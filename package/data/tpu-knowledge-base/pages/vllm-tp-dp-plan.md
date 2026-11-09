<!--
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
---
entry: vllm-tp-dp-plan
keyword: vllm-tp-dp-plan
url: https://github.com/google-pytorch/tpu-agent-suite/blob/main/package/scripts/common/engine_compatibility_check.py
authority: first-party-verified
retrieved: 2026-10-16
source_updated: 2026-10-16
generator: authored
---

# Choosing TP and DP for a vLLM deployment on torch_tpu

tpu-model-onboarding does not search for a layout. The
`vllm-engine-compatibility` check (`engine_compatibility_check.py`) writes
`tp_plan.json` with one tensor-parallel size (TP) and one data-parallel size
(DP), and the recipe copies them into `deployments.json` unless the user
stated a layout or the model is in the patch registry. The plan takes the
**largest** TP that fits, so one engine serves the model.

## Terms

The same words are used here, in `tp_plan.json` and in onboarding's
references:

- **Device**: what torch_tpu counts as one device, and what
  `compute inspect` reports as `job_devices`. A tpu7x chip is two devices; a
  v6e chip is one.
- **TP**: `--tensor-parallel-size`, the devices one engine shards the model
  across.
- **Replica**: one vLLM engine with its own full copy of the weights, its own
  KV cache and its own compiled graphs. **DP** (`--data-parallel-size`) is the
  number of replicas.
- **TP x DP** in the plan equals the devices one deployment gets. The manifest
  can leave some idle: a TP the user states with no `--data-parallel-size` in
  this vLLM build, or a build that cannot run replicas, starts one engine.

size-serving-topology (`index.py get playbook-size-serving-topology`) asks a
different question: how big a slice to buy. Its "model parallelism `Y`" is TP
here, and its slice is the device count, which onboarding never changes. Its
pitfall against data parallelism in a generate configuration follows the
scaling book, which says data parallelism within one model "replicates our
parameters and doesn't help us load parameters faster", and that you are
"better off spinning up multiple copies of the model instead"
(`index.py get scaling-book-inference`). A vLLM replica is such a copy: it
shares no weights or KV cache with the other replicas. So that pitfall does
not rule out DP here.

## Applies when

- Choosing `--tensor-parallel-size` and `--data-parallel-size` for vLLM on the
  `vllm_torchtpu` plugin.
- Reading `tp_plan.json`, or deciding whether to pass `--tp` to override it.
- A server fails at the planned TP and you need the next layout to try.

## The rule

Read from `plan_parallelism`, `head_problem` and `main` in the script, and
from `torch-inference-coder.md`, at tpu-agent-suite `5f408a087`, plus the
KV check (`oss_meshes.py`) and the traffic rule added after it.

1. **A TP the user states always wins** (`source: "user"`). The plan only
   records what it costs: idle devices, replica startup, an expected OOM.
   Next comes a model in the patch registry (`vllm_patches/registry.json`):
   its entry's `validated.serving` settings take precedence over the plan,
   and the manifest records `parallelism.source` as `registry`. The steps
   below are the plan, which decides every other model.
2. **Candidates** are the TPs that divide the device count and suit the
   heads:
   - query heads split evenly;
   - KV heads split evenly, or TP is a multiple of the KV-head count so they
     replicate (MLA models always replicate);
   - a hybrid model's Gated-DeltaNet key and value heads both split evenly
     (`linear_attention_heads`).
3. **Drop TPs the stack cannot start**: the ones the installed torch_tpu has
   no device layout for, and the ones vLLM rejects. The script asks
   `TpuPlatform._get_tpu_topology(tp)` itself, so it keeps no list of counts.
   For example, on a v6e 2x4 it refuses TP 2 with "No TPU topology found for
   count: 2". If that lookup fails any other way, no TP is ruled out. vLLM is
   asked only about the TP the plan picks, through a configuration dry run
   (`EngineArgs.create_engine_config`); a TP it refuses is dropped and the
   plan is redone. A dry run does not build the model's layers, so a failure
   that appears only when the engine starts is not caught here.
4. **Weight fit**: weights ÷ TP must be at most 80% of one device's HBM. The
   KV cache, a hybrid model's per-sequence state and working memory are not
   counted.
5. **Take the largest** remaining TP, with DP = devices ÷ TP. Replicas appear
   only when the heads, the layouts or vLLM rule out a TP that spans every
   device: a 12-head model on 8 devices gets TP 4 x DP 2. **Except for
   long-prompt traffic:** when `--points` include an input at least 4x its
   output at 128 or more concurrent requests, `--objective` is throughput
   (the default), and that largest TP spans every device, the plan halves it
   once (TP 8 x DP 1 becomes TP 4 x DP 2), if the smaller TP fits and its KV
   cache holds every point the larger TP's holds. `traffic_rule` records it with the layout passed
   over; see "TP versus replicas, measured" for the numbers.
6. **Record the alternative.** When a smaller TP also fits, the plan sets
   `smallest_fitting_tp` and `reason` says that it "may serve faster". Any
   benchmark of the deployment must then say it was measured at the larger
   TP. Measuring the smaller one is opt-in, with `--tp`.
7. **Check the KV cache at the target points.** With `--points`, the plan's
   `kv_warnings` lists each point whose sequences (concurrency ÷ DP of them,
   each input plus output length long) do not fit the HBM left on a device
   after the weights. A sliding-window layer counts only its window, as
   vLLM's hybrid KV-cache manager allocates it. The free HBM is an estimate
   (vLLM's default `--gpu-memory-utilization` of 0.92, less the weight shard
   and a 5 GiB runtime reserve, which matches a Gemma-3-27B TP 8 server log
   within 3%), so a warning is a point to watch, not a refusal. It needs the
   weight size, so it is skipped when that could not be read.

If the weight size cannot be read, the plan takes the largest
head-compatible TP without checking the fit, and `reason` says so.

## Why largest, not smallest

The plan used to take the smallest TP that fits, which passed the weight
check and then failed to start. The check's regression cases keep two
examples on a tpu7x-4 (8 devices x 96 GiB), where that rule chose
TP 1 x DP 8:

- **Qwen3.8-27B** (51.75 GiB): its linear-attention state slots ran one device
  out of HBM at server startup.
- **GPT-OSS-20B**: eight engines compiling at once on one host ran past
  vLLM's engine-ready timeout.

Both now plan TP 8 x DP 1. Onboarding needs a server that starts on the
first try. The trade-off is throughput: for
a small model, a smaller TP with more replicas may serve more tokens per
second.

## TP versus replicas, measured

Fewer TP ranks with more replicas helps one kind of traffic and hurts the
rest. Measured on Gemma-4-31B, tpu7x 2x2x1 (8 devices), vLLM 0.29.0,
`--max-model-len 10240`, the plan's
TP 4 x DP 2 against TP 2 x DP 4, two runs per point, both arms running at
the same time on the same cluster:

| Point (input/output, concurrency) | Output tok/s, TP 4 x DP 2 | TP 2 x DP 4 | Change |
| --- | --- | --- | --- |
| 8k/1k, 128 | 2928 | 3742 | +27.8% |
| 8k/1k, 32 | 1893 | 1792 | -5.3% |
| 1k/1k, 128 | 6654 | 6312 | -5.1% |
| 1k/1k, 32 | 2900 | 2177 | -24.9% |
| 1k/1k, 8 | 882 | 609 | -31.0% |

Repeated on vLLM 0.31.0 with
`--attention-backend CUSTOM`, as tpu-model-onboarding launches, same model
and flags, 2026-10-16:

| Point (input/output, concurrency) | Output tok/s, TP 4 x DP 2 | TP 2 x DP 4 | Change |
| --- | --- | --- | --- |
| 8k/1k, 128 | 3006 | 3835 | +27.6% |
| 1k/1k, 32 | 2870 | 2268 | -21.0% |

- At 8k/1k with 128 users the prefill step was collective-bound: TP
  all-reduces took 43% of it. Halving TP shortened them, and twice the
  replicas prefilled in parallel.
- Everywhere else decode dominates, and decode reads every weight each step.
  At TP 2 each device holds twice the weights, so each step is slower
  (inter-token latency +38% to +45% at 8 to 32 users), and the extra replicas
  do not make up for it.

### TP 8 against TP 4 x DP 2 and TP 2 x DP 4, two models

Same stack (vLLM 0.31.0), `--attention-backend
CUSTOM`, `--max-model-len 10240`, `--max-num-seqs 256`, two runs per point,
2026-10-16. Change in output tokens/s against TP 8 x DP 1:

| Point (input/output, concurrency) | Gemma-4-31B TP 4 x DP 2 | TP 2 x DP 4 | Mistral-Small-24B TP 4 x DP 2 | TP 2 x DP 4 |
| --- | --- | --- | --- | --- |
| 1k/1k, 32 | -7% | -27% | -12% | -33% |
| 1k/1k, 128 | +3% | -2% | -1% | -4% |
| 8k/1k, 32 | +9% | +6% | +2% | -5% |
| 8k/1k, 128 | +24% | +57% | +15% | +40% |

TP 8 output tokens/s at 8k/1k, 128: 2443 (Gemma-4), 3692 (Mistral). Up to
128 users TP 8 had the lowest median time to first token at 8k/1k (669 to
742 ms Gemma-4, 474 to 502 ms Mistral, against 1235 to 1255 and 784 to 894
at TP 4 x DP 2).

### Onboarding with and without the rule

Main's onboarding task, A = main (TP 8 x DP 1), B = with the traffic rule
(TP 4 x DP 2), same image, one run each, 2026-10-16/17, concurrency doubled
from 1 until throughput stopped rising. Change in output tokens/s, B
against A, at the same concurrency, and at each arm's peak (which can fall
at different concurrencies):

| Model, input/output | 32 users | 128 users | 256 users | Peak against peak |
| --- | --- | --- | --- | --- |
| Gemma-4-31B, 1k/1k | -8% | -2% | +8% | +16.5% (both at 512) |
| Gemma-4-31B, 8k/1k | +9% | +24% | +25% | +25.4% (both at 256) |
| Mistral-Small-24B, 1k/1k | -12% | -1% | +17% | +44.2% (A at 256, B at 1024) |
| Mistral-Small-24B, 8k/1k | +2% | +14% | not measured in A | +31.8% (A at 128, B at 512) |

At 128 users these agree with the table above within noise. From 1 to 16 users B was 15
to 20% slower. At 256 users and 8k/1k B's median time to first token was
lower (Gemma-4: 2.5 s against 11.3 s), because one TP 8 engine queues.

So the rule for this choice: keep the largest TP for short prompts, few
users, or when time to first token matters; for long prompts at high
concurrency, one TP halving with twice the replicas serves more, and the
check applies it by itself (`traffic_rule` in `tp_plan.json`).

**Why 4x and 128.** Only input/output 8 (8k/1k) and 1 (1k/1k) were
measured. At 128 users 8k/1k wins clearly and 1k/1k is even, so 128 is
where the long-prompt gain is clear, and 4 is a conservative point between
the two measured ratios, not a measured crossover. Short prompts also win
from about 256 users (above), which the rule leaves to the sweep: a sweep
that doubles concurrency reaches 128, so with a long-prompt pair the rule
picks the layout for the sweep's peak, at the cost of its low-concurrency
points. Halving again
wins more at 8k/1k with 128 users but loses up to a third at short prompts;
try it only when a profile shows collectives taking a large share of the
prefill step. Measure every target point before adopting it, because
one layout does not win them all. A layout change also changes the reduction
order: 3 of 8 greedy prompts gave different outputs at TP 2 x DP 4, so check
outputs before comparing quality.

## When the planned TP fails

These are tpu-model-onboarding's retry rules (`torch-inference-coder.md`):

- **Memory error**: rerun the check with `--tp` set to the next divisor of the
  device count above the failed TP, and take both TP and DP from the new
  plan. Never change TP alone.
- **No larger TP**, or the new plan's `ok` is false: keep the layout and shrink
  each engine. A hybrid model lowers `max_num_seqs` first when
  `--max-num-seqs` is in `vllm_flags.txt`, since it keeps one state slot per
  concurrent sequence; otherwise, or for any other model, lower
  `max_model_len`.
- **Stop after two resubmissions** and hand the decision back to the user.

The rules do not yet cover an engine assertion, as opposed to a memory error,
at the planned TP (see the first caveat).

## Evidence

| Run | Hardware | Rule | Layout | Outcome |
| --- | --- | --- | --- | --- |
| Onboarding runs under the smallest-TP rule, kept as the check's regression cases | tpu7x-4, 8 x 96 GiB | smallest | TP 1 x DP 8 | Qwen3.8-27B OOM at startup; GPT-OSS-20B past the engine-ready timeout |
| A check with a newer `vllm_torchtpu` than the GKE image, 2026-10-12 | tpu7x, 8 x 96 GiB | largest | TP 8 x DP 1 | Qwen3.8-27B hits a GDN state assertion in `pool_adapters.py`; TP 4 x DP 1 serves |
| Patch registry entry for GPT-OSS-20B, validated 2026-10-13 (vLLM 0.29.0, torch_tpu 0.1.1 release 2026.9.22) | tpu7x 2x2x1, 8 x 96 GiB | registry | TP 8 x DP 1 | Serves; TP 2 x DP 4 and TP 4 x DP 2 also verified; TP 1 has no torch_tpu layout on this stack. On the earlier vLLM 0.22.1 stack it served at TP 1 x DP 8 |
| An onboarding run under the smallest-TP rule, 2026-10-13 (tpu-agent-suite 0.5.0) | v6e 2x4, 8 x 32 GiB | smallest | TP 4 x DP 2 (TP 2 has no layout) | Qwen3.8-27B served and answered; no throughput, because every benchmark attempt lost the node or sat Pending |
| Qwen3.8-27B serves on vLLM 0.29.0, 2026-10-14 | tpu7x 2x2x1, 8 x 96 GiB | largest | TP 8 x DP 1 | Engine init fails: `Worker failed with error '(384, 256)'`; TP 4 x DP 2 serves |
| Gemma-4-31B TP 4 x DP 2 vs TP 2 x DP 4 on vLLM 0.29.0, 2026-10-14 | tpu7x 2x2x1, 8 x 96 GiB | measured | both | See "TP versus replicas, measured"; TP 1 x DP 8 fails at startup (`No TPU topology found for count: 1`) |

## Caveats

- **The largest TP does not start for Qwen3.8-27B on the GKE image.** The
  heads allow TP 8 x DP 1 (24 query heads, 4 KV heads replicated twice, and
  16 and 48 linear-attention heads all divide 8), but on the vLLM 0.29.0
  image engine init fails at TP 8, and a newer plugin hits a GDN state assertion
  there. The configuration dry run cannot catch either, so the plan still
  picks TP 8 (or TP 4 x DP 2 when the traffic rule halves it); TP 4 x DP 2
  serves. The observed constraint `qwen38_tp8_engine_init`
  (`vllm-torchtpu-mesh-constraints`) records the failure. GPT-OSS-20B, the
  other regression case, is validated at TP 8 on tpu7x (see Evidence).
- **Two models, one image for the full comparison.** TP 8 against
  TP 4 x DP 2 and TP 2 x DP 4 was measured for Gemma-4-31B and
  Mistral-Small-24B on vLLM 0.31.0; other models, images and
  concurrency levels between 32 and 128 are not measured.
- **DP replicas hang on some older images.** On vLLM 0.22.1
  (Gemma-3-27B) and a 2026-08-04 nightly development image (gpt-oss-20b),
  TP 4 x DP 2 reported healthy, then the
  first requests timed out (`TimeoutError: RPC call ... timed out`). The
  check does not know the image, so the traffic rule still halves there: on
  those images serve the plan's `instead_of`. The vLLM 0.29.0 and 0.31.0
  images measured above serve replicas.
- **The fit check counts weights only.** Expect less KV-cache room than 20% of
  HBM implies.

## Related

- How vLLM runs replicas (load balancing, one engine core per DP rank):
  `index.py get vllm-data-parallel-deployment`.
- The head-count rules vLLM itself applies: `index.py get vllm-model-config`
  and `index.py get vllm-qkv-parallel-linear`.
- Sizing a slice, batch and step-time floor from the model config:
  `index.py get playbook-size-serving-topology`.
- Every layout observed to fail at startup, with its failure signature:
  `index.py get vllm-torchtpu-mesh-constraints`.

## Sources

- `package/scripts/common/engine_compatibility_check.py`,
  `package/scripts/common/oss_meshes.py` and their tests,
  `package/references/tpu-model-onboarding/torch-inference-coder.md` and
  `package/scripts/common/vllm_patches/registry.json`, tpu-agent-suite
  `5f408a087`.
- The TP 8 assertion: a Qwen3.8-27B serve on tpu7x with a newer
  `vllm_torchtpu` than the GKE image, reported on 2026-10-12 when the
  largest-TP rule was reviewed.
- The smallest-TP onboarding run of 2026-10-13 on v6e 2x4: its
  `tp_plan.json`, serve attempts and benchmark failure table.
- The Gemma-4-31B and Qwen3.8-27B serves of 2026-10-14 on vLLM 0.29.0:
  `vllm bench serve` against the live server, random
  prompts, 5 x concurrency requests, two runs per point, baseline and
  candidate running at the same time; profiles taken in a separate run per
  point.
