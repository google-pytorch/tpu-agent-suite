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
entry: vllm-sharding-heuristics
keyword: vllm-sharding-heuristics
url: not applicable -- see Sources
authority: first-party-verified
retrieved: 2026-10-23
source_updated: 2026-10-23
generator: authored
---

# Sharding heuristics for vLLM on TPU: which layout change a trace argues for

How a running vLLM server's layout (tensor, data, expert parallelism) is
challenged from its own trace: which layouts are considered, which are
ruled out, how the rest are scored at each benchmark point, and what the
changes measured.

## Applies when

- A vLLM server on TPU is up at the planned layout and you want to know
  whether another layout serves your traffic faster.
- You see a layout candidate such as `m-tp2xdp4` with status
  `workload_dependent` and want what that means.
- For the layout onboarding starts from, read `vllm-tp-dp-plan`; for knobs
  other than the layout, `vllm-serving-heuristics`.

## Which layouts are considered

Every TP x DP split of the deployment's devices, with expert parallelism on
or off for MoE models (EP = TP x DP), plus pipeline, decode-context and
prefill-context parallel variants. Each is checked before it is scored:

- **Constraints**: layouts that fail at startup are rejected with their
  failure signature (no TP 1 layout on tpu7x[^tt-topo], pipeline parallelism with
  TP > 1, the Gated DeltaNet block rule, EP with more than 16 experts per
  group, known per-model failures such as Qwen3.8-27B at TP 8):
  `vllm-torchtpu-mesh-constraints`. A failure seen only on other images is
  a warning, not a rejection.
- **Fit**: weights split over TP must fit in 80% of one device's HBM, and
  the KV cache must hold the target points' sequences (sliding-window
  layers counted at their window).

## How a layout is scored

Relative to the baseline layout, at the traffic's regime:

| Term | Score | Why |
| --- | --- | --- |
| Changing TP by a factor of 2, or turning EP on or off | -0.25 each | a change has a cost |
| Collective-bound prefill (`D7`: collectives >= 25% of a prefill or mixed step), prefill-heavy or balanced traffic | +1 for one TP halving, -1 per TP doubling | a smaller all-reduce group, and twice the replicas to prefill in parallel |
| Decode-heavy traffic | -1 per TP step | decode reads every weight each step; at half the TP each device holds twice the weights |
| No collective-bound prefill | -0.5 per TP step | exploratory only |
| Padded KV rows (`D8`) | +1 when the layout leaves no KV-cache padding or replicas per device and the baseline does not, counting the global (full-attention) KV heads too | K and V interleave: two rows per KV head[^rpa-kv] (one at head_dim 64[^vtt-hd64]), at least one head per device, padded to the dtype packing (2 for bf16, 4 for 8-bit) |
| Adding EP | +0.5 | |
| KV-capacity warning (when the estimate is trusted) | -1 | |
| Pipeline / context parallelism | -0.25 flat (unverified on torch_tpu), with bounded bonuses when they shrink TP without growing per-device weights or relieve a short KV cache | |

Each point is scored separately, so a layout is a candidate if it scores
positive at the main point or at any benchmark point. When its points
disagree it is `workload_dependent`: expected to win at some points and lose
at others. A layout that scores at or below zero everywhere is listed as not
argued for. At most three candidates (layouts and knobs together) are
measured.

## Measured layout changes

Output tokens/s at each regime's baseline peak, tpu7x 2x2x1 (8 devices),
vLLM-TorchTPU with `--attention-backend CUSTOM`, one run per point:

| Model | Baseline -> candidate | 1k/1k peak | 8k/1k peak | At low concurrency | vLLM |
| --- | --- | --- | --- | --- | --- |
| Mistral-Small-24B | TP 4 x DP 2 -> TP 2 x DP 4 | +32.5% (c512) | +35.4% (c512) | -24% to -31% at 1 to 32 users | 0.32.0 |
| Gemma-4-31B | TP 4 x DP 2 -> TP 2 x DP 4 | +14.8% (c512) | +29.2% (c256) | -16% to -30% at 1 to 64 users | 0.32.0 |
| Gemma-4-31B | TP 4 x DP 2 -> TP 2 x DP 4 | +13.4% (c256) | +25.9% (c128) | -13% to -32% at 1 to 64 users | 0.31.0 |
| gpt-oss-20b | TP 8 x DP 1 -> TP 4 x DP 2 | +16.3% (c256) | +43.1% (c256) | -46% at 1 to 2 users | 0.31.0 |

- The Gemma-4-31B vLLM 0.31.0 row is the 2026-10-21 round; an earlier pair
  at 8k/1k with 128 users (2026-10-16) measured +27.6% (`vllm-tp-dp-plan`).
  Both are single runs, about two points apart.
- gpt-oss-20b TP 4 x DP 2 with expert parallelism failed at weight load on
  vLLM 0.31.0 (the MXFP4 loader passes no expert id to the EP map).
- Qwen3.8-27B: every other layout is ruled out (TP 2 by the Gated DeltaNet
  block rule, TP 8 by its known engine-init failure), so no layout is
  proposed.
- After Mistral-Small-24B moved to TP 2 x DP 4, the TP all-reduce fell from
  39% to 5.4% of the prefill step, and decode attention became the largest
  cost.

## Caveats

- Every layout win above is at peak load and costs throughput at low
  concurrency, with higher per-token latency: a per-regime choice, kept
  for the traffic the user named or each regime's peak.
- Fewer TP ranks also means fewer KV blocks per replica: Mistral-Small-24B
  at TP 2 x DP 4 preempted more requests (42 per engine over the sweep,
  against 7 at the baseline's 8k/1k sweep).
- Points are measured hours apart from the baseline on a shared cluster;
  noise is assumed at 3%, not measured.

## Related

- The plan onboarding starts from, and TP versus replicas measured:
  `index.py get vllm-tp-dp-plan`.
- Layouts that fail to start, with failure signatures:
  `index.py get vllm-torchtpu-mesh-constraints`.
- Knob changes from the same trace: `index.py get vllm-serving-heuristics`.

## Sources

- `package/scripts/common/oss_meshes.py` (`enumerate_oss_meshes`,
  `KERNEL_CONSTRAINTS`, the layout score) and
  `package/scripts/optimization-heuristics/propose.py`.
- Optimization-heuristics rounds on tpu7x 2x2x1: vLLM 0.31.0 (2026-10-19 to
  2026-10-21) and vLLM 0.32.0 (2026-10-23).

[^tt-topo]: [torch_tpu `_internal/utils/hardware.py`](https://github.com/google-pytorch/torch_tpu/blob/main/src/torch_tpu/_internal/utils/hardware.py). code-reference. Per-generation device topologies (v5e, v5p, v6e have 1 device; v7 starts at 2) and the `No TPU topology found for count` error; checked at commit 9b6ebfc24032. URL checked 2026-10-28.
[^rpa-kv]: [tpu-inference RPA v3 `util.py`, `get_kv_cache_shape`](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/ragged_paged_attention/v3/util.py). code-reference. KV cache rows `align_to(num_kv_heads * 2, packing)`, packing `32 // bits`; checked at commit 6b62267613be. URL checked 2026-10-28.
[^vtt-hd64]: [vllm-torchtpu RPA v3 `kernel_hd64.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/src/vllm_torchtpu/kernels/ragged_paged_attention/v3/kernel_hd64.py). code-reference. head_dim 64: K and V side by side in one 128-wide row per head; checked at commit a20f4a9b14e1. URL checked 2026-10-28.
