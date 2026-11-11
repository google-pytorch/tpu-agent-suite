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
entry: vllm-torchtpu-serving-hillclimb
keyword: vllm-torchtpu-serving-hillclimb
url: not applicable -- see Sources
authority: first-party-verified
retrieved: 2026-11-04
source_updated: 2026-11-06
generator: authored
---

# vLLM-TorchTPU disaggregated 1P1D & MoE serving hill-climb gates, measured

Measured root causes, failure modes, and decision rules from iterative
serving optimization on `vllm-torchtpu` over `tpu7x` slices (`GLM-5.2-FP8`
1P1D across two `2x2x2` slices and `Qwen3.5-397B-A17B-FP8` single-host `2x2x1`
under long-context multi-turn traffic with `aiperf`). Each section except the
predictor fit (section 5) ends with the rule to apply before launching an arm.

## Applies when

- Tuning `PREFILL_DCP_SIZE`, `DECODE_DCP_SIZE`, decode batch caps, or MTP
  speculative tokens on `vllm-torchtpu` 1P1D disaggregated or single-host
  deployments on `tpu7x`.
- Deciding whether to spend multi-slice TPU hours on a prefill-only knob, a
  larger decode batch cap, or an `absorb_q_head` MLA kernel flag.
- Stopping losing `aiperf` arms early or projecting multi-concurrency `Mtok/$`
  and `p90` interactivity before running a full sweep.

## 1. Asymmetric P/D DCP vs symmetric `DCP_SIZE=8` (`GLM-5.2-FP8` 1P1D, `tpu7x`)

Setting top-level `DCP_SIZE=8` on a `TP=16` 1P1D recipe applies `DCP=8` to both
legs, duplicating KV heads 2x on the prefill slice and halving the prefill KV
(prefix) cache from `11.89M` to `5.95M` tokens. Decode steps got faster, but
the halved prefill prefix cache cost more `Mtok/$` than that saved:

| Configuration | Prefill DCP / Decode DCP | Prefill KV capacity (tokens) | `Mtok/$` vs the `DCP=16` baseline |
| --- | --- | --- | --- |
| Symmetric baseline | `16 / 16` | `11.89M` | baseline |
| Symmetric `DCP_SIZE=8` | `8 / 8` | `5.95M` (`0.50x`) | `-24.8%` at `c=48`, decode cap `16`; `-1.7%` at `c=32`, cap `8` |
| Asymmetric P/D DCP (`TPU_KV_STRIPED_LAYOUT=0`) | `16 / 8` | `11.89M` (`1.00x`) | not reported |

When `PREFILL_DCP_SIZE != DECODE_DCP_SIZE` on models with a SparseCore / MLA
indexer K cache, both legs must also set `TPU_KV_STRIPED_LAYOUT=0` so KV
transfer resharding uses a consistent block layout, and the indexer cache must
not use `INDEXER_KV_CACHE_MODE=sal` (it lacks asymmetric byte ranges; use `has`).

Rule: with 32 or more sessions per prefill leg, keep prefill DCP at 16 and
lower only decode DCP (`PREFILL_DCP_SIZE=16 DECODE_DCP_SIZE=8`); do not set a
top-level `DCP_SIZE` of 8 or less.

## 2. Fused TP MoE decode kernel cap (`ROW_BUCKETS = (16, 32, 64)`)

The fused quantized TP MoE decode kernel (`moe_tp_decode`) only compiles
buckets up to `ROW_BUCKETS = (16, 32, 64)`. With the default decode buckets
(powers of two from 16), a step pads up to the next bucket, so any step of 64
tokens or fewer stays fused (48 runs in the 64 bucket). An explicit
`DECODE_TOKEN_BUCKET_EXTRA` size that is not a `ROW_BUCKETS` entry may not.
With `MTP=1` and `MTP_NUM_SPECULATIVE_TOKENS=3` (`1 + 3 = 4` query tokens per
sequence), raising the decode batch cap from `16` (`64` tokens) to `20`
(`20 x 4 = 80` tokens) crosses `64` and silently falls back to the unfused
GMM+psum path:

| Arm / kernel state | Decode tokens per step (`cap x (1 + speculative tokens)`) | Path executed | Per-layer MoE decode latency (microbenchmark) | End to end |
| --- | --- | --- | --- | --- |
| Batch cap `16`, 3 speculative tokens | `16 x 4 = 64` | Fused `moe_tp_decode` (`64` bucket) | not measured | baseline |
| Batch cap `20`, 3 speculative tokens (stock buckets) | `20 x 4 = 80` (`> 64`) | Unfused GMM+psum fallback | `438 us` | ITL p50 `10 -> 13 ms` and `-28%` `Mtok/$` at 11 min of the measured phase |
| Batch cap `20`, 3 speculative tokens (`ROW_BUCKETS` extended to `80`) | `20 x 4 = 80` | Fused `moe_tp_decode` (`80` bucket) | `175 us` (`2.5x` faster than the fallback) | not measured end to end |

The latencies come from a kernel microbenchmark on one 8-core `tpu7x` host at
`GLM-5.2-FP8` TP16 per-rank shapes. The fallback's end-to-end figures are a
reading 11 minutes into one run's measured phase, not a final result.

Rule: under TP MoE, keep `cap x (1 + speculative tokens)` at or below the
largest `ROW_BUCKETS` entry (cap `16` with 3 speculative tokens), or extend
`ROW_BUCKETS` and verify the new bucket with a 1-host microbenchmark first, or
switch to EP MoE.

## 3. 1P1D leg-bottleneck gating (prefill optimization under decode saturation)

In a disaggregated 1P1D topology, end-to-end throughput is bounded by
$\min(\mu_{\text{prefill}}, \mu_{\text{decode}})$. Optimizing the non-bottleneck
leg changes local step latency without moving served throughput. Measured on
`GLM-5.2-FP8` 1P1D at `c=48` with decode cap `16`:

| Metric | Baseline arm | Prefill-only arm | Change |
| --- | --- | --- | --- |
| Prefill drafter time per draft pass (`draft_propose`) | `7.5 ms` | `2.7 ms` | `2.8x` faster |
| End-to-end served throughput | `1.000x` | `0.999x` | no gain |

In both arms the prefill `vllm:num_requests_waiting` was 0 at p50 and p75:
requests did not queue at prefill, so decode was the limit.

Rule: before an arm that changes only one leg's knobs (for example only
`EXTRA_ENV_PREFILL`), read both legs' `/metrics`. If every prefill endpoint's
`vllm:num_requests_waiting` p75 is below 1 and the busiest prefill endpoint is
busy less than 70% of the run (the leg check's cutoff, not a measured limit),
the deployment is decode-limited and a prefill-only change will not move
served throughput. If the prefill waiting p50 is 1 or more, prefill is queued
and a decode-only change will not move it.

## 4. Equal-time `aiperf` early stopping

`aiperf` prints realtime blocks during the measured phase with the tokens of
the requests finished so far. Early in a full-hour run this cumulative rate
underestimates the final tok/s/chip by 10–15%, but runs of the same trace
share that bias. So an arm's cumulative total tok/s/chip (the `Total` row
divided by the chip count) over the baseline's at the same elapsed time
predicts the final ratio. On 15 finished full-hour `GLM-5.2-FP8` P/D runs
(105 pairs), the ratio at 15 minutes was this far from the final ratio:

| Pairs | Count | Median error at 15 min | Max error at 15 min |
| --- | --- | --- | --- |
| Same concurrency and chip count | `31` | `1.6%` | `5.7%` |
| Different concurrency or chip count | `74` | `5.3%` | `14.1%` |

A replay over two days of finished arms against a `c=48` 1P1D baseline used
the bands of that time, not the current ones below (same shape: ±7% from
10 min, ±5% from 30 min, ±3% from 50 min; different shapes: ±15%, then ±14%
from 20 min, ±12% from 30 min, ±6% from 50 min). It stopped each losing arm
about 10 minutes into the measured phase, freeing its chips about 50 minutes
early. Two reruns of the baseline, checked against a `1.028x` target, stopped
later, once the band ruled that win out:

| Arm | First `stop` (ratio) | Final vs baseline |
| --- | --- | --- |
| 1P2D at `c=96` on 24 chips | 10.5 min (`0.771`) | `-23.3%` |
| `DCP_SIZE=8` on both legs | 10.5 min (`0.747`) | `-24.8%` |
| Decode cap `18` (72-token steps padded to 128) | 10.1 min (`0.660`) | deleted before the end |
| Decode cap `20` (80-token steps padded to 128) | 10.3 min (`0.694`) | deleted before the end |
| Two reruns of the baseline | 30.2 and 55.1 min, once the band ruled out a `1.028x` win | `-2.5%` and `-0.9%` |

Rule (same concurrency and chip count): the error band is ±8% from 10 min,
±6% from 15 min and ±4% from 30 min. Stop an arm when its equal-time
tok/s/chip ratio plus the band is below `1.0`, that is below `0.92x` at
10 min, `0.94x` at 15 min or `0.96x` at 30 min. Also stop when its ITL is at
least `1.5x` the baseline's and the ratio is below `1.0`. The arm is
promising when the ratio minus the band is above `1.0`; otherwise keep
watching. Pairs that differ in concurrency or chip count use ±11% from
15 min, ±9% from 20 min and ±7% from 30 min, below the largest cross-shape
error at 15 min above (14.1%).

The 10-minute and 30-minute bands (cross shape: 20 and 30 minutes) are chosen
margins, not measured errors; the error was measured at 15 minutes only. The
10-minute band is wider than the largest same-shape 15-minute error (5.7%).
The later bands are narrower than the 15-minute maximum and have no measured
error behind them.

## 5. Closed queueing-network (CQN) predictor fit

A closed queueing network (client think time plus prefill and decode stations
with load-dependent rates, solved exactly by convolution), parameterized by
calibrated prefill/decode step times, the decode batch cap, and the MTP
acceptance length, projects throughput and tail interactivity across
concurrency sweeps on CPU. Single-host serving uses a chunked-prefill
fixed-point model instead. The errors below are in-sample: the parameters
were fit on the same runs they are scored on, so they show fit quality, not
accuracy on new configurations.

| Model & serving topology (`tpu7x`) | Runs (fit and scored) | `tok/s/chip` MAE | `p90` interactivity (`tok/s/user`) MAE |
| --- | --- | --- | --- |
| `GLM-5.2-FP8` disaggregated P/D (`c=32..96`) | `24` E2E runs | `1.35%` | `1.08%` |
| `Qwen3.5-397B-A17B-FP8` single-host (`2x2x1`) | `17` E2E runs | `1.95%` | not reported |

## 6. XLA VMEM OOM gate (`absorb_q_head` on `tpu7x`)

With `TPU_MLA_ABSORB_Q_FROM_HEAD` on, MLA query-head absorption
(`absorb_q_head`) allocates Pallas window buffers in VMEM. At 64 or more
replicated local Q heads per rank (`DCP_SIZE` of 16 or more with
`VLLM_DCP_Q_REPLICATE` on), those buffers take `68 MiB`, above the `64 MiB` of
`tpu7x` VMEM per core, and XLA compilation fails with `RESOURCE_EXHAUSTED`.

Rule: do not enable `TPU_MLA_ABSORB_Q_FROM_HEAD` on a leg with DCP of 16 or
more while `VLLM_DCP_Q_REPLICATE` is on.

## Related

- Measured `--async-scheduling` and `--max-num-batched-tokens` A/Bs: `index.py get vllm-torchtpu-measured-serving-knobs`.
- Disaggregated P/D setup and Raiden connector flags: `index.py get vllm-torchtpu-disaggregated-serving`.
- Trace-driven serving detector rules (`D1`–`D46`): `index.py get vllm-serving-heuristics`.
- Attributing collectives and KV capacity in XProf traces: `index.py get serving-trace-attribution`.
- In the experimental package, the `close-gaps` skill runs these rules as CPU
  checks (`hillclimb_cli.py`, described in its `serving-hillclimb-gates`
  reference).

## Sources

- Manual multi-concurrency `aiperf` A/B runs and standalone Pallas kernel
  microbenchmarks on `tpu7x` (`2x2x1` and `2x 2x2x2` 1P1D), October–November
  2026.
- `package/scripts/optimization-heuristics/hillclimb/` and
  `package/references/close-gaps/serving-hillclimb-gates.md`.
