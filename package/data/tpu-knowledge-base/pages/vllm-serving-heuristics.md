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
entry: vllm-serving-heuristics
keyword: vllm-serving-heuristics
url: not applicable -- see Sources
authority: first-party-verified
retrieved: 2026-11-06
source_updated: 2026-11-06
generator: authored
---

# Serving heuristics for vLLM on TPU: what a trace argues for

The fixed rules (detectors) that read an XProf trace of a running vLLM
server on TPU and say which serving knob, if any, the trace argues for; how
a change is chosen and judged; and what has been measured.

## Applies when

- You are deciding which serving setting (env var, vLLM flag, attention
  backend) to try on a vLLM server on TPU, and want the rule a trace
  signature maps to.
- You read a detector id (`D7_...`, `D45_...`) in a profiling result or a
  candidates file and want what it means.
- For layout changes (TP / DP / EP), read `vllm-sharding-heuristics`; this
  page covers the knob side.

## How a change is chosen

1. **Target.** The point to optimize: one the user picks from the baseline's
   measured (input/output length, concurrency) points, else each regime's
   peak, the concurrency where the baseline's output tokens/s peaked.
2. **Phase.** At each target point, the phase that dominates its time:
   per engine, prefill runs input length x concurrency / DP tokens in steps
   of the captured prefill bucket, decode runs output-length steps, each at
   its captured step time. Gemma-4-31B at 1k/1k with 512 users was 24%
   prefill (decode-dominated); at 8k/1k with 256 users, 65% prefill.
3. **Candidates.** Each detector that fires with a knob change gives one
   candidate (at most three in total, layouts included); candidates whose
   findings fired in a target point's dominant phase rank first. A knob the
   serving image does not define is reported, never launched.
4. **Measure and keep.** Each candidate is served and benchmarked at every
   point the baseline was, and the one with the largest gain in the user's
   metric at the target point is kept, provided no request fails there and
   any stated TTFT or ITL limit holds. Losses at other points are reported,
   not disqualifying. Results on other models never keep a candidate from
   being tried.

## The rules that most often apply to vLLM-TorchTPU

| Signature in the trace | Detector | What it proposes | Guardrail or known result |
| --- | --- | --- | --- |
| Collectives >= 25% of a prefill or mixed step | `D7_collective_bound` | a layout change (fewer TP ranks, more replicas) | see `vllm-sharding-heuristics` |
| A device's KV-cache rows (K and V interleaved: two per KV head[^rpa-kv], one at head_dim 64[^vtt-hd64]) do not fill the KV dtype packing (2 for bf16, 4 for 8-bit), or TP exceeds the KV heads | `D8_kv_head_padding` | the largest TP whose rows fill the packing, freed devices as replicas (never TP 1 on vLLM-TorchTPU) | one bf16 KV head per device fills the packing: 8 KV heads at TP 8 is not padding |
| Prefill steps interleaved with decode steps stall in-flight decodes | `D3_prefill_decode_interference` | a smaller prefill chunk (`--max-num-batched-tokens`) where the engine takes one | never below the benchmark's prompt length: a budget under the prompt splits every prefill (Gemma-3 at 8k inputs: TTFT +67%) |
| The device idles between decode steps and `--async-scheduling` is unset | `D29_async_scheduling_off` | `--async-scheduling` | no gain twice on Qwen3.8-27B (TTFT +18% at 128 users); a tie at every measured point on Mistral-Small-24B |
| Ops sized by `--max-num-seqs` while far fewer requests are live | `D27_max_num_seqs_inflation` | a smaller `--max-num-seqs`, sized from the busiest benchmark point per DP replica | no change when that is not smaller than the current value (the server must still serve its busiest point) |
| On a disaggregated decode engine (`kv_role` `kv_consumer`): sparse-attention indexer (`lightning_indexer` / `topk`) >= 8% of decode step time under `--decode-context-parallel-size` >= 16, `(TP * PCP) % 8 == 0`, and <= 16 concurrent decode sequences (or mean decode bucket <= 64 tokens) | `D18_attn_indexer` | `--decode-context-parallel-size=8` on the decode engine only | diagnostic on a unified or prefill engine, when the decode batch has more than 16 sequences and a mean decode bucket above 64 tokens, or when per-rank KV cache at DCP 8 exceeds `--kv-cache-memory-bytes` (36 GB assumed when unset); keep prefill DCP >= 16 (symmetric DCP 8 on GLM-5.2 1P1D halved KV capacity 11.89M -> 5.95M tokens, dropped prefix-cache hits 91.9% -> 80.2%, and lowered Mtok/$ 24.8% at concurrency 48) |
| MLA attention >= 5% of step time under `--tensor-parallel-size` > 1 | `D20_mla_attention_tp` | `--tensor-parallel-size=1` with `--data-parallel-size` replicas | suppressed when `--decode-context-parallel-size` > 1, `--prefill-context-parallel-size` > 1, or unsharded MLA KV cache would exceed `--kv-cache-memory-bytes` (36 GB assumed when unset) |
| Decode RPA v3 kernel whose compute chunk far exceeds the sliding window or the context (kernel name `RPAd-...-bkv_<block>_<chunk>`) | `D45_rpa_decode_block_oversized` | `TPU_RPA_DECODE_BLOCK_SIZES`: per layer kind, a block and chunk of one window for sliding layers, the smallest page-aligned chunk covering the context for global layers | needs the RPA v3 kernel (`--attention-backend FLASH_ATTN`) and a plugin patch when the image lacks the knob |
| Decode step spends >= 8% in XLA `all-reduce` at <= 64 rows | `D47_small_decode_allreduce` | `TPU_SMALL_ALL_REDUCE_MAX_ROWS` sized to the step bucket | requires an expert-parallel group and serves only supported row-parallel / MoE combine paths |
| Separate Q/K RMSNorm and RoPE ops >= 3% of step time | `D48_unfused_qk_norm_rope` | `TPU_FUSED_QK_NORM_ROPE=1` | proposed only for Qwen3-Next models (diagnostic on others); the fused kernel covers Qwen3-Next-style gated attention on steps of <= 64 rows |
| Replicated latent-MoE projections >= 8% of step time on dual-core chips | `D49_replicated_latent_moe_proj` | `TPU_LATENT_PROJ_INTRA_CHIP_TP=1` | shards column-parallel across the two cores of each chip with an on-package all-gather |
| Pallas `small_all_reduce` kernel (`oneshot` or `twoshot`) >= 8% of decode step | `D50_small_allreduce_form` | `TPU_SMALL_ALL_REDUCE_FORM` set to the other form (`oneshot` vs `twoshot`) | form trade-off depends on rows and devices |
| TP-sharded MoE decode with MoE GMM >= 20% of decode step time and >= 50% of decode time in steps above 64 rows (`moe_tp_decode` hard limit, including `1 + num_speculative_tokens` per sequence) | `D52_moe_tp_decode_row_limit` | `--enable-expert-parallel` | below 64 decode rows `moe_tp_decode` stays active; suppressed on prefill-heavy unified workloads (inputs >= 2x outputs), not on disaggregated `kv_consumer` decode engines |
| Decode attention >= 40% of the decode step on the batched RPA kernel (`--attention-backend CUSTOM`), whose decode blocks no knob tunes | `D46_batched_rpa_decode_untunable` | `--attention-backend FLASH_ATTN` with D45's `TPU_RPA_DECODE_BLOCK_SIZES` | reported without a change for models whose RPA v3 kernels fail to compile (Gemma-4-31B: VMEM, vLLM 0.29.0) |
| MoE token padding in the expert kernels at the benchmarked loads | `D2_padding_waste` | `TPU_MOE_SKIP_PADDED_TOKENS=1` (vLLM-TorchTPU)[^vtt-envs] | tied at every point on gpt-oss-20b |
| MoE experts sharded by TP in decode, no expert parallelism | `D24_moe_tp_without_ep` | expert parallelism | not on prefill-heavy traffic (inputs >= 2x outputs), where TP won |
| A transfer driven by per-shard host copies while the device idles | `D25_host_driven_transfer` | nothing (diagnostic): pack KV / prefix buffers into fewer arrays | disaggregated serving |
| Speculative draft ops (MTP / EAGLE / Medusa) take a share of the step | `D34_speculative_draft_overhead` | nothing (diagnostic): the accepted tokens per step needed to break even | compare with the server's acceptance rate |
| TensorCore idle inside the step program | `D19_in_step_wait` | nothing (diagnostic) | |

## All detectors

| Detector | What it flags |
| --- | --- |
| `D1_host_gap` | device idles between steps (host-bound) |
| `D2_padding_waste` | token-bucket padding waste |
| `D3_prefill_decode_interference` | prefill steps stall in-flight decodes |
| `D4_moe_topk_sort` | MoE router top-k / sort hotspot |
| `D5_moe_permute` | MoE token permute / unpermute is expensive |
| `D6_expert_imbalance` | expert load imbalance (barrier wait after MoE) |
| `D7_collective_bound` | step is collective-bound |
| `D7_ar_plus_slice` | reduce-scatter lowered as all-reduce + slice |
| `D7_moe_allgather_fp8` | MoE dispatch all-gather in bf16 |
| `D7_decode_weight_allgather` | decode re-gathers weights every step |
| `D8_kv_head_padding` | KV heads padded / replicated under TP attention |
| `D9_rope_cache_copy` | per-step copy of the full RoPE cache |
| `D11_recompilation` | XLA compilation inside the serving window |
| `D12_kv_cache_quant` | quantized KV cache dequantized outside the attention kernel |
| `D13_gmm_zero_init` | grouped-matmul output zero-initialization is expensive |
| `D14_profile_hygiene` | profile quality limits confidence |
| `D15_torchtpu_fused_ep_moe` | MoE runs as all-gather + grouped matmul + reduce-scatter |
| `D16_torchtpu_mla_layout` | sparse-MLA KV cache uses the TensorCore layout |
| `D17_data_formatting` | relayout copies / per-token KV writes dominate |
| `D18_attn_indexer` | sparse-attention indexer (scoring + top-k) >= 8% of step time; on the vLLM-TorchTPU decode engine of a disaggregated pair (`kv_consumer`) with DCP >= 16, `(TP * PCP) % 8 == 0` and <= 16 sequences (or mean decode bucket <= 64), proposes `--decode-context-parallel-size=8` when KV fits |
| `D19_in_step_wait` | TensorCore idles inside the step program |
| `D20_mla_attention_tp` | MLA attention >= 5% of step time under `--tensor-parallel-size` > 1 (suppressed when DCP > 1, PCP > 1, or unsharded MLA KV cache exceeds `--kv-cache-memory-bytes`, 36 GB assumed when unset) |
| `D22_moe_without_gmm` | MoE experts run without a grouped-matmul kernel |
| `D24_moe_tp_without_ep` | MoE experts sharded by TP instead of EP (decode) |
| `D25_host_driven_transfer` | a transfer driven by per-shard host copies while the device idles (diagnostic) |
| `D26_moe_combine_no_local_reduction` | MoE combine collective carries the top-k axis |
| `D27_max_num_seqs_inflation` | per-step shapes sized by `--max-num-seqs`, not live requests |
| `D28_moe_activation_unfused` | MoE activation runs as a separate pass between expert matmuls |
| `D29_async_scheduling_off` | scheduler may run serially with the device |
| `D30_batched_rpa` | decode attention without the batched RPA kernel |
| `D31_gmm_tile_misalignment` | MoE expert width under TP is not a multiple of 256 |
| `D32_excess_collectives_per_layer` | more collectives per layer than TP / EP sharding needs |
| `D33_kv_cache_block_size` | small KV-cache page size (`--block-size`) in Pallas attention |
| `D34_speculative_draft_overhead` | speculative draft head (MTP / EAGLE / Medusa) share of a step, against the acceptance it needs (diagnostic) |
| `D35_unfused_glue_programs` | unfused single-op jit programs outside the step executable |
| `D36_hbm_limited_decode_batch` | decode batch capped by HBM while expert matmuls are weight-bound |
| `D37_moe_weight_precision` | MoE expert weights are wider than 4 bits |
| `D40_mla_kernel_config` | MLA attention runs the v1 kernel |
| `D44_moe_combine_not_chunked` | MoE combine collective is not chunked |
| `D45_rpa_decode_block_oversized` | decode attention kernels whose compute chunk far exceeds the window or the context; proposes `TPU_RPA_DECODE_BLOCK_SIZES` with sizes computed from the kernel names and the workload's context (unmeasured) |
| `D46_batched_rpa_decode_untunable` | decode attention on the batched RPA kernel (`--attention-backend CUSTOM`), which no knob tunes, while the stock RPA v3 decode blocks would be oversized; proposes `--attention-backend FLASH_ATTN` with D45's `TPU_RPA_DECODE_BLOCK_SIZES` (unmeasured against a CUSTOM baseline; reported without a change for models whose RPA v3 kernels failed to compile) |
| `D47_small_decode_allreduce` | decode step spends >= 8% in XLA `all-reduce` at <= 64 rows; proposes `TPU_SMALL_ALL_REDUCE_MAX_ROWS` |
| `D48_unfused_qk_norm_rope` | separate Q/K RMSNorm and RoPE ops >= 3% of step time; proposes `TPU_FUSED_QK_NORM_ROPE=1` on Qwen3-Next models (diagnostic on others) |
| `D49_replicated_latent_moe_proj` | replicated latent-MoE projections >= 8% of step time on dual-core chips; proposes `TPU_LATENT_PROJ_INTRA_CHIP_TP=1` |
| `D50_small_allreduce_form` | Pallas `small_all_reduce` kernel (`oneshot` or `twoshot`) >= 8% of decode step time; proposes A/B testing the other `TPU_SMALL_ALL_REDUCE_FORM` |
| `D52_moe_tp_decode_row_limit` | TP-sharded MoE decode with MoE GMM >= 20% of decode step time and >= 50% of decode time in steps above 64 rows (`moe_tp_decode` hard limit); proposes `--enable-expert-parallel` |
| `D53_prefill_microbatch_overlap` | collective-heavy long prefill with no microbatch comm/compute overlap (diagnostic) |

## Knobs and where they are defined

| Framework | Kind | Where it is defined |
| --- | --- | --- |
| vLLM | CLI flag | `vllm/engine/arg_utils.py` |
| vLLM-TorchTPU | env var | `vllm_torchtpu/envs.py` |
| vLLM-TPU (tpu-inference) | env var, `--additional-config` key | `tpu_inference/envs.py` |

A knob is only proposed if the serving image's installed source defines it;
a server ignores an env var it does not know. `TPU_RPA_DECODE_BLOCK_SIZES`
exists only with a plugin patch that reads the per-layer sizes from the env
var and holds no sizes itself; it changes only the RPA v3 kernel.

## Caveats

- Every detector reads one chip of one replica, and a phase with few steps
  lowers confidence (`D14_profile_hygiene`).
- For a collective-bound prefill (`D7`), these rules propose only the
  layout change. The detector also notes hiding the all-reduce behind
  compute (collective matmul / async collectives) as a follow-up for kernel
  or compiler work: no such serving knob is in the catalog.

## Related

- Layout changes, constraints and their measured trade-offs:
  `index.py get vllm-sharding-heuristics`.
- Reading collectives and the KV cache in a trace:
  `index.py get serving-trace-attribution`.
- Knob A/B results: `index.py get vllm-torchtpu-measured-serving-knobs`.

## Sources

- `package/references/optimization-heuristics/detectors.md` and `knobs.md`,
  and `package/scripts/optimization-heuristics/` (`serving_analysis/`,
  `propose.py`).
- Rounds on tpu7x 2x2x1 with vLLM-TorchTPU: vLLM 0.31.0 (2026-10-19 and
  2026-10-21) and vLLM 0.32.0 (2026-10-23).
- Disaggregated 1P1D GLM-5.2 FP8 serving on two tpu7x 2x2x2 slices (TP=16,
  DCP=16 vs DCP=8, MTP=3) with vLLM-TorchTPU (2026-11).

[^vtt-envs]: [vllm-torchtpu `envs.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/src/vllm_torchtpu/envs.py). code-reference. Defines `TPU_MOE_SKIP_PADDED_TOKENS` (default off); checked at commit a20f4a9b14e1. URL checked 2026-10-28.
[^rpa-kv]: [tpu-inference RPA v3 `util.py`, `get_kv_cache_shape`](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/ragged_paged_attention/v3/util.py). code-reference. KV cache rows `align_to(num_kv_heads * 2, packing)`, packing `32 // bits`; checked at commit 6b62267613be. URL checked 2026-10-28.
[^vtt-hd64]: [vllm-torchtpu RPA v3 `kernel_hd64.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/src/vllm_torchtpu/kernels/ragged_paged_attention/v3/kernel_hd64.py). code-reference. head_dim 64: K and V side by side in one 128-wide row per head; checked at commit a20f4a9b14e1. URL checked 2026-10-28.
