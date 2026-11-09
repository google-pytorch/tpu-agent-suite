---
references:
  - references/optimization-heuristics/serving-trace-attribution.md
---
# Serving heuristics: detectors

Each detector is a fixed rule over an XProf trace of a vLLM server, run by
`analyze.py` on the Events DB parquet tpu-diagnostics exports. A finding
names its evidence (op shares, step times, kernel names) and, when a knob
can act on it, the knob change. `propose.py` turns findings into candidates
and ranks them for the target points' phase; every candidate is measured.
The same rules, for agents that read the knowledge base instead:
tpu-knowledge-base entries `vllm-serving-heuristics` and
`vllm-sharding-heuristics`.

## Status tiers

| Status | Meaning | Launched |
| --- | --- | --- |
| `workload_dependent` | A mesh whose benchmark points disagree | yes |
| `unmeasured` | Every other candidate | yes |

Results measured on other models do not keep a candidate from being tried:
each one is measured on this deployment.

A knob the serving image does not have (`knob_probe.py`) is reported, never
launched: the server would run as the baseline.

## Measured on vLLM-TorchTPU (tpu7x 2x2x1)

The mesh change in full: {{data:tpu-knowledge-base/pages/vllm-tp-dp-plan.md}}.

| Detector | Trace signature | Action | Result |
| --- | --- | --- | --- |
| `D7_collective_bound` | collectives >= 25% of a prefill or mixed step | mesh step: fewer TP ranks, more replicas | TP 4 x DP 2 -> TP 2 x DP 4 at the regime peaks, vLLM 0.32.0: Gemma-4-31B +14.8% (1k/1k) and +29.2% (8k/1k), Mistral-Small-24B +32.5% and +35.4%; -16% to -31% at 1 to 64 users |
| `D29_async_scheduling_off` | device idle between decode steps, `--async-scheduling` unset | `--async-scheduling` | no gain twice on Qwen3.8-27B (TTFT +18% at 128 users); a tie at every measured point on Mistral-Small-24B (vLLM 0.32.0) |
| (guardrail) | a finding proposing a higher `--max-num-batched-tokens` | dropped while TTFT matters | median TTFT +13% to +124% on three models, no throughput gain |

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
| `D18_attn_indexer` | sparse-attention indexer (scoring + top-k) is a hotspot |
| `D19_in_step_wait` | TensorCore idles inside the step program |
| `D20_mla_attention_tp` | MLA attention under tensor parallelism |
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

## Reading a trace for sharding

In full, with the KV-cache rules:
{{reference:optimization-heuristics/serving-trace-attribution.md}} (the same text as the
tpu-knowledge-base entry `serving-trace-attribution`).

- An all-reduce in an inference trace is a tensor-parallel collective: one
  per layer after the attention output and MLP down projections. There is no
  gradient all-reduce at inference.
- MoE token dispatch and combine show as all-gather / reduce-scatter pairs,
  not all-to-all; the expert count per device is in the grouped-matmul
  kernel name (`gmm_v2-g_<E>`).
- A long TTFT with a short prefill step is queueing: compute the prefill
  steps per replica, concurrency / DP x input length / max-num-batched-tokens,
  before blaming collectives.
- KV capacity per replica is in each engine's startup log
  (`GPU KV cache size: N tokens`); with DP, each rank logs its own.
