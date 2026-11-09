---
title: "Size a Serving Topology and Batch for Decode"
type: playbook
tags: [workload, performance, memory, precision, parallelism]
created: 2026-09-04
updated: 2026-10-27
sources: 3
human_reviewed: false
aliases: ["inference sizing", "serving recipe", "latency throughput tradeoff"]
target_imports:
- serving_topology
---
# Size a Serving Topology and Batch for Decode

From a model config, a target context length, a precision plan and a latency budget,
derive the smallest slice that fits, the batch that saturates it, the decode step-time
floor, how much model parallelism is allowed, and the prefill:generate server ratio. The
book's Part 7/8 procedure as steps ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md),
[scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)).

## Applies when

- Dense decoder LLM serving on TPU with a KV cache; metrics are TTFT, per-token latency
  (TPOT/ITL) and tokens/s/chip.
- Inputs: `L, D, F, N, K, H, V`, weight and KV dtypes, context `S`, generation `C`, HBM per
  chip and bandwidth, ICI per axis (see [critical-batch-size](../concepts/critical-batch-size.md)).

## Steps

1. **Bytes per token of KV cache**: `2 · bytes · K · H · L` (LLaMA 3-70B int8: 160 kB).
   Signal: per-sequence cache at `S` (5.3 GB at 32k) ([kv-cache](../concepts/kv-cache.md)).
2. **Parameter bytes** by dtype (70B: 140 GB bf16, 70 GB int8, 35 GB int4)
   ([scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)).
3. **Minimum slice by memory**: `ceil((params + B·S·KV) / HBM_per_chip)` rounded to a real
   topology. Signal: for LLaMA 3-70B on v5e (16 GiB): bf16 4×4, int8 4×2, int4 2×2 — with
   only ~43 KV caches of 8k left, i.e. tiny batches ([scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)).
4. **Critical batch** for the dtype plan: `B_crit = β · C / W_hbm` (240 bf16, 120 int8
   weights + bf16 FLOPs on v5e; per-generation on
   [critical-batch-size](../concepts/critical-batch-size.md)). Throughput saturates near it.
5. **Grow the topology until the batch approaches `B_crit`** if throughput is the goal:
   free HBM after weights ÷ per-sequence KV = max batch; per-chip throughput keeps rising
   with batch while step time is weight-load-bound (4×8 bf16 → batch 140, 1.6× the QPS/chip
   of 4×4) ([scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)).
6. **Step-time floor**: `B·KV / W_hbm_total + max(2·B·params / C_total, params / W_hbm_total)`.
   Signal: 17 ms at BS 32 int8 on a v5e 4×2; ~20 ms when the cache fills the chip's 16 GiB
   ([prefill-vs-decode](../concepts/prefill-vs-decode.md)).
7. **Model-parallel limits.** Compute-bound: `Y ≤ M_Y·F/α_ici` (≈16–26 for `F = 28.7k` on
   v5e with two axes). Latency-bound (small batch): `Y ≤ F / (B·β)`, `β = W_hbm/W_ici`
   (~8 v5e/v6e; much larger on TPU7x). Check `T_ici = 2BD/W_ici` < `T_hbm = 2DF/(Y·W_hbm)`
   for one matmul ([tensor-parallelism](../concepts/tensor-parallelism.md#inference-the-only-option-for-generation-with-different-limits)).
   Signal: 4×4 OK for bf16; int8/int4 configs fit pure model parallelism; 4×8 bf16 does
   not ([scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)).
8. **Shard the KV cache**: heads first (≤ `K` ways), then batch with two AllToAlls per
   layer; never replicate ([kv-cache](../concepts/kv-cache.md#sharding-the-cache)).
9. **Prefill budget**: `T_prefill ≈ 2·params·P / (N_chips · C · MFU)` at ~40% MFU (8k prompt
   on 16 v5e: ~0.9 s); shard prefill with TP up to the ICI bound then sequence parallel
   ([scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)).
10. **Disaggregation ratio**: prefill servers `P` and generate servers `G` balance when
    `P / T_prefill = B·G / (T_step · median_decode_len)` → ~3:1 for 8k/512 on v5e
    ([scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)).

## Pitfalls

- Sizing on the smallest slice that fits: it usually leaves no batch and terrible
  utilization ([scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)).
- Any FSDP or data parallelism in the generate config
  ([fsdp](../concepts/fsdp.md#when-fsdp-is-the-wrong-tool)).
- Ignoring working memory: the book's throughput numbers are ~2× optimistic
  ([scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)).
- Small-batch collectives are latency-bound below ~45 kB per shard
  ([collective-costs](../concepts/collective-costs.md#latency-bound-regime)).
- v5e-derived numbers reused on v6e (32 GiB) or TPU7x (192 GiB, far higher HBM bandwidth):
  recompute every step.

## Verification

Measured TPOT vs the step 6 floor (expect within ~1.5× when comms are modest); HBM and
KV-cache usage vs step 3/5; throughput flattening near `B_crit`; TTFT vs step 9. Trace per
[profile-tpu-program](profile-tpu-program.md): attention fusions bandwidth-bound, no
unexpected weight collectives.

## See also

- [Prefill vs decode](../concepts/prefill-vs-decode.md), [KV cache](../concepts/kv-cache.md),
  [Quantization](../concepts/quantization.md), [HBM](../concepts/hbm.md)

## Sources

- [How To Scale Your Model, Part 7: Inference (scaling-book)](../sources/2025-scaling-book-inference.md)
- [How To Scale Your Model, Part 8: Serving LLaMA 3 on TPUs (scaling-book)](../sources/2025-scaling-book-applied-inference.md)
- [How To Scale Your Model, Part 4: Transformer Math (scaling-book)](../sources/2025-scaling-book-transformers.md)
