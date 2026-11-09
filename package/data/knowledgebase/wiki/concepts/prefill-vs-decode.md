---
title: "Prefill vs decode (Transformer inference regimes)"
type: concept
tags: [workload, performance, memory, parallelism]
created: 2026-09-04
updated: 2026-10-27
sources: 3
human_reviewed: false
aliases: ["generation", "decode", "prefill", "TTFT", "disaggregated serving", "continuous batching", "speculative sampling"]
---

Transformer inference is two workloads. **Prefill** processes the prompt in one pass and
writes the [KV cache](kv-cache.md); it looks like training and is compute-bound.
**Generation** (decode) emits one token per step per request; it is bandwidth-bound on
weights and KV cache, needs cross-request batching to use the MXU, and admits only model
parallelism. Latency (TTFT, per-token) joins throughput as an objective
([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)).

## Rooflines per phase

| | Prefill | Generation |
|---|---|---|
| Linear ops | compute-bound once prompt > `B_crit` (~240 bf16 v5e); maximize MFU, TTFT follows | compute-bound only if total token batch across requests > `B_crit`; 240 concurrent KV caches is hard ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)) |
| Attention | intensity `T/2`: compute-bound past ~480 tokens | intensity ≈ 1: **always** HBM-bound ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)) |
| Memory | activations negligible (~80 MB for an 8k prompt at `D = 5k`) | params + `B × KV` ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)) |
| Sharding | anything from training: TP up to `F/α` per axis, then sequence parallel ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)) | model parallelism only — FSDP moves weights over ICI (fatal), DP replicates without speeding loads, no sequence to shard ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)) |

## Step time and throughput (the one formula to remember)

`T_step = B·KV_size / W_hbm_total + max(2·B·params / C_total, param_bytes / W_hbm_total)`

— attention term always bandwidth-bound, MLP term compute-bound above `B_crit`. Small-batch
latency floor: `(params + B·KV) / total HBM bandwidth`; throughput ceiling:
`B · W_hbm / (B·KV + params)` ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)).
Weight-load floor alone: 16 GiB (~1.7e10 B) HBM per v5e ÷ 8.2e11 B/s ≈ 21 ms per step (the book,
taking 16 GB as 1.6e10 B, gets ≈ 19 ms) when the cache fills
HBM; LLaMA 3-70B int8 on a v5e 4×2 at BS 32, 8k context: ~17 ms, ~235 tokens/s/chip
([scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)). Doubling the topology halves
latency but not per-chip throughput, and a bigger slice can *raise* throughput when it
lets the batch grow toward `B_crit` ("the smallest topology is not the most performant")
([scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)).

Pareto: batch is the knob — doubling per-token latency can cut per-token cost ~100×;
throughput flattens past `B_crit` (120 with int8 weights); at long context KV loading, not
FLOPs, dominates ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md),
[scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)). Caveats: XLA's overlap is
imperfect, and sharded matmuls take a latency hit above BS≈32
([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)).

## Engine designs

- **Batched prefill then generate**: couples the two batch sizes, terrible TTFT, padded
  prefills, one sharding for both — edge use only
  ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)).
- **Interleaved**: prefill at batch 1, generation batched; great for bulk generation, but
  every prefill pauses everyone's decode ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)).
- **Disaggregated**: separate prefill and generate servers (different topologies and
  shardings), KV shipped over the network; scale the pools independently — ~3 prefill
  servers per generate server for 8k prompts / 512-token decodes on v5e
  ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md),
  [scaling-book, Part 8](../sources/2025-scaling-book-applied-inference.md)).
- **Continuous batching**: compiled prefill (variable length → KV buffer) and generate
  (all active slots) functions under an orchestrator that refills freed slots
  ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)).
- **Prefix caching**: see [KV cache](kv-cache.md#prefix-caching-and-eviction).
- **Speculative sampling**: a cheap drafter proposes `K` tokens, the big model scores them
  in one batched pass — extra FLOPs are free below `B_crit`; latency win, throughput win at
  long context; exact distribution preserved (Metropolis-style rejection for non-greedy);
  drafter heads inside the model avoid needing a separate small model
  ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)).
- Reference engine: JetStream (prefill/generate/transfer threads; `prefill`, `insert`,
  `generate` interface; a PyTorch port exists) ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)).

Metrics these map to: TTFT is prefill latency; TPOT/ITL is `T_step`; per-chip throughput
is cost. Engineering the trade is the [size-serving-topology](../playbooks/size-serving-topology.md)
playbook.

## See also

- [KV cache](kv-cache.md), [Critical batch size](critical-batch-size.md),
  [Tensor parallelism](tensor-parallelism.md#inference-the-only-option-for-generation-with-different-limits),
  [HBM](hbm.md), [Quantization](quantization.md)

## Sources

- [How To Scale Your Model, Part 7: Inference (scaling-book)](../sources/2025-scaling-book-inference.md)
- [How To Scale Your Model, Part 8: Serving LLaMA 3 on TPUs (scaling-book)](../sources/2025-scaling-book-applied-inference.md)
- [How To Scale Your Model, Part 4: Transformer Math (scaling-book)](../sources/2025-scaling-book-transformers.md)
