---
title: "Tensor parallelism (Megatron / model parallelism)"
type: concept
tags: [parallelism, performance, interconnect]
created: 2026-09-04
updated: 2026-10-02
sources: 5
human_reviewed: true
aliases: ["TP", "Megatron sharding", "model parallelism", "1D model parallelism", "sequence parallelism"]
---

Tensor parallelism shards the feed-forward dimension of the weights and the model
dimension of the activations: `In[B, D_Y] · W_in[D, F_Y] · W_out[F_Y, D] → Out[B, D_Y]`.
It moves activations instead of weights — an AllGather of `In` before the first matmul and
a ReduceScatter of `Out` after the second, both on the critical path — and so unlocks a
smaller compute-bound batch than FSDP (`index.py get concept-fsdp`), at the price of a hard limit on how many
ways you can shard (scaling-book, Part 5 (`index.py get scaling-book-training`)).

## Training cost model

Forward pass per layer: `T_math = 4·B·D·F / (Y·C)`, `T_comms = 4·B·D / W_ici`, so
compute-bound when

`Y < M_Y · F / α`, with `α = C / W_ici`

— independent of batch size and of precision (int8 halves both comms and FLOPs time)
(scaling-book, Part 5 (`index.py get scaling-book-training`)). On v5p (`α = 2550`) that is 8–16 way
for most models: LLaMA-3 70B (`F ≈ 30k`) is fine 8-way and comms-bound 16-way; Gemma 7B
(`F ≈ 50k`) allows ~19-way (scaling-book, Part 5 (`index.py get scaling-book-training`)). The
per-generation `α` table on critical-batch-size (`index.py get concept-critical-batch-size`) shows the limit
is several times tighter on v6e and TPU7x. The chaining of the two matmuls is what makes
this cheap: gather once before `W_in`, reduce-scatter once after `W_out`, no AllReduce in
between (scaling-book, Part 5 (`index.py get scaling-book-training`)).

## Combining with FSDP

FSDP moves weights and TP moves activations, so each shrinks the other's transfer: TP's
`AllGather_Y([B_X, D_Y])` shrinks as `X` grows, FSDP's `AllGather_X([D_X, F_Y])` shrinks as
`Y` grows. With `N = X·Y` chips and `M_X`, `M_Y` axes, minimizing the larger of the two
comms gives

`X_opt = sqrt(B · M_X · N / (F · M_Y))`, and compute-bound when `B/N > α² / (M_X · M_Y · F)`

≈ 100 tokens/chip on v5p with `F = 32k` — about 8× lower than FSDP alone
(scaling-book, Part 5 (`index.py get scaling-book-training`)). Worked: LLaMA-3 70B, 4M batch, 8,960
v5p chips → `X_opt ≈ 1618` → 2048-way FSDP (1024 data × 2 sequence) + 4-way TP, just above
the 113-token floor (scaling-book, Part 6 (`index.py get scaling-book-applied-training`)).

## Sequence parallelism (section, not a page)

Sharding the sequence axis is data parallelism for the MLP ("tokens are tokens") and only
attention needs extra gathers, handled by ring attention; the book treats it as another DP
axis rather than a separate strategy (scaling-book, Part 5 (`index.py get scaling-book-training`)).
During prefill it is the second lever after TP once TP hits the ICI bound
(scaling-book, Part 7 (`index.py get scaling-book-inference`)).

## Inference: the only option for generation, with different limits

- Generation admits only model-parallel variants: FSDP moves weights over ICI (far slower
  than HBM), DP replicates without speeding loads, and a single token has no sequence to
  shard (scaling-book, Part 7 (`index.py get scaling-book-inference`)).
- **FLOPs-bound limit** as in training, `Y < M_Y·F/α` (~16 chips for LLaMA-3 70B with two
  axes on v5e) (scaling-book, Part 8 (`index.py get scaling-book-applied-inference`)).
- **Latency-bound limit**: when decode is HBM-bound, sharding further still helps latency
  until ICI time exceeds weight-load time, `Y > F / (B · β)` with `β = W_hbm / W_ici` (~8 on
  v5e/v6e) — e.g. `F = 16k, B = 32` allows 64-way with little throughput cost
  (scaling-book, Part 7 (`index.py get scaling-book-inference`)). On TPU7x `β` is several times
  larger, so this headroom shrinks (derived on critical-batch-size (`index.py get concept-critical-batch-size`)).
- Small activations make these collectives latency-bound (`Y > BD/45,000` in int8)
  (collective-costs (`index.py get concept-collective-costs`)).
- Attention: shard `W_Q`/`W_O` over heads Megatron-style; KV weights are small enough to
  replicate beyond `K`-way (scaling-book, Part 7 (`index.py get scaling-book-inference`)). Sharding
  the KV cache (`index.py get concept-kv-cache`) beyond the head count needs batch sharding plus two AllToAlls.
- **2D weight-stationary** (weights sharded on both `D` and `F`, activations on `D`; an
  AllGather over `YZ` then a ReduceScatter over `X`) beats 1D Megatron above
  `N > 32·(F/D)·(3/4)²` ≈ 72 chips for `F = 4D`, and its comms keep falling with `N`
  (scaling-book, Part 7 (`index.py get scaling-book-inference`)).

## What vLLM will start

vLLM checks a TP size against both head counts, at two points:

- **Attention heads** must be divisible by the TP size, or the engine refuses to
  start: "Total number of attention heads (N) must be divisible by tensor
  parallel size (T)" (`ModelConfig.verify_with_parallel_config`).
- **KV heads** must divide evenly one way or the other: the KV heads by the TP
  size when there are at least as many, or the TP size by the KV heads when
  there are fewer, in which case each KV head is replicated `tp / kv_heads`
  times. Otherwise building the attention layer fails an assertion
  (`QKVParallelLinear`, through `divide()`).

So 64 attention heads with 8 KV heads start at TP 8 or TP 16 (two workers to
each KV head); 48 attention heads with 8 KV heads pass the first check at TP 12
and then fail the second; 40 attention heads fail at TP 16 before anything is
built. Which of the sizes that start is fast is the limits above, not this
rule. From vLLM a810316c08:
[`verify_with_parallel_config`](https://github.com/vllm-project/vllm/blob/a810316c0894b8a43e53458ae5b14c9ad0b18b5d/vllm/config/model.py#L1508-L1520),
run whenever `VllmConfig` builds an engine's configuration, and
[`QKVParallelLinear`](https://github.com/vllm-project/vllm/blob/a810316c0894b8a43e53458ae5b14c9ad0b18b5d/vllm/model_executor/layers/linear.py#L1051-L1058),
used by the attention layers of most models. vllm-torchtpu (at `9725110e`)
adds no general rule of its own, and relaxes this one for Qwen2: when the
attention heads do not divide by TP, but divide by the KV heads and TP divides
by the KV heads, it zero-pads each KV group's query heads
([`gqa_head_padding.py`](https://github.com/vllm-project/vllm-torchtpu/blob/9725110e14e01caf8997519320ce5c6e1f5077c5/src/vllm_torchtpu/gqa_head_padding.py#L58-L110)), so 28 attention
heads with 4 KV heads start at TP 8. It is stricter in two places: with
`VLLM_XLA_USE_SPMD` the KV heads must divide by TP
([`runner/kv_cache_manager.py`](https://github.com/vllm-project/vllm-torchtpu/blob/9725110e14e01caf8997519320ce5c6e1f5077c5/src/vllm_torchtpu/runner/kv_cache_manager.py#L1090-L1093)),
and Kimi-K3 requires the attention heads to divide by TP
([`models/vllm/kimi_k3/attention.py`](https://github.com/vllm-project/vllm-torchtpu/blob/9725110e14e01caf8997519320ce5c6e1f5077c5/src/vllm_torchtpu/models/vllm/kimi_k3/attention.py#L226-L227)).

## Pitfalls

- TP collectives sit on the critical path; XLA overlaps them imperfectly, and the book
  reports a latency hit on sharded matmuls above BS≈32 in generation
  (scaling-book, Part 7 (`index.py get scaling-book-inference`)). Collective matmul (`index.py get concept-collective-matmul`)
  is the fix when it matters.
- On GPUs TP is confined to one NVLink node (~8-way, `Y < F/2475`)
  (scaling-book, Part 12 (`index.py get scaling-book-gpus`)); TPU pods lift that limit to the ICI
  bound above.

## See also

- Parallelism strategies (`index.py get concept-parallelism-strategies`) (hub), FSDP (`index.py get concept-fsdp`),
  Collective costs (`index.py get concept-collective-costs`), Prefill vs decode (`index.py get concept-prefill-vs-decode`)

## Sources

- How To Scale Your Model, Part 5: Training Parallelism (scaling-book) (`index.py get scaling-book-training`)
- How To Scale Your Model, Part 6: Training LLaMA 3 on TPUs (scaling-book) (`index.py get scaling-book-applied-training`)
- How To Scale Your Model, Part 7: Inference (scaling-book) (`index.py get scaling-book-inference`)
- How To Scale Your Model, Part 8: Serving LLaMA 3 on TPUs (scaling-book) (`index.py get scaling-book-applied-inference`)
- How To Scale Your Model, Part 12: How to Think About GPUs (scaling-book) (`index.py get scaling-book-gpus`)
