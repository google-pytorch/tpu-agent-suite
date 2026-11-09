---
title: "Data parallelism"
type: concept
tags: [parallelism, performance, memory]
created: 2026-09-04
updated: 2026-10-27
sources: 3
human_reviewed: false
aliases: ["DP", "pure data parallelism"]
---

Data parallelism shards activations along the batch (`In[B_X, D]`) and replicates weights
and optimizer state on every chip. The forward pass needs no communication; the backward
pass AllReduces each weight gradient, off the critical path
([scaling-book, Part 5](../sources/2025-scaling-book-training.md)). It is the right default whenever the
model fits one chip and the per-chip batch is large enough.

## Cost model

Per layer in the backward pass: `T_math = 8·B·D·F / (X·C)` and `T_comms = 8·D·F / W_ici`
(two AllReduces of `2DF` bytes each), so the scheme is compute-bound when

`B / X > C / W_ici = α`

i.e. the per-chip **token** batch must exceed the ICI arithmetic intensity: 2,550 on v5p
over one axis, 850 with all three axes of a v5p pod
([scaling-book, Part 5](../sources/2025-scaling-book-training.md)). Per-generation `α` values,
including the much higher v6e/TPU7x figures, are tabulated on
[critical-batch-size](critical-batch-size.md).

## Memory ceiling

With bf16 params and fp32 Adam state a parameter costs ~10 bytes, so pure DP can hold at
most `HBM / 10` parameters per chip — ~10 B on a 95 GiB v5p, before any activation
checkpoints ([scaling-book, Part 5](../sources/2025-scaling-book-training.md)). That is why DP alone is
rarely used for interesting models; [FSDP](fsdp.md) keeps the same roofline and drops the
replication.

## Variants and limits

- **Multiple mesh axes**: spanning `M_X` axes gives `M_X×` bandwidth, so the threshold
  becomes `α / M_X` ([scaling-book, Part 5](../sources/2025-scaling-book-training.md)).
- **Context / sequence parallelism** is data parallelism over the sequence axis: for the
  MLP "tokens are tokens"; attention needs KV/Q gathers, typically ring attention
  ([scaling-book, Part 5](../sources/2025-scaling-book-training.md)).
- **Across pods**: DP over [DCN](dcn.md) is compute-bound when the batch per ICI slice
  exceeds `C / W_dcn` (~73k tokens on v5p); DCN bandwidth grows with slice size because
  each host adds NICs ([scaling-book, Part 5](../sources/2025-scaling-book-training.md)).
- **Critical-batch paradox**: fewer tokens per step makes DP *more* comms-bound, so the
  design problem is fitting a scaling-law-chosen batch onto a large chip count
  ([scaling-book, Part 5](../sources/2025-scaling-book-training.md)).
- **MoE**: the threshold inflates by `E/k` because weights grow `E×` while FLOPs grow
  only `k×` ([expert-parallelism](expert-parallelism.md)).
- **Inference**: DP is pointless for generation — it replicates weights without loading
  them faster; run separate model replicas instead
  ([scaling-book, Part 7](../sources/2025-scaling-book-inference.md)).
- On GPUs the same rule gives ~2,200 tokens per H100 in-node, ~2,475 across nodes
  ([scaling-book, Part 12](../sources/2025-scaling-book-gpus.md)).

## See also

- [Parallelism strategies](parallelism-strategies.md) (hub), [FSDP](fsdp.md),
  [Collective costs](collective-costs.md)

## Sources

- [How To Scale Your Model, Part 5: Training Parallelism (scaling-book)](../sources/2025-scaling-book-training.md)
- [How To Scale Your Model, Part 7: Inference (scaling-book)](../sources/2025-scaling-book-inference.md)
- [How To Scale Your Model, Part 12: How to Think About GPUs (scaling-book)](../sources/2025-scaling-book-gpus.md)
