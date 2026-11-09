---
title: "Megacore"
type: concept
tags: [hardware]
created: 2026-08-22
updated: 2026-10-27
sources: 4
human_reviewed: false
---

Megacore is the chip organization where two TPU cores share memory and present as one
logical accelerator with doubled FLOPs — the layout on [v4](../hardware/tpu-v4.md)
(unified 32-GiB HBM across both TensorCores,
[v4 docs](../sources/2026-08-cloud-tpu-v4-docs.md)) and
[v5p](../hardware/tpu-v5p.md). Single-core chips (v5e, v6e) and v3-and-older
(two independent-memory cores = two accelerators) sit outside it
([source](../sources/2025-scaling-book-tpus.md)).

**Resolved contradiction** *(human-adjudicated 2026-08-24)*: the
[scaling-book](../sources/2025-scaling-book-tpus.md) claims megacore "is true for v4,
v5, and v6 TPUs", but the official
[v6e docs](../sources/2026-08-cloud-tpu-v6e-docs.md) state each v6e chip has **one**
TensorCore. Official docs win: **v6e is not megacore**. The book claim likely
anticipated a v6p that never shipped.

**TPU7x (Ironwood) replaces megacore with dual chiplets**: two chiplets with
*separate* memory spaces (1 TensorCore + 2 SparseCores + 96 GiB HBM each), joined by a
die-to-die link 6× faster than a 1D ICI link. Frameworks expose each chip as **two
devices** (JAX adds a 4th topology dimension) — a return to the pre-v4 programming
model ([TPU7x docs](../sources/2026-08-cloud-tpu-7x-docs.md)).

Practical consequence: "device" counts and per-device FLOPs/memory differ by
generation — a v5e host exposes 8 chips = 8 cores, while a training-chip tray is 4
chips ≈ 4 logical megacores — which matters when mapping meshes and interpreting
profiler output ([source](../sources/2025-scaling-book-tpus.md)).

## See also

- [TensorCore](tensorcore.md)

## Sources

- [How to Think About TPUs (scaling-book)](../sources/2025-scaling-book-tpus.md)
- [Cloud TPU v4 docs](../sources/2026-08-cloud-tpu-v4-docs.md),
  [v6e docs](../sources/2026-08-cloud-tpu-v6e-docs.md),
  [TPU7x docs](../sources/2026-08-cloud-tpu-7x-docs.md)
