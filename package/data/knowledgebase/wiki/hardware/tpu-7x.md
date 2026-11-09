---
title: "TPU7x (Ironwood)"
type: hardware
tags: [hardware]
aliases: [Ironwood, TPU7x]
created: 2026-08-22
updated: 2026-10-27
sources: 2
human_reviewed: false
---

First release of the Ironwood (seventh) generation: dual-chiplet scale-out chip for
large dense/MoE training, sampling, and decode-heavy inference; 9216-chip 3D-torus
pods. Specs from [Cloud TPU7x docs](../sources/2026-08-cloud-tpu-7x-docs.md).

## Specifications

| Spec | Value |
|---|---|
| Peak compute / chip | 2307 TFLOPs bf16; 4614 TFLOPs FP8 (2×) |
| HBM | 192 GiB, 7380 GBps (~7.37 TB/s) |
| TensorCores / chip | 2 — as **two chiplets**, separate memory spaces (no megacore) |
| [SparseCores](../concepts/sparsecore.md) / chip | 4 (2 per chiplet) |
| [ICI](../concepts/ici.md) | 1200 GBps bidi per chip; 3D torus, 200 GBps bidi per axis |
| DCN | 100 Gbps per chip |
| Pod | 9216 chips; slices ≥64 chips from 4x4x4 cubes |
| Host VM | 4 chips full-host only (`tpu7x-standard-4t`); 224 vCPU / 960 GB; 2 NUMA nodes |
| Frameworks | JAX, PyTorch; **TensorFlow not supported** |

## Dual-chiplet architecture

Each chiplet = 1 TensorCore + 2 SparseCores + 96 GiB HBM; the two chiplets connect via
a die-to-die link **6× faster than a 1D ICI link**, communicating through collectives.
Frameworks expose each chip as **two devices** (JAX adds a 4th topology dimension) —
a return to the pre-v4 programming model; mesh math and profiler output count
chiplets, not chips. See [Megacore](../concepts/megacore.md) for the contrast.

## Notes for optimization

- [VMEM](../concepts/vmem.md) buffer size is tunable;
  [Pallas](../codebases/pallas.md) kernel block sizes are constrained by available
  VMEM.
- Large HBM (192 GiB) enables big batches, but memory-bound vector ops can still
  bottleneck on HBM — see [Roofline analysis](../concepts/roofline-analysis.md).
- PCIe host offload (activations/optimizer state) is an explicit supported pattern.
- Hyperdisk Balanced/ML storage; All Capacity mode on GKE for full-topology
  visibility.

## See also

- [Critical batch size](../concepts/critical-batch-size.md) — derived compute-bound thresholds for this generation

- [TPU v5p](tpu-v5p.md) — the prior scale-out generation; [TPU 8t](tpu-8t.md) /
  [TPU 8i](tpu-8i.md) — announced successors

## Sources

- [Google Cloud TPU7x documentation](../sources/2026-08-cloud-tpu-7x-docs.md)
- HBM capacity unit (GiB): [tpu_info chip table](../sources/2026-10-tpu-info-chip-table.md)
