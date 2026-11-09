---
title: "TPU v6e (Trillium)"
type: hardware
tags: [hardware]
aliases: [Trillium]
created: 2026-08-22
updated: 2026-10-27
sources: 3
human_reviewed: false
---

Sixth-generation efficiency chip (codename Trillium; "v6e" on all technical
surfaces): single-TensorCore, 2D-torus, 256-chip pods; optimized for transformer,
text-to-image, and CNN training/fine-tuning/serving. Specs from
[Cloud TPU v6e docs](../sources/2026-08-cloud-tpu-v6e-docs.md).

## Specifications

| Spec | Value |
|---|---|
| Peak compute / chip | 918 TFLOPs bf16; 1836 TOPs int8 (2×) |
| HBM | 32 GiB, 1638 GBps |
| TensorCores / chip | **1** — not megacore; 2 [MXUs](../concepts/mxu.md) (256×256) |
| [SparseCores](../concepts/sparsecore.md) / chip | 2 |
| [ICI](../concepts/ici.md) | 800 GBps bidi per chip, 4 ports; 2D torus |
| Pod | 256 chips (16x16); 234.9 bf16 PFLOPs; all-reduce 102.4 TB/s; bisection 3.2 TB/s |
| Host | 8 chips; 1536 GiB DRAM; 4×200 Gbps NIC |
| Perf/TCO (expected) | 1× baseline ([TPU v5e](tpu-v5e.md) = 0.65×) |

## Configurations

- 2D slices 1x1 → 16x16 (256 chips). Standard VMs are half-host 4-chip
  (`ct6e-standard-4t`); `v6e-1` for testing; `v6e-8` full-host single-VM optimized
  for inference (all 8 chips in one serving workload); multi-host inference via
  Pathways.

Note: the 256×256 MXU means tensor dims below 256 under-fill the array — see
[Padding](../concepts/padding.md).

## See also

- [Critical batch size](../concepts/critical-batch-size.md) — derived compute-bound thresholds for this generation

- [TPU7x](tpu-7x.md) — next generation

## Sources

- [Google Cloud TPU v6e documentation](../sources/2026-08-cloud-tpu-v6e-docs.md)
- MXU array size (256×256): [scaling-book](../sources/2025-scaling-book-tpus.md)
- HBM capacity unit (GiB): [tpu_info chip table](../sources/2026-10-tpu-info-chip-table.md)
