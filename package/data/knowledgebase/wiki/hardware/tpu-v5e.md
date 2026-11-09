---
title: "TPU v5e"
type: hardware
tags: [hardware]
created: 2026-08-22
updated: 2026-10-27
sources: 2
human_reviewed: false
---

Fifth-generation efficiency chip: single-TensorCore, 2D-torus, 256-chip pods; combined
training + serving product (training jobs optimized for throughput, serving for
latency). Specs from [Cloud TPU v5e docs](../sources/2026-08-cloud-tpu-v5e-docs.md).

## Specifications

| Spec | Value |
|---|---|
| Peak compute / chip | 197 TFLOPs bf16; 393 TOPs int8 (2×) |
| HBM | 16 GiB, 800 GiBps |
| TensorCores / chip | 1; 4 MXUs |
| [ICI](../concepts/ici.md) | 400 GBps bidi per chip, 4 ports; 2D torus |
| Pod | 256 chips (16x16); 50.63 bf16 PFLOPs; all-reduce 51.2 TB/s; bisection 1.6 TB/s |
| Host | 8 chips (4x2); 512 GiB DRAM; 2×100 Gbps NIC |
| Perf/TCO (expected) | 0.65× ([TPU v6e](tpu-v6e.md) = 1×) |

## Configurations

- 2D slices 1x1 → 16x16; training up to 256 chips; single-host serving up to 8 chips
  (1x1/2x2/2x4), multi-host serving via Sax.
- VM types `ct5lp-hightpu-{1,4,8}t`; the 8-chip VM spans 2 NUMA nodes — keep CPU↔chip
  traffic NUMA-local.

## See also

- [Critical batch size](../concepts/critical-batch-size.md) — derived compute-bound thresholds for this generation

- [TPU v6e](tpu-v6e.md) — successor in the efficiency line

## Sources

- [Google Cloud TPU v5e documentation](../sources/2026-08-cloud-tpu-v5e-docs.md)
- HBM capacity unit (GiB): [tpu_info chip table](../sources/2026-10-tpu-info-chip-table.md)
