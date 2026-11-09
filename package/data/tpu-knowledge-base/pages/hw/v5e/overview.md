---
title: Cloud TPU v5e Hardware Overview
type: hardware
tags:
- hardware
created: '2026-10-03'
human_reviewed: false
source_ids:
- cloud-tpu-v5e
- xla-scoped-vmem
- openxla-tiled-layout
- openxla-sparsecore
- tpu-info-chips
generation_recipe: overview.recipe.json
---

# Cloud TPU v5e Hardware Overview

## Chip Compute and Architecture

Cloud TPU v5e is an accelerator designed for both training and inference workloads [^cloud-tpu-v5e]. Each v5e processor chip incorporates a single TensorCore (TC) containing four matrix-multiply units (MXUs), one vector unit, and one scalar unit [^cloud-tpu-v5e].

The generic OpenXLA layout reference describes 32-bit values using an `8x128` vector register layout [^openxla-tiled-layout]. In XLA, memory is mapped using 2D tiled formats matching these registers [^openxla-tiled-layout]. To optimize memory transfer and processing, lower-precision data formats are packed into native 32-bit elements: 16-bit brain floating-point (BF16) data uses `(8,128)(2,1)` tiling, and 8-bit integer (Int8) data uses `(8,128)(4,1)` tiling [^openxla-tiled-layout]. Reduced register tiling (`2x128` or `4x128`) is applied when minor dimensions are narrow to prevent unnecessary padding [^openxla-tiled-layout].

| Compute Metric | Value |
| :--- | :--- |
| TensorCores per Chip | 1 [^cloud-tpu-v5e] |
| MXUs per TensorCore | 4 [^cloud-tpu-v5e] |
| Peak Compute per Chip (BF16) | 197 TFLOPs [^cloud-tpu-v5e] |
| Peak Compute per Chip (Int8) | 393 TOPs [^cloud-tpu-v5e] |

## Memory Hierarchy and Buffer Limits

Each TPU v5e chip is provisioned with High Bandwidth Memory (HBM) alongside dedicated on-chip scratchpad storage [^cloud-tpu-v5e], [^xla-scoped-vmem].

* **High Bandwidth Memory (HBM):** 16 GiB capacity per chip with an interconnect bandwidth of 800 GiBps [^cloud-tpu-v5e] [^tpu-info-chips]. *(knowledge base note: the v5e page writes 16 GB; HBM capacities are binary, and the tpu_info chip table gives 16 GiB.)*
* **Host DRAM:** 512 GiB of main memory per physical 8-chip host [^cloud-tpu-v5e].
* **Vector Memory (VMEM):** Local on-chip scratchpad memory utilized exclusively by the TensorCore for instruction-scoped temporaries (such as register spills and double-buffered operand spans) and program-scoped values (intermediate HLO arrays) [^xla-scoped-vmem].
* **Usable Scoped VMEM Limit:** 131,072 KiB / 134,217,728 bytes (128 MiB, derived) [^xla-scoped-vmem].

*Memory Distinction:* Usable scoped VMEM is a compiler-enforced software ceiling for active instruction and kernel allocations, not a direct measurement of physical on-chip SRAM [^xla-scoped-vmem]. Hardware reserves an unspecified portion of physical SRAM for internal operations, restricting the total allocation available for scoped operations [^xla-scoped-vmem].

## Interconnect and Pod Scale

TPU v5e chips communicate over a dedicated Inter-Chip Interconnect (ICI) configured in a 2D torus topology [^cloud-tpu-v5e]. A full TPU Pod scales to 256 chips across 32 host systems [^cloud-tpu-v5e].

| Interconnect & Network Specification | Value |
| :--- | :--- |
| ICI Bidirectional Bandwidth (per chip) | 400 GBps [^cloud-tpu-v5e] |
| ICI Ports per Chip | 4 [^cloud-tpu-v5e] |
| Interconnect Topology | 2D torus [^cloud-tpu-v5e] |
| TPU Pod Size | 256 chips (32 hosts) [^cloud-tpu-v5e] |
| Pod Peak Compute (BF16) | 50.63 PFLOPs [^cloud-tpu-v5e] |
| Pod Peak Compute (Int8) | 100 PetaOps [^cloud-tpu-v5e] |
| Pod All-Reduce Bandwidth | 51.2 TB/s [^cloud-tpu-v5e] |
| Pod Bisection Bandwidth | 1.6 TB/s [^cloud-tpu-v5e] |
| Host NIC Configuration | 2 x 100 Gbps NICs per host [^cloud-tpu-v5e] |
| Data Center Network (DCN) Bandwidth per Pod | 6.4 Tbps [^cloud-tpu-v5e] |

## Slice and Host Configurations

Physical hosts integrate 8 v5e chips [^cloud-tpu-v5e]. Slices can be provisioned as fractional-host partitions (1 or 4 chips), single full hosts (8 chips), or multi-host training pods up to 256 chips [^cloud-tpu-v5e].

### VM Machine Types and NUMA Topologies

Host memory and vCPU resources scale based on the requested VM type [^cloud-tpu-v5e]. Single-chip and 4-chip VM allocations map to a single NUMA domain [^cloud-tpu-v5e]. The 8-chip VM features two NUMA domains; CPU-to-TPU communication is faster when pinned to local NUMA partitions (for example, `CPU0-Chip0` maintains lower latency than cross-node `CPU0-Chip4` transfers) [^cloud-tpu-v5e].

| VM Type | Machine Type | TPU Chips | vCPUs per VM | RAM per VM | NUMA Nodes per VM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1-chip VM | `ct5lp-hightpu-1t` | 1 | 24 | 48 GB | 1 |
| 4-chip VM | `ct5lp-hightpu-4t` | 4 | 112 | 192 GB | 1 |
| 8-chip VM | `ct5lp-hightpu-8t` | 8 | 224 | 384 GB | 2 |

*Note: All values in table directly sourced [^cloud-tpu-v5e].*

### Supported 2D Topology Slices

| Topology | Number of TPU Chips | Host Equivalent | Primary Provisioning Role |
| :--- | :--- | :--- | :--- |
| 1x1 | 1 | 1/8 | Serving (`ct5lp-hightpu-1t`) [^cloud-tpu-v5e] |
| 2x2 | 4 | 1/2 | Serving (`ct5lp-hightpu-4t`) [^cloud-tpu-v5e] |
| 2x4 | 8 | 1 | Serving (`ct5lp-hightpu-8t`) [^cloud-tpu-v5e] |
| 4x4 | 16 | 2 | Training (`ct5lp-hightpu-4t`) [^cloud-tpu-v5e] |
| 4x8 | 32 | 4 | Training (`ct5lp-hightpu-4t`) [^cloud-tpu-v5e] |
| 8x8 | 64 | 8 | Training (`ct5lp-hightpu-4t`) [^cloud-tpu-v5e] |
| 8x16 | 128 | 16 | Training (`ct5lp-hightpu-4t`) [^cloud-tpu-v5e] |
| 16x16 | 256 | 32 | Training (`ct5lp-hightpu-4t`) [^cloud-tpu-v5e] |

Multi-host training configurations exclusively deploy nodes using the `ct5lp-hightpu-4t` machine type across all multi-host slices (16 to 256 chips) [^cloud-tpu-v5e].

## Software and Operational Constraints

* **Platform and Orchestration:** Cloud TPU v5e is managed via Google Kubernetes Engine (GKE) and the Cloud TPU API [^cloud-tpu-v5e]. The Cloud TPU API is in maintenance mode and receives only security updates and bug fixes rather than active feature development [^cloud-tpu-v5e].
* **Workload-Targeted Provisioning:** Slices provisioned for training are optimized for throughput and availability, whereas slices provisioned for serving target latency minimization [^cloud-tpu-v5e]. Running training jobs on serving instances risks reduced availability; running serving jobs on training instances risks higher latency [^cloud-tpu-v5e].
* **Serving Scaling Constraints:** Single-host serving natively supports slices up to 8 chips (1x1, 2x2, 2x4) [^cloud-tpu-v5e]. Multi-host serving exceeding 8 chips requires external coordination using Sax [^cloud-tpu-v5e].
* **Scoped VMEM Limits and Debugging:** If an XLA HLO instruction or custom kernel requires more temporary memory than the 128MB ceiling, compilation aborts with error `E1001` (`RESOURCE_EXHAUSTED: Ran out of memory in memory space vmem`) [^xla-scoped-vmem]. Kernel authors must mitigate this by reducing block tiling sizes, passing a kernel-specific `vmem_limit_bytes` parameter, constraining buffer memory spaces, or setting the global flag `--xla_tpu_scoped_vmem_limit_kib` [^xla-scoped-vmem].

## Gaps and Unspecified Specifications

* **SparseCore Support:** The supplied SparseCore architecture reference does not establish generation-specific SparseCore specifications for TPU v5e [^openxla-sparsecore]. The supplied evidence contains no data indicating whether v5e contains SparseCores or lacks the unit.
* **Physical SRAM:** The raw physical on-chip SRAM capacity, hardware layout, and exact reserved overhead are unspecified; only the compiler-enforced usable scoped VMEM limit (128MB) is documented [^xla-scoped-vmem].
* **Semiconductor and Operating Parameters:** Process node geometry, die size, operating clock frequencies, thermal design power (TDP), and packaging details are absent from the provided technical sources.

## Detailed memory and profiling

See [the generation-specific memory and buffer reference](memory-buffers.md) for tile sizing, padding and XProf inspection.

<!-- sources:generated:start -->

## References

[^cloud-tpu-v5e]: [Google Cloud TPU v5e](https://docs.cloud.google.com/tpu/docs/v5e). specification. Generation specifications, topology and host configurations. URL checked 2026-10-03.

[^xla-scoped-vmem]: [OpenXLA E1001: scoped VMEM OOM](https://github.com/openxla/xla/blob/main/docs/errors/error_1001.md). compiler-reference. Typical usable scoped allocation limits, not physical SRAM totals. URL checked 2026-10-03.

[^openxla-tiled-layout]: [OpenXLA tiled layout](https://github.com/openxla/xla/blob/main/docs/tiled_layout.md). compiler-reference. Tiling and padding; not generation-specific physical capacities. URL checked 2026-10-03.

[^openxla-sparsecore]: [OpenXLA SparseCore](https://github.com/openxla/xla/blob/main/docs/sparsecore.md). architecture-reference. SparseCore organization, scratch sizing and embedding stack estimates. URL checked 2026-10-03.

<!-- sources:generated:end -->

[^tpu-info-chips]: [tpu_info chip table](https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/blob/main/tpu_info/tpu_info/device.py). code-reference. `TpuChip`: HBM per chip in GiB (`hbm_gib`) and devices per chip, the table `compute inspect` sizes from; checked at commit 35b5377566aa. URL checked 2026-10-27.
