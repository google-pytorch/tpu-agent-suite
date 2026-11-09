---
title: TPU v6e Hardware Overview
type: hardware
tags:
- hardware
created: '2026-10-03'
human_reviewed: false
source_ids:
- cloud-tpu-v6e
- cloud-tpu-7x
- xla-scoped-vmem
- openxla-tiled-layout
- openxla-sparsecore
- tpu-info-chips
generation_recipe: overview.recipe.json
---

# TPU v6e Hardware Overview

Cloud TPU v6e (code-named Trillium) is a Google accelerator designed for transformer, text-to-image, and convolutional neural network (CNN) training, fine-tuning, and serving [^cloud-tpu-v6e]. Across APIs and system logs, Trillium is formally designated as `v6e` [^cloud-tpu-v6e].

## Chip Compute and Architecture

Each TPU v6e chip incorporates one TensorCore and two SparseCores [^cloud-tpu-v6e] [^cloud-tpu-7x] [^openxla-sparsecore]. The internal compute engine within each TensorCore comprises two matrix-multiply units (MXUs), one vector unit, and one scalar unit [^cloud-tpu-v6e]. The generic OpenXLA layout reference describes 32-bit `8x128` vector registers [^openxla-tiled-layout]. Floating-point and integer datatypes leverage specialized nested tiling to pack into 32-bit hardware registers, including `(8,128)(2,1)` tiling for BF16 and `(8,128)(4,1)` tiling for 8-bit integers [^openxla-tiled-layout].

The SparseCore is an on-chip tiled coprocessor specialized for irregular, dynamic, and sparse memory access patterns, such as embedding lookups in large embedding models [^openxla-sparsecore]. Each SparseCore contains 16 execution tiles coordinated by a centralized sequencer and supports small-vector execution with a SIMD width of 8 elements for F32 and 16 elements for BF16 [^openxla-sparsecore].

Vendor-stated peak compute throughputs per chip are:
* **BF16 compute:** 918 TFLOPs [^cloud-tpu-v6e] [^cloud-tpu-7x]
* **FP8 compute:** 918 TFLOPs, from the TPU7x comparison table, equal to BF16 [^cloud-tpu-7x]. The v6e page itself lists only BF16 and Int8 [^cloud-tpu-v6e]. *(knowledge base note: neither page says whether v6e executes FP8 natively. The inference is that it does not, so FP8 runs at best at the BF16 rate: the same table lists v5p FP8 equal to its BF16 too, and FP8 matmuls through torch_tpu measured 6–7× slower than BF16, `index.py get v6e-measured-matmul-quantization`.)*
* **Int8 compute:** 1836 TOPs [^cloud-tpu-v6e]
* **Performance / Total Cost of Ownership (TCO):** 1 (expected vendor baseline metric) [^cloud-tpu-v6e]

## Memory Hierarchy and Buffer Limits

The memory subsystem on TPU v6e is divided into high-bandwidth memory (HBM), on-chip scratchpad memory for TensorCores, scratchpad memory for SparseCores, and host-attached main memory [^cloud-tpu-v6e] [^openxla-sparsecore]:

* **High-Bandwidth Memory (HBM):** Each v6e chip integrates 32 GiB of HBM providing 1638 GBps of memory bandwidth [^cloud-tpu-v6e] [^cloud-tpu-7x] [^openxla-sparsecore] [^tpu-info-chips]. *(knowledge base note: the v6e specification writes 32 GB; the comparison table and the tpu_info chip table give 32 GiB.)* HBM accommodates dense model weights, communication stack allocations, dynamically prefetched host data, and embedding tables [^openxla-sparsecore].
* **Vector Memory (VMEM):** VMEM serves as the local scratchpad memory used exclusively by the TensorCore [^xla-scoped-vmem]. While physical SRAM capacity is not stated, XLA enforces a usable instruction-scoped VMEM allocation ceiling of 131,008 KiB or 134,152,192 bytes (127.9375 MiB, derived; the source labels this approximately 127.9 MB) [^xla-scoped-vmem]. This scoped allocation limit can be tuned globally via compiler flags or per kernel in frameworks like Pallas [^xla-scoped-vmem].
* **SparseCore Scratchpad Memory (SPMEM):** Each SparseCore contains an explicitly managed local SRAM scratchpad that stages lookups, non-elementwise reductions, and temporary data to avoid high HBM latency [^openxla-sparsecore]. The SparseCore runtime lacks dynamic scratchpad allocation; buffer sizing inside SPMEM is strictly static and determined at compile time using ID limit settings [^openxla-sparsecore].
* **Host DRAM:** Physical hosts provide 1536 GiB of DRAM across their 8 hosted TPU chips [^cloud-tpu-v6e].

## Interconnect and Pod Scale

TPU v6e chips interconnect using direct inter-chip interconnect (ICI) links arranged in a 2-dimensional (2D) torus topology [^cloud-tpu-v6e]:

* **ICI Configuration:** Each chip exposes 4 ICI ports, supplying 800 GBps of bidirectional ICI bandwidth per chip [^cloud-tpu-v6e] [^cloud-tpu-7x].
* **Pod-Level Scale:** A full TPU v6e Pod contains 256 chips arranged in a 16x16 2D grid [^cloud-tpu-v6e].
* **Pod-Level Throughput Metrics:** A full 256-chip Pod yields 234.9 PFLOPs of peak BF16 compute [^cloud-tpu-v6e]. The Pod provides an all-reduce bandwidth of 102.4 TB/s and a bisection bandwidth of 3.2 TB/s [^cloud-tpu-v6e].
* **Data Center Network (DCN):** Each host includes four 200 Gbps network interface cards (4 x 200 Gbps NICs), resulting in 100 Gbps of DCN bandwidth per chip and 25.6 Tbps of aggregate DCN bandwidth across a 256-chip Pod [^cloud-tpu-v6e] [^cloud-tpu-7x].

## Slice and Host Configurations

A physical v6e host houses 8 TPU chips and 1536 GiB DRAM [^cloud-tpu-v6e]. Virtual machine (VM) instances slice physical hosts into sub-host, single-host, or multi-host partitions [^cloud-tpu-v6e]. The primary multi-host building block is a half-host VM instance containing 4 chips [^cloud-tpu-v6e].

### VM Instance Types

| VM Type | Chips per VM | vCPUs per VM | RAM (GB) per VM | NUMA Nodes per VM |
| :--- | :--- | :--- | :--- | :--- |
| 1-chip VM (`ct6e-standard-1t`) | 1 | 44 | 176 | 1 |
| 4-chip VM (`ct6e-standard-4t`) | 4 | 180 | 720 | 1 |
| 8-chip VM (`ct6e-standard-8t`) | 8 | 360 | 1440 | 2 |

*All VM specifications are directly reported by Cloud TPU documentation [^cloud-tpu-v6e] [^cloud-tpu-7x].*

### Supported 2D Slice Topologies

| Topology | Total Chips | Physical Hosts | VMs | Machine Type | Deployment Scope |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1x1 | 1 | 1/8 | 1 | `ct6e-standard-1t` | Sub-host |
| 2x2 | 4 | 1/2 | 1 | `ct6e-standard-4t` | Sub-host |
| 2x4 | 8 | 1 | 1 | `ct6e-standard-8t` | Single-host |
| 2x4 | 8 | 1 | 2 | `ct6e-standard-4t` | Multi-host (GKE API only) |
| 4x4 | 16 | 2 | 4 | `ct6e-standard-4t` | Multi-host |
| 4x8 | 32 | 4 | 8 | `ct6e-standard-4t` | Multi-host |
| 8x8 | 64 | 8 | 16 | `ct6e-standard-4t` | Multi-host |
| 8x16 | 128 | 16 | 32 | `ct6e-standard-4t` | Multi-host |
| 16x16 | 256 | 32 | 64 | `ct6e-standard-4t` | Multi-host |

*Topologies and configurations are defined in the Cloud TPU v6e documentation [^cloud-tpu-v6e].*

Single-host slices with 8 chips attached to one VM (`v6e-8`) are explicitly optimized for inference workloads [^cloud-tpu-v6e]. Multi-host inference can also be orchestrated across slices using Pathways on Cloud [^cloud-tpu-v6e].

## Software and Operational Constraints

* **API & Networking Restrictions:** Deploying an 8-chip (2x4) slice attached to two VMs is exclusively supported via the Google Kubernetes Engine (GKE) API [^cloud-tpu-v6e]. Utilizing a full-host VM (`v6e-8` with 1 VM) across dual networks is explicitly discouraged due to negative performance impacts [^cloud-tpu-v6e].
* **SparseCore Implementation:** The SparseCore runtime is proprietary, maintained internally by Google, and distributed as a binary via `libtpu.so` [^openxla-sparsecore]. APIs are exposed via TensorFlow Recommenders (`TPUEmbedding`), JAX (`jax-tpu-embedding`), and Keras [^openxla-sparsecore].
* **SparseCore Partition Limits:** Sharding across SparseCores operates via modulo hashing (`col_id % total_sparse_cores`) [^openxla-sparsecore]. Overrunning static compile-time limits (`max_ids_per_partition` or `max_unique_ids_per_partition`) triggers compilation failures or runtime errors unless `allow_id_dropping=True` is explicitly toggled, which drops exceeding IDs at the cost of model accuracy [^openxla-sparsecore].
* **Memory Pipelining Trade-Offs:** The compiler supports overlapping execution between TensorCores and SparseCores [^openxla-sparsecore]. Although this can accelerate device step time by up to an estimated 2x, SparseCores may compute against stale gradients (e.g., delayed by two execution steps), requiring empirical verification of model numerical convergence [^openxla-sparsecore].
* **Alignment Padding:** Embedding tables must be 32-byte aligned along the feature dimension (equivalent to 8 F32 elements), causing derived padding overhead of up to 87.5% (7/8) for 1-float feature dimensions [^openxla-sparsecore].

## Gaps and Unspecified Specifications

Several lower-level physical hardware specifications are not disclosed in the provided technical documentation:
* **Physical SRAM:** Total on-die physical SRAM capacity for TensorCore VMEM and SparseCore SPMEM is omitted; only the usable compiler-scoped limit (127.9 MB) is established for VMEM [^xla-scoped-vmem] [^openxla-sparsecore].
* **Physical Packaging & Clocks:** Die size, process node, thermal design power (TDP), voltage specifications, and base/boost operating clock frequencies for TensorCores, SparseCores, and memory buses are undocumented [^cloud-tpu-v6e] [^cloud-tpu-7x] [^openxla-sparsecore].
* **Host PCIe Topology:** The exact generation, lane count, and physical topology of the PCIe bus connecting the host CPU to the eight v6e accelerator chips are not specified [^cloud-tpu-v6e] [^cloud-tpu-7x].

## Detailed memory and profiling

See [the generation-specific memory and buffer reference](memory-buffers.md) for tile sizing, padding and XProf inspection.

<!-- sources:generated:start -->

## References

[^cloud-tpu-v6e]: [Google Cloud TPU v6e](https://docs.cloud.google.com/tpu/docs/v6e). specification. Generation specifications, topology and host configurations. URL checked 2026-10-03.

[^cloud-tpu-7x]: [Google Cloud TPU 7x](https://docs.cloud.google.com/tpu/docs/tpu7x). specification. Generation specifications, topology and host configurations. Includes cross-generation comparison figures. URL checked 2026-10-03.

[^xla-scoped-vmem]: [OpenXLA E1001: scoped VMEM OOM](https://github.com/openxla/xla/blob/main/docs/errors/error_1001.md). compiler-reference. Typical usable scoped allocation limits, not physical SRAM totals. URL checked 2026-10-03.

[^openxla-tiled-layout]: [OpenXLA tiled layout](https://github.com/openxla/xla/blob/main/docs/tiled_layout.md). compiler-reference. Tiling and padding; not generation-specific physical capacities. URL checked 2026-10-03.

[^openxla-sparsecore]: [OpenXLA SparseCore](https://github.com/openxla/xla/blob/main/docs/sparsecore.md). architecture-reference. SparseCore organization, scratch sizing and embedding stack estimates. URL checked 2026-10-03.

<!-- sources:generated:end -->

[^tpu-info-chips]: [tpu_info chip table](https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/blob/main/tpu_info/tpu_info/device.py). code-reference. `TpuChip`: HBM per chip in GiB (`hbm_gib`) and devices per chip, the table `compute inspect` sizes from; checked at commit 35b5377566aa. URL checked 2026-10-27.
