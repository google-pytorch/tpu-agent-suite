---
title: TPU v7x (Ironwood) Hardware Overview
type: hardware
tags:
- hardware
created: '2026-10-03'
human_reviewed: false
source_ids:
- cloud-tpu-7x
- xla-scoped-vmem
- openxla-tiled-layout
- openxla-sparsecore
- tpu-info-chips
generation_recipe: overview.recipe.json
---

# TPU v7x (Ironwood) Hardware Overview

## Chip Compute and Architecture

TPU7x (code-named Ironwood) is Google Cloud's seventh-generation TPU, targeted at large-scale artificial intelligence training and inference [^cloud-tpu-7x]. The physical package employs a dual-chiplet architecture rather than a single monolithic core, housing two distinct chiplets interconnected by a high-speed die-to-die (D2D) interface [^cloud-tpu-7x]. Vendor documentation states this D2D link delivers six times the bandwidth of a one-dimensional inter-chip interconnect (ICI) link [^cloud-tpu-7x].

Each individual chiplet constitutes a self-contained execution unit containing one TensorCore, two SparseCores, and its own dedicated memory space [^cloud-tpu-7x]. Across the full physical chip, compute resources total two TensorCores and four SparseCores [^cloud-tpu-7x]. TensorCores execute dense linear algebra via Matrix Multiply Units (MXUs), while SparseCores provide specialized tiled execution for memory-bound, irregular data access patterns, such as embedding lookups in large embedding models [^cloud-tpu-7x] [^openxla-sparsecore].

Under vendor-claimed peak ratings, a single TPU7x chip provides 2307 TFLOPs of BF16 compute and 4614 TFLOPs of FP8 compute [^cloud-tpu-7x]. The generic OpenXLA layout reference describes vector processing aligned with 32-bit vector registers structured around `8x128` tiles, applying packed sub-byte or multi-level tiling layouts for smaller datatypes such as BF16 `(8,128)(2,1)` and 8-bit `(8,128)(4,1)` [^openxla-tiled-layout].

| Attribute | Per Chiplet | Per Full Chip |
| :--- | :--- | :--- |
| TensorCores | 1 [^cloud-tpu-7x] | 2 [^cloud-tpu-7x] |
| SparseCores | 2 [^cloud-tpu-7x] | 4 [^cloud-tpu-7x] |
| Peak BF16 Compute (Vendor Claim) | Unspecified | 2307 TFLOPs [^cloud-tpu-7x] |
| Peak FP8 Compute (Vendor Claim) | Unspecified | 4614 TFLOPs [^cloud-tpu-7x] |
| Inter-Chiplet D2D Bandwidth | 6× 1D ICI link speed [^cloud-tpu-7x] | Not applicable |

## Memory Hierarchy and Buffer Limits

The TPU7x memory architecture is organized into three principal layers:

- **High-Bandwidth Memory (HBM):** Total on-package HBM capacity is 192 GiB per chip, 96 GiB per chiplet [^cloud-tpu-7x] [^tpu-info-chips]. *(knowledge base note: the tpu_info chip table also gives 192 GiB per chip, which the runtime exposes as two devices of 96 GiB.)* Total aggregate HBM bandwidth is rated at 7380 GBps per chip (described alternatively as approximately 7.37 TB/s) [^cloud-tpu-7x]. HBM holds primary model parameters, optimizer states, heap objects, and SparseCore embedding tables [^cloud-tpu-7x] [^openxla-sparsecore].
- **Vector Memory (VMEM) vs. Physical SRAM:** Each TensorCore couples to high-bandwidth on-chip static random-access memory (SRAM) operating as Vector Memory (VMEM), providing scratchpad staging directly for the MXU [^cloud-tpu-7x] [^xla-scoped-vmem]. While the total physical SRAM capacity is not disclosed, the usable software allocation limit for instruction-scoped VMEM managed by the XLA compiler is strictly bounded to 65,472 KiB or 67,043,328 bytes (63.9375 MiB, derived; the source labels this approximately 63.9 MB) [^xla-scoped-vmem]. Exceeding this usable ceiling during custom kernel compilation triggers an `E1001: Scoped Vmem OOM` error [^xla-scoped-vmem].
- **Host Memory:** Four TPU chips interface to host CPU memory via PCIe, providing a path to offload activations and optimizer states under heavy memory pressure, though at lower transfer speeds than HBM [^cloud-tpu-7x].

| Memory Tier | Allocation / Capacity | Bandwidth / Scope |
| :--- | :--- | :--- |
| HBM (Chip Total) | 192 GiB [^cloud-tpu-7x] [^tpu-info-chips] | 7380 GBps (~7.37 TB/s) [^cloud-tpu-7x] |
| HBM (Per Chiplet) | 96 GiB [^cloud-tpu-7x] [^tpu-info-chips] | Dedicated local address space [^cloud-tpu-7x] |
| Usable Scoped VMEM Limit | 65,472 KiB / 67,043,328 bytes (63.9375 MiB, derived) [^xla-scoped-vmem] | Compiler-managed per-instruction limit [^xla-scoped-vmem] |
| Physical On-Chip VMEM SRAM | Unspecified [^cloud-tpu-7x] [^xla-scoped-vmem] | Higher bandwidth than HBM to MXU [^cloud-tpu-7x] |

## Interconnect and Pod Scale

TPU7x chips form a 3D mesh and 3D torus network topology utilizing direct optical or electrical Inter-Chip Interconnect (ICI) channels [^cloud-tpu-7x].

- **ICI Bandwidth:** Each chip delivers 1200 GBps of bidirectional ICI bandwidth, providing 200 GBps bidirectional bandwidth along each physical axis (X, Y, Z) within a pod [^cloud-tpu-7x].
- **Pod Scale:** A full TPU7x Pod encompasses 9,216 chips [^cloud-tpu-7x].
- **Datacenter Network (DCN):** Each chip features an independent DCN network bandwidth of 100 Gbps for external communication across hosts and clusters [^cloud-tpu-7x].

## Slice and Host Configurations

All TPU7x deployments operate using full-host, 4-chip virtual machine nodes running the `tpu7x-standard-4t` machine type [^cloud-tpu-7x]. Topologies larger than 64 chips are built from modular 4x4x4 "cubes" containing 64 chips each [^cloud-tpu-7x].

### Host Machine Specifications
- **Machine Type:** `tpu7x-standard-4t` [^cloud-tpu-7x]
- **Chips per VM / Host:** 4 chips [^cloud-tpu-7x]
- **Host vCPUs:** 224 vCPUs [^cloud-tpu-7x]
- **Host RAM:** 960 GB [^cloud-tpu-7x]
- **NUMA Nodes:** 2 NUMA nodes per VM [^cloud-tpu-7x]
- **Attached Storage:** Hyperdisk Balanced (default VM boot disk) and Hyperdisk ML [^cloud-tpu-7x]

### Supported 3D Topology Slices

| Topology (Xx Yx Z) | TPU Chips | Hosts | VMs | Cubes | Deployment Scope |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2x2x1 | 4 | 1 | 1 | 1/16 [derived] | Single-host [^cloud-tpu-7x] |
| 2x2x2 | 8 | 2 | 2 | 1/8 [derived] | Multi-host [^cloud-tpu-7x] |
| 2x2x4 | 16 | 4 | 4 | 1/4 [derived] | Multi-host [^cloud-tpu-7x] |
| 2x4x4 | 32 | 8 | 8 | 1/2 [derived] | Multi-host [^cloud-tpu-7x] |
| 4x4x4 | 64 | 16 | 16 | 1 | Multi-host [^cloud-tpu-7x] |
| 4x4x8 | 128 | 32 | 32 | 2 | Multi-host [^cloud-tpu-7x] |
| 4x8x8 | 256 | 64 | 64 | 4 | Multi-host [^cloud-tpu-7x] |
| 8x8x8 | 512 | 128 | 128 | 8 | Multi-host [^cloud-tpu-7x] |
| 8x8x16 | 1024 | 256 | 256 | 16 | Multi-host [^cloud-tpu-7x] |
| 8x16x16 | 2048 | 512 | 512 | 32 | Multi-host [^cloud-tpu-7x] |

*(Note: Cube counts for topologies below 64 chips represent fractional cube multiples [derived]).*

## Software and Operational Constraints

- **Framework Compatibility:** Workloads must use JAX or PyTorch. TensorFlow is explicitly unsupported on TPU7x [^cloud-tpu-7x].
- **Device Exposure:** Software frameworks treat the dual-chiplet package as two distinct devices per physical chip [^cloud-tpu-7x]. In JAX, an extra fourth dimension is added to the topology matrix to differentiate between chiplets, requiring collective communication routines over the internal D2D link to bridge chiplet memory spaces [^cloud-tpu-7x].
- **Infrastructure Environments:** TPU7x is accessible through Google Kubernetes Engine (GKE) and Compute Engine [^cloud-tpu-7x]. It supports GKE "All Capacity mode" reservations, granting access to all reserved capacity without hold-backs, alongside visibility into physical topology, utilization, and health metrics [^cloud-tpu-7x].
- **Custom Kernel Tuning:** Pallas kernels running on TensorCores must fit instruction-scoped allocations within the 63.9MB VMEM limit; block and tile dimensions must be tailored accordingly using compiler flags (`--xla_tpu_scoped_vmem_limit_kib`) or kernel parameters (`vmem_limit_bytes`) [^cloud-tpu-7x] [^xla-scoped-vmem].

## Gaps and Unspecified Specifications

Certain technical specifications for TPU7x are not detailed in the supplied evidence:

- **Physical VMEM and SPMEM Size:** The gross physical SRAM capacity for TensorCore VMEM and the exact per-tile scratchpad memory (SPMEM) capacity for SparseCores are undisclosed [^cloud-tpu-7x] [^xla-scoped-vmem] [^openxla-sparsecore].
- **Clock Frequencies and Core Microarchitecture:** Operating clock frequencies, the exact physical dimensions and count of MXUs per TensorCore, SparseCore tile counts, and SIMD widths for TPU7x remain unspecified [^cloud-tpu-7x] [^openxla-sparsecore].
- **Thermal and Power Envelopes:** Thermal Design Power (TDP), electrical requirements, and physical server chassis form factors are undocumented [^cloud-tpu-7x].
- **Availability Timeline:** General availability dates, regional cloud availability, and pricing tiers are omitted [^cloud-tpu-7x].

## Detailed memory and profiling

See [the generation-specific memory and buffer reference](memory-buffers.md) for tile sizing, padding and XProf inspection.

<!-- sources:generated:start -->

## References

[^cloud-tpu-7x]: [Google Cloud TPU 7x](https://docs.cloud.google.com/tpu/docs/tpu7x). specification. Generation specifications, topology and host configurations. Includes cross-generation comparison figures. URL checked 2026-10-03.

[^xla-scoped-vmem]: [OpenXLA E1001: scoped VMEM OOM](https://github.com/openxla/xla/blob/main/docs/errors/error_1001.md). compiler-reference. Typical usable scoped allocation limits, not physical SRAM totals. URL checked 2026-10-03.

[^openxla-tiled-layout]: [OpenXLA tiled layout](https://github.com/openxla/xla/blob/main/docs/tiled_layout.md). compiler-reference. Tiling and padding; not generation-specific physical capacities. URL checked 2026-10-03.

[^openxla-sparsecore]: [OpenXLA SparseCore](https://github.com/openxla/xla/blob/main/docs/sparsecore.md). architecture-reference. SparseCore organization, scratch sizing and embedding stack estimates. URL checked 2026-10-03.

<!-- sources:generated:end -->

[^tpu-info-chips]: [tpu_info chip table](https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/blob/main/tpu_info/tpu_info/device.py). code-reference. `TpuChip`: HBM per chip in GiB (`hbm_gib`) and devices per chip, the table `compute inspect` sizes from; checked at commit 35b5377566aa. URL checked 2026-10-27.
