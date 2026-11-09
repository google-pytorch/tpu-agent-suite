# Knowledge Base Index

This is the master catalog. Read this first before answering any query — it tells you
what pages exist and what they cover. Rules and templates: [SCHEMA.md](SCHEMA.md).

---

## Hardware
*One spec page per generation; hardware blocks/mechanisms live under Concepts. Derived
compute-bound thresholds per generation: [Critical batch size](wiki/concepts/critical-batch-size.md).*

- [TPU v4](wiki/hardware/tpu-v4.md) — 275 TF bf16/int8, 32 GiB @ 1200 GBps, 4096-chip 3D-torus pods, megacore; twisted torus; legacy tier.
- [TPU v5e](wiki/hardware/tpu-v5e.md) — 197 TF bf16 / 393 TOPs int8, 16 GB @ 800 GiBps, single-core, 256-chip 2D torus; training+serving.
- [TPU v5p](wiki/hardware/tpu-v5p.md) — 459 TF bf16 (FP8 same), 95 GiB @ 2765 GBps, 8960-chip pods, megacore, 4 SparseCores, ICI resiliency.
- [TPU v6e (Trillium)](wiki/hardware/tpu-v6e.md) — 918 TF bf16 / 1836 TOPs int8, 32 GB @ 1638 GBps, 1 TensorCore × 2 MXUs (256×256), 256-chip 2D torus.
- [TPU7x (Ironwood)](wiki/hardware/tpu-7x.md) — 2307 TF bf16 / 4614 FP8, 192 GiB @ 7.37 TB/s, 9216-chip 3D torus, dual-chiplet (2 devices/chip), tunable VMEM.
- [TPU 8t](wiki/hardware/tpu-8t.md) — *announced 2026-04, not GA* — training chip; 9,600-chip / 2 PB HBM superpod claims; no datasheet yet.
- [TPU 8i](wiki/hardware/tpu-8i.md) — *announced 2026-04, not GA* — inference chip; 288 GB HBM + 384 MB SRAM, Boardfly topology claims; no datasheet yet.

## Codebases
*Planned: vLLM/tpu-inference, SGLang, MaxText.*

- [TorchTPU](wiki/codebases/torch-tpu.md) — hub for the PyTorch-on-TPU runtime: pin, routing, API pointers; the grounded subwiki starts at [torch_tpu overview](wiki/codebases/torch_tpu/overview.md) (commit 6d2a4575, release/0.1.2: 28 concept pages, 45 doc-derived pages from the READMEs and user guide, symbol index of 19,064 symbols).
- [TorchTitan](wiki/codebases/torchtitan.md) - TorchTitan, the PyTorch-native LLM training platform, as used on Cloud TPU (Google's `google-pytorch/torchtitan`, the only one with the TorchTPU backend; not the official `pytorch/torchtitan`, which has none and has diverged): Python config trees select a model spec and build one device mesh, FSDP/TP/PP/CP/EP, activation checkpointing and per-block torch.compile are applied to a meta-device model, and the experiments/tpu backend runs the step on torch_tpu either as compiled blocks (TPUTrainer) or as one whole-step XLA graph (TpuGraphTrainer). The grounded subwiki starts at [torchtitan overview](wiki/codebases/torchtitan/overview.md) (commit 033f1313: 24 concept pages, 56 doc-derived pages, symbol index of 13,713 symbols).
- [Tokamax](wiki/codebases/tokamax.md) - hub for the OpenXLA JAX kernel library (Apache-2.0): pin, routing; the grounded subwiki starts at [tokamax overview](wiki/codebases/tokamax/overview.md) (commit 3a0a3a34: 18 verified concept pages incl. a kernel catalog of every op x TPU/GPU/XLA backend, 14 doc-derived pages, symbol index of 5,738 symbols).
- [Pallas](wiki/codebases/pallas.md) — *stub* — JAX's TPU kernel-authoring extension; block sizes constrained by VMEM; the book's escape hatch below HLO.
- [JAX (parallelism APIs)](wiki/codebases/jax.md) — Auto / Explicit / Manual (`shard_map`) sharding modes, mesh + PartitionSpec vocabulary, collectives, pitfalls; docs not yet ingested.
- [xprof](wiki/codebases/xprof.md) — JAX/TensorBoard TPU profiler: capture with `jax.profiler.trace`, Trace Viewer, Graph Viewer, Memory Profile; repo not yet ingested.

## Concepts
### Architecture

- [TensorCore](wiki/concepts/tensorcore.md) — hub: the TPU compute core (MXU + VPU + VMEM) and its data flow.
- [MXU](wiki/concepts/mxu.md) — 128×128 (256×256 v6e) systolic-array matmul unit; 8-cycle `bf16[8,128]@bf16[128,128]`; vs GPU Tensor Cores.
- [VPU](wiki/concepts/vpu.md) — 8×128 SIMD vector unit (~30× below MXU FLOPs); VREGs, scalar core, XLU, lane/sublane reduction asymmetry; vs CUDA cores.
- [VMEM](wiki/concepts/vmem.md) — programmer-controlled scratchpad, ~22× HBM BW; prefetching; FLOPs-bound at batch ~11–20; `S(1)` in HLO; vs SMEM/L2.
- [HBM](wiki/concepts/hbm.md) — main chip memory (1–2 TB/s); decode-latency floor and step-time formula; intensity ~240 threshold.
- [ICI](wiki/concepts/ici.md) — nearest-neighbor chip interconnect; 2D/3D tori, wraparound rules, bidi bandwidth, collective costs, ICI arithmetic intensity, vs GPU fabrics.
- [DCN](wiki/concepts/dcn.md) — host-to-host network between slices; multi-slice training; cross-slice DP threshold (~73k tokens/slice on v5p).
- [PCIe](wiki/concepts/pcie.md) — host↔TPU link (~100× slower than HBM); offload limits.
- [Systolic array](wiki/concepts/systolic-array.md) — the MXU's compute primitive; diagonal operand flow, pipeline bubble.
- [Megacore](wiki/concepts/megacore.md) — two cores sharing memory as one accelerator (v4, v5p; contradiction on v6 human-adjudicated); TPU7x dual-chiplet successor.
- [SparseCore](wiki/concepts/sparsecore.md) — *stub* — sparse/embedding accelerator; per-chip counts v5p/v6e/7x.
- [GPU vs TPU](wiki/concepts/gpu-vs-tpu.md) — SM/TensorCore, SMEM/VMEM, NVLink/ICI mapping; H100/B200 rooflines; fat-tree collectives; GPU LLM scaling rules; what changes on onboarding.

### Parallelism

- [Parallelism strategies (hub)](wiki/concepts/parallelism-strategies.md) — DP / FSDP / TP / FSDP+TP / PP / EP as sharding formulas with comms per layer and compute-bound conditions; how to choose.
- [Sharding notation](wiki/concepts/sharding-notation.md) — `A[I_X, J_Y]` named-axis notation, mesh, local shape, `{U_X}`; the four sharded-matmul cases and their collectives; backward pass by the same rules.
- [Collective costs](wiki/concepts/collective-costs.md) — AllGather / ReduceScatter / AllReduce / AllToAll semantics and ring cost formulas; latency-bound regime (~45 kB); AG/RS duality; GPU comparison.
- [Data parallelism](wiki/concepts/data-parallelism.md) — batch-sharded activations, replicated weights; `B/X > C/W_ici`; HBM/10 parameter ceiling; DCN and MoE variants; useless for decode.
- [FSDP (ZeRO)](wiki/concepts/fsdp.md) — weights/optimizer sharded on the batch axis; same roofline as DP, `1/X` memory; prefetched AllGathers; never for generation; ZeRO-1 with pipelining.
- [Tensor parallelism](wiki/concepts/tensor-parallelism.md) — Megatron sharding; `Y < M_Y·F/α` (8–16 way v5p); `X_opt` for FSDP+TP; sequence parallelism; generation limits (FLOPs- vs latency-bound); 2D weight-stationary.
- [Pipeline parallelism](wiki/concepts/pipeline-parallelism.md) — layer-sharded stages; ~free comms; bubbles, zero-bubble schedules; breaks ZeRO-3; GPU-first, rarely needed on pods.
- [Expert parallelism / MoE](wiki/concepts/expert-parallelism.md) — `E/k` scaling of every roofline; expert sharding with two AllToAlls; ragged AllToAll; GPU EP regimes; JAX sketch.
- [Collective matmul](wiki/concepts/collective-matmul.md) — overlap AG/RS/AR with chunked matmuls via `ppermute`; 311 → 244 µs (224 µs baseline) on v5e-8; when to use.

### Performance & Profiling

- [Roofline analysis](wiki/concepts/roofline-analysis.md) — `T_lower = max(T_math, T_comms)`, arithmetic intensity, matmul and network rooflines; TPU bandwidth hierarchy; worked thresholds.
- [Critical batch size](wiki/concepts/critical-batch-size.md) — `B_crit = β·C/W`; the 240 rule; **derived per-generation table** (v5e/v5p/v6e/TPU7x HBM and ICI intensities); same ratio for DP/TP/DCN/MoE.
- [Compute–communication overlap](wiki/concepts/compute-communication-overlap.md) — pipelining principle: total ≈ slowest stage; critical-path vs async collectives; XLA overlap caveats.
- [Padding](wiki/concepts/padding.md) — ≥128 (256 v6e) MXU tiling rule; shard-size consequences; HLO tiling notation.
- [MFU](wiki/concepts/mfu.md) — model FLOPs utilization; `6ND` time estimates (LLaMA 3-70B: 44 days on a v5p pod at 40%); `8ND` with remat; what caps it.
- [Transformer FLOPs accounting](wiki/concepts/transformer-flops-accounting.md) — contraction rule, 6ND/2ND, per-layer params and FLOPs table, `T > 8D` attention crossover, LLaMA 3-70B worked count, attention intensity, Flash Attention.
- [KV cache](wiki/concepts/kv-cache.md) — `2·bytes·H·K·L` per token; why it dominates decode; GQA/local/sharing/quantization/paged attention; sharding heads then batch; prefix caching and eviction.
- [Prefill vs decode](wiki/concepts/prefill-vs-decode.md) — compute-bound prefill vs bandwidth-bound generation; step-time and throughput formulas; per-phase sharding rules; engine designs (interleaved, disaggregated, continuous batching, speculative sampling).
- [Gradient checkpointing](wiki/concepts/gradient-checkpointing.md) — remat policies (block, big-matmuls-only) and their FLOPs cost; checkpoints dominate the training memory budget; pod-scale training is not memory-bound.
- [Grids and BlockSpecs](wiki/concepts/grid_blockspec.md) — Grid mapping, block slicing, and index translation for TPU Pallas kernels.
- [Matrix Multiplication in Pallas](wiki/concepts/matmul.md) — Blocked matmul, MXU systolic array contraction, and tiling on TPUs.
- [TPU Pipelining](wiki/concepts/pipelining.md) — Overlapping DMA transfers (HBM <-> VMEM) with computation via double buffering.

### Precision & Numerics

- [Quantization](wiki/concepts/quantization.md) — native low-precision speedup **varies by generation** (v4/v5p: none; v5e/v6e/7x: 2×); effect on critical batch (int8 weights → 120), TP limits, memory fit, KV caches.

### Compilation

- [HLO](wiki/concepts/hlo.md) — JAX → StableHLO → HLO → LLO stack; anatomy of an HLO op line (shape, layout/tiling `T(8,128)(2,1)`, memory space `S(1)`); retile copies; reading sharding from local shapes and `replica_groups`.

## Playbooks
### Onboarding (CUDA → TorchTPU)

- [TPU Distributed & Environment Hygiene](wiki/playbooks/tpu-distributed-and-env-hygiene.md) — Switches nccl to tpu_dist and prevents destructive uv sync clobbering.
- [Static Shapes & Tracing](wiki/playbooks/static-shapes-and-tracing.md) — Enforces static shapes (dynamic=False) and step synchronization barriers for >97% cache hits.

### Pallas Kernels (tpu-kernel-development)

- [Kernel-Optimization Index](wiki/kernel-optimization-index.md) — Router and mental model for TPU Pallas kernel optimization.
- [Pallas Core Mental Model](wiki/kernel/pallas-core-mental-model.md) — SOTA mental model, SIMD execution, and memory hierarchy for Pallas.
- [Pallas TPU Reference](wiki/kernel/pallas-tpu-reference.md) — API reference for Pallas on TPU architectures.
- [Pallas Validation & XProf](wiki/kernel/pallas-validation-and-xprof.md) — Validating and profiling Pallas kernels with XProf and HLO dumps.
- [Matmul Pallas Architectures](wiki/kernel/matmul-pallas-architectures.md) — Program instance, kernel math, and precision constraints on TPU v6e.
- [Attention Pallas Data Management](wiki/kernel/attention-pallas-data-management.md) — Data management, asynchronous copies, and double-buffering on TPU v6e.
- [Kernel Design-Class Catalog](wiki/kernel/design-class-catalog.md) — Problem-invariant mechanisms across kernel classes.
- [Kernel Attention](wiki/kernel/kernel-attention.md) — Streaming-softmax and KV structure attention kernels.
- [Kernel Cross-Chip Collective](wiki/kernel/kernel-cross-chip-collective.md) — Ring attention and in-kernel collectives over ICI.
- [Kernel Dense Near Roofline](wiki/kernel/kernel-dense-near-roofline.md) — Dense compute near roofline for large pure GEMM/conv.
- [Kernel GEMM Conv Epilogue](wiki/kernel/kernel-gemm-conv-epilogue.md) — GEMM/conv with epilogue chains and fusion scope.
- [Kernel Grouped Ragged Indirection](wiki/kernel/kernel-grouped-ragged-indirection.md) — Grouped, ragged, MoE, and paged KV kernels.
- [Kernel State Carry Scan](wiki/kernel/kernel-state-carry-scan.md) — State-carry and chunked scans for SSM and linear attention.
- [Kernel Streaming Reduction](wiki/kernel/kernel-streaming-reduction.md) — Streaming reductions for norms, cross-entropy, and softmax.
- [Kernel Briefs](wiki/kernel/briefs.md) — Distilled rules for measurement, parity, and debugging.
- [Pallas Kernel](wiki/kernel/pallas-kernel.md) — Authoring custom Pallas kernels and compiling to Mosaic.

### End-to-end optimization

- [Profile a TPU Program Against Its Roofline](wiki/playbooks/profile-tpu-program.md) — Capture a trace, read the dominant HLO ops, predict each from the roofline, classify the gap; the verification step for every other playbook.
- [Size a Training Parallelism Configuration](wiki/playbooks/size-training-parallelism.md) — From model config, token batch, chip count and generation to the FSDP/TP/DP split, per-chip memory and step time.
- [Size a Serving Topology and Batch for Decode](wiki/playbooks/size-serving-topology.md) — KV bytes/token, memory fit by dtype, minimum slice, critical batch, step-time floor, model-parallel limits, prefill:generate ratio.

## Sources
### How To Scale Your Model (scaling-book, DeepMind 2025; snapshot 2026-09-04 @ `ac72f02`)

- [Book hub](wiki/sources/2025-scaling-book.md) — identity, pin, chapter map, gaps, further-reading fetch targets (covers intro + conclusion).
- [Part 1: Rooflines](wiki/sources/2025-scaling-book-roofline.md) — `max(T_math, T_comms)`, arithmetic intensity, the 240 rule, network rooflines.
- [Part 2: How to Think About TPUs](wiki/sources/2025-scaling-book-tpus.md) — TPU chip anatomy, networking, bandwidth hierarchy, worked roofline problems (ingested 2026-08-22; re-verified identical).
- [Part 3: Sharded Matrices](wiki/sources/2025-scaling-book-sharding.md) — sharding notation, four matmul cases, four collectives and their costs, collective matmul.
- [Part 4: Transformer Math](wiki/sources/2025-scaling-book-transformers.md) — 6ND, per-layer FLOPs/params, `T > 8D`, MoE, remat, KV cache, Flash Attention.
- [Part 5: Training Parallelism](wiki/sources/2025-scaling-book-training.md) — DP / FSDP / TP / mixed / PP / DCN thresholds with v5p constants.
- [Part 6: Training LLaMA 3](wiki/sources/2025-scaling-book-applied-training.md) — 70B on a v5p pod: 44 days at 40% MFU, memory budget, 2048-way FSDP + 4-way TP.
- [Part 7: Inference](wiki/sources/2025-scaling-book-inference.md) — prefill vs generation rooflines, step-time formula, KV cache techniques, sharding, engine design.
- [Part 8: Serving LLaMA 3](wiki/sources/2025-scaling-book-applied-inference.md) — 70B on v5e: fit by dtype, latency floors, model-parallel limits, 3:1 disaggregation, Pareto.
- [Part 9: Profiling](wiki/sources/2025-scaling-book-profiling.md) — software stack, xprof tabs, HLO op anatomy, roofline-matched worked profile.
- [Part 10: JAX](wiki/sources/2025-scaling-book-jax-stuff.md) — Auto / Explicit / `shard_map` modes; collective matmul 311 → 244 µs.
- [Part 12: GPUs](wiki/sources/2025-scaling-book-gpus.md) — GPU anatomy and mapping, NVLink/InfiniBand, collectives, GPU LLM scaling rooflines.

### Cloud TPU documentation and announcements

- [Cloud TPU v4 docs](wiki/sources/2026-08-cloud-tpu-v4-docs.md) — v4 spec sheet; twisted torus; NUMA binding.
- [Cloud TPU v5e docs](wiki/sources/2026-08-cloud-tpu-v5e-docs.md) — v5e spec sheet; serving/training configs, VM types.
- [Cloud TPU v5p docs](wiki/sources/2026-08-cloud-tpu-v5p-docs.md) — v5p spec sheet; cubes/wraparounds, ICI resiliency.
- [Cloud TPU v6e docs](wiki/sources/2026-08-cloud-tpu-v6e-docs.md) — v6e spec sheet; single-TensorCore (megacore contradiction).
- [Cloud TPU7x docs](wiki/sources/2026-08-cloud-tpu-7x-docs.md) — TPU7x spec sheet; dual-chiplet model, tunable VMEM.
- [tpu_info chip table](wiki/sources/2026-10-tpu-info-chip-table.md) — HBM per chip in GiB and devices per chip, the table `compute inspect` sizes from.
- [Google blog: 8th-gen TPUs](wiki/sources/2026-google-blog-tpu-8th-gen.md) — TPU 8t/8i announcement; vendor claims only.
- [tokamax repo snapshot (2026-10)](wiki/sources/2026-10-tokamax-repo.md) - provenance record of the tokamax ingest at 3a0a3a34: pin, license, indexing, what was produced, docs-vs-code gaps and possible upstream defects.
- [torch_tpu repo snapshot (2026-09)](wiki/sources/2026-09-torch-tpu-repo.md) — provenance record of the torch_tpu ingest at 4c0c7c57: pin, indexing, what was produced, gaps and caveats.
- [torchtitan repo snapshot (2026-10)](wiki/sources/2026-10-torchtitan-repo.md) - provenance record of the torchtitan re-ingest at 033f1313: the upstream rebase and new models, config remaps and re-seeds, doc scope (45 -> 56 doc-concept pages), verification, hygiene, doc-versus-code mismatches and gaps.
- [torch_tpu repo snapshot (2026-10, release/0.1.2)](wiki/sources/2026-10-torch-tpu-repo.md) — provenance record of the torch_tpu re-ingest at the release/0.1.2 head 6d2a4575: the repo restructure, reconcile, docs scope, verification and hygiene.
- [torchtitan repo snapshot (2026-09)](wiki/sources/2026-09-torchtitan-repo.md) - provenance record of the torchtitan ingest at 63e334d5: pin, indexing, coverage, verification, the experiments/ workaround, gaps and caveats.

## Analyses
- [30-Model Onboarding Learnings](wiki/analyses/30-model-onboarding-learnings.md) — Empirical rules and compilation cache receipts from onboarding 30 GPU architectures to TPU v5e/v6e.
- [Standard Intelligence Case Study](wiki/analyses/standard-intelligence-case-study.md) — Case study: Porting Hopper GPU-specialized LLM to Cloud TPU v7.
