---
title: "VMEM (Vector Memory)"
type: concept
tags: [hardware, memory, performance]
created: 2026-08-22
updated: 2026-10-07
sources: 6
human_reviewed: true
---

VMEM is the on-chip, **programmer-controlled** scratchpad inside the
TensorCore (`index.py get concept-tensorcore`) — like an L1/L2 cache but larger and explicitly managed.
All data must pass HBM → VMEM before the MXU (`index.py get concept-mxu`)/VPU (`index.py get vpu`) can touch it
(source (`index.py get scaling-book-tpus`)).

## Numbers that matter

- Capacity: small — 128 MiB on v5e vs 16 GiB HBM
  (source (`index.py get scaling-book-tpus`)).
- Bandwidth to the MXU: **~22× HBM bandwidth**. Consequence: an MXU op fed from VMEM
  needs arithmetic intensity of only ~10–20 for peak FLOPs, vs ~240 from
  HBM (`index.py get concept-hbm`) — VMEM-resident weights make matmuls FLOPs-bound at batch ~11–20
  (source (`index.py get scaling-book-tpus`)). See
  Roofline analysis (`index.py get concept-roofline-analysis`).

## VMEM prefetching

Loading weights into VMEM ahead of need masks weight-load cost: e.g. load the
feed-forward weights during attention, hiding the transfer when memory-bandwidth
bound. Requires the layer's weights (possibly sharded) to fit in VMEM with room to
spare (source (`index.py get scaling-book-tpus`)). A core lever for
memory-bound inference; see
Compute–communication overlap (`index.py get concept-compute-communication-overlap`).

## Optimization implications

- Algorithms with inherently low arithmetic intensity can still hit peak FLOPs if
  their working set fits in VMEM — the central idea behind kernel tiling strategies.
- VMEM's small size is usually the binding constraint; fitting a single layer is
  already a win.
- The VMEM buffer size is a **tunable parameter**, and
  Pallas (`index.py get codebase-pallas`) kernel block sizes are constrained by the available
  VMEM — buffer sizing is an explicit kernel-tuning knob. Stated in the
  TPU7x docs (`index.py get cloud-tpu-7x`); likely
  generation-general as a Pallas/XLA-stack property *(unverified — agent inference)*.

## The scoped limit

- The compiler caps the VMEM one instruction or kernel may allocate, and a kernel over it fails to
  compile with `CompileTimeScopedVmemOom` (E1001). Raise it for the whole program with
  `--xla_tpu_scoped_vmem_limit_kib`, or for one Pallas kernel with `vmem_limit_bytes` in its TPU
  compiler params (source (`index.py get xla-error-e1001`)).
- How far it can go is per generation: E1001 tables the usable limit, and each chip page carries
  its own (`index.py get hw-tpu-v6e`, `index.py get hw-tpu-7x`, …).
- What raising it does and does not buy — including that a flag win does not carry over to
  `vmem_limit_bytes` — is in BRIEFS (`index.py get briefs`).
- A value above the usable limit is not an error on TPU7x: libtpu 0.0.44 clamped it to the usable
  limit (`index.py get pallas-core-mental-model`). Recipes that pass 131072 on TPU7x, such as
  torchtitan's Ironwood Llama 3 70B ones, run at the cap (`index.py get torchtitan-doc-errata`).
- The default is not one number across sources: 32 MiB measured for a Pallas kernel on TPU7x,
  64 MB on v6e / v7x in torchtitan's onboarding notes. Read the limit from the compile log before
  tuning against it.

## In practice

- In HLO (`index.py get concept-hlo`) an array's memory space shows as `S(1)` for VMEM (`S(0)` = HBM);
  XLA's decision to keep a fusion operand in VMEM is visible on the op line
  (scaling-book, Part 9 (`index.py get scaling-book-profiling`)).
- Flash Attention keeps the `Q` chunk and the running max/sum/output in VMEM and streams
  KV chunks, which is what raises attention's arithmetic intensity
  (scaling-book, Part 4 (`index.py get scaling-book-transformers`)).
- Versus GPU: an H100 has 256 kB SMEM + 256 kB registers per SM (~33 MB each in total)
  plus a 50 MB L2 at ~5.5 TB/s that is not programmer-controlled; TPU VMEM is ~2× larger
  and ~40 TB/s, so weights/activations can be pinned on-chip in ways SMEM cannot
  (scaling-book, Part 12 (`index.py get scaling-book-gpus`)).

## See also

- HBM (`index.py get concept-hbm`) — the level above; MXU (`index.py get concept-mxu`) — the consumer

## Sources

- How to Think About TPUs (scaling-book) (`index.py get scaling-book-tpus`)
- Cloud TPU7x docs (`index.py get cloud-tpu-7x`)
- How To Scale Your Model, Part 4: Transformer Math (scaling-book) (`index.py get scaling-book-transformers`),
  How To Scale Your Model, Part 9: Profiling TPU Programs (scaling-book) (`index.py get scaling-book-profiling`),
  How To Scale Your Model, Part 12: How to Think About GPUs (scaling-book) (`index.py get scaling-book-gpus`)
- OpenXLA E1001: scoped VMEM OOM (`index.py get xla-error-e1001`)
