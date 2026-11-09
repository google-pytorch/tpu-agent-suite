---
title: "TorchTPU Runtime & OpenXLA Backend"
type: codebase
tags: [codebase, torch-tpu, openxla, pjrt, pytorch]
created: 2026-08-08
updated: 2026-10-13
sources: 4
human_reviewed: false
aliases: [torch_tpu_runtime, torch_tpu]
---

# TorchTPU Runtime & OpenXLA Backend

`torch_tpu` is Google's PyTorch backend for Cloud TPU (v4, v5e, v5p, v6e and TPU7x): it
registers a `tpu` device, lowers ATen ops to StableHLO and compiles them through OpenXLA and
PjRt. This hub page states the ingested pin and routes into the grounded subwiki; mechanism and
symbol-level detail live there, not here.

> [!IMPORTANT]
> **Deprecation of `torch_xla`:**
> `torch_xla` (and `pytorch_xla`) is **deprecated** and MUST NOT be used. All TPU hardware operations, device acquisition, and runtime execution must use native PyTorch TPU device bindings (`import torch`, `device = torch.device('tpu')`). Never attempt to install or import `torch_xla`.

---
## Pin

| | |
|---|---|
| Upstream | https://github.com/google-pytorch/torch_tpu (private; the source is not shipped with this knowledge base) |
| Ingested commit | `6d2a45754bdb52743883f75fbf753408345f6d1b` (head of `release/0.1.2`, 2026-10-12; previously `4c0c7c57`) |
| Ingest config | `config/torch_tpu.md` (wikify-repo; re-ingest at a new pin with `wikify prepare torch_tpu --ref <commit>`) |
| Provenance | [torch_tpu repo snapshot (2026-10, release/0.1.2)](../sources/2026-10-torch-tpu-repo.md); earlier: [2026-09](../sources/2026-09-torch-tpu-repo.md) |

## Start here (subwiki)
- [Overview](torch_tpu/overview.md): the system on one page, two architecture diagrams, a
  task table and the onboarding and optimization surfaces (env vars and APIs).
- [Concept index](torch_tpu/index.md): 28 grounded mechanism pages (dispatch, deferred
  execution, compilation cache and keys, MLIR optimization passes, precision, torch.compile,
  dynamic shapes, collectives, Pallas custom kernels, profiler and more), grouped by source area,
  plus one area page for `csrc/`. What changed since the previous pin:
  [changes/6d2a45754b](torch_tpu/changes/6d2a45754b.md).
- `torch_tpu/doc-concepts/`: 45 pages extracted from the repo's README files and its user guide
  under `docs/markdown/` (install, execution modes, lowering stack, compilation cache,
  torch.compile, precision and numerical parity, Pallas custom ops, distributed launch and GKE,
  profiling, error messages and debugging, environment variables, API stability policy); each
  is linked from the overview.
- `torch_tpu/catalog/`: the symbol index — `symbols.tsv` (one row per symbol: anchor,
  path, line, kind, rank, hash, caller count, citing pages, signature, doc line; 19,064 symbols
  in 15,910 rows, overloads sharing a row), `edges.tsv` (75,029 caller edges) and [`index.md`](torch_tpu/catalog/index.md)
  (module map, 716 modules; C++ under `csrc/`, Python under `src/torch_tpu/`). Look a symbol up with
  `grep -P '^<path>#<Symbol>\t' catalog/symbols.tsv`. Citations link the symbol's source line
  on GitHub at the pin (private: needs access) and carry its index key as the link title, so
  the row is one grep away without access.

## Core APIs and device bindings

```python
import torch
import torch_tpu  # registers the "tpu" device, torch.tpu, the tpu_dist backend and the profiler

device = torch.device("tpu")
```

- Device selection, streams, RNG state and autocast:
  [device-backend-registration](torch_tpu/concepts/device-backend-registration.md).
- Precision (`torch.set_float32_matmul_precision`, `torch_tpu.precision`, excess precision):
  [dtype-and-precision](torch_tpu/concepts/dtype-and-precision.md).
- Graph mode: after import, `torch.compile(model)` defaults to the `tpu` backend:
  [torch-compile-backend](torch_tpu/concepts/torch-compile-backend.md).
- Distributed: `init_process_group(backend="tpu_dist")` plus the launch variables:
  [process-group-and-collectives](torch_tpu/concepts/process-group-and-collectives.md),
  [distributed-launchers-and-spmd](torch_tpu/concepts/distributed-launchers-and-spmd.md).
- Custom kernels: `torch_tpu._internal.pallas.jax_op`:
  [pallas-custom-kernels](torch_tpu/concepts/pallas-custom-kernels.md).

## See also
- [TPU v6e (Trillium) Hardware Specs](../hardware/tpu-v6e.md)
- [Static Shapes & Tracing](../playbooks/static-shapes-and-tracing.md)
- [Pallas Kernel Authoring](pallas.md)
- [Compilation cache](torch_tpu/concepts/compilation-cache.md) and
  [cache keys](torch_tpu/concepts/compilation-cache-key.md): the mechanism behind the
  cache-hit receipts in [30-Model Onboarding Learnings](../analyses/30-model-onboarding-learnings.md).

## Sources
- [torch_tpu repo snapshot (2026-10, release/0.1.2)](../sources/2026-10-torch-tpu-repo.md): the
  ingested source at the current pin; grounds everything in the subwiki.
- [torch_tpu repo snapshot (2026-09)](../sources/2026-09-torch-tpu-repo.md): the previous pin.
- [30-Model Onboarding Learnings](../analyses/30-model-onboarding-learnings.md) and
  [Standard Intelligence Case Study](../analyses/standard-intelligence-case-study.md): the
  2026-08-28 playbook ingest that created this page.
