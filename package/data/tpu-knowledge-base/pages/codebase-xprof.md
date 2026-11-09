---
title: "xprof (JAX / TensorBoard TPU profiler)"
type: codebase
tags: [performance, framework]
created: 2026-09-04
updated: 2026-11-05
sources: 1
human_reviewed: false
aliases: ["JAX profiler", "TensorBoard profiler", "Trace Viewer", "Graph Viewer"]
---

xprof is the TensorBoard profiler plugin that renders traces captured with
`jax.profiler.trace`: a per-core timeline of XLA ops, the HLO graph, and memory over time.
Repo: <https://github.com/openxla/xprof> (not yet ingested); the book's walkthrough is the
only source so far (scaling-book, Part 9 (`index.py get scaling-book-profiling`)).

## Capture and open

```python
import jax
with jax.profiler.trace("/tmp/tensorboard"):
    y = x @ x
    y.block_until_ready()
# pip install -U xprof protobuf ; tensorboard --logdir=/tmp/tensorboard
```

(scaling-book, Part 9 (`index.py get scaling-book-profiling`).) Traces can also be shared as Perfetto
links (Trace Viewer only). For PyTorch on TPU, capture goes through `torch_tpu`'s profiler, not
`jax.profiler` (`index.py get torch-tpu-perf-profiling`).

## Tabs

| Tab | Shows | Use |
|---|---|---|
| Trace Viewer | timeline per TPU core: top row = XLA ops (HLO names), below = `jax.named_scope`/`named_call`/Python stack | find repeated layer blocks, click an op for source + Graph Viewer link; W/A/S/D navigation |
| Graph Viewer | the HLO graph with fusions expanded | decode compound ops (e.g. `all-reduce-scatter`), hover for source line |
| Memory Profile / Viewer | HBM in use over time | OOM debugging, headroom (7.5 GB params, 8.5 GB free in the book's example) |

(scaling-book, Part 9 (`index.py get scaling-book-profiling`).) Typically only TPU:0 is inspected
since all cores run the same program (scaling-book, Part 9 (`index.py get scaling-book-profiling`)).

## See also

- Profile a TPU program (`index.py get playbook-profile-tpu-program`) — the procedure
- HLO (`index.py get concept-hlo`) — reading what the ops say; jax (`index.py get codebase-jax`)

## Sources

- How To Scale Your Model, Part 9: Profiling TPU Programs (scaling-book) (`index.py get scaling-book-profiling`)
