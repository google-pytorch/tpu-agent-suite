---
title: Static Shapes & OpenXLA Graph Tracing
type: playbook
tags:
- optimization
- static-shapes
- torch-compile
- openxla
- synchronization
- cache-hits
created: 2026-08-08
updated: 2026-11-05
sources: 1
human_reviewed: false
aliases:
- static_shapes
- openxla_caching
target_imports:
- dynamic_shapes
---
# Static Shapes & OpenXLA Graph Tracing

This playbook describes how to configure static input shapes, padding boundaries, and step synchronization barriers to achieve >97% steady-state OpenXLA compilation cache hit rates on Google Cloud TPUs.

---

## Applies when

- Benchmarks or inference endpoints suffer from high per-step latency due to continuous OpenXLA recompilations.
- Profiler traces show excessive `pjrt_api` compilation bubbles.

---

## Steps & Transformation Recipes

### Step 1: Enforce Static Compilation (`dynamic=False`)
```python
import torch

# Compile with TPU backend enforcing static shapes
model = torch.compile(model, backend="tpu", dynamic=False)
```

### Step 2: Pad Sequences to Power-of-2 Buckets
Pad input token sequences to fixed bucket boundaries ($128, 256, 512, 1024, 2048$):
```python
import torch.nn.functional as F

def pad_to_bucket(input_ids: torch.Tensor, bucket_size: int = 512, pad_token_id: int = 0) -> torch.Tensor:
    seq_len = input_ids.shape[-1]
    if seq_len < bucket_size:
        return F.pad(input_ids, (0, bucket_size - seq_len), value=pad_token_id)
    return input_ids[:, :bucket_size]
```

### Step 3: Insert Step Barriers & Synchronization
```python
import torch

# Synchronize step execution
torch.accelerator.synchronize()
```

---

## See also

- Roofline Analysis (`index.py get concept-roofline-analysis`)
- Padding & Alignment (`index.py get padding`)
- Compute-Communication Overlap (`index.py get concept-compute-communication-overlap`)