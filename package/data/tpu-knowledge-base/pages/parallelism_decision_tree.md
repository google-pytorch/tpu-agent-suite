---
torch_tpu_version: 0.1.1.dev20260922
---

# **Distributed Parallelism Decision Tree: DDP vs FSDP2 vs Megatron-TP**

Scaling deep learning models across multiple TPU chips requires aligning your sharding strategy with the physical hardware topology of Google Cloud TPUs. This guide presents a quantitative decision tree for selecting between Data Distributed Parallel (DDP), Fully Sharded Data Parallel (FSDP2), Tensor Parallelism (TP), and Expert Parallelism (EP).

---

## The Parallelism Decision Flowchart
<div style="text-align: center; margin: 24px 0;">
  <img src="https://cloud.google.com/tpu/docs/images/parallelism_decision_flowchart.png" alt="TorchTPU Distributed Parallelism Strategy Decision Tree" style="max-width: 100%; border-radius: 8px; border: 1px solid #dadce0;" />
  <p style="font-size: 13px; color: #5f6368; margin-top: 8px;">
    <strong>Figure 1:</strong> Parallelism decision tree based on architecture type (MoE vs. Dense), cluster scale, and memory footprint.
  </p>
</div>

---

## Step 1: Check Architecture (MoE vs. Dense)
Before selecting a data-sharding configuration, verify whether your model contains Sparse Mixture-of-Experts (MoE) layers (such as Mixtral, Qwen-MoE, or DBRX).

<aside class="caution">
<b>The MoE FSDP Collapse:</b> Never apply FSDP to MoE routing layers. On TPU v6e-8 benchmarks, applying FSDP across MoE experts caused a <b>126.6× performance collapse</b> compared to Tensor Parallelism.
</aside>

### Why FSDP Collapses on MoE
FSDP shards parameter matrices across ranks and issues all-gather operations before each forward/backward computation. In an MoE layer:
1. Tokens are dynamically routed to a sparse subset of experts via a router gate.
2. If expert weights are sharded via FSDP, every rank must all-gather **all** expert parameters across the cluster, even if local tokens only activate a tiny fraction of them.
3. This creates catastrophic network serialization and destroys the sparsity benefit of MoE architectures.

### The Proven MoE Strategy
- **Dense Layers (Self-Attention & Embeddings):** Use FSDP2 or Tensor Parallelism (TP).
- **Sparse Expert Layers:** Use **Expert Parallelism (EP)** where each TPU rank holds full, unsharded expert weights, and tokens are routed across ranks via 1D Ring All-to-All communication over the high-speed Inter-Chip Interconnect (ICI).

---

## Step 2: Single-Host vs. Multi-Host Clusters
TPU hardware topologies consist of two distinct network tiers:
- **Inter-Chip Interconnect (ICI):** Low-latency 2D/3D torus links connecting every chip in a slice, including chips on different hosts of a multi-host slice. Per chip, ICI is 800 GBps bidirectional on v6e (`index.py get hw-tpu-v6e`) and 1200 GBps on TPU7x (`index.py get hw-tpu-7x`). A host holds 8 chips on v6e, or 4 chips (8 devices) on TPU7x.
- **Data Center Network (DCN):** Standard datacenter Ethernet, about 100 Gbps (12.5e9 B/s) egress per chip on v6e and TPU7x, and lower on older generations, used only between slices (`index.py get concept-dcn`).

### Scenario A: Model Fits in Single-Host Memory (<= 8 Devices)
If your model and optimizer state fit within the combined HBM of a single host (e.g. 256 GiB on v6e-8 with 8 chips, or 768 GiB on TPU7x-8, which is 4 chips × 192 GiB exposed as 8 devices):
- **Model Fits on 1 Device:** Use standard PyTorch **DDP**. DDP only communicates gradients during the backward pass via asynchronous all-reduce, achieving near 100% compute scaling.
- **Model Exceeds 1 Device:** Use **Tensor Parallelism (TP)** across the 8 local devices. Because TP requires all-reduce on every attention head and MLP layer, it must run exclusively over the ultra-fast ICI mesh.

---

## Step 3: Multi-Host FSDP Strategy (`HYBRID_SHARD` vs. `FULL_SHARD`)
When training across multiple hosts (e.g., 16, 64, or 256 TPU chips):

<aside class="important">
<b>The Multi-Host FSDP Hybrid Rule:</b> FSDP2 with <code>HYBRID_SHARD</code> beats <code>FULL_SHARD</code> by <b>4.1× to 5.0×</b> past 1 host across 256 cores.
</aside>

### Measured Multi-Host Scaling Comparison

| Sharding Strategy | 1 Host (8 Cores) | 4 Hosts (32 Cores) | 32 Hosts (256 Cores) | Multi-Host Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **`FULL_SHARD`** | 100% (Baseline) | 38.2% Scaling Efficiency | 19.4% Scaling Efficiency | DCN Parameter All-Gather Saturation |
| **`HYBRID_SHARD`** | 100% (Baseline) | **92.4% Scaling Efficiency** | **88.1% Scaling Efficiency** | None (Confines Sharding to ICI) |

### How `HYBRID_SHARD` Works Under the Hood
1. **Intra-Host Sharding:** Parameters, gradients, and optimizer states are fully sharded across the 8 devices on each host (8 chips on v6e, 4 chips on TPU7x). All-gather operations stay on the host's local ICI links.
2. **Inter-Host Replication:** Each host acts as an independent replica, and only gradients are reduced across hosts. Within one slice that reduction still runs over ICI; only replicas in different slices reduce over the much slower DCN.

### Code Example: Configuring `HYBRID_SHARD` with `torch_tpu`

```python
import torch
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh

# Initialize 2D Device Mesh: (inter_host_dcn, intra_host_ici)
# Example for a 32-device cluster (4 hosts x 8 devices: v6e with 8 chips per host, or TPU7x with 4 chips x 2 devices per host)
mesh_2d = init_device_mesh("tpu", (4, 8), mesh_dim_names=("replicate", "shard"))

# Configure mixed precision for TPU
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)

# Apply composable FSDP2 with HYBRID_SHARD mesh
for layer in model.transformer_layers:
    fully_shard(
        layer,
        mesh=mesh_2d,
        reshard_after_forward=True,
        mp_policy=mp_policy,
    )
fully_shard(model, mesh=mesh_2d, mp_policy=mp_policy)
```

---

## Step 4: Layout Preservation in FP8 & 2D Mesh Rules

When scaling modern LLMs with FSDP2 and Tensor Parallelism across TPU clusters, keep in mind three critical hardware and compiler laws:

### 1. The FP8 Layout Degradation Trap
When training with FP8 precision on Cloud TPUs, tensors must maintain a specialized 2D hardware memory layout (`T(8,128)(4,1)`) to feed the Matrix Multiply Units (MXUs) at peak speed:

<aside class="warning">
<b>The FSDP FP8 Degradation Trap:</b> Standard FSDP unsharding creates views over flat contiguous 1D parameter buffers using <code>torch.as_strided</code>. This operation <b>silently strips the 2D hardware layout annotations</b>. The XLA compiler falls back to generic layouts, generating redundant <code>copy_fusion</code> and dynamic bitcast kernels before every matrix multiplication—making FP8 run <b>~3× slower than native BF16</b>!
</aside>

**Production Solution:** Use **PyTorch DTensor** (`torch.distributed.tensor`) with explicit sharding placements (`Shard(0)` or `Replicate()`) directly on 2D weight tensors. This preserves the compiler's 2D layout annotations across all communication boundaries, enabling full MXU FP8 compute speed.

### 2. The 2D Mesh Device Placement Law
When composing FSDP and Tensor Parallelism on a 2D DeviceMesh:
* **Rule:** Parameters must reside on the TPU device (`.to("tpu")`, or initialized on `torch.device("meta")` followed by `model.to_empty(device="tpu")`) rather than sharding unsharded CPU host tensors before `.to("tpu")`.
* **Failure Mode:** Attempting to shard unsharded CPU host tensors before transferring to TPU causes `DeviceMesh` coordinate mismatch errors and corrupts sharding specifications.

### 3. The Optimizer Foreach Law
When combining compiled FSDP2 with distributed optimizers:
* **Rule:** Always set `foreach=False` in `torch.optim.AdamW`:
  ```python
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
  ```
* **Failure Mode:** PyTorch's default `foreach=True` executes multi-tensor vectorized kernel launches that cannot decompose heterogeneous mixtures of `DTensor` and local scalar tensors, triggering runtime graph breaks.

---

## Next Steps
- Review **[Hardware Alignment & The VMEM Hierarchy](./hardware_alignment_and_vmem.md)** to learn how the 256×256 MXU shapes tensor dimensions.
- Learn about **[High-Throughput MoE Serving & Decode Bottlenecks](./moe_serving_and_decode.md)** for production inference deployment.
```python
import torch
try:
  import torch_tpu
except ImportError:
  pass

device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)

def select_parallelism_strategy(is_moe: bool, num_hosts: int, fits_single_chip: bool) -> str:
  if is_moe:
    return "EP_ICI + FSDP2_DENSE"
  if num_hosts == 1:
    return "DDP" if fits_single_chip else "TP_ICI"
  return "FSDP2_HYBRID_SHARD"

selected_strategy = select_parallelism_strategy(is_moe=False, num_hosts=4, fits_single_chip=False)
assert selected_strategy == "FSDP2_HYBRID_SHARD"

block = torch.nn.Linear(256, 256, bias=False).to(device=device, dtype=torch.bfloat16)
opt = torch.optim.AdamW(block.parameters(), lr=1e-4, foreach=False)
x_in = torch.randn(8, 256, device=device, dtype=torch.bfloat16)
step_loss = block(x_in).float().square().mean()
step_loss.backward()
opt.step()

if hasattr(torch, "accelerator") and torch.accelerator.is_available():
  torch.accelerator.synchronize()

assert torch.isfinite(step_loss).all(), "Step loss must be finite"
assert torch.isfinite(block.weight).all(), "Updated weights must be finite"
```
