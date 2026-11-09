---
torch_tpu_version: 0.1.1.dev20260922
---

# **Hardware Alignment (128x8) and VMEM Allocation on Cloud TPU**

Achieving peak Model FLOPs Utilization (MFU) on Google Cloud TPUs requires structuring tensor dimensions and memory allocations to match the physical geometry of Matrix Multiply Units (MXUs), Vector Processing Units (VPUs), and the memory hierarchy.

Every recommendation in this guide is grounded in **empirically validated benchmarks across 5 independent replicas** on TPU v7 hardware (tested on both PyTorch 2.13 and PyTorch 2.15).

---

## The Modern TPU Hardware Hierarchy at a Glance
<div style="text-align: center; margin: 24px 0;">
  <img src="https://cloud.google.com/tpu/docs/images/tpu_memory_hierarchy_roofline.png" alt="Modern TPU Memory Hierarchy & Hardware Roofline" style="max-width: 100%; border-radius: 8px; border: 1px solid #dadce0;" />
  <p style="font-size: 13px; color: #5f6368; margin-top: 8px;">
    <strong>Figure 1:</strong> Memory latency hierarchy and arithmetic intensity roofline across TPU v6e and v7 hardware tiers.
  </p>
</div>

| Hardware Component | Geometry (TPU v7) | Execution Constraint | Optimal Alignment Target |
| :--- | :--- | :--- | :--- |
| **MXU (Matrix Multiply Unit)** | **256×256 physical systolic array** (four 128×128 quadrants) | Dense matrix multiplications process in 256×256 hardware tiles or packed 128×128 quadrants | Multiples of **256** (minimum 128) |
| **VPU (Vector Processing Unit)** | **(8, 128) vector registers**: 8 sublanes × 128 lanes = 1,024 32-bit elements (`index.py get vpu`) | Vector elementwise and reduction ALUs operate on whole registers; the last dimension pads to 128 lanes, the second-to-last to 8 sublanes | Last dimension a multiple of **128**, second-to-last a multiple of **8** |
| **VMEM (On-Chip Scratchpad)** | **64 MiB per core** (ultra-low latency; `index.py get vmem`) | Fast static RAM holds active operator fusions; spills to HBM if exceeded | Manage fusion cluster size |
| **HBM (High-Bandwidth Memory)** | **~3.7 TB/s per device** bandwidth (7.4 TB/s per chip, two devices per chip) | Feeds VMEM; throttles memory-bandwidth-bound operations | Maximize arithmetic intensity |
| **Peak Compute Ceiling** | **1,153.5 TFLOP/s (BF16)** per device | Speed-of-light hardware compute ceiling | Compute-bound kernels (I > ~310 FLOP/byte: 1,153.5 TFLOP/s ÷ ~3.7 TB/s) |

---

## 1. The MXU Systolic Array & Sublane Step Function: The +1 Element Cliff
Starting with TPU v6e (Trillium) and Cloud TPU v7 (Ironwood), the physical Matrix Multiply Unit (MXU) systolic array was expanded from 128×128 (on TPU v2–v5p) to **256×256**.

Internally, each 256×256 MXU is architected as a 2×2 grid of four **128×128 hardware quadrants**. This dual-granularity architecture allows the XLA compiler (`mxu_diagonal_packer`) to either:
1. **Full-Tile Dense Mode:** Execute a monolithic **256×256** matrix multiplication across the entire array for maximum compute density.
2. **Quadrant Packing Mode:** Diagonally pack two independent **128×128** matrix multiplications into opposing quadrants of the 256×256 MXU, preventing hardware idle time on smaller matrices.

Because matrix operations process in discrete 256×256 tiles and 128×128 quadrants, tensor dimensions that do not divide evenly by 128 or 256 pay the computational and memory cost of allocating the entire next hardware quadrant.

### Measured Hardware Reality: The +1 Element Cliff
We measured matrix multiplication latency across dimension N sweeping around 4,096 elements across 5 independent replicas on Cloud TPU v7 (replica spread 1.01×–1.03×):

| Dimension N | Alignment | Median Latency | Replica Spread | Step Ratio vs Previous | Hardware Execution State |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4,095** | Ragged (4095) | 724.8 µs | 1.03× | — | Sublane tail masking |
| **4,096** | **Exact multiple of 256 (16 × 256)** | **721.4 µs** | 1.03× | 0.995× | **Optimal full tile utilization (32 × 128 quadrants)** |
| **4,097** | **+1 element spill (4096 + 1)** | **760.1 µs** | 1.01× | **+5.4% (+38.7 µs)** | **New 128-wide quadrant allocated!** |
| **4,104** | +8 elements (4096 + 8) | 762.2 µs | 1.01× | 1.003× | Within 128-wide quadrant plateau |
| **4,128** | +32 elements (4096 + 32) | 761.3 µs | 1.02× | 0.999× | Within 128-wide quadrant plateau |
| **4,160** | +64 elements (4096 + 64) | 761.9 µs | 1.04× | 1.001× | Within 128-wide quadrant plateau |
| **4,224** | Next multiple of 128 (33 × 128) | 813.1 µs | 1.02× | **+6.7% (+51.2 µs)** | Next 128-wide quadrant boundary |

<aside class="important">
<b>The Sublane Plateau:</b> Adding just a single element (4,096 → 4,097) costs <b>+38.7 µs (+5.4%)</b>, after which latency remains completely flat up to 4,160. The hardware computes the full 128-wide quadrant regardless of whether your tensor uses it.
</aside>

## 2. The Vocabulary Alignment Rule: Free +7.5% Throughput
The most common manifestation of unaligned tensor dimensions in large language models is the **vocabulary size**.

For example, GPT-2 uses an unaligned vocabulary size of **50,257** (50,257 = 128 × 392 + 81). The final language model projection head (`lm_head`) projects hidden states into 50,257 logits, paying for 128 × 393 = 50,304 columns while masking out the remaining 47 columns.

### What Happens When You Pad to 128 or 256?
Rounding the vocabulary size up to the next multiple of 128 (50,304) or 256 (50,432) computes extra classes. Counterintuitively, **it runs measurably faster**:

| Vocabulary Size | Divisible by 128 / 256 | Median Latency (TPU v7) | MFU | Speedup vs Unaligned |
| :--- | :--- | :--- | :--- | :--- |
| **50,257 (Default GPT-2)** | ❌ No (Ragged) | **2,168.2 µs** | 67.4% | Baseline (unaligned) |
| **50,304 (Padded to 128)** | ✅ **Yes (Quadrant aligned)** | **2,006.4 µs** | **73.0%** | **1.081× FASTER (+7.5% gain)** |
| **50,432 (Padded to 256)** | ✅ **Yes (Full MXU tile aligned)** | **2,008.4 µs** | **73.1%** | **1.080× FASTER (+7.4% gain)** |
| **32,000 (LLaMA-2 Aligned)** | ✅ Yes (Divisible by 256) | 1,329.3 µs | 70.1% | Pre-aligned |
| **128,256 (LLaMA-3 Aligned)**| ✅ Yes (Divisible by 256) | 4,874.7 µs | 76.5% | Pre-aligned |

<aside class="tip">
<b>The Free Lunch Optimization:</b> Padding GPT-2's vocabulary from 50,257 to 50,304 (or 50,432) yields an immediate <b>7.5% to 7.6% speedup on Cloud TPU v7</b> while doing 0.09% more math! Unaligned trailing dimensions force sublane tail masking, which disables vectorization paths in the compiler.
</aside>

### Implementation: Aligned Vocabulary Embedding

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

class AlignedVocabEmbedding(torch.nn.Module):
  """Wraps an embedding table, automatically padding vocabulary to a multiple of 128."""

  def __init__(self, true_vocab_size: int, embedding_dim: int, alignment: int = 128):
    super().__init__()
    self.true_vocab_size = true_vocab_size
    remainder = true_vocab_size % alignment
    self.padded_vocab_size = true_vocab_size if remainder == 0 else true_vocab_size + (alignment - remainder)

    self.weight = torch.nn.Parameter(
        torch.randn(self.padded_vocab_size, embedding_dim, dtype=torch.bfloat16)
    )

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    # Safe forward pass; input_ids only index into [0, true_vocab_size)
    return torch.nn.functional.embedding(input_ids, self.weight)

  def lm_head_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # Matmul against padded weights, then slice off padding tokens
    logits = torch.matmul(hidden_states, self.weight.t())
    return logits[..., :self.true_vocab_size]

# Example: GPT-2 padded from 50,257 to 50,304
gpt2_aligned = AlignedVocabEmbedding(
    true_vocab_size=50257, embedding_dim=768, alignment=128
).to(device)
sample_ids = torch.randint(0, 50257, (4, 128), device=device)
hidden = gpt2_aligned(sample_ids)
logits = gpt2_aligned.lm_head_forward(hidden)

if hasattr(torch, "accelerator") and torch.accelerator.is_available():
  torch.accelerator.synchronize()

assert logits.shape == (4, 128, 50257), f"Unexpected shape: {logits.shape}"
assert torch.isfinite(logits).all(), "Logits must be finite"
```

---

## 3. Attention Head Dimension: The Arithmetic Intensity Truth
In Multi-Head Attention, common head dimensions include 64, 128, and 256. A persistent myth in deep learning is that `head_dim=64` performs poorly because it leaves an MXU systolic array half-empty due to zero-padding.

Our empirical roofline measurements reveal the actual hardware mechanism:

| Head Dimension (d_k) | Heads | Hardware Alignment Target | Median Latency | MFU | Achieved Bandwidth | Arithmetic Intensity (FLOP/byte) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **64** | 32 | Sub-quadrant packing | 1,598 µs | 7.1% | **2,562 GB/s** (~69% HBM peak) | 32 |
| **128** | 16 | Exact 128×128 Quadrant | 1,661 µs | 13.6% | **2,455 GB/s** (~66% HBM peak) | 64 |
| **256** | 8 | Native 256×256 Full MXU Tile | 1,483 µs | 30.2% | **2,722 GB/s** (~74% HBM peak) | 128 |

### The Key Takeaway: HBM Bandwidth Bound, Not Tile Padding Bound
Across all three head dimensions, the kernel achieves the **exact same memory bandwidth** (~2.5 – 2.7 TB/s), about 70% of the ~3.7 TB/s TPU v7 HBM peak per device. It is HBM-bound at every head dimension.

The reason MFU scales from 7.1% to 30.2% is not tile padding, but **Arithmetic Intensity** (I = FLOPs / BytesMoved). At `head_dim=64`, you move twice as much data from memory per arithmetic operation as at `head_dim=128`.

<aside class="important">
<b>Roofline Rule:</b> For memory-bandwidth-bound operations (like attention projection and decode), throughput is capped by HBM bandwidth (~3.7 TB/s peak per device on TPU v7). Increasing arithmetic intensity (larger batch size, standardizing on <code>head_dim=128</code> or <code>256</code>) shifts the kernel toward the compute-bound regime where the full 256×256 MXU systolic engines can be exploited.
</aside>

### Architectural Co-Design Case Study: The 16 × 256 Attention Head
Most open-source transformer architectures (such as LLaMA-2/3, Mistral, and original OLMo-3 7B) use `head_dim = 128` with 32 query heads for a 4,096-dimension hidden state (32 heads × 128 = 4,096).

In Google's large-scale TPU co-design studies and our empirical hardware verification benchmarks on TPU v7 (Ironwood), we tested a controlled iso-parameter, iso-FLOP ablation:
- **Baseline Architecture:** 32 heads × `head_dim = 128`
- **Co-Designed Architecture:** 16 heads × `head_dim = 256`
- **Model Parameters & Total FLOPs:** Strictly identical (16 × 256 = 4,096 hidden dimension; 67.1M attention parameters).

#### Empirical PyTorch Benchmark Measurements on TPU v7 Silicon:
Using standard PyTorch `torch.compile(backend="tpu")` benchmarks on Cloud TPU v7 hardware:

| Attention Workload Stage | Baseline: 32 Heads × 128 | Co-Designed: 16 Heads × 256 | Speedup (n=35 steps) | Hardware Mechanism |
| :--- | :--- | :--- | :---: | :--- |
| **Full Attention Layer (Forward)** | 2,224.1 µs | **1,682.9 µs** | **1.32× FASTER** (24.3% reduction) | 256×256 native systolic tiling + double arithmetic intensity |
| **Full Layer (Forward + Backward)** | 6,006.9 µs | **4,744.6 µs** | **1.27× FASTER** (21.0% reduction) | Backward gradient projection tiles cleanly across full MXU |
| **Isolated SDPA Attention Core** | 699.2 µs | **406.4 µs** | **1.72× FASTER** | Halves sequence-dimension tile fragmentation |

#### End-to-End Pre-Training Convergence:
1. **+12.4% Full-Model Speedup:** In MaxText full-model 7B pre-training (where attention speedup compounds with MLP layers), the model achieved a **+12.4% net training throughput gain**.
2. **Double Arithmetic Intensity:** Standardizing on `head_dim=256` doubles arithmetic intensity (from 64 to 128 FLOP/byte), moving attention projections closer to the compute-bound regime (ridge ~310 FLOP/byte on TPU7x), though still HBM-bound.
3. **Identical Convergence & Quality:** Across 120 billion pre-training tokens (30,000 steps), loss curves and downstream evaluations tracked the baseline identically within run-to-run variance.

<aside class="tip">
<b>Model Architecture Recommendation:</b> When designing or fine-tuning models for modern Cloud TPUs (v6e, v7), standardize attention query and key projection dimensions on <b><code>head_dim = 256</code></b> (adjusting head count to maintain hidden size). In PyTorch TorchTPU, this yields an immediate <b>1.27× to 1.32× acceleration</b> in attention layer wall time at zero parameter cost.
</aside>

## 4. The TPU v7 VMEM Shrink Reality
TPU generations exhibit significant architectural evolution in on-chip memory sizing:

| Hardware Specification | TPU v6e | TPU v7 (Ironwood) | Architectural Shift |
| :--- | :--- | :--- | :--- |
| **On-Chip Vector Memory (VMEM)** | 128 MiB | **64 MiB** | **50% Reduction in on-chip VMEM** |
| **High Bandwidth Memory (HBM3e)** | 32 GiB | **192 GiB** | **6× Increase in HBM Capacity** |
| **HBM Memory Bandwidth** | 1.6 TB/s | **7.4 TB/s** (chip) / **~3.7 TB/s** (device) | **4.5× Bandwidth Expansion** per chip |

### The Pallas Scoped VMEM Limit & Double Buffering Tax

When writing custom Pallas kernels or configuring manual tiling:

<aside class="caution">
<b>Scoped VMEM is a default, not a wall:</b> Physical VMEM on TPU v7 is 64 MiB per TensorCore, and no kernel can exceed it (e.g. <code>Used 64.01M of 63.94M vmem. Exceeded vmem capacity by 72.0K.</code>). Below that, each kernel is held to a <b>scoped VMEM limit</b>: its own <code>vmem_limit_bytes</code> (Pallas compiler params) if it passes one, otherwise the libtpu flag <code>--xla_tpu_scoped_vmem_limit_kib</code>, otherwise a default of 32 MiB. The kernel value wins over the flag, even when lower, and a value above capacity is clamped to 63.94 MiB without an error. This was measured on TPU7x (tpu7x-8) with libtpu 0.0.44 and 0.0.44.1, and differs from the JAX <code>pallas/mosaic/core.py</code> docstring, which says the kernel value must be paired with a larger flag; check your own build before relying on a number. A kernel over its scoped limit fails at compile time with <code>CompileTimeScopedVmemOom</code>.
</aside>

This is XLA error E1001 (`index.py get e1001`).

#### Double Buffering Multiplies VMEM Consumption by 2×
To overlap Direct Memory Access (DMA) transfers from HBM with vector/matrix computation, Pallas uses pipelined double buffering (`buffer_count=2`). However, double buffering doubles the on-chip footprint for all managed tiles:

<pre>Scoped VMEM = Sum(Tile Size in Bytes × buffer_count)</pre>

If your kernel declares:
* Input Tile A: 2048 × 2048 in BF16 = 8 MB
* Input Tile B: 2048 × 2048 in BF16 = 8 MB
* Output Tile C: 2048 × 2048 in BF16 = 8 MB

With `buffer_count=1`, Scoped VMEM = 24 MB.
With `buffer_count=2`, Scoped VMEM = **48 MB**, which fails with `CompileTimeScopedVmemOom` unless the scoped limit is raised to cover it.
**Solution:** Either raise the scoped limit (48 MB still fits in the 64 MiB physical VMEM), or reduce tile block shapes (for example, to 1024 × 2048) so that total double-buffered VMEM stays under the current scoped limit.

#### SPU vs DMA Engine Scheduling Overhead
Tiling too conservatively creates the opposite pathology: **SPU loop latency starvation**. If block dimensions along the major axis are set too small (e.g. `(8, 128)`), the Scalar Processor Unit (SPU) spends ~33 ns orchestrating the loop for only ~8 ns of vector math. Always ensure block sizes along contiguous dimensions are at least **128** elements to sustain >85% DMA transfer efficiency.

---

## Summary: The Golden Rules of TPU Alignment
1. **Rule of 256 / 128**: Always round matrix multiplication dimensions (M, K, N) to multiples of **256** for native full-tile MXU utilization, or at minimum multiples of **128** for quadrant packing. For vector elementwise operations, round to multiples of **128 / 256**.
2. **Pad Vocabularies**: Round vocabulary sizes up to the nearest multiple of 128 or 256 (50,257 → 50,304 or 50,432). It is free, avoids sublane tail masking, and yields a **7.5% speedup**.
3. **Know Your Scoped VMEM Limit**: When designing custom kernels or double buffering (`buffer_count=2`), keep aggregate tile residency under the scoped VMEM limit (`--xla_tpu_scoped_vmem_limit_kib`, raised per kernel with `vmem_limit_bytes`) to prevent `CompileTimeScopedVmemOom`, and never above physical VMEM (64 MiB per core on TPU v7).
4. **Tile Major Dimensions ≥ 128**: Keep block dimensions along major contraction axes ≥ 128 to amortize SPU instruction dispatch and maximize DMA pipeline bandwidth.
5. **Know Your Roofline**: Low MFU does not necessarily mean poor implementation. Check arithmetic intensity: on TPU v7 the HBM peak is ~3.7 TB/s per device, so an operation at ~2.5 TB/s is memory-bound at ~70% of peak, with some headroom left.
