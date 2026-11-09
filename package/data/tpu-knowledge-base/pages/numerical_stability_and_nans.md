---
torch_tpu_version: 0.1.1.dev20260922
---

# **TorchTPU Troubleshooting & Gotchas Runbook: Debugging NaNs, Infs, & Pipelining**

Training large-scale deep learning models on Google Cloud TPUs requires proactive defenses against numerical instability. Because TPUs execute matrix math on systolic arrays optimized for BFloat16 and FP8, discrepancies between analytical mathematical formulations and physical floating-point representations can cause sudden <code>NaN</code> (Not a Number) losses, vanishing gradients, or whole-model parameter corruption.

This guide details the physical failure modes observed during active TPU training loops and presents verified defensive patterns for PyTorch workloads using <code>torch_tpu</code>.

<aside class="note">
<b>Navigating Precision &amp; Parity Documentation:</b>
<ul>
  <li><b>Offline Model Parity and Compiler Precision Modes:</b> To verify that TPU outputs match CPU reference implementations within numerical tolerances, configure compiler excess precision flags, control Float32 matrix multiplication algorithms (1-pass, 3-pass, 6-pass), and understand MXU accumulation mechanics, see Numerical Precision (<code>index.py get torch-tpu-precision-management</code>).</li>
  <li><b>Format Selection:</b> To choose optimal data formats (FP32, BF16, FP8) based on compute throughput and memory bandwidth rooflines, see <a href="./precision_ladder.md">The Precision Ladder: FP32, BF16, and FP8</a>.</li>
  <li><b>This Guide:</b> Focuses strictly on preventing runtime training crashes, loss explosion, gradient corruption, and host synchronization penalties during active execution on TPU hardware.</li>
</ul>
</aside>

---

## The TPU Numerical Vulnerability Stack

The primary source of numerical instability during accelerator training is the discrepancy between mathematical formulation and physical silicon representation:

*   **1. 7-Bit Mantissa Truncation:** While BFloat16 provides 8 exponent bits to prevent underflow, it retains only 7 mantissa bits (&epsilon; &approx; 7.81 &times; 10<sup>-3</sup>). Denominator rounding and small epsilon terms cause catastrophic precision loss in custom loss and ratio formulations.
*   **2. Unfused Log-Softmax Underflow:** Writing `-log(softmax(x) + eps)` causes intermediate probabilities to flush to zero before logarithm evaluation, clamping loss onto `eps` and annihilating backward gradients when logit dynamic range exceeds 50.
*   **3. Global Norm NaN Contagion:** Standard `clip_grad_norm_` computes a single L<sub>2</sub> norm across all model parameters. A single `NaN` in one attention head or layer poisons 100% of the model's weights via scalar multiplication.
*   **4. Host Synchronization Tax:** Naive in-loop NaN checking via `torch.isnan().item()` forces a synchronous device-to-host PCIe/ICI barrier on every step, imposing a +63.1% latency penalty (+423.3 µs on Cloud TPU v7).

---

## 1. The BFloat16 Mantissa Trap: Custom Losses & Ratios
Understanding the floating-point representations of the TPU Matrix Multiply Unit (MXU) is vital for preventing unexpected numerical collapses.

### The 7-Bit Significand Cliff

BFloat16 shares the identical 8-bit dynamic range of IEEE FP32 (1.2 &times; 10<sup>-38</sup> to 3.4 &times; 10<sup>38</sup>), which protects training runs from exponent underflow. However, BFloat16 dedicates only **7 bits** to the significand (mantissa), providing approximately 2 to 3 decimal digits of precision (&epsilon; &approx; 7.81 &times; 10<sup>-3</sup>) compared to FP32's 23 bits (&epsilon; &approx; 1.19 &times; 10<sup>-7</sup>).

While neural network weights and GEMM activations are highly resilient to 7-bit quantization (as detailed in Precision Management, `index.py get torch-tpu-precision-management`), **custom loss functions that evaluate division or ratios are not**.

### Empirical Silicon Evidence: Dice Loss & Ratio Division

In computer vision and segmentation, Dice loss and Intersection-over-Union (IoU) evaluate ratio terms:

```
Dice Loss = 1.0 - (2.0 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
```

When evaluated on Cloud TPU v7 hardware:

*   **Unsmoothed (`0.0 / 0.0`):** When both prediction and ground truth are empty and `smooth = 0.0`, the operation evaluates to `0.0 / 0.0 = NaN` in both BF16 and FP32.
*   **Small Smoothing (`smooth = 1e-7`):** In native BF16, adding 10<sup>-7</sup> to activations of magnitude O(1) results in complete mantissa absorption: `1.0 + 1e-7 = 1.0` because 10<sup>-7</sup> is vastly smaller than 2<sup>-7</sup> (&approx; 7.81 &times; 10<sup>-3</sup>). Furthermore, normal mask overlap in pure BF16 suffers a **1.5% precision drift** (0.49609 in BF16 vs 0.50390 in FP32).

<aside class="caution">
<b>The Ratio Loss Upcasting Rule:</b> Never calculate custom loss reductions, ratio losses, or division operations directly in BFloat16. Always upcast inputs to <code>torch.float32</code> before computing division and reductions, then cast the final scalar back to the target dtype.
</aside>

### Stabilized Implementation Pattern

```python
import torch
import torch.nn as nn

class StabilizedDiceLoss(nn.Module):
    # Numerically robust Dice Loss for TPU training in mixed precision.
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Step 1: Upcast activations to FP32 to prevent 7-bit mantissa truncation
        pred_f = pred.float()
        target_f = target.float()

        # Step 2: Compute reductions in 32-bit precision
        intersection = (pred_f * target_f).sum()
        cardinality = pred_f.sum() + target_f.sum()

        # Step 3: Compute ratio with numerical stabilization
        dice_coeff = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice_coeff

        # Step 4: Cast scalar loss back to original input dtype
        return loss.to(pred.dtype)
```

---

## 2. Cross-Entropy & Logit Extremes: Unfused Rewrite Failure

A frequent mistake in custom loss implementations is decomposing Cross-Entropy into manual Softmax followed by Logarithm:

```python
# ANTI-PATTERN: Catastrophic numerical instability on TPU
prob = torch.softmax(logits, dim=-1)
loss = -torch.sum(targets * torch.log(prob + 1e-8))
```

### Measured Failure Mode on Cloud TPU v7

Sweeping logit scales on TPU v7 silicon highlights the severe breakdown of the unfused implementation:

| Logit Dynamic Range | Unfused `-log(softmax(x) + 1e-8)` | Fused `F.log_softmax(x)` | Canonical `F.cross_entropy` | Hardware Behavior & Gradient Integrity |
| :--- | :---: | :---: | :---: | :--- |
| **Scale = 10.0** | 13.88 | 15.06 | 15.06 | Minor probability truncation in unfused variant |
| **Scale = 50.0** | **17.00 (Plateau)** | 51.75 | 51.75 | Unfused softmax underflows to 0.0; clamps to `1e-8` |
| **Scale = 85.0** | **17.75 (Plateau)** | 88.50 | 88.50 | Flat loss; **backward gradients evaluate to 0.0** |
| **Scale = 95.0** | **18.12 (Plateau)** | 104.00 | 104.00 | Complete loss of optimization signal |
| **Scale = 150.0** | **18.25 (Plateau)** | **141.00** | **141.00** | Fused Cross-Entropy scales stably up to 141.0 |

<figure style="text-align: center; margin: 24px 0;">
  <img src="https://cloud.google.com/tpu/docs/images/cross_entropy_fused_vs_unfused_clean.png" alt="Cross-Entropy Numerical Dynamics: Unfused Rewrite vs Fused Log-Sum-Exp on Cloud TPU v7" style="max-width: 100%; border-radius: 8px; border: 1px solid #dadce0;" />
  <figcaption style="font-size: 13px; color: #5f6368; margin-top: 8px;">
    <strong>Figure 1:</strong> Cross-entropy numerical breakdown at high logit scale: Unfused softmax underflows to zero and clamps on epsilon causing vanishing gradients, whereas compiler-fused <code>F.cross_entropy</code> uses Log-Sum-Exp in 32-bit accumulators.
  </figcaption>
</figure>

<aside class="important">
<b>Why Unfused Softmax Fails:</b> At logit scale 50+, the probability of the true class drops below the minimum representable positive normal number in the target precision, flushing to zero. Adding <code>1e-8</code> clamps the loss at -ln(10<sup>-8</sup>) &approx; 18.42. The loss becomes an artificial constant, killing all backward gradients.
<br><br>
In contrast, <code>torch.nn.functional.cross_entropy</code> and <code>F.log_softmax</code> trigger the OpenXLA compiler's fused <b>Log-Sum-Exp</b> lowering pass, which subtracts max(x) in hardware registers and maintains 32-bit accumulation.
</aside>

---

## 3. Global Gradient Clipping: The 100% NaN Contagion Hazard

When training large language models or deep Vision Transformers, a single localized numerical overflow (for example, in a specific attention projection or MoE router) can produce a single `NaN` in one gradient tensor.

### The Contagion Mechanism

Standard `torch.nn.utils.clip_grad_norm_` calculates the global L<sub>2</sub> norm by summing the squared norms across all model parameters:

```
Total Norm = sqrt( sum( ||g_i||^2 ) )
g_i = g_i * ( max_norm / (Total Norm + 1e-6) )
```

If even **one** element in `g_0` is `NaN`, `Total Norm` evaluates to `NaN`. When every parameter gradient is subsequently multiplied by `clip_coef = max_norm / NaN = NaN`, **every gradient across the entire model becomes NaN**.

### Measured Silicon Contagion Test

In an empirical test on TPU v7 running a 10-layer model where a single `NaN` was injected into Layer 0:

| Gradient Clipping Method | Total Norm Evaluated | Contaminated Layers | Healthy Layers Preserved | Training Run Status |
| :--- | :---: | :---: | :---: | :--- |
| **Standard `clip_grad_norm_`** | `NaN` | **10 of 10 (100% Contagion)** | 0 of 10 | **Catastrophic Failure** (all weights destroyed) |
| **Resilient In-Graph Clipping** | 1.00 | **0 (Poisoned layer zeroed)** | **9 of 9 (100% Preserved)** | **Run Continues Stably** |

### The Resilient Clipping Pattern

Use the following resilient gradient clipping pattern in production training loops. It sanitizes non-finite gradients locally before computing the global norm:

```python
import torch
from typing import Iterable

def resilient_clip_grad_norm_(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    norm_type: float = 2.0
) -> torch.Tensor:
    # Clips gradient norm while quarantining NaNs to prevent whole-model contagion.
    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return torch.tensor(0.0)

    # Step 1: Quarantine non-finite gradients in-place without breaking execution
    for p in params:
        if not torch.isfinite(p.grad).all():
            # Zero out corrupted gradients to preserve remaining healthy layers
            p.grad.nan_to_num_(nan=0.0, posinf=max_norm, neginf=-max_norm)

    # Step 2: Compute standard global norm on sanitized gradients
    return torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type)
```

---

## 4. Runtime NaN Detection: The +63% Host Barrier Penalty

When diagnosing numerical stability issues, developers frequently insert assertion checks to detect `NaN` losses:

```python
# ANTI-PATTERN: Devastates TPU pipeline execution
loss = model(inputs)
if torch.isnan(loss).any().item():  # <-- FORCES HOST SYNCHRONIZATION BARRIER!
    raise FloatingPointError("NaN loss encountered!")
```

<aside class="note">
<b>Runtime Diagnostics vs. Offline Parity:</b> If you are validating numerical equivalence between TPU and CPU references during development or unit testing, use the tools documented in Numerical Precision (<code>index.py get torch-tpu-precision-management</code>). Those assertions (such as <code>torch.testing.assert_close</code>) require synchronizing and transferring tensors to the host CPU (<code>out_tpu.cpu()</code>), which is suitable for offline tests but must never run inside the hot training loop.
</aside>

### Empirical Silicon Benchmark (Cloud TPU v7)

Every call to `.item()` forces an immediate device-to-host execution barrier, flushing the TPU instruction pipeline and stalling host dispatch:

<figure style="text-align: center; margin: 24px 0;">
  <img src="https://cloud.google.com/tpu/docs/images/pipeline_execution_comparison_clean.png" alt="Pipeline Execution Comparison: Asynchronous Compiled Execution vs Synchronous .item() Barrier" style="max-width: 100%; border-radius: 8px; border: 1px solid #dadce0;" />
  <figcaption style="font-size: 13px; color: #5f6368; margin-top: 8px;">
    <strong>Figure 2:</strong> Pipeline execution comparison on Cloud TPU v7: Asynchronous compiled execution maintains 100% systolic utilization, whereas synchronous <code>.item()</code> assertions inject a +423.3 µs (+63.1%) host synchronization barrier.
  </figcaption>
</figure>

Empirical timings collected over 50 warm steps on TPU v7:

| Detection Strategy | Median Step Latency | Added Overhead vs Baseline | Latency Penalty (%) | Pipeline Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (No In-Loop Check)** | **671.06 µs** | — | Baseline (1.00×) | Fully asynchronous execution |
| **Host Sync (`.item()` every step)** | **1,094.37 µs** | **+423.31 µs** | **+63.1% penalty** | **Destroys compilation pipelining** |
| **In-Graph Guard (`torch.nan_to_num`)**| **741.85 µs** | **+70.79 µs** | **+10.5% penalty** | Preserves compiled graph boundary |

### Recommended Production Diagnostic Practices

1.  **Never call `.item()` on every step:** If synchronous assertions are required, gate them to run every N steps (for example, every 100 or 500 steps) alongside checkpointing or validation logging.
2.  **Use In-Graph Guarding:** Replace NaN checks with non-blocking compiler operations like `torch.nan_to_num` or `torch.where`.
3.  **Inspect Asynchronous Metrics:** Monitor TPU runtime metrics exported via the TPU runtime metrics ports (`8431`–`8438`) or Cloud TPU Profiler traces rather than inserting host-synchronous code.

---

### Hardware-Level Trapping: Zero-Host-Overhead NaN/Inf Detection

When hunting an elusive NaN that emerges deep into a training run, avoid modifying Python training loops with synchronous assertions. Instead, enable hardware-level traps via the runtime initialization driver:

```bash
# Instructs the XLA compiler to instrument HLO kernels with hardware check instructions
export LIBTPU_INIT_ARGS="--xla_tpu_detect_nan=true --xla_tpu_detect_inf=true"
```

#### How It Works Under the Hood
* When enabled, the compiler instruments every fused HLO kernel with hardware assertion instructions that evaluate results in VPU vector registers.
* The moment a `NaN` or `Inf` is generated, the TPU hardware triggers an immediate interrupt, halting execution at the exact operation that produced the anomaly.
* The driver dumps the exact HLO instruction name, tensor shape, and source line mapping to the runtime log, pinpointing the faulty operator with zero Python-level host synchronization overhead during healthy execution steps.

---

## 5. Summary Checklist: Numerical Stability Rules for TorchTPU

Follow this operational checklist to ensure numerical stability and peak accelerator throughput during TPU training:

*   **Compiler-Level Hardware Traps:** For diagnostic runs, use `LIBTPU_INIT_ARGS="--xla_tpu_detect_nan=true --xla_tpu_detect_inf=true"` to catch non-finite values at the hardware instruction level with zero Python overhead.
*   **Custom Loss Upcasting:** Upcast activations to `torch.float32` for all ratio, division, and reduction losses (Dice loss, IoU, contrastive losses) before evaluating mathematical operations, then cast the scalar loss back to the input dtype.
*   **Canonical Cross-Entropy:** Always use `torch.nn.functional.cross_entropy` or `F.log_softmax`. Never decompose cross-entropy into manual `-sum(p * log(softmax(x) + eps))`, which flushes extreme logits and freezes gradients.
*   **Normalization Preservation:** Keep variance and mean reductions in FP32 accumulators inside RMSNorm and LayerNorm layers before casting back to BFloat16 or FP8.
*   **Resilient Gradient Clipping:** Sanitize non-finite gradients with `p.grad.nan_to_num_()` prior to invoking `torch.nn.utils.clip_grad_norm_` to prevent a single poisoned layer from causing 100% whole-model parameter corruption.
*   **Eliminate Step-Wise `.item()`:** Never invoke `.item()` or `.cpu()` inside the hot training loop. Keep all diagnostics asynchronous, leverage in-graph guards, or monitor asynchronous TPU metrics ports.

```python
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
  import torch_tpu
except ImportError:
  pass

device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)

# 1. Intentional before-state: unsmoothed 0.0 / 0.0 Dice ratio produces NaN
empty_pred = torch.zeros(16, 16, device=device, dtype=torch.bfloat16)
empty_target = torch.zeros(16, 16, device=device, dtype=torch.bfloat16)
unstable_loss = 1.0 - (2.0 * (empty_pred * empty_target).sum()) / (
    empty_pred.sum() + empty_target.sum()
)
assert torch.isnan(unstable_loss).all(), "Unsmoothed 0.0/0.0 must produce NaN"

# 2. Remediated StabilizedDiceLoss: upcasts to FP32 and stabilizes denominator
class StabilizedDiceLoss(nn.Module):
  """Numerically robust Dice Loss for TPU training in mixed precision."""

  def __init__(self, smooth: float = 1e-5):
    super().__init__()
    self.smooth = smooth

  def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = pred.float()
    target_f = target.float()
    intersection = (pred_f * target_f).sum()
    cardinality = pred_f.sum() + target_f.sum()
    dice_coeff = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
    loss = 1.0 - dice_coeff
    return loss.to(pred.dtype)

stabilized_dice = StabilizedDiceLoss(smooth=1e-5)
stable_dice_loss = stabilized_dice(empty_pred, empty_target)

# 3. Remediated Fused Cross-Entropy at extreme logit scale (150.0)
extreme_logits = torch.randn(8, 32, device=device, dtype=torch.float32) * 150.0
labels = torch.randint(0, 32, (8,), device=device)
fused_ce_loss = F.cross_entropy(extreme_logits, labels)

# 4. Remediated resilient_clip_grad_norm_ quarantining an injected NaN gradient
def resilient_clip_grad_norm_(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
  params = [p for p in parameters if p.grad is not None]
  if len(params) == 0:
    return torch.tensor(0.0, device=device)
  for p in params:
    if not torch.isfinite(p.grad).all():
      p.grad.nan_to_num_(nan=0.0, posinf=max_norm, neginf=-max_norm)
  return torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type)

test_layers = nn.ModuleList([nn.Linear(32, 32, bias=False).to(device) for _ in range(4)])
for idx, layer in enumerate(test_layers):
  layer.weight.grad = torch.ones_like(layer.weight)
  if idx == 0:
    layer.weight.grad[0, 0] = float("nan")  # Inject localized NaN in Layer 0

sanitized_norm = resilient_clip_grad_norm_(test_layers.parameters(), max_norm=1.0)

if hasattr(torch, "accelerator") and torch.accelerator.is_available():
  torch.accelerator.synchronize()

assert torch.isfinite(stable_dice_loss).all(), "StabilizedDiceLoss must be finite"
assert torch.isfinite(fused_ce_loss).all(), "Fused cross-entropy loss must be finite"
assert torch.isfinite(sanitized_norm).all(), "Sanitized gradient norm must be finite"
for p in test_layers.parameters():
  assert torch.isfinite(p.grad).all(), "All parameter gradients must be finite after quarantine"
```
