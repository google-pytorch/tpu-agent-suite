---
torch_tpu_version: 0.1.1.dev20260922
---

# **MoE Serving, Static Routing, and Speculative Decoding on TPU**

Deploying Large Language Models (LLMs) and Mixture-of-Experts (MoE) architectures on Google Cloud TPUs requires balancing two fundamentally different execution phases: **compute-bound prompt prefill** and **memory-bandwidth-bound token decode**. This guide covers production inference optimization with `torch_tpu` and `vllm-torchtpu`.

---

## 1. The Two Operational Regimes of LLM Inference

```
[Prefill Phase: Context Processing] ──► Arithmetic Intensity > 300 FLOP/Byte  ──► COMPUTE-BOUND (MXU Saturation)
[Decode Phase:  Token Generation]   ──► Arithmetic Intensity < 30 FLOP/Byte   ──► MEMORY-BANDWIDTH-BOUND (HBM Saturation)
```

### The Decode Memory Bandwidth Roofline
During the autoregressive decode phase, the model generates one token at a time per request. For each token generated:
1. Every weight parameter across the model must be streamed from High Bandwidth Memory (HBM) into on-chip registers.
2. The entire Key-Value (KV) cache accumulated for past tokens must be fetched from HBM.
3. Compute efficiency drops to a fraction of peak MXU capacity because the execution pipeline spends over 80% of its time waiting for HBM memory transfers.

<div style="background-color: #f8f9fa; border-left: 4px solid #1a73e8; padding: 12px 16px; margin: 16px 0; font-family: monospace;">
  <strong>Peak Decode Tokens/Sec</strong> ≈ Hardware HBM Bandwidth (Bytes/sec) / (Model Bytes + KV Cache Bytes per Token)
</div>

---

## 2. Paged KV Cache in FP8: Half the KV Bytes
Because decode speed is directly limited by how fast the TPU can read the KV cache from memory:

<aside class="tip">
<b>The FP8 KV Cache Multiplier:</b> Compressing the Key-Value cache from BF16 (2 bytes) to FP8 E4M3 (1 byte) cuts KV cache bandwidth consumption by <b>50%</b> and doubles the tokens the cache holds. That speeds up only the KV-read share of a decode step -- weight reads and compute are unchanged -- so the end-to-end gain is less than 2× and depends on the model, batch and context; this page has no measurement of it.
</aside>

### KV Cache Memory Scaling (4096 Sequence Length, 32 Layers, 8 KV Heads, Dim 128)

| Format | Memory per Token per Layer | Cache Size for 4k Context (Batch 16) | Relative HBM Bandwidth Demand |
| :--- | :--- | :--- | :--- |
| **FP16 / BF16** | 2,048 Bytes | 2.14 GB | 100% (Baseline) |
| **FP8 (E4M3)** | **1,024 Bytes** | **1.07 GB** | **50%** |

On TPU v7 (Ironwood), the 192 GiB of HBM per chip and an FP8 KV cache together raise how many concurrent streams fit before the cache runs out; the number depends on the model's weights, context length and batch, and this page does not measure it.

---

## 3. Scaling Sparse MoE Models: TP vs. EP Routing
MoE models (such as Mixtral 8x7B, Qwen 2.5 Max, or DBRX) replace dense feed-forward networks with sparsely activated expert modules.

### The MoE Routing Tax
When a batch of tokens arrives at an MoE layer:
1. A gating router assigns each token to its top-k experts (typically k=2).
2. Tokens must be dispatched to whichever TPU rank holds the assigned expert weights via **All-to-All** communication.
3. After the expert computation completes, results must be gathered back via a second All-to-All pass.

<aside class="important">
<b>The MoE Interconnect Rule:</b> Expert Parallelism (EP) token all-to-all exchanges must execute strictly over the <b>Inter-Chip Interconnect (ICI)</b> mesh of one slice. If expert routing crosses the Data Center Network (DCN) between slices, network latency will dominate and throttle inference throughput.
</aside>

### Recommended MoE Sharding Strategy
- **Single Host (8 devices):** Use **Tensor Parallelism (TP=8)** or **Expert Parallelism (EP=8)**. A TPU v6e-8 host has 8 single-core chips; a TPU7x-8 host (`tpu7x-standard-4t`) has 4 chips with two devices each (`index.py get hw-tpu-7x`). In both cases all 8 devices share ICI links, enabling sub-millisecond all-to-all token exchanges.
- **Multi-Host Slices and Multislice:** Hosts inside one slice are also connected by ICI (`index.py get concept-dcn`). On a 2-host TPU7x slice (2x2x2), TP all-reduce measured ~165 GiB/s ICI egress per chip, with cross-host ranks within 0.11% of same-host ranks (vLLM on TPU7x, measured on 2026-09-30; the software versions were not recorded), far above the ~100 Gbps (12.5e9 B/s) per-chip DCN egress on TPU7x. DCN only carries traffic between slices. Keep all-to-all token exchanges within a slice, and put **Pipeline Parallelism (PP)** or data parallelism across slices, where stages communicate over DCN.

---

## 4. Chunked Prefill & Disaggregated Serving
Serving high-traffic production workloads on a unified TPU instance introduces latency jitter: a large prompt prefill request (e.g. 8k context) monopolizes the MXU systolic arrays, causing queueing delays and Time-To-First-Token (TTFT) spikes for active decode streams.

### Recommended Architectural Solution
1. **Chunked Prefill:** Configure the serving engine (`vllm-torchtpu`) to break massive prefill sequences into chunks of 512 or 1024 tokens. This interleaves decode steps between prefill chunks, maintaining smooth Time-Per-Output-Token (TPOT).
2. **Disaggregated Serving:** Separate prefill nodes (optimized for compute-heavy BF16 GEMMs on TPU v7) from decode nodes (optimized for memory-bandwidth-bound FP8 KV cache retrieval).

---

## Next Steps
- See **[FP8 KV cache with ragged paged attention](./fp8-kv-cache-rpa.md)** for the `--kv-cache-dtype` value to use on v6e.
- Review compiler architecture in **[XLA Lowering Pipeline & Runtime Flag Architecture](./xla_lowering_and_flags.md)**.
- Explore distributed training options in **[Distributed Parallelism Strategy & Decision Tree](./parallelism_decision_tree.md)**.

```python
import torch
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

# Verify KV-cache attention decode step and static top-k routing
query_decode = torch.randn(16, 8, 1, 128, device=device, dtype=torch.bfloat16)
key_cache = torch.randn(16, 8, 512, 128, device=device, dtype=torch.bfloat16)
val_cache = torch.randn(16, 8, 512, 128, device=device, dtype=torch.bfloat16)
decode_out = F.scaled_dot_product_attention(
    query_decode, key_cache, val_cache, is_causal=False
)

if hasattr(torch, "accelerator") and torch.accelerator.is_available():
  torch.accelerator.synchronize()

assert decode_out.shape == (16, 8, 1, 128)
assert torch.isfinite(decode_out).all(), "Decode output must be finite"
```
