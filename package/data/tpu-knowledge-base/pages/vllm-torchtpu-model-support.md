---
title: "vllm-torchtpu model support"
type: source
tags: [serving, vllm, models]
created: 2026-10-25
updated: 2026-10-25
sources: 1
human_reviewed: false
---

# `vllm-torchtpu` Model Support

This is the single source of truth for which models the `vllm-torchtpu` engine natively supports; update it here rather than repeating it elsewhere.

## 1. Native `vllm-torchtpu` Supported Models

These models are deployed and tuned through the `vllm-torchtpu` engine (engine serving parameters, tensor parallel sizing, Pallas paged attention kernels, continuous batching and scheduler arguments, and XLA compiler flags):
* **DeepSeek v2 / v4**
* **Gemma 4 (26B / 31B)**
* **GLM 5.2**
* **GPT-OSS 20B** (`openai/gpt-oss-20b`): only with its `vllm_torchtpu` patch applied at launch, as the shared vLLM patch registry lists it. The patch dequantizes the MXFP4 experts to bf16, so MXFP4 compute is not an available knob.
* **Kimi (K2.6 / 3)**
* **Qwen 3 Coder**
* **Qwen 3.5 (35B / 397B)**

## 2. Models Not Supported by `vllm-torchtpu`

Every model not listed in Section 1 runs as native PyTorch / `torch_tpu` without an inference engine. Representative models:
* **Qwen 3.8 (27B)**: Dense Transformer testing static KV-cache preallocation, single-step decode compilation, and 128-byte tile padding alignment.
* **Custom / Unsupported LLMs**: Arbitrary HuggingFace architectures lacking native `vllm-torchtpu` engine or kernel implementations.
