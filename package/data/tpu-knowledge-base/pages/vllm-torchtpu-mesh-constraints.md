<!--
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
---
entry: vllm-torchtpu-mesh-constraints
keyword: vllm-torchtpu-mesh-constraints
url: not applicable -- see Sources
authority: first-party-verified
retrieved: 2026-10-16
source_updated: 2026-10-16
generator: authored
---

# vLLM layouts that fail to start on vLLM-TorchTPU

Each layout below passes vLLM's configuration dry run and then fails when
the server starts, or soon after. They are the table `KERNEL_CONSTRAINTS` in
`oss_meshes.py`: tpu-diagnostics' serving candidates never propose a layout
one of them rules out on the serving image. Onboarding's TP plan does not
consult the table (it does not know the image digest), so a launch at a
planned layout can still hit one. Search this page by the error a failed
launch printed; each section quotes it.

A constraint is either **structural**, coming from the kernel or engine
code, so it holds on every image until someone re-verifies it, or **seen on
specific images**, so it rules a layout out only on those images and warns
on any other.

Layouts are written TP x DP (tensor-parallel size x data-parallel replicas)
on 8 devices: a tpu7x 2x2x1 slice, two devices per chip.

## Applies when

- A vLLM server on `vllm_torchtpu` fails to start, or hangs on its first
  requests, at a layout the dry run accepted.
- A serving-candidates report lists a layout under `rejected_meshes`.
- Choosing a layout by hand with `--tp`, `--data-parallel-size`,
  `--pipeline-parallel-size` or `--enable-expert-parallel`.

## TP 1 on tpu7x: no single-device layout

- Rule: on tpu7x, torch_tpu has no layout for a one-device engine (its v7
  topology table starts at 2 devices, one chip), so TP 1 fails worker init
  for every model, with or without replicas (TP 1 x DP 8 included). v5e,
  v5p and v6e have a 1-device layout[^tt-topo].
- Fails with: `No TPU topology found for count: 1`
- Seen on: vLLM 0.29.0, Qwen3.8-27B (2026-10-13) and Gemma-4-31B
  (2026-10-14).
- Onboarding's TP plan asks the installed torch_tpu for its layouts, so it
  never plans TP 1 there.

## Pipeline parallelism with TP above 1 (structural)

- Rule: pipeline parallelism on TPU runs only with one device per stage;
  `--pipeline-parallel-size` above 1 with `--tensor-parallel-size` above 1
  fails.
- Fails with: `NotImplementedError: Pipeline parallelism on TPU does not
  support tensor parallelism inside a stage yet`[^vtt-pp]
- Seen on: vLLM 0.29.0, Qwen3.8-27B TP 4 x PP 2 (2026-10-15).

## Gated DeltaNet v3 block size (structural)

- Rule: for Qwen3.5 / Qwen3.8 Gated DeltaNet layers, each device's share of
  the value heads sets the kernel's triangular block size,
  `max(2, min(16, 16 // ceil(v_local / 8)))`, which must divide 16[^vtt-gdn]. With 48
  value heads, TP 2 leaves 24 per device and fails; TP 4 and TP 8 pass this
  rule.
- Fails with: `array split does not result in an equal division`
- Seen on: vLLM 0.29.0, Qwen3.8-27B TP 2 (2026-10-14).

## Qwen3.8-27B at TP 8 (seen on two images)

- Rule: Qwen3.8-27B crashes during engine init at TP 8, although its heads
  allow it (24 query heads, 4 KV heads replicated twice, 16 and 48
  linear-attention heads).
- Fails with: `Worker failed with error '(384, 256)'`
- Seen on: vLLM 0.29.0 (2026-10-14) and vLLM 0.31.0 with
  `--attention-backend CUSTOM` (2026-10-16). A newer
  plugin fails at the same TP with a GDN state assertion in
  `pool_adapters.py` (2026-10-12). TP 4 x DP 2 serves.

## Expert parallelism with more than 16 experts (seen on one image)

- Rule: with `--enable-expert-parallel`, a model with more than 16 experts
  per expert-parallel group trips the MoE kernel's row-partition limit.
- Fails with: `AssertionError: num_row_partitions=32 must be <=
  num_simd_lanes=16`
- Seen on: a 2026-08-04 nightly build, gpt-oss-20b (32 experts) TP 2 x DP 4
  with expert parallelism (2026-10-14). TP 8 with expert parallelism served
  on vLLM 0.22.1, so this is not assumed to hold on other builds.

## Data-parallel replicas hang on first requests (seen on two images)

- Rule: with `--data-parallel-size` above 1 the server reports healthy, then
  the first benchmark requests never complete.
- Fails with: `TimeoutError: RPC call ... timed out`
- Seen on: vLLM 0.22.1, Gemma-3-27B TP 4 x DP 2, and a 2026-08-04 nightly
  build, gpt-oss-20b (2026-10-14). On vLLM 0.29.0, Gemma-4-31B and
  Qwen3.8-27B serve at TP 4 x DP 2.

## Prefill context parallelism (vLLM, structural)

- Rule: `--prefill-context-parallel-size` above 1 runs only on vLLM's Model
  Runner V2, for MLA models, without pipeline parallelism, data parallelism
  or multimodal inputs.
- Fails with: `NotImplementedError: MRV2 PCP currently supports MLA models
  only.`[^vllm-pcp]
- Source: vLLM v0.26.1rc0, `v1/worker/gpu/pcp_manager.py`,
  `config/parallel.py` and `config/vllm.py`. Not run on torch_tpu; re-check
  on newer images.

## Caveats

- Image-scoped rules are evidence from one or two serves each, not from the
  code. A rule seen only on other images is a reason to read the error
  carefully, not to avoid the layout.
- The table has no expiry: a newer image may lift a structural rule. When a
  layout listed here starts, record the image so the rule can be scoped.

## Related

- How the plan chooses TP and DP, and what fewer TP ranks with more
  replicas did to throughput: `index.py get vllm-tp-dp-plan`.
- The head-count rules vLLM itself applies: `index.py get vllm-model-config`.

## Sources

- `package/scripts/common/oss_meshes.py` (`KERNEL_CONSTRAINTS`) and its tests.
- Serves on tpu7x 2x2x1 of 2026-10-13 to 2026-10-15, on the images named in
  each section; each failure signature is quoted from that serve's server
  log.

[^tt-topo]: [torch_tpu `_internal/utils/hardware.py`](https://github.com/google-pytorch/torch_tpu/blob/main/src/torch_tpu/_internal/utils/hardware.py). code-reference. Per-generation device topologies (v5e, v5p, v6e have 1 device; v7 starts at 2) and the `No TPU topology found for count` error; checked at commit 9b6ebfc24032. URL checked 2026-10-28.
[^vtt-pp]: [vllm-torchtpu `platforms/pp_validation.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/src/vllm_torchtpu/platforms/pp_validation.py). code-reference. Rejects pipeline parallelism with tensor parallelism inside a stage; checked at commit a20f4a9b14e1. URL checked 2026-10-28.
[^vtt-gdn]: [vllm-torchtpu `kernels/gdn/v3/config.py`](https://github.com/vllm-project/vllm-torchtpu/blob/main/src/vllm_torchtpu/kernels/gdn/v3/config.py). code-reference. `triangular_block_size` from the per-device value heads; checked at commit a20f4a9b14e1. URL checked 2026-10-28.
[^vllm-pcp]: [vLLM `v1/worker/gpu/pcp_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/pcp_manager.py). code-reference. Prefill context parallelism limits (MLA only, no MM inputs); checked at commit e96b490b1204. URL checked 2026-10-28.
