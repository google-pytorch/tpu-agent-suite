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
entry: vllm-torchtpu-attention-backend
keyword: custom-attention-backend
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-20
source_updated: not applicable
generator: authored
---

# `--attention-backend CUSTOM` on vllm-torchtpu: which kernel it selects

On the vllm-torchtpu plugin, `--attention-backend` is what picks the paged
attention kernel. `CUSTOM` selects the plugin's batched-RPA backend, which runs
the plugin's own experimental `batched_rpa` kernel. Without the flag the plugin
falls back to `FLASH_ATTN` and the non-batched RPA kernel, and on Gemma-4-31B-it
that kernel fails to compile. This is why tpu-model-onboarding's vLLM recipe always
passes `--attention-backend CUSTOM`.

## Applies when

- The engine is `vllm` with the `vllm_torchtpu` platform plugin.
- You are deciding whether a launch needs `--attention-backend CUSTOM`, or
  which source file a paged attention kernel in a trace comes from.
- A server started without the flag fails at compile time with
  `Ran out of memory in memory space vmem` in an `RPA...` `pallas_call`.

## The dispatch chain

Read from vllm-project/vllm-torchtpu at `9725110e` (2026-10-20, vLLM 0.31.0).

1. `--attention-backend CUSTOM` sets `attention_config.backend` to
   `AttentionBackendEnum.CUSTOM`. `TpuPlatform.get_attn_backend_cls` keeps
   `FLASH_ATTN`, `CUSTOM` and `FLASH_ATTN_MLA`, forces MLA models to
   `FLASH_ATTN_MLA`, and turns anything else into `FLASH_ATTN`
   ([`platforms/tpu_platform.py`](https://github.com/vllm-project/vllm-torchtpu/blob/9725110e14e01caf8997519320ce5c6e1f5077c5/src/vllm_torchtpu/platforms/tpu_platform.py#L565-L594)).
2. `@register_backend(AttentionBackendEnum.CUSTOM)` registers
   `PallasBatchedRPAAttentionBackend`, whose Impl is
   `PallasBatchedRPAAttentionBackendImpl`
   ([`layers/adapter/attention.py`](https://github.com/vllm-project/vllm-torchtpu/blob/9725110e14e01caf8997519320ce5c6e1f5077c5/src/vllm_torchtpu/layers/adapter/attention.py#L720-L742)).
3. That Impl sets `_kernel_entry = _pallas_rpa_kernel_batched`
   ([L1644-L1647](https://github.com/vllm-project/vllm-torchtpu/blob/9725110e14e01caf8997519320ce5c6e1f5077c5/src/vllm_torchtpu/layers/adapter/attention.py#L1644-L1647)), which passes
   `rpa_func=ragged_paged_attention_batched` into `attention()`
   ([L541-L546](https://github.com/vllm-project/vllm-torchtpu/blob/9725110e14e01caf8997519320ce5c6e1f5077c5/src/vllm_torchtpu/layers/adapter/attention.py#L541-L546)). `FLASH_ATTN`'s
   Impl keeps the default `_pallas_rpa_kernel_default`, the RPA v3 kernel. The
   backend's docstring still describes a `use_batched_rpa` flag; no code sets
   one.
4. `attention()` calls `sharded_ragged_paged_attention`, which runs
   `rpa_func`, except that head_dim 64 always takes the hd64 kernel
   ([`layers/core/attention_interface.py`](https://github.com/vllm-project/vllm-torchtpu/blob/9725110e14e01caf8997519320ce5c6e1f5077c5/src/vllm_torchtpu/layers/core/attention_interface.py#L353-L409)).
5. `ragged_paged_attention_batched` is the plugin's own
   `vllm_torchtpu/kernels/experimental/batched_rpa`
   ([L32, L66](https://github.com/vllm-project/vllm-torchtpu/blob/9725110e14e01caf8997519320ce5c6e1f5077c5/src/vllm_torchtpu/layers/core/attention_interface.py#L66)). Nothing in the
   plugin imports tpu-inference. The 2026-10-05 reading of the task image's
   plugin (vLLM 0.29.0, torch_tpu 0.1.1+release.2026.9.22) attributed the
   kernel to tpu-inference; tpu-inference's own dispatch is
   `index.py get attention-kernel-dispatch`.

`CUSTOM` with `VLLM_TPU_BLOCK_MAJOR_KV=1` raises `NotImplementedError`
([L1228-L1238](https://github.com/vllm-project/vllm-torchtpu/blob/9725110e14e01caf8997519320ce5c6e1f5077c5/src/vllm_torchtpu/layers/adapter/attention.py#L1228-L1238)).

Without the flag:

- The plugin logs `Cannot use AttentionBackendEnum.<X> backend on TPU. Setting
  to FLASH_ATTN.` and then `Using FLASH_ATTN backend.` `<X>` is whatever
  upstream vLLM chose. For Gemma-4 that is `TRITON_ATTN`, because its layers
  have two head sizes (256 and 512).
- `FLASH_ATTN` maps to `PallasAttentionBackend`, whose Impl runs the RPA v3
  kernel.

With the flag, the startup log says `Using CUSTOM backend.` once per worker.

## Tell which kernel ran

- **Startup log:** `Using CUSTOM backend.` or `Using FLASH_ATTN backend.`
- **Kernel names**, in a trace or a compile error: the kernel that ran out of
  VMEM without the flag (below) was
  `RPAm-p_128-bq_512_512-bkv_2048_512-sw_1024`. That is RPA v3's
  `RPA<case>-p_<page_size>-bq_...` form. The plugin's `batched_rpa` kernel is
  named `RPA<mode>-<kv_layout>-p<page_size>-b...`
  ([`batched_rpa/kernel.py`](https://github.com/vllm-project/vllm-torchtpu/blob/9725110e14e01caf8997519320ce5c6e1f5077c5/src/vllm_torchtpu/kernels/experimental/batched_rpa/kernel.py#L746-L750)).
- When you profile a server, take the kernel's source file from the launch
  flags, not from a file with a similar name.

## Evidence: Gemma-4-31B-it with and without the flag

tpu-model-onboarding run `20261005T185158Z-40a60af3d0af`, on a TPU VM:

- **Hardware and stack:** `tpu7x-4`, 8 devices; vLLM 0.29.0; libtpu 0.0.47.
- **Launch:** `google/gemma-4-31B-it`, TP 2 x DP 4, `--max-model-len 2048`.

The two launch scripts differ only in the flag.

| | No flag | `--attention-backend CUSTOM` |
| --- | --- | --- |
| Backend in the log | `FLASH_ATTN` (upstream picked `TRITON_ATTN`) | `CUSTOM` |
| KV cache | block size 128, 117,749 tokens per replica | same |
| Outcome | Compile fails: `Ran out of memory in memory space vmem. Used 71.34M of 63.94M vmem.` The largest allocation is a 36 MiB scratch operand of `RPAm-p_128-bq_512_512-bkv_2048_512-sw_1024/pallas_call`; register spill slots add 11.34 MiB | Serves. All three requests return; the one-word France prompt answers `Paris` |

The same session's earlier no-flag attempts failed for reasons unrelated to
the attention backend, so they are not evidence either way:

- a tier-2 compilation cache directory it had no permission to create;
- the 600 s engine-ready timeout;
- `PjRtClient is not initialized`;
- a `head_dim` error from an older transformers in a different venv.

## Caveats

- **One model fails without the flag.** A hybrid linear-attention model in the
  same session started and generated without the flag at TP 4 x DP 2. So the
  flag is not always needed just to start a server. It is still the backend
  the recipe was validated with.
- **No performance comparison.** No run compared throughput or output agreement
  between the two backends on a model that serves under both.
- **TPU VM, not the GKE image.** The comparison ran on a TPU VM with its own
  plugin checkout. The 2026-10-05 dispatch chain was read from the GKE task
  image; the chain above is from the public repository.

## Sources

- Run `20261005T185158Z-40a60af3d0af` (tpu-model-onboarding, 2026-10-05): the
  plugin-source introspection (`attn_introspect2.txt`), and the TPU-VM launch
  scripts, serve logs and responses for the two Gemma-4 runs (`vm6`, no flag;
  `vm7`, `CUSTOM`).
