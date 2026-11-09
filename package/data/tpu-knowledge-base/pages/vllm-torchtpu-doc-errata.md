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
entry: vllm-torchtpu-doc-errata
keyword: vllm-torchtpu-doc-errata
url: not applicable -- see Sources
authority: first-party-verified
retrieved: 2026-10-20
source_updated: not applicable
generator: authored
---

# vllm-torchtpu docs that disagree with its source

Where vllm-torchtpu's own docs, read at `9725110e` (2026-10-20), say
something its source at the same commit does not do. Each item gives the doc's
claim, what the code does, and the line that shows it. Go by the source. For
everything else, the doc entries stand (`index.py get
vllm-torchtpu-disaggregated-serving`, `vllm-torchtpu-env-vars`,
`vllm-torchtpu-supported-models-upstream`).

## Applies when

- You are setting a variable or config value below from the docs and it is
  rejected, or does nothing.
- You are deciding which attention kernel `--attention-backend CUSTOM` runs.

## `TPU_KV_RESHARD_TRANSPORT`: `zmq` or `raiden`, not `tcp`

`docs/reference/env_vars.md` and `docs/user_guide/disaggregated_serving.md`
give `zmq` or `tcp`. `envs.py` accepts `zmq` (default) and `raiden`,
which selects the fail-closed controller path for PCP-to-DP pool resharding
([`envs.py`](https://github.com/vllm-project/vllm-torchtpu/blob/9725110e14e01caf8997519320ce5c6e1f5077c5/src/vllm_torchtpu/envs.py#L553-L557)). `tcp` is not a
choice.

## `TPU_RAIDEN_RESHARD_IMPL`: `controller` or `store`, not `host`

The same two docs give `controller` or `host`. `envs.py` accepts
`controller` (default, the external sidecar controller) and `store`, which
hosts the reshard service in each engine's rank-0 worker through an
in-process `KVCacheStore`
([`envs.py`](https://github.com/vllm-project/vllm-torchtpu/blob/9725110e14e01caf8997519320ce5c6e1f5077c5/src/vllm_torchtpu/envs.py#L572-L580)). The profiling doc
`docs/developers_guide/profiling/disaggregated.md` uses the right values.

## The Raiden connector's module path

`disaggregated_serving.md`'s `--kv-transfer-config` examples use
`"kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.raiden.raiden_connector"`.
There is no `raiden_connector.py` in `distributed/kv_transfer/raiden/`. The
platform registers `TPURaidenConnector` from
`vllm_torchtpu.distributed.kv_transfer.tpu_connector`
([`platforms/tpu_platform.py`](https://github.com/vllm-project/vllm-torchtpu/blob/9725110e14e01caf8997519320ce5c6e1f5077c5/src/vllm_torchtpu/platforms/tpu_platform.py#L365)),
so name the connector `TPURaidenConnector` and leave the module path out, or
give that module.

## `VLLM_XLA_CACHE_PATH` is not read

The README, `quickstart.md`, `kubernetes.md` and `env_vars.md` say to set
`VLLM_XLA_CACHE_PATH` to keep compiled artifacts across restarts, and point
to `compilation_cache.md`, which does not mention it. Nothing under the
plugin's `src/` reads it; vLLM defines it in `envs.py`, and on vLLM main
nothing else reads it. The plugin's tests only check that it is not
overwritten. The cache that
persists across restarts is torch_tpu's: set
`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`, which the plugin mirrors and pairs
with the tier-2 cache before libtpu loads
([`env_override.py`](https://github.com/vllm-project/vllm-torchtpu/blob/9725110e14e01caf8997519320ce5c6e1f5077c5/src/vllm_torchtpu/env_override.py#L25-L37);
`index.py get vllm-torchtpu-env-override`, `vllm-torchtpu-compilation-cache`).

## `--attention-backend CUSTOM` runs batched RPA, not RPA v3

`docs/user_guide/supported_models.md` lists "Ragged Paged Attention (RPA v3)"
as the default MHA / GQA backend, "`--attention-backend CUSTOM` in the
recipes". In source, `CUSTOM` runs the plugin's `batched_rpa` kernel and RPA
v3 is what `FLASH_ATTN`, the fallback, runs. The chain is
`index.py get vllm-torchtpu-attention-backend`.

## Speculative decoding: six methods

`docs/user_guide/speculative_decoding.md` says five methods.
`supported_models.md` lists `eagle3`, `mtp`, `dflash`, `dspark`,
`ngram` and `suffix`, with DSpark validated on Qwen3-4B. Not checked
against the source.

## Sources

- vllm-project/vllm-torchtpu at `9725110e14e01caf8997519320ce5c6e1f5077c5`:
  the docs named above under [`docs/`](https://github.com/vllm-project/vllm-torchtpu/blob/main/docs) and the source lines linked in
  each section; `grep -r VLLM_XLA_CACHE_PATH src/` finds nothing.
- vllm-project/vllm: `vllm/envs.py` defines `VLLM_XLA_CACHE_PATH` at v0.31.0,
  the version the plugin pins; a code search of main finds it only there, in
  `docs/usage/security.md` and in a test.
