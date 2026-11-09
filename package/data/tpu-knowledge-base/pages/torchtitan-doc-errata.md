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
entry: torchtitan-doc-errata
keyword: torchtitan-doc-errata
url: not applicable -- see Sources
authority: first-party-verified
retrieved: 2026-10-20
source_updated: not applicable
generator: authored
---

# torchtitan TPU docs that disagree with its source or recipes

Where google-pytorch/torchtitan's TPU docs under `torchtitan/experiments/tpu/`,
read at `a637209b` (2026-10-20), say something the code or the recipes at
the same commit do not. Go by the code. For everything else, the doc entries
stand (`index.py get torchtitan-tpu-optimization-guide`,
`torchtitan-tpu-recipes`, `torchtitan-trillium-recipes`).

## Applies when

- You are copying a launch command or a `--xla_tpu_scoped_vmem_limit_kib`
  value from a torchtitan TPU recipe.
- A recipe's CLI flag is rejected as unknown.

## Scoped VMEM values above the usable limit

The Ironwood `llama3-70b` recipes and the Trillium `llama3-8b` recipe pass
`--xla_tpu_scoped_vmem_limit_kib=131072`. That is above the usable limit
OpenXLA tables for TPU 7x (65,472 KiB) and for v6e (131,008 KiB). On TPU7x a
value above capacity raised no error and was clamped to the usable limit
(`index.py get pallas-core-mental-model`, libtpu 0.0.44), so those recipes
run at the cap, not at 128 MiB. v6e was not measured. The limits and how the
flag behaves are `index.py get vmem`.

The optimization guide says the Trillium recipes "converged on" 65536 KiB;
`trillium/llama3-8b/v6e-8` still uses 131072, and `trillium/qwen3-8b/v6e-8`
uses 65536.

## The scoped VMEM default is stated three ways

- The onboarding skill's `gotchas.md`: 16 MB on v5e, 64 MB on v6e / v7x.
- The optimization guide: "conservative", raise it to 65536 on v6e.
- Measured on TPU7x with libtpu 0.0.44 for a Pallas kernel with neither
  `vmem_limit_bytes` nor the flag: 32 MiB (`index.py get
  pallas-core-mental-model`).

They differ in libtpu version, and in whether they mean one kernel or the
whole program. Read the limit from your own compile log
(`Scoped allocation with size ... and limit ...`) before tuning against a
number.

## No `flux1` 4x8x8 recipe

The optimization guide lists `flux1` at 4x4x4 and 4x8x8 on v7x. Only
`recipes/ironwood/flux1/tpu7x-4x4x4` exists.

## Mamba3 recipe: removed training flags

The `mamba3-2.7b-256k` recipe
(`examples/mamba3/recipes/ironwood/mamba3-2.7b-256k/2x2x1/README.md`) passes
`--training.local_batch_size` and `--training.seq_len`. Upstream
`d771abb0` replaced them: `TrainingConfig` now has
`num_tokens_per_microbatch_per_dp_rank`, `num_tokens_per_train_step` and
`max_context_length`
([`config/configs.py`](https://github.com/google-pytorch/torchtitan/blob/a637209b3af6c4928f5f851ad9788e917b2e8598/torchtitan/config/configs.py#L36-L52)). The Trillium
`flux1/v6e-8` recipe notes the removal. For a batch of `B` sequences of
length `S` per DP rank, set `max_context_length=S` and
`num_tokens_per_microbatch_per_dp_rank=B*S`, as
`models/qwen3/config_registry.py` does.

## Sources

- google-pytorch/torchtitan at `a637209b3af6c4928f5f851ad9788e917b2e8598`:
  `torchtitan/experiments/tpu/torch-tpu-optimization-guide.md`, the recipe
  READMEs named above, `.claude/skills/tpu_model_onboarding/references/gotchas.md`,
  `torchtitan/config/configs.py`, `torchtitan/models/qwen3/config_registry.py`.
- OpenXLA E1001 usable limits (`index.py get xla-error-e1001`).
