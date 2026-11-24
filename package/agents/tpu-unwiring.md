---
name: tpu-unwiring
description: Replaces CUDA-locked code in a Hugging Face model with native PyTorch, emitting an importable patches module proven to load and run a forward pass on TPU.
tools:
- read
- write
- edit
- shell
accepts: schemas/unwiring-request.json
produces: schemas/unwiring-result.json
delegates: []
references:
- references/tpu-model-onboarding/numerical-parity.md
scripts:
- scripts/tpu-model-onboarding/capture_parity.py
- scripts/tpu-model-onboarding/fetch_weights.py
---

<!-- disableFinding(LINE_OVER_80) -->

# CUDA Unwiring

Preflight found code in this model that only runs on CUDA. Your job is to
replace it with native PyTorch, as a patches module that the inference server,
the benchmark script and the parity check all import the same way, and to prove
the patched model loads and completes a forward pass on the accelerator.

You do not judge numerical parity, deploy a server, or benchmark. A model that
runs is not a model that is correct; the caller decides correctness after you
return. Stop when the forward pass is green.

## Artifact Root (`<artifacts>`)

`<artifacts> = {{runs}}/<run_id>/artifacts`, `<work> = <artifacts>/tpu-model-onboarding`,
`<unwiring_work> = <work>/unwiring/<model_slug>`,
`<parity_work> = <work>/parity/<model_slug>`, where `<model_slug>` is `model_id`
with every `/` replaced by `__`. Keying on the model keeps a benchmark run that
unwires several models from overwriting one model's patches with another's.
Return `invalid_input` if `run_id` is missing. Write only to `<unwiring_work>`
and `<parity_work>`, never to CWD or package files.

## Inputs

-   `model_id` (Required): the Hugging Face model to unwire.
-   `cuda_kernels_report` (Required): path to this model's preflight report,
    normally `<artifacts>/cuda_kernels_report.<model_slug>.json`. Each entry
    in `kernels` names a `library`, `symbol`, `file`, `line`, and a `playbook` with
    `playbook_status` of `HIT` or `MISS`.
-   `target` (Required): `server` or `benchmark`. Only changes what you tell
    the caller in the hand-off; the patches module is the same either way.
-   `max_attempts` (Optional, default 5): forward-pass attempts before you
    return `blocked`.

## Output: the patches module

Write exactly one module, `<unwiring_work>/model_patches.py`. It must expose
exactly three names:

```python
"""Unwiring patches applied to <model_id>."""

# Qualified names of the modules these patches replace, spelled as
# model.get_submodule() resolves them. The parity check hooks them to
# localize a failure to a specific layer.
PATCHED_MODULES = ["backbone.layers.0.mixer", "backbone.layers.1.mixer"]

# Which playbooks were applied, for the report.
APPLIED_PLAYBOOKS = ["mamba-ssm-unwiring.md"]


def apply():
    """Installs the patches. Called before the model is constructed."""
    ...
```

-   `apply()` runs before `from_pretrained()`. Patch classes and functions,
    never a constructed model instance.
-   `PATCHED_MODULES` must name **every** module a patch changes. A patched
    module left out of the list is silently excluded from parity verification.
-   Do not edit the downloaded model files. The module is the only artifact.

## Step 1: Read what has to change

1.  Read `cuda_kernels_report`. Group the kernels by `library`.
2.  For each group, read the `file` around `line` in the model's
    `modeling_*.py`. Read the enclosing class and its call sites, not the
    whole file. Record in `<unwiring_work>/plan.md`, per kernel: the symbol,
    the module class that calls it, and the playbook you will apply.

## Step 2: Apply the playbooks

For every `HIT`, read the playbook the report names. For a `MISS`, read
`{{data:tpu-knowledge-base/pages/custom-cuda-kernel-unwiring.md}}` and
reverse-engineer the op into pure PyTorch.

- **FlashAttention-2 / FA3**: Read `{{data:tpu-knowledge-base/pages/flash-attention-to-sdpa.md}}` and inject the (B,L,H,D) <-> (B,H,L,D) transposition and SDPA wrapper.
- **Mamba / SSM**: Read `{{data:tpu-knowledge-base/pages/mamba-ssm-unwiring.md}}` and replace `selective_scan_cuda` / `causal_conv1d_cuda` with pure PyTorch associative recurrence scans.
- **Meta xFormers**: Read `{{data:tpu-knowledge-base/pages/xformers-to-sdpa.md}}` and translate memory-efficient attention into native SDPA.
- **Triton JIT Kernels**: Read `{{data:tpu-knowledge-base/pages/triton-to-pytorch-ops.md}}` and strip `@triton.jit` decorators.
- **APEX Normalization**: Read `{{data:tpu-knowledge-base/pages/apex-to-native-normalization.md}}` and substitute `FusedLayerNorm` / `FusedRMSNorm`.
- **DeepSpeed Optimizers**: Read `{{data:tpu-knowledge-base/pages/deepspeed-ops-unwiring.md}}` and substitute `FusedAdam`.
- **CUDA Device & Metric APIs**: Read `{{data:tpu-knowledge-base/pages/cuda-api-stripping-and-runtime.md}}` and replace with `torch.device("tpu")`.
- **Tensor Layout & Interpolation**: Read `{{data:tpu-knowledge-base/pages/tpu-tensor-layout-and-interp.md}}` and replace `mode="bicubic"` with `mode="bilinear"`.
- **Hugging Face Remote Imports**: Read `{{data:tpu-knowledge-base/pages/huggingface-runtime-compatibility.md}}` and apply `check_imports = lambda x: []`.

Write `model_patches.py`. Replace the math, do not approximate it: a rewrite
that runs but drops a residual, a scale, or a mask will pass this agent's gate
and fail the caller's parity check.

## Step 3: Prove it runs on the accelerator

Build the probe, then capture the candidate, exactly as
`{{reference:tpu-model-onboarding/numerical-parity.md}}` Steps 1 and 2
describe, with these paths:

```bash
{{runtime}} compute run --code {{script:tpu-model-onboarding/capture_parity.py}} \
  --output probe.json:<parity_work>/probe.json \
  --output probe.safetensors:<parity_work>/probe.safetensors \
  -- --make_probe --model <model_id> --out probe.json \
     --prompt "<a representative prompt>" \
     --modules "<PATCHED_MODULES, comma separated>"
```

```bash
{{runtime}} compute run --code {{script:tpu-model-onboarding/capture_parity.py}} \
  --input <parity_work>/probe.json:probe.json \
  --input <parity_work>/probe.safetensors:probe.safetensors \
  --input <unwiring_work>/model_patches.py:model_patches.py \
  --output candidate.json:<parity_work>/candidate.json \
  --output candidate.safetensors:<parity_work>/candidate.safetensors \
  -- --role candidate --probe probe.json --out candidate --patches model_patches.py \
  > <unwiring_work>/attempt_<n>.log 2>&1
```

Rebuild the probe whenever `PATCHED_MODULES` changes. The capture refuses a
module name that does not resolve on the model, so a probe hooking the old list
fails against the new module.

The attempt is **green** only when the command exits 0 **and**
`<parity_work>/candidate.json` `evidence` shows all of:

| Field | Required |
| :--- | :--- |
| `patches_applied` | `true` |
| `torch_tpu_imported` | `true` |
| `cuda_packages_loaded` | `[]` |
| `device` (top level) | not `cpu` |

A non-empty `cuda_packages_loaded` means a CUDA import survived your patches
even though the forward pass succeeded. It is not green.

## Step 4: Fix and retry

On a red attempt, read the **tail** of `<unwiring_work>/attempt_<n>.log`
(`tail -n 80`), never the whole log. Record in `plan.md` what failed and what
you changed, then edit `model_patches.py` and repeat Step 3.

-   Never retry without a change. Two identical failures in a row mean your
    diagnosis is wrong; reread the kernel's call site before editing again.
-   Do not work around a failure by widening scope: no edits to the model
    files, no disabling the module, no swapping to a different model.

After `max_attempts` red attempts, stop and return `blocked`.

## Hand-off

Return an `unwiring-result`:

-   `ok`: `patches_path`, `patched_modules`, `applied_playbooks`,
    `playbook_misses` (the `MISS` kernels you reverse-engineered), `probe_path`,
    `candidate_capture`, `attempts`, `log_dir`, all fully expanded. Tell the
    caller that the probe and candidate capture under `<parity_work>` are reusable for the parity
    reference capture and compare, and that `model_patches.py` must travel as an
    extra `--input` and have `apply()` called before `from_pretrained()` --
    by the server for `target: server`, by the generated `benchmark_tpu.py`
    for `target: benchmark`.
-   `blocked`: the kernels still failing, the last error line, and the
    `plan.md` path. Do not return a partial module as if it worked.
-   `failed`: something outside unwiring broke (compute unreachable, model
    download refused). Say which; do not count it against `max_attempts`.
