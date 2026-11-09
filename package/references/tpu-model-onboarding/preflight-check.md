---
scripts:
  - scripts/tpu-model-onboarding/preflight_check.py
  - scripts/tpu-model-onboarding/image_probe.py
  - scripts/tpu-model-onboarding/fetch_weights.py
  - scripts/tpu-model-onboarding/onboarding_preflight.py
  - scripts/tpu-model-onboarding/chipless_tpu.py
  - scripts/common/engine_compatibility_check.py
  - scripts/common/oss_meshes.py
  - scripts/common/discover_vllm_flags.sh
  - scripts/common/discover_sglang_flags.sh
  - scripts/common/vllm_patches/registry.json
  - scripts/common/vllm_patches/apply_vllm_patch.sh
  - scripts/common/vllm_patches/gpt_oss_mxfp4.py
---
# Pre-flight Compatibility Check Skill

This skill handles the local preflight validation of a Hugging Face model to
verify its compatibility with `torch_tpu` (on TPU target compute). This helps
prevent spending resource/hardware budget on incompatible architectures or
running into runtime crashes.

The core validator script is located at
`{{script:tpu-model-onboarding/preflight_check.py}}`.

## Artifact Root (`<artifacts>`)

`<artifacts> = {{runs}}/<run_id>/artifacts`.
Return `invalid_input` if `run_id` is missing. Write only to `<artifacts>`, never to CWD or package files.
`<inference_work>` is torch-tpu-inference's `<work>/inference`, under
`<artifacts>`; only a vLLM onboarding's Step 3 writes there.

## CUDA Kernels Report (`<cuda_report>`)

`<cuda_report> = <artifacts>/cuda_kernels_report.<model_slug>.json`, where
`<model_slug>` is `model_id` with `/` replaced by `--`
(`openai/gpt-oss-20b` → `openai--gpt-oss-20b`; a local checkpoint directory
uses its base name). The check writes it on every run, with `kernels: []`
when the scan is clean, so each model has its own: one shared file would let a
later candidate overwrite the report an earlier one's unwiring reads.

## Workflow

Before generating benchmark scripts or executing remote TPU/CPU runs, you
**MUST** validate the compatibility of **every single candidate model** using
this skill. **Never skip or bypass running the preflight check for any
candidate model**, even if it belongs to a standard library or architecture.

Run the steps in order. Steps 1 and 2 run on the host and need only
`huggingface_hub`, so they settle the cheap questions before anything queues
for a chip.

### Step 1: Resolve the Model Name

For a model named in the request, resolve the name to a Hub repository first,
unless it is a local checkpoint directory or a `gs://` checkpoint folder. Do this for an `org/name` id too: an
id that exists comes back `exact`. Candidates found by `smart-hf-search` are
already exact ids and skip this step.

```bash
python3 {{script:tpu-model-onboarding/preflight_check.py}} --resolve="{model_id}" > <artifacts>/model_resolution.json
```

It prints one JSON object and runs no check. Act on its `verdict`:

| `verdict` | Meaning | Action |
| :--- | :--- | :--- |
| `exact`, `single_match` | `repo_id` is the model. | Use `repo_id` as `model_id` in every later step. If it differs from the name asked for, say so in the summary: "`gpt oss 20b` resolved to `openai/gpt-oss-20b`". |
| `ambiguous` | Several repositories fit. `candidates` are ranked; the first is the suggested default. | Return `needs_input`. In `summary`, ask "Did you mean [<repo_id>](<url>)?" for the first candidate, list the others as links, and say how they differ (`variant` such as base vs instruct, `quantized`). Do not pick one. |
| `not_found` | Nothing on the Hub matches. | Return `needs_input` asking for the exact Hub id. Never resolve a name from memory. |
| `hub_error` | The Hub answered with an HTTP error (`http_status`: 429, 5xx). | It is transient, not `offline`: run the same command once more after a minute. If it fails again, return `failed` with `http_status` in `error`. |
| `offline` | The Hub could not be reached from the host. | Do not resolve from memory. If `model_id` is already an `org/name` id, continue with it: Step 2's access check reports whether the Hub is reachable. Otherwise return `needs_input` asking for the exact Hub id, and say the Hub was unreachable. |

### Step 2: Hub-side Checks (host)

```bash
python3 {{script:tpu-model-onboarding/preflight_check.py}} --model={model_id} --hub_only --allow_gpu_imports --cuda_report_json=<cuda_report> --mode={benchmark|inference}
```

`--hub_only` runs access and gating, weight format, library tier, the
quantization rule (read from `config.json`) and the custom-code import scan,
and ends with `PRE-FLIGHT HUB CHECKS PASSED`. It does not need `transformers`,
so it runs on a host without it. Act on its failures under the
Rules below before going further. A model it rejects -- a failed access,
format, tier or quantization check, so the run does not end with `PASSED` --
stops preflight here: return `failed` without Steps 3 and 4, so nothing
queues for a chip on a model the host already ruled out. Flagged imports are
not a rejection: they warn, or mark the model `ONBOARDING_REQUIRED`, and
preflight goes on.

**A `gs://` checkpoint folder** goes through the same command, without the Hub:
access, gating and library tier do not apply and are reported as skipped; the
format check reads the folder's file names; and the quantization rule and the
custom-code scan run on a copy of its small files (config, tokenizer, code; no
weights) that {{script:tpu-model-onboarding/fetch_weights.py}} makes beside the
script, with the host's own credentials. A `Could not read gs://...: no
permission to read` failure is the user's to fix (an identity that can read the
bucket); do not fall back to a Hub id.

### Step 3: Check in the Task Image

Every compute job takes a whole TPU node and queues for it on its own, so the
checks that need the task image run as **one** job: the image probe, the full
check (Step 4), and, when this onboarding will serve on vLLM, the vLLM engine
compatibility check and the vLLM flag discovery with it. Run it once per model
that reaches this step.

None of them uses a chip, so on `gke_job` the job runs on a CPU node: add
`--cpu-requests cpu=2,memory=8Gi` and
`--input {{script:tpu-model-onboarding/chipless_tpu.py}}:chipless_tpu.py`
before `--`, and skip the gate below. If the CPU node cannot stand in for the
TPU, compute runs the job on a TPU node itself (`details.cpu_fallback` says
why).

**Here, when this host has the software.** If this machine's own Python
already has torch_tpu and vLLM (`python3 -c 'import torch_tpu, vllm'` exits
0), run the check here instead of as a job, with no node and no queue.
Put the script and every file the job form below names with `--input` in one
directory under `<artifacts>`, and run it there with the same arguments and
the TPU generation of the request's `accelerator_type` (`v5e`, `v5p`, `v6e` or
`tpu7x`; `v7x` and `tpu-v6e-slice` name the same generations):

```bash
mkdir -p <artifacts>/preflight_run && cd <artifacts>/preflight_run
cp {{script:tpu-model-onboarding/onboarding_preflight.py}} \
   {{script:tpu-model-onboarding/chipless_tpu.py}} <the job form's --input files> .
TPU_AGENT_SUITE_CHIPLESS_TPU=<generation> timeout 540 \
  python3 onboarding_preflight.py <the arguments after -->
```

Its outputs land in that directory: move each to where the job form's
`--output` sends it. The check read this environment's versions, not the task
image's: say which in the report. If it exits `75` (the chipless self-test
failed), run the job form instead.

**Free chips** (other backends). The job takes a whole TPU node; capacity is
booked through `tpu_slots.py` (torch-tpu-inference's Step 0.1b), not read
here. Do not run `compute inspect` for free chips or poll for a slot:
dispatch. A profile with a `queue` holds the run's place in line; without one,
a full target shows as `never_started` below.

```bash
{{runtime}} compute run --code {{script:tpu-model-onboarding/onboarding_preflight.py}} \
  --timeout 540 \
  --input {{script:tpu-model-onboarding/image_probe.py}}:image_probe.py \
  --input {{script:tpu-model-onboarding/preflight_check.py}}:preflight_check.py \
  --input {{script:common/vllm_patches/registry.json}}:registry.json \
  --input {{script:common/vllm_patches/gpt_oss_mxfp4.py}}:gpt_oss_mxfp4.py \
  --output preflight.json:<artifacts>/preflight.json \
  --output image_probe.json:<artifacts>/image_probe.json \
  --output full_check.log:<artifacts>/full_check.log \
  -- --model <MODEL_ID> --mode {benchmark|inference}
```

For a `gs://` model, add
`--input {{script:tpu-model-onboarding/fetch_weights.py}}:fetch_weights.py`
before `--`: the checks copy the folder's small files with it.

For a vLLM onboarding (`--mode=inference` with an engine of `vllm`, or none
given for an LLM), also add, before `--`:

```bash
  --input {{script:common/engine_compatibility_check.py}}:engine_compatibility_check.py \
  --input {{script:common/oss_meshes.py}}:oss_meshes.py \
  --input {{script:common/discover_vllm_flags.sh}}:discover_vllm_flags.sh \
  [--input {{script:common/vllm_patches/apply_vllm_patch.sh}}:apply_vllm_patch.sh \
   --input {{script:common/vllm_patches/gpt_oss_mxfp4.py}}:gpt_oss_mxfp4.py] \
  --output tp_plan.json:<inference_work>/tp_plan.json \
  --output vllm_flags.txt:<inference_work>/vllm_flags.txt \
```

For an SGLang onboarding (`--mode=inference` with an engine of `sglang`), add,
before `--`:

```bash
  --input {{script:common/engine_compatibility_check.py}}:engine_compatibility_check.py \
  --input {{script:common/discover_sglang_flags.sh}}:discover_sglang_flags.sh \
  --output tp_plan.json:<inference_work>/tp_plan.json \
  --output sglang_flags.txt:<inference_work>/sglang_flags.txt \
```

and, after `--`, the compatibility check's arguments as the engine
compatibility check describes them (`engine-compatibility.md`, read beside
this one): `--engine sglang` when onboarding on SGLang, `--devices <JOB_DEVICES>
--hbm-gb-per-device <HBM_GB_PER_DEVICE>` as Step 0.1 sized them, `--tp <N>`
only when the user stated one, `--vllm-patch gpt_oss_mxfp4.py` for a model in
the patch registry, and always `--points <ISL:OSL:C,...>` (the request's target
points), which the check needs to check the KV cache at those points, to pick
the layout for that traffic, and to size `--max-num-seqs` so the server holds
them. For one measurement, the point is the one it measures (`1024:128:1` unless
the request names another). For a sweep that doubles concurrency until
throughput stops rising, the points are each ISL/OSL pair at the concurrencies
the sweep can reach, up to the request's limit or 256 when it names none: that
is the traffic the sweep's peak is measured at. Such a sweep always reaches 128
users, so with a long-prompt pair (input at least 4x output) the check halves TP
for the whole sweep, low-concurrency points included, where the halved layout is
slower; the plan's `traffic_rule` says so. Add `--objective latency` when the
request says time to first token or latency matters more than throughput; leave
it out otherwise.

The job runs every step even when one fails, and writes every file it is
given an `--output` for. `preflight.json` holds each step's `exit_code`
(`null`: not run) and, for one that failed, the `tail` of its output. Judge
each step by its own entry, in order: the probe below, the full check in
Step 4, then `tp_plan.json` as `engine-compatibility.md` says, then
`vllm_flags.txt` (or `sglang_flags.txt`) in torch-inference-coder's Task 1.

`--timeout 540` keeps the queue wait under a 10-minute shell limit, so on a
full cluster the run reports `never_started` itself rather than being killed
by your shell with nothing said. Give the shell at least 20 minutes anyway:
the budget applies to the queue and the run separately.

`never_started` in the result's details means it timed out before any node
took it and has been deleted. Do not resubmit or `job wait`: return
`needs_input`, quoting the reason from `stderr`: the target is full, not
broken, and the user decides whether to wait, free a job, or change the
compute profile.
If your shell killed the run first, `compute` prints one line saying the
same; treat it the same way.

Read `<artifacts>/image_probe.json`. With an `error` key, the probe itself
failed: return `failed` with `preflight.json`'s `image_probe.tail` in `error`.
Otherwise:

-   `torch_tpu_present: false`: the image cannot run `torch_tpu`. Return
    `failed`, listing the image's `distributions` in `error`.
-   `device.status: "fail"`: the TPU runtime did not initialise. Return
    `failed` with `device.detail` in `error`. `skipped` means the job ran on
    a CPU node: the server's start is the first use of a chip.
-   `distributions`: state the `torch`, `torch_tpu` and `transformers`
    versions in the `summary` of the `onboarding-result` you return, whether
    Step 4 then passes or fails, as in "Task image: torch 2.11.0+cpu,
    torch_tpu 0.1.1.dev20260714090201, transformers 5.8.0." Add the `vllm`,
    `vllm_torchtpu` and `jax` versions too, each one the image has: serving
    runs on all three, and a `vllm_torchtpu` older than the image's `jax` can
    call a jax API that no longer exists. For `--mode=inference`, name any of
    the three the image lacks as not installed. A model Step 2 rejected never
    reaches this step, and its summary has no versions.
-   `dependencies.vllm_stack.verdict`: whether the image's vLLM stack is one
    a published image was tested with.
    -   `broken`: the stack is known not to serve. For `--mode=inference`,
        return `failed`, with the `vllm` and `jax` versions and
        `vllm_stack.problems` in `error`, and say to run on the installer's
        default image, which has a tested stack. For `--mode=benchmark`,
        nothing runs on vLLM: state it in the summary as a warning and go on.
    -   `untested`: no published image was tested with this stack. State that
        in the summary as a warning, with `vllm_stack.problems`, and go on.
    -   `tested`: state the `vllm_stack.image` it matches in the summary.
    -   `absent`: the image has no vLLM; nothing more to say.
-   `dependencies.pip_check.stack_conflicts`: requirements the image's own
    onboarding packages declare and the image does not meet. List them in the
    summary as warnings; they do not fail preflight on their own, since a TPU
    image can carry another torch than vLLM pins, on purpose.
    `pip_check.status: "unavailable"` means the image has no `pip`; say
    nothing about it.
-   `preflight_runnable`: whether Step 4 can run in the image.
-   `custom_fastapi_runnable`: whether the image can serve on the
    `custom_fastapi` engine, which imports `fastapi`, `pydantic` and
    `uvicorn`. When it is `false`, name the ones `modules` shows missing:
    -   For `--mode=inference` with `custom_fastapi` as the engine (the user
        asked for it, or the model is one vLLM cannot serve), return `failed`
        with the missing modules in `error`, and say to install them in the
        task image (`requirements-tpu.txt` lists them). Do not install them on
        the host: the image is what the server runs in.
    -   For `--mode=inference` on `vllm`, state them in the summary as a
        warning: a fallback to `custom_fastapi` would fail on them.
    -   For `--mode=benchmark`, nothing serves; say nothing about it.
-   `vllm_platform.registered`: whether vLLM in this image will serve on the
    TorchTPU plugin. For `--mode=inference`, state it in the summary, and when
    it is `false`, list `vllm_platform.problems`: vLLM would then serve on its
    own TPU platform, or not at all, rather than on TorchTPU.
-   `vllm_patches`: closest-version patch selection and dry-run results from
    {{script:common/vllm_patches/registry.json}} against the image's `vllm` and
    `vllm_torchtpu`. For a `model_id` in `vllm_patches`:
    -   `version_match` and `candidates` list the closest validated version(s)
        to try (`exact`; `between`: closest `before` and `after`; `newer`: only
        the last validated `before` version; `older`: only the first `after`
        version), with `validated_vllm`, `position`, `patch`, `serving`,
        `applies`, `status`, and `failures`.
    -   For `--mode=inference`, do not stop early solely because `vllm_stack`
        is `untested` when `action` is `"try_candidates"`. When `action` is
        `"needs_user_confirmation"` (`applicable_candidates == 0`), finish Step
        4 (`preflight_check.py`) and then stop before
        `engine-compatibility` or `torch-inference-coder`: return
        `needs_input` with the installed `vllm` version, closest candidate(s)
        checked, and `failures`, waiting for user confirmation before spending
        compute cycles fixing the patch.
-   `device.loaded_by`: `autoload` when torch_tpu loaded with `torch`, as a
    workload relies on, or `import` when the probe had to import it.

### Step 4: Full Check

The config, processor and vocabulary checks load the model's files with
`transformers`, so the full check ran in the image, in Step 3's job. Its
output is in `<artifacts>/full_check.log` and its exit code is
`preflight.json`'s `full_check.exit_code`; act on them under the Rules below
as you did on Step 2's. It writes no CUDA kernels report: `<cuda_report>`
from Step 2 is the report.

If it stopped with `transformers is not installed here` (the probe's
`preflight_runnable: false`), the image cannot run it: return `failed`
naming the `modules` the image lacks. That is not a model failure. Do not
install `transformers` on the host to run it there instead: the image is what
the workload will use.


## Rules

When running the tool, monitor the stdout and exit code. Take the following
actions based on results:

### A. Access & Gating Check

-   If the check flags a model as requiring gated access (e.g., raising a
    `GatedRepoError` or missing valid access token), and the credentials are not
    provided/valid, the orchestrator **must immediately skip** this model,
    evaluate the next candidate in the search buffer (discovered via
    `smart-hf-search`), and run the local `preflight-check` on it to verify
    compatibility.

### B. Format & Library Tier Validation

The `preflight_check` tool classifies the model's Hugging Face library into one
of three tiers based on metadata and file extensions:

-   **Tier None (Incompatible / Unsupported)**:
    -   *Identification*: Detected via specific `library_name` tags
        (`tensorflow`, `jax`, `onnx`, `openvino`, `coreml`, `paddlepaddle`,
        `tflite`, `gguf`, `wav2letter`, `scikit-learn`) or if the repo files
        contain only non-PyTorch binaries (e.g. only `.gguf` or `.tflite`,
        without any PyTorch/Safetensors weights).
    -   *Action/Utility*: The check fails immediately and you **must abort** the
        onboarding. These models do not support PyTorch / TPU device
        mapping.
-   **Tier B (Medium/Low Priority)**:
    -   *Identification*: Detected via NLP pipeline packaging or agentic wrapper
        framework tags (`spacy`, `stanza`, `flair`, `allennlp`,
        `simpletransformers`, `bertopic`, `haystack`, `ml-agents`).
    -   *Action/Utility*: The check warns that the library is compatible but
        runs with medium/low benefit due to CPU-bound overhead or high-level
        pipeline scaffolding.
-   **Tier A (High Benefit - Target)**:
    -   *Identification*: Standard model libraries (e.g. `transformers`,
        `diffusers`, `timm`, `sentence-transformers`, `peft`, `trl`) not
        classified in Tier None or Tier B.
    -   *Action/Utility*: Fully supported and designated as high-priority
        targets for TPU VM acceleration.

### C. Configuration & Processor Check

-   **Quantization**: The rule follows `--mode`, because the two workflows load
    a quantized checkpoint differently. `--hub_only` applies it too.
    -   `inference` serves through the vLLM TorchTPU plugin, which implements
        `fp8`, `mxfp4`, `compressed-tensors`, NVFP4 (`modelopt` with an FP4
        `quant_algo`) and DeepSeek-V4's fp8; any other method, `awq` included,
        fails.
    -   `benchmark` loads through transformers, which needs GPU kernels for
        `awq`, `gptq`, `bitsandbytes`, `eetq`, `fbgemm_fp8`, `fp8`, `vptq` and
        `aqlm`; those fail. Any other method passes with "Verify TPU support":
        nothing on TPU implements it, so it may not run, or run slowly.
    -   `mxfp4` passes in both; for `benchmark`, transformers dequantizes it to
        bf16 on a TPU.
-   **Processor / Tokenizer**: The check verifies that the tokenizer or
    processor loads successfully. If it is non-instantiable due to mismatching
    config mapping, the check fails.
-   **Vocabulary Check**: Validates that the tokenizer vocabulary size does not
    exceed the model's embedding layers config parameters (to prevent index
    errors).
-   **Specialized Models**: If the model is tagged, named, or configured with
    specialized keywords (such as protein folding, document layout, table
    structures, GNNs), a preflight warning is raised indicating custom wrapper
    handling is required.

### D. Incompatible Imports Scan in Custom Code

-   For custom models (`trust_remote_code=True`), the script scans the
    downloaded python codebase.
-   If it flags forbidden imports such as:
    -   `triton`, `flash_attn`, `apex`, `deepspeed`, `xformers`, `mamba_ssm`:
        Since `--allow_gpu_imports` is used by default, the check will print a
        warning but pass. You **may proceed** with the onboarding, but you should
        be aware that TPU compilation or execution might fail if these code
        paths are triggered on the remote VM.

## Tool Options

-   `--model`: Target Hugging Face model ID (e.g.
    `meta-llama/Llama-3.1-8B-Instruct`) or local checkpoint directory.
-   `--resolve`: Loose model name to resolve (Step 1); prints the JSON verdict
    and runs no check. `--resolve_limit` caps the candidates (default 3).
-   `--hub_only`: Run only the checks that need `huggingface_hub` (Step 2).
-   `--force`: Bypass all preflight checking errors (marked as `[FORCE
    BYPASS]`).
-   Authentication uses `HF_TOKEN` or the locally saved Hugging Face login.

`image_probe.py` takes `--touch_device` (allocate on the TPU; needs a chip)
and `--output` (also write the report to this path).

## Onboarding-Required Workflow (CUDA Unwiring)

When scanning custom code (`modeling_*.py`), if known GPU-locked libraries (`flash_attn`, `mamba_ssm`, `triton`, `apex`, `deepspeed`, `xformers`) are detected, `preflight_check.py` matches them against the playbooks in `{{data:tpu-knowledge-base/pages}}/`, each of which lists the imports it removes under `target_imports`.

Instead of failing, the model is flagged as `[STATUS: ONBOARDING_REQUIRED]` with the target playbook link, allowing the orchestrator, `torch_benchmark_coder`, and `torch_inference_coder` to apply the unwiring recipe during script generation.
