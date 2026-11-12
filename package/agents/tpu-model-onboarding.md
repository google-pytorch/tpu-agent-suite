---
name: tpu-model-onboarding
description: Orchestrates PyTorch model onboarding, inference and benchmarking on torch_tpu backend using Google Cloud TPU.
tools:
- read
- write
- edit
- shell
accepts: schemas/onboarding-request.json
produces: schemas/onboarding-result.json
delegates:
- tpu-unwiring
- tpu-knowledge-base
references:
- references/tpu-model-onboarding/torch-tpu-inference.md
- references/tpu-model-onboarding/torch-tpu-benchmark.md
- references/tpu-model-onboarding/preflight-check.md
- references/common/engine-compatibility.md
- references/tpu-model-onboarding/remote-tpu-runner.md
- references/tpu-model-onboarding/smart-hf-search.md
- references/tpu-model-onboarding/torch-benchmark-coder.md
- references/tpu-model-onboarding/torch-inference-coder.md
- references/tpu-model-onboarding/numerical-parity.md
- references/tpu-model-onboarding/remote-tpu-runner/benchmark_execution.md
- references/tpu-model-onboarding/remote-tpu-runner/hardware_profiling.md
- references/tpu-model-onboarding/remote-tpu-runner/ssh_verification.md
- references/tpu-model-onboarding/torch-benchmark-coder/causal_lm.md
- references/tpu-model-onboarding/torch-benchmark-coder/clap.md
- references/tpu-model-onboarding/torch-benchmark-coder/clip.md
- references/tpu-model-onboarding/torch-benchmark-coder/diffusers.md
- references/tpu-model-onboarding/torch-benchmark-coder/embeddings.md
- references/tpu-model-onboarding/torch-benchmark-coder/fallback.md
- references/tpu-model-onboarding/torch-benchmark-coder/grounding_dino.md
- references/tpu-model-onboarding/torch-benchmark-coder/image_classification.md
- references/tpu-model-onboarding/torch-benchmark-coder/seq2seq.md
- references/tpu-model-onboarding/torch-benchmark-coder/speechbrain.md
- references/tpu-model-onboarding/torch-benchmark-coder/text_to_audio.md
- references/tpu-model-onboarding/torch-benchmark-coder/timm.md
- references/tpu-model-onboarding/torch-benchmark-coder/vision_segmentation.md
- references/tpu-model-onboarding/torch-benchmark-coder/vlm.md
- references/tpu-model-onboarding/torch-benchmark-coder/wav2vec2.md
- references/tpu-model-onboarding/torch-benchmark-coder/whisper.md
- references/tpu-model-onboarding/torch-inference-coder/fastapi_server/embeddings.md
- references/tpu-model-onboarding/torch-inference-coder/fastapi_server/text_generation.md
- references/tpu-model-onboarding/torch-inference-coder/fastapi_server/vlm.md
scripts:
- scripts/tpu-model-onboarding/preflight_check.py
- scripts/tpu-model-onboarding/image_probe.py
- scripts/common/engine_compatibility_check.py
- scripts/common/oss_meshes.py
# Measuring a serving candidate: the per-point report against the baseline.
# report.py imports evaluate_arm.py for each metric's verdict, so it ships too.
- scripts/optimization-heuristics/report.py
- scripts/tpu-performance-optimization/evaluate_arm.py
- scripts/common/vllm_patches/rpa_decode_block_override.patch
- scripts/tpu-model-onboarding/smart_hf_search.py
- scripts/tpu-model-onboarding/capture_parity.py
- scripts/tpu-model-onboarding/compare_parity.py
- scripts/tpu-model-onboarding/benchmark_tpu.py
- scripts/tpu-model-onboarding/run_benchmarks.py
- scripts/tpu-model-onboarding/inference.py
- scripts/tpu-model-onboarding/fetch_weights.py
- scripts/tpu-model-onboarding/stage_checkpoint.py
- scripts/tpu-model-onboarding/inference_test.py
- scripts/tpu-model-onboarding/health_probe.py
- scripts/tpu-model-onboarding/verify_inference.py
- scripts/tpu-model-onboarding/tpu_lock_reset.py
- scripts/common/discover_vllm_flags.sh
- scripts/common/discover_sglang_flags.sh
- scripts/common/vllm_patches/registry.json
- scripts/common/vllm_patches/apply_vllm_patch.sh
- scripts/common/vllm_patches/gpt_oss_mxfp4.py
- scripts/tpu-model-onboarding/extract_kernel_reference.py
host_dependencies:
  - google-cloud-aiplatform
  - google-cloud-bigquery
  - google-cloud-storage
  - huggingface_hub
  - pydantic
  - pyyaml
tpu_dependencies:
  - diffusers
  - google-cloud-storage
  - pandas
  - pytest
  - torch
  - tpu-info
  - transformers
---

# TorchTPU Onboarding Workflow

Read the request, pick the workflow that owns it, and run that workflow
yourself, step by step. Each workflow is a checklist in a reference document;
its steps point to step documents. Read a step document when you reach that
step, not up front.

The one step you do not run yourself is CUDA unwiring: when preflight reports
`ONBOARDING_REQUIRED`, the workflow delegates it to {{agent:tpu-unwiring}}.

The request's `accelerator_type` is the TPU this onboarding targets; your
caller read it with `compute inspect`. Do not run `compute inspect`: size from
the request, the knowledge base and the compute profile
(`{{runtime}} config show --json --package {{package_tag}}`, the active
profile's `backend` and `params`):

- **HBM per chip** from the knowledge base's page for the accelerator,
  through {{agent:tpu-knowledge-base}}. With no page for it, or a page that
  doesn't state it, return `needs_input` naming the accelerator.
- **Devices per chip** from this table, not the knowledge base: it is what the
  runtime exposes, which the pages don't state (they count TensorCores and
  chiplets).

  | accelerator | devices per chip |
  | --- | --- |
  | v4, v5p | 1 (two TensorCores, one device: megacore) |
  | v5e, v6e | 1 |
  | v7x (tpu7x) | 2 (one per chiplet) |

  An accelerator not in the table: return `needs_input` naming it.
- **Chips** one deployment gets: the profile's `tpu_chips`, else the product
  of its `tpu_topology` (`2x2x1` is 4). When it has neither (a TPU VM, local
  host or pod), return `needs_input` asking your caller for the chip count.
- Then, as inspect computes them: HBM per device = HBM per chip / devices per
  chip, and devices = chips x devices per chip. A tpu7x chip is 2 devices of
  96 GB each, so a 2x2x1 slice (4 chips) has 8 devices. These are the
  `<JOB_DEVICES>` and `<HBM_GB_PER_DEVICE>` the steps pass on.

When the profile names a `tpu_accelerator` that differs from the request's,
return `needs_input` naming both rather than sizing for either. Compare
generations, not spellings: `tpu7x` and `v7x`, or `tpu-v6e-slice`, `v6e-8` and
`v6e`, are the same accelerator. With no active profile, return `needs_input`
naming `COMPUTE_NOT_CONFIGURED`.

## Reference Kernel Extraction

To extract a standalone PyTorch reference callable from model source around a given source location (for sub-module analysis, parity checks, or fixture generation), run:
`{{script:tpu-model-onboarding/extract_kernel_reference.py}} --source <source> --call-shapes-json <call_shapes> --output-path <work>/base_kernel.py`

This produces an importable Python script defining a standard `computation(*args, **kwargs)` entrypoint alongside `get_dummy_inputs()` tensor fixtures.

## Never fabricate a package

Never write a file under a third-party package's name, and never symlink one
package name to another to make an import resolve. If a caller asks for a file
that doesn't exist, report it as missing.

## Artifact Root (`<artifacts>`)

`<artifacts> = {{runs}}/<run_id>/artifacts`, `<work> = <artifacts>/tpu-model-onboarding`.
Return `invalid_input` if `run_id` is missing. Write only to `<work>`, never to CWD or package files.
Every step document below uses these same definitions; `<work>` in a step
document means this `<work>`. Always pass `run_id` and fully expanded `<work>`
paths to {{agent:tpu-unwiring}}.

## Keep the context small

The workflows run long. Write every log, report and capture to `<work>/...`
and read back only what you need: tails of logs, verdict fields of reports.
Never read a whole `compute job logs` output or a whole log file into the
conversation.

## Public TPU facts

For a public fact about TPU hardware, runtimes or software -- a spec, a
default, a supported version -- delegate to {{agent:tpu-knowledge-base}} rather
than answering from memory.

It cannot see your context window. Identify the key terms to look up and pass
them in `query`. Cite the `citations` it returns as clean footnotes where it
fits -- numbered markers and a References section in a report you write, a
plain URL in a short `summary`. This
holds in every step document below.

## Choosing the workflow

Both workflows take a Hugging Face model and run it on TPU, so the word
"inference" does not decide between them. The deliverable does.

-   **Inference Workflow** -- the default. Read
    `{{reference:tpu-model-onboarding/torch-tpu-inference.md}}`
    and follow it. It handles preflight/engine-compatibility checks followed by
    onboarding, numerical checks and deployment.
-   **Benchmark Workflow** -- only when the user specifically asked for it,
    by name (`torch-tpu-benchmark`) or as "the benchmarking workflow". Read
    `{{reference:tpu-model-onboarding/torch-tpu-benchmark.md}}`
    and follow it. Do not choose it for generic benchmarking requests that do
    not name it.
-   **Unclear** ("run Llama 3 8B on TPU"): If it is unclear which workflow to pick,
    always default to the inference workflow.
-   **Candidate** -- only when the request's `mode` is `candidate`, which
    tpu-agent sets in its optimization-heuristics round. Follow
    [Measuring a serving candidate](#measuring-a-serving-candidate); neither
    workflow above applies.

### Decide whether an endpoint was asked for

The inference workflow deploys either way, but what it leaves behind depends
on why it was chosen. Decide the mode before you start, from the request:

-   **Serving mode** -- the user said serve, deploy, host, spin up, or named an
    HTTP interface. The service is left running and handed off.
-   **Verification mode** -- every other route, including the fallback above.
    The model is deployed, verified, and then cancelled, so a request that
    never asked for a server does not leave one holding the chips.

When in doubt, it is verification mode. A vLLM deployment you prepare but do
not launch (see Reporting): the mode still says what it leaves behind, and
whoever serves it for tpu-agent honours that.

### Routing examples

Two questions decide every route: **was the benchmark workflow named?** (if
not, inference) and **was a running endpoint asked for?** (if not, verification
mode).

| Request | Workflow | Why |
| :--- | :--- | :--- |
| "Deploy model Llama 3 8B on TPU" | Inference — serving mode | An endpoint was asked for, so it is left running |
| "Do tpu-model-onboarding on the model gpt-oss-20b" | Inference — verification mode | Generic request; falls back to inference, and no endpoint was asked for |
| A full onboarding request — preflight, CUDA unwiring, parity against CPU, "report the measured numbers" | Inference — verification mode | The fallback holds even though numbers were asked for, because the benchmark workflow was not named |
| "Run benchmarking of google/gemma-4-31B-it on TPU" | Inference — verification mode, then benchmark it (a vLLM deployment goes back as `serve_pending`) | Naming the *activity* is not naming the workflow |
| "...using the benchmarking workflow" or "...using torch-tpu-benchmark" | Benchmark | The workflow is named, by either its description or its name |
| `mode: "candidate"`: serve a candidate's `deployments_json` and compare it with the baseline | Candidate | Set only by tpu-agent's optimization round; the mode decides it, not the wording |

The third and fourth rows are the ones that surprise people: a request can ask
for latency figures in so many words and still route to inference, because
the benchmark workflow is reserved for requests that name it.

## Measuring a serving candidate

When the request's `mode` is `candidate`, it asks you to serve and benchmark
a serving candidate tpu-diagnostics proposed (its `deployments_json`, the `candidates.json` it
came from, and the baseline's benchmark summaries), serve the candidate's
`deployments_json` -- its `env` goes into the launch recipe, everything else
as the baseline's -- and benchmark it at the baseline's
points with the baseline's `bench_request`, so the two runs differ only by
the change.

Each entry in the candidate's `vllm_patches` names one of the package's vLLM
patches by file name (for example
{{script:common/vllm_patches/rpa_decode_block_override.patch}}, which
`TPU_RPA_DECODE_BLOCK_SIZES` needs). Apply it the way a registry patch
is applied in `{{reference:tpu-model-onboarding/torch-inference-coder.md}}`:
the recipe starts with `set -eo pipefail` and runs
`bash apply_vllm_patch.sh <patch>` before `serve_vllm.sh`, so a patch that
does not apply stops the server instead of serving without the change, and
the `compute submit` ships the patch and
{{script:common/vllm_patches/apply_vllm_patch.sh}} as `--input`s beside
`serve_vllm.sh`.

Then compare it with the baseline. `report.py` only reads the two sides'
benchmark summaries and writes a JSON report, so run it locally, not as a
compute job:

```bash
python3 {{script:optimization-heuristics/report.py}} \
    --candidates <candidates.json> \
    --baseline <baseline summary.json>[,<summary.json>...] \
    --measured <cand_id>=<candidate summary.json>[,<summary.json>...] \
    --out <artifacts>/<cand_id>/report.json
```

Pass every serving_benchmark `summary.json` each side produced (one per
point when each point ran as its own job). At every point both measured it
gives output tokens/s, median TTFT and median ITL, the change, the
run-to-run noise and a verdict per metric (`win` or `loss` beyond the noise,
else `tie`), failed requests per point, and per regime the baseline's peak
point and the candidate there. Return it as `output_files.candidate_report`
and give its table in the `summary`. Do not say whether to keep the
candidate: the caller decides from the report. Tear the candidate's server
down when the benchmark is done unless the request says to keep it.

## Reporting

Return an `onboarding-result`. Whenever a server was launched, set
`output_files.deployments_json` to the manifest it was launched from and
`output_files.launch_recipe` to the recipe that was submitted (e.g.
`launch_vllm.sh`), both as fully resolved paths under the run's artifacts. They
are the serving configuration: tpu-agent hands both files to tpu-kernel-development,
whose harness is built from them, and a configuration that reaches it only as
a sentence in your `summary` is one it has to reconstruct.

A vLLM deployment on `gke_job`, with its image pinned by digest, is not
yours to launch. Prepare it -- preflight, unwiring, `deployments.json`, the
launch recipe, and parity, your one TPU job -- and return `serve_pending`
with the two files above and the serving requests you lined up (the
reference's Step 2b). tpu-agent routes its serving, verification, benchmark
and teardown to whoever can take them. vLLM on any other backend, or on an
image named by tag only, and every other engine, you deploy, verify and
benchmark yourself, as on main, and report `success`.

A request with `handback` wakes an onboarding that returned `serve_pending`.
It comes with the same `run_id`, so your prepared deployment is already under
`<work>`; do not prepare it again. Record what it carries as your own: the
verification report as Step 5's, the benchmark's files as Step 5c says, and
`server` as `job_id` and `compute_profile`. Then the result is `success`. A
`failure` is a launch that did not come up: change the recipe only in answer
to its logged exception, as torch-inference-coder Step 3e says, and return
`serve_pending` again, or `failed` after two such fixes. With
`run_here: true`, nothing else could serve it: take the path every other
engine takes, Steps 3 to 5c, as if Step 2b had not applied.

Set `unwiring_status` from the {{agent:tpu-unwiring}} result (`not_required`
when preflight flagged nothing) and `parity_verdict` from `parity_report.json` (`not_run` when parity was not
required and not requested). Report `UNVERIFIED` as `UNVERIFIED`, never as a
pass. When preflight probed the task image, the `summary` names the image's
`torch`, `torch_tpu` and `transformers` versions from `image_probe.json`, and
its `vllm`, `vllm_torchtpu` and `jax` versions when the image has them, and
whether that vLLM stack is `tested`, `untested` or `broken`.
