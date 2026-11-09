---
scripts:
  - scripts/tpu-model-onboarding/run_benchmarks.py
references:
  - references/tpu-model-onboarding/remote-tpu-runner/ssh_verification.md
  - references/tpu-model-onboarding/remote-tpu-runner/hardware_profiling.md
  - references/tpu-model-onboarding/remote-tpu-runner/benchmark_execution.md
---
# RemoteTpuRunner Skill

This skill is designed to manage connectivity and automated hardware diagnostics
for remote TPU VM hosts, executing model benchmarks with optimized architecture
device flags tailored directly to the available accelerators.

## Artifact Root (`<artifacts>`)

`<artifacts> = {{runs}}/<run_id>/artifacts`, `<work> = <artifacts>/tpu-model-onboarding`.
Return `invalid_input` if `run_id` is missing. Write only to `<work>`, never to CWD or package files.

## Preconditions & Inputs

Before executing this skill, ensure you have:

-   **`run_benchmarks_script` (Required)**: Path to the orchestrator script
    (`{{script:tpu-model-onboarding/run_benchmarks.py}}`).
-   **`models_json_path` (Required)**: Path to the `models.json` file configuring
    the benchmark iterations (typically `<work>/models.json`).
-   **`benchmark_script` (Required)**: Path to the synthesized model execution
    logic script (typically `<work>/benchmark_tpu.py`).

--------------------------------------------------------------------------------

## Workflow

Perform the following stages sequentially to verify connection integrity,
profile systems, and invoke execution:

### Step 0: Critical Protocol Adherence

1.  Ensure that the project's rules are active to dynamically inject
    instructions on the fly. If not running in an environment that automatically
    injects local rules, load and follow instructions from associated AGENTS.md
    to ensure full conversation isolation.

2.  Enable planning and instruct tools/subagents to keep a track of workflow
    progress.

### Step 1: Compute Configuration Check

Confirm a compute profile is active with
`{{runtime}} config show --json --package {{package_tag}}`; do not run
`compute inspect`. See
`{{reference:tpu-model-onboarding/remote-tpu-runner/ssh_verification.md}}` for full instructions.

### Step 2: Accelerator Hardware Sizing

Size the accelerator without probing the target: the request's
`accelerator_type`, its per-chip specs from the knowledge base, and the chips
from that profile. See
`{{reference:tpu-model-onboarding/remote-tpu-runner/hardware_profiling.md}}`
for sizing instructions.

### Step 3: Adaptive Benchmark Execution

Select the target configuration scenario and run the benchmark command:

```bash
{{runtime}} compute run --code {{script:tpu-model-onboarding/run_benchmarks.py}} \
  --timeout 3600 \
  --input <work>/benchmark_tpu.py:benchmark_tpu.py \
  --input <models_json_path>:models.json \
  --output results.json:<work>/results.json \
  -- --models_json models.json --output_json results.json [--compile]
```

`--input` carries the synthesized `benchmark_tpu.py` and the models file to the
target; `--output` brings `results.json` directly back to `<work>/results.json`. See
`{{reference:tpu-model-onboarding/remote-tpu-runner/benchmark_execution.md}}` for
execution instructions.

### Step 4: Collect Benchmarking Log Payload

Capture the complete stdout/stderr payload outputs and store them into `<work>/benchmark.log`.
See `{{reference:tpu-model-onboarding/remote-tpu-runner/benchmark_execution.md}}#step-4-collect-benchmarking-log-payload`
for details.
