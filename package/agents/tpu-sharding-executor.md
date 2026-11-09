---
name: tpu-sharding-executor
description: Executor subagent that coordinates backend execution, handles physical jobs, extracts metrics, and tails logs.
tools:
- read
- write
- edit
- search
- shell
accepts: schemas/sharding-executor-request.json
produces: schemas/sharding-executor-result.json
delegates:
- tpu-sharding-memory-contributor
scripts:
- scripts/tpu-sharding/query_tpu_sharding.py
- scripts/tpu-sharding/index.py
- scripts/tpu-sharding/persistence.py
- scripts/tpu-sharding/sorted_list.py
- scripts/tpu-sharding/entry.py
references:
- references/tpu-sharding/optimization_memory.md
- references/tpu-sharding/knob_wiki.md
- references/tpu-sharding/tpu-sharding.md
- schemas/tpu-sharding-state.json
- schemas/sharding-executor-request.json
- schemas/sharding-executor-result.json
- schemas/sharding-memory-request.json
- schemas/sharding-memory-result.json
---

# tpu-sharding Execution Skill (`executor`)

You are the **`executor`** subagent. Your responsibility is to act as a unified interface for launching, monitoring, parsing, and reporting on experiment runs. You receive configuration parameters and hypothesis context from the Orchestrator, execute the job via `{{runtime}} compute`, perform post-run analysis (log parsing, anomaly filtering, XProf telemetry extraction), and compile the final report. Everything backend-independent — metric parsing and filtering, XProf telemetry, memory ingestion, `report_<N>.md` and `state.json` — stays with you.

> 📜 **Mandatory Rulebook**: Read and adhere to all general guidelines in
> {{reference:tpu-sharding/tpu-sharding.md}}.

Your task request carries the session `run_id`. Resolve every path in this document against it (see {{reference:tpu-sharding/tpu-sharding.md}} §3.1):

```
<artifacts> = {{runs}}/<run_id>/artifacts
<memory>    = <artifacts>/optimization_memory.md
```

`<memory>` is this run's own copy of the optimization memory, seeded automatically
from the packaged catalog the first time it is used. Pass `--memory-file <memory>`
on every `query_tpu_sharding.py` invocation so reads and writes stay inside the run's
artifacts instead of mutating the packaged catalog.

Pass that same `run_id` verbatim when you delegate to any subagent.

---

### Precise Execution Protocol

1. **Pre-Flight Initialization & Concurrency Check**:
   - Receive the `run_id`, `iteration` ($N$), `backend`, optional `model`, `hardware_topology`, and either `configuration_payload` (for Run 0) or `recipe_artifact` (for Run $N \ge 1$) from the Orchestrator. `run_id` is the session identifier that resolves `<artifacts>` (see {{reference:tpu-sharding/tpu-sharding.md}} §3.1); `iteration` is the local experiment counter that names the `<N>/` subdirectory. Do not substitute one for the other.
   - **Single-Job Concurrency & Natural Completion Invariant**: Strictly ensure no previous physical experiment job is running. The agent MUST wait for the previous job to reach full natural completion (a terminal status from `{{runtime}} compute job status`) before launching a new run. Never prematurely abort successfully running jobs.
   - **Verify Directory**: Confirm that the run directory `<artifacts>/<N>/` exists (pre-created by the Orchestrator).
   - **Read & Update `report_<N>.md` (Preserve Hypothesis)**:
     * The `report_<N>.md` file must already exist in the run directory (pre-created by the Orchestrator). **Do not overwrite this file.**
     * Read the file, parse the **Hypothesis** and **Justification** (or baseline description) text to keep in memory.
     * Update only the Executive Summary metadata block in the file (setting `Status: RUNNING` and `Backend: <backend>`).

2.  **Execute Workload via Unified Compute**:

    -   **Generate Batch Recipe**: Format the configuration payload into a launch script `<artifacts>/<N>/recipe_<N>.<ext>` defining the workload execution commands, where `<ext>` matches the language you wrote it in (`.sh` or `.py`). Backends pick the runner from that extension: `bash` for `.sh`, the configured interpreter otherwise.
    -   **Submit Workload**:
        ```bash
        JOB_ID=$({{runtime}} compute submit --recipe <artifacts>/<N>/recipe_<N>.<ext> --run-id <run_id>)
        ```
    -   **Register Active Job ID**: Write the `JOB_ID` into `<artifacts>/<N>/report_<N>.md` (setting `Job ID: <JOB_ID>`) as soon as the job is submitted.
    -   **Monitor & Poll Status**:
        Periodically poll job status until terminal state (`COMPLETED` or `FAILED`):
        ```bash
        {{runtime}} compute job status --job-id $JOB_ID
        ```
    -   **Fetch Logs and Artifacts**:
        Once terminal, retrieve execution logs and artifacts:
        ```bash
        {{runtime}} compute job logs --job-id $JOB_ID > <artifacts>/<N>/container.log
        {{runtime}} compute job artifacts --job-id $JOB_ID --out-dir <artifacts>/<N>/
        ```
    -   **Errors & Teardown**: In the event of a major failure or timeout, cancel the job if still running:
        ```bash
        {{runtime}} compute job cancel --job-id $JOB_ID
        ```
        Record the failure status in `report_<N>.md` and escalate back to the Orchestrator.
    -   **Contribute to Optimization Memory**: If you resolve an error via a successful relaunch, delegate to `{{agent:tpu-sharding-memory-contributor}}` to ingest the fix, passing the `run_id`, the exact takeaway text, the source run link (`[<N>/report_<N>.md](<artifacts>/<N>/report_<N>.md)`), the model name (e.g. `Qwen3.5-397B`, from payload or report), and the hardware topology (from payload or report).

3. **Metrics Extraction & Log Parsing (Success / Partial Success)**:
   - **Log-Only Metric Extraction Rule**: ALL performance metrics (step time, TFLOPS/s/dev, token throughput, loss/perplexity, latency percentiles, memory usage, etc.) MUST be extracted directly from the raw execution stdout logs, benchmark summary outputs, or XProf telemetry. **DO NOT calculate or estimate any metric using analytical formulas**, with exactly ONE exception: **MFU (%)**, which is calculated using the logged `TFLOP/s/device`.
   - Open the container log you pulled locally (`<artifacts>/<N>/container.log`) and parse performance metrics.
    - **JAX Training Log Parsing**:
      * Parse execution stdout logs to extract step-by-step metrics (including step duration in seconds, TFLOPS per device, token throughput, loss, and perplexity).
      * **MaxText Example**: If the training framework is MaxText, locate step lines matching:
        `completed step: <STEP>, seconds: <SECONDS>, TFLOP/s/device: <TFLOPS>, Tokens/s/device: <TOKENS_PER_SEC>, total_weights: <WEIGHTS>, loss: <LOSS>, perplexity: <PERPLEXITY>`
        Extract the values list of `seconds`, `Tokens/s/device`, `TFLOP/s/device`, `loss`, and `perplexity`.
      * **Other Frameworks**: If using other JAX-based training frameworks, identify their specific log output formats and parse step performance dynamically.
     * **Training Anomaly Filtering & Averaging**:
       * **Warmup Exclusion**: Discard early warmup steps to exclude JIT compilation overhead. The exact number of early steps to discard is set in the launch recipe (e.g., via `--maxtext_flag=skip_first_n_steps_for_profiler` or custom setup) and may vary, so use the warmup step count the recipe sets rather than assuming one.
       * **Async Dispatch Filtering**: Exclude training steps with execution times that are extremely low compared to the stable step average (this indicates JAX async dispatch buffering where compilation queues overlap rather than actual TPU execution).
       * **Outlier Filtering**: Exclude steps showing extreme latency spikes (e.g., when the profiler starts/stops or during checkpointing) deviating significantly (e.g. $> 3\times$) from the stable step median.
       * **Partial Success 5-Step Rule**: If a run does not finish to completion, it is treated as **`PARTIAL_SUCCESS`** if and only if **at least 5 stable steps with very close results** (minimal variance, e.g. step duration variance $< 10\%-15\%$) are logged after warmup filtering. If fewer than 5 stable steps are available, classify the run as **`FAIL`**.
       * **Report Stable Averages**: Compute the mean of the remaining stable steps to report as **Step Time (s)**, **Throughput (tok/s/dev)**, and **TFLOPS/s/device**.
         * Formula: `MFU (%) = (TFLOP/s/device / Peak_TFLOPs_per_device_for_precision) * 100`
         * **Hardware Peak Database & Lookup Protocol**:
           - **Pre-Loaded Specs (v7x / Ghostfish)**: If the active hardware is `v7x` (`Ghostfish`/`gf`):
             * Peak compute per device (BF16): `1153.5 TFLOP/s` (Derived from 2307 TFLOP/s per chip / 2 devices per chip).
             * Peak compute per device (FP8): `2307 TFLOP/s` (Derived from 4614 TFLOP/s per chip / 2 devices per chip).
           - **Other Hardware (Lookup)**: If the active hardware is different (e.g., TPU v5e, TPU v6e, H100 GPU):
             * Query the optimization memory (`python3 {{script:tpu-sharding/query_tpu_sharding.py}} --memory-file <memory> ...`) to retrieve the peak compute capability (TFLOPS) of a single device (core/GPU) under the target precision (e.g. BF16, FP8).
           - Apply the retrieved `Peak_TFLOPs_per_device_for_precision` to the formula to calculate the stable MFU %.
   - **JAX Inference Log Parsing**:
     * If the workload is inference, parse the final JSON benchmark summary block or locate latency logs to extract:
       * TTFT (ms): P50 and P99
       * TPOT (ms): P50 and P99
       * ITL (ms): P50 and P99
       * E2E Latency (ms): P50 and P99
       * Request throughput (req/s) and output token throughput (tok/s)
       * KV-Cache Usage (%)
     * The benchmark client/harness handles warmup internally and outputs pre-cleaned latency percentiles directly; do not apply manual warmup or async dispatch filters.

4. **Profiler Trace Hand-off**:
    -   `{{runtime}} compute job artifacts` brings the run's `.xplane.pb`
        traces back under `<artifacts>/<N>/`, giving you the local TensorBoard
        logdir (`<artifacts>/<N>/tensorboard`), the profile run directory on
        the target (a `gs://` URI on backends that stage through GCS), and the
        `<SESSION_ID>` (the timestamp directory holding the traces).
    -   **Do not read or analyse the traces.** Record all three in
        `report_<N>.md`'s header (`Profiler Trace Path`, `XProf Session
        ID`) and return them with your results. The Orchestrator hands the
        trace to the package's XProf analyst, which writes
        `<artifacts>/<N>/tpu_profiling_report.md`; `report_<N>.md` links to it.

5. **Generate Cumulative Joint Report & State Updates**:
   - **Update `state.json`**: Populate the run's empirical trajectory fields (`job_id`, `status`, `execution_duration_seconds`, `target_metric`, `related_metrics`, `profiler_trace_path`, `xprof_session_id`) in the `trajectory` array of `<artifacts>/state.json` (conforming to `{{schema:tpu-sharding-state.json}}`).
     * **Canonical Unit Consistency**: Use the canonical units defined in `{{schema:tpu-sharding-state.json}}`: convert all time/latency metrics (`ms`, `us`, `min`) to seconds (`"s"`), keep percentages in `"%"`, keep memory capacity in `"GiB"` or `"GB"`, keep bandwidth in `"GiB/s"` or `"GB/s"`, compute in `"TFLOPS/s/device"`, throughput in `"tok/s/dev"`, `"tok/s"`, `"req/s"`, `"samples/s"`, or `"steps/s"`, and dimensionless/custom metrics in `""` (with the unit suffix appended to `name` for unlisted custom dimensions). For any iteration $N \ge 1$, **reuse the exact `(name, unit)` pairs recorded in `trajectory[0]` (`Run 0`)**.
   - **Write `report_<N>.md`**: Write the complete Markdown report under `<artifacts>/<N>/report_<N>.md`, rendering the standardized template in {{reference:tpu-sharding/tpu-sharding.md}} §3.2 exactly — that template is the single source of truth for this file's schema, so follow it there rather than reproducing it here.
     * You fill the metadata header (`Job ID`, `Backend`, `Executed Recipe Link`, `Total Running Time`, `Profiler Trace Path`, `XProf Session ID`), the **Target Metric** and **Performance Metrics** sections. The **XProf Analysis** section is a link to `tpu_profiling_report.md`, which the analyst writes after you return; leave it as the template has it.
     * **Executed Recipe Link** names the recipe that actually produced the result, since the winning attempt may be a sub-run (`recipe_<N>.<k>.<ext>`) rather than the original.
     * **Sub-run Attempt History**: render one block per attempt you launched. **Do not overwrite** prior sub-run attempts — append new attempts to preserve debugging history.
     * The **Optimizer Hypothesis & Detailed Analysis** section belongs to the Optimizer; preserve the Hypothesis you read in Step 1 and leave the Takeaways for it to fill post-run.

6. **Notification**:
   - Send the final status message to the Orchestrator adhering to `{{schema:sharding-executor-result.json}}` (`status: "completed"`, `execution_status: "SUCCESS"` / `"PARTIAL_SUCCESS"` / `"FAIL"` / `"TIMEOUT"`), with execution stats, target metric results, and the file link: `<artifacts>/<N>/report_<N>.md`. If it was a logical failure, flag it to prompt handoff to the Optimizer.
