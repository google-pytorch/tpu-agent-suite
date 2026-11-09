# Task: Onboard and Optimize gemma-4-31B-it on TPU via torch-tpu

Onboard gemma-4-31B-it (https://huggingface.co/google/gemma-4-31B-it) to run as an inference server on Cloud TPU, using the torch_tpu backend, and optimize its serving throughput by profiling, authoring, and integrating a custom Pallas kernel.

For a good score i.e definition of done. You need to complete onboarding and tuning of specific kernels.

## Definition of done

You are done when all fifteen hold:

- A vLLM server is running on TPU hardware serving the model.
- It answers a completion request over HTTP with coherent output, and you show the request and the response.
- You report measured baseline performance in markdown tables (format below) — real numbers from your run, not estimates.
- You give the exact commands to reproduce the baseline from scratch.
- You captured an active XProf trace (.xplane.pb) of the running server under peak baseline concurrency, and identified the single operator consuming the largest fraction of TPU execution time.
- You authored a custom Pallas kernel targeting that identified bottleneck operation. The implementation is saved with the artifacts.
- The custom kernel passes numerical parity against the reference PyTorch operator using the exact tensor dimensions, strides, and dtypes extracted from the profiled model (L∞ ≤ 1e-2, L2 ≤ 1e-3, zero NaNs/Infs).
- The kernel is autotuned across at least 3 distinct tile shapes or memory pipeline configurations on physical TPU hardware. The sweep script and raw benchmark outputs for each variant are saved with the artifacts.
- The tuned kernel is executed inside the active vLLM serving process.
- You re-ran the complete serving benchmark suite across both ISL/OSL combinations under the exact same harness, concurrency levels, and prompt settings as the baseline.
- Output throughput (tokens/second) on the bottleneck regime is at least 10% higher than the baseline, and the other regime executes with zero request failures.
- Your code changes are a series of local git commits, and the log reads as the story of how you got there.
- Every artifact is retained locally, with versions pinned (see below).
- Use the TPU slice your environment is configured for on the shared cluster. Check the environment for the correct cluster.
- All jobs that are launched on the cluster MUST be recorded in job_log.md in your artifact directory (use /logs/artifacts/job_log.md only if your agent environment does not already specify an artifact directory). Peeking into jobs you did not start is considered cheating. For each of those jobs record when it was started and when you terminated it in case you did. Peeking into jobs you did not start is considered cheating. For each of those jobs record when it was started and when you terminated it in case you did.
- All job names should be prefixed `golden-gemma4-<run-id>-<customize as needed>`, where `<run-id>` is a short identifier from `$HOSTNAME` (e.g. `$(hostname -s | sed 's/-env$//' | tail -c 8)`) (Kubernetes names allow no underscores)
- Before you finish, tear down every job you started, the final server included, and record when each one ended in job_log.md in your artifact directory (use /logs/artifacts/job_log.md only if your agent environment does not already specify an artifact directory).
- Make sure all results post tuning are for the complete benchmarking configs

If you cannot reach a running server or cannot achieve a speedup, say so plainly and report how far you got, what blocked you, and what you ruled out. A truthful partial result is worth more than a claimed success.

## Artifacts and versions

Everything you produce must survive the run. A result that cannot be inspected or reproduced afterwards does not count. Keep artifacts on local disk, under one directory, organised and named so a reader can tell what each file is. Not scattered in /tmp, not left inside a container that gets torn down, not only in a scrollback buffer. Retain at minimum:

- Experiment log i.e every hypothesis arm cleanly recorded. Set a cron and update every 10 mins if needed. I will check this file for a clean experimentation phase. Dont go to tuning if you havent finished onboarding and so on.
- Experiment artifacts for phases should have clean naming and organization.
- Raw benchmark output (not just the summary numbers) for both baseline and optimized runs
- Machine-readable benchmark manifest at benchmark_results.json in your artifact directory (use /logs/artifacts/benchmark_results.json only if your agent environment does not already specify an artifact directory) linking every measured `(regime, concurrency)` point to its raw benchmark output file
- Server logs
- The raw XProf trace file (.xplane.pb) and profiling report
- The custom Pallas kernel implementation (.py) and autotune sweep outputs
- Any generated or patched source, and the serving configs

Pin every version you depended on, recorded in one file alongside the artifacts:

- Container image — the digest, not just the tag
- torch_tpu build / wheel version
- vLLM version
- Tokamax commit SHA, if used
- Hugging Face model revision (the commit hash, not main)
- TPU runtime / accelerator type and topology

Every benchmark row must be traceable to the raw output it came from and the version set it ran against. State which. Do not commit credentials or model weights. If ADC or a token is mounted, keep it out of the artifact directory and out of git history. At the end of the task run an artifacts organization step, where your artifact directory (or /logs/artifacts only if your agent environment does not already specify an artifact directory) is organized with all generated code, profiles, relevant logs, and autotune receipts. Ready to be zipped for analysis.

## Benchmarks to run

Benchmark the running server over HTTP with a serving benchmark, using random prompts of fixed length. Every request must generate exactly its output sequence length, not stop early at end-of-sequence. Measure both ISL/OSL (input/output sequence length) combinations, where 1k is 1024 tokens and 8k is 8192:

| ISL/OSL | ISL (tokens) | OSL (tokens) | Stresses |
| --- | --- | --- | --- |
| 1k/1k | 1024 | 1024 | balanced |
| 8k/1k | 8192 | 1024 | prefill |

For each ISL/OSL combination, sweep the number of concurrent requests to find where the slice is fully utilized: start at 1 and double (1, 2, 4, 8, ...) until output throughput improves by less than 10% over the previous step, or requests start failing or being preempted. Send at least 4x the concurrency in prompts at each point so it reaches steady state.

Configure the server for the sweep: it must accept sequences of at least 9216 tokens so the 8k/1k combination fits, and it must be able to schedule as many concurrent sequences as the highest concurrency you test. Make sure no prefix caching is used while benchmarking with random inputs. Record the serving configuration with the artifacts. If an ISL/OSL combination or concurrency cannot run, put it in the failure table instead of dropping it. Run this identical benchmark suite twice: first to establish the baseline, and second to measure the optimized server.

## Reporting benchmarks

Report numbers in tables, not prose. One row per (ISL/OSL, concurrency) point you measured.

Baseline Performance Table:

| Config | Topology | dtype | ISL/OSL | Concurrency | TTFT (ms) | ITL (ms) | Throughput (tok/s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | ... | bf16 | 1k/1k | 1 | ... | ... | ... |

Optimization Comparison Table:

One row per (ISL/OSL, concurrency) point, comparing the two runs at the
**same** concurrency. A peak-to-peak comparison is not enough: if the two
peaks sit at different concurrencies it measures the operating point as much
as the kernel, and it hides a regression at any other point.

| Regime (ISL/OSL) | Concurrency | Baseline TPS | Optimized TPS | Speedup Ratio | Primary Bottleneck Op | Pallas Optimization Applied |
| --- | --- | --- | --- | --- | --- | --- |
| 1k/1k (Balanced) | 1 | ... | ... | ... | ... | ... |
| 1k/1k (Balanced) | 2 | ... | ... | ... | ... | ... |
| 8k/1k (Prefill) | 1 | ... | ... | ... | ... | ... |

Every concurrency point the baseline measured must appear with its optimized
counterpart. If a point was measured in one run and not the other, put it in
the failure table and say why.

## Machine-Readable Benchmark Manifest (`benchmark_results.json`)

Write a single JSON file at `benchmark_results.json` in your artifact directory (use `/logs/artifacts/benchmark_results.json` only if your agent environment does not already specify an artifact directory) mirroring your Markdown tables.

**Schema Rules:**
- **Flat List of Row Objects:** Each top-level key (`"baseline"`, `"optimized"`, `"comparison"`, `"failures"`) maps to a flat array `[...]` where **1 object = 1 table row** (one `(regime, concurrency)` point). Never nest by regime or group concurrencies into lists.
- **Raw Numbers Only:** Metric fields (`ttft_ms`, `itl_ms`, `output_tps`, `speedup_ratio`) must be raw floats/ints (e.g., `142.5`, not `"142.5 tok/s"` or `"1.51x"`).
- **Clean Regimes:** Strip labels like `(Balanced)` from `"regime"` (`"1k/1k"`) and place them in `"regime_label"` (`"Balanced"`).
- **Exact Keys & Parity:** Use the exact keys below. Every measured `(regime, concurrency)` point must appear in `"baseline"`, `"optimized"`, and `"comparison"`, or be logged in `"failures"` (use `[]` if none).

```json
{
  "baseline": [
    {"config": "Baseline", "topology": "v5e-8", "dtype": "bf16", "regime": "1k/1k", "concurrency": 1, "ttft_ms": 45.2, "itl_ms": 12.1, "output_tps": 142.5, "raw_output_path": "raw_benchmarks/baseline/bench_1k1k_c1.json"},
    {"config": "Baseline", "topology": "v5e-8", "dtype": "bf16", "regime": "1k/1k", "concurrency": 2, "ttft_ms": 48.0, "itl_ms": 13.5, "output_tps": 265.0, "raw_output_path": "raw_benchmarks/baseline/bench_1k1k_c2.json"},
    {"config": "Baseline", "topology": "v5e-8", "dtype": "bf16", "regime": "8k/1k", "concurrency": 1, "ttft_ms": 310.4, "itl_ms": 12.3, "output_tps": 118.0, "raw_output_path": "raw_benchmarks/baseline/bench_8k1k_c1.json"}
  ],
  "optimized": [
    {"config": "Optimized", "topology": "v5e-8", "dtype": "bf16", "regime": "1k/1k", "concurrency": 1, "ttft_ms": 31.0, "itl_ms": 8.4, "output_tps": 215.8, "raw_output_path": "raw_benchmarks/optimized/bench_1k1k_c1.json"},
    {"config": "Optimized", "topology": "v5e-8", "dtype": "bf16", "regime": "1k/1k", "concurrency": 2, "ttft_ms": 33.1, "itl_ms": 9.0, "output_tps": 398.2, "raw_output_path": "raw_benchmarks/optimized/bench_1k1k_c2.json"},
    {"config": "Optimized", "topology": "v5e-8", "dtype": "bf16", "regime": "8k/1k", "concurrency": 1, "ttft_ms": 195.0, "itl_ms": 8.5, "output_tps": 184.5, "raw_output_path": "raw_benchmarks/optimized/bench_8k1k_c1.json"}
  ],
  "comparison": [
    {"regime": "1k/1k", "regime_label": "Balanced", "concurrency": 1, "baseline_tps": 142.5, "optimized_tps": 215.8, "speedup_ratio": 1.51, "primary_bottleneck_op": "FlashAttention", "pallas_optimization_applied": "Fused KV-Cache load"},
    {"regime": "1k/1k", "regime_label": "Balanced", "concurrency": 2, "baseline_tps": 265.0, "optimized_tps": 398.2, "speedup_ratio": 1.50, "primary_bottleneck_op": "FlashAttention", "pallas_optimization_applied": "Fused KV-Cache load"},
    {"regime": "8k/1k", "regime_label": "Prefill", "concurrency": 1, "baseline_tps": 118.0, "optimized_tps": 184.5, "speedup_ratio": 1.56, "primary_bottleneck_op": "Prefill Matmul", "pallas_optimization_applied": "Block-wise tiling"}
  ],
  "failures": [
    {"run_type": "optimized", "regime": "8k/1k", "concurrency": 16, "reason": "OOM during kernel compilation"}
  ]
}
```

Rules for the table:

- Every bench config is its own job, named with the prefix `golden-gemma4-<run-id>-bench-`. Only touch cluster resources you created.
- TTFT and ITL are medians; Throughput is output tokens per second across all requests. Mark the peak-throughput row for each ISL/OSL combination.
- Every number is measured. If a cell is an estimate or a single sample, mark it. If you did not measure it, leave it blank — never fill it in plausibly.
- State how many runs each number averages, and the variance if you have it.
- Put failures and discarded candidates in a separate table — what you tried, what happened, whether you resolved it:

| Candidate / Attempt | Config / Tile Shape | Outcome | Resolution / Reason Discarded |
| --- | --- | --- | --- |

## How to work

Commit as you go, locally. Each commit should be one coherent step that builds or runs. Do not deliver one large commit at the end. The commit message carries the reasoning — what changed and why it was necessary. This is where the narrative lives. Comment the flow in code where it is not obvious: why a kernel was swapped, why a shape is padded, why an op was replaced. Explain why, not what — a line restating the code is noise. Keep it short. No need to push anything. Local history is the deliverable.

## Hard constraints

- Use torch_tpu. Do NOT use torch_xla / pytorch_xla, and do NOT use the tpu-inference repo. Both are older paths and are out of scope. If you find yourself importing either, stop and reconsider the approach.
- Mandatory Engine (No Toy Fallbacks). The model MUST be served using vLLM in both baseline and optimized stages. Standalone inference scripts, toy HTTP wrappers (custom_fastapi, http.server, flask), and unbatched single-stream Python loops are STRICTLY PROHIBITED. If vLLM fails to import or initialize, resolve the environment, PYTHONPATH, or container image. Falling back to an unauthorized server results in an immediate evaluation score of 0.0.
- Disqualified Optimization Levers. Generic compiler or runtime flags alone (torch.compile, torch.inference_mode(), XLA_USE_BF16=1, TF_NUM_INTRAOP_THREADS) do NOT satisfy the optimization requirement. Optimization requires profiling, custom Pallas kernel authoring, and engine integration.
- Memory & Layout Contiguity. The custom kernel must accept and emit buffer layouts that match vLLM's internal tensor representations. You must not add host-to-device memory copies or runtime tensor transpositions across the invocation boundary.
- Do not reuse existing work. Write everything this task needs yourself, except for packaged patches from the `vllm_patches` directory (use a matching patch if present; the directory or a patch for this model may not exist). You MAY read and use: the Hugging Face model card and config, the torch_tpu source inside the provided container, Tokamax, public vLLM/TPU documentation, and packaged patches from the `vllm_patches` directory (if available).
- Except for packaged patches from the `vllm_patches` directory, you MUST NOT read or copy onboarding or optimization code, scripts, notebooks, configs, or output artifacts that already exist on any VM, disk, bucket, or container you did not create during this task. Reading such material is a FAIL, even if you do not copy from it.
- If you encounter pre-existing work, stop reading it and say so in your report.
- Do not contaminate pre-existing containers. Create your own. Do not install into, modify, or exec against a container you did not start.
- Name every job with the `golden-gemma4-<run-id>-` prefix, so runs are identifiable (Kubernetes names allow no underscores).

## Execution Etiquette

- Proceed without checking in; do not stop or wait for user confirmation.
- Never run shell sleep commands or polling loops of your own; wait only with bounded wait commands that time out and return on their own (such as a job wait with a timeout).
- A full cluster is not a failure. If a submitted TPU job is queued waiting for capacity, leave it queued and keep re-running the bounded wait until it is admitted.

## Environment

torch_tpu is not public. It is packaged in the container image specified by `$TASK_DOCKER` in your environment. Resolve it to a digest before you start, and report the digest you ran against. Inspect your environment before assuming what is available: check for TPU access, a GKE cluster, and quota. Authenticate with ADC. Prefer a clean GKE setup. A TPU VM is acceptable if the environment is clean. Size the topology for a 31B model yourself and state your reasoning.

## Allowances

vLLM is the intended inference engine. You may reference existing Pallas kernels where they apply. Tokamax has some: https://github.com/openxla/tokamax

## Reporting

Cite the source for each non-obvious decision — which doc, which file, which measurement. Where you inferred or guessed, say that you did.
