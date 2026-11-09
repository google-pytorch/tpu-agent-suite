# The onboarded baseline

This directory holds google/gemma-4-31B-it, already onboarded to vLLM on TPU
and benchmarked. It is the starting point that this task optimizes.

The files are the output of a tpu-model-onboarding run on 2026-10-21. They are
kept here so that every trial starts from the same baseline instead of
onboarding the model again.

## The baseline server job

The onboarded server is the job `job-20261021054117-48973b`.

The job's compute record and server log are installed under
`~/.tpu-agent-suite/runs/jobs/`. As a result, `compute job status` and
`compute job logs` read this job the same way they read a job you started.
Copies of the records are also in `jobs/`. The job has ended, and nothing from
it is still running on the cluster.

The onboarding ran four jobs:

| Job | What it was |
| --- | --- |
| `job-20261021054117-48973b` | the vLLM server (`onboarding/inference/launch_vllm.sh`) |
| `job-20261021054837-82d4ad` | benchmark client, 1k/1k at concurrency 512 |
| `job-20261021060724-8618a7` | benchmark client, 1k/8k at concurrency 256 |
| `job-20261021060211-c6b620` | benchmark client, 8k/1k at concurrency 64 |

All four jobs ran on the `local` compute backend, on the onboarding host:

- The server ran on one tpu7x chip (2 devices). It ran from a Python
  environment, not from a container. `image_probe.json` lists the versions in
  that environment.
- The benchmark clients ran on the same host and reached the server at
  `localhost:8000`.

In the logs and records, host paths were replaced with `/opt/onboarding-host`
and `/root`.

## Serving recipe

The serving recipe is `onboarding/inference/launch_vllm.sh`. It uses the
following settings:

- TP 2 x DP 1
- `--max-model-len 10240`
- prefix caching off
- `--language-model-only`
- `--attention-backend CUSTOM`

The recipe needs two scripts from the installed package's `scripts/common/`
directory: `fetch_weights.py` and `serve_vllm.sh`. Three other files describe
the deployment:

- `tp_plan.json` records the sizing.
- `deployments.json` records the deployment.
- `verification.json` records the known-answer check that the server passed.

## Benchmark

The benchmark has three cells. Each cell ran as its own client job. Each job
ran the package's `serving_benchmark.py` client, which wraps
`vllm bench serve`, through `onboarding/bench/bench_client.sh`.

The prompts are real text from ShareGPT, cut to an exact length, instead of
random tokens. `build_dataset.py` and `dataset_report.json` describe how the
prompts were made. `submit_bench.fn.sh` is the exact submission command that
the onboarding used.

| Cell | ISL/OSL | Concurrency | Prompts | Dataset (`onboarding/bench/datasets/`, gzipped) |
| --- | --- | --- | --- | --- |
| `1k1k_c512` | 1024/1024 | 512 | 2048 | `sharegpt_isl1024_c512.jsonl.gz` |
| `1k8k_c256` | 1024/8192 | 256 | 768 | `sharegpt_isl1024_c256.jsonl.gz` |
| `8k1k_c64` | 8192/1024 | 64 | 256 | `sharegpt_isl8192_c64.jsonl.gz` |

The onboarding's own results are in `onboarding/bench/baseline_results.json`
and in each cell's `serving_bench/summary.*` files. For example, the 1k/1k
cell reached 2806.8 output tokens per second.

These numbers were measured on the onboarding host.
