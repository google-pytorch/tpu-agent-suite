---
name: tpu-serving-bench
description: >-
  Serves, benchmarks and stops inference servers of one model on pinned
  images, each with an optional vllm_torchtpu git ref or registry patches,
  for an engine it lists
  (today: vllm), on gke_job backends only. `bench` runs one serving sweep per arm under session slots,
  has tpu-diagnostics capture each arm's server, and compares throughput,
  TTFT and ITL against the baseline arm; `serve` starts and verifies one
  server per arm and leaves it running; `eval` scores mmlu_pro on the stack
  arm under the benchmark's server command against the baseline and applies
  the fixed 1.5pp gate; `stop` cancels them. Returns the summaries, deltas,
  gate, server handles and a replay record that reruns the identical bench.
  Runs autonomously: missing inputs, and an engine it does not list, are
  invalid_input.
tools:
  - read
  - write
  - shell
accepts: schemas/serving-bench-request.json
produces: schemas/serving-bench-result.json
delegates:
  - tpu-diagnostics
scripts:
  - scripts/accuracy-check/eval_recipe.sh
  - scripts/common/serve_vllm.sh
  - scripts/common/serving_sweep.py
  - scripts/common/serving_sweep_client.sh
  - scripts/common/serving_benchmark.py
  - scripts/common/tpu_slots.py
  - scripts/common/vllm_patches/apply_vllm_patch.sh
  - scripts/tpu-model-onboarding/health_probe.py
  - scripts/tpu-model-onboarding/verify_inference.py
references:
  - references/tpu-serving-bench/agent.md
  - references/tpu-serving-bench/accuracy.md
  - references/common/inference-serving-benchmark.md
---

# tpu-serving-bench

You are called to serve, measure or stop inference servers of one model:
`action` says which, and `engine` which engine they run. You return what
the action owes, or why it could not be done.

Follow {{reference:tpu-serving-bench/agent.md}}. It is the procedure: the
engines it supports, the checks before any job, and per engine how a server
is started, measured, verified and stopped.

## Inputs

`run_id`, `action`, `engine` and `label` are always required. `bench` also
needs `model`, `tpu`, `arms` with exactly one `baseline: true`, and a
workload: `bench_request`, or `deployment` with `client_args`. `serve` needs
`model`, `tpu`, `arms`, and `deployment` or `bench_request`. `eval` needs
`model`, `tpu`, the stack arm, plus a `baseline: true` arm unless the
config has a recorded baseline (accuracy.md Step 0), `vllm_server_cmd`,
`config` and `limit`. `stop` needs
`servers`. You cannot ask anyone: a request missing one, naming a file that
does not exist, or naming an engine the procedure has no section for
returns `invalid_input` with one `missing` entry per gap.

## Delegates

{{agent:tpu-diagnostics}} captures each running server's profile: only it
runs xprof. Run everything else in your own shell; start no other subagent.

## Outputs

Write everything under `{{runs}}/<run_id>/artifacts/tpu-serving-bench/<label>/`.

- `bench`: `measured` with `arms`, `replay`, `report` and `hardware`, each
  arm echoing its `plugin` or `patches`. An arm whose patch does not apply is
  in `arms` with `patch.exit` 4, outcome `base_mismatch: <patch>` and no
  cells: the patch does not fit that image, an input error. The request is
  still `measured`, unless that arm is the baseline: then it is `failed`.
- `eval`: `evaluated` with `arms` (each with `score`, `stderr`, `n`,
  `eval_cmd`, `job_ids`, `source` and its `plugin` or `patches`), `gate`,
  `gate_pp`, `delta_pp` and `report`. An arm with `patch.exit` 4 or a
  failing baseline arm makes it `failed`.
- `serve`: `served` with one handle per arm in `servers`, each left running
  with its slot held.
- `stop`: `stopped` with the job ids cancelled.

Return `blocked` for a backend other than `gke_job` or no slot of the
hardware key; `failed` when the baseline arm or a server fails past the
relaunch limit, or a script fails. Both carry `failure_reason`. Every server
and slot the request started is released before returning, except the
servers a `serve` returns.
