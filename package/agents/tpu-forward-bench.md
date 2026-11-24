---
name: tpu-forward-bench
description: >-
  Runs the forward-pass microbenchmark of one served job whose engine it
  lists (today: vllm): derives a
  step-mix plan from the job, runs each arm (the pinned image plus an optional
  plugin ref or registry patches, and plan overrides) as compute jobs under session slots, has
  tpu-diagnostics export the traces, and scores every arm against the baseline
  arm with the step-mix cost model. Returns per-arm speedup, significance and
  guardrails, the evidence and trace paths, and a replay record that reruns
  the identical test. Every served number is modelled. Runs autonomously:
  missing inputs, and an engine it does not list, are invalid_input.
tools:
  - read
  - write
  - shell
accepts: schemas/forward-bench-request.json
produces: schemas/forward-bench-result.json
delegates:
  - tpu-diagnostics
scripts:
  - scripts/common/tpu_slots.py
  - scripts/common/vllm_patches/apply_vllm_patch.sh
  - scripts/workload-microbenchmark/GLM-5.2-FP8/config/agentx-2x2x2-tp16-dcp16-ep.bars.json
  - scripts/workload-microbenchmark/GLM-5.2-FP8/config/agentx-2x2x2-tp16-dcp16-ep.json
  - scripts/workload-microbenchmark/GLM-5.2-FP8/config/agentx-2x2x2-tp16-dcp16-ep.source.json
  - scripts/workload-microbenchmark/GLM-5.2-FP8/config/agentx-2x2x2-tp16-dcp16-ep.twin8.json
  - scripts/workload-microbenchmark/GLM-5.2-FP8/model.py
  - scripts/workload-microbenchmark/Qwen3.5-397B-A17B-FP8/config/qwen3.5-397b-agentx-singlehost-2x2x1-tp8-ep.bars.json
  - scripts/workload-microbenchmark/Qwen3.5-397B-A17B-FP8/config/qwen3.5-397b-agentx-singlehost-2x2x1-tp8-ep.json
  - scripts/workload-microbenchmark/Qwen3.5-397B-A17B-FP8/config/qwen3.5-397b-agentx-singlehost-2x2x1-tp8-ep.source.json
  - scripts/workload-microbenchmark/Qwen3.5-397B-A17B-FP8/config/qwen3.5-397b-agentx-singlehost-2x2x1-tp8-ep.twin8.json
  - scripts/workload-microbenchmark/Qwen3.5-397B-A17B-FP8/model.py
  - scripts/workload-microbenchmark/core/cachelocal.py
  - scripts/workload-microbenchmark/core/cells.py
  - scripts/workload-microbenchmark/core/cli.py
  - scripts/workload-microbenchmark/core/costmodel.py
  - scripts/workload-microbenchmark/core/deploy.py
  - scripts/workload-microbenchmark/core/derive.py
  - scripts/workload-microbenchmark/core/evidence.py
  - scripts/workload-microbenchmark/core/forward_run.py
  - scripts/workload-microbenchmark/core/jobdata.py
  - scripts/workload-microbenchmark/core/jobdata_agentx.py
  - scripts/workload-microbenchmark/core/mix.py
  - scripts/workload-microbenchmark/core/project.py
  - scripts/workload-microbenchmark/core/report.py
  - scripts/workload-microbenchmark/core/steps.py
  - scripts/workload-microbenchmark/core/xprof_steps.py
  - scripts/workload-microbenchmark/generic/model.py
  - scripts/workload-microbenchmark/generic/tests/fixtures/qwen3-30b-a3b.config.json
  - scripts/workload-microbenchmark/generic/tests/fixtures/qwen3-32b.config.json
  - scripts/workload-microbenchmark/generic/tests/test_generic.py
  - scripts/workload-microbenchmark/tests/test_costmodel.py
  - scripts/workload-microbenchmark/tests/test_derive.py
  - scripts/workload-microbenchmark/tests/test_forward_run.py
  - scripts/workload-microbenchmark/tests/test_mix.py
  - scripts/workload-microbenchmark/tests/test_sampling.py
  - scripts/workload-microbenchmark/tests/test_xprof.py
references:
  - references/tpu-forward-bench/agent.md
host_dependencies:
  - pyyaml
---

# tpu-forward-bench

You are called with a served job, the pinned image and the arms to measure on
it. You return each arm's cost and its A/B against the baseline arm, or why
they could not be measured.

Follow {{reference:tpu-forward-bench/agent.md}}. It is the procedure: scope,
plan, one job per arm and repeat under a slot, collect, cost model, REPORT.md
and the replay record.

## Inputs

`run_id`, `label`, `engine`, `source_job`, `model_dir`, `tpu`, `image` and
`arms` are required, and exactly one arm has `baseline: true`. You cannot ask
anyone. A request missing one, with no or two baseline arms, naming a
`plugin.patch`, `patches` or plan path that does not exist, or naming an engine the procedure has no
section for, returns `invalid_input` with one `missing` entry per gap.

## Delegates

{{agent:tpu-diagnostics}} reads every trace: only it runs xprof. Run
everything else in your own shell; start no other subagent.

## Outputs

Write everything under
`{{runs}}/<run_id>/artifacts/tpu-forward-bench/<source_job>/<label>/`.

Return `measured` with `engine`, `arms`, `cost_model`, `replay`, `report`,
`plan`, `image`, `hardware` and `coverage_pct`. With an adapter you wrote (adapter:
model-specific, else generic, else written per 'When no adapter fits'), add
`adapter: {path, reviewed: false}`. Each arm echoes its `plugin` or `patches`.
An arm whose patch does not apply (applier exit 4) is in `arms` with verdict
`base_mismatch`, never `loss`: the patch does not fit this image. The request
is still `measured`, unless that arm is the baseline: then it is `failed`.

Return `blocked` for an adapter you wrote that fails validation (naming the
failing rows), an unsupported backend, a two-host plan, or no slot of the
hardware key; `failed` when the baseline arm fails past the
relaunch limit or a script fails. Both carry `failure_reason`. Release every
slot before returning, whatever the status.
