# `core/` — model-agnostic code

Everything here counts, divides and formats. Model-specific facts and rules live in
`<MODEL_DIR>/model.py` (beside `core/` for the models the package ships, or under
`<work>/models/` for a model being onboarded), which `derive.load_model` loads. `SKILL.md` covers usage; this
file covers how the code fits together.

## The analysis model

A dynamic serving job is reduced to **the steps its engine ran, with counts**
(`T_busy ≈ Σ steps × step time`). Those steps are then timed on a small-slice twin of the
model.

1. **Snapshot** a finished job: server command, requests, server counters, and the per-second
   step histogram.
2. **Derive** the deployment and its per-rank shapes, and check that it is feasible.
3. **Project** the twin (one layer per kind) onto the slice it runs on: **scaled** mode on one
   2x2x1 host (split dims scaled so per-rank shapes match), or **mirror** mode on the served slice
   (the served sharding as is, over two hosts on Ray).
4. **Step mix**: classify every second of the job as decode (D1-2 / D3-4 / D5-8 requests),
   mixed (a prefill chunk rides the step) or idle. Then split each class into cells by context
   and batch size, each weighted by its share of measured busy time.
5. **Plan** one batch per cell (real tokens, served bucket rule): one self-contained
   `.steps.json` per cell, all sharing one compile cache.
6. **Run** each cell as its own job, in parallel, on the TPU host: time and profile the cell.
7. **Collect**: split the profile (the event export tpu-diagnostics writes) into layers, project to
   the served model (modelled), and
   write the evidence the agent's REPORT.md is built from.
8. **Cost model**: `cli.py mix-rollup` sums `T_mix = Σ steps × t` over the cells' jobs, and with a
   baseline gives the speedup (below).

## The representative workload model (`mix.py`)

**In short:** the model asks which step shapes the served engine spends its time on, and how
much of the time goes to each. It reads that from the job's own server metrics, classifies every
second of the measured window by the kind of step that ran, and splits each kind into a few
concrete step shapes (cells). Each cell carries its measured step count and share of busy time.
The cells are what gets benchmarked, and the result is weighted back by those counts.

**Inputs** (all in the snapshot, from `jobdata.py`):
- `step_slices`: aiperf's 1-second scrape of `vllm:iteration_tokens_total` (a histogram of tokens
  per engine step, recorded when the step **ends**) and the `num_requests_running` gauge, over the
  measured window;
- `requests`: each request's start, first-token and end times, and its lengths;
- the deployment (MTP k, `mnbt`, `max-num-seqs`) and the prefix-cache rule (`cells.with_hit`).

**Step 1: classify each second.** With MTP k, a decoding request contributes 1+k tokens per
step, so a step's token count tells its kind:

| tokens in the step | class |
| --- | --- |
| ≤ 8 | decode, 1–2 requests (`D1-2`) |
| ≤ 16 | decode, 3–4 requests (`D3-4`) |
| ≤ 32 | decode, 5–8 requests (`D5-8`) |
| > 32 | mixed: a prefill chunk rides the step |

- **Step time per decode class** is fitted on fully busy decode-only seconds, where
  `1 s = Σ steps_j × t_j`.
- **The rest of each second** goes to *mixed* when a long step is in flight: no step ended while
  requests were running, or a > 32-token step ends in this second or the next. Otherwise it is
  *idle*: the client thinking between turns, which no benchmark prices.
- **Validated** against a job that also logged every scheduler step (`j-c964a8b6`), after
  correcting that log for async scheduling's one-step lag: within about 5 points, with the error
  toward mixed.

**Step 2: cells.** Each class is split so that every cell is one concrete, runnable step:
- **by context:** three bins covering 0–50%, 50–90% and 90–100% of the class's time, each at its
  time-weighted median context. The context comes from the requests that were decoding in those
  seconds;
- **by batch:** the two whole batch sizes bracketing the class's mean number of decoding requests,
  with shares that reproduce the mean;
- **mixed cells:** the prefix is the prefilling request's cache hit under this deployment, binned
  the same way. The chunk fills the step budget after the decode rows: `mnbt − B × (1+k)`.

**Step 3: weights.** Every cell carries `steps` (its count in the job) and `share` (its fraction
of measured busy time). A benchmark result rolls up as `Σ steps × step time`, per class and in
total, beside the job's measured busy time. So a change's effect on the workload is its effect on
that sum, not on any one cell. `steps.py` drops cells under `--min-share` and reports the share
it kept.

**Known limits:**
- A re-sharded plan (`--override`) keeps the job's step counts. A faster sharding that would
  change the schedule itself is not modelled.
- The cache-hit rule assumes `block × DCP` granularity plus an MTP block drop. On newer images
  (`prefix_match_unit` 256) it over-predicts computed tokens, so mixed-cell prefixes come out low.
  This does not affect counts or shares, which come from the server.

**Alternative considered: request percentiles.** The earlier model built P50 / P90 points
(`cached / input / output @ C`) from the requests table. It assumes every step carries C
identical requests, and pairing marginal percentiles (P90 context with P90 concurrency) builds a
step that never ran. In GLM-5.2 AgentX at client concurrency 4, half the engine's time is decode
steps with 1–2 requests, while the P50 / P90 points ran 7–8 per step, which is 3% of the job. A
hand-typed point (`--point`) is still accepted as an explicit what-if.

## The cost model (`costmodel.py`)

One number per run: the job's engine-busy time at the twin's shapes,

```
T_mix = Σ over cells i of  steps_i × t_i       (steps from mix.py, t from forward_run, median over repeats)
speedup = T_mix(baseline) / T_mix(candidate)   (an A/B over the same cells)
```

On the **served basis** (primary), `t` is not the twin's wall step but the cell's step at the
served layer counts: `fixed + Σ_kind n_kind × layer_kind`, from the cell's profiles (pooled, per-kind
median; `costmodel.served_times`), with `n_kind` from the model. The twin wall basis is kept beside
it. SKILL.md "Cost model" explains why the twin's ratio does not carry over. The **served_device**
basis (`costmodel.served_device_times`, primary for kernel changes) drops the host wall: before /
after busy + per-step ops + `Σ n_kind × layer_kind` + drafter busy + sampling busy, one sample per
profile, and `costmodel.significance` (Welch's t) tests two arms on those samples. The sample stage
(plans with `sampling.run`) counts once per step on both served bases.

`cli.py mix-rollup` reads the cell plans, each job's `result.json` and `xprof_steps.json`, calls
`costmodel.evaluate` per arm and basis and `costmodel.compare` for an A/B, and writes
`cost_model.json` with:
- per arm: `T_mix`, per-class seconds, `T_mix` per repeat and its spread, coverage, missing cells;
- the A/B: speedup, whether it beats the repeat spread, per-class deltas, guardrails (the
  heaviest decode cell for typical TPOT, every mixed cell for TTFT and the tail), and the
  modelled served output gain `(busy + idle) / (busy * (covered_share / speedup + 1 - covered_share) + idle)`.

It is valid for changes that keep the job's step counts (kernels, collectives, env, layout). Changes
to the schedule (chunk size, `mnbt`, speculative tokens or acceptance, admission) change the counts
and need a server-level run. SKILL.md "Cost model" is the full statement.

## Files

| file | role | reads → writes |
| --- | --- | --- |
| `cli.py` | entry point for every mode (`onboard`, `derive`, `project`, `points`, `forward-cmd`, `collect`, …) | orchestrates the rest |
| `jobdata.py` | fetched job dir → snapshot: the server from vLLM's log, the client from aiperf or `summary.json` | `<work>/jobs/<job>` (`cli.py --jobs`) → `*.source.json` |
| `jobdata_agentx.py` | builds on `jobdata`: an AgentX job's JobSet manifest (`recipe.yaml`) → served command and env, and the checkpoint config it names; else `jobdata`'s vLLM-log readers. `cli` reads every job through its `snapshot` and `read_recipe` | `recipe.yaml` → `jobdata.snapshot` |
| `deploy.py` | server args → `Deploy`; per-rank shapes; `Facts` type | — |
| `cells.py` | cache-hit rule (`with_hit`), request-level cells and counts, closure check | snapshot requests |
| `derive.py` | snapshot (+ overrides) → real-workload config, feasibility | snapshot → `<name>.json` |
| `project.py` | real config → twin config, scaled (`twin8`) or mirror (`twin16`) | `<name>.json` → `<name>.twin<N>.json` |
| `mix.py` | snapshot → **step mix** (cells, counts, shares) | snapshot `step_slices` + requests |
| `steps.py` | mix (or `--point` shapes) → batches, buckets, plan | → `<sharding>-mix-<job>/<cell>.steps.json` |
| `costmodel.py` | per-cell step times → `T_mix`, spread, and an A/B (speedup, class deltas, guardrails) | results + plans → `cost_model.json` (via `mix-rollup`) |
| `forward_run.py` | **runs on the TPU host**, in the image under test: boot twin, time/profile the plan; `--node-prep` readies each host of a mirror run. The recipe runs it under `supervise`, which stops the job if engine start-up fails and the engine does not exit | plan → `result.json` and a profile; `failure.json` if the run ends without a result |
| `cachelocal.py` | local compile-cache store `~/.xla_cache` | job outputs ↔ cache |
| `xprof_steps.py` | twin profile → steps, layers, served projection | `trace_events.json` (exported by tpu-diagnostics) + plan + result → `xprof_steps.json` |
| `evidence.py` | a finished run → `evidence.json` (facts only, no prose) | result, plan, `xprof_steps.json`, `hlo_categories.json` (from tpu-diagnostics) |
| `report.py` | text tables for CLI output (summaries, selftest, request tables) | data it is given |

## How they connect

```
job dir ──jobdata──> snapshot ──derive (deploy, cells, model.py)──> real config
                        │                                              │
                        │                                          project
                        │                                              v
                        └──────── mix ─────> steps ───────────> twin config + plan (.steps.json)
                                                                        │
                                          forward-cmd (cachelocal) → recipe, compute submit
                                                                        v
                                         forward_run.py on the TPU host → result.json, profile
                                                                        │
                                          tpu-diagnostics: trace_events.json, hlo_categories.json
                                                                        │
                                                 collect: cachelocal, xprof_steps, evidence
                                                                        v
                                                 result.json + xprof_steps.json + evidence.json
                                                                        │
                                                                 agent writes REPORT.md
```

- `cli.py` is the only file that calls across stages. The library modules import only
  `deploy`, `cells` and `jobdata` (which `jobdata_agentx` builds on); `report` is used only by
  `cli`.
- `forward_run.py` and `xprof_steps.py` are standalone scripts: the first runs on the TPU host
  with the image's vLLM, the second on this host, on the event export. Neither imports another
  `core` module; the plan they get is self-contained.
- The cache-hit rule has one home, `cells.with_hit`, and `derive`, `mix` and `steps` all use
  it. A model can override it with `cache_hit` in its `model.py`.
