---
description: The hill-climb on one served job whose engine both benches list (today vllm), run by tpu-agent. Microbenchmark the job, have its gaps ranked with hypotheses, review them, test each as a branch of the pinned image's plugin history, and confirm the stack with an end-to-end serving bench and an accuracy check, delegating every microbenchmark, serving bench, profile, floor, kernel, code change and review.
scripts:
- scripts/common/tpu_slots.py
references:
- references/tpu-agent/hill-climb/LEDGER-TEMPLATE.md
delegates:
- tpu-forward-bench
- tpu-performance-optimization
- tpu-serving-bench
- tpu-diagnostics
- tpu-roofline
- tpu-pallas
- tpu-code-arm
- tpu-critical-review
- tpu-code-stack
---

# Hill-climb

One served job in, one end-to-end-verified stack of wins out. Follow this when the user asks to
close the gaps on a job, once query understanding has confirmed the plan. From then on it is an
autonomous loop: it never asks the user and never ends on its own. Step 8 starts the next round;
only `blocked` stops it.

```
0. baseline      tpu-forward-bench: the job twice on the pinned image
1. triage        tpu-performance-optimization: floors, ranked gaps, hypotheses
2. the list      the ranked gaps and their hypotheses → the ledger
3. review        the hypotheses → tpu-critical-review, before any TPU spend
4. each gap      dig → test → A/B against the stack head → stack or stop
5. combine       tpu-forward-bench: the stack vs the last verified stack → projected served gain
6. E2E           tpu-serving-bench: the stack alone, against the last verified stack's numbers
7. accuracy      tpu-serving-bench eval: the stack on mmlu_pro, against the recorded score
8. image         tpu-code-stack: push the stack as the next campaign's image
```

You run the shell steps below and write the ledger; everything else goes to a subagent. Every
microbenchmark goes to {{agent:tpu-forward-bench}}, every server and serving benchmark to
{{agent:tpu-serving-bench}}, the ranking and hypotheses to {{agent:tpu-performance-optimization}},
all XProf to {{agent:tpu-diagnostics}}, every later floor to {{agent:tpu-roofline}}, every kernel
to {{agent:tpu-pallas}}, every other plugin code change to {{agent:tpu-code-arm}}, every git action
on the campaign repo to {{agent:tpu-code-stack}} with `caller: hill-climb`. Delegations run one at a time.

## Start

Check these before step 0, in order; each failure stops `blocked` with the reason, before any job.

0. **The source is a finished job id.** tpu-forward-bench derives its plan from the job, so a
   local onboarding directory with no job id stops
   `blocked: the hill-climb needs a finished job id to start from`.
1. **The engine is explicit.** Read it from the source, never assume it: `deployments.json`
   `engine`, a `bench_request`'s `--engine`, the inventory's `engine`, else the recipe's server
   command (`vllm serve` is `vllm`). Record it in the ledger's setup block and pass it as `engine`
   on every tpu-forward-bench and tpu-serving-bench request. None found, or one either request
   schema's `engine` does not list (today both list only `vllm`), stops
   `blocked: engine <x> is not supported by the hill-climb's benches`.
2. **The backend is `gke_job`.** Each job is a fresh container from the pinned image, so a patch
   in one arm never reaches the next. Read it from the profile:

   ```bash
   {{runtime}} compute --package {{package_tag}} preflight --json
   ```

   Read `backend` and `status` from its JSON. Read only; it starts nothing. No source edits, no xprof CLI, no raw XPlane reads, no floors.
3. **The model has a microbenchmark adapter:** model-specific, else generic, else written per
   'When no adapter fits' ({{agent:tpu-forward-bench}} does this). Only an unreviewed adapter that
   fails validation stops the climb, with the `blocked` reason tpu-forward-bench returns.
4. **An earlier ledger exists** at `./.work/<MODEL_DIR>/hill-climb/<job-id>/ledger.md`: go to
   **Resume**. Otherwise create it from {{reference:tpu-agent/hill-climb/LEDGER-TEMPLATE.md}}.

Record at start: the source job, the model, the engine, the session `run_id`, the pinned image
(`compute.params.image`, by digest), and, for the vllm engine, the campaign repo: send
{{agent:tpu-code-stack}} `init` with `repo` `/workspace/vllm-torchtpu` when that directory is
on this host (a copy of the pinned image's plugin, edited and stacked in place), else
`./.work/<MODEL_DIR>/hill-climb/<job-id>/plugin`, the pinned image and the profile. Record its
`plugin` as the stack and its `worktree`. `blocked` stops `blocked: <reason>`.
Record the accuracy `config` too, `<model>-<quant>-<sharding>` from the source, and whether
{{agent:tpu-serving-bench}}'s `eval` has a baseline for it. None is not a stop: step 7 then sends
a baseline arm too, and its score becomes that config's baseline.

The session's TPU slots are the ones the `gke_job` profile declares in its `slots` param (one when
unset); the first delegate to claim a slot opens them. After step 0, record their count in the
ledger's setup block:

```bash
python3 {{script:common/tpu_slots.py}} show --run-id <run_id>
```

## The loop

```
while ledger.step != blocked:
    read the ledger; re-read the section "Re-read next" names
    do the next action
    write the ledger: artifacts, job ids, slots, request and result paths, verdict, next action,
    section to re-read
```

- **The ledger is the state.** Never trust memory of earlier turns; the run will be compacted.
  Write it after every artifact: a job id, a slot, a digest, a verdict. A number goes in only with
  the job id behind it.
- **Microbenchmarks.** Each is one `forward-bench-request`; save it and its result beside the
  ledger as `bench/<label>.request.json` and `bench/<label>.result.json`, and record both paths
  and each arm's `ml_diagnostics_urls`, the links that open its captures.
  The result's `replay` is a complete request that reruns the identical test: to rerun or vary it,
  send that object itself, changing only `label` and `arms`; never nest it inside another request.
  An arm with verdict `base_mismatch` (applier
  exit 4) is not a loss: its ref is off the image's base; fix the ref and re-send. `failed` on a
  baseline arm: send it once more, then stop `blocked`.
- **Launches are the only budget.** Count every TPU job and server any delegate starts, from its
  result's job ids, against the gap and against the campaign: the user's limit when query
  understanding recorded one, else none. Nothing else is budgeted: not A/B rounds, candidates or
  time. Keep enough for steps 5–7 (combine: 4 jobs; E2E: 2 servers per shape; eval: 2 jobs); when
  only that is left, every open gap stops as `budget` and the climb goes to step 5.
- **Subagent answers.** `invalid_input` is your omission: re-send once with the missing fields,
  then block the gap. `needs_input` blocks only its gap and is never put to the user. This
  overrides **After a profile** and **If a specialist returns `needs_input`** while the loop runs.
- **The profiling gate.** Step 0's microbenchmark runs with `profile: true`, and its traces are the
  climb's profile: they meet SKILL.md's **Mandatory profiling gate**, so the climb asks
  tpu-diagnostics for no separate capture of the running server.
- **Stop.** Only `blocked` (auth or quota a human must fix, the baseline failing twice), written
  with the exact resume condition once you hold no slot. A queued job is waiting, not blocked.
  Every gap blocked or closed ends the round, not the climb: go to step 5.

## Slots

You start no job and claim no slot: tpu-code-stack (its probe at start), tpu-forward-bench,
tpu-serving-bench, tpu-pallas and tpu-code-arm book their own from the session's slots.

- The stack hardware key is the `hardware` step 0's forward-bench result returns; record it in
  the ledger's stack block. Every later request uses it.
- **Before every delegation**, release any slot you hold: delegation is foreground, so a child
  waiting on a slot you hold never gets it.

  ```bash
  python3 {{script:common/tpu_slots.py}} release --run-id <run_id> --resource <resource> --holder <holder>
  ```

  Slot bookkeeping only. No source edits, no xprof CLI, no raw XPlane reads, no floors.

## Resume

Read the ledger. If its `run_id` is this session's, this is a compaction: continue at **Next
action**, re-sending a delegation the ledger shows waiting from its saved request. Otherwise the
old session's delegations ended with it:

1. A step-6 bench the ledger shows waiting may have left servers under the old run. Cancel them:

   ```bash
   {{runtime}} compute job reap --run-id <old run_id> --include-serving
   ```

   Only the old run's servers. No source edits, no xprof CLI, no raw XPlane reads, no floors.
2. Write the new `run_id` to the ledger. A delegation the ledger shows waiting is sent again from
   its saved request with this `run_id`: a microbenchmark from `bench/<label>.request.json`, the
   step-6 bench from `bench/e2e.request.json` (its result's `replay` when one came back).

## Baselines

Two references, never mixed:

- **The entry baseline** is where the climb starts: the source's served numbers per shape. A
  source that carries them (a serving-bench `summary.json` from the source job) is reused as is, never re-measured; a shape it lacks is measured once at entry
  (step 0). It is fixed for the whole climb and is only reported against: FINAL.md, the
  cumulative gain.
- **The last verified stack** is what each round is judged against: the stack whose step 6 and
  step 7 passed, with that step 6's numbers. Before any round passes, it is `base` with the entry
  baseline's numbers. Step 4's A/Bs use the stack head instead, measured in the same request.

## Step 0 — Baseline

Send {{agent:tpu-forward-bench}} one `forward-bench-request`, `label: base`: the `engine`, the
source job, `model_dir`, `tpu` (`topology` the source job's, `accelerator_type` the TPU query
understanding's probe found with that topology's chip count: `tpu7x` on `2x2x2` → `tpu7x-8`),
the pinned image, one arm `baseline` with no `plugin` or overrides, `plan: {top_k: 5}`,
`repeats: 2` and `profile: true`. The two boots are the noise floor every A/B is judged against.

Record from the result: baseline `T_mix` (served, modelled), its `noise_spread_pct`,
`coverage_pct`, the plan path, both job ids, `hardware` as the stack hardware key, and the
result path step 1 sends. If the job's own projection disagrees with its measured busy time by
more than ~15%, say so: every gain is projected through that model.

Then the entry baseline: record each shape's source `summary.json` in the ledger. For a shape
with none, send {{agent:tpu-serving-bench}} `action: bench`, `label: entry-<shape>`, one arm on
the pinned image with no `plugin` marked baseline, the source's workload at that shape; its
`summary_json` is that shape's entry. This is the only served baseline the climb measures.

## Step 1 — Triage

Send {{agent:tpu-performance-optimization}} one `perf-triage-request`: `engine`, `model` (the
source job's model id), `tpu` as in step 0 with `topology` from the stack hardware key,
`baseline` as step 0's result path, `source_job`, `checkout` as the `worktree` `init` returned
(the campaign repo's own work dir, on `stack`; read only), and `tried`:
one entry per lever already tested on this job, from earlier `FINAL.md` files under
`./.work/<MODEL_DIR>/hill-climb/`, and this ledger. Save it and its result beside the ledger as
`triage.request.json` and `triage.result.json`.

Record the result's `roofline`, `decomposition` and `traces` paths in the setup block. Every
later step reads its floors from that roofline result. `failed`: send it once more, then stop
`blocked`.

## Step 2 — The list

Copy one row per `gaps` entry into the ledger in rank order, and park the `no_floor` ops with
their reasons and the decomposition's "examined, not selected" ops. Write each row's hypotheses
in `gaps/<rank>-<slug>.md` beside the ledger, with its diagnosis, neighbours and evidence paths.
Drop a row only by rule (already accepted, or e2e ceiling under 0.5%) and write why.

## Step 3 — Review

Send {{agent:tpu-critical-review}} one `review-request`, `kind: claim`, before any TPU spend: the
hypothesis list as `claim`, the gap files, the decomposition, the roofline result and the baseline REPORT as
`artifacts`, the baseline job ids as `job_ids` and their spread as `noise_floor`
(`{"spread_pct": <spread>, "source": <the result it came from>}`). Drop each hypothesis it
refutes (class `refuted`, the review as its artifact) and carry its corrections into the gap file.

## Step 4 — Test each hypothesis, top to bottom

One gap at a time, in list order. Measure against the stack head (the pinned image plus branch
`stack`), not the original baseline. Each hypothesis is branch `hyp/<rank>-<slug>` off `stack`,
cut by a tpu-code-stack `branch`, each test a commit on it. An arm's `plugin` is always
"`plugin` from <ref>": the `plugin` the last tpu-code-stack action on that ref returned (`init`
or `win` for `stack`, the writer's result for a branch), or a `ref` for it. The stack arm's ref is
`stack`, a candidate's its branch. An arm still at `base` (no win yet) omits `plugin`. Copy the
object whole; never type or shorten a sha. When you hold no current one, get it from a
tpu-code-stack `ref` action. A code arm's `checkout.revision` is the `sha` of the `stack` object,
all 40 characters.

### 4.1 Re-confirm on the stack head

Read `before_ms` from the latest stack microbenchmark, `floor_ms` from step 1's roofline result,
and
`gap = before − floor`. A gap an earlier win closed to within noise is `closed-by:<rank>`.

### 4.2 Dig

Send {{agent:tpu-diagnostics}} one `profiling-request` per gap, on the latest stack arm's
`traces`: the op's time per occurrence, the 5 ops before and after it with times and shapes, its
HLO producers and consumers, and for a Pallas or Mosaic kernel its LLO analysis, each named in
`fields`. Each neighbour finding is folded into
this gap or added as row `<rank>.n-<slug>`, priced from step 1's roofline result
({{agent:tpu-roofline}} with `rows` and `cells` for one it lacks). Write the diagnosis in three
lines: where the time goes, why, what would remove it. A diagnosis that contradicts the
reviewed claim rewrites it and goes back to tpu-critical-review before the A/B.

### 4.3 Test

The levers are the vllm engine's, whose plugin is `vllm_torchtpu`.

| lever | test |
| --- | --- |
| XLA / libtpu flag, vllm-torchtpu env var | one arm per setting with `overrides: ["env.NAME=value"]` on the stack head, in the 4.4 request |
| tile, block or buffering of a Pallas kernel; a new or fused kernel | {{agent:tpu-pallas}}, then the 4.4 A/B with `plugin` from the gap's branch |
| any other code change in `vllm_torchtpu/` | {{agent:tpu-code-arm}}, then the 4.4 A/B with `plugin` from the gap's branch |
| a fix outside the `vllm_torchtpu` plugin | `blocked`, naming what it needs |

A flag sweep is a search, not a result: the winning setting gets its own 4.4 A/B at two boots.

For a kernel, release your slots, then send a `pallas-request` built from the gap: `block` from the
diagnosis; `checkout` from a tpu-code-stack `branch` with `name: hyp/<rank>-<slug>`: its
`worktree` as `path`, the branch, and `revision` the `stack` commit it starts from; `model`,
`cell` and `mesh` from the source job; `precisions` per operand; `target` with the chip and the
stack hardware key; `stack_head` as the image digest and `plugin` from `stack`; `floor` and
`diagnosis` as the result paths with one line each; `budget.jobs` (default 12); `run_id`. Copy
each `carried` item into the list as a row. tpu-pallas commits its kernel on the gap's branch
through tpu-code-stack; record its result's `plugin.sha`.
`failed` with a base mismatch is not a loss either: fix the request and re-send once, then block
the gap.

For any other code change, release your slots, then send a `code-arm-request`: `checkout` from a
`branch` as for a kernel; `gap` with its `op`, `layer_kind`, `floor_ms`, 4.1's `before_ms` as
`measured_ms`, the 4.2 diagnosis and the gap file as `evidence`; the `code` hypothesis; `engine`
and `model`; `stack_head` as the image digest and `plugin` from `stack`, omitted at `base`;
`target` as for a kernel; `budget.jobs` (default 8); `noise_floor_pct` step 0's spread; `repo` the
campaign repo; `run_id`. `changed` goes to the 4.4 A/B with its `plugin`; record its `plugin.sha`,
and on a loss there `discard` one test per entry of its `evidence.commits`. `refuted` is class
`refuted` with its `evidence`, and a `discard` with `what: hypothesis`. `blocked` blocks the gap
with its `reason`; `failed` with a base mismatch is handled as for a kernel.

You never edit source and never run git: every change to the repo is a tpu-code-stack action.

### 4.4 A/B against the stack head

Release your slots, then send {{agent:tpu-forward-bench}} one `forward-bench-request`,
`label: <rank>-<slug>-a<round>`, on step 0's plan path, `hardware` the stack key and
`noise_floor_pct` step 0's spread: a `stack` arm marked baseline (`plugin` from `stack`), and a
candidate arm per hypothesis (`plugin` from its branch, or from `stack` with the override).
Candidates of several gaps that are ready together may ride one request.

If `stack` moved after the branch was cut, send `branch` for it again first: an existing branch is
rebased onto `stack`. `conflict` is recorded and the hypothesis redone on the new stack.

1. Read each candidate arm's `speedup` and `significant`, its `guardrails` (a regressed decode or
   mixed class is a loss), and the op's `after_ms` from its evidence and traces.
2. Write `closed % = (before − after) / (before − floor)` and `gain % = speedup − 1` (served,
   modelled) with `noise_spread_pct` beside it, and the arm's `plugin.sha` and `patch` outcome
   under the stack block.
3. **Win** (verdict `win`): send `win` with the branch, record its `plugin` as the new `stack`, the
   request, result and job ids and cumulative `T_mix`. The next A/B runs on the new head.
   `conflict`: record the files and redo the hypothesis on the new stack. **Not a win**: `discard`
   the test (`what: test`, tagged `tried/…` and reset), record what was learned, back to 4.2. A
   hypothesis given up is a `discard` with `what: hypothesis` (tagged `discarded/…`).

A change to the schedule (mnbt, chunk size, spec tokens, admission) cannot be decided by `T_mix`:
park it `needs-e2e` and carry it to step 6 as its own arm.

### 4.5 When a gap is done

| class | means | artifact |
| --- | --- | --- |
| `floor` | `after` within ~10% of floor, or the rest is inside noise | the last A/B arm's `cost_model` |
| `ceiling` | what is left is worth under 0.5% served `T_mix` | the gap's e2e ceiling row |
| `refuted` | the hypothesis was wrong; the time belongs to another op, which becomes a row | the evidence or the review |
| `budget` | the gap's 12 launches spent: every TPU job and server started for it, A/B boots and writer probes alike | the spend row |
| `blocked` | the route above, or a subagent per **The loop** | what it needs |

A stop with a class other than `floor` and closed under 50% goes to {{agent:tpu-critical-review}}
with `kind: stop`, the claim, the A/B artifacts, job ids, noise floor, `termination_class` and
`budget_left`. Run its continuations as
one extension of 6 launches; a second stop on the gap is final. Then re-sort the open tail by e2e
ceiling and take the next row.

## Step 5 — Combine

When every row has a class, release your slots and send {{agent:tpu-forward-bench}} one
`forward-bench-request`, `label: combine`: a `verified` arm marked baseline (the pinned image with
the last verified stack's `plugin`, none in round 1), and a `stack` arm with `plugin` from `stack`, `plan: {full: true}`,
`repeats: 2`, `hardware` the stack key. Write `COMBINED.md`: per gap its ms, closed % and own
gain; the product of individual speedups beside the measured combined speedup (a shortfall means
wins overlap; name which); and the served gain, modelled, as
`(busy + idle) / (busy / speedup + idle)` with coverage and noise.

Send {{agent:tpu-critical-review}} `COMBINED.md` (`kind: claim`) with the job ids, the image digest,
the `stack` commit and the result's `replay`. A combined speedup that is not significant is a stop:
review it with `kind: stop` before skipping step 6.

## Step 6 — End-to-end serving bench

Read the source job's recipe back from `{{runs}}/<run_id>/artifacts/<job-id>/recipe.yaml`, fetching
it with `{{runtime}} compute job artifacts --job-id <job-id>` when absent. Write `e2e/launch.sh` beside the ledger: its server env, pre-steps and
server command, unchanged; tpu-serving-bench arms the profiler and applies each arm's
`plugin.patch`.

Then send {{agent:tpu-serving-bench}} one `serving-bench-request`, `action: bench`, `label: e2e`,
with only the `plugin` changed between arms, and save it and its result as
`bench/e2e.request.json` and `bench/e2e.result.json`:

- `engine` as recorded, `model` the source job's model id, `tpu` as in step 0, `hardware` the
  stack key, `profile: true`;
- `arms`: `stack` (marked baseline: the pinned image, `plugin` from `stack`) and one arm per
  `needs-e2e` row. No baseline arm: the last verified stack was measured when it passed;
- the workload: the source's `bench_request` (one request per shape when it has one per shape,
  `label: e2e-<shape>`), else `deployment` (`e2e/launch.sh`, the recipe's port and inputs) and
  `client_args`: recipe.yaml's lengths as `--isl-osl`, its concurrency as `--concurrency`,
  `--engine <engine>`, `--model` and `--tokenizer` as the model id; no exploratory flags.

1. **Check** the result is `measured` and each arm's `status` is `ok`. A stack arm with
   `patch.exit` 4 (`base_mismatch`) means the repo's `base` is not the pinned image's: stop
   `blocked` with its outcome. `failed`: re-send its `replay` (or the request) once as `e2e-r2`,
   then stop `blocked`.
2. **Profiles** came back as each arm's `traces`, captured by tpu-diagnostics in the delegate, with
   their `ml_diagnostics_urls`; record the links. The capture is never a measurement.
3. **Compare** the stack per cell against the last verified stack's step-6 `summary_json` (round 1:
   the entry baseline): `output_tokens_per_s`, `ttft_p50_ms` and `tpot_p50_ms`, beside the step-5
   projection. A miss over ~30% names what the microbenchmark missed. Report the same cells against
   the entry baseline as the cumulative gain. A stack that loses to the last verified stack is not
   verified: the round's wins are kept on `stack` for the next round, and the last verified stack
   stays.
4. {{agent:tpu-critical-review}} the E2E A/B (`kind: claim`) with the result, its `report`, each arm's
   `summary_json` and `bench_request`, and the `replay`.

## Step 7 — Accuracy

Send {{agent:tpu-serving-bench}} `action: eval`, `engine` as recorded, `label: accuracy`, `model`
the source job's model id, `tpu` as in step 0, `hardware` the stack key, `config` as recorded at
start, `arms` the `stack` arm (the pinned image, `plugin` from `stack`) plus, when the config has
no recorded baseline, a `baseline: true` arm on the unpatched image, `vllm_server_cmd` the
`vllm serve` command in `e2e/launch.sh`, `limit: 100`, and `run_id`. It gates against the
config's recorded score; with none, the baseline arm runs first and its score is recorded as the
config's baseline for every later round. Save both as `bench/accuracy.request.json` and
`bench/accuracy.result.json`.

1. A regression, a failed review, a `gate` other than `pass`, or an arm with `patch.exit` 4 stops
   `blocked` with the evidence; the stack is kept, and nothing is bisected.
2. Otherwise record the result under `{{runs}}/<run_id>/artifacts/hill-climb/<job-id>/` and write
   `FINAL.md` beside the ledger: each row's class and closed %, projected vs measured gain, the image
   digest, the `base` and `stack` commits, and what is still open.

## Step 8 — Image

After step 7 passes, send {{agent:tpu-code-stack}} `push` with `name: <model-slug>-hill-climb`.
`recorded` puts its `image` digest in `FINAL.md` as the next campaign's image. `skipped` is recorded
with its reason. If query understanding recorded docker unavailable or no registry, record the skip
without delegating.

Then start the next round: when steps 6 and 7 passed, the stack and its step-6 result become the
last verified stack. Increment `Round` in the ledger. Before triage, profile the stack itself: send
tpu-forward-bench step 0's request with its one `baseline: true` arm's `plugin` set to `stack`,
`profile: true` and `label: base-r<round>`, and go to step 1 with that result as `baseline` and
every row so far in `tried`. The stack keeps growing on
the same `base`; the entry baseline never changes. A round
whose triage finds no gap above its 0.5% ceiling still runs: ask for the next ranks
(`plan: {top_k}` doubled) and the shapes no round has profiled yet.

## Rules

- Measured vs modelled, always labelled. Only step 6 is a measured served result.
- Every arm is the pinned image digest plus one patch from `base`. Never `:latest`, never a build.
- Every TPU job goes through a delegate: microbenchmark jobs only through tpu-forward-bench, servers
  and serving benchmarks only through tpu-serving-bench.
- Never touch the benchmark harness, never edit source and never run git: commits come only from
  tpu-pallas and tpu-code-arm, through tpu-code-stack.
- A gap file with no neighbour section is unfinished. No stop without a class and an artifact.
