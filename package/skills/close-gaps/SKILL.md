---
name: close-gaps
description: A simple optimization loop for one served job — microbenchmark the job, rank its gaps with decompose-optimization, then work the list top to bottom, digging into each gap until it is closed (XLA / libtpu flag tuning, env knobs, a code change, a kernel via write-tpu-kernels, and the ops around it), verifying every gain with a workload-microbenchmark A/B, then stacking every win, projecting the combined gain and confirming it with one end-to-end rerun of the original job. Use when asked to "close the gaps on job j-…", "optimize this job end to end", "work the decompose list", "go deep on each problem". Runs as an autonomous loop from start to E2E with no human gates — one ledger file, stops only when done or blocked. Runs in the main thread.
scripts:
- scripts/common/serve_vllm.sh
- scripts/optimization-heuristics/hillclimb/__init__.py
- scripts/optimization-heuristics/hillclimb/code_gap.py
- scripts/optimization-heuristics/hillclimb/code_limits.py
- scripts/optimization-heuristics/hillclimb/early_verdict.py
- scripts/optimization-heuristics/hillclimb/hillclimb_cli.py
- scripts/optimization-heuristics/hillclimb/kv_replay.py
- scripts/optimization-heuristics/hillclimb/ledger.py
- scripts/optimization-heuristics/hillclimb/legs.py
- scripts/optimization-heuristics/hillclimb/lessons.py
- scripts/optimization-heuristics/hillclimb/metrics.py
- scripts/optimization-heuristics/hillclimb/microbench.py
- scripts/optimization-heuristics/hillclimb/model_plugins.py
- scripts/optimization-heuristics/hillclimb/ports.py
- scripts/optimization-heuristics/hillclimb/predictor.py
- scripts/optimization-heuristics/hillclimb/preflight.py
- scripts/optimization-heuristics/hillclimb/run_history.py
- scripts/optimization-heuristics/hillclimb/trace_roofline.py
- scripts/optimization-heuristics/hillclimb/verify.py
references:
- references/close-gaps/LEDGER-TEMPLATE.md
- references/close-gaps/serving-hillclimb-gates.md
delegates:
- tpu-diagnostics
---

# `close-gaps`

One job in, one end-to-end-verified stack of wins out. **It is an autonomous loop:** one
`/close-gaps start` runs every step below to the end without asking anyone, waits on its own jobs,
and stops only at `done` or a real blocker (see "The loop"). The plan is short:

```
0. microbenchmark the job        (baseline T_mix + profile)       {{skill:workload-microbenchmark}}
1. rank the gaps                 (decomposition.md → the list)    {{skill:decompose-optimization}}
2. for each gap, top to bottom:  dig → change → microbenchmark A/B → stack or stop
3. combine                       (whole stack vs baseline, one A/B → projected served gain)
4. E2E rerun of the original job (baseline vs stack) → {{skill:critical-review}} → {{skill:accuracy-check}} → record
```

**The key is step 2: depth on each gap.** The goal of a gap is to close it, measured `→` floor, not
to try one idea and move on. A gap is worked until it hits a termination class (see "When a gap is
done"), and the ops around it are searched on the way, because the cheapest wins are usually a
relayout, copy or un-fused neighbour sitting next to the op the profile named.

**Runs on the main thread.** It delegates to {{skill:roofline-analysis}}, {{agent:tpu-diagnostics}},
{{skill:critical-review}} and {{skill:exit-review}}, and submits its own TPU jobs with `{{runtime}} compute`;
nesting depth is 3, so it must not itself run inside one. /{{skill:write-tpu-kernels}} is a
main-thread skill too — invoke it from here, not from a subagent.

## Efficiency in Hill Climbing

* Targeted XProf & tpu-diagnosis: Profiling analysis can quickly become a bottleneck. Every XProf
  inspection must have a specific, deliberate target—running tpu-diagnosis without explicit bounds
  can trigger overly granular HLO/LLO inspections and large, inefficient data downloads.

* Judicious Full Accuracy Checks: Because full accuracy evaluations are expensive, avoid running
  them outside of the mandatory check after the final end-to-end (e2e) run unless strictly
  necessary.

* Microbenchmark usage: use microbenchmark to replace the full end-to-end (e2e) run except the final
  one.

### Serving hill-climb CPU gates (`hillclimb_cli.py`)

For vLLM-TorchTPU serving jobs, `python3 {{script:optimization-heuristics/hillclimb/hillclimb_cli.py}}`
runs deterministic stdlib checks on CPU (step mapping, flags and decision rules in
{{reference:close-gaps/serving-hillclimb-gates.md}}). They keep no state: they read past outcomes
from the ledger and gap files. They decide no wins: that stays `cost_model.json` in 2.4 and
{{skill:critical-review}} in step 4.

* **Digging & pre-launch (`arm_preflight`)** — `roofline` (tabular op floor join when raw `.xplane.pb` is not on hand; `decompose-optimization` and `roofline-analysis` stay primary on full profiles), `code-limits`, `code-gap`, `ports`, and for P/D jobs `legs` and `kv-replay` in 2.2; `preflight` before any 2.3 / `needs-e2e` serving arm (an ERROR blocks launch); `microbench` (`--repo`) to test a single kernel bucket extension with `float64` numerics before building an image for `workload-microbenchmark`; and `predict` for schedule-changing `needs-e2e` levers (`DECODE_BATCH_CAP`, speculative tokens, P/D leg balance) that fixed-step `T_mix` cannot score.
* **In-flight & post-run readings (`arm_verdict`)** — `microbench --bench-log` on 1-host kernel logs; `early-verdict`, `verify` (resolves `serving_analysis` detector IDs and slugs) and `noise` as advisory readings on step-4 arms, which still run to completion; `diagnose-failure` before a relaunch.

## Invocation

```
/close-gaps start <job-id> --model <MODEL_DIR> [--top K] [--mode scaled|mirror] [--max-jobs N] [--pause]
/close-gaps resume        # pick the loop back up from the ledger (after a blocker, a crash, a new session)
/close-gaps status        # print the ledger summary; does nothing
```

- `<job-id>`: the finished served job to optimize (the "original job"). Its recipe is what step 4
  reruns.
- `--top K` (default 5): cells in the microbenchmark plan.
- `--mode` as in /{{skill:workload-microbenchmark}}
  (scaled 2x2x1 for kernel work, mirror for sharding/collective gaps).
- `--max-jobs N` (default 200): total TPU jobs the whole loop may launch, E2E included. The loop's
  only spend brake besides the per-gap budget.
- `--pause`: opt back into two human stops — after the list (step 1) and before the E2E (step 4).
  Off by default.

## The loop

`start` and `resume` both enter the same loop and stay in it:

```
while ledger.step != done:
    read ledger                       # the state; never trust memory of earlier turns
    do the next action                # one step / sub-step below
    write ledger                      # artifacts, job ids, verdict, next action
    if jobs were launched:
        do other work that does not depend on them, else wait for them (below)
    if blocked: write the blocker, stop
```

- **Never stop to ask.** Every decision a human would make has a rule here: the list is taken as
  ranked, the E2E is launched when step 3 is done, a stop is gated by {{skill:exit-review}} (a clean-context
  agent), a win by {{skill:critical-review}}. Put the decision and its reason in the ledger and keep going.
  The user can read the ledger at any time with `/close-gaps status` from another session.
- **Waiting on jobs.** Run `{{runtime}} compute job wait --job-id <id> --until done` for the
  in-flight job ids as a background shell command, so you are re-invoked when it exits; exit code 3
  means run it again. A queued job is waiting for admission, not failed. While it runs, do work that
  does not depend on those jobs: dig the next gap (2.2), write the gap file, prepare the next flag
  sweep, read code. Never sleep in the foreground.
- **Pipeline across gaps.** Digging gap N+1 while gap N's A/B is in flight is expected. Its A/B
  must still be measured against the stack *after* gap N resolves (2.1), so build it on the stack
  head once gap N's verdict is in.
- **Context.** The run is long and will be compacted; that is safe because the ledger is the state.
  Re-read the ledger at the top of every iteration, and send bulk reading (traces, logs, code
  sweeps) to subagents so only conclusions enter the main context.
- **Failures are handled, not escalated.** A failed job: read its logs
  (`{{runtime}} compute job logs --job-id <id>`; for a server, both the server lines and the client
  lines), fix the cause if it is ours (bad flag, OOM from an override, build error) and relaunch —
  up to 2 relaunches per job. A change that will not build is a loss for that
  idea, not a blocker. Count every relaunch against `--max-jobs`.
- **Blocked = stop.** Only these end the loop early: auth or quota that needs a human (e.g.
  credentials that `{{runtime}} compute preflight` reports as invalid, no chips for > 6 h),
  the job budget spent (`--max-jobs`), the baseline itself failing twice, or every remaining gap
  `blocked`. Write `Step: blocked — <reason>` and the exact `resume` condition
  in the ledger, print a summary, stop.
- **Done.** Step 4 finished and `FINAL.md` written. Print the summary: list with classes, projected
  vs measured gain, stack branch + digest, accuracy verdict.

## The ledger: one file, the only state

`./.work/<MODEL_DIR>/close-gaps/<job-id>/ledger.md`, from {{reference:close-gaps/LEDGER-TEMPLATE.md}}.
Everything else (reports, profiles, candidate notes) lives beside it under the same directory.

- **Read it at the top of every loop iteration.** It holds the baseline, the list, each gap's
  status, the current stack (branch + head sha + image digest), the in-flight job ids, the jobs
  spent against `--max-jobs`, and the next action. A compaction or a new session loses nothing.
- **Write it after every step that produced an artifact** (a job id, a digest, a verdict). A number
  goes in only with the job id that produced it.
- Keep it under ~150 lines. Per-gap detail goes in `gaps/<rank>-<slug>.md`; the ledger keeps one
  row per gap and a link.

## Step 0 — Baseline microbenchmark

Follow /{{skill:workload-microbenchmark}} "Change → build → microbenchmark" steps 0, 3 and 4–6, for the
baseline arm only:

1. `derive <MODEL_DIR> --job <job-id>` → twin; `points <twin> --top K [--mode …]` → one plan in its
   own dir.
2. **Baseline image = vllm-torchtpu `origin/main` HEAD**, built the same way every later arm will be
   built (your team's vllm-torchtpu image build; this package ships no
   AgentX image build, so for AgentX the caller supplies the images, and a missing image is a
   blocker). Not the source job's
   image: the stack is built on HEAD, so the baseline must be too. Record `repo:tag@sha256:…` and
   the main sha.
3. Run the plan with `--profile-iters 5`. Run it **twice** (two jobs, launched together) — the
   second job is the noise floor every later A/B is judged against. `collect` both, `mix-rollup`
   both into `cost_model.json`, write `REPORT.md`.
4. Ledger: baseline `T_mix` (served basis), its spread, coverage of busy time, the plan path, both
   job ids, the image.

If the source job's own `T_mix` projection disagrees with the job's measured busy time by more than
~15%, say so in the ledger before going on: every later gain is projected through that model.

## Step 1 — Rank the gaps

Invoke a built-in subagent (`general-purpose` on Claude Code, `self` on Antigravity) that follows
{{skill:decompose-optimization}} on the baseline microbenchmark run (it has a section for {{skill:workload-microbenchmark}} jobs: per-layer-kind
join, served weighting). Its `decomposition.md` ranked table **is the list**.

Copy into the ledger, one row per candidate, in rank order: rank, slug, op/block, regimes, measured
served ms, floor ms, gap ms, e2e ceiling %, binding resource, and the route decompose named. Add
the "Examined, not selected" ops as a parked tail — neighbour work in step 2 often promotes them.

Then read the `FINAL.md` and ledger files of earlier close-gaps runs for the model, under
`./.work/<MODEL_DIR>/close-gaps/`: mark any row already tried (accepted, rejected, unverified) with
the file that records it, so a gap is not re-dug blind.

Take the list as ranked and go straight into step 2 (with `--pause`, show it and stop here once).
Drop a row only by rule — already `accepted` in an earlier close-gaps run, or e2e ceiling < 0.5% —
and write why.

## Step 2 — Close each gap, top to bottom

One gap at a time, in list order. Make `gaps/<rank>-<slug>.md` and work through the loop below.
Measure against the **current stack** (baseline + every win so far), not the original baseline:
that is the code the next change has to beat, and it means a gap a previous win already closed
shows up as closed.

### 2.1 Re-confirm the gap on the current stack

Read the op in the latest stack microbenchmark profile (the last accepted A/B's candidate arm, or
the baseline for the first gap). Record `before_ms` (served, modelled from the twin), `floor_ms`
from decompose, and `gap = before − floor`. If a previous win already closed it to within noise,
mark it `closed-by:<rank>` and move on.

### 2.2 Dig — understand why it is above the floor

Before changing anything, answer *why this op costs more than its floor* with evidence. Dispatch
the bulk reading; keep only the conclusions here.

- **The op itself.** What is it bound on vs what the roofline says it should be bound on? For an
  XLA op: the HLO (fusion kind, layouts, padding to tile, dtype converts). For a Pallas/Mosaic
  kernel: the LLO analysis from {{agent:tpu-diagnostics}} (bundles, VMEM, spills, MXU utilisation). For a collective: payload vs wait
  (decompose 5.3.3 — the excess is usually arrival skew from somewhere else).
- **Its neighbours. Mandatory.** List the ops immediately before and after it in the layer on the
  timeline ({{agent:tpu-diagnostics}} with the op as focus, ±5 ops), and its HLO producers and consumers.
  Look for: copies/transposes/relayouts feeding it, `convert`s around it, a reduction or
  elementwise op that could fuse in, an all-gather/reduce-scatter that could overlap it, the same
  weights read twice. Each finding is either folded into this gap (same change closes it) or added
  to the list as a new row `<rank>.n-<slug>` (neighbour-of), priced with its measured ms and a
  floor (ask {{skill:roofline-analysis}} for the affected rows only).
- **Across cells and layers.** Does the gap scale with the cell's work (decompose 5.4)? Is it the
  same in every layer of its kind? A gap that is fixed overhead in the light cell is a different
  fix from one that scales.
- **Code, last.** Only now read the vllm-torchtpu code that emits the op (at the stack's sha, never
  a stale branch), and cite `file:line`.

Both profile questions above go to {{agent:tpu-diagnostics}} as one `profiling-request` JSON object
per gap (see {{schema:profiling-request.json}}). Name every number you need in `fields`; the agent
returns only those. For example:

```json
{"prompt": "In the trace of close-gaps stack job <job-id>, for op <op>: its device time per occurrence, the 5 ops before and after it on the timeline with their device times and shapes, and its HLO producers and consumers. For a Pallas/Mosaic kernel, also its LLO analysis.", "run_id": "<run_id>", "traces": [{"path": "<rank-0 .xplane.pb of the job>", "label": "<job-id>"}], "report_path": "{{runs}}/<run_id>/artifacts/close-gaps/<job-id>/gaps/<rank>-<slug>/tpu_profiling_report.md", "fields": ["op_device_time_ms", "neighbour_ops", "hlo_producers_consumers", "llo_bundle_count", "vmem_usage", "spills", "mxu_utilization_percent"]}
```

Branch on its `status`. On `reported`, read the report it names; on any other status, write in the
gap file what could not be read and why.

Write the diagnosis in the gap file in three lines: *where the time goes*, *why*, *what would remove
it*. If you cannot write the *why*, keep digging; do not start changing things.

### 2.3 Change — cheapest lever first

Try the levers in cost order, and stop climbing as soon as one closes the gap:

| lever | examples | owner |
| --- | --- | --- |
| **XLA / libtpu flags** | `LIBTPU_INIT_ARGS` / `XLA_FLAGS` (fusion, layout assignment, collective offload/overlap, scoped VMEM limit), via `--override env.LIBTPU_INIT_ARGS=…` on `points` — no rebuild | this skill |
| **env knobs** | vllm-torchtpu env vars that pick a code path or kernel (`envs.py`) | this skill |
| **kernel tuning** | block sizes / tiling / buffering of an existing Pallas kernel | /{{skill:write-tpu-kernels}} |
| **code change** | remove a relayout, fuse a neighbour, change a layout or dtype, reorder for overlap, change sharding of one weight | this skill |
| **new / fused kernel** | the op (plus neighbours) as one Pallas kernel | /{{skill:write-tpu-kernels}} (main thread) |

- **Flags and env knobs are free to sweep.** One `points --override` plan per setting, all on the
  stack image, launched together. A flag sweep is a cheap search, not a result — the winning setting
  still gets the 2.4 A/B.
- **Hand a kernel to /{{skill:write-tpu-kernels}}** unattended — tell it the caller is an autonomous loop
  and to resolve its own stops by its rules rather than wait for a human — with the block, the cell shapes from the plan, the
  incumbent, the diagnosis and the floor. It returns `final_report.md` and a `carried:` list —
  copy any carried improvement into the list as a row.
- Every change is a commit on the stack branch `close-gaps/<job-id>/<rank>-<slug>` off the stack
  head, pushed (the build clones from the remote). One idea per commit.

### 2.4 Verify — microbenchmark A/B against the stack

1. Build the candidate image the same way as the stack image.
2. One job per arm, launched together, same plan: stack image vs candidate image
   (`forward-cmd --steps <plan> --image …`). Then `collect` and
   `mix-rollup --plans <plan> <candidate> --baseline <stack>`.
3. Read from `cost_model.json`: speedup and `significant`, per-class deltas, **guardrails** (a
   regressed mixed/decode guardrail is a loss even when `T_mix` wins), and the op's `after_ms`.
4. Compute and write in the gap file:
   - `closed % = (before − after) / (before − floor)` — how much of the gap this closed;
   - `gain % = speedup − 1` on served `T_mix` (modelled), with the noise beside it.
5. **Win** (significant, no guardrail regression) → merge into the stack branch, new stack image,
   ledger: stack sha, digest, the A/B job ids, cumulative `T_mix`. **Not a win** → revert, record
   what was learned in the gap file, back to 2.2 with it.
   Either way, and for a job that crashed, add one line to the gap file: `- outcome: win`,
   `not a win` or `crash`, then each changed knob or env flag as a backticked `KEY=VALUE`.

If the change alters the schedule (mnbt, chunk size, spec tokens, admission), `T_mix` cannot decide
it — park it as `needs-e2e` and carry it to step 4 as its own arm.

### 2.5 When a gap is done

Keep looping 2.2 → 2.4 until one of these holds, and write the class and its artifact in one line
on the gap's ledger row:

| class | means | artifact |
| --- | --- | --- |
| `floor` | `after` within ~10% of floor, or the remaining gap is inside the A/B noise | the last A/B's `cost_model.json` |
| `ceiling` | what is left of the gap is worth < 0.5% served `T_mix`, below what the next gap offers | the gap's e2e ceiling row |
| `refuted` | the diagnosis was wrong (the floor does not apply, or the time belongs to another op — then that op becomes a row) | the evidence that refuted it |
| `budget` | the gap's budget is spent | the ledger spend column |
| `blocked` | needs something unavailable (real weights, a served run, an upstream fix) | what it needs, named |

- **Budget per gap:** 6 A/B rounds or one {{skill:write-tpu-kernels}} run, whichever comes first. Raise it
  explicitly in the ledger if the gap is still closing fast.
- **Anything but `floor` with a closed % under 50% → {{skill:exit-review}} before moving on.** A stop is a
  claim; it gets a clean-context review that recomputes the floor and noise and either returns
  priced continuations or `EXIT JUSTIFIED`. Do the continuations as **one** budget extension
  (3 more A/B rounds); a second stop on the same gap is final, so the loop cannot circle forever.
- Then go straight to the next row. New neighbour rows are worked when their rank comes up —
  re-sort the open tail by e2e ceiling after each gap.

## Step 3 — Combine and project

When every row has a termination class:

1. The stack head is already the combination. Run the **full plan** (and, if the list touched cells
   outside the top K, the per-cell plans for full coverage) on the original baseline image and the
   final stack image, two jobs per arm, launched together. `mix-rollup` both arms.
2. Write `COMBINED.md`:
   - per-gap row: gap ms, closed %, individual gain (from its own A/B);
   - the product of individual speedups vs the measured combined speedup — a shortfall means wins
     overlap (same time saved twice) or interact; name which;
   - the combined served gain, **modelled**: `(busy + idle) / (busy / speedup + idle)` from
     `cost_model.json`, with coverage and noise.
3. {{skill:critical-review}} on `COMBINED.md` with the job ids, image digests and the `recipe.yaml`
   read-back of every job. It recomputes; fix what it finds.

Then launch step 4 directly (with `--pause`, stop here once). Skip the E2E only if the combined
speedup is not significant — that is a stop, so {{skill:exit-review}} it first.

## Step 4 — E2E rerun of the original job

1. Read back the original job's `recipe.yaml` verbatim: the resolved server and client commands
   and env, in `{{runs}}/<run_id>/artifacts/<job-id>/recipe.yaml`. If that directory does not hold
   it yet, fetch it with `{{runtime}} compute job artifacts --job-id <job-id>` and unpack
   `outputs.tgz` there (`tar -xzf outputs.tgz`), which also gives `job.log` and `outputs/`
   (`vllm.log`, `bench.log`, `metrics.json`, ...). Rerun it twice with **only the image changed**:
   once on the baseline (HEAD) image, once on the stack image, launched together so they share a
   day's noise. Each `needs-e2e` row from step 2 is a third arm. Each arm is two jobs:

   ```bash
   # <arm>/serve.sh: the server env and pre-steps from recipe.yaml, unchanged; its
   # `vllm serve <args>` line becomes `exec bash serve_vllm.sh <args>`
   {{runtime}} compute submit --recipe <arm>/serve.sh --interpreter bash --serve \
     --input {{script:common/serve_vllm.sh}}:serve_vllm.sh \
     --image <arm-image> --name <label>-server
   # <arm>/bench.sh: wait for http://<server_job_id>:<port>/health, then the client command from
   # recipe.yaml pointed at that address; it packs its logs and metrics into outputs.tgz
   {{runtime}} compute submit --recipe <arm>/bench.sh --interpreter bash \
     --colocate-with <server_job_id> --colocate-requests cpu=<n>,memory=<size> \
     --output outputs.tgz --name <label>-client
   ```

   Wait on the client with `{{runtime}} compute job wait --until done`, fetch it with
   `{{runtime}} compute job artifacts`, then `{{runtime}} compute job cancel` the server: a
   `--serve` job is not reaped at session end and holds the chips until it is cancelled.
   (If the active compute backend does not support `--colocate-with`, establish a local port-forward
   to the server container's HTTP and profile ports and run `bench.sh` against `127.0.0.1` instead.)
   The server starts through `serve_vllm.sh`, the launcher every vLLM server in this package uses,
   so close-gaps and the other skills start servers the same way.
   **Every arm runs with the profiler armed by default:** `serve_vllm.sh` arms the phased profiler
   (`USE_PHASED_PROFILER=true`, `PYTHON_TRACER_LEVEL=0`, `max_iterations=5`, and
   `--xla_xprof_register_llo_debug_info=true`), writing traces under `VLLM_PROFILE_DIR`
   (default `artifacts/vllm_profile`) and serving them on port `8790`. Measure the reportable
   metrics with a clean client pass, and delegate profile capture and analysis of each running
   server arm to `tpu-diagnostics` (passing `server: {job_id, port, files_port, load}`). The
   profiling pass is slower and is never a measurement. An E2E without a trace answers "how much"
   but not "where": when the measured gain misses the projection, the trace is what names the op,
   and a rerun to get it costs another 2.5–4 h. Capture on **all** arms, never one, so the arms
   differ only in the image. Check that `tpu-diagnostics` verified `/device:TPU:` planes on each
   arm's traces before reporting the A/B.
2. Measured served gain = stack vs baseline throughput (and TTFT/TPOT medians). Put it beside the
   step-3 projection; a gap over ~30% between them means the microbenchmark model missed something
   — name what (orchestration, routing skew, a deviation the twin listed).
3. {{skill:critical-review}} the E2E A/B. Then {{skill:accuracy-check}} the stack image
   (`mmlu_pro --limit 100`, 1.5pp gate, against the baseline {{skill:accuracy-check}} records for the model)
   with the image and `VLLM_SERVER_CMD` read back from the stack E2E job's `recipe.yaml`.
4. Record the E2E result in the run's artifacts directory, `{{runs}}/<run_id>/artifacts/close-gaps/<job-id>/`
   — each win with its A/B evidence, the stack as the combined result. Only now is anything
   `accepted`. In the ledger, add an `- outcome:` line (as in 2.4) for each `needs-e2e` arm.
5. Write `FINAL.md` beside the ledger: the list with each row's class and closed %, the projected vs
   measured gain, the stack branch and digest, and what is still open with its route.

## Rules

- **Measured vs modelled, always labelled.** Twin times are measured; served times from the twin,
  the combined projection and every floor are modelled. Only step 4 is a measured served result.
- **Every arm is a named, digest-pinned image built from a pushed commit.** Never `:latest`, never
  a local-only commit.
- **The A/B baseline is the current stack**, not the original baseline, in step 2.
- **Never touch the benchmark harness** (`<vllm-torchtpu>/scripts/vllm/benchmark/**`, `<vllm-torchtpu>/benchmarking/**`).
- **Neighbours are part of the gap.** A gap file with no neighbour section is unfinished.
- **No stop without a class and an artifact.**
