---
name: decompose-optimization
description: Turns a benchmark job with an xprof capture into a ranked list of optimization candidates, each diagnosed down to its root cause. Use for "optimize <model>", "what should we optimize in job <id>", "where is the time going", "profile this and rank the problems", "triage this model". Inventories every capture in the job, takes the model structure and the per-op floor from roofline-analysis, reconciles the observed op census against that structure, joins measured device time against the floor, then diagnoses each candidate by its distribution across layers, ranks and phases. Ranks and diagnoses. Proposes no fix, changes nothing, measures nothing new, logs nothing.
scripts:
- scripts/decompose-optimization/inventory.py
- scripts/decompose-optimization/q.py
- scripts/roofline-analysis/roofline.py
references:
- references/decompose-optimization/decompose_template.md
- references/decompose-optimization/candidate_report_template.md
delegates:
- tpu-diagnostics
host_dependencies:
- duckdb
- pandas
---

# decompose-optimization

Input: a finished benchmark job with an xprof capture, plus the model checkout.
Output: `decomposition.md`, a ranked list of operators, each with a problem statement and its
diagnosed root cause.

Every number in the output that comes from the roofline is **modelled**. Nothing here is a result.
Never record anything this skill produces as an experiment result.

The basic worflow of this work is to compare the information from xprof and roofline. You must get
the information from this two side first, find the candidate ops and then diagnose it. You must using
the comparison to zoom the scope of the further analyzing. The reading of source code, or searching 
of previous information etc, are only allowed for the specific ops diagnosing.

Make sure that re-capture itself cannot be suported as a candidate. You cannot just say it requires
recapture as a candidate unless the data you need is really not obtainable without re-run the benchmark.

Do not spend time on how to fix the problem. No report contains a fix, a fix direction, a fix
cost or a routing to a fix skill. The output stops at the diagnosed cause and its problem class
(Step 5.6).

## The two inputs this skill joins

| Source | Gives | Step |
| --- | --- | --- |
| The xprof capture, exported as an events parquet by {{agent:tpu-diagnostics}} and queried with {{script:decompose-optimization/q.py}} | what every op **actually cost**, per occurrence, per rank, per phase | 1, 3, 5 |
| {{skill:roofline-analysis}} | what every op **should have cost**, and the model's structure: the op list per layer, the layer type map, the shapes | 2 |

Neither alone answers the question. The profile says where the time went, the roofline says where it
should have gone, and only the join says what to work on: a block at 40% of step time sitting at 92%
of its floor is worth less than a collective at 8% that is entirely exposed.

The roofline is also the source of the **structural prediction** used in Step 3. Do not re-derive
the op list, the layer map or the shapes by hand.

## Run it this way

Orchestrate from the thread that runs this skill, whether that is the main thread or a subagent.
Dispatch every bulk operation to a subagent: the roofline, the trace downloads, the op census, the
source searches. Keep only tables in the orchestrating thread.

Unless a step names a package agent, "subagent" in this skill means a built-in subagent:
`general-purpose` on Claude Code, `self` on Antigravity. A built-in subagent is not a package
agent, so the "Direct delegation targets" line in your instructions does not forbid it. That line
lists only the package agents you may call.

The critical path is:

```
recipe read-back ─┬─> roofline (2a+2b, existing config) ──────────────┐
                  └─> events parquets (2 ranks/phase) ─> inventory.py ─> windows.csv ─┬─> opmap (3a) ─> join (3b) ─> Step 4 ─> Step 5 (parallel, writes reports) ─> decomposition.md
                                                                                      └─> other ranks: span-and-busy (second export request)
```

The roofline and the operator mapping (Step 3a) run **in parallel**. Neither needs the other until the join in Step 3b. 

Use {{script:decompose-optimization/q.py}} for all trace queries. XProf's own events query command
serialises concurrent invocations through a shared cache and deadlocks; `q.py` reads the same
parquet directly.

Use {{agent:tpu-diagnostics}} and {{skill:roofline-analysis}} to get floor and real measured time for ops.
Roofline-analysis is well written and you should not micromanage it. Two different things are
often confused here:

- **Do not send a modelled number back to argue with it.** The floor is guidance for diagnosis, not
  a result. Re-running it to get a number you prefer is micromanagement.
- **Do send back an op it did not model, or an operating point Step 3 changed**, and ask only for
  the affected rows. Step 3 and Step 4.1 both require this. It is a scoped request of a few
  minutes, and skipping it leaves an observed op with no floor, which means no rank.

You must have {{agent:tpu-diagnostics}} export all ranks and phases when available, but only two ranks
per phase are on the critical path. Request those two first and start Step 1.3 on them. Request the
remaining files in a second export request. Their span-and-busy check must return before Step 3
closes. 5.5 uses them in Step 5.

You should never read the source in advance to guessing any optimization strategy. Source code is
only used for when you diagnose specific ops to help you find the real problem. You should never 
guess optimization directions from source code in advance. The problems appeared in final candidates
can only from the real xprof trace analysis. Never write a candidate that are just your model from
code.

## Inputs the caller must supply

In this skill, `<job_dir>` is `{{runs}}/<run_id>/artifacts/<JOB_ID>/`, the job directory that
`{{runtime}} compute job artifacts` fills for the serving job: `recipe.yaml` (the resolved command and env), `job.log`, and
`outputs/` (`vllm.log`, `bench.log`, `metrics.json`, and `profile/` with the traces). If it does not
hold them yet, fetch them with `{{runtime}} compute job artifacts --job-id <id> --out-dir <job_dir>`
and `tar -xzf <job_dir>/outputs.tgz -C <job_dir>`.

| Item | Where to get it | If missing |
| --- | --- | --- |
| Job id with an xprof capture | the caller | stop and ask |
| Model checkout path and commit, or the image | the image in `<job_dir>/recipe.yaml` | stop and ask |
| Workload: input/output length, concurrency, recipe | read back verbatim from `<job_dir>/recipe.yaml` | stop and ask; never assume a default shape |
| Sharding: TP/DP/EP degrees and mesh | the server command in the recipe read-back | stop and ask |
| Precisions: weights, activations, KV cache, collectives | the server command and `config.json` | carry as `unknown`, state it |

Do not substitute a plausible value for a missing one. A ranking built on a guessed sharding ranks
a problem that does not exist.

## When the job is a {{skill:workload-microbenchmark}} run

A microbenchmark job is not a served model. It runs a **twin**: the served model with one layer
per layer kind (GLM-5.2: 3 layers standing for 78), on dummy weights, with no scheduler, and
several step shapes back to back in one profile. Every step below changes. **Read the run's
`REPORT.md` in full before Step 1**, then its per-job files (`plan.steps.json`, `evidence.json`,
`xprof_steps.json`, `result.json`, `cost_model.json`). If the run has no REPORT.md, stop and ask
for one; the microbenchmark skill writes it.

Take these from the REPORT and its files, never from the job's `recipe.yaml` (the job runs the
microbenchmark's forward recipe, not a server):

| item | where | why it matters |
| --- | --- | --- |
| twin layers, their kinds, and served layer count per kind | REPORT "what the twin is", `evidence.json` | the roofline prices the served model; the join is per layer kind, not per step |
| mode: mirror (served sharding on the served slice) or scaled (2x2x1, split dims shrunk) | REPORT, plan `sharding` | scaled mode has deviations (collective degree, DCP merge width, routing statistics); a floor at served degree does not apply to a scaled collective |
| deviations table | plan `deviations`, REPORT | every deviation is an op whose twin cost is not its served cost |
| the shapes in the profile, in order | plan `top` / `points`, `profile_order`, REPORT profile step map | one job holds several shapes; each is its own regime |
| each shape's weight | `cost_model.json` (steps × step time per cell), REPORT step mix | replaces the ISL/OSL weights of Step 4.2; coverage below 100% means the rest of busy time is unpriced |
| MTP drafter on or off | plan `twin.run_drafter`, REPORT setup | off means the drafter's served cost is absent from the trace |
| image and commit | REPORT, `result.json` `image_source` | the code to read when diagnosing; never a local checkout's branch |
| ranks captured | REPORT (mirror: pod 0 only, ranks 0–7 of 16) | bounds every cross-rank and barrier conclusion (5.3.3, 5.5) |
| noise | REPORT repeat spread and `profile_spread` | the Step 4.5 noise floor |

What changes in the steps:

- **Step 1: regimes are the plan's shapes, not phases.** Every decode shape runs the same compiled
  program (one bucket), so the op mix does not change between them and a histogram will not
  separate them. Split the trace by the step boundaries in `xprof_steps.json` (`start_ms`, label,
  `program`) and the `profile_order` labels, one regime per shape. Inside a step, split layers by
  the boundary rule in `model.py` `TRACE` (the census in `evidence.json` checks it). Treat the
  first profiled step of each shape as possible warmup (Rule 3.2).
- **Step 2: one roofline phase per shape**, at that shape's served operating point (tokens per step,
  batch, context, chunk on prefix), with the twin's deployment and env. Ask for **per-layer-kind**
  floors, not only per-step totals.
- **Step 3: predicted counts are twin counts.** An op of kind k appears once per twin layer of kind
  k, not `n_k` times. Ops outside the layers (embedding, LM head, sampling glue) appear once per
  step in both.
- **Step 4: join per layer kind, then scale.** For each op in twin layer kind k:
  `served_ms = n_k × measured twin-layer ms`, joined against `n_k × floor of one layer of kind k`.
  Per-step ops count once. Rank on the served numbers, weighted by each shape's share of busy
  time from `cost_model.json`. Never rank on the twin's wall step: a per-layer saving is `n_k`
  times larger served, and a per-step one is not.
- **Label every number twin (N layers), measured, or served, modelled.** Never mix them in one
  column. A served figure from a twin is modelled even though its inputs are measured.
- **Name what the twin cannot show:** routing skew and EP imbalance (dummy weights; the decode
  MoE layer can flip between per-boot modes), host and scheduler overhead, the drafter if it was
  off, and any listed deviation. A candidate resting on one of these is `blocked` on a served run,
  not ranked.

---

## Step 1 — Inventory every capture before forming any hypothesis

Do not query for an op until this step is complete. Skipping it is the single most common way this
analysis fails.

1. List every trace in the job: `find <job_dir>/outputs/profile -name '*.xplane.pb'` (glob
   `<phase>/plugins/profile/*/*.xplane.pb`: a phase can span more than one timestamp directory).
   Record the phase label and rank of each file. Export **at least two ranks per phase label**. If only one rank carries
   device planes, record it as a capture defect now, because every cross-rank test below will be
   unavailable. For each file, list the device planes it contains (`SELECT DISTINCT device`). One
   process can hold several cores. Count captured cores, not files with device planes, and give
   that count to 5.5.
2. Send {{agent:tpu-diagnostics}} one `profiling-request` JSON object (see
   {{schema:profiling-request.json}}) that exports the events parquet of each file:

   ```json
   {"prompt": "For each trace below, build XProf's events database: the events parquet with one row per event and the columns device, category, kernel_name, start_ns, end_ns, self_time_ns, input_tensors and output_tensors, with no event limit. Write it to <workdir>/events/<label>.parquet and list it in the trace's tool_outputs. Report the device planes each trace holds.", "run_id": "<run_id>", "traces": [{"path": "<job_dir>/outputs/profile/prefill_only/plugins/profile/<ts>/rank0_<job>.xplane.pb", "label": "prefill_only_r0"}, {"path": "<job_dir>/outputs/profile/decode_only/plugins/profile/<ts>/rank8_<job>.xplane.pb", "label": "decode_only_r8"}], "report_path": "<workdir>/tpu_profiling_events_report.md", "fields": ["events_parquet_path", "event_count"]}
   ```

   Branch on its `status`. On `reported`, check that every parquet exists and that each trace's
   `coverage` matches the device planes you expect. On `failed` with `trace_empty_or_partial`,
   record the file as a capture defect. On any other status, stop and report what it returned.
3. **Run `inventory.py` once on the two ranks per phase.** It does the work of the three
   queries below in one pass, in about 1 s per file: it finds the step module, lists every step
   (wall ms, busy ms, op count) per TensorCore device, groups steps into step types, and prints a
   census of the median step of each type with HLO instances merged per kernel and shape.

   ```bash
   python3 {{script:decompose-optimization/inventory.py}} --top 15 \
       --out <workdir>/steps_inventory.csv \
       prefill_only:r0=<parquet> prefill_only:r10=<parquet> decode_only:r8=<parquet> ...
   ```

   Use the same Python environment as `q.py`. It needs `duckdb`.

   Name the regimes from its step-type tables and write `windows.csv` from the rows of
   `steps_inventory.csv`. Do not re-run the census by hand. Use the SQL below only for a question
   the script does not answer, for example a sub-step histogram.

   The three queries, run on **two ranks per phase**, with only the span-and-busy
   query on the remaining files. Regime boundaries are a property of the phase, not the rank, so
   segmenting every file re-derives one answer once per rank. Span-and-busy on the rest is what
   confirms they hold the same regimes. Per-op cross-rank comparison is Step 5.5, and it runs on
   candidates, not on every op in every file.

   Span and busy time per device:
   ```sql
   SELECT device, count(*) n, min(start_ns)/1e6 t0_ms, max(end_ns)/1e6 t1_ms,
          sum(self_time_ns)/1e6 busy_ms
   FROM Events GROUP BY 1 ORDER BY 1
   ```

   Op census on the compute device:
   ```sql
   SELECT kernel_name, count(*) n, sum(self_time_ns)/1e6 tot_ms,
          median(CAST(end_ns AS BIGINT)-CAST(start_ns AS BIGINT))/1e3 med_us
   FROM Events WHERE device='<compute device>'
   GROUP BY 1 ORDER BY tot_ms DESC LIMIT 40
   ```

   Regime histogram — bucket the span into ~200 bins and watch the op mix change:
   ```sql
   SELECT CAST(start_ns/ <bin_ns> AS BIGINT) bin, kernel_name,
          count(*) n, sum(self_time_ns)/1e6 ms
   FROM Events WHERE device='<compute device>'
   GROUP BY 1,2 QUALIFY row_number() OVER (PARTITION BY bin ORDER BY ms DESC) <= 3
   ORDER BY bin
   ```

4. Segment each file into **regimes**. A regime boundary is where the set of dominant kernels or
   the event rate changes. Name each regime by its dominant ops and by the batch dimension read
   from the tensor shapes, then map it to a model phase yourself.

5. Write down the **analysis window** for each regime you will study: file, device, `[t0, t1]`, and
   the number of steps it contains. Every later number is reported against a named window.

**Rule 1.1.** The phase label in a capture filename records when the profiler was armed, not what
the file contains. A file labelled for one phase routinely contains a leading region of another
phase, with different shapes and a different op set. Never scope a query to a file because of its
name; scope it to a regime you found in step 1.4.

**Rule 1.2.** A time filter written to answer one question will silently exclude the answer to the
next. Restate the window in every query, and check the returned event count against the inventory.

---

## Step 2 — Get the structural prediction and the floor from /{{skill:roofline-analysis}}

The prediction does not come from an ad-hoc source read. It comes from the roofline, which already
enumerates every op per layer with its shapes, its bytes and its FLOPs. Run it here, before any
per-op timing is examined, not later when you need the floor.

### 2a. Build the configs. Dispatch this alongside Step 1, not after it.

Only the operating point depends on Step 1. Everything expensive in a roofline config is the model
block, the numerics read from the checkout, and the hardware pinning, and all three come from the
"Inputs the caller must supply" table. Dispatch the config build as soon as those inputs are in hand,
concurrently with the trace downloads and the census. Serialising it behind Step 1 adds about six
minutes to the critical path and changes no number.

**One dispatch per distinct `(model, numerics, sharding, hardware)` tuple, not one per regime.** A
roofline config carries a `phases` list, so a regime that runs the same code path at a different
operating point is one more entry in that list, not one more dispatch. Five prefill operating
points are one dispatch. Regimes need separate configs only when the forward path itself differs,
for example a different MoE dispatch strategy, a different fused-activation path, or a different
activation dtype in decode. Those fields are top-level in the config schema and cannot vary by
phase, which is the only thing that forces a second config.

State two things in the dispatch:

- **Start from an existing config** among the configs {{skill:roofline-analysis}} ships if one
  exists for this model and slice. Check it against the **recipe read-back**: weight, activation and KV dtypes, the
  TP/DP/EP degrees, and the MoE and attention environment flags in `VLLM_SERVER_CMD`. Read the
  checkout only for a field the recipe changes. Do not rebuild the config from source. A cached
  config can carry a stale activation dtype or an unresolved hardware conflict, and the recipe
  check finds both. Require the reply to name every field changed and why. On 2026-09-30 a config
  for the same model and slice existed and the subagent rebuilt it from source anyway; the
  roofline took 12.6 minutes against about 5 minutes for a reuse.
- **Pass inputs only**: job id, recipe read-back path, config path, operating points. Do not paste
  instructions that {{skill:roofline-analysis}} already states.
- **Name the ops the fused kernels hide.** If the census or the existing config shows a fused
  kernel that `roofline.py` cannot model directly (for example a fused EP MoE kernel that contains
  dispatch, GEMMs and combine), say so in the dispatch so the supplement is priced in the first
  pass and not in a second dispatch.
- **Supply the previous dispatch's `config.json`** on every dispatch after the first, and require
  the returned config to differ from it only in the fields whose code path actually differs. Two
  configs for one job that diverge on a field nobody chose is a defect, and it is invisible unless
  they are diffed.

### 2b. Run it, once Step 1 has named the regimes

Set each regime's `phases` entry from the shapes in the capture: its token count, KV length and
batch read from the trace, not the workload's nominal shape.

Require these back. Do not continue without them.

| Required from the roofline | Consumed by | How |
| --- | --- | --- |
| Section 1, model structure: layer count, and the **layer type map** — which layer indices run which optional block, with period and offset if it is periodic | Step 3, Step 5.3.1 | in the reply |
| Section 3, layer-by-layer: every op in a layer, with global and local operand and result shapes and dtypes | Step 3 count and shape reconciliation | **by path.** At 78 layers this is a large CSV. The Step 3 subagent reads it; it never enters the orchestrating thread |
| Section 4, operation table: per-op HBM bytes, FLOPs, collective volume, `T_mem` / `T_comp` / `T_comm`, and which term binds | Step 4 floor, Step 5.2 | in the reply |
| `roofline_data.json` | Step 3 and Step 4 join, so both joins are table operations and not arithmetic by hand | by path |
| `config.json` | the diff in 2a, and the floor audit. Without it you cannot tell which dtypes were pinned from which were defaulted | by path |

List every returned `config.json` path in the report's provenance section. `.work/` is gitignored,
so a config worth reusing has to be promoted deliberately; naming it is the minimum that makes that
possible.

Two things a roofline cannot give you. Read these from the source yourself:

1. **Every conditional gate in the forward path that can skip work** — the `if`, its condition, and
   the value of that condition for the captured workload. A roofline prices the work that runs; it
   cannot tell you that a gate switched a block off in the regime you captured. A capture taken with
   a gate closed is unrepresentative, and you must know that before you rank, not after.
2. **Which collective runs on which mesh axis**, if the roofline left the mapping implicit.

If the roofline's layer map disagrees with the source, the source wins and the roofline was built on
a wrong structure. Send it back before continuing; every floor below it is wrong too.

Output of this step: one row per op —
`op → layers it runs on → count per step → local operand and result shapes → floor ms`.

---

## Step 3 — Reconcile observed against predicted. This is a gate.

Join the Step 1 census to the Step 2 prediction **as a table operation, in a script**, against
`roofline_data.json`. Step 4 already requires this of the floor; the census join is the same join
one column earlier, so write one script that emits observed count, predicted count, their ratio,
measured ms and floor ms in a single pass. Comparing forty ops by hand is a manual pass that the
same script removes.

**Write the operator mapping once, covering every regime in the job.** The join needs a rule set
turning kernel names, source lines and operand shapes into operator names, and it is the most
error-prone part of this step. A second mapping written later for a second phase duplicates most
of the first and can disagree with it about which kernel is which operator, with nothing comparing
the two. Emit two artifacts and record their paths:

| artifact | holds | consumed by |
| --- | --- | --- |
| `opmap.sql` | the kernel → operator rules, all regimes in one file | Step 4 join, every Step 5 diagnosis |
| `windows.csv` | one row per regime: file, device, `t0`, `t1`, step count, step ms | Step 4 join, every Step 5 diagnosis |

Give both paths to every Step 5 subagent. A subagent that re-derives the window or the operator
mapping is redoing Step 1 and Step 3 inside Step 5, and its numbers are then not comparable with
the ranked table.

### 3a. Write `opmap.sql` while the roofline runs

The operator mapping needs the census and the list of operator names. It does not need any floor.
Start it as soon as `windows.csv` exists, in parallel with the roofline dispatch:

1. If Step 2a found an existing config, get the operator names in the orchestrating thread, in
   under 1 s: `python3 {{script:roofline-analysis/roofline.py}} <config> --csv <workdir>/ops_names.csv --no-layers`.
   These are the names `roofline_data.json` will carry, unless the roofline reply changes the
   model structure. If no config exists, the subagent maps kernels to operators named from the
   source lines and shapes in the census, and Step 3b renames them to the roofline rows. Do not
   wait for the roofline to start the mapping.
2. Dispatch one subagent with `windows.csv`, `steps_inventory.csv`, `ops_names.csv` and the
   parquet paths. It writes `opmap.sql` covering every kernel above 0.1% of step time in every
   regime, and reports the unmapped remainder in ms per regime.

### 3b. Join when `roofline_data.json` returns

One script reads `opmap.sql`, `windows.csv` and `roofline_data.json` and emits, per regime and
operator: observed count, predicted count, ratio, measured ms, floor ms, headroom, ceiling, and
the max/median and cross-rank columns of Step 4 item 6. It is a table operation of a few seconds.
It runs in the orchestrating thread or in a short subagent. It never rebuilds the mapping.

Act on the discrepancy as follows.

| Observation | What it means | What to do |
| --- | --- | --- |
| observed count = predicted | consistent | proceed |
| observed = k × predicted, k integer | one kernel name covers k distinct ops | split by result shape, then operand shape, then `hlo_op`, until each group matches a predicted row. Only then compute any average |
| observed count is 0 | **your window is wrong, not the workload** | search that op name in **every** trace file of the job. If it appears anywhere, return to Step 1 and re-derive the window. Only when it is absent from every file may you write "this op does not run", and you must then name the conditional gate recorded in Step 2 that skipped it |
| observed > 0 but not predicted | the roofline's structure is wrong, or it did not model this op | collect **every** such op for a regime and send them back to {{skill:roofline-analysis}} in one request with the operating point, then repeat Step 3. One request per op wastes a dispatch each time; one request for ten ops costs about the same as one for one |

**Rule 3.1.** Normalising by a count you assumed rather than a count you computed manufactures or
hides a k-fold effect. Compute the divisor inside the query, in the same pass as the sum.

**Rule 3.2.** Exclude warmup. For each top kernel, compare the duration distribution in the first
10% of the window against the remainder. If the early median differs by more than 2×, the early
region is warmup: shrink the window, state that you did, and never use a warmup duration in a
ranking or a root cause.

**Rule 3.3.** If two regimes of the same phase exist with different costs, **analyse the expensive
one**. Say in the report which regime every number belongs to.

---

## Step 4 — Join against the roofline and select candidates

1. The floor is already in hand from Step 2. Do not dispatch the roofline again. Send it back only
   if Step 3 changed the operating point or revealed an op it did not model, and then request only
   the affected rows.
2. **Name the target metric and the regime weights before joining.** The target metric comes from
   the request: e2e time or throughput (default), TTFT, or TPOT. For each regime `p` found in Step 3,
   state its weight `w_p`, the share of the target metric's time it accounts for:
   - e2e time: `(steps of p per request × wall ms per step of p) / Σ over regimes`.
   - TTFT: prefill regimes only. TPOT: decode regimes only.

   Label every weight measured or estimated and give its source (server log, chunk schedule,
   ISL/OSL arithmetic). A regime whose weight is unknown is priced at its largest plausible weight,
   not dropped. Padded and idle regimes (ranks padded to the DP-max bucket) are regimes too.
3. For each op in the census compute, per step, in the named window, **in every regime**:
   - `measured_ms` — measured self time (measured)
   - `floor_ms` — modelled floor (modelled)
   - `headroom_ms = measured_ms − floor_ms` (modelled)
   - `share = measured_ms / step_ms` (measured)
   - `ceiling_p = headroom_ms / step_ms` of regime `p` (modelled)
   - `e2e_ceiling = Σ_p w_p × ceiling_p` (modelled) — **the ranking key**. An op present in several
     regimes sums its contributions.

   Never rank on one regime's ceiling. An op that recovers 80% of a phase holding 30% of the time is
   worth 24% e2e; an op that recovers 5% of a phase holding 70% is worth 3.5%. The most expensive
   phase is not where the largest win is.

   Join against **the operating point the capture actually holds**, and only that one. A Step 2
   config priced at several operating points has one that matches the window; the others exist to
   bracket an op the capture could not observe. Joining measured time against those produces joins
   with no measured side. State in the report which operating point each floor came from, and use
   a bracketing point only in the one place it answers something. Each regime joins against its
   own operating point.
4. **Custom calls (Pallas, Mosaic, Tokamax) report 0 FLOPs and 0 bytes to XLA.** Never take FLOPs,
   bytes, MXU% or roofline position for them from XProf's overview, KPI or roofline-model tools,
   as {{agent:tpu-diagnostics}} reports them. Their floor comes only from {{skill:roofline-analysis}}
   modelling the kernel's mathematics. Never write "0 FLOPs, memory bound" for such an op.
5. Select as a candidate every op that clears **both** the benchmark's noise floor and 1% by
   `e2e_ceiling`. Cap the list at about 10.
6. **Additionally select, regardless of ceiling:** any op whose `max/median` across occurrences
   exceeds 2, and any op differing by more than 20% between traced ranks, **tested in every
   regime**. A list ranked on totals alone hides every non-uniform op, and non-uniform ops are where
   the large recoverable time is. An op selected here gets its own shortlist row. Never fold it into
   another candidate's row; a merged op is never distributed on its own and its structure is lost.

   Compute `max/median` per **kernel signature**: kernel name with the `.N` instance suffix
   removed, plus result shape. An operator that covers several signatures (for example a 1024-token
   and a 16-token variant) mixes two distributions, and its max/median then measures the mix, not a
   non-uniform op. The Step 3b join script emits this column, so this item costs no extra query.
7. **Cut the shortlist to at most ten before Step 5.** The cut happens here, once, in the main
   thread. Fill the slots in this order:
   1. **Full reports, by `e2e_ceiling`.** Take the rows that pass item 5, highest first.
   2. **Estimated-weight slot, regardless of item 5.** Any op whose ceiling is 10% or more in a
      regime whose weight is estimated (not measured) gets a full report. Its report states the
      weight at which it would enter the top five. Reason: its rank depends on the weight estimate,
      and a rank cannot show that. Example: a routing all-gather at 48% of the padded-rank step
      ranked 6th at an estimated padded weight of 0.02; it enters the top five at 0.031.
   3. **Short reports.** A row outside the top five by `e2e_ceiling` that has no 5.3.0 outlier, no
      max/median above 2 and no rank spread above 20% in any regime gets a short report: Section 4
      and one line of diagnosis, no Step 5 subagent. A uniform op below the top five has nothing
      for a full diagnosis to find.

   Every shortlist entry that does not survive the cut goes into the "Examined, Not Selected" table
   with its `e2e_ceiling` and its largest per-regime ceiling. Carrying more than ten into Step 5
   spends a full diagnosis pass per surplus candidate and produces reports that were never ranked.

---

## Step 5 — Diagnose each candidate, in this order

Dispatch one built-in subagent (`general-purpose` on Claude Code, `self` on Antigravity) per
full-report candidate from Step 4 item 7. Do not skip this step because tpu-diagnostics is your
only direct delegation target: built-in subagents are not package agents. Short-report candidates
get no subagent. Never generate more than 10 candidate reports. The following procedures must be strictly
followed to find the root cause.

Give every subagent the `opmap.sql` and `windows.csv` paths from Step 3, and the candidate's row
from the Step 4 table, for every regime. A subagent left to find its own window and its own
operator mapping will build them again, differently, and its numbers will not reconcile with the
ranked table.

**Pre-extract before dispatch.** In one pass over each analysis window, write every full-report
candidate's occurrences (step index, in-step index, kernel name, start, duration, self time,
device) to `diag/<id>/occ_<window>.parquet`. Give each subagent these paths. It opens the full
events parquet only for a neighbourhood query (5.3.3, 5.3.4) and for 5.5 on ranks outside the
windows.

**Bound each subagent.** Every command runs in the foreground. No `sleep` and no waiting on a
file that a finished command already wrote. Target 10 minutes. At 15 minutes, write the report
with the open items listed as unknown.

Give measurements, not explanations. A cause written into the prompt ("the decode ranks wait for
the prefilling ranks") decides which groupings the subagent tries, and the groupings it does not
try are where the missed structure is.

### 5.1 Distribution across occurrences. Mandatory, before reading any code.

An op's mean tells you nothing about why it is slow. Build the per-occurrence series first.

```sql
WITH e AS (
  SELECT CAST(end_ns AS BIGINT)-CAST(start_ns AS BIGINT) dur_ns, start_ns,
         row_number() OVER (ORDER BY start_ns) - 1 AS occ
  FROM Events
  WHERE device='<compute device>' AND kernel_name='<op>'
    AND start_ns BETWEEN <t0> AND <t1>)
SELECT occ % <ops_per_step> AS idx, count(*) n,
       median(dur_ns)/1e3 med_us, min(dur_ns)/1e3 min_us, max(dur_ns)/1e3 max_us
FROM e GROUP BY 1 ORDER BY 1
```

Compute `n`, median, mean, min, max, `max/median`, `p90/p10`. Report the per-index table itself,
not only these statistics, and state the first and the last occurrence in the step against the
median of the rest. Run this in every regime the op appears in.

- `max/median < 1.3` → **uniform**. Go to 5.2.
- `max/median ≥ 1.3` → **non-uniform**. Go to 5.3. Do not use the mean for anything until 5.3
  returns a structure.

### 5.2 Uniform: the op itself is the problem

1. Compare measured to floor. `measured/floor < 1.3` → the op is at its floor. It is not a kernel
   inefficiency. Record the diagnosis as "at floor".
2. `measured/floor ≥ 1.3` → find the inefficiency in this order, and stop at the first that
   explains the gap quantitatively:
   - **Layout.** Compare the operand layout the op receives against the layout its consumer or the
     kernel signature declares. A mismatch appears as a separate `copy`, `reshape`, `bitcast` or
     `transpose` adjacent in the timeline. Sum those neighbours; they often exceed the op.
   - **Padding.** Recompute the floor with each dimension rounded up to the tile or vector width.
     If the padded floor explains the gap, the cause is padding.
   - **Dtype.** A conversion on the critical path, or a compute dtype wider than needed.
3. If nothing above explains it, ask {{agent:tpu-diagnostics}} for the kernel's LLO analysis and
   report the unit census: a `profiling-request` with the trace in `traces`, the kernel name in
   `prompt`, and `fields` naming the census you need (for example `llo_unit_census`,
   `mxu_utilization_percent`, `alu_utilization_percent`). If the capture lacks LLO metadata (captured
   without `--xla_xprof_register_llo_debug_info=true`; the metric comes back with `tool_reason`
   `LLO_DATA_ABSENT`), say so, and write the kernel-internal cause as unknown.

### 5.3 Non-uniform: find the structure, then the cause

#### 5.3.0 Test for a single-index outlier. Before periodicity.

From the 5.1 per-index table, find any index whose median exceeds 1.3× the median of all other
indices. If one exists, the excess belongs to **that position in the step**. Name, from the code and
the timeline, what happens at that position and nowhere else: the first cross-rank synchronization
point in the step, the first layer after a dense block, a cache or table built once per step. Then
branch as 5.3.1 does: collective → 5.3.3, compute op → 5.3.4. Continue to 5.3.1 for the remaining
indices.

Periodicity grouping compares group medians, so one slow occurrence among 6 or more normal ones in
its group moves the group median by nothing. This test is the only one that sees it.

#### 5.3.1 Test for periodicity

For each `p` in 2..12, group occurrences by `idx % p` and compute
`separation = max(group median) / min(group median)`. Take the `p` with the highest separation.

- `separation > 1.3` and `p` matches a period in the Step 2 layer map → the split is **structural**.
  Name, from the code, exactly what the slow layers do that the fast layers do not.
- `separation > 1.3` but `p` matches nothing in the layer map → your Step 2 read is incomplete.
  Return to the source and find what recurs with period `p` before continuing.
- No `p` gives separation > 1.3 → go to 5.3.2.

**Rule 5.3.1.** When the excess is periodic, **it does not belong to the op that displays it.** It
belongs to the block that distinguishes the slow layers. An op whose slow and fast instances do
identical work is a symptom, not the cause.

Then branch: collective → 5.3.3, compute op → 5.3.4.

#### 5.3.2 No period: test these in order

1. Group by operand and result shape. If the slow group has different shapes, the cost is
   shape-driven, and the problem is the shape distribution, not the kernel.
2. Correlate duration with a data-dependent quantity in the trace (routed token counts, sequence
   lengths, ragged offsets). Report the correlation coefficient.
3. Group by rank. A rank-aligned split is imbalance; go to 5.5.
4. Group by position in time within the regime. A monotone drift is not an op problem.

#### 5.3.3 A collective displays the excess. It is almost never the collective.

1. **Compare the payload shape between the fast and the slow group.** Identical payload means the
   extra time is wait, not transfer. Confirm by computing transfer time from payload size and link
   bandwidth. If that is orders of magnitude below both groups, both are dominated by wait and the
   difference is pure arrival skew.
2. **Find the work unique to the slow group.** List every op running in the gap that precedes each
   slow instance and does not precede a fast instance. Verify the join is exact: every slow
   instance has it, no fast instance does. That op is the candidate cause.
3. **Decide local against remote. Do not skip this.**
   - Sum the candidate cause's own billed device time across the window on this rank. If that total
     is far smaller than the excess, the cost is **not on this rank**.
   - Compare the collective's duration across every traced rank. All traced ranks equal → the late
     arriver is a rank you did not capture. One rank lower than the others → that rank is the late
     arriver and its extra work is the cause.
4. **Rule.** A barrier's cost is set by its last arriver. If the capture does not contain every
   rank, you cannot name the last arriver. When that is the case, state that the last arriver is
   unknown, and state that the kernel you are about to blame is fast on every rank you can see.
5. Do not write "the work was issued asynchronously and surfaced at the barrier" unless billed time
   on the same rank accounts for it. Asynchronous local work is billed on some timeline of that
   rank. If no timeline holds it, it did not happen there.
6. Check every co-processor timeline (SparseCore or any secondary unit) over the same interval. If
   the secondary unit's busy time during the collective equals the collective's duration, the
   collective is running or spinning, not blocked behind separate work on that unit.

#### 5.3.4 A compute op displays the excess

1. Take one slow window and one fast window of equal structure. Sum self time per kernel name in
   each, difference them, and rank the difference. **The op carrying the excess is frequently not
   the op you started from.**
2. Compare window wall time to device busy time in that window. Busy ≈ wall → no stall, the excess
   is real extra work; find it in the code path unique to the slow layers. Wall > busy → there is a
   gap; identify what the gap waits for before naming a cause.

### 5.4 Cross-phase check (run for every candidate)

Compare the same op between two regimes with different work.
`scaling_efficiency = (heavy/light time ratio) / (heavy/light work ratio)`. Take both ratios in
the same direction, heavy over light; the other direction inverts every threshold below.

- `≈ 1.0` — scales with its work.
- `< 1.0` — the lighter phase is disproportionately slow. A fixed overhead dominates: a relayout, a
  launch, a synchronisation, or a dimension padded to a tile. Report every op below 0.5.

This is the only test that catches an op sitting near its floor in both phases yet badly wrong in
one of them.

### 5.5 Cross-rank check (run for every candidate)

Any op differing by more than 20% across traced ranks is imbalanced. Attribute it: different shapes
(input skew), different counts (routing skew), or neither (placement or hardware). State how many
of the total ranks were captured; an imbalance measured on 2 of N ranks bounds nothing.

### 5.6 Problem class (run for every candidate, last)

Assign exactly one class to the op or block that 5.1 to 5.5 named as the cause. Write it in the
candidate report's `Problem class` and `Class evidence` fields.

| Class | Condition | Required evidence |
| --- | --- | --- |
| `redundant-relayout` | The op is a `copy`, `reshape`, `bitcast`, `transpose` or layout-changing fusion that computes nothing the model needs. It exists only because the producer's output shape or layout differs from the shape or layout the consumer expects. | The producer op and its output shape and layout. The consumer op and the shape and layout it expects. The code location that introduces the mismatch. |
| `kernel` | Every other case. The op does required work, and its measured time is above its floor or it is the cause found by 5.3. | Measured ms, floor ms, and the cause found. |

Rules:

1. A relayout found in 5.2 item 2 as a neighbour of the candidate is its own cause. Classify the
   relayout ops as `redundant-relayout`, and report their summed ms, not the ms of the op they sit
   next to.
2. If the code location of the mismatch is not found, the class is still `redundant-relayout` when
   the producer and consumer shapes prove the mismatch. Write the code location as unknown.
3. A relayout that changes data the consumer needs in the new layout, and that no producer could
   emit directly, is required work. Classify it as `kernel`.

---

## Step 7 — Report

Write `decomposition.md` following {{reference:decompose-optimization/decompose_template.md}}.

In the `candidates` directory, write one report per candidate following {{reference:decompose-optimization/candidate_report_template.md}}. A short report fills the header, Section 4 and a one-line diagnosis, and writes `Short report: uniform, no Step 5 diagnosis.` in place of the other sections.

---

## Traps that have produced wrong answers

1. **A missing op means the wrong window**, not an absent op. Search all trace files before writing
   any negative statement.
2. **One kernel name can be several ops.** Reconcile counts against the structural prediction
   before computing any per-occurrence average.
3. **The mean hides periodicity.** Distribute across occurrences before ranking. An op whose slow
   instances are 9× its fast ones can sit mid-table on total time.
4. **A phase label names when capture was armed**, not what the file contains.
5. **A time filter written for one question excludes the answer to the next.**
6. **A negative you have already written down is still a hypothesis.** Once it is in the report it
   stops being tested; re-test it against every capture before it stays.
7. **Warmup is not steady state.** The same kernel can be 200× slower in the first second.
8. **Custom calls report zero FLOPs to XLA.** Invisible, not zero.
9. **A barrier's cost belongs to the rank you did not capture.**
10. **A period is a correlate, not a cause.** Name the code that differs between the groups, and
    confirm its billed time accounts for the excess, before you call it the cause.
11. **Ranking on one phase hides the largest total win.** A decode op that recovers 80% of a phase
    holding 30% of the time beats a prefill op that recovers 5%. Weight every regime by its share of
    the target metric and sum (Step 4).
12. **A group median hides one outlier per step.** A first-layer-only wait is invisible to
    `idx % p` grouping. Read the per-index table (5.1, 5.3.0).
13. **A hypothesis in the Step 5 prompt steers the diagnosis.** Give the subagent measurements.
14. **A microbenchmark twin is not the served model.** It has one layer per kind; a per-layer op
    joined against a per-step floor, or ranked on the twin's wall, is off by the layer count.
15. **One microbenchmark profile holds several shapes on one program.** Decode shapes share a
    bucket, so their ops look identical; only the step boundaries and `profile_order` say which
    shape a step is. Averaging across them mixes contexts and batches.
