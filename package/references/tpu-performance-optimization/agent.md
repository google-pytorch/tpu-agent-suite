---
description: The tpu-performance-optimization procedure. Turns a served job's profiled tpu-forward-bench baseline into gaps ranked by share x headroom against tpu-roofline floors, each diagnosed to its root cause, with orthogonal hypotheses for the top ones.
scripts:
- scripts/tpu-performance-optimization/inventory.py
- scripts/tpu-performance-optimization/q.py
references:
- references/tpu-performance-optimization/decompose_template.md
- references/tpu-performance-optimization/candidate_report_template.md
delegates:
- tpu-diagnostics
- tpu-roofline
- tpu-knowledge-base
host_dependencies:
- duckdb
- pandas
---

# tpu-performance-optimization

Input: the request, naming a `measured` forward-bench-result whose baseline arm was profiled.
Output: `decomposition.md`, a ranked list of operators, each with its diagnosed root cause, one
report per candidate, hypotheses for the top `top_n`, and the `perf-triage-result`.

`<workdir>` is `{{runs}}/<run_id>/artifacts/tpu-performance-optimization/<source_job or the
baseline's label>/`. Every query, summary and report goes under it.

Every number that comes from the roofline is **modelled**. Nothing here is a result; never record
anything this procedure produces as an experiment result.

The work compares two sides: what xprof measured and what the roofline says it should cost. Get
both first, find the candidate ops from the comparison, then diagnose each. Read source code or
earlier findings only while diagnosing a specific op; never read source in advance to guess an
optimization, and never write a candidate that is a model of the code rather than a finding in the
trace.

Steps 1–5 hold no fix, no fix direction and no fix cost: they stop at the diagnosed cause and its
problem class (5.7). Hypotheses come only in Step 6, each from a diagnosis.

## Scope: check these before anything else

Each failure returns `invalid_input` before any delegation.

- **The engine has a section below.** A request whose `engine` has no `## <engine>` section:
  `engine <x> is not supported here`.
- **The baseline is measured and profiled.** `baseline` reads as a forward-bench-result with
  status `measured`, and its baseline arm has `traces`, `evidence` and `cost_model`.
- **A `roofline` path, when given, exists** and reads as a `modeled` roofline-result.

## The two inputs this joins

| Source | Gives | Step |
| --- | --- | --- |
| The baseline arm's traces, exported as events parquets by {{agent:tpu-diagnostics}} and queried with {{script:tpu-performance-optimization/q.py}} | what every op **actually cost**, per occurrence, per rank, per regime | 1, 3, 5 |
| {{agent:tpu-roofline}} | what every op **should have cost**, and the model's structure: ops per layer, the layer type map, the shapes | 2 |

The profile says where the time went, the roofline says where it should have gone, and only the
join says what to work on: a block at 40% of step time at 92% of its floor is worth less than a
collective at 8% that is entirely exposed. Do not re-derive the op list, layer map or shapes by
hand.

## How the work runs

Every step runs in sequence, in your own shell or as one delegation at a time; nothing fans out.

- **Trace data comes only from {{agent:tpu-diagnostics}}.** It exports and reads; it is the only
  reader of `.xplane.pb` files and the only user of the xprof CLI.
- **Floors come only from {{agent:tpu-roofline}}**, one request covering every regime, plus at most
  a follow-up request for the affected rows (Step 3, Step 4.1). Never compute a floor yourself.
- **Facts a hypothesis rests on come from {{agent:tpu-knowledge-base}}**, cited.
- **Trace queries run in your shell** as {{script:tpu-performance-optimization/inventory.py}} and
  {{script:tpu-performance-optimization/q.py}} over the parquets tpu-diagnostics exported. Every bulk
  query writes its result to a file under `<workdir>`; read back only the ranked table, the
  per-candidate summaries and the few rows a decision needs. Raw censuses, per-occurrence series
  and the roofline's layer-by-layer CSV stay on disk.

Write bulk results with DuckDB's `COPY`, and keep stdout to a count:

```bash
python3 {{script:tpu-performance-optimization/q.py}} <parquet> \
    "COPY (<query over Events>) TO '<workdir>/<name>.csv' (HEADER)"
```

`q.py` reads the parquet directly. Do not use XProf's own events query command: it serialises
through a shared cache and deadlocks.

Do not send a modelled number back to argue with it; the floor is guidance for diagnosis. Do send
back an op it did not model, or an operating point Step 3 changed, asking only for those rows.

Never substitute a plausible value for a missing one: a ranking built on a guessed sharding ranks a
problem that does not exist. Carry an unknown precision as `unknown` and say so.

## vllm

The source job served with vllm, and the baseline is a tpu-forward-bench **twin**: the served
model with one layer per layer kind (GLM-5.2: 3 standing for 78), on dummy weights, with no
scheduler, and several step shapes back to back in one profile. **Read its `report` in full before
Step 1**, then its `plan`, the baseline arm's `cost_model`, each `evidence` path and the
`xprof_steps.json` and `result.json` beside it.

Take these from the REPORT and its files, never from the source job's `recipe.yaml`:

| item | where | why it matters |
| --- | --- | --- |
| twin layers, kinds, served count per kind | REPORT, `evidence.json` | the join is per layer kind, not per step |
| mode: mirror or scaled | REPORT, plan `sharding` | a floor at served degree does not apply to a scaled collective |
| deviations | plan `deviations`, REPORT | every deviation is an op whose twin cost is not its served cost |
| the shapes in the profile, in order | plan `top` / `points`, `profile_order`, REPORT step map | each shape is its own regime |
| each shape's weight | `cost_model.json`, REPORT step mix | the Step 4.2 weights; coverage below 100% leaves busy time unpriced |
| MTP drafter on or off | plan `twin.run_drafter` | off means the drafter's served cost is absent |
| image, commit, patch stack | REPORT, `result.json` `image_source`, `evidence.json` `patches` | the code to read when diagnosing |
| ranks captured | REPORT (mirror: pod 0 only, ranks 0–7 of 16) | bounds every cross-rank and barrier conclusion |
| noise | the arm's `noise_spread_pct`, REPORT repeat spread and `profile_spread` | the Step 4.5 noise floor |

What changes in the steps:

- **Step 1: regimes are the plan's shapes, not phases.** Decode shapes share one compiled program,
  so a histogram will not separate them. Split by the step boundaries in `xprof_steps.json`
  (`start_ms`, label, `program`) and the `profile_order` labels; inside a step, split layers by
  `model.py` `TRACE` (checked by the census in `evidence.json`). Treat the first profiled step of
  each shape as possible warmup (Rule 3.2).
- **Step 2: one roofline cell per shape**, at that shape's served operating point, with the twin's
  deployment and env, asking for **per-layer-kind** floors.
- **Step 3: predicted counts are twin counts**: once per twin layer of kind k, not `n_k` times.
  Ops outside the layers (embedding, LM head, sampling) appear once per step in both.
- **Step 4: join per layer kind, then scale:** `served_ms = n_k × measured twin-layer ms` against
  `n_k × floor of one layer of kind k`; per-step ops count once. Rank on served numbers weighted
  by each shape's busy share from `cost_model.json`, never on the twin's wall step.
- **Label every number twin (N layers), measured, or served, modelled**, never mixed in a column.
- **Name what the twin cannot show:** routing skew and EP imbalance, host and scheduler overhead,
  the drafter if it was off, and any deviation. A candidate resting on one of these is listed with
  `no_floor` naming it, not ranked.

The levers Step 6 proposes are this engine's, whose plugin is `vllm_torchtpu`, read in the
request's `checkout`:

| route | lever |
| --- | --- |
| `flag-env` | an XLA or libtpu flag, or a vllm-torchtpu env var the plugin reads at that code path; a setting, never a sweep |
| `kernel` | the tile, block or buffering of a Pallas or Mosaic kernel in `vllm_torchtpu/`, or a new or fused kernel for the block |
| `code` | any other change in `vllm_torchtpu/`, or a fix outside the plugin, named |

---

## Step 1 — Inventory every capture before forming any hypothesis

Do not query for an op until this step is complete.

1. List every trace: the baseline arm's `traces`, one `.xplane.pb` per profiled job. Record each
   file's job and rank. If only one rank carries device planes, record a capture defect now:
   every cross-rank test is then unavailable.
2. Send {{agent:tpu-diagnostics}} one `profiling-request` ({{schema:profiling-request.json}})
   exporting the events parquet of every trace:

   ```json
   {"prompt": "For each trace below, build XProf's events database: the events parquet with one row per event and the columns device, category, kernel_name, start_ns, end_ns, self_time_ns, input_tensors and output_tensors, with no event limit. Write it to <workdir>/events/<label>.parquet and list it in the trace's tool_outputs. Report the device planes each trace holds.", "run_id": "<run_id>", "traces": [{"path": "<baseline arm traces[0]>", "label": "base_j1_r0"}, {"path": "<baseline arm traces[1]>", "label": "base_j2_r0"}], "report_path": "<workdir>/tpu_profiling_events_report.md", "fields": ["events_parquet_path", "event_count"]}
   ```

   On `reported`, check every parquet exists and each trace's `coverage` matches the device planes
   you expect. On `failed` with `trace_empty_or_partial`, record a capture defect. On any other
   status, return `failed` with what it returned. Count captured cores (one process can hold
   several), not files, and give that count to 5.5.
3. **Run `inventory.py` once on the exported parquets.** It finds the step module, lists every
   step (wall ms, busy ms, op count) per TensorCore device, groups steps into step types, and
   writes a census of the median step of each type:

   ```bash
   python3 {{script:tpu-performance-optimization/inventory.py}} --top 15 \
       --out <workdir>/steps_inventory.csv \
       base_j1:r0=<parquet> base_j2:r0=<parquet> ... \
       > <workdir>/inventory.txt
   ```

   Read the step-type tables from `inventory.txt`, name the regimes, and write `windows.csv` from
   `steps_inventory.csv`. Use `q.py` only for a question the script does not answer. The repeat
   jobs must hold the same regimes; span-and-busy confirms it:

   ```sql
   SELECT device, count(*) n, min(start_ns)/1e6 t0_ms, max(end_ns)/1e6 t1_ms,
          sum(self_time_ns)/1e6 busy_ms
   FROM Events GROUP BY 1 ORDER BY 1
   ```

   For a regime boundary the script leaves unclear, a histogram over ~200 bins shows the op mix
   change:

   ```sql
   SELECT CAST(start_ns/ <bin_ns> AS BIGINT) bin, kernel_name,
          count(*) n, sum(self_time_ns)/1e6 ms
   FROM Events WHERE device='<compute device>'
   GROUP BY 1,2 QUALIFY row_number() OVER (PARTITION BY bin ORDER BY ms DESC) <= 3
   ORDER BY bin
   ```

4. Segment each file into **regimes**: a boundary is where the dominant kernels or the event rate
   change. Name each by its dominant ops and the batch dimension from the tensor shapes, then map
   it to a model phase.
5. Write the **analysis window** of each regime into `windows.csv`: file, device, `[t0, t1]`, step
   count, step ms. Every later number is reported against a named window.

**Rule 1.1.** A capture's phase label records when the profiler was armed, not what the file
contains. Scope a query to a regime from 1.4, never to a file by its name.

**Rule 1.2.** A time filter written for one question silently excludes the answer to the next.
Restate the window in every query and check the event count against the inventory.

---

## Step 2 — Structure and floors from tpu-roofline

The structural prediction comes from the roofline, which enumerates every op per layer with its
shapes, bytes and FLOPs. Get it before examining any per-op timing.

Floors come only from a `modeled` roofline-result. If {{agent:tpu-roofline}} returns
`invalid_input` (a hardware field the knowledge base lacks, or a missing input) or `failed`, return
`failed` with its status and its `missing` entries or `failure_reason`: nothing can be ranked
without floors, and a guessed floor ranks the wrong gap first.

When the request carries `roofline`, read that result: it must hold the rows below for every
regime Step 1 named. Ask {{agent:tpu-roofline}} only for what it lacks, with `rows` and `cells`.
Otherwise send it **one** `roofline-request` once Step 1 has named the regimes:

- `cells`: one per regime, its tokens per step, KV length and batch read from the trace, not the
  workload's nominal shape.
- `tpu`: the request's `tpu`.
- `context`: the source job and its `recipe.yaml` read-back when `source_job` is given (`{{runtime}}
  compute job artifacts --job-id <id> --out-dir <workdir>/source`), the model id, the image, the
  baseline plan and REPORT, and any existing config path. Ask it to start from a config it ships for this model and slice if one exists,
  checked against the recipe (weight, activation and KV dtypes, TP/DP/EP degrees, the MoE and
  attention flags in the server command), and to name every field it changed and why.
- In `prompt`: the fused kernels the census shows that the roofline cannot model directly (e.g. a
  fused EP MoE kernel holding dispatch, GEMMs and combine), so they are priced in this request;
  and any regimes whose forward path differs (MoE dispatch strategy, fused-activation path,
  activation dtype), which need separate configs. Pass inputs only; do not restate its procedure.

Require back, by path in `artifacts` unless noted:

| Required | Consumed by |
| --- | --- |
| model structure: layer count and the **layer type map** (which layer indices run which optional block, with period and offset), in `summary` | Step 3, 5.3.1 |
| layer-by-layer op list with local operand and result shapes and dtypes | Step 3 count and shape reconciliation; stays on disk |
| operation table: per-op bytes, FLOPs, collective volume, `T_mem` / `T_comp` / `T_comm`, binding term | Step 4, 5.2 |
| `roofline_data.json` | the Step 3 and Step 4 joins |
| `config.json` for each config | provenance, and the floor audit |

List every returned `config.json` path in the report's provenance section.

Two things a roofline cannot give you; read them from the request's `checkout` yourself, or write
them as unknown when there is none:

1. **Every conditional gate in the forward path that can skip work**: the `if`, its condition, and
   its value for the captured workload. A capture taken with a gate closed is unrepresentative.
2. **Which collective runs on which mesh axis**, if the roofline left it implicit.

If the roofline's layer map disagrees with the source, the source wins: send a follow-up request
with the correction before continuing.

Output of this step: one row per op —
`op → layers it runs on → count per step → local operand and result shapes → floor ms`. In
`roofline_data.json` these are `layer_kind`, `count_per_step`, `shape` and `floor_ms_per_pass`.

---

## Step 3 — Reconcile observed against predicted. This is a gate.

Write the operator mapping once, covering every regime, and keep it on disk:

| artifact | holds |
| --- | --- |
| `opmap.sql` | the kernel → operator rules (kernel names, source lines and operand shapes → the roofline's row names), all regimes in one file |
| `windows.csv` | one row per regime: file, device, `t0`, `t1`, step count, step ms |

Write `opmap.sql` from `steps_inventory.csv` and the row names in `roofline_data.json`, covering
every kernel above 0.1% of step time in every regime; record the unmapped remainder in ms per
regime. Then run the join as one `q.py` query that applies `opmap.sql` and reads `windows.csv`
and the roofline rows (DuckDB's `read_csv` / `read_json_auto`), writing `<workdir>/join.csv`: per
regime and operator, observed count, predicted count, ratio, measured ms, floor ms, headroom,
ceiling, `max/median`, and the cross-rank spread. Read back the rows where the counts disagree
and the ranked columns; never rebuild the mapping per regime.

| Observation | What it means | What to do |
| --- | --- | --- |
| observed count = predicted | consistent | proceed |
| observed = k × predicted, k integer | one kernel name covers k distinct ops | split by result shape, then operand shape, then `hlo_op`, until each group matches a predicted row's `shape`. Only then average |
| observed count is 0 | **your window is wrong, not the workload** | search the op in **every** trace file. If it appears anywhere, return to Step 1. Only when it is absent everywhere may you write "this op does not run", naming the gate from Step 2 that skipped it |
| observed > 0 but not predicted | the roofline's structure is wrong, or it did not model this op | collect **every** such op for every regime and send them to {{agent:tpu-roofline}} in one follow-up request with `rows` and the operating points, then repeat the join |

**Rule 3.1.** Normalising by an assumed count manufactures or hides a k-fold effect. Compute the
divisor inside the query, in the same pass as the sum.

**Rule 3.2.** Exclude warmup. For each top kernel, compare the duration distribution in the first
10% of the window against the rest. If the early median differs by more than 2×, shrink the
window, say so, and never use a warmup duration in a ranking or a root cause.

**Rule 3.3.** If two regimes of the same phase exist with different costs, **analyse the expensive
one**, and say which regime every number belongs to.

---

## Step 4 — Join against the roofline and select candidates

1. The floor is already in hand. Ask {{agent:tpu-roofline}} again only if Step 3 changed the
   operating point or found an op it did not model, and then only for the affected rows.
2. **Name the target metric and the regime weights before ranking.** The metric is the request's
   `metric`: e2e time or throughput (default), TTFT, or TPOT. For each regime `p`, state its weight
   `w_p`, its share of the metric's time:
   - e2e: `(steps of p per request × wall ms per step of p) / Σ over regimes`;
   - TTFT: prefill regimes only; TPOT: decode regimes only.

   Label each weight measured or estimated with its source. A regime of unknown weight is priced at
   its largest plausible weight, not dropped. Padded and idle regimes are regimes too.
3. Per op, per step, in the named window, **in every regime** (columns of `join.csv`):
   - `measured_ms` (measured), `floor_ms` (modelled), `gap_ms = measured_ms − floor_ms`
     (modelled), `share = measured_ms / step_ms` (measured), `headroom = 1 − floor_ms /
     measured_ms` (modelled), `ceiling_p = share × headroom = gap_ms / step_ms` (modelled);
   - the per-step floor comes from `floor_ms_per_pass`: a row's `floor_ms` sums its regime's
     `passes`, and one step runs one pass;
   - `e2e_ceiling = Σ_p w_p × ceiling_p` (modelled): **the ranking key**.
   - **No floor.** An op tpu-roofline did not price after the Step 3 follow-up is `no floor`, with
     the reason it gave. It gets no headroom, no ceiling and no rank, and is listed after the
     ranked rows by share. Never stand a utilization figure (MXU or HBM %, VMEM spills, collective
     wait) in for a floor: those are Step 5 evidence only.

   Never rank on one regime's ceiling: 80% of a phase holding 30% of the time is worth 24%; 5% of
   a phase holding 70% is worth 3.5%. Join each regime against the operating point its capture
   holds, and name that point in the report.
4. **Custom calls (Pallas, Mosaic, Tokamax) report 0 FLOPs and 0 bytes to XLA.** Never take their
   FLOPs, bytes, MXU% or roofline position from XProf's overview or roofline tools; their floor
   comes only from tpu-roofline modelling the kernel's mathematics. Never write "0 FLOPs, memory
   bound" for such an op.
5. Select every op that clears **both** the baseline's noise floor (`noise_spread_pct`) and 0.5% by
   `e2e_ceiling`. Cap at about 10.
6. **Also select, regardless of ceiling:** any op whose `max/median` across occurrences exceeds 2,
   and any op differing by more than 20% between traced ranks, tested in every regime. Each gets
   its own row; never fold it into another candidate. Compute `max/median` per **kernel
   signature** (name without the `.N` suffix, plus result shape), so two shape variants are not
   read as one non-uniform op.
7. **Cut the shortlist to at most ten**, once:
   1. **Full reports, by `e2e_ceiling`**: the rows that pass item 5, highest first.
   2. **Estimated-weight slot:** any op with a ceiling of 10% or more in a regime whose weight is
      estimated gets a full report stating the weight at which it would enter the top `top_n`.
   3. **Short reports:** a row outside the top `top_n` with no 5.3.0 outlier, no `max/median`
      above 2 and no rank spread above 20% gets Section 4 and a one-line diagnosis, no Step 5.

   Every entry cut goes into "Examined, Not Selected" with its `e2e_ceiling` and largest
   per-regime ceiling.

8. Mark each row the request's `tried` names with its lever and outcome. Drop a row only by rule
   (already accepted, or e2e ceiling under 0.5%) and write why.

Write the ranked shortlist to `<workdir>/ranked.csv`; it is the one table you keep in view.

---

## Step 5 — Diagnose each full-report candidate, one at a time

Diagnose candidates sequentially, in rank order, each from the same `opmap.sql`, `windows.csv`
and its `join.csv` row in every regime, so its numbers reconcile with the ranked table.

**Pre-extract once.** In one `q.py` pass per analysis window, write every full-report candidate's
occurrences (step index, in-step index, kernel name, start, duration, self time, device) to
`<workdir>/diag/<id>/occ_<window>.parquet`. Query those; open the full events parquet only for a
neighbourhood query (5.3.3, 5.3.4) and for 5.5 on ranks outside the windows.

**Per candidate,** write each query's output under `<workdir>/diag/<id>/`, then write the
candidate report and a summary (`diag/<id>/summary.md`: the per-index table, the structure found,
the cause, the class). Carry only the summary forward; do not keep raw series in view across
candidates. Budget about 10 minutes per candidate; at 15, write the report with the open items
listed as unknown.

Start each diagnosis from measurements, not from a cause you already suspect: the groupings you do
not try are where the missed structure is.

### 5.1 Distribution across occurrences. Mandatory, before reading any code.

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
and the first and last occurrence in the step against the median of the rest, in every regime the
op appears in.

- `max/median < 1.3` → **uniform**. Go to 5.2.
- `max/median ≥ 1.3` → **non-uniform**. Go to 5.3. Do not use the mean until 5.3 returns a
  structure.

### 5.2 Uniform: the op itself is the problem

1. `measured/floor < 1.3` → at its floor; not a kernel inefficiency. Record "at floor".
2. `measured/floor ≥ 1.3` → stop at the first of these that explains the gap quantitatively:
   - **Layout.** A mismatch between the operand layout received and the layout its consumer or
     the kernel declares shows as an adjacent `copy`, `reshape`, `bitcast` or `transpose`. Sum
     those neighbours; they often exceed the op.
   - **Padding.** Ask whether the floor with each dimension rounded up to the tile or vector width
     explains the gap; get padded floors from tpu-roofline in one `rows` request covering every
     candidate that needs one.
   - **Dtype.** A conversion on the critical path, or a compute dtype wider than needed.
3. If nothing explains it, ask {{agent:tpu-diagnostics}} for the kernel's LLO analysis: a
   `profiling-request` with the trace in `traces`, the kernel name in `prompt`, and `fields` such
   as `llo_unit_census`, `mxu_utilization_percent`, `alu_utilization_percent`. If it returns
   `tool_reason` `LLO_DATA_ABSENT` (captured without
   `--xla_xprof_register_llo_debug_info=true`), say so and write the kernel-internal cause as
   unknown.

### 5.3 Non-uniform: find the structure, then the cause

#### 5.3.0 Single-index outlier. Before periodicity.

From the 5.1 table, find any index whose median exceeds 1.3× the median of all others. Its excess
belongs to **that position in the step**: name, from the code and the timeline, what happens there
and nowhere else (the first cross-rank sync, the first layer after a dense block, a table built
once per step). Then branch: collective → 5.3.3, compute op → 5.3.4. Continue to 5.3.1 for the
remaining indices; periodicity grouping cannot see one slow occurrence per step.

#### 5.3.1 Periodicity

For each `p` in 2..12, group by `idx % p`; `separation = max(group median) / min(group median)`.
Take the `p` with the highest separation.

- `separation > 1.3` and `p` matches a period in the Step 2 layer map → **structural**. Name, from
  the code, what the slow layers do that the fast ones do not.
- `separation > 1.3` but no match → your Step 2 read is incomplete; find what recurs with period
  `p` in the source first.
- No `p` above 1.3 → 5.3.2.

**Rule 5.3.1.** Periodic excess **does not belong to the op that displays it**; it belongs to the
block that distinguishes the slow layers. Then branch: collective → 5.3.3, compute op → 5.3.4.

#### 5.3.2 No period: test in order

1. Group by operand and result shape. Different shapes in the slow group → the problem is the
   shape distribution, not the kernel.
2. Correlate duration with a data-dependent quantity in the trace (routed token counts, sequence
   lengths, ragged offsets); report the coefficient.
3. Group by rank. A rank-aligned split is imbalance; go to 5.5.
4. Group by time within the regime. A monotone drift is not an op problem.

#### 5.3.3 A collective displays the excess. It is almost never the collective.

1. **Compare payload shapes** of the fast and slow groups. Identical payload means wait, not
   transfer; confirm by computing transfer time from payload and link bandwidth.
2. **Find the work unique to the slow group**: every op in the gap before each slow instance and
   before no fast one. Verify the join is exact. That op is the candidate cause.
3. **Decide local against remote.** If the candidate cause's billed time on this rank is far
   smaller than the excess, the cost is not on this rank. All traced ranks equal → the late arriver
   is a rank you did not capture; one rank lower → that rank arrived late and its extra work is
   the cause.
4. A barrier's cost is set by its last arriver. Without every rank captured, state that the last
   arriver is unknown and that the kernel you would blame is fast on every rank you can see.
5. Do not write "issued asynchronously and surfaced at the barrier" unless billed time on the same
   rank accounts for it.
6. Check every co-processor timeline (SparseCore) over the same interval. Busy for the collective's
   whole duration → it is running or spinning, not blocked behind other work on that unit.

#### 5.3.4 A compute op displays the excess

1. Take one slow and one fast window of equal structure, sum self time per kernel in each,
   difference and rank. **The op carrying the excess is often not the one you started from.**
2. Busy ≈ wall in the window → real extra work; find it in the code path unique to the slow
   layers. Wall > busy → identify what the gap waits for before naming a cause.

### 5.4 Cross-phase check (every candidate)

`scaling_efficiency = (heavy/light time ratio) / (heavy/light work ratio)`, both heavy over light.
`≈ 1.0` scales with its work; `< 1.0` means a fixed overhead dominates the lighter phase (relayout,
launch, synchronisation, padding to a tile). Report every op below 0.5. This is the only test that
catches an op near its floor in both phases yet badly wrong in one.

### 5.5 Cross-rank check (every candidate)

Any op differing by more than 20% across traced ranks is imbalanced. Attribute it: different
shapes (input skew), different counts (routing skew), or neither (placement or hardware). State how
many of the total ranks were captured; an imbalance on 2 of N ranks bounds nothing.

### 5.6 Neighbours (every candidate)

Before the first candidate's 5.1, ask {{agent:tpu-diagnostics}}, in one `profiling-request` on
the baseline traces covering every full-report candidate, for the 5 ops before and after each
occurrence with times and shapes, its HLO producers and consumers, and for a Pallas or Mosaic
kernel its LLO analysis, each named in `fields`. Record
each neighbour with its relation and ms; one that carries part of the excess is folded into the
candidate or becomes its own row, priced from the Step 2 result.

### 5.7 Problem class (every candidate, last)

Assign exactly one class to the op or block 5.1–5.6 named as the cause, in the report's
`Problem class` and `Class evidence` fields.

| Class | Condition | Required evidence |
| --- | --- | --- |
| `redundant-relayout` | a `copy`, `reshape`, `bitcast`, `transpose` or layout-changing fusion that computes nothing the model needs, present only because producer and consumer layouts differ | the producer and its output shape and layout; the consumer and the layout it expects; the code location introducing the mismatch |
| `kernel` | every other case: required work above its floor, or the cause found by 5.3 | measured ms, floor ms, the cause |

1. A relayout neighbour found in 5.2 is its own cause: classify it `redundant-relayout` and report
   its summed ms, not the ms of the op beside it.
2. If the code location is not found but the shapes prove the mismatch, the class stands; write the
   location as unknown.
3. A relayout that changes data the consumer needs in the new layout, which no producer could emit
   directly, is `kernel`.

---

## Step 6 — Hypotheses for the top `top_n` rows

For each of the top `top_n` ranked rows that are not already accepted, propose one to three
hypotheses from its Step 5 diagnosis. A row with `no_floor`, or below the 0.5% threshold, gets
none.

1. **Read the code the diagnosis named,** in the request's `checkout` at the location 5.3 or 5.7
   found: the kernel, its call site, the flags and env vars that select it. Without a checkout,
   cite the op and write the location as unknown.
2. **Ask {{agent:tpu-knowledge-base}}** for the facts a lever rests on: what a flag does on this
   TPU generation, the hardware limit the floor binds on, a known kernel for the op. Pass `run_id`
   and the key terms; cite what it returns in `evidence`.
3. **For a `kernel` lever**, work from the {{agent:tpu-roofline}} row for this op: its binding
   term and the hardware limit behind its floor. Propose what the kernel should do differently
   (tiling, buffering, fusion, the unit it runs on) and why that reaches the floor. Its `route`
   is `kernel`: tpu-pallas, which owns the kernel strategies, designs it.
4. **Write each hypothesis** as `lever`, `route`, `claim` (why the op costs more than its floor and
   how the lever removes it, citing the diagnosis), `expected_gain_pct` and `cost` (what testing
   it spends: one env arm, one tpu-pallas run, a code change), with its `evidence` paths.
   `expected_gain_pct` is the fraction of the gap the lever removes times the row's
   `e2e_ceiling`, both stated in the claim; it never exceeds the ceiling.
5. **Keep them orthogonal.** Two hypotheses on one row act on different mechanisms, so either can
   be tested and stacked alone. Two that change the same flag or the same code are one hypothesis;
   keep the one the evidence supports better.
6. **Do not repeat what was tried.** A lever the request's `tried` lists on this op is proposed
   again only when the diagnosis shows why it lost, and the claim says so.

Each `hypotheses` entry is a proposal, not a measurement: the caller has it reviewed and tested.

## Step 7 — Report

Write `decomposition.md` following {{reference:tpu-performance-optimization/decompose_template.md}}
from `ranked.csv` and the candidate summaries, with the `no floor` ops and their reasons beside
the ranked table. In the `candidates` directory, one report per candidate following
{{reference:tpu-performance-optimization/candidate_report_template.md}}, its hypotheses in a section of
their own after the diagnosis. A short report fills the header, Section 4 and a one-line
diagnosis, and writes `Short report: uniform, no Step 5 diagnosis.` in place of the other
sections.

Return `triaged` with one `gaps` entry per ranked row in rank order, then the `no floor` ops by
share: `op`, `layer_kind`, `regimes`, `measured_ms` and `share`; `rank`, `floor_ms`, `gap_ms`,
`headroom` and `e2e_ceiling_pct` for a ranked row, `no_floor` for the others; `bound_by`,
`diagnosis` (where, why, what would remove it, class, evidence paths), `neighbours`, `tried`,
`report` and `hypotheses`. Add `decomposition`, `roofline` (the result the floors came from) and
`traces` (the traces and parquets the shares came from).

---

## Traps that have produced wrong answers

1. **A missing op means the wrong window**, not an absent op. Search every trace before writing a
   negative.
2. **One kernel name can be several ops.** Reconcile counts against the prediction before
   averaging.
3. **The mean hides periodicity.** Distribute across occurrences before ranking.
4. **A phase label names when capture was armed**, not what the file contains.
5. **A time filter written for one question excludes the answer to the next.**
6. **A negative already written down is still a hypothesis.** Re-test it against every capture.
7. **Warmup is not steady state.** The same kernel can be 200× slower in the first second.
8. **Custom calls report zero FLOPs to XLA.** Invisible, not zero.
9. **A barrier's cost belongs to the rank you did not capture.**
10. **A period is a correlate, not a cause.** Name the differing code and confirm its billed time
    accounts for the excess.
11. **Ranking on one phase hides the largest total win.** Weight every regime and sum (Step 4).
12. **A group median hides one outlier per step.** Read the per-index table (5.1, 5.3.0).
13. **A suspected cause steers the diagnosis.** Start from measurements.
14. **A microbenchmark twin is not the served model.** A per-layer op joined against a per-step
    floor, or ranked on the twin's wall, is off by the layer count.
15. **One microbenchmark profile holds several shapes on one program.** Only the step boundaries
    and `profile_order` say which shape a step is.
