# Optimization Decomposition Index: `<Model Name>`

<!--
Target length: ~2 pages maximum. Keep only index-level tables and ranking data here.
Put all per-candidate diagnoses, boundary arguments, and cost arithmetic in `candidates/<id>-<slug>.md`.
Write `unknown` for any missing metadata field — never guess.
-->

## 1. Run Metadata & Setup

| Field | Value |
| :--- | :--- |
| **Model** | `<model id / commit SHA>` |
| **Workload** | `<input_len / output_len, concurrency, chunk size, recipe name>` |
| **Sharding & Mesh** | `<TP=x, DP=y, EP=z, mesh shape, TPU slice e.g. v6e-16>` |
| **Precisions** | `Weights: <dtype> \| Activations: <dtype> \| KV Cache: <dtype> \| Collectives: <dtype>` |
| **Profile Job ID** | `<job_id>` |
| **Baseline & Logs** | `<baseline throughput/latency>` — `gs://<path/to/log_uri>` |
| **Phases Captured** | `Prefill: <file, device, [t0, t1] ms, N steps, ranks captured>`<br>`Decode: <file, device, [t0, t1] ms, N steps, ranks captured>` |
| **Capture Defects** | `<Missing ranks, missing LLO debug info, warmup trimmed, or "None">` |
| **Roofline Join Coverage** | `Prefill: <X>% of busy time matched \| Decode: <Y>% of busy time matched` |
| **Target Metric & Regime Weights** | `Target: <e2e time (default) \| TTFT \| TPOT>`<br>`<regime>: w=<0.xx> (<measured \| estimated>, <source>)` — one line per regime, padded/idle regimes included. Unknown weights priced at the largest plausible value. |

---

## 2. Definitions & Column Key

- **Device Step Time (`step_ms`)**: Total device busy time for one model step in the analyzed steady-state window.
- **Measured (`measured_ms`)**: Observed self-time of the operator per step from the xprof trace (measured).
- **Floor (`floor_ms`)**: Theoretical minimum time per step from tpu-roofline given hardware peak FLOPs, HBM bandwidth, and ICI bandwidth (modelled).
- **Gap (`gap_ms`)**: `measured_ms - floor_ms` — maximum recoverable time per step if the op reaches its hardware floor (modelled).
- **Step Share (`share %`)**: `measured_ms / step_ms` — fraction of total step time currently spent in this op (measured).
- **Ceiling (`ceiling %`)**: `gap_ms / step_ms` of one regime — maximum % reduction in that regime's step time if this op hits its floor (modelled).
- **Weight (`w_p`)**: the regime's share of the target metric's time (measured or estimated; state which).
- **E2E Ceiling (`e2e_ceiling %`)**: `Σ_p w_p × ceiling_p` over every regime the op appears in (modelled). **The ranking key.**
- **Problem Class**: `kernel` (the op does required work and is slow) or `redundant-relayout` (the op exists only because of a producer/consumer shape or layout mismatch). Assigned in Step 5.7.
- **Scaling Efficiency**: `(Heavy / Light time ratio) / (Heavy / Light work ratio)`. Values near `1.0` scale linearly with work; values `< 0.5` indicate fixed overhead (launch, padding, barrier, or relayout).

---

## 3. Ranked Optimization Candidates

**Device Step Time**:
- **Prefill**: `<X.XX> ms/step`
- **Decode**: `<Y.YY> ms/step`
- **Other regimes** (padded, idle, first-chunk): `<regime>: <ms/step>` each

<!--
One row per (candidate, regime) the op appears in. Rank and E2E Ceiling go on the candidate's first row only.
Rank by E2E Ceiling, never by one regime's Ceiling.
Write each candidate name as a relative markdown link to its report, candidates/<id>-<slug>.md.
Up to 10 candidates (Step 4 item 7). Mark estimated-weight candidates `(EW)` and short-report candidates `(short)` after the report link.
-->

| Rank | Candidate & Report | Regime | Weight | Kind | Measured (ms) | Floor (ms) | Gap (ms) | Step Share (%) | Ceiling (%) | **E2E Ceiling (%)** | Binding Resource | Problem Class |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `<id>-<slug>`, linked to its report | `<regime A>` | `<w>` | `<kind>` | `<val>` | `<val>` | `<val>` | `<val>%` | `<val>%` | **`<Σ w×ceiling>%`** | `<MXU/VPU/HBM/ICI>` | `<kernel \| redundant-relayout>` |
| | | `<regime B>` | `<w>` | | `<val>` | `<val>` | `<val>` | `<val>%` | `<val>%` | | `<MXU/VPU/HBM/ICI>` | |
| **2** | `<id>-<slug>`, linked to its report | `<regime>` | `<w>` | `<kind>` | `<val>` | `<val>` | `<val>` | `<val>%` | `<val>%` | **`<val>%`** | `<MXU/VPU/HBM/ICI>` | `<kernel \| redundant-relayout>` |
| **3** | `<id>-<slug>`, linked to its report | `<regime>` | `<w>` | `<kind>` | `<val>` | `<val>` | `<val>` | `<val>%` | `<val>%` | **`<val>%`** | `<MXU/VPU/HBM/ICI>` | `<kernel \| redundant-relayout>` |

### Examined, Not Selected
<!-- Every Step 4 shortlist entry that did not survive the cut, including every op selected for max/median > 2 or rank spread > 20%. -->

| Op | E2E Ceiling (%) | Largest per-regime Ceiling (%) (regime) | max/median (regime) | Reason not selected |
| :--- | :--- | :--- | :--- | :--- |
| `<op>` | `<val>%` | `<val>% (<regime>)` | `<val> (<regime>)` | `<at floor \| below cut \| ...>` |

### Gap Sum Check
- **E2E**: Sum of candidate `E2E Ceiling` = `<G>%`. Overlap adjustments: `<None | list>`.
- **Prefill**: Sum of candidate `Measured` = `<A> ms` (`<A/step>%`), Sum of `Floor` = `<B> ms`, Sum of `Gap` = `<C> ms` (`<C/step>%` max theoretical step reduction). Overlap adjustments: `<None | list overlapping candidates>`.
- **Decode**: Sum of candidate `Measured` = `<D> ms` (`<D/step>%`), Sum of `Floor` = `<E> ms`, Sum of `Gap` = `<F> ms` (`<F/step>%` max theoretical step reduction). Overlap adjustments: `<None | list overlapping candidates>`.

---

## 4. Cross-Phase Scaling Check

<!--
Include every operator that appears in two or more regimes/phases.
Do not delete this table if no rows are flagged — explicitly state "All multi-phase ops scale proportionally with work" when clean.
Flag any row with Scaling Efficiency < 0.5.
-->

| Operator | Light Phase (`<e.g. Decode>`) ms | Heavy Phase (`<e.g. Prefill>`) ms | Work Ratio (`Heavy / Light`) | Time Ratio (`Heavy / Light`) | Scaling Efficiency (`Time / Work`) | Status / Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `<op_name>` | `<val>` | `<val>` | `<val>` | `<val>` | `<val>` | `<OK (≈1.0) \| FLAGGED (<0.5): fixed overhead / padding / barrier>` |

---