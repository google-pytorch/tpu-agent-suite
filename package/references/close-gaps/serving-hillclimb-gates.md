---
scripts:
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
delegates:
  - tpu-diagnostics
---
# Serving Hill-Climb Gates (`hillclimb_cli.py`)

Use `{{script:optimization-heuristics/hillclimb/hillclimb_cli.py}}` during `/close-gaps` on vLLM-TorchTPU serving workloads (especially multi-host 1P1D disaggregated P/D and large MoE / MLA models on `tpu7x`) to catch arms on CPU that cannot start or cannot help before they spend multi-slice TPU hours.

All subcommands run with the Python standard library only. All accept `--json` for structured output, except `microbench` when it generates a benchmark: it prints the pytest file.

```bash
CLI="python3 {{script:optimization-heuristics/hillclimb/hillclimb_cli.py}}"
```

Three rules keep the gates inside the close-gaps loop:

- **No state of their own.** `ledger.md` and the `gaps/*.md` files beside it stay the only state. The gates read past outcomes from them (section 1.1) and write nothing.
- **No second win rule.** A win is still `cost_model.json` `significant` with no guardrail regression in 2.4, then critical-review in step 4. The outputs of `predict`, `early-verdict`, `noise` and `verify` are advisory. Put them in the gap file or the critical-review packet as supporting evidence.
- **A `preflight` ERROR is a launch blocker, not a verdict.** Treat it like a change that will not build: fix the config or drop the idea before spending a job. Never cancel a step-4 arm because of a gate's output.

---

## 1. Where each gate fits in `/close-gaps`

Each gate runs at the step where its inputs exist:

| `/close-gaps` step | Subcommand(s) | Inputs | How the output is used |
| --- | --- | --- | --- |
| **0 Baseline, 1 Rank** | none | | The gates need a gap to dig, an arm to launch or a running arm, so they start at 2.2 |
| **2.2 Dig** | `roofline` | Op table from the gap's `tpu-diagnostics` report | Per-op floors and ranked targets in the gap file |
| **2.2 Dig** | `code-limits`, `code-gap`, `ports` | vllm-torchtpu checkout at the stack sha | Hard-coded caps, gate chains and unmerged fixes, cited as `file:line` in the diagnosis |
| **2.2 Dig (P/D jobs)** | `legs`, `kv-replay` | The original job's per-leg `/metrics` export and request trace (`{{runtime}} compute job artifacts`) | Which leg binds. A change to the idle leg cannot move served throughput |
| **2.3, before a flag / env-knob sweep or a `needs-e2e` arm** | `preflight` | Candidate knobs, baseline knobs (`--defaults`), checkout (`--repo`), the original job's server metrics, `ledger.md` (`--ledger`) | ERROR: do not launch as configured. WARNING: fix it or write in the gap file why not |
| **2.3, kernel lever** | `microbench` | Crossed limit from `code-limits`; `RESULT` log of the 1-host job | Go / no-go for a kernel extension before its 2.4 A/B |
| **2.4 `needs-e2e` rows, step 3** | `predict` | Default or `--history`-calibrated parameters; microbench deltas | Advisory served projection for schedule changes `T_mix` cannot decide, written beside the close-gaps projection |
| **Step 4, while the arms run** | `early-verdict` | aiperf realtime telemetry in each arm's `bench.log` | Advisory reading in the ledger. The arms run to completion |
| **Step 4, after the arms finish** | `verify`, `noise` | Each arm's `metrics.json`, detector findings before and after, repeated served runs | Supporting evidence for critical-review |
| **Any failed job** | `diagnose-failure` | The failed server or client log | Picks the fix before one of the two relaunches |

### 1.1 Past outcomes come from close-gaps state

When an A/B, an E2E arm or a job ends, write its outcome in the gap file (or the ledger) as one line, naming every changed knob or env flag as a backticked `KEY=VALUE`:

```text
- outcome: not a win `DCP_SIZE=8` (needs-e2e arm in step 4: -24.8% Mtok/$)
- outcome: crash `TPU_MLA_ABSORB_Q_FROM_HEAD=1` (vmem_oom, see diagnose-failure)
- outcome: win `PREFILL_DCP_SIZE=16` `DECODE_DCP_SIZE=8`
```

The outcome is `win`, `not a win` or `crash`. close-gaps asks for this line in 2.4 and step 4, and `LEDGER-TEMPLATE.md` shows it, so there is nothing to keep in sync. `preflight --ledger <dir>/ledger.md` and `code-gap --ledger` read these lines from `ledger.md` and every `gaps/*.md` beside it. `--ledger` is repeatable, so earlier runs' ledgers can be passed too. `preflight` reports a `prior_outcome` WARNING when the arm repeats a setting recorded as `not a win` or `crash`, and INFO for a recorded `win`. If a `--ledger` path does not exist, `preflight` adds one INFO finding saying prior outcomes were not checked. It does not fail.

---

## 2. Pre-launch config & topology gates (`preflight`, `legs`, `kv-replay`)

### 2.1 `preflight` — static launch audit

Run `preflight` on every candidate config JSON (or `--knob KEY=VALUE` overrides) before submitting a server job:

```bash
$CLI preflight \
  --config candidate_arm.json \
  --defaults baseline_knobs.json \
  --plugin glm5_mla_moe \
  --repo <vllm-torchtpu> \
  --server-metrics server_metrics_export.csv \
  --ledger ./.work/<MODEL_DIR>/close-gaps/<job-id>/ledger.md \
  --json
```

Exit code 1 and `passed: false` mean at least one ERROR (`--no-fail-on-error` keeps exit code 0).

| Severity | Checks |
| --- | --- |
| ERROR | `vmem_absorb_q_head`, `dcp_size_prefill_kv_halved` (top-level `DCP_SIZE`), `off_leg` (with `--server-metrics`), `asym_dcp_indexer_layout`, `asym_dcp_striped_layout`, `asym_dcp_interleave`, `decode_bucket_dcp`, `kv_prefetch_cap_mismatch`, `host_ram_oom` (prefill compile buckets < 512), `unknown_env_flag` (`TPU_*` / `VLLM_*` flags missing from `envs.py`), `kv_budget` (backed by a failed run or an HBM-risk flag) |
| WARNING | `decode_tp_kernel`, `decode_bucket_padding`, `gate_off`, `moe_mode`, `hbm_risk`, `decode_moe_crossover`, `extra_prefill_leg` (only for knob maps with a measured concurrency limit; GLM-5 P/D: `c <= 96`), `plugin` (`--plugin` has no registered knob map, so checks fall back to GLM-5 P/D knob names), `dcp_size_prefill_kv_halved` (explicit `PREFILL_DCP_SIZE`), `host_ram_oom` (staging leases), `kv_budget`, `prior_outcome` (`not a win` / `crash`) |
| INFO | `numerics`, `unknown_env_flag` (non-serving variables), `prior_outcome` (`win`, or a missing ledger) |

A `prior_outcome` from a ledger line that changed several settings together says so: that outcome belongs to the bundle, not to the one setting the new arm shares with it.

Key checks:
- **Symmetric vs asymmetric P/D DCP (`dcp_size_prefill_kv_halved`)**: a prefill DCP of 8 or less at 32 or more sessions per prefill leg halves the prefill HBM prefix cache. Setting top-level `DCP_SIZE=8` on a `TP=16` 1P1D recipe did this on `GLM-5.2-FP8` `tpu7x` (`11.89M -> 5.95M` tokens, measured `-24.8%` Mtok/$ at `c=48` with decode cap 16). Keep `PREFILL_DCP_SIZE=16, DECODE_DCP_SIZE=8`. With asymmetric DCP, `TPU_KV_STRIPED_LAYOUT=1` and `INDEXER_KV_CACHE_MODE=sal` are errors; the measured asymmetric arm set `TPU_KV_STRIPED_LAYOUT=0`.
- **Asymmetric DCP interleave (`asym_dcp_interleave`)**: asymmetric KV transfer moves whole packed rows of 8 tokens, so `DECODE_INTERLEAVE` must be a multiple of 8. When neither the arm nor the recipe defaults set it, preflight assumes vLLM's default interleave (1) and blocks the arm; set `DECODE_INTERLEAVE` explicitly.
- **Fused TP MoE decode step cap (`decode_tp_kernel`)**: `step_tokens = cap * (1 + num_speculative_tokens)`. A step pads up to the next `ROW_BUCKETS` entry (read from `--repo`, else `(16, 32, 64)`), so 48 tokens run in the 64 bucket. Only a step above the largest bucket (for example `20 * 4 = 80`) leaves the fused kernel. If it is also at or below `FUSED_MOE_DECODE_KERNEL_MAX_TOKENS` (default 128), it falls back to the unfused GMM+psum path. A kernel microbenchmark at 80 tokens on one 8-core `tpu7x` host (`GLM-5.2-FP8` TP16 per-rank shapes) measured `438 us` per MoE layer on that fallback vs `175 us` with `ROW_BUCKETS` extended to 80. End to end, cap 20 on the fallback read ITL p50 `10 -> 13 ms` and `-28%` Mtok/$ at 11 min of the measured phase (not a final result); the extended buckets were not measured end to end. Known limitation: the check does not model explicit decode buckets, so a `DECODE_TOKEN_BUCKET_EXTRA` entry that is not in `ROW_BUCKETS` (such as the 48 that `decode_bucket_padding` can suggest) may take a step off the fused kernel without a finding.
- **`tpu7x` VMEM OOM (`vmem_absorb_q_head`)**: `TPU_MLA_ABSORB_Q_FROM_HEAD` at 64 or more replicated local Q heads per rank (DCP of 16 or more with `VLLM_DCP_Q_REPLICATE` on) allocates 68 MiB of Pallas windows in `absorb_q_head`, above the 64 MiB `tpu7x` VMEM, and compilation fails with `RESOURCE_EXHAUSTED`.
- **Inert flags (`moe_mode`, `gate_off`, WARNING)**: a flag that only runs under the other MoE mode, or whose prerequisite `envs.py` gate is off, has no effect, so the arm would measure the baseline again. Turn the gate on or drop the flag. These do not block, because the gate chains are parsed from `envs.py` comments.

### 2.2 `legs` — 1P1D prefill vs decode bottleneck diagnosis

Before proposing any leg-scoped change on a disaggregated 1P1D deployment, check which leg owns queueing and throughput:

```bash
$CLI legs \
  --server-metrics server_metrics_export.csv \
  --config candidate_arm.json \
  --defaults baseline_knobs.json \
  --json
```

- `decode_limited` when every prefill endpoint's `vllm:num_requests_waiting` p75 is below 1 and the busiest prefill endpoint is busy less than 70% of the run (`_BUSY_FRACTION_LIGHT`, a chosen cutoff, not a measurement). `prefill_queued` when the prefill waiting p50 is 1 or more. `balanced` otherwise.
- `off_leg_warning` is set when every changed knob targets the non-binding leg (`PREFILL` or `DECODE` in the knob name). With `--config`, pass `--defaults` so only the knobs that differ from the baseline count as changed.
- Measured on `GLM-5.2-FP8` 1P1D at `c=48`: a prefill-only arm that made the drafter `2.8x` faster (`7.5 -> 2.7 ms` per draft pass) gave `0.999x` end-to-end throughput, with the prefill queue empty at p50 and p75.

### 2.3 `kv-replay` — prefix-cache & Little's Law admission replay

When prefill queueing dominates TTFT, replay request sequences (`requests.jsonl`) across HBM + CPU KV tiers and check Little's Law lane saturation before launching a slice:

```bash
$CLI kv-replay \
  --requests requests.jsonl \
  --dcp-size 16 \
  --concurrency 48 \
  --decode-cap 16 \
  --throughput-req-s 0.45 \
  --mean-prefill-s 2.1 \
  --mean-decode-s 95.0 \
  --json
```

---

## 3. Digging gaps in Step 2.2 (`roofline`, `code-limits`, `code-gap`, `ports`)

### 3.1 `roofline` — analytical model-family floor vs tabular op timings

Join tabular op timing exports (CSV or JSON records emitted by `{{agent:tpu-diagnostics}}`) against `tpu7x` analytical rooflines (`glm5_mla_moe`, `qwen3_5_hybrid_moe`, `deepseek_mla_moe`):

```bash
$CLI roofline \
  --trace step_ops_tabular.csv \
  --plugin glm5_mla_moe \
  --batch-reqs 16 \
  --padded-rows 64 \
  --context-tokens 9216 \
  --weight-dtype fp8 \
  --json
```

Reports MXU / HBM / VMEM / ICI floor time per architecture pack, step compute vs exposed communication split, and ranked optimization targets with concrete knob suggestions. Each target's saving range is a heuristic estimate (measured category time times the lever's hand-set prior share), not a measured result. For 4-bit weight dtypes, pass a measured `--vpu-unpack-gweights-per-ms` to add the VPU unpack time to the compute floor; it has no default.

### 3.2 `code-limits` & `code-gap` — source limit & flag gate audit

Scan the served `vllm-torchtpu` checkout at the stack SHA before editing flags or kernels:

```bash
# Find hard-coded bucket tuples and shape caps crossed by the candidate step shape
$CLI code-limits \
  --repo <vllm-torchtpu>/src/vllm_torchtpu/kernels \
  --steps 80 \
  --base-steps 64 \
  --json

# Audit envs.py defaults, prerequisite gate chains, HBM risks, and inactive step ranges
$CLI code-gap \
  --repo <vllm-torchtpu> \
  --knob "EXTRA_ENV_DECODE=MOE_HUDSON_DECODE_MIN_TOKENS=16" \
  --steps 80 \
  --tp \
  --json
```

`code-gap` reads `src/vllm_torchtpu/envs.py` from `--repo` (or from `--served-ref` / `--upstream-ref` in that checkout, or from `--served-envs-file` / `--upstream-envs-file`). When the file is missing it exits 2 with a message naming the flags to pass; when a ref cannot be read it exits 2 with the `git show` error.

### 3.3 `ports` — unmerged PR branch triage

Rank local git refs / PR branches by active switches, touched kernel limits, and merge cleanliness before writing a patch from scratch:

```bash
$CLI ports \
  --repo <vllm-torchtpu> \
  --served-ref origin/main \
  --refs refs/pull/101/head refs/pull/102/head \
  --steps 80 \
  --json
```

---

## 4. Microbenchmarking & closed queueing-network projection (`microbench`, `predict`)

### 4.1 `microbench` — standalone kernel bucket verification

When `code-limits` identifies a crossed kernel cap (such as `ROW_BUCKETS = (16, 32, 64)` at 80 tokens):

```bash
# Emit a standalone pytest harness comparing current fallback vs extended cap
$CLI microbench \
  --repo <vllm-torchtpu>/src/vllm_torchtpu/kernels \
  --limit-name ROW_BUCKETS \
  --steps 80 \
  --base-steps 64 \
  --out /tmp/bench_moe_80.py

# Score RESULT lines from the run log (checks target speedup and non-target regressions)
$CLI microbench --bench-log bench_moe_80.log --steps 80 --json
```

### 4.2 `predict` — closed queueing-network serving throughput & p90 interactivity

Projects 1P1D or single-host throughput (`tok/s/chip`), TTFT, and `p90` interactivity (`tok/s/user`), plus `Mtok/$` when you pass your own chip-hour rate with `--chip-hour-usd` (no price is built in; relative changes need no rate). For P/D it solves a closed queueing network (a client think-time delay station plus prefill and decode stations with load-dependent rates) exactly by convolution; for single-host it iterates a chunked-prefill fixed point. The default parameters were fit to `GLM-5.2-FP8` P/D and `Qwen3.5-397B-A17B-FP8` single-host runs; their `1.35%` and `1.95%` `tok/s/chip` MAE are in-sample (fit and scored on the same runs). No fit error is reported for the `deepseek_mla_moe` defaults. `--history` recalibrates the workload tables from your own runs.

```bash
$CLI predict \
  --plugin glm5_mla_moe \
  --history baseline_runs.jsonl \
  --concurrency 48 \
  --decode-cap 20 \
  --delta-step-ms 4.2 \
  --chip-hour-usd <your rate> \
  --json
```

---

## 5. Step-4 readings (`early-verdict`, `noise`, `verify`, `diagnose-failure`)

### 5.1 `early-verdict` — equal-time `aiperf` comparison

While a step-4 arm runs, compare its `aiperf` realtime telemetry with the baseline arm's at the same elapsed time:

```bash
$CLI early-verdict \
  --baseline-log baseline_bench.log \
  --arm-log stack_bench.log \
  --json
```

`early_verdict.py` computes total tok/s/chip (the `Total` telemetry row divided by `--arm-chips` / `--baseline-chips`, default 16) at the arm's latest block and at the baseline block within 60 s of the same elapsed time, and returns:

- `stop` when `ratio + band < target` (`--target-ratio`, default 1.0). With the same-shape bands (same concurrency and chip count: ±8% from 10 min, ±6% from 15 min, ±4% from 30 min) that is a ratio below `0.92x` at 10 min, `0.94x` at 15 min and `0.96x` at 30 min. It is also `stop` when the arm's ITL is at least `1.5x` the baseline's and the ratio is below target (checked from 5 min). `--cross-shape` uses ±11% / ±9% / ±7% from 15 / 20 / 30 min. The 10-minute and 30-minute bands (cross shape: 20 and 30 minutes) are chosen margins, not measured errors; the error was measured at 15 minutes only. The 10-minute band is wider than the largest same-shape 15-minute error (5.7%). The later bands are narrower than the 15-minute maximum and have no measured error behind them.
- `promising` when `ratio - band > target` and ITL is under `1.5x`; `wait` otherwise; `too_early` before the first band starts (10 min, or 15 min with `--cross-shape`) or when the baseline has no block near that time.

In close-gaps this is a reading, not a decision: write it in the ledger beside the arm (for example `early-verdict <elapsed> min: stop, <ratio>x`) and let the arm finish, because critical-review needs the full A/B. A `stop` is a cue to start the arm's trace analysis with `{{agent:tpu-diagnostics}}` while it runs.

### 5.2 `noise`, `verify`, and `diagnose-failure`

```bash
# Strict same-(perf_knobs, concurrency) repeat spread, for verify --noise-rel
$CLI noise --history runs.jsonl --json

# Mechanism check + noise-aware KEEP / DISCARD / INCONCLUSIVE reading (advisory)
# --target-detector accepts canonical IDs (D8_kv_head_padding), unambiguous D<N>
# prefixes (D8), or semantic slugs (batched_rpa_decode_untunable) resolved
# dynamically against serving_analysis so renumberings stay compatible.
$CLI verify \
  --baseline baseline_metrics.json \
  --candidate candidate_metrics.json \
  --noise-rel 0.02 \
  --target-detector kv_head_padding \
  --before-findings before_findings.json \
  --after-findings after_findings.json \
  --json

# Classify crash / OOM / timeout logs (vmem_oom vs hbm_oom, bucket asserts, timeouts)
$CLI diagnose-failure --log failed_server.log --run-id arm-candidate --json
```

`verify` reads `serving_analysis` `findings.json` payloads natively (`detector_id`, `est_step_fraction_saved`, and list-valued `evidence`) as well as category summaries, and answers whether the targeted detector finding shrank and whether the metric moved outside the noise band. A throughput win that worsens a latency metric past its limit (5% by default) reads `INCONCLUSIVE`, not `KEEP`: it is a different operating point until both arms are compared at matched interactivity. Mismatched benchmark settings, concurrency included, also read `INCONCLUSIVE`. Give that to critical-review as evidence; its `KEEP` does not accept a change and its `DISCARD` does not reject one.

---

## 6. Lifecycle Phases, Skill Boundaries & Bundled Modules

Subcommands split logically into two phases over shared stateless readers (`ledger.py`, `run_history.py`, `lessons.py`, `metrics.py`):

- **`arm_preflight` (gap digging & pre-launch gates)**: `preflight`, `legs`, `kv-replay`, `roofline`, `code-limits`, `code-gap`, `ports`, `microbench --repo`, `predict`.
- **`arm_verdict` (in-flight & post-run readings)**: `early-verdict`, `verify`, `noise`, `diagnose-failure`, `microbench --bench-log`.

How the three modeling/benchmark helpers relate to existing skills:
- **`roofline` (`trace_roofline.py`) vs `roofline-analysis` & `decompose-optimization`**: `decompose-optimization` and `roofline-analysis` are the primary Step 1 / neighbour tools when `.xplane.pb` traces are present; `trace_roofline.py` is a stdlib-only helper for when only tabular op exports from `{{agent:tpu-diagnostics}}` are available, adding optional 4-bit VPU unpack floors and fixed-vs-proportional step-cost splitting.
- **`microbench` (`microbench.py`) vs `workload-microbenchmark`**: `workload-microbenchmark` runs the served model's forward cells in a vLLM engine on a container image; `microbench` runs *before* an image exists, on one TPU host, to check a single Pallas/JAX kernel bucket extension (`jax.lax.scan` latency + `float64` `rel_l2`/`cos` numerics).
- **`predict` (`predictor.py`) vs `workload-microbenchmark` `T_mix`**: `T_mix` holds the job's step counts fixed, so by its own documentation it does not decide changes that alter the schedule (chunk size or `mnbt`, speculative tokens, a re-sharding that changes admission, scheduler policy) and sends them to its measured L2/L3 levels; `predict` is a no-run analytical screen for those `needs-e2e` levers (also `DECODE_BATCH_CAP` and 1P1D leg balance), solving the closed queueing network (`solve_network`) or the single-host chunked-prefill fixed point. It ranks arms before a run is spent; the measured run still decides.

All sibling modules alongside `{{script:optimization-heuristics/hillclimb/hillclimb_cli.py}}` are pure Python standard library:

| Module | Phase | Responsibility |
| --- | --- | --- |
| `__init__.py` | shared | Package marker for `hillclimb` optimization gates. |
| `hillclimb_cli.py` | entry | Unified `argparse` CLI entry point for all 13 subcommands. |
| `metrics.py` | shared | `ServingMetrics`, `Mtok/$` at a caller-supplied chip-hour rate, and relative deltas that need no rate. |
| `run_history.py` | shared | `RunRecord` JSONL store and strict `(perf_knobs, concurrency)` `noise_estimate`. |
| `ledger.py` | shared | Reads `outcome:` lines from the close-gaps `ledger.md` and `gaps/*.md`; keeps no state. |
| `lessons.py` | shared | Failure log regex classifier (`vmem_oom` before `hbm_oom`, bucket asserts, timeouts) and prior-outcome matching. |
| `preflight.py` | `arm_preflight` | Pre-launch config validator (live `--repo` `ROW_BUCKETS` with `serving_analysis.limits` static fallback, asymmetric P/D DCP, VMEM OOM, off-leg arms, prior outcomes). |
| `legs.py` | `arm_preflight` | 1P1D prefill vs decode bottleneck analyzer from Prometheus `/metrics` or CSV exports. |
| `kv_replay.py` | `arm_preflight` | Deterministic prefix-cache HBM + CPU tier replay and Little's Law lane saturation check. |
| `code_limits.py` | `arm_preflight` | AST scanner for integer/tuple compile and kernel bucket limits (`ROW_BUCKETS`) in Python source. |
| `code_gap.py` | `arm_preflight` | `envs.py` registry parser, prerequisite gate-chain checker, HBM risk, and active step-range analyzer. |
| `ports.py` | `arm_preflight` | `git ls-remote` PR head parser and keyword/path relevance ranker for unmerged fixes. |
| `model_plugins.py` | `arm_preflight` | `TPU_7X` hardware spec and composable architecture packs (`GLM5_MLA_MOE`, `QWEN3_5_HYBRID_MOE`, `DEEPSEEK_MLA_MOE`). |
| `trace_roofline.py` | `arm_preflight` | Tabular op timing parser (CSV/JSON), pack floor comparator, and ranked optimization target generator. |
| `predictor.py` | `arm_preflight` | Closed queueing-network predictor: exact convolution for P/D, fixed point for single-host. |
| `microbench.py` | both | Standalone kernel microbenchmark generator (`--repo`) and `RESULT` log evaluator (`--bench-log`). |
| `early_verdict.py` | `arm_verdict` | Equal-time `aiperf` telemetry parser and `stop` / `promising` / `wait` / `too_early` classifier. |
| `verify.py` | `arm_verdict` | Provenance checks, `serving_analysis`-aware mechanism verification, and the advisory `KEEP` / `DISCARD` / `INCONCLUSIVE` reading. |
