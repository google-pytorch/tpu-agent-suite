# Hillclimber State Management (`framework/hc/`)

`framework/hc/` implements the protected coordination store, detached Git lineage repository, script sealer, CLI (`{{runtime}} hc ...`), and target-side job runner (`target_runner.py`) for iterative Hillclimber optimization campaigns.

Agent sandboxes deny direct read/write access to `~/.tpu-agent-suite/campaigns/`, so agents interact with campaign state, candidate history, and sealed evaluation protocols strictly through the `runtime.py hc` CLI while editing candidate source code in `<workspace>/src/`.

---

## Module Overview

| Module | Role |
| :--- | :--- |
| `constants.py` | Domain `StrEnum`s, limits (`MAX_TEXT_LENGTH`, `MAX_SHORT_LENGTH`), and structured `HcError` subclasses (`INVALID_STATE`, `VALIDATION_ERROR`, `NOT_FOUND`, `GIT_ERROR`). |
| `paths.py` | Layout helpers under `~/.tpu-agent-suite/campaigns/`, symlink rejection, and workspace subpath containment checks. |
| `db.py`, `schema.sql`, `queries.sql` | Normalized `STRICT` SQLite schema (`SCHEMA_VERSION = 1`), named `:param` queries, WAL + FK connection setup, and `BEGIN IMMEDIATE` / `SAVEPOINT` transactions. |
| `schemas.py` | Validators for Campaign, Protocol, and Hypothesis JSON configs (`framework/schemas/hc-*.json`) and canonical `__flags.json` serialization. |
| `telemetry.py` | Post-commit JSONL event buffer (`events.jsonl` and `events_unscoped.jsonl`) with `cmd.rejected` and optional `cmd.called` logging. |
| `git_ops.py` | Detached bare Git repository (`campaigns/<id>/git/`), `<workspace>/src/` working-tree population/3-way merges, `refs/candidates/<id>` pinning, and root-relative `patch_sha` computation. |
| `campaign_ops.py` | Campaign lifecycle (`open`, `set-microbenchmark-protocol`, `block`, `unblock`, `close`), manifest-driven script sealing, and active-campaign/open-round guards. |
| `round_ops.py` | Round lifecycle (`round open`, enforcing at most one `Open` round per active campaign). |
| `hypothesis_ops.py` | Hypothesis registration (`hypothesis add`) and immutable refutation (`hypothesis refute`). |
| `candidate_ops.py` | Candidate working-tree preparation (`candidate prepare`), immutable commit proposal (`candidate propose`), supersession, and abandonment (`candidate abandon`). |
| `review_ops.py` | Qualitative candidate and hypothesis reviews (`review add`) with harness-inferred reviewer identity. |
| `metrics.py` | Sealed benchmark and accuracy `summary.json` parsing, environment consistency checks, and trial-scoped raw-replicate measurement pooling. |
| `decisions.py` | Multi-cell weighted geometric mean benchmark verdicts (`Win`, `Viable`, `Neutral`, `Loss`, `Invalid`), per-cell regression guardrails, and accuracy gate evaluation. |
| `candidate_state.py` | On-the-fly strict-precedence `CandidateState` (`Promoted` down to `Untested`) and `prior_decisions_count` computation relative to a reference baseline. |
| `cli.py` | Argument parser (`build_parser`) and top-level CLI dispatch / rejection logging. |
| `target_runner.py` | Standalone stdlib-only runner staged into target TPU job working directories for `serve-vllm`, `serve-sglang`, `eval-vllm`, and `run-microbenchmark`. |

---

## Protected Store Layout

```text
~/.tpu-agent-suite/campaigns/
├── campaign.db                        # Normalized STRICT SQLite coordination store
├── logs/events_unscoped.jsonl         # Rejections before a campaign ID is resolved
├── <campaign_id>/
│   ├── git/                           # Detached bare Git repo + worktree_state.json
│   ├── scripts/                       # Immutable scripts sealed from package & workspace
│   │   ├── manifest.json              # SHA-256 digests + host e2e_sweep source paths
│   │   ├── target_runner.py           # Target-side job runner
│   │   ├── serve_vllm.sh              # Canonical engine launcher (vllm or sglang)
│   │   ├── apply_vllm_patch.sh        # Sealed when engine == "vllm"
│   │   ├── eval_recipe.sh             # Sealed when accuracy_gate is non-null
│   │   ├── serving_sweep.py           # Audit copy of host E2E sweep driver
│   │   ├── forward_run.py             # Sealed when microbenchmark_target is .json
│   │   └── microbenchmark_p<pid>/     # Sealed workspace microbenchmark target
│   ├── sweeps/                        # Sweep working directories
│   └── logs/events.jsonl              # Campaign-scoped JSONL telemetry
├── sealed_jobs/<job_id>/
│   └── sealed/                        # Write-once job outputs and logs (from compute)
└── runs/<run_id>/active_campaign_id   # Protected run -> campaign pointer
```

---

## Manifest-Driven Script Sealing & Protocols

Shipped benchmark and serving scripts are declared once in `package/package.release.json` under `"hillclimber"` (`e2e_sweep_script`, `e2e_sweep_companions`, `serving_scripts`, `vllm_patch_applier`, `accuracy_scripts`, `microbenchmark_runner`), verified at build time by `compiler.py verify`, bundled into distributions (including `--skills` subset builds), and published in `tpu-agent-suite-build.json`.

- **Main E2E protocol (`hc-protocol.json`):** Specifies `image` and optional `base_server_args`, `base_client_args`, and `base_eval_args` (checked at `campaign open` to be disjoint from flags managed by `eval_recipe.sh`, such as `--tasks` and `--limit`). It carries no script paths: `campaign open` resolves only the scripts needed for `(campaign.engine, protocol.accuracy_gate)` from `tpu-agent-suite-build.json`, seals them under canonical filenames in `campaigns/<id>/scripts/`, and records `e2e_sweep_script` + companion host paths and SHA-256 digests in `manifest.json` so `serving_sweep.py` can run in place on the host after digest verification. If `accuracy_gate` is non-null for an engine with no `accuracy_scripts` entry (e.g. `sglang`), `campaign open` fails fast.
- **Microbenchmark protocol:** Requires `execution_spec.microbenchmark_target` naming either a `<plan>.steps.json` file (which also seals `forward_run.py`) or a standalone single-file `.py` script inside `<workspace>/` (outside `<workspace>/src/`). Mutable via `campaign set-microbenchmark-protocol` only until the first `Microbenchmark` job is dispatched.

---

## State Transitions & Key Invariants

- **Campaigns (`CampaignState`):**
  - Open in `Unanchored` with `initial_version` imported from `--code-dir` into `campaigns/<id>/git/` and checked out at `<workspace>/src/` (including canonical `__flags.json`, with `.git` stripped).
  - Transition to `Active` once baseline measurements anchor `champion` and `entry_baseline`.
  - `campaign block <reason>` transitions `Unanchored` or `Active` -> `Blocked`; `campaign unblock` restores `Active` (if `entry_baseline` is set) or `Unanchored`.
  - `campaign close [--abort]` cancels remaining `Dispatched` jobs and transitions to terminal `Closed` (`Manual` or `Aborted`).
- **Rounds (`RoundState`):**
  - Require an `Active` campaign with a `champion`, and enforce at most one `Open` round per campaign (`Open` -> `Progressed` | `Barren`).
- **Hypotheses (`HypothesisState`):**
  - Created as `Open` in the current `Open` round (`lever_kind`, `target_scope`, and matching `canonical_ops` or `redundant_relayout` child rows).
  - `hypothesis refute <id> <reason>` transitions `Open` -> `Refuted`; `refutation_reason` is write-once and `Promoted` hypotheses cannot be refuted.
- **Candidates & Lineage:**
  - `candidate prepare <p1> [<p2>] [--discard]` populates `<workspace>/src/` from 1 or 2 distinct parent candidates; at least one parent must descend from (or equal) the current `champion`. Dirty edits in `<workspace>/src/` are refused unless `--discard` is passed.
  - `candidate propose` requires an `Open` round, verifies at least one staged parent descends from the round's `current_champion`, verifies linked `--hypotheses` are `Open`, commits `<workspace>/src/` + canonicalized `__flags.json`, computes `patch_sha` (`git diff initial_version..sha` excluding `__flags.json`), and pins `refs/candidates/<id>` inside the same `BEGIN IMMEDIATE` SQLite transaction.
  - `Abandoned` (`abandoned_reason IS NOT NULL`) and `Superseded` (`superseded_by IS NOT NULL`) are mutually exclusive; `initial_version` and `Promoted` candidates (ancestors of `champion`) can neither be abandoned nor superseded.
  - `CandidateState` is never stored in SQLite; `candidate_state.py` derives it on the fly relative to reference baseline `B` (`current_baseline` or `--baseline-measurement`) in strict precedence order: `Promoted` -> `Superseded` -> `Abandoned` -> `E2E_In_Flight` -> `E2E_Passed` / `E2E_Accuracy_Failed` / `E2E_Pending_Accuracy` / `E2E_Viable` / `E2E_Regressed` (`Neutral`, `Loss`, or `Invalid`) -> `Microbenchmark_In_Flight` -> `Microbench_Passed` / `Microbench_Regressed` (`Neutral`, `Loss`, or `Invalid`) -> `Untested`.
- **Measurements & Decisions (`DecisionVerdict`):**
  - Benchmark jobs emit canonical `summary.json` (`{"status": "ok", "cells": [{..., "replicates": [...]}], "environment": {...}}` from `serving_benchmark.py` / `serving_sweep.py` via `$PROFILE_DIR/server_env.json` or normalized by `target_runner.py` `run-microbenchmark`), and accuracy jobs emit `{"status": "ok", "scores": {<metric>: <float>}}` via `target_runner.py`.
  - `Measurements` are stateless snapshots pooling raw per-replicate samples across all verified `Succeeded` jobs for `(trial, subject, protocol)` with identical 10-field `environment` metadata.
  - `Decisions` evaluate `min_replicates` and `max_cell_regression_percentage` across every protocol cell (`Invalid` on violation), then classify the weighted geometric mean speedup delta `(prod_i (r_i ** w_i) - 1) * 100` against inclusive `[-noise_floor_percentage, +noise_floor_percentage]` (`Neutral`), `-noise_floor_percentage` (`Loss`), and `min_win_percentage` (`Win` vs. `Viable`).
- **Telemetry:**
  - Queued in `TelemetryBuffer` and flushed strictly after SQLite `COMMIT`. Failed CLI invocations emit `cmd.rejected` with `[error_code]` and exit code (`2` for `INVALID_ARGS`, `1` for domain errors).

---

## CLI Subcommands

Invoked as `{{runtime}} hc [--workspace PATH] [--run-id ID] <domain> <action> ...`:

| Command | Description |
| :--- | :--- |
| `campaign open <campaign.json> <protocol.json> --code-dir <dir> [--microbenchmark-protocol <micro.json>] [--package <pkg>]` | Open a new `Unanchored` campaign, seal scripts, and initialize `git/` + `<workspace>/src/`. |
| `campaign set-microbenchmark-protocol <micro.json> [--package <pkg>]` | Set or replace the microbenchmark protocol before any `Microbenchmark` job runs. |
| `campaign block <reason>` / `campaign unblock` | Pause a campaign pending human intervention, or resume a `Blocked` campaign. |
| `campaign close [--abort]` | Cancel `Dispatched` jobs and close the active campaign (`Manual` or `Aborted`). |
| `round open` | Open the next sequential round in an `Active` campaign. |
| `hypothesis add <hypothesis.json>` / `hypothesis refute <id> <reason>` | Register an `Open` hypothesis in the current round, or mark it `Refuted`. |
| `candidate prepare <p1> [<p2>] [--discard]` | Populate or 3-way merge `<workspace>/src/` from 1–2 parent candidates. |
| `candidate propose [--hypotheses <id> ...] [--supersedes <id>] --innovation <text> --short-innovation <text>` | Commit `<workspace>/src/` as an immutable candidate and pin `refs/candidates/<id>`. |
| `candidate abandon <id> <reason>` | Mark a non-`Promoted`, non-`Superseded` candidate as `Abandoned`. |
| `review add (--candidate <id> \| --hypothesis <id>) --approval <Pass\|ConditionalPass\|Fail> --comments <text> [--short-comments <text>] [--conditions <text>]` | Record a qualitative review (`--conditions` required iff `ConditionalPass`). |

---

## Target Job Runner (`target_runner.py`)

Target TPU serving and accuracy jobs run `target_runner.py` inside the staged job working directory alongside canonical sealed launchers (`serve_vllm.sh`, `serve_sglang.sh`, `eval_recipe.sh`):

- **`__hc_context.json` (required, campaign-fixed):** Supplies `target_model` / `weights_path`, `engine`, `base_server_args`, `base_eval_args`, `accuracy_metric`, and optional `eval_tasks`, `eval_limit`, and `server_port`.
- **`__flags.json` (optional, candidate-authored):** Supplies candidate `env` variables and extra `server_args`. Reserved model/accuracy keys (`model`, `target_model`, `eval_limit`, `base_eval_args`, etc.) and `--model` / `--model-path` / `--served-model-name` flags inside `server_args` are rejected so a candidate cannot swap out the model or weaken its accuracy gate.
- **`candidate.patch` (optional, non-empty):** Listed in `vllm_patches.txt` for `serve_vllm.sh` (`apply_vllm_patch.sh`) on `serve-vllm`, or applied idempotently via `git apply -p1` against installed `sglang` on `serve-sglang`. Empty/whitespace patches are skipped.
- **Modes:** `serve-vllm` and `serve-sglang` `exec` the staged bash launcher in place; `eval-vllm` invokes `eval_recipe.sh` (which starts the server through `target_runner.py serve-vllm` and runs `lm_eval`) and reduces `eval_out/**/results_*.json` into `summary.json`.
