# Model Onboarding & Hillclimbing Grader

A zero-install utility to grade or re-evaluate Harbor model onboarding and hillclimbing trials using the latest RewardKit verifier suite, display an immediate before-and-after **Reward Diff**, and export slim, upload-ready packages to Harbor Hub.

---

## Working Directory

Always execute commands from the **root of the `tpu-agent-suite` repository**:

```bash
cd <your tpu-agent-suite checkout>
```

*(All commands below assume you are standing in the repository root).*

---

## Concrete Examples (Copy & Paste to Run Immediately)

Here are working examples using a real trial in this repository:

### 1. Run via Local Path

```bash
uv run eval/verifiers/onboard_hillclimb/grade.py jobs/mhillclimbing/dd860156-9649-4608-ad59-53eb0b8b9226
```

### 2. Run via Job UUID

```bash
uv run eval/verifiers/onboard_hillclimb/grade.py 6537a7f6-cff2-47f5-9ca7-1d1179c89980
```

### 3. Run via Harbor Hub URL

```bash
uv run eval/verifiers/onboard_hillclimb/grade.py "https://hub.harborframework.com/jobs/6537a7f6-cff2-47f5-9ca7-1d1179c89980/trials/7345bc71-6b45-4c3b-a133-892f655557f1"
```

### 4. Regrade & Upload Directly to Harbor Hub

```bash
uv run eval/verifiers/onboard_hillclimb/grade.py 6537a7f6-cff2-47f5-9ca7-1d1179c89980 --upload
```

---

## Generic Templates (With Placeholders)

Replace `<PLACEHOLDERS>` with your specific target:

### From a Harbor Hub URL

```bash
uv run eval/verifiers/onboard_hillclimb/grade.py "https://hub.harborframework.com/jobs/<JOB_ID>/trials/<TRIAL_ID>"
```

### From a Job UUID

```bash
uv run eval/verifiers/onboard_hillclimb/grade.py "<JOB_ID>"
```

### From a Local Directory

```bash
uv run eval/verifiers/onboard_hillclimb/grade.py "jobs/mhillclimbing/<JOB_FOLDER_NAME>"
```

---

## Common Recipes

### Quick Terminal Check Only (Do Not Export Folder)
If you only want to view the Reward Diff in the terminal without writing a `jobs/regraded_*` directory:

```bash
uv run eval/verifiers/onboard_hillclimb/grade.py jobs/mhillclimbing/dd860156-9649-4608-ad59-53eb0b8b9226 --no-export
```

### Update Original Trial In-Place
If you want to update the original local directory's `verifier/reward.json` directly in-place:

```bash
uv run eval/verifiers/onboard_hillclimb/grade.py jobs/mhillclimbing/dd860156-9649-4608-ad59-53eb0b8b9226 --use-original-dir
```

### Explicit Model Name Override
If the trial directory lacks clear model metadata or you want to explicitly test a target model name against the `onboarding_success` LLM judge:

```bash
uv run eval/verifiers/onboard_hillclimb/grade.py jobs/mhillclimbing/dd860156-9649-4608-ad59-53eb0b8b9226 --onboarding-model-name "google/gemma-4-31B-it"
```

### View Full Verbose Logs
To view raw container test output and full LLM judge explanations from `reward-details.json`:

```bash
uv run eval/verifiers/onboard_hillclimb/grade.py jobs/mhillclimbing/dd860156-9649-4608-ad59-53eb0b8b9226 -v
```

---

## The 5 Pipeline Steps

When you execute a regrade, the console displays high-signal progress across 5 steps:

```text
[1] Using job at 'jobs/mhillclimbing/dd860156-9649-4608-ad59-53eb0b8b9226' for regrade
  -> Discovered 1 trial(s) in 'dd860156-9649-4608-ad59-53eb0b8b9226': dd860156-9649-4608-ad59-53eb0b8b9226__session

[2] Staging verifier suite for 'dd860156-9649-4608-ad59-53eb0b8b9226__session'

[3] Running verifiers via test.sh + RewardKit on 'dd860156-9649-4608-ad59-53eb0b8b9226__session'
  Reward Diff (original -> regraded):
    Metric                 Original     Regraded     Diff
    ────────────────────────────────────────────────────────
    reward                 0.3000       0.3000       =
    speedup                1.0753       1.0753       =
    onboarding_success     1.0000       1.0000       =
    hillclimb_success      0.0000       0.0000       =
    efficiency             0.9638       0.9638       =
    valid_tool_calls       0.9638       0.9638       =
    optimization_steps     1            1            =
    cheating               0.0000       0.0000       =
    execution_report       0.0000       0.0000       =

[4] Exporting slim regraded job to 'jobs/regraded_dd860156-9649-4608-ad59-53eb0b8b9226'
  -> Excluded bulky runtime artifacts (preserved original metadata + slim trajectory)
  -> Copied files to jobs/regraded_dd860156-9649-4608-ad59-53eb0b8b9226

[5] Harbor upload status
  -> No upload performed (local export only).
  -> To upload to Harbor platform, run:
  ->   harbor upload jobs/regraded_dd860156-9649-4608-ad59-53eb0b8b9226
```

---

## Why Export is Slim

Standard Harbor trial runs accumulate large checkpoint files, workspace copies, and xprof traces (often **150 MB - 500 MB+** per trial).

The slim export:
1. **Preserves 100% of original metadata**: `trial_name`, start/end timestamps, and `agent_info` (who ran it, agent name, agent version, model name).
2. **Excludes bulky artifacts**: Completely omits `artifacts/` and `workspace/` directories.
3. **Synthesizes a lightweight, valid ATIF 1.7 trajectory**: Contains clickable links pointing back to the original full Harbor trial.
4. **Reduces size by >99.9%**: Produces a clean **~50 KB** directory that uploads to Harbor Hub in milliseconds.

---

## CLI Options

| Option | Description |
| :--- | :--- |
| `target` | **(Required)** Job ID (UUID), Harbor Hub URL, or local path to job/trial directory. |
| `--upload` | Automatically runs `harbor upload` on the exported job upon completion. |
| `--slim` | Exports a slim copy omitting bulky `artifacts/` and replacing `trajectory.json` with a 2-step summary (alias: `--trim`). |
| `--full` | Preserves full original `artifacts/` and `trajectory.json` in the exported copy **(default)**. |
| `--no-export` | Skips creating the `jobs/regraded_<name>` directory (console diff check only). |
| `--use-original-dir` | Updates `verifier/` files directly in the source directory in place (aliases: `--in-place`, `--apply`). |
| `--onboarding-model-name` | Target model name override for `onboarding_success` verifier (e.g. `google/gemma-4-31B-it`). Auto-detected if omitted. (Alias: `--model-name`) |
| `--verifiers` | Comma-separated list of verifiers to run (default: `onboarding_success,efficiency,cheating,speedup,execution_report`). |
| `-v`, `--verbose` | Dumps full container execution logs and complete `reward-details.json` LLM judge reasoning. |
