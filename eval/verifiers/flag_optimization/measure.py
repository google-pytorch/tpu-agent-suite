# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The one measurement every rewardkit criterion of this task reads.

rewardkit runs the criteria under tests/ concurrently, in threads of one
process. Each criterion directory carries a copy of this module and the first
one on PYTHONPATH wins, so they all share one `evaluate()`, which measures the
agent's config once and caches the result:

  1. Validate <workspace>/output/best_config.json against the task's spec:
     $FLAGOPT_TASK in the root-owned /opt/flagopt/tasks.json, read with the
     root-owned /opt/flagopt/flagopt.py (the agent runs as an unprivileged
     user and cannot change either).
  2. Delete any Job the agent left behind in this trial.
  3. Baseline: tests/baselines/<task>.json, recorded once per task
     (eval/suites/flag_optimization/record_baseline.py). A task without a
     recording measures the baseline live (TODO: record them).
  4. Run the agent's config in a fresh Job, measured exactly as `flagopt run`
     (one retry if its step time is noisy; one retry with random init, against
     the recorded random-init baseline, if it fails with checkpoint weights).
  5. Accuracy gate: per-step loss and gradient norm over the first
     `gate_steps` steps must match the baseline's within tolerance.
  6. speedup = baseline p50 step time / candidate p50 step time; the reward
     is the improvement, speedup - 1 (`improvement_score`).

It never raises: an unexpected error is recorded as an infrastructure error,
so every criterion still gets an answer. verification.json and
verification.md under the verifier log dir hold every detail.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

# The verifier's environment comes from task.toml, not from the agent, so these
# overrides exist for the unit tests only.
FLAGOPT_HOME = Path(os.environ.get("FLAGOPT_HOME", "/opt/flagopt"))
TESTS = Path(os.environ.get("TESTS_DIR", Path(__file__).resolve().parents[1]))
if str(FLAGOPT_HOME) not in sys.path:
    sys.path.insert(0, str(FLAGOPT_HOME))
import flagopt  # noqa: E402  (the root-owned copy in the image)

BASELINES_DIR = TESTS / "baselines"
# eval/adapter/kube_gate_setup.sh installs this in every trial and puts
# /usr/local/bin/kubectl in front of it: every kubectl call is scoped to the
# trial's own resources (the `eval-trial-id` label).
KUBE_GATE_WRAPPER = Path("/usr/libexec/kubectl-wrapper")

_LOCK = threading.Lock()
_RESULT: dict | None = None


def log(msg: str) -> None:
    print(f"[measure {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_dir() -> Path:
    return Path(os.environ.get("FLAGOPT_LOG_DIR", "/logs/verifier"))


def improvement_score(result: dict) -> float:
    """The reward: the throughput improvement over the baseline.

    speedup - 1 (0.25 for 25% more tokens/s, i.e. a step time 1/1.25 of the
    baseline's) for a config that is valid, ran and passed the accuracy gate;
    0 for anything else, including a config slower than the baseline. There
    is no cap and no pass threshold: more improvement, more reward.
    """
    sp = result.get("speedup")
    if not (result.get("config_valid") and result.get("candidate_ran")
            and result.get("accuracy_gate")) or sp is None:
        return 0.0
    return float(max(0.0, sp - 1.0))


def evaluate(workspace: Path) -> dict:
    """Measure once per process; every criterion gets the same result."""
    global _RESULT
    with _LOCK:
        if _RESULT is None:
            try:
                _RESULT = _measure(Path(workspace))
            except Exception:  # noqa: BLE001 - criteria must still get an answer
                _RESULT = {
                    "config_valid": False,
                    "candidate_ran": False,
                    "accuracy_gate": False,
                    "infra_error": True,
                    "error": traceback.format_exc(),
                }
            _RESULT.setdefault("improvement_score", improvement_score(_RESULT))
            _write_report(_RESULT, Path(workspace))
        return _RESULT


def _kubectl_cmd() -> list[str]:
    env = os.environ.get("FLAGOPT_VERIFIER_KUBECTL")
    return env.split() if env else ["kubectl"]


def _kube_gated() -> bool:
    return KUBE_GATE_WRAPPER.is_file(
    ) and not os.environ.get("FLAGOPT_VERIFIER_KUBECTL")


def load_recorded_baseline(spec: dict, steps: int) -> dict | None:
    """{weights: {"metrics", "summary", ...}} from tests/baselines/, if valid.

    A recording made for different steps, warmup or image is ignored rather
    than compared against, since its step time would not be comparable.
    """
    baseline_file = BASELINES_DIR / f"{spec['task_id']}.json"
    if not baseline_file.is_file():
        return None
    try:
        data = json.loads(baseline_file.read_text(encoding="utf-8"))
    except ValueError as exc:
        log(f"ignoring unreadable {baseline_file.name}: {exc}")
        return None
    meas = spec["measurement"]
    expect = {
        "task_id": spec["task_id"],
        "steps": steps,
        "warmup_steps": meas["warmup_steps"],
        "image": spec["cluster"]["image"],
    }
    mismatch = {
        k: (data.get(k), v)
        for k, v in expect.items() if data.get(k) != v
    }
    if mismatch:
        log(f"ignoring {baseline_file.name}: recorded for {mismatch}")
        return None
    runs = {
        w: r
        for w, r in (data.get("runs") or {}).items()
        if r.get("metrics") and (r.get("summary") or {}).get("step_time_p50_s")
    }
    return runs or None


def _stable(spec: dict, res: dict) -> bool:
    cv = res.get("summary", {}).get("step_time_cv")
    return cv is not None and cv <= float(
        os.environ.get("MAX_STEP_TIME_CV",
                       spec["measurement"]["max_step_time_cv"]))


def _brief(res: dict) -> dict:
    keep = ("name", "status", "ok", "weights", "steps", "exit_code",
            "error_signatures", "pod_diagnosis", "summary", "wall_time_s",
            "queued_s", "ran_s", "detail", "error")
    out = {k: res[k] for k in keep if k in res}
    out["metrics_head"] = res.get("metrics", [])[:20]
    return out


def _measure(workspace: Path) -> dict:
    spec = flagopt.load_spec(FLAGOPT_HOME / "tasks.json",
                             os.environ.get("FLAGOPT_TASK"))
    meas = spec["measurement"]
    loss_rtol = float(os.environ.get("LOSS_RTOL", meas["loss_rtol"]))
    gn_rtol = float(os.environ.get("GRAD_NORM_RTOL", meas["grad_norm_rtol"]))
    steps = int(os.environ.get("VERIFY_STEPS", meas["steps"]))
    weights = os.environ.get("VERIFY_WEIGHTS",
                             meas.get("weights", "checkpoint"))
    deadline = time.time() + float(
        os.environ.get("VERIFY_DEADLINE_SEC",
                       spec["verifier_timeout_sec"] - 600))
    out: dict = {
        "task_id": spec["task_id"],
        "steps": steps,
        "loss_rtol": loss_rtol,
        "grad_norm_rtol": gn_rtol,
        "config_valid": False,
        "candidate_ran": False,
        "accuracy_gate": False,
        "infra_error": False,
    }

    # 1. The deliverable.
    deliverable = Path(
        os.environ.get("FLAGOPT_DELIVERABLE",
                       workspace / "output" / "best_config.json"))
    try:
        cand = json.loads(deliverable.read_text(encoding="utf-8"))
        resolved, xla = flagopt.validate_candidate(spec, cand)
    except FileNotFoundError:
        out["error"] = f"{deliverable} not found"
    except (ValueError, flagopt.ConfigError) as exc:
        out["error"] = f"invalid config: {exc}"
    else:
        out["config_valid"] = True
        out["candidate"] = cand
        out["xla_flags"] = xla
        out["changed_from_baseline"] = {
            k: v
            for k, v in resolved.items()
            if spec["baseline"]["maxtext_overrides"].get(k) != v
        }
    if not out["config_valid"]:
        log(out["error"])
        return out

    kube = flagopt.Kube(cmd=_kubectl_cmd(),
                        namespace=flagopt.cluster_settings(spec)["namespace"])
    runs_dir = log_dir() / "runs"

    # 2. Free the slice. Through the kube-gate wrapper, `--all` is rewritten
    # to this trial's label; with a plain kubectl (local testing) only flagopt
    # agent Jobs are touched.
    for kind in ("jobs", "services"):
        sel = ["--all"] if _kube_gated() and kind == "jobs" else [
            "-l", "app=tpu-flagopt,flagopt-role=agent"
        ]
        try:
            kube.run(["delete", kind] + sel + ["--wait=false"],
                     timeout=180,
                     check=False)
        except flagopt.KubeError as exc:
            log(f"cleanup of leftover {kind} failed: {exc}")

    def remaining() -> float:
        return deadline - time.time()

    def run(label: str, c: dict, w: str, runs_left: int) -> dict:
        # Share what is left of the deadline between the runs still to come:
        # all of it for the last run, half while a live baseline is pending.
        budget = max(600.0, remaining() / max(1, runs_left))
        run_timeout = int(min(meas["run_timeout_sec"], budget))
        res = flagopt.run_candidate(spec,
                                    c,
                                    label=label,
                                    steps=steps,
                                    weights=w,
                                    out_dir=runs_dir,
                                    role="verifier",
                                    kube=kube,
                                    queue_timeout_s=int(
                                        min(meas["queue_timeout_sec"] * 2,
                                            budget * 0.6)),
                                    run_timeout_s=run_timeout,
                                    log=log)
        res["run_timeout_s"] = run_timeout
        # A run cut short by the verifier's own deadline, rather than by the
        # task's run timeout, says nothing about the config.
        res["timeout_shortened"] = run_timeout < int(meas["run_timeout_sec"])
        return res

    def run_steady(label: str, c: dict, w: str, runs_left: int) -> dict:
        res = run(label, c, w, runs_left)
        if res["ok"] and not _stable(spec, res) and remaining() > 1800:
            log(f"{label}: step time noisy (cv={res['summary'].get('step_time_cv')}); re-running once"
                )
            again = run(label + "n", c, w, runs_left)
            inf = float("inf")
            if again["ok"] and again["summary"].get("step_time_cv",
                                                    inf) < res["summary"].get(
                                                        "step_time_cv", inf):
                res = again
        return res

    baseline_cand = flagopt.baseline_candidate(spec)
    stamp = time.strftime("%H%M%S")

    # 3. Baseline: recorded once when the task was built, so every trial is
    # scored against the same measurement and spends no TPU time on it.
    recorded = load_recorded_baseline(spec, steps)
    if recorded and weights in recorded:
        base = dict(recorded[weights], ok=True)
        out["baseline_source"] = "recorded"
        log(f"using the recorded {weights}-weights baseline "
            f"(p50 {base['summary']['step_time_p50_s']:.4f}s)")
    else:
        out["baseline_source"] = "live"
        log("no recorded baseline for this task; measuring it live")
        base = run_steady(f"vb{stamp}", baseline_cand, weights, 2)
        if not base["ok"] and weights == "checkpoint" and remaining() > 1800:
            log("baseline with checkpoint weights failed; using random init for both runs"
                )
            out["baseline_checkpoint_attempt"] = _brief(base)
            weights = "random"
            base = run_steady(f"vbr{stamp}", baseline_cand, weights, 2)
        if not base["ok"]:
            out["infra_error"] = True
            out["baseline_run"] = _brief(base)
            out["error"] = "baseline run failed: " + str(base.get("status"))
            log(out["error"])
            return out

    # 4. Candidate.
    if remaining() < 900:
        out["infra_error"] = True
        out["error"] = "verifier deadline reached before the candidate run"
        return out
    cand_res = run_steady(f"vc{stamp}", cand, weights, 1)
    if (not cand_res["ok"] and weights == "checkpoint"
            and out["baseline_source"] == "recorded"
            and "random" in (recorded or {}) and remaining() > 1800):
        log("candidate failed with checkpoint weights; retrying with random init against the recorded random-init baseline"
            )
        out["candidate_checkpoint_attempt"] = _brief(cand_res)
        weights = "random"
        base = dict(recorded["random"], ok=True)
        cand_res = run_steady(f"vcr{stamp}", cand, weights, 1)
    out["weights_used"] = weights
    out["baseline_run"] = _brief(base)
    out["candidate_run"] = _brief(cand_res)
    out["candidate_ran"] = bool(cand_res["ok"])
    if not cand_res["ok"]:
        out["error"] = (
            f"candidate run did not complete: status={cand_res.get('status')} "
            f"errors={cand_res.get('error_signatures')} diag={cand_res.get('pod_diagnosis')}"
        )
        status = cand_res.get("status")
        if (status in ("QUEUE_TIMEOUT", "ERROR", "VANISHED")
                and not cand_res.get("error_signatures")) or (
                    status == "RUN_TIMEOUT"
                    and cand_res.get("timeout_shortened")):
            out["infra_error"] = True
        log(out["error"])
        return out

    # 5. Accuracy gate.
    gate = flagopt.accuracy_gate(spec,
                                 base.get("metrics", []),
                                 cand_res["metrics"],
                                 loss_rtol=loss_rtol,
                                 grad_norm_rtol=gn_rtol)
    out["gate"] = gate
    out["accuracy_gate"] = bool(gate["pass"])

    # 6. Speedup.
    bs, cs = base["summary"], cand_res["summary"]
    if not (bs.get("step_time_p50_s") and cs.get("step_time_p50_s")):
        out["infra_error"] = True
        out["error"] = (f"no steady-state step time to compare "
                        f"({steps} steps, warmup {meas['warmup_steps']})")
        log(out["error"])
        return out
    sp = bs["step_time_p50_s"] / cs["step_time_p50_s"]
    out.update({
        "speedup": sp,
        "throughput_improvement_pct": (sp - 1.0) * 100.0,
        "baseline_step_time_p50_s": bs["step_time_p50_s"],
        "baseline_step_time_p90_s": bs.get("step_time_p90_s"),
        "candidate_step_time_p50_s": cs["step_time_p50_s"],
        "candidate_step_time_p90_s": cs.get("step_time_p90_s"),
        "candidate_step_time_cv": cs.get("step_time_cv"),
        "baseline_tokens_per_sec": bs.get("tokens_per_sec"),
        "candidate_tokens_per_sec": cs.get("tokens_per_sec"),
    })
    log(f"speedup {sp:.4f} ({(sp - 1) * 100:+.2f}% throughput), "
        f"gate={'PASS' if gate['pass'] else 'FAIL'}")
    return out


def _write_report(result: dict, workspace: Path) -> None:
    """verification.json / .md for humans, and the agent's own records."""
    try:
        d = log_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / "verification.json").write_text(
            json.dumps(result, indent=2, default=str))
        lines = [f"# Verification: {result.get('task_id')}", ""]
        for k in ("improvement_score", "speedup", "throughput_improvement_pct",
                  "config_valid", "candidate_ran", "accuracy_gate",
                  "infra_error", "baseline_source", "baseline_step_time_p50_s",
                  "candidate_step_time_p50_s", "candidate_step_time_p90_s"):
            if k in result:
                lines.append(f"- **{k}**: {result[k]}")
        if result.get("error"):
            lines += ["", f"**Error:** {result['error']}"]
        gate = result.get("gate")
        if gate:
            lines += [
                "", "## Accuracy gate", "",
                f"pass={gate['pass']} reasons={gate['reasons']}", "",
                "| step | loss base | loss cand | rel diff | gnorm base | gnorm cand | rel diff |",
                "|---|---|---|---|---|---|---|"
            ]
            for r in gate["per_step"]:
                lines.append(
                    f"| {r['step']} | {r['loss_base']:.6f} | {r['loss_cand']:.6f} | {r['loss_rel_diff']:.2e} | "
                    f"{r.get('grad_norm_base', float('nan')):.5f} | {r.get('grad_norm_cand', float('nan')):.5f} | "
                    f"{r.get('grad_norm_rel_diff', float('nan')):.2e} |")
        if result.get("changed_from_baseline") is not None:
            lines += [
                "", "## Agent config (changed from baseline)", "", "```json",
                json.dumps(
                    {
                        "maxtext_overrides": result["changed_from_baseline"],
                        "xla_flags": result.get("xla_flags")
                    },
                    indent=2), "```"
            ]
        (d / "verification.md").write_text("\n".join(lines) + "\n")
        artifacts = Path(
            os.environ.get("FLAGOPT_ARTIFACTS_DIR", "/logs/artifacts"))
        for name in ("output", "runs"):
            src = workspace / name
            if src.exists():
                shutil.copytree(src,
                                artifacts / "agent" / name,
                                dirs_exist_ok=True)
    except OSError as exc:
        log(f"could not write the verification report: {exc}")
