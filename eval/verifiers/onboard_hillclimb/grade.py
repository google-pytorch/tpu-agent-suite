#!/usr/bin/env python3
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

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "harbor>=0.23.0",
#     "harbor-rewardkit>=0.2.2",
#     "google-auth",
#     "numpy",
# ]
# ///
"""RewardKit Regrade Engine for Harbor Jobs and Serving Benchmarks.

Runs latest RewardKit verifiers on Harbor job directories or remote jobs,
computes deterministic speedup via calculate_speedup.py, prints a reward diff,
and produces finalized reward.json and reward-details.json verifier payloads.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / ".git").exists() or (parent / "eval").exists():
            return parent
    return Path.cwd()


_REPO_ROOT = _find_repo_root()
_TPU_INFERENCE_DIR = _REPO_ROOT / "eval" / "datasets" / "tpu-inference-vllm"
_NIGHTLY_TASK_TESTS: Dict[str, Path] = {
    "gemma":
    _TPU_INFERENCE_DIR / "tpu-inference-gemma-4-31b-it-vllm" / "tests",
    "gpt-oss": _TPU_INFERENCE_DIR / "tpu-inference-gpt-oss-20b-vllm" / "tests",
    "qwen": _TPU_INFERENCE_DIR / "tpu-inference-qwen3-8-27b-vllm" / "tests",
}

UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
DEFAULT_DOWNLOAD_DIR = Path.home() / ".cache" / "harbor_jobs"
DEFAULT_EXPORT_DIR = _REPO_ROOT / "jobs"

DEFAULT_VERIFIERS = [
    "onboarding_success",
    "efficiency",
    "cheating",
    "speedup",
    "execution_report",
    "invoked_subagent",
]
AVAILABLE_VERIFIERS = list(DEFAULT_VERIFIERS)

VERIFIER_WEIGHTS: Dict[str, float] = {
    "onboarding_success": 0.3,
    "hillclimb_success": 0.3,
    "speedup_score": 0.3,
    "speedup": 0.3,
    "efficiency": 0.1,
    "cheating": -1.0,
    "execution_report": 0.0,
    "invoked_subagent": 0.0,
}

# --- Small I/O & Logging Helpers ---


def log_step(step_num: int, title: str) -> None:
    print(f"\n[{step_num}] {title}", file=sys.stderr, flush=True)


def log_info(message: str) -> None:
    print(f"  -> {message}", file=sys.stderr, flush=True)


def log_block(text: str, indent: str = "  ") -> None:
    for line in text.splitlines():
        print(f"{indent}{line}", file=sys.stderr, flush=True)


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    """Safely reads a JSON dictionary from disk, returning {} on any failure."""
    if path and path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    return {}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _copy_if_different(src: Path, dst: Path) -> None:
    if src.is_file() and src.resolve() != dst.resolve():
        shutil.copy2(src, dst)


def _harbor_cmd(*args: str) -> List[str]:
    harbor_bin = shutil.which("harbor")
    base = [harbor_bin
            ] if harbor_bin else ["uvx", "--from", "harbor", "harbor"]
    return [*base, *args]


def _is_trial_dir(path: Path, include_traj: bool = True) -> bool:
    markers = ("agent", "artifacts", "verifier",
               *(["trajectory.json"] if include_traj else []))
    return any((path / m).exists() for m in markers)


def _write_verifier_outputs(verifier_dir: Path, scores: Dict[str, Any],
                            details: Dict[str, Any], log_text: str) -> None:
    verifier_dir.mkdir(parents=True, exist_ok=True)
    _write_json(verifier_dir / "reward.json", scores)
    _write_json(verifier_dir / "reward-details.json", details)
    (verifier_dir / "test-stdout.txt").write_text(log_text, encoding="utf-8")
    (verifier_dir / "regrade.log").write_text(log_text, encoding="utf-8")


# --- Discovery & Resolution ---


def extract_job_and_trial(target: str) -> Tuple[Optional[str], Optional[str]]:
    uuids = UUID_PATTERN.findall(target)
    return (uuids[0],
            uuids[1] if len(uuids) > 1 else None) if uuids else (None, None)


def locate_local_job_dir(target_id: str,
                         search_roots: List[Path]) -> Optional[Path]:
    target_lower = target_id.lower()
    for root in search_roots:
        if not root.exists():
            continue
        candidates: List[Path] = []
        try:
            for d in root.iterdir():
                if d.is_dir() and not d.name.startswith("regraded_"):
                    candidates.append(d)
                    try:
                        candidates.extend(s for s in d.iterdir() if s.is_dir())
                    except OSError:
                        pass
        except OSError:
            continue

        for cand in candidates:
            if target_lower in cand.name.lower():
                return cand
            if str(_read_json(cand / "result.json").get(
                    "id", "")).lower() == target_lower:
                return cand
            cfg = _read_json(cand / "config.json")
            if target_lower in (
                    str(cfg.get("job_id", "")).lower(),
                    str(cfg.get("job_name", "")).lower(),
            ):
                return cand
    return None


def download_job(job_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"Downloading job {job_id} into {output_dir} via harbor CLI...")
    proc = subprocess.run(
        _harbor_cmd("job", "download", job_id, "-o", str(output_dir)),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to download job {job_id}:\n{proc.stderr or proc.stdout}")

    job_dir = locate_local_job_dir(job_id, [output_dir])
    if not job_dir:
        raise FileNotFoundError(
            f"Job {job_id} downloaded, but not found under {output_dir}.")

    orig_name = _read_json(job_dir / "config.json").get("job_name")
    if orig_name and job_dir.name != orig_name:
        desired_dir = job_dir.parent / orig_name
        if not desired_dir.exists():
            job_dir.rename(desired_dir)
            job_dir = desired_dir
            log_info(
                f"Renamed downloaded folder to original job_name: {orig_name}")

    log_info(f"Downloaded and verified job directory: {job_dir}")
    return job_dir


def find_trial_directories(job_dir: Path,
                           target_trial_id: Optional[str] = None
                           ) -> List[Path]:
    if _is_trial_dir(job_dir):
        return [job_dir]

    trials: List[Path] = []
    target_lower = target_trial_id.lower() if target_trial_id else None
    for child in job_dir.iterdir():
        if not child.is_dir() or not _is_trial_dir(child):
            continue
        if target_lower:
            cid = (_read_json(child / "result.json").get("id")
                   or _read_json(child / "config.json").get("trial_id") or "")
            if str(cid).lower(
            ) == target_lower or target_lower in child.name.lower():
                return [child]
        trials.append(child)
    return trials


def find_trajectory_file(target_dir: Path) -> Optional[Path]:
    for rel in (
            "agent/trajectory.json",
            "logs/agent/trajectory.json",
            "trajectory.json",
            "logs/trajectory.json",
    ):
        c = target_dir / rel
        if c.is_file():
            return c.resolve()

    for path in target_dir.glob("**/trajectory.json"):
        if path.is_file() and ".git" not in path.parts:
            return path.resolve()

    ws_session = target_dir / "agent" / "sessions" / "projects" / "-workspace"
    if ws_session.is_dir() and any(ws_session.glob("*.jsonl")):
        agent_dir = target_dir / "agent"
        out_traj = agent_dir / "trajectory.json"
        recover_script = (
            "import json, sys; from pathlib import Path; "
            "from harbor.agents.installed.claude_code import ClaudeCode; "
            "agent_dir = Path(sys.argv[1]); "
            "cc = ClaudeCode(logs_dir=agent_dir); "
            "traj = cc._convert_events_to_trajectory(agent_dir / 'sessions' / 'projects' / '-workspace'); "
            "(agent_dir / 'trajectory.json').write_text(json.dumps(traj.to_json_dict(), indent=2, ensure_ascii=False))"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", recover_script,
                 str(agent_dir)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode == 0 and out_traj.is_file():
                log_info(
                    f"Recovered missing trajectory.json from {ws_session}")
                return out_traj.resolve()
        except Exception:
            pass
    return None


def _match_known_model(text: str) -> Optional[str]:
    lower = text.lower()
    mapping = [
        (("gemma", ), "google/gemma-4-31B-it"),
        (("gpt-oss", "gpt_oss"), "openai/gpt-oss-20b"),
        (("qwen", ), "Qwen/Qwen3.8-27B"),
        (("mistral", ), "mistralai/Mistral-Small-24B-Instruct-2501"),
        (("llama", ), "meta-llama/Llama-3.3-70B-Instruct"),
    ]
    for keys, model in mapping:
        if any(k in lower for k in keys):
            return model
    return None


def detect_model_name(target_dir: Path) -> str:
    name_text = f"{target_dir.name} {target_dir.parent.name if target_dir.parent else ''}"
    if matched := _match_known_model(name_text):
        return matched

    text_blobs: List[str] = []
    for d in (target_dir, target_dir.parent):
        if not d or not d.exists():
            continue
        for fname in ("config.json", "task.toml", "result.json"):
            p = d / fname
            if not p.is_file():
                continue
            if fname.endswith(".json"):
                data = _read_json(p)
                for k in ("job_name", "task_name", "trial_name"):
                    if matched := _match_known_model(str(data.get(k, ""))):
                        return matched
            try:
                text_blobs.append(p.read_text(encoding="utf-8").lower())
            except Exception:
                pass

    for cfg in target_dir.glob("artifacts/**/config.json"):
        try:
            text_blobs.append(cfg.read_text(encoding="utf-8")[:4096].lower())
        except Exception:
            pass

    for rel in ("agent/trajectory.json", "trajectory.json",
                "logs/agent/trajectory.json"):
        traj = target_dir / rel
        if traj.is_file():
            try:
                with traj.open("r", encoding="utf-8") as f:
                    text_blobs.append(f.read(8192).lower())
            except Exception:
                pass
            break

    return _match_known_model(" ".join(text_blobs)) or "the target model"


def resolve_verifiers_dir(model_name: Optional[str] = None,
                          trial_dir: Optional[Path] = None) -> Path:
    search_str = (model_name or "").lower()
    if trial_dir is not None and search_str in ("", "auto",
                                                "the target model"):
        search_str = detect_model_name(trial_dir).lower()
    if "gpt-oss" in search_str or "gpt_oss" in search_str:
        return _NIGHTLY_TASK_TESTS["gpt-oss"]
    if "qwen" in search_str:
        return _NIGHTLY_TASK_TESTS["qwen"]
    return _NIGHTLY_TASK_TESTS["gemma"]


def _is_agent_suite_trial(trial_dir: Optional[Path]) -> bool:
    if trial_dir is None:
        return False
    manifest = trial_dir / "artifacts" / "manifest.json"
    if manifest.is_file():
        try:
            m_text = manifest.read_text(encoding="utf-8")
            if "tpu-agent-suite" in m_text:
                return True
        except Exception:
            pass

    if any((trial_dir / "artifacts" / s).is_dir() for s in (
            "tpu-model-onboarding",
            "tpu-diagnostics",
            "tpu-kernel-development",
    )):
        return True

    suite_markers = ("/packages/", ".tpu-agent-suite/packages",
                     "/workspace/.claude")
    for d in (trial_dir, trial_dir.parent):
        if not d or not d.exists():
            continue
        cfg_file = d / "config.json"
        if cfg_file.is_file():
            try:
                text = cfg_file.read_text(encoding="utf-8")
                if any(m in text for m in suite_markers):
                    return True
                jname = str(json.loads(text).get("job_name", "")).lower()
                if "agent-suite" in jname:
                    return True
                if "baseline" in jname:
                    return False
            except Exception:
                pass
        if "agent-suite" in d.name.lower():
            return True
        if "baseline" in d.name.lower():
            return False
    return False


def _collect_suite_agent_names() -> List[str]:
    names = {"tpu-knowledge-base", "tpu-diagnostics"}
    agents_dir = _REPO_ROOT / "package" / "agents"
    if agents_dir.is_dir():
        for md_file in agents_dir.glob("*.md"):
            try:
                if m := re.search(
                        r'^name:\s*"?([^"\n]+?)"?\s*$',
                        md_file.read_text(encoding="utf-8"),
                        re.M,
                ):
                    names.add(m.group(1).strip())
            except Exception:
                pass
    return sorted(names)


# --- Staging & Execution ---


def stage_verifier_tests(
    stage_dir: Path,
    verifiers: List[str],
    trajectory_path: Optional[Path],
    model_name: str,
    judge_model: Optional[str] = None,
    verifiers_dir: Optional[Path] = None,
    workspace_dir: Optional[Path] = None,
    logs_dir: Optional[Path] = None,
) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    source_tests_dir = verifiers_dir or resolve_verifiers_dir(model_name)

    tests_str = str(stage_dir.resolve())
    ws_str = str((workspace_dir or stage_dir.parent).resolve())
    resolved_logs = (logs_dir or (stage_dir.parent / "logs")).resolve()
    logs_str = str(resolved_logs)
    staged_artifacts = resolved_logs / "artifacts"
    staged_artifacts.mkdir(parents=True, exist_ok=True)

    if workspace_dir is not None:
        for rel in (
                "artifacts/logs/artifacts",
                "artifacts/workspace_logs_artifacts",
                "artifacts",
                "logs/artifacts",
        ):
            cand_art = workspace_dir / rel
            if not cand_art.is_dir():
                continue
            for child in cand_art.iterdir():
                if (child.is_file()
                        and not (staged_artifacts / child.name).exists()
                        and child.stat().st_size <= 65536):
                    try:
                        shutil.copy2(child, staged_artifacts / child.name)
                    except Exception:
                        pass
                elif child.is_dir() and child.name not in ("serving_bench",
                                                           "serving_bench_opt",
                                                           "bench",
                                                           "benchmarks"):
                    try:
                        shutil.copytree(child,
                                        staged_artifacts / child.name,
                                        dirs_exist_ok=True)
                    except Exception:
                        pass
            for sub_name in ("serving_bench", "serving_bench_opt", "bench",
                             "benchmarks"):
                sub_dir = cand_art / sub_name
                if sub_dir.is_dir():
                    for md_f in sub_dir.glob("*.md"):
                        flat_dst = staged_artifacts / f"{sub_name}_{md_f.name}"
                        if not flat_dst.exists():
                            try:
                                shutil.copy2(md_f, flat_dst)
                            except Exception:
                                pass

    reward_toml_lines = [
        "[[reward]]", 'name = "reward"', 'aggregation = "weighted-sum"', "",
        "[reward.weights]"
    ]

    for v in verifiers:
        src = source_tests_dir / v
        if not src.exists():
            continue

        dst = stage_dir / v
        shutil.copytree(src, dst, symlinks=False)

        if v == "speedup":
            (dst / "judge.toml").unlink(missing_ok=True)
            calc_src = Path(__file__).resolve().parent / "calculate_speedup.py"
            if calc_src.is_file():
                shutil.copy2(calc_src, dst / "calculate_speedup.py")

        judge_file = dst / "judge.toml"
        if judge_file.exists():
            content = judge_file.read_text(encoding="utf-8")
            if trajectory_path:
                content = content.replace(
                    'atif-trajectory = "/logs/agent/trajectory.json"',
                    f'atif-trajectory = "{trajectory_path.resolve()}"',
                )
            content = content.replace('"/logs/artifacts/',
                                      f'"{staged_artifacts.resolve()}/')
            content = content.replace('"/logs/artifacts"',
                                      f'"{staged_artifacts.resolve()}"')
            content = content.replace("{MODEL_NAME}", model_name)
            if judge_model:
                content = re.sub(r'judge\s*=\s*"[^"]+"',
                                 f'judge = "{judge_model}"', content)
            judge_file.write_text(content, encoding="utf-8")

        if v == "efficiency":
            for py_file in dst.glob("*.py"):
                py_content = py_file.read_text(encoding="utf-8")
                py_file.write_text(
                    py_content.replace('"/logs/', f'"{logs_str}/').replace(
                        '"/tests/', f'"{tests_str}/'),
                    encoding="utf-8",
                )

        inv_check_file = dst / "check.py"
        if v == "invoked_subagent" and inv_check_file.exists():
            inv_content = inv_check_file.read_text(encoding="utf-8").replace(
                '"/logs/', f'"{logs_str}/')
            if _is_agent_suite_trial(workspace_dir):
                manifest_path = dst / "tpu-agent-suite-build.json"
                _write_json(
                    manifest_path, {
                        "contracts": {
                            name: {}
                            for name in _collect_suite_agent_names()
                        }
                    })
            else:
                manifest_path = dst / "no-suite" / "tpu-agent-suite-build.json"
            inv_content = inv_content.replace(
                '"/root/.tpu-agent-suite/packages/*/tpu-agent-suite-build.json"',
                f'"{manifest_path.resolve()}"',
            )
            inv_check_file.write_text(inv_content, encoding="utf-8")

        reward_toml_lines.append(f"{v} = {VERIFIER_WEIGHTS.get(v, 0.0)}")

    local_test_sh = Path(__file__).resolve().parent / "test.sh"
    test_sh_src = local_test_sh if local_test_sh.is_file() else (
        source_tests_dir / "test.sh")
    if test_sh_src.exists():
        content = test_sh_src.read_text(encoding="utf-8")
        if rewardkit_bin := shutil.which("rewardkit"):
            content = re.sub(
                r'uvx\s+--with\s+google-auth\s+--from\s+"harbor-rewardkit[^"]*"\s+rewardkit',
                rewardkit_bin,
                content,
            )
        for old, new in (
            (
                "--max-concurrent-llm 2",
                "--output /logs/verifier/reward.json --max-concurrent-llm 2",
            ),
            ("/logs", logs_str),
            ("/tests", tests_str),
            ("/workspace", ws_str),
        ):
            content = content.replace(old, new)
        staged_test_sh = stage_dir / "test.sh"
        staged_test_sh.write_text(content, encoding="utf-8")
        staged_test_sh.chmod(0o755)

    (stage_dir / "reward.toml").write_text("\n".join(reward_toml_lines) + "\n",
                                           encoding="utf-8")
    return stage_dir


# --- Diff & Formatting ---


def _normalize_scores(scores: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(scores)
    if "efficiency" not in out and "efficiency_raw" in out:
        out["efficiency"] = out["efficiency_raw"]
    return out


def compute_reward_diff(original: Dict[str, Any],
                        new: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    orig_copy = _normalize_scores(original)
    new_copy = _normalize_scores(new)
    ignored = {"speedup_ratio", "candidate_pool"}

    base_order = [
        "reward",
        "speedup",
        "speedup_score",
        "target_speedup",
        "onboarding_success",
        "hillclimb_success",
        "efficiency",
        "trajectory_score",
        "valid_tool_calls",
        "framework_directed_penalty",
        "cheating",
        "invoked_subagent",
    ]
    all_keys = list(
        dict.fromkeys([*base_order, *orig_copy.keys(), *new_copy.keys()]))

    diff: Dict[str, Dict[str, Any]] = {}
    for k in all_keys:
        if k in ignored or k.endswith(("_raw", "_gated")):
            continue
        old_v, new_v = orig_copy.get(k), new_copy.get(k)
        num_types = (int, float)
        is_num_old = isinstance(old_v,
                                num_types) and not isinstance(old_v, bool)
        is_num_new = isinstance(new_v,
                                num_types) and not isinstance(new_v, bool)
        delta = round(float(new_v) -
                      float(old_v), 4) if (is_num_old and is_num_new) else None
        diff[k] = {"original": old_v, "regraded": new_v, "delta": delta}
    return diff


def _fmt_metric_val(v: Any) -> str:
    if isinstance(v, bool) or isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4f}"
    return "none" if v is None else str(v)


def format_reward_diff(diff: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "Reward Diff (original -> regraded):",
        f"  {'Metric':<22} {'Original':<12} {'Regraded':<12} {'Diff'}",
        "  " + "─" * 56,
    ]
    priority = [
        "reward",
        "speedup",
        "speedup_score",
        "target_speedup",
        "onboarding_success",
        "hillclimb_success",
        "efficiency",
        "trajectory_score",
        "valid_tool_calls",
        "framework_directed_penalty",
        "optimization_steps",
        "cheating",
        "cheating_outcome",
        "invoked_subagent",
        "execution_report",
    ]
    ordered_keys = [k for k in priority if k in diff
                    ] + [k for k in sorted(diff) if k not in priority]

    for k in ordered_keys:
        old_v, new_v, delta = diff[k]["original"], diff[k]["regraded"], diff[
            k]["delta"]
        old_str, new_str = _fmt_metric_val(old_v), _fmt_metric_val(new_v)
        suffix = "x" if k == "speedup" else ""
        both_int = isinstance(old_v, int) and isinstance(new_v, int)

        if delta is not None:
            if delta == 0:
                diff_str = "="
            elif both_int:
                diff_str = f"{int(delta):+d}"
            else:
                diff_str = f"{delta:+.4f}{suffix}"
        elif old_v is None and new_v is not None:
            diff_str = "new"
        elif old_v is not None and new_v is None:
            diff_str = "removed"
        else:
            diff_str = ""

        lines.append(f"  {k:<22} {old_str:<12} {new_str:<12} {diff_str}")
    return "\n".join(lines)


# --- Core Regrade & Export ---


def regrade_trial(
    trial_dir: Path,
    verifiers: Optional[List[str]] = None,
    apply_changes: bool = False,
    model_name: Optional[str] = None,
    judge_model: Optional[str] = None,
    trial_index: int = 1,
    total_trials: int = 1,
) -> Dict[str, Any]:
    active_verifiers = verifiers or DEFAULT_VERIFIERS
    resolved_model = (detect_model_name(trial_dir) if not model_name
                      or model_name.lower() in ("auto", "the target model")
                      else model_name)
    verifiers_dir = resolve_verifiers_dir(resolved_model, trial_dir)
    trajectory_path = find_trajectory_file(trial_dir)
    verifier_dir = trial_dir / "verifier"

    trial_title = f"'{trial_dir.name}'" if total_trials == 1 else f"trial {trial_index}/{total_trials} ('{trial_dir.name}')"
    log_step(2, f"Staging verifier suite for {trial_title}")

    original_scores = _read_json(verifier_dir / "reward.json")
    if not original_scores:
        vr = _read_json(trial_dir / "result.json").get("verifier_result")
        if isinstance(vr, dict) and isinstance(vr.get("rewards"), dict):
            original_scores = vr["rewards"]

    verifier_env: Dict[str, str] = {}
    for p in (trial_dir / "config.json", trial_dir.parent / "config.json"):
        v_env = _read_json(p).get("verifier", {}).get("env")
        if isinstance(v_env, dict):
            verifier_env.update(v_env)

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        staged_tests, staged_logs = temp_dir / "tests", temp_dir / "logs"
        (staged_logs / "agent").mkdir(parents=True, exist_ok=True)
        (staged_logs / "verifier").mkdir(parents=True, exist_ok=True)

        if trajectory_path and trajectory_path.exists():
            shutil.copy2(trajectory_path,
                         staged_logs / "agent" / "trajectory.json")
            shutil.copy2(trajectory_path, staged_logs / "trajectory.json")

        traj_verifiers = {
            "onboarding_success", "hillclimb_success", "cheating",
            "execution_report", "efficiency", "invoked_subagent"
        }
        effective_verifiers = (list(active_verifiers) if trajectory_path else [
            v for v in active_verifiers if v not in traj_verifiers
        ])

        stage_verifier_tests(
            stage_dir=staged_tests,
            verifiers=effective_verifiers,
            trajectory_path=trajectory_path,
            model_name=resolved_model,
            judge_model=judge_model,
            verifiers_dir=verifiers_dir,
            workspace_dir=trial_dir,
            logs_dir=staged_logs,
        )

        env = os.environ.copy()
        env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
        if trajectory_path:
            env["ATIF_TRAJECTORY"] = str(trajectory_path.resolve())
        for k, v in {
                "VERTEXAI_PROJECT": "tpu-nexus-767207",
                "VERTEXAI_LOCATION": "global",
                "CLOUD_ML_REGION": "global",
                "DEFAULT_MAX_RETRIES": "10",
                "INITIAL_RETRY_DELAY": "2.0",
                "MAX_RETRY_DELAY": "60.0",
        }.items():
            env.setdefault(k, v)
        env.update(verifier_env)

        test_stdout = ""
        test_script = staged_tests / "test.sh"
        log_step(
            3, f"Running verifiers via test.sh + RewardKit on {trial_title}")
        if effective_verifiers and test_script.exists():
            proc = subprocess.run(
                ["bash", str(test_script.resolve())],
                env=env,
                capture_output=True,
                text=True,
            )
            test_stdout = "\n".join(filter(None, [proc.stdout, proc.stderr]))
            if proc.returncode != 0:
                log_info(
                    f"Verifier script exited with non-zero code {proc.returncode}"
                )

        final_scores = _read_json(staged_logs / "verifier" / "reward.json")
        final_details = _read_json(staged_logs / "verifier" /
                                   "reward-details.json")

        if not final_scores and test_stdout:
            tail = [
                line for line in test_stdout.strip().splitlines()
                if line.strip()
            ][-10:]
            log_info("Verifier produced no rewards. Recent output:\n" +
                     "\n".join(f"  {line}" for line in tail))

        reward_diff = compute_reward_diff(original_scores, final_scores)
        diff_table = format_reward_diff(reward_diff)
        log_block(diff_table)

        if apply_changes:
            _write_verifier_outputs(verifier_dir, final_scores, final_details,
                                    test_stdout)
            log_info(f"Updated source verifier files at: {verifier_dir}")

    return {
        "trial_name": trial_dir.name,
        "trial_path": str(trial_dir),
        "reward": final_scores,
        "original_reward": original_scores,
        "reward_diff": reward_diff,
        "diff_table": diff_table,
        "reward_details": final_details,
        "harbor_log": test_stdout,
        "active_verifiers": active_verifiers,
        "model_name": resolved_model,
        "trajectory_path": str(trajectory_path) if trajectory_path else None,
    }


def export_slim_regraded_job(
    job_dir: Path,
    regraded_results: List[Dict[str, Any]],
    export_parent_dir: Path = Path("jobs"),
    new_uuids: bool = True,
    orig_job_id: Optional[str] = None,
    slim: bool = False,
) -> Path:
    export_parent_dir.mkdir(parents=True, exist_ok=True)
    if _is_trial_dir(job_dir, include_traj=False) and (
        (job_dir.parent / "result.json").exists() or
        (job_dir.parent / "config.json").exists()):
        actual_job_dir = job_dir.parent
    else:
        actual_job_dir = job_dir

    root_cfg = _read_json(actual_job_dir / "config.json")
    orig_job_name = (root_cfg.get("job_name")
                     or actual_job_dir.name).removeprefix("regraded_")
    root_res = _read_json(actual_job_dir / "result.json")
    resolved_orig_job_id = root_res.get("id") or orig_job_id or root_cfg.get(
        "job_id")

    export_job_dir = export_parent_dir / f"regraded_{orig_job_name}"
    export_job_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(export_job_dir / ".harbor-upload", ignore_errors=True)

    log_step(
        4,
        f"Exporting {'slim ' if slim else ''}regraded job to '{export_job_dir}'"
    )
    log_info(
        "Excluded bulky runtime artifacts (preserved original metadata + slim trajectory)"
        if slim else
        "Preserved original job name, artifacts, full trajectory, and trial metadata"
    )

    _write_json(export_job_dir / "config.json", {
        **root_cfg, "job_name": orig_job_name
    })

    now_iso = datetime.now(timezone.utc).isoformat()
    export_root_res = {
        **root_res,
        **({
            "id": str(uuid.uuid4())
        } if new_uuids else {}),
        "started_at": now_iso,
        "updated_at": now_iso,
        "finished_at": now_iso,
    }
    _write_json(export_job_dir / "result.json", export_root_res)

    for extra in ("lock.json", "job.log", "analysis.md"):
        _copy_if_different(actual_job_dir / extra, export_job_dir / extra)

    for res in regraded_results:
        trial_path = Path(res["trial_path"])
        trial_name = res.get("trial_name") or trial_path.name
        dst_trial_dir = export_job_dir / trial_name
        dst_trial_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(dst_trial_dir / ".harbor-upload", ignore_errors=True)

        trial_cfg = _read_json(trial_path / "config.json")
        trial_res = _read_json(trial_path / "result.json")
        orig_trial_id = trial_res.get("id")

        if trial_cfg:
            export_trial_cfg = {
                **trial_cfg,
                **({
                    "job_id": export_root_res["id"]
                } if new_uuids and export_root_res.get("id") else {}),
            }
            _write_json(dst_trial_dir / "config.json", export_trial_cfg)
        else:
            _write_json(dst_trial_dir / "config.json",
                        {"trial_name": trial_name})

        src_lock, dst_lock = trial_path / "lock.json", dst_trial_dir / "lock.json"
        if src_lock.exists():
            _copy_if_different(src_lock, dst_lock)
        else:
            ai = trial_res.get("agent_info") or {}
            _write_json(
                dst_lock,
                {
                    "schema_version": 2,
                    "task": {
                        "name": trial_res.get("task_name", trial_name)
                    },
                    "agent": {
                        "name": ai.get("name", "agent"),
                        "model_name": ai.get("model_info", {}).get("name", "")
                    },
                    "verifier": {
                        "disable": False
                    },
                },
            )

        export_trial_res = dict(trial_res)
        export_trial_res.setdefault("trial_name", trial_name)
        export_trial_res.setdefault("task_name", orig_job_name)
        export_trial_res["verifier_result"] = {
            "rewards": res.get("reward", {})
        }

        if new_uuids:
            export_trial_res["id"] = str(uuid.uuid4())
            if isinstance(export_trial_res.get("config"),
                          dict) and export_root_res.get("id"):
                export_trial_res["config"] = {
                    **export_trial_res["config"], "job_id":
                    export_root_res["id"]
                }

        _write_json(dst_trial_dir / "result.json", export_trial_res)
        _write_verifier_outputs(
            dst_trial_dir / "verifier",
            res.get("reward", {}),
            res.get("reward_details", {}),
            res.get("harbor_log", ""),
        )

        dst_agent_dir = dst_trial_dir / "agent"
        dst_agent_dir.mkdir(parents=True, exist_ok=True)
        orig_traj_file = find_trajectory_file(trial_path)

        if not slim:
            for extra in ("trial.log", "analysis.md", "exception.txt"):
                _copy_if_different(trial_path / extra, dst_trial_dir / extra)
            for sub in ("artifacts", "agent", "steps"):
                src_sub, dst_sub = trial_path / sub, dst_trial_dir / sub
                if src_sub.is_dir() and src_sub.resolve() != dst_sub.resolve():
                    shutil.copytree(src_sub,
                                    dst_sub,
                                    symlinks=True,
                                    ignore_dangling_symlinks=True,
                                    dirs_exist_ok=True)
            if orig_traj_file:
                _copy_if_different(orig_traj_file,
                                   dst_agent_dir / "trajectory.json")
            continue

        # Slim mode
        shutil.rmtree(dst_trial_dir / "artifacts", ignore_errors=True)
        orig_traj = _read_json(orig_traj_file)
        ai = trial_res.get("agent_info") or {}
        agent_block = orig_traj.get("agent") or {
            "name":
            ai.get("name", "agent"),
            "version":
            ai.get("version", "1.0"),
            "model_name":
            ai.get("model_info", {}).get("name",
                                         res.get("model_name", "unknown")),
        }

        if resolved_orig_job_id and orig_trial_id:
            orig_url = f"https://hub.harborframework.com/jobs/{resolved_orig_job_id}/trials/{orig_trial_id}"
        elif resolved_orig_job_id:
            orig_url = f"https://hub.harborframework.com/jobs/{resolved_orig_job_id}"
        else:
            orig_url = f"original Harbor trial {trial_name}"

        steps = orig_traj.get("steps") or []
        first_ts = (steps[0].get("timestamp")
                    if steps else None) or trial_res.get("started_at")
        last_ts = (steps[-1].get("timestamp")
                   if steps else None) or trial_res.get("finished_at")

        slim_traj: Dict[str, Any] = {
            "schema_version":
            orig_traj.get("schema_version", "ATIF-v1.7"),
            "session_id":
            orig_traj.get("session_id") or trial_name,
            "agent":
            agent_block,
            "steps": [
                {
                    "step_id":
                    1,
                    "timestamp":
                    first_ts,
                    "source":
                    "system",
                    "message":
                    f"Manual regrade of original Harbor trial: {orig_url}. Please view the original trial for the full execution trajectory and artifacts.",
                },
                {
                    "step_id":
                    2,
                    "timestamp":
                    last_ts,
                    "source":
                    "agent",
                    "model_name":
                    agent_block.get("model_name") or res.get("model_name"),
                    "message":
                    f"Trial regraded with updated RewardKit verifiers. Full trajectory preserved at {orig_url}.",
                },
            ],
        }
        if "final_metrics" in orig_traj:
            slim_traj["final_metrics"] = orig_traj["final_metrics"]

        _write_json(dst_agent_dir / "trajectory.json", slim_traj)

    log_info(f"Copied files to {export_job_dir}")
    return export_job_dir


# --- CLI Entrypoint ---


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=
        "RewardKit Regrade Tool for TPU Serving benchmarks and Harbor jobs.")
    parser.add_argument(
        "target",
        type=str,
        help="Job ID (UUID), Harbor URL, or local path to job/trial directory."
    )
    parser.add_argument("--use-original-dir",
                        "--in-place",
                        "--apply",
                        dest="use_original_dir",
                        action="store_true",
                        help="Update verifier artifacts in-place.")
    parser.add_argument("--no-export",
                        action="store_true",
                        help="Skip exporting a regraded copy of the job.")
    parser.add_argument("--slim",
                        "--trim",
                        dest="slim",
                        action="store_true",
                        default=False,
                        help="Export a slim copy stripping runtime artifacts.")
    parser.add_argument(
        "--full",
        dest="slim",
        action="store_false",
        help="Preserve full original artifacts and trajectory (default).")
    parser.add_argument("--upload",
                        action="store_true",
                        help="Upload the regraded job using `harbor upload`.")
    parser.add_argument("--onboarding-model-name",
                        "--model-name",
                        dest="onboarding_model_name",
                        type=str,
                        default="the target model",
                        help="Target model name or 'auto'.")
    parser.add_argument(
        "--verifiers",
        type=str,
        default=",".join(DEFAULT_VERIFIERS),
        help=
        f"Comma-separated verifiers (Available: {','.join(AVAILABLE_VERIFIERS)})"
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help="Directory to search for or download Harbor jobs into.",
    )
    parser.add_argument("--verbose",
                        "-v",
                        action="store_true",
                        help="Display verbose output.")
    parser.add_argument("--keep-uuids",
                        action="store_true",
                        help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    target = args.target.strip()
    job_id, target_trial_id = extract_job_and_trial(target)

    if Path(target).exists():
        job_dir = Path(target)
        log_step(1, f"Using job at '{job_dir}' for regrade")
    elif job_id:
        job_dir = locate_local_job_dir(
            job_id, [args.download_dir, _REPO_ROOT / "jobs"])
        if job_dir:
            log_step(1, f"Using existing local job at '{job_dir}' for regrade")
        else:
            log_step(
                1,
                f"Downloading job '{job_id}' from Harbor platform into {args.download_dir}"
            )
            job_dir = download_job(job_id, args.download_dir)
    else:
        print(
            f"Error: Target '{target}' is not an existing path and contains no valid UUID.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    trials = find_trial_directories(job_dir, target_trial_id=target_trial_id)
    if not trials:
        print(
            f"Error: No trial directories found inside {job_dir}.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    log_info(
        f"Discovered {len(trials)} trial(s) in '{job_dir.name}': {', '.join(t.name for t in trials)}"
    )
    selected_verifiers = [
        v.strip() for v in args.verifiers.split(",") if v.strip()
    ]

    all_results: List[Dict[str, Any]] = []
    for idx, trial in enumerate(trials, start=1):
        try:
            all_results.append(
                regrade_trial(
                    trial_dir=trial,
                    verifiers=selected_verifiers,
                    apply_changes=args.use_original_dir,
                    model_name=args.onboarding_model_name,
                    trial_index=idx,
                    total_trials=len(trials),
                ))
        except Exception as e:
            print(
                f"Warning: Failed to regrade trial {trial.name}: {e}",
                file=sys.stderr,
                flush=True,
            )

    if not all_results:
        print(
            "Error: Could not regrade any trials successfully.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    exported_path: Optional[Path] = None
    if not args.no_export and (not args.use_original_dir or args.upload):
        exported_path = export_slim_regraded_job(
            job_dir=job_dir,
            regraded_results=all_results,
            export_parent_dir=DEFAULT_EXPORT_DIR,
            new_uuids=not getattr(args, "keep_uuids", False),
            orig_job_id=job_id,
            slim=args.slim,
        )
        for res in all_results:
            res["exported_job_dir"] = str(exported_path)
    else:
        log_step(
            4,
            f"Export regraded job skipped ({'--use-original-dir' if args.use_original_dir else '--no-export'})"
        )

    log_step(5, "Harbor upload status")
    upload_target = exported_path or (job_dir
                                      if args.use_original_dir else None)
    if args.upload and upload_target:
        proc = subprocess.run(_harbor_cmd("upload", str(upload_target)),
                              capture_output=True,
                              text=True)
        if proc.returncode == 0:
            log_info("Successfully uploaded regraded job to Harbor platform!")
            for line in (proc.stdout or "").strip().splitlines():
                log_info(line)
        else:
            log_info(f"Harbor upload failed with exit code {proc.returncode}:")
            for line in (proc.stderr or proc.stdout
                         or "Unknown upload error").strip().splitlines():
                log_info(f"  {line}")
    elif exported_path:
        log_info("No upload performed (local export only).")
        log_info(
            f"To upload to Harbor platform, run: harbor upload {exported_path}"
        )
    else:
        log_info("No upload performed.")

    for res in all_results:
        reward_dict = res.get("reward", {})
        speedup_val, reward_val = reward_dict.get("speedup"), reward_dict.get(
            "reward")
        print("=" * 80)
        print(f"REGRADED TRIAL: {res['trial_name']}")
        print(f"Path          : {res['trial_path']}")
        print(f"Model         : {res['model_name']}")
        print(f"Verifiers     : {', '.join(res['active_verifiers'])}")
        if res.get("trajectory_path"):
            print(f"Trajectory    : {res['trajectory_path']}")
        print(
            f"Speedup       : {f'{speedup_val:.4f}x' if isinstance(speedup_val, (int, float)) else 'none'}"
        )
        print(
            f"Overall Reward: {f'{reward_val:.4f}' if isinstance(reward_val, (int, float)) else 'none'}"
        )
        print("=" * 80)
        if args.verbose:
            if res.get("harbor_log"):
                print("--- Standard Harbor Log (test-stdout.txt) ---")
                print(res["harbor_log"].strip())
            print("--- RewardKit reward.json ---")
            print(json.dumps(reward_dict, indent=2))
            print("\n--- RewardKit reward-details.json ---")
            print(json.dumps(res["reward_details"], indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
