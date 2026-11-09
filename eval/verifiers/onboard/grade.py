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
#     "harbor-rewardkit>=0.2.1",
#     "google-auth",
# ]
# ///
"""RewardKit Regrade Engine for Onboarding-Only TPU Serving Benchmarks.

Adapted from PR #694 (`eval/verifiers/onboard_hillclimb/grade.py`) for
onboarding-only tasks (`eval/datasets/nightly-onboard-optimize/onboarding-*`).
Accepts a Harbor Job ID, URL, or local path, stages requested verifiers
(`onboarding_success`, `efficiency`, `cheating`, `regime_sweep`, `execution_report`),
preserves any legacy regex verifier output in `verifier_legacy_regex/`, invokes
RewardKit, and produces finalized `reward.json` and `reward-details.json` payloads.
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
from pathlib import Path
from typing import Any, Dict, List, Optional

_VERIFIERS_DIR = Path(__file__).resolve().parent / "tests"

UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE)


def _find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for p in [current] + list(current.parents):
        if (p / ".git").exists() or (p / "eval").exists():
            return p
    return Path.cwd()


_REPO_ROOT = _find_repo_root()
DEFAULT_DOWNLOAD_DIR = Path("/tmp/harbor_jobs")
DEFAULT_EXPORT_DIR = _REPO_ROOT / "jobs"

DEFAULT_VERIFIERS = [
    "onboarding_success",
    "efficiency",
    "cheating",
    "regime_sweep",
    "execution_report",
    "invoked_subagent",
]
AVAILABLE_VERIFIERS = [
    "onboarding_success",
    "efficiency",
    "cheating",
    "regime_sweep",
    "execution_report",
    "invoked_subagent",
]


def log_step(step_num: int, title: str) -> None:
    sys.stderr.write(f"\n[{step_num}] {title}\n")
    sys.stderr.flush()


def log_info(message: str) -> None:
    sys.stderr.write(f"  -> {message}\n")
    sys.stderr.flush()


def log_block(text: str, indent: str = "  ") -> None:
    for line in text.splitlines():
        sys.stderr.write(f"{indent}{line}\n")
    sys.stderr.flush()


def find_trial_directories(job_dir: Path,
                           target_trial_id: Optional[str] = None
                           ) -> List[Path]:
    """Finds trial directories within a job directory."""
    trials: List[Path] = []
    if ((job_dir / "agent").exists() or (job_dir / "artifacts").exists()
            or (job_dir / "verifier").exists()
            or (job_dir / "trajectory.json").exists()):
        trials.append(job_dir)
        return trials

    for child in sorted(job_dir.iterdir()):
        if not child.is_dir():
            continue
        if ((child / "agent").exists() or (child / "artifacts").exists()
                or (child / "verifier").exists()
                or (child / "trajectory.json").exists()):
            if target_trial_id and target_trial_id.lower(
            ) not in child.name.lower():
                continue
            trials.append(child)
    return trials


def find_trajectory_file(target_dir: Path) -> Optional[Path]:
    """Locates trajectory.json in standard and wonky directory layouts."""
    candidates = [
        target_dir / "agent" / "trajectory.json",
        target_dir / "logs" / "agent" / "trajectory.json",
        target_dir / "trajectory.json",
        target_dir / "logs" / "trajectory.json",
    ]
    for c in candidates:
        if c.is_file() and c.stat().st_size > 10:
            return c.resolve()

    for path in target_dir.glob("**/trajectory.json"):
        if path.is_file() and ".git" not in path.parts and path.stat(
        ).st_size > 10:
            return path.resolve()

    workspace_session_dir = (target_dir / "agent" / "sessions" / "projects" /
                             "-workspace")
    if workspace_session_dir.is_dir() and any(
            workspace_session_dir.glob("*.jsonl")):
        agent_dir = target_dir / "agent"
        out_traj = agent_dir / "trajectory.json"
        venv_python = _REPO_ROOT / "eval" / ".venv" / "bin" / "python"
        python_bin = str(
            venv_python) if venv_python.is_file() else sys.executable
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
                [python_bin, "-c", recover_script,
                 str(agent_dir)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode == 0 and out_traj.is_file():
                log_info(
                    f"Recovered missing trajectory.json from {workspace_session_dir}"
                )
                return out_traj.resolve()
        except Exception:
            pass

    return None


def detect_model_name(target_dir: Path) -> str:
    """Infers model name from directory name, config files, artifacts, or trajectory."""
    name_str = target_dir.name.lower()
    if target_dir.parent:
        name_str += " " + target_dir.parent.name.lower()

    if "gemma-4-31b" in name_str or "gemma4-31b" in name_str or "gemma-4" in name_str:
        return "google/gemma-4-31B-it"
    if "gemma-3-27b" in name_str or "gemma3-27b" in name_str:
        return "google/gemma-3-27b-it"
    if "qwen3-32b" in name_str:
        return "Qwen/Qwen3-32B"
    if "gpt-oss-20b" in name_str or "gpt_oss_20b" in name_str or "gpt-oss" in name_str:
        return "openai/gpt-oss-20b"
    if "mistral-small-24b" in name_str or "mistral" in name_str:
        return "mistralai/Mistral-Small-24B-Instruct-2501"

    text_to_search = name_str
    for d in (target_dir, target_dir.parent):
        if not d or not d.exists():
            continue
        for config_file in ("config.json", "task.toml", "result.json"):
            p = d / config_file
            if p.exists():
                try:
                    text_to_search += " " + p.read_text(
                        encoding="utf-8").lower()
                except Exception:
                    pass

    if "gemma-3-27b" in text_to_search or "gemma3-27b" in text_to_search:
        return "google/gemma-3-27b-it"
    if "qwen3-32b" in text_to_search:
        return "Qwen/Qwen3-32B"
    if "gpt-oss" in text_to_search or "gpt_oss" in text_to_search:
        return "openai/gpt-oss-20b"
    if "mistral" in text_to_search:
        return "mistralai/Mistral-Small-24B-Instruct-2501"
    if "gemma" in text_to_search:
        return "google/gemma-4-31B-it"
    if "qwen" in text_to_search:
        return "Qwen/Qwen3.8-27B"
    return "the target model"


def _is_agent_suite_trial(workspace_dir: Optional[Path]) -> bool:
    if workspace_dir is None:
        return False
    trial_dir = workspace_dir.parent if workspace_dir.name == "workspace" else workspace_dir
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


def stage_verifier_tests(
    stage_dir: Path,
    verifiers: List[str],
    trajectory_path: Optional[Path],
    model_name: str,
    judge_model: Optional[str] = None,
    workspace_dir: Optional[Path] = None,
    logs_dir: Optional[Path] = None,
) -> Path:
    """Stages RewardKit test folder with requested verifiers and paths dynamically mapped."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    resolved_logs = (logs_dir or (stage_dir.parent / "logs")).resolve()
    logs_str = str(resolved_logs)

    weights: Dict[str, float] = {
        "onboarding_success": 0.6,
        "efficiency": 0.4,
        "cheating": -1.0,
        "regime_sweep": 0.0,
        "execution_report": 0.0,
        "invoked_subagent": 0.0,
    }

    reward_toml_lines = [
        "[[reward]]",
        'name = "reward"',
        'aggregation = "weighted-sum"',
        "",
        "[reward.weights]",
    ]

    for v in verifiers:
        src = _VERIFIERS_DIR / v
        if not src.exists():
            continue

        dst = stage_dir / v
        shutil.copytree(src, dst, symlinks=False)

        judge_file = dst / "judge.toml"
        if judge_file.exists():
            content = judge_file.read_text(encoding="utf-8")
            if trajectory_path:
                content = content.replace(
                    'atif-trajectory = "/logs/agent/trajectory.json"',
                    f'atif-trajectory = "{trajectory_path.resolve()}"',
                )
            content = content.replace("{MODEL_NAME}", model_name)
            if judge_model:
                content = re.sub(r'judge\s*=\s*"[^"]+"',
                                 f'judge = "{judge_model}"', content)
            judge_file.write_text(content, encoding="utf-8")

        inv_check_file = dst / "check.py"
        if v == "invoked_subagent" and inv_check_file.exists():
            inv_content = inv_check_file.read_text(encoding="utf-8").replace(
                '"/logs/', f'"{logs_str}/')
            if _is_agent_suite_trial(workspace_dir):
                manifest_path = dst / "tpu-agent-suite-build.json"
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(
                    json.dumps(
                        {
                            "contracts": {
                                name: {}
                                for name in _collect_suite_agent_names()
                            }
                        },
                        indent=2),
                    encoding="utf-8",
                )
            else:
                manifest_path = dst / "no-suite" / "tpu-agent-suite-build.json"
            inv_content = inv_content.replace(
                '"/root/.tpu-agent-suite/packages/*/tpu-agent-suite-build.json"',
                f'"{manifest_path.resolve()}"',
            )
            inv_check_file.write_text(inv_content, encoding="utf-8")

        reward_toml_lines.append(f"{v} = {weights.get(v, 0.0)}")

    test_sh_src = _VERIFIERS_DIR / "test.sh"
    if test_sh_src.exists():
        shutil.copy2(test_sh_src, stage_dir / "test.sh")
        try:
            (stage_dir / "test.sh").chmod(0o755)
        except Exception:
            pass

    reward_toml_path = stage_dir / "reward.toml"
    reward_toml_path.write_text("\n".join(reward_toml_lines) + "\n",
                                encoding="utf-8")
    return stage_dir


def compute_reward_diff(original: Dict[str, Any],
                        new: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    diff: Dict[str, Dict[str, Any]] = {}
    orig_copy = dict(original)
    new_copy = dict(new)

    all_keys = list(
        dict.fromkeys([
            "reward",
            "legacy_regex_reward",
            "onboarding_success",
            "efficiency",
            "trajectory_score",
            "valid_tool_calls",
            "framework_directed_penalty",
            "cheating",
            "regime_coverage",
            "peak_tps_1k_1k",
            "peak_tps_1k_8k",
            "peak_tps_8k_1k",
            "vllm_server_on_tpu",
            "completion_over_http",
            "baseline_benchmarks_reported",
            "baseline_reproduce_commands",
            "invoked_subagent",
        ] + list(orig_copy.keys()) + list(new_copy.keys())))
    for k in all_keys:
        if k.endswith("_raw") or k.endswith("_gated"):
            continue
        old_v = orig_copy.get(k)
        new_v = new_copy.get(k)
        if old_v is None and new_v is None:
            continue
        delta = None
        if isinstance(old_v, (int, float)) and isinstance(new_v, (int, float)):
            delta = round(float(new_v) - float(old_v), 4)
        diff[k] = {
            "original": old_v,
            "regraded": new_v,
            "delta": delta,
        }
    return diff


def format_reward_diff(diff: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "Reward Diff (original -> regraded):",
        f"  {'Metric':<30} {'Original':<12} {'Regraded':<12} {'Diff'}",
        "  " + "─" * 64,
    ]
    for k, item in diff.items():
        old_v = item["original"]
        new_v = item["regraded"]
        delta = item["delta"]

        old_str = f"{old_v:.4f}" if isinstance(
            old_v, float) else ("none" if old_v is None else str(old_v))
        new_str = f"{new_v:.4f}" if isinstance(
            new_v, float) else ("none" if new_v is None else str(new_v))
        if delta is not None:
            diff_str = f"+{delta:.4f}" if delta > 0 else (
                f"{delta:.4f}" if delta < 0 else "=")
        elif old_v is None and new_v is not None:
            diff_str = "new"
        else:
            diff_str = ""
        lines.append(f"  {k:<30} {old_str:<12} {new_str:<12} {diff_str}")
    return "\n".join(lines)


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
    if not model_name or model_name.lower() == "auto":
        resolved_model = detect_model_name(trial_dir)
    else:
        resolved_model = model_name
    trajectory_path = find_trajectory_file(trial_dir)

    workspace_dir = trial_dir
    verifier_dir = trial_dir / "verifier"
    backup_dir = trial_dir / "verifier_legacy_regex"

    trial_title = f"'{trial_dir.name}'" if total_trials == 1 else f"trial {trial_index}/{total_trials} ('{trial_dir.name}')"
    log_step(
        2,
        f"Staging onboarding verifier suite for {trial_title} (model={resolved_model})"
    )

    original_scores: Dict[str, Any] = {}
    legacy_regex_reward: Optional[float] = None

    # Preserve legacy regex verifier directory before first regrade
    if verifier_dir.exists() and not backup_dir.exists():
        try:
            shutil.copytree(verifier_dir, backup_dir)
            log_info(f"Backed up legacy verifier directory to {backup_dir}")
        except Exception as e:
            log_info(f"Warning: could not back up verifier dir: {e}")

    if (backup_dir / "reward.txt").exists():
        try:
            legacy_regex_reward = float(
                (backup_dir /
                 "reward.txt").read_text(encoding="utf-8").strip())
        except Exception:
            pass

    existing_reward = verifier_dir / "reward.json"
    if existing_reward.exists():
        try:
            original_scores = json.loads(
                existing_reward.read_text(encoding="utf-8"))
        except Exception:
            pass
    if not original_scores and (trial_dir / "result.json").exists():
        try:
            res_json = json.loads(
                (trial_dir / "result.json").read_text(encoding="utf-8"))
            if isinstance(res_json.get("verifier_result"),
                          dict) and "rewards" in res_json["verifier_result"]:
                original_scores = res_json["verifier_result"]["rewards"]
        except Exception:
            pass
    if legacy_regex_reward is None and "reward" in original_scores and "onboarding_success" not in original_scores:
        legacy_regex_reward = float(original_scores["reward"])

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        staged_tests = temp_dir / "tests"
        staged_logs = temp_dir / "logs"
        (staged_logs / "agent").mkdir(parents=True, exist_ok=True)
        (staged_logs / "verifier").mkdir(parents=True, exist_ok=True)

        if trajectory_path and trajectory_path.exists():
            shutil.copy2(trajectory_path,
                         staged_logs / "agent" / "trajectory.json")
            shutil.copy2(trajectory_path, staged_logs / "trajectory.json")

        # Symlink trial artifacts into staged_logs/artifacts so /logs/artifacts works seamlessly
        trial_artifacts = trial_dir / "artifacts" / "logs" / "artifacts"
        if not trial_artifacts.is_dir():
            trial_artifacts = trial_dir / "artifacts"
        if trial_artifacts.is_dir():
            try:
                (staged_logs / "artifacts").symlink_to(
                    trial_artifacts.resolve(), target_is_directory=True)
            except Exception:
                pass

        effective_verifiers = list(active_verifiers)
        if not trajectory_path:
            traj_verifiers = {
                "onboarding_success", "cheating", "execution_report",
                "efficiency", "invoked_subagent"
            }
            effective_verifiers = [
                v for v in effective_verifiers if v not in traj_verifiers
            ]

        stage_verifier_tests(
            stage_dir=staged_tests,
            verifiers=effective_verifiers,
            trajectory_path=trajectory_path,
            model_name=resolved_model,
            judge_model=judge_model,
            workspace_dir=workspace_dir,
            logs_dir=staged_logs,
        )

        test_script = staged_tests / "test.sh"
        env = os.environ.copy()
        if trajectory_path:
            env["ATIF_TRAJECTORY"] = str(trajectory_path.resolve())
            env["TRAJECTORY_PATH"] = str(trajectory_path.resolve())
        env["ONBOARDING_MODEL_NAME"] = resolved_model
        env.setdefault("VERTEXAI_PROJECT", "tpu-nexus-767207")
        env.setdefault("VERTEXAI_LOCATION", "global")
        env.setdefault("CLOUD_ML_REGION", "global")
        env.setdefault("DEFAULT_MAX_RETRIES", "10")
        env.setdefault("INITIAL_RETRY_DELAY", "2.0")
        env.setdefault("MAX_RETRY_DELAY", "60.0")

        test_stdout = ""
        log_step(
            3, f"Running verifiers via test.sh + RewardKit on {trial_title}")
        if effective_verifiers and test_script.exists():
            cmd = [
                "bash",
                str(test_script.resolve()),
                str(staged_tests.resolve()),
                str(workspace_dir.resolve()),
                str(staged_logs.resolve()),
            ]
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            test_stdout = proc.stdout or ""
            if proc.stderr:
                test_stdout += ("\n" if test_stdout
                                and not test_stdout.endswith("\n") else
                                "") + proc.stderr
            if proc.returncode != 0:
                log_info(
                    f"Verifier script exited with non-zero code {proc.returncode}"
                )

        gen_reward = staged_logs / "verifier" / "reward.json"
        gen_details = staged_logs / "verifier" / "reward-details.json"

        is_partial_regrade = set(effective_verifiers) < set(DEFAULT_VERIFIERS)

        final_scores: Dict[str, Any] = dict(
            original_scores) if is_partial_regrade else {}
        if gen_reward.exists():
            try:
                new_scores = json.loads(gen_reward.read_text(encoding="utf-8"))
                if is_partial_regrade and "onboarding_success" not in effective_verifiers and "efficiency" not in effective_verifiers:
                    new_scores.pop("reward", None)
                final_scores.update(new_scores)
            except Exception:
                pass

        if legacy_regex_reward is not None:
            final_scores["legacy_regex_reward"] = legacy_regex_reward

        final_details: Dict[str, Any] = {}
        existing_details_file = verifier_dir / "reward-details.json"
        if is_partial_regrade and existing_details_file.exists():
            try:
                final_details = json.loads(
                    existing_details_file.read_text(encoding="utf-8"))
            except Exception:
                final_details = {}
        if gen_details.exists():
            try:
                new_details = json.loads(
                    gen_details.read_text(encoding="utf-8"))
                final_details.update(new_details)
            except Exception:
                pass

        reward_diff = compute_reward_diff(original_scores, final_scores)
        diff_table = format_reward_diff(reward_diff)
        log_block(diff_table)

        if apply_changes and final_scores:
            verifier_dir.mkdir(parents=True, exist_ok=True)
            (verifier_dir / "reward.json").write_text(json.dumps(final_scores,
                                                                 indent=2),
                                                      encoding="utf-8")
            (verifier_dir / "reward.txt").write_text(
                f"{final_scores.get('reward', 0.0)}\n", encoding="utf-8")
            (verifier_dir / "reward-details.json").write_text(json.dumps(
                final_details, indent=2),
                                                              encoding="utf-8")
            (verifier_dir / "test-stdout.txt").write_text(test_stdout,
                                                          encoding="utf-8")
            (verifier_dir / "regrade.log").write_text(test_stdout,
                                                      encoding="utf-8")

            # Also update trial result.json and parent job result.json
            trial_res_file = trial_dir / "result.json"
            if trial_res_file.exists():
                backup_res = trial_dir / "result.legacy_regex.json"
                if not backup_res.exists():
                    shutil.copy2(trial_res_file, backup_res)
                try:
                    t_res = json.loads(
                        trial_res_file.read_text(encoding="utf-8"))
                    t_res["verifier_result"] = {"rewards": final_scores}
                    trial_res_file.write_text(json.dumps(t_res, indent=2),
                                              encoding="utf-8")
                except Exception as e:
                    log_info(
                        f"Warning: could not update {trial_res_file}: {e}")

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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RewardKit Onboarding-Only Regrade Tool for Harbor trials."
    )
    parser.add_argument(
        "target",
        type=str,
        help="Local path to job/trial directory.",
    )
    parser.add_argument(
        "--use-original-dir",
        "--in-place",
        "--apply",
        dest="use_original_dir",
        action="store_true",
        help=
        "Update verifier artifacts directly inside the original trial directory.",
    )
    parser.add_argument(
        "--onboarding-model-name",
        "--model-name",
        dest="onboarding_model_name",
        type=str,
        default="auto",
        help=
        "Target model name for onboarding_success verifier prompt (default: 'auto').",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Optional override for the LLM judge model in judge.toml.",
    )
    parser.add_argument(
        "--verifiers",
        type=str,
        default=",".join(DEFAULT_VERIFIERS),
        help=
        f"Comma-separated list of verifiers to run (default: '{','.join(DEFAULT_VERIFIERS)}').",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Display full verbose output.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    target_path = Path(args.target.strip())
    if not target_path.exists():
        sys.stderr.write(
            f"Error: Target path '{target_path}' does not exist.\n")
        return 1

    log_step(1, f"Using job/trial at '{target_path}' for regrade")
    trials = find_trial_directories(target_path)
    if not trials:
        sys.stderr.write(
            f"Error: No trial directories found inside {target_path}.\n")
        return 1

    log_info(
        f"Discovered {len(trials)} trial(s): {', '.join(t.name for t in trials)}"
    )
    selected_verifiers = [
        v.strip() for v in args.verifiers.split(",") if v.strip()
    ]
    all_results: List[Dict[str, Any]] = []

    for idx, trial in enumerate(trials, start=1):
        res = regrade_trial(
            trial_dir=trial,
            verifiers=selected_verifiers,
            apply_changes=args.use_original_dir,
            model_name=args.onboarding_model_name,
            judge_model=args.judge_model,
            trial_index=idx,
            total_trials=len(trials),
        )
        all_results.append(res)

    for res in all_results:
        print("=" * 80)
        print(f"REGRADED TRIAL: {res['trial_name']}")
        print(f"Path          : {res['trial_path']}")
        print(f"Model         : {res['model_name']}")
        print(f"Reward JSON   : {json.dumps(res['reward'])}")
        print("=" * 80)
        if args.verbose and res.get("harbor_log"):
            print(res["harbor_log"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
