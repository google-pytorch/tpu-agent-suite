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
"""Verify that the local model checkpoint directory was staged to GCS.

Evidence is gathered from:
1. The checkpoint.json artifact produced by stage_checkpoint.py.
2. The commands executed in trajectory/stream logs running stage_checkpoint.py.
3. The deployments.json / launch_vllm.sh referencing the staged gs:// checkpoint.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

from rewardkit import criterion

_CMD_TOOLS = {
    "run_shell_command",
    "run_command",
    "bash",
    "execute_command",
    "sh",
    "terminal",
    "exec",
}

_STAGE_CMD_RE = re.compile(
    r"stage_checkpoint\.py.*?local-qwen3-0\.6b|stage_checkpoint\.py.*?--bucket\s+gs://",
    re.DOTALL)


def _logs_dir() -> Path:
    """The run's logs as the harness lays them out: $LOGS_DIR, else /logs."""
    return Path(os.environ.get("LOGS_DIR") or "/logs")


def _trajectory_path() -> Path:
    """The agent's trajectory, where the harness writes it."""
    return _logs_dir() / "agent" / "trajectory.json"


def _load_trajectory(path: str | Path) -> dict[str, Any] | None:
    """Load an ATIF trajectory JSON file. Returns None on error."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _collect_tool_calls(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect all tool calls across all steps."""
    calls: list[dict[str, Any]] = []
    for step in data.get("steps", []):
        for tc in step.get("tool_calls") or []:
            calls.append(tc)
    return calls


def _search_dirs(workspace: Path) -> list[Path]:
    return [workspace, _logs_dir() / "artifacts"]


def _find_files(workspace: Path, name: str) -> list[Path]:
    found: dict[str, Path] = {}
    for d in _search_dirs(workspace):
        try:
            if not d.is_dir():
                continue
            for p in d.glob(f"**/{name}"):
                if p.is_file():
                    found[str(p.resolve())] = p
        except Exception:
            pass
    return list(found.values())


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def _check_checkpoint_artifact(workspace: Path) -> bool:
    for path in _find_files(workspace, "checkpoint.json"):
        data = _load_json(path)
        if isinstance(data, dict):
            ckpt = str(data.get("checkpoint", ""))
            repo = str(data.get("repo", ""))
            if ckpt.startswith("gs://") and "local-qwen3-0.6b" in (ckpt +
                                                                   repo):
                return True
    return False


def _check_deployments_for_staged_gcs(workspace: Path) -> bool:
    for path in _find_files(workspace, "deployments.json"):
        data = _load_json(path)
        if isinstance(data, dict):
            for dep in data.get("deployments") or []:
                if isinstance(dep, dict):
                    model = str(dep.get("model", ""))
                    if model.startswith(
                            "gs://") and "checkpoints/local" in model:
                        return True
    return False


def _check_commands() -> bool:
    calls: list[dict[str, Any]] = []
    data = _load_trajectory(_trajectory_path())
    if data:
        calls.extend(_collect_tool_calls(data))

    log_dir = _logs_dir() / "agent"
    if log_dir.is_dir():
        for p in log_dir.glob("**/*"):
            if not (p.is_file() and p.suffix in (".log", ".json", ".jsonl")):
                continue
            try:
                for line in p.read_text(encoding="utf-8",
                                        errors="ignore").splitlines():
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    su = obj.get("step_update") or {}
                    for key in ("tool_info", "tool_call"):
                        tc = su.get(key) if isinstance(su, dict) else None
                        if isinstance(tc, dict):
                            calls.append(tc)
                    for tc in obj.get("tool_calls") or []:
                        if isinstance(tc, dict):
                            calls.append(tc)
            except Exception:
                continue

    for tc in calls:
        name = str(tc.get("function_name") or tc.get("name") or "").lower()
        args = tc.get("arguments") or tc.get("args") or {}
        if name in _CMD_TOOLS and isinstance(args, dict):
            cmd = str(
                args.get("command") or args.get("CommandLine")
                or args.get("cmd") or args.get("command_line") or "")
            if _STAGE_CMD_RE.search(cmd):
                return True
    return False


@criterion(description="Verify local checkpoint directory was staged to GCS")
def check_checkpoint_staged(workspace: Path) -> bool:
    if _check_checkpoint_artifact(workspace):
        return True
    if _check_deployments_for_staged_gcs(workspace):
        return True
    return _check_commands()
