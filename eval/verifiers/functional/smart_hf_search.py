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

import glob
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
    "Bash",
    "exec",
}


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


def _collect_all_calls() -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    # 1. The ATIF trajectory
    data = _load_trajectory(_trajectory_path())
    if data:
        calls.extend(_collect_tool_calls(data))

    # 2. Antigravity stream log (stream-json events)
    stream_logs = [
        _logs_dir() / "agent" / "stream.log",
        _logs_dir() / "agent" / "stream.jsonl",
        *(_logs_dir() / "agent").glob("*.log"),
    ]
    for sl in stream_logs:
        if not sl.exists():
            continue
        try:
            for line in sl.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                su = evt.get("step_update", {})
                ti = su.get("tool_info")
                if isinstance(ti, dict):
                    calls.append({
                        "function_name":
                        ti.get("name") or su.get("tool_name"),
                        "arguments":
                        ti.get("parameters") or {},
                    })
                sub_info = su.get("subagent_info", {})
                for sub in sub_info.get("subagents", []):
                    calls.append({
                        "function_name": "invoke_subagent",
                        "arguments": {
                            "Subagents": [sub],
                            "TypeName": sub.get("type_name"),
                            "Role": sub.get("role"),
                            "Prompt": sub.get("initial_prompt"),
                        },
                    })
        except Exception:
            pass

    # 3. Transcripts on disk (parent and subagents)
    patterns = [
        "/root/.gemini/antigravity-cli/brain/*/.system_generated/logs/transcript*.jsonl",
        str(Path.home() /
            ".gemini/antigravity-cli/brain/*/.system_generated/logs/transcript*.jsonl"
            ),
        f"{glob.escape(str(_logs_dir()))}/agent/*.jsonl",
        f"{glob.escape(str(_logs_dir()))}/*.jsonl",
    ]
    for pat in patterns:
        for f in glob.glob(pat):
            try:
                for line in Path(f).read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        step = json.loads(line)
                    except Exception:
                        continue
                    for tc in step.get("tool_calls", []):
                        calls.append(tc)
            except Exception:
                pass

    return calls


def _has_artifact(workspace: Path, patterns: list[str]) -> bool:
    search_dirs = [workspace, _logs_dir() / "artifacts"]
    for s_dir in search_dirs:
        try:
            if not s_dir.exists():
                continue
        except OSError:
            continue
        for pat in patterns:
            try:
                matches = list(s_dir.glob(pat)) + list(
                    s_dir.glob(f"*/artifacts/{pat}"))
                for m in matches:
                    if m.is_file() and m.stat().st_size > 0:
                        return True
            except Exception:
                pass
    return False


@criterion(description="Verify smart_hf_search command or script was executed")
def check_command_executed(workspace: Path) -> bool:
    regex = re.compile(
        r"smart_hf_search\.py\b.*\s--(?:max_params|k|sort)\b|smart[-_]hf[-_]search"
    )

    if _has_artifact(
            workspace,
        [
            "*smart_hf_search*", "*hf_search*", "*search_models*",
            "*models*.json"
        ],
    ):
        return True

    calls = _collect_all_calls()
    for tc in calls:
        tool_name = (tc.get("function_name") or tc.get("name") or "").lower()
        args = tc.get("arguments") or tc.get("args") or tc.get("input") or {}
        if tool_name in _CMD_TOOLS:
            cmd = str(
                args.get("command") or args.get("CommandLine")
                or args.get("cmd") or args.get("command_line") or "")
            if regex.search(cmd):
                return True

    return False
