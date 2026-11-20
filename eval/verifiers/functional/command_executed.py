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
"""Verify a workflow's commands or artifacts were produced.

Configured through `verifier_params.json`:
  - `artifact_globs`: file globs; any match in the workspace or artifact
    directories passes.
  - `command_regex`: regex matched against shell commands and delegation
    prompts.
  - `target_names`: names that pass when a subagent delegation mentions one.
"""

import itertools
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
_SUBAGENT_TOOLS = {
    "invoke_subagent",
    "subagent",
    "call_subagent",
    "delegate",
}


def _logs_dir() -> Path:
    """The run's logs as the harness lays them out: $LOGS_DIR, else /logs."""
    return Path(os.environ.get("LOGS_DIR") or "/logs")


def _tests_dir() -> Path:
    """The task's tests directory: $TESTS_DIR, else /tests."""
    return Path(os.environ.get("TESTS_DIR") or "/tests")


def _trajectory_path() -> Path:
    """The agent's trajectory, where the harness writes it."""
    return _logs_dir() / "agent" / "trajectory.json"


def _load_verifier_params() -> dict[str, Any]:
    """This verifier's table in the task's verifier_params.json, else {}.

    The table is named after the folder this verifier is staged in.
    """
    path = _tests_dir() / "verifier_params.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    params = data.get(Path(__file__).parent.name)
    return params if isinstance(params, dict) else {}


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
    agent_logs = _logs_dir() / "agent"
    stream_logs = [
        agent_logs / "stream.log",
        agent_logs / "stream.jsonl",
        *agent_logs.glob("*.log"),
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
                        "name":
                        ti.get("name") or ti.get("function_name"),
                        "args":
                        ti.get("args") or ti.get("arguments") or {},
                    })
                tc = su.get("tool_call")
                if isinstance(tc, dict):
                    calls.append({
                        "name":
                        tc.get("name") or tc.get("function_name"),
                        "args":
                        tc.get("args") or tc.get("arguments") or {},
                    })
        except Exception:
            pass

    # 3. Agent output text files for embedded JSON / tool invocations
    for p in agent_logs.glob("**/*"):
        if p.is_file() and p.suffix in (".log", ".json", ".jsonl", ".txt"):
            try:
                for line in p.read_text(encoding="utf-8",
                                        errors="ignore").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        if "tool_name" in obj or "function_name" in obj or "name" in obj:
                            calls.append({
                                "name":
                                obj.get("tool_name")
                                or obj.get("function_name") or obj.get("name"),
                                "args":
                                obj.get("arguments") or obj.get("args")
                                or obj.get("parameters") or {},
                            })
                        call_field = obj.get("tool_call") or obj.get("call")
                        if isinstance(call_field, dict):
                            calls.append({
                                "name":
                                call_field.get("name")
                                or call_field.get("function_name"),
                                "args":
                                call_field.get("args")
                                or call_field.get("arguments") or {},
                            })
            except Exception:
                pass

    return calls


def _strings(value: Any) -> list[str]:
    """The non-empty strings of a `verifier_params.json` list field."""
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str) and v]


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _has_artifact(workspace: Path, patterns: list[str]) -> bool:
    search_dirs = [
        workspace,
        workspace / "artifacts",
        _logs_dir() / "artifacts",
    ]
    for d in search_dirs:
        if not d.is_dir():
            continue
        for pat in patterns:
            for match in itertools.chain(d.glob(pat), d.glob(f"**/{pat}")):
                if _is_nonempty_file(match):
                    return True
    return False


@criterion(
    description="Verify workflow execution commands or artifacts were produced"
)
def check_command_executed(workspace: Path) -> bool:
    params = _load_verifier_params()
    artifact_globs = _strings(params.get("artifact_globs"))
    target_names = {n.lower() for n in _strings(params.get("target_names"))}
    command_regex = params.get("command_regex")
    if not isinstance(command_regex, str):
        command_regex = ""
    if not (artifact_globs or target_names or command_regex):
        print("command_executed: [FAIL] verifier_params.json gives this "
              "verifier no artifact_globs, target_names or command_regex")
        return False

    if _has_artifact(workspace, artifact_globs):
        return True

    regex = re.compile(command_regex) if command_regex else None
    calls = _collect_all_calls()
    for tc in calls:
        tool_name = (tc.get("function_name") or tc.get("name") or "").lower()
        args = tc.get("arguments") or tc.get("args") or {}
        if tool_name in _CMD_TOOLS:
            cmd = str(
                args.get("command") or args.get("CommandLine")
                or args.get("cmd") or args.get("command_line") or "")
            if regex and regex.search(cmd):
                return True
        if tool_name in _SUBAGENT_TOOLS:
            subagents = args.get("Subagents") or args.get("subagents") or []
            if isinstance(subagents, list):
                for sub in subagents:
                    if isinstance(sub, dict):
                        t_name = str(
                            sub.get("TypeName") or sub.get("type_name")
                            or sub.get("Role") or sub.get("role")
                            or "").lower()
                        prompt = str(
                            sub.get("Prompt") or sub.get("prompt")
                            or "").lower()
                        if any(n in t_name for n in target_names) or any(
                                n in prompt for n in target_names):
                            return True
                        if regex and (regex.search(prompt)
                                      or regex.search(t_name)):
                            return True
            args_str = str(args).lower()
            if any(n in args_str for n in target_names):
                return True

    return False
