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

import ast
import glob
import json
import os
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
_WRITE_TOOLS = {
    "write_file",
    "write",
    "create_file",
    "edit_file",
    "edit",
    "multiedit",
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


def _is_valid_benchmark_script(content: str, require_ast: bool = True) -> bool:
    if not content.strip(
    ) or "Bespoke loader not implemented" in content or "torch_xla" in content:
        return False
    if require_ast:
        try:
            ast.parse(content)
        except SyntaxError:
            return False
    return ("AutoModelForCausalLM" in content and "AutoTokenizer" in content
            and "run_benchmark_pass" in content
            and ("get_device" in content or "torch_tpu" in content
                 or 'device("tpu")' in content or "device('tpu')" in content))


@criterion(description=(
    "Verify torch-benchmark-coder Causal LM synthesis was executed"))
def check_command_executed(workspace: Path) -> bool:
    search_dirs = [workspace, _logs_dir() / "artifacts"]
    for s_dir in search_dirs:
        try:
            if not s_dir.exists():
                continue
            for m in s_dir.glob("**/benchmark_tpu.py"):
                if m.is_file() and "/skills/" not in str(
                        m) and "/packages/" not in str(m):
                    if _is_valid_benchmark_script(
                            m.read_text(encoding="utf-8", errors="ignore")):
                        return True
        except Exception:
            pass

    for tc in _collect_all_calls():
        tool_name = (tc.get("function_name") or tc.get("name") or "").lower()
        args = (tc.get("arguments") or tc.get("args") or tc.get("input") or {})
        if not isinstance(args, dict):
            continue
        if tool_name in _WRITE_TOOLS:
            path = str(
                args.get("Path") or args.get("path") or args.get("file_path")
                or args.get("TargetFile") or args.get("AbsolutePath") or "")
            content = str(
                args.get("Content") or args.get("content")
                or args.get("file_text") or args.get("CodeContent")
                or args.get("Replacement") or args.get("new_string") or "")
            is_edit = tool_name in {"edit_file", "edit", "multiedit"}
            if path.endswith(
                    "benchmark_tpu.py") and _is_valid_benchmark_script(
                        content, require_ast=not is_edit):
                return True
        if tool_name in _CMD_TOOLS:
            cmd = str(
                args.get("command") or args.get("CommandLine")
                or args.get("cmd") or args.get("command_line") or "")
            if "benchmark_tpu.py" in cmd and _is_valid_benchmark_script(
                    cmd, require_ast=False):
                return True

    return False
