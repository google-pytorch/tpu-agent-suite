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
import re
from pathlib import Path
from typing import Any

from rewardkit import criterion
from rewardkit.criteria._trajectory import collect_tool_calls, load_trajectory

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
_COMPUTE_RUN = re.compile(
    r"\bcompute(?:\s+--?[\w-]+(?:(?:=|\s+)(?!-)\S+)?)*\s+run\b")
_INSTALL_VLLM = re.compile(
    r"\b(pip3?|uv\s+pip)\s+install\b[^\n;&|]*\bvllm\b",
    re.IGNORECASE,
)


def _collect_all_calls() -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    for traj in ["/logs/agent/trajectory.json", "/logs/trajectory.json"]:
        data = load_trajectory(traj)
        if data:
            calls.extend(collect_tool_calls(data))

    for sl in [
            Path("/logs/agent/stream.log"),
            Path("/logs/agent/stream.jsonl"),
            *Path("/logs/agent").glob("*.log"),
    ]:
        if not sl.exists():
            continue
        try:
            for line in sl.read_text(encoding="utf-8",
                                     errors="ignore").splitlines():
                if not line.strip():
                    continue
                evt = json.loads(line)
                su = evt.get("step_update", {})
                ti = su.get("tool_info")
                if isinstance(ti, dict):
                    calls.append({
                        "function_name":
                        ti.get("name") or su.get("tool_name"),
                        "arguments":
                        ti.get("parameters") or ti.get("args") or {},
                    })
        except Exception:
            pass

    patterns = [
        "/root/.gemini/antigravity-cli/brain/*/.system_generated/logs/transcript*.jsonl",
        str(Path.home() /
            ".gemini/antigravity-cli/brain/*/.system_generated/logs/transcript*.jsonl"
            ),
        "/logs/agent/*.jsonl",
        "/logs/*.jsonl",
    ]
    for pat in patterns:
        for f in glob.glob(pat):
            try:
                for line in Path(f).read_text(encoding="utf-8",
                                              errors="ignore").splitlines():
                    if not line.strip():
                        continue
                    step = json.loads(line)
                    for tc in step.get("tool_calls", []):
                        calls.append(tc)
            except Exception:
                pass

    for pat in ["/logs/agent/**/subagent*.json", "/logs/subagent*.json"]:
        for f in glob.glob(pat, recursive=True):
            try:
                data = json.loads(
                    Path(f).read_text(encoding="utf-8", errors="ignore"))
                if isinstance(data, dict):
                    calls.extend(collect_tool_calls(data))
            except Exception:
                pass

    return calls


def _has_valid_artifact(workspace: Path) -> bool:
    for s_dir in [
            workspace,
            Path("/root/.tpu-agent-suite/runs"),
            Path.home() / ".tpu-agent-suite/runs",
    ]:
        try:
            if not s_dir.exists():
                continue
            for m in s_dir.glob("**/*engine_compat*"):
                if m.is_file() and m.stat(
                ).st_size > 0 and "/skills/" not in str(
                        m) and "/packages/" not in str(m):
                    return True
        except Exception:
            pass
    return False


@criterion(description=(
    "Verify engine_compatibility_check.py was executed on the remote "
    "TPU cluster via compute run without installing vllm"))
def check_command_executed(workspace: Path) -> bool:
    cmds: list[str] = []
    for tc in _collect_all_calls():
        tool_name = (tc.get("function_name") or tc.get("name") or "").lower()
        args = (tc.get("arguments") or tc.get("args") or tc.get("input") or {})
        if tool_name in _CMD_TOOLS and isinstance(args, dict):
            cmd = str(
                args.get("command") or args.get("CommandLine")
                or args.get("cmd") or args.get("command_line") or "")
            if cmd:
                cmds.append(cmd)

    if any(_INSTALL_VLLM.search(cmd) for cmd in cmds):
        return False

    if any("engine_compatibility_check.py" in cmd
           and "Qwen2.5-7B-Instruct" in cmd and bool(_COMPUTE_RUN.search(cmd))
           for cmd in cmds):
        return True

    return _has_valid_artifact(workspace)
