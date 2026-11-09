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
_COMPUTE_INSPECT = re.compile(
    r"\bcompute(?:\s+--?[\w-]+(?:(?:=|\s+)(?!-)\S+)?)*\s+inspect\b")
_CONFIG_SHOW = re.compile(
    r"\bconfig(?:\s+--?[\w-]+(?:(?:=|\s+)(?!-)\S+)?)*\s+show\b")


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


_NO_TPU_STATUS = re.compile(
    r'"status"\s*:\s*"no_accelerator"|"chips"\s*:\s*0\b')
_PRESENT_TPU = re.compile(
    r'"status"\s*:\s*"(?:healthy|busy|unknown_contention)"')


def _collect_all_outputs() -> list[str]:
    outputs: list[str] = []
    for traj in ["/logs/agent/trajectory.json", "/logs/trajectory.json"]:
        data = load_trajectory(traj)
        if not isinstance(data, dict):
            continue
        for step in data.get("steps", []):
            if not isinstance(step, dict):
                continue
            obs = step.get("observation")
            if isinstance(obs, dict):
                for res in obs.get("results", []):
                    if isinstance(res, dict) and res.get("content"):
                        outputs.append(str(res["content"]))
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
                    if line.strip():
                        step = json.loads(line)
                        if isinstance(step, dict) and step.get("content"):
                            outputs.append(str(step["content"]))
            except Exception:
                pass
    return outputs


@criterion(description=(
    "Verify compute inspect was executed and found TPU nodes on the cluster"))
def check_command_executed(workspace: Path) -> bool:
    ran_inspect = False
    for tc in _collect_all_calls():
        tool_name = (tc.get("function_name") or tc.get("name") or "").lower()
        args = (tc.get("arguments") or tc.get("args") or tc.get("input") or {})
        if tool_name in _CMD_TOOLS and isinstance(args, dict):
            cmd = str(
                args.get("command") or args.get("CommandLine")
                or args.get("cmd") or args.get("command_line") or "")
            if _COMPUTE_INSPECT.search(cmd) or _CONFIG_SHOW.search(cmd):
                ran_inspect = True
                break
    if not ran_inspect:
        return False
    outputs = _collect_all_outputs()
    if any(_PRESENT_TPU.search(out) for out in outputs):
        return True
    if any(_NO_TPU_STATUS.search(out) for out in outputs):
        return False
    return True
