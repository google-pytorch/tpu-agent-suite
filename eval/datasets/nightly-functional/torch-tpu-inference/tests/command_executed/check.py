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
_SUBAGENT_TOOLS = {
    "invoke_subagent",
    "subagent",
    "call_subagent",
    "delegate",
}


def _collect_all_calls() -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    # 1. ATIF trajectory files
    for traj in ["/logs/agent/trajectory.json", "/logs/trajectory.json"]:
        data = load_trajectory(traj)
        if data:
            calls.extend(collect_tool_calls(data))

    # 2. Antigravity stream log (stream-json events)
    stream_logs = [
        Path("/logs/agent/stream.log"),
        Path("/logs/agent/stream.jsonl"),
        *Path("/logs/agent").glob("*.log"),
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
    for p in Path("/logs/agent").glob("**/*"):
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


def _has_artifact(workspace: Path, patterns: list[str]) -> bool:
    search_dirs = [
        workspace,
        workspace / "artifacts",
        Path("/logs/artifacts"),
        Path("/workspace"),
    ]
    for d in search_dirs:
        if not d.is_dir():
            continue
        for pat in patterns:
            if list(d.glob(pat)) or list(d.glob(f"**/{pat}")):
                return True
    return False


@criterion(description=(
    "Verify inference workflow execution commands or artifacts were produced"))
def check_command_executed(workspace: Path) -> bool:
    if _has_artifact(workspace, [
            "*deployments*.json", "*inference*.py", "*fastapi_server*.py",
            "*launch_vllm*.sh", "*health*.json", "*verification*.json"
    ]):
        return True

    target_names = {
        "torch_tpu_inference",
        "torch-tpu-inference",
        "torch_inference_coder",
        "torch-inference-coder",
        "inference",
        "deployments",
        "fastapi_server",
        "launch_vllm",
        "tpu-model-onboarding",
        "tpu_model_onboarding",
    }
    regex = re.compile(
        r"torch_tpu_inference|torch-tpu-inference|deployments\.json|fastapi_server\.py|launch_vllm\.sh|health\.json|verification\.json"
    )

    calls = _collect_all_calls()
    for tc in calls:
        tool_name = (tc.get("function_name") or tc.get("name") or "").lower()
        args = tc.get("arguments") or tc.get("args") or {}
        if tool_name in _CMD_TOOLS:
            cmd = str(
                args.get("command") or args.get("CommandLine")
                or args.get("cmd") or args.get("command_line") or "")
            if regex.search(cmd):
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
                        if regex.search(prompt) or regex.search(t_name):
                            return True
            args_str = str(args).lower()
            if any(n in args_str for n in target_names):
                return True

    return False
