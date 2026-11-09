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
"""RewardKit verifier to check if an agent utilized a subagent."""

import json
from pathlib import Path
from typing import Any

from rewardkit import criterion

_SUBAGENT_TOOLS = {"invoke_subagent", "agent", "task", "subagent", "delegate"}


def _load_trajectory(path: str | Path) -> dict[str, Any] | None:
    """Load an ATIF trajectory JSON file. Returns None on error."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _load_verifier_params() -> dict[str, Any]:
    p = Path("/tests/verifier_params.json")
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        key = Path(__file__).parent.name
        val = data.get(key, data)
        return val if isinstance(val, dict) else data
    except (json.JSONDecodeError, OSError):
        return {}


def _collect_tool_calls(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect all tool calls across all steps."""
    calls: list[dict[str, Any]] = []
    for step in data.get("steps", []):
        for tc in step.get("tool_calls") or []:
            calls.append(tc)
    return calls


@criterion(description="Verify that a subagent was invoked during execution")
def check_agent_utilized(workspace: Path) -> bool:
    """Checks Harbor's standardized ATIF trajectory for subagent invocation."""
    traj = _load_trajectory("/logs/agent/trajectory.json") or _load_trajectory(
        "/logs/trajectory.json")
    if not traj:
        print(
            "check_agent_utilized: [FAIL] No trajectory found at /logs/agent/trajectory.json"
        )
        return False

    target_agent = _load_verifier_params().get("target_agent")
    if isinstance(target_agent, str):
        target_agent = [target_agent]

    for call in _collect_tool_calls(traj):
        tool_name = (call.get("function_name") or call.get("name")
                     or "").lower()
        if tool_name in _SUBAGENT_TOOLS:
            if not target_agent:
                print(
                    f"check_agent_utilized: [PASS] Subagent invoked via tool '{tool_name}'"
                )
                return True

            args_str = json.dumps(
                call.get("arguments") or call.get("args")
                or {}).lower().replace("_", "-")
            if any(target in args_str for target in target_agent):
                print(
                    f"check_agent_utilized: [PASS] Target subagent found in tool '{tool_name}'"
                )
                return True

    print(
        f"check_agent_utilized: [FAIL] No matching subagent invocation found (targets={target_agent or 'any'})"
    )
    return False
