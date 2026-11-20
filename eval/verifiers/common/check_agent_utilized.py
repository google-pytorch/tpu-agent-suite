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
import os
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
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


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
    traj = _load_trajectory(_trajectory_path())
    if not traj:
        print("check_agent_utilized: [FAIL] No trajectory found at "
              f"{_trajectory_path()}")
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
