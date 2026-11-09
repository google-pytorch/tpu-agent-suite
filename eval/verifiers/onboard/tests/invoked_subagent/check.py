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
"""invoked_subagent: the run delegated to a TPU agent suite subagent.

An agent-suite run can still do the work itself: it never delegates, it
starts a generic helper and hands it a suite agent's name as a role, or its
delegation fails (an agent installed where the harness does not look fails
with "not found or not allowed to be invoked") and it carries on inline. This
check passes only when at least one delegation started one of the suite's own
agents and did not fail.

It reads Harbor's ATIF trajectory, which holds each delegation and its result:

- Claude Code: an `Agent` call (`Task`, its earlier name) whose
  `subagent_type` is the agent, bare or plugin-namespaced
  (`tpu-agent-suite:tpu-model-onboarding`). A failed result has
  `extra.tool_result_is_error`.
- Antigravity: an `invoke_subagent` call naming each agent in
  `Subagents[].TypeName`, or in a top-level `TypeName` from agy 1.4.7, which
  sends the arguments flat. A failed result has `extra.status` ERROR.

A `Role` is free text, not an agent type, so it never counts. A delegation
with no recorded result (the run ended while the subagent worked) counts,
since it started.

The suite's agents are the keys of `contracts` in the compiled package's
tpu-agent-suite-build.json, the map the suite's runtime uses to validate
delegations. Agent-suite runs mount the package under
/root/.tpu-agent-suite/packages/; a baseline run has none, so it scores 0.

Each nightly tpu-inference-*-vllm task carries a byte-identical copy of this
file as tests/invoked_subagent/check.py; eval/verifiers/common/invoked_subagent.py
is the source, and eval/tests/test_invoked_subagent.py keeps the copies in
step.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

from rewardkit import criterion

_TRAJECTORY = "/logs/agent/trajectory.json"
_BUILD_MANIFESTS = "/root/.tpu-agent-suite/packages/*/tpu-agent-suite-build.json"
_FAILED_STATUSES = ("ERROR", "FAILED")


def suite_agents() -> set[str]:
    """The agents the installed package ships; empty without the suite."""
    names = set()
    for path in glob.glob(_BUILD_MANIFESTS):
        names |= set(json.loads(Path(path).read_text())["contracts"])
    return names


def agent_types(tool: str, args: dict[str, Any]) -> list[str]:
    """The agent types one tool call starts; empty if it is no delegation."""
    if tool in ("Agent", "Task"):
        named = [args.get("subagent_type")]
    elif tool == "invoke_subagent":
        named = [sub.get("TypeName") for sub in args.get("Subagents") or []]
        if not named:
            named = [args.get("TypeName")]
    else:
        return []
    return [n for n in named if isinstance(n, str) and n]


def delegations(trajectory: dict[str, Any]) -> list[tuple[str, bool]]:
    """(agent type, failed) for each delegation in an ATIF trajectory."""
    found = []
    for step in trajectory["steps"]:
        observation = step.get("observation") or {}
        results = {
            r["source_call_id"]: r
            for r in observation["results"]
        } if observation else {}
        for call in step.get("tool_calls") or []:
            result = results.get(call["tool_call_id"]) or {}
            extra = result.get("extra") or {}
            failed = (extra.get("tool_result_is_error") is True
                      or extra.get("status") in _FAILED_STATUSES)
            found += [(name, failed) for name in agent_types(
                call["function_name"], call["arguments"])]
    return found


def _check() -> bool:
    agents = suite_agents()
    found = delegations(json.loads(Path(_TRAJECTORY).read_text()))
    # A plugin install names an agent `<package>:<agent>`; the suite's runtime
    # strips the prefix the same way.
    started = sorted({
        name
        for name, failed in found
        if not failed and name.split(":", 1)[-1] in agents
    })
    print(f"invoked_subagent: suite agents: "
          f"{sorted(agents) or 'none (the suite is not installed)'}")
    print(f"invoked_subagent: delegations (agent, failed): {found or 'none'}")
    print(f"invoked_subagent: suite agents started: {started or 'none'}")
    return bool(started)


@criterion(description="The agent delegated to a TPU agent suite subagent "
           "(an agent the installed package ships), and the delegation did "
           "not fail")
def invoked_subagent(workspace: Path) -> bool:
    del workspace  # The logs it reads are outside the workspace.
    # rewardkit turns a criterion's exception into a failed verifier run,
    # which loses every other score in the trial. This check is only
    # reported, so logs it cannot read score 0 instead.
    try:
        return _check()
    except Exception as error:  # pylint: disable=broad-exception-caught
        print(f"invoked_subagent: could not read the run's logs: "
              f"{type(error).__name__}: {error}")
        return False
