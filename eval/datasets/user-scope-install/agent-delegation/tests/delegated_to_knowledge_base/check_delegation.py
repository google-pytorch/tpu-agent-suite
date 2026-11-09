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
"""Did the session delegate to the target agent?

eval/verifiers/common/check_agent_utilized.py reads only Harbor's ATIF
trajectory. Harbor writes none for a Claude Code run that leaves more than
one session directory, which a delegating run can, so a delegation that
happened would score as one that did not. Claude Code's own stream log,
/logs/agent/claude-code.txt, records every tool call the session made, and
is read when there is no trajectory.

    python3 check_delegation.py --trajectory t.json --stream claude-code.txt
"""

import argparse
import json
import pathlib
import sys

SUBAGENT_TOOLS = {"invoke_subagent", "agent", "task", "subagent", "delegate"}

TRAJECTORIES = (
    pathlib.Path("/logs/agent/trajectory.json"),
    pathlib.Path("/logs/trajectory.json"),
)
STREAM = pathlib.Path("/logs/agent/claude-code.txt")
PARAMS = pathlib.Path("/tests/verifier_params.json")


def targets() -> list[str]:
    try:
        params = json.loads(PARAMS.read_text("utf-8"))
    except (OSError, ValueError):
        return []
    return list((params.get(pathlib.Path(__file__).parent.name)
                 or {}).get("target_agent") or [])


def trajectory_calls(trajectory: dict) -> list[tuple[str, object]]:
    return [(str(call.get("function_name") or call.get("name")
                 or ""), call.get("arguments") or call.get("args") or {})
            for step in trajectory.get("steps", [])
            for call in step.get("tool_calls") or []]


def stream_calls(lines: list[str]) -> list[tuple[str, object]]:
    calls = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("type") != "assistant":
            continue
        for part in (event.get("message") or {}).get("content") or []:
            if isinstance(part, dict) and part.get("type") == "tool_use":
                calls.append((str(part.get("name") or ""), part.get("input")
                              or {}))
    return calls


def delegated(calls: list[tuple[str, object]],
              wanted: list[str]) -> str | None:
    """The tool that delegated to a wanted agent, or None."""
    for name, args in calls:
        if name.lower() not in SUBAGENT_TOOLS:
            continue
        said = json.dumps(args).lower().replace("_", "-")
        if not wanted or any(target in said for target in wanted):
            return name
    return None


def found_calls(trajectory: pathlib.Path | None = None,
                stream: pathlib.Path | None = None):
    """Tool calls from the trajectory, else the stream log, and the source."""
    for path in ([trajectory] if trajectory else list(TRAJECTORIES)):
        try:
            return trajectory_calls(json.loads(path.read_text("utf-8"))), path
        except (OSError, ValueError, AttributeError):
            continue
    path = stream or STREAM
    try:
        return stream_calls(path.read_text("utf-8").splitlines()), path
    except OSError:
        return None, None


def check(calls, source, wanted: list[str]) -> bool:
    if calls is None:
        print("  no trajectory and no Claude Code stream log to read")
        return False
    tool = delegated(calls, wanted)
    if tool:
        print(
            f"  [PASS] {tool} delegated to {wanted or 'an agent'} ({source})")
        return True
    print(f"  [FAIL] no delegation to {wanted or 'any agent'} in {source}")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trajectory", type=pathlib.Path)
    parser.add_argument("--stream", type=pathlib.Path)
    parser.add_argument("--target", action="append", default=[])
    args = parser.parse_args(argv)
    calls, source = found_calls(args.trajectory, args.stream)
    return 0 if check(calls, source, args.target) else 1


try:
    from rewardkit import criterion as _criterion
except ImportError:  # the ordinary case: a laptop, or the unit tests
    _criterion = None

if _criterion is not None:

    @_criterion(description="the session delegated to the target agent")
    def delegated_to_target(workspace) -> bool:
        del workspace  # the logs are at fixed paths, not in the workspace
        calls, source = found_calls()
        return check(calls, source, targets())


if __name__ == "__main__":
    sys.exit(main())
