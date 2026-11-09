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
"""Was the installed memory block in the session's context?

The two answers are said only in the package's memory block,
package/memory/tpu-knowledge-base.md; no agent or skill description
carries either, so a session can name them only if the harness loaded the
block from where the user-scope install wrote it. A test in
eval/tests keeps EXPECT and that file in step.

The answer is read from the file the task asks for, or, when the run wrote
none, from the agent's last message: a harness that says both answers but
holds back from writing a file was told "do not read, search or list any
file", and still had the block in context, which is what is measured.

Reading the answers off disk would pass the first check and prove nothing,
so the second fails a run whose tool calls named a memory file or the
installed package, searched, or globbed a harness directory -- whatever
else the call did, writing the answer file included.

    python3 check_memory.py --result answer.json --trajectory trajectory.json
"""

import argparse
import json
import pathlib
import re
import sys

EXPECT: dict = {
    "kernel_index": "kernel-optimization-index",
    "request_field": "entry_id",
}

# What a run that went looking would have named in a tool call: a memory
# file or the installed package. Not the answers: a call can carry them only
# once the run knows them, so they mark it stating its answer -- agy sends it
# as a message -- not looking for one. Not the harness directories as a
# whole either: the agents and skills installed there carry neither answer
# (a test in eval/tests holds their sources to that), and a harness
# reads its own files there -- agy its brain artifacts -- whatever it is
# asked.
LOOKED_AT = (
    "CLAUDE.md",
    "AGENTS.md",
    "GEMINI.md",
    ".tpu-agent-suite",
    "/opt/tpu-agent-suite",
)

# What reaches a memory file without naming it. The task forbids searching
# outright, so every search tool counts, and so does a shell search or a
# wildcard under a harness directory, where `cat ~/.gemini/A*` would read
# the block.
SEARCH_TOOLS = {"grep", "glob", "grep_search", "find_by_name", "search"}
SEARCHED = (
    re.compile(r"\bfind\s"),
    re.compile(r"\bgrep\s+(?:-\w*\s+)*-\w*[rR]"),
    re.compile(r"\b(?:rg|ag|ack)\s"),
    re.compile(r"\.(?:gemini|claude|codex)/[^\s\"']*[*?\[]"),
)

ANSWER = pathlib.Path("/logs/artifacts/answer.json")
TRAJECTORIES = (
    pathlib.Path("/logs/agent/trajectory.json"),
    pathlib.Path("/logs/trajectory.json"),
)


def _flat(text: str) -> str:
    return text.lower().replace("_", "-")


def check_answer(result: dict) -> list[str]:
    """Why the answer misses the block, or nothing if it names both."""
    why = []
    for key, wanted in EXPECT.items():
        said = str(result.get(key) or "")
        if _flat(wanted) not in _flat(said):
            why.append(f"{key} is {said!r}, not {wanted}")
    return why


def tool_calls(trajectory: dict) -> list[dict]:
    return [
        call for step in trajectory.get("steps", [])
        for call in step.get("tool_calls") or []
    ]


def check_looked(trajectory: dict) -> list[str]:
    """Tool calls that went to disk for what should have been in context."""
    why = []
    for call in tool_calls(trajectory):
        args = json.dumps(call.get("arguments") or call.get("args") or {})
        name = call.get("function_name") or call.get("name") or "?"
        if name.lower() in SEARCH_TOOLS:
            why.append(f"{name} searched: {args[:160]}")
            continue
        named = next((probe for probe in LOOKED_AT if probe in args), None)
        if named:
            why.append(f"{name} touched {named}: {args[:160]}")
            continue
        if any(pattern.search(args) for pattern in SEARCHED):
            why.append(f"{name} searched or globbed: {args[:160]}")
    return why


def load(path: pathlib.Path) -> dict | None:
    try:
        loaded = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def found_trajectory() -> dict | None:
    for path in TRAJECTORIES:
        loaded = load(path)
        if loaded is not None:
            return loaded
    return None


def last_message(trajectory: dict) -> str:
    """The agent's final reply, text parts joined; empty if it gave none."""
    for step in reversed(trajectory.get("steps", [])):
        if step.get("source") != "agent" or not step.get("message"):
            continue
        message = step["message"]
        if isinstance(message, list):
            message = "\n".join(
                str(part.get("text", "")) for part in message
                if isinstance(part, dict))
        return str(message)
    return ""


def found_answer(trajectory: dict | None) -> dict | None:
    """The answer file, or the last reply standing in for every key."""
    result = load(ANSWER)
    if result is not None:
        return result
    said = last_message(trajectory or {})
    if not said:
        return None
    print(f"  the run saved no {ANSWER}; scoring its last message")
    return {key: said for key in EXPECT}


def report(why: list[str]) -> bool:
    for reason in why:
        print(f"  {reason}")
    return not why


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", type=pathlib.Path, default=ANSWER)
    parser.add_argument("--trajectory", type=pathlib.Path)
    args = parser.parse_args(argv)
    result = load(args.result)
    if result is None:
        print(f"  the run saved no {args.result}")
        return 1
    ok = report(check_answer(result))
    trajectory = (load(args.trajectory)
                  if args.trajectory else found_trajectory())
    if trajectory is None:
        print("  no trajectory to check what the run read")
        ok = False
    else:
        ok = report(check_looked(trajectory)) and ok
    print("pass" if ok else "fail")
    return 0 if ok else 1


try:
    from rewardkit import criterion as _criterion
except ImportError:  # the ordinary case: a laptop, or the unit tests
    _criterion = None

if _criterion is not None:

    @_criterion(description="the answers came from the installed memory")
    def answered_from_memory(workspace) -> bool:
        del workspace  # the artifact path is fixed, not discovered
        result = found_answer(found_trajectory())
        if result is None:
            print(f"  the run saved no {ANSWER} and said nothing")
            return False
        return report(check_answer(result))

    @_criterion(description="nothing was read to find them")
    def read_nothing(workspace) -> bool:
        del workspace
        trajectory = found_trajectory()
        if trajectory is None:
            print("  no trajectory to check what the run read")
            return False
        return report(check_looked(trajectory))


if __name__ == "__main__":
    sys.exit(main())
