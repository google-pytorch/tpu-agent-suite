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
"""Did the reply answer the question, and say what grounds it?

Every task drops the same artifact -- `{"answer": ..., "reference": ...}`
at /logs/artifacts/answer.json -- and one checker scores it. What varies
is the expectation, which each dataset supplies:

    {"keys": ["concept-vmem"],
     "references": ["jax-ml.github.io/scaling-book"],
     "contains": [["128 KiB", "128KiB"]]}

`contains` are phrase groups the answer must say -- any spelling in a
group counts, so a figure written three ways still counts once.
`references` are substrings the citation must contain. `keys` is what to
look up: the offline test reads it, the scoring ignores it.

Scoring what the reply wrote, rather than an internal record it was asked
to copy out, keeps this a test of the answer and not of transcription.

This copy, for user-scope-install, also scores the agent's last message
when no artifact was written: agy's `tpu-agent` persona has no tool that
writes a file, so it answers in its reply. And it takes the v6e page under
either host, docs.cloud.google.com or cloud.google.com: this task asks
whether an installed agent was reachable, not how a citation is spelled.

    python3 check_answer.py --result answer.json --expect '{...}'
"""

import argparse
import json
import pathlib
import re
import sys

EXPECT: dict = {
    "keys": ["tpu-v6e-spec"],
    "contains": [["32 GB", "32GB"]],
    "references": ["cloud.google.com/tpu/docs/v6e"]
}


def spoken(text: str, group: list[str]) -> bool:
    flat = re.sub(r"\s+", " ", text.lower())
    return any(re.sub(r"\s+", " ", p.lower()) in flat for p in group)


def check(result: dict, expect: dict) -> tuple[bool, list[str]]:
    """Why this answer fails the expectation, or nothing if it does not."""
    why: list[str] = []

    answer = str(result.get("answer") or "")
    if not answer.strip():
        why.append("no answer")
    for group in expect.get("contains") or []:
        if not spoken(answer, group):
            why.append(f"the answer never says {' / '.join(group)}")

    # One citation or several; a reply may ground an answer in more than
    # one place, and the shape it chose is not what is being scored.
    cited = result.get("reference")
    cited = " ".join(cited) if isinstance(cited, list) else str(cited or "")
    if not cited.strip():
        why.append("nothing cited")
    for wanted in expect.get("references") or []:
        if wanted not in cited:
            why.append(f"never cites {wanted}")

    return not why, why


# The one path the instructions name. Nothing is searched for: a run that
# did not write it scores zero, which is the honest reading of a trial that
# was asked for an artifact and produced none.
ANSWER = pathlib.Path("/logs/artifacts/answer.json")
TRAJECTORIES = (
    pathlib.Path("/logs/agent/trajectory.json"),
    pathlib.Path("/logs/trajectory.json"),
)


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


def found_result(workspace: pathlib.Path) -> dict | None:
    """The record the run saved, else its last reply standing in for both
    fields, or None if it left neither."""
    del workspace  # the artifact path is fixed, not discovered
    try:
        loaded = json.loads(ANSWER.read_text("utf-8"))
    except (OSError, ValueError):
        loaded = None
    if isinstance(loaded, dict):
        return loaded
    for path in TRAJECTORIES:
        try:
            said = last_message(json.loads(path.read_text("utf-8")))
        except (OSError, ValueError, AttributeError):
            continue
        if said:
            print(f"  the run saved no {ANSWER}; scoring its last message")
            return {"answer": said, "reference": said}
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", type=pathlib.Path)
    parser.add_argument("--workspace",
                        type=pathlib.Path,
                        default=None,
                        help="Unused; the artifact path is fixed")
    parser.add_argument("--expect", default="{}")
    args = parser.parse_args(argv)
    result = (json.loads(args.result.read_text("utf-8"))
              if args.result else found_result(args.workspace))
    if result is None:
        print(f"  the run saved no {ANSWER}")
        return 1
    ok, why = check(result, json.loads(args.expect))
    for reason in why:
        print(f"  {reason}")
    print("pass" if ok else "fail")
    return 0 if ok else 1


try:
    from rewardkit import criterion as _criterion
except ImportError:  # the ordinary case: a laptop, or the unit tests
    _criterion = None

if _criterion is not None:

    @_criterion(description="the lookup reached the right page")
    def right_page(workspace) -> bool:
        result = found_result(pathlib.Path(workspace))
        if result is None:
            print(f"  the run saved no {ANSWER}")
            return False
        ok, why = check(result, EXPECT)
        for reason in why:
            print(f"  {reason}")
        return ok


if __name__ == "__main__":
    sys.exit(main())
