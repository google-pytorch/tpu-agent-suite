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
A missing artifact scores zero: the task asked for one.

    python3 check_answer.py --result answer.json --expect '{...}'
"""

import argparse
import json
import pathlib
import re
import sys

EXPECT: dict = {
    "keys": ["torch-tpu-get-started-gke"],
    "contains": [["torchtpu_singledevice_slice_job.yaml"],
                 ["zero code change", "no code change",
                  "without code change"]],
    "references": ["get_started_on_gke"]
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


def found_result(workspace: pathlib.Path) -> dict | None:
    """The record the run was told to save, or None if it saved none."""
    del workspace  # the artifact path is fixed, not discovered
    try:
        loaded = json.loads(ANSWER.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


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
