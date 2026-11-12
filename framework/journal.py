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
"""The session journal: one experiment timeline every agent in a run writes.

Entries are JSON lines in `<runs>/<run_id>/journal.jsonl`, beside the run's
`artifacts/`; `experiment.md` next to it is re-rendered after every write.
Writes hold one lock, so concurrent agents never interleave or lose an entry.
Not the study's `studies/<study_id>/journal.jsonl`, which outlives a run.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from framework import jsonio, paths, utils

JOURNAL_FILE = "journal.jsonl"
RENDERED_FILE = "experiment.md"
# Each kind and the letter its ids start with.
KINDS = {
    "hypothesis": "H",
    "experiment": "E",
    "result": "R",
    "decision": "D",
    "rejected": "X",
    "blocked": "B",
    "note": "N",
}
VERDICTS = ("supported", "refuted", "inconclusive")
# Kinds that answer to a hypothesis and must name it as their parent.
NEEDS_HYPOTHESIS = ("experiment", "result", "rejected")


class Refused(utils.TpuAgentSuiteError):
    """A write the journal will not take; the message says why."""


def build(entries: list[dict[str, Any]],
          run_id: str,
          agent: str,
          kind: str,
          text: str,
          parent: str | None = None,
          verdict: str | None = None,
          evidence: list[str] | None = None,
          metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    """Check one entry against the journal so far and return it."""
    if not agent.strip() or not text.strip():
        raise Refused("--agent and --text must not be empty")
    kinds = {e["id"]: e["kind"] for e in entries}
    if kind in NEEDS_HYPOTHESIS and kinds.get(parent) != "hypothesis":
        raise Refused(
            f"a {kind} names the hypothesis it answers: --parent "
            f"{parent or '<id>'} is not a hypothesis in this journal")
    if parent and parent not in kinds:
        raise Refused(f"--parent {parent} is not in this journal")
    if kind == "result" and (verdict not in VERDICTS or not evidence):
        raise Refused(f"a result needs --verdict ({', '.join(VERDICTS)}) and "
                      "--evidence: the path or job id its numbers came from")
    if verdict and kind != "result":
        raise Refused("--verdict belongs on a result")
    n = 1 + sum(e["kind"] == kind for e in entries)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "id": f"{KINDS[kind]}{n}",
        "run_id": run_id,
        "agent": " ".join(agent.split()),
        "kind": kind,
        "text": " ".join(text.split()),
        "parent": parent,
        "verdict": verdict,
        "evidence": evidence,
        "metrics": metrics,
    }
    return {k: v for k, v in entry.items() if v}


def status(hypothesis: dict[str, Any], entries: list[dict[str, Any]]) -> str:
    """open, or the latest verdict, or rejected."""
    state = "open"
    for e in entries:
        if e.get("parent") == hypothesis["id"] and e["kind"] in ("result",
                                                                 "rejected"):
            state = e.get("verdict", "rejected")
    return state


def render(entries: list[dict[str, Any]], run_id: str) -> str:
    """The experiment.md Markdown: a hypothesis table, then the timeline."""
    lines = [
        f"# Experiment journal: run {run_id}", "",
        f"Rendered from `{JOURNAL_FILE}` after every entry; do not edit.", ""
    ]
    hyps = [e for e in entries if e["kind"] == "hypothesis"]
    if hyps:
        lines += [
            "## Hypotheses", "",
            "| ID | Agent | Status | Hypothesis | Latest result |",
            "| --- | --- | --- | --- | --- |"
        ]
        for h in hyps:
            answers = [
                e["text"] for e in entries
                if e.get("parent") == h["id"] and e["kind"] in ("result",
                                                                "rejected")
            ]
            cells = [h["agent"], h["text"], answers[-1] if answers else ""]
            agent, text, last = (c.replace("|", "\\|") for c in cells)
            lines.append(f"| {h['id']} | {agent} | {status(h, entries)} "
                         f"| {text} | {last} |")
        lines.append("")
    lines += ["## Timeline", ""]
    for e in entries:
        head = f"- `{e['ts']}` **{e['agent']}** {e['kind']} {e['id']}"
        head += f" (for {e['parent']})" if e.get("parent") else ""
        head += f": **{e['verdict']}**" if e.get("verdict") else ""
        lines.append(f"{head}. {e['text']}")
        if e.get("metrics"):
            lines.append("  - " + ", ".join(f"{k} = {v}"
                                            for k, v in e["metrics"].items()))
        lines += [f"  - `{ev}`" for ev in e.get("evidence", [])]
    return "\n".join(lines) + "\n"


def _run_dir(run_id: str) -> Path:
    """The run's directory, which the session created; never made here."""
    directory = paths.get_run_root(run_id)
    if not directory.is_dir():
        raise Refused(f"run {run_id} has no directory at {directory}; use the "
                      "run_id your session was given")
    return directory


def read(run_id: str) -> list[dict[str, Any]]:
    """The run's journal entries, oldest first."""
    return jsonio.read_jsonl(_run_dir(run_id) / JOURNAL_FILE)


def write(run_id: str, **fields: Any) -> dict[str, Any] | None:
    """Append one entry (if fields are given) and re-render experiment.md.

    The journal file itself is the lock, held while it is read, so two agents
    writing at once can't take the same id or interleave lines.
    """
    directory = _run_dir(run_id)
    with open(directory / JOURNAL_FILE, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # Released when the file closes.
        entries = jsonio.read_jsonl(directory / JOURNAL_FILE)
        entry = build(entries, run_id, **fields) if fields else None
        if entry:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
            entries.append(entry)
        jsonio.write_text_atomic(directory / RENDERED_FILE,
                                 render(entries, run_id))
    return entry


def _metric(pair: str) -> tuple[str, Any]:
    key, sep, value = pair.partition("=")
    key, value = key.strip(), value.strip()
    if not sep or not key or not value:
        raise ValueError(pair)
    try:
        number = float(value)
    except ValueError:
        return key, value
    # nan/inf stay strings: json.dumps would write bare NaN/Infinity, which
    # strict JSON readers reject, and a NaN loss is still worth recording.
    return key, number if math.isfinite(number) else value


def register_journal_parser(sub: Any) -> None:
    """Attach `journal add|show|render` to the runtime's subparsers."""
    journal = sub.add_parser("journal",
                             help="The session's experiment journal for a run")
    verbs = journal.add_subparsers(dest="journal_action", required=True)
    for name in ("add", "show", "render"):
        verbs.add_parser(name).add_argument("--run-id", required=True)
    add, show = verbs.choices["add"], verbs.choices["show"]
    add.add_argument("--agent", required=True)
    add.add_argument("--kind", required=True, choices=list(KINDS))
    add.add_argument("--text", required=True)
    add.add_argument("--parent")
    add.add_argument("--verdict", choices=VERDICTS)
    add.add_argument("--evidence", action="append", help="repeatable")
    add.add_argument("--metric",
                     action="append",
                     type=_metric,
                     help="key=value, repeatable")
    show.add_argument("--open",
                      action="store_true",
                      help="only hypotheses with no result yet")
    show.add_argument("--agent")
    show.add_argument("--kind", choices=list(KINDS))


def dispatch_journal_command(args: Any) -> None:
    """Run one `journal` verb; a refused write raises Refused."""
    if args.journal_action == "add":
        entry = write(args.run_id,
                      agent=args.agent,
                      kind=args.kind,
                      text=args.text,
                      parent=args.parent,
                      verdict=args.verdict,
                      evidence=args.evidence,
                      metrics=dict(args.metric or ()))
        print(json.dumps(entry, sort_keys=True))
    elif args.journal_action == "render":
        write(args.run_id)
        print(paths.get_run_root(args.run_id) / RENDERED_FILE)
    else:
        entries = read(args.run_id)
        agent = " ".join(args.agent.split()) if args.agent else None
        for e in entries:
            if ((not args.open or
                 (e["kind"] == "hypothesis" and status(e, entries) == "open"))
                    and agent in (None, e["agent"])
                    and args.kind in (None, e["kind"])):
                print(json.dumps(e, sort_keys=True))
