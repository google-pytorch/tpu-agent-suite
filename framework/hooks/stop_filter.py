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
"""Decides whether a SubagentStop is a finished subagent that owes a result.

The event also fires on a turn boundary, a halt, or a stop after the result
was already handed back. `subagent_stop_skip_reason` tells those apart, so
`dispatch` lets them through rather than holding them to a contract.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Iterator

from framework import bg_ledger, harnesses
from framework.hooks import events

Path = pathlib.Path


def _jsonl_lines(path: str) -> list[str]:
    """Return the non-empty lines of a JSONL file, or `[]` if unreadable."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [line for line in map(str.strip, text.splitlines()) if line]


def _parse_record(line: str) -> Any:
    """Parse a JSONL line, returning None if malformed."""
    try:
        return json.loads(line)
    except ValueError:
        return None


def _iter_nested_dicts(node: Any) -> Iterator[dict[str, Any]]:
    """Yield `node` and every dict nested inside it."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_nested_dicts(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_nested_dicts(item)


def _is_call_to(node: dict[str, Any], tool: str) -> bool:
    """Whether a transcript record is a `tool_use` call to `tool` with a string id."""
    return (node.get("type") == "tool_use" and node.get("name") == tool
            and isinstance(node.get("id"), str))


def _current_run_lines(
    lines: list[str],
    prompt_id: str,
    prompt_id_key: str,
) -> list[str] | None:
    """Return the slice of transcript lines starting at `prompt_id`, or None if not found."""
    if prompt_id and prompt_id_key:
        for index, line in enumerate(lines):
            if prompt_id not in line:
                continue
            record = _parse_record(line)
            if isinstance(record,
                          dict) and record.get(prompt_id_key) == prompt_id:
                return lines[index:]
    return None


def _is_accepted_result(node: dict[str, Any], call_ids: set[str]) -> bool:
    """Whether a transcript record is a non-error `tool_result` for one of `call_ids`."""
    return (node.get("type") == "tool_result"
            and node.get("tool_use_id") in call_ids
            and node.get("is_error") is not True)


def _handback_delivered(
    transcript_path: str,
    tool: str,
    prompt_id: str = "",
    prompt_id_key: str = "",
) -> bool:
    """Return True if the current subagent run already succeeded in calling the hand-back `tool`."""
    if not transcript_path or not tool:
        return False
    all_lines = _jsonl_lines(transcript_path)
    if not all_lines:
        return False
    lines = _current_run_lines(all_lines, prompt_id, prompt_id_key)
    if lines is None:
        print(
            f"runtime: cannot locate the stopping run in {transcript_path} "
            f"(prompt id {prompt_id or 'missing'!r}, transcript key "
            f"{prompt_id_key or 'missing'!r}); checking the native stop.",
            file=sys.stderr)
        return False
    call_ids = {
        node["id"]
        for line in lines if tool in line
        for node in _iter_nested_dicts(_parse_record(line))
        if _is_call_to(node, tool)
    }
    return any(
        _is_accepted_result(node, call_ids) for line in lines
        if any(call_id in line for call_id in call_ids)
        for node in _iter_nested_dicts(_parse_record(line)))


def _handback_offered(
    transcript_path: str,
    tool: str,
    offer: Any = None,
    prompt_id: str = "",
    prompt_id_key: str = "",
) -> bool | None:
    """Return whether the current subagent run was offered the hand-back `tool`, or None if unknown.

    Checks the run's transcript slice for either the harness's `transcript_offer`
    announcement record (`offer['flag']` + `offer['text']`) or an actual call to `tool`.
    """
    if not isinstance(offer, dict):
        return None
    flag = offer.get("flag")
    text = offer.get("text")
    if not transcript_path or not tool or not flag or not text:
        return None
    all_lines = _jsonl_lines(transcript_path)
    if not all_lines:
        return None
    lines = _current_run_lines(all_lines, prompt_id, prompt_id_key)
    if lines is None:
        return None
    for line in lines:
        if tool not in line:
            continue
        record = _parse_record(line)
        if (isinstance(record, dict) and record.get(flag) is True
                and text in line):
            return True
        if any(_is_call_to(node, tool) for node in _iter_nested_dicts(record)):
            return True
    return False


HANDBACK_SKIP = ("subagent delivers through the hand-back tool; this native "
                 "stop is a turn boundary")
HANDBACK_ASSUMED_SKIP = (
    "auto mode, but the transcript could not show whether the subagent was "
    "given the hand-back tool; this native stop is taken as a turn boundary "
    "and its result is not checked")


def _handback_skip_reason(
    event_payload: dict[str, Any],
    harness: str,
) -> str | None:
    """Return a skip reason if a native SubagentStop is covered by a hand-back tool, else None.

    When a subagent delivers via a hand-back tool (e.g. `SubagentHandback` in
    Claude Code auto mode), its native `SubagentStop` events are turn boundaries
    or closing remarks rather than the deliverable itself.
    """
    handback_mapping = harnesses.subagent_handback_mapping(harness)
    if handback_mapping is None:
        return None
    raw_input = event_payload.get("raw_input") or {}
    fields = harnesses.event_mapping("SubagentStop", harness).get("fields", {})
    transcript_path, prompt_id, prompt_id_key = _stop_run_transcript(
        raw_input, fields, handback_mapping)
    required_mode = handback_mapping.get("requires_mode")
    if required_mode and events.extract_canonical_field(
            raw_input, fields.get("permission_mode", [])) == required_mode:
        offered = _handback_offered(transcript_path,
                                    handback_mapping["matcher"],
                                    handback_mapping.get("transcript_offer"),
                                    prompt_id=prompt_id,
                                    prompt_id_key=prompt_id_key)
        if offered is True:
            return HANDBACK_SKIP
        if offered is None:
            return HANDBACK_ASSUMED_SKIP
    if _handback_delivered(
            transcript_path,
            handback_mapping["matcher"],
            prompt_id=prompt_id,
            prompt_id_key=prompt_id_key,
    ):
        return HANDBACK_SKIP
    return None


def _stop_run_transcript(
    raw_input: dict[str, Any],
    fields: dict[str, Any],
    handback_mapping: dict[str, Any],
) -> tuple[str, str, str]:
    """Locate the stopping subagent's own transcript and the run within it.

    Returns three strings, each empty when the payload or mapping lacks it:

    * the subagent's transcript path (Claude Code: `agent_transcript_path`),
      read through the SubagentStop mapping's `fields`. The payload's
      top-level `transcript_path` is the parent session's file and never
      holds the subagent's tool calls;
    * the id of the prompt that started this run (`prompt_id`), so that a
      resumed subagent is judged on its current run, not an earlier one;
    * the transcript record key that carries that id, from the hand-back
      mapping's `transcript_prompt_id`.

    `_handback_offered` and `_handback_delivered` scan the transcript with
    these to decide whether a native stop is the delivery or a turn boundary.
    """
    transcript_path = str(
        events.extract_canonical_field(raw_input,
                                       fields.get("transcript_path", []))
        or "")
    prompt_id = str(
        events.extract_canonical_field(raw_input, fields.get("prompt_id", []))
        or "")
    prompt_id_key = str(handback_mapping.get("transcript_prompt_id") or "")
    return transcript_path, prompt_id, prompt_id_key


_COMPLETED_SUBAGENT_TERMINATION_REASONS = frozenset({
    "no_tool_call",
    "terminal_step_type",
    "terminal_custom_hook",
})


def _agent_loop_completed(reason: Any) -> bool:
    """Return True if `reason` indicates the subagent loop completed normally."""
    text = str(reason or "").strip().lower()
    return (text.removeprefix("termination_reason_")
            in _COMPLETED_SUBAGENT_TERMINATION_REASONS)


def _subagent_was_cut_short(event_payload: dict[str, Any],
                            harness: str) -> bool:
    """Return True if a Google harness subagent was halted or cancelled before finishing."""
    if not harnesses.is_google_harness(harness):
        return False

    args = event_payload.get("args")
    reason = args.get("stop_reason") if isinstance(args, dict) else None
    return bool(reason) and not _agent_loop_completed(reason)


NO_SUBAGENT_NAME_SKIP = "no subagent name (internal helper or root session)"


def subagent_stop_skip_reason(
    event_payload: dict[str, Any],
    harness: str,
    harness_resumes: bool | None = None,
) -> str | None:
    """Return a skip reason if `event_payload` is not a finished subagent delivery, else None.

    Skips nameless stops (internal helpers or root sessions), halted/cancelled
    subagents, non-idle turn boundaries (`fully_idle is False`) that the
    harness will resume (`bg_ledger.harness_resumes_agent`; otherwise the stop
    is final and `bg_ledger.idle_gate_verdict` refuses it), and native stops
    when delivery is handled by a hand-back tool.

    `harness_resumes` is `bg_ledger.harness_resumes_agent` for this event, if
    the caller already asked, as the dispatcher does for this and the idle
    gate.

    Claude Code's `background_tasks` is a signal only intersected with the
    ledger of this agent's launches (`bg_ledger.outstanding_work`): it lists
    every live agent of the session with no parent link.
    """
    if not str(event_payload.get("subagent") or "").strip():
        return NO_SUBAGENT_NAME_SKIP

    if events.is_handback_event(event_payload):
        return None

    if _subagent_was_cut_short(event_payload, harness):
        return "subagent was cut short before finishing"

    args = event_payload.get("args")
    if isinstance(args, dict) and args.get("fully_idle") is False:
        if harness_resumes is None:
            harness_resumes = bg_ledger.harness_resumes_agent(
                event_payload, harness)
        if harness_resumes:
            return "subagent still has work outstanding (not fully idle)"

    handback_skip = _handback_skip_reason(event_payload, harness)
    if handback_skip:
        return handback_skip

    return None
