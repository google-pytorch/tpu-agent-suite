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
"""Claude Code."""

from __future__ import annotations

import json
import pathlib
import sys
from collections.abc import Iterator
from typing import Any

from framework import utils
from framework.harnesses import base


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


def _load_run_lines(raw_input: dict[str, Any],
                    fields: dict[str, Any],
                    handback_mapping: dict[str, Any],
                    *,
                    warn_missing_run: bool = False) -> list[str] | None:
    """Locate the stopping subagent's own transcript and return the current run's lines.

    Reads the subagent's `transcript_path` (`agent_transcript_path`, never the
    parent session's top-level `transcript_path`) and slices from the record
    matching the stopping run's `prompt_id` (`transcript_prompt_id`).
    """
    transcript_path = str(
        utils.extract_first_field(raw_input, fields.get("transcript_path", []))
        or "")
    if not transcript_path:
        return None
    try:
        text = pathlib.Path(transcript_path).read_text(encoding="utf-8",
                                                       errors="replace")
    except OSError:
        return None
    lines = [line for line in map(str.strip, text.splitlines()) if line]
    if not lines:
        return None
    prompt_id = str(
        utils.extract_first_field(raw_input, fields.get("prompt_id", []))
        or "")
    prompt_id_key = str(handback_mapping.get("transcript_prompt_id") or "")
    if prompt_id and prompt_id_key:
        for index, line in enumerate(lines):
            if prompt_id in line:
                record = _parse_record(line)
                if isinstance(record,
                              dict) and record.get(prompt_id_key) == prompt_id:
                    return lines[index:]
    if warn_missing_run:
        print(
            f"runtime: cannot locate the stopping run in {transcript_path} "
            f"(prompt id {prompt_id or 'missing'!r}, transcript key "
            f"{prompt_id_key or 'missing'!r}); checking the native stop.",
            file=sys.stderr)
    return None


def _is_accepted_result(node: dict[str, Any], call_ids: set[str]) -> bool:
    """Whether a transcript record is a non-error `tool_result` for one of `call_ids`."""
    return (node.get("type") == "tool_result"
            and node.get("tool_use_id") in call_ids
            and node.get("is_error") is not True)


def _handback_delivered(lines: list[str], tool: str) -> bool:
    """Return True if `lines` contain an accepted call to the hand-back `tool`."""
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


def _handback_offered(lines: list[str], tool: str, offer: dict[str,
                                                               Any]) -> bool:
    """Return True if `lines` contain the `transcript_offer` record or a call to `tool`."""
    flag, text = offer["flag"], offer["text"]
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


class ClaudeCode(base.Harness):
    """Claude Code."""

    name = "claude-code"
    display_name = "Claude Code"
    config_dir = ".claude"

    # --- runtime: inbound --------------------------------------------------

    def handback_skip_reason(self, event_payload: dict[str,
                                                       Any]) -> str | None:
        """Skip native `SubagentStop` events whose delivery goes through `SubagentHandback`."""
        args = event_payload.get("args")
        if isinstance(args, dict) and args.get("handback") is True:
            return None
        handback_mapping = self.handback_mapping()
        if handback_mapping is None:
            return None
        raw_input = event_payload.get("raw_input") or {}
        fields = self.event_mapping("SubagentStop").get("fields", {})
        tool = handback_mapping["matcher"]
        required_mode = handback_mapping.get("requires_mode")
        if required_mode and utils.extract_first_field(
                raw_input, fields.get("permission_mode", [])) == required_mode:
            offer = handback_mapping.get("transcript_offer")
            if not isinstance(
                    offer,
                    dict) or not offer.get("flag") or not offer.get("text"):
                return base.HANDBACK_ASSUMED_SKIP
            lines = _load_run_lines(raw_input, fields, handback_mapping)
            if lines is None:
                return base.HANDBACK_ASSUMED_SKIP
            return HANDBACK_SKIP if _handback_offered(lines, tool,
                                                      offer) else None
        lines = _load_run_lines(raw_input,
                                fields,
                                handback_mapping,
                                warn_missing_run=True)
        if lines is not None and _handback_delivered(lines, tool):
            return HANDBACK_SKIP
        return None

    # --- runtime: outbound -------------------------------------------------
    #
    # Every reply opens with the hook's `message` as `systemMessage`. A
    # refusal is `decision: block` with a reason on the stop, session and
    # PostToolUse events, and `permissionDecision: deny` on PreToolUse. The
    # event's own block is `hookSpecificOutput`, which carries
    # `additionalContext`. SessionEnd and unknown events get the banner only.

    @staticmethod
    def _reply(reply: base.HookReply) -> dict[str, Any]:
        return {"systemMessage": reply.message} if reply.message else {}

    @staticmethod
    def _specific(event: str, context: str | None) -> dict[str, Any]:
        h_out: dict[str, Any] = {"hookEventName": event}
        if context:
            h_out["additionalContext"] = context
        return h_out

    def _blocking_reply(self, reply: base.HookReply, event: str,
                        blocks: bool) -> dict[str, Any]:
        response = self._reply(reply)
        if blocks:
            response["decision"] = "block"
            response["reason"] = reply.effective_reason
        response["hookSpecificOutput"] = self._specific(event, reply.context)
        return response

    def reply_session_end(self, reply: base.HookReply) -> dict[str, Any]:
        # The session is over; there is no loop left to block or resume,
        # so the only thing worth returning is the message.
        return self._reply(reply)

    def reply_session_start(self, reply: base.HookReply) -> dict[str, Any]:
        return self._blocking_reply(reply, "SessionStart", reply.is_deny)

    def reply_subagent_stop(self, reply: base.HookReply) -> dict[str, Any]:
        return self._blocking_reply(reply, "SubagentStop", reply.is_deny)

    def reply_stop(self, reply: base.HookReply) -> dict[str, Any]:
        blocks = reply.decision in ("retry", "deny")
        return self._blocking_reply(reply, "Stop", blocks)

    def reply_pre_tool_use(self, reply: base.HookReply) -> dict[str, Any]:
        h_out: dict[str, Any] = {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow" if not reply.is_deny else "deny",
        }
        if reply.is_deny and reply.reason:
            h_out["permissionDecisionReason"] = reply.reason
        if reply.updated_args is not None:
            h_out["updatedInput"] = reply.updated_args
        response = self._reply(reply)
        response["hookSpecificOutput"] = h_out
        return response

    def reply_post_tool_use(self, reply: base.HookReply) -> dict[str, Any]:
        # Not `_blocking_reply`: the reason here is the raw one, and only
        # when the hook gave one.
        response = self._reply(reply)
        if reply.is_deny:
            response["decision"] = "block"
            if reply.reason:
                response["reason"] = reply.reason
        response["hookSpecificOutput"] = self._specific(
            "PostToolUse", reply.context)
        return response

    def reply_unknown_event(self, reply: base.HookReply) -> dict[str, Any]:
        return self._reply(reply)
