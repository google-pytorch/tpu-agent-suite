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

from typing import Any

from framework.harnesses import base


class ClaudeCode(base.Harness):
    """Claude Code."""

    name = "claude-code"
    display_name = "Claude Code"
    config_dir = ".claude"

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
