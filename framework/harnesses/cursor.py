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
"""Cursor."""

from __future__ import annotations

from typing import Any

from framework.harnesses import base


class Cursor(base.Harness):
    """Cursor."""

    name = "cursor"
    display_name = "Cursor"
    config_dir = ".cursor"

    # --- runtime: outbound -------------------------------------------------
    #
    # A refusal is `decision: cancel`, except on SubagentStop, which carries
    # a `followup_message` back to the subagent instead. `message` goes to
    # stderr; `context` is forwarded as `additional_context` only where
    # Cursor reads it (SessionStart, PostToolUse).

    message_on_stderr = True

    @staticmethod
    def _cancel(reply: base.HookReply) -> dict[str, Any]:
        return {
            "decision": "cancel",
            "reason": reply.reason or f"Cancelled by hook {reply.hook_name}",
        }

    def _cancel_or_empty(self,
                         reply: base.HookReply,
                         with_context: bool = False) -> dict[str, Any]:
        if reply.is_deny:
            return self._cancel(reply)
        response: dict[str, Any] = {}
        if with_context and reply.context:
            response["additional_context"] = reply.context
        return response

    def reply_session_start(self, reply: base.HookReply) -> dict[str, Any]:
        return self._cancel_or_empty(reply, with_context=True)

    def reply_subagent_stop(self, reply: base.HookReply) -> dict[str, Any]:
        if reply.is_deny:
            return {"followup_message": reply.effective_reason}
        return {}

    def reply_stop(self, reply: base.HookReply) -> dict[str, Any]:
        if reply.decision in ("retry", "deny"):
            return self._cancel(reply)
        return {}

    def reply_pre_tool_use(self, reply: base.HookReply) -> dict[str, Any]:
        return self._cancel_or_empty(reply)

    def reply_post_tool_use(self, reply: base.HookReply) -> dict[str, Any]:
        return self._cancel_or_empty(reply, with_context=True)

    def reply_session_end(self, reply: base.HookReply) -> dict[str, Any]:
        return self._cancel_or_empty(reply)

    def reply_unknown_event(self, reply: base.HookReply) -> dict[str, Any]:
        return self._cancel_or_empty(reply)
