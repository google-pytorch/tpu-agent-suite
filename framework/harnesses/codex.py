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
"""Codex."""

from __future__ import annotations

from typing import Any

from framework.harnesses import base


class Codex(base.Harness):
    """Codex."""

    name = "codex"
    display_name = "Codex"
    config_dir = ".codex"

    # --- runtime: outbound -------------------------------------------------
    #
    # Replies are `decision` verdicts. SessionStart, SubagentStop and Stop
    # refuse with `block` and forward `context` as `additional_context`.
    # The tool events refuse with `deny` and forward rewritten arguments as
    # `tool_input`; only PostToolUse also forwards `context`. SessionEnd and
    # unknown events take the tool shape. `message` goes to stderr.

    message_on_stderr = True

    @staticmethod
    def _allow(reply: base.HookReply) -> dict[str, Any]:
        response: dict[str, Any] = {"decision": "allow"}
        if reply.context:
            response["additional_context"] = reply.context
        return response

    def _block_or_allow(self, reply: base.HookReply,
                        blocks: bool) -> dict[str, Any]:
        if blocks:
            return {"decision": "block", "reason": reply.effective_reason}
        return self._allow(reply)

    def _tool_reply(self, reply: base.HookReply,
                    with_context: bool) -> dict[str, Any]:
        if reply.is_deny:
            return {
                "decision": "deny",
                "reason": reply.reason or f"Denied by hook {reply.hook_name}",
            }
        response: dict[str, Any] = {"decision": "allow"}
        if with_context and reply.context:
            response["additional_context"] = reply.context
        if reply.updated_args is not None:
            response["tool_input"] = reply.updated_args
        return response

    def reply_session_start(self, reply: base.HookReply) -> dict[str, Any]:
        return self._block_or_allow(reply, reply.is_deny)

    def reply_subagent_stop(self, reply: base.HookReply) -> dict[str, Any]:
        return self._block_or_allow(reply, reply.is_deny)

    def reply_stop(self, reply: base.HookReply) -> dict[str, Any]:
        return self._block_or_allow(reply, reply.decision in ("retry", "deny"))

    def reply_pre_tool_use(self, reply: base.HookReply) -> dict[str, Any]:
        return self._tool_reply(reply, with_context=False)

    def reply_post_tool_use(self, reply: base.HookReply) -> dict[str, Any]:
        return self._tool_reply(reply, with_context=True)

    def reply_session_end(self, reply: base.HookReply) -> dict[str, Any]:
        return self._tool_reply(reply, with_context=False)

    def reply_unknown_event(self, reply: base.HookReply) -> dict[str, Any]:
        return self._tool_reply(reply, with_context=False)
