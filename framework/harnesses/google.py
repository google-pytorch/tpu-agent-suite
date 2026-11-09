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
"""The family of harnesses built on Google's agent surface.

Members share one mapping block and one layout; each states only its name
and its brain directory.
"""

from __future__ import annotations

from typing import Any

from framework.harnesses import base

# The member whose mapping block the family inherits.
GOOGLE_HARNESS_MAPPING_KEY = "antigravity"


class GoogleHarness(base.Harness):
    """Shared behaviour of the Google harness family."""

    config_dir = ".agents"
    mapping_key = GOOGLE_HARNESS_MAPPING_KEY
    brain_dir: str  # conversation brains live at `~/.gemini/<brain_dir>/brain`

    # --- runtime: outbound -------------------------------------------------
    #
    # Replies are `decision` verdicts, except SessionStart, which injects
    # `ephemeralMessage` steps. `message` goes to stderr, since no reply
    # field carries it. Stop and SessionEnd share a shape because the family
    # emulates SessionEnd on its per-turn Stop.

    message_on_stderr = True

    @staticmethod
    def _refusal(reply: base.HookReply) -> dict[str, Any]:
        return {
            "decision": "deny",
            "reason": reply.reason or f"Denied by hook {reply.hook_name}",
        }

    def _refuse_or_empty(self, reply: base.HookReply) -> dict[str, Any]:
        return self._refusal(reply) if reply.is_deny else {}

    def reply_stop(self, reply: base.HookReply) -> dict[str, Any]:
        # "continue" is what re-enters the loop, and it is the whole point:
        # the conversation survives, so a resumed run picks up where it
        # stopped instead of repeating everything it had already done.
        if reply.decision == "retry":
            return {
                "decision": "continue",
                "reason": reply.reason or f"Resumed by hook {reply.hook_name}",
            }
        if reply.is_deny:
            return self._refusal(reply)
        return {"decision": "stop"}

    def reply_session_end(self, reply: base.HookReply) -> dict[str, Any]:
        return self.reply_stop(reply)

    def reply_session_start(self, reply: base.HookReply) -> dict[str, Any]:
        inject_steps: list[dict[str, Any]] = []
        if reply.context:
            inject_steps.append({"ephemeralMessage": reply.context})
        if reply.is_deny:
            inject_steps.append({
                "ephemeralMessage":
                f"Session Start Denied: {reply.effective_reason}"
            })
        return {"injectSteps": inject_steps}

    def reply_subagent_stop(self, reply: base.HookReply) -> dict[str, Any]:
        if reply.is_deny:
            return {"decision": "continue", "reason": reply.effective_reason}
        return {"decision": "allow"}

    def reply_pre_tool_use(self, reply: base.HookReply) -> dict[str, Any]:
        if reply.is_deny:
            return self._refusal(reply)
        res: dict[str, Any] = {"decision": "allow"}
        if reply.updated_args is not None:
            res["overwrite"] = reply.updated_args
        return res

    def reply_post_tool_use(self, reply: base.HookReply) -> dict[str, Any]:
        return self._refuse_or_empty(reply)

    def reply_unknown_event(self, reply: base.HookReply) -> dict[str, Any]:
        return self._refuse_or_empty(reply)
