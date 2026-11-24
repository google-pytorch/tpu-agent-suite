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

import re
from typing import Any

from framework.harnesses import base

# Codex `multi_agent_v2` encrypts `collaborationspawn_agent` prompts as Fernet
# tokens (`0x80` version byte + 64-bit timestamp -> base64url `"gAAAAA..."`).
_ENCRYPTED_DELEGATION_PROMPT_RE = re.compile(r"gAAAAA[A-Za-z0-9_-]{80,}={0,2}")


class Codex(base.Harness):
    """Codex."""

    name = "codex"
    display_name = "Codex"
    config_dir = ".codex"

    # Runtime: inbound.
    #
    # MultiAgentV1 `spawn_agent` accepts either `message` or structured `items`
    # (`[{"type": "text", "text": "..."}]`). Under MultiAgentV2, the Responses
    # API server-encrypts `message` as a Fernet token (`gAAAAA...`), so
    # `validate-schema` skips `PreToolUse` and checks plaintext at `SubagentStop`.

    def normalize_tool_args(
        self,
        canonical_tool: str,
        normalized_args: dict[str, Any],
        raw_args: dict[str, Any],
    ) -> None:
        """Extract `prompt` from `MultiAgentV1` `spawn_agent` `items` blocks."""
        if canonical_tool != "delegate" or normalized_args.get("prompt"):
            return
        items = raw_args.get("items")
        if not isinstance(items, list):
            return
        texts = [
            item["text"] for item in items
            if isinstance(item, dict) and item.get("type", "text") == "text"
            and isinstance(item.get("text"), str) and item["text"].strip()
        ]
        if texts:
            normalized_args["prompt"] = "\n".join(texts)

    def is_encrypted_delegation(self, prompt: Any) -> bool:
        """Whether `prompt` is a `MultiAgentV2` Fernet-encrypted token."""
        return isinstance(prompt, str) and bool(
            _ENCRYPTED_DELEGATION_PROMPT_RE.fullmatch(prompt))

    def max_event_timeout(self, canonical_event: str) -> int | None:
        """Codex clamps `SessionEnd` hook timeouts to 3 seconds."""
        return 3 if canonical_event == "SessionEnd" else None

    # Runtime: outbound.
    #
    # Replies follow Codex's `ClaudeHooksEngine` schemas (`additionalProperties:
    # false`). `message` rides as `systemMessage`, and `context` goes in
    # `hookSpecificOutput.additionalContext`. Refusals use `continue: false` +
    # `stopReason` on SessionStart, `decision: "block"` + `reason` on Stop,
    # SubagentStop and PostToolUse, and `permissionDecision: "deny"` +
    # `permissionDecisionReason` on PreToolUse (where `permissionDecision:
    # "allow"` is valid only alongside `updatedInput`).

    @staticmethod
    def _reply(reply: base.HookReply) -> dict[str, Any]:
        return {"systemMessage": reply.message} if reply.message else {}

    @staticmethod
    def _specific(event: str, context: str | None) -> dict[str, Any]:
        h_out: dict[str, Any] = {"hookEventName": event}
        if context:
            h_out["additionalContext"] = context
        return h_out

    def _stop_reply(self, reply: base.HookReply,
                    blocks: bool) -> dict[str, Any]:
        response = self._reply(reply)
        if blocks:
            response["decision"] = "block"
            response["reason"] = reply.effective_reason
        return response

    def reply_session_start(self, reply: base.HookReply) -> dict[str, Any]:
        """Halt with `continue: false` on deny; attach `additionalContext` if present."""
        response = self._reply(reply)
        if reply.is_deny:
            response["continue"] = False
            response["stopReason"] = reply.effective_reason
        if reply.context:
            response["hookSpecificOutput"] = self._specific(
                "SessionStart", reply.context)
        return response

    def reply_subagent_stop(self, reply: base.HookReply) -> dict[str, Any]:
        """Block with `reason` on deny; omit `hookSpecificOutput`."""
        return self._stop_reply(reply, reply.is_deny)

    def reply_stop(self, reply: base.HookReply) -> dict[str, Any]:
        """Block with `reason` on retry or deny; omit `hookSpecificOutput`."""
        return self._stop_reply(reply, reply.decision in ("retry", "deny"))

    def reply_pre_tool_use(self, reply: base.HookReply) -> dict[str, Any]:
        """Deny with `permissionDecisionReason`, or allow with `updatedInput` / context."""
        response = self._reply(reply)
        h_out: dict[str, Any] = {}
        if reply.is_deny:
            h_out["permissionDecision"] = "deny"
            h_out["permissionDecisionReason"] = reply.effective_reason
        elif reply.updated_args is not None:
            h_out["permissionDecision"] = "allow"
            h_out["updatedInput"] = reply.updated_args
        if reply.context:
            h_out["additionalContext"] = reply.context
        if h_out:
            response["hookSpecificOutput"] = {
                "hookEventName": "PreToolUse",
                **h_out,
            }
        return response

    def reply_post_tool_use(self, reply: base.HookReply) -> dict[str, Any]:
        """Block with `reason` on deny, plus `additionalContext` if present."""
        response = self._reply(reply)
        if reply.is_deny:
            response["decision"] = "block"
            response["reason"] = reply.effective_reason
        if reply.context:
            response["hookSpecificOutput"] = self._specific(
                "PostToolUse", reply.context)
        return response

    def reply_session_end(self, reply: base.HookReply) -> dict[str, Any]:
        """Only the `systemMessage` banner; nothing can be blocked."""
        return self._reply(reply)

    def reply_unknown_event(self, reply: base.HookReply) -> dict[str, Any]:
        """Only the `systemMessage` banner."""
        return self._reply(reply)
