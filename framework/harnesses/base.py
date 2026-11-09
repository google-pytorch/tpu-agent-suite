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
"""The interface every harness adaptor implements.

An adaptor holds what is particular to one harness: today its identity,
which block of `tool-mappings.json` and `event-mappings.json` it reads, and
the native shape of a hook reply; the per-harness compile, install and
runtime logic still in the core modules moves here over time. Ships with
the runtime; imports no build tooling.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any


@dataclasses.dataclass(frozen=True)
class HookReply:
    """A hook's canonical verdict, ready for an adaptor to render.

    The fields are what the hook returned, except that the runtime has
    canonicalised `event` and filled `context` from a bare `run_id`.
    """

    event: str
    hook_name: str
    decision: str
    reason: str
    followup_message: str | None
    message: str | None
    context: str | None
    updated_args: Any

    @property
    def is_deny(self) -> bool:
        return self.decision == "deny"

    @property
    def effective_reason(self) -> str:
        """The follow-up message, else the reason, else a default naming the hook."""
        return (self.followup_message or self.reason
                or f"Denied by hook {self.hook_name}")


class Harness:
    """One harness's identity, mapping-table access, and hook reply shape."""

    name: str
    display_name: str
    config_dir: str  # e.g. `.claude`, under the workspace and under `~`
    # The table block read when there is none under the harness's own name.
    # A family base names the member that owns the shared block; otherwise
    # None, and the harness reads its own block only.
    mapping_key: str | None = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.name!r})"

    def mapping_block(self, harnesses: dict[str, Any]) -> dict[str, Any]:
        """This harness's block of one entry's `harnesses` object, or `{}`.

        The block under the harness's own name wins; otherwise the one under
        `mapping_key`, which is how a family member inherits.
        """
        block = harnesses.get(self.name)
        if block is None and self.mapping_key:
            block = harnesses.get(self.mapping_key)
        return block or {}

    # --- runtime: outbound -------------------------------------------------

    # Whether the hook's `message` is announced on stderr before the reply.
    # True for harnesses whose reply has no field for it; Claude Code carries
    # it in the reply instead.
    message_on_stderr: bool = False

    def render_reply(self, reply: HookReply) -> dict[str, Any]:
        """Render a hook's canonical verdict in this harness's native shape.

        Announces `message` if the harness wants it on stderr, then
        dispatches on the event to one `reply_*` method. Adaptors override
        those methods, not this one.
        """
        if self.message_on_stderr and reply.message:
            print(f"[{reply.hook_name}] {reply.message}", file=sys.stderr)
        if reply.event == "PreToolUse":
            return self.reply_pre_tool_use(reply)
        if reply.event == "PostToolUse":
            return self.reply_post_tool_use(reply)
        if reply.event == "SubagentStop":
            return self.reply_subagent_stop(reply)
        if reply.event == "SessionStart":
            return self.reply_session_start(reply)
        if reply.event == "Stop":
            return self.reply_stop(reply)
        if reply.event == "SessionEnd":
            return self.reply_session_end(reply)
        return self.reply_unknown_event(reply)

    def reply_pre_tool_use(self, reply: HookReply) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name}: PreToolUse")

    def reply_post_tool_use(self, reply: HookReply) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name}: PostToolUse")

    def reply_subagent_stop(self, reply: HookReply) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name}: SubagentStop")

    def reply_session_start(self, reply: HookReply) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name}: SessionStart")

    def reply_stop(self, reply: HookReply) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name}: Stop")

    def reply_session_end(self, reply: HookReply) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name}: SessionEnd")

    def reply_unknown_event(self, reply: HookReply) -> dict[str, Any]:
        """The reply to an event `render_reply` does not name.

        Reachable: a native event name the mapping tables do not know
        passes through canonicalisation unchanged. Each harness answers
        as it did before the adaptors; refusing instead is a behaviour
        change left for its own PR.
        """
        raise NotImplementedError(f"{self.name}: {reply.event}")
