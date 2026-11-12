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
the inbound, dispatch, and outbound sides of a hook call; the per-harness
compile, install and trajectory logic still in the core modules moves here
over time. Ships with the runtime; imports no build tooling.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any

from framework import utils


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

    @classmethod
    def allow(
        cls,
        event: str,
        hook_name: str,
        *,
        context: str | None = None,
    ) -> HookReply:
        """An empty `allow` verdict, optionally carrying `context`."""
        return cls(
            event=event,
            hook_name=hook_name,
            decision="allow",
            reason="",
            followup_message=None,
            message=None,
            context=context,
            updated_args=None,
        )

    @property
    def is_deny(self) -> bool:
        """Whether the hook refused (`decision` is `deny`)."""
        return self.decision == "deny"

    @property
    def effective_reason(self) -> str:
        """The follow-up message, else the reason, else a default naming the hook."""
        return (self.followup_message or self.reason
                or f"Denied by hook {self.hook_name}")


@dataclasses.dataclass(frozen=True)
class DispatchContext:
    """What `before_dispatch` and `after_dispatch` see of a hook invocation."""

    event: str
    canonical_event: str
    normalized_event: dict[str, Any]
    hook_name: str
    package: str
    session_id: str


HANDBACK_ASSUMED_SKIP = (
    "auto mode, but the transcript could not show whether the subagent was "
    "given the hand-back tool; this native stop is taken as a turn boundary "
    "and its result is not checked")


class Harness:
    """One harness's identity, mapping-table access, and runtime hook behaviour."""

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

    def event_mapping(self, canonical_event: str) -> dict[str, Any]:
        """This harness's block of `canonical_event` in `event-mappings.json`."""
        return self.mapping_block(utils.get_event_mappings().get(
            canonical_event, {}).get("harnesses", {}))

    def handback_mapping(self) -> dict[str, Any] | None:
        """The validated `handback` block of `SubagentStop`, or None."""
        handback = self.event_mapping("SubagentStop").get("handback")
        if (isinstance(handback, dict) and handback.get("event_name")
                and handback.get("matcher")):
            return handback
        return None

    # Runtime: inbound.

    # Whether a SubagentStop without a subagent name is logged to stderr.
    # True when nameless stops come from internal helper agents; False when
    # root-session turns share the same native Stop event and fire routinely.
    report_nameless_stop: bool = True

    def parent_session_id(self, session_id: str) -> str | None:
        """The parent session ID of a subagent session, if the harness links them."""
        return None

    def canonical_event(self, event: str, raw_payload: dict[str, Any]) -> str:
        """The canonical event name for inbound `event` and `raw_payload`."""
        del raw_payload
        event_mappings = utils.get_event_mappings()
        if event in event_mappings:
            return event
        for canonical, entry in event_mappings.items():
            if event == self.mapping_block(entry.get("harnesses",
                                                     {})).get("event_name"):
                return canonical
        return event

    def resolve_subagent(self,
                         session_id: str,
                         transcript_path: str = "") -> str | None:
        """The subagent name for a session when the payload omits it, or None."""
        del session_id, transcript_path
        return None

    def complete_subagent_stop(
        self,
        subagent: str,
        output: Any,
        *,
        session_id: str = "",
        transcript_path: str = "",
    ) -> tuple[str, Any]:
        """Fill in subagent metadata absent from the raw SubagentStop payload."""
        del session_id, transcript_path
        return subagent, output

    def subagent_was_cut_short(self, event_payload: dict[str, Any]) -> bool:
        """Whether `event_payload` describes a subagent halted before finishing."""
        del event_payload
        return False

    def handback_skip_reason(self, event_payload: dict[str,
                                                       Any]) -> str | None:
        """Skip reason when a native SubagentStop is covered by a hand-back tool."""
        del event_payload
        return None

    def is_session_over(self, event_payload: dict[str, Any]) -> bool:
        """Whether `event_payload` marks the actual end of the root session."""
        return event_payload.get("event") == "SessionEnd"

    # Runtime: dispatch lifecycle.

    def before_dispatch(self, ctx: DispatchContext) -> HookReply | None:
        """A canonical reply to render without running the hook, or None."""
        del ctx
        return None

    def after_dispatch(self, ctx: DispatchContext,
                       canonical_result: dict[str, Any]) -> None:
        """Record any per-turn state after the hook action has produced its verdict."""
        del ctx, canonical_result

    # Runtime: outbound.

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
        """The reply before a tool runs: allow or refuse the call.

        May forward `updated_args` (when set) as the call's rewritten
        arguments, where the harness supports it.
        """
        raise NotImplementedError(f"{self.name}: PreToolUse")

    def reply_post_tool_use(self, reply: HookReply) -> dict[str, Any]:
        """The reply after a tool ran: flag a refusal or add `context`."""
        raise NotImplementedError(f"{self.name}: PostToolUse")

    def reply_subagent_stop(self, reply: HookReply) -> dict[str, Any]:
        """The reply as a subagent finishes: let it stop or send it back."""
        raise NotImplementedError(f"{self.name}: SubagentStop")

    def reply_session_start(self, reply: HookReply) -> dict[str, Any]:
        """The reply as a session starts: inject `context` or refuse it."""
        raise NotImplementedError(f"{self.name}: SessionStart")

    def reply_stop(self, reply: HookReply) -> dict[str, Any]:
        """The reply as the agent stops: allow it, or block on retry or deny."""
        raise NotImplementedError(f"{self.name}: Stop")

    def reply_session_end(self, reply: HookReply) -> dict[str, Any]:
        """The reply as a session ends."""
        raise NotImplementedError(f"{self.name}: SessionEnd")

    def reply_unknown_event(self, reply: HookReply) -> dict[str, Any]:
        """The reply to an event `render_reply` does not name.

        Reachable: a native event name the mapping tables do not know
        passes through canonicalisation unchanged. Each harness answers
        as it did before the adaptors; refusing instead is a behaviour
        change left for its own PR.
        """
        raise NotImplementedError(f"{self.name}: {reply.event}")
