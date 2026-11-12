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

import glob
import json
import pathlib
import uuid
from typing import Any

from framework import jsonio, paths, utils
from framework.harnesses import base

# The member whose mapping block the family inherits.
GOOGLE_HARNESS_MAPPING_KEY = "antigravity"

# Ceiling on spooled context entries per session, oldest evicted first.
# Every live entry is replayed into emulated context on each invocation until
# pushed out by newer entries, so this bounds the per-call prompt overhead.
MAX_PENDING_CONTEXT_ENTRIES = 8

# Termination reasons that mean a subagent loop finished on its own terms
# rather than being halted, cancelled, or cut short by an error or budget cap.
_COMPLETED_SUBAGENT_TERMINATION_REASONS = frozenset({
    "no_tool_call",
    "terminal_step_type",
    "terminal_custom_hook",
})

# Termination reasons that mean a root session actually ended (as opposed to
# a per-turn Stop such as `no_tool_call`, `user_canceled`, or `error`, where
# the conversation continues and still needs its submitted jobs).
_COMPLETED_SESSION_TERMINATION_REASONS = frozenset({
    "terminal_step_type",
    "terminal_custom_hook",
})


def _termination_reason_in(reason: Any, allowed: frozenset[str]) -> bool:
    text = str(reason or "").strip().lower()
    return text.removeprefix("termination_reason_") in allowed


def extract_last_transcript_message(transcript_path: str) -> str | None:
    """Return the latest MODEL message content from a JSONL transcript, or None."""
    if not transcript_path:
        return None
    p = pathlib.Path(transcript_path)
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except ValueError:
                continue
            if (isinstance(data, dict) and data.get("source") == "MODEL"
                    and data.get("content")):
                return data["content"]
    except OSError:
        pass
    return None


def get_hook_context_file(session_id: str, hook_name: str) -> pathlib.Path:
    """Return path to a hook's cached SessionStart context for a session."""
    contexts_dir = (paths.get_sessions_root() /
                    f"{utils.safe_path_component(session_id)}.contexts")
    return contexts_dir / f"{utils.safe_path_component(hook_name)}.json"


def record_session_hook_context(
    session_id: str,
    hook_name: str,
    context_text: str,
    package: str = "",
) -> None:
    """Record cached hook context under ~/.tpu-agent-suite/sessions/<id>.contexts/."""
    if not session_id or not hook_name or not context_text:
        return
    jsonio.write_json(
        get_hook_context_file(session_id, hook_name), {
            "context": context_text,
            "hook_name": hook_name,
            "package": package,
            "recorded_at": utils.utc_now_iso(),
        })


def resolve_session_hook_context(
    session_id: str | None,
    hook_name: str,
) -> str | None:
    """Resolve cached hook context text for a session and hook name."""
    if not session_id or not hook_name:
        return None
    context_text = jsonio.load_json_or_empty(
        get_hook_context_file(session_id, hook_name)).get("context")
    return str(context_text) if context_text else None


def get_pending_context_dir(session_id: str) -> pathlib.Path:
    """Return the spool directory for hook context awaiting delivery.

    PostToolUse responses cannot carry context on Google harnesses; hooks spool
    context here and PreInvocation delivers it on subsequent turns.
    """
    return (paths.get_sessions_root() /
            f"{utils.safe_path_component(session_id)}.pending")


def record_pending_hook_context(
    session_id: str,
    hook_name: str,
    context_text: str,
    package: str = "",
) -> None:
    """Spool hook context into a per-entry file for delivery on a later event."""
    if not session_id or not context_text:
        return
    pending_dir = get_pending_context_dir(session_id)
    pending_dir.mkdir(parents=True, exist_ok=True)

    recorded_at = utils.utc_now_iso()
    entry_name = f"{recorded_at}-{uuid.uuid4().hex[:8]}.json"
    jsonio.write_json(
        pending_dir / utils.safe_path_component(entry_name), {
            "context": context_text,
            "hook_name": hook_name,
            "package": package,
            "recorded_at": recorded_at,
        })

    # Evict oldest entries past MAX_PENDING_CONTEXT_ENTRIES.
    existing = sorted(pending_dir.glob("*.json"))
    overflow = len(existing) - MAX_PENDING_CONTEXT_ENTRIES
    if overflow > 0:
        for stale in existing[:overflow]:
            stale.unlink(missing_ok=True)


def replay_pending_hook_context(session_id: str | None) -> str:
    """Return spooled context for `session_id` (without falling back to a parent session)."""
    if not session_id:
        return ""
    pending_dir = get_pending_context_dir(session_id)
    if not pending_dir.is_dir():
        return ""

    texts: list[str] = []
    for entry_file in sorted(pending_dir.glob("*.json")):
        entry = jsonio.load_json_or_empty(entry_file)
        context_text = entry.get("context")
        if not context_text:
            entry_file.unlink(missing_ok=True)
            continue
        texts.append(str(context_text))
    return "\n\n".join(texts)


class GoogleHarness(base.Harness):
    """Shared behaviour of the Google harness family."""

    config_dir = ".agents"
    mapping_key = GOOGLE_HARNESS_MAPPING_KEY
    brain_dir: str  # conversation brains live at `~/.gemini/<brain_dir>/brain`

    # Runtime: inbound.
    #
    # Stop, SessionEnd, and SubagentStop all ride the native `Stop` event, and
    # `SessionStart` rides per-turn `PreInvocation`. Brain descriptors under
    # `~/.gemini/<brain_dir>/brain` link subagent conversations to their parent
    # and carry subagent names when the payload omits them.

    # Every root-session turn emits a nameless Stop, so logging them floods
    # stderr.
    report_nameless_stop = False

    def _subagent_descriptor(self,
                             conversation_id: str) -> pathlib.Path | None:
        """The brain descriptor file for `conversation_id`, or None.

        Descriptors live at
        `~/.gemini/<brain_dir>/brain/<parent_cid>/.system_generated/subagents/<cid>.json`.
        """
        if (not conversation_id or "/" in conversation_id
                or "\\" in conversation_id or ".." in conversation_id
                or conversation_id == "."):
            return None
        brain_root = pathlib.Path.home() / ".gemini" / self.brain_dir / "brain"
        if not brain_root.is_dir():
            return None
        escaped = glob.escape(conversation_id)
        matches = list(
            brain_root.glob(f"*/.system_generated/subagents/{escaped}.json"))
        return matches[0] if matches else None

    def parent_session_id(self, session_id: str) -> str | None:
        """The parent conversation ID: the brain holding the descriptor."""
        descriptor = self._subagent_descriptor(session_id)
        return descriptor.parents[2].name if descriptor else None

    def canonical_event(self, event: str, raw_payload: dict[str, Any]) -> str:
        """`SubagentStop` for a Stop or SessionEnd a subagent fired.

        A subagent is recognized by a subagent name in the payload, a
        transcript under a subagent worktree, or a brain descriptor for the
        session. Other events map as in the base adaptor.
        """
        if event in ("Stop", "SessionEnd"):
            subagent_fields = self.event_mapping("SubagentStop").get(
                "fields", {})
            container: dict[str, Any] = {}
            for key in subagent_fields.get("container", []):
                value = raw_payload.get(key)
                if isinstance(value, dict):
                    container = value
                    break
            if utils.extract_first_field(raw_payload,
                                         subagent_fields.get("subagent", []),
                                         container=container):
                return "SubagentStop"
            session_id = str(
                utils.extract_first_field(raw_payload,
                                          subagent_fields.get(
                                              "session_id", []),
                                          container=container) or "")
            transcript_path = str(
                utils.extract_first_field(raw_payload,
                                          subagent_fields.get(
                                              "transcript_path", []),
                                          container=container) or "")
            if ("/worktrees/subagent-" in transcript_path
                    or self._subagent_descriptor(session_id) is not None):
                return "SubagentStop"
            return event
        return super().canonical_event(event, raw_payload)

    def resolve_subagent(self,
                         session_id: str,
                         transcript_path: str = "") -> str | None:
        """The subagent name for a session, or None if it can't be found.

        Returns "subagent" for a transcript under a subagent worktree, else the
        `typeName` or `role` from the session's brain descriptor.
        """
        if transcript_path and "/worktrees/subagent-" in transcript_path:
            return "subagent"
        descriptor = self._subagent_descriptor(session_id)
        if descriptor is not None:
            sub_desc = jsonio.load_json_or_empty(descriptor).get(
                "subagentDescriptor")
            if isinstance(sub_desc, dict):
                name = sub_desc.get("typeName") or sub_desc.get("role")
                if name:
                    return str(name)
        return None

    def complete_subagent_stop(
        self,
        subagent: str,
        output: Any,
        *,
        session_id: str = "",
        transcript_path: str = "",
    ) -> tuple[str, Any]:
        """Resolve a missing name; take missing output from the transcript."""
        if not subagent and (session_id or transcript_path):
            subagent = self.resolve_subagent(session_id, transcript_path) or ""
        if output is None and transcript_path:
            output = extract_last_transcript_message(transcript_path)
        return subagent, output

    def subagent_was_cut_short(self, event_payload: dict[str, Any]) -> bool:
        """Whether the stop has a `stop_reason` that is not a completion."""
        args = event_payload.get("args")
        reason = args.get("stop_reason") if isinstance(args, dict) else None
        return bool(reason) and not _termination_reason_in(
            reason, _COMPLETED_SUBAGENT_TERMINATION_REASONS)

    def is_session_over(self, event_payload: dict[str, Any]) -> bool:
        """Whether a SessionEnd is fully idle with a completed reason."""
        if event_payload.get("event") != "SessionEnd":
            return False
        term = event_payload.get("termination", {})
        return (term.get("fully_idle") is True and _termination_reason_in(
            term.get("reason"), _COMPLETED_SESSION_TERMINATION_REASONS))

    # Runtime: dispatch lifecycle.
    #
    # Root `Stop` and `SessionEnd` hooks ignore subagent stops riding the
    # shared `Stop` event. `PreInvocation` fires before every model call:
    # Turn 1 of the root session runs the `SessionStart` hook and caches its
    # context; later invocations and subagent calls replay cached and spooled
    # `PostToolUse` context.

    def before_dispatch(self,
                        ctx: base.DispatchContext) -> base.HookReply | None:
        """Allow subagent stops on root hooks; replay later SessionStarts."""
        if (ctx.event in ("Stop", "SessionEnd")
                and ctx.canonical_event == "SubagentStop"):
            return base.HookReply.allow(ctx.event, ctx.hook_name)

        if ctx.canonical_event == "SessionStart":
            args = ctx.normalized_event.get("args", {})
            if "initial_num_steps" not in args or "invocation_num" not in args:
                raise utils.TpuAgentSuiteError(
                    f"harness '{self.name}' requires 'initial_num_steps' and "
                    "'invocation_num' to determine SessionStart phase")
            is_subagent = bool(ctx.normalized_event.get("subagent"))
            if (is_subagent or args["invocation_num"] > 0
                    or args["initial_num_steps"] > 1):
                cached_ctx = ("" if is_subagent else
                              (resolve_session_hook_context(
                                  ctx.session_id, ctx.hook_name) or ""))
                pending_ctx = replay_pending_hook_context(ctx.session_id)
                merged = "\n\n".join(t for t in (cached_ctx, pending_ctx) if t)
                return base.HookReply.allow("SessionStart",
                                            ctx.hook_name,
                                            context=merged or None)
        return None

    def after_dispatch(self, ctx: base.DispatchContext,
                       canonical_result: dict[str, Any]) -> None:
        """Cache SessionStart context and spool PostToolUse context."""
        if not ctx.session_id:
            return
        if ctx.canonical_event == "SessionStart":
            ctx_text = (canonical_result.get("replay_context")
                        or canonical_result.get("context"))
            if ctx_text:
                record_session_hook_context(
                    ctx.session_id,
                    ctx.hook_name,
                    str(ctx_text),
                    ctx.package,
                )
        elif ctx.canonical_event == "PostToolUse":
            deferred_text = canonical_result.get("context")
            if deferred_text:
                record_pending_hook_context(
                    ctx.session_id,
                    ctx.hook_name,
                    str(deferred_text),
                    ctx.package,
                )

    # Runtime: outbound.
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
        """`continue` on retry, `deny` on deny, otherwise `stop`."""
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
        """The Stop reply, since the family emulates SessionEnd on Stop."""
        return self.reply_stop(reply)

    def reply_session_start(self, reply: base.HookReply) -> dict[str, Any]:
        """`ephemeralMessage` steps for the context and any refusal."""
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
        """`continue` with the reason on deny; otherwise `allow`."""
        if reply.is_deny:
            return {"decision": "continue", "reason": reply.effective_reason}
        return {"decision": "allow"}

    def reply_pre_tool_use(self, reply: base.HookReply) -> dict[str, Any]:
        """Deny, or allow with rewritten arguments as `overwrite`."""
        if reply.is_deny:
            return self._refusal(reply)
        res: dict[str, Any] = {"decision": "allow"}
        if reply.updated_args is not None:
            res["overwrite"] = reply.updated_args
        return res

    def reply_post_tool_use(self, reply: base.HookReply) -> dict[str, Any]:
        """Deny on deny; otherwise reply with nothing."""
        return self._refuse_or_empty(reply)

    def reply_unknown_event(self, reply: base.HookReply) -> dict[str, Any]:
        """Deny on deny; otherwise reply with nothing."""
        return self._refuse_or_empty(reply)
