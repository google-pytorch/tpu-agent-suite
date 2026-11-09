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
"""`runtime.py hook-dispatch`: run a hook for one harness event and reply.

Dispatch normalizes the payload (`events`), drops events no hook should
see, runs the hook (`actions`), and translates its result into the reply
the harness expects. It always answers, even when it fails itself.
"""

from __future__ import annotations

import copy
import json
import pathlib
import re
import sys
from typing import Any

from framework import bg_ledger, harnesses, run_state, utils
from framework.hooks import actions, events, stop_filter

Path = pathlib.Path
TpuAgentSuiteError = utils.TpuAgentSuiteError


def translate_canonical_output(
    canonical_result: dict[str, Any],
    harness: str,
    event: str,
    hook_name: str,
) -> dict[str, Any]:
    """Translate canonical hook output into target harness response format.

    Resolves the verdict -- a bare `run_id` becomes context, a native event
    name is canonicalised -- packs it into a `HookReply`, and hands it to
    the harness adaptor, which owns the native shape.
    """
    adaptor = harnesses.get(harness)
    if adaptor is None:
        raise TpuAgentSuiteError(
            f"unsupported harness for hook translation: '{harness}'")

    decision = canonical_result.get("decision", "allow")
    reason = canonical_result.get("reason", "")
    context_text = canonical_result.get("context")
    run_id = canonical_result.get("run_id")
    if run_id and not context_text:
        context_text = run_state.format_run_context(run_id)
    event = harnesses.to_canonical_event(event, harness)

    return adaptor.render_reply(
        harnesses.HookReply(
            event=event,
            hook_name=hook_name,
            decision=decision,
            reason=reason,
            followup_message=canonical_result.get("followup_message"),
            message=canonical_result.get("message"),
            context=context_text,
            updated_args=canonical_result.get("updated_args"),
        ))


def _translate_hook_reply(
    canonical_result: dict[str, Any],
    harness: str,
    canonical_event: str,
    hook_name: str,
    normalized_event: dict[str, Any],
) -> dict[str, Any]:
    """Translate a hook result into the reply expected by the firing native event.

    When a SubagentStop hook fires via a hand-back tool call (e.g. PreToolUse on
    `send_message`), its verdict must be formatted as a permission decision for
    that tool event rather than a native SubagentStop reply.
    """
    if canonical_event == "SubagentStop" and events.is_handback_event(
            normalized_event):
        args = normalized_event.get("args") or {}
        source_event = str(args["handback_event"])
        refused = canonical_result.get("decision") == "deny"
        reply: dict[str, Any] = {"decision": "deny" if refused else "allow"}
        reason = (canonical_result.get("followup_message")
                  or canonical_result.get("reason")
                  or (f"Denied by hook {hook_name}" if refused else None))
        if reason:
            reply["reason"] = reason
        if canonical_result.get("message"):
            reply["message"] = canonical_result["message"]
        # Omit `context` and `updated_args`: a finishing subagent does not need
        # additional context, and `updated_args` would rewrite the hand-back
        # tool's input arguments.
        return translate_canonical_output(reply, harness, source_event,
                                          hook_name)
    return translate_canonical_output(canonical_result, harness,
                                      canonical_event, hook_name)


def dispatch_hook(
    harness: str,
    package: str,
    hook_name: str,
    event: str,
    script_rel: str | None = None,
    command: str | None = None,
    builtin: str | None = None,
    timeout: float | None = utils.DEFAULT_HOOK_TIMEOUT,
    workspace: Path | None = None,
    raw_input_text: str | None = None,
    script_args: list[str] | None = None,
    matcher: str | None = None,
    on_failure: str = utils.DEFAULT_HOOK_ON_FAILURE,
    handback_event: str | None = None,
) -> tuple[dict[str, Any], int]:
    """Run a hook for a harness event and return `(harness_response, exit_code)`.

    Any unhandled error during dispatch is caught and settled according to
    `on_failure` so the harness always receives a well-formed response.
    """
    try:
        return _dispatch_hook_body(harness=harness,
                                   package=package,
                                   hook_name=hook_name,
                                   event=event,
                                   script_rel=script_rel,
                                   command=command,
                                   builtin=builtin,
                                   timeout=timeout,
                                   workspace=workspace,
                                   raw_input_text=raw_input_text,
                                   script_args=script_args,
                                   matcher=matcher,
                                   on_failure=on_failure)
    except Exception as exc:  # noqa: BLE001 - becomes the hook's failure
        return _dispatch_failure(exc,
                                 harness=harness,
                                 hook_name=hook_name,
                                 event=event,
                                 workspace=workspace,
                                 on_failure=on_failure,
                                 handback_event=handback_event)


def _dispatch_failure(exc: Exception,
                      *,
                      harness: str,
                      hook_name: str,
                      event: str,
                      workspace: Path | None,
                      on_failure: str,
                      handback_event: str | None,
                      kind: str = "dispatch") -> tuple[dict[str, Any], int]:
    """Record a dispatch or startup exception and format the policy's fallback reply."""
    try:
        event = harnesses.to_canonical_event(event, harness)
    except Exception:  # noqa: BLE001 - the raw name still labels the record
        pass
    verdict = actions.hook_failure(
        kind,
        f"Dispatch of hook '{hook_name}' failed: {type(exc).__name__}: {exc}",
        hook_name=hook_name,
        on_failure=on_failure,
        event_payload={"event": event},
        workspace=workspace or Path.cwd())
    handback = ({
        "args": {
            "handback": True,
            "handback_event": handback_event
        }
    } if handback_event else {})
    return _translate_hook_reply(verdict, harness, event, hook_name,
                                 handback), 0


class HookDispatchStartupError(Exception):
    """A `hook-dispatch` command line the dispatcher could not parse."""


def hook_dispatch_startup_failure(hints: dict[str, str | None],
                                  reason: str) -> int:
    """Handle a `hook-dispatch` startup/argument failure according to `on_failure`."""
    hook_name = hints.get("hook") or "?"
    harness = hints.get("harness") or ""
    event = hints.get("event") or ""
    on_failure = hints.get("on_failure")
    if on_failure not in utils.HOOK_FAILURE_POLICIES:
        on_failure = utils.DEFAULT_HOOK_ON_FAILURE
    workspace = Path(hints["workspace"]) if hints.get("workspace") else None
    try:
        if (harness in harnesses.supported_harnesses()
                and harnesses.to_canonical_event(
                    event, harness) in utils.get_supported_hook_events()):
            response, code = _dispatch_failure(
                HookDispatchStartupError(reason),
                harness=harness,
                hook_name=hook_name,
                event=event,
                workspace=workspace,
                on_failure=on_failure,
                handback_event=hints.get("handback_event"),
                kind="startup")
            print(json.dumps(response), flush=True)
            return code
    except Exception as exc:  # noqa: BLE001 - the exit code still carries the policy
        reason = f"{reason} (no reply could be shaped: {type(exc).__name__}: {exc})"
    print(
        f"runtime: [{hook_name}] hook failed (startup): {reason}; "
        f"on_failure={on_failure}",
        file=sys.stderr)
    return 0 if on_failure == "allow" else 2


def _dispatch_hook_body(
    harness: str,
    package: str,
    hook_name: str,
    event: str,
    script_rel: str | None = None,
    command: str | None = None,
    builtin: str | None = None,
    timeout: float | None = utils.DEFAULT_HOOK_TIMEOUT,
    workspace: Path | None = None,
    raw_input_text: str | None = None,
    script_args: list[str] | None = None,
    matcher: str | None = None,
    on_failure: str = utils.DEFAULT_HOOK_ON_FAILURE,
) -> tuple[dict[str, Any], int]:
    """`dispatch_hook` itself, before its guard against its own errors."""
    if timeout is None or timeout <= 0:
        timeout = utils.DEFAULT_HOOK_TIMEOUT

    event = harnesses.to_canonical_event(event, harness)

    # 1. Parse raw payload
    if raw_input_text is None:
        try:
            raw_input_text = sys.stdin.read()
        except Exception:
            raw_input_text = "{}"

    try:
        raw_payload = json.loads(
            raw_input_text) if raw_input_text.strip() else {}
    except Exception as exc:
        raw_payload = {"_raw": raw_input_text, "error": str(exc)}

    resolved_workspace = events.resolve_workspace_root(workspace, raw_payload)

    # 2. Inbound normalization
    normalized_event = events.normalize_inbound_event(
        raw_payload=raw_payload,
        harness=harness,
        event=event,
        package=package,
        workspace=resolved_workspace,
    )
    canonical_event = normalized_event.get("event") or event

    # Google harnesses emulate SessionStart, SessionEnd, and SubagentStop via
    # PreInvocation and Stop events.
    if harnesses.is_google_harness(harness):
        # Ignore subagent stops when dispatching a root SessionEnd hook.
        if event == "SessionEnd" and canonical_event == "SubagentStop":
            return translate_canonical_output({"decision": "allow"}, harness,
                                              event, hook_name), 0

        # PreInvocation fires on every model call. Run the SessionStart hook
        # only on the root session's first turn/invocation (`initial_num_steps == 1`
        # and `invocation_num == 0`); on subsequent calls, re-inject cached and
        # pending context instead.
        if canonical_event == "SessionStart":
            args = normalized_event.get("args", {})
            if "initial_num_steps" not in args or "invocation_num" not in args:
                raise TpuAgentSuiteError(
                    f"harness '{harness}' requires 'initial_num_steps' and 'invocation_num' "
                    f"to determine SessionStart phase")
            init_steps = args["initial_num_steps"]
            inv_num = args["invocation_num"]
            session_id = events.event_session_id(normalized_event)

            is_subagent = bool(normalized_event.get("subagent"))
            is_subagent_or_retry = is_subagent or inv_num > 0
            is_turn_2_plus = init_steps > 1

            if is_subagent_or_retry or is_turn_2_plus:
                cached_ctx = ""
                if not is_subagent:
                    cached_ctx = run_state.resolve_session_hook_context(
                        session_id, hook_name)
                pending_ctx = run_state.replay_pending_hook_context(session_id)
                merged = "\n\n".join(t for t in (cached_ctx, pending_ctx) if t)
                canonical: dict[str, Any] = {"decision": "allow"}
                if merged:
                    canonical["context"] = merged
                return translate_canonical_output(canonical, harness,
                                                  canonical_event,
                                                  hook_name), 0

    # Skip SubagentStop events that do not represent a completed subagent
    # delivery (e.g. turn boundaries, halts, root turns, or duplicate fires).
    if canonical_event == "SubagentStop":
        # The skip filter and the idle gate both ask whether a not-idle stop
        # is a pause the harness resumes; ask once.
        stop_args = normalized_event.get("args")
        harness_resumes = (bg_ledger.harness_resumes_agent(
            normalized_event, harness) if isinstance(stop_args, dict)
                           and stop_args.get("fully_idle") is False else None)
        skip_reason = stop_filter.subagent_stop_skip_reason(
            normalized_event, harness, harness_resumes=harness_resumes)
        handback_skip = skip_reason in (stop_filter.HANDBACK_SKIP,
                                        stop_filter.HANDBACK_ASSUMED_SKIP)
        if skip_reason and not handback_skip:
            if (skip_reason == stop_filter.NO_SUBAGENT_NAME_SKIP
                    and not harnesses.is_google_harness(harness)):
                print(f"[{hook_name}] SubagentStop skipped: {skip_reason}",
                      file=sys.stderr)
            return _translate_hook_reply({"decision": "allow"}, harness,
                                         canonical_event, hook_name,
                                         normalized_event), 0
        # A stop with the agent's own work live, on a run the harness will
        # not resume, is refused here before any body runs, so every hook on
        # the event gives the same reply. It comes before the hand-back skip:
        # which road the result takes says nothing about whether the work is
        # done, and a pass here lets the harness's own re-prompt drive the
        # subagent into a hand-back before its child has reported.
        idle_verdict = bg_ledger.idle_gate_verdict(
            normalized_event, harness, harness_resumes=harness_resumes)
        if idle_verdict:
            print(
                f"[{hook_name}] SubagentStop refused: subagent "
                f"'{normalized_event.get('subagent')}' stopped with background "
                "work live and the harness would not resume it",
                file=sys.stderr)
            return _translate_hook_reply(idle_verdict, harness,
                                         canonical_event, hook_name,
                                         normalized_event), 0
        if handback_skip:
            if skip_reason == stop_filter.HANDBACK_ASSUMED_SKIP:
                print(f"[{hook_name}] SubagentStop skipped: {skip_reason}",
                      file=sys.stderr)
            return _translate_hook_reply({"decision": "allow"}, harness,
                                         canonical_event, hook_name,
                                         normalized_event), 0

    # 3. Matcher filtering (for harnesses that do not filter all events natively)
    if matcher and matcher != "*":
        target = ""
        if canonical_event == "SubagentStop":
            target = normalized_event.get("subagent", "")
        elif canonical_event in ("PreToolUse", "PostToolUse"):
            target = normalized_event.get("tool", "")

        matched = False
        if target:
            patterns = [p.strip() for p in matcher.split("|") if p.strip()]
            for p in patterns:
                if p == target or p == "*":
                    matched = True
                    break
                try:
                    if re.search(rf"^{p}$", target):
                        matched = True
                        break
                except re.error:
                    pass
        if not matched:
            return _translate_hook_reply({"decision": "allow"}, harness,
                                         canonical_event, hook_name,
                                         normalized_event), 0

    # 4. Execute hook action (with multi-subagent batch handling)

    # If the harness invokes multiple subagents in a single call (e.g. the Google harnesses'
    # Subagents array), run the hook for each subagent. If some are denied, filter them out via
    # PreToolUse 'overwrite'.
    subs_list = []
    subagents_key = None
    if normalized_event.get("tool") == "delegate":
        h_args = harnesses.harness_mapping(
            utils.get_tool_mappings().get("delegate", {}).get("harnesses", {}),
            harness).get("args", {})
        raw_subs_key = h_args.get("subagents")
        if raw_subs_key and isinstance(raw_subs_key, str):
            subagents_key = raw_subs_key
            raw_subs = normalized_event.get("args", {}).get(raw_subs_key)
            if raw_subs is None:
                raw_subs = events.extract_arg_by_path(raw_payload,
                                                      raw_subs_key)
            if isinstance(raw_subs, str):
                try:
                    raw_subs = json.loads(raw_subs)
                except Exception:
                    raw_subs = []
            if isinstance(raw_subs, list) and len(raw_subs) > 1:
                subs_list = raw_subs

    if subs_list and event == "PreToolUse" and subagents_key:
        allowed_subs = []
        deny_reasons = []
        h_args = harnesses.harness_mapping(
            utils.get_tool_mappings().get("delegate", {}).get("harnesses", {}),
            harness).get("args", {})
        sub_candidates = h_args.get("subagent", [])
        if isinstance(sub_candidates, str):
            sub_candidates = [sub_candidates]
        prompt_candidates = h_args.get("prompt", [])
        if isinstance(prompt_candidates, str):
            prompt_candidates = [prompt_candidates]

        b_handler = None
        if builtin:
            b_handler = actions.BUILTIN_HOOK_ACTIONS.get(builtin)
            if not b_handler:
                raise TpuAgentSuiteError(
                    f"unsupported builtin hook action: '{builtin}', expected one of {sorted(actions.BUILTIN_HOOK_ACTIONS)}"
                )

        # Each subagent gets its own copy of the event, nested dicts included:
        # the runs below are concurrent and must not share mutable state.
        sub_runs = []
        for sub_entry in subs_list:
            sub_event = copy.deepcopy(normalized_event)

            matched_sub_k = None
            for cand in sub_candidates:
                if cand in sub_entry:
                    matched_sub_k = cand
                    break
                short_k = cand.split(".")[-1]
                if short_k in sub_entry:
                    matched_sub_k = short_k
                    break

            matched_prompt_k = None
            for cand in prompt_candidates:
                if cand in sub_entry:
                    matched_prompt_k = cand
                    break
                short_k = cand.split(".")[-1]
                if short_k in sub_entry:
                    matched_prompt_k = short_k
                    break

            if matched_sub_k:
                subagent_name = str(sub_entry[matched_sub_k])
                # Set both the canonical event envelope metadata and the tool call arg bag.
                sub_event["subagent"] = subagent_name
                sub_event["args"]["subagent"] = subagent_name
            if matched_prompt_k:
                sub_event["args"]["prompt"] = str(sub_entry[matched_prompt_k])
            sub_runs.append(
                (sub_entry, sub_event, matched_sub_k, matched_prompt_k))

        sub_results = actions.run_hook_action_per_subagent(
            [sub_event for _, sub_event, _, _ in sub_runs],
            hook_name=hook_name,
            workspace=resolved_workspace,
            package=package,
            timeout=timeout,
            on_failure=on_failure,
            builtin_handler=b_handler,
            command=command,
            script_rel=script_rel,
            script_args=script_args,
        )

        for (sub_entry, sub_event, matched_sub_k,
             matched_prompt_k), sub_res in zip(sub_runs, sub_results):
            if sub_res.get("decision") == "deny":
                label = sub_event["args"].get("subagent") or "subagent"
                deny_reasons.append(
                    f"{label}: {sub_res.get('reason', 'denied')}")
            else:
                # If hook returned updated_args, reflect updates back to the subagent entry
                up_args = sub_res.get("updated_args")
                if isinstance(up_args, dict):
                    updated_entry = dict(sub_entry)
                    if matched_sub_k and "subagent" in up_args:
                        updated_entry[matched_sub_k] = up_args["subagent"]
                    if matched_prompt_k and "prompt" in up_args:
                        updated_entry[matched_prompt_k] = up_args["prompt"]
                    allowed_subs.append(updated_entry)
                else:
                    allowed_subs.append(sub_entry)

        if not allowed_subs:
            canonical_result = {
                "decision": "deny",
                "reason": "; ".join(deny_reasons)
                or "All subagents were denied",
            }
        elif deny_reasons:
            # Partial allowance: allow tool call but overwrite with only allowed subagents
            canonical_result = {
                "decision": "allow",
                "reason":
                f"Filtered out denied subagents: {'; '.join(deny_reasons)}",
                "updated_args": {
                    subagents_key: allowed_subs
                },
            }
        else:
            canonical_result = {"decision": "allow"}
            if any(s != o for s, o in zip(allowed_subs, subs_list)):
                canonical_result["updated_args"] = {
                    subagents_key: allowed_subs
                }
    elif builtin:
        handler = actions.BUILTIN_HOOK_ACTIONS.get(builtin)
        if not handler:
            raise TpuAgentSuiteError(
                f"unsupported builtin hook action: '{builtin}', expected one of {sorted(actions.BUILTIN_HOOK_ACTIONS)}"
            )
        canonical_result = actions.run_builtin_hook_action(
            handler,
            package=package,
            workspace=resolved_workspace,
            event_payload=normalized_event,
            hook_name=hook_name,
            timeout=timeout,
            on_failure=on_failure,
        )
    else:
        canonical_result = actions.execute_single_hook_action(
            event_payload=normalized_event,
            hook_name=hook_name,
            workspace=resolved_workspace,
            package=package,
            command=command,
            script_rel=script_rel,
            timeout=timeout,
            script_args=script_args,
            on_failure=on_failure,
        )

    if harnesses.is_google_harness(harness):
        session_id = events.event_session_id(normalized_event)

        # Cache Turn 1 SessionStart context for re-injection on subsequent PreInvocation calls.
        if canonical_event == "SessionStart":
            ctx_text = (canonical_result.get("replay_context")
                        or canonical_result.get("context"))
            run_id = canonical_result.get("run_id")
            if run_id and not ctx_text:
                ctx_text = run_state.format_run_context(run_id)
            if session_id and ctx_text:
                run_state.record_session_hook_context(
                    session_id=session_id,
                    hook_name=hook_name,
                    context_text=ctx_text,
                    package=package,
                )

        # Spool PostToolUse context for delivery on the next PreInvocation call,
        # since Google harness PostToolUse responses cannot carry context directly.
        elif canonical_event == "PostToolUse":
            deferred_text = canonical_result.get("context")
            if deferred_text and session_id:
                run_state.record_pending_hook_context(
                    session_id=session_id,
                    hook_name=hook_name,
                    context_text=str(deferred_text),
                    package=package,
                )

    # 4. Translate to target harness format
    harness_response = _translate_hook_reply(
        canonical_result,
        harness,
        canonical_event,
        hook_name,
        normalized_event,
    )

    return harness_response, 0
