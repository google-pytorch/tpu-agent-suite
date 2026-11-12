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
"""Turns a harness's hook payload into the canonical event hooks read.

Every harness names its fields differently, and event-mappings.json says where
each one keeps them. `normalize_inbound_event` is the entry point.
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import Any

from framework import bg_ledger, harnesses, paths, run_state, utils

Path = pathlib.Path
TpuAgentSuiteError = utils.TpuAgentSuiteError
extract_canonical_field = utils.extract_first_field
extract_arg_by_path = utils.extract_by_path


def resolve_workspace_root(
    workspace: Path | str | None = None,
    raw_payload: dict[str, Any] | None = None,
) -> Path:
    """Determine the active workspace root directory across harnesses and contexts."""
    if workspace:
        return Path(workspace).resolve()

    # 1. Extract active workspace path directly from inbound harness payload if
    #    available
    if raw_payload and isinstance(raw_payload, dict):
        # Dynamically aggregate declared cwd/workspace candidate fields from
        # event-mappings.json
        candidate_fields = ["workspace"]
        try:
            for h_info in utils.get_event_mappings().get(
                    "SessionStart", {}).get("harnesses", {}).values():
                for field in h_info.get("fields", {}).get("cwd", []):
                    if field not in candidate_fields:
                        candidate_fields.append(field)
        except Exception:
            pass

        for key in candidate_fields:
            val = raw_payload.get(key)
            if isinstance(val, list) and val and isinstance(val[0], str):
                val = val[0]
            if isinstance(val, str) and val.strip():
                p = Path(val).resolve()
                if p.is_dir():
                    return p

    # 2. Fallback to process cwd and parent directory marker search
    cur = Path.cwd().resolve()
    if cur.name in harnesses.HARNESS_CONFIG_DIRS.values():
        return cur.parent
    # Find the nearest ancestor with an active-run marker.
    for p in [cur, *cur.parents]:
        if (p / paths.TPU_AGENT_SUITE_DIRNAME /
                paths.ACTIVE_RUN_FILE).is_file():
            return p
    return cur


def _build_event_context(
    workspace: Path,
    package: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Construct canonical context dictionary enriched with active session and run IDs."""
    context: dict[str, Any] = {
        "workspace": str(workspace),
        "package": package,
    }
    if session_id:
        context["session_id"] = session_id
        active_run_id = run_state.resolve_session_run_id(session_id)
        if active_run_id:
            context["run_id"] = active_run_id
    return context


def event_session_id(event_payload: dict[str, Any]) -> str:
    """Extract canonical session_id from event context."""
    return str(event_payload.get("context", {}).get("session_id") or "")


def _normalize_subagent_stop_event(
    raw_payload: dict[str, Any],
    harness: str,
    package: str,
    workspace: Path,
) -> dict[str, Any]:
    """Normalize a harness subagent stop payload into a canonical SubagentStop event."""
    event_mappings = utils.get_event_mappings()
    harness_config = harnesses.harness_mapping(
        event_mappings.get("SubagentStop", {}).get("harnesses", {}), harness)
    if not harness_config:
        raise TpuAgentSuiteError(
            f"unsupported harness for subagent stop: '{harness}'")

    fields = harness_config.get("fields", {})

    # Unwrap nested argument containers (e.g. `tool_input` on Claude Code).
    container_dict: dict[str, Any] = {}
    for c_key in fields.get("container", []):
        val = raw_payload.get(c_key)
        if isinstance(val, dict):
            container_dict = val
            break

    subagent = str(
        extract_canonical_field(
            raw_payload, fields.get("subagent", []), container=container_dict)
        or "")
    subagent_output = extract_canonical_field(raw_payload,
                                              fields.get("output", []),
                                              container=container_dict)
    session_id_val = str(
        extract_canonical_field(raw_payload,
                                fields.get("session_id", []),
                                container=container_dict) or "")

    adaptor = harnesses.get(harness)
    if adaptor is not None:
        tpath = str(
            extract_canonical_field(raw_payload,
                                    fields.get("transcript_path", []),
                                    container=container_dict) or "")
        subagent, subagent_output = adaptor.complete_subagent_stop(
            subagent,
            subagent_output,
            session_id=session_id_val,
            transcript_path=tpath)

    normalized_args: dict[str, Any] = {"subagent": subagent}
    if isinstance(raw_payload.get("args"), dict):
        normalized_args.update(raw_payload["args"])
    if container_dict:
        normalized_args.update(container_dict)
    for key, value in _passthrough_args(raw_payload,
                                        fields,
                                        skip=fields.get("container",
                                                        [])).items():
        normalized_args.setdefault(key, value)

    stop_reason = extract_canonical_field(raw_payload,
                                          fields.get("stop_reason", []),
                                          container=container_dict)
    if stop_reason is not None:
        normalized_args["stop_reason"] = str(stop_reason)

    # Distinguishes a finished subagent from a mid-task turn boundary (e.g.
    # Antigravity's `fullyIdle`).
    fully_idle = _extract_idle_flag(raw_payload,
                                    fields,
                                    harness,
                                    "SubagentStop",
                                    container=container_dict)
    if fully_idle is not None:
        normalized_args["fully_idle"] = fully_idle
        normalized_args["fully_idle_source"] = "harness"
    elif not fields.get("fully_idle"):
        # No idle field: derive it (`bg_ledger.outstanding_work`).
        # The source is kept because a `False` means different things: a
        # harness that reports idleness also resumes the paused agent; a
        # derived flag says nothing about that.
        fully_idle, outstanding = bg_ledger.outstanding_work(
            raw_payload, harness, "SubagentStop", session_id_val)
        if fully_idle is not None:
            normalized_args["fully_idle"] = fully_idle
            normalized_args["fully_idle_source"] = "ledger"
            if outstanding:
                normalized_args["outstanding_work"] = outstanding

    return _subagent_stop_event(subagent, subagent_output, normalized_args,
                                session_id_val, raw_payload, package,
                                workspace)


def _passthrough_args(
        raw_payload: dict[str, Any],
        fields: dict[str, Any],
        skip: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return raw payload keys to pass through in `args`, excluding canonicalized fields."""
    skip_keys = {"args", "event", "hook_event_name", *skip}
    skip_keys.update(spec.split(".")[0] for spec in fields.get("output", []))
    return {k: v for k, v in raw_payload.items() if k not in skip_keys}


def _subagent_stop_event(
    subagent: str,
    output: Any,
    args: dict[str, Any],
    session_id: str,
    raw_payload: dict[str, Any],
    package: str,
    workspace: Path,
) -> dict[str, Any]:
    return {
        "event": "SubagentStop",
        "subagent": subagent,
        "tool": "delegate",
        "args": args,
        "output": output,
        "context": _build_event_context(workspace, package, session_id),
        "raw_input": raw_payload,
    }


def _match_handback(
    raw_payload: dict[str, Any],
    harness: str,
) -> dict[str, Any] | None:
    """Return the subagent hand-back mapping if `raw_payload` matches its event and tool."""
    handback_mapping = harnesses.subagent_handback_mapping(harness)
    if handback_mapping is None or not isinstance(raw_payload, dict):
        return None
    native_event = raw_payload.get("hook_event_name")
    if native_event and native_event != handback_mapping["event_name"]:
        return None
    tool_event = harnesses.to_canonical_event(handback_mapping["event_name"],
                                              harness)
    tool_fields = harnesses.event_mapping(tool_event,
                                          harness).get("fields", {})
    tool_name = extract_canonical_field(raw_payload,
                                        tool_fields.get("tool", []))
    return (handback_mapping
            if tool_name == handback_mapping["matcher"] else None)


def _normalize_handback_event(
    raw_payload: dict[str, Any],
    handback_mapping: dict[str, Any],
    package: str,
    workspace: Path,
) -> dict[str, Any]:
    """Normalize a hand-back tool call into a canonical SubagentStop event with `args.handback = True`."""
    fields = handback_mapping.get("fields", {})
    subagent = str(
        extract_canonical_field(raw_payload, fields.get("subagent", [])) or "")
    output = extract_canonical_field(raw_payload, fields.get("output", []))
    session_id = str(
        extract_canonical_field(raw_payload, fields.get("session_id", []))
        or "")
    args = {
        "subagent": subagent,
        "handback": True,
        "handback_tool": handback_mapping["matcher"],
        "handback_event": handback_mapping["event_name"],
    }
    for key, value in _passthrough_args(raw_payload, fields).items():
        args.setdefault(key, value)
    return _subagent_stop_event(subagent, output, args, session_id,
                                raw_payload, package, workspace)


def is_handback_event(event_payload: dict[str, Any]) -> bool:
    """Whether a canonical SubagentStop arrived through a hand-back tool."""
    args = event_payload.get("args")
    return isinstance(args, dict) and args.get("handback") is True


def _normalize_tool_event(
    raw_payload: dict[str, Any],
    harness: str,
    canonical_event: str,
    package: str,
    workspace: Path,
) -> dict[str, Any]:
    """Normalize harness-specific PreToolUse or PostToolUse payload."""
    event_mappings = utils.get_event_mappings()
    harness_config = harnesses.harness_mapping(
        event_mappings.get(canonical_event, {}).get("harnesses", {}), harness)
    if not harness_config:
        raise TpuAgentSuiteError(
            f"unsupported harness for hook dispatch: '{harness}'")

    fields = harness_config.get("fields", {})

    raw_tool = str(
        extract_canonical_field(raw_payload, fields.get("tool", [])) or "")
    raw_args_val = extract_canonical_field(raw_payload, fields.get("args", []))
    raw_args: dict[str, Any] = raw_args_val if isinstance(raw_args_val,
                                                          dict) else {}
    raw_output = extract_canonical_field(raw_payload, fields.get("output", []))

    tool_mappings = utils.get_tool_mappings()
    normalized_tool = raw_tool
    normalized_args = dict(raw_args)

    # Reverse-lookup raw_tool in tool_mappings to find canonical tool name
    matched_canonical: str | None = None
    matched_harness_info: dict[str, Any] = {}
    for c_name, c_def in tool_mappings.items():
        h_info = harnesses.harness_mapping(c_def.get("harnesses", {}), harness)
        tools_list = h_info.get("tools", [])
        matcher = h_info.get("matcher", "")
        if raw_tool in tools_list or (matcher and re.search(
                rf"^({matcher})$", raw_tool)):
            matched_canonical = c_name
            matched_harness_info = h_info
            break

    if matched_canonical:
        normalized_tool = matched_canonical
        for canonical_k, raw_spec in matched_harness_info.get("args",
                                                              {}).items():
            val = extract_canonical_field(raw_args, raw_spec)
            if val is not None:
                if isinstance(val, str) and not val.strip():
                    continue
                normalized_args[canonical_k] = (str(val) if canonical_k
                                                in ("subagent",
                                                    "prompt") else val)

    session_id_val = str(
        extract_canonical_field(raw_payload, fields.get("session_id", []))
        or "")

    subagent_val = normalized_args.get(
        "subagent") if normalized_tool == "delegate" else None

    # Named as termination events already do it: a builtin that reads a
    # harness mapping (the background ledger) is handed only the event.
    context = _build_event_context(workspace, package, session_id_val)
    context["harness"] = harness
    event_dict: dict[str, Any] = {
        "event": canonical_event,
        "tool": normalized_tool,
        "subagent": subagent_val,
        "args": normalized_args,
        "output": raw_output if canonical_event == "PostToolUse" else None,
        "context": context,
        "raw_input": raw_payload,
    }

    # Check if the tool call succeeded based on the harness's declared error
    # field (harnesses without one route failures to a separate event).
    if canonical_event == "PostToolUse":
        event_dict["success"] = not extract_canonical_field(
            raw_payload, fields.get("error", []))

    return event_dict


def _extract_idle_flag(
    raw_payload: dict[str, Any],
    fields: dict[str, Any],
    harness: str,
    event_name: str,
    container: dict[str, Any] | None = None,
) -> bool | None:
    """Read the harness's `fully_idle` flag, warning on stderr if a declared field is missing."""
    candidates = fields.get("fully_idle", [])
    if not candidates:
        return None

    value = extract_canonical_field(raw_payload,
                                    candidates,
                                    container=container)
    if value is None:
        print(
            f"runtime: {harness} {event_name} payload carries none of "
            f"{list(candidates)}; treating the agent as possibly busy. If the "
            "harness renamed the field, stop hooks will fire on turn "
            "boundaries.",
            file=sys.stderr)
        return None
    return bool(value)


def _normalize_termination_event(
    raw_payload: dict[str, Any],
    harness: str,
    package: str,
    workspace: Path,
    event_name: str = "Stop",
) -> dict[str, Any]:
    """Normalize a root-loop termination into a canonical `Stop` or `SessionEnd` event."""
    fields = harnesses.harness_mapping(
        utils.get_event_mappings().get(event_name, {}).get("harnesses", {}),
        harness).get("fields", {})

    def first(keys: list[str], default: Any = "") -> Any:
        for key in keys:
            if raw_payload.get(key) is not None:
                return raw_payload[key]
        return default

    cid = str(
        first(fields.get("conversation_id", []), "")
        or first(fields.get("session_id", []), ""))
    context = _build_event_context(workspace, package, cid)
    context["conversation_id"] = cid
    context["harness"] = harness

    fully_idle = _extract_idle_flag(raw_payload, fields, harness, event_name)
    return {
        "event": event_name,
        "tool": "",
        "args": {},
        "output": None,
        "raw": raw_payload,
        "termination": {
            "reason": str(first(fields.get("stop_reason", []))),
            "error": str(first(fields.get("error", []))),
            "execution_num":
            int(first(fields.get("execution_num", []), 0) or 0),
            "fully_idle": fully_idle,
        },
        "context": context,
    }


def _normalize_session_start_event(
    raw_payload: dict[str, Any],
    harness: str,
    package: str,
    workspace: Path,
) -> dict[str, Any]:
    """Normalize a harness session start payload into a canonical SessionStart event."""
    event_mappings = utils.get_event_mappings()
    harness_config = harnesses.harness_mapping(
        event_mappings.get("SessionStart", {}).get("harnesses", {}), harness)
    if not harness_config:
        raise TpuAgentSuiteError(
            f"unsupported harness for session start: '{harness}'")

    fields = harness_config.get("fields", {})

    normalized_args: dict[str, Any] = {
        "session_id": "",
        "source": "startup",
        "cwd": str(workspace),
    }

    for canonical_key, candidate_fields in fields.items():
        val = extract_canonical_field(raw_payload, candidate_fields)
        if val is not None:
            normalized_args[canonical_key] = val

    # Coerce scalar string fields and preserve the workspace fallback if `cwd`
    # is empty.
    for scalar_key in ("cwd", "session_id", "transcript_path"):
        if scalar_key not in normalized_args:
            continue
        coerced = str(normalized_args[scalar_key] or "")
        if scalar_key == "cwd" and not coerced:
            coerced = str(workspace)
        normalized_args[scalar_key] = coerced

    session_id_val = str(normalized_args.get("session_id") or "")
    trans_path = str(normalized_args.get("transcript_path") or "")
    adaptor = harnesses.get(harness)
    subagent = (adaptor.resolve_subagent(session_id_val, trans_path)
                if adaptor is not None else None)

    return {
        "event": "SessionStart",
        "tool": None,
        "subagent": subagent,
        "args": normalized_args,
        "output": None,
        "context": _build_event_context(workspace, package, session_id_val),
        "raw_input": raw_payload,
    }


def normalize_inbound_event(
    raw_payload: dict[str, Any],
    harness: str,
    event: str,
    package: str,
    workspace: Path,
) -> dict[str, Any]:
    """Normalize harness-specific stdin payload into canonical Hook Input format."""
    if event == "SubagentStop":
        handback_mapping = _match_handback(raw_payload, harness)
        if handback_mapping is not None:
            return _normalize_handback_event(raw_payload, handback_mapping,
                                             package, workspace)
    adaptor = harnesses.get(harness)
    canonical_event = (adaptor.canonical_event(event, raw_payload) if adaptor
                       is not None else harnesses.to_canonical_event(
                           event, harness))

    if canonical_event in ("Stop", "SessionEnd"):
        return _normalize_termination_event(raw_payload,
                                            harness,
                                            package,
                                            workspace,
                                            event_name=canonical_event)
    if canonical_event == "SubagentStop":
        return _normalize_subagent_stop_event(raw_payload, harness, package,
                                              workspace)
    if canonical_event == "SessionStart":
        return _normalize_session_start_event(raw_payload, harness, package,
                                              workspace)
    if canonical_event in ("PreToolUse", "PostToolUse"):
        return _normalize_tool_event(raw_payload, harness, canonical_event,
                                     package, workspace)
    raise TpuAgentSuiteError(
        f"unsupported canonical event: '{canonical_event}'")
