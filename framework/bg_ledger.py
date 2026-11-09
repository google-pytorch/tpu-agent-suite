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
"""Per-agent ledger of background work, and the idle gate built on it.

Claude Code lists a session's live background tasks on every `SubagentStop`
but does not say which agent launched each one. This module records each
launch a `PostToolUse` reports under the agent that made it, one file per
task, and at a stop keeps the recorded tasks the harness still lists as
running. From that it derives `fully_idle` for the stop and, on a run the
harness will not resume, the verdict that refuses the stop while the agent's
own work is live. `hooks` wires these into event normalization and hook
dispatch; `harnesses.subagent_idle_mapping` says where a harness reports the
pieces.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

from framework import jsonio, paths, utils
from framework.harnesses import subagent_idle_mapping


def ledger_dir(session_id: str, agent_id: str) -> Path:
    """The directory holding one file per background task `agent_id` launched.

    One file per task: the hooks of launches made in one message run
    concurrently, and creating or unlinking distinct files needs no lock
    where a shared file would. The root conversation, which has no agent
    id, files under `root`.
    """
    return (paths.get_sessions_root() /
            f"{utils.safe_path_component(session_id)}.background" /
            utils.safe_path_component(agent_id or "root"))


def record_launch(session_id: str, agent_id: str, task_id: str,
                  task_type: str) -> None:
    """Record that `agent_id` launched background task `task_id`."""
    directory = ledger_dir(session_id, agent_id)
    jsonio.write_json(
        directory / f"{utils.safe_path_component(task_id)}.json", {
            "task_id": task_id,
            "task_type": task_type,
            "launched_at": utils.utc_now_iso(),
        })


def load_ledger(session_id: str, agent_id: str) -> dict[str, dict[str, Any]]:
    """The tasks `agent_id` launched and has not seen end, keyed by task id."""
    directory = ledger_dir(session_id, agent_id)
    if not directory.is_dir():
        return {}
    ledger: dict[str, dict[str, Any]] = {}
    for entry_file in sorted(directory.glob("*.json")):
        entry = jsonio.load_json_or_empty(entry_file)
        task_id = str(entry.get("task_id") or "")
        if task_id:
            ledger[task_id] = entry
        else:
            # Entries are written atomically, so this is not a torn write
            # but a foreign or corrupt file: left alone, and said, rather
            # than deleted on the ledger's behalf.
            print(f"runtime: ignoring unreadable ledger entry {entry_file}",
                  file=sys.stderr)
    return ledger


def forget_launch(session_id: str, agent_id: str, task_id: str) -> None:
    """Drop `task_id` from `agent_id`'s ledger; absent is fine."""
    (ledger_dir(session_id, agent_id) /
     f"{utils.safe_path_component(task_id)}.json").unlink(missing_ok=True)


def clear_ledger(session_id: str) -> None:
    """Drop every agent's ledger of a session, once the session is over."""
    shutil.rmtree(paths.get_sessions_root() /
                  f"{utils.safe_path_component(session_id)}.background",
                  ignore_errors=True)


def background_launch_without_shell(
    raw_payload: dict[str, Any],
    tool: str,
    harness: str,
    agent_tools: dict[str, list[str]],
) -> str:
    """The agent type asking `tool` for a background launch it cannot wait
    for (no shell to run `sleep` in, per `agent_tools`), or "" when the
    request is foreground, or from the root or an agent not in the table."""
    idle_mapping = subagent_idle_mapping(harness) or {}
    requested = (idle_mapping.get("launches", {}).get(tool)
                 or {}).get("requested")
    if not (requested and utils.extract_by_path(raw_payload, str(requested))):
        return ""
    agent_type = str(
        utils.extract_first_field(raw_payload,
                                  idle_mapping.get("agent_type", [])) or "")
    # Plugin installs name agents `<package>:<agent>`.
    tools = agent_tools.get(agent_type,
                            agent_tools.get(agent_type.rsplit(":", 1)[-1]))
    return agent_type if tools is not None and "shell" not in tools else ""


def outstanding_work(
    raw_payload: dict[str, Any],
    harness: str,
    event_name: str,
    session_id: str,
) -> tuple[bool | None, list[dict[str, Any]]]:
    """This agent's background tasks the harness still lists as running.

    Returns `(idle, outstanding)`:

    * `idle` is True when none of the tasks this agent launched is still
      listed as running, False when at least one is, and None when the
      question cannot be answered: the mapping declares no ledger, or the
      payload carries no live list (said on stderr, since every stop would
      otherwise pass as finished with nothing to explain why).
    * `outstanding` lists the still-running tasks, each as `task_id`,
      `task_type`, `launched_at`.

    Both sources are needed. The live list has no owner, so on its own a
    sibling's child would excuse this agent's stop; the ledger on its own
    goes stale when a task ends unreported. So a ledger entry the live list
    no longer carries is dropped here, and only the intersection counts.
    The agent's own id is left out of the live set: a background-launched
    subagent is listed as running at its own stop. There is no time limit:
    a package's runs outlast any ceiling, and dropping a live task on age
    would pass a stop while a healthy job runs.
    """
    idle_mapping = subagent_idle_mapping(harness)
    if idle_mapping is None or idle_mapping.get("source") != "ledger":
        return None, []
    live_tasks = utils.extract_by_path(raw_payload,
                                       str(idle_mapping["live_tasks"]))
    if not isinstance(live_tasks, list):
        print(
            f"runtime: {harness} {event_name} payload carries no "
            f"{idle_mapping['live_tasks']!r} list; cannot tell a paused "
            "subagent from a finished one and treating the stop as final.",
            file=sys.stderr)
        return None, []
    agent_id = str(
        utils.extract_first_field(raw_payload, idle_mapping.get(
            "agent_id", [])) or "")
    statuses = set(idle_mapping.get("live_statuses") or ["running"])
    live = {
        str(task.get("id"))
        for task in live_tasks
        if isinstance(task, dict) and task.get("status") in statuses
        and str(task.get("id") or "") not in ("", agent_id)
    }
    outstanding: list[dict[str, Any]] = []
    for task_id, entry in load_ledger(session_id, agent_id).items():
        if task_id in live:
            outstanding.append({
                "task_id":
                task_id,
                "task_type":
                str(entry.get("task_type") or "task"),
                "launched_at":
                str(entry.get("launched_at") or ""),
            })
        else:
            forget_launch(session_id, agent_id, task_id)
    return not outstanding, outstanding


def reported_launch(
    raw_payload: dict[str, Any],
    tool: str,
    harness: str,
) -> tuple[str, str] | None:
    """The (task_id, task_type) a PostToolUse on `tool` reports launching.

    Per the mapping's `launches` entry for the portable tool: the `task_id`
    path must hold a string, and every path in `requires` must hold its
    value. `requires` is for a tool whose every result carries the id
    (`Agent` returns `agentId` for a foreground run too, so its `status`
    must say `async_launched`); a tool that returns the id only for a
    background run needs none, which also catches a command the harness
    backgrounded on its own.
    """
    idle_mapping = subagent_idle_mapping(harness)
    if idle_mapping is None:
        return None
    launch = idle_mapping["launches"].get(tool)
    if not isinstance(launch, dict):
        return None
    requires = launch.get("requires") or {}
    if not isinstance(requires, dict) or any(
            utils.extract_by_path(raw_payload, str(path)) != value
            for path, value in requires.items()):
        return None
    task_id = utils.extract_by_path(raw_payload,
                                    str(launch.get("task_id") or ""))
    if not isinstance(task_id, str) or not task_id:
        return None
    return task_id, str(launch.get("task_type") or "task")


def harness_resumes_agent(event_payload: dict[str, Any], harness: str) -> bool:
    """Whether the harness wakes this subagent once its background work ends.

    If it will, a stop made with work still running is a pause and may pass
    unchecked; if not, the stop is final and must be refused.

    A harness that reports idleness itself (`fully_idle_source` is
    `harness`) resumes the agent. For a derived answer the idle mapping's
    `pause_resumes` decides: the named environment variable of this hook
    process must hold one of the listed values. Claude Code sets
    `CLAUDE_CODE_ENTRYPOINT` to `cli` in an interactive session, which
    resumes, and to `sdk-cli` headless, which does not. Anything else
    answers no: a wrong refusal costs a short wait, a wrong pass delivers
    the pause as the result and orphans the work.
    """
    args = event_payload.get("args")
    source = args.get("fully_idle_source") if isinstance(args, dict) else None
    if source != "ledger":
        return True
    policy = (subagent_idle_mapping(harness) or {}).get("pause_resumes")
    if isinstance(policy, dict) and policy.get("env"):
        value = os.environ.get(str(policy["env"]))
        return value is not None and value in (policy.get("values") or [])
    return False


def idle_gate_verdict(
    event_payload: dict[str, Any],
    harness: str,
    harness_resumes: bool | None = None,
) -> dict[str, Any] | None:
    """Refuse a stop made while the agent's own background work is live.

    On a run the harness will not resume (measured: Claude Code headless),
    such a stop is final: the caller's `Agent` call returns the pause text
    as the result and the work is killed or orphaned, so no later stop
    exists to validate. Refusing keeps the agent in its turn, where the
    completions do reach it, and the stop after them is the one checked.

    `harness_resumes` is `harness_resumes_agent` for this event, if the
    caller already asked, as the dispatcher does for this and the skip filter.
    """
    args = event_payload.get("args")
    if not (isinstance(args, dict) and args.get("fully_idle") is False
            and args.get("fully_idle_source") == "ledger"):
        return None
    if harness_resumes is None:
        harness_resumes = harness_resumes_agent(event_payload, harness)
    if harness_resumes:
        return None
    outstanding = [
        item for item in (args.get("outstanding_work") or [])
        if isinstance(item, dict) and item.get("task_id")
    ]
    if not outstanding:
        return None
    named = ", ".join(f"{item.get('task_type', 'task')} {item['task_id']}"
                      for item in outstanding)
    # 20 seconds: Claude Code blocks a bare `sleep` of 25 seconds or more.
    reason = (
        f"Not idle: background work you launched is still running ({named})."
        " Ending your turn now would make this message your final result"
        " and abandon that work. Run `sleep 20` in your shell tool, then"
        " try again: the result is delivered to you after a tool call."
        " If the work will not finish, stop it and report the failure.")
    return {"decision": "deny", "reason": reason, "followup_message": reason}
