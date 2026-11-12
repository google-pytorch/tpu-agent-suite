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
"""The harness adaptors, one module per harness, and their registry.

A harness is registered by one line in `HARNESSES`; today it also needs
its blocks in the two mapping tables and its entries in the core modules.
The registry is a literal, not a directory walk: hook dispatch reads it on
every tool call. Ships with the runtime; imports no build tooling.
"""

from __future__ import annotations

from typing import Any

from framework import utils
from framework.harnesses import (antigravity, base, claude_code, codex, cursor,
                                 google)

Harness = base.Harness
GoogleHarness = google.GoogleHarness
HookReply = base.HookReply
DispatchContext = base.DispatchContext

HARNESSES: dict[str, Harness] = {
    adaptor.name: adaptor
    for adaptor in (
        claude_code.ClaudeCode(),
        codex.Codex(),
        antigravity.Antigravity(),
        cursor.Cursor(),
    )
}


def names() -> list[str]:
    """Every registered harness name, sorted."""
    return sorted(HARNESSES)


def get(harness: str) -> Harness | None:
    """The adaptor registered as `harness`, or None."""
    return HARNESSES.get(harness)


def require(harness: str) -> Harness:
    """The adaptor registered as `harness`; raises if there is none."""
    adaptor = HARNESSES.get(harness)
    if adaptor is None:
        raise utils.TpuAgentSuiteError(
            f"unsupported harness: '{harness}' (supported: "
            f"{', '.join(names())})")
    return adaptor


# Facades derived from the registry, kept while callers migrate to it.

GOOGLE_HARNESSES: frozenset[str] = frozenset(
    name for name, adaptor in HARNESSES.items()
    if isinstance(adaptor, GoogleHarness))

# Where each Google harness keeps conversation brains, relative to ~/.gemini.
GOOGLE_HARNESS_BRAIN_DIRS: dict[str, str] = {
    name: adaptor.brain_dir
    for name, adaptor in HARNESSES.items()
    if isinstance(adaptor, GoogleHarness)
}

HARNESS_DISPLAY_NAMES: dict[str, str] = {
    name: adaptor.display_name
    for name, adaptor in HARNESSES.items()
}

HARNESS_CONFIG_DIRS: dict[str, str] = {
    name: adaptor.config_dir
    for name, adaptor in HARNESSES.items()
}


def is_google_harness(harness: str) -> bool:
    """Report whether `harness` belongs to the Google harness family."""
    return isinstance(get(harness), GoogleHarness)


def harness_mapping(harnesses: dict[str, Any], harness: str) -> dict[str, Any]:
    """The mapping block for `harness`, falling back within its family.

    An unregistered name reads its own block only.
    """
    adaptor = get(harness)
    if adaptor is None:
        return harnesses.get(harness) or {}
    return adaptor.mapping_block(harnesses)


def event_mapping(canonical_event: str, harness: str) -> dict[str, Any]:
    """The mapping block of `canonical_event` for `harness` (empty if none)."""
    adaptor = get(harness)
    return adaptor.event_mapping(
        canonical_event) if adaptor is not None else {}


def subagent_handback_mapping(harness: str) -> dict[str, Any] | None:
    """The mapping of the tool call a subagent on `harness` hands back through.

    A subagent's result normally reaches the caller when it ends its turn,
    which is what the native SubagentStop reports. Some harnesses also offer
    a dedicated tool for it -- in auto mode Claude Code injects
    `SubagentHandback` into its subagents, foreground or background, and
    only that call is delivered.
    The entry names the native event and tool (`matcher`) that carry such a
    hand-back, plus the fields to read it with. An entry missing either name
    counts as absent, so a half-written mapping registers nothing rather
    than a hook on every tool call. An optional `requires_mode` names the
    value of the stop's `permission_mode` field in which the harness gives
    the tool and accepts the call; a native stop in that mode is a turn
    boundary around the hand-back and goes unchecked. Any other stop is
    checked unless the stopping run's call was accepted.
    `transcript_prompt_id` names the transcript key that tags each record
    with its run's prompt id, which lets a resumed subagent's current run
    be read apart from earlier ones; without it no run can be located, and
    only `requires_mode` exempts a native stop.
    `transcript_offer` says how the harness records that it gave a run the
    tool: a record whose `flag` key is true and whose line holds `text`. In
    `requires_mode`, a stop of a run whose transcript shows neither that
    record nor a call is checked; without the entry the transcript cannot
    show it, and the mode alone exempts the stop.
    """
    adaptor = get(harness)
    return adaptor.handback_mapping() if adaptor is not None else None


def subagent_idle_mapping(harness: str) -> dict[str, Any] | None:
    """Return the `idle` block of `harness`'s SubagentStop mapping, or None.

    The block says how to tell whether a stopping subagent still has
    background work running. A harness that reports this maps `fully_idle`
    on the stop and needs no block. Claude Code does not, so the runtime
    derives it: launches reported by `PostToolUse` are recorded per agent,
    and at the stop the recorded task ids are checked against the live task
    list (the list has no owner; the record alone misses ended tasks).

    * `source`: always `ledger`.
    * `agent_id`: the payload field naming the agent.
    * `live_tasks`: the stop field holding the live-task list (`id`,
      `status`); `live_statuses`: which statuses count as running.
    * `launches`: per portable tool, how `PostToolUse` shows a launch:
      `requires` (payload paths and required values), `task_id` (path of the
      task id), `task_type` (label for messages).
    * `pause_resumes`: `{"env": VAR, "values": [...]}`; the harness resumes
      a paused subagent when VAR in the hook's environment holds a listed
      value. Claude Code: `CLAUDE_CODE_ENTRYPOINT` is `cli` interactive
      (resumes) or `sdk-cli` headless (does not). Otherwise the stop is
      refused.

    A block without `source`, `live_tasks`, or `launches` counts as absent.
    """
    idle_mapping = event_mapping("SubagentStop", harness).get("idle")
    if (isinstance(idle_mapping, dict) and idle_mapping.get("source")
            and idle_mapping.get("live_tasks")
            and isinstance(idle_mapping.get("launches"), dict)):
        return idle_mapping
    return None


def supported_harnesses() -> list[str]:
    """Harness names a hook can be dispatched for, sorted.

    Reads the event table on purpose, not only the registry: hook-dispatch
    builds its parser from this, so a store that has lost the table raises
    here, where the startup-failure path can still exit by policy.
    """
    found: set[str] = set(HARNESSES)
    for c_def in utils.get_event_mappings().values():
        found.update(c_def.get("harnesses", {}))
    return sorted(found)


def to_harness_event(canonical_event: str, harness: str) -> str:
    """Map a canonical event name to its harness-specific event identifier."""
    return event_mapping(canonical_event, harness).get("event_name",
                                                       canonical_event)


def to_canonical_event(harness_event: str, harness: str) -> str:
    """Map a harness-specific event identifier to its canonical event name."""
    adaptor = get(harness)
    return adaptor.canonical_event(
        harness_event, {}) if adaptor is not None else harness_event
