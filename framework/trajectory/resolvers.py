#!/usr/bin/env python3
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
"""Per-harness discovery of the transcripts belonging to one session.

Walks from the root session rather than relying on lifecycle hooks: this runs
when something has already gone wrong, possibly the hooks themselves. Every
function here is read-only and reports what it could not find rather than
raising.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import pathlib
import re
from typing import Any, Callable, Iterable

from framework import harnesses, jsonio, utils

Path = pathlib.Path
TpuAgentSuiteError = utils.TpuAgentSuiteError


@dataclasses.dataclass(frozen=True)
class Cut:
    """Where the bundle starts, and how that point was established."""

    detected: bool
    # skill-tool-use | slash-command | skill-file-read | user-consented-all |
    # none
    basis: str
    # Record index (claude-code) or step_index (Google harnesses).
    key: Any = None
    # How `key` selects records. "index" slices by position; any other value
    # names a record field compared with >=. Keeping this on the cut lets
    # slicing stay generic, so harness knowledge lives only in the resolver.
    mode: str = "index"


@dataclasses.dataclass(frozen=True)
class TranscriptRef:
    """One transcript file belonging to the session, root or subagent."""

    kind: str  # "root" | "subagent"
    label: str  # "root", or the delegate name
    agent_id: str
    parent_id: str
    depth: int
    path: Path
    order: int  # position of the launching call, for stable sorting
    bytes: int
    # Which on-disk variant this came from, where a harness keeps more than
    # one. Empty when the harness has only a single form.
    source: str = ""


@dataclasses.dataclass(frozen=True)
class SessionBundle:
    """Everything a resolver could find for one session."""

    harness: str
    session_id: str
    cut: Cut
    transcripts: list[TranscriptRef]
    harness_version: str = ""
    models: dict[str, Any] = dataclasses.field(default_factory=dict)
    missing: list[str] = dataclasses.field(default_factory=list)


def read_transcript(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL transcript, skipping lines that do not parse.

    A session killed mid-write leaves a truncated final line, so a bad line is
    expected rather than exceptional. Laxer than jsonio.read_jsonl, which it
    wraps: undecodable bytes are dropped, and a transcript that cannot be read
    at all reads as empty rather than raising, since every caller here reports
    what it could not find instead of failing.
    """
    try:
        return jsonio.read_jsonl(path, errors="ignore")
    except OSError:
        return []


def _tool_uses(record: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield the tool_use blocks of one transcript record."""
    content = (record.get("message") or {}).get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            yield block


def file_size(path: Path) -> int:
    """Size in bytes, 0 when the file is gone. Follows symlinks by design."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


# Claude Code resolver.


def _claude_root_transcript(session_id: str) -> Path | None:
    """Locate the root transcript by session id.

    Globbed rather than derived from the workspace path: the project slug
    encoding is the harness's business, and a glob stays correct if it changes.

    `CLAUDE_CONFIG_DIR` moves the whole config tree, which the eval harness
    does so transcripts land under /logs; `~/.claude` then holds nothing.
    """
    config = os.environ.get("CLAUDE_CONFIG_DIR")
    projects = Path(config or Path.home() / ".claude") / "projects"
    matches = sorted(projects.glob(f"*/{session_id}.jsonl"))
    return matches[0] if matches else None


# A skill the user typed as a slash command. Claude Code expands it inline and
# records the invocation on the user record as tagged text, not as a tool call.
_COMMAND_NAME = re.compile(r"<command-name>\s*/?([^<\s]+)\s*</command-name>")


def _slash_command(record: dict[str, Any]) -> str:
    """Name of the slash command a user record carries, or "" if none.

    Claude Code writes the typed command as a plain string, but content may
    also be a block list, so text blocks are read too. Tool results are not:
    a tool that echoes a transcript (the exporter itself, for one) would
    otherwise look like an activation.
    """
    if record.get("type") != "user":
        return ""
    content = (record.get("message") or {}).get("content")
    if isinstance(content, list):
        content = "\n".join(
            str(block.get("text") or "") for block in content
            if isinstance(block, dict) and block.get("type") == "text")
    if not isinstance(content, str):
        return ""
    match = _COMMAND_NAME.search(content)
    return match.group(1) if match else ""


def _claude_cut(records: list[dict[str, Any]], entrypoint: str) -> Cut:
    """Find the most recent activation of the entrypoint skill.

    An activation is either a Skill tool call the model made or a slash
    command the user typed; both are exact records, so the cut is exact.
    The name is matched whole, so `/tpu-agent-export-trajectory` is not
    an activation of `tpu-agent`.
    """
    cut_index = -1
    basis = "none"
    for index, record in enumerate(records):
        if _slash_command(record) == entrypoint:
            cut_index, basis = index, "slash-command"
            continue
        for block in _tool_uses(record):
            if block.get("name") != "Skill":
                continue
            if (block.get("input") or {}).get("skill") == entrypoint:
                cut_index, basis = index, "skill-tool-use"
    if cut_index < 0:
        return Cut(detected=False, basis="none")
    return Cut(detected=True, basis=basis, key=cut_index)


def _claude_subagent_dir(root_transcript: Path, session_id: str) -> Path:
    """Durable subagent directory beside the root transcript.

    The `outputFile` paths in the transcript point into /tmp, but those are
    symlinks into this directory; going direct avoids depending on /tmp.
    """
    return root_transcript.parent / session_id / "subagents"


def _claude_collect_agents(subagent_dir: Path) -> dict[str, dict[str, Any]]:
    """Load every subagent's metadata, keyed by agent id."""
    agents: dict[str, dict[str, Any]] = {}
    for meta_path in sorted(subagent_dir.glob("agent-*.meta.json")):
        transcript = meta_path.with_name(meta_path.name[:-len(".meta.json")] +
                                         ".jsonl")
        if not transcript.is_file():
            continue
        meta = jsonio.load_json_or_empty(meta_path)
        if not meta:
            continue
        agent_id = meta_path.name[len("agent-"):-len(".meta.json")]
        agents[agent_id] = {"meta": meta, "path": transcript}
    return agents


def _claude_owned_tool_uses(records: list[dict[str, Any]]) -> dict[str, int]:
    """Map each tool_use id in a transcript to its record index."""
    owned: dict[str, int] = {}
    for index, record in enumerate(records):
        for block in _tool_uses(record):
            if block.get("id"):
                owned[str(block["id"])] = index
    return owned


def _claude_included_agents(
    agents: dict[str, dict[str, Any]],
    root_tool_uses: dict[str, int],
    cut_index: int,
    root_id: str,
) -> dict[str, tuple[str, int]]:
    """Select subagents launched at or after the cut, including nested ones.

    An agent is in scope when the `Agent` call that launched it sits at or
    after the cut in the root, or anywhere inside an already-included agent.
    Returns agent id -> (parent id, launch order), where a top-level agent's
    parent is `root_id` so the tree joins on agent ids alone.
    """
    # Index which transcript contains each tool_use id, so a nested agent can
    # be attributed to the agent that launched it rather than to the root.
    agent_tool_uses = {
        agent_id: _claude_owned_tool_uses(read_transcript(entry["path"]))
        for agent_id, entry in agents.items()
    }

    included: dict[str, tuple[str, int]] = {}
    # Seed with agents launched directly by the root at or after the cut.
    for agent_id, entry in agents.items():
        tool_use_id = str(entry["meta"].get("toolUseId") or "")
        order = root_tool_uses.get(tool_use_id)
        if order is not None and order >= cut_index:
            included[agent_id] = (root_id, order)

    # Then walk outward: any agent launched from inside an included agent is
    # itself in scope, at any depth.
    changed = True
    while changed:
        changed = False
        for agent_id, entry in agents.items():
            if agent_id in included:
                continue
            tool_use_id = str(entry["meta"].get("toolUseId") or "")
            for parent_id in list(included):
                order = agent_tool_uses.get(parent_id, {}).get(tool_use_id)
                if order is None:
                    continue
                included[agent_id] = (parent_id, order)
                changed = True
                break
    return included


def resolve_claude_code(session_id: str, entrypoint: str) -> SessionBundle:
    """Resolve root and subagent transcripts for a Claude Code session."""
    missing: list[str] = []
    root_path = _claude_root_transcript(session_id)
    if root_path is None:
        raise TpuAgentSuiteError(
            f"no Claude Code transcript found for session {session_id}")

    records = read_transcript(root_path)
    cut = _claude_cut(records, entrypoint)
    cut_index = int(cut.key) if cut.detected else 0

    # The harness stamps its own version on transcript records, which is more
    # reliable than shelling out to `claude --version` after the fact.
    harness_version = ""
    models: dict[str, Any] = {"observed": [], "usage": {}, "source": "none"}
    for record in records:
        harness_version = str(record.get("version") or harness_version)
        model = (record.get("message") or {}).get("model")
        if model and model not in models["observed"]:
            models["observed"].append(model)
            models["source"] = "transcript"
        if isinstance(record.get("modelUsage"), dict):
            models["usage"].update(record["modelUsage"])

    transcripts = [
        TranscriptRef(kind="root",
                      label="root",
                      agent_id=session_id,
                      parent_id="",
                      depth=0,
                      path=root_path,
                      order=-1,
                      bytes=file_size(root_path))
    ]

    subagent_dir = _claude_subagent_dir(root_path, session_id)
    if subagent_dir.is_dir():
        agents = _claude_collect_agents(subagent_dir)
        included = _claude_included_agents(agents,
                                           _claude_owned_tool_uses(records),
                                           cut_index, session_id)
        # Sort by launch order so the bundle reads in the order work happened.
        for agent_id, (parent_id, order) in sorted(included.items(),
                                                   key=lambda kv: kv[1][1]):
            meta = agents[agent_id]["meta"]
            path = agents[agent_id]["path"]
            transcripts.append(
                TranscriptRef(kind="subagent",
                              label=str(meta.get("agentType") or "subagent"),
                              agent_id=agent_id,
                              parent_id=parent_id,
                              depth=int(meta.get("spawnDepth") or 1),
                              path=path,
                              order=order,
                              bytes=file_size(path)))
    else:
        missing.append(f"subagent directory not found: {subagent_dir}")

    return SessionBundle(harness="claude-code",
                         session_id=session_id,
                         cut=cut,
                         transcripts=transcripts,
                         harness_version=harness_version,
                         models=models,
                         missing=missing)


# Google harness resolver.
#
# Every harness in harnesses.GOOGLE_HARNESSES stores conversations the same way,
# so one resolver serves the family. The harness name selects which brain
# roots are searched and labels the bundle; nothing else branches on it.


def _google_brain_dir(harness: str, conversation_id: str) -> Path | None:
    """Locate a conversation's brain directory under `harness`'s brain roots.

    A brain root belongs to `harness` when its directory name under ~/.gemini
    contains the harness name, so conversations from another family member
    are never picked up. Globbed rather than read from
    harnesses.GOOGLE_HARNESS_BRAIN_DIRS, which registers one root per harness:
    a harness can keep conversations under several roots, and the registered
    one alone would not find the others.
    """
    pattern = f"*{harness}*/brain/{conversation_id}"
    matches = sorted((Path.home() / ".gemini").glob(pattern))
    return matches[0] if matches else None


def _google_transcript(brain_dir: Path) -> tuple[Path, str]:
    """A conversation's transcript, and which variant it turned out to be.

    `transcript_full.jsonl` is preferred, but it is not always written: agy's
    `--print` mode produces none at all, and a live session may not have
    flushed it yet. The compact `transcript.jsonl` carries the same records
    and step indices, so it is a usable substitute.

    It is not an equivalent one, though: it truncates long field values, and a
    tool-call path truncated before the skill document would cost the
    activation cut. So the variant is reported rather than silently
    substituted, and lands in the manifest where a missing cut can be
    explained by it.
    """
    logs = brain_dir / ".system_generated" / "logs"
    full = logs / "transcript_full.jsonl"
    if full.is_file():
        return full, "transcript_full"
    compact = logs / "transcript.jsonl"
    if compact.is_file():
        return compact, "transcript"
    # Neither exists: name the preferred one, so the error says what was
    # actually looked for.
    return full, "missing"


def _google_cut(records: list[dict[str, Any]], entrypoint: str) -> Cut:
    """Find the step index of the most recent entrypoint skill activation.

    Google harnesses have no skill tool, so activation shows up as a file read
    of the skill document -- the same heuristic the repo's eval checks use.
    """
    needle = f"skills/{entrypoint}/SKILL.md"
    cut_step: int | None = None
    for record in records:
        for call in record.get("tool_calls") or []:
            if call.get("name") != "view_file":
                continue
            target = str((call.get("args") or {}).get("AbsolutePath") or "")
            if needle in target:
                cut_step = int(record.get("step_index") or 0)
    if cut_step is None:
        return Cut(detected=False, basis="none")
    return Cut(detected=True,
               basis="skill-file-read",
               key=cut_step,
               mode="step_index")


@dataclasses.dataclass(frozen=True)
class _Descendant:
    """One subagent conversation found below the root, with its place in the tree."""

    conversation_id: str
    descriptor: dict[str, Any]
    brain_dir: Path
    spawn_step: int
    parent_id: str
    depth: int


def _google_descendants(
    harness: str,
    brain_dir: Path,
    cut_step: int,
    depth: int,
    parent_id: str,
    visited: set[str] | None = None,
) -> list[_Descendant]:
    """Recursively collect subagent conversations spawned at or after the cut.

    Children are wholly in scope once included, so the cut applies only at the
    level where spawning happened. Parent and depth are carried down so the
    manifest can describe the delegation tree rather than a flat list.

    `visited` guards the walk, seeded with the conversation being walked from
    so a descriptor naming its own ancestor cannot send the recursion back
    around. Without it a cyclic or self-referencing descriptor recurses until
    the stack limit -- in the one tool reached for when things are already
    broken. It also stops a conversation reachable from two parents being
    collected twice, which would duplicate a transcript in the bundle.
    """
    if visited is None:
        visited = {parent_id}
    found: list[_Descendant] = []
    descriptors = sorted(
        (brain_dir / ".system_generated" / "subagents").glob("*.json"))
    for descriptor_path in descriptors:
        descriptor = jsonio.load_json_or_empty(descriptor_path)
        if not descriptor:
            continue
        spawn_step = int(descriptor.get("spawnStepIndex") or 0)
        if spawn_step < cut_step:
            continue
        child_id = str(
            descriptor.get("conversationId") or descriptor_path.stem)
        if child_id in visited:
            continue
        child_brain = _google_brain_dir(harness, child_id)
        if child_brain is None:
            continue
        visited.add(child_id)
        found.append(
            _Descendant(conversation_id=child_id,
                        descriptor=descriptor,
                        brain_dir=child_brain,
                        spawn_step=spawn_step,
                        parent_id=parent_id,
                        depth=depth))
        # A subagent's own subagents are in scope unconditionally, so recurse
        # with a cut of 0 rather than the parent's.
        found.extend(
            _google_descendants(harness, child_brain, 0, depth + 1, child_id,
                                visited))
    return found


def resolve_google_harness(harness: str, session_id: str,
                           entrypoint: str) -> SessionBundle:
    """Resolve root and subagent transcripts for a Google harness conversation.

    `harness` limits the search to that harness's brain roots, and is
    recorded on the bundle so the manifest names the harness the session
    actually ran on rather than the family it belongs to.
    """
    missing: list[str] = []
    brain_dir = _google_brain_dir(harness, session_id)
    if brain_dir is None:
        display_name = harnesses.HARNESS_DISPLAY_NAMES.get(harness, harness)
        raise TpuAgentSuiteError(
            f"no {display_name} brain directory found for conversation"
            f" {session_id}")

    root_path, root_source = _google_transcript(brain_dir)
    records = read_transcript(root_path) if root_path.is_file() else []
    if not root_path.is_file():
        missing.append(f"root transcript not found: {root_path}")
    elif root_source != "transcript_full":
        # Recorded, not silently accepted: the compact variant truncates long
        # field values, so a cut missed here may be an artefact of the
        # fallback rather than a session that never activated the entrypoint.
        missing.append(
            f"read the compact transcript, {root_source}.jsonl; long tool-call "
            "paths may be truncated")
    cut = _google_cut(records, entrypoint)
    cut_step = int(cut.key) if cut.detected else 0

    transcripts = [
        TranscriptRef(kind="root",
                      label="root",
                      agent_id=session_id,
                      parent_id="",
                      depth=0,
                      path=root_path,
                      order=-1,
                      bytes=file_size(root_path),
                      source=root_source)
    ]

    # Sorted by spawn step so the bundle reads in the order work happened.
    for child in sorted(_google_descendants(harness, brain_dir, cut_step, 1,
                                            session_id),
                        key=lambda item: item.spawn_step):
        child_transcript, child_source = _google_transcript(child.brain_dir)
        sub = child.descriptor.get("subagentDescriptor") or {}
        transcripts.append(
            TranscriptRef(kind="subagent",
                          label=str(
                              sub.get("typeName") or sub.get("role")
                              or "subagent"),
                          agent_id=child.conversation_id,
                          parent_id=child.parent_id,
                          depth=child.depth,
                          path=child_transcript,
                          order=child.spawn_step,
                          bytes=file_size(child_transcript),
                          source=child_source))

    # Google harnesses record neither the model nor their own version anywhere
    # in the transcript, so both stay unknown.
    return SessionBundle(harness=harness,
                         session_id=session_id,
                         cut=cut,
                         transcripts=transcripts,
                         harness_version="",
                         models={
                             "observed": [],
                             "usage": {},
                             "source": "none"
                         },
                         missing=missing)


# A missing entry here is what produces the "not supported" refusal, so adding
# a harness is a resolver plus one line, with no change to any skill document.
# Google harnesses need neither: each member of harnesses.GOOGLE_HARNESSES is
# registered with the shared resolver, bound to its own name.
RESOLVERS: dict[str, Callable[[str, Path, str], SessionBundle]] = {
    "claude-code": resolve_claude_code,
    **{
        harness: functools.partial(resolve_google_harness, harness)
        for harness in sorted(harnesses.GOOGLE_HARNESSES)
    },
}


def supported(harness: str) -> bool:
    """Whether a resolver is registered for `harness`."""
    return harness in RESOLVERS


def unsupported_message(harness: str) -> str:
    """The single wording for a harness with no resolver."""
    return f"exporting trajectories is not supported on {harness} at this time"


def resolve_session(harness: str, session_id: str,
                    entrypoint: str) -> SessionBundle:
    """Dispatch to the resolver for `harness`, or refuse if there is none."""
    resolver = RESOLVERS.get(harness)
    if resolver is None:
        raise TpuAgentSuiteError(unsupported_message(harness))
    return resolver(session_id, entrypoint)
