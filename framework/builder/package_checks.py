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
"""Checks that reject a broken package while it loads.

Catches agents that call each other in a loop or nest too deep, unknown or
misspelled tools, hooks on events they can't run on, `{{...}}` references to
files the unit doesn't ship, and manifest fields that were removed.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Iterable, Sequence

from framework import utils
from framework.builder import package_discovery, tokens

Path = pathlib.Path


def get_harness_tool_aliases() -> dict[str, str]:
    """Derive legacy tool alias suggestions dynamically from tool-mappings.json as single source of truth."""
    mappings = utils.get_tool_mappings()
    canonical_tools = {"read", "write", "edit", "shell", "search"}
    aliases: dict[str, str] = {
        "sh": "shell",
        "bash": "shell",
        "grep": "search",
        "glob": "search",
        "edit_file": "edit",
        "apply_patch": "write",
    }
    for canonical, defn in mappings.items():
        if canonical not in canonical_tools or not isinstance(defn, dict):
            continue
        for harness, h_defn in defn.get("harnesses", {}).items():
            for t in h_defn.get("tools", []):
                t_lower = t.lower()
                if t_lower not in canonical_tools and t_lower not in aliases:
                    aliases[t_lower] = canonical
    return aliases


HARNESS_TOOL_ALIASES: dict[str, str] = get_harness_tool_aliases()

# `reap-jobs` cancels the run's leftover compute jobs, which is only right once
# the session is over.
VALID_BUILTIN_HOOK_ACTIONS: dict[str, frozenset[str]] = {
    "start-run":
    frozenset({"SessionStart"}),
    "validate-schema":
    frozenset({"PreToolUse", "SubagentStop"}),
    "reap-jobs":
    frozenset({"SessionEnd"}),
    # Records the active skill (PostToolUse) and blocks tools that skill
    # doesn't allow (PreToolUse).
    "skill-tools":
    frozenset({"PreToolUse", "PostToolUse"}),
    # Refuses a background launch the agent cannot wait for (PreToolUse),
    # records the launches made (PostToolUse) and clears the session's
    # ledger (SessionEnd). The stop gate reads the ledger itself.
    "track-background":
    frozenset({"PreToolUse", "PostToolUse", "SessionEnd"}),
    # The other builtins act: `start-run` initializes a run, `reap-jobs`
    # cancels compute. Each is pinned to the events where that action is
    # correct. `record-event` only appends to the study's event log, so it
    # changes nothing the session goes on to do and is legal on every event.
    # A recorder pinned the same way would simply be absent from any event
    # nobody anticipated, and a missing recorder looks identical to nothing
    # happening.
    "record-event":
    frozenset({
        "SessionStart", "PreToolUse", "PostToolUse", "SubagentStop", "Stop",
        "SessionEnd"
    }),
}

# Levels of delegation permitted below a released skill. Must match the harness
# nesting limit the install plan sets, or a package builds here and fails there.
MAX_DELEGATION_DEPTH = 3

# Hook events whose payload carries something for `matchers` to select: the
# tool on the tool events, the subagent on SubagentStop. The other events
# name neither, so the runtime's matcher emulation would skip every fire of
# a hook that filtered on them -- a hook that silently never runs.
MATCHABLE_HOOK_EVENTS = ("PreToolUse", "PostToolUse", "SubagentStop")


def check_hook_matchers(hook_name: str, event: str,
                        matchers: list[str] | None) -> None:
    """Reject `matchers` on a hook event that has nothing to match them on."""
    if matchers and event not in MATCHABLE_HOOK_EVENTS:
        raise utils.TpuAgentSuiteError(
            f"hook '{hook_name}' declares matchers, but {event} has nothing "
            f"to match them against; matchers apply only to "
            f"{', '.join(MATCHABLE_HOOK_EVENTS)}")


def covers(declared: Iterable[str], target: str, package_dir: Path) -> bool:
    """Whether `target` is among `declared`, under a declared directory, or a
    directory whose every file is declared."""
    entries = {entry.rstrip("/") for entry in declared}
    if target in entries:
        return True
    if any(target.startswith(entry + "/") for entry in entries):
        return True
    path = package_dir / target
    if path.is_dir():
        files = [
            package_discovery.relative_to_package(f, package_dir)
            for f in path.rglob("*") if f.is_file()
            and not package_discovery.dist_excluded(f.relative_to(package_dir))
        ]
        return bool(files) and all(
            covers(entries, name, package_dir) for name in files)
    return False


def validate_resource_tokens(
    text: str,
    package_dir: Path,
    label: str,
    declared: Iterable[str],
    assets: Iterable[str] = ()) -> None:
    """Fail when a token names a file its document does not ship.

    `declared` is what the document that holds `text` ships: what it declared
    in frontmatter, and what the references it reads declared. A file that
    exists but that only some other unit declares does not count, because a
    build that holds this document need not hold that one. A token that
    renders to a path nothing ships is the migration bug this layout exists
    to prevent, so it fails the build rather than the run.

    Data and schemas belong to the package rather than to a unit, so a data
    or schema token is held to `assets`, what package.release.json copies.
    """
    declared, assets = list(declared), list(assets)
    for kind, rel in tokens.resource_token_targets(text):
        target = utils.rooted(package_dir, rel, f"{label} {kind} token")
        if not target.exists():
            raise utils.TpuAgentSuiteError(
                f"{label} names {kind} {rel!r} in a token, but the package"
                " has no such file")
        if kind in ("data", "schema"):
            if not covers(assets, rel, package_dir):
                raise utils.TpuAgentSuiteError(
                    f"{label} names {kind} {rel!r} in a token, but no build"
                    " ships it: package.release.json does not list it under `copy`"
                )
            continue
        if not covers(declared, rel, package_dir):
            raise utils.TpuAgentSuiteError(
                f"{label} names {kind} {rel!r} in a token but does not"
                f" declare it: add it to `{kind}s` in its frontmatter, or"
                " to the frontmatter of a reference it reads")


def validate_delegation_graph(
    skills: Sequence[dict[str, Any]],
    agents_by_name: dict[str, dict[str, Any]],
    expected_agents: set[str],
) -> None:
    """Validate the delegation graph: cycles, depth, and reachability.

    Every released skill is a root of the graph. A skill may delegate to an
    agent or to another skill.

    Depth is counted from the skill. The skill is level 0, what it delegates
    to is level 1, and no chain may go deeper than MAX_DELEGATION_DEPTH:
    skill -> L1 -> L2 -> L3.

    The limit must stay in step with the harness. Claude Code enforces its own nesting
    limit via CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH, set by the install plan, and
    a package that builds here but exceeds that there fails at the point of
    delegation rather than at build time.
    """
    visited: set[str] = set()
    roots = list(skills)
    nodes: dict[str, dict[str, Any]] = dict(agents_by_name)
    for root in roots:
        nodes.setdefault(root["name"], root)

    def visit(agent_name: str, depth: int, path: tuple[str, ...]) -> None:
        if agent_name in path:
            cycle = " -> ".join((*path, agent_name))
            raise utils.TpuAgentSuiteError(
                f"delegation graph contains a cycle: {cycle}")
        if depth > MAX_DELEGATION_DEPTH:
            chain = " -> ".join((*path, agent_name))
            raise utils.TpuAgentSuiteError(
                f"delegation exceeds {MAX_DELEGATION_DEPTH} levels below"
                f" skill {path[0]}: {chain}")

        visited.add(agent_name)
        for delegated in nodes[agent_name].get("delegates", []):
            visit(delegated, depth + 1, (*path, agent_name))

    for root in roots:
        for delegated in root.get("delegates", []):
            visit(delegated, 1, (root["name"], ))
        # A skill that is also a selectable agent (its twin) is its own root:
        # nothing delegates to it, so reachability has to count it as reached
        # or the package fails to build for the one agent a user starts from.
        visited.add(root["name"])

    unreachable = expected_agents - visited
    if unreachable:
        raise utils.TpuAgentSuiteError(
            f"agents are unreachable from any skill: {sorted(unreachable)}")


def delegation_warnings(manifest: dict[str, Any]) -> list[str]:
    """Delegation the guide allows and expects to be rare.

    Two cases, neither of which fails a build: a skill that delegates to a
    skill, and a skill that is built only because something delegates to it.
    """
    skills = {skill["name"]: skill for skill in manifest.get("skills", [])}
    warnings: list[str] = []
    for name, skill in skills.items():
        for delegate in skill.get("delegates", []):
            if delegate in skills and delegate != name:
                warnings.append(
                    f"skill '{name}' delegates to skill '{delegate}'; a"
                    " skill should rarely delegate to a skill")
    units = (*manifest.get("skills", []), *manifest.get("agents", []))
    for name in manifest.get("delegated_skills", []):
        by = sorted({
            unit["name"]
            for unit in units if name in unit.get("delegates", [])
        })
        warnings.append(
            f"skill '{name}' is not one of the skills this build names; it"
            f" is built only because {', '.join(by)} delegates to it")
    return warnings


def print_delegation_warnings(manifest: dict[str, Any]) -> None:
    for warning in delegation_warnings(manifest):
        print(f"WARNING: {warning}", file=sys.stderr)


def validate_agent_tool(
    tool_name: str,
    declared_tools: dict[str, dict[str, Any]],
    agent_name: str,
    kind: str = "agent",
) -> dict[str, Any]:
    """Validate that a tool a skill or agent lists is declared and has
    action_level metadata, suggesting canonical names for legacy aliases.

    `kind` is what `agent_name` names, "agent" or "skill", for the message.
    """
    if tool_name not in declared_tools:
        canonical = HARNESS_TOOL_ALIASES.get(tool_name.lower())
        suggestion = (f" (did you mean canonical tool '{canonical}'?)"
                      if canonical else "")
        raise utils.TpuAgentSuiteError(
            f"{kind} '{agent_name}' references undeclared tool '{tool_name}'{suggestion}. "
            f"{kind.capitalize()}s must declare canonical tools "
            f"({', '.join(repr(t) for t in sorted(utils.PORTABLE_TOOLS))}) "
            "or declared package scripts/mcp_servers.")
    tool = declared_tools[tool_name]
    if "action_level" not in tool:
        raise utils.TpuAgentSuiteError(
            f"tool '{tool_name}' declared for {kind} '{agent_name}' lacks required 'action_level' metadata"
        )
    return tool


# Fields package.release.json once carried. Each is rejected by name, with where its
# content lives now, rather than as an anonymous unknown property.
REMOVED_MANIFEST_FIELDS = {
    "entrypoint":
    "name the skill in `skills`",
    "agents":
    "agents are discovered from the `delegates` of each skill; put each at"
    " agents/<name>.md",
    "include":
    "fragment files are gone; declare metadata in the frontmatter of the"
    " skill, agent or reference it belongs to",
    "schemas":
    "every schemas/*.json is registered automatically",
    "host_dependencies":
    "declare it in the frontmatter of the unit that needs it",
    "tpu_dependencies":
    "declare it in the frontmatter of the unit that needs it",
    "required_capabilities":
    "harness capabilities are uniform; nothing declares them",
}


def reject_removed_fields(manifest: dict[str, Any]) -> None:
    """Fail on a package.release.json field that is no longer read, saying why."""
    found = [key for key in REMOVED_MANIFEST_FIELDS if key in manifest]
    if found:
        raise utils.TpuAgentSuiteError(
            "package.release.json declares fields that are no longer"
            " supported:\n" +
            "\n".join(f"  {key}: {REMOVED_MANIFEST_FIELDS[key]}"
                      for key in found))
