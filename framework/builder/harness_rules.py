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
"""What differs between harnesses when building a package.

Each harness names its tools differently, has its own way for one agent to
call another, and expects its own agent file header. This module holds those
differences: tool name maps, per-harness delegation instructions, agent
frontmatter, and Codex's sandbox and approval settings. It also sets the shell
timeouts and subagent limits a harness has to allow.
"""

from __future__ import annotations

import sys
from typing import Any

from framework import harnesses, utils
from framework.builder import package_checks, package_loader

# Delegates a single session may run at once. A parent waiting on its child
# still holds a slot, so a nested chain needs one slot per level, and
# delegates running side by side need one slot each.
MAX_CONCURRENT_DELEGATES = 20

# Longest `compute run --timeout` a package prompt may ask for, in seconds.
# A synchronous run holds the shell call open for its whole duration, so the
# harness's shell timeout has to outlast it.
MAX_SYNC_COMPUTE_TIMEOUT_S = 3600

# Margin over the run's own timeout for staging inputs, pulling outputs, and
# the run reporting its own timeout before the shell kills it.
SHELL_TIMEOUT_HEADROOM_S = 300

# How long a harness shell call must be allowed to stay open.
SHELL_TIMEOUT_MS = (MAX_SYNC_COMPUTE_TIMEOUT_S +
                    SHELL_TIMEOUT_HEADROOM_S) * 1000

# Environment the Claude Code install plan writes into the workspace settings.
# - The two caps keep the harness in step with what the compiler enforces.
#   Background work is allowed: the runtime's idle gate
#   (`bg_ledger.idle_gate_verdict`) keeps a subagent's wait from passing as
#   its result.
# - Bash timeouts: Claude Code defaults to 2 minutes (max 10), too short for a
#   `compute run`, which gets killed along with its shell. Raise both to cover
#   the longest run; the default has to be set too or it stays at 2 minutes.
CLAUDE_CODE_SETTINGS_ENV: dict[str, str] = {
    "CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH":
    str(package_checks.MAX_DELEGATION_DEPTH),
    "CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS":
    str(MAX_CONCURRENT_DELEGATES),
    "BASH_DEFAULT_TIMEOUT_MS":
    str(SHELL_TIMEOUT_MS),
    "BASH_MAX_TIMEOUT_MS":
    str(SHELL_TIMEOUT_MS),
}


def resolve_agent_tools(
    agent: dict[str, Any],
    declared_tools: dict[str, dict[str, Any]],
    include_internal: bool,
) -> list[dict[str, Any]]:
    """Resolve tool metadata for an agent, filtering internal tools with warnings if unrequested."""
    resolved: list[dict[str, Any]] = []
    agent_name = agent.get("name", "unnamed")
    for tool_name in agent.get("tools", []):
        tool = package_checks.validate_agent_tool(tool_name, declared_tools,
                                                  agent_name)
        if package_loader.excluded_internal(tool, include_internal):
            print(
                f"WARNING: Skipping internal tool '{tool_name}' for agent"
                f" '{agent['name']}' (use --include-internal to include).",
                file=sys.stderr,
            )
            continue
        resolved.append(tool)
    return resolved


def load_harness_tool_map(harness: str) -> dict[str, list[str]]:
    """Derive the tool mapping for a harness from tool-mappings.json.

    `tool-mappings.json` is the single source of truth for tool vocabulary
    across both runtime hook interception and compiler frontmatter generation.
    """
    mappings = utils.get_tool_mappings()
    tool_map: dict[str, list[str]] = {}
    for canonical_name, spec in mappings.items():
        if canonical_name == "delegate" or not isinstance(spec, dict):
            continue
        blocks = spec.get("harnesses") or {}
        h_info = harnesses.harness_mapping(blocks, harness)
        tools = h_info.get("tools")
        if tools:
            tools_list = list(tools)
            if canonical_name == "shell" and harnesses.is_google_harness(
                    harness):
                if "manage_task" not in tools_list:
                    tools_list.append("manage_task")
            tool_map[canonical_name] = tools_list
    return tool_map


CLAUDE_TOOL_MAP: dict[str, list[str]] = load_harness_tool_map("claude-code")

# Two tools for a delegating agent, per how agy behaves headless (`-p`,
# which is how the eval runs):
# - `invoke_subagent` returns a handle; the result later arrives as a message
#   that wakes an idle parent.
# - `manage_subagents` stays for stopping a subagent, not for waiting.
# - But a headless run drops a *root* agent that stays idle for 30 minutes
#   (changelog 1.2.9; not exposed by any flag, setting or env var, and not
#   lifted by `--print-timeout`), with a clean exit and partial output. Short
#   delegations pass; long ones are silently abandoned. So the root (the
#   skill, and its agent twin that `--agent` starts from) is also granted
#   `schedule` and arms a heartbeat on the subagent id that wakes it before
#   the cap and cancels itself when the result arrives. Nested delegators
#   need none: their idleness ends nothing. Staying in the turn and reading
#   the transcript instead is a context-filling busy-wait.
GOOGLE_DELEGATION_TOOLS = ["invoke_subagent", "manage_subagents"]

# Root heartbeat interval: a 2x margin under the 30-minute cap, so an
# hour-long delegation costs a few near-empty turns. 10 minutes was tested.
ANTIGRAVITY_HEARTBEAT_SECONDS = 900

# Granted to the root alone: the heartbeat, and `manage_task` to cancel it
# when a subagent fails -- a headless run waits for a pending timer, so one
# left armed holds the run open until it fires.
GOOGLE_ROOT_TOOLS = ["schedule", "manage_task"]


def is_root_agent(manifest: dict[str, Any], agent: dict[str, Any]) -> bool:
    """Report whether `agent` is a skill's twin, the agent a run starts from.

    A skill that delegates is also built as an agent of the same name
    (`--agent <skill>`), and that is how the eval launches the root: its
    idleness is what ends a headless run, so it alone carries the heartbeat
    and the tool to arm it.
    """
    return any(skill["name"] == agent["name"]
               for skill in manifest.get("skills", []))


# A harness that runs every agent in one context, by construction. Cursor is
# the case: `translators.translate_cursor` inlines each agent into one rules
# file as a persona, so there is no subagent to call and saying otherwise
# would be a lie.
DELEGATION_NONE = "@none"

# A harness whose delegation call has not been confirmed. The contract stays
# silent about the mechanism rather than naming one that may not exist, and the
# build warns. This is a placeholder, not a resting state: an unnamed mechanism
# is what lets a model fall back to running an agent's instructions inline.
DELEGATION_UNVERIFIED = "@unverified"

# Harnesses whose delegation call the runtime intercepts but nobody has watched
# a model actually make. The name is known; that it works is not. Naming an
# unconfirmed mechanism in a contract is worse than saying nothing, because a
# model that tries it and fails falls back to running the agent inline.
UNVERIFIED_DELEGATION_HARNESSES: frozenset[str] = frozenset({"codex"})

# Harnesses that run every agent in one context by construction. Cursor is the
# case: `translators.translate_cursor` inlines each agent into one rules file
# as a persona, so there is no subagent to call and saying otherwise would be a
# lie. The delegate mapping still lists a tool for it; that tool is for
# intercepting a call this build never emits.
INLINING_HARNESSES: frozenset[str] = frozenset({"cursor"})


def load_harness_delegation_tools() -> dict[str, str]:
    """Read the delegation tool per harness from the shared tool mapping.

    `tool-mappings.json` already answers this for the runtime, which matches
    hook events against the same tool names. Deriving the compiler's answer
    from it keeps one registry: a second copy here would name the same tools
    today and silently disagree the first time a harness is added to one and
    not the other.
    """
    blocks = ((utils.get_tool_mappings() or {}).get("delegate")
              or {}).get("harnesses") or {}
    tools: dict[str, str] = {}
    for harness, spec in blocks.items():
        if harness in INLINING_HARNESSES:
            tools[harness] = DELEGATION_NONE
        elif harness in UNVERIFIED_DELEGATION_HARNESSES:
            tools[harness] = DELEGATION_UNVERIFIED
        else:
            names = spec.get("tools") or []
            if names:
                tools[harness] = names[0]

    # A Google harness that inherits its mapping block inherits the tool named
    # in it. Leaving it out here would build a package whose runtime
    # intercepts a delegation the contract never names.
    for harness in harnesses.GOOGLE_HARNESSES:
        inherited = harnesses.harness_mapping(blocks,
                                              harness).get("tools") or []
        if harness not in tools and inherited:
            tools[harness] = inherited[0]
    return tools


# Which tool call actually performs a delegation, per harness. A harness absent
# from this map cannot be built once anything routes to an agent -- see
# `delegation_tool_for`. Adding a harness means answering this question, and the
# place to answer it is `tool-mappings.json`.
HARNESS_DELEGATION_TOOLS: dict[str, str] = load_harness_delegation_tools()

CLAUDE_DELEGATION_TOOL: str = HARNESS_DELEGATION_TOOLS["claude-code"]


def delegation_tool_for(harness: str, delegates: list[str]) -> str:
    """Return the tool that performs a delegation on `harness`.

    Empty when nothing is delegated. Raises when work *is* routed to an agent
    but the harness has no entry: shipping that package produces instructions
    the runtime cannot honor, and the model's remaining move is to read the
    agent's definition and run it inline -- which silently collapses the
    isolation the delegation existed to provide. Failing the build is cheaper
    than discovering it in a transcript.
    """
    if not delegates:
        return ""
    tool = HARNESS_DELEGATION_TOOLS.get(harness)
    if not tool:
        raise utils.TpuAgentSuiteError(
            f"harness {harness} has no delegation mechanism declared, but"
            f" delegation to {sorted(delegates)} was requested; add it to"
            " HARNESS_DELEGATION_TOOLS or drop the routing")
    if tool == DELEGATION_UNVERIFIED:
        print(
            f"WARNING: harness {harness} delegates to {sorted(delegates)} but"
            " its delegation tool is unverified; the generated contract cannot"
            " name how to delegate.",
            file=sys.stderr,
        )
    return tool


def delegation_mechanism_lines(harness: str,
                               delegates: list[str],
                               *,
                               root: bool = False,
                               shell: bool = True) -> str:
    """Render the contract line naming *how* to delegate, not just to whom.

    `root` marks the agent whose idleness ends a headless run (the skill, on
    Antigravity); only it carries the heartbeat.

    Naming targets alone leaves the mechanism to be guessed, and the guess that
    costs the most is doing the work inline: it looks like progress, produces a
    plausible answer, and quietly spends the orchestrator's context on output
    that was supposed to stay behind an agent boundary.

    How to wait is the second guess, and the right answer differs by harness,
    so this is the one place that states it: a skill body is compiled for
    every harness and cannot know which one it is on.
    """
    tool = delegation_tool_for(harness, delegates)
    if not tool or tool == DELEGATION_UNVERIFIED:
        return ""
    if tool == DELEGATION_NONE:
        return ("\n- This harness runs every agent in one context: there is no"
                " subagent to call. Adopt each persona's contract yourself,"
                " in sequence.")
    how = (f"\n- Delegate by calling the `{tool}` tool. Reading an agent's"
           " definition and following it yourself is not delegation: it"
           " discards the isolation delegating exists to provide, and is"
           " never the fallback when the call seems unavailable.")
    if not harnesses.is_google_harness(harness):
        # How to wait on a handle is left to the agent: the idle gate refuses
        # a stop that would deliver the pause as the result, and says what to
        # do. Claude Code is foreground by default and returns the report;
        # only `run_in_background` returns a handle. Naming both keeps a
        # headless subagent from backgrounding everything and meeting the
        # gate. Other harnesses are unobserved and keep the handle-only line.
        if harness == "claude-code" and not shell:
            # Headless, a background child's report reaches its parent only
            # with a tool result the parent requests; an agent without a
            # shell has no call to make, so it launches in the foreground.
            # The runtime refuses a background launch from such an agent.
            returns = (
                f"\n- `{tool}` runs the subagent and returns its report as"
                " the tool result. Launch in the foreground: without a"
                " shell you cannot wait for background work.")
        elif harness == "claude-code":
            returns = (
                f"\n- `{tool}` runs the subagent and returns its report as"
                " the tool result. With `run_in_background: true` it returns"
                " a handle instead and the report reaches you later.")
        else:
            returns = (f"\n- `{tool}` returns a handle, not a result, while"
                       " the work runs; the result reaches you later.")
        return how + returns + (
            "\n- A launch that fails with `Concurrent subagent limit reached`"
            " started nothing: launch it again once one of your subagents"
            " has reported."
            "\n- Delegate in whole pieces: handing over the job costs one call"
            " plus what you choose to read.")
    # Antigravity: the call returns a handle while the work runs and the
    # result arrives as a message that wakes the parent. A headless run drops
    # an idle *root* after 30 minutes, so the root alone keeps a heartbeat;
    # see the note above GOOGLE_DELEGATION_TOOLS.
    wait = (
        f"\n- `{tool}` returns a handle (the subagent's conversation id), not"
        " a result. The result arrives later as a message that wakes you."
        " A subagent that dies sends no result, only a system message that"
        " it went idle without reporting, stopped, or failed to start: that"
        " message is its result, a failure, not a result still to come.")
    if root:
        wait += (
            "\n- Before ending the turn, call `schedule` with"
            f" `DurationSeconds: {ANTIGRAVITY_HEARTBEAT_SECONDS}` and"
            " `TimerCondition` set to that conversation id: a headless run"
            " drops a root agent idle for 30 minutes, and this wakes you"
            " first. It cancels itself when the result arrives."
            "\n- Then end your turn and wait idle: never read the subagent's"
            " transcript, and never act on a result that has not arrived."
            " What wakes you decides what you do. The result: act on it. A"
            " failure message: cancel the heartbeat with `manage_task`"
            " (`Action: kill`, the task id `schedule` returned) -- otherwise"
            " the run stays open until it fires -- then report. The"
            " heartbeat, with no result yet: one `manage_subagents` call"
            " (`Action: list`), then end the turn either way -- still running,"
            " re-arm the heartbeat first; not running, report it as stopped"
            " without a result and do not re-arm. Beyond that,"
            " `manage_subagents` is only for stopping a subagent no longer"
            " needed.")
    else:
        wait += (
            "\n- Then end your turn and wait idle: never read the subagent's"
            " transcript, never check on it, and never act on a result that"
            " has not arrived. Act on the result when its message wakes you."
            " Use `manage_subagents` only to stop a subagent no longer"
            " needed.")
    return how + wait + (
        "\n- Delegate in whole pieces -- say what to produce and how it will be"
        " checked -- since you will not see intermediate steps.")


def map_agent_tools(
    resolved_tools: list[dict[str, Any]],
    tool_map: dict[str, list[str]],
) -> list[str]:
    """Map an agent's resolved tools into one harness's native vocabulary.

    Order is the declaration order of the agent's tools, de-duplicated: a
    harness config is easier to review when it does not reshuffle between
    builds. Script and MCP tools pass through under their own names.
    """
    tools: list[str] = []
    for tool in resolved_tools:
        if tool["category"] == "native":
            for native in tool_map.get(tool["name"], [tool["name"]]):
                if native not in tools:
                    tools.append(native)
        elif tool["category"] in ("script", "mcp"):
            if tool["name"] not in tools:
                tools.append(tool["name"])
    return tools


def claude_frontmatter(agent: dict[str, Any],
                       resolved_tools: list[dict[str, Any]]) -> str:
    """Generate Claude agent YAML frontmatter with mapped native and custom tools."""
    tools = map_agent_tools(resolved_tools, CLAUDE_TOOL_MAP)

    if agent.get("delegates"):
        tools.append(CLAUDE_DELEGATION_TOOL)

    # Claude Code reads it from an agent's own definition; the Agent tool has
    # no per-call effort. Other harnesses get nothing until an equivalent is
    # confirmed, and run the agent at their default.
    optional = f"effort: {agent['effort']}\n" if agent.get("effort") else ""
    return ("---\n"
            f"name: {agent['name']}\n"
            f"description: {agent['description']}\n"
            f"tools: {', '.join(tools)}\n"
            f"{optional}"
            "background: false\n"
            "---\n\n")


def google_harness_frontmatter(
    agent: dict[str, Any],
    resolved_tools: list[dict[str, Any]],
    tool_map: dict[str, list[str]],
    *,
    root: bool = False,
) -> str:
    """Generate Google harness (Antigravity) YAML frontmatter."""
    tools = map_agent_tools(resolved_tools, tool_map)

    if agent.get("delegates"):
        tools.extend(t for t in GOOGLE_DELEGATION_TOOLS if t not in tools)
        if root:
            tools.extend(t for t in GOOGLE_ROOT_TOOLS if t not in tools)

    tools_yaml = "\n".join(f"  - {t}" for t in tools)
    return ("---\n"
            f"name: {agent['name']}\n"
            f'description: "{agent["description"]}"\n'
            f"tools:\n{tools_yaml}\n"
            "subagent: true\n"
            "---\n\n")


def toml_basic_string(value: str) -> str:
    """Escape a value for a TOML basic (double-quoted) string."""
    return (value.replace("\\", "\\\\").replace('"', '\\"').replace(
        "\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))


def toml_multiline_string(value: str) -> str:
    """Escape a value for a TOML multi-line basic string.

    Prompts are authored as markdown and routinely contain backslashes -- an
    escaped paren such as `1\\)` carried over from the source package is
    enough. TOML reads `\\` as the start of an escape sequence, so an
    unescaped prompt produces an agent config that does not parse at all.
    Backslashes are doubled first, then any `\"\"\"` that would otherwise
    terminate the literal early.
    """
    return value.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def codex_sandbox_mode(resolved_tools: list[dict[str, Any]]) -> str:
    """Determine Codex sandbox mode based on action_levels of resolved tools."""
    has_host_or_exec = any(
        t.get("action_level", {}).get("scope") == "host"
        or t.get("action_level", {}).get("effect") == "execute"
        or t.get("name") == "shell" for t in resolved_tools)
    if has_host_or_exec:
        return "danger-unrestricted"

    has_workspace_write = any(
        (t.get("action_level", {}).get("scope") == "workspace"
         and t.get("action_level", {}).get("effect") in ("write", "delete"))
        or t.get("name") in ("write", "edit") for t in resolved_tools)
    if has_workspace_write:
        return "workspace-write"

    return "read-only"


def codex_approval_policy(resolved_tools: list[dict[str, Any]]) -> str:
    """Determine Codex interactive approval policy based on tool action levels."""
    if any(
            t.get("action_level", {}).get("approval") == "always"
            for t in resolved_tools):
        return "always"
    if any(
            t.get("action_level", {}).get("approval") == "on-demand"
            for t in resolved_tools):
        return "on-demand"
    return "never"
