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
"""Per-harness hook configuration.

Compiles a manifest's hooks into each harness's hook settings, with the
fail-safe shell fallback, and the skill tool restrictions the skill-tools
hook enforces.
"""

from __future__ import annotations

import shlex
import sys
from typing import Any

from framework import harnesses, paths, utils
from framework.builder import package_checks, package_loader

# Margin the harness allows a hook over the hook's own timeout. hook-dispatch
# enforces the declared timeout on the action it runs, but finishes a little
# after it: interpreter start and event normalization come first, the verdict
# after. Given the same deadline, the harness kills dispatch before that
# verdict is written -- a hung hook's deny never arrives, and Claude Code runs
# the tool anyway. About 0.1s on an idle host, more on a loaded one; it costs
# nothing on the ordinary path, since the harness waits only for dispatch to
# exit, and bounds dispatch itself should it hang.
HOOK_DISPATCH_HEADROOM_S = 5

# Exit code of the compiled hook command when hook-dispatch itself exits
# non-zero. Dispatch always exits 0 with its verdict in JSON, so a non-zero
# exit is never a verdict, and the policy decides. `deny` is 2, the one code
# Claude Code, Cursor and Codex block on (others fail open); Antigravity
# refuses tool hooks on any non-zero. The policy applies on tool events only,
# where a block is one refusal: on Stop and SubagentStop a block means "keep
# going", with no way out once dispatch cannot run, and session events carry
# no decision, so those exit 0 whatever the policy.
HOOK_FAIL_SAFE_EXIT_CODES = {"allow": 0, "deny": 2}

HOOK_FAIL_SAFE_POLICY_EVENTS = frozenset({"PreToolUse", "PostToolUse"})


def fail_safe_hook_command(dispatch_cmd: str, hook_name: str, on_failure: str,
                           event: str) -> str:
    """Append a shell fallback applying `on_failure` if dispatch exits non-zero.

    Needs nothing installed, so it works when `runtime.py` is gone. The
    stderr line is the only trace of a fail-open: the dispatcher's failure
    log cannot be written by a dispatcher that never started. `event` is
    the canonical event; outside HOOK_FAIL_SAFE_POLICY_EVENTS the fallback
    exits 0 whatever the policy, and the line says so.
    """
    if on_failure not in HOOK_FAIL_SAFE_EXIT_CODES:
        raise utils.TpuAgentSuiteError(
            f"hook '{hook_name}' on_failure must be one of "
            f"{list(HOOK_FAIL_SAFE_EXIT_CODES)}, got {on_failure!r}")
    if event in HOOK_FAIL_SAFE_POLICY_EVENTS:
        exit_code = HOOK_FAIL_SAFE_EXIT_CODES[on_failure]
        after = f"before a verdict; on_failure={on_failure}"
    else:
        exit_code = 0
        after = (f"before a verdict; allowing: on_failure={on_failure} would "
                 f"loop on {event}")
    # `"$?"` is its own argument so it still holds the dispatcher's exit;
    # the literal parts are single-quoted. Hook names are schema-restricted
    # to `[a-zA-Z0-9_.-]`.
    before = shlex.quote(
        f"tpu-agent-suite: hook {hook_name}: hook-dispatch exited")
    return (f'{dispatch_cmd} || {{ echo {before} "$?" {shlex.quote(after)} '
            f">&2; exit {exit_code}; }}")


# How each harness handles tool restrictions declared in a skill's frontmatter.
# Claude Code enforces them with the skill-tools hook, which limits the main
# thread only. Other harnesses can't enforce them, so the build warns instead
# of silently dropping the restriction. A skill that delegates also ships as
# an agent of the same name, and its `tools` there are the agent's: a harness
# started with `--agent <skill>` (Antigravity's runner starts tpu-agent
# that way) enforces them as it does any agent's.
#
# Not Claude Code's own `disallowed-tools` skill field: that removes the tools
# from the session's pool for the rest of the turn, and subagents started in
# that turn draw from the same pool. A headless trial is one turn, so every
# specialist tpu-agent delegated to ran without Bash.
SKILL_TOOLS_HOOK = "hook"

SKILL_TOOLS_UNENFORCED = "unenforced"

SKILL_TOOL_RESTRICTION: dict[str, str] = {
    "claude-code": SKILL_TOOLS_HOOK,
    "codex": SKILL_TOOLS_UNENFORCED,
    "antigravity": SKILL_TOOLS_UNENFORCED,
    "cursor": SKILL_TOOLS_UNENFORCED,
}


def skill_tools_enforced(skill: dict[str, Any], harness: str) -> bool:
    """Whether the harness enforces the skill's `tools` limit; warns if it can't."""
    granted = skill.get("tools") or []
    if not granted:
        return False
    mode = SKILL_TOOL_RESTRICTION.get(harness)
    if mode is None:
        raise utils.TpuAgentSuiteError(
            f"Skill '{skill['name']}' declares tools {sorted(granted)}, but harness "
            f"'{harness}' is missing from SKILL_TOOL_RESTRICTION.")
    if mode == SKILL_TOOLS_UNENFORCED:
        print(
            f"WARNING: Skill '{skill['name']}' declares tools {sorted(granted)}, but harness "
            f"'{harness}' cannot enforce tool restrictions on skills. All tools will remain available.",
            file=sys.stderr,
        )
        return False
    return True


# Tools the skill-tools hook always allows, as the hook sees them (portable
# names for mapped tools, native names otherwise). None run commands or edit files.
SKILL_ALWAYS_ALLOWED_TOOLS: dict[str, list[str]] = {
    "claude-code": [
        "delegate",
        "Skill",
        "ToolSearch",
        "TodoWrite",
        "TaskCreate",
        "TaskGet",
        "TaskList",
        "TaskUpdate",
        "TaskOutput",
    ],
}


def hook_allowed_skill_tools(manifest: dict[str, Any],
                             harness: str) -> dict[str, list[str] | None]:
    """Map each skill to the tool names the skill-tools hook allows (None = no limit)."""
    enforced = SKILL_TOOL_RESTRICTION.get(harness) == SKILL_TOOLS_HOOK
    always = SKILL_ALWAYS_ALLOWED_TOOLS.get(harness, [])
    return {
        skill["name"]: (sorted({*skill["tools"], *always})
                        if enforced and skill.get("tools") else None)
        for skill in manifest["skills"]
    }


def build_harness_hooks_config(
    package_name: str,
    manifest: dict[str, Any],
    harness: str,
    include_internal: bool = False,
) -> dict[str, Any]:
    """Generate harness-specific hook definitions for lifecycle events and tool interception."""
    hooks_dict: dict[str, Any] = {}
    for hook in manifest.get("hooks", []):
        if package_loader.excluded_internal(hook, include_internal):
            continue
        # Skip the skill-tools hook on harnesses that can't enforce skill tools.
        if (hook.get("action") == "builtin:skill-tools"
                and SKILL_TOOL_RESTRICTION.get(harness) != SKILL_TOOLS_HOOK):
            continue

        hook_name = hook["name"]
        event = hook["event"]
        action = hook.get("action", "")
        timeout = hook.get("timeout", utils.DEFAULT_HOOK_TIMEOUT)

        pkg_version = manifest.get("version", "")
        pkg_tag = utils.format_package_tag(package_name, pkg_version)
        # The payload lives only in the central store, so hook commands are
        # home-anchored: correct from any cwd, expanded by the shell at run
        # time on the target machine.
        runtime_prefix = paths.TPU_AGENT_SUITE_HOME_SPELLING

        # Map event name to harness-specific lifecycle event identifier
        harness_event = harnesses.to_harness_event(event, harness)

        matchers = hook.get("matchers", [])
        package_checks.check_hook_matchers(hook_name, event, matchers)
        if not matchers:
            matcher_str = "*"
        elif event in ("PreToolUse", "PostToolUse"):
            tool_mappings = utils.get_tool_mappings()
            mapped = [
                harnesses.harness_mapping(
                    tool_mappings.get(m, {}).get("harnesses", {}),
                    harness).get("matcher", m) for m in matchers
            ]
            matcher_str = "|".join(mapped)
        else:
            # SubagentStop, the one other event
            # package_checks.check_hook_matchers allows.
            matcher_str = "|".join(matchers)

        # Always spelled out, default included: the command line is the one
        # place the policy is read from when dispatch cannot start.
        on_failure = hook.get("on_failure", utils.DEFAULT_HOOK_ON_FAILURE)
        on_failure_args = f" --on-failure {on_failure}"

        base_cmd = (
            f"python3 {runtime_prefix}/packages/{pkg_tag}/framework/runtime.py hook-dispatch "
            f"--harness {harness} --package {pkg_tag} --hook {hook_name} "
            f"--event {event} --timeout {timeout}{on_failure_args}")

        if action.startswith("builtin:"):
            builtin_name = action[len("builtin:"):]
            action_args = f" --builtin {shlex.quote(builtin_name)}"
        elif (script_rel := package_loader.hook_script(action)) is not None:
            action_args = f" --script {script_rel}"
            if hook.get("args"):
                escaped_args = " ".join(shlex.quote(a) for a in hook["args"])
                action_args = f"{action_args} -- {escaped_args}"
        elif hook.get("command"):
            raw_cmd = hook["command"]
            if hook.get("args"):
                escaped_args = " ".join(shlex.quote(a) for a in hook["args"])
                raw_cmd = f"{raw_cmd} {escaped_args}"
            action_args = f" --command {shlex.quote(raw_cmd)}"
        else:
            raise utils.TpuAgentSuiteError(
                f"hook '{hook_name}' must specify either 'action' (script: or builtin:) or 'command'"
            )

        # The flag that makes the runtime filter by matcher itself, for
        # registrations whose native matcher cannot select the subagent.
        emulated_matcher_args = ("" if matcher_str == "*" else
                                 f" --matcher {shlex.quote(matcher_str)}")

        # Emulate matcher filtering when host engine does not support matchers
        # natively (e.g. Stop on the Google harnesses)
        if harnesses.is_google_harness(harness) and harness_event == "Stop":
            dispatch_cmd = f"{base_cmd}{emulated_matcher_args}{action_args}"
        else:
            dispatch_cmd = f"{base_cmd}{action_args}"

        # The policy is applied by dispatch, which has to be running for
        # that. The shell applies it when dispatch is not.
        hook_entry = {
            "type":
            "command",
            "command":
            fail_safe_hook_command(dispatch_cmd, hook_name, on_failure, event),
            "timeout":
            timeout + HOOK_DISPATCH_HEADROOM_S,
        }

        if hook_name not in hooks_dict:
            hooks_dict[hook_name] = {}
        if harness_event not in hooks_dict[hook_name]:
            hooks_dict[hook_name][harness_event] = []

        # On the Google harnesses, 'PreInvocation', 'PostInvocation', and 'Stop' events
        # use a flat array of handler objects in hooks.json because they do not target specific tools.
        # Tool events ('PreToolUse', 'PostToolUse'), as well as other harnesses, require
        # grouping handlers under a 'matcher' object (e.g. {"matcher": "...", "hooks": [...]}).
        # Wrapping a loop-level handler in a matcher group makes it silently
        # never fire, which is why this is a shape difference and not a detail.
        if harnesses.is_google_harness(harness) and harness_event in (
                "Stop", "PreInvocation"):
            hooks_dict[hook_name][harness_event].append(hook_entry)
        else:
            hooks_dict[hook_name][harness_event].append({
                "matcher": matcher_str,
                "hooks": [hook_entry],
            })

        # A subagent's result can reach its caller by more than one road. On
        # Claude Code a foreground subagent ends its turn and the result is
        # its last message (native SubagentStop), but a background or
        # auto-mode subagent must call the injected `SubagentHandback` tool
        # instead, and the text it writes after that call is not delivered.
        # Registering the same hook body on that tool call keeps the package's
        # SubagentStop contract enforced whichever road the harness takes.
        # The command still says `--event SubagentStop`: the runtime recognizes
        # the tool payload and answers it in that event's native shape. It
        # also names that event with `--handback-event`, for a dispatch that
        # fails before it can read the payload: a reply in SubagentStop's
        # shape is rejected by the tool event, and the tool runs. The tool
        # matcher selects the call, so the subagent matchers move into the
        # command, where the runtime emulates them.
        handback_mapping = (harnesses.subagent_handback_mapping(harness)
                            if event == "SubagentStop" else None)
        if handback_mapping:
            handback_args = (" --handback-event "
                             f"{shlex.quote(handback_mapping['event_name'])}")
            hooks_dict[hook_name].setdefault(
                handback_mapping["event_name"], []).append({
                    "matcher":
                    handback_mapping["matcher"],
                    "hooks": [{
                        **hook_entry,
                        "command":
                        fail_safe_hook_command(
                            f"{base_cmd}{handback_args}"
                            f"{emulated_matcher_args}{action_args}", hook_name,
                            on_failure, event),
                    }],
                })

    return hooks_dict
