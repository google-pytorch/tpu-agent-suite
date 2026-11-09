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
"""Trajectory checks for tpu-model-onboarding's preflight sequence.

Every preflight-* task carries a copy of this file. Everything above the
TASK CRITERIA marker is shared and must stay identical across tasks; below it,
each task declares the `@criterion` functions it scores.

preflight-check.md runs four steps in order: resolve the name
(`--resolve`), the Hub-side checks on the host (`--hub_only`), the image
probe on the accelerator (`image_probe.py` through `compute run`), and the
full check where transformers is installed. These checks read which of those
commands ran, where, and in what order. An agent often chains steps in one
shell call, so each call is read statement by statement.

The commands run inside the tpu-model-onboarding subagent. Harbor's Claude Code
trajectory includes subagent steps, in time order. Its Antigravity trajectory
holds only the main session, so the checks also read Antigravity's own
per-session step transcripts, a subagent's included. Tool names and argument
keys are matched for Claude Code, Antigravity and Codex alike. Prose -- the verdict and how it is reported -- is the judge's to
score.
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from typing import Any

from rewardkit import criterion

_CMD_TOOLS = {
    "bash", "run_shell_command", "run_command", "execute_command", "sh",
    "terminal", "exec", "shell"
}
_CMD_KEYS = ("command", "commandline", "cmd", "command_line")
_SUBAGENT_TOOLS = {
    "agent", "task", "invoke_subagent", "subagent", "delegate", "spawn_agent"
}
# Where each harness names the agent a delegation starts.
_TARGET_KEYS = ("subagent_type", "subagent", "agent_name", "type_name",
                "typename", "role")
# Antigravity's step transcripts, one per session, subagents' included.
_AGY_TRANSCRIPTS = (
    "/root/.gemini/antigravity-cli/brain/*/.system_generated/logs/transcript*.jsonl",
    "/logs/agent/**/transcript*.jsonl")
# Where a run's artifacts can land: `{{runs}}` in the home directory, or a
# workspace-scoped TPU agent suite tree.
_ARTIFACT_GLOBS = ("/root/.tpu-agent-suite/runs/*/artifacts",
                   "/workspace/.tpu-agent-suite/runs/*/artifacts",
                   "/workspace/**/artifacts")

_PREFLIGHT = "preflight_check.py"
_PROBE = "image_probe.py"
# `{{runtime}} compute` renders as `compute --package <tag>`, so options can
# stand between `compute` and `run`.
_COMPUTE_RUN = re.compile(
    r"\bcompute(?:\s+--?[\w-]+(?:(?:=|\s+)(?!-)\S+)?)*\s+run\b")
_INSTALL_TRANSFORMERS = re.compile(
    r"\b(pip3?|uv\s+pip)\s+install\b[^\n;&|]*\btransformers\b")
# One shell call often runs several steps; these split it into statements.
_STATEMENT_BREAK = re.compile(r"\n|&&|\|\||;")
# NAME=value, the value bare or quoted: R="python3 runtime.py compute ...".
_ASSIGNMENT = re.compile(
    r"""^(?:export\s+)?([A-Za-z_]\w*)=(?:"([^"]*)"|'([^']*)'|(\S*))$""")
_VARIABLE = re.compile(r"\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))")


def load_trajectory(workspace: Path) -> dict[str, Any]:
    """The run's trajectory, from wherever the harness left it."""
    candidates = [
        os.environ.get("HARBOR_TRAJECTORY_PATH", ""),
        "/logs/agent/trajectory.json", "/logs/trajectory.json",
        str(workspace / "agent" / "trajectory.json"),
        str(workspace / "trajectory.json")
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            try:
                return json.loads(Path(candidate).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
    return {}


def load_agy_transcripts() -> list[dict[str, Any]]:
    """Tool calls from Antigravity's step transcripts, file by file in order."""
    calls: list[dict[str, Any]] = []
    paths = sorted({
        p
        for pattern in _AGY_TRANSCRIPTS
        for p in glob.glob(pattern, recursive=True)
    })
    for path in paths:
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                calls.extend(tc for tc in record.get("tool_calls") or []
                             if isinstance(tc, dict))
    return calls


def _call(name: Any, args: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"raw": args}
    return str(name or "").lower(), args if isinstance(args, dict) else {}


def tool_calls(workspace: Path) -> list[tuple[str, dict[str, Any]]]:
    """(lower-cased tool name, arguments) for every call.

    The trajectory's calls in order, then Antigravity's transcripts' calls,
    each transcript in its own order. Preflight runs in one subagent, so the
    order of its commands is kept either way.
    """
    raw: list[dict[str, Any]] = []
    trajectory = load_trajectory(workspace)
    for step in trajectory.get("steps") or []:
        if isinstance(step, dict):
            raw.extend(tc for tc in step.get("tool_calls") or []
                       if isinstance(tc, dict))
    raw.extend(load_agy_transcripts())
    return [
        _call(
            tc.get("function_name") or tc.get("name"),
            tc.get("arguments") or tc.get("args") or tc.get("input"))
        for tc in raw
    ]


def statements(command: str) -> list[str]:
    """The statements of one shell call, in order.

    Line continuations are joined, and a variable set by an assignment
    (`S=.../preflight_check.py`, or quoted, `R="python3 runtime.py compute"`)
    is expanded in the statements after it, so `python3 $S --resolve=...`
    still names the script and `$R run ...` still runs on the accelerator.
    """
    env: dict[str, str] = {}
    found = []
    for part in _STATEMENT_BREAK.split(command.replace("\\\n", " ")):
        part = _VARIABLE.sub(
            lambda m: env.get(m.group(1) or m.group(2), m.group(0)),
            part.strip())
        assigned = _ASSIGNMENT.match(part)
        if assigned:
            env[assigned.group(1)] = next(g for g in assigned.groups()[1:]
                                          if g is not None)
        elif part:
            found.append(part)
    return found


def commands(workspace: Path) -> list[str]:
    """Every shell statement the agent, or a subagent, ran, in order."""
    found = []
    for name, args in tool_calls(workspace):
        if name in _CMD_TOOLS:
            for k, v in args.items():
                if k.lower() in _CMD_KEYS and v:
                    found += statements(str(v))
    return found


def delegation_targets(workspace: Path) -> list[str]:
    """The agent each delegation started, in order; "" when none is named."""
    targets = []
    for name, args in tool_calls(workspace):
        if name not in _SUBAGENT_TOOLS:
            continue
        named = [
            str(v) for k, v in args.items()
            if k.lower() in _TARGET_KEYS and isinstance(v, str) and v
        ]
        # Antigravity starts several at once, as a list of subagents.
        for sub in args.get("Subagents") or args.get("subagents") or []:
            if isinstance(sub, dict):
                named += [
                    str(v) for k, v in sub.items()
                    if k.lower() in _TARGET_KEYS and isinstance(v, str) and v
                ]
        targets += [n.lower().replace("_", "-") for n in named] or [""]
    return targets


def first_index(cmds: list[str], matches) -> int | None:
    """Position of the first command `matches` accepts, if any."""
    return next((i for i, c in enumerate(cmds) if matches(c)), None)


def is_resolve(cmd: str) -> bool:
    return _PREFLIGHT in cmd and "--resolve" in cmd


def is_hub_only(cmd: str) -> bool:
    return _PREFLIGHT in cmd and "--hub_only" in cmd


def is_full_check(cmd: str) -> bool:
    """The full check: a --model run that is neither --hub_only nor --resolve."""
    return (_PREFLIGHT in cmd and "--model" in cmd and "--hub_only" not in cmd
            and "--resolve" not in cmd)


def on_accelerator(cmd: str) -> bool:
    return bool(_COMPUTE_RUN.search(cmd))


def is_probe(cmd: str) -> bool:
    return _PROBE in cmd and on_accelerator(cmd)


def installed_transformers(cmd: str) -> bool:
    return bool(_INSTALL_TRANSFORMERS.search(cmd))


def artifact(name_glob: str, nested: bool = False) -> list[Path]:
    """Artifact files of this run matching `name_glob`.

    At the artifacts root, where another agent reads them, unless `nested`:
    then in any directory under it too, for a file only its writer reads.
    """
    pattern = f"**/{name_glob}" if nested else name_glob
    return sorted({
        Path(p)
        for root in _ARTIFACT_GLOBS
        for p in glob.glob(f"{root}/{pattern}", recursive=True)
        if Path(p).is_file()
    })


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# ---- TASK CRITERIA: preflight-ambiguous-name --------------------------------
#
# "llama 3 8b" names several official repositories (Llama 3 and 3.1, base and
# instruct). Preflight resolves the name, finds it ambiguous, and hands the
# question back; no check runs on a repository it guessed.


@criterion(description="Preflight resolved the loose name with --resolve")
def resolved_name(workspace: Path) -> bool:
    return any(
        is_resolve(c) and "llama" in c.lower() for c in commands(workspace))


@criterion(description="No check ran on a repository chosen without the user")
def no_check_on_a_guess(workspace: Path) -> bool:
    return not any(
        is_hub_only(c) or is_full_check(c) or on_accelerator(c)
        for c in commands(workspace))
