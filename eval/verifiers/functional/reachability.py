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
"""Whether tpu-agent's delegation to tpu-knowledge-base reached a running subagent.

A delegation call proves nothing by itself: a harness that cannot find the
agent still records the call, and answers it with an error. The routing
checks elsewhere count calls, so a run whose every delegation failed with
"not found or not allowed to be invoked" scored as routed. This check needs
evidence that the subagent ran -- a session of its own, or a command only it
can run -- and no not-found error anywhere in the run.
"""

import glob
import json
import os
import re
from pathlib import Path
from typing import Any

from rewardkit import criterion

_TARGET = "tpu-knowledge-base"
_SUBAGENT_TOOLS = {
    "agent", "task", "invoke_subagent", "subagent", "delegate", "spawn_agent"
}
_TARGET_KEYS = ("subagent_type", "subagent", "agent_name", "type_name",
                "typename", "role")
# Antigravity: one step transcript per session, the parent's and each
# subagent's, each in its own brain directory.
_AGY_TRANSCRIPTS = (
    "/root/.gemini/antigravity-cli/brain/*/.system_generated/logs/transcript*.jsonl"
)
# What each harness says when the named agent is not available to it.
_UNREACHABLE = re.compile(
    r"not found or not allowed to be invoked"
    r"|agent type ['\"]?[\w-]+['\"]? not found", re.IGNORECASE)
# Claude Code writes a subagent's messages as a sidechain of the session.
_CLAUDE_SIDECHAIN = re.compile(r'"is_?sidechain"\s*:\s*true', re.IGNORECASE)
# tpu-agent holds no shell tool (enforced by skill-tools), so only the
# tpu-knowledge-base subagent can run kb_lookup.py.
_SHELL_TOOLS = {"bash", "shell", "run_command", "exec_command"}
_KB_LOOKUP = "kb_lookup.py"
_LOG_SUFFIXES = (".json", ".jsonl", ".log", ".txt")


def _logs_dir() -> Path:
    """The run's logs as the harness lays them out: $LOGS_DIR, else /logs."""
    return Path(os.environ.get("LOGS_DIR") or "/logs")


def _texts() -> dict[str, str]:
    """Every log the run left, by path."""
    paths = set(
        glob.glob(f"{glob.escape(str(_logs_dir()))}/agent/**/*",
                  recursive=True))
    paths |= set(glob.glob(_AGY_TRANSCRIPTS))
    texts = {}
    for path in sorted(paths):
        p = Path(path)
        if p.is_file() and p.suffix in _LOG_SUFFIXES:
            try:
                texts[path] = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                pass
    return texts


def _tool_calls(texts: dict[str, str]) -> list[tuple[str, dict[str, Any]]]:
    """(lower-cased name, arguments) of every tool call found in the logs."""
    calls = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            name = node.get("function_name") or node.get("name") or node.get(
                "tool_name")
            args = node.get("arguments") or node.get("args") or node.get(
                "input") or node.get("args_json")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = None
            if isinstance(name, str) and isinstance(args, dict):
                calls.append((name.lower(), args))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    for path, text in texts.items():
        docs = text.splitlines() if path.endswith(".jsonl") else [text]
        for doc in docs:
            try:
                visit(json.loads(doc))
            except json.JSONDecodeError:
                continue
    return calls


def _delegated_to_target(calls: list[tuple[str, dict[str, Any]]]) -> bool:
    for name, args in calls:
        if name not in _SUBAGENT_TOOLS:
            continue
        subs = [args] + [
            s for s in args.get("Subagents") or args.get("subagents") or []
            if isinstance(s, dict)
        ]
        for sub in subs:
            for key, value in sub.items():
                if (key.lower() in _TARGET_KEYS and isinstance(value, str)
                        and value.lower().replace("_", "-") == _TARGET):
                    return True
    return False


def _subagent_ran(texts: dict[str, str],
                  calls: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """The evidence that a subagent session ran, empty when there is none."""
    evidence = []
    sessions = {Path(p).parents[2].name for p in glob.glob(_AGY_TRANSCRIPTS)}
    if len(sessions) >= 2:
        evidence.append(f"{len(sessions)} Antigravity sessions")
    if any(_CLAUDE_SIDECHAIN.search(t) for t in texts.values()):
        evidence.append("Claude Code sidechain messages")
    if any(name in _SHELL_TOOLS and _KB_LOOKUP in json.dumps(args)
           for name, args in calls):
        evidence.append("kb_lookup.py shell call")
    return evidence


@criterion(description=(
    "tpu-agent delegated to tpu-knowledge-base, the subagent ran, and no"
    " delegation failed with the agent not found"))
def subagent_reachable(workspace: Path) -> bool:
    texts = _texts()
    calls = _tool_calls(texts)
    delegated = _delegated_to_target(calls)
    ran = _subagent_ran(texts, calls)
    unreachable = sorted(
        {m.group(0)
         for t in texts.values()
         for m in _UNREACHABLE.finditer(t)})
    print(f"[subagent-reachability] delegated to {_TARGET}: {delegated}")
    print(f"[subagent-reachability] subagent ran: {ran or 'no evidence'}")
    print(f"[subagent-reachability] not-found errors: {unreachable or 'none'}")
    return delegated and bool(ran) and not unreachable
