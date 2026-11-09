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
"""Deterministic result checks for nightly-functional/numerical-parity."""

from __future__ import annotations

import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from rewardkit import criterion

_CMD_TOOLS = {
    "bash",
    "run_shell_command",
    "run_command",
    "execute_command",
    "sh",
    "terminal",
    "exec",
    "shell",
}
_CMD_KEYS = ("command", "commandline", "cmd", "command_line")
_AGY_TRANSCRIPTS = (
    "/root/.gemini/antigravity-cli/brain/*/.system_generated/logs/transcript*.jsonl",
    "/logs/agent/**/transcript*.jsonl",
)
_ARTIFACT_GLOBS = (
    "/root/.tpu-agent-suite/runs/*/artifacts",
    "/workspace/.tpu-agent-suite/runs/*/artifacts",
    "/workspace/**/artifacts",
    "/logs/artifacts",
)
_REMOTE_EXEC = re.compile(
    r"\b(compute(?:\s+--?[\w-]+(?:(?:=|\s+)(?!-)\S+)?)*\s+run|kubectl\s+(?:apply|create|exec))\b"
)
_STATEMENT_BREAK = re.compile(r"\n|&&|\|\||;")
_ASSIGNMENT = re.compile(
    r"""^(?:export\s+)?([A-Za-z_]\w*)=(?:"([^"]*)"|'([^']*)'|(\S*))$""")
_VARIABLE = re.compile(r"\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))")

_EXPECTED_MODULES = (
    "model.layers.0.self_attn",
    "model.layers.29.self_attn",
)


def load_trajectory(workspace: Path) -> dict[str, Any]:
    """Loads the run's trajectory from wherever the harness left it."""
    candidates = [
        os.environ.get("HARBOR_TRAJECTORY_PATH", ""),
        "/logs/agent/trajectory.json",
        "/logs/trajectory.json",
        str(workspace / "agent" / "trajectory.json"),
        str(workspace / "trajectory.json"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            try:
                return json.loads(Path(candidate).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
    return {}


def load_agy_transcripts() -> list[dict[str, Any]]:
    """Tool calls from step transcripts, file by file in order."""
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
    """Returns (lower-cased tool name, arguments) for every call."""
    raw: list[dict[str, Any]] = []
    for step in load_trajectory(workspace).get("steps") or []:
        if isinstance(step, dict):
            raw.extend(tc for tc in step.get("tool_calls") or []
                       if isinstance(tc, dict))
    raw.extend(load_agy_transcripts())
    return [
        _call(
            tc.get("function_name") or tc.get("name") or tc.get("tool_name"),
            tc.get("arguments") or tc.get("args") or tc.get("input"),
        ) for tc in raw
    ]


def statements(command: str) -> list[str]:
    """Splits a shell call into statements with simple variable expansion."""
    env: dict[str, str] = {}
    found = []
    for part in _STATEMENT_BREAK.split(command.replace("\\\n", " ")):
        part = _VARIABLE.sub(
            lambda m: env.get(m.group(1) or m.group(2), m.group(0)),
            part.strip(),
        )
        assigned = _ASSIGNMENT.match(part)
        if assigned:
            env[assigned.group(1)] = next(g for g in assigned.groups()[1:]
                                          if g is not None)
        elif part:
            found.append(part)
    return found


def commands(workspace: Path) -> list[str]:
    """Every shell statement the agent ran, in order."""
    found = []
    for name, args in tool_calls(workspace):
        if name in _CMD_TOOLS:
            for k, v in args.items():
                if k.lower() in _CMD_KEYS and v:
                    found += statements(str(v))
    return found


def _valid_metric_block(block: Any) -> bool:
    """Checks that a tensor diff block has finite non-negative max_abs and max_rel within 5%."""
    if not isinstance(block, dict):
        return False
    if block.get("shape_mismatch") is True:
        return False
    max_abs = block.get("max_abs")
    max_rel = block.get("max_rel")
    if isinstance(max_rel, str) and max_rel.endswith("%"):
        try:
            max_rel = float(max_rel[:-1].strip()) / 100.0
        except ValueError:
            return False
    if not isinstance(max_abs, (int, float)) or isinstance(max_abs, bool):
        return False
    if not isinstance(max_rel, (int, float)) or isinstance(max_rel, bool):
        return False
    if not (math.isfinite(max_abs) and math.isfinite(max_rel)):
        return False
    if max_abs < 0.0 or max_rel < 0.0:
        return False
    # Accept either decimal fraction (<= 0.05) or percentage (<= 5.0).
    return max_rel <= 5.0


def _validate_results_dict(data: dict[str, Any]) -> bool:
    """Validates either parity_results.json schema or compare_parity.py's parity_report.json schema."""
    if not isinstance(data, dict):
        return False

    # Schema A: /workspace/parity_results.json
    if "parity_passed" in data or ("logits" in data and "modules" in data):
        passed = data.get("parity_passed")
        verdict = str(data.get("verdict", "")).upper()
        if passed is not True and verdict != "PASS":
            return False
        if not _valid_metric_block(data.get("logits")):
            return False
        modules = data.get("modules")
        if isinstance(modules, dict):
            return all(
                _valid_metric_block(modules.get(m)) for m in _EXPECTED_MODULES)
        if isinstance(modules, list):
            by_name = {
                m.get("name"): m
                for m in modules
                if isinstance(m, dict) and isinstance(m.get("name"), str)
            }
            return all(
                _valid_metric_block(by_name.get(m)) for m in _EXPECTED_MODULES)
        return False

    # Schema B: compare_parity.py's parity_report.json
    if (str(data.get("verdict", "")).upper() == "PASS"
            and isinstance(data.get("probes"), list) and data["probes"]):
        for probe in data["probes"]:
            if not isinstance(probe, dict):
                return False
            if not _valid_metric_block(probe.get("logits")):
                return False
            mods = probe.get("modules")
            if isinstance(mods, list):
                by_name = {
                    m.get("name"): m
                    for m in mods
                    if isinstance(m, dict) and isinstance(m.get("name"), str)
                }
            elif isinstance(mods, dict):
                by_name = mods
            else:
                return False
            if not all(
                    _valid_metric_block(by_name.get(m))
                    for m in _EXPECTED_MODULES):
                return False
        return True

    return False


def _candidate_result_files(workspace: Path) -> list[Path]:
    paths: list[Path] = []
    for p in (
            workspace / "parity_results.json",
            Path("/workspace/parity_results.json"),
            workspace / "parity_report.json",
            Path("/workspace/parity_report.json"),
    ):
        if p.is_file() and p not in paths:
            paths.append(p)
    for root in (*_ARTIFACT_GLOBS, str(workspace)):
        for name in ("parity_results.json", "parity_report.json"):
            for match in sorted(glob.glob(f"{root}/**/{name}",
                                          recursive=True)):
                mp = Path(match)
                if mp.is_file() and mp not in paths:
                    paths.append(mp)
    return paths


@criterion(
    description="Executed a parity workload against the remote TPU cluster")
def ran_on_tpu(workspace: Path) -> bool:
    return any(_REMOTE_EXEC.search(c) for c in commands(workspace))


@criterion(description=(
    "Produced valid passing parity metrics for logits and hooked modules"))
def parity_results_valid(workspace: Path) -> bool:
    for path in _candidate_result_files(workspace):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and _validate_results_dict(data):
            return True
    return False
