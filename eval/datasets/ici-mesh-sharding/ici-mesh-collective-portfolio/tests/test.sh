#!/bin/bash
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

# Scores the answer in /workspace/work/portfolio_recommendation.json, and
# only there: a JSON block in the agent's messages is reported
# (message_json_found) but never scored. The answer can be worked out by hand
# from the five ops in op_profile.json, so the agent's ATIF trajectory must
# also show a successful shell call of the skill's `cli.py rank-portfolio`
# (the portfolio ranking, which also attributes the collective time) whose
# tool result holds that command's own output. An echoed command, or output
# keys typed into the command, does not count. With no readable trajectory
# nothing can be checked, so both trajectory checks fail.
set -uo pipefail
LOGS="${LOGS_DIR:-/logs/verifier}"
WORK_DIR="${WORK_DIR:-/workspace/work}"
mkdir -p "$LOGS"

python3 - "$LOGS" "$WORK_DIR" "${AGENT_LOGS_DIR:-/logs/agent}" <<'PYEOF'
import itertools
import json
import math
import pathlib
import re
import sys

logs = pathlib.Path(sys.argv[1])
work_dir = pathlib.Path(sys.argv[2])
agent_logs = pathlib.Path(sys.argv[3])
rec_path = work_dir / "portfolio_recommendation.json"

# Shell tools: Claude Code's Bash, Antigravity's run_command, and the names
# other harnesses use.
SHELL_TOOLS = ("Bash", "run_command", "run_shell_command", "shell")
FAILED_STATUSES = ("ERROR", "FAILED")
SUBCOMMAND_RE = re.compile(
    r"(?<![\w./-])(enumerate|breakdown|rank-portfolio|check-topology|"
    r"plan-topology|validate-model|verify-compile)(?![\w-])"
    r"(?!\s+(?:-h|--help)\b)")
COMPUTE_RE = re.compile(
    r"\bcompute(?:\s+--?[\w-]+(?:(?:=|\s+)(?!-)\S+)?)*\s+(?:run|submit)\b"
    r"|\bkubectl\s+(?:apply|create)\b")
# Text only `cli.py rank-portfolio` prints: the title and proposal-table
# header of its Markdown report, or two keys of its --output-format json one.
RANK_PORTFOLIO_OUTPUT = (
    ("# Ranked mesh portfolio", "hypothesis (slot)"),
    ('"hypotheses"', '"estimated_gain_pct"'),
)


def load_steps(path: pathlib.Path) -> list | None:
    """The steps of an ATIF trajectory, or None if it cannot be read."""
    try:
        steps = json.loads(path.read_text(encoding="utf-8")).get("steps")
    except (OSError, ValueError, AttributeError):
        return None
    return steps if isinstance(steps, list) else None


def shell_command(call: dict) -> str:
    """The command a shell tool call ran, or '' for any other call."""
    if call.get("function_name") not in SHELL_TOOLS:
        return ""
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        return ""
    cmd = args.get("command") or args.get("CommandLine") or args.get("cmd")
    if isinstance(cmd, list):
        cmd = " ".join(str(part) for part in cmd)
    return cmd if isinstance(cmd, str) else ""


def result_text(result: dict) -> str:
    """The text of a tool result's content."""
    content = (result or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(c.get("text") or "") if isinstance(c, dict) else str(c)
            for c in content)
    return ""


def cli_subcommands(command: str) -> set[str]:
    """The cli.py subcommands a command runs, not --help."""
    if "cli.py" not in command:
        return set()
    return set(SUBCOMMAND_RE.findall(command))


def printed_rank_portfolio(command: str, output: str) -> bool:
    """True when a tool result holds rank-portfolio's own output.

    The markers must be in the result and not in the command, so echoing
    them does not count.
    """
    return any(
        all(m in output for m in group) and not any(m in command for m in group)
        for group in RANK_PORTFOLIO_OUTPUT)


# Harbor writes the trajectory to /logs/agent/trajectory.json; some runs keep
# it at /logs/trajectory.json. A delegated subagent may have its own file.
traj_path = next((p for p in (agent_logs / "trajectory.json",
                              agent_logs.parent / "trajectory.json")
                  if p.is_file()), None)
steps = load_steps(traj_path) if traj_path else None
trajectory_ok = steps is not None
all_steps = list(steps or [])
for sub in sorted(agent_logs.glob("**/subagent*.json")):
    all_steps += load_steps(sub) or []

subcommands: set[str] = set()
ran_rank_portfolio = False
launched_compute = False
message_json_found = False
for step in all_steps:
    if not isinstance(step, dict):
        continue
    message = step.get("message")
    if (step.get("source") == "agent" and isinstance(message, str)
            and re.search(r"```(?:json)?\s*\{.*?\}\s*```", message, re.S)):
        message_json_found = True
    results = {
        r.get("source_call_id"): r
        for r in (step.get("observation") or {}).get("results") or []
        if isinstance(r, dict)
    }
    for call in step.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        command = shell_command(call)
        if COMPUTE_RE.search(command):
            launched_compute = True
        result = results.get(call.get("tool_call_id")) or {}
        extra = result.get("extra") or {}
        if (extra.get("tool_result_is_error") is True
                or extra.get("status") in FAILED_STATUSES):
            continue
        ran = cli_subcommands(command)
        subcommands |= ran
        if "rank-portfolio" in ran and printed_rank_portfolio(
                command, result_text(result)):
            ran_rank_portfolio = True

AXES = ("data", "fsdp", "tensor", "sequence", "expert")


def is_power_of_two(val: int) -> bool:
    return isinstance(val, int) and not isinstance(val, bool) and val >= 1 and (val & (val - 1)) == 0


def fits_4x4x4_without_split(mesh: dict[str, int]) -> bool:
    active = [int(v) for v in mesh.values() if int(v) > 1]
    if math.prod(active or [1]) != 64:
        return False
    # Physical dims are (4, 4, 4): every non-1 logical axis must factor from
    # disjoint whole physical dimensions (i.e. each active axis in {4, 16, 64}).
    for perm in itertools.permutations((4, 4, 4)):
        idx = 0
        ok = True
        for size in active:
            prod = 1
            while prod < size and idx < len(perm):
                prod *= perm[idx]
                idx += 1
            if prod != size:
                ok = False
                break
        if ok and idx == len(perm):
            return True
    return False


checks = {
    "wrote_recommendation_json": False,
    "identified_expert_bottleneck": False,
    "identified_moe_routing_role": False,
    "accurate_collective_pct": False,
    "detected_baseline_split_axes": False,
    "valid_portfolio_bounds_and_uniqueness": False,
    "includes_torus_aligned_expert_reduction": False,
    "ran_rank_portfolio": trajectory_ok and ran_rank_portfolio,
    "no_compute_job_submitted": trajectory_ok and not launched_compute,
}
error_detail = None

payload = None
if rec_path.is_file():
    try:
        payload = json.loads(rec_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pylint: disable=broad-except
        error_detail = str(exc)

if isinstance(payload, dict):
    try:
        checks["wrote_recommendation_json"] = True
        axis = str(payload.get("primary_bottleneck_axis", "")).strip().lower()
        checks["identified_expert_bottleneck"] = axis == "expert"

        role = str(payload.get("primary_bottleneck_role", "")).strip().lower()
        checks["identified_moe_routing_role"] = ("moe" in role) or ("routing" in role)

        coll_pct = payload.get("collective_pct")
        if isinstance(coll_pct, (int, float)) and not isinstance(coll_pct, bool):
            checks["accurate_collective_pct"] = abs(float(coll_pct) - 57.0) <= 1.5

        split_req = payload.get("baseline_requires_split_physical_axes")
        checks["detected_baseline_split_axes"] = split_req is True

        raw_props = payload.get("proposals")
        if isinstance(raw_props, list) and 3 <= len(raw_props) <= 7:
            seen = set()
            all_legal = True
            aligned_reduced = False
            for entry in raw_props:
                mesh_obj = entry.get("mesh") if isinstance(entry, dict) and "mesh" in entry else entry
                if not isinstance(mesh_obj, dict):
                    all_legal = False
                    break
                norm = {k: int(mesh_obj.get(k, 1)) for k in AXES}
                if not all(is_power_of_two(v) for v in norm.values()):
                    all_legal = False
                    break
                if math.prod(norm.values()) != 64:
                    all_legal = False
                    break
                if 128 % (norm["data"] * norm["fsdp"]) != 0:
                    all_legal = False
                    break
                if 8192 % norm["sequence"] != 0:
                    all_legal = False
                    break
                if 32 % norm["tensor"] != 0:
                    all_legal = False
                    break
                key = json.dumps(norm, sort_keys=True)
                if key in seen:
                    all_legal = False
                    break
                seen.add(key)
                if norm["expert"] < 8 and fits_4x4x4_without_split(norm):
                    aligned_reduced = True
            checks["valid_portfolio_bounds_and_uniqueness"] = all_legal
            checks["includes_torus_aligned_expert_reduction"] = all_legal and aligned_reduced
    except Exception as exc:  # pylint: disable=broad-except
        error_detail = str(exc)

score = 1.0 if all(checks.values()) else 0.0
(logs / "reward.txt").write_text(f"{score}\n", encoding="utf-8")
(logs / "reward.json").write_text(json.dumps({"reward": score}), encoding="utf-8")
(logs / "reward-details.json").write_text(
    json.dumps(
        {
            "checks": {k: int(v) for k, v in checks.items()},
            "error": error_detail,
            "trajectory_present": trajectory_ok,
            "cli_subcommands": sorted(subcommands),
            "message_json_found": message_json_found,
        },
        indent=2,
    ),
    encoding="utf-8",
)
for name, passed in checks.items():
    print(f"  {name:<42} {'PASS' if passed else 'FAIL'}")
PYEOF
