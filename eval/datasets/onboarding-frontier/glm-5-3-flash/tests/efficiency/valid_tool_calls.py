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
"""RewardKit verifier to calculate the valid tool call ratio from trajectory."""

import json
import os
import re
from pathlib import Path
from typing import Any

from rewardkit import criterion

_ERROR_STATUSES = {
    "error",
    "errored",
    "fail",
    "failed",
    "failure",
    "invalid",
    "cancelled",
    "canceled",
    "timeout",
    "timed_out",
}

_READ_ONLY_FILE_TOOLS = {
    "view_file",
    "read_file",
    "read",
    "cat",
    "grep_search",
    "grep",
    "find_by_name",
    "list_dir",
    "glob",
}

_TOOL_RUNNER_ERROR_RE = re.compile(
    r"(?:"
    r"\[error\]\s*tool reported failure"
    r"|\[exit_code\]\s*-?[1-9]\d*"
    r"|The command exited with code\s+-?[1-9]\d*"
    r"|Encountered error in tool execution"
    r"|<tool_use_error>"
    r"|^(?:Error|ERROR|ToolError):\s+(?:executing tool|unknown tool|invalid tool|tool .* not found|missing required|invalid argument|failed to)"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

_EXECUTION_ERROR_RE = re.compile(
    r"(?:"
    r"Traceback \(most recent call last\):"
    r"|^\s*(?:SyntaxError|NameError|TypeError|AttributeError|ImportError|ModuleNotFoundError|KeyError|ValueError|FileNotFoundError|PermissionError|CalledProcessError):"
    r"|(?:^|:\s+)command not found\b"
    r"|(?:^|:\s+)(?:unrecognized arguments|no such option|unknown option|unknown flag|invalid choice):"
    r"|^(?:bash|sh|zsh|python3?):\s+.*:\s+No such file or directory"
    r"|(?:exit\s*code|return\s*code|exited\s+with\s+(?:code|status))\s*[:=]?\s*-?[1-9]\d*"
    r")",
    re.MULTILINE,
)

_EXPLICIT_EXIT_ZERO_RE = re.compile(
    r"(?:\[exit_code\]\s*0\b|The command exited with code\s+0\b)",
    re.IGNORECASE,
)


def _load_trajectory(path: str | Path) -> dict[str, Any] | None:
    """Load an ATIF trajectory JSON file. Returns None on error."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _load_verifier_params() -> dict[str, Any]:
    p = Path("/tests/verifier_params.json")
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        key = Path(__file__).parent.name
        val = data.get(key, data)
        return val if isinstance(val, dict) else data
    except (json.JSONDecodeError, OSError):
        return {}


def _is_nonzero_exit_code(val: Any) -> bool:
    if isinstance(val, bool) or val is None:
        return False
    if isinstance(val, (int, float)):
        return int(val) != 0
    if isinstance(val, str):
        s = val.strip()
        if s.lstrip("-").isdigit():
            return int(s) != 0
    return False


def _dict_has_error(d: dict[str, Any]) -> bool:
    """Checks whether a metadata or payload dict carries an explicit error."""
    if d.get("is_error") is True:
        return True
    if d.get("tool_use_is_error") is True or d.get(
            "tool_result_is_error") is True:
        return True
    if d.get("error"):
        return True
    status = str(d.get("status") or "").strip().lower()
    if status in _ERROR_STATUSES:
        return True
    for key in ("exit_code", "exitCode", "returncode", "return_code"):
        if key in d and _is_nonzero_exit_code(d.get(key)):
            return True
    return False


def _extract_content_text(content: Any) -> str:
    """Normalizes an ObservationResult content payload into a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("is_error") is True or item.get(
                        "type") == "tool_use_error":
                    parts.append("[error] tool reported failure")
                text = item.get("text") or item.get("content") or item.get(
                    "output")
                if text is not None:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content)
    return str(content)


def _is_tool_call_valid(
    call: dict[str, Any],
    result: dict[str, Any] | None,
    step: dict[str, Any],
) -> bool:
    """Determines whether a single tool call succeeded without error."""
    tool_name = (call.get("function_name") or call.get("name")
                 or "").strip().lower()
    if not tool_name or tool_name == "unknown":
        return False

    if _dict_has_error(call):
        return False
    call_extra = call.get("extra")
    if isinstance(call_extra, dict) and _dict_has_error(call_extra):
        return False

    raw_args = call.get("arguments") if "arguments" in call else call.get(
        "args")
    if isinstance(raw_args, str):
        try:
            parsed_args = json.loads(raw_args)
            if not isinstance(parsed_args, dict):
                return False
        except json.JSONDecodeError:
            return False

    step_calls = step.get("tool_calls") or []
    step_extra = step.get("extra")
    if isinstance(step_extra, dict):
        if len(step_calls) <= 1 and _dict_has_error(step_extra):
            return False
        meta = step_extra.get("tool_result_metadata")
        if isinstance(meta, dict):
            if meta.get("is_error") is True:
                return False
            tur = meta.get("tool_use_result")
            if isinstance(tur, dict) and _dict_has_error(tur):
                return False

    if not isinstance(result, dict):
        return True

    if _dict_has_error(result):
        return False
    res_extra = result.get("extra")
    if isinstance(res_extra, dict) and _dict_has_error(res_extra):
        return False

    raw_content = result.get("content")
    if isinstance(raw_content, dict) and _dict_has_error(raw_content):
        return False

    content_text = _extract_content_text(raw_content)
    if not content_text:
        return True

    stripped = content_text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed_content = json.loads(stripped)
            if isinstance(parsed_content,
                          dict) and _dict_has_error(parsed_content):
                return False
        except json.JSONDecodeError:
            pass

    if _TOOL_RUNNER_ERROR_RE.search(content_text):
        return False

    if tool_name in _READ_ONLY_FILE_TOOLS:
        return True

    if _EXECUTION_ERROR_RE.search(content_text):
        return False

    if _EXPLICIT_EXIT_ZERO_RE.search(content_text):
        return True

    return True


def _collect_tool_call_outcomes(
        data: dict[str, Any]) -> list[tuple[str, bool]]:
    """Collects (tool_name, is_valid) for every tool call across all steps."""
    steps = data.get("steps") or []

    results_by_call_id: dict[str, dict[str, Any]] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        obs = step.get("observation")
        if not isinstance(obs, dict):
            continue
        for res in obs.get("results") or []:
            if isinstance(res, dict):
                cid = res.get("source_call_id") or res.get("tool_call_id")
                if cid:
                    results_by_call_id[str(cid)] = res

    outcomes: list[tuple[str, bool]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        tool_calls = step.get("tool_calls") or []
        obs = step.get("observation")
        step_results = (obs.get("results") or []) if isinstance(obs,
                                                                dict) else []

        for idx, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                outcomes.append(("unknown", False))
                continue

            tool_name = (call.get("function_name") or call.get("name")
                         or "unknown").strip()

            call_id = call.get("tool_call_id") or call.get("id")
            matched_result: dict[str, Any] | None = None
            if call_id and str(call_id) in results_by_call_id:
                matched_result = results_by_call_id[str(call_id)]
            elif idx < len(step_results) and isinstance(
                    step_results[idx], dict):
                matched_result = step_results[idx]

            is_valid = _is_tool_call_valid(call, matched_result, step)
            outcomes.append((tool_name, is_valid))

    return outcomes


@criterion(
    description="Calculate the ratio of valid tool calls to total tool calls")
def valid_tool_calls(workspace: Path) -> float:
    """Calculates valid tool call ratio from Harbor's standardized ATIF trajectory."""
    traj_candidates = []
    if os.environ.get("ATIF_TRAJECTORY"):
        traj_candidates.append(Path(os.environ["ATIF_TRAJECTORY"]))
    if os.environ.get("TRAJECTORY_PATH"):
        traj_candidates.append(Path(os.environ["TRAJECTORY_PATH"]))
    if workspace:
        traj_candidates.extend([
            workspace / "trajectory.json",
            workspace / "agent" / "trajectory.json",
            workspace.parent / "agent" / "trajectory.json",
            workspace.parent / "trajectory.json",
        ])
        try:
            traj_candidates.extend(
                list(workspace.glob("**/trajectory.json"))[:5])
        except Exception:
            pass
    traj_candidates.extend([
        Path("/logs/agent/trajectory.json"),
        Path("/logs/trajectory.json"),
    ])

    traj = None
    for cand in traj_candidates:
        if cand and cand.exists():
            traj = _load_trajectory(cand)
            if traj:
                break

    if not traj:
        print("valid_tool_calls: [FAIL] No trajectory found in candidates")
        return 0.0

    params = _load_verifier_params()
    outcomes = _collect_tool_call_outcomes(traj)
    total_calls = len(outcomes)
    if total_calls == 0:
        empty_score = float(params.get("empty_score", 1.0))
        print(
            f"valid_tool_calls: [INFO] No tool calls found in trajectory (score={empty_score:.4f})"
        )
        return empty_score

    valid_calls = sum(1 for _, ok in outcomes if ok)
    failed_calls = total_calls - valid_calls
    ratio = valid_calls / total_calls

    failed_names = [name for name, ok in outcomes if not ok]
    print(f"valid_tool_calls: valid={valid_calls}/{total_calls} "
          f"(failed={failed_calls}, ratio={ratio:.4f})" +
          (f" failed_tools={failed_names}" if failed_names else ""))

    min_ratio = params.get("min_ratio")
    if min_ratio is not None and ratio < float(min_ratio):
        print(
            f"valid_tool_calls: [FAIL] Ratio {ratio:.4f} is below min_ratio {float(min_ratio):.4f}"
        )
        return 0.0

    return ratio
