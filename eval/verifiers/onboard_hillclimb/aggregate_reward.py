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
"""Multiplicative reward aggregator for onboarding + hillclimb verifiers.

RewardKit natively supports additive aggregation (`weighted-sum` /
`weighted_mean`) across criteria, whereas TPU Agent Suite onboarding + kernel
optimization benchmarks require multiplicative success and anti-cheating gates:

  hillclimb_success = (1.0 if speedup >= 1.01 else 0.0) * onboarding_success
  speedup_score = clamp(log(max(speedup, 1.0)) / log(target_speedup * 2.0),
                        0.0, 1.0) * onboarding_success * hillclimb_success
  efficiency = efficiency_raw * onboarding_success * hillclimb_success
  reward = clamp((w_onboarding * onboarding_success
                  + w_hillclimb * hillclimb_success
                  + w_speedup * speedup_score
                  + w_efficiency * efficiency) * (1 - cheating), 0.0, 1.0)

This script runs as a `post_verify` hook after RewardKit writes raw scores to
`${LOGS_DIR}/verifier/reward.json` and `${LOGS_DIR}/verifier/reward-details.json`.

The `w_*` weights are the constants in `_WEIGHTS`. When the gates cannot be
applied, the hook records no reward and exits non-zero.
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

_WEIGHTS = {
    "onboarding_success": 0.3,
    "hillclimb_success": 0.3,
    "speedup": 0.3,
    "efficiency": 0.1,
}
# RewardKit's scores that the gates read.
_GATE_INPUTS = ("onboarding_success", "efficiency", "cheating")

_HILLCLIMB_OUTPUT_COLS = (
    "onboarding_success",
    "hillclimb_success",
    "speedup_score",
    "efficiency",
    "trajectory_score",
    "valid_tool_calls",
    "framework_directed_penalty",
    "target_speedup",
    "speedup",
    "cheating",
    "invoked_subagent",
    "schema_compliance",
    "reward",
)


def _logs_dir() -> Path:
    """The run's logs as the harness lays them out: $LOGS_DIR, else /logs."""
    return Path(os.environ.get("LOGS_DIR") or "/logs")


def _tests_dir() -> Path:
    """The task's tests directory: $TESTS_DIR, else /tests."""
    return Path(os.environ.get("TESTS_DIR") or "/tests")


def _workspace_dir() -> Path:
    """The run's workspace directory: $WORKSPACE_DIR, else /workspace."""
    return Path(os.environ.get("WORKSPACE_DIR") or "/workspace")


_XPROF_SKIP_DIRS = frozenset(
    {".git", "venv", ".venv", "__pycache__", "node_modules"})
_XPROF_EXCLUDED_SEGMENTS = frozenset({"onboarding", "tpu-diagnostics"})


def _is_post_hillclimb_xplane(rel_parts: tuple[str, ...]) -> bool:
    """True when an `.xplane.pb` sits under `hillclimb/` and is not a baseline trace."""
    lower_parts = tuple(p.lower() for p in rel_parts)
    if "hillclimb" not in lower_parts:
        return False
    if any(seg in _XPROF_EXCLUDED_SEGMENTS for seg in lower_parts[:-1]):
        return False
    return "baseline" not in lower_parts[-1]


def _collect_post_hillclimb_xprof(
    logs_dir: Path,
    workspace_dir: Path,
) -> dict[str, Any]:
    """Counts non-empty post-hill-climb XProf traces and bottleneck reports."""
    search_roots = [
        p for p in (logs_dir / "artifacts", workspace_dir) if p.is_dir()
    ]
    xplanes: set[Path] = set()
    gap_docs: set[Path] = set()
    for root in search_roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _XPROF_SKIP_DIRS]
            cur_dir = Path(dirpath)
            for fname in filenames:
                if not (fname.endswith(".xplane.pb")
                        or fname == "xprof_bottlenecks_and_gaps.md"):
                    continue
                fpath = cur_dir / fname
                try:
                    if not fpath.is_file():
                        continue
                    size = fpath.stat().st_size
                    resolved = fpath.resolve()
                except OSError:
                    continue
                if fname.endswith(".xplane.pb") and size > 0:
                    try:
                        rel_parts = fpath.relative_to(root).parts
                    except ValueError:
                        continue
                    if _is_post_hillclimb_xplane(rel_parts):
                        xplanes.add(resolved)
                elif fname == "xprof_bottlenecks_and_gaps.md" and size >= 100:
                    gap_docs.add(resolved)
    return {
        "xplane_trace_count": len(xplanes),
        "gaps_report_count": len(gap_docs),
        "passed": bool(xplanes and gap_docs),
    }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_score(value: Any) -> bool:
    """Whether a gate can use `value`: a finite number, or a boolean."""
    return isinstance(value, (int, float)) and math.isfinite(value)


def _require_gate_inputs(scores: dict[str, Any]) -> None:
    """Raises unless every score the gates read is one they can use."""
    missing = [key for key in _GATE_INPUTS if not _is_score(scores.get(key))]
    if missing:
        raise ValueError("the gates need a number for each of "
                         f"{', '.join(missing)}, and RewardKit scored none")


def find_criterion(node: Any, name: str) -> dict[str, Any] | None:
    """Recursively locates a criterion dict matching `name` in reward-details."""
    if isinstance(node, dict):
        if node.get("name") == name and ("value" in node or "raw" in node):
            return node
        children = list(node.values())
    elif isinstance(node, list):
        children = node
    else:
        return None

    for child in children:
        found = find_criterion(child, name)
        if found is not None:
            return found
    return None


def read_target_speedup(tests_dir: Path, default: float = 1.0) -> float:
    """Reads the `speedup` verifier's `target_speedup` in `tests_dir`."""
    try:
        data = json.loads(
            (tests_dir / "verifier_params.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default
    speedup = data.get("speedup") if isinstance(data, dict) else None
    raw = speedup.get("target_speedup") if isinstance(speedup, dict) else None
    return float(raw) if _is_number(raw) else default


def aggregate_hillclimb_scores(
    scores: dict[str, Any],
    details: dict[str, Any],
    tests_dir: Path,
    headroom_multiplier: float = 2.0,
) -> dict[str, Any]:
    """Applies onboarding + hillclimb multiplicative gating and log-speedup scaling."""
    updated = dict(scores)

    speedup_node = find_criterion(details.get("speedup"), "tpu_speedup")
    if speedup_node is not None and _is_number(speedup_node.get("raw")):
        updated["speedup"] = round(float(speedup_node["raw"]), 4)

    target_speedup = read_target_speedup(tests_dir, default=1.0)
    target_with_headroom = target_speedup * headroom_multiplier
    if not math.isfinite(target_with_headroom) or target_with_headroom <= 1.0:
        raise ValueError(
            f"target_speedup {target_speedup} x headroom {headroom_multiplier} "
            "must exceed 1.0 to scale the speedup score")
    updated["target_speedup"] = target_speedup

    eff_details = details.get("efficiency")
    for eff_key in (
            "trajectory_score",
            "valid_tool_calls",
            "framework_directed_penalty",
    ):
        eff_node = find_criterion(eff_details, eff_key)
        if eff_node is not None:
            eff_val = eff_node.get("value")
            if not _is_number(eff_val):
                eff_val = eff_node.get("raw")
            if _is_number(eff_val):
                updated[eff_key] = round(float(eff_val), 4)

    _require_gate_inputs(updated)
    onboarding = updated["onboarding_success"]
    speedup_val = updated.get("speedup", 0.0)
    if not _is_number(speedup_val):
        speedup_val = 0.0

    hillclimb_raw = 1.0 if speedup_val >= 1.01 else 0.0
    hillclimb = hillclimb_raw * onboarding

    trajectory_score = float(updated.get("trajectory_score", 1.0))
    valid_tool_calls = float(
        updated.get("valid_tool_calls", updated["efficiency"]))
    framework_directed_penalty = float(
        updated.get("framework_directed_penalty", 1.0))
    updated["trajectory_score"] = round(trajectory_score, 4)
    updated["valid_tool_calls"] = round(valid_tool_calls, 4)
    updated["framework_directed_penalty"] = round(framework_directed_penalty,
                                                  4)
    efficiency_raw = (trajectory_score * valid_tool_calls *
                      framework_directed_penalty)

    speedup_score_raw = _clamp(
        math.log(max(float(speedup_val), 1.0)) /
        math.log(target_with_headroom),
        0.0,
        1.0,
    )
    speedup_score = speedup_score_raw * onboarding * hillclimb
    efficiency = efficiency_raw * onboarding * hillclimb

    updated["hillclimb_success"] = round(hillclimb, 4)
    updated["speedup_score"] = round(speedup_score, 4)
    updated["efficiency"] = round(efficiency, 4)

    base = (_WEIGHTS["onboarding_success"] * onboarding +
            _WEIGHTS["hillclimb_success"] * hillclimb +
            _WEIGHTS["speedup"] * speedup_score +
            _WEIGHTS["efficiency"] * efficiency)
    gated = base * (1.0 - updated["cheating"])
    updated["reward"] = round(_clamp(gated, 0.0, 1.0), 4)

    return {k: updated[k] for k in _HILLCLIMB_OUTPUT_COLS if k in updated}


def _print_criterion_reports(details: dict[str, Any]) -> None:
    for section in ("onboarding_success", "cheating", "execution_report"):
        report = details.get(section)
        if not isinstance(report, dict):
            continue
        criteria = report.get("criteria")
        if not isinstance(criteria, list):
            continue
        for crit in criteria:
            if not isinstance(crit, dict):
                continue
            print(
                f"\n=== {section}/{crit.get('name')}: raw={crit.get('raw')} ==="
            )
            print(
                crit.get("reasoning") or crit.get("error") or "(no reasoning)")


def withhold_reward(logs_dir: Path) -> str:
    """Deletes reward.json; returns what it held."""
    reward_path = logs_dir / "verifier" / "reward.json"
    try:
        held = reward_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        held = ""
    reward_path.unlink(missing_ok=True)
    return held.strip()


def _try_read_details(details_path: Path) -> dict[str, Any] | None:
    """Parses `reward-details.json`, or returns None if missing/malformed."""
    try:
        details = json.loads(details_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return details if isinstance(details, dict) else None


def _read_details(verifier_dir: Path) -> dict[str, Any]:
    """RewardKit's reward-details.json, or {} when missing or malformed."""
    parsed = _try_read_details(verifier_dir / "reward-details.json")
    return parsed if parsed is not None else {}


def run_aggregation(
    logs_dir: Path,
    tests_dir: Path,
    workspace_dir: Path | None = None,
) -> dict[str, Any]:
    """Reads RewardKit outputs, computes gated scores, and writes final files."""
    verifier_dir = logs_dir / "verifier"
    reward_path = verifier_dir / "reward.json"

    if not reward_path.is_file():
        raise FileNotFoundError(f"Missing reward.json at {reward_path}")

    scores = json.loads(reward_path.read_text(encoding="utf-8"))
    if not isinstance(scores, dict):
        raise ValueError(f"Expected JSON object in {reward_path}")

    details_path = verifier_dir / "reward-details.json"
    parsed_details = _try_read_details(details_path)
    details = parsed_details if parsed_details is not None else {}

    if parsed_details is not None:
        try:
            eff_workspace = (workspace_dir if workspace_dir is not None else
                             _workspace_dir())
            details["post_hillclimb_xprof"] = _collect_post_hillclimb_xprof(
                logs_dir=logs_dir,
                workspace_dir=eff_workspace,
            )
            summary_path = verifier_dir / "speedup_summary.json"
            if summary_path.is_file():
                try:
                    summary_data = json.loads(
                        summary_path.read_text(encoding="utf-8"))
                    if isinstance(summary_data, dict):
                        details["speedup_summary"] = summary_data
                except (OSError, ValueError):
                    pass
            details_path.write_text(json.dumps(details, indent=2),
                                    encoding="utf-8")
        except OSError:
            pass

    final_scores = aggregate_hillclimb_scores(
        scores=scores,
        details=details,
        tests_dir=tests_dir,
    )

    reward_path.write_text(json.dumps(final_scores, indent=2),
                           encoding="utf-8")
    return final_scores


def main() -> int:
    logs_dir = _logs_dir()
    tests_dir = _tests_dir()
    workspace_dir = _workspace_dir()
    try:
        final_scores = run_aggregation(
            logs_dir=logs_dir,
            tests_dir=tests_dir,
            workspace_dir=workspace_dir,
        )
    except Exception as exc:
        held = withhold_reward(logs_dir)
        print(
            f"[!] aggregate_reward could not complete ({exc}); no reward is "
            f"recorded. reward.json held: {held or '(nothing)'}",
            file=sys.stderr,
        )
        return 1
    # The gated reward is on disk by now, so a report that fails to print
    # cannot withhold it.
    print(json.dumps(final_scores, indent=2))
    _print_criterion_reports(_read_details(logs_dir / "verifier"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
