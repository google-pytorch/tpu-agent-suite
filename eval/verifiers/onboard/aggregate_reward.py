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
"""Multiplicative reward aggregator for serving onboarding verifiers.

RewardKit natively supports additive aggregation (`weighted-sum` /
`weighted_mean`) across criteria, whereas TPU Agent Suite serving onboarding
benchmarks require multiplicative success and anti-cheating gates:

  efficiency = (trajectory_score * valid_tool_calls
                * framework_directed_penalty * onboarding_success)
  reward = clamp((w_onboarding * onboarding_success + w_efficiency * efficiency)
                 * (1 - cheating), 0.0, 1.0)

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

_WEIGHTS = {"onboarding_success": 0.6, "efficiency": 0.4}
# RewardKit's scores that the gates read.
_GATE_INPUTS = ("onboarding_success", "efficiency", "cheating")

_ONBOARD_SUBCRITERIA = (
    "vllm_server_on_tpu",
    "completion_over_http",
    "baseline_benchmarks_reported",
    "baseline_reproduce_commands",
    "no_stale_servers",
)

_ONBOARD_OUTPUT_COLS = (
    "reward",
    "onboarding_success",
    "efficiency",
    "trajectory_score",
    "valid_tool_calls",
    "framework_directed_penalty",
    "cheating",
    "regime_coverage",
    "peak_tps_1k_1k",
    "peak_tps_1k_8k",
    "peak_tps_8k_1k",
    "vllm_server_on_tpu",
    "completion_over_http",
    "baseline_benchmarks_reported",
    "baseline_reproduce_commands",
    "no_stale_servers",
    "invoked_subagent",
    "schema_compliance",
)


def _logs_dir() -> Path:
    """The run's logs as the harness lays them out: $LOGS_DIR, else /logs."""
    return Path(os.environ.get("LOGS_DIR") or "/logs")


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


def _extract_binary_criterion(node: dict[str, Any]) -> float:
    val = node.get("value")
    if _is_number(val):
        return 1.0 if float(val) >= 0.5 else 0.0
    raw = str(node.get("raw", "")).strip().lower()
    return 1.0 if raw in ("yes", "true", "1", "1.0", "pass") else 0.0


def aggregate_onboard_scores(
    scores: dict[str, Any],
    details: dict[str, Any],
    logs_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Applies onboarding-only multiplicative gating and telemetry extraction."""
    updated = dict(scores)
    updated_details = dict(details)
    details_modified = False

    onboard_node = updated_details.get("onboarding_success")
    for cname in _ONBOARD_SUBCRITERIA:
        cnode = find_criterion(onboard_node, cname)
        if cnode is not None:
            updated[cname] = _extract_binary_criterion(cnode)

    steps_node = find_criterion(updated_details.get("execution_report"),
                                "steps")
    if steps_node is not None and _is_number(steps_node.get("raw")):
        updated["regimes_measured_judge"] = int(steps_node["raw"])

    sweep_file = logs_dir / "verifier" / "regime_sweep_summary.json"
    if sweep_file.is_file():
        try:
            sweep_data = json.loads(sweep_file.read_text(encoding="utf-8"))
            if isinstance(sweep_data, dict):
                for metric in (
                        "regime_coverage",
                        "peak_tps_1k_1k",
                        "peak_tps_1k_8k",
                        "peak_tps_8k_1k",
                ):
                    updated[metric] = float(sweep_data.get(metric, 0.0))
                updated_details["regime_sweep_details"] = sweep_data
                details_modified = True
        except (OSError, ValueError):
            pass

    eff_details = updated_details.get("efficiency")
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
    onboarding = round(float(updated["onboarding_success"]), 4)
    updated["onboarding_success"] = onboarding

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
    updated["efficiency_raw"] = round(efficiency_raw, 4)
    efficiency = round(efficiency_raw * onboarding, 4)
    updated["efficiency"] = efficiency

    base = (_WEIGHTS["onboarding_success"] * onboarding +
            _WEIGHTS["efficiency"] * efficiency)
    gated = base * (1.0 - float(updated["cheating"]))
    updated["reward"] = round(_clamp(gated, 0.0, 1.0), 4)

    filtered = {k: updated[k] for k in _ONBOARD_OUTPUT_COLS if k in updated}
    for key, val in updated.items():
        if (key not in filtered and not key.endswith("_raw")
                and not key.endswith("_gated")
                and key not in ("regime_sweep", "onboarding_all_pass")):
            filtered[key] = val

    return filtered, updated_details, details_modified


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


def _read_details(verifier_dir: Path) -> dict[str, Any]:
    """RewardKit's reward-details.json, or {} when missing or malformed."""
    try:
        details = json.loads(
            (verifier_dir / "reward-details.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return details if isinstance(details, dict) else {}


def run_aggregation(logs_dir: Path) -> dict[str, Any]:
    """Reads RewardKit outputs, computes gated scores, and writes final files."""
    verifier_dir = logs_dir / "verifier"
    reward_path = verifier_dir / "reward.json"
    details_path = verifier_dir / "reward-details.json"

    if not reward_path.is_file():
        raise FileNotFoundError(f"Missing reward.json at {reward_path}")

    scores = json.loads(reward_path.read_text(encoding="utf-8"))
    if not isinstance(scores, dict):
        raise ValueError(f"Expected JSON object in {reward_path}")

    details = _read_details(verifier_dir)

    final_scores, updated_details, details_modified = aggregate_onboard_scores(
        scores=scores,
        details=details,
        logs_dir=logs_dir,
    )
    if details_modified:
        details_path.write_text(json.dumps(updated_details, indent=2),
                                encoding="utf-8")

    reward_path.write_text(json.dumps(final_scores, indent=2),
                           encoding="utf-8")
    return final_scores


def main() -> int:
    logs_dir = _logs_dir()
    try:
        final_scores = run_aggregation(logs_dir=logs_dir)
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
