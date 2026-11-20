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
"""Speedup gate verifying candidate performance exceeds baseline by target multiplier."""

import json
import os
import sys
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_report import extract_report
from rewardkit import criterion


def _tests_dir() -> Path:
    """The task's tests directory: $TESTS_DIR, else /tests."""
    return Path(os.environ.get("TESTS_DIR") or "/tests")


def _load_verifier_params() -> dict[str, Any]:
    """This verifier's table in the task's verifier_params.json, else {}.

    The table is named after the folder this verifier is staged in.
    """
    path = _tests_dir() / "verifier_params.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    params = data.get(Path(__file__).parent.name)
    return params if isinstance(params, dict) else {}


def evaluate_kernel_speedup(
    workspace: Path,
    target_speedup: float | None = None,
    custom_speedup_patterns: list[str] | None = None,
) -> float:
    """Evaluates whether candidate execution speedup meets or exceeds target_speedup."""
    params = _load_verifier_params()
    eff_target_speedup = (
        target_speedup if target_speedup is not None else float(
            params.get("target_speedup", params.get("min_speedup_factor",
                                                    1.0))))

    eff_patterns = custom_speedup_patterns or params.get(
        "custom_speedup_patterns")

    report = extract_report(
        workspace,
        target_speedup=eff_target_speedup,
        custom_speedup_patterns=eff_patterns,
    )

    # 1. Multiplicative Correctness Gate: If correctness was broken, score 0.0
    corr = report.get("correctness", {})
    if not corr.get("passed", False):
        print(
            "[speedup] FAIL: Numerical correctness failed or unverified. Multiplicative gate applied (score=0.0)."
        )
        return 0.0

    # 2. Speedup Extraction
    sp = report.get("speedup", {})
    factor = sp.get("speedup_factor")
    cand_ms = sp.get("candidate_latency_ms")
    base_ms = sp.get("baseline_latency_ms")
    source = sp.get("source", "unknown")

    if factor is None and cand_ms is not None and base_ms is not None and cand_ms > 0:
        factor = base_ms / cand_ms

    if factor is None:
        print(
            "[speedup] FAIL: No benchmark timing or speedup metrics found in trajectory or artifacts."
        )
        return 0.0

    factor = float(factor)
    factor_str = f"{factor:.2f}x"
    target_str = f"{eff_target_speedup:.2f}x"

    if factor >= eff_target_speedup:
        print(
            f"[speedup] PASS: Speedup {factor_str} >= Target {target_str} (Source={source})"
        )
        return 1.0
    else:
        print(
            f"[speedup] FAIL: Speedup {factor_str} < Target {target_str} (Source={source})"
        )
        return 0.0


@criterion(
    description=
    "Verify kernel execution speedup meets or exceeds target multiplier")
def check_kernel_speedup(workspace: Path) -> float:
    """RewardKit criterion verifying kernel speedup."""
    return evaluate_kernel_speedup(workspace)
