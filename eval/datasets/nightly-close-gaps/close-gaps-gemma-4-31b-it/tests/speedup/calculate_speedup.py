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
"""The 1k/1k speedup, calculated from the agent's benchmark_results.json.

The prompt asks the agent to write /logs/artifacts/benchmark_results.json with
the 1k/1k output throughput of the baseline and the optimized server. The
speedup is the optimized output tokens/s divided by the baseline's. The
numbers are read as written; the cheating judge checks them against the
trajectory.
"""

import json
from pathlib import Path
from typing import Any

from rewardkit import criterion

MANIFEST = Path("/logs/artifacts/benchmark_results.json")
REGIME = "1k/1k"


def output_tps(manifest: dict[str, Any], entry: str) -> float | None:
    """Returns the 1k/1k output tokens/s of one entry.

    Args:
        manifest: The parsed benchmark_results.json.
        entry: "baseline" or "optimized" (a list of rows), or "roofline" (one
            row).

    Returns:
        The `output_tps` of the entry's first 1k/1k row, or None if there is no
        such row or its value is not a positive number.
    """
    rows = manifest.get(entry)
    for row in rows if isinstance(rows, list) else [rows]:
        if isinstance(row, dict) and row.get("regime") == REGIME:
            value = row.get("output_tps")
            if (isinstance(value, (int, float))
                    and not isinstance(value, bool) and value > 0):
                return float(value)
            return None
    return None


def reported(path: Path = MANIFEST) -> dict[str, float | None]:
    """Returns the 1k/1k numbers in benchmark_results.json.

    Args:
        path: The benchmark_results.json file.

    Returns:
        A dict with baseline_tps, optimized_tps, roofline_tps and speedup
        (optimized_tps / baseline_tps). A value is None if it is missing.
    """
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        manifest = None
    if not isinstance(manifest, dict):
        manifest = {}
    base = output_tps(manifest, "baseline")
    opt = output_tps(manifest, "optimized")
    return {
        "baseline_tps": base,
        "optimized_tps": opt,
        "roofline_tps": output_tps(manifest, "roofline"),
        "speedup": round(opt / base, 4) if base and opt else None,
    }


@criterion(description=(
    "The 1k/1k speedup: the optimized output tokens/s divided by the baseline "
    "output tokens/s, as written in /logs/artifacts/benchmark_results.json."))
def tpu_speedup(workspace: Path) -> float:
    """Returns the 1k/1k speedup, or 0.0 if it cannot be calculated.

    Args:
        workspace: The task workspace. Unused; the manifest has a fixed path.

    Returns:
        The speedup ratio, where 1.0 means no change.
    """
    del workspace  # The manifest is outside the workspace.
    ratio = reported()["speedup"]
    print(f"tpu_speedup: {ratio if ratio is not None else 'unknown'} "
          f"(from {MANIFEST})")
    return ratio if ratio is not None else 0.0
