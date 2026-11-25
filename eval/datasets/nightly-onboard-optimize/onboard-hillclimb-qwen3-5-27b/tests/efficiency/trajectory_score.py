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
"""RewardKit verifier to extract agent runtime, turns, and total tokens and compute P50-shifted Log-Weighted AUC."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from rewardkit import criterion

DEFAULT_STATS: dict[str, float] = {
    "agent_runtime_median": 120.0,  # minutes
    "agent_runtime_ceiling": 750.0,  # minutes
    "total_tokens_median": 12.0,  # millions of tokens
    "total_tokens_ceiling": 220.0,  # millions of tokens
    "agent_turns_median": 125.0,  # turns
    "agent_turns_ceiling": 1500.0,  # turns
}


def compute_p50_shifted_log_auc(
    value: float,
    p50_grace: float,
    ceiling: float,
    log_span: float = 100.0,
    task_success: bool = True,
) -> float:
    """Calculates the P50-shifted Log-Weighted AUC score in [0.0, 1.0].

    Args:
        value: The raw metric value achieved in the run.
        p50_grace: The grace threshold (P50). Any value <= this scores 1.0.
        ceiling: The zero-score ceiling (Max). Any value >= this scores 0.0.
        log_span: The coordinate log-space span (default 100.0 maps to [1, 100]).
        task_success: Gating boolean. If False, the score is forced to 0.0.

    Returns:
        Float efficiency score in [0.0, 1.0].
    """
    if not task_success:
        return 0.0
    if value <= p50_grace:
        return 1.0
    if value >= ceiling:
        return 0.0

    # Step 1: Shift and map the excess above P50 onto [1.0, log_span]
    excess_ratio = (value - p50_grace) / (ceiling - p50_grace)
    u = 1.0 + (log_span - 1.0) * excess_ratio

    # Step 2: Compute logarithmic decay (slope matches 1/u)
    score = max(1.0 - (math.log(u) / math.log(log_span)), 0)
    return round(float(score), 4)


def load_stats(stats_path: Path | None = None) -> dict[str, float]:
    """Loads P50 median (grace) and ceiling values from stats.txt."""
    stats = dict(DEFAULT_STATS)
    candidates: list[Path] = []
    if stats_path is not None:
        candidates.append(Path(stats_path))
    candidates.extend([
        Path(__file__).resolve().parent / "stats.txt",
        Path("/tests/efficiency/stats.txt"),
    ])

    for cand in candidates:
        if cand.is_file():
            try:
                for raw_line in cand.read_text(encoding="utf-8").splitlines():
                    line = raw_line.split("#", 1)[0].strip()
                    if not line or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    try:
                        stats[key.strip()] = float(val.strip())
                    except ValueError:
                        pass
                break
            except OSError:
                pass
    return stats


def _load_trajectory(path: str | Path) -> dict[str, Any] | None:
    """Loads an ATIF trajectory JSON file. Returns None on error."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _find_trajectory(workspace: Path | None = None) -> dict[str, Any] | None:
    """Finds and loads the ATIF trajectory JSON from standard candidate locations."""
    traj_candidates: list[Path] = []
    if os.environ.get("ATIF_TRAJECTORY"):
        traj_candidates.append(Path(os.environ["ATIF_TRAJECTORY"]))
    if os.environ.get("TRAJECTORY_PATH"):
        traj_candidates.append(Path(os.environ["TRAJECTORY_PATH"]))
    # Harness-owned paths only: a trajectory.json in the workspace is the
    # agent's to write, so it cannot be what scores the agent.
    del workspace
    traj_candidates.extend([
        Path("/logs/agent/trajectory.json"),
        Path("/logs/trajectory.json"),
    ])

    for cand in traj_candidates:
        if cand and cand.exists():
            traj = _load_trajectory(cand)
            if traj:
                return traj
    return None


def _parse_iso(ts: str) -> float | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def extract_trajectory_metrics(traj: dict[str, Any]) -> dict[str, float]:
    """Extracts agent_runtime (minutes), total_tokens (millions), and agent_turns from ATIF trajectory."""
    steps = [s for s in (traj.get("steps") or []) if isinstance(s, dict)]

    # 1. Agent runtime in minutes (span from first to last step timestamp)
    timestamps: list[float] = []
    for s in steps:
        ts_str = s.get("timestamp")
        if isinstance(ts_str, str) and ts_str.strip():
            ts_val = _parse_iso(ts_str.strip())
            if ts_val is not None:
                timestamps.append(ts_val)
    if len(timestamps) >= 2:
        agent_runtime_min = round((max(timestamps) - min(timestamps)) / 60.0,
                                  4)
    else:
        agent_runtime_min = 0.0

    # 2. Agent turns (number of model/agent steps in trajectory)
    agent_turns = sum(1 for s in steps
                      if str(s.get("source") or "").strip().lower() == "agent")

    # 3. Total tokens (input + output) in millions
    final_metrics = traj.get("final_metrics")
    prompt_tokens = 0
    completion_tokens = 0
    if isinstance(final_metrics, dict):
        prompt_tokens = int(final_metrics.get("total_prompt_tokens") or 0)
        completion_tokens = int(
            final_metrics.get("total_completion_tokens") or 0)

    if prompt_tokens + completion_tokens <= 0:
        prompt_tokens = 0
        completion_tokens = 0
        for s in steps:
            m = s.get("metrics")
            if isinstance(m, dict):
                prompt_tokens += int(m.get("prompt_tokens") or 0)
                completion_tokens += int(m.get("completion_tokens") or 0)

    total_tokens_millions = round((prompt_tokens + completion_tokens) / 1e6, 4)

    return {
        "agent_runtime_min": agent_runtime_min,
        "total_tokens_millions": total_tokens_millions,
        "agent_turns": float(agent_turns),
    }


def agent_runtime_auc(workspace: Path) -> float:
    """Calculates P50-shifted Log-Weighted AUC for agent runtime in minutes."""
    traj = _find_trajectory(workspace)
    if not traj:
        print("agent_runtime_auc: [FAIL] No trajectory found in candidates")
        return 0.0

    stats = load_stats()
    metrics = extract_trajectory_metrics(traj)
    val = metrics["agent_runtime_min"]
    p50 = stats["agent_runtime_median"]
    ceil = stats["agent_runtime_ceiling"]
    score = compute_p50_shifted_log_auc(val,
                                        p50_grace=p50,
                                        ceiling=ceil,
                                        task_success=(val > 0))
    print(f"agent_runtime_auc: runtime={val:.2f} min "
          f"(p50_grace={p50:.1f}, ceiling={ceil:.1f}) -> score={score:.4f}")
    return score


def total_tokens_auc(workspace: Path) -> float:
    """Calculates P50-shifted Log-Weighted AUC for total tokens (input + output, in millions)."""
    traj = _find_trajectory(workspace)
    if not traj:
        print("total_tokens_auc: [FAIL] No trajectory found in candidates")
        return 0.0

    stats = load_stats()
    metrics = extract_trajectory_metrics(traj)
    val = metrics["total_tokens_millions"]
    p50 = stats["total_tokens_median"]
    ceil = stats["total_tokens_ceiling"]
    score = compute_p50_shifted_log_auc(val,
                                        p50_grace=p50,
                                        ceiling=ceil,
                                        task_success=(val > 0))
    print(f"total_tokens_auc: total_tokens={val:.4f}M "
          f"(p50_grace={p50:.1f}M, ceiling={ceil:.1f}M) -> score={score:.4f}")
    return score


def agent_turns_auc(workspace: Path) -> float:
    """Calculates P50-shifted Log-Weighted AUC for agent turns."""
    traj = _find_trajectory(workspace)
    if not traj:
        print("agent_turns_auc: [FAIL] No trajectory found in candidates")
        return 0.0

    stats = load_stats()
    metrics = extract_trajectory_metrics(traj)
    val = metrics["agent_turns"]
    p50 = stats["agent_turns_median"]
    ceil = stats["agent_turns_ceiling"]
    score = compute_p50_shifted_log_auc(val,
                                        p50_grace=p50,
                                        ceiling=ceil,
                                        task_success=(val > 0))
    print(f"agent_turns_auc: turns={int(val)} "
          f"(p50_grace={p50:.0f}, ceiling={ceil:.0f}) -> score={score:.4f}")
    return score


@criterion(
    description=("Average P50-shifted Log-Weighted AUC across agent runtime, "
                 "total tokens, and agent turns"))
def trajectory_score(workspace: Path) -> float:
    """Calculates the average P50-shifted Log-Weighted AUC of runtime, tokens, and turns."""
    runtime_score = agent_runtime_auc(workspace)
    tokens_score = total_tokens_auc(workspace)
    turns_score = agent_turns_auc(workspace)
    trajectory_score = round(
        (runtime_score + tokens_score + turns_score) / 3.0, 4)
    print(f"trajectory_score: runtime_auc={runtime_score:.4f}, "
          f"tokens_auc={tokens_score:.4f}, turns_auc={turns_score:.4f} "
          f"-> trajectory_score={trajectory_score:.4f}")
    return trajectory_score
