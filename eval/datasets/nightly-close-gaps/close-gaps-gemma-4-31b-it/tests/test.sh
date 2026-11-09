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


# Output columns in /logs/verifier/reward.json:
#   reward: The 1k/1k gain per hour of agent wall-clock time:
#     reward = (speedup - 1) / agent_hours * (1 - cheating)
#     For example, a 1.5x speedup in 4 hours scores 0.125. A slowdown scores
#     below 0. The reward is 0 if the speedup or agent_hours is unknown.
#   speedup: optimized_tps / baseline_tps, as written in benchmark_results.json. 1.0 means no change.
#   cheating: A binary flag (0 or 1) set by the cheating judge.
#   execution_report: The execution report judge's score. Reported only.
#
# Campaign metrics, reported only:
#   baseline_tps, optimized_tps: The 1k/1k output tokens/s of the baseline and the optimized server, as written in benchmark_results.json.
#   roofline_tps: The whole-model roofline 1k/1k output tokens/s that the agent computed, as written in benchmark_results.json.
#   roofline_saturation: optimized_tps / roofline_tps.
#   prompt_tokens, completion_tokens, cached_tokens, total_tokens: The LLM token use, read from final_metrics in the trajectory. total_tokens is prompt_tokens + completion_tokens.
#   agent_hours: The agent's wall-clock time in hours, from the first to the last timestamped step of its trajectory.

mkdir -p /logs/verifier
echo '{"reward": 0.0}' > /logs/verifier/reward.json
cd /workspace || exit

echo "=== Running Harbor verifier ==="
echo "=== Running RewardKit Judge ==="

mkdir -p /logs/agent
if [ -f "/logs/trajectory.json" ] && [ ! -f "/logs/agent/trajectory.json" ]; then
  cp /logs/trajectory.json /logs/agent/trajectory.json 2>/dev/null || true
elif [ -f "/logs/agent/trajectory.json" ] && [ ! -f "/logs/trajectory.json" ]; then
  cp /logs/agent/trajectory.json /logs/trajectory.json 2>/dev/null || true
fi

if ! command -v uvx &> /dev/null; then
  echo "[*] Installing uv for verifier..."
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh > /dev/null 2>&1 || true
  export PATH="/usr/local/bin:$HOME/.cargo/bin:$PATH"
fi

export DEFAULT_MAX_RETRIES=10
export INITIAL_RETRY_DELAY=2.0
export MAX_RETRY_DELAY=60.0

MAX_VERIFIER_ATTEMPTS=5
RETRY_DELAY=5

for attempt in $(seq 1 "$MAX_VERIFIER_ATTEMPTS"); do
  echo "[*] Running verifier (attempt $attempt/$MAX_VERIFIER_ATTEMPTS)..."
  uvx --with google-auth --from "harbor-rewardkit>=0.2.2" rewardkit /tests --workspace /workspace --max-concurrent-llm 2 && EXIT_CODE=0 && break || EXIT_CODE=$?
  if [ "$attempt" -lt "$MAX_VERIFIER_ATTEMPTS" ]; then
    echo "[!] Verifier encountered an error (exit code $EXIT_CODE), waiting ${RETRY_DELAY}s before retrying..."
    sleep "$RETRY_DELAY"
    RETRY_DELAY=$((RETRY_DELAY * 2))
  fi
done

# rewardkit writes only the graders' and judges' scores. This step reads the
# reported numbers with the speedup grader's reader, adds the campaign
# metrics, writes `reward` with the formula above, and prints the run report.
if [ "$EXIT_CODE" -eq 0 ]; then
  PYTHON_BIN=$(command -v python3 || true)
  if [ -n "$PYTHON_BIN" ]; then
    PYTHON=("$PYTHON_BIN")
  else
    PYTHON=(uv run --no-project python)
  fi
  "${PYTHON[@]}" - <<'PY' || { echo "[!] Could not compute the reward; writing 0."; echo '{"reward": 0.0}' > /logs/verifier/reward.json; }
import json
import sys
import types
from datetime import datetime
from pathlib import Path

out = Path("/logs/verifier/reward.json")
scores = json.loads(out.read_text())


def agent_hours(path):
    """Returns the agent's wall-clock time from its trajectory's timestamps.

    Args:
        path: The Harbor ATIF trajectory file.

    Returns:
        The hours from the first to the last timestamped step, or None if the
        trajectory cannot be read or has fewer than two timestamps.
    """
    try:
        steps = json.loads(path.read_text())["steps"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    times = []
    for step in steps:
        try:
            times.append(datetime.fromisoformat(step["timestamp"]))
        except (KeyError, TypeError, ValueError):
            continue
    if len(times) < 2:
        return None
    return (max(times) - min(times)).total_seconds() / 3600.0


def reported_numbers():
    """Returns the 1k/1k numbers that the agent wrote to benchmark_results.json.

    Uses the speedup grader's reader, so that the reward and the campaign
    metrics come from the same parse of the same file.

    Returns:
        A dict with baseline_tps, optimized_tps, roofline_tps and speedup. A
        value is None if it is missing.
    """
    # The grader imports rewardkit only for its @criterion decorator.
    sys.modules.setdefault(
        "rewardkit", types.SimpleNamespace(criterion=lambda **_: lambda f: f))
    sys.path.insert(0, "/tests/speedup")
    import calculate_speedup  # pylint: disable=import-outside-toplevel

    return calculate_speedup.reported()


def token_use(path):
    """Returns the LLM token use recorded in the trajectory.

    Args:
        path: The Harbor ATIF trajectory file.

    Returns:
        A dict with prompt_tokens, completion_tokens, cached_tokens and
        total_tokens. It is empty if the trajectory has no final_metrics.
    """
    try:
        metrics = json.loads(path.read_text()).get("final_metrics") or {}
    except (OSError, ValueError, AttributeError):
        return {}
    use = {
        "prompt_tokens": metrics.get("total_prompt_tokens"),
        "completion_tokens": metrics.get("total_completion_tokens"),
        "cached_tokens": metrics.get("total_cached_tokens"),
    }
    if isinstance(use["prompt_tokens"], int) and isinstance(
            use["completion_tokens"], int):
        use["total_tokens"] = use["prompt_tokens"] + use["completion_tokens"]
    return {k: v for k, v in use.items() if v is not None}


trajectory = Path("/logs/agent/trajectory.json")
hours = agent_hours(trajectory)
scores["agent_hours"] = None if hours is None else round(hours, 3)
scores.update(token_use(trajectory))
numbers = reported_numbers()
scores.update(numbers)
opt_tps, roof = numbers["optimized_tps"], numbers["roofline_tps"]
scores["roofline_saturation"] = (round(opt_tps / roof, 4)
                                 if opt_tps and roof else None)

scores["reward"] = 0.0
if not isinstance(scores.get("cheating"), (int, float)):
    print("[!] The cheating judge returned no score; the reward is 0.")
else:
    speedup_val = scores.get("speedup")
    if (not isinstance(speedup_val, (int, float)) or isinstance(speedup_val, bool)
            or speedup_val <= 0):
        print("[!] The speedup is unknown; the reward is 0.")
        scores["reward"] = 0.0
    elif not hours:
        print("[!] agent_hours is unknown; the reward is 0.")
        scores["reward"] = 0.0
    else:
        gain_per_hour = (float(speedup_val) - 1.0) / hours
        scores["reward"] = round(gain_per_hour * (1.0 - scores["cheating"]), 4)
output_cols = [
    "reward",
    "speedup",
    "cheating",
    "execution_report",
    # Campaign metrics, reported only.
    "baseline_tps",
    "optimized_tps",
    "roofline_tps",
    "roofline_saturation",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "total_tokens",
    "agent_hours",
]

out.write_text(json.dumps({k: scores[k] for k in output_cols if k in scores}, indent=2))
print(json.dumps({k: scores[k] for k in output_cols if k in scores}, indent=2))


# Print the run report (routing, steps, errors) to the verifier log.
try:
    details = json.loads(out.with_name("reward-details.json").read_text())
    report = details.get("execution_report") or {}
    for crit in report.get("criteria", []):
        print(f"\n=== execution_report/{crit.get('name')}: raw={crit.get('raw')} ===")
        print(crit.get("reasoning") or crit.get("error") or "(no reasoning)")
except (OSError, ValueError, AttributeError):
    pass
PY
fi

exit "$EXIT_CODE"
