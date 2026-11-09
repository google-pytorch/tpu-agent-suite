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

# Branched from: eval/datasets/nightly-onboard-optimize/*/tests/test.sh
# Purpose: Canonical test runner for onboard & hillclimb regrading.
# Key modifications from original:
#   1. Deterministic hillclimb_success: Computed directly from verified speedup
#      (>= 1.01) rather than an LLM judge.
#   2. Fallback weights: Pre-populates default weights dictionary so scoring
#      succeeds reliably even if reward.toml is missing in test environments.

# Output columns in /logs/verifier/reward.json:
#   onboarding_success: Returns 1 if onboarding is successful, 0 otherwise.
#   hillclimb_success: Gated on onboarding_success (hillclimb_success_raw * onboarding_success). Returns 1 if both onboarding and hillclimb were successful, 0 otherwise.
#   speedup_score: Gated on onboarding_success and hillclimb_success, clamped between 0 and 1:
#     clip(log(max(speedup, 1.0)) / log(target_speedup * c), 0.0, 1.0) * onboarding_success * hillclimb_success (where c = 2.0).
#   efficiency: Gated on onboarding_success and hillclimb_success (efficiency_raw * onboarding_success * hillclimb_success). Returns the ratio of valid tool calls to total tool calls.
#   target_speedup: The target speedup ratio, read from a file in tests/speedup/.
#   speedup: The raw speedup ratio reported by the verifier judge (e.g., from TPUspeedometer), not yet clamped or scaled.
#   cheating: A binary flag (0 or 1) set by the verifier judge when the agent exhibits cheating behavior. Nullifies the reward when 1.
#   invoked_subagent: Returns 1 if a TPU agent suite subagent was invoked, 0 otherwise.
#   reward: The final reward, clamped to [0, 1]:
#     reward = (w_onboarding * onboarding_success
#               + w_hillclimb * hillclimb_success
#               + w_speedup * speedup_score
#               + w_efficiency * efficiency) * (1 - cheating)
#     where weights are read from reward.toml (0.3 * onboarding_success + 0.3 * hillclimb_success + 0.3 * speedup_score + 0.1 * efficiency).

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

# rewardkit has no multiplicative aggregation: rewrite `reward` with the
# scoring framework's exact formula (see reward.toml: hillclimb_success gated
# on onboarding_success, speedup_score and efficiency on both, then the cheating
# gate), and copy the judge's raw TPU speedup ratio over `speedup` and the run
# report's optimization step count to `optimization_steps`, then print the run
# report.
if [ "$EXIT_CODE" -eq 0 ]; then
  PYTHON_BIN=$(command -v python3 || true)
  if [ -n "$PYTHON_BIN" ]; then
    PYTHON=("$PYTHON_BIN")
  else
    PYTHON=(uv run --no-project python)
  fi
  "${PYTHON[@]}" - <<'PY' || echo "[!] Could not apply the cheating gate; keeping rewardkit's reward."
import json
import math
from pathlib import Path

out = Path("/logs/verifier/reward.json")
scores = json.loads(out.read_text())


def find_raw(node, name):
    if isinstance(node, dict):
        if node.get("name") == name and "raw" in node:
            return node["raw"]
        node = list(node.values())
    if isinstance(node, list):
        for child in node:
            raw = find_raw(child, name)
            if raw is not None:
                return raw
    return None


def find_criterion(node, name):
    if isinstance(node, dict):
        if node.get("name") == name and ("value" in node or "raw" in node):
            return node
        node = list(node.values())
    if isinstance(node, list):
        for child in node:
            res = find_criterion(child, name)
            if res is not None:
                return res
    return None


try:
    details = json.loads(out.with_name("reward-details.json").read_text())
    raw = find_raw(details.get("speedup"), "tpu_speedup")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        # rewardkit stores the ratio rescaled to 0-1 (divided by the
        # criterion's max); report the ratio itself (1.0 = baseline).
        scores["speedup"] = round(float(raw), 4)
    steps = find_raw(details.get("execution_report"), "steps")
    if isinstance(steps, (int, float)) and not isinstance(steps, bool):
        scores["optimization_steps"] = int(steps)
    eff_node = details.get("efficiency")
    for ename in ("trajectory_score", "valid_tool_calls", "framework_directed_penalty"):
        enode = find_criterion(eff_node, ename)
        if enode is not None:
            ev = enode.get("value")
            if not isinstance(ev, (int, float)) or isinstance(ev, bool):
                ev = enode.get("raw")
            if isinstance(ev, (int, float)) and not isinstance(ev, bool):
                scores[ename] = round(float(ev), 4)
except (OSError, ValueError):
    pass

c = 2.0
target_speedup = 1.0
for candidate in (
    Path("/tests/speedup/target_speedup.txt"),
    Path("tests/speedup/target_speedup.txt"),
):
    if candidate.is_file():
        try:
            target_speedup = float(candidate.read_text().strip())
            break
        except ValueError:
            pass
target_speedup_w_headroom = target_speedup * c
scores["target_speedup"] = target_speedup

weights = {
    "onboarding_success": 0.3,
    "hillclimb_success": 0.3,
    "speedup": 0.3,
    "efficiency": 0.1,
}
for candidate in (Path("/tests/reward.toml"), Path("tests/reward.toml")):
    if candidate.is_file():
        in_weights = False
        for line in candidate.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("["):
                in_weights = line == "[reward.weights]"
            elif in_weights and "=" in line:
                k, v = line.split("=", 1)
                weights[k.strip()] = float(v.strip())
        break

parts = ("onboarding_success", "efficiency", "cheating")
if all(isinstance(scores.get(p), (int, float)) for p in parts):
    onboarding = scores["onboarding_success"]
    speedup_val = scores.get("speedup", 0.0)
    if not isinstance(speedup_val, (int, float)) or isinstance(speedup_val, bool):
        speedup_val = 0.0
    # Hill-climbing succeeds deterministically if throughput speedup is >= 1% (>= 1.01).
    # No hill-climbing without an onboarded model, and no speedup or efficiency
    # credit unless both the onboarding and the hill-climbing succeeded.
    hillclimb_raw = 1.0 if speedup_val >= 1.01 else 0.0
    hillclimb = hillclimb_raw * onboarding
    scores["hillclimb_success_raw"] = hillclimb_raw
    trajectory_score = float(scores.get("trajectory_score", 1.0))
    valid_tool_calls = float(scores.get("valid_tool_calls", scores["efficiency"]))
    framework_directed_penalty = float(scores.get("framework_directed_penalty", 1.0))
    scores["trajectory_score"] = round(trajectory_score, 4)
    scores["valid_tool_calls"] = round(valid_tool_calls, 4)
    scores["framework_directed_penalty"] = round(framework_directed_penalty, 4)
    efficiency_raw = trajectory_score * valid_tool_calls * framework_directed_penalty
    scores["efficiency_raw"] = round(efficiency_raw, 4)
    speedup_score_raw = min(
        1.0,
        max(
            0.0,
            math.log(max(float(speedup_val), 1.0))
            / math.log(target_speedup_w_headroom),
        ),
    )
    speedup_score = speedup_score_raw * onboarding * hillclimb
    efficiency = efficiency_raw * onboarding * hillclimb
    scores["hillclimb_success"] = round(hillclimb, 4)
    scores["speedup_score"] = round(speedup_score, 4)
    scores["efficiency"] = round(efficiency, 4)
    base = (
        weights["onboarding_success"] * onboarding
        + weights["hillclimb_success"] * hillclimb
        + weights["speedup"] * speedup_score
        + weights["efficiency"] * efficiency
    )
    gated = base * (1.0 - scores["cheating"])
    scores["reward"] = round(min(1.0, max(0.0, gated)), 4)
output_cols = [
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
    "reward",
]

out.write_text(json.dumps({k: scores[k] for k in output_cols if k in scores}, indent=2))
print(json.dumps({k: scores[k] for k in output_cols if k in scores}, indent=2))


# Print the run report (routing, steps, errors) to the verifier log.
try:
    report = details.get("execution_report") or {}
    for crit in report.get("criteria", []):
        print(f"\n=== execution_report/{crit.get('name')}: raw={crit.get('raw')} ===")
        print(crit.get("reasoning") or crit.get("error") or "(no reasoning)")
except NameError:
    pass
PY
fi

exit "$EXIT_CODE"
