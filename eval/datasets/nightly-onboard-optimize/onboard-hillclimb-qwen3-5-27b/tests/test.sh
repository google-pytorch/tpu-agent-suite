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

# Canonical Harbor verifier for two-phase onboarding + per-chip hill-climbing
# tasks (`onboard-hillclimb-*`).
#
# Output columns in /logs/verifier/reward.json:
#   onboarding_success: Returns 1 if Phase 1 onboarding succeeds, 0 otherwise.
#   hillclimb_success: Gated on onboarding_success (hillclimb_success_raw * onboarding_success). Returns 1 if both onboarding succeeded and verified output_tps_per_chip speedup >= 1.01, 0 otherwise.
#   speedup_score: Gated on onboarding_success and hillclimb_success, clamped between 0 and 1:
#     clip(log(max(speedup, 1.0)) / log(target_speedup * c), 0.0, 1.0) * onboarding_success * hillclimb_success (where c = 2.0).
#   efficiency: Gated on onboarding_success and hillclimb_success (efficiency_raw * onboarding_success * hillclimb_success).
#   target_speedup: The target speedup ratio, read from tests/speedup/target_speedup.txt.
#   speedup: The raw verified output_tps_per_chip speedup ratio on 8k/1k (1.0 = baseline).
#   cheating: Binary flag (0 or 1) set by the anti-cheating judge; nullifies reward when 1.
#   invoked_subagent: Returns 1 if a TPU agent suite subagent was invoked, 0 otherwise.
#   schema_compliance: Fraction of TPU agent suite delegations whose request and result met their schemas.
#   reward: The final reward, clamped to [0, 1]:
#     reward = (0.3 * onboarding_success
#               + 0.3 * hillclimb_success
#               + 0.3 * speedup_score
#               + 0.1 * efficiency) * (1 - cheating)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TESTS_DIR="${1:-${TESTS_DIR:-${SCRIPT_DIR}}}"
export WORKSPACE_DIR="${2:-${WORKSPACE_DIR:-/workspace}}"
export LOGS_DIR="${3:-${LOGS_DIR:-/logs}}"

mkdir -p "${LOGS_DIR}/verifier"
echo '{"reward": 0.0}' > "${LOGS_DIR}/verifier/reward.json"
cd "${WORKSPACE_DIR}" 2>/dev/null || exit

echo "=== Running Harbor verifier (Onboard + Hill-Climb RewardKit Suite) ==="
echo "=== Running RewardKit Judge ==="

mkdir -p "${LOGS_DIR}/agent"
if [ -f "${LOGS_DIR}/trajectory.json" ] && [ ! -f "${LOGS_DIR}/agent/trajectory.json" ]; then
  cp "${LOGS_DIR}/trajectory.json" "${LOGS_DIR}/agent/trajectory.json" 2>/dev/null || true
elif [ -f "${LOGS_DIR}/agent/trajectory.json" ] && [ ! -f "${LOGS_DIR}/trajectory.json" ]; then
  cp "${LOGS_DIR}/agent/trajectory.json" "${LOGS_DIR}/trajectory.json" 2>/dev/null || true
fi

if ! command -v rewardkit &> /dev/null && ! command -v uvx &> /dev/null; then
  echo "[*] Installing uv for verifier..."
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh > /dev/null 2>&1 || true
  export PATH="/usr/local/bin:$HOME/.cargo/bin:$PATH"
fi

export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-global}"
export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-tpu-nexus-767207}"
export CLOUD_ML_REGION="${CLOUD_ML_REGION:-global}"
export DEFAULT_MAX_RETRIES="${DEFAULT_MAX_RETRIES:-10}"
export INITIAL_RETRY_DELAY="${INITIAL_RETRY_DELAY:-2.0}"
export MAX_RETRY_DELAY="${MAX_RETRY_DELAY:-60.0}"

MAX_VERIFIER_ATTEMPTS=5
RETRY_DELAY=5

for attempt in $(seq 1 "$MAX_VERIFIER_ATTEMPTS"); do
  echo "[*] Running verifier (attempt $attempt/$MAX_VERIFIER_ATTEMPTS)..."
  if command -v rewardkit &> /dev/null; then
    rewardkit "${TESTS_DIR}" --workspace "${WORKSPACE_DIR}" --output "${LOGS_DIR}/verifier/reward.json" --max-concurrent-llm 2 && EXIT_CODE=0 && break || EXIT_CODE=$?
  else
    uvx --with google-auth --from "harbor-rewardkit>=0.2.2" rewardkit "${TESTS_DIR}" --workspace "${WORKSPACE_DIR}" --output "${LOGS_DIR}/verifier/reward.json" --max-concurrent-llm 2 && EXIT_CODE=0 && break || EXIT_CODE=$?
  fi
  if [ "$attempt" -lt "$MAX_VERIFIER_ATTEMPTS" ]; then
    echo "[!] Verifier encountered an error (exit code $EXIT_CODE), waiting ${RETRY_DELAY}s before retrying..."
    sleep "$RETRY_DELAY"
    RETRY_DELAY=$((RETRY_DELAY * 2))
  fi
done

# rewardkit has no multiplicative aggregation: rewrite `reward` with the
# scoring framework's exact formula (see reward.toml: hillclimb_success gated
# on onboarding_success, speedup_score and efficiency on both, then the cheating
# gate), copy the judge's raw TPU per-chip speedup ratio over `speedup`, and
# print the run report.
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
import os
from pathlib import Path

logs_dir = Path(os.environ.get("LOGS_DIR", "/logs"))
tests_dir = Path(os.environ.get("TESTS_DIR", "/tests"))
workspace_dir = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
out = logs_dir / "verifier" / "reward.json"
scores = json.loads(out.read_text(encoding="utf-8"))


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


details = {}
try:
    details = json.loads(out.with_name("reward-details.json").read_text(encoding="utf-8"))
    raw = find_raw(details.get("speedup"), "tpu_speedup")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
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

# Record diagnostic Phase 2 XProf deliverables in reward-details.json.
try:
    search_roots = [p for p in (logs_dir / "artifacts", workspace_dir) if p.is_dir()]
    xplanes = {
        p.resolve()
        for r in search_roots
        for p in r.rglob("*.xplane.pb")
        if p.is_file() and p.stat().st_size > 0
    }
    gap_docs = {
        p.resolve()
        for r in search_roots
        for p in r.rglob("xprof_bottlenecks_and_gaps.md")
        if p.is_file() and p.stat().st_size >= 100
    }
    details["post_hillclimb_xprof"] = {
        "xplane_trace_count": len(xplanes),
        "gaps_report_count": len(gap_docs),
        "passed": bool(xplanes and gap_docs),
    }
    if out.with_name("reward-details.json").is_file():
        out.with_name("reward-details.json").write_text(
            json.dumps(details, indent=2), encoding="utf-8"
        )
except OSError:
    pass

c = 2.0
target_speedup = 1.0
for candidate in (
    tests_dir / "speedup" / "target_speedup.txt",
    Path("/tests/speedup/target_speedup.txt"),
    Path("tests/speedup/target_speedup.txt"),
):
    if candidate.is_file():
        try:
            target_speedup = float(candidate.read_text(encoding="utf-8").strip())
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
for candidate in (
    tests_dir / "reward.toml",
    Path("/tests/reward.toml"),
    Path("tests/reward.toml"),
):
    if candidate.is_file():
        in_weights = False
        for line in candidate.read_text(encoding="utf-8").splitlines():
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
    # Hill-climbing succeeds deterministically if verified per-chip throughput speedup is >= 1% (>= 1.01).
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
    "schema_compliance",
    "reward",
]

out.write_text(json.dumps({k: scores[k] for k in output_cols if k in scores}, indent=2), encoding="utf-8")
print(json.dumps({k: scores[k] for k in output_cols if k in scores}, indent=2))

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
