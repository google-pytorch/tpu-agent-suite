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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Support running the preserved legacy regex verifier via LEGACY_REGEX_VERIFIER=1
if [ "${LEGACY_REGEX_VERIFIER:-0}" = "1" ] && [ -f "${SCRIPT_DIR}/test_legacy_regex.sh" ]; then
  exec bash "${SCRIPT_DIR}/test_legacy_regex.sh" "$@"
fi

export TESTS_DIR="${1:-${TESTS_DIR:-${SCRIPT_DIR}}}"
export WORKSPACE_DIR="${2:-${WORKSPACE_DIR:-/workspace}}"
export LOGS_DIR="${3:-${LOGS_DIR:-/logs}}"

mkdir -p "${LOGS_DIR}/verifier"
echo '{"reward": 0.0}' > "${LOGS_DIR}/verifier/reward.json"
cd "${WORKSPACE_DIR}" 2>/dev/null || true

echo "=== Running Harbor verifier (Onboarding-Only RewardKit Suite) ==="
echo "=== Running RewardKit Judge ==="

mkdir -p "${LOGS_DIR}/agent"
if [ -n "$ATIF_TRAJECTORY" ] && [ -f "$ATIF_TRAJECTORY" ]; then
  cp "$ATIF_TRAJECTORY" "${LOGS_DIR}/agent/trajectory.json" 2>/dev/null || true
  cp "$ATIF_TRAJECTORY" "${LOGS_DIR}/trajectory.json" 2>/dev/null || true
elif [ -f "${LOGS_DIR}/trajectory.json" ] && [ ! -f "${LOGS_DIR}/agent/trajectory.json" ]; then
  cp "${LOGS_DIR}/trajectory.json" "${LOGS_DIR}/agent/trajectory.json" 2>/dev/null || true
elif [ -f "${LOGS_DIR}/agent/trajectory.json" ] && [ ! -f "${LOGS_DIR}/trajectory.json" ]; then
  cp "${LOGS_DIR}/agent/trajectory.json" "${LOGS_DIR}/trajectory.json" 2>/dev/null || true
fi

export TRAJECTORY_PATH="${ATIF_TRAJECTORY:-${LOGS_DIR}/agent/trajectory.json}"

# If running in a container or offline with a custom model name or trajectory path,
# stage tests to a temporary directory if judge.toml contains {MODEL_NAME} or if
# LOGS_DIR != /logs.
ACTIVE_TESTS_DIR="${TESTS_DIR}"
STAGE_TMP=""
if grep -q "{MODEL_NAME}" "${TESTS_DIR}"/*/judge.toml 2>/dev/null || [ "${LOGS_DIR}" != "/logs" ]; then
  STAGE_TMP="$(mktemp -d)"
  cp -r "${TESTS_DIR}/." "${STAGE_TMP}/"
  MODEL_LABEL="${ONBOARDING_MODEL_NAME:-${HF_MODEL_URL:-the target model}}"
  MODEL_LABEL="${MODEL_LABEL#https://huggingface.co/}"
  for jf in "${STAGE_TMP}"/*/judge.toml; do
    [ -f "$jf" ] || continue
    python3 - "$jf" "$TRAJECTORY_PATH" "$MODEL_LABEL" <<'PYSTAGE'
import pathlib, sys
jf = pathlib.Path(sys.argv[1])
traj = sys.argv[2]
model = sys.argv[3]
txt = jf.read_text(encoding="utf-8")
txt = txt.replace('atif-trajectory = "/logs/agent/trajectory.json"', f'atif-trajectory = "{traj}"')
txt = txt.replace("{MODEL_NAME}", model)
jf.write_text(txt, encoding="utf-8")
PYSTAGE
  done
  ACTIVE_TESTS_DIR="${STAGE_TMP}"
fi

if ! command -v uvx &> /dev/null; then
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
    rewardkit "${ACTIVE_TESTS_DIR}" --workspace "${WORKSPACE_DIR}" --output "${LOGS_DIR}/verifier/reward.json" --max-concurrent-llm 2 && EXIT_CODE=0 && break || EXIT_CODE=$?
  else
    uvx --python 3.12 --with google-auth --from "harbor-rewardkit>=0.2.1" rewardkit "${ACTIVE_TESTS_DIR}" --workspace "${WORKSPACE_DIR}" --output "${LOGS_DIR}/verifier/reward.json" --max-concurrent-llm 2 && EXIT_CODE=0 && break || EXIT_CODE=$?
  fi
  if [ "$attempt" -lt "$MAX_VERIFIER_ATTEMPTS" ]; then
    echo "[!] Verifier encountered an error (exit code $EXIT_CODE), waiting ${RETRY_DELAY}s before retrying..."
    sleep "$RETRY_DELAY"
    RETRY_DELAY=$((RETRY_DELAY * 2))
  fi
done

if [ -n "$STAGE_TMP" ] && [ -d "$STAGE_TMP" ]; then
  rm -rf "$STAGE_TMP"
fi

if [ "$EXIT_CODE" -eq 0 ]; then
  PYTHON_BIN=$(command -v python3 || true)
  if [ -n "$PYTHON_BIN" ]; then
    PYTHON=("$PYTHON_BIN")
  else
    PYTHON=(uv run --no-project python)
  fi
  "${PYTHON[@]}" - <<'PY' || echo "[!] Could not post-process reward.json; keeping rewardkit's reward."
import json
import os
from pathlib import Path

logs_dir = Path(os.environ.get("LOGS_DIR", "/logs"))
tests_dir = Path(os.environ.get("TESTS_DIR", "/tests"))
out = logs_dir / "verifier" / "reward.json"
scores = json.loads(out.read_text(encoding="utf-8"))


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
    onboard_Node = details.get("onboarding_success")
    for cname in (
        "vllm_server_on_tpu",
        "completion_over_http",
        "baseline_benchmarks_reported",
        "baseline_reproduce_commands",
    ):
        cnode = find_criterion(onboard_Node, cname)
        if cnode is not None:
            v = cnode.get("value")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                fval = 1.0 if float(v) >= 0.5 else 0.0
            else:
                r = str(cnode.get("raw", "")).strip().lower()
                fval = 1.0 if r in ("yes", "true", "1", "1.0", "pass") else 0.0
            scores[cname] = fval
    steps_node = find_criterion(details.get("execution_report"), "steps")
    if steps_node is not None and isinstance(steps_node.get("raw"), (int, float)):
        scores["regimes_measured_judge"] = int(steps_node["raw"])
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

# Merge deterministic regime_sweep_summary.json if produced
sweep_file = logs_dir / "verifier" / "regime_sweep_summary.json"
if sweep_file.is_file():
    try:
        sweep_data = json.loads(sweep_file.read_text(encoding="utf-8"))
        scores["regime_coverage"] = float(sweep_data.get("regime_coverage", 0.0))
        scores["peak_tps_1k_1k"] = float(sweep_data.get("peak_tps_1k_1k", 0.0))
        scores["peak_tps_1k_8k"] = float(sweep_data.get("peak_tps_1k_8k", 0.0))
        scores["peak_tps_8k_1k"] = float(sweep_data.get("peak_tps_8k_1k", 0.0))
        if sweep_file in details:
            pass
        details["regime_sweep_details"] = sweep_data
        out.with_name("reward-details.json").write_text(
            json.dumps(details, indent=2), encoding="utf-8"
        )
    except (OSError, ValueError):
        pass

weights = {}
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
                in_weights = line in ("[reward.weights]", "[weights]")
            elif in_weights and "=" in line:
                k, v = line.split("=", 1)
                try:
                    weights[k.strip()] = float(v.strip())
                except ValueError:
                    pass
        break

onboard_val = scores.get("onboarding_success")

if onboard_val is not None and isinstance(scores.get("efficiency"), (int, float)):
    onboarding = round(float(onboard_val), 4)
    scores["onboarding_success"] = onboarding

    # Only calculate efficiency if onboarding is a success (all DoD criteria pass)
    trajectory_score = float(scores.get("trajectory_score", 1.0))
    valid_tool_calls = float(scores.get("valid_tool_calls", scores["efficiency"]))
    framework_directed_penalty = float(scores.get("framework_directed_penalty", 1.0))
    scores["trajectory_score"] = round(trajectory_score, 4)
    scores["valid_tool_calls"] = round(valid_tool_calls, 4)
    scores["framework_directed_penalty"] = round(framework_directed_penalty, 4)
    efficiency_raw = trajectory_score * valid_tool_calls * framework_directed_penalty
    scores["efficiency_raw"] = round(efficiency_raw, 4)
    efficiency = round(efficiency_raw * onboarding, 4)
    scores["efficiency"] = efficiency

    w_onboarding = weights.get("onboarding_success", 0.6)
    w_efficiency = weights.get("efficiency", 0.4)

    base = w_onboarding * onboarding + w_efficiency * efficiency
    cheating = float(scores.get("cheating", 0.0))
    gated = base * (1.0 - cheating)
    scores["reward"] = round(min(1.0, max(0.0, gated)), 4)

output_cols = [
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
    "invoked_subagent",
]

filtered = {k: scores[k] for k in output_cols if k in scores}
for k, v in scores.items():
    if (
        k not in filtered
        and not k.endswith("_raw")
        and not k.endswith("_gated")
        and k not in ("regime_sweep", "onboarding_all_pass")
    ):
        filtered[k] = v

out.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
(logs_dir / "verifier" / "reward.txt").write_text(f"{filtered.get('reward', 0.0)}\n", encoding="utf-8")
print(json.dumps(filtered, indent=2))

try:
    for section in ("onboarding_success", "cheating", "execution_report"):
        report = details.get(section) or {}
        for crit in report.get("criteria", []):
            print(f"\n=== {section}/{crit.get('name')}: raw={crit.get('raw')} ===")
            print(crit.get("reasoning") or crit.get("error") or "(no reasoning)")
except Exception:
    pass
PY
fi

exit "$EXIT_CODE"
