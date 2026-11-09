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
  uvx --with google-auth --from "harbor-rewardkit>=0.1.7" rewardkit /tests --workspace /workspace --max-concurrent-llm 2 && EXIT_CODE=0 && break || EXIT_CODE=$?
  if [ "$attempt" -lt "$MAX_VERIFIER_ATTEMPTS" ]; then
    echo "[!] Verifier encountered an error (exit code $EXIT_CODE), waiting ${RETRY_DELAY}s before retrying..."
    sleep "$RETRY_DELAY"
    RETRY_DELAY=$((RETRY_DELAY * 2))
  fi
done

exit "$EXIT_CODE"
