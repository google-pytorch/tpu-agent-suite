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

mkdir -p /logs/verifier /logs/agent
echo '{"reward": 0.0}' > /logs/verifier/reward.json
cd /workspace || exit 1

echo "=== Running Harbor verifier ==="
echo "=== Running RewardKit Judge ==="

if [ -f "/logs/trajectory.json" ] && [ ! -f "/logs/agent/trajectory.json" ]; then
  cp /logs/trajectory.json /logs/agent/trajectory.json 2>/dev/null || true
elif [ -f "/logs/agent/trajectory.json" ] && [ ! -f "/logs/trajectory.json" ]; then
  cp /logs/agent/trajectory.json /logs/trajectory.json 2>/dev/null || true
fi

# Use native rewardkit if pre-installed in the container; fall back to uvx if missing
if command -v rewardkit &> /dev/null; then
  REWARDKIT_CMD="rewardkit"
else
  if ! command -v uvx &> /dev/null; then
    echo "[*] Installing uv for verifier..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh > /dev/null 2>&1 || true
    export PATH="/usr/local/bin:$HOME/.cargo/bin:$PATH"
  fi
  REWARDKIT_CMD="uvx --with google-auth --from harbor-rewardkit>=0.2.2 rewardkit"
fi

export LITELLM_NUM_RETRIES=8
export LITELLM_RETRY_AFTER=3

MAX_RETRIES=5
RETRY_DELAY=3

for ((i=1; i<=MAX_RETRIES; i++)); do
  $REWARDKIT_CMD /tests --workspace /workspace && exit 0
  EXIT_CODE=$?
  if [ "$i" -lt "$MAX_RETRIES" ]; then
    echo "[test.sh] Verifier transient error ($EXIT_CODE). Retrying in ${RETRY_DELAY}s ($i/$MAX_RETRIES)..."
    sleep "$RETRY_DELAY"
    RETRY_DELAY=$((RETRY_DELAY * 2))
  fi
done

exit "$EXIT_CODE"
