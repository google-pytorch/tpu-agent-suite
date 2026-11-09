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

export LITELLM_NUM_RETRIES=8
export LITELLM_RETRY_AFTER=3

MAX_RETRIES=5
RETRY_DELAY=3

for ((i=1; i<=MAX_RETRIES; i++)); do
  if rewardkit /tests --workspace /workspace; then
    exit 0
  fi
  EXIT_CODE=$?
  if [ "$i" -lt "$MAX_RETRIES" ]; then
    echo "[test.sh] Verifier transient error ($EXIT_CODE). Retrying in ${RETRY_DELAY}s ($i/$MAX_RETRIES)..."
    sleep "$RETRY_DELAY"
    RETRY_DELAY=$((RETRY_DELAY * 2))
  fi
done

exit "$EXIT_CODE"
