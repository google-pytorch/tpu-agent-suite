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

# Baseline vLLM server for google/gemma-4-31B-it: the serving recipe the
# onboarding settled on (TP 2 x DP 1, no sharding search, max-model-len 10240,
# prefix caching off, text-only via --language-model-only, CUSTOM attention
# backend). It ran on one tpu7x chip (2 devices).
#
# The original also pointed the weight and compile caches at the onboarding
# host's scratch disk and moved the torch_tpu tier-2 cache off /dev/shm; both
# were workarounds for that host and are left out so this runs in a compute
# job. Submit it with the package's fetch_weights.py and serve_vllm.sh
# (scripts/common/) as inputs; it calls both from its working directory.
set -eo pipefail
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export VLLM_SERVER_DEV_MODE=1
PY="${TPU_AGENT_SUITE_PYTHON:-python3}"
PY="$(command -v "${PY/#\~/$HOME}")" \
  || { echo "no Python at ${TPU_AGENT_SUITE_PYTHON:-python3} on the target" >&2; exit 1; }
"$PY" fetch_weights.py "google/gemma-4-31B-it" --path-file weights_path.txt || exit 1
export HF_HUB_OFFLINE=1
exec bash serve_vllm.sh "$(cat weights_path.txt)" \
  --served-model-name "google/gemma-4-31B-it" \
  --port 8000 \
  --tensor-parallel-size 2 \
  --data-parallel-size 1 \
  --max-model-len 10240 \
  --no-enable-prefix-caching \
  --language-model-only \
  --host 0.0.0.0 \
  --attention-backend CUSTOM
