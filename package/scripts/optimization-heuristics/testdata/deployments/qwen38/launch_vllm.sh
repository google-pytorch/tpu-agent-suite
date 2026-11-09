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

set -eo pipefail
exec bash serve_vllm.sh /model \
  --port 8000 \
  --tensor-parallel-size 4 \
  --data-parallel-size 2 \
  --max-model-len 10240 \
  --no-enable-prefix-caching \
  --dtype bfloat16 \
  --language-model-only \
  --attention-backend CUSTOM \
  --host 0.0.0.0
