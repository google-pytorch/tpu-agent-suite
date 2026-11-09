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

FROM --platform=linux/amd64 us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/nexus-host@sha256:7ff58a98ae2655dbcf25fa0aeb3e8081281bfb3061f97581a98951e98371fff4

RUN apt-get update && apt-get install -y --no-install-recommends openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
# No accelerate: it pulls in torch and its CUDA wheels (~3 GB), and torch
# only runs in the TPU job (TASK_DOCKER), never in this container.
RUN pip install --no-cache-dir -c /etc/nexus-host/constraints.txt transformers absl-py pytest

# Pre-stage a local checkpoint directory from Qwen/Qwen3-0.6B
# under a non-Hub path and strip _name_or_path / HF cache metadata so the agent
# must stage the local directory to GCS rather than passing a Hub ID.
RUN python3 -c "import json, pathlib; from huggingface_hub import snapshot_download; p = pathlib.Path(snapshot_download('Qwen/Qwen3-0.6B', local_dir='/workspace/checkpoints/local-qwen3-0.6b')); cfg = p / 'config.json'; data = json.loads(cfg.read_text()); data.pop('_name_or_path', None); cfg.write_text(json.dumps(data, indent=2)); (p / 'README.md').unlink(missing_ok=True)" \
    && rm -rf /workspace/checkpoints/local-qwen3-0.6b/.cache /root/.cache/huggingface

WORKDIR /workspace

RUN chmod -R 777 /workspace
