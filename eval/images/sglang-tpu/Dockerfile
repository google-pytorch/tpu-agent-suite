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

# This file is an image that TPU workloads can run inside on the containerized
# backends (e.g., gke_job, gke_pod). It is based on the public SGLang-TorchTPU
# release image and adds package/requirements-tpu.txt.
#
# This image is built in CI and pushed to:
#   us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/sglang-tpu:latest
# To build it yourself, run from the repository root:
#   docker build -f eval/images/sglang-tpu/Dockerfile -t sglang-tpu package

ARG BASE_IMAGE=lmsysorg/sglang:v0.1.0-tpu
FROM ${BASE_IMAGE}

COPY requirements-tpu.txt /tmp/tpu-agent-suite/requirements-tpu.txt

# Packages already in the base image are passed as constraints, so pip only
# installs missing packages and fails instead of replacing one.
RUN python3 -m pip list --format=freeze > /tmp/tpu-agent-suite/base-constraints.txt \
 && python3 -m pip install --no-cache-dir uv \
 && uv pip install --system --no-cache \
      -c /tmp/tpu-agent-suite/base-constraints.txt \
      -r /tmp/tpu-agent-suite/requirements-tpu.txt
