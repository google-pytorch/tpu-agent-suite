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
# backends (e.g., gke_job, gke_pod). This image is based on the latest
# vLLM-TorchTPU prod and includes everything in package/requirements-tpu.txt.
#
# This image is built nightly in CI and pushed to:
#   us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/nexus-tpu:latest
# To build it yourself, run from the repository root:
#   docker build -f eval/images/nexus-tpu/Dockerfile -t nexus-tpu package
#
# The base image (vLLM-TorchTPU prod) is private; pulling it needs Artifact Registry
# read access for cloud-ullm-inference-ci-cd/vllm-torchtpu.

ARG BASE_IMAGE=us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-prod:latest
FROM ${BASE_IMAGE}

COPY requirements-tpu.txt /tmp/tpu-agent-suite/requirements-tpu.txt

# We first record packages already included in the vLLM-TorchTPU base image in
# base-constraints.txt, and we pass these as "constraints" to `pip install`, so
# that `pip install` would only install packages that are *missing* in the base
# image. If requirements-tpu.txt needs an incompatible version of something already
# installed (torch, jax, libtpu, ...), pip fails the build instead of replacing it.
RUN python3 -m pip list --format=freeze > /tmp/tpu-agent-suite/base-constraints.txt \
 && grep -i '^setuptools==' /tmp/tpu-agent-suite/base-constraints.txt > /tmp/tpu-agent-suite/overrides.txt \
 && (command -v uv || python3 -m pip install --no-cache-dir uv) \
 && uv pip install --system --no-cache \
      -c /tmp/tpu-agent-suite/base-constraints.txt \
      --override /tmp/tpu-agent-suite/overrides.txt \
      -r /tmp/tpu-agent-suite/requirements-tpu.txt
