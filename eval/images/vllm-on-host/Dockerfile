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

# This file is the host-side image for eval tasks whose agent needs vLLM,
# torch_tpu and the rest of vLLM-TorchTPU in its own container (e.g., the
# onboarding tasks). It installs the same host-side dependencies as
# eval/images/nexus-host/Dockerfile (package/requirements-client.txt plus curl,
# git, jq, unzip, gcloud, kubectl, so on), but on top of the latest
# vLLM-TorchTPU prod image instead of python:3.12-slim, plus tokamax.
#
# Task environment/Dockerfiles start FROM this image, pinned by digest, and add
# only what the specific task needs. After a rebuild, re-pin every task to the
# new digest (`git grep vllm-on-host@sha256`).
#
# This image is built in CI and pushed to:
#   us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/vllm-on-host:latest
# To build it yourself, run from the repository root:
#   docker build -f eval/images/vllm-on-host/Dockerfile -t vllm-on-host package
#
# The base image (vLLM-TorchTPU prod) is private; pulling it needs Artifact
# Registry read access for cloud-ullm-inference-ci-cd/vllm-torchtpu.

ARG BASE_IMAGE=us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-prod:latest
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git gnupg jq nodejs npm unzip \
 && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      > /etc/apt/sources.list.d/google-cloud-sdk.list \
 && apt-get update && apt-get install -y --no-install-recommends \
      google-cloud-cli google-cloud-cli-gke-gcloud-auth-plugin \
 && npm install -g yarn@1.22.22 \
 && rm -rf /var/lib/apt/lists/*

ARG KUBECTL_VERSION=v1.31.4
RUN curl -fsSLo /usr/local/bin/kubectl \
      "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
 && chmod +x /usr/local/bin/kubectl

COPY requirements-client.txt /tmp/requirements-client.txt
# We first record packages already included in the vLLM-TorchTPU base image in
# base-constraints.txt, and we pass these as "constraints" to `pip install`, so
# that `pip install` would only install packages that are *missing* in the base
# image. If requirements-client.txt needs an incompatible version of something
# already installed (torch, jax, vllm, ...), pip fails the build instead of
# replacing it. The extra packages after requirements-client.txt match the ones
# eval/images/nexus-host/Dockerfile installs.
#
# Exception: the base image ships python-dateutil 3.0.0, but every usable
# google-cloud-bigquery release (requirements-client.txt) needs
# python-dateutil<3. We drop the dateutil pin from base-constraints.txt and
# override it to <3, which downgrades it in this image.
#
# /etc/vllm-on-host/constraints.txt pins each requirements-client.txt
# package to the version installed here, like /etc/nexus-host/constraints.txt.
RUN python3 -m pip list --format=freeze | grep -iv '^python-dateutil==' > /tmp/base-constraints.txt \
 && grep -i '^setuptools==' /tmp/base-constraints.txt > /tmp/overrides.txt \
 && echo 'python-dateutil<3' >> /tmp/overrides.txt \
 && (command -v uv || python3 -m pip install --no-cache-dir uv) \
 && uv pip install --system --no-cache \
      -c /tmp/base-constraints.txt \
      --override /tmp/overrides.txt \
      -r /tmp/requirements-client.txt \
      harbor-rewardkit google-auth cloud-diagnostics-xprof==0.1.1 \
 && mkdir -p /etc/vllm-on-host \
 && pip show $(sed -E 's/#.*//; s/[[:space:]<>=!~;\[].*//' /tmp/requirements-client.txt) \
      | awk '/^Name:/{n=$2} /^Version:/{print n"=="$2}' > /etc/vllm-on-host/constraints.txt \
 && rm /tmp/requirements-client.txt /tmp/base-constraints.txt /tmp/overrides.txt

RUN pip install --no-cache-dir tokamax "flax>=0.12.10"

WORKDIR /workspace
