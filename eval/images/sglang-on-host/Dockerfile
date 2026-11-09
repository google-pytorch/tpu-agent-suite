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

# This file is the host-side image for eval tasks whose agent needs SGLang,
# torch_tpu and the rest of SGLang-TorchTPU in its own container. It installs
# the same host-side dependencies as eval/images/nexus-host/Dockerfile
# (package/requirements-client.txt plus curl, git, jq, unzip, gcloud, kubectl,
# so on), plus tokamax and flax, on top of the public SGLang-TorchTPU release
# image that eval/images/sglang-tpu/Dockerfile also uses.
#
# Task environment/Dockerfiles start FROM this image, pinned by digest, and add
# only what the specific task needs. After a rebuild, re-pin every task to the
# new digest (`git grep sglang-on-host@sha256`).
#
# This image is built in CI and pushed to:
#   us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/sglang-on-host:latest
# To build it yourself, run from the repository root:
#   docker build -f eval/images/sglang-on-host/Dockerfile -t sglang-on-host package

ARG BASE_IMAGE=lmsysorg/sglang:v0.1.0-tpu
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
# Packages already in the base image are passed as constraints, so pip only
# installs missing packages and fails instead of replacing one (torch, jax,
# sglang, ...). The extra packages after requirements-client.txt match the
# ones eval/images/nexus-host/Dockerfile installs, plus tokamax and flax.
#
# Exception: tenacity is left out of the constraints. With the base image's
# protobuf 7.36.2 and google-cloud-storage 3.17.1, the only
# google-cloud-aiplatform releases (requirements-client.txt) that resolve
# need google-genai>=1.66.0, which needs tenacity<9.2.0. The base image has
# tenacity 9.2.1, so it is downgraded to 9.1.x.
#
# Exception: openai is constrained to <3 instead of the base image's 2.6.1.
# harbor-rewardkit needs litellm>=1.83, and every litellm release that
# resolves here needs openai>=2.8.0. SGLang pins openai==2.6.1 exactly, so
# `pip check` reports that one conflict
# (the CI image build ignores that one check). The <3 bound comes
# from sgl-eval, also in the base image, which needs openai>=1.0,<3.
#
# The base image has jax 0.10.2, so the constraints select the newest tokamax
# and flax releases that accept it: tokamax 0.0.15 and flax 0.12.8.
#
# /etc/sglang-on-host/constraints.txt pins each requirements-client.txt
# package to the version installed here, like /etc/nexus-host/constraints.txt.
RUN { python3 -m pip list --format=freeze | grep -ivE '^(tenacity|openai)=='; echo 'openai<3'; } \
      > /tmp/base-constraints.txt \
 && python3 -m pip install --no-cache-dir uv \
 && uv pip install --system --no-cache \
      -c /tmp/base-constraints.txt \
      -r /tmp/requirements-client.txt \
      harbor-rewardkit google-auth cloud-diagnostics-xprof==0.1.1 \
      tokamax flax \
 && mkdir -p /etc/sglang-on-host \
 && pip show $(sed -E 's/#.*//; s/[[:space:]<>=!~;\[].*//' /tmp/requirements-client.txt) \
      | awk '/^Name:/{n=$2} /^Version:/{print n"=="$2}' > /etc/sglang-on-host/constraints.txt \
 && rm /tmp/requirements-client.txt /tmp/base-constraints.txt

WORKDIR /workspace
