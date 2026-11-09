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

# This file is the host-side image that eval agents run in. It contains
# everything in package/requirements-client.txt plus the tools the
# harnesses, hooks, and verifiers call (curl, git, jq, unzip, so on).
# Task environment/Dockerfiles start FROM this image, pinned by digest, and add
# only what the specific task needs.
#
# After a rebuild, re-pin every task to the new digest (`git grep nexus-host@sha256`).
#
# This image is built in CI and pushed to:
#   us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/nexus-host:latest
# To build it yourself, run from the repository root:
#   docker build -f eval/images/nexus-host/Dockerfile -t nexus-host package

ARG BASE_IMAGE=python:3.12-slim
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
# /etc/nexus-host/constraints.txt pins each requirements-client.txt package to
# the version installed here. Task Dockerfiles pass it to `pip install -c` so
# their own installs cannot change those versions.
RUN pip install --no-cache-dir uv \
 && uv pip install --system --no-cache -r /tmp/requirements-client.txt \
      harbor-rewardkit google-auth cloud-diagnostics-xprof==0.1.1 \
 && mkdir -p /etc/nexus-host \
 && pip show $(sed -E 's/#.*//; s/[[:space:]<>=!~;\[].*//' /tmp/requirements-client.txt) \
      | awk '/^Name:/{n=$2} /^Version:/{print n"=="$2}' > /etc/nexus-host/constraints.txt \
 && rm /tmp/requirements-client.txt

WORKDIR /workspace
