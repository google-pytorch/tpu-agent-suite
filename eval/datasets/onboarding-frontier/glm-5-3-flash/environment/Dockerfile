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

# The agent runs in the vLLM + torch_tpu nightly image its TPU jobs use
# (nexus-tpu:latest), with the dev tooling laid over it.
FROM --platform=linux/amd64 us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/nexus-tpu:latest

# Node 22 from NodeSource: Claude Code needs >= 22 and bookworm ships 18.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git gnupg jq procps sudo unzip \
 && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      > /etc/apt/sources.list.d/google-cloud-sdk.list \
 && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
      | gpg --dearmor -o /usr/share/keyrings/nodesource.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
      > /etc/apt/sources.list.d/nodesource.list \
 && apt-get update && apt-get install -y --no-install-recommends \
      google-cloud-cli google-cloud-cli-gke-gcloud-auth-plugin nodejs \
 && npm install -g yarn@1.22.22 @anthropic-ai/claude-code@2.1.304 \
 && rm -rf /var/lib/apt/lists/*

ARG KUBECTL_VERSION=v1.31.4
RUN curl -fsSLo /usr/local/bin/kubectl \
      "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
 && chmod +x /usr/local/bin/kubectl

# agy: the release, checksum and binary patch of ANTIGRAVITY_CLI_PIN in
# eval/adapter/sitecustomize.py. Trials still run Harbor's install at
# $HOME/.local/bin/agy.
ARG AGY_VERSION=1.4.0
ARG AGY_URL=https://storage.googleapis.com/antigravity-public/antigravity-cli/1.4.0-5281842794987520/linux-x64/cli_linux_x64.tar.gz
ARG AGY_SHA512=6c4b298c2a69b6bd0f58567031cc56d8097a32be54c4b633aa80ad395871b389618ab3860d4ba0a12599831b8226b51b03e258ee2ec7fae105a45b8e0fce2188
ARG AGY_PATCH_BEFORE=84c07417488b54247848c7821802000001000000c6822101000001
ARG AGY_PATCH_AFTER=488b542478c682210100000184c0740b48c7821802000001000000
RUN d=$(mktemp -d) \
 && curl -fsSL -o "$d/cli.tar.gz" "$AGY_URL" \
 && echo "$AGY_SHA512  $d/cli.tar.gz" | sha512sum -c - \
 && tar -xzf "$d/cli.tar.gz" -C "$d" antigravity \
 && python3 -I -c 'import pathlib, sys; p = pathlib.Path(sys.argv[1]); b, a = (bytes.fromhex(x) for x in sys.argv[2:]); d = p.read_bytes(); assert d.count(b) == 1; p.write_bytes(d.replace(b, a))' \
      "$d/antigravity" "$AGY_PATCH_BEFORE" "$AGY_PATCH_AFTER" \
 && install -m 0755 "$d/antigravity" /usr/local/bin/agy \
 && [ "$(AGY_CLI_DISABLE_AUTO_UPDATE=true agy --version)" = "$AGY_VERSION" ] \
 && rm -rf "$d"

# The package/requirements-client.txt packages nexus-tpu lacks: the suite's
# session-start check stops the agent until python3 has all of them. Pinned to
# the base's versions so nothing torch or vLLM imports changes, except
# python-dateutil, which google-cloud-bigquery needs below 3.
RUN python3 -m pip list --format=freeze \
      | grep -viE '^python-dateutil==' > /tmp/base-constraints.txt \
 && uv pip install --system --no-cache -c /tmp/base-constraints.txt \
      'google-cloud-aiplatform>=1.40.0' 'google-cloud-bigquery>=3.20.0' \
      'vertexai>=1.0.0' \
 && rm /tmp/base-constraints.txt

# The base has no rewardkit; the verifier gets it in a venv of its own that
# does not see the system site-packages. test.sh finds it on PATH.
RUN python3 -m venv /opt/verifier \
 && VIRTUAL_ENV=/opt/verifier uv pip install --no-cache harbor-rewardkit google-auth \
 && ln -s /opt/verifier/bin/rewardkit /usr/local/bin/rewardkit

# The agent (uid 1000) writes here. /workspace/vllm-torchtpu is the base's
# source checkout; the vllm_torchtpu that python3 imports is a copy in the
# root-owned site-packages, not an editable install.
WORKDIR /workspace
RUN chmod 777 /workspace
