#!/usr/bin/env bash
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

# Environment variables for manual TPU agent E2E testing.
export WORK_ROOT="/tmp/tpu-agent-e2e"
export DOWNLOAD_DIR="$WORK_ROOT/downloads"
export LOCAL_REPO_DIR="$HOME/Workspace/github/google-pytorch/tpu-agent-suite"

export HARNESS="antigravity"
export GOLDEN_TASK="tpu-inference-gemma-4-31b-it-vllm"
export WORKSPACE_DIR="$WORK_ROOT/$HARNESS/$GOLDEN_TASK"

export GCS_ARCHIVE_DIR="gs://gh_zips/releases/nightly/latest"
# shellcheck disable=SC2155
export DIST_NAME=$(basename "$(gcloud storage ls "$GCS_ARCHIVE_DIR/*-$HARNESS-*.tar.gz")" .tar.gz)

export COMPUTE_BACKEND="gke_job"
export GKE_PROJECT="cloud-tpu-shared-capacity"
export GKE_LOCATION="us-central1"
export GKE_CLUSTER="bodaborg-tpu7x-nap"
export GKE_NAMESPACE="default"
export GKE_QUEUE="default"
export GKE_PRIORITY="low"
# time-limit-controller stops a pod at min(namespace cap,
# declared-duration-minutes * 1.5). The cap on `default` is 180, so declaring
# 180 makes the limit agents read from `compute inspect` the one enforced.
export GKE_DECLARED_DURATION_MINUTES="180"
export GKE_TPU_ACCELERATOR="tpu7x"
export GKE_TPU_TOPOLOGY="2x2x1"
export TASK_DOCKER="us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/nexus-tpu:latest"
