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

# Prepares the task workspace and installs the TPU agent package.
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/e2e_env.sh"

if [ -e "$WORKSPACE_DIR" ]; then
  echo "Error: $WORKSPACE_DIR already exists. Remove it first." >&2
  exit 1
fi

mkdir -p "$DOWNLOAD_DIR" "$WORKSPACE_DIR/logs/artifacts"
git -C "$WORKSPACE_DIR" init
cp "$LOCAL_REPO_DIR/eval/datasets/tpu-inference-vllm/$GOLDEN_TASK/instruction.md" "$WORKSPACE_DIR/PROMPT.md"

gcloud storage cp --no-clobber "$GCS_ARCHIVE_DIR/$DIST_NAME.tar.gz" "$DOWNLOAD_DIR/"
tar -xzf "$DOWNLOAD_DIR/$DIST_NAME.tar.gz" -C "$DOWNLOAD_DIR"
pip install -r "$DOWNLOAD_DIR/$DIST_NAME/requirements-client.txt"
python3 "$DOWNLOAD_DIR/$DIST_NAME/framework/installer.py" install "$DOWNLOAD_DIR/$DIST_NAME" --destination "$WORKSPACE_DIR" --non-interactive --force
