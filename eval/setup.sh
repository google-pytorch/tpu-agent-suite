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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment using uv..."
  uv venv --python 3.12 .venv
else
  echo "Virtual environment already exists."
fi

echo "Installing/verifying dependencies..."
uv pip install -e .

ACTIVATE_SCRIPT="$SCRIPT_DIR/.venv/bin/activate"
if [ -f "$ACTIVATE_SCRIPT" ] && ! grep -q "# TORCHTPU_EVAL_PYTHONPATH" "$ACTIVATE_SCRIPT"; then
  cat >> "$ACTIVATE_SCRIPT" <<EOF

# TORCHTPU_EVAL_PYTHONPATH
export PYTHONPATH="${REPO_ROOT}\${PYTHONPATH:+:\$PYTHONPATH}"
EOF
fi

echo "Checking GKE authentication plugins..."
if command -v gcloud &>/dev/null; then
  if ! command -v gke-gcloud-auth-plugin &>/dev/null; then
    echo "Installing gke-gcloud-auth-plugin via gcloud components..."
    gcloud components install gke-gcloud-auth-plugin kubectl --quiet || true
  fi
  sdk_root="$(gcloud info --format="value(installation.sdk_root)" 2>/dev/null || true)"
  if [ -n "$sdk_root" ]; then
    if [ -x "$sdk_root/bin/gke-gcloud-auth-plugin" ] && [ ! -e "$SCRIPT_DIR/.venv/bin/gke-gcloud-auth-plugin" ]; then
      ln -sf "$sdk_root/bin/gke-gcloud-auth-plugin" "$SCRIPT_DIR/.venv/bin/gke-gcloud-auth-plugin"
    fi
    if [ -x "$sdk_root/bin/kubectl" ] && [ ! -e "$SCRIPT_DIR/.venv/bin/kubectl" ]; then
      ln -sf "$sdk_root/bin/kubectl" "$SCRIPT_DIR/.venv/bin/kubectl"
    fi
  fi
fi

if ! command -v gke-gcloud-auth-plugin &>/dev/null && [ ! -x "$SCRIPT_DIR/.venv/bin/gke-gcloud-auth-plugin" ]; then
  echo "Warning: gke-gcloud-auth-plugin was not found."
  echo "  For GKE evaluations, install it via: gcloud components install gke-gcloud-auth-plugin"
fi

echo ""
echo "Setup complete. To activate the environment, run:"
echo "  source eval/.venv/bin/activate"
