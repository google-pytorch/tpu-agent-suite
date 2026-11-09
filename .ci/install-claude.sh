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
#
# Installs the Claude Code CLI pinned in pins.conf beside this file, from npm.
set -euo pipefail

# shellcheck disable=SC1091
. "$(dirname "${BASH_SOURCE[0]}")/pins.conf"
: "${CLAUDE_CODE_VERSION:?set in .ci/pins.conf}"

npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"
echo "Installed pinned Claude Code $(claude --version)"
