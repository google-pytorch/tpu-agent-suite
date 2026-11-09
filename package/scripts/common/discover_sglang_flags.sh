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
#
# Capture `python3 -m sglang.launch_server --help` from the SGLang installed on
# the target, so the launcher is generated against the flags that build
# actually accepts rather than against remembered ones.
#
# Runs on the accelerator host, dispatched with `compute run`.
#
# Usage:
#   compute run --code .../discover_sglang_flags.sh \
#     --output sglang_flags.txt:<local> -- sglang_flags.txt

set -uo pipefail

OUT="${1:-sglang_flags.txt}"

PY="${TPU_AGENT_SUITE_PYTHON:-python3}"
PY="$(command -v "${PY/#\~/$HOME}" || true)"

if [ -z "$PY" ]; then
  echo "python3 was not found on the target (looked for ${TPU_AGENT_SUITE_PYTHON:-python3})." >&2
  echo "python3 not found" > "$OUT"
  exit 1
fi

if ! "$PY" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('sglang') else 1)" >/dev/null 2>&1; then
  echo "sglang was not found on the target with $PY." >&2
  echo "sglang not found" > "$OUT"
  exit 1
fi

echo "# python: $PY -m sglang.launch_server" > "$OUT"
"$PY" -m sglang.launch_server --help >> "$OUT" 2>&1
STATUS=$?

cat "$OUT"
if [ "$STATUS" -ne 0 ]; then
  echo "\`$PY -m sglang.launch_server --help\` exited $STATUS; the output above is what it printed." >&2
fi
exit "$STATUS"
