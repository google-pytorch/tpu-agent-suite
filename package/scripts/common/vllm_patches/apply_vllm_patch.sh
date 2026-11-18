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
# Apply a vllm_torchtpu patch to the plugin installed on the target, in the
# container that is about to use it. The task image is never rebuilt, so every
# job that imports the plugin applies the patch again; this is idempotent.
#
#   bash apply_vllm_patch.sh <patch> [<plugin source root>]
#
# The source root is the directory holding `vllm_torchtpu/`. Without one, it is
# found by importing the plugin with $TPU_AGENT_SUITE_PYTHON (else python3), so an
# editable checkout and a site-packages install are patched alike.
#
# A patch is one of two kinds, told apart by its extension:
#
# - `.patch`, a fixed unified diff. Already applied is detected by the patch
#   itself reversing cleanly, not by a marker string, so an image that ships an
#   upstream fix with different code is reported as drift rather than silently
#   skipped or double-patched. It applies to the revision it was cut from only.
# - `.py`, a patch script, run with the same Python as `<script> <source root>`.
#   It finds its own anchors, so it can follow the plugin across revisions; it
#   must decide every change before writing any, keep these exit codes, and
#   print the same `vllm patch applied` / `already applied` line.
#
# Exit codes: 0 applied or already applied; 2 usage; 3 plugin not found;
# 4 the patch does not apply to this plugin revision.
set -euo pipefail

PATCH="${1:-}"
if [ -z "$PATCH" ] || [ ! -f "$PATCH" ]; then
  echo "usage: apply_vllm_patch.sh <patch> [<plugin source root>]" >&2
  exit 2
fi
PATCH="$(cd "$(dirname "$PATCH")" && pwd)/$(basename "$PATCH")"

# Resolved, since the default interpreter is a bare `python3`, and with a
# leading ~ expanded, since Kubernetes passes env values through as is. A
# shell is not a Python: a runtime that took `--interpreter bash` for one
# exported it here.
PY="${TPU_AGENT_SUITE_PYTHON:-python3}"
case "${PY##*/}" in bash | sh | dash | zsh | ksh) PY=python3 ;; esac
PY="$(command -v "${PY/#\~/$HOME}" || true)"

SRC="${2:-}"
if [ -z "$SRC" ]; then
  # find_spec locates the plugin without running it: importing vllm_torchtpu
  # starts the TPU runtime and prints banners to stdout, which would land in
  # SRC. The tag keeps anything a sitecustomize prints out of it all the same.
  SRC="$("$PY" -c 'import importlib.util, os; s = importlib.util.find_spec("vllm_torchtpu"); print("VLLM_TORCHTPU_ROOT=" + os.path.dirname(os.path.abspath(list(s.submodule_search_locations)[0])))' 2>/dev/null \
    | sed -n 's/^VLLM_TORCHTPU_ROOT=//p' | tail -n 1 || true)"
  if [ -z "$SRC" ]; then
    echo "vllm_torchtpu is not importable with $PY; cannot patch it" >&2
    exit 3
  fi
fi
if [ ! -d "$SRC/vllm_torchtpu" ]; then
  echo "no vllm_torchtpu/ under $SRC" >&2
  exit 3
fi

name="$(basename "$PATCH")"
# What serve_vllm.sh reads to give a patched build compile caches of its own:
# the patch changes compiled programs but not the image they are keyed on.
RECORD="$SRC/vllm_torchtpu/.tpu_agent_suite_patches"
record() {
  local line
  line="$(sha256sum "$PATCH" | cut -d' ' -f1) $name"
  grep -qxF "$line" "$RECORD" 2>/dev/null || echo "$line" >> "$RECORD"
}
if [ "${name##*.}" = "py" ]; then
  if [ -z "$PY" ]; then
    echo "no Python at ${TPU_AGENT_SUITE_PYTHON:-python3} to run $name" >&2
    exit 3
  fi
  # Not exec'd: a patch script that applied, or found itself applied, is
  # recorded like a diff so its build gets compile caches of its own too.
  rc=0
  "$PY" "$PATCH" "$SRC" || rc=$?
  if [ "$rc" -eq 0 ]; then
    record
  fi
  exit "$rc"
fi
if command -v patch >/dev/null; then
  reverses() { patch -p1 -R -F0 --dry-run --silent -f -d "$SRC" < "$PATCH" >/dev/null 2>&1; }
  applies() { patch -p1 -F0 --dry-run --silent -f -d "$SRC" < "$PATCH" >/dev/null 2>&1; }
  apply() { patch -p1 -F0 --forward -d "$SRC" < "$PATCH"; }
else
  reverses() { git -C "$SRC" apply -p1 -R --check "$PATCH" >/dev/null 2>&1; }
  applies() { git -C "$SRC" apply -p1 --check "$PATCH" >/dev/null 2>&1; }
  apply() { git -C "$SRC" apply -p1 --verbose "$PATCH"; }
fi

if reverses; then
  record
  echo "vllm patch already applied: $name ($SRC)"
  exit 0
fi
if ! applies; then
  echo "vllm patch does not apply: $name to $SRC." >&2
  echo "The plugin revision differs from the one the patch was validated on." >&2
  echo "Check whether this image already fixes the problem before re-basing it." >&2
  exit 4
fi
apply
record
echo "vllm patch applied: $name ($SRC)"
