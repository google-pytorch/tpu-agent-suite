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
# Installs the pinned Antigravity CLI as $HOME/.local/bin/agy.
#
# The pin comes from pins.conf beside this file. When the script text runs on
# its own, as it does when eval/adapter/sitecustomize.py sends it into a
# Harbor trial container with pins.conf prepended, the AGY_* variables are
# already set and there is no file to source.
#
# It does what the vendor installer does -- download, verify, extract to
# ~/.local/bin, then the CLI's own `install` step -- for one fixed release,
# applies the byte patch from pins.conf, and fails unless the installed binary
# reports the pinned version.
set -euo pipefail

if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "$(dirname "${BASH_SOURCE[0]}")/pins.conf" ]; then
  # shellcheck disable=SC1091
  . "$(dirname "${BASH_SOURCE[0]}")/pins.conf"
fi
: "${AGY_VERSION:?set in .ci/pins.conf}"
: "${AGY_URL:?set in .ci/pins.conf}"
: "${AGY_SHA512:?set in .ci/pins.conf}"
: "${AGY_PATCH_BEFORE=}"
: "${AGY_PATCH_AFTER=}"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64|Linux-amd64) ;;
  *) echo "Antigravity CLI $AGY_VERSION is pinned for linux-x64 only," \
       "not $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac
export AGY_CLI_DISABLE_AUTO_UPDATE=true
staging=$(mktemp -d)
trap 'rm -rf "$staging"' EXIT
curl -fsSL -o "$staging/cli.tar.gz" "$AGY_URL"
echo "$AGY_SHA512  $staging/cli.tar.gz" | sha512sum -c - >/dev/null
tar -xzf "$staging/cli.tar.gz" -C "$staging" antigravity
if [ -n "$AGY_PATCH_BEFORE" ] || [ -n "$AGY_PATCH_AFTER" ]; then
  AGY_PATCH_BEFORE="$AGY_PATCH_BEFORE" AGY_PATCH_AFTER="$AGY_PATCH_AFTER" \
    python3 - "$staging/antigravity" <<'PY'
import os, pathlib, sys
p = pathlib.Path(sys.argv[1])
data = p.read_bytes()
before = bytes.fromhex(os.environ["AGY_PATCH_BEFORE"])
after = bytes.fromhex(os.environ["AGY_PATCH_AFTER"])
if not before or len(before) != len(after):
    sys.exit("Antigravity CLI patch length mismatch")
count = data.count(before)
if count != 1:
    sys.exit(f"Expected 1 occurrence of AGY_PATCH_BEFORE in the Antigravity "
             f"CLI binary, found {count}")
p.write_bytes(data.replace(before, after, 1))
PY
fi
mkdir -p "$HOME/.local/bin"
install -m 0755 "$staging/antigravity" "$HOME/.local/bin/agy"
"$HOME/.local/bin/agy" install || true
installed=$("$HOME/.local/bin/agy" --version)
if [ "$installed" != "$AGY_VERSION" ]; then
  echo "Antigravity CLI reports $installed, expected $AGY_VERSION" >&2
  exit 1
fi
echo "Installed pinned Antigravity CLI $AGY_VERSION"
