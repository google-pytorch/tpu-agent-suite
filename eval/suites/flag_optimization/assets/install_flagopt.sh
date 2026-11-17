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

# Docker-build setup step (suite `setup_scripts`, run as root): install
# `flagopt` from the workspace staging dir into /opt/flagopt.
#
# suite_to_harbor copies `workspace_files` into the workdir before it runs the
# setup scripts, so the toolkit arrives as /workspace/.flagopt. It is moved
# out of the agent-writable workspace into a root-owned, read-only
# /opt/flagopt: the verifier imports flagopt.py and reads tasks.json from
# there, and the agent (uid 1000) cannot change either.
set -euo pipefail

src=/workspace/.flagopt
dst=/opt/flagopt
test -f "${src}/flagopt.py" && test -f "${src}/tasks.json"

rm -rf "${dst}"
mv "${src}" "${dst}"
rm -rf "${dst}/__pycache__"
chown -R root:root "${dst}"
find "${dst}" -type d -exec chmod 0755 {} +
find "${dst}" -type f -exec chmod 0644 {} +

# `-I -S`: nothing the agent installs can change what flagopt imports.
printf '#!/bin/sh\nexec python3 -I -S /opt/flagopt/flagopt.py "$@"\n' \
  >/usr/local/bin/flagopt
chmod 0755 /usr/local/bin/flagopt

# The MaxText install of the base image, where the instructions point.
ln -sfn /deps /opt/maxtext
mkdir -p /workspace/output /workspace/runs
