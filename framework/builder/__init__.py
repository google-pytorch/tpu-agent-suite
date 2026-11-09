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
"""Builds installable packages from package/. `compiler.py` is the CLI.

The build reads the package source, checks it, and writes one ready-to-install
folder per harness (Claude Code, Codex, Antigravity, Cursor). It only runs in
the repo and CI and is not shipped.

- `package_loader`: loads the package into one manifest; `load_package` is
  the entry point.
- `package_discovery`: starting from the skills, finds every agent,
  reference and script the package uses.
- `package_checks`: rejects broken packages while they load (agent call
  loops, unknown tools, bad hooks, references to missing files).
- `tokens`: replaces placeholders like `{{agent:name}}` with real names and
  install paths.
- `prompt_contracts`: writes the standard instructions added to every agent
  and skill (running TPU jobs, the run journal, calling other agents, the
  result format).
- `harness_rules`: what differs per harness: tool names, how one agent calls
  another, agent file headers, Codex sandbox settings.
- `hook_config`: writes each harness's hook settings.
- `translators`: writes the agent and skill files for one harness.
- `dist_assets`: copies everything else into the output: runtime code,
  scripts, docs, requirements files and the install plan.
- `verify`: the checks behind `compiler.py verify`.

The names below are the public API. Use them instead of reaching into the
modules above, which can change; tests of a module's internals import it
directly.
"""

from __future__ import annotations

from .overlays import stage_package_with_overlays
from .package_loader import (DEFAULT_MANIFEST, load_package,
                             resolve_manifest_composition)
from .translators import HARNESS_TRANSLATORS, translate
from .verify import verify_package

__all__ = [
    "DEFAULT_MANIFEST",
    "HARNESS_TRANSLATORS",
    "load_package",
    "resolve_manifest_composition",
    "stage_package_with_overlays",
    "translate",
    "verify_package",
]
