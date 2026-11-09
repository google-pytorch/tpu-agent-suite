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
"""Evaluation utility functions."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def find_repo_root(start: Path | None = None) -> Path:
    """Identify repository root using relative hierarchy and project sentinels.

    Does not depend on Git or subprocess execution. Checks in order:
    1. REPO_ROOT environment variable override.
    2. Parent directory walk looking for project sentinels (`package/package.release.json` or `AGENTS.md`).
    3. Direct file-relative hierarchy (one level above `eval`).
    """
    env_root = os.environ.get("REPO_ROOT")
    if env_root:
        candidate = Path(env_root).resolve()
        if candidate.is_dir():
            return candidate

    cur = (start or Path(__file__)).resolve()
    # Walk parents checking for unambiguous project sentinels
    for parent in [cur] + list(cur.parents):
        if (parent / "package" / "package.release.json").is_file() or (
                parent / "AGENTS.md").is_file():
            return parent

    # Fallback to direct relative hierarchy from eval/
    eval_dir = Path(__file__).resolve().parent
    return eval_dir.parent


def is_control_arm(config_path: Path | str) -> bool:
    """Whether a config is a control arm, which installs nothing.

    Decided by name alone: a stem with a `baseline` part, split on `_` or `-`,
    as in `claude-code/baseline.yaml`. Every other config installs the whole
    package.
    """
    return "baseline" in re.split(r"[-_]", Path(config_path).stem.lower())


RELEASE_MANIFEST = "package.release.json"

# package.<label>.json: <label> is the package's short name in job names.
_MANIFEST_NAME = re.compile(r"package\.[A-Za-z0-9_-]+\.json")


def package_manifest(agent_env: dict[str, Any]) -> str:
    """Return the PACKAGE_MANIFEST value in `agent_env`, unchecked.

    `agent_env` is a config's agents[].env with its `${VAR:-default}`
    templates already resolved. The value is the file name of the manifest
    in `package/` an agent-suite arm installs, e.g. package.experimental.json;
    surrounding whitespace is stripped, and package.release.json is returned
    when the key is missing or empty. Whether that file exists is checked by
    `resolve_manifest`, not here.
    """
    value = str(agent_env.get("PACKAGE_MANIFEST") or "").strip()
    return value or RELEASE_MANIFEST


def resolve_manifest(manifest: str, repo_root: Path | None = None) -> str:
    """Return `manifest` unchanged if it names a manifest in `package/`.

    `manifest` must be a bare file name of the form package.<label>.json,
    and that file must exist in `repo_root`/package (`repo_root` defaults to
    this checkout's root). Anything else -- a typo, another JSON file in
    package/, a path reaching outside it -- raises ValueError listing the
    manifests that do exist, rather than falling back to release.
    """
    package = (repo_root or find_repo_root()) / "package"
    if _MANIFEST_NAME.fullmatch(manifest) and (package / manifest).is_file():
        return manifest
    available = sorted(p.name for p in package.glob("package.*.json"))
    raise ValueError(
        f"PACKAGE_MANIFEST={manifest!r} is not a manifest in package/; "
        f"available: {', '.join(available)}")


def package_label(manifest: str) -> str:
    """Return a manifest file name's label: package.experimental.json -> experimental.

    Job names carry it, so runs of different packages stay apart on dashboards.
    """
    return manifest.removeprefix("package.").removesuffix(".json")


def read_manifest(repo_root: Path | None = None,
                  manifest: str = RELEASE_MANIFEST) -> dict[str, Any]:
    """Return the contents of the manifest file `manifest` in `package/`, as a dict.

    When the manifest has an `extends` key naming a base manifest, the
    returned dict is the merge compiler.py builds: the base's fields,
    overridden by the manifest's own, then the manifest's `add_skills`
    appended to the resulting `skills`; the `extends` and `add_skills` keys
    are dropped. Unlike compiler.py it does not validate the manifests.
    `repo_root` defaults to this checkout's root.
    """
    package = (repo_root or find_repo_root()) / "package"
    data = json.loads((package / manifest).read_text(encoding="utf-8"))
    if "extends" not in data:
        return data
    base = json.loads((package / data["extends"]).read_text(encoding="utf-8"))
    merged = {
        **base,
        **{
            k: v
            for k, v in data.items() if k not in ("extends", "add_skills")
        }
    }
    merged["skills"] = list(merged.get("skills", [])) + list(
        data.get("add_skills", []))
    return merged


def package_identity(manifest: str = RELEASE_MANIFEST,
                     repo_root: Path | None = None) -> tuple[str, str]:
    """Return the (name, version) the manifest file `manifest` gives its package.

    They are its `name` and `version` fields, after `extends` is merged (see
    `read_manifest`), and they name the directory the installer stores the
    package in, `<name>@<version>`: ("tpu-agent-suite", "0.7.0") for
    package.release.json, ("tpu-agent-suite-experimental", "0.7.0") for
    package.experimental.json. Raises KeyError if either field is missing.
    """
    data = read_manifest(repo_root, manifest)
    return str(data["name"]), str(data["version"])


# The agents in agents/user_scope.py, named here so that choosing a config's
# staging does not import Harbor.
USER_SCOPE_AGENT_MODULE = "eval.agents.user_scope"
# Where those agents find what they install: the compiled distribution and the
# installer's answers, both mounted read-only by the mount advisor.
USER_SCOPE_DIST = "/opt/tpu-agent-suite/dist"
USER_SCOPE_ANSWERS = "/opt/tpu-agent-suite/answers.json"


def installs_at_user_scope(config: dict[str, Any]) -> bool:
    """Whether the config's first agent installs the package itself, at user
    scope, rather than being handed a staged project-scope install."""
    agents = config.get("agents") or []
    import_path = str(agents[0].get("import_path") or "") if agents else ""
    return import_path.split(":", 1)[0] == USER_SCOPE_AGENT_MODULE


def harness_for_config(config: dict[str, Any]) -> str:
    """The compiler harness named by the config's first agent."""
    agents = config.get("agents") or []
    # A custom agent has an import path in place of a name; its class name
    # still says which harness it wraps.
    name = (str(agents[0].get("name") or agents[0].get("import_path")
                or "").lower() if agents else "")
    for token, harness in (("claude", "claude-code"), ("codex", "codex"),
                           ("cursor", "cursor")):
        if token in name:
            return harness
    return "antigravity"


def released_skills(repo_root: Path | None = None,
                    manifest: str = RELEASE_MANIFEST) -> list[str]:
    """Return the names of the skills the manifest file `manifest` releases.

    These are what the whole package installs: the `skills` list after
    `extends` is merged (see `read_manifest`). `manifest` defaults to
    package.release.json; pass a config's own, from `resolve_manifest`.
    """
    return list(read_manifest(repo_root, manifest)["skills"])


def skill_subset(names: Iterable[str], released: Iterable[str]) -> list[str]:
    """The skills named, comma- or space-separated, checked against `released`.

    The subset mode: a debugging run that installs only some of the package.
    """
    chosen = sorted(
        {n.strip()
         for value in names
         for n in value.split(",") if n.strip()})
    known = set(released)
    unknown = [n for n in chosen if n not in known]
    if unknown or not chosen:
        raise ValueError(
            f"--skills: unknown or empty: {', '.join(unknown) or '(none)'}. "
            f"The package releases: {', '.join(sorted(known))}")
    return chosen


def read_harness_pins(path: Path) -> dict[str, str]:
    """The KEY=VALUE pairs of .ci/pins.conf, without comments or blank lines.

    eval/adapter/sitecustomize.py carries its own copy of this parser: it
    loads at interpreter start inside Harbor, where an import error would
    silently drop every patch it applies.
    """
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"{path}: expected KEY=VALUE, got {raw!r}")
        pins[key.strip()] = value.strip()
    return pins
