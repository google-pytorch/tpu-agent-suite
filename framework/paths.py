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
"""Locating the enclosing git repository, the ~/.tpu-agent-suite store, and the
packages installed in it.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

from framework import jsonio, utils

Path = pathlib.Path

# Plan-relative prefix for central-store payloads (~/.tpu-agent-suite/packages/).
# Shared by the compiler's install plan and the installer's target routing.
TPU_AGENT_SUITE_DIRNAME = ".tpu-agent-suite"
PACKAGES_DIRNAME = "packages"
RUNS_DIRNAME = "runs"
SESSIONS_DIRNAME = "sessions"
STUDIES_DIRNAME = "studies"
CENTRAL_PACKAGES_PREFIX = f"{TPU_AGENT_SUITE_DIRNAME}/{PACKAGES_DIRNAME}"
# Per-install config files in .tpu-agent-suite/ (at most one package per
# workspace or user scope).
ENVIRONMENT_FILE = "environment.json"
OWNERSHIP_FILE = "ownership.json"
# Each profile's last `compute setup verify` result, beside environment.json.
VERIFICATION_FILE = "verification.json"

# Home-anchored paths for compiled prompts ("~/" expands on the target machine).
# Runtime code uses get_tpu_agent_suite_home() instead so $TPU_AGENT_SUITE_HOME works.
TPU_AGENT_SUITE_HOME_SPELLING = f"~/{TPU_AGENT_SUITE_DIRNAME}"
RUNS_HOME_SPELLING = f"{TPU_AGENT_SUITE_HOME_SPELLING}/{RUNS_DIRNAME}"

# The workspace file naming its active run (see record_active_run).
ACTIVE_RUN_FILE = "current_run_id"


def find_git_root(start: Path) -> Path | None:
    """Nearest ancestor of `start` (inclusive) containing `.git`, or None."""
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def default_tpu_agent_suite_home() -> Path:
    """Default central store path (~/.tpu-agent-suite)."""
    return (Path.home() / TPU_AGENT_SUITE_DIRNAME).resolve()


def get_tpu_agent_suite_home() -> Path:
    """Return the central storage root ($TPU_AGENT_SUITE_HOME or ~/.tpu-agent-suite).

    $TPU_AGENT_SUITE_HOME is for test isolation. Compiled hooks and prompts
    hardcode ~/.tpu-agent-suite, so real installs reject custom values unless
    --relocated-store is passed.
    """
    env_home = os.environ.get("TPU_AGENT_SUITE_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser().resolve()
    return default_tpu_agent_suite_home()


def get_package_storage_root(package: str = "", version: str = "") -> Path:
    """Return the central storage path for a package (~/.tpu-agent-suite/packages/<pkg>@<ver>/)."""
    tpu_agent_suite_home = get_tpu_agent_suite_home()
    if not package:
        return tpu_agent_suite_home
    utils.validate_package_name(package)
    pkg_tag = utils.format_package_tag(package, version)
    return tpu_agent_suite_home / PACKAGES_DIRNAME / pkg_tag


def tpu_agent_suite_dir_for(workspace: "Path | str") -> Path:
    """Return the .tpu-agent-suite/ config directory for a workspace or $HOME."""
    ws = Path(workspace).expanduser().resolve()
    if ws == Path.home().resolve():
        return get_tpu_agent_suite_home()
    return ws / TPU_AGENT_SUITE_DIRNAME


def resolve_environment(
        package: str,
        workspace: "Path | str | None" = None) -> "tuple[dict, Path | None]":
    """Resolve environment.json for `package`, checking `workspace` then user scope.

    Returns (config_dict, path), or ({}, None) if the package isn't installed
    in either scope. Raises TpuAgentSuiteError if ownership.json claims the
    package is installed but environment.json is missing.
    """
    if not package:
        raise utils.TpuAgentSuiteError(
            "a package is required to resolve environment config")
    base, _ = utils.parse_package_tag(package)

    levels = [tpu_agent_suite_dir_for(workspace)
              ] if workspace is not None else []
    levels.append(get_tpu_agent_suite_home())
    for tpu_agent_suite_dir in dict.fromkeys(levels):
        env_file = tpu_agent_suite_dir / ENVIRONMENT_FILE
        ledger_file = tpu_agent_suite_dir / OWNERSHIP_FILE
        owner = jsonio.load_json_or_empty(
            ledger_file) if ledger_file.is_file() else None
        if owner is not None and owner.get("package") != base:
            continue
        if env_file.is_file():
            return json.loads(env_file.read_text(encoding="utf-8")), env_file
        if owner is not None:
            raise utils.TpuAgentSuiteError(
                f"{base} is installed ({ledger_file}) but"
                f" {env_file} is missing -- re-run the installer to"
                " restore it, then set compute up again")

    return {}, None


def get_runs_root() -> Path:
    """Return the central runs directory (~/.tpu-agent-suite/runs/)."""
    return get_tpu_agent_suite_home() / RUNS_DIRNAME


def get_sessions_root() -> Path:
    """Return the central session records directory (~/.tpu-agent-suite/sessions/)."""
    return get_tpu_agent_suite_home() / SESSIONS_DIRNAME


def get_run_root(run_id: str) -> Path:
    """Return the run directory for `run_id` (~/.tpu-agent-suite/runs/<run_id>/)."""
    utils.validate_run_id(run_id)
    return get_runs_root() / run_id


def get_run_artifacts_dir(run_id: str) -> Path:
    """Return the artifacts directory for `run_id` (~/.tpu-agent-suite/runs/<run_id>/artifacts/)."""
    return get_run_root(run_id) / "artifacts"


def get_studies_root() -> Path:
    """Return the central studies directory (~/.tpu-agent-suite/studies/)."""
    return get_tpu_agent_suite_home() / STUDIES_DIRNAME


def get_study_root(study_id: str) -> Path:
    """Return the directory for `study_id` (~/.tpu-agent-suite/studies/<study_id>/)."""
    utils.validate_study_id(study_id)
    return get_studies_root() / study_id


def central_prefix_for(package: str, version: str = "") -> str:
    """Return the plan-relative central payload prefix for a package version."""
    return f"{CENTRAL_PACKAGES_PREFIX}/{utils.format_package_tag(package, version)}"


def central_home_spelling_for(package: str, version: str = "") -> str:
    """Return the home-anchored spelling of a central package dir for prose."""
    return f"~/{central_prefix_for(package, version)}"


def record_active_run(run_id: str, workspace: Path | str) -> None:
    """Record `run_id` as the workspace's active run in `.tpu-agent-suite/current_run_id`."""
    utils.validate_run_id(run_id)
    target = Path(workspace) / TPU_AGENT_SUITE_DIRNAME
    target.mkdir(parents=True, exist_ok=True)
    (target / ACTIVE_RUN_FILE).write_text(run_id, encoding="utf-8")


def get_active_run_id(workspace: Path | str | None = None) -> str | None:
    """Return the active run ID for `workspace` (or cwd), searching parent directories."""
    start = Path(workspace) if workspace else Path.cwd()
    for candidate in (start, *start.parents):
        marker = candidate / TPU_AGENT_SUITE_DIRNAME / ACTIVE_RUN_FILE
        if not marker.is_file():
            continue
        try:
            run_id = marker.read_text(encoding="utf-8").strip()
            utils.validate_run_id(run_id)
            return run_id
        except (OSError, utils.TpuAgentSuiteError):
            return None
    return None


def get_installed_package_root(package: str = "") -> Path:
    """Return the installed package directory under ~/.tpu-agent-suite/packages/.

    If `package` is unversioned, resolves to the highest installed version.
    Without `package`, returns the TPU agent suite home directory.
    """
    if not package:
        return get_tpu_agent_suite_home()

    pkg_base, pkg_version = utils.parse_package_tag(package)
    central_pkg = get_package_storage_root(pkg_base, pkg_version)
    if central_pkg.is_dir():
        return central_pkg
    if not pkg_version:
        installed = latest_installed_package(pkg_base)
        if installed is not None:
            return installed
    raise utils.TpuAgentSuiteError(
        f"package {package} is not installed in"
        f" {get_tpu_agent_suite_home() / PACKAGES_DIRNAME} --"
        " install the distribution first")


def get_environment(workspace: Path | str | None = None,
                    package: str = "") -> tuple[dict[str, Any], Path | None]:
    """Load environment.json for `package` in `workspace` or user scope."""
    try:
        return resolve_environment(package, workspace)
    except ValueError as exc:
        raise utils.TpuAgentSuiteError(
            f"environment.json is not valid JSON: {exc}. Fix it, or delete "
            "it and set compute up again.") from exc


def latest_installed_package(package: str) -> Path | None:
    """Return the highest installed version directory for `package`, or None."""
    if not package:
        return None
    packages_root = get_tpu_agent_suite_home() / PACKAGES_DIRNAME
    if not packages_root.is_dir():
        return None

    candidates: list[tuple[utils.SemVer, Path]] = []
    for entry in packages_root.iterdir():
        if not entry.is_dir():
            continue
        name, _, version = entry.name.partition("@")
        if name != package or not version:
            continue
        try:
            candidates.append((utils.SemVer.parse(version), entry))
        except (ValueError, utils.TpuAgentSuiteError):
            continue

    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0])[1]
