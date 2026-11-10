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
"""Run namespaces and the session records that point at them.

A run is a directory under ~/.tpu-agent-suite/runs/ holding one session's
artifacts. Sessions are recorded under ~/.tpu-agent-suite/sessions/: the run
each one uses and its active tool-limited skill.
"""

from __future__ import annotations

import datetime
import pathlib
import shutil
import sys
import uuid
from typing import Any, Iterator

from framework import compute, harnesses, jsonio, paths, utils

Path = pathlib.Path
TpuAgentSuiteError = utils.TpuAgentSuiteError
_safe_path_component = utils.safe_path_component


def generate_run_id() -> str:
    """Generate a collision-resistant timestamped run identifier."""
    timestamp = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:12]}"


def start_run(
    package: str = "",
    workspace: Path | str | None = None,
    compute_profile: str | None = None,
) -> dict[str, Any]:
    """Create a collision-resistant run namespace inside central store ~/.tpu-agent-suite/runs/."""
    package_str = str(package) if package else "no-package"
    # A run without a package has no environment to consult; the "no-package"
    # placeholder is a run-document label, not a resolvable package.
    environment: dict[str, Any] = {}
    if package:
        environment, _ = paths.get_environment(workspace, package_str)

    profile_name = (compute_profile or "").strip()
    if compute_profile is not None and not profile_name:
        raise compute.ComputeConfigError(
            "--profile must be a non-empty profile name.",
            error_code="INVALID_FIELD")
    if profile_name:
        # Refuse a name the run could never resolve, rather than recording it
        # and failing at the first compute command.
        if not package:
            raise compute.ComputeConfigError(
                "start-run --profile needs --package: a run without a package "
                "has no environment.json to look the profile up in.",
                error_code="MISSING_FIELD")
        profiles, _ = compute.normalize_environment_profiles(environment)
        if profile_name not in profiles:
            raise compute.unknown_profile_error(profile_name, "--profile",
                                                profiles)

    # Always allocate run in central storage (~/.tpu-agent-suite/runs/)
    runs_root = paths.get_runs_root()
    runs_root.mkdir(parents=True, exist_ok=True)

    for _ in range(10):
        run_id = generate_run_id()
        run_root = paths.get_run_root(run_id)
        try:
            run_root.mkdir()
        except FileExistsError:
            continue

        paths.get_run_artifacts_dir(run_id).mkdir()
        document: dict[str, Any] = {
            "created_at": utils.utc_now_iso(),
            # 0 means no environment was recorded (packageless run); real
            # environment revisions start at 1.
            "environment_revision": environment.get("revision", 0),
            "version": utils.TPU_AGENT_SUITE_VERSION,
            "package": package_str,
            "run_id": run_id,
        }
        if profile_name:
            document["compute_profile"] = profile_name
        jsonio.write_json(run_root / "run.json", document)
        if workspace:
            paths.record_active_run(run_id, workspace)
        return document

    raise TpuAgentSuiteError("could not allocate a unique run ID")


def format_run_context(run_id: str) -> str:
    """Render the run-namespace context injected into the model conversation."""
    # Resolve the actual path on disk so TPU_AGENT_SUITE_HOME overrides work.
    try:
        artifacts_dir: Any = paths.get_run_artifacts_dir(run_id)
    except TpuAgentSuiteError:
        artifacts_dir = f"{paths.RUNS_HOME_SPELLING}/{run_id}/artifacts"
    return (
        f"TPU agent suite run namespace: {run_id}\n"
        f"Artifacts: {artifacts_dir}/\n"
        f"Use this exact run_id for all subagent delegations and artifact creation."
    )


def get_session_file(session_id: str) -> Path:
    """Return path to session metadata file under ~/.tpu-agent-suite/sessions/."""
    return paths.get_sessions_root(
    ) / f"{_safe_path_component(session_id)}.json"


def record_session_run(
    session_id: str,
    run_id: str,
    package: str = "",
) -> None:
    """Record an active session-to-run mapping under ~/.tpu-agent-suite/sessions/."""
    if not session_id or not run_id:
        return
    session_file = get_session_file(session_id)
    existing = jsonio.load_json_or_empty(session_file)
    session_doc = {
        "created_at": existing.get("created_at") or utils.utc_now_iso(),
        "package": package,
        "run_id": run_id,
        "session_id": session_id,
    }
    if session_doc == existing:
        return
    jsonio.write_json(session_file, session_doc)


def _iter_session_ids(session_id: str) -> Iterator[str]:
    """Yield `session_id`, followed by its parent session ID if it is a subagent."""
    yield session_id
    for name in harnesses.names():
        parent_id = harnesses.HARNESSES[name].parent_session_id(session_id)
        if parent_id and parent_id != session_id:
            yield parent_id
            return


def resolve_session_run_id(session_id: str | None = None) -> str | None:
    """Resolve the active run_id associated with a session or child conversation."""
    if not session_id:
        return None
    for cid in _iter_session_ids(session_id):
        run_id = jsonio.load_json_or_empty(get_session_file(cid)).get("run_id")
        if run_id:
            return str(run_id)
    return None


def get_active_skill_file(session_id: str) -> Path:
    """Return the file that records a session's active tool-limited skill."""
    return (paths.get_sessions_root() /
            f"{_safe_path_component(session_id)}.active-skill.json")


def purge_data(force: bool = False) -> None:
    """Purge central run history and session records (~/.tpu-agent-suite/{runs,sessions}/)."""
    targets = [
        p for p in (paths.get_runs_root(), paths.get_sessions_root())
        if p.is_dir()
    ]
    if not targets:
        return
    if not force:
        listing = "\n".join(f"  {p}" for p in targets)
        try:
            choice = input(
                "Are you sure you want to permanently delete all run data, artifacts, "
                f"and session records in:\n{listing}\n[y/N]: ").strip().lower(
                )
        except (EOFError, KeyboardInterrupt):
            choice = "n"
        if choice not in ("y", "yes"):
            print("Purge data cancelled by user.", file=sys.stderr)
            return
    for target in targets:
        shutil.rmtree(target, ignore_errors=True)
    print("Purged all run data, artifacts, and session records.")


def session_for_run(run_id: str) -> str:
    """Return the id of the session whose record names `run_id`."""
    # Session records are keyed by session id, so the match runs the other
    # way: scan for records naming this run and take the newest.
    candidates = []
    for record_path in paths.get_sessions_root().glob("*.json"):
        document = jsonio.load_json_or_empty(record_path)
        if document.get("run_id") == run_id and document.get("session_id"):
            candidates.append((str(document.get("created_at")
                                   or ""), str(document["session_id"])))
    if not candidates:
        raise TpuAgentSuiteError(f"no session is recorded for run {run_id}")
    return sorted(candidates)[-1][1]
