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
each one uses, the context its hooks injected, and context still waiting to be
delivered on the Google harnesses.
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

# Ceiling on spooled context entries per session, oldest evicted first.
# Every live entry is replayed into emulated context on each invocation until
# pushed out by newer entries, so this bounds the per-call prompt overhead.
MAX_PENDING_CONTEXT_ENTRIES = 8


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


def _google_harness_subagent_descriptors(
        conversation_id: str) -> Iterator[Path]:
    """Yield subagent descriptors matching `conversation_id` across brain roots.

    Descriptors live at `<brain_dir>/<parent_cid>/.system_generated/subagents/<cid>.json`.
    """
    for brain_root in sorted(set(
            harnesses.GOOGLE_HARNESS_BRAIN_DIRS.values())):
        brain_dir = Path.home() / ".gemini" / brain_root / "brain"
        if brain_dir.is_dir():
            matches = list(
                brain_dir.glob(
                    f"*/.system_generated/subagents/{conversation_id}.json"))
            if matches:
                yield matches[0]


def get_google_harness_subagent_name(
    conversation_id: str,
    transcript_path: str = "",
) -> str | None:
    """Return the subagent name for `conversation_id`, or None for a root session."""
    if transcript_path and "/worktrees/subagent-" in transcript_path:
        return "subagent"
    if not conversation_id:
        return None

    for descriptor in _google_harness_subagent_descriptors(conversation_id):
        desc_data = jsonio.load_json_or_empty(descriptor)
        sub_desc = desc_data.get("subagentDescriptor")
        if isinstance(sub_desc, dict):
            name = sub_desc.get("typeName") or sub_desc.get("role")
            if name:
                return name

    return None


def get_google_harness_parent_conversation_id(
        conversation_id: str) -> str | None:
    """Return the parent conversation ID for a Google harness subagent, or None."""
    if not conversation_id:
        return None
    descriptor = next(_google_harness_subagent_descriptors(conversation_id),
                      None)
    return descriptor.parents[2].name if descriptor else None


def get_session_file(session_id: str) -> Path:
    """Return path to session metadata file under ~/.tpu-agent-suite/sessions/."""
    return paths.get_sessions_root(
    ) / f"{_safe_path_component(session_id)}.json"


def get_hook_context_file(session_id: str, hook_name: str) -> Path:
    """Return path to a hook's cached SessionStart context for a session."""
    contexts_dir = (paths.get_sessions_root() /
                    f"{_safe_path_component(session_id)}.contexts")
    return contexts_dir / f"{_safe_path_component(hook_name)}.json"


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


def record_session_hook_context(
    session_id: str,
    hook_name: str,
    context_text: str,
    package: str = "",
) -> None:
    """Record cached hook context under ~/.tpu-agent-suite/sessions/<id>.contexts/."""
    if not session_id or not hook_name or not context_text:
        return
    jsonio.write_json(
        get_hook_context_file(session_id, hook_name), {
            "context": context_text,
            "hook_name": hook_name,
            "package": package,
            "recorded_at": utils.utc_now_iso(),
        })


def get_pending_context_dir(session_id: str) -> Path:
    """Return the spool directory for hook context awaiting delivery.

    Used on Google harnesses where PostToolUse responses cannot carry context;
    hooks spool context here and PreInvocation delivers it on subsequent turns.
    """
    return (paths.get_sessions_root() /
            f"{_safe_path_component(session_id)}.pending")


def record_pending_hook_context(
    session_id: str,
    hook_name: str,
    context_text: str,
    package: str = "",
) -> None:
    """Spool hook context into a per-entry file for delivery on a later event."""
    if not session_id or not context_text:
        return
    pending_dir = get_pending_context_dir(session_id)
    pending_dir.mkdir(parents=True, exist_ok=True)

    recorded_at = utils.utc_now_iso()
    entry_name = f"{recorded_at}-{uuid.uuid4().hex[:8]}.json"
    jsonio.write_json(
        pending_dir / _safe_path_component(entry_name), {
            "context": context_text,
            "hook_name": hook_name,
            "package": package,
            "recorded_at": recorded_at,
        })

    # Evict oldest entries past MAX_PENDING_CONTEXT_ENTRIES.
    existing = sorted(pending_dir.glob("*.json"))
    overflow = len(existing) - MAX_PENDING_CONTEXT_ENTRIES
    if overflow > 0:
        for stale in existing[:overflow]:
            stale.unlink(missing_ok=True)


def replay_pending_hook_context(session_id: str | None) -> str:
    """Return spooled context for `session_id` (without falling back to a parent session)."""
    if not session_id:
        return ""
    pending_dir = get_pending_context_dir(session_id)
    if not pending_dir.is_dir():
        return ""

    texts: list[str] = []
    for entry_file in sorted(pending_dir.glob("*.json")):
        entry = jsonio.load_json_or_empty(entry_file)
        context_text = entry.get("context")
        if not context_text:
            entry_file.unlink(missing_ok=True)
            continue
        texts.append(str(context_text))
    return "\n\n".join(texts)


def _iter_session_ids(session_id: str) -> Iterator[str]:
    """Yield `session_id`, followed by its parent conversation ID if it is a subagent."""
    yield session_id
    parent_cid = get_google_harness_parent_conversation_id(session_id)
    if parent_cid:
        yield parent_cid


def resolve_session_hook_context(
    session_id: str | None,
    hook_name: str,
) -> str | None:
    """Resolve cached hook context text for a session and hook name."""
    if not session_id or not hook_name:
        return None
    for cid in _iter_session_ids(session_id):
        context_text = jsonio.load_json_or_empty(
            get_hook_context_file(cid, hook_name)).get("context")
        if context_text:
            return str(context_text)
    return None


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
