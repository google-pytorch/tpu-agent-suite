#!/usr/bin/env python3
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
"""Assembly and upload of one session's trajectory bundle.

Produces a gzipped tar of transcripts and capped artifacts, plus
an uncompressed manifest uploaded beside it so a reader can see how the bundle
was scoped without downloading it.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import io
import json
import pathlib
import platform
import shutil
import socket
import subprocess
import tarfile
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from framework import harnesses, journal, jsonio, utils

from .resolvers import (Cut, SessionBundle, TranscriptRef, file_size,
                        read_jsonl, resolve_session)

Path = pathlib.Path
TpuAgentSuiteError = utils.TpuAgentSuiteError

# Artifacts above this ride in the manifest's drop report instead of the
# bundle. Transcripts are never capped: one measured session was 1.6 MB of
# transcript alone, so a bundle-wide budget would truncate the payload.
ARTIFACT_BYTE_CAP = 10 * 1024 * 1024

BUNDLE_NAME = "bundle.tar.gz"
MANIFEST_NAME = "manifest.json"
PREFIX_ROOT = "tpu-agent-trajectories"
# Session ids are UUIDs; eight hex characters keep the leaf readable.
SESSION_ID_LEAF_CHARS = 8

# --------------------------------------------------------------------------
# Bundling
# --------------------------------------------------------------------------


def _safe_component(value: str) -> str:
    """Reduce an identifier to characters that are safe inside an object path."""
    kept = [c if (c.isalnum() or c in "-_.@") else "_" for c in value.strip()]
    return "".join(kept) or "unknown"


def _host_info() -> dict[str, str]:
    """Describe the machine the session ran on."""
    return {
        "os": platform.system(),
        "platform": platform.platform(),
        "arch": platform.machine(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
    }


def slice_root_records(cut: Cut,
                       records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop everything before the cut, leaving the entrypoint's own work.

    Generic over both harnesses: `cut.mode` says whether the key is a record
    position or the name of a field to compare against.
    """
    if not cut.detected or cut.key is None:
        return records
    if cut.mode == "index":
        return records[int(cut.key):]
    return [
        record for record in records
        if int(record.get(cut.mode) or 0) >= int(cut.key)
    ]


def _sliced_root_jsonl(cut: Cut, ref: TranscriptRef) -> str:
    """The root transcript cut at `cut`, re-serialized as JSONL text."""
    records = slice_root_records(cut, read_jsonl(ref.path))
    return "".join(json.dumps(record) + "\n" for record in records)


def _subagent_filename(ref: TranscriptRef) -> str:
    """File name a subagent transcript ships under, ordered by launch."""
    return f"{ref.order:04d}-{_safe_component(ref.label)}-{ref.agent_id}.jsonl"


def collect_run_files(directory: Path,
                      cap: int) -> tuple[list[Path], list[dict[str, Any]]]:
    """Split a run directory's files into those under `cap` and those over it.

    A `cap` of 0 means no limit.
    Oversized files are reported rather than silently omitted -- a dropped
    artifact is usually the one that explains the failure being reported.
    """
    included: list[Path] = []
    dropped: list[dict[str, Any]] = []
    if not directory.is_dir():
        return included, dropped
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        size = file_size(path)
        if cap and size > cap:
            dropped.append({
                "path": str(path),
                "bytes": size,
                "reason": f"exceeds {cap} byte per-artifact cap",
            })
            continue
        included.append(path)
    return included, dropped


def object_prefix(user_id: str, hostname: str, uploaded_at: str,
                  session_id: str) -> str:
    """Build the destination prefix.

    Timestamp and session id share the leaf so lexical order is chronological;
    user and host are separate levels so the bucket browses well. Only the
    first SESSION_ID_LEAF_CHARS of the session id go into the leaf: the
    timestamp already makes it unique, and the manifest carries the full id.
    """
    # Normalized to YYYYMMDDTHHMMSSZ so a prefix looks the same whichever
    # ISO spelling produced it, with or without fractional seconds.
    core = uploaded_at.split(".")[0].replace("-", "").replace(":", "")
    stamp = core.split("+")[0].rstrip("Z") + "Z"
    short_id = _safe_component(session_id)[:SESSION_ID_LEAF_CHARS]
    leaf = f"{_safe_component(stamp)}-{short_id}"
    return (f"{PREFIX_ROOT}/{_safe_component(user_id)}/"
            f"{_safe_component(hostname)}/{leaf}")


def build_manifest(
    *,
    bundle: SessionBundle,
    message: str,
    package: str,
    run_id: str,
    environment: dict[str, Any],
    uploaded_at: str,
    dropped: list[dict[str, Any]],
    artifact_count: int,
    bytes_total: int,
) -> dict[str, Any]:
    """Assemble the manifest uploaded uncompressed beside the bundle."""
    host = _host_info()
    return {
        # `provided` distinguishes "user said nothing" from "key missing",
        # since the note is optional.
        "note": {
            "text": message,
            "provided": bool(message.strip()),
        },
        "package": {
            "name": package,
            "version": utils.TPU_AGENT_SUITE_VERSION,
        },
        "harness": {
            "name": bundle.harness,
            "version": bundle.harness_version or "unknown",
            "version_source":
            "transcript" if bundle.harness_version else "none",
        },
        "models": bundle.models,
        "host": host,
        "environment": environment,
        "session": {
            "session_id": bundle.session_id,
            "run_id": run_id,
            "uploaded_at": uploaded_at,
        },
        # Records how the bundle was scoped, so a bundle always states whether
        # it is the entrypoint's work or a whole session shipped by consent.
        "cut": {
            "detected": bundle.cut.detected,
            "basis": bundle.cut.basis,
            "key": bundle.cut.key,
        },
        "contents": {
            "transcripts": [{
                "kind": ref.kind,
                "label": ref.label,
                "agent_id": ref.agent_id,
                "parent_id": ref.parent_id,
                "depth": ref.depth,
                "order": ref.order,
                "bytes": ref.bytes,
                "source": ref.source,
            } for ref in bundle.transcripts],
            "artifact_count":
            artifact_count,
            "bundle_bytes":
            bytes_total,
            "dropped":
            dropped,
            "missing":
            bundle.missing,
        },
    }


def write_bundle(bundle: SessionBundle, run_root: Path | None,
                 tar_path: Path) -> tuple[int, int, list[dict[str, Any]]]:
    """Write the gzipped tar and report (artifact count, bytes, dropped).

    The root transcript is sliced at the cut; subagent transcripts are whole,
    because an included subagent is entirely in scope by construction.
    """
    dropped: list[dict[str, Any]] = []
    artifact_count = 0
    with tarfile.open(tar_path, "w:gz") as tar:
        for ref in bundle.transcripts:
            if ref.kind == "root":
                # Sliced in memory so the source transcript is never touched.
                payload = _sliced_root_jsonl(bundle.cut, ref).encode()
                info = tarfile.TarInfo("transcripts/root.jsonl")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
                continue
            if not ref.path.is_file():
                continue
            tar.add(ref.path, arcname=f"transcripts/{_subagent_filename(ref)}")

        # Artifacts under the byte cap are bundled; oversized ones are dropped and reported.
        if run_root is not None:
            for name in (journal.JOURNAL_FILE, journal.RENDERED_FILE):
                path = run_root / name
                if path.is_file():
                    tar.add(path, arcname=name)
            artifacts, dropped = collect_run_files(run_root / "artifacts",
                                                   ARTIFACT_BYTE_CAP)
            for path in artifacts:
                tar.add(
                    path,
                    arcname=
                    f"artifacts/{path.relative_to(run_root / 'artifacts')}")
            artifact_count = len(artifacts)
    return artifact_count, file_size(tar_path), dropped


def _clean_bucket(bucket: str) -> str:
    """Bucket name without its `gs://` prefix or trailing slash.

    Reuses the compute drivers' normalizer rather than repeating the rule, so
    `gs://my-bucket`, `my-bucket/` and `my-bucket` all name the same place.
    The import is safe without google-cloud-storage: that dependency is
    resolved inside `get_storage_client`, not at module import.
    """
    from framework.compute.drivers import gcs_staging
    return gcs_staging.clean_bucket_name(bucket)


def _default_uploader(local_path: Path, bucket: str, blob_name: str) -> None:
    """Upload one file, reusing the compute drivers' GCS helper.

    Imported lazily: google-cloud-storage is optional, and a user who declined
    the bucket must never need it installed.
    """
    from framework.compute.drivers import gcs_staging
    gcs_staging.upload_file_to_gcs(local_path, bucket, blob_name)


def verify_bucket_exists(bucket: str) -> None:
    """Fail loudly before uploading if the destination cannot be seen.

    This is an existence check, not a permission check: it catches the
    mistyped bucket name that would otherwise route a raw trajectory
    somewhere the user never intended. A bucket that exists but refuses the
    write still fails at upload, with the storage client's own error.
    """
    from framework.compute.drivers import gcs_staging
    clean = gcs_staging.clean_bucket_name(bucket)
    # The bucket alone: `bucket` may name a folder in it, as uploads allow.
    name, _ = gcs_staging.locate(clean, "")
    client = gcs_staging.get_storage_client()
    if not client.bucket(name).exists():
        raise TpuAgentSuiteError(
            f"destination bucket gs://{clean} does not exist or is not readable"
        )


def resolve_for_export(harness: str, session_id: str, entrypoint: str,
                       include_full_session: bool) -> SessionBundle:
    """Resolve a session and apply the whole-session consent rule.

    Shared by both destinations so the `--all` refusal cannot apply to one
    and not the other.
    """
    bundle = resolve_session(harness, session_id, entrypoint)
    # An undetected cut means the bundle would be the whole session, which is
    # the user's call to make, not ours. Refuse with the exact re-invocation.
    if not bundle.cut.detected:
        if not include_full_session:
            raise TpuAgentSuiteError(
                "could not find where the session's work began, so the bundle "
                "would cover the entire session. Re-run with --all to send it "
                "anyway.")
        bundle = dataclasses.replace(bundle,
                                     cut=Cut(detected=False,
                                             basis="user-consented-all"))
    return bundle


def export_trajectory(
    *,
    harness: str,
    session_id: str,
    run_root: Path | None,
    run_id: str,
    entrypoint: str,
    message: str,
    bucket: str,
    user_id: str,
    package: str,
    environment: dict[str, Any],
    include_full_session: bool = False,
    uploader: Callable[[Path, str, str], None] | None = None,
    uploaded_at: str = "",
) -> dict[str, Any]:
    """Resolve, bundle and upload one session's trajectory and its note.

    Raises rather than uploading when the harness has no resolver, when no
    bucket was configured, or when the cut could not be found and the user has
    not consented to shipping the whole session.
    """
    # Checked first: with sharing never set up, whether to send the whole
    # session is a question for later.
    if not bucket:
        raise TpuAgentSuiteError(
            "no destination is configured, so trajectories cannot be "
            "exported. Trajectory sharing is not set up; turn it on with "
            "`config set trajectory_enabled=true trajectory_sink=gcs "
            "trajectory_bucket=gs://... --user-confirmed` (or "
            "trajectory_sink=harbor).")
    bundle = resolve_for_export(harness, session_id, entrypoint,
                                include_full_session)

    uploaded_at = uploaded_at or utils.utc_now_iso()
    bucket = _clean_bucket(bucket)
    prefix = object_prefix(user_id,
                           _host_info()["hostname"], uploaded_at, session_id)

    with tempfile.TemporaryDirectory() as staging:
        tar_path = Path(staging) / BUNDLE_NAME
        manifest_path = Path(staging) / MANIFEST_NAME
        artifact_count, bytes_total, dropped = write_bundle(
            bundle, run_root, tar_path)
        manifest = build_manifest(bundle=bundle,
                                  message=message,
                                  package=package,
                                  run_id=run_id,
                                  environment=environment,
                                  uploaded_at=uploaded_at,
                                  dropped=dropped,
                                  artifact_count=artifact_count,
                                  bytes_total=bytes_total)
        jsonio.write_json(manifest_path, manifest)

        objects = [
            {
                "name": f"{prefix}/{MANIFEST_NAME}",
                "bytes": file_size(manifest_path)
            },
            {
                "name": f"{prefix}/{BUNDLE_NAME}",
                "bytes": bytes_total
            },
        ]
        summary = {
            "uri": f"gs://{bucket}/{prefix}",
            "objects": objects,
            "cut": manifest["cut"],
            "transcripts": len(bundle.transcripts),
            "dropped": dropped,
        }

        # Upload the manifest last: a reader that sees it can trust the
        # bundle beside it is already complete.
        send = uploader or _default_uploader
        if uploader is None:
            verify_bucket_exists(bucket)
        send(tar_path, bucket, f"{prefix}/{BUNDLE_NAME}")
        send(manifest_path, bucket, f"{prefix}/{MANIFEST_NAME}")
    return summary


# --------------------------------------------------------------------------
# The Harbor hub
# --------------------------------------------------------------------------
#
# A hand-driven session has no job directory, so wrap its transcript in one:
# harbor's agent converts to ATIF and a single synthetic trial carries it.
# harbor is imported inside these functions so an environment without it still
# reaches the bucket above.


@functools.cache
def harbor_missing() -> str:
    """Name what harbor is missing, or "" when it is usable.

    Both halves are needed and they can be installed apart: the converter
    imports harbor.agents, while the upload execs the `harbor` binary. A pipx
    install puts the binary on PATH without making it importable, and a
    library-only install does the reverse. find_spec locates the package
    without executing it, so the import stays lazy.
    """
    if importlib.util.find_spec("harbor") is None:
        return ("the harbor package is not importable; install it into this"
                " interpreter, e.g. `pip install harbor==0.23.0`")
    if shutil.which("harbor") is None:
        return "the `harbor` command is not on PATH"
    return ""


def _hub_url(output: str) -> str | None:
    """Pull the hub URL out of `harbor upload` output, or None if it printed none."""
    for token in (output or "").split():
        cleaned = token.strip().lstrip("(<[").rstrip(".,)>]")
        if cleaned.startswith("http://") or cleaned.startswith("https://"):
            return cleaned
    return None


def stage_session_dir(bundle: SessionBundle, staging: Path) -> Path:
    """Lay one session's transcripts out the way harbor's converters glob.

    harbor merges every `*.jsonl` plus `subagents/*.jsonl` under the directory
    it is handed, so pointing it at the harness's own store would sweep in
    every other session filed beside this one. Staging a directory that holds
    this session and nothing else makes that structurally impossible.

    Mirrors write_bundle: the root transcript is sliced at the cut, subagent
    transcripts ship whole.
    """
    session_dir = staging / bundle.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    for ref in bundle.transcripts:
        if ref.kind == "root":
            (session_dir / f"{bundle.session_id}.jsonl").write_text(
                _sliced_root_jsonl(bundle.cut, ref), encoding="utf-8")
            continue
        if not ref.path.is_file():
            continue
        subagents = session_dir / "subagents"
        subagents.mkdir(exist_ok=True)
        (subagents / _subagent_filename(ref)).write_bytes(
            ref.path.read_bytes())
    return session_dir


def _build_trajectory(harness: str, session_dir: Path, logs_dir: Path):
    """Convert a harness transcript to ATIF using harbor's own converter."""
    # Each CLI converts differently, and not all of them from a directory:
    # Google harnesses read a step transcript into records first, because
    # harbor exposes no directory-shaped entry point for them. They share one
    # transcript format, so all of them go through harbor's Antigravity
    # converter.
    if harnesses.is_google_harness(harness):
        from harbor.agents.installed.antigravity_cli import AntigravityCli
        agent = AntigravityCli(logs_dir=logs_dir)
        # Top-level only: the staged directory holds this session's root
        # transcript and nothing else, so there is no neighbour to match.
        # Subagents are read too: their transcripts are part of the session,
        # and reading only the root would drop every delegated step from the
        # hub copy while the bundle beside it still carries them.
        records = []
        for candidate in (sorted(session_dir.glob("*.jsonl")) +
                          sorted(session_dir.rglob("subagents/*.jsonl"))):
            records.extend(
                agent._load_step_transcript(candidate)  # noqa: SLF001
                or [])
        trajectory = (
            agent._convert_step_transcript_to_atif(records)  # noqa: SLF001
            if records else None)
    else:
        if harness == "codex":
            from harbor.agents.installed.codex import Codex as cls
        else:
            from harbor.agents.installed.claude_code import ClaudeCode as cls
        agent = cls(logs_dir=logs_dir)
        trajectory = agent._convert_events_to_trajectory(
            session_dir)  # noqa: SLF001
    if trajectory is None:
        raise ValueError(
            f"No trajectory could be read from {session_dir}. A session that"
            " never took a step leaves nothing to convert.")
    return trajectory


def write_job_dir(
    job_dir: Path,
    harness: str,
    trajectory: Any,
    task_name: str = "sandbox-session",
) -> Path:
    """Lay out the job and trial files `harbor upload` expects to find."""
    from harbor.models.job.config import JobConfig, RetryConfig
    from harbor.models.job.lock import (JobLock, TaskLock, TrialLock,
                                        VerifierLock)
    from harbor.models.job.result import JobResult, JobStats
    from harbor.models.task.id import LocalTaskId
    from harbor.models.trial.config import (AgentConfig, EnvironmentConfig,
                                            TaskConfig, TrialConfig)
    from harbor.models.trial.result import AgentInfo, TrialResult

    started = datetime.now(timezone.utc)
    trial_name = f"{task_name.replace('/', '-')}__session"
    trial_dir = job_dir / trial_name
    (trial_dir / "agent").mkdir(parents=True, exist_ok=True)

    trajectory_path = trial_dir / "agent" / "trajectory.json"
    trajectory_path.write_text(trajectory.model_dump_json(indent=2,
                                                          exclude_none=True),
                               encoding="utf-8")

    # The checksum identifies the task content a trial ran against. Nothing
    # here was scored against a task, so it covers the trajectory instead:
    # stable for one session, distinct across sessions.
    checksum = utils.compute_digest(trajectory_path)

    agent = getattr(trajectory, "agent", None)
    model_name = getattr(agent, "model_name", "") or ""
    trial_config = TrialConfig(
        task=TaskConfig(name=task_name),
        trial_name=trial_name,
    )
    # No verifier_result: nothing scored this session, and an empty pass is a
    # worse lie than an absent one.
    trial_result = TrialResult(
        id=uuid.uuid4(),
        task_name=task_name,
        trial_name=trial_name,
        trial_uri=trial_dir.resolve().as_uri(),
        task_id=LocalTaskId(path=task_name),
        task_checksum=checksum,
        config=trial_config,
        agent_info=AgentInfo(
            name=harness,
            version=getattr(agent, "version", "") or "unknown",
            model_info={"name": model_name} if model_name else None),
        started_at=started,
        finished_at=datetime.now(timezone.utc),
    )
    (trial_dir / "config.json").write_text(trial_config.model_dump_json(
        indent=2, exclude_none=True),
                                           encoding="utf-8")
    # Both lock files are required: harbor upload refuses a trial without one,
    # and reads the job's to enumerate the trials it should send.
    trial_lock = TrialLock(
        task=TaskLock(name=task_name,
                      type="local",
                      digest=f"sha256:{checksum}",
                      path=task_name),
        agent=AgentConfig(name=harness, model_name=model_name or None),
        environment=EnvironmentConfig(),
        verifier=VerifierLock(disable=True),
    )
    (trial_dir / "lock.json").write_text(trial_lock.model_dump_json(
        indent=2, exclude_none=True),
                                         encoding="utf-8")
    (trial_dir / "result.json").write_text(trial_result.model_dump_json(
        indent=2, exclude_none=True),
                                           encoding="utf-8")

    job_result = JobResult(
        id=uuid.uuid4(),
        started_at=started,
        finished_at=datetime.now(timezone.utc),
        n_total_trials=1,
        stats=JobStats(n_completed_trials=1),
    )
    # harbor upload refuses a job directory without a job-level config.json,
    # before it ever reads the trial.
    job_config = JobConfig(job_name=job_dir.name,
                           jobs_dir=str(job_dir.parent),
                           n_attempts=1)
    (job_dir / "config.json").write_text(job_config.model_dump_json(
        indent=2, exclude_none=True),
                                         encoding="utf-8")
    job_lock = JobLock(n_concurrent_trials=1,
                       retry=RetryConfig(),
                       trials=[trial_lock])
    (job_dir / "lock.json").write_text(job_lock.model_dump_json(
        indent=2, exclude_none=True),
                                       encoding="utf-8")
    (job_dir / "result.json").write_text(job_result.model_dump_json(
        indent=2, exclude_none=True),
                                         encoding="utf-8")
    return trial_dir


# `harbor upload` is a network call over a bundle that can run to tens of MB,
# so the ceiling is generous; it exists to bound a hang, not to pace a slow
# link.
UPLOAD_TIMEOUT_SECONDS = 900


def export_to_harbor(
    *,
    bundle: SessionBundle,
    job_dir: Path,
    run_root: Path | None,
    run_id: str,
    message: str,
    package: str,
    environment: dict[str, Any],
    task_name: str = "",
    share_org: str = "",
    uploaded_at: str = "",
) -> dict[str, Any]:
    """Build a one-trial job from a resolved session and upload it to the hub.

    Takes the bundle the caller already resolved rather than re-deriving it:
    the session id is the only thing tying the upload back to the conversation,
    and guessing at one loses it.

    Harbor is an alternative to the bucket, not a first attempt at it, so the
    manifest and the bundle tar ride along in the job directory. The ATIF
    trajectory alone would drop the note, the consent basis and the run's
    artifacts, and with no bucket export behind it there is nowhere else they
    would appear.
    """
    task_name = task_name or bundle.session_id
    job_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        session_dir = stage_session_dir(bundle, Path(staging))
        trajectory = _build_trajectory(bundle.harness, session_dir, job_dir)
    trial_dir = write_job_dir(job_dir, bundle.harness, trajectory, task_name)

    # Same bundle and manifest the bucket would have received, written beside
    # the trajectory so the note, the consent basis and the run's artifacts
    # travel with the upload.
    artifact_count, bytes_total, dropped = write_bundle(
        bundle, run_root, trial_dir / BUNDLE_NAME)
    manifest = build_manifest(bundle=bundle,
                              message=message,
                              package=package,
                              run_id=run_id,
                              environment=environment,
                              uploaded_at=uploaded_at or utils.utc_now_iso(),
                              dropped=dropped,
                              artifact_count=artifact_count,
                              bytes_total=bytes_total)
    jsonio.write_json(trial_dir / MANIFEST_NAME, manifest)

    summary: dict[str, Any] = {
        "session_id": bundle.session_id,
        "job_dir": str(job_dir),
        "trajectory": str(trial_dir / "agent" / "trajectory.json"),
        "manifest": str(trial_dir / MANIFEST_NAME),
        "bundle": str(trial_dir / BUNDLE_NAME),
        "steps": len(getattr(trajectory, "steps", []) or []),
        "transcripts": len(bundle.transcripts),
        "cut": manifest["cut"],
        "dropped": dropped,
        "uploaded": False,
        "destination": str(job_dir),
    }

    # --yes: sharing with an org the uploader does not belong to otherwise
    # prompts for confirmation, and with stdout captured that prompt is written
    # into a pipe nobody reads, hanging the export.
    cmd = ["harbor", "upload", str(job_dir), "--private"]
    if share_org:
        cmd += ["--share-org", share_org, "--yes"]
    try:
        result = subprocess.run(cmd,
                                check=False,
                                capture_output=True,
                                text=True,
                                timeout=UPLOAD_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise TpuAgentSuiteError(
            f"`{' '.join(cmd)}` did not finish within"
            f" {UPLOAD_TIMEOUT_SECONDS}s. The job is staged at {job_dir}."
        ) from exc
    if result.returncode != 0:
        raise TpuAgentSuiteError(
            f"`{' '.join(cmd)}` exited {result.returncode}: "
            f"{(result.stderr or result.stdout or '').strip()[:400]}"
            f" The job is staged at {job_dir}.")
    summary["uploaded"] = True
    summary["command"] = " ".join(cmd)
    # harbor prints the hub URL it uploaded to. Report that rather than the
    # local job directory, which is the staging area and not where it landed.
    summary["destination"] = _hub_url(result.stdout) or "the Harbor hub"
    summary["output"] = (result.stdout or "").strip()
    return summary


# --------------------------------------------------------------------------
# End Harbor hub
# --------------------------------------------------------------------------
