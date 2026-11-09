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
"""Syncing a `gke_job` Job's cluster state to its job record.

Also deletes Jobs and keeps a last-seen note for Jobs that vanish.
"""

from __future__ import annotations

import json
import re
import sys
from typing import TYPE_CHECKING, Any

from framework import jsonio

from .. import base, gcs_staging, jobs
from ..kubectl import KubectlError
from .gke_common import is_not_found
from .log_retrieval import save_logs
from .manifest import TOKEN_SECRET_SUFFIX, renew_token_secret
from .status import live_status, live_status_retrying_new_job

if TYPE_CHECKING:
    from ..kubectl import Kubectl
    from .gke_common import Path, StagingClient
    from .profile_settings import GkeJobConfig

# Timeout for `kubectl delete`, which waits out the pod's grace period (30s
# by default). On overrun the record stays live and the reaper retries.
KUBECTL_DELETE_TIMEOUT_SEC = 90


def mark_terminal(record: jobs.JobRecord,
                  status: base.AsynchronousExecutionResult) -> None:
    """Write a terminal status and its reason to the record.

    Jobs that never start are deleted right away, so the record is the
    only place left to explain them.
    """
    extra = ({
        "never_started": True
    } if status.details.get("never_started") else {})
    record.mark(status.status,
                exit_code=status.exit_code,
                error=status.error,
                **extra)


def recorded_status(job_id: str) -> base.AsynchronousExecutionResult | None:
    """Return a gone Job's status from its record, or None if not settled."""
    record = jobs.JobRecord.load(job_id)
    if record is None or record.status not in jobs.TERMINAL_STATUSES:
        return None
    details: dict[str, Any] = {"from_record": True}
    if record.extra.get("never_started"):
        details["never_started"] = True
    return base.AsynchronousExecutionResult(job_id=job_id,
                                            status=record.status,
                                            start_time=record.start_time,
                                            end_time=record.end_time,
                                            exit_code=record.exit_code,
                                            error=record.extra.get("error"),
                                            details=details)


def _last_seen_path(job_id: str) -> Path:
    return jobs.jobs_dir() / job_id / "last_seen.json"


def note_last_seen(job_id: str,
                   status: base.AsynchronousExecutionResult) -> None:
    """Save what this read saw, in case the Job later vanishes.

    Stored beside job.json, not in it, so a poll can't overwrite a
    terminal status another process wrote. Best-effort.
    """
    path = _last_seen_path(job_id)
    if not (path.parent / "job.json").is_file():
        return
    seen = {
        "status": status.status,
        "at": jobs.utc_now(),
        "node": status.details.get("node_name"),
        "pod": status.details.get("pod_name"),
        "reason": status.error,
    }
    try:
        jsonio.write_text_atomic(path, json.dumps(seen), mkdir=False)
    except OSError:
        pass


def last_seen_note(job_id: str) -> str:
    """`; last seen RUNNING on node N at T (pod P), reason: R`, or ''."""
    seen = jsonio.load_json_or_empty(_last_seen_path(job_id))
    if not seen.get("status"):
        return ""
    note = f"; last seen {seen['status']}"
    if seen.get("node"):
        note += f" on node {seen['node']}"
    if seen.get("at"):
        note += f" at {seen['at']}"
    if seen.get("pod"):
        note += f" (pod {seen['pod']})"
    if seen.get("reason"):
        note += f", reason: {seen['reason']}"
    return note


def read_and_record_status(kube: Kubectl, config: GkeJobConfig,
                           storage: StagingClient,
                           job_id: str) -> base.AsynchronousExecutionResult:
    """Read the Job's status and update its record.

    A Kueue-suspended Job is PENDING with `suspended` set, so a full
    queue doesn't look like missing hardware. One Kueue refused is FAILED
    with Kueue's reason.
    """
    try:
        status = live_status_retrying_new_job(kube, job_id)
    except base.ComputeConfigError as exc:
        # We may have deleted it after it finished. If so, its settled
        # record is the answer. Other errors still raise.
        if not is_not_found(exc):
            raise
        recorded = recorded_status(job_id)
        if recorded is not None:
            return recorded
        record = jobs.JobRecord.load(job_id)
        if record is None:
            raise
        # Something else removed it. Keep the record, since the profile
        # may now point at a different cluster.
        raise base.ComputeConfigError(
            f"Job {job_id} not found in namespace {config.namespace!r}"
            f"{last_seen_note(job_id)}. compute didn't delete it (its "
            f"record says {record.status}). Another run or a person "
            "removed it, or the compute profile now points at a "
            f"different cluster. `job logs --job-id {job_id}` returns any "
            "saved output, or what Cloud Logging kept. To carry on, "
            "submit it again; don't look for it with kubectl. "
            f"({exc})",
            error_code="JOB_GONE",
        ) from exc
    note_last_seen(job_id, status)
    if status.status in ("PENDING", "RUNNING"):
        renew_token_secret(kube, config, job_id)
    if status.details.get("never_started"):
        # Its pod may hold a TPU node or start later, and the reaper skips
        # settled records, so delete it now. Record the status after, so a
        # later read of a vanishing Job can't lose the reason.
        try:
            delete_job(kube, config, storage, job_id, status)
        except base.ComputeConfigError as exc:
            # Left unsettled, so the reaper tries again.
            print(f"gke_job: could not delete Job {job_id} -- {exc}",
                  file=sys.stderr)
            return status
    settle_record(kube, config, storage, job_id, status)
    return status


def delete_job(kube: Kubectl,
               config: GkeJobConfig,
               storage: StagingClient,
               job_id: str,
               status: base.AsynchronousExecutionResult | None = None) -> None:
    """`cancel`, given the Job's status when the caller has just read it."""
    record = jobs.JobRecord.load(job_id)
    settled = record is not None and record.status in jobs.TERMINAL_STATUSES
    if not settled and status is None:
        try:
            status = live_status(kube, job_id)
        except base.ComputeConfigError:
            # No Job to read: never created, or already gone.
            status = None
    finished = settled or (status is not None
                           and status.status in jobs.TERMINAL_STATUSES)
    running = status is not None and status.status == "RUNNING"
    never_ran = bool(record and record.extra.get("never_started")) or (
        status is not None and status.details.get("never_started"))
    if record and (finished or running) and not never_ran:
        # Save logs before the delete takes the pod. Skip Jobs that never
        # ran, where asking blocks.
        save_logs(kube, config, storage, job_id)
    objects = [("job", job_id), ("configmap", job_id)]
    if record is not None and record.extra.get("token_secret"):
        # Only for Jobs with a token, so cancel doesn't need Secret access.
        objects.append(("secret", job_id + TOKEN_SECRET_SUFFIX))
    for kind, obj in objects:
        proc = kube.run(["delete", kind, obj] + kube.ns() +
                        ["--ignore-not-found"],
                        check=False,
                        timeout=KUBECTL_DELETE_TIMEOUT_SEC)
        # Also treat a missing namespace as gone. kubectl says either
        # `(NotFound)` or `not found`.
        if proc.returncode != 0 and not re.search(r"not ?found", proc.stderr
                                                  or "", re.IGNORECASE):
            raise KubectlError(["delete", kind, obj], proc.stderr)
    if record is not None and config.staging_bucket:
        # Delete inputs only. `fetch_artifacts` deletes outputs once they
        # are on disk, so uncollected results survive.
        prefix = record.extra.get("staged_prefix")
        if prefix:
            gcs_staging.delete_staged_prefix(
                gcs_staging.clean_bucket_name(config.staging_bucket),
                f"{prefix}in/", **storage.client_kwargs(True))
    if record is None or settled:
        return
    if status is not None and status.status in jobs.TERMINAL_STATUSES:
        mark_terminal(record, status)
    else:
        # The error is the Job's state at cancel time, not a failure.
        error = status.error if status is not None else None
        record.mark("CANCELLED",
                    error=f"Cancelled while: {error}" if error else None,
                    **({
                        "never_started": True
                    } if never_ran else {}))


def settle_record(kube: Kubectl, config: GkeJobConfig, storage: StagingClient,
                  job_id: str,
                  status: base.AsynchronousExecutionResult) -> None:
    """Write a terminal cluster status to the job record.

    Nothing else updates the record after dispatch, and `job list` and
    the reaper rely on it.
    """
    if status.status not in jobs.TERMINAL_STATUSES:
        return
    record = jobs.JobRecord.load(job_id)
    if record is None:
        return
    # Copy logs now, since the pod may not last. `save_logs` keeps only
    # the first copy.
    if not status.details.get("never_started"):
        save_logs(kube, config, storage, job_id)
    if record.status not in jobs.TERMINAL_STATUSES:
        mark_terminal(record, status)
