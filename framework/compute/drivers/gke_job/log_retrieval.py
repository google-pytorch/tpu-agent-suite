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
"""Retrieving a `gke_job` Job's logs from its pod, a copy or Cloud Logging."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

from framework import jsonio

from .. import base, gcs_staging, jobs
from ..kubectl import KUBECTL_READ_TIMEOUT_SEC
from .gke_common import parse_time, rfc3339
from .pod_diagnostics import failed, first_failures, node_lost
from .status import job_pods, live_status, pick_live_pod

if TYPE_CHECKING:
    from ..kubectl import Kubectl
    from .gke_common import Path, StagingClient
    from .profile_settings import GkeJobConfig

# Limits on the Cloud Logging fallback, so a chatty Job can't make one
# `job logs` take minutes.
CLOUD_LOGGING_MAX_LINES = 2000
CLOUD_LOGGING_TIMEOUT_SEC = 60
# Margin around the record's start and end times, which only approximate
# when the container ran. Applied only when the query has the Job's UID.
CLOUD_LOGGING_SLACK_SEC = 600
# The Job UID pod label as Cloud Logging names it (it turns dots into
# underscores). Kubernetes 1.27+ sets both keys.
CLOUD_LOGGING_UID_LABELS = ("k8s-pod/controller-uid",
                            "k8s-pod/batch_kubernetes_io/controller-uid")

# Wait this long after a node loss before asking Cloud Logging for the
# missing lines. Ingestion can lag up to a minute.
CLOUD_LOGGING_SETTLE_SEC = 90


def _with_cloud_tail(log: str, cloud: list[str]) -> tuple[str, int]:
    """Add the lines Cloud Logging has after the end of `log`.

    Returns the new log and how many lines were added. The pod's own copy
    is kept whole, since Cloud Logging's is capped. If `log`'s last lines
    can't be found in `cloud`, returns `log` unchanged.
    """
    lines = log.splitlines()
    if log and not log.endswith("\n"):
        lines = lines[:-1]  # Cut mid-line when the node went.
    tail = [line for line in lines if line.strip()][-3:]
    cloud = [line for line in cloud if line.strip()]
    if not tail:
        extra, lines = cloud, []
    else:
        for end in range(len(cloud), len(tail) - 1, -1):
            if cloud[end - len(tail):end] == tail:
                extra = cloud[end:]
                break
        else:
            return log, 0
    if not extra:
        return log, 0
    head = "".join(line + "\n" for line in lines)
    return (head + f"[compute] The node was lost, and its copy of this log "
            f"ends here; the {len(extra)} line(s) below are what Cloud "
            "Logging kept after it.\n" + "\n".join(extra) + "\n", len(extra))


def _cloud_logging_query(namespace: str,
                         job_id: str,
                         cluster: str | None,
                         start: float | None,
                         end: float | None,
                         uid: str | None = None) -> str:
    """Build the Cloud Logging filter for a Job's container output.

    With `uid`, the time bounds widen by CLOUD_LOGGING_SLACK_SEC. Without
    it, they stay exact so another Job with the same name isn't matched.
    """
    filters = [
        'resource.type="k8s_container"',
        f'resource.labels.namespace_name="{namespace}"',
        f'labels."k8s-pod/job-name"="{job_id}"',
    ]
    if uid:
        filters.append("(" +
                       " OR ".join(f'labels."{label}"="{uid}"'
                                   for label in CLOUD_LOGGING_UID_LABELS) +
                       ")")
    if cluster:
        filters.append(f'resource.labels.cluster_name="{cluster}"')
    slack = CLOUD_LOGGING_SLACK_SEC if uid else 0
    if start is not None:
        filters.append(f'timestamp>="{rfc3339(start - slack)}"')
    if end is not None:
        filters.append(f'timestamp<="{rfc3339(end + slack)}"')
    return " AND ".join(filters)


def _log_entry_text(entry: dict[str, Any]) -> str | None:
    """Return the printed line from a Cloud Logging entry, or None."""
    if isinstance(entry.get("textPayload"), str):
        return entry["textPayload"].rstrip("\n")
    payload = entry.get("jsonPayload")
    if isinstance(payload, dict):
        message = payload.get("message")
        return message.rstrip("\n") if isinstance(
            message, str) else json.dumps(payload, sort_keys=True)
    return None


def _log_entry_pod(entry: dict[str, Any]) -> str | None:
    return ((entry.get("resource") or {}).get("labels") or {}).get("pod_name")


def read_logs(kube: Kubectl, config: GkeJobConfig, storage: StagingClient,
              job_id: str) -> str:
    """Return the Job's pod output, or "" if it hasn't started.

    Prefers the `save_logs` copy, since the pod is GC'd once its TPU node
    scales away. Then tries the pod, the GCS copy and Cloud Logging, and
    raises if none has it.
    """
    saved = saved_log(job_id)
    if not saved.is_file():
        record = jobs.JobRecord.load(job_id)
        if record is not None and record.extra.get("never_started"):
            # No container ever ran, so don't point at Cloud Logging.
            raise base.ComputeConfigError(
                f"Job {job_id} never started, so it has no logs: "
                f"{record.extra.get('error') or 'its image never pulled'}",
                error_code="JOB_NEVER_STARTED",
            )
        proc = pod_logs(kube, job_id)
        if proc.returncode == 0:
            if not node_lost(job_pods(kube, job_id)):
                return proc.stdout
            # The node is gone, so this log is final. Save it.
            _keep_lost_node_log(job_id, proc.stdout)
        else:
            download_log(config, storage, job_id, saved)
    if saved.is_file():
        if lost_node_marker(job_id).is_file():
            return _settle_lost_node_log(kube, job_id)
        return saved.read_text(encoding="utf-8")
    try:
        live = live_status(kube, job_id).status
    except base.ComputeConfigError:
        live = None  # The Job itself is gone.
    if live == "PENDING":
        return ""
    record = jobs.JobRecord.load(job_id)
    why = ("It was cancelled before it finished, and cancelling deletes "
           "its pod"
           if record is not None and record.status == "CANCELLED" else
           "Its pod could not be read (reaped after its node scaled away, "
           "evicted, expired, or deleted outside compute)")
    logs, why_not = _cloud_logging_logs(kube, job_id, record)
    if logs is not None:
        return logs
    raise base.ComputeConfigError(
        f"Logs for Job {job_id} are unavailable: "
        f"{proc.stderr.strip()}. {why}, and no copy was "
        "saved locally or in the staging bucket. Cloud Logging: "
        f"{why_not}",
        error_code="LOGS_UNAVAILABLE",
    )


def _cloud_logging_logs(
        kube: Kubectl, job_id: str,
        record: jobs.JobRecord | None) -> tuple[str | None, str]:
    """Return Cloud Logging's copy of the output, or (None, why not).

    Never raises.
    """
    lines, where = _cloud_logging_lines(kube, job_id, record)
    if lines is None:
        return None, where
    header = (f"[compute] Job {job_id}'s pod is gone and no copy of its "
              f"log was saved: the {len(lines)} line(s) below are what "
              f"Cloud Logging kept ({where})")
    return header + ".\n" + "\n".join(lines) + "\n", ""


def _cloud_logging_lines(
        kube: Kubectl, job_id: str,
        record: jobs.JobRecord | None) -> tuple[list[str] | None, str]:
    """Return (lines oldest first, source) or (None, why not).

    Never raises.
    """
    target = base.gke_target(base.current_context(kube.cmd([])),
                             prefer_context=True)
    known = {
        part: value
        for part, value in target.items() if not value.startswith("<")
    }
    start = parse_time(record.start_time) if record else None
    end = parse_time(record.end_time) if record else None
    query = _cloud_logging_query(
        kube.namespace,
        job_id,
        known.get("cluster"),
        start,
        end,
        uid=record.extra.get("uid") if record else None)
    if not shutil.which("gcloud"):
        return None, ("gcloud is not on PATH to read it. Install the "
                      "Google Cloud CLI, or run this query in the Logs "
                      f"Explorer: {query}")
    argv = [
        "gcloud", "logging", "read", query,
        f"--limit={CLOUD_LOGGING_MAX_LINES}", "--order=desc", "--format=json"
    ]
    if start is None:
        # gcloud reads only the last day unless the query bounds it.
        argv.append("--freshness=30d")
    if "project" in known:
        argv.append(f"--project={known['project']}")
    proc = base.probe_run(argv, timeout=CLOUD_LOGGING_TIMEOUT_SEC)
    if proc.returncode != 0:
        return None, (f"`gcloud logging read` failed: "
                      f"{base.probe_error(proc)}. Query: {query}")
    try:
        entries = json.loads(proc.stdout or "[]")
    except ValueError:
        entries = None
    if not isinstance(entries, list):
        return None, ("`gcloud logging read` returned output that is not "
                      f"a list of entries. Query: {query}")
    # Fetched newest first so the limit keeps the end, then reversed.
    entries = [entry for entry in reversed(entries) if isinstance(entry, dict)]
    pods = {_log_entry_pod(entry) for entry in entries} - {None}
    lines = []
    for entry in entries:
        text = _log_entry_text(entry)
        if text is None:
            continue
        pod = _log_entry_pod(entry)
        # A Job that retried ran several pods; say whose line it is.
        lines.append(f"[{pod}] {text}" if len(pods) > 1 and pod else text)
    if not lines:
        return None, ("it holds no output for the Job: container logging "
                      "may be off for the cluster, the entries expired, "
                      f"or they are not ingested yet. Query: {query}")
    where = (f"project {known['project']}"
             if "project" in known else "gcloud's default project")
    if len(entries) >= CLOUD_LOGGING_MAX_LINES:
        where += f", the last {CLOUD_LOGGING_MAX_LINES} only"
    return lines, where


def _keep_lost_node_log(job_id: str, log: str) -> None:
    """Save a lost node's log and mark it for `read_logs` to complete."""
    # A marker file, since callers may overwrite the record. Written
    # first so the copy never looks final without it.
    lost_node_marker(job_id).touch()
    jsonio.write_text_atomic(saved_log(job_id), log)


def _settle_lost_node_log(kube: Kubectl, job_id: str) -> str:
    """Complete a lost node's log from Cloud Logging and save it.

    Waits out CLOUD_LOGGING_SETTLE_SEC first. If Cloud Logging fails,
    returns the copy as is and retries on the next read.
    """
    saved, marker = saved_log(job_id), lost_node_marker(job_id)
    text = saved.read_text(encoding="utf-8")
    wait = marker.stat().st_mtime + CLOUD_LOGGING_SETTLE_SEC - time.time()
    if wait > 0:
        print(
            f"gke_job: {job_id}'s node was lost, so its log on the node "
            f"misses its last lines; waiting {wait:.0f}s for Cloud "
            "Logging to have them.",
            file=sys.stderr,
            flush=True)
        time.sleep(wait)
    lines, why_not = _cloud_logging_lines(kube, job_id,
                                          jobs.JobRecord.load(job_id))
    if lines is None:
        print(
            f"gke_job: {job_id}'s log is the lost node's copy, which may "
            f"miss its last lines. Cloud Logging: {why_not}",
            file=sys.stderr)
        return text
    text = _with_cloud_tail(text, lines)[0]
    jsonio.write_text_atomic(saved, text)
    marker.unlink(missing_ok=True)
    return text


def save_logs(kube: Kubectl, config: GkeJobConfig, storage: StagingClient,
              job_id: str) -> None:
    """Copy a finished Job's logs beside its record, once."""
    saved = saved_log(job_id)
    if saved.is_file():
        return
    # Check for pods first: `kubectl logs` on a podless Job blocks ~20s.
    pods = job_pods(kube, job_id)
    if pods:
        proc = pod_logs(kube, job_id, pods)
        if proc.returncode == 0:
            if node_lost(pods):
                _keep_lost_node_log(job_id, proc.stdout)
            else:
                jsonio.write_text_atomic(saved, proc.stdout)
            return
    download_log(config, storage, job_id, saved)


def pod_logs(
    kube: Kubectl,
    job_id: str,
    pods: list[dict[str, Any]] | None = None,
) -> subprocess.CompletedProcess[str]:
    """`kubectl logs` of the Job, unchecked.

    For a multi-host Job, the logs of the first pod to fail if one did, else
    the leader's.
    """
    target = f"job/{job_id}"
    record = jobs.JobRecord.load(job_id)
    if (record.extra if record else {}).get("hosts", 1) > 1:
        first = first_failures(
            job_pods(kube, job_id) if pods is None else pods)
        pod = first[0] if first and failed(first[0]) else pick_live_pod(first)
        if pod and pod.get("metadata", {}).get("name"):
            target = f"pod/{pod['metadata']['name']}"
    return kube.run(["logs", target] + kube.ns(),
                    check=False,
                    timeout=KUBECTL_READ_TIMEOUT_SEC)


def saved_log(job_id: str) -> Path:
    """Where `save_logs` keeps a finished Job's log."""
    return jobs.jobs_dir() / job_id / "output.log"


def lost_node_marker(job_id: str) -> Path:
    """Marker for a saved log still missing its Cloud Logging tail."""
    return jobs.jobs_dir() / job_id / "output.log.node-lost"


def log_blob(job_id: str) -> str:
    """The staging-bucket object the pod uploads its log copy to."""
    return f"{job_id}/out/{gcs_staging.POD_LOG_FILE}"


def download_log(config: GkeJobConfig, storage: StagingClient, job_id: str,
                 dest: Path) -> None:
    """Fetch the log the pod uploaded, if a staging bucket holds one.

    Never raises, so a status read or cancel can't fail over a log copy.
    Downloads to a `.part` file so a partial `dest` never exists.
    """
    if not config.staging_bucket:
        return
    part = dest.with_name(dest.name + ".part")
    try:
        if gcs_staging.download_file_from_gcs(
                gcs_staging.clean_bucket_name(config.staging_bucket),
                log_blob(job_id), part, **storage.client_kwargs()):
            part.replace(dest)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        print(
            f"gke_job: could not fetch the log of {job_id} from GCS "
            f"-- {exc}",
            file=sys.stderr)
    finally:
        part.unlink(missing_ok=True)
