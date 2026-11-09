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
"""Read a `gke_job` Job's status from the cluster, and wait for it."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Callable

from .. import base, jobs
from ..base import TPU_RESOURCE
from ..kubectl import KUBECTL_READ_TIMEOUT_SEC
from .gke_common import (POLL_INTERVAL_SEC, QUEUE_LABEL, is_not_found,
                         parse_time, pod_index)
from .kueue_admission import kueue_admission
from .pod_diagnostics import (failed, first_failures, image_pull_problem,
                              pod_end, termination_error)

if TYPE_CHECKING:
    from ..kubectl import Kubectl

# A Job read right after it is created can come back NotFound. A Job whose
# record is younger than FRESH_JOB_SEC is read again after each delay.
FRESH_JOB_SEC = 60
NOT_FOUND_RETRY_DELAYS_SEC = (1, 2, 4)

# `wait`'s latest poll: (status, seconds since submit, ever RUNNING).
LastPoll = tuple[base.AsynchronousExecutionResult, int, bool]


def _just_submitted(record: jobs.JobRecord | None) -> bool:
    """Whether a live record is young enough that NotFound may be a lag."""
    if record is None or record.status in jobs.TERMINAL_STATUSES:
        return False
    started = parse_time(record.start_time)
    return started is not None and time.time() - started < FRESH_JOB_SEC


def live_status_retrying_new_job(
        kube: Kubectl, job_id: str) -> base.AsynchronousExecutionResult:
    """`live_status`, retrying NotFound while the Job is newly created."""
    delays = list(NOT_FOUND_RETRY_DELAYS_SEC)
    while True:
        try:
            return live_status(kube, job_id)
        except base.ComputeConfigError as exc:
            if not (delays and is_not_found(exc)
                    and _just_submitted(jobs.JobRecord.load(job_id))):
                raise
            time.sleep(delays.pop(0))


def live_status(kube: Kubectl,
                job_id: str) -> base.AsynchronousExecutionResult:
    """The driver's `get_status` from the cluster, without the record write."""
    job = kube.get_json(["get", "job", job_id] + kube.ns())
    spec = job.get("spec", {})
    job_status = job.get("status", {})
    details: dict[str, Any] = {"namespace": kube.namespace}
    queued = bool(job.get("metadata", {}).get("labels", {}).get(QUEUE_LABEL))

    pods = job_pods(kube, job_id)
    live_pod = pick_live_pod(pods)
    if live_pod:
        pod_name = live_pod.get("metadata", {}).get("name")
        if pod_name:
            details["pod_name"] = pod_name
        node_name = live_pod.get("spec", {}).get("nodeName")
        if node_name:
            details["node_name"] = node_name

    start_time = job_status.get("startTime")
    end_time = job_status.get("completionTime")
    # A multi-host Job is one pod per node, and is done when all are.
    hosts = (int(spec.get("completions") or 1)
             if spec.get("completionMode") == "Indexed" else 1)
    if hosts > 1:
        details["hosts"] = hosts

    if job_status.get("succeeded", 0) >= hosts:
        return base.AsynchronousExecutionResult(
            job_id=job_id,
            status="COMPLETED",
            start_time=start_time,
            end_time=end_time,
            exit_code=0,
            details=details,
        )

    if job_status.get("failed", 0):
        pods = first_failures(pods)
        end = pod_end(pods)
        if end.pod_reason:
            details["pod_reason"] = end.pod_reason
        error = termination_error(**end._asdict())
        if hosts > 1 and pods and failed(pods[0]):
            # The other hosts are stopped with it; theirs isn't the failure.
            details["failed_host"] = pod_index(pods[0])
            error = f"host {details['failed_host']} of {hosts}: {error}"
        return base.AsynchronousExecutionResult(
            job_id=job_id,
            status="FAILED",
            start_time=start_time,
            end_time=end_time,
            exit_code=end.exit_code,
            error=error,
            details=details,
        )

    if spec.get("suspend"):
        refused, kueue = kueue_admission(kube, job_id, job)
        details.update(kueue)
        if kueue.get("eviction_reason"):
            # The time limit stopped it after it ran. Keep the Job for logs.
            return base.AsynchronousExecutionResult(
                job_id=job_id,
                status="FAILED",
                start_time=start_time,
                error=("The cluster stopped the Job after it ran: "
                       f"{kueue['eviction_message']}. Kueue will not "
                       "requeue it; split the work into shorter jobs or "
                       "resume from a checkpoint."),
                details=details,
            )
        if refused:
            # Waiting won't help. `get_status` deletes the Job so it can't
            # run unwatched once the cluster is fixed.
            details["never_started"] = True
            return base.AsynchronousExecutionResult(
                job_id=job_id,
                status="FAILED",
                start_time=start_time,
                error=(f"Kueue will not admit the Job "
                       f"({kueue['kueue_reason']}): "
                       f"{kueue['kueue_message']}"),
                details=details,
            )
        details["suspended"] = True
        details["pending_reason"] = base.PENDING_QUEUED
        error = ("Kueue preempted the Job to make room for another "
                 "workload and requeued it; it runs again when quota "
                 "frees up." if kueue.get("kueue_reason") == "Preempted" else
                 "Job is suspended: the cluster admits work through "
                 "a queue and has not admitted this one. Free chips, not "
                 "configuration, is the usual cause.")
        if kueue.get("kueue_message"):
            error += f" Kueue says: {kueue['kueue_message']}"
        return base.AsynchronousExecutionResult(
            job_id=job_id,
            status="PENDING",
            start_time=start_time,
            error=error,
            details=details,
        )

    if job_status.get("active", 0):
        # `active` also counts pending pods, including ones that will never
        # schedule. Check the pod phase to tell them from running ones.
        pull_error, final = image_pull_problem(kube, pods)
        if final:
            # Still live and holding a TPU node: `get_status` deletes it.
            details["pod_reason"] = pull_error
            details["never_started"] = True
            return base.AsynchronousExecutionResult(
                job_id=job_id,
                status="FAILED",
                start_time=start_time,
                error=f"The image could not be pulled: {pull_error}",
                details=details,
            )
        phase, reason = _pod_phase(kube, job_id, pods)
        # A scheduled pod with a retrying pull: the pull is the reason.
        reason = reason or pull_error
        lag = _lagging_host(pods, hosts) if hosts > 1 else None
        if phase == "Running" and lag:
            # The leader running isn't the slice running: libtpu waits for
            # every host, so nothing has started yet.
            index, pod = lag
            details["lagging_host"] = index
            phase = "Pending"
            node = (pod or {}).get("spec", {}).get("nodeName")
            if node:
                details["node_name"] = node
            else:
                details.pop("node_name", None)
            reason = (f"host {index} of {hosts} is not running yet" +
                      (f": {_pod_phase(kube, job_id, [pod])[1]}"
                       if pod and not node else "."))
        if phase == "Pending":
            details["pod_phase"] = phase
            if reason:
                details["pod_reason"] = reason
            # On a node: starting. Off one, with a queue: quota held but
            # no node. Without a queue: unscheduled.
            details["pending_reason"] = (base.PENDING_STARTING
                                         if "node_name" in details else
                                         base.PENDING_ADMITTED_NO_NODE if
                                         queued else base.PENDING_UNSCHEDULED)
            if not reason:
                reason = (f"Pod is on node {details['node_name']} but not yet "
                          "running: the image is still pulling or the "
                          "container is starting." if "node_name" in details
                          else "Pod is not scheduled yet: no node matches the "
                          "selector, or the chips are taken.")
            return base.AsynchronousExecutionResult(
                job_id=job_id,
                status="PENDING",
                start_time=start_time,
                error=reason,
                details=details,
            )
        return base.AsynchronousExecutionResult(
            job_id=job_id,
            status="RUNNING",
            start_time=start_time,
            details=details,
        )

    return base.AsynchronousExecutionResult(
        job_id=job_id,
        status="PENDING",
        start_time=start_time,
        details=details,
    )


def job_pods(kube: Kubectl, job_id: str) -> list[dict[str, Any]]:
    """The Job's pods, or an empty list if they cannot be read."""
    proc = kube.run(
        ["get", "pods"] + kube.ns() +
        ["-l", f"job-name={job_id}", "-o", "json"],
        check=False,
        timeout=KUBECTL_READ_TIMEOUT_SEC,
    )
    if proc.returncode != 0:
        return []
    try:
        return json.loads(proc.stdout).get("items", [])
    except ValueError:
        return []


def pick_live_pod(pods: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The pod to read: Running, else Pending, else the first one. In a
    multi-host Job, the leader's (index 0), which serves and prints the
    result."""
    ranked = {"Running": 0, "Pending": 1}
    return min(
        pods,
        key=lambda p:
        (pod_index(p) not in
         (None, 0), ranked.get(p.get("status", {}).get("phase"), 2)),
        default=None,
    )


def _lagging_host(pods: list[dict[str, Any]],
                  hosts: int) -> tuple[int, dict[str, Any] | None] | None:
    """The first of `hosts` indexes with no pod Running (or done), and its
    pod if it has one; None when every host is up."""
    up = {
        pod_index(pod)
        for pod in pods
        if pod.get("status", {}).get("phase") in ("Running", "Succeeded")
    }
    for index in range(hosts):
        if index not in up:
            return index, next(
                (pod for pod in pods if pod_index(pod) == index), None)
    return None


def _pod_phase(
    kube: Kubectl,
    job_id: str,
    pods: list[dict[str, Any]] | None = None,
) -> tuple[str | None, str | None]:
    """The pod's phase and, if unscheduled, the scheduler's reason."""
    pod = pick_live_pod(job_pods(kube, job_id) if pods is None else pods)
    if pod is None:
        return None, None
    status = pod.get("status", {})
    reason = None
    for cond in status.get("conditions", []):
        if cond.get("type") == "PodScheduled" and cond.get(
                "status") == "False":
            reason = cond.get("message") or cond.get("reason")
            break
    return status.get("phase"), reason


def chips_in_use(kube: Kubectl) -> tuple[list[str], int | None]:
    """Pods holding TPU chips and the chip total, or `([], None)` if unread."""
    proc = kube.run(
        ["get", "pods", "--all-namespaces", "-o", "json"],
        check=False,
    )
    if proc.returncode != 0:
        return [], None
    try:
        payload = json.loads(proc.stdout)
    except ValueError:
        return [], None

    held: list[str] = []
    total = 0
    for pod in payload.get("items", []):
        phase = pod.get("status", {}).get("phase", "")
        if phase in ("Succeeded", "Failed"):
            continue
        # Only a pod on a node holds chips. Unscheduled ones are waiting.
        if not pod.get("spec", {}).get("nodeName"):
            continue
        requested = 0
        for container in pod.get("spec", {}).get("containers", []):
            limits = container.get("resources", {}).get("limits", {})
            requested += int(limits.get(TPU_RESOURCE, 0) or 0)
        if requested:
            meta = pod.get("metadata", {})
            held.append(f"{meta.get('namespace')}/{meta.get('name')}")
            total += requested
    return held, total


def wait(
        get_status: Callable[[str], base.AsynchronousExecutionResult],
        job_id: str, timeout: int,
        on_poll: Callable[[LastPoll],
                          None]) -> base.AsynchronousExecutionResult:
    """Poll until the Job settles, or return its last state at timeout.

    Polls instead of `kubectl wait` so a failure returns at once.
    `timeout` counts from when the pod starts running. Queue time gets
    its own `timeout` budget. Each poll goes to `on_poll` as a `LastPoll`.
    """
    submitted = time.monotonic()
    started: float | None = None
    status = get_status(job_id)
    while status.status in ("PENDING", "RUNNING"):
        now = time.monotonic()
        if started is None and status.status == "RUNNING":
            started = now
        on_poll((status, round(now - submitted), started is not None))
        # Once started, the run clock keeps going through preemptions.
        elapsed = now - (submitted if started is None else started)
        if elapsed >= timeout:
            queued = (now if started is None else started) - submitted
            return _timed_out(job_id, status, timeout, queued, started
                              is not None)
        time.sleep(POLL_INTERVAL_SEC)
        status = get_status(job_id)
    return status


def _timed_out(job_id: str, status: base.AsynchronousExecutionResult,
               timeout: int, queued: float,
               started: bool) -> base.AsynchronousExecutionResult:
    """Mark `status` timed out, noting queue time and whether it ran."""
    status.details["timed_out_after_sec"] = timeout
    status.details["queued_sec"] = round(queued)
    if started:
        message = (f"Job {job_id} was still {status.status} after "
                   f"running for {timeout}s. It had also waited "
                   f"{round(queued)}s in the queue, which does not count "
                   "toward --timeout.")
    else:
        # No container ran: `run` and `read_logs` skip the pod.
        status.details["never_started"] = True
        message = (f"Job {job_id} was still {status.status} after "
                   f"waiting {timeout}s in the queue. It never ran, so "
                   "no work was lost.")
    status.error = (f"{message} {status.error}" if status.error else message)
    return status


def killed_message(job_id: str, last_poll: LastPoll | None) -> str:
    """One line for a `run` killed from outside, from its last poll."""
    if last_poll is None:
        return (f"gke_job: stopped before Job {job_id}'s first status "
                "read; deleting it.")
    status, since, ever_ran = last_poll
    why = f" {status.error.rstrip('.')}." if status.error else ""
    tail = ("Logs so far are saved for `job logs` if the pod can still be "
            "read." if ever_ran else
            "Nothing ran. A resubmit would wait the same way; check "
            "`compute inspect` first.")
    return (f"gke_job: stopped while Job {job_id} was {status.status}, "
            f"{since}s after submission; deleting it.{why} {tail}")
