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
"""Why Kueue has not admitted a `gke_job` Job, and whether it ever will."""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

from ..kubectl import KUBECTL_READ_TIMEOUT_SEC, KubectlError
from .gke_common import PRIORITY_LABEL, parse_time

if TYPE_CHECKING:
    from ..kubectl import Kubectl

# Kueue tracks each Job through a Workload labelled with the Job's uid. Only
# the Workload tells a refused Job from one queued behind a full cluster.
WORKLOAD_RESOURCE = "workloads.kueue.x-k8s.io"
JOB_UID_LABEL = "kueue.x-k8s.io/job-uid"
# No Workload this long after creation means Kueue couldn't build one
# (usually a missing WorkloadPriorityClass). The Job stays suspended forever.
KUEUE_WORKLOAD_GRACE_SEC = 60
# QuotaReserved=False reasons that waiting won't fix. Queue holds
# (`Suspended`), `Pending` and `WaitingForQuota` are left out.
KUEUE_REFUSALS = frozenset({
    "Inadmissible",
    "Misconfigured",
    "NoMatchingFlavor",
    "ExceedsMaxQuota",
})
# Queue problems the scheduler may report under plain `Pending`. A held
# queue also reads "is inactive"; see `_queue_held`.
KUEUE_QUEUE_PROBLEM = re.compile(
    r"\b(LocalQueue|ClusterQueue) (\S+) (doesn't exist|is inactive|"
    r"is terminating or missing)")
# Set by a time-limit controller that stopped a Workload for running too long.
# Its `finished-gc` reason is cleanup of a finished Workload, not a kill.
EVICTION_REASON_ANNOTATION = "quota-system/eviction-reason"
EVICTION_MESSAGE_ANNOTATION = "quota-system/eviction-message"
EVICTED_AT_ANNOTATION = "quota-system/evicted-at"
FINISHED_GC_REASON = "finished-gc"


def _time_limit_stop(workload: dict[str, Any]) -> dict[str, str] | None:
    """Eviction details if the time-limit controller stopped it, else None."""
    annotations = workload.get("metadata", {}).get("annotations") or {}
    reason = annotations.get(EVICTION_REASON_ANNOTATION)
    if (workload.get("spec", {}).get("active") is not False or not reason
            or reason == FINISHED_GC_REASON):
        return None
    message = (annotations.get(EVICTION_MESSAGE_ANNOTATION)
               or f"evicted ({reason})")
    stop = {"eviction_reason": reason, "eviction_message": message}
    if annotations.get(EVICTED_AT_ANNOTATION):
        stop["evicted_at"] = annotations[EVICTED_AT_ANNOTATION]
    return stop


def _workload_verdict(
        workload: dict[str, Any]) -> tuple[bool, str, str] | None:
    """(refused, reason, message) for a Workload Kueue has not admitted.

    Refused means Kueue will never admit it as written. Not refused means
    waiting for quota or requeued after preemption. None means admitted
    or not yet looked at.
    """
    conditions = {
        cond.get("type"): cond
        for cond in workload.get("status", {}).get("conditions", [])
    }
    finished = conditions.get("Finished") or {}
    if finished.get("status") == "True":
        # Finished while the Job is suspended: never admitted.
        return (True, finished.get("reason")
                or "Finished", finished.get("message")
                or "Kueue finished the Workload without admitting it.")
    if workload.get("spec", {}).get("active") is False:
        # Kueue won't requeue it until reactivated. The cause decides which
        # condition explains it.
        for kind in ("DeactivationTarget", "Evicted", "Requeued"):
            cond = conditions.get(kind) or {}
            if cond.get("message"):
                return (True, cond.get("reason")
                        or "Deactivated", cond["message"])
        return True, "Deactivated", "Kueue deactivated the Workload."
    # Older Kueue records preemption only as the eviction's reason.
    for kind in ("Preempted", "Evicted"):
        cond = conditions.get(kind) or {}
        if cond.get("status") == "True" and "Preempted" in (
                kind, cond.get("reason")):
            return False, "Preempted", cond.get("message") or "Preempted."
    reserved = conditions.get("QuotaReserved") or {}
    if reserved.get("status") != "False":
        return None
    reason = reserved.get("reason") or ""
    message = reserved.get("message") or ""
    if reason in KUEUE_REFUSALS or KUEUE_QUEUE_PROBLEM.search(message):
        return True, reason or "Inadmissible", message or reason
    return (False, reason or "Pending", message) if message else None


def _event_time(event: dict[str, Any]) -> float:
    """When an event last happened, for ordering; 0 when it does not say."""
    return parse_time(
        event.get("lastTimestamp") or
        (event.get("series") or {}).get("lastObservedTime")
        or event.get("eventTime") or event.get("firstTimestamp")) or 0.0


def kueue_admission(kube: Kubectl, job_id: str,
                    job: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Whether Kueue has refused a suspended Job for good, and why.

    Returns (refused, details) with `kueue_*` keys. A failed read refuses
    nothing, so the Job stays PENDING. Waits are not refused, but Kueue's
    message is still returned.
    """
    metadata = job.get("metadata", {})
    uid = metadata.get("uid")
    # Jobs owned by a JobSet, LWS or Pathways have no Workload of their own.
    if not uid or any(
            ref.get("controller")
            for ref in metadata.get("ownerReferences", [])):
        return False, {}
    workloads = _workloads(kube, uid)
    if workloads == []:
        created = parse_time(metadata.get("creationTimestamp"))
        if (created is None
                or time.time() - created < KUEUE_WORKLOAD_GRACE_SEC):
            return False, {}
        # Older Kueue doesn't set the uid label. Fall back to owner refs.
        workloads = _workloads(kube, uid, labelled=False)
        if workloads == []:
            return True, _no_workload(kube, job_id, metadata)
    waiting: dict[str, Any] = {}
    for workload in workloads or []:
        stop = _time_limit_stop(workload)
        if stop:
            return True, stop
        verdict = _workload_verdict(workload)
        if verdict is None:
            continue
        refused, reason, message = verdict
        if refused and _queue_held(kube, message):
            refused, reason = False, "Stopped"
        name = workload.get("metadata", {}).get("name")
        details = {"kueue_reason": reason, "kueue_message": message}
        if name:
            details["kueue_workload"] = name
        if not refused:
            waiting = waiting or details
            continue
        events = _warning_events(kube, "Workload", name) if name else []
        if events:
            details["kueue_events"] = events
        return True, details
    return False, waiting


def _queue_held(kube: Kubectl, message: str) -> bool:
    """Whether an "inactive" queue named in `message` is only paused.

    Kueue also calls a queue paused by its `stopPolicy` "inactive", but the
    pause can be lifted. A paused queue's Active condition has reason
    `Stopped`. If the queue can't be read, assume it's paused.
    """
    match = KUEUE_QUEUE_PROBLEM.search(message)
    if not match or match.group(3) != "is inactive":
        return False
    kind, name = match.group(1), match.group(2)
    queue = kube.get_json_or_none(
        ["get", f"{kind.lower()}s.kueue.x-k8s.io", name] +
        (kube.ns() if kind == "LocalQueue" else []))
    return queue is None or any(
        cond.get("type") == "Active" and cond.get("reason") == "Stopped"
        for cond in queue.get("status", {}).get("conditions", []))


def _no_workload(kube: Kubectl, job_id: str,
                 metadata: dict[str, Any]) -> dict[str, Any]:
    """Why a Job with no Workload will never be admitted.

    Uses Warning events on the Job when newer Kueue emits them. Otherwise
    blames the priority class only if the cluster confirms it's missing.
    """
    details: dict[str, Any] = {"kueue_reason": "NoWorkload"}
    events = _warning_events(kube, "Job", metadata.get("name") or job_id)
    if events:
        details.update(kueue_message=events[0], kueue_events=events)
        return details
    priority = metadata.get("labels", {}).get(PRIORITY_LABEL)
    cause = ("Kueue could not build one for it, and says why only in the "
             "kueue-controller-manager log (namespace kueue-system).")
    if priority:
        try:
            proc = kube.run(
                ["get", "workloadpriorityclasses.kueue.x-k8s.io", priority],
                check=False,
                timeout=KUBECTL_READ_TIMEOUT_SEC)
        except KubectlError:
            proc = None
        if proc is not None and "NotFound" in proc.stderr:
            cause = (f"The Job names the WorkloadPriorityClass "
                     f"'{priority}' ({PRIORITY_LABEL}), which the cluster "
                     "does not define -- `kubectl get "
                     "workloadpriorityclasses` lists those it does.")
    details["kueue_message"] = (
        f"no Kueue Workload was created for the Job within "
        f"{KUEUE_WORKLOAD_GRACE_SEC}s, so it will never be admitted. "
        f"{cause}")
    return details


def _workloads(kube: Kubectl,
               uid: str,
               labelled: bool = True) -> list[dict[str, Any]] | None:
    """The Kueue Workloads of the Job with `uid`, or None if unreadable.

    None (no CRD, no rights, no answer) does not mean "no Workload".
    Without `labelled`, all Workloads are matched by owner reference.
    """
    argv = ["get", WORKLOAD_RESOURCE] + kube.ns()
    if labelled:
        argv += ["-l", f"{JOB_UID_LABEL}={uid}"]
    items = (kube.get_json_or_none(argv) or {}).get("items")
    if not isinstance(items, list):
        return None
    if labelled:
        return items
    return [
        workload for workload in items if any(
            ref.get("uid") == uid
            for ref in workload.get("metadata", {}).get("ownerReferences", []))
    ]


def _warning_events(kube: Kubectl, kind: str, name: str) -> list[str]:
    """Up to five Warning events as `Reason: message`, newest first."""
    events = (kube.get_json_or_none(["get", "events"] + kube.ns() + [
        "--field-selector",
        f"involvedObject.kind={kind},involvedObject.name={name}"
    ]) or {}).get("items") or []
    seen: list[str] = []
    for event in sorted(events, key=_event_time, reverse=True):
        if event.get("type") != "Warning" or not event.get("message"):
            continue
        line = f"{event.get('reason') or 'Warning'}: {event['message']}"
        if line not in seen:
            seen.append(line)
    return seen[:5]
