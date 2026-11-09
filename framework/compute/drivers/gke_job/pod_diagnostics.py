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
"""Why a `gke_job` pod ended, or cannot start."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, NamedTuple

from .gke_common import parse_time

if TYPE_CHECKING:
    from ..kubectl import Kubectl

# Container waiting reasons for an image that will never pull as written.
IMAGE_REF_FAILURES = ("InvalidImageName", "ErrImageNeverPull")
# Retried pull failures. The kubelet retries a missing image forever while
# the pod holds its chips, so pulls failing this long count as final.
IMAGE_PULL_RETRIES = ("ErrImagePull", "ImagePullBackOff")
PULL_FAILURE_GRACE_SEC = 60


class PodEnd(NamedTuple):
    """How a Job's pod ended. Fields match `termination_error`'s arguments."""
    exit_code: int | None = None
    reason: str | None = None
    pod_reason: str | None = None
    pod_message: str | None = None


# Exit 137 is SIGKILL; with no reason anywhere, these are what send it.
SIGKILL_HINT = "SIGKILL: out of memory, an eviction, or the node going away"


def termination_error(exit_code: int | None,
                      reason: str | None,
                      pod_reason: str | None = None,
                      pod_message: str | None = None) -> str | None:
    """Why a finished container failed, in words a person can act on.

    The generic `Error` reason says little, so the exit code leads and only
    specific reasons are kept. If the kubelet or node ended the pod, as in
    an eviction, that becomes the headline and the exit code the detail.
    """
    if exit_code is None:
        container = reason or None
    else:
        specific = reason if reason not in (None, "", "Error",
                                            "Completed") else None
        container = f"container exited with code {exit_code}"
        if specific:
            container += f" ({specific})"
        elif exit_code == 137 and not (pod_reason or pod_message):
            container += f" ({SIGKILL_HINT})"
    if _is_eviction(pod_reason, pod_message):
        evicted = (f"pod was evicted: {pod_message}"
                   if pod_message else "pod was evicted")
        return f"{evicted} ({container})" if container else evicted
    pod = ": ".join(part for part in (pod_reason, pod_message) if part)
    if not pod:
        return container
    return f"{container}; pod {pod}" if container else f"pod {pod}"


def _is_eviction(pod_reason: str | None, pod_message: str | None) -> bool:
    """Whether the kubelet evicted the pod, by its reason or message."""
    return pod_reason == "Evicted" or (pod_message or "").startswith(
        ("The node was low on resource", "Pod ephemeral local storage usage"))


def _pod_status_reason(pod: dict[str, Any]) -> tuple[str | None, str | None]:
    """The pod's own (reason, message) for ending, or (None, None).

    Falls back to the DisruptionTarget condition set by newer kubelets.
    """
    status = pod.get("status", {})
    if status.get("reason") or status.get("message"):
        return status.get("reason"), status.get("message")
    for cond in status.get("conditions", []):
        if cond.get("type") == "DisruptionTarget" and cond.get(
                "status") == "True":
            return cond.get("reason"), cond.get("message")
    return None, None


def _terminated(pod: dict[str, Any]) -> dict[str, Any]:
    """The first container's `state.terminated` in `pod`, or {}."""
    return next((cs["state"]["terminated"]
                 for cs in pod.get("status", {}).get("containerStatuses", [])
                 if (cs.get("state") or {}).get("terminated")), {})


def failed(pod: dict[str, Any]) -> bool:
    """Whether the pod's container exited non-zero or the kubelet ended it."""
    return (_terminated(pod).get("exitCode") not in (None, 0)
            or any(_pod_status_reason(pod)))


def first_failures(pods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`pods`, with the first to fail ahead of the rest.

    In a multi-host Job the controller stops the other pods after one
    fails, so theirs isn't the failure to read.
    """
    return sorted(pods,
                  key=lambda pod:
                  (not failed(pod), _terminated(pod).get("finishedAt") or "~"))


def pod_end(pods: list[dict[str, Any]]) -> PodEnd:
    """How the Job's pod ended, from the first pod that says (the first to
    fail before any other); else all None."""
    for pod in first_failures(pods):
        terminated = _terminated(pod)
        pod_reason, pod_message = _pod_status_reason(pod)
        if terminated or pod_reason or pod_message:
            return PodEnd(terminated.get("exitCode"), terminated.get("reason"),
                          pod_reason, pod_message)
    return PodEnd()


# Container reasons when the kubelet lost the container's state because the
# node rebooted or went away (seen as exit 255 after a VM reset).
NODE_LOST_REASONS = ("Unknown", "ContainerStatusUnknown")


def node_lost(pods: list[dict[str, Any]]) -> bool:
    """Whether the Job's pod ended because its node was lost.

    The node's log then misses unflushed lines, but Cloud Logging has them.
    """
    return pod_end(pods).reason in NODE_LOST_REASONS


def image_pull_problem(kube: Kubectl,
                       pods: list[dict[str, Any]]) -> tuple[str | None, bool]:
    """Return why an image isn't pulling, and whether to give up on it.

    Gives up on a bad image reference right away, and on a retrying pull
    once its failures span `PULL_FAILURE_GRACE_SEC`. The span runs from the
    first failure to the latest, not to now, since a slow retry may still
    be pulling.
    """
    for pod in pods:
        status = pod.get("status", {})
        for cs in (status.get("initContainerStatuses", []) +
                   status.get("containerStatuses", [])):
            waiting = cs.get("state", {}).get("waiting") or {}
            reason = waiting.get("reason")
            if reason not in IMAGE_REF_FAILURES + IMAGE_PULL_RETRIES:
                continue
            detail = waiting.get("message") or cs.get("image")
            message = f"{reason}: {detail}" if detail else reason
            if reason in IMAGE_REF_FAILURES:
                return message, True
            span = _pull_failure_span(kube,
                                      pod.get("metadata", {}).get("name"))
            return message, (span is not None
                             and span >= PULL_FAILURE_GRACE_SEC)
    return None, False


def _pull_failure_span(kube: Kubectl, pod_name: str | None) -> float | None:
    """Seconds from the pod's first failed pull to its latest, or None.

    Uses the kubelet's per-attempt `Failed to pull image` events, since the
    waiting state has no timestamp. `BackOff` events are skipped. None
    (events unreadable) means the pull is never treated as final.
    """
    if not pod_name:
        return None
    proc = kube.run(
        ["get", "events"] + kube.ns() + [
            "--field-selector",
            f"involvedObject.kind=Pod,involvedObject.name={pod_name}", "-o",
            "json"
        ],
        check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        events = json.loads(proc.stdout).get("items", [])
    except ValueError:
        return None
    failed = [
        event for event in events if event.get("reason") == "Failed" and (
            event.get("message") or "").startswith("Failed to pull image")
    ]
    firsts = [
        parse_time(event.get("firstTimestamp") or event.get("eventTime"))
        for event in failed
    ]
    lasts = [
        parse_time(
            event.get("lastTimestamp")
            or (event.get("series") or {}).get("lastObservedTime")
            or event.get("eventTime")) for event in failed
    ]
    firsts = [stamp for stamp in firsts if stamp is not None]
    lasts = [stamp for stamp in lasts if stamp is not None]
    return max(lasts) - min(firsts) if firsts and lasts else None
