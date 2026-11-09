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
"""Tests for gke_job status: pending, image pulls, Kueue, pod failures."""

from __future__ import annotations

import contextlib
import datetime
import io
import json
import pathlib
import subprocess
import sys
import time
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

# Imports below follow this sys.path setup, so the file runs directly.
# ruff: noqa: E402
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.compute import ComputeConfigError
from framework.compute.drivers import base, gke_job, jobs
from framework.tests.compute.gke_job.helpers import (FakeKubectl,
                                                     GkeJobLifecycleBase,
                                                     failed_pod, job_json,
                                                     job_store, make_driver,
                                                     set_up_module)


def setUpModule():  # pylint: disable=invalid-name
    set_up_module()


class TestGkeJobStatus(GkeJobLifecycleBase):
    """Status mapping of a Job: done, active, pending."""

    def test_active_job_with_unreadable_pods_stays_running(self):
        # No phase to read is not evidence of a stuck pod; the Job is active.
        fake = FakeKubectl({
            "get job": (0, job_json(active=1), ""),
            "get pods": (1, "", "Forbidden"),
        })
        self.assertEqual(self.status(fake).status, "RUNNING")


class TestGkeJobImagePull(GkeJobLifecycleBase):
    """Image pulls that fail, and when that is final."""

    FIRST_FAILURE = "2026-01-01T00:04:00Z"

    @staticmethod
    def _pulling_pods(reason: str) -> str:
        waiting = dict(reason=reason, message="manifest for img:1 not found")
        pod = failed_pod(phase="Pending", node=None)
        pod["status"]["containerStatuses"] = [
            dict(image="img:1", state=dict(waiting=waiting))
        ]
        return json.dumps({"items": [pod]})

    @classmethod
    def _pull_events(cls, span: float) -> str:
        """The kubelet's events for pulls failing from FIRST_FAILURE for
        `span` seconds, and for back-offs that went on for an hour."""
        first = gke_job.gke_common.parse_time(cls.FIRST_FAILURE)

        def stamp(offset: float) -> str:
            return datetime.datetime.fromtimestamp(
                first + offset,
                datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Neither of these is a pull attempt: they repeat while it waits.
        waiting = [("BackOff", 'Back-off pulling image "img:1"'),
                   ("Failed", "Error: ImagePullBackOff")]
        return json.dumps({
            "items": [{
                "reason": "Pulling",
                "message": 'Pulling image "img:1"',
                "firstTimestamp": stamp(-240)
            }, {
                "reason": "Failed",
                "message": 'Failed to pull image "img:1": not found',
                "firstTimestamp": stamp(0),
                "lastTimestamp": stamp(span),
            }] + [{
                "reason": reason,
                "message": message,
                "firstTimestamp": stamp(0),
                "lastTimestamp": stamp(3600),
            } for reason, message in waiting]
        })

    def _pulling(self,
                 reason: str = "ImagePullBackOff",
                 span: float = gke_job.pod_diagnostics.PULL_FAILURE_GRACE_SEC,
                 **responses: Any) -> FakeKubectl:
        return FakeKubectl({
            "get job": (0, job_json(active=1), ""),
            "get pods": (0, self._pulling_pods(reason), ""),
            "get events": (0, self._pull_events(span), ""),
            **responses,
        })

    def test_a_pull_failure_is_final_only_once_it_lasts(self):
        """The waiting reason outlasts the failure while a retry pulls; a
        multi-gigabyte retry after one blip must be left to finish. Without
        readable events (forbidden, or aged out) it is never final."""
        cases = [
            ("ErrImagePull", dict(span=0)),
            ("ImagePullBackOff", dict(span=0)),
            ("ImagePullBackOff", {
                "get events": (1, "", "Forbidden")
            }),
            ("ImagePullBackOff", {
                "get events": (0, '{"items": []}', "")
            }),
        ]
        for reason, kwargs in cases:
            with self.subTest(reason=reason, kwargs=kwargs):
                fake = self._pulling(reason, **kwargs)
                result = self.status(fake)
                self.assertEqual(result.status, "PENDING")
                self.assertIn("manifest for img:1 not found", result.error)
                self.assertIsNone(fake.find("delete job"))

    def test_an_image_reference_that_can_never_pull_fails_at_once(self):
        fake = self._pulling("InvalidImageName")
        self.assertEqual(self.status(fake).status, "FAILED")
        self.assertIsNone(fake.find("get events"))

    def test_a_pull_failure_on_a_submitted_job_is_deleted_and_recorded(self):
        """`job status` settles the record, and the reaper skips a settled
        record: without the delete, the Job would pull forever on a TPU
        node with nothing left to clean it up.

        The status read is the one recorded: reading the Job again to
        delete it could find it vanishing and record a bare CANCELLED.
        Later reads answer from that record, not with a kubectl NotFound,
        and logs say there are none.
        """
        jobs.JobRecord.create(job_id="run-1", backend="gke_job")
        fake = self._pulling()
        self.assertEqual(self.status(fake).status, "FAILED")
        self.assertIn("involvedObject.name=run-1-abcde",
                      " ".join(fake.find("get events")))
        self.assertIsNotNone(fake.find("delete job run-1"))
        self.assertIsNotNone(fake.find("delete configmap run-1"))
        self.assertEqual(
            sum("get job run-1" in " ".join(cmd) for cmd in fake.calls), 1)
        record = jobs.JobRecord.load("run-1")
        self.assertEqual(record.status, "FAILED")
        self.assertTrue(record.extra["never_started"])
        gone = FakeKubectl({"get job": self.GONE})
        driver = make_driver()
        with patch.object(subprocess, "run", gone):
            result = driver.get_status("run-1")
            self.assertEqual(result.status, "FAILED")
            self.assertEqual(
                result.error, "The image could not be pulled: "
                "ImagePullBackOff: manifest for img:1 not found")
            self.assertTrue(result.details["never_started"])
            self.assertTrue(result.details["from_record"])
            with self.assertRaises(ComputeConfigError) as ctx:
                driver.fetch_logs("run-1")
            self.assertEqual(ctx.exception.error_code, "JOB_NEVER_STARTED")
            self.assertIn("manifest for img:1 not found",
                          ctx.exception.message)
        # Nothing is asked of a container that never ran.
        self.assertIsNone(gone.find("logs"))
        self.assertIsNone(gone.find("delete"))

    def test_a_pull_failure_whose_delete_fails_stays_unsettled(self):
        """Left unsettled, the run-end reaper still tries to delete it."""
        record = MagicMock(status="RUNNING")
        fake = self._pulling(**{"delete job": (1, "", "Forbidden")})
        with patch.object(jobs.JobRecord, "load",
                          return_value=record), \
                patch.object(gke_job.records, "save_logs"), \
                contextlib.redirect_stderr(io.StringIO()):
            result = self.status(fake)
        self.assertEqual(result.status, "FAILED")
        record.mark.assert_not_called()


class TestGkeJobMissingJob(GkeJobLifecycleBase):
    """A Job that is no longer on the cluster."""

    def test_a_missing_job_is_only_answered_by_a_settled_record(self):
        driver = make_driver()
        # No record, or one still live: the Job's absence is unexplained.
        for create in (False, True):
            with self.subTest(record=create), job_store(), \
                    patch.object(time, "sleep") as sleep:
                if create:
                    record = jobs.JobRecord.create(job_id="run-1",
                                                   backend="gke_job",
                                                   status="RUNNING")
                    record.start_time = "2026-01-01T00:00:00+00:00"
                    record.save()
                with patch.object(subprocess, "run",
                                  FakeKubectl({"get job": self.GONE})):
                    with self.assertRaises(ComputeConfigError) as ctx:
                        driver.get_status("run-1")
                sleep.assert_not_called()  # Not just submitted: no retry.
                if create:
                    # Removed outside compute: said so, and the record
                    # is kept, since the Job may run on another cluster.
                    self.assertEqual(ctx.exception.error_code, "JOB_GONE")
                    self.assertEqual(
                        jobs.JobRecord.load("run-1").status, "RUNNING")
        # A settled record does not hide a cluster that cannot be asked.
        jobs.JobRecord.create(job_id="run-1",
                              backend="gke_job").mark("COMPLETED", exit_code=0)
        unauthorized = FakeKubectl(
            {"get job": (1, "", "error: You must be logged in to the server")})
        with patch.object(subprocess, "run", unauthorized):
            with self.assertRaises(ComputeConfigError):
                driver.get_status("run-1")

    def test_a_just_submitted_job_is_read_again_on_not_found(self):
        """The read right after submit can race the create."""
        jobs.JobRecord.create(job_id="run-1", backend="gke_job")
        found = FakeKubectl({"get job": (0, job_json(), "")})
        answers = iter([FakeKubectl({"get job": self.GONE})] * 2)

        def kubectl(cmd: list[str], **kwargs: Any):
            if "get job run-1" in " ".join(cmd):
                return next(answers, found)(cmd, **kwargs)
            return found(cmd, **kwargs)

        with patch.object(time, "sleep") as sleep:
            self.assertEqual(self.status(kubectl).status, "PENDING")
        self.assertEqual(sleep.call_count, 2)


class TestGkeJobKueue(GkeJobLifecycleBase):
    """Kueue holding, refusing or stopping a Job."""

    # -- Kueue refusing a suspended Job ------------------------------------

    CREATED = "2026-01-01T00:00:00Z"

    WORKLOAD = "job-run-1-3f2a1"

    LABELLED = f"-l {gke_job.kueue_admission.JOB_UID_LABEL}=uid-1"

    UNLABELLED = f"{gke_job.kueue_admission.WORKLOAD_RESOURCE} -n default -o json"

    NO_LQ = "LocalQueue lq doesn't exist"

    QUOTA_WAIT = ("couldn't assign flavors to pod set main: insufficient "
                  "unused quota for google.com/tpu in flavor tpu7x, 4 more "
                  "needed")

    PREEMPTED = ("Preempted to accommodate a workload (UID: w-2, JobUID: "
                 "uid-2) due to prioritization in the ClusterQueue")

    @classmethod
    def _suspended_job(cls, **metadata: Any) -> str:
        labels = {
            gke_job.gke_common.QUEUE_LABEL: "lq",
            gke_job.gke_common.PRIORITY_LABEL: "medium"
        }
        return json.dumps({
            "metadata":
            dict(name="run-1",
                 uid="uid-1",
                 creationTimestamp=cls.CREATED,
                 labels=labels,
                 **metadata),
            "spec":
            dict(suspend=True),
        })

    @staticmethod
    def _items(*items: dict[str, Any]) -> tuple[int, str, str]:
        return 0, json.dumps({"items": list(items)}), ""

    @staticmethod
    def _cond(kind: str,
              reason: str,
              message: str,
              status: str = "True") -> dict[str, Any]:
        return dict(type=kind, status=status, reason=reason, message=message)

    @classmethod
    def _workload(cls,
                  *conditions: dict[str, Any],
                  owner: str = "uid-1",
                  **spec: Any) -> dict[str, Any]:
        return dict(metadata=dict(
            name=cls.WORKLOAD, ownerReferences=[dict(kind="Job", uid=owner)]),
                    spec=spec,
                    status=dict(conditions=list(conditions)))

    @classmethod
    def _unreserved(cls, reason: str, message: str) -> dict[str, Any]:
        """A Workload Kueue has not reserved quota for, and why."""
        return cls._workload(
            cls._cond("QuotaReserved", reason, message, status="False"))

    @staticmethod
    def _queue(reason: str) -> tuple[int, str, str]:
        """A LocalQueue or ClusterQueue Kueue has made inactive for `reason`."""
        active = {"type": "Active", "status": "False", "reason": reason}
        return 0, json.dumps({"status": {"conditions": [active]}}), ""

    @staticmethod
    def _warnings(*lines: tuple[str, str]) -> tuple[int, str, str]:
        return 0, json.dumps({
            "items": [{
                "type": "Normal",
                "reason": "CreatedWorkload",
                "message": "Created Workload: default/job-run-1-3f2a1",
                "lastTimestamp": "2026-01-01T00:00:01Z",
            }] + [{
                "type": "Warning",
                "reason": reason,
                "message": message,
                "lastTimestamp": f"2026-01-01T00:00:0{i + 2}Z",
            } for i, (reason, message) in enumerate(lines)]
        }), ""

    def _kueue(self,
               *workloads: dict[str, Any],
               job: str | None = None,
               **responses: Any) -> FakeKubectl:
        """A suspended Job in a Kueue queue, its Workloads, and the rest."""
        if workloads:
            responses = {self.LABELLED: self._items(*workloads), **responses}
        return FakeKubectl({
            "get job": (0, job or self._suspended_job(), ""),
            **responses,
        })

    def kueue_status(self, fake: Any, age: float = 3600):
        created = gke_job.gke_common.parse_time(self.CREATED)
        with patch.object(time, "time", return_value=created + age):
            return self.status(fake)

    def test_a_workload_kueue_refuses_fails_the_job_with_kueues_reason(self):
        """An inadmissible Workload is never admitted, however long the wait.

        Polling it burned the whole fifteen minutes to report a "full queue"
        whose real cause was only in the Workload's conditions. It fails at
        once, with Kueue's own words, and is deleted: left queued, it would
        run with nobody watching the day the queue is fixed.
        """
        jobs.JobRecord.create(job_id="run-1", backend="gke_job")
        fake = self._kueue(
            self._unreserved("Inadmissible", self.NO_LQ), **{
                "involvedObject.kind=Workload":
                self._warnings(("Inadmissible", self.NO_LQ))
            })
        result = self.status(fake)
        self.assertEqual(result.status, "FAILED")
        self.assertEqual(
            result.error, f"Kueue will not admit the Job (Inadmissible): "
            f"{self.NO_LQ}")
        self.assertEqual(result.details["kueue_reason"], "Inadmissible")
        self.assertEqual(result.details["kueue_workload"], self.WORKLOAD)
        self.assertEqual(result.details["kueue_events"],
                         [f"Inadmissible: {self.NO_LQ}"])
        self.assertNotIn("suspended", result.details)
        self.assertIn(f"involvedObject.name={self.WORKLOAD}",
                      " ".join(fake.find("involvedObject.kind=Workload")))
        self.assertIsNotNone(fake.find("delete job run-1"))
        record = jobs.JobRecord.load("run-1")
        self.assertEqual(record.status, "FAILED")
        self.assertTrue(record.extra["never_started"])
        # Nothing ran, so there are no logs to save or look for.
        self.assertIsNone(fake.find("logs"))

    def test_every_kind_of_kueue_refusal_fails_the_job(self):
        refusals = {
            "granular reason":
            self._unreserved("Misconfigured", "ClusterQueue cq doesn't exist"),
            # Inactive for want of its flavor, not held: nobody will lift it.
            "inactive queue":
            self._unreserved("Inadmissible", "ClusterQueue cq is inactive"),
            "request no quota can hold":
            self._unreserved("ExceedsMaxQuota", "request > maximum capacity"),
            # The scheduler can report an inactive queue as plain Pending.
            "queue problem under Pending":
            self._unreserved("Pending", "LocalQueue lq is inactive"),
            "deactivated":
            self._workload(self._cond("Evicted", "Deactivated",
                                      "deactivated by an AdmissionCheck"),
                           active=False),
            "finished unadmitted":
            self._workload(
                self._cond("Finished", "AdmissionChecksRejected",
                           "Admission checks rejected")),
        }
        for case, workload in refusals.items():
            with self.subTest(case), job_store():
                result = self.status(
                    self._kueue(
                        workload,
                        **{"queues.kueue": self._queue("FlavorNotFound")}))
                self.assertEqual(result.status, "FAILED")
                self.assertTrue(result.details["never_started"])
                self.assertIn(result.details["kueue_message"], result.error)

    def test_a_job_the_clusters_time_limit_stopped_fails_and_is_kept(self):

        def stopped(reason: str, message: str) -> dict[str, Any]:
            w = self._workload(self._cond("Evicted", "Deactivated",
                                          "Deactivated"),
                               active=False)
            w["metadata"]["annotations"] = {
                gke_job.kueue_admission.EVICTION_REASON_ANNOTATION:
                reason,
                gke_job.kueue_admission.EVICTION_MESSAGE_ANNOTATION:
                message,
                gke_job.kueue_admission.EVICTED_AT_ANNOTATION:
                "2026-01-01T02:00:24Z",
            }
            return w

        msg = "Suspended by time-limit-controller: elapsed 120.4m"
        jobs.JobRecord.create(job_id="run-1", backend="gke_job")
        fake = self._kueue(stopped("timeout", msg))
        result = self.status(fake)
        self.assertEqual(result.status, "FAILED")
        self.assertIn(f"stopped the Job after it ran: {msg}", result.error)
        self.assertEqual(result.details["eviction_reason"], "timeout")
        self.assertNotIn("never_started", result.details)
        self.assertIsNone(fake.find("delete job run-1"))
        self.assertIsNone(
            gke_job.kueue_admission._time_limit_stop(
                stopped("finished-gc", "GC")))

    def test_kueue_waits_that_end_on_their_own_stay_pending(self):
        """No free quota, a queue someone has put on hold, and a preemption
        Kueue requeued are all waits, not refusals. Kueue's message says
        which, and is passed on."""
        held = self._unreserved("Inadmissible", "ClusterQueue cq is inactive")
        stopped = self._queue("Stopped")
        # (case, the Workload, its queue's answer, kueue_reason, in the error)
        waits = [
            ("no quota", self._unreserved("Pending", self.QUOTA_WAIT), None,
             "Pending", self.QUOTA_WAIT),
            ("no quota, newer Kueue",
             self._unreserved("WaitingForQuota", self.QUOTA_WAIT), None,
             "WaitingForQuota", self.QUOTA_WAIT),
            ("held ClusterQueue", held, stopped, "Stopped", "cq is inactive"),
            ("held LocalQueue",
             self._unreserved("Inadmissible", "LocalQueue lq is inactive"),
             stopped, "Stopped", "lq is inactive"),
            # Unreadable, the queue may be held: only an answer fails a Job.
            ("unreadable queue", held, (1, "", "Forbidden"), "Stopped",
             "cq is inactive"),
            ("preempted",
             self._workload(
                 self._cond("Preempted", "InClusterQueue", self.PREEMPTED),
                 self._cond("QuotaReserved", "Pending", self.QUOTA_WAIT,
                            "False")), None, "Preempted",
             f"Kueue preempted the Job to make room for another workload and "
             f"requeued it; it runs again when quota frees up. Kueue says: "
             f"{self.PREEMPTED}"),
            ("preempted, older Kueue",
             self._workload(self._cond("Evicted", "Preempted",
                                       self.PREEMPTED)), None, "Preempted",
             self.PREEMPTED),
            ("not looked at yet", self._workload(), None, None, "Free chips"),
        ]
        for case, workload, queue, reason, said in waits:
            with self.subTest(case):
                fake = self._kueue(
                    workload, **({
                        "queues.kueue": queue
                    } if queue else {}))
                result = self.kueue_status(fake)
                self.assertEqual(result.status, "PENDING")
                self.assertTrue(result.details["suspended"])
                # What `job wait` reads: queued, not lost hardware.
                self.assertEqual(result.details["pending_reason"],
                                 base.PENDING_QUEUED)
                self.assertEqual(result.details.get("kueue_reason"), reason)
                self.assertIn(said, result.error)
                self.assertIsNone(fake.find("delete job"))

    def test_no_workload_after_the_grace_period_fails_the_job(self):
        """Kueue makes no Workload for a Job naming a priority class the
        cluster lacks; that Job stays suspended forever."""
        late = gke_job.kueue_admission.KUEUE_WORKLOAD_GRACE_SEC + 1
        none = {self.LABELLED: self._items(), self.UNLABELLED: self._items()}
        fake = self._kueue(
            **{
                **none,
                # A Workload another Job owns is not this Job's.
                self.UNLABELLED:
                self._items(self._workload(owner="uid-2")),
                "workloadpriorityclasses":
                (1, "", "Error from server (NotFound): "
                 'workloadpriorityclasses.kueue.x-k8s.io "medium" not found'),
            })
        result = self.kueue_status(fake, age=late)
        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.details["kueue_reason"], "NoWorkload")
        self.assertIn("no Kueue Workload was created", result.error)
        self.assertIn("WorkloadPriorityClass 'medium'", result.error)
        # The label is checked by owner too: an older Kueue does not set it.
        self.assertIsNotNone(fake.find(self.UNLABELLED))
        self.assertIsNotNone(fake.find("delete job run-1"))

        # A class the cluster does define is not blamed.
        result = self.kueue_status(self._kueue(**none), age=late)
        self.assertEqual(result.status, "FAILED")
        self.assertNotIn("'medium'", result.error)
        self.assertIn("kueue-controller-manager", result.error)

        # Kueue's own event on the Job, when it records one, says it better.
        found = 'WorkloadPriorityClass "medium" not found'
        fake = self._kueue(
            **{
                **none, "involvedObject.kind=Job":
                self._warnings(("WorkloadPriorityClassNotFound", found))
            })
        result = self.kueue_status(fake, age=late)
        self.assertEqual(result.details["kueue_message"],
                         f"WorkloadPriorityClassNotFound: {found}")

    def test_no_workload_is_pending_within_the_grace_period_or_for_a_child(
            self):
        # Kueue has not reconciled the Job yet: that is not a refusal.
        fake = self._kueue(**{self.LABELLED: self._items()})
        result = self.kueue_status(
            fake, age=gke_job.kueue_admission.KUEUE_WORKLOAD_GRACE_SEC - 1)
        self.assertEqual(result.status, "PENDING")
        self.assertIsNone(fake.find(self.UNLABELLED))
        # A JobSet's Job never has a Workload: Kueue admits the JobSet.
        child = self._suspended_job(ownerReferences=[{
            "kind": "JobSet",
            "uid": "js-1",
            "controller": True
        }])
        fake = self._kueue(job=child, **{self.LABELLED: self._items()})
        self.assertEqual(self.kueue_status(fake).status, "PENDING")
        self.assertIsNone(fake.find(gke_job.kueue_admission.WORKLOAD_RESOURCE))

    def test_an_unreadable_workload_list_stays_pending(self):
        """Only an answer the cluster gave may fail a Job: no Kueue CRD, no
        right to list Workloads, or no answer at all says nothing."""
        unreadable = {
            "no CRD": (1, "", 'error: the server doesn\'t have a resource '
                       'type "workloads"'),
            "forbidden": (1, "", "Error from server (Forbidden)"),
            "garbage": (0, "not json", ""),
        }
        for case, answer in unreadable.items():
            with self.subTest(case):
                fake = self._kueue(**{self.LABELLED: answer})
                result = self.kueue_status(fake)
                self.assertEqual(result.status, "PENDING")
                self.assertTrue(result.details["suspended"])
        # Found by neither label nor owner, but the owner scan failed.
        fake = self._kueue(**{
            self.LABELLED: self._items(),
            self.UNLABELLED: (1, "", "Forbidden")
        })
        self.assertEqual(self.kueue_status(fake).status, "PENDING")
        # A read that times out raises in `_kubectl`; it must not escape.
        answering = self._kueue()

        def hang_on_workloads(cmd: list[str], **kwargs: Any):
            if gke_job.kueue_admission.WORKLOAD_RESOURCE in cmd:
                raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
            return answering(cmd, **kwargs)

        self.assertEqual(
            self.kueue_status(hang_on_workloads).status, "PENDING")


class TestGkeJobFailedPod(GkeJobLifecycleBase):
    """Why a failed Job's pod ended."""

    LOW_DISK = ("The node was low on resource: ephemeral-storage. Threshold "
                "quantity: 10Gi, available: 1Gi.")

    SHUTDOWN = "Pod was terminated in response to imminent node shutdown."

    def test_failed_job_says_why_the_pod_ended(self):
        """Exit 137 and no traceback reads as a crash; the pod's own reason
        (an eviction for a full disk, a node shutdown) is the diagnosis."""
        killed = dict(exit_code=137, container_reason="Error")
        evicted = dict(reason="Evicted", message=self.LOW_DISK)
        disrupted = dict(type="DisruptionTarget",
                         status="True",
                         reason="TerminationByKubelet",
                         message=self.LOW_DISK)
        evicted_137 = (f"pod was evicted: {self.LOW_DISK} "
                       "(container exited with code 137)")
        cases = [
            # (failed_pod arguments, error)
            (dict(killed, **evicted), evicted_137),
            # Evicted before the kubelet recorded the container's end.
            (evicted, f"pod was evicted: {self.LOW_DISK}"),
            # Only the DisruptionTarget condition kept the kubelet's words.
            (dict(killed, conditions=[disrupted]), evicted_137),
            (dict(killed, reason="Terminated", message=self.SHUTDOWN),
             f"container exited with code 137; pod Terminated: {self.SHUTDOWN}"
             ),
            # No reason anywhere: say what sends a SIGKILL.
            (killed, ("container exited with code 137 (SIGKILL: out of "
                      "memory, an eviction, or the node going away)")),
            # A specific reason needs no hint.
            (dict(exit_code=137, container_reason="OOMKilled"),
             "container exited with code 137 (OOMKilled)"),
        ]
        for pod, error in cases:
            with self.subTest(error=error), job_store():
                jobs.JobRecord.create(job_id="run-1", backend="gke_job")
                pods = json.dumps({"items": [failed_pod(**pod)]})
                fake = FakeKubectl({
                    "get job": (0, job_json(failed=1), ""),
                    "get pods": (0, pods, ""),
                    "logs": (0, "", ""),
                })
                result = self.status(fake)
                self.assertEqual(result.status, "FAILED")
                self.assertEqual(result.error, error)
                self.assertEqual(result.exit_code, pod.get("exit_code"))
                # Kept on the record, for once the pod is gone.
                self.assertEqual(
                    jobs.JobRecord.load("run-1").extra["error"], error)

    def test_a_vanished_job_says_where_it_was_last_seen(self):
        """A `--serve` Job deleted out from under the run used to give only
        kubectl's NotFound."""
        jobs.JobRecord.create(job_id="run-1", backend="gke_job", serve=True)
        record = jobs.JobRecord.load("run-1")
        record.start_time = "2026-01-01T00:00:00+00:00"
        record.save()
        pods = json.dumps({"items": [failed_pod(phase="Running")]})
        self.status(
            FakeKubectl({
                "get job": (0, job_json(active=1), ""),
                "get pods": (0, pods, ""),
            }))
        with self.assertRaises(ComputeConfigError) as ctx:
            self.status(FakeKubectl({"get job": self.GONE}))
        self.assertEqual(ctx.exception.error_code, "JOB_GONE")
        self.assertRegex(
            ctx.exception.message,
            r"^Job run-1 not found in namespace 'default'; last seen RUNNING "
            r"on node gke-tpu-node-1 at \S+ \(pod run-1-abcde\)\. ")


class TestGkeJobPodRanking(GkeJobLifecycleBase):
    """Which of a Job's pods status reports."""

    def test_the_pod_that_explains_the_job_speaks_for_it(self):
        """A retried Job has several pods; only one of them is the news.

        A live pod means progress, whatever else is queued. Otherwise the
        Pending pod is why the caller is still waiting, and a Succeeded or
        Failed attempt sorting ahead of it would report the Job as healthy
        progress until the timeout expired.
        """
        why = "0/2 nodes are available: 2 Insufficient google.com/tpu"
        unscheduled = dict(type="PodScheduled", status="False", message=why)
        pending = failed_pod(phase="Pending",
                             conditions=[unscheduled],
                             node=None)
        failed = failed_pod(name="job-pod-attempt-1")
        running = failed_pod(phase="Running", name="job-pod-live")

        def status(*pods: dict[str, Any], queue: bool = False):
            job = json.loads(job_json(active=1))
            if queue:
                job["metadata"] = dict(
                    labels={gke_job.gke_common.QUEUE_LABEL: "lq"})
            return self.status(
                FakeKubectl({
                    "get job": (0, json.dumps(job), ""),
                    "get pods": (0, json.dumps({"items": list(pods)}), ""),
                }))

        result = status(failed, pending)
        self.assertEqual((result.status, result.details["pod_phase"]),
                         ("PENDING", "Pending"))
        self.assertIn(why, result.details["pod_reason"])
        # Why it waits is what `job wait` reads: unbound with no queue,
        # admitted by a queue but with no node, or bound to a node, starting.
        bound = {**pending, "spec": {"nodeName": "gke-tpu-node-1"}}
        for pods, queue, why in (([failed,
                                   pending], False, base.PENDING_UNSCHEDULED),
                                 ([failed, pending], True,
                                  base.PENDING_ADMITTED_NO_NODE),
                                 ([bound], True, base.PENDING_STARTING)):
            self.assertEqual(
                status(*pods, queue=queue).details["pending_reason"], why)
        result = status(pending, running)
        self.assertEqual((result.status, result.details["pod_name"]),
                         ("RUNNING", "job-pod-live"))


if __name__ == "__main__":
    unittest.main()
