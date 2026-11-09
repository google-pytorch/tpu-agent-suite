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
"""Tests for gke_job logs, artifacts, cancellation, port-forward and run."""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import threading
import time
import unittest
from typing import Any
from unittest.mock import patch

# Imports below follow this sys.path setup, so the file runs directly.
# ruff: noqa: E402
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.compute import ComputeConfigError
from framework.compute.drivers import base, gcs_staging, gke_job, jobs

from framework.tests.compute.gke_job.helpers import (  # isort: skip
    FakeKubectl, GkeJobLifecycleBase, failed_pod, hello_script, job_json,
    job_store, make_driver, set_up_module)

Path = pathlib.Path


def setUpModule():  # pylint: disable=invalid-name
    set_up_module()


def phase_pods(*pods: tuple[str, str]) -> str:
    """Pods, each (name, phase), as `kubectl get pods -o json` lists them."""
    return json.dumps({
        "items": [
            dict(metadata=dict(name=name), status=dict(phase=phase))
            for name, phase in pods
        ]
    })


_RUNNING_POD = json.dumps({"items": [{"status": {"phase": "Running"}}]})


class TestGkeJobLogs(GkeJobLifecycleBase):
    """Logs from the pod, Cloud Logging or the uploaded copy."""

    @staticmethod
    def _entry(text: str, **payload: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "resource": {
                "labels": {
                    "pod_name": "run-1-abcde"
                }
            }
        }
        entry.update({"jsonPayload": payload}
                     if payload else {"textPayload": text + "\n"})
        return entry

    def _fetch_gone_pod_logs(self,
                             answer: tuple[int, str, str],
                             on_path: bool = True) -> tuple[Any, Any]:
        """fetch_logs for a Job whose pod and copies are gone, with gcloud
        answering `answer`. Returns (logs or the raised error, the fake)."""
        record = jobs.JobRecord.create(job_id="run-1", backend="gke_job")
        record.start_time = "2026-01-01T00:20:00+00:00"
        record.extra["uid"] = "uid-1"
        record.save()
        fake = FakeKubectl({
            "logging read": answer,
            "get job": self.GONE,
            "logs": (1, "", 'pods "run-1-abcde" not found'),
        })
        with patch.object(subprocess, "run", fake), \
                patch.object(shutil, "which",
                             return_value="/bin/gcloud" if on_path else None):
            try:
                return make_driver().fetch_logs("run-1"), fake
            except ComputeConfigError as exc:
                return exc, fake

    def test_logs_of_a_gone_pod_come_from_cloud_logging(self):
        """No pod, no saved copy, no GCS copy: Cloud Logging's lines, oldest
        first, under a line saying where they came from."""
        newest_first = json.dumps([
            self._entry("", message="Killed"),
            self._entry("Downloading shards: 40%"),
        ])
        logs, fake = self._fetch_gone_pod_logs((0, newest_first, ""))
        first, *rest = logs.splitlines()
        self.assertTrue(first.startswith("[compute] Job run-1's pod is gone"))
        self.assertIn("Cloud Logging kept", first)
        self.assertEqual(rest, ["Downloading shards: 40%", "Killed"])
        query = fake.find("logging read")
        self.assertIn("--order=desc", query)
        self.assertIn(
            f"--limit={gke_job.log_retrieval.CLOUD_LOGGING_MAX_LINES}", query)
        # The record's UID and start reach the query (the window itself is
        # covered by test_cloud_logging_query_keeps_to_one_job).
        self.assertIn('labels."k8s-pod/controller-uid"="uid-1"', query[3])
        self.assertIn('timestamp>="2026-01-01T00:10:00Z"', query[3])

    def test_no_cloud_logging_lines_says_why(self):
        cases = [
            # (gcloud on PATH, its answer, words in the error)
            (False, (0, "[]", ""), "gcloud is not on PATH"),
            (True, (1, "", "ERROR: (gcloud.logging.read) PERMISSION_DENIED"),
             "PERMISSION_DENIED"),
            (True, (0, "[]", ""), "holds no output for the Job"),
        ]
        for on_path, answer, expected in cases:
            with self.subTest(expected=expected), job_store():
                exc, _ = self._fetch_gone_pod_logs(answer, on_path)
                self.assertIsInstance(exc, ComputeConfigError)
                self.assertEqual(exc.error_code, "LOGS_UNAVAILABLE")
                self.assertIn(expected, exc.message)
                # The query to run by hand is always given.
                self.assertIn('labels."k8s-pod/job-name"="run-1"', exc.message)

    def test_cloud_logging_query_keeps_to_one_job(self):
        """Review on #680: cancel and resubmit with the same --name, and a
        window widened around one Job's times takes in the other's output.

        With the UID, only this Job's pods match, so the window can be wide.
        Without it (an older record), the window is exactly the record's.
        """
        start = gke_job.gke_common.parse_time("2026-01-01T00:20:00Z")
        end = gke_job.gke_common.parse_time("2026-01-01T01:00:00Z")
        base = ('resource.type="k8s_container" AND '
                'resource.labels.namespace_name="ns" AND '
                'labels."k8s-pod/job-name"="run-1"')
        self.assertEqual(
            gke_job.log_retrieval._cloud_logging_query("ns", "run-1", "c1",
                                                       start, end, "uid-2"),
            base + ' AND (labels."k8s-pod/controller-uid"="uid-2" OR '
            'labels."k8s-pod/batch_kubernetes_io/controller-uid"="uid-2") '
            'AND resource.labels.cluster_name="c1" AND '
            'timestamp>="2026-01-01T00:10:00Z" AND '
            'timestamp<="2026-01-01T01:10:00Z"')
        self.assertEqual(
            gke_job.log_retrieval._cloud_logging_query("ns", "run-1", None,
                                                       start, end),
            base + ' AND timestamp>="2026-01-01T00:20:00Z" AND '
            'timestamp<="2026-01-01T01:00:00Z"')

    def _lost_node_fake(self,
                        cloud: list[str],
                        node_log: str = "c1 385\nc2 729\n") -> FakeKubectl:
        """A Job whose node was reset: the pod is left `Unknown`/255, its log
        on the node is `node_log`, and Cloud Logging has `cloud`."""
        pods = json.dumps(
            {"items": [failed_pod(exit_code=255, container_reason="Unknown")]})
        return FakeKubectl({
            "logging read":
            (0, json.dumps([self._entry(line)
                            for line in reversed(cloud)]), ""),
            "get job": (0, job_json(failed=1), ""),
            "get pods": (0, pods, ""),
            "logs": (0, node_log, ""),
        })

    def test_a_lost_nodes_log_gets_cloud_loggings_tail(self):
        """Hard-resetting the node mid-benchmark left `job logs` with c1 and
        c2, while Cloud Logging also had c4 and c8. The node's copy can end
        mid-line; that line is read from Cloud Logging whole. Status reads
        never ask Cloud Logging; the first log read waits until it has had
        time to ingest the tail, asks once, and keeps the answer."""
        jobs.JobRecord.create(job_id="run-1", backend="gke_job")
        driver = make_driver()
        cloud = ["c1 385", "c2 729", "c4 1393", "c8 2613"]
        status = self._lost_node_fake(cloud, "c1 385\nc2 72")
        with patch.object(subprocess, "run", status), \
                patch.object(shutil, "which",
                             return_value="/bin/gcloud"):
            driver.get_status("run-1")
        self.assertIsNone(status.find("logging read"))
        read = self._lost_node_fake(cloud)
        with patch.object(subprocess, "run", read), \
                patch.object(shutil, "which",
                             return_value="/bin/gcloud"), \
                patch.object(time, "sleep") as sleep:
            logs = driver.fetch_logs("run-1").splitlines()
        self.assertAlmostEqual(sleep.call_args[0][0],
                               gke_job.log_retrieval.CLOUD_LOGGING_SETTLE_SEC,
                               delta=5)
        self.assertEqual(logs[0], "c1 385")
        self.assertTrue(logs[1].startswith("[compute] The node was lost"))
        self.assertEqual(logs[2:], ["c2 729", "c4 1393", "c8 2613"])
        settled = FakeKubectl({"logs": (1, "", "timed out")})
        with patch.object(subprocess, "run", settled):
            self.assertEqual(driver.fetch_logs("run-1").splitlines(), logs)
        self.assertIsNone(settled.find("logging read"))

    def test_a_lost_nodes_log_read_before_the_job_settles(self):
        """`job logs` on a pod left Unknown, before any status read, keeps
        and settles the copy the same way; a copy with no complete line
        takes all of Cloud Logging's."""
        jobs.JobRecord.create(job_id="run-1", backend="gke_job")
        driver = make_driver()
        with patch.object(subprocess, "run",
                          self._lost_node_fake(["c1 385", "c2 729"], "c1 3")), \
                patch.object(shutil, "which",
                             return_value="/bin/gcloud"), \
                patch.object(time, "sleep"):
            logs = driver.fetch_logs("run-1").splitlines()
        self.assertTrue(logs[0].startswith("[compute] The node was lost"))
        self.assertEqual(logs[1:], ["c1 385", "c2 729"])
        self.assertFalse(
            gke_job.log_retrieval.lost_node_marker("run-1").exists())

    def test_logs_fall_back_to_the_copy_the_pod_uploaded(self):
        fake = FakeKubectl({"logs": (1, "", "timed out")})

        def download(bucket, blob, dest):
            self.assertEqual((bucket, blob),
                             ("b", "run-1/out/.compute-output.log"))
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_text("step_time_ms: 41.7\n", encoding="utf-8")
            return True

        with \
                patch.object(subprocess, "run", fake), \
                patch.object(gcs_staging, "download_file_from_gcs", download):
            self.assertEqual(
                make_driver(staging_bucket="gs://b").fetch_logs("run-1"),
                "step_time_ms: 41.7\n")


class TestGkeJobArtifacts(GkeJobLifecycleBase):
    """Fetching a Job's artifacts."""

    def test_a_gcs_failure_breaks_neither_status_nor_cancel(self):
        fake = FakeKubectl({"get job": (0, job_json(succeeded=1), "")})

        def download(bucket, blob, dest):
            Path(dest).write_text("half a lo", encoding="utf-8")
            raise OSError("mTLS handshake failed")

        for observe in ("get_status", "cancel"):
            with self.subTest(observe), job_store(), \
                    patch.object(subprocess, "run", fake), \
                    patch.object(gcs_staging, "download_file_from_gcs",
                                 download), \
                    patch.object(gcs_staging, "delete_staged_prefix") as rm, \
                    contextlib.redirect_stderr(io.StringIO()):
                jobs.JobRecord.create(job_id="run-1",
                                      backend="gke_job",
                                      extra={"staged_prefix": "run-1/"})
                getattr(make_driver(staging_bucket="gs://b"), observe)("run-1")
                if observe == "cancel":
                    # Inputs only: outputs nobody fetched survive a reap.
                    rm.assert_called_once_with("b", "run-1/in/")
                self.assertEqual(
                    jobs.JobRecord.load("run-1").status, "COMPLETED")
                # A partial download must not pass for the saved log.
                self.assertEqual(
                    list((jobs.jobs_dir() / "run-1").glob("output.log*")), [])
        self.assertIsNotNone(fake.find("delete job run-1"))


class TestGkeJobCancel(GkeJobLifecycleBase):
    """Cancelling a Job and settling its record."""

    def test_cancel_removes_job_and_configmap(self):
        # Both: a Job left holding a TPU node keeps the next run pending, and
        # a stranded ConfigMap blocks the name.
        fake = FakeKubectl()
        driver = make_driver()
        with patch.object(subprocess, "run", fake):
            driver.cancel("run-1")
        self.assertIsNotNone(fake.find("delete job run-1"))
        self.assertIsNotNone(fake.find("delete configmap run-1"))
        # Already gone is not an error: cancel is safe to repeat.
        with patch.object(subprocess, "run",
                          FakeKubectl({"delete": (1, "", "NotFound")})):
            driver.cancel("run-1")

    def test_cancel_that_cannot_delete_raises_and_keeps_the_record_live(self):
        """A CANCELLED record over a running Job is a leak; refuse to write it."""
        fake = FakeKubectl({
            "get job": (0, job_json(active=1), ""),
            "get pods": (0, _RUNNING_POD, ""),
            "delete": (1, "", "error: You must be logged in to the server "
                       "(Unauthorized)"),
        })
        driver = make_driver()
        jobs.JobRecord.create(job_id="run-1",
                              job_name="run-1",
                              backend="gke_job")
        with patch.object(subprocess, "run", fake):
            with self.assertRaises(gke_job.KubectlError):
                driver.cancel("run-1")
        self.assertEqual(jobs.JobRecord.load("run-1").status, "PENDING")


class TestGkeJobPortForward(GkeJobLifecycleBase):
    """Port-forwarding to a Job's pod."""

    def test_port_forward_skips_a_dead_pod(self):
        """`-l job-name=X` returns finished pods too, in no useful order."""
        pods = phase_pods(("job-pod-old", "Failed"),
                          ("job-pod-live", "Running"))
        fake = FakeKubectl({
            "get pods": (0, pods, ""),
            "port-forward": (0, "Forwarding\n", ""),
        })
        driver = make_driver()
        with patch.object(subprocess, "run", fake):
            rc = driver.port_forward(local_port=8080,
                                     remote_port=8000,
                                     job_id="my-job")
        self.assertEqual(rc, 0)
        self.assertIsNotNone(
            fake.find("port-forward pod/job-pod-live 8080:8000"))
        self.assertIsNone(fake.find("job-pod-old"))

        # Ctrl-C is how a foreground forward is meant to end.
        def interrupted(cmd, *args, **kwargs):
            if "port-forward" in cmd:
                raise KeyboardInterrupt()
            return fake(cmd, *args, **kwargs)

        with patch.object(subprocess, "run", interrupted):
            self.assertEqual(
                driver.port_forward(local_port=8080,
                                    remote_port=8000,
                                    job_id="my-job"), 0)

        # No pod running: forward to the Job and let kubectl pick.
        fake.responses["get pods"] = (0,
                                      phase_pods(
                                          ("job-pod-gone", "Succeeded")), "")
        with patch.object(subprocess, "run", fake):
            driver.port_forward(local_port=8080,
                                remote_port=8000,
                                job_id="my-job")
        self.assertIsNotNone(fake.find("port-forward job/my-job"))


class TestGkeJobRun(unittest.TestCase):

    def setUp(self):
        self.script = hello_script(self)
        self.tmp = self.script.parent
        self.enterContext(job_store())

    UNSCHEDULABLE = ("0/15 nodes are available: 8 Insufficient "
                     "google.com/tpu. No preemption victims found.")

    def _pending_fake(self) -> FakeKubectl:
        """A cluster with no free chips: the Job stays active, its pod
        Pending with the scheduler's reason, for as long as anyone asks."""
        unschedulable = dict(type="PodScheduled",
                             status="False",
                             reason="Unschedulable",
                             message=self.UNSCHEDULABLE)
        pod = failed_pod(phase="Pending",
                         conditions=[unschedulable],
                         node=None)
        return FakeKubectl({
            "get job": (0, job_json(active=1), ""),
            "get pods": (0, json.dumps({"items": [pod]}), ""),
        })

    def test_a_run_that_times_out_in_the_queue_is_deleted_not_leaked(self):
        # Timed out before any node took it: no log read, Job deleted.
        fake = self._pending_fake()
        driver = make_driver()
        out = self.tmp / "probe.json"
        with patch.object(subprocess, "run", fake), \
             patch.object(time, "sleep"):
            result = driver.run(
                str(self.script),
                timeout=0,
                outputs=[base.StagedFile(local=str(out), remote="probe.json")])
        self.assertTrue(result.details["never_started"])
        self.assertIn(self.UNSCHEDULABLE, result.stderr)
        self.assertIn("The Job has been deleted.", result.stderr)
        self.assertIn("probe.json", result.stderr)
        self.assertIsNone(fake.find("logs"))
        self.assertIsNotNone(fake.find("delete job"))
        record = jobs.JobRecord.load(result.details["job_id"])
        self.assertEqual(record.status, "CANCELLED")
        self.assertTrue(record.extra["never_started"])
        self.assertTrue(record.extra["error"].startswith("Cancelled while: "))
        self.assertIn(self.UNSCHEDULABLE, record.extra["error"])

    def test_sigterm_during_the_wait_still_deletes_the_job(self):
        # The caller's shell timeout kills with SIGTERM.
        if threading.current_thread() is not threading.main_thread():
            self.skipTest("the guard is main-thread only; the kill would "
                          "take the runner down")
        before = signal.getsignal(signal.SIGTERM)
        fake = self._pending_fake()
        driver = make_driver()

        def sleep(_sec: float) -> None:
            os.kill(os.getpid(), signal.SIGTERM)

        with patch.object(subprocess, "run", fake), \
             patch.object(time, "sleep", sleep), \
             contextlib.redirect_stderr(io.StringIO()) as err, \
             self.assertRaises(SystemExit) as ctx:
            driver.run(str(self.script), timeout=3600)
        self.assertEqual(ctx.exception.code, 128 + signal.SIGTERM)
        self.assertIsNotNone(fake.find("delete job"))
        # The caller sees why, not just its own timeout.
        self.assertIn("was PENDING", err.getvalue())
        self.assertIn(self.UNSCHEDULABLE, err.getvalue())
        self.assertIn("A resubmit would wait the same way", err.getvalue())
        # Whatever handler the runner had is put back.
        self.assertIs(signal.getsignal(signal.SIGTERM), before)

    def _wait_through(self, timeout: int, *phases: tuple[float, str]):
        """Run `_wait` on a fake clock against a Job that moves through
        `phases`, each (seconds, status); the last one never ends."""
        clock = [0.0]

        def status(job_id: str) -> base.AsynchronousExecutionResult:
            at, state = clock[0], phases[-1][1]
            for seconds, phase in phases:
                if at < seconds:
                    state = phase
                    break
                at -= seconds
            return base.AsynchronousExecutionResult(job_id=job_id,
                                                    status=state)

        def sleep(seconds: float) -> None:
            clock[0] += seconds

        driver = make_driver(queue="lq")
        with patch.object(driver, "get_status", side_effect=status), \
                patch.object(time, "sleep", side_effect=sleep), \
                patch.object(time, "monotonic",
                             side_effect=lambda: clock[0]):
            return driver._wait("job-1", timeout)

    def test_only_running_time_counts_against_the_timeout(self):
        # 40 minutes queued, then 25 of work, on a 1h timeout.
        result = self._wait_through(3600, (2400, "PENDING"), (1500, "RUNNING"),
                                    (1, "COMPLETED"))
        self.assertEqual(result.status, "COMPLETED")
        # A timeout says how long it queued first.
        result = self._wait_through(3600, (600, "PENDING"), (1, "RUNNING"))
        self.assertEqual(result.details["queued_sec"], 600)
        self.assertIn("running for 3600s", result.error)
        # A Job that never starts still times out.
        result = self._wait_through(300, (1, "PENDING"))
        self.assertEqual(result.status, "PENDING")
        self.assertIn("never ran", result.error)


if __name__ == "__main__":
    unittest.main()
