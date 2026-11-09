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
"""Unit tests for run leases: owner detection, the watcher, and the sweep."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

_repo_root = pathlib.Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from framework import jsonio, paths  # noqa: E402
from framework.compute import lease  # noqa: E402
from framework.compute.drivers import jobs  # noqa: E402

Path = pathlib.Path

RUN = "20260101T000000Z-000000000001"
OTHER_RUN = "20260101T000000Z-000000000002"


def _gone() -> dict:
    """A lease holder whose process has exited."""
    proc = subprocess.Popen(["true"])
    proc.wait()
    return {"pid": proc.pid, "started": "1"}


def _me() -> dict:
    return {"pid": os.getpid(), "started": jobs.process_started(os.getpid())}


class _LeaseTestCase(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        patcher = patch.dict(os.environ,
                             {"TPU_AGENT_SUITE_HOME": str(self.tmp)})
        patcher.start()
        self.addCleanup(patcher.stop)
        for run_id in (RUN, OTHER_RUN):
            paths.get_run_root(run_id).mkdir(parents=True)
        self.resolve = MagicMock()
        self.resolve.return_value.cancel.side_effect = (
            lambda jid: jobs.JobRecord.load(jid).mark("CANCELLED"))

    def _claim(self, holder: dict | None, run_id: str = RUN) -> None:
        lease.claim(run_id, holder, self.tmp / "ws-a", "pkg@1")

    def _live(self, job_id: str, run_id: str = RUN, serve: bool = False):
        return jobs.JobRecord.create(job_id=job_id,
                                     run_id=run_id,
                                     backend="gke_job",
                                     serve=serve).mark("RUNNING")

    def _set_watcher(self, holder: dict) -> None:
        jsonio.write_json(paths.get_run_root(RUN) / lease.WATCHER_FILE, holder)


class TestProcessInfo(unittest.TestCase):
    """The `ps` fallback (macOS) tells a gone process from an unanswered one."""

    def _ps(self, **run):
        with patch.object(jobs, "Path") as path, \
                patch.object(jobs.subprocess, "run", **run):
            path.return_value.read_text.side_effect = OSError
            path.return_value.is_dir.return_value = False
            return jobs.process_info(42)

    def test_ps(self):
        ps = subprocess.CompletedProcess
        self.assertEqual(
            self._ps(return_value=ps([], 0, "1 Thu Sep 24 05:53:38 2026 x")),
            (1, "Thu Sep 24 05:53:38 2026", "x"))
        self.assertIsNone(self._ps(return_value=ps([], 1, "")))
        with self.assertRaises(jobs.ProcessUnknown):
            self._ps(return_value=ps([], 0, "garbled"))
        with self.assertRaises(jobs.ProcessUnknown):
            self._ps(side_effect=subprocess.TimeoutExpired("ps", 5))


class TestSessionOwner(unittest.TestCase):

    def _owner(self, *chain: str):
        """The owner above pid 40, whose ancestors are named `chain`."""
        tree = {
            40 - 10 * i: (30 - 10 * i, str(i), name)
            for i, name in enumerate(chain)
        }
        with patch.object(jobs, "process_info", side_effect=tree.get):
            return lease.session_owner(40)

    def test_only_a_known_harness_process_is_the_owner(self):
        owner = self._owner("bash", "-zsh", "claude")
        self.assertEqual((owner["pid"], owner["started"]), (20, "2"))
        # A wrapper would exit with the hook: the run goes unleased instead.
        self.assertIsNone(self._owner("sh", "uv", "node"))
        self.assertIsNone(self._owner())  # unreadable
        with patch.object(jobs,
                          "process_info",
                          side_effect=jobs.ProcessUnknown):
            self.assertIsNone(lease.session_owner(40))


class TestSessionRun(_LeaseTestCase):
    """The run leased to the harness above a command, whatever its cwd."""

    def _lease(self, run_id: str, holder: dict, claimed_at: str,
               workspace: str) -> None:
        (self.tmp / workspace).mkdir(exist_ok=True)
        jsonio.write_json(
            paths.get_run_root(run_id) / lease.OWNER_FILE, {
                **holder, "host": lease._HOST,
                "workspace": str(self.tmp / workspace),
                "package": "pkg@1",
                "claimed_at": claimed_at
            })

    def test_finds_the_newest_run_leased_to_this_harness(self):
        harness = {"pid": 4242, "started": "s", "host": lease._HOST}
        third = "20260101T000000Z-000000000003"
        paths.get_run_root(third).mkdir(parents=True)
        self._lease(RUN, harness, "2026-01-01T00:00:01Z", "ws-old")
        # The same process after `/clear`: its newer session is the one live.
        self._lease(OTHER_RUN, harness, "2026-01-01T00:00:02Z", "ws-new")
        # Another session's harness, however recent, is not ours.
        self._lease(third, dict(harness, pid=99), "2026-01-01T00:00:09Z",
                    "ws-other")
        with patch.object(lease, "session_owner", return_value=harness):
            self.assertEqual(lease.session_run(),
                             (OTHER_RUN, self.tmp / "ws-new"))
            # A workspace deleted since its claim has nothing to resolve.
            (self.tmp / "ws-new").rmdir()
            self.assertEqual(lease.session_run(), (RUN, self.tmp / "ws-old"))
        # A pid reused by a later process is not the harness that leased.
        with patch.object(lease,
                          "session_owner",
                          return_value=dict(harness, started="t")):
            self.assertIsNone(lease.session_run())


class TestIsOrphaned(_LeaseTestCase):

    def test_only_a_gone_owner_with_no_watcher_is_orphaned(self):
        cases = {
            # Before leases, or an unidentifiable owner: nothing says it ended.
            "unleased": (None, None, False),
            "live owner": (_me(), None, False),
            "owner pid reused": (dict(_me(), started="0"), None, True),
            "owner gone": (_gone(), None, True),
            # A shared home directory: that host judges its own sessions.
            "owner on another host": (dict(_gone(), host="b"), None, False),
            # The watcher's grace period is what lets a crash be resumed.
            "owner gone, watcher live": (_gone(), _me(), False),
        }
        for name, (owner, watcher, expected) in cases.items():
            with self.subTest(name):
                shutil.rmtree(paths.get_run_root(RUN))
                paths.get_run_root(RUN).mkdir()
                self._claim(owner)
                if watcher:
                    self._set_watcher(watcher)
                self.assertEqual(lease.is_orphaned(RUN), expected)
        # `ps` could not answer: never reaped on a guess.
        (paths.get_run_root(RUN) / lease.WATCHER_FILE).unlink()
        with patch.object(jobs,
                          "process_info",
                          side_effect=jobs.ProcessUnknown):
            self.assertFalse(lease.is_orphaned(RUN))


class TestReapOrphanedJobs(_LeaseTestCase):

    def test_reaps_orphaned_batch_jobs_with_the_runs_own_config(self):
        self._claim(_gone())
        self._claim(_me(), run_id=OTHER_RUN)
        self._live("job-orphan")
        self._live("job-serve", serve=True)
        self._live("job-mine", run_id=OTHER_RUN)
        jobs.JobRecord.create(job_id="job-odd", run_id="../x").mark("RUNNING")

        reaped, failed = lease.reap_orphaned_jobs(self.resolve)

        self.assertEqual((reaped, failed), ({RUN: ["job-orphan"]}, {}))
        self.resolve.assert_called_with(workspace=self.tmp / "ws-a",
                                        package="pkg@1",
                                        run_id=RUN,
                                        job_id="job-orphan")
        for job_id in ("job-serve", "job-mine", "job-odd"):
            self.assertEqual(jobs.JobRecord.load(job_id).status, "RUNNING")

    def test_gives_up_on_a_run_after_failed_sweeps(self):
        self._claim(_gone())
        self._live("job-stuck")
        self.resolve.return_value.cancel.side_effect = RuntimeError("denied")
        for _ in range(lease.SWEEP_ATTEMPTS - 1):
            _, failed = lease.reap_orphaned_jobs(self.resolve)
            self.assertEqual(failed["job-stuck"], "denied")
        _, failed = lease.reap_orphaned_jobs(self.resolve)
        self.assertIn("giving up", failed["job-stuck"])
        self.assertEqual(lease.reap_orphaned_jobs(self.resolve), ({}, {}))


class TestWatch(_LeaseTestCase):

    def _watch(self, on_sleep=lambda now: None):
        """Run `watch` on a clock that only advances while it sleeps."""
        self.now = 0.0

        def sleep(seconds):
            self.now += seconds
            on_sleep(self.now)

        return lease.watch(RUN,
                           self.resolve,
                           grace=120,
                           poll=15,
                           retry=600,
                           sleep=sleep,
                           clock=lambda: self.now)

    def test_a_session_resumed_within_the_grace_period_keeps_its_jobs(self):
        self._claim(_gone())
        self._live("job-1")

        def on_sleep(now):
            if now == 60:
                self._claim(_me())
            elif now == 600:
                self._claim(_gone())

        self.assertEqual(self._watch(on_sleep), (["job-1"], {}))
        self.assertGreaterEqual(self.now, 600 + 120)

    def test_reaps_once_the_owner_is_gone_retrying_failed_cancels(self):
        self._claim(_gone())
        self._live("job-1")
        self._live("job-2")
        cancel = self.resolve.return_value.cancel
        flaky = {"job-2": 2}  # fails twice, then goes through

        def flaky_cancel(jid):
            if flaky.get(jid):
                flaky[jid] -= 1
                raise RuntimeError("blip")
            jobs.JobRecord.load(jid).mark("CANCELLED")

        cancel.side_effect = flaky_cancel
        self.assertEqual(self._watch(), (["job-1", "job-2"], {}))
        self.assertGreaterEqual(self.now, 120)
        self.assertFalse(
            (paths.get_run_root(RUN) / lease.WATCHER_FILE).exists())
        self._live("job-3")
        cancel.side_effect = RuntimeError("gone for good")
        self.assertEqual(self._watch(), ([], {"job-3": "gone for good"}))
        self.assertGreaterEqual(self.now, 120 + 600)

    def test_leaves_the_run_alone(self):
        root = paths.get_run_root(RUN)
        cases = (
            ("another watcher has it", lambda: self._set_watcher(_me())),
            ("unleased", lambda: self._claim(None)),
            ("no jobs left", lambda: self._live("job-1").mark("COMPLETED")),
            ("run deleted", lambda: shutil.rmtree(root)),
        )
        for name, setup in cases:
            with self.subTest(name):
                root.mkdir(exist_ok=True)
                self._claim(_gone())
                self._live("job-1")
                setup()
                self.assertEqual(self._watch(), ([], {}))
        self.resolve.assert_not_called()


class TestSpawn(_LeaseTestCase):

    def test_a_watcher_only_for_a_leased_run_without_one(self):
        # Only lease's own subprocess: jobs reads `ps` through it off Linux.
        with patch.object(lease, "subprocess") as mock_subprocess:
            popen = mock_subprocess.Popen
            lease.spawn_watcher(RUN)  # unleased
            self._claim(_me())
            lease.spawn_watcher(RUN)
            self._set_watcher(_me())
            lease.spawn_watcher(RUN)  # has one
            lease.spawn_sweep(RUN)
        self.assertEqual(
            [c.args[0][2:] for c in popen.call_args_list],
            [["compute", "job", "reap", "--watch", "--run-id", RUN],
             ["compute", "job", "reap", "--orphaned"]])
        self.assertTrue(popen.call_args.kwargs["start_new_session"])


if __name__ == "__main__":
    unittest.main()
