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
"""Unit tests for sealed jobs: records in protected storage, write-once outputs."""

from __future__ import annotations

import hashlib
import io
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from framework import jsonio, paths, utils
from framework.compute import cli as compute_cli
from framework.compute import config as compute_config
from framework.compute.drivers import gke_job, jobs
from framework.compute.drivers import local as local_driver
from framework.compute.drivers import shell as shell_driver

Path = pathlib.Path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class _ProbeShellDriver(shell_driver.RemoteShellDriver):
    """A real RemoteShellDriver over a scripted status probe.

    Not a mock: what these tests exercise is the driver's own `get_status`
    -- how it parses the probe's output and reconciles it with the record
    (the CANCELLED-while-still-RUNNING case) -- which a MagicMock would
    stand in for rather than run. Only the transport is stubbed, as
    `test_drivers.TestRemoteShellDrivers` does with `subprocess.run`.
    """
    target_description = "stub-ssh"

    def __init__(self):
        super().__init__()
        self.probe_out = "RUNNING\n"

    @classmethod
    def from_params(cls, params):
        return cls()

    def inspect(self):
        return {}

    def _sh(self, script: str, timeout: int = 60, **_kw):
        return subprocess.CompletedProcess(["ssh"], 0, self.probe_out, "")

    def _copy_in(self, local_src: Path, remote_dest: str) -> None:
        pass

    def _copy_out(self, remote_src: str, local_dest: Path) -> None:
        pass

    def _copy_out_file(self, remote_src: str, local_dest: Path) -> None:
        pass


class TestSealedJobs(unittest.TestCase):
    """Where a job's record lives, how it is found, and when its outputs seal."""

    def setUp(self):
        self.home_dir = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.home_dir, ignore_errors=True)
        patcher = mock.patch.dict(os.environ,
                                  {"TPU_AGENT_SUITE_HOME": str(self.home_dir)})
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(jobs.JOBS_DIR_ENV, None)
        self.run_id = "20261110T070000Z-abcdef012345"
        self.sealed_root = paths.get_sealed_jobs_root()

    def _sealed_env(self):
        """The environment hc.py gives the compute commands it runs."""
        return mock.patch.dict(os.environ,
                               {jobs.JOBS_DIR_ENV: str(self.sealed_root)})

    def _create_sealed(self, job_id: str, **fields) -> jobs.JobRecord:
        with self._sealed_env():
            fields.setdefault("run_id", self.run_id)
            return jobs.JobRecord.create(job_id=job_id, **fields)

    def test_sealed_records_live_in_protected_storage_and_win_every_lookup(
            self):
        """hc.py's env puts a job under sealed_jobs/; every reader finds it there first."""
        ordinary = jobs.JobRecord.create(job_id="job-a",
                                         run_id=self.run_id,
                                         backend="local")
        sealed = self._create_sealed("job-b",
                                     backend="gke_job",
                                     status="RUNNING")
        self._create_sealed("job-c", backend="local", run_id="other-run")

        # Created where the env says, with the root kept out of job.json.
        self.assertEqual(ordinary.directory, jobs.default_jobs_dir() / "job-a")
        self.assertFalse(ordinary.is_sealed)
        self.assertEqual(sealed.directory, self.sealed_root / "job-b")
        self.assertTrue(sealed.is_sealed)
        self.assertFalse((jobs.default_jobs_dir() / "job-b").exists())
        self.assertNotIn("_root",
                         jsonio.load_json(sealed.directory / "job.json"))

        # An agent plants a record of the same id in the writable root,
        # claiming another backend and a finished status. Without the env,
        # load() still finds the protected one, and only that one.
        forged = jobs.default_jobs_dir() / "job-b"
        forged.mkdir(parents=True)
        jsonio.write_json(
            forged / "job.json", {
                "job_id": "job-b",
                "backend": "local",
                "status": "COMPLETED",
                "run_id": self.run_id
            })
        self.assertNotIn(jobs.JOBS_DIR_ENV, os.environ)
        loaded = jobs.JobRecord.load("job-b")
        assert loaded is not None
        self.assertEqual((loaded.is_sealed, loaded.backend, loaded.status),
                         (True, "gke_job", "RUNNING"))
        self.assertEqual(jobs.job_directory("job-b"),
                         self.sealed_root / "job-b")

        # So resolve_driver cannot be redirected onto LocalDriver by it...
        chosen: list[str] = []
        with (mock.patch.object(compute_config,
                                "find_environment_data",
                                return_value={}),
              mock.patch.object(compute_config,
                                "resolve_profile_name",
                                return_value=("p", "default")),
              mock.patch.object(compute_config,
                                "parse_compute_section",
                                return_value=("local", {})),
              mock.patch.object(compute_config,
                                "get_driver",
                                side_effect=lambda b:
                                (chosen.append(b), mock.MagicMock())[1])):
            compute_config.resolve_driver(job_id="job-b")
        self.assertEqual(chosen, ["gke_job"])

        # ...and the reaper's list_jobs sees sealed jobs, each id once.
        listed = jobs.list_jobs(run_id=self.run_id)
        self.assertEqual([(r.job_id, r.is_sealed) for r in listed],
                         [("job-a", False), ("job-b", True)])
        self.assertEqual([r.job_id for r in jobs.list_jobs()],
                         ["job-a", "job-b", "job-c"])

        # Traversal ids and records whose own job_id disagrees are refused.
        jsonio.write_json(jobs.default_jobs_dir() / "job.json", {
            "job_id": "..",
            "backend": "local",
        })
        jsonio.write_json(ordinary.directory / "job.json", {
            "job_id": "other-id",
            "backend": "local",
        })
        for bad in ("../../runs/jobs", "..", ".", "a/b", r"a\b"):
            self.assertIsNone(jobs.JobRecord.load(bad))
            with self.assertRaises(utils.TpuAgentSuiteError):
                jobs.job_directory(bad)
        self.assertIsNone(jobs.JobRecord.load("job-a"))

    def test_a_sealed_job_seals_once_when_terminal_then_serves_the_copy(self):
        """Through `job artifacts` on a local job: not before it ends, once after, never again."""
        job_id = "job-20261110070000-990011"
        declared = [{"local": "summary.json", "remote": "summary.json"}]
        live_pid = os.getpid()
        record = self._create_sealed(
            job_id,
            backend="local",
            status="RUNNING",
            pid=live_pid,
            pid_started=jobs.process_started(live_pid),
            extra={"outputs": declared},
        )
        sealed_dir = record.directory / jobs.SEALED_ARTIFACTS_DIRNAME
        served = paths.get_run_artifacts_dir(self.run_id) / job_id
        (record.directory / "output.log").write_text("partial\n")
        (record.directory / "summary.json").write_bytes(b'{"status": "part')
        drv = local_driver.LocalDriver()
        args = mock.MagicMock(job_action="artifacts",
                              job_id=job_id,
                              out_dir=None,
                              run_id=self.run_id,
                              session_run_id=None,
                              workspace=None,
                              json=True)

        def fetch() -> int:
            with redirect_stdout(io.StringIO()):
                return compute_cli._action_job(drv, args)

        # Still running (this process is alive): fetched, not sealed.
        fetch()
        self.assertFalse(
            jobs.JobRecord.load(job_id).extra.get("artifacts_sealed"))
        self.assertFalse(sealed_dir.exists())

        # Finished: the complete output seals, log included, and is served,
        # without reverting the terminal status `get_status` settled on disk.
        final = b'{"status": "ok", "latency_ms": 9.5}\n'
        (record.directory / "summary.json").write_bytes(final)
        (record.directory / "output.log").write_text("all done\n")
        (record.directory / "exit_code.txt").write_text("0\n")
        with mock.patch.object(jobs, "process_is_alive", return_value=False):
            self.assertEqual(fetch(), 0)
        after = jobs.JobRecord.load(job_id)
        assert after is not None
        self.assertEqual((after.status, after.exit_code), ("COMPLETED", 0))
        self.assertTrue(after.extra["artifacts_sealed"])
        self.assertEqual(after.extra["sealed_artifacts"],
                         {"summary.json": _sha256_bytes(final)})
        self.assertEqual((sealed_dir / "summary.json").read_bytes(), final)
        self.assertEqual((sealed_dir / f"{job_id}.log").read_text(),
                         "all done\n")
        self.assertEqual((served / "summary.json").read_bytes(), final)

        # Never collected again: tampering with the source or the served
        # copy changes nothing, and the driver is not asked.
        (record.directory / "summary.json").write_bytes(b'{"status":"fake"}')
        (served / "summary.json").write_bytes(b"hacked")
        args.out_dir = str(self.home_dir / "custom_out")
        with mock.patch.object(drv,
                               "fetch_artifacts",
                               side_effect=AssertionError("must not refetch")):
            self.assertEqual(fetch(), 0)
        self.assertEqual(
            (self.home_dir / "custom_out" / "summary.json").read_bytes(),
            final)

        # A symlink anywhere in the job directory -- pre-planted in sealed/ or
        # brought in by fetch_artifacts -- refuses the seal and leaves nothing.
        secret = paths.get_campaigns_root() / "campaign.db"
        secret.write_bytes(b"SQLite format 3\x00 secrets")
        linked = self._create_sealed("job-20261110070000-990012",
                                     backend="local",
                                     status="COMPLETED",
                                     extra={"outputs": declared})
        linked_sealed = linked.directory / jobs.SEALED_ARTIFACTS_DIRNAME
        linked_sealed.mkdir()
        (linked_sealed / "summary.json").symlink_to(secret)
        with self.assertRaisesRegex(utils.TpuAgentSuiteError, "symlinks"):
            jobs.seal_artifacts(drv, linked, self.home_dir / "out")
        (linked_sealed / "summary.json").unlink()
        with mock.patch.object(drv,
                               "fetch_artifacts",
                               side_effect=lambda _j, d:
                               (Path(d) / "summary.json").symlink_to(secret)):
            with self.assertRaisesRegex(utils.TpuAgentSuiteError, "symlinks"):
                jobs.seal_artifacts(drv, linked, self.home_dir / "out")
        self.assertFalse(
            jobs.JobRecord.load(linked.job_id).extra.get("artifacts_sealed"))
        self.assertFalse(linked_sealed.exists())
        self.assertFalse((self.home_dir / "out" / "summary.json").exists())

    def test_the_backend_not_the_record_decides_when_a_seal_may_happen(self):
        """A record claiming the job is over seals nothing while the target says otherwise."""
        # Shell: the record says CANCELLED, the probe says RUNNING. get_status
        # reports RUNNING without overwriting CANCELLED on disk, so the seal
        # waits; once the probe says the process is gone, it is CANCELLED
        # for good and the seal goes through.
        sh_id = "job-20261110070000-990022"
        sh_record = self._create_sealed(sh_id,
                                        backend="ssh",
                                        status="RUNNING",
                                        extra={"outputs": []})
        sh_record.status = "CANCELLED"
        sh_record.save()
        drv = _ProbeShellDriver()
        self.assertEqual(drv.get_status(sh_id).status, "RUNNING")
        self.assertEqual(jobs.JobRecord.load(sh_id).status, "CANCELLED")
        jobs.seal_artifacts(drv, jobs.JobRecord.load(sh_id),
                            self.home_dir / "sh_mid")
        self.assertFalse(
            jobs.JobRecord.load(sh_id).extra.get("artifacts_sealed"))

        drv.probe_out = "FAILED 143\n"
        self.assertEqual(drv.get_status(sh_id).status, "CANCELLED")
        self.assertEqual(jobs.JobRecord.load(sh_id).status, "CANCELLED")
        jobs.seal_artifacts(drv, jobs.JobRecord.load(sh_id),
                            self.home_dir / "sh_done")
        self.assertTrue(jobs.JobRecord.load(sh_id).extra["artifacts_sealed"])

        # GKE: the record says COMPLETED while the Job is still active on
        # the cluster. Only the cluster's word seals, and then the final
        # pod log is what is sealed.
        gke = gke_job.GkeJobDriver.from_params({
            "namespace": "default",
            "image": "registry/torch-tpu@sha256:abc",
            "tpu_accelerator": "tpu7x",
            "tpu_topology": "2x2x1",
        })
        gke_id = "job-20261110070000-990044"
        gke_record = self._create_sealed(gke_id,
                                         backend="gke_job",
                                         status="RUNNING",
                                         extra={"outputs": []})
        gke_record.status = "COMPLETED"
        gke_record.save()
        cluster = {"job": '{"status": {"active": 1}}', "log": "partial\n"}

        def fake_kubectl(cmd, **_kw):
            joined = " ".join(cmd)
            if "logs" in joined:
                return subprocess.CompletedProcess(cmd, 0, cluster["log"], "")
            if "get job" in joined:
                return subprocess.CompletedProcess(cmd, 0, cluster["job"], "")
            return subprocess.CompletedProcess(cmd, 0, '{"items": []}', "")

        with mock.patch("subprocess.run", side_effect=fake_kubectl):
            jobs.seal_artifacts(gke, jobs.JobRecord.load(gke_id),
                                self.home_dir / "gke_mid")
        self.assertFalse(
            jobs.JobRecord.load(gke_id).extra.get("artifacts_sealed"))

        cluster.update(job='{"status": {"succeeded": 1}}', log="complete\n")
        with mock.patch("subprocess.run", side_effect=fake_kubectl):
            jobs.seal_artifacts(gke, jobs.JobRecord.load(gke_id),
                                self.home_dir / "gke_done")
        self.assertTrue(jobs.JobRecord.load(gke_id).extra["artifacts_sealed"])
        self.assertEqual(
            (self.home_dir / "gke_done" / f"{gke_id}.log").read_text(),
            "complete\n")


if __name__ == "__main__":
    unittest.main()
