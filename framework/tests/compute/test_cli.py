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
"""Unit tests for `runtime.py compute`.

Covers CLI behaviour, output envelopes, the job subcommands (wait, reap), and
the generated agent contract.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

# Imports below follow this sys.path setup, so the file runs directly.
# ruff: noqa: E402
_repo_root = pathlib.Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from framework import compute, paths, runtime
from framework.builder import prompt_contracts
from framework.compute import (AsynchronousExecutionResult, ComputeConfigError,
                               LocalDriver, StagedFile)
from framework.compute.drivers import base, jobs
from framework.tests.compute.helpers import (MockTestDriver, ScriptMixin,
                                             cli_args, patch_home,
                                             set_up_module, tear_down_module,
                                             temp_home)

Path = pathlib.Path


def setUpModule():  # noqa: N802 - unittest's spelling
    set_up_module()


def tearDownModule():  # noqa: N802 - unittest's spelling
    tear_down_module()


# A well-formed run id.
_RUN = "20260101T000000Z-123456789abc"


class TestRuntimeCliCompute(ScriptMixin, unittest.TestCase):
    """`runtime.py compute` argument parsing and dispatch."""

    def _run_cli(self, argv: list[str]) -> tuple[int, str]:
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            rc = runtime.main(argv)
            return rc, buf.getvalue()

    def test_run_input_and_output_specs_are_parsed_and_validated(self):
        """Verify `--input LOCAL[:REMOTE]` and `--output REMOTE[:LOCAL]` reach the driver."""
        self.assertEqual(StagedFile.parse_input("a/b.json"),
                         StagedFile(local="a/b.json", remote="b.json"))
        self.assertEqual(StagedFile.parse_input("a/b.json:sub/c.json"),
                         StagedFile(local="a/b.json", remote="sub/c.json"))
        self.assertEqual(StagedFile.parse_output("r.json"),
                         StagedFile(local="r.json", remote="r.json"))
        self.assertEqual(StagedFile.parse_output("r.json:out/r.json"),
                         StagedFile(local="out/r.json", remote="r.json"))
        for bad in ("/abs/path.json", "../up.json", "a/../../b"):
            with self.assertRaises(ComputeConfigError):
                StagedFile.parse_output(bad)
        # `~` after the colon reaches us unexpanded (the shell leaves it), and
        # is how a prompt lands an output in its run directory; the target
        # side is never a home directory, so `~` there stays a plain name.
        home = os.path.expanduser("~")
        self.assertEqual(
            StagedFile.parse_output("x.pb:~/runs/r/x.pb"),
            StagedFile(local=f"{home}/runs/r/x.pb", remote="x.pb"))
        self.assertEqual(StagedFile.parse_input("~/in.json"),
                         StagedFile(local=f"{home}/in.json", remote="in.json"))
        self.assertEqual(StagedFile.parse_output("~x.json"),
                         StagedFile(local="~x.json", remote="~x.json"))

        path = self._script("print('x')\n")
        with io.StringIO() as err, contextlib.redirect_stderr(err):
            rc, _ = self._run_cli([
                "compute", "--backend", "mock_test", "run", "--code", path,
                "--input", "/definitely/missing.json", "--json"
            ])
            self.assertNotEqual(rc, 0)
            self.assertEqual(
                json.loads(err.getvalue())["error_code"], "INPUT_NOT_FOUND")

    def test_submit_without_json_prints_only_the_job_id(self):
        """Verify plain `submit` output is capturable as `JOB_ID=$(...)`.

        The executor prompts do exactly that and then pass the value to
        `--job-id`, so anything but the bare id on stdout breaks polling.
        """
        path = self._script("print('submitted')\n")
        argv = [
            "compute", "--backend", "mock_test", "submit", "--recipe", path,
            "--run-id", "20260101T000000Z-123456789abc"
        ]
        rc, out = self._run_cli(argv)
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), "mock-job-999")
        # The id still reaches stdout when the first status read fails; why
        # it failed goes to stderr rather than an unobserved PENDING.
        with patch.object(MockTestDriver,
                          "get_status",
                          side_effect=RuntimeError("cluster unreachable")), \
                io.StringIO() as err, contextlib.redirect_stderr(err):
            rc, out = self._run_cli(argv)
            self.assertIn(
                "Could not read initial status: RuntimeError: cluster "
                "unreachable", err.getvalue())
        self.assertEqual((rc, out.strip()), (0, "mock-job-999"))

    def test_submit_and_job_lifecycle_subcommands(self):
        """Verify submit, job status, job logs, and job cancel all route through."""
        path = self._script("print('submitted')\n")
        mock = ["compute", "--backend", "mock_test"]
        job = ["--job-id", "mock-job-999"]
        # A real driver's submit leaves a record, which cancel checks.
        with temp_home():
            jobs.JobRecord.create(job_id="mock-job-999", backend="mock_test")
            for argv, key, expected in (
                (["submit", "--recipe", path, "--run-id",
                  _RUN], "job_id", "mock-job-999"),
                (["job", "status", *job], "status", "RUNNING"),
                (["job", "logs", *job], "logs", "mock logs for mock-job-999"),
                (["job", "cancel", *job], "status", "CANCELLED"),
            ):
                rc, out = self._run_cli([*mock, *argv, "--json"])
                self.assertEqual((rc, json.loads(out)[key]), (0, expected))
        # Without --json, logs are plain text.
        rc, out = self._run_cli([*mock, "job", "logs", *job])
        self.assertEqual(out.strip(), "mock logs for mock-job-999")


class TestComputeCliContracts(ScriptMixin, unittest.TestCase):
    """handle_compute_cli behaviour reachable without the runtime parser."""

    def _args(self, **overrides: Any) -> argparse.Namespace:
        return cli_args(**{"job_id": "job-1", "out_dir": None, **overrides})

    def _submit_args(self, recipe: str,
                     **overrides: Any) -> argparse.Namespace:
        """`compute submit` of `recipe`, unnamed and without extra args."""
        return self._args(
            **{
                "compute_action": "submit",
                "recipe": recipe,
                "name": None,
                "submit_args": [],
                "interpreter": None,
                **overrides
            })

    def _invoke(self, args: argparse.Namespace, driver: Any):
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            rc = compute.handle_compute_cli(args, resolve=lambda **_: driver)
            return rc, buf.getvalue()

    def test_missing_tool_is_a_structured_error_not_a_traceback(self):
        """A missing kubectl crashed inspect with a traceback."""
        driver = MagicMock(required_tools=base.GKE_TOOLS)
        with patch.object(base.shutil, "which", return_value=None):
            with self.assertRaises(ComputeConfigError) as ctx:
                self._invoke(self._args(), driver)
            self.assertEqual(ctx.exception.error_code, "MISSING_TOOL")
            self.assertIn("gcloud components install kubectl",
                          ctx.exception.message)
            self.assertIn("gke-gcloud-auth-plugin", ctx.exception.message)
            driver.inspect.assert_not_called()
            # Preflight reports the tools itself, so it still runs.
            driver.preflight.return_value.to_dict.return_value = {
                "status": "fail"
            }
            self._invoke(self._args(compute_action="preflight"), driver)
            driver.preflight.assert_called_once()

    def test_job_cancel_only_deletes_this_runs_jobs(self):
        """`job cancel` only cancels jobs this run submitted."""
        mine, other = _RUN, "20261007T150000Z-ba9876543210"
        driver = MagicMock()

        def cancel(job_id, run_id=mine):
            return self._invoke(
                self._args(compute_action="job",
                           job_action="cancel",
                           job_id=job_id,
                           run_id=run_id), driver)

        with temp_home():
            jobs.JobRecord.create(job_id="theirs", run_id=other)
            jobs.JobRecord.create(job_id="ours", run_id=mine)
            for job_id, why in (("someone-elses", "tell them compute can't"),
                                ("theirs", f"belongs to run {other}")):
                with self.subTest(job_id), self.assertRaises(
                        ComputeConfigError) as ctx:
                    cancel(job_id)
                self.assertEqual(ctx.exception.error_code,
                                 compute.cli.NOT_OWN_JOB)
                self.assertIn(why, ctx.exception.message)
            driver.cancel.assert_not_called()
            # This run's job, and the other run's when it is named outright.
            for job_id, run_id in (("ours", mine), ("theirs", other)):
                rc, out = cancel(job_id, run_id)
                self.assertEqual((rc, json.loads(out)), (0, {
                    "job_id": job_id,
                    "status": "CANCELLED"
                }))
                driver.cancel.assert_called_with(job_id)

    def test_job_action_requires_job_id(self):
        """status, logs, artifacts, cancel require --job-id.

        Raised as a ComputeConfigError so the runtime's one error handler
        renders it -- JSON under --json, prose otherwise -- like every
        other configuration failure.
        """
        driver = MagicMock()
        for action in ("status", "logs", "artifacts", "cancel"):
            with self.subTest(action):
                with self.assertRaises(ComputeConfigError) as ctx:
                    self._invoke(
                        self._args(compute_action="job",
                                   job_action=action,
                                   job_id=None), driver)
                self.assertEqual(ctx.exception.error_code, "MISSING_FIELD")
                self.assertIn("--job-id", str(ctx.exception))

    def test_job_artifacts_fails_when_a_declared_output_never_arrived(self):
        """Exit 0 with nothing to read is what makes an agent invent numbers.

        What did arrive is reported where it landed, not at a submit-time
        ':LOCAL', which is where agents used to look for it.
        """

        def fetch(jid, dest):
            (dest / "r.json").write_text("{}")
            return [dest / "r.json"]

        driver = MagicMock()
        driver.fetch_artifacts.side_effect = fetch
        declared = [{
            "remote": "r.json",
            "local": "x/r.json"
        }] + [{
            "remote": "summary.json",
            "local": "summary.json"
        }]
        with temp_home() as home:
            jobs.JobRecord.create(job_id="job-1", extra={"outputs": declared})
            rc, out = self._invoke(
                self._args(compute_action="job",
                           job_action="artifacts",
                           out_dir=str(home)), driver)
        payload = json.loads(out)
        self.assertEqual((rc, payload["missing_outputs"]),
                         (1, ["summary.json"]))
        self.assertEqual(payload["outputs"], {"r.json": str(home / "r.json")})

    def test_submit_maps_outputs_and_warns_that_local_is_ignored(self):
        """Only a spec with a ':LOCAL' warns; every output is mapped."""
        driver = MagicMock()
        driver.submit.return_value = "job-s"
        driver.get_status.return_value = AsynchronousExecutionResult(
            "job-s", "RUNNING")
        run_id = "20260101T000000Z-cccccccccccc"
        with temp_home() as tmp, \
                patch.object(compute.cli.lease, "spawn_watcher"), \
                io.StringIO() as err, contextlib.redirect_stderr(err):
            recipe = tmp / "recipe.py"
            recipe.write_text("print('hi')\n")
            rc, out = self._invoke(
                self._submit_args(str(recipe),
                                  name="s",
                                  run_id=run_id,
                                  output=["results.json:out/r.json",
                                          "a.json"]), driver)
            stderr = err.getvalue()
            landing = paths.get_run_artifacts_dir(run_id) / "job-s"
        self.assertEqual(rc, 0)
        self.assertEqual(
            json.loads(out)["outputs"], {
                "results.json": str(landing / "results.json"),
                "a.json": str(landing / "a.json")
            })
        self.assertIn(
            f"ignores ':out/r.json'; results.json will land at "
            f"{landing / 'results.json'}", stderr)
        self.assertEqual(stderr.count("warning:"), 1)

    def test_job_artifacts_defaults_to_run_artifact_dir(self):
        """When --out-dir is omitted, artifacts default to {{runs}}/<run_id>/artifacts/<job_id>/."""
        driver = MagicMock()
        driver.fetch_artifacts.side_effect = lambda jid, dest: []
        run_id = "20260101T000000Z-123456789abc"

        def out_dir(job_id: str, **flags: Any) -> str:
            rc, out = self._invoke(
                self._args(compute_action="job",
                           job_action="artifacts",
                           job_id=job_id,
                           **flags), driver)
            self.assertEqual(rc, 0)
            return json.loads(out)["out_dir"]

        with temp_home() as home:
            jobs.JobRecord.create(job_id="job-42", run_id=run_id)
            expected = paths.get_run_artifacts_dir(run_id) / "job-42"
            self.assertEqual(out_dir("job-42"), str(expected))
            self.assertTrue(expected.is_dir())
            # No run anywhere: cwd / job_id, so two jobs still cannot collide.
            jobs.JobRecord.create(job_id="job-no-run", run_id="")
            with contextlib.chdir(home):
                self.assertEqual(
                    out_dir("job-no-run", run_id=None, workspace=home),
                    str(Path.cwd() / "job-no-run"))

    def test_job_list_and_reap(self):
        """Verify compute job list filters records and compute job reap cancels active jobs."""
        run_a = "20260101T000000Z-aaaaaaaaaaaa"

        def invoke(action, driver=None, **flags):
            flags = {"run_id": None, "status": None, **flags}
            rc, out = self._invoke(
                self._args(compute_action="job", job_action=action, **flags),
                driver or MagicMock())
            self.assertEqual(rc, 0)
            return out

        def listed(**flags) -> set[str]:
            return {r["job_id"] for r in json.loads(invoke("list", **flags))}

        with temp_home():
            for job_id, run_id, status in (("job-101", run_a, "RUNNING"),
                                           ("job-102", run_a, "COMPLETED"),
                                           ("job-103", "20260101T000000Z-b",
                                            "RUNNING")):
                jobs.JobRecord.create(job_id=job_id,
                                      job_name="train",
                                      run_id=run_id).mark(status)
            self.assertEqual(listed(run_id=run_a), {"job-101", "job-102"})
            self.assertEqual(listed(status="running"), {"job-101", "job-103"})
            self.assertIn("train", invoke("list", json=False))

            # A driver's cancel records the outcome; the reaper reads it back.
            driver = MagicMock()
            driver.cancel.side_effect = (
                lambda jid: jobs.JobRecord.load(jid).mark("CANCELLED"))
            payload = json.loads(invoke("reap", driver, run_id=run_a))
            self.assertEqual((payload["run_id"], payload["reaped"]),
                             (run_a, ["job-101"]))
            driver.cancel.assert_called_once_with("job-101")

    def test_job_list_refreshes_the_active_runs_unfinished_jobs(self):
        """`job list` first asks the backend of each unfinished job in the run.

        A record says PENDING from submit until the job ends unless a status
        read updates it. A backend that cannot answer is asked only once.
        """
        replies: dict[str, Any] = {}

        def status(jid: str) -> AsynchronousExecutionResult:
            reply = replies.get(jid,
                                AsynchronousExecutionResult(jid, "RUNNING"))
            if isinstance(reply, Exception):
                raise reply
            return reply

        driver = MagicMock()
        driver.get_status.side_effect = status
        with temp_home() as tmp:
            paths.record_active_run(_RUN, workspace=tmp)
            # A target is a backend and profile: "c" is on a second one.
            for jid, profile in (("a", None), ("b", None), ("c", "other")):
                jobs.JobRecord.create(job_id=jid,
                                      run_id=_RUN,
                                      backend="x",
                                      compute_profile=profile)
            jobs.JobRecord.create(job_id="done", run_id=_RUN,
                                  backend="x").mark("COMPLETED")
            jobs.JobRecord.create(job_id="old-run",
                                  run_id="20260101T000000Z-000000000000",
                                  backend="x")

            def listed(**flags: Any) -> dict[str, Any]:
                driver.get_status.reset_mock()
                rc, out = self._invoke(
                    self._args(compute_action="job",
                               job_action="list",
                               run_id=None,
                               status=None,
                               workspace=tmp,
                               **flags), driver)
                self.assertEqual(rc, 0)
                return {r["job_id"]: r for r in json.loads(out)}

            self.assertEqual(listed(no_refresh=True)["a"]["status"], "PENDING")
            driver.get_status.assert_not_called()
            # Only the active run's unfinished jobs are asked about; an older
            # run's record may name a cluster today's config does not.
            rows = listed()
            self.assertEqual([rows[j]["status"] for j in rows],
                             ["RUNNING"] * 3 + ["COMPLETED", "PENDING"])
            self.assertEqual(driver.get_status.call_count, 3)
            # A target that cannot answer -- as UNKNOWN with an error, or by
            # raising -- is asked once, and every job on it says so.
            for jid in "abc":
                jobs.JobRecord.load(jid).mark("PENDING")
            timed_out = AsynchronousExecutionResult("a",
                                                    "UNKNOWN",
                                                    error="ssh: timed out")
            replies.update(a=timed_out, b=timed_out, c=RuntimeError("no vm"))
            rows = listed()
            self.assertEqual(driver.get_status.call_count, 2)
            self.assertEqual(rows["b"]["status"], "PENDING")
            self.assertIn("ssh: timed out", rows["b"]["refresh_error"])
            self.assertIn("no vm", rows["c"]["refresh_error"])
            self.assertNotIn("refresh_error", rows["old-run"])
            # A Job deleted outside compute is its own error: the target
            # answered for the rest.
            replies.update(a=ComputeConfigError(
                "deleted", error_code=compute.cli.JOB_GONE),
                           b=AsynchronousExecutionResult("b", "RUNNING"))
            rows = listed()
            self.assertEqual(rows["a"]["refresh_error"], "deleted")
            self.assertEqual(rows["b"]["status"], "RUNNING")

    def test_job_reap_by_lease(self):
        """`--orphaned` needs no run, `--watch` one.

        Both are tested in test_lease.
        """
        resolve = MagicMock()

        def reap(**flags):
            flags.setdefault("run_id", None)
            args = self._args(compute_action="job", job_action="reap", **flags)
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                self.assertEqual(compute.handle_compute_cli(args, resolve), 0)
                return json.loads(buf.getvalue())

        sweep = ({"run-a": ["job-1"]}, {})
        with patch.object(compute.lease,
                          "reap_orphaned_jobs",
                          return_value=sweep):
            self.assertEqual(reap(orphaned=True)["reaped"], sweep[0])
        watched = ([], {"job-2": "x"})
        with patch.object(compute.lease, "watch",
                          return_value=watched) as watch:
            self.assertEqual(
                reap(watch=True, run_id="run-b")["failed"], watched[1])
        watch.assert_called_once_with("run-b", resolve)

    def test_submit_after_the_session_ended(self):
        """A batch job starts its run's watcher.

        An ended session's active run is no fallback, as a job tagged with it
        would be reaped.
        """
        driver = MagicMock()
        driver.submit.return_value = "job-1"
        driver.get_status.return_value = AsynchronousExecutionResult(
            job_id="job-1", status="RUNNING")
        run_id = "20260101T000000Z-123456789abc"
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        paths.record_active_run(run_id, workspace=tmp)

        def submit(**overrides):
            return self._invoke(
                self._submit_args(self._script("echo hi\n", suffix=".sh"),
                                  workspace=str(tmp),
                                  **overrides), driver)[0]

        with patch.object(compute.lease, "spawn_watcher") as spawn, \
                patch.object(compute.lease, "has_ended", return_value=True):
            with self.assertRaisesRegex(base.ComputeConfigError, "has ended"):
                submit(run_id=None)
            self.assertEqual(submit(run_id=run_id), 0)
            self.assertEqual(submit(run_id=run_id, serve=True), 0)
        spawn.assert_called_once_with(run_id)

    def test_port_forward_cli_dispatches_to_driver(self):
        """Verify compute port-forward parses [LOCAL:]REMOTE and calls driver."""
        driver = MagicMock()
        for port, job_id, background, local in (("8000", None, False, 8000),
                                                ("8080:8000", "job-4", True,
                                                 8080)):
            driver.port_forward.return_value = 12345 if background else 0
            rc, out = self._invoke(
                self._args(compute_action="port-forward",
                           port=port,
                           job_id=job_id,
                           background=background), driver)
            self.assertEqual(rc, 0)
            driver.port_forward.assert_called_with(local_port=local,
                                                   remote_port=8000,
                                                   job_id=job_id,
                                                   background=background)
        self.assertIn("12345", out)

    def test_port_forward_rejects_a_port_that_is_not_one(self):
        """Rejected here, not deep inside kubectl.

        There, the message names neither the flag nor the number.
        """
        driver = MagicMock()
        for spec in ("0", "-1", "99999", "8080:70000", "not_a_port", "8080:x"):
            with self.assertRaises(compute.ComputeConfigError) as ctx:
                self._invoke(
                    self._args(
                        compute_action="port-forward",
                        port=spec,
                        job_id=None,
                    ),
                    driver,
                )
            self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")
        driver.port_forward.assert_not_called()

    def test_port_forward_gce_tpu_vm(self):
        driver = compute.GceTpuVmDriver(tpu_name="v4-node",
                                        zone="us-central2-b",
                                        project="test-proj")
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[],
                                                                returncode=0)
            rc = driver.port_forward(local_port=8080, remote_port=8000)
            self.assertEqual(rc, 0)
            cmd = mock_run.call_args[0][0]
            self.assertIn("-L8080:localhost:8000", cmd)
            self.assertIn("-N", cmd)

    def test_port_forward_local_refuses_a_remapping(self):
        """Nothing would be listening on the local port it was asked for."""
        driver = compute.LocalDriver()
        with self.assertRaises(compute.ComputeConfigError) as ctx:
            driver.port_forward(local_port=9000,
                                remote_port=8000,
                                background=True)
        self.assertEqual(ctx.exception.error_code, "UNSUPPORTED_OPERATION")

    def test_invalid_run_id_fails_before_the_driver_is_asked(self):
        """A malformed --run-id is INVALID_FIELD, and nothing is dispatched.

        Checked after the dispatch, it left a job running on the target
        that no run could find, and the job id was never printed.
        """
        driver = MagicMock()
        script = self._script("print('hi')\n")
        cases = [
            self._args(compute_action="run",
                       code=script,
                       timeout=None,
                       run_args=[],
                       interpreter=None),
            self._submit_args(script),
        ]
        for args in cases:
            with self.subTest(args.compute_action), \
                    patch.object(compute.lease, "spawn_watcher") as spawn:
                with self.assertRaises(ComputeConfigError) as ctx:
                    args.run_id = "test-run1"
                    self._invoke(args, driver)
                self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")
                self.assertIn("invalid run ID", ctx.exception.message)
                spawn.assert_not_called()
        driver.run.assert_not_called()
        driver.submit.assert_not_called()

    def test_submit_without_any_run_is_refused(self):
        """No --run-id and no active run: refuse rather than tag with ''.

        A job tagged with no run is one the run-end reaper can never find.
        """
        with temp_home() as tmp_path:
            recipe_path = tmp_path / "recipe.py"
            recipe_path.write_text("print('hi')\n")
            with self.assertRaises(ComputeConfigError) as ctx:
                self._invoke(
                    self._submit_args(
                        str(recipe_path),
                        name="untagged",
                        run_id=None,
                        workspace=tmp_path,
                    ),
                    LocalDriver(),
                )
            self.assertEqual(ctx.exception.error_code, "MISSING_FIELD")
            self.assertIn("--run-id", str(ctx.exception))
            # The directory the search started from, not the cwd.
            self.assertIn(str(tmp_path), str(ctx.exception))
            self.assertEqual(jobs.list_jobs(), [])

    def test_two_sessions_in_one_workspace_each_keep_their_own_run(self):
        """A second session's start overwrites the active-run marker.

        The marker is the workspace's; the lease is per harness process, so
        this session's `job list` still refreshes its own jobs, not the other
        session's.
        """
        mine, theirs = _RUN, "20260101T000000Z-bbbbbbbbbbbb"
        driver = MagicMock()
        driver.get_status.side_effect = (
            lambda jid: AsynchronousExecutionResult(jid, "RUNNING"))
        with temp_home() as ws, contextlib.chdir(ws), \
                patch.object(compute.cli.lease, "session_run",
                             return_value=(mine, ws)):
            paths.record_active_run(theirs, workspace=ws)
            for job_id, run_id in (("mine", mine), ("theirs", theirs)):
                jobs.JobRecord.create(job_id=job_id,
                                      run_id=run_id,
                                      backend="x")
            rc, out = self._invoke(
                self._args(compute_action="job",
                           job_action="list",
                           run_id=None,
                           status=None), driver)
        self.assertEqual((rc, {
            r["job_id"]: r["status"]
            for r in json.loads(out)
        }), (0, {
            "mine": "RUNNING",
            "theirs": "PENDING"
        }))
        driver.get_status.assert_called_once_with("mine")


class _ScriptedJob:
    """A driver whose job moves through `states`, one per status read.

    A state is a status, a (status, error) pair, a (status, error, details)
    triple, or an exception the read raises. `logs` is the text `fetch_logs`
    returns, or a function of how many reads there have been, for logs that
    grow as the job runs. Each read advances `clock` by `poll_sec`.
    """

    def __init__(self, states: list[Any], logs: Any = "") -> None:
        self.states = states
        self.logs = logs
        self.reads = 0
        self.clock = [0.0]
        self.poll_sec = 0.0

    def get_status(self, job_id: str) -> AsynchronousExecutionResult:
        state = self.states[min(self.reads, len(self.states) - 1)]
        self.reads += 1
        self.clock[0] += self.poll_sec
        if isinstance(state, Exception):
            raise state
        details: dict[str, Any] = {}
        if isinstance(state, tuple):
            status, error, *rest = state
            if rest:
                details = rest[0]
        else:
            status, error = state, None
        return AsynchronousExecutionResult(job_id=job_id,
                                           status=status,
                                           error=error,
                                           details=details)

    def fetch_logs(self, job_id: str) -> str:
        return self.logs(self.reads) if callable(self.logs) else self.logs


class TestJobWait(unittest.TestCase):
    """`compute job wait`: one exit code per outcome.

    That way agents stop writing sleep loops that outlive the shell or miss
    a crash.
    """

    def wait(self, job: _ScriptedJob, **flags: Any) -> tuple[int, dict]:
        args = cli_args(
            **{
                "compute_action": "job",
                "job_action": "wait",
                "job_id": "serve-1",
                "until": None,
                "fail_on_log": None,
                "no_default_fail_patterns": False,
                "timeout": None,
                **flags
            })
        cli = sys.modules[compute.handle_compute_cli.__module__]
        clock = job.clock

        def sleep(sec: float) -> None:
            clock[0] += sec

        with io.StringIO() as buf, contextlib.redirect_stdout(buf), \
                contextlib.redirect_stderr(io.StringIO()), \
                patch.object(cli.time, "sleep", sleep), \
                patch.object(cli.time, "monotonic", lambda: clock[0]):
            rc = compute.handle_compute_cli(args, resolve=lambda **_: job)
            return rc, json.loads(buf.getvalue())

    def test_each_outcome_has_its_exit_code(self):
        on_ready = dict(until="log:READY")

        def ready_on_read_2(first: str):
            return lambda reads: first + ("READY\n" if reads >= 2 else "")

        cases = [
            # (states, logs, flags, exit code, outcome, line)
            (["PENDING",
              "RUNNING"], "", dict(until="running"), 0, "reached", None),
            (["RUNNING"], ready_on_read_2("loading\n"), on_ready, 0, "reached",
             "READY"),
            (["RUNNING", "COMPLETED"], "", {}, 0, "reached", None),
            # A warning-like line does not fail a job that succeeded.
            (["COMPLETED"], "No space left on device", {}, 0, "reached", None),
            # The run that went on for 8 minutes: the compile cache filled
            # /dev/shm while the Job still read RUNNING.
            (["RUNNING"], "RESOURCE_EXHAUSTED: write", on_ready, 2,
             "failure_logged", "RESOURCE_EXHAUSTED: write"),
            (["RUNNING"], "HBM OOM", dict(on_ready, fail_on_log=["OOM"]), 2,
             "failure_logged", "HBM OOM"),
            # A TP=8 attempt failed, then TP=4 came up: the server is up.
            (["RUNNING"], "RuntimeError: cancelled\nREADY", on_ready, 0,
             "reached", "READY"),
            (["RUNNING"], ready_on_read_2("RuntimeError: cancelled\n"),
             on_ready, 2, "failure_logged", "RuntimeError: cancelled"),
            (["RUNNING"], ready_on_read_2("RuntimeError: cancelled\n"),
             dict(on_ready,
                  no_default_fail_patterns=True), 0, "reached", "READY"),
            # Up, then dead before the wait looked: not up.
            (["FAILED"], "READY\nRaising signal 7", on_ready, 4, "job_ended",
             "Raising signal 7"),
            (["COMPLETED"], "bye", on_ready, 4, "job_ended", None),
            (["PENDING"], "", dict(timeout=0), 3, "timed_out", None),
        ]
        for states, logs, flags, code, outcome, line in cases:
            with self.subTest(states=states, logs=logs, **flags):
                rc, result = self.wait(_ScriptedJob(states, logs), **flags)
                self.assertEqual((rc, result["outcome"], result.get("line")),
                                 (code, outcome, line))

    def test_failed_reads_are_retried_then_raised(self):
        # An API blip does not end the wait; a job no backend knows does,
        # with the runtime's error exit rather than a nine-minute timeout.
        rc, _ = self.wait(_ScriptedJob([RuntimeError("blip"), "RUNNING"]),
                          until="running")
        self.assertEqual(rc, 0)
        job = _ScriptedJob([("UNKNOWN", "Job not found")])
        with self.assertRaises(ComputeConfigError) as ctx:
            self.wait(job)
        self.assertIn("Job not found", ctx.exception.message)
        self.assertEqual(job.reads, 3)

    def test_a_long_pending_job_is_told_why_before_waiting_again(self):
        # An unqueued pod on a full cluster, past the ceiling.
        cli = sys.modules[compute.handle_compute_cli.__module__]
        long = cli.WAIT_PENDING_GIVE_UP_SEC + 180  # the wait stops a poll early
        reason = "0/15 nodes are available: 8 Insufficient google.com/tpu."
        job = _ScriptedJob([("PENDING", reason, {
            "pending_reason": base.PENDING_UNSCHEDULED
        })])
        job.poll_sec = 60
        _, result = self.wait(job, timeout=long)
        self.assertGreaterEqual(result["since_submit_sec"],
                                cli.WAIT_PENDING_GIVE_UP_SEC)
        self.assertIn("job cancel --job-id serve-1", result["next"])
        self.assertIn(reason, result["next"])

        # Same ceiling, but queued, admitted, starting, running, or a backend
        # that gives no reason: wait.
        kept = [("PENDING", None, {
            "pending_reason": r
        }) for r in (base.PENDING_QUEUED, base.PENDING_ADMITTED_NO_NODE,
                     base.PENDING_STARTING)] + ["RUNNING", "PENDING"]
        for state in kept:
            with self.subTest(state=state):
                job = _ScriptedJob([state])
                job.poll_sec = 60
                _, result = self.wait(job, until="log:READY", timeout=long)
                self.assertIn("run the same `job wait` again", result["next"])
                self.assertNotIn("job cancel", result["next"])

    def test_bad_until_is_rejected(self):
        for until in ("ready", "log:", "log:(unclosed"):
            with self.assertRaises(ComputeConfigError):
                self.wait(_ScriptedJob(["RUNNING"]), until=until)


class TestReapJobs(unittest.TestCase):
    """`reap_jobs` hands every live job to its own backend's driver."""

    RUN = "20260101T000000Z-000000000001"

    def setUp(self):
        patch_home(self)

    def _live(self,
              job_id: str,
              backend: str,
              serve: bool = False) -> jobs.JobRecord:
        rec = jobs.JobRecord.create(job_id=job_id,
                                    job_name=job_id,
                                    run_id=self.RUN,
                                    backend=backend,
                                    serve=serve)
        return rec.mark("RUNNING")

    def test_serving_jobs_are_exempt_from_reap_unless_included(self):
        """Session end leaves a serve=True job running, and reports it.

        `job reap --include-serving` is a decision to clear the run.
        """
        self._live("job-serve-1", "local", serve=True)
        self._live("job-old-1", "local", serve=True).mark("COMPLETED")
        reaped, failed = jobs.reap_jobs(self.RUN, lambda _: LocalDriver())
        self.assertEqual((reaped, failed), ([], {}))
        self.assertEqual(jobs.JobRecord.load("job-serve-1").status, "RUNNING")
        self.assertEqual(jobs.serving_jobs(self.RUN), ["job-serve-1"])
        reaped, failed = jobs.reap_jobs(self.RUN,
                                        lambda _: LocalDriver(),
                                        include_serving=True)
        self.assertEqual((reaped, failed), (["job-serve-1"], {}))
        self.assertEqual(
            jobs.JobRecord.load("job-serve-1").status, "CANCELLED")


class TestContractGeneration(unittest.TestCase):
    """The generated agent contract must reflect the real registry."""

    def test_contract_matches_the_job_cli(self):
        """Every `job` action is in the section, and its flags are on `job`.

        `job wait` shipped in the CLI but not in this section, so agents wrote
        their own sleep loops. Every action the `job` parser accepts must be
        listed, and every flag the section names for `job` must exist on it.
        """
        parser = runtime.build_runtime_parser()
        for name in ("compute", "job"):
            parser = next(
                a for a in parser._actions
                if isinstance(a, argparse._SubParsersAction)).choices[name]
        section = prompt_contracts.compute_contract_section(
            "RUNTIME", "pkg@1.0.0")
        job_lines = " ".join(line for line in section.splitlines()
                             if "pkg@1.0.0 job " in line)
        choices = next(a.choices for a in parser._actions
                       if a.dest == "job_action")
        for action in choices:
            self.assertRegex(job_lines,
                             rf"job (\[[^\]]*\b{action}\b|{action}\b)")
        options = {o for a in parser._actions for o in a.option_strings}
        for flag in set(re.findall(r"--[a-z][a-z-]*",
                                   job_lines)) - {"--package"}:
            self.assertIn(flag, options)


if __name__ == "__main__":
    unittest.main()
