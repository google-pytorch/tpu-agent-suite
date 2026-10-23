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
"""Tests for gke_job file staging: GCS, the log fallback, and tokens."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

# Imports below follow this sys.path setup, so the file runs directly.
# ruff: noqa: E402
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.compute import ComputeConfigError
from framework.compute.drivers import base, gcs_staging, gke_job, jobs
from framework.tests.compute.gke_job.helpers import (_fake_google_auth, _KeyCreds,
                                                     FakeKubectl, hello_script,
                                                     job_json, job_store,
                                                     make_driver,
                                                     set_up_module)

Path = pathlib.Path


def setUpModule():  # pylint: disable=invalid-name
    set_up_module()


# Taken before setUpModule stubs it out, for the test of the token's rules.
_REAL_MINT_BUCKET_TOKEN = gcs_staging.mint_bucket_token


def workload_command(container: dict[str, Any]) -> list[str]:
    """The container's command behind the cache prologue, if it has one."""
    command = container["command"]
    if command[:2] == ["sh", "-c"] and command[3] == "sh":
        return command[4:]
    return command


try:
    from google.auth import exceptions as _google_auth_errors
except ImportError:
    _google_auth_errors = None


_DONE_POD = json.dumps({"items": [{"status": {"phase": "Succeeded"}}]})


class _UserCreds:
    """A person's ADC login: a token, and no way to sign."""
    token = "mock-user-token"


class TestGkeJobStaging(unittest.TestCase):
    """File staging via GCS and signed URLs on gke_job."""

    def setUp(self):
        # Dispatch writes job records; none of them belong in the real store.
        self.enterContext(job_store())

    @staticmethod
    def _client_with(creds: Any) -> tuple[MagicMock, MagicMock]:
        client = MagicMock()
        client._credentials = creds
        blob = MagicMock()
        client.bucket.return_value.blob.return_value = blob
        blob.generate_signed_url.return_value = (
            "https://storage.googleapis.com/test-signed-url")
        return client, blob

    def test_iam_signing_failure_names_the_missing_role(self):
        """A GCE/GKE identity has no key; it is refreshed for its real
        email and token, and those go to the IAM signBlob path."""

        class GceCreds:
            service_account_email = "default"
            token = None

            def refresh(self, req):
                del req
                self.service_account_email = "my-sa@proj.iam.gserviceaccount.com"
                self.token = "fake-token"

        creds = GceCreds()
        client, blob = self._client_with(creds)
        blob.generate_signed_url.side_effect = Exception(
            "403 Forbidden: Permission denied")
        with _fake_google_auth(creds), \
                self.assertRaises(ComputeConfigError) as ctx:
            gcs_staging.generate_v4_signed_url("my-bucket",
                                               "in/data.txt",
                                               client=client)
        kwargs = blob.generate_signed_url.call_args.kwargs
        self.assertEqual(kwargs["service_account_email"],
                         "my-sa@proj.iam.gserviceaccount.com")
        self.assertEqual(kwargs["access_token"], "fake-token")
        self.assertEqual(ctx.exception.error_code, "SIGNING_FAILED")
        self.assertIn("serviceAccountTokenCreator", str(ctx.exception))
        self.assertIn("my-sa@proj.iam.gserviceaccount.com", str(ctx.exception))

    @contextlib.contextmanager
    def _fake_storage(self, creds: Any):
        client, blob = self._client_with(creds)
        storage = types.ModuleType("google.cloud.storage")
        storage.Client = MagicMock(return_value=client)
        cloud = types.ModuleType("google.cloud")
        cloud.storage = storage
        with _fake_google_auth(creds), patch.dict(
                sys.modules, {
                    "google.cloud": cloud,
                    "google.cloud.storage": storage
                }):
            yield client, blob

    def test_preflight_signs_and_lists_the_bucket(self):
        with self._fake_storage(_KeyCreds()) as (client, blob):
            checks = gcs_staging.preflight_checks("b")
        self.assertEqual([(c.name, c.status) for c in checks],
                         [("staging:credentials", "ok"),
                          ("staging:signing", "ok"), ("staging:bucket", "ok")])
        blob.upload_from_filename.assert_not_called()
        client.list_blobs.assert_called_once_with("b",
                                                  prefix=None,
                                                  max_results=1)
        # A folder in the bucket is listed as one, not as a bucket name. A
        # staging credential is used as given, never swapped for ADC.
        with self._fake_storage(_KeyCreds()) as (client, _):
            storage = sys.modules["google.cloud.storage"]
            with patch("google.auth.default") as default:
                checks = gcs_staging.preflight_checks("gs://b/runs/",
                                                      credentials=_KeyCreds())
        client.list_blobs.assert_called_once_with("b",
                                                  prefix="runs/",
                                                  max_results=1)
        default.assert_not_called()
        self.assertIsInstance(storage.Client.call_args.kwargs["credentials"],
                              _KeyCreds)
        self.assertEqual(checks[0].status, "ok")

    def test_preflight_does_not_ask_a_pod_token_setup_to_sign(self):
        """With pod_service_account nothing is signed, so a
        person's login, which cannot sign, is not a warning."""
        with self._fake_storage(_UserCreds()) as (_, blob):
            checks = gcs_staging.preflight_checks("b",
                                                  pod_service_account="ksa")
        self.assertEqual([c.status for c in checks], ["ok"] * 3)
        blob.generate_signed_url.assert_not_called()

    def test_preflight_reports_credentials_that_cannot_sign(self):
        with self._fake_storage(_UserCreds()) as (client, _):
            error = Exception("403 no list")
            error.code = 403
            client.list_blobs.side_effect = error
            checks = {c.name: c for c in gcs_staging.preflight_checks("b")}
        self.assertEqual(checks["staging:signing"].status, "fail")
        self.assertIn("--impersonate-service-account",
                      checks["staging:signing"].fix)
        # Forbidden to list is not proof the bucket is missing.
        self.assertEqual(checks["staging:bucket"].status, "warn")

    def test_staging_credential_validation(self):
        self.assertIsNone(gcs_staging.staging_credentials(None, None))
        self.assertIsNone(gcs_staging.staging_credentials(" ", ""))
        for fn in (
                lambda: gcs_staging.staging_credentials(
                    "sa@p.iam.gserviceaccount.com", "/k.json"),
                lambda: gke_job.GkeJobDriver.from_params({
                    gcs_staging.SERVICE_ACCOUNT_PARAM:
                    "sa@p.iam.gserviceaccount.com",
                    gcs_staging.KEY_FILE_PARAM:
                    "/k.json",
                }),
                lambda: gcs_staging.staging_credentials(
                    key_file="/nonexistent/key.json"),
        ):
            with self.assertRaises(ComputeConfigError) as ctx:
                fn()
            self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")
        self.assertIn(gcs_staging.KEY_FILE_PARAM, str(ctx.exception))
        driver = make_driver(
            staging_bucket="gs://b",
            staging_service_account_key="/nonexistent/key.json")
        self.assertEqual(driver.storage.client_kwargs(best_effort=True), {})
        with self.assertRaises(ComputeConfigError):
            driver.storage.client_kwargs()

    @unittest.skipUnless(_google_auth_errors, "needs google-auth")
    def test_impersonated_staging_credential_signs_locally(self):
        """Impersonation from ADC, built without touching the environment."""
        from google.auth import impersonated_credentials  # noqa: PLC0415

        source = MagicMock()
        with patch("google.auth.default",
                   return_value=(source,
                                 "proj")) as default, patch.dict(os.environ,
                                                                 {},
                                                                 clear=False):
            os.environ.pop(base.GAC_ENV, None)
            creds = gcs_staging.staging_credentials(
                "stager@p.iam.gserviceaccount.com")
            self.assertNotIn(base.GAC_ENV, os.environ)
        default.assert_called_once()
        self.assertIsInstance(creds, impersonated_credentials.Credentials)
        self.assertEqual(creds.signer_email,
                         "stager@p.iam.gserviceaccount.com")
        self.assertEqual(gcs_staging._signing_kwargs(creds), {})
        client, blob = self._client_with(creds)
        blob.generate_signed_url.side_effect = Exception("403 signBlob denied")
        with self.assertRaises(ComputeConfigError) as ctx:
            gcs_staging.generate_v4_signed_url("b", "x", client=client)
        self.assertEqual(ctx.exception.error_code, "SIGNING_FAILED")
        self.assertIn("serviceAccountTokenCreator", str(ctx.exception))
        self.assertIn("stager@p.iam.gserviceaccount.com", str(ctx.exception))
        with self.assertRaises(ComputeConfigError) as ctx:
            with gcs_staging._gcs_errors("upload to", "gs://b/x", client):
                raise _google_auth_errors.RefreshError("getAccessToken denied")
        self.assertEqual(ctx.exception.error_code, "GCS_FORBIDDEN")
        self.assertIn("serviceAccountTokenCreator", str(ctx.exception))

    @unittest.skipUnless(_google_auth_errors, "needs google-auth")
    def test_credentials_that_will_not_refresh_cannot_sign(self):
        """An ambient identity whose token cannot be had, off GCE or with
        no network: a ComputeConfigError, which callers fall back on."""

        class AmbientCreds:
            service_account_email = "default"
            token = None

            def refresh(self, request):
                raise _google_auth_errors.RefreshError("metadata unreachable")

        client, _ = self._client_with(AmbientCreds())
        with self.assertRaises(ComputeConfigError) as ctx:
            gcs_staging.generate_v4_signed_url("b", "x", client=client)
        self.assertEqual(ctx.exception.error_code, "SIGNING_FAILED")
        self.assertIn("metadata unreachable", str(ctx.exception))

    @staticmethod
    def _script_and_input(directory: str) -> tuple[Path, Path]:
        """A workload script and one input file for it, in `directory`."""
        script_path = Path(directory) / "script.py"
        script_path.write_text("print('test')")
        in_file = Path(directory) / "input.dat"
        in_file.write_text("in-data")
        return script_path, in_file

    @staticmethod
    def _finished_job() -> FakeKubectl:
        """A cluster on which the Job succeeds and logs `hello output`."""
        return FakeKubectl({
            "get job": (0, job_json(succeeded=1), ""),
            "wait": (0, "job.batch/run-1 condition met", ""),
            "get pods": (0, _DONE_POD, ""),
            "logs": (0, "hello output\n", ""),
        })

    @patch.object(gcs_staging, "CONFIGMAP_MAX_BYTES", 0)
    def test_run_requires_staging_bucket_when_staging(self):
        script = str(hello_script(self))
        with self.assertRaises(ComputeConfigError) as ctx:
            make_driver().run(script,
                              inputs=[base.StagedFile(script, "in.txt")])
        self.assertEqual((ctx.exception.error_code, ctx.exception.missing),
                         ("STAGING_NOT_CONFIGURED", ["staging_bucket"]))

    @patch.object(gcs_staging, "CONFIGMAP_MAX_BYTES", 0)
    @patch("framework.compute.drivers.gcs_staging.get_storage_client")
    @patch("framework.compute.drivers.gcs_staging.staging_credentials")
    @patch("framework.compute.drivers.gcs_staging.delete_staged_prefix")
    @patch("framework.compute.drivers.gcs_staging.download_file_from_gcs")
    @patch("framework.compute.drivers.gcs_staging.generate_v4_signed_url")
    @patch("framework.compute.drivers.gcs_staging.upload_file_to_gcs")
    def test_run_stages_with_its_own_credential(self, mock_upload, mock_sign,
                                                mock_download, mock_delete,
                                                mock_creds, mock_client):
        """With a staging credential configured, every staging
        call acts as it, and nothing sets $GOOGLE_APPLICATION_CREDENTIALS
        for kubectl to inherit."""
        mock_sign.return_value = "https://storage.googleapis.com/test-url"
        client = mock_client.return_value
        fake = self._finished_job()
        driver = make_driver(
            staging_bucket="gs://test-staging-bucket",
            staging_service_account="stager@p.iam.gserviceaccount.com",
        )
        driver.run_id = "r1"
        mock_download.side_effect = lambda _b, _k, dest, **_kw: bool(
            Path(dest).write_text("downloaded-result"))
        environ = dict(os.environ)
        with tempfile.TemporaryDirectory() as td:
            script_path, in_file = self._script_and_input(td)
            out_file = Path(td) / "o.dat"
            with patch.object(subprocess, "run", fake):
                res = driver.run(
                    str(script_path),
                    inputs=[base.StagedFile(str(in_file), "input.dat")],
                    outputs=[base.StagedFile(str(out_file), "o.dat")],
                )
            self.assertEqual(out_file.read_text(), "downloaded-result")
        self.assertEqual((res.exit_code, res.stdout), (0, "hello output\n"))
        mock_upload.assert_called_once()
        mock_delete.assert_called_with("test-staging-bucket",
                                       f"{res.details['job_id']}/",
                                       client=client)
        mock_creds.assert_called_once_with("stager@p.iam.gserviceaccount.com",
                                           None)
        mock_client.assert_called_once_with(
            credentials=mock_creds.return_value)
        for mock in (mock_upload, mock_sign, mock_download, mock_delete):
            self.assertTrue(mock.call_args_list, mock)
            for call in mock.call_args_list:
                self.assertIs(call.kwargs.get("client"), client)
        self.assertEqual(dict(os.environ), environ)
        # The run-end reaper finds jobs by run id; logs go with the pod.
        self.assertEqual(
            jobs.JobRecord.load(res.details["job_id"]).run_id, "r1")
        verbs = [" ".join(c) for c in fake.calls]
        self.assertLess(
            next(i for i, v in enumerate(verbs) if "logs" in v),
            next(i for i, v in enumerate(verbs) if "delete job" in v))

    @patch.object(gcs_staging, "CONFIGMAP_MAX_BYTES", 0)
    @patch("framework.compute.drivers.gcs_staging.delete_staged_prefix")
    @patch("framework.compute.drivers.gcs_staging.generate_v4_signed_url")
    @patch("framework.compute.drivers.gcs_staging.upload_file_to_gcs")
    def test_run_cleans_up_an_upload_whose_signing_fails(
            self, mock_upload, mock_sign, mock_delete):
        """`run` stages before `submit`, which cleans up only its own."""
        mock_sign.side_effect = ComputeConfigError(
            "cannot sign", error_code="SIGNING_UNSUPPORTED")
        fake = FakeKubectl()
        driver = make_driver(staging_bucket="gs://test-staging-bucket")
        with tempfile.TemporaryDirectory() as td:
            script_path, in_file = self._script_and_input(td)
            with patch.object(subprocess, "run", fake):
                with self.assertRaises(ComputeConfigError):
                    driver.run(
                        str(script_path),
                        inputs=[base.StagedFile(str(in_file), "input.dat")],
                    )
        mock_upload.assert_called_once()
        mock_delete.assert_called_once()
        self.assertEqual(mock_delete.call_args[0][0], "test-staging-bucket")
        self.assertTrue(mock_delete.call_args[0][1].endswith("/"))
        # Only the lookup that checks the name is free; nothing was created.
        self.assertEqual([c for c in fake.calls if "get" not in c], [])

    @patch.object(gcs_staging, "CONFIGMAP_MAX_BYTES", 0)
    @patch("framework.compute.drivers.gcs_staging.delete_staged_prefix")
    @patch("framework.compute.drivers.gcs_staging.generate_v4_signed_url",
           return_value="https://u")
    @patch("framework.compute.drivers.gcs_staging.upload_file_to_gcs")
    def test_a_name_taken_after_the_lookup_keeps_the_other_jobs_files(
            self, mock_upload, mock_sign, mock_delete):
        """The ConfigMap and staged prefix under the name are another job's."""
        fake = FakeKubectl({
            "create configmap": (1, "", 'error: failed to create configmap: '
                                 'configmaps "run-1" already exists')
        })
        with tempfile.TemporaryDirectory() as td:
            script_path, in_file = self._script_and_input(td)
            with patch.object(subprocess, "run", fake):
                with self.assertRaises(ComputeConfigError):
                    make_driver(staging_bucket="gs://b").submit(
                        str(script_path),
                        job_name="run-1",
                        inputs=[base.StagedFile(str(in_file), "input.dat")])
        mock_upload.assert_called_once()
        mock_delete.assert_not_called()
        self.assertIsNone(fake.find("delete configmap"))
        # A submitted job may outlive any wait, so its URLs must too.
        self.assertEqual(mock_sign.call_args.kwargs["expiration_seconds"],
                         gke_job.gke_common.SUBMIT_URL_EXP_SEC)


def _framed(index: int, data: bytes, job: str = "run-1") -> str:
    """An output as the pod's bootstrap prints it into `job`'s log."""
    body = base64.b64encode(data).decode()
    lines = [body[i:i + 76] for i in range(0, len(body), 76)]
    return "\n".join([
        f"{gcs_staging.output_frame(job)} begin {index} "
        f"{hashlib.sha256(data).hexdigest()}", *lines,
        f"{gcs_staging.output_frame(job)} end {index}"
    ]) + "\n"


# The pod's note that output 0 of `run-1` did not fit in its log.
_TOO_LARGE = f"{gcs_staging.output_frame('run-1')} too-large 0 21600000\n"


class TestGkeJobSmallFiles(unittest.TestCase):
    """Inputs in the ConfigMap and outputs in the log: no bucket, no signing."""

    def setUp(self):
        self.enterContext(job_store())
        # `run` names its Job, and the frames in a scripted log name it too.
        self.enterContext(
            patch.object(jobs, "new_job_id", return_value="run-1"))
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.script = self.tmp / "job.py"
        self.script.write_text("print('hi')\n")
        self.config = self.tmp / "config.json"
        self.config.write_text('{"lr": 0.1}')

    def test_small_inputs_ride_in_the_configmap(self):
        fake = FakeKubectl()
        with patch.object(subprocess, "run", fake), \
                patch.object(gcs_staging, "upload_file_to_gcs") as upload:
            make_driver().submit(
                str(self.script),
                inputs=[base.StagedFile(str(self.config), "cfg/c.json")])
        upload.assert_not_called()
        self.assertIn(f"--from-file=tpu-agent-suite-input-0={self.config}",
                      fake.find("create configmap"))
        container = fake.applied_container()
        script = workload_command(container)[2]
        self.assertIn("mkdir -p cfg", script)
        self.assertIn("cp /app/tpu-agent-suite-input-0 cfg/c.json", script)

    def test_run_returns_outputs_through_the_log_without_a_bucket(self):
        result_file = self.tmp / "local" / "result.json"
        fake = FakeKubectl({
            "get job": (0, job_json(succeeded=1), ""),
            "get pods": (0, _DONE_POD, ""),
            "logs": (0, "step 1\n" + _framed(0, b'{"ok": 1}'), ""),
        })
        with patch.object(subprocess, "run", fake):
            res = make_driver().run(
                str(self.script),
                outputs=[
                    base.StagedFile(str(result_file), "result.json"),
                    base.StagedFile(str(self.tmp / "never"), "never.txt"),
                ])
        self.assertEqual(result_file.read_bytes(), b'{"ok": 1}')
        self.assertEqual(res.stdout, "step 1\n")
        self.assertEqual(res.exit_code, 1)
        self.assertIn(
            "'never.txt' was not produced on the target: was not "
            "created", res.stderr)

    @unittest.skipUnless(_google_auth_errors, "needs google-auth")
    def test_outputs_fall_back_to_the_log_without_any_credentials(self):
        """A laptop that never ran `application-default login`."""
        from google.cloud import storage  # noqa: PLC0415
        fake = FakeKubectl()
        with patch.object(subprocess, "run", fake), \
                patch.object(storage, "Client", side_effect=(
                    _google_auth_errors.DefaultCredentialsError("no ADC"))), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            job_id = make_driver(staging_bucket="gs://b").submit(
                str(self.script), outputs=[base.StagedFile("r", "r.json")])
        self.assertIn("NO_CREDENTIALS: could not sign upload URLs",
                      err.getvalue())
        self.assertIn("not keeping a GCS copy", err.getvalue())
        self.assertTrue(jobs.JobRecord.load(job_id).extra["outputs_in_log"])

    def _run_losing_an_output(self, log: str, **driver_kw: Any):
        """`run` declaring one output, whose Job's log is `log`."""
        fake = FakeKubectl({
            "get job": (0, job_json(succeeded=1), ""),
            "get pods": (0, _DONE_POD, ""),
            "logs": (0, log, ""),
        })
        with patch.object(subprocess, "run", fake), \
                patch.object(gcs_staging, "delete_staged_prefix"), \
                contextlib.redirect_stderr(io.StringIO()):
            return make_driver(**driver_kw).run(
                str(self.script),
                outputs=[base.StagedFile(str(self.tmp / "p"), "p.tgz")])

    def test_an_output_too_large_for_the_log_asks_for_a_pod_upload(self):
        """A bucket set, but ADC that cannot sign, lost a 20 MiB
        profile, and the error asked for the bucket that was already set."""
        refused = ComputeConfigError("cannot sign",
                                     error_code="SIGNING_UNSUPPORTED")
        with patch.object(gcs_staging,
                          "generate_v4_signed_url",
                          side_effect=refused):
            res = self._run_losing_an_output(_TOO_LARGE,
                                             staging_bucket="gs://b")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("signing its URL failed (SIGNING_UNSUPPORTED)",
                      res.stderr)
        self.assertEqual(res.details["setup_fields"],
                         list(gcs_staging.POD_UPLOAD_FIELDS))
        # With no bucket, only the bucket: `setup verify` then asks for
        # signing only if this host cannot.
        res = self._run_losing_an_output(_TOO_LARGE)
        self.assertEqual(res.details["setup_fields"], ["staging_bucket"])

    def test_artifacts_lost_in_the_log_fail_with_why_at_submit(self):
        """The reason is the one recorded at submit: a bucket set since then
        would not have helped that Job."""
        jobs.JobRecord.create(job_id="run-1",
                              backend="gke_job",
                              extra={
                                  "outputs": [{
                                      "remote": "p.tgz"
                                  }],
                                  "outputs_in_log":
                                  gke_job.gke_common.NO_BUCKET
                              })
        fake = FakeKubectl({"logs": (0, _TOO_LARGE, "")})
        with patch.object(subprocess, "run", fake), \
                contextlib.redirect_stderr(io.StringIO()) as err, \
                self.assertRaises(base.NotConfigured) as ctx:
            make_driver(staging_bucket="gs://b").fetch_artifacts(
                "run-1", self.tmp / "out")
        self.assertEqual((ctx.exception.error_code, ctx.exception.missing),
                         ("STAGING_NOT_CONFIGURED", ["staging_bucket"]))
        self.assertIn("no 'staging_bucket' was set", err.getvalue())
        self.assertTrue((self.tmp / "out" / "run-1.log").is_file())

    def test_artifacts_of_a_submitted_job_come_from_its_log(self):
        jobs.JobRecord.create(job_id="run-1",
                              backend="gke_job",
                              extra={
                                  "outputs": [{
                                      "local": "r.json",
                                      "remote": "eval/r.json"
                                  }],
                                  "outputs_in_log":
                                  True
                              })
        fake = FakeKubectl({"logs": (0, "done\n" + _framed(0, b"42"), "")})
        with patch.object(subprocess, "run", fake):
            written = make_driver().fetch_artifacts("run-1", self.tmp / "out")
        self.assertEqual((self.tmp / "out" / "run-1.log").read_text(),
                         "done\n")
        self.assertEqual((self.tmp / "out" / "eval" / "r.json").read_bytes(),
                         b"42")
        self.assertEqual(len(written), 2)
        # Outputs staged to a bucket that is no longer configured.
        jobs.JobRecord.create(job_id="run-2",
                              backend="gke_job",
                              extra={"outputs": [{
                                  "remote": "r.json"
                              }]})
        with patch.object(subprocess, "run", fake), \
                self.assertRaises(ComputeConfigError) as ctx:
            make_driver().fetch_artifacts("run-2", self.tmp / "out")
        self.assertEqual((ctx.exception.error_code, ctx.exception.missing),
                         ("STAGING_NOT_CONFIGURED", ["staging_bucket"]))

    def test_a_frame_is_accepted_only_whole(self):
        good = _framed(0, b"payload")
        damaged = good.replace(good.splitlines()[1], "AAAA")
        for log, expected in (
            (good, b"payload"),
            (damaged, "arrived damaged in the pod's log"),
            (good.rsplit("\n", 2)[0], "was cut off in the pod's log"),
        ):
            with self.subTest(expected=expected):
                kept, found = gcs_staging.split_logged_outputs(
                    "a\n" + log, "run-1")
                self.assertEqual(kept, "a\n")
                self.assertEqual(found, {0: expected})
        # A frame whose start rotated out of the log says so.
        kept, found = gcs_staging.split_logged_outputs(
            good.split("\n", 1)[1], "run-1")
        self.assertIn("log rotated", found[0])

    @unittest.skipUnless(
        all(shutil.which(t) for t in ("bash", "base64", "sha256sum")),
        "needs bash, base64 and sha256sum")
    def test_bootstrap_sends_outputs_when_the_pod_is_stopped(self):
        """SIGTERM (a timeout, a preemption) reaches the workload, and its
        outputs come back before the grace period's SIGKILL."""
        mount = self.tmp / "mount"
        mount.mkdir()
        (mount / "job.py").write_text("import time\n"
                                      "open('r.txt', 'w').write('kept')\n"
                                      "print('started', flush=True)\n"
                                      "time.sleep(60)\n")
        with patch.object(gcs_staging, "POD_WORK_DIR", str(self.tmp / "w")):
            command = gcs_staging.build_pod_bootstrap_command(
                script_mount=str(mount),
                script_name="job.py",
                interpreter=sys.executable,
                logged_outputs=["r.txt"],
                job="run-1",
            )
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        self.assertEqual(proc.stdout.readline(), "started\n")
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=10)
        # The workload's own exit: killed by the signal.
        self.assertEqual(proc.returncode, 128 + signal.SIGTERM)
        self.assertEqual(
            gcs_staging.split_logged_outputs(out, "run-1")[1], {0: b"kept"})

    @unittest.skipUnless(
        all(
            shutil.which(t)
            for t in ("bash", "base64", "sha256sum", "curl", "sed")),
        "needs bash, base64, sha256sum, curl and sed")
    def test_bootstrap_frames_an_output_the_bucket_refuses(self):
        """An upload that fails comes back in the log and its GCS copy; so
        does one with no Workload Identity token to send."""
        mount = self.tmp / "mount"
        mount.mkdir()
        (mount / "job.py").write_text("open('r.bin', 'wb').write(b'res')\n")
        refused = "http://127.0.0.1:9/refused"  # Nothing listens on port 9.
        for pod_token in (False, True):
            with self.subTest(pod_token=pod_token):
                work = self.tmp / f"w-{pod_token}"
                with patch.object(gcs_staging, "POD_WORK_DIR", str(work)), \
                        patch.object(gcs_staging, "POD_TOKEN_URL", refused):
                    command = gcs_staging.build_pod_bootstrap_command(
                        script_mount=str(mount),
                        script_name="job.py",
                        interpreter=sys.executable,
                        output_urls=[("r.bin", refused)],
                        log_url=refused,
                        job="run-1",
                        pod_token=pod_token,
                    )
                proc = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stdout)
                # curl's error and the notice share the frame's stream.
                log, found = gcs_staging.split_logged_outputs(
                    proc.stdout, "run-1")
                self.assertEqual(found, {0: b"res"})
                self.assertIn("returning it in the log", log)
                self.assertEqual(
                    "The GKE metadata server returned no Google token (curl: "
                    in log, pod_token)
                self.assertNotIn("--workload-metadata", log)
                self.assertEqual(proc.stderr, "")
                log_copy = (work / gcs_staging.POD_LOG_FILE).read_text()
                self.assertEqual(
                    gcs_staging.split_logged_outputs(log_copy, "run-1")[1],
                    {0: b"res"})


class TestGcsStagingBucket(unittest.TestCase):

    def test_a_refused_request_is_a_config_error(self):

        class GoogleAPICallError(Exception):
            code, message = 403, "storage.objects.create denied"

        exceptions = types.SimpleNamespace(
            GoogleAPICallError=GoogleAPICallError)
        api_core = types.SimpleNamespace(exceptions=exceptions)
        client = MagicMock()
        client.bucket.return_value.blob.return_value.upload_from_filename \
            .side_effect = GoogleAPICallError()
        with tempfile.NamedTemporaryFile() as f, patch.dict(
                sys.modules, {
                    "google": types.SimpleNamespace(api_core=api_core),
                    "google.api_core": api_core,
                }):
            with self.assertRaises(ComputeConfigError) as ctx:
                gcs_staging.upload_file_to_gcs(f.name, "b", "x", client=client)
        self.assertEqual(ctx.exception.error_code, "GCS_FORBIDDEN")
        self.assertIn("storage.objects.create denied", ctx.exception.message)


class TestGkeJobWorkloadIdentity(unittest.TestCase):
    """With `pod_service_account`, the pod moves staged files
    with its own Workload Identity token, so nothing has to be signed."""

    @patch.object(gcs_staging, "CONFIGMAP_MAX_BYTES", 0)
    def test_submit_runs_the_pod_as_the_account_and_signs_nothing(self):
        self.enterContext(job_store())
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        script = tmp / "job.py"
        script.write_text("print('x')\n")
        data = tmp / "in.bin"
        data.write_bytes(b"in")
        fake = FakeKubectl()
        driver = make_driver(staging_bucket="gs://b/runs",
                             pod_service_account="ksa")
        with patch.object(subprocess, "run", fake), \
                patch.object(gcs_staging, "upload_file_to_gcs") as upload, \
                patch.object(gcs_staging, "generate_v4_signed_url",
                             side_effect=AssertionError("signed")):
            job = driver.submit(
                str(script),
                inputs=[base.StagedFile(str(data), "a b.bin")],
                outputs=[base.StagedFile(str(tmp / "o.bin"), "o.bin")])
        upload.assert_called_once()
        spec = fake.applied_pod_spec()
        self.assertEqual(spec["serviceAccountName"], "ksa")
        self.assertEqual(
            spec["nodeSelector"][gke_job.gke_common.WORKLOAD_IDENTITY_LABEL],
            "true")
        pod_script = workload_command(spec["containers"][0])[2]
        # Plain object URLs under the bucket's folder, names URL-encoded.
        prefix = f"https://storage.googleapis.com/b/runs/{job}"
        for url in (f"{prefix}/in/a%20b.bin", f"{prefix}/out/o.bin",
                    f"{prefix}/out/{gcs_staging.POD_LOG_FILE}"):
            self.assertIn(url, pod_script)
        # The input, the output (as it changes, and at the end) and the log
        # copy each send a fresh token.
        self.assertEqual(
            pod_script.count('-H "Authorization: Bearer $(gcs_token)"'), 4)


class TestGkeJobPodToken(unittest.TestCase):
    """With a staging bucket and no pod_service_account, the pod gets a GCS
    token limited to its own folder and the bucket's checkpoints, in a Secret
    the Job owns."""

    TOKEN = "fake-pod-token-value"

    def setUp(self):
        self.enterContext(job_store())

    def submit(self, job_name=None, **params):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        script = tmp / "job.py"
        script.write_text("print('x')\n")
        fake = FakeKubectl()
        driver = make_driver(**params)
        with patch.object(subprocess, "run", fake), \
                patch.object(gcs_staging, "mint_bucket_token",
                             return_value=(self.TOKEN, time.time() + 3600)
                             ) as mint, \
                contextlib.redirect_stderr(io.StringIO()):
            job = driver.submit(str(script), job_name=job_name)
        applied = {
            o["kind"]: o
            for o in (json.loads(s) for c, s in zip(fake.calls, fake.stdins)
                      if "apply" in c and s)
        }
        return job, applied, mint

    def test_the_pod_token_reaches_its_folder_and_reads_checkpoints(self):
        # Its own folder, where `stage_files` puts its files; checkpoints beside
        # it, read-only; both under the staging folder when there is one.
        for bucket, folder in (("gs://b", ""), ("gs://b/runs", "runs/")):
            with self.subTest(staging_bucket=bucket):
                job, applied, mint = self.submit(staging_bucket=bucket)
                mint.assert_called_once_with(
                    "b", f"{folder}{job}/",
                    (f"{folder}{gke_job.manifest.CHECKPOINTS_PREFIX}", ))
        self.assertEqual(applied["Secret"]["stringData"],
                         {"token": self.TOKEN})
        self.assertNotIn(self.TOKEN, json.dumps(applied["Job"]))
        c = applied["Job"]["spec"]["template"]["spec"]["containers"][0]
        self.assertIn(
            {
                "name": "GCS_TOKEN_FILE",
                "value": f"{gke_job.manifest.TOKEN_MOUNT}/token"
            }, c["env"])
        # subPath mounts don't pick up renewed tokens.
        mount = next(m for m in c["volumeMounts"] if m["name"] == "gcs-token")
        self.assertNotIn("subPath", mount)

    def test_a_job_named_after_the_checkpoints_folder_is_refused(self):
        # Its folder would be writable, so its pod could rewrite checkpoints.
        with self.assertRaises(ComputeConfigError) as cm:
            self.submit(staging_bucket="gs://b", job_name="checkpoints")
        self.assertEqual(cm.exception.error_code, "INVALID_FIELD")


class TestPodTokenRules(unittest.TestCase):
    """What a pod token may reach: read/write under one folder, read-only
    under others, nothing else."""

    def test_the_minted_token_carries_those_rules(self):
        # A prefix without its slash would let job-1 reach job-10/; an empty
        # one would be the whole bucket.
        for prefix in ("", "runs/job-1"):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                _REAL_MINT_BUCKET_TOKEN("b", prefix)
        try:
            from google.auth import \
                downscoped  # pylint: disable=import-outside-toplevel
        except ImportError:
            self.skipTest("google-auth is not installed")
        with patch("google.auth.default", return_value=(MagicMock(), "p")), \
                patch.object(downscoped, "Credentials") as credentials:
            credentials.return_value.expiry = None
            _REAL_MINT_BUCKET_TOKEN("b", "runs/job-1/", ("checkpoints/", ))
        boundary = credentials.call_args.kwargs["credential_access_boundary"]

        def rule(role, prefix):
            return {
                "availableResource":
                "//storage.googleapis.com/projects/_/buckets/b",
                "availablePermissions": [f"inRole:roles/storage.{role}"],
                "availabilityCondition": {
                    "expression":
                    "resource.name.startsWith('projects/_/buckets/b/objects/"
                    f"{prefix}') || api.getAttribute('storage.googleapis.com/"
                    f"objectListPrefix', '').startsWith('{prefix}')"
                },
            }

        self.assertEqual(
            boundary.to_json()["accessBoundary"]["accessBoundaryRules"], [
                rule("objectUser", "runs/job-1/"),
                rule("objectViewer", "checkpoints/")
            ])


if __name__ == "__main__":
    unittest.main()
