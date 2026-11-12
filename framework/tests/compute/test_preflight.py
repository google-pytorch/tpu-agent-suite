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
"""Unit tests for compute preflight checks.

Also covers the per-session compute check and ADC retry.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import MagicMock, patch

# Imports below follow this sys.path setup, so the file runs directly.
# ruff: noqa: E402
_repo_root = pathlib.Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from framework import compute, paths, utils
from framework.compute import (GceTpuVmDriver, GkeJobDriver, GkePodDriver,
                               LocalDriver, get_registered_drivers)
from framework.compute.drivers import auth_utils, base, gke_pod, probe
from framework.compute.drivers import registry as driver_registry
from framework.tests.compute.helpers import (MockTestDriver, cli_args,
                                             completed, patch_home,
                                             profile_env, set_up_module,
                                             tear_down_module)

Path = pathlib.Path


def setUpModule():  # noqa: N802 - unittest's spelling
    set_up_module()


def tearDownModule():  # noqa: N802 - unittest's spelling
    tear_down_module()


class TestPreflight(unittest.TestCase):
    """`compute preflight`: the same report shape for every backend."""

    def _run(self, driver, responses=None, tools="/bin/tool"):
        """Preflight with subprocess answering by a substring of the argv."""
        calls: list[str] = []

        def fake(cmd, **_kwargs):
            calls.append(" ".join(cmd))
            for key, (code, out) in (responses or {}).items():
                if key in calls[-1]:
                    return completed(out, code, "boom")
            return completed()

        with patch("shutil.which", return_value=tools), \
                patch("subprocess.run", fake):
            report = driver.preflight()
        return report, {c.name: c for c in report.checks}, calls

    ACCOUNT = {"auth list": (0, "me@example.com\n")}

    def _gce(self):
        return GceTpuVmDriver(tpu_name="vm", zone="us-east5-b", project="p")

    def test_a_missing_tool_is_a_failure_and_the_target_is_not_checked(self):
        report, checks, calls = self._run(self._gce(), tools=None)
        self.assertEqual((report.reachable, report.status), (False, "fail"))
        self.assertIn("docs/install.md", checks["tool:gcloud"].fix)
        self.assertEqual(calls, [])

    def test_local(self):
        with patch.object(probe, "list_devices", return_value=([], [])):
            report, checks, _ = self._run(LocalDriver())
        self.assertFalse(report.reachable)
        self.assertEqual(checks["tpu_devices"].status, "fail")
        with patch.object(probe, "list_devices",
                          return_value=(["/dev/accel0"], [])), \
                patch.object(probe, "scan_holders",
                             return_value=(["PID 7"], set(), set())):
            report, checks, _ = self._run(LocalDriver())
        self.assertEqual((report.reachable, report.status), (True, "warn"))

    def test_gce(self):
        report, checks, calls = self._run(self._gce(), {
            **self.ACCOUNT, "describe": (0, "READY\tv6e-8\n")
        })
        self.assertEqual((report.reachable, report.status), (True, "ok"))
        self.assertFalse(any(" ssh " in call for call in calls))
        _, checks, _ = self._run(self._gce(), {
            **self.ACCOUNT, "describe": (0, "STOPPED v6e-8\n")
        })
        self.assertIn("tpu-vm start vm --zone=us-east5-b",
                      checks["tpu_vm"].fix)
        _, checks, _ = self._run(self._gce(), {"auth list": (0, "")})
        self.assertIn("gcloud auth login", checks["gcloud:account"].fix)
        self.assertNotIn("tpu_vm", checks)

    def test_gke_pod(self):
        cluster = {
            "current-context": (0, "ctx"),
            "version": (0, '"serverVersion"')
        }
        _, checks, calls = self._run(GkePodDriver(pod="p1", namespace="ns"), {
            **cluster, "get pod": (0, "Pending")
        })
        self.assertIn("is Pending", checks["pod"].detail)
        # The cluster is probed with the same kubectl the driver runs with.
        kubectl = [call for call in calls if call.startswith("kubectl")]
        self.assertTrue(kubectl)
        self.assertTrue(
            all(call.startswith("kubectl -n ns ") for call in kubectl))
        self.assertIn("gcloud:account", checks)

    @staticmethod
    def _cluster(kubectl=("kubectl", ), kubeconfig=None):
        """The `cluster` check `auth_utils.cluster_check` appends."""
        checks: list = []
        auth_utils.cluster_check(types.SimpleNamespace(), list(kubectl),
                                 checks, kubeconfig)
        return next(c for c in checks if c.name == "cluster")

    def _cluster_check(self, env, kubeconfig=None, context=None):
        """The cluster check with kubectl knowing no context, under `env`."""
        with patch.dict(os.environ, env, clear=True), \
                patch("subprocess.run", return_value=completed("", 1)):
            return self._cluster(base.kubectl_prefix(kubeconfig, context),
                                 kubeconfig)

    def test_a_missing_kubeconfig_is_named(self):
        """A missing kubeconfig reads as "no context" to kubectl; the path is the fault."""
        with tempfile.TemporaryDirectory() as home:
            gone = os.path.join(home, "missing-kubeconfig")
            there = os.path.join(home, "config")
            Path(there).write_text("", encoding="utf-8")
            # Given with --kubeconfig: named, and credentials written there.
            check = self._cluster_check({"HOME": home}, kubeconfig=gone)
            self.assertEqual(check.status, "fail")
            self.assertIn(f"kubeconfig {gone}, which does not exist",
                          check.detail)
            self.assertTrue(check.fix.startswith(f"KUBECONFIG={gone} gcloud"))
            # Named by $KUBECONFIG, alongside one that is there.
            both = f"{gone}{os.pathsep}{there}"
            check = self._cluster_check({"HOME": home, "KUBECONFIG": both})
            self.assertIn(f"$KUBECONFIG names {gone}, which does not exist",
                          check.detail)
            self.assertNotIn(there, check.detail)
            self.assertIn(f"KUBECONFIG={gone} ", check.fix)
            # Neither: the default file, written without naming it.
            check = self._cluster_check({"HOME": home})
            self.assertIn("~/.kube/config, which does not exist", check.detail)
            self.assertTrue(check.fix.startswith("gcloud container clusters"))
            # The file is there but has no context: nothing to name.
            check = self._cluster_check({"HOME": home, "KUBECONFIG": there})
            self.assertIn("no cluster is configured", check.detail)
            # A pinned context is named, and its cluster is in the fix.
            check = self._cluster_check({"HOME": home},
                                        kubeconfig=gone,
                                        context="gke_p_us-central1_c9")
            self.assertIn(
                "context 'gke_p_us-central1_c9' cannot be used: "
                f"kubeconfig {gone}", check.detail)
            self.assertIn("get-credentials c9", check.fix)

    def test_kube_context_check(self):
        """A pinned context: ok if current, a warning if another is.

        It is a failure if the kubeconfig does not have it or cannot be read.
        """
        with tempfile.TemporaryDirectory() as home:
            config = os.path.join(home, "config")
            Path(config).write_text("{}", encoding="utf-8")

            def check(context, contexts="a\nb\n", current="b", code=0):
                out = {"name": contexts, "current-context": current}

                def run(cmd, **_):
                    return completed(out.get(cmd[-1], ""), code, "boom")

                with patch.object(base, "probe_run", side_effect=run):
                    return base.kube_context_check(config, context)

            self.assertIsNone(check(None))
            self.assertEqual(check("b").status, "ok")
            moved = check("a")
            self.assertEqual(moved.status, "warn")
            self.assertIn("'a', not kubectl's current context 'b'",
                          moved.detail)
            for contexts, code, detail, others in (
                ("a\nb\n", 0, "not in the kubeconfig",
                 "one of: a, b"), ("", 0, "not in the kubeconfig", None),
                ("", 1, "could not read the kubeconfig's contexts: boom",
                 None)):
                with self.subTest(contexts=contexts, code=code):
                    failed = check("gke_p_us-central1_c9", contexts, code=code)
                    self.assertEqual((failed.name, failed.status),
                                     ("kube_context", "fail"))
                    self.assertIn(detail, failed.detail)
                    self.assertIn("get-credentials c9", failed.fix)
                    self.assertEqual("one of" in failed.fix, bool(others))
                    if others:
                        self.assertIn(others, failed.fix)

    SA = "123-compute@developer.gserviceaccount.com"

    def _metadata(self, email, scopes):
        """base._metadata_get answering as a VM whose account is `email`."""
        values = {"email": email, "scopes": "\n".join(scopes)}
        return patch.object(base,
                            "_metadata_get",
                            side_effect=lambda p: values[p.rsplit("/", 1)[-1]])

    def _account_check(self, account, other=None):
        with patch("subprocess.run", return_value=completed(account, 0)):
            return base.gcloud_account_check(other)

    def test_a_vm_account_without_cloud_platform_scope(self):
        """A VM's service account is limited to the VM's scopes.

        That holds whatever IAM grants it: say so, rather than let every call
        be refused.
        """
        storage = "https://www.googleapis.com/auth/devstorage.read_only"
        with self._metadata(self.SA, [storage]):
            check = self._account_check(self.SA)
            self.assertEqual(check.status, "fail")
            self.assertIn("does not grant it the cloud-platform scope",
                          check.detail)
            self.assertIn("devstorage.read_only", check.detail)
            self.assertIn("gcloud auth login", check.fix)
            self.assertIn("--scopes=cloud-platform", check.fix)
            # Another credential that works: a warning, and it is named.
            check = self._account_check(self.SA, "kubectl reaches the cluster")
            self.assertEqual(check.status, "warn")
            self.assertIn("not blocking: kubectl reaches the cluster",
                          check.detail)
        with self._metadata(self.SA, [storage, base.CLOUD_PLATFORM_SCOPE]):
            self.assertEqual(self._account_check(self.SA).status, "ok")
        # Some other service account (a key gcloud activated): not the VM's.
        with self._metadata("other@p.iam.gserviceaccount.com", [storage]):
            self.assertEqual(self._account_check(self.SA).status, "ok")
        # Off GCE: nothing to check.
        with patch.object(base, "_metadata_get", side_effect=OSError("no")):
            self.assertEqual(self._account_check(self.SA).status, "ok")
        # A user account is never limited by scopes, so it is not looked up.
        with patch.object(base,
                          "_metadata_get",
                          side_effect=AssertionError("looked up")):
            self.assertEqual(
                self._account_check("me@example.com").status, "ok")
        # No account at all: the same rule.
        self.assertEqual(self._account_check("").status, "fail")
        self.assertEqual(self._account_check("", "$X works").status, "warn")

    def test_application_credentials_are_reported_with_a_warning(self):
        """$GOOGLE_APPLICATION_CREDENTIALS is a credential kubectl will use.

        List it, but warn rather than recommend.
        """
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(base.application_credentials_check(),
                             (None, None))
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        (tmp / "bad.json").write_text("{")

        def check(name, body=None):
            if body:
                (tmp / name).write_text(json.dumps(body))
            with patch.dict(os.environ, {base.GAC_ENV: str(tmp / name)},
                            clear=True):
                return base.application_credentials_check()

        gac, usable = check("u.json", {"type": "authorized_user"})
        self.assertEqual(gac.status, "warn")
        self.assertIn(base.KUBECTL_IDENTITY_WARNING, gac.detail)
        self.assertIn("env -u GOOGLE_APPLICATION_CREDENTIALS", gac.fix)
        self.assertIn("authorized_user", usable)
        gac, usable = check("sa.json", {
            "type": "service_account",
            "client_email": "bot@p"
        })
        self.assertIn("service_account credential bot@p", gac.detail)
        self.assertIsNotNone(usable)
        # Reported, but not one the auth plugin is known to use.
        for name, body, says in (("odd.json", {
                "type": "gdch"
        }, "gdch"), ("gone.json", None, "No such file"), ("bad.json", None,
                                                          "not a readable")):
            gac, usable = check(name, body)
            self.assertIn(says, gac.detail)
            self.assertIsNone(usable)

    def test_the_credentials_repro(self):
        """The missing KUBECONFIG file is the blocker; the scope is not.

        KUBECONFIG names a missing file, the gcloud account is a VM's without
        cloud-platform scope, and an authorized_user
        $GOOGLE_APPLICATION_CREDENTIALS is set.
        """
        with tempfile.TemporaryDirectory() as tmp:
            gone = os.path.join(tmp, "config")
            gac = os.path.join(tmp, "adc.json")
            Path(gac).write_text('{"type": "authorized_user"}')
            env = {"HOME": tmp, "KUBECONFIG": gone, base.GAC_ENV: gac}
            env.update(GKE_CLUSTER="c", GKE_LOCATION="l", GKE_PROJECT="p")
            for driver in (GkePodDriver(pod="p1", namespace="ns"),
                           GkeJobDriver(kubeconfig=None)):
                with self.subTest(driver=driver.backend_name), \
                        patch.dict(os.environ, env, clear=True), \
                        self._metadata(self.SA, []):
                    report, checks, _ = self._run(driver, {
                        "current-context": (1, ""),
                        "auth list": (0, self.SA)
                    })
                self.assertFalse(report.reachable)
                self.assertEqual([
                    checks[c].status
                    for c in ("cluster", "gcloud:account", "credentials:gac")
                ], ["fail", "warn", "warn"])
                self.assertIn(f"{gone}, which does not exist",
                              checks["cluster"].detail)
                self.assertIn("get-credentials c --location l --project p",
                              checks["cluster"].fix)
                self.assertIn(
                    "not blocking: $GOOGLE_APPLICATION_CREDENTIALS holds "
                    "another credential (authorized_user credential)",
                    checks["gcloud:account"].detail)

    def test_cli_prints_the_report_and_exits_1_on_failure(self):
        from framework.compute import cli as compute_cli
        driver = MagicMock()
        driver.preflight.return_value = base.PreflightReport(
            "local", "this host", False,
            [base.PreflightCheck("tpu_devices", "fail", "none", "use a VM")])
        for as_json in (False, True):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = compute_cli._action_preflight(
                    driver, argparse.Namespace(json=as_json))
            self.assertEqual(code, 1)
            self.assertIn(
                '"status": "fail"' if as_json else "[FAIL] tpu_devices  none",
                out.getvalue())


class _CountingDriver(MockTestDriver):
    """Mock driver with settable preflight and inspect results.

    It also keeps a call counter for preflight.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._checks = []
        self._calls = 0
        self._reachable = True
        self._hardware = base.HardwareProfile("v6e",
                                              "2x2",
                                              4,
                                              32.0,
                                              "healthy",
                                              details={
                                                  "accelerator_type": "v6e-4",
                                                  "job_devices": 4,
                                                  "hbm_gb_per_device": 32.0,
                                              })

    def preflight(self):
        self._calls += 1
        return base.PreflightReport("mock_test",
                                    f"mock target {self.endpoint}",
                                    self._reachable, list(self._checks))

    def inspect(self):
        return self._hardware


class TestComputeCheckPerSession(unittest.TestCase):
    """Session start reports the setup state; a changed profile is rechecked."""

    RUN_ID = "20260101T000000Z-123456789abc"

    def setUp(self):
        root = patch_home(self)
        paths.get_run_root(self.RUN_ID).mkdir(parents=True)
        self.workspace = root / "ws"
        self.env_file = self.workspace / ".tpu-agent-suite" / "environment.json"
        self.env_file.parent.mkdir(parents=True)
        self._write_env({"python_interpreter": "a"})
        self.driver = _CountingDriver()
        patcher = patch.object(compute.cli.config,
                               "resolve_driver",
                               side_effect=lambda **_: self.driver)
        self.resolve = patcher.start()
        self.addCleanup(patcher.stop)
        # What this machine has installed changes the note; tests that care
        # set it.
        self.host_patcher = patch.object(compute.cli.setup,
                                         "host_requirements_note",
                                         return_value="")
        self.host_patcher.start()
        self.addCleanup(self.host_patcher.stop)

    def _write_env(self, params, backend="local"):
        self.env_file.write_text(
            json.dumps({
                "revision": 1,
                **profile_env(backend, **params)
            }))

    def _verified(self, passed=True, failed_checks=()):
        """Record a verify result for the current default profile."""
        profile = json.loads(self.env_file.read_text())["profiles"]["default"]
        (self.env_file.parent / paths.VERIFICATION_FILE).write_text(
            json.dumps({
                compute.config.profile_hash(profile): {
                    "profile": "default",
                    "passed": passed,
                    "version": utils.TPU_AGENT_SUITE_VERSION,
                    "failed_checks": list(failed_checks),
                }
            }))

    def _session_start(self, hook_timeout=30):
        with contextlib.redirect_stderr(io.StringIO()):
            return compute.cli.check_target_at_session_start(
                "pkg", self.workspace, self.RUN_ID, hook_timeout)

    def _command(self):
        args = cli_args(package="pkg",
                        workspace=self.workspace,
                        run_id=self.RUN_ID)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            rc = compute.handle_compute_cli(args,
                                            resolve=lambda **_: self.driver)
        return rc, err.getvalue()

    READY = ("Compute ready: profile 'default'. Compute target checked: "
             "mock target default-endpoint (ok)\nHardware: accelerator_type "
             "v6e-4, 4 devices, 32 GB HBM per device. Plan from these "
             "numbers, not from the profile, VM or cluster name.")

    def test_host_requirements_come_from_the_file_shipped_with_the_runtime(
            self):
        real = self.host_patcher.temp_original  # setUp patched it out
        reqs = self.workspace / utils.HOST_REQUIREMENTS
        reqs.write_text("# host\npyyaml>=6.0\nno-such-package-xyz>=1.0\n")
        with patch.object(compute.setup, "_package_file", return_value=reqs):
            note = real()
        self.assertIn(
            f"`{sys.executable} -m pip install 'no-such-package-xyz>=1.0'`",
            note)
        self.assertNotIn("pyyaml", note)
        missing = compute.setup.ComputeConfigError("no file")
        with patch.object(compute.setup, "_package_file", side_effect=missing):
            self.assertEqual(real(), "")

    def test_session_start_gives_one_message_per_setup_state(self):
        """Only a READY profile is preflighted; the rest say what to run."""
        plan = "compute --package pkg setup plan"
        use = "config --package pkg use <name>"
        gke = profile_env("gke_job", tpu_accelerator="t", tpu_topology="1x1")
        cases = [
            ("no profile", '{"profiles": {}}',
             ["Compute is not set up yet.",
              str(compute.cli.SETUP_GUIDE)]),
            ("none selected", '{"profiles": {"vm": {"backend": "local"}}}',
             ["environment.json has vm", use]),
            ("unknown selected", '{"default": "gone", "profiles": {}}',
             ["'gone'", use, f"{plan} --profile <name> --backend <backend>"]),
            ("incomplete", json.dumps(profile_env("gke_pod")), [
                "profile 'default' is not fully set up",
                "setup apply --profile default --answers '<json>'"
            ]),
            ("unverified", json.dumps(profile_env("local")),
             ["setup verify`.", "!--smoke"]),
            ("unverified, schedules Jobs", json.dumps(gke),
             ["add `--smoke-test` only if they agree"]),
        ]
        built_in_only = {
            k: v
            for k, v in get_registered_drivers().items()
            if k not in driver_registry.EXTERNAL_DRIVERS
        }
        for name, env, expected in cases:
            with self.subTest(name):
                self.env_file.write_text(env)
                with patch.object(compute.config,
                                  "get_registered_drivers",
                                  return_value=built_in_only):
                    note = self._session_start()
                self.assertTrue(note.startswith("## Compute setup\n"), note)
                for text in expected:
                    if text.startswith("!"):
                        self.assertNotIn(text[1:], note)
                    else:
                        self.assertIn(text, note)

        # FAILED, from the last verify: its checks, fixes and fields.
        self._write_env({})
        self._verified(False, [
            dict(name="software",
                 detail="no torch_tpu\nmore",
                 fix="install it",
                 fields=["python_interpreter"])
        ])
        note = self._session_start()
        self.assertIn("- software: no torch_tpu Fix: install it", note)
        self.assertIn(f"{plan} --profile default --fields python_interpreter",
                      note)
        self.assertEqual(self.driver._calls, 0)

        # A READY profile whose quick preflight fails reads as FAILED; one
        # that did not answer, with no failing check, names that.
        self._verified()
        self.driver._checks = [
            base.PreflightCheck("image", "fail", "gone", "push it", ["image"])
        ]
        self.assertIn("failed its check.\n- image: gone Fix: push it",
                      self._session_start())
        self.driver._checks, self.driver._reachable = [], False
        self.assertIn("- reachable: mock target default-endpoint did not",
                      self._session_start())
        self.assertEqual(self.driver._calls, 2)
        # A probe without devices or HBM says why, and is never guessed from.
        self.driver._reachable = True
        self.driver._hardware = base.HardwareProfile(
            "tpu", "unknown", 4, 0.0, "healthy", details={"hbm_gb_note": "x"})
        self.assertIn("Hardware not read at session start (x). Don't guess ",
                      self._session_start())
        self.env_file.unlink()
        self.assertFalse(self._session_start())

    def test_only_a_changed_environment_is_checked_again(self):
        """The recheck runs once, reports a failure, and the command still runs."""
        self._verified()
        self._session_start()
        self._command()
        self.assertEqual(self.driver._calls, 1)
        self.driver._checks = [
            base.PreflightCheck("image", "fail", "gone", "push it")
        ]
        self._write_env({"python_interpreter": "b"})
        rc, err = self._command()
        self.assertEqual((rc, self.driver._calls), (0, 2))
        self.assertIn("the compute profile changed", err)
        self.assertIn("[FAIL] image  gone", err)
        self.assertIn("fix: push it", err)
        self._command()
        self.assertEqual(self.driver._calls, 2)
        # A change outside the profile is not a change to the target.
        doc = json.loads(self.env_file.read_text())
        self.env_file.write_text(json.dumps({**doc, "xprof_interpreter": "x"}))
        self._command()
        self.assertEqual(self.driver._calls, 2)

    def test_a_session_check_that_did_not_finish_is_left_to_the_next_command(
            self):
        self._verified()
        release = threading.Event()
        self.addCleanup(release.set)
        slow = self.driver.preflight
        self.driver.preflight = lambda: (release.wait(5), slow())[1]
        for timeout in (compute.cli.SESSION_CHECK_HEADROOM_S,
                        compute.cli.SESSION_CHECK_HEADROOM_S + 0.05):
            self.assertTrue(
                self._session_start(timeout).startswith(
                    "Compute ready: profile 'default'.\n"
                    "Hardware not read at session start."))
        self.assertIsNone(compute.cli._last_checked_hash(self.RUN_ID))
        release.set()
        _, err = self._command()
        self.assertIn("Compute target checked", err)
        self.assertIsNotNone(compute.cli._last_checked_hash(self.RUN_ID))


class TestAdcRetry(unittest.TestCase):
    """A cluster that refuses the gcloud account is retried with ADC."""

    ACCOUNT = "sa@proj.iam.gserviceaccount.com"
    ADC_USER = "me@example.com"

    def setUp(self):
        patcher = patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(auth_utils.ACCESS_TOKEN_ENV, None)
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        for target, value in (("shutil.which", "/usr/bin/tool"), (
                "framework.compute.drivers.base.vm_account_scopes", None)):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.calls = []

    def _cluster(self, *allowed, adc=True):
        """A fake kubectl/gcloud whose cluster lets in only `allowed` credentials."""

        def run(cmd, **kwargs):
            token = os.environ.get(auth_utils.ACCESS_TOKEN_ENV)
            cred = "adc" if token == "adc-token" else "default"
            text = " ".join(cmd)
            self.calls.append((cred, text))
            if "print-access-token" in text:
                return subprocess.CompletedProcess(
                    cmd, 0 if adc else 1, "adc-token\n" if adc else "",
                    "" if adc else "ERROR: Your default credentials were "
                    "not found.")
            if cmd[0] == "gcloud":
                return subprocess.CompletedProcess(cmd, 0, self.ACCOUNT + "\n",
                                                   "")
            if "current-context" in text:
                return subprocess.CompletedProcess(
                    cmd, 0, "gke_proj_us-central1_cluster-a\n", "")
            if " version " in f" {text} ":
                return subprocess.CompletedProcess(cmd, 0,
                                                   '{"serverVersion": {}}', "")
            if cred in allowed:
                out = "Running" if "status.phase" in text else "hi\n"
                if "exec" in cmd:
                    out = f"{gke_pod.EXEC_STARTED}\n{out}"
                return subprocess.CompletedProcess(cmd, 0, out, "")
            user = self.ADC_USER if cred == "adc" else self.ACCOUNT
            return subprocess.CompletedProcess(
                cmd, 1, "",
                "Error from server (Forbidden): pods is forbidden: "
                f'User "{user}" cannot list resource "pods"')

        return patch("subprocess.run", run)

    def test_preflight_falls_back_to_adc_and_says_so(self):
        driver = GkePodDriver(pod="p")
        with self._cluster("adc"):
            report = driver.preflight().to_dict()
        checks = {c["name"]: c for c in report["checks"]}
        self.assertTrue(driver.using_adc)
        self.assertEqual(os.environ[auth_utils.ACCESS_TOKEN_ENV], "adc-token")
        self.assertEqual(checks["cluster"]["status"], "ok")
        self.assertIn("answers with application default credentials",
                      checks["cluster"]["detail"])
        self.assertEqual(checks["credentials:adc"]["status"], "warn")
        self.assertIn(f"refused the gcloud account {self.ACCOUNT}",
                      checks["credentials:adc"]["detail"])
        self.assertEqual(checks["pod"]["status"], "ok")  # read with ADC

    def test_preflight_names_both_identities_and_the_fix_when_adc_fails(self):
        with self._cluster():
            report = GkePodDriver(pod="p").preflight().to_dict()
        cluster = next(c for c in report["checks"] if c["name"] == "cluster")
        self.assertEqual(cluster["status"], "fail")
        self.assertIn(f"the gcloud account {self.ACCOUNT}", cluster["detail"])
        self.assertIn(f"application default credentials ({self.ADC_USER})",
                      cluster["detail"])
        self.assertIn("gcloud auth application-default login", cluster["fix"])
        self.assertIn(
            "gcloud projects add-iam-policy-binding proj "
            f"--member=serviceAccount:{self.ACCOUNT} "
            "--role=roles/container.developer", cluster["fix"])
        self.assertNotIn(auth_utils.ACCESS_TOKEN_ENV, os.environ)

    def test_compute_commands_raise_a_structured_error_when_adc_fails(self):
        with self._cluster(adc=False), self.assertRaises(
                auth_utils.ClusterAccessDenied) as ctx:
            GkeJobDriver(image="img").kube.run(["get", "pods"])
        payload = ctx.exception.to_dict()
        self.assertEqual(payload["error_code"], "CLUSTER_ACCESS_DENIED")
        self.assertEqual(payload["identities"], [
            f"the gcloud account {self.ACCOUNT}",
            "application default credentials (none usable: ERROR: Your "
            "default credentials were not found.)"
        ])
        self.assertIn("gcloud auth application-default login", payload["fix"])

    def test_port_forward_uses_adc_after_the_gcloud_account_is_refused(self):
        drivers = (GkePodDriver(pod="p"), GkeJobDriver(image="img"))
        for driver in drivers:
            os.environ.pop(auth_utils.ACCESS_TOKEN_ENV, None)
            with self.subTest(driver=type(driver).__name__), \
                    self._cluster("adc"):
                cmd, _ = driver._port_forward_command(8080, 80, "job-1")
                self.assertEqual(os.environ[auth_utils.ACCESS_TOKEN_ENV],
                                 "adc-token")
                self.assertIn("port-forward", cmd)
                self.assertNotIn("adc-token", " ".join(cmd))


if __name__ == "__main__":
    unittest.main()
