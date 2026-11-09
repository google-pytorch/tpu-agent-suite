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
"""Unit tests for compute configuration: 3-tier resolution,
environment.json schemas and profiles, CLI override precedence, and the
target software checks."""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import shutil
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

from framework import compute, paths, run_state, runtime, utils
from framework.compute import (ComputeConfigError, GkeJobDriver,
                               SynchronousExecutionResult)
from framework.compute import config as compute_config
from framework.compute import get_driver, resolve_driver
from framework.compute.drivers import base, jobs
from framework.tests.compute.helpers import (GKE_ANSWERS, MockTestDriver,
                                             ScriptMixin, cli_args, patch_home,
                                             profile_env, set_up_module,
                                             tear_down_module)

Path = pathlib.Path


def setUpModule():  # noqa: N802 - unittest's spelling
    set_up_module()


def tearDownModule():  # noqa: N802 - unittest's spelling
    tear_down_module()


class TestComputeConfig(ScriptMixin, unittest.TestCase):
    """Driver resolution, environment.json schemas, and override precedence."""

    @contextlib.contextmanager
    def _environment(self, data: dict[str, Any]):
        """Resolve against a stubbed environment.json payload."""
        with patch.object(compute_config,
                          "find_environment_data",
                          return_value=data):
            yield

    def test_missing_or_unknown_backend_raises(self):
        """Verify missing or unrecognized backend names fail with proper error codes."""
        with self._environment(
            {}), self.assertRaises(ComputeConfigError) as ctx:
            resolve_driver()
        self.assertEqual(ctx.exception.error_code, "NO_COMPUTE_BACKEND")
        self.assertIn("--package <name@version>", str(ctx.exception))

        for backend in ("auto", "nonexistent_backend_xyz"):
            with self.subTest(backend), self.assertRaises(
                    ComputeConfigError) as ctx:
                resolve_driver(backend_override=backend)
            self.assertEqual(ctx.exception.error_code, "INVALID_BACKEND")

    def test_cli_passes_the_job_and_fills_in_a_fields_fix(self):
        """The job's id reaches the resolver; a driver's `<GROUP>_NOT_CONFIGURED`
        gets the `setup plan --fields` command for this package and the
        driver's profile as its fix."""
        driver, seen = MockTestDriver(), {}
        driver.compute_profile = "c"
        args = cli_args(job_id="job-1")
        with patch.object(driver,
                          "inspect",
                          side_effect=base.NotConfigured(
                              "unset", "STAGING_NOT_CONFIGURED",
                              ["staging_bucket"])):
            with self.assertRaises(ComputeConfigError) as ctx:
                compute.handle_compute_cli(
                    args, resolve=lambda **kw: seen.update(kw) or driver)
        self.assertEqual(seen["job_id"], "job-1")
        error = ctx.exception.to_dict()
        self.assertEqual((error["error_code"], error["missing"]),
                         ("STAGING_NOT_CONFIGURED", ["staging_bucket"]))
        self.assertTrue(error["fix"].endswith(
            "compute --package pkg@1 setup plan --profile c --fields "
            "staging_bucket"))

    def test_run_turns_setup_fields_into_a_fix(self):
        """An output lost in a gke_job pod's log comes back as `setup_fields`;
        `run` adds the command that asks for them, as for NotConfigured."""
        driver = MockTestDriver()
        driver.compute_profile = "c"
        lost = SynchronousExecutionResult(
            1, "", "", details={"setup_fields": ["pod_service_account"]})
        args = cli_args(compute_action="run",
                        code=self._script(""),
                        timeout=None,
                        run_args=[])
        with patch.object(driver, "run", return_value=lost), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            compute.handle_compute_cli(args, resolve=lambda **_: driver)
        fix = json.loads(out.getvalue())["details"]["fix"]
        self.assertTrue(
            fix.endswith("compute --package pkg@1 setup plan --profile c "
                         "--fields pod_service_account"))


class TestEnvironmentResolution(unittest.TestCase):
    """environment.json lookup must stay package-agnostic."""

    def _install(self, directory: Path, backend: str | None,
                 package: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if backend:
            (directory / "environment.json").write_text(
                json.dumps(profile_env(backend)))
        (directory / "ownership.json"
         ).write_text(json.dumps({"package": package}) if package else "{")

    def test_environment_resolution_precedence_and_fallback(self):
        """Ownership-keyed resolution: workspace is authoritative, central
        serves the not-installed-here case, and nothing else is consulted."""
        home = patch_home(self)
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self._install(tmp / "ws" / ".tpu-agent-suite", "scoped", "demo-pkg")
        self._install(home, "central", "demo-pkg")
        # Another package's install, or an unreadable ledger, is not ours.
        self._install(tmp / "other" / ".tpu-agent-suite", "x", "other-pkg")
        self._install(tmp / "unreadable" / ".tpu-agent-suite", "x", "")
        (tmp / "elsewhere").mkdir()

        def backend(ws: str, package: str) -> str | None:
            data = compute_config.find_environment_data(workspace=tmp / ws,
                                                        package=package)
            return data and data["profiles"]["default"]["backend"]

        self.assertEqual(backend("ws", "demo-pkg@2.0.0"), "scoped")
        # The central store is not version-namespaced: a bare name reaches it.
        for ws in ("elsewhere", "other", "unreadable"):
            self.assertEqual(backend(ws, "demo-pkg"), "central", ws)
        self.assertEqual(
            compute_config.find_environment_data(workspace=tmp / "ws"), {})
        self.assertEqual(backend("elsewhere", "ghost-pkg@1.0.0"), {})
        # Broken install: ledger present, config gone -> loud error.
        self._install(tmp / "broken" / ".tpu-agent-suite", None, "demo-pkg")
        with self.assertRaises(runtime.TpuAgentSuiteError):
            backend("broken", "demo-pkg@2.0.0")
        with self.assertRaises(runtime.TpuAgentSuiteError):
            runtime.paths.resolve_environment("", tmp / "ws")


class TestMultiProfileResolution(unittest.TestCase):
    """Multi-profile environment.json: shapes, precedence, and named errors."""

    PKG = "pkg@1.0.0"

    def setUp(self):
        self.workspace = patch_home(self) / "ws"
        self.env_file = self.workspace / ".tpu-agent-suite" / "environment.json"
        self.env_file.parent.mkdir(parents=True)

    @staticmethod
    def _profiles(default: str | None, *names: str) -> dict[str, Any]:
        """A multi-profile document whose profiles run `/<name>/python`."""
        return {
            "revision": 1,
            "default": default,
            "profiles": {
                n: {
                    "backend": "local",
                    "params": {
                        "python_interpreter": f"/{n}/python"
                    }
                }
                for n in names
            },
        }

    def _interpreter(self, env: dict[str, Any] | str, **kwargs: Any) -> str:
        """Write `env` (a dict or JSON text); the resolved driver's Python."""
        self.env_file.write_text(
            env if isinstance(env, str) else json.dumps(env), encoding="utf-8")
        return resolve_driver(workspace=self.workspace,
                              package=self.PKG,
                              **kwargs).python_interpreter

    def test_precedence_is_flag_then_job_then_run_then_default(self):
        abc = self._profiles("a", "a", "b", "c")
        self.assertEqual(self._interpreter(abc), "/a/python")
        run_id = run_state.start_run(package=self.PKG,
                                     workspace=self.workspace,
                                     compute_profile="b")["run_id"]
        self.assertEqual(self._interpreter(abc, run_id=run_id), "/b/python")
        self.assertEqual(
            self._interpreter(abc, run_id=run_id, profile_override="c"),
            "/c/python")
        # A job is reached through the profile it was submitted under; a
        # record from before jobs kept one falls back to the run's.
        job = jobs.JobRecord.create(compute_profile="c").job_id
        old = jobs.JobRecord.create().job_id
        self.assertEqual(self._interpreter(abc, run_id=run_id, job_id=job),
                         "/c/python")
        self.assertEqual(self._interpreter(abc, run_id=run_id, job_id=old),
                         "/b/python")
        self.assertEqual(
            self._interpreter(abc, job_id=job, profile_override="a"),
            "/a/python")
        # A job whose profile was deleted falls back, so it can be cancelled.
        gone = jobs.JobRecord.create(compute_profile="gone").job_id
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(
                self._interpreter(abc, run_id=run_id, job_id=gone),
                "/b/python")
        self.assertIn("'gone' no longer exists", err.getvalue())
        # A driver built for another backend is not the profile's target.
        self.assertIsNone(
            resolve_driver(workspace=self.workspace,
                           package=self.PKG,
                           backend_override="mock_test").compute_profile)
        # The CLI forwards --profile and the active run to the resolver.
        with patch.object(runtime.compute.config, "resolve_driver") as m, \
                contextlib.redirect_stdout(io.StringIO()):
            m.return_value.inspect.return_value = MockTestDriver().inspect()
            runtime.main([
                "compute", "--package", self.PKG, "--workspace",
                str(self.workspace), "--profile", "c", "inspect", "--json"
            ])
        self.assertEqual((m.call_args.kwargs["profile_override"],
                          m.call_args.kwargs["run_id"]), ("c", run_id))

    def test_unresolvable_profiles_are_named_errors(self):
        for env, kwargs, code in [
            ('{"default": "typo", "profiles": {"a": {}}}', {},
             "UNKNOWN_PROFILE"),
            ("{}", dict(profile_override="x"), "UNKNOWN_PROFILE"),
            ('{"default": null, "profiles": {}}', {},
             "COMPUTE_NOT_CONFIGURED"),
            ('{"profiles": {}}', dict(profile_override=" "), "INVALID_FIELD"),
            ('{"backend": "mock_test", "endpoint": "flat"}', {},
             "COMPUTE_NOT_CONFIGURED"),
            ('{"profiles": []}', {}, "INVALID_FIELD"),
            ('{"profiles": {"vm": "gce"}}', {}, "INVALID_FIELD"),
            ('{"default": "", "profiles": {}}', {}, "INVALID_FIELD"),
        ]:
            with self.subTest(env):
                with self.assertRaises(ComputeConfigError) as ctx:
                    self._interpreter(env, **kwargs)
                self.assertEqual(ctx.exception.error_code, code)
                if env == '{"default": null, "profiles": {}}':
                    self.assertEqual(ctx.exception.missing, ["backend"])

        # start-run refuses an unknown name, or a name without a package,
        # before recording a run.
        self.env_file.write_text(json.dumps(self._profiles("a", "a")))
        for kwargs, code, text in [
            (dict(package=self.PKG,
                  workspace=self.workspace,
                  compute_profile="nope"), "UNKNOWN_PROFILE",
             "available profiles: a"),
            (dict(compute_profile="a"), "MISSING_FIELD", "--package"),
            (dict(compute_profile=" "), "INVALID_FIELD", "non-empty"),
        ]:
            with self.assertRaises(ComputeConfigError) as ctx:
                run_state.start_run(**kwargs)
            self.assertEqual(ctx.exception.error_code, code)
            self.assertIn(text, ctx.exception.message)
        self.assertFalse(paths.get_runs_root().is_dir()
                         and any(paths.get_runs_root().iterdir()))

    def test_driver_flags_complete_the_profile_they_are_checked_with(self):
        # CLI flags can supply missing setup fields (e.g. `--image` for gke_job).
        # We test with tpu_topology here because it has no default value.
        self.env_file.write_text(
            json.dumps(
                profile_env("gke_job",
                            image="img:1",
                            tpu_accelerator="tpu-v6e-slice")))
        with self.assertRaises(ComputeConfigError) as ctx:
            resolve_driver(workspace=self.workspace, package=self.PKG)
        self.assertEqual((ctx.exception.error_code, ctx.exception.missing),
                         ("COMPUTE_NOT_CONFIGURED", ["tpu_topology"]))

        # None does not overwrite existing values, and run-only flags
        # (like mldiagnostics) are ignored during validation.
        driver = resolve_driver(workspace=self.workspace,
                                package=self.PKG,
                                param_overrides={
                                    "tpu_topology": "2x4",
                                    "image": None,
                                    "mldiagnostics": True,
                                })
        self.assertEqual((driver.config.tpu_topology, driver.config.image),
                         ("2x4", "img:1"))


class TestTargetSoftwareCheck(unittest.TestCase):
    """The software check `compute setup verify` runs on the target."""

    def setUp(self):
        super().setUp()
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        self.requirements = tmp / utils.TARGET_REQUIREMENTS
        self.requirements.write_text(
            "# comment\njax>=0.11.0\n--index-url https://x\n",
            encoding="utf-8")

    def _check(self,
               target_run=None,
               *,
               backend="local",
               params=None,
               scheduled=False,
               smoke_test=False):
        """Run `compute.setup.check_target` on a mock driver for `backend`."""
        real = get_driver(backend)
        driver = MagicMock(quick_run=target_run is not None and not scheduled,
                           scheduled_run=scheduled,
                           backend_name=backend,
                           runs_in_container=real.runs_in_container,
                           SETUP_FIELDS=real.SETUP_FIELDS)
        driver.preflight.return_value = base.PreflightReport(
            backend, "this host", True,
            [base.PreflightCheck("tpu_devices", "ok")])
        driver.run.return_value = target_run
        out = io.StringIO()
        report = compute.setup.check_target(
            driver, {"python_interpreter": "/venv/bin/python"}
            if params is None else params,
            self.requirements,
            smoke_test=smoke_test,
            out=out)
        return report, out.getvalue(), driver

    def test_a_scheduled_backend_smoke_tests_only_when_asked(self):
        failed = base.SynchronousExecutionResult(
            1, "", "The image could not be pulled: ImagePullBackOff")
        gke = dict(backend="gke_job",
                   params={"image": "img:1"},
                   scheduled=True)
        report, out, driver = self._check(failed, **gke)
        driver.run.assert_not_called()
        self.assertIn("pass --smoke-test", out)
        # Skipping the smoke test leaves a target that is otherwise ready.
        self.assertEqual(report["status"], "ok")

        # Asked for: the run's error is shown, and a failure counts.
        report, out, driver = self._check(failed, smoke_test=True, **gke)
        self.assertEqual(driver.run.call_args.kwargs["timeout"],
                         compute.setup.SMOKE_TEST_TIMEOUT_SEC)
        # A pip install would not outlast the pod; fixes name the image.
        self.assertEqual(driver.run.call_args.kwargs["args"][-1], "--image")
        self.assertIn("ImagePullBackOff", out)
        self.assertEqual(report["status"], "fail")

    def test_the_software_check_is_skipped_where_it_cannot_run_quickly(self):
        failed = base.SynchronousExecutionResult(1, "", "")
        # A backend that can neither run a script quickly nor schedule one.
        _, _, driver = self._check()
        driver.run.assert_not_called()
        # A build without the requirements to check against.
        self.requirements.unlink()
        _, _, driver = self._check(failed)
        driver.run.assert_not_called()

    def test_a_software_check_that_cannot_run_is_a_failure(self):
        """It asks again for the profile's image, or else its interpreter."""
        for backend, field in (("gke_job", "image"), ("gke_pod",
                                                      "python_interpreter")):
            broken = MagicMock(backend_name=backend,
                               SETUP_FIELDS=get_driver(backend).SETUP_FIELDS)
            broken.run.side_effect = OSError("kubectl cp failed")
            check, versions = compute.setup._run_target_check(
                broken, self.requirements)
            self.assertEqual((check.status, check.fields), ("fail", [field]))
            self.assertIsNone(versions)

    def test_local_without_an_interpreter_skips_the_software_check(self):
        # It would check this Python, not the workload's.
        _, out, driver = self._check(base.SynchronousExecutionResult(
            1, "", ""),
                                     params={})
        driver.run.assert_not_called()
        self.assertIn("set python_interpreter", out)


class TestTargetVersions(unittest.TestCase):
    """`compute versions`: the releases on the target, never this machine's."""

    PKG = "pkg@1.0.0"
    GKE = GKE_ANSWERS

    def setUp(self):
        super().setUp()
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.workspace = self.tmp / "ws"
        self.env_file = (self.workspace / ".tpu-agent-suite" /
                         "environment.json")
        self.env_file.parent.mkdir(parents=True)
        self.env_file.write_text('{"profiles": {}}')
        patch_home(self)

    def _python_with(self, **packages: str) -> str:
        """An interpreter whose installed packages are exactly `packages`."""
        site = self.tmp / "site"
        for name, version in packages.items():
            meta = site / f"{name}-{version}.dist-info"
            meta.mkdir(parents=True)
            (meta / "METADATA").write_text(
                f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
                encoding="utf-8")
        python = self.tmp / "python"
        # -S: no site-packages, so only `site` is installed.
        python.write_text(
            f'#!/bin/sh\nPYTHONPATH={site} exec {sys.executable} -S "$@"\n',
            encoding="utf-8")
        python.chmod(0o755)
        return str(python)

    def _apply(self, answers: dict[str, Any]) -> None:
        result = compute.apply_setup(profile="default",
                                     answers=answers,
                                     package=self.PKG,
                                     workspace=self.workspace)
        self.assertEqual("ok", result["status"], result)

    def _versions(self, *flags: str) -> dict[str, Any]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = runtime.main([
                "compute", "versions", "--package", self.PKG, "--workspace",
                str(self.workspace), "--json", *flags
            ])
        self.assertEqual(0, rc)
        return json.loads(out.getvalue())

    def _smoke_test(self,
                    report: str,
                    smoke_test: bool = True) -> dict[str, Any]:
        """`compute setup verify --smoke-test` on the gke_job profile, with
        the Job's output `report`; without the flag if not `smoke_test`."""
        driver = MagicMock(quick_run=False,
                           scheduled_run=True,
                           backend_name="gke_job",
                           runs_in_container=True,
                           SETUP_FIELDS=GkeJobDriver.SETUP_FIELDS)
        driver.preflight.side_effect = lambda: base.PreflightReport(
            "gke_job", "cluster", True, [])
        driver.run.return_value = base.SynchronousExecutionResult(
            0, report, "")
        out = io.StringIO()
        with patch.object(GkeJobDriver, "from_params", return_value=driver), \
                contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(io.StringIO()):
            runtime.main([
                "compute", "setup", "verify", "--package", self.PKG,
                "--workspace",
                str(self.workspace), "--json",
                *(["--smoke-test"] if smoke_test else [])
            ])
        return json.loads(out.getvalue())

    def test_a_local_target_is_asked_with_its_interpreter_not_this_one(self):
        python = self._python_with(jax="0.11.2")
        self._apply({"backend": "local", "python_interpreter": python})
        # This process has another jax; the target's is the one reported.
        with patch("importlib.metadata.version", return_value="0.10.2"):
            found = self._versions()
        self.assertEqual(("live", "0.11.2", None),
                         (found["source"], found["packages"]["jax"],
                          found["packages"]["torch_tpu"]))

    def test_a_record_for_another_image_is_not_the_answer(self):
        """The job image can change after the check: then it's unknown."""
        self._apply(self.GKE)
        self._smoke_test('Status: ok\nVersions: {"jax": "0.11.0"}\n')
        compute.config_set(assignments=["image=us-docker.pkg.dev/p/r/img:v2"],
                           profile=None,
                           package=self.PKG,
                           workspace=self.workspace)
        found = self._versions()
        self.assertEqual((None, {}), (found["source"], found["packages"]))
        self.assertEqual((self.GKE["image"], "us-docker.pkg.dev/p/r/img:v2", {
            "jax": "0.11.0"
        }), (found["stale"]["image"], found["stale"]["configured_image"],
             found["stale"]["packages"]))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            runtime.main([
                "compute", "versions", "--package", self.PKG, "--workspace",
                str(self.workspace)
            ])
        self.assertIn("unknown: the profile changed", out.getvalue())
        self.assertIn("img:v2", out.getvalue())

    def test_a_verify_without_the_smoke_test_keeps_the_releases(self):
        """The profile is unchanged, so what was recorded still holds."""
        self._apply(self.GKE)
        first = self._smoke_test('Status: ok\nVersions: {"jax": "0.11.0"}\n')
        res = self._smoke_test("", smoke_test=False)
        self.assertFalse(res["smoke_test"])
        found = self._versions()
        self.assertEqual(
            ("recorded", {
                "jax": "0.11.0"
            }, first["checked_at"]),
            (found["source"], found["packages"], found["checked"]))


if __name__ == "__main__":
    unittest.main()
