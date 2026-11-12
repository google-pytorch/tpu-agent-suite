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
"""The software check: of the target at setup, and of this host at install."""

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SCRIPT = Path(__file__).resolve().parents[2] / "compute" / "env_check.py"
_spec = importlib.util.spec_from_file_location("env_check", _SCRIPT)
env_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(env_check)


class TestEnvCheck(unittest.TestCase):

    def test_each_requirement_is_checked_with_its_fix(self):
        versions = {"jax": "0.10.0", "libtpu": "0.0.44", "pkg": "1.2"}
        with mock.patch.object(env_check,
                               "version_of",
                               side_effect=versions.get):
            results = {
                req: env_check.check_requirement(req)
                for req in ("diffusers>=0.28.0", "jax>=0.11.0",
                            "libtpu>=0.0.41", "torch>=2.13.0,<2.14",
                            "pkg[extra]>=1; python_version < '3.13'")
            }
        self.assertEqual(results["diffusers>=0.28.0"][1:],
                         ("fail", "not installed; needs >=0.28.0",
                          f"{sys.executable} -m pip install "
                          "'diffusers>=0.28.0'"))
        self.assertEqual(results["jax>=0.11.0"][2],
                         "0.10.0 installed; needs >=0.11.0")
        self.assertEqual(results["libtpu>=0.0.41"][1], "ok")
        # Extras and markers are not part of the name.
        self.assertEqual(results["pkg[extra]>=1; python_version < '3.13'"],
                         ("pkg", "ok", "1.2 >=1", ""))
        # torch comes with torch_tpu, never on its own.
        self.assertIn("reinstall torch_tpu", results["torch>=2.13.0,<2.14"][3])

    def test_an_image_target_is_fixed_by_its_image_not_pip(self):
        """A package installed into a pod is gone with it."""
        with mock.patch.object(env_check, "version_of", return_value=None):
            checks = {
                name: fix
                for name, _, _, fix in env_check.run_checks(
                    ["jax>=0.11.0", "torch>=2.13.0,<2.14"], image=True)
            }
        for name in ("torch_tpu", "jax", "torch"):
            self.assertIn(env_check.IMAGE_DOC, checks[name])
            self.assertNotIn("pip install", checks[name])
        self.assertIn("'jax>=0.11.0'", checks["jax"])

    def test_a_system_python_is_told_to_use_a_venv_not_pip(self):
        """Review on #679: the hooks may run on a system Python pip refuses.

        With the deps in a venv the user didn't activate, the hooks run on the
        system Python, which pip refuses (PEP 668).
        """
        with tempfile.TemporaryDirectory() as stdlib:
            marker = Path(stdlib) / "EXTERNALLY-MANAGED"
            for in_venv, marked, managed in ((False, True, True), (True, True,
                                                                   False),
                                             (False, False, False)):
                with self.subTest(in_venv=in_venv, marked=marked), \
                        mock.patch.object(env_check.sysconfig, "get_path",
                                          return_value=stdlib), \
                        mock.patch.object(env_check.sys, "base_prefix",
                                          "/usr" if in_venv else sys.prefix):
                    if marked:
                        marker.touch()
                    else:
                        marker.unlink(missing_ok=True)
                    self.assertEqual(env_check.externally_managed(), managed)
                    note = env_check.ask_to_install(["x"])
                    if managed:
                        self.assertIn("activate it and restart this session",
                                      note)
                        self.assertIn("pip install 'x'`, then restart", note)
                        self.assertNotIn("-m pip install", note)
                    else:
                        self.assertIn(f"`{sys.executable} -m pip install 'x'`",
                                      note)
        # Debian's system Python has no ensurepip until python3-venv is
        # installed, and `venv` fails without it.
        for has_ensurepip, apt in ((True, False), (False, True)):
            with self.subTest(has_ensurepip=has_ensurepip), \
                    mock.patch.object(env_check, "externally_managed",
                                      return_value=True), \
                    mock.patch.object(env_check.importlib.util, "find_spec",
                                      return_value=object() if has_ensurepip
                                      else None):
                note = env_check.ask_to_install(["x"])
                self.assertEqual(
                    "`sudo apt install python3-venv && python3 -m venv"
                    in note, apt)
        # An interpreter named explicitly is never second-guessed.
        with mock.patch.object(env_check,
                               "externally_managed",
                               return_value=True):
            self.assertIn(
                "`/v/bin/python -m pip install 'x'`",
                env_check.ask_to_install(["x"], python="/v/bin/python"))

    def test_the_script_runs_standalone_and_exits_1_on_failure(self):
        """A Job carries the script but, without a bucket, no files.

        So `--require` works alongside or instead of `--requirements`.
        """
        with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                         delete=False) as handle:
            handle.write("# comment\nno-such-package-xyz>=1.0\n")
        self.addCleanup(Path(handle.name).unlink)
        proc = subprocess.run([
            sys.executable,
            str(_SCRIPT), "--requirements", handle.name,
            "--require=no-such-package-abc>=1.0",
            "--require=--index-url https://x"
        ],
                              capture_output=True,
                              text=True,
                              check=False)
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertIn("[FAIL] no-such-package-xyz", proc.stdout)
        self.assertIn("[FAIL] no-such-package-abc", proc.stdout)
        self.assertNotIn("index-url", proc.stdout)
        self.assertIn("Status: fail", proc.stdout)

    def test_versions_prints_each_tracked_release_and_checks_nothing(self):
        versions = {"jax": "0.11.2", "torch_tpu": "0.1.1"}
        out = io.StringIO()
        with mock.patch.object(env_check,
                               "version_of",
                               side_effect=versions.get), \
                contextlib.redirect_stdout(out):
            code = env_check.main(["--versions"])
        self.assertEqual(code, 0)
        self.assertEqual(
            {name: versions.get(name)
             for name in env_check.TRACKED}, json.loads(out.getvalue()))

    def test_a_target_check_ends_with_the_releases_it_found(self):
        """`compute setup verify` records them from this line."""
        out = io.StringIO()
        with mock.patch.object(env_check, "version_of",
                               return_value=None), \
                contextlib.redirect_stdout(out):
            env_check.main(["--require=jax"])
        last = out.getvalue().splitlines()[-1]
        self.assertTrue(last.startswith(env_check.VERSIONS_PREFIX), last)
        self.assertIsNone(
            json.loads(last[len(env_check.VERSIONS_PREFIX):])["jax"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            env_check.main(["--host", "--require=jax"])
        self.assertNotIn(env_check.VERSIONS_PREFIX, out.getvalue())
        # Host mode skips the TPU software and devices.
        self.assertNotIn("torch_tpu", out.getvalue())


class TestConflicts(unittest.TestCase):
    """Tests for a requirement installed next to a conflicting distribution.

    The case under test is xprof-nightly installed next to xprof. Both
    install the same import package, and pip reports no conflict between
    them.
    """

    REQUIREMENTS = ["pyyaml>=6.0", "xprof>=2.24.0", "pydantic>=2.0.0"]
    BOTH = {
        "pyyaml": "6.0",
        "xprof": "2.24.0",
        "xprof-nightly": "2.25.0a20261017"
    }

    def _versions(self, versions):
        return mock.patch.object(env_check,
                                 "version_of",
                                 side_effect=versions.get)

    def test_the_check_and_fix_follow_what_is_installed(self):
        # The fix uninstalls only what is installed. Uninstalling xprof too
        # matters: xprof already meets its bound, so a plain install would
        # leave xprof-nightly in place.
        nightly = {"xprof-nightly": "2.25.0a20261017"}
        xprof_only = {"xprof": "2.24.0"}
        cases = [
            # (installed, status, fix_command)
            (self.BOTH, "fail", "/v/py -m pip uninstall -y xprof-nightly "
             "xprof && /v/py -m pip install 'xprof>=2.24.0'"),
            (nightly, "fail", "/v/py -m pip uninstall -y xprof-nightly && "
             "/v/py -m pip install 'xprof>=2.24.0'"),
            (xprof_only, "ok", "/v/py -m pip install 'xprof>=2.24.0'"),
        ]
        for installed, status, fix in cases:
            with self.subTest(installed=installed), self._versions(installed):
                self.assertEqual(
                    env_check.check_requirement("xprof>=2.24.0")[1], status)
                self.assertEqual(
                    env_check.fix_command(["xprof>=2.24.0"], python="/v/py"),
                    fix)
        with self._versions(self.BOTH):
            detail = env_check.check_requirement("xprof>=2.24.0")[2]
            image_fix = env_check.check_requirement("xprof>=2.24.0",
                                                    image=True)[3]
        self.assertIn("xprof-nightly 2.25.0a20261017", detail)
        self.assertIn(env_check.IMAGE_DOC, image_fix)
        self.assertNotIn("pip", image_fix)

    def test_the_session_note_uninstalls_before_it_installs(self):
        # The session start lists what fails and asks for one command. A
        # plain `pip install 'xprof>=2.24.0'` would leave xprof-nightly in
        # place, because the installed xprof already meets the bound.
        with self._versions(self.BOTH):
            failing = env_check.failing_requirements(self.REQUIREMENTS)
            note = env_check.ask_to_install(failing, python="/v/py")
        self.assertEqual(failing, ["xprof>=2.24.0", "pydantic>=2.0.0"])
        self.assertIn(
            "ask the user to run `/v/py -m pip uninstall -y xprof-nightly "
            "xprof && /v/py -m pip install 'xprof>=2.24.0' "
            "'pydantic>=2.0.0'`", note)

    def test_main_reports_the_conflict_and_changes_nothing(self):
        out = io.StringIO()
        with self._versions(self.BOTH), \
                mock.patch.object(subprocess, "run") as run, \
                contextlib.redirect_stdout(out):
            code = env_check.main(["--host", "--require=xprof>=2.24.0"])
        run.assert_not_called()
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] xprof", out.getvalue())
        # The fix names the interpreter that was checked, not a bare pip.
        self.assertIn(
            f"fix: {sys.executable} -m pip uninstall -y xprof-nightly xprof",
            out.getvalue())


if __name__ == "__main__":
    unittest.main()
