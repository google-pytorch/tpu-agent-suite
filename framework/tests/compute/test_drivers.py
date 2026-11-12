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
"""Unit tests for the compute drivers.

Covers plugin registration, driver CLI flags and auto-discovery, host probing,
the local job lifecycle, and the lifecycle GceTpuVmDriver and GkePodDriver
share, exercised once against both transports, plus each transport's own
commands.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import importlib
import json
import os
import pathlib
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from typing import Any
from unittest.mock import patch

# Imports below follow this sys.path setup, so the file runs directly.
# ruff: noqa: E402
_repo_root = pathlib.Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from framework.compute import (AsynchronousExecutionResult, ComputeConfigError,
                               GceTpuVmDriver, GkePodDriver, LocalDriver,
                               StagedFile, get_registered_driver_cli_flags,
                               resolve_driver)
from framework.compute.drivers import _auto_discover_drivers, base, jobs, probe
from framework.compute.drivers import registry as driver_registry
from framework.compute.drivers import shell
from framework.compute.drivers.registry import list_drivers
from framework.tests.compute.helpers import (MockTestDriver, ScriptMixin,
                                             completed, patch_home,
                                             set_up_module, tear_down_module,
                                             temp_home)

Path = pathlib.Path


def setUpModule():  # noqa: N802 - unittest's spelling
    set_up_module()


def tearDownModule():  # noqa: N802 - unittest's spelling
    tear_down_module()


def _decode_inner_script(ssh_command: str) -> str:
    """Pull the workload back out of the base64 payload `run` stages.

    `run` no longer passes the command inline: it writes a script to the
    target and runs that, so the command text a test wants to assert on is
    inside the base64 blob rather than in the SSH invocation.
    """
    blob = re.search(r"printf %s '([A-Za-z0-9+/=]+)'", ssh_command)
    assert blob, f"no staged script found in: {ssh_command}"
    return base64.b64decode(blob.group(1)).decode("utf-8")


class TestDriverRegistry(unittest.TestCase):
    """Driver registration, discovery, and CLI flag declaration."""

    def test_registered_backends_are_declared_in_schema(self):
        """Verify every shipped driver is a documented environment.json backend."""
        schema_path = (pathlib.Path(__file__).resolve().parents[3] /
                       "package" / "schemas" / "environment.json")
        if not schema_path.is_file():
            self.skipTest("package schema not available in this layout")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        compute_props = schema["properties"]["profiles"][
            "additionalProperties"]["properties"]
        declared = set(compute_props["backend"]["enum"])
        shipped = {n
                   for n in list_drivers() if not n.startswith("mock")} - set(
                       driver_registry.EXTERNAL_DRIVERS)
        self.assertTrue(
            shipped <= declared,
            f"backends missing from environment.json schema: {shipped - declared}"
        )

    def test_add_cli_arg_enforces_rules(self):
        """Verify driver defaults are stripped and reserved names are rejected."""
        parser = argparse.ArgumentParser()

        class _DefaultingDriver(MockTestDriver):

            @classmethod
            def register_cli_args(cls, target: Any) -> None:
                cls.add_cli_arg(target,
                                "--zone-probe",
                                default="us-central2-b",
                                help="zone")

        self.addCleanup(driver_registry.REGISTERED_DRIVER_CLI_FLAGS.discard,
                        "zone_probe")
        _DefaultingDriver.register_cli_args(parser)
        self.assertFalse(hasattr(parser.parse_args([]), "zone_probe"))

        with self.assertRaises(ValueError) as ctx:
            MockTestDriver.add_cli_arg(argparse.ArgumentParser(),
                                       "--workspace",
                                       help="clash")
        self.assertIn("reserved", str(ctx.exception))
        self.assertNotIn("workspace", get_registered_driver_cli_flags())

    def test_auto_discovery_handles_errors_and_subpackages(self):
        """Verify auto-discovery records import errors and discovers subpackages."""
        self.addCleanup(driver_registry.DISCOVERY_ERRORS.pop, "zz_broken",
                        None)
        with patch("pkgutil.iter_modules",
                   return_value=[(None, "zz_broken", False)]), \
             patch("importlib.import_module",
                   side_effect=RuntimeError("probe boom")):
            _auto_discover_drivers()
        errors = driver_registry.get_discovery_errors()
        self.assertIn("probe boom", errors["zz_broken"])
        # A driver that failed to import is named in INVALID_BACKEND, rather
        # than vanishing and leaving the user told only "invalid".
        with self.assertRaises(ComputeConfigError) as ctx:
            resolve_driver(backend_override="zz_broken")
        self.assertEqual(ctx.exception.error_code, "INVALID_BACKEND")
        self.assertIn("probe boom", ctx.exception.message)

        with patch("pkgutil.iter_modules",
                   return_value=[(None, "zz_probe_pkg", True)]), \
             patch("importlib.import_module") as mock_import:
            _auto_discover_drivers()
        parent_pkg = driver_registry.__name__.rsplit(".", 1)[0]
        mock_import.assert_called_with(f"{parent_pkg}.zz_probe_pkg")


_ACCEL2 = ["/dev/accel0", "/dev/accel1"]

_VFIO8 = [f"/dev/vfio/{i}" for i in range(8)]

_V6E2 = {"ACCELERATOR_TYPE": "v6e-2"}


class TestHostProbe(unittest.TestCase):
    """The host probe LocalDriver runs in-process and remote drivers ship out."""

    def setUp(self):
        self.driver = LocalDriver(python_interpreter=sys.executable)

    @contextlib.contextmanager
    def _probed(self,
                accel=(),
                vfio=(),
                pci=(0, None),
                meta=None,
                holders=([], set(), set()),
                locked=False):
        """Patch every input the host probe reads.

        The probe consults metadata, device nodes, PCI, and /proc. A test that
        stubs only some of those inherits the real machine for the rest, which
        is how these tests used to pass in CI and fail on a TPU host.
        """
        patches = (
            patch.object(probe, "read_metadata", return_value=meta or {}),
            patch.object(probe,
                         "list_devices",
                         return_value=(list(accel), list(vfio))),
            patch.object(probe, "scan_pci", return_value=pci),
            patch.object(probe, "scan_holders", return_value=holders),
            patch.object(probe, "_lockfile_is_locked", return_value=locked),
        )
        with contextlib.ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            yield

    def _holders(self, devices):
        """Build a scan_holders return value for the given held devices."""
        held = set(devices)
        return ([f"PID 42 (trainer) on {d}"
                 for d in sorted(held)], held, set())

    def _assert_profiles(self, cases):
        """Probe each `(name, probe_args, expected)`; check the fields."""
        for name, probe_args, expected in cases:
            with self.subTest(name):
                with self._probed(**probe_args):
                    profile = self.driver.inspect()
                for field, value in expected.items():
                    self.assertEqual(getattr(profile, field), value, field)

    def test_accelerator_identification(self):
        """Verify generation, chip count, topology, and HBM across device styles.

        A host with no devices reports 'none' and no capacity. tpu7x exposes
        two devices per chip, so its chip count is half the device
        count; GKE describes the same hardware through kube-label metadata
        rather than an ACCELERATOR_TYPE, and both must land on the same answer.
        """
        gke = {
            "cloud.google.com/gke-tpu-accelerator": "tpu-v6e-slice",
            "cloud.google.com/gke-tpu-topology": "2x4",
            "cloud.google.com/machine-family": "ct6e",
        }
        cases = [
            ("no devices", {},
             dict(accelerator="none", chips=0, status="no_accelerator")),
            ("v6e accel nodes", dict(accel=_ACCEL2, meta=_V6E2),
             dict(accelerator="v6e", chips=2, status="healthy")),
            ("tpu7x halves devices per chip",
             dict(vfio=_VFIO8,
                  meta={
                      "ACCELERATOR_TYPE": "tpu7x-8",
                      "TOPOLOGY": "2x2x1"
                  }),
             dict(accelerator="tpu7x",
                  chips=4,
                  topology="2x2x1",
                  hbm_gb=192.0,
                  available_chips=4)),
            ("gke kube-label metadata", dict(vfio=_VFIO8, meta=gke),
             dict(accelerator="v6e", chips=8, topology="2x4", hbm_gb=32.0)),
            ("unrecognised chip count is not guessed at",
             dict(accel=[f"/dev/accel{i}" for i in range(6)],
                  meta={"ACCELERATOR_TYPE":
                        "v6e-6"}), dict(chips=6, topology="unknown")),
        ]
        self._assert_profiles(cases)

    def test_contention_counts_chips_not_devices(self):
        """Verify availability subtracts held chips, converting devices first.

        Four held tpu7x devices are two physical chips, not four; a lockfile
        held by libtpu means nothing is usable however free the devices look.
        The holder descriptions reach the caller, not just a count.
        """
        cases = [
            ("some accel nodes held",
             dict(accel=_ACCEL2,
                  meta=_V6E2,
                  holders=self._holders(_ACCEL2[:1])),
             dict(available_chips=1,
                  status="partially_busy",
                  held_by=["PID 42 (trainer) on /dev/accel0"])),
            ("all accel nodes held",
             dict(accel=_ACCEL2, meta=_V6E2, holders=self._holders(_ACCEL2)),
             dict(available_chips=0, status="busy")),
            ("tpu7x devices convert to chips",
             dict(vfio=_VFIO8,
                  meta={"ACCELERATOR_TYPE": "tpu7x-8"},
                  holders=self._holders(_VFIO8[:4])),
             dict(chips=4, available_chips=2, status="partially_busy")),
            ("held lockfile overrides free devices",
             dict(accel=_ACCEL2, meta=_V6E2, locked=True),
             dict(available_chips=0,
                  status="busy",
                  held_by=["libtpu_lockfile is locked by an active process"])),
        ]
        self._assert_profiles(cases)

    def test_devices_and_hbm_per_device_are_reported(self):
        """Verify TP inputs match gke_job's keys, and unknown chips say why."""
        cases = [
            ("v6e is one device per chip",
             dict(accel=["/dev/accel0"], meta={"ACCELERATOR_TYPE": "v6e-1"}),
             dict(job_devices=1, hbm_gb_per_device=32.0)),
            ("tpu7x chips are two devices each",
             dict(vfio=[f"/dev/vfio/{i}" for i in range(8)],
                  meta={"ACCELERATOR_TYPE":
                        "tpu7x-4"}), dict(job_devices=8,
                                          hbm_gb_per_device=96.0)),
        ]
        for name, probe_args, expected in cases:
            with self.subTest(name):
                with self._probed(**probe_args):
                    details = self.driver.inspect().details
                for field, value in expected.items():
                    self.assertEqual(details[field], value, field)
                self.assertNotIn("hbm_gb_note", details)

        with self._probed(accel=["/dev/accel0"],
                          meta={"ACCELERATOR_TYPE": "tpu9z-1"}):
            details = self.driver.inspect().details
        self.assertNotIn("job_devices", details)
        self.assertNotIn("hbm_gb_per_device", details)
        self.assertIn("tpu9z", details["hbm_gb_note"])

    def test_list_devices_filters_unbacked_accel_nodes(self):
        """Verify static unbacked /dev/accel* nodes do not shadow /dev/vfio/[0-9]*."""

        def fake_glob(pattern: str) -> list[str]:
            if pattern == "/dev/vfio/[0-9]*":
                return [f"/dev/vfio/{i}" for i in range(806, 814)]
            if pattern.startswith("/dev/accel"):
                return ["/dev/accel-p2p"] + [
                    f"/dev/accel{i}" for i in range(396)
                ] + [f"/dev/accel_block{i}" for i in range(396)]
            return []

        with patch.object(probe.glob, "glob", side_effect=fake_glob):
            accel, vfio = probe.list_devices()
        self.assertEqual(accel, [])
        self.assertEqual(len(vfio), 8)

    def test_missing_tpu_info_raises_runtime_error(self):
        """Import must survive a missing tpu_info; only probing itself raises.

        The compute CLI import chain pulls this module in on controller
        machines that never probe local hardware; an import-time raise there
        takes down every runtime command, including hook dispatch.
        """
        with patch.dict(sys.modules, {
                "tpu_info": None,
                "tpu_info.device": None
        }):
            importlib.reload(probe)  # must not raise
            with self.assertRaises(probe.MissingPackageError) as ctx:
                probe.spec_for("v6e")
            self.assertEqual(ctx.exception.package, "tpu_info")
            self.assertEqual(
                str(ctx.exception),
                f"tpu_info is not installed for {sys.executable}")
        importlib.reload(probe)


def _terminate_quietly(pid: int) -> None:
    """Stop a test-spawned process group without failing if it already exited."""
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass


def _terminate_job_quietly(job_id: str) -> None:
    """Stop a submitted job's process group, if its record has one."""
    record = jobs.JobRecord.load(job_id)
    if record and record.pid:
        _terminate_quietly(record.pid)


class TestLocalDriver(ScriptMixin, unittest.TestCase):
    """LocalDriver's synchronous and detached execution on this host."""

    def setUp(self):
        self.driver = LocalDriver(python_interpreter=sys.executable)

    def test_shell_script_gets_the_interpreter_as_tpu_agent_suite_python(self):
        """Verify a .sh runs under bash with TPU_AGENT_SUITE_PYTHON set."""
        result = self.driver.run(
            self._script('echo "py=$TPU_AGENT_SUITE_PYTHON"\n', suffix=".sh"))
        self.assertEqual(result.exit_code, 0, result.stderr)
        self.assertIn(f"py={sys.executable}", result.stdout)

    def _await_terminal(self, job_id: str, timeout: float = 5.0):
        """Poll a submitted job until it reaches a terminal state."""
        deadline = time.monotonic() + timeout
        status = self.driver.get_status(job_id)
        while status.status not in ("COMPLETED",
                                    "FAILED") and time.monotonic() < deadline:
            time.sleep(0.05)
            status = self.driver.get_status(job_id)
        _terminate_job_quietly(job_id)
        return status

    @staticmethod
    def _record_declaring(job_id: str, name: str) -> jobs.JobRecord:
        """A local job record whose one declared output is `name`."""
        return jobs.JobRecord.create(
            job_id=job_id,
            backend="local",
            extra={"outputs": [{
                "local": name,
                "remote": name,
            }]})

    def test_declared_outputs_survive_a_missing_artifacts_dir(self):
        """Outputs sit beside `artifacts/`, so its absence must not hide them."""
        with temp_home():
            record = self._record_declaring("job-out-1", "results.json")
            (record.directory / "results.json").write_text('{"n": 42}')
            shutil.rmtree(record.directory / "artifacts")
            with tempfile.TemporaryDirectory() as dest:
                files = self.driver.fetch_artifacts("job-out-1", Path(dest))
                self.assertEqual([f.name for f in files], ["results.json"])
                self.assertEqual(files[0].read_text(), '{"n": 42}')

    def test_run_stages_inputs_and_outputs_like_a_remote(self):
        """Verify the local backend honours the same --input/--output spelling.

        A prompt written for a TPU VM names the payload by bare name and
        expects the result at a workspace path; running it on `local` must
        not require a second spelling.
        """
        work = pathlib.Path(tempfile.mkdtemp())
        payload = work / "in" / "payload.json"
        payload.parent.mkdir()
        payload.write_text(json.dumps({"n": 41}))
        script = self._script(
            "import json, sys, os\n"
            "n = json.load(open(sys.argv[1]))['n']\n"
            "json.dump({'n': n + 1, 'cwd': os.getcwd()}, open(sys.argv[2], 'w'))\n"
        )
        result_local = work / "out" / "results.json"

        res = self.driver.run(
            script,
            args=["payload.json", "results.json"],
            inputs=[StagedFile(local=str(payload), remote="payload.json")],
            outputs=[
                StagedFile(local=str(result_local), remote="results.json")
            ],
        )
        self.assertEqual(res.exit_code, 0, res.stderr)
        produced = json.loads(result_local.read_text())
        self.assertEqual(produced["n"], 42)
        # Ran in a scratch directory, not the caller's, and that directory
        # did not outlive the run.
        self.assertNotEqual(produced["cwd"], os.getcwd())
        self.assertFalse(pathlib.Path(produced["cwd"]).exists())

    def test_run_kills_the_whole_process_tree_on_timeout(self):
        """A timed-out workload must not leave children holding the device.

        `subprocess.run` kills the direct child and nothing below it. The
        workload this package times out most often -- an autotune sweep --
        spawns a subprocess per trial, so the children are the part that
        matters: orphaned, they keep the accelerator and the next run collides
        with them.

        The assertion is on elapsed time, because killing only the parent has
        two symptoms and this is the sharper one: the grandchild inherits the
        stdout pipe, so `communicate()` blocks until it exits on its own. The
        orphan does not merely survive, it hangs the caller for as long as it
        runs. A test that only checked "is the pid gone" would pass against the
        broken version too -- eventually.
        """
        marker = pathlib.Path(tempfile.mkdtemp()) / "child_alive.txt"
        child_lifetime = 30
        path = self._script(
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c',\n"
            f"    \"import time,pathlib;\"\n"
            f"    \"pathlib.Path(r'{marker}').write_text('1');\"\n"
            f"    \"time.sleep({child_lifetime})\"])\n"
            "print(child.pid, flush=True)\n"
            f"time.sleep({child_lifetime})\n")

        started = time.monotonic()
        res = self.driver.run(path, timeout=3)
        elapsed = time.monotonic() - started

        self.assertEqual(res.exit_code, 124)
        self.assertIn("process group was terminated", res.stderr)
        self.assertLess(
            elapsed, child_lifetime / 2,
            f"run() took {elapsed:.1f}s for a 3s timeout: it waited on a "
            "grandchild that should have been killed with the group")

        # The grandchild must have started, or the test proves nothing.
        self.assertTrue(marker.is_file(), "the grandchild never started")

        # Poll rather than check once: the signal lands immediately but the pid
        # survives as a zombie until init reaps it, which is not instant.
        child_pid = int(res.stdout.split()[0])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except OSError:
                break
            time.sleep(0.05)
        else:
            self.fail("the grandchild survived and still holds the device")

    def test_unreadable_input_reports_permission_cause(self):
        """Verify an inaccessible parent directory reports why it could not be stat'd."""
        with patch.object(Path,
                          "is_file",
                          side_effect=PermissionError(13,
                                                      "Permission denied")):
            with self.assertRaises(ComputeConfigError) as ctx:
                self.driver.run("/restricted/dir/script.py")
            self.assertEqual(ctx.exception.error_code, "INPUT_NOT_FOUND")
            self.assertIn("is not accessible", ctx.exception.message)
            self.assertIn("Permission denied", ctx.exception.message)

    def test_submitted_job_tracks_status_logs_and_exit_code(self):
        """Verify a detached job reaches a terminal state with its output kept."""
        for name, exit_code, expected in (("success", 0, "COMPLETED"),
                                          ("failure", 42, "FAILED")):
            with self.subTest(name):
                path = self._script(
                    f"import sys; print('logged_output'); sys.exit({exit_code})\n"
                )
                # Through ComputeDriver._new_record, which every driver uses.
                self.driver.compute_profile = "vm"
                job_id = self.driver.submit(path)
                self.assertTrue(job_id.startswith("job-"))
                self.assertEqual(
                    jobs.JobRecord.load(job_id).compute_profile, "vm")

                status = self._await_terminal(job_id)
                self.assertIsInstance(status, AsynchronousExecutionResult)
                self.assertEqual(status.status, expected)
                self.assertEqual(status.exit_code, exit_code)
                self.assertIn("logged_output", self.driver.fetch_logs(job_id))

    def test_a_reused_pid_is_neither_running_nor_signalled(self):
        """A pid now held by another process is not the job, and is left be.

        The reapers run long after submit, even after a reboot, when a
        recorded pid can belong to anything; SIGTERM must not reach it.
        """
        job_id = self.driver.submit(
            self._script("import time\ntime.sleep(10)\n"))
        record = jobs.JobRecord.load(job_id)
        self.addCleanup(_terminate_quietly, record.pid)
        self.assertEqual(record.pid_started, jobs.process_started(record.pid))
        record.pid_started = "another process"
        record.save()

        with patch("os.killpg") as killpg, patch("os.kill") as kill:
            self.driver.cancel(job_id)
        killpg.assert_not_called()
        self.assertNotIn(signal.SIGTERM, [c.args[1] for c in kill.mock_calls])
        self.assertNotEqual(jobs.JobRecord.load(job_id).status, "RUNNING")


_PROBE_PAYLOAD = {
    "accelerator": "v4",
    "topology": "2x2x1",
    "chips": 4,
    "hbm_gb": 32.0,
    "status": "healthy",
    "available_chips": 4,
    "held_by": [],
    "details": {
        "accelerator_type": "v4-8"
    },
}


def _shell_drivers() -> dict[str, Any]:
    """The RemoteShellDriver subclasses, keyed by backend name."""
    return {
        "gce_tpu_vm":
        lambda: GceTpuVmDriver(
            tpu_name="test-tpu", zone="us-central2-b", project="test-proj"),
        "gke_pod":
        lambda: GkePodDriver(namespace="default", pod="tpu-pod"),
    }


class TestRemoteShellDrivers(ScriptMixin, unittest.TestCase):
    """The job lifecycle GceTpuVmDriver and GkePodDriver inherit and share.

    Both reach their target as "run this shell command there", so the launch
    wrapper, the pid/exit-code status protocol, log retrieval, and cancellation
    are one implementation; these run it against both transports.
    """

    @patch("subprocess.run")
    def test_unreachable_host_is_not_free_capacity(self, mock_run):
        """Verify a host that could not be probed reports unknown availability.

        A missing `tpu_info` looks identical to a dead ssh link in the status
        field; the interpreter in `details` is what points at the venv.
        """
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.return_value = completed(
                    returncode=1, stderr="RuntimeError: tpu_info is required")
                driver = make_driver()
                driver.python_interpreter = "/opt/venv/bin/python3"
                profile = driver.inspect()
                self.assertEqual(profile.status, "unreachable")
                self.assertIsNone(profile.available_chips)
                self.assertEqual(profile.chips, 0)
                self.assertEqual(profile.details["python_interpreter"],
                                 "/opt/venv/bin/python3")
                self.assertIn("tpu_info", profile.details["error"])

    @patch("subprocess.run")
    def test_probe_runs_under_the_configured_interpreter(self, mock_run):
        """Verify inspect probes with `python_interpreter`, like run does.

        `tpu_info` is installed into the user's venv per `requirements-tpu.txt`,
        so a probe that shells out to the target's bare `python3` reports a
        healthy VM as unreachable while `run` on the same VM succeeds.
        """
        venv_python = "~/tpu_venv/bin/python3"
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.reset_mock()
                mock_run.return_value = completed(json.dumps(_PROBE_PAYLOAD))
                driver = make_driver()
                driver.python_interpreter = venv_python
                driver.inspect()
                remote = " ".join(mock_run.call_args.args[0])
                self.assertIn(f"{venv_python} -c ", remote)
                self.assertNotIn(" python3 -c ", remote)

    @patch("subprocess.run")
    def test_run_stages_inputs_and_copies_outputs_back(self, mock_run):
        """Verify --input files land beside the script and --output files return.

        The autotune and profiling prompts hand a script a payload by bare
        name and read a result by bare name; on a remote backend that only
        works if both travel with the script and it runs where they landed.
        """
        payload = self._script("{}", suffix=".json")
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                out_local = Path(tempfile.mkdtemp()) / "nested" / "r.json"
                calls = []

                def fake_run(cmd, *a, **k):
                    calls.append(" ".join(str(x) for x in cmd))
                    # The copy-out names the local path; "deliver" the file.
                    if str(out_local) in cmd:
                        out_local.parent.mkdir(parents=True, exist_ok=True)
                        out_local.write_text("{}")
                    return completed("ran\n" if len(calls) == 4 else "")

                mock_run.side_effect = fake_run
                res = make_driver().run(
                    self._script("print('hi')\n"),
                    args=["payload.json", "r.json"],
                    inputs=[StagedFile(local=payload, remote="payload.json")],
                    outputs=[
                        StagedFile(local=str(out_local), remote="r.json")
                    ],
                )
                self.assertEqual((res.exit_code, res.stdout), (0, "ran\n"))
                self.assertTrue(out_local.is_file())
                # The script ran inside the run directory, by bare name, with
                # the payload copied to that same directory. The workload
                # travels base64-encoded, so assert on what the target will
                # actually execute rather than on the SSH command around it.
                inner = _decode_inner_script(calls[3])
                self.assertIn("cd /tmp/tpu_agent_suite_run_", inner)
                self.assertIn("script.py payload.json r.json", inner)
                self.assertIn("/payload.json", calls[2])

    @patch("subprocess.run")
    def test_run_reaps_the_remote_process_when_it_times_out(self, mock_run):
        """A dead SSH client does not stop the workload; it must be killed.

        This is the failure that silently costs an accelerator: the client
        gives up, the caller sees exit 124 and moves on, and the workload is
        still running on the target holding the chips the next run wants.
        """
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                calls = []

                def fake_run(cmd, **kwargs):
                    joined = " ".join(str(x) for x in cmd)
                    calls.append(joined)
                    if "base64 -d" in joined:
                        raise subprocess.TimeoutExpired(cmd, 5)
                    return completed("REAPED\n")

                mock_run.side_effect = fake_run
                res = make_driver().run(self._script("print('hi')\n"),
                                        timeout=5)

                self.assertEqual(res.exit_code, 124)
                self.assertIn("timed out", res.stderr)
                self.assertIn("process group was terminated", res.stderr)
                reap = calls[-1]
                self.assertIn("kill -TERM", reap)
                self.assertIn("kill -KILL", reap)
                self.assertIn("ps -o pgid=", reap)
                mock_run.reset_mock()

    @patch("subprocess.run")
    def test_runs_serialise_on_the_target_lock_and_busy_never_started(
            self, mock_run):
        """Concurrent runs against one accelerator must not interleave.

        Losing the lock race is not a failed run, nor may it read as one.
        """
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.side_effect = [
                    completed(),
                    completed(returncode=shell.LOCK_BUSY_EXIT),
                ]
                res = make_driver().run(self._script("print('hi')\n"),
                                        timeout=600)
                execute = " ".join(
                    str(x) for x in mock_run.call_args_list[-1].args[0])
                self.assertIn(shell.RUN_LOCK_PATH, execute)
                # Half the call's timeout, so a contended target still leaves
                # the workload most of its budget.
                self.assertIn("flock -w 300", execute)
                # Absent flock, the run proceeds rather than failing.
                self.assertIn("command -v flock", execute)
                self.assertEqual(res.exit_code, shell.LOCK_BUSY_EXIT)
                self.assertIn("busy", res.stderr)
                self.assertIn("no measurement was taken", res.stderr)
                mock_run.reset_mock()

    @patch("subprocess.run")
    def test_cancel_of_a_finished_job_keeps_its_exit_status(self, mock_run):
        """The status probe says the job already ended; cancel records that."""
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.return_value = completed()
                driver = make_driver()
                job_id = driver.submit(self._script("echo hi\n", suffix=".sh"))
                mock_run.return_value = completed("COMPLETED 0\n")
                driver.cancel(job_id)
                self.assertEqual(
                    jobs.JobRecord.load(job_id).status, "COMPLETED")
                for call in mock_run.call_args_list:
                    self.assertNotIn("kill -TERM", " ".join(call[0][0]))

    @patch("subprocess.run")
    def test_cancel_that_cannot_reach_the_target_raises(self, mock_run):
        """Unreachable target: the record stays live rather than lying."""
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.return_value = completed()
                driver = make_driver()
                job_id = driver.submit(self._script("echo hi\n", suffix=".sh"))
                mock_run.side_effect = [
                    completed("RUNNING\n"),
                    completed(returncode=255, stderr="ssh: connect refused"),
                ]
                with self.assertRaises(ComputeConfigError) as ctx:
                    driver.cancel(job_id)
                mock_run.side_effect = None
                self.assertEqual(ctx.exception.error_code, "CANCEL_FAILED")
                self.assertIn("connect refused", str(ctx.exception))
                self.assertEqual(jobs.JobRecord.load(job_id).status, "RUNNING")

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "reads /proc")
    def test_only_the_launched_process_counts_as_the_job(self):
        """The target-side check behind status and cancel, run for real."""
        proc = subprocess.Popen(["sleep", "30"])
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        job_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, job_dir)
        (job_dir / "pid.txt").write_text(f"{proc.pid}\n")
        # The target reads field 22 of /proc/<pid>/stat, as process_started.
        (job_dir /
         "pid_started.txt").write_text(f"{jobs.process_started(proc.pid)}\n")

        def live() -> bool:
            check = f"{shell._live_pid(str(job_dir))} echo \"$LIVE\""
            return subprocess.run(["sh", "-c", check],
                                  capture_output=True,
                                  text=True).stdout.strip() == "1"

        self.assertTrue(live())
        (job_dir / "pid_started.txt").write_text("1\n")  # pid reused
        self.assertFalse(live())
        (job_dir / "pid_started.txt").unlink()  # launched before the file
        self.assertTrue(live())

    @patch("subprocess.run")
    def test_cancelled_job_stays_cancelled(self, mock_run):
        """Verify cancelling then polling does not relabel the job as FAILED.

        Killing the remote process makes the status probe report `FAILED 1`;
        cancellation is the truer account of why it stopped.
        """
        mock_run.return_value = completed()
        driver = GkePodDriver(namespace="default", pod="tpu-pod")
        job_id = driver.submit(self._script("echo hi\n", suffix=".sh"))
        driver.cancel(job_id)

        mock_run.return_value = completed("FAILED 1\n")
        self.assertEqual(driver.get_status(job_id).status, "CANCELLED")
        _terminate_job_quietly(job_id)

    def test_gce_internal_ip_reaches_ssh_and_scp_only(self):
        """Verify internal_ip adds --internal-ip to ssh and scp, not describe."""
        driver = GceTpuVmDriver.from_params({
            "tpu_name": "t",
            "zone": "z",
            "project": "p",
            "internal_ip": "true"
        })
        self.assertIn("--internal-ip", driver._gcloud_base("ssh"))
        self.assertNotIn("--internal-ip", driver._gcloud_base("describe"))
        with patch("subprocess.run") as run:
            driver._scp("a", "t:b")
        self.assertIn("--internal-ip", run.call_args.args[0])

    def test_gke_pod_command_carries_namespace_and_container(self):
        """Verify kubectl calls are scoped to the pod's namespace and container."""
        driver = GkePodDriver(namespace="ns-1", pod="p-1", container="trainer")
        cmd = driver._exec_prefix()
        self.assertEqual(cmd[:5], ["kubectl", "-n", "ns-1", "exec", "p-1"])
        self.assertEqual(cmd[5:], ["-c", "trainer"])
        # A pinned context goes on every call; unset, none does.
        pinned = GkePodDriver.from_params({"pod": "p-1", "kube_context": "c"})
        self.assertEqual(pinned._kubectl_base(),
                         ["kubectl", "--context", "c", "-n", "default"])


def _free_port() -> int:
    """A loopback port nothing is bound to, for a forward to bind or miss."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class TestSpawnPortForward(unittest.TestCase):
    """The liveness check behind every backgrounded port forward."""

    def setUp(self):
        patch_home(self)

    def test_a_forward_that_binds_its_port_is_ready(self):
        """The socket answering is the proof; the poll ends as soon as it does."""
        script = ("import socket, time;"
                  " s = socket.socket();"
                  " s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1);"
                  " s.bind(('127.0.0.1', PORT)); s.listen(5); time.sleep(30)")
        port = _free_port()
        pid = base.spawn_port_forward(
            [sys.executable, "-c",
             script.replace("PORT", str(port))],
            "ready-1",
            local_port=port,
        )
        self.addCleanup(_terminate_quietly, pid)
        self.assertGreater(pid, 0)

    def test_a_forward_that_never_binds_is_not_called_established(self):
        """Alive is not the same as serving -- this is the silent failure."""
        port = _free_port()
        with self.assertRaises(ComputeConfigError) as ctx:
            base.spawn_port_forward(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                "never-1",
                local_port=port,
                ready_timeout_sec=0.5,
            )
        self.assertEqual(ctx.exception.error_code, "PORT_FORWARD_FAILED")
        self.assertIn("nothing is listening", str(ctx.exception))

    def test_a_forward_that_dies_late_is_still_caught(self):
        """The old fixed sleep returned a pid for anything slower than it.

        The pid alone proves nothing: Popen returns one either way.
        """
        port = _free_port()
        with self.assertRaises(ComputeConfigError) as ctx:
            base.spawn_port_forward([
                sys.executable, "-c", "import sys, time; time.sleep(1.5);"
                " sys.stderr.write('ssh: connect to host: timed out\\n');"
                " sys.exit(255)"
            ],
                                    "late-1",
                                    local_port=port,
                                    ready_timeout_sec=10.0)
        self.assertEqual(ctx.exception.error_code, "PORT_FORWARD_FAILED")
        self.assertIn("code 255", str(ctx.exception))
        self.assertIn("timed out", str(ctx.exception))
        # The output is kept, not sent to /dev/null, which is why this
        # failure used to be undiagnosable.
        log_path = Path(str(ctx.exception).rsplit("Log: ", 1)[1].strip())
        self.assertIn("timed out", log_path.read_text())


if __name__ == "__main__":
    unittest.main()
