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
"""Unit tests for the framework compute runtime layer.

Test Coverage Overview:
- TestDriverRegistry: plugin registration, driver CLI flags, and driver
  auto-discovery.
- TestLocalDriver: host probing and the local job lifecycle.
- TestRemoteShellDrivers: the lifecycle GceTpuVmDriver and GkePodDriver share,
  exercised once against both transports, plus each transport's own commands.
- TestComputeConfig / TestEnvironmentResolution: 3-tier configuration
  resolution, environment.json schemas, and CLI override precedence.
- TestComputeCli / TestContractGeneration: `runtime.py compute` behaviour,
  output envelopes, and the generated agent contract.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import importlib
import io
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
import threading
import time
import types
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

# Allow imports both from within the repository root and standalone
_framework_dir = pathlib.Path(__file__).resolve().parents[1]
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))
_repo_root = pathlib.Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from framework import compute, paths, run_state, runtime, utils
    from framework.builder import prompt_contracts
    from framework.compute import (AsynchronousExecutionResult,
                                   ComputeConfigError, ComputeDriver,
                                   GceTpuVmDriver, GkeJobDriver, GkePodDriver,
                                   HardwareProfile, LocalDriver, StagedFile,
                                   SynchronousExecutionResult)
    from framework.compute import config as compute_config
    from framework.compute import (get_driver, get_registered_driver_cli_flags,
                                   get_registered_drivers, register_driver,
                                   resolve_driver)
    from framework.compute.drivers import (_auto_discover_drivers, auth_utils,
                                           base, external, gke_pod, jobs,
                                           probe)
    from framework.compute.drivers import registry as driver_registry
    from framework.compute.drivers import shell
    from framework.compute.drivers.registry import (DRIVER_REGISTRY,
                                                    list_drivers)
except ImportError:
    import compute
    import run_state
    import runtime
    import utils
    from builder import prompt_contracts
    from compute import (AsynchronousExecutionResult, ComputeConfigError,
                         ComputeDriver, GceTpuVmDriver, GkeJobDriver,
                         GkePodDriver, HardwareProfile, LocalDriver,
                         StagedFile, SynchronousExecutionResult)
    from compute import config as compute_config
    from compute import (get_driver, get_registered_driver_cli_flags,
                         get_registered_drivers, register_driver,
                         resolve_driver)
    from compute.drivers import (_auto_discover_drivers, auth_utils, base,
                                 external, gke_pod, jobs, probe)
    from compute.drivers import registry as driver_registry
    from compute.drivers import shell
    from compute.drivers.registry import DRIVER_REGISTRY, list_drivers

Path = pathlib.Path

# The compute CLI falls back to the session leased to the harness process
# above it. Run from inside an agent session, that would be the developer's
# own run; tests that exercise the fallback patch it back in themselves.
_NO_SESSION = patch.object(compute.cli.lease, "session_run", return_value=None)

# Nor may a test reach the developer's real state some other way. Jobs and runs
# are recorded in the central store, and a command given no workspace finds one
# by climbing from the current directory, which in a checkout reaches its
# active run. Both point into a temp directory for the whole module; a test that
# needs its own store still patches `TPU_AGENT_SUITE_HOME` over this one.
_ISOLATION = contextlib.ExitStack()


def setUpModule():  # noqa: N802 - unittest's spelling
    _NO_SESSION.start()
    scratch = Path(tempfile.mkdtemp())
    _ISOLATION.callback(shutil.rmtree, scratch, ignore_errors=True)
    _ISOLATION.enter_context(
        patch.dict(os.environ,
                   {"TPU_AGENT_SUITE_HOME": str(scratch / "store")}))
    # chdir rather than patching Path.cwd, so that subprocesses the tests start
    # resolve their workspace inside the temp directory too.
    _ISOLATION.callback(os.chdir, os.getcwd())
    os.chdir(scratch)


def tearDownModule():  # noqa: N802 - unittest's spelling
    _ISOLATION.close()
    _NO_SESSION.stop()


def profile_env(backend: str, **params: Any) -> dict[str, Any]:
    """An environment.json holding one profile, `default`, the default."""
    profile = {"backend": backend, **({"params": params} if params else {})}
    return {"default": "default", "profiles": {"default": profile}}


def _cli_args(**overrides: Any) -> argparse.Namespace:
    """`compute` arguments as the parser leaves them, for `pkg@1`."""
    return argparse.Namespace(
        **{
            "compute_action": "inspect",
            "backend": None,
            "package": "pkg@1",
            "workspace": None,
            "json": True,
            "profile": None,
            "run_id": None,
            "job_id": None,
            **overrides
        })


def _decode_inner_script(ssh_command: str) -> str:
    """Pull the workload back out of the base64 payload `run` stages.

    `run` no longer passes the command inline: it writes a script to the
    target and runs that, so the command text a test wants to assert on is
    inside the base64 blob rather than in the SSH invocation.
    """
    blob = re.search(r"printf %s '([A-Za-z0-9+/=]+)'", ssh_command)
    assert blob, f"no staged script found in: {ssh_command}"
    return base64.b64decode(blob.group(1)).decode("utf-8")


def _completed(stdout: str = "",
               returncode: int = 0,
               stderr: str = "") -> subprocess.CompletedProcess:
    """Build a CompletedProcess stand-in for a mocked subprocess call."""
    return subprocess.CompletedProcess(args=[],
                                       returncode=returncode,
                                       stdout=stdout,
                                       stderr=stderr)


def _patch_home(test: unittest.TestCase) -> Path:
    """Point the central store at a fresh temp dir for the rest of `test`."""
    home = Path(tempfile.mkdtemp())
    test.addCleanup(shutil.rmtree, home, ignore_errors=True)
    patcher = patch.dict(os.environ, {"TPU_AGENT_SUITE_HOME": str(home)})
    patcher.start()
    test.addCleanup(patcher.stop)
    return home


@contextlib.contextmanager
def _temp_home():
    """Point the central store at a fresh temp dir inside the block."""
    with tempfile.TemporaryDirectory() as home, patch.dict(
            os.environ, {"TPU_AGENT_SUITE_HOME": home}):
        yield Path(home)


# A complete gke_job profile, as the answers `compute setup apply` takes.
_GKE_ANSWERS = {
    "backend": "gke_job",
    "image": "us-docker.pkg.dev/p/r/img:v1",
    "tpu_accelerator": "tpu7x",
    "tpu_topology": "2x2x1",
}
# A well-formed run id.
_RUN = "20260101T000000Z-123456789abc"
# A complete gce_tpu_vm profile.
_GCE_ANSWERS = {
    "backend": "gce_tpu_vm",
    "project": "p",
    "zone": "us-east5-a",
    "tpu_name": "v",
}


class _ScriptMixin:
    """Writes throwaway scripts that are removed when the test finishes."""

    def _script(self, body: str, suffix: str = ".py") -> str:
        with tempfile.NamedTemporaryFile("w", suffix=suffix,
                                         delete=False) as handle:
            handle.write(body)
            path = handle.name
        self.addCleanup(os.remove, path)
        return path


@register_driver("mock_test")
class MockTestDriver(ComputeDriver):
    """Mock compute driver used to verify base framework contracts and CLI routing."""

    def __init__(self, endpoint: str = "default-endpoint"):
        self.endpoint = endpoint
        self.python_interpreter = "python3"

    @classmethod
    def from_params(cls, params: dict) -> MockTestDriver:
        return cls(params.get("endpoint", "default-endpoint"))

    @classmethod
    def register_cli_args(cls, parser: Any) -> None:
        cls.add_cli_arg(parser, "--endpoint", type=str, help="Mock endpoint")

    def inspect(self) -> HardwareProfile:
        return HardwareProfile("mock_tpu", "2x2x1", 4, 64.0, "healthy")

    def run(self, code_path: str, *args,
            **kwargs) -> SynchronousExecutionResult:
        return SynchronousExecutionResult(0, f"mock run executed {code_path}",
                                          "")

    def submit(self, recipe_path: str, *args, **kwargs) -> str:
        return "mock-job-999"

    def get_status(self, job_id: str) -> AsynchronousExecutionResult:
        return AsynchronousExecutionResult(job_id=job_id, status="RUNNING")

    def fetch_logs(self, job_id: str) -> str:
        return f"mock logs for {job_id}"

    def fetch_artifacts(self, job_id: str, dest_dir: str) -> list[str]:
        return []

    def cancel(self, job_id: str) -> None:
        pass


class TestDriverRegistry(unittest.TestCase):
    """Driver registration, discovery, and CLI flag declaration."""

    def test_registered_backends_are_declared_in_schema(self):
        """Verify every shipped driver is a documented environment.json backend."""
        schema_path = (pathlib.Path(__file__).resolve().parent.parent.parent /
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

# A minimal external driver. It reuses the local driver for execution and
# adds one required setup field, `target`, so that profile validation has a
# backend-specific field to check.
_EXTERNAL_DRIVER_SOURCE = """
from framework.compute.drivers import base, local, registry


@registry.register_driver("{name}")
class ExternalProbeDriver(local.LocalDriver):
    SETUP_FIELDS = base.SetupFields(required=("target",))

    def __init__(self, target):
        super().__init__()
        self.target = target

    @classmethod
    def from_params(cls, params):
        if not params.get("target"):
            raise base.ComputeConfigError("{name} requires target",
                                          error_code="MISSING_FIELD")
        return cls(params["target"])
"""


class TestExternalDrivers(unittest.TestCase):
    """Loading compute drivers from outside the package (drivers/external.py)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        # Each test gets an empty store and no environment variable, so the
        # developer's own external drivers are never loaded here.
        env = patch.dict(
            os.environ, {
                "TPU_AGENT_SUITE_HOME": str(self.tmp / "store"),
                external.ENV_VAR: "",
            })
        env.start()
        self.addCleanup(env.stop)
        repo_patch = patch.object(external,
                                  "repo_google_drivers_dir",
                                  return_value=None)
        repo_patch.start()
        self.addCleanup(repo_patch.stop)
        dict_registries = (
            DRIVER_REGISTRY,
            driver_registry.EXTERNAL_DRIVERS,
            driver_registry.DISCOVERY_ERRORS,
            driver_registry.DRIVER_CLI_FLAGS,
            driver_registry.DRIVER_REQUIRED_FLAGS,
        )
        dict_snapshots = [{
            k: (list(v) if isinstance(v, list) else
                set(v) if isinstance(v, set) else v)
            for k, v in r.items()
        } for r in dict_registries]
        cli_flags_snapshot = set(driver_registry.REGISTERED_DRIVER_CLI_FLAGS)
        modules = set(sys.modules)

        def restore():
            for live, snapshot in zip(dict_registries, dict_snapshots):
                live.clear()
                live.update(snapshot)
            driver_registry.REGISTERED_DRIVER_CLI_FLAGS.clear()
            driver_registry.REGISTERED_DRIVER_CLI_FLAGS.update(
                cli_flags_snapshot)
            for name in set(sys.modules) - modules:
                if name.startswith(external.MODULE_PREFIX):
                    del sys.modules[name]

        self.addCleanup(restore)

    def write_driver(self, path: Path, name: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_EXTERNAL_DRIVER_SOURCE.format(name=name),
                        encoding="utf-8")
        return path

    def load(self, *env_paths: Path) -> None:
        with patch.dict(
                os.environ,
            {external.ENV_VAR: os.pathsep.join(str(p) for p in env_paths)}):
            external.load_external_drivers()

    def test_loads_file_named_in_env_var(self):
        source = self.write_driver(self.tmp / "ext_a.py", "ext_a")
        self.load(source)
        self.assertIn("ext_a", compute_config.get_valid_backends())
        self.assertEqual(driver_registry.EXTERNAL_DRIVERS["ext_a"],
                         str(source.resolve()))
        driver = resolve_driver(backend_override="ext_a",
                                param_overrides={"target": "t1"})
        self.assertEqual(driver.target, "t1")

    def test_loads_store_directory_and_skips_private_names(self):
        drivers_dir = external.default_drivers_dir()
        self.write_driver(drivers_dir / "ext_b.py", "ext_b")
        self.write_driver(drivers_dir / "_helper.py", "ext_hidden")
        (drivers_dir / "notes.txt").write_text("not a driver")
        self.load()
        self.assertIn("ext_b", list_drivers())
        self.assertNotIn("ext_hidden", list_drivers())
        self.assertFalse([
            k for k in driver_registry.DISCOVERY_ERRORS
            if k.startswith(str(drivers_dir))
        ])

    def test_loads_package_directory_with_relative_import(self):
        package = self.tmp / "ext_pkg"
        package.mkdir()
        (package / "helpers.py").write_text('NAME = "ext_c"\n')
        (package / "__init__.py"
         ).write_text("from .helpers import NAME\n" +
                      _EXTERNAL_DRIVER_SOURCE.format(name="ext_c").replace(
                          'register_driver("ext_c")', "register_driver(NAME)"))
        self.load(package)
        self.assertIn("ext_c", list_drivers())

    def test_missing_env_path_is_reported(self):
        missing = self.tmp / "absent.py"
        self.load(missing)
        self.assertIn(str(missing), driver_registry.DISCOVERY_ERRORS)
        with self.assertRaises(ComputeConfigError) as ctx:
            resolve_driver(backend_override="absent")
        self.assertIn(str(missing), ctx.exception.message)

    def test_builtin_backend_name_is_rejected(self):
        source = self.write_driver(self.tmp / "takeover.py", "local")
        self.load(source)
        self.assertIs(get_driver("local"), LocalDriver)
        self.assertNotIn("local", driver_registry.EXTERNAL_DRIVERS)
        self.assertIn("already uses",
                      driver_registry.DISCOVERY_ERRORS[str(source.resolve())])

    def test_source_that_registers_nothing_is_reported(self):
        source = self.tmp / "empty.py"
        source.write_text("VALUE = 1\n")
        self.load(source)
        self.assertIn("registers no backend",
                      driver_registry.DISCOVERY_ERRORS[str(source.resolve())])

    def test_import_failure_rolls_back_partial_registration(self):
        source = self.write_driver(self.tmp / "broken.py", "ext_broken")
        source.write_text(source.read_text() +
                          "\nraise RuntimeError('boom')\n")
        self.load(source)
        self.assertNotIn("ext_broken", list_drivers())
        self.assertIn("boom",
                      driver_registry.DISCOVERY_ERRORS[str(source.resolve())])

    def test_duplicate_module_name_keeps_first(self):
        first = self.write_driver(self.tmp / "a" / "ext_dup.py", "ext_dup_a")
        second = self.write_driver(self.tmp / "b" / "ext_dup.py", "ext_dup_b")
        self.load(first, second)
        self.assertIn("ext_dup_a", list_drivers())
        self.assertNotIn("ext_dup_b", list_drivers())
        self.assertIn("already loaded",
                      driver_registry.DISCOVERY_ERRORS[str(second.resolve())])

    def test_setup_accepts_external_backend_and_fields(self):
        profile = {"backend": "ext_d", "params": {"target": "t1"}}
        self.write_driver(self.tmp / "ext_d.py", "ext_d")
        before = compute.setup.validate_profile("p", profile)
        self.assertNotEqual(before["status"], "ok")
        self.load(self.tmp / "ext_d.py")
        self.assertEqual(
            compute.setup.validate_profile("p", profile)["status"], "ok")
        missing = compute.setup.validate_profile("p", {
            "backend": "ext_d",
            "params": {}
        })
        self.assertEqual([q["name"] for q in missing["missing"]], ["target"])

    def test_failed_driver_rolls_back_cli_flags(self):
        source = self.tmp / "flag_leak.py"
        source.write_text(
            _EXTERNAL_DRIVER_SOURCE.format(name="ext_leak") + """
import argparse
parser = argparse.ArgumentParser()
ExtDriver.add_cli_arg(parser, "--leaked-flag", dest="leaked_flag", help="leak")
raise RuntimeError("boom after cli flag registration")
""",
            encoding="utf-8",
        )
        self.load(source)
        self.assertNotIn("ext_leak", list_drivers())
        self.assertNotIn("leaked_flag",
                         driver_registry.REGISTERED_DRIVER_CLI_FLAGS)
        self.assertFalse(
            any("--leaked-flag" in flags
                for flags in driver_registry.DRIVER_CLI_FLAGS.values()))

    def test_idempotent_reload_of_same_source(self):
        source = self.write_driver(self.tmp / "ext_idem.py", "ext_idem")
        self.load(source)
        self.assertIn("ext_idem", list_drivers())
        self.assertNotIn(str(source.resolve()),
                         driver_registry.DISCOVERY_ERRORS)
        # Loading the exact same source a second time must be a no-op rather
        # than recording a self-collision in DISCOVERY_ERRORS.
        self.load(source)
        self.assertIn("ext_idem", list_drivers())
        self.assertNotIn(str(source.resolve()),
                         driver_registry.DISCOVERY_ERRORS)


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


class TestLocalDriver(_ScriptMixin, unittest.TestCase):
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
        with _temp_home():
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


class TestRemoteShellDrivers(_ScriptMixin, unittest.TestCase):
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
                mock_run.return_value = _completed(
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
                mock_run.return_value = _completed(json.dumps(_PROBE_PAYLOAD))
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
                    return _completed("ran\n" if len(calls) == 4 else "")

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
                    return _completed("REAPED\n")

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
        """Concurrent runs against one accelerator must not interleave, and
        losing the lock race is not a failed run, nor may it read as one."""
        for name, make_driver in _shell_drivers().items():
            with self.subTest(name):
                mock_run.side_effect = [
                    _completed(),
                    _completed(returncode=shell.LOCK_BUSY_EXIT),
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
                mock_run.return_value = _completed()
                driver = make_driver()
                job_id = driver.submit(self._script("echo hi\n", suffix=".sh"))
                mock_run.return_value = _completed("COMPLETED 0\n")
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
                mock_run.return_value = _completed()
                driver = make_driver()
                job_id = driver.submit(self._script("echo hi\n", suffix=".sh"))
                mock_run.side_effect = [
                    _completed("RUNNING\n"),
                    _completed(returncode=255, stderr="ssh: connect refused"),
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
        mock_run.return_value = _completed()
        driver = GkePodDriver(namespace="default", pod="tpu-pod")
        job_id = driver.submit(self._script("echo hi\n", suffix=".sh"))
        driver.cancel(job_id)

        mock_run.return_value = _completed("FAILED 1\n")
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


class TestComputeConfig(_ScriptMixin, unittest.TestCase):
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
        args = _cli_args(job_id="job-1")
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
        args = _cli_args(compute_action="run",
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
        home = _patch_home(self)
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


class TestRuntimeCliCompute(_ScriptMixin, unittest.TestCase):
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
        with _temp_home():
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


def _free_port() -> int:
    """A loopback port nothing is bound to, for a forward to bind or miss."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class TestSpawnPortForward(unittest.TestCase):
    """The liveness check behind every backgrounded port forward."""

    def setUp(self):
        _patch_home(self)

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


class TestComputeCliContracts(_ScriptMixin, unittest.TestCase):
    """handle_compute_cli behaviour reachable without the runtime parser."""

    def _args(self, **overrides: Any) -> argparse.Namespace:
        return _cli_args(**{"job_id": "job-1", "out_dir": None, **overrides})

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

        with _temp_home():
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
        with _temp_home() as home:
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
        with _temp_home() as tmp, \
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

        with _temp_home() as home:
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

        with _temp_home():
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
        """A record says PENDING from submit until the job ends unless a
        status read updates it, so `job list` first asks the backend of each
        unfinished job in the run -- once per backend if it cannot answer."""
        replies: dict[str, Any] = {}

        def status(jid: str) -> AsynchronousExecutionResult:
            reply = replies.get(jid,
                                AsynchronousExecutionResult(jid, "RUNNING"))
            if isinstance(reply, Exception):
                raise reply
            return reply

        driver = MagicMock()
        driver.get_status.side_effect = status
        with _temp_home() as tmp:
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
        """`--orphaned` needs no run, `--watch` one; both are tested in
        test_lease."""
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
        """A batch job starts its run's watcher; an ended session's active
        run is no fallback, as a job tagged with it would be reaped."""
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
        """Rejected here, not deep inside kubectl, where the message names
        neither the flag nor the number."""
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
        with _temp_home() as tmp_path:
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
        """A second session's start overwrites the workspace's active-run
        marker; the lease is per harness process, so this session's `job
        list` still refreshes its own jobs, not the other session's."""
        mine, theirs = _RUN, "20260101T000000Z-bbbbbbbbbbbb"
        driver = MagicMock()
        driver.get_status.side_effect = (
            lambda jid: AsynchronousExecutionResult(jid, "RUNNING"))
        with _temp_home() as ws, contextlib.chdir(ws), \
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
    """`compute job wait`: one exit code per outcome, so agents stop writing
    sleep loops that outlive the shell or miss a crash."""

    def wait(self, job: _ScriptedJob, **flags: Any) -> tuple[int, dict]:
        args = _cli_args(
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
        _patch_home(self)

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
        """Session end leaves a serve=True job running, and reports it;
        `job reap --include-serving` is a decision to clear the run."""
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
        """`job wait` shipped in the CLI but not in this section, so agents
        wrote their own sleep loops. Every action the `job`
        parser accepts must be listed, and every flag the section names for
        `job` must exist on it."""
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


class TestPreflight(unittest.TestCase):
    """`compute preflight`: the same report shape for every backend."""

    def _run(self, driver, responses=None, tools="/bin/tool"):
        """Preflight with subprocess answering by a substring of the argv."""
        calls: list[str] = []

        def fake(cmd, **_kwargs):
            calls.append(" ".join(cmd))
            for key, (code, out) in (responses or {}).items():
                if key in calls[-1]:
                    return _completed(out, code, "boom")
            return _completed()

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
                patch("subprocess.run", return_value=_completed("", 1)):
            return self._cluster(base.kubectl_prefix(kubeconfig, context),
                                 kubeconfig)

    def test_a_missing_kubeconfig_is_named(self):
        """kubectl calls a missing file "no context"; the path is the fault."""
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
        """A pinned context: ok if current, a warning if another is, and a
        failure if the kubeconfig does not have it or cannot be read."""
        with tempfile.TemporaryDirectory() as home:
            config = os.path.join(home, "config")
            Path(config).write_text("{}", encoding="utf-8")

            def check(context, contexts="a\nb\n", current="b", code=0):
                out = {"name": contexts, "current-context": current}

                def run(cmd, **_):
                    return _completed(out.get(cmd[-1], ""), code, "boom")

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
        with patch("subprocess.run", return_value=_completed(account, 0)):
            return base.gcloud_account_check(other)

    def test_a_vm_account_without_cloud_platform_scope(self):
        """A VM's service account is limited to the VM's scopes, whatever
        IAM grants it: say so, rather than let every call be refused."""
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
        """$GOOGLE_APPLICATION_CREDENTIALS is a credential kubectl will use:
        list it, but warn rather than recommend."""
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
        """KUBECONFIG names a missing file, the gcloud
        account is a VM's without cloud-platform scope, and an
        authorized_user $GOOGLE_APPLICATION_CREDENTIALS is set. The missing
        file is the blocker; the scope is not."""
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
    """Mock driver with settable preflight and inspect results and a call
    counter for preflight."""

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
        root = _patch_home(self)
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
        args = _cli_args(package="pkg",
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


class TestSetupFieldsAndValidation(unittest.TestCase):
    """SETUP_FIELDS metadata, format validation, and schema/questions consistency."""

    def test_setup_fields_categories_are_disjoint_and_consistent(self):
        pkg_root = pathlib.Path(__file__).resolve().parents[2] / "package"
        schema_path = pkg_root / "schemas" / "environment.json"
        questions_path = pkg_root / "environment" / "questions.json"
        if not schema_path.is_file() or not questions_path.is_file():
            self.skipTest("package schema/questions not available")
        param_props = json.loads(
            schema_path.read_text())["properties"]["profiles"][
                "additionalProperties"]["properties"]["params"]["properties"]
        # A missing `when` means "every backend" in questions.json, which is
        # never what a backend-specific param wants, so each one must say.
        questions = {
            q["path"].removeprefix("compute.params."): q
            for q in json.loads(questions_path.read_text())["questions"]
            if q.get("path", "").startswith("compute.params.")
        }
        required_flags = compute.describe_required_flags()
        samples = {**_GKE_ANSWERS, **_GCE_ANSWERS, "pod": "p"}
        shipped = {
            k: v.SETUP_FIELDS
            for k, v in get_registered_drivers().items()
            if not k.startswith("mock")
            and k not in driver_registry.EXTERNAL_DRIVERS
        }
        # Nothing may exist only in the schema or questions.json: a param the
        # driver reads but no SETUP_FIELDS declares is invisible to setup.
        declared = set().union(*(sf.all_fields() for sf in shipped.values()))
        self.assertLessEqual(set(param_props) | set(questions), declared)
        for backend, sf in shipped.items():
            all_fields = sf.all_fields()
            with self.subTest(backend=backend):
                self.assertEqual(
                    len(sf.required) + len(sf.optional) + len(sf.conditional) +
                    len(sf.on_demand), len(all_fields))
                self.assertLessEqual(set(sf.conditional.values()), all_fields)
                self.assertLessEqual(all_fields, set(param_props))
                # on_demand fields are set when a command first needs them,
                # not at install, so they need no prompt here.
                asked = {
                    f
                    for f, q in questions.items()
                    if backend in q["when"]["backend"]
                }
                self.assertLessEqual(all_fields - set(sf.on_demand), asked)
                self.assertLessEqual(asked, all_fields)
                for field in sf.required:
                    self.assertNotEqual(questions[field].get("default"), "")
                self.assertEqual(
                    required_flags.get(backend, set()),
                    {f"--{f.replace('_', '-')}"
                     for f in sf.required})
                for missing in sf.required:
                    with self.assertRaises(ComputeConfigError) as ctx:
                        drv = get_registered_drivers()[backend].from_params({
                            f:
                            samples[f]
                            for f in sf.required if f != missing
                        })
                        if isinstance(drv, GkeJobDriver):
                            drv.config.require_dispatch()
                    self.assertEqual(ctx.exception.error_code, "MISSING_FIELD")

    def test_validate_field_format(self):
        v = compute.validate_field_format
        for field, good, bad in [
            ("zone", "us-central1-a", "not_a_zone"),
            ("tpu_topology", "2x2x1", "v6e-8"),
            ("tpu_topology", "2x4", "0x4"),
            ("priority", "low", "urgent"),
            ("staging_bucket", "gs://my-bucket/path", "my-bucket"),
            ("trajectory_bucket", "gs://my-bucket", "http://my-bucket"),
        ]:
            self.assertIsNone(v(field, good), field)
            self.assertIsNotNone(v(field, bad), field)

        region_err = v("zone", "us-central1")
        self.assertIn("is a region, not a zone", region_err["reason"])
        self.assertIn("us-central1-a", region_err["choices"])
        # Priority choices come out in a fixed order, highest first.
        self.assertEqual(
            v("priority", "urgent")["choices"],
            ["", "medium", "low", "very-low"],
        )
        # An unfamiliar accelerator label is left to preflight, which only
        # warns, so a newer TPU generation is not rejected at setup.
        self.assertIsNone(v("tpu_accelerator", "tpu-v99-slice"))

        # Images are accepted whether pinned by digest or by tag.
        self.assertIsNone(v("image", "gcr.io/p/img@sha256:deadbeef"))
        self.assertIsNone(v("image", "gcr.io/p/img:latest"))


class TestMultiProfileResolution(unittest.TestCase):
    """Multi-profile environment.json: shapes, precedence, and named errors."""

    PKG = "pkg@1.0.0"

    def setUp(self):
        self.workspace = _patch_home(self) / "ws"
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


class TestComputeSetupApplyAndConfig(unittest.TestCase):
    """`compute setup apply` and `config [show|set|use]`."""

    PKG = "pkg@1.0.0"
    GKE = _GKE_ANSWERS

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = pathlib.Path(self.temp_dir.name)
        self.workspace = root / "ws"
        self.env_file = self.workspace / ".tpu-agent-suite" / "environment.json"
        self.write({"revision": 1, "default": None, "profiles": {}})
        _patch_home(self)

    def write(self, doc):
        self.env_file.parent.mkdir(parents=True, exist_ok=True)
        self.env_file.write_text(json.dumps(doc), encoding="utf-8")

    def saved(self):
        return json.loads(self.env_file.read_text(encoding="utf-8"))

    def apply(self, profile, answers, **kwargs):
        return compute.apply_setup(profile=profile,
                                   answers=answers,
                                   package=self.PKG,
                                   workspace=self.workspace,
                                   **kwargs)

    def config_set(self, *assignments, profile=None):
        return compute.config_set(assignments=list(assignments),
                                  profile=profile,
                                  package=self.PKG,
                                  workspace=self.workspace)

    def test_apply_incomplete_writes_nothing(self):
        vm = {"backend": "gce_tpu_vm", "project": "p"}
        gke = self.GKE
        cases = [
            ({}, "missing", ["backend"]),
            (dict(backend="nope"), "invalid", ["backend"]),
            (vm, "missing", ["zone", "tpu_name"]),
            (dict(_GCE_ANSWERS, zone="us-central1"), "invalid", ["zone"]),
            (dict(gke, queue="q"), "newly_applicable", ["priority"]),
            (dict(gke, tpu_chips="abc"), "invalid", ["tpu_chips"]),
            (dict(gke,
                  staging_service_account="sa@p",
                  staging_service_account_key="/k"), "invalid", ["params"]),
            (dict(backend="local",
                  xprof_interpreter="py"), "invalid", ["xprof_interpreter"]),
        ]
        before = self.env_file.read_text(encoding="utf-8")
        for answers, key, names in cases:
            with self.subTest(answers=answers):
                res = self.apply("p", answers)
                self.assertEqual(res["status"], "INCOMPLETE")
                self.assertEqual(sorted(e["name"] for e in res[key]),
                                 sorted(names))
        with self.assertRaises(compute.ComputeConfigError) as ctx:
            self.apply("my profile", {"backend": "local"})
        self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")
        self.assertEqual(self.env_file.read_text(encoding="utf-8"), before)

    def test_config_set_validates_like_apply(self):
        # A blank field that has a default, like gke_job's image, takes it.
        self.assertEqual(
            self.apply("blank", {
                **self.GKE, "image": ""
            }, dry_run=True)["params"]["image"],
            compute.setup._questions()["image"]["default"])
        self.apply("gke", self.GKE)
        before = self.saved()
        self.assertEqual(
            self.config_set("tpu_topology=")["missing"][0]["name"],
            "tpu_topology")
        self.assertEqual(self.saved(), before)
        res = self.config_set("declared_duration_minutes=30",
                              "xprof_interpreter=/py")
        self.assertEqual((res["status"], res["revision"]), ("ok", 3))
        self.assertEqual(
            self.saved()["profiles"]["gke"]["params"]
            ["declared_duration_minutes"], 30)
        self.assertEqual(self.saved()["xprof_interpreter"], "/py")
        for args, kwargs, code in [(("trajectory_enabled=true", ), {},
                                    "INVALID_FIELD"),
                                   (("queue=x", ), {
                                       "profile": "nope"
                                   }, "UNKNOWN_PROFILE"),
                                   (("no-equals", ), {}, "INVALID_FIELD")]:
            with self.subTest(args=args), self.assertRaises(
                    compute.ComputeConfigError) as ctx:
                self.config_set(*args, **kwargs)
            self.assertEqual(ctx.exception.error_code, code)

    def test_environment_location_is_never_guessed(self):
        self.env_file.write_text("{not json", encoding="utf-8")
        with self.assertRaises(compute.ComputeConfigError) as ctx:
            self.apply("dev", {"backend": "local"})
        self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")
        # Other runtime commands, such as trajectory export, say so too.
        with self.assertRaisesRegex(runtime.TpuAgentSuiteError,
                                    "not valid JSON"):
            paths.get_environment(self.workspace, self.PKG)
        # An ownership ledger without environment.json is repaired ...
        self.env_file.unlink()
        (self.env_file.parent / "ownership.json").write_text(json.dumps(
            {"package": "pkg"}),
                                                             encoding="utf-8")
        self.assertEqual(
            self.apply("dev", {"backend": "local"})["status"], "ok")
        # ... but a package installed nowhere is an error.
        with self.assertRaises(compute.ComputeConfigError) as ctx:
            compute.config_show(package=self.PKG,
                                workspace=pathlib.Path(self.temp_dir.name))
        self.assertEqual(ctx.exception.error_code, "MISSING_FIELD")

    def test_cli(self):
        where = ["--package", self.PKG, "--workspace", str(self.workspace)]
        err = io.StringIO()

        def run(*argv):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(err):
                rc = runtime.main([*argv, *where, "--json"])
            return rc, (json.loads(out.getvalue()) if out.getvalue() else None)

        answers = pathlib.Path(self.temp_dir.name) / "answers.json"
        answers.write_text(json.dumps(self.GKE), encoding="utf-8")
        self.assertEqual(
            run("compute", "setup", "apply", "--profile", "gke", "--answers",
                str(answers))[0], 0)
        self.apply("dev", self.GKE)
        self.assertEqual(run("config", "set", "queue=q")[0], 1)
        rc, shown = run("config", "show")
        self.assertEqual((rc, shown["active_profile"], shown["active_source"]),
                         (0, "gke", "default"))
        run_id = run_state.start_run(package=self.PKG,
                                     workspace=self.workspace)["run_id"]
        self.assertEqual(run("config", "use", "dev")[0], 0)
        self.assertEqual(run("config", "use", "nope")[0], 1)
        # With no --profile, `set` changes the run's profile, not the default.
        rc, res = run("config", "set", "declared_duration_minutes=45")
        self.assertEqual((rc, res["profile"]), (0, "dev"))
        shown = run("config", "show", "--run-id", run_id)[1]
        self.assertEqual((shown["active_profile"], shown["active_source"]),
                         ("dev", "run"))
        self.assertNotIn("declared_duration_minutes",
                         shown["profiles"]["gke"]["params"])
        self.assertEqual(run("config", "use", "dev", "--default")[0], 0)
        self.assertEqual(self.saved()["default"], "dev")
        # Trajectory sharing changes only once the user has agreed, and lands
        # where the export reads it.
        sharing = ("trajectory_enabled=true", "trajectory_bucket=gs://b")
        self.assertEqual(run("config", "set", *sharing)[0], 1)
        self.assertIn("--user-confirmed", err.getvalue())
        self.assertNotIn("trajectory", self.saved())
        run("config", "set", *sharing, "--user-confirmed")
        self.assertEqual(self.saved()["trajectory"]["params"], {
            "enabled": True,
            "bucket": "gs://b"
        })
        # A Harbor org is checked with the hub.
        with patch.object(SETUP, "_harbor_org_exists", return_value=False):
            rc, _ = run("config", "set", "trajectory_sink=harbor",
                        "trajectory_share_org=nope", "--user-confirmed")
        self.assertEqual(rc, 1)
        self.assertIn("no organization named 'nope'", err.getvalue())
        # Turning sharing off takes the destination with it, and the export
        # then refuses rather than uploading to the old bucket.
        _, res = run("config", "set", "trajectory_enabled=false",
                     "--user-confirmed")
        self.assertEqual(self.saved()["trajectory"]["params"],
                         {"enabled": False})
        self.assertIn("trajectory_bucket", res["removed"][0])
        with contextlib.redirect_stderr(err):
            rc = runtime.main(
                ["trajectory", "export", "--harness", "claude-code", *where])
        self.assertEqual(rc, 1)
        self.assertIn("trajectory sharing is off", err.getvalue())

    def test_config_remove(self):
        for name, answers in (("gke", self.GKE), ("dev", {
                "backend": "local"
        }), ("aux", {
                "backend": "local"
        })):
            self.apply(name, answers)

        def remove(profile, **kwargs):
            return compute.config_remove(profile=profile,
                                         package=self.PKG,
                                         workspace=self.workspace,
                                         **kwargs)

        before = self.saved()
        for profile, confirmed, code in (("  ", True, "MISSING_FIELD"),
                                         ("aux", False,
                                          "CONFIRMATION_REQUIRED"),
                                         ("nope", True, "UNKNOWN_PROFILE")):
            with self.subTest(code), self.assertRaises(
                    compute.ComputeConfigError) as ctx:
                remove(profile, user_confirmed=confirmed)
            self.assertEqual(ctx.exception.error_code, code)
        self.assertEqual(self.saved(), before)

        # Removing a non-default profile preserves default and leaves other
        # pinned runs alone.
        run_dev = run_state.start_run(package=self.PKG,
                                      workspace=self.workspace)["run_id"]
        compute.config_use(profile="dev",
                           package=self.PKG,
                           workspace=self.workspace,
                           run_id=run_dev)
        res = remove("aux", user_confirmed=True, run_id=run_dev)
        self.assertEqual((res["default"], res.get("unpinned_run")),
                         ("gke", None))
        run_file = paths.get_run_root(run_dev) / "run.json"
        self.assertEqual(
            json.loads(run_file.read_text())["compute_profile"], "dev")
        # Removing the default profile promotes a remaining profile ("dev").
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = runtime.main([
                "config", "remove", "gke", "--user-confirmed", "--package",
                self.PKG, "--workspace",
                str(self.workspace), "--json"
            ])
        self.assertEqual((rc, json.loads(out.getvalue())["default"]),
                         (0, "dev"))
        self.assertEqual(list(self.saved()["profiles"]), ["dev"])
        # Removing the last profile sets default to None, passes schema
        # validation, and unpins a run pinned to that profile.
        res = remove("dev", user_confirmed=True, run_id=run_dev)
        self.assertEqual((res["default"], res["unpinned_run"]),
                         (None, run_dev))
        self.assertEqual(
            utils.validate_instance(self.saved(), compute.setup._schema()), [])
        self.assertNotIn("compute_profile", json.loads(run_file.read_text()))

    def test_verify_records_a_result_per_profile_hash(self):
        """A result counts until the profile's settings change."""
        self.apply("vm", _GCE_ANSWERS)
        driver = MagicMock(quick_run=True, scheduled_run=False)
        driver.preflight.side_effect = lambda: base.PreflightReport(
            "gce_tpu_vm", "vm", True, [])

        def verify(stdout, *flags):
            if stdout is not None:
                driver.run.return_value = base.SynchronousExecutionResult(
                    0 if "ok" in stdout else 1, stdout, "")
            out = io.StringIO()
            with patch.object(compute.GceTpuVmDriver,
                              "from_params",
                              return_value=driver), \
                    contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(io.StringIO()):
                rc = runtime.main([
                    "compute", "setup", "verify", "--package", self.PKG,
                    "--workspace",
                    str(self.workspace), "--json", *flags
                ])
            res = json.loads(out.getvalue())
            return rc, res["status"], res.get("smoke_test")

        def state():
            return compute.config_show(package=self.PKG,
                                       workspace=self.workspace)["setup_state"]

        self.assertEqual(state(), {"vm": "UNVERIFIED"})
        self.assertEqual(verify("[FAIL] torch_tpu")[:2], (1, "FAILED"))
        self.assertEqual(state(), {"vm": "FAILED"})
        # The failing checks are kept, for session start to show.
        results = self.env_file.parent / "verification.json"
        (saved, ) = json.loads(results.read_text()).values()
        self.assertEqual([c["fields"] for c in saved["failed_checks"]],
                         [["python_interpreter"]])
        self.assertEqual(verify("Status: ok", "--profile", "vm"),
                         (0, "ok", False))
        self.assertEqual(state(), {"vm": "READY"})
        self.assertEqual(len(json.loads(results.read_text())), 1)
        # A result from another package version doesn't count.
        with patch.object(SETUP.utils, "TPU_AGENT_SUITE_VERSION", "0.0.1"):
            self.assertEqual(state(), {"vm": "UNVERIFIED"})
        # A smoke test still pending at its timeout passes, but didn't run.
        driver.configure_mock(quick_run=False, scheduled_run=True)
        driver.run.return_value = base.SynchronousExecutionResult(
            1, "", "", {
                "timed_out_after_sec": 900,
                "job_status": "PENDING"
            })
        self.assertEqual(verify(None, "--smoke-test"), (0, "ok", False))
        self.config_set("zone=us-east5-b")
        self.assertEqual(state(), {"vm": "UNVERIFIED"})
        # An incomplete profile is reported, not checked.
        self.write(profile_env("gce_tpu_vm"))
        self.assertEqual(verify("")[1], "INCOMPLETE")
        self.assertEqual(state(), {"default": "NOT_CONFIGURED"})

    def test_plan_lists_questions_with_suggestions(self):
        outputs = {
            "kubectl get nodes": "tpu7x 2x2x1\ntpu7x 2x2x1\ntpu-v6e-slice 2x4",
            "kubectl get localqueues": "lq other",
            "gcloud config get-value": "other-proj",
            "gcloud compute tpus": "vm1 us-west4-a",
            "kubectl config current-context": "gke_my-proj_us-central1_c1",
        }
        ran = []

        def run(cmd):
            ran.append(" ".join(cmd))
            return outputs.get(
                " ".join(cmd[:3]),
                ""), ("kubectl is not installed" if cmd[2] == "view" else None)

        def plan(*flags, before=()):
            out = io.StringIO()
            with patch.object(compute.probes, "_run", side_effect=run), \
                    patch.object(compute.probes.probe, "list_devices",
                                 return_value=([], [])), \
                    contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(io.StringIO()):
                rc = runtime.main([
                    "compute", *before, "setup", "plan", "--package", self.PKG,
                    "--workspace",
                    str(self.workspace), "--json", *flags
                ])
            res = json.loads(out.getvalue() or "{}")
            return rc, res, {q["name"]: q for q in res.get("questions", [])}

        # No backend: only the backend question, and the profiles to pick.
        self.write({"default": None, "profiles": {"vm": {"backend": "local"}}})
        _, res, qs = plan()
        self.assertEqual(
            (res["profiles"], list(qs), qs["backend"]["suggested"]),
            ({
                "vm": "local"
            }, ["backend"], "gke_job"))
        # A backend's setup fields, suggested by probes or defaults. Flags
        # before the subcommand count too.
        _, res, qs = plan(before=("--profile", "gke", "--backend", "gke_job"))
        self.assertEqual((res["profile"], res["backend"]), ("gke", "gke_job"))
        nodes = "kubectl get nodes"
        self.assertEqual([(f, q.get("suggested_from")) for f, q in qs.items()],
                         [("image", "default"), ("tpu_accelerator", nodes),
                          ("tpu_topology", nodes),
                          ("kube_context", "kubectl config current-context"),
                          ("namespace", None),
                          ("queue", "kubectl get localqueues"),
                          ("hf_cache_pvc", None), ("priority", None)])
        self.assertEqual([
            qs[f]["suggested"] for f in ("tpu_accelerator", "tpu_topology",
                                         "kube_context", "queue")
        ], ["tpu7x", "2x2x1", "gke_my-proj_us-central1_c1", "lq"])
        self.assertEqual(qs["priority"]["asked_if"], "queue")
        # A missing tool is reported, deduplicated.
        self.assertEqual(res["probe_warnings"], ["kubectl is not installed"])
        # With a backend chosen, a cluster with no TPU nodes is reported.
        saved_nodes = outputs.pop(nodes)
        _, res, _ = plan("--profile", "g2", "--backend", "gke_job")
        self.assertIn("no TPU nodes", " ".join(res["probe_warnings"]))
        outputs[nodes] = saved_nodes
        # Stored values are suggested as they are, except with --fields,
        # which asks again for values verify rejected.
        self.apply("gke", {**self.GKE, "kubeconfig": "/k"})
        _, res, qs = plan("--profile", "gke")
        self.assertEqual(qs["tpu_accelerator"]["suggested_from"], "profile")
        _, res, qs = plan("--profile", "gke", "--fields",
                          "tpu_accelerator,queue")
        self.assertEqual(list(qs), ["tpu_accelerator", "queue"])
        self.assertEqual((qs["tpu_accelerator"]["current"],
                          qs["tpu_accelerator"]["suggested_from"]),
                         ("tpu7x", "kubectl get nodes"))
        # A saved kubeconfig is used by the kubectl probes.
        self.assertIn(
            "kubectl get localqueues -o jsonpath="
            "{.items[*].metadata.name} --kubeconfig /k", ran)
        # Unknown, consent-only (trajectory) or backend-less --fields fail.
        for flags in (("gke", "--fields", "zone"), ("gke", "--fields",
                                                    "trajectory_enabled"),
                      ("new", "--fields", "zone"), ("my vm", "--backend",
                                                    "gke_job")):
            self.assertEqual(plan("--profile", *flags)[0], 1, flags)
        # gcloud prints "(unset)" when it has no project. A TPU VM is only
        # suggested when it's clear which one is meant; otherwise the VMs are
        # offered as choices, each with its zone.
        outputs["gcloud config get-value"] = "(unset)"
        _, _, qs = plan("--profile", "tv", "--backend", "gce_tpu_vm")
        self.assertEqual(qs["tpu_name"]["suggested"], "vm1")
        self.assertNotIn("suggested", qs["project"])
        outputs["gcloud compute tpus"] = "vm1 us-west4-a\nme-vm us-east5-a"
        with patch.dict(os.environ, {"USER": "nobody"}):
            _, res, qs = plan("--profile", "tv", "--backend", "gce_tpu_vm")
        self.assertEqual([c["tpu_name"] for c in qs["tpu_name"]["choices"]],
                         ["vm1", "me-vm"])
        self.assertNotIn("suggested", qs["zone"])
        self.assertIn("free text", res["note"])
        # After a backend change, the values apply would keep are current.
        self.apply("vm", {**_GCE_ANSWERS, "python_interpreter": "/py"})
        _, _, qs = plan("--profile", "vm", "--backend", "gke_pod")
        self.assertEqual(
            [qs[f].get("current") for f in ("python_interpreter", "pod")],
            ["/py", None])

    def test_tpu_vm_choices(self):
        """$USER's VMs come first, the list is capped, and a zone is
        suggested only when every VM, listed or not, shares it."""
        many = "\n".join(f"vm{i} z{i % 2}" for i in range(20))
        with patch.dict(os.environ, {"USER": "ann"}):
            found = compute.probes._tpu_vm(f"{many}\nann-a z9\nann-b z9")
        self.assertEqual([c["tpu_name"] for c in found["tpu_name"][:3]],
                         ["ann-a", "ann-b", "vm0"])
        self.assertEqual(len(found["tpu_name"]), compute.probes.MAX_CHOICES)
        self.assertNotIn("zone", found)
        self.assertEqual(compute.probes._tpu_vm(""), {})
        self.assertEqual(compute.probes._tpu_vm("a z\nb z")["zone"], "z")
        cut = "\n".join(f"vm{i} z" for i in range(20)) + "\nlast elsewhere"
        self.assertNotIn("zone", compute.probes._tpu_vm(cut))
        # The same name in two zones stays two distinct choices.
        self.assertEqual(
            len(compute.probes._tpu_vm("dup za\ndup zb")["tpu_name"]), 2)
        # Several VMs still point to gce_tpu_vm as the backend.
        self.assertEqual(
            compute.probes._backend({}, "a z\nb y", ["gce_tpu_vm"])[0],
            "gce_tpu_vm")

    def test_a_probe_reports_a_missing_tool_or_a_timeout(self):
        self.assertEqual(compute.probes._run(["no-such-tool-xyz"]),
                         ("", "no-such-tool-xyz is not installed"))
        with patch.object(compute.probes, "PROBE_TIMEOUT_SEC", 0.2):
            self.assertEqual(compute.probes._run(["sleep", "5"]),
                             ("", "`sleep 5` did not answer within 0.2s"))
        # A command that runs and fails just has nothing to suggest.
        self.assertEqual(compute.probes._run(["false"]), ("", None))


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


# The module `compute setup` and `config` run in.
SETUP = sys.modules[compute.config_show.__module__]


class ShareOrgCheckTest(unittest.TestCase):
    """A Harbor org the hub does not have is refused when saved, not at
    upload."""

    def test_an_org_that_cannot_be_checked_is_kept_with_a_warning(
            self) -> None:
        """No harbor, no login or no network is not evidence the org is wrong."""
        answers = {
            "trajectory": {
                "params": {
                    "sink": "harbor",
                    "share_org": "t"
                }
            }
        }
        for error, says in ((ImportError("harbor"), "harbor is not installed"),
                            (TimeoutError(), "could not check the Harbor")):
            with patch.object(SETUP, "_harbor_org_exists", side_effect=error), \
                    contextlib.redirect_stderr(io.StringIO()) as stderr:
                SETUP.check_share_org(answers, None)
            self.assertIn(says, stderr.getvalue())


class TestTargetVersions(unittest.TestCase):
    """`compute versions`: the releases on the target, never this machine's."""

    PKG = "pkg@1.0.0"
    GKE = _GKE_ANSWERS

    def setUp(self):
        super().setUp()
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.workspace = self.tmp / "ws"
        self.env_file = (self.workspace / ".tpu-agent-suite" /
                         "environment.json")
        self.env_file.parent.mkdir(parents=True)
        self.env_file.write_text('{"profiles": {}}')
        _patch_home(self)

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
