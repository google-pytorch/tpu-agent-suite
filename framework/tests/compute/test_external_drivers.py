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
"""Unit tests for loading compute drivers from outside the package
(`framework/compute/drivers/external.py`)."""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

# Imports below follow this sys.path setup, so the file runs directly.
# ruff: noqa: E402
_repo_root = pathlib.Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from framework import compute
from framework.compute import ComputeConfigError, LocalDriver
from framework.compute import config as compute_config
from framework.compute import get_driver, resolve_driver
from framework.compute.drivers import external
from framework.compute.drivers import registry as driver_registry
from framework.compute.drivers.registry import DRIVER_REGISTRY, list_drivers
from framework.tests.compute.helpers import set_up_module, tear_down_module

Path = pathlib.Path


def setUpModule():  # noqa: N802 - unittest's spelling
    set_up_module()


def tearDownModule():  # noqa: N802 - unittest's spelling
    tear_down_module()


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


if __name__ == "__main__":
    unittest.main()
