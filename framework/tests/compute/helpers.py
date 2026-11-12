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
"""Shared fixtures for the compute runtime unit tests in this package.

Test Coverage Overview:
- test_drivers: plugin registration, driver CLI flags, driver auto-discovery,
  host probing, the local job lifecycle, and the lifecycle GceTpuVmDriver
  and GkePodDriver share, plus each transport's own commands.
- test_config / test_setup: driver resolution from the environment file,
  environment.json schemas, CLI override precedence, and `compute setup`.
- test_cli / test_preflight: `runtime.py compute` behaviour, output
  envelopes, preflight checks, and the generated agent contract.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

from framework import compute
from framework.compute import (AsynchronousExecutionResult, ComputeDriver,
                               HardwareProfile, SynchronousExecutionResult,
                               register_driver)

Path = pathlib.Path

# The compute CLI falls back to the session leased to the harness process
# above it. Run from inside an agent session, that would be the developer's
# own run; tests that exercise the fallback patch it back in themselves.
NO_SESSION = patch.object(compute.cli.lease, "session_run", return_value=None)

# Nor may a test reach the developer's real state some other way. Jobs and runs
# are recorded in the central store, and a command given no workspace finds one
# by climbing from the current directory, which in a checkout reaches its
# active run. Both point into a temp directory for the whole module; a test that
# needs its own store still patches `TPU_AGENT_SUITE_HOME` over this one.
ISOLATION = contextlib.ExitStack()


def set_up_module() -> None:
    """Isolate a test module from the developer's session and state."""
    NO_SESSION.start()
    scratch = Path(tempfile.mkdtemp())
    ISOLATION.callback(shutil.rmtree, scratch, ignore_errors=True)
    ISOLATION.enter_context(
        patch.dict(os.environ,
                   {"TPU_AGENT_SUITE_HOME": str(scratch / "store")}))
    # chdir rather than patching Path.cwd, so that subprocesses the tests start
    # resolve their workspace inside the temp directory too.
    ISOLATION.callback(os.chdir, os.getcwd())
    os.chdir(scratch)


def tear_down_module() -> None:
    """Undo `set_up_module`."""
    ISOLATION.close()
    NO_SESSION.stop()


def profile_env(backend: str, **params: Any) -> dict[str, Any]:
    """An environment.json holding one profile, `default`, the default."""
    profile = {"backend": backend, **({"params": params} if params else {})}
    return {"default": "default", "profiles": {"default": profile}}


def cli_args(**overrides: Any) -> argparse.Namespace:
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


def completed(stdout: str = "",
              returncode: int = 0,
              stderr: str = "") -> subprocess.CompletedProcess:
    """Build a CompletedProcess stand-in for a mocked subprocess call."""
    return subprocess.CompletedProcess(args=[],
                                       returncode=returncode,
                                       stdout=stdout,
                                       stderr=stderr)


def patch_home(test: unittest.TestCase) -> Path:
    """Point the central store at a fresh temp dir for the rest of `test`."""
    home = Path(tempfile.mkdtemp())
    test.addCleanup(shutil.rmtree, home, ignore_errors=True)
    patcher = patch.dict(os.environ, {"TPU_AGENT_SUITE_HOME": str(home)})
    patcher.start()
    test.addCleanup(patcher.stop)
    return home


@contextlib.contextmanager
def temp_home():
    """Point the central store at a fresh temp dir inside the block."""
    with tempfile.TemporaryDirectory() as home, patch.dict(
            os.environ, {"TPU_AGENT_SUITE_HOME": home}):
        yield Path(home)


# A complete gke_job profile, as the answers `compute setup apply` takes.
GKE_ANSWERS = {
    "backend": "gke_job",
    "image": "us-docker.pkg.dev/p/r/img:v1",
    "tpu_accelerator": "tpu7x",
    "tpu_topology": "2x2x1",
}


class ScriptMixin:
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
