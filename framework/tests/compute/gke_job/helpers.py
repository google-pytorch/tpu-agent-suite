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
"""Shared fakes and fixtures for the gke_job driver tests.

kubectl is replaced by a scripted fake rather than mocked per call, so the
assertions are about the commands the driver issues and the manifest it sends
-- the two things a cluster actually sees. Nothing here talks to a cluster.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
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
from framework.compute.drivers import gcs_staging, gke_job

Path = pathlib.Path


def set_up_module() -> None:
    """Stop tests from minting real tokens with the developer's ADC.

    Any submit with a staging bucket tries to mint one. By default, act like
    CI (no credentials). Token tests patch this themselves.
    """
    patcher = patch.object(gcs_staging,
                           "mint_bucket_token",
                           side_effect=ComputeConfigError(
                               "no credentials in tests",
                               error_code="NO_CREDENTIALS"))
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class FakeKubectl:
    """Scripted stand-in for kubectl that records what it was asked to do.

    Responses are keyed by a substring of the joined argv rather than the
    whole command, so a test states the one verb it cares about and stays
    readable when an unrelated flag is added to the driver.
    """

    def __init__(self,
                 responses: dict[str, tuple[int, str, str]] | None = None):
        self.responses = responses or {}
        self.calls: list[list[str]] = []
        self.stdins: list[str | None] = []

    def __call__(self, cmd: list[str], **kwargs: Any):
        self.calls.append(cmd)
        self.stdins.append(kwargs.get("input"))
        joined = " ".join(cmd)
        for key, (code, out, err) in self.responses.items():
            if key in joined:
                return subprocess.CompletedProcess(cmd, code, out, err)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def find(self, needle: str) -> list[str] | None:
        """The first recorded call whose argv contains `needle`."""
        for cmd in self.calls:
            if needle in " ".join(cmd):
                return cmd
        return None

    def applied_manifest(self) -> dict[str, Any]:
        """The Job sent to `kubectl apply -f -`."""
        for cmd, stdin in zip(self.calls, self.stdins):
            if "apply" in cmd and stdin and json.loads(stdin)["kind"] == "Job":
                return json.loads(stdin)
        raise AssertionError("no manifest was applied")

    def applied_pod_spec(self) -> dict[str, Any]:
        """The pod spec of the Job sent to `kubectl apply -f -`."""
        return self.applied_manifest()["spec"]["template"]["spec"]

    def applied_container(self) -> dict[str, Any]:
        """The first container of the applied Job's pod."""
        return self.applied_pod_spec()["containers"][0]


def make_driver(**overrides: Any) -> gke_job.GkeJobDriver:
    params: dict[str, Any] = {
        "namespace": "default",
        "image": "registry/torch-tpu@sha256:abc",
        "tpu_accelerator": "tpu7x",
        "tpu_topology": "2x2x1",
    }
    params.update(overrides)
    return gke_job.GkeJobDriver.from_params(params)


def hello_script(test: unittest.TestCase) -> Path:
    """A `hello_tpu.py` in a temporary directory removed when `test` ends."""
    tmp = Path(test.enterContext(tempfile.TemporaryDirectory()))
    script = tmp / "hello_tpu.py"
    script.write_text("print('hi')\n", encoding="utf-8")
    return script


def job_json(**status: Any) -> str:
    spec = {"suspend": status.pop("suspend", False)}
    return json.dumps({"spec": spec, "status": status})


def failed_pod(*,
               exit_code: int | None = None,
               container_reason: str | None = None,
               reason: str | None = None,
               message: str | None = None,
               conditions: list[dict[str, Any]] | None = None,
               phase: str = "Failed",
               node: str | None = "gke-tpu-node-1",
               name: str = "run-1-abcde") -> dict[str, Any]:
    """A Job's pod as `kubectl get pods -o json` lists it.

    The container is terminated only when `exit_code` or
    `container_reason` is given; `reason`, `message` and `conditions` are
    the pod's own status. Defaults to a failed pod on a TPU node; pass
    `phase="Running"` for a live one.
    """
    status: dict[str, Any] = {"phase": phase}
    if exit_code is not None or container_reason is not None:
        terminated = {"exitCode": exit_code, "reason": container_reason}
        status["containerStatuses"] = [{
            "state": {
                "terminated": {
                    k: v
                    for k, v in terminated.items() if v is not None
                }
            }
        }]
    for key, value in (("reason", reason), ("message", message), ("conditions",
                                                                  conditions)):
        if value is not None:
            status[key] = value
    pod: dict[str, Any] = {"metadata": {"name": name}, "status": status}
    if node:
        pod["spec"] = {"nodeName": node}
    return pod


@contextlib.contextmanager
def job_store():
    """A throwaway job-record store, so tests never touch ~/.tpu-agent-suite."""
    with tempfile.TemporaryDirectory() as home, patch.dict(
            os.environ, {"TPU_AGENT_SUITE_HOME": home}):
        yield


class GkeJobLifecycleBase(unittest.TestCase):
    """Job-store isolation and shared fakes for the lifecycle tests."""

    def setUp(self):
        self.enterContext(job_store())

    def status(self, fake: FakeKubectl):
        driver = make_driver()
        with patch.object(subprocess, "run", fake):
            return driver.get_status("run-1")

    GONE = (1, "", 'Error from server (NotFound): jobs.batch "run-1" not '
            "found")


@contextlib.contextmanager
def fake_google_auth(creds: Any):
    """Stand in for `google.auth` so the ambient-credential path is testable.

    It works with or without the SDK installed. The code under test
    imports it lazily; `import google.auth` consults `sys.modules` for the
    submodule but then reads it as an attribute of the `google` package,
    so both are patched.
    """
    auth = types.ModuleType("google.auth")
    auth.default = MagicMock(return_value=(creds, "proj"))
    transport = types.ModuleType("google.auth.transport")
    requests = types.ModuleType("google.auth.transport.requests")
    requests.Request = MagicMock()
    auth.transport = transport
    transport.requests = requests
    google_pkg = sys.modules.get("google") or types.ModuleType("google")
    with patch.dict(
            sys.modules, {
                "google": google_pkg,
                "google.auth": auth,
                "google.auth.transport": transport,
                "google.auth.transport.requests": requests,
            }), patch.object(google_pkg, "auth", auth, create=True):
        yield


class KeyCreds:
    """Credentials with their own signer: a key or impersonation."""
    signer_email = "sa@proj.iam.gserviceaccount.com"

    def sign_bytes(self, data):
        del data
        return b"sig"
