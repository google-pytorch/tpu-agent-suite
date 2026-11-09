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
"""Shared fixtures for the hook and runtime unit tests."""

import contextlib
import io
import json
import os
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

from framework import runtime
from framework.hooks import dispatch

Path = pathlib.Path

_no_lease_sweep = mock.patch("framework.compute.lease.spawn_sweep")


def set_up_module():
    # Every builtin start-run starts a detached sweep of the real job store.
    _no_lease_sweep.start()


def tear_down_module():
    _no_lease_sweep.stop()


def isolated_workspace(test: unittest.TestCase,
                       home_name: str) -> tuple[Path, Path]:
    """A temp dir holding a workspace, with the store pointed beside it.

    Returns the temp dir and its `workspace`; `TPU_AGENT_SUITE_HOME` names
    `home_name` inside the temp dir, which is not created. Both the directory
    and the variable are restored when `test` ends.
    """
    temp_dir = Path(tempfile.mkdtemp())
    test.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
    workspace = temp_dir / "workspace"
    workspace.mkdir()
    home = mock.patch.dict(os.environ,
                           {"TPU_AGENT_SUITE_HOME": str(temp_dir / home_name)})
    home.start()
    test.addCleanup(home.stop)
    return temp_dir, workspace


class RuntimeTestCase(unittest.TestCase):
    """Base for the topic classes split out of the former `TestRuntime`.

    Unit tests for runtime namespace allocation, hooks, and validations: a
    workspace, a central store and an installed `test-pkg@1.0.0`.
    """

    def setUp(self):
        super().setUp()
        self.temp_dir, self.workspace_dir = isolated_workspace(
            self, "central_tpu_agent_suite")
        self.central_home = self.temp_dir / "central_tpu_agent_suite"
        self.central_home.mkdir()

        # Set up installed package structure: the payload under its version,
        # and a user-scope environment.json at the central store's root.
        self.pkg_root = self.central_home / "packages" / "test-pkg@1.0.0"
        self.pkg_root.mkdir(parents=True)
        (self.central_home / "environment.json").write_text(json.dumps({
            "revision":
            1,
            "target":
            "local"
        }),
                                                            encoding="utf-8")
        (self.pkg_root / "schemas").mkdir()
        (self.pkg_root / "schemas" / "out.json").write_text(
            json.dumps({
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string"
                    }
                },
                "required": ["status"],
            }),
            encoding="utf-8",
        )

    def _write_script(self, name: str, source: str) -> Path:
        """Writes `source` as the package's `scripts/<name>`."""
        script = self.pkg_root / "scripts" / name
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(source, encoding="utf-8")
        return script

    def _dispatch(self, **kwargs) -> tuple[dict, int]:
        """`dispatch_hook` for the test package, in the test workspace."""
        kwargs.setdefault("package", "test-pkg@1.0.0")
        kwargs.setdefault("workspace", self.workspace_dir)
        return dispatch.dispatch_hook(**kwargs)

    def _main_capturing(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = runtime.main(argv)
        return code, out.getvalue(), err.getvalue()
