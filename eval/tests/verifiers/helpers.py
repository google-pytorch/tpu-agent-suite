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
"""Shared fixtures for the verifier unit tests.

A verifier finds its inputs under $LOGS_DIR and $TESTS_DIR. `harness` points
both at a temporary directory, so a check under test reads that in place of
the host's /logs and /tests.
"""

import contextlib
import dataclasses
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock


def _write(path: Path, content: Any) -> Path:
    """Writes text as it is, and anything else as JSON."""
    text = content if isinstance(content, str) else json.dumps(content)
    path.write_text(text, encoding="utf-8")
    return path


@dataclasses.dataclass(frozen=True)
class Harness:
    """The directories a verifier reads, under one temporary root."""

    logs: Path
    tests: Path
    workspace: Path

    def write_trajectory(self, trajectory: Any) -> Path:
        return _write(self.logs / "agent" / "trajectory.json", trajectory)

    def write_params(self, params: Any) -> Path:
        return _write(self.tests / "verifier_params.json", params)


@contextlib.contextmanager
def harness() -> Iterator[Harness]:
    """A temporary harness, with LOGS_DIR and TESTS_DIR pointing at it.

    Its path holds glob metacharacters, which a verifier has to take as plain
    text. HOME is an empty directory under it, so a verifier that looks in the
    agent's home finds nothing of the host's.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "run[1]"
        home = root / "home"
        dirs = Harness(root / "logs", root / "tests", root / "workspace")
        for directory in (dirs.logs / "agent", dirs.tests, dirs.workspace,
                          home):
            directory.mkdir(parents=True)
        with mock.patch.dict(
                os.environ, {
                    "LOGS_DIR": str(dirs.logs),
                    "TESTS_DIR": str(dirs.tests),
                    "HOME": str(home),
                }):
            yield dirs
