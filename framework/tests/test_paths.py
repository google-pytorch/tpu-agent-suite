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
"""Unit tests for the workspace's active-run lookup."""

import pathlib
import shutil
import tempfile
import unittest

from framework import paths


class TestActiveRun(unittest.TestCase):
    """The workspace's active run: written once, found from any subdirectory."""

    def setUp(self):
        self.workspace = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.workspace, ignore_errors=True)

    def test_corrupted_file_reads_as_no_run(self):
        marker = self.workspace / ".tpu-agent-suite" / "current_run_id"
        marker.parent.mkdir(parents=True)
        marker.write_text("not-a-valid-run-id", encoding="utf-8")
        self.assertIsNone(paths.get_active_run_id(self.workspace))


if __name__ == "__main__":
    unittest.main()
