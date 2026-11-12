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
"""Compiles every suite under eval/suites/.

The compiler's unit tests use synthetic fixtures; this is the one test that
reads the repository's suites.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest
from unittest import mock

from eval.suite_to_harbor import gcs_utils, suite_to_harbor
from eval.utils import find_repo_root

Path = pathlib.Path

_SUITES_DIR = find_repo_root() / "eval" / "suites"


class RepoSuitesCompileTest(unittest.TestCase):

    def test_every_repo_suite_compiles(self) -> None:
        suite_files = sorted(_SUITES_DIR.glob("**/*.yaml"))
        self.assertTrue(suite_files, f"no suites found under {_SUITES_DIR}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            def _placeholder_gcs_asset(gcs_uri: str,
                                       cache_root: Path | None = None) -> Path:
                # Keeps the test offline: a gs:// asset resolves to an empty
                # local file instead of a download.
                del cache_root
                placeholder = (tmp_path / "gcs" /
                               gcs_uri.removeprefix("gs://").strip("/"))
                placeholder.parent.mkdir(parents=True, exist_ok=True)
                placeholder.touch()
                return placeholder

            with mock.patch.object(gcs_utils,
                                   "fetch_gcs_asset",
                                   side_effect=_placeholder_gcs_asset):
                for suite_file in suite_files:
                    rel = suite_file.relative_to(_SUITES_DIR).with_suffix("")
                    with self.subTest(suite=str(rel)):
                        self.assertTrue(
                            suite_to_harbor.compile_suite(
                                suite_file, tmp_path / "out" / rel))


if __name__ == "__main__":
    unittest.main()
