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
"""Tests for gcs_utils."""

from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from eval.suite_to_harbor import gcs_utils

Path = pathlib.Path


class GcsUtilsTest(unittest.TestCase):

    def test_get_gcs_cache_root(self) -> None:
        self.assertEqual(
            gcs_utils.get_gcs_cache_root(),
            gcs_utils.DEFAULT_GCS_CACHE_DIR,
        )
        with mock.patch.dict(os.environ,
                             {"EVAL_GCS_CACHE_DIR": "/tmp/custom_eval_cache"}):
            self.assertEqual(
                gcs_utils.get_gcs_cache_root(),
                Path("/tmp/custom_eval_cache").resolve(),
            )

    def test_fetch_gcs_asset_invalid_uri(self) -> None:
        with self.assertRaises(ValueError):
            gcs_utils.fetch_gcs_asset("https://storage.googleapis.com/test")

    @mock.patch("shutil.which", return_value=None)
    def test_fetch_gcs_asset_missing_gcloud(
            self, mock_which: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError) as ctx:
                gcs_utils.fetch_gcs_asset(
                    "gs://bucket/asset.pb",
                    cache_root=Path(tmp_dir),
                )
            self.assertIn("gcloud", str(ctx.exception))

    def test_fetch_gcs_asset_cache_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir)
            cached_file = cache_root / "my-bucket" / "traces" / "v1" / "trace.pb"
            cached_file.parent.mkdir(parents=True, exist_ok=True)
            cached_file.write_bytes(b"existing_trace_data")

            with mock.patch("subprocess.run") as mock_run:
                result = gcs_utils.fetch_gcs_asset(
                    "gs://my-bucket/traces/v1/trace.pb",
                    cache_root=cache_root,
                )
                self.assertEqual(result, cached_file)
                mock_run.assert_not_called()

    @mock.patch("shutil.which", return_value="/usr/bin/gcloud")
    def test_fetch_gcs_asset_cache_miss_success(
            self, mock_which: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir)

            def fake_download(cmd, **kwargs):
                dest = Path(cmd[-1])
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "trace.pb").write_bytes(b"downloaded_bytes")
                return mock.MagicMock(returncode=0)

            with mock.patch("subprocess.run",
                            side_effect=fake_download) as mock_run:
                result = gcs_utils.fetch_gcs_asset(
                    "gs://my-bucket/traces/v1/trace.pb",
                    cache_root=cache_root,
                )
                expected_dest = cache_root / "my-bucket" / "traces" / "v1" / "trace.pb"
                self.assertEqual(result, expected_dest)
                self.assertTrue(expected_dest.is_file())
                self.assertEqual(expected_dest.read_bytes(),
                                 b"downloaded_bytes")
                mock_run.assert_called_once()

    @mock.patch("shutil.which", return_value="/usr/bin/gcloud")
    def test_fetch_gcs_asset_download_failure(
            self, mock_which: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir)

            error = subprocess.CalledProcessError(
                returncode=1,
                cmd=["gcloud"],
                stderr="AccessDenied: Bucket not found",
            )
            with mock.patch("subprocess.run", side_effect=error):
                with self.assertRaises(RuntimeError) as ctx:
                    gcs_utils.fetch_gcs_asset(
                        "gs://bad-bucket/trace.pb",
                        cache_root=cache_root,
                    )
                self.assertIn("AccessDenied", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
