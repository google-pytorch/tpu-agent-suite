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
"""Unit tests for `framework.hc.paths`."""

import os
import pathlib
import tempfile
import unittest
from unittest import mock

from framework import paths as base_paths
from framework.hc import constants, paths


class TestCampaignPaths(unittest.TestCase):
    """The protected campaign store's layout and active-campaign pointer."""

    def test_layout_active_campaign_and_id_validation(self):
        run_id = "20261110T070000Z-abcdef012345"
        job_id = "job-20261110070000-a1b2c3"
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
                os.environ,
            {"TPU_AGENT_SUITE_HOME": str(pathlib.Path(tmp).resolve())}):
            root = pathlib.Path(tmp).resolve() / "campaigns"
            self.assertEqual(
                (paths.get_campaign_db_path(), paths.get_campaign_git_dir("7"),
                 paths.get_campaign_scripts_dir(7),
                 paths.get_campaign_sweeps_dir(7),
                 paths.get_campaign_events_log(7),
                 paths.get_unscoped_events_log(),
                 paths.get_sealed_job_artifacts_dir(job_id)),
                (root / "campaign.db", root / "7/git", root / "7/scripts",
                 root / "7/sweeps", root / "7/logs/events.jsonl",
                 root / "logs/events_unscoped.jsonl",
                 root / f"sealed_jobs/{job_id}/sealed"),
            )

            self.assertIsNone(paths.get_active_campaign_id(run_id))
            paths.record_active_campaign(run_id, 42)
            self.assertEqual(paths.get_active_campaign_id(run_id), 42)
            marker = paths.get_active_campaign_marker(run_id)
            self.assertEqual(marker,
                             root / f"runs/{run_id}/active_campaign_id")
            marker.write_text("corrupted\n", encoding="utf-8")
            self.assertIsNone(paths.get_active_campaign_id(run_id))
            marker.unlink()
            legacy = base_paths.get_run_root(run_id) / "active_campaign_id"
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_text("7\n", encoding="utf-8")
            self.assertIsNone(paths.get_active_campaign_id(run_id))

            for bad_job in ("", "../x", "a/b", "f/", r"a\b", "a b", ".h",
                            "/a"):
                with self.assertRaises(constants.ValidationError):
                    paths.get_sealed_job_artifacts_dir(bad_job)
            for bad_cid in (0, -3, True, False, "", "abc", "../1", "1.5", "²"):
                with self.assertRaises(constants.ValidationError):
                    paths.validate_campaign_id(
                        bad_cid)  # type: ignore[arg-type]
            for bad_run in ("", "../x", "a/b"):
                with self.assertRaises(constants.ValidationError):
                    paths.get_active_campaign_marker(bad_run)

            ws = pathlib.Path(tmp).resolve() / "ws"
            (ws / "src").mkdir(parents=True)
            sym_anc = pathlib.Path(tmp).resolve() / "sym_anc"
            sym_anc.symlink_to(pathlib.Path(tmp).resolve())
            ws_sym = sym_anc / "ws"
            for base_ws, sub, rel in ((ws, ws / "s/1", "s/1"),
                                      (ws_sym, pathlib.Path("s/2"), "s/2"),
                                      (ws_sym, ws_sym / "s/3", "s/3")):
                self.assertEqual(
                    paths.verify_workspace_subpath(base_ws, sub, "--out-dir"),
                    ws / rel)
            outside = pathlib.Path(tmp).resolve() / "secret"
            outside.write_text("x", encoding="utf-8")
            (ws / "sym_out").symlink_to(outside)
            for bad_sub in (outside, ws, ws / "src/in_src",
                            ws / "sym_out/sub"):
                with self.assertRaises(constants.ValidationError):
                    paths.verify_workspace_subpath(ws, bad_sub, "--out-dir")


if __name__ == "__main__":
    unittest.main()
