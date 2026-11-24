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
"""Unit tests for `runtime.py hc` and `framework.hc.campaign_ops`."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import unittest

from framework import jsonio
from framework import paths as base_paths
from framework.hc import campaign_ops, constants, db, paths
from framework.tests.hc import _helpers

Path = pathlib.Path


class TestCampaignLifecycleCli(_helpers.HcCliTestCase):
    """Tests for `runtime.py hc` CLI dispatch and campaign lifecycle commands."""

    def test_open_guards_lifecycle_and_telemetry(self) -> None:
        fake_run = "20260101T000000Z-000000000000"
        c_arg, p_arg, d_arg = (str(self.camp_json), str(self.proto_json),
                               str(self.code_dir))
        self.assertNotEqual(
            self._run_hc("campaign", "open", c_arg, p_arg, "--code-dir", d_arg,
                         "--run-id", fake_run)[0], 0)
        self.assertFalse(base_paths.get_run_root(fake_run).exists())
        self.assertNotEqual(
            self._run_hc("campaign", "open", c_arg, p_arg, "--code-dir",
                         str(self.base / "missing"))[0], 0)
        (self.workspace / "src").write_text("regular file", encoding="utf-8")
        self.assertEqual(
            self._run_hc("campaign", "open", c_arg, p_arg, "--code-dir",
                         d_arg)[0], 1)
        (self.workspace / "src").unlink()

        self.assertEqual(
            self._run_hc("campaign", "open", c_arg, p_arg, "--code-dir",
                         d_arg)[0], 0)
        self.assertEqual(paths.get_active_campaign_id(self.run_id), 1)
        self.assertTrue(
            paths.get_active_campaign_marker(self.run_id).is_file())
        self.assertFalse((base_paths.get_run_root(self.run_id) /
                          "active_campaign_id").exists())
        self.assertTrue(
            (self.workspace / "src" / constants.FLAGS_FILENAME).is_file())

        scripts_dir = paths.get_campaign_scripts_dir(1)
        for sealed in ("target_runner.py", "serve_vllm.sh",
                       "apply_vllm_patch.sh", "eval_recipe.sh",
                       "serving_sweep.py", "manifest.json"):
            self.assertTrue((scripts_dir / sealed).is_file(), sealed)
        self.assertFalse((scripts_dir / "serve_sglang.sh").exists())
        manifest_doc = jsonio.load_json(scripts_dir / "manifest.json")
        self.assertTrue(
            Path(manifest_doc["e2e_sweep"]["source_path"]).is_file())
        self.assertTrue(manifest_doc["e2e_sweep"]["companions"])

        self.assertNotEqual(
            self._run_hc("campaign", "open", c_arg, p_arg, "--code-dir",
                         d_arg)[0], 0)
        self.assertEqual(
            self._run_hc("candidate", "abandon", "²", "bad id")[0], 1)

        for _ in range(2):
            self.assertEqual(
                self._run_hc("campaign", "set-microbenchmark-protocol",
                             str(self.micro_json))[0], 0)
        self.assertTrue((scripts_dir / "forward_run.py").is_file())
        self.assertTrue((scripts_dir / "microbenchmark_p3" /
                         "micro_plan.steps.json").is_file())
        with db.open_db() as conn, db.transaction(conn):
            conn.execute(
                "INSERT INTO Jobs (job_id, round, candidate, protocol, trial, kind, status) "
                "VALUES ('job-micro-1', NULL, 1, 3, 'trial_1', 'Microbenchmark', 'Dispatched')"
            )
        self.assertNotEqual(
            self._run_hc("campaign", "set-microbenchmark-protocol",
                         str(self.micro_json))[0], 0)

        self.assertEqual(
            self._run_hc("campaign", "block", "  waiting on quota  ")[0], 0)
        self.assertEqual(self._run_hc("campaign", "unblock")[0], 0)
        self.assertEqual(self._run_hc("campaign", "close", "--abort")[0], 0)

        with db.open_db() as conn:
            camp = conn.execute(
                "SELECT state, close_reason, closed_at FROM Campaigns WHERE id = 1"
            ).fetchone()
            self.assertEqual((camp["state"], camp["close_reason"]),
                             ("Closed", "Aborted"))
            self.assertIsNotNone(camp["closed_at"])
            job = conn.execute(
                "SELECT status FROM Jobs WHERE job_id = 'job-micro-1'"
            ).fetchone()
            self.assertEqual(job["status"], "Cancelled")

        events_list = jsonio.read_jsonl(paths.get_campaign_events_log(1))
        names = [e["event_name"] for e in events_list]
        for expected in ("campaign.opened",
                         "campaign.microbenchmark_protocol_set",
                         "campaign.blocked", "campaign.unblocked",
                         "job.settled", "campaign.closed"):
            self.assertIn(expected, names)
        opened = next(e for e in events_list
                      if e["event_name"] == "campaign.opened")
        self.assertIn("target_runner.py", opened["script_digests"])

    def test_manifest_script_sealing_microbenchmark_targets_and_engine_guards(
            self) -> None:
        c_arg, p_arg, d_arg = (str(self.camp_json), str(self.proto_json),
                               str(self.code_dir))

        # SGLang has no accuracy_scripts entry today: non-null accuracy_gate fails fast.
        sglang_camp = self.base / "sglang_campaign.json"
        jsonio.write_json(
            sglang_camp,
            dict(name="sgl",
                 owner="r",
                 target_model="m",
                 engine="sglang",
                 accelerator="v6e"),
        )
        self.assertNotEqual(
            self._run_hc("campaign", "open", str(sglang_camp), p_arg,
                         "--code-dir", d_arg)[0], 0)

        # Installed package without 'hillclimber' in build.json (e.g. train stub) fails fast.
        no_hc_pkg = base_paths.get_package_storage_root("train-pkg", "1.0.0")
        no_hc_pkg.mkdir(parents=True)
        jsonio.write_json(no_hc_pkg / "tpu-agent-suite-build.json", {
            "version": "1.0.0",
            "generated": True
        })
        self.assertNotEqual(
            self._run_hc("campaign", "open", c_arg, p_arg, "--code-dir", d_arg,
                         "--package", "train-pkg")[0], 0)

        # Installed unversioned package lookup via tpu-agent-suite-build.json['hillclimber'].
        pkg_root = base_paths.get_package_storage_root("demo-pkg", "1.2.0")
        for rel in ("scripts/s/sweep.py", "scripts/s/serve.sh",
                    "scripts/s/patch.sh", "scripts/s/eval.sh",
                    "scripts/s/fwd.py"):
            f = pkg_root / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(f"# {rel}\n", encoding="utf-8")
        jsonio.write_json(
            pkg_root / "tpu-agent-suite-build.json",
            {
                "version": "1.2.0",
                "generated": True,
                "hillclimber": {
                    "e2e_sweep_script": "scripts/s/sweep.py",
                    "serving_scripts": {
                        "vllm": "scripts/s/serve.sh"
                    },
                    "vllm_patch_applier": "scripts/s/patch.sh",
                    "accuracy_scripts": {
                        "vllm": "scripts/s/eval.sh"
                    },
                    "microbenchmark_runner": "scripts/s/fwd.py",
                },
            },
        )
        resolved_root, _, _ = campaign_ops.resolve_package_hillclimber(
            self.run_id, "demo-pkg")
        self.assertEqual(resolved_root, pkg_root)

        # Standalone .py microbenchmark target: symlink rejected, regular file sealed & tamper-proof.
        ws_micro = self.workspace / "bench_kernel.py"
        outside = self.base / "outside.py"
        outside.write_text("x = 1\n", encoding="utf-8")
        ws_micro.symlink_to(outside)
        micro_cfg = _helpers.make_protocol_cfg(is_micro=True)
        micro_cfg["execution_spec"] = dict(
            image="us-docker.pkg.dev/proj/vllm:latest",
            microbenchmark_target="bench_kernel.py",
        )
        jsonio.write_json(self.micro_json, micro_cfg)
        self.assertNotEqual(
            self._run_hc("campaign", "open", c_arg, p_arg, "--code-dir",
                         d_arg, "--microbenchmark-protocol",
                         str(self.micro_json), "--package", "demo-pkg")[0], 0)
        self.assertIsNone(paths.get_active_campaign_id(self.run_id))

        ws_micro.unlink()
        ws_micro.write_text("KERNEL_V = 1\n", encoding="utf-8")
        self.assertEqual(
            self._run_hc("campaign", "open", c_arg, p_arg, "--code-dir",
                         d_arg, "--microbenchmark-protocol",
                         str(self.micro_json), "--package", "demo-pkg")[0], 0)

        sealed_dir = paths.get_campaign_scripts_dir(1)
        sealed_micro = sealed_dir / "microbenchmark_p2" / "bench_kernel.py"
        self.assertIn("scripts/s/serve.sh",
                      (sealed_dir /
                       "serve_vllm.sh").read_text(encoding="utf-8"))
        self.assertFalse((sealed_dir / "forward_run.py").exists())
        ws_micro.write_text("KERNEL_V = 999\n", encoding="utf-8")
        self.assertEqual(sealed_micro.read_text(encoding="utf-8"),
                         "KERNEL_V = 1\n")

    def test_runtime_hc_subprocess_and_argparse_rejection_logging(
            self) -> None:
        runtime_script = Path(__file__).resolve().parents[2] / "runtime.py"
        cmd = [
            sys.executable,
            str(runtime_script),
            "hc",
            "--workspace",
            str(self.workspace),
            "campaign",
            "bogus",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={
                **os.environ, "TPU_AGENT_SUITE_HOME": str(self.home)
            },
            check=False,
        )
        self.assertEqual(proc.returncode, 2)
        records = jsonio.read_jsonl(paths.get_unscoped_events_log())
        self.assertEqual((len(records), records[0]["error_code"]),
                         (1, "INVALID_ARGS"))


if __name__ == "__main__":
    unittest.main()
