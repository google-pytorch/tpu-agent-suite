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
"""analyze.py -> propose.py through their command lines, judge.py, and knob_probe.py."""

import json
import os
import sys
import tempfile
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(_ROOT), "common"))

import analyze  # noqa: E402  # pylint: disable=wrong-import-position
import knob_probe  # noqa: E402  # pylint: disable=wrong-import-position
import propose  # noqa: E402  # pylint: disable=wrong-import-position

_DATA = os.path.join(_ROOT, "testdata")
_CONFIGS = os.path.join(os.path.dirname(_ROOT), "common", "testdata",
                        "oss_configs")


class CliTest(unittest.TestCase):

    def test_analyze_then_propose_writes_manifests(self):
        d = os.path.join(_DATA, "deployments", "gemma4")
        dep_path = os.path.join(d, "deployments.json")
        recipe = os.path.join(d, "launch_vllm.sh")
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        points = ",".join(meta["points"])
        with tempfile.TemporaryDirectory() as out:
            events = os.path.join(_DATA, "events",
                                  "gemma4_8k1k_prefill.parquet")
            self.assertEqual(
                analyze.main([
                    "--deployments", dep_path, "--launch-recipe", recipe,
                    "--points", points, "--events",
                    f"{events},role=prefill,point=8192:1024:128", "--out-dir",
                    out
                ]), 0)
            self.assertTrue(os.path.exists(os.path.join(out, "report.md")))
            self.assertEqual(
                propose.main([
                    "--findings",
                    os.path.join(out, "findings.json"), "--deployments",
                    dep_path, "--launch-recipe", recipe, "--points", points,
                    "--image", meta["image"], "--hf-config",
                    os.path.join(_CONFIGS, "gemma-4-31B-it.json"),
                    "--hbm-gib-per-device", "94.75", "--out",
                    os.path.join(out, "candidates.json"), "--changes-dir",
                    os.path.join(out, "changes")
                ]), 0)
            with open(os.path.join(out, "candidates.json"),
                      encoding="utf-8") as f:
                (cand, ) = json.load(f)["candidates"]
            with open(cand["deployments_json"], encoding="utf-8") as f:
                (dep, ) = json.load(f)["deployments"]
            with open(dep_path, encoding="utf-8") as f:
                (base, ) = json.load(f)["deployments"]
            self.assertEqual(
                (dep["tensor_parallel_size"], dep["data_parallel_size"]),
                (2, 4))
            # Everything else is the baseline's, including the flags only the
            # recipe held, now in the manifest so the launch path keeps them.
            for key in ("model_id", "engine", "port", "max_model_len",
                        "parallelism"):
                self.assertEqual(dep[key], base[key], key)
            self.assertEqual(dep["attention_backend"], "CUSTOM")
            self.assertIs(dep["no_enable_prefix_caching"], True)
            self.assertNotIn("host", dep)
            # The recipe's env is unchanged, so the manifest holds none.
            self.assertNotIn("env", dep)

    def test_serve_command_from_baseline(self):
        cmd = analyze.serve_command({
            "model_path":
            "/model",
            "tp":
            4,
            "dp":
            2,
            "env": {
                "A": "1"
            },
            "vllm_args": ["--max-num-batched-tokens", "4096"]
        })
        self.assertEqual(
            cmd, "A=1 vllm serve /model --tensor-parallel-size 4 "
            "--data-parallel-size 2 --max-num-batched-tokens 4096")

    def test_bad_events_option(self):
        with self.assertRaises(ValueError):
            analyze.parse_events("x.parquet,role=warmup")


class KnobProbeTest(unittest.TestCase):

    def test_tokens(self):
        self.assertEqual(knob_probe.tokens("--async-scheduling", "cli"),
                         ("--async-scheduling", "async_scheduling"))
        self.assertEqual(knob_probe.tokens("TPU_X", "env"),
                         ('"TPU_X"', "'TPU_X'"))

    def test_finds_knobs_in_source(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "envs.py"), "w",
                      encoding="utf-8") as f:
                f.write('environment_variables = {"TPU_PRESENT": int}\n')
            text = knob_probe.source_text([root])
            self.assertTrue(
                any(t in text
                    for t in knob_probe.tokens("TPU_PRESENT", "env")))
            self.assertFalse(
                any(t in text for t in knob_probe.tokens("TPU_ABSENT", "env")))


if __name__ == "__main__":
    unittest.main()
