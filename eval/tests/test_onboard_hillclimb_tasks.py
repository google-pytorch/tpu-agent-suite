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
"""Unit tests for nightly-onboard-optimize/onboard-hillclimb-* tasks and verifiers."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import tomllib
import types
import unittest
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parents[2]
DATASET = REPO / "eval" / "datasets" / "nightly-onboard-optimize"
REF_DOCKERFILE = (DATASET / "onboarding-gemma-4-31b-it" / "environment" /
                  "Dockerfile")

TASKS = (
    "onboard-hillclimb-gemma-4-26b-a4b-it",
    "onboard-hillclimb-glm-5-2-fp8",
    "onboard-hillclimb-glm-5-3-flash",
    "onboard-hillclimb-qwen3-5-27b",
    "onboard-hillclimb-qwen3-6-27b",
)

DISALLOWED_PROMPT_TERMS = (
    r"(?<![\w-])tpu-agent(?![\w-])",
    r"(?<![\w-])tpu-agent-suite(?![\w-])",
    r"(?<![\w])/close-gaps(?![\w-])",
    r"(?<![\w-])hill-climb\.md(?![\w-])",
    r"(?<![\w-])tpu-forward-bench(?![\w-])",
    r"(?<![\w-])tpu-roofline(?![\w-])",
    r"(?<![\w-])tpu-diagnostics(?![\w-])",
    r"(?<![\w-])tpu-performance-optimization(?![\w-])",
    r"(?<![\w-])tpu-kernel-development(?![\w-])",
    r"(?<![\w-])tpu-pallas(?![\w-])",
    r"(?<![\w-])tpu-code-arm(?![\w-])",
    r"(?<![\w-])tpu-serving-bench(?![\w-])",
    r"\bserving_sweep\.py\b",
    r"\bruntime\.py\b",
    r"\bgke_job\b",
    r"\bshm_size_gb\b",
    r"\binstalled agent suite\b",
)

SHARED_TEST_RELPATHS = (
    "reward.toml",
    "test.sh",
    "cheating/judge.toml",
    "efficiency/framework_directed_penalty.py",
    "efficiency/stats.txt",
    "efficiency/trajectory_score.py",
    "efficiency/valid_tool_calls.py",
    "invoked_subagent/check.py",
    "schema_compliance/check.py",
    "speedup/calculate_speedup.py",
    "speedup/target_speedup.txt",
)


def _load_speedup_module(path: pathlib.Path):
    stub = types.ModuleType("rewardkit")

    def _criterion(fn=None, **kwargs):
        del kwargs
        return (lambda f: f) if fn is None else fn

    stub.criterion = _criterion
    with mock.patch.dict(sys.modules, {"rewardkit": stub}):
        spec = importlib.util.spec_from_file_location("onboard_hc_speedup",
                                                      path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules["onboard_hc_speedup"] = module
        spec.loader.exec_module(module)
    return module


class OnboardHillclimbTasksTest(unittest.TestCase):

    def test_dockerfiles_use_canonical_nexus_host(self) -> None:
        expected = REF_DOCKERFILE.read_text("utf-8")
        self.assertIn("nexus-host@sha256:", expected)
        for slug in TASKS:
            df = DATASET / slug / "environment" / "Dockerfile"
            with self.subTest(task=slug):
                self.assertEqual(df.read_text("utf-8"), expected)

    def test_instructions_are_neutral_across_arms(self) -> None:
        # Sanity-check boundary behavior on suite names vs GCS checkpoint bucket URLs.
        for sample in ("Use tpu-agent", "Install tpu-agent-suite",
                       "Run /close-gaps"):
            self.assertTrue(
                any(re.search(p, sample) for p in DISALLOWED_PROMPT_TERMS),
                f"Expected DISALLOWED_PROMPT_TERMS to flag {sample!r}",
            )
        bucket_url = "gs://tpu-agent-eval/checkpoints/zai-org/GLM-5.2-FP8/"
        for pat in DISALLOWED_PROMPT_TERMS:
            self.assertIsNone(re.search(pat, bucket_url))

        for slug in TASKS:
            text = (DATASET / slug / "instruction.md").read_text("utf-8")
            with self.subTest(task=slug):
                for pat in DISALLOWED_PROMPT_TERMS:
                    self.assertIsNone(
                        re.search(pat, text),
                        f"{slug}/instruction.md contains suite-specific pattern {pat!r}",
                    )
                self.assertIn("output_tps_per_chip", text)
                self.assertIn("benchmark_results.json", text)
                self.assertIn("$TASK_DOCKER", text)

    def test_shared_verifier_files_have_byte_parity_and_canonical_weights(
            self) -> None:
        anchor = DATASET / TASKS[0] / "tests"
        reward = tomllib.loads((anchor / "reward.toml").read_text("utf-8"))
        weights = reward["reward"][0]["weights"]
        self.assertEqual(
            weights,
            {
                "onboarding_success": 0.3,
                "speedup": 0.3,
                "efficiency": 0.1,
                "cheating": -1.0,
                "execution_report": 0.0,
                "invoked_subagent": 0.0,
                "schema_compliance": 0.0,
            },
        )
        for slug in TASKS[1:]:
            other = DATASET / slug / "tests"
            for rel in SHARED_TEST_RELPATHS:
                with self.subTest(task=slug, file=rel):
                    self.assertEqual(
                        (other / rel).read_bytes(),
                        (anchor / rel).read_bytes(),
                    )

    def test_per_chip_speedup_calculation_and_chip_scaling_guard(self) -> None:
        mod = _load_speedup_module(DATASET / TASKS[0] / "tests" / "speedup" /
                                   "calculate_speedup.py")

        def write_raw(path: pathlib.Path, conc: int, tps: float) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({
                    "isl": 8192,
                    "osl": 1024,
                    "concurrency": conc,
                    "completed": conc * 4,
                    "failed": 0,
                    "output_tps": tps,
                }),
                encoding="utf-8",
            )

        # Case 1: Same chip count (4 -> 4), 1200 -> 1800 tok/s => 1.5x per-chip speedup.
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp)
            write_raw(ws / "onboarding" / "base_c16.json", 16, 1200.0)
            write_raw(ws / "hillclimb" / "opt_c16.json", 16, 1800.0)
            (ws / "benchmark_results.json").write_text(
                json.dumps({
                    "optimization_metric":
                    "output_tps_per_chip",
                    "baseline": [{
                        "regime": "8k/1k",
                        "concurrency": 16,
                        "num_chips": 4,
                        "output_tps": 1200.0,
                        "output_tps_per_chip": 300.0,
                        "raw_output_path": "onboarding/base_c16.json",
                    }],
                    "optimized": [{
                        "regime": "8k/1k",
                        "concurrency": 16,
                        "num_chips": 4,
                        "output_tps": 1800.0,
                        "output_tps_per_chip": 450.0,
                        "raw_output_path": "hillclimb/opt_c16.json",
                    }],
                }),
                encoding="utf-8",
            )
            self.assertAlmostEqual(mod.tpu_speedup(ws), 1.5, places=4)

        # Case 2: Doubling chips (4 -> 8) with linear total TPS (1200 -> 2400) => 1.0x per-chip speedup (no fake 2.0x).
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp)
            write_raw(ws / "onboarding" / "base_c16.json", 16, 1200.0)
            write_raw(ws / "hillclimb" / "opt_c16.json", 16, 2400.0)
            (ws / "benchmark_results.json").write_text(
                json.dumps({
                    "optimization_metric":
                    "output_tps_per_chip",
                    "baseline": [{
                        "regime": "8k/1k",
                        "concurrency": 16,
                        "num_chips": 4,
                        "output_tps": 1200.0,
                        "output_tps_per_chip": 300.0,
                        "raw_output_path": "onboarding/base_c16.json",
                    }],
                    "optimized": [{
                        "regime": "8k/1k",
                        "concurrency": 16,
                        "num_chips": 8,
                        "output_tps": 2400.0,
                        "output_tps_per_chip": 300.0,
                        "raw_output_path": "hillclimb/opt_c16.json",
                    }],
                }),
                encoding="utf-8",
            )
            self.assertAlmostEqual(mod.tpu_speedup(ws), 1.0, places=4)

        # Case 3: Missing num_chips is rejected (cannot bypass per-chip normalization).
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp)
            write_raw(ws / "onboarding" / "base_c16.json", 16, 1200.0)
            write_raw(ws / "hillclimb" / "opt_c16.json", 16, 2400.0)
            (ws / "benchmark_results.json").write_text(
                json.dumps({
                    "baseline": [{
                        "regime": "8k/1k",
                        "concurrency": 16,
                        "output_tps": 1200.0,
                        "raw_output_path": "onboarding/base_c16.json",
                    }],
                    "optimized": [{
                        "regime": "8k/1k",
                        "concurrency": 16,
                        "output_tps": 2400.0,
                        "raw_output_path": "hillclimb/opt_c16.json",
                    }],
                }),
                encoding="utf-8",
            )
            self.assertEqual(mod.tpu_speedup(ws), 0.0)

    def test_test_sh_applies_canonical_gated_reward_formula(self) -> None:
        test_sh = DATASET / TASKS[0] / "tests" / "test.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            ws = root / "workspace"
            logs = root / "logs"
            fake_bin = root / "bin"
            ws.mkdir()
            (logs / "verifier").mkdir(parents=True)
            (logs / "artifacts" / "hillclimb" / "xprof").mkdir(parents=True)
            (logs / "artifacts" / "hillclimb" / "xprof" /
             "trace.xplane.pb").write_bytes(b"xplane")
            (logs / "artifacts" / "xprof_bottlenecks_and_gaps.md").write_text(
                "x" * 150, encoding="utf-8")
            fake_bin.mkdir()

            # Stub rewardkit CLI to emit raw criterion scores without network/LLM calls.
            fake_rk = fake_bin / "rewardkit"
            fake_rk.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, pathlib, sys\n"
                "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
                "spd = float(os.environ['TEST_SPEEDUP'])\n"
                "cheat = float(os.environ.get('TEST_CHEATING', '0.0'))\n"
                "out.write_text(json.dumps({\n"
                "  'onboarding_success': 1.0,\n"
                "  'speedup': 0.5,\n"
                "  'efficiency': 0.8,\n"
                "  'cheating': cheat,\n"
                "  'invoked_subagent': 1.0,\n"
                "  'schema_compliance': 1.0,\n"
                "  'reward': 0.5,\n"
                "}))\n"
                "out.with_name('reward-details.json').write_text(json.dumps({\n"
                "  'speedup': {'criteria': [{'name': 'tpu_speedup', 'raw': spd}]},\n"
                "  'efficiency': {'criteria': [\n"
                "    {'name': 'trajectory_score', 'value': 0.8},\n"
                "    {'name': 'valid_tool_calls', 'value': 1.0},\n"
                "    {'name': 'framework_directed_penalty', 'value': 1.0},\n"
                "  ]},\n"
                "}))\n",
                encoding="utf-8",
            )
            fake_rk.chmod(0o755)

            env_base = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
            }

            # Slower optimized server (0.8x) -> hillclimb_success=0.0, reward=0.3 (not 1.0).
            subprocess.run(
                [
                    "bash",
                    str(test_sh),
                    str(test_sh.parent),
                    str(ws),
                    str(logs)
                ],
                env={
                    **env_base, "TEST_SPEEDUP": "0.8"
                },
                check=True,
                capture_output=True,
            )
            r_slow = json.loads(
                (logs / "verifier" / "reward.json").read_text("utf-8"))
            self.assertEqual(r_slow["hillclimb_success"], 0.0)
            self.assertEqual(r_slow["speedup_score"], 0.0)
            self.assertEqual(r_slow["reward"], 0.3)

            # 2.0x per-chip speedup -> hillclimb_success=1.0, speedup_score=1.0, reward=0.98.
            subprocess.run(
                [
                    "bash",
                    str(test_sh),
                    str(test_sh.parent),
                    str(ws),
                    str(logs)
                ],
                env={
                    **env_base, "TEST_SPEEDUP": "2.0"
                },
                check=True,
                capture_output=True,
            )
            r_fast = json.loads(
                (logs / "verifier" / "reward.json").read_text("utf-8"))
            self.assertEqual(r_fast["hillclimb_success"], 1.0)
            self.assertEqual(r_fast["speedup_score"], 1.0)
            self.assertEqual(r_fast["reward"], 0.98)

            # Cheating=1.0 -> reward=0.0.
            subprocess.run(
                [
                    "bash",
                    str(test_sh),
                    str(test_sh.parent),
                    str(ws),
                    str(logs)
                ],
                env={
                    **env_base, "TEST_SPEEDUP": "2.0",
                    "TEST_CHEATING": "1.0"
                },
                check=True,
                capture_output=True,
            )
            r_cheat = json.loads(
                (logs / "verifier" / "reward.json").read_text("utf-8"))
            self.assertEqual(r_cheat["reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
