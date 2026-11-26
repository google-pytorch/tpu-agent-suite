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
"""Tests for the nightly-onboard-optimize YAML suite and hill-climb tasks."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import types
import unittest
from unittest import mock

from eval.suite_to_harbor import suite_to_harbor

REPO = pathlib.Path(__file__).resolve().parents[2]
SUITE_DIR = REPO / "eval" / "suites" / "nightly-onboard-optimize"

ONBOARD_CASES = (
    "onboarding-gemma-4-31b-it",
    "onboarding-gpt-oss-20b",
    "onboarding-mistral-small-24b",
    "onboarding-qwen3-8-27b",
)

HILLCLIMB_CASES = {
    "onboard-hillclimb-gemma-4-26b-a4b-it": {
        "url": "https://huggingface.co/google/gemma-4-26B-A4B-it",
        "memory_mb": 65536,
        "storage_mb": 102400,
        "timeout_sec": 21600.0,
    },
    "onboard-hillclimb-qwen3-5-27b": {
        "url": "https://huggingface.co/Qwen/Qwen3.5-27B",
        "memory_mb": 65536,
        "storage_mb": 102400,
        "timeout_sec": 21600.0,
    },
    "onboard-hillclimb-qwen3-6-27b": {
        "url": "https://huggingface.co/Qwen/Qwen3.6-27B",
        "memory_mb": 65536,
        "storage_mb": 102400,
        "timeout_sec": 21600.0,
    },
    "onboard-hillclimb-glm-5-2-fp8": {
        "url": "https://huggingface.co/zai-org/GLM-5.2-FP8",
        "memory_mb": 65536,
        "storage_mb": 102400,
        "timeout_sec": 36000.0,
    },
    "onboard-hillclimb-glm-5-3-flash": {
        "url": "https://huggingface.co/zai-org/GLM-5.3-Flash",
        "memory_mb": 65536,
        "storage_mb": 102400,
        "timeout_sec": 36000.0,
    },
}

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


def _load_speedup_module(path: pathlib.Path):
    stub = types.ModuleType("rewardkit")

    def _criterion(fn=None, **kwargs):
        del kwargs
        return (lambda f: f) if fn is None else fn

    stub.criterion = _criterion
    with mock.patch.dict(sys.modules, {"rewardkit": stub}):
        spec = importlib.util.spec_from_file_location("compiled_hc_speedup",
                                                      path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules["compiled_hc_speedup"] = module
        spec.loader.exec_module(module)
    return module


class OnboardHillclimbSuiteTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = pathlib.Path(tempfile.mkdtemp(prefix="noo_suite_test_"))
        cls.out_dir = cls._tmp / "nightly-onboard-optimize"
        assert suite_to_harbor.compile_suite_spec(SUITE_DIR, cls.out_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def test_compiled_suite_contains_all_nine_cases(self) -> None:
        actual = sorted(p.name for p in self.out_dir.iterdir() if p.is_dir())
        expected = sorted((*ONBOARD_CASES, *HILLCLIMB_CASES))
        self.assertEqual(actual, expected)

    def test_hillclimb_task_tomls_and_dockerfiles(self) -> None:
        for slug, cfg in HILLCLIMB_CASES.items():
            task_dir = self.out_dir / slug
            with self.subTest(task=slug):
                toml_data = tomllib.loads(
                    (task_dir / "task.toml").read_text("utf-8"))
                self.assertEqual(toml_data["task"]["name"],
                                 f"nightly-onboard-optimize/{slug}")
                self.assertEqual(toml_data["environment"]["cpus"], 2)
                self.assertEqual(toml_data["environment"]["memory_mb"],
                                 cfg["memory_mb"])
                self.assertEqual(toml_data["environment"]["storage_mb"],
                                 cfg["storage_mb"])
                self.assertEqual(toml_data["environment"]["env"],
                                 {"HF_MODEL_URL": cfg["url"]})
                self.assertEqual(toml_data["agent"]["timeout_sec"],
                                 cfg["timeout_sec"])
                self.assertEqual(toml_data["verifier"]["timeout_sec"], 1800.0)
                self.assertEqual(toml_data["verifier"]["env"],
                                 {"HF_MODEL_URL": cfg["url"]})

                df = (task_dir / "environment" /
                      "Dockerfile").read_text("utf-8")
                self.assertIn(
                    "us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/nexus-host@sha256:7ff58a98ae2655dbcf25fa0aeb3e8081281bfb3061f97581a98951e98371fff4",
                    df,
                )
                self.assertIn("openssh-client", df)
                self.assertIn(
                    "-c /etc/nexus-host/constraints.txt transformers absl-py pytest",
                    df,
                )

    def test_instructions_are_neutral_across_arms(self) -> None:
        for sample in ("Use tpu-agent", "Install tpu-agent-suite",
                       "Run /close-gaps"):
            self.assertTrue(
                any(re.search(p, sample) for p in DISALLOWED_PROMPT_TERMS),
                f"Expected DISALLOWED_PROMPT_TERMS to flag {sample!r}",
            )
        bucket_url = "gs://tpu-agent-eval/checkpoints/zai-org/GLM-5.2-FP8/"
        for pat in DISALLOWED_PROMPT_TERMS:
            self.assertIsNone(re.search(pat, bucket_url))

        for slug, cfg in HILLCLIMB_CASES.items():
            text = (self.out_dir / slug / "instruction.md").read_text("utf-8")
            with self.subTest(task=slug):
                for pat in DISALLOWED_PROMPT_TERMS:
                    self.assertIsNone(
                        re.search(pat, text),
                        f"{slug}/instruction.md contains suite-specific pattern {pat!r}",
                    )
                self.assertIsNone(
                    re.search(r"\{\{.*?\}\}", text),
                    f"{slug}/instruction.md contains unexpanded template placeholder",
                )
                self.assertIn(cfg["url"], text)
                self.assertIn("output_tps_per_chip", text)
                self.assertIn("benchmark_results.json", text)
                self.assertIn('"accelerator"', text)
                self.assertIn("$TASK_DOCKER", text)
                self.assertIn("### Model-Specific Notes", text)

    def test_compiled_verifiers_weights_params_and_common_parity(self) -> None:
        invoked_src = (REPO / "eval" / "verifiers" / "common" /
                       "invoked_subagent.py").read_bytes()
        schema_src = (REPO / "eval" / "verifiers" / "common" /
                      "schema_compliance.py").read_bytes()

        for slug in HILLCLIMB_CASES:
            tests = self.out_dir / slug / "tests"
            with self.subTest(task=slug):
                reward = tomllib.loads(
                    (tests / "reward.toml").read_text("utf-8"))
                self.assertEqual(reward["reward"][0]["name"], "reward")
                self.assertEqual(
                    {p.name
                     for p in tests.iterdir() if p.is_dir()},
                    {
                        "onboarding_success",
                        "speedup",
                        "efficiency",
                        "cheating",
                        "execution_report",
                        "invoked_subagent",
                        "schema_compliance",
                    },
                )
                params = json.loads(
                    (tests / "verifier_params.json").read_text("utf-8"))
                self.assertEqual(
                    params["speedup"],
                    {
                        "target_speedup": 1.0,
                        "primary_regime": "8k/1k",
                        "per_chip": True,
                        "require_manifest": True,
                    },
                )
                self.assertEqual(
                    params["efficiency"],
                    {
                        "agent_runtime_median": 250.0,
                        "agent_runtime_ceiling": 1200.0,
                        "total_tokens_median": 45.0,
                        "total_tokens_ceiling": 500.0,
                        "agent_turns_median": 310.0,
                        "agent_turns_ceiling": 3900.0,
                    },
                )
                self.assertEqual(
                    (tests / "invoked_subagent" /
                     "invoked_subagent.py").read_bytes(),
                    invoked_src,
                )
                self.assertEqual(
                    (tests / "schema_compliance" /
                     "schema_compliance.py").read_bytes(),
                    schema_src,
                )
                test_sh = (tests / "test.sh").read_text("utf-8")
                self.assertIn("aggregate_reward.py", test_sh)

    def test_compiled_speedup_and_test_sh_end_to_end(self) -> None:
        tests_dir = (self.out_dir / "onboard-hillclimb-gemma-4-26b-a4b-it" /
                     "tests")
        mod = _load_speedup_module(tests_dir / "speedup" /
                                   "calculate_speedup.py")

        def write_raw(path: pathlib.Path, isl: int, osl: int, conc: int,
                      tps: float) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({
                    "isl": isl,
                    "osl": osl,
                    "concurrency": conc,
                    "completed": conc * 4,
                    "failed": 0,
                    "output_tps": tps,
                }),
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            # 1. Per-chip speedup with mixed Phase 1 1k/1k + 8k/1k rows.
            ws1 = root / "ws_same_chips"
            write_raw(ws1 / "onboarding" / "base_1k1k_c16.json", 1024, 1024,
                      16, 1500.0)
            write_raw(ws1 / "onboarding" / "base_8k1k_c16.json", 8192, 1024,
                      16, 1200.0)
            write_raw(ws1 / "hillclimb" / "opt_8k1k_c16.json", 8192, 1024, 16,
                      1800.0)
            (ws1 / "benchmark_results.json").write_text(
                json.dumps({
                    "baseline": [
                        {
                            "regime": "1k/1k",
                            "concurrency": 16,
                            "num_chips": 4,
                            "output_tps": 1500.0,
                            "output_tps_per_chip": 375.0,
                            "raw_output_path": "onboarding/base_1k1k_c16.json",
                        },
                        {
                            "regime": "8k/1k",
                            "concurrency": 16,
                            "num_chips": 4,
                            "output_tps": 1200.0,
                            "output_tps_per_chip": 300.0,
                            "raw_output_path": "onboarding/base_8k1k_c16.json",
                        },
                    ],
                    "optimized": [{
                        "regime":
                        "8k/1k",
                        "concurrency":
                        16,
                        "num_chips":
                        4,
                        "output_tps":
                        1800.0,
                        "output_tps_per_chip":
                        450.0,
                        "raw_output_path":
                        "hillclimb/opt_8k1k_c16.json",
                    }],
                }),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"TESTS_DIR": str(tests_dir)}):
                self.assertAlmostEqual(mod.tpu_speedup(ws1), 1.5, places=4)

            # 2. Chip-scaling guard: 4 -> 8 chips with 1200 -> 2400 TPS => 1.0x.
            ws2 = root / "ws_scaled_chips"
            write_raw(ws2 / "onboarding" / "base_8k1k_c16.json", 8192, 1024,
                      16, 1200.0)
            write_raw(ws2 / "hillclimb" / "opt_8k1k_c16.json", 8192, 1024, 16,
                      2400.0)
            (ws2 / "benchmark_results.json").write_text(
                json.dumps({
                    "baseline": [{
                        "regime":
                        "8k/1k",
                        "concurrency":
                        16,
                        "num_chips":
                        4,
                        "output_tps":
                        1200.0,
                        "output_tps_per_chip":
                        300.0,
                        "raw_output_path":
                        "onboarding/base_8k1k_c16.json",
                    }],
                    "optimized": [{
                        "regime":
                        "8k/1k",
                        "concurrency":
                        16,
                        "num_chips":
                        8,
                        "output_tps":
                        2400.0,
                        "output_tps_per_chip":
                        300.0,
                        "raw_output_path":
                        "hillclimb/opt_8k1k_c16.json",
                    }],
                }),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"TESTS_DIR": str(tests_dir)}):
                self.assertAlmostEqual(mod.tpu_speedup(ws2), 1.0, places=4)

            # 3. Compiled tests/test.sh applies gated reward formula + XProf diagnostic.
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
                "TESTS_DIR": str(tests_dir),
                "WORKSPACE_DIR": str(ws),
                "LOGS_DIR": str(logs),
            }
            test_sh = tests_dir / "test.sh"

            subprocess.run(["bash", str(test_sh)],
                           env={
                               **env_base, "TEST_SPEEDUP": "0.8"
                           },
                           check=True,
                           capture_output=True)
            r_slow = json.loads(
                (logs / "verifier" / "reward.json").read_text("utf-8"))
            self.assertEqual(r_slow["hillclimb_success"], 0.0)
            self.assertEqual(r_slow["speedup_score"], 0.0)
            self.assertEqual(r_slow["reward"], 0.3)

            subprocess.run(["bash", str(test_sh)],
                           env={
                               **env_base, "TEST_SPEEDUP": "2.0"
                           },
                           check=True,
                           capture_output=True)
            r_fast = json.loads(
                (logs / "verifier" / "reward.json").read_text("utf-8"))
            self.assertEqual(r_fast["hillclimb_success"], 1.0)
            self.assertEqual(r_fast["speedup_score"], 1.0)
            self.assertEqual(r_fast["reward"], 0.98)
            details = json.loads(
                (logs / "verifier" / "reward-details.json").read_text("utf-8"))
            self.assertTrue(details["post_hillclimb_xprof"]["passed"])

            subprocess.run(["bash", str(test_sh)],
                           env={
                               **env_base, "TEST_SPEEDUP": "2.0",
                               "TEST_CHEATING": "1.0"
                           },
                           check=True,
                           capture_output=True)
            r_cheat = json.loads(
                (logs / "verifier" / "reward.json").read_text("utf-8"))
            self.assertEqual(r_cheat["reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
