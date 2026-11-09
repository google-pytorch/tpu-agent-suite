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
"""Regression test suite for package/tpu-performance-optimization scripts in CI test discovery."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent
_PERF_SCRIPTS_DIR = _PACKAGE_ROOT / "scripts" / "tpu-performance-optimization"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework import study, utils  # noqa: E402
from framework.builder import translate  # noqa: E402


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_eval_arm_test = _load_script_module(
    "evaluate_arm_test", _PERF_SCRIPTS_DIR / "evaluate_arm_test.py")
_arm_microbench_test = _load_script_module(
    "arm_microbench_test", _PERF_SCRIPTS_DIR / "arm_microbench_test.py")


class TestTpuPerformanceEvaluateArm(_eval_arm_test.TestEvaluateArm):
    """Wire evaluate_arm_test into package discovery."""


class TestTpuPerformanceArmMicrobench(_arm_microbench_test.TestArmMicrobench):
    """Wire arm_microbench_test into package discovery."""


class TestTpuPerformanceSchemaParity(unittest.TestCase):
    """Enforce single source of truth for bottleneck kinds, arm schemas, and shipped scripts."""

    def test_bottleneck_kinds_match_across_framework_and_package_schemas(self):
        study_round_schema = json.loads(
            (_REPO_ROOT / "framework" / "schemas" /
             "study-round.json").read_text(encoding="utf-8"))
        arm_request_schema = json.loads(
            (_PACKAGE_ROOT / "schemas" /
             "arm-request.json").read_text(encoding="utf-8"))
        profiling_result_schema = json.loads(
            (_PACKAGE_ROOT / "schemas" /
             "profiling-result.json").read_text(encoding="utf-8"))

        round_kinds = tuple(study_round_schema["properties"]["arms"]["items"]
                            ["properties"]["kind"]["enum"])
        arm_kinds = tuple(arm_request_schema["properties"]["kind"]["enum"])
        lever = (profiling_result_schema["properties"]["traces"]["items"]
                 ["properties"]["levers"]["items"])
        lever_kinds = tuple(lever["properties"]["kind"]["enum"])

        self.assertEqual(study.BOTTLENECK_KINDS, round_kinds)
        self.assertEqual(study.BOTTLENECK_KINDS, arm_kinds)
        # A profile can find a cost that none of the study's kinds describe.
        self.assertEqual(study.BOTTLENECK_KINDS + ("other", ), lever_kinds)

    def test_arm_request_omits_redundant_round_property(self):
        arm_request_schema = json.loads(
            (_PACKAGE_ROOT / "schemas" /
             "arm-request.json").read_text(encoding="utf-8"))
        self.assertIn("run_id", arm_request_schema["properties"])
        self.assertNotIn("round", arm_request_schema["properties"])

    def test_shipped_evaluate_arm_runs_outside_repo_and_validates_arm_result_schema(
            self):
        arm_result_schema = json.loads(
            (_PACKAGE_ROOT / "schemas" /
             "arm-result.json").read_text(encoding="utf-8"))
        protocol_schema = study.load_study_protocol_schema()
        example_protocol = protocol_schema["examples"][0]
        baseline = {
            "metric": {
                "name": "output_tokens_per_s",
                "value": 37000.0,
                "unit": "tok/s",
                "replicates": 3,
                "provenance": "measured",
            }
        }
        gates = [
            {
                "name": "ttft-ceiling",
                "value": 3278.4,
                "execution_id": "j-shipped-01",
            },
            {
                "name": "accuracy-floor",
                "value": 0.96,
                "execution_id": "j-shipped-01",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            dist_dir = Path(tmp_dir) / "dist_claude"
            translate(_PACKAGE_ROOT, "claude-code", dist_dir)

            # tpu-performance-optimization is not in package.release.json; stage
            # evaluate_arm.py beside the bundled framework/ to check its
            # build-layout import path outside the repository.
            shipped_script = (dist_dir / "scripts" /
                              "tpu-performance-optimization" /
                              "evaluate_arm.py")
            shipped_script.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_PERF_SCRIPTS_DIR / "evaluate_arm.py", shipped_script)
            self.assertTrue(shipped_script.is_file())

            proto_file = Path(tmp_dir) / "protocol.json"
            base_file = Path(tmp_dir) / "baseline.json"
            gates_file = Path(tmp_dir) / "gates.json"
            proto_file.write_text(json.dumps(example_protocol),
                                  encoding="utf-8")
            base_file.write_text(json.dumps(baseline), encoding="utf-8")
            gates_file.write_text(json.dumps(gates), encoding="utf-8")

            clean_env = {
                k: v
                for k, v in os.environ.items() if k != "PYTHONPATH"
            }
            proc = subprocess.run(
                [
                    sys.executable,
                    str(shipped_script),
                    "--arm",
                    "a-shipped",
                    "--execution-id",
                    "j-shipped-01",
                    "--replicates",
                    "38580.0",
                    "38585.0",
                    "38582.5",
                    "--protocol",
                    str(proto_file),
                    "--baseline",
                    str(base_file),
                    "--gates",
                    str(gates_file),
                ],
                cwd=tmp_dir,
                env=clean_env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["arm"], "a-shipped")
            self.assertEqual(payload["verdict"], "win")
            self.assertEqual(
                utils.validate_instance(payload, arm_result_schema), [])

    def test_documented_arm_microbench_invocations_execute_successfully(self):
        """Assert that all command lines documented in executor prompt execute cleanly."""
        doc_path = (_PACKAGE_ROOT / "agents" /
                    "tpu-performance-optimization-executor.md")
        doc_text = doc_path.read_text(encoding="utf-8")
        self.assertNotIn("--run --cmd", doc_text)

        script_path = _PERF_SCRIPTS_DIR / "arm_microbench.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_code = Path(tmp_dir) / "sample_kernel.py"
            sample_code.write_text(
                "import torch\nx = torch.ones((128, 128))\n",
                encoding="utf-8",
            )

            # Test 1: Static syntax and import validation invocation
            res1 = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--file",
                    str(sample_code),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res1.returncode, 0, msg=res1.stderr)
            out1 = json.loads(res1.stdout)
            self.assertTrue(out1["valid"])
            self.assertEqual(out1["phase"], "syntax_check")

            # Test 2: Timed execution invocation with trailing positional command
            res2 = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--file",
                    str(sample_code),
                    "--run",
                    f"{sys.executable} -c 'print(1)'",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res2.returncode, 0, msg=res2.stderr)
            out2 = json.loads(res2.stdout)
            self.assertTrue(out2["valid"])
            self.assertEqual(out2["phase"], "microbench_run")


if __name__ == "__main__":
    unittest.main()
