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
"""Tests for nightly-functional/numerical-parity checks (tests/checks/checks.py)."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

# Stub rewardkit before importing checks
_rk = types.ModuleType("rewardkit")
_rk.criterion = lambda **_: (lambda fn: fn)
sys.modules.setdefault("rewardkit", _rk)

_CHECKS_PATH = (Path(__file__).resolve().parents[1] / "datasets" /
                "nightly-functional" / "numerical-parity" / "tests" /
                "checks" / "checks.py")
_spec = importlib.util.spec_from_file_location("numerical_parity_checks",
                                               _CHECKS_PATH)
checks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checks)


class NumericalParityChecksTest(unittest.TestCase):

    def setUp(self) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix="test_num_parity_"))
        self.addCleanup(shutil.rmtree, self.workspace, ignore_errors=True)

    def _write_trajectory(self, commands: list[str]) -> None:
        steps = []
        for cmd in commands:
            steps.append({
                "tool_calls": [{
                    "function_name": "run_shell_command",
                    "arguments": {
                        "command": cmd
                    },
                }]
            })
        (self.workspace / "trajectory.json").write_text(json.dumps(
            {"steps": steps}),
                                                        encoding="utf-8")

    def test_ran_on_tpu_accepts_compute_run_and_kubectl(self) -> None:
        for cmd in (
                "compute run --code capture_parity.py",
                "compute --package pkg run --code capture_parity.py -- --make_probe",
                "kubectl exec -it tpu-pod -- python3 capture_parity.py",
        ):
            with self.subTest(cmd=cmd):
                self._write_trajectory([cmd])
                self.assertTrue(checks.ran_on_tpu(self.workspace))

    def test_ran_on_tpu_rejects_pure_local_commands(self) -> None:
        for cmd in (
                "python3 capture_parity.py --make_probe",
                "pip install transformers",
                "cat parity_results.json",
        ):
            with self.subTest(cmd=cmd):
                self._write_trajectory([cmd])
                self.assertFalse(checks.ran_on_tpu(self.workspace))

    def test_parity_results_valid_accepts_valid_json(self) -> None:
        data = {
            "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "parity_passed": True,
            "top1_token_agreement": "13/13",
            "logits": {
                "max_abs": 0.45,
                "max_rel": 0.012,
                "rms": 0.12,
            },
            "modules": {
                "model.layers.0.self_attn": {
                    "max_abs": 0.05,
                    "max_rel": 0.008,
                },
                "model.layers.29.self_attn": {
                    "max_abs": 0.11,
                    "max_rel": 0.015,
                },
            },
        }
        (self.workspace / "parity_results.json").write_text(json.dumps(data),
                                                            encoding="utf-8")
        self.assertTrue(checks.parity_results_valid(self.workspace))

    def test_parity_results_valid_accepts_percentage_units(self) -> None:
        data = {
            "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "verdict": "PASS",
            "logits": {
                "max_abs": 0.45,
                "max_rel": "1.2%",
            },
            "modules": {
                "model.layers.0.self_attn": {
                    "max_abs": 0.05,
                    "max_rel": "0.8%",
                },
                "model.layers.29.self_attn": {
                    "max_abs": 0.11,
                    "max_rel": 1.5,
                },
            },
        }
        (self.workspace / "parity_results.json").write_text(json.dumps(data),
                                                            encoding="utf-8")
        self.assertTrue(checks.parity_results_valid(self.workspace))

    def test_parity_results_valid_accepts_compare_parity_report_format(
        self, ) -> None:
        report = {
            "model_id":
            "HuggingFaceTB/SmolLM2-135M-Instruct",
            "verdict":
            "PASS",
            "probes": [{
                "id":
                "p0",
                "logits": {
                    "max_abs": 0.45,
                    "max_rel": 0.012,
                    "shape_mismatch": False,
                },
                "modules": [
                    {
                        "name": "model.layers.0.self_attn",
                        "max_abs": 0.05,
                        "max_rel": 0.008,
                        "shape_mismatch": False,
                    },
                    {
                        "name": "model.layers.29.self_attn",
                        "max_abs": 0.11,
                        "max_rel": 0.015,
                        "shape_mismatch": False,
                    },
                ],
            }],
        }
        art_dir = self.workspace / "artifacts"
        art_dir.mkdir()
        (art_dir / "parity_report.json").write_text(json.dumps(report),
                                                    encoding="utf-8")
        self.assertTrue(checks.parity_results_valid(self.workspace))

    def test_parity_results_valid_rejects_missing_module(self) -> None:
        data = {
            "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "parity_passed": True,
            "logits": {
                "max_abs": 0.45,
                "max_rel": 0.012
            },
            "modules": {
                "model.layers.0.self_attn": {
                    "max_abs": 0.05,
                    "max_rel": 0.008
                }
                # missing layer 29
            },
        }
        (self.workspace / "parity_results.json").write_text(json.dumps(data),
                                                            encoding="utf-8")
        self.assertFalse(checks.parity_results_valid(self.workspace))

    def test_parity_results_valid_rejects_tolerance_breach(self) -> None:
        data = {
            "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "parity_passed": True,
            "logits": {
                "max_abs": 0.45,
                "max_rel": 0.012
            },
            "modules": {
                "model.layers.0.self_attn": {
                    "max_abs": 0.05,
                    "max_rel": 0.008
                },
                "model.layers.29.self_attn": {
                    "max_abs": 0.11,
                    "max_rel": 10.5
                },
            },
        }
        (self.workspace / "parity_results.json").write_text(json.dumps(data),
                                                            encoding="utf-8")
        self.assertFalse(checks.parity_results_valid(self.workspace))

    def test_parity_results_valid_rejects_failed_verdict(self) -> None:
        data = {
            "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "parity_passed": False,
            "logits": {
                "max_abs": 0.45,
                "max_rel": 0.012
            },
            "modules": {
                "model.layers.0.self_attn": {
                    "max_abs": 0.05,
                    "max_rel": 0.008
                },
                "model.layers.29.self_attn": {
                    "max_abs": 0.11,
                    "max_rel": 0.015
                },
            },
        }
        (self.workspace / "parity_results.json").write_text(json.dumps(data),
                                                            encoding="utf-8")
        self.assertFalse(checks.parity_results_valid(self.workspace))


if __name__ == "__main__":
    unittest.main()
