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
"""End-to-end unit tests for all 7 subcommands of `cli.py`."""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cli  # noqa: E402


class CliTest(unittest.TestCase):
    """Tests every subcommand of `cli.py` in JSON and table modes."""

    def _run_cli(self, argv: list[str]) -> tuple[int, str]:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cli.main(argv)
        return code, stdout.getvalue()

    def test_enumerate_subcommand(self):
        code, out = self._run_cli([
            "enumerate",
            "--mesh",
            '{"fsdp":8,"tensor":4,"expert":2}',
            "--topology",
            "4x4x4",
            "--batch-size",
            "128",
            "--num-heads",
            "32",
            "--output-format",
            "json",
        ])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertGreater(payload["num_candidates"], 0)

    def test_breakdown_and_rank_portfolio_subcommands(self):
        with tempfile.TemporaryDirectory() as tmp:
            ops_path = pathlib.Path(tmp) / "op_profile.json"
            ops_path.write_text(
                json.dumps([
                    {
                        "name":
                        "all-gather.1",
                        "category":
                        "all-gather",
                        "raw_time_ps":
                        500_000_000.0,
                        "occurrences":
                        32,
                        "expression":
                        ("%all-gather.1 = bf16[8192,4096] all-gather(bf16[1024,4096] %p),"
                         " replica_groups=[8,8]<=[64]"),
                        "provenance":
                        "jit(train_step)/params/fsdp",
                    },
                    {
                        "name": "dot.2",
                        "category": "convolution",
                        "raw_time_ps": 500_000_000.0,
                        "occurrences": 32,
                        "expression": "%dot.2 = bf16[1024,4096] dot(...)",
                        "provenance": "jit(train_step)/mlp",
                    },
                ]),
                encoding="utf-8",
            )
            code_bd, out_bd = self._run_cli([
                "breakdown",
                "--ops-json",
                str(ops_path),
                "--mesh",
                '{"fsdp":8,"tensor":4,"expert":2}',
                "--output-format",
                "json",
            ])
            self.assertEqual(code_bd, 0)
            bd = json.loads(out_bd)
            self.assertAlmostEqual(bd["collective_pct"], 50.0, places=1)

            code_rk, out_rk = self._run_cli([
                "rank-portfolio",
                "--ops-json",
                str(ops_path),
                "--mesh",
                '{"fsdp":8,"tensor":4,"expert":2}',
                "--topology",
                "4x4x4",
                "--batch-size",
                "128",
                "--sequence-length",
                "4096",
                "--num-heads",
                "32",
                "--num-kv-heads",
                "8",
                "--output-format",
                "json",
            ])
            self.assertEqual(code_rk, 0)
            rk = json.loads(out_rk)
            self.assertGreater(len(rk["proposals"]), 0)

    def test_check_topology_and_plan_topology_subcommands(self):
        code_ct, out_ct = self._run_cli([
            "check-topology",
            "--mesh",
            '{"fsdp":4,"tensor":4,"expert":4}',
            "--topology",
            "4x4x4",
            "--platform",
            "v5p",
            "--output-format",
            "json",
        ])
        self.assertEqual(code_ct, 0)
        fit = json.loads(out_ct)
        self.assertTrue(fit["fits_without_split"])

        code_pt, out_pt = self._run_cli([
            "plan-topology",
            "--source-chips",
            "64",
            "--source-platform",
            "v5p",
            "--target-platform",
            "tpu7x",
            "--memory-per-chip-gib",
            "64.0",
            "--output-format",
            "json",
        ])
        self.assertEqual(code_pt, 0)
        plan = json.loads(out_pt)
        self.assertEqual(plan["target_platform"], "tpu7x")
        self.assertGreaterEqual(plan["recommended_chips"], 1)

    def test_validate_model_and_verify_compile_subcommands(self):
        code_vm, out_vm = self._run_cli([
            "validate-model",
            "--proposal",
            '{"tp":4,"dp":2}',
            "--model-config",
            '{"num_attention_heads":32,"num_key_value_heads":8,"hidden_size":4096,"num_hidden_layers":32,"vocab_size":32000}',
            "--total-tpu-chips",
            "8",
            "--tpu-type",
            "v6e",
            "--output-format",
            "json",
        ])
        self.assertEqual(code_vm, 0)
        vm = json.loads(out_vm)
        self.assertTrue(vm["is_valid"])

        with tempfile.TemporaryDirectory() as tmp:
            code_vc, out_vc = self._run_cli([
                "verify-compile",
                "--candidate-dir",
                tmp,
                "--compile-log",
                "RESOURCE_EXHAUSTED: CompileTimeHbmOom exceeds available HBM",
                "--exit-code",
                "1",
                "--output-format",
                "json",
            ])
            self.assertEqual(code_vc, 0)
            vc = json.loads(out_vc)
            self.assertEqual(vc["verdict"], "DROPPED")


if __name__ == "__main__":
    unittest.main()
