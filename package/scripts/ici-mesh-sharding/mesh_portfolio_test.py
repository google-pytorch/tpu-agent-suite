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
"""Unit tests for `mesh_portfolio.py` and `rank_mesh_proposals.py`."""

from __future__ import annotations

import json
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import collective_breakdown  # noqa: E402
import mesh_portfolio  # noqa: E402
import rank_mesh_proposals  # noqa: E402


class MeshPortfolioTest(unittest.TestCase):
    """Tests analytical communication-volume scoring and 7-slot portfolio ranking."""

    def _sample_ops(self) -> list[dict[str, object]]:
        return [
            {
                "operation":
                "all-gather.10",
                "category":
                "all-gather",
                "raw_time_ps":
                450_000_000.0,
                "occurrences":
                32,
                "expression":
                ("%all-gather.10 = bf16[8192,4096] all-gather(bf16[1024,4096] %p),"
                 " replica_groups=[8,8]<=[64]"),
                "provenance":
                "jit(train_step)/params/fsdp",
            },
            {
                "operation":
                "all-reduce.11",
                "category":
                "all-reduce",
                "raw_time_ps":
                200_000_000.0,
                "occurrences":
                32,
                "expression":
                ("%all-reduce.11 = bf16[1024,4096] all-reduce(bf16[1024,4096] %x),"
                 " replica_groups=[16,4]<=[64]"),
                "provenance":
                "jit(train_step)/mlp/wo",
            },
            {
                "operation": "dot.12",
                "category": "convolution",
                "raw_time_ps": 350_000_000.0,
                "occurrences": 32,
                "expression": "%dot.12 = bf16[1024,4096] dot(...)",
                "provenance": "jit(train_step)/mlp/wi",
            },
        ]

    def test_build_portfolio_returns_diverse_unique_proposals(self):
        baseline = {
            "data": 1,
            "fsdp": 8,
            "tensor": 4,
            "sequence": 1,
            "expert": 2
        }
        ops = self._sample_ops()
        bd = collective_breakdown.summarize(ops, baseline)
        portfolio = mesh_portfolio.build_portfolio(
            baseline,
            bd,
            topology="4x4x4",
            batch_size=128,
            sequence_length=8192,
            num_heads=32,
            num_kv_heads=8,
        )
        proposals = portfolio["proposals"]
        self.assertGreaterEqual(len(proposals), 3)
        self.assertLessEqual(len(proposals), 7)
        seen_meshes = set()
        for prop in proposals:
            key = json.dumps(prop["mesh"], sort_keys=True)
            self.assertNotIn(key, seen_meshes)
            seen_meshes.add(key)
        # At least one proposal should shrink the dominant fsdp=8 collective axis.
        self.assertTrue(any(p["mesh"]["fsdp"] < 8 for p in proposals))
        md = mesh_portfolio.format_portfolio_markdown(portfolio)
        self.assertIn("fsdp", md)

    def test_torus_mapping_names_wraparound_only_where_the_slice_has_it(self):
        v6e_16 = mesh_portfolio.torus_mapping({"fsdp": 4, "tensor": 4}, "4x4")
        self.assertIn("4-long X ring (no wrap-around links on this slice)",
                      v6e_16)
        self.assertNotIn("(wrap-around links)", v6e_16)
        cube = mesh_portfolio.torus_mapping(
            {
                "fsdp": 4,
                "tensor": 4,
                "expert": 4
            }, "4x4x4")
        self.assertIn("4-long X ring (wrap-around links)", cube)
        long_axis = mesh_portfolio.torus_mapping({
            "data": 8,
            "tensor": 16
        }, "8x16")
        self.assertIn("8-long X ring (no wrap-around links on this slice)",
                      long_axis)
        self.assertIn("16-long Y ring (wrap-around links)", long_axis)

    def test_rank_mesh_proposals_guards_serving_disaggregation_axes(self):
        with self.assertRaisesRegex(ValueError, "validate-model"):
            rank_mesh_proposals.rank_mesh_proposals(
                self._sample_ops(),
                {
                    "tp": 4,
                    "dp": 4
                },
            )

    def test_rank_mesh_proposals_json_and_table_formats(self):
        res = rank_mesh_proposals.rank_mesh_proposals(
            self._sample_ops(),
            {
                "fsdp": 8,
                "tensor": 4,
                "expert": 2
            },
            topology="4x4x4",
            batch_size=128,
            sequence_length=4096,
            num_heads=32,
            num_kv_heads=8,
        )
        as_json = json.loads(rank_mesh_proposals.format_result(res, "json"))
        self.assertIn("proposals", as_json)
        self.assertGreater(len(as_json["proposals"]), 0)
        self.assertEqual(as_json["estimate_type"], "analytical_volume_ranking")
        assumptions = as_json["model_assumptions"]
        self.assertEqual(assumptions["shared_torus_dim_penalty"], 2.0)
        self.assertIn("L / 4", assumptions["all_to_all_ring_mean_hops"])
        self.assertEqual(assumptions["memory_caution_ratio"], 4.0)
        self.assertEqual(assumptions["memory_risk_ratio"], 8.0)
        self.assertEqual(assumptions["oom_hbm_fraction"], 0.95)
        self.assertIn("exceeds max(oom_hbm_fraction of HBM",
                      assumptions["memory_rule"])
        self.assertIn("only an upper bound", assumptions["memory_rule"])
        as_md = rank_mesh_proposals.format_result(res, "table")
        self.assertIn(
            "Model assumptions (uncalibrated ranking priors, not predictions):",
            as_md,
        )
        self.assertIn("exceeds max(95% of HBM, the measured peak)", as_md)
        self.assertIn(">=8x the baseline local tokens", as_md)


if __name__ == "__main__":
    unittest.main()
