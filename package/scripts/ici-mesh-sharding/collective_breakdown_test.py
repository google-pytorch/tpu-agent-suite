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
"""Unit tests for `collective_breakdown.py`."""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import collective_breakdown  # noqa: E402


class CollectiveBreakdownTest(unittest.TestCase):
    """Tests 4-tier collective axis attribution and role classification."""

    def test_mesh_axes_metadata_takes_highest_precedence(self):
        mesh = {"fsdp": 4, "sequence": 4, "tensor": 2}
        ops = [
            {
                "operation":
                "custom-call.10",
                "category":
                "all-gather",
                "raw_time_ps":
                500_000_000.0,
                "occurrences":
                10,
                "expression":
                ("%custom-call.10 = bf16[4,8192,128] all-gather(bf16[4,2048,128] %kv),"
                 " backend_config={\"mesh_axes\": [\"sequence\"]}"),
                "provenance":
                "jit(train_step)/self_attn/splash_attention",
            },
            {
                "operation": "fusion.1",
                "category": "convolution",
                "raw_time_ps": 500_000_000.0,
                "occurrences": 10,
                "expression": "%fusion.1 = bf16[4,8192,4096] dot(...)",
                "provenance": "jit(train_step)/mlp/wi",
            },
        ]
        summary = collective_breakdown.summarize(ops, mesh)
        self.assertAlmostEqual(summary["collective_pct"], 50.0, places=1)
        self.assertIn("sequence", summary["per_axis"])
        self.assertNotIn("fsdp", summary["per_axis"])

    def test_iota_explicit_and_global_reduction_attribution(self):
        mesh = {"fsdp": 8, "tensor": 4, "expert": 2}
        ops = [
            {
                "operation":
                "all-gather.1",
                "category":
                "all-gather",
                "raw_time_ps":
                400_000_000.0,
                "occurrences":
                32,
                "expression":
                ("%all-gather.1 = bf16[8192,4096] all-gather(bf16[1024,4096] %p),"
                 " replica_groups=[8,8]<=[64]"),
                "provenance":
                "jit(train_step)/remat/params/fsdp_gather",
            },
            {
                "operation":
                "all-to-all.2",
                "category":
                "all-to-all",
                "raw_time_ps":
                200_000_000.0,
                "occurrences":
                32,
                "expression":
                ("%all-to-all.2 = bf16[1024,4096] all-to-all(bf16[1024,4096] %x),"
                 " replica_groups={{0,1},{2,3}}"),
                "provenance":
                "jit(train_step)/moe/routed_experts",
            },
            {
                "operation":
                "all-reduce.3",
                "category":
                "all-reduce",
                "raw_time_ps":
                100_000_000.0,
                "occurrences":
                1,
                "expression": ("%all-reduce.3 = f32[] all-reduce(f32[] %loss),"
                               " replica_groups=[1,64]<=[64]"),
                "provenance":
                "jit(train_step)/grad_norm",
            },
            {
                "operation": "dot.4",
                "category": "convolution",
                "raw_time_ps": 300_000_000.0,
                "occurrences": 32,
                "expression": "%dot.4 = bf16[1024,4096] dot(...)",
                "provenance": "jit(train_step)/mlp/wo",
            },
        ]
        summary = collective_breakdown.summarize(ops, mesh)
        self.assertAlmostEqual(summary["collective_pct"], 70.0, places=1)
        self.assertAlmostEqual(summary["per_axis"]["fsdp"], 40.0, places=1)
        self.assertAlmostEqual(summary["per_axis"]["expert"], 20.0, places=1)
        self.assertEqual(summary["largest_reducible"]["axes"], "fsdp")
        md = collective_breakdown.format_markdown(summary)
        self.assertIn("fsdp", md)
        self.assertIn("expert", md)

    def test_ambiguous_equal_size_axes_are_not_guessed(self):
        mesh = {"fsdp": 4, "tensor": 4}
        ops = [{
            "operation":
            "all-reduce.1",
            "category":
            "all-reduce",
            "raw_time_ps":
            300_000_000.0,
            "occurrences":
            16,
            "expression":
            ("%all-reduce.1 = bf16[1024,4096] all-reduce(bf16[1024,4096] %x),"
             " replica_groups=[4,4]<=[16]"),
            "provenance":
            "jit(train_step)/layer/block",
        }]
        summary = collective_breakdown.summarize(ops, mesh)
        group_axes = summary["groups"][0]["axes"]
        self.assertIn("fsdp", group_axes)
        self.assertIn("tensor", group_axes)

    def test_shape_factor_fallback_when_replica_groups_elided(self):
        mesh = {"fsdp": 8, "tensor": 2}
        ops = [{
            "operation":
            "all-gather.9",
            "category":
            "all-gather",
            "raw_time_ps":
            250_000_000.0,
            "occurrences":
            8,
            "expression":
            ("%all-gather.9 = bf16[8192,4096] all-gather(bf16[1024,4096] %w),"
             " dimensions={0}"),
            "provenance":
            "jit(train_step)/weights",
        }]
        summary = collective_breakdown.summarize(ops, mesh)
        self.assertEqual(summary["groups"][0]["axes"], "fsdp")
        self.assertIn("x8", summary["groups"][0]["evidence"])


if __name__ == "__main__":
    unittest.main()
