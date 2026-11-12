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
"""Unit tests for `mesh_search.py`."""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mesh_search  # noqa: E402


class MeshSearchTest(unittest.TestCase):
    """Tests pairwise power-of-two factor moves, legality checks, and torus fit."""

    def test_parse_mesh_json_and_kv(self):
        self.assertEqual(
            mesh_search.parse_mesh('{"fsdp": 8, "tensor": 4, "expert": 2}'),
            {
                "fsdp": 8,
                "tensor": 4,
                "expert": 2
            },
        )
        self.assertEqual(
            mesh_search.parse_mesh("fsdp=8,tensor=4,expert=2"),
            {
                "fsdp": 8,
                "tensor": 4,
                "expert": 2
            },
        )

    def test_parse_topology_2d_and_3d(self):
        self.assertEqual(mesh_search.parse_topology("4x4"), [4, 4])
        self.assertEqual(mesh_search.parse_topology("4x4x4"), [4, 4, 4])
        with self.assertRaises(ValueError):
            mesh_search.parse_topology("invalid")

    def test_enumerate_preserves_device_product_and_powers_of_two(self):
        baseline = {
            "data": 1,
            "fsdp": 8,
            "tensor": 4,
            "sequence": 1,
            "expert": 2
        }
        res = mesh_search.enumerate_mesh_candidates(
            baseline,
            batch_size=128,
            sequence_length=8192,
            num_heads=32,
            num_kv_heads=8,
            topology="4x4x4",
        )
        self.assertGreater(res["num_candidates"], 0)
        for cand in res["candidates"]:
            mesh = cand["mesh"]
            product = 1
            for size in mesh.values():
                self.assertGreaterEqual(size, 1)
                self.assertEqual(size & (size - 1), 0)
                product *= size
            self.assertEqual(product, 64)

    def test_head_and_batch_divisibility_filters_illegal_moves(self):
        baseline = {"fsdp": 8, "tensor": 4}
        res = mesh_search.enumerate_mesh_candidates(
            baseline,
            batch_size=8,
            sequence_length=4096,
            num_heads=8,
            num_kv_heads=8,
        )
        for cand in res["candidates"]:
            mesh = cand["mesh"]
            self.assertLessEqual(mesh["tensor"], 8)
            self.assertEqual(8 % mesh["tensor"], 0)
            self.assertEqual(8 % mesh["fsdp"], 0)

    def test_kv_head_replication_warning_when_tensor_exceeds_kv_heads(self):
        baseline = {"fsdp": 8, "tensor": 4}
        res = mesh_search.enumerate_mesh_candidates(
            baseline,
            batch_size=64,
            sequence_length=4096,
            num_heads=32,
            num_kv_heads=4,
        )
        tp8 = [c for c in res["candidates"] if c["mesh"]["tensor"] == 8]
        self.assertTrue(tp8)
        self.assertTrue(
            any("KV" in w or "kv" in w.lower()
                for w in tp8[0].get("warnings", []))
            or "KV" in tp8[0].get("legality", ""))

    def test_sequence_and_context_are_sequence_axes(self):
        for axis in ("sequence", "context"):
            with self.subTest(axis=axis):
                self.assertEqual(mesh_search.axis_family(axis), "sequence")
                res = mesh_search.enumerate_mesh_candidates(
                    {
                        "fsdp": 8,
                        axis: 2
                    },
                    sequence_length=8,
                )
                self.assertTrue(res["candidates"])
                for cand in res["candidates"]:
                    self.assertEqual(8 % cand["mesh"][axis], 0)

    def test_fixed_axes_remain_unchanged(self):
        baseline = {"replica": 2, "fsdp": 8, "tensor": 4}
        res = mesh_search.enumerate_mesh_candidates(
            baseline,
            fixed_axes=["replica"],
            batch_size=64,
            num_heads=32,
        )
        for cand in res["candidates"]:
            self.assertEqual(cand["mesh"]["replica"], 2)

    def test_torus_fit_and_table_formatting(self):
        fit_aligned = mesh_search.torus_fit({"fsdp": 4, "tensor": 4}, [4, 4])
        self.assertEqual(fit_aligned["shared"], [])
        fit_split = mesh_search.torus_fit({"fsdp": 2, "tensor": 8}, [4, 4])
        self.assertTrue(fit_split["shared"])

        res = mesh_search.enumerate_mesh_candidates(
            {
                "fsdp": 4,
                "tensor": 4
            },
            batch_size=32,
            sequence_length=2048,
            num_heads=16,
            topology="4x4",
        )
        table = mesh_search.format_candidates_table(res)
        self.assertIn("fsdp", table)
        self.assertIn("tensor", table)

    def test_wraparound_dims_follow_the_documented_rule(self):
        # 2D (v5e, v6e): only a 16-chip dimension wraps.
        self.assertEqual(mesh_search.wraparound_dims([4, 4]), (False, False))
        self.assertEqual(mesh_search.wraparound_dims([8, 16]), (False, True))
        self.assertEqual(mesh_search.wraparound_dims([16, 16]), (True, True))
        # 3D: whole 4x4x4 cubes wrap on every dimension, smaller slices do not.
        self.assertEqual(mesh_search.wraparound_dims([4, 4, 8]), (True, ) * 3)
        self.assertEqual(mesh_search.wraparound_dims([2, 4, 4]), (False, ) * 3)


if __name__ == "__main__":
    unittest.main()
