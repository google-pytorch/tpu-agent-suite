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
"""Unit tests for roofline_data.py, on a layer CSV roofline.py writes now."""

from __future__ import annotations

import csv
import importlib.util
import pathlib
import subprocess
import sys
import tempfile
import unittest

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
CONFIG = _SCRIPTS_DIR / "configs" / "glm-5.2-fp8-8k1k-2x2x2.json"

try:
    from . import roofline_data  # noqa: E402
except ImportError:
    _spec = importlib.util.spec_from_file_location(
        "roofline_data", _SCRIPTS_DIR / "roofline_data.py")
    roofline_data = importlib.util.module_from_spec(_spec)
    sys.modules["roofline_data"] = roofline_data
    _spec.loader.exec_module(roofline_data)
roofline = roofline_data.roofline


class TestRooflineData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.csv = pathlib.Path(cls.tmp.name) / "layers.csv"
        subprocess.run([
            sys.executable,
            str(_SCRIPTS_DIR / "roofline.py"),
            str(CONFIG), "--layer-csv",
            str(cls.csv)
        ],
                       check=True,
                       capture_output=True)
        with open(cls.csv, newline="") as f:
            cls.csv_rows = list(csv.DictReader(f))
        cls.data = roofline_data.build([(str(CONFIG), str(cls.csv))])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_rows_reproduce_each_phase_total(self):
        """Summing counted rows (not sub, not total) gives the CSV's TOTALs."""
        for regime in ("prefill", "decode"):
            with self.subTest(regime=regime):
                want = sum(
                    float(r["Layer Total (ms)"]) for r in self.csv_rows
                    if r["Phase"] == regime and r["Role"] == "total")
                rows = [r for r in self.data["rows"] if r["regime"] == regime]
                counted = sum(r["layer_total_ms"] for r in rows
                              if r["role"] in ("op", "fused"))
                self.assertGreater(want, 0)
                self.assertAlmostEqual(counted, want, places=6)

    def test_sub_rows_name_their_fused_parent(self):
        qk_rope = next(
            r for r in self.data["rows"]
            if r["regime"] == "decode" and r["layer_kind"] == "moe_idx"
            and r["operator"] == "MLA: QK Rope")
        self.assertEqual(qk_rope["role"], "sub")
        self.assertEqual(qk_rope["parent"], "MLA: Flash Attention QK+AV")
        self.assertEqual(qk_rope["count_per_step"], 18)
        self.assertIn(qk_rope["bound_by"], ("HBM", "MXU", "VPU", "collective"))
        self.assertTrue(
            all(r["parent"] is None for r in self.data["rows"]
                if r["role"] != "sub"))

    def test_collectives_are_bound_by_collective(self):
        bound = {r["bound_by"] for r in self.data["rows"]}
        self.assertIn("collective", bound)
        self.assertNotIn("ICI", bound)

    def test_layer_type_map_covers_every_layer_once(self):
        kinds = self.data["layer_types"]
        self.assertEqual(kinds["dense_idx"]["layers"], [0, 1, 2])
        self.assertEqual(kinds["moe_idx"]["layers"][:3], [6, 10, 14])
        self.assertEqual({
            k: v["count"]
            for k, v in kinds.items()
        }, {
            "dense_idx": 3,
            "moe_idx": 18,
            "moe": 57
        })
        every = sorted(i for v in kinds.values() for i in v["layers"])
        self.assertEqual(every, list(range(78)))

    def test_regime_carries_its_operating_point(self):
        prefill = self.data["regimes"]["prefill"]
        self.assertEqual((prefill["tokens"], prefill["kv_len"],
                          prefill["batch"], prefill["passes"]),
                         (8192, 8192, 1, 8))

    def test_every_mxu_row_carries_a_shape_matching_its_flops(self):
        cfg = roofline.load(str(CONFIG))
        for name, ph in ((p.name, p) for p in cfg.phases):
            sub = roofline.chunks_of(ph)[0]
            ops = {o.name: o for o in roofline.build_ops(cfg, sub)}
            mxu = [
                r for r in self.data["rows"]
                if r["regime"] == name and r["role"] in (
                    "op", "sub") and ops[r["operator"]].kind == "compute"
                and ops[r["operator"]].unit == "mxu"
            ]
            self.assertGreater(len(mxu), 10)
            for r in mxu:
                with self.subTest(regime=name, op=r["operator"]):
                    s = r["shape"]
                    self.assertIsNotNone(s, r["comment"])
                    k = max(s["k"], cfg.hardware.mxu_min_dim)
                    self.assertAlmostEqual(
                        2 * s["batch"] * s["m"] * k * s["n"] * s["n_mat"],
                        ops[r["operator"]].flops,
                        delta=1e-9 * ops[r["operator"]].flops)
                    self.assertTrue(s["dtype"])

    def test_floor_per_pass_times_passes_is_the_floor(self):
        prefill = [r for r in self.data["rows"] if r["regime"] == "prefill"]
        self.assertTrue(all(r["passes"] == 8 for r in prefill))
        for r in prefill:
            self.assertAlmostEqual(r["floor_ms_per_pass"] * r["passes"],
                                   r["floor_ms"],
                                   places=9)
        self.assertTrue(
            all(r["passes"] == 1 for r in self.data["rows"]
                if r["regime"] == "decode"))

    def test_collectives_have_no_shape_and_say_why(self):
        colls = [
            r for r in self.data["rows"]
            if r["bound_by"] == "collective" and r["role"] == "op"
        ]
        self.assertTrue(colls)
        for r in colls:
            self.assertIsNone(r["shape"])
            self.assertIn("shape", r["comment"])

    def test_a_csv_from_another_model_is_refused(self):
        bad = pathlib.Path(self.tmp.name) / "bad.csv"
        rows = [dict(r) for r in self.csv_rows]
        for r in rows:
            if r["Layer Kind"] == "moe":
                r["Layer Count"] = "56"
        with open(bad, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        with self.assertRaises(ValueError):
            roofline_data.build([(str(CONFIG), str(bad))])


if __name__ == "__main__":
    unittest.main()
