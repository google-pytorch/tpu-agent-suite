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
"""Unit tests for `mesh_topology.py` and `tpu_specs.py`."""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mesh_topology  # noqa: E402
import tpu_specs  # noqa: E402


class MeshTopologyTest(unittest.TestCase):
    """Tests TPU hardware specs, physical torus fit, and target slice onboarding."""

    def test_public_platform_specs_and_devices_per_chip(self):
        for platform in ("v4", "v5e", "v5p", "v6e", "tpu7x"):
            spec = tpu_specs.get_platform_spec(platform)
            self.assertIsNotNone(spec)
            self.assertGreater(spec["hbm_gib"], 0)
            self.assertIn(spec["topology_dim"], (2, 3))
        self.assertEqual(tpu_specs.get_devices_per_chip("v6e"), 1)
        self.assertEqual(tpu_specs.get_devices_per_chip("v5p"), 1)
        self.assertEqual(tpu_specs.get_devices_per_chip("tpu7x"), 2)

    def test_check_mesh_topology_fit_whole_axes_vs_split(self):
        aligned = mesh_topology.check_mesh_topology_fit(
            {
                "fsdp": 4,
                "tensor": 4,
                "expert": 4
            }, "4x4x4", cores_per_chip=1)
        self.assertTrue(aligned["device_count_matches"])
        self.assertTrue(aligned["fits_without_split"])
        self.assertFalse(aligned["allow_split_physical_axes"])

        split_needed = mesh_topology.check_mesh_topology_fit(
            {
                "fsdp": 8,
                "tensor": 8
            }, "4x4x4", cores_per_chip=1)
        self.assertTrue(split_needed["device_count_matches"])
        self.assertFalse(split_needed["fits_without_split"])
        self.assertTrue(split_needed["allow_split_physical_axes"])

    def test_check_mesh_topology_fit_dual_core_tpu7x(self):
        # 8 chips of tpu7x = 16 logical TensorCore devices (2x2x2 chips * 2 cores/chip).
        fit = mesh_topology.check_mesh_topology_fit({
            "fsdp": 4,
            "tensor": 4
        },
                                                    "2x2x2",
                                                    cores_per_chip=2)
        self.assertTrue(fit["device_count_matches"])
        self.assertTrue(fit["buildable"])

    def test_decide_allow_split_physical_axes_overrides_false_when_required(
            self):
        val, reason = mesh_topology.decide_allow_split_physical_axes(
            {
                "fsdp": 8,
                "tensor": 8
            },
            "4x4x4",
            cores_per_chip=1,
            explicit=False)
        self.assertTrue(val)
        self.assertIn("allow_split_physical_axes", reason)

    def test_plan_target_topology_v5p_to_tpu7x_and_v6e(self):
        plan_3d = mesh_topology.plan_target_topology(
            source_chips=64,
            source_platform="v5p",
            target_platform="tpu7x",
            total_model_memory_per_chip_gib=70.0,
        )
        self.assertEqual(plan_3d.source_dim, 3)
        self.assertEqual(plan_3d.target_dim, 3)
        self.assertGreaterEqual(plan_3d.recommended_chips, 1)
        self.assertEqual(len(plan_3d.recommended_topology.split("x")), 3)
        self.assertIn("Topology Onboarding Plan", str(plan_3d))

        plan_2d = mesh_topology.plan_target_topology(
            source_chips=16,
            source_platform="v5e",
            target_platform="v6e",
            total_model_memory_per_chip_gib=12.0,
        )
        self.assertEqual(plan_2d.target_dim, 2)
        self.assertEqual(len(plan_2d.recommended_topology.split("x")), 2)

    def test_optimal_2d_and_3d_topology_helpers(self):
        self.assertEqual(mesh_topology.get_optimal_2d_topology(16), "4x4")
        self.assertEqual(mesh_topology.get_optimal_3d_topology(64), "4x4x4")


if __name__ == "__main__":
    unittest.main()
