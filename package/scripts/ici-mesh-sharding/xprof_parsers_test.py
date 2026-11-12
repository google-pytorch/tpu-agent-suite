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
"""Unit tests for `xprof_parsers.py`."""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import xprof_parsers  # noqa: E402

_GIB = float(1 << 30)


class XprofParsersTest(unittest.TestCase):
    """Tests pure-stdlib XProf JSON parsers and serving HBM reservation suppression."""

    def test_parse_hlo_op_profile_nested_tree_and_flat_list(self):
        nested = {
            "byCategory": {
                "name":
                "root",
                "children": [{
                    "name":
                    "all-gather",
                    "metrics": {
                        "occurrences": 1,
                        "rawTime": 500.0
                    },
                    "xla": {
                        "category": "all-gather"
                    },
                    "children": [{
                        "name": "all-gather.42",
                        "metrics": {
                            "occurrences": 16,
                            "rawTime": 400_000_000.0,
                            "avgTimePs": 25_000_000.0,
                        },
                        "xla": {
                            "category": "all-gather",
                            "expression":
                            "%all-gather.42 = bf16[8192,4096] all-gather(...)",
                            "provenance": "jit(train_step)/fsdp",
                        },
                    }],
                }],
            }
        }
        ops = xprof_parsers.parse_hlo_op_profile(nested)
        self.assertTrue(any(o["operation"] == "all-gather.42" for o in ops))

        flat = [{
            "name": "all-reduce.1",
            "category": "all-reduce",
            "self_time_us": 120.0
        }]
        flat_ops = xprof_parsers.parse_hlo_op_profile(flat)
        self.assertEqual(len(flat_ops), 1)
        self.assertAlmostEqual(flat_ops[0]["raw_time_ps"], 120.0 * 1e6)

    def test_summarize_memory_headroom_training_vs_serving_kv_reservation(
            self):
        mem_profile = {
            "memoryProfilePerAllocator": {
                "TPU_HBM": {
                    "profileSummary": {
                        "memoryCapacity": 95 * _GIB,
                        "peakStats": {
                            "peakBytesInUse": 90 * _GIB,
                            "heapAllocatedBytes": 88 * _GIB,
                            "stackReservedBytes": 2 * _GIB,
                        },
                    }
                }
            }
        }
        train_viewer = {
            "moduleName": "jit_train_step",
            "totalBufferAllocationMib": 70000.0,
            "entryComputationParametersMib": 24000.0,
        }
        train_headroom = xprof_parsers.summarize_memory_headroom(
            mem_profile, memory_viewer_module=train_viewer)
        self.assertFalse(train_headroom["is_serving"])
        self.assertIsNotNone(train_headroom["peak_fraction"])
        self.assertGreater(train_headroom["peak_fraction"], 0.9)

        serve_viewer = {
            "moduleName": "vllm_paged_attention_step",
            "totalBufferAllocationMib": 90000.0,
            "entryComputationParametersMib": 18000.0,
        }
        serve_headroom = xprof_parsers.summarize_memory_headroom(
            mem_profile, memory_viewer_module=serve_viewer)
        self.assertTrue(serve_headroom["is_serving"])
        # Static KV reservation peak must not be reported as training peak_fraction.
        self.assertIsNone(serve_headroom["peak_fraction"])

    def test_summarize_dcn_stats_from_megascale_and_overview(self):
        megascale = {
            "cols": [
                {
                    "id": "rendezvous_name"
                },
                {
                    "id": "transfer_type"
                },
                {
                    "id": "occurrences"
                },
                {
                    "id": "observed_duration"
                },
                {
                    "id": "stall_duration"
                },
                {
                    "id": "slack_time"
                },
                {
                    "id": "net_tx_bytes"
                },
            ],
            "rows": [{
                "c": [
                    {
                        "v": "replica_all_reduce_0"
                    },
                    {
                        "v": "ALL_REDUCE"
                    },
                    {
                        "v": 10
                    },
                    {
                        "v": 25.0
                    },
                    {
                        "v": 15.0
                    },
                    {
                        "v": 2.0
                    },
                    {
                        "v": 104857600
                    },
                ]
            }],
        }
        overview = {"step_time_ms": 100.0, "num_steps": 10}
        dcn = xprof_parsers.summarize_dcn_stats(megascale,
                                                overview_page=overview,
                                                num_hosts=1)
        self.assertTrue(dcn["available"])
        self.assertEqual(dcn["num_collectives"], 1)
        self.assertAlmostEqual(dcn["dcn_pct"], 15.0, places=1)


if __name__ == "__main__":
    unittest.main()
