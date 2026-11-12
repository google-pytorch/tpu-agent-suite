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
"""Unit tests for `oss_sharding_rules.py` and `sharding_adapter.py`."""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import oss_sharding_rules  # noqa: E402
import sharding_adapter  # noqa: E402


class OssShardingRulesTest(unittest.TestCase):
    """Tests OSS/serving sharding rules, MLA detection, derived EP/DCP, and HBM fit."""

    def test_power_of_two_validation(self):
        self.assertTrue(oss_sharding_rules.is_power_of_two(1))
        self.assertTrue(oss_sharding_rules.is_power_of_two(8))
        self.assertFalse(oss_sharding_rules.is_power_of_two(0))
        self.assertFalse(oss_sharding_rules.is_power_of_two(6))

        ok, errors, _ = oss_sharding_rules.validate_oss_model_sharding(
            {
                "tp": 6,
                "dp": 1
            }, {"num_attention_heads": 24}, total_tpu_chips=6)
        self.assertFalse(ok)
        self.assertTrue(any("power of 2" in e.lower() for e in errors))

    def test_head_divisibility_and_kv_replication_warning(self):
        cfg = {
            "num_attention_heads": 32,
            "num_key_value_heads": 6,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "accelerator_type": "v6e",
        }
        # TP=8 > num_key_value_heads=6 (non-divisible) -> warning about KV replication,
        # while query heads (32) remain divisible by TP=8.
        ok, errors, warnings = oss_sharding_rules.validate_oss_model_sharding(
            {
                "tp": 8,
                "dp": 1
            }, cfg, total_tpu_chips=8)
        self.assertTrue(ok, msg=f"Unexpected errors: {errors}")
        self.assertTrue(
            any("kv" in w.lower() or "replicat" in w.lower()
                for w in warnings),
            msg=f"Expected KV replication warning, got {warnings}",
        )

    def test_mla_detection_and_non_tp_kv_sharding_hbm(self):
        mla_cfg = {
            "num_attention_heads": 128,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "vocab_size": 129280,
            "n_routed_experts": 256,
            "moe_intermediate_size": 2048,
            "max_model_len": 32768,
            "max_num_seqs": 128,
            "dtype": "fp8",
            "mode": "inference",
            "runtime_engine": "vllm",
        }
        self.assertTrue(
            sharding_adapter.ShardingNomenclatureAdapter.is_mla_model(mla_cfg))
        # On a tiny single chip (v5e, 16 GiB), huge MLA model must report HBM overflow.
        hbm_errors = oss_sharding_rules.estimate_hbm_capacity(
            {
                "tp": 1,
                "dp": 1
            }, mla_cfg, tpu_type="v5e")
        self.assertTrue(hbm_errors)

    def test_vllm_derived_ep_and_dcp_rank_reuse(self):
        adapter = sharding_adapter.ShardingNomenclatureAdapter
        # vLLM DCP reuses TP ranks: TP=4, DP=2, DCP=2 -> world size is 4*2 = 8 (not 16).
        proposal = {
            "tp": 4,
            "dp": 2,
            "pcp": 1,
            "dcp": 2,
            "pp": 1,
            "enable_expert_parallel": 1,
        }
        flags = adapter.to_vllm_flags(proposal)
        self.assertIn("--tensor-parallel-size", flags["cli_string"])
        self.assertIsNone(
            adapter.vllm_dcp_error(
                adapter.standardize_proposal({
                    "tp": 4,
                    "dcp": 2
                })))
        self.assertIsNotNone(
            adapter.vllm_dcp_error(
                adapter.standardize_proposal({
                    "tp": 2,
                    "dcp": 4
                })))

    def test_roofline_ridge_point_and_bottleneck_classification(self):
        ridge_v6e = sharding_adapter.get_roofline_ridge_point("v6e")
        self.assertGreater(ridge_v6e, 100.0)
        self.assertEqual(
            sharding_adapter.classify_bottleneck(1e9, 1e9, "v6e"),
            "memory_bound",
        )
        self.assertEqual(
            sharding_adapter.classify_bottleneck(1e15, 1e9, "v6e"),
            "compute_bound",
        )


if __name__ == "__main__":
    unittest.main()
