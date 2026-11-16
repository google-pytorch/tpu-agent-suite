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
"""CPU tests for generic/model.py and core/derive.py's fallback to it.

Fixtures are the checkpoints' HF config.json files (huggingface.co/Qwen/<model>/raw/main).
Run: `python3 -m unittest discover -s generic/tests` from scripts/workload-microbenchmark/.
"""

import json
import os
import sys
import unittest
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, SCRIPTS)

from core import derive  # noqa: E402
from core.deploy import Deploy, per_rank  # noqa: E402


def _load(p):
    with open(p) as fh:
        return json.load(fh)


def _fixture(name):
    return _load(os.path.join(HERE, "fixtures", name))


def _ckpt(model_dir):
    import glob
    p = glob.glob(os.path.join(SCRIPTS, model_dir, "config",
                               "*.source.json"))[0]
    return _load(p)["checkpoint_config"]


G = derive.load_model("generic")
DENSE = _fixture("qwen3-32b.config.json")
MOE = _fixture("qwen3-30b-a3b.config.json")


def _rows(mod, f, d, pool=None):
    return per_rank(f, d, pool, mod.per_rank_rows,
                    mod.kv_bytes_per_token_per_rank)


class Qwen3Dense(unittest.TestCase):

    def test_facts(self):
        f = G.facts(DENSE, "auto")
        self.assertEqual(
            (f.layers, f.hidden, f.heads, f.kv_heads, f.dense_ffn, f.vocab),
            (64, 5120, 64, 8, 25600, 151936))
        self.assertEqual((f.experts, f.mla, f.layer_kinds), (0, False, {
            "dense": 64
        }))
        # K + V x 8 heads x 128 x bf16
        self.assertEqual(f.kv_cache_bytes_per_token_per_layer, 4096)
        self.assertEqual(f.kv_bytes_token, 4096 * 64)
        self.assertEqual(
            G.facts(DENSE, "fp8").kv_cache_bytes_per_token_per_layer, 2048)

    def test_per_rank_tp8_and_kv_replication_past_kv_heads(self):
        f = G.facts(DENSE, "auto")
        r8 = _rows(G, f, Deploy(tp=8))
        self.assertEqual(r8["q_heads_projected"], 8)
        self.assertEqual(r8["dense_ffn_per_rank"], 3200)
        self.assertEqual(r8["attn_kv_heads_per_rank"], 1)
        self.assertFalse(r8["attn_kv_replicated"])
        self.assertEqual(r8["kv_bytes_per_token_per_rank"], 32768)
        r16 = _rows(G, f, Deploy(tp=16))
        self.assertEqual(r16["attn_kv_heads_per_rank"], 1)
        self.assertTrue(r16["attn_kv_replicated"])
        self.assertEqual(r16["kv_bytes_per_token_per_rank"], 32768)
        self.assertEqual(G.feasibility_checks(f, Deploy(tp=16)), [])
        self.assertTrue(G.feasibility_checks(f, Deploy(tp=3)))

    def _twin_matches(self, f, real, world):
        t = G.twin_overrides(f, real, world)
        ft = G.twin_facts(f, t["hf_overrides"])
        sv = t["server_args"]
        dt = replace(real,
                     tp=sv["tensor-parallel-size"],
                     dp=sv["data-parallel-size"],
                     dcp=sv["decode-context-parallel-size"])
        pr, prt = _rows(G, f, real), _rows(G, ft, dt)
        for k in ("q_heads_projected", "dense_ffn_per_rank",
                  "experts_per_rank", "shared_expert_ffn_per_rank",
                  *G.PROJECTION_ROWS):
            if k in pr:
                self.assertEqual(pr[k], prt[k], k)
        self.assertEqual(f.topk / real.ep_group, ft.topk / dt.ep_group)
        return t, ft

    def test_twin_mirror_keeps_sizes_and_one_layer(self):
        f = G.facts(DENSE, "auto")
        t, ft = self._twin_matches(f, Deploy(tp=8), 8)
        self.assertEqual(t["hf_overrides"], {"num_hidden_layers": 1})
        self.assertEqual(t["layers"], ["dense"])
        self.assertEqual(ft.layer_kinds, {"dense": 1})

    def test_twin_scaled_keeps_per_rank_sizes(self):
        f = G.facts(DENSE, "auto")
        t, ft = self._twin_matches(f, Deploy(tp=16), 8)
        self.assertEqual(t["server_args"]["tensor-parallel-size"], 8)
        self.assertEqual((ft.heads, ft.kv_heads, ft.dense_ffn, ft.layers),
                         (32, 4, 12800, 1))
        self.assertEqual(t["problems"], [])


class Qwen3Moe(unittest.TestCase):

    def test_facts_and_ep(self):
        f = G.facts(MOE, "auto")
        self.assertEqual(
            (f.experts, f.topk, f.expert_ffn, f.shared_ffn, f.dense_ffn),
            (128, 8, 768, 0, 0))
        self.assertEqual(f.layer_kinds, {"moe": 48})
        r = _rows(G, f, Deploy(tp=4, ep=True))
        self.assertEqual(r["experts_per_rank"], 32)
        self.assertEqual(r["attn_kv_heads_per_rank"], 1)

    def test_mixed_kinds_twin_has_one_layer_per_kind(self):
        f = G.facts({**MOE, "mlp_only_layers": [0, 1]}, "auto")
        self.assertEqual(f.layer_kinds, {"dense": 2, "moe": 46})
        self.assertEqual(f.dense_ffn, 6144)
        t, ft = Qwen3Dense._twin_matches(self, f, Deploy(tp=8, ep=True), 8)
        self.assertEqual(t["layers"], ["dense", "moe"])
        self.assertEqual(t["hf_overrides"]["mlp_only_layers"], [0])
        self.assertEqual(ft.kinds, ("dense", "moe"))

    def test_scaled_moe_scales_experts_with_topk(self):
        f = G.facts(MOE, "auto")
        t, ft = Qwen3Dense._twin_matches(self, f, Deploy(tp=16, ep=True), 8)
        self.assertEqual((ft.experts, ft.topk), (64, 4))


class Classify(unittest.TestCase):

    def test_standard(self):
        self.assertEqual(G.classify(DENSE), [])
        self.assertEqual(G.classify(MOE), [])

    def test_non_standard_named(self):
        q35 = " ".join(G.classify(_ckpt("Qwen3.5-397B-A17B-FP8")))
        for want in ("linear_attention", "MTP", "multimodal"):
            self.assertIn(want, q35)
        glm = " ".join(G.classify(_ckpt("GLM-5.2-FP8")))
        for want in ("MLA", "indexer", "MTP"):
            self.assertIn(want, glm)
        self.assertIn(
            "sliding", " ".join(
                G.classify({
                    **DENSE, "sliding_window": 4096,
                    "use_sliding_window": True
                })))
        with self.assertRaises(ValueError):
            G.facts(_ckpt("GLM-5.2-FP8"))


class Fallback(unittest.TestCase):

    def test_missing_adapter_falls_back_to_generic(self):
        mod = derive.load_model("Qwen3-32B", DENSE)
        self.assertTrue(mod.__file__.endswith("generic/model.py"))

    def test_non_standard_without_adapter_names_why(self):
        with self.assertRaises(FileNotFoundError) as e:
            derive.load_model("GLM-9", _ckpt("GLM-5.2-FP8"))
        self.assertIn("MLA", str(e.exception))

    def test_model_specific_adapter_wins(self):
        mod = derive.load_model("GLM-5.2-FP8", _ckpt("GLM-5.2-FP8"))
        self.assertEqual(mod.MODEL_DIR, "GLM-5.2-FP8")


if __name__ == "__main__":
    unittest.main()
