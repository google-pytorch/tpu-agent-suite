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
"""Real traces, trimmed: the findings and candidates their A/B runs confirmed.

The fixtures are Events DB parquets cut from XProf traces of vLLM-TorchTPU
servers on tpu7x 2x2x1 (8 devices), one device each:

- Gemma-3-27B TP 8, stock decode blocks (`bkv_10240_10240`) and smaller ones
  (`bkv_1536_1536`): D45 fires on the first, with sizes computed from the
  kernel names and the context, and not on the second.
- Gemma-4-31B TP 4 x DP 2, 8k/1k prefill: collectives 43% of the step;
  TP 2 x DP 4 gave +28% at 8k/1k c128 and lost at short prompts.
- Qwen3.8-27B TP 4 x DP 2, 1k/1k prefill: every other layout fails to start.
"""

import json
import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(_ROOT), "common"))

import analyze  # noqa: E402  # pylint: disable=wrong-import-position
import deployment  # noqa: E402  # pylint: disable=wrong-import-position
import propose  # noqa: E402  # pylint: disable=wrong-import-position

_DATA = os.path.join(_ROOT, "testdata")
_CONFIGS = os.path.join(os.path.dirname(_ROOT), "common", "testdata",
                        "oss_configs")
_HBM = 94.75


def _load(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return json.load(f)


def _baseline(model):
    d = os.path.join(_DATA, "deployments", model)
    meta = _load(d, "meta.json")
    with open(os.path.join(d, "launch_vllm.sh"), encoding="utf-8") as f:
        recipe = f.read()
    return deployment.baseline(
        deployment.load(os.path.join(d, "deployments.json")), recipe,
        meta["image"], [analyze.parse_point(p) for p in meta["points"]])


def _findings(model, *profiles):
    bench = _baseline(model)
    events = [
        analyze.parse_events(
            os.path.join(_DATA, "events", name) + f",role={role},point={pt}")
        for name, role, pt in profiles
    ]
    return analyze.run(events, analyze.serve_command(bench), None, None, None,
                       True), bench


def _ids(payload, label):
    p = next(p for p in payload["profiles"] if p["label"] == label)
    return [f["detector_id"] for f in p["result"]["findings"]]


class EventsTest(unittest.TestCase):

    def test_gemma3_stock_fires_d45_and_tuned_does_not(self):
        payload, _ = _findings(
            "gemma3", ("gemma3_stock.parquet", "mixed", "1024:1024:64"))
        self.assertIn("D45_rpa_decode_block_oversized", _ids(payload, "mixed"))
        d45 = [
            f for p in payload["profiles"] for f in p["result"]["findings"]
            if f["detector_id"] == "D45_rpa_decode_block_oversized"
        ][0]
        # Sizes come from the kernel names and the 1k/1k context: one window
        # for the sliding layers, a chunk covering 2048 tokens for the global.
        self.assertEqual(
            [(k["name"], k["suggested_value"], k["requires_patch"])
             for k in d45["knob_changes"]],
            [("TPU_RPA_DECODE_BLOCK_SIZES",
              "global=10240:2048,sliding=1024:1024",
              "scripts/common/vllm_patches/"
              "rpa_decode_block_override.patch")])
        tuned, _ = _findings("gemma3",
                             ("gemma3_tuned.parquet", "mixed", "1024:1024:64"))
        self.assertNotIn("D45_rpa_decode_block_oversized",
                         _ids(tuned, "mixed"))

    def test_gemma3_d45_candidate_carries_the_patch(self):
        payload, bench = _findings(
            "gemma3", ("gemma3_stock.parquet", "mixed", "1024:1024:64"))
        out = propose.propose(payload,
                              bench,
                              _load(_CONFIGS, "gemma-3-27b-it.json"),
                              _HBM,
                              devices=8)
        d45 = [
            c for c in out["candidates"] + out["reported_only"]
            if "D45_rpa_decode_block_oversized" in c["provenance"]["findings"]
        ]
        self.assertTrue(d45)

    def test_gemma4_prefill_proposes_tp2xdp4_workload_dependent(self):
        payload, bench = _findings(
            "gemma4",
            ("gemma4_8k1k_prefill.parquet", "prefill", "8192:1024:128"))
        self.assertIn("D7_collective_bound", _ids(payload, "prefill"))
        out = propose.propose(payload, bench,
                              _load(_CONFIGS, "gemma-4-31B-it.json"), _HBM)
        self.assertTrue(out["prefill_collective_bound"])
        (cand, ) = out["candidates"]
        self.assertEqual(
            (cand["serving_change"]["tp"], cand["serving_change"]["dp"]),
            (2, 4))
        self.assertEqual(cand["status"], "workload_dependent")
        directions = {
            k: v["expected_direction"]
            for k, v in cand["per_point_assessment"].items()
        }
        self.assertEqual(directions["8k1k c128"], "positive")
        self.assertEqual(directions["1k1k c32"], "negative")
        self.assertIn("TP1xDP8", [m["label"] for m in out["rejected_meshes"]])

    def test_qwen38_proposes_nothing(self):
        payload, bench = _findings(
            "qwen38",
            ("qwen38_1k1k_prefill.parquet", "prefill", "1024:1024:1024"))
        self.assertIn("D7_collective_bound", _ids(payload, "prefill"))
        out = propose.propose(payload, bench,
                              _load(_CONFIGS, "Qwen3.8-27B.json"), _HBM)
        self.assertEqual(out["candidates"], [])
        rejected = {
            m["label"]: [r["id"] for r in m["rejected_by"]]
            for m in out["rejected_meshes"]
        }
        self.assertIn("qwen38_tp8_engine_init", rejected["TP8xDP1"])
        self.assertIn("gdn_v3_triangular_block", rejected["TP2xDP4"])


class D27ModelDimsTest(unittest.TestCase):
    """D27 needs the model's dimensions to tell max_num_seqs from a model axis."""

    def _ids(self, model, **ctx_kw):
        return [f.detector_id for f in self._findings(model, **ctx_kw)]

    def _findings(self, model, **ctx_kw):
        from serving_analysis import \
            analyze as analyze_lib  # pylint: disable=import-outside-toplevel
        from serving_analysis import \
            context as context_lib  # pylint: disable=import-outside-toplevel
        from serving_analysis import \
            trace_table  # pylint: disable=import-outside-toplevel
        cmd = (
            "vllm serve /model --tensor-parallel-size 4 "
            "--data-parallel-size 2 --max-num-seqs 256 --max-model-len 10240")
        ctx = context_lib.ServingContext(
            framework="vllm_torchtpu",
            model=model,
            serving_config=context_lib.parse_serving_command(cmd),
            serving_command=cmd,
            input_len=8192,
            output_len=1024,
            concurrency=128,
            ttft_sensitive=True,
            **ctx_kw)
        table = trace_table.TraceTable.from_parquet(os.path.join(
            _DATA, "events", "gemma4_8k1k_prefill.parquet"),
                                                    session_id="g4")
        return analyze_lib.analyze(table, ctx, session_id="g4").findings

    def test_head_dim_256_is_not_max_num_seqs_padding(self):
        import dataclasses  # pylint: disable=import-outside-toplevel

        from serving_analysis import \
            context as context_lib  # pylint: disable=import-outside-toplevel
        real = context_lib.ModelConfig.from_hf_dict(
            _load(_CONFIGS, "gemma-4-31B-it.json"))
        d27 = "D27_max_num_seqs_inflation"
        # Gemma-4's head_dim is 256, like --max-num-seqs here: no finding.
        self.assertNotIn(d27, self._ids(real))
        # Without the config the axis cannot be told apart: no finding either.
        self.assertNotIn(d27, self._ids(context_lib.ModelConfig()))
        # The trace does carry 256-sized axes: a head_dim that is not 256
        # attributes them to --max-num-seqs.
        self.assertIn(d27, self._ids(dataclasses.replace(real, head_dim=128)))

    def test_proposal_serves_the_busiest_benchmark_point(self):
        import dataclasses  # pylint: disable=import-outside-toplevel

        from serving_analysis import \
            context as context_lib  # pylint: disable=import-outside-toplevel
        model = dataclasses.replace(context_lib.ModelConfig.from_hf_dict(
            _load(_CONFIGS, "gemma-4-31B-it.json")),
                                    head_dim=128)

        def change(conc):
            (d27, ) = [
                f for f in self._findings(model, benchmark_concurrencies=conc)
                if f.detector_id == "D27_max_num_seqs_inflation"
            ]
            return [(k.name, k.suggested_value) for k in d27.knob_changes]

        # Captured at 128 users; the busiest point is 256 over 2 replicas:
        # 128 per replica, half of --max-num-seqs 256.
        self.assertEqual(change((1, 64, 128, 256)),
                         [("--max-num-seqs", "128")])
        # The busiest point needs 256 per replica: nothing to shrink.
        self.assertEqual(change((1, 64, 128, 512)), [])
        # No benchmark list: the captured concurrency (64 per replica).
        self.assertEqual(change(()), [("--max-num-seqs", "64")])


class TargetPhaseTest(unittest.TestCase):

    @staticmethod
    def _payload(prefill_ms, decode_ms, bucket=2048, point=(8192, 1024, 256)):
        return {
            "profiles": [
                {
                    "role": "prefill",
                    "point": list(point),
                    "result": {
                        "phase_stats": [{
                            "phase": "prefill_or_mixed",
                            "mean_dur_ms": prefill_ms,
                            "mean_bucket": bucket
                        }],
                        "findings": [{
                            "detector_id": "D7_collective_bound",
                            "phase": "prefill_or_mixed"
                        }]
                    }
                },
                {
                    "role": "decode",
                    "point": list(point),
                    "result": {
                        "phase_stats": [{
                            "phase": "decode",
                            "mean_dur_ms": decode_ms
                        }],
                        "findings": [{
                            "detector_id": "D45_rpa_decode_block_oversized",
                            "phase": "decode"
                        }]
                    }
                },
            ]
        }

    def test_mix_comes_from_step_times_and_request_shape(self):
        # 8192 x 256 / 2048 = 1024 prefill steps of 100 ms against 1024
        # decode steps of 50 ms: two thirds prefill.
        mix = propose.phase_mix(self._payload(100, 50), (8192, 1024, 256))
        self.assertEqual(mix["phase"], "prefill")
        self.assertAlmostEqual(mix["prefill_share"], 0.667, places=3)
        # Replicas split the prompts: at DP 4 decode dominates.
        self.assertEqual(
            propose.phase_mix(self._payload(100, 50), (8192, 1024, 256),
                              dp=4)["phase"], "decode")

    def test_mix_falls_back_to_lengths_without_captures(self):
        mix = propose.phase_mix({"profiles": []}, (1024, 1024, 64))
        self.assertEqual((mix["phase"], mix["prefill_share"]),
                         ("decode", None))

    def test_findings_are_grouped_by_phase(self):
        fired = propose._finding_phases(self._payload(100, 50),
                                        [(8192, 1024, 256)])
        self.assertEqual(fired["D7_collective_bound"], {"prefill"})
        self.assertEqual(fired["D45_rpa_decode_block_oversized"], {"decode"})

    def test_gemma4_target_records_phase_and_match(self):
        payload, bench = _findings(
            "gemma4",
            ("gemma4_8k1k_prefill.parquet", "prefill", "8192:1024:128"))
        out = propose.propose(payload,
                              bench,
                              _load(_CONFIGS, "gemma-4-31B-it.json"),
                              _HBM,
                              target_phase="prefill")
        self.assertEqual(out["target"]["phase"], "prefill")
        self.assertEqual(out["target"]["points"], ["8k1k c128"])
        self.assertTrue(out["candidates"][0]["target_phase_match"])
        out = propose.propose(payload,
                              bench,
                              _load(_CONFIGS, "gemma-4-31B-it.json"),
                              _HBM,
                              target_phase="decode")
        self.assertFalse(out["candidates"][0]["target_phase_match"])


class BatchedRpaDecodeTest(unittest.TestCase):
    """D46: a CUSTOM baseline gets RPA v3 with per-layer decode blocks."""

    def _run(self, config, backend="CUSTOM"):
        bench = _baseline("gemma3")
        ev = [
            analyze.parse_events(
                os.path.join(_DATA, "events", "gemma3_stock.parquet") +
                ",role=mixed,point=1024:1024:64")
        ]
        cmd = analyze.serve_command(bench).replace("FLASH_ATTN", backend)
        payload = analyze.run(ev, cmd, os.path.join(_CONFIGS, config), None,
                              None, True)
        return [
            f for p in payload["profiles"] for f in p["result"]["findings"]
            if f["detector_id"] == "D46_batched_rpa_decode_untunable"
        ]

    def test_custom_baseline_gets_flash_attn_with_decode_blocks(self):
        (d46, ) = self._run("gemma-3-27b-it.json")
        self.assertEqual(
            {k["name"]: k["suggested_value"]
             for k in d46["knob_changes"]}, {
                 "--attention-backend": "FLASH_ATTN",
                 "TPU_RPA_DECODE_BLOCK_SIZES":
                 "global=10240:2048,sliding=1024:1024"
             })

    def test_known_rpa_v3_failure_is_reported_without_a_change(self):
        (d46, ) = self._run("gemma-4-31B-it.json")
        self.assertEqual(d46["knob_changes"], [])
        self.assertIn("Not proposed", d46["summary"])

    def test_candidate_manifest_switches_the_backend(self):
        bench = _baseline("gemma3")
        bench["vllm_args"] = [
            "CUSTOM" if a == "FLASH_ATTN" else a
            for a in bench.get("vllm_args") or []
        ]
        ev = [
            analyze.parse_events(
                os.path.join(_DATA, "events", "gemma3_stock.parquet") +
                ",role=mixed,point=1024:1024:64")
        ]
        payload = analyze.run(ev, analyze.serve_command(bench),
                              os.path.join(_CONFIGS, "gemma-3-27b-it.json"),
                              None, None, True)
        out = propose.propose(payload,
                              bench,
                              _load(_CONFIGS, "gemma-3-27b-it.json"),
                              _HBM,
                              devices=8)
        (cand, ) = [
            c for c in out["candidates"] + out["reported_only"] if
            "D46_batched_rpa_decode_untunable" in c["provenance"]["findings"]
        ]
        dep = deployment.load(
            os.path.join(_DATA, "deployments", "gemma3", "deployments.json"))
        manifest = deployment.apply_change(dep, cand["serving_change"], bench)
        self.assertEqual(manifest["attention_backend"], "FLASH_ATTN")
        self.assertEqual(manifest["env"]["TPU_RPA_DECODE_BLOCK_SIZES"],
                         "global=10240:2048,sliding=1024:1024")

    def test_quiet_when_the_baseline_is_not_custom(self):
        self.assertEqual(self._run("gemma-3-27b-it.json", "FLASH_ATTN"), [])


class MixedTargetPhaseTest(unittest.TestCase):

    def test_targets_in_both_phases_rank_both(self):
        payload = TargetPhaseTest._payload(100, 50)
        payload["profiles"] += TargetPhaseTest._payload(10,
                                                        50,
                                                        point=(1024, 1024,
                                                               64))["profiles"]
        mixes = {
            p: propose.phase_mix(payload, p)["phase"]
            for p in ((8192, 1024, 256), (1024, 1024, 64))
        }
        self.assertEqual(set(mixes.values()), {"prefill", "decode"})
        fired = propose._finding_phases(payload, list(mixes))
        self.assertEqual(fired["D45_rpa_decode_block_oversized"], {"decode"})


class DuplicateCandidateIdTest(unittest.TestCase):
    """D46 at two points with different block sizes: two ids, two manifests."""

    def test_same_detector_two_points_gets_distinct_ids(self):
        bench = _baseline("gemma3")
        bench["vllm_args"] = [
            "CUSTOM" if a == "FLASH_ATTN" else a
            for a in bench.get("vllm_args") or []
        ]
        ev = [
            analyze.parse_events(
                os.path.join(_DATA, "events", "gemma3_stock.parquet") +
                f",role=mixed,point={pt}")
            for pt in ("1024:1024:64", "8192:1024:64")
        ]
        payload = analyze.run(ev, analyze.serve_command(bench),
                              os.path.join(_CONFIGS, "gemma-3-27b-it.json"),
                              None, None, True)
        out = propose.propose(payload,
                              bench,
                              _load(_CONFIGS, "gemma-3-27b-it.json"),
                              _HBM,
                              devices=8,
                              max_candidates=6)
        d46 = [
            c for c in out["candidates"] + out["reported_only"] if
            "D46_batched_rpa_decode_untunable" in c["provenance"]["findings"]
        ]
        ids = [c["cand_id"] for c in d46]
        self.assertGreaterEqual(len(ids), 2)
        self.assertEqual(len(ids), len(set(ids)), ids)
        sizes = {
            c["serving_change"]["env"]["TPU_RPA_DECODE_BLOCK_SIZES"]
            for c in d46 if "reason" not in c
        }
        self.assertEqual(len(sizes),
                         len([c for c in d46 if "reason" not in c]))


if __name__ == "__main__":
    unittest.main()
