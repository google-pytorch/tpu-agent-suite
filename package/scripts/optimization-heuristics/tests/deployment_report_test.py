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
"""deployment.py (manifest and recipe in, candidate manifest out) and report.py."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import deployment  # noqa: E402  # pylint: disable=wrong-import-position
import fetch_hf_config  # noqa: E402  # pylint: disable=wrong-import-position
import report  # noqa: E402  # pylint: disable=wrong-import-position

_RECIPE = """#!/bin/bash
set -eo pipefail
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export USE_MOE_SPARSE_CORE=0
bash apply_vllm_patch.sh gpt_oss_mxfp4.py
exec bash serve_vllm.sh "openai/gpt-oss-20b" \\
  --port 8000 \\
  --tensor-parallel-size 8 \\
  --data-parallel-size 1 \\
  --max-model-len 8192 \\
  --host 0.0.0.0 \\
  --attention-backend CUSTOM
"""
_DEP = {
    "model_id": "openai/gpt-oss-20b",
    "engine": "vllm",
    "port": 8000,
    "tensor_parallel_size": 8,
    "data_parallel_size": 1,
    "max_model_len": 8192,
    "max_num_batched_tokens": 4096,
    "parallelism": {
        "source": "recommended"
    },
}


class DeploymentTest(unittest.TestCase):

    def test_recipe_gives_env_patches_and_flags(self):
        r = deployment.parse_recipe(_RECIPE)
        self.assertEqual(
            r["env"], {
                "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "false",
                "USE_MOE_SPARSE_CORE": "0"
            })
        self.assertEqual(r["patches"], ["gpt_oss_mxfp4.py"])
        self.assertIn("--attention-backend", r["args"])
        self.assertNotIn("openai/gpt-oss-20b", r["args"])

    def test_baseline_merges_manifest_and_recipe(self):
        b = deployment.baseline(_DEP, _RECIPE, "sha256:abc",
                                [(1024, 1024, 64)])
        self.assertEqual((b["tp"], b["dp"], b["max_model_len"]), (8, 1, 8192))
        self.assertEqual(b["patches"], ["gpt_oss_mxfp4.py"])
        self.assertEqual(b["points"], [[1024, 1024, 64]])
        args = b["vllm_args"]
        self.assertEqual(args[args.index("--max-num-batched-tokens") + 1],
                         "4096")
        self.assertEqual(args[args.index("--attention-backend") + 1], "CUSTOM")
        # Where the server listens is never a change to measure.
        self.assertNotIn("--host", args)
        self.assertNotIn("--port", args)

    def test_apply_change_writes_a_manifest(self):
        base = deployment.baseline(_DEP, _RECIPE)
        change = {
            "env": dict(base["env"], TPU_X="1"),
            "vllm_args":
            ["--attention-backend", "CUSTOM", "--async-scheduling"],
            "tp": 4,
            "dp": 2,
            "max_model_len": 8192,
            "patches": base["patches"] + ["extra.patch"],
        }
        out = deployment.apply_change(_DEP, change, base)
        self.assertEqual(
            (out["tensor_parallel_size"], out["data_parallel_size"]), (4, 2))
        self.assertIs(out["async_scheduling"], True)
        self.assertEqual(out["attention_backend"], "CUSTOM")
        # A flag the change no longer carries is gone.
        self.assertNotIn("max_num_batched_tokens", out)
        # Only what differs from the recipe is recorded.
        self.assertEqual(out["env"], {"TPU_X": "1"})
        self.assertEqual(out["vllm_patches"], ["extra.patch"])
        self.assertEqual(out["parallelism"], _DEP["parallelism"])


def _summary(values, failed=0):
    """values: {cell: [(tok_s, ttft_ms, itl_ms), ...]} -> summary.json."""
    return {
        "cells": [{
            "isl":
            i,
            "osl":
            o,
            "concurrency":
            c,
            "replicates": [{
                "output_tokens_per_s": tok_s,
                "ttft_p50_ms": ttft,
                "itl_p50_ms": itl,
                "failed": failed
            } for tok_s, ttft, itl in reps]
        } for (i, o, c), reps in values.items()]
    }


_LONG, _SHORT = (8192, 1024, 128), (1024, 1024, 32)
# Gemma-4 on sha256:8e22f91d, CUSTOM, two runs per point,
# 2026-10-16: TP4xDP2 (baseline) and TP2xDP4.
_BASE = _summary({
    _LONG: [(3006.4, 1338.0, 39.0), (3005.9, 1338.5, 39.1)],
    _SHORT: [(2869.1, 333.0, 10.6), (2871.2, 333.4, 10.6)],
})
_TP2 = _summary({
    _LONG: [(3834.7, 1357.0, 30.1), (3835.6, 1356.8, 30.0)],
    _SHORT: [(2268.4, 275.0, 13.6), (2267.9, 275.2, 13.7)],
})


class ReportTest(unittest.TestCase):

    def _report(self, measured):
        cands = {
            "candidates": [{
                "cand_id": k,
                "kind": "mesh"
            } for k in measured]
        }
        return report.report(cands, _BASE, measured)

    def test_mixed_layout_is_reported_per_point(self):
        (c, ) = self._report({"m-tp2xdp4": _TP2})["candidates"]
        self.assertEqual(c["throughput_wins_at"], ["8192:1024:128"])
        self.assertEqual(c["throughput_losses_at"], ["1024:1024:32"])
        long = next(r for r in c["points"] if r["point"] == "8192:1024:128")
        self.assertAlmostEqual(long["output_tokens_per_s"]["delta_pct"],
                               27.6,
                               places=0)
        # Lower is better for latency: a shorter TTFT at 1k/1k is a win.
        short = next(r for r in c["points"] if r["point"] == "1024:1024:32")
        self.assertEqual(short["ttft_p50_ms"]["verdict"], "win")
        self.assertEqual(short["itl_p50_ms"]["verdict"], "loss")
        self.assertTrue(short["output_tokens_per_s"]["noise_measured"])
        # Nothing is adopted here: the report decides nothing.
        self.assertNotIn("adopted", self._report({"m-tp2xdp4": _TP2}))

    def test_without_candidates_each_measured_arm_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for name, summary in (("base", _BASE), ("stack", _TP2)):
                paths[name] = os.path.join(tmp, f"{name}.json")
                with open(paths[name], "w", encoding="utf-8") as f:
                    json.dump(summary, f)
            out = os.path.join(tmp, "report.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    report.main([
                        "--baseline", paths["base"], "--measured",
                        f"stack={paths['stack']}", "--out", out
                    ]), 0)
            with open(out, encoding="utf-8") as f:
                (arm, ) = json.load(f)["candidates"]
        self.assertEqual((arm["cand_id"], arm["measured"]), ("stack", True))

    def test_one_replicate_assumes_the_noise(self):
        one = _summary({_LONG: [(3100.0, 1300.0, 38.0)]})
        (c, ) = self._report({"k": one})["candidates"]
        (row, ) = c["points"]
        self.assertFalse(row["output_tokens_per_s"]["noise_measured"])
        self.assertEqual(row["output_tokens_per_s"]["noise_pct"], 3.0)
        # +3.1% clears the assumed 3% noise: a win.
        self.assertEqual(row["output_tokens_per_s"]["verdict"], "win")
        self.assertEqual(c["missing_points"], ["1024:1024:32"])

    def test_failures_and_unmeasured_candidates(self):
        failing = _summary(
            {
                _LONG: [(3500.0, 1300.0, 35.0)] * 2,
                _SHORT: [(2900.0, 330.0, 10.5)] * 2
            },
            failed=1)
        out = self._report({"k": failing})
        self.assertEqual(out["candidates"][0]["failed_requests"], 4)
        cands = {"candidates": [{"cand_id": "x"}]}
        self.assertFalse(
            report.report(cands, _BASE, {})["candidates"][0]["measured"])


# Gemma-4 layouts on sha256:8e22f91d, CUSTOM, two runs per point,
# 2026-10-16: TP 8 (baseline), TP 4 x DP 2, TP 2 x DP 4.
_G4_TP8 = _summary({
    (1024, 1024, 128): [(6558.0, 341.3, 15.38), (6639.1, 343.1, 15.36)],
    (1024, 1024, 32): [(3093.1, 314.7, 9.32), (3103.0, 315.2, 9.32)],
    (8192, 1024, 128): [(2441.9, 742.5, 21.24), (2444.6, 742.2, 21.25)],
    (8192, 1024, 32): [(1735.7, 669.6, 10.73), (1738.5, 669.3, 10.73)],
})
_G4_TP4 = _summary({
    (1024, 1024, 128): [(6777.4, 485.3, 15.31), (6812.9, 462.9, 15.32)],
    (1024, 1024, 32): [(2864.4, 343.1, 10.06), (2903.7, 351.0, 10.03)],
    (8192, 1024, 128): [(3019.8, 1178.1, 18.16), (3016.0, 1291.6, 18.2)],
    (8192, 1024, 32): [(1883.0, 1395.7, 10.77), (1912.9, 1114.7, 10.73)],
})
_G4_TP2 = _summary({
    (1024, 1024, 128): [(6418.0, 494.8, 17.99), (6523.4, 433.6, 17.86)],
    (1024, 1024, 32): [(2255.9, 176.7, 13.51), (2276.4, 252.6, 13.45)],
    (8192, 1024, 128): [(3845.3, 1351.8, 20.85), (3837.1, 1348.5, 20.79)],
    (8192, 1024, 32): [(1840.4, 1044.3, 14.26), (1830.1, 989.7, 14.18)],
})


class RegimeTest(unittest.TestCase):

    def test_each_regime_is_summarized(self):
        cands = {
            "candidates": [{
                "cand_id": "m-tp4xdp2"
            }, {
                "cand_id": "m-tp2xdp4"
            }]
        }
        out = report.report(cands, _G4_TP8, {
            "m-tp4xdp2": _G4_TP4,
            "m-tp2xdp4": _G4_TP2
        })
        by_id = {c["cand_id"]: c["regimes"] for c in out["candidates"]}
        short, long = by_id["m-tp4xdp2"]["1024:1024"], by_id["m-tp4xdp2"][
            "8192:1024"]
        self.assertEqual(short["throughput_losses_at"], ["1024:1024:32"])
        self.assertEqual(long["throughput_wins_at"],
                         ["8192:1024:32", "8192:1024:128"])
        self.assertLess(short["mean_throughput_delta_pct"], 0)
        self.assertGreater(long["mean_throughput_delta_pct"], 15)
        # TP 2 x DP 4: the largest long-prompt gain, a loss at short prompts.
        tp2 = by_id["m-tp2xdp4"]
        self.assertGreater(tp2["8192:1024"]["mean_throughput_delta_pct"],
                           long["mean_throughput_delta_pct"])
        self.assertIn("1024:1024:32", tp2["1024:1024"]["throughput_losses_at"])
        # Long prompts at 1.7x the TTFT: a latency loss the report names.
        self.assertIn("8192:1024:128", long["ttft_losses_at"])
        # Judged where the baseline peaks: 1k/1k peaks at c128, where
        # TP 4 x DP 2 wins (+3%) and TP 2 x DP 4 loses.
        self.assertEqual(short["peak_point"], "1024:1024:128")
        self.assertEqual(short["at_peak"]["output_tokens_per_s"]["verdict"],
                         "win")
        self.assertEqual(
            tp2["1024:1024"]["at_peak"]["output_tokens_per_s"]["verdict"],
            "loss")
        self.assertEqual(long["peak_point"], "8192:1024:128")


class FetchHfConfigTest(unittest.TestCase):

    def test_model_and_revision_come_from_the_deployment(self):
        import json  # pylint: disable=import-outside-toplevel
        import tempfile  # pylint: disable=import-outside-toplevel
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "model")
            os.makedirs(model)
            with open(os.path.join(model, "config.json"),
                      "w",
                      encoding="utf-8") as f:
                json.dump({"hidden_size": 5376, "head_dim": 256}, f)
            dep = os.path.join(d, "deployments.json")
            with open(dep, "w", encoding="utf-8") as f:
                json.dump(
                    {"deployments": [{
                        "model_id": model,
                        "revision": "abc"
                    }]}, f)
            self.assertEqual(fetch_hf_config.resolve(dep, None, None),
                             (model, "abc"))
            out = os.path.join(d, "hf_config.json")
            self.assertEqual(
                fetch_hf_config.main(["--deployments", dep, "--out", out]), 0)
            with open(out, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["head_dim"], 256)

    def test_gcs_and_missing_model_fail_clearly(self):
        with self.assertRaises(ValueError):
            fetch_hf_config.resolve(None, None, None)
        with self.assertRaises(ValueError):
            fetch_hf_config.fetch("gs://b/m", None, "/tmp/x.json")


class ManifestNotesTest(unittest.TestCase):
    """Notes onboarding writes into a manifest never become vLLM flags."""

    def test_free_text_and_record_keys_stay_records(self):
        dep = {
            "model_id": "/model",
            "tensor_parallel_size": 8,
            "dtype": "bfloat16",
            "revision": "9216db5781bf",
            "checkpoint": "gs://bucket/model/",
            "sizing": "61.0 GiB weights; TP8: 7.6 GiB per device",
            "history": "attempt 1 ran max_num_seqs 256",
        }
        base = deployment.baseline(dep)
        args = base["vllm_args"]
        for flag_name in ("--sizing", "--history", "--checkpoint"):
            self.assertNotIn(flag_name, args)
        self.assertIn("--revision", args)
        self.assertIn("--dtype", args)
        out = deployment.apply_change(dep, dict(base, tp=4, dp=2), base)
        self.assertEqual(out["sizing"], dep["sizing"])
        self.assertEqual(out["checkpoint"], dep["checkpoint"])
        self.assertEqual(out["tensor_parallel_size"], 4)


if __name__ == "__main__":
    unittest.main()
