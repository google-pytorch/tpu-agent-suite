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
"""CPU tests for core/steps.py and core/mix.py: steps, step mix, mix plan, repeat merging and token paddings.

Run from scripts/workload-microbenchmark/: `python3 -m unittest discover tests`.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

from core.deploy import Deploy  # noqa: E402


class Steps(unittest.TestCase):
    D = Deploy(tp=16,
               dcp=16,
               ep=True,
               block_size=256,
               mnbt=4096,
               mns=8,
               spec_method="mtp",
               spec_tokens=3,
               prefix_caching=True)

    def test_parse_point(self):
        from core.steps import Point, parse_point
        self.assertEqual(parse_point("P90=299K/6K/1.4K@8"),
                         Point("P90", 299000, 6000, 1400, 8))
        with self.assertRaises(ValueError):
            parse_point("P90=299K/6K@8")

    def test_point_steps(self):
        from core.steps import Point, point_steps
        s = point_steps(Point("x", 65000, 1000, 200, 7), self.D, acc=3.0)
        # hit: 15 whole 4096-blocks of 65000, minus the MTP-dropped block
        self.assertEqual(s["hit"], 14 * 4096)
        self.assertEqual(s["computed"], 66000 - 14 * 4096)
        self.assertEqual(s["chunk_tokens"],
                         4096 - 6 * 4)  # 6 neighbours x 4 rows
        self.assertEqual(len(s["mixed"]["batches"]), s["chunks_full"])
        b = s["mixed"]["batches"][-1]
        self.assertEqual(sum(r["q"] for r in b), 4096)
        self.assertEqual((s["decode"]["tokens"], s["decode"]["bucket"]),
                         (28, 32))
        self.assertEqual(s["last_chunk"]["bucket"], 1024)  # 896 + 24 tokens

    def test_step_mix(self):
        from core.steps import step_mix
        m = step_mix(C=4, chunks=2, output=300, acc=3.0)
        # 8 mixed steps give 24 decode slots of the 400 needed; 376 / 4 decode-only steps
        self.assertEqual((m["mixed_steps"], m["decode_steps"]), (8, 94.0))


class StepMix(unittest.TestCase):
    """core/mix.py: a job's per-second step histogram + its requests -> the step mix."""
    D = Deploy(tp=16,
               dcp=16,
               ep=True,
               block_size=256,
               mnbt=4096,
               mns=8,
               spec_method="mtp",
               spec_tokens=3,
               prefix_caching=True)

    @staticmethod
    def _snap():
        # 0-99 s: one request decoding alone, 25 steps/s (40 ms, bucket <=8)
        # 100-102 s: a 3 s prefill step (no steps end, then one > 512-token step ends)
        # 103-109 s: nothing running (idle)
        rows = [[t, 0, 25, 0, 0, 0, 0, 1, 1, 1.0] for t in range(100)]
        rows += [[100, 0, 0, 0, 0, 0, 0, 2, 2, 2.0],
                 [101, 0, 0, 0, 0, 0, 0, 2, 2, 2.0],
                 [102, 0, 0, 0, 0, 0, 1, 2, 2, 2.0]]
        rows += [[t, 0, 0, 0, 0, 0, 0, 0, 0, 0.0] for t in range(103, 110)]

        def req(conv, turn, t0, t1, ttft, isl, osl):
            return {
                "conv": conv,
                "turn": turn,
                "kind": "main",
                "phase": "profiling",
                "t0": t0,
                "t1": t1,
                "isl": isl,
                "osl": osl,
                "ttft_ms": ttft,
                "error": False
            }

        return {
            "step_slices": {
                "rows": rows
            },
            "requests": [
                req("a", 1, 0.0, 103.0, 10.0, 100_000, 7000),
                req("b", 1, 99.5, 110.0, 3500.0, 50_000, 10)
            ]
        }

    def test_time_split_and_counts(self):
        from core import mix
        m = mix.build(self._snap(), self.D)
        self.assertAlmostEqual(m["measured_step_ms"]["D1-2"], 40.0, places=1)
        self.assertEqual(m["steps"], {
            "D1-2": 2500,
            "D3-4": 0,
            "D5-8": 0,
            "mixed": 1
        })
        sh = m["measured_share_of_wall"]
        self.assertAlmostEqual(sh["decode D1-2"], 100 / 110, places=3)
        self.assertAlmostEqual(sh["mixed"], 3 / 110, places=3)
        self.assertAlmostEqual(sh["idle"], 7 / 110, places=3)

    def test_cells_conserve_steps_and_share(self):
        from core import mix
        m = mix.build(self._snap(), self.D)
        dec = [c for c in m["cells"] if c["kind"] == "decode"]
        self.assertAlmostEqual(sum(c["steps"] for c in dec), 2500, delta=0.5)
        self.assertAlmostEqual(sum(c["share"] for c in m["cells"]),
                               1.0,
                               places=3)
        self.assertTrue(all(c["batch"] == 1
                            for c in dec))  # one request decoding
        mx = [c for c in m["cells"] if c["kind"] == "mixed"]
        self.assertTrue(mx and all(c["chunk"] == 4096 - c["batch"] * 4
                                   for c in mx))

    def test_no_slices_is_an_error_not_a_guess(self):
        from core import mix
        with self.assertRaises(ValueError):
            mix.build({"requests": [], "step_slices": None}, self.D)


class DecodeClasses(unittest.TestCase):
    """The decode classes follow from the job: tokens per request (1 + MTP k) and mns."""
    EDGES = ["1.0", "8.0", "16.0", "32.0", "512.0", "+Inf"]

    @staticmethod
    def _deploy(spec_tokens, mns):
        return Deploy(tp=8,
                      mnbt=4096,
                      mns=mns,
                      spec_method="mtp" if spec_tokens else None,
                      spec_tokens=spec_tokens)

    def _classes(self, spec_tokens, mns):
        from core import mix
        return [(n, lo, hi, bins) for n, lo, hi, bins in mix.decode_classes(
            self.EDGES, self._deploy(spec_tokens, mns))]

    def test_mtp3_mns8_is_the_reference_split(self):
        # the <=1-token bin holds no whole 4-token request: it joins D1-2
        self.assertEqual(self._classes(3, 8), [("D1-2", 1, 2, [0, 1]),
                                               ("D3-4", 3, 4, [2]),
                                               ("D5-8", 5, 8, [3])])

    def test_no_mtp_counts_one_token_per_request(self):
        self.assertEqual(self._classes(0, 8), [("D1", 1, 1, [0]),
                                               ("D2-8", 2, 8, [1])])

    def test_more_running_requests_add_classes(self):
        self.assertEqual([c[0] for c in self._classes(3, 16)],
                         ["D1-2", "D3-4", "D5-8", "D9-16"])
        self.assertEqual([c[0] for c in self._classes(0, 64)],
                         ["D1", "D2-8", "D9-16", "D17-32", "D33-64"])

    def test_no_mtp_mix_reads_batch_from_the_right_bin(self):
        # 0-99 s: four requests decoding without MTP, 25 steps/s of 4 tokens (bin <=8)
        from core import mix
        rows = [[t, 0, 25, 0, 0, 0, 0, 4, 4, 4.0] for t in range(100)]

        def req(c):
            return {
                "conv": c,
                "turn": 1,
                "kind": "main",
                "phase": "profiling",
                "t0": 0.0,
                "t1": 100.0,
                "ttft_ms": 1.0,
                "isl": 1000,
                "osl": 2500,
                "error": False
            }

        m = mix.build(
            {
                "step_slices": {
                    "edges": self.EDGES,
                    "rows": rows
                },
                "requests": [req(c) for c in "abcd"]
            }, self._deploy(0, 8))
        self.assertEqual(m["steps"], {"D1": 0, "D2-8": 2500, "mixed": 0})
        dec = [c for c in m["cells"] if c["kind"] == "decode"]
        self.assertTrue(dec and all(c["class"] == "D2-8" for c in dec))
        batch = sum(c["batch"] * c["steps"] for c in dec) / 2500
        self.assertAlmostEqual(batch, 4.0, places=2)  # MTP-4 rule would say 1


class MixPlan(unittest.TestCase):
    D = StepMix.D

    def _mix(self):

        def cell(name, kind, cls, b, ctx, steps, share, **kw):
            return {
                "name": name,
                "kind": kind,
                "class": cls,
                "batch": b,
                "context": ctx,
                "steps": steps,
                "share": share,
                "measured_step_ms": 1.0,
                **kw
            }

        return {
            "rows_per_decode":
            4,
            "busy_s":
            100.0,
            "idle_s":
            5.0,
            "steps": {},
            "measured_share_of_wall": {},
            "measured_step_ms": {},
            "cells": [
                cell("d1", "decode", "D1-2", 1, 100_000, 1000, 0.6),
                cell("d2", "decode", "D1-2", 2, 300_000, 400, 0.3),
                cell("m2",
                     "mixed",
                     "mixed",
                     2,
                     150_000,
                     10,
                     0.099,
                     chunk=4088,
                     prefix=40_000),
                cell("tiny", "decode", "D5-8", 6, 100_000, 1, 0.001)
            ]
        }

    def test_entries(self):
        from core.steps import mix_steps, token_paddings
        e = {
            x["point"]["name"]: x
            for x in mix_steps(
                self._mix(), self.D, token_paddings(4096), min_share=0.002)
        }
        self.assertNotIn("tiny", e)  # under min_share
        self.assertEqual(
            (e["d1"]["decode"]["tokens"], e["d1"]["decode"]["bucket"]),
            (4, 16))
        self.assertEqual(e["d1"]["mixed"]["batches"], [])
        b = e["m2"]["mixed"]["batches"][0]
        self.assertEqual(sum(r["q"] for r in b),
                         4096)  # decode first, chunk fills
        self.assertEqual(b[-1], {"q": 4088, "seq": 40_000 + 4088})
        self.assertEqual(len(e["m2"]["decode"]["batch"]),
                         2)  # its decode rows alone
        self.assertEqual([n for n, x in e.items() if x["profile"]],
                         ["d1", "m2"])

    def test_top_keeps_the_heaviest_and_a_floor_kind_and_profiles_all(self):
        from core.steps import (mix_steps, steps_per_shape, token_paddings,
                                top_cells)
        mx = self._mix()
        mx["cells"][2][
            "share"] = 0.12  # mixed carries >= 10% but is lighter than d1, d2
        mx["cells"].insert(2, {**mx["cells"][1], "name": "d3", "share": 0.2})
        self.assertEqual([c["name"] for c in top_cells(mx["cells"], 2)],
                         ["d1", "m2"])
        self.assertEqual([c["name"] for c in top_cells(mx["cells"], 3)],
                         ["d1", "d2", "m2"])
        with self.assertRaises(ValueError):
            top_cells(mx["cells"], 6)
        e = mix_steps(mx, self.D, token_paddings(4096), top=3)
        self.assertTrue(all(x["profile"] for x in e))
        # decode: 1 batch x 10 x 2 + 5 profiled; mixed: (chunk + its decode rows) x that
        self.assertEqual([steps_per_shape(x, 10, 2, 5) for x in e],
                         [25, 25, 50])

    def test_forward_run_times_cells_and_rolls_up(self):
        from core import forward_run as fr
        from core.steps import plan

        class Engine:  # ms = 10 per request + 1000 for the 4096 bucket

            def collective_rpc(self, fn, args):
                if fn is fr._last_inputs:  # the batch's input record (rank 0)
                    return [None]
                rows, bucket, iters = args[:3]  # + the verify-window width
                return [
                    [10.0 * len(rows) +
                     (1000.0 if bucket == 4096 else 0.0)] * iters
                ]

        pl = plan([], self.D, 3.0, "given", mix=self._mix(), min_share=0.002)

        class A:
            iters = 2

        reps = [fr._time_points(Engine(), pl, 0.5, A, r)[0] for r in range(2)]
        pts = fr._merge_repeats(reps)
        m = {p["point"]["name"]: p["twin_model"] for p in pts}
        self.assertEqual((m["d1"]["step_ms"], m["d2"]["step_ms"]),
                         (10.0, 20.0))
        self.assertEqual((m["m2"]["step_ms"], m["m2"]["chunk_ms"]),
                         (1030.0, 1010.0))
        r = fr._mix_rollup(pl, pts)
        self.assertAlmostEqual(r["busy_twin_s"],
                               (1000 * 10 + 400 * 20 + 10 * 1030) / 1e3)
        window = fr._time_points(Engine(), pl, 0.5, A, 0)[1]
        self.assertEqual([w[0] for w in window],
                         ["d1/decode", "m2/mixed/chunk0", "m2/decode"])
        # the sample-stage warm-up covers every TIMED shape, profiled or not (d2 is not)
        self.assertEqual([(x[0], x[2]) for x in fr._plan_shapes(pl, 0.5)],
                         [("d1/decode", 16), ("d2/decode", 16),
                          ("m2/mixed/chunk0", 4096), ("m2/decode", 16)])


class MergeRepeats(unittest.TestCase):

    def test_median_of_repeat_medians_and_all_kept(self):
        from core.forward_run import _merge_repeats

        def rep(m, d):
            return [{
                "point": {
                    "name": "P"
                },
                "twin_model": {},
                "mixed_ms": [{
                    "chunk": 0,
                    "median_after_first": m,
                    "all": [m + 5, m, m]
                }],
                "decode_ms": {
                    "median_after_first": d,
                    "all": [d, d, d]
                },
                "last_chunk_ms":
                None
            }]

        out = _merge_repeats(
            [rep(300.0, 4.0),
             rep(310.0, 5.0),
             rep(305.0, 4.5)])
        x = out[0]["mixed_ms"][0]
        self.assertEqual(x["median_after_first"], 305.0)
        self.assertEqual(x["by_repeat"], [300.0, 310.0, 305.0])
        self.assertEqual(len(x["all"]), 3)
        self.assertAlmostEqual(x["spread_pct"], 100 * 10 / 305, places=1)
        self.assertEqual(out[0]["decode_ms"]["median_after_first"], 4.5)


class TokenPaddings(unittest.TestCase):

    def test_runner_rule(self):
        from core.steps import bucket, token_paddings
        self.assertEqual(token_paddings(4096),
                         [16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
        self.assertEqual(token_paddings(5000)[-2:],
                         [4096, 5000])  # last bucket = mnbt
        self.assertEqual(
            token_paddings(4096, {"TPU_TOKEN_BUCKET_EXTRA": "24,9000"})[:3],
            [16, 24, 32])
        self.assertEqual(token_paddings(4096, compile_sizes=[4096, 32]),
                         [32, 4096])
        self.assertEqual(bucket(28, token_paddings(4096)), 32)
        self.assertEqual(bucket(4096, token_paddings(4096)), 4096)
        with self.assertRaises(ValueError):
            bucket(4097, token_paddings(4096))


if __name__ == "__main__":
    unittest.main()
