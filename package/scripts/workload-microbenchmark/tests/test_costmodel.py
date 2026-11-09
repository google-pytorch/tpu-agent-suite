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
"""CPU tests for core/costmodel.py and core/project.py: cost model, drafter busy time and device basis.

Run from scripts/workload-microbenchmark/: `python3 -m unittest discover tests`.
"""

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

from core import jobdata  # noqa: E402


class CostModel(unittest.TestCase):
    CELLS = {
        "d1": {
            "class": "D1-2",
            "kind": "decode",
            "steps": 1000,
            "share": 0.6
        },
        "d4": {
            "class": "D3-4",
            "kind": "decode",
            "steps": 200,
            "share": 0.2
        },
        "m": {
            "class": "mixed",
            "kind": "mixed",
            "steps": 10,
            "share": 0.2
        }
    }

    @staticmethod
    def _t(d1, d4, m, jitter=0.0):
        return {
            "d1": {
                "median_ms": d1,
                "by_repeat": [d1, d1 * (1 + jitter)]
            },
            "d4": {
                "median_ms": d4,
                "by_repeat": [d4, d4]
            },
            "m": {
                "median_ms": m,
                "by_repeat": [m, m]
            }
        }

    def test_t_mix_is_the_weighted_sum(self):
        from core.costmodel import evaluate
        e = evaluate(self.CELLS, self._t(10.0, 20.0, 1000.0))
        self.assertAlmostEqual(e["t_mix_s"],
                               (1000 * 10 + 200 * 20 + 10 * 1000) / 1e3)
        self.assertEqual(e["missing"], [])
        self.assertAlmostEqual(e["covered_share"], 1.0)

    def test_compare_speedup_classes_guardrails_and_noise(self):
        from core.costmodel import compare, evaluate
        base = evaluate(self.CELLS, self._t(10.0, 20.0, 1000.0))
        cand = evaluate(self.CELLS,
                        self._t(8.0, 20.0,
                                1100.0))  # decode faster, mixed slower
        c = compare(self.CELLS, base, cand, busy_s=90.0, idle_s=10.0)
        self.assertAlmostEqual(c["speedup"], 24.0 / 23.0, places=4)
        self.assertAlmostEqual(c["by_class_delta_s"]["D1-2"], 2.0)
        self.assertAlmostEqual(c["by_class_delta_s"]["mixed"], -1.0)
        roles = {g["cell"]: g for g in c["guardrails"]}
        self.assertEqual(set(roles), {"d1", "m"})  # heaviest decode + mixed
        self.assertTrue(roles["m"]["regressed"])
        self.assertAlmostEqual(c["covered_share"], 1.0)
        self.assertAlmostEqual(c["served_output_gain_modelled"],
                               100.0 / (90.0 / (24.0 / 23.0) + 10.0),
                               places=4)
        self.assertIn("covered_share / speedup + 1 - covered_share",
                      c["served_gain_rule"])
        self.assertTrue(c["significant"])
        noisy = evaluate(self.CELLS, self._t(8.0, 20.0, 1100.0, jitter=0.5))
        self.assertFalse(
            compare(self.CELLS, base, noisy, 90.0, 10.0)["significant"])

    def test_compare_partial_coverage_leaves_unmeasured_busy_unchanged(self):
        from core.costmodel import compare, evaluate
        tb = self._t(10.0, 20.0, 1000.0)
        tc = self._t(5.0, 10.0, 1000.0)
        del tb["m"], tc["m"]  # d1 (0.6) + d4 (0.2) = 0.8 covered, 2.0x speedup
        base = evaluate(self.CELLS, tb)
        cand = evaluate(self.CELLS, tc)
        c = compare(self.CELLS, base, cand, busy_s=90.0, idle_s=10.0)
        self.assertEqual(c["cells_compared"], 2)
        self.assertAlmostEqual(c["covered_share"], 0.8)
        self.assertAlmostEqual(c["speedup"], 2.0)
        # busy_after = 90 * (0.8 / 2.0 + 0.2) = 54.0 -> 100 / (54 + 10) = 1.5625
        # (vs full-extrapolation 100 / (45 + 10) = 1.8182)
        self.assertAlmostEqual(c["served_output_gain_modelled"],
                               100.0 / (90.0 * (0.8 / 2.0 + 0.2) + 10.0),
                               places=4)

    def test_a_missing_cell_is_listed_not_zero(self):
        from core.costmodel import evaluate
        t = self._t(10.0, 20.0, 1000.0)
        del t["m"]
        e = evaluate(self.CELLS, t)
        self.assertEqual(e["missing"], ["m"])
        self.assertAlmostEqual(e["covered_share"], 0.8)

    @staticmethod
    def _xsteps(label, per_kind, counts, before=0.5, after=0.0):
        return {
            "label": label,
            "projection": {
                "per_kind_ms": per_kind,
                "real_counts": counts
            },
            "before": {
                "ms": before
            },
            "after": {
                "ms": after
            }
        }

    # d1 and d2 share a class (only d1 is profiled); d4 is its own class, never profiled
    SCELLS = {
        "d1": {
            "class": "D1-2",
            "kind": "decode",
            "steps": 1000,
            "share": 0.5
        },
        "d2": {
            "class": "D1-2",
            "kind": "decode",
            "steps": 300,
            "share": 0.1
        },
        "d4": {
            "class": "D3-4",
            "kind": "decode",
            "steps": 200,
            "share": 0.2
        },
        "m": {
            "class": "mixed",
            "kind": "mixed",
            "steps": 10,
            "share": 0.2
        }
    }

    @staticmethod
    def _st(d1, d2, d4, m):
        return {
            n: {
                "median_ms": v,
                "by_repeat": [v, v]
            }
            for n, v in (("d1", d1), ("d2", d2), ("d4", d4), ("m", m))
        }

    @staticmethod
    def _xsteps(label, per_kind, counts):
        return {
            "label": label,
            "projection": {
                "per_kind_ms": per_kind,
                "real_counts": counts
            }
        }

    def test_served_basis_scales_layers_not_the_fixed_part(self):
        """A per-layer saving is n_layers x larger served; a fixed (per-step) one is not. The
        twin's wall ratio would score both the same."""
        from core.costmodel import cell_profiles, evaluate, served_times
        counts = {"a": 3, "b": 57}  # from the model, not here
        xs = [
            self._xsteps("d1/decode", {
                "a": 0.4,
                "b": 0.6
            }, counts),
            self._xsteps("m/mixed/chunk0", {
                "a": 20.0,
                "b": 30.0
            }, counts),
            self._xsteps("m/decode", {
                "a": 9.0,
                "b": 9.0
            }, counts)
        ]  # not m's step
        pf = cell_profiles(xs, self.SCELLS)
        self.assertEqual(set(pf), {"d1", "m"})
        self.assertEqual(len(pf["m"]), 1)
        st = served_times(self.SCELLS, self._st(3.0, 3.5, 4.0, 60.0), pf)
        # own profile: fixed = wall - layers = 2.0; served = 2.0 + 3*0.4 + 57*0.6
        self.assertAlmostEqual(st["d1"]["median_ms"],
                               2.0 + 1.2 + 34.2,
                               places=3)
        self.assertEqual(st["d1"]["profile"], "own")
        # same class, no profile: d1's fixed, d1's served/twin layer ratio on the rest
        k = (1.2 + 34.2) / 1.0
        self.assertAlmostEqual(st["d2"]["median_ms"], 2.0 + 1.5 * k, places=3)
        self.assertEqual(st["d2"]["profile"], "via d1")
        self.assertAlmostEqual(st["m"]["median_ms"],
                               10.0 + 60.0 + 1710.0,
                               places=3)
        # a 0.1 ms saving in d2's layer body is worth 0.1 x k served
        b = served_times(self.SCELLS, self._st(3.0, 3.4, 4.0, 60.0),
                         pf)["d2"]["median_ms"]
        self.assertAlmostEqual(st["d2"]["median_ms"] - b, 0.1 * k, places=3)
        # a class with no profiled cell is missing on this basis, never borrowed across classes
        self.assertNotIn("d4", st)
        self.assertEqual(evaluate(self.SCELLS, st)["missing"], ["d4"])

    def test_repeat_jitter_is_not_scaled_by_the_layer_ratio(self):
        """A repeat 0.1 ms slower than the median is 0.1 ms slower served, not 0.1 x k: the
        repeats share the profiled layers, so their difference is per-step (host) time."""
        from core.costmodel import cell_profiles, served_times
        counts = {"a": 3, "b": 57}
        pf = cell_profiles(
            [self._xsteps("d1/decode", {
                "a": 0.4,
                "b": 0.6
            }, counts)], self.SCELLS)
        t = {"d1": {"median_ms": 3.0, "by_repeat": [3.0, 3.1]}}
        st = served_times(self.SCELLS, t, pf)["d1"]
        self.assertAlmostEqual(st["median_ms"], 2.0 + 1.2 + 34.2, places=3)
        self.assertAlmostEqual(st["by_repeat"][1] - st["by_repeat"][0],
                               0.1,
                               places=3)

    def test_the_drafter_is_in_the_step_once_and_not_scaled(self):
        """Under MTP the drafter's propose replay follows every served step, so it belongs in
        T_mix. It is one layer served and in the twin: it sits in `fixed`, never times n_k."""
        from core.costmodel import (cell_profiles, cell_times, compare,
                                    evaluate, served_times)
        pts = [{
            "point": {
                "name": "d1"
            },
            "twin_model": {
                "step_ms": 3.0
            },
            "mixed_ms": [],
            "decode_ms": {
                "median_after_first": 3.0,
                "by_repeat": [3.0, 3.2],
                "draft_ms": {
                    "median_after_first": 4.0,
                    "by_repeat": [4.0, 4.4]
                }
            }
        }, {
            "point": {
                "name": "m"
            },
            "twin_model": {
                "step_ms": 60.0
            },
            "mixed_ms": [{
                "median_after_first": 60.0,
                "by_repeat": [60.0, 60.0],
                "draft_ms": {
                    "median_after_first": 20.0,
                    "by_repeat": [20.0, 20.0]
                }
            }]
        }]
        t = cell_times(pts)
        self.assertAlmostEqual(t["d1"]["median_ms"], 7.0)
        self.assertEqual(t["d1"]["by_repeat"], [7.0, 7.6])
        self.assertEqual((t["d1"]["backbone_ms"], t["d1"]["draft_ms"]),
                         (3.0, 4.0))
        counts = {"a": 3, "b": 57}
        pf = cell_profiles([
            self._xsteps("d1/decode", {
                "a": 0.4,
                "b": 0.6
            }, counts),
            self._xsteps("m/mixed/chunk0", {
                "a": 20.0,
                "b": 30.0
            }, counts)
        ], self.SCELLS)
        st = served_times(self.SCELLS, t, pf)
        # fixed = (3.0 backbone + 4.0 drafter) - 1.0 layers = 6.0; served = 6.0 + 1.2 + 34.2
        self.assertAlmostEqual(st["d1"]["median_ms"],
                               6.0 + 1.2 + 34.2,
                               places=3)
        # the same backbone without the drafter is exactly the drafter's 4.0 ms cheaper
        nod = cell_times([{
            **pts[0], "decode_ms": {
                k: v
                for k, v in pts[0]["decode_ms"].items() if k != "draft_ms"
            }
        }])
        self.assertIsNone(nod["d1"]["draft_ms"])
        self.assertAlmostEqual(
            st["d1"]["median_ms"] -
            served_times(self.SCELLS, nod, pf)["d1"]["median_ms"],
            4.0,
            places=3)
        # comparing an arm with the drafter against one without is flagged
        a, b = evaluate(self.SCELLS,
                        st), evaluate(self.SCELLS,
                                      served_times(self.SCELLS, nod, pf))
        self.assertTrue(
            compare(self.SCELLS, b, a, None, None)["drafter_mismatch"])
        self.assertFalse(
            compare(self.SCELLS, a, a, None, None)["drafter_mismatch"])

    def test_pooled_profiles_take_the_per_kind_median_and_measure_their_spread(
            self):
        """One profile is one step; the served basis multiplies its per-layer wobble by the
        layer count, so a second profile of the same cell must pool, and its disagreement
        must show up as noise."""
        from core.costmodel import cell_profiles, profile_spread, served_times
        counts = {"a": 3, "b": 57}
        pf = cell_profiles([
            self._xsteps("d1/decode", {
                "a": 0.4,
                "b": 0.6
            }, counts),
            self._xsteps("m/mixed/chunk0", {
                "a": 20.0,
                "b": 30.0
            }, counts)
        ], self.SCELLS)
        # job "j2" profiled d1 three times in one boot: identical steps, one sample, not three
        cell_profiles(
            [self._xsteps("d1/decode", {
                "a": 0.4,
                "b": 0.8
            }, counts)] * 3, self.SCELLS, pf, "j2")
        cell_profiles(
            [self._xsteps("d1/decode", {
                "a": 0.4,
                "b": 0.5
            }, counts)], self.SCELLS, pf, "j3")
        self.assertEqual(len(pf["d1"]), 5)
        t = self._st(3.0, 3.5, 4.0, 60.0)
        st = served_times(self.SCELLS, t, pf)
        self.assertEqual(st["d1"]["profile_jobs"], 3)
        # median over jobs of per-job medians {0.6, 0.8, 0.5} = 0.6 (a per-step median would
        # let one boot's three copies win: 0.8)
        self.assertEqual(st["d1"]["per_kind_ms"], {"a": 0.4, "b": 0.6})
        ps = profile_spread(self.SCELLS, t, pf)
        self.assertEqual(ps["profiled_jobs_max"], 3)
        self.assertGreater(ps["spread_pct"], 0)


class DrafterBusy(unittest.TestCase):
    CELLS = {
        "d1": {
            "class": "D1-2",
            "kind": "decode",
            "steps": 1000,
            "share": 0.5
        }
    }

    def test_busy_is_the_union_of_ops(self):
        from core.xprof_steps import _busy_ns
        self.assertEqual(_busy_ns([(0, 10, "a"), (5, 10, "b"), (100, 5, "c")]),
                         20)

    def test_served_basis_takes_drafter_busy_not_replay_wall(self):
        from core.costmodel import (cell_profiles, cell_times, evaluate,
                                    served_times)
        pts = [{
            "point": {
                "name": "d1"
            },
            "twin_model": {
                "step_ms": 3.0
            },
            "mixed_ms": [],
            "decode_ms": {
                "median_after_first": 3.0,
                "by_repeat": [3.0, 3.2],
                "draft_ms": {
                    "median_after_first": 4.0,
                    "by_repeat": [4.0, 4.4]
                }
            }
        }]
        t = cell_times(pts)
        counts = {"a": 3, "b": 57}
        x = {
            "label": "d1/decode",
            "projection": {
                "per_kind_ms": {
                    "a": 0.4,
                    "b": 0.6
                },
                "real_counts": counts,
                "draft_busy_ms": 1.2
            }
        }
        st = served_times(self.CELLS, t, cell_profiles([x], self.CELLS))["d1"]
        # backbone fixed = 3.0 - 1.0 = 2.0; + 3 x 0.4 + 57 x 0.6 + drafter busy 1.2 (not 4.0 wall)
        self.assertAlmostEqual(st["median_ms"],
                               2.0 + 1.2 + 34.2 + 1.2,
                               places=3)
        self.assertEqual((st["draft_wall_ms"], st["draft_busy_ms"]),
                         (4.0, 1.2))
        self.assertEqual(st["drafter_basis"], "profile device busy (once)")
        # repeat jitter comes from the backbone repeats, unscaled
        self.assertAlmostEqual(st["by_repeat"][1] - st["by_repeat"][0],
                               0.2,
                               places=3)
        self.assertEqual(
            evaluate(self.CELLS, {"d1": st})["cells"]["d1"]["draft_busy_ms"],
            1.2)
        # a profile without drafter busy keeps the replay wall, labelled
        x2 = {
            "label": "d1/decode",
            "projection": {
                "per_kind_ms": {
                    "a": 0.4,
                    "b": 0.6
                },
                "real_counts": counts
            }
        }
        st2 = served_times(self.CELLS, t, cell_profiles([x2],
                                                        self.CELLS))["d1"]
        self.assertAlmostEqual(st2["median_ms"], 6.0 + 1.2 + 34.2, places=3)
        self.assertTrue(st2["drafter_basis"].startswith("twin replay wall"))

    def test_project_adds_drafter_busy_once(self):
        from core import xprof_steps as xs
        sp = {
            "before": {
                "ms": 0.1
            },
            "after": {
                "ms": 0.0
            },
            "draft_ms": 4.0,
            "draft_busy_ms": 1.1,
            "draft_modules": 12,
            "layers": [{
                "ms": 0.2
            }, {
                "ms": 0.3
            }]
        }
        p = xs.project(sp, {
            "twin": ["gdn", "attn"],
            "real": {
                "gdn": 45,
                "attn": 15
            }
        })
        self.assertAlmostEqual(p["served_step_ms"],
                               0.1 + 45 * 0.2 + 15 * 0.3 + 1.1,
                               places=3)


class DeviceBasis(unittest.TestCase):
    """`served_device`: every term of the served step from device time, and a significance test
    on per-profile T_mix samples."""
    CELLS = {
        "d1": {
            "class": "D1-2",
            "kind": "decode",
            "steps": 1000,
            "share": 0.5
        },
        "d2": {
            "class": "D1-2",
            "kind": "decode",
            "steps": 500,
            "share": 0.1
        },
        "m": {
            "class": "mixed",
            "kind": "mixed",
            "steps": 10,
            "share": 0.2
        }
    }

    @staticmethod
    def _x(label, b, gdn, attn, job="j1", draft=1.2, samp=0.65):
        return {
            "label": label,
            "projection": {
                "per_kind_ms": {
                    "gdn": gdn,
                    "attn": attn
                },
                "real_counts": {
                    "gdn": 45,
                    "attn": 15
                },
                "draft_busy_ms": draft,
                "sampling_busy_ms": samp,
                "before_busy_ms": b,
                "after_busy_ms": 0.05,
                "per_step_in_layers_ms": 0.1
            }
        }

    @staticmethod
    def _t(bb, reps=None):
        return {
            "median_ms": bb + 4.0,
            "by_repeat": reps or [bb + 4.0, bb + 4.0],
            "backbone_ms": bb,
            "backbone_by_repeat": [bb, bb],
            "draft_ms": 4.0,
            "sampling_wall_ms": 1.0
        }

    def test_terms_and_no_host_wall(self):
        from core.costmodel import cell_profiles, evaluate, served_device_times
        pf = cell_profiles([self._x("d1/decode", 0.08, 0.115, 0.16)] * 1 +
                           [self._x("d1/decode", 0.08, 0.117, 0.16)] +
                           [self._x("d1/decode", 0.08, 0.116, 0.16)],
                           self.CELLS)
        # the host wall of the backbone jitters by 0.3 ms between repeats: irrelevant here
        t = {"d1": self._t(3.0, [7.0, 7.3]), "d2": self._t(6.0)}
        dv = served_device_times(self.CELLS, t, pf)
        exp = 0.08 + 0.05 + 0.1 + 45 * 0.116 + 15 * 0.16 + 1.2 + 0.65
        self.assertAlmostEqual(dv["d1"]["median_ms"], exp, places=4)
        self.assertEqual(dv["d1"]["parts_ms"]["sampling"], 0.65)
        self.assertEqual(len(dv["d1"]["by_repeat"]),
                         3)  # one sample per profile
        # d2 has no profile: through d1, scaled by the backbone wall ratio 6/3, labelled
        self.assertAlmostEqual(dv["d2"]["median_ms"], 2 * exp, places=3)
        self.assertTrue(dv["d2"]["profile"].startswith("via d1"))
        e = evaluate(self.CELLS, dv)
        self.assertEqual(e["missing"], ["m"])
        self.assertEqual(len(e["t_mix_by_repeat_s"]), 3)
        # an old xprof_steps.json without before/after busy: no device sample, cell missing
        old = {
            "label": "d1/decode",
            "projection": {
                "per_kind_ms": {
                    "gdn": 0.1,
                    "attn": 0.1
                },
                "real_counts": {
                    "gdn": 45,
                    "attn": 15
                }
            }
        }
        self.assertEqual(
            served_device_times(self.CELLS, t, cell_profiles([old],
                                                             self.CELLS)), {})

    def test_t_distribution_p_values(self):
        from core.costmodel import t_two_sided_p
        self.assertAlmostEqual(t_two_sided_p(2.776445, 4), 0.05, places=5)
        self.assertAlmostEqual(t_two_sided_p(2.228139, 10), 0.05, places=5)
        self.assertAlmostEqual(t_two_sided_p(1.959964, 1e7), 0.05, places=4)
        self.assertAlmostEqual(t_two_sided_p(0.0, 5), 1.0, places=6)

    def test_significance_finds_a_small_win_and_not_an_aa(self):
        from core.costmodel import significance
        base = [100.0, 100.2, 99.9, 100.1, 99.8]
        cand = [x * 0.985 for x in [100.1, 99.9, 100.0, 100.2, 99.85]
                ]  # -1.5%, same noise
        s = significance(base, cand)
        self.assertTrue(s["significant"])
        self.assertLess(s["p_two_sided"], 1e-4)
        lo, hi = s["ci95_speedup"]
        self.assertTrue(lo < s["speedup_of_means"] < hi and lo > 1.0)
        aa = significance(base, [100.05, 99.95, 100.1, 99.9, 100.0])
        self.assertFalse(aa["significant"])
        # a boot offset the within-boot samples cannot see: p is tiny, the floor refuses it
        off = significance(base, [x * 0.996 for x in base])
        self.assertLess(off["p_two_sided"], 0.05)
        self.assertEqual(off["boot_noise"], "unmeasured")
        self.assertFalse(
            significance(base, [x * 0.996 for x in base],
                         floor_pct=0.5)["significant"])
        self.assertTrue(significance(base, cand, floor_pct=0.5)["significant"])
        self.assertIsNone(significance([1.0], [1.0, 2.0])["significant"])

    def test_mix_rollup_writes_and_prints_the_device_basis(self):
        import argparse
        import contextlib
        import io
        import tempfile

        from core import cli
        cells = {
            "d1": {
                "name": "d1",
                "class": "D1-2",
                "kind": "decode",
                "steps": 1000,
                "share": 0.5
            }
        }
        with tempfile.TemporaryDirectory() as root:
            plan = os.path.join(root, "p.steps.json")
            json.dump(
                {
                    "points": [{
                        "point": {
                            "name": "d1"
                        },
                        "cell": cells["d1"]
                    }],
                    "mix": {
                        "busy_s": 100.0,
                        "idle_s": 10.0
                    }
                }, open(plan, "w"))

            def job(jid, attn):
                d = os.path.join(root, jid, "outputs", "x", "wmb", "forward")
                os.makedirs(d)
                pt = {
                    "point": {
                        "name": "d1"
                    },
                    "twin_model": {
                        "step_ms": 3.0
                    },
                    "mixed_ms": [],
                    "decode_ms": {
                        "median_after_first": 3.0,
                        "by_repeat": [3.0, 3.3],
                        "sampling_ms": {
                            "median_after_first": 1.0,
                            "by_repeat": [1, 1]
                        }
                    }
                }
                json.dump({"points": [pt]},
                          open(os.path.join(d, "result.json"), "w"))
                # 5 profiles with +-0.4% per-layer wobble; the candidate's attention layer is
                # 4% faster and its sampler 0.5 ms cheaper
                steps = [
                    self._x("d1/decode",
                            0.08,
                            0.116,
                            attn * (1 + 0.002 * w),
                            samp=0.65 if attn > 0.16 else 0.15)
                    for w in (-2, -1, 0, 1, 2)
                ]
                json.dump({"steps": steps},
                          open(os.path.join(d, "xprof_steps.json"), "w"))

            job("j-base", 0.162)
            job("j-cand", 0.155)
            old = jobdata.DEFAULT_ROOT
            jobdata.DEFAULT_ROOT = root
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    rc = cli.cmd_mix_rollup(
                        argparse.Namespace(plans=plan,
                                           jobs=["j-cand"],
                                           baseline="j-base",
                                           profiles=None,
                                           baseline_profiles=None,
                                           json=os.path.join(root, "cm.json"),
                                           noise_floor_pct=0.4))
            finally:
                jobdata.DEFAULT_ROOT = old
            self.assertEqual(rc, 0)
            out = json.load(open(os.path.join(root, "cm.json")))
            self.assertEqual(
                out["primary_for"]["kernel / device-time changes"],
                "served_device")
            c = out["served_device"]["compare"]
            self.assertGreater(c["speedup"], 1.0)
            self.assertTrue(c["significance"]["significant"])
            self.assertEqual(c["significance"]["boot_noise_floor_pct"], 0.4)
            self.assertEqual(
                out["served_device"]["baseline"]["boot_spread"]
                ["t_mix_by_job_s"].keys(), {"j-base"})
            self.assertEqual(
                out["served_device"]["candidate"]["cells"]["d1"]["parts_ms"]
                ["sampling"], 0.15)
            self.assertIn(
                "[served_device]  <- primary for kernel / device-time changes",
                buf.getvalue())
            self.assertIn("Welch t", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
