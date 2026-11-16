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
"""CPU tests for the sampling stage: core/cli.py, core/costmodel.py and the runner glue.

Run from scripts/workload-microbenchmark/: `python3 -m unittest discover tests`.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

from core import jobdata  # noqa: E402


class SamplingStage(unittest.TestCase):
    """The serving step's sample stage in the twin step: plan block, layout, profile booking and
    the cost model's once-per-step rule."""
    LOG = (
        "(APIServer pid=385) WARNING 10-01 18:38:58 [model.py:1759] Default vLLM sampling "
        "parameters have been overridden by the model's `generation_config.json`: "
        "`{'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}`. If this is not intended, please "
        "relaunch vLLM instance with `--generation-config vllm`.")

    def test_server_log_line_parses(self):
        self.assertEqual(jobdata.parse_sampling_line(self.LOG), {
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95
        })
        self.assertIsNone(
            jobdata.parse_sampling_line("INFO something else {'a': 1}"))

    def test_plan_block_defaults_on_for_non_greedy_and_flag_turns_it_off(self):
        from core.cli import sampling_block

        class M:
            SAMPLING = {
                "temperature": 0.6,
                "top_k": 20,
                "top_p": 0.95,
                "source": "model.py"
            }

        class NoM:
            pass

        snap = {
            "job_id": "j-none",
            "sampling_defaults": {
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.8,
                "source": "log"
            }
        }
        b = sampling_block(snap, M, None, job_root="/nonexistent")
        self.assertEqual((b["run"], b["temperature"], b["top_k"], b["source"]),
                         (True, 0.7, 40, "log"))
        self.assertFalse(
            sampling_block(snap, M, False, job_root="/nonexistent")["run"])
        # no snapshot line, no synced log: the model's default, labelled
        b = sampling_block({"job_id": "j-none"},
                           M,
                           None,
                           job_root="/nonexistent")
        self.assertEqual((b["run"], b["top_p"], b["source"]),
                         (True, 0.95, "model.py"))
        # greedy job: on, so the LM head and argmax stay in every step
        g = sampling_block({"sampling_defaults": {
            "temperature": 0.0
        }}, NoM, None)
        self.assertTrue(g["run"])
        self.assertTrue(g["greedy"])
        # --greedy: clients at temperature 0 against a server whose defaults sample
        g = sampling_block(snap, M, None, job_root="/nonexistent", greedy=True)
        self.assertEqual((g["run"], g["greedy"], g["temperature"]),
                         (True, True, 0.0))
        with self.assertRaises(SystemExit):
            sampling_block(snap, M, False, greedy=True)
        # unknown params: off, and --sampling is refused
        self.assertFalse(sampling_block({}, NoM, None)["run"])
        with self.assertRaises(SystemExit):
            sampling_block({}, NoM, True)

    def test_verify_layout_matches_the_runner_padding(self):
        from core import forward_run as fr
        pads = [4, 8, 16, 16384]
        # B=1 MTP-3 decode: 4 logits rows (token bucket 4), bonus rows padded to 8 requests,
        # target rows 4 -- the [8, V] bonus and [4, V] rejection calls of the served trace
        lay = fr._sampling_layout([[4, 95502]], 4, pads, 8)
        self.assertEqual((lay["path"], lay["logits_rows"], lay["bonus_rows"],
                          lay["target_rows"]), ("spec_verify", 4, 8, 4))
        self.assertEqual(lay["num_draft_tokens"], [3])
        # mixed: two verify windows + a chunk with no drafts -> 4 + 4 + 1 = 9 rows -> bucket 16
        lay = fr._sampling_layout([[4, 100], [4, 100], [16376, 20000]], 4,
                                  pads, 8)
        self.assertEqual((lay["num_draft_tokens"], lay["logits_rows"]),
                         ([3, 3, 0], 16))
        # no MTP: one row per request, padded to 8
        lay = fr._sampling_layout([[1, 100], [1, 100]], 1, pads, 8)
        self.assertEqual((lay["path"], lay["logits_rows"]),
                         ("sample_from_logits", 8))
        self.assertEqual(
            fr._sampling_layout([[1, 1]] * 9, 1, pads, 16)["padded_num_reqs"],
            16)

    def test_markers_agree_between_pod_and_profile_reader(self):
        from core import forward_run as fr
        from core import xprof_steps as xs
        self.assertEqual(fr.SAMPLING_MARKS, xs.SAMPLING_MARKS)
        self.assertEqual(
            xs._marker([
                (0, 1, "%fusion.3 = s32[7919]{0} fusion(s32[7919]{0} %p)")
            ]), "mark")
        self.assertEqual(
            xs._marker([
                (0, 1, "%add.1 = s32[7907]{0:T(1024)} add(s32[7907] %a)")
            ]), "mark")
        self.assertIsNone(
            xs._marker([(0, 1, "%f = s32[7919,2]{1,0} fusion(%p)")]))

    @staticmethod
    def _mods(with_marks: bool):
        bb = [(0, 10, "%all-reduce.1 = bf16[4,8] all-reduce(%x)")]

        def mk(c):
            return [(0, 1, f"%m = s32[{c}]{{0}} fusion(s32[{c}] %b)")]

        def op(t0, d):
            return [(t0, d, "%f = f32[4,248320] fusion(%x)")]

        mods = [(0, 100, "fwd", bb)]
        if with_marks:
            mods.append((110, 111, "mark", mk(7919)))
        mods += [(120, 150, "samp1", op(120, 30)),
                 (160, 180, "samp2", op(160, 20))]
        if with_marks:
            mods.append((190, 191, "mark", mk(7907)))
        mods += [(300, 340, "draft1", op(300, 40)), (400, 500, "fwd", bb),
                 (510, 520, "draft-next", op(510, 10))]
        return mods

    def test_profile_books_the_stage_between_markers_apart_from_the_drafter(
            self):
        from core import xprof_steps as xs

        def is_step(name, o, n):
            return name == "fwd"

        st = xs._assign(self._mods(True), is_step, xs._marker)
        self.assertEqual(len(st), 2)
        self.assertEqual(
            (st[0]["sampling_modules"], st[0]["sampling_busy_ns"]), (2, 50))
        self.assertEqual((st[0]["draft_modules"], st[0]["draft_busy_ns"]),
                         (1, 40))
        self.assertEqual(st[0]["sampling_marks"], 2)
        self.assertEqual((st[1]["sampling_modules"], st[1]["draft_busy_ns"]),
                         (0, 10))
        # negative control: without the markers the same modules all land in the drafter
        st = xs._assign(self._mods(False), is_step, xs._marker)
        self.assertEqual((st[0]["sampling_modules"], st[0]["draft_modules"]),
                         (0, 3))
        # torch-tpu ran both markers as one program, so the trace showed s32[7919] twice
        # (j-3e69c2dd): the second marker still closes the stage
        same = [(a, b, n, [(0, 1, "%m = s32[7919]{0} fusion(s32[7919] %b)")]
                 if n == "mark" else o) for a, b, n, o in self._mods(True)]
        st = xs._assign(same, is_step, xs._marker)
        self.assertEqual((st[0]["sampling_busy_ns"], st[0]["draft_busy_ns"]),
                         (50, 40))

    def test_project_counts_sampling_busy_once(self):
        from core import xprof_steps as xs
        sp = {
            "before": {
                "ms": 0.1,
                "busy_ms": 0.08
            },
            "after": {
                "ms": 0.05,
                "busy_ms": 0.04
            },
            "draft_ms":
            4.0,
            "draft_busy_ms":
            1.1,
            "draft_modules":
            12,
            "sampling_busy_ms":
            0.65,
            "sampling_modules":
            6,
            "layers": [{
                "ms": 0.2,
                "per_step_ms": 0.0
            }, {
                "ms": 0.3,
                "per_step_ms": 0.1
            }]
        }
        p = xs.project(sp, {
            "twin": ["gdn", "attn"],
            "real": {
                "gdn": 45,
                "attn": 15
            }
        })
        body = 45 * 0.2 + 15 * 0.2
        self.assertAlmostEqual(p["served_step_ms"],
                               0.15 + 0.1 + body + 1.1 + 0.65,
                               places=3)
        self.assertAlmostEqual(p["served_device_step_ms"],
                               0.12 + 0.1 + body + 1.1 + 0.65,
                               places=3)
        self.assertEqual(p["sampling_busy_ms"], 0.65)
        sp["sampling_modules"] = 0  # not run: no sampling term
        self.assertIsNone(
            xs.project(sp, {
                "twin": ["gdn", "attn"],
                "real": {
                    "gdn": 45,
                    "attn": 15
                }
            })["sampling_busy_ms"])

    def test_forward_run_times_the_stage_apart_from_the_forward(self):
        from core import forward_run as fr

        class Engine:

            def collective_rpc(self, fn, args):
                if fn is fr._last_inputs:
                    return [None]
                if fn is fr._last_sampling:  # per rank: rank 1 is the slower one
                    return [[0.7, 0.6, 0.6], [0.9, 0.65, 0.7]]
                self.args = args
                return [[5.0, 4.0, 4.0]] * 2

        e = Engine()
        old = fr.SAMPLING
        try:
            fr.SAMPLING = {
                "run": True,
                "temperature": 0.6,
                "top_k": 20,
                "top_p": 0.95
            }
            out = fr._time(e, [[4, 100]], 4, 3)
        finally:
            fr.SAMPLING = old
        self.assertEqual(e.args[4]["top_k"],
                         20)  # the plan block reaches _run_batch
        self.assertEqual(out["median_after_first"], 4.0)  # the forward alone
        self.assertEqual(out["sampling_ms"]["all"],
                         [0.9, 0.65, 0.7])  # max over ranks
        self.assertAlmostEqual(out["sampling_ms"]["median_after_first"], 0.675)
        m = fr._merge_repeats([[{
            "point": {
                "name": "d"
            },
            "mixed_ms": [],
            "last_chunk_ms": None,
            "decode_ms": dict(out),
            "twin_model": {}
        }]] * 2)
        self.assertEqual(m[0]["decode_ms"]["sampling_ms"]["by_repeat"],
                         [0.675, 0.675])

    def test_cost_model_twin_has_the_wall_served_has_the_busy_once(self):
        from core.costmodel import cell_profiles, cell_times, served_times
        cells = {
            "d1": {
                "class": "D1-2",
                "kind": "decode",
                "steps": 1000,
                "share": 0.5
            }
        }
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
                },
                "sampling_ms": {
                    "median_after_first": 1.5,
                    "by_repeat": [1.5, 1.6]
                }
            }
        }]
        t = cell_times(pts)["d1"]
        self.assertAlmostEqual(t["median_ms"], 8.5)
        self.assertEqual([round(x, 3) for x in t["by_repeat"]], [8.5, 9.2])
        self.assertEqual((t["backbone_ms"], t["sampling_wall_ms"]), (3.0, 1.5))
        x = {
            "label": "d1/decode",
            "projection": {
                "per_kind_ms": {
                    "a": 0.4,
                    "b": 0.6
                },
                "real_counts": {
                    "a": 3,
                    "b": 57
                },
                "draft_busy_ms": 1.2,
                "sampling_busy_ms": 0.65
            }
        }
        st = served_times(cells, {"d1": t}, cell_profiles([x], cells))["d1"]
        # fixed from the backbone wall alone (3.0 - 1.0); + layers + drafter busy + sampling busy
        self.assertAlmostEqual(st["median_ms"],
                               2.0 + 1.2 + 34.2 + 1.2 + 0.65,
                               places=3)
        self.assertEqual((st["sampling_wall_ms"], st["sampling_busy_ms"]),
                         (1.5, 0.65))


class SamplingGlue(unittest.TestCase):
    """`_prepare_sampling` + `_sample_stage` against a fake runner with each image's signatures:
    0fcc1f696 (`_build_padded_sampling_params` -> 3 tensors, `sample_from_logits` without
    k_bound) and c8ee856e5 (4: + the static k_bound, passed on). Needs torch (skipped without)."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def _runner(self, with_k_bound: bool, spec: bool):
        import types

        import torch
        calls = []

        class MD:

            def __init__(self, nd, cu, preqs):
                rows = int((nd + 1).sum())
                plen = next(p for p in (4, 8, 16, 16384) if p >= rows)
                self.final_logits_indices = torch.zeros(plen,
                                                        dtype=torch.int32)
                self.bonus_logits_indices = torch.zeros(preqs,
                                                        dtype=torch.int32)
                self.target_logits_indices = torch.zeros(plen,
                                                         dtype=torch.int32)

        class R:
            vocab_size, max_num_reqs, device = 1000, 8, "cpu"
            input_ids_cpu = torch.zeros(64, dtype=torch.int32)
            input_batch = types.SimpleNamespace(
                temperature_cpu_tensor=torch.ones(8),
                top_k_cpu_tensor=torch.zeros(8, dtype=torch.int32),
                top_p_cpu_tensor=torch.ones(8))
            rejection_sampler = object() if spec else None
            spec_decode_manager = types.SimpleNamespace(
                get_spec_decode_metadata=lambda nd, cu, preqs: (calls.append(
                    ("md", nd.tolist(), cu.tolist(), preqs)), MD(
                        nd, cu, preqs))[1])

            def _build_padded_sampling_params(self, a, b, logits):
                n = logits.shape[0]
                t = self.input_batch.temperature_cpu_tensor[a:b]
                out = (torch.ones(n, 1), torch.zeros(n, 1, dtype=torch.int32),
                       torch.ones(n, 1))
                calls.append(("params", a, b, n, round(float(t[0]), 3)))
                return out + (64, ) if with_k_bound else out

            def compute_selected_logits(self, h, idx):
                calls.append(("logits", tuple(h.shape), int(idx.shape[0])))
                return torch.zeros(int(idx.shape[0]), self.vocab_size)

            def _get_sampling_generator(self):
                return torch.Generator()

            def _greedy_sample(self, logits):
                calls.append(("greedy", tuple(logits.shape)))
                return torch.zeros(logits.shape[0], 1, dtype=torch.int64)

            def _sample_spec_verify_chunk(self, logits, md, grammar, sched, a,
                                          b, greedy, gen):
                calls.append(
                    ("verify", tuple(logits.shape), a, b, greedy, grammar))
                p = self._build_padded_sampling_params(
                    a, b, torch.zeros(md.bonus_logits_indices.shape[0], 1))
                return torch.zeros(b, 4, dtype=torch.int32)

            def sample_from_logits_func(self,
                                        logits,
                                        t,
                                        u,
                                        k,
                                        p,
                                        all_greedy=False,
                                        **kw):
                if kw and not with_k_bound:
                    raise TypeError("old image: unexpected keyword k_bound")
                calls.append(("sample", tuple(logits.shape), all_greedy,
                              kw.get("k_bound")))
                return torch.zeros(logits.shape[0], 1, dtype=torch.int64)

        return R(), calls

    def _fake_tr(self):
        import types
        planner = types.SimpleNamespace(
            maybe_select_logits_hidden_states=lambda h, p, i: None,
            finalize_hidden_states=lambda h, p: h)
        tr = types.SimpleNamespace(
            _get_padded_num_reqs_with_upper_limit=lambda x, u: min(
                8 if x <= 8 else 1 << (x - 1).bit_length(), u),
            _get_sequence_layout_planner_for_runner=lambda r: planner,
            synchronize_tensors=lambda t: None)
        pkg = types.ModuleType("vllm_torchtpu")
        run = types.ModuleType("vllm_torchtpu.runner")
        run.tpu_runner = tr
        return {
            "vllm_torchtpu": pkg,
            "vllm_torchtpu.runner": run,
            "vllm_torchtpu.runner.tpu_runner": tr
        }

    def test_both_images_spec_and_plain(self):
        from unittest import mock

        import torch
        from core import forward_run as fr
        samp = {"run": True, "temperature": 0.6, "top_k": 20, "top_p": 0.95}
        with mock.patch.dict(sys.modules, self._fake_tr()):
            for kb in (False, True):
                # MTP verify, B=1: logits on the 4 verify rows, the runner's verify chunk
                r, calls = self._runner(kb, spec=True)
                ids = torch.arange(4, dtype=torch.int32)
                st = fr._prepare_sampling(r, [[4, 100]], ids, 4, samp)
                self.assertEqual(st["rec"]["image_returns_k_bound"], kb)
                self.assertEqual(st["rec"]["k_bound"], 64 if kb else None)
                self.assertEqual(
                    (st["rec"]["logits_rows"], st["rec"]["bonus_rows"]),
                    (4, 8))
                self.assertEqual(int(r.input_batch.top_k_cpu_tensor[0]), 20)
                self.assertAlmostEqual(float(
                    r.input_batch.temperature_cpu_tensor[0]),
                                       0.6,
                                       places=5)
                self.assertEqual(calls[0], ("md", [3], [4], 8))
                fr._sample_stage(r, torch.zeros(4, 16), None, st)
                kinds = [c[0] for c in calls]
                self.assertIn("verify", kinds)
                self.assertEqual(next(c for c in calls if c[0] == "logits"),
                                 ("logits", (4, 16), 4))
                self.assertEqual(
                    next(c for c in calls if c[0] == "verify")[4:],
                    (False, None))
                # no drafts (no MTP): LM head on one row per request, then sample_from_logits,
                # k_bound passed only where the image returns it
                r, calls = self._runner(kb, spec=False)
                st = fr._prepare_sampling(r, [[1, 50], [1, 60]],
                                          torch.arange(2, dtype=torch.int32),
                                          1, samp)
                fr._sample_stage(r, torch.zeros(4, 16), None, st)
                self.assertEqual(next(c for c in calls if c[0] == "logits"),
                                 ("logits", (4, 16), 8))
                self.assertEqual(next(c for c in calls if c[0] == "sample"),
                                 ("sample",
                                  (8, 1000), False, 64 if kb else None))
        # greedy plan: the LM head, then the runner's argmax, never the top-k/top-p sampler
        greedy = {
            "run": True,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "greedy": True
        }
        with mock.patch.dict(sys.modules, self._fake_tr()):
            r, calls = self._runner(False, spec=False)
            st = fr._prepare_sampling(r, [[1, 50], [1, 60]],
                                      torch.arange(2, dtype=torch.int32), 1,
                                      greedy)
            del calls[:]
            fr._sample_stage(r, torch.zeros(4, 16), None, st)
            self.assertEqual([c[0] for c in calls], ["logits", "greedy"])
            r, calls = self._runner(False, spec=True)
            st = fr._prepare_sampling(r, [[4, 100]],
                                      torch.arange(4, dtype=torch.int32), 4,
                                      greedy)
            fr._sample_stage(r, torch.zeros(4, 16), None, st)
            self.assertTrue(next(c for c in calls if c[0] == "verify")[4])
        # top_k disabled (0 / -1) is stored as the vocab size, as vLLM's InputBatch does
        with mock.patch.dict(sys.modules, self._fake_tr()):
            r, _ = self._runner(False, spec=False)
            fr._prepare_sampling(r, [[1, 5]], torch.arange(1,
                                                           dtype=torch.int32),
                                 1, {
                                     **samp, "top_k": -1
                                 })
            self.assertEqual(int(r.input_batch.top_k_cpu_tensor[0]), 1000)


if __name__ == "__main__":
    unittest.main()
