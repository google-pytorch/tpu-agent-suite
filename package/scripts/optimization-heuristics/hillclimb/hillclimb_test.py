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
"""Unit tests for the `hillclimb` serving optimization gates and CLI."""

import contextlib
import dataclasses
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# pylint: disable=g-import-not-at-top,wrong-import-position
from hillclimb import code_gap  # noqa: E402
from hillclimb import code_limits  # noqa: E402
from hillclimb import early_verdict  # noqa: E402
from hillclimb import hillclimb_cli  # noqa: E402
from hillclimb import kv_replay  # noqa: E402
from hillclimb import ledger  # noqa: E402
from hillclimb import legs  # noqa: E402
from hillclimb import lessons  # noqa: E402
from hillclimb import metrics  # noqa: E402
from hillclimb import microbench  # noqa: E402
from hillclimb import model_plugins  # noqa: E402
from hillclimb import ports  # noqa: E402
from hillclimb import predictor  # noqa: E402
from hillclimb import preflight  # noqa: E402
from hillclimb import run_history  # noqa: E402
from hillclimb import trace_roofline  # noqa: E402
from hillclimb import verify  # noqa: E402

# Arbitrary rates for the arithmetic only: they are not prices or measurements.
_TEST_CHIP_HOUR_USD = 2.0
_TEST_VPU_RATE = 5.0

SAMPLE_AIPERF_BASELINE = """
│ Realtime Telemetry │
│ Profiling 120/500 │ 24.0% │ 00:10:30 │ 00:33:00 │
│ Requests │ 0.20 │ 0.19 │ 0.21 │ 0.20 │ 0.22 │ 0.23 │
│ Output   │ 400.0 │ 390.0 │ 410.0 │ 405.0 │ 420.0 │ 430.0 │
│ Total    │ 3200.0 │ 3100.0 │ 3300.0 │ 3250.0 │ 3400.0 │ 3500.0 │
│ TTFT     │ 850.0 │ 800.0 │ 900.0 │ 820.0 │ 1200.0 │ 1500.0 │
│ ITL      │ 24.0 │ 22.0 │ 26.0 │ 23.5 │ 28.0 │ 31.0 │
"""

SAMPLE_AIPERF_LOSER = """
│ Realtime Telemetry │
│ Profiling 90/500 │ 18.0% │ 00:10:30 │ 00:47:00 │
│ Requests │ 0.15 │ 0.14 │ 0.16 │ 0.15 │ 0.17 │ 0.18 │
│ Output   │ 300.0 │ 290.0 │ 310.0 │ 305.0 │ 320.0 │ 330.0 │
│ Total    │ 2406.4 │ 2300.0 │ 2500.0 │ 2420.0 │ 2550.0 │ 2600.0 │
│ TTFT     │ 1450.0 │ 1300.0 │ 1600.0 │ 1400.0 │ 2100.0 │ 2600.0 │
│ ITL      │ 32.0 │ 30.0 │ 35.0 │ 31.5 │ 39.0 │ 45.0 │
"""

SAMPLE_SERVER_METRICS_CSV = """endpoint_url,metric_name,avg,p50,p75,p90,max
http://prefill-0:8081/metrics,vllm:num_requests_waiting,0.05,0.0,0.0,0.0,1.0
http://prefill-0:8081/metrics,vllm:num_requests_running,0.28,0.0,1.0,1.0,1.0
http://decode-0:8082/metrics,vllm:num_requests_waiting,4.2,4.0,6.0,8.0,12.0
http://decode-0:8082/metrics,vllm:num_requests_running,15.8,16.0,16.0,16.0,16.0
"""

SAMPLE_ENVS_SOURCE = '''
environment_variables = {
    # Master switch for fused MoE kernel path.
    "ENABLE_FUSED_MOE_KERNEL": lambda: os.getenv("ENABLE_FUSED_MOE_KERNEL", "0") == "1",
    # Requires ENABLE_FUSED_MOE_KERNEL=1. Active for <= 64 tokens; allocates extra HBM buffer.
    "FUSED_MOE_FAST_SUBCHANNEL": lambda: os.getenv("FUSED_MOE_FAST_SUBCHANNEL", "0") == "1",
    # Hudson EP-only decode optimization; requires EP mode.
    "MOE_HUDSON_EP_OPT": lambda: os.getenv("MOE_HUDSON_EP_OPT", "0") == "1",
}
'''

SAMPLE_KERNEL_SOURCE = """
ROW_BUCKETS = (16, 32, 64)
MAX_DECODE_TOKENS = int(os.environ.get("MAX_DECODE_TOKENS", "128"))

def run_kernel(num_tokens: int) -> bool:
    if num_tokens > 64:
        return False
    return num_tokens in ROW_BUCKETS
"""


def _run_cli(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = hillclimb_cli.main(argv)
    return rc, buf.getvalue()


class MetricsAndHistoryTest(unittest.TestCase):

    def test_mtok_per_usd_and_compare_runs(self):
        val = metrics.mtok_per_usd(400.0, _TEST_CHIP_HOUR_USD)
        expected = (400.0 * 3600.0 / 1e6) / _TEST_CHIP_HOUR_USD
        self.assertAlmostEqual(val, expected, places=5)
        with self.assertRaises(ValueError):
            metrics.mtok_per_usd(400.0, 0.0)

        b = metrics.ServingMetrics(
            run_id="base",
            token_throughput_per_chip=320.0,
            p90_interactivity_tok_s=28.0,
            mean_ttft_ms=900.0,
            median_itl_ms=25.0,
        )
        c = metrics.ServingMetrics(
            run_id="cand",
            token_throughput_per_chip=360.0,
            p90_interactivity_tok_s=31.0,
            mean_ttft_ms=810.0,
            median_itl_ms=22.0,
        )
        delta = metrics.compare_runs(b, c)
        self.assertAlmostEqual(delta.tok_s_chip_pct_change, 12.5)
        self.assertEqual(delta.mtok_per_usd_pct_change,
                         delta.tok_s_chip_pct_change)
        self.assertIsNone(delta.baseline_mtok_per_usd)
        self.assertGreater(delta.p90_interactivity_pct_change, 10.0)
        # A chip-hour rate adds absolute Mtok/$ and leaves every delta alone.
        priced = metrics.compare_runs(b, c, _TEST_CHIP_HOUR_USD)
        self.assertEqual(priced.tok_s_chip_pct_change,
                         delta.tok_s_chip_pct_change)
        assert priced.baseline_mtok_per_usd is not None
        assert priced.candidate_mtok_per_usd is not None
        self.assertAlmostEqual(
            priced.candidate_mtok_per_usd / priced.baseline_mtok_per_usd,
            360.0 / 320.0)

    def test_strict_per_concurrency_noise_estimate(self):
        rows = [
            {
                "run_id": "rep-c32-a",
                "recipe": "glm5_1p1d",
                "status": "SUCCEEDED",
                "knobs": {
                    "DECODE_BATCH_CAP": "16",
                    "CLIENT_CONCURRENCIES": "32"
                },
                "metrics": {
                    "token_throughput_per_chip": 300.0,
                    "concurrency": 32
                },
            },
            {
                "run_id": "rep-c32-b",
                "recipe": "glm5_1p1d",
                "status": "SUCCEEDED",
                "knobs": {
                    "DECODE_BATCH_CAP": "16",
                    "CLIENT_CONCURRENCIES": "32",
                    "BENCHMARK_DURATION_SECONDS": "1800",
                },
                "metrics": {
                    "token_throughput_per_chip": 303.0,
                    "concurrency": 32
                },
            },
            {
                "run_id": "rep-c64-a",
                "recipe": "glm5_1p1d",
                "status": "SUCCEEDED",
                "knobs": {
                    "DECODE_BATCH_CAP": "16",
                    "CLIENT_CONCURRENCIES": "64"
                },
                "metrics": {
                    "token_throughput_per_chip": 460.0,
                    "concurrency": 64
                },
            },
            {
                "run_id": "rep-c64-b",
                "recipe": "glm5_1p1d",
                "status": "SUCCEEDED",
                "knobs": {
                    "DECODE_BATCH_CAP": "16",
                    "CLIENT_CONCURRENCIES": "64"
                },
                "metrics": {
                    "token_throughput_per_chip": 464.0,
                    "concurrency": 64
                },
            },
        ]
        lines = [json.dumps(r) for r in rows]
        runs = run_history.load_jsonl(lines)
        est = run_history.noise_estimate(runs, "glm5_1p1d")
        self.assertIsNotNone(est)
        assert est is not None
        self.assertEqual(est.n_groups, 2)
        self.assertLess(est.rel_half_range, 0.02)
        # Pooled sweep SD across c=32 and c=64 is huge (~25%+), proving strict grouping works.
        assert est.pooled_sweep_rel_sd is not None
        self.assertGreater(est.pooled_sweep_rel_sd, 0.20)
        # Scored on tok/s/chip without a rate; Mtok/$ needs one and gives the
        # same relative noise.
        self.assertEqual(runs[0].score, 300.0)
        self.assertIsNone(runs[0].mtok_per_dollar)
        priced = run_history.load_jsonl(lines,
                                        price_usd=_TEST_CHIP_HOUR_USD,
                                        metric=run_history.MTOK_PER_DOLLAR)
        priced_est = run_history.noise_estimate(priced, "glm5_1p1d")
        assert priced_est is not None
        self.assertAlmostEqual(priced_est.rel_half_range, est.rel_half_range)

        matches = run_history.find_baselines(
            "glm5_1p1d",
            {
                "DECODE_BATCH_CAP": "20",
                "CLIENT_CONCURRENCIES": "64"
            },
            runs,
        )
        self.assertTrue(matches)
        self.assertEqual(matches[0].differing_knobs, ("DECODE_BATCH_CAP", ))

        merged, added, updated = run_history.upsert_rows(lines[:2], rows[1:3])
        self.assertEqual(added, 1)
        self.assertEqual(updated, 1)
        self.assertEqual(len(merged), 3)

    def test_prior_outcomes_read_from_close_gaps_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "close-gaps" / "run-a"
            (run_dir / "gaps").mkdir(parents=True)
            (run_dir / "ledger.md").write_text(
                "# close-gaps ledger\n\n"
                "- outcome: crash `TPU_MLA_ABSORB_Q_FROM_HEAD=1` (vmem_oom)\n",
                encoding="utf-8",
            )
            (run_dir / "gaps" / "2-decode-dcp.md").write_text(
                "## A/B 1\n"
                "- outcome: not a win `DCP_SIZE=8` (cost_model.json)\n"
                "- outcome: win `DECODE_DCP_SIZE=8` `PREFILL_DCP_SIZE=16`\n"
                "- outcome: not a win (no setting, skipped)\n",
                encoding="utf-8",
            )
            loaded = ledger.read(run_dir / "ledger.md")
            self.assertEqual(ledger.read(run_dir), loaded)
            self.assertEqual(
                [(e.payload["decision"], e.payload["settings"])
                 for e in loaded],
                [
                    ("crash", ["TPU_MLA_ABSORB_Q_FROM_HEAD=1"]),
                    ("not a win", ["DCP_SIZE=8"]),
                    ("win", ["DECODE_DCP_SIZE=8", "PREFILL_DCP_SIZE=16"]),
                ],
            )
            self.assertEqual(loaded[1].payload["run_id"],
                             "gaps/2-decode-dcp.md:2")
            self.assertEqual(
                lessons.flag_verdicts(loaded)["DCP_SIZE"].decision,
                ledger.NOT_A_WIN)

            findings = preflight.check_launch(
                "1p1d_long_context",
                {"DCP_SIZE": "8"},
                ledger_entries=loaded,
            )
            prior = [f for f in findings if f.check == "prior_outcome"]
            self.assertEqual(len(prior), 1)
            self.assertEqual(prior[0].severity, preflight.WARNING)
            self.assertIn("gaps/2-decode-dcp.md:2: not a win",
                          prior[0].message)

            rc, out = _run_cli([
                "preflight",
                "--knob",
                "DCP_SIZE=8",
                "--ledger",
                str(run_dir / "ledger.md"),
                "--ledger",
                str(pathlib.Path(tmp) / "absent" / "ledger.md"),
                "--no-fail-on-error",
                "--json",
            ])
            self.assertEqual(rc, 0)
            by_sev = {(f["check"], f["severity"])
                      for f in json.loads(out)["findings"]}
            self.assertIn(("prior_outcome", "WARNING"), by_sev)
            self.assertIn(("prior_outcome", "INFO"), by_sev)

    def test_verify_mechanism_gate(self):
        before = {
            "meta": {
                "concurrency": 48,
                "num_chips": 16
            },
            "metrics": {
                "token_throughput_per_chip": 310.0,
                "mean_ttft_ms": 950.0,
                "median_itl_ms": 25.0,
            },
        }
        after = {
            "meta": {
                "concurrency": 48,
                "num_chips": 16
            },
            "metrics": {
                "token_throughput_per_chip": 345.0,
                "mean_ttft_ms": 880.0,
                "median_itl_ms": 23.0,
            },
        }
        comp = verify.compare(before, after)
        mech_ok = verify.check_mechanism(
            ["D8"],
            [{
                "detector": "D8",
                "share_pct": 22.0
            }],
            [{
                "detector": "D8",
                "share_pct": 8.0
            }],
        )
        self.assertTrue(mech_ok.confirmed)
        verdict_keep = verify.decide_verdict(comp,
                                             noise_rel=0.015,
                                             mechanism=mech_ok)
        self.assertEqual(verdict_keep.decision, "KEEP")

        mech_bad = verify.check_mechanism(
            ["D8"],
            [{
                "detector": "D8",
                "share_pct": 22.0
            }],
            [{
                "detector": "D8",
                "share_pct": 24.0
            }],
        )
        self.assertFalse(mech_bad.confirmed)
        verdict_unconfirmed = verify.decide_verdict(comp,
                                                    noise_rel=0.015,
                                                    mechanism=mech_bad)
        self.assertNotEqual(verdict_unconfirmed.decision, "KEEP")

        verdict_eval_fail = verify.decide_verdict(comp,
                                                  noise_rel=0.015,
                                                  mechanism=mech_ok,
                                                  eval_drop=0.03)
        self.assertEqual(verdict_eval_fail.decision, "DISCARD")

    def test_verify_resolves_serving_analysis_schema_slugs_and_renumbers(self):
        before_sa = [{
            "detector_id": "D8_kv_head_padding",
            "est_step_fraction_saved": 0.22,
            "evidence": ["kv_heads=1 padded to 8"],
        }]
        after_sa = [{
            "detector_id": "D8_kv_head_padding",
            "est_step_fraction_saved": 0.06,
            "evidence": ["kv_heads=8 aligned"],
        }]
        for target in ("kv_head_padding", "D8", "D8_kv_head_padding"):
            with self.subTest(target=target):
                mech = verify.check_mechanism([target], before_sa, after_sa)
                self.assertTrue(mech.confirmed)
                self.assertEqual(mech.targeted[0].detector,
                                 "D8_kv_head_padding")
                self.assertAlmostEqual(mech.targeted[0].before_pct, 22.0)
                self.assertAlmostEqual(mech.targeted[0].after_pct, 6.0)
                self.assertEqual(mech.targeted[0].before_evidence,
                                 {"items": ["kv_heads=1 padded to 8"]})

        # A run that recorded the same detector under another number (a
        # hypothetical D99 here) still matches by its unique slug.
        before_renum = [{
            "detector_id": "D46_batched_rpa_decode_untunable",
            "est_step_fraction_saved": 0.18,
            "evidence": ["rpa_decode untuned"],
        }]
        after_renum = [{
            "detector_id": "D99_batched_rpa_decode_untunable",
            "est_step_fraction_saved": 0.03,
            "evidence": ["rpa_decode tuned"],
        }]
        for target in ("batched_rpa_decode_untunable",
                       "D46_batched_rpa_decode_untunable",
                       "D99_batched_rpa_decode_untunable"):
            with self.subTest(renumber_target=target):
                mech_renum = verify.check_mechanism([target], before_renum,
                                                    after_renum)
                self.assertTrue(mech_renum.confirmed)
                self.assertEqual(mech_renum.targeted[0].detector,
                                 "D99_batched_rpa_decode_untunable")
                self.assertAlmostEqual(mech_renum.targeted[0].before_pct, 18.0)
                self.assertAlmostEqual(mech_renum.targeted[0].after_pct, 3.0)

        # Ambiguous numeric prefix (multiple D7_* detectors) is not guessed,
        # and generic category names ("attn") stay untouched.
        self.assertEqual(verify.resolve_detector_id("D7"), "D7")
        self.assertEqual(verify.resolve_detector_id("attn"), "attn")

    def test_preflight_live_repo_row_buckets_override_static_fallback(self):
        arm = {
            "DECODE_BATCH_CAP": "20",
            "MTP": "1",
            "MTP_NUM_SPECULATIVE_TOKENS": "3",
            "DECODE_MOE": "tp",
        }
        static_hits = [
            f for f in preflight.check_launch("1p1d_long_context", arm)
            if f.check == "decode_tp_kernel"
        ]
        self.assertEqual(len(static_hits), 1)
        with tempfile.TemporaryDirectory() as tmp:
            kern_dir = pathlib.Path(tmp) / "src" / "vllm_torchtpu" / "kernels"
            kern_dir.mkdir(parents=True)
            (kern_dir / "moe_tp_decode.py").write_text(
                "ROW_BUCKETS = (16, 32, 64, 80)\n", encoding="utf-8")
            extended_hits = [
                f for f in preflight.check_launch(
                    "1p1d_long_context", arm, repo_dir=tmp)
                if f.check == "decode_tp_kernel"
            ]
            self.assertEqual(extended_hits, [])
            above_extended = [
                f for f in preflight.check_launch(
                    "1p1d_long_context",
                    {
                        **arm, "DECODE_BATCH_CAP": "24"
                    },
                    repo_dir=tmp,
                ) if f.check == "decode_tp_kernel"
            ]
            self.assertEqual(len(above_extended), 1)
            self.assertIn("96 tokens", above_extended[0].message)


class DisaggGatesAndReplayTest(unittest.TestCase):

    def test_early_verdict_stops_losing_arm_at_10_5_min(self):
        base_blocks = early_verdict.parse_blocks(SAMPLE_AIPERF_BASELINE)
        lose_blocks = early_verdict.parse_blocks(SAMPLE_AIPERF_LOSER)
        self.assertEqual(len(base_blocks), 1)
        self.assertEqual(base_blocks[0].elapsed_s, 630)
        verdict = early_verdict.assess(lose_blocks, 16, base_blocks, 16)
        self.assertEqual(verdict.status, early_verdict.STOP)
        assert verdict.ratio is not None
        self.assertAlmostEqual(verdict.ratio, 0.752, places=3)
        self.assertIn("stop", early_verdict.render(verdict).lower())

    def test_legs_rejects_prefill_only_arm_when_decode_saturated(self):
        p_loads, d_loads = legs.parse_server_metrics(SAMPLE_SERVER_METRICS_CSV)
        verdict = legs.assess(p_loads, d_loads)
        self.assertEqual(verdict.kind, legs.DECODE_LIMITED)
        warn = legs.warn_off_leg_arm(verdict, ["EXTRA_ENV_PREFILL"])
        self.assertIsNotNone(warn)
        self.assertIsNone(legs.warn_off_leg_arm(verdict, ["EXTRA_ENV_DECODE"]))

    def test_kv_replay_and_lane_saturation(self):
        reqs = [
            kv_replay.ReplayRequest(isl=512,
                                    block_hashes=[101, 102, 103, 104],
                                    osl=256),
            kv_replay.ReplayRequest(
                isl=768, block_hashes=[101, 102, 103, 104, 105, 106], osl=256),
        ]
        cfg = kv_replay.CacheConfig(unit_tokens=64,
                                    units_per_block=2,
                                    gpu_blocks=8,
                                    cpu_blocks=8)
        outcomes = kv_replay.replay(reqs, cfg)
        summary = kv_replay.totals(reqs, outcomes)
        self.assertGreater(summary["hbm_tokens"], 0)
        lanes = kv_replay.lane_saturation(
            client_concurrency=48,
            decode_batch_cap=16,
            throughput_req_s=0.35,
            mean_prefill_s=1.8,
            mean_decode_s=46.0,
            mean_admit_wait_s=4.0,
        )
        self.assertGreater(lanes.decode_fill_ratio, 0.9)
        self.assertAlmostEqual(kv_replay.littles_law_seconds(2.0, 20, 100.0),
                               10.0)

    def test_preflight_catches_symmetric_dcp_row_cap_and_vmem_oom(self):
        with tempfile.TemporaryDirectory() as tmp:
            kdir = pathlib.Path(
                tmp) / "src" / "vllm_torchtpu" / "kernels" / "fused_moe"
            kdir.mkdir(parents=True)
            (kdir / "moe_tp_decode.py").write_text(SAMPLE_KERNEL_SOURCE,
                                                   encoding="utf-8")
            envs_path = pathlib.Path(tmp) / "src" / "vllm_torchtpu" / "envs.py"
            envs_path.write_text(SAMPLE_ENVS_SOURCE, encoding="utf-8")
            p_loads, d_loads = legs.parse_server_metrics(
                SAMPLE_SERVER_METRICS_CSV)
            leg_v = legs.assess(p_loads, d_loads)

            bad_knobs = {
                "DCP_SIZE": "8",
                "DECODE_BATCH_CAP": "20",
                "MTP": "1",
                "MTP_NUM_SPECULATIVE_TOKENS": "3",
                "EXTRA_ENV_DECODE": "FUSED_MOE_FAST_SUBCHANNEL=1",
            }
            findings = preflight.check_launch(
                "1p1d_long_context",
                bad_knobs,
                code_flags=code_gap.parse_env_registry(SAMPLE_ENVS_SOURCE),
                repo_dir=tmp,
                leg_verdict=leg_v,
            )
            checks = {f.check for f in findings}
            self.assertIn("dcp_size_prefill_kv_halved", checks)
            self.assertIn("decode_tp_kernel", checks)
            self.assertIn("gate_off", checks)

            # Pure prefill-only arm is flagged as off_leg when decode is saturated.
            prefill_only_findings = preflight.check_launch(
                "1p1d_long_context",
                {"EXTRA_ENV_PREFILL": "DRAFTER_TP=4"},
                recipe_defaults={},
                repo_dir=tmp,
                leg_verdict=leg_v,
            )
            self.assertIn("off_leg", {f.check for f in prefill_only_findings})

            # Asymmetric P/D DCP with sal indexer or striped layout=1 flags errors.
            asym_bad_layout = {
                "PREFILL_DCP_SIZE": "16",
                "DECODE_DCP_SIZE": "8",
                "INDEXER_KV_CACHE_MODE": "sal",
                "EXTRA_ENV": "TPU_KV_STRIPED_LAYOUT=1",
                "DECODE_BATCH_CAP": "16",
            }
            asym_findings = preflight.check_launch("1p1d_long_context",
                                                   asym_bad_layout,
                                                   repo_dir=tmp)
            asym_checks = {f.check for f in asym_findings}
            self.assertIn("asym_dcp_indexer_layout", asym_checks)
            self.assertIn("asym_dcp_striped_layout", asym_checks)

            # DECODE_INTERLEAVE unset: the engine default (1) is not a whole
            # packed row, so the arm is blocked; a recipe default of 4 is too.
            asym_ok_layout = {
                "PREFILL_DCP_SIZE": "16",
                "DECODE_DCP_SIZE": "8",
                "INDEXER_KV_CACHE_MODE": "has",
                "EXTRA_ENV": "TPU_KV_STRIPED_LAYOUT=0",
                "DECODE_BATCH_CAP": "16",
            }

            def interleave_flagged(knobs, **kw):
                return "asym_dcp_interleave" in {
                    f.check
                    for f in preflight.check_launch(
                        "1p1d_long_context", knobs, repo_dir=tmp, **kw)
                }

            self.assertTrue(interleave_flagged(asym_ok_layout))
            self.assertTrue(
                interleave_flagged(asym_ok_layout,
                                   recipe_defaults={"DECODE_INTERLEAVE": "4"}))
            self.assertFalse(
                interleave_flagged({
                    **asym_ok_layout, "DECODE_INTERLEAVE": "8"
                }))

            # VMEM absorb_q_head OOM check at DCP >= 16.
            vmem_knobs = {
                "DECODE_DCP_SIZE": "16",
                "EXTRA_ENV_DECODE": "TPU_MLA_ABSORB_Q_FROM_HEAD=1",
            }
            vmem_findings = preflight.check_launch("1p1d_long_context",
                                                   vmem_knobs,
                                                   repo_dir=tmp)
            self.assertIn("vmem_absorb_q_head",
                          {f.check
                           for f in vmem_findings})

            # An inert flag (prerequisite gate off) warns but does not block.
            inert = preflight.check_launch(
                "1p1d_long_context",
                {
                    "DECODE_BATCH_CAP": "16",
                    "EXTRA_ENV_DECODE": "FUSED_MOE_FAST_SUBCHANNEL=1",
                },
                code_flags=code_gap.parse_env_registry(SAMPLE_ENVS_SOURCE),
                repo_dir=tmp,
            )
            self.assertIn(("gate_off", preflight.WARNING),
                          {(f.check, f.severity)
                           for f in inert})
            self.assertFalse(any(f.severity == preflight.ERROR for f in inert))

    def test_decode_tp_kernel_flags_only_steps_above_largest_row_bucket(self):

        def tp_kernel(cap: str) -> list[preflight.Finding]:
            return [
                f for f in preflight.check_launch(
                    "1p1d_long_context",
                    {
                        "DECODE_BATCH_CAP": cap,
                        "MTP": "1",
                        "MTP_NUM_SPECULATIVE_TOKENS": "3",
                        "DECODE_MOE": "tp",
                    },
                ) if f.check == "decode_tp_kernel"
            ]

        self.assertEqual(tp_kernel("12"), [])  # 48 tokens pad into 64.
        self.assertEqual(tp_kernel("16"), [])  # 64 tokens.
        flagged = tp_kernel("20")  # 80 tokens.
        self.assertEqual(len(flagged), 1)
        self.assertIn("80 tokens", flagged[0].message)
        self.assertIn("keep cap <= 16", flagged[0].message)

    def test_legs_config_diffs_against_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            metrics_file = root / "server_metrics.csv"
            metrics_file.write_text(SAMPLE_SERVER_METRICS_CSV,
                                    encoding="utf-8")
            base = {
                "CLIENT_CONCURRENCIES": "48",
                "DCP_SIZE": "16",
                "DECODE_BATCH_CAP": "16",
                "MTP": "1",
                "MTP_NUM_SPECULATIVE_TOKENS": "3",
            }
            base_file = root / "base.json"
            base_file.write_text(json.dumps(base), encoding="utf-8")
            prefill_arm = root / "prefill_arm.json"
            prefill_arm.write_text(json.dumps({
                **base, "EXTRA_ENV_PREFILL":
                "DRAFTER_TP=4"
            }),
                                   encoding="utf-8")
            decode_arm = root / "decode_arm.json"
            decode_arm.write_text(json.dumps({
                **base, "DECODE_DCP_SIZE": "8"
            }),
                                  encoding="utf-8")

            def off_leg(config: pathlib.Path, *extra: str) -> str | None:
                rc, out = _run_cli([
                    "legs", "--server-metrics",
                    str(metrics_file), "--config",
                    str(config), *extra, "--json"
                ])
                self.assertEqual(rc, 0)
                return json.loads(out)["off_leg_warning"]

            warning = off_leg(prefill_arm, "--defaults", str(base_file))
            self.assertIsNotNone(warning)
            self.assertIn("(EXTRA_ENV_PREFILL)", warning or "")
            self.assertIsNone(off_leg(decode_arm, "--defaults",
                                      str(base_file)))
            # Without the baseline every knob counts as changed.
            self.assertIsNone(off_leg(prefill_arm))


class SourceAuditAndMicrobenchTest(unittest.TestCase):

    def test_code_limits_and_microbench_generation_and_verdict(self):
        limits = code_limits.scan_text(SAMPLE_KERNEL_SOURCE,
                                       "moe_tp_decode.py")
        found = code_limits.new_crossings(limits, [80], [64])
        self.assertTrue(any(c.limit.name == "ROW_BUCKETS" for c in found))
        row_crossing = [c for c in found if c.limit.name == "ROW_BUCKETS"][0]
        spec = microbench.spec_from_crossing(row_crossing, base_steps=[64])
        self.assertIn(80, microbench.extended_value(spec, 80))
        script = microbench.render_pytest(spec)
        self.assertIn("ROW_BUCKETS", script)

        result_payload = {
            "device_kind": "TPU v7x",
            "devices": 8,
            "chain_overhead_us": 4.0,
            "kernel": "moe_tp_decode",
            "cases": {
                "80:candidate": {
                    "admitted": True,
                    "per_layer_us": 179.0,
                    "rel_l2": 0.012,
                    "cos": 0.9995,
                    "compile_s": 14.2,
                },
                "80:fallback": {
                    "admitted": True,
                    "per_layer_us": 442.0,
                },
            },
        }
        stdout = "RESULT " + json.dumps(result_payload) + "\n"
        parsed = microbench.parse_result(stdout)
        verdict = microbench.evaluate_result(parsed, steps=[80])
        self.assertTrue(verdict.passed)
        self.assertAlmostEqual(verdict.steps[0].speedup or 0.0,
                               438.0 / 175.0,
                               places=2)

    def test_code_gap_ports_and_lessons(self):
        served = code_gap.parse_env_registry(SAMPLE_ENVS_SOURCE)
        upstream_src = SAMPLE_ENVS_SOURCE.replace(
            '"MOE_HUDSON_EP_OPT", "0"',
            '"MOE_HUDSON_EP_OPT", "0")\n    # Extra upstream switch\n    "NEW_MLA_TILE": lambda: os.getenv("NEW_MLA_TILE", "0"',
        )
        upstream = code_gap.parse_env_registry(upstream_src)
        gaps = code_gap.find_gaps(served,
                                  upstream,
                                  arm_env={},
                                  steps=[80],
                                  ep_enabled=False)
        by_name = {g.flag: g for g in gaps}
        self.assertIn("FUSED_MOE_FAST_SUBCHANNEL", by_name)
        self.assertTrue(
            code_gap.hbm_risk(by_name["FUSED_MOE_FAST_SUBCHANNEL"].doc))

        # VMEM OOM must match ahead of generic HBM OOM.
        vmem_diag = lessons.diagnose_log(
            "RESOURCE_EXHAUSTED: Ran out of memory in memory space vmem (68 MiB > 64 MiB)"
        )
        self.assertEqual(vmem_diag.pattern, "vmem_oom")
        hbm_diag = lessons.diagnose_log(
            "RESOURCE_EXHAUSTED: Out of memory allocating 12884901888 bytes in hbm"
        )
        self.assertEqual(hbm_diag.pattern, "hbm_oom")

        heads = ports.parse_ls_remote(
            "aaaaaaaa11111111aaaaaaaa11111111aaaaaaaa\trefs/pull/42/head\n")
        self.assertEqual(len(heads), 1)
        self.assertEqual(heads[0].number, 42)

    def test_code_gap_outside_checkout_fails_with_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc, out = _run_cli(
                    ["code-gap", "--repo", tmp, "--steps", "80", "--json"])
            self.assertEqual(rc, 2)
            self.assertEqual(out, "")
            self.assertIn("not found: pass --repo <vllm-torchtpu checkout>",
                          err.getvalue())


class RooflineAndPredictorTest(unittest.TestCase):

    def test_model_plugins_and_trace_roofline(self):
        for name in ("glm5_mla_moe", "qwen3_5_hybrid_moe", "deepseek_mla_moe"):
            plugin = model_plugins.get_plugin(name)
            self.assertEqual(plugin.hardware.name, "tpu7x")

        csv_trace = ("name,group,ms_per_step,calls_per_step\n"
                     "ragged_all_to_all_gmm_moe,moe,12.4,78\n"
                     "mla_ragged_paged_attention,attn,6.8,78\n"
                     "sparse_core_topk_indexer,indexer,2.1,78\n"
                     "all_reduce_fusion,comm,3.2,78\n")
        records = trace_roofline.parse_tabular_ops(csv_trace)
        self.assertEqual(len(records), 4)
        plugin = model_plugins.get_plugin("glm5_mla_moe")
        ctx = model_plugins.StepContext(batch_reqs=16,
                                        padded_rows=64,
                                        context_tokens=32768)
        report = trace_roofline.analyze_records(records, plugin, ctx)
        self.assertGreater(report.step_ms, 20.0)
        self.assertTrue(report.pack_comparisons)
        self.assertTrue(report.ranked_targets)
        markdown = report.render_markdown()
        self.assertIn("Roofline", markdown)
        # Lever saving shares are hand-set priors, so the report labels every
        # printed saving range as heuristic.
        self.assertIn("hand-set prior share", markdown)
        self.assertEqual(markdown.count("ms/step** (heuristic; up to"),
                         min(5, len(report.ranked_targets)))

    def test_closed_queueing_network_predictor_1p1d_and_single_host(self):
        pd_cfg = predictor.PredictConfig(
            plugin_name="glm5_mla_moe",
            topology="1P1D",
            concurrency=48,
            cap=16,
        )
        pd_res = predictor.predict_serving(pd_cfg)
        self.assertGreater(pd_res.tok_s_chip, 50.0)
        self.assertIsNone(pd_res.mtok_per_usd)
        self.assertGreater(pd_res.p90_interactivity, 5.0)
        priced = predictor.predict_serving(
            dataclasses.replace(pd_cfg, chip_hour_usd=_TEST_CHIP_HOUR_USD))
        self.assertEqual(priced.tok_s_chip, pd_res.tok_s_chip)
        assert priced.mtok_per_usd is not None
        self.assertAlmostEqual(
            priced.mtok_per_usd,
            metrics.mtok_per_usd(pd_res.tok_s_chip, _TEST_CHIP_HOUR_USD))

        faster_pd = predictor.predict_serving(
            predictor.PredictConfig(
                plugin_name="glm5_mla_moe",
                topology="1P1D",
                concurrency=48,
                cap=16,
                step_delta_ms=4.0,
            ))
        self.assertGreater(faster_pd.tok_s_chip, pd_res.tok_s_chip)
        self.assertGreater(faster_pd.p90_interactivity,
                           pd_res.p90_interactivity)

        sh_res = predictor.predict_serving(
            predictor.PredictConfig(
                plugin_name="qwen3_5_hybrid_moe",
                topology="single_host",
                concurrency=32,
                cap=16,
            ))
        self.assertGreater(sh_res.tok_s_chip, 20.0)

    def test_vpu_unpack_rate_is_an_explicit_parameter(self):
        plugin = model_plugins.get_plugin("glm5_mla_moe")
        self.assertIsNone(plugin.hardware.vpu_unpack_gweights_per_ms_per_tc)
        ctx = model_plugins.StepContext(batch_reqs=16,
                                        padded_rows=64,
                                        weight_dtype="w4a8",
                                        moe_mode="tp")

        def moe_roofline(p: model_plugins.ModelPlugin):
            return next(r for r in p.rooflines(ctx).values()
                        if "MoE" in r.buckets)

        unrated = moe_roofline(plugin)
        self.assertTrue(
            any("VPU widen time not included" in w for w in unrated.warnings))
        rated = moe_roofline(
            dataclasses.replace(
                plugin,
                hardware=dataclasses.replace(
                    plugin.hardware,
                    vpu_unpack_gweights_per_ms_per_tc=(_TEST_VPU_RATE))))
        self.assertGreater(rated.compute_floor_ms_per_step,
                           unrated.compute_floor_ms_per_step)
        self.assertFalse(any("not included" in w for w in rated.warnings))


class HillclimbCliAllSubcommandsTest(unittest.TestCase):

    def test_all_13_cli_subcommands_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            # Initialize a tiny local git repo so `code-gap` and `ports` exercise git refs too.
            repo = root / "repo"
            (repo / "src" / "vllm_torchtpu" / "kernels").mkdir(parents=True)
            envs_file = repo / "src" / "vllm_torchtpu" / "envs.py"
            envs_file.write_text(SAMPLE_ENVS_SOURCE, encoding="utf-8")
            (repo / "src" / "vllm_torchtpu" / "kernels" /
             "moe_tp_decode.py").write_text(SAMPLE_KERNEL_SOURCE,
                                            encoding="utf-8")
            subprocess.run(["git", "init", "-q", "-b", "main"],
                           cwd=repo,
                           check=True)
            subprocess.run(
                [
                    "git", "config", "user.email",
                    "test@users.noreply.github.com"
                ],
                cwd=repo,
                check=True,
            )
            subprocess.run(["git", "config", "user.name", "Tester"],
                           cwd=repo,
                           check=True)
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "base"],
                           cwd=repo,
                           check=True)
            subprocess.run(["git", "checkout", "-q", "-b", "pr-80-bucket"],
                           cwd=repo,
                           check=True)
            envs_file.write_text(
                SAMPLE_ENVS_SOURCE.replace(
                    "environment_variables = {",
                    'environment_variables = {\n    "EXTEND_TP_MOE_80": lambda: os.getenv("EXTEND_TP_MOE_80", "0") == "1",',
                ),
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "commit", "-q", "-am", "add 80 bucket flag"],
                cwd=repo,
                check=True)
            subprocess.run(["git", "checkout", "-q", "main"],
                           cwd=repo,
                           check=True)

            # 1. preflight
            cfg_file = root / "cand.json"
            cfg_file.write_text(
                json.dumps({
                    "DCP_SIZE": "8",
                    "DECODE_BATCH_CAP": "20"
                }),
                encoding="utf-8",
            )
            rc, out = _run_cli([
                "preflight",
                "--config",
                str(cfg_file),
                "--repo",
                str(repo),
                "--no-fail-on-error",
                "--json",
            ])
            self.assertEqual(rc, 0)
            pre_json = json.loads(out)
            self.assertTrue(
                any(f["check"] == "dcp_size_prefill_kv_halved"
                    for f in pre_json["findings"]))

            # 2. early-verdict
            base_log = root / "base_aiperf.log"
            arm_log = root / "arm_aiperf.log"
            base_log.write_text(SAMPLE_AIPERF_BASELINE, encoding="utf-8")
            arm_log.write_text(SAMPLE_AIPERF_LOSER, encoding="utf-8")
            rc, out = _run_cli([
                "early-verdict",
                "--baseline-log",
                str(base_log),
                "--arm-log",
                str(arm_log),
                "--json",
            ])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(out)["status"], "stop")

            # 3. legs
            sm_file = root / "server_metrics.csv"
            sm_file.write_text(SAMPLE_SERVER_METRICS_CSV, encoding="utf-8")
            rc, out = _run_cli([
                "legs",
                "--server-metrics",
                str(sm_file),
                "--knob",
                "EXTRA_ENV_PREFILL=DRAFTER_TP=4",
                "--json",
            ])
            self.assertEqual(rc, 0)
            legs_json = json.loads(out)
            self.assertEqual(legs_json["kind"], "decode_limited")
            self.assertTrue(legs_json["off_leg_warning"])

            # 4. kv-replay
            req_file = root / "reqs.jsonl"
            req_file.write_text(
                json.dumps({
                    "isl": 4096,
                    "block_hashes": [1, 2],
                    "osl": 128
                }) + "\n",
                encoding="utf-8",
            )
            rc, out = _run_cli([
                "kv-replay",
                "--requests",
                str(req_file),
                "--throughput-req-s",
                "0.3",
                "--mean-prefill-s",
                "1.5",
                "--mean-decode-s",
                "40.0",
                "--json",
            ])
            self.assertEqual(rc, 0)
            self.assertIn("as_served", json.loads(out)["policies"])

            # 5. code-limits
            rc, out = _run_cli([
                "code-limits",
                "--repo",
                str(repo),
                "--steps",
                "80",
                "--base-steps",
                "64",
                "--json",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(
                any(c["name"] == "ROW_BUCKETS" for c in json.loads(out)))

            # 6. code-gap
            rc, out = _run_cli([
                "code-gap",
                "--repo",
                str(repo),
                "--served-ref",
                "main",
                "--upstream-ref",
                "pr-80-bucket",
                "--steps",
                "80",
                "--json",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(
                any(g["flag"] == "EXTEND_TP_MOE_80" for g in json.loads(out)))

            # 7. ports
            rc, out = _run_cli([
                "ports",
                "--repo",
                str(repo),
                "--served-ref",
                "main",
                "--refs",
                "pr-80-bucket",
                "--steps",
                "80",
                "--json",
            ])
            self.assertEqual(rc, 0)
            self.assertEqual(len(json.loads(out)), 1)

            # 8a & 8b. microbench (generate + evaluate)
            mb_out = root / "bench_gen.py"
            rc, _ = _run_cli([
                "microbench",
                "--repo",
                str(repo),
                "--limit-name",
                "ROW_BUCKETS",
                "--steps",
                "80",
                "--base-steps",
                "64",
                "--out",
                str(mb_out),
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(mb_out.is_file())

            bench_log = root / "bench_res.log"
            bench_log.write_text(
                "RESULT " + json.dumps({
                    "device_kind": "TPU v7x",
                    "devices": 8,
                    "chain_overhead_us": 3.0,
                    "kernel": "moe_tp_decode",
                    "cases": {
                        "80:candidate": {
                            "admitted": True,
                            "per_layer_us": 178.0,
                            "rel_l2": 0.01,
                            "cos": 0.999,
                        },
                        "80:fallback": {
                            "admitted": True,
                            "per_layer_us": 441.0,
                        },
                    },
                }) + "\n",
                encoding="utf-8",
            )
            rc, out = _run_cli([
                "microbench",
                "--bench-log",
                str(bench_log),
                "--steps",
                "80",
                "--json",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(json.loads(out)["passed"])

            # 9. roofline
            trace_csv = root / "trace.csv"
            trace_csv.write_text(
                "name,group,ms_per_step,calls_per_step\n"
                "ragged_all_to_all_gmm_moe,moe,11.5,78\n"
                "mla_ragged_paged_attention,attn,6.0,78\n",
                encoding="utf-8",
            )
            rc, out = _run_cli([
                "roofline",
                "--trace",
                str(trace_csv),
                "--plugin",
                "glm5_mla_moe",
                "--batch-reqs",
                "16",
                "--padded-rows",
                "64",
                "--json",
            ])
            self.assertEqual(rc, 0)
            self.assertGreater(json.loads(out)["step_ms"], 15.0)

            # 10. predict
            rc, out = _run_cli([
                "predict",
                "--plugin",
                "glm5_mla_moe",
                "--concurrency",
                "48",
                "--decode-cap",
                "16",
                "--chip-hour-usd",
                str(_TEST_CHIP_HOUR_USD),
                "--json",
            ])
            self.assertEqual(rc, 0)
            self.assertGreater(json.loads(out)["mtok_per_usd"], 0.1)
            rc, out = _run_cli(["predict", "--json"])
            self.assertEqual(rc, 0)
            self.assertIsNone(json.loads(out)["mtok_per_usd"])
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc, _ = _run_cli(["predict", "--chip-hour-usd", "0"])
            self.assertEqual(rc, 2)
            self.assertIn("--chip-hour-usd must be positive", err.getvalue())

            # 11. verify
            b_json = root / "b.json"
            c_json = root / "c.json"
            b_json.write_text(
                json.dumps({"metrics": {
                    "token_throughput_per_chip": 300.0
                }}),
                encoding="utf-8",
            )
            c_json.write_text(
                json.dumps({"metrics": {
                    "token_throughput_per_chip": 330.0
                }}),
                encoding="utf-8",
            )
            rc, out = _run_cli([
                "verify",
                "--baseline",
                str(b_json),
                "--candidate",
                str(c_json),
                "--noise-rel",
                "0.015",
                "--json",
            ])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(out)["decision"], "KEEP")

            # 12. noise
            hist_file = root / "history.jsonl"
            hist_file.write_text(
                "\n".join([
                    json.dumps({
                        "run_id": "r1",
                        "recipe": "1p1d",
                        "status": "SUCCEEDED",
                        "knobs": {
                            "CLIENT_CONCURRENCIES": "48"
                        },
                        "metrics": {
                            "token_throughput_per_chip": 310.0,
                            "concurrency": 48
                        },
                    }),
                    json.dumps({
                        "run_id": "r2",
                        "recipe": "1p1d",
                        "status": "SUCCEEDED",
                        "knobs": {
                            "CLIENT_CONCURRENCIES": "48"
                        },
                        "metrics": {
                            "token_throughput_per_chip": 312.0,
                            "concurrency": 48
                        },
                    }),
                ]) + "\n",
                encoding="utf-8",
            )
            rc, out = _run_cli(
                ["noise", "--history",
                 str(hist_file), "--json"])
            self.assertEqual(rc, 0)
            self.assertIsNotNone(json.loads(out)["noise"])

            # 13. diagnose-failure
            fail_log = root / "fail.log"
            fail_log.write_text(
                "RESOURCE_EXHAUSTED: Ran out of memory in memory space vmem\n",
                encoding="utf-8",
            )
            rc, out = _run_cli([
                "diagnose-failure",
                "--log",
                str(fail_log),
                "--json",
            ])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(out)["pattern"], "vmem_oom")


class VerdictAndScopeRulesTest(unittest.TestCase):
    """Rules carried over from measured serving A/B runs."""

    @staticmethod
    def _bench(tok_per_chip: float, ttft_ms: float,
               itl_ms: float) -> dict[str, object]:
        return {
            "meta": {
                "concurrency": 48,
                "num_chips": 16
            },
            "metrics": {
                "token_throughput_per_chip": tok_per_chip,
                "mean_ttft_ms": ttft_ms,
                "median_itl_ms": itl_ms,
            },
        }

    def test_keep_with_interactivity_regression_is_inconclusive(self):
        comp = verify.compare(self._bench(310.0, 950.0, 25.0),
                              self._bench(345.0, 960.0, 29.0))
        verdict = verify.decide_verdict(comp, noise_rel=0.015)
        self.assertEqual(verdict.decision, verify.INCONCLUSIVE)
        self.assertIn("matched interactivity", verdict.reason)

        clean = verify.compare(self._bench(310.0, 950.0, 25.0),
                               self._bench(345.0, 940.0, 24.5))
        self.assertEqual(
            verify.decide_verdict(clean, noise_rel=0.015).decision,
            verify.KEEP)

    def test_extra_prefill_leg_only_where_calibrated(self):
        knobs = {
            "NUM_PREFILLS": "2",
            "NUM_DECODES": "1",
            "CLIENT_CONCURRENCIES": "96",
        }
        glm = preflight.check_disagg_knobs(knobs,
                                           knob_map=preflight.GLM5_PD_KNOB_MAP)
        self.assertIn("extra_prefill_leg", {f.check for f in glm})
        other = preflight.check_disagg_knobs(
            knobs, knob_map=preflight.DEEPSEEK_PD_KNOB_MAP)
        self.assertNotIn("extra_prefill_leg", {f.check for f in other})
        above = preflight.check_disagg_knobs(
            {
                **knobs, "CLIENT_CONCURRENCIES": "128"
            },
            knob_map=preflight.GLM5_PD_KNOB_MAP)
        self.assertNotIn("extra_prefill_leg", {f.check for f in above})

    def test_prior_outcome_of_a_bundle_is_not_one_settings(self):
        entries = ledger.parse_outcomes(
            "- outcome: win `DECODE_DCP_SIZE=8` `PREFILL_DCP_SIZE=16`\n"
            "- outcome: not a win `DECODE_DCP_SIZE=8`\n", "ledger.md")
        priors = lessons.prior_outcomes(entries, ["DECODE_DCP_SIZE=8"])
        self.assertEqual([p.bundle_size for p in priors], [1, 2])
        findings = preflight.check_launch(
            "1p1d_long_context",
            {"DECODE_DCP_SIZE": "8"},
            ledger_entries=entries,
        )
        messages = [f.message for f in findings if f.check == "prior_outcome"]
        self.assertTrue(any("one of 2 settings" in m for m in messages))
        self.assertTrue(any(m.endswith("not a win.") for m in messages),
                        messages)

    def test_plugin_aliases_resolve_to_their_knob_map(self):
        kmap, warning = hillclimb_cli._knob_map("qwen3_5_single")
        self.assertIs(kmap, preflight.QWEN35_SINGLE_KNOB_MAP)
        self.assertEqual(warning, "")
        kmap, warning = hillclimb_cli._knob_map("no_such_model")
        self.assertIs(kmap, preflight.GLM5_PD_KNOB_MAP)
        self.assertIn("No knob map is registered", warning)
        rc, out = _run_cli([
            "preflight", "--plugin", "no_such_model", "--knob", "DCP_SIZE=16",
            "--no-fail-on-error", "--json"
        ])
        self.assertEqual(rc, 0)
        self.assertIn(("plugin", "WARNING"),
                      {(f["check"], f["severity"])
                       for f in json.loads(out)["findings"]})


if __name__ == "__main__":
    unittest.main()
