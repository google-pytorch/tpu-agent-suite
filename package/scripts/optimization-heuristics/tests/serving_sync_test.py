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
"""Unit tests for new serving_analysis detectors, guardrails, and utilities."""

import dataclasses
import functools
import math
import os
import sys
import unittest

import pandas as pd

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TESTS_DIR)
for _p in (_TESTS_DIR, _ROOT, os.path.join(os.path.dirname(_ROOT), 'common')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import synthetic_events  # noqa: E402
from serving_analysis import analyze as analyze_lib  # noqa: E402
from serving_analysis import catalog_detectors  # noqa: E402
from serving_analysis import code_locations  # noqa: E402
from serving_analysis import context  # noqa: E402
from serving_analysis import detectors  # noqa: E402
from serving_analysis import features  # noqa: E402
from serving_analysis import findings  # noqa: E402
from serving_analysis import planner  # noqa: E402
from serving_analysis import segmenter  # noqa: E402
from serving_analysis import trace_table  # noqa: E402
from serving_analysis import detector_utils as du  # noqa: E402

_SID = 'sess-1'
_GMM_FILE = 'tpu_inference/kernels/megablox/gmm.py'


@functools.cache
def _default() -> features.Features:
    return synthetic_events.compute_features(session_id=_SID)


def _knobs(found: list[findings.Finding]) -> dict[str, str]:
    return {
        kc.name: kc.suggested_value
        for f in found
        for kc in f.knob_changes
    }


def _with_op(events: pd.DataFrame, target_op: str, /,
             **updates: object) -> pd.DataFrame:
    out = events.copy()
    mask = out['hlo_op'] == target_op
    for col, val in updates.items():
        out.loc[mask, col] = val
    return out


def _host_tokens(events: pd.DataFrame, reqs: int, toks: int) -> pd.DataFrame:
    """Decode steps of `make_events(decode_reqs=reqs)` annotated with `toks`."""
    out = events.copy()
    host = out['category'] == 'host'
    out.loc[host, 'kernel_name'] = out.loc[host, 'kernel_name'].str.replace(
        f'{reqs} reqs, {reqs} toks', f'{reqs} reqs, {toks} toks', regex=False)
    return out


class CodeLocationsTest(unittest.TestCase):

    def test_normalize_path(self):
        cases = (
            (
                '/home/u/tpu-inference/tpu_inference/layers/vllm/attention.py:285',
                ('tpu_inference/layers/vllm/attention.py', 285),
            ),
            (
                'vllm/lib/python3.12/site-packages/torchax/ops/jaten.py:280',
                ('torchax/ops/jaten.py', 280),
            ),
            (
                '/workspace/vllm-torchtpu/src/vllm_torchtpu/layers/vllm/x.py:17',
                ('vllm_torchtpu/layers/vllm/x.py', 17),
            ),
            (
                'maxtext/src/maxtext/layers/moe.py:2301',
                ('maxtext/layers/moe.py', 2301),
            ),
            (
                '/home/u/tpu_inference/tpu_inference/kernels/mla/v1/kernel.py:102',
                ('tpu_inference/kernels/mla/v1/kernel.py', 102),
            ),
            (
                'vllm/vllm/model_executor/layers/fused_moe/layer.py:403',
                ('vllm/model_executor/layers/fused_moe/layer.py', 403),
            ),
        )
        for recorded, expected in cases:
            with self.subTest(recorded=recorded):
                self.assertEqual(code_locations.normalize_path(recorded),
                                 expected)

        for dropped in (
                '/usr/lib/python3.12/site-packages/torch_tpu/_internal/pallas/pallas.py:684',
                '/x/site-packages/torch/_library/autograd.py:41',
                '/x/site-packages/qwix/_src/core/pallas.py:76',
                '',
                'kernel.py',
        ):
            with self.subTest(dropped=dropped):
                self.assertIsNone(code_locations.normalize_path(dropped))

    def test_is_library_and_function_name(self):
        self.assertTrue(code_locations.is_library('torchax/ops/jaten.py'))
        self.assertTrue(code_locations.is_library('flax/nnx/nn/linear.py'))
        self.assertFalse(
            code_locations.is_library(
                'tpu_inference/layers/common/fused_moe_gmm.py'))
        self.assertEqual(
            code_locations.function_name(
                'jit(step_fun)/FusedMoE/jit(fused_moe_func)/shard_map/psum'),
            'fused_moe_func',
        )
        self.assertEqual(
            code_locations.function_name(
                'jit(step_fun)/RowParallelLinear/shard_map/psum'),
            'RowParallelLinear',
        )
        self.assertEqual(code_locations.function_name('mm/as_strided'), '')

    def test_location_index_and_analyze_attach(self):
        index = code_locations.LocationIndex(_default())
        locs = index.top(
            'decode',
            code_locations.DETECTOR_OPS['D26_moe_combine_no_local_reduction'],
        )
        self.assertEqual(len(locs), 1)
        loc = locs[0]
        self.assertEqual(
            (loc.path, loc.line, loc.op, loc.op_type, loc.function),
            (
                synthetic_events.MOE_FILE,
                synthetic_events.MOE_COMBINE_LINE,
                'psum',
                'psum',
                'fused_moe_func',
            ),
        )
        hot = index.hotspots(n=3)
        self.assertCountEqual(hot, ['decode', 'mixed'])
        self.assertEqual(hot['decode'][0].path, _GMM_FILE)

        table = trace_table.TraceTable.from_events_frame(
            synthetic_events.make_events(), _SID)
        result = analyze_lib.analyze(table, synthetic_events.serving_context())
        self.assertEqual(
            [e for e in result.errors if e.startswith('code_locations')], [])
        self.assertIn('decode', result.hotspots)
        summary = result.summary_dict()
        self.assertEqual(summary['code_hotspots']['decode'][0]['path'],
                         _GMM_FILE)
        md = result.to_markdown()
        self.assertIn('**Hot code**', md)
        self.assertIn(f'`{_GMM_FILE}:622` in `gmm_pipeline`', md)


class DetectorUtilsTest(unittest.TestCase):

    def test_units_dims_and_public_links(self):
        self.assertEqual(du.ms(2_500_000), 2.5)
        self.assertEqual(du.us(2_500), 2.5)
        self.assertEqual(du.pct(0.125), 12.5)
        self.assertEqual(du.parse_dims('64x1024'), [64, 1024])
        self.assertEqual(du.n_elems('64x1024'), 64 * 1024)
        self.assertEqual(du.parse_dims('64xN'), [])
        self.assertIsNone(du.xprof_link(_SID, 1_000_000, 2_000_000))
        self.assertIsNone(du.step_link(_SID, _default(), segmenter.DECODE))

    def test_framework_and_knob_readers(self):
        f = _default()
        detected = synthetic_events.serving_context()
        torchtpu = synthetic_events.serving_context(framework='vllm_torchtpu')
        self.assertEqual(du.framework(detected, f), 'vllm_tpu')
        self.assertTrue(du.is_vllm(detected, f))
        self.assertFalse(du.is_torchtpu(detected, f))
        self.assertTrue(du.is_torchtpu(torchtpu, f))

        ctx = synthetic_events.serving_context(
            serving_config={
                'MLA_KV_PACKING_SIZE': ' 8 ',
                'gpu-memory-utilization': '0.95',
            })
        self.assertEqual(
            du.knob_int(ctx, 'vllm_tpu.env.MLA_KV_PACKING_SIZE', 32), 8)
        self.assertEqual(
            du.knob_float(ctx, 'vllm.cli.gpu_memory_utilization', 0.5), 0.95)

    def test_kv_cache_dtype_and_dequant(self):
        f = _default()
        self.assertEqual(du.kv_cache_dtype(f), 'bf16')
        fp8_meta = f.meta.assign(
            kernel_text=f.meta['kernel_text'].str.replace('bf16', 'f8e4m3fn'))
        self.assertEqual(
            du.kv_cache_dtype(dataclasses.replace(f, meta=fp8_meta)), 'fp8')
        int8_meta = f.meta.assign(
            kernel_text=f.meta['kernel_text'].str.replace('bf16', 's8'))
        self.assertEqual(
            du.kv_cache_dtype(dataclasses.replace(f, meta=int8_meta)), 'int8')

        ctx_fp8 = synthetic_events.serving_context(
            serving_config={'kv_cache_dtype': 'fp8'})
        found, why = du.kv_dequant_outside_kernel(f, ctx_fp8)
        self.assertTrue(found)
        self.assertIn('kv_cache_dtype=fp8', why)

    def test_phase_stats_and_evidence(self):
        f = _default()
        self.assertEqual(du.stat(f, segmenter.DECODE, 'n_steps_per_device'), 8)
        self.assertTrue(math.isnan(du.stat(f, 'no_such_phase', 'mean_dur_ns')))
        self.assertEqual(du.active_phases(f), [segmenter.DECODE])
        ev = du.step_evidence(f, segmenter.DECODE, _SID)
        self.assertEqual(ev.metric, 'decode step time (mean)')
        self.assertIsNone(ev.xprof_link)


class ContextAndSpeculativeSegmenterTest(unittest.TestCase):

    def test_parse_serving_command_dcp_pcp_speculative_and_short_flags(self):
        cmd = (
            'TPU_SMALL_ALL_REDUCE_MAX_ROWS=64 vllm serve /model -tp 8 '
            '--decode-context-parallel-size 16 '
            '--prefill-context-parallel-size 2 '
            '--kv-cache-memory-bytes 36000000000 '
            '--speculative-config \'{"method":"mtp","num_speculative_tokens":3}\''
        )
        cfg = context.parse_serving_command(cmd)
        self.assertEqual(cfg['TPU_SMALL_ALL_REDUCE_MAX_ROWS'], '64')
        self.assertEqual(cfg['tensor-parallel-size'], '8')
        self.assertEqual(cfg['decode-context-parallel-size'], '16')
        self.assertEqual(cfg['prefill-context-parallel-size'], '2')
        ctx = context.ServingContext(serving_config=cfg).finalize()
        self.assertEqual(ctx.tp, 8)
        self.assertEqual(ctx.dcp, 16)
        self.assertEqual(ctx.pcp, 2)
        self.assertEqual(ctx.kv_cache_memory_bytes, 36_000_000_000)
        self.assertEqual(ctx.speculative_method, 'mtp')
        self.assertEqual(ctx.num_speculative_tokens, 3)
        self.assertEqual(ctx.decode_tokens_per_req, 4)

    def test_async_scheduling_log_and_d29(self):
        self.assertTrue(
            bool(
                context.async_scheduling_from_log(
                    'INFO: Asynchronous scheduling is enabled')))
        self.assertFalse(
            bool(
                context.async_scheduling_from_log(
                    'WARNING: Async scheduling disabled')))
        self.assertFalse(bool(context.async_scheduling_from_log('')))
        ctx_on = synthetic_events.serving_context(
            server_log='INFO: Asynchronous scheduling is enabled')
        self.assertEqual(
            catalog_detectors.d29_async_scheduling_off(_default(), ctx_on,
                                                       _SID),
            [],
        )

    def test_kv_transfer_role_marks_disaggregated_decode_engine(self):
        for role, is_decode in (('kv_consumer', True), ('kv_producer', False),
                                ('kv_both', False)):
            cmd = ('vllm serve /model -tp 8 --kv-transfer-config'
                   ' \'{"kv_connector":"TPURaidenConnector","kv_role":"'
                   f'{role}"}}\'')
            ctx = context.ServingContext(serving_command=cmd).finalize()
            with self.subTest(role=role):
                self.assertEqual(ctx.kv_role, role)
                self.assertEqual(ctx.is_pd_decode, is_decode)
        unified = context.ServingContext(
            serving_command='vllm serve /model -tp 8').finalize()
        self.assertIsNone(unified.kv_role)
        self.assertFalse(unified.is_pd_decode)

    def test_speculative_tokens_keep_mtp_decode_bucket_in_decode_phase(self):
        # 16 requests x 4 tokens/req (3 MTP draft tokens) = 64-token step bucket.
        events = _host_tokens(
            synthetic_events.make_events('DDD', decode_reqs=16), 16, 64)
        ctx_no_spec = synthetic_events.serving_context(
            concurrency=16, max_num_seqs=16).finalize()
        feats_no_spec = synthetic_events.compute_features(events, ctx_no_spec)
        self.assertEqual(set(feats_no_spec.seg.steps['phase']),
                         {segmenter.MIXED})

        ctx = synthetic_events.serving_context(
            concurrency=16, max_num_seqs=16,
            num_speculative_tokens=3).finalize()
        feats = synthetic_events.compute_features(events, ctx)
        self.assertEqual(set(feats.seg.steps['phase']), {segmenter.DECODE})

    def test_mtp_step_short_of_full_drafts_is_mixed(self):
        # 63 tokens for 16 requests: MTP gives every decoding request 4 tokens,
        # so the step holds a short prefill chunk beside the decodes.
        events = _host_tokens(
            synthetic_events.make_events('DDD', decode_reqs=16), 16, 63)
        mtp = synthetic_events.serving_context(concurrency=16,
                                               max_num_seqs=16,
                                               num_speculative_tokens=3,
                                               speculative_method='mtp')
        feats = synthetic_events.compute_features(events, mtp)
        self.assertEqual(set(feats.seg.steps['phase']), {segmenter.MIXED})

        # N-gram drafts vary in length per request: 4 tokens is only a bound.
        ngram = synthetic_events.serving_context(concurrency=16,
                                                 max_num_seqs=16,
                                                 num_speculative_tokens=3,
                                                 speculative_method='ngram')
        feats = synthetic_events.compute_features(events, ngram)
        self.assertEqual(set(feats.seg.steps['phase']), {segmenter.DECODE})


class AsymmetricDcpAndMlaGuardTest(unittest.TestCase):

    def _indexer_features(self, decode_bucket: int = 64) -> features.Features:
        events = _with_op(
            synthetic_events.make_events('DDD'),
            'RPA-bq_32.1',
            hlo_op='streamidx_decode.1',
            tf_op_name='jit(step_fun)/streamidx_decode',
        )
        if decode_bucket != synthetic_events.DECODE_BUCKET:
            dec = events['hlo_module'] == synthetic_events.DECODE_MODULE
            events.loc[dec, 'kernel_name'] = events.loc[
                dec, 'kernel_name'].str.replace(
                    f'[{synthetic_events.DECODE_BUCKET},',
                    f'[{decode_bucket},',
                    regex=False,
                )
        return synthetic_events.compute_features(events)

    def _dcp16_ctx(self, **overrides: object) -> context.ServingContext:
        defaults: dict[str, object] = dict(
            framework='vllm_torchtpu',
            tp=8,
            pcp=2,
            dcp=16,
            concurrency=16,
            max_model_len=262144,
            kv_cache_memory_bytes=36_000_000_000,
            serving_config={'kv-cache-dtype': 'fp8'},
            kv_role='kv_consumer',
        )
        defaults.update(overrides)
        ctx = synthetic_events.serving_context(**defaults)
        ctx.model.num_layers = 78
        ctx.model.kv_lora_rank = 512
        return ctx.finalize()

    def test_d18_recommends_decode_only_dcp_8_at_low_concurrency(self):
        found = detectors.d18_indexer(self._indexer_features(64),
                                      self._dcp16_ctx(), _SID)
        self.assertEqual(_knobs(found),
                         {'--decode-context-parallel-size': '8'})
        self.assertEqual(found[0].risk, findings.Risk.PERF_ONLY)
        self.assertIn('decode engine only', found[0].validation)

    def test_d18_lowers_dcp_only_on_a_disaggregated_decode_engine(self):
        # A unified engine (no kv_role) or the prefill engine of a P/D pair
        # also holds the prefix cache, and DCP 8 halves its KV capacity.
        for kv_role in (None, 'kv_producer'):
            with self.subTest(kv_role=kv_role):
                found = detectors.d18_indexer(
                    self._indexer_features(64),
                    self._dcp16_ctx(kv_role=kv_role),
                    _SID,
                )
                self.assertEqual(_knobs(found), {})
                self.assertEqual(found[0].risk, findings.Risk.DIAGNOSTIC)

    def test_d18_stays_diagnostic_when_batch_is_large_or_kv_overflows(self):
        large = detectors.d18_indexer(
            self._indexer_features(192),
            self._dcp16_ctx(concurrency=48),
            _SID,
        )
        self.assertEqual(_knobs(large), {})
        self.assertEqual(large[0].risk, findings.Risk.DIAGNOSTIC)

        tight_kv = detectors.d18_indexer(
            self._indexer_features(64),
            self._dcp16_ctx(kv_cache_memory_bytes=15_000_000_000),
            _SID,
        )
        self.assertEqual(_knobs(tight_kv), {})
        self.assertEqual(tight_kv[0].risk, findings.Risk.DIAGNOSTIC)

    def test_d20_suppressed_under_context_parallel_or_tight_kv_budget(self):
        ctx_dcp = self._dcp16_ctx()
        self.assertEqual(
            detectors.d20_mla_attention_tp(_default(), ctx_dcp, _SID), [])

        # TP=8, 16 sequences: 2 per rank x 202752 tokens of fp8 latent KV
        # (78 layers x 576) is ~18.2 GB per rank.
        ctx_tp8 = synthetic_events.serving_context(
            tp=8,
            concurrency=16,
            max_model_len=202752,
            kv_cache_memory_bytes=20_000_000_000,
            serving_config={'kv-cache-dtype': 'fp8'},
        )
        ctx_tp8.model.num_layers = 78
        ctx_tp8.model.kv_lora_rank = 512
        ctx_tp8 = ctx_tp8.finalize()
        self.assertTrue(
            detectors.d20_mla_attention_tp(_default(), ctx_tp8, _SID))

        ctx_pcp = dataclasses.replace(ctx_tp8, pcp=2).finalize()
        self.assertEqual(
            detectors.d20_mla_attention_tp(_default(), ctx_pcp, _SID), [])

        ctx_tight = dataclasses.replace(
            ctx_tp8, kv_cache_memory_bytes=12_000_000_000).finalize()
        self.assertEqual(
            detectors.d20_mla_attention_tp(_default(), ctx_tight, _SID), [])

    def test_mla_kv_dim_reads_qk_rope_head_dim_when_present(self):
        # Default fallback when qk_rope_head_dim is absent is 64 (512 + 64 = 576).
        default_model = context.ModelConfig.from_hf_dict({
            'num_hidden_layers': 78,
            'kv_lora_rank': 512,
        })
        self.assertIsNone(default_model.qk_rope_head_dim)
        self.assertEqual(default_model.mla_kv_dim, 576)

        # Explicit qk_rope_head_dim=128 widens per-token MLA KV from 576 to 640:
        # at 16 seqs across DCP=8 (262144 tokens, 78 layers, fp8), per-rank KV
        # grows from ~23.56 GB (fits 25 GB) to ~26.17 GB (overflows 25 GB).
        wide_rope_model = context.ModelConfig.from_hf_dict({
            'text_config': {
                'num_hidden_layers': 78,
                'kv_lora_rank': 512,
                'qk_rope_head_dim': 128,
            }
        })
        self.assertEqual(wide_rope_model.qk_rope_head_dim, 128)
        self.assertEqual(wide_rope_model.mla_kv_dim, 640)

        ctx_default = self._dcp16_ctx(kv_cache_memory_bytes=25_000_000_000)
        ctx_default.model = default_model
        found_default = detectors.d18_indexer(self._indexer_features(64),
                                              ctx_default, _SID)
        self.assertEqual(_knobs(found_default),
                         {'--decode-context-parallel-size': '8'})

        ctx_wide = self._dcp16_ctx(kv_cache_memory_bytes=25_000_000_000)
        ctx_wide.model = wide_rope_model
        found_wide = detectors.d18_indexer(self._indexer_features(64),
                                           ctx_wide, _SID)
        self.assertEqual(_knobs(found_wide), {})
        self.assertEqual(found_wide[0].risk, findings.Risk.DIAGNOSTIC)


class NewCatalogDetectorsTest(unittest.TestCase):

    def test_d53_prefill_microbatch_overlap_is_diagnostic(self):
        f = synthetic_events.compute_features(
            synthetic_events.make_events(pattern='DDDPPP', barrier_us=250))
        ctx = synthetic_events.serving_context(
            max_num_batched_tokens=4096,
            input_len=4096,
            output_len=512,
            serving_config={},
        )
        (finding, ) = catalog_detectors.d53_prefill_microbatch_overlap(
            f, ctx, _SID)
        self.assertEqual(finding.detector_id, 'D53_prefill_microbatch_overlap')
        self.assertEqual(finding.knob_changes, [])
        self.assertEqual(finding.risk, findings.Risk.DIAGNOSTIC)

        short_ctx = dataclasses.replace(ctx, input_len=1024)
        self.assertEqual(
            catalog_detectors.d53_prefill_microbatch_overlap(
                f, short_ctx, _SID),
            [],
        )

    def test_d47_small_decode_allreduce(self):
        ctx = synthetic_events.serving_context(framework='vllm_torchtpu',
                                               serving_config={})
        (finding, ) = catalog_detectors.d47_small_decode_allreduce(
            _default(), ctx, _SID)
        self.assertEqual(finding.detector_id, 'D47_small_decode_allreduce')
        self.assertEqual(
            _knobs([finding]),
            {
                'TPU_SMALL_ALL_REDUCE_MAX_ROWS':
                str(synthetic_events.DECODE_BUCKET)
            },
        )

        events = synthetic_events.make_events()
        events.loc[events['hlo_op'] == 'psum.1',
                   'hlo_op'] = ('small_all_reduce_twoshot_r16.1')
        f_pallas = synthetic_events.compute_features(events)
        self.assertEqual(
            catalog_detectors.d47_small_decode_allreduce(f_pallas, ctx, _SID),
            [],
        )

    @staticmethod
    def _qk_norm_rope_events() -> pd.DataFrame:
        """Default events with Q/K norm, RoPE and latent-projection scopes."""
        events = synthetic_events.make_events()
        events.loc[events['hlo_op'] == 'top_k.1',
                   'tf_op_name'] = ('jit_model/q_norm/reduce')
        events.loc[events['hlo_op'] == 'gather.1',
                   'tf_op_name'] = ('jit_model/rotary_emb/mul')
        events.loc[events['hlo_op'] == 'gmm-g_20.1', 'tf_op_name'] = (
            'jit_model/layers.3/mlp/routed_expert_down_proj/dot_general')
        return events

    def test_d48_unfused_qk_norm_rope_and_d49_latent_proj(self):
        events = self._qk_norm_rope_events()
        f = synthetic_events.compute_features(events)
        ctx = synthetic_events.serving_context(framework='vllm_torchtpu',
                                               serving_config={})
        ctx.model.name = 'qwen3_next'

        d48 = catalog_detectors.d48_unfused_qk_norm_rope(f, ctx, _SID)
        self.assertEqual(_knobs(d48), {'TPU_FUSED_QK_NORM_ROPE': '1'})

        d49 = catalog_detectors.d49_replicated_latent_moe_proj(f, ctx, _SID)
        self.assertEqual(_knobs(d49), {'TPU_LATENT_PROJ_INTRA_CHIP_TP': '1'})

        events.loc[events['hlo_op'] == 'psum.1',
                   'hlo_op'] = ('qk_norm_rope_gate.1')
        f_fused = synthetic_events.compute_features(events)
        self.assertEqual(
            catalog_detectors.d48_unfused_qk_norm_rope(f_fused, ctx, _SID), [])

    def test_d48_proposes_the_fused_kernel_only_for_qwen3_next(self):
        f = synthetic_events.compute_features(self._qk_norm_rope_events())
        for name, proposes in (('Qwen/Qwen3-Next-80B-A3B-Instruct', True),
                               ('zai-org/GLM-5-FP8', False), ('', False)):
            ctx = synthetic_events.serving_context(framework='vllm_torchtpu',
                                                   serving_config={})
            ctx.model.name = name
            with self.subTest(model=name):
                (finding, ) = catalog_detectors.d48_unfused_qk_norm_rope(
                    f, ctx, _SID)
                if proposes:
                    self.assertEqual(_knobs([finding]),
                                     {'TPU_FUSED_QK_NORM_ROPE': '1'})
                    self.assertEqual(finding.risk, findings.Risk.PERF_ONLY)
                else:
                    self.assertEqual(finding.knob_changes, [])
                    self.assertEqual(finding.risk, findings.Risk.DIAGNOSTIC)
                    self.assertIn('Not proposed', finding.summary)

    def test_d50_small_allreduce_form(self):
        events = synthetic_events.make_events()
        rows = events['hlo_op'] == 'psum.2'
        events.loc[rows, 'hlo_op'] = 'small_all_reduce_twoshot_r16.2'
        events.loc[rows, 'self_time_ns'] = int(200 * synthetic_events.US)
        f = synthetic_events.compute_features(events)
        ctx = synthetic_events.serving_context(framework='vllm_torchtpu',
                                               serving_config={})
        (finding, ) = catalog_detectors.d50_small_allreduce_form(f, ctx, _SID)
        self.assertEqual(finding.detector_id, 'D50_small_allreduce_form')
        self.assertEqual(_knobs([finding]),
                         {'TPU_SMALL_ALL_REDUCE_FORM': 'oneshot'})

    @staticmethod
    def _d52_features() -> features.Features:
        """Decode steps in a 128-row bucket (above the 64-row limit)."""
        events = synthetic_events.make_events(decode_reqs=32)
        dec_rows = events['hlo_module'] == synthetic_events.DECODE_MODULE
        events.loc[dec_rows,
                   'kernel_name'] = (events.loc[dec_rows,
                                                'kernel_name'].str.replace(
                                                    '[64,',
                                                    '[128,',
                                                    regex=False).str.replace(
                                                        '[512,',
                                                        '[1024,',
                                                        regex=False))
        return synthetic_events.compute_features(events)

    @staticmethod
    def _d52_ctx(**overrides: object) -> context.ServingContext:
        ctx = synthetic_events.serving_context(framework='vllm_torchtpu',
                                               serving_config={},
                                               **overrides)
        ctx.model.kv_lora_rank = 512
        return ctx

    def test_d52_moe_tp_decode_row_limit(self):
        f_128 = self._d52_features()
        ctx = self._d52_ctx()

        (finding, ) = catalog_detectors.d52_moe_tp_decode_row_limit(
            f_128, ctx, _SID)
        self.assertEqual(finding.detector_id, 'D52_moe_tp_decode_row_limit')
        self.assertEqual(_knobs([finding]),
                         {'--enable-expert-parallel': 'true'})

        self.assertEqual(
            catalog_detectors.d52_moe_tp_decode_row_limit(
                _default(), ctx, _SID),
            [],
        )
        self.assertEqual(
            catalog_detectors.d52_moe_tp_decode_row_limit(
                f_128, dataclasses.replace(ctx, ep=8), _SID),
            [],
        )

    def test_d52_needs_half_of_decode_time_above_64_rows(self):
        f_128 = self._d52_features()
        ctx = self._d52_ctx()
        steps = f_128.seg.steps
        is_dec = steps['phase'] == segmenter.DECODE
        starts = sorted(steps.loc[is_dec, 'start_ns'].unique())

        def with_small(n_small: int) -> features.Features:
            # The first `n_small` decode steps (all devices) back at 64 rows.
            st = steps.copy()
            st.loc[is_dec & st['start_ns'].isin(starts[:n_small]),
                   'bucket'] = 64
            return dataclasses.replace(f_128,
                                       seg=dataclasses.replace(f_128.seg,
                                                               steps=st))

        one_large = with_small(len(starts) - 1)  # 1 of 8 steps above 64 rows.
        self.assertEqual(
            catalog_detectors.d52_moe_tp_decode_row_limit(
                one_large, ctx, _SID), [])
        half_large = with_small(len(starts) // 2)
        (finding, ) = catalog_detectors.d52_moe_tp_decode_row_limit(
            half_large, ctx, _SID)
        self.assertIn(('decode time in steps above 64 rows', 50.0),
                      [(e.metric, e.value) for e in finding.evidence])

    def test_d52_not_suppressed_on_disaggregated_decode_engine(self):
        f_128 = self._d52_features()
        prefill_heavy = self._d52_ctx(input_len=32768, output_len=1024)
        self.assertEqual(
            catalog_detectors.d52_moe_tp_decode_row_limit(
                f_128, prefill_heavy, _SID), [])
        pd_decode = dataclasses.replace(prefill_heavy, kv_role='kv_consumer')
        (finding, ) = catalog_detectors.d52_moe_tp_decode_row_limit(
            f_128, pd_decode, _SID)
        self.assertEqual(_knobs([finding]),
                         {'--enable-expert-parallel': 'true'})

    def test_d31_moe_tp_decode_width_and_small_batch_guard(self):
        events = _with_op(
            synthetic_events.make_events('DDD', decode_reqs=16),
            'gmm-g_20.1',
            hlo_op='moe_tp_decode-e_256-m_16-h_6144-i_128-n_16.1',
        )
        ctx_small = synthetic_events.serving_context(framework='vllm_torchtpu',
                                                     tp=16,
                                                     concurrency=16)
        ctx_small.model.moe_intermediate_size = 2048
        f_small = synthetic_events.compute_features(events, ctx_small)
        (d31_small, ) = catalog_detectors.d31_gmm_tile_misalignment(
            f_small, ctx_small, _SID)
        self.assertEqual(d31_small.confidence, findings.Confidence.MEDIUM)
        self.assertIn('Full EP is not suggested while decode steps stay',
                      d31_small.kernel_followup or '')

        # Whole-server concurrency=48 across DP=2 with spec=1 is 96 total rows,
        # but each DP replica runs ceil(48 / 2) * 2 = 48 rows (<= 64).
        ctx_dp2 = dataclasses.replace(ctx_small,
                                      dp=2,
                                      concurrency=48,
                                      num_speculative_tokens=1,
                                      speculative_method='mtp')
        events_dp2 = _host_tokens(
            _with_op(
                synthetic_events.make_events('DDD', decode_reqs=24),
                'gmm-g_20.1',
                hlo_op='moe_tp_decode-e_256-m_48-h_6144-i_128-n_16.1',
            ), 24, 48)
        f_dp2 = synthetic_events.compute_features(events_dp2, ctx_dp2)
        self.assertEqual(set(f_dp2.seg.steps['phase']), {segmenter.DECODE})
        (d31_dp2, ) = catalog_detectors.d31_gmm_tile_misalignment(
            f_dp2, ctx_dp2, _SID)
        self.assertIn('Full EP is not suggested while decode steps stay',
                      d31_dp2.kernel_followup or '')

        # 20 sequences x (1 + 3 MTP tokens) = 80 rows: past the 64-row limit
        # D52 proposes EP for, so D31 must not advise against EP.
        ctx_mtp = dataclasses.replace(ctx_small,
                                      concurrency=20,
                                      num_speculative_tokens=3,
                                      speculative_method='mtp')
        events_mtp = _host_tokens(
            _with_op(
                synthetic_events.make_events('DDD', decode_reqs=20),
                'gmm-g_20.1',
                hlo_op='moe_tp_decode-e_256-m_20-h_6144-i_128-n_16.1',
            ), 20, 80)
        f_mtp = synthetic_events.compute_features(events_mtp, ctx_mtp)
        self.assertEqual(set(f_mtp.seg.steps['phase']), {segmenter.DECODE})
        (d31_mtp, ) = catalog_detectors.d31_gmm_tile_misalignment(
            f_mtp, ctx_mtp, _SID)
        self.assertIn('--enable-expert-parallel', d31_mtp.kernel_followup
                      or '')
        self.assertNotIn('not suggested', d31_mtp.kernel_followup or '')

    def test_planner_rechecks_d52_and_dcp_detectors(self):
        active = [
            {
                'detector_id': 'D52_moe_tp_decode_row_limit'
            },
            {
                'detector_id': 'D18_attn_indexer'
            },
            {
                'detector_id': 'D20_mla_attention_tp'
            },
        ]
        recheck_ep = planner.compute_recheck_findings(
            {
                'tp': 8,
                'ep': 8,
                'dcp': 16
            },
            {
                'tp': 8,
                'ep': 1,
                'dcp': 16
            },
            active,
        )
        self.assertEqual(recheck_ep, ['D52_moe_tp_decode_row_limit'])

        recheck_dcp = planner.compute_recheck_findings(
            {
                'tp': 8,
                'ep': 1,
                'dcp': 8
            },
            {
                'tp': 8,
                'ep': 1,
                'dcp': 16
            },
            active,
        )
        self.assertEqual(recheck_dcp,
                         ['D18_attn_indexer', 'D20_mla_attention_tp'])


if __name__ == '__main__':
    unittest.main()
