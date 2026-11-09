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
"""Synthetic end-to-end tests for the serving analysis pipeline.

Builds small Events DB-shaped frames that mimic a vLLM-TPU MoE trace (8
TensorCores, decode + prefill buckets, host annotations) and checks that the
segmenter and detectors fire (or stay quiet) as expected.
"""

import dataclasses
import json
import os
import sys
import types
from unittest import mock

import pandas as pd
from absl.testing import absltest

# The serving_analysis package sits one directory up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving_analysis import (analyze, catalog_detectors, context, detectors,
                              findings, guardrails, headroom_detectors,
                              intervals, knobs, op_classes, segmenter,
                              trace_table)

_TESTDATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testdata')
_US = 1000
_HIDDEN = 1024
_N_DEV = 8
_MOE = 'jit(step_fun)/SharedFusedMoE/jit(fused_moe_func)/shard_map'


@dataclasses.dataclass
class _Op:
    name: str
    dtype: str
    dims: tuple[int, ...]
    opcode: str
    scope: str
    tf_type: str
    category: str
    us: float


def _step_ops(t: int, dev: int, scale: float, barrier_us: float,
              gmm_skew: float) -> list[_Op]:
    return [
        _Op(
            'fusion.1',
            'bf16',
            (t, _HIDDEN),
            'fusion',
            'jit(step_fun)/QKVParallelLinear/dot_general',
            'dot_general',
            'convolution fusion',
            200 * scale,
        ),
        _Op(
            'RPA-bq_32.1',
            'bf16',
            (t, 8, 128),
            'custom-call',
            'jit(step_fun)/jit(_jax_attn_func)/shard_map/'
            'jit(ragged_paged_attention)/RPA-bq_32',
            'pallas_call',
            'custom-call',
            300 * scale,
        ),
        _Op(
            'psum.1',
            'bf16',
            (t, _HIDDEN),
            'all-reduce',
            'jit(step_fun)/RowParallelLinear/shard_map/psum',
            'psum',
            'all-reduce',
            100 * scale,
        ),
        _Op(
            'top_k.1',
            's32',
            (t, 8),
            'custom-call',
            f'{_MOE}/top_k',
            'top_k',
            'sort',
            60 * scale,
        ),
        _Op(
            'gather.1',
            'bf16',
            (t * 8, _HIDDEN),
            'gather',
            f'{_MOE}/gather',
            'gather',
            'async-done',
            150 * scale,
        ),
        _Op(
            'gmm-g_20.1',
            'bf16',
            (t * 8, _HIDDEN),
            'custom-call',
            f'{_MOE}/jit(gmm_pipeline)/pallas_call',
            'pallas_call',
            'custom-call',
            300 * scale * (gmm_skew if dev == 3 else 1.0),
        ),
        _Op(
            'psum.2',
            'bf16',
            (t, _HIDDEN),
            'all-reduce',
            f'{_MOE}/psum',
            'psum',
            'all-reduce',
            (100 + barrier_us) * scale,
        ),
    ]


def _row(**kw) -> dict[str, object]:
    base: dict[str, object] = dict(
        host='h0',
        category='',
        kernel_name='',
        hlo_module='',
        hlo_op='',
        tf_op_name='',
        tf_op_type='',
        trace_args='',
        thread_name='',
        source_line='',
    )
    base.update(kw)
    base['self_time_ns'] = kw.get('self_time_ns',
                                  kw['end_ns'] - kw['start_ns'])
    return base


def make_events(
    pattern: str = 'DDDPDDPDDD',
    idle_us: float = 400,
    barrier_us: float = 150,
    gmm_skew: float = 1.6,
    decode_reqs: int = 32,
) -> pd.DataFrame:
    rows = []
    modules = {
        'D': ('jit_step_fun(111)', 64, 1.0),
        'P': ('jit_step_fun(222)', 2048, 20.0),
    }
    t0 = 1_000 * _US
    for kind in pattern:
        mod, bucket, scale = modules[kind]
        reqs, toks = (decode_reqs, decode_reqs) if kind == 'D' else (4, bucket)
        rows.append(
            _row(
                device='cpu:0',
                category='host',
                thread_name='VLLM::EngineCor',
                kernel_name=f'execute_model: {reqs} reqs, {toks} toks',
                start_ns=t0 - 100 * _US,
                end_ns=t0 + 20 * _US,
            ))
        step_end = t0
        for dev in range(_N_DEV):
            cur = t0
            for op in _step_ops(bucket, dev, scale, barrier_us, gmm_skew):
                d = int(op.us * _US)
                dims = ','.join(map(str, op.dims))
                rows.append(
                    _row(
                        device=f'TPU:{dev}',
                        category='XLA Ops',
                        hlo_module=mod,
                        hlo_op=op.name,
                        kernel_name=
                        (f'%{op.name} = {op.dtype}[{dims}]{{1,0}} {op.opcode}(%p.0)'
                         ),
                        tf_op_name=op.scope,
                        tf_op_type=op.tf_type,
                        trace_args=f'hlo_category={op.category},program_id=1',
                        start_ns=cur,
                        end_ns=cur + d,
                    ))
                cur += d
            rows.append(
                _row(
                    device=f'TPU:{dev}',
                    category='XLA Modules',
                    kernel_name=f'HLO Module:{mod}',
                    start_ns=t0,
                    end_ns=cur,
                ))
            rows.append(
                _row(
                    device=f'TPU:{dev}',
                    category='XLA Modules',
                    kernel_name='HLO Module:jit_compute_logits_func(333)',
                    start_ns=cur,
                    end_ns=cur + 50 * _US,
                ))
            step_end = max(step_end, cur + 50 * _US)
        rows.append(
            _row(
                device='cpu:0',
                category='host',
                thread_name='VLLM::EngineCor',
                kernel_name='$tpu_runner.py:683 _execute_model',
                start_ns=step_end,
                end_ns=step_end + int(idle_us * _US),
            ))
        t0 = step_end + int(idle_us * _US)
    return pd.DataFrame(rows)


def _ctx(**kw) -> context.ServingContext:
    model = context.ModelConfig(
        hidden_size=_HIDDEN,
        num_kv_heads=8,
        num_experts=160,
        num_experts_per_tok=8,
        max_position_embeddings=262144,
    )
    kw.setdefault('serving_config', {})
    return context.ServingContext(model=model, tp=8, **kw)


def _ids(result) -> set[str]:
    return {f.detector_id for f in result.findings}


class IntervalsTest(absltest.TestCase):

    def test_union_gaps(self):
        u = intervals.union([[0, 5], [3, 8], [10, 12]])
        self.assertEqual(u.tolist(), [[0, 8], [10, 12]])
        self.assertEqual(intervals.gaps(u).tolist(), [[8, 10]])
        self.assertEqual(intervals.total_length(u), 10)
        self.assertEqual(
            intervals.subtract([[0, 10]], [[2, 3], [5, 7]]).tolist(),
            [[0, 2], [3, 5], [7, 10]],
        )


class OpClassesTest(absltest.TestCase):

    def test_classify(self):
        t = trace_table.TraceTable.from_events_frame(make_events('D'))
        meta = op_classes.classify(t.op_meta).set_index('op')
        self.assertEqual(meta.loc['psum.1', 'op_class'], op_classes.COLLECTIVE)
        self.assertFalse(meta.loc['psum.1', 'moe_scope'])
        self.assertTrue(meta.loc['psum.2', 'moe_scope'])
        self.assertEqual(meta.loc['top_k.1', 'op_class'],
                         op_classes.MOE_TOPK_SORT)
        self.assertEqual(meta.loc['gather.1', 'op_class'],
                         op_classes.MOE_PERMUTE)
        self.assertEqual(meta.loc['gmm-g_20.1', 'op_class'],
                         op_classes.MOE_GMM)
        self.assertEqual(meta.loc['RPA-bq_32.1', 'op_class'],
                         op_classes.ATTENTION_KERNEL)
        self.assertEqual(meta.loc['psum.1', 'out_dims'], (64, _HIDDEN))

    def test_attention_scope_copy_is_data_formatting(self):
        attn = 'jit(step_fun)/jit(_jax_attn_func)/shard_map'
        self.assertEqual(
            op_classes.classify_row(
                'copy.3',
                '%copy.3 = bf16[64,8,128]{2,1,0} copy(%p.0)',
                f'{attn}/copy',
                'copy',
                'copy',
            )[0],
            op_classes.DATA_FORMATTING,
        )
        self.assertEqual(
            op_classes.classify_row(
                'xpose_full_shape_1.1',
                '',
                f'{attn}/xpose_full_shape_1',
                'pallas_call',
                'custom-call',
            )[0],
            op_classes.DATA_FORMATTING,
        )
        self.assertEqual(
            op_classes.classify_row('fusion.9', '', f'{attn}/mul', 'mul',
                                    'loop fusion')[0],
            op_classes.ATTENTION_OTHER,
        )

    def test_split_rpa_kernels_are_attention(self):
        for name in ('RPAd-p256', 'RPAp-p256', 'RPAm-p256'):
            self.assertEqual(
                op_classes.classify_row(
                    f'{name}.1',
                    '',
                    f'jit(step_fun)/{name}',
                    'pallas_call',
                    'custom-call',
                )[0],
                op_classes.ATTENTION_KERNEL,
            )

    def test_moe_bincount_and_sparsecore_permute(self):
        self.assertEqual(
            op_classes.classify_row(
                'scatter.5',
                '%scatter.5 = s32[160]{0} scatter(%p.0)',
                f'{_MOE}/scatter-add',
                'scatter-add',
                'scatter',
            )[0],
            op_classes.MOE_OTHER,
        )
        self.assertEqual(
            op_classes.classify_row(
                'scatter.6',
                f'%scatter.6 = bf16[512,{_HIDDEN}]{{1,0}} scatter(%p.0)',
                f'{_MOE}/scatter-add',
                'scatter-add',
                'scatter',
            )[0],
            op_classes.MOE_PERMUTE,
        )
        for name in (
                'sc_ragged_gather_v2',
                'sc_ragged_scatter',
                'sc_ragged_gather_reduce_v2',
        ):
            self.assertEqual(
                op_classes.classify_row(
                    f'{name}.1',
                    '',
                    f'jit(step_fun)/{name}',
                    'pallas_call',
                    'custom-call',
                )[0],
                op_classes.MOE_PERMUTE,
            )


class SegmenterTest(absltest.TestCase):

    def test_buckets_and_phases(self):
        t = trace_table.TraceTable.from_events_frame(make_events())
        seg = segmenter.segment(t, _ctx())
        self.assertEqual(seg.framework, 'vllm_tpu')
        self.assertEqual(seg.step_families, ['jit_step_fun'])
        v = seg.variants.set_index('module')
        self.assertEqual(v.loc['jit_step_fun(111)', 'bucket'], 64)
        self.assertEqual(v.loc['jit_step_fun(222)', 'bucket'], 2048)
        self.assertEqual(v.loc['jit_step_fun(111)', 'phase'], segmenter.DECODE)
        self.assertEqual(v.loc['jit_step_fun(222)', 'phase'], segmenter.MIXED)
        self.assertLen(seg.steps, 10 * _N_DEV)
        dev0 = seg.steps[seg.steps['device'] == 'TPU:0']
        self.assertEqual(dev0['reqs'].iloc[0], 32)
        # Idle = 400us gap per cycle on the slowest device (logits program counts
        # as busy). Synthetic devices don't sync at collectives, so faster ones
        # also idle for the GMM skew.
        dev3 = seg.steps[seg.steps['device'] == 'TPU:3']
        self.assertAlmostEqual(dev3['idle_ns'].dropna().iloc[0],
                               400 * _US,
                               delta=_US)

    def test_phase_from_bucket_without_annotations(self):
        ev = make_events()
        ev = ev[~ev['kernel_name'].str.startswith('execute_model')]
        seg = segmenter.segment(trace_table.TraceTable.from_events_frame(ev),
                                _ctx())
        v = seg.variants.set_index('module')
        self.assertEqual(v.loc['jit_step_fun(222)', 'phase'],
                         segmenter.PREFILL_OR_MIXED)


class DetectorsTest(absltest.TestCase):

    def test_baseline_fires_expected(self):
        t = trace_table.TraceTable.from_events_frame(make_events(), 'sid-1')
        r = analyze.analyze(t, _ctx())
        self.assertEmpty(r.errors)
        ids = _ids(r)
        for want in (
                'D1_host_gap',
                'D2_padding_waste',
                'D3_prefill_decode_interference',
                'D4_moe_topk_sort',
                'D5_moe_permute',
                'D6_expert_imbalance',
        ):
            self.assertIn(want, ids)
        d5 = next(f for f in r.findings if f.detector_id == 'D5_moe_permute'
                  and f.phase == segmenter.DECODE)
        self.assertEqual(d5.knob_changes[0].name,
                         'ONEHOT_MOE_PERMUTE_THRESHOLD')
        self.assertEqual(d5.knob_changes[0].suggested_value, str(64 * 8))
        d1 = next(f for f in r.findings if f.detector_id == 'D1_host_gap')
        self.assertEqual(d1.knob_changes[0].name, 'enable_continue_decode')
        # Traces here are local files: evidence carries no hosted link.
        self.assertIsNone(d1.evidence[0].xprof_link)
        md = r.to_markdown()
        self.assertIn('## Serving Optimizations', md)

    def test_quiet_when_healthy(self):
        ev = make_events(pattern='DDDDDD',
                         idle_us=5,
                         barrier_us=0,
                         gmm_skew=1.0,
                         decode_reqs=64)
        r = analyze.analyze(trace_table.TraceTable.from_events_frame(ev),
                            _ctx())
        self.assertEmpty(r.errors)
        ids = _ids(r)
        for unwanted in (
                'D1_host_gap',
                'D2_padding_waste',
                'D3_prefill_decode_interference',
                'D6_expert_imbalance',
        ):
            self.assertNotIn(unwanted, ids)

    def test_d8_quiet_when_kv_rows_fill_the_packing(self):
        # 8 KV heads at TP 8: one head, two KV-cache rows (K and V) per
        # device, which fills the bf16 packing of 2.
        r = analyze.analyze(
            trace_table.TraceTable.from_events_frame(make_events()), _ctx())
        self.assertNotIn('D8_kv_head_padding', _ids(r))

    def test_d8_fires_when_kv_heads_are_replicated(self):
        ctx = _ctx()
        ctx.model.num_kv_heads = 4
        r = analyze.analyze(
            trace_table.TraceTable.from_events_frame(make_events()), ctx)
        d8 = [f for f in r.findings if f.detector_id == 'D8_kv_head_padding']
        self.assertTrue(d8)
        self.assertIn('replicated on 2 devices', d8[0].summary)

    def test_d8_on_torchtpu_targets_a_tp_whose_rows_fill_the_packing(self):
        ctx = _ctx(framework='vllm_torchtpu')
        ctx.model.num_kv_heads = 4
        r = analyze.analyze(
            trace_table.TraceTable.from_events_frame(make_events()), ctx)
        got = _knob_values(r, 'D8_kv_head_padding')
        self.assertEqual((got.get('--tensor-parallel-size'),
                          got.get('--data-parallel-size')), ('4', '2'))

    def test_d8_on_torchtpu_never_targets_tp1(self):
        # MQA: only TP 1 keeps a whole KV head per device, and torch_tpu has
        # no TP 1 layout on tpu7x, so D8 reports the padding with no change.
        ctx = _ctx(framework='vllm_torchtpu')
        ctx.model.num_kv_heads = 1
        r = analyze.analyze(
            trace_table.TraceTable.from_events_frame(make_events()), ctx)
        d8 = [f for f in r.findings if f.detector_id == 'D8_kv_head_padding']
        self.assertTrue(d8)
        self.assertEqual(_knob_values(r, 'D8_kv_head_padding'), {})
        self.assertIn('Only TP 1', d8[0].summary)

    def test_kv_packing_is_32_over_bits(self):
        for dtype, packing in (('bf16', 2), ('', 2), ('fp8', 4),
                               ('fp8_e4m3', 4), ('int8', 4), ('s8', 4)):
            self.assertEqual(detectors.kv_packing(dtype), packing, dtype)

    def test_d3_never_lowers_the_token_budget_below_the_prompt(self):
        ctx = _ctx(framework='vllm_tpu',
                   input_len=8192,
                   output_len=1024,
                   max_num_batched_tokens=4096)
        r = analyze.analyze(
            trace_table.TraceTable.from_events_frame(make_events()), ctx)
        d3 = [
            f for f in r.findings
            if f.detector_id == 'D3_prefill_decode_interference'
        ]
        self.assertTrue(d3)
        self.assertNotIn('--max-num-batched-tokens',
                         _knob_values(r, 'D3_prefill_decode_interference'))
        self.assertIn('--max-num-batched-tokens stays', d3[0].summary)

    def test_d3_lowers_the_token_budget_to_no_less_than_the_prompt(self):
        ctx = _ctx(framework='vllm_tpu',
                   input_len=1024,
                   output_len=1024,
                   max_num_batched_tokens=8192)
        r = analyze.analyze(
            trace_table.TraceTable.from_events_frame(make_events()), ctx)
        self.assertEqual(
            _knob_values(r, 'D3_prefill_decode_interference').get(
                '--max-num-batched-tokens'), '4096')

    def test_d24_prefill_heavy_needs_twice_the_output(self):

        def fired(input_len):
            ctx = _ctx(input_len=input_len, output_len=1024)
            r = analyze.analyze(
                trace_table.TraceTable.from_events_frame(make_events()), ctx)
            return 'D24_moe_tp_without_ep' in _ids(r)

        self.assertTrue(fired(1536))
        self.assertFalse(fired(2048))

    def test_d34_reports_speculative_draft_share(self):
        ev = make_events()
        # Put the QKV projection inside an MTP draft block.
        qkv = ev['tf_op_name'].astype(str).str.contains('QKVParallelLinear')
        ev.loc[qkv, 'tf_op_name'] = ev.loc[qkv, 'tf_op_name'].str.replace(
            'jit(step_fun)/', 'jit(step_fun)/mtp_block/', regex=False)
        r = analyze.analyze(trace_table.TraceTable.from_events_frame(ev),
                            _ctx())
        d34 = [
            f for f in r.findings
            if f.detector_id == 'D34_speculative_draft_overhead'
        ]
        self.assertTrue(d34)
        self.assertEmpty(d34[0].knob_changes)
        self.assertNotIn(
            'D34_speculative_draft_overhead',
            _ids(
                analyze.analyze(
                    trace_table.TraceTable.from_events_frame(make_events()),
                    _ctx())))

    def test_d25_flags_host_driven_transfers(self):

        def found(**transfer):
            stub = types.SimpleNamespace(transfer=transfer)
            return detectors.d25_host_driven_transfer(stub, _ctx(), 'sid')

        (f, ) = found(n_copies=4096,
                      span_ms=200.0,
                      device_busy_ms=5.0,
                      copy_ms=150.0)
        self.assertEqual(f.detector_id, 'D25_host_driven_transfer')
        self.assertEmpty(f.knob_changes)
        # A busy device or few copies is not host-bound.
        self.assertEmpty(
            found(n_copies=4096, span_ms=200.0, device_busy_ms=100.0))
        self.assertEmpty(found(n_copies=10, span_ms=200.0, device_busy_ms=5.0))

    def test_dp_attention_suppresses_d8(self):
        ctx = _ctx()
        ctx.model.num_kv_heads = 4
        ctx.serving_config = {
            'sharding.sharding_strategy.enable_dp_attention': 'true'
        }
        r = analyze.analyze(
            trace_table.TraceTable.from_events_frame(make_events()), ctx)
        self.assertNotIn('D8_kv_head_padding', _ids(r))

    def test_already_set_knob_scores_zero(self):
        ctx = _ctx()
        ctx.serving_config = {'MOE_APPROX_TOPK': '1'}
        r = analyze.analyze(
            trace_table.TraceTable.from_events_frame(make_events()), ctx)
        d4 = [f for f in r.findings if f.detector_id == 'D4_moe_topk_sort']
        self.assertNotEmpty(d4)
        self.assertTrue(all(f.score == 0.0 for f in d4))

    def test_catalog_detectors_fire(self):
        ev = make_events()
        # Replace MoE combine psum.2 output shape with flattened [t*8, H] to
        # trigger D26, and add an eevee_draft op to trigger D34.
        mask = ev['hlo_op'] == 'psum.2'
        ev.loc[mask, 'kernel_name'] = (
            f'%psum.2 = bf16[{64 * 8},{_HIDDEN}]{{1,0}} all-reduce(%p.0)')
        ctx = _ctx(
            max_model_len=16384,
            serving_config=context.parse_serving_command(
                'vllm serve m --no-async-scheduling --block-size 32'),
        )
        ctx.model.moe_intermediate_size = 2560  # 2560 // 8 = 320 (not % 256)
        r = analyze.analyze(trace_table.TraceTable.from_events_frame(ev), ctx)
        self.assertEmpty(r.errors)
        ids = _ids(r)
        for want in (
                'D26_moe_combine_no_local_reduction',
                'D29_async_scheduling_off',
                'D30_batched_rpa',
                'D31_gmm_tile_misalignment',
                'D33_kv_cache_block_size',
        ):
            self.assertIn(want, ids)
        d29 = next(f for f in r.findings
                   if f.detector_id == 'D29_async_scheduling_off')
        self.assertEqual(d29.confidence.value, 'high')

    def _catalog_ctx(self, **kw) -> context.ServingContext:
        ctx = _ctx(
            max_model_len=16384,
            serving_config=context.parse_serving_command(
                'vllm serve m --no-async-scheduling --block-size 32'),
            **kw,
        )
        ctx.model.moe_intermediate_size = 2560
        return ctx

    def test_d30_quiet_for_head_dim_64(self):
        ctx = self._catalog_ctx()
        ctx.model.head_dim = 64
        r = analyze.analyze(
            trace_table.TraceTable.from_events_frame(make_events()), ctx)
        self.assertNotIn('D30_batched_rpa', _ids(r))

    def test_d30_quiet_for_hd64_kernel(self):
        ev = make_events()
        ev['tf_op_name'] = ev['tf_op_name'].str.replace('RPA-bq_32',
                                                        'RPA-HD_64-bq_32',
                                                        regex=False)
        r = analyze.analyze(trace_table.TraceTable.from_events_frame(ev),
                            self._catalog_ctx())
        self.assertNotIn('D30_batched_rpa', _ids(r))

    def test_d31_no_ep_suggestion_when_prefill_heavy(self):
        r = analyze.analyze(
            trace_table.TraceTable.from_events_frame(make_events()),
            self._catalog_ctx(input_len=8192, output_len=1024),
        )
        d31 = [
            f for f in r.findings
            if f.detector_id == 'D31_gmm_tile_misalignment'
        ]
        self.assertNotEmpty(d31)
        for f in d31:
            self.assertNotIn('Switch MoE from TP to EP', f.kernel_followup
                             or '')
            self.assertIn('pads to 512', f.kernel_followup or '')

    def test_d12_reports_dequant_instead_of_fp8(self):
        base = analyze.analyze(
            trace_table.TraceTable.from_events_frame(make_events()), _ctx())
        d12 = [
            f for f in base.findings if f.detector_id == 'D12_kv_cache_quant'
        ]
        self.assertNotEmpty(d12)
        self.assertEqual(d12[0].knob_changes[0].name, '--kv-cache-dtype')
        r = analyze.analyze(
            trace_table.TraceTable.from_events_frame(make_events()),
            _ctx(serving_config={'kv_cache_dtype': 'fp8'}),
        )
        d12 = [f for f in r.findings if f.detector_id == 'D12_kv_cache_quant']
        self.assertLen(d12, 1)
        self.assertEmpty(d12[0].knob_changes)
        self.assertEqual(d12[0].risk.value, 'diagnostic')
        self.assertIn('dequantized', d12[0].title)


def _knob_values(r, detector_id: str) -> dict[str, str]:
    out = {}
    for f in r.findings:
        if f.detector_id == detector_id:
            for k in f.knob_changes:
                out[k.name] = k.suggested_value
    return out


class HeadroomDetectorsTest(absltest.TestCase):

    def test_d36_full_batch_raises_memory_and_seqs(self):
        ctx = _ctx(serving_config=context.parse_serving_command(
            'vllm serve m --max-num-seqs 32'))
        with mock.patch.object(headroom_detectors, 'D36_MIN_GMM_SHARE', 0.1):
            r = analyze.analyze(
                trace_table.TraceTable.from_events_frame(make_events()), ctx)
        got = _knob_values(r, 'D36_hbm_limited_decode_batch')
        self.assertEqual(got.get('--gpu-memory-utilization'), '0.98')
        self.assertEqual(got.get('--kv-cache-dtype'), 'fp8')
        self.assertEqual(got.get('--max-num-seqs'), '64')

    def test_d36_quiet_when_load_limited(self):
        ctx = _ctx(serving_config=context.parse_serving_command(
            'vllm serve m --max-num-seqs 256'))
        with mock.patch.object(headroom_detectors, 'D36_MIN_GMM_SHARE', 0.1):
            r = analyze.analyze(
                trace_table.TraceTable.from_events_frame(make_events()), ctx)
        self.assertNotIn('D36_hbm_limited_decode_batch', _ids(r))

    def test_d37_moe_weights(self):
        r = analyze.analyze(
            trace_table.TraceTable.from_events_frame(make_events()), _ctx())
        got = _knob_values(r, 'D37_moe_weight_precision')
        self.assertEqual(got.get('MOE_REQUANTIZE_WEIGHT_DTYPE'), 'fp4')
        self.assertEqual(got.get('MOE_REQUANTIZE_BLOCK_SIZE'), '512')
        r = analyze.analyze(
            trace_table.TraceTable.from_events_frame(make_events()),
            _ctx(serving_config={'MOE_REQUANTIZE_WEIGHT_DTYPE': 'fp4'}),
        )
        self.assertNotIn('D37_moe_weight_precision', _ids(r))

    def test_d40_mla_v1_kernel_and_packing(self):
        ev = make_events()
        for col in ('hlo_op', 'kernel_name', 'tf_op_name'):
            ev[col] = ev[col].str.replace('RPA-bq_32',
                                          'MLA-RPA-bq_32',
                                          regex=False)
        ctx = _ctx(serving_config={'MLA_KV_PACKING_SIZE': '4'})
        ctx.model.kv_lora_rank = 512
        r = analyze.analyze(trace_table.TraceTable.from_events_frame(ev), ctx)
        d40 = [
            f for f in r.findings if f.detector_id == 'D40_mla_kernel_config'
        ]
        self.assertLen(d40, 2)
        self.assertIn('v1 kernel', d40[0].title)
        self.assertEqual(
            _knob_values(r, 'D40_mla_kernel_config'),
            {'MLA_KV_PACKING_SIZE': '32'},
        )

    def test_d44_moe_combine_not_chunked(self):
        ev = make_events(pattern='DDDPPPDDD')
        r = analyze.analyze(trace_table.TraceTable.from_events_frame(ev),
                            _ctx())
        self.assertEqual(
            _knob_values(r, 'D44_moe_combine_not_chunked'),
            {'VLLM_MOE_CHUNK_SIZE': '1024'},
        )
        r = analyze.analyze(
            trace_table.TraceTable.from_events_frame(ev),
            _ctx(serving_config={'VLLM_MOE_CHUNK_SIZE': '1024'}),
        )
        self.assertNotIn('D44_moe_combine_not_chunked', _ids(r))

    def test_d29_quiet_when_server_log_shows_async_enabled(self):
        cfg = context.parse_serving_command('vllm serve m --block-size 32')
        ev = make_events()
        r = analyze.analyze(
            trace_table.TraceTable.from_events_frame(ev),
            _ctx(serving_config=cfg),
        )
        self.assertIn('D29_async_scheduling_off', _ids(r))
        r = analyze.analyze(
            trace_table.TraceTable.from_events_frame(ev),
            _ctx(
                serving_config=cfg,
                server_log=
                ('INFO 10-14 05:31:02 [vllm.py:612] Asynchronous scheduling is'
                 ' enabled.\n'),
            ),
        )
        self.assertNotIn('D29_async_scheduling_off', _ids(r))

    def test_d45_fires_on_oversized_decode_block(self):
        ev = make_events(pattern='DDDDDD')
        for col in ('hlo_op', 'kernel_name', 'tf_op_name'):
            ev[col] = ev[col].str.replace(
                'RPA-bq_32',
                'RPAd-p_64-bq_1_1-bkv_16384_16384-sw_1024',
                regex=False)
        ctx = _ctx(
            max_model_len=16384,
            input_len=1024,
            output_len=1024,
        )
        ctx.framework = 'vllm_torchtpu'
        with mock.patch.object(catalog_detectors, 'D45_MIN_ATTN_SHARE', 0.1):
            r = analyze.analyze(trace_table.TraceTable.from_events_frame(ev),
                                ctx)
        self.assertEmpty(r.errors)
        d45 = [
            f for f in r.findings
            if f.detector_id == 'D45_rpa_decode_block_oversized'
        ]
        self.assertLen(d45, 1)
        # Sliding layer, window 1024: one window, rounded to the 64-token page.
        self.assertEqual([(k.name, k.suggested_value)
                          for k in d45[0].knob_changes],
                         [('TPU_RPA_DECODE_BLOCK_SIZES', 'sliding=1024:1024')])
        self.assertIn('window', d45[0].kernel_followup)
        # Sliding layer, window 1024, context 2048: 15/16 of the chunk unused.
        self.assertGreater(d45[0].est_step_fraction_saved, 0.0)
        # Below the attention-share gate it stays quiet.
        r = analyze.analyze(trace_table.TraceTable.from_events_frame(ev), ctx)
        self.assertNotIn('D45_rpa_decode_block_oversized', _ids(r))


class RpaDecodeBlockTest(absltest.TestCase):
    """D45 rule on real Gemma-3 27B TP8 decode traces (stock vs tuned RPA)."""

    def _decode_inputs(self, name: str) -> tuple[dict[str, float], float]:
        with open(os.path.join(_TESTDATA, name)) as f:
            result = json.load(f)['profiles'][0]['result']
        shares = {
            o['base_op']: o['share']
            for o in result['top_ops'] if o['phase'] == 'decode'
        }
        attn = next(
            c['share'] for c in result['class_share']
            if c['phase'] == 'decode' and c['op_class'] == 'attention_kernel')
        return shares, attn

    def test_parse_kernel_name(self):
        k = catalog_detectors.parse_rpa_decode_kernel(
            'RPAd-p_64-bq_1_1-bkv_10240_10240-sw_1024')
        assert k is not None
        self.assertEqual((k.block, k.chunk, k.window), (10240, 10240, 1024))
        k = catalog_detectors.parse_rpa_decode_kernel(
            'RPAd-p_64-bq_1_1-bkv_10240_2560')
        assert k is not None
        self.assertEqual((k.block, k.chunk, k.window), (10240, 2560, None))
        self.assertIsNone(
            catalog_detectors.parse_rpa_decode_kernel(
                'RPAm-p_64-bq_32_32-bkv_2048_512-sw_1024'))

    def test_stock_kernels_fire(self):
        shares, attn = self._decode_inputs('gemma3_tp8_stock_c64_case.json')
        share, hits = catalog_detectors.rpa_decode_block_oversized(
            shares, attn, max_model_len=10240, context_len=2048)
        self.assertEqual(
            [k.name for k, _, _ in hits],
            [
                'RPAd-p_64-bq_1_1-bkv_10240_10240-sw_1024',
                'RPAd-p_64-bq_1_1-bkv_10240_10240',
            ],
        )
        self.assertAlmostEqual(share, 0.681, places=2)
        self.assertIn('max_model_len', hits[0][2][0])

    def test_block_sizes_from_kernel_and_context(self):
        k = catalog_detectors.parse_rpa_decode_kernel
        # Window 700 on 128-token pages rounds up to 768.
        self.assertEqual(
            catalog_detectors.rpa_decode_block_sizes(
                [k('RPAd-p_128-bq_1_1-bkv_8192_8192-sw_700')], 3000),
            'sliding=768:768')
        # Global: smallest page-aligned divisor of 12288 covering 3000.
        self.assertEqual(
            catalog_detectors.rpa_decode_block_sizes(
                [k('RPAd-p_64-bq_1_1-bkv_12288_12288')], 3000),
            'global=12288:3072')
        # No context: a global layer is left alone.
        self.assertEqual(
            catalog_detectors.rpa_decode_block_sizes(
                [k('RPAd-p_64-bq_1_1-bkv_12288_12288')], None), '')
        # Chunk already covers the context: nothing to shrink.
        self.assertEqual(
            catalog_detectors.rpa_decode_block_sizes(
                [k('RPAd-p_64-bq_1_1-bkv_12288_2048')], 2048), '')
        # No page size in the name: no sizes.
        self.assertEqual(
            catalog_detectors.rpa_decode_block_sizes(
                [k('RPAd-bq_1_1-bkv_8192_8192-sw_1024')], 2048), '')

    def test_tuned_kernels_quiet(self):
        shares, attn = self._decode_inputs('gemma3_tp8_tuned_c64_case.json')
        self.assertGreater(attn, catalog_detectors.D45_MIN_ATTN_SHARE)
        _, hits = catalog_detectors.rpa_decode_block_oversized(
            shares, attn, max_model_len=10240, context_len=2048)
        self.assertEmpty(hits)


class GuardrailsTest(absltest.TestCase):

    def _d3(self, stall_pct: float, suggested: str) -> findings.Finding:
        return findings.Finding(
            detector_id='D3_prefill_decode_interference',
            title='t',
            phase='mixed',
            summary='s',
            evidence=[findings.Evidence('decode stall', stall_pct, '%')],
            knob_changes=[
                knobs.make_change(
                    'vllm_tpu.cli.max_num_batched_tokens',
                    suggested,
                    {'max-num-batched-tokens': '4096'},
                )
            ],
            risk=findings.Risk.LATENCY_TRADEOFF,
            confidence=findings.Confidence.MEDIUM,
        )

    def test_mnbt_increase_dropped_when_ttft_matters(self):
        ctx = context.ServingContext(max_num_batched_tokens=4096)
        out, notes = guardrails.apply([self._d3(5.0, '8192')], ctx)
        self.assertEmpty(out)
        self.assertLen(notes, 1)
        self.assertIn('TTFT', notes[0])

    def test_mnbt_increase_kept_with_large_stall_or_offline(self):
        ctx = context.ServingContext(max_num_batched_tokens=4096)
        out, _ = guardrails.apply([self._d3(30.0, '8192')], ctx)
        self.assertLen(out, 1)
        ctx.ttft_sensitive = False
        out, _ = guardrails.apply([self._d3(5.0, '8192')], ctx)
        self.assertLen(out, 1)

    def test_mnbt_decrease_kept(self):
        ctx = context.ServingContext(max_num_batched_tokens=4096)
        out, notes = guardrails.apply([self._d3(5.0, '2048')], ctx)
        self.assertLen(out, 1)
        self.assertEmpty(notes)


class FrameworkKnobsTest(absltest.TestCase):
    """TorchTPU / MaxText knob variants and command synthesis."""

    def _run(self, framework: str, cfg=None, pattern='DDDPDDPDDD', **kw):
        ctx = _ctx(serving_config=cfg if cfg is not None else {}, **kw)
        ctx.framework = framework
        return analyze.analyze(
            trace_table.TraceTable.from_events_frame(make_events(pattern)),
            ctx)

    def test_d4_torchtpu_rowmax_only_when_sort(self):
        r = self._run('vllm_torchtpu', {'TPU_MOE_ROUTER_TOPK': 'sort'})
        self.assertEqual(_knob_values(r, 'D4_moe_topk_sort'),
                         {'TPU_MOE_ROUTER_TOPK': 'rowmax'})
        d4 = [f for f in r.findings if f.detector_id == 'D4_moe_topk_sort']
        self.assertEqual(d4[0].risk, findings.Risk.PERF_ONLY)
        for cfg in ({}, {'TPU_MOE_ROUTER_TOPK': 'rowmax'}):
            r = self._run('vllm_torchtpu', cfg)
            self.assertIn('D4_moe_topk_sort', _ids(r))
            self.assertEqual(_knob_values(r, 'D4_moe_topk_sort'), {}, cfg)
        r_unknown = analyze.analyze(
            trace_table.TraceTable.from_events_frame(make_events()),
            dataclasses.replace(_ctx(serving_config=None),
                                framework='vllm_torchtpu'),
        )
        self.assertEqual(_knob_values(r_unknown, 'D4_moe_topk_sort'), {})

    def test_d2_torchtpu_skips_padded_tokens(self):
        r = self._run('vllm_torchtpu')
        got = _knob_values(r, 'D2_padding_waste')
        self.assertEqual(got, {'TPU_MOE_SKIP_PADDED_TOKENS': '1'})

    def test_d3_maxtext_sets_prefill_chunk_size(self):
        # Keep vLLM step segmentation for the synthetic trace; only the detector
        # sees MaxText.
        with mock.patch.object(detectors,
                               '_framework',
                               autospec=True,
                               return_value='maxtext'):
            r = self._run('auto')
        got = _knob_values(r, 'D3_prefill_decode_interference')
        self.assertEqual(got.get('use_chunked_prefill'), 'true')
        self.assertEqual(got.get('prefill_chunk_size'), '2048')

    def test_d44_torchtpu_collection_chunk_size(self):
        r = self._run('vllm_torchtpu', pattern='DDDPPPDDD')
        self.assertEqual(
            _knob_values(r, 'D44_moe_combine_not_chunked'),
            {'TPU_MOE_COLLECTION_CHUNK_SIZE': '1024'},
        )
        r = self._run(
            'vllm_torchtpu',
            {'TPU_MOE_COLLECTION_CHUNK_SIZE': '512'},
            pattern='DDDPPPDDD',
        )
        self.assertNotIn('D44_moe_combine_not_chunked', _ids(r))

    def test_recommended_command_patches_launch_command(self):
        cmd = 'MOE_APPROX_TOPK=false vllm serve m --max-num-seqs 32'
        r = self._run(
            'vllm_tpu',
            context.parse_serving_command(cmd),
            serving_command=cmd,
        )
        rec = r.recommended_command
        patched = context.parse_serving_command(rec.splitlines()[0])
        self.assertEqual(patched['MOE_APPROX_TOPK'], 'true')
        self.assertEqual(patched['enable_continue_decode'], 'true')
        self.assertEqual(r.summary_dict()['recommended_command'], rec)
        md = r.to_markdown()
        self.assertIn('## Ready-to-run configuration patch', md)
        self.assertIn(rec, md)

    def test_serving_command_alone_fills_serving_config(self):
        cmd = 'MOE_APPROX_TOPK=true vllm serve m --max-num-seqs 32'
        ctx = context.ServingContext(serving_command=cmd).finalize()
        self.assertEqual(ctx.config_get('MOE_APPROX_TOPK'), 'true')
        self.assertEqual(ctx.max_num_seqs, 32)


if __name__ == '__main__':
    absltest.main()
