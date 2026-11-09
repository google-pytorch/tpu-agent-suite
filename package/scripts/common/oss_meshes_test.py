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
"""Tests for oss_meshes, on the HF configs of real vLLM-TorchTPU baselines."""

import dataclasses
import json
import os
import unittest

import oss_meshes

_CONFIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testdata',
                        'oss_configs')
_HBM = 94.75
# A collective-bound prefill diagnosis, in the serving analyzer's output shape.
_D7_PREFILL = {
    'profiles': [{
        'role': 'prefill',
        'result': {
            'findings': [{
                'detector_id': 'D7_collective_bound',
                'phase': 'prefill_or_mixed',
            }]
        },
    }]
}


def _config(name: str) -> dict[str, object]:
    with open(os.path.join(_CONFIGS, name)) as f:
        return json.load(f)


def _by_label(result):
    return {m['label']: m for m in result['meshes']}


class OssMeshesTest(unittest.TestCase):

    def test_model_shape_reads_text_config_and_layer_types(self):
        m = oss_meshes.ModelShape.from_hf_config(_config('Qwen3.8-27B.json'))
        self.assertEqual((m.num_heads, m.num_kv_heads), (24, 4))
        self.assertEqual(m.linear_num_value_heads, 48)
        self.assertEqual(m.layer_types.count('full_attention'), 16)
        g4 = oss_meshes.ModelShape.from_hf_config(
            _config('gemma-4-31B-it.json'))
        self.assertEqual(g4.num_global_kv_heads, 4)

    def test_gemma4_global_heads_from_per_layer_config(self):
        # transformers 5.17's to_dict() drops num_global_key_value_heads and
        # global_head_dim; the full-attention layers carry them instead.
        cfg = _config('gemma-4-31B-it.json')
        text = dict(cfg['text_config'])
        heads = text.pop('num_global_key_value_heads')
        dim = text.pop('global_head_dim')
        text['per_layer_config'] = {
            f'{i:02d}': {
                'head_dim': dim,
                'num_key_value_heads': heads
            }
            for i, lt in enumerate(text['layer_types'])
            if lt == 'full_attention'
        }
        g4 = oss_meshes.ModelShape.from_hf_config(dict(cfg, text_config=text))
        self.assertEqual((g4.num_global_kv_heads, g4.global_head_dim),
                         (heads, dim))

    def test_gemma3_hbm_matches_server_log(self):
        # Server log (TP8): 6.86 GiB weights/device, 1,273,536 KV tokens.
        m = oss_meshes.ModelShape.from_hf_config(
            _config('gemma-3-27b-it.json'))
        est = oss_meshes.hbm_estimate(m,
                                      oss_meshes.Mesh(8, 1),
                                      _HBM,
                                      weights_gib=6.86 * 8)
        self.assertAlmostEqual(est.weights_gib_per_device, 6.86, places=2)
        self.assertEqual(est.kv_bytes_per_token_per_device,
                         62 * 2 * 2 * 128 * 2)
        self.assertAlmostEqual(est.kv_tokens_per_replica,
                               1_273_536,
                               delta=30_000)

    def test_kv_need_counts_sliding_windows(self):
        m = oss_meshes.ModelShape.from_hf_config(
            _config('gemma-4-31B-it.json'))
        self.assertEqual(m.sliding_window, 1024)
        mesh = oss_meshes.Mesh(8, 1)
        est = oss_meshes.hbm_estimate(m,
                                      mesh,
                                      _HBM,
                                      weights_gib=58.3,
                                      max_len=9216,
                                      concurrency=128)
        self.assertTrue(est.kv_ok)
        full = dataclasses.replace(m, sliding_window=None)
        self.assertGreater(oss_meshes.kv_bytes_per_sequence(full, mesh, 9216),
                           2 * oss_meshes.kv_bytes_per_sequence(m, mesh, 9216))

    def test_qwen38_tp8_is_image_scoped(self):
        r = oss_meshes.enumerate_oss_meshes(
            _config('Qwen3.8-27B.json'),
            devices=8,
            hbm_gib=_HBM,
            baseline={
                'tp': 4,
                'dp': 2
            },
            image='sha256:0123abcd',
        )
        rec = _by_label(r)['TP8xDP1']
        self.assertTrue(rec['feasible'])
        self.assertIn('qwen38_tp8_engine_init',
                      [x['id'] for x in rec['warnings']])

    def test_qwen38_only_pp_alternative(self):
        r = oss_meshes.enumerate_oss_meshes(
            _config('Qwen3.8-27B.json'),
            devices=8,
            hbm_gib=_HBM,
            baseline={
                'tp': 4,
                'dp': 2
            },
            serving=_D7_PREFILL,
            image='sha256:107753fc',
            input_len=1024,
            output_len=1024,
            concurrency=1024,
        )
        meshes = _by_label(r)
        self.assertEqual(meshes['TP2xDP4']['rejected_by'][0]['id'],
                         'gdn_v3_triangular_block')
        self.assertEqual(meshes['TP8xDP1']['rejected_by'][0]['id'],
                         'qwen38_tp8_engine_init')
        self.assertFalse(meshes['TP1xDP8']['feasible'])
        self.assertTrue(meshes['TP4xDP2']['feasible'])
        # DCP for GQA needs TP > 4 KV heads; PCP is MLA-only in vLLM.
        self.assertEqual(meshes['TP4xDP2+DCP2']['rejected_by'][0]['id'],
                         'head_divisibility')
        self.assertIn(
            'vllm_pcp_limits',
            [x['id'] for x in meshes['TP4xPCP2xDP1']['rejected_by']],
        )
        feasible = sorted(label for label, m in meshes.items()
                          if m["feasible"])
        self.assertEqual(feasible, ['TP4xDP2'])
        # Every alternative is infeasible once torchtpu_pp_with_tp_unsupported is
        # enforced; when ignored (or on non-torchtpu), TP4xPP2xDP1 is proposed.
        self.assertIn(
            'torchtpu_pp_with_tp_unsupported',
            [x['id'] for x in meshes['TP4xPP2xDP1']['rejected_by']],
        )
        self.assertFalse(r['candidates'])
        r_ignore = oss_meshes.enumerate_oss_meshes(
            _config('Qwen3.8-27B.json'),
            devices=8,
            hbm_gib=_HBM,
            baseline={
                'tp': 4,
                'dp': 2
            },
            serving=_D7_PREFILL,
            input_len=1024,
            output_len=1024,
            concurrency=1024,
            image='sha256:107753fc',
            ignore_constraints=('torchtpu_pp_with_tp_unsupported', ),
        )
        self.assertEqual(len(r_ignore['candidates']), 1)
        c = r_ignore['candidates'][0]
        self.assertEqual(c['label'], 'TP4xPP2xDP1')
        self.assertEqual(
            c['argv_set'],
            {
                '--tensor-parallel-size': '4',
                '--data-parallel-size': '1',
                '--pipeline-parallel-size': '2',
            },
        )
        self.assertEqual(c['unverified_on_torch_tpu'], ['pp'])
        self.assertIn('unverified on torch_tpu', ' '.join(c['risks']))
        self.assertIn('exploratory', c['rationale'])

    def test_qwen38_pp_not_proposed_without_diagnosis(self):
        r = oss_meshes.enumerate_oss_meshes(
            _config('Qwen3.8-27B.json'),
            devices=8,
            hbm_gib=_HBM,
            baseline={
                'tp': 4,
                'dp': 2
            },
            input_len=1024,
            output_len=1024,
            concurrency=1024,
        )
        self.assertFalse(r['candidates'])

    def test_gemma4_prefill_bound_proposes_tp2pp2_then_tp2dp4(self):
        r = oss_meshes.enumerate_oss_meshes(
            _config('gemma-4-31B-it.json'),
            devices=8,
            hbm_gib=_HBM,
            baseline={
                'tp': 4,
                'dp': 2
            },
            serving=_D7_PREFILL,
            input_len=8192,
            output_len=1024,
            concurrency=128,
        )
        self.assertEqual(r['regime'], 'prefill_heavy')
        labels = [c['label'] for c in r['candidates']]
        # On vllm_torchtpu, PP>1 with TP>1 is rejected by
        # torchtpu_pp_with_tp_unsupported, leaving TP2xDP4 as the sole candidate.
        self.assertEqual(labels, ['TP2xDP4'])
        only = r['candidates'][0]
        self.assertEqual(
            only['vllm_args'],
            {
                '--tensor-parallel-size': 2,
                '--data-parallel-size': 4
            },
        )
        self.assertIn('decode-regressing', only['risks'][0])
        self.assertFalse(only['unverified_on_torch_tpu'])
        self.assertNotIn('TP8xDP1', labels)
        # When torchtpu_pp_with_tp_unsupported is ignored, TP2xPP2xDP2 ranks first.
        r_pp = oss_meshes.enumerate_oss_meshes(
            _config('gemma-4-31B-it.json'),
            devices=8,
            hbm_gib=_HBM,
            baseline={
                'tp': 4,
                'dp': 2
            },
            serving=_D7_PREFILL,
            input_len=8192,
            output_len=1024,
            concurrency=128,
            ignore_constraints=('torchtpu_pp_with_tp_unsupported', ),
        )
        first, second = r_pp['candidates'][:2]
        self.assertEqual(
            first['vllm_args'],
            {
                '--tensor-parallel-size': 2,
                '--data-parallel-size': 2,
                '--pipeline-parallel-size': 2,
            },
        )
        self.assertNotIn('decode-regressing', ' '.join(first['risks']))
        self.assertIn('PP=2 unverified on torch_tpu', ' '.join(first['risks']))
        self.assertEqual(
            second['vllm_args'],
            {
                '--tensor-parallel-size': 2,
                '--data-parallel-size': 4
            },
        )

    def test_gemma4_without_diagnosis_keeps_mesh(self):
        r = oss_meshes.enumerate_oss_meshes(
            _config('gemma-4-31B-it.json'),
            devices=8,
            hbm_gib=_HBM,
            baseline={
                'tp': 4,
                'dp': 2
            },
            input_len=1024,
            output_len=1024,
        )
        self.assertFalse(r['candidates'])

    def test_pp_bubbles_at_low_concurrency(self):
        r = oss_meshes.enumerate_oss_meshes(
            _config('gemma-4-31B-it.json'),
            devices=8,
            hbm_gib=_HBM,
            baseline={
                'tp': 4,
                'dp': 2
            },
            serving=_D7_PREFILL,
            input_len=8192,
            output_len=1024,
            concurrency=8,
        )
        # 2 sequences per stage: PP bubbles cancel the D7 gain.
        self.assertEqual([c['label'] for c in r['candidates']], ['TP2xDP4'])

    def test_all_meshes_vllm_semantics(self):
        m = oss_meshes.ModelShape(num_heads=32, num_kv_heads=8)
        meshes = oss_meshes.all_meshes(m, 8)
        self.assertTrue(all(x.devices == 8 for x in meshes))
        self.assertTrue(all(x.tp % x.dcp == 0 for x in meshes))
        labels = {x.label for x in meshes}
        self.assertLessEqual(
            {'TP4xPP2xDP1', 'TP8xDP1+DCP2', 'TP4xPCP2xDP1', 'TP2xPP2xDP2'},
            labels,
        )
        self.assertNotIn('TP4xDP1+DCP2', labels)  # DCP adds no devices.

    def test_dcp_gqa_rules(self):
        for name, kv, tp, dcp, error in (
            ('tp_not_above_kv', 8, 4, 2, 'needs tp=4 > num_key_value_heads=8'),
            ('ok', 4, 8, 2, None),
            ('dcp_too_big', 4, 8, 4, 'dcp=4 > tp // num_key_value_heads=2'),
        ):
            with self.subTest(name):
                m = oss_meshes.ModelShape(num_heads=32, num_kv_heads=kv)
                errors = oss_meshes.head_divisibility(
                    m, oss_meshes.Mesh(tp, 1, dcp=dcp))
                if error is None:
                    self.assertFalse(errors)
                else:
                    self.assertTrue(any(error in e for e in errors), errors)

    def test_dcp_proposed_for_long_context_decode(self):
        cfg = {
            'model_type': 'llama',
            'hidden_size': 4096,
            'intermediate_size': 14336,
            'num_hidden_layers': 32,
            'num_attention_heads': 32,
            'num_key_value_heads': 4,
            'vocab_size': 128256,
        }
        r = oss_meshes.enumerate_oss_meshes(
            cfg,
            devices=8,
            hbm_gib=_HBM,
            baseline={
                'tp': 8,
                'dp': 1
            },
            input_len=4096,
            output_len=65536,
            concurrency=64,
        )
        first = r['candidates'][0]
        self.assertEqual(first['label'], 'TP8xDP1+DCP2')
        self.assertEqual(first['argv_set']['--decode-context-parallel-size'],
                         '2')
        self.assertEqual(first['unverified_on_torch_tpu'], ['dcp'])
        rec = _by_label(r)['TP8xDP1+DCP2']
        base = _by_label(r)['TP8xDP1']
        self.assertAlmostEqual(
            rec['hbm']['kv_bytes_per_token_per_device'] * 2,
            base['hbm']['kv_bytes_per_token_per_device'],
        )

    def test_pcp_allowed_for_mla_text_model(self):
        m = oss_meshes.ModelShape.from_hf_config({
            'model_type': 'deepseek_v3',
            'num_attention_heads': 128,
            'num_key_value_heads': 128,
            'kv_lora_rank': 512,
        })
        self.assertTrue(m.use_mla)
        rec = oss_meshes.evaluate_mesh(
            m,
            oss_meshes.Mesh(4, 1, pcp=2),
            hbm_gib=_HBM,
            framework='vllm_torchtpu',
            image=None,
        )
        self.assertNotIn('vllm_pcp_limits',
                         [x['id'] for x in rec['rejected_by']])
        self.assertIn('PCP=2 unverified on torch_tpu', rec['risks'][0])

    def test_baseline_pp_emits_explicit_pp1(self):
        mesh = oss_meshes.Mesh(4, 2)
        self.assertEqual(
            mesh.argv_set(oss_meshes.Mesh(4, 1, pp=2)),
            {
                '--tensor-parallel-size': '4',
                '--data-parallel-size': '2',
                '--pipeline-parallel-size': '1',
            },
        )

    def test_gptoss_ep_row_partitions_is_image_scoped(self):
        for image, rejected in (
            ('sha256:ec855bc5', True),
            ('sha256:581bcde5', False),
            (None, False),
        ):
            with self.subTest(image=image):
                r = oss_meshes.enumerate_oss_meshes(
                    _config('gpt-oss-20b.json'),
                    devices=8,
                    hbm_gib=_HBM,
                    baseline={
                        'tp': 2,
                        'dp': 4
                    },
                    image=image,
                )
                rec = _by_label(r)['TP8xDP1+EP8']
                where = 'rejected_by' if rejected else 'warnings'
                self.assertIn('torchtpu_ep_row_partitions',
                              [x['id'] for x in rec[where]])

    def test_gptoss_kv_head_padding_proposes_tp4(self):
        # D8 on gpt-oss TP 8: 8 KV heads give 1 per device, below the bf16
        # packing of 2. Without D7 the mesh step used to argue for nothing.
        d8 = {
            'findings': [{
                'detector_id':
                'D8_kv_head_padding',
                'phase':
                'decode',
                'evidence': [{
                    'name': 'KV packing factor',
                    'value': 2
                }],
            }]
        }
        kwargs = dict(devices=8,
                      hbm_gib=_HBM,
                      baseline={
                          'tp': 8,
                          'dp': 1
                      },
                      image='sha256:7cb2b493')
        without = oss_meshes.enumerate_oss_meshes(_config('gpt-oss-20b.json'),
                                                  **kwargs)
        self.assertNotIn('TP4xDP2',
                         [c['label'] for c in without['candidates']])
        self.assertIn('TP4xDP2',
                      [x['label'] for x in without['not_argued_for']])
        r = oss_meshes.enumerate_oss_meshes(_config('gpt-oss-20b.json'),
                                            serving=d8,
                                            **kwargs)
        self.assertEqual(r['kv_head_padding'], 2)
        labels = [c['label'] for c in r['candidates']]
        self.assertIn('TP4xDP2', labels)
        self.assertNotIn('TP2xDP4', labels)  # one halving reaches 2 heads
        tp4 = next(c for c in r['candidates'] if c['label'] == 'TP4xDP2')
        self.assertIn('D8:', tp4['rationale'])

    def test_measured_weights_kept_without_shape_estimate(self):
        # No hidden_size (a d_model / n_embd config): the shape estimate is
        # 0, so the measured weights are split over TP instead of dropped.
        m = oss_meshes.ModelShape(num_layers=10,
                                  num_heads=8,
                                  num_kv_heads=8,
                                  head_dim=128)
        est = oss_meshes.hbm_estimate(m,
                                      oss_meshes.Mesh(tp=4, dp=1),
                                      _HBM,
                                      weights_gib=40.0)
        self.assertAlmostEqual(est.weights_gib_per_device, 10.0)

    def test_tp1_is_scoped_to_tpu7x(self):
        m = oss_meshes.ModelShape(num_heads=8, num_kv_heads=8, head_dim=128)

        def rejected(accelerator):
            rec = oss_meshes.evaluate_mesh(m,
                                           oss_meshes.Mesh(tp=1, dp=1),
                                           hbm_gib=_HBM,
                                           framework='vllm_torchtpu',
                                           image='sha256:0123abcd',
                                           accelerator=accelerator)
            return [r['id'] for r in rec['rejected_by']]

        self.assertIn('torchtpu_tp1_no_layout', rejected('tpu7x'))
        self.assertIn('torchtpu_tp1_no_layout', rejected(None))  # unknown
        self.assertNotIn('torchtpu_tp1_no_layout', rejected('tpu-v6e-slice'))
        self.assertEqual(oss_meshes.generation('v5litepod'), 'v5e')

    def test_d8_packing_read_from_findings_json(self):
        # The analyzer's findings.json labels evidence `metric`, not `name`.
        d8 = {
            'findings': [{
                'detector_id':
                'D8_kv_head_padding',
                'evidence': [{
                    'metric': 'KV packing factor',
                    'value': 4,
                    'unit': '',
                    'detail': 'fp8'
                }],
            }]
        }
        self.assertEqual(oss_meshes.kv_head_padding(d8), 4)

    def test_d8_packing_unknown_without_evidence(self):
        d8 = {'findings': [{'detector_id': 'D8_kv_head_padding'}]}
        self.assertIsNone(oss_meshes.kv_head_padding(d8))

    def test_kv_waste_counts_global_heads(self):
        # Local heads fill the bf16 packing at TP 8 (2 per device), but 4
        # global heads leave half a head per device: replicated rows.
        ctx = oss_meshes._Context(  # pylint: disable=protected-access
            **{
                f.name: None
                for f in dataclasses.fields(oss_meshes._Context)  # pylint: disable=protected-access
                if f.default is dataclasses.MISSING
            },
            kv_packing=2,
            kv_heads=16,
            head_dim=256,
            global_kv_heads=4,
            global_head_dim=512)
        self.assertGreater(oss_meshes._kv_waste(ctx, 8), 0)  # pylint: disable=protected-access
        local_only = dataclasses.replace(ctx, global_kv_heads=0)
        self.assertEqual(oss_meshes._kv_waste(local_only, 8), 0)  # pylint: disable=protected-access

    def test_expert_parallel_costs_one_change(self):
        # EP = TP x PCP x DP follows from the mesh: turning it on is one flag.
        r = oss_meshes.enumerate_oss_meshes(_config('gpt-oss-20b.json'),
                                            devices=8,
                                            hbm_gib=_HBM,
                                            baseline={
                                                'tp': 8,
                                                'dp': 1
                                            },
                                            image='sha256:7cb2b493')
        self.assertIn('TP8xDP1+EP8', [c['label'] for c in r['candidates']])

    def test_dp_hang_is_image_scoped(self):
        for image, feasible in (
            ('sha256:581bcde5aa', False),
            ('us-docker.pkg.dev/x/vllm@sha256:ec855bc5ff', False),
            ('sha256:107753fc', True),
            (None, True),
        ):
            with self.subTest(image=image):
                r = oss_meshes.enumerate_oss_meshes(
                    _config('gemma-3-27b-it.json'),
                    devices=8,
                    hbm_gib=_HBM,
                    baseline={
                        'tp': 8,
                        'dp': 1
                    },
                    image=image,
                )
                rec = _by_label(r)['TP4xDP2']
                self.assertEqual(rec['feasible'], feasible)
                ids = [x['id'] for x in rec['rejected_by'] + rec['warnings']]
                self.assertIn('dp_engine_hang', ids)

    def test_head_divisibility(self):
        m = oss_meshes.ModelShape(num_heads=24, num_kv_heads=4)
        self.assertEqual(
            oss_meshes.head_divisibility(m, oss_meshes.Mesh(16, 1)),
            ['num_attention_heads=24 % tp=16 != 0'],
        )
        self.assertFalse(oss_meshes.head_divisibility(m, oss_meshes.Mesh(8,
                                                                         1)))

    def test_constraint_table_records_image(self):
        for c in oss_meshes.KERNEL_CONSTRAINTS:
            self.assertTrue(c.failure_signature, c.id)
            self.assertTrue(c.source, c.id)
            if not c.structural:
                self.assertTrue(c.observed_on, c.id)

    def test_gemma4_per_point_regime_split(self):
        # Measured 1016a/1016b (TP4xDP2 -> TP2xDP4): +27.8% at 8k1k c128,
        # -24.9% at 1k1k c32.
        r = oss_meshes.enumerate_oss_meshes(
            _config('gemma-4-31B-it.json'),
            devices=8,
            hbm_gib=_HBM,
            baseline={
                'tp': 4,
                'dp': 2
            },
            serving=_D7_PREFILL,
            points=[
                {
                    'isl': 8192,
                    'osl': 1024,
                    'c': 128
                },
                {
                    'isl': 1024,
                    'osl': 1024,
                    'c': 32
                },
            ],
        )
        self.assertEqual([p['label'] for p in r['points']],
                         ['8k1k c128', '1k1k c32'])
        # The first point is the main point when no --input_len is given.
        self.assertEqual(r['regime'], 'prefill_heavy')
        by_label = {c['label']: c for c in r['candidates']}
        c = by_label['TP2xDP4']
        pp = c['per_point_assessment']
        self.assertEqual(pp['8k1k c128']['regime'], 'prefill_heavy')
        self.assertEqual(pp['8k1k c128']['expected_direction'], 'positive')
        self.assertIn('D7', pp['8k1k c128']['rationale'])
        self.assertEqual(pp['1k1k c32']['regime'], 'balanced')
        self.assertEqual(pp['1k1k c32']['expected_direction'], 'negative')
        self.assertIn('D7 not credited', pp['1k1k c32']['rationale'])
        self.assertIn('HBM-bound decode', pp['1k1k c32']['rationale'])
        self.assertTrue(c['workload_dependent'])
        self.assertIn('reduction order changes', c['determinism_risk'])
        self.assertIn('greedy outputs may differ', ' '.join(c['risks']))

    def test_points_only_low_concurrency_balanced_keeps_mesh(self):
        r = oss_meshes.enumerate_oss_meshes(
            _config('gemma-4-31B-it.json'),
            devices=8,
            hbm_gib=_HBM,
            baseline={
                'tp': 4,
                'dp': 2
            },
            serving=_D7_PREFILL,
            input_len=1024,
            output_len=1024,
            concurrency=32,
            points=[{
                'input_len': 1024,
                'output_len': 1024,
                'concurrency': 32
            }],
        )
        # Single-point scoring still credits D7 on balanced traffic; the per-point
        # assessment is what shows the decode regression.
        by_label = {c['label']: c for c in r['candidates']}
        a = by_label['TP2xDP4']['per_point_assessment']['1k1k c32']
        self.assertEqual(a['expected_direction'], 'negative')
        self.assertFalse(by_label['TP2xDP4']['workload_dependent'])

    def test_single_point_output_unchanged_without_points(self):
        r = oss_meshes.enumerate_oss_meshes(
            _config('gemma-4-31B-it.json'),
            devices=8,
            hbm_gib=_HBM,
            baseline={
                'tp': 4,
                'dp': 2
            },
            serving=_D7_PREFILL,
            input_len=8192,
            output_len=1024,
            concurrency=128,
        )
        c = r['candidates'][0]
        self.assertNotIn('per_point_assessment', c)
        self.assertNotIn('workload_dependent', c)
        self.assertEqual(r['points'], [])
        self.assertEqual(c['score'], 0.75)

    def test_point_parsing(self):
        p = oss_meshes.Point.from_dict({'isl': 1000, 'osl': 2048, 'c': 4})
        self.assertEqual(p.label, '1000/2k c4')
        p = oss_meshes.Point.from_dict({
            'isl': 1024,
            'osl': 1024,
            'c': 4,
            'label': 'chat'
        })
        self.assertEqual(p.label, 'chat')
        with self.assertRaisesRegex(ValueError, 'lacks'):
            oss_meshes.Point.from_dict({'isl': 1024, 'c': 4})
        with self.assertRaisesRegex(ValueError, 'positive'):
            oss_meshes.Point.from_dict({'isl': 0, 'osl': 1, 'c': 1})

    def test_determinism_risk_only_for_tp_dp_changes(self):
        base = oss_meshes.Mesh(4, 2)
        self.assertIsNone(
            oss_meshes.determinism_risk(oss_meshes.Mesh(4, 2), base))
        self.assertIsNone(
            oss_meshes.determinism_risk(oss_meshes.Mesh(4, 2, ep=8), base))
        self.assertIn(
            'TP4xDP2 -> TP2xDP4',
            oss_meshes.determinism_risk(oss_meshes.Mesh(2, 4), base),
        )

    def test_a_truncated_digest_matches_nothing(self):
        digests = ('sha256:581bcde5', )
        self.assertTrue(
            oss_meshes._image_matches('repo/img@sha256:581bcde5ff', digests))
        self.assertFalse(oss_meshes._image_matches('sha256:5', digests))
        self.assertFalse(oss_meshes._image_matches('sha256:581b', digests))


if __name__ == '__main__':
    unittest.main()
