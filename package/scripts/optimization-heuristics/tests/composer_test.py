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
"""Tests for the independent-first composer (compose_plan)."""

import json
import os
import sys

from absl.testing import absltest

# The serving_analysis package sits one directory up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving_analysis import composer

_TESTDATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testdata')
# Keys of the overnight harness candidates.json (candidate_to_spec.py).
_HARNESS_KEYS = {
    'cand_id',
    'rank',
    'argv_set',
    'argv_add_flags',
    'argv_remove_flags',
    'env_set',
    'check_env_knobs',
    'check_cli_flags',
    'rationale',
    'evidence',
    'expected_mechanism',
    'target_points',
    'risks',
    'provenance',
}
# `enumerate_oss_meshes` output for Gemma-4 TP4xDP2 8k1k (abridged).
_GEMMA4_OSS_MESHES = {
    'prefill_collective_bound':
    True,
    'candidates': [
        {
            'rank': 1,
            'mesh': {
                'tp': 2,
                'dp': 4,
                'ep': 1
            },
            'label': 'TP2xDP4',
            'target_regime': 'prefill_heavy',
            'score': 0.75,
            'rationale': 'D7: prefill is collective-bound',
            'risks': ['decode-regressing: per-device weight bytes x2'],
        },
        {
            'rank': 2,
            'mesh': {
                'tp': 1,
                'dp': 8,
                'ep': 1
            },
            'label': 'TP1xDP8',
            'target_regime': 'prefill_heavy',
            'score': 0.5,
            'rationale': 'D7: prefill is collective-bound',
            'risks': ['decode-regressing: per-device weight bytes x4'],
        },
    ],
}


def _load(name: str) -> dict[str, object]:
    with open(os.path.join(_TESTDATA, name)) as f:
        return json.load(f)


def _finding(det, name, value, kind='env'):
    return {
        'detector_id':
        det,
        'title':
        det,
        'phase':
        'decode',
        'summary':
        's',
        'evidence': [],
        'knob_changes': [{
            'knob': f'k.{name}',
            'kind': kind,
            'name': name,
            'suggested_value': value,
            'current_value': '0',
            'already_set': False,
        }],
        'risk':
        'perf_only',
        'confidence':
        'high',
        'score':
        0.2,
    }


class ComposerTest(absltest.TestCase):

    def test_gemma3_proposes_rpa_tuning_only(self):
        plan = composer.compose_plan(
            _load('gemma3_tp8_stock_c64_golden.json'),
            baseline_mesh={
                'tp': 8,
                'dp': 1
            },
        )
        # The golden predates D45's knob: its finding carries no change.
        self.assertFalse([
            c for c in plan.candidates
            if 'D45_rpa_decode_block_oversized' in c.findings
        ])
        self.assertNotIn('max-num-batched-tokens', plan.markdown)

    def test_gemma4_mesh_candidates_from_enumerator(self):
        plan = composer.compose_plan(
            _load('gemma4_tp4dp2_8k1k_c128_golden.json'),
            oss_meshes=_GEMMA4_OSS_MESHES,
            baseline_mesh={
                'tp': 4,
                'dp': 2
            },
            target_points=['8k1k_c128', '1k1k_c128'],
        )
        self.assertEqual([c.cand_id for c in plan.candidates],
                         ['m-tp2xdp4', 'm-tp1xdp8'])
        c = plan.candidates[0]
        self.assertEqual(
            c.argv_set,
            {
                '--tensor-parallel-size': '2',
                '--data-parallel-size': '4'
            },
        )
        self.assertEqual(c.remeasure, ['D7_collective_bound'])
        self.assertEqual(c.findings, ['D7_collective_bound'])
        self.assertEqual(c.target_regime, 'prefill_heavy')
        self.assertIn('decode-regressing', c.risks[0])
        entry = plan.candidates_json()[0]
        self.assertEqual(entry['target_points'], ['8k1k_c128', '1k1k_c128'])
        self.assertEqual(entry['provenance']['tool'],
                         'oss_meshes.enumerate_oss_meshes')
        # TP/DP changes get a determinism risk; no points -> no per-point rows.
        self.assertIn('reduction order changes (TP, DP changed)', c.risks[-1])
        self.assertFalse(entry['workload_dependent'])
        self.assertNotIn('## Per-point assessment', plan.markdown)

    def test_per_point_rows_and_workload_dependent_label(self):
        oss = {
            'prefill_collective_bound':
            True,
            'points': [
                {
                    'isl': 8192,
                    'osl': 1024,
                    'c': 128,
                    'label': '8k1k c128'
                },
                {
                    'isl': 1024,
                    'osl': 1024,
                    'c': 32,
                    'label': '1k1k c32'
                },
            ],
            'candidates': [
                {
                    'rank':
                    1,
                    'mesh': {
                        'tp': 2,
                        'dp': 4,
                        'ep': 1
                    },
                    'label':
                    'TP2xDP4',
                    'score':
                    0.75,
                    'risks': [
                        'reduction order changes (TP4xDP2 -> TP2xDP4): greedy'
                        ' outputs may differ'
                    ],
                    'per_point_assessment': {
                        '8k1k c128': {
                            'regime': 'prefill_heavy',
                            'score': 0.75,
                            'expected_direction': 'positive',
                            'rationale': 'D7: prefill is collective-bound',
                        },
                        '1k1k c32': {
                            'regime': 'balanced',
                            'score': -1.75,
                            'expected_direction': 'negative',
                            'rationale': 'D7 not credited',
                        },
                    },
                },
                {
                    'rank': 2,
                    'mesh': {
                        'tp': 4,
                        'dp': 2,
                        'ep': 8
                    },
                    'label': 'TP4xDP2+EP8',
                    'score': 0.25,
                    'per_point_assessment': {
                        '8k1k c128': {
                            'regime': 'prefill_heavy',
                            'score': 0.25,
                            'expected_direction': 'positive',
                            'rationale': 'ep',
                        },
                        '1k1k c32': {
                            'regime': 'balanced',
                            'score': 0.25,
                            'expected_direction': 'positive',
                            'rationale': 'ep',
                        },
                    },
                },
            ],
        }
        plan = composer.compose_plan(None,
                                     oss_meshes=oss,
                                     baseline_mesh={
                                         'tp': 4,
                                         'dp': 2
                                     })
        tp2, ep = plan.candidates
        self.assertTrue(tp2.workload_dependent)
        self.assertFalse(ep.workload_dependent)
        md = plan.markdown
        self.assertIn('## Per-point assessment', md)
        self.assertIn(
            '| 1 | `m-tp2xdp4` (workload-dependent) | 8k1k c128 | prefill_heavy |'
            ' positive | 0.75 |',
            md,
        )
        self.assertIn('| 1k1k c32 | balanced | negative | -1.75 |', md)
        self.assertIn('| 2 | `m-tp4xdp2-ep8` | 1k1k c32 |', md)
        self.assertIn('(`m-tp2xdp4`) [workload-dependent]', md)
        # The enumerator's determinism risk is not duplicated; EP-only changes
        # keep TP and DP, so no determinism risk.
        self.assertLen(
            [r for r in tp2.risks if 'reduction order changes' in r], 1)
        self.assertNotIn('reduction order', ' '.join(ep.risks))
        self.assertIn('workload-dependent: expected positive at 8k1k c128', md)
        entry = plan.candidates_json()[0]
        self.assertEqual(entry['target_points'], ['8k1k c128', '1k1k c32'])
        self.assertEqual(
            entry['per_point_assessment']['1k1k c32']['expected_direction'],
            'negative',
        )

        oss = {
            'prefill_collective_bound':
            True,
            'candidates': [
                {
                    'mesh': {
                        'tp': 2,
                        'dp': 2,
                        'ep': 1,
                        'pp': 2,
                        'dcp': 1,
                        'pcp': 1,
                    },
                    'label': 'TP2xPP2xDP2',
                    'argv_set': {
                        '--tensor-parallel-size': '2',
                        '--data-parallel-size': '2',
                        '--pipeline-parallel-size': '2',
                    },
                    'target_regime': 'prefill_heavy',
                    'score': 1.0,
                    'rationale': 'D7',
                    'risks': ['PP=2 unverified on torch_tpu'],
                },
                {
                    'mesh': {
                        'tp': 4,
                        'dp': 2,
                        'ep': 1,
                        'pp': 1,
                        'dcp': 2,
                        'pcp': 1,
                    },
                    'label': 'TP4xDP2+DCP2',
                    'score': 0.5,
                    'rationale': 'long context',
                    'risks': ['DCP=2 unverified on torch_tpu'],
                },
            ],
        }
        # TP2xDP4 in the plan is not a duplicate of TP2xPP2xDP2.
        md = '### Candidate 1: TP2\n- **ICI Mesh**: {"tp": 2, "dp": 4}\n'
        plan = composer.compose_plan(
            None,
            sharding_markdown=md,
            oss_meshes=oss,
            baseline_mesh={
                'tp': 4,
                'dp': 2
            },
        )
        by_id = {c.cand_id: c for c in plan.candidates}
        self.assertCountEqual(by_id,
                              ['m-tp2', 'm-tp2xpp2xdp2', 'm-tp4xdp2-dcp2'])
        self.assertEqual(
            by_id['m-tp2xpp2xdp2'].argv_set['--pipeline-parallel-size'], '2')
        # DCP alone changes the mesh (not "same as the baseline").
        dcp = by_id['m-tp4xdp2-dcp2']
        self.assertEqual(
            dcp.argv_set,
            {
                '--tensor-parallel-size': '4',
                '--data-parallel-size': '2',
                '--decode-context-parallel-size': '2',
            },
        )
        self.assertNotIn('--pipeline-parallel-size', by_id['m-tp2'].argv_set)
        self.assertIn(
            '--decode-context-parallel-size',
            plan.candidates_json()[2]['check_cli_flags'],
        )
        # A baseline with PP gets an explicit PP1 on a non-PP candidate.
        plan = composer.compose_plan(None,
                                     sharding_markdown=md,
                                     baseline_mesh={
                                         'tp': 4,
                                         'dp': 1,
                                         'pp': 2
                                     })
        self.assertEqual(
            plan.candidates[0].argv_set['--pipeline-parallel-size'], '1')

    def test_duplicate_keeps_enumerator_risks(self):
        md = ('### Candidate 1: TP2 PP2\n'
              '- **ICI Mesh**: {"tp": 2, "pp": 2, "dp": 2}\n')
        oss = {
            'prefill_collective_bound':
            True,
            'candidates': [{
                'mesh': {
                    'tp': 2,
                    'dp': 2,
                    'ep': 1,
                    'pp': 2,
                    'dcp': 1,
                    'pcp': 1
                },
                'label': 'TP2xPP2xDP2',
                'target_regime': 'prefill_heavy',
                'score': 1.0,
                'rationale': 'D7',
                'risks': ['PP=2 unverified on torch_tpu'],
            }],
        }
        plan = composer.compose_plan(
            None,
            sharding_markdown=md,
            oss_meshes=oss,
            baseline_mesh={
                'tp': 4,
                'dp': 2
            },
        )
        self.assertLen(plan.candidates, 1)
        c = plan.candidates[0]
        self.assertEqual(c.cand_id, 'm-tp2-pp2')
        self.assertEqual(c.risks[0], 'PP=2 unverified on torch_tpu')
        # TP4 -> TP2 also changes the reduction order.
        self.assertLen(c.risks, 2)
        self.assertIn('reduction order changes (TP changed)', c.risks[1])
        self.assertEqual(c.target_regime, 'prefill_heavy')
        self.assertIn('D7_collective_bound', c.findings)
        self.assertIn('duplicate', ' '.join(plan.rejected))
        self.assertIn('PP=2 unverified on torch_tpu',
                      plan.candidates_json()[0]['risks'])

    def test_gptoss_serving_mesh_knob_rejected(self):
        plan = composer.compose_plan(_load('gptoss_1k1k_c512_case.json'))
        ids = [c.cand_id for c in plan.candidates]
        self.assertEqual(ids, ['s-async-scheduling-off'])
        self.assertEqual(plan.candidates[0].argv_add_flags,
                         ['--async-scheduling'])
        self.assertEqual(plan.candidates[0].profiles, ['prefill'])
        self.assertTrue(
            any('D24_moe_tp_without_ep' in r for r in plan.rejected),
            plan.rejected,
        )

    def test_coupled_pair_paired_with_ep_mesh_or_rejected(self):
        serving = [
            _finding('D15_torchtpu_fused_ep_moe', 'USE_MOE_FUSED_EP_KERNEL',
                     '1')
        ]
        md = ('### Candidate 1: EP\n'
              '- **ICI Mesh**: {"tp": 2, "dp": 4, "ep": 8}\n'
              '### Candidate 2: TP4\n'
              '- **ICI Mesh**: {"tp": 4, "dp": 2}\n')
        plan = composer.compose_plan(serving,
                                     sharding_markdown=md,
                                     baseline_mesh={
                                         'tp': 2,
                                         'dp': 4
                                     })
        combo = plan.candidates[0]
        self.assertEqual(combo.skill, 'serving-optimizer+sharding')
        self.assertEqual(combo.env_set, {'USE_MOE_FUSED_EP_KERNEL': '1'})
        self.assertIn('--enable-expert-parallel', combo.argv_add_flags)
        self.assertLen(plan.candidates, 3)  # combo + two mesh candidates.

        plan = composer.compose_plan(
            serving,
            sharding_markdown=
            '### C1: TP4\n- **ICI Mesh**: {"tp": 4, "dp": 2}\n',
            baseline_mesh={
                'tp': 2,
                'dp': 4
            },
        )
        self.assertEqual([c.skill for c in plan.candidates], ['sharding'])
        self.assertIn('rule a', ' '.join(plan.rejected))

    def test_sharding_markdown_serving_knobs_dropped_and_duplicates(self):
        md = (
            '### Candidate 1: TP2\n'
            '- **ICI Mesh**: {"tp": 2, "dp": 4}\n'
            '- **Serving Knobs**: {"--async-scheduling": true}\n'
            '- **Rationale**: addresses D7_collective_bound.\n'
            '- **Provenance**: skill=sharding; findings=D7_collective_bound\n'
            '### Candidate 2: TP2 again\n'
            '- **ICI Mesh**: {"model": 2, "data": 4}\n'
            '### Candidate 3: baseline\n'
            '- **ICI Mesh**: {"tp": 4, "dp": 2}\n')
        plan = composer.compose_plan(None,
                                     sharding_markdown=md,
                                     baseline_mesh={
                                         'tp': 4,
                                         'dp': 2
                                     })
        self.assertLen(plan.candidates, 1)
        c = plan.candidates[0]
        self.assertEqual(c.findings, ['D7_collective_bound'])
        self.assertNotIn('--async-scheduling', c.argv_add_flags)
        self.assertIn('skill=sharding; findings=D7_collective_bound', c.tool)
        text = ' '.join(plan.rejected + plan.notes)
        self.assertIn('dropped its Serving Knobs line', text)
        self.assertIn('duplicate', text)
        self.assertIn('same as the baseline mesh', text)


if __name__ == '__main__':
    absltest.main()
