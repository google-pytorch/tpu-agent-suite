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
"""Tests for the deterministic Stage 2/3 serving + sharding planner."""

import json
import os
import sys

from absl.testing import absltest

# The serving_analysis package sits one directory up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving_analysis import findings as findings_lib
from serving_analysis import planner

_TESTDATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testdata')


def _load_case(name: str) -> dict[str, object]:
    with open(os.path.join(_TESTDATA, name)) as f:
        return json.load(f)


class PlannerTest(absltest.TestCase):

    def _sample_findings(self) -> list[findings_lib.Finding]:
        return [
            findings_lib.Finding(
                detector_id='D8_kv_head_padding',
                title='KV-head padding under TP=16 (GQA replication)',
                phase='decode',
                summary='num_kv_heads=8 is padded/replicated across TP=16.',
                evidence=[],
                knob_changes=[
                    findings_lib.KnobChange(
                        knob='maxtext.yml.enable_dp_attention',
                        kind='maxtext_yml',
                        name='enable_dp_attention',
                        suggested_value='true',
                        current_value='false',
                    )
                ],
                risk=findings_lib.Risk.PERF_ONLY,
                confidence=findings_lib.Confidence.HIGH,
                est_step_fraction_saved=0.18,
            ),
            findings_lib.Finding(
                detector_id='D23_decode_fsdp_weight_ag',
                title='Decode step stalls on FSDP weight all-gather',
                phase='decode',
                summary='All-gather takes 45% of decode step time.',
                evidence=[],
                knob_changes=[],
                risk=findings_lib.Risk.PERF_ONLY,
                confidence=findings_lib.Confidence.HIGH,
                est_step_fraction_saved=0.35,
            ),
            findings_lib.Finding(
                detector_id='D4_moe_topk_hotspot',
                title='MoE softmax/top-k routing hotspot',
                phase='decode',
                summary='Use approximate top-k.',
                evidence=[],
                knob_changes=[
                    findings_lib.KnobChange(
                        knob='tpu_inference.env.MOE_APPROX_TOPK',
                        kind='env',
                        name='MOE_APPROX_TOPK',
                        suggested_value='1',
                        current_value=None,
                    )
                ],
                risk=findings_lib.Risk.QUALITY_GATED,
                confidence=findings_lib.Confidence.HIGH,
                est_step_fraction_saved=0.12,
                validation='Run accuracy eval on GSM8K/MMLU.',
            ),
            findings_lib.Finding(
                detector_id='D31_gmm_tile_misalignment',
                title='GMM tile alignment under TP',
                phase='decode',
                summary=
                ('Per-device expert intermediate size is not a multiple of 128.'
                 ),
                evidence=[],
                knob_changes=[],
                risk=findings_lib.Risk.PERF_ONLY,
                confidence=findings_lib.Confidence.MEDIUM,
                est_step_fraction_saved=0.08,
                kernel_followup='Align expert tile size to 128.',
            ),
            findings_lib.Finding(
                detector_id='D31_gmm_rhs_tile_alignment',
                title='RHS GMM tile alignment under TP',
                phase='decode',
                summary=
                'Per-device expert intermediate size is not a multiple of 128.',
                evidence=[],
                knob_changes=[],
                risk=findings_lib.Risk.PERF_ONLY,
                confidence=findings_lib.Confidence.MEDIUM,
                est_step_fraction_saved=0.08,
                kernel_followup='Align expert tile size to 128.',
            ),
        ]

    def test_finding_default_coupling_metadata(self):
        items = self._sample_findings()
        d8, d23, d4, d31, d31_rhs = items
        self.assertEqual(d8.lever, 'serving_knob')
        self.assertEqual(d8.couples_with, ['mesh', 'logical_sharding'])
        self.assertEqual(d23.lever, 'mesh')
        self.assertEqual(d23.couples_with, ['mesh', 'logical_sharding'])
        self.assertEqual(d4.lever, 'serving_knob')
        self.assertEqual(d4.couples_with, [])
        self.assertEqual(d31.lever, 'serving_knob')
        self.assertEqual(d31.couples_with, ['mesh'])
        self.assertEqual(d31_rhs.lever, 'serving_knob')
        self.assertEqual(d31_rhs.couples_with, ['mesh'])

    def test_extract_sharding_constraints(self):
        constraints = planner.extract_sharding_constraints(
            self._sample_findings())
        self.assertEqual(
            constraints['co_proposed_serving_knobs'],
            {'enable_dp_attention': True},
        )
        mesh_ids = [f['detector_id'] for f in constraints['mesh_findings']]
        self.assertIn('D8_kv_head_padding', mesh_ids)
        self.assertIn('D23_decode_fsdp_weight_ag', mesh_ids)
        self.assertTrue(constraints['prompt_guidance'])

    def test_merge_sharding_and_serving_plan_rules_r1_to_r5(self):
        sharding_md = (
            '### Candidate 1: Co-designed TP8 + EP2\n- **ICI Mesh**: `{"data": 1,'
            ' "fsdp": 1, "expert": 2, "model": 8}`\n- **Allow Split Physical'
            ' Axes**: False\n- **Logical Sharding Overrides**: `{"ffw_weight_df":'
            ' ["expert", "model"]}`\n')
        baseline_mesh = {'data': 1, 'fsdp': 16, 'expert': 1, 'model': 1}
        merged = planner.merge_sharding_and_serving_plan(
            sharding_md,
            self._sample_findings(),
            baseline_mesh=baseline_mesh,
        )
        # R1: D8 enable_dp_attention is injected into Candidate 1 as Serving Knobs.
        self.assertIn(
            '- **Serving Knobs**: {"enable_dp_attention": true}',
            merged.augmented_sharding_markdown,
        )
        # R2: D8 and D23 are absorbed into sharding, not duplicated in orthogonal.
        absorbed_ids = [f['detector_id'] for f in merged.absorbed_findings]
        self.assertIn('D8_kv_head_padding', absorbed_ids)
        self.assertIn('D23_decode_fsdp_weight_ag', absorbed_ids)
        orth_ids = [f['detector_id'] for f in merged.orthogonal_findings]
        self.assertEqual(orth_ids, ['D4_moe_topk_hotspot'])
        # R3: Candidate 1 changed tp, ep, fsdp -> recheck the axis-dependent
        # detector IDs that fired (D8, D31); unknown IDs are never listed.
        self.assertEqual(
            merged.recheck_by_candidate['Candidate 1'],
            ['D31_gmm_tile_misalignment', 'D8_kv_head_padding'],
        )

        # R5: User review contains Quality Gated, Unknown Current State,
        # and Mesh Recheck.
        review_text = '\n'.join(merged.user_review_items)
        self.assertIn('[Quality Gated]', review_text)
        self.assertIn('[Unknown Current State]', review_text)
        self.assertIn('[Mesh Recheck]', review_text)

    def test_case_output_findings_and_phase_stats(self):
        # Real --case output (gpt-oss-20b TP2xDP4, 1k1k c512): 3 profiles.
        case = _load_case('gptoss_1k1k_c512_case.json')
        findings = planner.normalize_findings(case)
        roles = {(f['detector_id'], f['profile_role']) for f in findings}
        self.assertIn(('D24_moe_tp_without_ep', 'decode'), roles)
        self.assertIn(('D29_async_scheduling_off', 'prefill'), roles)
        constraints = planner.extract_sharding_constraints(case)
        self.assertEqual(
            constraints['co_proposed_serving_knobs'],
            {'--enable-expert-parallel': True},
        )
        self.assertNotEmpty(constraints['mesh_findings'])
        self.assertNotEmpty(constraints['phase_stats'])
        self.assertEqual(
            {r['profile_role']
             for r in constraints['phase_stats']},
            {'prefill', 'decode', 'mixed'},
        )

    def test_case_output_qwen38_constraints(self):
        case = _load_case('qwen38_8k1k_c256_case.json')
        constraints = planner.extract_sharding_constraints(case)
        mesh_ids = {f['detector_id'] for f in constraints['mesh_findings']}
        self.assertIn('D7_collective_bound', mesh_ids)
        self.assertNotEmpty(constraints['phase_stats'])

    def test_r1_attaches_ep_flag_only_to_ep_candidates(self):
        case = _load_case('gptoss_1k1k_c512_case.json')
        sharding_md = ('### C1: Expert parallelism on the baseline mesh\n'
                       '*   **ICI Mesh:** `{"tp": 2, "dp": 4, "ep": 8}`\n'
                       '*   **Reasoning:** EP over TPxDP.\n'
                       '\n'
                       '## Candidate 2: Baseline mesh\n'
                       '- **ICI Mesh**: {"tp": 2, "dp": 4}\n'
                       '\n'
                       '### Recommended Configuration\n'
                       'C1.\n')
        merged = planner.merge_sharding_and_serving_plan(sharding_md,
                                                         case,
                                                         baseline_mesh={
                                                             'tp': 2,
                                                             'dp': 4
                                                         })
        md = merged.augmented_sharding_markdown
        c1, c2 = md.split('## Candidate 2')
        self.assertIn('**Serving Knobs**: {"--enable-expert-parallel": true}',
                      c1)
        self.assertNotIn('**Serving Knobs**', c2)
        self.assertIn('**Serving Knob Conflicts**', c2)
        self.assertEqual(list(merged.knob_conflicts), ['Candidate 2'])
        self.assertIn('needs EP>1', merged.knob_conflicts['Candidate 2'][0])
        self.assertIn('[Knob Conflict] Candidate 2',
                      '\n'.join(merged.user_review_items))
        # R3: only C1 changes a mesh axis (ep 1 -> 8).
        self.assertEqual(merged.recheck_by_candidate,
                         {'Candidate 1': ['D24_moe_tp_without_ep']})

    def test_r1_never_injects_d8_tp_dp_into_other_mesh(self):
        d8 = findings_lib.Finding(
            detector_id='D8_kv_head_padding',
            title='KV-head padding',
            phase='decode',
            summary='s',
            evidence=[],
            knob_changes=[
                findings_lib.KnobChange(
                    'vllm.cli.tensor_parallel_size',
                    'cli',
                    '--tensor-parallel-size',
                    '1',
                    '4',
                ),
                findings_lib.KnobChange(
                    'vllm.cli.data_parallel_size',
                    'cli',
                    '--data-parallel-size',
                    '8',
                    '2',
                ),
            ],
            risk=findings_lib.Risk.PERF_ONLY,
            confidence=findings_lib.Confidence.HIGH,
        )
        sharding_md = ('### Proposed ICI Mesh Configuration 1 - TP4\n'
                       '- **ICI Mesh**: {"model": 4, "data": 2}\n'
                       '### Proposed ICI Mesh Configuration 2 - DP8\n'
                       '- **ICI Mesh**: {"model": 1, "data": 8}\n')
        merged = planner.merge_sharding_and_serving_plan(sharding_md, [d8])
        blocks = merged.augmented_sharding_markdown.split('Configuration 2')
        self.assertNotIn('**Serving Knobs**', blocks[0])
        self.assertIn('--tensor-parallel-size=1', blocks[0])
        self.assertIn('**Serving Knobs**', blocks[1])
        self.assertNotIn('Conflicts', blocks[1])

    def test_existing_serving_knobs_line_is_checked_not_duplicated(self):
        case = _load_case('gptoss_1k1k_c512_case.json')
        sharding_md = (
            '### Candidate 1: Baseline mesh\n'
            '- **ICI Mesh**: {"tp": 2, "dp": 4}\n'
            '- **Serving Knobs**: {"--enable-expert-parallel": true}\n')
        merged = planner.merge_sharding_and_serving_plan(sharding_md, case)
        md = merged.augmented_sharding_markdown
        self.assertEqual(md.count('**Serving Knobs**'), 1)
        self.assertIn('Candidate 1', merged.knob_conflicts)


if __name__ == '__main__':
    absltest.main()
