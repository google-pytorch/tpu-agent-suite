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
"""Tests for knobs.py, findings.py and context.parse_serving_command."""

import os
import sys

from absl.testing import absltest

# The serving_analysis package sits one directory up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving_analysis import context, findings, knobs


class KnobsTest(absltest.TestCase):

    def test_registry_well_formed(self):
        names = set()
        for key, k in knobs.REGISTRY.items():
            self.assertEqual(key, k.key)
            self.assertIn(
                k.kind,
                (
                    'env',
                    'cli',
                    'additional_config',
                    'maxtext_yml',
                ),
            )
            # Relative to the installed package, so the image check can find it.
            self.assertFalse(k.source_path.startswith(('/', 'third_party/')),
                             key)
            self.assertEqual(
                k.source_path.split('/')[0],
                k.source_path.split('/')[0].lower(), key)
            self.assertTrue(k.source_token, key)
            if k.requires_patch:
                self.assertEqual(k.framework, 'vllm_torchtpu', key)
            self.assertNotIn((k.framework, k.name), names, key)
            names.add((k.framework, k.name))

    def test_parse_export_chain_and_expert_parallel(self):
        cfg = context.parse_serving_command(
            'export VLLM_ENGINE_READY_TIMEOUT_S=21600 && '
            'export USE_MOE_FUSED_EP_KERNEL=1 && vllm serve ${MODEL_PATH} '
            '--tensor-parallel-size=1 --data-parallel-size=16 '
            '--enable-expert-parallel --max-num-seqs 1 --kv-cache-dtype fp8')
        self.assertEqual(cfg['USE_MOE_FUSED_EP_KERNEL'], '1')
        self.assertEqual(cfg['kv-cache-dtype'], 'fp8')
        ctx = context.ServingContext(serving_config=cfg).finalize()
        self.assertEqual((ctx.tp, ctx.dp, ctx.ep, ctx.max_num_seqs),
                         (1, 16, 16, 1))
        c = knobs.make_change('vllm_torchtpu.env.USE_MOE_FUSED_EP_KERNEL', '1',
                              cfg)
        self.assertTrue(c.already_set)

    def test_unknown_knob_raises(self):
        with self.assertRaises(KeyError):
            knobs.get('vllm_tpu.env.DOES_NOT_EXIST')

    def test_current_value_unknown_without_config(self):
        self.assertIsNone(
            knobs.current_value('vllm_tpu.env.MOE_APPROX_TOPK', None))

    def test_current_value_default_when_absent(self):
        self.assertEqual(
            knobs.current_value('vllm_tpu.env.MIN_TOKEN_BUCKET', {}), '16')

    def test_make_change_already_set_normalizes_bools(self):
        cfg = context.parse_serving_command(
            'MOE_APPROX_TOPK=1 vllm serve m --max-num-seqs 256 '
            '--additional-config \'{"sharding": {"sharding_strategy": '
            '{"enable_dp_attention": true}}, "enable_continue_decode": false}\''
        )
        c = knobs.make_change('vllm_tpu.env.MOE_APPROX_TOPK', 'true', cfg)
        self.assertTrue(c.already_set)
        c = knobs.make_change('vllm_tpu.additional_config.enable_dp_attention',
                              'true', cfg)
        self.assertTrue(c.already_set)
        c = knobs.make_change(
            'vllm_tpu.additional_config.enable_continue_decode', 'true', cfg)
        self.assertFalse(c.already_set)
        c = knobs.make_change('vllm_tpu.cli.max_num_seqs', '128', cfg)
        self.assertEqual(c.current_value, '256')

    def test_finding_score_zero_when_already_set(self):
        cfg = {'MOE_APPROX_TOPK': 'True'}
        f = findings.Finding(
            detector_id='x',
            title='t',
            phase='decode',
            summary='s',
            evidence=[],
            knob_changes=[
                knobs.make_change('vllm_tpu.env.MOE_APPROX_TOPK', 'true', cfg)
            ],
            risk=findings.Risk.QUALITY_GATED,
            confidence=findings.Confidence.HIGH,
            est_step_fraction_saved=0.1,
        )
        self.assertEqual(f.score, 0.0)
        self.assertIn('ALREADY SET', findings.findings_to_markdown([f]))

    def test_negated_boolean_cli_flag(self):
        cfg = context.parse_serving_command(
            'vllm serve m --no-async-scheduling --block-size 64')
        self.assertEqual(cfg['async-scheduling'], 'false')
        self.assertEqual(cfg['async_scheduling'], 'false')
        self.assertEqual(knobs.current_value('vllm.cli.async_scheduling', cfg),
                         'false')
        self.assertEqual(knobs.current_value('vllm.cli.block_size', cfg), '64')


class ImageCheckTest(absltest.TestCase):
    """--verify_knobs / --knob_source_dir against an image source tree."""

    def _image(self) -> knobs.ImageSource:
        root = self.create_tempdir()
        root.create_file(
            'vllm_torchtpu/envs.py',
            content=('environment_variables = {\n'
                     '    "USE_MOE_FUSED_EP_KERNEL": lambda: os.getenv(\n'
                     '        "USE_MOE_FUSED_EP_KERNEL", "0"),\n'
                     '}\n'),
        )
        root.create_file(
            'vllm/engine/arg_utils.py',
            content=('class EngineArgs:\n'
                     '  max_num_batched_tokens: int | None = None\n'
                     '  async_scheduling: bool | None = None\n'),
        )
        return knobs.ImageSource(root.full_path)

    def test_has_knob(self):
        img = self._image()
        self.assertTrue(
            img.has_knob('vllm_torchtpu.env.USE_MOE_FUSED_EP_KERNEL'))
        self.assertTrue(img.has_knob('vllm.cli.max_num_batched_tokens'))
        self.assertFalse(
            img.has_knob('vllm_torchtpu.env.TPU_ROPE_CACHE_TRUNCATE'))

    def test_verify_image_lists_missing_with_patch(self):
        missing = knobs.verify_image(self._image(), 'vllm_torchtpu')
        text = '\n'.join(missing)
        self.assertIn('TPU_ROPE_CACHE_TRUNCATE', text)
        self.assertNotIn('USE_MOE_FUSED_EP_KERNEL', text)
        self.assertNotIn('MOE_APPROX_TOPK',
                         text)  # vllm_tpu knob, other image.

    def test_drop_knobs_absent_from_image(self):
        cfg = {}

        def finding(det, *keys):
            return findings.Finding(
                detector_id=det,
                title='t',
                phase='decode',
                summary='s',
                evidence=[],
                knob_changes=[knobs.make_change(k, '1', cfg) for k in keys],
                risk=findings.Risk.PERF_ONLY,
                confidence=findings.Confidence.MEDIUM,
            )

        found = [
            finding('D9_rope_cache_copy',
                    'vllm_torchtpu.env.TPU_ROPE_CACHE_TRUNCATE'),
            finding(
                'D29_async_scheduling_off',
                'vllm.cli.async_scheduling',
                'vllm_torchtpu.env.TPU_ROPE_CACHE_TRUNCATE',
            ),
        ]
        out, notes = knobs.drop_knobs_absent_from_image(found, self._image())
        self.assertEqual([f.detector_id for f in out],
                         ['D29_async_scheduling_off'])
        self.assertEqual([kc.name for kc in out[0].knob_changes],
                         ['--async-scheduling'])
        self.assertLen(notes, 2)
        self.assertIn('TPU_ROPE_CACHE_TRUNCATE', notes[0])

    def test_image_source_requires_python(self):
        with self.assertRaises(ValueError):
            knobs.ImageSource(self.create_tempdir().full_path)


class CommandSynthesisTest(absltest.TestCase):
    """context.apply_knob_changes (AnalysisResult.recommended_command)."""

    def test_apply_knob_changes_updates_and_synthesizes(self):
        cmd = ('MOE_APPROX_TOPK=0 vllm serve m --max-num-seqs 256 '
               '--no-async-scheduling --additional-config '
               '\'{"enable_continue_decode": false}\'')
        cfg = context.parse_serving_command(cmd)
        changes = [
            knobs.make_change('vllm_tpu.env.MOE_APPROX_TOPK', 'true', cfg),
            knobs.make_change('vllm_tpu.env.USE_BATCHED_RPA_KERNEL', 'true',
                              cfg),
            knobs.make_change('vllm.cli.max_num_seqs', '128', cfg),
            knobs.make_change('vllm.cli.async_scheduling', 'true', cfg),
            knobs.make_change(
                'vllm_tpu.additional_config.enable_continue_decode', 'true',
                cfg),
            knobs.make_change('vllm_tpu.additional_config.enable_dp_attention',
                              'true', cfg),
        ]
        patched = context.apply_knob_changes(cmd, changes, 'vllm_tpu')
        patched_cfg = context.parse_serving_command(patched)
        self.assertEqual(patched_cfg['MOE_APPROX_TOPK'], 'true')
        self.assertEqual(patched_cfg['USE_BATCHED_RPA_KERNEL'], 'true')
        self.assertEqual(patched_cfg['max-num-seqs'], '128')
        self.assertEqual(patched_cfg['async-scheduling'], 'true')
        self.assertEqual(patched_cfg['enable_continue_decode'], 'true')
        self.assertEqual(
            patched_cfg['sharding.sharding_strategy.enable_dp_attention'],
            'true')
        snippet = context.apply_knob_changes(None, changes, 'vllm_tpu')
        self.assertIn('MOE_APPROX_TOPK=true', snippet)
        self.assertIn('--max-num-seqs=128', snippet)

    def test_already_set_and_duplicate_changes_skipped(self):
        cmd = 'MOE_APPROX_TOPK=true vllm serve m'
        cfg = context.parse_serving_command(cmd)
        changes = [
            knobs.make_change('vllm_tpu.env.MOE_APPROX_TOPK', 'true', cfg),
            knobs.make_change('vllm.cli.max_num_seqs', '128', cfg),
            knobs.make_change('vllm.cli.max_num_seqs', '64', cfg),
        ]
        self.assertEqual(
            context.apply_knob_changes(cmd, changes, 'vllm_tpu'),
            'MOE_APPROX_TOPK=true vllm serve m --max-num-seqs=128',
        )
        self.assertEqual(context.apply_knob_changes(cmd, [], 'vllm_tpu'), cmd)

    def test_unspliced_kinds_and_patches_become_comments(self):
        cmd = 'vllm serve m'
        changes = [
            findings.KnobChange(knob='vllm_torchtpu.env.TPU_X',
                                kind='env',
                                name='TPU_X',
                                suggested_value='1',
                                requires_patch='x.patch'),
            findings.KnobChange(
                knob='xla.flag.x',
                kind='xla_flag',
                name='--xla_x',
                suggested_value='1',
            ),
        ]
        out = context.apply_knob_changes(cmd, changes, 'vllm_torchtpu')
        lines = out.splitlines()
        self.assertEqual(lines[0], 'TPU_X=1 vllm serve m')
        self.assertIn('# Also set (xla_flag): --xla_x=1', lines)
        self.assertIn('# TPU_X requires x.patch', lines)


if __name__ == '__main__':
    absltest.main()
