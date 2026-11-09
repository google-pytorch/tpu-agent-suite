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
"""Tests for guardrails.py (the --max-num-batched-tokens TTFT guardrail)."""

import os
import sys

from absl.testing import absltest

# The serving_analysis package sits one directory up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving_analysis import context, findings, guardrails, knobs

_MNBT = 'vllm.cli.max_num_batched_tokens'


def _finding(*changes, stall_pct=None):
    evidence = []
    if stall_pct is not None:
        evidence.append(findings.Evidence('decode stall', stall_pct, '%'))
    return findings.Finding(
        detector_id='D3_prefill_decode_interference',
        title='t',
        phase='mixed',
        summary='s',
        evidence=evidence,
        knob_changes=list(changes),
        risk=findings.Risk.PERF_ONLY,
        confidence=findings.Confidence.MEDIUM,
    )


def _ctx(ttft_sensitive=True):
    cfg = context.parse_serving_command(
        'vllm serve m --max-num-batched-tokens 4096')
    return context.ServingContext(
        framework='vllm_torchtpu',
        serving_config=cfg,
        ttft_sensitive=ttft_sensitive,
    )


class ApplyTest(absltest.TestCase):

    def test_raising_batched_tokens_is_dropped_when_ttft_matters(self):
        ctx = _ctx()
        f = _finding(knobs.make_change(_MNBT, '8192', ctx.serving_config))
        out, notes = guardrails.apply([f], ctx)
        self.assertEqual(out, [])
        self.assertLen(notes, 1)
        self.assertIn('raised median TTFT', notes[0])

    def test_allowed_when_ttft_does_not_matter(self):
        ctx = _ctx(ttft_sensitive=False)
        f = _finding(knobs.make_change(_MNBT, '8192', ctx.serving_config))
        out, notes = guardrails.apply([f], ctx)
        self.assertLen(out, 1)
        self.assertEqual(notes, [])

    def test_allowed_when_prefill_stalls_decode(self):
        ctx = _ctx()
        f = _finding(knobs.make_change(_MNBT, '8192', ctx.serving_config),
                     stall_pct=40.0)
        out, _ = guardrails.apply([f], ctx)
        self.assertLen(out[0].knob_changes, 1)

    def test_lowering_is_not_guarded(self):
        ctx = _ctx()
        f = _finding(knobs.make_change(_MNBT, '2048', ctx.serving_config))
        out, notes = guardrails.apply([f], ctx)
        self.assertLen(out, 1)
        self.assertEqual(notes, [])


if __name__ == '__main__':
    absltest.main()
