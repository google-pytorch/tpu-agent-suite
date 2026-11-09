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
"""Outcome-backed guardrails applied to detector findings before ranking.

Each guardrail removes a knob change that measured A/B runs showed to be
harmful in a given regime, unless the trace shows the specific evidence that
would justify it. Removed changes are reported as notes so the agent can see
what was suppressed and why.
"""

from collections.abc import Sequence

from serving_analysis import context as context_lib
from serving_analysis import findings as findings_lib

_MNBT_KNOBS = frozenset({
    'vllm.cli.max_num_batched_tokens',
    'vllm_tpu.cli.max_num_batched_tokens',
})
# Decode stall caused by prefill chunks (D3 'decode stall' evidence, % of
# profiled step time) above which a larger prefill budget may be proposed
# even when TTFT matters.
MNBT_MIN_PREFILL_STALL_PCT = 25.0
_MNBT_OUTCOME = (
    'raising --max-num-batched-tokens raised median TTFT on 3 of 3 vLLM-'
    'TorchTPU models (Gemma-3 +13..+88%, Gemma-4 +42..+124%, Qwen3.8 '
    '+24..+89%) with no throughput gain (A/B on tpu7x against a same-session control,'
    ' 2026-10-14)')


def _as_int(v: str | None) -> int | None:
    try:
        return int(str(v).strip()) if v is not None else None
    except ValueError:
        return None


def _prefill_stall_pct(f: findings_lib.Finding) -> float:
    for e in f.evidence:
        if 'stall' in e.metric and isinstance(e.value, (int, float)):
            return float(e.value)
    return 0.0


def _is_mnbt_increase(kc: findings_lib.KnobChange,
                      ctx: context_lib.ServingContext) -> bool:
    if kc.knob not in _MNBT_KNOBS:
        return False
    new = _as_int(kc.suggested_value)
    cur = _as_int(kc.current_value)
    if cur is None:
        cur = ctx.max_num_batched_tokens
    if new is None:
        return False
    return cur is None or new > cur


def apply(
    found: Sequence[findings_lib.Finding], ctx: context_lib.ServingContext
) -> tuple[list[findings_lib.Finding], list[str]]:
    """Applies guardrails; returns (findings, notes about suppressed changes).

  A finding left with no knob change, no kernel follow-up and a non-diagnostic
  risk is dropped (it would otherwise be an empty recommendation).

  Args:
    found: Detector findings.
    ctx: Serving context.
  """
    out, notes = [], []
    for f in found:
        kept = []
        for kc in f.knob_changes:
            if (ctx.ttft_sensitive and _is_mnbt_increase(kc, ctx)
                    and _prefill_stall_pct(f) < MNBT_MIN_PREFILL_STALL_PCT):
                notes.append(
                    f'Guardrail: dropped `{kc.name}={kc.suggested_value}` from'
                    f' {f.detector_id}: {_MNBT_OUTCOME}. Allowed only when TTFT does'
                    ' not matter or prefill stalls decode by'
                    f' >={MNBT_MIN_PREFILL_STALL_PCT:.0f}% of step time.')
                continue
            kept.append(kc)
        if len(kept) != len(f.knob_changes):
            f.knob_changes = kept
            if not kept and not f.kernel_followup:
                continue
        out.append(f)
    return out, notes
