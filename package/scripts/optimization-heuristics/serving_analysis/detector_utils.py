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
"""Helpers shared by the serving detector modules.

Phase statistics, evidence builders, op and program predicates, knob readers,
unit formatting, and the source references that findings cite. Used by
`detectors.py`, `catalog_detectors.py` and `headroom_detectors.py`.
"""

import math
import re

import pandas as pd
from serving_analysis import context as context_lib
from serving_analysis import features as features_lib
from serving_analysis import findings, knobs, op_classes, segmenter

# Source study labels cited in `Finding.source_refs`.
DOC_PREFILL_MICROBATCH_OVERLAP = 'Prefill microbatching & dual-batch overlap study'
DOC_GLM52_SERVING_OPT = 'GLM-5.2 serving optimization study'

# Phases with fewer steps/device are skipped when another phase has more.
MIN_PHASE_STEPS = 3
# Decode all-gathers at least this big gather weights, not activations
# (16M elements = 32 MB in bf16; decode activations are batch x hidden).
MIN_WEIGHT_AG_ELEMS = 1 << 24
# Phases whose steps carry prefill tokens.
PREFILL_LIKE = (segmenter.PREFILL, segmenter.PREFILL_OR_MIXED, segmenter.MIXED)
# vLLM-TorchTPU fused expert-parallel MoE kernel (`fused_ep_moe*`).
FUSED_EP_OP = r'fused[-_]ep[-_]moe'
# Programs that only copy or convert buffers (transfer profiles).
_COPY_PROGRAM = r'^jit__?(?:identity|copy|device_put|convert_element_type)\b'
# fp8 / int8 operands in HLO text, and KV-cache op names.
_KV_QUANT_OPERAND = re.compile(r'f8e4m3|f8e5m2|\bs8\[')
_KV_NAME = re.compile(r'kv|cache', re.I)


def ms(ns: float) -> float:
    """Nanoseconds to milliseconds."""
    return float(ns) / 1e6


def us(ns: float) -> float:
    """Nanoseconds to microseconds."""
    return float(ns) / 1e3


def pct(fraction: float) -> float:
    """Fraction to percent."""
    return 100.0 * float(fraction)


def parse_dims(dims: str) -> list[int]:
    """[A, B, C] from a `collective_groups` 'AxBxC' dims string ([] if invalid)."""
    out = []
    for d in str(dims or '').split('x'):
        if not d.isdigit():
            return []
        out.append(int(d))
    return out


def n_elems(dims: str) -> int:
    """Element count of a `collective_groups` 'AxBxC' dims string (0 if invalid)."""
    parsed = parse_dims(dims)
    return math.prod(parsed) if parsed else 0


def framework(ctx: context_lib.ServingContext,
              f: features_lib.Features) -> str:
    """The serving framework: the context's if given, else the detected one."""
    return ctx.framework if ctx.framework != 'auto' else f.seg.framework


def is_vllm(ctx: context_lib.ServingContext, f: features_lib.Features) -> bool:
    return framework(ctx, f) in ('vllm_tpu', 'vllm_torchtpu')


def is_torchtpu(ctx: context_lib.ServingContext,
                f: features_lib.Features) -> bool:
    return framework(ctx, f) == 'vllm_torchtpu'


def is_configured_vllm_tpu(ctx: context_lib.ServingContext,
                           f: features_lib.Features) -> bool:
    """Whether this is vLLM-TPU with a serving config (knob states known)."""
    return framework(ctx, f) == 'vllm_tpu' and ctx.serving_config is not None


def knob_value(ctx: context_lib.ServingContext, key: str) -> str:
    """A registered knob's current value, stripped ('' without a config)."""
    return str(knobs.current_value(key, ctx.serving_config) or '').strip()


def knob_int(ctx: context_lib.ServingContext, key: str, default: int) -> int:
    """A knob's current value as an int; `default` if unknown or not an int."""
    try:
        return int(knob_value(ctx, key))
    except ValueError:
        return default


def knob_float(ctx: context_lib.ServingContext, key: str,
               default: float) -> float:
    """A knob's current value as a float; `default` if unknown or not a number."""
    try:
        return float(knob_value(ctx, key))
    except ValueError:
        return default


# Inputs at least this many times the outputs make a workload prefill-heavy.
PREFILL_HEAVY_MIN_RATIO = 2


def is_prefill_heavy(ctx: context_lib.ServingContext) -> bool:
    """Whether the benchmark's inputs are at least twice its outputs.

    Never on the decode engine of a disaggregated P/D pair: it runs no prefill,
    so the input/output ratio does not change its decode batches.
    """
    return bool(not ctx.is_pd_decode and ctx.input_len and ctx.output_len
                and ctx.input_len >= PREFILL_HEAVY_MIN_RATIO * ctx.output_len)


def stat(f: features_lib.Features, phase: str, col: str) -> float:
    """A `Features.phase_stats` column for a phase (NaN if the phase is absent)."""
    ps = f.phase_stats
    row = ps[ps['phase'] == phase]
    return float(row[col].iloc[0]) if len(row) else float('nan')


def active_phases(f: features_lib.Features) -> list[str]:
    """Phases with enough steps to analyze.

    A phase with fewer than MIN_PHASE_STEPS steps/device is skipped when another
    phase has at least that many (e.g. the one warm-up prefill captured in a
    decode profile). If every phase is small (a 256K-context prefill profile
    may hold 1-2 steps), all are kept and D14 flags the low step count.
    """
    n = {p: stat(f, p, 'n_steps_per_device') for p in f.seg.phases}
    big = [p for p, v in n.items() if v >= MIN_PHASE_STEPS]
    return big if big else list(f.seg.phases)


def phase_ops(f: features_lib.Features, phase: str) -> pd.DataFrame:
    """Step ops of a phase ('all' for every phase)."""
    so = f.step_ops
    return so if phase == 'all' else so[so['phase'] == phase]


def n_steps(f: features_lib.Features, phase: str) -> int:
    """Number of steps of a phase, over all devices ('all' for every phase)."""
    st = f.seg.steps
    return len(st) if phase == 'all' else int((st['phase'] == phase).sum())


def share(f: features_lib.Features, phase: str, ops: pd.DataFrame) -> float:
    """Self time of `ops` per step, as a fraction of the mean step time."""
    n, dur = n_steps(f, phase), stat(f, phase, 'mean_dur_ns')
    if not n or not dur or ops.empty:
        return 0.0
    return float(ops['self_ns'].sum()) / n / dur


def name_mask(ops: pd.DataFrame, pattern: str) -> pd.Series:
    """Ops whose base op name (op name if absent) matches `pattern`."""
    base = ops['base_op'].fillna('') if 'base_op' in ops.columns else ops['op']
    return base.str.contains(pattern, case=False, regex=True)


def is_copy_program(f: features_lib.Features) -> bool:
    """Whether every step op runs in a copy / convert program."""
    so = f.step_ops
    return (not so.empty
            and so['module'].astype(str).str.contains(_COPY_PROGRAM).all())


def kv_dequant_outside_kernel(
        f: features_lib.Features,
        ctx: context_lib.ServingContext) -> tuple[bool, str]:
    """Whether quantized KV is converted to bf16 before the attention kernel.

    Args:
      f: Features.
      ctx: Serving context.

    Returns:
      (detected, reason).
    """
    m = f.meta
    if not m.empty:
        non_kernel = m['op_class'] != op_classes.ATTENTION_KERNEL
        names = m['tf_op_name'].fillna('')
        in_attn = names.str.contains(op_classes.ATTENTION_SCOPE)
        # fp8 attention weights also convert to bf16 in this scope; require a
        # KV-cache name to keep weight dequant out.
        kv_like = (names + ' ' + m['op'].fillna('')).str.contains(_KV_NAME)
        quant_in = m['kernel_text'].fillna('').str.contains(_KV_QUANT_OPERAND)
        deq = m[non_kernel
                & in_attn
                & kv_like
                & quant_in
                & (m['out_dtype'] == 'bf16')]
        if not deq.empty:
            return True, f'`{deq["base_op"].iloc[0]}` converts fp8/int8 to bf16'
    cur = (knob_value(ctx, 'vllm_tpu.cli.kv_cache_dtype')
           if is_configured_vllm_tpu(ctx, f) else '')
    if cur.lower().startswith('fp8') and kv_cache_dtype(f) == 'bf16':
        return True, f'kv_cache_dtype={cur} but the attention kernel reads bf16'
    return False, ''


def kv_cache_dtype(f: features_lib.Features) -> str:
    """'fp8' | 'int8' | 'bf16' | '' from the attention kernel operand text."""
    meta = f.meta[f.meta['op_class'] == op_classes.ATTENTION_KERNEL]
    text = ' '.join(meta['kernel_text'].dropna().astype(str).head(20))
    if 'f8e4m3' in text or 'f8e5m2' in text:
        return 'fp8'
    if re.search(r'\bs8\[', text):
        return 'int8'
    if 'bf16' in text:
        return 'bf16'
    return ''


def xprof_link(sid: str, start_ns: float, end_ns: float) -> str | None:
    """Trace-viewer deep link; None, since these traces are local files.

    Kept so evidence keeps a place for a link when a hosted trace viewer
    exists for the session.
    """
    del sid, start_ns, end_ns
    return None


def step_link(sid: str, f: features_lib.Features, phase: str) -> str | None:
    """Trace-viewer link to a median-length step of the phase (None standalone)."""
    del sid, f, phase
    return None


def step_evidence(f: features_lib.Features, phase: str,
                  sid: str) -> findings.Evidence:
    """Mean step time of a phase, linked to a representative step."""
    return findings.Evidence(
        metric=f'{phase} step time (mean)',
        value=ms(stat(f, phase, 'mean_dur_ns')),
        unit=' ms',
        detail=f'{stat(f, phase, "n_steps_per_device"):.0f} steps/device',
        xprof_link=step_link(sid, f, phase),
    )


def top_op_evidence(f: features_lib.Features,
                    phase: str,
                    op_class: str,
                    n: int = 2) -> list[findings.Evidence]:
    """Per-step time of the top `n` ops of a class in a phase."""
    return [
        findings.Evidence(
            metric=f'top op `{r["op"]}`',
            value=us(r['ns_per_step']),
            unit=' µs/step',
            detail=f'{r["count"]:.0f}×/step',
        ) for _, r in features_lib.top_ops(f.step_ops, f.seg.steps, phase,
                                           op_class, n).iterrows()
    ]


def base_op_evidence(f: features_lib.Features,
                     phase: str,
                     ops: pd.DataFrame,
                     n: int = 3) -> list[findings.Evidence]:
    """Per-step time of the top `n` base ops (instance suffixes stripped)."""
    steps = n_steps(f, phase)
    if ops.empty or not steps or 'base_op' not in ops.columns:
        return []
    agg = ops.groupby('base_op')['self_ns'].agg(['sum', 'size'])
    agg = agg.sort_values(by='sum', ascending=False).head(n)
    return [
        findings.Evidence(
            f'op `{str(b)[:90]}`',
            us(float(r['sum']) / steps),
            ' µs/step',
            detail=f'{float(r["size"]) / steps:.0f}×/step',
        ) for b, r in agg.iterrows()
    ]
