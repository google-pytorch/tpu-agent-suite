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
"""Deterministic serving detectors (D1-D22, D24).

Each detector inspects `features.Features` + `context.ServingContext` and emits
zero or more `findings.Finding`s. Detectors only recommend knobs registered in
knobs.py (via knobs.make_change); anything without a knob is emitted as a
kernel/compiler follow-up or a diagnostic.

Thresholds are module constants so they can be tuned against golden sessions.
"""

# pylint: disable=missing-function-docstring,missing-class-docstring,g-doc-args,g-doc-return-or-yield,g-docstring-has-escape,g-long-ternary,g-explicit-length-test,g-complex-comprehension,protected-access,cell-var-from-loop,comparison-with-itself,unused-argument

import math
import re
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from serving_analysis import context as context_lib
from serving_analysis import detector_utils as du
from serving_analysis import features as features_lib
from serving_analysis import findings as F
from serving_analysis import knobs, limits
from serving_analysis import op_classes as oc
from serving_analysis import segmenter

Finding = F.Finding
Evidence = F.Evidence

_DOC_QWEN3_CODER = 'Qwen3-Coder-480B serving study on TPU v7x'

# D1 host gap.
D1_MIN_IDLE_FRAC = 0.10
D1_MIN_MEAN_IDLE_NS = 100_000
# D2 padding.
D2_MIN_PAD_RATIO = 0.25
# MoE expert matmuls must be at least this share of the decode step.
D2_MIN_MOE_SHARE = 0.05
# D3 prefill/decode interference.
D3_MIN_DUR_RATIO = 3.0
D3_MIN_TRANSITIONS = 2
# D4 top-k / sort.
D4_MIN_SHARE = 0.015
# D5 permute.
D5_MIN_DECODE_SHARE = 0.05
D5_MIN_PREFILL_SHARE = 0.10
D5_MAX_ONEHOT_BATCH = 8192
# D6 imbalance.
D6_MIN_MAX_OVER_MEAN = 1.25
D6_MIN_BARRIER_RATIO = 1.3
D6_MIN_EXCESS_SHARE = 0.03
# D7 collectives.
D7_MIN_AR_SLICE_SHARE = 0.02
D7_MIN_MOE_AG_SHARE = 0.03
D7_MIN_COLLECTIVE_SHARE = 0.25
# Decode all-gathers at least this big gather weights, not activations
# (16M elements = 32 MB in bf16; decode activations are batch x hidden).
D7_MIN_WEIGHT_AG_ELEMS = du.MIN_WEIGHT_AG_ELEMS
D7_MIN_WEIGHT_AG_SHARE = 0.10
# D8 KV-head padding.
D8_MIN_ATTN_SHARE = 0.10
D8_MAX_GAIN = 0.45  # Docs: 1.64-1.8x RPA speedup.
# D9 rope.
D9_MIN_SHARE = 0.005
# D12 KV quantization.
D12_MIN_ATTN_SHARE = 0.20
# D13 kernel follow-ups.
D13_MIN_SHARE = 0.02
# D14 hygiene.
D14_MIN_STEPS = 5
D14_STOP_TRACE_NS = 100_000_000
# D15 fused EP MoE (vLLM-TorchTPU).
D15_MIN_GMM_SHARE = 0.02
D15_MIN_COLLECTIVE_SHARE = 0.05
D15_GAIN_FRACTION = 0.6
# D16 sparse-MLA layout.
D16_MIN_SHARE = 0.05
# D17 data formatting / KV writes.
D17_MIN_SHARE = 0.10
D17_MIN_KV_WRITE_OPS = 5000
# D18 indexer.
D18_MIN_SHARE = 0.08
# Small decode batches where halving the decode engine's DCP from 16 to 8
# halves the DCP top-k / KV merge fan-in. Only the decode engine of a
# disaggregated pair (`kv_role` kv_consumer) is lowered: symmetric DCP 8 on
# both legs halved the prefill prefix cache.
_D18_SMALL_DECODE_SEQS = limits.SMALL_DECODE_MAX_SEQS
_D18_SMALL_DECODE_BUCKET = limits.SMALL_DECODE_MAX_BUCKET
# D19 in-step waits. Below D19_MIN_COVERAGE attributed op time, op events are
# probably missing and D14 reports it instead.
D19_MIN_COVERAGE = 0.5
D19_MIN_WAIT_SHARE = 0.2
# D20 MLA attention TP.
D20_MIN_ATTN_SHARE = 0.05
# Assumed attention share when op coverage is too low to measure it
# (conservative; GLM-5.2 TP16->DP16 was a 7.3x throughput gain).
D20_CONFIG_ONLY_EST = 0.3
# Default per-rank KV budget (36 GB) when --kv-cache-memory-bytes is unset.
_D20_DEFAULT_KV_BUDGET_BYTES = limits.DEFAULT_KV_BUDGET_BYTES
# MLA KV latent + RoPE dim fallback (512 + 64 = 576 for DeepSeek-V3 / GLM-5.2).
_D20_DEFAULT_MLA_LATENT_DIM = limits.DEFAULT_MLA_LATENT_DIM
# D21 sequential loops.
D21_MIN_OPS_PER_STEP = 2000
D21_MIN_SHARE = 0.10
# Loops with at least this share of collective / expert-GMM time are
# model-level loops (generate loop, scan over layers), not recurrences.
D21_MAX_MODEL_LOOP_SHARE = 0.05
# D22 MoE without GMM.
D22_MAX_GMM_SHARE = 0.005
D22_MIN_DENSE_SHARE = 0.3
# D24 MoE TP without EP.
D24_MIN_MOE_SHARE = 0.15
D24_MAX_GAIN = 0.12
_FUSED_EP_OP = du.FUSED_EP_OP
_KV_WRITE_OP = r'dynamic-update-slice|scatter'
_DOC_GLM_HILLCLIMB = 'GLM-5.2 FP8 serving hill-climb on TPU v7x'
_DOC_GLM_BASELINE = 'GLM-5.2 FP8 serving baseline on TPU v7x'
_DOC_QWEN_CODER = 'Qwen3-Coder-480B serving study'
_DOC_Q32_STRIKE = 'Qwen3-32B serving study on TPU v7x'

_PREFILL_LIKE = du.PREFILL_LIKE

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
# Shared with catalog_detectors and headroom_detectors through detector_utils;
# the private names keep this module's call sites unchanged.
_ms = du.ms
_us = du.us
_pct = du.pct
_n_elems = du.n_elems
_framework = du.framework
_prefill_heavy = du.is_prefill_heavy


def _is_true(value) -> bool:
    """Whether a knob value read from a serving config means on."""
    return str(value or '').strip().lower() in ('1', 'true', 'yes', 'on')


def _cfg(ctx: context_lib.ServingContext) -> dict[str, str] | None:
    return ctx.serving_config


_stat = du.stat
xprof_link = du.xprof_link
_step_link = du.step_link
_active_phases = du.active_phases
_step_evidence = du.step_evidence
_top_op_evidence = du.top_op_evidence
_kv_cache_dtype = du.kv_cache_dtype
_kv_dequant_outside_kernel = du.kv_dequant_outside_kernel
_phase_ops = du.phase_ops
_n_steps = du.n_steps
_share = du.share
_base_op_evidence = du.base_op_evidence
_name_mask = du.name_mask
_is_torchtpu = du.is_torchtpu
_is_vllm = du.is_vllm
_is_copy_program = du.is_copy_program


def _knob_ref(key: str) -> str:
    k = knobs.get(key)
    return f'`{k.name}` ({k.source_path})'


def kv_packing(kv_dtype: str) -> int:
    """KV-cache rows per packed word: 32 // bits (RPA v3 get_dtype_packing).

    2 for bf16 / fp16 (and unknown), 4 for an 8-bit cache (fp8, int8).
    """
    return 4 if re.search(r'f8|fp8|int8|\bi8|\bs8', kv_dtype.lower()) else 2


# -----------------------------------------------------------------------------
# D1: host gap / host-bound decode
# -----------------------------------------------------------------------------
def d1_host_gap(f: features_lib.Features, ctx: context_lib.ServingContext,
                sid: str) -> list[Finding]:
    out = []
    fw = _framework(ctx, f)
    for phase in _active_phases(f):
        idle_frac = _stat(f, phase, 'idle_frac')
        mean_idle = _stat(f, phase, 'mean_idle_ns')
        if not (idle_frac >= D1_MIN_IDLE_FRAC
                and mean_idle >= D1_MIN_MEAN_IDLE_NS):
            continue
        ev = [
            Evidence(
                'device idle between steps',
                _pct(idle_frac),
                '% of cycle',
                detail=f'mean {_us(mean_idle):.0f} µs idle per step',
                xprof_link=_step_link(sid, f, phase),
            ),
            _step_evidence(f, phase, sid),
        ]
        hg = f.host_gap.get(phase, pd.DataFrame())
        for _, r in hg.head(3).iterrows():
            ev.append(
                Evidence(
                    f'host `{r["name"][:80]}` [{r["thread_name"]}] during idle',
                    _pct(r['overlap_frac_of_idle']),
                    '% of idle',
                ))
        changes = []
        fused = segmenter.CONTINUE_DECODE_FAMILY in f.seg.step_families
        if fw == 'vllm_tpu' and phase == segmenter.DECODE:
            if not fused:
                changes.append(
                    knobs.make_change(
                        'vllm_tpu.additional_config.enable_continue_decode',
                        'true',
                        _cfg(ctx),
                    ))
                changes.append(
                    knobs.make_change(
                        'vllm_tpu.additional_config.max_decode_steps', '10',
                        _cfg(ctx)))
            else:
                changes.append(
                    knobs.make_change(
                        'vllm_tpu.env.CONTINUE_DECODE_EOS_CHECK_INTERVAL',
                        '4',
                        _cfg(ctx),
                    ))
        out.append(
            Finding(
                detector_id='D1_host_gap',
                title='Device idles between steps (host-bound)',
                phase=phase,
                summary=
                (f'The TPU is idle {_pct(idle_frac):.1f}% of each {phase} step'
                 f' cycle (~{_us(mean_idle):.0f} µs/step) waiting on the host'
                 ' (scheduling, input prep, dispatch). ' +
                 ('Fusing multiple decode steps on device amortizes host '
                  'overhead.' if changes else
                  ('Attribute the host work with the host tracer and'
                   ' remove per-step Python overhead.'))),
                evidence=ev,
                knob_changes=changes,
                risk=F.Risk.LATENCY_TRADEOFF if changes else F.Risk.DIAGNOSTIC,
                confidence=F.Confidence.MEDIUM
                if len(hg) else F.Confidence.LOW,
                est_step_fraction_saved=float(idle_frac),
                validation=
                ('A/B at the same ISL/OSL/concurrency; expect higher output'
                 ' tok/s and lower TPOT. Re-profile and confirm idle_frac drops.'
                 ' New-request admission happens every max_decode_steps, so'
                 ' check TTFT.'),
                kernel_followup=None if changes else
                ('Profile with PYTHON_TRACER_LEVEL>=1 and check `check_host_'
                 'boundness` (tpu-diagnostics).'),
            ))
    return out


# -----------------------------------------------------------------------------
# D2: padding-bucket waste
# -----------------------------------------------------------------------------
def _next_pow2(n: float) -> int:
    return 1 << max(0, math.ceil(math.log2(max(n, 1))))


def _speculative_tokens(ctx: context_lib.ServingContext) -> int:
    """Draft tokens per request per decode step set by context or command."""
    if ctx.num_speculative_tokens:
        return int(ctx.num_speculative_tokens)
    for k, v in (ctx.serving_config or {}).items():
        key = k.replace('-', '_').lower()
        if key.rsplit('.', 1)[-1] == 'num_speculative_tokens':
            return int(v) if str(v).isdigit() else 0
        if 'speculative_config' in key:
            m = _SPEC_TOKENS.search(str(v))
            if m:
                return int(m.group(1))
    return 0


# vLLM-TorchTPU compiles token buckets doubling from 16 up to the batch-token
# budget (platforms/tpu_platform.py `_get_token_paddings`); the server log
# prints the list as "Using token paddings: [...]".
D2_TORCHTPU_MIN_BUCKET = 16
D2_LATENCY_BOUND_MAX_BUCKET = 64
_TOKEN_PADDINGS_RE = re.compile(r'Using token paddings: \[([0-9, ]+)\]')
_SPEC_TOKENS = re.compile(r'num_speculative_tokens[\'"]?\s*[:=]\s*(\d+)')


def _torchtpu_token_paddings(ctx: context_lib.ServingContext) -> list[int]:
    m = _TOKEN_PADDINGS_RE.search(ctx.server_log or '')
    if m:
        return sorted(int(v) for v in m.group(1).split(',') if v.strip())
    top = ctx.max_num_batched_tokens or 8192
    out, n = [], D2_TORCHTPU_MIN_BUCKET
    while n < top:
        out.append(n)
        n *= 2
    return out + [top]


def _d2_points_padding(f: features_lib.Features,
                       ctx: context_lib.ServingContext,
                       sid: str) -> list[Finding]:
    """Decode padding at every benchmarked load, from the bucket ladder.

  A profile is taken at one load (often the peak, where buckets are full).
  A decode step carries one token per running sequence on each DP replica,
  padded to the next bucket, so the ladder gives the padding at loads the
  profile did not see. Padded tokens route through the MoE experts like real
  ones unless TPU_MOE_SKIP_PADDED_TOKENS is set.
  """
    phase = segmenter.DECODE
    moe_share = features_lib.share_of(f.class_time, phase, oc.MOE_GMM)
    if not moe_share or moe_share != moe_share or moe_share < D2_MIN_MOE_SHARE:
        return []
    ladder = _torchtpu_token_paddings(ctx)
    dp = ctx.dp or 1
    per_req = 1 + _speculative_tokens(ctx)
    padded = []
    for c in ctx.benchmark_concurrencies:
        live = -(-c // dp) * per_req
        bucket = next((b for b in ladder if b >= live), None)
        if not bucket:
            continue
        pad = 1.0 - live / bucket
        if pad >= D2_MIN_PAD_RATIO:
            padded.append((c, live, bucket, pad))
    if not padded:
        return []
    worst = max(p for _, _, _, p in padded)
    loads = ', '.join(f'{c} ({_pct(p):.0f}%)' for c, _, _, p in padded)
    return [
        Finding(
            detector_id='D2_padding_waste',
            title='Token-bucket padding waste at low load',
            phase=phase,
            summary=
            (f'At {len(padded)} benchmarked load(s) a decode step pads its'
             f' live tokens to a bucket at least {_pct(D2_MIN_PAD_RATIO):.0f}%'
             f' larger (concurrency, padding: {loads}); buckets start at'
             f' {ladder[0]} and double. Padded tokens route through the'
             f' experts like real ones; MoE expert matmuls are'
             f' {_pct(moe_share):.1f}% of the profiled decode step.'),
            evidence=[
                Evidence('padded loads (concurrency: padding)', loads),
                Evidence('token buckets', ', '.join(map(str, ladder[:8]))),
                Evidence('MoE expert matmul share (decode)', _pct(moe_share),
                         '%'),
                _step_evidence(f, phase, sid),
            ],
            knob_changes=[
                knobs.make_change(
                    'vllm_torchtpu.env.TPU_MOE_SKIP_PADDED_TOKENS', '1',
                    _cfg(ctx))
            ],
            risk=F.Risk.PERF_ONLY,
            confidence=F.Confidence.MEDIUM,
            est_step_fraction_saved=worst * moe_share,
            validation=('Measure at the padded loads and at the peak: the gain'
                        ' is at low load, and the peak should not move.'),
            source_refs=[_DOC_QWEN3_CODER],
        )
    ]


def d2_padding(f: features_lib.Features, ctx: context_lib.ServingContext,
               sid: str) -> list[Finding]:
    out = []
    fw = _framework(ctx, f)
    if fw not in ('vllm_tpu', 'vllm_torchtpu'):
        return out
    st = f.seg.steps.dropna(subset=['bucket'])
    for phase in _active_phases(f):
        g = st[st['phase'] == phase]
        live = g['toks']
        inferred = live.notna().sum() == 0
        if inferred:
            if phase == segmenter.DECODE and ctx.concurrency:
                per_request = 1 + _speculative_tokens(ctx)
                live = pd.Series(
                    np.minimum(ctx.concurrency, ctx.max_num_seqs
                               or ctx.concurrency) * per_request,
                    index=g.index,
                    dtype=float,
                )
            else:
                continue
        ok = live.notna() & (g['bucket'].astype(float) > 0)
        if not ok.any():
            continue
        bucket = g.loc[ok, 'bucket'].astype(float)
        pad_ratio = float(((bucket - live[ok]).clip(lower=0) / bucket).mean())
        if pad_ratio < D2_MIN_PAD_RATIO:
            continue
        classes = [oc.MOE_GMM, oc.MATMUL]
        if float(bucket.median()) > D2_LATENCY_BOUND_MAX_BUCKET:
            classes.append(oc.COLLECTIVE)
        token_share = features_lib.share_of(f.class_time, phase, *classes)
        min_bucket_used = int(bucket.min())
        med_live = float(live[ok].median())
        changes = []
        if fw == 'vllm_tpu':
            cur_min = knobs.current_value('vllm_tpu.env.MIN_TOKEN_BUCKET',
                                          _cfg(ctx))
            min_default = int(cur_min) if cur_min and cur_min.isdigit() else 16
            if min_bucket_used <= min_default and med_live < min_bucket_used:
                changes.append(
                    knobs.make_change(
                        'vllm_tpu.env.MIN_TOKEN_BUCKET',
                        str(max(4, _next_pow2(med_live))),
                        _cfg(ctx),
                    ))
            elif min_bucket_used > 256:
                changes.append(
                    knobs.make_change(
                        'vllm_tpu.env.VLLM_TPU_BUCKET_PADDING_GAP', '128',
                        _cfg(ctx)))
            attn_dp = knobs.current_value(
                'vllm_tpu.additional_config.enable_dp_attention', _cfg(ctx))
            if ctx.model.is_moe and not _is_true(attn_dp) and (ctx.dp
                                                               or 1) == 1:
                changes.append(
                    knobs.make_change(
                        'vllm_tpu.env.MOE_ROUTE_PADDING_TO_EXPERT0', 'true',
                        _cfg(ctx)))
        elif fw == 'vllm_torchtpu':
            step = du.knob_int(ctx, 'vllm_torchtpu.env.TPU_TOKEN_BUCKET_STEP',
                               0)
            if min_bucket_used > 256 and not 0 < step < 128:
                changes.append(
                    knobs.make_change(
                        'vllm_torchtpu.env.TPU_TOKEN_BUCKET_STEP', '128',
                        _cfg(ctx)))
            if ctx.model.is_moe:
                changes.append(
                    knobs.make_change(
                        'vllm_torchtpu.env.TPU_MOE_SKIP_PADDED_TOKENS', '1',
                        _cfg(ctx)))
        note = ''
        if inferred:
            note = (
                ' Live tokens are inferred from the benchmark concurrency'
                ' (times 1 + draft tokens when speculative decoding is set);'
                ' with MTP or another drafter not shown in the command, the'
                ' padding is smaller than this.')
        out.append(
            Finding(
                detector_id='D2_padding_waste',
                title='Token-bucket padding waste',
                phase=phase,
                summary=
                (f'{phase} steps run padded token buckets'
                 f' ~{_pct(pad_ratio):.0f}% larger than the live token count'
                 f' (median live {med_live:.0f}, smallest bucket'
                 f' {min_bucket_used}). Tighter buckets and routing pad tokens'
                 ' to a single expert cut wasted compute.' + note),
                evidence=[
                    Evidence('mean padding ratio', _pct(pad_ratio), '%'),
                    Evidence(
                        'buckets used',
                        ', '.join(
                            str(int(b)) for b in sorted(bucket.unique())[:8]),
                    ),
                    Evidence(
                        'token-proportional share ({})'.format(
                            ' + '.join(classes)),
                        _pct(token_share),
                        '%',
                    ),
                    _step_evidence(f, phase, sid),
                ],
                knob_changes=changes,
                risk=F.Risk.PERF_ONLY,
                confidence=F.Confidence.MEDIUM
                if g['toks'].notna().any() else F.Confidence.LOW,
                est_step_fraction_saved=pad_ratio * token_share,
                validation=(
                    'A/B at low and high concurrency (docs: +7% at c4, +2.6% '
                    'at c256). Smaller buckets add compiled variants '
                    '(longer warmup).'),
                source_refs=[_DOC_QWEN3_CODER],
            ))
    if (fw == 'vllm_torchtpu' and ctx.model.is_moe
            and ctx.benchmark_concurrencies
            and not any(kc.name == 'TPU_MOE_SKIP_PADDED_TOKENS' for x in out
                        for kc in x.knob_changes)):
        out += _d2_points_padding(f, ctx, sid)
    return out


# -----------------------------------------------------------------------------
# D3: prefill / decode interference
# -----------------------------------------------------------------------------
def d3_interference(f: features_lib.Features, ctx: context_lib.ServingContext,
                    sid: str) -> list[Finding]:
    st = f.seg.steps
    if st.empty:
        return []
    first = st.iloc[0]
    dev = st[(st['host'] == first['host']) & (st['device'] == first['device'])]
    is_dec = dev['phase'] == segmenter.DECODE
    is_pre = dev['phase'].isin(_PREFILL_LIKE)
    if is_dec.sum() < 2 or is_pre.sum() < 1:
        return []
    transitions = int((is_dec.astype(int).diff().abs() > 0).sum())
    dec_mean = float(dev.loc[is_dec, 'dur_ns'].mean())
    pre_mean = float(dev.loc[is_pre, 'dur_ns'].mean())
    ratio = pre_mean / dec_mean if dec_mean else 0.0
    if transitions < D3_MIN_TRANSITIONS or ratio < D3_MIN_DUR_RATIO:
        return []
    total = float(dev['dur_ns'].sum())
    stall = float(is_pre.sum()) * (pre_mean - dec_mean)
    stall_frac = stall / total if total else 0.0
    changes = []
    mbt = ctx.max_num_batched_tokens
    chunk_target = str(max(512, (mbt // 2) if mbt else 2048))
    budget_note = ''
    if _framework(ctx, f) == 'vllm_tpu':
        # A token budget below the prompt length splits every prefill across
        # steps and raises TTFT (Gemma-3 8K inputs: +67%), so stay at or above
        # the benchmark input length and only suggest an actual reduction.
        chunk, prompt = int(chunk_target), ctx.input_len or 0
        budget = max(chunk, prompt)
        # With the budget unknown, keep the default suggestion; never raise it.
        lowers = budget < mbt if mbt else budget == chunk
        if lowers:
            changes.append(
                knobs.make_change(
                    'vllm_tpu.cli.max_num_batched_tokens',
                    str(budget),
                    _cfg(ctx),
                ))
        elif prompt > chunk:
            budget_note = (
                ' --max-num-batched-tokens stays: a budget under the'
                f' {prompt}-token prompt would split every prefill and raise'
                ' TTFT.')
        if (ctx.dp or 1) > 1:
            changes.append(
                knobs.make_change('vllm_tpu.env.DP_SCHED_BATCH_PREFILL',
                                  'true', _cfg(ctx)))
    elif _framework(ctx, f) == 'maxtext':
        changes.append(
            knobs.make_change('maxtext.yml.use_chunked_prefill', 'true',
                              _cfg(ctx)))
        changes.append(
            knobs.make_change('maxtext.yml.prefill_chunk_size', chunk_target,
                              _cfg(ctx)))
    pre_phase = dev.loc[is_pre, 'phase'].mode().iloc[0]
    return [
        Finding(
            detector_id='D3_prefill_decode_interference',
            title='Prefill steps stall in-flight decodes',
            phase='mixed',
            summary=
            (f'{int(is_pre.sum())} {pre_phase} steps ({_ms(pre_mean):.1f} ms,'
             f' {ratio:.1f}× a decode step) are interleaved with decode steps'
             f' ({transitions} phase switches). Every in-flight request waits'
             f' for them, inflating TPOT by ~{_pct(stall_frac):.0f}% of'
             ' profiled step time. Smaller prefill chunks (and batching'
             ' prefills across DP ranks) bound the stall.' + budget_note),
            evidence=[
                Evidence(
                    'decode step mean',
                    _ms(dec_mean),
                    ' ms',
                    xprof_link=_step_link(sid, f, segmenter.DECODE),
                ),
                Evidence(
                    f'{pre_phase} step mean',
                    _ms(pre_mean),
                    ' ms',
                    xprof_link=_step_link(sid, f, pre_phase),
                ),
                Evidence('phase switches (first device)', transitions),
                Evidence('decode stall', _pct(stall_frac), '% of step time'),
            ],
            knob_changes=changes,
            risk=F.Risk.LATENCY_TRADEOFF if changes else F.Risk.DIAGNOSTIC,
            confidence=F.Confidence.MEDIUM,
            est_step_fraction_saved=min(0.3, 0.5 * stall_frac),
            validation=
            ('Sweep max-num-batched-tokens at target concurrency; watch '
             'TTFT (rises) vs TPOT/throughput (docs: +14-25% at '
             'c64-c256 with DP prefill batching). A/B DP_SCHED_BATCH_PREFILL '
             'on vs off: the DeepSeek-V3 best command set it to 0.'),
            source_refs=[_DOC_QWEN3_CODER],
        )
    ]


# -----------------------------------------------------------------------------
# D4: MoE top-k / sort hotspot
# -----------------------------------------------------------------------------
def d4_topk(f: features_lib.Features, ctx: context_lib.ServingContext,
            sid: str) -> list[Finding]:
    out = []
    so = f.step_ops
    if so.empty:
        return out
    fw = _framework(ctx, f)
    for phase in _active_phases(f):
        share = features_lib.share_of(f.class_time, phase, oc.MOE_TOPK_SORT)
        if share < D4_MIN_SHARE:
            continue
        g = so[(so['phase'] == phase) & (so['op_class'] == oc.MOE_TOPK_SORT)]
        n_steps = int((f.seg.steps['phase'] == phase).sum())
        dur = _stat(f, phase, 'mean_dur_ns')
        topk_ns = g.loc[g['tf_op_type'].isin(['top_k', 'approx_top_k']),
                        'self_ns'].sum() / max(n_steps, 1)
        topk_share = topk_ns / dur if dur else 0.0
        changes = []
        if fw == 'vllm_tpu' and topk_share >= D4_MIN_SHARE / 2:
            changes.append(
                knobs.make_change('vllm_tpu.env.MOE_APPROX_TOPK', 'true',
                                  _cfg(ctx)))
        elif (fw == 'vllm_torchtpu' and topk_share >= D4_MIN_SHARE / 2
              and ctx.serving_config is not None and str(
                  knobs.current_value('vllm_torchtpu.env.TPU_MOE_ROUTER_TOPK',
                                      _cfg(ctx)) or '').lower() == 'sort'):
            changes.append(
                knobs.make_change('vllm_torchtpu.env.TPU_MOE_ROUTER_TOPK',
                                  'rowmax', _cfg(ctx)))
        elif fw == 'maxtext':
            changes.append(
                knobs.make_change('maxtext.yml.use_ragged_sort', 'true',
                                  _cfg(ctx)))
        out.append(
            Finding(
                detector_id='D4_moe_topk_sort',
                title='MoE router top-k / sort hotspot',
                phase=phase,
                summary=
                (f'Expert-selection top-k and sort take {_pct(share):.1f}% of'
                 f' the {phase} step (top-k alone {_pct(topk_share):.1f}%).'
                 ' Approximate top-k replaces the exact sort-based selection.'
                 ),
                evidence=[
                    Evidence('top-k + sort share', _pct(share), '%'),
                    Evidence('top-k share', _pct(topk_share), '%'),
                    *_top_op_evidence(f, phase, oc.MOE_TOPK_SORT),
                    _step_evidence(f, phase, sid),
                ],
                knob_changes=changes,
                risk=(F.Risk.PERF_ONLY if fw == 'vllm_torchtpu' and changes
                      else F.Risk.QUALITY_GATED),
                confidence=F.Confidence.HIGH if changes else F.Confidence.LOW,
                est_step_fraction_saved=0.8 *
                (topk_share if fw in ('vllm_tpu', 'vllm_torchtpu') else share),
                validation=(
                    "Run accuracy evals (e.g. the model's standard eval "
                    'suite) at MOE_APPROX_TOPK_RECALL_TARGET=0.9 and 0.95; '
                    'docs: 559 µs -> ~50 µs, +2-3% e2e.'),
                source_refs=[_DOC_QWEN3_CODER],
            ))
    return out


# -----------------------------------------------------------------------------
# D5: MoE permute / unpermute (gather/scatter)
# -----------------------------------------------------------------------------
def d5_permute(f: features_lib.Features, ctx: context_lib.ServingContext,
               sid: str) -> list[Finding]:
    out = []
    fw = _framework(ctx, f)
    topk = ctx.model.num_experts_per_tok or 1
    for phase in _active_phases(f):
        share = features_lib.share_of(f.class_time, phase, oc.MOE_PERMUTE)
        is_dec = phase == segmenter.DECODE
        if share < (D5_MIN_DECODE_SHARE if is_dec else D5_MIN_PREFILL_SHARE):
            continue
        buckets = f.seg.steps.loc[f.seg.steps['phase'] == phase,
                                  'bucket'].dropna()
        max_bucket = int(buckets.max()) if len(buckets) else 0
        batch = max_bucket * topk
        changes, follow = [], None
        if fw == 'vllm_tpu' and is_dec and 0 < batch <= D5_MAX_ONEHOT_BATCH:
            changes.append(
                knobs.make_change('vllm_tpu.env.ONEHOT_MOE_PERMUTE_THRESHOLD',
                                  str(batch), _cfg(ctx)))
        elif (fw == 'vllm_torchtpu' and ctx.serving_config is not None
              and is_dec and 0 < max_bucket <= 64):
            changes.append(
                knobs.make_change(
                    'vllm_torchtpu.env.TPU_MOE_LOWTOK_MAX_TOKENS',
                    str(max_bucket), _cfg(ctx)))
        else:
            follow = (
                'Permute/unpermute gathers are slow at this batch size. Check the '
                'SparseCore gather for HBM page thrashing (sorted/ragged gather) '
                'and file a kernel bug with the linked ops; per the Qwen3-Coder '
                'doc, a SparseCore ragged gather fixed this for prefill.')
        out.append(
            Finding(
                detector_id='D5_moe_permute',
                title='MoE token permute/unpermute is expensive',
                phase=phase,
                summary=
                (f'MoE dispatch gather / combine scatter take {_pct(share):.1f}%'
                 f' of the {phase} step (max bucket {max_bucket} tokens ×'
                 f' top-{topk}). ' +
                 ('At small batch, one-hot + matmul permute is faster than '
                  'gather/scatter.' if changes else '')),
                evidence=[
                    Evidence('permute share', _pct(share), '%'),
                    *_top_op_evidence(f, phase, oc.MOE_PERMUTE),
                    _step_evidence(f, phase, sid),
                ],
                knob_changes=changes,
                risk=F.Risk.PERF_ONLY if changes else F.Risk.DIAGNOSTIC,
                confidence=F.Confidence.MEDIUM
                if changes else F.Confidence.LOW,
                est_step_fraction_saved=0.6 * share,
                validation=
                ('Threshold is compared to the per-shard permute batch (tokens ×'
                 ' top_k; divide tokens by attention-DP size). A/B decode TPOT;'
                 ' docs: +4-14% decode.'
                 if changes else 'Compare per-op time after a kernel change.'),
                kernel_followup=follow,
                source_refs=[_DOC_QWEN3_CODER],
            ))
    return out


# -----------------------------------------------------------------------------
# D6: expert imbalance (cross-device + barrier wait)
# -----------------------------------------------------------------------------
def d6_imbalance(f: features_lib.Features, ctx: context_lib.ServingContext,
                 sid: str) -> list[Finding]:
    if f.step_ops.empty:
        return []
    out = []
    coll = f.collectives
    for phase in _active_phases(f):
        imb = features_lib.cross_device_imbalance(f.step_ops, phase,
                                                  oc.MOE_GMM)
        bar = features_lib.moe_barrier_excess(coll, phase)
        dur = _stat(f, phase, 'mean_dur_ns')
        excess = float(bar['excess_ns_per_step'].sum()) if len(bar) else 0.0
        excess_share = excess / dur if dur else 0.0
        ratio = imb.get('weighted_max_over_mean', 1.0)
        bar_ratio = float(bar['ratio'].max()) if len(bar) else 1.0
        fire = ratio >= D6_MIN_MAX_OVER_MEAN or (
            bar_ratio >= D6_MIN_BARRIER_RATIO
            and excess_share >= D6_MIN_EXCESS_SHARE)
        if not fire:
            continue
        ev = []
        if imb:
            ev.append(
                Evidence(
                    'GMM max/mean across devices (per occurrence)',
                    ratio,
                    '×',
                    detail=
                    (f'{imb["n_devices"]} devices; slowest {imb["slowest_device"]}'
                     ),
                ))
        for _, r in bar.iterrows():
            ev.append(
                Evidence(
                    f'{r["collective"]} {r["out_dtype"]}[{r["dims"]}] in MoE vs '
                    f'{r["ref_scopes"] or "non-MoE"}',
                    r['ratio'],
                    '×',
                    detail=(f'{_us(r["moe_mean_ns"]):.0f} vs '
                            f'{_us(r["ref_median_ns"]):.0f} µs; excess '
                            f'{_ms(r["excess_ns_per_step"]):.2f} ms/step'),
                ))
        ev.append(_step_evidence(f, phase, sid))
        out.append(
            Finding(
                detector_id='D6_expert_imbalance',
                title='Expert load imbalance (barrier wait after MoE)',
                phase=phase,
                summary=
                ('Under expert parallelism the slowest shard gates every MoE'
                 f' layer: GMM time varies {ratio:.2f}× across devices and the'
                 f' collective after MoE is {bar_ratio:.2f}× slower than the'
                 f' same-shape collective elsewhere (~{_pct(excess_share):.1f}%'
                 ' of step is barrier wait). Lower EP degree / hybrid EP×TP'
                 ' reduces the variance.'),
                evidence=ev,
                knob_changes=[],
                risk=F.Risk.DIAGNOSTIC,
                confidence=F.Confidence.MEDIUM,
                est_step_fraction_saved=excess_share,
                validation=
                ('Measure the upper bound with'
                 f' {_knob_ref("vllm_tpu.env.FORCE_MOE_RANDOM_ROUTING")}=1'
                 ' (benchmark only, changes outputs). Docs: EP8 ratio 1.4-1.8,'
                 ' EP2 1.1.'),
                kernel_followup=
                ('Hand off to the mesh step (oss_meshes): evaluate lower EP / EP×TP'
                 ' hybrid meshes for the MoE layers.'),
                source_refs=[_DOC_QWEN3_CODER],
            ))
    return out


# -----------------------------------------------------------------------------
# D7: exposed / mis-lowered collectives
# -----------------------------------------------------------------------------
def d7_collectives(f: features_lib.Features, ctx: context_lib.ServingContext,
                   sid: str) -> list[Finding]:
    if f.step_ops.empty:
        return []
    out = []
    fw = _framework(ctx, f)
    ars = features_lib.ar_then_slice(f.step_ops, f.meta, f.seg.steps)
    coll = f.collectives
    moe_ops = set(
        zip(
            f.meta.loc[f.meta['moe_scope'], 'module'],
            f.meta.loc[f.meta['moe_scope'], 'op'],
        ))
    for phase in _active_phases(f):
        dur = _stat(f, phase, 'mean_dur_ns')
        if not dur:
            continue
        # (a) all-reduce + slice == reduce-scatter lowered as all-reduce.
        if len(ars):
            a = ars[ars['phase'] == phase]
            share = float(a['ns_per_step'].sum()) / dur
            if share >= D7_MIN_AR_SLICE_SHARE:
                in_moe = any(
                    (m, o) in moe_ops for m, o in zip(a['module'], a['op']))
                changes = []
                if fw == 'vllm_tpu' and in_moe:
                    changes.append(
                        knobs.make_change('vllm_tpu.env.ENABLE_RS_KERNEL',
                                          'true', _cfg(ctx)))
                out.append(
                    Finding(
                        detector_id='D7_ar_plus_slice',
                        title='Reduce-scatter lowered as all-reduce + slice',
                        phase=phase,
                        summary=
                        (f'{int(a["count"].sum())} all-reduces are immediately'
                         f' sliced, costing {_pct(share):.1f}% of the {phase} step.'
                         ' A native reduce-scatter moves ~1/N of the bytes.'),
                        evidence=[
                            Evidence(
                                f'`{r["op"]}` -> `{r["next_op"]}`',
                                _us(r['ns_per_step']),
                                ' µs/step',
                            ) for _, r in
                            (a.sort_values('ns_per_step',
                                           ascending=False).head(3).iterrows())
                        ] + [_step_evidence(f, phase, sid)],
                        knob_changes=changes,
                        risk=F.Risk.PERF_ONLY,
                        confidence=F.Confidence.MEDIUM
                        if changes else F.Confidence.LOW,
                        est_step_fraction_saved=0.5 * share,
                        validation=
                        ('Re-profile; the all-reduce should become reduce-scatter.'
                         ),
                        kernel_followup=None if changes else
                        ('Use psum_scatter / reduce-scatter in the model code for'
                         ' this layer or file an XLA bug (collective pattern).'
                         ),
                    ))
        # (b) MoE all-gather in bf16 -> fp8.
        if fw in ('vllm_tpu', 'maxtext') and len(coll):
            ag = coll[(coll['phase'] == phase)
                      & coll['moe_scope']
                      & (coll['collective'] == 'all-gather')
                      & (coll['out_dtype'] == 'bf16')
                      & (coll['dims'].map(_n_elems) < D7_MIN_WEIGHT_AG_ELEMS)]
            share = float(ag['ns_per_step'].sum()) / dur
            if share >= D7_MIN_MOE_AG_SHARE:
                if fw == 'vllm_tpu':
                    ag_changes = [
                        knobs.make_change(
                            'vllm_tpu.env.MOE_ALL_GATHER_ACTIVATION_DTYPE',
                            'float8_e4m3fn',
                            _cfg(ctx),
                        )
                    ]
                else:
                    ag_changes = [
                        knobs.make_change(
                            'maxtext.yml.moe_quantize_token_all_gather',
                            'true',
                            _cfg(ctx),
                        )
                    ]
                out.append(
                    Finding(
                        detector_id='D7_moe_allgather_fp8',
                        title='MoE dispatch all-gather in bf16',
                        phase=phase,
                        summary=
                        (f'bf16 all-gathers inside MoE take {_pct(share):.1f}% of'
                         f' the {phase} step. Gathering activations in fp8 halves'
                         ' the bytes.'),
                        evidence=[
                            Evidence('MoE bf16 all-gather share', _pct(share),
                                     '%'),
                            _step_evidence(f, phase, sid),
                        ],
                        knob_changes=ag_changes,
                        risk=F.Risk.QUALITY_GATED,
                        confidence=F.Confidence.MEDIUM,
                        est_step_fraction_saved=0.5 * share,
                        validation=
                        ('Accuracy eval + A/B. Only applies to GMM_EP/GMM_TP '
                         'MoE backends.'),
                    ))
        # (d) Decode re-gathers weights every step (FSDP-style sharding).
        if phase == segmenter.DECODE and len(coll):
            ag = coll[(coll['phase'] == phase)
                      & (coll['collective'] == 'all-gather')]
            ag = ag[ag['dims'].map(_n_elems) >= D7_MIN_WEIGHT_AG_ELEMS]
            share = float(ag['ns_per_step'].sum()) / dur
            if share >= D7_MIN_WEIGHT_AG_SHARE:
                changes = []
                cfg = _cfg(ctx)
                fsdp = (knobs.current_value('maxtext.yml.ici_fsdp_parallelism',
                                            cfg) if fw == 'maxtext' else None)
                # FSDP already 1: another mesh axis shards these weights.
                fsdp_off = fsdp == '1'
                if fw == 'maxtext' and not fsdp_off:
                    size = (fsdp if fsdp and fsdp.isdigit() and int(fsdp) > 1
                            else '-1')
                    changes = [
                        knobs.make_change('maxtext.yml.ici_fsdp_parallelism',
                                          '1', cfg),
                        knobs.make_change(
                            'maxtext.yml.ici_expert_parallelism'
                            if ctx.model.num_experts else
                            'maxtext.yml.ici_tensor_parallelism',
                            size,
                            cfg,
                        ),
                    ]
                remedy = 'move the FSDP axis to tensor/expert parallelism.'
                if fsdp_off:
                    remedy = ('ici_fsdp_parallelism is already 1, so find the'
                              ' mesh axis that shards them.')
                top = ag.sort_values('ns_per_step', ascending=False).head(3)
                out.append(
                    Finding(
                        detector_id='D7_decode_weight_allgather',
                        title='Decode re-gathers weights every step',
                        phase=phase,
                        summary=
                        ('All-gathers with weight-sized outputs take'
                         f' {_pct(share):.1f}% of the decode step: the weights are'
                         ' sharded FSDP-style and rebuilt on every token. Serving'
                         ' should keep weights resident: ' + remedy),
                        evidence=[
                            Evidence(
                                f'all-gather {r["out_dtype"]}[{r["dims"]}] '
                                f'x{r["per_step"]:.0f}/step',
                                _us(r['ns_per_step']),
                                ' µs/step',
                            ) for _, r in top.iterrows()
                        ] + [_step_evidence(f, phase, sid)],
                        knob_changes=changes,
                        risk=F.Risk.PERF_ONLY,
                        confidence=F.Confidence.MEDIUM,
                        est_step_fraction_saved=0.8 * share,
                        validation=
                        ('Check HBM fits the TP/EP-sharded weights, then '
                         're-profile: the weight all-gathers should disappear.'
                         ),
                        kernel_followup=None if changes else
                        ('Hand off to the mesh step (oss_meshes) with a serving (no-FSDP) '
                         'mesh constraint.'),
                    ))
        # (c) Collective-dominated step: hand off.
        cshare = features_lib.share_of(f.class_time, phase, oc.COLLECTIVE)
        if cshare >= D7_MIN_COLLECTIVE_SHARE:
            out.append(
                Finding(
                    detector_id='D7_collective_bound',
                    title='Step is collective-bound',
                    phase=phase,
                    summary=
                    (f'Collectives are {_pct(cshare):.1f}% of the {phase} step '
                     '(exposed on the TensorCore timeline). Review the mesh / '
                     'collective overlap.'),
                    evidence=[
                        Evidence('collective share', _pct(cshare), '%'),
                        *_top_op_evidence(f, phase, oc.COLLECTIVE, 3),
                        _step_evidence(f, phase, sid),
                    ],
                    knob_changes=[],
                    risk=F.Risk.DIAGNOSTIC,
                    confidence=F.Confidence.LOW,
                    est_step_fraction_saved=0.0,
                    kernel_followup=
                    ('Hand off to the mesh step (oss_meshes) and XLA flag tuning'
                     ' (collective-matmul / async collective flags). Subtract the'
                     ' D6 barrier excess first: part of this may be imbalance'
                     ' wait.'),
                ))
    return out


# -----------------------------------------------------------------------------
# D8: KV-head padding / replication
# -----------------------------------------------------------------------------
def d8_kv_head_padding(f: features_lib.Features,
                       ctx: context_lib.ServingContext,
                       sid: str) -> list[Finding]:
    m = ctx.model
    if not m.num_kv_heads or m.is_mla or not _is_vllm(ctx, f):
        return []
    torchtpu = _is_torchtpu(ctx, f)
    if torchtpu:
        if not ctx.tp or ctx.tp <= 1:
            return []
    else:
        attn_dp = knobs.current_value(
            'vllm_tpu.additional_config.enable_dp_attention', _cfg(ctx))
        if _is_true(attn_dp):
            return []
    tp = ctx.tp
    tp_note = ''
    if not tp and not f.seg.steps.empty:
        tp = int(f.seg.steps.groupby(['host', 'device']).ngroups)
        tp_note = f' (TP assumed = {tp} devices; pass the serving config)'
    if not tp or tp < 1:
        return []
    kv_dtype = _kv_cache_dtype(f) or (ctx.config_get('kv-cache-dtype') or '')
    packing = kv_packing(kv_dtype)
    kv_per_dev = m.num_kv_heads / tp
    # The RPA KV cache interleaves K and V: a device stores two rows per KV
    # head (one for head_dim 64, where K and V share a row) and at least one
    # head (replicated when TP exceeds the KV heads), padded up to the dtype
    # packing (tpu_inference kernels/ragged_paged_attention/v3
    # get_kv_cache_shape).
    rows_per_head = 1 if m.head_dim == 64 else 2

    def _waste(heads_per_dev: float) -> float:
        rows = max(heads_per_dev, 1.0) * rows_per_head
        padded = math.ceil(rows / packing) * packing
        return 1.0 - rows / padded * min(heads_per_dev, 1.0)

    rows = max(kv_per_dev, 1.0) * rows_per_head
    padded = math.ceil(rows / packing) * packing
    waste = _waste(kv_per_dev)
    if waste <= 0:
        return []
    replicas = (f' (each head replicated on {1 / kv_per_dev:g} devices)'
                if kv_per_dev < 1 else '')
    target = ''
    if torchtpu:
        # The largest TP whose KV-cache rows fill the packing on each device,
        # with the freed devices as DP replicas.
        n = tp * (ctx.dp or 1)
        # TP 1 has no torch_tpu layout on tpu7x (oss_meshes'
        # torchtpu_tp1_no_layout), so it is never the target.
        fits = [
            d for d in range(2, n + 1)
            if n % d == 0 and _waste(m.num_kv_heads / d) <= 0
        ]
        new_tp = max(fits) if fits else None
        if new_tp is None:
            # Only TP 1 would keep whole KV heads (e.g. MQA): report the
            # padding, propose no layout.
            target = ' Only TP 1 keeps whole KV heads, which torch_tpu cannot serve.'
            changes = []
        else:
            target = f' TP {new_tp} x DP {n // new_tp} keeps whole KV heads.'
            changes = [
                knobs.make_change('vllm.cli.data_parallel_size',
                                  str(n // new_tp), _cfg(ctx)),
                knobs.make_change('vllm.cli.tensor_parallel_size', str(new_tp),
                                  _cfg(ctx)),
            ]
        if m.is_moe and changes:
            changes.append(
                knobs.make_change('vllm.cli.enable_expert_parallel', 'true',
                                  _cfg(ctx)))
    else:
        changes = [
            knobs.make_change('vllm_tpu.additional_config.enable_dp_attention',
                              'true', _cfg(ctx))
        ]
    out = []
    for phase in _active_phases(f):
        share = features_lib.share_of(f.class_time, phase, oc.ATTENTION_KERNEL)
        if share < D8_MIN_ATTN_SHARE:
            continue
        out.append(
            Finding(
                detector_id='D8_kv_head_padding',
                title='KV heads padded/replicated under TP attention',
                phase=phase,
                summary=
                (f'{m.num_kv_heads} KV heads over TP={tp}{tp_note} gives'
                 f' {kv_per_dev:g} head(s)/device{replicas}. With K and V'
                 f' interleaved that is {rows:g} KV-cache row(s) per device,'
                 f' padded to {padded} for the {kv_dtype or "KV"} packing of'
                 f' {packing}: {_pct(waste):.0f}% of the KV bytes attention'
                 ' reads are padding or replicas. Attention is'
                 f' {_pct(share):.1f}% of the {phase} step.' +
                 (target or ' Attention DP keeps whole KV heads per device.')),
                evidence=[
                    Evidence('attention kernel share', _pct(share), '%'),
                    Evidence('KV heads per device', kv_per_dev),
                    Evidence('KV-cache rows per device (K and V interleaved)',
                             rows,
                             detail=f'padded to {padded}'),
                    Evidence('KV packing factor', packing, detail=kv_dtype),
                    *_top_op_evidence(f, phase, oc.ATTENTION_KERNEL, 1),
                    _step_evidence(f, phase, sid),
                ],
                knob_changes=changes,
                risk=F.Risk.MEMORY if changes else F.Risk.DIAGNOSTIC,
                confidence=F.Confidence.HIGH
                if not tp_note else F.Confidence.MEDIUM,
                est_step_fraction_saved=share * min(waste, D8_MAX_GAIN),
                validation=(
                    'Co-propose with the mesh step (changes the attention'
                    ' mesh; set --max-num-batched-tokens per DP rank). Docs:'
                    ' 1.64-1.8× RPA, +6-7% e2e.'),
                source_refs=[_DOC_QWEN3_CODER],
            ))
    return out


# -----------------------------------------------------------------------------
# D9: rope cache layout copy
# -----------------------------------------------------------------------------
def d9_rope_cache(f: features_lib.Features, ctx: context_lib.ServingContext,
                  sid: str) -> list[Finding]:
    mp = ctx.model.max_position_embeddings
    fw = _framework(ctx, f)
    if not mp or fw not in ('vllm_tpu', 'vllm_torchtpu'):
        return []
    if ctx.max_model_len and ctx.max_model_len >= mp:
        return []
    hits = features_lib.ops_with_dim(f.step_ops, f.meta, mp, f.seg.steps)
    if hits.empty:
        return []
    out = []
    for phase in _active_phases(f):
        h = hits[hits['phase'] == phase]
        dur = _stat(f, phase, 'mean_dur_ns')
        share = float(h['ns_per_step'].sum()) / dur if dur else 0.0
        if share < D9_MIN_SHARE:
            continue
        if fw == 'vllm_torchtpu':
            trunc_chg = knobs.make_change(
                'vllm_torchtpu.env.TPU_ROPE_CACHE_TRUNCATE', '1', _cfg(ctx))
            if trunc_chg.already_set:
                changes = [
                    knobs.make_change(
                        'vllm_torchtpu.env.TPU_ROPE_CACHE_ROW_MAJOR', '1',
                        _cfg(ctx))
                ]
            else:
                changes = [trunc_chg]
        else:
            slice_chg = knobs.make_change('vllm_tpu.env.SLICE_ROPE_CACHE',
                                          'true', _cfg(ctx))
            if slice_chg.already_set:
                changes = [
                    knobs.make_change('vllm_tpu.env.ROPE_CACHE_ROW_MAJOR',
                                      'true', _cfg(ctx))
                ]
            else:
                changes = [slice_chg]
        out.append(
            Finding(
                detector_id='D9_rope_cache_copy',
                title='Per-step copy of the full RoPE cache',
                phase=phase,
                summary=
                (f'Ops touching a [{mp}, ...] table (max_position_embeddings)'
                 f' take {_pct(share):.2f}% of the {phase} step; slicing the'
                 ' cos/sin cache to max_model_len at load removes the per-step'
                 ' layout copy.'),
                evidence=[
                    Evidence(
                        f'`{r["op"]}` {r["out_dtype"]}{list(r["out_dims"])}',
                        _us(r['ns_per_step']),
                        ' µs/step',
                    ) for _, r in (h.sort_values(
                        'ns_per_step', ascending=False).head(3).iterrows())
                ] + [_step_evidence(f, phase, sid)],
                knob_changes=changes,
                risk=F.Risk.PERF_ONLY,
                confidence=F.Confidence.MEDIUM,
                est_step_fraction_saved=share,
                validation=(
                    'Text / 1-D RoPE only (ignored for MRoPE). If the copy'
                    ' remains, also try'
                    f' {_knob_ref("vllm_tpu.env.ROPE_CACHE_ROW_MAJOR")}.'),
            ))
    return out


# -----------------------------------------------------------------------------
# D11: recompilation during serving
# -----------------------------------------------------------------------------
def d11_recompile(f: features_lib.Features, ctx: context_lib.ServingContext,
                  sid: str) -> list[Finding]:
    ce = f.compile_events
    if ce.empty:
        return []
    total = float((ce['end_ns'] - ce['start_ns']).sum())
    changes = []
    if _framework(ctx, f) == 'vllm_tpu':
        changes = [
            knobs.make_change('vllm_tpu.env.SKIP_JAX_PRECOMPILE', 'false',
                              _cfg(ctx)),
            knobs.make_change('vllm_tpu.env.VLLM_XLA_CHECK_RECOMPILATION', '1',
                              _cfg(ctx)),
        ]
    first = ce.iloc[0]
    return [
        Finding(
            detector_id='D11_recompilation',
            title='XLA compilation inside the serving window',
            phase='all',
            summary=(
                f'{len(ce)} compile events ({_ms(total):.1f} ms) happen after '
                'serving started; each blocks the engine loop.'),
            evidence=[
                Evidence(
                    'compile events',
                    len(ce),
                    detail=str(first['name'])[:100],
                    xprof_link=xprof_link(sid, first['start_ns'],
                                          first['end_ns']),
                )
            ],
            knob_changes=changes,
            risk=F.Risk.DIAGNOSTIC,
            confidence=F.Confidence.MEDIUM,
            est_step_fraction_saved=0.0,
            validation=(
                'Warm up all buckets before profiling; confirm no runtime '
                'compiles with the recompilation check enabled.'),
        )
    ]


# -----------------------------------------------------------------------------
# D12: KV cache quantization headroom (decode)
# -----------------------------------------------------------------------------
def d12_kv_quant(f: features_lib.Features, ctx: context_lib.ServingContext,
                 sid: str) -> list[Finding]:
    if segmenter.DECODE not in f.seg.phases:
        return []
    if _kv_cache_dtype(f) != 'bf16':
        return []
    share = features_lib.share_of(f.class_time, segmenter.DECODE,
                                  oc.ATTENTION_KERNEL)
    if share < D12_MIN_ATTN_SHARE:
        return []
    dequant, why = _kv_dequant_outside_kernel(f, ctx)
    if dequant:
        return [
            Finding(
                detector_id='D12_kv_cache_quant',
                title=
                'Quantized KV cache dequantized outside the attention kernel',
                phase=segmenter.DECODE,
                summary=(
                    f'The KV cache is quantized ({why}), but attention is'
                    f' {_pct(share):.1f}% of the decode step on bf16 operands.'
                    ' Dequantizing outside the kernel adds a full KV read and'
                    ' write per step, so fp8 KV is slower here than bf16.'),
                evidence=[
                    Evidence('attention kernel share', _pct(share), '%'),
                    Evidence('dequant signal', 1, detail=why),
                    _step_evidence(f, segmenter.DECODE, sid),
                ],
                knob_changes=[],
                risk=F.Risk.DIAGNOSTIC,
                confidence=F.Confidence.MEDIUM,
                est_step_fraction_saved=0.0,
                validation=
                ('Compare decode step time against bf16 KV at the same batch,'
                 ' and the max batch each fits. Qwen3-32B on TPU v7x:'
                 ' fp8 KV with a bf16 convert raised bsz=32 decode from 18.88'
                 ' to 35.28 ms, though batch could grow to 240.'),
                kernel_followup=
                ('Use an attention kernel that reads fp8/int8 KV and'
                 ' dequantizes in VMEM (RPA v3 k_scale/v_scale), or keep bf16'
                 ' KV until one is available.'),
                source_refs=[_DOC_Q32_STRIKE],
            )
        ]
    fw = _framework(ctx, f)
    if fw == 'vllm_tpu':
        changes = [
            knobs.make_change('vllm_tpu.cli.kv_cache_dtype', 'fp8', _cfg(ctx))
        ]
    elif fw == 'maxtext':
        changes = [
            knobs.make_change('maxtext.yml.quantize_kvcache', 'true',
                              _cfg(ctx))
        ]
    else:
        return []
    return [
        Finding(
            detector_id='D12_kv_cache_quant',
            title='bf16 KV cache in a KV-bandwidth-bound decode',
            phase=segmenter.DECODE,
            summary=
            (f'Attention (bf16 KV) is {_pct(share):.1f}% of the decode step.'
             ' An fp8/int8 KV cache halves KV reads and doubles KV capacity.'),
            evidence=[
                Evidence('attention kernel share', _pct(share), '%'),
                _step_evidence(f, segmenter.DECODE, sid),
            ],
            knob_changes=changes,
            risk=F.Risk.QUALITY_GATED,
            confidence=F.Confidence.MEDIUM,
            est_step_fraction_saved=0.4 * share,
            validation='Accuracy eval on long-context tasks + decode A/B.',
        )
    ]


# -----------------------------------------------------------------------------
# D13: kernel follow-ups (no knob)
# -----------------------------------------------------------------------------
def d13_kernel_followups(f: features_lib.Features,
                         ctx: context_lib.ServingContext,
                         sid: str) -> list[Finding]:
    out = []
    for phase in _active_phases(f):
        share = features_lib.share_of(f.class_time, phase,
                                      oc.MOE_GMM_ZERO_INIT)
        if share >= D13_MIN_SHARE:
            out.append(
                Finding(
                    detector_id='D13_gmm_zero_init',
                    title=
                    'GMM output zero-initialization (broadcast) is expensive',
                    phase=phase,
                    summary=
                    ('broadcast (zero-init) ops inside the GMM pipeline take '
                     f'{_pct(share):.1f}% of the {phase} step. A GMM kernel that '
                     'initializes its output in-kernel removes them.'),
                    evidence=[
                        Evidence('GMM zero-init share', _pct(share), '%'),
                        *_top_op_evidence(f, phase, oc.MOE_GMM_ZERO_INIT),
                        _step_evidence(f, phase, sid),
                    ],
                    knob_changes=[],
                    risk=F.Risk.DIAGNOSTIC,
                    confidence=F.Confidence.MEDIUM,
                    est_step_fraction_saved=0.9 * share,
                    kernel_followup=
                    ('Check whether the deployed tpu_inference GMM kernel fuses '
                     'output zero-init (the Qwen3-Coder doc\'s "GMM v2"); upgrade '
                     'or file a kernel bug with these ops.'),
                    source_refs=[_DOC_QWEN3_CODER],
                ))
    return out


# -----------------------------------------------------------------------------
# D14: profile hygiene
# -----------------------------------------------------------------------------
def d14_hygiene(f: features_lib.Features, ctx: context_lib.ServingContext,
                sid: str) -> list[Finding]:
    ev, notes = [], []
    if f.meta.empty and f.seg.steps.empty:
        notes.append(
            'the trace has no TPU device events (host-only capture): enable the '
            'device tracer (xprof TPU tracer / torch_tpu profiler with device '
            'activities; for Ray workers profile inside the worker process) and '
            're-capture; no device findings are possible from this profile')
    active = _active_phases(f)
    for phase in f.seg.phases:
        n = _stat(f, phase, 'n_steps_per_device')
        if phase not in active:
            ev.append(Evidence(f'{phase} steps per device (skipped)', n))
            notes.append(
                f'{n:.0f} {phase} step(s) ignored as a minority phase')
        elif n < D14_MIN_STEPS:
            ev.append(Evidence(f'{phase} steps per device', n))
            notes.append(f'only {n:.0f} {phase} steps')
    for phase in active:
        cov = float(f.class_time.loc[f.class_time['phase'] == phase,
                                     'share'].sum())
        if cov < D19_MIN_COVERAGE:
            ev.append(Evidence(f'{phase} op-time coverage', _pct(cov), '%'))
            notes.append(
                f'ops cover only {_pct(cov):.0f}% of {phase} step time (op events '
                'missing, e.g. trace event limit or unprofiled cores); op-share '
                'findings are unreliable')
    if not f.stop_trace.empty:
        d = (f.stop_trace['end_ns'] - f.stop_trace['start_ns']).max()
        if d >= D14_STOP_TRACE_NS:
            ev.append(Evidence('stop_trace duration', _ms(d), ' ms'))
            notes.append('stop_trace stalls the engine (exclude the tail)')
    if (not f.seg.steps.empty and f.seg.steps['toks'].isna().all()
            and _framework(ctx, f) == 'vllm_tpu'):
        notes.append('no `execute_model:` host annotations (real batch sizes '
                     'unknown; use a newer tpu_inference or enable the Python '
                     'tracer)')
    if not notes:
        return []
    both = segmenter.DECODE in f.seg.phases and any(p in f.seg.phases
                                                    for p in _PREFILL_LIKE)
    changes = []
    if both and _framework(ctx, f) == 'vllm_tpu':
        changes.append(
            knobs.make_change(
                'vllm_tpu.env.PHASED_PROFILING_DIR',
                '/tmp/phased_profile',
                _cfg(ctx),
            ))
    return [
        Finding(
            detector_id='D14_profile_hygiene',
            title='Profile quality limits confidence',
            phase='all',
            summary='Re-profile to raise confidence: ' + '; '.join(notes) +
            '.',
            evidence=ev or [Evidence('notes', len(notes))],
            knob_changes=changes,
            risk=F.Risk.DIAGNOSTIC,
            confidence=F.Confidence.HIGH,
            est_step_fraction_saved=0.0,
        )
    ]


# -----------------------------------------------------------------------------
# D15: vLLM-TorchTPU fused expert-parallel MoE kernel
# -----------------------------------------------------------------------------
def d15_torchtpu_fused_ep(f: features_lib.Features,
                          ctx: context_lib.ServingContext,
                          sid: str) -> list[Finding]:
    if not _is_torchtpu(ctx, f) or f.step_ops.empty:
        return []
    so = f.step_ops
    if _name_mask(so, _FUSED_EP_OP).any():
        return []  # Fused EP MoE kernel already runs.
    if not (so['op_class'] == oc.MOE_GMM).any():
        return []
    out = []
    cur = knobs.current_value('vllm_torchtpu.env.USE_MOE_FUSED_EP_KERNEL',
                              _cfg(ctx))
    armed = cur is not None and F.KnobChange('', '', '', '1', cur).already_set
    for phase in _active_phases(f):
        g = _phase_ops(f, phase)
        gmm = _share(f, phase, g[g['op_class'] == oc.MOE_GMM])
        coll = g[(g['op_class'] == oc.COLLECTIVE)
                 & g['collective'].isin(['all-gather', 'reduce-scatter'])]
        cshare = _share(f, phase, coll)
        if gmm < D15_MIN_GMM_SHARE or cshare < D15_MIN_COLLECTIVE_SHARE:
            continue
        changes = []
        note = ''
        if armed:
            # Env set but the kernel is absent: node-wide token cap below the
            # arming threshold.
            cap = (ctx.max_num_batched_tokens or 0) * (ctx.dp or 1)
            changes.append(
                knobs.make_change(
                    'vllm_torchtpu.env.MOE_FUSED_EP_KERNEL_MIN_TOKENS',
                    str(cap) if cap else '256',
                    _cfg(ctx),
                ))
            note = (
                ' USE_MOE_FUSED_EP_KERNEL is set but no fused kernel ran: the '
                'node-wide token cap is below MOE_FUSED_EP_KERNEL_MIN_TOKENS.')
        else:
            changes.append(
                knobs.make_change('vllm_torchtpu.env.USE_MOE_FUSED_EP_KERNEL',
                                  '1', _cfg(ctx)))
        out.append(
            Finding(
                detector_id='D15_torchtpu_fused_ep_moe',
                title=
                'MoE runs as all-gather + grouped matmul + reduce-scatter',
                phase=phase,
                summary=
                ('The expert-parallel MoE is bracketed by all-gather /'
                 f' reduce-scatter collectives ({_pct(cshare):.1f}% of the'
                 f' {phase} step, exposed) around the grouped matmuls'
                 f' ({_pct(gmm):.1f}%). The fused EP MoE kernel runs routing,'
                 ' dispatch, both expert matmuls and combine as one SPMD Pallas'
                 ' program and overlaps the transport with the matmuls.' +
                 note),
                evidence=[
                    Evidence('MoE all-gather + reduce-scatter share',
                             _pct(cshare), '%'),
                    Evidence('grouped-matmul share', _pct(gmm), '%'),
                    *_base_op_evidence(f, phase, coll, 2),
                    _step_evidence(f, phase, sid),
                ],
                knob_changes=changes,
                risk=F.Risk.PERF_ONLY,
                confidence=F.Confidence.MEDIUM,
                est_step_fraction_saved=min(0.35, D15_GAIN_FRACTION * cshare),
                validation=
                ('A/B at the target ISL/OSL/concurrency. GLM-5.2 DP16+EP16 on'
                 ' v7x: +12% total tok/s at 256K/1K c16 and +32% at 8K/1K c1024'
                 ' (TPOT 47.7 -> 38.7 ms). Re-profile: expect a `fused_ep_moe*`'
                 ' kernel and no MoE reduce-scatter. Knob verified against a'
                 ' non-canonical vllm-torchtpu snapshot; confirm it exists in'
                 ' your build.'),
                source_refs=[_DOC_GLM_HILLCLIMB],
            ))
    return out


# -----------------------------------------------------------------------------
# D16: vLLM-TorchTPU sparse-MLA KV layout
# -----------------------------------------------------------------------------
def d16_torchtpu_mla_layout(f: features_lib.Features,
                            ctx: context_lib.ServingContext,
                            sid: str) -> list[Finding]:
    if not _is_torchtpu(ctx, f) or f.step_ops.empty:
        return []
    out = []
    for phase in _active_phases(f):
        g = _phase_ops(f, phase)
        mla = g[_name_mask(g, r'^MLA-')]
        if mla.empty or _name_mask(mla, 'sparsecore').any():
            continue
        share = _share(f, phase, mla)
        if share < D16_MIN_SHARE:
            continue
        out.append(
            Finding(
                detector_id='D16_torchtpu_mla_layout',
                title='Sparse-MLA KV cache uses the TensorCore layout',
                phase=phase,
                summary=
                (f'Sparse-MLA attention kernels ({_pct(share):.1f}% of the'
                 f' {phase} step) read a TensorCore-layout NoPE/RoPE KV cache.'
                 ' The SparseCore layout is the GLM-5.2 launch configuration;'
                 ' A/B it.'),
                evidence=[
                    Evidence('MLA kernel share', _pct(share), '%'),
                    *_base_op_evidence(f, phase, mla, 2),
                    _step_evidence(f, phase, sid),
                ],
                knob_changes=[
                    knobs.make_change(
                        'vllm_torchtpu.env.TPU_SPARSE_MLA_NOPE_LAYOUT',
                        'sparsecore',
                        _cfg(ctx),
                    ),
                    knobs.make_change(
                        'vllm_torchtpu.env.TPU_SPARSE_MLA_ROPE_LAYOUT',
                        'sparsecore',
                        _cfg(ctx),
                    ),
                ],
                risk=F.Risk.PERF_ONLY,
                confidence=F.Confidence.LOW,
                est_step_fraction_saved=min(0.05, 0.2 * share),
                validation=
                ('A/B end to end, not per kernel: GLM-5.2 256K/1K c16 gained +5%'
                 ' total tok/s, but the SparseCore-layout kernel is not always'
                 ' faster on the TensorCore timeline (it moves work to'
                 ' SparseCore).'),
                source_refs=[_DOC_GLM_HILLCLIMB],
            ))
    return out


# -----------------------------------------------------------------------------
# D17: data-formatting / KV-cache write hotspots (kernel follow-up)
# -----------------------------------------------------------------------------
def d17_data_formatting(f: features_lib.Features,
                        ctx: context_lib.ServingContext,
                        sid: str) -> list[Finding]:
    if _is_copy_program(f):
        return []  # A transfer-only capture says nothing about formatting.
    out = []
    for phase in _active_phases(f):
        g = _phase_ops(f, phase)
        fmt = g[g['op_class'] == oc.DATA_FORMATTING]
        in_loop = g['tf_op_name'].fillna('').str.contains('/while/',
                                                          regex=False)
        in_moe = g['moe_scope'].fillna(False).astype(bool)
        dus = g[_name_mask(g, _KV_WRITE_OP) & ~in_loop & ~in_moe]
        n = max(_n_steps(f, phase), 1)
        dus_per_step = len(dus) / n
        share = _share(f, phase, fmt)
        dus_share = _share(f, phase, dus)
        if share < D17_MIN_SHARE and not (dus_per_step >= D17_MIN_KV_WRITE_OPS
                                          and dus_share >= D13_MIN_SHARE):
            continue
        total = share + (dus_share
                         if dus_per_step >= D17_MIN_KV_WRITE_OPS else 0.0)
        out.append(
            Finding(
                detector_id='D17_data_formatting',
                title='Relayout copies / per-token KV writes dominate',
                phase=phase,
                summary=(
                    'Data-formatting ops (copy / reshape / transpose) take '
                    f'{_pct(share):.1f}% of the {phase} step' +
                    (f' and {dus_per_step:.0f} dynamic-update-slice / scatter'
                     f' ops per step take {_pct(dus_share):.1f}%'
                     if dus_per_step >= D17_MIN_KV_WRITE_OPS else '') +
                    '. These usually come from a KV-cache layout that does not'
                    ' match the attention kernel, or from KV writes lowered to'
                    ' loops.'),
                evidence=[
                    Evidence('data-formatting share', _pct(share), '%'),
                    Evidence(
                        'KV-write ops per step',
                        dus_per_step,
                        detail=f'{_pct(dus_share):.1f}% of step',
                    ),
                    *_base_op_evidence(f, phase, fmt, 3),
                    *_base_op_evidence(f, phase, dus, 1),
                    _step_evidence(f, phase, sid),
                ],
                knob_changes=[],
                risk=F.Risk.DIAGNOSTIC,
                confidence=F.Confidence.MEDIUM,
                est_step_fraction_saved=0.7 * total,
                kernel_followup=
                ('Find the producer/consumer of the top copies in the trace. On'
                 ' GLM-5.2 (vLLM-TorchTPU) these were removed by a native split'
                 ' sparse-MLA KV cache (no per-step relayout) and a SparseCore'
                 ' KV scatter kernel (replacing dynamic-update-slice loops).'
                 ' File a kernel/model bug with the linked ops.'),
                source_refs=[_DOC_GLM_BASELINE],
            ))
    return out


# -----------------------------------------------------------------------------
# D18: sparse-attention indexer hotspot
# -----------------------------------------------------------------------------
def _dcp_fits_kv_budget(ctx: context_lib.ServingContext,
                        target_dcp: int) -> bool:
    """Whether `target_dcp` shards can hold `ctx.concurrency` sequences in HBM."""
    model = ctx.model
    seq_len = ctx.max_model_len or ((ctx.input_len or 0) +
                                    (ctx.output_len or 0))
    if not seq_len or not model.num_layers:
        return True
    tp_pcp = max(1, (ctx.tp or 1) * max(1, ctx.pcp or 1))
    non_dcp_ranks = max(1, tp_pcp // max(1, target_dcp))
    if model.mla_kv_dim:
        per_token_elems = model.mla_kv_dim
        kv_head_shards = 1
    elif model.num_kv_heads and model.head_dim:
        per_token_elems = 2 * model.num_kv_heads * model.head_dim
        kv_head_shards = min(model.num_kv_heads, non_dcp_ranks)
    else:
        per_token_elems = _D20_DEFAULT_MLA_LATENT_DIM
        kv_head_shards = 1
    kv_dtype = (ctx.config_get('--kv-cache-dtype', 'kv-cache-dtype',
                               'kv_cache_dtype') or '').lower()
    bytes_per_elem = 1 if ('fp8' in kv_dtype or 'f8' in kv_dtype) else 2
    dp_replicas = max(1, ctx.dp or 1)
    seqs_per_dp = max(1, math.ceil((ctx.concurrency or 1) / dp_replicas))
    kv_per_rank = (model.num_layers * per_token_elems * seq_len * seqs_per_dp *
                   bytes_per_elem) / max(1, target_dcp * kv_head_shards)
    budget = ctx.kv_cache_memory_bytes or _D20_DEFAULT_KV_BUDGET_BYTES
    return kv_per_rank <= budget


def d18_indexer(f: features_lib.Features, ctx: context_lib.ServingContext,
                sid: str) -> list[Finding]:
    out = []
    dcp = ctx.dcp or du.knob_int(ctx, 'vllm.cli.decode_context_parallel_size',
                                 1)
    tp_pcp = (ctx.tp or 1) * max(1, ctx.pcp or 1)
    target_dcp = limits.SMALL_BATCH_DECODE_TARGET_DCP
    for phase in _active_phases(f):
        g = _phase_ops(f, phase)
        idx = g[g['op_class'] == oc.ATTN_INDEXER]
        share = _share(f, phase, idx)
        if share < D18_MIN_SHARE:
            continue
        sc = _name_mask(idx, 'sc_topk').any()
        sort = g[(g['op_class'] != oc.ATTN_INDEXER)
                 & ~g['moe_scope'].fillna(False).astype(bool)
                 & g['tf_op_type'].isin(['top_k', 'approx_top_k', 'sort'])]
        sort_share = _share(f, phase, sort)
        mean_bucket = _stat(f, phase, 'mean_bucket')
        small_decode = phase == segmenter.DECODE and (
            bool(ctx.concurrency and ctx.concurrency <= _D18_SMALL_DECODE_SEQS)
            or (not math.isnan(mean_bucket)
                and mean_bucket <= _D18_SMALL_DECODE_BUCKET))
        changes = []
        dcp_note = ''
        validation = ''
        est_saved = 0.5 * (share + sort_share)
        if (small_decode and ctx.is_pd_decode and _is_torchtpu(ctx, f)
                and ctx.serving_config is not None
                and dcp >= limits.MIN_ASYMMETRIC_PREFILL_DCP
                and tp_pcp % target_dcp == 0
                and _dcp_fits_kv_budget(ctx, target_dcp)):
            changes = [
                knobs.make_change(
                    'vllm.cli.decode_context_parallel_size',
                    str(target_dcp),
                    ctx.serving_config,
                )
            ]
            dcp_note = (
                f' At small decode batch with DCP={dcp}, a decode engine at'
                ' `--decode-context-parallel-size=8` halves the DCP top-k / KV'
                ' merge fan-in, which lowered GLM-5.2 P/D decode TPOT. Under'
                ' disaggregated P/D lower it on the decode leg only'
                ' (`DECODE_DCP_SIZE=8`, keeping `PREFILL_DCP_SIZE=16`):'
                ' symmetric DCP 8 on both legs halved KV capacity, lowered the'
                ' prefill prefix-cache hit rate and regressed Mtok/$.')
            validation = (
                'A/B decode p90 interactivity and Mtok/$ with'
                ' `--decode-context-parallel-size=8` on the decode engine only.'
                ' Under disaggregated P/D keep `PREFILL_DCP_SIZE=16` with an'
                ' asymmetric KV transfer path (`DECODE_DCP_SIZE=8`); symmetric'
                ' DCP 8 on both legs halved KV capacity (11.89M -> 5.95M'
                ' tokens), dropped the prefill HBM prefix-cache hit rate from'
                ' 91.9% to 80.2%, and lowered throughput per cost by 24.8% at'
                ' concurrency 48 and 1.7% at concurrency 32.')
            est_saved = min(0.25, 0.35 * (share + sort_share))
        out.append(
            Finding(
                detector_id='D18_attn_indexer',
                title='Sparse-attention indexer (scoring + top-k) is a hotspot',
                phase=phase,
                summary=
                (f'The DSA-style indexer takes {_pct(share):.1f}% of the {phase}'
                 ' step' + (f' (+{_pct(sort_share):.1f}% non-MoE top-k/sort)'
                            if sort_share >= D13_MIN_SHARE else '') +
                 '. Its cost grows with context length, so it dominates long-'
                 'context decode.' +
                 ('' if sc else
                  (' The top-k is not using the SparseCore top-k kernel.')) +
                 dcp_note),
                evidence=[
                    Evidence('indexer share', _pct(share), '%'),
                    Evidence('non-MoE top-k / sort share', _pct(sort_share),
                             '%'),
                    *([Evidence('DCP', dcp)] if changes else []),
                    *_base_op_evidence(f, phase, idx, 3),
                    _step_evidence(f, phase, sid),
                ],
                knob_changes=changes,
                risk=F.Risk.PERF_ONLY if changes else F.Risk.DIAGNOSTIC,
                confidence=F.Confidence.LOW
                if changes else F.Confidence.MEDIUM,
                est_step_fraction_saved=est_saved,
                validation=validation,
                kernel_followup=
                ('GLM-5.2 moved the indexer top-k to a SparseCore top-k kernel'
                 ' (`sc_topk_direct_*`) and streams indexer scoring'
                 ' (`StreamIdxTC-*`). If those already run, the remaining lever'
                 ' is overlapping the SparseCore top-k with TensorCore work or'
                 ' lowering index_topk; hand off to the kernel owners.'),
                source_refs=[_DOC_GLM_BASELINE],
            ))
    return out


# -----------------------------------------------------------------------------
# D19: TensorCore waits inside the step (async offload waits / op coverage)
# -----------------------------------------------------------------------------
def d19_in_step_wait(f: features_lib.Features, ctx: context_lib.ServingContext,
                     sid: str) -> list[Finding]:
    out = []
    for phase in _active_phases(f):
        ct = f.class_time[f.class_time['phase'] == phase]
        attributed = float(ct['share'].sum())
        wait = features_lib.share_of(f.class_time, phase, oc.ASYNC_WAIT)
        if attributed < D19_MIN_COVERAGE:
            continue  # Reported by D14 (op events likely missing).
        gap = max(0.0, 1.0 - attributed)
        total = wait + gap
        if total < D19_MIN_WAIT_SHARE:
            continue
        g = _phase_ops(f, phase)
        waits = g[g['op_class'] == oc.ASYNC_WAIT]
        out.append(
            Finding(
                detector_id='D19_in_step_wait',
                title='TensorCore idles inside the step program',
                phase=phase,
                summary=
                (f'{_pct(total):.1f}% of the {phase} step is the TensorCore'
                 f' waiting: {_pct(wait):.1f}% in async-done ops (SparseCore'
                 f' gather/scatter offloads, DMA copies) and {_pct(gap):.1f}%'
                 ' with no op running inside the program.'),
                evidence=[
                    Evidence('async-done wait share', _pct(wait), '%'),
                    Evidence('no-op gap inside step', _pct(gap), '%'),
                    *_base_op_evidence(f, phase, waits, 3),
                    _step_evidence(f, phase, sid),
                ],
                knob_changes=[],
                risk=F.Risk.DIAGNOSTIC,
                confidence=F.Confidence.LOW,
                est_step_fraction_saved=0.5 * total,
                kernel_followup=
                ('Open the linked step in the trace viewer with the SparseCore '
                 'lines expanded: find what the TensorCore waits on and whether '
                 'independent TensorCore work can be scheduled before the '
                 'async-done (XLA latency-hiding scheduler / kernel overlap).'
                 ),
            ))
    return out


# -----------------------------------------------------------------------------
# D20: MLA attention sharded with TP (KV latent replicated)
# -----------------------------------------------------------------------------
def _mla_kv_needs_dcp(ctx: context_lib.ServingContext) -> bool:
    """Whether a single rank's KV cache cannot hold the configured context."""
    m = ctx.model
    seq_len = ctx.max_model_len or ((ctx.input_len or 0) +
                                    (ctx.output_len or 0))
    if not seq_len or not m.num_layers:
        return False
    latent_dim = m.mla_kv_dim or _D20_DEFAULT_MLA_LATENT_DIM
    kv_dtype = (ctx.config_get('--kv-cache-dtype', 'kv-cache-dtype',
                               'kv_cache_dtype') or '').lower()
    bytes_per_elem = 1 if ('fp8' in kv_dtype or 'f8' in kv_dtype) else 2
    dp_ranks = max(1, (ctx.tp or 1) * (ctx.dp or 1) * max(1, ctx.pcp or 1))
    seqs_per_rank = max(1, math.ceil((ctx.concurrency or 1) / dp_ranks))
    kv_bytes_needed = (m.num_layers * latent_dim * seq_len * seqs_per_rank *
                       bytes_per_elem)
    budget = ctx.kv_cache_memory_bytes or _D20_DEFAULT_KV_BUDGET_BYTES
    return kv_bytes_needed > budget


def d20_mla_attention_tp(f: features_lib.Features,
                         ctx: context_lib.ServingContext,
                         sid: str) -> list[Finding]:
    m = ctx.model
    if not m.is_mla or not _is_vllm(ctx, f) or not ctx.tp or ctx.tp <= 1:
        return []
    if (ctx.dp or 1) > 1 and ctx.tp <= 2:
        return []  # Hybrid DPxTP2 already; not the pathological case.
    dcp = ctx.dcp or du.knob_int(ctx, 'vllm.cli.decode_context_parallel_size',
                                 1)
    if dcp > 1 or (ctx.pcp or 1) > 1 or _mla_kv_needs_dcp(ctx):
        # Context parallelism (or a KV cache one rank cannot hold) already
        # shards the MLA KV cache across ranks instead of replicating it, and
        # a TP=1 / DP mesh would drop that sharding.
        return []
    dp_ranks = ctx.tp * (ctx.dp or 1)
    if ctx.concurrency and ctx.concurrency < dp_ranks:
        return []  # Fewer requests than DP ranks would leave ranks idle.
    out = []
    for phase in _active_phases(f):
        share = features_lib.share_of(
            f.class_time,
            phase,
            oc.ATTENTION_KERNEL,
            oc.ATTENTION_OTHER,
            oc.ATTN_INDEXER,
        )
        cov = float(f.class_time.loc[f.class_time['phase'] == phase,
                                     'share'].sum())
        # With most op events missing the attention share is unmeasurable; the
        # config alone (MLA + TP>2) is the documented pathology, so still fire.
        config_only = cov < D19_MIN_COVERAGE
        if share < D20_MIN_ATTN_SHARE and not config_only:
            continue
        if _is_torchtpu(ctx, f):
            n = ctx.tp * (ctx.dp or 1)
            changes = [
                knobs.make_change('vllm.cli.data_parallel_size', str(n),
                                  _cfg(ctx)),
                knobs.make_change('vllm.cli.tensor_parallel_size', '1',
                                  _cfg(ctx)),
            ]
            if m.is_moe:
                changes.append(
                    knobs.make_change('vllm.cli.enable_expert_parallel',
                                      'true', _cfg(ctx)))
        else:
            changes = [
                knobs.make_change(
                    'vllm_tpu.additional_config.enable_dp_attention',
                    'true',
                    _cfg(ctx),
                )
            ]
        if config_only:
            share_txt = (
                'attention share unmeasurable: ops cover only '
                f'{_pct(cov):.0f}% of the {phase} step; the estimate is a'
                f' prior (an assumed {_pct(D20_CONFIG_ONLY_EST):.0f}%'
                ' attention share), not a measurement of this trace')
            share_ev = Evidence(f'{phase} op-time coverage', _pct(cov), '%')
            est = D20_CONFIG_ONLY_EST * (1 - 1 / ctx.tp)
        else:
            share_txt = f'{_pct(share):.1f}% of the {phase} step'
            share_ev = Evidence('attention share', _pct(share), '%')
            est = min(0.5, share * (1 - 1 / ctx.tp))
        out.append(
            Finding(
                detector_id='D20_mla_attention_tp',
                title='MLA attention under tensor parallelism',
                phase=phase,
                summary=
                ('The model uses MLA (one shared KV latent per token) with'
                 f' attention TP={ctx.tp}: every TP rank reads the full latent'
                 f' KV cache, so attention ({share_txt}) does TP× redundant KV'
                 ' reads. Attention data parallelism splits requests (and their'
                 ' KV) across ranks instead.'),
                evidence=[
                    share_ev,
                    Evidence('attention TP', ctx.tp),
                    Evidence('kv_lora_rank', m.kv_lora_rank or 0),
                    _step_evidence(f, phase, sid),
                ],
                knob_changes=changes,
                risk=F.Risk.MEMORY,
                confidence=(F.Confidence.MEDIUM
                            if config_only else F.Confidence.HIGH),
                est_step_fraction_saved=est,
                validation=
                ('Co-propose with the mesh step; set --max-num-seqs per DP '
                 'rank. GLM-5.2 256K/1K c16 on 16 v7x chips: DP16+EP16 served '
                 '7.3× the tok/s of TP16+EP16 (18.7K vs 2.6K).'),
                source_refs=[_DOC_GLM_HILLCLIMB],
            ))
    return out


# -----------------------------------------------------------------------------
# D21: sequential per-token loop (non-chunked recurrence)
# -----------------------------------------------------------------------------
def d21_sequential_loop(f: features_lib.Features,
                        ctx: context_lib.ServingContext,
                        sid: str) -> list[Finding]:
    out = []
    for phase in _active_phases(f):
        g = _phase_ops(f, phase)
        loop = g[g['tf_op_name'].fillna('').str.contains('/while/body/',
                                                         regex=False)]
        n = max(_n_steps(f, phase), 1)
        per_step = len(loop) / n
        share = _share(f, phase, loop)
        if per_step < D21_MIN_OPS_PER_STEP or share < D21_MIN_SHARE:
            continue
        lin = (g['op_class'] == oc.LINEAR_ATTENTION).any() or bool(
            re.search(r'gdn|delta|linear|mamba|kda', ctx.model.name, re.I))
        names = loop['tf_op_name'].fillna('')
        w = loop['self_ns'].astype(float)
        tot = max(float(w.sum()), 1.0)
        # A loop whose body holds collectives or expert matmuls runs whole layers
        # (MaxText on-device generate loop, scan over layers): not a recurrence.
        model_loop = (float(w[loop['op_class'].isin(
            [oc.COLLECTIVE, oc.MOE_GMM])].sum()) / tot)
        if model_loop >= D21_MAX_MODEL_LOOP_SHARE:
            continue
        attn_mask = (
            loop['op_class'].isin([oc.ATTENTION_KERNEL, oc.ATTENTION_OTHER])
            | names.str.contains(oc.ATTENTION_SCOPE)) & ~names.str.contains(
                oc.LINEAR_ATTENTION_SCOPE)
        attn_loop = (not lin) and float(w[attn_mask].sum()) / tot >= 0.5
        # Innermost jit(...) scope above the loop (e.g. mla_ragged_paged_attention).
        scopes = names.str.extract(r'jit\(([^)]+)\)[^()]*/while/body/',
                                   expand=False).fillna('')
        by_scope = w.groupby(scopes).sum().sort_values(ascending=False)
        top_scope = str(by_scope.index[0]) if len(by_scope) else ''
        if top_scope in ('step_fun', 'run_model', 'prefill_step',
                         'decode_step'):
            top_scope = ''  # The whole step program; not informative.
        scope_txt = f' (scope `{top_scope}`)' if top_scope else ''
        if attn_loop:
            title = 'Attention lowered to an XLA while loop'
            what = (f'the attention op{scope_txt} runs as an XLA while loop '
                    '(reference / non-Pallas path) instead of a fused kernel.')
            followup = (
                'Switch to the Pallas ragged-paged-attention kernel for this '
                'attention type (check the backend/attention selection and '
                "kernel support for the model's head dims), or file a kernel "
                'bug; confirm the loop scope in the linked step.')
            est = 0.5 * share
        else:
            title = 'Per-token while loop in the step (non-chunked recurrence)'
            what = f'a recurrence{scope_txt} is being stepped token by token. ' + (
                'For linear attention (GDN/KDA), a chunked kernel '
                'processes a whole chunk per iteration with matmuls.' if lin
                else 'Replace the loop with a chunked / parallel-scan kernel.')
            followup = (
                'Use the chunked GDN / linear-attention prefill kernel (Qwen3.5 '
                'serving work: chunked GDN was one of the landed wins) or file a '
                'model bug; confirm the loop scope in the linked step.')
            est = 0.8 * share
        out.append(
            Finding(
                detector_id='D21_sequential_loop',
                title=title,
                phase=phase,
                summary=
                (f'{per_step:.0f} tiny ops/step run inside while-loop bodies and'
                 f' take {_pct(share):.1f}% of the {phase} step: {what}'),
                evidence=[
                    Evidence('loop ops per step', per_step),
                    Evidence('loop share', _pct(share), '%'),
                    *([Evidence('dominant loop scope', top_scope)]
                      if top_scope else []),
                    *_base_op_evidence(f, phase, loop, 3),
                    _step_evidence(f, phase, sid),
                ],
                knob_changes=[],
                risk=F.Risk.DIAGNOSTIC,
                confidence=F.Confidence.MEDIUM,
                est_step_fraction_saved=est,
                kernel_followup=followup,
            ))
    return out


# -----------------------------------------------------------------------------
# D22: MoE model without a grouped-matmul kernel
# -----------------------------------------------------------------------------
def d22_moe_without_gmm(f: features_lib.Features,
                        ctx: context_lib.ServingContext,
                        sid: str) -> list[Finding]:
    if not ctx.model.is_moe or f.step_ops.empty:
        return []
    so = f.step_ops
    if _name_mask(so, _FUSED_EP_OP + r'|fused_moe|megablox').any():
        return []
    out = []
    fw = _framework(ctx, f)
    for phase in _active_phases(f):
        gmm = features_lib.share_of(f.class_time, phase, oc.MOE_GMM)
        if gmm >= D22_MAX_GMM_SHARE:
            continue
        g = _phase_ops(f, phase)
        dense = g[g['op_class'].isin([oc.MATMUL, oc.OTHER, oc.MOE_OTHER])]
        gathers = dense[dense['tf_op_type'].isin(['gather', 'scatter'])]
        dshare = _share(f, phase, dense)
        if dshare < D22_MIN_DENSE_SHARE:
            continue
        changes = []
        if fw == 'maxtext':
            changes = [
                knobs.make_change('maxtext.yml.megablox', 'true', _cfg(ctx)),
                knobs.make_change('maxtext.yml.sparse_matmul', 'true',
                                  _cfg(ctx)),
            ]
        out.append(
            Finding(
                detector_id='D22_moe_without_gmm',
                title='MoE experts run without a grouped-matmul kernel',
                phase=phase,
                summary=
                (f'The model has {ctx.model.num_experts} experts but'
                 f' grouped-matmul kernels are {_pct(gmm):.1f}% of the {phase}'
                 f' step while dense matmul / gather ops are {_pct(dshare):.1f}%'
                 f' (gather/scatter {_pct(_share(f, phase, gathers)):.1f}%). The'
                 ' experts are likely computed densely or by gathering expert'
                 ' weights per token.'),
                evidence=[
                    Evidence('grouped-matmul share', _pct(gmm), '%'),
                    Evidence('dense matmul + gather share', _pct(dshare), '%'),
                    *_base_op_evidence(f, phase, dense, 3),
                    _step_evidence(f, phase, sid),
                ],
                knob_changes=changes,
                risk=F.Risk.PERF_ONLY if changes else F.Risk.DIAGNOSTIC,
                confidence=F.Confidence.MEDIUM
                if changes else F.Confidence.LOW,
                est_step_fraction_saved=min(0.5, 0.5 * dshare),
                kernel_followup=None if changes else
                ('Switch the MoE layer to a ragged grouped-matmul (GMM /'
                 ' megablox) path with expert parallelism; on Kimi the optimized'
                 ' build (EP + GMM kernels) replaced the dense TP bf16 MoE.'),
            ))
    return out


# -----------------------------------------------------------------------------
# D24: vLLM MoE under tensor parallelism without expert parallelism
# -----------------------------------------------------------------------------
def d24_moe_tp_without_ep(f: features_lib.Features,
                          ctx: context_lib.ServingContext,
                          sid: str) -> list[Finding]:
    m = ctx.model
    if (not m.is_moe or m.is_mla or not _is_vllm(ctx, f) or not ctx.tp
            or ctx.tp <= 1 or ctx.ep or ctx.serving_config is None):
        return []
    ep_kernel = knobs.current_value('vllm_tpu.env.USE_MOE_EP_KERNEL',
                                    _cfg(ctx))
    if _is_true(ep_kernel) or (not f.step_ops.empty
                               and _name_mask(f.step_ops, _FUSED_EP_OP).any()):
        return []
    out = []
    if _prefill_heavy(ctx):
        return []  # EP->TP (GMM v2) won on Qwen3-Coder 8k/1k.
    for phase in _active_phases(f):
        if phase != segmenter.DECODE:
            continue
        share = features_lib.share_of(f.class_time, phase, oc.MOE_GMM,
                                      oc.MOE_PERMUTE, oc.MOE_OTHER)
        if share < D24_MIN_MOE_SHARE:
            continue
        width = (m.moe_intermediate_size //
                 ctx.tp if m.moe_intermediate_size else None)
        width_txt = (
            f' Each expert is split {ctx.tp} ways, so every device runs '
            f'all {m.num_experts} experts on {width}-wide slices.'
            if width else '')
        decode_heavy = bool(ctx.input_len and ctx.output_len
                            and ctx.output_len > ctx.input_len)
        out.append(
            Finding(
                detector_id='D24_moe_tp_without_ep',
                title='MoE experts sharded by TP instead of EP (decode)',
                phase=phase,
                summary=
                (f'MoE ops are {_pct(share):.1f}% of the decode step with '
                 f'TP={ctx.tp} and no expert parallelism.{width_txt} With '
                 '--enable-expert-parallel each device owns whole experts '
                 f'(E/{ctx.tp}), giving larger grouped matmuls. Measured on '
                 'Qwen3-Coder-480B TP8 (v7x): +12% at 1k in / 8k out, but the '
                 'prefill-heavy 8k/1k setup moved EP -> TP with GMM v2 '
                 '(about +13-17%, unstable because of a GMM v2 race; verify '
                 'accuracy), so A/B at the target workload.'),
                evidence=[
                    Evidence('MoE share', _pct(share), '%'),
                    Evidence('TP', ctx.tp),
                    *([Evidence('per-device expert width', width)]
                      if width else []),
                    _step_evidence(f, phase, sid),
                ],
                knob_changes=[
                    knobs.make_change('vllm.cli.enable_expert_parallel',
                                      'true', _cfg(ctx))
                ],
                risk=F.Risk.PERF_ONLY,
                confidence=(F.Confidence.MEDIUM
                            if decode_heavy else F.Confidence.LOW),
                est_step_fraction_saved=min(D24_MAX_GAIN, 0.3 * share),
                validation=
                ('A/B output tok/s at the target input/output lengths and '
                 'concurrency; EP adds a token all-to-all and exposes expert '
                 'imbalance (Qwen3-Coder: 1.5x GMM time spread at EP=8), so '
                 'check D6 after. Sources: Qwen3-Coder-480B on TPU v7x'
                 f' (+12%); {_DOC_QWEN_CODER} (EP->TP).'),
            ))
    return out


# -----------------------------------------------------------------------------
# D25: host-driven transfer (per-shard host copies, device idle)
# -----------------------------------------------------------------------------
D25_MIN_COPIES = 1000
D25_MAX_DEVICE_BUSY = 0.2


def d25_host_driven_transfer(f: features_lib.Features,
                             ctx: context_lib.ServingContext,
                             sid: str) -> list[Finding]:
    del ctx, sid  # Unused.
    t = f.transfer or {}
    n, span = t.get('n_copies', 0.0), t.get('span_ms', 0.0)
    busy = t.get('device_busy_ms', 0.0)
    if n < D25_MIN_COPIES or not span or busy / span > D25_MAX_DEVICE_BUSY:
        return []
    copy_ms = t.get('copy_ms', 0.0)
    return [
        Finding(
            detector_id='D25_host_driven_transfer',
            title='Transfer is driven by per-shard host copies',
            phase='all',
            summary=
            (f'The host issues {n:.0f} per-shard device copies ({copy_ms:.1f}'
             f' ms of host time) while each TPU is busy only {busy:.2f} ms of'
             f' the {span:.1f} ms window ({_pct(busy / span):.1f}%). The'
             ' transfer is bounded by host dispatch and transport, not device'
             ' compute: pack the KV/prefix buffers into fewer, larger arrays'
             ' (e.g. one array per layer group instead of per page/shard) or'
             ' use a device-to-device transfer path, so the per-copy overhead'
             ' is paid fewer times.'),
            evidence=[
                Evidence('host copy calls', n),
                Evidence('host time in copies', copy_ms, ' ms'),
                Evidence('device busy per TPU', busy, ' ms'),
                Evidence('profile window', span, ' ms'),
            ],
            knob_changes=[],
            risk=F.Risk.DIAGNOSTIC,
            confidence=F.Confidence.LOW,
            est_step_fraction_saved=0.0,
            validation=(
                'Compare the transfer-time metric over >=3 runs; also capture'
                ' the full transfer: this trace window may be shorter than it.'
            ),
            kernel_followup=(
                'Check the host lanes for the transfer loop (CopyToDevices /'
                ' DevicePutWithSharding / IciTransport) to find how many arrays'
                ' are moved per request.'),
        )
    ]


ALL_DETECTORS: tuple[Callable[..., list[Finding]], ...] = (
    d1_host_gap,
    d2_padding,
    d3_interference,
    d4_topk,
    d5_permute,
    d6_imbalance,
    d7_collectives,
    d8_kv_head_padding,
    d9_rope_cache,
    d11_recompile,
    d12_kv_quant,
    d13_kernel_followups,
    d14_hygiene,
    d15_torchtpu_fused_ep,
    d16_torchtpu_mla_layout,
    d17_data_formatting,
    d18_indexer,
    d19_in_step_wait,
    d20_mla_attention_tp,
    d21_sequential_loop,
    d22_moe_without_gmm,
    d24_moe_tp_without_ep,
    d25_host_driven_transfer,
)


def run_all(
    f: features_lib.Features,
    ctx: context_lib.ServingContext,
    session_id: str = '',
    detectors: Sequence[Callable[
        [features_lib.Features, context_lib.ServingContext, str],
        list[Finding],
    ]] = ALL_DETECTORS,
) -> tuple[list[Finding], list[str]]:
    """Runs detectors; returns (findings, errors). One failure never aborts."""
    results, errors = [], []
    for d in detectors:
        try:
            results.extend(d(f, ctx, session_id))
        except Exception as e:  # pylint: disable=broad-exception-caught
            errors.append(f'{d.__name__}: {type(e).__name__}: {e}')
    return results, errors
