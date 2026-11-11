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
"""Detectors from the GPT-OSS / Qwen3-Coder headroom, DeepSeek-V3, and Dense S V2 docs.

Covers D36 (decode batch capped by HBM while MoE GMM is weight-bound),
D37 (MoE expert weights wider than 4 bits), D40 (MLA attention kernel
generation and config) and D44 (unchunked MoE combine collective in vLLM-TPU /
vLLM-TorchTPU). Gains and caveats are quoted from the source docs;
see `source_refs` on each finding.
"""

# pylint: disable=missing-function-docstring,missing-class-docstring,g-doc-args,g-doc-return-or-yield,g-docstring-has-escape,g-long-ternary,g-explicit-length-test,g-complex-comprehension,protected-access,cell-var-from-loop,comparison-with-itself,unused-argument

import math
import re

from serving_analysis import context as context_lib
from serving_analysis import detector_utils as du
from serving_analysis import features as features_lib
from serving_analysis import findings as F
from serving_analysis import knobs
from serving_analysis import op_classes as oc
from serving_analysis import segmenter

Finding = F.Finding
Evidence = F.Evidence

_DOC_DSV3_SERVING = 'DeepSeek-V3 serving study'
_DOC_DSV3_MOE = 'DeepSeek-V3 MoE strategy study on TPU v7x'
_DOC_GPT_OSS = 'gpt-oss serving headroom study'
_DOC_QCH = 'Qwen3-Coder-480B serving headroom study'

# D36: decode GMM is weight-bound and the batch is capped by HBM.
D36_MIN_GMM_SHARE = 0.25
# Routed tokens per expert per step below which a grouped matmul is
# weight-bandwidth bound (v7x bf16/fp8 ridge is a few hundred rows).
D36_MAX_TOKENS_PER_EXPERT = 160
D36_SATURATED_FRAC = 0.9
D36_TARGET_GMU = 0.98
D36_SEQS_GROWTH = 1.25
D36_SEQS_ALIGN = 64
# D37: MoE weights wider than 4 bits.
D37_MIN_GMM_SHARE = 0.10
# D40: MLA attention kernel generation / config.
D40_MIN_ATTN_SHARE = 0.05
D40_TUNED_KV_PACKING = 32
# D44: MoE combine collective not chunked.
D44_MIN_SHARE = 0.05
D44_MIN_ROWS = 2048
D44_CHUNK = 1024

_FP4_WEIGHT = re.compile(r'\b(f4e2m1fn|s4|u4)\[')
_ANY_OPERAND = re.compile(r'custom-call\([^)]*\b[a-z][a-z0-9]*\[')
_FP4_ALIASES = ('fp4', 'float4_e2m1fn', 'f4e2m1fn', 'int4', 's4')
_MLA_V1 = re.compile(r'(?:^|/)MLA-RPA-bq_')
_DSV4 = re.compile(r'deepseek[-_]?v4|dsv4', re.I)
_TRUE = ('1', 'true', 'yes', 'on')


def _pct(x: float) -> float:
    return 100.0 * float(x)


def _knob(ctx: context_lib.ServingContext, key: str) -> str:
    return str(knobs.current_value(key, ctx.serving_config) or '').strip()


def _int_or(s: str, default: int) -> int:
    try:
        return int(s)
    except ValueError:
        return default


def _float_or(s: str, default: float) -> float:
    try:
        return float(s)
    except ValueError:
        return default


def _is_vllm_tpu(ctx: context_lib.ServingContext,
                 f: features_lib.Features) -> bool:
    return (du.framework(ctx, f) == 'vllm_tpu'
            and ctx.serving_config is not None)


# -----------------------------------------------------------------------------
# D36: decode batch limited by HBM while MoE GMM is weight-bound
# -----------------------------------------------------------------------------
def d36_hbm_limited_decode_batch(f: features_lib.Features,
                                 ctx: context_lib.ServingContext,
                                 sid: str) -> list[Finding]:
    m = ctx.model
    if (not _is_vllm_tpu(ctx, f) or not m.is_moe or not m.num_experts_per_tok
            or segmenter.DECODE not in f.seg.phases):
        return []
    gmm_share = features_lib.share_of(f.class_time, segmenter.DECODE,
                                      oc.MOE_GMM)
    if gmm_share < D36_MIN_GMM_SHARE:
        return []
    bucket = du.stat(f, segmenter.DECODE, 'mean_bucket')
    if not bucket or math.isnan(bucket):
        return []
    dp = ctx.dp or 1
    per_expert = bucket * dp * m.num_experts_per_tok / (m.num_experts or 1)
    if per_expert >= D36_MAX_TOKENS_PER_EXPERT:
        return []
    # Only a batch that is already full is capped by memory; a load-limited
    # batch (concurrency below max_num_seqs) does not grow with more KV.
    mns = ctx.max_num_seqs
    reqs = du.stat(f, segmenter.DECODE, 'mean_reqs')
    has_reqs = not math.isnan(reqs) and reqs > 0
    saturated = None
    sat_txt = 'batch saturation is unknown (no --max-num-seqs or request counts)'
    if mns and has_reqs:
        saturated = reqs >= D36_SATURATED_FRAC * mns
        sat_txt = (
            f'the batch is full ({reqs:.0f} of --max-num-seqs={mns} requests)')
    elif mns and ctx.concurrency:
        per_replica = ctx.concurrency / dp
        saturated = per_replica >= D36_SATURATED_FRAC * mns
        sat_txt = (
            f'the batch is full (benchmark concurrency {per_replica:.0f} per DP'
            f' replica vs --max-num-seqs={mns})')
    if saturated is not None and not saturated:
        return []
    cfg = ctx.serving_config
    changes = []
    gmu = _float_or(_knob(ctx, 'vllm.cli.gpu_memory_utilization'), 0.92)
    if gmu < D36_TARGET_GMU:
        changes.append(
            knobs.make_change('vllm.cli.gpu_memory_utilization',
                              str(D36_TARGET_GMU), cfg))
    kv_cfg = _knob(ctx, 'vllm_tpu.cli.kv_cache_dtype').lower()
    dequant, _ = du.kv_dequant_outside_kernel(f, ctx)
    if (du.kv_cache_dtype(f) != 'fp8' and not kv_cfg.startswith('fp8')
            and not dequant):
        changes.append(
            knobs.make_change('vllm_tpu.cli.kv_cache_dtype', 'fp8', cfg))
    is_dsv4 = bool(_DSV4.search(m.name or ''))
    if (m.is_mla and not is_dsv4 and
            not F.is_true(_knob(ctx, 'vllm_tpu.env.MLA_TRANSPOSE_KV_CACHE'))):
        changes.append(
            knobs.make_change('vllm_tpu.env.MLA_TRANSPOSE_KV_CACHE', 'True',
                              cfg))
    if mns and saturated:
        target = (math.ceil(mns * D36_SEQS_GROWTH / D36_SEQS_ALIGN) *
                  D36_SEQS_ALIGN)
        changes.append(
            knobs.make_change('vllm.cli.max_num_seqs', str(target), cfg))
    if not changes:
        return []
    return [
        Finding(
            detector_id='D36_hbm_limited_decode_batch',
            title=
            ('Decode batch capped by HBM while expert matmuls are weight-bound'
             ),
            phase=segmenter.DECODE,
            summary=
            (f'Grouped expert matmuls are {_pct(gmm_share):.1f}% of the'
             f' decode step with ~{per_expert:.0f} routed tokens per expert'
             f' (bucket {bucket:.0f} x DP {dp} x top-{m.num_experts_per_tok}'
             f' / {m.num_experts} experts), well below the'
             f' ~{D36_MAX_TOKENS_PER_EXPERT}-row ridge, so each step re-reads'
             f' expert weights for few tokens; {sat_txt}. Freeing HBM for KV'
             ' (higher memory utilization, fp8 KV, transposed MLA KV) lets'
             ' the decode batch grow at almost no extra GMM time.'),
            evidence=[
                Evidence('MoE GMM share', _pct(gmm_share), '%'),
                Evidence('routed tokens per expert per step', per_expert),
                Evidence('decode token bucket (mean)', bucket),
                *([Evidence('decode requests (mean)', reqs)]
                  if has_reqs else []),
                *du.top_op_evidence(f, segmenter.DECODE, oc.MOE_GMM, 2),
                du.step_evidence(f, segmenter.DECODE, sid),
            ],
            knob_changes=changes,
            risk=F.Risk.MEMORY,
            confidence=(F.Confidence.MEDIUM
                        if saturated else F.Confidence.LOW),
            est_step_fraction_saved=min(0.08, 0.3 * gmm_share),
            validation=(
                'A/B output tok/s at the target concurrency; check the engine'
                ' log for KV capacity, preemptions and HBM OOM. DeepSeek-V3 on'
                ' v7x: --gpu-memory-utilization 0.96 -> 0.98 +4.4-7.7%;'
                ' MLA_TRANSPOSE_KV_CACHE let batch go 896 -> 1024, +6.1%'
                ' (3297 -> 3511 tok/s/chip). fp8 KV is quality-gated.'),
            source_refs=[_DOC_DSV3_SERVING, _DOC_DSV3_MOE],
        )
    ]


# -----------------------------------------------------------------------------
# D37: MoE expert weights wider than 4 bits
# -----------------------------------------------------------------------------
def d37_moe_weight_precision(f: features_lib.Features,
                             ctx: context_lib.ServingContext,
                             sid: str) -> list[Finding]:
    if not _is_vllm_tpu(ctx, f) or not ctx.model.is_moe:
        return []
    weight_dtype = _knob(ctx, 'vllm_tpu.env.MOE_REQUANTIZE_WEIGHT_DTYPE')
    if weight_dtype.lower() in _FP4_ALIASES:
        return []
    # moe_weights.py asserts when requantization is disabled.
    if F.is_true(ctx.config_get('DISABLE_WEIGHT_REQUANTIZATION')):
        return []
    gmm_meta = f.meta[f.meta['op_class'] == oc.MOE_GMM]
    text = ' '.join(gmm_meta['kernel_text'].fillna(''))
    if _FP4_WEIGHT.search(text):
        return []
    operands_visible = bool(_ANY_OPERAND.search(text))
    phases = du.active_phases(f)
    shares = {
        p: features_lib.share_of(f.class_time, p, oc.MOE_GMM)
        for p in phases
    }
    if not shares:
        return []
    phase = max(shares, key=lambda p: shares[p])
    share = shares[phase]
    if share < D37_MIN_GMM_SHARE:
        return []
    cfg = ctx.serving_config
    changes = [
        knobs.make_change('vllm_tpu.env.MOE_REQUANTIZE_WEIGHT_DTYPE', 'fp4',
                          cfg)
    ]
    if _int_or(_knob(ctx, 'vllm_tpu.env.MOE_REQUANTIZE_BLOCK_SIZE'), 0) < 512:
        changes.append(
            knobs.make_change('vllm_tpu.env.MOE_REQUANTIZE_BLOCK_SIZE', '512',
                              cfg))
    if operands_visible:
        where = ' in the GMM operands'
    else:
        where = (' (GMM operand dtypes are not in the trace; based on'
                 ' MOE_REQUANTIZE_WEIGHT_DTYPE)')
    return [
        Finding(
            detector_id='D37_moe_weight_precision',
            title='MoE expert weights are wider than 4 bits',
            phase=phase,
            summary=(
                f'Grouped expert matmuls are {_pct(share):.1f}% of the {phase}'
                ' step and no fp4/int4 expert weights were found' + where +
                '. Decode GMM is weight-bandwidth bound, so fp4 weights with'
                ' 512-wide quantization blocks roughly halve expert weight'
                ' reads.'),
            evidence=[
                Evidence('MoE GMM share', _pct(share), '%'),
                *du.top_op_evidence(f, phase, oc.MOE_GMM, 2),
                du.step_evidence(f, phase, sid),
            ],
            knob_changes=changes,
            risk=F.Risk.QUALITY_GATED,
            confidence=(F.Confidence.MEDIUM
                        if operands_visible else F.Confidence.LOW),
            est_step_fraction_saved=min(0.15, 0.3 * share),
            validation=
            ('Accuracy eval before rollout (block 512 was quality-neutral in'
             ' the GPT-OSS study; smaller blocks cost TPU perf). Measured:'
             ' DeepSeek-V3 roofline +20%; Qwen3-Coder GMM 1.92x faster, +2-3%'
             ' e2e. Do not combine with DISABLE_WEIGHT_REQUANTIZATION.'),
            source_refs=[_DOC_DSV3_MOE, _DOC_GPT_OSS, _DOC_QCH],
        )
    ]


# -----------------------------------------------------------------------------
# D40: MLA attention kernel generation and config
# -----------------------------------------------------------------------------
def d40_mla_kernel_config(f: features_lib.Features,
                          ctx: context_lib.ServingContext,
                          sid: str) -> list[Finding]:
    m = ctx.model
    if not _is_vllm_tpu(ctx, f) or not m.is_mla or f.step_ops.empty:
        return []
    share = features_lib.share_of(f.class_time, segmenter.DECODE,
                                  oc.ATTENTION_KERNEL)
    if share < D40_MIN_ATTN_SHARE:
        return []
    out = []
    attn = f.step_ops[f.step_ops['op_class'] == oc.ATTENTION_KERNEL]
    names = attn['op'].fillna('') + ' ' + attn['tf_op_name'].fillna('')
    v1 = names.str.contains(_MLA_V1)
    if v1.any():
        out.append(
            Finding(
                detector_id='D40_mla_kernel_config',
                title='MLA attention runs the v1 kernel',
                phase=segmenter.DECODE,
                summary=
                (f'MLA attention ({_pct(share):.1f}% of the decode step) runs'
                 ' `MLA-RPA-*`, the v1 kernel. Current tpu_inference always'
                 ' uses the v2 kernel (batched decode, tuned block sizes,'
                 ' optional transposed KV).'),
                evidence=[
                    Evidence('attention kernel share', _pct(share), '%'),
                    Evidence('v1 kernel',
                             1,
                             detail=str(attn.loc[v1, 'op'].iloc[0])),
                    du.step_evidence(f, segmenter.DECODE, sid),
                ],
                knob_changes=[],
                risk=F.Risk.DIAGNOSTIC,
                confidence=F.Confidence.HIGH,
                est_step_fraction_saved=min(0.25, 0.5 * share),
                validation=(
                    'Re-profile on the new build. DeepSeek-V3: MLA v1 -> v2 was'
                    ' 1.34x e2e (797 -> 1077 tok/s/chip).'),
                kernel_followup=(
                    'Upgrade tpu_inference to a build with the MLA v2 kernel'
                    ' (kernels/mla/v2).'),
                source_refs=[_DOC_DSV3_SERVING],
            ))
    if _DSV4.search(m.name or ''):
        return out  # DSv4 forces packing 4 / no transpose (kv_cache_manager.py).
    cfg = ctx.serving_config
    changes = []
    notes = []
    packing = _int_or(_knob(ctx, 'vllm_tpu.env.MLA_KV_PACKING_SIZE'),
                      D40_TUNED_KV_PACKING)
    if packing < D40_TUNED_KV_PACKING:
        changes.append(
            knobs.make_change('vllm_tpu.env.MLA_KV_PACKING_SIZE',
                              str(D40_TUNED_KV_PACKING), cfg))
        notes.append(f'MLA_KV_PACKING_SIZE={packing} (tuned tables use'
                     f' {D40_TUNED_KV_PACKING})')
    no_q_quant = F.is_true(
        _knob(ctx, 'vllm_tpu.env.DISABLE_MLA_Q_ACTIVATION_QUANTIZATION'))
    if no_q_quant:
        changes.append(
            knobs.make_change(
                'vllm_tpu.env.DISABLE_MLA_Q_ACTIVATION_QUANTIZATION', 'False',
                cfg))
        notes.append('Q activation quantization is disabled (bf16 Q)')
    if changes:
        out.append(
            Finding(
                detector_id='D40_mla_kernel_config',
                title='MLA attention kernel configured off the tuned path',
                phase=segmenter.DECODE,
                summary=
                (f'MLA attention is {_pct(share):.1f}% of the decode step and '
                 + '; '.join(notes) +
                 '. The v2 kernel is tuned for fp8 Q with 32-way KV packing.'),
                evidence=[
                    Evidence('attention kernel share', _pct(share), '%'),
                    *du.top_op_evidence(f, segmenter.DECODE,
                                        oc.ATTENTION_KERNEL, 2),
                    du.step_evidence(f, segmenter.DECODE, sid),
                ],
                knob_changes=changes,
                risk=F.Risk.QUALITY_GATED if no_q_quant else F.Risk.PERF_ONLY,
                confidence=F.Confidence.MEDIUM,
                est_step_fraction_saved=min(0.1, 0.1 * len(changes) * share),
                validation=
                ('A/B decode step time; the packing changes the KV-cache'
                 ' layout, so check the tuned-params log line for a tuned'
                 ' key. DeepSeek-V3: KV packing 32 took MLA decode 481 -> 435'
                 ' µs. Re-enabling fp8 Q needs an accuracy eval.'),
                source_refs=[_DOC_DSV3_SERVING],
            ))
    return out


# -----------------------------------------------------------------------------
# D44: MoE combine collective not chunked / overlapped
# -----------------------------------------------------------------------------
def d44_moe_combine_not_chunked(f: features_lib.Features,
                                ctx: context_lib.ServingContext,
                                sid: str) -> list[Finding]:
    m = ctx.model
    fw = du.framework(ctx, f)
    if (not du.is_vllm(ctx, f) or ctx.serving_config is None or not m.is_moe
            or f.step_ops.empty):
        return []
    knob_key = ('vllm_tpu.env.VLLM_MOE_CHUNK_SIZE' if fw == 'vllm_tpu' else
                'vllm_torchtpu.env.TPU_MOE_COLLECTION_CHUNK_SIZE')
    if _int_or(_knob(ctx, knob_key), 0) > 0:
        return []
    if du.name_mask(f.step_ops, du.FUSED_EP_OP).any():
        return []  # The fused EP kernel does its own combine.
    coll = f.collectives
    if coll.empty:
        return []
    onehot = (_int_or(_knob(ctx, 'vllm_tpu.env.ONEHOT_MOE_PERMUTE_THRESHOLD'),
                      0) if fw == 'vllm_tpu' else 0)
    topk = m.num_experts_per_tok or 1
    out = []
    for phase in du.active_phases(f):
        if phase not in du.PREFILL_LIKE:
            continue
        bucket = du.stat(f, phase, 'mean_bucket')
        if onehot > 0 and bucket and bucket * topk <= onehot:
            continue  # One-hot combine path never chunks (fused_moe_gmm.py).
        c = coll[(coll['phase'] == phase)
                 & coll['moe_scope'].astype(bool)
                 & coll['collective'].isin(['all-reduce', 'reduce-scatter'])]
        rows = c['dims'].map(lambda d: (du.parse_dims(d) or [0])[0])
        c = c[rows >= D44_MIN_ROWS]
        if c.empty:
            continue
        dur = du.stat(f, phase, 'mean_dur_ns')
        share = float(c['ns_per_step'].sum()) / dur if dur else 0.0
        if share < D44_MIN_SHARE:
            continue
        top = c.sort_values('ns_per_step', ascending=False).head(2)
        out.append(
            Finding(
                detector_id='D44_moe_combine_not_chunked',
                title='MoE combine collective is not chunked',
                phase=phase,
                summary=
                ('MoE combine all-reduce / reduce-scatter on'
                 f' {int(rows.max())}-row tensors takes {_pct(share):.1f}% of'
                 f' the {phase} step with {knob_key.split(".")[-1]} unset.'
                 ' Chunking splits the gather-reduce and combine into'
                 f" {D44_CHUNK}-token pieces so each chunk's collective"
                 " overlaps the next chunk's gather-reduce."),
                evidence=[
                    Evidence('MoE combine share', _pct(share), '%'),
                    *[
                        Evidence(
                            f'{r["collective"]} {r["out_dtype"]}[{r["dims"]}]',
                            r['ns_per_step'] / 1e3,
                            ' µs/step',
                        ) for _, r in top.iterrows()
                    ],
                    du.step_evidence(f, phase, sid),
                ],
                knob_changes=[
                    knobs.make_change(knob_key, str(D44_CHUNK),
                                      ctx.serving_config)
                ],
                risk=F.Risk.PERF_ONLY,
                confidence=F.Confidence.MEDIUM,
                est_step_fraction_saved=min(0.2, 0.4 * share),
                validation=(
                    f'A/B prefill step time at {D44_CHUNK} and {2 * D44_CHUNK}.'
                    ' Chunking only applies on the GMM path when tokens exceed'
                    ' chunk x scatter-axis size. DeepSeek-V3 roofline +17-20%;'
                    ' Qwen3-Coder token reduction 1.59 -> 1 ms per block.'),
                source_refs=[_DOC_DSV3_MOE, _DOC_QCH],
            ))
    return out


ALL_DETECTORS = (
    d36_hbm_limited_decode_batch,
    d37_moe_weight_precision,
    d40_mla_kernel_config,
    d44_moe_combine_not_chunked,
)
