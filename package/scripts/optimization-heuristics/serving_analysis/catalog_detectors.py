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
"""Detectors derived from the documented TPU inference optimization catalog.

Each detector encodes one optimization that landed (with a measured gain) in
the Qwen3-Coder / Qwen3.5 / DeepSeek-V3 / GLM-5.2 / Qwen3 inference work, and
the trace or config signature that shows it is missing. Gains and caveats are
quoted from the source docs; see `source_refs` on each finding.
"""

# pylint: disable=missing-function-docstring,missing-class-docstring,g-doc-args,g-doc-return-or-yield,g-docstring-has-escape,g-long-ternary,g-explicit-length-test,g-complex-comprehension,protected-access,cell-var-from-loop,comparison-with-itself,unused-argument

import dataclasses
import math
import re

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

_DOC_QCH = 'Qwen3-Coder-480B serving headroom study'
_DOC_QWEN35 = 'Qwen3.5-397B serving study'
_DOC_Q32 = 'Qwen3 serving study on TPU v7x'

# D26 MoE combine without local top-k reduction.
D26_MIN_SHARE = 0.02
# D27 shapes sized by an over-provisioned --max-num-seqs.
D27_MIN_RATIO = 4
D27_MIN_SEQS = 64
D27_MIN_SHARE = 0.05
# D28 MoE activation not fused into the grouped matmul.
D28_MIN_SHARE = 0.01
_ACT_SCOPE = r'silu|swiglu|swi_glu|gelu|act_fn|activation'
# A bare sigmoid / logistic is also the router of sigmoid-scored MoEs
# (DeepSeek-V3, GLM), so it counts only outside router scopes and when its
# output is not the [tokens, num_experts] score matrix.
_SIGMOID_SCOPE = r'logistic|sigmoid'
_ROUTER_SCOPE = r'router|gating_network|(?:^|/)gate(?:/|$)|score'
# D29 async scheduling.
D29_MIN_IDLE = 0.03
# D30 batched ragged paged attention.
D30_MIN_ATTN_SHARE = 0.10
# D45 RPA decode KV compute chunk sized far beyond the live context.
D45_MIN_ATTN_SHARE = 0.40
D45_FAR_EXCEED_RATIO = 2.0
# Upper bound on the estimated saving: the estimate assumes the kernel's time
# scales with its compute chunk, which overstates it for memory-bound kernels.
D45_MAX_GAIN = 0.5
# `RPAd-p_64-bq_1_1-bkv_8192_8192-sw_1024`: decode RPA, KV block 8192,
# compute chunk 8192, sliding window 1024.
_RPA_DECODE_KERNEL = re.compile(
    r'RPAd-.*?bkv_(\d+)_(\d+)(?:-sw_(\d+))?(?![\w])')
_RPA_PAGE = re.compile(r'RPAd-p_(\d+)')
# D53 collective-heavy long prefill with no microbatch overlap.
D53_MIN_COLL_SHARE = 0.15
D53_MIN_BATCHED_TOKENS = 2048
# D47 small-row decode all-reduce on vLLM-TorchTPU.
D47_MIN_AR_SHARE = 0.08
D47_MAX_SMALL_ROWS = 64
_SMALL_AR_KERNEL = re.compile(r'^small_all_reduce_', re.I)
_SRC_SMALL_ALL_REDUCE = (
    'vLLM-TorchTPU small all-reduce kernel and call sites,'
    ' vllm_torchtpu/kernels/collectives/small_all_reduce.py,'
    ' vllm_torchtpu/layers/vllm/small_all_reduce_op.py')
# D48 unfused Q/K RMSNorm + partial RoPE on vLLM-TorchTPU.
D48_MIN_SHARE = 0.03
_QK_NORM_ROPE_RE = re.compile(r'q_norm|k_norm|qk_norm|rotary_emb', re.I)
_FUSED_QK_NORM_ROPE_OP = r'qk_norm_rope_gate|fused_qk_norm_rope'
# The fused kernel serves Qwen3-Next attention only. `ModelConfig.name` is the
# HF config's `_name_or_path` or `model_type`: `Qwen/Qwen3-Next-...` or
# `qwen3_next`.
_QWEN3_NEXT_MODEL = re.compile(r'qwen3[-_]?next', re.I)
_SRC_FUSED_QK_NORM_ROPE = (
    'vLLM-TorchTPU fused split + Q/K RMSNorm + RoPE + gate (Qwen3-Next'
    ' attention, <= 64 rows),'
    ' vllm_torchtpu/layers/vllm/fused_qk_norm_rope_tpu.py,'
    ' vllm_torchtpu/__init__.py (_patch_qwen3_next_fused_qk_norm_rope_gate)')
# D49 replicated latent-MoE projections on dual-core chips (Kimi-K3).
D49_MIN_SHARE = 0.08
_LATENT_PROJ_RE = re.compile(
    r'latent_proj|down_proj_latent|up_proj_latent'
    r'|routed_expert_(?:down|up)_proj',
    re.I,
)
_SRC_LATENT_PROJ_INTRA_CHIP = (
    'vLLM-TorchTPU intra-chip TP for the latent-MoE projections,'
    ' vllm_torchtpu/layers/vllm/latent_proj_intra_chip.py')
# D50 Pallas small all-reduce form A/B.
D50_MIN_SHARE = 0.08
_SMALL_AR_FORM = re.compile(r'^small_all_reduce_(oneshot|twoshot)', re.I)
# D52 vLLM-TorchTPU moe_tp_decode <= 64-row kernel cliff on decode steps.
D52_MIN_GMM_SHARE = 0.20
D52_MOE_TP_DECODE_MAX_ROWS = limits.MOE_TP_DECODE_MAX_ROWS
# Fraction of decode step time in steps above the row limit before EP is
# proposed: below it, most decode steps still run `moe_tp_decode`.
D52_MIN_DECODE_TIME_OVER_ROWS = 0.5
_SRC_MOE_TP_DECODE = ('vLLM-TorchTPU fused TP MoE decode kernel (<= 64 rows),'
                      ' vllm_torchtpu/kernels/moe_tp_decode/')
# D31 MoE GMM tile misalignment under TP.
D31_TILE_ALIGN = 256
D31_MIN_GMM_SHARE = 0.10
# D32 excess collectives per layer.
D32_MIN_COLL_PER_LAYER = 4.0
D32_MIN_COLL_SHARE = 0.10
# Expert-parallel layers add token dispatch and combine, so one more per layer
# is expected there.
D32_EP_EXTRA_PER_LAYER = 1.0
# D33 small KV-cache block_size.
D33_MIN_ATTN_SHARE = 0.10
# D34 speculative draft head overhead.
D34_MIN_DRAFT_SHARE = 0.05
_DRAFT_SCOPE = (r'mtp_block|eagle_draft|medusa_head|spec_decode_draft')
# D35 unfused single-op jit glue programs.
D35_MIN_EAGER_FAMILIES = 2
D35_MIN_EAGER_RATIO = 0.10
_EAGER_PRIMITIVE_RE = re.compile(
    r'^jit__?(scatter|select_n|convert_element_type|less|greater|equal|add|sub|'
    r'mul|div|slice|dynamic_slice|dynamic_update_slice|reshape|transpose|'
    r'broadcast_in_dim|concatenate|pad|gather|iota|reduce)\b')


def _pct(x: float) -> float:
    return 100.0 * float(x)


def _share_of_ns(f: features_lib.Features, phase: str,
                 ns_per_step: float) -> float:
    dur = du.stat(f, phase, 'mean_dur_ns')
    return float(ns_per_step) / dur if dur else 0.0


def _dims(d: str) -> list[int]:
    out = []
    for x in str(d or '').split('x'):
        if not x.isdigit():
            return []
        out.append(int(x))
    return out


# -----------------------------------------------------------------------------
# D26: MoE combine collective carries the top-k axis (no local reduction)
# -----------------------------------------------------------------------------
def d26_moe_combine_without_local_reduction(f: features_lib.Features,
                                            ctx: context_lib.ServingContext,
                                            sid: str) -> list[Finding]:
    """All-reduce / all-to-all over [T, top_k, H] instead of [T, H]."""
    m = ctx.model
    topk, hidden = m.num_experts_per_tok, m.hidden_size
    if not m.is_moe or not topk or topk < 2 or not hidden or f.step_ops.empty:
        return []
    coll = f.collectives
    if coll.empty:
        return []
    tp = ctx.tp or 1
    out = []

    def carries_topk(d: str) -> bool:
        dims = _dims(d)
        return (len(dims) >= 2 and dims[-1] in (hidden, hidden // tp)
                and topk in dims[:-1])

    def flattened_topk(d: str, ref_dims: set[tuple[int, ...]]) -> bool:
        dims = _dims(d)
        if len(dims) != 2 or dims[1] not in (hidden, hidden // tp):
            return False
        n = dims[0]
        return (n % topk == 0 and (n // topk, dims[1]) in ref_dims
                and tuple(dims) not in ref_dims)

    for phase in du.active_phases(f):
        c = coll[(coll['phase'] == phase)
                 & coll['moe_scope']
                 & coll['collective'].isin([
                     'all-reduce',
                     'all-to-all',
                     'ragged-all-to-all',
                     'reduce-scatter',
                 ])]
        if c.empty:
            continue
        # Flattened [T*top_k, H]: anchor on a same-phase non-MoE [T, H]
        # collective (e.g. the attention o-proj all-reduce) so T is taken from
        # the trace. If a non-MoE collective also has the [T*top_k, H] shape,
        # T*top_k is just another token bucket, so it is not evidence.
        pc = coll[(coll['phase'] == phase) & ~coll['moe_scope'].astype(bool)]
        ref_dims = {tuple(_dims(d)) for d in pc['dims']}
        hit = c[[
            carries_topk(d) or flattened_topk(d, ref_dims) for d in c['dims']
        ]]
        if hit.empty:
            continue
        share = _share_of_ns(f, phase, hit['ns_per_step'].sum())
        if share < D26_MIN_SHARE:
            continue
        kinds = sorted(set(hit['collective']))
        out.append(
            Finding(
                detector_id='D26_moe_combine_no_local_reduction',
                title='MoE combine collective carries the top-k axis',
                phase=phase,
                summary=
                (f'{"/".join(kinds)} inside the MoE block move tensors with a'
                 f' top-{topk} axis ({_pct(share):.1f}% of the {phase} step):'
                 ' the per-token sum over the selected experts happens after'
                 f' the collective, so it moves {topk}x the bytes. Reduce over'
                 ' top-k locally first (TP: sum before the all-reduce; EP:'
                 ' local reduce instead of the all-to-all unpermute).'),
                evidence=[
                    Evidence(
                        f'{r["collective"]} {r["out_dtype"]}[{r["dims"]}]',
                        r['ns_per_step'] / 1e3, ' µs/step')
                    for _, r in (hit.sort_values(
                        'ns_per_step', ascending=False).head(3).iterrows())
                ] + [du.step_evidence(f, phase, sid)],
                knob_changes=[],
                risk=F.Risk.DIAGNOSTIC,
                confidence=F.Confidence.MEDIUM,
                est_step_fraction_saved=share * (1 - 1 / topk),
                kernel_followup=
                ('vLLM-TPU: use a tpu-inference build with TP local reduction '
                 '(PR 1516) / EP local reduction (PR 1562); Qwen3-Coder went '
                 '19.7% -> 33.0% -> 41.2% of B200. Other frameworks: sum over '
                 'top-k before the psum / all-to-all.'),
                source_refs=[_DOC_QCH],
            ))
    return out


# -----------------------------------------------------------------------------
# D27: per-step shapes sized by an over-provisioned --max-num-seqs
# -----------------------------------------------------------------------------
def d27_max_num_seqs_inflation(f: features_lib.Features,
                               ctx: context_lib.ServingContext,
                               sid: str) -> list[Finding]:
    """Ops whose shapes carry max_num_seqs far above the live concurrency."""
    mns, conc = ctx.max_num_seqs, ctx.concurrency
    if not mns or not conc or mns < D27_MIN_SEQS or f.step_ops.empty:
        return []
    per_rank = max(1, conc // max(ctx.dp or 1, 1))
    if mns < D27_MIN_RATIO * per_rank:
        return []
    m = ctx.model
    if not (m.hidden_size and m.head_dim):
        # Without the model's dimensions, a max_num_seqs-sized axis cannot be
        # told from a model one (Gemma's head_dim is 256): say nothing.
        return []
    model_dims = {
        m.hidden_size,
        m.head_dim,
        m.moe_intermediate_size,
        m.num_experts,
        m.num_attention_heads,
        m.num_kv_heads,
    }
    if mns in model_dims:
        return []  # Ambiguous: the dim may be a model dimension.
    hits = features_lib.ops_with_dim(f.step_ops, f.meta, mns, f.seg.steps)
    if hits.empty:
        return []
    # MoE ops carry [tokens * top_k] rows; they are sized by the token bucket.
    moe_keys = set(
        zip(
            f.step_ops.loc[f.step_ops['moe_scope'].fillna(False).astype(bool),
                           'module'],
            f.step_ops.loc[f.step_ops['moe_scope'].fillna(False).astype(bool),
                           'op'],
        ))
    hits = hits[[k not in moe_keys for k in zip(hits['module'], hits['op'])]]
    hits = hits[~hits['op_class'].isin([
        oc.MOE_GMM,
        oc.MOE_GMM_ZERO_INIT,
        oc.MOE_PERMUTE,
        oc.MOE_TOPK_SORT,
        oc.MOE_OTHER,
    ])]
    if hits.empty:
        return []
    topk = m.num_experts_per_tok or 1
    steps = f.seg.steps
    out = []
    for phase in du.active_phases(f):
        buckets = {
            int(b)
            for b in steps.loc[steps['phase'] == phase, 'bucket'].dropna()
        }
        if mns in buckets or mns in {b * topk for b in buckets}:
            continue  # The axis is explained by the token bucket, not seqs.
        h = hits[hits['phase'] == phase]
        if h.empty:
            continue
        share = _share_of_ns(f, phase, h['ns_per_step'].sum())
        if share < D27_MIN_SHARE:
            continue
        # The server has to keep serving the busiest benchmark point, so size
        # the proposal from the highest benchmarked concurrency per DP rank,
        # not from the concurrency this trace was captured at.
        peak = max((ctx.benchmark_concurrencies or ()) + (conc, ))
        peak_per_rank = max(1, peak // max(ctx.dp or 1, 1))
        target = 1
        while target < peak_per_rank:
            target *= 2
        target = max(target, 16)
        changes = ([
            knobs.make_change('vllm.cli.max_num_seqs', str(target),
                              ctx.serving_config)
        ] if target < mns else [])
        keep_note = ('' if changes else
                     (f' It stays: the busiest benchmark point needs {peak}'
                      f' concurrent requests (~{peak_per_rank} per DP rank).'))
        linear = (h['tf_op_name'].fillna('').str.contains(
            r'gdn|gated_delta|linear_attn|mamba|ssm|conv1d', case=False).any())
        keys = set(zip(h['module'], h['op']))
        so = f.step_ops[f.step_ops['phase'] == phase]
        hit_ops = so[[k in keys for k in zip(so['module'], so['op'])]]
        gdn_txt = (
            ' This includes linear-attention/GDN state loops, whose trip count'
            ' scales with the padded batch.' if linear else '')
        out.append(
            Finding(
                detector_id='D27_max_num_seqs_inflation',
                title=(
                    'Per-step shapes sized by --max-num-seqs, not live requests'
                ),
                phase=phase,
                summary=
                (f'Ops with a {mns}-sized axis (--max-num-seqs={mns}) take '
                 f'{_pct(share):.1f}% of the {phase} step, while the benchmark '
                 f'runs {conc} concurrent requests (~{per_rank} per DP rank). '
                 'Buffers, loops and state updates are padded to max_num_seqs.'
                 + gdn_txt + keep_note),
                evidence=[
                    Evidence('share of ops with the max_num_seqs axis',
                             _pct(share), '%'),
                    Evidence('max_num_seqs', mns),
                    Evidence('concurrency per DP rank', per_rank),
                    Evidence('peak benchmark concurrency per DP rank',
                             peak_per_rank),
                    *du.base_op_evidence(f, phase, hit_ops, 3),
                    du.step_evidence(f, phase, sid),
                ],
                knob_changes=changes,
                risk=F.Risk.PERF_ONLY if changes else F.Risk.DIAGNOSTIC,
                confidence=F.Confidence.MEDIUM,
                est_step_fraction_saved=(0.5 * share * (1 - target / mns)
                                         if changes else 0.0),
                validation=
                ('Keep max_num_seqs >= the peak concurrency per DP rank or '
                 'requests will queue. Qwen3.5-397B: GDN inter-chunk loop ran '
                 '640 (512 seqs + 128) iterations instead of 128, 7.9 ms/layer '
                 '(35%).'),
                source_refs=[_DOC_QWEN35],
            ))
    return out


def _last_out_dim(f: features_lib.Features, ops: pd.DataFrame) -> pd.Series:
    """Last output dimension of each op in `ops` (0 if unknown)."""
    last = {
        (m, o): d[-1]
        for m, o, d in zip(f.meta['module'], f.meta['op'], f.meta['out_dims'])
        if d
    }
    return pd.Series([last.get(k, 0) for k in zip(ops['module'], ops['op'])],
                     index=ops.index)


# -----------------------------------------------------------------------------
# D28: MoE activation (SiLU * up) not fused into the grouped matmul
# -----------------------------------------------------------------------------
def d28_moe_activation_unfused(f: features_lib.Features,
                               ctx: context_lib.ServingContext,
                               sid: str) -> list[Finding]:
    if not ctx.model.is_moe or f.step_ops.empty:
        return []
    so = f.step_ops
    out = []
    for phase in du.active_phases(f):
        g = so[(so['phase'] == phase)
               & so['moe_scope'].fillna(False).astype(bool)
               & so['op_class'].isin([oc.MOE_OTHER, oc.OTHER])]
        if g.empty:
            continue
        tf = g['tf_op_name'].fillna('').astype(str)
        sigmoid = tf.str.contains(
            _SIGMOID_SCOPE,
            case=False) & ~tf.str.contains(_ROUTER_SCOPE, case=False)
        shard_inter = None
        if ctx.model.moe_intermediate_size and ctx.tp and (ctx.ep or 1) <= 1:
            shard_inter = ctx.model.moe_intermediate_size // ctx.tp
        if ctx.model.num_experts and shard_inter != ctx.model.num_experts:
            sigmoid &= _last_out_dim(f, g) != ctx.model.num_experts
        act = g[(tf.str.contains(_ACT_SCOPE, case=False)
                 | sigmoid).astype(bool)]
        share = du.share(f, phase, act)
        if share < D28_MIN_SHARE:
            continue
        out.append(
            Finding(
                detector_id='D28_moe_activation_unfused',
                title=
                ('MoE activation runs as a separate pass between expert matmuls'
                 ),
                phase=phase,
                summary=
                ('Standalone activation fusions (SiLU/GELU x up) inside the MoE'
                 f' block take {_pct(share):.1f}% of the {phase} step. They read'
                 ' and write the full [tokens x top_k, intermediate] tensor'
                 ' between the gate/up and down grouped matmuls.'),
                evidence=[
                    Evidence('activation share', _pct(share), '%'),
                    *du.base_op_evidence(f, phase, act, 3),
                    du.step_evidence(f, phase, sid),
                ],
                knob_changes=[],
                risk=F.Risk.DIAGNOSTIC,
                confidence=F.Confidence.MEDIUM,
                est_step_fraction_saved=0.8 * share,
                kernel_followup=
                ('Fuse the activation into the gate/up GMM epilogue. vLLM-TPU: '
                 'tpu-inference PR 1928 (Qwen3-Coder activation 0.358 -> 0.057 '
                 'ms per block).'),
                source_refs=[_DOC_QCH],
            ))
    return out


# -----------------------------------------------------------------------------
# D29: vLLM without --async-scheduling while the device idles between steps
# -----------------------------------------------------------------------------
def d29_async_scheduling_off(f: features_lib.Features,
                             ctx: context_lib.ServingContext,
                             sid: str) -> list[Finding]:
    logged = context_lib.async_scheduling_from_log(ctx.server_log)
    if not du.is_vllm(ctx, f) or (ctx.serving_config is None
                                  and logged is None):
        return []
    cur = knobs.current_value('vllm.cli.async_scheduling', ctx.serving_config)
    # Recent vLLM (e.g. 0.22+) auto-enables async scheduling without the flag;
    # trust the server log over the (absent) CLI flag.
    enabled = logged.enabled if logged is not None else F.is_true(cur)
    if enabled:
        return []
    explicit_off = bool(str(cur or '').strip())
    idle = du.stat(f, segmenter.DECODE, 'idle_frac')
    if math.isnan(idle) or idle < D29_MIN_IDLE:  # NaN-safe.
        return []
    stuck = logged is not None and (logged.blocked or F.is_true(cur))
    evidence = [Evidence('decode idle fraction', _pct(idle), '%')]
    if logged is not None:
        evidence.append(
            Evidence('server log', 'async scheduling off', '', logged.line))
    evidence.append(du.step_evidence(f, segmenter.DECODE, sid))
    if logged is not None and logged.blocked:
        why = ('the server log shows vLLM turned async scheduling off for an'
               ' incompatible option (speculative-decoding method, executor'
               ' backend, ...); change that option first, since vLLM rejects'
               ' --async-scheduling next to most of them')
    elif logged is not None and stuck:
        why = ('the server log shows async scheduling off even though'
               ' --async-scheduling is set')
    elif logged is not None:
        why = 'the server log shows async scheduling off'
    elif explicit_off:
        why = '--async-scheduling is explicitly disabled'
    else:
        why = (
            '--async-scheduling is not set (recent vLLM auto-enables it; older'
            ' releases and some executors/spec-decode methods do not - pass'
            ' --server_log to read the resolved state)')
    changes: list[F.KnobChange] = []
    if not stuck:
        changes.append(
            knobs.make_change('vllm.cli.async_scheduling', 'true',
                              ctx.serving_config))
    return [
        Finding(
            detector_id='D29_async_scheduling_off',
            title=(
                'Scheduler may run serially with the device (async scheduling)'
            ),
            phase=segmenter.DECODE,
            summary=(
                f'The TPU is idle {_pct(idle):.1f}% of each decode cycle and'
                f' {why}. Without async scheduling the scheduler for step N+1'
                ' runs after step N finishes instead of overlapping it.'),
            evidence=evidence,
            knob_changes=changes,
            risk=F.Risk.DIAGNOSTIC if stuck else F.Risk.PERF_ONLY,
            confidence=(F.Confidence.HIGH if logged is not None or explicit_off
                        else F.Confidence.LOW),
            est_step_fraction_saved=min(idle, 0.08),
            validation='A/B throughput; Qwen3 on TPU v7x: +7.7% and +2.1%.',
            source_refs=[_DOC_Q32],
        )
    ]


# -----------------------------------------------------------------------------
# D30: vLLM-TPU decode attention without the batched RPA kernel
# -----------------------------------------------------------------------------
def d30_batched_rpa(f: features_lib.Features, ctx: context_lib.ServingContext,
                    sid: str) -> list[Finding]:
    if not du.is_configured_vllm_tpu(ctx, f) or ctx.model.is_mla:
        return []
    cur = knobs.current_value('vllm_tpu.env.USE_BATCHED_RPA_KERNEL',
                              ctx.serving_config)
    if F.is_true(cur):
        return []
    share = features_lib.share_of(f.class_time, segmenter.DECODE,
                                  oc.ATTENTION_KERNEL)
    if share < D30_MIN_ATTN_SHARE:
        return []
    # tpu_inference routes head_dim 64 to ragged_paged_attention_hd64
    # regardless of USE_BATCHED_RPA_KERNEL (attention_interface.py).
    if ctx.model.head_dim == 64:
        return []
    attn = f.step_ops[f.step_ops['op_class'] == oc.ATTENTION_KERNEL]
    attn_names = attn['op'].fillna('') + ' ' + attn['tf_op_name'].fillna('')
    # Kernel scope `RPA-HD_64-bq_...`, jit `ragged_paged_attention_hd64`.
    if attn_names.str.contains(r'hd_?64', case=False, regex=True).any():
        return []
    return [
        Finding(
            detector_id='D30_batched_rpa',
            title='Decode attention without the batched RPA kernel',
            phase=segmenter.DECODE,
            summary=
            (f'Ragged paged attention is {_pct(share):.1f}% of the decode step'
             ' and USE_BATCHED_RPA_KERNEL is off. The batched kernel'
             ' processes several sequences per invocation, which helps most'
             ' for long decode (many short per-sequence blocks).'),
            evidence=[
                Evidence('attention kernel share', _pct(share), '%'),
                *du.top_op_evidence(
                    f,
                    segmenter.DECODE,
                    oc.ATTENTION_KERNEL,
                    2,
                ),
                du.step_evidence(f, segmenter.DECODE, sid),
            ],
            knob_changes=[
                knobs.make_change('vllm_tpu.env.USE_BATCHED_RPA_KERNEL',
                                  'true', ctx.serving_config)
            ],
            risk=F.Risk.PERF_ONLY,
            confidence=F.Confidence.MEDIUM,
            est_step_fraction_saved=min(0.3, 0.3 * share),
            validation=(
                'A/B output tok/s. Qwen3 on TPU v7x: batched RPA +6-10%'
                ' (1k/1k), +51% (1k/8k).'),
            source_refs=[_DOC_Q32, _DOC_QCH],
        )
    ]


# -----------------------------------------------------------------------------
# D31: MoE expert width under TP is not a multiple of 256 (MXU tile padding)
# -----------------------------------------------------------------------------
def _gmm_expert_widths(f: features_lib.Features) -> set[int]:
    """Per-device expert widths implied by the trace's GMM op names."""
    ops = f.step_ops
    if ops.empty or 'op_class' not in ops.columns or 'op' not in ops.columns:
        return set()
    names = ops.loc[ops['op_class'] == oc.MOE_GMM, 'op'].dropna().astype(str)
    widths = set()
    for name in names.unique():
        dims = {}
        for tok in name.split('.')[0].split('-'):
            m = _GMM_DIM_TOKEN.match(tok)
            if m:
                dims[m.group(1)] = int(m.group(2))
        if name.startswith('moe_tp_'):
            if 'i' in dims:
                widths.add(dims['i'])
            continue
        if 'n' in dims:
            widths.update({dims['n'], dims['n'] // 2})
        if 'k' in dims:
            widths.add(dims['k'])
    return widths


def _is_small_decode_moe_tp(f: features_lib.Features,
                            ctx: context_lib.ServingContext,
                            phase: str) -> bool:
    """Whether decode steps fit `moe_tp_decode` (<= 64 token rows per engine),
    where TP MoE beats EP. A row is a token: speculative decoding adds rows per
    sequence, and `ctx.concurrency` is split across `ctx.dp` engines."""
    if phase != segmenter.DECODE:
        return False
    if ctx.concurrency:
        seqs_per_dp = math.ceil(ctx.concurrency / max(ctx.dp or 1, 1))
        rows = seqs_per_dp * ctx.decode_tokens_per_req
        return rows <= D52_MOE_TP_DECODE_MAX_ROWS
    ops = du.phase_ops(f, phase)
    has_moe_tp_decode = (
        not ops.empty and
        ops['op'].dropna().astype(str).str.startswith('moe_tp_decode').any())
    mean_bucket = du.stat(f, phase, 'mean_bucket')
    return bool(has_moe_tp_decode and not math.isnan(mean_bucket)
                and mean_bucket <= D52_MOE_TP_DECODE_MAX_ROWS)


def d31_gmm_tile_misalignment(f: features_lib.Features,
                              ctx: context_lib.ServingContext,
                              sid: str) -> list[Finding]:
    m = ctx.model
    if not m.is_moe or not m.moe_intermediate_size or f.step_ops.empty:
        return []
    tp = ctx.tp or int(f.seg.steps.groupby(['host', 'device']).ngroups)
    if tp <= 1 or (ctx.ep or 1) > 1:
        return []
    shard_inter = m.moe_intermediate_size // tp
    if shard_inter <= 0 or shard_inter % D31_TILE_ALIGN == 0:
        return []
    widths = _gmm_expert_widths(f)
    if m.moe_intermediate_size in widths and shard_inter not in widths:
        return []
    confirmed = shard_inter in widths
    caveat = ('' if confirmed else (
        ' The per-shard width comes from the launch flags; no GMM op shape'
        ' in the trace confirms it.'))
    if du.name_mask(f.step_ops, du.FUSED_EP_OP).any():
        return []
    coll = f.collectives
    if coll.empty:
        return []
    padded = ((shard_inter + D31_TILE_ALIGN - 1) //
              D31_TILE_ALIGN) * (D31_TILE_ALIGN)
    pad_waste = (padded - shard_inter) / padded
    tp_fix = (
        'a TP degree that divides moe_intermediate_size into a multiple of 256.'
        ' Qwen3-Coder-480B: 2560 / TP8 = 320, which pads to 512 (37.5% GMM tile'
        ' padding).')
    out = []
    for phase in du.active_phases(f):
        mc = coll[(coll['phase'] == phase) & coll['moe_scope'].astype(bool)]
        if mc.empty:
            continue
        kinds = set(mc['collective'])
        if 'all-reduce' not in kinds or (
                kinds
                & {'all-to-all', 'ragged-all-to-all', 'reduce-scatter'}):
            continue
        gmm_share = features_lib.share_of(f.class_time, phase, oc.MOE_GMM)
        if gmm_share < D31_MIN_GMM_SHARE:
            continue
        # EP->TP (GMM v2) won on prefill-heavy Qwen3-Coder 8k/1k; see D24.
        if du.is_prefill_heavy(ctx):
            followup = (
                f'Choose {tp_fix} EP is not suggested: the workload is'
                ' prefill-heavy, where TP MoE was faster than EP in the'
                ' Qwen3-Coder sweeps.')
        elif _is_small_decode_moe_tp(f, ctx, phase):
            followup = (
                f'Choose {tp_fix} Full EP is not suggested while decode steps'
                f' stay within {D52_MOE_TP_DECODE_MAX_ROWS} token rows'
                ' (sequences per DP replica x tokens per sequence), where'
                ' `moe_tp_decode` beat EP16 on GLM-5.2; switch to EP (or hybrid'
                ' EP4xTP4 via `TPU_MOE_EP_TP_SIZE=4`) once decode steps exceed'
                ' it (D52).')
        else:
            followup = (
                'Switch MoE from TP to EP (--enable-expert-parallel keeps the'
                f' full {m.moe_intermediate_size} width per expert), or choose'
                f' {tp_fix}')
        out.append(
            Finding(
                detector_id='D31_gmm_tile_misalignment',
                title=(
                    'MoE expert width under TP is not a multiple of 256 (MXU'
                    ' padding)'),
                phase=phase,
                summary=(
                    f'With moe_intermediate_size={m.moe_intermediate_size}'
                    f' split across TP={tp}, each shard has expert width'
                    f' {shard_inter}, which pads to {padded} on a'
                    f' {D31_TILE_ALIGN}-aligned TPU GMM tile'
                    f' ({_pct(pad_waste):.0f}% tile waste across'
                    f' {_pct(gmm_share):.1f}% of the {phase} step).{caveat}'),
                evidence=[
                    Evidence('MoE GMM share', _pct(gmm_share), '%'),
                    Evidence(
                        'per-shard expert width (moe_intermediate_size / TP)',
                        shard_inter,
                    ),
                    Evidence('256-aligned MXU tile width', padded),
                    Evidence(
                        'GMM op shapes in the trace show this width',
                        'yes' if confirmed else 'no',
                    ),
                    du.step_evidence(f, phase, sid),
                ],
                knob_changes=[],
                risk=F.Risk.DIAGNOSTIC,
                confidence=(F.Confidence.MEDIUM
                            if confirmed else F.Confidence.LOW),
                est_step_fraction_saved=0.5 * gmm_share * pad_waste,
                kernel_followup=followup,
                source_refs=[_DOC_QCH, du.DOC_GLM52_SERVING_OPT],
            ))
    return out


# -----------------------------------------------------------------------------
# D32: more collectives per layer than expected for TP/EP sharding
# -----------------------------------------------------------------------------
def d32_excess_collectives_per_layer(f: features_lib.Features,
                                     ctx: context_lib.ServingContext,
                                     sid: str) -> list[Finding]:
    layers = ctx.model.num_layers
    if not layers or layers < 4 or f.step_ops.empty:
        return []
    if (f.step_ops['tf_op_name'].fillna('').astype(str).str.contains(
            r'/while/body/', case=False, regex=True).any()):
        return []
    coll = f.collectives
    if coll.empty:
        return []
    ep = bool(ctx.ep and ctx.ep > 1)
    limit = D32_MIN_COLL_PER_LAYER + (D32_EP_EXTRA_PER_LAYER if ep else 0.0)
    out = []
    for phase in du.active_phases(f):
        c = coll[coll['phase'] == phase]
        if c.empty:
            continue
        # Exclude weight-sized all-gathers covered by D7_decode_weight_allgather.
        c_act = c[~((c['collective'] == 'all-gather')
                    & (c['dims'].map(du.n_elems) >= du.MIN_WEIGHT_AG_ELEMS)
                    ).astype(bool)]
        if c_act.empty:
            continue
        per_layer = float(c_act['per_step'].sum()) / layers
        cshare = features_lib.share_of(f.class_time, phase, oc.COLLECTIVE)
        if per_layer < limit or cshare < D32_MIN_COLL_SHARE:
            continue
        by_kind = (c_act.groupby('collective')['per_step'].sum() /
                   layers).to_dict()
        detail = ', '.join(f'{k}: {v:.1f}/layer'
                           for k, v in sorted(by_kind.items()))
        out.append(
            Finding(
                detector_id='D32_excess_collectives_per_layer',
                title=
                ('More collectives per layer than expected for TP/EP sharding'
                 ),
                phase=phase,
                summary=(
                    f'The {phase} step executes {per_layer:.1f} activation'
                    f' collectives per layer across {layers} layers ({detail}),'
                    f' costing {_pct(cshare):.1f}% of the step. Standard TP'
                    ' layers need 2-3 collectives/layer (1 attention o-proj +'
                    ' 1-2 FFN/MoE); expert parallelism adds token dispatch and'
                    ' combine, so the EP threshold is one higher. Extra'
                    ' collectives indicate GSPMD resharding from mismatched'
                    ' weight or KV sharding specs.'),
                evidence=[
                    Evidence('activation collectives per layer',
                             per_layer,
                             detail=detail),
                    Evidence('collective share', _pct(cshare), '%'),
                    du.step_evidence(f, phase, sid),
                ],
                knob_changes=[],
                risk=F.Risk.DIAGNOSTIC,
                confidence=F.Confidence.MEDIUM,
                est_step_fraction_saved=0.3 * cshare,
                kernel_followup=(
                    'Inspect the extra collective ops in the trace viewer and'
                    ' verify PartitionSpec annotations on QKV, o_proj, and MoE'
                    ' gate/up/down weights so GSPMD does not insert implicit'
                    ' all-gather/all-reduce reshards.'),
                source_refs=[_DOC_QCH, _DOC_QWEN35],
            ))
    return out


# -----------------------------------------------------------------------------
# D33: small KV-cache page size (--block-size) in Pallas attention
# -----------------------------------------------------------------------------
def d33_kv_cache_block_size(f: features_lib.Features,
                            ctx: context_lib.ServingContext,
                            sid: str) -> list[Finding]:
    if not du.is_configured_vllm_tpu(ctx, f) or ctx.model.is_mla:
        return []
    bs = du.knob_value(ctx, 'vllm.cli.block_size')
    bs_int = int(bs) if bs.isdigit() else None
    max_len = ctx.max_model_len or 0
    # In tpu_inference/layers/vllm/backends/flash_attn.py:get_page_size, when
    # max_model_len > 8192 and --block-size is unset, page_size defaults to 16.
    if bs_int is not None and bs_int >= 128:
        return []
    if bs_int is None and max_len <= 8192:
        return []
    share = features_lib.share_of(f.class_time, segmenter.DECODE,
                                  oc.ATTENTION_KERNEL)
    if share < D33_MIN_ATTN_SHARE:
        return []
    changes = [
        knobs.make_change('vllm.cli.block_size', '256', ctx.serving_config)
    ]
    if du.knob_float(ctx, 'vllm.cli.gpu_memory_utilization', 0.92) < 0.90:
        changes.append(
            knobs.make_change('vllm.cli.gpu_memory_utilization', '0.95',
                              ctx.serving_config))
    if bs_int is not None:
        why = f'--block-size={bs_int} is small'
    else:
        why = (
            f'--block-size is unset with max_model_len={max_len} (>8192, where '
            'PallasAttentionBackend.get_page_size falls back to page_size=16)')
    return [
        Finding(
            detector_id='D33_kv_cache_block_size',
            title='Small KV-cache page size (--block-size) in Pallas attention',
            phase=segmenter.DECODE,
            summary=
            (f'Ragged paged attention is {_pct(share):.1f}% of the decode step'
             f' and {why}. Larger KV-cache pages (128-256) shrink block_tables'
             ' in SMEM, avoid SREG spills, and reduce page-gather overhead.'),
            evidence=[
                Evidence('attention kernel share', _pct(share), '%'),
                Evidence(
                    'configured block_size',
                    bs_int or 16,
                    detail=('explicit' if bs_int is not None else
                            'flash_attn.py default for max_model_len > 8192')),
                du.step_evidence(f, segmenter.DECODE, sid),
            ],
            knob_changes=changes,
            risk=F.Risk.PERF_ONLY,
            confidence=(F.Confidence.MEDIUM
                        if bs_int is not None else F.Confidence.LOW),
            est_step_fraction_saved=min(0.05, 0.15 * share),
            validation=(
                'A/B output tok/s and check for VMEM OOM at 256 (use 128 if'
                ' VMEM is tight). Qwen3 on TPU v7x: --block-size=256 +2.0%,'
                ' --gpu-memory-utilization=0.95 +4.4%.'),
            source_refs=[_DOC_Q32],
        )
    ]


# -----------------------------------------------------------------------------
# D46: batched decode attention, whose decode blocks no knob tunes
# -----------------------------------------------------------------------------
# Models whose RPA v3 (FLASH_ATTN) kernels failed to compile: the finding is
# reported without a change. Name pattern -> what failed.
_RPA_V3_KNOWN_FAILURES = {
    r'gemma-?4': ('Gemma-4-31B failed to compile with FLASH_ATTN on vLLM'
                  ' 0.29.0: VMEM 71.34M of 63.94M in the RPA v3 mixed kernel'),
}
# KV page size when the server does not set --block-size.
D46_DEFAULT_PAGE = 128

_GMM_DIM_TOKEN = re.compile(r'^([kni])_(\d+)$')


def d46_batched_rpa_decode_untunable(f: features_lib.Features,
                                     ctx: context_lib.ServingContext,
                                     sid: str) -> list[Finding]:
    if (not du.is_torchtpu(ctx, f) or ctx.serving_config is None
            or ctx.model.is_mla or ctx.model.head_dim == 64
            or not ctx.max_model_len):
        return []
    backend = knobs.current_value('vllm.cli.attention_backend',
                                  ctx.serving_config)
    if str(backend or '').upper() != 'CUSTOM':
        return []
    share = features_lib.share_of(f.class_time, segmenter.DECODE,
                                  oc.ATTENTION_KERNEL)
    if share < D45_MIN_ATTN_SHARE:
        return []
    page = du.knob_int(ctx, 'vllm.cli.block_size', D46_DEFAULT_PAGE)
    # The stock RPA v3 decode branch sets block = chunk = max_model_len.
    stock = [
        RpaDecodeKernel('global', ctx.max_model_len, ctx.max_model_len, None,
                        page)
    ]
    if ctx.model.sliding_window:
        stock.append(
            RpaDecodeKernel('sliding', ctx.max_model_len, ctx.max_model_len,
                            int(ctx.model.sliding_window), page))
    context_len = (ctx.input_len + ctx.output_len
                   if ctx.input_len and ctx.output_len else None)
    reasons = [
        (k, r) for k in stock
        for r in rpa_decode_oversize_reasons(k, ctx.max_model_len, context_len)
    ]
    sizes = rpa_decode_block_sizes(stock, context_len)
    if not reasons or not sizes:
        return []
    failed = next((why for pat, why in _RPA_V3_KNOWN_FAILURES.items()
                   if re.search(pat, ctx.model.name or '', re.I)), None)
    waste = max(rpa_decode_waste(k, context_len) for k in stock)
    changes = ([] if failed else [
        knobs.make_change('vllm.cli.attention_backend', 'FLASH_ATTN',
                          ctx.serving_config),
        knobs.make_change('vllm_torchtpu.env.TPU_RPA_DECODE_BLOCK_SIZES',
                          sizes, ctx.serving_config),
    ])
    return [
        Finding(
            detector_id='D46_batched_rpa_decode_untunable',
            title='Batched decode attention has no decode-block knob',
            phase=segmenter.DECODE,
            summary=
            (f'Attention is {_pct(share):.1f}% of the decode step on the'
             ' batched RPA kernel (--attention-backend CUSTOM), which no knob'
             ' tunes. The RPA v3 kernel (FLASH_ATTN) takes per-layer decode'
             f' blocks: {sizes} instead of its stock max_model_len'
             f' ({ctx.max_model_len}), because ' +
             '; '.join(f'{k.name}: {r}' for k, r in reasons) + '.' +
             (f' Not proposed: {failed}.' if failed else
              (' Measure it: the two kernels differ in more than block size.'))
             ),
            evidence=[
                Evidence('attention kernel share', _pct(share), '%'),
                Evidence('proposed decode blocks', sizes),
                *du.top_op_evidence(f, segmenter.DECODE, oc.ATTENTION_KERNEL,
                                    2),
                du.step_evidence(f, segmenter.DECODE, sid),
            ],
            knob_changes=changes,
            risk=F.Risk.PERF_ONLY if changes else F.Risk.DIAGNOSTIC,
            confidence=F.Confidence.LOW,
            est_step_fraction_saved=(share * min(waste, D45_MAX_GAIN)
                                     if changes else 0.0),
            validation=(
                'A/B against the CUSTOM baseline at every benchmark point;'
                ' check greedy outputs match, and that the server compiles (RPA'
                ' v3 uses more VMEM than the batched kernel).'),
            kernel_followup=(None if changes else
                             'Tune the batched RPA kernel itself.'),
        )
    ]


# -----------------------------------------------------------------------------
# D34: speculative draft head (MTP / EAGLE / Medusa) overhead inside a step
# -----------------------------------------------------------------------------
def d34_speculative_draft_overhead(f: features_lib.Features,
                                   ctx: context_lib.ServingContext,
                                   sid: str) -> list[Finding]:
    del ctx
    if f.step_ops.empty:
        return []
    out = []
    for phase in du.active_phases(f):
        g = du.phase_ops(f, phase)
        if g.empty:
            continue
        draft = g[g['tf_op_name'].fillna('').astype(str).str.contains(
            _DRAFT_SCOPE, case=False, regex=True).astype(bool)]
        if draft.empty:
            continue
        share = du.share(f, phase, draft)
        if share < D34_MIN_DRAFT_SHARE:
            continue
        dur_ms = du.stat(f, phase, 'mean_dur_ns') / 1e6
        draft_ms = share * dur_ms
        breakeven = share / max(1.0 - share, 1e-6)
        by_cls = draft.groupby('op_class')['self_ns'].sum() / max(
            float(draft['self_ns'].sum()), 1.0)
        top_cls = ', '.join(
            f'{k}: {_pct(v):.0f}%'
            for k, v in by_cls.sort_values(ascending=False).head(3).items())
        out.append(
            Finding(
                detector_id='D34_speculative_draft_overhead',
                title='Speculative draft head overhead',
                phase=phase,
                summary=
                ('Speculative draft ops (MTP / EAGLE / Medusa) take'
                 f' {_pct(share):.1f}% ({draft_ms:.2f} ms) of the'
                 f' {dur_ms:.2f} ms {phase} step ({top_cls}). Speculative'
                 ' decoding reduces net latency per token only when the mean'
                 ' accepted draft tokens per step exceeds'
                 f' {breakeven:.2f}; otherwise the draft compute and the KV /'
                 ' MoE bandwidth inside the draft head are pure overhead.'),
                evidence=[
                    Evidence('speculative draft share of step',
                             _pct(share),
                             '%',
                             detail=f'{draft_ms:.2f} ms/step ({top_cls})'),
                    Evidence('break-even accepted draft tokens/step',
                             round(breakeven, 3)),
                    *du.base_op_evidence(f, phase, draft, 3),
                    du.step_evidence(f, phase, sid),
                ],
                knob_changes=[],
                risk=F.Risk.DIAGNOSTIC,
                confidence=F.Confidence.HIGH,
                est_step_fraction_saved=0.0,
                kernel_followup=(
                    'Compare the break-even above with the acceptance rate in'
                    ' the server\'s speculative-decoding metrics; below it,'
                    ' fewer speculative tokens or no drafting serves faster.'),
            ))
    return out


# -----------------------------------------------------------------------------
# D35: unfused single-op jit glue programs outside the step executable
# -----------------------------------------------------------------------------
def d35_unfused_glue_programs(f: features_lib.Features,
                              ctx: context_lib.ServingContext,
                              sid: str) -> list[Finding]:
    del ctx
    if f.seg.steps.empty or f.seg.families.empty:
        return []
    ns = f.seg.families[~f.seg.families['family'].isin(f.seg.step_families)]
    if ns.empty:
        return []
    eager = ns[ns['family'].astype(str).str.match(_EAGER_PRIMITIVE_RE).astype(
        bool)]
    if len(eager) < D35_MIN_EAGER_FAMILIES:
        return []
    step_fams = f.seg.families[f.seg.families['family'].isin(
        f.seg.step_families)]
    step_total_ns = float(step_fams['total_ns'].sum())
    eager_total_ns = float(eager['total_ns'].sum())
    if step_total_ns <= 0 or eager_total_ns / step_total_ns < D35_MIN_EAGER_RATIO:
        return []
    ratio = eager_total_ns / step_total_ns
    share_of_all = eager_total_ns / (step_total_ns + eager_total_ns)
    phases = du.active_phases(f)
    phase = phases[0] if len(phases) == 1 else 'all'
    names = ', '.join(f'`{r["family"]}` ({r["total_ns"] / 1e6:.1f} ms total)'
                      for _, r in eager.head(5).iterrows())
    ev = [
        Evidence('eager primitive jit families', len(eager), detail=names),
        Evidence('eager glue / step device time ratio', _pct(ratio), '%'),
    ]
    if phase in phases:
        ev.append(du.step_evidence(f, phase, sid))
    return [
        Finding(
            detector_id='D35_unfused_glue_programs',
            title=
            ('Unfused single-op jit glue programs outside the step executable'
             ),
            phase=phase,
            summary=
            (f'{len(eager)} single-primitive JAX executables ({names}) run'
             f' outside `{", ".join(f.seg.step_families)}` and add'
             f' {_pct(ratio):.1f}% extra device time'
             f' ({_pct(share_of_all):.1f}% of total module time) plus separate'
             ' PjitFunction / shard_args host dispatches per call. Wrap these'
             ' scratchpad/mask/dtype ops inside the jitted step function so'
             ' XLA fuses them.'),
            evidence=ev,
            knob_changes=[],
            risk=F.Risk.DIAGNOSTIC,
            confidence=F.Confidence.HIGH,
            est_step_fraction_saved=share_of_all,
            kernel_followup=
            ('Move the pre/post-step array updates (scatter, select_n, '
             'convert_element_type, comparisons) inside the jitted step '
             'function (e.g. `insert_with_scratchpad`) to eliminate separate '
             'PjitFunction dispatches and DevicePutWithSharding fallbacks.'),
        )
    ]


# -----------------------------------------------------------------------------
# D45: RPA decode KV block / compute chunk sized to max_model_len
# -----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class RpaDecodeKernel:
    name: str
    block: int  # KV block (tokens DMA'd per grid step).
    chunk: int  # Compute chunk (tokens each step computes over).
    window: int | None  # Sliding window, if the kernel is a sliding layer.
    page: int | None = None  # KV page size in tokens.


def parse_rpa_decode_kernel(name: str) -> RpaDecodeKernel | None:
    m = _RPA_DECODE_KERNEL.search(name or '')
    if not m:
        return None
    page = _RPA_PAGE.search(name or '')
    return RpaDecodeKernel(
        name=m.group(0),
        block=int(m.group(1)),
        chunk=int(m.group(2)),
        window=int(m.group(3)) if m.group(3) else None,
        page=int(page.group(1)) if page else None,
    )


def rpa_decode_oversize_reasons(k: RpaDecodeKernel, max_model_len: int | None,
                                context_len: int | None) -> list[str]:
    """Why a decode RPA kernel's compute chunk is oversized ([] if it is not).

  The stock vllm_torchtpu v7 decode branch sets block = chunk =
  max_model_len, so every decode step computes over max_model_len slots and
  sliding-window layers cannot skip blocks outside their window.

  Args:
    k: Parsed kernel.
    max_model_len: Engine max_model_len, if known.
    context_len: Benchmark input + output length, if known.
  """
    out = []
    if max_model_len and k.chunk >= max_model_len:
        out.append(f'compute chunk {k.chunk} >= max_model_len {max_model_len}')
    if k.window and k.chunk >= D45_FAR_EXCEED_RATIO * k.window:
        out.append(
            f'compute chunk {k.chunk} is {k.chunk / k.window:.1f}x the sliding'
            f' window {k.window}')
    if context_len and k.chunk >= D45_FAR_EXCEED_RATIO * context_len:
        out.append(
            f'compute chunk {k.chunk} is {k.chunk / context_len:.1f}x the'
            f' longest context {context_len}')
    return out


def rpa_decode_block_oversized(
    kernel_shares: dict[str, float],
    attn_share: float,
    max_model_len: int | None,
    context_len: int | None,
) -> tuple[float, list[tuple[RpaDecodeKernel, float, list[str]]]]:
    """Core D45 rule over (kernel base name -> share of the decode step).

  Args:
    kernel_shares: Decode-phase op base names and their step-time shares.
    attn_share: Attention-kernel share of the decode step.
    max_model_len: Engine max_model_len, if known.
    context_len: Benchmark input + output length, if known.

  Returns:
    (share of the oversized kernels, [(kernel, share, reasons)]); empty when
    attention is below D45_MIN_ATTN_SHARE or no kernel is oversized.
  """
    if attn_share < D45_MIN_ATTN_SHARE:
        return 0.0, []
    hits = []
    for name, share in sorted(kernel_shares.items(), key=lambda x: -x[1]):
        k = parse_rpa_decode_kernel(name)
        if k is None:
            continue
        reasons = rpa_decode_oversize_reasons(k, max_model_len, context_len)
        if reasons:
            hits.append((k, share, reasons))
    return sum(s for _, s, _ in hits), hits


def _round_up(n: int, page: int) -> int:
    return -(-n // page) * page


def rpa_decode_block_sizes(kernels: list[RpaDecodeKernel],
                           context_len: int | None) -> str:
    """TPU_RPA_DECODE_BLOCK_SIZES for oversized decode kernels, or ''.

    Sizes come from the kernels' own shapes and the workload, not from a
    tuned table: a sliding layer gets a block and a chunk of one window,
    rounded up to a page; a global layer keeps its block (the DMA moves only
    valid tokens) and gets the smallest page-aligned chunk that divides the
    block and covers the context. A layer kind with nothing to shrink, or
    whose page size is not in the kernel name, is left out.
    """
    entries = {}
    for k in kernels:
        if not k.page:
            continue
        if k.window:
            size = min(k.block, _round_up(k.window, k.page))
            if size < k.chunk:
                entries.setdefault('sliding', (size, size))
        elif context_len:
            need = _round_up(context_len, k.page)
            chunk = next((d for d in range(need, k.block + 1, k.page)
                          if k.block % d == 0), k.block)
            if chunk < k.chunk:
                entries.setdefault('global', (k.block, chunk))
    return ','.join(f'{name}={b}:{c}'
                    for name, (b, c) in sorted(entries.items()))


def rpa_decode_waste(k: RpaDecodeKernel, context_len: int | None) -> float:
    """Fraction of a decode kernel's compute chunk the live tokens do not use.

    A sliding layer needs at most its window; any layer needs at most the
    context. Unknown context counts nothing as wasted beyond the window.
    """
    needed = [n for n in (k.window, context_len) if n]
    if not needed or not k.chunk:
        return 0.0
    return max(0.0, 1.0 - min(needed) / k.chunk)


def d45_rpa_decode_block_oversized(f: features_lib.Features,
                                   ctx: context_lib.ServingContext,
                                   sid: str) -> list[Finding]:
    if not du.is_vllm(ctx, f) or f.step_ops.empty:
        return []
    phase = segmenter.DECODE
    g = du.phase_ops(f, phase)
    g = g[g['op_class'] == oc.ATTENTION_KERNEL]
    if g.empty or 'base_op' not in g.columns:
        return []
    n_steps = du.n_steps(f, phase)
    dur = du.stat(f, phase, 'mean_dur_ns')
    if not n_steps or not dur or math.isnan(dur):
        return []
    shares = {
        str(b): float(ns) / n_steps / dur
        for b, ns in g.groupby('base_op')['self_ns'].sum().items()
    }
    attn_share = features_lib.share_of(f.class_time, phase,
                                       oc.ATTENTION_KERNEL)
    ctx_len = (ctx.input_len +
               ctx.output_len if ctx.input_len and ctx.output_len else None)
    share, hits = rpa_decode_block_oversized(shares, attn_share,
                                             ctx.max_model_len, ctx_len)
    if not hits:
        return []
    saved = sum(s * rpa_decode_waste(k, ctx_len) for k, s, _ in hits)
    sizes = rpa_decode_block_sizes([k for k, _, _ in hits], ctx_len)
    changes = []
    if sizes and du.is_torchtpu(ctx, f):
        changes.append(
            knobs.make_change('vllm_torchtpu.env.TPU_RPA_DECODE_BLOCK_SIZES',
                              sizes, ctx.serving_config))
    evidence = [Evidence('attention kernel share', _pct(attn_share), '%')]
    for k, s, reasons in hits[:3]:
        evidence.append(
            Evidence(f'kernel `{k.name}`', _pct(s), '% of decode step',
                     '; '.join(reasons)))
    evidence.append(du.step_evidence(f, phase, sid))
    return [
        Finding(
            detector_id='D45_rpa_decode_block_oversized',
            title=
            'Decode attention computes over more KV slots than the context',
            phase=phase,
            summary=
            (f'RPA decode kernels whose KV compute chunk far exceeds what the'
             f' live tokens need are {_pct(share):.1f}% of the decode step'
             f' (attention {_pct(attn_share):.1f}%). Each step computes over'
             ' the whole chunk whatever the sequence length, and a sliding'
             ' layer whose chunk is several windows long cannot skip the'
             ' blocks outside its window.'),
            evidence=evidence,
            knob_changes=changes,
            risk=F.Risk.PERF_ONLY,
            confidence=F.Confidence.MEDIUM,
            est_step_fraction_saved=min(D45_MAX_GAIN, saved),
            validation=(
                'A/B against a same-session control at every point: the sizes'
                " are chosen for this workload's context, so a longer context"
                ' may need larger chunks. Re-profile: the decode kernels named'
                ' here should show the new block and chunk, and the decode'
                ' step should shrink by about the estimated share.'),
            kernel_followup=(
                'Size the RPA decode blocks to the live context: about the'
                ' window for sliding layers, compute chunks that track the'
                ' sequence length for global layers. Use the attention'
                " backend's block-size override when the image has one;"
                ' otherwise it is a change to the plugin or the kernel.'),
            source_refs=[],
        )
    ]


# -----------------------------------------------------------------------------
# D53: collective-heavy long prefill with no microbatch comm/compute overlap
# -----------------------------------------------------------------------------
def d53_prefill_microbatch_overlap(f: features_lib.Features,
                                   ctx: context_lib.ServingContext,
                                   sid: str) -> list[Finding]:
    """D53: collectives dominate long prefills (diagnostic, no knob)."""
    if (not du.is_vllm(ctx, f) or ctx.serving_config is None
            or f.step_ops.empty):
        return []
    mbt = ctx.max_num_batched_tokens or 0
    inp = ctx.input_len or 0
    if mbt < D53_MIN_BATCHED_TOKENS or inp < D53_MIN_BATCHED_TOKENS:
        return []
    phases = [p for p in du.active_phases(f) if p in du.PREFILL_LIKE]
    if not phases:
        return []
    phase = phases[0]
    cshare = features_lib.share_of(f.class_time, phase, oc.COLLECTIVE)
    if cshare < D53_MIN_COLL_SHARE:
        return []
    return [
        Finding(
            detector_id='D53_prefill_microbatch_overlap',
            title='Collective-heavy prefill with no microbatch overlap',
            phase=phase,
            summary=(
                f'Collectives take {_pct(cshare):.1f}% of the {phase} step'
                f' (--max-num-batched-tokens={mbt}). A microbatch (batch-split)'
                " implementation could overlap one chunk's collectives with"
                " another's compute; --long-prefill-token-threshold only caps"
                " each prompt's tokens per step and overlaps nothing on its"
                ' own.'),
            evidence=[
                Evidence('prefill collective share', _pct(cshare), '%'),
                Evidence('max_num_batched_tokens', mbt),
                du.step_evidence(f, phase, sid),
            ],
            knob_changes=[],
            risk=F.Risk.DIAGNOSTIC,
            confidence=F.Confidence.LOW,
            est_step_fraction_saved=min(0.15, 0.4 * cshare),
            kernel_followup=(
                'Prototype a prefill batch split: run each prefill step as two'
                " microbatches ordered so that one microbatch's collectives"
                " overlap the other's compute (the cited prototype used"
                ' --long-prefill-token-threshold to get two or more chunks per'
                ' step).'),
            validation=(
                'If a batch split is prototyped, A/B throughput and P99 TTFT'
                ' against the unsplit baseline. A vLLM-TorchTPU batch-split'
                ' prototype (not --long-prefill-token-threshold alone)'
                ' reported throughput and P99 TTFT gains on Qwen3.5;'
                ' the same notes say the gains vanished on a release'
                ' candidate whose Pallas kernels inline the collectives.'
                ' Guardrail: do not batch-split HBM-bound small-batch decode'
                ' (splitting B -> B/2 loads weights twice).'),
            source_refs=[du.DOC_PREFILL_MICROBATCH_OVERLAP],
        )
    ]


# -----------------------------------------------------------------------------
# D47: small-row decode all-reduce on vLLM-TorchTPU
# -----------------------------------------------------------------------------
def d47_small_decode_allreduce(f: features_lib.Features,
                               ctx: context_lib.ServingContext,
                               sid: str) -> list[Finding]:
    """D47: small-row decode all-reduces using XLA instead of small_all_reduce."""
    if (not du.is_torchtpu(ctx, f) or ctx.serving_config is None
            or f.step_ops.empty or f.collectives.empty
            or segmenter.DECODE not in du.active_phases(f)):
        return []
    if (du.knob_int(ctx, 'vllm_torchtpu.env.TPU_SMALL_ALL_REDUCE_MAX_ROWS', 0)
            > 0):
        return []
    decode_ops = du.phase_ops(f, segmenter.DECODE)
    if decode_ops['op'].fillna('').astype(str).str.match(
            _SMALL_AR_KERNEL).any():
        return []
    c = f.collectives[(f.collectives['phase'] == segmenter.DECODE)
                      & (f.collectives['collective'] == 'all-reduce')]
    if c.empty:
        return []
    rows = c['dims'].map(lambda d: (_dims(d) or [0])[0])
    small = c[(rows >= 1) & (rows <= D47_MAX_SMALL_ROWS)]
    if small.empty:
        return []
    share = _share_of_ns(f, segmenter.DECODE, small['ns_per_step'].sum())
    if share < D47_MIN_AR_SHARE:
        return []
    max_rows = max(int(rows[(rows >= 1) & (rows <= D47_MAX_SMALL_ROWS)].max()),
                   16)
    top = small.sort_values('ns_per_step', ascending=False).head(2)
    return [
        Finding(
            detector_id='D47_small_decode_allreduce',
            title='Small-batch decode all-reduces run via XLA collective',
            phase=segmenter.DECODE,
            summary=(
                f'Decode all-reduces with <={max_rows} rows take'
                f' {_pct(share):.1f}% of the decode step as XLA collectives'
                ' while TPU_SMALL_ALL_REDUCE_MAX_ROWS is unset. That variable'
                ' sets the row cap of the Pallas `small_all_reduce` kernel,'
                ' which needs an expert-parallel group and serves only some'
                ' collectives (row-parallel layers with a bias, the fused GDN'
                ' output projection, the MoE combine on replicated'
                ' sequence-parallel steps, the data-parallel combine).'),
            evidence=[
                Evidence('small all-reduce share', _pct(share), '%'),
                *[
                    Evidence(
                        f'all-reduce {r["out_dtype"]}[{r["dims"]}]',
                        r['ns_per_step'] / 1e3,
                        ' µs/step',
                    ) for _, r in top.iterrows()
                ],
                du.step_evidence(f, segmenter.DECODE, sid),
            ],
            knob_changes=[
                knobs.make_change(
                    'vllm_torchtpu.env.TPU_SMALL_ALL_REDUCE_MAX_ROWS',
                    str(max_rows),
                    ctx.serving_config,
                )
            ],
            risk=F.Risk.PERF_ONLY,
            confidence=F.Confidence.MEDIUM,
            est_step_fraction_saved=min(0.15, 0.35 * share),
            validation=(
                'A/B decode TPOT at small concurrency and re-profile to confirm'
                f' `small_all_reduce` replaces the XLA all-reduce on'
                f' <={max_rows}-row decode blocks; if it does not, the'
                ' all-reduce is on a path the kernel does not serve. Builds'
                ' differ on an unset cap: the envs.py comment says 0 keeps'
                " XLA's all-reduce, while current"
                ' layers/vllm/small_all_reduce_op.py treats 0 as 64 rows.'),
            source_refs=[du.DOC_GLM52_SERVING_OPT, _SRC_SMALL_ALL_REDUCE],
        )
    ]


# -----------------------------------------------------------------------------
# D48: unfused Q/K RMSNorm + partial RoPE on vLLM-TorchTPU
# -----------------------------------------------------------------------------
def d48_unfused_qk_norm_rope(f: features_lib.Features,
                             ctx: context_lib.ServingContext,
                             sid: str) -> list[Finding]:
    """D48: separate Q/K RMSNorm and RoPE ops on vLLM-TorchTPU."""
    if (not du.is_torchtpu(ctx, f) or ctx.serving_config is None
            or f.step_ops.empty):
        return []
    if du.knob_int(ctx, 'vllm_torchtpu.env.TPU_FUSED_QK_NORM_ROPE', 0) > 0:
        return []
    qwen3_next = bool(_QWEN3_NEXT_MODEL.search(ctx.model.name or ''))
    if qwen3_next:
        not_proposed = ''
    elif ctx.model.name:
        not_proposed = (f' Not proposed: {ctx.model.name} is not a Qwen3-Next'
                        ' model.')
    else:
        not_proposed = (' Not proposed: the model is unknown (no HF config)'
                        ' and the knob serves Qwen3-Next only.')
    out = []
    for phase in du.active_phases(f):
        g = du.phase_ops(f, phase)
        if du.name_mask(g, _FUSED_QK_NORM_ROPE_OP).any():
            continue
        tf = g['tf_op_name'].fillna('').astype(str)
        has_qknorm = tf.str.contains(r'q_norm|k_norm|qk_norm',
                                     case=False).any()
        has_rope = tf.str.contains(r'rotary_emb', case=False).any()
        if not (has_qknorm and has_rope):
            continue
        hits = g[tf.str.contains(_QK_NORM_ROPE_RE)]
        share = du.share(f, phase, hits)
        if share < D48_MIN_SHARE:
            continue
        changes = ([
            knobs.make_change(
                'vllm_torchtpu.env.TPU_FUSED_QK_NORM_ROPE',
                '1',
                ctx.serving_config,
            )
        ] if qwen3_next else [])
        out.append(
            Finding(
                detector_id='D48_unfused_qk_norm_rope',
                title='Q/K RMSNorm and RoPE run as separate unfused ops',
                phase=phase,
                summary=(
                    f'Separate Q/K RMSNorm and RoPE ops take'
                    f' {_pct(share):.1f}% of the {phase} step while'
                    ' TPU_FUSED_QK_NORM_ROPE is unset.'
                    ' TPU_FUSED_QK_NORM_ROPE=1 runs the q/gate split, Q/K'
                    ' RMSNorm, RoPE and gate as one Pallas call per attention'
                    ' layer, but only for Qwen3-Next-style attention (vLLM'
                    ' `Qwen3NextAttention`: text-only, gated, neox RoPE) and'
                    ' only on steps of <= 64 rows.' + not_proposed),
                evidence=[
                    Evidence('Q/K norm + RoPE share', _pct(share), '%'),
                    *du.base_op_evidence(f, phase, hits, 2),
                    du.step_evidence(f, phase, sid),
                ],
                knob_changes=changes,
                risk=F.Risk.PERF_ONLY if changes else F.Risk.DIAGNOSTIC,
                confidence=F.Confidence.MEDIUM,
                est_step_fraction_saved=min(0.10, 0.5 * share),
                validation=(
                    'A/B step time and re-profile to confirm the'
                    ' `qk_norm_rope_gate` Pallas kernel replaces the separate'
                    ' Q/K norm and rotary ops (steps above 64 rows keep them).'
                    if changes else ''),
                source_refs=[_SRC_FUSED_QK_NORM_ROPE],
            ))
    return out


# -----------------------------------------------------------------------------
# D49: replicated latent-MoE projections on dual-core chips (Kimi-K3)
# -----------------------------------------------------------------------------
def d49_replicated_latent_moe_proj(f: features_lib.Features,
                                   ctx: context_lib.ServingContext,
                                   sid: str) -> list[Finding]:
    """D49: replicated latent-MoE projections without intra-chip TP."""
    if (not du.is_torchtpu(ctx, f) or ctx.serving_config is None
            or f.step_ops.empty):
        return []
    if F.is_true(
            knobs.current_value(
                'vllm_torchtpu.env.TPU_LATENT_PROJ_INTRA_CHIP_TP',
                ctx.serving_config,
            )):
        return []
    out = []
    for phase in du.active_phases(f):
        g = du.phase_ops(f, phase)
        tf = g['tf_op_name'].fillna('').astype(str)
        hits = g[tf.str.contains(_LATENT_PROJ_RE)]
        if hits.empty:
            continue
        share = du.share(f, phase, hits)
        if share < D49_MIN_SHARE:
            continue
        out.append(
            Finding(
                detector_id='D49_replicated_latent_moe_proj',
                title=('Latent-MoE projections are replicated across dual-core'
                       ' chips'),
                phase=phase,
                summary=(
                    'Latent-MoE down/up projections take'
                    f' {_pct(share):.1f}% of the {phase} step while'
                    ' TPU_LATENT_PROJ_INTRA_CHIP_TP is unset. Sharding them'
                    ' column-parallel across the two cores of each chip with'
                    ' an on-package all-gather halves the HBM weight read.'),
                evidence=[
                    Evidence('latent projection share', _pct(share), '%'),
                    *du.base_op_evidence(f, phase, hits, 2),
                    du.step_evidence(f, phase, sid),
                ],
                knob_changes=[
                    knobs.make_change(
                        'vllm_torchtpu.env.TPU_LATENT_PROJ_INTRA_CHIP_TP',
                        '1',
                        ctx.serving_config,
                    )
                ],
                risk=F.Risk.PERF_ONLY,
                confidence=F.Confidence.MEDIUM,
                est_step_fraction_saved=0.45 * share,
                validation=(
                    'A/B decode TPOT and re-profile: the routed-expert latent'
                    ' projections should read half the weight bytes per core'
                    ' (no measured gain is cited).'),
                source_refs=[_SRC_LATENT_PROJ_INTRA_CHIP],
            ))
    return out


# -----------------------------------------------------------------------------
# D50: the Pallas small all-reduce dominates decode: A/B its other form
# -----------------------------------------------------------------------------
def d50_small_allreduce_form(f: features_lib.Features,
                             ctx: context_lib.ServingContext,
                             sid: str) -> list[Finding]:
    """D50: the small all-reduce kernel is a decode hotspot; try its other form."""
    if (not du.is_torchtpu(ctx, f) or f.step_ops.empty
            or segmenter.DECODE not in du.active_phases(f)):
        return []
    ops = du.phase_ops(f, segmenter.DECODE)
    names = ops['op'].fillna('').astype(str)
    hits = ops[names.str.match(_SMALL_AR_FORM)]
    if hits.empty:
        return []
    share = du.share(f, segmenter.DECODE, hits)
    if share < D50_MIN_SHARE:
        return []
    forms = set()
    for name in hits['op'].astype(str):
        m = _SMALL_AR_FORM.match(name)
        if m:
            forms.add(m.group(1).lower())
    if len(forms) != 1:
        return []
    form = forms.pop()
    other = 'oneshot' if form == 'twoshot' else 'twoshot'
    return [
        Finding(
            detector_id='D50_small_allreduce_form',
            title='Small all-reduce kernel is a decode hotspot',
            phase=segmenter.DECODE,
            summary=(
                f'The Pallas small all-reduce (`{form}` form) takes'
                f' {_pct(share):.1f}% of the decode step.'
                ' TPU_SMALL_ALL_REDUCE_FORM forces oneshot (all-gather + sum)'
                ' or twoshot (reduce-scatter + all-gather) instead of the'
                ' block-size default; which is faster depends on the rows and'
                f' devices, so A/B the {other} form.'),
            evidence=[
                Evidence('small all-reduce share', _pct(share), '%'),
                Evidence('form in the trace', form),
                du.step_evidence(f, segmenter.DECODE, sid),
            ],
            knob_changes=[
                knobs.make_change(
                    'vllm_torchtpu.env.TPU_SMALL_ALL_REDUCE_FORM',
                    other,
                    ctx.serving_config,
                )
            ],
            risk=F.Risk.PERF_ONLY,
            confidence=F.Confidence.LOW,
            est_step_fraction_saved=share,
            validation=(
                'A/B decode TPOT with >=3 interleaved runs per arm; keep the'
                ' form only if it beats the noise band, and re-profile to'
                f' confirm `small_all_reduce_{other}` ran.'),
        )
    ]


# -----------------------------------------------------------------------------
# D52: vLLM-TorchTPU moe_tp_decode <= 64-row kernel cliff on decode steps
# -----------------------------------------------------------------------------
def d52_moe_tp_decode_row_limit(f: features_lib.Features,
                                ctx: context_lib.ServingContext,
                                sid: str) -> list[Finding]:
    """D52: TP MoE decode steps exceed `moe_tp_decode`'s 64-row limit."""
    m = ctx.model
    tp = ctx.tp or 1
    if (not m.is_moe or not du.is_torchtpu(ctx, f) or tp <= 1 or ctx.ep
            or ctx.serving_config is None or f.step_ops.empty
            or segmenter.DECODE not in du.active_phases(f)
            or du.is_prefill_heavy(ctx)):
        return []
    if du.name_mask(f.step_ops, du.FUSED_EP_OP).any():
        return []
    ops = du.phase_ops(f, segmenter.DECODE)
    has_moe_tp_decode = (
        not ops.empty and
        ops['op'].dropna().astype(str).str.startswith('moe_tp_decode').any())
    if not m.is_mla and not has_moe_tp_decode:
        return []
    gmm_share = features_lib.share_of(f.class_time, segmenter.DECODE,
                                      oc.MOE_GMM)
    if gmm_share < D52_MIN_GMM_SHARE:
        return []
    dec_steps = (f.seg.steps[(f.seg.steps['phase'] == segmenter.DECODE)
                             & f.seg.steps['bucket'].notna()]
                 if not f.seg.steps.empty and 'bucket' in f.seg.steps else
                 pd.DataFrame())
    dec_ns = float(dec_steps['dur_ns'].sum()) if not dec_steps.empty else 0.0
    if not dec_ns:
        return []
    rows = dec_steps['bucket'].astype(float)
    over = rows > D52_MOE_TP_DECODE_MAX_ROWS
    over_share = float(dec_steps.loc[over, 'dur_ns'].sum()) / dec_ns
    if over_share < D52_MIN_DECODE_TIME_OVER_ROWS:
        return []
    peak_rows = float(rows.max())
    gmm_ops = ops[ops['op_class'] == oc.MOE_GMM]
    return [
        Finding(
            detector_id='D52_moe_tp_decode_row_limit',
            title='TP MoE decode exceeds moe_tp_decode 64-row kernel limit',
            phase=segmenter.DECODE,
            summary=(
                f'MoE grouped matmuls take {_pct(gmm_share):.1f}% of the'
                f' decode step under TP={tp} without expert parallelism, and'
                f' {_pct(over_share):.0f}% of decode time is in steps above'
                f" `moe_tp_decode`'s {D52_MOE_TP_DECODE_MAX_ROWS}-row limit"
                f' (token buckets up to {peak_rows:.0f} rows; e.g. batch >= 17'
                ' with 3 speculative tokens). Those steps fall back to'
                ' TP-sliced GMM, which reads expert weights at a lower HBM'
                ' bandwidth than EP.'),
            evidence=[
                Evidence('MoE GMM share', _pct(gmm_share), '%'),
                Evidence(
                    f'decode time in steps above {D52_MOE_TP_DECODE_MAX_ROWS}'
                    ' rows', _pct(over_share), '%'),
                Evidence('max decode bucket rows', peak_rows, 'tokens'),
                Evidence('TP', tp),
                *du.base_op_evidence(f, segmenter.DECODE, gmm_ops, 2),
                du.step_evidence(f, segmenter.DECODE, sid),
            ],
            knob_changes=[
                knobs.make_change(
                    'vllm.cli.enable_expert_parallel',
                    'true',
                    ctx.serving_config,
                )
            ],
            risk=F.Risk.PERF_ONLY,
            confidence=F.Confidence.MEDIUM,
            est_step_fraction_saved=min(0.30, 0.35 * gmm_share),
            validation=(
                'A/B decode throughput and p90 interactivity with'
                ' `--enable-expert-parallel` at the decode batch sizes whose'
                f' steps exceed {D52_MOE_TP_DECODE_MAX_ROWS} token rows (more'
                ' than 16 sequences with 3 speculative tokens); keep TP MoE'
                ' where every step stays within'
                f' {D52_MOE_TP_DECODE_MAX_ROWS} rows.'),
            kernel_followup=(
                'If TP MoE is desired at 65-128 decode rows, extend'
                ' `moe_tp_decode` tile shapes past 64 rows in vLLM-TorchTPU.'),
            source_refs=[_SRC_MOE_TP_DECODE, du.DOC_GLM52_SERVING_OPT],
        )
    ]


ALL_DETECTORS = (
    d26_moe_combine_without_local_reduction,
    d27_max_num_seqs_inflation,
    d28_moe_activation_unfused,
    d29_async_scheduling_off,
    d30_batched_rpa,
    d31_gmm_tile_misalignment,
    d32_excess_collectives_per_layer,
    d33_kv_cache_block_size,
    d34_speculative_draft_overhead,
    d35_unfused_glue_programs,
    d45_rpa_decode_block_oversized,
    d46_batched_rpa_decode_untunable,
    d47_small_decode_allreduce,
    d48_unfused_qk_norm_rope,
    d49_replicated_latent_moe_proj,
    d50_small_allreduce_form,
    d52_moe_tp_decode_row_limit,
    d53_prefill_microbatch_overlap,
)
