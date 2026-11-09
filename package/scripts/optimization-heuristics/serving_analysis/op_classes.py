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
"""Framework-agnostic op classification for serving traces.

Classifies each (module, op) using only signals that are stable across models:
the HLO category, the framework op type, JAX named-scope paths (tf_op_name) and
well-known kernel names (RPA, GMM/megablox, splash/flash). We intentionally do
NOT try to map every op to a transformer component; detectors only need the
coarse classes below.
"""

# pylint: disable=missing-function-docstring,missing-class-docstring,g-doc-args,g-doc-return-or-yield,g-docstring-has-escape,g-long-ternary,g-explicit-length-test,g-complex-comprehension,protected-access,cell-var-from-loop,comparison-with-itself,unused-argument

import re

import pandas as pd

# Op classes.
COLLECTIVE = 'collective'
ATTENTION_KERNEL = 'attention_kernel'
ATTENTION_OTHER = 'attention_other'
LINEAR_ATTENTION = 'linear_attention'
MOE_GMM = 'moe_gmm'
MOE_GMM_ZERO_INIT = 'moe_gmm_zero_init'
MOE_TOPK_SORT = 'moe_topk_sort'
MOE_PERMUTE = 'moe_permute'
MOE_OTHER = 'moe_other'
MATMUL = 'matmul'
DATA_FORMATTING = 'data_formatting'
ROPE = 'rope'
ASYNC_BOOKKEEPING = 'async_bookkeeping'
# Sparse-attention (DeepSeek DSA / GLM) indexer: scoring + top-k selection.
ATTN_INDEXER = 'attn_indexer'
# TensorCore blocked on an async op it launched (SparseCore gather/scatter
# offload, host/DMA copies). Self time here is waiting, not compute.
ASYNC_WAIT = 'async_wait'
OTHER = 'other'

_COLLECTIVE_TYPES = (
    'all-reduce',
    'all-gather',
    'reduce-scatter',
    'all-to-all',
    'collective-permute',
    'ragged-all-to-all',
)
_COLLECTIVE_TF_TYPES = {
    'psum': 'all-reduce',
    'pmean': 'all-reduce',
    'all_gather': 'all-gather',
    'psum_scatter': 'reduce-scatter',
    'reduce_scatter': 'reduce-scatter',
    'all_to_all': 'all-to-all',
    'ragged_all_to_all': 'ragged-all-to-all',
    'ppermute': 'collective-permute',
    # torch_tpu (vLLM-TorchTPU) distributed ops.
    'distributed.all_reduce': 'all-reduce',
    'distributed.all_gather_into_tensor': 'all-gather',
    'distributed.reduce_scatter_tensor': 'reduce-scatter',
}
_MOE_SCOPE = re.compile(r'moe|expert|router|megablox|gmm', re.I)
_GMM = re.compile(r'(^|[/_\-(])gmm|megablox|grouped_mat|fused[-_](ep[-_])?moe',
                  re.I)
# Per-expert kernels (`.../expert_3/gating_3/pallas_call`).
_EXPERT_MATMUL = re.compile(r'/expert_\d+/', re.I)
# Dense shared-expert MLP inside an MoE block (DeepSeek-V3, GLM-5, Qwen3.5).
_SHARED_EXPERT = re.compile(r'(^|/)shared_(expert|mlp)', re.I)
# SparseCore token dispatch / combine kernels (gather, gather-reduce, scatter).
_MOE_SC_PERMUTE = re.compile(
    r'sc_ragged_(gather|scatter)|(ragged|dense)_gather_reduce', re.I)
# Ragged paged attention: `RPA-` (fused) and the split decode / prefill /
# mixed variants `RPAd-`, `RPAp-`, `RPAm-`.
_ATTN = re.compile(
    r'ragged_paged_attention|(^|/)RPA[dpm]?-|attn|attention|flash|splash|mla',
    re.I,
)
# MLA v2 KV-cache transpose kernels (tpu_inference kernels/mla/v2/transpose).
_XPOSE_KERNEL = re.compile(r'(^|/)xpose_(pipeline|full)_shape', re.I)
_COPY_CATEGORIES = ('data formatting', 'copy', 'transpose')
# Integer scatter-add outputs inside MoE are expert-load histograms
# (bincount), not token dispatch.
_INT_DTYPES = ('s32', 'u32')
# Collectives implemented as Pallas kernels (dense / forward all-to-all,
# latency-optimized all-gather / psum-scatter). Matched on op name and scope.
_KERNEL_COLLECTIVE = re.compile(
    r'(^|[/_])(start|wait)_(dense|forward)_a2a|dense_all_to_all|'
    r'a2a_forward|latency_(all_gather|psum_scatter|psum|all_reduce)|'
    r'run_per_expert_shard/(.*/)?(finish|next_chunk|retile_slice)/',
    re.I,
)
_KERNEL_COLLECTIVE_TYPE = (
    (re.compile(r'all_gather', re.I), 'all-gather'),
    (re.compile(r'psum_scatter', re.I), 'reduce-scatter'),
    (re.compile(r'psum|all_reduce', re.I), 'all-reduce'),
)
# Pallas matmul kernels: `dot` / `.../linear/` custom calls.
_KERNEL_MATMUL = re.compile(r'^(dot|matmul)|/linear/', re.I)
# Flash-attention style kernel output: (out, running max, running sum), e.g.
# (bf16[64,16,256], f32[64,16,128], f32[64,16,128]).
_LAYOUT = r'(?:\{[^}]*\})?'
_FLASH_OUT = re.compile(r'=\s*\(\w+\[[\d,]+\]' + _LAYOUT +
                        r',\s*f32\[([\d,]+),128\]' + _LAYOUT +
                        r',\s*f32\[\1,128\]')
_INDEXER = re.compile(r'streamidx|indexer|sc_topk|lightning_index', re.I)
_LINEAR_ATTN = re.compile(r'gdn|gated_delta|conv1d|linear_attn|kda|mamba|ssm',
                          re.I)
_ROPE = re.compile(r'rotary|rope|cos_sin', re.I)
_SHAPE = re.compile(r'=\s*\(?([a-z0-9]+)\[([0-9,]*)\]')
_OPCODE = re.compile(r'\}\s+([a-z][a-z0-9\-]*)\(')
_SCOPE_SPLIT = re.compile(r'/')


def parse_output_shape(kernel_text: str) -> tuple[str, tuple[int, ...]]:
    """Returns (dtype, dims) of the first output in an HLO instruction string."""
    m = _SHAPE.search(kernel_text or '')
    if not m:
        return '', ()
    dims = tuple(int(x) for x in m.group(2).split(',') if x)
    return m.group(1), dims


def parse_opcode(kernel_text: str) -> str:
    m = _OPCODE.search(kernel_text or '')
    return m.group(1) if m else ''


def collective_type(hlo_category: str, tf_op_type: str,
                    kernel_text: str) -> str:
    cat = (hlo_category or '').lower()
    for t in _COLLECTIVE_TYPES:
        if cat.startswith(t) or cat == f'{t} fusion':
            return t
    if 'all-reduce-scatter' in cat:
        return 'reduce-scatter'
    if tf_op_type in _COLLECTIVE_TF_TYPES:
        return _COLLECTIVE_TF_TYPES[tf_op_type]
    opcode = parse_opcode(kernel_text)
    for t in _COLLECTIVE_TYPES:
        if opcode.startswith(t):
            return t
    return ''


def layer_scope(tf_op_name: str) -> str:
    """Top framework scope below the jit wrapper, e.g. 'SharedFusedMoE'."""
    parts = [p for p in _SCOPE_SPLIT.split(tf_op_name or '') if p]
    for p in parts:
        if p.startswith('jit(') and p.endswith(')'):
            continue
        return re.sub(r'\.\d+$', '', p)
    return ''


def classify_row(
    op: str,
    kernel_text: str,
    tf_op_name: str,
    tf_op_type: str,
    hlo_category: str,
) -> tuple[str, str]:
    """Returns (op_class, collective_type)."""
    cat = (hlo_category or '').lower()
    tf = tf_op_name or ''
    if not tf and cat in ('copy-start', 'copy-done', 'async-start',
                          'async-done'):
        return ASYNC_BOOKKEEPING, ''
    in_moe = (bool(_MOE_SCOPE.search(tf)) or bool(_GMM.search(op or ''))
              or bool(_MOE_SC_PERMUTE.search(op or '')))
    coll = collective_type(cat, tf_op_type, kernel_text)
    if coll:
        return COLLECTIVE, coll
    if cat == 'async-start':
        return ASYNC_BOOKKEEPING, ''
    is_kernel = cat == 'custom-call' or tf_op_type == 'pallas_call'
    if is_kernel and (_KERNEL_COLLECTIVE.search(op or '')
                      or _KERNEL_COLLECTIVE.search(tf)):
        for pat, t in _KERNEL_COLLECTIVE_TYPE:
            if pat.search(tf) or pat.search(op or ''):
                return COLLECTIVE, t
        return COLLECTIVE, 'all-to-all'
    if _INDEXER.search(tf) or _INDEXER.search(op or ''):
        return ATTN_INDEXER, ''
    if in_moe:
        in_gmm = bool(_GMM.search(tf)) or bool(_GMM.search(op or ''))
        if in_gmm and cat == 'broadcast':
            return MOE_GMM_ZERO_INIT, ''
        if _MOE_SC_PERMUTE.search(op or ''):
            return MOE_PERMUTE, ''
        if in_gmm and is_kernel:
            return MOE_GMM, ''
        if is_kernel and (_EXPERT_MATMUL.search(tf)
                          or _EXPERT_MATMUL.search(op or '')):
            return MOE_GMM, ''
        if (tf_op_type in ('top_k', 'approx_top_k', 'sort', 'argsort')
                or (cat == 'sort') or '/jit(argsort)/' in tf):
            return MOE_TOPK_SORT, ''
        if (tf_op_type in ('scatter', 'scatter-add', 'scatter_add')
                and parse_output_shape(kernel_text)[0] in _INT_DTYPES):
            return MOE_OTHER, ''
        is_permute_type = tf_op_type in (
            'gather',
            'scatter',
            'scatter-add',
            'scatter_add',
            'one_hot',
        )
        is_permute_fusion = cat == 'custom fusion' and ('scatter' in tf
                                                        or 'gather' in tf)
        if is_permute_type or is_permute_fusion:
            return MOE_PERMUTE, ''
        if in_gmm:
            return MOE_GMM, ''
        if _SHARED_EXPERT.search(tf) and (
                cat.startswith('convolution')
                or tf_op_type in ('dot_general', 'einsum') or
            (is_kernel and
             (_KERNEL_MATMUL.search(op or '') or _KERNEL_MATMUL.search(tf)))):
            return MATMUL, ''
        return MOE_OTHER, ''
    if cat in ('async-done', 'copy-done'):
        # SparseCore collectives are named with underscores
        # (`ragged_all_to_all_sc.call-done`).
        name = (op or '').lower().replace('_', '-')
        for t in _COLLECTIVE_TYPES:
            if name.startswith(t):
                return COLLECTIVE, t
        return ASYNC_WAIT, ''
    if _LINEAR_ATTN.search(tf):
        return LINEAR_ATTENTION, ''
    if is_kernel and _FLASH_OUT.search(kernel_text or ''):
        return ATTENTION_KERNEL, ''
    if is_kernel and (_KERNEL_MATMUL.search(op or '')
                      or _KERNEL_MATMUL.search(tf)):
        return MATMUL, ''
    is_copy = cat in _COPY_CATEGORIES or cat.startswith('copy')
    if is_kernel and (_XPOSE_KERNEL.search(op or '')
                      or _XPOSE_KERNEL.search(tf)):
        return DATA_FORMATTING, ''
    if _ATTN.search(tf) or _ATTN.search(op or ''):
        if is_kernel:
            return ATTENTION_KERNEL, ''
        # KV / Q relayout copies inside the attention scope are data formatting,
        # not attention compute.
        if is_copy:
            return DATA_FORMATTING, ''
        return ATTENTION_OTHER, ''
    if _ROPE.search(tf):
        return ROPE, ''
    if cat.startswith('convolution') or tf_op_type in ('dot_general',
                                                       'einsum'):
        return MATMUL, ''
    if is_copy:
        return DATA_FORMATTING, ''
    return OTHER, ''


def classify(op_meta: pd.DataFrame) -> pd.DataFrame:
    """Adds op_class, collective, layer_scope, out_dtype, out_dims columns."""
    df = op_meta.copy()
    res = [
        classify_row(o, k, t, ty, c) for o, k, t, ty, c in zip(
            df['op'],
            df['kernel_text'],
            df['tf_op_name'],
            df['tf_op_type'],
            df['hlo_category'],
        )
    ]
    df['op_class'] = [r[0] for r in res]
    df['collective'] = [r[1] for r in res]
    df['layer_scope'] = df['tf_op_name'].map(layer_scope)
    df['moe_scope'] = [
        bool(_MOE_SCOPE.search(t or '')) or bool(_GMM.search(o or ''))
        for t, o in zip(df['tf_op_name'], df['op'])
    ]
    shapes = df['kernel_text'].map(parse_output_shape)
    df['out_dtype'] = shapes.map(lambda s: s[0])
    df['out_dims'] = shapes.map(lambda s: s[1])
    df['opcode'] = df['kernel_text'].map(parse_opcode)
    df['base_op'] = df['op'].map(base_op_name)
    return df


_INSTANCE_TOKEN = re.compile(r'\.(?:\d+|clone|cloned|remat\d*)(?=\.|$|-)')


def base_op_name(op: str) -> str:
    """Strips per-instance tokens.

  'psum.687' -> 'psum'; 'fusion.2489.cloned.1.call-done' ->
  'fusion.call-done'; 'RPA-bq_32-bkvp_64-p_64.21' -> 'RPA-bq_32-bkvp_64-p_64'.

  Args:
    op: HLO op name.
  """
    prev = None
    s = op or ''
    while prev != s:
        prev, s = s, _INSTANCE_TOKEN.sub('', s)
    return s
