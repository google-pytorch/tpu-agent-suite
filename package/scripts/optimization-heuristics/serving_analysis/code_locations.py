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
"""Code locations of hot ops: where in the served code step time is spent.

For each op, XLA records the Python line that emitted it (`source_line`, the
innermost frame outside JAX) and the framework scope (`tf_op_name`, e.g.
`jit(step_fun)/FusedMoE/jit(fused_moe_func)/shard_map/psum`). This module
groups step-program ops by that line and ranks the lines by step time, for the
whole step (`LocationIndex.hotspots`) and for each finding (`attach`), so that
a finding names the code to open and not only an op class.

Paths are recorded on the machine that ran the server, e.g.
`/home/<user>/tpu-inference/tpu_inference/layers/common/fused_moe_gmm.py`.
They are cut down to the import path (`tpu_inference/layers/...`), which the
optimizer then looks up in the served source. Lines of generic launchers
(torch_tpu's Pallas call wrapper, torch's op dispatch) say nothing about the
caller, so ops recorded there are grouped by kernel name instead.
"""

import dataclasses
import re
from collections.abc import Collection, Mapping, Sequence

import pandas as pd
from serving_analysis import features as features_lib
from serving_analysis import findings as findings_lib
from serving_analysis import op_classes as oc

# Top-level Python packages that serving code and its libraries come from; a
# recorded path is cut to start at one of them.
PACKAGES = frozenset({
    'flax',
    'jax',
    'jetstream',
    'maxtext',
    'MaxText',
    'qwix',
    'tokamax',
    'torch',
    'torch_tpu',
    'torchax',
    'tpu_inference',
    'vllm',
    'vllm_torchtpu',
})
# Of those, the libraries: code that a serving change does not edit.
LIBRARIES = frozenset({'flax', 'jax', 'qwix', 'torch', 'torch_tpu', 'torchax'})
# Files whose recorded line is a dispatcher or kernel launcher, not the caller.
_GENERIC = re.compile(
    r'(?:^|/)(?:torch_tpu/_internal/pallas/|torch/_library/|torch/_ops\.py'
    r'|qwix/_src/core/pallas\.py|jax/_src/)')
_SOURCE_LINE = re.compile(r'^(?P<path>\S+?\.py):(?P<line>\d+)')
_PACKAGE_DIR = re.compile(r'(?:^|/)(' +
                          '|'.join(sorted(PACKAGES, key=len, reverse=True)) +
                          r')/')
_JIT = re.compile(r'^jit\((\w+)\)$')
_LAYER = re.compile(r'^[A-Z][A-Za-z0-9]*$')
# Ops named in evidence metrics, e.g. "top op `sort.153`", "kernel `RPAd-...`".
_EVIDENCE_OP = re.compile(r'\b(?:op|kernel) `([^`]+)`')


def normalize_path(source_line: str) -> tuple[str, int] | None:
    """The import path and line of a recorded source line.

    Args:
      source_line: As recorded, e.g.
        '/home/u/tpu-inference/tpu_inference/layers/common/moe.py:188'.

    Returns:
      (path, line), e.g. ('tpu_inference/layers/common/moe.py', 188). None for
      an empty or unparsable line and for generic launchers.
    """
    m = _SOURCE_LINE.match(source_line.strip())
    if not m:
        return None
    path, line = m.group('path'), int(m.group('line'))
    dirs = list(_PACKAGE_DIR.finditer(path))
    installed = [
        d for d in dirs
        if path[:d.start(1)].endswith(('site-packages/', 'src/'))
    ]
    if installed:
        path = path[installed[-1].start(1):]
    elif dirs and (path.startswith('/') or dirs[0].start(1) == 0):
        path = path[dirs[0].start(1):]
        package = dirs[0].group(1) + '/'
        while path.startswith(package * 2):
            path = path[len(package):]
    if _GENERIC.search(path):
        return None
    return path, line


def is_library(path: str) -> bool:
    """Whether a normalized path is in a library (jax, torch, ...)."""
    return path.split('/')[0] in LIBRARIES


def function_name(tf_op_name: str) -> str:
    """The innermost jitted function or layer of a framework scope."""
    name = ''
    for part in tf_op_name.split('/'):
        part = re.sub(r'\.\d+$', '', part)
        m = _JIT.match(part)
        if m:
            name = m.group(1)
        elif _LAYER.match(part):
            name = part
    return name


@dataclasses.dataclass(frozen=True)
class OpFilter:
    """The step ops that a detector's findings are about."""

    classes: frozenset[str]
    moe_only: bool = False  # Only ops in an MoE block (e.g. its collectives).


def _ops(*classes: str, moe_only: bool = False) -> OpFilter:
    return OpFilter(frozenset(classes), moe_only)


# Op classes per detector. Detectors that name their ops in the evidence
# (`op `...``, `kernel `...``) get those as well.
DETECTOR_OPS: Mapping[str, OpFilter] = {
    'D4_moe_topk_sort': _ops(oc.MOE_TOPK_SORT),
    'D5_moe_permute': _ops(oc.MOE_PERMUTE),
    'D6_expert_imbalance': _ops(oc.MOE_GMM),
    'D7_ar_plus_slice': _ops(oc.COLLECTIVE, moe_only=True),
    'D7_collective_bound': _ops(oc.COLLECTIVE),
    'D7_decode_weight_allgather': _ops(oc.COLLECTIVE),
    'D7_moe_allgather_fp8': _ops(oc.COLLECTIVE, moe_only=True),
    'D8_kv_head_padding': _ops(oc.ATTENTION_KERNEL),
    'D12_kv_cache_quant': _ops(oc.ATTENTION_KERNEL),
    'D13_gmm_zero_init': _ops(oc.MOE_GMM_ZERO_INIT),
    'D15_torchtpu_fused_ep_moe': _ops(oc.COLLECTIVE, oc.MOE_GMM,
                                      moe_only=True),
    'D16_torchtpu_mla_layout': _ops(oc.ATTENTION_KERNEL),
    'D17_data_formatting': _ops(oc.DATA_FORMATTING),
    'D18_attn_indexer': _ops(oc.ATTN_INDEXER),
    'D19_in_step_wait': _ops(oc.ASYNC_WAIT),
    'D20_mla_attention_tp': _ops(oc.ATTENTION_KERNEL, oc.ATTENTION_OTHER),
    'D21_sequential_loop': _ops(oc.LINEAR_ATTENTION),
    'D22_moe_without_gmm': _ops(oc.MOE_OTHER),
    'D24_moe_tp_without_ep': _ops(oc.MOE_GMM),
    'D26_moe_combine_no_local_reduction': _ops(oc.COLLECTIVE, moe_only=True),
    'D28_moe_activation_unfused': _ops(oc.MOE_OTHER),
    'D30_batched_rpa': _ops(oc.ATTENTION_KERNEL),
    'D31_gmm_tile_misalignment': _ops(oc.MOE_GMM),
    'D32_excess_collectives_per_layer': _ops(oc.COLLECTIVE),
    'D33_kv_cache_block_size': _ops(oc.ATTENTION_KERNEL),
    'D36_hbm_limited_decode_batch': _ops(oc.MOE_GMM),
    'D37_moe_weight_precision': _ops(oc.MOE_GMM),
    'D40_mla_kernel_config': _ops(oc.ATTENTION_KERNEL),
    'D44_moe_combine_not_chunked': _ops(oc.COLLECTIVE, moe_only=True),
    'D45_rpa_decode_block_oversized': _ops(oc.ATTENTION_KERNEL),
    'D53_prefill_microbatch_overlap': _ops(oc.COLLECTIVE),
    'D47_small_decode_allreduce': _ops(oc.COLLECTIVE),
    'D48_unfused_qk_norm_rope': _ops(oc.ROPE, oc.ATTENTION_OTHER),
    'D49_replicated_latent_moe_proj': _ops(oc.MOE_OTHER,
                                           oc.MATMUL,
                                           moe_only=True),
    'D50_small_allreduce_form': _ops(oc.COLLECTIVE),
    'D52_moe_tp_decode_row_limit': _ops(oc.MOE_GMM),
    'D46_batched_rpa_decode_untunable': _ops(oc.ATTENTION_KERNEL),
}

_GROUP_KEYS = ['phase', 'op_class', 'moe', 'path', 'line', 'op']


def _text(so: pd.DataFrame, col: str) -> pd.Series:
    if col not in so:
        return pd.Series('', index=so.index, dtype=object)
    return so[col].fillna('').astype(str)


def _group(so: pd.DataFrame) -> pd.DataFrame:
    """Step-op time per (phase, class, MoE, source location, op)."""
    cols = _GROUP_KEYS + ['self_ns', 'count', 'op_type', 'scope']
    if so.empty:
        return pd.DataFrame(columns=cols)
    src = _text(so, 'source_line')
    locs = {s: normalize_path(s) for s in pd.unique(src)}
    base = _text(so, 'base_op')
    df = pd.DataFrame({
        'phase':
        so['phase'].astype(str),
        'op_class':
        so['op_class'].astype(str),
        'moe': (so['moe_scope'].fillna(False).astype(bool)
                if 'moe_scope' in so else False),
        'path':
        src.map({
            s: loc[0] if loc else ''
            for s, loc in locs.items()
        }),
        'line':
        src.map({
            s: loc[1] if loc else 0
            for s, loc in locs.items()
        }),
        'op':
        base.where(base.astype(bool), _text(so, 'op')),
        'op_type':
        _text(so, 'tf_op_type'),
        'scope':
        _text(so, 'tf_op_name'),
        'self_ns':
        so['self_ns'].astype('int64'),
    })
    return (df.groupby(_GROUP_KEYS, sort=False).agg(
        self_ns=('self_ns', 'sum'),
        count=('self_ns', 'size'),
        op_type=('op_type', 'first'),
        scope=('scope', 'first'),
    ).reset_index())


class LocationIndex:
    """Step-program op time grouped by source location (built once)."""

    def __init__(self, f: features_lib.Features):
        steps = f.seg.steps
        self._steps: dict[str, tuple[int, float]] = {}
        if not steps.empty:
            for phase, g in steps.groupby('phase'):
                self._steps[str(phase)] = (len(g), float(g['dur_ns'].mean()))
            self._steps['all'] = (len(steps), float(steps['dur_ns'].mean()))
        self._groups = _group(f.step_ops)

    @property
    def phases(self) -> list[str]:
        """Step phases, without 'all'."""
        return [p for p in self._steps if p != 'all']

    def top(
        self,
        phase: str,
        op_filter: OpFilter | None = None,
        ops: Collection[str] = (),
        n: int = 3,
        min_share: float = 0.001,
    ) -> list[findings_lib.CodeLocation]:
        """The source locations with the most step time among selected ops."""
        g = self._groups
        if phase != 'all':
            g = g[g['phase'] == phase]
        mask = pd.Series(False, index=g.index)
        if op_filter is not None:
            sel = g['op_class'].isin(op_filter.classes)
            if op_filter.moe_only:
                sel &= g['moe']
            mask |= sel
        if ops:
            mask |= g['op'].isin(list(ops))
        return self._rank(g[mask], phase, n, min_share)

    def hotspots(
        self,
        n: int = 8,
        min_share: float = 0.005
    ) -> dict[str, list[findings_lib.CodeLocation]]:
        """Per phase, the source locations with the most step time."""
        out = {}
        for phase in self.phases:
            g = self._groups[self._groups['phase'] == phase]
            out[phase] = self._rank(g, phase, n, min_share)
        return out

    def _rank(self, g: pd.DataFrame, phase: str, n: int,
              min_share: float) -> list[findings_lib.CodeLocation]:
        """Sums `g` per location (source line, else op name) and ranks it."""
        n_steps, mean_dur = self._steps.get(phase, (0, 0.0))
        if g.empty or not n_steps or not mean_dur:
            return []
        key = g['path'] + ':' + g['line'].astype(str)
        key = key.where(g['path'].astype(bool), '#' + g['op'])
        g = g.assign(key=key)
        totals = g.groupby('key')['self_ns'].sum().sort_values(ascending=False)
        out = []
        for k, ns in totals.items():
            share = float(ns) / n_steps / mean_dur
            if len(out) >= n or share < min_share:
                break
            rows = g[g['key'] == k]
            rep = rows.loc[rows['self_ns'].idxmax()]
            path = str(rep['path'])
            out.append(
                findings_lib.CodeLocation(
                    path=path,
                    line=int(rep['line']),
                    function=function_name(str(rep['scope'])),
                    op=str(rep['op']),
                    op_type=str(rep['op_type']),
                    op_class=str(
                        rows.groupby('op_class')['self_ns'].sum().idxmax()),
                    phase=phase,
                    ms_per_step=round(float(ns) / n_steps / 1e6, 4),
                    share=round(share, 4),
                    library=bool(path) and is_library(path),
                ))
        return out


def evidence_ops(f: findings_lib.Finding) -> set[str]:
    """Ops a finding's evidence names, without instance suffixes."""
    out: set[str] = set()
    for e in f.evidence:
        for m in _EVIDENCE_OP.finditer(e.metric):
            out.add(oc.base_op_name(m.group(1)))
    return out


def attach(found: Sequence[findings_lib.Finding],
           index: LocationIndex,
           n: int = 3) -> None:
    """Fills each finding's `locations` with its ops' hottest source lines."""
    for f in found:
        op_filter = DETECTOR_OPS.get(f.detector_id)
        ops = evidence_ops(f)
        if op_filter is None and not ops:
            continue
        f.locations = index.top(f.phase, op_filter, ops, n=n)
