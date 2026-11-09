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
"""Timeline features for serving detectors.

Every function is pure (DataFrames in, DataFrames/dicts out) so detectors can
be unit-tested with small synthetic frames. Features are computed over *step*
programs only (see segmenter.py); auxiliary programs (logits, sampling) are
counted in step busy time but not in per-op shares.
"""

# pylint: disable=missing-function-docstring,missing-class-docstring,g-doc-args,g-doc-return-or-yield,g-docstring-has-escape,g-long-ternary,g-explicit-length-test,g-complex-comprehension,protected-access,cell-var-from-loop,comparison-with-itself,unused-argument

import dataclasses
import re
from typing import Any

import numpy as np
import pandas as pd
from serving_analysis import intervals, op_classes, segmenter
from serving_analysis import trace_table as trace_table_lib

_META_JOIN_COLS = [
    'module',
    'op',
    'op_class',
    'collective',
    'layer_scope',
    'moe_scope',
    'tf_op_type',
    'hlo_category',
    'opcode',
    'base_op',
    'tf_op_name',
]
_COMPILE_RE = re.compile(
    r'backend_compile|compile_or_get_cached|XlaCompile|PjRtCompile|'
    r'pjit_compile|jit_compile|_cached_compilation',
    re.I,
)
_STOP_TRACE_RE = re.compile(r'stop_trace', re.I)
# Host events longer than this many median step cycles are enclosing loops,
# not attributable work.
_HOST_ENCLOSING_CYCLES = 3.0
# Step cycles longer than this many median cycles are excluded from idle stats.
OUTLIER_CYCLE_FACTOR = 5.0


@dataclasses.dataclass
class Features:
    """All features consumed by detectors."""

    seg: segmenter.Segmentation
    meta: pd.DataFrame  # classified op_meta
    step_ops: pd.DataFrame  # ops in step programs with class + step_idx + phase
    phase_stats: pd.DataFrame
    class_time: pd.DataFrame
    host_gap: dict[str, pd.DataFrame]  # phase -> top host events in idle gaps
    compile_events: pd.DataFrame
    stop_trace: pd.DataFrame
    # host_transfers() summary.
    transfer: dict[str, float] = dataclasses.field(default_factory=dict)


# -----------------------------------------------------------------------------
# Op <-> step assignment
# -----------------------------------------------------------------------------
def assign_ops_to_steps(ops: pd.DataFrame, meta: pd.DataFrame,
                        steps: pd.DataFrame) -> pd.DataFrame:
    """Returns step-program ops joined with class metadata and step index.

  Adds columns: step_idx (n-th step on that device), phase, dur_ns, plus the
  `_META_JOIN_COLS` from the classified metadata.

  Args:
    ops: TraceTable.ops.
    meta: Classified op_meta (op_classes.classify output).
    steps: Segmentation.steps.
  """
    if steps.empty or ops.empty:
        return ops.iloc[0:0].assign(step_idx=[], phase=[], dur_ns=[])
    step_modules = set(steps['module'])
    o = ops[ops['module'].isin(step_modules)]
    cols = [c for c in _META_JOIN_COLS if c in meta.columns]
    o = o.merge(meta[cols], on=['module', 'op'], how='left')
    st = steps[['host', 'device', 'start_ns', 'end_ns', 'phase']].copy()
    st['step_idx'] = st.groupby(['host', 'device']).cumcount()
    st = st.rename(columns={'start_ns': 'step_start', 'end_ns': 'step_end'})
    o = o.sort_values('start_ns')
    st = st.sort_values('step_start')
    merged = pd.merge_asof(
        o,
        st,
        left_on='start_ns',
        right_on='step_start',
        by=['host', 'device'],
        direction='backward',
    )
    merged = merged[merged['start_ns'] < merged['step_end']]
    merged = merged.drop(columns=['step_start', 'step_end'])
    merged['dur_ns'] = merged['end_ns'] - merged['start_ns']
    merged['op_class'] = merged['op_class'].fillna(op_classes.OTHER)
    return merged.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Step statistics
# -----------------------------------------------------------------------------
def phase_stats(steps: pd.DataFrame) -> pd.DataFrame:
    """Per-phase step duration / idle statistics (over all devices)."""
    rows = []
    for phase, g in list(steps.groupby('phase')) + [('all', steps)]:
        if g.empty:
            continue
        n_dev = g.groupby(['host', 'device']).ngroups
        dur = g['dur_ns'].astype(float)
        both = g[['cycle_ns', 'idle_ns']].dropna().astype(float)
        # Cycles far longer than typical are profiler pauses or an idle server
        # (no requests), not per-step host overhead.
        if len(both):
            med = float(both['cycle_ns'].median())
            keep = both['cycle_ns'] <= OUTLIER_CYCLE_FACTOR * med
            n_outliers = int((~keep).sum())
            both = both[keep]
        else:
            n_outliers = 0
        cyc = both['cycle_ns']
        idle = both['idle_ns']
        rows.append({
            'phase':
            phase,
            'n_steps_per_device':
            len(g) / max(n_dev, 1),
            'n_devices':
            n_dev,
            'n_variants':
            g['module'].nunique(),
            'mean_dur_ns':
            dur.mean(),
            'p50_dur_ns':
            dur.quantile(0.5),
            'p90_dur_ns':
            dur.quantile(0.9),
            'cv_dur':
            float(dur.std() / dur.mean()) if dur.mean() else 0.0,
            'mean_cycle_ns':
            cyc.mean() if len(cyc) else np.nan,
            'mean_idle_ns':
            idle.mean() if len(idle) else np.nan,
            'idle_frac':
            float(idle.sum() / cyc.sum()) if cyc.sum() else np.nan,
            'n_outlier_cycles':
            n_outliers,
            'mean_bucket':
            g['bucket'].astype(float).mean(),
            'mean_toks':
            g['toks'].astype(float).mean(),
            'mean_reqs':
            g['reqs'].astype(float).mean(),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Op class time shares
# -----------------------------------------------------------------------------
def class_time(step_ops: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    """Self time per step and share of step time, per (phase, op_class)."""
    if step_ops.empty:
        return pd.DataFrame(
            columns=['phase', 'op_class', 'ns_per_step', 'share'])
    n_steps = steps.groupby('phase').size()
    mean_dur = steps.groupby('phase')['dur_ns'].mean()
    frames = []
    for phase, g in list(step_ops.groupby('phase')) + [('all', step_ops)]:
        ns = n_steps.sum() if phase == 'all' else n_steps.get(phase, 0)
        dur = steps['dur_ns'].mean() if phase == 'all' else mean_dur.get(phase)
        if not ns or not dur:
            continue
        s = g.groupby('op_class')['self_ns'].sum() / ns
        frames.append(
            pd.DataFrame({
                'phase': phase,
                'op_class': s.index,
                'ns_per_step': s.values,
                'share': s.values / dur,
            }))
    if not frames:
        return pd.DataFrame(
            columns=['phase', 'op_class', 'ns_per_step', 'share'])
    out = pd.DataFrame(pd.concat(frames, ignore_index=True))
    return out.sort_values(by=['phase', 'ns_per_step'],
                           ascending=[True, False],
                           ignore_index=True)


def share_of(ct: pd.DataFrame, phase: str, *classes: str) -> float:
    g = ct[(ct['phase'] == phase) & ct['op_class'].isin(classes)]
    return float(g['share'].sum())


def ns_per_step_of(ct: pd.DataFrame, phase: str, *classes: str) -> float:
    g = ct[(ct['phase'] == phase) & ct['op_class'].isin(classes)]
    return float(g['ns_per_step'].sum())


def top_ops(
    step_ops: pd.DataFrame,
    steps: pd.DataFrame,
    phase: str,
    op_class: str | None = None,
    n: int = 5,
) -> pd.DataFrame:
    """Top (module, op) by self time per step within a phase/class."""
    g = step_ops if phase == 'all' else step_ops[step_ops['phase'] == phase]
    if op_class:
        g = g[g['op_class'] == op_class]
    n_steps = (len(steps) if phase == 'all' else int(
        (steps['phase'] == phase).sum()))
    if g.empty or not n_steps:
        return pd.DataFrame(columns=['module', 'op', 'ns_per_step', 'count'])
    agg = g.groupby(['module', 'op']).agg(self_ns=('self_ns', 'sum'),
                                          count=('self_ns', 'size'))
    agg['ns_per_step'] = agg['self_ns'] / n_steps
    agg['count'] = agg['count'] / n_steps
    return agg.sort_values('ns_per_step',
                           ascending=False).head(n).reset_index()


# -----------------------------------------------------------------------------
# Cross-device imbalance (L2)
# -----------------------------------------------------------------------------
def cross_device_imbalance(step_ops: pd.DataFrame, phase: str,
                           op_class: str) -> dict[str, Any]:
    """Per-occurrence max/mean of op duration across devices.

  Devices run the same SPMD program in lockstep, so the k-th occurrence of an
  op in the n-th step is the same logical op on every device. Imbalance in
  data-dependent kernels (e.g. GMM under expert parallelism) shows as spread
  across devices for the same occurrence.

  Args:
    step_ops: Output of assign_ops_to_steps.
    phase: Phase to restrict to ('all' for every phase).
    op_class: Op class to analyze.

  Returns:
    dict with weighted_max_over_mean, worst_device_share, n_devices,
    n_occurrences, mean_ns, max_ns (per occurrence averages).
  """
    g = step_ops[step_ops['op_class'] == op_class]
    if phase != 'all':
        g = g[g['phase'] == phase]
    if g.empty:
        return {}
    g = g.assign(dev=g['host'] + '/' + g['device'])
    g = g.assign(occ=g.groupby(['dev', 'step_idx', 'op']).cumcount())
    piv = g.pivot_table(
        index=['step_idx', 'op', 'occ'],
        columns='dev',
        values='self_ns',
        aggfunc='sum',
    ).dropna()
    if piv.shape[1] < 2 or piv.empty:
        return {}
    mean = piv.mean(axis=1)
    mx = piv.max(axis=1)
    weights = mean / mean.sum()
    per_dev_total = piv.sum(axis=0)
    return {
        'weighted_max_over_mean': float(((mx / mean) * weights).sum()),
        'total_max_over_mean':
        float(per_dev_total.max() / per_dev_total.mean()),
        'slowest_device': str(per_dev_total.idxmax()),
        'n_devices': int(piv.shape[1]),
        'n_occurrences': int(piv.shape[0]),
        'mean_ns': float(mean.mean()),
        'max_ns': float(mx.mean()),
    }


# -----------------------------------------------------------------------------
# Collectives
# -----------------------------------------------------------------------------
def collective_groups(step_ops: pd.DataFrame, meta: pd.DataFrame,
                      steps: pd.DataFrame) -> pd.DataFrame:
    """Per (phase, collective, dtype, dims, moe_scope, layer_scope) stats."""
    c = step_ops[step_ops['op_class'] == op_classes.COLLECTIVE]
    if c.empty:
        return pd.DataFrame()
    c = c.merge(
        meta[['module', 'op', 'out_dtype', 'out_dims']],
        on=['module', 'op'],
        how='left',
    )
    c = c.assign(dims=c['out_dims'].map(lambda d: 'x'.join(map(str, d or ()))))
    n_steps = steps.groupby('phase').size()
    agg = (c.groupby([
        'phase',
        'collective',
        'out_dtype',
        'dims',
        'moe_scope',
        'layer_scope',
    ]).agg(
        count=('self_ns', 'size'),
        total_self_ns=('self_ns', 'sum'),
        mean_self_ns=('self_ns', 'mean'),
        median_self_ns=('self_ns', 'median'),
    ).reset_index())
    agg['per_step'] = agg['count'] / agg['phase'].map(n_steps)
    agg['ns_per_step'] = agg['total_self_ns'] / agg['phase'].map(n_steps)
    return agg


def moe_barrier_excess(coll: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Compares same-shape collectives inside vs outside MoE scope.

  The collective right after the MoE block absorbs the wait for the slowest
  expert shard. If a same-type, same-shape collective outside MoE (e.g. the
  attention o-proj all-reduce) is much faster, the difference is barrier wait
  caused by expert/load imbalance, not bandwidth.

  Args:
    coll: collective_groups output.
    phase: Phase to analyze.

  Returns:
    Rows with collective, dims, moe_mean_ns, ref_mean_ns, excess_ns_per_step.
  """
    if coll.empty:
        return pd.DataFrame()
    c = coll[coll['phase'] == phase]
    rows = []
    keys = ['collective', 'out_dtype', 'dims']
    for key, g in c.groupby(keys):
        moe = g[g['moe_scope']]
        ref = g[~g['moe_scope']]
        if moe.empty or ref.empty:
            continue
        moe_mean = moe['total_self_ns'].sum() / moe['count'].sum()
        ref_med = float(np.average(ref['median_self_ns'],
                                   weights=ref['count']))
        excess = max(0.0, moe_mean - ref_med)
        rows.append({
            'collective': key[0],
            'out_dtype': key[1],
            'dims': key[2],
            'moe_scopes': ','.join(sorted(set(moe['layer_scope']))),
            'ref_scopes': ','.join(sorted(set(ref['layer_scope']))),
            'moe_mean_ns': moe_mean,
            'ref_median_ns': ref_med,
            'ratio': moe_mean / ref_med if ref_med else np.nan,
            'excess_ns_per_step': excess * moe['per_step'].sum(),
        })
    return pd.DataFrame(rows)


def ar_then_slice(ops: pd.DataFrame, meta: pd.DataFrame,
                  steps: pd.DataFrame) -> pd.DataFrame:
    """All-reduces whose next op on the device slices their result.

  This is the "reduce-scatter lowered as all-reduce + dynamic-slice" pattern;
  a native reduce-scatter moves ~1/N of the bytes.

  Args:
    ops: step ops (assign_ops_to_steps output), must include op_class/opcode.
    meta: classified op_meta with kernel_text.
    steps: Segmentation.steps (for per-step normalization).
  """
    if ops.empty:
        return pd.DataFrame()
    kt = meta.set_index(['module', 'op'])['kernel_text']
    rows = []
    for _, g in ops.groupby(['host', 'device']):
        g = g.sort_values('start_ns')
        ar_pos = np.flatnonzero((g['collective'] == 'all-reduce').to_numpy())
        if not len(ar_pos):
            continue
        mods = g['module'].to_numpy()
        names = g['op'].to_numpy()
        opcodes = g['opcode'].fillna('').to_numpy()
        selfs = g['self_ns'].to_numpy()
        phases = g['phase'].to_numpy()
        for i in ar_pos:
            if i + 1 >= len(g):
                continue
            nxt_op, nxt_code = names[i + 1], opcodes[i + 1]
            is_slice = nxt_code in ('dynamic-slice',
                                    'slice') or ('slice' in str(nxt_op))
            if not is_slice:
                continue
            nxt_text = kt.get((mods[i + 1], nxt_op), '')
            ar_name = str(names[i]).split('.')[0]
            if names[i] not in nxt_text and ar_name not in nxt_text:
                continue
            rows.append((phases[i], mods[i], names[i], nxt_op, selfs[i]))
        break  # One device is representative for SPMD programs.
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows,
                      columns=['phase', 'module', 'op', 'next_op', 'self_ns'])
    steps_dev = steps.groupby(['host', 'device']).ngroups or 1
    n_steps = steps.groupby('phase').size() / steps_dev
    agg = (df.groupby(['phase', 'module', 'op', 'next_op'
                       ]).agg(count=('self_ns', 'size'),
                              total_self_ns=('self_ns', 'sum')).reset_index())
    agg['ns_per_step'] = agg['total_self_ns'] / agg['phase'].map(n_steps)
    return agg


# -----------------------------------------------------------------------------
# Host activity during device idle (L3)
# -----------------------------------------------------------------------------
def host_activity_in_gaps(
    steps: pd.DataFrame,
    modules: pd.DataFrame,
    host: pd.DataFrame,
    phase: str = 'all',
    top_n: int = 10,
) -> pd.DataFrame:
    """Host events overlapping device-idle windows on the first device.

  Args:
    steps: Segmentation.steps.
    modules: TraceTable.modules (all programs, used for device busy).
    host: TraceTable.host.
    phase: Restrict to idle windows that precede steps of this phase.
    top_n: Number of host event names to return.

  Returns:
    DataFrame [thread_name, name, overlap_ns, overlap_frac_of_idle].
  """
    if steps.empty or host.empty:
        return pd.DataFrame()
    first = steps.iloc[0]
    dev_steps = steps[(steps['host'] == first['host'])
                      & (steps['device'] == first['device'])]
    if phase != 'all':
        dev_steps = dev_steps[dev_steps['phase'] == phase]
    if dev_steps.empty:
        return pd.DataFrame()
    lo, hi = int(dev_steps['start_ns'].min()), int(
        (dev_steps['start_ns'] + dev_steps['cycle_ns'].fillna(0)).max())
    dev_mods = modules[(modules['host'] == first['host'])
                       & (modules['device'] == first['device'])]
    busy = dev_mods[['start_ns', 'end_ns']].to_numpy(np.int64)
    windows = np.stack(
        [
            dev_steps['start_ns'].to_numpy(np.int64),
            (dev_steps['start_ns'] +
             dev_steps['cycle_ns'].fillna(dev_steps['dur_ns'])).to_numpy(
                 np.int64),
        ],
        axis=1,
    )
    idle = intervals.subtract(windows, busy)
    idle_total = int((idle[:, 1] - idle[:, 0]).sum()) if len(idle) else 0
    if not idle_total:
        return pd.DataFrame()
    h = host[(host['host'] == first['host'])
             & (host['end_ns'] > lo)
             & (host['start_ns'] < hi)]
    med_cycle = float(dev_steps['cycle_ns'].median() or 0)
    if med_cycle:
        h = h[(h['end_ns'] - h['start_ns']) < _HOST_ENCLOSING_CYCLES *
              med_cycle]
    if h.empty:
        return pd.DataFrame()
    h = h.sort_values('start_ns')
    hs = h['start_ns'].to_numpy(np.int64)
    he = h['end_ns'].to_numpy(np.int64)
    overlap = np.zeros(len(h), dtype=np.int64)
    max_len = int((he - hs).max())
    for s, e in idle:
        i0 = np.searchsorted(hs, s - max_len, side='left')
        i1 = np.searchsorted(hs, e, side='left')
        if i1 <= i0:
            continue
        seg_s = np.maximum(hs[i0:i1], s)
        seg_e = np.minimum(he[i0:i1], e)
        overlap[i0:i1] += np.clip(seg_e - seg_s, 0, None)
    h = h.assign(overlap_ns=overlap)
    agg = (h[h['overlap_ns'] > 0].groupby(
        ['thread_name', 'name'])['overlap_ns'].sum().sort_values(
            ascending=False).head(top_n).reset_index())
    agg['overlap_frac_of_idle'] = agg['overlap_ns'] / idle_total
    return agg


def compile_events(host: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    """Host compile events inside the serving window (after the first step)."""
    if host.empty or steps.empty:
        return pd.DataFrame(
            columns=['thread_name', 'name', 'start_ns', 'end_ns'])
    lo = int(steps['start_ns'].min())
    hi = int(steps['end_ns'].max())
    h = host[(host['start_ns'] >= lo) & (host['start_ns'] <= hi)]
    return h[h['name'].str.contains(_COMPILE_RE)].reset_index(drop=True)


def stop_trace_events(host: pd.DataFrame) -> pd.DataFrame:
    if host.empty:
        return host
    return host[host['name'].str.contains(_STOP_TRACE_RE)].reset_index(
        drop=True)


_HOST_COPY_RE = (r'^(ArrayImpl::CopyToDevices|DevicePutWithSharding|'
                 r'BufferFromPyval|TransferToDevice|TransferLiteralToDevice|'
                 r'PjRtClient::BufferFromHostBuffer)')


def host_transfers(table: trace_table_lib.TraceTable) -> dict[str, float]:
    """Host-initiated host->device copies vs device busy time in the profile.

  Returns:
    n_copies (per-shard copy calls), copy_ms (host time in them), span_ms
    (profile span over devices), device_busy_ms (module time per device) and
    n_devices.
  """
    host, mods = table.host, table.modules
    out = {
        'n_copies': 0.0,
        'copy_ms': 0.0,
        'span_ms': 0.0,
        'device_busy_ms': 0.0,
        'n_devices': float(len(table.devices)),
    }
    if not mods.empty:
        out['span_ms'] = float(mods['end_ns'].max() -
                               mods['start_ns'].min()) / 1e6
        out['device_busy_ms'] = (float(
            (mods['end_ns'] - mods['start_ns']).sum()) / 1e6 /
                                 max(len(table.devices), 1))
    if not host.empty:
        c = host[host['name'].str.contains(_HOST_COPY_RE, regex=True)]
        cp = c[c['name'].str.startswith('ArrayImpl::CopyToDevices')]
        out['n_copies'] = float(len(cp) if len(cp) else len(c))
        out['copy_ms'] = float((c['end_ns'] - c['start_ns']).sum()) / 1e6
    return out


# -----------------------------------------------------------------------------
# Shape searches
# -----------------------------------------------------------------------------
def ops_with_dim(step_ops: pd.DataFrame, meta: pd.DataFrame, dim: int,
                 steps: pd.DataFrame) -> pd.DataFrame:
    """Step ops whose output shape contains `dim` (e.g.

  max_position_embeddings).
  """
    hits = meta[meta['out_dims'].map(lambda d: dim in (d or ()))]
    if hits.empty:
        return pd.DataFrame()
    keys = set(zip(hits['module'], hits['op']))
    g = step_ops[[k in keys for k in zip(step_ops['module'], step_ops['op'])]]
    if g.empty:
        return pd.DataFrame()
    n_steps = steps.groupby('phase').size()
    agg = (g.groupby(['phase', 'module', 'op', 'op_class'
                      ]).agg(total_self_ns=('self_ns', 'sum'),
                             count=('self_ns', 'size')).reset_index())
    agg['ns_per_step'] = agg['total_self_ns'] / agg['phase'].map(n_steps)
    return agg.merge(
        meta[['module', 'op', 'out_dtype', 'out_dims', 'tf_op_name']],
        on=['module', 'op'],
        how='left',
    )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def compute(
    table: trace_table_lib.TraceTable,
    seg: segmenter.Segmentation,
    meta: pd.DataFrame,
) -> Features:
    """Computes all features."""
    step_ops = assign_ops_to_steps(table.ops, meta, seg.steps)
    host_gap = {}
    for phase in seg.phases + ['all']:
        host_gap[phase] = host_activity_in_gaps(seg.steps, table.modules,
                                                table.host, phase)
    return Features(
        seg=seg,
        meta=meta,
        step_ops=step_ops,
        phase_stats=phase_stats(seg.steps),
        class_time=class_time(step_ops, seg.steps),
        host_gap=host_gap,
        compile_events=compile_events(table.host, seg.steps),
        stop_trace=stop_trace_events(table.host),
        transfer=host_transfers(table),
    )
