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
"""Serving step segmentation over a TraceTable.

A serving "step" is one execution of the model-forward XLA program on a device.
vLLM-TPU compiles one program variant (fingerprint) per padded token bucket, so
we group modules by *family* (name without the fingerprint) and treat every
fingerprint of the step family as a step, rather than keeping only the dominant
variant.

For each step we compute:
  * the padded token bucket of its program variant (from HLO output shapes),
  * a phase label (decode / prefill / mixed / prefill_or_mixed / unknown),
  * the cycle to the next step, device busy time inside the cycle (union of all
    modules, incl. logits / sampling programs) and idle time,
  * real (unpadded) request / token counts when the host emitted the vLLM-TPU
    `execute_model: N reqs, M toks` TraceAnnotation.
"""

# pylint: disable=missing-function-docstring,missing-class-docstring,g-doc-args,g-doc-return-or-yield,g-docstring-has-escape,g-long-ternary,g-explicit-length-test,g-complex-comprehension,protected-access,cell-var-from-loop,comparison-with-itself,unused-argument

import collections
import dataclasses
import math
import re

import numpy as np
import pandas as pd
from serving_analysis import context as context_lib
from serving_analysis import intervals, op_classes
from serving_analysis import trace_table as trace_table_lib

# Phase labels.
DECODE = 'decode'
PREFILL = 'prefill'
MIXED = 'mixed'
PREFILL_OR_MIXED = 'prefill_or_mixed'
UNKNOWN = 'unknown'

# Known model-forward program families, per framework.
#   vLLM-TPU torchax path: vllm_model_wrapper.step_fun -> 'jit_step_fun'.
#   vLLM-TPU JAX path:     model_loader.run_model      -> 'jit_run_model'.
#   vLLM-TPU fused multi-step decode (additional_config enable_continue_decode):
#                          decode_loop._decode_core    -> 'jit__decode_core'.
#   MaxText MaxEngine:     prefill / generate / insert programs.
CONTINUE_DECODE_FAMILY = 'jit__decode_core'
_VLLM_STEP_FAMILIES = ('jit_step_fun', 'jit_run_model', CONTINUE_DECODE_FAMILY)
_MAXTEXT_PREFILL = re.compile(r'prefill', re.I)
_MAXTEXT_DECODE = re.compile(r'generate', re.I)
# vLLM-TorchTPU (torch_tpu) compiled graphs.
_TORCHTPU_PREFIX = 'tt_jit_'
_TORCHTPU_AUX = re.compile(
    r'sample|gemm_mm|uniform|as_strided|scatter|placeholder|logits', re.I)
_TORCHTPU_MIN_FRACTION = 0.25
_JAX_STEP = re.compile(r'(prefill|decode)_step', re.I)
_FAMILY_RE = re.compile(r'^(.*?)\((\d+)\)$')
_EXEC_MODEL_RE = re.compile(r'execute_model:\s*(\d+)\s*reqs,\s*(\d+)\s*toks')
# Default decode/prefill split for vLLM buckets when max_num_seqs is unknown.
_DEFAULT_DECODE_BUCKET_MAX = 512
# Max distance between a host annotation and the device step it launched.
_ANNOTATION_TOLERANCE_NS = 2_000_000_000
# Speculative decoding methods whose drafts vary in length per request (n-gram
# and suffix lookup). The other drafters (MTP, EAGLE, draft models) schedule
# exactly num_speculative_tokens drafts for each decoding request.
_VARIABLE_DRAFT_METHODS = ('ngram', 'suffix')


def module_family(module: str) -> str:
    """'jit_step_fun(1688...)' -> 'jit_step_fun'."""
    m = _FAMILY_RE.match(module or '')
    return m.group(1) if m else (module or '')


@dataclasses.dataclass
class Segmentation:
    """Result of step segmentation."""

    # 'vllm_tpu' | 'vllm_torchtpu' | 'maxtext' | 'jax' | 'generic'
    framework: str
    step_families: list[str]
    # One row per (host, device, step):
    #   host, device, module, family, start_ns, end_ns, dur_ns, bucket, phase,
    #   cycle_ns, busy_ns, idle_ns, gap_before_ns, reqs, toks
    steps: pd.DataFrame
    # One row per compiled step variant: module, family, bucket, phase,
    # n_steps (per device, max), mean_dur_ns
    variants: pd.DataFrame
    # One row per module family: family, n_fingerprints, n_exec, total_ns
    families: pd.DataFrame
    notes: list[str] = dataclasses.field(default_factory=list)

    def phase_steps(self, phase: str) -> pd.DataFrame:
        if phase == 'all':
            return self.steps
        return self.steps[self.steps['phase'] == phase]

    @property
    def phases(self) -> list[str]:
        return (sorted(self.steps['phase'].unique().tolist())
                if len(self.steps) else [])


def family_table(modules: pd.DataFrame) -> pd.DataFrame:
    df = modules.assign(
        family=modules['module'].map(module_family),
        dur=modules['end_ns'] - modules['start_ns'],
    )
    g = df.groupby('family')
    return (pd.DataFrame({
        'n_fingerprints': g['module'].nunique(),
        'n_exec': g.size(),
        'total_ns': g['dur'].sum(),
    }).reset_index().sort_values('total_ns',
                                 ascending=False,
                                 ignore_index=True))


def _torchtpu_step_families(families: pd.DataFrame) -> list[str]:
    """Model-forward programs of vLLM-TorchTPU.

  torch_tpu names each compiled graph `tt_jit_<file>_L<line>C<col>_<first op>`
  (e.g. `tt_jit_layernorm_L62C0_fused_add_rms_norm_mul` for the whole decoder
  stack), so the name is not stable across models. The forward is the
  top-time family; other families within a fraction of it are also forwards
  (e.g. a separately compiled prefill graph). Sampler / lm_head / scatter
  helpers are excluded by name.
  """
    tt = families[families['family'].str.startswith(_TORCHTPU_PREFIX)]
    tt = tt[~tt['family'].str.contains(_TORCHTPU_AUX)]
    if tt.empty:
        return []
    top = float(tt['total_ns'].iloc[0])
    return tt.loc[tt['total_ns'] >= _TORCHTPU_MIN_FRACTION * top,
                  'family'].tolist()


def pick_step_families(
    families: pd.DataFrame,
    framework: str = 'auto',
) -> tuple[str, list[str]]:
    """Returns (framework, step families) using known names, else top family."""
    names = set(families['family'])
    vllm = [f for f in _VLLM_STEP_FAMILIES if f in names]
    if framework in ('auto', 'vllm_tpu') and vllm:
        return 'vllm_tpu', vllm
    if framework in ('auto', 'vllm_torchtpu'):
        tt = _torchtpu_step_families(families)
        if tt:
            return 'vllm_torchtpu', tt
    mt = [
        f for f in families['family']
        if _MAXTEXT_PREFILL.search(f) or _MAXTEXT_DECODE.search(f)
    ]
    has_generate = any(_MAXTEXT_DECODE.search(f) for f in mt)
    if mt and (framework == 'maxtext' or
               (framework == 'auto' and has_generate)):
        return 'maxtext', mt
    # Custom JAX engines (e.g. forge / raiden_lm): prefill_step / decode_step.
    js = [f for f in families['family'] if _JAX_STEP.search(f)]
    if framework in ('auto', 'jax') and js:
        return 'jax', js
    if families.empty:
        return 'generic', []
    return ('generic' if framework == 'auto' else framework), [
        families['family'].iloc[0]
    ]


def _tokens(dims: tuple[int, ...]) -> int:
    return int(math.prod(dims[:-1])) if len(dims) >= 2 else 0


def infer_token_buckets(
    classified_meta: pd.DataFrame,
    step_modules: list[str],
    hidden_size: int | None,
) -> dict[str, int]:
    """Padded token count per compiled step variant.

  Uses the mode of prod(dims[:-1]) over HLO outputs whose last dim equals the
  model hidden size (residual stream activations are [T, H] or [B, S, H]).
  Falls back to all-reduce outputs (TP/EP partial-sum of [T, H]).

  Args:
    classified_meta: op_meta after `op_classes.classify`.
    step_modules: Module names (with fingerprint) of step programs.
    hidden_size: Model hidden size, if known.

  Returns:
    Mapping module -> bucket (tokens); missing if not inferable.
  """
    out: dict[str, int] = {}
    meta = classified_meta[classified_meta['module'].isin(step_modules)]
    for module, grp in meta.groupby('module'):
        counts: collections.Counter[int] = collections.Counter()
        if hidden_size:
            for dims in grp['out_dims']:
                if len(dims) >= 2 and dims[-1] == hidden_size:
                    counts[_tokens(dims)] += 1
        if not counts:
            ar = grp[grp['collective'] == 'all-reduce']
            for dims in ar['out_dims']:
                t = _tokens(dims)
                if t:
                    counts[t] += 1
        counts.pop(0, None)
        if counts:
            out[module] = counts.most_common(1)[0][0]
    return out


def _phase_for_variant(
    framework: str,
    family: str,
    bucket: int | None,
    decode_bucket_max: int,
) -> str:
    if framework in ('maxtext', 'jax'):
        if _MAXTEXT_PREFILL.search(family):
            return PREFILL
        if _MAXTEXT_DECODE.search(family) or re.search(r'decode', family,
                                                       re.I):
            return DECODE
        return UNKNOWN
    if bucket is None:
        return DECODE if family == CONTINUE_DECODE_FAMILY else UNKNOWN
    if family == CONTINUE_DECODE_FAMILY:
        return DECODE
    return DECODE if bucket <= decode_bucket_max else PREFILL_OR_MIXED


def _annotations(host: pd.DataFrame) -> pd.DataFrame:
    if host.empty:
        return pd.DataFrame(columns=['host', 'start_ns', 'reqs', 'toks'])
    mask = host['name'].str.contains('execute_model:', regex=False, na=False)
    ann = host[mask]
    parsed = ann['name'].str.extract(_EXEC_MODEL_RE)
    df = pd.DataFrame({
        'host': ann['host'],
        'start_ns': ann['start_ns'].astype('int64'),
        'reqs': pd.to_numeric(parsed[0], errors='coerce'),
        'toks': pd.to_numeric(parsed[1], errors='coerce'),
    }).dropna()
    return df.sort_values('start_ns', ignore_index=True)


def _per_device_timing(dev_modules: pd.DataFrame,
                       step_mask: np.ndarray) -> pd.DataFrame:
    """cycle / busy / idle / gap_before for step rows of one device."""
    mods = dev_modules
    starts = mods['start_ns'].to_numpy(np.int64)
    ends = mods['end_ns'].to_numpy(np.int64)
    step_idx = np.flatnonzero(step_mask)
    rows = []
    prev_end_running = np.maximum.accumulate(ends) if len(ends) else ends
    for n, i in enumerate(step_idx):
        s = starts[i]
        gap_before = (s - prev_end_running[i - 1]) if i > 0 else np.nan
        if n + 1 < len(step_idx):
            nxt = starts[step_idx[n + 1]]
            in_cycle = (starts >= s) & (starts < nxt)
            busy_iv = np.stack(
                [starts[in_cycle],
                 np.minimum(ends[in_cycle], nxt)], axis=1)
            busy = intervals.total_length(busy_iv)
            cycle = nxt - s
            rows.append((cycle, busy, cycle - busy, gap_before))
        else:
            rows.append((np.nan, np.nan, np.nan, gap_before))
    return pd.DataFrame(
        rows, columns=['cycle_ns', 'busy_ns', 'idle_ns', 'gap_before_ns'])


def segment(
    table: trace_table_lib.TraceTable,
    ctx: context_lib.ServingContext | None = None,
    classified_meta: pd.DataFrame | None = None,
) -> Segmentation:
    """Segments a TraceTable into serving steps."""
    ctx = (ctx or context_lib.ServingContext()).finalize()
    notes: list[str] = []
    families = family_table(table.modules)
    framework, step_fams = pick_step_families(families, ctx.framework)
    if framework == 'generic' and step_fams:
        notes.append('No known serving program found; using top module family '
                     f'{step_fams[0]!r} as the step.')
    if classified_meta is None:
        classified_meta = op_classes.classify(table.op_meta)

    mods = table.modules.assign(
        family=table.modules['module'].map(module_family))
    step_modules = sorted(mods.loc[mods['family'].isin(step_fams),
                                   'module'].unique())
    buckets = infer_token_buckets(classified_meta, step_modules,
                                  ctx.model.hidden_size)
    toks_per_seq = ctx.decode_tokens_per_req
    decode_max = (ctx.max_num_seqs * toks_per_seq
                  if ctx.max_num_seqs else _DEFAULT_DECODE_BUCKET_MAX)
    if framework in ('vllm_tpu', 'vllm_torchtpu') and not ctx.max_num_seqs:
        notes.append(
            f'max_num_seqs unknown; token buckets <= {decode_max} are labelled '
            'decode.')

    step_frames = []
    for _, dev in mods.groupby(['host', 'device'], sort=False):
        dev = dev.sort_values('start_ns', ignore_index=True)
        mask = dev['family'].isin(step_fams).to_numpy()
        if not mask.any():
            continue
        timing = _per_device_timing(dev, mask)
        st = dev[mask].reset_index(drop=True)
        st = pd.concat([st, timing], axis=1)
        step_frames.append(st)
    if not step_frames:
        empty = pd.DataFrame(columns=[
            'host',
            'device',
            'module',
            'family',
            'start_ns',
            'end_ns',
            'dur_ns',
            'bucket',
            'phase',
            'cycle_ns',
            'busy_ns',
            'idle_ns',
            'gap_before_ns',
            'reqs',
            'toks',
        ])
        notes.append('No step programs found on any TPU device.')
        return Segmentation(framework, step_fams, empty, pd.DataFrame(),
                            families, notes)
    steps = pd.DataFrame(pd.concat(step_frames, ignore_index=True))
    steps['dur_ns'] = steps['end_ns'] - steps['start_ns']
    steps['bucket'] = steps['module'].map(buckets).astype('Int64')
    steps['phase'] = [
        _phase_for_variant(framework, f, None if pd.isna(b) else int(b),
                           decode_max)
        for f, b in zip(steps['family'], steps['bucket'])
    ]

    # Real request/token counts from host annotations (vLLM-TPU only).
    ann = _annotations(table.host)
    steps['reqs'] = np.nan
    steps['toks'] = np.nan
    if not ann.empty:
        steps = steps.sort_values(by='start_ns', ignore_index=True)
        merged = pd.merge_asof(
            steps[['host', 'start_ns']].reset_index(),
            ann,
            on='start_ns',
            by='host',
            direction='backward',
            tolerance=_ANNOTATION_TOLERANCE_NS,
        ).set_index('index')
        steps['reqs'] = merged['reqs']
        steps['toks'] = merged['toks']
        # Refine vLLM labels: a step whose tokens exceed its requests has prefill
        # tokens in it. With speculative decoding a decoding request carries
        # 1 + num_speculative_tokens tokens, so a step is decode only when its
        # tokens are exactly that many per request: fewer can be a short
        # prefill chunk beside the decodes. Drafters whose drafts vary in
        # length only bound the count, so any step within the bound is decode.
        has = steps['reqs'].notna() & (framework == 'vllm_tpu')
        full = steps['reqs'] * toks_per_seq
        method = str(ctx.speculative_method or '').lower()
        if toks_per_seq > 1 and method not in _VARIABLE_DRAFT_METHODS:
            is_decode = steps['toks'] == full
        else:
            is_decode = steps['toks'] <= full
        steps.loc[has & is_decode, 'phase'] = DECODE
        steps.loc[has & ~is_decode, 'phase'] = MIXED
        notes.append(
            f'Matched {int(has.sum())}/{len(steps)} steps to host execute_model '
            'annotations.')
    else:
        notes.append(
            'No host `execute_model:` annotations; real batch sizes unknown, '
            'phases inferred from token buckets.')

    steps = steps[[
        'host',
        'device',
        'module',
        'family',
        'start_ns',
        'end_ns',
        'dur_ns',
        'bucket',
        'phase',
        'cycle_ns',
        'busy_ns',
        'idle_ns',
        'gap_before_ns',
        'reqs',
        'toks',
    ]].sort_values(['host', 'device', 'start_ns'], ignore_index=True)

    per_dev = steps.groupby(['module', 'host', 'device']).size().rename('n')
    vg = steps.groupby('module')
    variants = (pd.DataFrame({
        'family':
        vg['family'].first(),
        'bucket':
        vg['bucket'].first(),
        'phase':
        vg['phase'].agg(lambda s: s.mode().iloc[0]),
        'n_steps':
        per_dev.groupby('module').max(),
        'mean_dur_ns':
        vg['dur_ns'].mean(),
    }).reset_index().sort_values('bucket', ignore_index=True))
    return Segmentation(framework, step_fams, steps, variants, families, notes)
