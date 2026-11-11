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
"""End-to-end serving analysis pipeline.

TraceTable -> classify ops -> segment steps -> features -> detectors ->
ranked findings (+ JSON/Markdown and a step summary for the agent).
"""

# pylint: disable=missing-function-docstring,missing-class-docstring,g-doc-args,g-doc-return-or-yield,g-docstring-has-escape,g-long-ternary,g-explicit-length-test,g-complex-comprehension,protected-access,cell-var-from-loop,comparison-with-itself,unused-argument

import dataclasses
import json
from typing import Any

import pandas as pd
from serving_analysis import catalog_detectors, code_locations
from serving_analysis import context as context_lib
from serving_analysis import detectors
from serving_analysis import features as features_lib
from serving_analysis import findings as findings_lib
from serving_analysis import (guardrails, headroom_detectors, op_classes,
                              planner, segmenter)
from serving_analysis import trace_table as trace_table_lib

# Hot code locations per phase in the Markdown report.
_HOT_CODE_ROWS = 5


@dataclasses.dataclass
class AnalysisResult:
    session_id: str
    framework: str
    findings: list[findings_lib.Finding]
    features: features_lib.Features
    notes: list[str]
    errors: list[str]
    serving_command: str | None = None
    # Phase -> the source lines with the most step time (code_locations).
    hotspots: dict[str, list[findings_lib.CodeLocation]] = dataclasses.field(
        default_factory=dict)

    @property
    def ranked(self) -> list[findings_lib.Finding]:
        return sorted(self.findings, key=lambda f: f.score, reverse=True)

    @property
    def recommended_command(self) -> str:
        changes = [kc for x in self.ranked for kc in x.knob_changes]
        return context_lib.apply_knob_changes(self.serving_command, changes,
                                              self.framework)

    def summary_dict(self) -> dict[str, Any]:
        """Compact, JSON-serializable overview (for the agent / report)."""
        f = self.features
        ps = f.phase_stats.copy()
        for c in ps.columns:
            if c.endswith('_ns'):
                ps[c.replace('_ns', '_ms')] = ps.pop(c) / 1e6
        ct = f.class_time[f.class_time['share'] >= 0.01].copy()
        ct['ms_per_step'] = ct.pop('ns_per_step') / 1e6
        variants = f.seg.variants.copy()
        if 'mean_dur_ns' in variants:
            variants['mean_dur_ms'] = variants.pop('mean_dur_ns') / 1e6
        phase_records = _records(ps)
        ranked_dicts = [x.to_dict() for x in self.ranked]
        return {
            'session_id':
            self.session_id,
            'framework':
            self.framework,
            'step_families':
            f.seg.step_families,
            'phase_stats':
            phase_records,
            'step_variants':
            _records(variants),
            'class_share':
            _records(ct),
            'top_ops':
            _top_ops_records(f),
            'code_hotspots': {
                phase: [dataclasses.asdict(loc) for loc in locs]
                for phase, locs in self.hotspots.items()
            },
            'notes':
            self.notes,
            'errors':
            self.errors,
            'findings':
            ranked_dicts,
            'sharding_constraints':
            planner.extract_sharding_constraints({
                'phase_stats': phase_records,
                'findings': ranked_dicts,
            }),
            'recommended_command':
            self.recommended_command,
        }

    def to_markdown(self) -> str:
        f = self.features
        lines = [f'# Serving analysis: `{self.session_id}`', '']
        lines.append(f'Framework: **{self.framework}**; step programs: ' +
                     ', '.join(f'`{s}`' for s in f.seg.step_families))
        lines.append('')
        lines.append(
            '| Phase | Steps/dev | Variants | Mean step (ms) | p90 (ms) |'
            ' Idle % | Mean bucket |')
        lines.append('|---|---|---|---|---|---|---|')
        for _, r in f.phase_stats.iterrows():
            lines.append(
                f'| {r["phase"]} | {r["n_steps_per_device"]:.0f} |'
                f' {r["n_variants"]} | {r["mean_dur_ns"] / 1e6:.2f} |'
                f' {r["p90_dur_ns"] / 1e6:.2f} |'
                f' {100 * (r["idle_frac"] if pd.notna(r["idle_frac"]) else 0):.1f} |'
                f' {r["mean_bucket"]:.0f} |')
        lines.append('')
        if self.notes:
            lines.append('Notes: ' + ' '.join(self.notes))
            lines.append('')
        if self.errors:
            lines.append('Detector errors: ' + '; '.join(self.errors))
            lines.append('')
        hot = [(p, locs[:_HOT_CODE_ROWS]) for p, locs in self.hotspots.items()]
        if any(locs for _, locs in hot):
            lines.append(
                '**Hot code** (step time by the source line that emitted it)')
            lines.append('')
            for phase, locs in hot:
                if locs:
                    lines.append(f'{phase}:')
                    lines.append('')
                    lines.extend(
                        findings_lib.location_markdown(loc) for loc in locs)
                    lines.append('')
        rec_cmd = self.recommended_command
        if rec_cmd and rec_cmd != (self.serving_command or ''):
            lines.append('## Ready-to-run configuration patch')
            lines.append('```bash')
            lines.append(rec_cmd)
            lines.append('```')
            lines.append('')
        lines.append(findings_lib.findings_to_markdown(self.findings))
        return '\n'.join(lines)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    return json.loads(
        df.to_json(orient='records', default_handler=str) or '[]')


_TOP_OPS_PER_PHASE = 25


def _top_ops_records(f: features_lib.Features) -> list[dict[str, Any]]:
    """Top ops (summed over per-layer instances) per phase, with class."""
    so = f.step_ops
    steps = f.seg.steps
    if so.empty or steps.empty:
        return []
    tf = (so['tf_op_name'].fillna('')
          if 'tf_op_name' in so else pd.Series('', index=so.index))
    so = so.assign(
        base_op=so['base_op'].fillna(so['op'])
        if 'base_op' in so else so['op'],
        scope=tf.map(op_classes.base_op_name).str.slice(0, 160),
        hlo_category=so['hlo_category'].fillna(''),
        tf_op_type=so['tf_op_type'].fillna(''),
    )
    n_steps = steps.groupby('phase').size()
    mean_dur = steps.groupby('phase')['dur_ns'].mean()
    frames = []
    for phase, g in so.groupby('phase'):
        ns = n_steps.get(phase, 0)
        dur = mean_dur.get(phase, 0)
        if not ns or not dur:
            continue
        agg = (g.groupby(
            ['base_op', 'op_class', 'hlo_category', 'tf_op_type',
             'scope']).agg(self_ns=('self_ns', 'sum'),
                           count=('self_ns', 'size')).reset_index())
        agg['ms_per_step'] = agg['self_ns'] / ns / 1e6
        agg['count_per_step'] = agg['count'] / ns
        agg['share'] = agg['self_ns'] / ns / dur
        agg['phase'] = phase
        agg = agg.sort_values(by='self_ns',
                              ascending=False).head(_TOP_OPS_PER_PHASE)
        frames.append(agg.drop(columns=['self_ns', 'count']))
    if not frames:
        return []
    return _records(pd.DataFrame(pd.concat(frames, ignore_index=True)))


def analyze(
    table: trace_table_lib.TraceTable,
    ctx: context_lib.ServingContext | None = None,
    session_id: str | None = None,
) -> AnalysisResult:
    """Runs the full pipeline on a loaded TraceTable."""
    ctx = (ctx or context_lib.ServingContext()).finalize()
    sid = session_id if session_id is not None else table.session_id
    meta = op_classes.classify(table.op_meta)
    seg = segmenter.segment(table, ctx, meta)
    feats = features_lib.compute(table, seg, meta)
    found, errors = detectors.run_all(
        feats,
        ctx,
        sid,
        detectors=(detectors.ALL_DETECTORS + catalog_detectors.ALL_DETECTORS +
                   headroom_detectors.ALL_DETECTORS),
    )
    found, guard_notes = guardrails.apply(found, ctx)
    fw = ctx.framework if ctx.framework != 'auto' else seg.framework
    notes = list(seg.notes) + guard_notes
    if ctx.serving_config is None:
        notes.append('No serving config provided: knob states are unknown.')
    hotspots: dict[str, list[findings_lib.CodeLocation]] = {}
    try:
        index = code_locations.LocationIndex(feats)
        code_locations.attach(found, index)
        hotspots = index.hotspots()
    except Exception as e:  # pylint: disable=broad-exception-caught
        errors.append(f'code_locations: {type(e).__name__}: {e}')
    return AnalysisResult(sid,
                          fw,
                          found,
                          feats,
                          notes,
                          errors,
                          serving_command=ctx.serving_command,
                          hotspots=hotspots)
