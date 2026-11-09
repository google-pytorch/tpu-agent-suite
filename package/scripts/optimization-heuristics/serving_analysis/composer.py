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
"""Independent-first Stage 3: compose serving and sharding candidates.

Knob and mesh proposals are made on their own:

  * serving candidates change only serving / kernel knobs on the baseline mesh
    (one per serving finding with an unset knob change);
  * mesh candidates change only the mesh (from `oss_meshes.enumerate_oss_meshes`
    or a mesh plan in markdown), with serving knobs left at baseline.

`compose_plan` keeps them as separate candidates, each tested alone against a
same-session control, and applies three small rules:

  (a) coupled pairs: a knob that only works together with a mesh property
      (e.g. the fused-EP MoE kernel needs `--enable-expert-parallel`) is paired
      with a matching mesh candidate into one candidate, or rejected if no
      mesh candidate provides the property;
  (b) every mesh change marks the kernel-shape findings that depend on the
      changed axes as "re-measure" (planner R3);
  (c) conflicting proposals are rejected: serving candidates that change mesh
      axes (that is the mesh step's job), candidates that set one flag to
      two values, and duplicate meshes.

Mesh candidates from `enumerate_oss_meshes` with `points` carry a
per-benchmark-point assessment; the plan renders one row per (candidate x
point) and labels a candidate "workload-dependent" when the expected
directions disagree. Every mesh candidate that changes TP or DP gets a risk
line: the reduction order changes, so greedy outputs may differ.

Every candidate carries a provenance record (skill, finding IDs, tool
version). Combinations of winners are not predicted here; they are one more
measured trial after the A/B results are in.
"""

import dataclasses
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from serving_analysis import planner

TOOL_VERSION = 'compose_plan/1'

_PROVENANCE_LINE_RE = re.compile(
    r'(?im)^[ \t]*[-*]?\s*\*{0,2}Provenance\*{0,2}\s*:\s*\*{0,2}\s*(.+)$')
_DETECTOR_ID_RE = re.compile(r'\bD\d+[a-z]?_[a-z0-9_]+')


@dataclasses.dataclass(frozen=True)
class CoupledPair:
    """A knob that only takes effect together with a mesh property."""

    knob: str  # Literal knob name.
    value: str  # Value that needs the mesh property.
    axis: str  # A planner axis class: 'ep' | 'tp' | 'dp' | 'pp' | ...
    min_size: int
    why: str


COUPLED_PAIRS: tuple[CoupledPair, ...] = (
    CoupledPair(
        'USE_MOE_FUSED_EP_KERNEL',
        '1',
        'ep',
        2,
        'the fused EP MoE kernel runs only with --enable-expert-parallel',
    ),
    CoupledPair(
        'MOE_FUSED_EP_KERNEL_MIN_TOKENS',
        '*',
        'ep',
        2,
        'only read by the fused EP MoE kernel',
    ),
    CoupledPair(
        'sharding.sharding_strategy.enable_dp_attention',
        'true',
        'tp',
        2,
        'attention DP replaces KV-head replication under TP>1',
    ),
)

# Knobs that set a mesh axis; serving candidates may not carry them.
_MESH_KNOB_KEYS = frozenset({
    'tensor_parallel_size',
    'data_parallel_size',
    'enable_expert_parallel',
    'ici_tensor_parallelism',
    'ici_data_parallelism',
    'ici_expert_parallelism',
    'ici_fsdp_parallelism',
    'pipeline_parallel_size',
    'decode_context_parallel_size',
    'prefill_context_parallel_size',
})
# Mesh axes compared for duplicates (planner axis classes).
_MESH_AXES = ('tp', 'dp', 'ep', 'fsdp', 'pp', 'dcp', 'pcp')
# vLLM CLI flag per value axis (vllm/engine/arg_utils.py).
_VLLM_AXIS_FLAGS = (
    ('tp', '--tensor-parallel-size'),
    ('dp', '--data-parallel-size'),
    ('pp', '--pipeline-parallel-size'),
    ('dcp', '--decode-context-parallel-size'),
    ('pcp', '--prefill-context-parallel-size'),
)


def _knob_key(name: str) -> str:
    return name.lstrip('-').replace('-', '_').lower().rsplit('.', 1)[-1]


@dataclasses.dataclass
class Candidate:
    """One A/B trial: a delta against the baseline launch."""

    cand_id: str
    skill: str  # 'serving-optimizer' | 'sharding'
    findings: list[str]
    title: str
    rationale: str
    argv_set: dict[str, str] = dataclasses.field(default_factory=dict)
    argv_add_flags: list[str] = dataclasses.field(default_factory=list)
    argv_remove_flags: list[str] = dataclasses.field(default_factory=list)
    env_set: dict[str, str] = dataclasses.field(default_factory=dict)
    additional_config_set: dict[str,
                                str] = dataclasses.field(default_factory=dict)
    mesh: dict[str, int] | None = None
    evidence: list[str] = dataclasses.field(default_factory=list)
    expected_mechanism: str = ''
    risks: list[str] = dataclasses.field(default_factory=list)
    remeasure: list[str] = dataclasses.field(default_factory=list)
    target_points: list[str] = dataclasses.field(default_factory=list)
    # Profile roles whose findings back the candidate (prefill/decode/mixed).
    profiles: list[str] = dataclasses.field(default_factory=list)
    target_regime: str = ''
    score: float = 0.0
    tool: str = ''
    # Benchmark point label -> {regime, score, expected_direction, rationale}.
    per_point: dict[str, dict[str,
                              Any]] = dataclasses.field(default_factory=dict)

    @property
    def workload_dependent(self) -> bool:
        dirs = {
            str(a.get('expected_direction'))
            for a in self.per_point.values()
        }
        return {'positive', 'negative'} <= dirs

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            'skill': self.skill,
            'findings': list(self.findings),
            'tool': self.tool,
            'tool_version': TOOL_VERSION,
        }

    @property
    def provenance_line(self) -> str:
        f = ', '.join(self.findings) or 'none'
        return (
            f'Provenance: skill={self.skill}; findings={f}; tool={self.tool};'
            f' version={TOOL_VERSION}')

    def flag_values(self) -> dict[str, str]:
        out = dict(self.argv_set)
        out.update({f: 'true' for f in self.argv_add_flags})
        out.update(self.env_set)
        out.update(self.additional_config_set)
        return out

    def to_json(self, rank: int) -> dict[str, Any]:
        """Entry in the overnight harness `candidates.json` format."""
        return {
            'cand_id':
            self.cand_id,
            'rank':
            rank,
            'argv_set':
            dict(self.argv_set),
            'argv_add_flags':
            list(self.argv_add_flags),
            'argv_remove_flags':
            list(self.argv_remove_flags),
            'env_set':
            dict(self.env_set),
            'check_env_knobs':
            sorted(self.env_set),
            'check_cli_flags':
            sorted(set(self.argv_set) | set(self.argv_add_flags)),
            'rationale':
            self.rationale,
            'evidence':
            list(self.evidence),
            'expected_mechanism':
            self.expected_mechanism,
            'target_points':
            list(self.target_points),
            'risks':
            '; '.join(self.risks),
            'remeasure':
            list(self.remeasure),
            'profiles':
            list(self.profiles),
            'target_regime':
            self.target_regime,
            'additional_config_set':
            dict(self.additional_config_set),
            'per_point_assessment':
            dict(self.per_point),
            'workload_dependent':
            self.workload_dependent,
            'provenance':
            self.provenance,
        }


def _slug(text: str, n: int = 24) -> str:
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')[:n].strip('-')


def _apply_knob(c: Candidate, kc: Mapping[str, Any]) -> None:
    name, kind = str(kc['name']), str(kc.get('kind', ''))
    value = str(kc['suggested_value'])
    if kind == 'env':
        c.env_set[name] = value
    elif kind == 'cli':
        flag = name if name.startswith('--') else f'--{name}'
        if value.lower() == 'true':
            c.argv_add_flags.append(flag)
        elif value.lower() == 'false':
            c.argv_remove_flags.append(flag)
            c.argv_add_flags.append('--no-' + flag[2:])
        else:
            c.argv_set[flag] = value
    else:
        c.additional_config_set[name] = value


def serving_candidates(
    serving_payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[list[Candidate], list[str]]:
    """One candidate per serving finding with knob changes not already set.

  Findings of the same detector across profiles (e.g. prefill and decode
  captures) that recommend the same changes are merged into one candidate.

  Args:
    serving_payload: serving_analyze_main output (single or --case).

  Returns:
    (candidates, rejection notes)
  """
    by_key: dict[str, Candidate] = {}
    rejected: list[str] = []
    for f in planner.normalize_findings(serving_payload):
        changes = [
            kc for kc in f.get('knob_changes') or []
            if kc.get('already_set') is not True
        ]
        if not changes:
            continue
        det = str(f.get('detector_id', ''))
        mesh_knobs = [
            kc['name'] for kc in changes
            if _knob_key(kc['name']) in _MESH_KNOB_KEYS
        ]
        if mesh_knobs:
            rejected.append(
                f'{det}: serving candidate would change the mesh ({", ".join(mesh_knobs)});'
                ' mesh changes come from the mesh step (rule c).')
            changes = [kc for kc in changes if kc['name'] not in mesh_knobs]
            if not changes:
                continue
        sig = det + json.dumps(
            sorted((kc['name'], str(kc['suggested_value'])) for kc in changes))
        role = f.get('profile_role') or ''
        if sig in by_key:
            c = by_key[sig]
            if role and role not in c.profiles:
                c.profiles.append(role)
            c.score = max(c.score, float(f.get('score', 0.0)))
            continue
        c = Candidate(
            cand_id='s-' + _slug(det.split('_', 1)[-1] or det),
            skill='serving-optimizer',
            findings=[det],
            title=str(f.get('title', det)),
            rationale=str(f.get('summary', '')),
            evidence=[
                f'{e.get("metric")}: {e.get("value")}{e.get("unit", "")}'
                for e in (f.get('evidence') or [])[:3]
            ],
            expected_mechanism=str(f.get('validation', '')),
            profiles=[role] if role else [],
            score=float(f.get('score', 0.0)),
            tool='serving_analyze_main',
        )
        for kc in changes:
            _apply_knob(c, kc)
            if kc.get('requires_patch'):
                c.risks.append(
                    f'`{kc["name"]}` exists only with {kc["requires_patch"]}; verify'
                    ' the image carries it before launch')
        if f.get('risk') not in (None, '', 'perf_only'):
            c.risks.append(f'risk class: {f.get("risk")}')
        by_key[sig] = c
    return sorted(by_key.values(), key=lambda c: -c.score), rejected


def _mesh_args(
    mesh: Mapping[str, int],
    baseline: Mapping[str, int] | None = None
) -> tuple[dict[str, str], list[str]]:
    """vLLM flags for a mesh: TP and DP always, PP / DCP / PCP when > 1 here.

  PP / DCP / PCP are also emitted (as 1) when the baseline sets them, so the
  candidate overrides the baseline launch.

  Args:
    mesh: Candidate mesh.
    baseline: Baseline mesh.
  """
    size = planner._axis_size  # pylint: disable=protected-access
    argv = {}
    for axis, flag in _VLLM_AXIS_FLAGS:
        n = size(mesh, axis)
        if axis in ('tp', 'dp') or n > 1 or size(baseline or {}, axis) > 1:
            argv[flag] = str(n)
    return argv, (['--enable-expert-parallel'] if size(mesh, 'ep') > 1 else [])


def _merge_unique(first: list[str], more: Sequence[str]) -> list[str]:
    return first + [x for x in more if x not in first]


def sharding_candidates_from_markdown(
    md: str, ) -> tuple[list[Candidate], list[str]]:
    """Mesh candidates from a markdown mesh plan (serving knobs stripped)."""
    lines = md.splitlines()
    out, notes = [], []
    for label, start, end in planner._candidate_blocks(lines):  # pylint: disable=protected-access
        block = '\n'.join(lines[start:end])
        m = planner._ICI_MESH_LINE_RE.search(block)  # pylint: disable=protected-access
        if not m:
            continue
        mesh = planner._parse_simple_mesh(m.group(2))  # pylint: disable=protected-access
        if planner._SERVING_KNOBS_LINE_RE.search(block):  # pylint: disable=protected-access
            notes.append(
                f'sharding {label}: dropped its Serving Knobs line (independent'
                ' mode: sharding candidates keep serving knobs at baseline).')
        prov = _PROVENANCE_LINE_RE.search(block)
        findings = sorted(set(_DETECTOR_ID_RE.findall(block)))
        header = lines[start].lstrip('#').strip()
        c = Candidate(
            cand_id='m-' + _slug(header.split(':', 1)[-1] or label),
            skill='sharding',
            findings=findings,
            title=header,
            rationale=re.sub(r'\s+', ' ', block)[:600],
            mesh=mesh,
            tool='mesh plan',
        )
        if prov:
            c.tool = f'mesh plan ({prov.group(1).strip()})'
        out.append(c)
    return out, notes


def sharding_candidates_from_oss_meshes(
    result: Mapping[str, Any], ) -> list[Candidate]:
    """Mesh candidates from `oss_meshes.enumerate_oss_meshes` output."""
    out = []
    findings = ['D7_collective_bound'
                ] if result.get('prefill_collective_bound') else []
    for cand in result.get('candidates') or []:
        out.append(
            Candidate(
                cand_id='m-' + _slug(str(cand['label'])),
                skill='sharding',
                findings=list(findings),
                title=f'Mesh {cand["label"]}',
                rationale=str(cand.get('rationale', '')),
                mesh={
                    k: int(v)
                    for k, v in cand['mesh'].items()
                },
                argv_set={
                    str(k): str(v)
                    for k, v in (cand.get('argv_set') or {}).items()
                },
                argv_add_flags=list(cand.get('argv_add_flags') or []),
                risks=list(cand.get('risks') or []),
                target_regime=str(cand.get('target_regime', '')),
                score=float(cand.get('score', 0.0)),
                tool='oss_meshes.enumerate_oss_meshes',
                per_point={
                    str(k): dict(v)
                    for k, v in (
                        cand.get('per_point_assessment') or {}).items()
                },
            ))
    return out


@dataclasses.dataclass
class ComposedPlan:
    candidates: list[Candidate]
    rejected: list[str]
    notes: list[str]
    markdown: str

    def candidates_json(self) -> list[dict[str, Any]]:
        return [c.to_json(i) for i, c in enumerate(self.candidates, 1)]


_DETERMINISM_MARKER = 'reduction order changes'


def _determinism_risk(mesh: Mapping[str, int],
                      baseline: Mapping[str, int] | None) -> str | None:
    """Risk line when a mesh changes TP or DP vs. the baseline."""
    if baseline is None:
        return None
    size = planner._axis_size  # pylint: disable=protected-access
    changed = [a for a in ('tp', 'dp') if size(mesh, a) != size(baseline, a)]
    if not changed:
        return None
    return (
        f'{_DETERMINISM_MARKER} ({", ".join(a.upper() for a in changed)}'
        ' changed): greedy outputs may differ from the baseline; check greedy'
        ' parity before comparing quality')


def _coupling(c: Candidate) -> list[CoupledPair]:
    vals = c.flag_values()
    out = []
    for p in COUPLED_PAIRS:
        v = vals.get(p.knob)
        if v is not None and (p.value == '*' or v.lower() == p.value):
            out.append(p)
    return out


def compose_plan(
    serving_payload: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    target_points: Sequence[str] = (),
    sharding_markdown: str = '',
    oss_meshes: Mapping[str, Any] | None = None,
    baseline_mesh: Mapping[str, int] | None = None,
    vllm: bool = True,
) -> ComposedPlan:
    """Builds the independent-first candidate list and implementation plan.

  Args:
    serving_payload: serving_analyze_main output (single or --case).
    sharding_markdown: A markdown mesh plan (mesh-only candidates).
    oss_meshes: `enumerate_oss_meshes` output (alternative mesh source).
    baseline_mesh: Baseline mesh, for the re-measure rule.
    vllm: Emit vLLM CLI flags for mesh candidates.
    target_points: Benchmark points every candidate is measured at.

  Returns:
    The composed plan.
  """
    rejected: list[str] = []
    notes: list[str] = []
    serving: list[Candidate] = []
    if serving_payload:
        serving, rej = serving_candidates(serving_payload)
        rejected += rej
    meshes: list[Candidate] = []
    if sharding_markdown.strip():
        md_c, md_notes = sharding_candidates_from_markdown(sharding_markdown)
        meshes += md_c
        notes += md_notes
    if oss_meshes:
        meshes += sharding_candidates_from_oss_meshes(oss_meshes)
    findings = (planner.normalize_findings(serving_payload)
                if serving_payload else [])

    # (c) duplicate / baseline meshes.
    uniq: list[Candidate] = []
    seen: dict[tuple[tuple[str, int], ...], Candidate] = {}
    for c in meshes:
        assert c.mesh is not None
        key = tuple(
            sorted((a, planner._axis_size(c.mesh, a))  # pylint: disable=protected-access
                   for a in _MESH_AXES))
        if baseline_mesh and not planner._changed_axis_classes(
                c.mesh, baseline_mesh):  # pylint: disable=protected-access
            notes.append(f'{c.title}: same as the baseline mesh; not a trial.')
            continue
        if key in seen:
            # Keep the first copy but do not lose what the duplicate knew (e.g.
            # the enumerator's risks and target regime under a plan candidate).
            first = seen[key]
            first.findings = sorted(set(first.findings) | set(c.findings))
            first.risks = _merge_unique(first.risks, c.risks)
            first.argv_set = {**c.argv_set, **first.argv_set}
            first.target_regime = first.target_regime or c.target_regime
            first.score = max(first.score, c.score)
            first.per_point = first.per_point or c.per_point
            rejected.append(f'{c.title}: duplicate of {first.title} (rule c).')
            continue
        seen[key] = c
        uniq.append(c)
    meshes = uniq

    # (b) re-measure after a mesh change; vLLM flags.
    for c in meshes:
        assert c.mesh is not None
        c.remeasure = planner.compute_recheck_findings(c.mesh, baseline_mesh,
                                                       findings)
        if vllm:
            # Mesh flags are recomputed from the mesh; other argv (e.g. from the
            # enumerator) is kept.
            mesh_argv, c.argv_add_flags = _mesh_args(c.mesh, baseline_mesh)
            c.argv_set = {**c.argv_set, **mesh_argv}
            if not planner._axis_size(c.mesh, 'ep') > 1:  # pylint: disable=protected-access
                c.argv_remove_flags = ['--enable-expert-parallel']
        c.expected_mechanism = c.expected_mechanism or (
            'Re-profile and re-run the serving analyzer: kernel-shape findings '
            + (', '.join(c.remeasure) if c.remeasure else 'none') +
            ' must be re-measured on the new mesh.')
        det = _determinism_risk(c.mesh, baseline_mesh)
        if det and not any(_DETERMINISM_MARKER in r for r in c.risks):
            c.risks.append(det)
        if c.workload_dependent:
            c.risks.append('workload-dependent: expected ' +
                           ', '.join(f'{a.get("expected_direction")} at {p}'
                                     for p, a in c.per_point.items()) +
                           '; measure every point before adopting the mesh')

    # (a) coupled pairs.
    final_serving: list[Candidate] = []
    for c in serving:
        pairs = _coupling(c)
        base_ok = all(
            planner._axis_size(baseline_mesh or {}, p.axis) >= p.min_size  # pylint: disable=protected-access
            for p in pairs)
        if not pairs or base_ok:
            final_serving.append(c)
            continue
        partner = next(
            (
                m for m in meshes if m.mesh and all(
                    planner._axis_size(m.mesh, p.axis) >= p.min_size  # pylint: disable=protected-access
                    for p in pairs)),
            None,
        )
        why = '; '.join(p.why for p in pairs)
        if partner is None:
            rejected.append(
                f'{c.cand_id} ({", ".join(c.findings)}): coupled knob ({why}) and no'
                ' mesh candidate provides it (rule a).')
            continue
        combo = dataclasses.replace(
            partner,
            cand_id=f'{partner.cand_id}+{c.cand_id[2:]}'[:40],
            skill='serving-optimizer+sharding',
            findings=sorted(set(partner.findings) | set(c.findings)),
            title=f'{partner.title} + {c.title} (coupled)',
            rationale=f'Coupled pair ({why}). {c.rationale}',
            env_set={
                **partner.env_set,
                **c.env_set
            },
            argv_set={
                **partner.argv_set,
                **c.argv_set
            },
            argv_add_flags=partner.argv_add_flags + c.argv_add_flags,
            additional_config_set={
                **partner.additional_config_set,
                **c.additional_config_set,
            },
            risks=partner.risks + c.risks,
            tool=f'{c.tool} + {partner.tool}',
        )
        notes.append(
            f'{c.cand_id} paired with {partner.cand_id} into {combo.cand_id}'
            f' (rule a: {why}).')
        final_serving.append(combo)

    # (c) a flag set to two values within one candidate.
    out: list[Candidate] = []
    point_labels = [
        str(p.get('label')) for p in (oss_meshes or {}).get('points') or []
        if p.get('label')
    ]
    for c in final_serving + meshes:
        clash = set(
            c.argv_add_flags) & {'--no-' + f[2:]
                                 for f in c.argv_add_flags}
        if clash:
            rejected.append(
                f'{c.cand_id}: sets {sorted(clash)} both ways (rule c).')
            continue
        c.target_points = list(
            target_points) or c.target_points or point_labels
        out.append(c)
    return ComposedPlan(out, rejected, notes, _render(out, rejected, notes))


def _render(cands: Sequence[Candidate], rejected: Sequence[str],
            notes: Sequence[str]) -> str:
    """implementation_plan.md for independent mode."""
    lines = ['# Performance Optimization Implementation Plan', '']
    lines += [
        'Mode: **independent**. Each candidate is tested alone against a'
        ' same-session baseline control. Only winners are combined, and a'
        ' combination is one more measured trial.',
        '',
        '## Summary',
        '',
        '| # | Candidate | Skill | Change | Findings | Re-measure |',
        '|---|---|---|---|---|---|',
    ]
    for i, c in enumerate(cands, 1):
        change = ', '.join(
            [f'`{k}={v}`' for k, v in {
                **c.env_set,
                **c.argv_set
            }.items()] + [f'`{f}`' for f in c.argv_add_flags] +
            [f'`{k}={v}`' for k, v in c.additional_config_set.items()])
        lines.append(
            f'| {i} | `{c.cand_id}` | {c.skill} | {change} |'
            f' {", ".join(c.findings) or "-"} | {", ".join(c.remeasure) or "-"} |'
        )
    if not cands:
        lines.append('| - | none | - | keep the baseline | - | - |')
    per_point_rows = [(i, c, p, a) for i, c in enumerate(cands, 1)
                      for p, a in c.per_point.items()]
    if per_point_rows:
        lines += [
            '',
            '## Per-point assessment',
            '',
            '| # | Candidate | Point | Regime | Expected | Score | Rationale |',
            '|---|---|---|---|---|---|---|',
        ]
        for i, c, p, a in per_point_rows:
            tag = ' (workload-dependent)' if c.workload_dependent else ''
            lines.append(
                f'| {i} | `{c.cand_id}`{tag} | {p} | {a.get("regime", "")} |'
                f' {a.get("expected_direction", "")} | {a.get("score", "")} |'
                f' {a.get("rationale", "")} |')
    lines += ['', '## Candidates', '']
    for i, c in enumerate(cands, 1):
        tag = ' [workload-dependent]' if c.workload_dependent else ''
        lines.append(f'### Candidate {i}: {c.title} (`{c.cand_id}`){tag}')
        if c.mesh:
            lines.append(
                f'- **ICI Mesh**: {json.dumps(c.mesh, sort_keys=True)}')
        if c.env_set:
            lines.append(f'- **Env**: {json.dumps(c.env_set, sort_keys=True)}')
        if c.argv_set or c.argv_add_flags:
            lines.append('- **CLI**: ' +
                         ' '.join([f'{k} {v}' for k, v in c.argv_set.items()] +
                                  c.argv_add_flags))
        if c.additional_config_set:
            lines.append('- **additional_config**: ' +
                         json.dumps(c.additional_config_set, sort_keys=True))
        lines.append(f'- **Rationale**: {c.rationale}')
        if c.target_regime:
            lines.append(f'- **Target regime**: {c.target_regime}')
        for p, a in c.per_point.items():
            lines.append(
                f'- **At {p}**: expected {a.get("expected_direction")}'
                f' ({a.get("regime")}, score {a.get("score")})')
        if c.profiles:
            lines.append('- **Backed by profiles**: ' + ', '.join(c.profiles))
        if c.evidence:
            lines.append('- **Evidence**: ' + '; '.join(c.evidence))
        if c.expected_mechanism:
            lines.append(f'- **Validate**: {c.expected_mechanism}')
        if c.remeasure:
            lines.append('- **Re-measure after the mesh change**: ' +
                         ', '.join(c.remeasure))
        if c.risks:
            lines.append('- **Risks**: ' + '; '.join(c.risks))
        lines.append(f'- {c.provenance_line}')
        lines.append('')
    lines += ['## User Review Required', '']
    lines += [f'* {x}' for x in list(rejected) + list(notes)] or ['* None.']
    lines += [
        '',
        '## Combination trials',
        '',
        'After the A/B results: combine only the winners (for example the best'
        ' serving candidate on the best mesh) as one extra measured trial.'
        ' Kernel-shape findings listed under Re-measure must be re-checked on'
        ' the new mesh before a knob tuned on the baseline mesh is carried over.',
        '',
    ]
    return '\n'.join(lines)
