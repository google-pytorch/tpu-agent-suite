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
"""Deterministic Stage 2/3 planner merging sharding proposals and serving findings.

Implements the 4-stage serving architecture (Diagnose -> Propose ->
Plan -> Execute):
  * `extract_sharding_constraints`: Summarizes mesh- and
  logical-sharding-coupled
    serving findings (D6, D7, D8, D20, D23, D24, D31, D32, D37) and phase
    breakdown
    so the Sharding Proposer co-designs ICI Mesh and Logical Sharding Overrides.
  * `merge_sharding_and_serving_plan`: Applies deterministic rules R1-R5:
    - R1 (Co-propose): Attaches mesh-coupled serving knobs (e.g. D8
      `enable_dp_attention`) directly to sharding candidates as
      `- **Serving Knobs**: {...}`.
    - R2 (Absorb): Folds pure mesh/logical-sharding symptom detectors (D6, D7,
      D20, D23, D24, D32) into the sharding rationale instead of emitting
      duplicate standalone trials.
    - R3 (Invalidate): Flags padding/tile/headroom findings (D1, D8, D9, D31,
      D37) that must be re-checked when TP, EP, or DP/FSDP axes change.
    - R4 (Rank): Produces a unified ranked table across co-designed mesh+knob
      candidates and orthogonal runtime serving knobs.
    - R5 (Gate): Collects quality-gated, memory-risk, unknown-state, and
      mesh-invalidation items under `## User Review Required`.
"""

import dataclasses
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from serving_analysis import findings as findings_lib

# Detectors whose thresholds/shapes depend on TP/EP/DP axis sizes and must be
# re-checked after a mesh change (Rule R3). IDs are the ones the detectors in
# detectors.py / catalog_detectors.py / headroom_detectors.py actually emit.
_AXIS_DEPENDENT_DETECTORS: dict[str, tuple[str, ...]] = {
    'tp': (
        'D7_ar_plus_slice',
        'D7_collective_bound',
        'D8_kv_head_padding',
        'D20_mla_attention_tp',
        'D26_moe_combine_no_local_reduction',
        'D31_gmm_tile_misalignment',
        'D32_excess_collectives_per_layer',
        'D36_hbm_limited_decode_batch',
    ),
    'ep': (
        'D5_moe_permute',
        'D6_expert_imbalance',
        'D7_moe_allgather_fp8',
        'D15_torchtpu_fused_ep_moe',
        'D24_moe_tp_without_ep',
        'D31_gmm_tile_misalignment',
        'D37_moe_weight_precision',
        'D44_moe_combine_not_chunked',
    ),
    'dp': (
        'D2_padding_waste',
        'D3_prefill_decode_interference',
        'D27_max_num_seqs_inflation',
        'D36_hbm_limited_decode_batch',
    ),
    'fsdp': (
        'D7_decode_weight_allgather',
        'D36_hbm_limited_decode_batch',
        'D37_moe_weight_precision',
    ),
    # vLLM pipeline / decode- / prefill-context parallelism.
    'pp': (
        'D3_prefill_decode_interference',
        'D27_max_num_seqs_inflation',
        'D36_hbm_limited_decode_batch',
    ),
    'dcp': (
        'D8_kv_head_padding',
        'D36_hbm_limited_decode_batch',
    ),
    'pcp': ('D7_collective_bound', ),
}

# Mesh axis aliases (vLLM CLI, MaxText yml, candidate mesh dicts).
_AXIS_ALIASES: dict[str, tuple[str, ...]] = {
    'tp': ('tp', 'tensor', 'model', 'ici_tensor_parallelism'),
    'ep': ('ep', 'expert', 'ici_expert_parallelism'),
    'dp': ('dp', 'data', 'replica', 'ici_data_parallelism'),
    'fsdp': ('fsdp', 'ici_fsdp_parallelism'),
    'pp': ('pp', 'pipeline', 'pipeline_parallel_size'),
    'dcp': ('dcp', 'decode_context_parallel_size'),
    'pcp': ('pcp', 'prefill_context_parallel_size'),
}
# Serving knobs that set a mesh axis. R1 only attaches them to a candidate
# whose mesh has the same axis size.
_AXIS_KNOBS: dict[str, str] = {
    'tensor_parallel_size': 'tp',
    'ici_tensor_parallelism': 'tp',
    'data_parallel_size': 'dp',
    'ici_data_parallelism': 'dp',
    'ici_expert_parallelism': 'ep',
    'ici_fsdp_parallelism': 'fsdp',
    'pipeline_parallel_size': 'pp',
    'decode_context_parallel_size': 'dcp',
    'prefill_context_parallel_size': 'pcp',
}

# Candidate headers of a markdown mesh plan:
# `### Candidate 1: ...`, `## Candidate 1`, `### C1: ...`, `### Proposal 1`,
# `### Proposed ICI Mesh Configuration 1`.
_CANDIDATE_HEADER_RE = re.compile(
    r'(?i)^(#{2,4})\s*(?:Candidate\s*(\d+)|Proposal\s*(\d+)|C(\d+)\b'
    r'|Proposed\s+ICI\s+Mesh\s+Configuration\s*(\d+))')
# `- **ICI Mesh**: {...}`, `*   **ICI Mesh:** `{...}``, `ICI Mesh: {...}`.
_ICI_MESH_LINE_RE = re.compile(
    r'(?im)^([ \t]*[-*]?\s*\*{0,2}ICI[\s_]*Mesh\s*:?\s*\*{0,2}\s*:?\s*`?)'
    r'(\{[^\n`]*\})')
_SERVING_KNOBS_LINE_RE = re.compile(
    r'(?im)^[ \t]*[-*]?\s*\*{0,2}Serving[\s_]*Knobs(?::\*{0,2}|\*{0,2}\s*:)'
    r'\s*`?(\{[^\n`]*\})?')


@dataclasses.dataclass
class MergedPlan:
    """Structured output of the deterministic Stage 3 serving + sharding planner."""

    co_proposed_knobs: dict[str, Any]
    absorbed_findings: list[dict[str, Any]]
    orthogonal_findings: list[dict[str, Any]]
    kernel_followups: list[dict[str, Any]]
    diagnostic_findings: list[dict[str, Any]]
    recheck_by_candidate: dict[str, list[str]]
    user_review_items: list[str]
    augmented_sharding_markdown: str
    merged_markdown: str
    # Candidate label -> co-proposed knobs that contradict its mesh (R1).
    knob_conflicts: dict[str,
                         list[str]] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _coerce_finding_dict(
    item: findings_lib.Finding | Mapping[str, Any], ) -> dict[str, Any]:
    """Coerces a finding into a dict."""
    if isinstance(item, findings_lib.Finding):
        return item.to_dict()
    d = dict(item)
    detector_id = str(d.get('detector_id', ''))
    prefix = detector_id.split('_', 1)[0]
    default_lever, default_couples = (
        findings_lib._DEFAULT_DETECTOR_COUPLING.get(  # pylint: disable=protected-access
            prefix, ('', ())))
    if not d.get('lever'):
        if default_lever:
            d['lever'] = default_lever
        elif d.get('risk') == findings_lib.Risk.DIAGNOSTIC.value:
            d['lever'] = 'diagnostic'
        elif d.get('kernel_followup') and not d.get('knob_changes'):
            d['lever'] = 'kernel'
        else:
            d['lever'] = 'serving_knob'
    if 'couples_with' not in d or not d['couples_with']:
        d['couples_with'] = list(default_couples)
    return d


def _case_profiles(payload: Any) -> list[Mapping[str, Any]]:
    """Returns `profiles[*]` of a `serving_analyze_main --case` output, else []."""
    if not isinstance(payload, Mapping):
        return []
    profiles = payload.get('profiles')
    if not isinstance(profiles, list):
        return []
    return [p for p in profiles if isinstance(p, Mapping) and 'result' in p]


def normalize_findings(
    raw_findings: (Sequence[findings_lib.Finding | Mapping[str, Any]]
                   | Mapping[str, Any]),
) -> list[dict[str, Any]]:
    """Normalizes Finding objects or JSON dicts into dicts.

  Accepts a list of findings, a single-profile `summary_dict()` (top-level
  `findings`), or a `--case` output (`profiles[*].result.findings`). Findings
  from a case are tagged with `profile_role` / `profile_key` so that
  prefill-only and decode-only diagnoses stay distinguishable.

  Args:
    raw_findings: Findings in any of the forms above.

  Returns:
    A list of finding dicts with `lever` and `couples_with` filled in.
  """
    profiles = _case_profiles(raw_findings)
    if profiles:
        out = []
        for p in profiles:
            for f in (p.get('result') or {}).get('findings', []):
                d = _coerce_finding_dict(f)
                d.setdefault('profile_role', str(p.get('role', '')))
                d.setdefault('profile_key', str(p.get('key', '')))
                out.append(d)
        return out
    if isinstance(raw_findings, Mapping):
        items = raw_findings.get('findings', [])
    else:
        items = raw_findings
    return [_coerce_finding_dict(f) for f in items]


def _phase_stats(payload: Any) -> list[dict[str, Any]]:
    """Phase stats of a summary_dict, or of every profile of a case output."""
    profiles = _case_profiles(payload)
    if profiles:
        out = []
        for p in profiles:
            for row in (p.get('result') or {}).get('phase_stats', []):
                out.append({
                    **row,
                    'profile_role': str(p.get('role', '')),
                    'profile_key': str(p.get('key', '')),
                })
        return out
    if isinstance(payload, Mapping):
        return list(payload.get('phase_stats', []))
    return []


def extract_sharding_constraints(
    analysis_or_findings: (Sequence[findings_lib.Finding | Mapping[str, Any]]
                           | Mapping[str, Any]),
) -> dict[str, Any]:
    """Extracts Stage 2 sharding & logical-sharding constraints from serving findings.

  Args:
    analysis_or_findings: The `summary_dict()` JSON emitted by
      `serving_analyze_main`, its `--case` output (one result per profile), or
      a sequence of `Finding` objects/dicts.

  Returns:
    A JSON-serializable dictionary containing `phase_stats`, `mesh_findings`,
    `logical_sharding_findings`, `co_proposed_serving_knobs`, and a concise
    `prompt_guidance` list for the Sharding Proposer.
  """
    phase_stats = _phase_stats(analysis_or_findings)
    findings = normalize_findings(analysis_or_findings)

    mesh_findings: list[dict[str, Any]] = []
    logical_sharding_findings: list[dict[str, Any]] = []
    co_proposed_serving_knobs: dict[str, Any] = {}
    guidance: list[str] = []

    for f in findings:
        couples = set(f.get('couples_with') or [])
        lever = f.get('lever', '')
        det_id = str(f.get('detector_id', ''))
        if lever == 'mesh' or 'mesh' in couples:
            mesh_findings.append(f)
        if lever == 'logical_sharding' or 'logical_sharding' in couples:
            logical_sharding_findings.append(f)

        if 'mesh' in couples:
            for kc in f.get('knob_changes') or []:
                if kc.get('already_set'):
                    continue
                co_proposed_serving_knobs[kc['name']] = _coerce_scalar(
                    kc['suggested_value'])

        if det_id.startswith('D8'):
            guidance.append(
                'D8 (KV-head padding / GQA replication): Co-propose'
                ' `enable_dp_attention` (or reduce attention TP / shard Q on'
                ' `model_q` while keeping KV unsharded across `model` when'
                ' `num_kv_heads < tp`).')
        elif det_id.startswith('D23'):
            guidance.append(
                'D23 (Decode FSDP weight all-gather / stage imbalance): Move decode'
                ' weight sharding off `fsdp`/`data` onto `model`/`expert` so weights'
                ' stay resident in HBM instead of all-gathering every decode step.'
            )
        elif det_id.startswith('D24') or det_id.startswith('D6'):
            guidance.append(
                f'{det_id} (MoE routing / TP-vs-EP): Prefer Expert Parallelism (`ep`'
                ' / `expert`) over Tensor Parallelism on MoE layers when tile sizes'
                ' remain >= 128.')
        elif det_id.startswith('D31'):
            guidance.append(
                'D31 (GMM tile alignment): Keep per-device expert intermediate size'
                ' `(moe_intermediate_size / tp)` a multiple of 128.')
        elif det_id.startswith('D32'):
            guidance.append(
                'D32 (Excess collectives per layer): Align activation and weight'
                ' logical sharding specs (`activation_btd`, `ffw_hidden_btf`,'
                ' `query_btnh`, `keyvalue_bskh`) with the mesh to eliminate'
                ' per-layer reshard collectives.')

    return {
        'phase_stats': phase_stats,
        'mesh_findings': mesh_findings,
        'logical_sharding_findings': logical_sharding_findings,
        'co_proposed_serving_knobs': co_proposed_serving_knobs,
        'prompt_guidance': guidance,
    }


def _coerce_scalar(val: Any) -> Any:
    """Coerces a string to a bool, int, or float if possible."""
    if not isinstance(val, str):
        return val
    s = val.strip()
    if s.lower() in ('true', 'yes', 'on'):
        return True
    if s.lower() in ('false', 'no', 'off'):
        return False
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def _parse_simple_mesh(mesh_text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in re.finditer(r"['\"]?(\w+)['\"]?\s*[:=]\s*(\d+)", mesh_text):
        out[m.group(1).lower()] = int(m.group(2))
    return out


def _axis_size(mesh: Mapping[str, int], axis: str) -> int:
    """Size of an axis class (a key of `_AXIS_ALIASES`) in a mesh; 1 if absent."""
    norm = {str(k).lower(): int(v) for k, v in mesh.items()}
    return next((norm[a] for a in _AXIS_ALIASES[axis] if a in norm), 1)


def _changed_axis_classes(
    candidate_mesh: Mapping[str, int],
    baseline_mesh: Mapping[str, int] | None,
) -> set[str]:
    """Identifies which axis classes (tp, ep, dp, fsdp, pp, dcp, pcp) changed."""
    if not baseline_mesh:
        return set(_AXIS_ALIASES)
    return {
        axis
        for axis in _AXIS_ALIASES
        if _axis_size(baseline_mesh, axis) != _axis_size(candidate_mesh, axis)
    }


def compute_recheck_findings(
    candidate_mesh: Mapping[str, int],
    baseline_mesh: Mapping[str, int] | None,
    findings: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Rule R3: Returns detector IDs that must be re-checked after a mesh change."""
    changed_classes = _changed_axis_classes(candidate_mesh, baseline_mesh)
    active_detectors = {str(f.get('detector_id', '')) for f in findings}
    recheck: list[str] = []
    for cls_name in sorted(changed_classes):
        for det_id in _AXIS_DEPENDENT_DETECTORS.get(cls_name, ()):
            if det_id in active_detectors and det_id not in recheck:
                recheck.append(det_id)
    return recheck


def knob_mesh_conflict(name: str, value: Any,
                       mesh: Mapping[str, int]) -> str | None:
    """Returns why a serving knob contradicts a candidate mesh, or None.

  Examples: `--enable-expert-parallel` on a mesh without EP>1, or D8's
  `--tensor-parallel-size=1` on a TP4 candidate.

  Args:
    name: Literal knob name (`--enable-expert-parallel`, `enable_dp_attention`).
    value: Suggested value.
    mesh: Candidate mesh (any of the axis aliases in `_AXIS_ALIASES`).
  """
    key = name.lstrip('-').replace('-', '_').lower().rsplit('.', 1)[-1]
    val = _coerce_scalar(value)
    if key == 'enable_expert_parallel' and val is True:
        ep = _axis_size(mesh, 'ep')
        if ep <= 1:
            return f'`{name}` needs EP>1 but the candidate mesh has ep={ep}'
    elif key == 'enable_dp_attention' and val is True:
        if _axis_size(mesh, 'tp') <= 1:
            return f'`{name}` only applies with TP>1 (candidate mesh has tp=1)'
    elif key in _AXIS_KNOBS and isinstance(val, int):
        axis = _AXIS_KNOBS[key]
        size = _axis_size(mesh, axis)
        if size != val:
            return f'`{name}={val}` but the candidate mesh has {axis}={size}'
    return None


def _candidate_blocks(lines: Sequence[str]) -> list[tuple[str, int, int]]:
    """Splits sharding markdown into (label, first_line, end_line) blocks.

  A block starts at a candidate header and ends at the next header of the same
  or a higher level. Without candidate headers, every ICI Mesh line is its own
  block.

  Args:
    lines: Markdown lines.
  """
    heads: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        m = _CANDIDATE_HEADER_RE.match(line)
        if m:
            num = next(g for g in m.groups()[1:] if g)
            heads.append((i, len(m.group(1)), f'Candidate {int(num)}'))
    if not heads:
        meshes = [
            i for i, line in enumerate(lines) if _ICI_MESH_LINE_RE.match(line)
        ]
        return [(f'Candidate {n}', i, i + 1) for n, i in enumerate(meshes, 1)]
    blocks = []
    for start, level, label in heads:
        end = len(lines)
        for j in range(start + 1, len(lines)):
            m = re.match(r'^(#{1,6})\s', lines[j])
            if m and len(m.group(1)) <= level:
                end = j
                break
        blocks.append((label, start, end))
    return blocks


def _apply_serving_knobs_to_candidates(
    sharding_markdown: str,
    co_proposed_knobs: Mapping[str, Any],
) -> tuple[str, dict[str, dict[str, int]], dict[str, list[str]]]:
    """Rule R1: attaches mesh-compatible serving knobs to each candidate.

  A knob is written into a candidate as `- **Serving Knobs**: {...}` only when
  it is compatible with that candidate's mesh (`knob_mesh_conflict`). Knobs
  that contradict the mesh are listed under `- **Serving Knob Conflicts**:`
  instead. Candidates that already carry a Serving Knobs line keep it; its
  knobs are only checked for conflicts.

  Args:
    sharding_markdown: Sharding proposals markdown.
    co_proposed_knobs: Mesh-coupled serving knobs from the serving findings.

  Returns:
    (augmented markdown, candidate label -> mesh, candidate label -> conflicts)
  """
    lines = sharding_markdown.splitlines()
    meshes: dict[str, dict[str, int]] = {}
    conflicts: dict[str, list[str]] = {}
    inserts: dict[int, list[str]] = {}
    for label, start, end in _candidate_blocks(lines):
        mesh_idx = next(
            (i
             for i in range(start, end) if _ICI_MESH_LINE_RE.match(lines[i])),
            None,
        )
        if mesh_idx is None:
            continue
        mesh_line = lines[mesh_idx]
        mesh_match = _ICI_MESH_LINE_RE.match(mesh_line)
        assert mesh_match is not None
        mesh = _parse_simple_mesh(mesh_match.group(2))
        meshes[label] = mesh
        existing: dict[str, Any] | None = None
        for i in range(start, end):
            km = _SERVING_KNOBS_LINE_RE.match(lines[i])
            if km:
                try:
                    existing = json.loads(km.group(1)) if km.group(1) else {}
                except json.JSONDecodeError:
                    existing = {}
                break
        knobs = existing if existing is not None else co_proposed_knobs
        compatible: dict[str, Any] = {}
        bad: list[str] = []
        for name, value in knobs.items():
            why = knob_mesh_conflict(name, value, mesh)
            if why:
                bad.append(why)
            else:
                compatible[name] = value
        indent_match = re.match(r'^([ \t]*)', mesh_line)
        indent = indent_match.group(1) if indent_match else ''
        bullet = '- ' if mesh_line.lstrip().startswith(('-', '*')) else ''
        new_lines = []
        if existing is None and compatible:
            new_lines.append(f'{indent}{bullet}**Serving Knobs**:'
                             f' {json.dumps(compatible, sort_keys=True)}')
        if bad:
            conflicts[label] = bad
            new_lines.append(f'{indent}{bullet}**Serving Knob Conflicts**: ' +
                             '; '.join(bad))
        if new_lines:
            inserts[mesh_idx] = new_lines
    out: list[str] = []
    for i, line in enumerate(lines):
        out.append(line)
        out.extend(inserts.get(i, []))
    return '\n'.join(out), meshes, conflicts


def merge_sharding_and_serving_plan(
    sharding_markdown: str,
    analysis_or_findings: (Sequence[findings_lib.Finding | Mapping[str, Any]]
                           | Mapping[str, Any]),
    baseline_mesh: Mapping[str, int] | None = None,
) -> MergedPlan:
    """Deterministically merges sharding proposals and serving analyzer findings."""
    findings = normalize_findings(analysis_or_findings)
    constraints = extract_sharding_constraints(analysis_or_findings)
    co_proposed_knobs = dict(constraints['co_proposed_serving_knobs'])

    absorbed: list[dict[str, Any]] = []
    orthogonal: list[dict[str, Any]] = []
    kernel_followups: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    user_review: list[str] = []

    for f in sorted(findings,
                    key=lambda x: float(x.get('score', 0.0)),
                    reverse=True):
        lever = f.get('lever', 'serving_knob')
        couples = set(f.get('couples_with') or [])
        knob_changes = f.get('knob_changes') or []
        risk = f.get('risk', '')
        det_id = str(f.get('detector_id', ''))
        title = str(f.get('title', det_id))

        # Rule R5: Gate quality-gated, memory-risk, and unknown-state findings.
        if risk == findings_lib.Risk.QUALITY_GATED.value:
            user_review.append(
                f'[Quality Gated] `{det_id}` ({title}):'
                f' {f.get("validation") or "Requires accuracy evaluation before production rollout."}'
            )
        elif risk == findings_lib.Risk.MEMORY.value:
            user_review.append(
                f'[Memory Risk] `{det_id}` ({title}): May increase HBM usage or'
                ' reduce KV cache capacity.')
        if any(kc.get('already_set') is None for kc in knob_changes):
            unknown_names = ', '.join(f"`{kc['name']}`" for kc in knob_changes
                                      if kc.get('already_set') is None)
            user_review.append(
                f'[Unknown Current State] `{det_id}`: Current value of'
                f' {unknown_names} was not present in launch command; verify before'
                ' applying.')

        # Rule R2: Absorb pure mesh/logical-sharding findings or mesh-coupled
        # findings whose knobs were co-proposed into the candidate block.
        if lever in ('mesh', 'logical_sharding') and not knob_changes:
            absorbed.append(f)
            continue
        if 'mesh' in couples and knob_changes:
            absorbed.append(f)
            continue
        if lever == 'diagnostic' or risk == findings_lib.Risk.DIAGNOSTIC.value:
            diagnostics.append(f)
            continue
        if lever == 'kernel' or (f.get('kernel_followup')
                                 and not knob_changes):
            kernel_followups.append(f)
            continue
        orthogonal.append(f)

    # Rule R1: Attach mesh-compatible serving knobs to sharding candidates.
    if sharding_markdown.strip():
        augmented_sharding, cand_meshes, knob_conflicts = (
            _apply_serving_knobs_to_candidates(sharding_markdown,
                                               co_proposed_knobs))
    else:
        augmented_sharding, cand_meshes, knob_conflicts = sharding_markdown, {}, {}
    for cand_label, reasons in knob_conflicts.items():
        user_review.append(f'[Knob Conflict] {cand_label}: ' +
                           '; '.join(reasons))

    # Rule R3: Compute per-candidate invalidation / recheck list.
    recheck_by_candidate: dict[str, list[str]] = {}
    for cand_label, cand_mesh in cand_meshes.items():
        recheck = compute_recheck_findings(cand_mesh, baseline_mesh, findings)
        if recheck:
            recheck_by_candidate[cand_label] = recheck
    if recheck_by_candidate:
        for cand_label, det_ids in recheck_by_candidate.items():
            user_review.append(
                f'[Mesh Recheck] {cand_label} modifies TP/EP/DP; re-verify ' +
                ', '.join(f'`{d}`'
                          for d in det_ids) + ' after applying the new mesh.')

    # Rule R4 & R5: Render unified implementation_plan.md.
    lines = ['# Performance Optimization Implementation Plan', '']
    lines.append('## Summary')
    lines.append('')
    lines.append(
        '| Priority | Category | Optimization | Phase | Est. Gain (≤) | Risk |'
    )
    lines.append('|---|---|---|---|---|---|')
    rank_idx = 1
    if augmented_sharding.strip():
        co_desc = (f' + Serving Knobs `{json.dumps(co_proposed_knobs)}`'
                   if co_proposed_knobs else '')
        lines.append(
            f'| {rank_idx} | Co-Designed Mesh & Sharding | ICI Mesh + Logical'
            f' Sharding Proposals{co_desc} | all | see candidates | perf_only |'
        )
        rank_idx += 1
    for f in orthogonal:
        pct = 100.0 * float(f.get('est_step_fraction_saved', 0.0))
        lines.append(
            f'| {rank_idx} | Serving Knob (`{f["detector_id"]}`) | {f["title"]} |'
            f' {f.get("phase", "all")} | {pct:.1f}% |'
            f' {f.get("risk", "perf_only")} |')
        rank_idx += 1
    lines.append('')

    lines.append('## User Review Required')
    lines.append('')
    if user_review:
        for item in user_review:
            lines.append(f'* {item}')
    else:
        lines.append(
            '* None — all proposed changes are perf-only with known state.')
    lines.append('')

    if augmented_sharding.strip():
        lines.append('## Sharding Proposals')
        lines.append('')
        if absorbed:
            lines.append(
                '### Absorbed Serving Bottlenecks (Addressed by Sharding)')
            for f in absorbed:
                lines.append(
                    f'* **`{f["detector_id"]}` ({f["title"]})**: {f.get("summary", "")}'
                )
            lines.append('')
        lines.append(augmented_sharding.strip())
        lines.append('')

    lines.append('## Serving Optimizations')
    lines.append('')
    if orthogonal:
        for i, f in enumerate(orthogonal, 1):
            pct = 100.0 * float(f.get('est_step_fraction_saved', 0.0))
            role = f' [{f["profile_role"]}]' if f.get('profile_role') else ''
            lines.append(
                f'### {i}. {f["title"]} (`{f["detector_id"]}`, ≤{pct:.1f}%){role}'
            )
            lines.append('')
            lines.append(str(f.get('summary', '')))
            lines.append('')
            for kc in f.get('knob_changes') or []:
                lines.append(
                    f'* **Change**: `{kc["name"]}={kc["suggested_value"]}`'
                    f' ({kc["kind"]})')
            if f.get('validation'):
                lines.append(f'* **Validate**: {f["validation"]}')
            lines.append('')
    else:
        lines.append(
            'No standalone serving knob changes outside the co-designed sharding'
            ' candidates.')
        lines.append('')

    if kernel_followups:
        lines.append('### Follow-ups (Kernel / Compiler)')
        for f in kernel_followups:
            lines.append(f'* **`{f["detector_id"]}` ({f["title"]})**:'
                         f' {f.get("kernel_followup")}')
        lines.append('')

    return MergedPlan(
        co_proposed_knobs=co_proposed_knobs,
        absorbed_findings=absorbed,
        orthogonal_findings=orthogonal,
        kernel_followups=kernel_followups,
        diagnostic_findings=diagnostics,
        recheck_by_candidate=recheck_by_candidate,
        user_review_items=user_review,
        augmented_sharding_markdown=augmented_sharding,
        merged_markdown='\n'.join(lines),
        knob_conflicts=knob_conflicts,
    )
