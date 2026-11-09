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
"""Structured findings emitted by serving detectors.

A `Finding` is the unit of output: a deterministic, evidence-backed statement
that a known optimization applies to the analyzed workload. The LLM layer
(skill / coordinator) only ranks and explains findings; it never invents them.
"""

import dataclasses
import enum
import json
from typing import Any


class Risk(enum.Enum):
    """Risk class of applying the recommended change."""

    PERF_ONLY = 'perf_only'  # Numerics unchanged.
    QUALITY_GATED = 'quality_gated'  # Needs an accuracy / eval check.
    MEMORY = 'memory'  # May raise HBM usage or reduce KV capacity.
    LATENCY_TRADEOFF = 'latency_tradeoff'  # Throughput vs TTFT/TPOT trade-off.
    DIAGNOSTIC = 'diagnostic'  # Re-profile / measure; not a production change.


class Confidence(enum.Enum):
    HIGH = 'high'  # Signature is unambiguous and knob state known.
    MEDIUM = 'medium'  # Signature clear, knob state or magnitude uncertain.
    LOW = 'low'  # Heuristic; worth a quick experiment.


@dataclasses.dataclass
class KnobChange:
    """A concrete configuration change."""

    knob: str  # Registry key, e.g. 'tpu_inference.env.MOE_APPROX_TOPK'.
    kind: str  # 'env' | 'cli' | 'additional_config' | 'maxtext_yml' | 'xla_flag'
    name: str  # Literal name the user sets, e.g. 'MOE_APPROX_TOPK'.
    suggested_value: str
    current_value: str | None = None  # None => unknown (no serving config).
    # Out-of-tree patch the image must carry for this knob to exist.
    requires_patch: str = ''

    @property
    def already_set(self) -> bool | None:
        if self.current_value is None:
            return None
        return _normalize(self.current_value) == _normalize(
            self.suggested_value)


_TRUE = frozenset({'1', 'true', 'yes', 'on'})
_FALSE = frozenset({'0', 'false', 'no', 'off', ''})


def _normalize(v: str) -> str:
    s = str(v).strip().lower()
    if s in _TRUE:
        return 'true'
    if s in _FALSE:
        return 'false'
    return s


@dataclasses.dataclass
class Evidence:
    """A single quantitative observation backing a finding."""

    metric: str
    value: float | str
    unit: str = ''
    detail: str = ''
    xprof_link: str | None = None


# Default (lever, couples_with) metadata for detectors that couple with ICI mesh
# or model-level logical sharding.
_DEFAULT_DETECTOR_COUPLING: dict[str, tuple[str, tuple[str, ...]]] = {
    'D6': ('mesh', ('mesh', )),
    'D7': ('mesh', ('mesh', )),
    'D8': ('serving_knob', ('mesh', 'logical_sharding')),
    'D20': ('mesh', ('mesh', )),
    'D23': ('mesh', ('mesh', 'logical_sharding')),
    'D24': ('mesh', ('mesh', )),
    'D31': ('serving_knob', ('mesh', )),
    'D32': ('logical_sharding', ('mesh', 'logical_sharding')),
    'D36': ('serving_knob', ('mesh', )),
    'D37': ('serving_knob', ('mesh', )),
}


@dataclasses.dataclass
class Finding:
    """An actionable serving optimization with its supporting evidence."""

    detector_id: str  # e.g. 'D4_moe_topk_hotspot'
    title: str
    phase: str  # 'decode' | 'prefill' | 'mixed' | 'all'
    summary: str
    evidence: list[Evidence]
    knob_changes: list[KnobChange]
    risk: Risk
    confidence: Confidence
    # Upper bound on step-time saving as a fraction of the phase step time.
    est_step_fraction_saved: float = 0.0
    validation: str = ''
    kernel_followup: str | None = None  # Filled when no knob exists.
    source_refs: list[str] = dataclasses.field(default_factory=list)
    # Optimization lever class: 'mesh' | 'logical_sharding' | 'serving_knob' |
    # 'xla_flag' | 'kernel' | 'model' | 'diagnostic'.
    lever: str = ''
    # Cross-domain coupling tags, e.g. ['mesh', 'logical_sharding'].
    couples_with: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        prefix = self.detector_id.split('_', 1)[0]
        default_lever, default_couples = _DEFAULT_DETECTOR_COUPLING.get(
            prefix, ('', ()))
        if not self.lever:
            if default_lever:
                self.lever = default_lever
            elif self.risk == Risk.DIAGNOSTIC:
                self.lever = 'diagnostic'
            elif self.kernel_followup and not self.knob_changes:
                self.lever = 'kernel'
            elif self.knob_changes and all(k.kind == 'xla_flag'
                                           for k in self.knob_changes):
                self.lever = 'xla_flag'
            else:
                self.lever = 'serving_knob'
        if not self.couples_with and default_couples:
            self.couples_with = list(default_couples)

    @property
    def score(self) -> float:
        w = {Confidence.HIGH: 1.0, Confidence.MEDIUM: 0.6, Confidence.LOW: 0.3}
        already = (all(k.already_set for k in self.knob_changes) if
                   (self.knob_changes) else False)
        return 0.0 if already else self.est_step_fraction_saved * w[
            self.confidence]

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d['risk'] = self.risk.value
        d['confidence'] = self.confidence.value
        d['score'] = round(self.score, 4)
        for kc, obj in zip(d['knob_changes'], self.knob_changes):
            kc['already_set'] = obj.already_set
        return d


def findings_to_json(findings: list[Finding]) -> str:
    return json.dumps([f.to_dict() for f in findings], indent=2, default=str)


def _fmt_value(v: float | str) -> str:
    if isinstance(v, float):
        return f'{v:.3g}'
    return str(v)


def findings_to_markdown(findings: list[Finding]) -> str:
    """Renders findings as the `Serving Optimizations` plan section."""
    if not findings:
        return '## Serving Optimizations\n\nNo serving findings triggered.\n'
    ranked = sorted(findings, key=lambda f: f.score, reverse=True)
    lines = ['## Serving Optimizations', '']
    lines.append('| # | Finding | Phase | Est. gain (≤) | Risk | Confidence |')
    lines.append('|---|---|---|---|---|---|')
    for i, f in enumerate(ranked, 1):
        lines.append(
            f'| {i} | {f.title} | {f.phase} |'
            f' {100 * f.est_step_fraction_saved:.1f}% | {f.risk.value} |'
            f' {f.confidence.value} |')
    lines.append('')
    for i, f in enumerate(ranked, 1):
        lines.append(f'### {i}. {f.title} (`{f.detector_id}`)')
        lines.append('')
        lines.append(f.summary)
        lines.append('')
        lines.append('**Evidence**')
        lines.append('')
        for e in f.evidence:
            link = f' ({e.xprof_link})' if e.xprof_link else ''
            detail = f' — {e.detail}' if e.detail else ''
            lines.append(
                f'* {e.metric}: {_fmt_value(e.value)}{e.unit}{detail}{link}')
        lines.append('')
        if f.knob_changes:
            lines.append('**Change**')
            lines.append('')
            for k in f.knob_changes:
                state = {
                    None: 'current value unknown',
                    True: 'ALREADY SET',
                    False: f'currently `{k.current_value}`',
                }[k.already_set]
                patch = f'; requires {k.requires_patch}' if k.requires_patch else ''
                lines.append(
                    f'* `{k.name}={k.suggested_value}` ({k.kind}; {state}{patch})'
                )
            lines.append('')
        if f.kernel_followup:
            lines.append(f'**Kernel/compiler follow-up:** {f.kernel_followup}')
            lines.append('')
        if f.validation:
            lines.append(f'**Validate:** {f.validation}')
            lines.append('')
        if f.source_refs:
            lines.append('**Refs:** ' + '; '.join(f.source_refs))
            lines.append('')
    return '\n'.join(lines)
