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
"""Deterministic mechanism verification and noise-aware A/B verdicts.

Separates two post-run questions:
  1. Mechanism (`check_mechanism`): did the targeted trace bottleneck (a
     detector id such as `D8`, or a category share) shrink on the candidate
     trace?
  2. Verdict (`decide_verdict`): given baseline vs candidate serving metrics,
     replicate noise (`noise_rel`), and mechanism evidence, is the arm `KEEP`,
     `DISCARD`, or `INCONCLUSIVE`?
"""

import ast
import dataclasses
import functools
import math
import pathlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

KEEP = "KEEP"
DISCARD = "DISCARD"
INCONCLUSIVE = "INCONCLUSIVE"

_MATERIAL_DROP = 0.20
_MATERIAL_ABS_DROP_PCT = 2.0
_DEFAULT_THROUGHPUT_GAIN = 0.02
_DEFAULT_LATENCY_DROP = 0.03
_DEFAULT_LATENCY_REGRESSION = 0.05
_DEFAULT_THROUGHPUT_REGRESSION = 0.02

_DETECTOR_ID_RE = re.compile(r"^(D\d+)_([a-z0-9_]+)$", re.IGNORECASE)
_DETECTOR_NUM_RE = re.compile(r"^(D\d+)$", re.IGNORECASE)


@functools.lru_cache(maxsize=4)
def _serving_detector_catalog(
    serving_dir: str | None = None, ) -> tuple[str, ...]:
    """Discovers canonical `D<N>_<slug>` IDs from sibling `serving_analysis`."""
    base = (pathlib.Path(serving_dir) if serving_dir is not None else
            pathlib.Path(__file__).resolve().parents[1] / "serving_analysis")
    if not base.is_dir():
        return ()
    found: dict[str, None] = {}
    for filename in (
            "detectors.py",
            "catalog_detectors.py",
            "headroom_detectors.py",
    ):
        path = base / filename
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"),
                             filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                m = _DETECTOR_ID_RE.match(node.value)
                if m:
                    canon = f"{m.group(1).upper()}_{m.group(2).lower()}"
                    found.setdefault(canon, None)
    return tuple(found)


def resolve_detector_id(
    name: str,
    catalog: Sequence[str] | None = None,
) -> str:
    """Resolves a detector ID, short `D<N>`, or semantic slug to a catalog ID.

    Every detector in `serving_analysis` has a unique `<slug>` after its
    `D<N>_` prefix, so matching by slug keeps mechanism checks stable even when
    a run recorded the detector under a different number, and lets callers
    pass a slug (`kv_head_padding`) or an unambiguous short ID (`D8`). An
    ambiguous short ID is returned unchanged rather than guessed.
    """
    raw = str(name or "").strip()
    if not raw:
        return ""
    entries = (_serving_detector_catalog()
               if catalog is None else tuple(catalog))
    if not entries:
        return raw

    exact_lower: dict[str, str] = {}
    by_slug: dict[str, str] = {}
    by_num: dict[str, list[str]] = {}
    for entry in entries:
        exact_lower.setdefault(entry.lower(), entry)
        m = _DETECTOR_ID_RE.match(entry)
        if m:
            num, slug = m.group(1).upper(), m.group(2).lower()
            by_slug.setdefault(slug, entry)
            bucket = by_num.setdefault(num, [])
            if entry not in bucket:
                bucket.append(entry)

    m_id = _DETECTOR_ID_RE.match(raw)
    if m_id:
        slug = m_id.group(2).lower()
        if slug in by_slug:
            return by_slug[slug]
        return f"{m_id.group(1).upper()}_{slug}"

    m_num = _DETECTOR_NUM_RE.match(raw)
    if m_num:
        matches = by_num.get(m_num.group(1).upper(), [])
        if len(matches) == 1:
            return matches[0]
        return m_num.group(1).upper()

    norm_slug = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    if norm_slug in by_slug:
        return by_slug[norm_slug]
    return exact_lower.get(raw.lower(), raw)


@dataclasses.dataclass(frozen=True)
class FindingChange:
    """Before/after comparison for one detector or category finding."""

    detector: str
    before_pct: float
    after_pct: float
    before_evidence: Mapping[str,
                             Any] = dataclasses.field(default_factory=dict)
    after_evidence: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def abs_change_pct(self) -> float:
        return self.after_pct - self.before_pct

    @property
    def rel_change(self) -> float | None:
        if self.before_pct <= 0:
            return None
        return (self.after_pct - self.before_pct) / self.before_pct

    def to_dict(self) -> dict[str, Any]:
        rel = self.rel_change
        return {
            "detector": self.detector,
            "before_pct": round(self.before_pct, 2),
            "after_pct": round(self.after_pct, 2),
            "abs_change_pct": round(self.abs_change_pct, 2),
            "rel_change": round(rel, 4) if rel is not None else None,
            "before_evidence": dict(self.before_evidence),
            "after_evidence": dict(self.after_evidence),
        }


@dataclasses.dataclass(frozen=True)
class MechanismVerdict:
    """Whether the targeted bottleneck actually moved in the candidate trace."""

    proposal_id: str
    confirmed: bool
    summary: str
    targeted: tuple[FindingChange, ...]
    collateral: tuple[FindingChange, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "confirmed": self.confirmed,
            "summary": self.summary,
            "targeted": [t.to_dict() for t in self.targeted],
            "collateral": [c.to_dict() for c in self.collateral],
        }


def _finding_pct(item: Mapping[str, Any] | None) -> float:
    if item is None:
        return 0.0
    for key in ("affected_pct", "share_of_step_pct", "share_pct"):
        val = item.get(key)
        if val is not None:
            return float(val)
    frac = item.get("est_step_fraction_saved")
    if frac is not None:
        return float(frac) * 100.0
    return 0.0


def _finding_evidence(item: Mapping[str, Any] | None) -> dict[str, Any]:
    if item is None:
        return {}
    raw = item.get("evidence")
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return {"items": list(raw)} if raw else {}
    if raw is not None:
        return {"value": raw}
    return {}


def _finding_raw_detector(item: Mapping[str, Any]) -> str:
    return str(
        item.get("detector_id") or item.get("detector") or item.get("category")
        or "")


def _merged_catalog(
    before_findings: Sequence[Mapping[str, Any]],
    after_findings: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """IDs to resolve against: the candidate run's, the catalog's, the baseline's.

    The first ID seen for a slug wins, so a baseline that recorded a detector
    under another number maps onto the number the candidate run reports.
    """

    def ids(findings: Sequence[Mapping[str, Any]]) -> list[str]:
        out = []
        for item in findings:
            m = _DETECTOR_ID_RE.match(_finding_raw_detector(item).strip())
            if m:
                out.append(f"{m.group(1).upper()}_{m.group(2).lower()}")
        return out

    merged: dict[str, None] = {}
    for cid in (*ids(after_findings), *_serving_detector_catalog(),
                *ids(before_findings)):
        merged.setdefault(cid, None)
    return tuple(merged)


def _findings_by_detector(
    findings: Sequence[Mapping[str, Any]],
    catalog: Sequence[str] | None = None,
) -> dict[str, Mapping[str, Any]]:
    by_det: dict[str, Mapping[str, Any]] = {}
    for item in findings:
        det = resolve_detector_id(_finding_raw_detector(item), catalog=catalog)
        if not det:
            continue
        cur = _finding_pct(item)
        prev = by_det.get(det)
        prev_pct = _finding_pct(prev) if prev is not None else -1.0
        if cur > prev_pct:
            by_det[det] = item
    return by_det


def _materially_reduced(change: FindingChange) -> bool:
    if change.before_pct <= 0:
        return False
    if change.after_pct == 0.0:
        return True
    rel = change.rel_change
    if rel is not None and rel <= -_MATERIAL_DROP:
        return True
    if change.abs_change_pct <= -_MATERIAL_ABS_DROP_PCT:
        return True
    return False


def check_mechanism(
    target_detectors: Sequence[str],
    before_findings: Sequence[Mapping[str, Any]],
    after_findings: Sequence[Mapping[str, Any]],
    proposal_id: str = "candidate",
) -> MechanismVerdict:
    """Checks whether `target_detectors` shrank from `before` to `after`."""
    catalog = _merged_catalog(before_findings, after_findings)
    before_map = _findings_by_detector(before_findings, catalog=catalog)
    after_map = _findings_by_detector(after_findings, catalog=catalog)
    resolved_targets = [
        resolve_detector_id(det, catalog=catalog) for det in target_detectors
    ]
    targets = tuple(dict.fromkeys(t for t in resolved_targets if t))

    targeted: list[FindingChange] = []
    for det in targets:
        b = before_map.get(det)
        a = after_map.get(det)
        b_pct = _finding_pct(b)
        a_pct = _finding_pct(a)
        targeted.append(
            FindingChange(
                detector=det,
                before_pct=b_pct,
                after_pct=a_pct,
                before_evidence=_finding_evidence(b),
                after_evidence=_finding_evidence(a),
            ))

    collateral: list[FindingChange] = []
    for det, a in after_map.items():
        if det in targets:
            continue
        b = before_map.get(det)
        b_pct = _finding_pct(b)
        a_pct = _finding_pct(a)
        fc = FindingChange(detector=det, before_pct=b_pct, after_pct=a_pct)
        if fc.abs_change_pct >= _MATERIAL_ABS_DROP_PCT and (
                fc.rel_change is None or fc.rel_change >= _MATERIAL_DROP):
            collateral.append(fc)
    collateral.sort(key=lambda c: c.abs_change_pct, reverse=True)

    present_before = [t for t in targeted if t.before_pct > 0]
    if not targets:
        return MechanismVerdict(
            proposal_id=proposal_id,
            confirmed=False,
            summary="No target detector specified; mechanism unchecked.",
            targeted=tuple(targeted),
            collateral=tuple(collateral),
        )
    if not present_before:
        return MechanismVerdict(
            proposal_id=proposal_id,
            confirmed=False,
            summary=(
                f"Target detector(s) {', '.join(targets)} did not fire on the "
                "baseline trace (0.0% affected)."),
            targeted=tuple(targeted),
            collateral=tuple(collateral),
        )
    reduced = [t for t in present_before if _materially_reduced(t)]
    parts = [
        f"{t.detector}: {t.before_pct:.1f}% -> {t.after_pct:.1f}%"
        for t in targeted
    ]
    confirmed = bool(reduced)
    prefix = "Mechanism confirmed" if confirmed else "Mechanism did not fire"
    return MechanismVerdict(
        proposal_id=proposal_id,
        confirmed=confirmed,
        summary=f"{prefix} ({'; '.join(parts)}).",
        targeted=tuple(targeted),
        collateral=tuple(collateral),
    )


def check_provenance(
    before_meta: Mapping[str, Any],
    after_meta: Mapping[str, Any],
    keys: Sequence[str] = (
        "dataset",
        "concurrency",
        "max_concurrency",
        "input_len",
        "output_len",
        "num_chips",
    ),
) -> list[str]:
    """Returns mismatch descriptions if benchmark harness parameters changed."""
    issues: list[str] = []
    for key in keys:
        if key in before_meta and key in after_meta:
            if before_meta[key] != after_meta[key]:
                issues.append(
                    f"benchmark {key} changed: {before_meta[key]!r} -> "
                    f"{after_meta[key]!r}")
    return issues


@dataclasses.dataclass(frozen=True)
class MetricChange:
    """Before/after delta on one serving KPI."""

    name: str
    before: float
    after: float
    higher_is_better: bool

    @property
    def rel_change(self) -> float | None:
        if self.before == 0:
            return None
        return (self.after - self.before) / abs(self.before)

    @property
    def improved_fraction(self) -> float | None:
        rel = self.rel_change
        if rel is None:
            return None
        return rel if self.higher_is_better else -rel

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":
            self.name,
            "before":
            self.before,
            "after":
            self.after,
            "higher_is_better":
            self.higher_is_better,
            "rel_change": (round(self.rel_change, 4)
                           if self.rel_change is not None else None),
        }


@dataclasses.dataclass(frozen=True)
class Comparison:
    """Metric-by-metric comparison between baseline and candidate."""

    throughput: tuple[MetricChange, ...] = ()
    latency: tuple[MetricChange, ...] = ()
    provenance_issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "throughput": [m.to_dict() for m in self.throughput],
            "latency": [m.to_dict() for m in self.latency],
            "provenance_issues": list(self.provenance_issues),
        }


_THROUGHPUT_SPECS: tuple[tuple[str, bool], ...] = (
    ("mtok_per_dollar", True),
    ("token_throughput_per_chip", True),
    ("output_throughput", True),
    ("total_token_throughput", True),
    ("request_throughput", True),
)

_LATENCY_SPECS: tuple[tuple[str, bool], ...] = (
    ("p90_interactivity_tok_s", True),
    ("median_itl_ms", False),
    ("p90_itl_ms", False),
    ("p99_itl_ms", False),
    ("mean_ttft_ms", False),
    ("p90_ttft_ms", False),
    ("p99_ttft_ms", False),
    ("mean_e2el_ms", False),
)


def _num(m: Mapping[str, Any], key: str) -> float | None:
    val = m.get(key)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    fval = float(val)
    if not math.isfinite(fval):
        return None
    return fval


def compare(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> Comparison:
    """Compares `before` and `after` metric dictionaries."""
    b_inner = before.get("metrics")
    a_inner = after.get("metrics")
    bm: Mapping[str, Any] = b_inner if isinstance(b_inner, Mapping) else before
    am: Mapping[str, Any] = a_inner if isinstance(a_inner, Mapping) else after

    tp: list[MetricChange] = []
    for key, hib in _THROUGHPUT_SPECS:
        bv, av = _num(bm, key), _num(am, key)
        if bv is not None and av is not None and bv > 0:
            tp.append(MetricChange(key, bv, av, hib))

    lat: list[MetricChange] = []
    for key, hib in _LATENCY_SPECS:
        bv, av = _num(bm, key), _num(am, key)
        if bv is not None and av is not None and bv > 0:
            lat.append(MetricChange(key, bv, av, hib))

    prov = check_provenance(bm, am)
    return Comparison(
        throughput=tuple(tp),
        latency=tuple(lat),
        provenance_issues=tuple(prov),
    )


@dataclasses.dataclass(frozen=True)
class Verdict:
    """Noise- and mechanism-aware post-run A/B decision."""

    decision: str  # KEEP | DISCARD | INCONCLUSIVE
    reason: str
    noise_band_rel: float
    comparison: Comparison
    mechanism: MechanismVerdict | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision":
            self.decision,
            "reason":
            self.reason,
            "noise_band_rel":
            round(self.noise_band_rel, 4),
            "comparison":
            self.comparison.to_dict(),
            "mechanism":
            (self.mechanism.to_dict() if self.mechanism is not None else None),
        }


def decide_verdict(
    comparison: Comparison,
    *,
    noise_rel: float = 0.0,
    mechanism: MechanismVerdict | None = None,
    min_throughput_gain: float = _DEFAULT_THROUGHPUT_GAIN,
    min_latency_drop: float = _DEFAULT_LATENCY_DROP,
    max_throughput_regression: float = _DEFAULT_THROUGHPUT_REGRESSION,
    max_latency_regression: float = _DEFAULT_LATENCY_REGRESSION,
    eval_drop: float | None = None,
    max_eval_drop: float = 0.015,
) -> Verdict:
    """Decides `KEEP`, `DISCARD`, or `INCONCLUSIVE` from `comparison`."""
    band = max(min_throughput_gain, 2.0 * max(0.0, noise_rel))

    if comparison.provenance_issues:
        return Verdict(
            decision=INCONCLUSIVE,
            reason=("Benchmark provenance mismatch: " +
                    "; ".join(comparison.provenance_issues)),
            noise_band_rel=band,
            comparison=comparison,
            mechanism=mechanism,
        )

    if eval_drop is not None and eval_drop > max_eval_drop:
        return Verdict(
            decision=DISCARD,
            reason=(
                f"Accuracy dropped by {eval_drop:.3f}, exceeding max allowed "
                f"drop {max_eval_drop:.3f}."),
            noise_band_rel=band,
            comparison=comparison,
            mechanism=mechanism,
        )

    if not comparison.throughput and not comparison.latency:
        return Verdict(
            decision=INCONCLUSIVE,
            reason="No overlapping throughput or latency metrics to compare.",
            noise_band_rel=band,
            comparison=comparison,
            mechanism=mechanism,
        )

    # Check hard regressions first.
    tp_reg = [
        m for m in comparison.throughput if m.improved_fraction is not None
        and m.improved_fraction <= -max(max_throughput_regression, band)
    ]
    lat_reg = [
        m for m in comparison.latency if m.improved_fraction is not None
        and m.improved_fraction <= -max_latency_regression
    ]
    if tp_reg:
        worst = min(tp_reg, key=lambda m: m.improved_fraction or 0.0)
        pct = (worst.improved_fraction or 0.0) * 100.0
        return Verdict(
            decision=DISCARD,
            reason=(f"{worst.name} regressed {pct:+.2f}% "
                    f"({worst.before:.2f} -> {worst.after:.2f}), outside "
                    f"±{band * 100:.2f}% band."),
            noise_band_rel=band,
            comparison=comparison,
            mechanism=mechanism,
        )

    tp_win = [
        m for m in comparison.throughput
        if m.improved_fraction is not None and m.improved_fraction >= band
    ]
    lat_win = [
        m for m in comparison.latency if m.improved_fraction is not None
        and m.improved_fraction >= max(min_latency_drop, band)
    ]

    if lat_reg and not tp_win:
        worst = min(lat_reg, key=lambda m: m.improved_fraction or 0.0)
        pct = (worst.improved_fraction or 0.0) * 100.0
        return Verdict(
            decision=DISCARD,
            reason=(
                f"{worst.name} worsened {pct:+.2f}% "
                f"({worst.before:.2f} -> {worst.after:.2f}) with no throughput "
                "gain."),
            noise_band_rel=band,
            comparison=comparison,
            mechanism=mechanism,
        )

    if not tp_win and not lat_win:
        primary = (comparison.throughput[0]
                   if comparison.throughput else comparison.latency[0])
        pct = (primary.improved_fraction or 0.0) * 100.0
        if mechanism is not None and not mechanism.confirmed:
            return Verdict(
                decision=DISCARD,
                reason=
                (f"{primary.name} moved {pct:+.2f}% (within ±{band * 100:.2f}% "
                 f"noise band) and {mechanism.summary}"),
                noise_band_rel=band,
                comparison=comparison,
                mechanism=mechanism,
            )
        return Verdict(
            decision=INCONCLUSIVE,
            reason=(
                f"{primary.name} moved {pct:+.2f}%, within ±{band * 100:.2f}% "
                "replicate noise band; repeat both arms before deciding."),
            noise_band_rel=band,
            comparison=comparison,
            mechanism=mechanism,
        )

    best = tp_win[0] if tp_win else lat_win[0]
    pct = (best.improved_fraction or 0.0) * 100.0
    if mechanism is not None and not mechanism.confirmed:
        return Verdict(
            decision=INCONCLUSIVE,
            reason=(
                f"{best.name} improved {pct:+.2f}%, but {mechanism.summary} "
                "Verify mechanism before banking."),
            noise_band_rel=band,
            comparison=comparison,
            mechanism=mechanism,
        )

    if lat_reg:
        # A keep that its own checks flag is not banked: a throughput win that
        # costs interactivity is a different operating point, not a speedup.
        worst_lat = min(lat_reg, key=lambda m: m.improved_fraction or 0.0)
        return Verdict(
            decision=INCONCLUSIVE,
            reason=(
                f"{best.name} improved {pct:+.2f}%, but {worst_lat.name} "
                f"worsened {(worst_lat.improved_fraction or 0.0) * 100:+.2f}% "
                f"(past the -{max_latency_regression * 100:.1f}% limit). "
                "Compare both arms at matched interactivity before banking."),
            noise_band_rel=band,
            comparison=comparison,
            mechanism=mechanism,
        )
    return Verdict(
        decision=KEEP,
        reason=(f"{best.name} improved {pct:+.2f}% "
                f"({best.before:.2f} -> {best.after:.2f}), clearing "
                f"±{band * 100:.2f}% band."),
        noise_band_rel=band,
        comparison=comparison,
        mechanism=mechanism,
    )


def render_verdict(verdict: Verdict) -> str:
    """Formats `Verdict` as Markdown."""
    lines = [
        f"### Verdict: **{verdict.decision}**",
        "",
        f"- **Reason**: {verdict.reason}",
        f"- **Decision band**: `±{verdict.noise_band_rel * 100:.2f}%`",
    ]
    if verdict.mechanism is not None:
        lines.append(f"- **Mechanism**: {verdict.mechanism.summary}")
    all_metrics = (*verdict.comparison.throughput, *verdict.comparison.latency)
    if all_metrics:
        lines.extend([
            "",
            "| Metric | Baseline | Candidate | Relative Change |",
            "|---|---:|---:|---:|",
        ])
        for m in all_metrics:
            rel = (f"{m.rel_change * 100:+.2f}%"
                   if m.rel_change is not None else "n/a")
            lines.append(
                f"| `{m.name}` | {m.before:.3f} | {m.after:.3f} | {rel} |")
    return "\n".join(lines) + "\n"
