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
"""Trace-to-roofline HBM/ICI/MXU headroom analyzer (`trace_roofline.py`).

Stdlib-only companion to `decompose-optimization` and `roofline-analysis` when
only tabular op summaries (CSV/TSV/JSON/Markdown from `tpu-diagnostics`, without
a raw `.xplane.pb`) are on hand: classifies every op/group through a
`ModelPlugin`, compares each architectural block against its HBM, ICI, matrix,
and optional 4-bit VPU unpack floors (for example `96.7%` MoE HBM roofline
attainment on `GLM-5.2-FP8` decode vs launch-overhead-bound MLA DCP collectives),
splits fixed/padded vs live-request-proportional step cost, and ranks candidate
levers.
"""

import csv
import dataclasses
import io
import json
import pathlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from hillclimb import model_plugins


@dataclasses.dataclass(frozen=True)
class TraceOpRecord:
    """Normalized per-step trace op or op-group timing entry."""

    name: str
    group: str
    ms_per_step: float
    calls_per_step: float = 0.0
    early_ms_per_step: float | None = None


@dataclasses.dataclass(frozen=True)
class PackComparison:
    """Comparison of measured trace time vs analytical `PackRoofline`."""

    pack_name: str
    buckets: tuple[str, ...]
    measured_ms: float
    share_of_step_pct: float
    floor_ms: float
    hbm_floor_ms: float
    ici_floor_ms: float
    compute_floor_ms: float
    limiting_resource: str
    roofline_attainment_pct: float
    excess_over_floor_ms: float
    diagnosis: str
    warnings: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class StepCostSplit:
    """Fixed/padded vs live-request-proportional decomposition of step latency."""

    fixed_or_padded_ms: float
    live_request_proportional_ms: float
    fixed_share_pct: float
    slope_ms_per_req: float
    method: str


@dataclasses.dataclass(frozen=True)
class OptimizationTarget:
    """Ranked optimization lever instantiated against measured trace timings.

    `measured_target_ms` and `share_of_step_pct` come from the trace. The
    `est_saving_ms_low`, `est_saving_ms_high` and `est_step_speedup_pct_high`
    fields are heuristic estimates: the measured target time times the
    lever's hand-set `est_saving_share` prior, not measured savings.
    """

    lever_name: str
    title: str
    target_categories: tuple[str, ...]
    measured_target_ms: float
    share_of_step_pct: float
    est_saving_ms_low: float
    est_saving_ms_high: float
    est_step_speedup_pct_high: float
    mechanism: str
    action: str
    requires_accuracy_gate: bool


@dataclasses.dataclass(frozen=True)
class RooflineReport:
    """Complete trace-to-roofline diagnostic report for one trace snapshot."""

    plugin_name: str
    display_name: str
    trace_source: str
    section: str
    step_ms: float
    wall_step_ms: float
    context: model_plugins.StepContext
    bucket_totals_ms: Mapping[str, float]
    category_totals_ms: Mapping[str, float]
    pack_comparisons: tuple[PackComparison, ...]
    cost_split: StepCostSplit
    ranked_targets: tuple[OptimizationTarget, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_name":
            self.plugin_name,
            "display_name":
            self.display_name,
            "trace_source":
            self.trace_source,
            "section":
            self.section,
            "step_ms":
            round(self.step_ms, 4),
            "wall_step_ms":
            round(self.wall_step_ms, 4),
            "context":
            dataclasses.asdict(self.context),
            "bucket_totals_ms": {
                k: round(v, 4)
                for k, v in self.bucket_totals_ms.items()
            },
            "category_totals_ms": {
                k: round(v, 4)
                for k, v in self.category_totals_ms.items()
            },
            "pack_comparisons":
            [dataclasses.asdict(pc) for pc in self.pack_comparisons],
            "cost_split":
            dataclasses.asdict(self.cost_split),
            "ranked_targets":
            [dataclasses.asdict(t) for t in self.ranked_targets],
        }

    def render_markdown(self, top_n_targets: int = 5) -> str:
        lines = [
            f"# Trace-to-Roofline Report: {self.display_name} (`{self.plugin_name}`)",
            "",
            f"- **Source**: `{self.trace_source}` (section `{self.section}`)",
            (f"- **Busy device step time**: **{self.step_ms:.3f} ms/step** "
             f"(wall span `{self.wall_step_ms:.3f} ms/step`)"),
            (f"- **Step context**: `batch_reqs={self.context.batch_reqs}`, "
             f"`padded_rows={self.context.padded_rows}`, "
             f"`context_tokens={self.context.context_tokens}`, "
             f"`weight_dtype={self.context.weight_dtype}`"),
            ("- **Fixed vs proportional**: "
             f"**{self.cost_split.fixed_or_padded_ms:.2f} ms** "
             f"({self.cost_split.fixed_share_pct:.1f}%) fixed/padded + "
             f"**{self.cost_split.live_request_proportional_ms:.2f} ms** "
             "live-request proportional "
             f"(`{self.cost_split.slope_ms_per_req:.3f} ms/req`, "
             f"`{self.cost_split.method}`)"),
            "",
            "## Pack Roofline Comparison",
            "",
            ("| Pack | Measured (ms) | Step % | Floor (ms) | Bound | "
             "Attainment | Diagnosis |"),
            "|---|---:|---:|---:|---|---:|---|",
        ]
        for pc in self.pack_comparisons:
            lines.append(
                f"| `{pc.pack_name}` | {pc.measured_ms:.3f} | "
                f"{pc.share_of_step_pct:.1f}% | {pc.floor_ms:.3f} | "
                f"`{pc.limiting_resource}` | {pc.roofline_attainment_pct:.1f}% | "
                f"{pc.diagnosis} |")
        lines.extend([
            "",
            "## Category Breakdown",
            "",
            "| Category | ms/step | Share % |",
            "|---|---:|---:|",
        ])
        sorted_cats = sorted(self.category_totals_ms.items(),
                             key=lambda kv: kv[1],
                             reverse=True)
        for cat, ms in sorted_cats:
            pct = (ms / self.step_ms * 100.0) if self.step_ms > 0 else 0.0
            lines.append(f"| `{cat}` | {ms:.3f} | {pct:.1f}% |")

        lines.extend([
            "",
            f"## Top {top_n_targets} Ranked Optimization Targets",
            "",
            "Saving ranges are heuristic estimates: the measured time of the "
            "targeted categories times each lever's hand-set prior share, "
            "not measured results.",
            "",
        ])
        for idx, target in enumerate(self.ranked_targets[:top_n_targets],
                                     start=1):
            gate = (" *(requires accuracy gate)*"
                    if target.requires_accuracy_gate else "")
            lines.append(
                f"{idx}. **{target.title}** (`{target.lever_name}`){gate}: "
                f"targets `{', '.join(target.target_categories)}` "
                f"({target.measured_target_ms:.2f} ms / "
                f"{target.share_of_step_pct:.1f}% of step) -> est. save "
                f"**{target.est_saving_ms_low:.2f}–{target.est_saving_ms_high:.2f} "
                "ms/step** (heuristic; up to "
                f"{target.est_step_speedup_pct_high:.1f}% step reduction). "
                f"Action: {target.action}")
        return "\n".join(lines) + "\n"


def _parse_batch_composition_hint(
    hint: str, ) -> tuple[int | None, int | None, int | None, float | None]:
    early_reqs: float | None = None
    m_early_range = re.search(r"(\d+)\s*-\s*(\d+)\s*reqs", hint)
    if m_early_range:
        early_reqs = 0.5 * (float(m_early_range.group(1)) +
                            float(m_early_range.group(2)))
    else:
        m_early_single = re.search(r"batches\s+1-\d+:\s*(\d+)\s*reqs", hint)
        if m_early_single:
            early_reqs = float(m_early_single.group(1))

    all_reqs = re.findall(r"(\d+)\s*reqs\s*=\s*\d+\s*tok", hint)
    full_reqs = int(all_reqs[-1]) if all_reqs else None

    m_pad = re.search(r"padded\s+(\d+)", hint)
    padded_rows = int(m_pad.group(1)) if m_pad else None

    kv_matches = re.findall(r"(\d+(?:\.\d+)?)\s*[kK]", hint)
    context_tokens = (int(round(float(kv_matches[-1]) *
                                1000.0)) if kv_matches else None)
    return full_reqs, padded_rows, context_tokens, early_reqs


def _diagnose_attainment(attainment_pct: float, limiting: str) -> str:
    if attainment_pct >= 80.0:
        return (
            f"{limiting.upper()}-saturated ({attainment_pct:.0f}% of floor): "
            "cut weight/KV bytes (FP4/W4A8 or sparsity), not kernel scheduling"
        )
    if attainment_pct < 45.0:
        return (
            f"Launch/collective-overhead bound ({attainment_pct:.0f}% of "
            f"{limiting.upper()} floor): fuse collectives, overlap waits, or "
            "eliminate relayout copies")
    return (
        f"Moderate efficiency gap ({attainment_pct:.0f}% of {limiting.upper()} "
        "floor): tune tile/piece pipelining and row buckets")


def _load_rank_comparison_json(
    raw: Mapping[str, Any],
    section: str,
) -> tuple[
        list[TraceOpRecord],
        float,
        float,
        tuple[int | None, int | None, int | None, float | None],
]:
    sec_obj = raw.get(section)
    if not isinstance(sec_obj, Mapping):
        raise ValueError(f"Section {section!r} not found in trace JSON.")
    rank_obj = sec_obj.get("rank0_tpu0")
    if not isinstance(rank_obj, Mapping):
        for candidate in sec_obj.values():
            if isinstance(candidate,
                          Mapping) and ("groups_ms_per_step_full_batch"
                                        in candidate):
                rank_obj = candidate
                break
    if not isinstance(rank_obj, Mapping):
        raise ValueError(
            f"No rank breakdown dict found under section {section!r}.")

    step_dict = rank_obj.get("device_op_ms_per_step", {})
    groups_full = rank_obj.get("groups_ms_per_step_full_batch", {})
    groups_early = rank_obj.get("groups_ms_per_step_early", {})
    top_ops = rank_obj.get("top_ops_full_batch", [])

    rep_op_by_group: dict[str, str] = {}
    calls_by_group: dict[str, float] = {}
    if isinstance(top_ops, Sequence):
        for item in top_ops:
            if not isinstance(item, Mapping):
                continue
            grp = str(item.get("group", ""))
            op_name = str(item.get("op", ""))
            if grp and grp not in rep_op_by_group:
                rep_op_by_group[grp] = op_name
                calls_by_group[grp] = float(
                    item.get("calls_per_step", 0.0) or 0.0)

    records: list[TraceOpRecord] = []
    for grp_name, ms_val in groups_full.items():
        grp_str = str(grp_name)
        early_val = (float(groups_early[grp_str])
                     if isinstance(groups_early, Mapping)
                     and grp_str in groups_early else None)
        records.append(
            TraceOpRecord(
                name=rep_op_by_group.get(grp_str, grp_str),
                group=grp_str,
                ms_per_step=float(ms_val),
                calls_per_step=calls_by_group.get(grp_str, 0.0),
                early_ms_per_step=early_val,
            ))

    step_ms = float(
        step_dict.get("full_batch") or sum(r.ms_per_step for r in records))
    meta = raw.get("meta", {})
    hint_str = ""
    if isinstance(meta, Mapping):
        bc = meta.get("batch_composition", {})
        if isinstance(bc, Mapping):
            hint_str = str(bc.get(section, ""))
    hints = _parse_batch_composition_hint(hint_str)
    return records, step_ms, step_ms, hints


def _load_summary_json(
    raw: Mapping[str, Any], ) -> tuple[list[TraceOpRecord], float, float]:
    summary = raw.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("Missing `summary` object in breakdown JSON.")
    buckets = summary.get("buckets", {})
    top_ops = summary.get("top_ops", [])

    records: list[TraceOpRecord] = []
    top_sum_by_bucket: dict[str, float] = {}
    if isinstance(top_ops, Sequence):
        for item in top_ops:
            if not isinstance(item, Mapping):
                continue
            op_name = str(item.get("name", ""))
            bkt = str(item.get("bucket", "other"))
            ms = float(item.get("ms_per_step", 0.0) or 0.0)
            calls = float(item.get("count_per_step", 0.0) or 0.0)
            records.append(
                TraceOpRecord(
                    name=op_name,
                    group=bkt,
                    ms_per_step=ms,
                    calls_per_step=calls,
                ))
            top_sum_by_bucket[bkt] = top_sum_by_bucket.get(bkt, 0.0) + ms

    if isinstance(buckets, Mapping):
        for bkt_name, bkt_info in buckets.items():
            if not isinstance(bkt_info, Mapping):
                continue
            total_bkt_ms = float(bkt_info.get("avg_ms_per_step", 0.0) or 0.0)
            residual = total_bkt_ms - top_sum_by_bucket.get(str(bkt_name), 0.0)
            if residual > 1e-4:
                records.append(
                    TraceOpRecord(
                        name=str(bkt_name),
                        group=str(bkt_name),
                        ms_per_step=residual,
                    ))

    records_sum = sum(r.ms_per_step for r in records)
    avg_steps = float(summary.get("avg_steps", 1.0) or 1.0)
    raw_busy = float(summary.get("avg_busy_ops_time_ms", 0.0) or 0.0)
    if records_sum > 0:
        busy_ms = records_sum
    elif avg_steps > 1.0 and raw_busy > 0:
        busy_ms = raw_busy / avg_steps
    else:
        busy_ms = raw_busy
    wall_ms = float(summary.get("avg_step_time_ms") or busy_ms)
    return records, busy_ms, wall_ms


def parse_tabular_ops(text: str) -> list[TraceOpRecord]:
    """Parses CSV, TSV, or pipe-delimited Markdown op timing tables."""
    stripped_lines = [
        ln.strip() for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not stripped_lines:
        return []

    if stripped_lines[0].startswith("|"):
        rows: list[list[str]] = []
        for ln in stripped_lines:
            if not ln.startswith("|"):
                continue
            cells = [c.strip().strip("`") for c in ln.strip("|").split("|")]
            if all(not c or set(c) <= {"-", ":"} for c in cells):
                continue
            rows.append(cells)
        if len(rows) < 2:
            return []
        header = [h.lower() for h in rows[0]]
        dict_rows = [dict(zip(header, r)) for r in rows[1:]]
    else:
        delimiter = "\t" if "\t" in stripped_lines[0] else ","
        reader = csv.DictReader(io.StringIO("\n".join(stripped_lines)),
                                delimiter=delimiter)
        dict_rows = [{
            (k or "").strip().lower(): (v or "").strip()
            for k, v in row.items()
        } for row in reader]

    records: list[TraceOpRecord] = []
    for row in dict_rows:
        name = (row.get("op") or row.get("name") or row.get("hlo_op")
                or row.get("kernel") or row.get("group") or "")
        group = row.get("group") or row.get("bucket") or row.get(
            "category") or ""
        ms_raw = (row.get("ms_per_step") or row.get("self_time_ms")
                  or row.get("avg_ms_per_step") or row.get("time_ms") or "")
        us_raw = row.get("us_per_step") or row.get("self_time_us") or ""
        if ms_raw:
            ms = float(ms_raw)
        elif us_raw:
            ms = float(us_raw) / 1000.0
        else:
            continue
        calls_raw = row.get("calls_per_step") or row.get(
            "count_per_step") or "0"
        records.append(
            TraceOpRecord(
                name=name,
                group=group,
                ms_per_step=ms,
                calls_per_step=float(calls_raw or 0.0),
            ))
    return records


def _compute_cost_split(
    records: Sequence[TraceOpRecord],
    classifications: Sequence[model_plugins.Classification],
    step_ms: float,
    ctx: model_plugins.StepContext,
    early_reqs: float | None,
) -> StepCostSplit:
    has_early = (early_reqs is not None and ctx.batch_reqs > early_reqs + 0.5
                 and all(r.early_ms_per_step is not None for r in records)
                 and bool(records))
    if has_early and early_reqs is not None:
        early_total = sum(float(r.early_ms_per_step or 0.0) for r in records)
        full_total = sum(r.ms_per_step for r in records)
        delta_reqs = float(ctx.batch_reqs) - early_reqs
        slope = max(0.0, (full_total - early_total) / delta_reqs)
        proportional = min(step_ms, slope * float(ctx.batch_reqs))
        fixed_ms = max(0.0, step_ms - proportional)
        fixed_pct = (fixed_ms / step_ms * 100.0) if step_ms > 0 else 0.0
        return StepCostSplit(
            fixed_or_padded_ms=fixed_ms,
            live_request_proportional_ms=proportional,
            fixed_share_pct=fixed_pct,
            slope_ms_per_req=slope,
            method="within_trace_early_vs_full",
        )

    proportional = 0.0
    for rec, cls in zip(records, classifications):
        if cls.scaling == "live_rows":
            proportional += rec.ms_per_step
        elif cls.scaling == "expert_coverage":
            proportional += 0.20 * rec.ms_per_step
        elif cls.scaling == "padded_rows_x_ctx":
            proportional += 0.25 * rec.ms_per_step
    proportional = min(step_ms, proportional)
    fixed_ms = max(0.0, step_ms - proportional)
    slope = proportional / max(1, ctx.batch_reqs)
    fixed_pct = (fixed_ms / step_ms * 100.0) if step_ms > 0 else 0.0
    return StepCostSplit(
        fixed_or_padded_ms=fixed_ms,
        live_request_proportional_ms=proportional,
        fixed_share_pct=fixed_pct,
        slope_ms_per_req=slope,
        method="scaling_rule_model",
    )


def analyze_records(
    records: Sequence[TraceOpRecord],
    plugin: model_plugins.ModelPlugin,
    ctx: model_plugins.StepContext,
    *,
    step_ms: float | None = None,
    wall_step_ms: float | None = None,
    trace_source: str = "<memory>",
    section: str = "default",
    early_reqs: float | None = None,
) -> RooflineReport:
    """Runs roofline comparison, step-cost splitting, and lever ranking."""
    classifications = [plugin.classify_op(r.name, r.group) for r in records]
    total_ops_ms = sum(r.ms_per_step for r in records)
    eff_step_ms = (step_ms
                   if step_ms is not None and step_ms > 0 else total_ops_ms)
    eff_wall_ms = (wall_step_ms if wall_step_ms is not None
                   and wall_step_ms > 0 else eff_step_ms)

    bucket_totals: dict[str, float] = {}
    category_totals: dict[str, float] = {}
    for rec, cls in zip(records, classifications):
        bucket_totals[cls.bucket] = (bucket_totals.get(cls.bucket, 0.0) +
                                     rec.ms_per_step)
        category_totals[cls.category] = (
            category_totals.get(cls.category, 0.0) + rec.ms_per_step)

    rooflines = plugin.rooflines(ctx)
    comparisons: list[PackComparison] = []
    for pack in plugin.packs:
        rf = rooflines[pack.name]
        measured = sum(bucket_totals.get(b, 0.0) for b in rf.buckets)
        share_pct = (measured / eff_step_ms *
                     100.0) if eff_step_ms > 0 else 0.0
        attainment = (min(100.0, rf.floor_ms_per_step / measured *
                          100.0) if measured > 0 else 0.0)
        excess = max(0.0, measured - rf.floor_ms_per_step)
        comparisons.append(
            PackComparison(
                pack_name=pack.name,
                buckets=rf.buckets,
                measured_ms=measured,
                share_of_step_pct=share_pct,
                floor_ms=rf.floor_ms_per_step,
                hbm_floor_ms=rf.hbm_floor_ms_per_step,
                ici_floor_ms=rf.ici_floor_ms_per_step,
                compute_floor_ms=rf.compute_floor_ms_per_step,
                limiting_resource=rf.limiting_resource,
                roofline_attainment_pct=attainment,
                excess_over_floor_ms=excess,
                diagnosis=_diagnose_attainment(attainment,
                                               rf.limiting_resource),
                warnings=rf.warnings,
            ))

    cost_split = _compute_cost_split(records, classifications, eff_step_ms,
                                     ctx, early_reqs)

    targets: list[OptimizationTarget] = []
    for lever in plugin.all_levers():
        target_ms = sum(
            category_totals.get(cat, 0.0) for cat in lever.target_categories)
        if target_ms <= 0:
            continue
        low_frac, high_frac = lever.est_saving_share
        save_low = target_ms * low_frac
        save_high = target_ms * high_frac
        share_pct = (target_ms / eff_step_ms *
                     100.0) if eff_step_ms > 0 else 0.0
        speedup_high = ((save_high / eff_step_ms *
                         100.0) if eff_step_ms > 0 else 0.0)
        targets.append(
            OptimizationTarget(
                lever_name=lever.name,
                title=lever.title,
                target_categories=lever.target_categories,
                measured_target_ms=target_ms,
                share_of_step_pct=share_pct,
                est_saving_ms_low=save_low,
                est_saving_ms_high=save_high,
                est_step_speedup_pct_high=speedup_high,
                mechanism=lever.mechanism,
                action=lever.action,
                requires_accuracy_gate=lever.requires_accuracy_gate,
            ))
    targets.sort(key=lambda t: t.est_saving_ms_high, reverse=True)

    return RooflineReport(
        plugin_name=plugin.name,
        display_name=plugin.display_name,
        trace_source=trace_source,
        section=section,
        step_ms=eff_step_ms,
        wall_step_ms=eff_wall_ms,
        context=ctx,
        bucket_totals_ms=bucket_totals,
        category_totals_ms=category_totals,
        pack_comparisons=tuple(comparisons),
        cost_split=cost_split,
        ranked_targets=tuple(targets),
    )


def analyze_trace_file(
    trace_path: str | pathlib.Path,
    plugin: model_plugins.ModelPlugin
    | str = model_plugins.DEFAULT_PLUGIN_NAME,
    *,
    section: str = "new",
    batch_reqs: int | None = None,
    padded_rows: int | None = None,
    context_tokens: int | None = None,
    weight_dtype: str = "fp8",
    quant_block_size: int = 0,
    row_compacted: bool = False,
    moe_mode: str = "",
) -> RooflineReport:
    """Loads a JSON or CSV/TSV/Markdown op profile and returns `RooflineReport`."""
    resolved_plugin = (model_plugins.get_plugin(plugin) if isinstance(
        plugin, str) else plugin)
    path = pathlib.Path(trace_path)
    text = path.read_text(encoding="utf-8")

    hint_reqs: int | None = None
    hint_rows: int | None = None
    hint_ctx: int | None = None
    early_reqs: float | None = None

    if path.suffix.lower() == ".json" or text.lstrip().startswith("{"):
        raw = json.loads(text)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Expected JSON object in {path}.")
        if "summary" in raw and isinstance(raw["summary"], Mapping):
            records, step_ms, wall_ms = _load_summary_json(raw)
            eff_section = "summary"
        else:
            chosen_sec = section if section in raw else "new"
            records, step_ms, wall_ms, hints = _load_rank_comparison_json(
                raw, chosen_sec)
            hint_reqs, hint_rows, hint_ctx, early_reqs = hints
            eff_section = chosen_sec
    else:
        records = parse_tabular_ops(text)
        if not records:
            raise ValueError(f"No parseable trace op rows found in {path}.")
        step_ms = sum(r.ms_per_step for r in records)
        wall_ms = step_ms
        eff_section = "table"

    eff_reqs = batch_reqs or hint_reqs or 16
    default_rows = eff_reqs * (1 +
                               max(0, resolved_plugin.dims.mtp_draft_tokens))
    eff_rows = padded_rows or hint_rows or default_rows
    eff_ctx = context_tokens or hint_ctx or 50400

    ctx = model_plugins.StepContext(
        batch_reqs=eff_reqs,
        padded_rows=eff_rows,
        context_tokens=eff_ctx,
        weight_dtype=weight_dtype,
        quant_block_size=quant_block_size,
        row_compacted=row_compacted,
        moe_mode=moe_mode,
    )
    return analyze_records(
        records,
        resolved_plugin,
        ctx,
        step_ms=step_ms,
        wall_step_ms=wall_ms,
        trace_source=str(path),
        section=eff_section,
        early_reqs=early_reqs,
    )
