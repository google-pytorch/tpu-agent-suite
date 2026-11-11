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
"""1P1D disaggregated prefill-vs-decode bottleneck attribution from `/metrics`.

Before spending multi-slice chip-hours on an arm that accelerates only one leg
of a disaggregated deployment, check whether that leg actually limits end-to-end
throughput. On a GLM-5.2 1P1D run at `c=48` whose prefill
`vllm:num_requests_waiting` was 0 at p50 and p75, making the prefill drafter
`2.8x` faster (`7.5 ms -> 2.7 ms` per draft pass) yielded `0.999x` end-to-end
throughput: decode was the limit.
"""

import csv
import dataclasses
import io
import json
import re
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

_WAITING = "vllm:num_requests_waiting"
_RUNNING = "vllm:num_requests_running"

_QUEUE_P75_IDLE = 1.0
_QUEUE_P50_SATURATED = 1.0
_BUSY_FRACTION_LIGHT = 0.70

DECODE_LIMITED = "decode_limited"
PREFILL_QUEUED = "prefill_queued"
BALANCED = "balanced"

_PROM_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')


@dataclasses.dataclass(frozen=True)
class LegLoad:
    """Queueing and occupancy statistics for one prefill or decode endpoint."""

    endpoint: str
    role: str  # prefill | decode
    waiting_mean: float
    waiting_p50: float
    waiting_p75: float
    waiting_p90: float
    running_mean: float = 0.0
    busy_fraction: float | None = None


# Alias retained for callers reading prefill endpoints specifically.
PrefillLoad = LegLoad


@dataclasses.dataclass(frozen=True)
class Verdict:
    """Which leg limits a disaggregated P/D deployment."""

    kind: str  # decode_limited | prefill_queued | balanced
    prefills: tuple[LegLoad, ...]
    decodes: tuple[LegLoad, ...] = ()
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "summary": self.summary,
            "prefills": [dataclasses.asdict(p) for p in self.prefills],
            "decodes": [dataclasses.asdict(d) for d in self.decodes],
        }


def _opt_float(value: Any) -> float | None:
    try:
        return (None if value is None or value == ""
                or isinstance(value, bool) else float(value))
    except (TypeError, ValueError):
        return None


def _role_from_endpoint(endpoint: str,
                        labels: Mapping[str, str] | None = None) -> str | None:
    text = endpoint.lower()
    if labels:
        role_lbl = str(
            labels.get("role") or labels.get("leg") or labels.get("pod")
            or labels.get("service") or "").lower()
        text = f"{text} {role_lbl}"
    if "prefill" in text or ":8081" in text or ":8100" in text:
        return "prefill"
    if "decode" in text or ":8082" in text or ":8200" in text:
        return "decode"
    return None


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(len(sorted_vals) - 1, lo + 1)
    w = pos - lo
    return (1.0 - w) * sorted_vals[lo] + w * sorted_vals[hi]


def loads_from_csv(text: str) -> tuple[list[LegLoad], list[LegLoad]]:
    """Extracts `(prefill_loads, decode_loads)` from `server_metrics_export.csv`."""
    waiting_rows: dict[tuple[str, str], Mapping[str, str]] = {}
    running_rows: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        endpoint = row.get("endpoint_url") or row.get("endpoint") or ""
        role = _role_from_endpoint(endpoint, row)
        if role is None:
            continue
        metric = row.get("metric_name", "")
        if metric == _WAITING and row.get("avg") not in (None, ""):
            waiting_rows[(role, endpoint)] = row
        elif metric == _RUNNING and row.get("avg") not in (None, ""):
            running_rows[(role, endpoint)] = row

    prefills: list[LegLoad] = []
    decodes: list[LegLoad] = []
    for (role, endpoint), w in sorted(waiting_rows.items()):
        r = running_rows.get((role, endpoint), {})
        run_mean = _opt_float(r.get("avg")) or 0.0
        run_max = _opt_float(r.get("max"))
        busy: float | None = None
        if run_max is not None and 0 < run_max <= 1.0:
            busy = run_mean
        elif run_max is not None and run_max > 1.0:
            busy = min(1.0, run_mean / run_max)
        load = LegLoad(
            endpoint=endpoint,
            role=role,
            waiting_mean=_opt_float(w.get("avg")) or 0.0,
            waiting_p50=_opt_float(w.get("p50")) or 0.0,
            waiting_p75=_opt_float(w.get("p75")) or 0.0,
            waiting_p90=_opt_float(w.get("p90")) or 0.0,
            running_mean=run_mean,
            busy_fraction=busy,
        )
        if role == "prefill":
            prefills.append(load)
        else:
            decodes.append(load)
    return prefills, decodes


def prefill_loads_from_csv(text: str) -> list[LegLoad]:
    """Compatibility helper returning only prefill `LegLoad`s from CSV."""
    prefills, _ = loads_from_csv(text)
    return prefills


def loads_from_export(
    export: Mapping[str, Any], ) -> tuple[list[LegLoad], list[LegLoad]]:
    """Extracts `(prefill_loads, decode_loads)` from `server_metrics_export.json`."""

    def series_by_endpoint(
            metric: str) -> dict[tuple[str, str], Mapping[str, Any]]:
        found: dict[tuple[str, str], Mapping[str, Any]] = {}
        entry = export.get(metric)
        series_list = entry.get("series") if isinstance(entry,
                                                        Mapping) else entry
        if not isinstance(series_list, list):
            return found
        for item in series_list:
            if not isinstance(item, Mapping) or item.get("labels"):
                continue
            endpoint = str(
                item.get("endpoint_url") or item.get("endpoint") or "")
            role = _role_from_endpoint(endpoint, item)
            stats = item.get("stats")
            if role is not None and isinstance(stats, Mapping):
                found[(role, endpoint)] = stats
        return found

    waiting = series_by_endpoint(_WAITING)
    running = series_by_endpoint(_RUNNING)
    prefills: list[LegLoad] = []
    decodes: list[LegLoad] = []
    for (role, endpoint), w in sorted(waiting.items()):
        r = running.get((role, endpoint), {})
        run_mean = _opt_float(r.get("avg")) or 0.0
        run_max = _opt_float(r.get("max"))
        busy = (run_mean if run_max is not None and 0 < run_max <= 1.0 else (
            min(1.0, run_mean /
                run_max) if run_max is not None and run_max > 1.0 else None))
        load = LegLoad(
            endpoint=endpoint,
            role=role,
            waiting_mean=_opt_float(w.get("avg")) or 0.0,
            waiting_p50=_opt_float(w.get("p50")) or 0.0,
            waiting_p75=_opt_float(w.get("p75")) or 0.0,
            waiting_p90=_opt_float(w.get("p90")) or 0.0,
            running_mean=run_mean,
            busy_fraction=busy,
        )
        if role == "prefill":
            prefills.append(load)
        else:
            decodes.append(load)
    return prefills, decodes


def prefill_loads_from_export(export: Mapping[str, Any]) -> list[LegLoad]:
    """Compatibility helper returning only prefill `LegLoad`s from JSON."""
    prefills, _ = loads_from_export(export)
    return prefills


def loads_from_prometheus(text: str) -> tuple[list[LegLoad], list[LegLoad]]:
    """Parses one or more Prometheus `/metrics` scrape snapshots."""
    waiting_samples: dict[tuple[str, str], list[float]] = {}
    running_samples: dict[tuple[str, str], list[float]] = {}
    busy_override: dict[tuple[str, str], list[float]] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE_RE.match(line)
        if not m:
            continue
        metric_name, label_str, val_str = m.groups()
        labels = dict(_LABEL_RE.findall(label_str or ""))
        endpoint = (labels.get("endpoint") or labels.get("instance")
                    or labels.get("pod") or labels.get("role")
                    or labels.get("leg") or "")
        role = _role_from_endpoint(endpoint, labels)
        if role is None:
            continue
        val = float(val_str)
        key = (role, endpoint or role)
        if metric_name == _WAITING and "reason" not in labels:
            waiting_samples.setdefault(key, []).append(val)
        elif metric_name == _RUNNING:
            running_samples.setdefault(key, []).append(val)
        elif metric_name in ("vllm:leg_busy_ratio",
                             "vllm:gpu_cache_usage_perc"):
            if metric_name == "vllm:leg_busy_ratio":
                busy_override.setdefault(key, []).append(val)

    prefills: list[LegLoad] = []
    decodes: list[LegLoad] = []
    for (role, endpoint), w_vals in sorted(waiting_samples.items()):
        ws = sorted(w_vals)
        r_vals = running_samples.get((role, endpoint), [])
        run_mean = statistics.fmean(r_vals) if r_vals else 0.0
        busy: float | None = None
        if (role, endpoint) in busy_override:
            busy = statistics.fmean(busy_override[(role, endpoint)])
        elif len(r_vals) >= 2:
            busy = sum(1 for v in r_vals if v > 0) / len(r_vals)
        elif len(r_vals) == 1 and role == "prefill":
            busy = 1.0 if r_vals[0] >= 1.0 else max(0.0, r_vals[0])
        load = LegLoad(
            endpoint=endpoint,
            role=role,
            waiting_mean=statistics.fmean(ws),
            waiting_p50=_percentile(ws, 0.50),
            waiting_p75=_percentile(ws, 0.75),
            waiting_p90=_percentile(ws, 0.90),
            running_mean=run_mean,
            busy_fraction=busy,
        )
        if role == "prefill":
            prefills.append(load)
        else:
            decodes.append(load)
    return prefills, decodes


def parse_server_metrics(text: str) -> tuple[list[LegLoad], list[LegLoad]]:
    """Auto-detects JSON, CSV, or Prometheus `/metrics` text."""
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return loads_from_export(json.loads(stripped))
    first_line = stripped.splitlines()[0] if stripped else ""
    if "," in first_line and ("metric_name" in first_line
                              or "endpoint" in first_line):
        return loads_from_csv(text)
    prom_p, prom_d = loads_from_prometheus(text)
    if prom_p or prom_d:
        return prom_p, prom_d
    return loads_from_csv(text)


def assess(
        prefills: Sequence[LegLoad],
        decodes: Sequence[LegLoad] = (),
) -> Verdict:
    """Classifies whether prefill or decode limits a disaggregated P/D run."""
    if not prefills:
        raise ValueError(
            "No prefill endpoint metrics found: expected a disaggregated P/D "
            "metrics export containing vllm:num_requests_waiting.")
    max_p75 = max(p.waiting_p75 for p in prefills)
    max_p50 = max(p.waiting_p50 for p in prefills)
    max_p90 = max(p.waiting_p90 for p in prefills)
    busiest = max(
        (p.busy_fraction for p in prefills if p.busy_fraction is not None),
        default=None,
    )
    busy_note = f", busy {busiest:.0%} of the run" if busiest is not None else ""
    decode_note = ""
    if decodes:
        dec_run = max(d.running_mean for d in decodes)
        dec_wait = max(d.waiting_p50 for d in decodes)
        decode_note = (
            f"; decode running mean {dec_run:.1f}, waiting p50 {dec_wait:.0f}")

    light = busiest is None or busiest < _BUSY_FRACTION_LIGHT
    if max_p75 < _QUEUE_P75_IDLE and light:
        return Verdict(
            kind=DECODE_LIMITED,
            prefills=tuple(prefills),
            decodes=tuple(decodes),
            summary=
            (f"Prefill queue p50={max_p50:.0f}, p75={max_p75:.0f} "
             f"(p90={max_p90:.0f}{busy_note}{decode_note}): requests rarely "
             "wait at prefill, so decode throughput or client concurrency "
             "limits end-to-end Mtok/$. Prefill-only speedups will not move "
             "end-to-end throughput."),
        )
    if max_p50 >= _QUEUE_P50_SATURATED:
        return Verdict(
            kind=PREFILL_QUEUED,
            prefills=tuple(prefills),
            decodes=tuple(decodes),
            summary=(
                f"Prefill queue p50={max_p50:.0f}, p75={max_p75:.0f} "
                f"(p90={max_p90:.0f}{busy_note}{decode_note}): requests queue "
                "behind an in-flight prefill at least half the time, so "
                "prefill latency and KV cache hit rate directly move Mtok/$."),
        )
    return Verdict(
        kind=BALANCED,
        prefills=tuple(prefills),
        decodes=tuple(decodes),
        summary=(
            f"Prefill queue p50={max_p50:.0f}, p75={max_p75:.0f} "
            f"(p90={max_p90:.0f}{busy_note}{decode_note}): bursts queue at "
            "prefill without steady saturation; check both legs."),
    )


def _knob_leg(knob_name: str) -> str | None:
    tokens = knob_name.upper().split("_")
    has_p = "PREFILL" in tokens or "PREFILLS" in tokens
    has_d = "DECODE" in tokens or "DECODES" in tokens
    if has_p and not has_d:
        return "prefill"
    if has_d and not has_p:
        return "decode"
    return None


def arm_leg(changed_knobs: Sequence[str]) -> str | None:
    """Returns `'prefill'` or `'decode'` when every changed knob targets one leg."""
    if not changed_knobs:
        return None
    roles = {_knob_leg(k) for k in changed_knobs}
    if roles == {"prefill"}:
        return "prefill"
    if roles == {"decode"}:
        return "decode"
    return None


def warn_off_leg_arm(
    verdict: Verdict,
    changed_knobs: Sequence[str],
) -> str | None:
    """Returns a warning when `changed_knobs` only optimize the non-binding leg."""
    leg = arm_leg(changed_knobs)
    if leg == "prefill" and verdict.kind == DECODE_LIMITED:
        return (
            f"Arm changes only prefill knobs ({', '.join(changed_knobs)}), "
            f"but baseline is decode-limited ({verdict.summary})")
    if leg == "decode" and verdict.kind == PREFILL_QUEUED:
        return (f"Arm changes only decode knobs ({', '.join(changed_knobs)}), "
                f"but baseline is prefill-queued ({verdict.summary})")
    return None


def render(verdict: Verdict) -> str:
    """Formats a 1P1D leg-bottleneck `Verdict` as Markdown."""
    lines = [
        f"leg_verdict: **{verdict.kind}**",
        "",
        verdict.summary,
        "",
        ("| Role | Endpoint | Waiting mean | p50 | p75 | p90 | "
         "Running mean | Busy |"),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in (*verdict.prefills, *verdict.decodes):
        busy = f"{p.busy_fraction:.0%}" if p.busy_fraction is not None else "n/a"
        lines.append(
            f"| {p.role} | `{p.endpoint}` | {p.waiting_mean:.2f} | "
            f"{p.waiting_p50:.0f} | {p.waiting_p75:.0f} | {p.waiting_p90:.0f} | "
            f"{p.running_mean:.2f} | {busy} |")
    return "\n".join(lines) + "\n"
