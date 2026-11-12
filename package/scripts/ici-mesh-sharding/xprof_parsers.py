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
"""Pure-stdlib parsers for exported XProf JSON tool payloads.

Parses JSON files or strings exported by `tpu-diagnostics` (for example
`hlo_op_profile`, `memory_profile`, `memory_viewer`, `megascale_stats`,
`overview_page`, and `kpi_metrics`) into normalized Python dictionaries
consumed by `collective_breakdown` and `rank_mesh_proposals`.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

_MIB = float(1 << 20)
_GIB = float(1 << 30)

# Serving / inference module-name patterns (vLLM-TPU, TorchTPU, JAX serving).
# Serving runtimes pre-reserve HBM for the KV cache at startup, so
# `memory_profile.json`'s peak reflects static reservation rather than training
# activation + weight peak.
_SERVING_MODULE_RE = re.compile(
    r"^(?:tt_jit_|(?:jit_step_fun|jit_run_model|jit__decode_core"
    r"|jit_generate|jit_insert_with_scratchpad)\b)"
    r"|vllm|paged_attention|ragged_paged_attention",
    re.IGNORECASE,
)

_KPI_VALUE_KEYS = ("doubleValue", "uintValue", "intValue", "value", "rawValue")
_KPI_STEP_TIME_MS = "/tensorflow/training/step_time_ms"
_KPI_PEAK_MEM_MIB = "/tensorflow/memory/peak_per_device_mem_mb"
# megascale_stats reports INT_MAX required bandwidth when slack is zero.
_ZERO_SLACK_BANDWIDTH = 2147483647


def _num(value: Any) -> float | None:
    """Returns a finite float for a JSON number or numeric string, else None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_json_payload(payload: Any) -> Any:
    """Loads a JSON payload from a file path, JSON string/bytes, or Python object."""
    if isinstance(payload, pathlib.Path):
        return json.loads(payload.read_text(encoding="utf-8"))
    if isinstance(payload, bytes):
        return json.loads(payload.decode("utf-8"))
    if isinstance(payload, str):
        stripped = payload.strip()
        if not stripped:
            raise ValueError("Empty JSON payload.")
        if os.path.isfile(stripped):
            with open(stripped, encoding="utf-8") as f:
                return json.load(f)
        return json.loads(stripped)
    return payload


def _traverse_and_collect_ops_dict(
    node: Mapping[str, Any],
    collected_ops: list[dict[str, Any]],
    level: int,
) -> None:
    """Traverses an XProf op-profile node tree and collects ops with occurrences > 0."""
    metrics = node.get("metrics") or {}
    xla_info = node.get("xla") or {}
    if isinstance(metrics, Mapping) and isinstance(xla_info, Mapping):
        if (_num(metrics.get("occurrences")) or 0) > 0 and xla_info:
            source_info = xla_info.get("sourceInfo") or {}
            collected_ops.append({
                "operation":
                str(node.get("name", "N/A")),
                "category":
                str(xla_info.get("category", node.get("name", "N/A"))),
                "expression":
                str(xla_info.get("expression", "N/A")),
                "provenance":
                str(xla_info.get("provenance", "N/A")),
                "raw_time_ps":
                float(_num(metrics.get("rawTime")) or 0.0),
                "occurrences":
                int(_num(metrics.get("occurrences")) or 0),
                "avg_time_ps":
                float(_num(metrics.get("avgTimePs")) or 0.0),
                "flops":
                float(_num(metrics.get("flops")) or 0.0),
                "source_file":
                str(
                    source_info.get("fileName", "N/A") if isinstance(
                        source_info, Mapping) else "N/A"),
                "source_line":
                int((_num(source_info.get("lineNumber")) or 0
                     ) if isinstance(source_info, Mapping) else 0),
                "rawBytesAccessedArray":
                metrics.get("rawBytesAccessedArray"),
                "bandwidthUtils":
                metrics.get("bandwidthUtils"),
            })
    if level > 0:
        for child in node.get("children") or []:
            if isinstance(child, Mapping):
                _traverse_and_collect_ops_dict(child, collected_ops, level - 1)


def _normalize_flat_op(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalizes a pre-flattened op record into the `collective_breakdown` schema."""
    raw_time = _num(record.get("raw_time_ps"))
    if raw_time is None:
        raw_us = _num(record.get("total_self_time_us")) or _num(
            record.get("self_time_us"))
        raw_time = (raw_us * 1e6) if raw_us is not None else 0.0
    occurrences = int(_num(record.get("occurrences")) or 1)
    avg_time = _num(record.get("avg_time_ps"))
    if avg_time is None:
        avg_time = raw_time / max(occurrences, 1)
    op_name = str(record.get("operation") or record.get("name") or "N/A")
    return {
        "operation":
        op_name,
        "category":
        str(record.get("category") or op_name),
        "expression":
        str(record.get("expression") or "N/A"),
        "provenance":
        str(record.get("provenance") or record.get("tf_op_name") or "N/A"),
        "raw_time_ps":
        float(raw_time),
        "occurrences":
        occurrences,
        "avg_time_ps":
        float(avg_time),
        "flops":
        float(_num(record.get("flops")) or 0.0),
        "source_file":
        str(record.get("source_file") or "N/A"),
        "source_line":
        int(_num(record.get("source_line")) or 0),
        "rawBytesAccessedArray":
        record.get("rawBytesAccessedArray"),
        "bandwidthUtils":
        record.get("bandwidthUtils"),
    }


def parse_hlo_op_profile(payload: Any, level: int = 2) -> list[dict[str, Any]]:
    """Parses an XProf `hlo_op_profile` / `op_profile` JSON payload or flat op list.

    Accepts:
    - Raw XProf `{"byCategory": {...}}` tree JSON;
    - A single XProf category/root node dict (`{"name": ..., "children": [...]}`);
    - A list of XProf category nodes; or
    - A pre-flattened list of op dicts (or `{"ops": [...]}`).
    """
    data = _load_json_payload(payload)
    if isinstance(data, Mapping) and isinstance(data.get("ops"), list):
        data = data["ops"]

    if isinstance(data, list):
        if not data:
            return []
        first = data[0]
        if isinstance(first, Mapping) and ("operation" in first
                                           or "raw_time_ps" in first
                                           or "total_self_time_us" in first
                                           or "self_time_us" in first):
            return [
                _normalize_flat_op(item) for item in data
                if isinstance(item, Mapping)
            ]
        collected: list[dict[str, Any]] = []
        for node in data:
            if isinstance(node, Mapping):
                _traverse_and_collect_ops_dict(node, collected, level=level)
        return collected

    if isinstance(data, Mapping):
        collected = []
        if isinstance(data.get("byCategory"), Mapping):
            _traverse_and_collect_ops_dict(data["byCategory"],
                                           collected,
                                           level=level)
            return collected
        if data.get("children") or data.get("name"):
            _traverse_and_collect_ops_dict(data, collected, level=level)
            return collected

    raise ValueError("Unsupported HLO op profile JSON format.")


def parse_memory_profile(payload: Any) -> dict[str, float]:
    """Parses `memory_profile.json` into stats (`capacity`, `peak`, `heap`, `stack`, `free`, `fragmentation`) of the max-peak device."""
    data = _load_json_payload(payload)
    items = data if isinstance(data, list) else [data]
    first = items[0] if items and isinstance(items[0], Mapping) else {}
    if "memoryProfileSummary" in first:
        summaries = [
            d.get("memoryProfileSummary") or {} for d in items
            if isinstance(d, Mapping)
        ]
        peak_key = "peakBytesUsageHbm"
    elif "memoryProfilePerAllocator" in first:
        allocators = first.get("memoryProfilePerAllocator") or {}
        summaries = [
            allocator.get("profileSummary") or {}
            for name, allocator in allocators.items() if
            isinstance(allocator, Mapping) and "host" not in str(name).lower()
        ]
        peak_key = "peakBytesInUse"
    else:
        raise ValueError("Unknown memory profile JSON format.")

    stats: dict[str, float] = {}
    for summary in summaries:
        if not isinstance(summary, Mapping):
            continue
        peak_stats = summary.get("peakStats") or {}
        if not isinstance(peak_stats, Mapping):
            peak_stats = {}
        capacity = _num(summary.get("memoryCapacity")) or 0.0
        if capacity > stats.get("capacity", 0.0):
            stats["capacity"] = capacity
        peak = max(
            _num(peak_stats.get(peak_key)) or 0.0,
            _num(summary.get("peakBytesUsageLifetime")) or 0.0,
        )
        if peak > stats.get("peak", 0.0):
            stats["peak"] = peak
            for key, field in (
                ("heap", "heapAllocatedBytes"),
                ("stack", "stackReservedBytes"),
                ("free", "freeMemoryBytes"),
                ("fragmentation", "fragmentation"),
            ):
                value = _num(peak_stats.get(field))
                if value is None:
                    stats.pop(key, None)
                else:
                    stats[key] = value
    return stats


def _top_buffers(data: Mapping[str, Any], top_k: int) -> list[dict[str, Any]]:
    """Returns the `top_k` largest buffers live at a module's memory peak."""
    buffers: list[dict[str, Any]] = []
    heap = data.get("maxHeapBySize") or data.get("maxHeap") or []
    if isinstance(heap, Sequence):
        for obj in heap:
            if not isinstance(obj, Mapping):
                continue
            size = _num(obj.get("logicalBufferSizeMib"))
            if size is not None:
                buffers.append({
                    "name":
                    obj.get("instructionName") or obj.get("label"),
                    "size_mib":
                    round(size, 2),
                    "shape":
                    obj.get("shapeString") or obj.get("shape"),
                    "op_code":
                    obj.get("opCode"),
                    "tf_op_name":
                    str(obj.get("tfOpName") or "")[:160] or None,
                })
    if not buffers:
        buf_assign = data.get("bufferAssignment")
        logical = (buf_assign.get("logicalBuffers", []) if isinstance(
            buf_assign, Mapping) else [])
        for buf in logical:
            if not isinstance(buf, Mapping):
                continue
            size = _num(buf.get("size"))
            if size is not None:
                defined_at = buf.get("definedAt")
                buffers.append({
                    "name": (defined_at.get("instructionName") if isinstance(
                        defined_at, Mapping) else None),
                    "size_mib":
                    round(size / _MIB, 2),
                    "shape":
                    None,
                    "op_code":
                    None,
                    "tf_op_name":
                    None,
                })
    return sorted(buffers, key=lambda b: b["size_mib"], reverse=True)[:top_k]


def parse_memory_viewer_module(payload: Any,
                               top_k: int = 10) -> dict[str, Any]:
    """Parses a per-module `memory_viewer.json` payload (sizes in MiB)."""
    data = _load_json_payload(payload)
    if not isinstance(data, Mapping):
        raise ValueError("Expected JSON object for memory_viewer module.")
    total = _num(data.get("totalBufferAllocationMib"))
    peak_heap = _num(data.get("peakHeapMib"))
    return {
        "module": data.get("moduleName") or data.get("module"),
        "size": total or peak_heap or 0.0,
        "module_total_mib": total,
        "peak_heap_mib": peak_heap,
        "parameters_mib": _num(data.get("entryComputationParametersMib")),
        "top_buffers": _top_buffers(data, top_k),
    }


def is_serving_modules(modules: Sequence[str]) -> bool:
    """True when module names match known vLLM / serving step programs."""
    for module in modules:
        name = str(module).split("(", 1)[0].strip()
        if name and _SERVING_MODULE_RE.search(name):
            return True
    return False


def parse_kpi_metrics(payload: Any) -> dict[str, float]:
    """Parses `kpi_metrics.json` (`metricsPerSession`) into `{metric_name: float}`."""
    data = _load_json_payload(payload)
    if not isinstance(data, Mapping):
        return {}
    metrics = data.get("metricsPerSession")
    if not isinstance(metrics, Sequence):
        return {
            str(k): v
            for k, raw in data.items() if (v := _num(raw)) is not None
        }
    values: dict[str, float] = {}
    for metric in metrics:
        if not isinstance(metric, Mapping) or not metric.get("name"):
            continue
        value = next((_num(metric[k]) for k in _KPI_VALUE_KEYS if k in metric),
                     None)
        if value is None and isinstance(metric.get("pctValue"), Mapping):
            value = _num(metric["pctValue"].get("mean"))
        if value is not None:
            values[str(metric["name"])] = value
    return values


def summarize_memory_headroom(
    memory_profile: Any = None,
    *,
    memory_viewer_module: Any = None,
    memory_viewer_modules: Sequence[Any] | None = None,
    module_names: Sequence[str] = (),
    kpi_metrics: Any = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Summarizes per-device HBM headroom and resident parameter share.

    When serving modules are detected (`is_serving=True`), `peak_fraction` and
    `parameter_fraction` are returned as `None` (with `reserved_fraction` set)
    so static KV-cache pre-reservations are never mistaken for training
    activation/weight peaks.
    """
    notes: list[str] = []
    stats: dict[str, float] = {}
    source: str | None = None

    if memory_profile is not None:
        stats = parse_memory_profile(memory_profile)
        if stats:
            source = "memory_profile"

    kpis = parse_kpi_metrics(kpi_metrics) if kpi_metrics is not None else {}
    if "peak" not in stats:
        kpi_peak_mib = kpis.get(_KPI_PEAK_MEM_MIB)
        if kpi_peak_mib:
            stats["peak"] = kpi_peak_mib * _MIB
            source = "kpi"
            notes.append(f"Peak from KPI {_KPI_PEAK_MEM_MIB} (no heap/stack).")

    parsed_modules: list[dict[str, Any]] = []
    if memory_viewer_module is not None:
        parsed_modules.append(
            parse_memory_viewer_module(memory_viewer_module, top_k=top_k))
    for mod_payload in memory_viewer_modules or ():
        parsed_modules.append(
            parse_memory_viewer_module(mod_payload, top_k=top_k))

    best_module: dict[str, Any] | None = None
    for mod in parsed_modules:
        size = float(mod.get("size") or 0.0)
        if size >= 1.0 and (best_module is None
                            or size > float(best_module.get("size") or 0.0)):
            best_module = mod

    inferred_names = list(module_names)
    for mod in parsed_modules:
        if mod.get("module"):
            inferred_names.append(str(mod["module"]))
    is_serving = is_serving_modules(inferred_names)

    if not stats and best_module is None:
        raise ValueError("No usable memory_profile or memory_viewer data.")

    peak = stats.get("peak")
    capacity = stats.get("capacity")
    heap = stats.get("heap")
    if not (peak and capacity):
        notes.append("HBM capacity or peak unknown: no peak_fraction.")
    raw_fraction = round(peak / capacity, 4) if peak and capacity else None
    if is_serving:
        notes.append(
            "Serving session detected: memory_profile peak reflects static"
            " KV-cache pre-reservation, not training activation/weight peak.")

    result: dict[str, Any] = {
        "capacity_bytes": int(capacity) if capacity else None,
        "peak_bytes": int(peak) if peak else None,
        "peak_fraction": None if is_serving else raw_fraction,
        "is_serving": is_serving,
        "reserved_fraction": raw_fraction if is_serving else None,
    }
    for key in ("capacity", "peak", "heap", "stack", "free"):
        result[f"{key}_gib"] = (round(stats[key] /
                                      _GIB, 3) if key in stats else None)
    fragmentation = stats.get("fragmentation")
    result.update({
        "fragmentation_pct":
        (None if fragmentation is None else round(fragmentation * 100, 4)),
        "source":
        source,
        "module":
        None,
        "module_total_mib":
        None,
        "peak_heap_mib":
        None,
        "parameters_mib":
        None,
        "parameter_fraction":
        None,
        "parameter_fraction_basis":
        None,
        "top_buffers": [],
        "notes":
        notes,
    })

    fraction: float | None = None
    basis: str | None = None
    if best_module is not None:
        result.update({
            k: round(v, 2) if isinstance(v, float) else v
            for k, v in best_module.items() if k != "size"
        })
        params = best_module.get("parameters_mib")
        module_peak = best_module.get("peak_heap_mib") or (peak / _MIB
                                                           if peak else None)
        if params is not None and module_peak:
            fraction, basis = float(params) / float(
                module_peak), "module_params"
    if fraction is None and heap is not None and peak:
        fraction, basis = heap / peak, "device_heap"
    if fraction is not None and not is_serving:
        result["parameter_fraction"] = round(min(max(fraction, 0.0), 1.0), 4)
        result["parameter_fraction_basis"] = basis
    return result


def parse_megascale_stats(payload: Any) -> list[dict[str, Any]]:
    """Parses `megascale_stats.json` DataTable into per-collective totals."""
    data = _load_json_payload(payload)
    table = (data[0] if data else {}) if isinstance(data, list) else data
    if not isinstance(table, Mapping):
        raise ValueError("megascale_stats.json is not a JSON object or list.")
    cols = {
        c.get("id"): i
        for i, c in enumerate(table.get("cols") or [])
        if isinstance(c, Mapping)
    }
    if "rendezvous_name" not in cols or "occurrences" not in cols:
        raise ValueError("megascale_stats.json is not the expected DataTable.")
    collectives: list[dict[str, Any]] = []
    for row in table.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        cells = row.get("c") or []
        value = {
            c: (cells[i] or {}).get("v")
            for c, i in cols.items()
            if i < len(cells) and isinstance(cells[i], Mapping | type(None))
        }
        if not value.get("rendezvous_name"):
            continue
        occurrences = int(_num(value.get("occurrences")) or 0)
        stall_ms = _num(value.get("total_stall"))
        avg_stall_ms = _num(value.get("stall_duration"))
        if stall_ms is None and avg_stall_ms is not None:
            stall_ms = avg_stall_ms * occurrences
        tx_bytes = _num(value.get("net_tx_bytes"))
        bandwidth = _num(value.get("required_bandwidth"))
        collectives.append({
            "name":
            str(value["rendezvous_name"]),
            "transfer_type":
            value.get("transfer_type"),
            "occurrences":
            occurrences,
            "observed_us_total":
            ((_num(value.get("observed_duration")) or 0.0) * 1e3 *
             occurrences),
            "stall_us_total":
            None if stall_ms is None else stall_ms * 1e3,
            "slack_us_avg": (_num(value.get("slack_time")) or 0.0) * 1e3,
            "bytes":
            None if tx_bytes is None else int(tx_bytes),
            "required_bandwidth_gbps":
            (bandwidth if (bandwidth or 0) < _ZERO_SLACK_BANDWIDTH else None),
        })
    return collectives


def parse_overview_page(payload: Any) -> dict[str, Any]:
    """Parses `overview_page.json` into `step_time_ms`, `num_steps`, and `profile_duration_ms`."""
    data = _load_json_payload(payload)
    if isinstance(data, Mapping) and ("step_time_ms" in data
                                      or "profile_duration_ms" in data):
        step_ms = _num(data.get("step_time_ms"))
        num_steps_raw = _num(data.get("num_steps"))
        duration_ms = _num(data.get("profile_duration_ms"))
        return {
            "step_time_ms": step_ms if step_ms else None,
            "num_steps": int(num_steps_raw) if num_steps_raw else None,
            "profile_duration_ms": duration_ms if duration_ms else None,
        }

    tables = data if isinstance(data, list) else [data]
    step_ms: float | None = None
    num_steps: int | None = None
    duration_ms: float | None = None
    for table in tables:
        if not isinstance(table, Mapping):
            continue
        props = table.get("p") or {}
        if not isinstance(props, Mapping):
            props = {}
        average = _num(props.get("steptime_ms_average"))
        cols = {
            c.get("id")
            for c in (table.get("cols") or []) if isinstance(c, Mapping)
        }
        if "stepnum" in cols and average and step_ms is None:
            step_ms = average
            num_steps = len(table.get("rows") or []) or None
        duration_ms = duration_ms or _num(props.get("profile_duration_ms"))
    return {
        "step_time_ms": step_ms,
        "num_steps": num_steps,
        "profile_duration_ms": duration_ms or None,
    }


def summarize_dcn_stats(
    megascale_stats: Any = None,
    *,
    overview_page: Any = None,
    kpi_metrics: Any = None,
    num_hosts: int = 1,
) -> dict[str, Any]:
    """Summarizes cross-slice (DCN / megascale) collectives and exposed step-time share."""
    notes: list[str] = []
    result: dict[str, Any] = {
        "available": False,
        "num_hosts": None,
        "num_collectives": 0,
        "collectives": [],
        "total_observed_us": None,
        "total_stall_us": None,
        "step_time_ms": None,
        "num_steps": None,
        "dcn_pct": None,
        "dcn_pct_basis": None,
        "notes": notes,
    }
    if megascale_stats is None:
        notes.append("No megascale_stats payload provided.")
        return result

    collectives = parse_megascale_stats(megascale_stats)
    if not collectives:
        notes.append(
            "No megascale (DCN) collectives: expected for single-slice"
            " profiles; otherwise megascale tracing was not captured.")
        return result

    hosts = max(int(num_hosts or 1), 1)
    collectives.sort(
        key=lambda c: (c["stall_us_total"] or 0.0, c["observed_us_total"]),
        reverse=True,
    )
    total_observed = sum(c["observed_us_total"] for c in collectives)
    total_stall: float | None = None
    if any(c["stall_us_total"] is not None for c in collectives):
        total_stall = sum(c["stall_us_total"] or 0.0 for c in collectives)

    overview = (parse_overview_page(overview_page)
                if overview_page is not None else {})
    kpis = parse_kpi_metrics(kpi_metrics) if kpi_metrics is not None else {}
    step_ms = overview.get("step_time_ms") or kpis.get(_KPI_STEP_TIME_MS)
    num_steps = overview.get("num_steps")
    duration_ms = overview.get("profile_duration_ms")

    result.update({
        "available":
        True,
        "num_hosts":
        hosts,
        "num_collectives":
        len(collectives),
        "collectives": [{
            k: round(v, 3) if isinstance(v, float) else v
            for k, v in c.items()
        } for c in collectives[:20]],
        "total_observed_us":
        round(total_observed, 1),
        "total_stall_us":
        None if total_stall is None else round(total_stall, 1),
        "step_time_ms":
        step_ms,
        "num_steps":
        num_steps,
    })

    exposed_us, basis = total_stall, "stall"
    if exposed_us is None:
        exposed_us, basis = total_observed, "observed"
        notes.append("No stall data; using observed time (upper bound).")
    if step_ms and num_steps:
        window_ms, denominator = float(step_ms) * int(
            num_steps), "num_steps*step_time"
    elif duration_ms:
        window_ms, denominator = float(duration_ms), "profile_duration"
        notes.append(
            "No step count/time: dcn_pct is relative to the profile duration (a"
            " lower bound on the step-time share).")
    else:
        notes.append("No step time or profile duration: no dcn_pct.")
        return result

    dcn_pct = 100.0 * float(exposed_us) / hosts / (window_ms * 1e3)
    if dcn_pct > 100.0:
        notes.append(f"dcn_pct clamped from {dcn_pct:.1f}% to 100%.")
        dcn_pct = 100.0
    result["dcn_pct"] = round(dcn_pct, 3)
    result["dcn_pct_basis"] = f"{basis}/({denominator})"
    return result
