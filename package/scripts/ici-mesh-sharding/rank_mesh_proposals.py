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
"""Builds a ranked ICI mesh portfolio from local XProf JSON exports.

File-based glue between `xprof_parsers.py`, `collective_breakdown.py`, and
`mesh_portfolio.py`. Memory, DCN (megascale), and architecture JSON inputs are
optional: when omitted or unreadable, the portfolio is still built from the HLO
op profile and the missing evidence is noted in `result["notes"]`.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import collective_breakdown  # noqa: E402
import mesh_portfolio  # noqa: E402
import mesh_search  # noqa: E402
import xprof_parsers  # noqa: E402

DEFAULT_NUM_PROPOSALS = mesh_portfolio.DEFAULT_NUM_PROPOSALS

# Axes of vLLM / OSS serving meshes. A baseline made only of these is a serving
# mesh, which should be validated via `oss_sharding_rules` rather than the
# trainer communication-volume portfolio.
SERVING_AXES = frozenset({"tp", "dp", "ep", "sp", "pcp", "dcp", "pp"})

_EVIDENCE_ERRORS = (
    ValueError,
    KeyError,
    LookupError,
    TypeError,
    AttributeError,
    ArithmeticError,
    OSError,
    RuntimeError,
)


def _axes(text: str | Sequence[str] | None) -> list[str]:
    if not text:
        return []
    if isinstance(text, str):
        return [axis.strip() for axis in text.split(",") if axis.strip()]
    return [str(axis).strip() for axis in text if str(axis).strip()]


def _load_mapping(payload: Any) -> dict[str, Any]:
    if isinstance(payload, pathlib.Path):
        data = json.loads(payload.read_text(encoding="utf-8"))
    elif isinstance(payload, str):
        stripped = payload.strip()
        if os.path.isfile(stripped):
            with open(stripped, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(stripped)
    elif isinstance(payload, Mapping):
        data = dict(payload)
    else:
        raise ValueError(
            f"Expected JSON mapping or path, got {type(payload).__name__}.")
    if not isinstance(data, Mapping):
        raise ValueError("Expected top-level JSON object.")
    return dict(data)


def memory_evidence(
    memory_profile: Any = None,
    *,
    memory_viewer_module: Any = None,
    module_names: Sequence[str] = (),
    kpi_metrics: Any = None,
) -> tuple[dict[str, float] | None, str]:
    """Returns `({'peak_fraction', 'parameter_fraction'} | None, summary_note)`."""
    if (isinstance(memory_profile, Mapping)
            and "peak_fraction" in memory_profile
            and "memoryProfileSummary" not in memory_profile
            and "memoryProfilePerAllocator" not in memory_profile):
        headroom = dict(memory_profile)
    else:
        try:
            headroom = xprof_parsers.summarize_memory_headroom(
                memory_profile,
                memory_viewer_module=memory_viewer_module,
                module_names=module_names,
                kpi_metrics=kpi_metrics,
            )
        except _EVIDENCE_ERRORS as e:
            return None, f"memory profile unavailable ({type(e).__name__})"

    if headroom.get("is_serving"):
        reserved = headroom.get("reserved_fraction")
        pct = f" ({100 * float(reserved):.0f}% of HBM)" if reserved else ""
        return (
            None,
            (f"memory profile is a serving KV-cache pre-reservation{pct};"
             " skipped for training headroom"),
        )
    peak = headroom.get("peak_fraction")
    if not peak:
        return None, "memory profile has no peak / capacity"
    params = headroom.get("parameter_fraction")
    note = f"memory: peak {100 * float(peak):.0f}% of HBM"
    if params is not None:
        note += f", resident buffers {100 * float(params):.0f}% of peak"
    evidence: dict[str, float] = {"peak_fraction": float(peak)}
    if params is not None:
        evidence["parameter_fraction"] = float(params)
    return evidence, note


def dcn_evidence(
    megascale_stats: Any = None,
    *,
    overview_page: Any = None,
    kpi_metrics: Any = None,
    num_hosts: int = 1,
) -> tuple[float, str]:
    """Returns `(exposed_dcn_pct, summary_note)`."""
    try:
        stats = xprof_parsers.summarize_dcn_stats(
            megascale_stats,
            overview_page=overview_page,
            kpi_metrics=kpi_metrics,
            num_hosts=num_hosts,
        )
    except _EVIDENCE_ERRORS as e:
        return 0.0, f"DCN stats unavailable ({type(e).__name__})"
    pct = stats.get("dcn_pct")
    if not stats.get("available") or pct is None:
        return 0.0, "no DCN (megascale) timing in the profile"
    return float(pct), f"DCN: {float(pct):.1f}% of the step"


def arch_evidence(architecture: Any) -> tuple[dict[str, Any] | None, str]:
    """Loads optional architecture metadata (`token_axes`, `fsdp_axes`, `partition_config`, `moe`)."""
    try:
        arch = _load_mapping(architecture)
    except _EVIDENCE_ERRORS as e:
        return None, f"architecture metadata unavailable ({type(e).__name__})"
    if not arch:
        return None, "architecture metadata returned no fields"
    return arch, ""


def rank_mesh_proposals(
    ops: Any,
    mesh: str | Mapping[str, Any],
    *,
    memory_profile: Any = None,
    memory_viewer_module: Any = None,
    module_names: str | Sequence[str] | None = None,
    megascale_stats: Any = None,
    overview_page: Any = None,
    kpi_metrics: Any = None,
    num_hosts: int = 1,
    dcn_pct: float | None = None,
    architecture: Any = None,
    topology: str | None = None,
    num_slices: int = 1,
    batch_size: int | None = None,
    sequence_length: int | None = None,
    num_heads: int | None = None,
    num_kv_heads: int | None = None,
    extra_axes: str | Sequence[str] | None = None,
    fixed_axes: str | Sequence[str] | None = None,
    token_axes: str | Sequence[str] | None = None,
    fsdp_axes: str | Sequence[str] | None = None,
    partition_config: Mapping[str, Any] | None = None,
    allow_reshape: bool = False,
    allow_slice_change: bool = False,
    num_proposals: int = DEFAULT_NUM_PROPOSALS,
) -> dict[str, Any]:
    """Parses local profile evidence and builds the ranked 7-proposal portfolio."""
    baseline = mesh_search.parse_mesh(mesh)
    if baseline and set(baseline) <= SERVING_AXES:
        raise ValueError(
            "rank_mesh_proposals ranks trainer meshes; for a vLLM / OSS serving"
            f" mesh ({', '.join(sorted(baseline))}) use `validate-model` and"
            " follow references/ici-mesh-sharding/proposal.md instead")

    parsed_ops = xprof_parsers.parse_hlo_op_profile(ops)
    breakdown = collective_breakdown.summarize(parsed_ops,
                                               baseline,
                                               top_k=0,
                                               sequence_length=sequence_length
                                               or None)
    notes: list[str] = []
    if breakdown.get("mesh_mismatch"):
        notes.append(
            "the profile's replica groups imply other axis sizes than --mesh;"
            " check that the profile matches the baseline mesh")

    memory = None
    if (memory_profile is not None or memory_viewer_module is not None
            or kpi_metrics is not None):
        memory, mem_note = memory_evidence(
            memory_profile,
            memory_viewer_module=memory_viewer_module,
            module_names=_axes(module_names),
            kpi_metrics=kpi_metrics,
        )
        notes.append(mem_note)

    effective_dcn_pct = float(dcn_pct) if dcn_pct is not None else 0.0
    if dcn_pct is None and megascale_stats is not None and num_slices > 1:
        effective_dcn_pct, dcn_note = dcn_evidence(
            megascale_stats,
            overview_page=overview_page,
            kpi_metrics=kpi_metrics,
            num_hosts=num_hosts,
        )
        notes.append(dcn_note)

    if architecture is not None:
        arch, arch_note = arch_evidence(architecture)
        if arch_note:
            notes.append(arch_note)
        arch = arch or {}
        if not partition_config and isinstance(arch.get("partition_config"),
                                               Mapping):
            partition_config = dict(arch["partition_config"])
        if not token_axes and arch.get("token_axes"):
            token_axes = [
                a for a in arch["token_axes"]
                if a not in mesh_search.NON_ICI_AXES
            ]
        if not fsdp_axes and arch.get("fsdp_axes"):
            fsdp_axes = [
                a for a in arch["fsdp_axes"]
                if a not in mesh_search.NON_ICI_AXES
            ]
        gen_mesh = arch.get("generate_ici_mesh")
        moe_cfg = (dict(arch["moe"])
                   if isinstance(arch.get("moe"), Mapping) else {})
        stage_experts = moe_cfg.get("batch_pipeline_stage_experts")
        if isinstance(gen_mesh, Mapping) or stage_experts:
            parts = []
            if isinstance(gen_mesh, Mapping):
                parts.append(f"generate_ici_mesh={dict(gen_mesh)}")
            if stage_experts:
                counts = tuple(int(x) for x in stage_experts)
                parts.append(
                    f"batch_pipeline_stage_experts={counts} (per-stage expert"
                    " counts; they must sum to the routed experts per expert"
                    f" shard, {sum(counts)} for this mesh, so re-derive them"
                    " for a proposal that changes the number of expert shards)"
                )
            notes.insert(0, "MPMD serving config: " + "; ".join(parts))
        elif moe_cfg.get("routing_group"):
            rg = list(moe_cfg["routing_group"])
            ew = list(moe_cfg.get("expert_weight_axes") or rg)
            notes.append(
                f"MoE architecture: routing_group={rg}, expert_weight_axes={ew}"
            )

    tensor_ar = breakdown.get("tensor_ar") or {}
    if tensor_ar.get("pct", 0.0) >= collective_breakdown.MIN_PRINTED_PCT:
        axis_label = ", ".join(tensor_ar.get("axes") or ["model"])
        parts = [
            f"{pct:.2f}% {kind}"
            for kind, pct in (tensor_ar.get("by_kind") or {}).items()
        ]
        notes.append(
            f"Tensor-axis ({axis_label}) collectives: {tensor_ar['pct']:.2f}%"
            f" of profiled op time ({' + '.join(parts)})")

    fixed = _axes(fixed_axes) or list(mesh_search.DEFAULT_FIXED_AXES)
    result = mesh_portfolio.build_portfolio(
        baseline,
        breakdown,
        topology=topology or None,
        num_slices=num_slices or 1,
        batch_size=batch_size or None,
        sequence_length=sequence_length or None,
        num_heads=num_heads or None,
        num_kv_heads=num_kv_heads or None,
        extra_axes=_axes(extra_axes),
        fixed_axes=fixed,
        token_axes=_axes(token_axes),
        fsdp_axes=_axes(fsdp_axes),
        partition_config=partition_config,
        memory=memory,
        dcn_pct=effective_dcn_pct,
        allow_reshape=allow_reshape,
        allow_slice_change=allow_slice_change,
        num_proposals=num_proposals,
    )
    result["notes"] = notes + list(result.get("notes") or [])
    return result


def format_result(result: Mapping[str, Any],
                  output_format: str = "table") -> str:
    """Renders the portfolio as `'table'` (Markdown) or `'json'`."""
    if output_format == "json":
        return json.dumps(result, indent=2)
    return mesh_portfolio.format_portfolio_markdown(result)
