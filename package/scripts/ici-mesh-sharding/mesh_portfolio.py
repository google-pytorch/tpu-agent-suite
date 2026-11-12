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
"""Ranks a diverse, evidence-driven portfolio of ICI mesh proposals.

The baseline profile's collective breakdown (`collective_breakdown.py`) says
which collectives cost step time, what they are for (tensor-parallel
activations, KV gathers, MoE routing, FSDP weights, ...) and which mesh axes
they span. This module turns that evidence into bottleneck hypotheses, lists
the legal meshes that address each one (`mesh_search`), estimates every
candidate's step-time change analytically -- no simulation -- and fills the
portfolio from a fixed, explainable template:

1.  the strongest hypothesis' best move at its first step (k = 2) and at its
    boundary (the largest k that passes the memory check), because the best
    magnitude is usually at one end of a move's range;
2.  the same bottleneck relieved through another donor or receiver (e.g. the
    other axis of an MoE routing group), at its first step;
3.  the second hypothesis' best move at its first step;
4.  placement: re-factoring sibling axes that a bottleneck spans (size swap or
    first step towards the primary axis), the best re-shaped slice, fewer,
    larger slices, or the first step out of a movable pipeline axis -- only
    those the evidence or the prompt supports;
5.  the best compound remedy (one receiver fed by two donors, one donor split
    across two receivers, or a move plus a sibling fold), then the third
    hypothesis.

No mesh is repeated. Each proposal carries its justification: the evidence,
the collectives it shrinks or grows, the memory arithmetic and the estimate.

The estimate scales each collective group's share of the step with its
per-device communication volume:

*   tensor-parallel activation collectives ~ local tokens x (t - 1) / t over
    the tensor shards t (they vanish at t = 1);
*   KV gathers and ring attention ~ per-device batch x KV heads per device x
    (s - 1) / s over the group (the full sequence is gathered whatever the
    shard count);
*   MoE routing (all-to-all) ~ local tokens x (g - 1) / g x mean hop count
    of the group on the torus (all-to-all traffic crosses the group, so it
    grows with the group's extent, unlike ring collectives);
*   other activation collectives ~ local tokens x (g - 1) / g;
*   FSDP weight gathers and gradient reductions ~ 1 / (tensor x expert
    shards) x (g - 1) / g;
*   cross-slice (DCN) reductions ~ (r - 1) / r over the slices r, divided by
    the growth of the per-slice mesh;

each multiplied by 2 when the group's axes do not fill whole torus dimensions
on their own (no dedicated ring or plane). That factor of 2
(`_SHARED_DIM_PENALTY`) and the hop counts (L / 4 per dimension of length L,
the mean on a ring with wrap-around links; see `_hops`) are model assumptions,
not measurements. Global reductions, host
transfers and compute are held constant; tokens are sharded by the batch,
sequence and expert axes unless the caller passes the trainer's token axes.
The numbers rank candidates; they are not a prediction of the new step time.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import collective_breakdown  # noqa: E402
import mesh_search  # noqa: E402
import mesh_topology  # noqa: E402

# Bottleneck hypotheses. The first group shrinks communication volume, the
# second changes placement.
TENSOR = "tensor"
SEQUENCE = "sequence"
ROUTING = "routing"
WEIGHTS = "weights"
BATCH = "batch"
MIXED = "mixed"
LAYOUT = "layout"
SLICES = "slices"
PIPELINE = "pipeline"
# Evidence that no mesh move relieves.
GLOBAL = "global"
HOST = "host"
UNKNOWN = "unknown"

VOLUME_HYPOTHESES = (TENSOR, SEQUENCE, ROUTING, WEIGHTS, BATCH, MIXED)
HYPOTHESIS_TITLES = {
    TENSOR: "tensor-parallel activation collectives",
    SEQUENCE: "sequence-parallel KV gathers / ring attention",
    ROUTING: "MoE routing (all-to-all) group",
    WEIGHTS: "FSDP weight gathers / gradient reductions",
    BATCH: "batch-axis activation collectives",
    MIXED: "activation collectives over several axis families",
    LAYOUT: "torus placement",
    SLICES: "cross-slice (DCN) traffic",
    PIPELINE: "pipeline depth (stages)",
}

DEFAULT_NUM_PROPOSALS = 7
PIPELINE_AXES = ("stage", "pp")
MIN_HYPOTHESIS_PCT = 2.0
_PRIOR_PCT = 3.0
_SHORT_CONTEXT = 16384
_LONG_CONTEXT = 65536
_SHARED_DIM_PENALTY = 2.0
_RESHAPE_MIN_GAIN_PCT = 1.0
_REGRESSION_PCT = 0.5
_DIM_NAMES = "XYZ"
_MEMORY_RISK_RATIO = 8.0
_MEMORY_CAUTION_RATIO = 4.0
_OOM_FRACTION = 0.95

_SEQUENCE_SOURCE_RE = re.compile(
    r"attention|context_parallel|ring|flash|splash", re.I)
_TENSOR_SOURCE_RE = re.compile(r"megablox|matmul|mlp|ffn|dense", re.I)
_ROUTING_SOURCE_RE = re.compile(r"moe|expert|routed|dispatch|combine", re.I)
_MOVE_RE = re.compile(r"(\w+)/(\d+)\s*->\s*(\w+)\s*x(\d+)")
_ROLE_HYPOTHESIS = {
    collective_breakdown.ROLE_ROUTING: ROUTING,
    collective_breakdown.ROLE_KV: SEQUENCE,
    collective_breakdown.ROLE_RING: SEQUENCE,
    collective_breakdown.ROLE_MATMUL: TENSOR,
    collective_breakdown.ROLE_WEIGHTS: WEIGHTS,
    collective_breakdown.ROLE_GLOBAL: GLOBAL,
    collective_breakdown.ROLE_HOST: HOST,
}
_FAMILY_HYPOTHESIS = {
    "tensor": TENSOR,
    "sequence": SEQUENCE,
    "batch": BATCH,
    "expert": ROUTING,
}


@dataclasses.dataclass(frozen=True)
class Move:
    """Divides `donor` by `factor` and multiplies `receiver` by `factor`."""

    donor: str
    receiver: str
    factor: int

    def __str__(self) -> str:
        return f"{self.donor}/{self.factor} -> {self.receiver} x{self.factor}"


@dataclasses.dataclass(frozen=True)
class EvidenceGroup:
    """One group of baseline collectives, as evidence for a hypothesis."""

    pct: float
    hypothesis: str
    axes: tuple[str, ...]
    label: str


@dataclasses.dataclass
class Hypothesis:
    """A bottleneck explanation with its share of the step and its evidence."""

    name: str
    score: float
    axes: tuple[str, ...] = ()
    evidence: list[str] = dataclasses.field(default_factory=list)
    slots: int = 0


@dataclasses.dataclass
class Candidate:
    """A proposed mesh (and slice shape / count) with its estimate."""

    mesh: dict[str, int]
    moves: tuple[Move, ...] = ()
    topology: str | None = None
    num_slices: int = 1
    batch_size: int | None = None
    change: str = "move"  # 'move', 'reshape' or 'slices'.
    bracket: str = ""
    warnings: tuple[str, ...] = ()
    gain_pct: float = 0.0
    effects: list[tuple[str, float,
                        float]] = dataclasses.field(default_factory=list)
    local_tokens_ratio: float = 1.0
    weight_bytes_ratio: float = 1.0
    tensor_shards_base: int = 1
    tensor_shards_new: int = 1
    tokens_upper_bound: bool = False
    weights_assumed_fsdp: bool = False
    allow_split_physical_axes: bool | None = None
    memory_fraction: float | None = None
    oom_risk: bool = False
    layout_rank: tuple[int, ...] = ()
    hypothesis: str = ""
    role: str = ""
    current_shape_gain_pct: float | None = None

    def key(self) -> tuple[Any, ...]:
        return (
            tuple(sorted(self.mesh.items())),
            self.topology or "",
            self.num_slices,
        )


@dataclasses.dataclass
class _Context:
    """Baseline, workload and evidence shared by the estimators."""

    base: dict[str, int]
    dims: list[int] | None
    topology: str | None
    num_slices: int
    batch_size: int | None
    sequence_length: int | None
    num_heads: int | None
    num_kv_heads: int | None
    groups: list[EvidenceGroup]
    token_axes: tuple[str, ...]
    fsdp_axes: tuple[str, ...]
    dcn_pct: float
    memory: Mapping[str, Any] | None
    token_axes_known: bool = False
    pipeline_axes: tuple[str, ...] = ()
    kv_excluded: int = 0


def _parse_moves(texts: Sequence[str]) -> tuple[Move, ...]:
    moves = []
    for text in texts:
        match = _MOVE_RE.fullmatch(str(text).strip())
        if match:
            moves.append(
                Move(match.group(1), match.group(3), int(match.group(2))))
    return tuple(moves)


def _shards(mesh: Mapping[str, int], axes: Sequence[str]) -> int:
    return math.prod(int(mesh.get(axis, 1)) for axis in set(axes))


def _ring(n: int) -> float:
    return (n - 1) / n if n > 1 else 0.0


def _ring_ratio(n0: int, n1: int) -> float:
    """(n1 - 1) / n1 over (n0 - 1) / n0; 1 when the group does not change."""
    if n0 == n1 or n0 <= 1:
        return 1.0
    return _ring(n1) / _ring(n0)


def _family(axis: str) -> str:
    return mesh_search.axis_family(axis)


def _active(mesh: Mapping[str, int], axes: Sequence[str]) -> tuple[str, ...]:
    """Axes of `axes` with size > 1 in `mesh` that lie on ICI links."""
    return tuple(
        axis for axis in axes
        if int(mesh.get(axis, 1)) > 1 and axis not in mesh_search.NON_ICI_AXES)


def _alternatives(label: str) -> list[tuple[str, ...]]:
    """Axis tuples of an 'ambiguous: a x b or c x d' label."""
    text = label.split(":", 1)[1] if ":" in label else label
    return [
        tuple(axis.strip() for axis in option.split(" x ") if axis.strip())
        for option in text.split(" or ")
    ]


def evidence_groups(
    breakdown: Mapping[str, Any] | None,
    base: Mapping[str, int],
) -> list[EvidenceGroup]:
    """Maps collective-breakdown groups to hypotheses and axes."""
    if not breakdown:
        return []
    rows = list(breakdown.get("groups") or [])
    all_active = _active(base, list(base))
    routing_axes = collections.Counter(
        _active(base,
                row.get("exact_axes") or ()) for row in rows
        if row.get("role") == collective_breakdown.ROLE_ROUTING
        and row.get("exact_axes") and _active(base,
                                              row.get("exact_axes") or
                                              ()) != all_active)
    default_routing = routing_axes.most_common(1)[0][0] if routing_axes else ()
    out = []
    for row in rows:
        pct = float(row.get("pct", 0.0))
        if pct < 0.1:
            continue
        role = str(row.get("role", ""))
        label = str(row.get("axes", ""))
        kind = str(row.get("kind", ""))
        axes = _active(base, row.get("exact_axes") or ())
        if not axes and label.startswith("ambiguous"):
            options = [
                _active(base, option) for option in _alternatives(label)
            ]
            common = set(options[0]).intersection(
                *options[1:]) if options else set()
            axes = tuple(axis for axis in base if axis in common)
        if label == "all axes (global)" or (len(all_active) > 2
                                            and axes == all_active):
            hypothesis = GLOBAL
        else:
            hypothesis = _ROLE_HYPOTHESIS.get(role)
        if hypothesis == TENSOR:
            axes = _active(base, mesh_search.TENSOR_AXES) or axes
        elif hypothesis == ROUTING and not axes:
            axes = default_routing
        elif hypothesis == SEQUENCE:
            axes = tuple(
                a for a in axes if a in mesh_search.SEQUENCE_AXES) or _active(
                    base, mesh_search.SEQUENCE_AXES)
        elif hypothesis is None:
            source = str(row.get("source", ""))
            families = {_family(axis) for axis in axes} - {"other"}
            if _SEQUENCE_SOURCE_RE.search(source) and (not axes or "sequence"
                                                       in families):
                hypothesis = SEQUENCE
                axes = tuple(a for a in axes
                             if a in mesh_search.SEQUENCE_AXES) or _active(
                                 base, mesh_search.SEQUENCE_AXES)
            elif _ROUTING_SOURCE_RE.search(
                    source) and axes and axes != all_active:
                hypothesis = ROUTING
            elif _TENSOR_SOURCE_RE.search(source):
                hypothesis = TENSOR
                axes = _active(base, mesh_search.TENSOR_AXES) or axes
            elif len(families) == 1:
                hypothesis = _FAMILY_HYPOTHESIS.get(families.pop(), MIXED)
            elif families:
                hypothesis = MIXED
            else:
                hypothesis = UNKNOWN
        if hypothesis in (TENSOR, SEQUENCE) and not axes:
            hypothesis = UNKNOWN
        shown_role = collective_breakdown.display_role(
            row) if role else "other"
        text = f"{kind} ({shown_role}) over {label}"
        source = row.get("source")
        if source and source != "-":
            text += f" in {source}"
        out.append(EvidenceGroup(pct, hypothesis, axes, text))
    return out


def _chip_mesh(mesh: Mapping[str, int],
               dims: Sequence[int] | None) -> dict[str, int]:
    """Inter-chip mesh when 2 TensorCores per chip are folded into an axis."""
    out = {k: int(v) for k, v in mesh.items()}
    if not dims:
        return out
    chips = math.prod(int(d) for d in dims)
    ici = [
        a for a, v in out.items()
        if v > 1 and a not in mesh_search.NON_ICI_AXES
    ]
    product = math.prod(out[a] for a in ici)
    if product == 2 * chips and int(out.get("intrachip", 1)) == 1:
        for axis in ("model", "tp") + tuple(reversed(ici)):
            if int(out.get(axis, 1)) >= 2 and int(out[axis]) % 2 == 0:
                out[axis] //= 2
                break
    return out


@functools.lru_cache(maxsize=65536)
def _group_block_cached(
    mesh_items: tuple[tuple[str, int], ...],
    dims: tuple[int, ...],
    axes: tuple[str, ...],
) -> tuple[int, ...] | None:
    """Physical dims of the block the axes fill together; () if they cannot."""
    mesh = _chip_mesh(dict(mesh_items), dims)
    fit = mesh_search.block_layout(mesh, list(dims), axes)
    if "note" in fit:
        return None
    entry = fit.get("colocate") or {}
    block = entry.get("block")
    if not block:
        return ()
    return tuple(int(side) for side in str(block).split("x"))


def _group_block(mesh: Mapping[str, int], dims: Sequence[int] | None,
                 axes: Sequence[str]) -> tuple[int, ...] | None:
    ici = _active(mesh, axes)
    if not dims or not ici:
        return None
    return _group_block_cached(tuple((k, int(v)) for k, v in mesh.items()),
                               tuple(dims), ici)


def _placement(
    mesh: Mapping[str, int],
    dims: Sequence[int] | None,
    axes: Sequence[str],
    base: Mapping[str, int] | None = None,
) -> float:
    """Placement multiplier of a collective group: 1 or `_SHARED_DIM_PENALTY`."""
    block = _group_block(mesh, dims, axes)
    if block is None:
        return 1.0
    if not block:
        return _SHARED_DIM_PENALTY
    if (base is not None and len(block) == len(dims or ())
            and _active(base, axes) != _active(base, list(base))
            and _active(mesh, axes) == _active(mesh, list(mesh))
            and bool(_needs_split_physical_axes(mesh, dims))):
        return _SHARED_DIM_PENALTY
    return 1.0


def _compact_sides(size: int, dims: Sequence[int]) -> list[int]:
    """Sides of a compact block of `size` chips carved out of `dims`."""
    sides = []
    remaining = size
    for dim in sorted(dims, reverse=True):
        side = math.gcd(remaining, dim)
        if side > 1:
            sides.append(side)
            remaining //= side
    if remaining > 1:
        sides.append(remaining)
    return sides


def _hops(mesh: Mapping[str, int], dims: Sequence[int] | None,
          axes: Sequence[str]) -> float:
    """Mean torus hop count of all-to-all traffic inside the group.

    Each side of length L counts L / 4 hops, the mean distance on a ring with
    wrap-around links. Slices without them (`mesh_search.wraparound_dims`) are
    not modeled separately, so their counts are low: a model assumption, like
    `_SHARED_DIM_PENALTY` for groups that share a dimension.
    """
    chip_mesh = _chip_mesh(mesh, dims)
    ici = _active(chip_mesh, axes)
    size = _shards(chip_mesh, ici)
    if size <= 1:
        return 0.0
    if not dims:
        return size / 4.0
    block = _group_block(chip_mesh, dims, ici)
    if block:
        return sum(side / 4.0 for side in block)
    sides = _compact_sides(size, dims)
    penalty = _SHARED_DIM_PENALTY if block is not None else 1.0
    return penalty * sum(side / 4.0 for side in sides)


def _kv_per_device(mesh: Mapping[str, int], num_kv_heads: int | None) -> float:
    """KV heads per device (replicated beyond the head count; 1 when unknown)."""
    if not num_kv_heads:
        return 1.0
    shards = _shards(
        mesh, [axis for axis in mesh_search.TENSOR_AXES if axis != "model_q"])
    return num_kv_heads / min(shards, num_kv_heads)


def _dims_of(candidate: Candidate, ctx: _Context) -> list[int] | None:
    if candidate.topology == ctx.topology:
        return ctx.dims
    return mesh_search.parse_topology(candidate.topology)


def _tokens_over(
    base: Mapping[str, int],
    new: Mapping[str, int],
    b0: float,
    b1: float,
    axes: Sequence[str],
) -> float:
    return (b1 / _shards(new, axes)) / (b0 / _shards(base, axes))


def _needs_split_physical_axes(mesh: Mapping[str, int],
                               dims: Sequence[int] | None) -> bool | None:
    """True when the mesh axes cannot each own whole torus dimensions."""
    if not dims:
        return None
    fit = mesh_search.block_layout(_chip_mesh(mesh, dims), dims)
    if "layout" not in fit:
        return None
    return int(fit.get("max_axes_per_block", 0)) > 1


def _origin_note(mesh: Mapping[str, int], dims: Sequence[int] | None) -> str:
    """Names a slice shape the mesh fits when it does not fit `dims`."""
    if not dims or len(dims) != 3 or not _needs_split_physical_axes(
            mesh, dims):
        return ""
    ici = [
        axis for axis, size in mesh.items()
        if int(size) > 1 and axis not in mesh_search.NON_ICI_AXES
    ]
    devices = math.prod(int(mesh[axis]) for axis in ici)
    chips = math.prod(int(d) for d in dims)
    if devices == chips:
        cores, scale = "2 TensorCores per chip", 0.5
    elif devices == 2 * chips:
        cores, scale = "1 TensorCore per chip", 2.0
    else:
        return ""
    for i in reversed(range(3)):
        if scale < 1 and int(dims[i]) % 2:
            continue
        alt = [int(d) for d in dims]
        alt[i] = int(alt[i] * scale)
        needs_split = _needs_split_physical_axes(mesh, alt)
        if needs_split is not None and not needs_split:
            shape = "x".join(str(d) for d in alt)
            return (
                "these axis sizes cannot each own whole dimensions of this slice but"
                f" can on a {shape} slice with {cores}: the mesh may have been tuned"
                " for that platform or slice shape")
    return ""


def estimate(candidate: Candidate, ctx: _Context) -> None:
    """Fills the candidate's gain, effects and memory arithmetic in place."""
    base, new = ctx.base, candidate.mesh
    dims = _dims_of(candidate, ctx)
    b0 = float(ctx.batch_size or 1)
    b1 = float(candidate.batch_size or ctx.batch_size or 1)
    batch_axes = [a for a in ctx.token_axes if _family(a) == "batch"]
    per_device0 = b0 / _shards(base, batch_axes)
    per_device1 = b1 / _shards(new, batch_axes)

    non_tensor_tokens = [a for a in ctx.token_axes if _family(a) != "tensor"]
    tokens = _tokens_over(base, new, b0, b1, non_tensor_tokens)
    routed_tokens = _tokens_over(
        base,
        new,
        b0,
        b1,
        list(dict.fromkeys(non_tensor_tokens + list(mesh_search.EXPERT_AXES))),
    )
    memory_tokens = _tokens_over(base, new, b0, b1, ctx.token_axes)
    batch_ratio = per_device1 / per_device0
    t0 = _shards(base, mesh_search.TENSOR_AXES)
    t1 = _shards(new, mesh_search.TENSOR_AXES)
    e0 = _shards(base, mesh_search.EXPERT_AXES)
    e1 = _shards(new, mesh_search.EXPERT_AXES)
    kv_ratio = _kv_per_device(new, ctx.num_kv_heads) / _kv_per_device(
        base, ctx.num_kv_heads)
    traffic = (t0 * e0) / (t1 * e1)
    resident = 1.0
    if ctx.fsdp_axes:
        resident_axes = (list(mesh_search.TENSOR_AXES) +
                         list(mesh_search.EXPERT_AXES) + list(ctx.fsdp_axes))
        resident = _shards(base, resident_axes) / _shards(new, resident_axes)
    gain = 0.0
    effects = []
    for group in ctx.groups:
        if group.hypothesis in (GLOBAL, HOST, UNKNOWN):
            continue
        n0, n1 = _shards(base, group.axes), _shards(new, group.axes)
        place = _placement(new, dims, group.axes, base) / _placement(
            base, ctx.dims, group.axes)
        if group.hypothesis == TENSOR:
            scale = tokens * _ring_ratio(t0, t1) * place if t1 > 1 else 0.0
        elif group.hypothesis == SEQUENCE:
            scale = (batch_ratio * kv_ratio * _ring_ratio(n0, n1) *
                     place if n1 > 1 else 0.0)
        elif group.hypothesis == WEIGHTS:
            scale = traffic * _ring_ratio(n0, n1) * place
        elif group.hypothesis == ROUTING:
            h0 = _hops(base, ctx.dims, group.axes)
            h1 = _hops(new, dims, group.axes)
            hops = h1 / h0 if h0 > 0 else 1.0
            scale = routed_tokens * _ring_ratio(n0,
                                                n1) * hops if n1 > 1 else 0.0
        elif group.axes and n1 <= 1:
            scale = 0.0
        else:
            scale = tokens * _ring_ratio(n0, n1) * place
        gain += group.pct * (1.0 - scale)
        if abs(scale - 1.0) >= 0.05 and group.pct >= 0.5:
            effects.append((group.label, group.pct, scale))
    if candidate.num_slices != ctx.num_slices and ctx.dcn_pct > 0:
        ici0, ici1 = _shards(base, list(base)), _shards(new, list(new))
        scale = _ring_ratio(ctx.num_slices, candidate.num_slices) * ici0 / ici1
        gain += ctx.dcn_pct * (1.0 - scale)
        effects.append(("cross-slice (DCN) collectives", ctx.dcn_pct, scale))
    effects.sort(key=lambda effect: -abs(effect[1] * (1.0 - effect[2])))
    candidate.gain_pct = gain
    candidate.effects = effects
    candidate.local_tokens_ratio = memory_tokens
    candidate.weight_bytes_ratio = resident
    candidate.tensor_shards_base = t0
    candidate.tensor_shards_new = t1
    candidate.weights_assumed_fsdp = not ctx.fsdp_axes and traffic > 1.0
    candidate.allow_split_physical_axes = _needs_split_physical_axes(new, dims)
    candidate.tokens_upper_bound = not ctx.token_axes_known and any(
        _family(move.receiver) in ("tensor", "expert")
        for move in candidate.moves)
    peak = (ctx.memory or {}).get("peak_fraction")
    if peak and ctx.token_axes_known:
        params = (ctx.memory or {}).get("parameter_fraction") or 0.0
        params = min(max(float(params), 0.0), 1.0)
        candidate.memory_fraction = float(peak) * (
            params * resident + (1.0 - params) * memory_tokens)
        candidate.oom_risk = candidate.memory_fraction > max(
            _OOM_FRACTION,
            float(peak) + 1e-6)
    else:
        ratio = max(memory_tokens, resident)
        candidate.oom_risk = (ratio > _MEMORY_RISK_RATIO
                              if candidate.tokens_upper_bound else ratio
                              >= _MEMORY_RISK_RATIO)


def _layout_rank(candidate: Candidate, ctx: _Context,
                 colocate: Sequence[str]) -> tuple[int, ...]:
    """Smaller is better: colocated block (plane, ring, none), finer layout."""
    dims = _dims_of(candidate, ctx)
    if not dims:
        return (9, )
    fit = mesh_search.block_layout(candidate.mesh, dims, colocate)
    if "layout" not in fit:
        return (9, )
    block_dims = (fit.get("colocate") or {}).get("block_dims", 0)
    group_rank = {2: 0, 1: 1, 3: 2}.get(block_dims, 3) if colocate else 0
    return (group_rank, -fit["num_blocks"], fit["max_axes_per_block"])


def _add_hypothesis(
    by_name: dict[str, Hypothesis],
    base: Mapping[str, int],
    name: str,
    pct: float,
    axes: Sequence[str],
    text: str,
) -> None:
    """Accumulates `pct`, `axes` and `text` onto `by_name[name]`."""
    hyp = by_name.setdefault(name, Hypothesis(name, 0.0))
    hyp.score += pct
    merged = set(hyp.axes) | set(axes)
    hyp.axes = tuple(axis for axis in base if axis in merged)
    if text:
        hyp.evidence.append(text)


def _hypotheses(ctx: _Context, allow_reshape: bool,
                allow_slice_change: bool) -> list[Hypothesis]:
    """Scores the hypotheses from the evidence groups and regime priors."""
    by_name: dict[str, Hypothesis] = {}
    for group in ctx.groups:
        if group.hypothesis in (GLOBAL, HOST, UNKNOWN):
            continue
        text = f"{group.pct:.1f}% {group.label}"
        if group.hypothesis == MIXED:
            families = sorted({_family(axis)
                               for axis in group.axes} - {"other"})
            for family in families:
                name = _FAMILY_HYPOTHESIS.get(family, MIXED)
                axes = [axis for axis in group.axes if _family(axis) == family]
                _add_hypothesis(by_name, ctx.base, name,
                                group.pct / len(families), axes, text)
        else:
            _add_hypothesis(by_name, ctx.base, group.hypothesis, group.pct,
                            group.axes, text)
        families = collections.Counter(_family(axis) for axis in group.axes)
        split = [f for f, n in families.items() if n > 1 and f != "other"]
        if split and group.hypothesis in (SEQUENCE, WEIGHTS, BATCH, MIXED):
            _add_hypothesis(
                by_name,
                ctx.base,
                LAYOUT,
                group.pct,
                group.axes,
                f"{text} (spans sibling axes)",
            )

    seq_axes = _active(ctx.base, mesh_search.SEQUENCE_AXES)
    tensor_axes = _active(ctx.base, mesh_search.TENSOR_AXES)
    length = ctx.sequence_length or 0
    if seq_axes and length and length <= _SHORT_CONTEXT:
        _add_hypothesis(
            by_name,
            ctx.base,
            SEQUENCE,
            _PRIOR_PCT,
            seq_axes,
            f"prior: short context ({length} tokens) rarely needs sequence"
            " parallelism",
        )
    if tensor_axes and length >= _LONG_CONTEXT:
        _add_hypothesis(
            by_name,
            ctx.base,
            TENSOR,
            _PRIOR_PCT,
            tensor_axes,
            f"prior: long context ({length} tokens): tensor all-reduce volume"
            " grows with local tokens",
        )
    if allow_reshape and ctx.dims:
        _add_hypothesis(
            by_name,
            ctx.base,
            LAYOUT,
            0.0,
            (),
            "the prompt allows re-shaping the slice",
        )
    if allow_slice_change and ctx.num_slices > 1:
        if ctx.dcn_pct > 0:
            text = (f"{ctx.dcn_pct:.1f}% cross-slice (DCN) collectives over"
                    f" {ctx.num_slices} slices")
        else:
            text = f"the prompt allows changing the {ctx.num_slices}-slice count"
        _add_hypothesis(by_name, ctx.base, SLICES, ctx.dcn_pct, (), text)
    if ctx.pipeline_axes:
        _add_hypothesis(
            by_name,
            ctx.base,
            PIPELINE,
            0.0,
            ctx.pipeline_axes,
            "the pipeline depth may change: every microbatch or prefill chunk"
            " traverses all stages, which the volume model does not count",
        )
    volume = [
        h for h in by_name.values()
        if h.name in VOLUME_HYPOTHESES and h.score >= MIN_HYPOTHESIS_PCT
    ]
    volume.sort(key=lambda h: -h.score)
    placement = [
        by_name[n] for n in (LAYOUT, SLICES, PIPELINE) if n in by_name
    ]
    return volume + placement


def _subfamily(axis: str) -> str:
    """Sub-family for sibling grouping (`fsdp` is distinct from `data`/`data2`)."""
    if axis == "fsdp":
        return "fsdp"
    return _family(axis)


def _primary(axes: Sequence[str]) -> str:
    """'seq' before 'seq2', 'data' before 'data2' (shortest, then lowest)."""
    return sorted(axes, key=lambda a: (len(a), a))[0]


def _family_primary(mesh: Mapping[str, int], family: str) -> str | None:
    """The primary movable axis of a sub-family in the mesh."""
    axes = [
        a for a in mesh
        if _subfamily(a) == family and a not in mesh_search.DEFAULT_FIXED_AXES
    ]
    return _primary(axes) if axes else None


def _canonical(move: Move, mesh: Mapping[str, int]) -> bool:
    """A move between families targets the receiving family's primary axis."""
    family = _subfamily(move.receiver)
    if family in ("other", _subfamily(move.donor)):
        return True
    return move.receiver == _family_primary(mesh, family)


def _refactor(move: Move) -> bool:
    """True for a move between two sibling axes of one family."""
    family = _subfamily(move.donor)
    return family != "other" and family == _subfamily(move.receiver)


def _remedy(move: Move, hyp: Hypothesis) -> bool:
    """True when one move relieves the hypothesis' bottleneck on its own."""
    if _refactor(move) or move.receiver in PIPELINE_AXES:
        return False
    donor, receiver = _family(move.donor), _family(move.receiver)
    axes = set(hyp.axes)
    if hyp.name == TENSOR:
        return donor == "tensor" and receiver != "tensor"
    if hyp.name == SEQUENCE:
        return (move.donor in axes or not axes) and receiver != "sequence"
    if hyp.name == BATCH:
        return move.donor in axes and receiver != "batch"
    if hyp.name == ROUTING:
        collapse = (donor == "tensor" and receiver == "batch"
                    and {move.donor, move.receiver} <= axes)
        return (collapse or (move.donor in axes and move.receiver not in axes)
                or (not axes and receiver == "expert"))
    if hyp.name == WEIGHTS:
        return receiver in ("tensor", "expert") and donor not in (
            "tensor",
            "expert",
        )
    if hyp.name == MIXED:
        return move.donor in axes and move.receiver not in axes
    return False


def _single_remedy(candidate: Candidate, hyp: Hypothesis,
                   mesh: Mapping[str, int]) -> bool:
    return (candidate.change == "move" and len(candidate.moves) == 1
            and _remedy(candidate.moves[0], hyp)
            and _canonical(candidate.moves[0], mesh))


def _compound_remedy(candidate: Candidate, hyp: Hypothesis,
                     mesh: Mapping[str, int]) -> bool:
    """Two moves that relieve `hyp` together."""
    if candidate.change != "move" or len(candidate.moves) != 2:
        return False
    if not all(_canonical(m, mesh) for m in candidate.moves):
        return False
    remedies = [m for m in candidate.moves if _remedy(m, hyp)]
    if len(remedies) == 2:
        return (remedies[0].receiver == remedies[1].receiver
                or remedies[0].donor == remedies[1].donor)
    return len(remedies) == 1 and all(
        _refactor(m) and int(mesh.get(m.donor, 1)) > 1 and int(
            mesh.get(m.receiver, 1)) > 1 and m.receiver == remedies[0].receiver
        for m in candidate.moves if m not in remedies)


def _move_pool(
    ctx: _Context,
    extra_axes: Sequence[str],
    fixed_axes: Sequence[str],
    max_moves: int,
) -> tuple[dict[str, int], list[Candidate]]:
    """Legal move candidates on the current slice, with bracket roles."""
    result = mesh_search.enumerate_mesh_candidates(
        ctx.base,
        extra_axes=extra_axes,
        fixed_axes=fixed_axes,
        batch_size=ctx.batch_size,
        sequence_length=ctx.sequence_length,
        num_heads=ctx.num_heads,
        num_kv_heads=ctx.num_kv_heads,
        topology=ctx.topology,
        max_moves=max_moves,
    )
    roles = {}
    for pair in result["bracketed_pairs"]:
        for point in pair["points"]:
            roles[(pair["donor"], pair["receiver"],
                   point["factor"])] = point["bracket"]
    out = []
    for entry in result["candidates"]:
        mesh = dict(entry["mesh"])
        if not _kv_heads_ok(mesh, ctx):
            ctx.kv_excluded += 1
            continue
        moves = _parse_moves(entry["moves"])
        bracket = ""
        if len(moves) == 1:
            move = moves[0]
            bracket = roles.get((move.donor, move.receiver, move.factor),
                                "middle")
        out.append(
            Candidate(
                mesh=mesh,
                moves=moves,
                topology=ctx.topology,
                num_slices=ctx.num_slices,
                batch_size=ctx.batch_size,
                bracket=bracket,
                warnings=tuple(entry.get("warnings") or ()),
            ))
    return dict(result["baseline"]["mesh"]), out


def _reshape_pool(ctx: _Context) -> list[Candidate]:
    """The baseline mesh on every other same-chip-count slice shape."""
    if not ctx.dims:
        return []
    return [
        Candidate(
            mesh=dict(ctx.base),
            topology=shape,
            num_slices=ctx.num_slices,
            batch_size=ctx.batch_size,
            change="reshape",
            bracket=f"re-shape {ctx.topology} -> {shape}",
        ) for shape in mesh_search.alternative_topologies(ctx.dims)
    ]


def _shapes_with(chips: int, ndim: int = 3) -> list[str]:
    """Power-of-two slice shapes with `chips` chips (dims >= 4)."""
    if chips & (chips - 1):
        return []
    if ndim == 2:
        if chips < 16:
            return []
        shapes = []
        x_dim = 4
        while x_dim * x_dim <= chips:
            shapes.append(f"{x_dim}x{chips // x_dim}")
            x_dim *= 2
        return shapes
    if chips < 64:
        return []
    first = [4, 4, chips // 16]
    return ["x".join(str(d) for d in first)
            ] + mesh_search.alternative_topologies(first)


def _kv_heads_ok(mesh: Mapping[str, int], ctx: _Context) -> bool:
    """Returns whether the mesh meets the default KV-head policy."""
    if not ctx.num_kv_heads:
        return True
    kv_axes = [axis for axis in mesh_search.TENSOR_AXES if axis != "model_q"]
    kv = _shards(mesh, kv_axes)
    return kv == _shards(ctx.base, kv_axes) or ctx.num_kv_heads % kv == 0


def _legal(mesh: Mapping[str, int], batch: int | None, ctx: _Context) -> bool:
    """Returns whether `mesh` passes the batch, sequence and head checks."""
    checks = (
        (batch, mesh_search.BATCH_AXES),
        (ctx.sequence_length, mesh_search.SEQUENCE_AXES),
        (ctx.num_heads, mesh_search.QUERY_HEAD_AXES),
    )
    if not all(value is None or value % _shards(mesh, axes) == 0
               for value, axes in checks):
        return False
    return _kv_heads_ok(mesh, ctx)


def _slice_pool(ctx: _Context) -> list[Candidate]:
    """Fewer, larger slices; the freed factor goes to a batch, sequence or tensor axis."""
    if not ctx.dims or ctx.num_slices < 2:
        return []
    out = []
    for factor in (2, 4):
        if ctx.num_slices % factor:
            continue
        shapes = _shapes_with(math.prod(ctx.dims) * factor, len(ctx.dims))
        batch = ctx.batch_size * factor if ctx.batch_size else None
        for family_axes in (
                mesh_search.BATCH_AXES,
                mesh_search.SEQUENCE_AXES,
                mesh_search.TENSOR_AXES,
        ):
            present = [
                axis for axis in ctx.base
                if axis in family_axes and axis not in mesh_search.NON_ICI_AXES
            ]
            if not present:
                continue
            receiver = _primary(present)
            mesh = dict(ctx.base)
            mesh[receiver] *= factor
            variants: list[tuple[tuple[Move, ...], dict[str,
                                                        int]]] = [((), mesh)]
            for family in ("sequence", "batch"):
                axes = [
                    a for a in mesh if _family(a) == family and mesh[a] > 1
                    and a not in mesh_search.NON_ICI_AXES
                ]
                if len(axes) > 1:
                    primary = _primary(axes)
                    others = [a for a in axes if a != primary]
                    folded = dict(mesh)
                    moves = tuple(Move(a, primary, folded[a]) for a in others)
                    for a in others:
                        folded[primary] *= folded[a]
                        folded[a] = 1
                    variants.append((moves, folded))
            for moves, variant in variants:
                if not _legal(variant, batch, ctx):
                    continue
                for shape in shapes:
                    out.append(
                        Candidate(
                            mesh=variant,
                            moves=moves,
                            topology=shape,
                            num_slices=ctx.num_slices // factor,
                            batch_size=batch,
                            change="slices",
                            bracket=
                            (f"{ctx.num_slices} -> {ctx.num_slices // factor}"
                             f" slices of {shape}, {receiver} x{factor}"),
                        ))
    return out


class _Portfolio:
    """Collects proposals without repeating a mesh."""

    def __init__(self, size: int):
        self.size = size
        self.picks: list[Candidate] = []
        self.taken: set[tuple[Any, ...]] = set()

    def full(self) -> bool:
        return len(self.picks) >= self.size

    def take(self, candidate: Candidate | None, hyp: str, role: str) -> bool:
        if candidate is None or self.full() or candidate.key() in self.taken:
            return False
        candidate.hypothesis, candidate.role = hyp, role
        self.picks.append(candidate)
        self.taken.add(candidate.key())
        return True


def _rank_pair(points: Sequence[Candidate],
               mesh: Mapping[str, int]) -> tuple[bool, float, bool]:
    """Sort key for a (donor, receiver) pair: active receiver, gain, secondary donor."""
    best = max(c.gain_pct for c in points if not c.oom_risk)
    move = points[0].moves[0]
    donor, receiver = move.donor, move.receiver
    inactive_receiver = int(mesh.get(receiver, 1)) <= 1
    is_primary = donor == _family_primary(mesh, _subfamily(donor))
    return (inactive_receiver, -round(best * 2) / 2, is_primary)


def _pairs(pool: Sequence[Candidate], hyp: Hypothesis,
           mesh: Mapping[str, int]) -> list[list[Candidate]]:
    """Single-move remedies grouped by (donor, receiver), best pair first."""
    grouped: dict[tuple[str, str],
                  list[Candidate]] = collections.defaultdict(list)
    for candidate in pool:
        if _single_remedy(candidate, hyp, mesh):
            move = candidate.moves[0]
            grouped[(move.donor, move.receiver)].append(candidate)
    pairs = []
    for points in grouped.values():
        points.sort(key=lambda c: c.moves[0].factor)
        if any(not c.oom_risk for c in points):
            pairs.append(points)
    pairs.sort(key=lambda points: _rank_pair(points, mesh))
    return pairs


def _first_step(points: Sequence[Candidate]) -> Candidate | None:
    return next((c for c in points if not c.oom_risk), None)


def _boundary(points: Sequence[Candidate]) -> Candidate | None:
    """The largest factor that passes the memory check."""
    return next((c for c in reversed(points) if not c.oom_risk), None)


def _sibling_order(c: Candidate) -> tuple[int, int]:
    """Sort key for sibling re-factors: swap, k=2 to primary, k=2 away, fold."""
    move = c.moves[0]
    towards_primary = move.receiver == _primary([move.donor, move.receiver])
    if "swap" in c.bracket:
        return (0, 0)
    if c.bracket.startswith("k=") and towards_primary:
        return (1, 0)
    if c.bracket.startswith("k="):
        return (2, 0)
    if "boundary" in c.bracket and towards_primary:
        return (3, 0)
    return (4, move.factor)


def _sibling_options(pool: Sequence[Candidate],
                     hyp: Hypothesis) -> list[Candidate]:
    """Sibling re-factors of the axes the hypothesis' groups span."""
    axes = set(hyp.axes)
    options = [
        c for c in pool
        if c.change == "move" and len(c.moves) == 1 and _refactor(c.moves[0])
        and {c.moves[0].donor, c.moves[0].receiver} <= axes and not c.oom_risk
    ]
    return sorted(options, key=_sibling_order)


def _hyp_compounds(pool: Sequence[Candidate], hyp: Hypothesis,
                   base: Mapping[str, int]) -> list[Candidate]:
    """Compound remedies for `hyp` passing the memory check, best gain first."""
    return sorted(
        (c for c in pool if not c.oom_risk and _compound_remedy(c, hyp, base)),
        key=lambda c: -c.gain_pct,
    )


def _pipeline_options(pool: Sequence[Candidate],
                      ctx: _Context) -> list[Candidate]:
    """First steps (k = 2) from a movable pipeline axis into an SPMD axis."""
    options = [
        c for c in pool if c.change == "move" and len(c.moves) == 1
        and c.moves[0].donor in ctx.pipeline_axes and c.moves[0].receiver
        not in PIPELINE_AXES and c.moves[0].factor == 2 and not c.oom_risk
    ]
    return sorted(
        options,
        key=lambda c: (
            int(ctx.base.get(c.moves[0].receiver, 1)) <= 1,
            -c.gain_pct,
        ),
    )


def _request_slot(
    portfolio: _Portfolio,
    candidates: Sequence[Candidate | None],
    hyp: str,
    role: str,
) -> None:
    """Takes the first candidate in `candidates` not yet in `portfolio`."""
    for candidate in candidates:
        if portfolio.take(candidate, hyp, role):
            return


def _fill(
    portfolio: _Portfolio,
    hyps: Sequence[Hypothesis],
    pool: Sequence[Candidate],
    ctx: _Context,
) -> None:
    """Fills the portfolio from the template (see the module docstring)."""
    volume = [h for h in hyps if h.name in VOLUME_HYPOTHESES]
    by_name = {h.name: h for h in hyps}
    pairs = {h.name: _pairs(pool, h, ctx.base) for h in volume}
    first_steps = [[_first_step(p) for p in pairs[h.name]] for h in volume]

    top: list[Candidate] = []
    alt: list[Candidate] = []
    if volume and pairs[volume[0].name]:
        ranked = pairs[volume[0].name]
        top = ranked[0]
        donor, receiver = top[0].moves[0].donor, top[0].moves[0].receiver
        alt = (next(
            (p for p in ranked[1:]
             if _subfamily(p[0].moves[0].donor) != _subfamily(donor)),
            [],
        ) or next(
            (p for p in ranked[1:] if p[0].moves[0].receiver != receiver), [])
               or next(
                   (p
                    for p in ranked[1:] if p[0].moves[0].donor != donor), []))
    h1 = volume[0].name if volume else ""
    h2 = volume[1].name if len(volume) > 1 else ""
    h3 = volume[2].name if len(volume) > 2 else ""

    placement: list[tuple[Candidate, str, str]] = []
    if LAYOUT in by_name:
        for option in _sibling_options(pool, by_name[LAYOUT])[:2]:
            placement.append((option, LAYOUT, "sibling re-factor"))
        colocate = next(
            (g.axes for g in sorted(ctx.groups, key=lambda g: -g.pct)
             if g.hypothesis in (ROUTING, SEQUENCE, TENSOR, MIXED) and g.axes),
            (),
        )
        reshapes = sorted(
            (c for c in pool if c.change == "reshape"),
            key=lambda c: (_layout_rank(c, ctx, colocate), -c.gain_pct),
        )
        if reshapes:
            placement.insert(min(1, len(placement)),
                             (reshapes[0], LAYOUT, "re-shaped slice"))
    if SLICES in by_name:
        slices = sorted(
            (c for c in pool if c.change == "slices" and not c.oom_risk),
            key=lambda c: (
                c.local_tokens_ratio > 1.0,
                _layout_rank(c, ctx, ()),
                -c.num_slices,
                -c.gain_pct,
            ),
        )
        if slices:
            placement.insert(0, (slices[0], SLICES, "fewer, larger slices"))
            if ctx.dcn_pct > 0:
                for extra_slice in slices[1:]:
                    if extra_slice.num_slices != slices[0].num_slices:
                        placement.insert(
                            1, (extra_slice, SLICES, "fewer, larger slices"))
                        break
    if PIPELINE in by_name:
        for option in _pipeline_options(pool, ctx)[:1]:
            placement.insert(0,
                             (option, PIPELINE, "pipeline depth, first step"))

    h1_compounds = _hyp_compounds(pool, volume[0], ctx.base) if volume else []
    h2_compounds = (_hyp_compounds(pool, volume[1], ctx.base)
                    if len(volume) > 1 and volume[1].score >= 15.0 else [])

    h1_axes = set(volume[0].axes) if volume else set()
    h2_axes = (set(volume[1].axes)
               if len(volume) > 1 and volume[1].score >= 15.0 else set())
    top_b = _boundary(top)
    if (top_b is not None and top_b.moves[0].receiver in h2_axes
            and top_b.gain_pct < 0):
        top_b = None
    alt_b = _boundary(alt)
    if (alt_b is not None and alt_b.moves[0].receiver in h2_axes
            and alt_b.gain_pct < 0):
        alt_b = None
    top_donor = top[0].moves[0].donor if top else ""
    top_donor_steps = [
        c for p in (pairs[h1] if h1 else [])
        if p and p[0].moves[0].donor == top_donor and (c := _first_step(p))
        is not None and not (c.moves[0].receiver in h2_axes and c.gain_pct < 0)
    ]
    alt_donor = alt[0].moves[0].donor if alt else ""
    alt_donor_steps = [
        c for p in (pairs[h1] if h1 else [])
        if p and p[0].moves[0].donor == alt_donor and (c := _first_step(p))
        is not None and not (c.moves[0].receiver in h2_axes and c.gain_pct < 0)
    ]
    h3_steps = [
        c for c in (first_steps[2] if len(first_steps) > 2 else [])
        if c is not None and not (c.moves[0].receiver in
                                  (h1_axes | h2_axes) and c.gain_pct < 0)
    ]
    h2_b = _boundary(pairs[h2][0]) if h2 and pairs[h2] else None
    if (h2_b is not None and h2_b.moves[0].receiver in h1_axes
            and h2_b.gain_pct < 0):
        h2_b = None

    _request_slot(portfolio, [_first_step(top)], h1,
                  "strongest move, first step")
    _request_slot(portfolio, [top_b] + top_donor_steps, h1,
                  "strongest move, boundary")
    _request_slot(
        portfolio,
        [_first_step(alt)],
        h1,
        "same bottleneck, another axis, first step",
    )
    _request_slot(
        portfolio,
        [alt_b] + alt_donor_steps,
        h1,
        "same bottleneck, another axis, boundary",
    )
    _request_slot(
        portfolio,
        first_steps[1] if len(first_steps) > 1 else [],
        h2,
        "second hypothesis, first step",
    )
    if placement:
        _request_slot(portfolio, [placement[0][0]], placement[0][1],
                      placement[0][2])
    if h1_compounds:
        _request_slot(portfolio, h1_compounds, h1, "compound remedy")
    if h2_compounds:
        _request_slot(portfolio, h2_compounds, h2,
                      "second hypothesis, compound remedy")
    _request_slot(portfolio, h3_steps, h3, "third hypothesis, first step")
    if h2_b is not None:
        _request_slot(portfolio, [h2_b], h2, "second hypothesis, boundary")
    for option, hyp, role in placement[1:]:
        _request_slot(portfolio, [option], hyp, role)
    for index, steps in enumerate(first_steps):
        _request_slot(portfolio, steps, volume[index].name, "next first step")
    if top and h1:
        for point in reversed(top):
            if not point.oom_risk:
                _request_slot(portfolio, [point], h1,
                              "strongest move, intermediate factor")
    rest = sorted(
        (c for c in pool if c.change == "move" and not c.oom_risk and all(
            _canonical(m, ctx.base) and not _refactor(m)
            and m.receiver not in PIPELINE_AXES
            for m in c.moves) and not (len(c.moves) > 1 and any(
                int(ctx.base.get(m.receiver, 1)) <= 1 for m in c.moves))),
        key=lambda c: (
            any(int(ctx.base.get(m.receiver, 1)) <= 1 for m in c.moves),
            len(c.moves),
            -c.gain_pct,
        ),
    )
    for candidate in rest:
        if portfolio.full():
            break
        portfolio.take(candidate, "any", "best remaining estimate")


def _fineness(mesh: Mapping[str, int],
              dims: Sequence[int] | None) -> tuple[int, int] | None:
    """(-blocks, most axes in a block) of the finest layout; smaller is finer."""
    if not dims:
        return None
    fit = mesh_search.block_layout(_chip_mesh(mesh, dims), dims)
    if "layout" not in fit:
        return None
    return (-fit["num_blocks"], fit["max_axes_per_block"])


def _best_shape(
    candidate: Candidate,
    ctx: _Context,
    reshapes: Mapping[str, Candidate],
) -> Candidate | None:
    """The candidate on the slice shape that suits this move, if any."""
    fine_now = _fineness(candidate.mesh, ctx.dims)
    base_fine_now = _fineness(ctx.base, ctx.dims)
    options = []
    for shape, reshaped_base in reshapes.items():
        dims = mesh_search.parse_topology(shape)
        trial = dataclasses.replace(
            candidate,
            topology=shape,
            change="reshape",
            bracket=f"re-shape {ctx.topology} -> {shape}",
            effects=[],
        )
        estimate(trial, ctx)
        if trial.oom_risk and not candidate.oom_risk:
            continue
        improvement = trial.gain_pct - candidate.gain_pct
        interaction = improvement - reshaped_base.gain_pct
        fine = _fineness(candidate.mesh, dims)
        base_fine = _fineness(ctx.base, dims)
        finer = (fine is not None and fine_now is not None and fine < fine_now
                 and not (base_fine is not None and base_fine_now is not None
                          and base_fine < base_fine_now))
        helps = min(improvement, interaction) >= _RESHAPE_MIN_GAIN_PCT
        neutral = 0.0 <= improvement < _RESHAPE_MIN_GAIN_PCT
        if helps or (neutral and finer):
            options.append((-trial.gain_pct, fine or (0, 0), shape, trial))
    if not options:
        return None
    best = min(options, key=lambda option: option[:3])[3]
    best.current_shape_gain_pct = candidate.gain_pct
    return best


def _reshape_moves(portfolio: _Portfolio, ctx: _Context) -> None:
    """Proposes each mesh move on the slice shape that suits it (`_best_shape`)."""
    reshapes = {}
    for shape in mesh_search.alternative_topologies(ctx.dims or []):
        reshaped = Candidate(
            mesh=dict(ctx.base),
            topology=shape,
            num_slices=ctx.num_slices,
            batch_size=ctx.batch_size,
            change="reshape",
        )
        estimate(reshaped, ctx)
        reshapes[shape] = reshaped
    if not reshapes:
        return
    for index, pick in enumerate(portfolio.picks):
        if pick.change != "move":
            continue
        better = _best_shape(pick, ctx, reshapes)
        if better is None or better.key() in portfolio.taken:
            continue
        portfolio.taken.discard(pick.key())
        portfolio.taken.add(better.key())
        portfolio.picks[index] = better


def _routing_group(groups: Sequence[EvidenceGroup]) -> tuple[str, ...]:
    """Axes of the largest MoE routing evidence group, or ()."""
    routing = [g for g in groups if g.hypothesis == ROUTING and g.axes]
    return max(routing, key=lambda g: g.pct).axes if routing else ()


def _dims_text(
    block: Sequence[int],
    dims: Sequence[int],
    dim_letters: str = "",
) -> str:
    """Names a block of whole torus dims, e.g. '4x4 X-Y plane', '32-long Z ring'."""
    names = f"{dim_letters} " if dim_letters else ""
    if not names:
        have = collections.Counter(int(dim) for dim in dims)
        want = collections.Counter(int(dim) for dim in block)
        if len(dims) <= len(_DIM_NAMES) and all(
                have[size] == count for size, count in want.items()):
            names = (
                "-".join(_DIM_NAMES[i]
                         for i, dim in enumerate(dims) if int(dim) in want) +
                " ")
    if len(block) == 1:
        return f"{block[0]}-long {names}ring"
    sizes = "x".join(str(dim) for dim in block)
    if len(block) == 2:
        return f"{sizes} {names}plane"
    return f"whole {sizes} slice"


def _wraps(block: Sequence[int], dims: Sequence[int]) -> bool:
    """True when every side of a block of whole dims has wrap-around links."""
    wraps = mesh_search.wraparound_dims(dims)
    return all(
        any(wrap and int(dim) == int(side) for wrap, dim in zip(wraps, dims))
        for side in block)


def _place_text(block: Sequence[int],
                dims: Sequence[int],
                dim_letters: str = "") -> str:
    """'the 16-long Z ring (wrap-around links)', 'the 4x4 X-Y plane (2D ...)'."""
    where = f"the {_dims_text(block, dims, dim_letters)}"
    if len(block) == 1:
        if not _wraps(block, dims):
            return f"{where} (no wrap-around links on this slice)"
        return f"{where} (wrap-around links)"
    return f"{where} ({len(block)}D collectives)"


def _assign_block_letters(
    blocks: Sequence[tuple[tuple[int, ...], tuple[str, ...]]],
    dims: Sequence[int],
) -> list[str]:
    """Assigns X, Y, Z letters in order across blocks that partition `dims`."""
    if len(dims) > len(_DIM_NAMES) or len(blocks) <= 1:
        return ["" for _ in blocks]
    used: set[int] = set()
    out: list[str] = []
    for block, _ in blocks:
        letters = []
        for side in block:
            idx = next(
                (i for i, d in enumerate(dims)
                 if int(d) == int(side) and i not in used),
                None,
            )
            if idx is not None:
                used.add(idx)
                letters.append(_DIM_NAMES[idx])
        out.append("-".join(letters) if len(letters) == len(block) else "")
    return out


def _cores_per_chip(mesh: Mapping[str, int],
                    dims: Sequence[int] | None) -> int:
    """Infers XLA devices per chip (1 or 2): mesh product vs slice chips."""
    if int(mesh.get("intrachip", 1) or 1) == 2:
        return 2
    if not dims:
        return 1
    chips = math.prod(int(d) for d in dims)
    ici = [
        a for a, v in mesh.items()
        if int(v) > 1 and a not in mesh_search.NON_ICI_AXES
    ]
    devices = math.prod(int(mesh[a]) for a in ici)
    return 2 if chips > 0 and devices == 2 * chips else 1


def torus_mapping(
    mesh: Mapping[str, int],
    topology: str | None,
    group: Sequence[str] = (),
    group_name: str = "MoE routing group",
    partition_config: Mapping[str, Any] | None = None,
) -> str:
    """Says in words where the mesh axes land on the physical torus."""
    dims = mesh_search.parse_topology(topology)
    if not dims:
        return ""
    chip_mesh = _chip_mesh(mesh, dims)
    folded_note = ""
    if chip_mesh != {k: int(v) for k, v in mesh.items()}:
        folded_axis = next(k for k in mesh if chip_mesh.get(k) != int(mesh[k]))
        folded_note = (
            f" (`{folded_axis}`={mesh[folded_axis]} spans 2 TensorCores per chip"
            f" on-chip and {chip_mesh[folded_axis]} across chips)")
    layouts = mesh_search.tied_layouts(chip_mesh, dims, group)
    if not layouts:
        note = mesh_search.block_layout(chip_mesh, dims).get("note", "")
        return f"no torus mapping ({note})"
    blocks = layouts[0]
    block_letters = _assign_block_letters(blocks, dims)
    parts = []
    for (block, axes), letters in zip(blocks, block_letters):
        other_places = set()
        for tied in layouts[1:]:
            for other, other_axes in tied:
                if other_axes == axes and other != block:
                    other_places.add(other)
        places = [block] + sorted(other_places)
        names = " x ".join(f"`{axis}`={chip_mesh[axis]}" +
                           (f" (mesh {mesh[axis]})" if chip_mesh[axis] !=
                            int(mesh[axis]) else "") for axis in axes)
        if len(axes) == 1:
            if len(places) > 1:
                parts.append(f"{names} owns " + " or ".join(
                    _place_text(place, dims) for place in places))
            elif len(block) == 1:
                parts.append(
                    f"{names} owns {_place_text(block, dims, letters)}")
            else:
                parts.append(
                    f"{names} spans {_place_text(block, dims, letters)}")
            continue
        if len(places) > 1:
            where = " or ".join(_place_text(place, dims) for place in places)
        else:
            where = _place_text(block, dims, letters)
        owners = (mesh_search.torus_fit(
            {
                axis: chip_mesh[axis]
                for axis in axes
            }, block).get("whole") or {})
        sharers = [axis for axis in axes if axis not in owners]
        if not sharers:
            parts.append(
                f"{names} tile {where} as a grid of rings (each axis owns whole"
                " dimensions)")
        else:
            longest = max(max(place) for place in places)
            long_axes = [axis for axis in sharers if chip_mesh[axis] > longest]
            notes = [
                ", ".join(f"`{axis}`" for axis in sharers) +
                f" own{'s' if len(sharers) == 1 else ''} no whole dimension:"
                " strided or partial rings, allow_split_physical_axes"
            ]
            if long_axes:
                notes.append(
                    ", ".join(f"`{axis}`={chip_mesh[axis]}"
                              for axis in long_axes) +
                    f" {'is' if len(long_axes) == 1 else 'are'} longer than any"
                    " one dimension, so it spans several")
            if all(
                    len(place) == 1 and _wraps(place, dims)
                    for place in places):
                notes.insert(0, "together with the ring's wrap-around links")
            parts.append(f"{names} share {where} ({'; '.join(notes)})")
    text = ("; ".join(parts) or "no inter-chip axes") + folded_note
    active = [
        axis for axis in group if int(chip_mesh.get(axis, 1)) > 1
        and axis not in mesh_search.NON_ICI_AXES
    ]
    if active:
        label = (f"{group_name} " + " x ".join(f"`{axis}`"
                                               for axis in active) +
                 f" ({_shards(mesh, active)} devices)")
        mean_hops = _hops(mesh, dims, active)
        hops = (
            f"~{mean_hops:.3g} hop{'' if mean_hops == 1 else 's'} per all-to-all"
            " transfer")
        block_idx = next(
            (i for i, (_, axes) in enumerate(blocks)
             if set(axes) == set(active)),
            None,
        )
        if block_idx is None:
            text += (
                f". {label} fills no whole ring or plane: its all-to-all crosses"
                f" links that other axes use ({hops})")
        else:
            block = blocks[block_idx][0]
            letters = block_letters[block_idx]
            text += (
                f". {label} fills the {_dims_text(block, dims, letters)} ({hops};"
                " its all-to-all uses only that block's links)")
    pc_text = mesh_topology.describe_partition_layout(
        partition_config,
        mesh,
        dims,
        cores_per_chip=_cores_per_chip(mesh, dims),
        group=group,
        group_name=group_name,
    )
    if pc_text:
        text += f". {pc_text[0].upper() + pc_text[1:]}"
    return text


def _legality_text(candidate: Candidate, ctx: _Context) -> str:
    """The batch, sequence, head and KV-head divisibility of a proposal."""
    mesh = candidate.mesh
    parts = []
    for name, value, axes, unit in (
        (
            "batch",
            candidate.batch_size or ctx.batch_size,
            mesh_search.BATCH_AXES,
            "per batch shard",
        ),
        (
            "sequence",
            ctx.sequence_length,
            mesh_search.SEQUENCE_AXES,
            "tokens per sequence shard",
        ),
        ("heads", ctx.num_heads, mesh_search.QUERY_HEAD_AXES,
         "per tensor shard"),
    ):
        if not value:
            continue
        shards = _shards(mesh, axes)
        if value % shards:
            parts.append(f"{name} {value} is not divisible by {shards} shards")
        else:
            parts.append(
                f"{name} {value} / {shards} = {value // shards} {unit}")
    if ctx.num_kv_heads:
        kv = _shards(
            mesh,
            [axis for axis in mesh_search.TENSOR_AXES if axis != "model_q"])
        if ctx.num_kv_heads % kv:
            parts.append(
                f"KV heads {ctx.num_kv_heads} on {kv} shards: replicated or uneven"
                " (needs shard_kv_heads_along_model_axis=False)")
        else:
            parts.append(f"KV heads {ctx.num_kv_heads} / {kv} ="
                         f" {ctx.num_kv_heads // kv} per tensor shard")
    return "; ".join(parts)


def build_portfolio(
    baseline_mesh: Mapping[str, int],
    breakdown: Mapping[str, Any] | None,
    *,
    topology: str | None = None,
    num_slices: int = 1,
    batch_size: int | None = None,
    sequence_length: int | None = None,
    num_heads: int | None = None,
    num_kv_heads: int | None = None,
    extra_axes: Sequence[str] = (),
    fixed_axes: Sequence[str] = mesh_search.DEFAULT_FIXED_AXES,
    token_axes: Sequence[str] = (),
    fsdp_axes: Sequence[str] = (),
    partition_config: Mapping[str, Any] | None = None,
    memory: Mapping[str, Any] | None = None,
    dcn_pct: float = 0.0,
    allow_reshape: bool = False,
    allow_slice_change: bool = False,
    num_proposals: int = DEFAULT_NUM_PROPOSALS,
    max_moves: int = 2,
) -> dict[str, Any]:
    """Builds a ranked, diverse portfolio of mesh proposals."""
    base = mesh_search.parse_mesh(baseline_mesh)
    dims = mesh_search.parse_topology(topology)
    for axis in extra_axes:
        base.setdefault(axis, 1)
    groups = evidence_groups(breakdown, base)
    token_axes_known = bool(token_axes)
    if not token_axes:
        token_axes = [
            axis for axis in base if _family(axis) in ("batch", "sequence")
        ]
    else:
        token_axes = list(
            dict.fromkeys(
                list(token_axes) +
                [axis for axis in base if _family(axis) == "batch"]))
    ctx = _Context(
        base=base,
        dims=dims,
        topology="x".join(str(d) for d in dims) if dims else None,
        num_slices=max(1, int(num_slices or 1)),
        batch_size=batch_size or None,
        sequence_length=sequence_length or None,
        num_heads=num_heads or None,
        num_kv_heads=num_kv_heads or None,
        groups=groups,
        token_axes=tuple(token_axes),
        fsdp_axes=tuple(fsdp_axes),
        dcn_pct=float(dcn_pct or 0.0),
        memory=memory,
        token_axes_known=token_axes_known,
        pipeline_axes=tuple(
            axis for axis in PIPELINE_AXES
            if int(base.get(axis, 1)) > 1 and axis not in fixed_axes),
    )
    base_full, pool = _move_pool(ctx, extra_axes, fixed_axes, max_moves)
    ctx.base = base_full
    if allow_reshape:
        pool += _reshape_pool(ctx)
    if allow_slice_change:
        pool += _slice_pool(ctx)
    for candidate in pool:
        estimate(candidate, ctx)
    hyps = _hypotheses(ctx, allow_reshape, allow_slice_change)
    notes = []
    if not groups:
        notes.append("No collective breakdown: only regime priors were used.")
    peak = (memory or {}).get("peak_fraction")
    if peak and float(peak) >= 0.9:
        notes.append(
            f"Baseline peak HBM is {100 * float(peak):.0f}% of capacity: moves"
            " that grow per-device activations or weights may need more remat or"
            " a smaller per-device batch; flag them for review (ahead-of-time"
            " compilation checks memory).")
    if peak and not token_axes_known:
        notes.append(
            "The trainer's activation sharding is unknown (no token axes), so no"
            " proposal is excluded on memory and 'up to' marks local-token growth"
            " from moves into tensor axes, which also shard activations in"
            " sequence-parallel trainers.")
    if ctx.kv_excluded:
        notes.append(
            f"{ctx.kv_excluded} candidate meshes were left out because their"
            " KV-head shards (tensor shards except model_q) do not divide"
            f" num_kv_heads={ctx.num_kv_heads}: splitting KV heads unevenly or"
            " replicating them needs shard_kv_heads_along_model_axis=False.")
    pinned = [
        axis for axis in PIPELINE_AXES
        if int(base.get(axis, 1)) > 1 and axis in fixed_axes
    ]
    if pinned:
        notes.append(
            f"Pipeline axis {', '.join(f'{a}={base[a]}' for a in pinned)} is"
            " pinned (fixed axes). If the pipeline depth may change, re-run with"
            " --fixed_axes set to the axes that must keep their size, without"
            f" {', '.join(pinned)}, to also rank first steps from"
            f" {'it' if len(pinned) == 1 else 'them'} into an SPMD axis.")
    portfolio = _Portfolio(num_proposals)
    _fill(portfolio, hyps, pool, ctx)
    if allow_reshape and ctx.dims:
        _reshape_moves(portfolio, ctx)
    counts = collections.Counter(c.hypothesis for c in portfolio.picks)
    for hyp in hyps:
        hyp.slots = counts.get(hyp.name, 0)
    proposals = sorted(
        portfolio.picks,
        key=lambda c: (
            c.oom_risk,
            c.gain_pct < -_REGRESSION_PCT,
            c.hypothesis in (LAYOUT, PIPELINE),
            -c.gain_pct,
        ),
    )
    group = _routing_group(groups)
    entries = []
    for candidate in proposals:
        entry = _proposal_dict(candidate)
        entry["torus_mapping"] = torus_mapping(
            candidate.mesh,
            candidate.topology,
            group,
            partition_config=partition_config,
        )
        entry["legality"] = _legality_text(candidate, ctx)
        entries.append(entry)
    base_mapping = torus_mapping(base_full,
                                 ctx.topology,
                                 group,
                                 partition_config=partition_config)
    origin = _origin_note(base_full, ctx.dims)
    if base_mapping and origin:
        base_mapping += f" ({origin})"
    return {
        "estimate_type": "analytical_volume_ranking",
        "model_assumptions": {
            "shared_torus_dim_penalty":
            _SHARED_DIM_PENALTY,
            "all_to_all_ring_mean_hops":
            "L / 4 per torus side (wrap-around ring assumed)",
            "memory_caution_ratio":
            _MEMORY_CAUTION_RATIO,
            "memory_risk_ratio":
            _MEMORY_RISK_RATIO,
            "oom_hbm_fraction":
            _OOM_FRACTION,
            "memory_rule":
            ("With a memory profile: OOM risk when the estimated peak"
             " exceeds max(oom_hbm_fraction of HBM, the measured baseline"
             " peak). Without one: caution at >= memory_caution_ratio and"
             " OOM risk at >= memory_risk_ratio times the baseline local"
             " tokens or weight bytes (> when token counts are only an"
             " upper bound)."),
        },
        "baseline": {
            "mesh": base_full,
            "topology": ctx.topology,
            "num_slices": ctx.num_slices,
            "batch_size": ctx.batch_size,
            "collective_pct": (breakdown or {}).get("collective_pct"),
            "token_axes": list(ctx.token_axes),
            "fsdp_axes": list(ctx.fsdp_axes),
            "torus_mapping": base_mapping,
        },
        "hypotheses": [dataclasses.asdict(h) for h in hyps],
        "proposals": entries,
        "notes": notes,
    }


def _proposal_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "mesh":
        candidate.mesh,
        "moves": [str(move) for move in candidate.moves],
        "topology":
        candidate.topology,
        "num_slices":
        candidate.num_slices,
        "batch_size":
        candidate.batch_size,
        "change":
        candidate.change,
        "bracket":
        candidate.bracket,
        "hypothesis":
        candidate.hypothesis,
        "role":
        candidate.role,
        "estimated_gain_pct":
        round(candidate.gain_pct, 2),
        "current_shape_gain_pct":
        (None if candidate.current_shape_gain_pct is None else round(
            candidate.current_shape_gain_pct, 2)),
        "effects": [{
            "group": label,
            "pct": round(pct, 2),
            "scale": round(scale, 3)
        } for label, pct, scale in candidate.effects[:4]],
        "local_tokens_ratio":
        round(candidate.local_tokens_ratio, 3),
        "weight_bytes_ratio":
        round(candidate.weight_bytes_ratio, 3),
        "tensor_shards_base":
        candidate.tensor_shards_base,
        "tensor_shards_new":
        candidate.tensor_shards_new,
        "tokens_upper_bound":
        candidate.tokens_upper_bound,
        "weights_assumed_fsdp":
        candidate.weights_assumed_fsdp,
        "allow_split_physical_axes":
        candidate.allow_split_physical_axes,
        "memory_fraction": (None if candidate.memory_fraction is None else
                            round(candidate.memory_fraction, 3)),
        "oom_risk":
        candidate.oom_risk,
        "warnings":
        list(candidate.warnings),
    }


def _mesh_text(mesh: Mapping[str, int]) -> str:
    return "{" + ", ".join(f"'{k}': {v}" for k, v in mesh.items()) + "}"


def _why(proposal: Mapping[str, Any]) -> str:
    """One-line justification from the proposal's effects and memory."""
    parts = []
    for effect in proposal["effects"][:3]:
        scale = effect["scale"]
        verb = "removes" if scale == 0 else (
            "shrinks" if scale < 1 else "grows")
        amount = "" if scale == 0 else f" x{scale:.2f}"
        parts.append(
            f"{verb} {effect['group']} ({effect['pct']:.1f}%){amount}")
    if not parts:
        parts.append("placement change; neutral by the volume model")
    if proposal.get("current_shape_gain_pct") is not None:
        parts.append("on the current slice shape the same mesh estimates"
                     f" {proposal['current_shape_gain_pct']:+.1f}%")
    bound = "up to " if proposal.get("tokens_upper_bound") else ""
    weights_note = (" (assuming FSDP)"
                    if proposal.get("weights_assumed_fsdp") else "")
    memory = (
        f"local tokens {bound}x{proposal['local_tokens_ratio']:g}, resident"
        f" weights x{proposal['weight_bytes_ratio']:g}{weights_note}")
    t0 = int(proposal.get("tensor_shards_base", 1) or 1)
    t1 = int(proposal.get("tensor_shards_new", 1) or 1)
    if t1 < t0:
        memory += (
            f", tensor shards {t0}->{t1} (buffers that only tensor axes shard grow"
            f" x{t0 / t1:g})")
    if proposal["memory_fraction"] is not None:
        memory += f", est. peak {100 * proposal['memory_fraction']:.0f}% of HBM"
    elif (max(proposal["local_tokens_ratio"], proposal["weight_bytes_ratio"])
          >= _MEMORY_CAUTION_RATIO):
        memory += " (check memory)"
    if proposal["oom_risk"]:
        memory += " (OOM risk)"
    if proposal.get("allow_split_physical_axes"):
        memory += "; allow_split_physical_axes=True"
    text = "; ".join(parts) + f". Memory: {memory}."
    if proposal.get("warnings"):
        text += " Warnings: " + "; ".join(proposal["warnings"]) + "."
    return text


def format_portfolio_markdown(result: Mapping[str, Any]) -> str:
    """Renders `build_portfolio` output as Markdown for the agent."""
    base = result["baseline"]
    header = (f"Baseline mesh: {_mesh_text(base['mesh'])} on"
              f" {base.get('topology') or 'unknown topology'} x"
              f" {base.get('num_slices', 1)} slice(s)")
    if base.get("collective_pct") is not None:
        header += f"; collectives {base['collective_pct']:.1f}% of profiled time"
    lines = [
        "# Ranked mesh portfolio (analytical estimate, no simulation)",
        "",
        header,
        (f"Token axes: {', '.join(base.get('token_axes') or []) or '-'};"
         " weight (FSDP) axes:"
         f" {', '.join(base.get('fsdp_axes') or []) or '-'}"),
    ]
    if base.get("torus_mapping"):
        lines.append(f"Baseline torus mapping: {base['torus_mapping']}.")
    lines += [
        "",
        "Hypotheses (share of the step the evidence attributes):",
        "",
        "| hypothesis | % of step | axes | proposals | evidence |",
        "|---|---|---|---|---|",
    ]
    for hyp in result["hypotheses"]:
        title = HYPOTHESIS_TITLES.get(hyp["name"], hyp["name"])
        evidence = "; ".join(hyp["evidence"][:3]) or "-"
        lines.append(
            f"| {title} | {hyp['score']:.1f} | {', '.join(hyp['axes']) or '-'} |"
            f" {hyp['slots']} | {evidence} |")
    lines += [
        "",
        ("Proposals, ranked by estimated step-time gain (% of the baseline"
         " step; the volume model ranks, it does not predict):"),
        ("Model assumptions (uncalibrated ranking priors, not predictions):"
         f" shared torus-dim factor {_SHARED_DIM_PENALTY:g}x, all-to-all mean"
         " hops L/4 per side. Memory: with a profile, OOM risk when the"
         f" estimated peak exceeds max({100 * _OOM_FRACTION:.0f}% of HBM, the"
         " measured peak); without one, caution at"
         f" >={_MEMORY_CAUTION_RATIO:g}x and OOM risk at"
         f" >={_MEMORY_RISK_RATIO:g}x the baseline local tokens or weight"
         f" bytes (>{_MEMORY_RISK_RATIO:g}x when token counts are an upper"
         " bound)."),
        "",
        ("| # | mesh | topology x slices | change | hypothesis (slot) | est."
         " gain | why |"),
        "|---|---|---|---|---|---|---|",
    ]
    for index, proposal in enumerate(result["proposals"], 1):
        change = "; ".join(proposal["moves"]) or proposal["bracket"] or "-"
        if proposal["change"] != "move" and proposal["moves"]:
            change = f"{proposal['bracket']}; {change}"
        title = HYPOTHESIS_TITLES.get(proposal["hypothesis"],
                                      proposal["hypothesis"])
        lines.append(
            f"| {index} | {_mesh_text(proposal['mesh'])} |"
            f" {proposal['topology'] or '-'} x {proposal['num_slices']} |"
            f" {change} | {title} ({proposal['role']}) |"
            f" {proposal['estimated_gain_pct']:+.1f}% | {_why(proposal)} |")
    if any(
            p.get("torus_mapping") or p.get("legality")
            for p in result["proposals"]):
        lines += [
            "",
            ("Per proposal: torus mapping (best placement of the axes; the"
             " trainer's partition config decides the real one) and legality:"
             ),
            "",
        ]
        for index, proposal in enumerate(result["proposals"], 1):
            details = []
            if proposal.get("torus_mapping"):
                details.append(f"- Torus: {proposal['torus_mapping']}.")
            if proposal.get("legality"):
                details.append(f"- Legality: {proposal['legality']}.")
            lines.append(
                f"{index}. {_mesh_text(proposal['mesh'])} on"
                f" {proposal['topology'] or '-'} x {proposal['num_slices']} " +
                " ".join(details))
    for note in result.get("notes") or []:
        lines.append(f"\nNote: {note}")
    lines.append(
        "\nEvery hypothesis with evidence gets a proposal and the strongest"
        " move is bracketed at both ends. Replace a proposal only with evidence"
        " this portfolio lacks, and say why.")
    return "\n".join(lines)
