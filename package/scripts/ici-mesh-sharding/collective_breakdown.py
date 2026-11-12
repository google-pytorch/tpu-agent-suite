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
"""Groups HLO op-profile collective time by the mesh axes it runs over.

Turns per-op HLO profile records (`xprof_parsers.parse_hlo_op_profile`) into a
table of collective and async-wait time keyed by the evidence that identifies
the mesh axes involved:

*   kernel `mesh_axes` metadata (SparseCore / Pallas kernels such as MoE
    ragged all-to-all routing),
*   named replica groups (`replica_groups=mesh['data'=8,'seq'=4] {'seq'}`),
*   the group size of iota (`[32,8]<=[256]`) or explicit (`{{0,1,...}}`)
    replica groups, and
*   the gather / scatter factor between operand and result shapes.

Sizes are matched against the baseline mesh to name the axis (or product of
axes) a collective spans. A size that several axis subsets share is reported
as ambiguous rather than guessed.

Roles (MoE routing, KV gather, FSDP weights, ...) come from op names, source
files and kernel metadata. Given the sequence length, gathers and scatters that
none of these explain may get a role inferred from their shapes, marked as
such (`_shape_role`).
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import itertools
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mesh_search  # noqa: E402

_KIND_PATTERNS = (
    (
        "ragged-all-to-all",
        re.compile(r"ragged[_-]?all[_-]?to[_-]?all|ra2a", re.I),
    ),
    (
        "reduce-scatter",
        re.compile(r"reduce[_-]scatter|psum[_-]?scatter", re.I),
    ),
    ("all-reduce", re.compile(r"all[_-]reduce|psum", re.I)),
    ("all-gather", re.compile(r"all[_-]gather", re.I)),
    ("all-to-all", re.compile(r"all[_-]to[_-]all", re.I)),
    ("collective-permute", re.compile(r"collective[_-]permute|ppermute",
                                      re.I)),
    (
        "send/recv",
        re.compile(r"(?:^|[^a-z])(?:send|recv)(?:-done|-start)?(?:$|[^a-z])",
                   re.I),
    ),
    ("barrier", re.compile(r"barrier_semaphore", re.I)),
    ("mpmd_map", re.compile(r"mpmd_map", re.I)),
    ("async-collective", re.compile(r"async[_-]collective", re.I)),
)
_COLLECTIVE_CATEGORIES = frozenset({
    "all-reduce",
    "all-gather",
    "reduce-scatter",
    "all-to-all",
    "all-reduce-scatter fusion",
    "collective-permute",
    "collective-permute-done",
    "async-done",
    "send",
    "recv",
    "send-done",
    "recv-done",
})
_MESH_AXES_RE = re.compile(r"mesh_axes[\\\"']*\s*:\s*[\\\"']*\[([^\]]*)\]")
_NAMED_GROUPS_RE = re.compile(r"replica_groups=mesh\[([^\]]*)\]\s*\{([^}]*)\}")
_IOTA_GROUPS_RE = re.compile(
    r"replica_groups=\[([\d,]+)\]<=\[([\d,]+)\](?:T\(([\d,]+)\))?")
_EXPLICIT_GROUPS_RE = re.compile(r"replica_groups=\{\{([\d,\s]+)\}")
_SHAPE_RE = re.compile(r"[a-z]+\d*\[([\d,]*)\]")
_DTYPE_RE = re.compile(r"(?<![\w.])([a-z][a-z0-9]*)\[[\d,]*\]")
_FLOAT_DTYPE_RE = re.compile(r"(?:bf|f)\d")
_CHANGE_RE = re.compile(r"d\d+:(\d+)->(\d+)")
_NAME_RE = re.compile(r"[A-Za-z_]\w*")
_POSITIONAL_AXIS_RE = re.compile(r"^axis_\d+$")


@dataclasses.dataclass
class Evidence:
    """What one op reveals about the mesh axes it spans."""

    kind: str
    source: str
    axes: tuple[str, ...] = ()  # Exact axes (kernel metadata / named groups).
    group_size: int = 0  # From replica groups or the gather factor.
    detail: str = ""  # Human-readable evidence string.
    # Dimension a gather / scatter changes and the operand rank (-1 / 0 when
    # unknown); used to break ties between same-size axes of different families.
    dim: int = -1
    rank: int = 0


def collective_kind_from_text(text: str) -> str | None:
    """Returns the collective family named in an op name or HLO category."""
    for kind, pattern in _KIND_PATTERNS:
        if pattern.search(text):
            return kind
    return None


def collective_kind(op: Mapping[str, Any]) -> str | None:
    """Returns the collective family of an op, or None if it is not one."""
    name = str(op.get("operation", ""))
    if name.lower().startswith("copy"):
        return None
    kind = collective_kind_from_text(name)
    if kind:
        return kind
    category = str(op.get("category", "")).lower()
    if category in _COLLECTIVE_CATEGORIES:
        kind = collective_kind_from_text(category)
        if kind:
            return kind
        return "async-wait" if category == "async-done" else category
    return None


def _shape_change(expression: str) -> tuple[int, int, int, str]:
    """Returns (factor, dim, operand rank, 'dN:a->b') between operand and result."""
    if "=" not in expression:
        return 0, -1, 0, ""
    rhs = expression.split("=", 1)[1]
    result = _SHAPE_RE.search(rhs)
    args = rhs[rhs.find("(") + 1:] if "(" in rhs else ""
    operand = _SHAPE_RE.search(args)
    if not result or not operand:
        return 0, -1, 0, ""
    a = [int(x) for x in operand.group(1).split(",") if x]
    b = [int(x) for x in result.group(1).split(",") if x]
    if len(a) != len(b):
        return 0, -1, 0, ""
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b))
             if x != y and x and y]
    if len(diffs) != 1:
        return 0, -1, 0, ""
    i, x, y = diffs[0]
    big, small = max(x, y), min(x, y)
    if big % small:
        return 0, -1, 0, ""
    return big // small, i, len(a), f"d{i}:{x}->{y}"


def _operand_dtype(expression: str) -> str:
    """Returns the element type of the first operand, e.g. 'bf16', or ''."""
    rhs = expression.split("=", 1)[1] if "=" in expression else ""
    if "(" not in rhs:
        return ""
    m = _DTYPE_RE.search(rhs[rhs.find("(") + 1:])
    return m.group(1) if m else ""


def _family_of_dim(dim: int, rank: int) -> str | None:
    """Returns the axis family that conventionally shards activation dim `dim`."""
    if rank == 2:
        return "tensor" if dim == 1 else None
    if rank not in (3, 4) or not 0 <= dim <= 2:
        return None
    return ("batch", "sequence", "tensor")[dim]


def device_order(ops: Sequence[Mapping[str, Any]]) -> list[tuple[str, int]]:
    """Returns the device mesh axes in device order, read from the profile."""
    for op in ops:
        expression = op.get("expression")
        if not isinstance(expression, str):
            continue
        m = _NAMED_GROUPS_RE.search(expression)
        if not m:
            continue
        pairs = [(k, int(v))
                 for k, v in re.findall(r"'?(\w+)'?\s*=\s*(\d+)", m.group(1))]
        if pairs and not any(_POSITIONAL_AXIS_RE.match(k) for k, _ in pairs):
            return pairs
    return []


def _coords(device: int, sizes: Sequence[int]) -> list[int]:
    out = []
    for size in reversed(sizes):
        out.append(device % size)
        device //= size
    return out[::-1]


def _axes_of_members(members: Sequence[int],
                     order: Sequence[tuple[str, int]]) -> tuple[str, ...]:
    """Returns the axes along which one replica group's devices vary."""
    sizes = [s for _, s in order]
    total = math.prod(sizes)
    if not members or max(members) >= total:
        return ()
    coords = [_coords(d, sizes) for d in members]
    varying = tuple(order[i][0] for i in range(len(sizes))
                    if len({c[i]
                            for c in coords}) > 1)
    if math.prod(dict(order)[a] for a in varying) != len(members):
        return ()
    return varying


def _iota_first_group(text: str) -> list[int]:
    """Returns group 0 of an iota replica group `[G,S]<=[dims]T(perm)`."""
    m = _IOTA_GROUPS_RE.search(text)
    if not m:
        return []
    shape = [int(x) for x in m.group(1).split(",") if x]
    dims = [int(x) for x in m.group(2).split(",") if x]
    perm = ([int(x) for x in m.group(3).split(",")
             if x] if m.group(3) else list(range(len(dims))))
    if not shape or len(perm) != len(dims):
        return []
    group_size = shape[-1]
    permuted = [dims[p] for p in perm]
    members = []
    for flat in range(group_size):
        idx_t = _coords(flat, permuted)
        idx = [0] * len(dims)
        for t, p in enumerate(perm):
            idx[p] = idx_t[t]
        device = 0
        for i, d in zip(idx, dims):
            device = device * d + i
        members.append(device)
    return members


def _positional_axes(
    sizes: Mapping[str, int],
    chosen: Sequence[str],
    order: Sequence[tuple[str, int]],
) -> tuple[str, ...]:
    """Maps positional mesh axes (axis_0, ...) onto runs of real axes."""
    real = [(a, s) for a, s in order if s > 1]
    runs, i = {}, 0
    for name, size in sizes.items():
        run, prod = [], 1
        while prod < size and i < len(real):
            run.append(real[i][0])
            prod *= real[i][1]
            i += 1
        if prod != size:
            return ()
        runs[name] = run
    if i != len(real):
        return ()
    out = []
    for name in chosen:
        out.extend(runs.get(name, ()))
    return tuple(out)


def extract_evidence(
        op: Mapping[str, Any],
        kind: str,
        order: Sequence[tuple[str, int]] = (),
) -> Evidence:
    """Extracts axis evidence from an op's HLO expression and metadata."""
    expression = op.get("expression")
    expression = expression if isinstance(expression, str) else ""
    source = os.path.basename(str(op.get("source_file") or "")) or "-"
    if source in ("N/A", "A"):
        source = "-"
    ev = Evidence(kind=kind, source=source)

    m = _MESH_AXES_RE.search(expression)
    if m:
        axes = tuple(_NAME_RE.findall(m.group(1)))
        if axes:
            ev.axes = axes
            ev.detail = "kernel mesh_axes=" + ",".join(axes)
            return ev

    m = _NAMED_GROUPS_RE.search(expression)
    if m:
        sizes = dict(
            (k, int(v))
            for k, v in re.findall(r"'?(\w+)'?\s*=\s*(\d+)", m.group(1)))
        chosen = tuple(_NAME_RE.findall(m.group(2)))
        if chosen and not any(_POSITIONAL_AXIS_RE.match(a) for a in chosen):
            ev.axes = chosen
            ev.detail = "groups over " + ",".join(chosen)
            return ev
        if chosen:
            ev.group_size = math.prod(sizes.get(a, 1) for a in chosen)
            axes = _positional_axes(sizes, chosen, order) if order else ()
            if axes:
                ev.axes = axes
                ev.detail = f"groups over {','.join(chosen)} = {','.join(axes)}"
            else:
                ev.detail = f"group size {ev.group_size}"
            return ev

    members = _iota_first_group(expression)
    if not members:
        m = _EXPLICIT_GROUPS_RE.search(expression)
        if m:
            members = [
                int(x) for x in m.group(1).replace(" ", "").split(",") if x
            ]
    if members:
        ev.group_size = len(members)
        axes = _axes_of_members(members, order) if order else ()
        if axes:
            ev.axes = axes
            ev.detail = f"group of {len(members)} devices varies {','.join(axes)}"
        else:
            ev.detail = f"group size {ev.group_size}"
        return ev

    if kind in ("all-gather", "reduce-scatter", "all-to-all"):
        factor, dim, rank, text = _shape_change(expression)
        if factor > 1:
            ev.group_size = factor
            ev.dim, ev.rank = dim, rank
            ev.detail = f"x{factor} ({text})"
    return ev


def parse_mesh(text: str | None) -> dict[str, int] | None:
    """Parses a `--mesh` value with `mesh_search.parse_mesh`."""
    if not text:
        return None
    return mesh_search.parse_mesh(text)


def axes_for_size(size: int, mesh: Mapping[str, int]) -> list[tuple[str, ...]]:
    """Returns the axis subsets (up to 3 axes) of `mesh` whose product is size."""
    axes = [a for a, n in mesh.items() if n > 1]
    out = []
    for r in range(1, 4):
        for combo in itertools.combinations(axes, r):
            if math.prod(mesh[a] for a in combo) == size:
                out.append(combo)
        if out:
            break
    return out


def _label(ev: Evidence,
           sizes: Mapping[str, int] | None) -> tuple[str, tuple[str, ...]]:
    """Returns (axes label, exact axes) for an op's evidence."""
    if ev.axes:
        axes = ev.axes
        if sizes:
            axes = tuple(a for a in axes if sizes.get(a, 1) > 1)
            if not axes:
                return "size-1 axes only", ()
            active = {a for a, n in sizes.items() if n > 1}
            if len(axes) > 1 and set(axes) >= active:
                return "all axes (global)", axes
        return " x ".join(axes), axes
    if ev.group_size and sizes:
        if ev.group_size == math.prod(sizes.values()):
            active = tuple(a for a, n in sizes.items() if n > 1)
            return "all axes (global)", active
        subsets = axes_for_size(ev.group_size, sizes)
        if len(subsets) == 1:
            return " x ".join(subsets[0]), subsets[0]
        if subsets:
            family = (None if ev.source.startswith(_WEIGHT_FILES) else
                      _family_of_dim(ev.dim, ev.rank))
            matching = [
                subset for subset in subsets if family and all(
                    mesh_search.axis_family(axis) == family for axis in subset)
            ]
            if len(matching) == 1:
                return f"{' x '.join(matching[0])} (from d{ev.dim})", matching[
                    0]
            return "ambiguous: " + " or ".join(" x ".join(s)
                                               for s in subsets), ()
        return f"no axis product = {ev.group_size}", ()
    return "unattributed", ()


@functools.lru_cache(maxsize=4096)
def _all_axes_for_size_cached(
        size: int, mesh_items: tuple[tuple[str, int],
                                     ...]) -> tuple[tuple[str, ...], ...]:
    axes = [a for a, n in mesh_items if n > 1]
    sizes = dict(mesh_items)
    out = []
    for r in range(1, len(axes) + 1):
        for combo in itertools.combinations(axes, r):
            if math.prod(sizes[a] for a in combo) == size:
                out.append(combo)
    return tuple(out)


def _all_axes_for_size(size: int, mesh: Mapping[str,
                                                int]) -> list[tuple[str, ...]]:
    """Returns every subset of the axes > 1 whose sizes multiply to `size`."""
    return list(
        _all_axes_for_size_cached(size,
                                  tuple((a, int(n)) for a, n in mesh.items())))


def _candidate_axes(
    ev: Evidence,
    sizes: Mapping[str, int] | None,
    exact_axes: tuple[str, ...],
) -> list[tuple[str, ...]]:
    """Returns the axis subsets a collective's group may span."""
    if ev.axes:
        return [exact_axes] if exact_axes else []
    if ev.group_size and sizes:
        if ev.group_size == math.prod(sizes.values()):
            return [exact_axes] if exact_axes else []
        return _all_axes_for_size(ev.group_size, sizes)
    return []


def _is_routing(op: Mapping[str, Any], kind: str) -> bool:
    text = " ".join(
        str(op.get(k, "")) for k in ("operation", "provenance", "source_file"))
    return kind == "ragged-all-to-all" or "ragged_all_to_all" in text


ROLE_ROUTING = "MoE routing"
ROLE_KV = "KV gather (sequence)"
ROLE_WEIGHTS = "weights: FSDP gather / gradient"
ROLE_MATMUL = "tensor: collective matmul"
ROLE_GLOBAL = "global (all devices)"
ROLE_RING = "ring / permute"
ROLE_HOST = "host offload"
ROLE_OTHER = "activations / other"
# Roles a move between mesh axes does not shrink: weight gathers and gradient
# reductions scale with weight bytes / tensor shards, global reductions span
# every device, and host transfers are not collectives.
_NOT_PER_AXIS = frozenset({ROLE_WEIGHTS, ROLE_GLOBAL, ROLE_HOST})
_ROUTING_FILES = (
    "ragged_all_to_all",
    "scheduling_groups",
    "sparse_a2a",
    "dense_a2a",
    "fused_moe",
    "moe_routing",
    "token_dispatcher",
    "megablox",
    "gmm",
)
_WEIGHT_FILES = (
    "fsdp_utils",
    "fsdp",
    "fully_sharded",
    "zero_",
    "layer_stack",
    "rmsprop",
    "adam",
    "optimizer",
    "optax",
    "maxtext_fsdp",
)
_HOST_RE = re.compile(r"(?:^|[^a-z])host[-_]")
# Appended to a role inferred from a gather / scatter's shapes (`_shape_role`)
# instead of an op name, source file or kernel metadata.
BY_SHAPE = " (by shape)"
# Shares below this print as 0.0% (or 0.00%); lines for them are left out.
MIN_PRINTED_PCT = 0.05


def _shape_role(
    expression: str,
    kind: str,
    candidates: Sequence[tuple[str, ...]],
    sizes: Mapping[str, int] | None,
    sequence_length: int | None,
) -> str | None:
    """Returns the role a gather / scatter's shapes imply, or None."""
    if (kind not in ("all-gather", "reduce-scatter") or not sequence_length
            or not candidates or not sizes):
        return None
    factor, dim, rank, text = _shape_change(expression)
    change = _CHANGE_RE.fullmatch(text)
    if (factor <= 1 or not change
            or not _FLOAT_DTYPE_RE.match(_operand_dtype(expression))):
        return None
    families = [{mesh_search.axis_family(axis)
                 for axis in subset} for subset in candidates]
    if any("expert" in f for f in families):
        return None
    if _family_of_dim(dim, rank) == "sequence":
        chunk = min(int(change.group(1)), int(change.group(2)))
        sequence_shards = math.prod(
            n for a, n in sizes.items()
            if mesh_search.axis_family(a) == "sequence")
        sequence_only = any(
            f == {"sequence"} and math.prod(sizes.get(a, 1)
                                            for a in subset) == factor
            for f, subset in zip(families, candidates))
        if sequence_only and chunk * sequence_shards == sequence_length:
            return ROLE_KV
        return None
    first_weight_dim = 1 if rank == 2 else 2
    if (rank >= 2 and dim >= first_weight_dim
            and all(f & {"batch", "sequence"} and "tensor" not in f
                    for f in families)):
        return ROLE_WEIGHTS
    return None


def classify_role(op: Mapping[str, Any], kind: str, label: str) -> str:
    """Returns what a collective is for, from its name, kind and source file."""
    name = str(op.get("operation", "")).lower()
    source = os.path.basename(str(op.get("source_file") or "")).lower()
    if _HOST_RE.search(name):
        return ROLE_HOST
    if (_is_routing(op, kind) or "barrier_semaphore" in name
            or source.startswith(_ROUTING_FILES)):
        return ROLE_ROUTING
    if "k_hbm_ref" in name or "v_hbm_ref" in name:
        return ROLE_KV
    if source.startswith(_WEIGHT_FILES):
        return ROLE_WEIGHTS
    if "matmul" in name:
        return ROLE_MATMUL
    if label == "all axes (global)":
        return ROLE_GLOBAL
    if kind == "collective-permute" or "permute" in name:
        return ROLE_RING
    return ROLE_OTHER


def _is_mesh_reducible(group: Mapping[str, Any]) -> bool:
    """True for a group the per-axis reducible-time table charges to its axes."""
    return (bool(group["exact_axes"]) and group["axes"] != "all axes (global)"
            and group["role"] not in _NOT_PER_AXIS)


def display_role(group: Mapping[str, Any]) -> str:
    """A group's role, marked when it was inferred from shapes."""
    return str(group["role"]) + (BY_SHAPE if group.get("by_shape") else "")


def _largest_reducible(
        rows: Sequence[Mapping[str, Any]],
        sizes: Mapping[str, int] | None) -> dict[str, Any] | None:
    """The largest mesh-reducible collective group, summed over its rows."""
    totals: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for g in rows:
        if not _is_mesh_reducible(g):
            continue
        exact_axes = tuple(g["exact_axes"])
        total = totals.get((g["role"], exact_axes))
        if total is None:
            new_total: dict[str, Any] = {
                "role":
                g["role"],
                "axes":
                " x ".join(exact_axes),
                "exact_axes":
                exact_axes,
                "devices":
                (math.prod(int(sizes.get(a, 1))
                           for a in exact_axes) if sizes else 0),
                "pct":
                0.0,
                "by_shape_pct":
                0.0,
                "rows":
                0,
                "kinds": [],
                "kind":
                g["kind"],
                "evidence":
                g["evidence"],
                "example":
                g["example"],
            }
            totals[(g["role"], exact_axes)] = total = new_total
        total["pct"] += g["pct"]
        if g.get("by_shape"):
            total["by_shape_pct"] += g["pct"]
        total["rows"] += 1
        if g["kind"] not in total["kinds"]:
            total["kinds"].append(g["kind"])
    if not totals:
        return None
    return max(totals.values(), key=lambda t: t["pct"])


def summarize(
    ops: Sequence[Mapping[str, Any]],
    mesh: Mapping[str, int] | None = None,
    top_k: int = 25,
    sequence_length: int | None = None,
) -> dict[str, Any]:
    """Aggregates collective time by evidence, by role and by mesh axis.

    Args:
      ops: Op dicts from `xprof_parsers.parse_hlo_op_profile`.
      mesh: Baseline ICI mesh (axis -> size).
      top_k: Number of groups to keep, by time (0 keeps all).
      sequence_length: Sequence length of the workload, if known; enables
        `_shape_role` inference on unlabeled gathers / scatters.
    """
    total = sum(float(op.get("raw_time_ps", 0.0)) for op in ops) or 1.0
    order = device_order(ops)
    sizes = dict(mesh) if mesh else (dict(order) if order else None)
    mismatch = False
    if order and mesh:
        mismatch = {
            a: n
            for a, n in order if n > 1
        } != {
            a: n
            for a, n in mesh.items() if n > 1
        }
    items = []
    for op in ops:
        kind = collective_kind(op)
        if kind is not None:
            items.append((op, kind, extract_evidence(op, kind, order)))

    routing_axes = {
        tuple(a for a in ev.axes if not sizes or sizes.get(a, 1) > 1)
        for op, kind, ev in items if ev.axes and _is_routing(op, kind)
    }
    if len(routing_axes) == 1:
        (axes, ) = routing_axes
        for op, kind, ev in items:
            if not ev.axes and not ev.group_size and _is_routing(op, kind):
                ev.axes = axes
                ev.detail = "same routing kernel as mesh_axes=" + ",".join(
                    axes)

    groups: collections.OrderedDict[Any,
                                    dict[str,
                                         Any]] = collections.OrderedDict()
    for op, kind, ev in items:
        label, exact_axes = _label(ev, sizes)
        role = classify_role(op, kind, label)
        by_shape = False
        if role == ROLE_OTHER:
            expression = op.get("expression")
            inferred = _shape_role(
                expression if isinstance(expression, str) else "",
                kind,
                _candidate_axes(ev, sizes, exact_axes),
                sizes,
                sequence_length,
            )
            if inferred:
                role, by_shape = inferred, True
        key = (kind, ev.source, label, ev.detail, role, by_shape)
        g = groups.setdefault(
            key,
            {
                "kind": kind,
                "source": ev.source,
                "axes": label,
                "role": role,
                "by_shape": by_shape,
                "exact_axes": exact_axes,
                "evidence": ev.detail or "-",
                "time_ps": 0.0,
                "ops": 0,
                "occurrences": 0,
                "example": str(op.get("operation", ""))[:60],
            },
        )
        g["time_ps"] += float(op.get("raw_time_ps", 0.0))
        g["ops"] += 1
        g["occurrences"] += int(op.get("occurrences", 0))

    rows = sorted(groups.values(), key=lambda g: -g["time_ps"])
    for g in rows:
        g["pct"] = 100.0 * g["time_ps"] / total
    axes: collections.Counter[str] = collections.Counter()
    roles: collections.Counter[str] = collections.Counter()
    per_axis: collections.Counter[str] = collections.Counter()
    ambiguous = unattributed = 0.0
    for g in rows:
        roles[display_role(g)] += g["pct"]
        if g["exact_axes"]:
            axes[g["axes"]] += g["pct"]
            if _is_mesh_reducible(g):
                for a in g["exact_axes"]:
                    per_axis[a] += g["pct"]
        elif g["axes"].startswith(("ambiguous", "no axis")):
            ambiguous += g["pct"]
        else:
            unattributed += g["pct"]
    largest_reducible = _largest_reducible(rows, sizes)
    global_rows = [g for g in rows if g["axes"] == "all axes (global)"]
    global_summary = {
        "pct": sum(g["pct"] for g in global_rows),
        "occurrences": sum(g["occurrences"] for g in global_rows),
        "kinds": sorted({g["kind"]
                         for g in global_rows}),
        "sources":
        sorted({g["source"]
                for g in global_rows if g["source"] != "-"}),
        "devices": math.prod(sizes.values()) if sizes else 0,
    }
    tensor_ar_rows = [
        g for g in rows
        if g["axes"] != "all axes (global)" and g["role"] not in _NOT_PER_AXIS
        and g["role"] != ROLE_ROUTING and (g["role"] == ROLE_MATMUL or (
            g["exact_axes"] and all(a in mesh_search.TENSOR_AXES
                                    for a in g["exact_axes"]) and g["kind"] in
            ("all-reduce", "reduce-scatter", "all-gather")))
    ]
    by_kind: collections.Counter[str] = collections.Counter()
    tensor_axes: set[str] = set()
    for g in tensor_ar_rows:
        by_kind[g["kind"]] += g["pct"]
        tensor_axes.update(g["exact_axes"])
    if not tensor_axes and sizes:
        tensor_axes = {
            a
            for a, n in sizes.items() if n > 1 and a in mesh_search.TENSOR_AXES
        }
    tensor_ar_summary = {
        "pct": sum(g["pct"] for g in tensor_ar_rows),
        "axes": sorted(tensor_axes),
        "by_kind": dict(by_kind.most_common()),
    }
    return {
        "total_ms": total / 1e9,
        "collective_pct": sum(g["pct"] for g in rows),
        "groups": rows[:top_k] if top_k > 0 else rows,
        "axes": dict(axes.most_common()),
        "roles": dict(roles.most_common()),
        "per_axis": dict(per_axis.most_common()),
        "largest_reducible": largest_reducible,
        "ambiguous_pct": ambiguous,
        "unattributed_pct": unattributed,
        "global": global_summary,
        "tensor_ar": tensor_ar_summary,
        "device_order": order,
        "mesh_mismatch": mismatch,
        "sequence_length": sequence_length or None,
    }


def format_markdown(summary: Mapping[str, Any]) -> str:
    """Renders a summary from `summarize` as Markdown."""
    order = summary.get("device_order") or []
    lines = [
        (f"Collective and async-wait time: {summary['collective_pct']:.1f}% of"
         f" {summary['total_ms']:.1f} ms profiled program time."),
        "Device axis order (from named replica groups): " +
        (", ".join(a for a, _ in order) if order else
         "not in the profile; group sizes matched against --mesh"),
    ]
    if summary.get("mesh_mismatch"):
        lines.append(
            "Warning: the profile's replica groups imply other axis sizes than"
            " --mesh. Check that this session profiles the baseline mesh; axis"
            " names come from the profile, sizes from --mesh.")
    lines += [
        "",
        "Time by role:",
        "",
        "| role | % of profiled op time |",
        "|---|---|",
    ]
    for role, pct in summary.get("roles", {}).items():
        lines.append(f"| {role} | {pct:.1f}% |")
    global_summary = summary.get("global") or {}
    if global_summary.get("pct", 0.0) >= MIN_PRINTED_PCT:
        devices = global_summary.get("devices") or 0
        sources = ", ".join(global_summary.get("sources")
                            or []) or "unknown source"
        occ = int(global_summary.get("occurrences") or 0)
        lines += [
            "",
            (f"All-device collectives ({', '.join(global_summary['kinds'])}" +
             (f", groups of {devices} devices" if devices else "") +
             f"; {sources}): {global_summary['pct']:.1f}% of profiled op"
             f" time over {occ} occurrences."),
        ]
    tensor_ar = summary.get("tensor_ar") or {}
    if tensor_ar.get("pct", 0.0) >= MIN_PRINTED_PCT:
        axis_label = ", ".join(tensor_ar.get("axes") or ["model"])
        parts = [
            f"{pct:.2f}% {kind}"
            for kind, pct in (tensor_ar.get("by_kind") or {}).items()
        ]
        lines += [
            "",
            (f"Tensor-axis ({axis_label}) collectives: {tensor_ar['pct']:.2f}%"
             f" of profiled op time ({' + '.join(parts)}). Under sequence"
             " parallelism a per-layer tensor all-reduce runs as a"
             " reduce-scatter + all-gather pair, whose sum is then that"
             " all-reduce's time."),
        ]
    lines += [
        "",
        ("Reducible time per axis (routing, KV, ring, collective and"
         " activation groups that span the axis; weight gathers, gradient and"
         " global reductions and host transfers are excluded; rows overlap):"),
        "",
        "| axis | % of profiled op time |",
        "|---|---|",
    ]
    for axis, pct in summary.get("per_axis", {}).items():
        lines.append(f"| {axis} | {pct:.1f}% |")
    text = _largest_text(summary.get("largest_reducible"))
    if summary.get("ambiguous_pct", 0.0) >= MIN_PRINTED_PCT:
        text += (" Groups of ambiguous size are not included"
                 f" ({summary['ambiguous_pct']:.1f}% in total).")
    lines += ["", text]
    lines += [
        "",
        ("Time by axis group (exact; '(from dN)' rows resolve a same-size tie"
         " by the activation dimension the collective gathers or scatters:"
         " d0 batch, d1 sequence, d2 tensor):"),
        "",
        "| axes | % of profiled op time |",
        "|---|---|",
    ]
    for label, pct in summary["axes"].items():
        if pct >= MIN_PRINTED_PCT:
            lines.append(f"| {label} | {pct:.1f}% |")
    lines.append(f"| ambiguous size | {summary['ambiguous_pct']:.1f}% |")
    lines.append(f"| unattributed | {summary['unattributed_pct']:.1f}% |")
    lines += [
        "",
        "Groups (largest first):",
        "",
        ("| % of profiled op time | collective | role | source file | axes |"
         " evidence | ops | occurrences | example op |"),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for g in summary["groups"]:
        role = display_role(g) if g.get("role") else "-"
        lines.append(
            f"| {g['pct']:.2f}% | {g['kind']} | {role} |"
            f" {g['source']} | {g['axes']} | {g['evidence']} | {g['ops']} |"
            f" {g['occurrences']} | `{g['example']}` |")
    lines += [
        "",
        ("Kernel mesh_axes, named or decoded replica groups are exact. A size"
         " several axis subsets share is ambiguous: break the tie with the"
         " source file and role. Weight gathers and gradient reductions (FSDP)"
         " move about the same bytes per device whatever the size of their"
         " group, so moving factors between batch and sequence axes does not"
         " shrink them; more tensor or expert shards do. KV gathers shrink"
         " when sequence shards move to batch axes. Collective matmuls are"
         " tensor-parallel; their kernel metadata lists every axis."),
    ]
    sequence_length = summary.get("sequence_length")
    if sequence_length:
        lines.append(
            f"Roles marked '{BY_SHAPE.strip()}' are inferred from the shapes of"
            " gathers / scatters that no op name, source file or kernel metadata"
            " explains, assuming activations laid out [batch, sequence, ...] or"
            " [tokens, ...] and every sequence axis sharding the sequence: over"
            " sequence axes only, dimension 1 of a rank-3 or rank-4 tensor"
            f" growing from the per-device chunk of the {sequence_length}-token"
            " sequence is a KV gather; over batch and sequence axes only, a"
            " floating-point dimension that cannot hold tokens is an FSDP weight"
            " gather or gradient reduction. They are a heuristic: confirm them"
            " with the op's provenance before relying on them.")
    else:
        lines.append(
            "Pass --sequence_length to also infer KV and FSDP roles from the"
            " shapes of gathers / scatters that no op name or source file"
            " explains.")
    return "\n".join(lines)


def _largest_text(largest: Mapping[str, Any] | None) -> str:
    """Renders `_largest_reducible` output as one sentence."""
    if not largest:
        return "Largest mesh-reducible group: none."
    devices = f" ({largest['devices']} devices)" if largest.get(
        "devices") else ""
    rows = int(largest.get("rows") or 1)
    kinds = ", ".join(largest.get("kinds") or [largest["kind"]])
    text = (
        f"Largest mesh-reducible group: {largest['role']} over"
        f" {largest['axes']}{devices}: {largest['pct']:.1f}% of profiled op"
        f" time in {rows} row{'' if rows == 1 else 's'} ({kinds})")
    by_shape = float(largest.get("by_shape_pct") or 0.0)
    if by_shape >= MIN_PRINTED_PCT:
        text += f", {by_shape:.1f}% of it with roles inferred by shape"
    evidence = largest.get("evidence") or "-"
    detail = f" ({evidence})" if evidence != "-" else ""
    return (f"{text}; largest row: {largest['kind']}{detail}, e.g."
            f" `{largest['example']}`.")
