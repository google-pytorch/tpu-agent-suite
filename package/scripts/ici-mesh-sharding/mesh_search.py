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
"""Deterministic enumeration of legal ICI mesh candidates around a baseline.

Given a baseline ICI mesh, this module lists every mesh reachable by one (or
optionally a few) pairwise factor moves. A move divides a donor axis by a
power-of-two factor and multiplies a receiver axis by the same factor, so the
mesh product (the device count) is preserved, e.g. 'a/4 -> b x4'.
Candidates are filtered by hard legality constraints (powers of two; batch,
sequence and head divisibility) and annotated with cheap first-principles
features: per-shard batch, per-device activation tokens relative to the
baseline, tensor shards, KV-head warnings, axes that collapse to 1 and the
block layout on the physical slice (which groups of axes exactly fill whole
torus dims, optionally checking that a given axis group such as the MoE
routing axes fills one compact block).

Ranking of candidates is intentionally left to the caller (an LLM agent or a
human), who has the profile evidence about which collectives dominate the step
time. Candidates are returned in a fixed order (number of moves, then donor
axis in baseline order, then receiver axis, then factor ascending) and are
never sorted by a hidden score. `by_donor` groups them by the axis they shrink;
`bracketed_pairs` gives every single-move (donor, receiver) pair's bracket
points (first step, boundary and, for sibling axes, the size swap), so a
chosen move is proposed at both ends of its range instead of at one guessed
magnitude. Only `topology_fits` sorts meshes, by the documented block-layout
criteria.

Example:

  result = mesh_search.enumerate_mesh_candidates(
      {"data": 4, "model": 2, "seq": 8},
      batch_size=16,
      sequence_length=32768,
      num_heads=16,
      num_kv_heads=4,
      topology="4x4x4",
  )
  for index in result["by_donor"].get("seq", []):
    print(result["candidates"][index]["moves"])
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import itertools
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

# Axis families. Names follow TPU trainer mesh axes plus common OSS spellings.
BATCH_AXES = ("replica", "data", "data2", "fsdp", "dp")
SEQUENCE_AXES = ("seq", "seq1", "seq2", "seq3", "seq4", "sp", "pcp", "dcp",
                 "sequence", "context")
TENSOR_AXES = ("model", "model2", "model_q", "model_kv", "tp", "tensor")
# Tensor axes that shard the query-head dimension. `model2` is left out: in 2D
# tensor parallelism it can shard another dimension, so a head count it does
# not divide only gets a warning.
QUERY_HEAD_AXES = tuple(axis for axis in TENSOR_AXES if axis != "model2")
EXPERT_AXES = ("expert", "ep")

# Axes pinned by default: `intrachip` is the TensorCore factor inside a chip,
# `replica` is the data-parallel axis across slices (DCN) and `stage` / `pp`
# are pipeline stages.
DEFAULT_FIXED_AXES = ("intrachip", "replica", "stage", "pp")

# Axes that are not laid out on inter-chip ICI links (see `torus_fit`).
NON_ICI_AXES = ("intrachip", "replica")

# Largest supported `max_moves`; the number of candidates grows quickly.
MAX_MOVES_LIMIT = 3

_AXIS_FAMILIES = (
    ("batch", BATCH_AXES),
    ("sequence", SEQUENCE_AXES),
    ("tensor", TENSOR_AXES),
    ("expert", EXPERT_AXES),
)
_CONTIGUOUS_TOPOLOGY_RE = re.compile(r"\d+(?:[x,]\d+)+")
_TOPOLOGY_RE = re.compile(r"\d+(?:\s*[x,]\s*\d+)+")
_AXIS_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MESH_PAIR_RE = re.compile(r"['\"]?([A-Za-z_]\w*)['\"]?\s*[:=]\s*(\d+)")

# (axis, size) pairs and (axis, owned dim sizes) pairs used by `torus_fit`.
_AxisSizes = tuple[tuple[str, int], ...]
_WholeDims = tuple[tuple[str, tuple[int, ...]], ...]

# A block is (sizes of the whole physical dims it covers, mesh axes laid out on
# it); a layout tiles every physical dim of the slice with blocks.
_Block = tuple[tuple[int, ...], tuple[str, ...]]
_Layout = tuple[_Block, ...]

# Preference for the block holding a co-located axis group: a whole 2D plane
# (most links and bisection per device), then a whole ring, then the slice.
_GROUP_BLOCK_RANK = {2: 0, 1: 1}
_NO_GROUP_BLOCK_RANK = 3


@dataclasses.dataclass(frozen=True)
class _Move:
    """Divides `donor` by `factor` and multiplies `receiver` by `factor`."""

    donor: str
    receiver: str
    factor: int

    def __str__(self) -> str:
        return f"{self.donor}/{self.factor} -> {self.receiver} x{self.factor}"


@dataclasses.dataclass(frozen=True)
class _Workload:
    """Optional workload dimensions that constrain which meshes are legal."""

    batch_size: int | None = None
    sequence_length: int | None = None
    num_heads: int | None = None
    num_kv_heads: int | None = None


def axis_family(axis: str) -> str:
    """Returns the family of a mesh axis.

    Args:
      axis: Mesh axis name, e.g. 'seq2'.

    Returns:
      One of 'batch', 'sequence', 'tensor', 'expert' or 'other'.
    """
    for family, axes in _AXIS_FAMILIES:
        if axis in axes:
            return family
    return "other"


def _is_power_of_two(value: int) -> bool:
    return value >= 1 and value & (value - 1) == 0


def _shards(mesh: Mapping[str, int], axes: Sequence[str]) -> int:
    return math.prod(mesh.get(axis, 1) for axis in axes)


def _kv_head_shards(mesh: Mapping[str, int]) -> int:
    """Shards of the KV-head dimension: every tensor axis except `model_q`."""
    return _shards(mesh, [axis for axis in TENSOR_AXES if axis != "model_q"])


def _token_shards(mesh: Mapping[str, int]) -> int:
    """Number of ways the batch x sequence activation tokens are split."""
    return _shards(mesh, BATCH_AXES) * _shards(mesh, SEQUENCE_AXES)


def _per_shard_batch(mesh: Mapping[str, int],
                     batch_size: int | None) -> int | float | None:
    """Returns `batch_size` / batch shards, or None if `batch_size` is None."""
    if batch_size is None:
        return None
    batch_shards = _shards(mesh, BATCH_AXES)
    if batch_size % batch_shards == 0:
        return batch_size // batch_shards
    return batch_size / batch_shards


def _positive_int(value: Any, name: str) -> int:
    """Converts `value` (int, integral float or digit string) to an int >= 1."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _optional_positive_int(value: Any, name: str) -> int | None:
    return None if value is None else _positive_int(value, name)


def _axis_list(axes: str | Sequence[str]) -> list[str]:
    """Returns axis names from a sequence or from comma-separated text."""
    if isinstance(axes, str):
        return _AXIS_NAME_RE.findall(axes)
    return [str(axis).strip() for axis in axes if str(axis).strip()]


def _normalize_mesh(mesh: Mapping[Any, Any]) -> dict[str, int]:
    """Validates a mesh mapping and returns it as an ordered {axis: size}."""
    if not isinstance(mesh, Mapping):
        raise TypeError(
            f"A mesh must be a mapping of axis -> size, got {type(mesh).__name__}."
        )
    if not mesh:
        raise ValueError("The mesh is empty.")
    return {
        str(axis): _positive_int(size, f"Mesh axis {axis!r}")
        for axis, size in mesh.items()
    }


def parse_mesh(mesh: str | Mapping[str, Any]) -> dict[str, int]:
    """Parses a mesh given as a mapping, JSON, Python-dict or 'axis=size' text.

    Args:
      mesh: E.g. {'data': 16, 'model': 8}, '{"data": 16, "model": 8}', "{'data':
        16, 'model': 8}" or 'data=16, model=8'.

    Returns:
      The mesh as an ordered {axis: size} dict.

    Raises:
      ValueError: If the text cannot be parsed or a size is not a positive
        integer.
    """
    if isinstance(mesh, Mapping):
        return _normalize_mesh(mesh)
    text = str(mesh).strip()
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except ValueError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, TypeError, ValueError):
            pairs = _MESH_PAIR_RE.findall(text)
            parsed = {
                axis: int(size)
                for axis, size in pairs
            } if pairs else None
    if not isinstance(parsed, Mapping):
        raise ValueError(
            f"Cannot parse mesh {text!r}; expected JSON or Python-dict text such"
            ' as \'{"data": 16, "model": 8}\'.')
    return _normalize_mesh(parsed)


def parse_topology(topology: str | None) -> list[int] | None:
    """Parses a slice topology such as '4x4x16' or '4x8x16_twisted'.

    Args:
      topology: Topology string in chips. Text around the 'AxBxC' shape (e.g. a
        platform prefix or a '_twisted' suffix) is ignored. None or '' means
        unknown.

    Returns:
      The physical dims, e.g. [4, 4, 16], or None if `topology` is None or
      empty.

    Raises:
      ValueError: If `topology` is non-empty but contains no 'AxB[xC]' shape.
    """
    if topology is None or not str(topology).strip():
        return None
    text = str(topology).lower()
    contiguous = _CONTIGUOUS_TOPOLOGY_RE.findall(text)
    if contiguous:
        token = max(contiguous, key=lambda s: s.count("x") + s.count(","))
    else:
        match = _TOPOLOGY_RE.search(text)
        if match is None:
            raise ValueError(
                f"Cannot parse topology {topology!r}; expected e.g. '4x4x16'.")
        token = match.group(0)
    dims = [int(dim) for dim in re.split(r"\s*[x,]\s*", token)]
    if any(dim < 1 for dim in dims):
        raise ValueError(f"Topology {topology!r} has a zero dimension.")
    return dims


def wraparound_dims(dims: Sequence[int]) -> tuple[bool, ...]:
    """Says which physical dims of a slice have wrap-around ICI links.

    Public Cloud TPU documentation gives a 3D slice (v4, v5p) wrap-around links
    on every dimension once it is made of whole 4x4x4 cubes, and none below
    that; a 2D slice (v5e, v6e) gets them only on a dimension of 16 chips.
    tpu7x slices are also built from 4x4x4 cubes; the 3D rule is assumed for
    them.

    Args:
      dims: Physical dims of one slice in chips, e.g. [4, 4, 16].

    Returns:
      One flag per dimension, e.g. (False, True) for [8, 16].
    """
    sizes = [int(dim) for dim in dims]
    if len(sizes) == 3:
        return (all(dim % 4 == 0 for dim in sizes), ) * 3
    return tuple(len(sizes) == 2 and dim == 16 for dim in sizes)


def torus_fit(
    mesh: Mapping[str, int],
    dims: Sequence[int],
    ignore_axes: Sequence[str] = NON_ICI_AXES,
) -> dict[str, Any]:
    """Finds the assignment of mesh axes to whole physical torus dimensions.

    An axis that owns exactly one whole physical dimension forms a ring (closed
    by wrap-around links where the slice has them, see `wraparound_dims`); an
    axis that owns two or more whole dimensions gets a
    2D/3D collective (more links per chip). Every other axis shares a dimension
    with other axes (a partial line or a strided ring, i.e. less bandwidth).
    The search is exhaustive: it maximizes the number of axes on whole
    dimensions, then the number of axes spanning two or more dimensions; ties
    keep the first assignment found (axes in mesh order, smaller dimension
    subsets first). The result is informational: a framework may lay the axes
    out differently.

    Args:
      mesh: Mesh axis sizes.
      dims: Physical torus dims of one slice in chips, e.g. [4, 4, 16].
      ignore_axes: Axes that are not laid out on inter-chip ICI links.

    Returns:
      {'whole': {axis: [sizes of the dims it owns]}, 'shared': [axes that share
      dims], 'shared_dims': [sizes of the dims left to the shared axes]}, or
      {'note': ...} when the inter-chip axes do not multiply to the chip count.
    """
    axes, physical, note = _inter_chip(mesh, dims, ignore_axes)
    if note:
        return {"note": note}
    whole, shared_dims = _best_torus_assignment(axes, physical)
    owners = {axis for axis, _ in whole}
    return {
        "whole": {
            axis: list(sizes)
            for axis, sizes in whole
        },
        "shared": [axis for axis, _ in axes if axis not in owners],
        "shared_dims": list(shared_dims),
    }


def _inter_chip(
    mesh: Mapping[str, int],
    dims: Sequence[int],
    ignore_axes: Sequence[str],
) -> tuple[_AxisSizes, tuple[int, ...], str | None]:
    """Returns (inter-chip axes > 1, physical dims > 1, note or None)."""
    axes = tuple((axis, int(size)) for axis, size in mesh.items()
                 if int(size) > 1 and axis not in ignore_axes)
    physical = tuple(int(dim) for dim in dims if int(dim) > 1)
    product = math.prod(size for _, size in axes)
    chips = math.prod(physical)
    if product == chips:
        return axes, physical, None
    note = (
        f"inter-chip mesh product {product} (all axes except"
        f" {', '.join(ignore_axes) or 'none'}) != {chips} chips; no torus fit")
    if product == 2 * chips:
        note += (
            "; the mesh probably folds the 2 TensorCores per chip into another"
            " axis, model them as an intrachip axis of size 2 to get a fit")
    return axes, physical, note


@functools.lru_cache(maxsize=4096)
def _best_torus_assignment(
        axes: _AxisSizes,
        dims: tuple[int, ...]) -> tuple[_WholeDims, tuple[int, ...]]:
    """Exhaustive search behind `torus_fit`; memoized because inputs repeat."""
    subsets = []
    for subset_size in range(1, len(dims) + 1):
        for combo in itertools.combinations(enumerate(dims), subset_size):
            subsets.append((
                frozenset(index for index, _ in combo),
                math.prod(size for _, size in combo),
            ))
    best_score = (-1, -1)
    best: dict[str, frozenset[int]] = {}

    def search(index: int, used: frozenset[int],
               owned: dict[str, frozenset[int]]) -> None:
        nonlocal best_score, best
        if index == len(axes):
            multi_dim = sum(1 for dim_ids in owned.values()
                            if len(dim_ids) > 1)
            score = (len(owned), multi_dim)
            if score > best_score:
                best_score, best = score, dict(owned)
            return
        axis, size = axes[index]
        for subset, product in subsets:
            if product == size and not subset & used:
                owned[axis] = subset
                search(index + 1, used | subset, owned)
                del owned[axis]
        search(index + 1, used,
               owned)  # The axis shares a dim with other axes.

    search(0, frozenset(), {})
    used = frozenset().union(*best.values())
    # `best` keeps the axis order of `axes` because owners are added in order.
    whole = tuple((axis, tuple(sorted(dims[dim_id] for dim_id in dim_ids)))
                  for axis, dim_ids in best.items())
    shared_dims = tuple(
        sorted(size for dim_id, size in enumerate(dims) if dim_id not in used))
    return whole, shared_dims


def alternative_topologies(dims: Sequence[int],
                           min_dim: int | None = None) -> list[str]:
    """Lists other 3D power-of-two slice shapes with the same chip count.

    3D-torus TPU slices of 64 or more chips are assembled from 4x4x4 cubes, so by
    default every dimension must be at least 4 (a multiple of 4 for power-of-two
    shapes); smaller slices allow dimensions down to 2. These are plausible
    shapes, not a guarantee of availability (e.g. twisted-torus variants differ
    per platform): check that the platform offers a shape before proposing it.

    Args:
      dims: Physical dims of the current slice, e.g. [4, 4, 16].
      min_dim: Smallest allowed dimension; defaults to 4 for slices of 64+ chips
        and 2 otherwise.

    Returns:
      Shapes such as '4x8x8' sorted by dims, excluding the current shape. Empty
      unless `dims` is 3D with a power-of-two chip count.
    """
    chips = math.prod(dims)
    if len(dims) != 3 or not _is_power_of_two(chips):
        return []
    if min_dim is None:
        min_dim = 4 if chips >= 64 else 2
    current = tuple(sorted(dims))
    powers = [
        2**k for k in range(chips.bit_length()) if min_dim <= 2**k <= chips
    ]
    shapes = []
    for x_dim in powers:
        for y_dim in powers:
            if y_dim < x_dim or chips % (x_dim * y_dim):
                continue
            z_dim = chips // (x_dim * y_dim)
            if z_dim >= y_dim and (x_dim, y_dim, z_dim) != current:
                shapes.append(f"{x_dim}x{y_dim}x{z_dim}")
    return shapes


def _set_partitions(items: Sequence[int], ) -> Iterator[list[tuple[int, ...]]]:
    """Yields every partition of `items` into non-empty blocks."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for partition in _set_partitions(rest):
        yield [(first, )] + partition
        for i, block in enumerate(partition):
            yield partition[:i] + [(first, ) + block] + partition[i + 1:]


def _block_order(
        block: _Block) -> tuple[int, tuple[int, ...], tuple[str, ...]]:
    """Display order of blocks: planes before rings, then by dim sizes."""
    return -len(block[0]), block[0], block[1]


def _layout_rank(layout: _Layout) -> tuple[int, int, _Layout]:
    """Finest first: more blocks, then fewer axes in the most crowded block."""
    return -len(layout), max(len(axes) for _, axes in layout), layout


def _fill_blocks(
    axes: _AxisSizes,
    dims: tuple[int, ...],
    partition: Sequence[tuple[int, ...]],
    remaining: list[int],
    groups: list[list[str]],
    k: int,
    found: set[_Layout],
) -> None:
    """Assigns `axes[k:]` to blocks of `partition`; records exact tilings."""
    if k == len(axes):
        if all(rest == 1 for rest in remaining):
            blocks = ((tuple(sorted(dims[i] for i in block)), tuple(group))
                      for block, group in zip(partition, groups))
            found.add(tuple(sorted(blocks, key=_block_order)))
        return
    axis, size = axes[k]
    for b in range(len(partition)):
        if remaining[b] % size == 0:
            remaining[b] //= size
            groups[b].append(axis)
            _fill_blocks(axes, dims, partition, remaining, groups, k + 1,
                         found)
            groups[b].pop()
            remaining[b] *= size


@functools.lru_cache(maxsize=16384)
def _exact_layouts(axes: _AxisSizes, dims: tuple[int,
                                                 ...]) -> tuple[_Layout, ...]:
    """Every tiling of `dims` by groups of `axes` that each fill whole dims."""
    found: set[_Layout] = set()
    for partition in _set_partitions(tuple(range(len(dims)))):
        remaining = [math.prod(dims[i] for i in block) for block in partition]
        groups: list[list[str]] = [[] for _ in partition]
        _fill_blocks(axes, dims, partition, remaining, groups, 0, found)
    return tuple(sorted(found, key=_layout_rank))


def _render_layout(layout: _Layout) -> str:
    """Renders a layout as 'data+model@4x4 / seq@32' (no 'axis=N' text)."""
    return " / ".join(f"{'+'.join(axes)}@{'x'.join(str(dim) for dim in dims)}"
                      for dims, axes in layout)


def _choose_layout(
    mesh: Mapping[str, int],
    dims: Sequence[int],
    colocate: Sequence[str],
    ignore_axes: Sequence[str],
) -> tuple[_Layout, dict[str, Any] | None] | str:
    """Picks the layout `block_layout` reports."""
    axes, physical, note = _inter_chip(mesh, dims, ignore_axes)
    if note:
        return note
    if not physical:
        return (), None
    layouts = _exact_layouts(axes, physical)
    chosen = layouts[0]
    entry: dict[str, Any] | None = None
    if colocate:
        group = frozenset(
            axis for axis in colocate
            if int(mesh.get(axis, 1)) > 1 and axis not in ignore_axes)
        entry = {
            "axes": [axis for axis in colocate if axis in group],
            "block": None,
            "block_dims": 0,
        }
        matches = []
        if group:
            for layout in layouts:
                for block in layout:
                    if frozenset(block[1]) == group:
                        matches.append((block, layout))
        if matches:
            block, chosen = min(
                matches,
                key=lambda match: (
                    _GROUP_BLOCK_RANK.get(len(match[0][0]), 2),
                    _layout_rank(match[1]),
                ),
            )
            entry.update(
                block="x".join(str(dim) for dim in block[0]),
                block_dims=len(block[0]),
            )
    return chosen, entry


def block_layout(
    mesh: Mapping[str, int],
    dims: Sequence[int],
    colocate: Sequence[str] = (),
    ignore_axes: Sequence[str] = NON_ICI_AXES,
) -> dict[str, Any]:
    """Finds how groups of mesh axes can tile whole dims of the physical slice."""
    chosen = _choose_layout(mesh, dims, colocate, ignore_axes)
    if isinstance(chosen, str):
        return {"note": chosen}
    layout, entry = chosen
    if not layout:
        return {"layout": "-", "num_blocks": 0, "max_axes_per_block": 0}
    out: dict[str, Any] = {
        "layout": _render_layout(layout),
        "num_blocks": len(layout),
        "max_axes_per_block": max(len(group) for _, group in layout),
    }
    if entry is not None:
        out["colocate"] = entry
    return out


def tied_layouts(
    mesh: Mapping[str, int],
    dims: Sequence[int],
    colocate: Sequence[str] = (),
    ignore_axes: Sequence[str] = NON_ICI_AXES,
) -> list[list[tuple[tuple[int, ...], tuple[str, ...]]]]:
    """Returns the layout `block_layout` reports and the layouts as fine as it."""
    chosen = _choose_layout(mesh, dims, colocate, ignore_axes)
    if isinstance(chosen, str):
        return []
    layout, entry = chosen
    if not layout:
        return [[]]
    axes, physical, _ = _inter_chip(mesh, dims, ignore_axes)
    rank = _layout_rank(layout)[:2]
    group: frozenset[str] = frozenset()
    group_dims = 0
    if entry and entry["block"]:
        group = frozenset(entry["axes"])
        group_dims = int(entry["block_dims"])
    out = [layout]
    for other in _exact_layouts(axes, physical):
        if other == layout or _layout_rank(other)[:2] != rank:
            continue
        if group and not any(
                frozenset(names) == group and len(sizes) == group_dims
                for sizes, names in other):
            continue
        out.append(other)
    return [[(tuple(sizes), tuple(names)) for sizes, names in tied]
            for tied in out]


def _layout_key(fit: Mapping[str, Any],
                colocated: bool) -> tuple[int, int, int]:
    """Sort key of a `block_layout` result (smaller is better)."""
    group_rank = 0
    if colocated:
        block_dims = fit.get("colocate", {}).get("block_dims", 0)
        group_rank = (_GROUP_BLOCK_RANK.get(block_dims, 2)
                      if block_dims else _NO_GROUP_BLOCK_RANK)
    return group_rank, -fit["num_blocks"], fit["max_axes_per_block"]


def _topology_fits(
        base: Mapping[str, int],
        dims: Sequence[int],
        candidates: Sequence[Mapping[str, Any]] = (),
        colocate: Sequence[str] = (),
        top_n: int = 3,
) -> list[dict[str, Any]]:
    """Ranks the baseline and candidates by block layout, per slice shape."""
    current = "x".join(str(int(dim)) for dim in dims)
    out = []
    for shape in [current] + alternative_topologies(dims):
        shape_dims = [int(dim) for dim in shape.split("x")]
        entries = []
        for index, mesh, moves in [(-1, dict(base), [])
                                   ] + [(i, cand["mesh"], cand["moves"])
                                        for i, cand in enumerate(candidates)]:
            fit = block_layout(mesh, shape_dims, colocate)
            if "layout" not in fit:
                continue
            entries.append((
                (*_layout_key(fit, bool(colocate)), len(moves), index),
                {
                    "index": index,
                    "mesh": dict(mesh),
                    "moves": list(moves),
                    "block_layout": fit,
                },
            ))
        entries.sort(key=lambda entry: entry[0])
        out.append({
            "topology": shape,
            "current": shape == current,
            "baseline": block_layout(base, shape_dims, colocate),
            "best": [entry for _, entry in entries[:top_n]],
        })
    return out


def _single_moves(mesh: Mapping[str, int],
                  movable: Sequence[str]) -> Iterator[_Move]:
    """Yields every single move in (donor, receiver, factor) order."""
    for donor in movable:
        size = mesh[donor]
        for receiver in movable:
            if receiver == donor:
                continue
            factor = 2
            while size % factor == 0:
                yield _Move(donor, receiver, factor)
                factor *= 2


def _order_key(
        moves: Sequence[_Move],
        rank: Mapping[str, int]) -> tuple[int, list[tuple[int, int, int]]]:
    """Sort key: number of moves, then (donor, receiver, factor) per move."""
    return len(moves), [(rank[move.donor], rank[move.receiver], move.factor)
                        for move in moves]


def _explore(base: Mapping[str, int], movable: Sequence[str],
             max_moves: int) -> list[tuple[dict[str, int], tuple[_Move, ...]]]:
    """Breadth-first search over canonical move sequences."""
    seen = {tuple(base.values())}
    frontier: list[tuple[dict[str, int], tuple[_Move,
                                               ...]]] = [(dict(base), ())]
    reached = []
    for _ in range(max_moves):
        next_frontier = []
        for mesh, moves in frontier:
            donors = {move.donor for move in moves}
            receivers = {move.receiver for move in moves}
            for move in _single_moves(mesh, movable):
                if move.donor in receivers or move.receiver in donors:
                    continue
                new_mesh = dict(mesh)
                new_mesh[move.donor] //= move.factor
                new_mesh[move.receiver] *= move.factor
                key = tuple(new_mesh.values())
                if key in seen:
                    continue
                seen.add(key)
                next_frontier.append((new_mesh, moves + (move, )))
        reached.extend(next_frontier)
        frontier = next_frontier
    rank = {axis: index for index, axis in enumerate(base)}
    reached.sort(key=lambda entry: _order_key(entry[1], rank))
    return reached


def _kv_head_warnings(mesh: Mapping[str, int], num_kv_heads: int) -> list[str]:
    """Warns when the KV heads cannot be split evenly across their shards."""
    kv_shards = _kv_head_shards(mesh)
    label = ("tensor shards" if mesh.get("model_q", 1) == 1 else
             "KV-head shards (tensor shards except model_q)")
    warnings = []
    if kv_shards > num_kv_heads:
        warnings.append(
            f"{label} {kv_shards} > num_kv_heads {num_kv_heads}: KV heads are"
            " replicated across shards (more KV memory per device)")
    if num_kv_heads % kv_shards:
        warnings.append(
            f"num_kv_heads {num_kv_heads} is not divisible by {label}"
            f" {kv_shards}: rejected by strict SplashAttention KV-head"
            " divisibility validators")
    return warnings


def _check_mesh(mesh: Mapping[str, int], base: Mapping[str, int],
                workload: _Workload) -> tuple[list[str], list[str]]:
    """Returns (violations, warnings) of a mesh; any violation makes it illegal."""
    violations = [
        f"{axis}={size} is not a power of two" for axis, size in mesh.items()
        if size != base[axis] and not _is_power_of_two(size)
    ]
    divisibility_checks = (
        ("batch_size", workload.batch_size, BATCH_AXES, "batch"),
        ("sequence_length", workload.sequence_length, SEQUENCE_AXES,
         "sequence"),
        ("num_heads", workload.num_heads, QUERY_HEAD_AXES, "query-head"),
    )
    for name, value, axes, family in divisibility_checks:
        shards = _shards(mesh, axes)
        if value is not None and value % shards:
            violations.append(
                f"{name} {value} is not divisible by {shards} {family} shards")
    warnings = []
    if workload.num_heads is not None and mesh.get("model2", 1) > 1:
        shards = _shards(mesh, TENSOR_AXES)
        if workload.num_heads % shards:
            warnings.append(
                f"num_heads {workload.num_heads} is not divisible by {shards} tensor"
                " shards including model2: legal only if model2 does not shard heads"
            )
    if workload.num_kv_heads is not None:
        warnings.extend(_kv_head_warnings(mesh, workload.num_kv_heads))
    if mesh.get("intrachip") != base.get("intrachip"):
        warnings.append(
            f"intrachip {base.get('intrachip')} -> {mesh.get('intrachip')}:"
            " intrachip is the per-chip TensorCore factor, not an ICI axis")
    return violations, warnings


def _move_kind(
    mesh: Mapping[str, int],
    base: Mapping[str, int],
    moves: Sequence[_Move],
) -> str:
    """Classifies a candidate as 'sibling_refactor', 'collapse' or 'shift'."""
    if all(
            axis_family(move.donor) == axis_family(move.receiver) != "other"
            for move in moves):
        return "sibling_refactor"
    if any(size == 1 and base[axis] > 1 for axis, size in mesh.items()):
        return "collapse"
    return "shift"


def _move_family(
    mesh: Mapping[str, int],
    base: Mapping[str, int],
    moves: Sequence[_Move],
) -> str:
    """Names the axis families each move connects, e.g. 'tensor->batch'."""
    names = []
    for move in moves:
        donor, receiver = axis_family(move.donor), axis_family(move.receiver)
        if donor == receiver != "other":
            swapped = (mesh[move.donor] == base[move.receiver]
                       and mesh[move.receiver] == base[move.donor])
            names.append(f"{'swap' if swapped else 'sibling'}:{donor}")
        else:
            names.append(f"{donor}->{receiver}")
    return "+".join(names)


def _describe_baseline(
        base: Mapping[str, int],
        movable: Sequence[str],
        workload: _Workload,
        dims: Sequence[int] | None,
        colocate: Sequence[str] = (),
) -> dict[str, Any]:
    """Returns the JSON summary of the baseline mesh."""
    violations, warnings = _check_mesh(base, base, workload)
    warnings.extend(
        f"{axis}={base[axis]} is not a power of two; candidates keep it unchanged"
        for axis in movable if not _is_power_of_two(base[axis]))
    return {
        "mesh": dict(base),
        "devices": math.prod(base.values()),
        "legal": not violations,
        "violations": violations,
        "warnings": warnings,
        "per_shard_batch": _per_shard_batch(base, workload.batch_size),
        "tensor_shards": _shards(base, TENSOR_AXES),
        "layout": block_layout(base, dims, colocate) if dims else None,
    }


def _describe_candidate(
        mesh: dict[str, int],
        moves: Sequence[_Move],
        base: Mapping[str, int],
        workload: _Workload,
        warnings: list[str],
        dims: Sequence[int] | None,
        colocate: Sequence[str] = (),
) -> dict[str, Any]:
    """Returns the JSON summary of one legal candidate mesh."""
    return {
        "mesh":
        mesh,
        "moves": [str(move) for move in moves],
        "kind":
        _move_kind(mesh, base, moves),
        "family":
        _move_family(mesh, base, moves),
        "collapsed_axes":
        [axis for axis, size in mesh.items() if size == 1 and base[axis] > 1],
        "per_shard_batch":
        _per_shard_batch(mesh, workload.batch_size),
        "activation_tokens_ratio":
        _token_shards(base) / _token_shards(mesh),
        "tensor_shards":
        _shards(mesh, TENSOR_AXES),
        "warnings":
        warnings,
        "layout":
        block_layout(mesh, dims, colocate) if dims else None,
    }


def _group_by_donor(
    candidate_donors: Sequence[set[str]],
    num_returned: int,
    axis_order: Sequence[str],
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Returns (donor -> returned candidate indices, donor -> total count)."""
    indices: dict[str, list[int]] = {axis: [] for axis in axis_order}
    counts = {axis: 0 for axis in axis_order}
    for index, donors in enumerate(candidate_donors):
        for donor in donors:
            counts[donor] += 1
            if index < num_returned:
                indices[donor].append(index)
    return (
        {
            axis: found
            for axis, found in indices.items() if found
        },
        {
            axis: count
            for axis, count in counts.items() if count
        },
    )


def _pair_family(donor: str, receiver: str) -> str:
    """Names the axis families of a single move, e.g. 'tensor->batch'."""
    donor_family, receiver_family = axis_family(donor), axis_family(receiver)
    if donor_family == receiver_family != "other":
        return f"sibling:{donor_family}"
    return f"{donor_family}->{receiver_family}"


def _bracketed_pairs(
    base: Mapping[str, int],
    single_moves: Mapping[tuple[str, str], Sequence[tuple[int, int]]],
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Returns the bracket points of every legal single-move pair."""
    pairs = []
    for (donor, receiver), moves in single_moves.items():
        index_of = dict(moves)
        first, last = moves[0][0], moves[-1][0]
        roles: dict[int, list[str]] = {first: [f"k={first}"]}
        roles.setdefault(last, []).append("boundary")
        family = _pair_family(donor, receiver)
        swap = base[donor] // base[receiver]
        if (family.startswith("sibling:") and swap > 1
                and swap * base[receiver] == base[donor] and swap in index_of):
            roles.setdefault(swap, []).append("swap")
        pairs.append({
            "donor":
            donor,
            "receiver":
            receiver,
            "family":
            family,
            "points": [{
                "bracket": ", ".join(roles[factor]),
                "factor": factor,
                "index": index_of[factor],
                "mesh": dict(candidates[index_of[factor]]["mesh"]),
            } for factor in sorted(roles)],
        })
    return pairs


def _check_search_limits(max_moves: int, max_candidates: int | None) -> None:
    """Raises ValueError if `max_moves` or `max_candidates` is out of range."""
    if (isinstance(max_moves, bool) or not isinstance(max_moves, int)
            or not 1 <= max_moves <= MAX_MOVES_LIMIT):
        raise ValueError(
            f"max_moves must be between 1 and {MAX_MOVES_LIMIT}, got {max_moves!r}."
        )
    if max_candidates is not None and (isinstance(max_candidates, bool)
                                       or not isinstance(max_candidates, int)
                                       or max_candidates < 0):
        raise ValueError(
            f"max_candidates must be None or >= 0, got {max_candidates!r}.")


def enumerate_mesh_candidates(
        baseline_mesh: Mapping[str, int],
        *,
        extra_axes: Sequence[str] = (),
        fixed_axes: Sequence[str] = DEFAULT_FIXED_AXES,
        batch_size: int | None = None,
        sequence_length: int | None = None,
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
        topology: str | None = None,
        max_moves: int = 1,
        max_candidates: int | None = None,
        colocate: Sequence[str] = (),
) -> dict[str, Any]:
    """Enumerates legal meshes within `max_moves` pairwise factor moves."""
    base = _normalize_mesh(baseline_mesh)
    for axis in _axis_list(extra_axes):
        base.setdefault(axis, 1)
    fixed = set(_axis_list(fixed_axes))
    movable = [axis for axis in base if axis not in fixed]
    _check_search_limits(max_moves, max_candidates)
    workload = _Workload(
        batch_size=_optional_positive_int(batch_size, "batch_size"),
        sequence_length=_optional_positive_int(sequence_length,
                                               "sequence_length"),
        num_heads=_optional_positive_int(num_heads, "num_heads"),
        num_kv_heads=_optional_positive_int(num_kv_heads, "num_kv_heads"),
    )
    dims = parse_topology(topology)
    group = _axis_list(colocate)

    candidates = []
    candidate_donors = []
    single_moves: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for mesh, moves in _explore(base, movable, max_moves):
        violations, warnings = _check_mesh(mesh, base, workload)
        if violations:
            continue
        if len(moves) == 1:
            single_moves.setdefault((moves[0].donor, moves[0].receiver),
                                    []).append(
                                        (moves[0].factor, len(candidates)))
        candidates.append(
            _describe_candidate(mesh, moves, base, workload, warnings, dims,
                                group))
        candidate_donors.append({move.donor for move in moves})
    returned = (candidates
                if max_candidates is None else candidates[:max_candidates])
    by_donor, num_by_donor = _group_by_donor(candidate_donors, len(returned),
                                             list(base))
    return {
        "baseline":
        _describe_baseline(base, movable, workload, dims, group),
        "movable_axes":
        movable,
        "fixed_axes": [axis for axis in base if axis in fixed],
        "topology":
        "x".join(str(dim) for dim in dims) if dims else None,
        "colocate":
        group,
        "num_candidates":
        len(candidates),
        "num_returned":
        len(returned),
        "candidates":
        returned,
        "by_donor":
        by_donor,
        "num_candidates_by_donor":
        num_by_donor,
        "bracketed_pairs":
        _bracketed_pairs(base, single_moves, candidates),
        "topology_fits":
        (_topology_fits(base, dims, candidates, group) if dims else []),
    }


def _parse_fixed_axes_text(fixed_axes: str) -> list[str]:
    """'' -> DEFAULT_FIXED_AXES, 'none' -> [], else the comma-separated axes."""
    axes = _axis_list(fixed_axes or "")
    if not axes:
        return list(DEFAULT_FIXED_AXES)
    if [axis.lower() for axis in axes] == ["none"]:
        return []
    return axes


def _compact_layout(fit: Mapping[str, Any] | None) -> str:
    """Renders a `block_layout` result, e.g. 'data+model@4x4 / seq@32'."""
    if not fit:
        return "-"
    if "note" in fit:
        return str(fit["note"])
    return str(fit.get("layout", "-"))


def _colocate_cell(fit: Mapping[str, Any] | None) -> str:
    """Renders the co-located group's block, e.g. '4x4 plane' or '-'."""
    if not fit or "colocate" not in fit:
        return "-"
    block = fit["colocate"].get("block")
    if not block:
        return "-"
    kind = {
        1: "ring",
        2: "plane"
    }.get(fit["colocate"].get("block_dims"), "slice")
    return f"{block} {kind}"


def _cell(value: Any) -> str:
    """Renders a table cell: None as '-', floats without trailing zeros."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _index_ranges(indices: Sequence[int]) -> str:
    """Renders sorted indices compactly, e.g. [0, 1, 2, 5, 7, 8] -> '0-2,5,7-8'."""
    parts = []
    start = previous = None
    for index in indices:
        if previous is not None and index == previous + 1:
            previous = index
            continue
        if start is not None:
            parts.append(
                str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    if start is not None:
        parts.append(
            str(start) if start == previous else f"{start}-{previous}")
    return ",".join(parts)


def _point_cell(point: Mapping[str, Any]) -> str:
    """Renders one bracket point, e.g. '{data:16, model:1} (k=4, #3)'."""
    mesh = ", ".join(f"{k}:{v}" for k, v in point["mesh"].items())
    return f"{{{mesh}}} (k={point['factor']}, #{point['index']})"


_SHORTLIST_HEADER = (
    "Bracketed shortlist (single moves; fixed order, NOT ranked): every legal"
    " donor -> receiver pair at its first step (normally k=2) and at its"
    " boundary (largest legal k; the donor collapses to 1 when legal), plus the"
    " size swap for sibling axes. Pick pairs from the profile evidence; '= ...'"
    " marks a point that coincides with an earlier column; #n is the candidate"
    " index.")


def _shortlist_lines(pairs: Sequence[Mapping[str, Any]]) -> list[str]:
    """Renders `bracketed_pairs` as a markdown table; [] when there are none."""
    if not pairs:
        return []
    lines = [
        "",
        _SHORTLIST_HEADER,
        "",
        "| pair | family | first step | boundary | swap |",
        "|---|---|---|---|---|",
    ]
    for pair in pairs:
        points = pair["points"]
        first = points[0]
        boundary = next(p for p in points if "boundary" in p["bracket"])
        swap = next((p for p in points if "swap" in p["bracket"]), None)
        boundary_cell = ("= first step"
                         if boundary is first else _point_cell(boundary))
        if swap is None:
            swap_cell = "-"
        elif swap is first:
            swap_cell = "= first step"
        elif swap is boundary:
            swap_cell = "= boundary"
        else:
            swap_cell = _point_cell(swap)
        lines.append(
            f"| {pair['donor']}->{pair['receiver']} | {pair['family']} |"
            f" {_point_cell(first)} | {boundary_cell} | {swap_cell} |")
    return lines


def format_candidates_table(result: Mapping[str, Any]) -> str:
    """Renders enumerate_mesh_candidates output as compact markdown."""
    base = result["baseline"]
    group = result.get("colocate") or []
    group_label = "+".join(group) if group else ""
    baseline_line = (
        f"Baseline mesh: {base['mesh']} | devices {base['devices']} |"
        f" per-shard batch {_cell(base.get('per_shard_batch'))} | tensor shards"
        f" {_cell(base.get('tensor_shards'))} | layout"
        f" {_compact_layout(base.get('layout'))}")
    if group:
        baseline_line += (
            f" | {group_label} block {_colocate_cell(base.get('layout'))}")
    lines = [baseline_line]
    for issue in base.get("violations", []) + base.get("warnings", []):
        lines.append(f"Baseline note: {issue}")
    shortlist = _shortlist_lines(result.get("bracketed_pairs") or [])
    if shortlist:
        lines.extend(shortlist)
        lines.append("")
    lines.append(
        f"Candidates: {result['num_returned']} of {result['num_candidates']}"
        " (fixed order: moves, donor, receiver, factor; NOT ranked). Movable"
        f" axes: {', '.join(result['movable_axes'])}; fixed:"
        f" {', '.join(result['fixed_axes']) or '-'}. Layout = groups of axes"
        " that exactly fill whole physical dims (axes@dims, blocks separated by"
        " '/'). This table is a search aid: copy only the meshes you propose"
        " into the deliverables.")
    lines.append("")
    lines.append(
        "| # | move(s) | family | mesh | per-shard batch | act. tokens x |"
        f" tensor | layout | {group_label or 'colocate'} block | warnings |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for index, cand in enumerate(result["candidates"]):
        mesh = ", ".join(f"{k}:{v}" for k, v in cand["mesh"].items())
        lines.append(f"| {index} | {'; '.join(cand['moves'])} |"
                     f" {cand.get('family', '-')} | {{{mesh}}} |"
                     f" {_cell(cand.get('per_shard_batch'))} |"
                     f" {_cell(cand.get('activation_tokens_ratio'))} |"
                     f" {_cell(cand.get('tensor_shards'))} |"
                     f" {_compact_layout(cand.get('layout'))} |"
                     f" {_colocate_cell(cand.get('layout'))} |"
                     f" {'; '.join(cand.get('warnings', [])) or '-'} |")
    totals = result.get("num_candidates_by_donor", {})
    if totals:
        shown = result.get("by_donor", {})
        lines.append("")
        lines.append("By donor (indices shown / total legal): " + "; ".join(
            f"{axis}: {_index_ranges(shown.get(axis, [])) or 'none shown'}"
            f" / {total}" for axis, total in totals.items()))
    fits = result.get("topology_fits") or []
    if fits:
        rank = ""
        if group:
            rank = (
                f"the {group_label} block (whole plane, then whole ring, then none),"
                " then ")
        lines.append("")
        lines.append(
            "Best block layouts per slice shape (current shape first; re-shapes"
            " keep the chip count, verify platform support). Ranked by"
            f" {rank}the finest layout (more blocks, fewer axes per block), then"
            " fewer moves; #-1 = baseline:")
        for fit in fits:
            label = " (current)" if fit.get("current") else ""
            baseline = fit.get("baseline")
            extra = ""
            if group:
                extra = f", {group_label} block {_colocate_cell(baseline)}"
            lines.append(f"- {fit['topology']}{label}: baseline"
                         f" {_compact_layout(baseline)}{extra}")
            for entry in fit.get("best", []):
                mesh = ", ".join(f"{k}:{v}" for k, v in entry["mesh"].items())
                moves = "; ".join(entry["moves"]) or "no move"
                extra = ""
                if group:
                    extra = (
                        f" [{group_label} block {_colocate_cell(entry['block_layout'])}]"
                    )
                lines.append(
                    f"  - #{entry['index']} {moves} -> {{{mesh}}}:"
                    f" {_compact_layout(entry['block_layout'])}{extra}")
    return "\n".join(lines)


def enumerate_mesh_candidates_tool_wrapper(
    baseline_mesh: str,
    batch_size: int = 0,
    sequence_length: int = 0,
    num_heads: int = 0,
    num_kv_heads: int = 0,
    topology: str = "",
    extra_axes: str = "",
    fixed_axes: str = "",
    max_moves: int = 1,
    max_candidates: int = 60,
    colocate: str = "",
) -> str:
    """Lists legal ICI meshes reachable from a baseline by pairwise factor moves.

    Args:
      baseline_mesh: Baseline ICI mesh as JSON or Python-dict text.
      batch_size: Batch sharded by the ICI batch axes (0 = unknown).
      sequence_length: Sequence length (0 = unknown).
      num_heads: Number of query heads (0 = unknown).
      num_kv_heads: Number of KV heads. Only adds warnings (KV heads replicated
        when tensor shards exceed them; strict SplashAttention validators require
        them divisible by the tensor shards). 0 = unknown.
      topology: Physical slice shape in chips, e.g. '4x4x4' ('' = unknown).
      extra_axes: Comma-separated extra axes that start at size 1.
      fixed_axes: Comma-separated axes that must keep their size ('' = default,
        'none' = pin nothing).
      max_moves: Moves per candidate: 1 (default), 2 or 3.
      max_candidates: Maximum number of candidates returned (<= 0 returns all).
      colocate: Comma-separated axis group to keep on one compact block.

    Returns:
      JSON string summarizing baseline, legal candidates, bracketed pairs, and
      topology fits, or '{"error": "..."}' for invalid inputs.
    """
    try:
        result = enumerate_mesh_candidates(
            parse_mesh(baseline_mesh),
            extra_axes=_axis_list(extra_axes or ""),
            fixed_axes=_parse_fixed_axes_text(fixed_axes),
            batch_size=batch_size or None,
            sequence_length=sequence_length or None,
            num_heads=num_heads or None,
            num_kv_heads=num_kv_heads or None,
            topology=topology or None,
            max_moves=max_moves,
            max_candidates=(max_candidates if max_candidates
                            and max_candidates > 0 else None),
            colocate=_axis_list(colocate or ""),
        )
    except (TypeError, ValueError) as e:
        return json.dumps({"error": str(e)})
    return json.dumps(result)
