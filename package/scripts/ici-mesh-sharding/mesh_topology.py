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
"""Physical TPU topology fit checks, stacked partition layouts, and target slice planning.

JAX `mesh_utils.create_device_mesh` (and JAX/MaxText/PyTorch-XLA partition
configs built on it) default to `allow_split_physical_axes=False`. With that
default a logical mesh can only be placed on the physical topology if every
physical axis is assigned whole to one logical axis and every logical axis size
equals the product of its physical axes. Otherwise compilation fails with a
mesh-creation or sharding lowering error, and the mesh is only buildable with
`allow_split_physical_axes=True`.

Example: on `v5p` `8x8x8` (512 devices) `{"data": 8, "seq": 8, "model": 8}` and
`{"data": 64, "seq": 1, "model": 8}` fit without splitting, whereas
`{"data": 16, "seq": 4, "model": 8}` does not (16 and 4 are not products of
whole 8-sized axes) and requires `allow_split_physical_axes=True`.

This module also provides:
- `TopologyOnboardingPlan` and `plan_target_topology(...)` for sizing a target
  TPU generation slice using peak-FLOPs compute parity and safe HBM capacity.
- Stacked-partition layout placement helpers (`describe_partition_layout`, etc.)
  that describe where stacked `RING` / `GRID` partition configs place each mesh
  axis on the physical 2D/3D torus.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mesh_search  # noqa: E402
import tpu_specs  # noqa: E402

# Usable fraction of nameplate HBM; the remainder covers XLA runtime buffers,
# scratch memory, and allocator fragmentation.
HBM_UTILIZATION_SAFETY_FACTOR = 0.85

_DIM_NAMES = "XYZ"

# 2D tile shapes inferred for stacked partition axes of each power-of-two size.
_TILE_SHAPES = {
    1: (1, 1),
    2: (1, 2),
    4: (2, 2),
    8: (2, 4),
    16: (4, 4),
    32: (4, 8),
    64: (8, 8),
    128: (8, 16),
    256: (16, 16),
}

# Stackable types (inner to outer) of stacked-partition presets.
_PRESET_TYPES = {
    "GridOfRingsPartitionConfig": ("RING", "GRID"),
    "RingOfRingsPartitionConfig": ("RING", "RING"),
    "RingOfRingsOfRingsPartitionConfig": ("RING", "RING", "RING"),
    "GridOfGridsOfRingsPartitionConfig": ("RING", "GRID", "GRID"),
}


def parse_topology(topology: str | Sequence[int]) -> list[int]:
    """Parses a topology such as `8x8x8` (optionally prefixed) into dims.

    Args:
      topology: A topology string (e.g. `"8x8x8"`, `"v5p=8x8x8"`,
        `"4x8x16_twisted"`) or a sequence of dimension sizes.

    Returns:
      The list of physical dimension sizes.

    Raises:
      ValueError: If the topology cannot be parsed or has a dimension < 1.
    """
    if isinstance(topology, str):
        dims = mesh_search.parse_topology(topology)
        if dims is None:
            raise ValueError(
                f"Invalid topology {topology!r}; expected e.g. 8x8x8.")
    else:
        dims = [int(d) for d in topology]
    if not dims or any(d < 1 for d in dims):
        raise ValueError(f"Invalid topology {topology!r}.")
    return dims


def get_devices_per_chip(platform: str | None) -> int | None:
    """Returns logical JAX/XLA devices per physical TPU chip for `platform`.

    Megacore and single-TensorCore generations (`v4`, `v5e`, `v5p`, `v6e`)
    expose 1 logical device per chip; dual-chiplet `tpu7x` exposes 2 logical
    devices per chip.
    """
    return tpu_specs.get_devices_per_chip(platform)


def check_mesh_topology_fit(
    mesh: Mapping[str, int],
    topology: str | Sequence[int],
    cores_per_chip: int = 1,
    independent_axes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Checks whether `mesh` maps onto `topology` without splitting axes.

    Follows `jax.experimental.mesh_utils.create_device_mesh` with its default
    `allow_split_physical_axes=False`: every physical axis must be assigned whole
    to one logical axis, and each logical axis size must equal the product of
    its physical axes. Size-1 logical axes need no physical axis. When no such
    assignment exists but the device counts match, the mesh is still buildable
    with `allow_split_physical_axes=True` (at some collective bandwidth cost).

    Args:
      mesh: Logical axis name to size, e.g. `{"data": 16, "seq": 4}`.
      topology: Physical chip topology, e.g. `"8x8x8"` or `[8, 8, 8]`.
      cores_per_chip: Mesh devices per chip; adds a physical axis of this size.
      independent_axes: For a `GridOfRingsPartitionConfig` (a stack of partition
        configs), the axes that are not stacked. Their product must equal one
        whole (unsplit) physical axis, since each stacked slice is laid out
        along one physical dimension; `allow_split_physical_axes` cannot fix a
        mismatch.

    Returns:
      A dict with `fits_without_split`, `allow_split_physical_axes`,
      `device_count_matches`, `buildable`, `assignment`, and `reason` (plus
      `independent_axes_product` and `independent_axes_fit` when
      `independent_axes` is supplied).
    """
    chip_dims = parse_topology(topology)
    physical = list(chip_dims)
    if cores_per_chip > 1 and len(physical) < 4:
        physical = physical + [cores_per_chip]
    sizes = {str(k): int(v) for k, v in mesh.items()}
    if any(v < 1 for v in sizes.values()):
        raise ValueError(f"Mesh axis sizes must be >= 1, got {sizes}.")
    num_mesh_devices = math.prod(sizes.values())
    num_physical_devices = math.prod(physical)
    result: dict[str, Any] = {
        "mesh": sizes,
        "physical_axes": physical,
        "num_mesh_devices": num_mesh_devices,
        "num_physical_devices": num_physical_devices,
        "device_count_matches": num_mesh_devices == num_physical_devices,
        "fits_without_split": False,
        "allow_split_physical_axes": False,
        "assignment": None,
        "buildable": num_mesh_devices == num_physical_devices,
    }
    if num_mesh_devices != num_physical_devices:
        result["reason"] = (
            f"Mesh has {num_mesh_devices} devices but the topology has"
            f" {num_physical_devices}; no mesh can be built.")
        return result
    if independent_axes:
        product = math.prod(sizes.get(str(a), 1) for a in independent_axes)
        result["independent_axes_product"] = product
        result["independent_axes_fit"] = (
            product == 1 or product in chip_dims or product == cores_per_chip
            or (cores_per_chip > 1 and product % cores_per_chip == 0
                and product // cores_per_chip in chip_dims))
        if not result["independent_axes_fit"]:
            result["buildable"] = False
            result["reason"] = (
                "GridOfRingsPartitionConfig: the independent axes"
                f" {list(independent_axes)} have product {product}, which is not a"
                f" single physical axis of {physical}; each stacked slice must lie"
                " along one whole physical dimension (allow_split_physical_axes"
                " does not help). Choose sizes whose product is one of"
                f" {sorted(set(chip_dims))}" +
                (f" (times {cores_per_chip} cores per chip)."
                 if cores_per_chip > 1 else "."))
            return result

    fit = mesh_search.torus_fit(sizes, physical, ignore_axes=())
    if "note" not in fit and not fit.get("shared"):
        result["fits_without_split"] = True
        result["assignment"] = {
            name: list(fit["whole"].get(name, []))
            for name in sizes
        }
        result["reason"] = (
            "Each logical axis maps to whole physical axes; keep"
            " allow_split_physical_axes at its default (False).")
        return result

    result["allow_split_physical_axes"] = True
    result["reason"] = (
        f"No assignment of whole physical axes {physical} gives {sizes}; at"
        " least one physical axis must be split across logical axes. Set"
        " allow_split_physical_axes=True for this candidate (collectives on the"
        " split axes run as strided or partial rings), or pick a mesh/topology"
        " that fits.")
    return result


def requires_split_physical_axes(
    mesh: Mapping[str, int],
    topology: str | Sequence[int],
    cores_per_chip: int = 1,
) -> bool:
    """Returns True iff `mesh` is buildable only with split physical axes."""
    return check_mesh_topology_fit(mesh, topology,
                                   cores_per_chip)["allow_split_physical_axes"]


def decide_allow_split_physical_axes(
    mesh: Mapping[str, int],
    topology: str | Sequence[int] | None,
    cores_per_chip: int | None,
    explicit: bool | None = None,
) -> tuple[bool | None, str]:
    """Decides the `allow_split_physical_axes` value for a candidate trial."""
    if not mesh or not topology or not cores_per_chip:
        return explicit, (
            "Topology fit not checked (missing mesh, topology or platform);"
            f" using proposal value {explicit}.")
    try:
        fit = check_mesh_topology_fit(mesh, topology, cores_per_chip)
    except (TypeError, ValueError) as e:
        return explicit, (
            f"Topology fit not checked ({e}); using proposal value {explicit}."
        )
    if not fit["device_count_matches"]:
        return explicit, f"{fit['reason']} Using proposal value {explicit}."
    if fit["allow_split_physical_axes"]:
        if explicit is not None and not explicit:
            return True, (
                "Proposal set allow_split_physical_axes=False but the mesh needs it:"
                f" {fit['reason']}")
        return True, fit["reason"]
    return explicit, fit["reason"]


# ---------------------------------------------------------------------------
# Target TPU Generation Topology Onboarding Planner
# ---------------------------------------------------------------------------


def _get_next_power_of_two(n: int) -> int:
    """Returns the smallest power of 2 greater than or equal to n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def get_optimal_3d_topology(n_chips: int) -> str:
    """Generates a balanced 3D torus topology (`XxYxZ`) for a power-of-2 chip count."""
    if n_chips < 1:
        return "1x1x1"
    rounded = _get_next_power_of_two(n_chips)
    k = int(math.log2(rounded))
    base = k // 3
    remainder = k % 3
    c = base + (1 if remainder > 0 else 0)
    b = base + (1 if remainder > 1 else 0)
    a = base
    return f"{2**a}x{2**b}x{2**c}"


def get_optimal_2d_topology(n_chips: int) -> str:
    """Generates a near-square 2D torus topology (`XxY`) for a power-of-2 chip count."""
    if n_chips < 1:
        return "1x1"
    rounded = _get_next_power_of_two(n_chips)
    k = int(math.log2(rounded))
    base = k // 2
    remainder = k % 2
    x = base
    y = base + remainder
    return f"{2**x}x{2**y}"


_get_optimal_3d_topology = get_optimal_3d_topology
_get_optimal_2d_topology = get_optimal_2d_topology


@dataclasses.dataclass(frozen=True)
class TopologyOnboardingPlan:
    """Target TPU generation slice sizing and topology recommendation."""

    source_platform: str
    source_chips: int
    source_dim: int
    source_devices: int
    source_hbm_gib: float
    target_platform: str
    target_dim: int
    target_hbm_gib: float
    recommended_chips: int
    recommended_topology: str
    recommended_slices: int
    chips_per_slice: int = 0
    chips_needed_for_compute_parity: int = 0
    chips_needed_for_memory: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def __str__(self) -> str:
        return (
            f"Topology Onboarding Plan ({self.source_platform} ->"
            f" {self.target_platform}):\n1. Analysis:\n   - Source:"
            f" {self.source_chips} chips on {self.source_platform}"
            f" ({self.source_dim}D), {self.source_devices} logical devices,"
            f" Memory Capacity: {self.source_hbm_gib:g} GiB/chip\n"
            f"   - Target: {self.target_platform} ({self.target_dim}D),"
            f" Memory Capacity: {self.target_hbm_gib:g} GiB/chip\n"
            f"2. Recommendation:\n   - Platform: {self.target_platform}"
            f" ({self.target_dim}D)\n   - Total Target Chips:"
            f" {self.recommended_chips}\n   - Chips per Slice:"
            f" {self.chips_per_slice}\n   - Topology:"
            f" {self.recommended_topology}\n   - Number of Slices:"
            f" {self.recommended_slices}\n   - Binding constraint: compute"
            f" parity needs {self.chips_needed_for_compute_parity} chips,"
            f" memory needs {self.chips_needed_for_memory} chips")


def plan_target_topology(
    *,
    source_chips: int,
    source_platform: str,
    target_platform: str,
    total_model_memory_per_chip_gib: float | None = None,
) -> TopologyOnboardingPlan:
    """Sizes a target TPU topology using peak-FLOPs parity and safe HBM checks.

    Args:
      source_chips: Number of physical chips in the source topology (>= 1).
      source_platform: Public source TPU generation (e.g. `'v5p'`, `'v6e'`).
      target_platform: Public target TPU generation (e.g. `'tpu7x'`, `'v6e'`).
      total_model_memory_per_chip_gib: Measured HBM used per source chip in GiB
        (optional; defaults to full source chip HBM when omitted).

    Returns:
      A `TopologyOnboardingPlan` describing the recommended target slice shape
      and slice count.
    """
    if source_chips < 1:
        raise ValueError(f"source_chips must be >= 1, got {source_chips!r}.")
    src_canonical = tpu_specs.normalize_platform(source_platform)
    tgt_canonical = tpu_specs.normalize_platform(target_platform)
    if src_canonical is None:
        raise ValueError(
            f"Unknown source_platform {source_platform!r}; expected one of"
            f" {tpu_specs.PUBLIC_PLATFORMS}.")
    if tgt_canonical is None:
        raise ValueError(
            f"Unknown target_platform {target_platform!r}; expected one of"
            f" {tpu_specs.PUBLIC_PLATFORMS}.")
    src_spec = tpu_specs.PLATFORM_SPECS[src_canonical]
    tgt_spec = tpu_specs.PLATFORM_SPECS[tgt_canonical]

    total_source_devices = source_chips * int(src_spec["devices_per_chip"])
    src_flops = float(src_spec.get("peak_flops", 0.0))
    tgt_flops = float(tgt_spec.get("peak_flops", 0.0))
    if src_flops > 0 and tgt_flops > 0:
        chips_needed_compute = math.ceil(source_chips * src_flops / tgt_flops)
    else:
        chips_needed_compute = math.ceil(total_source_devices /
                                         int(tgt_spec["devices_per_chip"]))

    source_mem_per_chip_gib = float(src_spec["hbm_gib"])
    target_mem_per_chip_gib = float(tgt_spec["hbm_gib"])
    mem_to_consider = (
        source_mem_per_chip_gib if total_model_memory_per_chip_gib is None else
        min(float(total_model_memory_per_chip_gib), source_mem_per_chip_gib))
    total_model_memory_gib = mem_to_consider * source_chips

    usable_target_mem_per_chip_gib = (target_mem_per_chip_gib *
                                      HBM_UTILIZATION_SAFETY_FACTOR)
    raw_chips_mem = max(
        1.0, total_model_memory_gib) / usable_target_mem_per_chip_gib
    chips_needed_mem = _get_next_power_of_two(math.ceil(raw_chips_mem))

    chips_needed = max(chips_needed_compute, chips_needed_mem)
    final_chip_count = _get_next_power_of_two(chips_needed)

    target_dim = int(tgt_spec["topology_dim"])
    num_slices = 1
    chips_per_slice = final_chip_count
    if target_dim == 3:
        topology_str = get_optimal_3d_topology(final_chip_count)
    else:
        if final_chip_count > 256:  # 256 chips is the single-slice limit on 2D pods.
            num_slices = final_chip_count // 256
            chips_per_slice = 256
        topology_str = get_optimal_2d_topology(chips_per_slice)

    return TopologyOnboardingPlan(
        source_platform=src_canonical,
        source_chips=source_chips,
        source_dim=int(src_spec["topology_dim"]),
        source_devices=total_source_devices,
        source_hbm_gib=source_mem_per_chip_gib,
        target_platform=tgt_canonical,
        target_dim=target_dim,
        target_hbm_gib=target_mem_per_chip_gib,
        recommended_chips=final_chip_count,
        recommended_topology=topology_str,
        recommended_slices=num_slices,
        chips_per_slice=chips_per_slice,
        chips_needed_for_compute_parity=chips_needed_compute,
        chips_needed_for_memory=chips_needed_mem,
    )


# ---------------------------------------------------------------------------
# Stacked Partition Layout Helpers (for GridOfRings / RingOfRings trainers)
# ---------------------------------------------------------------------------


class LayoutError(ValueError):
    """The stacked partition config cannot lay out the mesh on the slice."""


@dataclasses.dataclass(frozen=True)
class Placement:
    """Where one mesh axis lands on the physical torus, in words."""

    axis: str
    size: int
    text: str
    dims: tuple[str, ...] = ()


def _size(mesh: Mapping[str, int], axis: str) -> int:
    return int(mesh.get(axis, 1) or 1)


def _sharding_axes(cfg: Mapping[str, Any]) -> list[str]:
    axes = [str(a) for a in cfg.get("sharding_axes") or [] if a]
    if not axes and cfg.get("inner_axis_name") and cfg.get("outer_axis_name"):
        axes = [str(cfg["inner_axis_name"]), str(cfg["outer_axis_name"])]
    return axes


def _stackable_types(cfg: Mapping[str, Any]) -> list[str]:
    raw = cfg.get("stackable_types") or []
    found = [str(t).rsplit(".", 1)[-1].upper() for t in raw]
    if not found:
        found = list(
            _PRESET_TYPES.get(str(cfg.get("partition_type") or ""), ()))
    return found


def _independent_axes(cfg: Mapping[str, Any]) -> list[str]:
    axes = [str(a) for a in cfg.get("independent_axes") or [] if a]
    if not axes and cfg.get("independent_axis"):
        axes = [str(cfg["independent_axis"])]
    return axes


def is_partition_layout_modeled(cfg: Mapping[str, Any] | None) -> bool:
    """Whether `partition_layout` models this partition config's device placement."""
    if not cfg:
        return False
    ptype = str(cfg.get("partition_type") or "")
    if "3D" in ptype or "2DRing" in ptype:
        return False
    axes = _sharding_axes(cfg)
    return bool(axes) and len(_stackable_types(cfg)) == len(axes)


def _tile_shape(size: int, remaining: tuple[int, int]) -> tuple[int, int]:
    """Infers the 2D tile shape for a stacked axis of `size`."""
    if size not in _TILE_SHAPES:
        raise LayoutError(
            f"stacked partition layout has no 2D tile for an axis of size {size}"
        )
    shape = _TILE_SHAPES[size]
    rev = (shape[1], shape[0])

    def fits(s: tuple[int, int]) -> bool:
        return s[0] <= remaining[0] and s[1] <= remaining[1]

    if fits(shape) and fits(rev):
        left = (remaining[0] // shape[0], remaining[1] // shape[1])
        left_rev = (remaining[0] // rev[0], remaining[1] // rev[1])
        if abs(left[0] - left[1]) <= abs(left_rev[0] - left_rev[1]):
            return shape
        return rev
    if fits(shape):
        return shape
    if fits(rev):
        return rev
    raise LayoutError(
        f"a stacked axis of size {size} does not fit the remaining"
        f" {remaining[0]}x{remaining[1]} chips")


def _infer_shapes(axes: Sequence[str], mesh: Mapping[str, int],
                  plane: tuple[int, int]) -> list[tuple[int, int]]:
    """Infers 2D tile shapes when `partition_config` has no `shapes` override."""
    last = 0
    for i, axis in enumerate(axes):
        if _size(mesh, axis) > 1:
            last = i
    shapes = []
    remaining = plane
    for axis in axes[:last]:
        shape = _tile_shape(_size(mesh, axis), remaining)
        shapes.append(shape)
        remaining = (remaining[0] // shape[0], remaining[1] // shape[1])
    shapes.append(remaining)
    shapes += [(1, 1)] * (len(axes) - last - 1)
    return shapes


def _line_text(count: int, stride: int, length: int, dim: str,
               wraps: bool) -> str:
    """Describes how `count` chips `stride` apart sit on a `length`-long ring."""
    if stride == 1 and count == length:
        if not wraps:
            return (f"owns the {length}-long {dim} ring (no wrap-around links"
                    " on this slice)")
        return f"owns the {length}-long {dim} ring (wrap-around links)"
    if stride == 1:
        return (
            f"takes {count} adjacent chips of the {length}-long {dim} ring (a"
            " segment without a wrap-around link)")
    return (f"takes {count} chips {stride} hops apart along the {length}-long"
            f" {dim} ring")


def _tile_text(
    kind: str,
    shape: tuple[int, int],
    stride: tuple[int, int],
    plane: tuple[int, int],
    names: tuple[str, str],
    wraps: tuple[bool, bool],
) -> tuple[str, tuple[str, ...]]:
    """Describes one stacked axis' tile on the 2D plane."""
    a, b = shape
    if a == 1 or b == 1:
        i = 0 if b == 1 else 1
        count = a if b == 1 else b
        if kind == "RING" and count > 2:
            raise LayoutError(
                f"a RING tile of {a}x{b} chips cannot close (snake rings need"
                " two rows or columns)")
        return _line_text(count, stride[i], plane[i], names[i],
                          wraps[i]), (names[i], )
    plane_name = f"{names[0]}-{names[1]}"
    whole = (a, b) == plane and stride == (1, 1)
    where = (
        f"the whole {a}x{b} {plane_name} plane" if whole else
        f"a {a}x{b} block of the {plane[0]}x{plane[1]} {plane_name} plane")
    spread = ("" if stride == (1, 1) else
              f", one chip per {stride[0]}x{stride[1]} inner tile")
    if kind == "RING":
        text = f"forms one closed ring snaking through {where}{spread}"
    else:
        text = f"forms a {a}x{b} grid over {where}{spread} (2D collectives)"
    return text, names


def _independent_dim(
    cfg: Mapping[str, Any],
    mesh: Mapping[str, int],
    d3: Sequence[int],
    cores: int,
) -> tuple[int | None, list[str], str | None]:
    """Finds the physical dimension along which independent axes are placed."""
    independent = _independent_axes(cfg)
    if not independent or (d3[2] <= 1 and cores <= 1):
        if d3[2] > 1 or cores > 1:
            raise LayoutError(
                "without independent axes a stacked partition config needs a 2D"
                " slice with 1 TensorCore per chip")
        return None, [], None
    merge = bool(cfg.get("merge_core_axis_into_independent_ici_axis"))
    core_axis = None
    if cores > 1 and not merge:
        core_axis = next(
            (a for a in reversed(independent) if _size(mesh, a) == cores),
            None)
        if core_axis is None:
            raise LayoutError(
                f"{cores} TensorCores per chip need an independent axis of size"
                f" {cores}; the independent axes are {', '.join(independent)}")
        independent = [a for a in independent if a != core_axis]
    target = math.prod(_size(mesh, a) for a in independent)
    lengths = [d * (cores if merge else 1) for d in d3]
    idx = next((i for i, d in enumerate(lengths) if d == target), None)
    if idx is None:
        raise LayoutError("the independent axes " +
                          " x ".join(f"{a}={_size(mesh, a)}"
                                     for a in independent) +
                          f" (= {target}) match no dimension of the"
                          f" {'x'.join(str(d) for d in d3)} slice")
    return idx, independent, core_axis


def partition_layout(
    cfg: Mapping[str, Any],
    mesh: Mapping[str, int],
    dims: Sequence[int],
    cores_per_chip: int = 1,
) -> list[Placement]:
    """Places every mesh axis according to a stacked partition config.

    Args:
      cfg: The trainer's `partition_config` dictionary.
      mesh: ICI mesh axis sizes of one slice.
      dims: Physical torus dims of the slice in chips, e.g. `[4, 4, 16]`.
      cores_per_chip: Logical XLA devices per chip (2 on `tpu7x`, 1 elsewhere).
    """
    axes = _sharding_axes(cfg)
    types = _stackable_types(cfg)
    if not axes or len(types) != len(axes):
        raise LayoutError("the partition config has no stacked sharding axes")
    independent = _independent_axes(cfg)
    if set(independent) & set(axes):
        raise LayoutError("an axis is both stacked and independent")
    others = [
        a for a, v in mesh.items()
        if int(v) > 1 and a not in axes and a not in independent
    ]
    if others:
        raise LayoutError(
            f"{', '.join(f'{a}={mesh[a]}' for a in others)} is neither a stacked"
            " nor an independent axis of the partition config")
    d3 = [int(d) for d in dims][:3] + [1] * max(0, 3 - len(dims))
    wraps = list(mesh_search.wraparound_dims(dims))[:3]
    wraps += [False] * (3 - len(wraps))
    cores = max(1, int(cores_per_chip))
    idx, independent, core_axis = _independent_dim(cfg, mesh, d3, cores)
    placements: list[Placement] = []
    if idx is not None:
        stride = 1
        line = []
        for axis in reversed(independent):
            n = _size(mesh, axis)
            if n > 1:
                line.append(
                    Placement(
                        axis,
                        n,
                        _line_text(n, stride, d3[idx], _DIM_NAMES[idx],
                                   wraps[idx]),
                        (_DIM_NAMES[idx], ),
                    ))
            stride *= n
        placements += reversed(line)
    if core_axis is not None:
        placements.append(
            Placement(
                core_axis,
                cores,
                f"spans the {cores} TensorCores of each chip",
                ("cores", ),
            ))
    plane_idx = [i for i in range(3) if i != (2 if idx is None else idx)]
    plane = (d3[plane_idx[0]], d3[plane_idx[1]])
    names = (_DIM_NAMES[plane_idx[0]], _DIM_NAMES[plane_idx[1]])
    raw_shapes = cfg.get("shapes")
    if raw_shapes:
        shapes = [(int(s[0]), int(s[1])) for s in raw_shapes]
        if len(shapes) != len(axes):
            raise LayoutError(
                "the shapes override has one entry per stacked axis")
    else:
        shapes = _infer_shapes(axes, mesh, plane)
    acc = (1, 1)
    for axis, kind, shape in zip(axes, types, shapes):
        n = _size(mesh, axis)
        if shape[0] * shape[1] != n:
            if raw_shapes:
                raise LayoutError(
                    f"the shapes override {list(shape)} for {axis} does not match"
                    f" {axis}={n}")
            raise LayoutError(
                f"the remaining {shape[0]}x{shape[1]} chips of the"
                f" {plane[0]}x{plane[1]} {names[0]}-{names[1]} plane do not match"
                f" {axis}={n}")
        new_acc = (acc[0] * shape[0], acc[1] * shape[1])
        if plane[0] % new_acc[0] or plane[1] % new_acc[1]:
            raise LayoutError(
                f"{axis}'s {shape[0]}x{shape[1]} tile does not divide the"
                f" {plane[0]}x{plane[1]} {names[0]}-{names[1]} plane")
        if n > 1:
            text, spans = _tile_text(
                kind, shape, acc, plane, names,
                (wraps[plane_idx[0]], wraps[plane_idx[1]]))
            placements.append(Placement(axis, n, text, spans))
        acc = new_acc
    if acc != plane:
        raise LayoutError(
            f"the stacked axes cover {acc[0]}x{acc[1]} chips of the"
            f" {plane[0]}x{plane[1]} {names[0]}-{names[1]} plane")
    return placements


def _config_label(cfg: Mapping[str, Any]) -> str:
    """Names the config's type, stacked and independent axes and overrides."""
    parts = []
    ptype = str(cfg.get("partition_type") or "partition config")
    if cfg.get("inner_axis_name") and cfg.get("outer_axis_name"):
        parts.append(
            f"inner {cfg['inner_axis_name']}, outer {cfg['outer_axis_name']}")
    else:
        stacked = _sharding_axes(cfg)
        types = _stackable_types(cfg)
        if stacked:
            parts.append("stacked " +
                         ", ".join(f"{a} {t}" for a, t in zip(stacked, types)))
    if cfg.get("shapes"):
        parts.append(f"shapes {[list(s) for s in cfg['shapes']]}")
    independent = _independent_axes(cfg)
    if independent:
        parts.append("independent " + ", ".join(independent))
    return f"{ptype} ({'; '.join(parts)})" if parts else ptype


def _rederived_shapes(cfg: Mapping[str, Any],
                      mesh: Mapping[str, int]) -> list[list[int]] | None:
    """Resizes a `shapes` override to `mesh`, preserving 1D orientations."""
    raw = cfg.get("shapes")
    if not raw:
        return None
    out = []
    for axis, shape in zip(_sharding_axes(cfg), raw):
        a, b = int(shape[0]), int(shape[1])
        n = _size(mesh, axis)
        if a * b == n:
            out.append([a, b])
        elif b == 1:
            out.append([n, 1])
        elif a == 1:
            out.append([1, n])
        else:
            return None
    return out


def _fits(
    cfg: Mapping[str, Any],
    mesh: Mapping[str, int],
    dims: Sequence[int],
    cores_per_chip: int,
) -> bool:
    try:
        partition_layout(cfg, mesh, dims, cores_per_chip)
    except LayoutError:
        return False
    return True


def partition_layout_shapes(
    cfg: Mapping[str, Any],
    mesh: Mapping[str, int],
    dims: Sequence[int],
    cores_per_chip: int = 1,
) -> list[list[int]]:
    """Returns the 2D `[rows, cols]` tiles used for each stacked axis."""
    if cfg.get("shapes"):
        return [[int(s[0]), int(s[1])] for s in cfg["shapes"]]
    d3 = [int(d) for d in dims][:3] + [1] * max(0, 3 - len(dims))
    idx, _, _ = _independent_dim(cfg, mesh, d3, max(1, int(cores_per_chip)))
    plane_idx = [i for i in range(3) if i != (2 if idx is None else idx)]
    plane = (d3[plane_idx[0]], d3[plane_idx[1]])
    return [list(s) for s in _infer_shapes(_sharding_axes(cfg), mesh, plane)]


def suggest_partition_layout_change(
    cfg: Mapping[str, Any],
    mesh: Mapping[str, int],
    dims: Sequence[int],
    cores_per_chip: int = 1,
) -> tuple[str, dict[str, Any] | None]:
    """Finds the smallest partition-config edit that lays out `mesh`."""
    shape_options: list[str] = []
    best_cfg: dict[str, Any] | None = None
    if cfg.get("shapes") and _fits(dict(cfg, shapes=None), mesh, dims,
                                   cores_per_chip):
        no_shapes = dict(cfg, shapes=None)
        inferred = partition_layout_shapes(no_shapes, mesh, dims,
                                           cores_per_chip)
        shape_options.append(
            f"drop the shapes override (the trainer then infers {inferred})")
        best_cfg = no_shapes
    shapes = _rederived_shapes(cfg, mesh)
    if shapes and shapes != [list(s) for s in cfg.get("shapes") or []]:
        rederived_cfg = dict(cfg, shapes=shapes)
        if _fits(rederived_cfg, mesh, dims, cores_per_chip):
            shape_options.append(f"re-derive the shapes override as {shapes}")
            if best_cfg is None:
                best_cfg = rederived_cfg
    if shape_options:
        return " or ".join(shape_options), best_cfg
    stacked = _sharding_axes(cfg)
    independent = _independent_axes(cfg)
    for i, s_axis in enumerate(stacked):
        for j, i_axis in enumerate(independent):
            new_stacked = list(stacked)
            new_stacked[i] = i_axis
            new_independent = list(independent)
            new_independent[j] = s_axis
            trial: dict[str, Any] = dict(
                cfg,
                sharding_axes=new_stacked,
                independent_axes=new_independent,
                shapes=None,
            )
            if cfg.get("inner_axis_name") and cfg.get("outer_axis_name"):
                trial["inner_axis_name"], trial[
                    "outer_axis_name"] = new_stacked[:2]
            if _fits(trial, mesh, dims, cores_per_chip):
                role = (("inner_axis_name",
                         "outer_axis_name")[i] if cfg.get("inner_axis_name")
                        and i < 2 else f"stacked axis {i}")
                drop = " and drop the shapes override" if cfg.get(
                    "shapes") else ""
                return (
                    f"set {role}={i_axis} and make {s_axis} an independent axis{drop}",
                    trial,
                )
    return "", None


def _format_placements(
        placements: Sequence[Placement],
        group: Sequence[str] = (),
        group_name: str = "MoE routing group",
) -> str:
    parts = [f"`{p.axis}`={p.size} {p.text}" for p in placements]
    text = "; ".join(parts) or "no axis larger than 1"
    by_axis = {p.axis: p for p in placements}
    active = [a for a in group if a in by_axis]
    if active:
        spans = []
        for axis in active:
            for d in by_axis[axis].dims:
                if d not in spans:
                    spans.append(d)
        where = " and ".join("the chips' TensorCores" if d == "cores" else d
                             for d in spans)
        devices = math.prod(by_axis[a].size for a in active)
        text += (f". {group_name} " + " x ".join(f"`{a}`" for a in active) +
                 f" ({devices} devices) spans {where}")
    return text


def describe_partition_layout(
    cfg: Mapping[str, Any] | None,
    mesh: Mapping[str, int],
    dims: Sequence[int] | None,
    cores_per_chip: int = 1,
    group: Sequence[str] = (),
    group_name: str = "MoE routing group",
    label: bool = True,
) -> str | None:
    """Describes where a stacked `partition_config` places each mesh axis."""
    if cfg is None or not dims or not is_partition_layout_modeled(cfg):
        return None
    prefix = f"per the trainer's {_config_label(cfg)}: " if label else ""
    try:
        placements = partition_layout(cfg, mesh, dims, cores_per_chip)
    except LayoutError as e:
        change, new_cfg = suggest_partition_layout_change(
            cfg, mesh, dims, cores_per_chip)
        fix = f"; to use it, {change}" if change else ""
        if new_cfg is not None:
            after = _format_placements(
                partition_layout(new_cfg, mesh, dims, cores_per_chip),
                group,
                group_name,
            )
            fix += f" (then {after})"
        return f"{prefix}does not fit this mesh ({e}){fix}"
    return prefix + _format_placements(placements, group, group_name)


# Short aliases matching `partition_layout` call sites.
is_modeled = is_partition_layout_modeled
layout = partition_layout
layout_shapes = partition_layout_shapes
suggest_change = suggest_partition_layout_change
describe = describe_partition_layout
