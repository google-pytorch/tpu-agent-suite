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
"""Public TPU hardware specifications and platform normalization utilities.

Hardware numbers match the verified TPU architecture reference pages in
`package/data/tpu-knowledge-base/pages/hw/{v4,v5e,v5p,v6e,v7x}/overview.md`:

- `v4`: 32 GiB HBM per chip (megacore: 1 logical XLA device per chip), 3D ICI
  torus, 275 TFLOPs BF16, 1200 GB/s HBM bandwidth, 240 GB/s bidirectional ICI.
- `v5e`: 16 GiB HBM per chip (1 TensorCore per chip), 2D ICI torus (up to
  16x16 = 256 chips), 197 TFLOPs BF16, 819 GB/s HBM bandwidth, 400 GB/s ICI.
- `v5p`: 95 GiB HBM per chip (megacore: 1 logical XLA device per chip), 3D ICI
  torus, 459 TFLOPs BF16, 2765 GB/s HBM bandwidth, 1200 GB/s bidirectional ICI.
- `v6e` (Trillium): 32 GiB HBM per chip (1 TensorCore per chip), 2D ICI torus
  (up to 16x16 = 256 chips), 918 TFLOPs BF16, 1638 GB/s HBM bandwidth,
  800 GB/s bidirectional ICI.
- `tpu7x` (Ironwood): dual-chiplet package (192 GiB HBM per chip, 2 logical XLA
  devices per chip at 96 GiB HBM per device), 3D ICI torus, 2307 TFLOPs BF16
  (4614 TFLOPs FP8) per chip, 7380 GB/s HBM bandwidth per chip (3690 GB/s per
  chiplet), 1200 GB/s bidirectional ICI.
"""

from __future__ import annotations

import re
import types
from collections.abc import Mapping
from typing import Any

# Canonical public TPU generation keys.
PUBLIC_PLATFORMS = ("v4", "v5e", "v5p", "v6e", "tpu7x")

PLATFORM_SPECS: Mapping[str, Mapping[str, Any]] = types.MappingProxyType({
    "v4":
    types.MappingProxyType({
        "devices_per_chip": 1,
        "hbm_gib": 32.0,
        "hbm_gib_per_core": 32.0,
        "topology_dim": 3,
        "default_axis_order": ("x", "y", "z"),
        "peak_flops": 275e12,
        "peak_bf16_tflops": 275.0,
        "mem_bw_gbs": 1200.0,
        "ici_bw_bidirectional_gbs": 240.0,
        "tile_alignment": 128,
        "max_untuned_k": 8192,
    }),
    "v5e":
    types.MappingProxyType({
        "devices_per_chip": 1,
        "hbm_gib": 16.0,
        "hbm_gib_per_core": 16.0,
        "topology_dim": 2,
        "default_axis_order": ("x", "y"),
        "peak_flops": 197e12,
        "peak_bf16_tflops": 197.0,
        "mem_bw_gbs": 819.0,
        "ici_bw_bidirectional_gbs": 400.0,
        "tile_alignment": 128,
        "min_moe_model_shards": 8,
        "max_untuned_k": 4096,
    }),
    "v5p":
    types.MappingProxyType({
        "devices_per_chip": 1,
        "hbm_gib": 95.0,
        "hbm_gib_per_core": 95.0,
        "topology_dim": 3,
        "default_axis_order": ("x", "y", "z"),
        "peak_flops": 459e12,
        "peak_bf16_tflops": 459.0,
        "mem_bw_gbs": 2765.0,
        "ici_bw_bidirectional_gbs": 1200.0,
        "tile_alignment": 128,
        "max_untuned_k": 8192,
    }),
    "v6e":
    types.MappingProxyType({
        "devices_per_chip": 1,
        "hbm_gib": 32.0,
        "hbm_gib_per_core": 32.0,
        "topology_dim": 2,
        "default_axis_order": ("x", "y"),
        "peak_flops": 918e12,
        "peak_bf16_tflops": 918.0,
        "mem_bw_gbs": 1638.0,
        "ici_bw_bidirectional_gbs": 800.0,
        "tile_alignment": 128,
        "max_untuned_k": 8192,
    }),
    "tpu7x":
    types.MappingProxyType({
        "devices_per_chip": 2,
        "hbm_gib": 192.0,
        "hbm_gib_per_core": 96.0,
        "topology_dim": 3,
        "default_axis_order": ("x", "y", "z"),
        "peak_flops": 2307e12,
        "peak_bf16_tflops": 2307.0,
        "peak_fp8_tflops": 4614.0,
        "mem_bw_gbs": 7380.0,
        "ici_bw_bidirectional_gbs": 1200.0,
        "tile_alignment": 256,
        "max_untuned_k": 8192,
    }),
})

# Public accelerator string aliases mapped to canonical keys in PLATFORM_SPECS.
PLATFORM_ALIASES: Mapping[str, str] = types.MappingProxyType({
    "v4":
    "v4",
    "tpuv4":
    "v4",
    "v5e":
    "v5e",
    "tpuv5e":
    "v5e",
    "v5litepod":
    "v5e",
    "tpuv5litepod":
    "v5e",
    "v5p":
    "v5p",
    "v5":
    "v5p",
    "tpuv5p":
    "v5p",
    "tpuv5":
    "v5p",
    "v6e":
    "v6e",
    "v6":
    "v6e",
    "tpuv6e":
    "v6e",
    "trillium":
    "v6e",
    "tpu7x":
    "tpu7x",
    "tpu7":
    "tpu7x",
    "tpuv7":
    "tpu7x",
    "tpuv7x":
    "tpu7x",
    "v7":
    "tpu7x",
    "v7x":
    "tpu7x",
    "ironwood":
    "tpu7x",
})

# Public MaxText ahead-of-time (AOT) compilation baseline flags.
MAXTEXT_BASELINE_COMPILE_FLAGS: tuple[str, ...] = (
    "compile_topology_num_slices=1",
    "dataset_type=synthetic",
    "steps=1",
    "enable_checkpointing=False",
)

_CHIP_COUNT_SUFFIX_RE = re.compile(r"[-_]\d+$")


def normalize_platform(platform: str | None) -> str | None:
    """Resolves a public TPU generation string (e.g. 'tpu7x-8', 'v6e-16') to its canonical key."""
    if not platform or not isinstance(platform, str):
        return None
    cleaned = platform.strip().lower()
    if not cleaned:
        return None
    without_count = _CHIP_COUNT_SUFFIX_RE.sub("", cleaned)
    for candidate in (cleaned, without_count):
        compact = re.sub(r"[\s_-]+", "", candidate)
        if compact in PLATFORM_ALIASES:
            return PLATFORM_ALIASES[compact]
    # Substring fallback for composite public names like "tpu-v5litepod-256".
    compact_all = re.sub(r"[\s_-]+", "", without_count)
    if "7x" in compact_all or "v7" in compact_all or "ironwood" in compact_all:
        return "tpu7x"
    if "v6e" in compact_all or "trillium" in compact_all:
        return "v6e"
    if "v5lite" in compact_all or "v5e" in compact_all:
        return "v5e"
    if "v5p" in compact_all or compact_all.endswith("v5"):
        return "v5p"
    if "v4" in compact_all:
        return "v4"
    return None


def get_platform_spec(platform: str | None) -> Mapping[str, Any] | None:
    """Returns the verified hardware specification dict for `platform`, or None."""
    canonical = normalize_platform(platform)
    if canonical is None:
        return None
    return PLATFORM_SPECS.get(canonical)


def get_devices_per_chip(platform: str | None) -> int | None:
    """Returns logical XLA devices per physical TPU chip (2 on tpu7x, 1 elsewhere), or None."""
    spec = get_platform_spec(platform)
    if spec is None:
        return None
    return int(spec["devices_per_chip"])
