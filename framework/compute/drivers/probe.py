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
"""Self-contained TPU host probe powered by tpu_info."""

from __future__ import annotations

import glob
import json
import math
import os
import sys
import urllib.request
from typing import Any


class MissingPackageError(RuntimeError):
    """A package the probe imports is not installed for this interpreter.

    Plain words, since this file also runs alone on targets where nothing
    else of the runtime is importable; callers on the controller turn
    `package` into an ask-the-user message.
    """

    def __init__(self, package: str) -> None:
        self.package = package
        super().__init__(f"{package} is not installed for {sys.executable}")


def _tpu_device():
    """Import tpu_info at first use, not at module import.

    This module is pulled in by the whole compute CLI import chain, which
    also runs on controller machines (laptops driving a remote TPU) that
    never probe local hardware. An import-time failure there takes down
    every runtime command -- including hook dispatch -- so the dependency
    is enforced only where probing actually happens.
    """
    try:
        from tpu_info import device as tpu_device
    except ImportError as exc:
        raise MissingPackageError(exc.name or "tpu_info") from exc
    return tpu_device


_METADATA_HOST = os.environ.get("GCE_METADATA_HOST",
                                "metadata.google.internal")
METADATA_URL = f"http://{_METADATA_HOST}/computeMetadata/v1"
_METADATA_ROOT = f"{METADATA_URL}/instance/attributes"
_LIBTPU_LOCKFILE = "/tmp/libtpu_lockfile"

_ALIASES: dict[str, tuple[str, ...]] = {
    "v4": ("v4", ),
    "v5e": ("v5e", "v5lite", "ct5lp", "ct5e"),
    "v5p": ("v5p", "ct5p"),
    "v6e": ("v6e", "ct6e", "trillium"),
    "tpu7x": ("tpu7x", "tpu7", "v7", "7x"),
}

# 2D torus for v5e/v6e; 3D torus for v4/v5p/tpu7x.
_TOPOLOGY_2D = {1: "1x1", 4: "2x2", 8: "2x4", 16: "4x4"}
_TOPOLOGY_3D = {1: "1x1x1", 4: "2x2x1", 8: "2x2x2", 16: "2x2x4"}


def generation_for_name(accel_type: str) -> str | None:
    """Resolve a canonical generation name from an accelerator type string."""
    lowered = (accel_type or "").lower().replace("-", "").replace("_", "")
    for name, aliases in _ALIASES.items():
        if any(a.replace("-", "") in lowered for a in aliases):
            return name
    return None


def spec_for(generation: str | None) -> dict[str, Any]:
    """Return the hardware spec for a generation."""
    if generation:
        enum_name = "V" + generation.lower().removeprefix("tpu").removeprefix(
            "v").upper()
        chip = getattr(_tpu_device().TpuChip, enum_name, None)
        if chip and hasattr(chip, "value"):
            return {
                "hbm_gb": float(chip.value.hbm_gib),
                "devices_per_chip": chip.value.devices_per_chip,
            }
    return {"hbm_gb": 0.0, "devices_per_chip": 1}


def topology_for(chips: int, generation: str | None = None) -> str:
    """Return the topology a chip count implies, or 'unknown'/'none'."""
    if chips <= 0:
        return "none"
    table = _TOPOLOGY_2D if generation in ("v5e", "v6e") else _TOPOLOGY_3D
    return table.get(chips, "unknown")


def product(text: str, sep: str = "x") -> int:
    """Product of the positive integers in `text` split on `sep`, or 0."""
    try:
        return max(0, math.prod(int(d) for d in text.lower().split(sep)))
    except (AttributeError, ValueError):
        return 0


def read_metadata() -> dict[str, str]:
    """Read TPU attributes from the GCE metadata server."""
    meta: dict[str, str] = {}
    try:
        req = urllib.request.Request(
            f"{_METADATA_ROOT}/?recursive=true&alt=json",
            headers={"Metadata-Flavor": "Google"})
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, str):
                        meta[k] = v
                if "tpu-env" in data and isinstance(data["tpu-env"], str):
                    for line in data["tpu-env"].splitlines():
                        if ":" in line:
                            k, v = line.split(":", 1)
                            meta[k.strip()] = v.strip().strip("'\"")
                if "accelerator-type" in data and isinstance(
                        data["accelerator-type"], str):
                    meta.setdefault("ACCELERATOR_TYPE",
                                    data["accelerator-type"].split("/")[-1])
                if "kube-labels" in data and isinstance(
                        data["kube-labels"], str):
                    for pair in data["kube-labels"].split(","):
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            meta.setdefault(k.strip(), v.strip().strip("'\""))
    except Exception:
        pass
    return meta


def scan_pci() -> tuple[int, str | None]:
    """Count Google TPU PCI functions and identify the generation via tpu_info."""
    try:
        chip, count = _tpu_device().get_local_chips()
        if chip:
            return count, generation_for_name(chip.value.name)
    except Exception:
        pass
    return 0, None


def list_devices() -> tuple[list[str], list[str]]:
    """Return (accel devices, vfio devices) present on this host."""
    vfio = sorted(d for d in glob.glob("/dev/vfio/[0-9]*")
                  if not d.endswith("/vfio"))
    if vfio:
        return [], vfio
    sys_accel = "/sys/class/accel"
    has_sys_accel = os.path.isdir(sys_accel)
    accel = sorted(
        d for d in glob.glob("/dev/accel[0-9]*") if not has_sys_accel
        or os.path.exists(os.path.join(sys_accel, os.path.basename(d))))
    return accel, vfio


def scan_holders() -> tuple[list[str], set[str], set[str]]:
    """Find processes holding TPU devices or the libtpu lockfile."""
    descriptions: list[str] = []
    held_devices: set[str] = set()
    lock_holders: set[str] = set()

    for pid_dir in glob.glob("/proc/[0-9]*"):
        pid = os.path.basename(pid_dir)
        fd_dir = os.path.join(pid_dir, "fd")
        if not os.path.isdir(fd_dir):
            continue
        try:
            entries = os.listdir(fd_dir)
        except (PermissionError, FileNotFoundError, OSError):
            continue
        command = None
        for fd in entries:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except (OSError, UnicodeError):
                continue
            if target.startswith("/dev/accel") or (
                    target.startswith("/dev/vfio/")
                    and not target.endswith("/vfio")):
                held_devices.add(target)
                if command is None:
                    command = _read_cmd(pid_dir, pid)
                descriptions.append(f"PID {pid} ({command}) on {target}")
            elif target == _LIBTPU_LOCKFILE:
                lock_holders.add(pid)

    if lock_holders:
        descriptions.append("libtpu_lockfile open by PIDs: " +
                            ", ".join(sorted(lock_holders)))
    return descriptions, held_devices, lock_holders


def _read_cmd(pid_dir: str, pid: str) -> str:
    try:
        with open(os.path.join(pid_dir, "cmdline"), "rb") as handle:
            return handle.read().replace(b"\x00", b" ").decode(
                errors="replace").strip()[:80] or f"PID {pid}"
    except Exception:
        return f"PID {pid}"


def _lockfile_is_locked() -> bool:
    """Report whether /tmp/libtpu_lockfile is held by an flock."""
    if not os.path.exists(_LIBTPU_LOCKFILE):
        return False
    try:
        import fcntl

        with open(_LIBTPU_LOCKFILE, "a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle, fcntl.LOCK_UN)
                return False
            except (BlockingIOError, OSError):
                return True
    except Exception:
        return False


def probe_host() -> dict[str, Any]:
    """Probe this host and return a HardwareProfile-shaped dictionary."""
    meta = read_metadata()
    accel_devs, vfio_devs = list_devices()
    pci_count, pci_generation = scan_pci()

    meta_type = (meta.get("ACCELERATOR_TYPE") or meta.get("TYPE")
                 or meta.get("cloud.google.com/gke-tpu-accelerator")
                 or meta.get("cloud.google.com/machine-family") or "")
    generation = generation_for_name(meta_type) or pci_generation
    spec = spec_for(generation)
    per_chip = spec["devices_per_chip"]

    chips = (len(accel_devs) or len(vfio_devs) or pci_count) // per_chip
    topology = meta.get("TOPOLOGY") or meta.get(
        "cloud.google.com/gke-tpu-topology")
    if not topology and meta.get("CHIPS_PER_HOST_BOUNDS"):
        topology = meta["CHIPS_PER_HOST_BOUNDS"].replace(",", "x")
    if not topology:
        topology = topology_for(chips, generation)

    descriptions, held_devices, lock_holders = scan_holders()
    held_chips = -(-len(held_devices) // per_chip)
    locked = bool(lock_holders) or _lockfile_is_locked()
    if locked and not lock_holders:
        descriptions.append("libtpu_lockfile is locked by an active process")

    available = 0 if (chips <= 0 or locked) else max(0, chips - held_chips)
    if chips <= 0:
        status = "no_accelerator"
    elif available == 0:
        status = "busy"
    elif available < chips:
        status = "partially_busy"
    else:
        status = "healthy"

    details: dict[str, Any] = {
        "accelerator_type": meta_type or generation or "",
        "accel_devices": accel_devs,
        "vfio_devices": vfio_devs,
        "pci_functions": pci_count,
        "devices_per_chip": per_chip,
    }
    # Tensor parallelism is sized to devices, not chips, so report the device
    # count and per-device HBM under the same keys gke_job uses. An unknown
    # generation falls back to hbm_gb 0 and one device per chip; say so
    # rather than let that fallback read as a measurement.
    if spec["hbm_gb"]:
        details["job_devices"] = chips * per_chip
        details["hbm_gb_per_device"] = spec["hbm_gb"] / per_chip
    elif chips > 0:
        details["hbm_gb_note"] = (
            f"unknown accelerator '{meta_type or generation or 'tpu'}'; "
            "hbm_gb and devices not reported")

    return {
        "accelerator": generation or ("tpu" if chips > 0 else "none"),
        "topology": topology,
        "chips": chips,
        "hbm_gb": spec["hbm_gb"],
        "status": status,
        "available_chips": available,
        "held_by": descriptions,
        "details": details,
    }


if __name__ == "__main__":
    print(json.dumps(probe_host()))
