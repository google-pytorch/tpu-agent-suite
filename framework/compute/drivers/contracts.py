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
"""Data contracts returned by compute drivers."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class HardwareProfile:
    """Hardware topology and capacity profile returned by compute inspect.

    Why this is needed:
        Workload agents (e.g. kernel autotuning, tensor sharding, and model onboarding)
        must know physical accelerator characteristics to generate valid execution
        plans and avoid runtime crashes. Specifically:
        - Accelerator generation and HBM capacity determine maximum model parameter count,
          batch sizes, and peak theoretical TFLOPS for MFU (Model FLOPs Utilization) calculation.
        - Physical mesh topology (e.g., "2x2x1") dictates valid shard mesh dimensions
          and communication axes for distributed training/inference (e.g. FSDP, TP, SP).
        - Chip availability (`available_chips` vs `chips`, `held_by`, `status`) detects
          hardware contention before job launch so agents do not clash with active workloads.

    Where it is used:
        - Returned by `ComputeDriver.inspect()` across all drivers (local, GCE TPU VM, GKE).
        - Emitted in CLI format via `runtime.py compute inspect [--json]`.
        - Consumed by optimization recipes (e.g. tpu-kernel-development benchmarking, tpu-sharding layout
          selection) to validate device health and compute peak performance baselines.
    """
    accelerator: str  # e.g. "v6e", "v5p", "v5e", "v4", "v7", "none"
    topology: str  # e.g. "2x2x1", "1x1", "none"
    chips: int  # number of chips
    hbm_gb: float  # HBM per chip in GB
    status: str  # "healthy", "busy", "no_accelerator", "degraded", etc.
    # Unheld chips ready for execution. `None` means the driver could not
    # measure contention -- deliberately distinct from 0. Coercing an
    # unmeasured value to `chips` would report a fully occupied host as free,
    # and coercing it to 0 would report a healthy host as saturated; both
    # mislead a scheduler, so the unknown case stays visible as null.
    available_chips: int | None = None
    held_by: list[str] = dataclasses.field(
        default_factory=list)  # PIDs/processes holding chips
    details: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The fields as a JSON-ready dict."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class SynchronousExecutionResult:
    """Result of a synchronous execution via compute run."""
    exit_code: int
    stdout: str
    stderr: str
    details: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The fields as a JSON-ready dict."""
        return dataclasses.asdict(self)


# Why a job is PENDING, as `details["pending_reason"]` on a status read. A
# driver that can tell sets one; a reader that cannot tell them apart should
# wait. Absent means the driver does not know, which also reads as wait.
PENDING_QUEUED = "queued"  # a queue holds it and admits in order
PENDING_STARTING = "starting"  # on a node, pulling or starting
# A queue admitted it; no node fits.
PENDING_ADMITTED_NO_NODE = "admitted_no_node"
PENDING_UNSCHEDULED = "unscheduled"  # no node, no queue: no claim on chips
PENDING_REASONS = frozenset({
    PENDING_QUEUED, PENDING_STARTING, PENDING_ADMITTED_NO_NODE,
    PENDING_UNSCHEDULED
})


@dataclasses.dataclass
class AsynchronousExecutionResult:
    """Status information for an asynchronous job.

    While `status` is PENDING, `details["pending_reason"]` may say why, one
    of PENDING_REASONS.
    """
    job_id: str
    status: str  # "PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"
    start_time: str | None = None
    end_time: str | None = None
    exit_code: int | None = None
    error: str | None = None
    details: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The fields as a JSON-ready dict."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class PreflightCheck:
    """The result of checking one prerequisite of a backend.

    `status` is `ok`, `warn` (degraded or unconfirmed; never blocks) or
    `fail` (the backend cannot work until it is fixed). `fix` is what a person
    runs or changes to resolve a problem, and `fields` are the profile settings
    to ask about again when it fails.
    """
    name: str
    status: str
    detail: str = ""
    fix: str = ""
    fields: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PreflightReport:
    """What `compute preflight` found: the target, and every check on it."""
    backend: str
    target: str  # described for a person to recognise and confirm
    reachable: bool  # whether the target answered at all
    checks: list[PreflightCheck] = dataclasses.field(default_factory=list)

    @property
    def status(self) -> str:
        """The worst check: `fail`, else `warn`, else `ok`."""
        statuses = {check.status for check in self.checks}
        return next((s for s in ("fail", "warn") if s in statuses), "ok")

    def to_dict(self) -> dict[str, Any]:
        """The fields and the overall `status` as a JSON-ready dict."""
        return {**dataclasses.asdict(self), "status": self.status}
