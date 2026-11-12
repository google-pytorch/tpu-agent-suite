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
"""Constants, errors and helpers shared by the `gke_job` modules.

Also `StagingPlan`, how one Job's files travel, and `StagingClient`, the GCS
client the staging calls share.
"""

from __future__ import annotations

import dataclasses
import datetime
import pathlib
import re
from typing import Any

from .. import base, gcs_staging

Path = pathlib.Path

# Node labels used for placement. The TPU labels themselves are in `base`.
HOSTNAME_LABEL = "kubernetes.io/hostname"
# Set on nodes with --workload-metadata=GKE_METADATA. Only these nodes give a
# pod its own Workload Identity token.
WORKLOAD_IDENTITY_LABEL = "iam.gke.io/gke-metadata-server-enabled"
NODEPOOL_LABEL = "cloud.google.com/gke-nodepool"

# Kueue admission labels. Applied only when a queue is configured.
QUEUE_LABEL = "kueue.x-k8s.io/queue-name"
# Opts a workload in to GKE managed ML diagnostics. Without it the workload
# is missing from the Console Diagnostics view. Harmless without the addon.
MLDIAGNOSTICS_LABEL = "managed-mldiagnostics-gke"
PRIORITY_LABEL = "kueue.x-k8s.io/priority-class"

# The torch_tpu images keep their own software under /workspace, so a script
# volume mounted there fails container creation. /app is free.
SCRIPT_MOUNT = "/app"
# ConfigMap key for the i-th --input. Keys can't contain `/`, and numbering
# keeps them clear of the script's key.
INPUT_KEY = "tpu-agent-suite-input-{}"
# Why outputs went to the log when no bucket is set. Otherwise the reason is
# the error code that blocked signing their upload URLs.
NO_BUCKET = "NO_BUCKET"

DEFAULT_NAMESPACE = "default"
# Fallback for a topology that can't be parsed; prefer `chips_for_topology`.
# A wrong count fails quietly: GKE turns it into a node selector no node
# matches, and the pod sits Pending as if the cluster were full.
DEFAULT_CHIPS = 4
# The pod label (on older clusters, only the annotation) an Indexed Job puts
# its index under. 0 is the leader of a multi-host Job.
COMPLETION_INDEX_KEY = "batch.kubernetes.io/job-completion-index"
# Blanked on a multi-host Job. It is one slice, and GKE's TPU webhook may
# otherwise set them as if it were several.
MEGASCALE_ENV = ("MEGASCALE_NUM_SLICES", "MEGASCALE_SLICE_ID",
                 "MEGASCALE_PORT", "MEGASCALE_COORDINATOR_ADDRESS")
DEFAULT_WAIT_SEC = 900
SUBMIT_URL_EXP_SEC = 604800
# A serving job should outlive the session, so one crash must not end it.
SERVE_BACKOFF_LIMIT = 6
POLL_INTERVAL_SEC = 5

# Size in GiB of the memory-backed /dev/shm. The runtime default of 64 MB is
# too small for vLLM. Without hf_cache_pvc it also holds the model and compile
# caches (Gemma-4-31B needs 76 GiB). The pod requests no memory, so it is
# evicted first under pressure. Keep the cap below the smallest 4-chip host
# (v5e, 192 GB).
SHM_MOUNT = "/dev/shm"
DEFAULT_SHM_SIZE_GB = 128
# The cap in bytes, for sizing downloads. statvfs in the pod may report the
# node's whole memory instead.
SHM_LIMIT_ENV = "TPU_AGENT_SUITE_SHM_LIMIT_BYTES"
# Set in a --cpu-requests Job to the profile's TPU generation (v6e, tpu7x,
# ...), so checks that only plan for that TPU can run without a chip.
CHIPLESS_TPU_ENV = "TPU_AGENT_SUITE_CHIPLESS_TPU"
# A --cpu-requests `run` that exits with this, or never starts, is retried
# once on a TPU node.
NEEDS_TPU_EXIT_CODE = 75


def k8s_name(raw: str) -> str:
    """`raw` lowercased, with runs of other than [a-z0-9-] made '-'."""
    return re.sub(r"[^a-z0-9-]+", "-", raw.strip().lower()).strip("-")


def positive_int(field: str, value: Any, minimum: int = 1) -> int:
    """`value` as an int >= `minimum`, or an INVALID_FIELD error naming it."""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise base.ComputeConfigError(
            f"Backend 'gke_job' needs an integer '{field}', got {value!r}.",
            error_code="INVALID_FIELD",
        ) from exc
    if number < minimum:
        raise base.ComputeConfigError(
            f"Backend 'gke_job' needs '{field}' >= {minimum}, got {number}.",
            error_code="INVALID_FIELD",
        )
    return number


def parse_requests(value: Any,
                   field: str = "colocate_requests") -> dict[str, str]:
    """`NAME=QTY,...` (or a dict) as pod resource requests; blank is {}."""
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    pairs = [
        p.partition("=") for p in str(value or "").split(",") if p.strip()
    ]
    if any(not name.strip() or not qty.strip() for name, _, qty in pairs):
        raise base.ComputeConfigError(
            f"Backend 'gke_job' needs '{field}' as "
            f"NAME=QTY,..., got {value!r}.",
            error_code="INVALID_FIELD",
        )
    return {name.strip(): qty.strip() for name, _, qty in pairs}


def pod_index(pod: dict[str, Any]) -> int | None:
    """An Indexed Job pod's index, or None for a pod of a plain Job."""
    meta = pod.get("metadata", {})
    raw = ((meta.get("labels") or {}).get(COMPLETION_INDEX_KEY)
           or (meta.get("annotations") or {}).get(COMPLETION_INDEX_KEY))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def is_not_found(exc: Exception) -> bool:
    """Whether kubectl said the object is not there. It spells it both ways."""
    return bool(re.search(r"not ?found", str(exc), re.IGNORECASE))


def parse_time(stamp: str | None) -> float | None:
    """A Kubernetes RFC 3339 timestamp as epoch seconds, or None."""
    if not stamp:
        return None
    try:
        return datetime.datetime.fromisoformat(stamp.replace(
            "Z", "+00:00")).timestamp()
    except ValueError:
        return None


def rfc3339(epoch: float) -> str:
    """Epoch seconds as the UTC timestamp a Cloud Logging query compares."""
    return datetime.datetime.fromtimestamp(
        epoch, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclasses.dataclass
class StagingPlan:
    """How a Job's files travel, as `file_transfer.stage_files` arranged it."""

    configmap_files: list[str] = dataclasses.field(default_factory=list)
    mounted_inputs: list[tuple[str,
                               str]] = dataclasses.field(default_factory=list)
    in_urls: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    out_urls: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    logged_outputs: list[str] = dataclasses.field(default_factory=list)
    # Why outputs go to the log: NO_BUCKET, or the error code that kept
    # their upload URLs from being signed.
    logged_because: str = ""
    # Set before the first upload, so a failure after it knows to clean up.
    uploaded: bool = False

    def moves_files(self) -> bool:
        """Whether the job has any input or output to stage."""
        return bool(self.mounted_inputs or self.in_urls or self.out_urls
                    or self.logged_outputs)


class StagingClient:
    """GCS client for staging calls, built once from the staging credential."""

    def __init__(self, service_account: str | None, key: str | None) -> None:
        self.service_account = service_account
        self.key = key
        self._client: Any = None

    def client_kwargs(self, best_effort: bool = False) -> dict[str, Any]:
        """`client=` kwargs for gcs_staging calls, or {} to use ADC.

        One cached client, built from the staging credential, serves every
        call. With `best_effort`, a credential that fails to build falls back
        to ADC instead of raising.
        """
        if not (self.service_account or self.key):
            return {}
        if self._client is None:
            try:
                self._client = gcs_staging.get_storage_client(
                    credentials=gcs_staging.staging_credentials(
                        self.service_account, self.key))
            except base.ComputeConfigError:
                if best_effort:
                    return {}
                raise
        return {"client": self._client}
