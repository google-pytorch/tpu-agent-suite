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
"""`gke_job` driver settings: `GkeJobConfig`, `config_from_params` and the
CLI flags from `add_cli_flags`.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Callable

from .. import base, gcs_staging
from ..setup_fields import VALID_PRIORITIES
from .gke_common import (DEFAULT_CHIPS, DEFAULT_NAMESPACE, DEFAULT_SHM_SIZE_GB,
                         k8s_name, parse_requests, positive_int)
from .manifest import (COMPILE_CACHE_MAX_AGE_DAYS, COMPILE_CACHE_SUBDIR,
                       HF_CACHE_ENV, HF_CACHE_MOUNT, chips_for_topology,
                       hosts_for)

# What a PersistentVolumeClaim's name may be: a DNS-1123 subdomain.
PVC_NAME = re.compile(r"[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?")

# A Job's name is copied into a pod label, and label values max out at 63.
MAX_JOB_NAME_LEN = 63
# Leaves room for the job's own name and the run suffix.
MAX_JOB_NAME_PREFIX_LEN = 30


@dataclasses.dataclass
class GkeJobConfig:
    """A `gke_job` profile's settings, normalised.

    `image`, `tpu_accelerator` and `tpu_topology` are only required at
    dispatch (`require_dispatch`). Exec-into-pod callers and `inspect` on a
    cluster without a chosen node pool don't need them.
    """

    namespace: str | None = None
    image: str | None = None
    tpu_accelerator: str | None = None
    tpu_topology: str | None = None
    # Chips per pod. A slice larger than one node is `tpu_hosts` pods.
    tpu_chips: int = DEFAULT_CHIPS
    tpu_hosts: int = 1
    # Set by hand, the split is kept. Derived from the topology, it is a
    # guess the nodes can correct (see `fit_to_nodes`).
    chips_explicit: bool = False
    queue: str | None = None
    priority: str | None = None
    declared_duration_minutes: int | None = None
    kubeconfig: str | None = None
    # Every kubectl call names this context; see `Kubectl`.
    kube_context: str | None = None
    python_interpreter: str = "python3"
    mldiagnostics: bool = False
    staging_bucket: str | None = None
    shm_size_gb: int = DEFAULT_SHM_SIZE_GB
    job_name_prefix: str | None = None
    staging_service_account: str | None = None
    staging_service_account_key: str | None = None
    colocate_with: str | None = None
    colocate_requests: dict[str, str] = dataclasses.field(default_factory=dict)
    # The pod's Kubernetes service account. If set, the pod stages files with
    # its own Workload Identity token, so nothing is signed.
    pod_service_account: str | None = None
    hf_cache_pvc: str | None = None
    cpu_requests: dict[str, str] = dataclasses.field(default_factory=dict)
    compile_cache_max_age_days: int = COMPILE_CACHE_MAX_AGE_DAYS
    # A gs:// folder torch_tpu keeps its compiled programs in, shared by
    # every Job (see manifest._tier3_cache_command).
    compile_cache_url: str | None = None

    def __post_init__(self) -> None:
        # Blank means `default`. Never derived from the queue: Kueue needs
        # the LocalQueue in the Job's namespace, and Kueue's quickstart puts
        # its queue in `default`.
        self.namespace = (self.namespace or "").strip() or DEFAULT_NAMESPACE
        self.tpu_hosts = max(1, self.tpu_hosts)
        if self.priority is not None and self.priority not in VALID_PRIORITIES:
            raise base.ComputeConfigError(
                f"Backend 'gke_job' priority must be one of "
                f"{sorted(VALID_PRIORITIES)}, got {self.priority!r}.",
                error_code="INVALID_FIELD",
            )
        self.pod_service_account = (self.pod_service_account
                                    or "").strip() or None
        self.colocate_requests = self.colocate_requests or {}
        self.cpu_requests = self.cpu_requests or {}

    def require_dispatch(self) -> None:
        """Raise MISSING_FIELD unless the settings a Job needs are set."""
        missing = [
            field for field, value in (
                ("image", self.image),
                ("tpu_accelerator", self.tpu_accelerator),
                ("tpu_topology", self.tpu_topology),
            ) if not value
        ]
        if missing:
            raise base.ComputeConfigError(
                "Backend 'gke_job' needs " + ", ".join(f"'{f}'"
                                                       for f in missing) +
                " to dispatch a Job. Set them in the compute profile"
                " (`config set`) or pass " +
                " ".join(f"--{f.replace('_', '-')}" for f in missing) + ".",
                error_code="MISSING_FIELD",
            )


def config_from_params(params: dict[str, Any]) -> GkeJobConfig:
    """Build a `GkeJobConfig` from profile params.

    Dispatch-only fields are checked later by `require_dispatch`, so
    exec-into-pod callers and `inspect` work without them.
    """
    # None means derive from the topology. An explicit 0 must still be
    # rejected, so don't use `or`. Either way `tpu_chips` is per pod: on a
    # slice larger than one node, the topology's chips over that many nodes.
    total_chips = chips_for_topology(params.get("tpu_topology", ""))
    chips = params.get("tpu_chips")
    if chips is None:
        hosts, chips = hosts_for(
            params.get("tpu_accelerator") or None, total_chips)
    else:
        chips = positive_int("tpu_chips", chips)
        hosts = max(1, total_chips // chips) if total_chips % chips == 0 else 1

    declared_duration = params.get("declared_duration_minutes")
    if declared_duration is not None and declared_duration != "":
        try:
            declared_duration = int(declared_duration)
        except (TypeError, ValueError) as exc:
            raise base.ComputeConfigError(
                f"Backend 'gke_job' needs an integer "
                f"'declared_duration_minutes', got {declared_duration!r}.",
                error_code="INVALID_FIELD",
            ) from exc
        if declared_duration == 0:
            declared_duration = None
        elif declared_duration < 0:
            raise base.ComputeConfigError(
                f"Backend 'gke_job' needs 'declared_duration_minutes' >= 1, "
                f"got {declared_duration}.",
                error_code="INVALID_FIELD",
            )
    else:
        declared_duration = None

    priority = params.get("priority") or None
    if priority is not None and priority not in VALID_PRIORITIES:
        raise base.ComputeConfigError(
            f"Backend 'gke_job' priority must be one of "
            f"{sorted(VALID_PRIORITIES)}, got {priority!r}.",
            error_code="INVALID_FIELD",
        )

    queue = params.get("queue") or None

    # Blank takes the default. 0 is rejected: a pod without /dev/shm is the
    # failure this setting prevents.
    shm_size_gb = params.get("shm_size_gb")
    if shm_size_gb is None or shm_size_gb == "":
        shm_size_gb = DEFAULT_SHM_SIZE_GB
    shm_size_gb = positive_int("shm_size_gb", shm_size_gb)

    raw_prefix = (params.get("job_name_prefix") or "").strip()
    job_name_prefix = k8s_name(raw_prefix) or None
    if raw_prefix and not (job_name_prefix and len(job_name_prefix)
                           <= MAX_JOB_NAME_PREFIX_LEN):
        raise base.ComputeConfigError(
            f"Backend 'gke_job' 'job_name_prefix' needs 1 to "
            f"{MAX_JOB_NAME_PREFIX_LEN} letters, digits or '-', got "
            f"{raw_prefix!r}.",
            error_code="INVALID_FIELD",
        )

    colocate_with = params.get("colocate_with") or None
    colocate_requests = parse_requests(params.get("colocate_requests"))
    # Kueue admits nothing that requests nothing, and the right size depends
    # on the client, so there is no default.
    if colocate_with and not colocate_requests:
        raise base.ComputeConfigError(
            "Backend 'gke_job' needs 'colocate_requests' "
            "(--colocate-requests NAME=QTY,...) with 'colocate_with'.",
            error_code="MISSING_FIELD",
        )

    max_age = params.get("compile_cache_max_age_days")
    if max_age is None or max_age == "":
        max_age = COMPILE_CACHE_MAX_AGE_DAYS
    max_age = positive_int("compile_cache_max_age_days", max_age, 0)
    compile_cache_url = (params.get("compile_cache_url")
                         or "").strip().rstrip("/")
    # A folder, not a bucket: serve_vllm.sh appends -patch-<key> to the root,
    # which on a bare gs://bucket would name a different bucket.
    if compile_cache_url and not re.fullmatch(r"gs://[^/]+/.+",
                                              compile_cache_url):
        raise base.ComputeConfigError(
            "Backend 'gke_job' 'compile_cache_url' must be a folder in a "
            f"bucket, gs://<bucket>/<folder>, got {compile_cache_url!r}.",
            error_code="INVALID_FIELD",
        )

    cpu_requests = parse_requests(params.get("cpu_requests"), "cpu_requests")
    if cpu_requests and colocate_with:
        # A colocated Job runs chipless on another Job's TPU node. A CPU Job
        # runs on no TPU node at all.
        raise base.ComputeConfigError(
            "Backend 'gke_job' takes 'cpu_requests' or 'colocate_with', "
            "not both.",
            error_code="INVALID_FIELD",
        )

    # Blank means no shared cache: each pod downloads into its own disk.
    hf_cache_pvc = (params.get("hf_cache_pvc") or "").strip() or None
    if hf_cache_pvc and not PVC_NAME.fullmatch(hf_cache_pvc):
        raise base.ComputeConfigError(
            f"Backend 'gke_job' 'hf_cache_pvc' must be the name of a "
            f"PersistentVolumeClaim (lowercase letters, digits, '-' or "
            f"'.'), got {hf_cache_pvc!r}.",
            error_code="INVALID_FIELD",
        )

    return GkeJobConfig(
        namespace=params.get("namespace"),
        image=params.get("image") or None,
        tpu_accelerator=params.get("tpu_accelerator") or None,
        tpu_topology=params.get("tpu_topology") or None,
        tpu_chips=chips,
        tpu_hosts=hosts,
        chips_explicit=params.get("tpu_chips") is not None,
        queue=queue,
        priority=priority,
        declared_duration_minutes=declared_duration,
        kubeconfig=params.get("kubeconfig") or None,
        kube_context=(params.get("kube_context") or "").strip() or None,
        python_interpreter=params.get("python_interpreter") or "python3",
        mldiagnostics=bool(params.get("mldiagnostics")),
        staging_bucket=params.get("staging_bucket") or None,
        shm_size_gb=shm_size_gb,
        job_name_prefix=job_name_prefix,
        staging_service_account=params.get(gcs_staging.SERVICE_ACCOUNT_PARAM)
        or None,
        staging_service_account_key=params.get(gcs_staging.KEY_FILE_PARAM)
        or None,
        colocate_with=colocate_with,
        colocate_requests=colocate_requests,
        pod_service_account=params.get(gcs_staging.POD_SERVICE_ACCOUNT_PARAM)
        or None,
        hf_cache_pvc=hf_cache_pvc,
        cpu_requests=cpu_requests,
        compile_cache_max_age_days=max_age,
        compile_cache_url=compile_cache_url or None,
    )


def add_cli_flags(parser: Any, add_cli_arg: Callable[..., None]) -> None:
    """Declare `GkeJobConfig` flags via `ComputeDriver.add_cli_arg`."""
    add_cli_arg(
        parser,
        "--compile-cache-url",
        help="gs:// folder torch_tpu keeps its compiled programs in, shared "
        "by every Job; pods read and write it as their own identity",
    )
    add_cli_arg(
        parser,
        "--namespace",
        help="Kubernetes namespace to dispatch into (default: default)",
    )
    add_cli_arg(
        parser,
        "--mldiagnostics",
        action="store_true",
        help="Label the Job for GKE managed ML diagnostics, so "
        "the workload appears in the Console Diagnostics view",
    )
    add_cli_arg(
        parser,
        "--image",
        help="Container image the dispatched Job runs, unmodified",
    )
    add_cli_arg(
        parser,
        "--tpu-accelerator",
        help=("Value of the cloud.google.com/gke-tpu-accelerator node "
              "label, e.g. tpu7x or tpu-v6e-slice"),
    )
    add_cli_arg(
        parser,
        "--tpu-topology",
        help=("Value of the cloud.google.com/gke-tpu-topology node label, "
              "e.g. 2x2x1 or 2x2"),
    )
    add_cli_arg(
        parser,
        "--tpu-chips",
        type=int,
        help=f"Chips to request per pod (default: {DEFAULT_CHIPS})",
    )
    add_cli_arg(
        parser,
        "--queue",
        help="Kueue LocalQueue to admit through; omit on a direct cluster",
    )
    add_cli_arg(
        parser,
        "--priority",
        help="Kueue priority class, one of " +
        ", ".join(sorted(VALID_PRIORITIES)) +
        "; omit to leave the Job unlabelled and take the cluster's default",
    )
    add_cli_arg(
        parser,
        "--declared-duration-minutes",
        type=int,
        help="Declared runtime in minutes for Kueue suspension enforcement",
    )
    add_cli_arg(
        parser,
        "--kubeconfig",
        help="Kubeconfig to use instead of the ambient one",
    )
    add_cli_arg(
        parser,
        "--kube-context",
        help=base.KUBE_CONTEXT_HELP,
    )
    add_cli_arg(
        parser,
        "--staging-bucket",
        help="GCS bucket URI for file staging via signed URLs",
    )
    add_cli_arg(
        parser,
        "--staging-service-account",
        help="Service account that staging acts as, using your own "
        "login (no key file); kubectl keeps your identity",
    )
    add_cli_arg(
        parser,
        "--staging-service-account-key",
        help="Key file of the service account staging uses, when you "
        "cannot use --staging-service-account; kubectl keeps your "
        "identity, unlike $GOOGLE_APPLICATION_CREDENTIALS",
    )
    add_cli_arg(
        parser,
        "--pod-service-account",
        help="Kubernetes service account the Job's pod runs as. On a "
        "cluster with Workload Identity, the pod then moves staged files "
        "with its own token, so nothing is signed",
    )
    add_cli_arg(
        parser,
        "--shm-size-gb",
        type=int,
        help=f"Size in GiB of the memory-backed /dev/shm mounted in the "
        f"pod (default: {DEFAULT_SHM_SIZE_GB})",
    )
    add_cli_arg(
        parser,
        "--job-name-prefix",
        help="Prefix added to the name of every Job this backend "
        "creates, e.g. 'agent'",
    )
    add_cli_arg(
        parser,
        "--hf-cache-pvc",
        help="ReadWriteMany PersistentVolumeClaim, made by a cluster "
        f"admin, mounted at {HF_CACHE_MOUNT} in every pod as "
        f"{HF_CACHE_ENV} (with vLLM's and torch_tpu's compile caches "
        f"under {HF_CACHE_MOUNT}/{COMPILE_CACHE_SUBDIR}/), so model "
        "weights download once and compiled programs are kept across Jobs",
    )
    add_cli_arg(
        parser,
        "--compile-cache-max-age-days",
        type=int,
        help="With --hf-cache-pvc, each pod removes compile-cache image "
        "folders no pod has used in this many days (default: "
        f"{COMPILE_CACHE_MAX_AGE_DAYS}; 0 keeps them)",
    )
    add_cli_arg(
        parser,
        "--colocate-with",
        metavar="JOB_ID",
        help="Run on the node of running Job JOB_ID, without chips (it "
        "holds them). For a client of a server Job, e.g. a benchmark: it "
        "reaches the server at http://JOB_ID:PORT without exec or a "
        "port-forward",
    )
    add_cli_arg(
        parser,
        "--cpu-requests",
        metavar="NAME=QTY,...",
        help="Run on a CPU node, holding no TPU: the Job requests this "
        "CPU and memory instead of chips, e.g. cpu=2,memory=8Gi, and has "
        "no TPU node selector or toleration, so it books no TPU node. "
        "For work that needs the task image but not a chip. The Job gets "
        "the profile's TPU generation in TPU_AGENT_SUITE_CHIPLESS_TPU; a "
        "`run` that exits 75 or never starts is run again on a TPU node",
    )
    add_cli_arg(
        parser,
        "--colocate-requests",
        metavar="NAME=QTY,...",
        help="Required with --colocate-with: the CPU and memory it "
        "requests instead of chips, e.g. cpu=2,memory=8Gi, sized to the "
        "client. Kueue admits no pod that requests nothing",
    )
