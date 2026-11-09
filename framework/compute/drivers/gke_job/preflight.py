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
"""Preflight checks for `gke_job`.

Checks the cluster, TPU node pools, image access, namespace RBAC, and the
Kueue, staging and HF cache PVC setup the profile asks for.
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

from .. import auth_utils, base, gcs_staging
from ..base import ACCELERATOR_LABEL
from .gke_common import (DEFAULT_NAMESPACE, NODEPOOL_LABEL,
                         WORKLOAD_IDENTITY_LABEL)
from .manifest import HF_CACHE_ACCESS_MODE, node_selector

if TYPE_CHECKING:
    from ..kubectl import Kubectl
    from .profile_settings import GkeJobConfig

# Accelerator labels GKE puts on TPU nodes. Used to flag a `tpu_accelerator`
# typo before a Job sits Pending. Chip specs come from `probe.spec_for`.
KNOWN_ACCELERATOR_LABELS = (
    "tpu-v4-podslice",
    "tpu-v5-lite-podslice",
    "tpu-v5p-slice",
    "tpu-v6e-slice",
    "tpu7x",
)


def _hf_cache_pvc_check(namespaced: list[str], kubectl: list[str], claim: str,
                        namespace: str) -> base.PreflightCheck:
    """Whether the shared cache claim exists, is ReadWriteMany, and binds."""
    name = "hf_cache_pvc"
    create = (f"Ask a cluster admin to create a {HF_CACHE_ACCESS_MODE} "
              f"PersistentVolumeClaim '{claim}' in namespace '{namespace}' "
              "(e.g. from a Filestore storage class), or unset hf_cache_pvc")
    proc = base.probe_run(namespaced + ["get", "pvc", claim, "-o", "json"])
    if proc.returncode != 0:
        error = base.probe_error(proc)
        if "forbidden" in error.lower():
            return base.PreflightCheck(
                name, "warn", f"cannot read PVC '{claim}': {error}",
                "Ask a cluster admin for `get` on persistentvolumeclaims, or "
                "check the claim yourself")
        # A pod naming a missing claim never starts.
        return base.PreflightCheck(
            name, "fail", f"PVC '{claim}' in namespace '{namespace}': "
            f"{error}", create)
    try:
        pvc = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return base.PreflightCheck(name, "warn",
                                   f"could not parse PVC '{claim}'")
    spec, status = pvc.get("spec") or {}, pvc.get("status") or {}
    modes = status.get("accessModes") or spec.get("accessModes") or []
    if HF_CACHE_ACCESS_MODE not in modes:
        return base.PreflightCheck(
            name, "fail",
            f"PVC '{claim}' is {', '.join(modes) or 'without an access mode'}"
            f", not {HF_CACHE_ACCESS_MODE}: Jobs on different nodes cannot "
            "all mount it, so a second one stays Pending", create)
    phase = status.get("phase") or "Pending"
    capacity = (status.get("capacity") or {}).get("storage")
    if phase == "Bound":
        return base.PreflightCheck(
            name, "ok", f"PVC '{claim}' is Bound, {HF_CACHE_ACCESS_MODE}" +
            (f", {capacity}" if capacity else ""))
    storage_class = spec.get("storageClassName")
    if phase == "Pending" and storage_class:
        # A WaitForFirstConsumer class binds on first use, so Pending is
        # normal for an unused claim.
        sc = base.probe_run(kubectl + [
            "get", "storageclass", storage_class, "-o",
            "jsonpath={.volumeBindingMode}"
        ])
        if sc.returncode != 0:
            return base.PreflightCheck(
                name, "warn",
                f"PVC '{claim}' is Pending, and its storage class "
                f"'{storage_class}' could not be read: {base.probe_error(sc)}",
                "Fine if the class binds on first use; otherwise ask a "
                "cluster admin why the claim is not Bound")
        if sc.stdout.strip() == "WaitForFirstConsumer":
            return base.PreflightCheck(
                name, "warn", f"PVC '{claim}' is Pending until the first Job "
                f"mounts it (storage class '{storage_class}' binds on first "
                "use)")
    return base.PreflightCheck(
        name, "fail", f"PVC '{claim}' is {phase}, not Bound: a Job that "
        "mounts it stays Pending",
        "Ask a cluster admin why the claim is not Bound "
        f"(kubectl describe pvc {claim} -n {namespace}), or unset "
        "hf_cache_pvc")


def _kube_get_check(kubectl: list[str], name: str, what: str, argv: list[str],
                    fix: str) -> base.PreflightCheck:
    """Whether `kubectl get <argv>` finds `what`."""
    proc = base.probe_run(kubectl + ["get"] + argv + ["-o", "name"])
    if proc.returncode == 0:
        return base.PreflightCheck(name, "ok", f"{what} exists")
    error = base.probe_error(proc)
    status = "warn" if "forbidden" in error.lower() else "fail"
    return base.PreflightCheck(name, status, f"{what}: {error}", fix)


# Registries gcloud can describe: Artifact Registry and AR-served gcr.io.
_GCLOUD_REGISTRY_SUFFIXES = ("-docker.pkg.dev", "gcr.io")


def _image_check(image: str) -> base.PreflightCheck:
    """Whether `image` resolves in its registry, as this machine's account.

    Quickly catches a missing tag, digest or repository. Nodes pull as their
    own service account, which `_node_pull_check` and `--smoke-test` cover.
    """
    registry = image.split("/", 1)[0]
    if not registry.endswith(_GCLOUD_REGISTRY_SUFFIXES):
        return base.PreflightCheck(
            "image", "warn", f"{image}: not in Artifact Registry, so it "
            "cannot be checked without pulling it")
    if not shutil.which("gcloud"):
        return base.PreflightCheck(
            "image", "warn", f"{image}: gcloud is not on PATH to check it")
    proc = base.probe_run([
        "gcloud", "artifacts", "docker", "images", "describe", image,
        "--format=value(image_summary.digest)"
    ],
                          timeout=30)
    if proc.returncode == 0:
        return base.PreflightCheck("image", "ok",
                                   f"{image} ({proc.stdout.strip()})")
    error = base.probe_error(proc)
    # The whole output, not the one line: which line says so varies.
    if re.search(r"not found|NOT_FOUND|manifest unknown",
                 f"{proc.stderr}\n{proc.stdout}", re.IGNORECASE):
        return base.PreflightCheck(
            "image", "fail", f"{image}: {error}",
            "Check the image's tag or digest against `gcloud artifacts docker "
            f"images list {image.split('@')[0].rsplit(':', 1)[0]} "
            "--include-tags`")
    # No read access, or gcloud failed. The nodes may still pull it.
    return base.PreflightCheck("image", "warn",
                               f"{image}: could not check it: {error}")


_AR_READER = "roles/artifactregistry.reader"

# An Artifact Registry image: captures location, project and repository.
_AR_IMAGE = re.compile(r"([a-z0-9-]+)-docker\.pkg\.dev/([^/]+)/([^/]+)/.+")


def _gcloud_json(argv: list[str]) -> Any:
    """`gcloud <argv> --format=json`, parsed; None when it fails."""
    proc = base.probe_run(["gcloud"] + argv + ["--format=json"], timeout=30)
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout or "null")
    except ValueError:
        return None


def _node_service_accounts(target: dict[str, str], pools: set[str],
                           topology: str | None) -> list[str] | None:
    """Service accounts the cluster's nodes run as, or None if unknown.

    Limited to `pools`, else to `topology`'s pools: a TPU pool scaled to zero
    has no nodes to name it.
    """
    listed = _gcloud_json([
        "container", "node-pools", "list", "--cluster", target["cluster"],
        "--location", target["location"], "--project", target["project"]
    ])
    if not isinstance(listed, list):
        return None
    chosen = ([pool for pool in listed if pool.get("name") in pools] or [
        pool for pool in listed if topology and
        (pool.get("placementPolicy") or {}).get("tpuTopology") == topology
    ])
    if not chosen:
        return None
    accounts = {(pool.get("config") or {}).get("serviceAccount") or "default"
                for pool in chosen}
    if "default" in accounts:
        # GKE's "default" is the project's Compute Engine default account.
        described = _gcloud_json(["projects", "describe", target["project"]
                                  ]) or {}
        number = described.get("projectNumber")
        if not number:
            return None
        accounts.discard("default")
        accounts.add(f"{number}-compute@developer.gserviceaccount.com")
    return sorted(accounts)


def _list_node_pools(kubectl: list[str],
                     selector: str) -> subprocess.CompletedProcess:
    """`kubectl get nodes -l selector`: one line per node, naming its pool."""
    return base.probe_run(kubectl + [
        "get", "nodes", "-l", selector, "-o",
        "jsonpath={range .items[*]}{.metadata.labels." +
        NODEPOOL_LABEL.replace(".", r"\.") + '}{"\\n"}{end}'
    ])


def _node_pull_check(image: str, context: str, pools: set[str],
                     topology: str | None) -> base.PreflightCheck | None:
    """Whether the nodes can likely pull `image`, or None if not checkable.

    Only images in another project's Artifact Registry are checked.
    """
    match = _AR_IMAGE.fullmatch(image)
    target = base.gke_target(context, prefer_context=True)
    if not match or any(v.startswith("<") for v in target.values()):
        return None
    location, project, repository = match.groups()
    if project == target["project"]:
        return None
    accounts = _node_service_accounts(target, pools, topology)
    who = (", ".join(accounts)
           if accounts else "the node pools' service account")
    grant = (f"gcloud artifacts repositories add-iam-policy-binding "
             f"{repository} --location {location} --project {project} "
             f"--role {_AR_READER} --member "
             f"serviceAccount:{accounts[0] if accounts else '<node SA>'}")
    policy = _gcloud_json([
        "artifacts", "repositories", "get-iam-policy", repository,
        "--location", location, "--project", project
    ])
    if isinstance(policy, dict):
        readers = {
            member
            for binding in policy.get("bindings") or []
            if binding.get("role") == _AR_READER
            for member in binding.get("members") or []
        }
        nodes = {f"serviceAccount:{account}" for account in accounts or []}
        if (nodes and nodes
                <= readers) or readers & {"allUsers", "allAuthenticatedUsers"}:
            return base.PreflightCheck(
                "image:node_pull", "ok",
                f"repository {project}/{repository} grants the nodes read")
        seen = "its policy does not name them"
    else:
        seen = "this account cannot read its policy to check"
    return base.PreflightCheck(
        "image:node_pull", "warn",
        f"the nodes pull {image.split('@')[0]} as {who}, from project "
        f"{project} rather than the cluster's ({target['project']}), and "
        f"{seen}; without read access the Job fails with ErrImagePull 403",
        f"Have an admin of {project} grant it {_AR_READER} "
        f"({grant}), or copy the image into a registry in {target['project']}")


def _kube_can_i_check(kubectl: list[str], verb: str,
                      resource: str) -> base.PreflightCheck:
    """Whether this account may `verb` `resource` where `kubectl` points."""
    proc = base.probe_run(kubectl + ["auth", "can-i", verb, resource])
    answer = proc.stdout.strip()
    name = f"rbac:{verb} {resource}"
    if answer == "yes":
        return base.PreflightCheck(name, "ok", "allowed")
    if answer.startswith("no"):
        return base.PreflightCheck(
            name, "fail", f"this account may not {verb} {resource}",
            f"Ask a cluster admin for a Role that grants {verb} on "
            f"{resource} in this namespace")
    return base.PreflightCheck(
        name, "warn", f"could not ask the cluster: {base.probe_error(proc)}")


def target_checks(kube: Kubectl, config: GkeJobConfig,
                  checks: list[base.PreflightCheck]) -> bool:
    """Check the cluster, dispatch settings, image, queue and bucket.

    Returns whether the cluster is reachable.
    """
    kubectl = kube.cmd([])
    # Profile settings are checked even when the cluster is unreachable.
    reachable = auth_utils.cluster_check(kube, kubectl, checks,
                                         config.kubeconfig, config.namespace)
    try:
        config.require_dispatch()
    except base.ComputeConfigError as exc:
        base.fail(checks, "dispatch_settings", str(exc))
    image = _image_check(config.image) if config.image else None
    if image:
        checks.append(image)
    if (config.tpu_accelerator
            and config.tpu_accelerator not in KNOWN_ACCELERATOR_LABELS):
        # Only a warning: a generation newer than this table is right.
        checks.append(
            base.PreflightCheck(
                "tpu_accelerator", "warn",
                f"'{config.tpu_accelerator}' is not a known "
                f"{ACCELERATOR_LABEL} value; a Job that asks for a "
                "label no node has stays Pending",
                "Use one of " + ", ".join(sorted(KNOWN_ACCELERATOR_LABELS)) +
                ", or ignore this if it is a newer generation"))
    if reachable:
        pools = cluster_checks(config, kubectl, checks)
        if image and image.status == "ok":
            pull = _node_pull_check(config.image,
                                    base.current_context(kubectl), pools,
                                    config.tpu_topology)
            if pull:
                checks.append(pull)
    if config.staging_bucket:
        checks.extend(staging_checks(config))
    return reachable


def staging_checks(config: GkeJobConfig) -> list[base.PreflightCheck]:
    """The staging checks, made with the credential staging will use."""
    bucket = gcs_staging.clean_bucket_name(config.staging_bucket or "")
    try:
        credentials = gcs_staging.staging_credentials(
            config.staging_service_account, config.staging_service_account_key)
    except base.ComputeConfigError as exc:
        return [
            base.PreflightCheck(
                "staging:credentials", "fail", str(exc),
                f"Fix or unset '{gcs_staging.SERVICE_ACCOUNT_PARAM}' / "
                f"'{gcs_staging.KEY_FILE_PARAM}'")
        ]
    return gcs_staging.preflight_checks(
        bucket,
        credentials=credentials,
        pod_service_account=config.pod_service_account)


def cluster_checks(config: GkeJobConfig, kubectl: list[str],
                   checks: list[base.PreflightCheck]) -> set[str]:
    """The namespace, the right to create Jobs, and Kueue's objects.

    Returns the node pools of the TPU nodes a Job would land on.
    """
    namespaced = kubectl + ["-n", config.namespace]
    if config.namespace != DEFAULT_NAMESPACE:
        checks.append(
            _kube_get_check(
                kubectl, "namespace", f"namespace '{config.namespace}'",
                ["namespace", config.namespace],
                f"kubectl create namespace {config.namespace}, or set "
                "namespace to an existing one"))
    pools: set[str] = set()
    if config.tpu_accelerator and config.tpu_topology:
        nodes, pools = _nodes_check(config, kubectl)
        checks.append(nodes)
    # A run creates the ConfigMap carrying the script, then the Job.
    for resource in ("configmaps", "jobs"):
        checks.append(_kube_can_i_check(namespaced, "create", resource))
    if config.staging_bucket and not config.pod_service_account:
        # The pod token is stored in a Secret. Only warn: without it the
        # Job still runs, it just can't reach GCS.
        secrets = _kube_can_i_check(namespaced, "create", "secrets")
        if secrets.status == "fail":
            secrets = dataclasses.replace(
                secrets,
                status="warn",
                detail=f"{secrets.detail}, so pods won't get a token for "
                "the staging bucket")
        checks.append(secrets)
    if config.pod_service_account:
        # A Job naming a missing account is accepted but never gets a pod.
        checks.append(
            _kube_get_check(
                namespaced, "pod_service_account",
                f"service account '{config.pod_service_account}' in "
                f"namespace '{config.namespace}'",
                ["serviceaccount", config.pod_service_account],
                f"kubectl create serviceaccount "
                f"{config.pod_service_account} -n {config.namespace}, then "
                "grant roles/storage.objectAdmin on the staging bucket "
                "either to its Workload Identity principal directly, or "
                "to a Google service account it names in an "
                "iam.gke.io/gcp-service-account annotation (setup step "
                "4 in docs/install.md)"))
    if config.hf_cache_pvc:
        checks.append(
            _hf_cache_pvc_check(namespaced, kubectl, config.hf_cache_pvc,
                                config.namespace))
    if not config.queue:
        return pools
    # One get shows both that Kueue is installed and that the queue exists,
    # without the cluster-wide read that listing CRDs needs.
    checks.append(
        _kube_get_check(
            namespaced, "kueue:queue",
            f"LocalQueue '{config.queue}' in namespace '{config.namespace}'",
            ["localqueue", config.queue],
            "Install Kueue on the cluster and create the LocalQueue "
            "(https://kueue.sigs.k8s.io/docs/installation/), or unset "
            "queue to schedule Jobs directly"))
    # The priority label is applied only on a queued Job.
    if config.priority:
        checks.append(
            _kube_get_check(
                kubectl, "kueue:priority",
                f"WorkloadPriorityClass '{config.priority}'",
                ["workloadpriorityclass", config.priority],
                "Ask a cluster admin to create the "
                "WorkloadPriorityClass, or unset priority"))
    return pools


def _nodes_check(config: GkeJobConfig,
                 kubectl: list[str]) -> tuple[base.PreflightCheck, set[str]]:
    """Whether any node has the Job's selector labels, and their node pools.

    Never fails: a pool that scales from zero or is auto-provisioned has no
    nodes until a Job asks. TPU nodes missing only the Workload Identity
    label get their own warning.
    """
    selector = node_selector(config)
    proc = _list_node_pools(kubectl, selector)
    if proc.returncode != 0:
        return base.PreflightCheck(
            "tpu_nodes", "warn",
            f"could not list nodes: {base.probe_error(proc)}"), set()
    nodes = proc.stdout.splitlines()
    if nodes:
        return base.PreflightCheck(
            "tpu_nodes", "ok",
            f"{len(nodes)} node(s) match {selector}"), set(nodes) - {""}
    if config.pod_service_account:
        tpu_selector = selector.removesuffix(
            f",{WORKLOAD_IDENTITY_LABEL}=true")
        tpu = _list_node_pools(kubectl, tpu_selector)
        pools = sorted(set(tpu.stdout.splitlines()) - {""})
        if tpu.returncode == 0 and pools:
            return base.PreflightCheck(
                "tpu_nodes", "warn",
                f"TPU node pool(s) {', '.join(pools)} match "
                f"{tpu_selector} but don't run the GKE metadata server, "
                f"which {gcs_staging.POD_SERVICE_ACCOUNT_PARAM} needs: "
                "Jobs won't schedule on them",
                "gcloud container node-pools update <pool> "
                "--workload-metadata=GKE_METADATA, or unset "
                f"{gcs_staging.POD_SERVICE_ACCOUNT_PARAM}. Fine if "
                "auto-provisioning will add a pool with Workload "
                "Identity"), set()
    return base.PreflightCheck(
        "tpu_nodes", "warn", f"no node matches {selector} right now",
        "Fine for a node pool that scales from zero or is auto-"
        "provisioned; otherwise check tpu_accelerator and tpu_topology "
        "against the TPU node pool's labels"), set()
