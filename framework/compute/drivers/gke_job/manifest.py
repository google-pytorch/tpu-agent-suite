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
"""The Kubernetes Job a `gke_job` dispatch creates.

Builds the manifest and manages pod caches and the GCS token Secret.
"""

from __future__ import annotations

import json
import shlex
import sys
import time
from typing import TYPE_CHECKING, Any

from framework import utils

from ... import env_check
from .. import base, gcs_staging, jobs, probe
from ..base import ACCELERATOR_LABEL, TOPOLOGY_LABEL, TPU_RESOURCE
from ..kubectl import KUBECTL_READ_TIMEOUT_SEC
from .gke_common import (CHIPLESS_TPU_ENV, COMPLETION_INDEX_KEY, DEFAULT_CHIPS,
                         MEGASCALE_ENV, MLDIAGNOSTICS_LABEL, NODEPOOL_LABEL,
                         PRIORITY_LABEL, QUEUE_LABEL, SCRIPT_MOUNT,
                         SERVE_BACKOFF_LIMIT, SHM_LIMIT_ENV, SHM_MOUNT,
                         WORKLOAD_IDENTITY_LABEL, StagingPlan, k8s_name)

if TYPE_CHECKING:
    from ..kubectl import Kubectl
    from .profile_settings import GkeJobConfig

# Without a pod_service_account, pods mount a GCS token from this Secret. It
# can write the Job's own folder and only read the checkpoints folder.
TOKEN_SECRET_SUFFIX = "-gcs-token"
TOKEN_MOUNT = "/var/run/gcs-token"
# Under the folder staging_bucket names, or at the bucket root.
CHECKPOINTS_PREFIX = "checkpoints/"
# Tokens last about an hour. Renew this early so kubelet has time to sync
# the new Secret into the pod.
TOKEN_RENEW_MARGIN_SEC = 20 * 60
TOKEN_RENEW_RETRY_SEC = 5 * 60

# A shared PVC, so models download once. Only HF_HUB_CACHE uses it, since
# HF_HOME would expose the pod's Hugging Face token to other Jobs.
HF_CACHE_MOUNT = "/hf-cache"
HF_CACHE_ENV = "HF_HUB_CACHE"
HF_CACHE_VOLUME = "hf-cache"
# Jobs on different nodes mount it at once. With ReadWriteOnce the second
# stays Pending on a Multi-Attach error.
HF_CACHE_ACCESS_MODE = "ReadWriteMany"
# vLLM and torch_tpu compile caches, per TPU shape and image. torch_tpu has
# two names for its root (it reads the INTERNAL one); if the image sets
# either, both are left alone. Defaults only: values the image or recipe
# sets win. (Names, subfolder.)
COMPILE_CACHE_SUBDIR = "compile-cache"
COMPILE_CACHE_ENV = (
    (("VLLM_CACHE_ROOT", ), "vllm"),
    (("TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT",
      "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"), "torch_tpu"),
)
# Old folders pile up on the weights' volume. A pod stamps its own folder,
# since reads don't update mtime, and removes others idle this long.
COMPILE_CACHE_STAMP = ".last_used"
COMPILE_CACHE_MAX_AGE_DAYS = 14
# Above this, compile caches go to /dev/shm to leave room for weights.
COMPILE_CACHE_MAX_USED_PCT = 90
# Without the shared cache, caches go on /dev/shm. A TPU node's disk can be
# smaller than one model, and filling it evicts the pod (exit 137).
# Defaults only: values the image or recipe sets win.
# Name -> (default, the variables that move it).
POD_CACHE_DIR = f"{SHM_MOUNT}/cache"
POD_CACHE_ENV = {
    HF_CACHE_ENV:
    (f"{POD_CACHE_DIR}/huggingface/hub", ("HF_HOME", "HUGGINGFACE_HUB_CACHE")),
    "VLLM_CACHE_ROOT": (f"{POD_CACHE_DIR}/vllm", ("XDG_CACHE_HOME", )),
    "XDG_CACHE_HOME": (POD_CACHE_DIR, ()),
    "TORCHINDUCTOR_CACHE_DIR": (f"{POD_CACHE_DIR}/torchinductor", ()),
    "TMPDIR": (SHM_MOUNT, ()),
}
# Marks the hub cache as pod-local, so scripts may prune it. Without it,
# HF_HUB_CACHE may be shared and is left alone.
POD_HF_CACHE_ENV = "TPU_AGENT_SUITE_POD_HF_CACHE"


def _pod_cache_command(command: list[str]) -> list[str]:
    """Wrap `command` to set any missing POD_CACHE_ENV defaults.

    Set in a shell, not the pod spec, so the image's ENV still wins.
    """
    lines = []
    for name, (default, movers) in POD_CACHE_ENV.items():
        unset = "".join(f"${{{var}+x}}" for var in (name, *movers))
        export = f"export {name}={shlex.quote(default)}"
        if name == HF_CACHE_ENV:
            export += f" {POD_HF_CACHE_ENV}={shlex.quote(default)}"
        lines.append(f'if [ -z "{unset}" ]; then {export}; fi')
    return ["sh", "-c", "\n".join(lines + ['exec "$@"']), "sh", *command]


def _compile_cache_exports(root: str) -> list[str]:
    """Shell lines pointing unset COMPILE_CACHE_ENV vars at `root`."""
    lines = []
    for names, subdir in COMPILE_CACHE_ENV:
        unset = "".join(f"${{{name}+x}}" for name in names)
        value = shlex.quote(f"{root}/{subdir}")
        exports = " ".join(f"{name}={value}" for name in names)
        lines.append(f'if [ -z "{unset}" ]; then export {exports}; fi')
    return lines


def _compile_cache_command(command: list[str],
                           root: str,
                           max_age_days: int = COMPILE_CACHE_MAX_AGE_DAYS,
                           mount: str = HF_CACHE_MOUNT) -> list[str]:
    """Wrap `command` to point unset COMPILE_CACHE_ENV vars at `root`.

    Also stamps `root` and prunes stale image folders unless `max_age_days`
    is 0. If `mount` is nearly full, caches stay in the pod.
    """
    q = shlex.quote
    lines = [
        f"used=$(df -P {q(mount)} 2>/dev/null | awk 'NR==2 {{print $5+0}}')",
        f'if [ "${{used:-0}}" -lt {COMPILE_CACHE_MAX_USED_PCT} ]; then',
        *_compile_cache_exports(root),
        f"mkdir -p {q(root)} 2>/dev/null && "
        f"touch {q(f'{root}/{COMPILE_CACHE_STAMP}')} 2>/dev/null",
        "else",
        f'echo "gke_job: {mount} is ${{used}}% full; compile caches stay '
        'in the pod" >&2',
        *_compile_cache_exports(POD_CACHE_DIR),
        "fi",
    ]
    if max_age_days > 0:
        # Keep folders that find can't read. Only the leader of a multi-host
        # Job prunes, since its pods share the volume.
        lines.append(
            f"{gcs_staging.NOT_LEADER} || "
            f"(for d in {q(mount)}/{COMPILE_CACHE_SUBDIR}/*/*/; do\n"
            f'd="${{d%/}}"; [ -d "$d" ] && [ "$d" != {q(root)} ] || continue\n'
            f'new=$(find "$d" -mtime -{max_age_days} 2>/dev/null) || continue\n'
            '[ -z "$new" ] || continue\n'
            f'echo "gke_job: removing compile cache $d, unused for '
            f'{max_age_days}+ days" >&2\n'
            'rm -rf "$d" || echo "gke_job: could not remove $d" >&2\n'
            "done) </dev/null &")
    return ["sh", "-c", "\n".join(lines + ['exec "$@"']), "sh", *command]


def compile_cache_key(image: str) -> str:
    """The compile-cache folder for `image`.

    A digest is used as is. A tag can move, so it also gets a hash.
    """
    ref = image.strip()
    _, _, digest = ref.partition("@")
    if digest:
        algorithm, _, value = digest.partition(":")
        return k8s_name(f"{algorithm}-{value[:32]}")
    name = k8s_name(ref.rsplit("/", 1)[-1])[:48].strip("-")
    return f"{name}-{utils.digest_text(ref)[:12]}"


def chips_for_topology(topology: str) -> int:
    """Chips in the whole slice of `topology`, e.g. `2x4` -> 8, `2x2x2` -> 8.

    The slice may span several nodes (see `hosts_for`). Returns
    `DEFAULT_CHIPS` if unparseable. Scheduling will report it better.
    """
    return probe.product(topology) or DEFAULT_CHIPS


def hosts_for(accelerator: str | None, total_chips: int) -> tuple[int, int]:
    """(nodes, chips per node) for a slice of `total_chips`.

    One node while the slice fits on one (up to 8 chips on v5e and v6e, 4 on
    3D generations), else `DEFAULT_CHIPS` per node. An unknown accelerator,
    or a count that doesn't divide, stays on one node and lets the scheduler
    say why it can't place it.
    """
    gen = probe.generation_for_name(accelerator or "")
    single = 8 if gen in ("v5e", "v6e") else DEFAULT_CHIPS
    if not gen or total_chips <= single or total_chips % DEFAULT_CHIPS:
        return 1, total_chips
    return total_chips // DEFAULT_CHIPS, DEFAULT_CHIPS


def _chip_spec(accelerator: str) -> tuple[dict[str, Any] | None, str]:
    """HBM and devices per chip for a node label, or why they are unknown.

    Runs on the controller, where tpu_info may be missing. Unknown specs are
    reported, not guessed, since a wrong guess sizes a model wrong.
    """
    try:
        spec = probe.spec_for(probe.generation_for_name(accelerator))
    except probe.MissingPackageError as exc:
        return None, ("hbm_gb and devices not reported: " +
                      env_check.ask_to_install([exc.package]))
    if not spec["hbm_gb"]:
        return None, (f"unknown accelerator '{accelerator}'; hbm_gb and "
                      "devices not reported")
    return spec, ""


def job_sizing(config: GkeJobConfig, accelerator: str,
               details: dict[str, Any]) -> float:
    """Add one Job's share to `details`. Return HBM per chip, or 0.0."""
    # One Job gets `tpu_chips` on each of its `tpu_hosts` nodes, not the
    # whole pool.
    job_chips = config.tpu_chips * config.tpu_hosts
    details["job_chips"] = job_chips
    if config.tpu_hosts > 1:
        details["job_hosts"] = config.tpu_hosts
        details["chips_per_host"] = config.tpu_chips
    spec, note = _chip_spec(accelerator)
    if not spec:
        details["hbm_gb_note"] = note
        return 0.0
    devices_per_chip = spec["devices_per_chip"]
    details["devices_per_chip"] = devices_per_chip
    details["job_devices"] = job_chips * devices_per_chip
    details["hbm_gb_per_device"] = spec["hbm_gb"] / devices_per_chip
    return spec["hbm_gb"]


def node_selector(config: GkeJobConfig) -> str:
    """The label selector for the nodes a Job of `config` lands on."""
    selector = f"{ACCELERATOR_LABEL}"
    if config.tpu_accelerator:
        selector = f"{ACCELERATOR_LABEL}={config.tpu_accelerator}"
    if config.tpu_topology:
        selector += f",{TOPOLOGY_LABEL}={config.tpu_topology}"
    if config.pod_service_account:
        selector += f",{WORKLOAD_IDENTITY_LABEL}=true"
    return selector


def fit_to_nodes(kube: Kubectl,
                 config: GkeJobConfig,
                 nodes: list[dict[str, Any]] | None = None) -> None:
    """Split the slice in `config` by the chips its nodes really have.

    Only a split guessed from the topology changes, and only to one with the
    same total: a v6e `2x4` guessed as one 8-chip node becomes two pods of 4
    on `ct6e-standard-4t` nodes. Without `nodes` (at submit), reads the pool
    best-effort, and only for v5e and v6e, where the topology is ambiguous.
    """
    if config.chips_explicit:
        return
    if nodes is None:
        if probe.generation_for_name(config.tpu_accelerator
                                     or "") not in ("v5e", "v6e"):
            return
        nodes = (kube.get_json_or_none(
            ["get", "nodes", "-l", node_selector(config)])
                 or {}).get("items") or []
    chips = [
        node.get("status", {}).get("allocatable", {}).get(TPU_RESOURCE)
        for node in nodes
    ]
    per_node = next((int(c) for c in chips if int(c or 0) > 0), None)
    total = config.tpu_chips * config.tpu_hosts
    if not per_node or per_node == config.tpu_chips or total % per_node:
        return
    config.tpu_chips, config.tpu_hosts = per_node, total // per_node


def token_bucket(config: GkeJobConfig) -> str | None:
    """The bucket for the Job's pod token.

    None without a staging bucket or with a pod_service_account.
    """
    if not config.staging_bucket or config.pod_service_account:
        return None
    return gcs_staging.locate(config.staging_bucket, "")[0]


def token_prefixes(config: GkeJobConfig,
                   name: str) -> tuple[str, tuple[str, ...]]:
    """The Job's own folder and its read-only folders in the bucket.

    Stable per Job, so a renewed token reaches the same folders.
    """
    own = gcs_staging.locate(config.staging_bucket, f"{name}/")[1]
    checkpoints = gcs_staging.locate(config.staging_bucket,
                                     CHECKPOINTS_PREFIX)[1]
    if own.startswith(checkpoints) or checkpoints.startswith(own):
        # The own folder is writable, so it can't overlap checkpoints.
        raise base.ComputeConfigError(
            f"Job {name!r}'s folder {own!r} overlaps the checkpoints "
            f"folder {checkpoints!r}, which its pod may only read. Choose "
            "a different --name.",
            error_code="INVALID_FIELD",
        )
    return own, (checkpoints, )


def put_token_secret(kube: Kubectl, config: GkeJobConfig,
                     name: str) -> float | None:
    """Mint the Job's pod token and store it in its Secret.

    Returns the expiry, or None if no Secret was written. Failures only
    warn, since many Jobs never use GCS.
    """
    bucket = token_bucket(config)
    if bucket is None:
        return None
    own, read_only = token_prefixes(config, name)
    try:
        token, expires = gcs_staging.mint_bucket_token(bucket, own, read_only)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(
            f"warning: couldn't create a GCS token for {name}, so its "
            f"pods can't use gs://{bucket}: {exc}",
            file=sys.stderr)
        return None
    # Client-side apply would copy the token into an annotation.
    try:
        proc = kube.run(
            ["apply", "--server-side", "--force-conflicts", "-f", "-"] +
            kube.ns(),
            stdin=json.dumps({
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {
                    "name": name + TOKEN_SECRET_SUFFIX,
                    "namespace": config.namespace
                },
                "type": "Opaque",
                "stringData": {
                    "token": token
                },
            }),
            check=False,
            timeout=KUBECTL_READ_TIMEOUT_SEC)
        # In case kubectl echoes the object back in its error.
        error = (proc.stderr or "").replace(token, "<redacted>").strip()
    except base.ComputeConfigError as exc:
        # A timeout. Our message never includes the token.
        proc, error = None, str(exc)
    if proc is None or proc.returncode != 0:
        print(
            f"warning: couldn't save the GCS token for {name}, so its "
            f"pods can't use gs://{bucket}: {error}",
            file=sys.stderr)
        return None
    return expires


def own_token_secret(kube: Kubectl, name: str, uid: str) -> None:
    """Make the Job own its token Secret, so they're deleted together."""
    try:
        kube.run(["patch", "secret", name + TOKEN_SECRET_SUFFIX] + kube.ns() +
                 [
                     "--type=merge", "-p",
                     json.dumps({
                         "metadata": {
                             "ownerReferences": [{
                                 "apiVersion": "batch/v1",
                                 "kind": "Job",
                                 "name": name,
                                 "uid": uid
                             }]
                         }
                     })
                 ],
                 timeout=KUBECTL_READ_TIMEOUT_SEC)
    # The Job is already running, so this must not fail the submit.
    except base.ComputeConfigError as exc:
        print(
            f"warning: the GCS token for {name} won't be deleted with "
            f"the Job; `compute job cancel` deletes it: {exc}",
            file=sys.stderr)


def renew_token_secret(kube: Kubectl, config: GkeJobConfig,
                       job_id: str) -> None:
    """Re-mint a running Job's pod token if it expires soon.

    Runs on every status read. Kubelet updates the mounted token file, so
    the pod doesn't need a restart. If nobody checks the Job for an hour the
    token expires, so long unattended Jobs should use pod_service_account.
    """
    record = jobs.JobRecord.load(job_id)
    if record is None or not record.extra.get("token_secret"):
        return
    now = time.time()
    expires = record.extra.get("token_expires") or 0
    if (expires - now > TOKEN_RENEW_MARGIN_SEC
            or now < (record.extra.get("token_retry_at") or 0)):
        return
    renewed = put_token_secret(kube, config, job_id)
    if renewed is None:
        # Back off so `job wait` doesn't retry and warn on every poll.
        record.extra["token_retry_at"] = now + TOKEN_RENEW_RETRY_SEC
    else:
        record.extra["token_expires"] = renewed
        record.extra.pop("token_retry_at", None)
    record.save()


def job_manifest(
    config: GkeJobConfig,
    name: str,
    script_name: str,
    interpreter: str,
    args: list[str],
    staging: StagingPlan | None = None,
    log_url: str | None = None,
    serve: bool = False,
    placement: dict[str, Any] | None = None,
    shell_python: str | None = None,
    token_secret: bool = False,
) -> dict[str, Any]:
    """Build the Job manifest."""
    metadata: dict[str, Any] = {"name": name, "namespace": config.namespace}
    template_metadata: dict[str, Any] = {}
    if config.queue:
        # Kueue suspends a Job that names no queue.
        metadata_labels: dict[str, str] = {QUEUE_LABEL: config.queue}
        # Pod labels for Kueue admission and quota tracking.
        pod_labels: dict[str, str] = {"namespace": config.namespace}
        # Kueue refuses a Job whose priority class doesn't exist.
        if config.priority:
            metadata_labels[PRIORITY_LABEL] = config.priority
            pod_labels["priority"] = config.priority
        metadata["labels"] = metadata_labels
        if config.declared_duration_minutes:
            pod_labels["declared-duration-minutes"] = str(
                config.declared_duration_minutes)
        template_metadata["labels"] = pod_labels

    # Set with or without Kueue.
    if config.mldiagnostics:
        template_metadata.setdefault("labels",
                                     {})[MLDIAGNOSTICS_LABEL] = "true"
        metadata.setdefault("labels", {})[MLDIAGNOSTICS_LABEL] = "true"

    hosts = job_hosts(config, placement)
    staging = staging or StagingPlan()
    command = (gcs_staging.build_pod_bootstrap_command(
        script_mount=SCRIPT_MOUNT,
        script_name=script_name,
        interpreter=interpreter,
        args=args,
        input_urls=staging.in_urls,
        output_urls=staging.out_urls,
        log_url=log_url,
        mounted_inputs=staging.mounted_inputs,
        logged_outputs=staging.logged_outputs,
        job=name,
        pod_token=bool(config.pod_service_account),
        service_account=config.pod_service_account or "",
        namespace=config.namespace,
        bucket=gcs_staging.clean_bucket_name(config.staging_bucket or ""),
    ) if staging.moves_files() or log_url else [
        interpreter,
        f"{SCRIPT_MOUNT}/{script_name}",
    ] + args)

    env = [{
        "name": SHM_LIMIT_ENV,
        "value": str(config.shm_size_gb * 2**30),
    }]
    scheduling: dict[str, Any]
    if placement:
        # Colocated on another Job's node, without chips. Kueue gives no
        # flavor to a pod that requests nothing.
        scheduling = dict(placement)
        resources: dict[str, Any] = {"requests": config.colocate_requests}
    elif config.cpu_requests:
        # A CPU Job. Without the TPU toleration it lands on a CPU node.
        scheduling = {}
        resources = {"requests": config.cpu_requests}
        generation = probe.generation_for_name(config.tpu_accelerator or "")
        if generation:
            env.append({"name": CHIPLESS_TPU_ENV, "value": generation})
    else:
        # TPU nodes have a NoSchedule taint, hence the toleration.
        scheduling = {
            "nodeSelector": {
                ACCELERATOR_LABEL: config.tpu_accelerator,
                TOPOLOGY_LABEL: config.tpu_topology,
            },
            "tolerations": [{
                "key": TPU_RESOURCE,
                "operator": "Exists",
                "effect": "NoSchedule",
            }],
        }
        # Without a chip request the pod silently runs on CPU.
        resources = {"limits": {TPU_RESOURCE: config.tpu_chips}}
    if hosts > 1:
        env.extend(_multi_host_env(config, name, hosts))

    template: dict[str, Any] = {
        "spec": {
            "restartPolicy":
            "OnFailure" if serve and hosts == 1 else "Never",
            # Other teams' Service env vars can clash with ours (VLLM_PORT).
            "enableServiceLinks":
            False,
            **scheduling,
            "containers": [{
                "name":
                "workload",
                "image":
                config.image,
                "command":
                command,
                "volumeMounts": [{
                    "name": "script",
                    "mountPath": SCRIPT_MOUNT,
                }, {
                    "name": "dshm",
                    "mountPath": SHM_MOUNT,
                }],
                "env":
                env,
                "resources":
                resources,
            }],
            "volumes": [
                {
                    "name": "script",
                    "configMap": {
                        "name": name
                    },
                },
                {
                    # The runtime's 64 MB default breaks vLLM and torch_tpu.
                    "name": "dshm",
                    "emptyDir": {
                        "medium": "Memory",
                        "sizeLimit": f"{config.shm_size_gb}Gi",
                    },
                }
            ],
        }
    }
    if config.pod_service_account:
        spec = template["spec"]
        spec["serviceAccountName"] = config.pod_service_account
        # Without the GKE metadata server the pod gets the node's account
        # and fails with a vague 403. Pending is clearer.
        if not placement:
            spec.setdefault("nodeSelector",
                            {})[WORKLOAD_IDENTITY_LABEL] = "true"
    if token_secret:
        # Not a subPath: kubelet doesn't update those, so renewals would fail.
        container = template["spec"]["containers"][0]
        container["volumeMounts"].append({
            "name": "gcs-token",
            "mountPath": TOKEN_MOUNT,
            "readOnly": True,
        })
        env = container.setdefault("env", [])
        for var in ("CLOUDSDK_AUTH_ACCESS_TOKEN_FILE", "GCS_TOKEN_FILE"):
            env.append({"name": var, "value": f"{TOKEN_MOUNT}/token"})
        template["spec"]["volumes"].append({
            "name": "gcs-token",
            "secret": {
                "secretName": name + TOKEN_SECRET_SUFFIX
            },
        })
    if config.hf_cache_pvc:
        container = template["spec"]["containers"][0]
        container["volumeMounts"].append({
            "name": HF_CACHE_VOLUME,
            "mountPath": HF_CACHE_MOUNT,
        })
        container.setdefault("env", []).append({
            "name": HF_CACHE_ENV,
            "value": HF_CACHE_MOUNT
        })
        if config.image:
            # Keyed on this container's image. See COMPILE_CACHE_ENV.
            container["command"] = _compile_cache_command(
                container["command"],
                f"{HF_CACHE_MOUNT}/{COMPILE_CACHE_SUBDIR}/"
                f"{config.tpu_accelerator}-{config.tpu_topology}/"
                f"{compile_cache_key(config.image)}",
                config.compile_cache_max_age_days)
        template["spec"]["volumes"].append({
            "name": HF_CACHE_VOLUME,
            "persistentVolumeClaim": {
                "claimName": config.hf_cache_pvc
            },
        })
    else:
        # See POD_CACHE_ENV.
        container = template["spec"]["containers"][0]
        container["command"] = _pod_cache_command(container["command"])
    if shell_python:
        # Kubernetes doesn't expand `~`. The scripts do.
        template["spec"]["containers"][0].setdefault("env", []).append({
            "name":
            "TPU_AGENT_SUITE_PYTHON",
            "value":
            shell_python
        })
    if template_metadata:
        template["metadata"] = template_metadata
    job_spec: dict[str, Any] = {
        "backoffLimit": SERVE_BACKOFF_LIMIT if serve and hosts == 1 else 0,
        "template": template,
    }
    if hosts > 1:
        # One pod per node, started together, each named
        # `<name>-<index>.<name>` by the headless Service `_expose` adds.
        # libtpu opens no device until every host is up. Any pod failing
        # fails the Job, even with --serve, so a slice never runs with a
        # host missing.
        job_spec.update({
            "completionMode": "Indexed",
            "completions": hosts,
            "parallelism": hosts,
        })
        template["spec"]["subdomain"] = name
        # Keeps two multi-host Jobs from each landing a pod on the same node
        # pool and deadlocking each other in Pending.
        template["spec"]["affinity"] = {
            "podAntiAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": [{
                    "labelSelector": {
                        "matchExpressions": [
                            {
                                "key": "job-name",
                                "operator": "Exists",
                            },
                            {
                                "key": "job-name",
                                "operator": "NotIn",
                                "values": [name],
                            },
                        ],
                    },
                    "topologyKey":
                    NODEPOOL_LABEL,
                }],
            },
        }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": metadata,
        "spec": job_spec,
    }


def job_hosts(config: GkeJobConfig,
              placement: dict[str, Any] | None = None) -> int:
    """Pods in a Job: one per node of the slice, or one for a colocated or
    CPU Job, which spans no slice."""
    return 1 if placement or config.cpu_requests else config.tpu_hosts


def _multi_host_env(config: GkeJobConfig, name: str,
                    hosts: int) -> list[dict[str, Any]]:
    """Env telling each pod of a multi-host Job which host it is and where
    the leader is.

    `TPU_WORKER_ID` (and `TPU_NODE_ID`, read by vllm_torchtpu) is the pod's
    index; 0 is the leader, which runs the server or coordinator the others
    join. libtpu needs `TPU_WORKER_HOSTNAMES`, which GKE doesn't set for a
    Job.
    """
    index = {
        "fieldRef": {
            "fieldPath": f"metadata.annotations['{COMPLETION_INDEX_KEY}']"
        }
    }
    env = [{
        "name": n,
        "valueFrom": index
    } for n in ("TPU_WORKER_ID", "TPU_NODE_ID")]
    hostnames = [
        f"{name}-{i}.{name}.{config.namespace}.svc" for i in range(hosts)
    ]
    for n, v in (("TPU_NUM_HOSTS", str(hosts)), ("TPU_CHIPS_PER_HOST",
                                                 str(config.tpu_chips)),
                 ("TPU_HEAD_HOST", hostnames[0]), ("TPU_WORKER_HOSTNAMES",
                                                   ",".join(hostnames)),
                 *((m, "") for m in MEGASCALE_ENV)):
        env.append({"name": n, "value": v})
    return env
