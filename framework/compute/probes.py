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
"""Suggested answers for `compute setup plan`, found by read-only probes.

Each probe runs one read-only command (gcloud, kubectl, python3) and reads
fields from its output. The probes run in parallel, each for at most
PROBE_TIMEOUT_SEC. A probe that finds nothing just leaves its questions
without a suggestion; one whose tool is missing or that runs out of time is
also reported, so the user knows why.
"""

from __future__ import annotations

import collections
import concurrent.futures
import itertools
import json
import os
import shutil
from typing import Any, Callable

from .drivers import base, probe, registry
from .drivers.base import ACCELERATOR_LABEL, TOPOLOGY_LABEL, TPU_RESOURCE

# Seconds each probe may take.
PROBE_TIMEOUT_SEC = 10

# The probes that can tell which backend to use (see `_backend`).
_BACKEND_PROBES = ("context", "tpu_nodes", "tpu_pods", "tpu_vms")

# Most values a probe offers as choices when none is clearly meant.
MAX_CHOICES = 10


def _source(cmd: list[str]) -> str:
    """The command up to its first flag, e.g. "kubectl get nodes"."""
    return " ".join(itertools.takewhile(lambda t: not t.startswith("-"), cmd))


def _run(cmd: list[str]) -> tuple[str, str | None]:
    """Run a probe; return its output, or why it couldn't run.

    A command that runs but fails (say, `kubectl get localqueues` on a
    cluster without Kueue) isn't reported; it just suggests nothing.
    """
    if not shutil.which(cmd[0]):
        return "", f"{cmd[0]} is not installed"
    proc = base.probe_run(cmd, PROBE_TIMEOUT_SEC)
    if proc.returncode == 0:
        return proc.stdout.strip(), None
    # probe_run returns -1 when the command timed out or couldn't start.
    return "", (f"`{_source(cmd)}` {proc.stderr}"
                if proc.returncode == -1 else None)


def _escape(label: str) -> str:
    """A label name as a jsonpath key, its dots escaped."""
    return label.replace(".", "\\.")


def _pairs(out: str) -> list[tuple[str, ...]]:
    """The lines of `out` that are two words, split."""
    return [
        tuple(line.split()) for line in out.splitlines()
        if len(line.split()) == 2
    ]


def _tpu_nodes(out: str) -> dict[str, str]:
    """The most common accelerator and topology among the TPU nodes."""
    pairs = collections.Counter(_pairs(out))
    if not pairs:
        return {}
    accelerator, topology = pairs.most_common(1)[0][0]
    return {"tpu_accelerator": accelerator, "tpu_topology": topology}


def _tpu_vm(out: str) -> dict[str, Any]:
    """The TPU VM meant: the only one, else the only one named for $USER.

    In a shared project the first VM listed is likely someone else's, so
    with no clear pick the VMs are offered as `tpu_name` choices, $USER's
    first, each naming its zone too: a name is unique only within a zone.
    The zone is suggested only if every VM listed, offered or not, is in it.
    """
    vms = _pairs(out)
    user = os.environ.get("USER")
    mine = [vm for vm in vms if user and user in vm[0]]
    for pick in (vms, mine):
        if len(pick) == 1:
            return {"tpu_name": pick[0][0], "zone": pick[0][1]}
    if not vms:
        return {}
    listed = (mine + [vm for vm in vms if vm not in mine])[:MAX_CHOICES]
    found: dict[str, Any] = {
        "tpu_name": [{
            "tpu_name": name,
            "zone": zone
        } for name, zone in listed]
    }
    zones = {zone for _, zone in vms}
    if len(zones) == 1:
        found["zone"] = zones.pop()
    return found


def _tpu_pod(out: str) -> dict[str, str]:
    """The first Running pod with a container that requests TPU chips."""
    try:
        pods = json.loads(out).get("items", []) if out else []
    except ValueError:
        return {}
    for pod in pods:
        if (pod.get("status") or {}).get("phase") != "Running":
            continue
        containers = (pod.get("spec") or {}).get("containers") or []
        for container in containers:
            limits = (container.get("resources") or {}).get("limits") or {}
            if TPU_RESOURCE in limits:
                found = {"pod": pod["metadata"]["name"]}
                if len(containers) > 1:
                    found["container"] = container["name"]
                return found
    return {}


def _probes(
    known: dict[str, Any]
) -> dict[str, tuple[list[str], tuple[str, ...], Callable | None]]:
    """Each probe: its command, the fields it fills in, and how to read them.

    Without a reader, each word on the output's first line fills the next
    field. `known` holds the answers so far, so a probe can use them, e.g.
    to list TPU VMs in the chosen project or queues in the chosen namespace.
    """
    project = [f"--project={known['project']}"] if known.get("project") else []
    namespace = ["-n", known["namespace"]] if known.get("namespace") else []
    # Last, so `_source` still names the subcommand.
    kube = base.kubectl_prefix(known.get("kubeconfig"),
                               known.get("kube_context"))[1:]
    # The context itself is suggested, for a profile to pin.
    probes = {
        "context":
        # `current-context` ignores --context, so only the kubeconfig.
        (["kubectl", "config", "current-context"] +
         base.kubectl_prefix(known.get("kubeconfig"))[1:], ("kube_context", ),
         (lambda out: {
             "kube_context": out
         } if out else {})),
        "project": (["gcloud", "config", "get-value",
                     "project"], ("project", ), lambda out: {
                         "project": out
                     } if out not in ("", "(unset)") else {}),
        "tpu_vms": ([
            "gcloud", "compute", "tpus", "tpu-vm", "list", "--zone=-",
            "--format=value(name.basename(),name.segment(-3))"
        ] + project, ("tpu_name", "zone"), _tpu_vm),
        "namespace": ([
            "kubectl", "config", "view", "--minify", "-o",
            "jsonpath={..namespace}"
        ] + kube, ("namespace", ), None),
        "tpu_nodes": ([
            "kubectl", "get", "nodes", "-l", ACCELERATOR_LABEL, "-o",
            "jsonpath={range .items[*]}"
            f"{{.metadata.labels.{_escape(ACCELERATOR_LABEL)}}} "
            f"{{.metadata.labels.{_escape(TOPOLOGY_LABEL)}}}"
            '{"\\n"}{end}'
        ] + kube, ("tpu_accelerator", "tpu_topology"), _tpu_nodes),
        "queues": (["kubectl", "get", "localqueues"] + namespace +
                   ["-o", "jsonpath={.items[*].metadata.name}"] + kube,
                   ("queue", ), None),
        "tpu_pods":
        (["kubectl", "get", "pods"] + namespace + ["-o", "json"] + kube,
         ("pod", "container"), _tpu_pod),
    }
    if known.get("backend") == "local":
        probes["torch_tpu"] = ([
            "python3", "-c", "import sys, torch_tpu; print(sys.executable)"
        ], ("python_interpreter", ), None)
    return probes


def _backend(found: dict[str, dict[str, Any]], vms: str,
             backends: list[str]) -> tuple[str, str] | None:
    """The backend the probes point to, in the order the design doc gives."""
    accel, vfio = probe.list_devices()
    if (accel or vfio) and "local" in backends:
        return "local", "TPU devices on this host"
    for name in backends:
        drv = registry.get_driver(name)
        if drv is not None and (source := drv.suggest_backend(found)):
            return name, source
    for backend, hit, source in (
        ("gke_job", found.get("tpu_nodes"),
         "TPU nodes in the kubectl context"),
        ("gke_pod", found.get("tpu_pods"), "running pods that request TPUs"),
        ("gce_tpu_vm", vms, "gcloud compute tpus tpu-vm list"),
    ):
        if hit and backend in backends:
            return backend, source
    return None


def suggest(
        fields: list[str], known: dict[str, Any],
        backends: list[str]) -> tuple[dict[str, tuple[Any, str]], list[str]]:
    """Suggestions for `fields`, and the probes that couldn't run.

    Returns `{field: (value, source)}` for each field a probe could answer
    (a list `value` holds choices: several found, none clearly meant),
    and one line per missing tool or timed-out probe, or a cluster with no
    TPU nodes. Fields in `known`
    already have answers and aren't probed; `backends` are the registered
    backends `backend` may suggest.
    """
    needed = {f for f in fields if f not in known}
    probes = {
        name: p
        for name, p in _probes(known).items() if needed
        & set(p[1]) or ("backend" in needed and name in _BACKEND_PROBES)
    }
    runs: dict[str, tuple[str, str | None]] = {}
    if probes:
        with concurrent.futures.ThreadPoolExecutor(len(probes)) as pool:
            runs = dict(
                zip(probes, pool.map(_run, (p[0] for p in probes.values()))))
    found: dict[str, dict[str, Any]] = {}
    result: dict[str, tuple[Any, str]] = {}
    for name, (out, _) in runs.items():
        cmd, gives, read = probes[name]
        found[name] = (read(out) if read else dict(
            zip(gives,
                out.splitlines()[0].split() if out else [])))
        for field, value in found[name].items():
            if field in needed and field in gives:
                result.setdefault(field, (value, _source(cmd)))
    if "backend" in needed:
        vms = runs.get("tpu_vms", ("", None))[0]
        if (choice := _backend(found, vms, backends)) is not None:
            result["backend"] = choice
    warnings = {why for _, why in runs.values() if why}
    # With a backend chosen, no TPU node means the cluster has none yet, not
    # that another backend fits.
    if "tpu_nodes" in runs and not found["tpu_nodes"] and (
            "backend" not in needed) and runs["tpu_nodes"][1] is None:
        warnings.add("`kubectl get nodes` found no TPU nodes in the kubectl "
                     "context; a node pool may still be starting.")
    return result, sorted(warnings)
