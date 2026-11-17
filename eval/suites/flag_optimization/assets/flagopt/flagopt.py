#!/usr/bin/env python3
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
"""flagopt: launch, measure and compare MaxText training runs on a GKE TPU slice.

The same file is the agent's toolkit (`flagopt ...` in the task container) and
the verifier's library (it imports the root-owned /opt/flagopt copy), so what
the agent measures during the session is measured the same way at grading
time. The task comes from the catalog /opt/flagopt/tasks.json, selected by
$FLAGOPT_TASK (see `expand_task`).

Stdlib only, so it runs with `python3 -I -S`.

A *candidate config* is a JSON object:

    {
      "maxtext_overrides": {"ici_fsdp_parallelism": 16, "remat_policy": "full"},
      "xla_flags": ["--xla_tpu_scoped_vmem_limit_kib=98304"],
      "notes": "free text, ignored"
    }

Only knobs listed under `tunables` in the task spec may appear in
`maxtext_overrides`; anything not given keeps its baseline value. Everything
else (model, precision, batch, sequence length, optimizer, data, seeds) is
fixed by the spec and cannot be overridden.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_SPEC_PATH = os.environ.get("FLAGOPT_SPEC", "/opt/flagopt/tasks.json")
DEFAULT_RUNS_DIR = os.environ.get("FLAGOPT_RUNS_DIR", "/workspace/runs")
METRIC_MARKER = "FLAGOPT_METRIC "
EXIT_MARKER = "FLAGOPT_EXIT_CODE "
DONE_MARKER = "FLAGOPT_DONE"
COMPLETION_INDEX_KEY = "batch.kubernetes.io/job-completion-index"
ACCELERATOR_LABEL = "cloud.google.com/gke-tpu-accelerator"
TOPOLOGY_LABEL = "cloud.google.com/gke-tpu-topology"
TPU_RESOURCE = "google.com/tpu"
QUEUE_LABEL = "kueue.x-k8s.io/queue-name"
PRIORITY_LABEL = "kueue.x-k8s.io/priority-class"
MEGASCALE_ENV = ("MEGASCALE_NUM_SLICES", "MEGASCALE_SLICE_ID",
                 "MEGASCALE_PORT", "MEGASCALE_COORDINATOR_ADDRESS")
ICI_AXES = (
    "ici_data_parallelism",
    "ici_fsdp_parallelism",
    "ici_fsdp_transpose_parallelism",
    "ici_sequence_parallelism",
    "ici_context_parallelism",
    "ici_tensor_parallelism",
    "ici_tensor_sequence_parallelism",
    "ici_expert_parallelism",
    "ici_pipeline_parallelism",
    "ici_autoregressive_parallelism",
)
XLA_FLAG_RE = re.compile(
    r"^--(xla|megascale|libtpu)_[a-z0-9_]+(=[A-Za-z0-9_.,:+\-]*)?$")
K8S_NAME_RE = re.compile(r"^[a-z]([-a-z0-9]*[a-z0-9])?$")


class ConfigError(ValueError):
    """The candidate config is not valid for this task."""


class KubeError(RuntimeError):
    """A kubectl call failed."""


# --------------------------------------------------------------------------
# Spec and candidate validation
# --------------------------------------------------------------------------


def _derived_choices(rule: str, *, num_devices: int,
                     per_device_batch_size: float) -> list[Any]:
    """Knob choices that depend on the slice or the batch, not on the knob."""
    if rule == "powers_of_two_up_to_num_devices":
        out, v = [], 1
        while v <= num_devices:
            out.append(v)
            v *= 2
        return out
    if rule == "per_device_batch_size_fractions":
        return [per_device_batch_size / d for d in (8, 4, 2, 1)]
    raise RuntimeError(f"unknown choices rule {rule!r} in the task catalog")


def expand_task(catalog: dict[str, Any], task_id: str) -> dict[str, Any]:
    """Resolve one task of the catalog (tasks.json) into its full spec.

    A task entry names a model, a topology, the per-device batch, the knob
    groups it unlocks and its baseline mesh; everything else comes from the
    catalog: the model's facts, fixed settings, applicable knobs and default
    values, the topology's hardware, and the shared MaxText, measurement and
    cluster settings. A knob of a locked group keeps MaxText's default, except
    those `locked_fixed` pins to the baseline value.
    """
    tasks = catalog["tasks"]
    if task_id not in tasks:
        raise RuntimeError(
            f"unknown task {task_id!r}; the catalog has {sorted(tasks)}")
    t = tasks[task_id]
    model = catalog["models"][t["model"]]
    hw = dict(catalog["topologies"][t["topology"]])
    batch_pdbs = t["per_device_batch_size"]
    pdbs = float(batch_pdbs)
    gas = int(t.get("gradient_accumulation_steps", 1))
    groups = list(t["groups"])
    every = "all" in groups
    unlocked = set(catalog["knob_groups"]) if every else set(groups)

    defaults = {
        **model["baseline"],
        **t.get("mesh", {}),
        "per_device_batch_size": pdbs,
        "gradient_accumulation_steps": gas,
    }
    tunables: dict[str, Any] = {}
    for name, knob in catalog["knobs"].items():
        if name in model.get("exclude_knobs",
                             []) or knob["group"] not in unlocked:
            continue
        knob = dict(knob)
        if isinstance(knob.get("choices"), str):
            knob["choices"] = _derived_choices(knob["choices"],
                                               num_devices=int(
                                                   hw["num_devices"]),
                                               per_device_batch_size=pdbs)
        tunables[name] = knob

    fixed = {**model["fixed"], **catalog["fixed"]}
    pinned = {**defaults, "per_device_batch_size": batch_pdbs}
    for group, names in catalog.get("locked_fixed", {}).items():
        if group not in unlocked:
            for name in names:
                if name not in model.get("exclude_knobs", []):
                    fixed[name] = pinned[name]

    cluster = {
        **catalog["cluster"],
        "topology": t["topology"],
        "ckpt_uri": model["info"]["ckpt_uri_default"],
    }
    budget = int(t["time_budget_sec"])
    return {
        "task_id": task_id,
        "title": t["title"],
        "doc_row": t["doc_row"],
        "model": dict(model["info"]),
        "hardware": hw,
        "cluster": cluster,
        "maxtext": dict(catalog["maxtext"]),
        "batch": {
            "per_device_batch_size": batch_pdbs,
            "gradient_accumulation_steps": gas,
            "global_batch_size": int(pdbs * gas * int(hw["num_devices"])),
            "seq_len": int(catalog["fixed"]["max_target_length"]),
        },
        "fixed": fixed,
        "tunables": tunables,
        "xla_flags": {
            "tunable": every or "xla" in groups,
            **catalog["xla_flags"]
        },
        "baseline": {
            "maxtext_overrides": {
                k: defaults[k]
                for k in tunables
            },
            "xla_flags": list(t.get("baseline_xla_flags", [])),
        },
        "constraints": list(catalog.get("constraints", [])),
        "measurement": dict(catalog["measurement"]),
        "time_budget_sec": budget,
        "agent_timeout_sec": budget,
        "verifier_timeout_sec": int(t["verifier_timeout_sec"]),
        "groups": groups,
        "notes": t.get("notes", ""),
    }


def load_spec(path: str | os.PathLike[str] | None = None,
              task_id: str | None = None) -> dict[str, Any]:
    """The task spec: one task of the catalog, or an already resolved spec.

    `path` defaults to $FLAGOPT_SPEC or /opt/flagopt/tasks.json, and the task
    to $FLAGOPT_TASK, which each task sets in its environment.
    """
    p = Path(path or DEFAULT_SPEC_PATH)
    spec = json.loads(p.read_text(encoding="utf-8"))
    if "tasks" in spec:
        task_id = task_id or os.environ.get("FLAGOPT_TASK")
        if not task_id:
            raise RuntimeError(
                f"{p} holds several tasks; set FLAGOPT_TASK to one of "
                f"{sorted(spec['tasks'])}")
        spec = expand_task(spec, task_id)
    overlap = set(spec["fixed"]) & set(spec["tunables"])
    if overlap:
        raise RuntimeError(
            f"task spec lists {sorted(overlap)} as both fixed and tunable")
    return spec


def cluster_settings(spec: dict[str, Any]) -> dict[str, str]:
    """Cluster target: the spec's defaults, overridable by env vars.

    Namespace, queue, priority and declared duration follow the harness
    config's GKE_* variables, which describe the cluster the trial's kubeconfig
    points at. The slice shape does not: a config's GKE_TPU_ACCELERATOR /
    GKE_TPU_TOPOLOGY are its default slice (e.g. 2x2x1), not this task's, and
    reach the agent's environment ahead of the task's own. The slice is part of
    the task, so only the FLAGOPT_TPU_* names override it.
    """
    c = dict(spec["cluster"])
    env_map = {
        "namespace": "GKE_NAMESPACE",
        "queue": "GKE_QUEUE",
        "priority": "GKE_PRIORITY",
        "declared_duration_minutes": "GKE_DECLARED_DURATION_MINUTES",
        "accelerator": "FLAGOPT_TPU_ACCELERATOR",
        "topology": "FLAGOPT_TPU_TOPOLOGY",
        "image": "TPU_IMAGE",
        "ckpt_uri": "MODEL_CKPT_URI",
        "gcs_scratch": "FLAGOPT_GCS_SCRATCH",
    }
    for key, env in env_map.items():
        val = os.environ.get(env)
        if val:
            c[key] = val
    return c


def _check_value(name: str, knob: dict[str, Any], value: Any) -> Any:
    kind = knob["type"]
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{name}: expected an integer, got {value!r}")
        if "choices" in knob and value not in knob["choices"]:
            raise ConfigError(
                f"{name}: {value} is not one of {knob['choices']}")
        if "min" in knob and value < knob["min"]:
            raise ConfigError(f"{name}: {value} < min {knob['min']}")
        if "max" in knob and value > knob["max"]:
            raise ConfigError(f"{name}: {value} > max {knob['max']}")
        if knob.get("multiple_of") and value % knob["multiple_of"]:
            raise ConfigError(
                f"{name}: {value} is not a multiple of {knob['multiple_of']}")
        return value
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{name}: expected a number, got {value!r}")
        value = float(value)
        if "choices" in knob and value not in knob["choices"]:
            raise ConfigError(
                f"{name}: {value} is not one of {knob['choices']}")
        if "min" in knob and value < knob["min"]:
            raise ConfigError(f"{name}: {value} < min {knob['min']}")
        if "max" in knob and value > knob["max"]:
            raise ConfigError(f"{name}: {value} > max {knob['max']}")
        return value
    if kind == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"{name}: expected true/false, got {value!r}")
        return value
    if kind == "enum":
        if value not in knob["choices"]:
            raise ConfigError(
                f"{name}: {value!r} is not one of {knob['choices']}")
        return value
    raise RuntimeError(f"unknown knob type {kind!r} for {name}")


def baseline_candidate(spec: dict[str, Any]) -> dict[str, Any]:
    """The task's current (baseline) config, as a candidate."""
    return {
        "maxtext_overrides": {},
        "xla_flags": list(spec["baseline"].get("xla_flags", []))
    }


def validate_candidate(spec: dict[str, Any],
                       cand: Any) -> tuple[dict[str, Any], list[str]]:
    """Return (resolved MaxText overrides, XLA flags) or raise ConfigError."""
    if not isinstance(cand, dict):
        raise ConfigError("config must be a JSON object")
    extra = set(cand) - {"maxtext_overrides", "xla_flags", "notes"}
    if extra:
        raise ConfigError(
            f"unknown top-level keys {sorted(extra)}; allowed: maxtext_overrides, xla_flags, notes"
        )
    overrides = cand.get("maxtext_overrides", {}) or {}
    if not isinstance(overrides, dict):
        raise ConfigError("maxtext_overrides must be an object")
    tunables = spec["tunables"]
    for key in overrides:
        if key in spec["fixed"]:
            raise ConfigError(
                f"{key} is fixed by the task and cannot be changed")
        if key not in tunables:
            raise ConfigError(
                f"{key} is not a tunable knob of this task (see `flagopt spec`)"
            )
    resolved = dict(spec["baseline"]["maxtext_overrides"])
    for key, value in overrides.items():
        resolved[key] = _check_value(key, tunables[key], value)

    # The mesh must cover every device of the slice exactly once.
    num_devices = int(spec["hardware"]["num_devices"])
    product = 1
    for axis in ICI_AXES:
        v = resolved.get(axis, spec["fixed"].get(axis, 1))
        if not isinstance(v, int) or v < 1:
            raise ConfigError(
                f"{axis} must be a positive integer, got {v!r} (-1 auto-fill is not accepted)"
            )
        product *= v
    if product != num_devices:
        raise ConfigError(
            f"product of ici_*_parallelism is {product}, but the slice has {num_devices} devices"
        )

    # Tokens per optimizer step are fixed.
    tokens = spec["batch"]
    pdbs = resolved.get("per_device_batch_size",
                        spec["fixed"].get("per_device_batch_size"))
    gas = resolved.get("gradient_accumulation_steps",
                       spec["fixed"].get("gradient_accumulation_steps", 1))
    if not math.isclose(
            float(pdbs) * float(gas),
            float(tokens["per_device_batch_size"]) *
            float(tokens.get("gradient_accumulation_steps", 1))):
        raise ConfigError(
            f"per_device_batch_size * gradient_accumulation_steps must stay "
            f"{tokens['per_device_batch_size'] * tokens.get('gradient_accumulation_steps', 1)} "
            f"(global batch {tokens['global_batch_size']} x seq {tokens['seq_len']} is fixed); got {pdbs} * {gas}"
        )

    for rule in spec.get("constraints", []):
        _check_rule(rule, resolved, spec)

    # Like the MaxText overrides, XLA flags left out keep the baseline's.
    xla = cand.get("xla_flags", spec["baseline"].get("xla_flags", [])) or []
    if not isinstance(xla, list) or not all(isinstance(f, str) for f in xla):
        raise ConfigError("xla_flags must be a list of strings")
    xla = [f.strip() for f in xla if f.strip()]
    xcfg = spec["xla_flags"]
    if not xcfg["tunable"]:
        if xla and xla != spec["baseline"].get("xla_flags", []):
            raise ConfigError(
                "XLA flags are not tunable in this task; leave xla_flags empty"
            )
        xla = list(spec["baseline"].get("xla_flags", []))
    else:
        if len(xla) > xcfg.get("max_flags", 96):
            raise ConfigError(f"at most {xcfg.get('max_flags', 96)} XLA flags")
        seen = set()
        for f in xla:
            if not XLA_FLAG_RE.match(f):
                raise ConfigError(
                    f"malformed XLA/libtpu flag {f!r}; expected --xla_name=value"
                )
            fname = f.split("=", 1)[0]
            if fname in seen:
                raise ConfigError(f"duplicate flag {fname}")
            seen.add(fname)
            for bad in xcfg.get("denied_substrings", []):
                if bad in fname:
                    raise ConfigError(
                        f"flag {fname} is not allowed in this task ({bad!r} flags are denied)"
                    )
            if fname in xcfg.get("denied_flags", []):
                raise ConfigError(f"flag {fname} is not allowed in this task")
        if len(" ".join(xla)) > 16000:
            raise ConfigError("XLA flag string too long")
    return resolved, xla


def _check_rule(rule: dict[str, Any], resolved: dict[str, Any],
                spec: dict[str, Any]) -> None:
    kind = rule["kind"]
    if kind == "divides":
        # resolved[knob] must divide `value`.
        v = resolved.get(rule["knob"], spec["fixed"].get(rule["knob"]))
        if isinstance(v, int) and v > 0 and rule["value"] % v:
            raise ConfigError(
                f"{rule['knob']}={v} must divide {rule['value']} ({rule.get('why', '')})"
            )
    elif kind == "max_product":
        prod = 1
        for k in rule["knobs"]:
            prod *= int(resolved.get(k, spec["fixed"].get(k, 1)))
        if prod > rule["value"]:
            raise ConfigError(
                f"product of {rule['knobs']} = {prod} exceeds {rule['value']} ({rule.get('why', '')})"
            )
    else:
        raise RuntimeError(f"unknown constraint kind {kind}")


# --------------------------------------------------------------------------
# MaxText command and Kubernetes manifests
# --------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value)) if abs(value) >= 1 else repr(value)
    return str(value)


def maxtext_args(spec: dict[str, Any], resolved: dict[str, Any], *,
                 run_name: str, steps: int, weights: str,
                 cluster: dict[str, str]) -> list[str]:
    """`key=value` overrides passed to MaxText's trainer, fixed settings last so they win."""
    args: dict[str, Any] = {}
    args.update(resolved)
    args.update(spec["fixed"])
    args["run_name"] = run_name
    args["steps"] = steps
    args["metrics_file"] = "/tmp/flagopt/metrics.jsonl"
    scratch = cluster.get("gcs_scratch") or ""
    args["base_output_directory"] = scratch.rstrip(
        "/") + "/" + run_name if scratch else "/tmp/flagopt/out"
    if weights == "checkpoint":
        uri = cluster.get("ckpt_uri") or ""
        if not uri:
            raise ConfigError(
                "weights=checkpoint needs MODEL_CKPT_URI (or cluster.ckpt_uri in the spec)"
            )
        if uri.startswith("hf://") and not scratch.startswith("gs://"):
            raise ConfigError(
                "an hf:// checkpoint needs FLAGOPT_GCS_SCRATCH=gs://... for MaxText's HF cache"
            )
        args["load_parameters_path"] = uri
        args["source_checkpoint_layout"] = "safetensors_dynamic"
    elif weights != "random":
        raise ConfigError(
            f"weights must be 'checkpoint' or 'random', got {weights!r}")
    return [f"{k}={_fmt(v)}" for k, v in args.items()]


def pod_script(spec: dict[str, Any],
               args: list[str],
               xla_flags: list[str],
               *,
               hold_sec: int = 0,
               profile: bool = False) -> str:
    """The bash program each TPU pod runs (identical on every host)."""
    entry = spec["maxtext"]["entrypoint"]
    base_yml = spec["maxtext"]["base_config"]
    extra = []
    if profile:
        extra = [
            "profiler=xplane", "skip_first_n_steps_for_profiler=2",
            "profiler_steps=3"
        ]
    cmd = ["python3", "-u", "-m", entry, base_yml] + args + extra
    lines = [
        "set -u",
        "mkdir -p /tmp/flagopt/out",
        f"export LIBTPU_INIT_ARGS={shlex.quote(' '.join(xla_flags))}",
        "export TPU_STDERR_LOG_LEVEL=${TPU_STDERR_LOG_LEVEL:-2}",
        'echo "FLAGOPT_START host=${TPU_WORKER_ID:-0} $(date -u +%FT%TZ)"',
        'echo "FLAGOPT_LIBTPU_INIT_ARGS ${LIBTPU_INIT_ARGS}"',
        "cd " + shlex.quote(spec["maxtext"]["workdir"]),
        " ".join(shlex.quote(c)
                 for c in cmd) + " 2>&1 | tee /tmp/flagopt/train.log",
        "rc=${PIPESTATUS[0]}",
        'if [ "${TPU_WORKER_ID:-0}" = "0" ]; then',
        '  if [ -f /tmp/flagopt/metrics.jsonl ]; then',
        f'    while IFS= read -r line; do echo "{METRIC_MARKER}$line"; done < /tmp/flagopt/metrics.jsonl',
        "  fi",
        f'  echo "{EXIT_MARKER}$rc"',
        f'  echo "{DONE_MARKER} $(date -u +%FT%TZ)"',
    ]
    if hold_sec > 0:
        lines += [
            f'  echo "FLAGOPT_HOLD {hold_sec}s: kubectl cp the pod\'s /tmp/flagopt now"',
            f"  sleep {int(hold_sec)}",
        ]
    lines += ["fi", "exit $rc"]
    return "\n".join(lines) + "\n"


def job_manifests(spec: dict[str, Any], cluster: dict[str, str], name: str,
                  script: str, role: str) -> dict[str, Any]:
    hw = spec["hardware"]
    hosts = int(hw["hosts"])
    chips_per_host = int(hw["chips_per_host"])
    ns = cluster["namespace"]
    labels = {
        "app": "tpu-flagopt",
        "flagopt-role": role,
        "flagopt-task": spec["task_id"][:63]
    }
    meta_labels = dict(labels)
    pod_labels = dict(labels)
    if cluster.get("queue"):
        meta_labels[QUEUE_LABEL] = cluster["queue"]
        pod_labels["namespace"] = ns
        if cluster.get("priority"):
            meta_labels[PRIORITY_LABEL] = cluster["priority"]
            pod_labels["priority"] = cluster["priority"]
        if cluster.get("declared_duration_minutes"):
            pod_labels["declared-duration-minutes"] = str(
                cluster["declared_duration_minutes"])
    env: list[dict[str, Any]] = [{"name": "PYTHONUNBUFFERED", "value": "1"}]
    if hosts > 1:
        index = {
            "fieldRef": {
                "fieldPath": f"metadata.annotations['{COMPLETION_INDEX_KEY}']"
            }
        }
        env += [{
            "name": n,
            "valueFrom": index
        } for n in ("TPU_WORKER_ID", "TPU_NODE_ID")]
        hostnames = [f"{name}-{i}.{name}.{ns}.svc" for i in range(hosts)]
        env += [
            {
                "name": "TPU_NUM_HOSTS",
                "value": str(hosts)
            },
            {
                "name": "TPU_CHIPS_PER_HOST",
                "value": str(chips_per_host)
            },
            {
                "name": "TPU_HEAD_HOST",
                "value": hostnames[0]
            },
            {
                "name": "TPU_WORKER_HOSTNAMES",
                "value": ",".join(hostnames)
            },
        ]
        env += [{"name": m, "value": ""} for m in MEGASCALE_ENV]
    else:
        env.append({"name": "TPU_WORKER_ID", "value": "0"})
    pod_spec: dict[str, Any] = {
        "restartPolicy":
        "Never",
        "enableServiceLinks":
        False,
        "terminationGracePeriodSeconds":
        30,
        "nodeSelector": {
            ACCELERATOR_LABEL: cluster["accelerator"],
            TOPOLOGY_LABEL: cluster["topology"]
        },
        "tolerations": [{
            "key": TPU_RESOURCE,
            "operator": "Exists",
            "effect": "NoSchedule"
        }],
        "containers": [{
            "name":
            "maxtext",
            "image":
            cluster["image"],
            "command": ["bash", "-c", script],
            "env":
            env,
            "resources": {
                "limits": {
                    TPU_RESOURCE: chips_per_host
                }
            },
            "volumeMounts": [{
                "name": "dshm",
                "mountPath": "/dev/shm"
            }],
        }],
        "volumes": [{
            "name": "dshm",
            "emptyDir": {
                "medium": "Memory",
                "sizeLimit": "128Gi"
            }
        }],
    }
    job_spec: dict[str, Any] = {
        "backoffLimit":
        0,
        "activeDeadlineSeconds":
        int(spec["measurement"].get("job_deadline_sec", 7200)),
        "ttlSecondsAfterFinished":
        600,
        "template": {
            "metadata": {
                "labels": pod_labels
            },
            "spec": pod_spec
        },
    }
    if hosts > 1:
        job_spec.update({
            "completionMode": "Indexed",
            "completions": hosts,
            "parallelism": hosts
        })
        pod_spec["subdomain"] = name
        pod_spec["affinity"] = {
            "podAntiAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": [{
                    "labelSelector": {
                        "matchExpressions": [
                            {
                                "key": "job-name",
                                "operator": "Exists"
                            },
                            {
                                "key": "job-name",
                                "operator": "NotIn",
                                "values": [name]
                            },
                        ]
                    },
                    "topologyKey":
                    "cloud.google.com/gke-nodepool",
                }]
            }
        }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": ns,
            "labels": meta_labels
        },
        "spec": job_spec
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": name,
            "namespace": ns,
            "labels": dict(labels)
        },
        "spec": {
            "clusterIP": "None",
            "publishNotReadyAddresses": True,
            "selector": {
                "job-name": name
            }
        }
    }
    return {"job": job, "service": service if hosts > 1 else None}


# --------------------------------------------------------------------------
# kubectl
# --------------------------------------------------------------------------


class Kube:
    """Thin kubectl runner. `cmd` is the kubectl argv prefix."""

    def __init__(self,
                 cmd: list[str] | None = None,
                 namespace: str | None = None):
        if cmd is None:
            env_cmd = os.environ.get("FLAGOPT_KUBECTL")
            cmd = shlex.split(env_cmd) if env_cmd else ["kubectl"]
        self.cmd = cmd
        self.namespace = namespace

    def run(self,
            args: list[str],
            *,
            stdin: str | None = None,
            timeout: float = 120,
            check: bool = True) -> subprocess.CompletedProcess[str]:
        argv = list(self.cmd) + list(args)
        if self.namespace and "-n" not in args and "--namespace" not in args:
            argv += ["-n", self.namespace]
        try:
            proc = subprocess.run(argv,
                                  input=stdin,
                                  capture_output=True,
                                  text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise KubeError(
                f"kubectl {' '.join(args[:3])} timed out after {timeout}s"
            ) from exc
        if check and proc.returncode != 0:
            raise KubeError(
                f"kubectl {' '.join(args[:4])} failed ({proc.returncode}): {proc.stderr.strip()[:2000]}"
            )
        return proc

    def get_json(self,
                 args: list[str],
                 timeout: float = 120) -> dict[str, Any] | None:
        proc = self.run(args + ["-o", "json"], timeout=timeout, check=False)
        if proc.returncode != 0:
            if "NotFound" in proc.stderr or "not found" in proc.stderr:
                return None
            raise KubeError(
                f"kubectl {' '.join(args)} failed: {proc.stderr.strip()[:2000]}"
            )
        return json.loads(proc.stdout or "{}")


# --------------------------------------------------------------------------
# Log parsing and statistics
# --------------------------------------------------------------------------

_STEP_LINE = re.compile(
    r"completed step: (?P<step>\d+), seconds: (?P<sec>[0-9.]+)"
    r"(?:.*?TFLOP/s/device: (?P<tflops>[0-9.]+))?(?:.*?Tokens/s/device: (?P<tps>[0-9.]+))?"
    r".*?loss: (?P<loss>[0-9.naif+-]+)")


def parse_log(text: str) -> dict[str, Any]:
    metrics: dict[int, dict[str, float]] = {}
    exit_code = None
    done = False
    for line in text.splitlines():
        idx = line.find(METRIC_MARKER)
        if idx >= 0:
            try:
                rec = json.loads(line[idx + len(METRIC_MARKER):])
            except ValueError:
                continue
            step = int(rec.get("step", -1))
            # MaxText's `learning/grad_norm` is the norm *after* clipping (pinned at the
            # clip threshold once gradients exceed it); `learning/raw_grad_norm` is before.
            gn = rec.get("learning/raw_grad_norm",
                         rec.get("learning/grad_norm"))
            metrics[step] = {
                "step":
                step,
                "loss":
                rec.get("learning/loss"),
                "grad_norm":
                gn,
                "step_time":
                rec.get("perf/step_time_seconds"),
                "tflops_per_device":
                rec.get("perf/per_device_tflops_per_sec"),
                "tokens_per_sec_per_device":
                rec.get("perf/per_device_tokens_per_sec"),
                "source":
                "metrics_file",
            }
            continue
        idx = line.find(EXIT_MARKER)
        if idx >= 0:
            try:
                exit_code = int(line[idx + len(EXIT_MARKER):].split()[0])
            except (ValueError, IndexError):
                pass
            continue
        if DONE_MARKER in line:
            done = True
    if not metrics:
        # Fallback: MaxText's own per-step log line (loss has only 3 decimals).
        for m in _STEP_LINE.finditer(text):
            step = int(m.group("step"))
            metrics.setdefault(
                step, {
                    "step": step,
                    "loss": _float(m.group("loss")),
                    "grad_norm": None,
                    "step_time": _float(m.group("sec")),
                    "tflops_per_device": _float(m.group("tflops")),
                    "tokens_per_sec_per_device": _float(m.group("tps")),
                    "source": "stdout",
                })
    errors = []
    for pat in ("RESOURCE_EXHAUSTED", "Out of memory", "out of memory",
                "Ran out of memory", "FAILED_PRECONDITION", "INVALID_ARGUMENT",
                "Traceback (most recent call last)", "ValueError:",
                "AssertionError", "NaN", "Segmentation fault",
                "Aborting training"):
        if pat in text:
            errors.append(pat)
    return {
        "metrics": [metrics[k] for k in sorted(metrics)],
        "exit_code": exit_code,
        "done": done,
        "error_signatures": errors
    }


def _float(s: str | None) -> float | None:
    try:
        return float(s) if s is not None else None
    except ValueError:
        return None


def _quantile(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    if not ys:
        return float("nan")
    pos = (len(ys) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def summarize(spec: dict[str, Any],
              metrics: list[dict[str, Any]],
              *,
              warmup: int | None = None) -> dict[str, Any]:
    meas = spec["measurement"]
    warmup = meas["warmup_steps"] if warmup is None else warmup
    times = [
        m["step_time"] for m in metrics
        if m.get("step_time") is not None and m["step"] >= warmup
    ]
    tokens = spec["batch"]["global_batch_size"] * spec["batch"]["seq_len"]
    out: dict[str, Any] = {
        "steps_logged": len(metrics),
        "steady_state_steps": len(times),
        "warmup_steps_excluded": warmup,
        "tokens_per_step": tokens
    }
    if not times:
        return out
    p50 = statistics.median(times)
    mean = statistics.fmean(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    out.update({
        "step_time_p50_s":
        p50,
        "step_time_p90_s":
        _quantile(times, 0.9),
        "step_time_mean_s":
        mean,
        "step_time_std_s":
        std,
        "step_time_cv":
        std / mean if mean else float("nan"),
        "tokens_per_sec":
        tokens / p50,
        "tokens_per_sec_per_device":
        tokens / p50 / spec["hardware"]["num_devices"],
    })
    tfl = [
        m["tflops_per_device"] for m in metrics
        if m.get("tflops_per_device") and m["step"] >= warmup
    ]
    if tfl:
        out["tflops_per_device_p50"] = statistics.median(tfl)
    return out


def accuracy_gate(spec: dict[str, Any],
                  base: list[dict[str, Any]],
                  cand: list[dict[str, Any]],
                  *,
                  loss_rtol: float | None = None,
                  grad_norm_rtol: float | None = None) -> dict[str, Any]:
    """Per-step loss / grad-norm agreement over the first `gate_steps` steps.

    Every gated step needs a finite loss and a finite gradient norm in both
    runs; a missing or non-finite value fails the gate rather than skipping
    the comparison. The gradient norm only comes from MaxText's metrics file,
    so a run whose metrics were scraped from stdout cannot pass.
    """
    meas = spec["measurement"]
    n = int(meas["gate_steps"])
    loss_rtol = float(meas["loss_rtol"] if loss_rtol is None else loss_rtol)
    grad_norm_rtol = float(meas["grad_norm_rtol"] if grad_norm_rtol is
                           None else grad_norm_rtol)
    b = {m["step"]: m for m in base}
    c = {m["step"]: m for m in cand}
    rows = []
    worst_loss = 0.0
    worst_gn = 0.0
    reasons = []

    def finite(v: Any) -> bool:
        return (isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v))

    for step in range(n):
        if step not in b or step not in c:
            reasons.append(
                f"step {step} missing from {'baseline' if step not in b else 'candidate'}"
            )
            continue
        bl, cl = b[step].get("loss"), c[step].get("loss")
        bg, cg = b[step].get("grad_norm"), c[step].get("grad_norm")
        bad = [
            f"{what} {val}"
            for what, val in (("baseline loss", bl), ("candidate loss", cl),
                              ("baseline grad_norm",
                               bg), ("candidate grad_norm", cg))
            if not finite(val)
        ]
        if bad:
            reasons.append(f"step {step}: missing or not finite: " +
                           ", ".join(bad))
            continue
        dl = abs(cl - bl) / max(abs(bl), 1e-12)
        dg = abs(cg - bg) / max(abs(bg), 1e-12)
        worst_loss = max(worst_loss, dl)
        worst_gn = max(worst_gn, dg)
        rows.append({
            "step": step,
            "loss_base": bl,
            "loss_cand": cl,
            "loss_rel_diff": dl,
            "grad_norm_base": bg,
            "grad_norm_cand": cg,
            "grad_norm_rel_diff": dg
        })
    if worst_loss > loss_rtol:
        reasons.append(
            f"max loss relative diff {worst_loss:.3e} > {loss_rtol:.1e}")
    if worst_gn > grad_norm_rtol:
        reasons.append(
            f"max grad-norm relative diff {worst_gn:.3e} > {grad_norm_rtol:.1e}"
        )
    return {
        "pass": not reasons,
        "reasons": reasons,
        "gate_steps": n,
        "loss_rtol": loss_rtol,
        "grad_norm_rtol": grad_norm_rtol,
        "max_loss_rel_diff": worst_loss,
        "max_grad_norm_rel_diff": worst_gn,
        "per_step": rows
    }


# --------------------------------------------------------------------------
# Running a job end to end
# --------------------------------------------------------------------------


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_job_name(prefix: str, label: str, suffix: str = "") -> str:
    """A valid Job name of at most 40 characters; `suffix` is never cut."""
    tail = f"-{suffix}" if suffix else ""
    raw = f"{prefix}-{label}".lower()
    raw = re.sub(r"[^a-z0-9-]+", "-", raw).strip("-")
    if not K8S_NAME_RE.match(raw):
        raw = "fo-" + raw
    return raw[:40 - len(tail)].rstrip("-") + tail


def unique_suffix() -> str:
    """Time of day plus two random hex digits, so repeated runs never collide."""
    return time.strftime("%H%M%S") + os.urandom(1).hex()


def run_prefix() -> str:
    host = os.environ.get("HOSTNAME") or os.uname().nodename
    rid = re.sub(r"[^a-z0-9]", "", host.lower())[-6:] or "local"
    return f"fo-{rid}"


def _job_log(runs_dir: Path, line: str) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    p = runs_dir / "job_log.md"
    if not p.exists():
        p.write_text(
            "| time (UTC) | event | job | details |\n|---|---|---|---|\n",
            encoding="utf-8")
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _leader_pod(kube: Kube, name: str) -> dict[str, Any] | None:
    data = kube.get_json(["get", "pods", "-l", f"job-name={name}"])
    pods = (data or {}).get("items", [])
    best = None
    for p in pods:
        meta = p.get("metadata", {})
        raw = (meta.get("labels") or {}).get(COMPLETION_INDEX_KEY) or (
            meta.get("annotations") or {}).get(COMPLETION_INDEX_KEY)
        if raw in (None, "0"):
            if best is None or meta.get("creationTimestamp",
                                        "") > best["metadata"].get(
                                            "creationTimestamp", ""):
                best = p
    return best


def _pod_diagnosis(pods_json: dict[str, Any] | None) -> list[str]:
    out = []
    for p in (pods_json or {}).get("items", []):
        name = p["metadata"]["name"]
        st = p.get("status", {})
        for cs in st.get("containerStatuses", []) or []:
            term = (cs.get("state", {}).get("terminated")
                    or cs.get("lastState", {}).get("terminated"))
            if term and (term.get("exitCode")
                         or term.get("reason") not in (None, "Completed")):
                out.append(
                    f"{name}: terminated reason={term.get('reason')} exit={term.get('exitCode')}"
                )
            wait = cs.get("state", {}).get("waiting")
            if wait and wait.get("reason") not in ("ContainerCreating",
                                                   "PodInitializing"):
                out.append(
                    f"{name}: waiting reason={wait.get('reason')} {wait.get('message', '')[:200]}"
                )
        for cond in st.get("conditions", []) or []:
            if cond.get("type") == "PodScheduled" and cond.get(
                    "status") == "False":
                out.append(
                    f"{name}: unschedulable: {cond.get('message', '')[:300]}")
    return out


def run_candidate(spec: dict[str, Any],
                  cand: dict[str, Any],
                  *,
                  label: str,
                  steps: int | None = None,
                  weights: str | None = None,
                  out_dir: str | os.PathLike[str] | None = None,
                  role: str = "agent",
                  kube: Kube | None = None,
                  queue_timeout_s: int | None = None,
                  run_timeout_s: int | None = None,
                  keep: bool = False,
                  hold_sec: int = 0,
                  profile: bool = False,
                  name: str | None = None,
                  log=print) -> dict[str, Any]:
    """Submit one TPU Job for `cand`, wait for it, collect metrics, clean up."""
    resolved, xla = validate_candidate(spec, cand)
    cluster = cluster_settings(spec)
    meas = spec["measurement"]
    steps = int(steps or meas["steps"])
    weights = weights or meas.get("weights", "checkpoint")
    queue_timeout_s = int(queue_timeout_s
                          or meas.get("queue_timeout_sec", 3600))
    run_timeout_s = int(run_timeout_s or meas.get("run_timeout_sec", 5400))
    # Every run gets its own Job name and run directory: re-running a label
    # never overwrites an earlier result or collides with a Job that is still
    # terminating.
    name = name or make_job_name(run_prefix(), label, unique_suffix())
    runs_dir = Path(out_dir or DEFAULT_RUNS_DIR)
    rdir = runs_dir / name
    rdir.mkdir(parents=True, exist_ok=True)
    args = maxtext_args(spec,
                        resolved,
                        run_name=name,
                        steps=steps,
                        weights=weights,
                        cluster=cluster)
    script = pod_script(spec, args, xla, hold_sec=hold_sec, profile=profile)
    mans = job_manifests(spec, cluster, name, script, role)
    (rdir / "candidate.json").write_text(json.dumps(cand, indent=2),
                                         encoding="utf-8")
    (rdir / "job.json").write_text(json.dumps(mans["job"], indent=2),
                                   encoding="utf-8")
    (rdir / "pod_script.sh").write_text(script, encoding="utf-8")
    kube = kube or Kube(namespace=cluster["namespace"])
    result: dict[str, Any] = {
        "name": name,
        "label": label,
        "role": role,
        "steps": steps,
        "weights": weights,
        "resolved_overrides": resolved,
        "xla_flags": xla,
        "submitted_at": _now(),
        "status": "SUBMITTING",
        "cluster": {
            k: cluster[k]
            for k in ("namespace", "accelerator", "topology", "image")
        }
    }
    t_submit = time.time()
    try:
        kube.run(["apply", "-f", str(rdir / "job.json")], timeout=180)
        _job_log(
            runs_dir,
            f"| {_now()} | submitted | {name} | role={role} steps={steps} weights={weights} |"
        )
        log(f"[flagopt] submitted Job {name} ({cluster['accelerator']} {cluster['topology']}, {spec['hardware']['hosts']} host(s))"
            )
        if mans["service"] is not None:
            job = kube.get_json(["get", "job", name]) or {}
            uid = job.get("metadata", {}).get("uid", "")
            svc = mans["service"]
            if uid:
                svc["metadata"]["ownerReferences"] = [{
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": name,
                    "uid": uid
                }]
            (rdir / "service.json").write_text(json.dumps(svc, indent=2),
                                               encoding="utf-8")
            kube.run(["apply", "-f", str(rdir / "service.json")], timeout=180)
        status = _wait(kube, name, queue_timeout_s, run_timeout_s, log)
        result["status"] = status["phase"]
        result.update({k: v for k, v in status.items() if k != "phase"})
    except (KubeError, ConfigError) as exc:
        result["status"] = "ERROR"
        result["error"] = str(exc)
        log(f"[flagopt] {name}: {exc}")
    # Logs from the leader, whatever happened.
    text = ""
    try:
        pod = _leader_pod(kube, name)
        if pod is not None:
            proc = kube.run(["logs", pod["metadata"]["name"], "--tail=-1"],
                            timeout=300,
                            check=False)
            text = proc.stdout or ""
            if proc.returncode != 0 and not text:
                result.setdefault("log_error", proc.stderr.strip()[:500])
        result["pod_diagnosis"] = _pod_diagnosis(
            kube.get_json(["get", "pods", "-l", f"job-name={name}"]))
    except KubeError as exc:
        result["log_error"] = str(exc)
    (rdir / "leader.log").write_text(text, encoding="utf-8")
    parsed = parse_log(text)
    result["exit_code"] = parsed["exit_code"]
    result["done_marker"] = parsed["done"]
    result["error_signatures"] = parsed["error_signatures"]
    result["metrics"] = parsed["metrics"]
    result["summary"] = summarize(spec, parsed["metrics"])
    result["wall_time_s"] = round(time.time() - t_submit, 1)
    # A run counts only if it finished cleanly and logged every step it was
    # asked for, with a steady-state step time whenever it ran past warmup.
    logged = {m["step"] for m in parsed["metrics"]}
    missing = [s for s in range(steps) if s not in logged]
    ok = (parsed["done"] and parsed["exit_code"] == 0 and not missing
          and (steps <= int(meas["warmup_steps"])
               or "step_time_p50_s" in result["summary"]))
    if missing:
        result["missing_steps"] = missing
    result["ok"] = bool(ok)
    if not ok and result["status"] not in ("ERROR", ):
        result["status"] = "FAILED" if result["status"] in (
            "SUCCEEDED", "FAILED", "DONE") else result["status"]
    if not keep:
        for kind in ("job", "service"):
            try:
                kube.run([
                    "delete", kind, name, "--ignore-not-found", "--wait=false"
                ],
                         timeout=120,
                         check=False)
            except KubeError:
                pass
        _job_log(
            runs_dir,
            f"| {_now()} | deleted | {name} | status={result['status']} ok={result['ok']} |"
        )
    result["finished_at"] = _now()
    (rdir / "result.json").write_text(json.dumps(result, indent=2),
                                      encoding="utf-8")
    s = result["summary"]
    if "step_time_p50_s" in s:
        log(f"[flagopt] {name}: ok={result['ok']} p50 step {s['step_time_p50_s']:.4f}s "
            f"p90 {s['step_time_p90_s']:.4f}s cv {s['step_time_cv']:.3f} "
            f"tokens/s {s['tokens_per_sec']:.0f} ({len(parsed['metrics'])} steps)"
            )
    else:
        log(f"[flagopt] {name}: ok={result['ok']} status={result['status']} no step metrics; "
            f"errors={parsed['error_signatures']} diag={result.get('pod_diagnosis')}"
            )
    return result


def _wait(kube: Kube, name: str, queue_timeout_s: int, run_timeout_s: int,
          log) -> dict[str, Any]:
    """Wait for the Job: bounded while queued (suspended / pending) and while running."""
    t0 = time.time()
    started = None
    last_report = 0.0
    while True:
        job = kube.get_json(["get", "job", name])
        if job is None:
            return {
                "phase": "VANISHED",
                "detail": "Job disappeared (deleted or TTL-expired)"
            }
        st = job.get("status", {})
        spec = job.get("spec", {})
        for cond in st.get("conditions", []) or []:
            if cond.get("status") == "True" and cond.get("type") in (
                    "Complete", "SuccessCriteriaMet"):
                return {
                    "phase": "SUCCEEDED",
                    "queued_s": round((started or time.time()) - t0, 1),
                    "ran_s": round(time.time() - (started or t0), 1)
                }
            if cond.get("status") == "True" and cond.get("type") in (
                    "Failed", "FailureTarget"):
                return {
                    "phase": "FAILED",
                    "detail":
                    f"{cond.get('reason')}: {cond.get('message', '')[:300]}",
                    "queued_s": round((started or time.time()) - t0, 1),
                    "ran_s": round(time.time() - (started or t0), 1)
                }
        suspended = bool(spec.get("suspend"))
        active = int(st.get("active") or 0)
        ready = int(st.get("ready") or 0)
        if started is None and not suspended and ready > 0:
            started = time.time()
            log(f"[flagopt] {name}: running after {started - t0:.0f}s in queue"
                )
        now = time.time()
        if started is None and now - t0 > queue_timeout_s:
            return {
                "phase":
                "QUEUE_TIMEOUT",
                "detail":
                f"not running after {queue_timeout_s}s (suspended={suspended}, active={active})"
            }
        if started is not None and now - started > run_timeout_s:
            return {
                "phase": "RUN_TIMEOUT",
                "detail": f"still running after {run_timeout_s}s",
                "queued_s": round(started - t0, 1)
            }
        if now - last_report > 120:
            state = "queued (Kueue suspended)" if suspended else (
                "running"
                if started else f"pending (active={active}, ready={ready})")
            log(f"[flagopt] {name}: {state}, {now - t0:.0f}s since submit")
            last_report = now
        time.sleep(15)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _load_cand(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cmd_spec(a, spec):
    if a.json:
        print(json.dumps(spec, indent=2))
        return 0
    hw, b = spec["hardware"], spec["batch"]
    print(f"Task: {spec['task_id']}\n{spec['title']}\n")
    print(
        f"Model: {spec['model']['maxtext_model_name']} (HF {spec['model']['hf_repo']} @ {spec['model']['hf_revision'][:12]})"
    )
    print(
        f"Hardware: {hw['accelerator_display']} {hw['topology']}: {hw['chips']} chips, {hw['num_devices']} JAX devices, {hw['hosts']} host(s)"
    )
    print(
        f"Batch: global {b['global_batch_size']} x seq {b['seq_len']} = {b['global_batch_size'] * b['seq_len']} tokens/step"
    )
    print(
        f"Time budget: {spec.get('time_budget_sec') or 'open ended (session limit only)'}\n"
    )
    print("Fixed (not tunable):")
    for k, v in spec["fixed"].items():
        print(f"  {k} = {_fmt(v)}")
    print("\nTunable knobs (baseline value in brackets):")
    for k, knob in spec["tunables"].items():
        rng = knob.get("choices") or [knob.get("min"), knob.get("max")]
        base = spec["baseline"]["maxtext_overrides"].get(
            k, knob.get("default"))
        print(
            f"  {k} [{_fmt(base)}] {knob['type']} {rng}  {knob.get('help', '')}"
        )
    print(
        f"\nXLA flags tunable: {spec['xla_flags']['tunable']}; baseline: {spec['baseline'].get('xla_flags', [])}"
    )
    return 0


def _cmd_validate(a, spec):
    cand = _load_cand(a.config)
    try:
        resolved, xla = validate_candidate(spec, cand)
    except ConfigError as exc:
        print(f"INVALID: {exc}")
        return 2
    cluster = cluster_settings(spec)
    args = maxtext_args(spec,
                        resolved,
                        run_name="validate",
                        steps=spec["measurement"]["steps"],
                        weights=spec["measurement"].get(
                            "weights", "checkpoint"),
                        cluster=cluster)
    print("VALID")
    print("LIBTPU_INIT_ARGS=" + " ".join(xla))
    print("MaxText overrides:\n  " + "\n  ".join(args))
    return 0


def _cmd_render(a, spec):
    cand = _load_cand(a.config)
    resolved, xla = validate_candidate(spec, cand)
    cluster = cluster_settings(spec)
    name = make_job_name(run_prefix(), a.label)
    args = maxtext_args(spec,
                        resolved,
                        run_name=name,
                        steps=a.steps or spec["measurement"]["steps"],
                        weights=a.weights
                        or spec["measurement"].get("weights", "checkpoint"),
                        cluster=cluster)
    mans = job_manifests(
        spec, cluster, name,
        pod_script(spec, args, xla, hold_sec=a.hold_sec, profile=a.profile),
        "agent")
    print(json.dumps(mans, indent=2))
    return 0


def _cmd_run(a, spec):
    cand = baseline_candidate(spec) if a.config == "baseline" else _load_cand(
        a.config)
    res = run_candidate(spec,
                        cand,
                        label=a.label,
                        steps=a.steps,
                        weights=a.weights,
                        out_dir=a.runs_dir,
                        keep=a.keep,
                        hold_sec=a.hold_sec,
                        profile=a.profile,
                        queue_timeout_s=a.queue_timeout,
                        run_timeout_s=a.run_timeout)
    print(
        json.dumps(
            {
                k: res[k]
                for k in ("name", "status", "ok", "summary", "exit_code",
                          "error_signatures", "pod_diagnosis") if k in res
            },
            indent=2))
    print(
        f"[flagopt] details: {Path(a.runs_dir or DEFAULT_RUNS_DIR) / res['name'] / 'result.json'}"
    )
    return 0 if res["ok"] else 1


def _load_result(path: str) -> dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / "result.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _cmd_compare(a, spec):
    base, cand = _load_result(a.baseline), _load_result(a.candidate)
    gate = accuracy_gate(spec, base["metrics"], cand["metrics"])
    sb, sc = base["summary"], cand["summary"]
    out = {
        "accuracy_gate": {
            k: gate[k]
            for k in ("pass", "reasons", "max_loss_rel_diff",
                      "max_grad_norm_rel_diff")
        }
    }
    if "step_time_p50_s" in sb and "step_time_p50_s" in sc:
        sp = sb["step_time_p50_s"] / sc["step_time_p50_s"]
        out.update({
            "baseline_p50_s": sb["step_time_p50_s"],
            "candidate_p50_s": sc["step_time_p50_s"],
            "speedup": sp,
            "throughput_improvement_pct": (sp - 1) * 100
        })
    print(json.dumps(out, indent=2))
    return 0


def _cmd_logs(a, spec):
    kube = Kube(namespace=cluster_settings(spec)["namespace"])
    pod = _leader_pod(kube, a.name)
    if pod is None:
        print("no pod found for", a.name)
        return 1
    proc = kube.run(["logs", pod["metadata"]["name"], f"--tail={a.tail}"],
                    timeout=300,
                    check=False)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


def _cmd_delete(a, spec):
    kube = Kube(namespace=cluster_settings(spec)["namespace"])
    for kind in ("job", "service"):
        kube.run(
            ["delete", kind, a.name, "--ignore-not-found", "--wait=false"],
            check=False)
    _job_log(Path(a.runs_dir or DEFAULT_RUNS_DIR),
             f"| {_now()} | deleted | {a.name} | manual |")
    print("deleted", a.name)
    return 0


def _cmd_list(a, spec):
    kube = Kube(namespace=cluster_settings(spec)["namespace"])
    proc = kube.run(["get", "jobs", "-l", "app=tpu-flagopt"], check=False)
    sys.stdout.write(proc.stdout or proc.stderr)
    return 0


def _container_start() -> float | None:
    """Wall-clock start of this container's PID 1, i.e. when the session began."""
    try:
        with open("/proc/stat", encoding="ascii") as f:
            btime = next(
                int(line.split()[1]) for line in f
                if line.startswith("btime "))
        with open("/proc/1/stat", encoding="ascii") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        # Field 22 of /proc/<pid>/stat (starttime, in clock ticks since boot);
        # fields[0] here is field 3, after the parenthesised command name.
        return btime + int(fields[19]) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def session_start() -> float:
    """When the time budget started: the container start, recorded once."""
    start_file = Path(
        os.environ.get("FLAGOPT_START_FILE",
                       "/workspace/.flagopt_session_start"))
    try:
        return float(start_file.read_text().strip())
    except (OSError, ValueError):
        pass
    start = _container_start() or time.time()
    try:
        start_file.parent.mkdir(parents=True, exist_ok=True)
        start_file.write_text(str(start))
    except OSError:
        pass
    return start


def _cmd_timer(a, spec):
    budget = spec.get("time_budget_sec")
    elapsed = time.time() - session_start()
    if not budget:
        print(f"elapsed {elapsed / 60:.1f} min; no time budget set")
    else:
        print(
            f"elapsed {elapsed / 60:.1f} min of {budget / 60:.0f} min; remaining {(budget - elapsed) / 60:.1f} min"
        )
    return 0


def _cmd_example(a, spec):
    print(
        json.dumps(
            {
                "maxtext_overrides": dict(
                    spec["baseline"]["maxtext_overrides"]),
                "xla_flags": list(spec["baseline"].get("xla_flags", [])),
                "notes": "baseline configuration of this task"
            },
            indent=2))
    return 0


def _cmd_aot(a, spec):
    """Ahead-of-time compile a candidate for the task's slice on this CPU machine."""
    cand = baseline_candidate(spec) if a.config == "baseline" else _load_cand(
        a.config)
    resolved, xla = validate_candidate(spec, cand)
    cluster = cluster_settings(spec)
    args = maxtext_args(spec,
                        resolved,
                        run_name="aot",
                        steps=5,
                        weights="random",
                        cluster=cluster)
    args = [
        x for x in args
        if not x.startswith(("metrics_file=", "base_output_directory="))
    ]
    workdir = spec["maxtext"]["workdir"]
    if not Path(workdir, spec["maxtext"]["base_config"]).exists():
        print(
            f"MaxText not found at {workdir}; `flagopt aot` needs the task container's MaxText install",
            file=sys.stderr)
        return 2
    cmd = [
        "python3", "-m", "maxtext.trainers.pre_train.train_compile",
        spec["maxtext"]["base_config"],
        f"compile_topology={spec['hardware']['aot_topology']}",
        "compile_topology_num_slices=1"
    ] + args
    env = dict(os.environ, LIBTPU_INIT_ARGS=" ".join(xla), JAX_PLATFORMS="cpu")
    env.pop("PYTHONNOUSERSITE", None)
    print("[flagopt] " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd,
                          cwd=workdir,
                          env=env,
                          capture_output=True,
                          text=True,
                          timeout=a.timeout)
    text = proc.stdout + proc.stderr
    out = Path(a.log or (Path(DEFAULT_RUNS_DIR) / f"aot-{int(t0)}.log"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    m = re.search(r"Peak memory: ([0-9.]+) GiB", text)
    ok = "Finished train_compile.py successfully" in text
    hbm = spec["hardware"]["hbm_gib_per_device"]
    if ok and m:
        peak = float(m.group(1))
        verdict = "FITS" if peak <= hbm else "DOES NOT FIT"
        print(
            f"[flagopt] AOT compile OK in {time.time() - t0:.0f}s: peak {peak:.2f} GiB per device vs {hbm} GiB HBM -> {verdict}"
        )
    else:
        tail = "\n".join(text.strip().splitlines()[-25:])
        print(
            f"[flagopt] AOT compile FAILED in {time.time() - t0:.0f}s (log: {out}):\n{tail}"
        )
    print(f"[flagopt] full log: {out}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flagopt",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec",
                    default=None,
                    help="task catalog or resolved spec JSON "
                    "(default: $FLAGOPT_SPEC or /opt/flagopt/tasks.json)")
    ap.add_argument("--task",
                    default=None,
                    help="task id in the catalog (default: $FLAGOPT_TASK)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser(
        "spec", help="show the task: fixed settings, tunable knobs, baseline")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser(
        "validate",
        help="check a candidate config and print the resolved MaxText command")
    p.add_argument("config")
    p = sub.add_parser(
        "render",
        help="print the Kubernetes manifests for a candidate (no submit)")
    p.add_argument("config")
    p.add_argument("--label", default="render")
    p.add_argument("--steps", type=int)
    p.add_argument("--weights", choices=["checkpoint", "random"])
    p.add_argument("--hold-sec", type=int, default=0)
    p.add_argument("--profile", action="store_true")
    p = sub.add_parser(
        "run",
        help=
        "run a candidate config (or 'baseline') on the TPU slice and measure it"
    )
    p.add_argument("config",
                   help="candidate JSON path, or the word 'baseline'")
    p.add_argument("--label",
                   default="run",
                   help="short name used in the Job name")
    p.add_argument(
        "--steps",
        type=int,
        help="training steps (default: the task's measurement steps)")
    p.add_argument(
        "--weights",
        choices=["checkpoint", "random"],
        help=
        "load the staged HF checkpoint (default) or use random init (faster to start, same step time)"
    )
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--keep",
                   action="store_true",
                   help="do not delete the Job when done")
    p.add_argument(
        "--hold-sec",
        type=int,
        default=0,
        help=
        "keep the leader pod alive this long after training (to kubectl cp profiles)"
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help=
        "capture an XProf trace (steps 2-4) into the pod's /tmp/flagopt/out")
    p.add_argument("--queue-timeout", type=int)
    p.add_argument("--run-timeout", type=int)
    p = sub.add_parser(
        "compare", help="accuracy gate + speedup between two finished runs")
    p.add_argument("baseline",
                   help="run dir or result.json of the reference run")
    p.add_argument("candidate",
                   help="run dir or result.json of the candidate run")
    p = sub.add_parser("logs", help="print the leader pod's logs of a Job")
    p.add_argument("name")
    p.add_argument("--tail", type=int, default=200)
    p = sub.add_parser("delete", help="delete a Job (and its Service)")
    p.add_argument("name")
    p.add_argument("--runs-dir", default=None)
    sub.add_parser("list", help="list this trial's flagopt Jobs")
    sub.add_parser("timer", help="time used / remaining in the session budget")
    sub.add_parser("example",
                   help="print the baseline config in the deliverable format")
    p = sub.add_parser(
        "aot",
        help=
        "ahead-of-time compile a config for the task's slice on this CPU machine (memory check, no TPU)"
    )
    p.add_argument("config",
                   help="candidate JSON path, or the word 'baseline'")
    p.add_argument("--timeout", type=int, default=5400)
    p.add_argument("--log", default=None)
    a = ap.parse_args(argv)
    try:
        spec = load_spec(a.spec, a.task)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"flagopt: cannot load the task spec: {exc}", file=sys.stderr)
        return 2
    handler = {
        "spec": _cmd_spec,
        "validate": _cmd_validate,
        "render": _cmd_render,
        "run": _cmd_run,
        "compare": _cmd_compare,
        "logs": _cmd_logs,
        "delete": _cmd_delete,
        "list": _cmd_list,
        "timer": _cmd_timer,
        "example": _cmd_example,
        "aot": _cmd_aot
    }[a.cmd]
    try:
        return handler(a, spec)
    except ConfigError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    except KubeError as exc:
        print(f"KUBECTL ERROR: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
