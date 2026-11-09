#!/usr/bin/env -S python3 -I
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
"""`kubectl` wrapper that keeps concurrent eval trials out of each other's way.

Why this file exists
--------------------
An eval run starts several trials at once. A trial is one AI agent working on
one task in its own container. To run code on a TPU, the agent submits
Kubernetes Jobs (and the Pods and ConfigMaps that go with them) to a shared
GKE cluster. Every trial uses the same cluster, the same namespace (`default`)
and the same credentials, and per-trial namespaces are not available. So with
plain `kubectl`, an agent in one trial could list, read the logs of, exec into,
or delete another trial's Jobs, and change that trial's result.

How a `kubectl` call reaches this file
--------------------------------------
`adapter/kube_gate_setup.sh` prepares each trial container, as root, so that
this wrapper is the only way to reach the cluster:

- The agent runs as the ordinary Linux user `agent`, not as root.
- The cluster credentials (`/etc/kube-gate/kubeconfig` and
  `/etc/kube-gate/adc.json`) and the real `kubectl` binary
  (`/usr/libexec/kubectl-real`) can only be read or run by a separate system
  user, `kube-gate`.
- `/usr/local/bin/kubectl` is a two-line script that runs this file as
  `kube-gate` with `sudo -n -u kube-gate /usr/libexec/kubectl-wrapper "$@"`. A
  sudo rule allows exactly that command. sudo also clears the caller's
  environment, so the agent cannot set the `KUBE_GATE_*` variables below; the
  tests use them to point the wrapper at a fake `kubectl-real`.

The agent types `kubectl` commands as usual. This file receives the same
arguments, decides whether and how to run them, and then calls `kubectl-real`.

What this file does with each command
-------------------------------------
Each trial has an ID, written to `/etc/kube-gate/trial_id` at startup. The
wrapper marks everything the trial creates with the Kubernetes label
`eval-trial-id=<trial ID>`, and uses that label to decide what the trial may
touch:

1. Read-only cluster information (`version`, `config`, `get nodes`, Kueue
   queues, ...): run unchanged.
2. `apply`, `create`, `run`: add the label to every resource, including the
   Pod templates of Jobs and CronJobs so the Pods they start carry it too.
   Refuse if the name already belongs to another trial.
3. `get`, `delete`, `wait`, `logs`, `top` without a resource name: add
   `-l eval-trial-id=<trial ID>`, so only this trial's resources are listed or
   affected.
4. Commands that name a resource (`logs job/x`, `exec`, `attach`, `cp`,
   `port-forward`, ...): look up the resource's label first. If it belongs to
   another trial, answer `NotFound` and do not run the command.

Any other subcommand (`edit`, `replace`, `proxy`, `drain`, ...) is refused.
`get pods --all-namespaces -o json` is a special case: the agent-suite compute
driver uses it to count busy TPU chips, so it runs unfiltered and the wrapper
removes everything except node, TPU limits and phase from other trials' Pods.

See "Trial Isolation" in `eval/README.md` for the overall design, and
`eval/tests/test_kubectl_wrapper.py` for the tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

TRIAL_ID_PATH = Path(
    os.environ.get("KUBE_GATE_TRIAL_ID_FILE", "/etc/kube-gate/trial_id"))
KUBECONFIG_PATH = Path(
    os.environ.get("KUBE_GATE_KUBECONFIG", "/etc/kube-gate/kubeconfig"))
ADC_PATH = Path(os.environ.get("KUBE_GATE_ADC", "/etc/kube-gate/adc.json"))
KUBECTL_REAL = os.environ.get("KUBE_GATE_KUBECTL_REAL",
                              "/usr/libexec/kubectl-real")

TRIAL_LABEL_KEY = "eval-trial-id"

# Subcommands that read cluster-level or client-level metadata without touching
# trial workloads.
PASSTHROUGH_SUBCOMMANDS = frozenset({
    "version",
    "config",
    "auth",
    "api-resources",
    "api-versions",
    "explain",
})

# Resource kinds whose `get`/`describe` operations are read-only cluster or
# shared infrastructure metadata and pass through without trial label filtering.
METADATA_RESOURCE_KINDS = frozenset({
    "node",
    "nodes",
    "no",
    "namespace",
    "namespaces",
    "ns",
    "storageclass",
    "storageclasses",
    "sc",
    "persistentvolumeclaim",
    "persistentvolumeclaims",
    "pvc",
    "serviceaccount",
    "serviceaccounts",
    "sa",
    "localqueue",
    "localqueues",
    "localqueues.kueue.x-k8s.io",
    "lq",
    "clusterqueue",
    "clusterqueues",
    "clusterqueues.kueue.x-k8s.io",
    "cq",
    "workloadpriorityclass",
    "workloadpriorityclasses",
    "workloadpriorityclasses.kueue.x-k8s.io",
    "workload",
    "workloads",
    "workloads.kueue.x-k8s.io",
    "event",
    "events",
    "ev",
})

CREATE_SUBCOMMANDS = frozenset({"apply", "create", "run"})
LIST_OR_NAMED_SUBCOMMANDS = frozenset(
    {"get", "describe", "delete", "wait", "logs", "top"})
NAMED_ONLY_SUBCOMMANDS = frozenset(
    {"exec", "attach", "cp", "port-forward", "patch", "label", "annotate"})
# Subcommands where `-f` (`--follow`) and `-p` (`--previous`) are booleans,
# rather than `--filename` and `--patch`.
FOLLOW_SUBCOMMANDS = frozenset({"logs"})
FOLLOW_BOOLEAN_SHORT_FLAGS = frozenset("fp")

# Global or subcommand flags that consume the next argument token when not
# passed with `=`.
FLAGS_WITH_VALUE = frozenset({
    "-n",
    "--namespace",
    "-o",
    "--output",
    "-l",
    "--selector",
    "-L",
    "--label-columns",
    "-f",
    "--filename",
    "-c",
    "--container",
    "--context",
    "--cluster",
    "--user",
    "--kubeconfig",
    "--request-timeout",
    "--field-selector",
    "--sort-by",
    "--template",
    "--timeout",
    "--for",
    "--tail",
    "--since",
    "--since-time",
    "--limit",
    "--chunk-size",
    "--grace-period",
    "--cascade",
    "--wait",
    "--dry-run",
    "--field-manager",
    "--prune-allowlist",
    "--subresource",
    "--type",
    "-p",
    "--patch",
    "--patch-file",
    "--image",
    "--restart",
    "--overrides",
    "--env",
    "--port",
    "--hostport",
    "--labels",
    "--serviceaccount",
})


def _load_trial_id() -> str:
    try:
        trial_id = TRIAL_ID_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        sys.stderr.write(
            f"kubectl-wrapper: cannot read trial ID from {TRIAL_ID_PATH}: {exc}\n"
        )
        sys.exit(1)
    if not trial_id:
        sys.stderr.write(
            f"kubectl-wrapper: empty trial ID in {TRIAL_ID_PATH}\n")
        sys.exit(1)
    return trial_id


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    gate_dir = KUBECONFIG_PATH.parent
    if gate_dir.is_dir():
        env["HOME"] = str(gate_dir)
        gcloud_dir = gate_dir / "gcloud"
        try:
            gcloud_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            env["CLOUDSDK_CONFIG"] = str(gcloud_dir)
        except OSError:
            pass
    if KUBECONFIG_PATH.is_file():
        env["KUBECONFIG"] = str(KUBECONFIG_PATH)
    if ADC_PATH.is_file():
        env["GOOGLE_APPLICATION_CREDENTIALS"] = str(ADC_PATH)
        env["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"] = str(ADC_PATH)
    return env


def _strip_kubeconfig_args(argv: list[str]) -> list[str]:
    """Remove caller-supplied --kubeconfig flags so KUBECONFIG_PATH is used."""
    if not KUBECONFIG_PATH.is_file():
        return list(argv)
    out: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            out.extend(argv[i:])
            break
        if arg == "--kubeconfig":
            i += 2
            continue
        if arg.startswith("--kubeconfig="):
            i += 1
            continue
        out.append(arg)
        i += 1
    return out


# One-letter flags that take a value. kubectl (pflag) accepts the value
# attached (`-fjob.yaml`, `-ojson`), after `=`, or as the next argument.
SHORT_FLAGS_WITH_VALUE = frozenset("nolLfcpksv")

# Flags that change which server kubectl talks to, or with what identity.
# `kubectl --server=https://<agent's own server>` would send kube-gate's GKE
# token there. The agent-suite driver passes only `--kubeconfig`, which
# `_strip_kubeconfig_args` drops.
CONNECTION_FLAGS = frozenset({
    "-s",
    "--server",
    "--token",
    "--as",
    "--as-group",
    "--as-uid",
    "--user",
    "--username",
    "--password",
    "--cluster",
    "--context",
    "--certificate-authority",
    "--client-certificate",
    "--client-key",
    "--insecure-skip-tls-verify",
    "--tls-server-name",
})
# `apply -k <dir>` builds objects from a kustomization the wrapper cannot
# read, so they would be created without the trial label or an owner check.
KUSTOMIZE_FLAGS = frozenset({"-k", "--kustomize"})


def _expand_short_flags(argv: list[str]) -> list[str]:
    """Rewrite one-letter flag groups the way pflag reads them.

    `-fjob.yaml` -> `-f job.yaml`, `-ojson` -> `-o json`, `-it` -> `-i -t`,
    `-l=app=x` -> `-l app=x`. Everything after `--` is left alone. `-f` takes
    no value under `logs`, where it means `--follow`; nor does `-p`
    (`--previous`).
    """
    out: list[str] = []
    subcommand: str | None = None
    i = 0
    n = len(argv)
    while i < n:
        tok = argv[i]
        if tok == "--":
            out.extend(argv[i:])
            break
        if not tok.startswith("-") or tok == "-":
            out.append(tok)
            if subcommand is None and tok != "-":
                subcommand = tok
            i += 1
            continue
        if tok.startswith("--"):
            out.append(tok)
            # A long flag's separate value must not be read as the subcommand.
            if "=" not in tok and tok in FLAGS_WITH_VALUE and i + 1 < n:
                out.append(argv[i + 1])
                i += 1
            i += 1
            continue
        body = tok[1:]
        for j, letter in enumerate(body):
            takes_value = (letter in SHORT_FLAGS_WITH_VALUE
                           and not (letter in FOLLOW_BOOLEAN_SHORT_FLAGS
                                    and subcommand in FOLLOW_SUBCOMMANDS))
            out.append(f"-{letter}")
            if takes_value:
                rest = body[j + 1:]
                if rest.startswith("="):
                    rest = rest[1:]
                if rest:
                    out.append(rest)
                elif i + 1 < n:
                    out.append(argv[i + 1])
                    i += 1
                break
        i += 1
    return out


def _forbidden_flag(argv: list[str]) -> str | None:
    """The first connection or kustomize flag in `argv`, or None."""
    for tok in argv:
        if tok == "--":
            break
        name = tok.partition("=")[0]
        if name in CONNECTION_FLAGS or name in KUSTOMIZE_FLAGS:
            return name
    return None


class ParsedCommand:
    """Parsed representation of a kubectl invocation."""

    def __init__(self, argv: list[str]) -> None:
        self.argv = argv
        self.subcommand: str | None = None
        self.subcommand_index: int = -1
        self.positionals: list[str] = []
        self.namespace: str | None = None
        self.all_namespaces: bool = False
        self.output: str | None = None
        self.selector: str | None = None
        self.filenames: list[str] = []
        self.ignore_not_found: bool = False
        self.has_all_flag: bool = False
        # argv indices of positionals and of `-f <file>` tokens, used by
        # `_owned_only_argv` to keep every other token when rebuilding.
        self.positional_indices: list[int] = []
        self.filename_indices: list[int] = []
        self._parse()

    def _parse(self) -> None:
        i = 0
        n = len(self.argv)
        while i < n:
            token = self.argv[i]
            if token == "--":
                break
            if token.startswith("-"):
                name, has_eq, val = token.partition("=")
                if name in ("-n", "--namespace"):
                    if has_eq:
                        self.namespace = val
                    elif i + 1 < n:
                        self.namespace = self.argv[i + 1]
                        i += 1
                elif name in ("-A", "--all-namespaces"):
                    self.all_namespaces = not has_eq or val.lower() != "false"
                elif name in ("-o", "--output"):
                    if has_eq:
                        self.output = val
                    elif i + 1 < n:
                        self.output = self.argv[i + 1]
                        i += 1
                elif name in ("-l", "--selector"):
                    if has_eq:
                        self.selector = val
                    elif i + 1 < n:
                        self.selector = self.argv[i + 1]
                        i += 1
                elif (name in ("-f", "-p")
                      and self.subcommand in FOLLOW_SUBCOMMANDS):
                    pass
                elif name in ("-f", "--filename"):
                    self.filename_indices.append(i)
                    if has_eq:
                        self.filenames.append(val)
                    elif i + 1 < n:
                        self.filenames.append(self.argv[i + 1])
                        self.filename_indices.append(i + 1)
                        i += 1
                elif name == "--ignore-not-found":
                    self.ignore_not_found = not has_eq or val.lower(
                    ) != "false"
                elif name == "--all":
                    self.has_all_flag = not has_eq or val.lower() != "false"
                elif not has_eq and (name in FLAGS_WITH_VALUE or
                                     (name.startswith("--from-")
                                      and name != "--from-literal")):
                    # Handle flags like --from-file foo if passed without =
                    if name in FLAGS_WITH_VALUE and i + 1 < n:
                        i += 1
                i += 1
                continue

            if self.subcommand is None:
                self.subcommand = token
                self.subcommand_index = i
            else:
                self.positionals.append(token)
                self.positional_indices.append(i)
            i += 1


def _exec_real(argv: list[str],
               env: dict[str, str],
               stdin_bytes: bytes | None = None) -> int:
    proc = subprocess.run(
        [KUBECTL_REAL, *argv],
        input=stdin_bytes,
        env=env,
        check=False,
    )
    return proc.returncode


def _query_real(
        argv: list[str],
        env: dict[str, str],
        stdin_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [KUBECTL_REAL, *argv],
        input=stdin_text,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _ns_args(parsed: ParsedCommand, doc_ns: str | None = None) -> list[str]:
    ns = parsed.namespace or doc_ns
    return ["-n", ns] if ns else []


def _resource_label(kind: str,
                    name: str,
                    parsed: ParsedCommand,
                    env: dict[str, str],
                    doc_ns: str | None = None) -> tuple[bool, str | None]:
    """Return `(exists, trial_id_label)` for `<kind> <name>` on the cluster."""
    cmd = [
        "get",
        kind,
        name,
        *_ns_args(parsed, doc_ns),
        "-o",
        f"jsonpath={{.metadata.labels.{TRIAL_LABEL_KEY}}}",
    ]
    proc = _query_real(cmd, env)
    if proc.returncode != 0:
        return False, None
    return True, proc.stdout.strip() or None


def _is_metadata_resource(kind_token: str) -> bool:
    parts = [p.strip().lower() for p in kind_token.split(",") if p.strip()]
    return bool(parts) and all(
        p.split("/", 1)[0] in METADATA_RESOURCE_KINDS for p in parts)


def _extract_targets(parsed: ParsedCommand) -> list[tuple[str, str]]:
    """Extract `(kind, name)` targets from a non-`-f` command's positionals."""
    pos = parsed.positionals
    sub = parsed.subcommand
    if not pos:
        return []

    if sub in ("exec", "attach"):
        # `kubectl exec|attach (pod | type/name) ...`
        first = pos[0]
        if "/" in first:
            kind, name = first.split("/", 1)
            return [(kind, name)]
        return [("pod", first)]

    if sub == "cp":
        # `kubectl cp <src> <dest>`: whichever side contains `:` is `[ns/]pod:path`.
        for item in pos[:2]:
            if ":" in item and not item.startswith(("/", ".", "~")):
                pod_spec = item.split(":", 1)[0]
                if "/" in pod_spec:
                    # In `kubectl cp namespace/pod:/path`, the prefix is namespace.
                    _, pod_name = pod_spec.rsplit("/", 1)
                else:
                    pod_name = pod_spec
                return [("pod", pod_name)]
        return []

    if sub == "logs":
        # `kubectl logs [-f] (pod | type/name | type name) [container]`
        first = pos[0]
        if "/" in first:
            kind, name = first.split("/", 1)
            return [(kind, name)]
        if len(pos) >= 2 and first.lower() in ("pod", "pods", "po", "job",
                                               "jobs", "deployment",
                                               "deployments", "deploy",
                                               "service", "services", "svc"):
            return [(first, pos[1])]
        return [("pod", first)]

    if sub == "port-forward":
        # `kubectl port-forward (pod | type/name | type name) [LOCAL:]REMOTE`
        first = pos[0]
        if "/" in first:
            kind, name = first.split("/", 1)
            return [(kind, name)]
        if len(pos) >= 2 and ":" not in pos[1] and not pos[1].isdigit():
            return [(first, pos[1])]
        return [("pod", first)]

    # Standard `<kind> [<name>...]` or `<kind>/<name>...` syntax (get, describe,
    # delete, wait, top, patch, label, annotate).
    first = pos[0]
    if "/" in first:
        targets: list[tuple[str, str]] = []
        for token in pos:
            if "/" in token and "=" not in token and not token.endswith("-"):
                kind, name = token.split("/", 1)
                targets.append((kind, name))
        return targets

    if len(pos) == 1:
        return []

    kinds = [k.strip() for k in first.split(",") if k.strip()]
    names: list[str] = []
    for token in pos[1:]:
        # Stop at label/annotation key=val or key- args on `label` / `annotate`.
        if sub in ("label", "annotate") and ("=" in token
                                             or token.endswith("-")):
            break
        if "/" in token:
            k, n = token.split("/", 1)
            kinds = [k]
            names.append(n)
        else:
            names.append(token)
    return [(k, n) for n in names for k in kinds]


def _child_dict(parent: Any, key: str) -> dict[str, Any] | None:
    """`parent[key]` if both are dicts, else None."""
    if not isinstance(parent, dict):
        return None
    child = parent.get(key)
    return child if isinstance(child, dict) else None


def _set_trial_label(obj: dict[str, Any], trial_id: str) -> None:
    """Set `obj.metadata.labels[TRIAL_LABEL_KEY]`.

    An empty `metadata:` or `labels:` in YAML loads as None, so a missing or
    non-dict value is replaced with `{}` rather than kept via `setdefault`.
    """
    meta = obj.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
        obj["metadata"] = meta
    labels = meta.get("labels")
    if not isinstance(labels, dict):
        labels = {}
        meta["labels"] = labels
    labels[TRIAL_LABEL_KEY] = trial_id


def _inject_label_into_doc(doc: dict[str, Any], trial_id: str) -> None:
    _set_trial_label(doc, trial_id)

    spec = _child_dict(doc, "spec")
    # Job / Deployment: pods are created from `spec.template`.
    template = _child_dict(spec, "template")
    if template is not None:
        _set_trial_label(template, trial_id)
    # CronJob: Jobs are created from `spec.jobTemplate`, and their pods from
    # `spec.jobTemplate.spec.template`.
    job_template = _child_dict(spec, "jobTemplate")
    if job_template is not None:
        _set_trial_label(job_template, trial_id)
        job_pod_template = _child_dict(_child_dict(job_template, "spec"),
                                       "template")
        if job_pod_template is not None:
            _set_trial_label(job_pod_template, trial_id)


def _check_manifest_conflicts(docs: list[dict[str, Any]],
                              parsed: ParsedCommand, trial_id: str,
                              env: dict[str, str]) -> int | None:
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        kind = doc.get("kind")
        meta = doc.get("metadata") or {}
        name = meta.get("name")
        doc_ns = meta.get("namespace")
        if not kind or not name:
            continue
        exists, owner = _resource_label(kind, name, parsed, env, doc_ns=doc_ns)
        if exists and owner != trial_id:
            sys.stderr.write(
                f'Error from server (AlreadyExists): {kind.lower()}s "{name}" '
                f"already exists and belongs to another trial\n")
            return 1
    return None


def _replace_filenames_with_stdin(argv: list[str]) -> list[str]:
    """Rewrite `-f <path>` / `--filename=<path>` flags to a single `-f -`."""
    out: list[str] = []
    replaced = False
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in ("-f", "--filename"):
            if not replaced:
                out.extend(["-f", "-"])
                replaced = True
            i += 2
            continue
        if token.startswith("-f=") or token.startswith("--filename="):
            if not replaced:
                out.extend(["-f", "-"])
                replaced = True
            i += 1
            continue
        out.append(token)
        i += 1
    return out


def _load_manifest_docs(
        filenames: list[str]) -> tuple[list[dict[str, Any]], str | None]:
    """Read the objects in `-f` manifests, unpacking `kind: List`.

    Returns `(docs, stdin_text)`. stdin can be read only once, so the text read
    for `-f -` is returned for the caller to pass on to `kubectl-real`.
    Raises ValueError with a kubectl-style message if a file cannot be read or
    parsed.
    """
    docs: list[dict[str, Any]] = []
    stdin_text: str | None = None
    for fname in filenames:
        if fname == "-":
            if stdin_text is None:
                stdin_text = sys.stdin.read()
            raw = stdin_text
        else:
            try:
                raw = Path(fname).read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(
                    f"error: unable to read {fname}: {exc}") from exc
        try:
            loaded = list(yaml.safe_load_all(raw))
        except yaml.YAMLError as exc:
            raise ValueError(f"error: error parsing {fname}: {exc}") from exc
        for item in loaded:
            if not isinstance(item, dict):
                continue
            if item.get("kind") == "List" and isinstance(
                    item.get("items"), list):
                docs.extend(i for i in item["items"] if isinstance(i, dict))
            else:
                docs.append(item)
    return docs, stdin_text


def _handle_create_or_apply(parsed: ParsedCommand, trial_id: str,
                            env: dict[str, str]) -> int:
    if parsed.filenames:
        try:
            docs, _ = _load_manifest_docs(parsed.filenames)
        except ValueError as exc:
            sys.stderr.write(f"{exc}\n")
            return 1

        conflict_rc = _check_manifest_conflicts(docs, parsed, trial_id, env)
        if conflict_rc is not None:
            return conflict_rc

        for doc in docs:
            _inject_label_into_doc(doc, trial_id)

        payload = "\n---\n".join(json.dumps(doc) for doc in docs) + "\n"
        new_argv = _replace_filenames_with_stdin(parsed.argv)
        return _exec_real(new_argv, env, stdin_bytes=payload.encode("utf-8"))

    # Imperative `kubectl run <name> ...` or `kubectl create <kind> <name> ...`
    if parsed.subcommand == "run":
        if not parsed.positionals:
            return _exec_real(parsed.argv, env)
        pod_name = parsed.positionals[0]
        exists, owner = _resource_label("pod", pod_name, parsed, env)
        if exists and owner != trial_id:
            sys.stderr.write(
                f'Error from server (AlreadyExists): pods "{pod_name}" '
                f"already exists and belongs to another trial\n")
            return 1
        return _exec_with_injected_selector_or_labels(parsed,
                                                      trial_id,
                                                      env,
                                                      flag="-l")

    if parsed.subcommand == "create" and len(parsed.positionals) >= 2:
        kind = parsed.positionals[0]
        name = parsed.positionals[1]
        if kind.lower() in ("namespace", "ns", "clusterrole",
                            "clusterrolebinding", "role", "rolebinding"):
            sys.stderr.write(
                f"Error from server (Forbidden): creating {kind} is not "
                "permitted in an isolated eval trial\n")
            return 1
        exists, owner = _resource_label(kind, name, parsed, env)
        if exists and owner != trial_id:
            sys.stderr.write(
                f'Error from server (AlreadyExists): {kind} "{name}" '
                f"already exists and belongs to another trial\n")
            return 1
        # Use `--dry-run=client -o json` to materialize the manifest, inject
        # `eval-trial-id`, and pipe to `kubectl-real create -f -`. The flags go
        # before any `--`: after it they are arguments to the container command
        # (`create job x --image=busybox -- sleep 10`), and kubectl would
        # create the object for real, unlabelled.
        cut = (parsed.argv.index("--")
               if "--" in parsed.argv else len(parsed.argv))
        own_args = parsed.argv[:cut]
        dry_argv = [
            *own_args, "--dry-run=client", "-o", "json", *parsed.argv[cut:]
        ]
        dry_proc = _query_real(dry_argv, env)
        if dry_proc.returncode != 0:
            if dry_proc.stdout:
                sys.stdout.write(dry_proc.stdout)
            if dry_proc.stderr:
                sys.stderr.write(dry_proc.stderr)
            return dry_proc.returncode
        try:
            doc = json.loads(dry_proc.stdout)
        except ValueError:
            # Running the original command would create it without the label.
            sys.stderr.write(
                "error: could not add the trial label: `kubectl create "
                "--dry-run=client -o json` did not print JSON\n")
            return 1
        _inject_label_into_doc(doc, trial_id)
        create_argv = ["create", *_ns_args(parsed), "-f", "-"]
        # Keep the caller's `--dry-run`: `create configmap ... --dry-run=client
        # -o yaml | kubectl apply -f -` is how agents update a ConfigMap, and
        # without it this step would create the object on the server.
        create_argv.extend(
            tok for tok in own_args
            if tok == "--dry-run" or tok.startswith("--dry-run="))
        if parsed.output:
            create_argv.extend(["-o", parsed.output])
        return _exec_real(
            create_argv,
            env,
            stdin_bytes=(json.dumps(doc) + "\n").encode("utf-8"),
        )

    return _exec_real(parsed.argv, env)


def _exec_with_injected_selector_or_labels(parsed: ParsedCommand,
                                           trial_id: str,
                                           env: dict[str, str],
                                           flag: str = "-l") -> int:
    label_expr = f"{TRIAL_LABEL_KEY}={trial_id}"
    out: list[str] = []
    merged = False
    i = 0
    argv = parsed.argv
    while i < len(argv):
        token = argv[i]
        if token == "--":
            if not merged:
                out.extend([flag, label_expr])
                merged = True
            out.extend(argv[i:])
            break
        if parsed.subcommand in ("delete", "wait") and (token == "--all" or
                                                        token == "--all=true"):
            # `kubectl delete|wait <kind> --all -l ...` is rejected by kubectl
            # because `--all` and `-l` are mutually exclusive; `-l` alone
            # selects all matching resources.
            i += 1
            continue
        if token in ("-l", "--selector", "--labels"):
            if i + 1 < len(argv):
                existing = argv[i + 1]
                out.extend([token, f"{existing},{label_expr}"])
                merged = True
                i += 2
                continue
        if (token.startswith("-l=") or token.startswith("--selector=")
                or token.startswith("--labels=")):
            prefix, _, existing = token.partition("=")
            out.append(f"{prefix}={existing},{label_expr}")
            merged = True
            i += 1
            continue
        out.append(token)
        i += 1
    if not merged:
        out.extend([flag, label_expr])
    return _exec_real(out, env)


def _is_all_namespaces_pods_json(parsed: ParsedCommand) -> bool:
    """True when the command is `kubectl get pods --all-namespaces -o json`
    without a selector or target name (used by `gke_job.status.chips_in_use`)."""
    if parsed.subcommand != "get" or not parsed.all_namespaces:
        return False
    if parsed.selector is not None or len(parsed.positionals) != 1:
        return False
    if parsed.positionals[0].lower() not in ("pod", "pods", "po"):
        return False
    return (parsed.output or "").lower() == "json"


def _handle_all_namespaces_pods_json(parsed: ParsedCommand, trial_id: str,
                                     env: dict[str, str]) -> int:
    """Run `get pods --all-namespaces -o json` and redact non-trial pods so
    `compute inspect` counts busy TPU chips across the cluster without leaking
    other trials' pod names, commands, or env vars."""
    proc = _query_real(parsed.argv, env)
    if proc.returncode != 0:
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        return proc.returncode
    try:
        payload = json.loads(proc.stdout)
    except ValueError:
        sys.stdout.write(proc.stdout)
        return 0

    items = payload.get("items")
    if isinstance(items, list):
        redacted_items: list[dict[str, Any]] = []
        for pod in items:
            if not isinstance(pod, dict):
                continue
            meta = pod.get("metadata") or {}
            labels = meta.get("labels") or {}
            if labels.get(TRIAL_LABEL_KEY) == trial_id:
                redacted_items.append(pod)
                continue
            spec = pod.get("spec") or {}
            status = pod.get("status") or {}
            containers = [{
                "resources": {
                    "limits": (c.get("resources") or {}).get("limits") or {},
                }
            } for c in (spec.get("containers") or []) if isinstance(c, dict)]
            redacted_items.append({
                "metadata": {
                    "namespace": meta.get("namespace", "default"),
                    "name": "redacted",
                },
                "spec": {
                    "nodeName": spec.get("nodeName"),
                    "containers": containers,
                },
                "status": {
                    "phase": status.get("phase", ""),
                },
            })
        payload["items"] = redacted_items

    sys.stdout.write(json.dumps(payload))
    return 0


def _singular_kind(kind: str) -> str:
    k = kind.lower()
    aliases = {
        "po": "pod",
        "pods": "pod",
        "jobs": "job",
        "cm": "configmap",
        "configmaps": "configmap",
        "svc": "service",
        "services": "service",
        "deploy": "deployment",
        "deployments": "deployment",
    }
    return aliases.get(k, k.rstrip("s") if k.endswith("s") else k)


def _owned_only_argv(parsed: ParsedCommand,
                     targets: list[tuple[str, str]]) -> list[str]:
    """Rebuild `parsed.argv` naming only `targets`, as `<kind>/<name>`.

    Drops the original positionals and `-f <file>` tokens and keeps every
    other token (flags, `--ignore-not-found`, `-o`, ...) in place.
    """
    dropped = set(parsed.positional_indices) | set(parsed.filename_indices)
    rest = [
        tok for idx, tok in enumerate(parsed.argv)
        if idx != parsed.subcommand_index and idx not in dropped
    ]
    names = [f"{kind.lower()}/{name}" for kind, name in targets]
    return [str(parsed.subcommand), *names, *rest]


def _handle_named_targets(parsed: ParsedCommand,
                          targets: list[tuple[str, str]],
                          trial_id: str,
                          env: dict[str, str],
                          stdin_text: str | None = None,
                          target_ns: list[str | None] | None = None) -> int:
    """Forward the command only for targets labelled with `trial_id`.

    `stdin_text` is the manifest the caller already read from `-f -`; stdin is
    empty by now, so it is passed to `kubectl-real` explicitly. `target_ns`
    holds each `-f` object's `metadata.namespace`, used where `-n` is absent.
    """
    namespaces = target_ns or [None] * len(targets)
    allowed: list[tuple[str, str]] = []
    allowed_ns: set[str | None] = set()
    # `get --ignore-not-found -o name` is how `GkeJobDriver._name_unused`
    # checks whether a name is free. A name held by another trial is reported
    # as taken (only the name, nothing else about it); otherwise the driver
    # picks it and the create is rejected with AlreadyExists.
    report_taken = (parsed.subcommand == "get" and parsed.ignore_not_found
                    and (parsed.output or "").lower() == "name")
    taken: list[str] = []
    for (kind, name), doc_ns in zip(targets, namespaces):
        if _is_metadata_resource(kind) and parsed.subcommand in ("get",
                                                                 "describe"):
            allowed.append((kind, name))
            allowed_ns.add(doc_ns)
            continue
        exists, owner = _resource_label(kind, name, parsed, env, doc_ns=doc_ns)
        if exists and owner == trial_id:
            allowed.append((kind, name))
            allowed_ns.add(doc_ns)
            continue
        if parsed.ignore_not_found and parsed.subcommand in ("get", "delete"):
            if exists and report_taken:
                taken.append(f"{_singular_kind(kind)}/{name}")
            continue
        singular = _singular_kind(kind)
        sys.stderr.write(
            f'Error from server (NotFound): {singular}s "{name}" not found\n')
        return 1

    if len(allowed) == len(targets):
        stdin_bytes = (stdin_text.encode("utf-8")
                       if stdin_text is not None else None)
        return _exec_real(parsed.argv, env, stdin_bytes=stdin_bytes)

    # Only reachable with `--ignore-not-found` on get/delete: some targets
    # belong to another trial or do not exist. Forwarding the original argv
    # would act on them too, so name only the allowed targets.
    if taken:
        sys.stdout.write("".join(f"{line}\n" for line in taken))
        # `kubectl-real` below writes to the same stdout file descriptor.
        sys.stdout.flush()
    if not allowed:
        return 0
    owned_argv = _owned_only_argv(parsed, allowed)
    if parsed.namespace is None:
        # `<kind>/<name>` arguments cannot carry the manifest's namespace.
        doc_namespaces = allowed_ns - {None}
        if len(doc_namespaces) > 1:
            sys.stderr.write(
                "error: objects from several namespaces in one -f with "
                "--ignore-not-found; pass them separately\n")
            return 1
        if doc_namespaces:
            owned_argv += ["-n", doc_namespaces.pop()]
    return _exec_real(owned_argv, env)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    env = _child_env()
    clean_argv = _expand_short_flags(_strip_kubeconfig_args(raw_argv))
    forbidden = _forbidden_flag(clean_argv)
    if forbidden:
        sys.stderr.write(
            f"Error from server (Forbidden): kubectl flag '{forbidden}' is not "
            "permitted in an isolated eval trial\n")
        return 1
    parsed = ParsedCommand(clean_argv)
    # Only kubectl's own arguments: `exec pod -- ls -h` is not a help request.
    own_args = (clean_argv[:clean_argv.index("--")]
                if "--" in clean_argv else clean_argv)

    # Bare `kubectl`, `kubectl --help` / `-h`, or `kubectl version --client`
    # (used by CI image verification before `/etc/kube-gate/trial_id` exists).
    if (parsed.subcommand is None or "-h" in own_args or "--help" in own_args
            or (parsed.subcommand == "version" and
                ("--client" in own_args or "--client=true" in own_args))):
        return _exec_real(clean_argv, env)

    trial_id = _load_trial_id()
    sub = parsed.subcommand

    # Rule 1: Read-only cluster metadata
    if sub in PASSTHROUGH_SUBCOMMANDS:
        return _exec_real(clean_argv, env)

    if sub in ("get",
               "describe") and parsed.positionals and _is_metadata_resource(
                   parsed.positionals[0]):
        return _exec_real(clean_argv, env)

    # Rule 2: Create / Apply
    if sub in CREATE_SUBCOMMANDS:
        return _handle_create_or_apply(parsed, trial_id, env)

    # Rule 3 & Rule 4: List / Bulk vs. Named Resource Operations
    if sub in LIST_OR_NAMED_SUBCOMMANDS or sub in NAMED_ONLY_SUBCOMMANDS:
        if parsed.filenames:
            # e.g. `kubectl delete -f manifest.yaml` or `kubectl get -f ...`
            try:
                docs, stdin_text = _load_manifest_docs(parsed.filenames)
            except ValueError as exc:
                sys.stderr.write(f"{exc}\n")
                return 1
            targets: list[tuple[str, str]] = []
            target_ns: list[str | None] = []
            for d in docs:
                meta = _child_dict(d, "metadata") or {}
                kind = d.get("kind")
                name = meta.get("name")
                if not kind or not name:
                    # Its owner cannot be checked, and forwarding the file
                    # would let `kubectl-real` act on it unchecked.
                    sys.stderr.write(
                        "error: every object passed with -f needs a kind and "
                        "metadata.name in an isolated eval trial\n")
                    return 1
                targets.append((str(kind), str(name)))
                target_ns.append(
                    str(meta["namespace"]) if meta.get("namespace") else None)
            return _handle_named_targets(parsed,
                                         targets,
                                         trial_id,
                                         env,
                                         stdin_text=stdin_text,
                                         target_ns=target_ns)

        targets = _extract_targets(parsed)
        if targets:
            return _handle_named_targets(parsed, targets, trial_id, env)

        if sub in NAMED_ONLY_SUBCOMMANDS:
            sys.stderr.write(f"error: {sub} requires a target resource name\n")
            return 1

        # Special case for `compute inspect` (`gke_job.status.chips_in_use`):
        if _is_all_namespaces_pods_json(parsed):
            return _handle_all_namespaces_pods_json(parsed, trial_id, env)

        # List / bulk operation without a specific resource name:
        # `kubectl delete jobs --all` or `kubectl get pods` -> scope by `-l`.
        return _exec_with_injected_selector_or_labels(parsed,
                                                      trial_id,
                                                      env,
                                                      flag="-l")

    sys.stderr.write(
        f"Error from server (Forbidden): kubectl subcommand '{sub}' is not "
        "permitted in an isolated eval trial\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
