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
"""Abstract base class for compute drivers."""

from __future__ import annotations

import abc
import argparse
import http.client
import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import time
import urllib.request
from typing import Any, NamedTuple

from framework import paths

from . import jobs, probe
# Driver modules import this module as `base` and reach the contracts and the
# error type through it (`base.HardwareProfile`, `base.ComputeConfigError`), so
# one import line gives a driver everything the interface deals in.
from .contracts import PENDING_ADMITTED_NO_NODE as PENDING_ADMITTED_NO_NODE
from .contracts import PENDING_QUEUED as PENDING_QUEUED
from .contracts import PENDING_REASONS as PENDING_REASONS
from .contracts import PENDING_STARTING as PENDING_STARTING
from .contracts import PENDING_UNSCHEDULED as PENDING_UNSCHEDULED
from .contracts import (AsynchronousExecutionResult, HardwareProfile,
                        PreflightCheck, PreflightReport,
                        SynchronousExecutionResult)
from .errors import ComputeConfigError, NotConfigured
from .registry import record_cli_flag
from .setup_fields import SetupFields

Path = pathlib.Path

# How long a backgrounded port forward may take to bind its local port before
# it is called a failure
READY_TIMEOUT_SEC = 30.0
READY_POLL_SEC = 0.1


class StagedFile:
    """One file carried across the local/target boundary by `run`.

    `local` is a path on this host; `remote` a path relative to the script's
    working directory on the target. Parsed from the CLI's `LOCAL[:REMOTE]`
    (inputs) and `REMOTE[:LOCAL]` (outputs), where the omitted side defaults to
    the same relative spelling as the given one.
    """

    __slots__ = ("local", "remote")

    def __init__(self, local: str, remote: str) -> None:
        self.local = local
        self.remote = remote

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StagedFile(local={self.local!r}, remote={self.remote!r})"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, StagedFile) and self.local == other.local
                and self.remote == other.remote)

    @staticmethod
    def as_records(files: list[StagedFile]) -> list[dict[str, str]]:
        """Declared outputs in the shape a `JobRecord`'s `extra` stores them."""
        return [{"local": f.local, "remote": f.remote} for f in files]

    @staticmethod
    def artifact_path(dest_dir: Path, record: dict[str, str]) -> Path:
        """Where a declared output of `submit` lands under `dest_dir`."""
        target = Path(dest_dir) / record["remote"]
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @classmethod
    def parse_input(cls, spec: str) -> StagedFile:
        """`LOCAL[:REMOTE]` -> StagedFile; REMOTE defaults to LOCAL's basename."""
        local, sep, remote = spec.partition(":")
        if not local:
            raise ComputeConfigError(
                f"--input {spec!r}: a local path is required.",
                error_code="INVALID_FIELD")
        remote = remote if sep and remote else pathlib.PurePosixPath(
            local).name
        return cls(local=os.path.expanduser(local),
                   remote=_relative_remote(remote, "--input"))

    @classmethod
    def parse_output(cls, spec: str) -> StagedFile:
        """`REMOTE[:LOCAL]` -> StagedFile; LOCAL defaults to REMOTE's spelling."""
        remote, sep, local = spec.partition(":")
        if not remote:
            raise ComputeConfigError(
                f"--output {spec!r}: a path on the target is required.",
                error_code="INVALID_FIELD")
        return cls(
            local=os.path.expanduser(local) if sep and local else remote,
            remote=_relative_remote(remote, "--output"))


def _relative_remote(remote: str, flag: str) -> str:
    """Reject target paths that escape the script's working directory."""
    posix = pathlib.PurePosixPath(remote)
    if posix.is_absolute() or ".." in posix.parts:
        raise ComputeConfigError(
            f"{flag} {remote!r}: the target-side path must be relative to the "
            f"script's working directory and may not contain '..'.",
            error_code="INVALID_FIELD")
    return posix.as_posix()


class ComputeDriver(abc.ABC):
    """Abstract interface implemented by all compute backends.

    A driver is an implementation of what a backend must supply: how to describe the
    hardware (`inspect`), how to run something synchronously (`run`), and how to run
    something asynchronously and follow it afterwards (`submit`,
    `get_status`, `fetch_logs`, `fetch_artifacts`, `cancel`).

    Two class-level hooks keep that true:
    - `from_params` owns the backend's own validation, so the resolver can hand
      any driver a merged parameter dict without knowing what it needs.
    - `register_cli_args` declares the backend's flags, so the shared parser
      mounts them by asking the registered drivers rather than hardcoding a
      list that drifts as backends come and go.
    """

    backend_name: str = ""
    SETUP_FIELDS: SetupFields = SetupFields()
    # Command-line tools the backend shells out to. `preflight` checks them
    # before anything that would need them.
    required_tools: tuple[str, ...] = ()
    # Whether `run` takes seconds rather than scheduling a workload, so
    # `compute setup verify` can run the software check on the target.
    quick_run: bool = False
    # Whether `run` works but schedules a workload, which can take minutes
    # (a node scaling up, an image pulling), so `compute setup verify` runs
    # the software check through it as a smoke test only when asked.
    scheduled_run: bool = False
    # Whether workloads run in a container image, so software installed into
    # a running container is lost: the software check's fixes name the image.
    runs_in_container: bool = False
    # The profile settings to ask about again when a preflight check fails,
    # by check name; `preflight` copies them into `PreflightCheck.fields`.
    CHECK_FIELDS: dict[str, tuple[str, ...]] = {}
    # The environment.json profile this driver was built from, if any. The
    # resolver sets it, and `submit` records it on the job.
    compute_profile: str | None = None
    # The run this command belongs to, if any. The resolver sets it, so a
    # driver whose `run` schedules a job tags that job as `submit` would,
    # and the run-end reaper finds it if the run is interrupted.
    run_id: str | None = None
    # The Python a recipe runs under on the target when the command names
    # none. Drivers that run recipes through `_runner_for` set it.
    python_interpreter: str | None = None

    @classmethod
    def suggest_backend(cls, found: dict[str, dict[str, Any]]) -> str | None:
        """Return a probe-source string when `found` points to this backend."""
        del found
        return None

    @classmethod
    def populate_default_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        """Return `params` with any backend-inferred defaults filled in."""
        return dict(params)

    @classmethod
    def collect_environment_answers(
            cls, env_lookup: dict[str, str]) -> dict[str, Any]:
        """Return profile answers inferred from `env_lookup` for this backend."""
        del env_lookup
        return {}

    def not_configured(self, field: str, detail: str = "") -> NotConfigured:
        """The error for a setting that's asked only when needed, and is unset.

        The error code is named after the group of checks the setting
        belongs to in CHECK_FIELDS: `staging_bucket` is in `staging:bucket`,
        so its code is STAGING_NOT_CONFIGURED. A setting in no group gets a
        code named after itself. The message says why the setting is needed,
        then adds `detail` about this command.
        """
        group = next(
            (g for g, fields in self.CHECK_FIELDS.items() if field in fields),
            field)
        reason = self.SETUP_FIELDS.on_demand[field]
        return NotConfigured(
            f"'{field}' is not set; it is needed for {reason}. {detail}".strip(
            ), f"{group.split(':')[0].upper()}_NOT_CONFIGURED", [field])

    def _new_record(self, **fields: Any) -> jobs.JobRecord:
        """Create a job record that names this driver's compute profile."""
        return jobs.JobRecord.create(compute_profile=self.compute_profile,
                                     **fields)

    def _runner_for(self, path: Path, python_interpreter: str | None) -> str:
        """Pick the program that executes a recipe, by file extension.

        A shell recipe gets the configured Python as `TPU_AGENT_SUITE_PYTHON`, which
        is how it finds tools installed beside it (e.g. a venv's `vllm`).
        Unquoted, like the Python runner, so `~` expands on the target.
        """
        python = python_interpreter or self.python_interpreter or "python3"
        if path.suffix == ".sh":
            return f"TPU_AGENT_SUITE_PYTHON={python} bash"
        return python

    @classmethod
    @abc.abstractmethod
    def from_params(cls, params: dict[str, Any]) -> ComputeDriver:
        """Validate required parameters and construct the driver instance.

        This is the single entry point the config resolver uses. It exists so
        the resolver can stay backend-agnostic: it merges environment.json with
        CLI overrides into one untyped dict and hands it over.

        Args:
            params: Parameters merged from environment.json and CLI overrides.

        Returns:
            Configured ComputeDriver instance.

        Raises:
            ComputeConfigError: If required backend parameters are missing or invalid.
        """
        raise NotImplementedError

    @classmethod
    def register_cli_args(cls, parser: Any) -> None:
        """Contribute backend-specific CLI flags to the shared compute argument parser.

        Override this method to expose backend-specific options (e.g., `--zone`, `--project`).
        Use `cls.add_cli_arg(parser, ...)` to register flags safely and record their names
        for automatic extraction into driver `param_overrides`.
        """
        pass

    @classmethod
    def add_cli_arg(cls, parser: Any, *args: Any, **kwargs: Any) -> None:
        """Safely add a CLI flag if not already registered, and record its attribute name.

        Deduplicates shared flags across drivers (such as `--namespace` or
        `--kube-context` used by both GKE pods and GKE Jobs) and records the
        destination parameter into `REGISTERED_DRIVER_CLI_FLAGS` for whitelist
        parameter extraction.

        Flags are always registered with `default=argparse.SUPPRESS`, and any `default`
        a caller supplies is ignored. An argparse default would land in the namespace on
        every invocation, and the CLI forwards every registered flag it finds into
        `param_overrides`, which take precedence over `environment.json` -- so a default
        here would silently override configured values. Put backend defaults in
        `from_params` instead, where they apply only when nothing else supplied a value.

        A flag whose destination is in `SETUP_FIELDS.required` is recorded as
        required for this backend, so generated docs can tell required options
        from optional ones. It never makes argparse enforce the flag -- the
        value may legitimately come from environment.json.

        Raises:
            ValueError: If the flag's destination is reserved by the runtime.
        """
        dest = kwargs.get("dest")
        if not dest:
            for opt in args:
                if opt.startswith("--"):
                    dest = opt[2:].replace("-", "_")
                    break
        if dest:
            record_cli_flag(dest, tuple(args), cls.backend_name, dest
                            in cls.SETUP_FIELDS.required)

        kwargs["default"] = argparse.SUPPRESS

        option_strings = getattr(parser, "_option_string_actions", {})
        for opt in args:
            if opt in option_strings:
                return
        parser.add_argument(*args, **kwargs)

    @abc.abstractmethod
    def inspect(self) -> HardwareProfile:
        """Probe accelerator generation, chip count, topology, and health."""

    @abc.abstractmethod
    def run(
        self,
        code_path: str,
        timeout: int | None = None,
        args: list[str] | None = None,
        python_interpreter: str | None = None,
        inputs: list[StagedFile] | None = None,
        outputs: list[StagedFile] | None = None,
        job_name: str | None = None,
    ) -> SynchronousExecutionResult:
        """Synchronously execute code or a script on the target accelerator.

        `inputs` are files placed next to the script before it starts and
        `outputs` files copied back once it exits; both are addressed on the
        target by a path relative to the script's working directory. A driver
        whose target is this host still honours them, so one command spelling
        works on every backend. Without either, the script runs alone in the
        caller's working directory, as before.

        `job_name` names whatever the driver creates for the run, e.g. a
        Kubernetes Job. Drivers that create nothing to name ignore it.
        """

    @abc.abstractmethod
    def submit(
        self,
        recipe_path: str,
        job_name: str | None = None,
        args: list[str] | None = None,
        python_interpreter: str | None = None,
        run_id: str | None = None,
        inputs: list[StagedFile] | None = None,
        outputs: list[StagedFile] | None = None,
        serve: bool = False,
    ) -> str:
        """Dispatch an asynchronous workload; returns a durable job_id."""

    @abc.abstractmethod
    def get_status(self, job_id: str) -> AsynchronousExecutionResult:
        """Query status of a submitted job."""

    @abc.abstractmethod
    def fetch_logs(self, job_id: str) -> str:
        """Retrieve stdout/stderr logs of a submitted job."""

    @abc.abstractmethod
    def fetch_artifacts(self, job_id: str, dest_dir: Path) -> list[Path]:
        """Download output artifacts from a submitted job into dest_dir."""

    @abc.abstractmethod
    def cancel(self, job_id: str) -> None:
        """Terminate a running or pending job."""

    def describe_target(self) -> str:
        """Where this backend runs work, for a person to recognise and confirm."""
        return f"backend '{self.backend_name}'"

    def needed_tools(self) -> tuple[str, ...]:
        """The required tools this setup actually calls."""
        return self.required_tools

    def preflight(self) -> PreflightReport:
        """Check the tools, credentials and target this backend needs.

        Quick and read-only: it changes nothing. The software installed on
        the target is `env_check.py`'s job, not this.
        """
        checks = [tool_check(tool) for tool in self.needed_tools()]
        reachable = (all(check.status == "ok" for check in checks)
                     and self._preflight_target(checks))
        for check in checks:
            if check.status == "fail":
                check.fields = list(self.CHECK_FIELDS.get(check.name, ()))
        return PreflightReport(self.backend_name, self.describe_target(),
                               reachable, checks)

    def _preflight_target(self, checks: list[PreflightCheck]) -> bool:
        """Append the target's checks to `checks`; return whether it answered."""
        del checks
        return False

    def _port_forward_command(
        self,
        local_port: int,
        remote_port: int,
        job_id: str | None,
    ) -> tuple[list[str], str]:
        """The tunnelling command and a label to name its log file by.

        A driver that can forward a port implements this and nothing else:
        `port_forward` owns what running it means, so foreground and
        background behave the same way on every backend.
        """
        raise ComputeConfigError(
            f"Backend '{self.backend_name}' does not support port-forwarding.",
            error_code="UNSUPPORTED_OPERATION",
        )

    def port_forward(
        self,
        local_port: int,
        remote_port: int,
        job_id: str | None = None,
        background: bool = False,
    ) -> int:
        """Forward local_port to remote_port on the target device, pod, or job.

        Returns the child's pid when backgrounded, and its exit code when it
        ran in the foreground. Ctrl-C is how a foreground forward is meant to
        end, so it is a zero rather than a traceback.
        """
        cmd, label = self._port_forward_command(local_port, remote_port,
                                                job_id)
        if background:
            return spawn_port_forward(cmd, label, local_port=local_port)
        try:
            return subprocess.run(cmd).returncode
        except KeyboardInterrupt:
            return 0


def _listening(local_port: int) -> bool:
    """Whether something accepts a connection on the loopback port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", local_port)) == 0


def spawn_port_forward(cmd: list[str],
                       label: str,
                       local_port: int,
                       ready_timeout_sec: float = READY_TIMEOUT_SEC) -> int:
    """Start `cmd` detached and return its pid once the forward is serving."""
    log_dir = paths.get_runs_root() / "port-forward"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label}.log"
    with open(log_path, "wb") as handle:
        proc = subprocess.Popen(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _died() -> None:
        try:
            detail = log_path.read_text(encoding="utf-8",
                                        errors="replace").strip()
        except OSError:
            detail = ""
        raise ComputeConfigError(
            f"Port forward exited immediately (code {proc.returncode}): "
            f"{detail or 'no output'}. Log: {log_path}",
            error_code="PORT_FORWARD_FAILED",
        )

    deadline = time.monotonic() + ready_timeout_sec
    while True:
        if proc.poll() is not None:
            _died()
        if _listening(local_port):
            return proc.pid
        if time.monotonic() >= deadline:
            break
        time.sleep(READY_POLL_SEC)

    proc.terminate()
    raise ComputeConfigError(
        f"Port forward is running (pid {proc.pid}) but nothing is listening "
        f"on 127.0.0.1:{local_port} after {ready_timeout_sec:g}s; it has been "
        f"stopped rather than reported as established. Log: {log_path}",
        error_code="PORT_FORWARD_FAILED",
    )


# Where each tool, and how to install it, is documented.
TOOLS_DOC = "docs/install.md#1-host-prerequisites"

# The one command that installs a tool, where there is one.
TOOL_INSTALL = {
    "kubectl": "gcloud components install kubectl",
    "gke-gcloud-auth-plugin":
    "gcloud components install gke-gcloud-auth-plugin",
}


def _install_hint(tool: str) -> str:
    cmd = TOOL_INSTALL.get(tool)
    return f"`{cmd}` (see {TOOLS_DOC})" if cmd else f"see {TOOLS_DOC}"


def tool_check(tool: str) -> PreflightCheck:
    """Whether `tool` is on PATH, as a preflight check."""
    found = shutil.which(tool)
    if found:
        return PreflightCheck(f"tool:{tool}", "ok", found)
    return PreflightCheck(f"tool:{tool}", "fail", f"'{tool}' is not on PATH",
                          f"Install it: {_install_hint(tool)}")


def require_tools(tools: tuple[str, ...]) -> None:
    """Raise a MISSING_TOOL error naming each of `tools` not on PATH.

    Without this, the first subprocess call surfaces as a bare
    FileNotFoundError traceback that names neither the tool nor the fix.
    """
    missing = [tool for tool in tools if not shutil.which(tool)]
    if missing:
        raise ComputeConfigError(
            "Not on PATH: " + ", ".join(missing) + ". Install: " +
            "; ".join(f"{t}: {_install_hint(t)}" for t in missing) + ".",
            error_code="MISSING_TOOL")


def fail(checks: list[PreflightCheck],
         name: str,
         detail: str,
         fix: str = "") -> bool:
    """Record a failed check; returns False, for `return fail(...)`."""
    checks.append(PreflightCheck(name, "fail", detail, fix))
    return False


def probe_run(cmd: list[str],
              timeout: int = 20) -> subprocess.CompletedProcess[str]:
    """Run a read-only probe, never raising.

    Preflight reports and never raises, so one hung probe loses one check. A
    probe that could not run or timed out comes back as a failed process
    whose stderr says why.
    """
    try:
        return subprocess.run(cmd,
                              capture_output=True,
                              text=True,
                              timeout=timeout,
                              check=False)
    except subprocess.TimeoutExpired:
        reason = f"did not answer within {timeout}s"
    except (OSError, subprocess.SubprocessError) as exc:
        reason = str(exc)
    return subprocess.CompletedProcess(cmd, -1, "", reason)


def probe_error(proc: subprocess.CompletedProcess[str]) -> str:
    """The reason a probe failed, in one line."""
    text = (proc.stderr or proc.stdout or "").strip()
    if not text:
        return f"exit {proc.returncode}"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return next((line for line in lines
                 if line.lower().startswith(("error:", "error from server"))),
                lines[-1])


# kubectl authenticates to GKE through gke-gcloud-auth-plugin when the
# kubeconfig says so; without it every call fails, and the cluster check could
# only say the cluster does not answer. A kubeconfig holding a token does not
# need it.
GKE_TOOLS = ("kubectl", "gke-gcloud-auth-plugin")

# Node labels GKE puts on TPU nodes, and the resource chips are requested as.
ACCELERATOR_LABEL = "cloud.google.com/gke-tpu-accelerator"
TOPOLOGY_LABEL = "cloud.google.com/gke-tpu-topology"
TPU_RESOURCE = "google.com/tpu"


class TpuNodes(NamedTuple):
    """What a list of GKE TPU nodes adds up to."""
    accelerator: str | None
    topology: str | None
    chips: int
    names: list[str]


def tpu_nodes(items: list[dict[str, Any]]) -> TpuNodes:
    """Sum up the `items` of `kubectl get nodes -o json`.

    Labels come from the first node that has them, or None if none do.
    Chips are the nodes' allocatable `TPU_RESOURCE`, added up.
    """
    accelerator = topology = None
    chips = 0
    names = []
    for node in items:
        labels = node.get("metadata", {}).get("labels", {})
        accelerator = accelerator or labels.get(ACCELERATOR_LABEL)
        topology = topology or labels.get(TOPOLOGY_LABEL)
        names.append(node.get("metadata", {}).get("name", ""))
        allocatable = node.get("status", {}).get("allocatable", {})
        chips += int(allocatable.get(TPU_RESOURCE, 0) or 0)
    return TpuNodes(accelerator, topology, chips, names)


def gke_tools(kubectl: list[str]) -> tuple[str, ...]:
    """GKE_TOOLS, without the plugin when the kubeconfig's user does not call it."""
    if not shutil.which("kubectl"):
        return GKE_TOOLS
    proc = probe_run(kubectl + [
        "config", "view", "--minify", "-o",
        "jsonpath={.users[0].user.exec.command}"
    ],
                     timeout=10)
    if proc.returncode == 0 and "gke-gcloud-auth-plugin" not in proc.stdout:
        return ("kubectl", )
    return GKE_TOOLS


CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def _metadata_get(path: str) -> str:
    """One value from the GCE metadata server; raises off GCE."""
    request = urllib.request.Request(f"{probe.METADATA_URL}/{path}",
                                     headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.read().decode("utf-8").strip()


def vm_account_scopes(account: str) -> list[str] | None:
    """The OAuth scopes this VM was created with, when `account` is the
    VM's own service account; None for any other account, or off GCE.

    Tokens for a VM's service account carry only those scopes, so without
    cloud-platform they are refused whatever IAM grants the account. A user
    account has no such limit, so it is not looked up.
    """
    if not account.endswith(".gserviceaccount.com"):
        return None
    prefix = "instance/service-accounts/default/"
    try:
        if _metadata_get(prefix + "email") != account:
            return None
        return _metadata_get(prefix + "scopes").split()
    except (OSError, ValueError, http.client.HTTPException):
        return None


def _probe_active_gcloud_account(
) -> tuple[subprocess.CompletedProcess[str], str]:
    """The `gcloud auth list` probe for the active account, and that
    account; "" when there is none or the probe failed."""
    proc = probe_run([
        "gcloud", "auth", "list", "--filter=status:ACTIVE",
        "--format=value(account)"
    ])
    lines = proc.stdout.strip().splitlines() if proc.returncode == 0 else []
    return proc, lines[0].strip() if lines else ""


def active_gcloud_account() -> str:
    """The active gcloud account, or "" if there is none."""
    return _probe_active_gcloud_account()[1]


def gcloud_account_check(
        other_credential: str | None = None) -> PreflightCheck:
    """Whether gcloud has an active account to act as, with cloud-platform
    scope when it is a VM's service account.

    `other_credential` names a credential already known to work, such as a
    cluster that answers kubectl. With one, a problem here is a warning, not
    a blocker.
    """

    def problem(detail: str, fix: str) -> PreflightCheck:
        if other_credential:
            return PreflightCheck(
                "gcloud:account", "warn",
                f"{detail}; not blocking: {other_credential}", fix)
        return PreflightCheck("gcloud:account", "fail", detail, fix)

    proc, account = _probe_active_gcloud_account()
    if proc.returncode != 0 or not account:
        return problem(
            "gcloud has no active account"
            if proc.returncode == 0 else probe_error(proc),
            "gcloud auth login && gcloud auth application-default login")
    scopes = vm_account_scopes(account)
    if scopes is not None and CLOUD_PLATFORM_SCOPE not in scopes:
        return problem(
            f"{account} is this VM's service account, and the VM does not "
            "grant it the cloud-platform scope, so its tokens are refused "
            "whatever IAM allows (scopes: "
            f"{', '.join(s.rsplit('/', 1)[-1] for s in scopes) or 'none'})",
            "Log gcloud in as yourself (gcloud auth login; a user account is "
            "not limited by the VM's scopes), or recreate the VM or GKE node "
            "pool with --scopes=cloud-platform")
    return PreflightCheck("gcloud:account", "ok", account)


GAC_ENV = "GOOGLE_APPLICATION_CREDENTIALS"

KUBECTL_IDENTITY_WARNING = (
    f"gke-gcloud-auth-plugin honours ${GAC_ENV}, so it also decides who "
    "kubectl authenticates as; a credential without RBAC on the cluster "
    "gets an opaque HTML 403")

# The credential types gke-gcloud-auth-plugin can authenticate with.
USABLE_CREDENTIAL_TYPES = ("authorized_user", "service_account",
                           "external_account", "impersonated_service_account")


def application_credentials_check(
) -> tuple[PreflightCheck | None, str | None]:
    """$GOOGLE_APPLICATION_CREDENTIALS, reported as the credential it is.

    Returns the check (None when the variable is unset) and, when the file
    holds a usable credential, a description of it for
    `gcloud_account_check`. Always a warning, never a recommendation: see
    KUBECTL_IDENTITY_WARNING.
    """
    path = os.environ.get(GAC_ENV, "")
    if not path:
        return None, None
    name = "credentials:gac"
    try:
        info = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return PreflightCheck(
            name, "warn", f"${GAC_ENV} names {path}, which is not a "
            f"readable credential file ({exc}); {KUBECTL_IDENTITY_WARNING}",
            f"unset {GAC_ENV}, or point it at a credential file"), None
    info = info if isinstance(info, dict) else {}
    kind = str(info.get("type") or "unknown")
    who = str(info.get("client_email") or info.get("account") or "")
    credential = f"{kind} credential" + (f" {who}" if who else "")
    check = PreflightCheck(
        name, "warn", f"${GAC_ENV}: {credential} ({path}). "
        f"{KUBECTL_IDENTITY_WARNING}",
        f"If kubectl is refused, run it without the variable (env -u "
        f"{GAC_ENV} ...) to use the gcloud account, or grant this credential "
        "RBAC on the cluster")
    usable = kind in USABLE_CREDENTIAL_TYPES
    return check, (f"${GAC_ENV} holds another credential ({credential})"
                   if usable else None)


def gke_credential_checks(cluster: PreflightCheck) -> list[PreflightCheck]:
    """The credential checks for a backend whose kubectl authenticates
    through gke-gcloud-auth-plugin, given its cluster check.

    A cluster that answers proves kubectl has a working credential, and a
    usable $GOOGLE_APPLICATION_CREDENTIALS is one the plugin can use, so
    either turns a gcloud account problem into a warning.
    """
    gac, usable = application_credentials_check()
    other = ("kubectl reaches the cluster"
             if cluster.status == "ok" else usable)
    # Without gcloud, the plugin cannot use a gcloud account either.
    checks = [gcloud_account_check(other)] if shutil.which("gcloud") else []
    if gac:
        checks.append(gac)
    return checks


# How a request that never reached the API server fails, as opposed to one
# the server refused.
NETWORK_TIMEOUT_MARKERS = ("deadline exceeded", "i/o timeout",
                           "did not answer within", "connection timed out",
                           "no route to host")

GKE_CONTEXT = re.compile(r"gke_([^_]+)_([^_]+)_([^_]+)")


def gke_target(context: str | None = None,
               prefer_context: bool = False) -> dict[str, str]:
    """The cluster, location and project a get-credentials fix names.

    All three from one source, never mixed: GKE_CLUSTER / GKE_LOCATION /
    GKE_PROJECT when any is set, as CI and eval environments set them, or a
    `gke_<project>_<location>_<cluster>` context; `prefer_context` tries the
    context first, for a cluster already known to answer. Placeholders for
    whatever the chosen source lacks.
    """
    parts = ("cluster", "location", "project")
    match = GKE_CONTEXT.fullmatch(context or "")
    from_context = dict(zip(parts[::-1], match.groups())) if match else {}
    from_env = {
        part: os.environ[f"GKE_{part.upper()}"]
        for part in parts if os.environ.get(f"GKE_{part.upper()}")
    }
    first, second = ((from_context, from_env) if prefer_context else
                     (from_env, from_context))
    source = first or second
    return {part: source.get(part) or f"<{part}>" for part in parts}


def get_credentials_fix(kubeconfig: str | None = None,
                        extra: str = "",
                        context: str | None = None) -> str:
    """The `gcloud ... get-credentials` command that configures the cluster.

    Names the cluster `gke_target` finds for `context`. With `kubeconfig`,
    it writes to that file rather than the default one.
    """
    target = gke_target(context)
    cmd = (f"gcloud container clusters get-credentials {target['cluster']} "
           f"--location {target['location']} --project {target['project']}"
           f"{extra}")
    return f"KUBECONFIG={kubeconfig} {cmd}" if kubeconfig else cmd


def missing_kubeconfig(kubeconfig: str | None,
                       context: str | None = None) -> PreflightCheck | None:
    """A failed check naming the kubeconfig kubectl read that is not there.

    With a pinned `context`, it is named, and so is its cluster in the fix.
    """
    env = os.environ.get("KUBECONFIG", "")
    chosen = [kubeconfig
              ] if kubeconfig else [f for f in env.split(os.pathsep) if f]
    missing = [
        f for f in chosen or ["~/.kube/config"]
        if not os.path.isfile(os.path.expanduser(f))
    ]
    if not missing:
        return None
    named_by = "$KUBECONFIG names" if env and not kubeconfig else "kubeconfig"
    verb = "does" if len(missing) == 1 else "do"
    lead = (f"context '{context}' cannot be used"
            if context else "kubectl has no current context")
    return PreflightCheck(
        "cluster",
        "fail",
        f"{lead}: {named_by} "
        f"{', '.join(missing)}, which {verb} not exist",
        # The default file gets written without being named; a chosen one
        # must be, or get-credentials writes where kubectl does not read.
        get_credentials_fix(missing[0] if chosen else None, context=context))


KUBE_CONTEXT_HELP = ("kubectl context every call names, instead of whichever "
                     "one is current")


def kubectl_prefix(kubeconfig: str | None = None,
                   context: str | None = None) -> list[str]:
    """`kubectl` with the kubeconfig and context a profile names, if any."""
    return (["kubectl"] +
            (["--kubeconfig", kubeconfig] if kubeconfig else []) +
            (["--context", context] if context else []))


def pinned_context(cmd: list[str]) -> str | None:
    """The context a kubectl command line names with `--context`, if any."""
    return cmd[cmd.index("--context") + 1] if "--context" in cmd else None


def current_context(kubectl: list[str]) -> str:
    """The context `kubectl` (a command prefix) talks to, or "" if none.

    `kubectl config current-context` ignores `--context`, so a pinned
    command's context is read off its argv instead.
    """
    if pinned := pinned_context(kubectl):
        return pinned
    proc = probe_run(kubectl + ["config", "current-context"], timeout=10)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def kube_context_check(kubeconfig: str | None,
                       context: str | None) -> PreflightCheck | None:
    """Whether a pinned context exists, and if kubectl's current one differs.

    None when nothing is pinned. A failure means kubectl cannot use the
    context, so there is no cluster to check.
    """
    if not context:
        return None
    config = kubectl_prefix(kubeconfig) + ["config"]
    listed = probe_run(config + ["get-contexts", "-o", "name"], timeout=10)
    names = listed.stdout.split() if listed.returncode == 0 else []
    if context not in names:
        missing = missing_kubeconfig(kubeconfig, context)
        if missing:
            return missing
        add = "Add it: " + get_credentials_fix(kubeconfig, context=context)
        if listed.returncode != 0:
            return PreflightCheck(
                "kube_context", "fail",
                f"could not read the kubeconfig's contexts: "
                f"{probe_error(listed)}", add)
        return PreflightCheck(
            "kube_context", "fail",
            f"context '{context}' is not in the kubeconfig",
            add + (", or set kube_context to one of: " +
                   ", ".join(names) if names else ""))
    current = current_context(config[:-1])
    if current == context:
        return PreflightCheck("kube_context", "ok",
                              f"context '{context}' is current")
    return PreflightCheck(
        "kube_context", "warn",
        f"compute commands use the profile's context '{context}', not "
        f"kubectl's current context '{current or '(none)'}'",
        "Nothing to do; to use the current context instead, set "
        "kube_context to it")
