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
"""Compute Engine (GCE) TPU VM compute driver."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextvars
import json
import pathlib
import shlex
import subprocess
from typing import Any, Callable

from . import base, probe, registry, shell

Path = pathlib.Path

# Reuse an existing SSH master channel across consecutive commands (and IAP
# tunnels) instead of opening a new handshake on each status poll.
_SSH_MULTIPLEX_FLAGS = (
    "--ssh-flag=-o ControlMaster=auto",
    "--ssh-flag=-o ControlPath=/tmp/ssh-tpu-agent-suite-%C",
    "--ssh-flag=-o ControlPersist=600",
)

_TRUE_VALUES = ("1", "true", "yes")
_NOT_FOUND_MARKERS = ("was not found", "not found:",
                      "could not fetch resource")
_STATUS_UNREACHABLE: contextvars.ContextVar[list[str]
                                            | None] = (contextvars.ContextVar(
                                                "_gce_status_unreachable",
                                                default=None))


def _clean_id(value: Any, field: str, *, allow_colon: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        raise base.ComputeConfigError(
            f"Backend 'gce_tpu_vm' requires '{field}' in the compute profile "
            f"(`config set`) or via --{field.replace('_', '-')}.",
            error_code="MISSING_FIELD",
        )
    bad = ",/" if allow_colon else ":,/"
    if text.startswith("-") or any(c.isspace() or c in bad for c in text):
        raise base.ComputeConfigError(
            f"Invalid '{field}' value {value!r} for backend 'gce_tpu_vm'.",
            error_code="INVALID_FIELD",
        )
    return text


def _parse_hosts(value: Any) -> tuple[str, ...]:
    raw = str(value or "").strip()
    if not raw:
        _clean_id("", "tpu_name")
    hosts = tuple(_clean_id(p, "tpu_name") for p in raw.split(","))
    if len(set(hosts)) != len(hosts):
        raise base.ComputeConfigError(
            f"Duplicate host in 'tpu_name': {value!r}.",
            error_code="INVALID_FIELD",
        )
    return hosts


def _has_external_ip(nics: list[Any]) -> bool:
    return any(
        isinstance(n, dict) and (n.get("ipv6AccessConfigs") or any(
            isinstance(c, dict) and c.get("natIP")
            for c in (n.get("accessConfigs") or []))) for n in nics)


def _worker_rank(item: dict[str, Any], fallback: int) -> tuple[int, str]:
    name = str(item.get("name") or "")
    for entry in (item.get("metadata") or {}).get("items") or []:
        if isinstance(entry,
                      dict) and entry.get("key") == "agent-worker-number":
            val = str(entry.get("value") or "").strip()
            if val.isdigit():
                return int(val), name
    return fallback, name


@registry.register_driver("gce_tpu_vm")
class GceTpuVmDriver(shell.RemoteShellDriver):
    """Driver executing on Compute Engine TPU VM(s) or MIGs over `gcloud compute`."""

    target_description = "GCE TPU VM"
    required_tools = ("gcloud", )
    CHECK_FIELDS = {"tpu_vm": ("tpu_name", "zone", "project")}
    SETUP_FIELDS = base.SetupFields(
        required=("project", "zone", "tpu_name"),
        optional=("python_interpreter", "tunnel_through_iap", "internal_ip"),
        on_demand={
            "ml_diagnostics_cluster":
            "uploading profiles to an ML Diagnostics dashboard",
            "ml_diagnostics_gcs":
            "telling ML Diagnostics which GCS folder to read profiles from",
        },
    )

    def __init__(
        self,
        tpu_name: str,
        zone: str,
        project: str,
        python_interpreter: str | None = None,
        internal_ip: bool = False,
        tunnel_through_iap: bool = False,
    ) -> None:
        super().__init__(python_interpreter=python_interpreter)
        hosts = tuple(h.strip() for h in str(tpu_name).split(",") if h.strip())
        self.hosts = hosts or (tpu_name, )
        self._mig_checked = len(self.hosts) > 1
        self.tpu_name = ",".join(self.hosts)
        self.zone = zone
        self.project = project
        self.internal_ip = internal_ip
        self.tunnel_through_iap = tunnel_through_iap

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> GceTpuVmDriver:
        """Build the driver; `tpu_name`, `zone`, and `project` are required."""
        hosts = _parse_hosts(params.get("tpu_name"))
        zone = _clean_id(params.get("zone"), "zone")
        project = _clean_id(params.get("project"), "project", allow_colon=True)
        internal_ip = str(params.get("internal_ip",
                                     "")).lower() in _TRUE_VALUES
        iap = str(params.get("tunnel_through_iap", "")).lower() in _TRUE_VALUES
        if internal_ip and iap:
            raise base.ComputeConfigError(
                "Cannot enable both 'internal_ip' and 'tunnel_through_iap'; "
                "choose one.",
                error_code="INVALID_FIELD",
            )
        return cls(
            tpu_name=",".join(hosts),
            zone=zone,
            project=project,
            python_interpreter=params.get("python_interpreter"),
            internal_ip=internal_ip,
            tunnel_through_iap=iap,
        )

    @classmethod
    def register_cli_args(cls, parser: Any) -> None:
        """Add `--tpu-name`, `--zone`, `--project`, and routing flags."""
        cls.add_cli_arg(
            parser,
            "--tpu-name",
            help="GCE TPU VM, MIG, or comma-separated VM names (gce_tpu_vm)")
        cls.add_cli_arg(parser, "--zone", help="GCP zone of compute resource")
        cls.add_cli_arg(parser, "--project", help="GCP project ID")
        cls.add_cli_arg(parser,
                        "--internal-ip",
                        action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Connect over internal VPC IP")
        cls.add_cli_arg(parser,
                        "--tunnel-through-iap",
                        action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Tunnel SSH/SCP through Cloud IAP")

    def _resolve_mig_if_needed(self) -> bool:
        """If single `tpu_name` names a MIG, populate `self.hosts` sorted by ICI rank."""
        if self._mig_checked or len(self.hosts) > 1:
            return False
        self._mig_checked = True
        proc = base.probe_run([
            "gcloud", "compute", "instance-groups", "managed",
            "list-instances", self.hosts[0], f"--zone={self.zone}",
            f"--project={self.project}", "--format=value(instance.basename())"
        ],
                              timeout=30)
        names = [
            n.strip() for n in (proc.stdout or "").splitlines() if n.strip()
        ] if proc.returncode == 0 else []
        if not names:
            return False
        meta_proc = base.probe_run([
            "gcloud", "compute", "instances", "list",
            f"--filter=name:({' '.join(names)})", f"--zones={self.zone}",
            f"--project={self.project}", "--format=json(name,metadata.items)"
        ],
                                   timeout=30)
        if meta_proc.returncode == 0:
            try:
                raw_items = json.loads(meta_proc.stdout or "[]")
                name_set = set(names)
                items = [
                    it for it in raw_items
                    if isinstance(it, dict) and it.get("name") in name_set
                ] if isinstance(raw_items, list) else []
                if len(items) == len(names):
                    names = [
                        n for _, n in sorted(
                            _worker_rank(it, i) for i, it in enumerate(items))
                        if n
                    ]
            except ValueError:
                pass
        self.hosts = tuple(names)
        return True

    def _retry_mig(
        self, fn: Callable[[], subprocess.CompletedProcess]
    ) -> subprocess.CompletedProcess:
        res = fn()
        if (res.returncode != 0 and not self._mig_checked
                and any(m in (res.stderr or "").lower()
                        for m in _NOT_FOUND_MARKERS)
                and self._resolve_mig_if_needed()):
            return fn()
        return res

    def _net_flags(self) -> list[str]:
        if self.tunnel_through_iap:
            return ["--tunnel-through-iap"]
        return ["--internal-ip"] if self.internal_ip else []

    def _gcloud_base(self,
                     subcmd: str = "ssh",
                     host: str | None = None) -> list[str]:
        """Build a `gcloud compute` command addressing `host` (default leader)."""
        nouns = ("instances",
                 "describe") if subcmd == "describe" else (subcmd, )
        cmd = [
            "gcloud", "compute", *nouns, host or self.hosts[0],
            f"--zone={self.zone}", f"--project={self.project}"
        ]
        if subcmd == "ssh":
            cmd.extend((*_SSH_MULTIPLEX_FLAGS, *self._net_flags()))
        return cmd

    def _worker_env(self, idx: int) -> str:
        n = len(self.hosts)
        if n <= 1:
            return ""
        hosts_csv = ",".join(self.hosts)
        meta = "http://metadata.google.internal/computeMetadata/v1/instance"
        return (
            f"_w=$(curl -sf -m 1 -H Metadata-Flavor:Google {meta}/attributes/agent-worker-number || true); "
            f"_ep=$(curl -sf -m 1 -H Metadata-Flavor:Google {meta}/attributes/worker-network-endpoints "
            f"| tr ',' '\\n' | awk -F: '{{print $NF}}' | paste -sd, - || true); "
            f"if [ \"$(printf '%s' \"$_ep\" | awk -F, 'NF{{print NF}}')\" != \"{n}\" ]; then _w=; _ep=; fi; "
            f"_mt=$(curl -sf -m 1 -H Metadata-Flavor:Google {meta}/machine-type "
            f"| sed -n 's/.*-\\([0-9][0-9]*\\)t$/\\1/p' || true); "
            f"_wid=\"${{_w:-{idx}}}\"; _hcsv=\"${{_ep:-{shlex.quote(hosts_csv)}}}\"; "
            f"_head=\"${{_hcsv%%,*}}\"; "
            f"export TPU_NUM_HOSTS={n} TPU_WORKER_ID=\"$_wid\" "
            f"TPU_NODE_ID=\"$_wid\" TPU_HEAD_HOST=\"$_head\" "
            f"TPU_WORKER_HOSTNAMES=\"$_hcsv\"; "
            f"_c=$(ls -d /dev/vfio/[0-9]* 2>/dev/null | wc -l | tr -d ' '); "
            f"[ \"${{_c:-0}}\" -eq 0 ] && _c=$(ls -d /dev/accel[0-9]* 2>/dev/null | wc -l | tr -d ' '); "
            f"_chips=\"${{_mt:-$([ \"${{_c:-0}}\" -gt 0 ] && echo \"$_c\" || true)}}\"; "
            f"[ -n \"$_chips\" ] && export TPU_CHIPS_PER_HOST=\"${{TPU_CHIPS_PER_HOST:-$_chips}}\"; "
        )

    def _parallel(
        self, fn: Callable[[int, str], subprocess.CompletedProcess]
    ) -> subprocess.CompletedProcess:
        if len(self.hosts) == 1:
            return self._retry_mig(lambda: fn(0, self.hosts[0]) if len(
                self.hosts) == 1 else self._parallel(fn))
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(self.hosts)) as pool:
            procs = list(
                pool.map(lambda item: fn(item[0], item[1]),
                         enumerate(self.hosts)))
        code = next((p.returncode for p in procs if p.returncode != 0), 0)
        unreachable = _STATUS_UNREACHABLE.get()
        parts: list[str] = []
        for i, p in enumerate(procs):
            s = (p.stdout or "").rstrip("\n")
            if s:
                parts.append(s)
            elif unreachable is not None and p.returncode != 0:
                parts.append(f"@{i} UNKNOWN")
                unreachable.append(self.hosts[i])
        out = "\n".join(parts)
        err = "\n".join(
            s.rstrip("\n") for p in procs
            if (s := (p.stderr or ""))).rstrip("\n")
        return subprocess.CompletedProcess(procs[0].args, code,
                                           f"{out}\n" if out else "",
                                           f"{err}\n" if err else "")

    def get_status(self, job_id: str) -> base.AsynchronousExecutionResult:
        """Query per-host job status, marking unreachable hosts as UNKNOWN."""
        unreachable: list[str] = []
        tok = _STATUS_UNREACHABLE.set(unreachable)
        try:
            res = super().get_status(job_id)
        finally:
            _STATUS_UNREACHABLE.reset(tok)
        if res.status == "UNKNOWN" and unreachable and not res.error:
            res.error = f"unreachable host(s): {', '.join(unreachable)}"
        return res

    def _ssh_one(self,
                 idx: int,
                 host: str,
                 script: str,
                 timeout: int | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            self._gcloud_base("ssh", host) +
            [f"--command={self._worker_env(idx)}{script}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _sh(self,
            script: str,
            timeout: int | None = None) -> subprocess.CompletedProcess:
        return self._parallel(
            lambda idx, host: self._ssh_one(idx, host, script, timeout))

    def _sh_leader(self,
                   script: str,
                   timeout: int | None = None,
                   worker: int | None = None) -> subprocess.CompletedProcess:

        def _run() -> subprocess.CompletedProcess:
            idx = worker if worker is not None and 0 <= worker < len(
                self.hosts) else 0
            return self._ssh_one(idx, self.hosts[idx], script, timeout)

        return self._retry_mig(_run)

    def _env_preamble(self) -> str:
        return ":" if len(self.hosts) > 1 else ""

    def _copy_in(self, src: Path, dest: str) -> subprocess.CompletedProcess:
        return self._parallel(
            lambda _idx, host: self._scp(str(src), f"{host}:{dest}"))

    def _copy_out(self, src: str, dest: Path) -> subprocess.CompletedProcess:
        return self._retry_mig(lambda: self._scp(
            f"{self.hosts[0]}:{src}/*", str(dest), recurse=True))

    def _copy_out_file(self, src: str,
                       dest: Path) -> subprocess.CompletedProcess:
        return self._retry_mig(
            lambda: self._scp(f"{self.hosts[0]}:{src}", str(dest)))

    def _scp(self,
             source: str,
             destination: str,
             recurse: bool = False) -> subprocess.CompletedProcess:
        cmd = [
            "gcloud", "compute", "scp", *(["--recurse"] if recurse else []),
            source, destination, f"--zone={self.zone}",
            f"--project={self.project}", *self._net_flags()
        ]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    def _port_forward_command(
        self,
        local_port: int,
        remote_port: int,
        job_id: str | None,
    ) -> tuple[list[str], str]:
        """Tunnel to the leader GCE TPU VM over SSH."""
        del job_id
        if not self._mig_checked and len(self.hosts) == 1:
            chk = base.probe_run(
                self._gcloud_base("describe", self.hosts[0]) +
                ["--format=value(status)"],
                timeout=15,
            )
            if chk.returncode != 0:
                self._resolve_mig_if_needed()
            else:
                self._mig_checked = True
        cmd = self._gcloud_base("ssh", self.hosts[0]) + [
            "--", "-N", f"-L{local_port}:localhost:{remote_port}"
        ]
        return cmd, f"{self.hosts[0]}-{local_port}"

    def _profile_details(self) -> dict[str, Any]:
        return {
            "tpu_name": self.tpu_name,
            "zone": self.zone,
            "project": self.project,
        }

    def describe_target(self) -> str:
        """Return the GCE TPU VM/MIG name(s), zone, and project."""
        noun = "GCE TPU VMs" if len(self.hosts) > 1 else "GCE TPU VM"
        return (f"{noun} '{self.tpu_name}' (zone {self.zone}, project "
                f"{self.project})")

    def _preflight_target(self, checks: list[base.PreflightCheck]) -> bool:
        """An active gcloud account, and every GCE host in RUNNING state."""
        checks.append(base.gcloud_account_check())
        if checks[-1].status == "fail":
            return False

        def _desc(host: str) -> subprocess.CompletedProcess[str]:
            return base.probe_run(
                self._gcloud_base("describe", host) +
                ["--format=json(status,machineType,networkInterfaces)"],
                timeout=30,
            )

        procs = [_desc(self.hosts[0])] if len(self.hosts) == 1 else []
        if procs and procs[0].returncode != 0 and self._resolve_mig_if_needed(
        ):
            procs = []
        if not procs:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(self.hosts)) as pool:
                procs = list(pool.map(_desc, self.hosts))

        machine, missing_ext_ip = "?", []
        for host, proc in zip(self.hosts, procs):
            if proc.returncode != 0:
                return base.fail(
                    checks, "tpu_vm",
                    f"could not describe GCE TPU VM '{host}': "
                    f"{base.probe_error(proc)}",
                    "Check tpu_name, zone and project against "
                    f"`gcloud compute instances list --zones={self.zone} "
                    f"--project={self.project}`")
            try:
                info = json.loads(proc.stdout or "{}")
            except ValueError:
                info = {}
            status = str((info if isinstance(info, dict) else {}).get("status")
                         or "?")
            machine = str(
                (info if isinstance(info, dict) else {}).get("machineType")
                or "?").rsplit("/", 1)[-1]
            if status != "RUNNING":
                verb = {
                    "TERMINATED": "start",
                    "SUSPENDED": "resume"
                }.get(status)
                fix = (f"gcloud compute instances {verb} {host} "
                       f"--zone={self.zone} --project={self.project}"
                       if verb else "Wait for it to reach RUNNING")
                prefix = f"{host} " if len(self.hosts) > 1 else ""
                return base.fail(checks, "tpu_vm",
                                 f"{prefix}status is {status} ({machine})",
                                 fix)
            nics = info.get("networkInterfaces") if isinstance(info,
                                                               dict) else None
            if (isinstance(nics, list) and nics and not self.internal_ip
                    and not self.tunnel_through_iap
                    and not _has_external_ip(nics)):
                missing_ext_ip.append(host)

        summary = (f"{len(self.hosts)}/{len(self.hosts)} RUNNING ({machine})"
                   if len(self.hosts) > 1 else f"RUNNING ({machine})")
        checks.append(base.PreflightCheck("tpu_vm", "ok", summary))
        if machine != "?" and not probe.generation_for_name(machine):
            checks.append(
                base.PreflightCheck(
                    "tpu_vm:machine_type", "warn",
                    f"machine type '{machine}' is not a known TPU family",
                    "Verify 'tpu_name' names a TPU VM (e.g. ct6e-*, tpu7x-*)"))
        if missing_ext_ip:
            checks.append(
                base.PreflightCheck(
                    "tpu_vm:network", "warn",
                    f"GCE TPU VM '{missing_ext_ip[0]}' has no external IP",
                    "Set tunnel_through_iap=true or internal_ip=true"))
        return True

    def inspect(self) -> base.HardwareProfile:
        """Probe live TPU hardware and scale capacity across multi-host slices."""
        prof = self._remote_profile()
        n = len(self.hosts)
        if n <= 1 or prof.chips <= 0 or "job_hosts" in prof.details:
            return prof
        per_host, total = prof.chips, prof.chips * n
        avail = None if prof.available_chips is None else prof.available_chips * n
        topo = (probe.topology_for(
            total, prof.accelerator) if prof.topology == probe.topology_for(
                per_host, prof.accelerator) else prof.topology)
        details = {
            **prof.details, "job_hosts": n,
            "chips_per_host": per_host,
            "job_chips": total
        }
        if "job_devices" in details:
            details["job_devices"] = total * int(
                details.get("devices_per_chip", 1))
        return base.HardwareProfile(
            accelerator=prof.accelerator,
            topology=topo,
            chips=total,
            hbm_gb=prof.hbm_gb,
            status=prof.status,
            available_chips=avail,
            held_by=prof.held_by,
            details=details,
        )
