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
"""GCE Cloud TPU VM compute driver."""

from __future__ import annotations

import pathlib
import subprocess
from typing import Any

from . import base, registry, shell

Path = pathlib.Path

# This flag combos allows re-use of an existing SSH connection, instead
# of needing to re-launch one for each command
_SSH_MULTIPLEX_FLAGS = (
    # The first ssh opens a master connection; later ones ride it as extra
    # channels instead of dialling out themselves.
    "--ssh-flag=-o ControlMaster=auto",
    # The unix socket master and clients meet on. %C hashes user/host/port, so
    # two different TPU VMs never collide on one socket.
    "--ssh-flag=-o ControlPath=/tmp/ssh-tpu-agent-suite-%C",
    # The master lingers 600s after the last command, so status polls spread
    # over minutes still find it alive and reuse it.
    "--ssh-flag=-o ControlPersist=600",
)


@registry.register_driver("gce_tpu_vm")
class GceTpuVmDriver(shell.RemoteShellDriver):
    """Driver executing on a Cloud TPU VM over `gcloud compute tpus tpu-vm`.

    Supplies the `RemoteShellDriver` transport primitives as `gcloud ... ssh`
    and `gcloud ... scp`; the job lifecycle built on them is inherited. Jobs
    land under the base `JOB_ROOT` (`/tmp`, the writable scratch space on a
    TPU VM image).
    """

    target_description = "TPU VM"
    required_tools = ("gcloud", )
    CHECK_FIELDS = {"tpu_vm": ("tpu_name", "zone", "project")}
    SETUP_FIELDS = base.SetupFields(
        required=("project", "zone", "tpu_name"),
        optional=("python_interpreter", "internal_ip"),
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
    ) -> None:
        super().__init__(python_interpreter=python_interpreter)
        self.tpu_name = tpu_name
        self.zone = zone
        self.project = project
        # For a host on the VM's VPC whose route to the external address is
        # firewalled
        self.internal_ip = internal_ip

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> GceTpuVmDriver:
        """Build the driver; tpu_name, zone and project are all required."""
        tpu_name = params.get("tpu_name")
        zone = params.get("zone")
        project = params.get("project")
        for value, requirement, flag in (
            (tpu_name, "'tpu_name'", "--tpu-name"),
            (zone, "'zone'", "--zone"),
            (project, "'project'", "--project"),
        ):
            if not value:
                raise base.ComputeConfigError(
                    f"Backend 'gce_tpu_vm' requires {requirement} in the "
                    f"compute profile (`config set`) or via {flag}.",
                    error_code="MISSING_FIELD",
                )
        return cls(
            tpu_name=tpu_name,
            zone=zone,
            project=project,
            python_interpreter=params.get("python_interpreter"),
            internal_ip=str(params.get("internal_ip", "")).lower()
            in ("1", "true", "yes"),
        )

    @classmethod
    def register_cli_args(cls, parser: Any) -> None:
        """Add `--tpu-name`, `--zone` and `--project`."""
        cls.add_cli_arg(parser,
                        "--tpu-name",
                        help="Cloud TPU VM resource name (gce_tpu_vm)")
        cls.add_cli_arg(parser, "--zone", help="GCP zone of compute resource")
        cls.add_cli_arg(parser, "--project", help="GCP project ID")

    def _gcloud_base(self, subcmd: str = "ssh") -> list[str]:
        """Build a `gcloud compute tpus tpu-vm` command addressing this VM."""
        cmd = [
            "gcloud", "compute", "tpus", "tpu-vm", subcmd, self.tpu_name,
            f"--zone={self.zone}", f"--project={self.project}"
        ]
        if subcmd == "ssh":
            cmd.extend(_SSH_MULTIPLEX_FLAGS)
            # Only ssh and scp take it; `describe` rejects the flag.
            if self.internal_ip:
                cmd.append("--internal-ip")
        return cmd

    def _sh(self,
            script: str,
            timeout: int | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            self._gcloud_base("ssh") + [f"--command={script}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _copy_in(self, src: Path, dest: str) -> subprocess.CompletedProcess:
        return self._scp(str(src), f"{self.tpu_name}:{dest}")

    def _copy_out(self, src: str, dest: Path) -> subprocess.CompletedProcess:
        return self._scp(f"{self.tpu_name}:{src}/*",
                         str(dest),
                         recurse=True,
                         timeout=120)

    def _copy_out_file(self, src: str,
                       dest: Path) -> subprocess.CompletedProcess:
        return self._scp(f"{self.tpu_name}:{src}", str(dest), timeout=120)

    def _scp(self,
             source: str,
             destination: str,
             recurse: bool = False,
             timeout: int = 120) -> subprocess.CompletedProcess:
        cmd = ["gcloud", "compute", "tpus", "tpu-vm", "scp"]
        if recurse:
            cmd.append("--recurse")
        cmd.extend([
            source, destination, f"--zone={self.zone}",
            f"--project={self.project}"
        ])
        if self.internal_ip:
            cmd.append("--internal-ip")
        return subprocess.run(cmd,
                              capture_output=True,
                              text=True,
                              timeout=timeout)

    def _port_forward_command(
        self,
        local_port: int,
        remote_port: int,
        job_id: str | None,
    ) -> tuple[list[str], str]:
        """Tunnel to the TPU VM over SSH; there is one target, so no job id."""
        cmd = self._gcloud_base("ssh") + [
            "--",
            "-N",
            f"-L{local_port}:localhost:{remote_port}",
        ]
        return cmd, f"{self.tpu_name}-{local_port}"

    def _profile_details(self) -> dict[str, Any]:
        """Name the VM the probe numbers were read from."""
        return {
            "tpu_name": self.tpu_name,
            "zone": self.zone,
            "project": self.project,
        }

    def describe_target(self) -> str:
        """The TPU VM's name, zone and project."""
        return (f"Cloud TPU VM '{self.tpu_name}' (zone {self.zone}, project "
                f"{self.project})")

    def _preflight_target(self, checks: list[base.PreflightCheck]) -> bool:
        """An active gcloud account, and a TPU VM in the READY state."""
        checks.append(base.gcloud_account_check())
        if checks[-1].status == "fail":
            return False
        proc = base.probe_run(self._gcloud_base("describe") +
                              ["--format=value(state,acceleratorType)"],
                              timeout=30)
        if proc.returncode != 0:
            return base.fail(
                checks, "tpu_vm",
                f"could not describe TPU VM '{self.tpu_name}': "
                f"{base.probe_error(proc)}",
                "Check the profile's tpu_name, zone and project against "
                "`gcloud compute tpus tpu-vm list "
                f"--zone={self.zone} --project={self.project}`")
        state, accelerator = (proc.stdout.split() + ["?", "?"])[:2]
        if state != "READY":
            return base.fail(
                checks, "tpu_vm", f"state is {state} ({accelerator})",
                f"gcloud compute tpus tpu-vm start {self.tpu_name} "
                f"--zone={self.zone} --project={self.project}"
                if state == "STOPPED" else "Wait for it to reach READY")
        checks.append(
            base.PreflightCheck("tpu_vm", "ok", f"READY ({accelerator})"))
        return True

    def inspect(self) -> base.HardwareProfile:
        """Probe the TPU VM's live hardware state over SSH.

        The probe runs on the VM rather than reading `gcloud describe`, because
        the API reports provisioning state, not whether the chips are currently
        held by another process.
        """
        return self._remote_profile()
