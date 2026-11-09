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
"""GKE single-pod compute driver."""

from __future__ import annotations

import pathlib
import subprocess
from typing import Any

from . import auth_utils, base, registry, shell
from .kubectl import Kubectl

Path = pathlib.Path

# The pod's shell prints this first, so output without it means the exec never
# started.
EXEC_STARTED = "compute-exec-started"


def _refused_before_exec(proc: subprocess.CompletedProcess[str]) -> bool:
    """True if kubectl was refused before the pod's shell started."""
    started = (proc.stdout or "").startswith(f"{EXEC_STARTED}\n")
    return not started and auth_utils.credential_refused(proc)


@registry.register_driver("gke_pod")
class GkePodDriver(shell.RemoteShellDriver):
    """Driver executing inside an existing GKE TPU workload pod via kubectl.

    Supplies the `RemoteShellDriver` transport primitives as `kubectl exec` and
    `kubectl cp`; the job lifecycle built on them is inherited. The pod must
    already be running -- this driver attaches to a workload, it does not create
    one (`GkeJobDriver` is the backend that launches cluster workloads).
    """

    # app is the workload's own writable directory.
    JOB_ROOT = "/app/tpu_agent_suite_jobs"
    target_description = "pod"
    required_tools = base.GKE_TOOLS
    runs_in_container = True
    CHECK_FIELDS = {
        "pod": ("pod", "namespace"),
        "kube_context": ("kube_context", )
    }
    SETUP_FIELDS = base.SetupFields(
        required=("pod", ),
        optional=("kube_context", "namespace", "container",
                  "python_interpreter"),
    )

    def __init__(
        self,
        namespace: str = "default",
        pod: str = "",
        container: str | None = None,
        python_interpreter: str | None = None,
        kube_context: str | None = None,
    ) -> None:
        super().__init__(python_interpreter=python_interpreter)
        self.kube = Kubectl(None, kube_context, namespace)
        self.pod = pod
        self.container = container

    @property
    def namespace(self) -> str:
        return self.kube.namespace

    @property
    def kube_context(self) -> str | None:
        return self.kube.kube_context

    @property
    def using_adc(self) -> bool:
        """Whether the cluster refused the gcloud account but let ADC in."""
        return self.kube.using_adc

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> GkePodDriver:
        pod = params.get("pod")
        if not pod:
            raise base.ComputeConfigError(
                "Backend 'gke_pod' requires 'pod' in the compute profile "
                "(`config set`) or via --pod.",
                error_code="MISSING_FIELD",
            )
        return cls(
            namespace=params.get("namespace", "default"),
            pod=pod,
            container=params.get("container"),
            python_interpreter=params.get("python_interpreter"),
            kube_context=(params.get("kube_context") or "").strip() or None,
        )

    @classmethod
    def register_cli_args(cls, parser: Any) -> None:
        cls.add_cli_arg(parser,
                        "--pod",
                        help="GKE TPU workload pod name (gke_pod)")
        cls.add_cli_arg(parser,
                        "--namespace",
                        help="Kubernetes namespace of the workload pod")
        cls.add_cli_arg(
            parser,
            "--container",
            help="Container within the pod, for multi-container pods")
        cls.add_cli_arg(parser, "--kube-context", help=base.KUBE_CONTEXT_HELP)

    def needed_tools(self) -> tuple[str, ...]:
        return base.gke_tools(self._kubectl_base())

    def _kubectl_base(self) -> list[str]:
        """Build a kubectl command scoped to this driver's context and
        namespace."""
        return self.kube.cmd(self.kube.ns())

    def _exec_prefix(self) -> list[str]:
        """Build the `kubectl exec` prefix addressing this pod's container."""
        cmd = self._kubectl_base() + ["exec", self.pod]
        if self.container:
            cmd.extend(["-c", self.container])
        return cmd

    def _sh(self,
            script: str,
            timeout: int | None = None) -> subprocess.CompletedProcess:
        proc = auth_utils.run_kubectl(
            self.kube,
            self._exec_prefix() +
            ["--", "sh", "-c", f"echo {EXEC_STARTED}\n{script}"],
            refused=_refused_before_exec,
            timeout=timeout)
        if proc.stdout:
            proc.stdout = proc.stdout.removeprefix(f"{EXEC_STARTED}\n")
        return proc

    def _copy_in(self, src: Path, dest: str) -> subprocess.CompletedProcess:
        return self._cp(str(src), f"{self.pod}:{dest}")

    def _copy_out(self, src: str, dest: Path) -> subprocess.CompletedProcess:
        return self._cp(f"{self.pod}:{src}", str(dest))

    def _copy_out_file(self, src: str,
                       dest: Path) -> subprocess.CompletedProcess:
        return self._cp(f"{self.pod}:{src}", str(dest))

    def _cp(self, source: str,
            destination: str) -> subprocess.CompletedProcess:
        container = ["-c", self.container] if self.container else []
        return auth_utils.run_kubectl(self.kube,
                                      self._kubectl_base() +
                                      ["cp", source, destination] + container,
                                      timeout=120)

    def _port_forward_command(
        self,
        local_port: int,
        remote_port: int,
        job_id: str | None,
    ) -> tuple[list[str], str]:
        """Tunnel to the configured pod; the driver already names its target."""
        auth_utils.resolve_credentials(
            self.kube,
            self._kubectl_base() + ["get", f"pod/{self.pod}", "-o", "name"])
        cmd = self._kubectl_base() + [
            "port-forward",
            f"pod/{self.pod}",
            f"{local_port}:{remote_port}",
        ]
        return cmd, f"{self.pod}-{local_port}"

    def _profile_details(self) -> dict[str, Any]:
        """Name the pod the probe numbers were read from."""
        return {"pod": self.pod, "namespace": self.namespace}

    def describe_target(self) -> str:
        where = (f"context '{self.kube_context}'"
                 if self.kube_context else "the current kubectl context")
        return (f"running pod '{self.pod}' in namespace '{self.namespace}' "
                f"of {where}")

    def _preflight_target(self, checks: list[base.PreflightCheck]) -> bool:
        """The cluster answers and the pod is Running."""
        if not auth_utils.cluster_check(
                self.kube, self._kubectl_base(), checks,
                namespace=self.namespace):
            return False
        proc = base.probe_run(
            self._kubectl_base() +
            ["get", "pod", self.pod, "-o", "jsonpath={.status.phase}"])
        phase = proc.stdout.strip() if proc.returncode == 0 else ""
        if phase != "Running":
            return base.fail(
                checks, "pod", f"pod '{self.pod}' is {phase}" if phase else
                f"could not read pod '{self.pod}': {base.probe_error(proc)}",
                "gke_pod attaches to a running pod; set pod and namespace "
                "to one `kubectl get pods -A` lists as Running")
        checks.append(base.PreflightCheck("pod", "ok", "Running"))
        return True

    def inspect(self) -> base.HardwareProfile:
        """Probe the pod's live hardware state with the shared host probe."""
        return self._remote_profile(timeout=30)
