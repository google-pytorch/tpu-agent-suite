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
"""The kubectl client the GKE drivers share."""

from __future__ import annotations

import json
import subprocess
from typing import Any

from . import auth_utils, base

# Timeout for one kubectl read, so an unreachable API server or a stalled
# auth plugin can't hang `job status` or `job list`.
KUBECTL_READ_TIMEOUT_SEC = 30


class KubectlError(base.ComputeConfigError):
    """A kubectl invocation failed.

    Subclasses ComputeConfigError so the CLI renders it as a structured
    payload, with kubectl's stderr in the message.
    """

    def __init__(self, argv: list[str], stderr: str) -> None:
        super().__init__(
            f"kubectl {' '.join(argv)} failed: {stderr.strip()}",
            error_code="KUBECTL_FAILED",
        )


class Kubectl:
    """kubectl, pointed at one kubeconfig, context and namespace.

    `auth_utils` sets `using_adc` once the cluster refuses gcloud but takes
    ADC, so later calls skip gcloud.
    """

    def __init__(self, kubeconfig: str | None, kube_context: str | None,
                 namespace: str) -> None:
        self.kubeconfig = kubeconfig
        # Pinned so `kubectl config use-context` can't move our Jobs.
        self.kube_context = kube_context
        self.namespace = namespace
        self.using_adc = False

    def run(
        self,
        argv: list[str],
        stdin: str | None = None,
        check: bool = True,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run one kubectl command with this client's kubeconfig and context."""
        try:
            proc = auth_utils.run_kubectl(self,
                                          self.cmd(argv),
                                          kubeconfig=self.kubeconfig,
                                          raise_refused=check,
                                          input=stdin,
                                          timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise KubectlError(argv, f"no answer within {timeout}s") from exc
        if check and proc.returncode != 0:
            raise KubectlError(argv, proc.stderr)
        return proc

    def cmd(self, argv: list[str]) -> list[str]:
        """The full kubectl command line for `argv`."""
        return base.kubectl_prefix(self.kubeconfig, self.kube_context) + argv

    def get_json(self, argv: list[str]) -> dict[str, Any]:
        """`kubectl <argv> -o json`, parsed; raises if it fails."""
        proc = self.run(argv + ["-o", "json"],
                        timeout=KUBECTL_READ_TIMEOUT_SEC)
        try:
            return json.loads(proc.stdout)
        except ValueError as exc:
            raise base.ComputeConfigError(
                f"kubectl {' '.join(argv)} returned unparseable JSON.",
                error_code="KUBECTL_BAD_OUTPUT",
            ) from exc

    def ns(self) -> list[str]:
        """The `-n <namespace>` arguments for a namespaced command."""
        return ["-n", self.namespace]

    def get_json_or_none(self, argv: list[str]) -> dict[str, Any] | None:
        """`kubectl <argv> -o json`, or None if it could not be read.

        For Kueue objects, where a failed read (no CRD, no access, timeout)
        must not fail the Job.
        """
        try:
            proc = self.run(argv + ["-o", "json"],
                            check=False,
                            timeout=KUBECTL_READ_TIMEOUT_SEC)
            payload = json.loads(proc.stdout) if proc.returncode == 0 else None
        except (KubectlError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None
