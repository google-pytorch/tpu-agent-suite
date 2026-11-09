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
"""kubectl credential helpers: spot a refusal and retry with ADC."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any, Callable

from . import base
from .contracts import PreflightCheck
from .errors import ClusterAccessDenied

# gcloud, and so gke-gcloud-auth-plugin, authenticates with this token instead
# of the active account. eval/run.py sets it the same way for ADC.
ACCESS_TOKEN_ENV = "CLOUDSDK_AUTH_ACCESS_TOKEN"

# kubectl's refusal prefixes, from StandardErrorMessage in kubectl's helpers.go.
REFUSED_PATTERN = re.compile(
    r"^(?:Error from server \(Forbidden\): "
    r"|error: You must be logged in to the server \()", re.MULTILINE)

# How a 403 names the identity the cluster saw.
REFUSED_USER = re.compile(r'User "([^"]+)"')

ADC = "application default credentials"

Refused = Callable[[subprocess.CompletedProcess[str]], bool]


def credential_refused(proc: subprocess.CompletedProcess[str]) -> bool:
    """True if kubectl failed because the cluster refused its credential."""
    return proc.returncode != 0 and bool(
        REFUSED_PATTERN.search(proc.stderr or ""))


def adc_token() -> tuple[str, str]:
    """An ADC access token and "", or "" and why there is none."""
    proc = base.probe_run(
        ["gcloud", "auth", "application-default", "print-access-token"])
    token = proc.stdout.strip() if proc.returncode == 0 else ""
    return token, "" if token else base.probe_error(proc)


def _set_token(token: str | None) -> None:
    if token is None:
        os.environ.pop(ACCESS_TOKEN_ENV, None)
    else:
        os.environ[ACCESS_TOKEN_ENV] = token


def retry_with_adc(
    driver: Any,
    run: Callable[[], subprocess.CompletedProcess[str]],
    refused: Refused = credential_refused
) -> tuple[subprocess.CompletedProcess[str] | None, str]:
    """Run `run` again as ADC, and keep ADC for this process if it gets in.

    Returns the retry (None if there is no ADC) and ADC named for an error.
    """
    token, why = adc_token()
    if not token:
        return None, f"{ADC} (none usable: {why})"
    previous = os.environ.get(ACCESS_TOKEN_ENV)
    _set_token(token)
    proc = run()
    if not refused(proc):
        driver.using_adc = True
        return proc, ADC
    _set_token(previous)
    user = REFUSED_USER.search(proc.stderr or "")
    return proc, f"{ADC} ({user.group(1)})" if user else ADC


def _grant_fix(account: str, context: str | None) -> str:
    """The command that grants `account` access to the cluster's project."""
    member = ("serviceAccount:" if account.endswith(".gserviceaccount.com")
              else "user:") + (account or "<account>")
    return (f"gcloud projects add-iam-policy-binding "
            f"{base.gke_target(context)['project']} --member={member} "
            "--role=roles/container.developer")


def access_denied(adc: str,
                  context: str | None = None,
                  kubeconfig: str | None = None) -> ClusterAccessDenied:
    """The error for a cluster that refused the gcloud account and ADC."""
    account = base.active_gcloud_account()
    identities = [
        f"the gcloud account {account}"
        if account else "gcloud (no active account)", adc
    ]
    fix = ("gcloud auth application-default login (as an account with access "
           "to the cluster); or grant the account access: " +
           _grant_fix(account, context) +
           "; or, if the kubeconfig's token expired, refresh it: " +
           base.get_credentials_fix(kubeconfig, context=context))
    return ClusterAccessDenied(identities, fix)


def cluster_check(driver: Any,
                  kubectl: list[str],
                  checks: list[PreflightCheck],
                  kubeconfig: str | None = None,
                  namespace: str = "default") -> bool:
    """The context, cluster and credential checks a GKE driver's preflight
    shares, appended to `checks`; True if the cluster answered.

    `kubectl` is the driver's command prefix, with the --kubeconfig (also
    passed as `kubeconfig`, so a missing one can be named) and the pinned
    --context it runs with. The credential checks run whatever happens,
    since they help explain a cluster that does not answer.
    """
    pinned = base.kube_context_check(kubeconfig, base.pinned_context(kubectl))
    if pinned:
        checks.append(pinned)
    if pinned and pinned.status == "fail":
        # A context kubectl cannot use leaves no cluster to ask.
        cluster = pinned
    else:
        cluster = _cluster(driver, kubectl, kubeconfig, namespace)
        checks.append(cluster)
    checks.extend(base.gke_credential_checks(cluster))
    checks.extend(adc_fallback_check(driver, kubectl))
    return cluster.status == "ok"


def _cluster(driver: Any, kubectl: list[str], kubeconfig: str | None,
             namespace: str) -> PreflightCheck:
    """Whether the cluster `kubectl` points at answers and lets this account
    in, retrying with ADC if it refuses the gcloud account."""
    name = base.current_context(kubectl)
    if not name:
        return base.missing_kubeconfig(kubeconfig) or PreflightCheck(
            "cluster", "fail",
            "kubectl has no current context: no cluster is configured",
            base.get_credentials_fix(kubeconfig))
    version = base.probe_run(
        kubectl + ["version", "-o", "json", "--request-timeout=10s"],
        timeout=20)
    answered = '"serverVersion"' in version.stdout
    if answered or credential_refused(version):
        # A cluster that answers can still refuse to list pods, so access
        # is checked with a request that needs it.
        pods = kubectl + [
            "get", "pods", "-n", namespace, "--limit=1", "-o", "name",
            "--request-timeout=10s"
        ]
        refused = base.probe_run(pods) if answered else version
        if credential_refused(refused):
            retry, adc = retry_with_adc(driver, lambda: base.probe_run(pods))
            if retry is not None and not credential_refused(retry):
                return PreflightCheck("cluster", "ok",
                                      f"context '{name}' answers with {ADC}")
            denied = access_denied(adc, name, kubeconfig)
            return PreflightCheck(
                "cluster", "fail", f"context '{name}' refused "
                f"{' and then '.join(denied.identities)}: "
                f"{base.probe_error(refused)}", denied.fix)
    if answered:
        return PreflightCheck("cluster", "ok", f"context '{name}' answers")
    if any(marker in (version.stderr or "").lower()
           for marker in base.NETWORK_TIMEOUT_MARKERS):
        # Credentials would have been refused, not left unanswered: the
        # request never reached the API server.
        return PreflightCheck(
            "cluster", "fail",
            f"context '{name}' does not answer from this machine: "
            f"{base.probe_error(version)}",
            "The API server is out of this machine's network reach. Common "
            "causes: its IP is not in the cluster's authorized networks, "
            "the cluster has only a private endpoint, or a VPN is needed. "
            "The DNS endpoint is authorized by IAM rather than IP: " +
            base.get_credentials_fix(kubeconfig, " --dns-endpoint", name) +
            " (enable it with gcloud container clusters update "
            f"{base.gke_target(name)['cluster']} "
            "--enable-dns-access)")
    return PreflightCheck(
        "cluster", "fail",
        f"context '{name}' does not answer: {base.probe_error(version)}",
        "Check the cluster exists and refresh credentials: " +
        base.get_credentials_fix(kubeconfig, context=name))


def adc_fallback_check(driver: Any,
                       kubectl: list[str]) -> list[PreflightCheck]:
    """A warning when the cluster refused the gcloud account but let ADC in."""
    if not getattr(driver, "using_adc", False):
        return []
    account = base.active_gcloud_account()
    context = base.current_context(kubectl) or None
    return [
        PreflightCheck(
            "credentials:adc", "warn",
            f"the cluster refused the gcloud account {account or '(none)'}, "
            f"so compute commands use {ADC}",
            "Nothing to do; to use the gcloud account instead: " +
            _grant_fix(account, context))
    ]


def run_kubectl(driver: Any,
                cmd: list[str],
                kubeconfig: str | None = None,
                raise_refused: bool = True,
                refused: Refused = credential_refused,
                **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run kubectl, retrying with ADC if refused; raise if ADC is refused too."""

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(cmd,
                              capture_output=True,
                              text=True,
                              check=False,
                              **kwargs)

    proc = run()
    if not refused(proc):
        return proc
    adc = ADC
    if not getattr(driver, "using_adc", False):
        retry, adc = retry_with_adc(driver, run, refused)
        if retry is not None:
            proc = retry
            if not refused(proc):
                return proc
    if not raise_refused:
        return proc
    context = base.current_context(
        base.kubectl_prefix(kubeconfig, base.pinned_context(cmd)))
    raise access_denied(adc, context or None, kubeconfig)


def resolve_credentials(driver: Any,
                        cmd: list[str],
                        kubeconfig: str | None = None) -> None:
    """Switch to ADC, or raise, before a kubectl that can't be retried."""
    try:
        run_kubectl(driver, cmd, kubeconfig=kubeconfig, timeout=30)
    except subprocess.TimeoutExpired:
        pass  # the command that follows reports the unreachable cluster
