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
"""The `gke_job` compute driver.

Runs, submits, names and places Jobs. Delegates the rest to `status`,
`records`, `file_transfer`, `log_retrieval` and `preflight`.
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from typing import TYPE_CHECKING, Any

from .. import auth_utils, base, gcs_staging, jobs, registry, shell
from ..kubectl import KUBECTL_READ_TIMEOUT_SEC, Kubectl, KubectlError
from .file_transfer import (collect_outputs, fetch_job_artifacts,
                            log_upload_url, stage_files)
from .gke_common import (DEFAULT_SHM_SIZE_GB, DEFAULT_WAIT_SEC, HOSTNAME_LABEL,
                         NEEDS_TPU_EXIT_CODE, SUBMIT_URL_EXP_SEC, Path,
                         StagingClient, StagingPlan, k8s_name)
from .log_retrieval import read_logs
from .manifest import (COMPILE_CACHE_MAX_AGE_DAYS, TOKEN_SECRET_SUFFIX,
                       fit_to_nodes, job_hosts, job_manifest, job_sizing,
                       node_selector, own_token_secret, put_token_secret,
                       token_bucket, token_prefixes)
from .preflight import target_checks
from .profile_settings import (MAX_JOB_NAME_LEN, GkeJobConfig, add_cli_flags,
                               config_from_params)
from .records import delete_job, read_and_record_status, settle_record
from .status import chips_in_use, job_pods, killed_message, pick_live_pod, wait

if TYPE_CHECKING:
    from .status import LastPoll

# Suffixed names tried when a job outside this run holds the requested one.
NAME_SUFFIX_ATTEMPTS = 3


@registry.register_driver("gke_job")
class GkeJobDriver(base.ComputeDriver):
    """Creates a Kubernetes Job on a TPU node and follows it to completion.

    Not built on `shell.RemoteShellDriver`, which assumes an existing host.
    Here the Job creates the host, and its lifecycle is the Job's.
    """

    required_tools = base.GKE_TOOLS
    scheduled_run = True
    runs_in_container = True
    CHECK_FIELDS = {
        "cluster": ("kubeconfig", "kube_context"),
        "kube_context": ("kube_context", ),
        "image": ("image", ),
        "namespace": ("namespace", ),
        "kueue:queue": ("queue", "namespace"),
        "kueue:priority": ("priority", ),
        "pod_service_account": ("pod_service_account", ),
        "hf_cache_pvc": ("hf_cache_pvc", ),
        "staging:bucket": ("staging_bucket", ),
        "staging:signing":
        gcs_staging.POD_UPLOAD_FIELDS,
        "staging:credentials":
        ("staging_service_account", "staging_service_account_key"),
    }
    SETUP_FIELDS = base.SetupFields(
        required=("image", "tpu_accelerator", "tpu_topology"),
        optional=("kube_context", "namespace", "queue", "hf_cache_pvc"),
        conditional={
            "priority": "queue",
        },
        on_demand={
            "staging_bucket":
            "staging files too large to send inline",
            "staging_service_account":
            "signing staging URLs as a service account you can impersonate",
            "staging_service_account_key":
            "signing staging URLs with a key file when impersonation is "
            "unavailable",
            "pod_service_account":
            "Workload Identity GCS access without signing",
            "declared_duration_minutes":
            "Kueue suspension enforcement",
            "kubeconfig":
            "a non-default kubeconfig path",
            "python_interpreter":
            "a non-default Python inside the container",
            "tpu_chips":
            "overriding the chip count derived from tpu_topology",
            "shm_size_gb":
            f"a /dev/shm larger or smaller than {DEFAULT_SHM_SIZE_GB} GiB",
            "compile_cache_max_age_days":
            "keeping unused compile caches on hf_cache_pvc other than "
            f"{COMPILE_CACHE_MAX_AGE_DAYS} days (0 never removes them)",
            "job_name_prefix":
            "a recognisable prefix on generated Job names",
            "ml_diagnostics_cluster":
            "uploading profiles to an ML Diagnostics dashboard",
            "ml_diagnostics_gcs":
            "telling ML Diagnostics which GCS folder to read profiles from",
        },
    )

    def __init__(self,
                 config: GkeJobConfig | None = None,
                 **settings: Any) -> None:
        """Take the settings as a `GkeJobConfig`, or as its fields."""
        self.config = config if config is not None else GkeJobConfig(
            **settings)
        if (self.config.staging_service_account
                and self.config.staging_service_account_key):
            gcs_staging.staging_credentials(
                self.config.staging_service_account,
                self.config.staging_service_account_key)
        self.kube = Kubectl(self.config.kubeconfig, self.config.kube_context,
                            self.config.namespace)
        self.storage = StagingClient(self.config.staging_service_account,
                                     self.config.staging_service_account_key)
        # `_wait`'s latest poll, reported if `run` is killed mid-wait.
        self._last_poll: LastPoll | None = None

    @property
    def using_adc(self) -> bool:
        """Whether the cluster refused the gcloud account but let ADC in."""
        return self.kube.using_adc

    # -- construction ------------------------------------------------------

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> GkeJobDriver:
        """Build a driver from profile params; see `config_from_params`."""
        return cls(config_from_params(params))

    @classmethod
    def register_cli_args(cls, parser: Any) -> None:
        add_cli_flags(parser, cls.add_cli_arg)

    def needed_tools(self) -> tuple[str, ...]:
        return base.gke_tools(self.kube.cmd([]))

    # -- inspect -----------------------------------------------------------

    def inspect(self) -> base.HardwareProfile:
        """Report the TPU node pool this driver would dispatch onto.

        Reads node labels, since scheduling keys on them.
        """
        selector = node_selector(self.config)
        nodes = self.kube.get_json(["get", "nodes", "-l", selector])
        items = nodes.get("items", [])
        if not items:
            details: dict[str, Any] = {"selector": selector}
            if self.config.tpu_accelerator:
                # Pool may be scaled to zero. Size the Job from the profile.
                details["tpu_accelerator"] = self.config.tpu_accelerator
                details["nodes_note"] = (
                    "no node matches the selector: the pool may be scaled to "
                    "zero, and a Job waits for one; job_chips, job_devices "
                    "and hbm_gb_per_device come from the profile")
                job_sizing(self.config, self.config.tpu_accelerator, details)
            return base.HardwareProfile(
                accelerator="none",
                topology="none",
                chips=0,
                hbm_gb=0.0,
                status="no_accelerator",
                available_chips=0,
                details=details,
            )

        found = base.tpu_nodes(items)
        accelerator = found.accelerator or "unknown"
        topology = found.topology or "unknown"
        chips = found.chips
        node_names = found.names

        fit_to_nodes(self.kube, self.config, items)

        held_by, used = chips_in_use(self.kube)
        # None if pods can't be listed: don't report unknown chips as free.
        available = None if used is None else max(chips - used, 0)

        details: dict[str, Any] = {
            "nodes": node_names,
            "namespace": self.config.namespace,
            "selector": selector,
            "tpu_accelerator": accelerator,
        }
        hbm_gb = job_sizing(self.config, accelerator, details)
        if self.config.queue:
            details["queue"] = self.config.queue
        if self.config.priority:
            details["priority"] = self.config.priority
        if self.config.declared_duration_minutes:
            details[
                "declared_duration_minutes"] = self.config.declared_duration_minutes

        status = "healthy"
        if available == 0:
            status = "busy"
        elif available is None:
            status = "unknown_contention"

        return base.HardwareProfile(
            accelerator=accelerator,
            topology=topology,
            chips=chips,
            hbm_gb=hbm_gb,
            status=status,
            available_chips=available,
            held_by=held_by,
            details=details,
        )

    def describe_target(self) -> str:
        where = (f"kubeconfig {self.config.kubeconfig}"
                 if self.config.kubeconfig else "the current kubectl context")
        if self.config.kube_context:
            where = f"context '{self.config.kube_context}'" + (
                f" of kubeconfig {self.config.kubeconfig}"
                if self.config.kubeconfig else "")
        queue = f", Kueue queue '{self.config.queue}'" if self.config.queue else ""
        return (f"GKE cluster of {where}, namespace '{self.config.namespace}'"
                f"{queue}")

    def _preflight_target(self, checks: list[base.PreflightCheck]) -> bool:
        """See `preflight.target_checks`."""
        return target_checks(self.kube, self.config, checks)

    # -- dispatch ----------------------------------------------------------

    def _clean_bucket(self, needed: bool) -> str:
        """Return the staging bucket. Raise only if `needed` and unset."""
        clean = (gcs_staging.clean_bucket_name(self.config.staging_bucket)
                 if self.config.staging_bucket else "")
        if needed and not clean:
            raise self.not_configured(
                "staging_bucket", "These --input files are over "
                f"{gcs_staging.CONFIGMAP_MAX_BYTES // 1024} KiB in total "
                "(script included), too large for the Job's ConfigMap.")
        return clean

    def run(
        self,
        code_path: str,
        timeout: int | None = None,
        args: list[str] | None = None,
        python_interpreter: str | None = None,
        inputs: list[base.StagedFile] | None = None,
        outputs: list[base.StagedFile] | None = None,
        job_name: str | None = None,
    ) -> base.SynchronousExecutionResult:
        """Execute a script and wait for its output.

        With `cpu_requests`, a run that never starts or exits
        NEEDS_TPU_EXIT_CODE is retried once on a TPU node.
        `details.cpu_fallback` records why.
        """
        kwargs = dict(code_path=code_path,
                      timeout=timeout,
                      args=args,
                      python_interpreter=python_interpreter,
                      inputs=inputs,
                      outputs=outputs,
                      job_name=job_name)
        result = self._run_once(**kwargs)
        if not self.config.cpu_requests:
            return result
        if result.details.get("never_started"):
            why = "no CPU node took it"
        elif result.exit_code == NEEDS_TPU_EXIT_CODE:
            why = (f"it exited {NEEDS_TPU_EXIT_CODE}: it needs a chip, "
                   "or could not do without one")
        else:
            return result
        print(
            f"gke_job: the CPU run {result.details.get('job_id')} did not "
            f"do the work ({why}); running it again on a TPU node.",
            file=sys.stderr,
            flush=True)
        cpu_requests, self.config.cpu_requests = self.config.cpu_requests, {}
        try:
            result = self._run_once(**kwargs)
        finally:
            self.config.cpu_requests = cpu_requests
        result.details["cpu_fallback"] = why
        return result

    def _run_once(
        self,
        code_path: str,
        timeout: int | None = None,
        args: list[str] | None = None,
        python_interpreter: str | None = None,
        inputs: list[base.StagedFile] | None = None,
        outputs: list[base.StagedFile] | None = None,
        job_name: str | None = None,
    ) -> base.SynchronousExecutionResult:
        """One `run`: execute a script and wait for its output.

        Reads the logs before deleting the Job, since the delete removes the
        pod and its logs.
        """
        outputs = outputs or []
        clean_bucket = self._clean_bucket(needed=False)

        # A requested name gets a random suffix so parallel runs don't collide.
        job_name = self._job_name(job_name, unique=True)
        # URLs must outlive the wait: queue plus run, up to twice timeout.
        job_id = self.submit(
            code_path,
            job_name=job_name,
            args=args,
            python_interpreter=python_interpreter
            or self.config.python_interpreter,
            # Lets the run-end reaper find it if this run is killed.
            run_id=self.run_id,
            inputs=inputs,
            outputs=outputs,
            url_exp_sec=min(2 * (timeout or DEFAULT_WAIT_SEC),
                            SUBMIT_URL_EXP_SEC),
        )
        # `is None`, not falsy: `--timeout 0` means don't wait.
        wait_for = DEFAULT_WAIT_SEC if timeout is None else timeout
        status: base.AsynchronousExecutionResult | None = None
        never_ran = False
        self._last_poll = None
        # On SIGTERM, report the Job's state and still run the delete.
        with shell.sigterm_raises():
            try:
                status = self._wait(job_id, wait_for)
                settle_record(self.kube, self.config, self.storage, job_id,
                              status)
                # No container ran, so there are no logs or outputs to read.
                never_ran = bool(status.details.get("never_started"))
                logs, carried = (("", {}) if never_ran else
                                 gcs_staging.split_logged_outputs(
                                     self._read_logs(job_id), job_id))
                stderr = status.error or ""
                if never_ran:
                    # The Job is deleted in `finally`.
                    stderr = f"{stderr} The Job has been deleted.".strip()
                result = base.SynchronousExecutionResult(
                    exit_code=status.exit_code if status.exit_code is not None
                    else (0 if status.status == "COMPLETED" else 1),
                    stdout=logs,
                    stderr=stderr,
                    details={
                        "job_id": job_id,
                        "job_status": status.status,
                        **status.details,
                    },
                )
                if never_ran:
                    for staged in outputs:
                        result = shell.missing_output(
                            result, staged, "the Job never ran, so it was "
                            "never written")
                    return result
                job = jobs.JobRecord.load(job_id)
                missing, fields = collect_outputs(
                    self.config, self.storage, job_id,
                    base.StagedFile.as_records(outputs), carried,
                    job.extra.get("outputs_in_log") if job else None,
                    lambda record: Path(record["local"]))
                for staged in outputs:
                    if staged.remote in missing:
                        result = shell.missing_output(result, staged,
                                                      missing[staged.remote])
                if fields:
                    result.details["setup_fields"] = fields
                return result
            except SystemExit:
                # Say what the Job was doing, to tell a full queue from a hang.
                print(killed_message(job_id, self._last_poll), file=sys.stderr)
                raise
            finally:
                # Always delete the Job so it can't hold a TPU node or start
                # later unwatched. If the delete fails, warn and return the
                # result anyway; the run-end reaper retries it.
                try:
                    delete_job(self.kube, self.config, self.storage, job_id,
                               status if never_ran else None)
                except base.ComputeConfigError as exc:
                    print(f"gke_job: could not delete Job {job_id} -- {exc}",
                          file=sys.stderr)
                if clean_bucket:
                    gcs_staging.delete_staged_prefix(
                        clean_bucket, f"{job_name}/",
                        **self.storage.client_kwargs(True))

    def submit(
        self,
        recipe_path: str,
        job_name: str | None = None,
        args: list[str] | None = None,
        python_interpreter: str | None = None,
        run_id: str | None = None,
        inputs: list[base.StagedFile] | None = None,
        outputs: list[base.StagedFile] | None = None,
        serve: bool = False,
        url_exp_sec: int = SUBMIT_URL_EXP_SEC,
    ) -> str:
        """Dispatch the script as a Job and return its name as the job id.

        The script ships in a ConfigMap mounted into the stock image, so no
        image is built. The ConfigMap shares the Job's name.
        """
        self.config.require_dispatch()
        # Check first so a missing script raises not-found, not OSError.
        shell.require_file(recipe_path, "recipe_path")
        recipe = Path(recipe_path)
        name = self._job_name(job_name, unique=False)
        # By extension. A shell recipe gets Python as TPU_AGENT_SUITE_PYTHON.
        python = python_interpreter or self.config.python_interpreter
        shell_recipe = recipe.suffix == ".sh"
        interpreter = "bash" if shell_recipe else python

        inputs, outputs = inputs or [], outputs or []
        clean_bucket = self._clean_bucket(needed=False)
        staging = StagingPlan()
        configmap = False
        job_applied = False
        token_expires: float | None = None
        # Before staging, so cleanup can't delete another job's files.
        name = self._unused_job_name(name, job_name, run_id)
        if token_bucket(self.config) is not None:
            # Refuse a name whose folder overlaps the checkpoints.
            token_prefixes(self.config, name)
        placement = None
        if not self.config.cpu_requests and not self.config.colocate_with:
            fit_to_nodes(self.kube, self.config)
        if self.config.colocate_with:
            placement = self._placement_of(self.config.colocate_with)
            # Give the server an address now. A --serve one already has it.
            self._expose(self.config.colocate_with)
        hosts = job_hosts(self.config, placement)

        try:
            # Inside the try, so a failure cleans up inputs already uploaded.
            stage_files(self.config, self.storage, self._clean_bucket, staging,
                        name, recipe, inputs, outputs, url_exp_sec)
            log_url = log_upload_url(self.config, self.storage, clean_bucket,
                                     name)
            configmap = True
            try:
                self.kube.run(["create", "configmap", name] + self.kube.ns() +
                              [f"--from-file={recipe.name}={recipe_path}"] +
                              staging.configmap_files)
            except KubectlError as exc:
                # Taken since the check, so don't clean up another job's files.
                if re.search(r"already ?exists", exc.message, re.IGNORECASE):
                    configmap = staging.uploaded = False
                raise
            token_expires = put_token_secret(self.kube, self.config, name)
            manifest = job_manifest(
                self.config,
                name=name,
                script_name=recipe.name,
                interpreter=interpreter,
                args=args or [],
                staging=staging,
                log_url=log_url,
                serve=serve,
                placement=placement,
                shell_python=python if shell_recipe else None,
                token_secret=token_expires is not None,
            )
            # JSON, not templated YAML, so odd values can't break the
            # document. The UID tells apart Jobs that reuse a name.
            uid = self.kube.run(
                ["apply", "-f", "-", "-o", "jsonpath={.metadata.uid}"],
                stdin=json.dumps(manifest)).stdout.strip()
            job_applied = True
            extra: dict[str, Any] = {}
            if token_expires is not None:
                # Lets cancel and renewal skip Secrets for tokenless Jobs.
                extra["token_secret"] = True
                extra["token_expires"] = token_expires
            if uid:
                extra["uid"] = uid
                if token_expires is not None:
                    own_token_secret(self.kube, name, uid)
            if staging.in_urls or staging.out_urls:
                extra["staged_prefix"] = f"{name}/"
            if outputs:
                extra["outputs"] = base.StagedFile.as_records(outputs)
            if staging.logged_outputs:
                extra["outputs_in_log"] = staging.logged_because
            if hosts > 1:
                extra["hosts"] = hosts
                # Its pods find each other through the Service.
                self._expose(name, uid, required=True)
            self._new_record(
                job_id=name,
                job_name=job_name or name,
                recipe_path=str(recipe),
                run_id=run_id or "",
                backend="gke_job",
                status="PENDING",
                serve=serve,
                extra=extra,
            )
        except Exception:
            # Undo what was created: no record points at it yet, and a leftover
            # ConfigMap would make a retry under the same name fail.
            if job_applied:
                self.kube.run(["delete", "job", name] + self.kube.ns() +
                              ["--ignore-not-found"],
                              check=False)
            if configmap:
                self.kube.run(["delete", "configmap", name] + self.kube.ns() +
                              ["--ignore-not-found"],
                              check=False)
            if token_expires is not None:
                self.kube.run(
                    ["delete", "secret", name + TOKEN_SECRET_SUFFIX] +
                    self.kube.ns() + ["--ignore-not-found"],
                    check=False)
            if staging.uploaded:
                gcs_staging.delete_staged_prefix(
                    clean_bucket, f"{name}/",
                    **self.storage.client_kwargs(True))
            raise
        # Outside the cleanup: a single-host --serve Job is already reachable
        # by pod IP, so a Service failure must not undo it.
        if serve and hosts == 1:
            self._expose(name, uid)
        return name

    def _unused_job_name(self, name: str, requested: str | None,
                         run_id: str | None) -> str:
        """`name`, or `name` with a random suffix if another run holds it.

        Raises NAME_TAKEN if this run's live job holds it."""
        if self._name_unused(name):
            return name
        # Cleanup and trimming can map two --name values to one Job name.
        asked = (f" (from --name {requested!r})"
                 if requested not in (None, name) else "")
        mine = jobs.JobRecord.load(name)
        if (mine is not None and run_id and mine.run_id == run_id
                and mine.status not in jobs.TERMINAL_STATUSES):
            raise base.ComputeConfigError(
                f"Job name {name!r}{asked} is already this run's job, still "
                f"{mine.status}. Wait for it with `job wait --job-id {name}`, "
                f"cancel it first with `job cancel --job-id {name}`, or "
                "choose a different --name.",
                error_code="NAME_TAKEN",
            )
        for _ in range(NAME_SUFFIX_ATTEMPTS):
            candidate = self._job_name(name, unique=True)
            if self._name_unused(candidate):
                print(
                    f"gke_job: Job name {name!r}{asked} is used by a job "
                    f"outside this run in namespace {self.config.namespace!r}; "
                    f"submitting as {candidate!r}.",
                    file=sys.stderr)
                return candidate
        raise base.ComputeConfigError(
            f"Job name {name!r}{asked} and {NAME_SUFFIX_ATTEMPTS} suffixed "
            f"variants are all in use in namespace {self.config.namespace!r}. "
            "Choose a different --name. Don't cancel or delete jobs this run "
            "didn't submit.",
            error_code="NAME_TAKEN",
        )

    def _name_unused(self, name: str) -> bool:
        """Whether no Job or ConfigMap called `name` exists.

        A failed lookup returns True; the ConfigMap create catches clashes."""
        try:
            proc = self.kube.run(["get", "configmap,job", name] +
                                 self.kube.ns() +
                                 ["--ignore-not-found", "-o", "name"],
                                 check=False,
                                 timeout=KUBECTL_READ_TIMEOUT_SEC)
        except KubectlError:
            return True
        return proc.returncode != 0 or not proc.stdout.strip()

    def _placement_of(self, job_id: str) -> dict[str, Any]:
        """The server pod's node, nodeSelector and tolerations.

        Copying them lands the client on the server's node and makes Kueue
        admit it under the server's TPU flavor, not a CPU pool's.
        """
        spec = (pick_live_pod(job_pods(self.kube, job_id))
                or {}).get("spec", {})
        if not spec.get("nodeName"):
            raise base.ComputeConfigError(
                f"--colocate-with {job_id!r}: no scheduled pod. Wait for it "
                f"with `job wait --job-id {job_id} --until running`.",
                error_code="INVALID_FIELD",
            )
        return {
            "nodeSelector": {
                **spec.get("nodeSelector", {}),
                HOSTNAME_LABEL: spec["nodeName"],
            },
            "tolerations": spec.get("tolerations", []),
        }

    def _expose(self,
                name: str,
                uid: str = "",
                required: bool = False) -> None:
        """Give a Job the address `<name>.<namespace>.svc`.

        Via a headless Service the Job owns. Best effort unless `required`:
        a multi-host Job's pods resolve `TPU_WORKER_HOSTNAMES` through it.
        Only --serve, not the address, exempts the Job from reaping.
        """
        try:
            uid = uid or self.kube.run(["get", "job", name] + self.kube.ns() +
                                       ["-o", "jsonpath={.metadata.uid}"],
                                       timeout=KUBECTL_READ_TIMEOUT_SEC).stdout
            self.kube.run(
                ["apply", "-f", "-"],
                stdin=json.dumps({
                    "apiVersion": "v1",
                    "kind": "Service",
                    "metadata": {
                        "name":
                        name,
                        "namespace":
                        self.config.namespace,
                        "ownerReferences": [{
                            "apiVersion": "batch/v1",
                            "kind": "Job",
                            "name": name,
                            "uid": uid.strip(),
                        }],
                    },
                    "spec": {
                        "clusterIP": "None",
                        # Workers look up a still-starting leader.
                        "publishNotReadyAddresses": True,
                        "selector": {
                            "job-name": name
                        }
                    },
                }),
                timeout=KUBECTL_READ_TIMEOUT_SEC)
        # An RBAC 403 surfaces as ClusterAccessDenied, not KubectlError.
        except base.ComputeConfigError as exc:
            if required:
                raise
            print(f"warning: no Service for {name}: {exc}", file=sys.stderr)

    def _job_name(self, requested: str | None, unique: bool) -> str:
        """`requested` (or a fresh job id) as a valid, prefixed Job name.

        With `unique`, a requested name gets a random suffix. Trimmed to
        MAX_JOB_NAME_LEN. Idempotent, so `run` can pass its chosen name on to
        `submit`.
        """
        name = k8s_name(requested or "")
        if requested and not name:
            raise base.ComputeConfigError(
                f"Job name needs at least one letter or digit, got "
                f"{requested!r}.",
                error_code="INVALID_FIELD",
            )
        suffix = f"-{uuid.uuid4().hex[:6]}" if name and unique else ""
        name = name or jobs.new_job_id()
        prefix = self.config.job_name_prefix
        if prefix and name != prefix and not name.startswith(f"{prefix}-"):
            name = f"{prefix}-{name}"
        return name[:MAX_JOB_NAME_LEN - len(suffix)].rstrip("-") + suffix

    # -- job lifecycle -----------------------------------------------------

    def get_status(self, job_id: str) -> base.AsynchronousExecutionResult:
        """See `records.read_and_record_status`."""
        return read_and_record_status(self.kube, self.config, self.storage,
                                      job_id)

    def _wait(self, job_id: str,
              timeout: int) -> base.AsynchronousExecutionResult:
        """`status.wait`, saving each poll in `self._last_poll`."""
        return wait(self.get_status, job_id, timeout, self._keep_poll)

    def _keep_poll(self, poll: LastPoll) -> None:
        self._last_poll = poll

    def fetch_logs(self, job_id: str) -> str:
        """Return the Job's pod output, without any outputs framed in it."""
        return gcs_staging.split_logged_outputs(self._read_logs(job_id),
                                                job_id)[0]

    def _read_logs(self, job_id: str) -> str:
        """See `log_retrieval.read_logs`."""
        return read_logs(self.kube, self.config, self.storage, job_id)

    def fetch_artifacts(self, job_id: str, dest_dir: Path) -> list[Path]:
        """See `file_transfer.fetch_job_artifacts`."""
        return fetch_job_artifacts(self.kube, self.config, self.storage,
                                   self.not_configured, job_id, dest_dir)

    def _port_forward_command(
        self,
        local_port: int,
        remote_port: int,
        job_id: str | None,
    ) -> tuple[list[str], str]:
        """Tunnel to the Job's running pod, falling back to the Job itself.

        Naming the pod stops kubectl from picking a dead pod of a retried Job.
        """
        if not job_id:
            raise base.ComputeConfigError(
                "Backend 'gke_job' requires '--job-id' for port-forwarding.",
                error_code="MISSING_FIELD",
            )
        auth_utils.resolve_credentials(
            self.kube,
            self.kube.cmd(["get", "job", job_id, "-o", "name"] +
                          self.kube.ns()), self.config.kubeconfig)
        target = f"job/{job_id}"
        live_pod = pick_live_pod(job_pods(self.kube, job_id))
        if live_pod and live_pod.get("status", {}).get("phase") == "Running":
            pod_name = live_pod.get("metadata", {}).get("name")
            if pod_name:
                target = f"pod/{pod_name}"
        cmd = self.kube.cmd(
            self.kube.ns() +
            ["port-forward", target, f"{local_port}:{remote_port}"])
        return cmd, f"{job_id}-{local_port}"

    def cancel(self, job_id: str) -> None:
        """Delete the Job and its ConfigMap, and record what that meant."""
        delete_job(self.kube, self.config, self.storage, job_id)
