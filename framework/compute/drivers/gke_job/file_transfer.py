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
"""Moving a `gke_job` Job's input files in and its outputs back.

Inputs go through the ConfigMap or the staging bucket. Outputs come back
through the bucket or framed in the pod's log.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Callable, Sequence

from .. import base, gcs_staging, jobs, shell
from .gke_common import INPUT_KEY, NO_BUCKET, SUBMIT_URL_EXP_SEC, Path
from .log_retrieval import log_blob, read_logs

if TYPE_CHECKING:
    from ..kubectl import Kubectl
    from .gke_common import StagingClient, StagingPlan
    from .profile_settings import GkeJobConfig


def _why_not_uploaded(in_log: str | bool) -> str:
    """Why the pod sent its outputs to its log rather than to GCS."""
    if in_log == NO_BUCKET:
        return "no 'staging_bucket' was set for the pod to upload it to"
    if isinstance(in_log, str):
        return f"the pod could not upload it: signing its URL failed ({in_log})"
    # A record from before the reason was kept.
    return "the pod had no way to upload it"


def stage_files(
    config: GkeJobConfig,
    storage: StagingClient,
    clean_bucket: Callable[..., str],
    staging: StagingPlan,
    name: str,
    script: Path,
    inputs: Sequence[base.StagedFile],
    outputs: Sequence[base.StagedFile],
    url_exp: int,
) -> None:
    """Stage inputs and set up the outputs' return path in `staging`.

    Small inputs go in the ConfigMap and outputs fall back to the pod's
    log, so neither needs a bucket or signing credentials.
    """
    for staged in inputs:
        shell.require_file(staged.local, f"--input {staged.local}")
    size = sum(
        Path(path).stat().st_size
        for path in [script, *(staged.local for staged in inputs)])
    mount = size <= gcs_staging.CONFIGMAP_MAX_BYTES
    bucket = clean_bucket(needed=bool(inputs) and not mount)
    for index, staged in enumerate(inputs):
        if mount:
            key = INPUT_KEY.format(index)
            staging.configmap_files.append(f"--from-file={key}={staged.local}")
            staging.mounted_inputs.append((staged.remote, key))
            continue
        blob_name = f"{name}/in/{staged.remote}"
        staging.uploaded = True
        gcs_staging.upload_file_to_gcs(staged.local, bucket, blob_name,
                                       **storage.client_kwargs())
        staging.in_urls.append((staged.remote,
                                pod_url(config, storage, bucket, blob_name,
                                        "GET", url_exp)))
    if outputs and bucket:
        try:
            staging.out_urls = [(staged.remote,
                                 pod_url(config, storage, bucket,
                                         f"{name}/out/{staged.remote}", "PUT",
                                         url_exp)) for staged in outputs]
        except base.ComputeConfigError as exc:
            staging.logged_because = exc.error_code
            print(
                "gke_job: returning --output files through the pod's "
                f"log -- {exc.error_code}: could not sign upload URLs; "
                "outputs over "
                f"{gcs_staging.LOGGED_OUTPUT_MAX_BYTES // 1024} KiB in all "
                "will not come back (see `compute preflight`)",
                file=sys.stderr)
    if outputs and not staging.out_urls:
        staging.logged_outputs = [staged.remote for staged in outputs]
        staging.logged_because = staging.logged_because or NO_BUCKET


def pod_url(config: GkeJobConfig, storage: StagingClient, bucket: str,
            blob_name: str, method: str, expiration: int) -> str:
    """Signed URL for `blob_name`, or a plain one for Workload Identity."""
    if config.pod_service_account:
        return gcs_staging.object_url(bucket, blob_name)
    return gcs_staging.generate_v4_signed_url(bucket,
                                              blob_name,
                                              method=method,
                                              expiration_seconds=expiration,
                                              **storage.client_kwargs())


def log_upload_url(config: GkeJobConfig, storage: StagingClient,
                   clean_bucket: str, job_id: str) -> str | None:
    """Return a PUT URL for the pod's log copy, or None.

    Best-effort: the copy is only a backup, so failing to sign must not
    fail the Job.
    """
    if not clean_bucket:
        return None
    try:
        return pod_url(config, storage, clean_bucket, log_blob(job_id), "PUT",
                       SUBMIT_URL_EXP_SEC)
    except base.ComputeConfigError as exc:
        print(
            f"gke_job: not keeping a GCS copy of {job_id}'s log -- "
            f"{exc.error_code}: could not sign an upload URL",
            file=sys.stderr)
        return None


def fetch_job_artifacts(kube: Kubectl, config: GkeJobConfig,
                        storage: StagingClient,
                        not_configured: Callable[..., base.NotConfigured],
                        job_id: str, dest_dir: Path) -> list[Path]:
    """Write the Job's log and declared outputs to `dest_dir`.

    The log is always returned, since the pod's filesystem doesn't
    outlive it. Raises if declared outputs can't be retrieved.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / f"{job_id}.log"
    logs, carried = gcs_staging.split_logged_outputs(
        read_logs(kube, config, storage, job_id), job_id)
    target.write_text(logs, encoding="utf-8")
    retrieved = [target]
    record = jobs.JobRecord.load(job_id)
    declared = record.extra.get("outputs") if record else None
    if not declared:
        return retrieved
    in_log = record.extra.get("outputs_in_log")
    if not in_log and not config.staging_bucket:
        # The bucket was set at submit time but isn't now. Returning just
        # the log would look like the Job produced nothing else.
        raise not_configured(
            "staging_bucket",
            f"Job {job_id} staged its --output files to GCS: set the same "
            "bucket it was submitted with to retrieve them.")
    missing, fields = collect_outputs(
        config, storage, job_id, declared, carried, in_log,
        lambda record: base.StagedFile.artifact_path(dest, record))
    for out_spec in declared:
        if out_spec["remote"] in missing:
            print(
                f"gke_job: output {out_spec['remote']!r} of {job_id} not "
                f"retrieved -- {missing[out_spec['remote']]}",
                file=sys.stderr)
        else:
            retrieved.append(base.StagedFile.artifact_path(dest, out_spec))
    # Delete the staged outputs only once all are on disk. Anything not
    # fetched is the caller's only copy.
    if not in_log and not missing:
        gcs_staging.delete_staged_prefix(
            gcs_staging.clean_bucket_name(config.staging_bucket),
            f"{job_id}/out/", **storage.client_kwargs(True))
    if fields:
        # The pod is gone, so only a rerun that can upload recovers them.
        raise base.NotConfigured(
            f"Job {job_id} produced outputs too large for the pod's log, "
            "and the pod could not upload them, so they are lost; its "
            f"log is in {target}. Set up an upload path and rerun.",
            "STAGING_NOT_CONFIGURED", fields)
    return retrieved


def collect_outputs(
    config: GkeJobConfig,
    storage: StagingClient,
    job_id: str,
    declared: list[dict[str, str]],
    carried: dict[int, bytes | str],
    in_log: str | bool | None,
    dest_of: Callable[[dict[str, str]], Path],
) -> tuple[dict[str, str], list[str]]:
    """Save each declared output to `dest_of(output)`, from the log or GCS.

    `in_log` is why the pod put all outputs in its log instead of GCS (the
    record's `outputs_in_log`), or empty if it didn't. Returns the missing
    outputs with the reason for each, and the settings that would let the
    pod upload the ones lost in the log.
    """
    bucket = gcs_staging.clean_bucket_name(config.staging_bucket or "")
    missing: dict[str, str] = {}
    lost = False
    for index, out_spec in enumerate(declared):
        dest, remote = dest_of(out_spec), out_spec["remote"]
        # Check the log first: a pod whose upload failed frames it there.
        got = carried.get(index)
        if isinstance(got, bytes):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(got)
            continue
        if got and in_log:
            lost = True
            missing[remote] = f"{got}, and {_why_not_uploaded(in_log)}"
            continue
        if got:
            # Framed only because its upload failed; the log says why.
            missing[remote] = (f"{got}, after the pod failed to upload it "
                               f"to gs://{bucket} (see the job's log)")
            continue
        blob_name = f"{job_id}/out/{remote}"
        if in_log or not gcs_staging.download_file_from_gcs(
                bucket, blob_name, dest, **storage.client_kwargs()):
            missing[remote] = ("was not created" if in_log else
                               f"gs://{bucket}/{blob_name} was not created")
    if not lost:
        return missing, []
    return missing, (["staging_bucket"] if in_log == NO_BUCKET else list(
        gcs_staging.POD_UPLOAD_FIELDS))
