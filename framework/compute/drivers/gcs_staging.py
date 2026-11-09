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
"""GCS file staging and V4 signed URL helper for GKE compute drivers.

Workload pods running on GKE (e.g. `gke_job`) lack embedded GCP
credentials in user container images. This module enables file staging by:
1. Uploading input files to GCS via the controller's ambient credentials.
2. Generating V4 signed GET and PUT URLs with bounded expiration.
3. Constructing bootstrap commands that use `curl` inside the pod to download
   inputs before running the script and upload outputs after execution.
4. Downloading declared outputs back to the local host.

Small files skip GCS, since signing needs credentials a person's own login
does not have: inputs that fit ride in the Job's ConfigMap, and outputs come
back framed in the pod's log (`split_logged_outputs`).

With `pod_service_account`, nothing is signed: the pod runs as that
Kubernetes service account, gets a token from the GKE metadata server
(Workload Identity), and sends it with the same `curl` calls to plain object
URLs (`object_url`). The host still uploads and downloads with its own
credentials, which a person's login can do; it just cannot sign.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import datetime
import functools
import hashlib
import pathlib
import re
import shlex
import urllib.parse
from typing import Any, Callable, Iterator, Sequence

from .. import env_check
from . import base

Path = pathlib.Path

DEFAULT_URL_EXPIRATION_SECONDS = 3600
STAGING_CONTENT_TYPE = "application/octet-stream"
# How long each HTTP call made while minting a pod token may take.
MINT_TIMEOUT_SEC = 30
# The environment.json keys that give staging a credential of its own.
SERVICE_ACCOUNT_PARAM = "staging_service_account"
KEY_FILE_PARAM = "staging_service_account_key"
# The environment.json key naming the Kubernetes service account the pod runs
# as, so it moves staged files with a Workload Identity token, unsigned.
POD_SERVICE_ACCOUNT_PARAM = "pod_service_account"
# Any one of these lets the pod reach the bucket when this host's own
# credentials cannot sign.
POD_UPLOAD_FIELDS = (POD_SERVICE_ACCOUNT_PARAM, SERVICE_ACCOUNT_PARAM,
                     KEY_FILE_PARAM)
# Where a pod on a Workload Identity node pool gets its Google token.
POD_TOKEN_URL = ("http://metadata.google.internal/computeMetadata/v1/"
                 "instance/service-accounts/default/token")
GCS_OBJECT_ENDPOINT = "https://storage.googleapis.com"
# Where a staged run's files live inside the pod, and its working directory:
# the shell drivers give a staged script a fresh directory so it reads and
# writes by bare name, and this keeps that contract on a throwaway pod whose
# image WORKDIR may be read-only or hold the image's own software.
POD_WORK_DIR = "/tmp/tpu_agent_suite_run"
# The workload's output, teed in the pod and uploaded to `<job>/out/` under
# this name, dot-prefixed so it cannot collide with a declared output.
POD_LOG_FILE = ".compute-output.log"
# Where the workload's output reaches `tee`, and where an output is copied
# before it is sent while the workload runs; dot-prefixed for the same reason.
LOG_FIFO = ".compute-output.fifo"
LIVE_COPY = ".compute-live-output"
LIVE_STOP = ".compute-live-stop"
# A ConfigMap holds at most 1 MiB, the script included; the rest is headroom
# for the object's own metadata.
CONFIGMAP_MAX_BYTES = 900 * 1024
# Outputs share the pod's log with the workload's own output, which kubelet
# rotates at 10 MiB by default, and base64 grows them by a third.
LOGGED_OUTPUT_MAX_BYTES = 1024 * 1024
# While the workload runs, the pod looks every second for a declared output
# whose file changed and sends a fresh copy of it, each file at most this
# often, so a pod lost before it ends (its node reclaimed or reset) still
# returns the outputs as they last were. A copy for
# the log is sent only up to the first size, and all copies only up to the
# second: the final copies, sent after the workload ends, matter more.
LIVE_OUTPUT_INTERVAL_SEC = 30
LIVE_OUTPUT_MAX_BYTES = 256 * 1024
LIVE_OUTPUT_BUDGET_BYTES = 4 * 1024 * 1024
# Marks an output's frame in the pod's log. It names the Job, so a workload
# that prints another run's log -- its own frames do not exist while it runs
# -- cannot open or close one; and a frame is only accepted whole and
# matching its sha256.
OUTPUT_FRAME = "[compute-output:{job}]"
# What a frame's body is made of; any other line in one was written to the
# log by something else meanwhile, and is left in the log.
_BASE64_LINE = re.compile(r"[A-Za-z0-9+/=]*")


def output_frame(job: str) -> str:
    """The marker of `job`'s output frames."""
    return OUTPUT_FRAME.format(job=job)


def clean_bucket_name(bucket: str) -> str:
    """Normalize bucket URI by removing 'gs://' prefix and trailing slashes."""
    cleaned = (bucket or "").strip()
    if cleaned.startswith("gs://"):
        cleaned = cleaned[5:]
    return cleaned.rstrip("/")


def locate(bucket: str, blob_name: str) -> tuple[str, str]:
    """Split `bucket[/folder]` into the bucket and the blob's full name."""
    name, _, folder = clean_bucket_name(bucket).partition("/")
    return name, f"{folder}/{blob_name}" if folder else blob_name


def object_url(bucket: str, blob_name: str) -> str:
    """The unsigned URL of a blob, for a pod that sends its own token.

    GCS's XML API: GET and PUT on it work as on a signed URL, so the pod's
    `curl` calls only gain an Authorization header.
    """
    name, full_blob = locate(bucket, blob_name)
    return (f"{GCS_OBJECT_ENDPOINT}/{name}/"
            f"{urllib.parse.quote(full_blob, safe='/')}")


@contextlib.contextmanager
def _gcs_errors(action: str, uri: str, client: Any = None) -> Iterator[None]:
    """Report a refused GCS request as a ComputeConfigError, not a traceback."""
    try:
        yield
    except Exception as exc:
        if _is_auth_error(exc):
            imp = _impersonated(_credentials_of(client)) if client else None
            if imp:
                raise base.ComputeConfigError(
                    f"Could not {action} {uri} acting as '{imp}': {exc}. "
                    "To act as it, your own account (this host's Application "
                    "Default Credentials) needs "
                    f"'roles/iam.serviceAccountTokenCreator' on {imp}.",
                    error_code="GCS_FORBIDDEN") from exc
            raise base.ComputeConfigError(f"Could not {action} {uri}: {exc}",
                                          error_code="NO_CREDENTIALS") from exc
        try:
            from google.api_core import exceptions  # noqa: PLC0415
        except ImportError:
            raise exc from None
        if not isinstance(exc, exceptions.GoogleAPICallError):
            raise
        code, hint = {
            403: ("GCS_FORBIDDEN",
                  " The credentials on this host lack access to the staging "
                  "bucket."),
            404: ("GCS_NOT_FOUND",
                  " Check that 'staging_bucket' names an existing bucket, "
                  "optionally with a folder in it (gs://bucket/folder)."),
        }.get(exc.code, ("GCS_ERROR", ""))
        raise base.ComputeConfigError(
            f"Could not {action} {uri}: {exc.message}.{hint}",
            error_code=code) from exc


def _is_auth_error(exc: Exception) -> bool:
    """Whether `exc` is google-auth's: no ADC, or a token that will not refresh.

    Callers fall back on a ComputeConfigError (outputs to the log, no log
    copy); these would otherwise pass them by as a traceback.
    """
    try:
        from google.auth import exceptions  # noqa: PLC0415
    except ImportError:
        return False
    return isinstance(exc, exceptions.GoogleAuthError)


def get_storage_client(project: str | None = None,
                       credentials: Any = None) -> Any:
    """Instantiate a Google Cloud Storage client.

    `credentials` is what `staging_credentials` built; None uses ADC.
    """
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise base.ComputeConfigError(
            "GCS staging needs google-cloud-storage. " +
            env_check.ask_to_install(["google-cloud-storage"]),
            error_code="MISSING_DEPENDENCY",
        ) from exc
    try:
        return storage.Client(project=project, credentials=credentials)
    except Exception as exc:
        if not _is_auth_error(exc):
            raise
        raise base.ComputeConfigError(
            f"No Application Default Credentials to reach GCS with: {exc}",
            error_code="NO_CREDENTIALS",
        ) from exc


# Signing is the one thing here that plain authentication does not cover. The
# other drivers shell out to `gcloud` and `kubectl`, which carry their
# own auth; this module talks to GCS through the client library, and a signed
# URL needs either a private key on this host or a service account the IAM
# Credentials API will sign for. Which applies depends on how the credentials
# were obtained:
#
# - a service-account key file, or impersonated credentials (`gcloud auth
#   application-default login --impersonate-service-account=...`), sign
#   locally;
# - a GCE/GKE host's ambient service account has no key, so it signs through
#   IAM `signBlob`, which needs `roles/iam.serviceAccountTokenCreator` on
#   itself;
# - a person's own credentials (`gcloud auth application-default login`) have
#   neither, and no error from the library says so usefully.
#
# Ambient credentials are process-wide, though, and kubectl inherits the same
# environment: pointing $GOOGLE_APPLICATION_CREDENTIALS at a key to make
# signing work also changes who kubectl is (base.KUBECTL_IDENTITY_WARNING).
# `staging_credentials` builds a credential only this module uses, from
# environment.json, so staging can sign as a service account while kubectl
# keeps the user's identity.


def staging_credentials(service_account: str | None = None,
                        key_file: str | None = None) -> Any | None:
    """The credential staging uses instead of ADC, or None to use ADC.

    `service_account` is one the host's ADC may act as (Google calls this
    impersonation; it needs roles/iam.serviceAccountTokenCreator on it);
    `key_file` is a service-account key. Either signs, and neither touches
    the process environment.
    """
    service_account = (service_account or "").strip() or None
    key_file = (key_file or "").strip() or None
    if service_account and key_file:
        raise base.ComputeConfigError(
            f"Set at most one of '{SERVICE_ACCOUNT_PARAM}' and "
            f"'{KEY_FILE_PARAM}'.",
            error_code="INVALID_FIELD",
        )
    if not service_account and not key_file:
        return None
    # A key path that isn't there is wrong whatever is installed, so say so
    # before asking for google-auth: fixing the dependency first would only
    # lead to this error next.
    key_path = Path(key_file).expanduser() if key_file else None
    if key_path is not None and not key_path.is_file():
        raise base.ComputeConfigError(
            f"'{KEY_FILE_PARAM}' names {key_file}, which is not a file.",
            error_code="INVALID_FIELD",
        )
    try:
        import google.auth  # noqa: PLC0415 - optional dependency
        from google.auth import impersonated_credentials  # noqa: PLC0415
        from google.oauth2 import service_account as sa_key  # noqa: PLC0415
    except ImportError as exc:
        raise base.ComputeConfigError(
            "A staging credential needs google-auth, which "
            "google-cloud-storage brings. " +
            env_check.ask_to_install(["google-cloud-storage"]),
            error_code="MISSING_DEPENDENCY",
        ) from exc
    if key_path is not None:
        try:
            return sa_key.Credentials.from_service_account_file(
                str(key_path), scopes=[base.CLOUD_PLATFORM_SCOPE])
        except (OSError, ValueError) as exc:
            raise base.ComputeConfigError(
                f"'{KEY_FILE_PARAM}' names {key_file}, which is not a "
                f"readable service-account key: {exc}",
                error_code="INVALID_FIELD",
            ) from exc
    try:
        source, _ = google.auth.default(scopes=[base.CLOUD_PLATFORM_SCOPE])
    except Exception as exc:
        if not _is_auth_error(exc):
            raise
        raise base.ComputeConfigError(
            f"'{SERVICE_ACCOUNT_PARAM}' acts as {service_account} using this "
            f"host's Application Default Credentials, and there are none: "
            f"{exc}. Run `gcloud auth application-default login`.",
            error_code="NO_CREDENTIALS",
        ) from exc
    return impersonated_credentials.Credentials(
        source_credentials=source,
        target_principal=service_account,
        target_scopes=[base.CLOUD_PLATFORM_SCOPE],
    )


def _cel_string(value: str) -> str:
    """`value` as a single-quoted CEL string literal."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def prefix_condition(bucket: str, prefix: str) -> str:
    """A CEL condition true only for objects under `prefix` in `bucket`, and
    for listings whose prefix is under it.

    A listing is checked against the bucket, not an object, so
    `resource.name` alone would refuse it; its prefix is in the
    `objectListPrefix` attribute instead (empty for other requests).
    """
    if not prefix or not prefix.endswith("/"):
        # Without the slash, `job-1` would also match `job-10/`.
        raise ValueError(f"a token prefix must end in '/', got {prefix!r}")
    objects = _cel_string(f"projects/_/buckets/{bucket}/objects/{prefix}")
    return (f"resource.name.startsWith({objects}) || "
            f"api.getAttribute('storage.googleapis.com/objectListPrefix', '')"
            f".startsWith({_cel_string(prefix)})")


def pod_token_rules(
        bucket: str,
        read_write_prefix: str,
        read_only_prefixes: Sequence[str] = (),
) -> list[tuple[str, str]]:
    """The (role, availability condition) pairs a pod token is limited to.

    Read, write, list and delete under `read_write_prefix`; read and list
    under each of `read_only_prefixes`; nothing else in `bucket` or in any
    other bucket.
    """
    rules = [("roles/storage.objectUser",
              prefix_condition(bucket, read_write_prefix))]
    rules += [("roles/storage.objectViewer", prefix_condition(bucket, prefix))
              for prefix in read_only_prefixes]
    return rules


def mint_bucket_token(
        bucket: str,
        read_write_prefix: str,
        read_only_prefixes: Sequence[str] = (),
) -> tuple[str, float]:
    """Mint a token limited to two kinds of folder in `bucket`.

    It can read, write, list and delete objects under `read_write_prefix`,
    and only read and list them under each of `read_only_prefixes` (see
    `pod_token_rules`). Uses this host's ADC, downscoped with a Credential
    Access Boundary. Returns the token and its expiry (epoch seconds),
    usually an hour out.
    """
    rules = pod_token_rules(bucket, read_write_prefix, read_only_prefixes)
    try:
        import google.auth  # noqa: PLC0415 - optional dependency
        import google.auth.transport.requests  # noqa: PLC0415
        from google.auth import downscoped  # noqa: PLC0415
    except ImportError as exc:
        raise base.ComputeConfigError(
            "Creating a GCS token for the pod needs google-auth, which "
            "comes with google-cloud-storage. " +
            env_check.ask_to_install(["google-cloud-storage"]),
            error_code="MISSING_DEPENDENCY",
        ) from exc
    boundary = downscoped.CredentialAccessBoundary(rules=[
        downscoped.AccessBoundaryRule(
            available_resource=(
                f"//storage.googleapis.com/projects/_/buckets/{bucket}"),
            available_permissions=[f"inRole:{role}"],
            availability_condition=downscoped.AvailabilityCondition(
                expression=condition),
        ) for role, condition in rules
    ])
    request = functools.partial(google.auth.transport.requests.Request(),
                                timeout=MINT_TIMEOUT_SEC)
    try:
        source, _ = google.auth.default(scopes=[base.CLOUD_PLATFORM_SCOPE])
        creds = downscoped.Credentials(source_credentials=source,
                                       credential_access_boundary=boundary)
        creds.refresh(request)
    except Exception as exc:
        if not _is_auth_error(exc):
            raise
        raise base.ComputeConfigError(
            f"Couldn't create a GCS token for the pod: {exc}. If your "
            "Google Cloud login has expired, run "
            "`gcloud auth application-default login`.",
            error_code="NO_CREDENTIALS",
        ) from exc
    # `expiry` is a naive datetime in UTC.
    if creds.expiry is None:
        expires = (datetime.datetime.now(datetime.timezone.utc).timestamp() +
                   DEFAULT_URL_EXPIRATION_SECONDS)
    else:
        expires = creds.expiry.replace(
            tzinfo=datetime.timezone.utc).timestamp()
    return creds.token, expires


def _impersonated(credentials: Any) -> str | None:
    """The service account `credentials` impersonate, if they do."""
    if type(credentials).__module__.endswith("impersonated_credentials"):
        return getattr(credentials, "signer_email", None)
    return None


def _credentials_of(client: Any) -> Any:
    creds = getattr(client, "_credentials", None)
    if creds is not None:
        return creds
    import google.auth  # noqa: PLC0415 - optional dependency, imported lazily
    creds, _ = google.auth.default()
    return creds


def _signing_kwargs(credentials: Any) -> dict[str, Any]:
    """Extra `generate_signed_url` arguments these credentials need.

    Empty when they sign locally; the service-account email and token when
    the IAM Credentials API has to sign for them. Raises for credentials
    that can do neither.
    """
    # Anything carrying its own signer: a key file or impersonation.
    if callable(getattr(credentials, "sign_bytes", None)) and getattr(
            credentials, "signer_email", None):
        return {}

    # A GCE/GKE ambient identity: an email (once refreshed) and a token, no
    # key. The IAM Credentials API signs on its behalf.
    email = getattr(credentials, "service_account_email", None)
    if email is not None:
        if not getattr(credentials, "token", None) or email == "default":
            import google.auth.transport.requests  # noqa: PLC0415
            credentials.refresh(google.auth.transport.requests.Request())
            email = getattr(credentials, "service_account_email", None)
        if email and email != "default":
            return {
                "service_account_email": email,
                "access_token": getattr(credentials, "token", None),
            }

    raise base.ComputeConfigError(
        "GCS staging signs URLs the workload pod uses without credentials of "
        "its own; it is needed for --input files over "
        f"{CONFIGMAP_MAX_BYTES // 1024} KiB in total, which are too large for "
        "the Job's ConfigMap, and --output files over "
        f"{LOGGED_OUTPUT_MAX_BYTES // 1024} KiB in total, which are too large "
        "for the pod's log. The credentials on this host cannot sign: they "
        "look like "
        "a person's (`gcloud auth application-default login`), which have no "
        "private key and no service account to sign through. Use one of:\n"
        f"  - `config set {SERVICE_ACCOUNT_PARAM}=<sa>` on the compute "
        "profile, so staging alone acts as <sa> and kubectl keeps "
        "your identity (you need roles/iam.serviceAccountTokenCreator on "
        "<sa>; <sa> needs roles/storage.objectAdmin on the staging bucket);\n"
        "  - `gcloud auth application-default login "
        "--impersonate-service-account=<sa>` (gcloud's name for acting as "
        "<sa>), with the same roles;\n"
        f"  - '{KEY_FILE_PARAM}' naming a service-account key -- not "
        f"{base.GAC_ENV}, since {base.KUBECTL_IDENTITY_WARNING};\n"
        "  - running from a GCE/GKE host whose service account has "
        "roles/iam.serviceAccountTokenCreator on itself;\n"
        f"  - on a cluster with Workload Identity, '{POD_SERVICE_ACCOUNT_PARAM}: "
        "<ksa>' (gke_job), so the pod moves files with its own token and "
        "nothing is signed (<ksa> needs roles/storage.objectAdmin on the "
        "staging bucket).",
        error_code="SIGNING_UNSUPPORTED",
    )


def preflight_checks(
        bucket: str,
        credentials: Any = None,
        pod_service_account: str | None = None) -> list[base.PreflightCheck]:
    """Whether this host can stage files through `bucket`, as preflight checks.

    The client library, the credentials staging uses -- `credentials` (from
    `staging_credentials`) or else Application Default Credentials (which is
    what the library uses, not `gcloud auth list`'s account) -- a signed URL
    made with them, and the bucket. Nothing is uploaded. Credentials that
    cannot sign fail: the pod then cannot reach the bucket at all, so inputs
    too large for the ConfigMap cannot go and outputs too large for the
    pod's log cannot come back. With `pod_service_account` nothing is
    signed, so signing is not checked.
    """
    try:
        import google.auth  # noqa: PLC0415 - optional dependency
        from google.cloud import storage  # noqa: PLC0415
    except ImportError:
        return [
            base.PreflightCheck(
                "staging:library", "fail",
                "google-cloud-storage is not installed",
                env_check.ask_to_install(["google-cloud-storage"]))
        ]
    project = None
    if credentials is None:
        try:
            credentials, project = google.auth.default()
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            return [
                base.PreflightCheck(
                    "staging:credentials", "warn",
                    f"no Application Default Credentials, so only files small "
                    f"enough to skip the bucket can move: {exc}",
                    f"gcloud auth application-default login, and set "
                    f"'{SERVICE_ACCOUNT_PARAM}' to a service account (plain "
                    "`application-default login` cannot sign URLs)")
            ]
    who = _impersonated(credentials) or getattr(credentials,
                                                "service_account_email", None)
    checks = [
        base.PreflightCheck(
            "staging:credentials", "ok",
            type(credentials).__name__ +
            (f" ({who})" if isinstance(who, str) and who != "default" else ""))
    ]
    client = storage.Client(project=project, credentials=credentials)
    if pod_service_account:
        checks.append(
            base.PreflightCheck(
                "staging:signing", "ok",
                f"not needed: the pod moves files as '{pod_service_account}'"))
    else:
        try:
            # Signed the way a run signs: through IAM for an ambient identity,
            # which is where a missing Token Creator role shows up.
            generate_v4_signed_url(bucket,
                                   "preflight",
                                   expiration_seconds=60,
                                   client=client)
            checks.append(
                base.PreflightCheck("staging:signing", "ok", "can sign"))
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            detail, _, fix = str(exc).partition("Use one of:")
            checks.append(
                base.PreflightCheck("staging:signing", "fail", detail.strip(),
                                    "Use one of:" + fix if fix else ""))
    try:
        # Listing needs only object permissions, which staging needs anyway;
        # describing the bucket would need more.
        name, prefix = locate(bucket, "")
        next(
            iter(client.list_blobs(name, prefix=prefix or None,
                                   max_results=1)), None)
        checks.append(
            base.PreflightCheck("staging:bucket", "ok", f"gs://{bucket}"))
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        code = getattr(exc, "code", None)
        checks.append(
            base.PreflightCheck(
                "staging:bucket", "warn" if code == 403 else "fail",
                f"gs://{bucket}: {str(exc).splitlines()[0]}",
                "Set staging_bucket to a bucket these credentials hold "
                "roles/storage.objectAdmin on"))
    return checks


def generate_v4_signed_url(
    bucket_name: str,
    blob_name: str,
    method: str = "GET",
    expiration_seconds: int = DEFAULT_URL_EXPIRATION_SECONDS,
    client: Any = None,
) -> str:
    """Generate a V4 signed URL for downloading (GET) or uploading (PUT)."""
    clean_bucket = clean_bucket_name(bucket_name)
    if not clean_bucket:
        raise base.ComputeConfigError(
            "Staging bucket must not be empty.",
            error_code="INVALID_FIELD",
        )
    storage_client = client or get_storage_client()
    name, full_blob = locate(clean_bucket, blob_name)
    blob = storage_client.bucket(name).blob(full_blob)
    try:
        credentials = _credentials_of(storage_client)
        signing = _signing_kwargs(credentials)
    except Exception as exc:
        if not _is_auth_error(exc):
            raise
        raise base.ComputeConfigError(
            f"Failed to sign gs://{clean_bucket}/{blob_name}: the credentials "
            f"on this host could not be used: {exc}",
            error_code="SIGNING_FAILED",
        ) from exc

    kwargs: dict[str, Any] = {
        "version": "v4",
        "expiration": datetime.timedelta(seconds=expiration_seconds),
        "method": method.upper(),
        **signing,
    }
    if method.upper() == "PUT":
        kwargs["content_type"] = STAGING_CONTENT_TYPE

    try:
        return blob.generate_signed_url(**kwargs)
    except Exception as exc:
        impersonated = _impersonated(credentials)
        if impersonated:
            raise base.ComputeConfigError(
                f"Failed to sign gs://{clean_bucket}/{blob_name} acting as "
                f"'{impersonated}': {exc}. To act as it, your own account "
                "(this host's Application Default Credentials) needs "
                f"'roles/iam.serviceAccountTokenCreator' on {impersonated}.",
                error_code="SIGNING_FAILED",
            ) from exc
        if "service_account_email" in signing:
            raise base.ComputeConfigError(
                f"Failed to sign gs://{clean_bucket}/{blob_name} through the "
                f"IAM Credentials API as '{signing['service_account_email']}': "
                f"{exc}. That service account needs "
                "'roles/iam.serviceAccountTokenCreator' "
                "(iam.serviceAccounts.signBlob) on itself.",
                error_code="SIGNING_FAILED",
            ) from exc
        raise base.ComputeConfigError(
            f"Failed to sign gs://{clean_bucket}/{blob_name}: {exc}",
            error_code="SIGNING_FAILED",
        ) from exc


def upload_file_to_gcs(
    local_path: str | Path,
    bucket_name: str,
    blob_name: str,
    client: Any = None,
) -> None:
    """Upload a local file to GCS."""
    clean_bucket = clean_bucket_name(bucket_name)
    path = Path(local_path).resolve()
    if not path.is_file():
        raise base.ComputeConfigError(
            f"Local file to stage does not exist: {local_path}",
            error_code="FILE_NOT_FOUND",
        )
    storage_client = client or get_storage_client()
    name, full_blob = locate(clean_bucket, blob_name)
    with _gcs_errors("upload to", f"gs://{clean_bucket}/{blob_name}",
                     storage_client):
        storage_client.bucket(name).blob(full_blob).upload_from_filename(
            str(path))


def download_file_from_gcs(
    bucket_name: str,
    blob_name: str,
    dest_path: str | Path,
    client: Any = None,
) -> bool:
    """Download a GCS blob to a local file path. Returns True if found and downloaded."""
    clean_bucket = clean_bucket_name(bucket_name)
    dest = Path(dest_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    storage_client = client or get_storage_client()
    name, full_blob = locate(clean_bucket, blob_name)
    blob = storage_client.bucket(name).blob(full_blob)
    with _gcs_errors("download", f"gs://{clean_bucket}/{blob_name}",
                     storage_client):
        if not blob.exists():
            return False
        blob.download_to_filename(str(dest))
    return dest.is_file()


def delete_staged_prefix(
    bucket_name: str,
    prefix: str,
    client: Any = None,
) -> None:
    """Delete all objects under a staging prefix."""
    clean_bucket = clean_bucket_name(bucket_name)
    if not clean_bucket or not prefix:
        return
    try:
        storage_client = client or get_storage_client()
        name, full_prefix = locate(clean_bucket, prefix)
        bucket = storage_client.bucket(name)
        blobs = list(bucket.list_blobs(prefix=full_prefix))
        if blobs:
            bucket.delete_blobs(blobs)
    except Exception:
        pass


def _gs_uri(url: str) -> str:
    """The `gs://` name of an `object_url`, or `url` itself if not one."""
    prefix = f"{GCS_OBJECT_ENDPOINT}/"
    if not url.startswith(prefix):
        return url
    return "gs://" + urllib.parse.unquote(url[len(prefix):])


def _pod_token_functions(service_account: str, namespace: str,
                         bucket: str) -> str:
    """Shell functions that fetch the pod's token and explain a refusal.

    `gcs_curl WHAT URI ARGS...` runs `curl ARGS...`; if GCS answers 403, it
    says which object was refused, to whom, and the grant that is missing.
    """
    who = (f"Kubernetes service account '{service_account}'"
           if service_account else "the pod's Kubernetes service account")
    if service_account and namespace:
        who += f" in namespace '{namespace}'"
    name = locate(bucket, "")[0] if bucket else ""
    where = f"gs://{name}" if name else "the staging bucket"
    refused = (
        f"[compute] Grant roles/storage.objectAdmin on {where} to {who}: to "
        "its Workload Identity principal, or to the Google service account "
        "its iam.gke.io/gcp-service-account annotation names (setup step 4 "
        "in docs/install.md).")
    no_token = (
        "so the GCS transfer that needs it will fail. If "
        f"{who} has the iam.gke.io/gcp-service-account annotation, that "
        "Google service account must grant it roles/iam.workloadIdentityUser.")
    # sed, not jq or the job's Python: the image is the user's, unchanged.
    # A failure leaves the header empty, so the curl after it fails, and this
    # says why rather than leaving a bare 401. `|| :` because the failure is
    # told by what the reply lacks, and reported here.
    return f"""gcs_token() {{
  local out t
  out=$(curl -fsS --retry 3 -H 'Metadata-Flavor: Google' {POD_TOKEN_URL} 2>&1) || :
  t=$(printf '%s' "$out" | sed -n 's/.*"access_token" *: *"\\([^"]*\\)".*/\\1/p')
  if [ -z "$t" ]; then
    echo "[compute] The GKE metadata server returned no Google token \\
($(printf '%s\\n' "$out" | tail -n 1))," {shlex.quote(no_token)} >&2
    return 1
  fi
  printf '%s' "$t"
}}
gcs_curl() {{
  local what=$1 uri=$2 code rc
  shift 2
  code=$(curl -w '%{{http_code}}' "$@") && return 0
  rc=$?
  if [ "$code" = 403 ]; then
    echo "[compute] GCS refused to $what $uri with the pod's Workload \\
Identity token (HTTP 403)." >&2
    echo {shlex.quote(refused)} >&2
  fi
  return $rc
}}"""


# True on a pod of a multi-host Job other than its leader.
NOT_LEADER = '[ "${JOB_COMPLETION_INDEX:-0}" != 0 ]'


def build_pod_bootstrap_command(
    script_mount: str,
    script_name: str,
    interpreter: str,
    args: Sequence[str] | None = None,
    input_urls: Sequence[tuple[str, str]] | None = None,
    output_urls: Sequence[tuple[str, str]] | None = None,
    log_url: str | None = None,
    mounted_inputs: Sequence[tuple[str, str]] | None = None,
    logged_outputs: Sequence[str] | None = None,
    job: str = "",
    pod_token: bool = False,
    service_account: str = "",
    namespace: str = "",
    bucket: str = "",
) -> list[str]:
    """Construct a container bootstrap shell command.

    `input_urls`, `output_urls` and `log_url` are where the pod moves staged
    files: signed URLs, or with `pod_token`, plain `object_url`s.
    `log_url` keeps a copy of the workload's output in GCS, which outlives
    the pod; a failed upload of it is reported but does not fail the job.
    Every pod of a multi-host Job runs this; only the leader
    (`JOB_COMPLETION_INDEX` 0, or none) returns outputs and the log, which
    the others would write over.
    `mounted_inputs` pairs each input's path with its key in the ConfigMap
    mounted at `script_mount`; `logged_outputs` are printed to the log in
    frames, by their index in it, for `split_logged_outputs` to recover. A
    staged output whose upload fails is framed the same way, by its index in
    `output_urls`, so an upload the bucket refuses still comes back.
    While the workload runs, an output whose file changed is sent again the
    same way, within a second and at most every LIVE_OUTPUT_INTERVAL_SEC, so a pod lost before the end
    still returns it as it last was; `split_logged_outputs` keeps the last
    whole copy. SIGTERM (a timeout, a preemption) is passed on to the
    workload, and the outputs are sent once it ends, within the grace period.
    `job` names the Job in the frames' marker. With `pod_token`, every
    transfer sends the pod's Workload Identity token, fetched just before it
    since a token lasts about an hour, and a refusal names the object, the
    pod's `service_account` in `namespace`, and the grant on `bucket` it
    needs.
    """
    lines: list[str] = [
        "set -e",
        # One stream: the runtime reads stdout and stderr separately, so a
        # line on stderr (curl's error, a notice) could land inside a frame.
        "exec 2>&1",
        f"mkdir -p {POD_WORK_DIR} && cd {POD_WORK_DIR}",
    ]
    auth = ""
    if pod_token:
        lines.append(_pod_token_functions(service_account, namespace, bucket))
        auth = '-H "Authorization: Bearer $(gcs_token)" '

    def curl(what: str, url: str) -> str:
        # Through `gcs_curl` with a token, so a refusal is explained.
        if not pod_token:
            return "curl "
        return f"gcs_curl {what} {shlex.quote(_gs_uri(url))} "

    # 1. Download staged inputs, or copy them out of the read-only mount
    fetches = [(remote_rel, f"{curl('download', url)}-fsSL --retry 3 {auth}"
                f"-o {shlex.quote(remote_rel)} {shlex.quote(url)}")
               for remote_rel, url in (input_urls or [])]
    fetches += [(remote_rel,
                 f"cp {script_mount}/{key} {shlex.quote(remote_rel)}")
                for remote_rel, key in (mounted_inputs or [])]
    for remote_rel, fetch in fetches:
        parent = pathlib.PurePosixPath(remote_rel).parent.as_posix()
        if parent and parent != ".":
            lines.append(f"mkdir -p {shlex.quote(parent)}")
        lines.append(fetch)

    # 2. Run workload, in the background: this shell is the container's
    # PID 1, which ignores SIGTERM unless it traps it, and bash runs a trap
    # only between commands, not while one runs in the foreground.
    workload_cmd = " ".join([interpreter, f"{script_mount}/{script_name}"] +
                            [shlex.quote(a) for a in (args or [])])
    lines.append("set +e")
    if log_url:
        # Unbuffered, or a pipe holds back output a live `kubectl logs` shows.
        # Through a FIFO, not a pipe, so the workload is the background job
        # the signal goes to, not a subshell around it.
        lines.append(f"mkfifo {LOG_FIFO}\n"
                     f"tee {POD_LOG_FILE} < {LOG_FIFO} &\n"
                     "TEE_PID=$!\n"
                     f"PYTHONUNBUFFERED=1 {workload_cmd} > {LOG_FIFO} 2>&1 &")
    else:
        lines.append(f"{workload_cmd} &")
    lines.append("WORKLOAD_PID=$!")
    lines.append("trap 'kill -TERM $WORKLOAD_PID 2>/dev/null' TERM")
    live = _live_outputs(logged_outputs or [], output_urls or [], job, curl,
                         auth, "-o /dev/null " if pod_token else "")
    if live:
        lines.append(live)
        lines.append(f"{{ {NOT_LEADER} || live_outputs; }} &\nLIVE_PID=$!")
    # The trap cuts a wait short; waiting again gives the workload's own
    # exit status.
    lines.append("while kill -0 $WORKLOAD_PID 2>/dev/null; do\n"
                 "  wait $WORKLOAD_PID\n"
                 "done\n"
                 "wait $WORKLOAD_PID\n"
                 "WORKLOAD_EXIT_CODE=$?")
    if live:
        # Asked to stop, not killed, so a copy it is sending is not cut off
        # in the log ahead of the final ones.
        lines.append(f"touch {LIVE_STOP}\nwait $LIVE_PID")
    if log_url:
        lines.append("wait $TEE_PID")
    lines.append(f"! {NOT_LEADER} || exit $WORKLOAD_EXIT_CODE")
    # 3. Return outputs: framed into the log, or uploaded and framed only if
    # the upload fails. Frames also go into the log copy the pod uploads.
    if logged_outputs or output_urls:
        lines.append(f"OUT_LOG={POD_LOG_FILE if log_url else '/dev/null'}")
        # No newline before the first frame: after output that ends without
        # one, the marker starts mid-line and `split_logged_outputs` splits
        # it there, so the workload's output comes back as it was printed.
        marker = output_frame(job)
        lines.append(f"""LOGGED=0
log_output() {{
  SIZE=$(wc -c < "$2")
  if [ $((LOGGED + SIZE)) -gt {LOGGED_OUTPUT_MAX_BYTES} ]; then
    echo "{marker} too-large $1 $SIZE" | tee -a "$OUT_LOG"
    return 1
  fi
  LOGGED=$((LOGGED + SIZE))
  {{ echo "{marker} begin $1 $(sha256sum < "$2" | cut -d' ' -f1)"
    base64 -w 76 "$2" | sed "s|^|{marker} |"
    echo "{marker} end $1"; }} | tee -a "$OUT_LOG"
}}""")
    for index, remote_rel in enumerate(logged_outputs or []):
        path = shlex.quote(remote_rel)
        lines.append(f"[ ! -f {path} ] || log_output {index} {path}")
    lines.append("UPLOAD_FAILED=0")
    # `gcs_curl` keeps the status code `-w` prints; the reply has no body.
    sink = "-o /dev/null " if pod_token else ""
    for index, (remote_rel, url) in enumerate(output_urls or []):
        path = shlex.quote(remote_rel)
        lines.append(
            f"if [ -f {path} ] && ! {curl('upload', url)}-fsSL --retry 3 "
            f"-X PUT {auth}{sink}"
            f"-H 'Content-Type: {STAGING_CONTENT_TYPE}' --upload-file {path} "
            f"{shlex.quote(url)}; then\n"
            f"  echo \"[compute] Failed to upload output {path}; returning "
            f"it in the log\" >&2\n"
            f"  log_output {index} {path} || UPLOAD_FAILED=1\n"
            f"fi")
    if log_url:
        lines.append(
            f"{curl('upload', log_url)}-fsS --retry 3 -X PUT {auth}{sink}"
            f"-H 'Content-Type: {STAGING_CONTENT_TYPE}' "
            f"--upload-file {POD_LOG_FILE} {shlex.quote(log_url)} || "
            "echo '[compute] Failed to upload the log to GCS' >&2")

    lines.append("if [ $WORKLOAD_EXIT_CODE -ne 0 ]; then")
    lines.append("  exit $WORKLOAD_EXIT_CODE")
    lines.append("fi")
    lines.append("if [ $UPLOAD_FAILED -ne 0 ]; then")
    lines.append(
        "  echo \"[compute] Failed to upload output files to GCS\" >&2")
    lines.append("  exit 1")
    lines.append("fi")
    lines.append("exit 0")
    script = "\n".join(lines) + "\n"
    return ["bash", "-c", script]


def _live_outputs(logged_outputs: Sequence[str],
                  output_urls: Sequence[tuple[str, str]], job: str,
                  curl: Callable[[str, str],
                                 str], auth: str, sink: str) -> str:
    """The bash function `live_outputs`, which sends each output again as
    its file changes while the workload runs; "" if there are none.

    A copy is taken first, so a file the workload is rewriting is sent whole
    or not at all. A staged output is uploaded over its GCS copy; one carried
    in the log is framed there, each line written alone and set apart by
    newlines, so output the workload prints meanwhile, even half a line, lands
    between the frame's lines rather than inside one.
    """
    if not logged_outputs and not output_urls:
        return ""
    marker = output_frame(job)
    sends = []
    for index, remote_rel in enumerate(logged_outputs):
        sends.append((index, remote_rel, f'live_frame {index} "$COPY"'))
    for index, (remote_rel, url) in enumerate(output_urls):
        sends.append(
            (len(logged_outputs) + index, remote_rel,
             f"{curl('upload', url)}-fsS --retry 3 -X PUT {auth}{sink}"
             f"-H 'Content-Type: {STAGING_CONTENT_TYPE}' "
             f"--upload-file \"$COPY\" {shlex.quote(url)} >/dev/null"))
    checks = "\n".join(f"    send_if_changed {key} {shlex.quote(remote_rel)} "
                       f"{shlex.quote(send)}"
                       for key, remote_rel, send in sends)
    return f"""live_frame() {{
  SIZE=$(wc -c < "$2")
  if [ "$SIZE" -gt {LIVE_OUTPUT_MAX_BYTES} ] || \\
     [ $((LIVE + SIZE)) -gt {LIVE_OUTPUT_BUDGET_BYTES} ]; then
    return 0
  fi
  LIVE=$((LIVE + SIZE))
  printf '\\n%s begin %s %s\\n' "{marker}" "$1" \\
    "$(sha256sum < "$2" | cut -d' ' -f1)"
  base64 -w 76 "$2" | while IFS= read -r LINE; do
    printf '\\n%s %s\\n' "{marker}" "$LINE"
  done
  printf '\\n%s end %s\\n' "{marker}" "$1"
}}
send_if_changed() {{
  [ -f "$2" ] || return 0
  [ $((TICK - ${{SENT_AT[$1]:--{LIVE_OUTPUT_INTERVAL_SEC}}})) -ge \\
    {LIVE_OUTPUT_INTERVAL_SEC} ] || return 0
  STAMP=$(stat -c '%y %s' "$2" 2>/dev/null) || return 0
  [ "$STAMP" != "${{SENT[$1]:-}}" ] || return 0
  COPY={LIVE_COPY}.$1
  cp "$2" "$COPY" 2>/dev/null || return 0
  SENT[$1]=$STAMP
  SENT_AT[$1]=$TICK
  eval "$3" || true
}}
live_outputs() {{
  LIVE=0
  SENT=()
  SENT_AT=()
  TICK=0
  while [ ! -e {LIVE_STOP} ]; do
    sleep 1
    TICK=$((TICK + 1))
{checks}
  done
}}"""


def split_logged_outputs(log: str,
                         job: str) -> tuple[str, dict[int, bytes | str]]:
    """Separate `job`'s output frames from its pod's log.

    Returns the log without them, and each framed output by its index among
    the logged outputs: its bytes, or why it did not arrive whole.
    """
    marker = output_frame(job)
    kept: list[str] = []
    found: dict[int, bytes | str] = {}
    frame: tuple[int, str, list[str]] | None = None
    for line in log.splitlines(keepends=True):
        at = line.find(marker)
        if at > 0:
            # Output that ended without a newline, then a marker.
            kept.append(line[:at])
            line = line[at:]
        words = line.split()
        marked = (len(words) in (3, 4) and words[0] == marker
                  and words[2].isdigit())
        # A frame's body line: the marker, then base64. Untagged base64
        # lines are the body of a frame from before tagging.
        body = (len(words) == 2 and words[0] == marker
                and _BASE64_LINE.fullmatch(words[1]))
        if words and words[0] == marker and kept and not kept[-1].strip():
            # The newline a copy sent while the workload ran starts each of
            # its lines with, so none starts on the workload's half line.
            kept.pop()
        if (frame is not None and marked and words[1] == "begin"
                and len(words) == 4):
            frame = None  # One never closed, its sender stopped: dropped.
        if frame is not None:
            index, digest, tagged, untagged = frame
            if not (marked and words[1:3] == ["end", str(index)]):
                if body:
                    tagged.append(words[1])
                elif not line.strip():
                    pass  # The newlines setting a sent copy's lines apart.
                elif _BASE64_LINE.fullmatch(line.strip()):
                    untagged.append(line)
                else:
                    kept.append(line)
                continue
            if tagged:
                # Untagged lines in a tagged frame are the workload's own
                # output that happened to look like base64 (`ok`, `100`).
                kept.extend(untagged)
                chunks = tagged
            else:
                chunks = [line.strip() for line in untagged]
            try:
                data = base64.b64decode("".join(chunks), validate=True)
            except binascii.Error:
                data = b""
            if hashlib.sha256(data).hexdigest() == digest:
                found[index] = data
            else:
                _unless_whole(found, index, "arrived damaged in the pod's log")
            frame = None
        elif marked and words[1] == "begin" and len(words) == 4:
            frame = (int(words[2]), words[3], [], [])
        elif marked and words[1] == "end" and len(words) == 3:
            found.setdefault(
                int(words[2]),
                "lost its start when the pod's log rotated, after the "
                "workload printed more than the log keeps")
        elif marked and words[1] == "too-large" and len(words) == 4:
            _unless_whole(
                found, int(words[2]),
                f"is {words[3]} bytes, more than the pod's log carries "
                f"({LOGGED_OUTPUT_MAX_BYTES} bytes of outputs in all)")
        elif body:
            pass  # A body line whose frame's start rotated away.
        else:
            kept.append(line)
    if frame is not None:
        # The pod stopped mid-frame, as when its node is lost.
        _unless_whole(found, frame[0], "was cut off in the pod's log")
        kept.extend(frame[3] if frame[2] else [])
    return "".join(kept), found


def _unless_whole(found: dict[int, bytes | str], index: int, why: str) -> None:
    """Record why output `index` did not arrive, unless an earlier copy of it
    did: a copy sent while the workload ran is its last state, and beats
    none."""
    if not isinstance(found.get(index), bytes):
        found[index] = why
