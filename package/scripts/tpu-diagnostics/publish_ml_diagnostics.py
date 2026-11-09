#!/usr/bin/env python3
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
"""Publish XProf captures to an ML Diagnostics dashboard.

    publish_ml_diagnostics.py --cluster projects/<p>/locations/<r>/clusters/<c> \\
        --gcs gs://<bucket>/<prefix> --run-group <run_id> \\
        --trace <label>=<dir> [--trace <label>=<dir> ...]

Each `<dir>` holds a capture's `plugins/profile/<timestamp>/` files. A capture
with no `/device:TPU:` plane is skipped. For the rest this copies the files to
`<gcs>/<run id>/plugins/profile/` (the backend appends `<run id>/plugins/
profile/` to the run's gcsPath), registers an ML Diagnostics run, and lists
its profiler sessions, which makes the backend scan the upload, until one
carries a dashboardUri. Run ids come from the run group and label, so
publishing again lands on the same run. The request shapes match the
google-cloud-mldiagnostics SDK's ControlPlaneClient (v1alpha). Prints
`{"published": {label: link}, "skipped": {...}, "failed": {...}}` and exits 1
if any capture failed. Authenticates with application-default credentials.
Standard library only, plus `gcloud`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API = "https://hypercomputecluster.googleapis.com/v1alpha"
POLL_TRIES = 12
POLL_SECONDS = 10
ALREADY_EXISTS = 6  # google.rpc.Code, in a failed create operation
CLUSTER = re.compile(r"^projects/([^/]+)/locations/([^/]+)/clusters/[^/]+$")


def location(cluster: str) -> tuple[str, str]:
    """(project, region) of a `projects/<p>/locations/<r>/clusters/<c>` path."""
    m = CLUSTER.match(cluster)
    if not m:
        raise ValueError(f"not an ML Diagnostics cluster path: {cluster!r}")
    return m[1], m[2]


def run_id(group: str, label: str = "") -> str:
    """A run id the API accepts: lowercase letters, digits and '-', <= 63."""
    rid = re.sub(r"[^a-z0-9]+", "-", f"{group}-{label}".lower())
    return rid.strip("-")[:63].rstrip("-")


def has_tpu_plane(profile_dir: Path) -> bool:
    """Whether any *.xplane.pb under `profile_dir` has a TPU device plane."""
    return any(b"/device:TPU:" in f.read_bytes()
               for f in profile_dir.rglob("*.xplane.pb"))


def dashboard_uri(sessions: Any) -> str | None:
    """The first non-empty dashboardUri in a profilerSessions listing."""
    for session in (sessions or {}).get("profilerSessions") or []:
        if session.get("dashboardUri"):
            return session["dashboardUri"]
    return None


def _run(argv: list[str], what: str, timeout: int = 3600) -> str:
    proc = subprocess.run(argv,
                          capture_output=True,
                          text=True,
                          stdin=subprocess.DEVNULL,
                          timeout=timeout,
                          check=False)
    if proc.returncode:
        raise RuntimeError(f"{what} failed: {proc.stderr.strip()[-400:]}")
    return proc.stdout


def _call(method: str, url: str, token: str, body: Any = None):
    req = urllib.request.Request(
        url,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        })
    try:
        with urllib.request.urlopen(req, timeout=120) as reply:
            return reply.status, json.loads(reply.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode("utf-8", "replace")[:400]}
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return 0, {"error": str(exc)}


def register(base: str, rid: str, group: str, gcs: str, token: str) -> None:
    """Create run `rid`, or accept it if an earlier publish made it."""
    code, op = _call(
        "POST", f"{base}?machine_learning_run_id={rid}", token, {
            "displayName": rid,
            "runSet": group,
            "runPhase": "COMPLETED",
            "artifacts": {
                "gcsPath": gcs
            },
            "tools": [{
                "xprof": {}
            }],
        })
    if code == 409:
        return
    if code != 200:
        raise RuntimeError(f"register {rid}: HTTP {code} {op}")
    # Create returns a long-running operation; its error is the real answer.
    for _ in range(POLL_TRIES):
        if op.get("done"):
            err = op.get("error")
            if err and err.get("code") != ALREADY_EXISTS:
                raise RuntimeError(f"register {rid}: {err}")
            return
        time.sleep(POLL_SECONDS)
        code, op = _call("GET", f"{API}/{op.get('name')}", token)
        if code != 200:
            raise RuntimeError(f"register {rid}: operation HTTP {code} {op}")
    raise RuntimeError(f"register {rid}: operation did not finish")


def publish_one(cluster: str, gcs: str, group: str, label: str, src: Path,
                token: str) -> str:
    """Upload one capture, register its run and return the dashboard link."""
    project, region = location(cluster)
    rid, prefix = run_id(group, label), gcs.rstrip("/")
    _run([
        "gcloud", "storage", "rsync", "-r",
        str(src / "plugins" / "profile"), f"{prefix}/{rid}/plugins/profile/"
    ], f"upload {src}")
    base = f"{API}/projects/{project}/locations/{region}/machineLearningRuns"
    register(base, rid, run_id(group), prefix, token)
    # Listing is what makes the backend scan the upload into a session.
    for _ in range(POLL_TRIES):
        link = dashboard_uri(
            _call("GET", f"{base}/{rid}/profilerSessions", token)[1])
        if link:
            return link
        time.sleep(POLL_SECONDS)
    raise RuntimeError(f"{rid} registered, but no session appeared")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--gcs", required=True)
    ap.add_argument("--run-group", required=True)
    ap.add_argument("--trace",
                    action="append",
                    required=True,
                    metavar="LABEL=DIR")
    args = ap.parse_args(argv)
    location(args.cluster)
    if not args.gcs.startswith("gs://"):
        ap.error(f"--gcs must be a gs:// path: {args.gcs!r}")
    result: dict[str, dict[str, str]] = {
        "published": {},
        "skipped": {},
        "failed": {}
    }
    todo = []
    for item in args.trace:
        label, _, src = item.partition("=")
        profile = Path(src) / "plugins" / "profile"
        if not profile.is_dir():
            result["failed"][label] = f"no {profile}"
        elif has_tpu_plane(profile):
            todo.append((label, Path(src)))
        else:
            result["skipped"][label] = "no /device:TPU: plane (host-only)"
    token = _run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        "ADC token",
        timeout=60).strip() if todo else ""
    for label, src in todo:
        # One bad capture must not cost the others their links.
        try:
            result["published"][label] = publish_one(args.cluster, args.gcs,
                                                     args.run_group, label,
                                                     src, token)
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            result["failed"][label] = str(exc)
    print(json.dumps(result, indent=2))
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
