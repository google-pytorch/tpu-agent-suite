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
"""On-disk record of asynchronous compute jobs."""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import pathlib
import subprocess
import uuid
from typing import Any, Callable

from framework import jsonio, paths

Path = pathlib.Path


def jobs_dir() -> Path:
    """Return the directory holding job records, creating it if needed."""
    path = paths.get_runs_root() / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# Statuses a job does not come back from. A record in one of these is left
# alone by later status reads and by cancel.
TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})


def new_job_id() -> str:
    """Mint a sortable, collision-resistant job identifier."""
    stamp = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"job-{stamp}-{uuid.uuid4().hex[:6]}"


def read_exit_code(path: Path) -> int | None:
    """The exit code a launch wrapper wrote to `path`.

    None while the file is missing; 1 when it cannot be read or holds no
    number, so a mangled file never reads as success.
    """
    if not path.is_file():
        return None
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return 1


class ProcessUnknown(OSError):
    """`ps` could not say whether a process is running; never read as gone."""


def process_info(pid: int) -> tuple[int, str, str] | None:
    """(parent pid, start-time token, name) of a live process, else None.

    The start-time token is opaque, only ever compared with an earlier token
    for the same pid: equal means the same process, and anything else means
    the kernel has since handed the pid to something new. Reads /proc where
    there is one and falls back to `ps` elsewhere (macOS), raising
    ProcessUnknown when `ps` fails rather than calling the process gone.
    """
    if pid <= 0:
        return None
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        raw = ""
    if raw:
        # The name is parenthesized and may hold spaces; after the last ')'
        # come state, ppid, ..., starttime (field 22 of stat).
        fields = raw[raw.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return int(fields[1]), fields[19], raw[raw.find("(") +
                                               1:raw.rfind(")")]
    if Path("/proc/self").is_dir():
        return None  # /proc is authoritative: no entry, no process.
    try:
        proc = subprocess.run(
            ["ps", "-o", "ppid=,lstart=,comm=", "-p",
             str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProcessUnknown(str(exc)) from exc
    out = proc.stdout.split()
    if proc.returncode and not out:
        return None  # ps found no such pid
    # lstart is always five words: "Thu Sep 24 05:53:38 2026".
    if len(out) < 7:
        raise ProcessUnknown(f"unreadable ps output: {proc.stdout!r}")
    return int(out[0]), " ".join(out[1:6]), " ".join(out[6:])


def process_started(pid: int) -> str | None:
    """The start-time token of a live process, or None (see process_info).

    None too when `ps` cannot tell, leaving later checks to the pid alone.
    """
    try:
        info = process_info(pid)
    except ProcessUnknown:
        return None
    return info[1] if info else None


def process_is_alive(pid: int, started: str | None = None) -> bool:
    """Report whether a submitted process is still running.

    The job is a direct child, so it must be reaped: a zombie still answers
    `kill -0` and would otherwise be reported as running forever. With
    `started`, a pid now running a different process is not alive either --
    that process is not the job, and must not be signalled as though it were.
    """
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return False
    except (ChildProcessError, OSError):
        pass

    try:
        os.kill(pid, 0)
    except OSError:
        return False

    # process_info reads a zombie as not running, as it must be here.
    try:
        info = process_info(pid)
    except ProcessUnknown:
        return True  # `kill -0` found it; unverified, it is judged by pid alone
    return info is not None and started in (None, info[1])


@dataclasses.dataclass
class JobRecord:
    """Metadata a driver persists about one submitted job.

    Drivers write this once at submit time and update it as the job reaches a
    terminal state, so `get_status`, `fetch_logs`, `fetch_artifacts`, and
    `cancel` can recover what they need (remote directory, backend identifiers)
    without re-deriving it.
    """
    job_id: str
    job_name: str = ""
    status: str = "PENDING"
    run_id: str = ""
    backend: str = ""
    pid: int | None = None
    # Start-time token of `pid` (see `process_info`), so a reused pid is not
    # mistaken for the job. None on records written before it was kept.
    pid_started: str | None = None
    start_time: str = dataclasses.field(default_factory=utc_now)
    end_time: str | None = None
    exit_code: int | None = None
    recipe_path: str = ""
    log_path: str = ""
    remote_dir: str = ""
    serve: bool = False
    # The compute profile the job was submitted under, so later commands for
    # it reach its target even after the run switches profile. None on
    # records written before it was kept.
    compute_profile: str | None = None
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def directory(self) -> Path:
        """This job's directory under `jobs_dir()`."""
        return jobs_dir() / self.job_id

    @classmethod
    def create(cls,
               job_name: str = "",
               job_id: str | None = None,
               run_id: str = "",
               backend: str = "",
               serve: bool = False,
               **fields: Any) -> JobRecord:
        """Create a record with a fresh id and its on-disk directory."""
        record = cls(
            job_id=job_id or new_job_id(),
            job_name=job_name,
            run_id=run_id or "",
            backend=backend,
            serve=serve,
            **fields,
        )
        (record.directory / "artifacts").mkdir(parents=True, exist_ok=True)
        record.save()
        return record

    @classmethod
    def load(cls, job_id: str) -> JobRecord | None:
        """Load a record by id, or None when it was never written."""
        data = jsonio.load_json_or_empty(jobs_dir() / job_id / "job.json")
        if not data:
            return None
        known = {f.name for f in dataclasses.fields(cls)}
        extra = data.pop("extra", {}) or {}
        extra.update({k: v for k, v in data.items() if k not in known})
        return cls(**{
            k: v
            for k, v in data.items() if k in known
        },
                   extra=extra)

    def save(self) -> None:
        """Persist the record to its job directory."""
        # Atomic, and in the same indent=2 layout job.json has always had.
        jsonio.write_text_atomic(
            self.directory / "job.json",
            json.dumps(dataclasses.asdict(self), indent=2))

    def mark(self,
             status: str,
             exit_code: int | None = None,
             error: str | None = None,
             **extra: Any) -> JobRecord:
        """Record a terminal (or updated) status and persist it.

        `error` and `extra` go in `extra`: once the Job is deleted, the record
        is the only place left to say why it ended.
        """
        self.status = status
        if exit_code is not None:
            self.exit_code = exit_code
        if error:
            self.extra["error"] = error
        self.extra.update(extra)
        if status in TERMINAL_STATUSES:
            self.end_time = self.end_time or utc_now()
        self.save()
        return self


def list_jobs(
    run_id: str | None = None,
    status: str | None = None,
) -> list[JobRecord]:
    """Persisted job records, oldest first, optionally filtered."""
    base = jobs_dir()
    if not base.is_dir():
        return []

    records: list[JobRecord] = []
    for item in base.iterdir():
        record = JobRecord.load(item.name) if item.is_dir() else None
        if record is None:
            continue
        if run_id and record.run_id != run_id:
            continue
        if status and record.status.upper() != status.upper():
            continue
        records.append(record)

    records.sort(key=lambda r: (r.start_time, r.job_id))
    return records


def serving_jobs(run_id: str) -> list[str]:
    """Ids of a run's live serving jobs -- the ones a session reap leaves."""
    return [
        record.job_id for record in list_jobs(run_id=run_id)
        if record.serve and record.status not in TERMINAL_STATUSES
    ]


def reap_jobs(
    run_id: str,
    driver_resolver: Callable[[str], Any],
    include_serving: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """Cancel every job of `run_id` that is not already over.

    Each job is cancelled by its own backend's driver, resolved from its job
    id (the record names its backend and profile), so the driver decides
    what cancelling means there
    (process group, Kubernetes Job) and what the record ends
    up saying. Returns the ids that ended CANCELLED and, separately, the ids
    whose cancel raised with the reason: a job the reaper could not stop is
    the one thing the caller must not be left to assume is gone.

    Serving jobs outlive the session that started them, so the run-end hook
    leaves them alone. `include_serving` is for a caller that means it -- an
    explicit `job reap` is a decision to clear the run, and a serving job it
    refused to stop would hold TPU chips with nothing left to point at it.
    """
    reaped: list[str] = []
    failed: dict[str, str] = {}
    for record in list_jobs(run_id=run_id):
        if record.status in TERMINAL_STATUSES:
            continue
        if record.serve and not include_serving:
            continue
        try:
            driver_resolver(record.job_id).cancel(record.job_id)
        except Exception as exc:  # noqa: BLE001 - reported per job
            failed[record.job_id] = str(exc)
            continue
        after = JobRecord.load(record.job_id)
        if after is not None and after.status == "CANCELLED":
            reaped.append(record.job_id)
    return reaped, failed
