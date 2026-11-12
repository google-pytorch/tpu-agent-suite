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
"""Run leases: reaping the jobs of sessions that ended without SessionEnd.

A killed session (Ctrl+C, a closed terminal, a crash) never fires SessionEnd,
so its batch jobs keep holding TPU chips. Each run is leased to the harness
process that started it, and two things act once that process is gone: the
detached watcher `submit` starts, after `LEASE_GRACE_SECONDS`, and failing
that the sweep the next session starts once it has claimed its own run. Both
reap with the workspace and package the run was started under, so each job
is cancelled on the cluster it was submitted to rather than on whichever one
the current session points at.
"""

from __future__ import annotations

import os
import pathlib
import socket
import subprocess
import sys
import time
from typing import Any, Callable

from framework import jsonio, paths, utils

from .config import RUNTIME
from .drivers import jobs

Path = pathlib.Path

OWNER_FILE = "lease-owner.json"
WATCHER_FILE = "lease-watcher.json"
SWEEP_FILE = "lease-sweep.json"

# Long enough for `--resume` after a crash to take the run back, short enough
# that TPU work does not queue for long behind a session nobody is resuming.
LEASE_GRACE_SECONDS = 120
LEASE_POLL_SECONDS = 15
# How long a watcher retries cancels that fail (an expired token, a network
# blip) before leaving them to the next session's sweep.
REAP_RETRY_SECONDS = 30 * 60
# Sweeps that may fail to cancel a run's jobs before it is left to an
# explicit `compute job reap`, so a deleted cluster does not slow every start.
SWEEP_ATTEMPTS = 3

# The harness processes a run can be leased to. Any other process chain
# leaves the run unleased: a wrapper taken for the owner exits as soon as the
# hook does, and would get a live session's jobs reaped.
_OWNERS = frozenset({"claude", "codex"})

# Leases sit under the home directory, which may be shared between machines.
_HOST = socket.gethostname()

# `config.resolve_driver` or a stand-in.
Resolver = Callable[..., Any]
# Reaped job ids, and the ids whose cancel failed with the reason.
Reaped = tuple[list[str], dict[str, str]]


def _holder(pid: int, started: str | None) -> dict[str, Any]:
    """A lease document naming a process on this host."""
    return {"pid": pid, "started": started, "host": _HOST}


def session_owner(pid: int | None = None) -> dict[str, Any] | None:
    """The harness process above `pid` (default: this process's parent).

    None when it cannot be identified: the run then goes unleased rather than
    tied to a guess that may outlive the session, or die before it.
    """
    current = os.getppid() if pid is None else pid
    for _ in range(8):
        try:
            info = jobs.process_info(current) if current > 1 else None
        except jobs.ProcessUnknown:
            return None
        if info is None:
            return None
        parent, started, name = info
        if Path(name.lstrip("-")).name in _OWNERS:
            return _holder(current, started)
        current = parent
    return None


def is_alive(holder: dict[str, Any]) -> bool:
    """Whether a lease document's pid is still the process it names.

    A holder on another host counts as alive, as does one `ps` cannot answer
    for: this host cannot tell they are gone, and reaping on a guess would
    cancel a live session's work.
    """
    if holder.get("host", _HOST) != _HOST:
        return True
    pid, started = holder.get("pid"), holder.get("started")
    if not isinstance(pid, int) or not started:
        return False
    try:
        info = jobs.process_info(pid)
    except jobs.ProcessUnknown:
        return True
    return info is not None and info[1] == started


def _path(run_id: str, name: str) -> Path:
    return paths.get_run_root(run_id) / name


def _doc(run_id: str, name: str) -> dict[str, Any]:
    return jsonio.load_json_or_empty(_path(run_id, name))


def read_owner(run_id: str) -> dict[str, Any]:
    """The run's lease owner record; {} if the run is not leased."""
    return _doc(run_id, OWNER_FILE)


def _watcher_alive(run_id: str) -> bool:
    return is_alive(_doc(run_id, WATCHER_FILE))


def claim(run_id: str, owner: dict[str, Any] | None, workspace: Path,
          package: str) -> None:
    """Lease a run to `owner`; a resumed session takes its run over.

    With no owner the run is released instead, so a resumed session whose
    process cannot be identified does not leave its run to be reaped for the
    dead process that last held it. Either way the run starts over on its
    failed sweeps: the session they gave up on is not this one.
    """
    _path(run_id, SWEEP_FILE).unlink(missing_ok=True)
    if owner is None:
        _path(run_id, OWNER_FILE).unlink(missing_ok=True)
        return
    jsonio.write_json(
        _path(run_id, OWNER_FILE), {
            **owner, "workspace": str(workspace),
            "package": package,
            "claimed_at": jobs.utc_now()
        })


def has_ended(run_id: str) -> bool:
    """Whether a run is leased and its owner is gone.

    An unleased run never has: nothing says its session is over.
    """
    owner = read_owner(run_id)
    return bool(owner.get("workspace")) and not is_alive(owner)


def session_run(pid: int | None = None) -> tuple[str, Path] | None:
    """The run, and its workspace, leased to the session `pid` runs under.

    The session-start hook leases its run to the harness process, and every
    command the agent runs is that process's descendant wherever it `cd`s, so
    the lease finds the session's run when the current directory cannot.
    When one harness process has started several sessions (`/clear`), the
    newest claim is the session still running. None when the harness cannot
    be identified or holds no lease.
    """
    me = session_owner(pid)
    if me is None:
        return None
    found: list[tuple[str, str, Path]] = []
    for owner_file in paths.get_runs_root().glob(f"*/{OWNER_FILE}"):
        owner = jsonio.load_json_or_empty(owner_file)
        if not owner.get("workspace") or any(
                owner.get(key) != me[key]
                for key in ("pid", "started", "host")):
            continue
        run_id = owner_file.parent.name
        try:
            utils.validate_run_id(run_id)
        except utils.TpuAgentSuiteError:
            continue
        workspace = Path(owner["workspace"])
        if not workspace.is_dir():
            continue  # Deleted or moved since: nothing there to resolve.
        found.append((str(owner.get("claimed_at") or ""), run_id, workspace))
    if not found:
        return None
    _, run_id, workspace = max(found)
    return run_id, workspace


def _failed_sweeps(run_id: str) -> int:
    return _doc(run_id, SWEEP_FILE).get("failed", 0)


def is_orphaned(run_id: str) -> bool:
    """Whether a run has ended with no watcher left to reap it.

    A run the sweep has given up on is not orphaned.
    """
    return (has_ended(run_id) and not _watcher_alive(run_id)
            and _failed_sweeps(run_id) < SWEEP_ATTEMPTS)


def _batch_jobs(run_id: str | None = None) -> list[jobs.JobRecord]:
    """The jobs a lease guards: batch jobs that are not over yet."""
    return [
        record for record in jobs.list_jobs(run_id=run_id)
        if not record.serve and record.status not in jobs.TERMINAL_STATUSES
    ]


def _reap(run_id: str, owner: dict[str, Any], resolve: Resolver) -> Reaped:
    return jobs.reap_jobs(
        run_id, lambda job_id: resolve(workspace=Path(owner["workspace"]),
                                       package=owner.get("package"),
                                       run_id=run_id,
                                       job_id=job_id))


def reap_orphaned_jobs(
        resolve: Resolver) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Reap the batch jobs of every orphaned run.

    Returns the reaped ids per run, and the ids whose cancel failed with the
    reason. A run whose sweeps fail SWEEP_ATTEMPTS times in a row is given up
    on, until a session claims it again.
    """
    reaped: dict[str, list[str]] = {}
    failed: dict[str, str] = {}
    for run_id in sorted({r.run_id for r in _batch_jobs() if r.run_id}):
        try:
            if not is_orphaned(run_id):
                continue
        except utils.TpuAgentSuiteError:  # a record with a malformed run id
            continue
        run_reaped, run_failed = _reap(run_id, read_owner(run_id), resolve)
        if run_reaped:
            reaped[run_id] = run_reaped
        if not run_failed:
            # A clean sweep ends the streak; earlier failures were transient.
            _path(run_id, SWEEP_FILE).unlink(missing_ok=True)
            continue
        sweeps = _failed_sweeps(run_id) + 1
        jsonio.write_json(_path(run_id, SWEEP_FILE), {"failed": sweeps})
        note = " (giving up on the run)" if sweeps >= SWEEP_ATTEMPTS else ""
        failed.update({jid: why + note for jid, why in run_failed.items()})
    return reaped, failed


def _spawn(log: Path, *argv: str) -> None:
    """Start `runtime.py <argv>` detached, appending its output to `log`.

    In a session of its own, so the terminal hanging up or the harness killing
    its process group -- the moments it exists for -- do not take it along.
    """
    with open(log, "ab") as out:
        subprocess.Popen(  # pylint: disable=consider-using-with
            [sys.executable, str(RUNTIME), *argv],
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=out,
            start_new_session=True)


def spawn_watcher(run_id: str) -> None:
    """Start a watcher (`job reap --watch`) for a leased run that has none.

    `submit` calls it, so a watcher exists only while a run has jobs to guard.
    """
    if read_owner(run_id).get("workspace") and not _watcher_alive(run_id):
        _spawn(_path(run_id, "lease-watch.log"), "compute", "job", "reap",
               "--watch", "--run-id", run_id)


def spawn_sweep(run_id: str) -> None:
    """Reap orphaned runs in the background, logging under `run_id`.

    Cancels can take minutes, and a session start must not wait on them.
    """
    _spawn(_path(run_id, "lease-sweep.log"), "compute", "job", "reap",
           "--orphaned")


def watch(
    run_id: str,
    resolve: Resolver,
    grace: float = LEASE_GRACE_SECONDS,
    poll: float = LEASE_POLL_SECONDS,
    retry: float = REAP_RETRY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Reaped:
    """Reap a run's batch jobs once its owner has been gone for `grace`.

    Cancels that fail are retried, backing off, for up to `retry`. Returns
    nothing when another watcher has the run, or the run is unleased, deleted
    or has no batch jobs left to guard; the next `submit` starts a watcher.
    """
    run_root = paths.get_run_root(run_id)
    if not run_root.is_dir() or _watcher_alive(run_id):
        return [], {}
    me = _holder(os.getpid(), jobs.process_started(os.getpid()))
    jsonio.write_json(run_root / WATCHER_FILE, me)
    try:
        lost_at: float | None = None
        while run_root.is_dir() and _batch_jobs(run_id):
            owner = read_owner(run_id)
            if not owner.get("workspace"):
                return [], {}
            if is_alive(owner):
                lost_at = None
            elif lost_at is None:
                lost_at = clock()
            elif clock() - lost_at >= grace:
                break
            sleep(poll)
        else:
            return [], {}
        deadline = clock() + retry
        reaped: list[str] = []
        while True:
            done, failed = _reap(run_id, owner, resolve)
            reaped += done
            if not failed or clock() >= deadline:
                return reaped, failed
            sleep(poll)
            poll = min(poll * 2, 300)
    finally:
        if _doc(run_id, WATCHER_FILE).get("pid") == me["pid"]:
            (run_root / WATCHER_FILE).unlink(missing_ok=True)
