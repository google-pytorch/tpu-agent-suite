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
"""Job lifecycle shared by drivers that reach a host through a shell.

`GceTpuVmDriver` and `GkePodDriver` differ only in how they run a command on
the target and how they move files to and from it. Everything else -- the
launch wrapper, the pid/exit-code status protocol, log retrieval, artifact
copy-back, and cancellation -- is identical, so it lives here once. A subclass
supplies four primitives (`_sh`, `_copy_in`, `_copy_out`, `_copy_out_file`), a
`JOB_ROOT`, and its own `inspect()`.
"""

from __future__ import annotations

import base64
import contextlib
import importlib.resources
import json
import os
import pathlib
import re
import shlex
import signal
import subprocess
import threading
from typing import Any

from . import base, jobs

Path = pathlib.Path

_PROBE_SOURCE = (
    importlib.resources.files(__package__).joinpath("probe.py").read_text(
        encoding="utf-8"))

# The probe is shipped base64-encoded so no quoting layer between here and the
# remote shell can mangle it.
_PROBE_B64 = base64.b64encode(_PROBE_SOURCE.encode("utf-8")).decode("ascii")


def probe_command(python_interpreter: str = "python3") -> str:
    """Build the shell command that runs the host probe on the target."""
    return (f"{python_interpreter} -c \"import base64;"
            f"exec(base64.b64decode('{_PROBE_B64}').decode('utf-8'))\"")


@contextlib.contextmanager
def sigterm_raises():
    """SIGTERM raises SystemExit(143) inside the block, so `finally` runs.

    Signal handlers are main-thread only; other threads run unguarded.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def raise_exit(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, raise_exit)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


# One advisory lock per target, held for the duration of a `run`.
#
# `run` is synchronous and every caller's contract says to run sequentially,
# but nothing enforced that ACROSS callers: two sessions pointed at one TPU VM
# both SSH in and both start a workload, and the accelerator is not
# time-shared. Colliding runs do not merely fail -- an autotune sweep whose
# trials collide reports a `best_config` skewed toward whichever trials
# happened to run alone, which is a wrong answer rather than an error.
#
# `flock` is the right primitive because the kernel releases the lock when the
# holder dies, so a crashed run cannot strand it. Where `flock` is missing the
# run proceeds unserialised rather than failing: losing the guarantee is worse
# than losing the run, but not by enough to refuse.
RUN_LOCK_PATH = "/tmp/tpu_agent_suite_compute_run.lock"

# `flock`'s exit code when `-w` expires. 75 is EX_TEMPFAIL: the run did not
# fail, it never started, and retrying later is the correct response.
LOCK_BUSY_EXIT = 75

# Ceiling on how long to wait for the lock. Half the call's own timeout, so a
# contended target still leaves the workload the majority of its budget, and
# never more than this however generous the timeout.
MAX_LOCK_WAIT_SECONDS = 300
DEFAULT_LOCK_WAIT_SECONDS = 120


def _lock_wait_seconds(timeout: int | None) -> int:
    """How long to block for the run lock before giving up."""
    if timeout is None:
        return DEFAULT_LOCK_WAIT_SECONDS
    return max(15, min(MAX_LOCK_WAIT_SECONDS, timeout // 2))


def serialized_run_command(inner_path: str, lock_wait: int,
                           use_lock: bool) -> str:
    """Wrap the staged inner script in the target-wide run lock."""
    runner = f"sh {shlex.quote(inner_path)}"
    if not use_lock:
        return runner
    return (f"if command -v flock >/dev/null 2>&1; then "
            f"flock -w {lock_wait} -E {LOCK_BUSY_EXIT} "
            f"{shlex.quote(RUN_LOCK_PATH)} {runner}; "
            f"else {runner}; fi")


_LEADING_ASSIGNMENTS = re.compile(r"((?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*)(.*)",
                                  re.DOTALL)


def run_inner_script(command: str, pidfile: str, cd_dir: str | None) -> str:
    """The script the target actually executes.

    It records its own pid before `exec`ing the workload, so the pid in the
    file IS the workload rather than a shell that wrapped it. That is what
    makes a timeout reapable: without it the only handle on a runaway process
    is the SSH client, and killing that leaves the workload holding chips.
    """
    assignments, workload = _LEADING_ASSIGNMENTS.fullmatch(command).groups()
    lines = [f"echo $$ > {shlex.quote(pidfile)}"]
    if cd_dir:
        lines.append(f"cd {shlex.quote(cd_dir)} || exit 1")
    lines.append(f"{assignments}exec {workload}")
    return "\n".join(lines) + "\n"


class RemoteShellDriver(base.ComputeDriver):
    """Base for drivers whose target is reachable via shell.

    A subclass needs to implement the following:
    - how to run a  shell script on the target (`_sh`)
    - how to put a file there (`_copy_in`)
    - how to get a file from there (`_copy_out`)
    - how to get one file from there (`_copy_out_file`)
    - how to inspect the target (`inspect`)

    `RemoteShellDriver` provides the following:

    - `run` copies the script over and waits for it.
    - `submit` creates `JOB_ROOT/<job_id>/` on the target, copies the recipe in,
      and launches it detached (`nohup setsid`) writing `output.log`, `pid.txt`,
      and, on exit, `exit_code.txt`. Nothing needs to stay connected afterwards.
    - `get_status` reads those two files back: a live pid means RUNNING, an
      exit code means COMPLETED/FAILED.
    - `fetch_logs`, `fetch_artifacts`, and `cancel` address the same directory.

    Local `JobRecord`s mirror this so a caller can look a job up by id in a
    later process without remembering which host it landed on.
    """

    # Directory on the target host under which job directories are created.
    JOB_ROOT = "/tmp/tpu_agent_suite_jobs"
    # The target already exists, so `run` takes seconds.
    quick_run = True

    # What to call the target in user-facing errors ("Copy to TPU VM failed").
    target_description = "remote host"

    def __init__(self, python_interpreter: str | None = None) -> None:
        self.python_interpreter = python_interpreter or "python3"

    def _sh(self,
            script: str,
            timeout: int | None = None) -> subprocess.CompletedProcess:
        """Run a shell script on the target and return the completed process.

        Must not raise on a non-zero exit -- the lifecycle reads return codes.
        """
        raise NotImplementedError

    def _copy_in(self, src: Path, dest: str) -> subprocess.CompletedProcess:
        """Copy a local file to an absolute path on the target."""
        raise NotImplementedError

    def _copy_out(self, src: str, dest: Path) -> subprocess.CompletedProcess:
        """Copy a directory's contents from the target into a local directory."""
        raise NotImplementedError

    def _copy_out_file(self, src: str,
                       dest: Path) -> subprocess.CompletedProcess:
        """Copy one file from the target to a local path."""
        raise NotImplementedError

    def _remote_profile(self, timeout: int = 60) -> base.HardwareProfile:
        """Run the shared probe on the target and parse its profile."""
        interpreter = self.python_interpreter or "python3"
        try:
            proc = self._sh(probe_command(interpreter), timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - reported in the profile
            return _unreachable_profile(f"error: {exc}", {"error": str(exc)})

        if proc.returncode != 0:
            detail = (proc.stderr or "").strip()
            # Name the interpreter: a missing `tpu_info` under a system
            # `python3` reads as a connectivity failure otherwise, and the
            # reader's first move is to check ssh rather than the venv.
            return _unreachable_profile("unreachable", {
                "error": detail,
                "python_interpreter": interpreter,
            })

        try:
            data = json.loads((proc.stdout or "").strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            return _unreachable_profile(
                "unreachable", {
                    "error": f"could not parse probe output: {exc}",
                })
        return profile_from_dict(data, self._profile_details())

    def _remote_dir(self, job_id: str) -> str:
        """Directory on the target holding a job's recipe, log, and artifacts."""
        record = jobs.JobRecord.load(job_id)
        if record and record.remote_dir:
            return record.remote_dir
        return f"{self.JOB_ROOT}/{job_id}"

    def _profile_details(self) -> dict[str, Any]:
        """Transport-specific fields to merge into the profile details.

        The probe reports what it found on the host but has no idea which host
        that was, so each subclass names its target here (VM and zone, pod and
        namespace). Without it a profile is a bare chip count with no way to
        tell two backends' results apart in a log or a cached JSON payload.
        """
        return {}

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
        """Copy code (and any inputs) to the target, execute, copy outputs back.

        Everything lands in one run directory on the target, which is also the
        script's working directory, so a script reads `payload.json` and
        writes `results.json` by bare name on every backend.
        """
        del job_name  # The target already exists; nothing is created to name.
        src_path = require_file(code_path, "code_path")
        inputs, outputs = inputs or [], outputs or []
        for staged in inputs:
            require_file(staged.local, f"--input {staged.local}")
        run_dir = f"/tmp/tpu_agent_suite_run_{jobs.new_job_id()}"
        script_name = f"script{src_path.suffix}"

        # A run without staging keeps the old shape (one file, no cd), so
        # scripts that relied on the login directory keep working.
        staged_run = bool(inputs or outputs)
        if staged_run:
            mk = self._sh(f"mkdir -p {shlex.quote(run_dir)}", timeout=60)
            if mk.returncode != 0:
                return _copy_failure(self.target_description, mk)
        remote_script = (f"{run_dir}/{script_name}"
                         if staged_run else f"{run_dir}{src_path.suffix}")

        copy_res = self._copy_in(src_path, remote_script)
        if copy_res.returncode != 0:
            return _copy_failure(self.target_description, copy_res)
        for staged in inputs:
            remote = f"{run_dir}/{staged.remote}"
            parent = pathlib.PurePosixPath(remote).parent.as_posix()
            if parent != run_dir:
                self._sh(f"mkdir -p {shlex.quote(parent)}", timeout=60)
            copy_res = self._copy_in(Path(staged.local).resolve(), remote)
            if copy_res.returncode != 0:
                return _copy_failure(self.target_description,
                                     copy_res,
                                     what=staged.local)

        runner = self._runner_for(src_path, python_interpreter)
        command = " ".join([runner, shlex.quote(remote_script)] +
                           [shlex.quote(a) for a in (args or [])])

        # Stage the workload as its own script rather than passing it inline.
        # Two things need it: the pid has to be recorded by the same shell that
        # becomes the workload, and the whole thing has to be wrappable in the
        # run lock. Base64 so no quoting layer between here and the remote
        # shell can mangle it -- the same reason the probe travels that way.
        pidfile = f"{run_dir}.pid"
        inner_path = f"{run_dir}.launch.sh"
        inner = run_inner_script(command, pidfile,
                                 run_dir if staged_run else None)
        inner_b64 = base64.b64encode(inner.encode("utf-8")).decode("ascii")
        lock_wait = _lock_wait_seconds(timeout)
        use_lock = not os.environ.get("TPU_AGENT_SUITE_COMPUTE_NO_RUN_LOCK")
        launch = (f"printf %s '{inner_b64}' | base64 -d > "
                  f"{shlex.quote(inner_path)} && " +
                  serialized_run_command(inner_path, lock_wait, use_lock))

        try:
            proc = self._sh(launch, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            # The SSH client is gone but the workload is not: it is a process
            # on the target, and nothing about a dead client stops it. Reap it
            # explicitly or it keeps the accelerator until someone notices.
            reaped = self._terminate_remote_run(pidfile)
            return base.SynchronousExecutionResult(
                exit_code=124,
                stdout=exc.stdout if isinstance(exc.stdout, str) else "",
                stderr=(f"Execution timed out after {timeout} seconds. "
                        f"{reaped}"),
            )
        if use_lock and proc.returncode == LOCK_BUSY_EXIT:
            return base.SynchronousExecutionResult(
                exit_code=LOCK_BUSY_EXIT,
                stdout=proc.stdout or "",
                stderr=(f"{self.target_description} is busy: another run held "
                        f"the accelerator for longer than {lock_wait}s, so "
                        f"this one never started. Nothing was executed and no "
                        f"measurement was taken -- retry, or give the run a "
                        f"target of its own."),
            )
        result = base.SynchronousExecutionResult(
            exit_code=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )

        for staged in outputs:
            dest = Path(staged.local)
            dest.parent.mkdir(parents=True, exist_ok=True)
            out = self._copy_out_file(f"{run_dir}/{staged.remote}", dest)
            if out.returncode != 0 or not dest.is_file():
                result = missing_output(result, staged, out.stderr or "")
        return result

    def _terminate_remote_run(self, pidfile: str) -> str:
        """Kill the process group of a `run` whose client already gave up.

        The same TERM-then-KILL on the process group that `cancel` uses for a
        submitted job. The group rather than the pid, because a workload that
        spawned its own children -- an autotune sweep is exactly that -- leaves
        them holding the accelerator when only the parent dies.

        Returns:
            a sentence for the caller's stderr, since this runs on a path that
            has already failed and must not raise a second time.
        """
        quoted = shlex.quote(pidfile)
        script = (f"if [ ! -f {quoted} ]; then echo NO_PIDFILE; exit 0; fi; "
                  f"PID=$(cat {quoted}); "
                  f"PGID=$(ps -o pgid= -p $PID 2>/dev/null | tr -d ' '); "
                  f"if [ -n \"$PGID\" ]; then TARGET=-$PGID; "
                  f"else TARGET=$PID; fi; "
                  f"kill -TERM $TARGET 2>/dev/null; sleep 2; "
                  f"kill -KILL $TARGET 2>/dev/null; "
                  f"rm -f {quoted}; echo REAPED")
        try:
            proc = self._sh(script, timeout=60)
        except Exception:  # noqa: BLE001 - never mask the original timeout
            return ("The remote process could not be reached to reap it; it "
                    "may still be holding the accelerator.")
        out = proc.stdout or ""
        if "REAPED" in out:
            return "The remote process group was terminated."
        if "NO_PIDFILE" in out:
            return "No remote process had started; nothing to reap."
        return ("Reaping the remote process reported nothing; it may still be "
                "holding the accelerator.")

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
    ) -> str:
        """Launch a detached job on the target and return its job id."""
        src_path = require_file(recipe_path, "recipe_path")
        record = self._new_record(
            job_name=job_name or src_path.stem,
            recipe_path=str(src_path),
            run_id=run_id or "",
            backend=self.backend_name,
            serve=serve,
        )
        remote_dir = f"{self.JOB_ROOT}/{record.job_id}"
        record.remote_dir = remote_dir
        record.log_path = f"{remote_dir}/output.log"
        if outputs:
            record.extra["outputs"] = base.StagedFile.as_records(outputs)

        self._sh(f"mkdir -p {shlex.quote(remote_dir)}/artifacts", timeout=60)

        recipe_name = f"recipe{src_path.suffix or '.py'}"
        copy_res = self._copy_in(src_path, f"{remote_dir}/{recipe_name}")
        if copy_res.returncode != 0:
            record.mark("FAILED", exit_code=1)
            raise base.ComputeConfigError(
                f"Could not copy recipe to {self.target_description}: "
                f"{copy_res.stderr}",
                error_code="COPY_FAILED",
            )

        for staged in (inputs or []):
            remote = f"{remote_dir}/{staged.remote}"
            parent = pathlib.PurePosixPath(remote).parent.as_posix()
            if parent != remote_dir:
                self._sh(f"mkdir -p {shlex.quote(parent)}", timeout=60)
            in_copy = self._copy_in(Path(staged.local).resolve(), remote)
            if in_copy.returncode != 0:
                record.mark("FAILED", exit_code=1)
                raise base.ComputeConfigError(
                    f"Could not copy input {staged.local} to {self.target_description}: "
                    f"{in_copy.stderr}",
                    error_code="COPY_FAILED",
                )

        runner = self._runner_for(src_path, python_interpreter)
        quoted_args = " ".join(shlex.quote(a) for a in (args or []))
        quoted_dir = shlex.quote(remote_dir)
        quoted_recipe = shlex.quote(recipe_name)
        inner = (f"{runner} {quoted_recipe} {quoted_args} > output.log 2>&1; "
                 f"echo $? > exit_code.txt")
        launch = (f"cd {quoted_dir} && "
                  f"( nohup setsid sh -c {shlex.quote(inner)} "
                  f"< /dev/null > /dev/null 2>&1 & echo $! > pid.txt; "
                  f"{_started('$!')} > pid_started.txt )")
        self._sh(launch, timeout=60)

        record.status = "RUNNING"
        record.save()
        return record.job_id

    def get_status(self, job_id: str) -> base.AsynchronousExecutionResult:
        """Report job state from the target's pid and exit-code files."""
        record = jobs.JobRecord.load(job_id)
        remote_dir = self._remote_dir(job_id)
        quoted = shlex.quote(remote_dir)
        script = (f"if [ -f {quoted}/pid.txt ]; then "
                  f"{_live_pid(quoted)}"
                  f"if [ -n \"$LIVE\" ]; then echo RUNNING; "
                  f"elif [ -f {quoted}/exit_code.txt ]; then "
                  f"CODE=$(cat {quoted}/exit_code.txt); "
                  f"if [ \"$CODE\" = \"0\" ]; then echo \"COMPLETED 0\"; "
                  f"else echo \"FAILED $CODE\"; fi; "
                  f"else echo \"FAILED 1\"; fi; "
                  f"else echo UNKNOWN; fi")
        try:
            proc = self._sh(script, timeout=30)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            return base.AsynchronousExecutionResult(job_id=job_id,
                                                    status="UNKNOWN",
                                                    error=str(exc))

        status, exit_code = _parse_status_line((proc.stdout or "").strip())
        if record and record.status == "CANCELLED":
            # The remote process was killed, so the probe now reports FAILED.
            # Cancellation is the truer account of why it stopped, so keep it.
            return base.AsynchronousExecutionResult(
                job_id=job_id,
                status="CANCELLED",
                exit_code=record.exit_code,
                start_time=record.start_time,
                end_time=record.end_time,
            )
        if record and status != "UNKNOWN":
            record.mark(status, exit_code)
        return base.AsynchronousExecutionResult(
            job_id=job_id,
            status=status,
            exit_code=exit_code,
            start_time=record.start_time if record else None,
            end_time=record.end_time if record else None,
        )

    def fetch_logs(self, job_id: str) -> str:
        """Read the job's log file off the target."""
        remote_dir = self._remote_dir(job_id)
        proc = self._sh(
            f"cat {shlex.quote(remote_dir)}/output.log 2>/dev/null",
            timeout=60)
        return proc.stdout or ""

    def fetch_artifacts(self, job_id: str, dest_dir: Path) -> list[Path]:
        """Copy the job's artifacts directory back, returning what arrived.

        Only paths the copy actually added are reported, so a caller reusing a
        destination directory is not handed files from an earlier job.
        """
        remote_dir = self._remote_dir(job_id)
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        before = set(dest_dir.iterdir())
        self._copy_out(f"{remote_dir}/artifacts", dest_dir)
        record = jobs.JobRecord.load(job_id)
        if record and record.extra.get("outputs"):
            for out_spec in record.extra["outputs"]:
                target_file = base.StagedFile.artifact_path(dest_dir, out_spec)
                self._copy_out_file(f"{remote_dir}/{out_spec['remote']}",
                                    target_file)
        return sorted(p for p in dest_dir.iterdir() if p not in before)

    def cancel(self, job_id: str) -> None:
        """Terminate the job's whole process group on the target.

        The recipe is launched under `setsid`, so signalling the process group
        reaches the children it spawned; the recorded pid is only the fallback.

        A job that has already exited keeps saying how it exited. A live one
        is recorded CANCELLED only once the signal has been delivered: if the
        target cannot be reached the record stays live and this raises, so
        nothing reads a still-running job as gone.
        """
        if self.get_status(job_id).status in jobs.TERMINAL_STATUSES:
            return
        record = jobs.JobRecord.load(job_id)
        quoted = shlex.quote(self._remote_dir(job_id))
        try:
            proc = self._sh(
                f"if [ -f {quoted}/pid.txt ]; then "
                f"{_live_pid(quoted)}"
                f"if [ -n \"$LIVE\" ]; then "
                f"PGID=$(ps -o pgid= -p $PID 2>/dev/null | tr -d ' '); "
                f"if [ -n \"$PGID\" ]; then kill -TERM -$PGID 2>/dev/null; "
                f"else kill -TERM $PID 2>/dev/null; fi; fi; fi",
                timeout=30)
        except Exception as exc:  # noqa: BLE001 - the job may still be live
            raise base.ComputeConfigError(
                f"Could not reach {self.target_description} to cancel "
                f"{job_id}: {exc}",
                error_code="CANCEL_FAILED") from exc
        if proc.returncode != 0:
            raise base.ComputeConfigError(
                f"Cancelling {job_id} on {self.target_description} failed "
                f"(exit {proc.returncode}): {(proc.stderr or '').strip()}",
                error_code="CANCEL_FAILED")
        if record:
            record.mark("CANCELLED")


def _started(pid: str) -> str:
    """Shell printing a pid's start time: field 22 of /proc/<pid>/stat.

    The targets are Linux, and the job's process name (`nohup`, `setsid`,
    `sh`) holds no spaces, so the fields split cleanly.
    """
    return f"cut -d ' ' -f 22 /proc/{pid}/stat 2>/dev/null"


def _live_pid(quoted_dir: str) -> str:
    """Shell setting PID from pid.txt, and LIVE only if it is still the job."""
    return (
        f"PID=$(cat {quoted_dir}/pid.txt); LIVE=; "
        f"if kill -0 $PID 2>/dev/null && {{ "
        f"[ ! -f {quoted_dir}/pid_started.txt ] || "
        f"[ \"$({_started('$PID')})\" = "
        f"\"$(cat {quoted_dir}/pid_started.txt)\" ]; }}; then LIVE=1; fi; ")


def _copy_failure(target: str,
                  proc: subprocess.CompletedProcess,
                  what: str | None = None) -> base.SynchronousExecutionResult:
    """The result of a `run` that never started because staging failed."""
    subject = f"Copy of {what} to {target}" if what else f"Copy to {target}"
    return base.SynchronousExecutionResult(
        exit_code=proc.returncode or 1,
        stdout=proc.stdout or "",
        stderr=f"{subject} failed: {proc.stderr}",
    )


def missing_output(result: base.SynchronousExecutionResult,
                   staged: base.StagedFile,
                   detail: str) -> base.SynchronousExecutionResult:
    """Record that an `--output` file the caller asked for was never produced.

    Adds a line to stderr naming the missing file. If the script itself
    exited 0, the exit code is changed to 1: the caller was going to read
    that file next, so "succeeded but produced nothing" is a failure. If the
    script already exited non-zero, its exit code is left alone -- the
    missing file is a consequence of that failure, not a separate one.
    """
    note = (f"declared output {staged.remote!r} was not produced on the "
            f"target" + (f": {detail.strip()}" if detail.strip() else ""))
    stderr = result.stderr + ("\n" if result.stderr and
                              not result.stderr.endswith("\n") else "") + note
    return base.SynchronousExecutionResult(
        exit_code=result.exit_code if result.exit_code != 0 else 1,
        stdout=result.stdout,
        stderr=stderr + "\n",
        details=result.details,
    )


def require_file(path: str, field: str) -> Path:
    """Resolve a caller-supplied path, raising the shared not-found error.

    `run` and `submit` answer a missing input the same way so callers do not
    have to special-case which one they invoked.
    """
    try:
        resolved = Path(path).resolve()
        is_regular_file = resolved.is_file()
    except OSError as exc:
        raise base.ComputeConfigError(
            f"{field} '{path}' is not accessible ({exc.strerror or exc}).",
            error_code="INPUT_NOT_FOUND",
        ) from exc
    if not is_regular_file:
        raise base.ComputeConfigError(
            f"{field} '{path}' does not exist.",
            error_code="INPUT_NOT_FOUND",
        )
    return resolved


def _parse_status_line(raw: str) -> tuple[str, int | None]:
    """Parse the status probe's `STATUS [exit_code]` line."""
    if raw == "RUNNING":
        return "RUNNING", None
    for prefix, fallback in (("COMPLETED", 0), ("FAILED", 1)):
        if raw.startswith(prefix):
            parts = raw.split()
            code = fallback
            if len(parts) > 1:
                try:
                    code = int(parts[1])
                except ValueError:
                    code = fallback
            return prefix, code
    return "UNKNOWN", None


def profile_from_dict(data: dict[str, Any],
                      extra_details: dict[str, Any]) -> base.HardwareProfile:
    """Build a HardwareProfile from a probe payload."""
    details = dict(data.get("details", {}))
    details.update(extra_details)
    return base.HardwareProfile(
        accelerator=str(data.get("accelerator", "none")),
        topology=str(data.get("topology", "none")),
        chips=int(data.get("chips", 0)),
        hbm_gb=float(data.get("hbm_gb", 0.0)),
        status=str(data.get("status", "unknown")),
        available_chips=_optional_int(data.get("available_chips")),
        held_by=list(data.get("held_by", [])),
        details=details,
    )


def _optional_int(value: Any) -> int | None:
    """Coerce a probe value to int, preserving `None` as 'not measured'."""
    return None if value is None else int(value)


def _unreachable_profile(status: str,
                         details: dict[str, Any]) -> base.HardwareProfile:
    """Profile returned when the target could not be probed.

    `available_chips` is left unset rather than zeroed: the host was never
    measured, and reporting a measured-looking 0 would be a different claim.
    """
    return base.HardwareProfile(
        accelerator="unknown",
        topology="none",
        chips=0,
        hbm_gb=0.0,
        status=status,
        available_chips=None,
        held_by=[],
        details=details,
    )
