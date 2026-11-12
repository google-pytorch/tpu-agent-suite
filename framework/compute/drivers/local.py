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
"""Local compute driver executing directly on the host machine."""

from __future__ import annotations

import os
import pathlib
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from typing import Any

from . import base, jobs, probe, registry, shell

Path = pathlib.Path


@registry.register_driver("local")
class LocalDriver(base.ComputeDriver):
    """Driver executing compute directly on the local host.

    Implements the lifecycle directly rather than inheriting `RemoteShellDriver`
    since no major overhead (copying, etc.) is needed.
    """

    quick_run = True
    SETUP_FIELDS = base.SetupFields(optional=("python_interpreter", ), )

    def __init__(self, python_interpreter: str | None = None) -> None:
        self.python_interpreter = python_interpreter or sys.executable

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> LocalDriver:
        """Build the driver; only `python_interpreter` is read, and optional."""
        return cls(python_interpreter=params.get("python_interpreter"))

    @classmethod
    def register_cli_args(cls, parser: Any) -> None:
        """Add nothing: the local backend has no flags of its own."""

    def inspect(self) -> base.HardwareProfile:
        """Probe local hardware for TPU chips, topology, and availability."""
        return shell.profile_from_dict(probe.probe_host(), {})

    def describe_target(self) -> str:
        """This host, by hostname."""
        return f"this host ({socket.gethostname()})"

    def _preflight_target(self, checks: list[base.PreflightCheck]) -> bool:
        """This host has TPU devices, and no other process holds them."""
        accel, vfio = probe.list_devices()
        if not accel + vfio:
            return base.fail(
                checks, "tpu_devices", "no TPU devices on this host",
                "The local backend runs work here, so this must be a TPU VM; "
                "otherwise choose a remote backend")
        checks.append(
            base.PreflightCheck("tpu_devices", "ok",
                                f"{len(accel + vfio)} device(s)"))
        holders, _, _ = probe.scan_holders()
        if holders:
            checks.append(
                base.PreflightCheck(
                    "tpu_free", "warn", "held: " + "; ".join(holders),
                    "Only one process can hold the chips; stop it or wait"))
        return True

    def _command_for(self, path: Path, args: list[str] | None,
                     python_interpreter: str | None) -> list[str]:
        """Build the argv that executes a script, chosen by file extension."""
        runner = (python_interpreter or self.python_interpreter
                  or sys.executable)
        # No shell sits between this argv and exec, so `~/venv/bin/python`
        # from environment.json would be looked up literally. Remote drivers
        # hand the runner to the target's shell unquoted, which expands it
        # against the *remote* home -- the same spelling works for both.
        runner = os.path.expanduser(runner)
        if path.suffix == ".sh":
            # The Python a shell recipe finds its tools beside.
            return [
                "env", f"TPU_AGENT_SUITE_PYTHON={runner}", "bash",
                str(path), *(args or [])
            ]
        return [runner, str(path), *(args or [])]

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
        """Synchronously execute a script on this host.

        With inputs or outputs the script runs in a scratch directory holding
        copies of the inputs, exactly as a remote driver would, so a prompt's
        `--input x.json -- x.json` spelling behaves the same here as on a VM.
        Without either it runs in the caller's directory, as before.
        """
        del job_name  # A local run creates nothing to name.
        target_path = shell.require_file(code_path, "code_path")
        inputs, outputs = inputs or [], outputs or []
        for staged in inputs:
            shell.require_file(staged.local, f"--input {staged.local}")
        cmd = self._command_for(target_path, args, python_interpreter)

        if not (inputs or outputs):
            return self._exec(cmd, timeout, cwd=None)

        with tempfile.TemporaryDirectory(
                prefix="tpu_agent_suite_run_") as run_dir:
            for staged in inputs:
                dest = pathlib.Path(run_dir, staged.remote)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(staged.local, dest)
            result = self._exec(cmd, timeout, cwd=run_dir)
            for staged in outputs:
                produced = pathlib.Path(run_dir, staged.remote)
                if not produced.is_file():
                    result = shell.missing_output(result, staged, "")
                    continue
                dest = pathlib.Path(staged.local)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(produced, dest)
        return result

    def _exec(self, cmd: list[str], timeout: int | None,
              cwd: str | None) -> base.SynchronousExecutionResult:
        # Own process group, so a timeout can take the whole tree. `subprocess`
        # kills only the direct child on `TimeoutExpired`, which is not enough
        # here: a workload that spawns its own children -- an autotune sweep
        # runs one subprocess per trial -- leaves them alive and holding the
        # accelerator, and the next run then collides with its own orphans.
        try:
            with subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd,
                    env=dict(os.environ),
                    start_new_session=True,
            ) as proc:
                try:
                    stdout, stderr = proc.communicate(timeout=timeout)
                except subprocess.TimeoutExpired as exc:
                    _kill_process_group(proc)
                    stdout, stderr = proc.communicate()
                    return base.SynchronousExecutionResult(
                        exit_code=124,
                        stdout=(exc.stdout if isinstance(exc.stdout, str) else
                                stdout or ""),
                        stderr=(f"Execution timed out after {timeout} "
                                f"seconds. The process group was terminated."),
                    )
            return base.SynchronousExecutionResult(
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        except OSError as exc:
            return base.SynchronousExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"Execution failed: {exc}",
            )

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
        """Dispatch a detached process group on this host."""
        target_path = shell.require_file(recipe_path, "recipe_path")
        record = self._new_record(
            job_name=job_name or target_path.stem,
            recipe_path=str(target_path),
            run_id=run_id or "",
            backend="local",
            serve=serve,
        )
        log_path = record.directory / "output.log"
        exit_code_path = record.directory / "exit_code.txt"
        record.log_path = str(log_path)
        if outputs:
            record.extra["outputs"] = base.StagedFile.as_records(outputs)

        for staged in (inputs or []):
            dest = record.directory / staged.remote
            dest.parent.mkdir(parents=True, exist_ok=True)
            src_p = Path(staged.local).resolve()
            if src_p.is_dir():
                shutil.copytree(src_p, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(src_p, dest)

        cmd = self._command_for(target_path, args, python_interpreter)
        quoted = " ".join(shlex.quote(part) for part in cmd)
        wrapped = (f"({quoted}) > {shlex.quote(str(log_path))} 2>&1; "
                   f"echo $? > {shlex.quote(str(exit_code_path))}")

        proc = subprocess.Popen(
            ["sh", "-c", wrapped],
            start_new_session=True,
            cwd=str(record.directory),
        )
        record.pid = proc.pid
        record.pid_started = jobs.process_started(proc.pid)
        record.status = "RUNNING"
        record.save()
        return record.job_id

    def get_status(self, job_id: str) -> base.AsynchronousExecutionResult:
        """Report job state, settling a finished process's status on first read.

        A process that has exited leaves the record saying RUNNING until someone
        looks, so the exit code is read and recorded here.
        """
        record = jobs.JobRecord.load(job_id)
        if record is None:
            return base.AsynchronousExecutionResult(job_id=job_id,
                                                    status="UNKNOWN",
                                                    error="Job not found")

        if record.status == "RUNNING" and record.pid:
            if not jobs.process_is_alive(record.pid, record.pid_started):
                record.mark(*_terminal_state(record.directory))

        return base.AsynchronousExecutionResult(
            job_id=job_id,
            status=record.status,
            start_time=record.start_time,
            end_time=record.end_time,
            exit_code=record.exit_code,
            details={
                "pid": record.pid,
                "job_name": record.job_name
            },
        )

    def fetch_logs(self, job_id: str) -> str:
        """Read the job's captured stdout/stderr from its record directory."""
        record = jobs.JobRecord.load(job_id)
        if record is None:
            return ""
        log_path = record.directory / "output.log"
        if not log_path.is_file():
            return ""
        return log_path.read_text(encoding="utf-8")

    def fetch_artifacts(self, job_id: str, dest_dir: Path) -> list[Path]:
        """Copy the job's `artifacts/` directory into dest_dir."""
        record = jobs.JobRecord.load(job_id)
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        if record is None:
            return []
        retrieved: list[Path] = []
        source = record.directory / "artifacts"
        if source.is_dir():
            for item in sorted(source.iterdir()):
                target = dest_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target)
                retrieved.append(target)
        # Declared outputs sit beside `artifacts/`, not in it, so a missing
        # or emptied artifacts directory must not take them down with it.
        if record.extra.get("outputs"):
            for out_spec in record.extra["outputs"]:
                src_out = record.directory / out_spec["remote"]
                if src_out.exists():
                    target = base.StagedFile.artifact_path(dest_dir, out_spec)
                    if src_out.is_dir():
                        shutil.copytree(src_out, target, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_out, target)
                    # An output declared under `artifacts/` is copied by both
                    # halves of this method, and the same path twice reads as
                    # two files retrieved.
                    if target not in retrieved:
                        retrieved.append(target)
        return retrieved

    def port_forward(
        self,
        local_port: int,
        remote_port: int,
        job_id: str | None = None,
        background: bool = False,
    ) -> int:
        """Port forward is a no-op on the local host."""
        if local_port != remote_port:
            raise base.ComputeConfigError(
                f"Backend 'local' cannot map localhost:{local_port} to "
                f"localhost:{remote_port}: the workload runs on this host and "
                f"is already reachable on {remote_port}.",
                error_code="UNSUPPORTED_OPERATION",
            )
        print(
            f"Local backend: workload is already running on localhost:{remote_port}.",
            file=sys.stderr,
        )
        return 0

    def cancel(self, job_id: str) -> None:
        """Signal the job's process group, falling back to the pid alone.

        A job that has already exited is left saying how it exited; only a
        live one becomes CANCELLED.
        """
        if self.get_status(job_id).status in jobs.TERMINAL_STATUSES:
            return
        record = jobs.JobRecord.load(job_id)
        if record is None:
            return
        if record.pid:
            try:
                os.killpg(os.getpgid(record.pid), signal.SIGTERM)
            except OSError:
                try:
                    os.kill(record.pid, signal.SIGTERM)
                except OSError:
                    pass
        record.mark("CANCELLED")


def _terminal_state(job_dir: Path) -> tuple[str, int]:
    """Read the recorded exit code and map it to a terminal status."""
    exit_code = jobs.read_exit_code(job_dir / "exit_code.txt")
    if exit_code is None:
        exit_code = 1
    return ("COMPLETED" if exit_code == 0 else "FAILED"), exit_code


def _kill_process_group(proc: "subprocess.Popen") -> None:
    """TERM then KILL the whole group, tolerating a race with natural exit."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            proc.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            continue
