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
"""Runtime Lifecycle Adapter for Harbor GKE Environment.

Intercepts GKEEnvironment.start() to stage files declared in environment.mounts
into the remote GKE pod using GKEEnvironment's native upload_file() and upload_dir()
methods with secure permissions (chmod 600 for credentials).
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path

try:
    from harbor.environments.gke import GKEEnvironment
except ImportError:
    GKEEnvironment = None

try:
    from harbor.trial.hooks import TrialEvent
    from harbor.trial.trial import Trial as _HarborTrial
except ImportError:
    _HarborTrial = None
    TrialEvent = None

# harbor imports without its `gke` extra; kubernetes then is not installed, and
# no GKE trial can run, so there is nothing to patch.
try:
    from kubernetes.stream import stream
except ImportError:
    stream = None

HOST_REQUIREMENTS = "requirements-client.txt"
# tpu-diagnostics accepts only the xprof release. Drop a system xprof-nightly,
# and xprof with it: they share a package directory, so removing one breaks
# the other. The requirements install then puts xprof back.
DROP_XPROF_NIGHTLY = (
    "if python3 -m pip show -q xprof-nightly >/dev/null 2>&1; then "
    "python3 -m pip uninstall -y -q xprof-nightly xprof; fi; ")
# Large wheels (google-cloud-aiplatform, xprof) on a cold pod.
HOST_REQUIREMENTS_TIMEOUT_SEC = 900


async def _install_host_requirements(env, mounts) -> None:
    """Install the staged package's host requirements into the pod's python3.

    The package's session-start note tells the agent to stop until they are
    installed, and a task image carries only what its own checks need, so
    without this every trial stops before its first step. The file is the
    staged package's own, trimmed to the skills that build holds. Only when
    the staged package was among the mounts; a failed install is logged and
    the trial goes on, where the agent then reports what is missing.
    """
    host_pkg = os.environ.get("HOST_CENTRAL_PKG")
    pod_pkg = os.environ.get("HOST_CENTRAL_PKG_TARGET")
    if not host_pkg or not pod_pkg:
        return
    if not (Path(host_pkg) / HOST_REQUIREMENTS).is_file():
        return
    if not any(
            os.path.expandvars(mount.get("target") or "") == pod_pkg
            for mount in mounts):
        return
    requirements = f"{pod_pkg}/{HOST_REQUIREMENTS}"
    env.logger.info(
        f"[GKE-Mount-Adapter] Installing {requirements} into pod '{env.pod_name}'..."
    )
    result = await env.exec(
        DROP_XPROF_NIGHTLY +
        "python3 -m pip install --no-cache-dir --quiet -r "
        f"{shlex.quote(requirements)}",
        # An env var rather than --root-user-action, which a pip older than
        # 22.1 rejects; an older pip ignores the variable.
        env={"PIP_ROOT_USER_ACTION": "ignore"},
        user="root",
        timeout_sec=HOST_REQUIREMENTS_TIMEOUT_SEC,
    )
    if result.return_code != 0:
        detail = (result.stderr or result.stdout or "").strip()[-2000:]
        env.logger.warning(
            f"[GKE-Mount-Adapter] Installing {requirements} failed "
            f"(exit {result.return_code}); the agent will be told what is "
            f"missing: {detail}")


try:
    from harbor.environments.docker.docker import DockerEnvironment
except ImportError:
    DockerEnvironment = None

KUBECTL_WRAPPER_HOST_PATH = (Path(__file__).resolve().parent.parent /
                             "kubectl_wrapper.py")
KUBECTL_WRAPPER_CONTAINER_PATH = "/usr/libexec/kubectl-wrapper"
KUBE_GATE_SETUP_SCRIPT_PATH = (Path(__file__).resolve().parent /
                               "kube_gate_setup.sh")


def _trial_id_for_env(env) -> str:
    """Return a valid Kubernetes label value (<=63 chars) identifying the trial."""
    raw = (getattr(env, "pod_name", None) or getattr(env, "session_id", None)
           or "trial")
    sanitized = re.sub(r"[^a-z0-9A-Z_.-]+", "-", str(raw)).strip("-_.")
    if not sanitized:
        sanitized = "trial"
    if len(sanitized) <= 63:
        return sanitized
    digest = hashlib.sha256(str(raw).encode("utf-8")).hexdigest()[:8]
    prefix = sanitized[:63 - len(digest) - 1].rstrip("-_.")
    return f"{prefix}-{digest}"


async def _setup_kube_gate(env) -> None:
    """Provision `agent` and `kube-gate` isolation inside the container.

    Works on both newly built `nexus-host` images and older pinned task images:
    ensures `agent` (uid 1000) and `kube-gate` (uid 1001) exist, installs
    `/usr/libexec/kubectl-wrapper` and the `/usr/local/bin/kubectl` sudo stub,
    moves the mounted kubeconfig and copies ADC into `/etc/kube-gate/` (0600,
    owned by `kube-gate:kube-gate`), and writes `/etc/kube-gate/trial_id`.
    The shell steps are in `kube_gate_setup.sh`, which reads the trial ID from
    `KUBE_GATE_TRIAL_ID`.
    """
    if KUBECTL_WRAPPER_HOST_PATH.is_file():
        # Root-only directory: see the note in kube_gate_setup.sh.
        await env.exec("mkdir -p /usr/libexec", user="root")
        await env.upload_file(KUBECTL_WRAPPER_HOST_PATH,
                              KUBECTL_WRAPPER_CONTAINER_PATH)

    trial_id = _trial_id_for_env(env)
    setup_script = (f"KUBE_GATE_TRIAL_ID={shlex.quote(trial_id)}\n" +
                    KUBE_GATE_SETUP_SCRIPT_PATH.read_text())
    res = await env.exec(setup_script, user="root")
    if res.return_code != 0:
        env.logger.warning(
            f"[Kube-Gate-Adapter] Setup exited {res.return_code}: "
            f"{(res.stderr or res.stdout or '').strip()}")


# ------------------------------------------------------------------------------
# GKE Environment Mount Staging Adapter
# ------------------------------------------------------------------------------
if GKEEnvironment is not None and not getattr(
        GKEEnvironment, "_eval_gke_mounts_patched", False):
    _orig_gke_start = GKEEnvironment.start

    @functools.wraps(_orig_gke_start)
    async def _staged_gke_start(self, force_build: bool) -> None:
        # Execute standard GKE start (pod creation, wait for exec ready, etc.)
        await _orig_gke_start(self, force_build)

        mounts = getattr(self, "_mounts", None)
        if not mounts:
            await _setup_kube_gate(self)
            return

        self.logger.info(
            f"[GKE-Mount-Adapter] Staging {len(mounts)} mount entries into pod '{self.pod_name}'..."
        )

        for mount in mounts:
            raw_source = mount.get("source")
            raw_target = mount.get("target")
            if not raw_source or not raw_target:
                continue

            # Expand environment variables using the current host environment
            source_str = os.path.expandvars(raw_source)
            target_str = os.path.expandvars(raw_target)
            source_path = Path(source_str)

            if not source_path.exists():
                raise FileNotFoundError(
                    f"[GKE-Mount-Adapter] Required mount source path does not exist: '{source_path}' (raw: '{raw_source}')"
                )

            # Ensure destination parent directory exists inside the container
            parent_dir = target_str if source_path.is_dir() else str(
                Path(target_str).parent)
            await self.exec(f"mkdir -p '{parent_dir}'", user="root")

            self.logger.info(
                f"[GKE-Mount-Adapter] Uploading: '{source_path}' -> '{target_str}'"
            )
            try:
                if source_path.is_dir():
                    await self.upload_dir(source_path, target_str)
                else:
                    await self.upload_file(source_path, target_str)

                # Set restrictive permissions (600) on credentials inside the container
                if "adc.json" in target_str or ("config" in target_str
                                                and ".kube" in target_str):
                    await self.exec(
                        f"chmod 600 '{target_str}' 2>/dev/null || true",
                        user="root",
                    )
                elif mount.get("read_only"):
                    await self.exec(
                        f"chmod -R a+rX '{target_str}' 2>/dev/null || true",
                        user="root",
                    )
                else:
                    await self.exec(
                        f"chmod -R a+rwX '{target_str}' 2>/dev/null || true",
                        user="root",
                    )

                if target_str.endswith(".sh") or "/bin/" in target_str:
                    await self.exec(
                        f"chmod a+x '{target_str}' 2>/dev/null || true",
                        user="root",
                    )
            except Exception as e:
                self.logger.error(
                    f"[GKE-Mount-Adapter] Failed to upload '{source_path}' -> '{target_str}': {e}"
                )
                raise

        self.logger.info(
            f"[GKE-Mount-Adapter] All mounts successfully staged into pod '{self.pod_name}'."
        )
        await _install_host_requirements(self, mounts)
        await _setup_kube_gate(self)

    GKEEnvironment.start = _staged_gke_start
    GKEEnvironment._eval_gke_mounts_patched = True

if DockerEnvironment is not None and not getattr(
        DockerEnvironment, "_eval_docker_kube_gate_patched", False):
    _orig_docker_start = DockerEnvironment.start

    @functools.wraps(_orig_docker_start)
    async def _kube_gate_docker_start(self, force_build: bool) -> None:
        await _orig_docker_start(self, force_build)
        await _setup_kube_gate(self)

    DockerEnvironment.start = _kube_gate_docker_start
    DockerEnvironment._eval_docker_kube_gate_patched = True

# ------------------------------------------------------------------------------
# GKE Tar Transfer Integrity & Chunked Upload Adapter
# ------------------------------------------------------------------------------
# 1. GKEEnvironment.download_dir() and download_file() read `tar cf -` from a
#    text-mode exec websocket, which UTF-8 decodes each frame on its own with
#    errors="replace". A multi-byte character split across two frames (a
#    progress bar's "█" in claude-code.txt) comes back as two U+FFFD, which
#    shifts every byte after it. tarfile then reads the next, garbled header as
#    the end of the archive and stops without an error, so every member after it
#    is dropped -- for Claude Code, the session files Harbor builds
#    trajectory.json from.
# 2. GKEEnvironment.upload_dir() and upload_file() send the entire uncompressed
#    tar buffer in a single `resp.write_stdin(tar_buffer.read())` WebSocket
#    frame. When `/logs/agent` exceeds the GKE/Kubernetes WebSocket single-frame
#    size limit (~32 MiB, e.g. with `antigravity-conversation.db-wal`), the
#    server drops the TLS connection (`ssl.SSLEOFError` /
#    `WebSocketConnectionClosedException: socket is already closed`).
#
# These replacements read/write tar streams as binary frames, chunk stdin writes
# into 64 KiB frames, and require tar's exit status so truncated transfers fail
# and retry cleanly.
_DOWNLOAD_ATTEMPTS = 3
_UPLOAD_ATTEMPTS = 3
_UPLOAD_CHUNK_BYTES = 64 * 1024
_UPLOAD_DRAIN_TIMEOUT_SEC = 30.0

if GKEEnvironment is not None and stream is not None and not getattr(
        GKEEnvironment, "_eval_gke_download_patched", False):
    import asyncio
    import io
    import shlex
    import tarfile
    import time

    from harbor.environments.tar_transfer import (extract_dir_from_bytes,
                                                  pack_dir_to_bytes)

    class _TarExitError(RuntimeError):
        """tar ran to completion and failed: retrying would not help."""

    def _validate_tar_bytes(raw: bytes) -> bytes:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
            tar.getmembers()
            last_end = tar.offset
        if len(raw) - last_end < 1024 or any(raw[last_end:last_end + 1024]):
            raise tarfile.ReadError(
                "incomplete tar archive (missing EOF blocks)")
        return raw

    def _read_tar_bytes(self, resp, command: list[str]) -> bytes:
        stdout = bytearray()
        stderr = bytearray()
        try:
            while resp.is_open():
                resp.update(timeout=1)
                if resp.peek_stdout():
                    stdout += resp.read_stdout()
                if resp.peek_stderr():
                    stderr += resp.read_stderr()
            # Raises when the stream closed before tar's exit status arrived.
            return_code = self._exec_return_code(resp)
        finally:
            resp.close()
        if return_code != 0:
            err_text = stderr.decode("utf-8", "replace").strip()
            # GNU tar exits 1 ("Some files differed" / "file changed as we read
            # it") when a live log or SQLite WAL file is appended to during
            # archiving. The tar stream on stdout is still valid and complete.
            if return_code == 1 and stdout:
                self.logger.warning(
                    f"[GKE-Download-Adapter] `{' '.join(command)}` exited 1"
                    f" (live file modified during read) in pod"
                    f" '{self.pod_name}'; keeping {len(stdout)} archive bytes:"
                    f" {err_text}")
                return _validate_tar_bytes(bytes(stdout))
            raise _TarExitError(
                f"`{' '.join(command)}` exited {return_code} in pod"
                f" {self.pod_name}: {err_text}")
        return _validate_tar_bytes(bytes(stdout))

    def _write_tar_bytes(self, resp, tar_data: bytes,
                         command: list[str]) -> None:
        stderr = bytearray()
        try:
            mv = memoryview(tar_data)
            for offset in range(0, len(mv), _UPLOAD_CHUNK_BYTES):
                resp.write_stdin(bytes(mv[offset:offset +
                                          _UPLOAD_CHUNK_BYTES]))
                if resp.peek_stderr():
                    stderr += resp.read_stderr()
            resp.close_channel(0)
            deadline = time.monotonic() + _UPLOAD_DRAIN_TIMEOUT_SEC
            while resp.is_open():
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out after {_UPLOAD_DRAIN_TIMEOUT_SEC}s waiting"
                        f" for `{' '.join(command)}` to exit in pod"
                        f" {self.pod_name}")
                resp.update(timeout=1)
                if resp.peek_stderr():
                    stderr += resp.read_stderr()
            # Raises when the stream closed before tar's exit status arrived.
            return_code = self._exec_return_code(resp)
        finally:
            resp.close()
        if return_code != 0:
            raise _TarExitError(
                f"`{' '.join(command)}` exited {return_code} in pod"
                f" {self.pod_name}: {stderr.decode('utf-8', 'replace').strip()}"
            )

    async def _download_tar(self, command: list[str]) -> bytes:
        await self._ensure_client()
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            try:
                resp = await asyncio.to_thread(
                    stream,
                    self._api.connect_get_namespaced_pod_exec,
                    self.pod_name,
                    self.namespace,
                    command=command,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False,
                    binary=True,
                    _preload_content=False,
                )
                return await asyncio.to_thread(_read_tar_bytes, self, resp,
                                               command)
            except Exception as e:
                if isinstance(e,
                              _TarExitError) or attempt == _DOWNLOAD_ATTEMPTS:
                    raise
                self.logger.warning(
                    f"[GKE-Download-Adapter] `{' '.join(command)}` from pod"
                    f" '{self.pod_name}' failed (attempt {attempt}/"
                    f"{_DOWNLOAD_ATTEMPTS}), retrying: {e}")
                await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")

    async def _upload_tar(self, tar_data: bytes, target_dir: str) -> None:
        await self._ensure_client()
        command = ["tar", "xf", "-", "-C", target_dir]
        dir_created = False
        for attempt in range(1, _UPLOAD_ATTEMPTS + 1):
            try:
                if not dir_created:
                    await self._wait_for_container_exec_ready()
                    mkdir_result = await self.exec(
                        f"mkdir -p {shlex.quote(target_dir)}", user="root")
                    if mkdir_result.return_code != 0:
                        raise _TarExitError(
                            f"Failed to create target directory {target_dir}:"
                            f" {mkdir_result.stderr}")
                    dir_created = True
                resp = await asyncio.to_thread(
                    stream,
                    self._api.connect_get_namespaced_pod_exec,
                    self.pod_name,
                    self.namespace,
                    command=command,
                    stderr=True,
                    stdin=True,
                    stdout=True,
                    tty=False,
                    binary=True,
                    _preload_content=False,
                )
                await asyncio.to_thread(_write_tar_bytes, self, resp, tar_data,
                                        command)
                return
            except Exception as e:
                if isinstance(e, _TarExitError) or attempt == _UPLOAD_ATTEMPTS:
                    raise
                self.logger.warning(
                    f"[GKE-Upload-Adapter] `{' '.join(command)}` to pod"
                    f" '{self.pod_name}' failed (attempt {attempt}/"
                    f"{_UPLOAD_ATTEMPTS}), retrying: {e}")
                await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")

    _orig_gke_download_dir = GKEEnvironment.download_dir
    _orig_gke_download_file = GKEEnvironment.download_file
    _orig_gke_upload_dir = GKEEnvironment.upload_dir
    _orig_gke_upload_file = GKEEnvironment.upload_file

    @functools.wraps(_orig_gke_download_dir)
    async def _binary_gke_download_dir(self, source_dir: str,
                                       target_dir: Path | str):
        if self._compose_mode:
            return await _orig_gke_download_dir(self, source_dir, target_dir)
        tar_data = await _download_tar(
            self, ["sh", "-c", f"cd {shlex.quote(source_dir)} && tar cf - ."])
        extract_dir_from_bytes(tar_data, target_dir)

    @functools.wraps(_orig_gke_download_file)
    async def _binary_gke_download_file(self, source_path: str,
                                        target_path: Path | str):
        if self._compose_mode:
            return await _orig_gke_download_file(self, source_path,
                                                 target_path)
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tar_data = await _download_tar(self, ["tar", "cf", "-", source_path])
        with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r") as tar:
            for member in tar.getmembers():
                if member.name == source_path or member.name.startswith(
                        source_path.lstrip("/")):
                    member.name = target_path.name
                    tar.extract(member,
                                path=str(target_path.parent),
                                filter="data")
                    break

    @functools.wraps(_orig_gke_upload_dir)
    async def _chunked_gke_upload_dir(self, source_dir: Path | str,
                                      target_dir: str):
        if self._compose_mode:
            return await _orig_gke_upload_dir(self, source_dir, target_dir)
        source_dir = Path(source_dir)
        if not source_dir.is_dir():
            self.logger.warning(f"No files to upload from {source_dir}")
            return
        tar_data = pack_dir_to_bytes(source_dir).getvalue()
        await _upload_tar(self, tar_data, target_dir)
        self.logger.debug(
            f"Successfully uploaded {source_dir} ({len(tar_data)} bytes) to"
            f" {target_dir}")

    @functools.wraps(_orig_gke_upload_file)
    async def _chunked_gke_upload_file(self, source_path: Path | str,
                                       target_path: str):
        if self._compose_mode:
            return await _orig_gke_upload_file(self, source_path, target_path)
        source_path = Path(source_path)
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            tar.add(str(source_path), arcname=Path(target_path).name)
        target_dir = str(Path(target_path).parent)
        await _upload_tar(self, tar_buffer.getvalue(), target_dir)

    GKEEnvironment.download_dir = _binary_gke_download_dir
    GKEEnvironment.download_file = _binary_gke_download_file
    GKEEnvironment.upload_dir = _chunked_gke_upload_dir
    GKEEnvironment.upload_file = _chunked_gke_upload_file
    GKEEnvironment._eval_gke_download_patched = True

# ------------------------------------------------------------------------------
# GKE Durable Exec Adapter
# ------------------------------------------------------------------------------
# GKEEnvironment.exec() runs the command as the exec websocket's own process,
# so the command lives exactly as long as that one connection. On Autopilot the
# connection runs apiserver -> konnectivity-agent -> kubelet, and the cluster
# autoscaler drains konnectivity-agent pods whenever it removes a node. Every
# exec tunnelled through a drained agent closes, and the apiserver reports the
# closed stream as a successful exit. An agent run hours long then "finishes"
# with exit 0 mid tool call: Harbor records no exception, downloads the logs and
# scores a half-done trial (four trials cut at the same second on 2026-10-06).
#
# Here the command runs detached in the pod, its output and exit status written
# to files, and the harness attaches to it with short polling execs. A dropped
# poll is retried, and only a status the command itself wrote ends the wait.
_DURABLE_EXEC_DIR = "/tmp/.eval-exec"
# Each poll attaches for at most this long, so a stream that died without
# failing its keepalive ping costs one window, not the run.
_DURABLE_EXEC_POLL_SEC = 600
_DURABLE_EXEC_ATTEMPTS = 10

if GKEEnvironment is not None and not getattr(
        GKEEnvironment, "_eval_gke_durable_exec_patched", False):
    import asyncio
    import shlex
    import uuid

    from harbor.environments.base import ExecResult
    from harbor.environments.gke import GKEExecStreamClosedError

    _orig_gke_exec = GKEEnvironment.exec

    def _durable_launch_script(command: str, run_dir: str) -> str:
        # `mkdir` without -p makes a retried launch a no-op once one attempt
        # has started the command. The outer shell records the status, so an
        # `exit` inside the command cannot skip it.
        d = shlex.quote(run_dir)
        exec_dir = shlex.quote(_DURABLE_EXEC_DIR)
        body = (f"bash -c {shlex.quote(command)}; echo $? > {d}/rc.tmp"
                f" && mv {d}/rc.tmp {d}/rc")
        return (
            f"mkdir -p -m 1777 {exec_dir} && (chmod 1777 {exec_dir} 2>/dev/null"
            f" || true) && if mkdir {d} 2>/dev/null; then"
            f" if command -v setsid >/dev/null 2>&1; then s=setsid; else"
            f" set -m; s=; fi; $s bash -c {shlex.quote(body)} </dev/null"
            f" >{d}/out 2>{d}/err & echo $! > {d}/pid; fi")

    def _durable_wait_script(run_dir: str, sentinel: str, missing: str,
                             max_wait_sec: int) -> str:
        # Polls fast at first, so a short command returns about as soon as it
        # would have attached directly. A process gone without a status was
        # killed outright, and is reported as 255.
        d = shlex.quote(run_dir)
        return (
            f"if [ ! -f {d}/pid ]; then echo {shlex.quote(missing)}; exit 0;"
            f" fi; while [ ! -f {d}/rc ] && [ $SECONDS -lt {max_wait_sec} ];"
            f" do if ! kill -0 \"$(cat {d}/pid)\" 2>/dev/null; then sleep 1;"
            f" if [ ! -f {d}/rc ]; then echo 255 > {d}/rc;"
            f" echo '[GKE-Exec-Adapter] command exited without recording a status'"
            f" >> {d}/err; fi; break; fi;"
            f" if [ $SECONDS -lt 10 ]; then sleep 0.2; else sleep 2; fi; done;"
            f" if [ -f {d}/rc ]; then cat {d}/err >&2; cat {d}/out;"
            f" printf '\\n%s%s\\n' {shlex.quote(sentinel)} \"$(cat {d}/rc)\";"
            f" fi")

    def _split_sentinel(stdout: str | None,
                        sentinel: str) -> tuple[str, int] | None:
        stdout = stdout or ""
        idx = stdout.rfind("\n" + sentinel)
        if idx < 0:
            return None
        try:
            return_code = int(stdout[idx + 1 + len(sentinel):].strip())
        except ValueError:
            return None
        return stdout[:idx], return_code

    @functools.wraps(_orig_gke_exec)
    async def _durable_gke_exec(self,
                                command: str,
                                cwd: str | None = None,
                                env: dict[str, str] | None = None,
                                timeout_sec: int | None = None,
                                user: str | int | None = None) -> ExecResult:
        # GKEEnvironment.exec wraps non-root users in `su <user> -s /bin/bash -c`
        # without `-`. Set HOME, USER, and LOGNAME explicitly when running as
        # `agent` (whose home is `/root`), so `~`, `$HOME` and `$USER` are right
        # whatever the image's `su` passes through.
        effective_user = (user if user is not None else getattr(
            self, "default_user", None))
        if str(effective_user) in ("agent", "1000"):
            env = {
                "HOME": "/root",
                "USER": "agent",
                "LOGNAME": "agent",
                **(env or {}),
            }

        if self._compose_mode:
            return await _orig_gke_exec(self,
                                        command,
                                        cwd=cwd,
                                        env=env,
                                        timeout_sec=timeout_sec,
                                        user=user)

        token = uuid.uuid4().hex
        run_dir = f"{_DURABLE_EXEC_DIR}/{token}"
        sentinel = f"__EVAL_EXEC_{token}_RC="
        missing = f"__EVAL_EXEC_{token}_MISSING"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_sec if timeout_sec else None

        async def _run(script: str, **kwargs) -> ExecResult | None:
            """One attach; None when the stream closed under it."""
            try:
                return await _orig_gke_exec(self, script, user=user, **kwargs)
            except GKEExecStreamClosedError as e:
                self.logger.warning(
                    f"[GKE-Exec-Adapter] Exec stream to pod '{self.pod_name}'"
                    f" closed early: {e}")
                return None

        async def _kill() -> None:
            d = shlex.quote(run_dir)
            try:
                await asyncio.wait_for(
                    _orig_gke_exec(
                        self,
                        f"p=$(cat {d}/pid 2>/dev/null) && {{ kill -TERM -- -$p"
                        f" 2>/dev/null || {{ pkill -TERM -P $p; kill -TERM $p;"
                        f" }} 2>/dev/null; for _ in $(seq 1 30); do"
                        f" kill -0 -- -$p 2>/dev/null || kill -0 $p"
                        f" 2>/dev/null || break; sleep 0.5; done; }}; true",
                        user=user), 30)
            except Exception as e:
                self.logger.warning(
                    f"[GKE-Exec-Adapter] Could not stop {run_dir} in pod"
                    f" '{self.pod_name}': {e}")

        try:
            launched = False
            failures = 0
            last_error: object = None
            while True:
                if not launched:
                    launch = await _run(_durable_launch_script(
                        command, run_dir),
                                        cwd=cwd,
                                        env=env)
                    if launch is not None and launch.return_code != 0:
                        # Never started (pod not ready, bad user): report it
                        # as the direct exec would have.
                        return launch
                    launched = launch is not None

                if launched:
                    max_wait = _DURABLE_EXEC_POLL_SEC
                    if deadline is not None:
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            await _kill()
                            return ExecResult(stdout=None,
                                              stderr=f"Command timed out after"
                                              f" {timeout_sec} seconds",
                                              return_code=124)
                        max_wait = max(1, min(max_wait, int(remaining) + 1))
                    started = loop.time()
                    poll = await _run(
                        _durable_wait_script(run_dir, sentinel, missing,
                                             max_wait))
                    done = _split_sentinel(poll.stdout if poll else None,
                                           sentinel)
                    if done is not None:
                        stdout, return_code = done
                        try:
                            await asyncio.wait_for(
                                _orig_gke_exec(
                                    self,
                                    f"rm -rf {shlex.quote(run_dir)}",
                                    user=user), 30)
                        except Exception:
                            pass
                        return ExecResult(stdout=stdout,
                                          stderr=poll.stderr,
                                          return_code=return_code)
                    if poll is not None and missing in (poll.stdout or ""):
                        # The launch reported success but never ran.
                        launched = False
                    elif (poll is not None and poll.return_code == 0
                          and loop.time() - started >= max_wait - 1):
                        # Still running when the window closed: attach again.
                        failures = 0
                        continue
                    last_error = poll.stderr if poll else "stream closed"
                else:
                    last_error = "launch stream closed"

                # A cut stream, a failed poll or a launch that never ran. Each
                # is retried; a run of them means the pod is gone.
                failures += 1
                if failures >= _DURABLE_EXEC_ATTEMPTS:
                    return ExecResult(
                        stdout=None,
                        stderr=f"Lost contact with {run_dir} in pod"
                        f" {self.pod_name} after {failures} attempts:"
                        f" {last_error}",
                        return_code=1)
                self.logger.warning(
                    f"[GKE-Exec-Adapter] Re-attaching to {run_dir} in pod"
                    f" '{self.pod_name}' ({failures}/{_DURABLE_EXEC_ATTEMPTS}):"
                    f" {last_error}")
                await asyncio.sleep(min(2**failures, 30))
        except asyncio.CancelledError:
            # Harbor cancels the agent run at its timeout. Stop the detached
            # command too, as it would have died with its stream before.
            await _kill()
            raise

    GKEEnvironment.exec = _durable_gke_exec
    GKEEnvironment._eval_gke_durable_exec_patched = True

# ------------------------------------------------------------------------------
# GKE Image Cache Adapter
# ------------------------------------------------------------------------------
# GKEEnvironment._image_exists() asks `gcloud artifacts docker images describe`
# whether the environment's content-addressed tag is already in the registry.
# describe also looks up the image's Container Analysis metadata, through the
# Service Usage API, which a service account bills to its own project. On the
# CI runner that is ml-velocity-actions-production, which has not enabled it,
# so describe fails, Harbor discards its stderr, reads the failure as "not
# found", and rebuilds and pushes the same tag on every run.
#
# `gcloud artifacts docker tags list` only calls Artifact Registry. A failure
# still falls back to building, but is logged.
if GKEEnvironment is not None and not getattr(
        GKEEnvironment, "_eval_gke_image_check_patched", False):
    import asyncio

    @functools.wraps(GKEEnvironment._image_exists)
    async def _tag_list_image_exists(self) -> bool:
        image, tag = self._get_image_url().rsplit(":", 1)
        proc = await asyncio.create_subprocess_exec(
            "gcloud",
            "artifacts",
            "docker",
            "tags",
            "list",
            image,
            "--project",
            self.project_id,
            f"--filter=tag~/{tag}$",
            "--format=value(tag)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            self.logger.warning(
                f"[GKE-Image-Adapter] Could not check for {image}:{tag}, will"
                f" build it: {stderr.decode('utf-8', 'replace').strip()}")
            return False
        return tag in stdout.decode().split()

    GKEEnvironment._image_exists = _tag_list_image_exists
    GKEEnvironment._eval_gke_image_check_patched = True

# ------------------------------------------------------------------------------
# Antigravity CLI Version Pin
# ------------------------------------------------------------------------------
# Harbor's antigravity-cli agent installs whatever the CLI's installer serves
# as the latest release, so a CLI release would change every trial without a
# change here. Instead each trial container runs .ci/install-agy.sh with the
# pin from .ci/pins.conf: the same script and pin the harness E2E presubmit job
# installs from, so presubmit and the evals exercise one binary. pins.conf
# explains the release choice and the byte patch the script applies. The
# container turns the CLI's self-update off (AGY_CLI_DISABLE_AUTO_UPDATE, set
# by mount_advisor.py and the session supervisor) so the pin holds for the
# whole trial. EVAL_ANTIGRAVITY_CLI_VERSION=latest restores Harbor's own
# install.
CI_PINS_DIR = Path(__file__).resolve().parents[2] / ".ci"
HARNESS_PINS_FILE = CI_PINS_DIR / "pins.conf"
ANTIGRAVITY_CLI_INSTALL_SCRIPT = CI_PINS_DIR / "install-agy.sh"
ANTIGRAVITY_CLI_VERSION_ENV = "EVAL_ANTIGRAVITY_CLI_VERSION"


def antigravity_cli_pin_requested() -> bool:
    """False only when the run asked for the installer's latest release."""
    requested = os.environ.get(ANTIGRAVITY_CLI_VERSION_ENV, "")
    return requested.strip().lower() != "latest"


def read_harness_pins(path: Path = HARNESS_PINS_FILE) -> dict[str, str]:
    """The KEY=VALUE pairs of .ci/pins.conf, without comments or blank lines."""
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"{path}: expected KEY=VALUE, got {raw!r}")
        pins[key.strip()] = value.strip()
    return pins


def antigravity_cli_install_command(
        pins_file: Path = HARNESS_PINS_FILE,
        script: Path = ANTIGRAVITY_CLI_INSTALL_SCRIPT) -> str:
    """The shell script a trial container runs to install the pinned CLI.

    install-agy.sh sources pins.conf from beside itself, and the container has
    neither file, so the pin's assignments are prepended and the script finds
    its AGY_* variables already set.
    """
    return (pins_file.read_text(encoding="utf-8") + "\n" +
            script.read_text(encoding="utf-8"))


try:
    from harbor.agents.installed.antigravity_cli import AntigravityCli
except ImportError:
    AntigravityCli = None

if AntigravityCli is not None and not getattr(AntigravityCli,
                                              "_eval_cli_pin_patched", False):
    _orig_antigravity_install = AntigravityCli.install
    _orig_validate_reasoning_effort = AntigravityCli._validate_reasoning_effort

    @functools.wraps(_orig_antigravity_install)
    async def _pinned_antigravity_install(self, environment) -> None:
        if not antigravity_cli_pin_requested():
            await _orig_antigravity_install(self, environment)
            return
        await self.ensure_system_dependencies(environment, ("curl", "python3"))
        # Under bash whatever shell the environment execs with: the script
        # relies on pipefail and `case` patterns, as the upstream one does.
        await self.exec_as_agent(
            environment,
            command=f"bash -c {shlex.quote(antigravity_cli_install_command())}"
        )

    @staticmethod
    def _patched_validate_reasoning_effort(
        reasoning_effort: str | None,
        model_name: str | None,
    ) -> None:
        # Gemini 4 (gemini-4-argon) supports low, medium, and high reasoning
        # efforts, whereas Harbor 0.23.0 restricts medium to flash models.
        if (reasoning_effort == "medium" and model_name is not None
                and ("gemini-4" in model_name or "argon" in model_name)):
            return
        _orig_validate_reasoning_effort(reasoning_effort, model_name)

    AntigravityCli.install = _pinned_antigravity_install
    AntigravityCli._validate_reasoning_effort = (
        _patched_validate_reasoning_effort)
    AntigravityCli._eval_cli_pin_patched = True


# ------------------------------------------------------------------------------
# Trial Pre-Upload Scrubbing Adapter
# ------------------------------------------------------------------------------
# Harbor's Trial.run() calls `await self._finalize()` (which emits
# `TrialEvent.END` and triggers `HarborHubUploadPlugin._streaming_upload_cb`)
# before calling `self._scrub_jobs_dir()`. Running `_scrub_jobs_dir()` right
# before `TrialEvent.END` is emitted ensures sensitive env values are scrubbed
# from `trial_dir` before per-trial streaming upload archives the trial.
#
# - If Harbor renames `Trial._emit` or `Trial._scrub_jobs_dir`, `sitecustomize`
#   raises `RuntimeError` at Harbor startup rather than silently running
#   unpatched.
# - If `_scrub_jobs_dir()` throws on `TrialEvent.END`, it logs a warning and
#   returns without emitting `TrialEvent.END`, skipping the streaming upload
#   hook for that trial.
def _patch_trial_pre_emit_scrub(trial_cls: object,
                                trial_event_cls: object) -> None:
    if (trial_cls is None or trial_event_cls is None
            or getattr(trial_cls, "_eval_pre_emit_scrub_patched", False)):
        return
    import logging as _logging

    if not callable(getattr(trial_cls, "_emit", None)) or not callable(
            getattr(trial_cls, "_scrub_jobs_dir", None)):
        msg = ("[Trial-Scrub-Adapter] Harbor Trial is missing _emit or"
               " _scrub_jobs_dir; refusing to run without pre-upload secret"
               " scrubbing.")
        _logging.getLogger(__name__).warning(msg)
        raise RuntimeError(msg)

    orig_trial_emit = trial_cls._emit

    @functools.wraps(orig_trial_emit)
    async def _scrubbing_trial_emit(self, event: TrialEvent) -> None:
        if event == trial_event_cls.END:
            logger = getattr(self, "logger",
                             None) or _logging.getLogger(__name__)
            scrub_fn = getattr(self, "_scrub_jobs_dir", None)
            if not callable(scrub_fn):
                logger.warning(
                    "[Trial-Scrub-Adapter] Harbor Trial._scrub_jobs_dir not"
                    " found; skipping TrialEvent.END to prevent unscrubbed"
                    " trial upload.")
                return
            try:
                scrub_fn()
            except Exception as e:
                logger.warning(
                    f"[Trial-Scrub-Adapter] Pre-upload scrub failed; skipping"
                    f" TrialEvent.END to prevent unscrubbed trial upload: {e}")
                return
        await orig_trial_emit(self, event)

    trial_cls._emit = _scrubbing_trial_emit
    trial_cls._eval_pre_emit_scrub_patched = True


_patch_trial_pre_emit_scrub(_HarborTrial, TrialEvent)


# ------------------------------------------------------------------------------
# Trial Non-Root Agent User Adapter
# ------------------------------------------------------------------------------
# Runs the agent phase (`agent.setup()` and `agent.run()`) as the non-root
# `agent` user (`uid=1000`, `/root`) when `task.config.agent.user` is not
# explicitly set, while leaving `task.config.verifier.user` as `None` (root) so
# `/tests/test.sh` and `rewardkit` can write `/logs/verifier`.
def _patch_trial_agent_user(trial_cls: object) -> None:
    if trial_cls is None or getattr(trial_cls, "_eval_agent_user_patched",
                                    False):
        return
    orig_prepare = getattr(trial_cls, "_prepare", None)
    if not callable(orig_prepare):
        return

    @functools.wraps(orig_prepare)
    async def _agent_user_prepare(self) -> None:
        task_cfg = getattr(getattr(self, "task", None), "config", None)
        agent_cfg = getattr(task_cfg, "agent", None)
        if agent_cfg is not None and getattr(agent_cfg, "user", None) is None:
            agent_cfg.user = os.environ.get("EVAL_AGENT_USER", "agent")
        task_env = getattr(getattr(task_cfg, "environment", None), "env",
                           None) or {}
        task_persona = task_env.get("TASK_AGENT_PERSONA")
        agent_obj = getattr(self, "agent", None)
        extra_env = getattr(agent_obj, "extra_env", None)
        if (task_persona and isinstance(extra_env, dict)
                and "AGENT_PERSONA" in extra_env):
            extra_env["TASK_AGENT_PERSONA"] = str(task_persona)
        await orig_prepare(self)

    trial_cls._prepare = _agent_user_prepare
    trial_cls._eval_agent_user_patched = True


_patch_trial_agent_user(_HarborTrial)

# Trial Agent-Suite Artifact Staging Adapter
# The TPU agent suite writes run outputs under
# `/root/.tpu-agent-suite/runs/<run_id>/artifacts/` (`check_completed_has_artifact`
# and `rooted` require specialists to keep outputs inside their run root), and
# the top-level `tpu-agent` router has `tools: [read, ask]` so it cannot shell
# out to copy them into Harbor's `/logs/artifacts/`.
#
# Intercepting `Trial._collect_artifacts_phased` copies each run's `artifacts/`
# directory into `agent_env_paths.artifacts_dir` (`/logs/artifacts`) inside the
# agent container right after the agent completes and before Harbor:
#   1. collects task-declared artifacts,
#   2. downloads `/logs/artifacts` to `<trial_dir>/artifacts/logs/artifacts/`,
#   3. uploads `/logs/artifacts` into a separate verifier container (if used),
#   4. runs the verifier (`test.sh`, `check.py`, `judge.toml`).
SUITE_RUNS_DIR = "/root/.tpu-agent-suite/runs"
DEFAULT_ENV_ARTIFACTS_DIR = "/logs/artifacts"
STAGE_ARTIFACTS_TIMEOUT_SEC = 60


def _stage_suite_artifacts_command(
        runs_dir: str = SUITE_RUNS_DIR,
        target_dir: str = DEFAULT_ENV_ARTIFACTS_DIR) -> str:
    quoted_runs = shlex.quote(runs_dir)
    quoted_target = shlex.quote(target_dir)
    return ("for d in " + quoted_runs + "/*/artifacts; do "
            '[ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ] || continue; '
            f"mkdir -p {quoted_target} && cp -rf \"$d\"/. {quoted_target}/; "
            "done")


def _patch_trial_stage_suite_artifacts(trial_cls: object) -> None:
    if trial_cls is None or getattr(
            trial_cls, "_eval_stage_suite_artifacts_patched", False):
        return
    import logging as _logging

    if not callable(getattr(trial_cls, "_collect_artifacts_phased", None)):
        msg = ("[Trial-Artifacts-Adapter] Harbor Trial is missing"
               " _collect_artifacts_phased; refusing to run without"
               " agent-suite artifact staging.")
        _logging.getLogger(__name__).warning(msg)
        raise RuntimeError(msg)

    orig_collect = trial_cls._collect_artifacts_phased

    @functools.wraps(orig_collect)
    async def _staging_collect_artifacts_phased(self, *, artifacts_dir: Path,
                                                **kwargs) -> None:
        env = getattr(self, "agent_environment", None)
        if env is not None and callable(getattr(env, "exec", None)):
            logger = getattr(self, "logger",
                             None) or _logging.getLogger(__name__)
            env_paths = getattr(self, "agent_env_paths", None)
            env_artifacts = getattr(env_paths, "artifacts_dir", None)
            target_dir = (env_artifacts.as_posix() if hasattr(
                env_artifacts, "as_posix") else str(
                    env_artifacts or DEFAULT_ENV_ARTIFACTS_DIR))
            try:
                result = await env.exec(
                    _stage_suite_artifacts_command(SUITE_RUNS_DIR, target_dir),
                    user="root",
                    timeout_sec=STAGE_ARTIFACTS_TIMEOUT_SEC,
                )
                if getattr(result, "return_code", 0) != 0:
                    detail = (getattr(result, "stderr", None)
                              or getattr(result, "stdout", None)
                              or "").strip()[-500:]
                    logger.warning(
                        f"[Trial-Artifacts-Adapter] Staging suite artifacts"
                        f" into {target_dir} exited"
                        f" {result.return_code}: {detail}")
            except Exception as e:
                logger.warning(
                    f"[Trial-Artifacts-Adapter] Failed to stage suite"
                    f" artifacts into {target_dir}: {e}")
        await orig_collect(self, artifacts_dir=artifacts_dir, **kwargs)

    trial_cls._collect_artifacts_phased = _staging_collect_artifacts_phased
    trial_cls._eval_stage_suite_artifacts_patched = True


_patch_trial_stage_suite_artifacts(_HarborTrial)

# ------------------------------------------------------------------------------
# Trial Fallback Trajectory Adapter (RC5 / RC6)
# ------------------------------------------------------------------------------
# 1. When an agent CLI crashes early (auth/model/quota/startup failure) before
#    writing its native session transcript (`antigravity-cli.trajectory.jsonl`
#    or `sessions/projects/<slug>/<sid>.jsonl`), Harbor's
#    `populate_context_post_run` returns without writing
#    `agent/trajectory.json` even though stream/stderr logs exist in
#    `agent_dir`.
# 2. When `Trial._prepare()` fails before `_run_agent_phase()`
#    (`EnvironmentStartTimeoutError`, `AgentSetupTimeoutError`, Cloud Build or
#    image pull `RuntimeError`), `self.result.agent_result` remains `None` and
#    `Trial._populate_agent_context` returns immediately without producing
#    `agent/trajectory.json`.
#
# This adapter:
# - Initializes `target.agent_result = AgentContext()` in `_sync_agent_output`
#   if `None` so `_recover_outputs()` does not skip `_populate_agent_context`.
# - Stages fallback native transcripts from `antigravity-stream.jsonl` /
#   `antigravity-cli.err` or `claude-code.txt` before
#   `populate_context_post_run` executes so Harbor's native ATIF converters run
#   whenever stream logs are present.
# - Synthesizes a valid ATIF v1.7 diagnostic `trajectory.json` when no
#   `trajectory.json` exists after `populate_context_post_run` (including
#   pre-agent environment/setup failures).
_FALLBACK_TAIL_CHARS = 4000


def _iso_utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _trial_instruction(trial: object) -> str:
    task = getattr(trial, "task", None)
    instruction = getattr(task, "instruction", None)
    if isinstance(instruction, str) and instruction.strip():
        return instruction
    return "[Trial task instruction unavailable]"


def _agy_file_has_valid_transcript(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if path.suffix == ".json":
        try:
            obj = json.loads(text)
            if (isinstance(obj, dict) and isinstance(obj.get("messages"), list)
                    and obj["messages"]):
                return True
        except Exception:
            pass
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if (isinstance(rec, dict) and isinstance(rec.get("step_index"), int)
                and not isinstance(rec.get("step_index"), bool)
                and str(rec.get("type")
                        or "").upper() in ("USER_INPUT", "PLANNER_RESPONSE")):
            return True
    return False


def _stage_agy_fallback_transcript(trial: object, logs_dir: Path) -> None:
    for candidate in ("antigravity-cli.trajectory.jsonl",
                      "antigravity-cli.trajectory.json"):
        if _agy_file_has_valid_transcript(logs_dir / candidate):
            return

    existing_jsonl = logs_dir / "antigravity-cli.trajectory.jsonl"
    stream_path = logs_dir / "antigravity-stream.jsonl"
    err_path = logs_dir / "antigravity-cli.err"
    sup_path = logs_dir / "session_supervisor.log"
    if not any(p.is_file() and p.stat().st_size > 0
               for p in (stream_path, err_path, sup_path, existing_jsonl)):
        return

    now_ts = _iso_utc_now()
    records: list[dict[str, object]] = [{
        "step_index": 0,
        "type": "USER_INPUT",
        "source": "USER_EXPLICIT",
        "status": "DONE",
        "created_at": now_ts,
        "content": _trial_instruction(trial),
    }]

    stream_errors: list[str] = []
    if existing_jsonl.is_file() and existing_jsonl.stat().st_size > 0:
        for raw_line in existing_jsonl.read_text(
                encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict) and str(rec.get("type")
                                             or "").upper() == "SYSTEM_ERROR":
                err_val = rec.get("error") or rec.get("content")
                if isinstance(err_val, str) and err_val.strip():
                    stream_errors.append(err_val.strip())

    if stream_path.is_file():
        for raw_line in stream_path.read_text(encoding="utf-8",
                                              errors="replace").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if not isinstance(ev, dict):
                continue
            ev_type = ev.get("event")
            if ev_type == "step_update" and isinstance(ev.get("step_update"),
                                                       dict):
                su = ev["step_update"]
                step_type = str(su.get("step_type") or "").lower()
                text_delta = su.get("text_delta")
                tool_name = su.get("tool_name")
                if step_type == "agent_response" or tool_name:
                    tool_calls = None
                    if tool_name:
                        tool_info = su.get("tool_info")
                        args = tool_info if isinstance(tool_info, dict) else {}
                        tool_calls = [{"name": str(tool_name), "args": args}]
                    records.append({
                        "step_index": len(records),
                        "type": "PLANNER_RESPONSE",
                        "source": "MODEL",
                        "status": str(su.get("state") or "DONE"),
                        "created_at": now_ts,
                        "content": str(text_delta or ""),
                        "tool_calls": tool_calls,
                    })
            elif ev_type == "result" and isinstance(ev.get("result"), dict):
                res = ev["result"]
                for key in ("error", "response"):
                    val = res.get(key)
                    if isinstance(val, str) and val.strip():
                        stream_errors.append(val.strip())

    err_parts = list(stream_errors)
    for p in (err_path, sup_path):
        if p.is_file() and p.stat().st_size > 0:
            txt = p.read_text(encoding="utf-8", errors="replace").strip()
            if txt:
                err_parts.append(txt[-_FALLBACK_TAIL_CHARS:])

    if err_parts or len(records) == 1:
        err_msg = "\n".join(err_parts)[-_FALLBACK_TAIL_CHARS:] or (
            "Antigravity CLI exited before writing a native transcript.")
        records.append({
            "step_index": len(records),
            "type": "SYSTEM_ERROR",
            "source": "SYSTEM",
            "status": "ERROR",
            "created_at": now_ts,
            "content": f"[Early CLI Crash / Error Output]\n{err_msg}",
            "error": err_msg,
        })

    existing_jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n",
                              encoding="utf-8")


def _claude_file_has_message_step(jsonl_path: Path) -> bool:
    if not jsonl_path.is_file() or jsonl_path.stat().st_size == 0:
        return False
    try:
        for raw_line in jsonl_path.read_text(encoding="utf-8",
                                             errors="replace").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if (isinstance(ev, dict)
                    and ev.get("type") in ("user", "assistant")
                    and isinstance(ev.get("message"), dict)):
                return True
    except OSError:
        return False
    return False


def _claude_file_has_user_step(jsonl_path: Path) -> bool:
    if not jsonl_path.is_file() or jsonl_path.stat().st_size == 0:
        return False
    try:
        for raw_line in jsonl_path.read_text(encoding="utf-8",
                                             errors="replace").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if not (isinstance(ev, dict) and ev.get("type") == "user"
                    and isinstance(ev.get("message"), dict)):
                continue
            content = ev["message"].get("content")
            if isinstance(content, str) and content.strip():
                return True
            if isinstance(content, list) and any(not (
                    isinstance(b, dict) and b.get("type") == "tool_result")
                                                 for b in content):
                return True
    except OSError:
        return False
    return False


def _ensure_claude_user_step(trial: object, jsonl_path: Path) -> None:
    if _claude_file_has_user_step(jsonl_path):
        return
    try:
        raw_lines = [
            ln for ln in jsonl_path.read_text(encoding="utf-8",
                                              errors="replace").splitlines()
            if ln.strip()
        ]
    except OSError:
        return

    parsed_events: list[dict[str, object]] = []
    for ln in raw_lines:
        try:
            ev = json.loads(ln)
            if isinstance(ev, dict):
                parsed_events.append(ev)
        except Exception:
            continue

    min_ts = min(
        (str(ev.get("timestamp") or "")
         for ev in parsed_events if isinstance(ev.get("message"), dict)),
        default="",
    )
    user_ev: dict[str, object] = {
        "type": "user",
        "sessionId": jsonl_path.stem,
        "message": {
            "role": "user",
            "content": _trial_instruction(trial),
        },
    }
    if min_ts:
        user_ev["timestamp"] = min_ts

    jsonl_path.write_text(
        "\n".join([json.dumps(user_ev), *raw_lines]) + "\n",
        encoding="utf-8",
    )


def _stage_claude_fallback_transcript(trial: object, logs_dir: Path) -> None:
    projects_root = logs_dir / "sessions" / "projects"
    if projects_root.is_dir():
        for jsonl_file in list(projects_root.rglob("*.jsonl")):
            try:
                if jsonl_file.is_file() and jsonl_file.stat().st_size == 0:
                    jsonl_file.unlink()
            except OSError:
                pass

    root_files = ([
        f for f in projects_root.rglob("*.jsonl")
        if "subagents" not in f.parent.parts
    ] if projects_root.is_dir() else [])
    if len(root_files) > 1 and any(
            _claude_file_has_user_step(f) for f in root_files):
        kept_roots: list[Path] = []
        for f in root_files:
            if _claude_file_has_user_step(f):
                kept_roots.append(f)
            else:
                try:
                    f.unlink()
                except OSError:
                    kept_roots.append(f)
        root_files = kept_roots
    subagent_files = ([
        f for f in projects_root.rglob("subagents/*.jsonl")
        if _claude_file_has_message_step(f)
    ] if projects_root.is_dir() else [])

    for f in root_files:
        if _claude_file_has_message_step(f):
            _ensure_claude_user_step(trial, f)
            return

    raw_txt_path = logs_dir / "claude-code.txt"
    sup_path = logs_dir / "session_supervisor.log"
    has_raw_txt = raw_txt_path.is_file() and raw_txt_path.stat().st_size > 0
    if not has_raw_txt and not subagent_files:
        return

    # Choose the single project directory Harbor's ClaudeCode._get_session_dir expects.
    proj_dir: Path | None = None
    if root_files:
        proj_dir = root_files[0].parent
    elif subagent_files:
        # `<projects_root>/<slug>/<session_id>/subagents/agent-xxx.jsonl`
        candidate = subagent_files[0].parent.parent
        proj_dir = (candidate.parent
                    if candidate.parent != projects_root else candidate)
    elif projects_root.is_dir():
        existing_dirs = sorted(d for d in projects_root.iterdir()
                               if d.is_dir())
        if len(existing_dirs) == 1:
            proj_dir = existing_dirs[0]
    if proj_dir is None:
        proj_dir = projects_root / "-workspace"
    proj_dir.mkdir(parents=True, exist_ok=True)

    target_jsonl = (root_files[0] if root_files else proj_dir /
                    "fallback-session.jsonl")
    extracted_lines: list[str] = []
    plain_errors: list[str] = []
    if has_raw_txt:
        for raw_line in raw_txt_path.read_text(encoding="utf-8",
                                               errors="replace").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                ev = json.loads(stripped)
            except Exception:
                plain_errors.append(stripped)
                continue
            if isinstance(ev, dict) and ev.get("type") is not None:
                extracted_lines.append(json.dumps(ev))
                for key in ("result", "error", "message"):
                    val = ev.get(key)
                    if isinstance(val, str) and val.strip():
                        plain_errors.append(val.strip())

    if extracted_lines and not target_jsonl.exists():
        target_jsonl.write_text("\n".join(extracted_lines) + "\n",
                                encoding="utf-8")

    if not _claude_file_has_message_step(target_jsonl):
        if sup_path.is_file() and sup_path.stat().st_size > 0:
            plain_errors.append(
                sup_path.read_text(
                    encoding="utf-8",
                    errors="replace").strip()[-_FALLBACK_TAIL_CHARS:])
        err_body = "\n".join(plain_errors)[-_FALLBACK_TAIL_CHARS:] or (
            "Claude Code exited before writing any user/assistant session steps."
        )
        now_ts = _iso_utc_now()
        user_ev = {
            "type": "user",
            "sessionId": target_jsonl.stem,
            "timestamp": now_ts,
            "message": {
                "role": "user",
                "content": _trial_instruction(trial),
            },
        }
        asst_ev = {
            "type": "assistant",
            "sessionId": target_jsonl.stem,
            "timestamp": now_ts,
            "message": {
                "role":
                "assistant",
                "content": [{
                    "type":
                    "text",
                    "text":
                    f"[Early CLI Crash / Error Output]\n{err_body}",
                }],
            },
        }
        with target_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(user_ev) + "\n")
            handle.write(json.dumps(asst_ev) + "\n")
    else:
        _ensure_claude_user_step(trial, target_jsonl)


def _stage_native_fallback_inputs(
    trial: object,
    logs_dir: Path,
    agent_obj: object | None = None,
) -> None:
    running_agent = (agent_obj or getattr(trial, "user_agent", None)
                     or getattr(trial, "agent", None))
    agent_name = ""
    if running_agent is not None and callable(
            getattr(running_agent, "name", None)):
        try:
            agent_name = str(running_agent.name() or "")
        except Exception:
            agent_name = ""
    cls_name = type(running_agent).__name__.lower() if running_agent else ""

    if "antigravity" in agent_name or "antigravity" in cls_name or (
            logs_dir / "antigravity-stream.jsonl").exists() or (
                logs_dir / "antigravity-cli.err").exists():
        _stage_agy_fallback_transcript(trial, logs_dir)
    if "claude" in agent_name or "claude" in cls_name or (
            logs_dir / "claude-code.txt").exists() or (logs_dir / "sessions" /
                                                       "projects").exists():
        _stage_claude_fallback_transcript(trial, logs_dir)


def _trial_has_exception_source(trial: object) -> bool:
    result = getattr(trial, "result", None)
    if getattr(result, "exception_info", None) is not None:
        return True
    if sys.exc_info()[1] is not None:
        return True
    paths = getattr(trial, "paths", None)
    exc_path = getattr(paths, "exception_message_path", None)
    return (isinstance(exc_path, Path) and exc_path.is_file()
            and exc_path.stat().st_size > 0)


def _should_write_fallback_trajectory(traj_path: Path, trial: object) -> bool:
    if not traj_path.is_file() or traj_path.stat().st_size == 0:
        return True
    if not _trial_has_exception_source(trial):
        return False
    try:
        existing = json.loads(
            traj_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return False
    if not isinstance(existing, dict):
        return False
    steps = existing.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    last_step = steps[-1]
    if not isinstance(last_step, dict):
        return False
    extra = last_step.get("extra")
    return (isinstance(extra, dict)
            and extra.get("fallback_trajectory") is True
            and extra.get("exception_type") is None)


def _write_fallback_atif_trajectory(
    trial: object,
    logs_dir: Path,
    agent_obj: object | None = None,
) -> None:
    traj_path = logs_dir / "trajectory.json"
    if not _should_write_fallback_trajectory(traj_path, trial):
        return

    running_agent = (agent_obj or getattr(trial, "user_agent", None)
                     or getattr(trial, "agent", None))
    agent_name = "unknown"
    agent_version = "unknown"
    model_name = getattr(running_agent, "model_name", None)
    session_id = str(
        getattr(running_agent, "session_id", None)
        or getattr(getattr(trial, "result", None), "id", None)
        or "fallback-session")
    if running_agent is not None:
        if callable(getattr(running_agent, "name", None)):
            try:
                agent_name = str(running_agent.name() or "unknown")
            except Exception:
                pass
        if callable(getattr(running_agent, "version", None)):
            try:
                agent_version = str(running_agent.version() or "unknown")
            except Exception:
                pass

    diag_parts: list[str] = []
    result = getattr(trial, "result", None)
    exc_info = getattr(result, "exception_info", None)
    active_exc = sys.exc_info()[1] if exc_info is None else None
    exc_type_name: str | None = None
    if exc_info is not None:
        exc_type_name = getattr(exc_info, "exception_type",
                                None) or "Exception"
        exc_msg = getattr(exc_info, "exception_message", None) or ""
        diag_parts.append(f"{exc_type_name}: {exc_msg}".strip())
    elif active_exc is not None:
        exc_type_name = type(active_exc).__name__
        exc_msg = str(active_exc)
        diag_parts.append(f"{exc_type_name}: {exc_msg}".strip())

    paths = getattr(trial, "paths", None)
    exc_path = getattr(paths, "exception_message_path", None)
    if (not diag_parts and isinstance(exc_path, Path) and exc_path.is_file()
            and exc_path.stat().st_size > 0):
        diag_parts.append(
            exc_path.read_text(
                encoding="utf-8",
                errors="replace").strip()[-_FALLBACK_TAIL_CHARS:])

    candidate_logs = [
        logs_dir / "antigravity-cli.err",
        logs_dir / "antigravity-stream.jsonl",
        logs_dir / "claude-code.txt",
        logs_dir / "session_supervisor.log",
        logs_dir / "setup" / "stderr.txt",
        logs_dir / "setup" / "stdout.txt",
        *(sorted(logs_dir.glob("command-*/stderr.txt"))
          if logs_dir.is_dir() else []),
        *(sorted(logs_dir.glob("command-*/stdout.txt"))
          if logs_dir.is_dir() else []),
        logs_dir.parent / "setup" / "stderr.txt",
        logs_dir.parent / "setup" / "stdout.txt",
    ]
    for p in candidate_logs:
        if p.is_file() and p.stat().st_size > 0:
            snippet = p.read_text(encoding="utf-8", errors="replace").strip()
            if snippet:
                label = (f"{p.parent.name}/{p.name}"
                         if p.parent != logs_dir else p.name)
                diag_parts.append(
                    f"[{label}]\n{snippet[-_FALLBACK_TAIL_CHARS:]}")

    diag_message = "\n\n".join(diag_parts)[-_FALLBACK_TAIL_CHARS:] or (
        "Trial terminated before the agent produced a native trajectory.")
    now_ts = _iso_utc_now()

    payload = {
        "schema_version":
        "ATIF-v1.7",
        "session_id":
        session_id,
        "agent": {
            "name": agent_name,
            "version": agent_version,
            **({
                "model_name": model_name
            } if model_name else {}),
        },
        "steps": [
            {
                "step_id": 1,
                "timestamp": now_ts,
                "source": "user",
                "message": _trial_instruction(trial),
            },
            {
                "step_id": 2,
                "timestamp": now_ts,
                "source": "system",
                "message": f"[Fallback Diagnostic Trajectory]\n{diag_message}",
                "extra": {
                    "fallback_trajectory": True,
                    "exception_type": exc_type_name,
                },
            },
        ],
        "final_metrics": {
            "total_steps": 2,
        },
    }

    try:
        from harbor.models.trajectories import Trajectory
        validated = Trajectory.model_validate(payload)
        payload = validated.to_json_dict()
    except Exception:
        pass

    logs_dir.mkdir(parents=True, exist_ok=True)
    traj_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                         encoding="utf-8")


def _patch_trial_fallback_trajectory(trial_cls: object) -> None:
    if trial_cls is None or getattr(
            trial_cls, "_eval_fallback_trajectory_patched", False):
        return
    import logging as _logging

    orig_sync = getattr(trial_cls, "_sync_agent_output", None)
    orig_populate = getattr(trial_cls, "_populate_agent_context", None)
    if not callable(orig_sync) or not callable(orig_populate):
        return

    @functools.wraps(orig_sync)
    async def _fallback_sync_agent_output(self, target) -> None:
        if target is not None and getattr(target, "agent_result",
                                          None) is None:
            try:
                from harbor.models.agent.context import AgentContext
                target.agent_result = AgentContext()
            except Exception:
                pass
        await orig_sync(self, target)

    def _populate_role_trajectory(
        trial: object,
        agent_obj: object | None,
        logs_path: Path | None,
        populate_fn: object,
        logger: object,
    ) -> None:
        if logs_path is not None:
            traj_path = logs_path / "trajectory.json"
            if not traj_path.is_file() or traj_path.stat().st_size == 0:
                try:
                    logs_path.mkdir(parents=True, exist_ok=True)
                    _stage_native_fallback_inputs(trial,
                                                  logs_path,
                                                  agent_obj=agent_obj)
                except Exception as e:
                    logger.warning(
                        f"[Trial-Trajectory-Adapter] Staging fallback inputs"
                        f" failed: {e}")

        populate_fn()

        if logs_path is not None:
            traj_path = logs_path / "trajectory.json"
            if _should_write_fallback_trajectory(traj_path, trial):
                try:
                    _write_fallback_atif_trajectory(trial,
                                                    logs_path,
                                                    agent_obj=agent_obj)
                    logger.info(
                        f"[Trial-Trajectory-Adapter] Synthesized fallback"
                        f" trajectory at {traj_path}")
                except Exception as e:
                    logger.warning(
                        f"[Trial-Trajectory-Adapter] Failed to write fallback"
                        f" trajectory at {traj_path}: {e}")

    @functools.wraps(orig_populate)
    def _fallback_populate_agent_context(self, agent_result) -> None:
        logger = getattr(self, "logger", None) or _logging.getLogger(__name__)
        user_agent = getattr(self, "user_agent", None)
        coding_agent = getattr(self, "agent", None)
        running_agent = user_agent or coding_agent
        logs_dir = (getattr(running_agent, "logs_dir", None) or getattr(
            getattr(self, "paths", None), "agent_dir", None))
        logs_path = Path(logs_dir) if logs_dir is not None else None

        _populate_role_trajectory(
            self,
            running_agent,
            logs_path,
            lambda: orig_populate(self, agent_result),
            logger,
        )

        # In bridged trials Harbor's Trial._populate_agent_context only calls
        # populate_context_post_run on user_agent, leaving agent/trajectory.json
        # unwritten even though Uploader archives agent/trajectory.json.
        if user_agent is not None and coding_agent is not None:
            coding_dir = (getattr(coding_agent, "logs_dir", None) or getattr(
                getattr(self, "paths", None), "agent_dir", None))
            coding_path = Path(coding_dir) if coding_dir is not None else None
            if coding_path is not None and coding_path != logs_path:

                def _populate_coding() -> None:
                    pop = getattr(coding_agent, "populate_context_post_run",
                                  None)
                    if not callable(pop):
                        return
                    try:
                        from harbor.models.agent.context import AgentContext
                        ctx = AgentContext()
                    except Exception:
                        ctx = type("_DummyContext", (), {})()
                    try:
                        pop(ctx)
                    except Exception as e:
                        logger.warning(
                            f"[Trial-Trajectory-Adapter] Coding agent"
                            f" populate_context_post_run failed: {e}")

                _populate_role_trajectory(
                    self,
                    coding_agent,
                    coding_path,
                    _populate_coding,
                    logger,
                )

    trial_cls._sync_agent_output = _fallback_sync_agent_output
    trial_cls._populate_agent_context = _fallback_populate_agent_context
    trial_cls._eval_fallback_trajectory_patched = True


_patch_trial_fallback_trajectory(_HarborTrial)
