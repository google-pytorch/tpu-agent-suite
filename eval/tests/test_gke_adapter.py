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
"""Unit tests for the GKE download adapter in eval/adapter/sitecustomize.py."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from harbor.environments.base import ExecResult
from harbor.environments.gke import GKEEnvironment
from harbor.environments.tar_transfer import extract_dir_from_bytes

from eval.adapter import sitecustomize as _ADAPTER

_SUCCESS = b"status: Success\n"


class _FakeExecStream:
    """A binary-mode kubernetes WSClient replaying fixed frames."""

    def __init__(
        self,
        frames: list[bytes],
        error_channel: bytes | None,
        *,
        fail_write_after: int | None = None,
        hang_on_drain: bool = False,
    ):
        self._frames = list(frames)
        self._error_channel = error_channel
        self._stdout = b""
        self._fail_write_after = fail_write_after
        self._hang_on_drain = hang_on_drain
        self.written_chunks: list[bytes] = []
        self.stdin_closed = False
        self.closed = False

    def is_open(self) -> bool:
        if self._hang_on_drain and not self.closed:
            return True
        return bool(self._frames) or bool(self._stdout)

    def update(self, timeout=0) -> None:
        if self._frames:
            self._stdout += self._frames.pop(0)

    def peek_stdout(self) -> bool:
        return bool(self._stdout)

    def read_stdout(self) -> bytes:
        data, self._stdout = self._stdout, b""
        return data

    def peek_stderr(self) -> bool:
        return False

    def read_stderr(self) -> bytes:
        return b""

    def write_stdin(self, data: bytes) -> None:
        if (self._fail_write_after is not None
                and len(self.written_chunks) >= self._fail_write_after):
            raise RuntimeError("socket is already closed.")
        self.written_chunks.append(data)

    def close_channel(self, channel: int) -> None:
        if channel == 0:
            self.stdin_closed = True

    @property
    def returncode(self) -> int | None:
        if self._error_channel is None:
            return None
        if self._error_channel == _SUCCESS:
            return 0
        if self._error_channel == b"exit1":
            return 1
        return 2

    def close(self) -> None:
        self.closed = True


def _make_env() -> GKEEnvironment:
    env = GKEEnvironment.__new__(GKEEnvironment)
    env._compose_mode = False
    env.pod_name = "pod"
    env.namespace = "default"
    env._core_api = MagicMock()
    env.logger = logging.getLogger("test_gke_adapter")

    async def _ensure_client():
        return None

    async def _wait_for_container_exec_ready():
        return None

    async def _exec(command: str, **kwargs):
        return ExecResult(stdout="", stderr="", return_code=0)

    env._ensure_client = _ensure_client
    env._wait_for_container_exec_ready = _wait_for_container_exec_ready
    env.exec = _exec
    return env


def _agent_dir_tar() -> tuple[bytes, int]:
    """A tar of an agent log dir, and the offset of a "█" inside it."""
    buf = io.BytesIO()
    log = ("Fetching 7 files: 100%|██████████| 7/7\n" * 50).encode()
    members = {
        "claude-code.txt": log,
        "session_supervisor.log": b"[ClaudeSupervisor] done\n",
        "sessions/projects/-workspace/s.jsonl": b'{"type":"user"}\n',
    }
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    data = buf.getvalue()
    return data, data.index("█".encode())


class TestGkeDownloadAdapter(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_multibyte_char_split_across_frames_is_kept(self) -> None:
        # The CI failure: "█" split over two frames. Decoded per frame as
        # text, it shifted the tar stream and every member after
        # claude-code.txt was silently dropped.
        data, at = _agent_dir_tar()
        frames = [data[:at + 1], data[at + 1:]]
        env = _make_env()
        with patch.object(_ADAPTER,
                          "stream",
                          return_value=_FakeExecStream(frames, _SUCCESS)):
            asyncio.run(env.download_dir("/logs/agent", self.temp_dir))

        self.assertTrue((self.temp_dir / "session_supervisor.log").is_file())
        self.assertTrue(
            (self.temp_dir / "sessions/projects/-workspace/s.jsonl").is_file())
        log = (self.temp_dir / "claude-code.txt").read_text(encoding="utf-8")
        self.assertNotIn("�", log)

    def test_stream_closed_before_exit_status_is_retried(self) -> None:
        data, _ = _agent_dir_tar()
        truncated = _FakeExecStream([data[:len(data) // 2]], None)
        complete = _FakeExecStream([data], _SUCCESS)
        env = _make_env()
        with patch.object(_ADAPTER, "stream",
                          side_effect=[truncated, complete]) as mock_stream, \
                patch.object(_ADAPTER.asyncio, "sleep") as mock_sleep:
            mock_sleep.return_value = None
            asyncio.run(env.download_dir("/logs/agent", self.temp_dir))

        self.assertEqual(mock_stream.call_count, 2)
        self.assertTrue(mock_stream.call_args.kwargs["binary"])
        self.assertTrue(
            (self.temp_dir / "sessions/projects/-workspace/s.jsonl").is_file())

    def test_stream_never_completing_raises(self) -> None:
        data, _ = _agent_dir_tar()
        env = _make_env()
        streams = [
            _FakeExecStream([data[:1024]], None)
            for _ in range(_ADAPTER._DOWNLOAD_ATTEMPTS)
        ]
        with patch.object(_ADAPTER, "stream", side_effect=streams), \
                patch.object(_ADAPTER.asyncio, "sleep") as mock_sleep:
            mock_sleep.return_value = None
            with self.assertRaises(Exception):
                asyncio.run(env.download_dir("/logs/agent", self.temp_dir))
        self.assertFalse((self.temp_dir / "claude-code.txt").exists())

    def test_tar_failure_is_not_retried(self) -> None:
        env = _make_env()
        with patch.object(_ADAPTER,
                          "stream",
                          return_value=_FakeExecStream([], b"failure")) as m:
            with self.assertRaises(RuntimeError):
                asyncio.run(env.download_dir("/missing", self.temp_dir))
        self.assertEqual(m.call_count, 1)

    def test_tar_exit_1_with_nonempty_stdout_is_kept(self) -> None:
        # GNU tar exits 1 when a live log or SQLite WAL file is appended to
        # while reading; the archive on stdout must still be extracted.
        data, _ = _agent_dir_tar()
        env = _make_env()
        with patch.object(_ADAPTER,
                          "stream",
                          return_value=_FakeExecStream(
                              [data], b"exit1")) as mock_stream:
            asyncio.run(env.download_dir("/logs/agent", self.temp_dir))
        self.assertEqual(mock_stream.call_count, 1)
        self.assertTrue((self.temp_dir / "session_supervisor.log").is_file())
        self.assertTrue(
            (self.temp_dir / "sessions/projects/-workspace/s.jsonl").is_file())

    def test_tar_exit_1_with_truncated_stdout_is_retried(self) -> None:
        data, _ = _agent_dir_tar()
        truncated = _FakeExecStream([data[:len(data) // 2]], b"exit1")
        complete = _FakeExecStream([data], _SUCCESS)
        env = _make_env()
        with patch.object(_ADAPTER, "stream",
                          side_effect=[truncated, complete]) as mock_stream, \
                patch.object(_ADAPTER.asyncio, "sleep") as mock_sleep:
            mock_sleep.return_value = None
            asyncio.run(env.download_dir("/logs/agent", self.temp_dir))
        self.assertEqual(mock_stream.call_count, 2)
        self.assertTrue(
            (self.temp_dir / "sessions/projects/-workspace/s.jsonl").is_file())

    def test_truncated_tar_reported_as_success_is_retried(self) -> None:
        # When konnectivity-agent/apiserver drops an exec stream mid-transfer,
        # the apiserver can report status: Success on ERROR_CHANNEL even though
        # stdout was cut mid-member.
        data, _ = _agent_dir_tar()
        truncated = _FakeExecStream([data[:len(data) // 2]], _SUCCESS)
        complete = _FakeExecStream([data], _SUCCESS)
        env = _make_env()
        with patch.object(_ADAPTER, "stream",
                          side_effect=[truncated, complete]) as mock_stream, \
                patch.object(_ADAPTER.asyncio, "sleep") as mock_sleep:
            mock_sleep.return_value = None
            asyncio.run(env.download_dir("/logs/agent", self.temp_dir))
        self.assertEqual(mock_stream.call_count, 2)
        self.assertTrue(
            (self.temp_dir / "sessions/projects/-workspace/s.jsonl").is_file())

    def test_tar_cut_on_member_boundary_without_eof_blocks_is_retried(
            self) -> None:
        data, _ = _agent_dir_tar()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
            first = tar.getmembers()[0]
        first_end = first.offset_data + ((first.size + 511) // 512) * 512
        cut_between = _FakeExecStream([data[:first_end]], _SUCCESS)
        complete = _FakeExecStream([data], _SUCCESS)
        env = _make_env()
        with patch.object(_ADAPTER,
                          "stream",
                          side_effect=[cut_between, complete]) as mock_stream, \
                patch.object(_ADAPTER.asyncio, "sleep") as mock_sleep:
            mock_sleep.return_value = None
            asyncio.run(env.download_dir("/logs/agent", self.temp_dir))
        self.assertEqual(mock_stream.call_count, 2)
        self.assertTrue(
            (self.temp_dir / "sessions/projects/-workspace/s.jsonl").is_file())

    def test_download_dir_quotes_source_dir(self) -> None:
        data, _ = _agent_dir_tar()
        env = _make_env()
        with patch.object(_ADAPTER,
                          "stream",
                          return_value=_FakeExecStream(
                              [data], _SUCCESS)) as mock_stream:
            asyncio.run(env.download_dir("/logs/agent dir", self.temp_dir))
        cmd = mock_stream.call_args.kwargs["command"]
        self.assertIn("'/logs/agent dir'", cmd[2])

    def test_download_file_reads_bytes(self) -> None:
        payload = "progress █ done\n".encode()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo("logs/agent/trajectory.json")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        data = buf.getvalue()
        at = data.index("█".encode())
        env = _make_env()
        target = self.temp_dir / "out" / "trajectory.json"
        with patch.object(_ADAPTER,
                          "stream",
                          return_value=_FakeExecStream(
                              [data[:at + 1], data[at + 1:]], _SUCCESS)):
            asyncio.run(
                env.download_file("/logs/agent/trajectory.json", target))
        self.assertEqual(target.read_bytes(), payload)

    def test_upload_dir_writes_in_chunks_and_closes_stdin(self) -> None:
        src_dir = self.temp_dir / "src"
        src_dir.mkdir()
        payload = os.urandom(_ADAPTER._UPLOAD_CHUNK_BYTES * 3 + 1234)
        (src_dir / "antigravity-conversation.db-wal").write_bytes(payload)
        (src_dir / "trajectory.json").write_text('{"steps":[]}',
                                                 encoding="utf-8")

        fake_stream = _FakeExecStream([], _SUCCESS)
        env = _make_env()
        with patch.object(_ADAPTER, "stream",
                          return_value=fake_stream) as mock_stream:
            asyncio.run(env.upload_dir(src_dir, "/logs/agent"))

        self.assertTrue(mock_stream.call_args.kwargs["binary"])
        self.assertGreater(len(fake_stream.written_chunks), 3)
        self.assertTrue(
            all(
                len(c) <= _ADAPTER._UPLOAD_CHUNK_BYTES
                for c in fake_stream.written_chunks))
        self.assertTrue(fake_stream.stdin_closed)

        extracted = self.temp_dir / "extracted"
        extracted.mkdir()
        extract_dir_from_bytes(b"".join(fake_stream.written_chunks), extracted)
        self.assertEqual(
            (extracted / "antigravity-conversation.db-wal").read_bytes(),
            payload,
        )
        self.assertEqual(
            (extracted / "trajectory.json").read_text(encoding="utf-8"),
            '{"steps":[]}',
        )

    def test_upload_dir_retries_on_closed_socket(self) -> None:
        src_dir = self.temp_dir / "src"
        src_dir.mkdir()
        (src_dir / "file.bin").write_bytes(b"x" *
                                           (_ADAPTER._UPLOAD_CHUNK_BYTES * 2))

        dropped = _FakeExecStream([], _SUCCESS, fail_write_after=1)
        complete = _FakeExecStream([], _SUCCESS)
        env = _make_env()
        with patch.object(_ADAPTER, "stream",
                          side_effect=[dropped, complete]) as mock_stream, \
                patch.object(_ADAPTER.asyncio, "sleep") as mock_sleep:
            mock_sleep.return_value = None
            asyncio.run(env.upload_dir(src_dir, "/logs/agent"))

        self.assertEqual(mock_stream.call_count, 2)
        self.assertGreater(len(complete.written_chunks), 1)

    def test_upload_dir_retries_when_readiness_check_fails(self) -> None:
        src_dir = self.temp_dir / "src"
        src_dir.mkdir()
        (src_dir / "trajectory.json").write_text('{"steps":[]}',
                                                 encoding="utf-8")

        complete = _FakeExecStream([], _SUCCESS)
        env = _make_env()
        attempts = 0

        async def _flaky_ready():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError(51, "Network is unreachable")

        env._wait_for_container_exec_ready = _flaky_ready
        with patch.object(_ADAPTER, "stream",
                          return_value=complete) as mock_stream, \
                patch.object(_ADAPTER.asyncio, "sleep") as mock_sleep:
            mock_sleep.return_value = None
            asyncio.run(env.upload_dir(src_dir, "/logs/agent"))

        self.assertEqual(attempts, 2)
        self.assertEqual(mock_stream.call_count, 1)

    def test_upload_dir_tar_failure_is_not_retried(self) -> None:
        src_dir = self.temp_dir / "src"
        src_dir.mkdir()
        (src_dir / "file.txt").write_text("hello", encoding="utf-8")

        env = _make_env()
        with patch.object(_ADAPTER,
                          "stream",
                          return_value=_FakeExecStream([], b"failure")) as m:
            with self.assertRaises(RuntimeError):
                asyncio.run(env.upload_dir(src_dir, "/logs/agent"))
        self.assertEqual(m.call_count, 1)

    def test_upload_file_writes_in_chunks(self) -> None:
        src_file = self.temp_dir / "big.bin"
        payload = b"a" * (_ADAPTER._UPLOAD_CHUNK_BYTES * 2 + 500)
        src_file.write_bytes(payload)

        fake_stream = _FakeExecStream([], _SUCCESS)
        env = _make_env()
        with patch.object(_ADAPTER, "stream", return_value=fake_stream):
            asyncio.run(env.upload_file(src_file, "/logs/agent/big.bin"))

        self.assertGreater(len(fake_stream.written_chunks), 2)
        self.assertTrue(
            all(
                len(c) <= _ADAPTER._UPLOAD_CHUNK_BYTES
                for c in fake_stream.written_chunks))
        extracted = self.temp_dir / "extracted_file"
        extracted.mkdir()
        extract_dir_from_bytes(b"".join(fake_stream.written_chunks), extracted)
        self.assertEqual((extracted / "big.bin").read_bytes(), payload)

    def test_upload_dir_quotes_target_dir_with_spaces(self) -> None:
        src_dir = self.temp_dir / "src"
        src_dir.mkdir()
        (src_dir / "file.txt").write_text("hello", encoding="utf-8")

        fake_stream = _FakeExecStream([], _SUCCESS)
        env = _make_env()
        exec_commands: list[str] = []

        async def _recording_exec(command: str, **kwargs):
            exec_commands.append(command)
            return ExecResult(stdout="", stderr="", return_code=0)

        env.exec = _recording_exec
        with patch.object(_ADAPTER, "stream", return_value=fake_stream):
            asyncio.run(env.upload_dir(src_dir, "/logs/agent dir"))

        self.assertEqual(exec_commands, ["mkdir -p '/logs/agent dir'"])

    def test_upload_dir_drain_deadline_raises_timeout(self) -> None:
        src_dir = self.temp_dir / "src"
        src_dir.mkdir()
        (src_dir / "file.txt").write_text("hello", encoding="utf-8")

        streams = [
            _FakeExecStream([], _SUCCESS, hang_on_drain=True)
            for _ in range(_ADAPTER._UPLOAD_ATTEMPTS)
        ]
        env = _make_env()
        with patch.object(_ADAPTER, "stream", side_effect=streams), \
                patch.object(_ADAPTER, "_UPLOAD_DRAIN_TIMEOUT_SEC", 0.0), \
                patch.object(_ADAPTER.asyncio, "sleep") as mock_sleep:
            mock_sleep.return_value = None
            with self.assertRaises(TimeoutError):
                asyncio.run(env.upload_dir(src_dir, "/logs/agent"))
        self.assertTrue(all(s.closed for s in streams))


_real_sleep = asyncio.sleep


async def _no_backoff(*_) -> None:
    await _real_sleep(0)


class _LocalPodExec:
    """Stands in for the unpatched GKEEnvironment.exec: runs the script in a
    local bash, the way the pod would, except for the calls listed in `cut`,
    which return what a stream cut by a drained konnectivity-agent returns."""

    def __init__(self, cut: dict[int, object] | None = None):
        self.cut = cut or {}
        self.scripts: list[str] = []

    async def __call__(self,
                       environment,
                       command: str,
                       cwd=None,
                       user=None,
                       **kwargs) -> ExecResult:
        call = len(self.scripts)
        self.scripts.append(command)
        if call in self.cut:
            outcome = self.cut[call]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        return ExecResult(stdout=stdout.decode(),
                          stderr=stderr.decode(),
                          return_code=proc.returncode)


class TestGkeDurableExecAdapter(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        patcher = patch.object(_ADAPTER, "_DURABLE_EXEC_DIR",
                               str(self.temp_dir / "exec"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _exec(self, fake: _LocalPodExec, command: str, **kwargs) -> ExecResult:
        env = _make_env()
        with patch.object(_ADAPTER, "_orig_gke_exec", fake):
            return asyncio.run(GKEEnvironment.exec(env, command, **kwargs))

    def test_output_and_status_pass_through(self) -> None:
        result = self._exec(_LocalPodExec(), "echo out; echo err >&2; exit 3")
        self.assertEqual(result.stdout, "out\n")
        self.assertEqual(result.stderr, "err\n")
        self.assertEqual(result.return_code, 3)
        self.assertEqual(list((self.temp_dir / "exec").iterdir()), [])

    def test_cut_stream_reported_as_success_is_reattached(self) -> None:
        # The CI failure: the agent's exec closed after 18 minutes with
        # status Success and no output, while the agent was mid tool call.
        fake = _LocalPodExec(
            cut={1: ExecResult(stdout="", stderr="", return_code=0)})
        with patch.object(_ADAPTER.asyncio, "sleep", _no_backoff):
            result = self._exec(fake, "sleep 1; echo finished")
        self.assertEqual(result.stdout, "finished\n")
        self.assertEqual(result.return_code, 0)
        self.assertGreaterEqual(len(fake.scripts), 3)

    def test_closed_stream_error_is_reattached(self) -> None:
        fake = _LocalPodExec(cut={1: _ADAPTER.GKEExecStreamClosedError("eof")})
        with patch.object(_ADAPTER.asyncio, "sleep", _no_backoff):
            result = self._exec(fake, "echo finished")
        self.assertEqual(result.stdout, "finished\n")
        self.assertEqual(result.return_code, 0)

    def test_launch_that_never_ran_is_relaunched(self) -> None:
        fake = _LocalPodExec(
            cut={0: ExecResult(stdout="", stderr="", return_code=0)})
        with patch.object(_ADAPTER.asyncio, "sleep", _no_backoff):
            result = self._exec(fake,
                                "echo once >> " + str(self.temp_dir / "count"))
        self.assertEqual(result.return_code, 0)
        self.assertEqual((self.temp_dir / "count").read_text(), "once\n")

    def test_failed_launch_is_returned(self) -> None:
        refused = ExecResult(stdout=None,
                             stderr="pod not ready",
                             return_code=1)
        fake = _LocalPodExec(cut={0: refused})
        self.assertIs(self._exec(fake, "true"), refused)
        self.assertEqual(len(fake.scripts), 1)

    def test_timeout_stops_the_command(self) -> None:
        marker = self.temp_dir / "marker"
        result = self._exec(_LocalPodExec(),
                            f"sleep 3; touch {marker}",
                            timeout_sec=1)
        self.assertEqual(result.return_code, 124)
        time.sleep(3)
        self.assertFalse(marker.exists())

    def test_timeout_waits_for_sigterm_cleanup_trap(self) -> None:
        # Mirrors the pod command shape (`... | tee ...`) where a child traps
        # SIGTERM to flush partial trajectories after the outer shell dies.
        flushed = self.temp_dir / "flushed"
        script = (f"(trap 'sleep 1; echo done > {flushed}; exit 0' TERM;"
                  " while :; do sleep 0.2; done) | cat")
        result = self._exec(_LocalPodExec(), script, timeout_sec=1)
        self.assertEqual(result.return_code, 124)
        self.assertTrue(flushed.is_file())
        self.assertEqual(flushed.read_text(), "done\n")

    def test_killed_without_status_reports_255(self) -> None:
        # kill -9 of the recording shell: nothing is left to write a status.
        result = self._exec(_LocalPodExec(), "kill -9 $PPID; sleep 5")
        self.assertEqual(result.return_code, 255)
        self.assertIn("without recording a status", result.stderr)

    def test_compose_mode_uses_the_direct_exec(self) -> None:
        env = _make_env()
        env._compose_mode = True
        direct = ExecResult(stdout="x", stderr="", return_code=0)
        fake = _LocalPodExec(cut={0: direct})
        with patch.object(_ADAPTER, "_orig_gke_exec", fake):
            result = asyncio.run(GKEEnvironment.exec(env, "echo x"))
        self.assertIs(result, direct)
        self.assertEqual(fake.scripts, ["echo x"])

    def test_launch_creates_exec_dir_world_writable(self) -> None:
        self._exec(_LocalPodExec(), "true")
        mode = (self.temp_dir / "exec").stat().st_mode & 0o7777
        self.assertEqual(mode, 0o1777)

    def test_agent_user_defaults_home_env(self) -> None:
        env = _make_env()
        env.default_user = "agent"
        captured_envs: list[dict | None] = []
        fake = _LocalPodExec()
        orig_call = fake.__call__

        async def _spy(environment, command: str, **kwargs) -> ExecResult:
            captured_envs.append(kwargs.get("env"))
            return await orig_call(environment, command, **kwargs)

        with patch.object(_ADAPTER, "_orig_gke_exec", _spy):
            res = asyncio.run(
                GKEEnvironment.exec(env, "true", env={"CUSTOM": "1"}))
        self.assertEqual(res.return_code, 0)
        self.assertEqual(
            captured_envs[0],
            {
                "HOME": "/root",
                "USER": "agent",
                "LOGNAME": "agent",
                "CUSTOM": "1",
            },
        )


class TestHostRequirementsInstall(unittest.TestCase):
    """The staged package's host requirements are installed into the pod."""

    POD_PKG = "/root/.tpu-agent-suite/packages/tpu-agent-suite@0.5.0"

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temp_dir)
        self.host_pkg = self.temp_dir / "tpu-agent-suite@0.5.0"
        self.host_pkg.mkdir()
        (self.host_pkg / "requirements-client.txt").write_text(
            "pyyaml>=6.0\n", encoding="utf-8")
        patcher = patch.dict(
            os.environ, {
                "HOST_CENTRAL_PKG": str(self.host_pkg),
                "HOST_CENTRAL_PKG_TARGET": self.POD_PKG,
            })
        patcher.start()
        self.addCleanup(patcher.stop)
        self.mounts = [{"source": str(self.host_pkg), "target": self.POD_PKG}]

    def _run(self, mounts, return_code: int = 0):
        env = _make_env()
        calls: list[tuple[str, dict]] = []

        async def _recording_exec(command: str, **kwargs):
            calls.append((command, kwargs))
            return ExecResult(stdout="",
                              stderr="no matching distribution",
                              return_code=return_code)

        env.exec = _recording_exec
        asyncio.run(_ADAPTER._install_host_requirements(env, mounts))
        return calls

    def test_installs_the_staged_requirements_as_root(self) -> None:
        [(command, kwargs)] = self._run(self.mounts)
        self.assertEqual(
            command, _ADAPTER.DROP_XPROF_NIGHTLY +
            "python3 -m pip install --no-cache-dir --quiet -r "
            f"{self.POD_PKG}/requirements-client.txt")
        self.assertEqual(kwargs["user"], "root")
        self.assertEqual(kwargs["env"], {"PIP_ROOT_USER_ACTION": "ignore"})

    def test_drops_xprof_nightly_before_installing(self) -> None:
        """A pod image built while the package tracked xprof-nightly would
        otherwise keep it beside the release, which tpu-diagnostics refuses."""
        [(command, _)] = self._run(self.mounts)
        drop, install = command.split("; fi; ")
        self.assertIn("pip show -q xprof-nightly", drop)
        self.assertIn("pip uninstall -y -q xprof-nightly xprof", drop)
        self.assertTrue(install.startswith("python3 -m pip install"))

    def test_skipped_when_the_package_is_not_mounted(self) -> None:
        self.assertEqual(self._run([{"source": "/x", "target": "/y"}]), [])

    def test_skipped_when_the_package_ships_no_requirements(self) -> None:
        (self.host_pkg / "requirements-client.txt").unlink()
        self.assertEqual(self._run(self.mounts), [])

    def test_failed_install_is_logged_not_raised(self) -> None:
        with self.assertLogs("test_gke_adapter", level="WARNING") as logs:
            self.assertEqual(len(self._run(self.mounts, return_code=1)), 1)
        self.assertIn("no matching distribution", logs.output[0])


class TestKubeGateAdapter(unittest.TestCase):
    """Unit tests for kube-gate provisioning and non-root agent user default."""

    def test_trial_id_for_env_truncates_and_sanitizes(self) -> None:
        env = _make_env()
        env.pod_name = "harbor-trial-pod-123"
        self.assertEqual(_ADAPTER._trial_id_for_env(env),
                         "harbor-trial-pod-123")

        env.pod_name = "a" * 80
        tid = _ADAPTER._trial_id_for_env(env)
        self.assertLessEqual(len(tid), 63)
        self.assertTrue(tid.startswith("a"))

    def test_setup_kube_gate_uploads_wrapper_and_runs_setup(self) -> None:
        env = _make_env()
        env.pod_name = "trial-xyz"
        uploaded: list[tuple[str, str]] = []
        exec_calls: list[tuple[str, dict]] = []

        async def _upload_file(src, dst):
            uploaded.append((str(src), str(dst)))

        async def _exec(command: str, **kwargs):
            exec_calls.append((command, kwargs))
            return ExecResult(stdout="", stderr="", return_code=0)

        env.upload_file = _upload_file
        env.exec = _exec
        asyncio.run(_ADAPTER._setup_kube_gate(env))

        self.assertEqual(len(uploaded), 1)
        self.assertEqual(uploaded[0][1], "/usr/libexec/kubectl-wrapper")
        self.assertEqual(len(exec_calls), 2)
        self.assertIn("mkdir -p /usr/libexec", exec_calls[0][0])
        script, kwargs = exec_calls[1]
        self.assertEqual(kwargs.get("user"), "root")
        self.assertIn("trial-xyz", script)
        self.assertIn("/etc/kube-gate/trial_id", script)
        self.assertIn("/usr/libexec/kubectl-real", script)
        self.assertIn("NOPASSWD: /usr/libexec/kubectl-wrapper", script)

    def test_patch_trial_agent_user_defaults_unset_agent_user_to_agent(
            self) -> None:

        class _DummyTrial:

            async def _prepare(self) -> None:
                pass

        _ADAPTER._patch_trial_agent_user(_DummyTrial)
        t = _DummyTrial()
        t.task = MagicMock()
        t.task.config.agent.user = None
        asyncio.run(t._prepare())
        self.assertEqual(t.task.config.agent.user, "agent")

        # Explicit user is preserved
        t.task.config.agent.user = "custom-user"
        asyncio.run(t._prepare())
        self.assertEqual(t.task.config.agent.user, "custom-user")

    def test_patch_trial_agent_user_propagates_task_agent_persona(
            self) -> None:

        class _DummyTrial:

            async def _prepare(self) -> None:
                pass

        _ADAPTER._patch_trial_agent_user(_DummyTrial)
        t = _DummyTrial()
        t.task = MagicMock()
        t.task.config.agent.user = None
        t.task.config.environment.env = {
            "TASK_AGENT_PERSONA": "tpu-model-onboarding"
        }
        t.agent = MagicMock()
        t.agent.extra_env = {"AGENT_PERSONA": "tpu-agent"}
        asyncio.run(t._prepare())
        self.assertEqual(
            t.agent.extra_env.get("TASK_AGENT_PERSONA"),
            "tpu-model-onboarding",
        )

        # Baseline arm without AGENT_PERSONA in extra_env is untouched
        t.agent.extra_env = {}
        asyncio.run(t._prepare())
        self.assertNotIn("TASK_AGENT_PERSONA", t.agent.extra_env)


class TestFallbackTrajectoryAdapter(unittest.TestCase):
    """Unit tests for RC5 (early CLI crash) and RC6 (pre-agent setup failure) fallback trajectories."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.logs_dir = self.tmp / "agent"
        self.logs_dir.mkdir(parents=True)

    def _make_trial(self, agent_name: str = "antigravity-cli"):
        trial = MagicMock()
        trial.task.instruction = "Optimize 8p-gemm kernel on TPU v6e."
        trial.paths.agent_dir = self.logs_dir
        trial.paths.exception_message_path = self.tmp / "exception.txt"
        trial.user_agent = None
        trial.agent.logs_dir = self.logs_dir
        trial.agent.name.return_value = agent_name
        trial.agent.version.return_value = "1.3.2"
        trial.agent.model_name = "google/gemini-3-flash"
        trial.agent.session_id = "test-session-123"
        trial.result.id = "trial-id-123"
        trial.result.exception_info = None
        return trial

    def test_agy_early_crash_stages_step_transcript_from_stream_and_err(
            self) -> None:
        import json

        (self.logs_dir / "antigravity-stream.jsonl").write_text(
            '{"event":"result","result":{"conversation_id":"","status":"ERROR",'
            '"response":"","error":"authentication failed or timed out"}}\n',
            encoding="utf-8",
        )
        (self.logs_dir / "antigravity-cli.err").write_text(
            "Error: authentication required. Run 'agy-real' to log in.\n",
            encoding="utf-8",
        )

        trial = self._make_trial("antigravity-cli")
        _ADAPTER._stage_native_fallback_inputs(trial, self.logs_dir)

        staged = self.logs_dir / "antigravity-cli.trajectory.jsonl"
        self.assertTrue(staged.is_file())
        records = [
            json.loads(line)
            for line in staged.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["type"], "USER_INPUT")
        self.assertIn("Optimize 8p-gemm", records[0]["content"])
        self.assertEqual(records[1]["type"], "SYSTEM_ERROR")
        self.assertIn("authentication failed or timed out",
                      records[1]["content"])
        self.assertIn("authentication required", records[1]["content"])

    def test_claude_early_crash_removes_zero_byte_jsonl_and_stages_fallback(
            self) -> None:
        import json

        proj_dir = self.logs_dir / "sessions" / "projects" / "-workspace"
        proj_dir.mkdir(parents=True)
        zero_file = proj_dir / "empty-session.jsonl"
        zero_file.write_text("", encoding="utf-8")
        (self.logs_dir / "claude-code.txt").write_text(
            "Error: Vertex AI quota exceeded for claude-opus-4-6\n",
            encoding="utf-8",
        )

        trial = self._make_trial("claude-code")
        _ADAPTER._stage_native_fallback_inputs(trial, self.logs_dir)

        self.assertFalse(zero_file.exists())
        fallback = proj_dir / "fallback-session.jsonl"
        self.assertTrue(fallback.is_file())
        records = [
            json.loads(line)
            for line in fallback.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(records[0]["type"], "user")
        self.assertEqual(records[1]["type"], "assistant")
        self.assertIn(
            "Vertex AI quota exceeded",
            records[1]["message"]["content"][0]["text"],
        )

    def test_agy_non_step_checkpoint_transcript_is_replaced_by_fallback(
            self) -> None:
        import json

        staged = self.logs_dir / "antigravity-cli.trajectory.jsonl"
        staged.write_text(
            '{"step_index":0,"type":"CHECKPOINT","created_at":"2026-10-20T00:00:00Z"}\n',
            encoding="utf-8",
        )
        (self.logs_dir / "antigravity-stream.jsonl").write_text(
            '{"event":"result","result":{"conversation_id":"","status":"ERROR",'
            '"response":"Model gemini-invalid not found","error":""}}\n',
            encoding="utf-8",
        )

        trial = self._make_trial("antigravity-cli")
        _ADAPTER._stage_native_fallback_inputs(trial, self.logs_dir)

        records = [
            json.loads(line)
            for line in staged.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["type"], "USER_INPUT")
        self.assertEqual(records[1]["type"], "SYSTEM_ERROR")
        self.assertIn("Model gemini-invalid not found", records[1]["content"])

    def test_claude_assistant_only_session_prepends_user_step(self) -> None:
        import json

        proj_dir = self.logs_dir / "sessions" / "projects" / "-workspace"
        proj_dir.mkdir(parents=True)
        session_file = proj_dir / "8161b680-cf5b-480d-bb2f-5e40d7af3a99.jsonl"
        session_file.write_text(
            '{"type":"system","session_id":"8161b680-cf5b-480d-bb2f-5e40d7af3a99"}\n'
            '{"type":"assistant","sessionId":"8161b680-cf5b-480d-bb2f-5e40d7af3a99",'
            '"message":{"role":"assistant","content":[{"type":"text","text":'
            '"[Depth 0] [Early CLI Crash / Error Output]\\nAPI Error: 401"}]}}\n',
            encoding="utf-8",
        )

        trial = self._make_trial("claude-code")
        _ADAPTER._stage_native_fallback_inputs(trial, self.logs_dir)

        records = [
            json.loads(line)
            for line in session_file.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(records[0]["type"], "user")
        self.assertIn("Optimize 8p-gemm", records[0]["message"]["content"])
        assistant_records = [
            r for r in records if r.get("type") == "assistant"
        ]
        self.assertEqual(len(assistant_records), 1)

    def test_pre_agent_setup_failure_initializes_agent_result_and_writes_atif_trajectory(
            self) -> None:
        import json

        class _DummyTrial:

            async def _sync_agent_output(self, target) -> None:
                self._populate_agent_context(target.agent_result)

            def _populate_agent_context(self, agent_result) -> None:
                pass

        _ADAPTER._patch_trial_fallback_trajectory(_DummyTrial)
        t = _DummyTrial()
        base = self._make_trial("claude-code")
        t.task = base.task
        t.paths = base.paths
        t.user_agent = None
        t.agent = base.agent
        t.result = MagicMock()
        t.result.id = "trial-rc6"
        t.result.agent_result = None
        t.result.exception_info = MagicMock(
            exception_type="EnvironmentStartTimeoutError",
            exception_message="Pod failed to start within 900s",
        )
        setup_dir = self.logs_dir / "setup"
        setup_dir.mkdir(parents=True)
        (setup_dir / "stderr.txt").write_text(
            "Back-off pulling image us-central2-docker.pkg.dev/...\n",
            encoding="utf-8",
        )
        cmd_dir = self.logs_dir / "command-0"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "stderr.txt").write_text(
            "Fatal CLI startup error in command-0\n",
            encoding="utf-8",
        )

        asyncio.run(t._sync_agent_output(t.result))

        self.assertIsNotNone(t.result.agent_result)
        traj_path = self.logs_dir / "trajectory.json"
        self.assertTrue(traj_path.is_file())
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
        self.assertEqual(traj["schema_version"], "ATIF-v1.7")
        self.assertEqual(traj["agent"]["name"], "claude-code")
        self.assertEqual(len(traj["steps"]), 2)
        self.assertEqual(traj["steps"][0]["source"], "user")
        self.assertIn("Optimize 8p-gemm", traj["steps"][0]["message"])
        self.assertEqual(traj["steps"][1]["source"], "system")
        self.assertIn("EnvironmentStartTimeoutError",
                      traj["steps"][1]["message"])
        self.assertIn("[setup/stderr.txt]", traj["steps"][1]["message"])
        self.assertIn("Back-off pulling image", traj["steps"][1]["message"])
        self.assertIn("[command-0/stderr.txt]", traj["steps"][1]["message"])
        self.assertIn("Fatal CLI startup error", traj["steps"][1]["message"])

    def test_uncaught_agent_exception_and_placeholder_refresh(self) -> None:
        import json

        trial = self._make_trial("antigravity-cli")
        # 1. Active exception via sys.exc_info() when trial.result.exception_info is still None
        try:
            raise ValueError("Unsupported model identifier")
        except ValueError:
            _ADAPTER._write_fallback_atif_trajectory(trial, self.logs_dir)

        traj_path = self.logs_dir / "trajectory.json"
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
        self.assertEqual(
            traj["steps"][1]["extra"]["exception_type"],
            "ValueError",
        )
        self.assertIn("ValueError: Unsupported model identifier",
                      traj["steps"][1]["message"])

        # 2. Placeholder fallback (exception_type=None) is refreshed when exception_info arrives later
        traj_path.unlink()
        _ADAPTER._write_fallback_atif_trajectory(trial, self.logs_dir)
        placeholder = json.loads(traj_path.read_text(encoding="utf-8"))
        self.assertIsNone(placeholder["steps"][1]["extra"]["exception_type"])

        trial.result.exception_info = MagicMock(
            exception_type="VerifierTimeoutError",
            exception_message="Verifier timed out after 600s",
        )
        _ADAPTER._write_fallback_atif_trajectory(trial, self.logs_dir)
        refreshed = json.loads(traj_path.read_text(encoding="utf-8"))
        self.assertEqual(
            refreshed["steps"][1]["extra"]["exception_type"],
            "VerifierTimeoutError",
        )
        self.assertIn("VerifierTimeoutError: Verifier timed out",
                      refreshed["steps"][1]["message"])

    def test_existing_trajectory_json_is_never_overwritten(self) -> None:
        traj_path = self.logs_dir / "trajectory.json"
        original = '{"schema_version": "ATIF-v1.7", "existing": true}\n'
        traj_path.write_text(original, encoding="utf-8")

        trial = self._make_trial("antigravity-cli")
        trial.result.exception_info = MagicMock(
            exception_type="RuntimeError",
            exception_message="Should not overwrite real trajectory",
        )
        _ADAPTER._write_fallback_atif_trajectory(trial, self.logs_dir)
        self.assertEqual(traj_path.read_text(encoding="utf-8"), original)

    def test_bridged_trial_populates_coding_agent_trajectory(self) -> None:
        import json

        class _DummyBridgedTrial:

            async def _sync_agent_output(self, target) -> None:
                self._populate_agent_context(target.agent_result)

            def _populate_agent_context(self, agent_result) -> None:
                running = self.user_agent or self.agent
                running.populate_context_post_run(agent_result)

        _ADAPTER._patch_trial_fallback_trajectory(_DummyBridgedTrial)
        user_dir = Path(self.tmp.name) / "user-agent"
        agent_dir = Path(self.tmp.name) / "coding-agent"
        user_dir.mkdir(parents=True)
        agent_dir.mkdir(parents=True)

        t = _DummyBridgedTrial()
        base = self._make_trial("claude-code")
        t.task = base.task
        t.paths = MagicMock(agent_dir=agent_dir)
        t.result = MagicMock(id="trial-bridged", exception_info=None)

        t.user_agent = MagicMock()
        t.user_agent.logs_dir = user_dir
        t.user_agent.name.return_value = "claude-code"
        t.user_agent.version.return_value = "1.0"
        t.user_agent.populate_context_post_run.side_effect = (
            lambda _ctx: (user_dir / "trajectory.json").write_text(
                '{"schema_version": "ATIF-v1.7", "steps": [{"step_id": 1}]}',
                encoding="utf-8",
            ))

        t.agent = MagicMock()
        t.agent.logs_dir = agent_dir
        t.agent.name.return_value = "claude-code"
        t.agent.version.return_value = "1.0"
        t.agent.populate_context_post_run.side_effect = (lambda _ctx: (
            agent_dir / "trajectory.json"
        ).write_text(
            '{"schema_version": "ATIF-v1.7", "steps": [{"step_id": 1}, {"step_id": 2}]}',
            encoding="utf-8",
        ))

        proj_dir = agent_dir / "sessions" / "projects" / "-workspace"
        proj_dir.mkdir(parents=True)
        probe_jsonl = proj_dir / "probe-crash.jsonl"
        probe_jsonl.write_text(
            '{"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "[Early CLI Crash / Error Output]"}]}}\n',
            encoding="utf-8",
        )
        real_jsonl = proj_dir / "real-session.jsonl"
        real_jsonl.write_text(
            '{"type": "user", "message": {"role": "user", "content": "Run the task"}}\n'
            '{"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Done"}]}}\n',
            encoding="utf-8",
        )

        t._populate_agent_context(MagicMock())

        t.user_agent.populate_context_post_run.assert_called_once()
        t.agent.populate_context_post_run.assert_called_once()
        self.assertFalse(probe_jsonl.exists())
        self.assertTrue(real_jsonl.is_file())
        self.assertTrue((user_dir / "trajectory.json").is_file())
        self.assertTrue((agent_dir / "trajectory.json").is_file())
        coding_traj = json.loads(
            (agent_dir / "trajectory.json").read_text(encoding="utf-8"))
        self.assertEqual(len(coding_traj["steps"]), 2)


if __name__ == "__main__":
    unittest.main()
