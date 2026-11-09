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
"""Tests for Antigravity CLI subagent token accounting in session_supervisor.sh."""

from __future__ import annotations

import json
import os
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

try:
    from harbor.agents.installed.antigravity_cli import AntigravityCli
    from harbor.models.agent.context import AgentContext
except ImportError:
    AntigravityCli = None
    AgentContext = None

_SUPERVISOR = (Path(__file__).resolve().parents[1] / "hooks" / "agy" /
               "session_supervisor.sh")
_ROOT_ID = "f000eccc-e194-41f8-b141-afe4b52843d9"
_SUB1_ID = "0100ef57-ac1d-417c-8c0a-180033a534b6"
_SUB2_ID = "91b4c2d0-1122-4333-9444-556677889900"


def _varint(val: int) -> bytes:
    out = bytearray()
    while val >= 0x80:
        out.append((val & 0x7F) | 0x80)
        val >>= 7
    out.append(val)
    return bytes(out)


def _vfield(field_num: int, val: int) -> bytes:
    return (_varint(field_num << 3) + _varint(val)) if val else b""


def _bfield(field_num: int, payload: bytes) -> bytes:
    return _varint((field_num << 3) | 2) + _varint(len(payload)) + payload


def _encode_turn_blob(inp: int,
                      out: int,
                      think: int = 0,
                      cache: int = 0) -> bytes:
    """Build a CortexStepGeneratorMetadata.chat_model.usage protobuf."""
    usage_pb = b"".join([
        _vfield(1, 1318),
        _vfield(2, inp),
        _vfield(3, out),
        _vfield(5, cache),
        _vfield(6, 3),
        _vfield(9, think),
        _vfield(10, max(0, out - think)),
    ])
    return _bfield(1, _bfield(4, usage_pb))


def _write_db(db_path: Path, turns: list[tuple[int, int, int, int]]) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "CREATE TABLE gen_metadata (idx INTEGER PRIMARY KEY, data BLOB, size INTEGER)"
        )
        for idx, (inp, out, think, cache) in enumerate(turns):
            blob = _encode_turn_blob(inp, out, think, cache)
            conn.execute("INSERT INTO gen_metadata VALUES (?, ?, ?)",
                         (idx, blob, len(blob)))


class TestAntigravitySessionSupervisorUsage(unittest.TestCase):

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.home = self.root / "home"
        self.conv_dir = self.home / ".gemini" / "antigravity-cli" / "conversations"
        self.logs_dir = self.root / "logs" / "agent"
        self.workspace = self.root / "workspace"
        self.bin_dir = self.root / "bin"
        for d in (self.conv_dir, self.logs_dir, self.workspace, self.bin_dir):
            d.mkdir(parents=True, exist_ok=True)
        if not shutil.which("jq"):
            jq_stub = self.bin_dir / "jq"
            jq_stub.write_text(
                '#!/bin/sh\n'
                'if [ "$3" = "p" ]; then echo "{}"; '
                'elif [ "$6" = "content" ]; then printf \'{"content":"%s"}\\n\' "$7"; '
                'fi\n',
                encoding="utf-8")
            jq_stub.chmod(0o755)
        self.run_marker = self.root / ".harbor-antigravity-run-start"
        self.run_marker.touch()
        (self.logs_dir / "antigravity-cli.trajectory.jsonl").write_text(
            '{"step_index":0,"type":"USER_INPUT","source":"USER_EXPLICIT","status":"DONE","created_at":"2026-10-22T00:00:01Z","content":"task"}\n'
            '{"step_index":1,"type":"PLANNER_RESPONSE","source":"MODEL","status":"DONE","created_at":"2026-10-22T00:00:02Z","content":"done"}\n',
            encoding="utf-8",
        )

    def _make_cli(self, body: str) -> tuple[Path, dict[str, str]]:
        cli = self.root / "fake-cli"
        cli.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
        cli.chmod(0o755)
        env = {
            "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
            "HOME": str(self.home),
            "AGY_REAL_BIN": str(cli),
            "LOGS_AGENT_DIR": str(self.logs_dir),
            "HARBOR_RUN_MARKER": str(self.run_marker),
        }
        return cli, env

    def _run_cli(self, body: str) -> dict[str, int]:
        _, env = self._make_cli(body)
        done = subprocess.run(
            ["bash", str(_SUPERVISOR), "--prompt=task"],
            cwd=self.workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False)
        self.assertEqual(done.returncode, 0, done.stderr)
        return self._last_result_usage()

    def _last_result_usage(self) -> dict[str, int]:
        stream = (self.logs_dir /
                  "antigravity-stream.jsonl").read_text(encoding="utf-8")
        results = [
            json.loads(ln)["result"]["usage"] for ln in stream.splitlines()
            if ln.strip() and json.loads(ln).get("event") == "result"
        ]
        self.assertTrue(results)
        return results[-1]

    def test_single_agent_run_does_not_double_count_root_usage(self) -> None:
        _write_db(self.conv_dir / f"{_ROOT_ID}.db", [(1000, 200, 50, 0),
                                                     (800, 150, 30, 900)])
        usage = self._run_cli(
            f'printf \'{{"event":"session_start","session_id":"{_ROOT_ID}"}}\\n\'\n'
            f'printf \'{{"event":"result","session_id":"{_ROOT_ID}","result":{{"conversation_id":"{_ROOT_ID}","status":"DONE","usage":{{"input_tokens":1800,"output_tokens":350,"thinking_tokens":80,"cache_read_tokens":900,"total_tokens":2150}}}}}}\\n\''
        )
        self.assertEqual(
            usage, {
                "input_tokens": 1800,
                "output_tokens": 350,
                "thinking_tokens": 80,
                "cache_read_tokens": 900,
                "total_tokens": 2150,
            })

    def test_multi_subagent_run_aggregates_all_current_run_dbs(self) -> None:
        _write_db(self.conv_dir / f"{_ROOT_ID}.db", [(1200, 100, 40, 0),
                                                     (600, 80, 20, 1100)])
        _write_db(self.conv_dir / f"{_SUB1_ID}.db", [
            (5000, 900, 400, 0),
            (3000, 1100, 500, 15000),
            (2000, 500, 100, 20000),
        ])
        _write_db(self.conv_dir / f"{_SUB2_ID}.db", [(4000, 600, 250, 8000)])
        stale_db = self.conv_dir / "stale-prior-run.db"
        _write_db(stale_db, [(999999, 999999, 0, 999999)])
        os.utime(stale_db, (time.time() - 3600, ) * 2)

        usage = self._run_cli(
            f'printf \'{{"event":"session_start","session_id":"{_ROOT_ID}"}}\\n\'\n'
            f'printf \'{{"event":"result","session_id":"{_ROOT_ID}","result":{{"conversation_id":"{_ROOT_ID}","status":"DONE","usage":{{"input_tokens":1800,"output_tokens":180,"thinking_tokens":60,"cache_read_tokens":1100,"total_tokens":1980}}}}}}\\n\''
        )
        self.assertEqual(
            usage, {
                "input_tokens": 15800,
                "output_tokens": 3280,
                "thinking_tokens": 1310,
                "cache_read_tokens": 44100,
                "total_tokens": 19080,
            })
        if AntigravityCli is not None and AgentContext is not None:
            ctx = AgentContext()
            AntigravityCli(
                logs_dir=self.logs_dir,
                model_name="google/gemini-3-flash").populate_context_post_run(
                    ctx)
            self.assertEqual(
                (ctx.n_input_tokens, ctx.n_output_tokens, ctx.n_cache_tokens),
                (59900, 3280, 44100))
            traj = json.loads((self.logs_dir /
                               "trajectory.json").read_text(encoding="utf-8"))
            self.assertEqual(traj["final_metrics"]["total_prompt_tokens"],
                             59900)
            self.assertEqual(traj["final_metrics"]["total_completion_tokens"],
                             3280)
            self.assertEqual(traj["final_metrics"]["total_cached_tokens"],
                             44100)

    def test_sigterm_timeout_aggregates_usage_without_native_result(
            self) -> None:
        _write_db(self.conv_dir / f"{_ROOT_ID}.db", [(2000, 300, 100, 500)])
        _write_db(self.conv_dir / f"{_SUB1_ID}.db", [(8000, 1200, 600, 12000),
                                                     (6000, 900, 400, 18000)])
        _, env = self._make_cli(
            "IFS= read -r _\n"
            f'printf \'{{"event":"init","conversation_id":"{_ROOT_ID}","init":{{}}}}\\n\'\n'
            f'printf \'{{"event":"step_update","step_update":{{"conversation_id":"{_ROOT_ID}","step_index":1,"usage":{{"input_tokens":2000,"output_tokens":300,"thinking_tokens":100,"cache_read_tokens":500}}}}}}\\n\'\n'
            "sleep 30")
        proc = subprocess.Popen(
            ["bash", str(_SUPERVISOR), "--prompt=task"],
            cwd=self.workspace,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True)
        assert proc.stdout is not None
        assert proc.stderr is not None
        proc.stdout.readline()
        proc.stdout.readline()
        proc.stdout.close()
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=15)
        proc.stderr.close()

        self.assertEqual(
            self._last_result_usage(), {
                "input_tokens": 16000,
                "output_tokens": 2400,
                "thinking_tokens": 1100,
                "cache_read_tokens": 30500,
                "total_tokens": 18400,
            })

    def test_empty_raw_stream_falls_back_to_stream_file(self) -> None:
        _write_db(self.conv_dir / f"{_ROOT_ID}.db", [(2000, 300, 100, 500)])
        _write_db(self.conv_dir / f"{_SUB1_ID}.db", [(8000, 1200, 600, 12000)])
        (self.logs_dir / "antigravity-stream.jsonl").write_text(
            f'{{"event":"step_update","step_update":{{"conversation_id":"{_ROOT_ID}","step_index":1,"usage":{{"input_tokens":2000,"output_tokens":300,"thinking_tokens":100,"cache_read_tokens":500}}}}}}\n',
            encoding="utf-8",
        )
        _, env = self._make_cli(
            "IFS= read -r _\nprintf 'READY_FOR_TERM\\n' >&2\nsleep 30")
        proc = subprocess.Popen(
            ["bash", str(_SUPERVISOR), "--prompt=task"],
            cwd=self.workspace,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True)
        assert proc.stdout is not None
        assert proc.stderr is not None
        while "READY_FOR_TERM" not in proc.stderr.readline():
            pass
        proc.stdout.close()
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=15)
        proc.stderr.close()

        self.assertEqual(
            self._last_result_usage(), {
                "input_tokens": 10000,
                "output_tokens": 1500,
                "thinking_tokens": 700,
                "cache_read_tokens": 12500,
                "total_tokens": 11500,
            })

    def test_task_agent_persona_overrides_agent_persona(self) -> None:
        _write_db(self.conv_dir / f"{_ROOT_ID}.db", [(1000, 200, 50, 0)])
        args_log = self.root / "args.log"
        _, env = self._make_cli(
            f'echo "$*" >> "{args_log}"\n'
            f'printf \'{{"event":"session_start","session_id":"{_ROOT_ID}"}}\\n\'\n'
            f'printf \'{{"event":"result","session_id":"{_ROOT_ID}","result":{{"conversation_id":"{_ROOT_ID}","status":"DONE","usage":{{"input_tokens":1000,"output_tokens":200,"thinking_tokens":50,"cache_read_tokens":0,"total_tokens":1200}}}}}}\\n\''
        )
        env["AGENT_PERSONA"] = "tpu-agent"
        env["TASK_AGENT_PERSONA"] = "tpu-model-onboarding"
        (self.logs_dir /
         "antigravity-cli.trajectory.jsonl").unlink(missing_ok=True)
        done = subprocess.run(
            ["bash", str(_SUPERVISOR), "--prompt=task"],
            cwd=self.workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        args_text = args_log.read_text(encoding="utf-8")
        self.assertIn("--agent tpu-model-onboarding", args_text)
        self.assertNotIn("--agent tpu-agent", args_text)
        traj_text = (self.logs_dir /
                     "antigravity-cli.trajectory.jsonl").read_text(
                         encoding="utf-8")
        self.assertIn("[Depth 0 | tpu-model-onboarding]", traj_text)


if __name__ == "__main__":
    unittest.main()
