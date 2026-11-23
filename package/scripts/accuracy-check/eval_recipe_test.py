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
"""Tests for eval_recipe.sh single-host and multi-host execution."""

import os
import shutil
import signal
import socket
import subprocess
import tarfile
import tempfile
import textwrap
import time
import unittest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "eval_recipe.sh")
TIMEOUT_ENVS = ("VLLM_ENGINE_READY_TIMEOUT_S",
                "VLLM_ENGINE_ITERATION_TIMEOUT_S",
                "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS")

FAKE_SERVER_PY = textwrap.dedent("""\
    import http.server
    import json
    import os
    import signal
    import sys

    port = int(sys.argv[1])
    env_out = sys.argv[2]
    stop_flag = sys.argv[3] if len(sys.argv) > 3 else ""

    with open(env_out, "w", encoding="utf-8") as fh:
        for key in (
            "VLLM_ENGINE_READY_TIMEOUT_S",
            "VLLM_ENGINE_ITERATION_TIMEOUT_S",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
        ):
            fh.write(f"{key}={os.environ.get(key, '')}\\n")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            body = json.dumps({
                "choices": [{
                    "finish_reason": "stop",
                    "message": {"content": "391"},
                }],
                "usage": {"completion_tokens": 1},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # pylint: disable=redefined-builtin
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)

    def _stop(_signum, _frame):
        if stop_flag:
            with open(stop_flag, "w", encoding="utf-8") as fh:
                fh.write("stopped\\n")
        os._exit(0)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    srv.serve_forever()
""")

FAKE_EVAL_PY = textwrap.dedent("""\
    import json
    import os
    import sys

    out_dir = os.environ["EVAL_OUTPUT_DIR"]
    marker = sys.argv[1]
    exit_code = int(sys.argv[2])
    with open(marker, "a", encoding="utf-8") as fh:
        fh.write(f"ran worker={os.environ.get('TPU_WORKER_ID', '0')}\\n")
    os.makedirs(out_dir, exist_ok=True)
    if exit_code == 0:
        results = {
            "groups": {
                "mmlu_pro": {
                    "alias": "mmlu_pro",
                    "exact_match,custom-extract": 0.75,
                    "exact_match_stderr,custom-extract": 0.01,
                }
            },
            "results": {
                "mmlu_pro": {
                    "alias": "mmlu_pro",
                    "exact_match,custom-extract": 0.75,
                    "exact_match_stderr,custom-extract": 0.01,
                },
                "mmlu_pro_math": {
                    "exact_match,custom-extract": 0.75,
                    "exact_match_stderr,custom-extract": 0.01,
                },
            },
            "n-samples": {
                "mmlu_pro_math": {"effective": 1400},
            },
            "versions": {
                "mmlu_pro": "2.0",
                "mmlu_pro_math": "3.0",
            },
        }
        with open(
            os.path.join(out_dir, "results_2026-01-01.json"),
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(results, fh)
        with open(
            os.path.join(out_dir, "samples_mmlu_pro_math.jsonl"),
            "w",
            encoding="utf-8",
        ) as fh:
            fh.write(json.dumps({"resps": [["(A)"]]}) + "\\n")
    sys.exit(exit_code)
""")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class EvalRecipeTest(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        # Stub `lm_eval` so the recipe skips pip install.
        self.stub_py = os.path.join(self.root, "pylib")
        os.makedirs(self.stub_py)
        with open(os.path.join(self.stub_py, "lm_eval.py"),
                  "w",
                  encoding="utf-8") as fh:
            fh.write("__version__ = '0.5.0.dev1'\n")
        self.fake_server = os.path.join(self.root, "fake_server.py")
        with open(self.fake_server, "w", encoding="utf-8") as fh:
            fh.write(FAKE_SERVER_PY)
        self.fake_eval = os.path.join(self.root, "fake_eval.py")
        with open(self.fake_eval, "w", encoding="utf-8") as fh:
            fh.write(FAKE_EVAL_PY)

    def _base_env(self, extra=None):
        env = os.environ.copy()
        for key in (*TIMEOUT_ENVS, "TPU_NUM_HOSTS", "TPU_WORKER_ID",
                    "TPU_HEAD_HOST", "JOB_COMPLETION_INDEX", "EVAL_COORD_PORT",
                    "EVAL_COORD_GRACE_S", "EVAL_COORD_POLL_S",
                    "EVAL_COORD_EXIT_RETRIES"):
            env.pop(key, None)
        existing_py = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (f"{self.stub_py}:{existing_py}"
                             if existing_py else self.stub_py)
        env["SERVER_HEALTH_POLL_S"] = "0.1"
        if extra:
            env.update(extra)
        return env

    def _make_workdir(self, name, server_cmd, eval_cmd):
        work = os.path.join(self.root, name)
        os.makedirs(work, exist_ok=True)
        with open(os.path.join(work, "vllm_server_cmd.txt"),
                  "w",
                  encoding="utf-8") as fh:
            fh.write(server_cmd)
        with open(os.path.join(work, "eval_cmd.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(eval_cmd)
        return work

    def test_single_host_happy_path_and_timeout_defaults(self):
        port = _free_port()
        env_out = os.path.join(self.root, "server_env.txt")
        eval_marker = os.path.join(self.root, "eval_ran.txt")
        work = self._make_workdir(
            "single",
            f"python3 {self.fake_server} {port} {env_out}",
            f"python3 {self.fake_eval} {eval_marker} 0 # base_url=${{EVAL_BASE_URL}}",
        )
        proc = subprocess.run(
            [
                "bash",
                SCRIPT,
                "SERVED_MODEL_NAME=test/model",
                f"SERVER_PORT={port}",
                "SERVER_WAIT_TIMEOUT_S=30",
            ],
            cwd=work,
            env=self._base_env(),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("smoke: content=391", proc.stdout)
        self.assertIn("eval-summary: null-content answers=0", proc.stdout)
        self.assertIn("eval-summary: mmlu_pro exact_match,custom-extract=0.75",
                      proc.stdout)
        self.assertIn("eval exited rc=0", proc.stdout)
        with open(env_out, encoding="utf-8") as fh:
            recorded = dict(line.strip().split("=", 1) for line in fh
                            if "=" in line)
        self.assertEqual(recorded["VLLM_ENGINE_READY_TIMEOUT_S"], "3600")
        self.assertEqual(recorded["VLLM_ENGINE_ITERATION_TIMEOUT_S"], "1800")
        self.assertEqual(recorded["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"],
                         "1800")
        with tarfile.open(os.path.join(work, "eval_out.tar.gz")) as tar:
            names = tar.getnames()
        self.assertTrue(any("results_2026-01-01.json" in n for n in names),
                        names)

    def test_timeout_env_override_preserved(self):
        port = _free_port()
        env_out = os.path.join(self.root, "server_env_override.txt")
        eval_marker = os.path.join(self.root, "eval_override.txt")
        work = self._make_workdir(
            "override",
            f"python3 {self.fake_server} {port} {env_out}",
            f"python3 {self.fake_eval} {eval_marker} 0 # base_url=${{EVAL_BASE_URL}}",
        )
        proc = subprocess.run(
            [
                "bash",
                SCRIPT,
                "SERVED_MODEL_NAME=test/model",
                f"SERVER_PORT={port}",
            ],
            cwd=work,
            env=self._base_env({"VLLM_ENGINE_READY_TIMEOUT_S": "9600"}),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        with open(env_out, encoding="utf-8") as fh:
            recorded = dict(line.strip().split("=", 1) for line in fh
                            if "=" in line)
        self.assertEqual(recorded["VLLM_ENGINE_READY_TIMEOUT_S"], "9600")

    def test_multihost_leader_and_worker_happy_path(self):
        server_port = _free_port()
        coord_port = _free_port()
        env_out = os.path.join(self.root, "mh_env.txt")
        eval_marker = os.path.join(self.root, "mh_eval_ran.txt")
        worker_started = os.path.join(self.root, "worker_started.txt")
        worker_stopped = os.path.join(self.root, "worker_stopped.txt")

        # Same command on both hosts, branching on TPU_WORKER_ID like serve_vllm.sh.
        server_cmd = textwrap.dedent(f"""\
            if [ "${{TPU_WORKER_ID:-0}}" = "0" ]; then
              exec python3 {self.fake_server} {server_port} {env_out}
            else
              trap 'echo stopped > {worker_stopped}; exit 0' TERM INT
              echo started > {worker_started}
              while true; do sleep 0.2; done
            fi
        """)
        eval_cmd = (f"python3 {self.fake_eval} {eval_marker} 0 "
                    "# base_url=${EVAL_BASE_URL}")
        leader_dir = self._make_workdir("mh_leader", server_cmd, eval_cmd)
        worker_dir = self._make_workdir("mh_worker", server_cmd, eval_cmd)

        common_args = [
            "bash",
            SCRIPT,
            "SERVED_MODEL_NAME=test/model",
            f"SERVER_PORT={server_port}",
            f"EVAL_COORD_PORT={coord_port}",
        ]
        shared_env = {
            "TPU_NUM_HOSTS": "2",
            "TPU_HEAD_HOST": "127.0.0.1",
            "EVAL_COORD_POLL_S": "0.2",
            "EVAL_COORD_GRACE_S": "15",
        }
        worker_proc = subprocess.Popen(
            common_args,
            cwd=worker_dir,
            env=self._base_env({
                **shared_env, "TPU_WORKER_ID": "1"
            }),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        leader_proc = subprocess.Popen(
            common_args,
            cwd=leader_dir,
            env=self._base_env({
                **shared_env, "TPU_WORKER_ID": "0"
            }),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            l_out, l_err = leader_proc.communicate(timeout=30)
            w_out, w_err = worker_proc.communicate(timeout=30)
        finally:
            for p in (leader_proc, worker_proc):
                if p.poll() is None:
                    p.kill()

        self.assertEqual(leader_proc.returncode, 0, l_out + l_err)
        self.assertEqual(worker_proc.returncode, 0, w_out + w_err)
        self.assertIn("eval exited rc=0", l_out)
        self.assertIn("received leader done", w_out)
        self.assertIn("eval worker exited rc=0", w_out)
        self.assertTrue(os.path.exists(worker_started))
        self.assertTrue(os.path.exists(worker_stopped))
        with open(eval_marker, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        self.assertEqual(lines, ["ran worker=0"])

    def test_multihost_worker_server_exits_nonzero_on_leader_stop(self):
        server_port = _free_port()
        coord_port = _free_port()
        env_out = os.path.join(self.root, "race_env.txt")
        leader_stopped = os.path.join(self.root, "leader_stopped.txt")
        eval_marker = os.path.join(self.root, "race_eval.txt")

        # Worker exits rc=1 as soon as leader stops its server (simulating an
        # mp headless peer disconnect before the next poll interval).
        server_cmd = textwrap.dedent(f"""\
            if [ "${{TPU_WORKER_ID:-0}}" = "0" ]; then
              exec python3 {self.fake_server} {server_port} {env_out} {leader_stopped}
            else
              while [ ! -f {leader_stopped} ]; do sleep 0.05; done
              exit 1
            fi
        """)
        eval_cmd = (f"python3 {self.fake_eval} {eval_marker} 0 "
                    "# base_url=${EVAL_BASE_URL}")
        leader_dir = self._make_workdir("race_leader", server_cmd, eval_cmd)
        worker_dir = self._make_workdir("race_worker", server_cmd, eval_cmd)

        common_args = [
            "bash",
            SCRIPT,
            "SERVED_MODEL_NAME=test/model",
            f"SERVER_PORT={server_port}",
            f"EVAL_COORD_PORT={coord_port}",
        ]
        shared_env = {
            "TPU_NUM_HOSTS": "2",
            "TPU_HEAD_HOST": "127.0.0.1",
            "EVAL_COORD_POLL_S": "5",
            "EVAL_COORD_GRACE_S": "15",
            "EVAL_COORD_EXIT_RETRIES": "10",
        }
        worker_proc = subprocess.Popen(
            common_args,
            cwd=worker_dir,
            env=self._base_env({
                **shared_env, "TPU_WORKER_ID": "1"
            }),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        leader_proc = subprocess.Popen(
            common_args,
            cwd=leader_dir,
            env=self._base_env({
                **shared_env, "TPU_WORKER_ID": "0"
            }),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            l_out, l_err = leader_proc.communicate(timeout=30)
            w_out, w_err = worker_proc.communicate(timeout=30)
        finally:
            for p in (leader_proc, worker_proc):
                if p.poll() is None:
                    p.kill()

        self.assertEqual(leader_proc.returncode, 0, l_out + l_err)
        self.assertEqual(worker_proc.returncode, 0, w_out + w_err)
        expected = ("after leader done; exiting 0", "received leader done")
        self.assertTrue(any(msg in w_out for msg in expected), w_out + w_err)

    def test_multihost_leader_eval_failure_still_stops_worker_cleanly(self):
        server_port = _free_port()
        coord_port = _free_port()
        env_out = os.path.join(self.root, "fail_env.txt")
        eval_marker = os.path.join(self.root, "fail_eval.txt")

        server_cmd = textwrap.dedent(f"""\
            if [ "${{TPU_WORKER_ID:-0}}" = "0" ]; then
              exec python3 {self.fake_server} {server_port} {env_out}
            else
              while true; do sleep 0.2; done
            fi
        """)
        eval_cmd = (f"python3 {self.fake_eval} {eval_marker} 1 "
                    "# base_url=${EVAL_BASE_URL}")
        leader_dir = self._make_workdir("fail_leader", server_cmd, eval_cmd)
        worker_dir = self._make_workdir("fail_worker", server_cmd, eval_cmd)

        common_args = [
            "bash",
            SCRIPT,
            "SERVED_MODEL_NAME=test/model",
            f"SERVER_PORT={server_port}",
            f"EVAL_COORD_PORT={coord_port}",
        ]
        shared_env = {
            "TPU_NUM_HOSTS": "2",
            "TPU_HEAD_HOST": "127.0.0.1",
            "EVAL_COORD_POLL_S": "0.2",
            "EVAL_COORD_GRACE_S": "15",
        }
        worker_proc = subprocess.Popen(
            common_args,
            cwd=worker_dir,
            env=self._base_env({
                **shared_env, "TPU_WORKER_ID": "1"
            }),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        leader_proc = subprocess.Popen(
            common_args,
            cwd=leader_dir,
            env=self._base_env({
                **shared_env, "TPU_WORKER_ID": "0"
            }),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            l_out, l_err = leader_proc.communicate(timeout=30)
            w_out, w_err = worker_proc.communicate(timeout=30)
        finally:
            for p in (leader_proc, worker_proc):
                if p.poll() is None:
                    p.kill()

        self.assertEqual(leader_proc.returncode, 1, l_out + l_err)
        self.assertEqual(worker_proc.returncode, 0, w_out + w_err)
        self.assertIn("eval exited rc=1", l_out)

    def test_multihost_worker_early_crash_fails_fast(self):
        coord_port = _free_port()
        work = self._make_workdir(
            "early_crash",
            "echo 'vllm patch does not apply: bad.patch'; exit 4",
            "echo base_url",
        )
        proc = subprocess.run(
            [
                "bash",
                SCRIPT,
                "SERVED_MODEL_NAME=test/model",
                f"EVAL_COORD_PORT={coord_port}",
            ],
            cwd=work,
            env=self._base_env({
                "TPU_NUM_HOSTS": "2",
                "TPU_WORKER_ID": "1",
                "TPU_HEAD_HOST": "127.0.0.1",
                "EVAL_COORD_POLL_S": "0.2",
                "EVAL_COORD_EXIT_RETRIES": "2",
            }),
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(proc.returncode, 4, proc.stdout + proc.stderr)
        self.assertIn("server exited early (rc=4)", proc.stdout)
        self.assertIn("vllm patch does not apply: bad.patch", proc.stdout)

    def test_multihost_missing_head_host_fails_fast(self):
        work = self._make_workdir("no_head", "true", "echo base_url")
        t0 = time.monotonic()
        proc = subprocess.run(
            ["bash", SCRIPT, "SERVED_MODEL_NAME=test/model"],
            cwd=work,
            env=self._base_env({
                "TPU_NUM_HOSTS": "2",
                "TPU_WORKER_ID": "0"
            }),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(proc.returncode, 1)
        self.assertLess(time.monotonic() - t0, 5.0)
        self.assertIn("requires TPU_HEAD_HOST", proc.stderr)

    def test_multihost_leader_sigterm_stops_beacon_and_closes_pipe(self):
        server_port = _free_port()
        coord_port = _free_port()
        env_out = os.path.join(self.root, "sigterm_env.txt")
        leader_stopped = os.path.join(self.root, "sigterm_leader_stopped.txt")
        eval_marker = os.path.join(self.root, "sigterm_eval.txt")

        server_cmd = (f"exec python3 {self.fake_server} {server_port} "
                      f"{env_out} {leader_stopped}")
        eval_cmd = (f"python3 {self.fake_eval} {eval_marker} 0 "
                    "# base_url=${EVAL_BASE_URL}")
        leader_dir = self._make_workdir("sigterm_leader", server_cmd, eval_cmd)
        ready_file = os.path.join(leader_dir, ".eval_coord_ready")

        leader_proc = subprocess.Popen(
            [
                "bash",
                SCRIPT,
                "SERVED_MODEL_NAME=test/model",
                f"SERVER_PORT={server_port}",
                f"EVAL_COORD_PORT={coord_port}",
            ],
            cwd=leader_dir,
            env=self._base_env({
                "TPU_NUM_HOSTS": "2",
                "TPU_WORKER_ID": "0",
                "TPU_HEAD_HOST": "127.0.0.1",
                "EVAL_COORD_GRACE_S": "60",
            }),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                ready = os.path.exists(ready_file)
                stopped = os.path.exists(leader_stopped)
                if ready and stopped:
                    break
                if leader_proc.poll() is not None:
                    break
                time.sleep(0.05)
            self.assertTrue(os.path.exists(ready_file), "beacon not ready")
            self.assertTrue(os.path.exists(leader_stopped), "server still up")

            t0 = time.monotonic()
            leader_proc.send_signal(signal.SIGTERM)
            # communicate() waits for pipe EOF across all inheritors, mirroring
            # gcs_staging.py's FIFO/tee wait on pod teardown.
            l_out, _ = leader_proc.communicate(timeout=10)
            elapsed = time.monotonic() - t0
        finally:
            if leader_proc.poll() is None:
                leader_proc.kill()

        self.assertLess(elapsed, 5.0, f"hung {elapsed:.2f}s:\n{l_out}")
        self.assertFalse(os.path.exists(ready_file))
        with self.assertRaises(OSError):
            with socket.create_connection(("127.0.0.1", coord_port), 1.0):
                pass

    def test_unknown_key_fails(self):
        work = self._make_workdir("bad_key", "true", "echo base_url")
        proc = subprocess.run(
            ["bash", SCRIPT, "SERVED_MODEL_NAME=test/model", "BOGUS=1"],
            cwd=work,
            env=self._base_env(),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("unknown key 'BOGUS'", proc.stderr)


if __name__ == "__main__":
    unittest.main()
