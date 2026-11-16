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
"""Tests for serve_vllm.sh's vLLM timeout and LIBTPU_INIT_ARGS defaults,
its patch list, and its multi-host leader/worker split.

The script runs for real, against a stub `python3`, `vllm` and `ray` beside
it: the stub vLLM writes the arguments and environment it was started with
to a file, and the stub Ray each call it gets.
"""

import json
import os
import pathlib
import shutil
import socket
import subprocess
import tempfile
import unittest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "serve_vllm.sh")
TIMEOUT_ENVS = ("VLLM_ENGINE_READY_TIMEOUT_S",
                "VLLM_ENGINE_ITERATION_TIMEOUT_S",
                "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS")
LLO_ENVS = ("LIBTPU_INIT_ARGS", "VLLM_PROFILE_LLO")
PROFILER_ENVS = ("USE_PHASED_PROFILER", "PYTHON_TRACER_LEVEL",
                 "VLLM_PROFILE_MAX_ITERATIONS")
LLO_ON = "--xla_xprof_register_llo_debug_info=true"

# Answers every `-c` probe serve_vllm.sh makes as the build in
# STUB_HAS_DIST_FLAG would, and idles as the profile file server.
STUB_PYTHON = """#!/bin/bash
case "$*" in
  *http.server*) exec sleep 30 ;;
  *distributed_timeout_seconds*) [ "$STUB_HAS_DIST_FLAG" = 1 ] ;;
  *"- worker-0.svc"*) echo "2002:db8::1" ;;
  *) exit 0 ;;
esac
"""
STUB_VLLM = """#!/bin/bash
{
  for arg in "$@"; do printf 'ARG %s\\n' "$arg"; done
  env | grep -E '^(VLLM_(ENGINE|EXECUTE)|LIBTPU_INIT_ARGS=|RAY_ADDRESS=|USE_PHASED_PROFILER=|PYTHON_TRACER_LEVEL=)' | sed 's/^/ENV /'
} > "$STUB_OUT"
"""
# Logs each patch it is given, and fails the one named in STUB_BAD_PATCH as
# apply_vllm_patch.sh does a patch cut from another plugin revision.
STUB_APPLIER = """#!/bin/bash
echo "$1" >> "$STUB_APPLIED"
[ "$1" = "$STUB_BAD_PATCH" ] && exit 4
exit 0
"""
# Records each call, answers `status` with RAY_STATUS_TPUS chips, and in
# `start --block` exits with RAY_BLOCK_EXIT (default 0).
STUB_RAY = """#!/bin/bash
echo "RAY $*" >> "$STUB_RAY_OUT"
case "$*" in
  status*) echo " 0.0/${RAY_STATUS_TPUS:-8}.0 TPU" ;;
  *--block*) exit "${RAY_BLOCK_EXIT:-0}" ;;
esac
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServeVllmTimeoutsTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        bin_dir = os.path.join(self.dir, "bin")
        os.makedirs(bin_dir)
        for name, text in (("python3", STUB_PYTHON), ("vllm", STUB_VLLM),
                           ("ray", STUB_RAY)):
            path = os.path.join(bin_dir, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            os.chmod(path, 0o755)
        self.python = os.path.join(bin_dir, "python3")
        self.out = os.path.join(self.dir, "vllm.out")
        self.ray_out = os.path.join(self.dir, "ray.out")

    def _run(self, *args, has_dist_flag=True, script=SCRIPT, **env):
        """Run the script; return (returncode, vllm args, vllm env, ray calls, stderr)."""
        for p in (self.out, self.ray_out):
            if os.path.exists(p):
                os.remove(p)
        full_env = {
            k: v
            for k, v in os.environ.items()
            if k not in TIMEOUT_ENVS + LLO_ENVS + PROFILER_ENVS and k !=
            "VLLM_DISTRIBUTED_TIMEOUT_S" and not k.startswith(("TPU_", "RAY_"))
        }
        full_env.update({
            "TPU_AGENT_SUITE_PYTHON": self.python,
            "VLLM_PROFILE_DIR": os.path.join(self.dir, "profile"),
            "VLLM_PROFILE_FILES_PORT": str(_free_port()),
            "STUB_OUT": self.out,
            "STUB_RAY_OUT": self.ray_out,
            "STUB_HAS_DIST_FLAG": "1" if has_dist_flag else "0",
            **env,
        })
        proc = subprocess.run(["bash", script, "/model", *args],
                              cwd=self.dir,
                              env=full_env,
                              capture_output=True,
                              text=True,
                              timeout=60,
                              check=False)
        lines = (pathlib.Path(self.out).read_text(
            encoding="utf-8").splitlines() if os.path.exists(self.out) else [])
        ray_lines = (pathlib.Path(self.ray_out).read_text(
            encoding="utf-8").splitlines()
                     if os.path.exists(self.ray_out) else [])
        argv = [line[4:] for line in lines if line.startswith("ARG ")]
        envs = dict(line[4:].split("=", 1) for line in lines
                    if line.startswith("ENV "))
        return proc.returncode, argv, envs, [line[4:] for line in ray_lines
                                             ], proc.stderr

    def _serve(self, *args, has_dist_flag=True, script=SCRIPT, rc=0, **env):
        """Run the script; return (vllm args, vllm env we set, stderr)."""
        code, argv, envs, _, stderr = self._run(*args,
                                                has_dist_flag=has_dist_flag,
                                                script=script,
                                                **env)
        self.assertEqual(code, rc, stderr)
        if rc:
            return None, None, stderr
        return argv, envs, stderr

    def test_defaults_are_tpu_sized(self):
        argv, envs, stderr = self._serve("--port", "8000")
        self.assertEqual(
            envs, {
                "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
                "VLLM_ENGINE_ITERATION_TIMEOUT_S": "1800",
                "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1800",
                "LIBTPU_INIT_ARGS": LLO_ON,
                "USE_PHASED_PROFILER": "true",
                "PYTHON_TRACER_LEVEL": "0",
            })
        self.assertEqual(argv[:4], ["serve", "/model", "--port", "8000"])
        i = argv.index("--distributed-timeout-seconds")
        self.assertEqual(argv[i + 1], "1800")
        p = argv.index("--profiler-config")
        self.assertEqual(
            json.loads(argv[p + 1]), {
                "profiler": "torch",
                "torch_profiler_dir": os.path.join(self.dir, "profile"),
                "max_iterations": 5,
                "ignore_frontend": True,
            })
        self.assertIn("serve_vllm: timeouts engine_ready=3600s", stderr)
        self.assertIn("USE_PHASED_PROFILER=true", stderr)
        self.assertIn("max_iterations=5", stderr)

    def test_caller_values_are_kept(self):
        argv, envs, _ = self._serve(VLLM_ENGINE_READY_TIMEOUT_S="9600",
                                    VLLM_DISTRIBUTED_TIMEOUT_S="2400",
                                    USE_PHASED_PROFILER="false",
                                    PYTHON_TRACER_LEVEL="1")
        self.assertEqual(envs["VLLM_ENGINE_READY_TIMEOUT_S"], "9600")
        self.assertEqual(envs["VLLM_ENGINE_ITERATION_TIMEOUT_S"], "1800")
        self.assertEqual(envs["USE_PHASED_PROFILER"], "false")
        self.assertEqual(envs["PYTHON_TRACER_LEVEL"], "1")
        i = argv.index("--distributed-timeout-seconds")
        self.assertEqual(argv[i + 1], "2400")
        p = argv.index("--profiler-config")
        self.assertNotIn("max_iterations", json.loads(argv[p + 1]))
        argv, _, _ = self._serve(USE_PHASED_PROFILER="false",
                                 VLLM_PROFILE_MAX_ITERATIONS="15")
        p = argv.index("--profiler-config")
        self.assertEqual(json.loads(argv[p + 1])["max_iterations"], 15)

    def test_caller_flag_is_not_duplicated(self):
        for flag in (["--distributed-timeout-seconds",
                      "600"], ["--distributed-timeout-seconds=600"]):
            with self.subTest(flag=flag):
                argv, _, _ = self._serve(*flag)
                self.assertEqual(
                    sum(
                        a.startswith("--distributed-timeout-seconds")
                        for a in argv), 1)
                self.assertNotIn("1800", argv)

    def test_flag_left_out_where_vllm_lacks_it(self):
        argv, envs, _ = self._serve(has_dist_flag=False)
        self.assertNotIn("--distributed-timeout-seconds", argv)
        self.assertEqual(envs["VLLM_ENGINE_READY_TIMEOUT_S"], "3600")

    def test_llo_flag_is_appended_to_caller_args(self):
        _, envs, stderr = self._serve(LIBTPU_INIT_ARGS="--xla_foo=1")
        self.assertEqual(envs["LIBTPU_INIT_ARGS"], "--xla_foo=1 " + LLO_ON)
        self.assertIn("serve_vllm: LLO debug info on", stderr)

    def test_caller_llo_value_is_kept(self):
        for args in ("--xla_xprof_register_llo_debug_info=false",
                     "--xla_foo=1 --xla_xprof_register_llo_debug_info",
                     "--xla_xprof_register_llo_debug_info=true --xla_foo=1"):
            with self.subTest(args=args):
                _, envs, stderr = self._serve(LIBTPU_INIT_ARGS=args)
                self.assertEqual(envs["LIBTPU_INIT_ARGS"], args)
                self.assertIn("keeping the caller's", stderr)

    def test_llo_off_leaves_libtpu_args_alone(self):
        _, envs, _ = self._serve(VLLM_PROFILE_LLO="0")
        self.assertNotIn("LIBTPU_INIT_ARGS", envs)
        _, envs, _ = self._serve(VLLM_PROFILE_LLO="0",
                                 LIBTPU_INIT_ARGS="--xla_foo=1")
        self.assertEqual(envs["LIBTPU_INIT_ARGS"], "--xla_foo=1")

    def test_caller_profiler_config_is_not_overridden(self):
        argv, envs, stderr = self._serve(
            "--profiler-config.profiler=torch",
            "--profiler-config.torch_profiler_dir=/custom",
            "--data-parallel-address", "worker-0.svc")
        self.assertNotIn("--profiler-config", argv)
        self.assertNotIn("USE_PHASED_PROFILER", envs)
        self.assertEqual(envs["PYTHON_TRACER_LEVEL"], "0")
        self.assertEqual(argv[argv.index("--data-parallel-address") + 1],
                         "2002:db8::1")
        self.assertIn("keeping the caller's --profiler-config", stderr)

    def _with_applier(self, *patches):
        """A job's layout: the script, the applier beside it, the list."""
        script = os.path.join(self.dir, "serve_vllm.sh")
        shutil.copy(SCRIPT, script)
        with open(os.path.join(self.dir, "apply_vllm_patch.sh"),
                  "w",
                  encoding="utf-8") as f:
            f.write(STUB_APPLIER)
        if patches:
            with open(os.path.join(self.dir, "vllm_patches.txt"),
                      "w",
                      encoding="utf-8") as f:
                f.write("".join(n + "\n" for n in patches))
        return script

    def _applied(self):
        path = os.path.join(self.dir, "applied")
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return f.read().splitlines()

    def test_no_list_applies_nothing(self):
        self._serve(script=self._with_applier(),
                    STUB_APPLIED=os.path.join(self.dir, "applied"))
        self.assertEqual(self._applied(), [])

    def test_patches_apply_in_order_before_vllm(self):
        script = self._with_applier("01-rpa.patch", "02-gate.py")
        argv, _, stderr = self._serve(script=script,
                                      STUB_APPLIED=os.path.join(
                                          self.dir, "applied"))
        self.assertEqual(self._applied(), ["01-rpa.patch", "02-gate.py"])
        self.assertEqual(argv[:2], ["serve", "/model"])
        self.assertIn("serve_vllm: applied 2 patch(es)", stderr)

    def test_a_patch_that_does_not_apply_fails_the_server(self):
        script = self._with_applier("01-rpa.patch", "02-gate.py",
                                    "03-late.patch")
        _, _, stderr = self._serve(script=script,
                                   rc=4,
                                   STUB_APPLIED=os.path.join(
                                       self.dir, "applied"),
                                   STUB_BAD_PATCH="02-gate.py")
        self.assertEqual(self._applied(), ["01-rpa.patch", "02-gate.py"])
        self.assertIn("serve_vllm: patch does not apply: 02-gate.py", stderr)
        self.assertFalse(os.path.exists(self.out))

    def test_multihost_leader_and_worker(self):

        def host(worker: int, *extra: str, **env):
            return self._run("--port",
                             "8000",
                             *extra,
                             TPU_NUM_HOSTS="2",
                             TPU_CHIPS_PER_HOST="4",
                             TPU_WORKER_ID=str(worker),
                             TPU_HEAD_HOST="127.0.0.1",
                             TPU_MULTIHOST_JOIN_TIMEOUT_S="0",
                             **env)[:4]

        code, argv, envs, ray = host(0)
        self.assertEqual(
            (code, argv[:2], envs["RAY_ADDRESS"], ray[0], ray[-1]),
            (0, ["serve", "/model"], "127.0.0.1:6379",
             "start --head --port=6379", "stop --force"))
        self.assertEqual(host(0, RAY_STATUS_TPUS="4")[:2], (1, []))

        with socket.socket() as head:
            head.bind(("127.0.0.1", 0))
            head.listen()
            port = head.getsockname()[1]
            code, argv, _, ray = host(1, TPU_MULTIHOST_RAY_PORT=str(port))
            crashed = host(1,
                           TPU_MULTIHOST_RAY_PORT=str(port),
                           RAY_BLOCK_EXIT="1")[0]
        self.assertEqual(
            (code, crashed, argv, ray),
            (0, 1, [], [f"start --address=127.0.0.1:{port} --block"]))
        self.assertEqual(
            host(0, "--data-parallel-size", "3",
                 TPU_MULTIHOST_BACKEND="mp")[0], 2)

        for worker in (0, 1):
            code, argv, _, ray = host(worker, TPU_MULTIHOST_BACKEND="mp")
            self.assertEqual((code, ray, "--headless" in argv),
                             (0, [], worker == 1))
            i = argv.index("--nnodes")
            self.assertEqual(argv[i:i + 4],
                             ["--nnodes", "2", "--node-rank",
                              str(worker)])
            code, argv, _, ray = host(worker,
                                      "--data-parallel-size",
                                      "8",
                                      "--api-server-count",
                                      "4",
                                      TPU_MULTIHOST_BACKEND="mp")
            self.assertEqual(
                (code, "--nnodes" in argv, "--headless"
                 in argv, "--data-parallel-start-rank" in argv, argv[-3:]
                 == ["--api-server-count", "0", "--headless"]),
                (0, False, worker == 1, worker == 1, worker == 1))
            i = argv.index("--data-parallel-size-local")
            self.assertEqual(argv[i:i + 4], [
                "--data-parallel-size-local", "4", "--data-parallel-address",
                "127.0.0.1"
            ])


if __name__ == "__main__":
    unittest.main()
