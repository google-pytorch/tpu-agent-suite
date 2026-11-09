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
"""Tests for serve_sglang.sh's SGLang timeout defaults.

The script runs for real, against a stub `python3` beside it: the stub's
`-m sglang.launch_server` writes the arguments and environment it was started
with to a file.
"""

import os
import shutil
import socket
import subprocess
import tempfile
import unittest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "serve_sglang.sh")
TIMEOUT_ENVS = ("SGLANG_WARMUP_TIMEOUT", "SGLANG_WATCHDOG_TIMEOUT_S",
                "SGLANG_DIST_TIMEOUT_S")
ALL_FIELDS = "watchdog_timeout dist_timeout"

# Answers the ServerArgs probe with the fields in STUB_FIELDS, records the
# launch_server call, and idles as the profile file server.
STUB_PYTHON = """#!/bin/bash
case "$*" in
  *http.server*) exec sleep 30 ;;
  *sglang.launch_server*)
    shift 2
    {
      for arg in "$@"; do printf 'ARG %s\\n' "$arg"; done
      env | grep -E '^SGLANG_WARMUP_TIMEOUT=' | sed 's/^/ENV /'
    } > "$STUB_OUT"
    ;;
  *ServerArgs*) echo "$STUB_FIELDS" ;;
  *) exit 0 ;;
esac
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServeSglangTimeoutsTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        bin_dir = os.path.join(self.dir, "bin")
        os.makedirs(bin_dir)
        self.python = os.path.join(bin_dir, "python3")
        with open(self.python, "w", encoding="utf-8") as f:
            f.write(STUB_PYTHON)
        os.chmod(self.python, 0o755)
        self.out = os.path.join(self.dir, "sglang.out")

    def _serve(self, *args, fields=ALL_FIELDS, **env):
        """Run the script; return (launch_server args, env we set, stderr)."""
        full_env = {
            k: v
            for k, v in os.environ.items() if k not in TIMEOUT_ENVS
        }
        full_env.update({
            "TPU_AGENT_SUITE_PYTHON": self.python,
            "SGLANG_PROFILE_DIR": os.path.join(self.dir, "profile"),
            "SGLANG_PROFILE_FILES_PORT": str(_free_port()),
            "STUB_OUT": self.out,
            "STUB_FIELDS": fields,
        })
        full_env.update(env)
        proc = subprocess.run(["bash", SCRIPT, "/model", *args],
                              cwd=self.dir,
                              env=full_env,
                              capture_output=True,
                              text=True,
                              timeout=60,
                              check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(self.out, encoding="utf-8") as f:
            lines = f.read().splitlines()
        argv = [line[4:] for line in lines if line.startswith("ARG ")]
        envs = dict(line[4:].split("=", 1) for line in lines
                    if line.startswith("ENV "))
        return argv, envs, proc.stderr

    def test_defaults_are_tpu_sized(self):
        argv, envs, stderr = self._serve("--port", "8000")
        self.assertEqual(envs, {"SGLANG_WARMUP_TIMEOUT": "3600"})
        self.assertEqual(argv[:4],
                         ["--model-path", "/model", "--port", "8000"])
        i = argv.index("--watchdog-timeout")
        self.assertEqual(argv[i + 1], "3600")
        i = argv.index("--dist-timeout")
        self.assertEqual(argv[i + 1], "1800")
        self.assertIn(
            "serve_sglang: timeouts warmup=3600s watchdog=3600s dist=1800s",
            stderr)

    def test_caller_values_are_kept(self):
        argv, envs, _ = self._serve(SGLANG_WARMUP_TIMEOUT="9600",
                                    SGLANG_WATCHDOG_TIMEOUT_S="7200",
                                    SGLANG_DIST_TIMEOUT_S="2400")
        self.assertEqual(envs["SGLANG_WARMUP_TIMEOUT"], "9600")
        i = argv.index("--watchdog-timeout")
        self.assertEqual(argv[i + 1], "7200")
        i = argv.index("--dist-timeout")
        self.assertEqual(argv[i + 1], "2400")

    def test_caller_flag_is_not_duplicated(self):
        for flags in (["--watchdog-timeout", "600", "--dist-timeout", "900"],
                      ["--watchdog-timeout=600", "--dist-timeout=900"]):
            with self.subTest(flags=flags):
                argv, _, stderr = self._serve(*flags)
                for flag in ("--watchdog-timeout", "--dist-timeout"):
                    self.assertEqual(sum(a.startswith(flag) for a in argv), 1)
                self.assertNotIn("3600", argv)
                self.assertNotIn("1800", argv)
                self.assertIn("watchdog=caller's dist=caller's", stderr)

    def test_flags_left_out_where_sglang_lacks_them(self):
        argv, envs, stderr = self._serve(fields="")
        self.assertNotIn("--watchdog-timeout", argv)
        self.assertNotIn("--dist-timeout", argv)
        self.assertEqual(envs["SGLANG_WARMUP_TIMEOUT"], "3600")
        self.assertIn("no --watchdog-timeout in this SGLang", stderr)
        self.assertIn("no --dist-timeout in this SGLang", stderr)
        argv, _, _ = self._serve(fields="dist_timeout")
        self.assertNotIn("--watchdog-timeout", argv)
        self.assertIn("--dist-timeout", argv)


if __name__ == "__main__":
    unittest.main()
