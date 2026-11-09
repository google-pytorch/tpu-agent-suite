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
"""Tests for onboarding_preflight.py, with each step a stub that records how
it was called."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))

# Run as a script, records its argv and environment, prints
# STUB_<NAME>_SAY, writes the file after --output or --plan-json unless
# STUB_<NAME>_RC is set, then exits that code. Imported (chipless_tpu, from
# the sitecustomize), it does nothing.
STUB = '''import json, os, sys
if __name__ == "__main__":
    name = os.path.basename(sys.argv[0]).split(".")[0]
    with open("calls.jsonl", "a") as f:
        f.write(json.dumps({"step": name, "argv": sys.argv[1:],
            "python": os.environ.get("TPU_AGENT_SUITE_PYTHON"),
            "devices": os.environ.get("CHIPLESS_TPU_DEVICES"),
            "pythonpath": os.environ.get("PYTHONPATH", "")}) + "\\n")
    print(os.environ.get("STUB_" + name.upper() + "_SAY", name + " ran"))
    rc = int(os.environ.get("STUB_" + name.upper() + "_RC", "0"))
    for flag in ("--output", "--plan-json"):
        if flag in sys.argv and not rc:
            with open(sys.argv[sys.argv.index(flag) + 1], "w") as f:
                f.write('{"ok": true}')
    sys.exit(rc)
'''


class OnboardingPreflightTest(unittest.TestCase):

    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.work)
        for name in ("image_probe.py", "preflight_check.py",
                     "engine_compatibility_check.py", "vllm_flags.py",
                     "sglang_flags.py", "chipless_tpu.py"):
            with open(os.path.join(self.work, name), "w",
                      encoding="utf-8") as f:
                f.write(STUB)
        with open(os.path.join(self.work, "discover_vllm_flags.sh"),
                  "w",
                  encoding="utf-8") as f:
            f.write(f'"{sys.executable}" vllm_flags.py "$@" || exit $?\n'
                    'echo "--tensor-parallel-size" > "$1"\n')
        with open(os.path.join(self.work, "discover_sglang_flags.sh"),
                  "w",
                  encoding="utf-8") as f:
            f.write(f'"{sys.executable}" sglang_flags.py "$@" || exit $?\n'
                    'echo "--tp-size" > "$1"\n')

    def _run(self, *args, chipless=None, exit_code=0, **stubs):
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("STUB_", "TPU_AGENT_SUITE_CHIPLESS"))
        }
        env.update({f"STUB_{k.upper()}": str(v) for k, v in stubs.items()})
        if chipless:
            env["TPU_AGENT_SUITE_CHIPLESS_TPU"] = chipless
        proc = subprocess.run([
            sys.executable,
            os.path.join(HERE, "onboarding_preflight.py"), *args
        ],
                              cwd=self.work,
                              env=env,
                              capture_output=True,
                              check=False)
        self.assertEqual(proc.returncode, exit_code, proc.stdout)
        calls = {}
        if os.path.exists(os.path.join(self.work, "calls.jsonl")):
            with open(os.path.join(self.work, "calls.jsonl"),
                      encoding="utf-8") as f:
                for line in f:
                    call = json.loads(line)
                    calls[call["step"]] = call
        if exit_code:
            return None, calls
        return self._json("preflight.json"), calls

    def _json(self, name):
        with open(os.path.join(self.work, name), encoding="utf-8") as f:
            return json.load(f)

    def test_without_devices_only_the_probe_and_full_check_run(self):
        report, calls = self._run("--model", "org/m", "--mode", "benchmark")
        self.assertEqual(sorted(calls), ["image_probe", "preflight_check"])
        self.assertIn("--touch_device", calls["image_probe"]["argv"])
        self.assertEqual(
            calls["preflight_check"]["argv"],
            ["--model=org/m", "--allow_gpu_imports", "--mode=benchmark"])
        self.assertEqual(
            {
                name: step["exit_code"]
                for name, step in report.items()
            }, {
                "chipless": None,
                "image_probe": 0,
                "full_check": 0,
                "engine_compatibility_check": None,
                "discover_vllm_flags": None
            })
        with open(os.path.join(self.work, "full_check.log"),
                  encoding="utf-8") as f:
            self.assertIn("preflight_check ran", f.read())

    def test_with_devices_every_step_runs_with_its_arguments(self):
        report, calls = self._run("--model", "org/m", "--mode", "inference",
                                  "--devices", "4", "--hbm-gb-per-device",
                                  "32", "--tp", "2", "--vllm-patch", "p.py")
        self.assertEqual(calls["engine_compatibility_check"]["argv"], [
            "--plan-json", "tp_plan.json", "--model", "org/m", "--devices",
            "4", "--hbm-gb-per-device", "32", "--tp", "2", "--vllm-patch",
            "p.py"
        ])
        self.assertTrue(
            all(step["exit_code"] == 0 for name, step in report.items()
                if name != "chipless"))
        # The steps' scripts are told the Python this ran on.
        self.assertEqual(calls["vllm_flags"]["python"], sys.executable)

    def test_points_reach_the_engine_check(self):
        _, calls = self._run("--model", "org/m", "--mode", "inference",
                             "--devices", "8", "--hbm-gb-per-device", "94",
                             "--points", "1024:1024:64,8192:1024:128")
        argv = calls["engine_compatibility_check"]["argv"]
        self.assertEqual(argv[-2:], ["--points", "1024:1024:64,8192:1024:128"])

    def test_sglang_engine_passes_engine_and_discovers_sglang_flags(self):
        report, calls = self._run("--model", "org/m", "--mode", "inference",
                                  "--engine", "sglang", "--trust-remote-code",
                                  "--devices", "8", "--hbm-gb-per-device",
                                  "96")
        self.assertEqual(calls["engine_compatibility_check"]["argv"], [
            "--plan-json", "tp_plan.json", "--model", "org/m", "--devices",
            "8", "--hbm-gb-per-device", "96", "--engine", "sglang",
            "--trust-remote-code"
        ])
        self.assertIn("sglang_flags", calls)
        self.assertNotIn("vllm_flags", calls)
        self.assertEqual(calls["sglang_flags"]["argv"], ["sglang_flags.txt"])
        self.assertEqual(report["discover_sglang_flags"]["exit_code"], 0)
        self.assertNotIn("discover_vllm_flags", report)
        with open(os.path.join(self.work, "sglang_flags.txt"),
                  encoding="utf-8") as f:
            self.assertIn("--tp-size", f.read())

    def test_a_failed_step_still_leaves_every_file(self):
        report, calls = self._run(
            "--model",
            "org/m",
            "--mode",
            "inference",
            "--devices",
            "1",
            "--hbm-gb-per-device",
            "16",
            image_probe_rc=1,
            engine_compatibility_check_rc=1,
            engine_compatibility_check_say=(
                "PRE-FLIGHT FAILED: TorchTPU Pallas Kernel Compatibility Error"
                "\n     Reason: no kernel for ALiBi"),
            vllm_flags_rc=2)
        # A failed step does not stop the next.
        self.assertIn("vllm_flags", calls)
        self.assertEqual(report["engine_compatibility_check"]["exit_code"], 1)
        self.assertIn("     Reason: no kernel for ALiBi",
                      report["engine_compatibility_check"]["tail"])
        plan = self._json("tp_plan.json")
        self.assertFalse(plan["ok"])
        self.assertIn("Reason: no kernel for ALiBi", plan["reason"])
        self.assertIn("error", self._json("image_probe.json"))
        with open(os.path.join(self.work, "vllm_flags.txt"),
                  encoding="utf-8") as f:
            self.assertTrue(f.read().startswith("# flag discovery exited 2"))

    def test_on_a_cpu_node_torch_tpu_is_told_the_profiles_tpu(self):
        """With the generation compute sets in a --cpu-requests job, the
        self-test runs first, every later step loads the shim, and the probe
        leaves the device alone."""
        report, calls = self._run("--model",
                                  "org/m",
                                  "--mode",
                                  "inference",
                                  "--devices",
                                  "4",
                                  "--hbm-gb-per-device",
                                  "32",
                                  chipless="v6e")
        self.assertEqual(report["chipless"]["exit_code"], 0)
        self.assertEqual(calls["chipless_tpu"]["argv"], ["--check"])
        self.assertNotIn("--touch_device", calls["image_probe"]["argv"])
        shim = os.path.join(os.path.realpath(self.work), ".chipless")
        for step in ("image_probe", "preflight_check",
                     "engine_compatibility_check", "vllm_flags"):
            self.assertEqual(calls[step]["devices"], "4")
            self.assertEqual(
                os.path.realpath(calls[step]["pythonpath"].split(
                    os.pathsep)[0]), shim)

    def test_a_failed_self_test_asks_for_a_tpu_node(self):
        """Nothing else runs, and the exit is the one gke_job reruns on a TPU
        node for, so a check without the device never reads as the model's
        failure."""
        _, calls = self._run("--model",
                             "org/m",
                             "--mode",
                             "inference",
                             chipless="v6e",
                             exit_code=75,
                             chipless_tpu_rc=3)
        self.assertEqual(sorted(calls), ["chipless_tpu"])


if __name__ == "__main__":
    unittest.main()
