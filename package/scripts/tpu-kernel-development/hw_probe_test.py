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
"""Tests for hw_probe.py. Needs neither jax nor a TPU."""

import ast
import contextlib
import importlib
import io
import json
import os
import tempfile
import unittest

try:
    # Hyphenated package dir: not reachable via the `import` statement.
    hw_probe = importlib.import_module(
        "package.scripts.tpu-kernel-development.hw_probe")
except ImportError:
    import hw_probe

STAMP = "TPU_v6_lite/jax0.0.1-jaxlib0.0.1-libtpu0.0.2"

# What a job prints: the suite banner, one three-line report per probe, and
# the closing summary. Shaped exactly like harness.report/report_summary.
OUTPUT = """\
========================================================================
probe suite: 3 of 168 probes
modules: environment, partitioning
========================================================================

[environment] software_versions
    Q: Which jax, jaxlib and libtpu is this image on?
    PASS (0.01s): python=3.12.0 jax=0.0.1 jaxlib=0.0.1 libtpu=0.0.2

[environment] device_inventory
    Q: What devices does jax see, and what does it call this chip?
    PASS (0.20s): count=4 platform=tpu kinds=['TPU v6 lite']

[partitioning] dispatch_floor
    Q: What does one pallas_call cost?
    PASS (3.10s): per call 1.0 us vs an empty jit
      second line of detail
      . next_probe arm: region 300us under floor, retrying at batch=40

========================================================================
SUMMARY
========================================================================
  PASS         3
  PASS         [environment] software_versions
  PASS         [environment] device_inventory
  PASS         [partitioning] dispatch_floor
========================================================================
"""


def _without_report(output, name):
    """`output` as if the job had not run probe `name`."""
    start = output.index(f"\n[partitioning] {name}\n")
    end = output.index("\n\n=", start)
    return output[:start] + output[end:]


# What a job with nothing to measure prints: the stamp probes alone.
STAMP_ONLY_OUTPUT = _without_report(OUTPUT, "dispatch_floor").replace(
    "probe suite: 3 of", "probe suite: 2 of")


def _job_argv(job):
    """The argv a generated job script passes to the probe runner."""
    with open(job, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    return next(
        ast.literal_eval(node.value.args[0]) for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "run_probes.main")


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = hw_probe.main(argv)
    return code, out.getvalue(), err.getvalue()


class ParseTest(unittest.TestCase):

    def test_stamp_key_names_device_and_toolchain(self):
        self.assertEqual(hw_probe.stamp_key(OUTPUT), STAMP)

    def test_stamp_key_needs_both_stamp_probes(self):
        cut = OUTPUT.replace("[environment] device_inventory", "[x] other")
        self.assertIsNone(hw_probe.stamp_key(cut))

    def test_split_keeps_multiline_detail_and_stops_at_summary(self):
        blocks = hw_probe.split_by_probe(OUTPUT)
        self.assertEqual(
            sorted(blocks),
            ["device_inventory", "dispatch_floor", "software_versions"])
        self.assertTrue(blocks["dispatch_floor"].endswith("second line of "
                                                          "detail"))
        self.assertNotIn("SUMMARY", blocks["dispatch_floor"])
        self.assertNotIn("retrying", blocks["dispatch_floor"])
        self.assertEqual(hw_probe.verdict_of(blocks["dispatch_floor"]), "PASS")


class BuildTest(unittest.TestCase):

    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.job = os.path.join(self.work, "hw_probes", "job.py")

    def test_list_needs_no_device(self):
        code, out, _ = _run(["list", "--group", "partitioning"])
        self.assertEqual(code, 0)
        self.assertIn("dispatch_floor", out)
        self.assertNotIn("software_versions", out)

    def test_unknown_name_is_refused(self):
        code, _, err = _run([
            "build", "--pick", "dispatch_flor", "--work", self.work,
            "--output", self.job
        ])
        self.assertEqual(code, 2)
        self.assertIn("dispatch_flor", err)
        self.assertFalse(os.path.exists(self.job))

    def test_destructive_probe_is_refused(self):
        code, _, err = _run([
            "build", "--pick", "undrained_semaphore_halts_core", "--work",
            self.work, "--output", self.job
        ])
        self.assertEqual(code, 2)
        self.assertIn("destructive", err)
        self.assertFalse(os.path.exists(self.job))

    def test_job_is_self_contained_and_stamped(self):
        code, out, _ = _run([
            "build", "--pick", "dispatch_floor", "--work", self.work,
            "--output", self.job
        ])
        self.assertEqual(code, 0)
        self.assertIn("JOB: ", out)
        with open(self.job, encoding="utf-8") as f:
            script = f.read()
        tree = ast.parse(script)
        files = next(
            ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and node.targets[0].id == "FILES")
        self.assertEqual(sorted(files), [
            "checks/__init__.py", "checks/environment.py",
            "checks/partitioning.py", "harness.py", "run_probes.py"
        ])
        argv = next(
            ast.literal_eval(node.value.args[0]) for node in tree.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "run_probes.main")
        self.assertEqual(argv, [
            "--traceback", "--pick", "software_versions", "--pick",
            "device_inventory", "--pick", "dispatch_floor"
        ])


class RecordTest(unittest.TestCase):

    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.job = os.path.join(self.work, "hw_probes", "job.py")
        self.stdout = os.path.join(self.work, "hw_probes", "job.out")

    def _build(self, *picks):
        argv = ["build", "--work", self.work, "--output", self.job]
        for pick in picks:
            argv += ["--pick", pick]
        return _run(argv)

    def _record(self, output):
        os.makedirs(os.path.dirname(self.stdout), exist_ok=True)
        with open(self.stdout, "w", encoding="utf-8") as f:
            f.write(output)
        return _run(["record", "--stdout", self.stdout, "--work", self.work])

    def test_record_files_and_caches_each_answer(self):
        self._build("dispatch_floor")
        code, out, _ = self._record(OUTPUT)
        self.assertEqual(code, 0, out)
        answer = os.path.join(self.work, "hw_probes", "dispatch_floor.txt")
        with open(answer, encoding="utf-8") as f:
            text = f.read()
        self.assertTrue(text.startswith(f"environment: {STAMP}\n"))
        self.assertIn("per call 1.0 us", text)
        cached = os.path.join(self.work, "hw_probes", "cache", STAMP,
                              "dispatch_floor.txt")
        self.assertTrue(os.path.exists(cached))
        self.assertFalse(
            os.path.exists(os.path.join(self.work, "hw_probes",
                                        "pending.json")))

    def test_a_module_brings_the_check_modules_it_imports(self):
        # sparsecore_gaps borrows helpers from sparsecore; a job carrying only
        # the first fails before any probe runs.
        self.assertIn("checks.sparsecore",
                      hw_probe.with_dependencies({"checks.sparsecore_gaps"}))
        script = hw_probe.build_script(["sc_mpmd_vmem_scratch"],
                                       {"checks.sparsecore_gaps"}, False)
        self.assertIn("checks/sparsecore.py", script)

    def _answer(self, name):
        return os.path.join(self.work, "hw_probes", f"{name}.txt")

    def _cached(self, name, stamp=STAMP):
        return os.path.join(self.work, "hw_probes", "cache", stamp,
                            f"{name}.txt")

    def test_second_build_is_served_from_cache(self):
        self._build("dispatch_floor")
        self._record(OUTPUT)
        os.remove(self.job)
        code, out, _ = self._build("dispatch_floor")
        self.assertEqual(code, 0)
        self.assertIn("CACHED: dispatch_floor", out)
        self.assertTrue(os.path.exists(self._answer("dispatch_floor")))

    def test_all_cached_build_still_sends_the_stamp_probes(self):
        # Only the target knows which device and toolchain it is now, so even
        # a fully cached build sends a job: the stamp probes, nothing else.
        self._build("dispatch_floor")
        self._record(OUTPUT)
        os.remove(self.job)
        self._build("dispatch_floor")
        self.assertEqual(_job_argv(self.job), [
            "--traceback", "--pick", "software_versions", "--pick",
            "device_inventory"
        ])
        with open(os.path.join(self.work, "hw_probes", "pending.json"),
                  encoding="utf-8") as f:
            pending = json.load(f)
        self.assertEqual(pending["picks"], [])
        self.assertEqual(pending["served"], ["dispatch_floor"])
        self.assertEqual(pending["served_stamp"], STAMP)

    def test_same_target_confirms_the_served_answers(self):
        self._build("dispatch_floor")
        self._record(OUTPUT)
        self._build("dispatch_floor")
        code, out, err = self._record(STAMP_ONLY_OUTPUT)
        self.assertEqual(code, 0, out + err)
        self.assertNotIn("STALE", out)
        self.assertTrue(os.path.exists(self._answer("dispatch_floor")))
        self.assertFalse(
            os.path.exists(os.path.join(self.work, "hw_probes",
                                        "pending.json")))

    def test_answers_cached_on_another_target_are_stale(self):
        # The slot last saw a v6e; the target is now a v5e. The v6e answer is
        # served by build, then removed by record, and the next build
        # measures it on the v5e.
        self._build("dispatch_floor")
        self._record(OUTPUT)
        self._build("dispatch_floor")
        code, out, err = self._record(
            STAMP_ONLY_OUTPUT.replace("TPU v6 lite", "TPU v5 lite"))
        self.assertEqual(code, 1)
        self.assertIn("STALE: dispatch_floor", out)
        self.assertIn("Run build again", err)
        self.assertFalse(os.path.exists(self._answer("dispatch_floor")))
        code, out, _ = self._build("dispatch_floor")
        self.assertNotIn("CACHED", out)
        self.assertIn("dispatch_floor", _job_argv(self.job))

    def test_fail_and_invalid_are_filed_but_not_cached(self):
        for verdict in ("FAIL", "INVALID"):
            with self.subTest(verdict=verdict):
                self.setUp()
                self._build("dispatch_floor")
                code, out, _ = self._record(
                    OUTPUT.replace("PASS (3.10s)", f"{verdict} (3.10s)"))
                self.assertEqual(code, 0, out)
                self.assertIn(f"{verdict}: dispatch_floor", out)
                self.assertIn("(not cached)", out)
                self.assertTrue(os.path.exists(self._answer("dispatch_floor")))
                self.assertFalse(os.path.exists(
                    self._cached("dispatch_floor")))
                _, out, _ = self._build("dispatch_floor")
                self.assertNotIn("CACHED", out)

    def test_unsupported_is_cached(self):
        self._build("dispatch_floor")
        self._record(OUTPUT.replace("PASS (3.10s)", "UNSUPPORTED (3.10s)"))
        self.assertTrue(os.path.exists(self._cached("dispatch_floor")))

    def test_refresh_remeasures_one_cached_answer(self):
        self._build("dispatch_floor")
        self._record(OUTPUT)
        code, out, _ = self._run_build(["--refresh", "dispatch_floor"])
        self.assertEqual(code, 0)
        self.assertNotIn("CACHED", out)
        self.assertIn("dispatch_floor", _job_argv(self.job))

    def test_refreshed_probe_that_now_fails_drops_the_old_answer(self):
        self._build("dispatch_floor")
        self._record(OUTPUT)
        self._run_build(["--refresh", "dispatch_floor"])
        self._record(OUTPUT.replace("PASS (3.10s)", "FAIL (3.10s)"))
        self.assertFalse(os.path.exists(self._cached("dispatch_floor")))

    def test_build_needs_a_probe(self):
        code, _, err = self._run_build([])
        self.assertEqual(code, 2)
        self.assertIn("--pick or --refresh", err)

    def _run_build(self, extra):
        return _run(["build", "--work", self.work, "--output", self.job] +
                    extra)

    def test_partial_cache_runs_only_the_rest(self):
        self._build("dispatch_floor")
        self._record(OUTPUT)
        code, out, _ = self._build("dispatch_floor", "hbm_bandwidth_arms")
        self.assertEqual(code, 0)
        self.assertIn("CACHED: dispatch_floor", out)
        with open(os.path.join(self.work, "hw_probes", "pending.json"),
                  encoding="utf-8") as f:
            self.assertEqual(json.load(f)["picks"], ["hbm_bandwidth_arms"])

    def test_missing_answer_fails_and_keeps_pending(self):
        self._build("dispatch_floor", "hbm_bandwidth_arms")
        code, _, err = self._record(OUTPUT)
        self.assertEqual(code, 1)
        self.assertIn("hbm_bandwidth_arms", err)
        self.assertTrue(
            os.path.exists(os.path.join(self.work, "hw_probes",
                                        "pending.json")))

    def test_job_that_ran_nothing_records_nothing(self):
        self._build("dispatch_floor")
        code, _, err = self._record("Traceback: ImportError: no jax\n")
        self.assertEqual(code, 1)
        self.assertIn("did not run any probe", err)
        self.assertFalse(
            os.path.exists(
                os.path.join(self.work, "hw_probes", "dispatch_floor.txt")))

    def test_unstamped_output_records_nothing(self):
        self._build("dispatch_floor")
        unstamped = OUTPUT.replace("[environment] device_inventory",
                                   "[environment] other")
        code, _, err = self._record(unstamped)
        self.assertEqual(code, 1)
        self.assertIn("cannot be tied to a device", err)
        self.assertFalse(
            os.path.exists(
                os.path.join(self.work, "hw_probes", "dispatch_floor.txt")))
        self.assertFalse(
            os.path.exists(os.path.join(self.work, "hw_probes", "cache")))


if __name__ == "__main__":
    unittest.main()
