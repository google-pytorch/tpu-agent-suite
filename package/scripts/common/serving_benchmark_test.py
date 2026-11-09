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
"""Unit tests for serving_benchmark.py, against a fake server and fake client."""

from __future__ import annotations

import contextlib
import http.server
import importlib.util
import io
import json
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
import threading
import unittest
import unittest.mock

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent

try:
    from . import serving_benchmark  # noqa: E402
except ImportError:
    _spec = importlib.util.spec_from_file_location(
        "serving_benchmark", _SCRIPTS_DIR / "serving_benchmark.py")
    serving_benchmark = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(serving_benchmark)

# A vllm bench serve result, trimmed to the keys the summary reads.
RAW_RESULT = {
    "duration": 7.0,
    "completed": 10,
    "failed": 0,
    "total_input_tokens": 5120,
    "total_output_tokens": 1280,
    "request_throughput": 1.4,
    "output_throughput": 180.0,
    "total_token_throughput": 900.0,
    "mean_ttft_ms": 20.0,
    "median_ttft_ms": 18.0,
    "p50_ttft_ms": 18.0,
    "p90_ttft_ms": 21.0,
    "p99_ttft_ms": 39.0,
    "mean_tpot_ms": 5.0,
    "median_tpot_ms": 5.0,
    "p99_tpot_ms": 5.5,
    "mean_itl_ms": 5.0,
    "median_itl_ms": 5.0,
    "p99_itl_ms": 5.7,
    "mean_e2el_ms": 700.0,
    "median_e2el_ms": 700.0,
    "p99_e2el_ms": 720.0,
    "errors": [""] * 10,
}

# Stands in for the vllm CLI: writes RAW_RESULT (with an override from the
# FAKE_RESULT_OVERRIDE file next to it, when present) where --result-dir and
# --result-filename say, and records its argv. Hangs when a "hang" file exists.
FAKE_VLLM = r'''
import json, pathlib, sys, time
argv = sys.argv[1:]
here = pathlib.Path(__file__).parent
with open(here / "calls.jsonl", "a") as f:
    f.write(json.dumps(argv) + "\n")
if (here / "hang").exists():
    time.sleep(60)
result = json.loads((here / "raw_result.json").read_text())
override = here / "override.json"
if override.exists():
    result.update(json.loads(override.read_text()))
out = pathlib.Path(argv[argv.index("--result-dir") + 1])
name = argv[argv.index("--result-filename") + 1]
by_name = here / "overrides_by_file.json"
if by_name.exists():
    mapping = json.loads(by_name.read_text())
    if name in mapping:
        result.update(mapping[name])
(out / name).write_text(json.dumps(result))
'''

SGLANG_RAW_RESULT = {
    "duration": 7.0,
    "completed": 10,
    "num_prompts": 10,
    "total_input_tokens": 5120,
    "total_output_tokens": 1280,
    "request_throughput": 1.4,
    "output_throughput": 180.0,
    "total_throughput": 900.0,
    "mean_ttft_ms": 20.0,
    "median_ttft_ms": 18.0,
    "p99_ttft_ms": 39.0,
    "mean_tpot_ms": 5.0,
    "median_tpot_ms": 5.0,
    "p99_tpot_ms": 5.5,
    "mean_itl_ms": 5.0,
    "median_itl_ms": 5.0,
    "p99_itl_ms": 5.7,
    "mean_e2e_latency_ms": 700.0,
    "median_e2e_latency_ms": 700.0,
    "p99_e2e_latency_ms": 720.0,
}

FAKE_SGLANG = r'''
import json, pathlib, sys
argv = sys.argv[1:]
here = pathlib.Path(__file__).parent
with open(here / "calls.jsonl", "a") as f:
    f.write(json.dumps(argv) + "\n")
result = json.loads((here / "sglang_raw_result.json").read_text())
override = here / "override.json"
if override.exists():
    result.update(json.loads(override.read_text()))
out_file = pathlib.Path(argv[argv.index("--output-file") + 1])
out_file.write_text(json.dumps(result) + "\n")
'''


class _FakeServer(http.server.BaseHTTPRequestHandler):
    models = {
        "object":
        "list",
        "data": [{
            "id": "org/model",
            "root": "hub-org/model",
            "max_model_len": 2048,
        }],
    }
    server_info = None

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            body = b""
        elif self.path == "/v1/models":
            body = json.dumps(self.models).encode()
        elif self.path == "/get_server_info" and self.server_info is not None:
            body = json.dumps(self.server_info).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        # Answers only when the test enabled it, as vLLM registers the route
        # only under VLLM_SERVER_DEV_MODE=1 and SGLang registers /flush_cache.
        if (self.path == "/reset_prefix_cache"
                and self.server.resets is not None):
            self.server.resets += 1
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"success": True}).encode())
            return
        if (self.path == "/flush_cache"
                and getattr(self.server, "flushes", None) is not None):
            self.server.flushes += 1
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Cache flushed.\n")
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args):
        pass


# Stand-ins for torch and the vllm CLI, as packages, for --as-vllm-cli. The
# vllm CLI runs the script named by FAKE_VLLM_SCRIPT when set; otherwise it
# prints what the process looked like when it ran.
FAKE_TORCH = r'''
import os, types
autoload_at_import = os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD")
renamed = []
utils = types.SimpleNamespace(rename_privateuse1_backend=renamed.append)
'''

FAKE_VLLM_CLI = r'''
import json, os, runpy, sys
def main():
    script = os.environ.get("FAKE_VLLM_SCRIPT")
    if script:
        runpy.run_path(script, run_name="__main__")
        return 0
    import torch
    blocked = {}
    for name in ("libtpu", "torch_tpu"):
        try:
            __import__(name)
            blocked[name] = False
        except ImportError:
            blocked[name] = True
    print(json.dumps({
        "argv": sys.argv,
        "autoload_at_torch_import": torch.autoload_at_import,
        "vllm_plugins": os.environ.get("VLLM_PLUGINS"),
        "renamed": torch.renamed,
        "blocked": blocked,
    }))
    return 0
'''


def _write_fake_packages(root: pathlib.Path) -> None:
    (root / "torch").mkdir()
    (root / "torch" / "__init__.py").write_text(FAKE_TORCH, encoding="utf-8")
    cli = root / "vllm" / "entrypoints" / "cli"
    cli.mkdir(parents=True)
    for package in (root / "vllm", root / "vllm" / "entrypoints", cli):
        (package / "__init__.py").write_text("", encoding="utf-8")
    (cli / "main.py").write_text(FAKE_VLLM_CLI, encoding="utf-8")


class TestAsVllmCli(unittest.TestCase):

    def test_keeps_the_tpu_stack_out_whatever_the_caller_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_fake_packages(pathlib.Path(tmp))
            env = dict(os.environ,
                       PYTHONPATH=tmp,
                       TORCH_DEVICE_BACKEND_AUTOLOAD="1",
                       VLLM_PLUGINS="torchtpu")
            proc = subprocess.run([
                sys.executable, serving_benchmark.__file__, "--as-vllm-cli",
                "bench", "serve", "--seed", "7"
            ],
                                  capture_output=True,
                                  text=True,
                                  env=env,
                                  check=True)
        seen = json.loads(proc.stdout)
        self.assertEqual(seen["argv"],
                         ["vllm", "bench", "serve", "--seed", "7"])
        self.assertEqual(seen["autoload_at_torch_import"], "0")
        self.assertEqual(seen["vllm_plugins"], "")
        self.assertEqual(seen["renamed"], ["tpu"])
        self.assertEqual(seen["blocked"], {"libtpu": True, "torch_tpu": True})


class TestPureFunctions(unittest.TestCase):

    def test_parse_cells(self):
        self.assertEqual(serving_benchmark.parse_cells("1024:128, 64:8"),
                         [(1024, 128), (64, 8)])
        with self.assertRaises(ValueError):
            serving_benchmark.parse_cells("1024")

    def test_parse_ints_rejects_zero(self):
        self.assertEqual(serving_benchmark.parse_ints("1,4,16"), [1, 4, 16])
        with self.assertRaises(ValueError):
            serving_benchmark.parse_ints("0")

    def test_normalize_maps_to_protocol_names(self):
        m = serving_benchmark.normalize(RAW_RESULT)
        self.assertEqual(m["ttft_p50_ms"], 18.0)
        self.assertEqual(m["ttft_p99_ms"], 39.0)
        self.assertEqual(m["ttft_mean_ms"], 20.0)
        self.assertEqual(m["itl_p99_ms"], 5.7)
        self.assertEqual(m["output_tokens_per_s"], 180.0)
        self.assertEqual(m["total_tokens_per_s"], 900.0)
        self.assertEqual(m["requests_per_s"], 1.4)
        self.assertAlmostEqual(m["output_tokens_per_s_per_user"], 200.0)

    def test_normalize_maps_sglang_keys(self):
        m = serving_benchmark.normalize(SGLANG_RAW_RESULT)
        self.assertEqual(m["total_tokens_per_s"], 900.0)
        self.assertEqual(m["e2el_mean_ms"], 700.0)
        self.assertEqual(m["e2el_p50_ms"], 700.0)
        self.assertEqual(m["e2el_p99_ms"], 720.0)
        self.assertEqual(m["failed"], 0)
        self.assertAlmostEqual(m["output_tokens_per_s_per_user"], 200.0)

    def test_pareto_frontier_and_reproduction(self):
        points = [
            {
                "concurrency": 1,
                "output_tokens_per_s": 70.0,
                "output_tokens_per_s_per_user": 70.0,
            },
            {
                "concurrency": 4,
                "output_tokens_per_s": 212.0,
                "output_tokens_per_s_per_user": 53.0,
            },
            {
                "concurrency": 8,
                "output_tokens_per_s": 200.0,
                "output_tokens_per_s_per_user": 25.0,
            },
            {
                "concurrency": 16,
                "output_tokens_per_s": 511.0,
                "output_tokens_per_s_per_user": 32.0,
            },
        ]
        frontier = serving_benchmark.compute_pareto_frontier(points)
        # Concurrency 8 is dominated by concurrency 16 (511 > 200 and 32 > 25)
        self.assertEqual([p["concurrency"] for p in frontier], [1, 4, 16])

        curves = {"8192:1024": {"points": points, "pareto_frontier": frontier}}
        repro_ok = serving_benchmark.evaluate_pareto_reproduction(
            curves, {"curves": curves}, tolerance_pct=5.0)
        self.assertTrue(repro_ok["consistent"])
        self.assertTrue(repro_ok["promote_to_bkc"])

        # Drop concurrency 16 throughput by 20% -> fails reproduction check
        degraded_points = [
            dict(p, output_tokens_per_s=400.0) if p["concurrency"] == 16 else p
            for p in points
        ]
        degraded_curves = {"8192:1024": {"points": degraded_points}}
        repro_fail = serving_benchmark.evaluate_pareto_reproduction(
            degraded_curves, {"curves": curves}, tolerance_pct=5.0)
        self.assertFalse(repro_fail["consistent"])
        self.assertFalse(repro_fail["promote_to_bkc"])

    def test_aggregate_reports_spread(self):
        agg = serving_benchmark.aggregate([{"x": 100.0}, {"x": 110.0}])
        self.assertAlmostEqual(agg["x"]["mean"], 105.0)
        self.assertEqual(agg["x"]["values"], [100.0, 110.0])
        self.assertGreater(agg["x"]["cv_pct"], 0)
        single = serving_benchmark.aggregate([{"x": 1.0}])
        self.assertEqual(single["x"]["stdev"], 0.0)

    def test_failure_reason(self):
        self.assertIsNone(serving_benchmark.failure_reason(RAW_RESULT, 0.0))
        none_done = dict(RAW_RESULT,
                         completed=0,
                         failed=10,
                         errors=["connection refused"] * 10)
        self.assertIn("connection refused",
                      serving_benchmark.failure_reason(none_done, 0.0))
        some_failed = dict(RAW_RESULT, completed=9, failed=1)
        self.assertIsNotNone(serving_benchmark.failure_reason(some_failed, 0))
        self.assertIsNone(serving_benchmark.failure_reason(some_failed, 20))

    def test_base_url_strips_v1(self):
        args = serving_benchmark.parse_args(
            ["--output-dir", "/tmp/x", "--base-url", "http://h:1/v1/"])
        self.assertEqual(args.base_url, "http://h:1")

    def test_vllm_runs_as_this_script_by_default(self):
        with unittest.mock.patch.object(serving_benchmark.importlib.util,
                                        "find_spec",
                                        return_value=object()):
            cmd = serving_benchmark.resolve_vllm_cmd(None)
        self.assertEqual(cmd, [
            sys.executable,
            str(pathlib.Path(serving_benchmark.__file__).resolve()),
            "--as-vllm-cli"
        ])

    def test_vllm_without_vllm_here_warns(self):
        stderr = io.StringIO()
        with unittest.mock.patch.object(serving_benchmark.importlib.util,
                                        "find_spec",
                                        return_value=None), \
                unittest.mock.patch.object(serving_benchmark.shutil, "which",
                                           return_value="/usr/bin/vllm"), \
                contextlib.redirect_stderr(stderr):
            cmd = serving_benchmark.resolve_vllm_cmd(None)
        self.assertEqual(cmd, ["/usr/bin/vllm"])
        self.assertIn("has no vllm", stderr.getvalue())
        self.assertIn("without TPU chips", stderr.getvalue())

    def test_vllm_cmd_given_is_used_as_is(self):
        self.assertEqual(serving_benchmark.resolve_vllm_cmd("vllm"), ["vllm"])

    def test_parse_avoid_rules_and_filter_c_range(self):
        rules = serving_benchmark.parse_avoid_rules("8192:*:64, *:1024:32")
        self.assertEqual(
            serving_benchmark.avoid_rule_for(8192, 128, 64, rules),
            "8192:*:64")
        self.assertEqual(
            serving_benchmark.avoid_rule_for(1024, 1024, 32, rules),
            "*:1024:32")
        self.assertIsNone(
            serving_benchmark.avoid_rule_for(1024, 128, 16, rules))

        kept, skipped = serving_benchmark.filter_c_range(
            8192,
            128,
            [1, 16, 32, 64, 128],
            max_model_len=16384,
            max_concurrency_limit=64,
            avoid_rules=rules,
        )
        self.assertEqual(kept, [1, 16, 32])
        self.assertEqual([s["concurrency"] for s in skipped], [64, 128])
        self.assertIn("--avoid rule 8192:*:64", skipped[0]["error"])
        self.assertIn("max_concurrency_limit 64", skipped[1]["error"])

    def test_check_early_stop(self):

        def _cell(conc: int, tps: float) -> dict[str, object]:
            return {
                "concurrency": conc,
                "status": "ok",
                "metrics": {
                    "output_tokens_per_s": {
                        "mean": tps
                    }
                },
            }

        c1, c2 = _cell(1, 100.0), _cell(2, 105.0)
        # One low-gain step does not stop; a later big gain resets the streak.
        self.assertIsNone(
            serving_benchmark.check_early_stop([c1],
                                               c2,
                                               stop_below_gain_pct=10.0))
        self.assertIsNone(
            serving_benchmark.check_early_stop([c1, c2],
                                               _cell(4, 130.0),
                                               stop_below_gain_pct=10.0))
        # Two consecutive low-gain steps vs best throughput trigger stop.
        err = serving_benchmark.check_early_stop([c1, c2],
                                                 _cell(4, 108.0),
                                                 stop_below_gain_pct=10.0)
        self.assertIn("throughput saturated at c=4", err)
        # Stop-on-failure is opt-in.
        failed = {"concurrency": 8, "status": "failed"}
        self.assertIsNone(serving_benchmark.check_early_stop([c1], failed))
        fail_err = serving_benchmark.check_early_stop([c1],
                                                      failed,
                                                      stop_on_failure=True)
        self.assertIn("earlier concurrency c=8 failed", fail_err)


class TestEndToEnd(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        (self.dir / "fake_vllm.py").write_text(FAKE_VLLM, encoding="utf-8")
        (self.dir / "fake_sglang.py").write_text(FAKE_SGLANG, encoding="utf-8")
        (self.dir / "raw_result.json").write_text(json.dumps(RAW_RESULT),
                                                  encoding="utf-8")
        (self.dir / "sglang_raw_result.json").write_text(
            json.dumps(SGLANG_RAW_RESULT), encoding="utf-8")
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                                      _FakeServer)
        self.server.resets = None
        self.server.flushes = None
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.out = self.dir / "out"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def _main(self, *extra: str) -> int:
        vllm_cmd = shlex.join([sys.executable, str(self.dir / "fake_vllm.py")])
        return serving_benchmark.main([
            "--base-url", self.base_url, "--output-dir",
            str(self.out), "--vllm-cmd", vllm_cmd, "--ready-timeout", "5",
            *extra
        ])

    def _calls(self) -> list[list[str]]:
        path = self.dir / "calls.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_sweep_writes_summary_with_discovered_model(self):
        rc = self._main("--isl-osl", "512:128", "--concurrency", "1,4",
                        "--replicates", "2")
        self.assertEqual(rc, serving_benchmark.EXIT_OK)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["model"]["id"], "org/model")
        self.assertEqual([c["cell"] for c in summary["cells"]],
                         ["isl512_osl128_c1", "isl512_osl128_c4"])
        cell = summary["cells"][1]
        self.assertEqual(len(cell["replicates"]), 2)
        self.assertEqual(cell["metrics"]["ttft_p50_ms"]["mean"], 18.0)
        self.assertTrue((self.out / "summary.md").exists())

        calls = self._calls()
        self.assertEqual(len(calls), 4)
        argv = calls[-1]
        self.assertEqual(argv[:2], ["bench", "serve"])
        self.assertEqual(argv[argv.index("--model") + 1], "org/model")
        # The tokenizer is whatever the server loaded, so no model is named.
        self.assertEqual(argv[argv.index("--tokenizer") + 1], "hub-org/model")
        self.assertEqual(argv[argv.index("--max-concurrency") + 1], "4")
        self.assertEqual(argv[argv.index("--num-prompts") + 1], "40")
        self.assertIn("--ignore-eos", argv)
        self.assertEqual(argv[argv.index("--temperature") + 1], "0.0")
        # Without a cache reset, no run resends an earlier run's prompts.
        seeds = [c[c.index("--seed") + 1] for c in calls]
        self.assertEqual(seeds, ["42", "43", "44", "45"])
        self.assertEqual(cell["seeds"], [44, 45])
        self.assertEqual(summary["settings"]["prefix_cache"], "seed per run")

    def test_a_sweep_cut_short_keeps_the_cells_it_finished(self):
        """summary.json is replaced after each cell, as `partial`."""
        real = serving_benchmark.run_cell
        done = []

        def lost_after_one(*args, **kwargs):
            if done:
                raise KeyboardInterrupt  # The job is stopped mid-sweep.
            done.append(1)
            return real(*args, **kwargs)

        with unittest.mock.patch.object(serving_benchmark, "run_cell",
                                        lost_after_one), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                self._main("--isl-osl", "512:128", "--concurrency", "1,4")
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["status"], "partial")
        self.assertEqual([c["cell"] for c in summary["cells"]],
                         ["isl512_osl128_c1"])
        self.assertIn("isl512_osl128_c1",
                      (self.out / "summary.md").read_text())
        self.assertFalse(list(self.out.glob(".*.tmp")))

    def test_sweep_runs_vllm_as_this_script_by_default(self):
        packages = self.dir / "packages"
        packages.mkdir()
        _write_fake_packages(packages)
        env = {
            "PYTHONPATH": str(packages),
            "FAKE_VLLM_SCRIPT": str(self.dir / "fake_vllm.py"),
        }
        with unittest.mock.patch.dict(os.environ, env), \
                unittest.mock.patch.object(serving_benchmark.importlib.util,
                                           "find_spec",
                                           return_value=object()):
            rc = serving_benchmark.main([
                "--base-url", self.base_url, "--output-dir",
                str(self.out), "--ready-timeout", "5", "--isl-osl", "512:128"
            ])
        self.assertEqual(rc, serving_benchmark.EXIT_OK)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["status"], "ok")
        self.assertIn("--as-vllm-cli", summary["cells"][0]["command"])
        self.assertEqual(self._calls()[0][:2], ["bench", "serve"])

    def test_prefix_cache_reset_before_every_run_when_available(self):
        self.server.resets = 0
        rc = self._main("--isl-osl", "512:128", "--concurrency", "1,4",
                        "--replicates", "2")
        self.assertEqual(rc, serving_benchmark.EXIT_OK)
        # One probe before the sweep, then one reset per run.
        self.assertEqual(self.server.resets, 1 + 4)
        seeds = [c[c.index("--seed") + 1] for c in self._calls()]
        self.assertEqual(seeds, ["42"] * 4)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["settings"]["prefix_cache"], "reset per run")

    def test_cell_over_max_model_len_is_skipped_not_run(self):
        rc = self._main("--isl-osl", "2048:256")
        self.assertEqual(rc, serving_benchmark.EXIT_CELL_FAILED)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["cells"][0]["status"], "skipped")
        self.assertEqual(self._calls(), [])

    def test_all_requests_failed_fails_the_cell(self):
        (self.dir / "override.json").write_text(
            json.dumps({
                "completed": 0,
                "failed": 10,
                "errors": ["boom"] * 10
            }))
        rc = self._main("--isl-osl", "64:8")
        self.assertEqual(rc, serving_benchmark.EXIT_CELL_FAILED)
        cell = json.loads((self.out / "summary.json").read_text())["cells"][0]
        self.assertEqual(cell["status"], "failed")
        self.assertIn("boom", cell["error"])

    def test_hung_run_times_out_and_the_sweep_goes_on(self):
        (self.dir / "hang").write_text("")
        rc = self._main("--isl-osl", "64:8,128:8", "--run-timeout", "1")
        self.assertEqual(rc, serving_benchmark.EXIT_CELL_FAILED)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual([c["status"] for c in summary["cells"]],
                         ["failed", "failed"])
        self.assertIn("did not finish within 1s", summary["cells"][0]["error"])
        self.assertIsNone(summary["error"])

    def test_server_down_after_a_failure_stops_the_sweep(self):
        (self.dir / "hang").write_text("")
        with unittest.mock.patch.object(serving_benchmark,
                                        "server_alive",
                                        return_value=False):
            rc = self._main("--isl-osl", "64:8,128:8", "--run-timeout", "1")
        self.assertEqual(rc, serving_benchmark.EXIT_SETUP)
        self.assertEqual(len(self._calls()), 1)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual([c["status"] for c in summary["cells"]],
                         ["failed", "skipped"])
        self.assertIn("stopped answering", summary["cells"][1]["error"])
        self.assertIn("isl64_osl8_c1", summary["error"])

    def test_server_path_missing_here_needs_tokenizer(self):
        # A server in a pod loaded a path that a port-forwarded client lacks.
        models = {
            "data": [{
                **_FakeServer.models["data"][0], "root":
                "/weights/not-on-this-host"
            }]
        }
        with unittest.mock.patch.object(_FakeServer, "models", models):
            rc = self._main("--isl-osl", "64:8")
            self.assertEqual(rc, serving_benchmark.EXIT_SETUP)
            self.assertEqual(self._calls(), [])

            rc = self._main("--isl-osl", "64:8", "--tokenizer",
                            "hub-org/model")
        self.assertEqual(rc, serving_benchmark.EXIT_OK)
        argv = self._calls()[-1]
        self.assertEqual(argv[argv.index("--tokenizer") + 1], "hub-org/model")

    def _fake_fetch_weights(self, copies):
        local = self.dir / "tokenizer-copy"

        def fetch_gcs(url, small_only=False):
            copies.append((url, small_only))
            print("WEIGHTS READY (stdout of the copy)")
            local.mkdir(exist_ok=True)
            return str(local)

        module = unittest.mock.MagicMock()
        module.fetch_gcs = fetch_gcs
        return module, str(local)

    def test_a_gcs_tokenizer_is_copied_here_and_used_locally(self):
        copies = []
        module, local = self._fake_fetch_weights(copies)
        with unittest.mock.patch.dict(sys.modules, {"fetch_weights": module}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = self._main("--isl-osl", "64:8", "--tokenizer",
                            "gs://bucket/models/m")
        self.assertEqual(rc, serving_benchmark.EXIT_OK)
        self.assertEqual(copies, [("gs://bucket/models/m", True)])
        argv = self._calls()[-1]
        self.assertEqual(argv[argv.index("--tokenizer") + 1], local)
        # The copy's progress lines stay off stdout, which is for JSON.
        self.assertNotIn("WEIGHTS READY", out.getvalue())

    def test_a_server_that_loaded_a_gcs_folder_needs_no_tokenizer(self):
        copies = []
        module, local = self._fake_fetch_weights(copies)
        models = {
            "data": [{
                **_FakeServer.models["data"][0], "root": "gs://bucket/m"
            }]
        }
        with unittest.mock.patch.object(_FakeServer, "models", models), \
                unittest.mock.patch.dict(sys.modules,
                                         {"fetch_weights": module}):
            rc = self._main("--isl-osl", "64:8")
        self.assertEqual(rc, serving_benchmark.EXIT_OK)
        self.assertEqual(copies, [("gs://bucket/m", True)])
        argv = self._calls()[-1]
        self.assertEqual(argv[argv.index("--tokenizer") + 1], local)

    def test_a_gcs_tokenizer_that_cannot_be_copied_is_a_setup_error(self):
        module = unittest.mock.MagicMock()
        module.fetch_gcs.side_effect = PermissionError(
            13, "no permission to read gs://bucket/m")
        with unittest.mock.patch.dict(sys.modules, {"fetch_weights": module}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = self._main("--isl-osl", "64:8", "--tokenizer",
                            "gs://bucket/m")
        self.assertEqual(rc, serving_benchmark.EXIT_SETUP)
        self.assertEqual(self._calls(), [])
        error = json.loads(out.getvalue())["error"]
        self.assertIn("could not copy the tokenizer from gs://bucket/m", error)
        self.assertIn("no permission", error)

    def test_dry_run_never_copies(self):
        module = unittest.mock.MagicMock()
        module.fetch_gcs.side_effect = AssertionError("copied")
        with unittest.mock.patch.dict(sys.modules, {"fetch_weights": module}), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = self._main("--isl-osl", "64:8", "--tokenizer", "gs://b/m",
                            "--model", "m", "--dry-run")
        self.assertEqual(rc, serving_benchmark.EXIT_OK)

    def test_missing_path_error_names_a_gcs_folder_as_an_option(self):
        models = {
            "data": [{
                **_FakeServer.models["data"][0], "root":
                "/weights/elsewhere"
            }]
        }
        with unittest.mock.patch.object(_FakeServer, "models", models), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = self._main("--isl-osl", "64:8")
        self.assertEqual(rc, serving_benchmark.EXIT_SETUP)
        self.assertIn("gs:// folder", json.loads(out.getvalue())["error"])

    def test_unreachable_server_is_a_setup_error(self):
        rc = serving_benchmark.main([
            "--base-url", "http://127.0.0.1:1", "--output-dir",
            str(self.out), "--ready-timeout", "0"
        ])
        self.assertEqual(rc, serving_benchmark.EXIT_SETUP)

    def test_sweep_static_filtering_and_dynamic_early_stopping(self):
        overrides = {
            "isl64_osl8_c1_r1.json": {
                "output_throughput": 100.0
            },
            "isl64_osl8_c2_r1.json": {
                "output_throughput": 105.0
            },
            "isl64_osl8_c4_r1.json": {
                "output_throughput": 108.0
            },
            "isl128_osl8_c4_r1.json": {
                "completed": 0,
                "failed": 10,
                "errors": ["timeout"],
            },
        }
        (self.dir / "overrides_by_file.json").write_text(json.dumps(overrides))
        rc = self._main(
            "--isl-osl",
            "64:8,128:8",
            "--concurrency",
            "1,2,4,8,16",
            "--max-concurrency-limit",
            "8",
            "--avoid",
            "128:*:2",
            "--stop-below-gain-pct",
            "10.0",
            "--stop-on-failure",
        )
        self.assertEqual(rc, serving_benchmark.EXIT_OK)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["settings"]["sweep_mode"], "exploratory")
        by_cell = {c["cell"]: c for c in summary["cells"]}
        # Shape 64:8: c=1,2,4 ok; c=8 skipped (2 low-gain steps); c=16 > limit 8.
        self.assertEqual(by_cell["isl64_osl8_c4"]["status"], "ok")
        self.assertEqual(by_cell["isl64_osl8_c8"]["status"], "skipped")
        self.assertIn("throughput saturated at c=4",
                      by_cell["isl64_osl8_c8"]["error"])
        self.assertIn("max_concurrency_limit 8",
                      by_cell["isl64_osl8_c16"]["error"])
        # Shape 128:8: c=1 ok; c=2 avoided; c=4 failed; c=8 early-stopped.
        self.assertEqual(by_cell["isl128_osl8_c1"]["status"], "ok")
        self.assertIn("--avoid rule 128:*:2",
                      by_cell["isl128_osl8_c2"]["error"])
        self.assertEqual(by_cell["isl128_osl8_c4"]["status"], "failed")
        self.assertIn("earlier concurrency c=4 failed",
                      by_cell["isl128_osl8_c8"]["error"])
        # Seeds stay anchored to each cell's fixed grid index despite skips.
        self.assertEqual(by_cell["isl128_osl8_c1"]["seeds"], [42 + 5])
        self.assertEqual(by_cell["isl128_osl8_c4"]["seeds"], [42 + 7])

    def test_sglang_bench_client_and_flush_cache(self):
        self.server.flushes = 0
        models = {"data": [{"id": "org/model", "root": "hub-org/model"}]}
        server_info = {"context_length": 4096}
        sglang_cmd = shlex.join(
            [sys.executable, str(self.dir / "fake_sglang.py")])
        with unittest.mock.patch.object(_FakeServer, "models", models), \
                unittest.mock.patch.object(_FakeServer, "server_info", server_info):
            rc = self._main("--bench-client", "sglang", "--sglang-cmd",
                            sglang_cmd, "--isl-osl", "512:128",
                            "--concurrency", "1,4")
        self.assertEqual(rc, serving_benchmark.EXIT_OK)
        self.assertEqual(self.server.flushes, 1 + 2)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["client"], "sglang.bench_serving")
        self.assertEqual(summary["model"]["max_model_len"], 4096)
        self.assertEqual(summary["settings"]["prefix_cache"], "reset per run")
        self.assertIn("512:128", summary["curves"])
        argv = self._calls()[-1]
        self.assertEqual(argv[argv.index("--backend") + 1], "sglang-oai")
        self.assertEqual(argv[argv.index("--max-concurrency") + 1], "4")

    def test_pareto_reproduction_exports_bkc_when_consistent(self):
        ref_path = self.dir / "ref_summary.json"
        ref_path.write_text(
            json.dumps({
                "curves": {
                    "512:128": {
                        "pareto_frontier": [{
                            "concurrency":
                            1,
                            "output_tokens_per_s":
                            175.0,
                            "output_tokens_per_s_per_user":
                            195.0,
                        }]
                    }
                }
            }),
            encoding="utf-8",
        )
        bkc_path = self.dir / "bkc.json"
        rc = self._main("--isl-osl", "512:128", "--concurrency", "1",
                        "--reference-curve", str(ref_path), "--export-bkc",
                        str(bkc_path))
        self.assertEqual(rc, serving_benchmark.EXIT_OK)
        self.assertTrue(bkc_path.exists())
        bkc = json.loads(bkc_path.read_text(encoding="utf-8"))
        self.assertEqual(bkc["status"], "best_known_config")
        self.assertTrue(bkc["reproduction"]["consistent"])
        self.assertTrue(bkc["reproduction"]["promote_to_bkc"])

    def test_export_bkc_requires_reference_curve(self):
        bkc_path = self.dir / "bkc.json"
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = self._main("--isl-osl", "512:128", "--concurrency", "1",
                            "--export-bkc", str(bkc_path))
        self.assertEqual(rc, serving_benchmark.EXIT_SETUP)
        self.assertFalse(bkc_path.exists())
        self.assertEqual(self._calls(), [])
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["status"], "error")
        self.assertIn("--reference-curve", payload["error"])


class _FakeFastAPIServer(http.server.BaseHTTPRequestHandler):
    """Answers like torch-inference-coder's FastAPI recipe: /health and
    /generate, and no /v1/models or /reset_prefix_cache."""

    def do_GET(self):  # noqa: N802
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
        if self.path == "/health":
            self.wfile.write(b'{"status": "ok"}')

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.posts.append((self.path, body))
        if self.path != "/generate" or self.server.fail_every and (
                len(self.server.posts) % self.server.fail_every == 0):
            self.send_response(500 if self.path == "/generate" else 404)
            self.end_headers()
            return
        data = json.dumps({"result": "one two three"}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class TestFastAPIEngine(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = pathlib.Path(self.tmp.name) / "out"
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                                      _FakeFastAPIServer)
        self.server.posts = []
        self.server.fail_every = 0
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def _main(self, *extra: str) -> int:
        return serving_benchmark.main([
            "--engine", "custom_fastapi", "--model", "org/model", "--base-url",
            self.base_url, "--output-dir",
            str(self.out), "--ready-timeout", "5", "--isl-osl", "8:4",
            "--num-prompts", "3", *extra
        ])

    def test_summary_has_the_vllm_names_without_streaming_metrics(self):
        rc = self._main("--concurrency", "1,2", "--replicates", "2")
        self.assertEqual(rc, serving_benchmark.EXIT_OK)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["client"], "benchmark_fastapi_server.py")
        self.assertEqual(summary["settings"]["engine"], "custom_fastapi")
        self.assertIn("none", summary["settings"]["prefix_cache"])
        metrics = summary["cells"][1]["metrics"]
        for name in ("e2el_p50_ms", "e2el_p90_ms", "e2el_p99_ms",
                     "requests_per_s", "completed"):
            self.assertIn(name, metrics)
        self.assertEqual(metrics["completed"]["values"], [3, 3])
        self.assertFalse(
            [m for m in metrics if m.startswith(("ttft", "tpot"))])

    def test_every_request_is_new_and_held_to_the_output_length(self):
        self._main("--concurrency", "1", "--replicates", "2")
        prompts = [body["prompt"] for _, body in self.server.posts]
        # Two runs of 1 warmup + 3 measured requests, each prompt unseen.
        self.assertEqual(len(prompts), 8)
        self.assertEqual(len(set(prompts)), 8)
        self.assertTrue(
            all(body["ignore_eos"] and body["max_tokens"] == 4
                for _, body in self.server.posts))

    def test_failed_requests_are_judged_by_max_failed_pct(self):
        self.server.fail_every = 3  # The warmup passes; one in three fails.
        rc = self._main("--concurrency", "1")
        self.assertEqual(rc, serving_benchmark.EXIT_CELL_FAILED)
        cell = json.loads((self.out / "summary.json").read_text())["cells"][0]
        self.assertEqual(cell["status"], "failed")
        self.assertIn("requests failed", cell["error"])

        self.server.posts = []
        rc = self._main("--concurrency", "1", "--max-failed-pct", "50")
        self.assertEqual(rc, serving_benchmark.EXIT_OK)

    def test_model_is_required(self):
        with self.assertRaises(SystemExit):
            serving_benchmark.parse_args(
                ["--engine", "custom_fastapi", "--output-dir",
                 str(self.out)])

    def test_vqa_needs_an_image(self):
        with self.assertRaises(SystemExit):
            serving_benchmark.parse_args([
                "--engine", "custom_fastapi", "--model", "m", "--task", "vqa",
                "--output-dir",
                str(self.out)
            ])
        with self.assertRaises(SystemExit):
            serving_benchmark.parse_args([
                "--engine", "custom_fastapi", "--model", "m", "--output-dir",
                str(self.out), "--stop-below-gain-pct", "10"
            ])


# An AgentX aggregate result (process_agentic_result.py's), trimmed to the
# keys the summary reads. Latencies are in seconds.
AGENTX_RESULT = {
    "num_requests_successful": 73,
    "request_accounting": {
        "records_error_dropped": 0
    },
    "request_metrics": {
        "qps": {
            "mean": 0.04
        },
        "latency": {
            "ttft": {
                "mean": 2.0,
                "p50": 1.5,
                "p90": 4.0,
                "p95": 5.0
            },
            "tpot": {
                "mean": 0.1,
                "p50": 0.09,
                "p90": 0.2,
                "p95": 0.25
            },
            "itl": {
                "mean": 0.1,
                "p50": 0.09
            },
            "e2el": {
                "mean": 30.0,
                "p50": 25.0
            },
        },
        "throughput": {
            "input": {
                "tokens_per_second": 4000.0
            },
            "output": {
                "tokens_per_second": 20.0
            },
            "total": {
                "tokens_per_second": 4020.0
            },
            "duration_seconds": 1800.0,
        },
        "cache": {
            "theoretical_cache_hit_rate": 0.95
        },
    },
    "server_metrics": {
        "cache": {
            "overall_cache_hit_rate": 0.9
        }
    },
}

# Stands in for benchmark_lib.sh's replay: writes what the real one writes
# (the replay command and the aggregate result), records the
# variables it was run with, and exits FAKE_RC.
FAKE_AGENTX_REPLAY = r'''set -e
python3 - <<'PY'
import json, os, pathlib
env = os.environ
result_dir = pathlib.Path(env["RESULT_DIR"])
artifacts = result_dir / "aiperf_artifacts"
artifacts.mkdir(parents=True, exist_ok=True)
(result_dir / "benchmark_command.txt").write_text(
    "aiperf profile --scenario inferencex-agentx-mvp --concurrency "
    + env["CONC"])
out = pathlib.Path(env["AGENTIC_OUTPUT_DIR"]) / (env["RESULT_FILENAME"] + ".json")
out.write_text(pathlib.Path(env["FAKE_AGENTX_RESULT"]).read_text())
with open(env["FAKE_AGENTX_CALLS"], "a") as f:
    f.write(json.dumps({"cwd": os.getcwd(), "env": dict(env)}) + "\n")
PY
exit "${FAKE_RC:-0}"
'''


class TestAgentXWorkload(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                                      _FakeServer)
        self.server.resets = 0
        # A context the 256k corpus fits.
        self.server.RequestHandlerClass = type(
            "_AgentXServer", (_FakeServer, ), {
                "models": {
                    "data": [{
                        "id": "org/model",
                        "root": "hub-org/model",
                        "max_model_len": 262144,
                    }]
                }
            })
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.out = self.dir / "out"
        self.cache = self.dir / "cache"
        (self.cache /
         f"InferenceX-{serving_benchmark.INFERENCEX_SHA[:9]}").mkdir(
             parents=True)
        result = self.dir / "agentx_result.json"
        result.write_text(json.dumps(AGENTX_RESULT), encoding="utf-8")
        self.calls_path = self.dir / "calls.jsonl"
        self.prepared = []
        patches = [
            unittest.mock.patch.object(serving_benchmark,
                                       "AGENTX_REPLAY_SCRIPT",
                                       FAKE_AGENTX_REPLAY),
            unittest.mock.patch.object(
                serving_benchmark, "prepare_agentx",
                lambda args, raw_dir: self.prepared.append(raw_dir)),
            unittest.mock.patch.dict(
                os.environ, {
                    "FAKE_AGENTX_RESULT": str(result),
                    "FAKE_AGENTX_CALLS": str(self.calls_path),
                    "PYTHONPATH": "/somewhere/torch_tpu",
                }),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        for name in ("HF_HOME", "HF_HUB_CACHE"):
            os.environ.pop(name, None)  # patch.dict restores them.

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def _main(self, *extra: str) -> tuple[int, str]:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(
                io.StringIO()):
            rc = serving_benchmark.main([
                "--base-url", self.base_url, "--output-dir",
                str(self.out), "--ready-timeout", "5", "--workload", "agentx",
                "--agentx-cache-dir",
                str(self.cache), *extra
            ])
        return rc, err.getvalue()

    def _calls(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.calls_path.read_text().splitlines()
        ]

    def test_replays_each_concurrency_and_maps_the_metrics(self):
        rc, _ = self._main("--concurrency", "4,1")
        self.assertEqual(rc, serving_benchmark.EXIT_OK)
        self.assertEqual(len(self.prepared), 1)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["client"], "aiperf (InferenceX AgentX)")
        self.assertEqual(summary["settings"]["workload"], "agentx")
        self.assertEqual(summary["settings"]["prefix_cache"], "reset per run")
        agentx = summary["settings"]["agentx"]
        self.assertEqual(agentx["inferencex_sha"],
                         serving_benchmark.INFERENCEX_SHA)
        self.assertEqual(agentx["corpus"],
                         serving_benchmark.AGENTX_CORPUS_256K)
        self.assertTrue(agentx["canonical"])
        self.assertEqual([c["cell"] for c in summary["cells"]],
                         ["agentx_c1", "agentx_c4"])
        cell = summary["cells"][0]
        self.assertEqual(
            cell["command"],
            "aiperf profile --scenario inferencex-agentx-mvp --concurrency 1")
        metrics = {k: v["mean"] for k, v in cell["metrics"].items()}
        self.assertAlmostEqual(metrics["ttft_p50_ms"], 1500.0)
        self.assertAlmostEqual(metrics["ttft_p95_ms"], 5000.0)
        self.assertAlmostEqual(metrics["tpot_mean_ms"], 100.0)
        self.assertAlmostEqual(metrics["output_tokens_per_s_per_user"], 10.0)
        self.assertEqual(metrics["output_tokens_per_s"], 20.0)
        self.assertEqual(metrics["total_tokens_per_s"], 4020.0)
        self.assertEqual(metrics["total_input_tokens"], 7200000)
        self.assertEqual(metrics["completed"], 73)
        self.assertEqual(metrics["failed"], 0)
        self.assertEqual(metrics["requests_per_s"], 0.04)
        self.assertEqual(metrics["prefix_cache_hit_rate"], 0.9)
        self.assertEqual(metrics["theoretical_prefix_cache_hit_rate"], 0.95)
        self.assertNotIn("ttft_p99_ms", metrics)
        self.assertIn("prefix hit", (self.out / "summary.md").read_text())
        # One reset at setup, then one before every replay.
        self.assertEqual(self.server.resets, 3)

    def test_each_replay_gets_the_variables_an_agentx_recipe_sets(self):
        rc, _ = self._main("--concurrency", "2", "--tokenizer", "/tok")
        self.assertEqual(rc, serving_benchmark.EXIT_OK)
        [call] = self._calls()
        env = call["env"]
        self.assertEqual(
            call["cwd"],
            str(self.cache.resolve() / f"InferenceX-"
                f"{serving_benchmark.INFERENCEX_SHA[:9]}"))
        self.assertEqual(env["MODEL"], "/tok")
        self.assertEqual(env["SERVED_MODEL_NAME"], "org/model")
        self.assertEqual(env["AIPERF_SERVER_URL"], self.base_url)
        self.assertEqual(env["CONC"], "2")
        self.assertEqual(env["DURATION"], "1800")
        self.assertEqual(env["MAX_MODEL_LEN"], "262144")
        self.assertEqual(env["WEKA_LOADER_OVERRIDE"],
                         serving_benchmark.AGENTX_CORPUS_256K)
        self.assertEqual(env["AIPERF_FAILED_REQUEST_THRESHOLD"], "0.1")
        self.assertEqual(env["ENABLE_AGENTX_POWER"], "0")
        self.assertEqual(env["KV_OFFLOADING"], "none")
        self.assertEqual(env["FRAMEWORK"], "vllm")
        self.assertEqual(env["RESULT_FILENAME"], "agentx_c2_r1")
        self.assertEqual(env["AIPERF_VENV"],
                         str(self.cache.resolve() / "venv"))
        self.assertEqual(env["HF_HUB_CACHE"],
                         str(self.cache.resolve() / "hf_hub"))
        self.assertNotIn("PYTHONPATH", env)

    def test_a_failed_replay_fails_its_cell(self):
        with unittest.mock.patch.dict(os.environ, {"FAKE_RC": "1"}):
            rc, _ = self._main("--concurrency", "1")
        self.assertEqual(rc, serving_benchmark.EXIT_CELL_FAILED)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["cells"][0]["status"], "failed")
        self.assertIn("exited 1", summary["cells"][0]["error"])

    def test_dry_run_prints_the_replay_and_sets_nothing_up(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(
                io.StringIO()):
            rc = serving_benchmark.main([
                "--base-url", self.base_url, "--output-dir",
                str(self.out), "--workload", "agentx", "--dry-run", "--model",
                "org/model", "--concurrency", "8"
            ])
        self.assertEqual(rc, serving_benchmark.EXIT_OK)
        self.assertEqual(self.prepared, [])
        self.assertIn("CONC=8", out.getvalue())
        self.assertIn("SERVED_MODEL_NAME=org/model", out.getvalue())

    def test_corpus_follows_the_servers_context(self):
        corpus = serving_benchmark.agentx_corpus
        self.assertEqual(corpus(1_048_576, None),
                         serving_benchmark.AGENTX_CORPUS_FULL)
        self.assertEqual(corpus(262_144, None),
                         serving_benchmark.AGENTX_CORPUS_256K)
        self.assertEqual(corpus(None, None),
                         serving_benchmark.AGENTX_CORPUS_256K)
        self.assertEqual(corpus(8192, serving_benchmark.AGENTX_CORPUS_FULL),
                         serving_benchmark.AGENTX_CORPUS_FULL)

    def test_flags(self):

        def parse(*extra):
            return serving_benchmark.parse_args(
                ["--output-dir", str(self.out), *extra])

        args = parse("--workload", "agentx")
        self.assertEqual(args.max_failed_pct, 10.0)
        self.assertEqual(
            args.run_timeout, serving_benchmark.AGENTX_DURATION_S +
            serving_benchmark.AGENTX_TIMEOUT_SLACK_S)
        self.assertEqual(args.agentx_cache_dir, self.out / "agentx_cache")
        self.assertEqual(
            parse("--workload", "agentx", "--agentx-fast").run_timeout,
            1200 + serving_benchmark.AGENTX_TIMEOUT_SLACK_S)
        random = parse()
        self.assertEqual((random.max_failed_pct, random.run_timeout),
                         (0.0, 1800.0))
        for bad in (["--isl-osl", "1024:128"], ["--stop-on-failure"],
                    ["--engine", "custom_fastapi", "--model",
                     "m"], ["--agentx-coverage-ratio", "1.5"]):
            with self.subTest(bad=bad), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                parse("--workload", "agentx", *bad)


if __name__ == "__main__":
    unittest.main()
