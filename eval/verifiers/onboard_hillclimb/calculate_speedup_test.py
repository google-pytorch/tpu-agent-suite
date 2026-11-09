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
"""Unit tests for calculate_speedup verifier."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_speedup_module():
    """Loads calculate_speedup.py with an identity @criterion decorator for testing."""
    rk = types.ModuleType("rewardkit")

    def _criterion(fn=None, *, description: str | None = None):

        def _decorator(func):
            func._criterion_description = description
            return func

        if fn is not None:
            return _decorator(fn)
        return _decorator

    rk.criterion = _criterion  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"rewardkit": rk}):
        script_path = Path(__file__).resolve().parent / "calculate_speedup.py"
        spec = importlib.util.spec_from_file_location("calculate_speedup",
                                                      script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


speedup_mod = _load_speedup_module()


class CalculateSpeedupTest(unittest.TestCase):
    """Tests manifest verification, regime/concurrency checks, failed runs, and legacy fallback."""

    def _write_raw_json(
        self,
        path: Path,
        *,
        output_tps: float,
        concurrency: int = 1,
        isl: int = 1024,
        osl: int = 1024,
        completed: int = 16,
        failed: int = 0,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "completed": completed,
            "failed": failed,
            "max_concurrency": concurrency,
            "random_input_len": isl,
            "random_output_len": osl,
            "output_throughput": output_tps,
            "median_ttft_ms": 45.0,
            "median_itl_ms": 12.0,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def test_verified_manifest_matches_raw_outputs(self) -> None:
        """Valid benchmark_results.json matching raw JSON and stdout logs produces verified speedup."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            self._write_raw_json(
                artifacts / "raw" / "base_1k1k_c1.json",
                output_tps=100.0,
                concurrency=1,
            )
            self._write_raw_json(
                artifacts / "raw" / "base_1k1k_c4.json",
                output_tps=200.04,
                concurrency=4,
            )
            self._write_raw_json(
                artifacts / "raw" / "opt_1k1k_c1.json",
                output_tps=150.0,
                concurrency=1,
            )
            # Test plain-text vllm bench serve log for optimized c=4
            raw_log = artifacts / "raw" / "opt_1k1k_c4.log"
            raw_log.write_text(
                "============ Serving Benchmark Result ============\n"
                "Successful requests:                     16\n"
                "Failed requests:                         0\n"
                "Maximum request concurrency:             4\n"
                "Total input tokens:                      16384\n"
                "Total generated tokens:                  16384\n"
                "Output token throughput (tok/s):         300.06\n"
                "Median TTFT (ms):                        35.2\n"
                "Median ITL (ms):                         8.4\n",
                encoding="utf-8",
            )

            manifest = {
                "baseline": [
                    {
                        "regime": "1k/1k",
                        "concurrency": 1,
                        "output_tps": 100.0,
                        "raw_output_path": "raw/base_1k1k_c1.json",
                    },
                    {
                        "regime": "1k/1k",
                        "concurrency": 4,
                        # Within rounding tolerance of 200.04
                        "output_tps": 200.0,
                        "raw_output_path":
                        "/logs/artifacts/raw/base_1k1k_c4.json",
                    },
                ],
                "optimized": [
                    {
                        "regime": "1k/1k",
                        "concurrency": 1,
                        "output_tps": 150.0,
                        "raw_output_path": "raw/opt_1k1k_c1.json",
                    },
                    {
                        "regime": "1k/1k",
                        "concurrency": 4,
                        "output_tps": 300.1,
                        "raw_output_path": "raw/opt_1k1k_c4.log",
                    },
                ],
            }
            (artifacts / "benchmark_results.json").write_text(json.dumps(
                manifest, indent=2),
                                                              encoding="utf-8")

            discovery = speedup_mod.discover_or_verify_runs(root)
            self.assertFalse(discovery.inferred)
            self.assertEqual(discovery.verification_errors, [])
            self.assertEqual(len(discovery.baseline_runs), 2)
            self.assertEqual(len(discovery.optimized_runs), 2)

            evaluator = speedup_mod.DeterministicSpeedupEvaluator(
                discovery.baseline_runs,
                discovery.optimized_runs,
                inferred=discovery.inferred,
            )
            res = evaluator.evaluate(target_regimes=["1k/1k"])
            self.assertFalse(res["speedup_inferred"])
            # Uses raw TPS 300.06 / 200.04 = 1.5
            self.assertAlmostEqual(res["final_speedup"], 1.5, places=4)
            self.assertAlmostEqual(speedup_mod.tpu_speedup(root),
                                   1.5,
                                   places=4)

    def test_manifest_in_specialist_subdirectory_preferred_over_fallback(
            self) -> None:
        """benchmark_results.json inside a specialist subdirectory is discovered and preferred over top-level fallback."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Create a top-level fallback manifest
            top_manifest = root / "benchmark_results.json"
            top_manifest.write_text('{"name": "fallback"}', encoding="utf-8")
            # Create a specialist subdirectory manifest
            specialist_dir = root / "artifacts" / "tpu-model-onboarding"
            specialist_dir.mkdir(parents=True)
            specialist_manifest = specialist_dir / "benchmark_results.json"
            specialist_manifest.write_text('{"name": "specialist"}',
                                           encoding="utf-8")
            # Specialist manifest should take precedence
            self.assertEqual(speedup_mod.find_manifest_file(root),
                             specialist_manifest)

    def test_mistyped_tps_in_manifest_is_rejected(self) -> None:
        """A mistyped or inflated TPS in benchmark_results.json is rejected during raw cross-check."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_raw_json(
                root / "raw" / "base_c4.json",
                output_tps=200.0,
                concurrency=4,
            )
            self._write_raw_json(
                root / "raw" / "opt_c4.json",
                output_tps=202.0,
                concurrency=4,
            )
            # Also plant a markdown report with the inflated number to ensure
            # it does not fall back to markdown when benchmark_results.json exists
            (root / "benchmark_report.md").write_text(
                "| Regime | Concurrency | Baseline TPS | Optimized TPS |\n"
                "| --- | --- | --- | --- |\n"
                "| 1k/1k | 4 | 200.0 | 350.0 |\n",
                encoding="utf-8",
            )

            manifest = {
                "baseline": [{
                    "regime": "1k/1k",
                    "concurrency": 4,
                    "output_tps": 200.0,
                    "raw_output_path": "raw/base_c4.json",
                }],
                "optimized": [{
                    "regime": "1k/1k",
                    "concurrency": 4,
                    # Mistyped/inflated TPS (350.0 vs raw 202.0)
                    "output_tps": 350.0,
                    "raw_output_path": "raw/opt_c4.json",
                }],
            }
            (root / "benchmark_results.json").write_text(json.dumps(manifest,
                                                                    indent=2),
                                                         encoding="utf-8")

            discovery = speedup_mod.discover_or_verify_runs(root)
            self.assertFalse(discovery.inferred)
            self.assertEqual(len(discovery.baseline_runs), 1)
            self.assertEqual(len(discovery.optimized_runs), 0)
            self.assertEqual(len(discovery.verification_errors), 1)
            self.assertIn("does not match raw output TPS",
                          discovery.verification_errors[0])

            # With no valid optimized runs, speedup falls back to 1.0 (no hillclimb credit)
            self.assertEqual(speedup_mod.tpu_speedup(root), 1.0)

    def test_mismatched_regime_and_concurrency_rejected(self) -> None:
        """Manifest entries pointing to raw outputs from a different regime or concurrency are rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_raw_json(
                root / "raw" / "base_1k1k_c4.json",
                output_tps=200.0,
                concurrency=4,
                isl=1024,
                osl=1024,
            )
            # Raw file is actually 8k/1k (isl=8192, osl=1024)
            self._write_raw_json(
                root / "raw" / "opt_wrong_regime.json",
                output_tps=300.0,
                concurrency=4,
                isl=8192,
                osl=1024,
            )
            # Raw file is actually concurrency=8
            self._write_raw_json(
                root / "raw" / "opt_wrong_conc.json",
                output_tps=320.0,
                concurrency=8,
                isl=1024,
                osl=1024,
            )

            manifest = {
                "baseline": [{
                    "regime": "1k/1k",
                    "concurrency": 4,
                    "output_tps": 200.0,
                    "raw_output_path": "raw/base_1k1k_c4.json",
                }],
                "optimized": [
                    {
                        "regime": "1k/1k",
                        "concurrency": 4,
                        "output_tps": 300.0,
                        "raw_output_path": "raw/opt_wrong_regime.json",
                    },
                    {
                        "regime": "1k/1k",
                        "concurrency": 4,
                        "output_tps": 320.0,
                        "raw_output_path": "raw/opt_wrong_conc.json",
                    },
                ],
            }
            (root / "benchmark_results.json").write_text(json.dumps(manifest,
                                                                    indent=2),
                                                         encoding="utf-8")

            discovery = speedup_mod.discover_or_verify_runs(root)
            self.assertFalse(discovery.inferred)
            self.assertEqual(len(discovery.baseline_runs), 1)
            self.assertEqual(len(discovery.optimized_runs), 0)
            self.assertEqual(len(discovery.verification_errors), 2)
            self.assertTrue(
                any("mismatched regime" in e
                    for e in discovery.verification_errors))
            self.assertTrue(
                any("mismatched concurrency" in e
                    for e in discovery.verification_errors))

    def test_mismatched_regimes_in_evaluator(self) -> None:
        """Evaluator does not cross-compare runs from different regimes."""
        base_8k1k = speedup_mod.BenchmarkRunMetrics(regime="8k/1k",
                                                    concurrency=4,
                                                    output_tps=100.0,
                                                    completed=16)
        opt_1k8k = speedup_mod.BenchmarkRunMetrics(regime="1k/8k",
                                                   concurrency=4,
                                                   output_tps=250.0,
                                                   completed=16)

        evaluator = speedup_mod.DeterministicSpeedupEvaluator([base_8k1k],
                                                              [opt_1k8k])
        # Target bottleneck regime 1k/1k was never measured in baseline -> 0.0
        res_1k1k = evaluator.evaluate(target_regimes=["1k/1k"])
        self.assertEqual(res_1k1k["final_speedup"], 0.0)

        # Target regime 8k/1k has baseline but no matching 8k/1k optimized run -> 1.0
        res_8k1k = evaluator.evaluate(target_regimes=["8k/1k"])
        self.assertEqual(res_8k1k["final_speedup"], 1.0)

    def test_failed_runs_rejected_in_manifest_and_evaluator(self) -> None:
        """Runs with failed requests or zero completed requests are rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_raw_json(
                root / "raw" / "base_c4.json",
                output_tps=200.0,
                concurrency=4,
                completed=16,
                failed=0,
            )
            self._write_raw_json(
                root / "raw" / "opt_c4_failed.json",
                output_tps=320.0,
                concurrency=4,
                completed=14,
                failed=2,
            )
            self._write_raw_json(
                root / "raw" / "opt_c8_zero_completed.json",
                output_tps=350.0,
                concurrency=8,
                completed=0,
                failed=0,
            )

            manifest = {
                "baseline": [{
                    "regime": "1k/1k",
                    "concurrency": 4,
                    "output_tps": 200.0,
                    "raw_output_path": "raw/base_c4.json",
                }],
                "optimized": [
                    {
                        "regime": "1k/1k",
                        "concurrency": 4,
                        "output_tps": 320.0,
                        "raw_output_path": "raw/opt_c4_failed.json",
                    },
                    {
                        "regime": "1k/1k",
                        "concurrency": 8,
                        "output_tps": 350.0,
                        "raw_output_path": "raw/opt_c8_zero_completed.json",
                    },
                ],
            }
            (root / "benchmark_results.json").write_text(json.dumps(manifest,
                                                                    indent=2),
                                                         encoding="utf-8")

            discovery = speedup_mod.discover_or_verify_runs(root)
            self.assertEqual(len(discovery.baseline_runs), 1)
            self.assertEqual(len(discovery.optimized_runs), 0)
            self.assertEqual(len(discovery.verification_errors), 2)
            self.assertTrue(
                any("failed requests" in e
                    for e in discovery.verification_errors))
            self.assertTrue(
                any("0 completed requests" in e
                    for e in discovery.verification_errors))
            self.assertEqual(speedup_mod.tpu_speedup(root), 1.0)

        # Direct evaluator check: failed runs passed to DeterministicSpeedupEvaluator are excluded
        good_base = speedup_mod.BenchmarkRunMetrics(regime="1k/1k",
                                                    concurrency=4,
                                                    output_tps=200.0,
                                                    completed=16,
                                                    failed=0)
        failed_opt = speedup_mod.BenchmarkRunMetrics(regime="1k/1k",
                                                     concurrency=4,
                                                     output_tps=400.0,
                                                     completed=12,
                                                     failed=4)
        ev = speedup_mod.DeterministicSpeedupEvaluator([good_base],
                                                       [failed_opt])
        res = ev.evaluate(target_regimes=["1k/1k"])
        self.assertEqual(res["failed_runs_excluded"], 1)
        self.assertEqual(res["final_speedup"], 1.0)

    def test_legacy_pattern_discovery_marks_speedup_as_inferred(self) -> None:
        """Old runs without benchmark_results.json use pattern discovery and set speedup_inferred=True."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_raw_json(
                root / "raw_benchmarks" / "baseline" / "bench_1k1k_c4.json",
                output_tps=100.0,
                concurrency=4,
            )
            self._write_raw_json(
                root / "raw_benchmarks" / "optimized" / "bench_1k1k_c4.json",
                output_tps=135.0,
                concurrency=4,
            )

            discovery = speedup_mod.discover_or_verify_runs(root)
            self.assertTrue(discovery.inferred)
            self.assertIn("[INFERRED]", discovery.summary)
            self.assertEqual(discovery.verification_errors, [])

            evaluator = speedup_mod.DeterministicSpeedupEvaluator(
                discovery.baseline_runs,
                discovery.optimized_runs,
                inferred=discovery.inferred,
            )
            res = evaluator.evaluate(target_regimes=["1k/1k"])
            self.assertTrue(res["speedup_inferred"])
            self.assertAlmostEqual(res["final_speedup"], 1.35, places=4)
            trace = speedup_mod.format_debug_trace(res)
            self.assertIn("Verification  : INFERRED", trace)

    def test_markdown_raw_output_and_all_baseline_failed(self) -> None:
        """Markdown files cannot be used as raw outputs, and failed baseline runs yield 0.0."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            md_path = root / "table.md"
            md_path.write_text(
                "| ISL/OSL | Concurrency | Throughput (tok/s) |\n"
                "| --- | --- | --- |\n"
                "| 1k/1k | 4 | 200.0 |\n",
                encoding="utf-8",
            )
            self._write_raw_json(
                root / "base_failed.json",
                output_tps=200.0,
                concurrency=4,
                completed=10,
                failed=6,
            )
            manifest = {
                "baseline": [
                    {
                        "regime": "1k/1k",
                        "concurrency": 4,
                        "output_tps": 200.0,
                        "raw_output_path": "table.md",
                    },
                    {
                        "regime": "1k/1k",
                        "concurrency": 4,
                        "output_tps": 200.0,
                        "raw_output_path": "base_failed.json",
                    },
                ],
                "optimized": [],
            }
            (root / "benchmark_results.json").write_text(json.dumps(manifest,
                                                                    indent=2),
                                                         encoding="utf-8")

            discovery = speedup_mod.discover_or_verify_runs(root)
            self.assertFalse(discovery.inferred)
            self.assertEqual(discovery.baseline_runs, [])
            self.assertEqual(discovery.optimized_runs, [])
            self.assertEqual(len(discovery.verification_errors), 2)
            self.assertTrue(
                any("cannot be a markdown file" in e
                    for e in discovery.verification_errors))
            self.assertTrue(
                any("failed requests" in e
                    for e in discovery.verification_errors))
            self.assertEqual(speedup_mod.tpu_speedup(root), 0.0)

    def test_full_manifest_schema_with_comparison_and_failures(self) -> None:
        """Verifies the instruction.md schema containing throughput_tps, comparison, and failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_dir = root / "raw_benchmarks" / "baseline"
            base_dir.mkdir(parents=True)
            self._write_raw_json(
                base_dir / "bench_1k1k_c1.json",
                output_tps=142.5,
                completed=10,
                failed=0,
            )
            opt_dir = root / "raw_benchmarks" / "optimized"
            opt_dir.mkdir(parents=True)
            self._write_raw_json(
                opt_dir / "bench_1k1k_c1.json",
                output_tps=215.8,
                completed=10,
                failed=0,
            )
            manifest = {
                "baseline": [{
                    "config":
                    "Baseline",
                    "topology":
                    "v5e-8",
                    "dtype":
                    "bf16",
                    "regime":
                    "1k/1k",
                    "concurrency":
                    1,
                    "ttft_ms":
                    45.2,
                    "itl_ms":
                    12.1,
                    "output_tps":
                    142.5,
                    "raw_output_path":
                    "raw_benchmarks/baseline/bench_1k1k_c1.json",
                }],
                "optimized": [{
                    "config":
                    "Optimized",
                    "topology":
                    "v5e-8",
                    "dtype":
                    "bf16",
                    "regime":
                    "1k/1k",
                    "concurrency":
                    1,
                    "ttft_ms":
                    31.0,
                    "itl_ms":
                    8.4,
                    "throughput_tps":
                    215.8,
                    "raw_output_path":
                    "raw_benchmarks/optimized/bench_1k1k_c1.json",
                }],
                "comparison": [{
                    "regime":
                    "1k/1k",
                    "regime_label":
                    "Balanced",
                    "concurrency":
                    1,
                    "baseline_tps":
                    142.5,
                    "optimized_tps":
                    215.8,
                    "speedup_ratio":
                    1.51,
                    "primary_bottleneck_op":
                    "FlashAttention",
                    "pallas_optimization_applied":
                    "Fused KV-Cache quantized load",
                }],
                "failures": [{
                    "run_type": "optimized",
                    "regime": "8k/1k",
                    "concurrency": 16,
                    "reason": "OOM during prefill kernel compilation",
                }],
            }
            (root / "benchmark_results.json").write_text(json.dumps(manifest,
                                                                    indent=2),
                                                         encoding="utf-8")
            discovery = speedup_mod.discover_or_verify_runs(root)
            self.assertFalse(discovery.inferred)
            self.assertEqual(len(discovery.baseline_runs), 1)
            self.assertEqual(len(discovery.optimized_runs), 1)
            self.assertEqual(discovery.verification_errors, [])
            speedup = speedup_mod.tpu_speedup(root)
            self.assertAlmostEqual(speedup, 215.8 / 142.5, places=4)

    def test_single_canonical_calculate_speedup_exists(self) -> None:
        """Ensures the single canonical calculate_speedup.py lives next to grade.py."""
        script_path = Path(__file__).resolve().parent / "calculate_speedup.py"
        self.assertTrue(script_path.is_file())

    def test_manifest_output_tokens_per_s_and_raw_output_tps_unified(
            self) -> None:
        """Verifies that manifests using output_tokens_per_s and raw files using output_tps are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            # Raw file using "output_tps" and "completed_prompts"
            raw_base = artifacts / "raw" / "base_1k1k_c1.json"
            raw_base.parent.mkdir(parents=True, exist_ok=True)
            raw_base.write_text(json.dumps({
                "concurrency": 1,
                "isl": 1024,
                "osl": 1024,
                "completed_prompts": 4,
                "failed_prompts": 0,
                "output_tps": 180.0,
            }),
                                encoding="utf-8")

            raw_opt = artifacts / "raw" / "opt_1k1k_c1.json"
            raw_opt.write_text(json.dumps({
                "concurrency": 1,
                "isl": 1024,
                "osl": 1024,
                "num_successful": 4,
                "failed": 0,
                "output_tokens_per_s": 270.0,
            }),
                               encoding="utf-8")

            # Manifest using "output_tokens_per_s"
            manifest = {
                "baseline": [{
                    "regime": "1k/1k",
                    "concurrency": 1,
                    "output_tokens_per_s": 180.0,
                    "raw_output_path": "raw/base_1k1k_c1.json",
                }],
                "optimized": [{
                    "regime": "1k/1k",
                    "concurrency": 1,
                    "output_tps": 270.0,
                    "raw_output_path": "raw/opt_1k1k_c1.json",
                }],
            }
            (artifacts / "benchmark_results.json").write_text(json.dumps(
                manifest, indent=2),
                                                              encoding="utf-8")

            discovery = speedup_mod.discover_or_verify_runs(artifacts)
            self.assertFalse(discovery.inferred)
            self.assertEqual(len(discovery.baseline_runs), 1)
            self.assertEqual(len(discovery.optimized_runs), 1)
            self.assertAlmostEqual(discovery.baseline_runs[0].output_tps,
                                   180.0)
            self.assertAlmostEqual(discovery.optimized_runs[0].output_tps,
                                   270.0)

    def test_auto_discover_runs_searches_logs_artifacts_when_workspace(
            self) -> None:
        """Trial root searches logs/artifacts, and container /workspace searches _CONTAINER_ARTIFACTS_DIR."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            ws_dir = tmp_root / "workspace"
            ws_dir.mkdir()
            logs_artifacts = tmp_root / "logs" / "artifacts"
            logs_artifacts.mkdir(parents=True)

            self._write_raw_json(
                logs_artifacts / "raw_benchmarks" / "baseline" / "c1.json",
                output_tps=100.0,
                concurrency=1,
            )
            self._write_raw_json(
                logs_artifacts / "raw_benchmarks" / "optimized" / "c1.json",
                output_tps=150.0,
                concurrency=1,
            )

            base, opt, _ = speedup_mod.auto_discover_runs(tmp_root)
            self.assertEqual(len(base), 1)
            self.assertEqual(len(opt), 1)
            self.assertAlmostEqual(base[0].output_tps, 100.0)
            self.assertAlmostEqual(opt[0].output_tps, 150.0)

            with patch.object(speedup_mod, "_CONTAINER_ARTIFACTS_DIR",
                              logs_artifacts):
                c_base, c_opt, _ = speedup_mod.auto_discover_runs(
                    Path("/workspace"))
                self.assertEqual(len(c_base), 1)
                self.assertEqual(len(c_opt), 1)

    def test_manifest_fallback_to_auto_discover_when_zero_runs(self) -> None:
        """When manifest has 0 verified runs, discover_or_verify_runs falls back to auto_discover_runs."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Manifest points to non-existent raw file
            manifest = {
                "baseline": [{
                    "regime": "1k/1k",
                    "concurrency": 1,
                    "output_tps": 100.0,
                    "raw_output_path": "non_existent.json",
                }],
                "optimized": [],
            }
            (root / "benchmark_results.json").write_text(json.dumps(manifest,
                                                                    indent=2),
                                                         encoding="utf-8")

            # But raw benchmarks exist on disk in auto-discoverable layout
            self._write_raw_json(
                root / "raw_benchmarks" / "baseline" / "bench_c1.json",
                output_tps=100.0,
                concurrency=1,
            )
            self._write_raw_json(
                root / "raw_benchmarks" / "optimized" / "bench_c1.json",
                output_tps=150.0,
                concurrency=1,
            )

            discovery = speedup_mod.discover_or_verify_runs(root)
            self.assertTrue(discovery.inferred)
            self.assertEqual(len(discovery.baseline_runs), 1)
            self.assertEqual(len(discovery.optimized_runs), 1)

    def test_parse_raw_benchmark_with_nested_replicates_and_metrics(
            self) -> None:
        """Verifies raw benchmark files with nested replicates and metrics (agent suite format) are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            raw_base = artifacts / "raw" / "isl1024_osl1024_c1_r1.json"
            raw_base.parent.mkdir(parents=True, exist_ok=True)
            raw_payload = {
                "cell":
                "isl1024_osl1024_c1",
                "isl":
                1024,
                "osl":
                1024,
                "concurrency":
                1,
                "status":
                "ok",
                "replicates": [{
                    "output_tokens_per_s": 148.3,
                    "completed": 4,
                    "failed": 0,
                    "ttft_p50_ms": 42.5,
                    "itl_p50_ms": 11.2,
                }],
                "metrics": {
                    "output_tokens_per_s": {
                        "mean": 148.3,
                        "stdev": 0.0,
                        "values": [148.3],
                    },
                    "completed": {
                        "mean": 4.0,
                        "stdev": 0.0,
                        "values": [4],
                    },
                    "failed": {
                        "mean": 0.0,
                        "stdev": 0.0,
                        "values": [0],
                    },
                },
            }
            raw_base.write_text(json.dumps(raw_payload, indent=2),
                                encoding="utf-8")

            # 1. Test parse_raw_benchmark_file directly
            parsed = speedup_mod.parse_raw_benchmark_file(raw_base)
            self.assertIsNotNone(parsed)
            self.assertIsNone(parsed.error)
            self.assertEqual(parsed.detected_regime, "1k/1k")
            self.assertEqual(parsed.detected_concurrency, 1)
            self.assertIsNotNone(parsed.metrics)
            self.assertAlmostEqual(parsed.metrics.output_tps, 148.3)
            self.assertEqual(parsed.metrics.completed, 4)
            self.assertEqual(parsed.metrics.failed, 0)
            self.assertAlmostEqual(parsed.metrics.ttft_p50, 42.5)
            self.assertAlmostEqual(parsed.metrics.itl_p50, 11.2)

            # 2. Test BenchmarkRunMetrics.from_dict directly
            metrics = speedup_mod.BenchmarkRunMetrics.from_dict(raw_payload)
            self.assertAlmostEqual(metrics.output_tps, 148.3)
            self.assertEqual(metrics.completed, 4)
            self.assertEqual(metrics.failed, 0)
            self.assertAlmostEqual(metrics.ttft_p50, 42.5)
            self.assertAlmostEqual(metrics.itl_p50, 11.2)

            # 3. Test verification via manifest
            manifest = {
                "baseline": [{
                    "regime":
                    "1k/1k",
                    "concurrency":
                    1,
                    "output_tps":
                    148.3,
                    "raw_output_path":
                    "raw/isl1024_osl1024_c1_r1.json",
                }],
                "optimized": [{
                    "regime": "1k/1k",
                    "concurrency": 1,
                    "output_tps": 200.0,
                    "raw_output_path": "raw/opt.json",
                }],
            }
            self._write_raw_json(
                artifacts / "raw" / "opt.json",
                output_tps=200.0,
                concurrency=1,
            )
            (artifacts / "benchmark_results.json").write_text(json.dumps(
                manifest, indent=2),
                                                              encoding="utf-8")

            discovery = speedup_mod.discover_or_verify_runs(artifacts)
            self.assertFalse(discovery.inferred)
            self.assertEqual(len(discovery.baseline_runs), 1)
            self.assertEqual(len(discovery.optimized_runs), 1)
            self.assertAlmostEqual(discovery.baseline_runs[0].output_tps,
                                   148.3)
            self.assertAlmostEqual(discovery.optimized_runs[0].output_tps,
                                   200.0)
            self.assertEqual(discovery.verification_errors, [])

    def test_complete_manifest_preferred_over_specialist_onboarding_manifest(
            self) -> None:
        """Complete manifest with baseline and optimized runs is preferred over Stage 1 onboarding manifest."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)

            self._write_raw_json(
                artifacts / "raw" / "base_c1.json",
                output_tps=100.0,
                concurrency=1,
            )
            self._write_raw_json(
                artifacts / "raw" / "opt_c1.json",
                output_tps=150.0,
                concurrency=1,
            )

            # 1. Specialist subdirectory with onboarding-only manifest
            specialist_dir = artifacts / "tpu-model-onboarding"
            specialist_dir.mkdir(parents=True, exist_ok=True)
            specialist_manifest = {
                "stage":
                "baseline",
                "baseline": [{
                    "regime": "1k/1k",
                    "concurrency": 1,
                    "output_tps": 100.0,
                    "raw_output_path": "raw/base_c1.json",
                }],
            }
            (specialist_dir / "benchmark_results.json").write_text(
                json.dumps(specialist_manifest, indent=2), encoding="utf-8")

            # 2. Complete top-level manifest with baseline and optimized runs
            top_manifest = {
                "baseline": [{
                    "regime": "1k/1k",
                    "concurrency": 1,
                    "output_tps": 100.0,
                    "raw_output_path": "raw/base_c1.json",
                }],
                "optimized": [{
                    "regime": "1k/1k",
                    "concurrency": 1,
                    "output_tps": 150.0,
                    "raw_output_path": "raw/opt_c1.json",
                }],
            }
            top_manifest_path = artifacts / "benchmark_results.json"
            top_manifest_path.write_text(json.dumps(top_manifest, indent=2),
                                         encoding="utf-8")

            # Complete manifest should be selected
            self.assertEqual(speedup_mod.find_manifest_file(root),
                             top_manifest_path)

            discovery = speedup_mod.discover_or_verify_runs(root)
            self.assertFalse(discovery.inferred)
            self.assertEqual(len(discovery.baseline_runs), 1)
            self.assertEqual(len(discovery.optimized_runs), 1)
            self.assertEqual(discovery.verification_errors, [])
            self.assertAlmostEqual(discovery.baseline_runs[0].output_tps,
                                   100.0)
            self.assertAlmostEqual(discovery.optimized_runs[0].output_tps,
                                   150.0)

    def test_downloaded_trial_artifacts_logs_artifacts_manifest(self):
        """Verifies that manifests in <trial_dir>/artifacts/logs/artifacts are discovered and verified."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_dir = root / "artifacts" / "logs" / "artifacts"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            raw_dir = manifest_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            (raw_dir / "base_c1.json").write_text(
                json.dumps({
                    "completed": 10,
                    "failed": 0,
                    "request_throughput": 50.0,
                    "output_throughput": 100.0,
                    "input_lens": [1024] * 10,
                    "output_lens": [1024] * 10,
                }),
                encoding="utf-8",
            )
            (raw_dir / "opt_c1.json").write_text(
                json.dumps({
                    "completed": 10,
                    "failed": 0,
                    "request_throughput": 75.0,
                    "output_throughput": 150.0,
                    "input_lens": [1024] * 10,
                    "output_lens": [1024] * 10,
                }),
                encoding="utf-8",
            )

            manifest = {
                "baseline": [{
                    "regime": "1k/1k",
                    "concurrency": 1,
                    "output_tps": 100.0,
                    "raw_output_path": "raw/base_c1.json",
                }],
                "optimized": [{
                    "regime":
                    "1k/1k",
                    "concurrency":
                    1,
                    "output_tps":
                    150.0,
                    "raw_output_path":
                    "/logs/artifacts/raw/opt_c1.json",
                }],
            }
            manifest_path = manifest_dir / "benchmark_results.json"
            manifest_path.write_text(json.dumps(manifest, indent=2),
                                     encoding="utf-8")

            # Manifest should be discovered
            found = speedup_mod.find_manifest_file(root)
            self.assertEqual(found, manifest_path)

            # Runs should be verified, not inferred
            discovery = speedup_mod.discover_or_verify_runs(root)
            self.assertFalse(discovery.inferred)
            self.assertEqual(len(discovery.baseline_runs), 1)
            self.assertEqual(len(discovery.optimized_runs), 1)
            self.assertEqual(discovery.verification_errors, [])

            # Speedup calculation works
            res = speedup_mod.DeterministicSpeedupEvaluator(
                discovery.baseline_runs,
                discovery.optimized_runs,
                inferred=discovery.inferred,
            ).evaluate(target_regimes=["1k/1k"])
            self.assertAlmostEqual(res["final_speedup"], 1.5, places=4)

    def test_root_tpu_agent_suite_path_resolution_and_permission_safety(
            self) -> None:
        """Verifies that /root/.tpu-agent-suite/... container paths resolve safely without PermissionError."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Harbor downloaded trial layout with artifacts/logs/artifacts
            base_rel = "02_baseline_bench/raw/1k1k_c1/isl1024_osl1024_c1_r1.json"
            opt_rel = "03_optimized_bench/raw/1k1k_c1/isl1024_osl1024_c1_r1.json"

            base_file = root / "artifacts" / "logs" / "artifacts" / base_rel
            opt_file = root / "artifacts" / "logs" / "artifacts" / opt_rel
            base_file.parent.mkdir(parents=True, exist_ok=True)
            opt_file.parent.mkdir(parents=True, exist_ok=True)

            base_file.write_text(
                json.dumps({
                    "completed": 10,
                    "failed": 0,
                    "request_throughput": 50.0,
                    "output_throughput": 100.0,
                    "input_lens": [1024] * 10,
                    "output_lens": [1024] * 10,
                }),
                encoding="utf-8",
            )
            opt_file.write_text(
                json.dumps({
                    "completed": 10,
                    "failed": 0,
                    "request_throughput": 80.0,
                    "output_throughput": 160.0,
                    "input_lens": [1024] * 10,
                    "output_lens": [1024] * 10,
                }),
                encoding="utf-8",
            )

            manifest_dir = root / "artifacts" / "logs" / "artifacts"
            manifest = {
                "baseline": [{
                    "regime":
                    "1k/1k",
                    "concurrency":
                    1,
                    "output_tps":
                    100.0,
                    "raw_output_path":
                    f"/root/.tpu-agent-suite/runs/20261020T142625Z-c4c5a7a82c36/artifacts/{base_rel}",
                }],
                "optimized": [{
                    "regime":
                    "1k/1k",
                    "concurrency":
                    1,
                    "output_tps":
                    160.0,
                    "raw_output_path":
                    f"/root/.tpu-agent-suite/runs/20261020T142625Z-c4c5a7a82c36/artifacts/{opt_rel}",
                }],
            }
            manifest_path = manifest_dir / "benchmark_results.json"
            manifest_path.write_text(json.dumps(manifest, indent=2),
                                     encoding="utf-8")

            # 1. Direct resolution test
            resolved_base = speedup_mod.resolve_raw_output_path(
                manifest["baseline"][0]["raw_output_path"],
                manifest_path,
                root,
            )
            self.assertEqual(resolved_base, base_file)

            # 2. Manifest discovery and verification
            discovery = speedup_mod.discover_or_verify_runs(root)
            self.assertFalse(discovery.inferred)
            self.assertEqual(len(discovery.baseline_runs), 1)
            self.assertEqual(len(discovery.optimized_runs), 1)
            self.assertEqual(discovery.verification_errors, [])

            # 3. Deterministic speedup evaluation
            speedup = speedup_mod.tpu_speedup(root)
            self.assertAlmostEqual(speedup, 1.6, places=4)

    def test_root_tpu_agent_suite_missing_file_no_permission_error(
            self) -> None:
        """Verifies that missing /root/.tpu-agent-suite/... files do not raise PermissionError."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_dir = root / "artifacts" / "logs" / "artifacts"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "baseline": [{
                    "regime":
                    "1k/1k",
                    "concurrency":
                    1,
                    "output_tps":
                    100.0,
                    "raw_output_path":
                    "/root/.tpu-agent-suite/runs/xyz/artifacts/nonexistent.json",
                }],
            }
            manifest_path = manifest_dir / "benchmark_results.json"
            manifest_path.write_text(json.dumps(manifest, indent=2),
                                     encoding="utf-8")

            # Should not raise PermissionError; should gracefully record verification error
            discovery = speedup_mod.discover_or_verify_runs(root)
            self.assertEqual(len(discovery.baseline_runs), 0)
            self.assertTrue(
                any("Raw output file not found" in err
                    for err in discovery.verification_errors))
            speedup = speedup_mod.tpu_speedup(root)
            self.assertEqual(speedup, 0.0)

    def test_specialist_manifest_two_levels_deep_resolved_with_parent_paths(
            self) -> None:
        """Verifies that specialist manifests 2 levels deep are discovered and raw paths resolved via parent roots."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Specialist manifest 2 levels deep under artifacts: tpu-model-onboarding/inference/
            spec_dir = root / "artifacts" / "tpu-model-onboarding" / "inference"
            spec_dir.mkdir(parents=True, exist_ok=True)

            raw_dir = root / "artifacts" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "base.json").write_text(
                json.dumps({
                    "completed": 10,
                    "failed": 0,
                    "request_throughput": 50.0,
                    "output_throughput": 100.0,
                    "input_lens": [1024] * 10,
                    "output_lens": [1024] * 10,
                }),
                encoding="utf-8",
            )
            (raw_dir / "opt.json").write_text(
                json.dumps({
                    "completed": 10,
                    "failed": 0,
                    "request_throughput": 75.0,
                    "output_throughput": 150.0,
                    "input_lens": [1024] * 10,
                    "output_lens": [1024] * 10,
                }),
                encoding="utf-8",
            )

            manifest = {
                "baseline": [{
                    "regime": "1k/1k",
                    "concurrency": 1,
                    "output_tps": 100.0,
                    "raw_output_path": "raw/base.json",
                }],
                "optimized": [{
                    "regime":
                    "1k/1k",
                    "concurrency":
                    1,
                    "output_tps":
                    150.0,
                    "raw_output_path":
                    "/logs/artifacts/raw/opt.json",
                }],
            }
            manifest_path = spec_dir / "benchmark_results.json"
            manifest_path.write_text(json.dumps(manifest, indent=2),
                                     encoding="utf-8")

            # 1. Manifest 2 levels deep should be discovered
            found = speedup_mod.find_manifest_file(root)
            self.assertEqual(found, manifest_path)

            # 2. Raw outputs should be resolved via manifest_dir.parent.parent or search roots
            discovery = speedup_mod.discover_or_verify_runs(root)
            self.assertFalse(discovery.inferred)
            self.assertEqual(len(discovery.baseline_runs), 1)
            self.assertEqual(len(discovery.optimized_runs), 1)
            self.assertEqual(discovery.verification_errors, [])

            # 3. Speedup calculation succeeds
            speedup = speedup_mod.tpu_speedup(root)
            self.assertAlmostEqual(speedup, 1.5, places=4)

    def test_search_stays_within_given_root(self) -> None:
        """Discovery searches strictly under the given root without climbing to parent directories."""
        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = Path(tmp) / "trial"
            ws_dir = trial_dir / "workspace"
            ws_dir.mkdir(parents=True)
            self._write_raw_json(
                trial_dir / "artifacts" / "raw_benchmarks" / "baseline" /
                "b.json",
                output_tps=100.0,
            )
            self._write_raw_json(
                trial_dir / "artifacts" / "raw_benchmarks" / "optimized" /
                "o.json",
                output_tps=150.0,
            )

            # Passing workspace/ does not climb up to sibling artifacts/
            ws_base, ws_opt, _ = speedup_mod.auto_discover_runs(ws_dir)
            self.assertEqual(ws_base, [])
            self.assertEqual(ws_opt, [])

            # Passing trial_dir searches downward into artifacts/
            self.assertAlmostEqual(speedup_mod.tpu_speedup(trial_dir), 1.5)

    def test_artifacts_preferred_with_workspace_fallback(self) -> None:
        """Uses artifacts runs first and falls back to workspace when artifacts has none."""
        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = Path(tmp) / "trial"
            art_dir = trial_dir / "artifacts"
            ws_dir = trial_dir / "workspace"
            art_dir.mkdir(parents=True)

            # 1. Empty artifacts/ -> falls back to workspace/ (1.2x)
            self._write_raw_json(
                ws_dir / "raw_benchmarks" / "baseline" / "b.json",
                output_tps=100.0,
            )
            self._write_raw_json(
                ws_dir / "raw_benchmarks" / "optimized" / "o.json",
                output_tps=120.0,
            )
            self.assertAlmostEqual(speedup_mod.tpu_speedup(trial_dir), 1.2)

            # 2. Populated artifacts/ -> takes priority over workspace/ (1.5x)
            self._write_raw_json(
                art_dir / "raw_benchmarks" / "baseline" / "b.json",
                output_tps=100.0,
            )
            self._write_raw_json(
                art_dir / "raw_benchmarks" / "optimized" / "o.json",
                output_tps=150.0,
            )
            self.assertAlmostEqual(speedup_mod.tpu_speedup(trial_dir), 1.5)


if __name__ == "__main__":
    unittest.main()
