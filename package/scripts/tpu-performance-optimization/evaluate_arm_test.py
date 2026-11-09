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
"""Unit tests for evaluate_arm.py and protocol compliance."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
# The directory holding the framework package: the package's own in a build,
# the repository root in the repository.
for _root in (_SCRIPTS_DIR.parents[1], _SCRIPTS_DIR.parents[2]):
    if (_root / "framework").is_dir():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break

from framework import jsonio, study, utils  # noqa: E402

try:
    from . import evaluate_arm  # noqa: E402
except ImportError:
    _spec = importlib.util.spec_from_file_location(
        "evaluate_arm", _SCRIPTS_DIR / "evaluate_arm.py")
    evaluate_arm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(evaluate_arm)


class TestEvaluateArm(unittest.TestCase):

    def setUp(self):
        self.protocol = {
            "primary_metric": {
                "name": "tokens_per_s",
                "direction": "maximize",
                "unit": "tok/s",
            },
            "measurement_cell": {
                "topology": "v6e-2x2x1",
            },
            "replicate_count":
            3,
            "noise_floor_pct":
            0.5,
            "stopping_rule": {
                "max_rounds": 10,
            },
            "admissibility_gates": [
                {
                    "name": "eval_loss",
                    "metric": "eval_loss",
                    "comparator": "lte",
                    "bound": 3.20,
                    "same_execution": True,
                },
            ],
            "mode":
            "auto",
        }
        self.assertEqual(study.validate_study_protocol(self.protocol), [])
        self.baseline = {
            "metric": {
                "name": "tokens_per_s",
                "value": 1000.0,
                "unit": "tok/s",
                "replicates": 3,
                "provenance": "measured",
            }
        }

    def test_win_exceeds_noise_floor_and_passes_gates(self):
        result = evaluate_arm.evaluate_arm(
            arm="a1",
            execution_id="job-01",
            replicates=[1020.0, 1025.0,
                        1022.0],  # median 1022.0 (+2.2% > 0.5%)
            protocol=self.protocol,
            baseline=self.baseline,
            gate_measurements=[{
                "name": "eval_loss",
                "value": 3.10,
                "execution_id": "job-01",
            }],
        )
        self.assertEqual(result["verdict"], "win")
        self.assertEqual(result["metric"]["replicates"], 3)
        self.assertEqual(result["metric"]["provenance"], "measured")
        self.assertTrue(result["metric"]["delta_pct"] > 0.5)

        # Invariant check against framework study validator
        record = {
            "arms": [{
                "arm": "a1",
                "kind": "kernel-block",
                "status": "decided",
                "claim": "fuse norm",
                "result": evaluate_arm.to_round_result(result),
            }]
        }
        arm_result_schema = jsonio.load_json(_SCRIPTS_DIR.parents[1] /
                                             "schemas" / "arm-result.json")
        self.assertEqual(study.validate_instance(result, arm_result_schema),
                         [])
        errors = (
            study.validate_instance(record, study.load_study_round_schema()) +
            study.check_round_against_protocol(record, self.protocol))
        self.assertEqual(errors, [])

    def test_tie_within_noise_floor(self):
        result = evaluate_arm.evaluate_arm(
            arm="a2",
            execution_id="job-02",
            replicates=[1002.0, 1003.0,
                        1004.0],  # median 1003.0 (+0.3% <= 0.5%)
            protocol=self.protocol,
            baseline=self.baseline,
            gate_measurements=[{
                "name": "eval_loss",
                "value": 3.10,
                "execution_id": "job-02",
            }],
        )
        self.assertEqual(result["verdict"], "tie")

    def test_loss_below_noise_floor(self):
        result = evaluate_arm.evaluate_arm(
            arm="a3",
            execution_id="job-03",
            replicates=[980.0, 982.0, 981.0],  # median 981.0 (-1.9%)
            protocol=self.protocol,
            baseline=self.baseline,
            gate_measurements=[{
                "name": "eval_loss",
                "value": 3.10,
                "execution_id": "job-03",
            }],
        )
        self.assertEqual(result["verdict"], "loss")

    def test_invalid_when_gate_fails(self):
        # A configuration that is faster on the primary metric but breaks an admissibility gate is inadmissible ('invalid').
        result = evaluate_arm.evaluate_arm(
            arm="a4",
            execution_id="job-04",
            replicates=[1050.0, 1055.0, 1052.0],  # +5.2% win candidate
            protocol=self.protocol,
            baseline=self.baseline,
            gate_measurements=[{
                "name": "eval_loss",
                "value": 3.45,  # FAILS gate lte 3.20
                "execution_id": "job-04",
            }],
        )
        self.assertEqual(result["verdict"], "invalid")
        self.assertEqual(result["gates"][0]["passed"], False)

    def test_invalid_when_same_execution_violated(self):
        result = evaluate_arm.evaluate_arm(
            arm="a5",
            execution_id="job-05",
            replicates=[1050.0, 1055.0, 1052.0],
            protocol=self.protocol,
            baseline=self.baseline,
            gate_measurements=[{
                "name": "eval_loss",
                "value": 3.10,
                "execution_id": "different-job-id",  # VIOLATES same_execution
            }],
        )
        self.assertEqual(result["verdict"], "invalid")
        self.assertEqual(result["gates"][0]["passed"], False)

    def test_omitted_execution_id_does_not_fabricate_colocation(self):
        # A gate measurement that omits execution_id must NOT be attributed to the metric's execution
        result = evaluate_arm.evaluate_arm(
            arm="a5b",
            execution_id="job-05b",
            replicates=[1050.0, 1055.0, 1052.0],
            protocol=self.protocol,
            baseline=self.baseline,
            gate_measurements=[{
                "name": "eval_loss",
                "value": 3.10,
                # execution_id is omitted!
            }],
        )
        self.assertEqual(result["verdict"], "invalid")
        self.assertEqual(result["gates"][0]["passed"], False)
        self.assertNotEqual(result["gates"][0]["execution_id"], "job-05b")

    def test_omitted_gate_value_does_not_pass_ceiling(self):
        # A gate omitting value must not default to 0.0 to pass ceiling gates
        ceiling_protocol = {
            "primary_metric": {
                "name": "tokens_per_s",
                "direction": "maximize",
                "unit": "tok/s",
            },
            "measurement_cell": {
                "topology": "v6e-2x2x1",
            },
            "replicate_count":
            3,
            "noise_floor_pct":
            0.5,
            "stopping_rule": {
                "max_rounds": 10,
            },
            "admissibility_gates": [{
                "name": "ttft",
                "metric": "ttft_ms",
                "comparator": "lt",
                "bound": 5000.0,
                "same_execution": True,
            }],
            "mode":
            "auto",
        }
        self.assertEqual(study.validate_study_protocol(ceiling_protocol), [])
        result = evaluate_arm.evaluate_arm(
            arm="a5c",
            execution_id="job-05c",
            replicates=[1050.0, 1055.0, 1052.0],
            protocol=ceiling_protocol,
            baseline=self.baseline,
            gate_measurements=[{
                "name": "ttft",
                "execution_id": "job-05c",
                # value is omitted!
            }],
        )
        self.assertEqual(result["verdict"], "invalid")
        self.assertEqual(result["gates"][0]["passed"], False)

    def test_comparator_enum_variants(self):
        # Verifies all comparators accepted by study-protocol.json enum: lt, lte, gt, gte
        self.assertTrue(evaluate_arm.compare_value(10.0, "lte", 10.0))
        self.assertTrue(evaluate_arm.compare_value(9.9, "lte", 10.0))
        self.assertFalse(evaluate_arm.compare_value(10.1, "lte", 10.0))

        self.assertTrue(evaluate_arm.compare_value(10.0, "gte", 10.0))
        self.assertTrue(evaluate_arm.compare_value(10.1, "gte", 10.0))
        self.assertFalse(evaluate_arm.compare_value(9.9, "gte", 10.0))

        self.assertTrue(evaluate_arm.compare_value(9.9, "lt", 10.0))
        self.assertFalse(evaluate_arm.compare_value(10.0, "lt", 10.0))

        self.assertTrue(evaluate_arm.compare_value(10.1, "gt", 10.0))
        self.assertFalse(evaluate_arm.compare_value(10.0, "gt", 10.0))

        with self.assertRaises(ValueError):
            evaluate_arm.compare_value(10.0, "invalid_comp", 10.0)

    def test_shipped_schema_example_evaluates_cleanly(self):
        schema = study.load_study_protocol_schema()
        example_protocol = schema["examples"][0]
        self.assertEqual(study.validate_study_protocol(example_protocol), [])

        baseline = {
            "metric": {
                "name": "output_tokens_per_s",
                "value": 37000.0,
                "unit": "tok/s",
                "replicates": 3,
                "provenance": "measured",
            }
        }
        # candidate is +4.277% improvement: 38582.5 vs 37000.0
        # ttft-ceiling is lt 5000, value is 3278.4 -> passes
        # accuracy-floor is gte 0.94, value is 0.96 -> passes
        result = evaluate_arm.evaluate_arm(
            arm="a-example",
            execution_id="j-example-01",
            replicates=[38580.0, 38585.0, 38582.5],
            protocol=example_protocol,
            baseline=baseline,
            gate_measurements=[
                {
                    "name": "ttft-ceiling",
                    "value": 3278.4,
                    "execution_id": "j-example-01",
                },
                {
                    "name": "accuracy-floor",
                    "value": 0.96,
                    "execution_id": "j-example-01",
                },
            ],
        )
        self.assertEqual(result["verdict"], "win")
        self.assertAlmostEqual(result["metric"]["delta_pct"], 4.277, places=2)
        self.assertTrue(all(g["passed"] for g in result["gates"]))

        round_record = {
            "arms": [{
                "arm": "a-example",
                "kind": "kernel-block",
                "status": "decided",
                "claim": "fused epilogue",
                "result": evaluate_arm.to_round_result(result),
            }]
        }
        arm_result_schema = jsonio.load_json(_SCRIPTS_DIR.parents[1] /
                                             "schemas" / "arm-result.json")
        self.assertEqual(study.validate_instance(result, arm_result_schema),
                         [])
        errors = (
            study.validate_instance(round_record,
                                    study.load_study_round_schema()) +
            study.check_round_against_protocol(round_record, example_protocol))
        self.assertEqual(errors, [])

    def test_minimize_direction(self):
        latency_protocol = {
            "primary_metric": {
                "name": "step_time_ms",
                "direction": "minimize",
                "unit": "ms",
            },
            "measurement_cell": {
                "topology": "v6e-2x2x1",
            },
            "replicate_count": 2,
            "noise_floor_pct": 1.0,
            "stopping_rule": {
                "max_rounds": 10,
            },
            "admissibility_gates": [],
            "mode": "auto",
        }
        self.assertEqual(study.validate_study_protocol(latency_protocol), [])
        latency_baseline = {
            "metric": {
                "name": "step_time_ms",
                "value": 50.0,
                "unit": "ms",
                "replicates": 2,
                "provenance": "measured",
            }
        }
        # 45.0ms is 10% faster than 50.0ms baseline
        result = evaluate_arm.evaluate_arm(
            arm="a6",
            execution_id="job-06",
            replicates=[45.0, 45.2],
            protocol=latency_protocol,
            baseline=latency_baseline,
        )
        self.assertEqual(result["verdict"], "win")
        self.assertAlmostEqual(result["metric"]["delta_pct"], 9.6, delta=0.5)

    def test_gate_self_asserted_pass_without_value_is_invalid(self):
        # Asserting passed=True without providing a numeric value cannot clear the gate.
        result = evaluate_arm.evaluate_arm(
            arm="a_asserted",
            execution_id="job-assert",
            replicates=[1050.0, 1055.0, 1052.0],
            protocol=self.protocol,
            baseline=self.baseline,
            gate_measurements=[{
                "name": "eval_loss",
                "execution_id": "job-assert",
                "passed": True,  # Asserting pass without numeric value!
            }],
        )
        self.assertEqual(result["verdict"], "invalid")
        self.assertEqual(result["gates"][0]["passed"], False)
        self.assertNotIn("value", result["gates"][0])

    def test_missing_bound_or_comparator_raises_key_error(self):
        protocol_bad_gate = {
            "primary_metric": {
                "name": "tokens_per_s",
                "direction": "maximize",
            },
            "admissibility_gates": [{
                "name": "loss_gate",
                # missing "comparator" and "bound"
            }],
        }
        with self.assertRaises(KeyError):
            evaluate_arm.evaluate_arm(
                arm="a1",
                execution_id="job-1",
                replicates=[100.0, 100.0, 100.0],
                protocol=protocol_bad_gate,
                baseline=self.baseline,
                gate_measurements=[{
                    "name": "loss_gate",
                    "value": 1.0,
                }],
            )

    def test_empty_replicates_returns_failed_without_fabricated_metric(self):
        result = evaluate_arm.evaluate_arm(
            arm="a_empty",
            execution_id="job-empty",
            replicates=[],
            protocol=self.protocol,
            baseline=self.baseline,
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["arm"], "a_empty")
        self.assertEqual(result["verdict"], "invalid")
        self.assertEqual(result["execution_id"], "job-empty")
        self.assertNotIn("metric", result)
        self.assertEqual(result["gates"][0]["passed"], False)

    def test_arm_result_matches_arm_result_schema(self):
        result = evaluate_arm.evaluate_arm(
            arm="a1",
            execution_id="job-01",
            replicates=[1020.0, 1025.0, 1022.0],
            protocol=self.protocol,
            baseline=self.baseline,
            gate_measurements=[{
                "name": "eval_loss",
                "value": 3.10,
                "execution_id": "job-01",
            }],
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["arm"], "a1")
        self.assertEqual(result["verdict"], "win")

        # Invariant check: to_round_result strips status and arm for study-round.json
        round_res = evaluate_arm.to_round_result(result)
        self.assertNotIn("status", round_res)
        self.assertNotIn("arm", round_res)
        self.assertIn("verdict", round_res)
        self.assertIn("execution_id", round_res)
        self.assertIn("metric", round_res)
        self.assertIn("gates", round_res)

    def test_compute_arm_fingerprint_canonicalizes_numbers_and_key_order(self):
        fp1 = evaluate_arm.compute_arm_fingerprint(
            kind="kernel-block",
            target_parameters={
                "block_m": 128,
                "block_n": 256.0,
                "stages": [1, 2.0]
            },
            claim="Fuse epilogue version A",
        )
        # Stringified numeric values ("256", "1.0") canonicalize identically to numbers.
        fp2 = evaluate_arm.compute_arm_fingerprint(
            kind="KERNEL-BLOCK ",
            target_parameters={
                "stages": ["1.0", 2],
                "block_n": "256",
                "block_m": 128.0
            },
            claim="Completely different wording for the same parameters",
        )
        self.assertEqual(fp1, fp2)

        # When target_parameters is empty, claim text distinguishes arms.
        fp_claim_a = evaluate_arm.compute_arm_fingerprint(
            kind="kernel-block",
            target_parameters={},
            claim="Fuse RMSNorm into QKV",
        )
        fp_claim_b = evaluate_arm.compute_arm_fingerprint(
            kind="kernel-block",
            target_parameters={},
            claim="Fuse SwiGLU into down_proj",
        )
        self.assertNotEqual(fp_claim_a, fp_claim_b)

    def test_find_duplicate_arm_detects_decided_and_in_flight_ignores_open(
            self):
        prior_arms = [
            {
                "arm": "a0-open",
                "kind": "kernel-block",
                "status": "open",
                "target_parameters": {
                    "chunk_size": 512
                },
            },
            {
                "arm": "a1-decided",
                "kind": "kernel-block",
                "status": "decided",
                "target_parameters": {
                    "chunk_size": 2048,
                    "num_warps": 4
                },
                "result": {
                    "verdict": "loss"
                },
            },
        ]
        # Matching an open arm is not a duplicate execution.
        self.assertIsNone(
            evaluate_arm.find_duplicate_arm(
                kind="kernel-block",
                target_parameters={"chunk_size": 512},
                prior_arms=prior_arms,
            ))
        # Matching a decided arm (even with float 2048.0) is flagged.
        dup = evaluate_arm.find_duplicate_arm(
            kind="kernel-block",
            target_parameters={
                "num_warps": 4,
                "chunk_size": 2048.0
            },
            prior_arms=prior_arms,
            current_arm_id="a2-retry",
        )
        self.assertIsNotNone(dup)
        self.assertEqual(dup["arm"], "a1-decided")

    def test_evaluate_arm_rejects_duplicate_hypothesis_as_invalid(self):
        prior_arms = [{
            "arm": "a1",
            "kind": "kernel-block",
            "status": "decided",
            "target_parameters": {
                "tile_size": 128
            },
            "result": {
                "verdict": "loss"
            },
        }]
        result = evaluate_arm.evaluate_arm(
            arm="a2",
            execution_id="job-01",
            replicates=[1100.0, 1110.0, 1105.0],
            protocol=self.protocol,
            baseline=self.baseline,
            gate_measurements=[{
                "name": "eval_loss",
                "value": 3.10,
                "execution_id": "job-01",
            }],
            kind="kernel-block",
            target_parameters={"tile_size": 128.0},
            prior_arms=prior_arms,
        )
        self.assertEqual(result["verdict"], "invalid")
        self.assertIn("duplicate of prior arm 'a1'", result["summary"])

    def test_check_harness_integrity_detects_tampering_and_invalidates_win(
            self):
        import hashlib
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            harness = pathlib.Path(tmpdir) / "benchmark_harness.py"
            harness.write_text("THRESHOLD = 0.99999\n")
            good_sha = hashlib.sha256(harness.read_bytes()).hexdigest()

            # Clean checksum passes and yields win.
            clean_res = evaluate_arm.evaluate_arm(
                arm="a-clean",
                execution_id="job-01",
                replicates=[1050.0, 1055.0, 1052.0],
                protocol=self.protocol,
                baseline=self.baseline,
                gate_measurements=[{
                    "name": "eval_loss",
                    "value": 3.10,
                    "execution_id": "job-01",
                }],
                protected_paths=[harness],
                expected_sha256={str(harness): good_sha},
            )
            self.assertEqual(clean_res["verdict"], "win")

            # Outside git without checksum: unverifiable file is treated as a violation.
            unverifiable_res = evaluate_arm.evaluate_arm(
                arm="a-unverifiable",
                execution_id="job-01",
                replicates=[1050.0, 1055.0, 1052.0],
                protocol=self.protocol,
                baseline=self.baseline,
                gate_measurements=[{
                    "name": "eval_loss",
                    "value": 3.10,
                    "execution_id": "job-01",
                }],
                protected_paths=[harness],
            )
            self.assertEqual(unverifiable_res["verdict"], "invalid")
            self.assertIn("unverifiable", unverifiable_res["summary"])

            # Tampering with the benchmark harness invalidates the win.
            harness.write_text("THRESHOLD = 0.50\n")
            tampered_res = evaluate_arm.evaluate_arm(
                arm="a-tampered",
                execution_id="job-01",
                replicates=[1500.0, 1510.0, 1505.0],
                protocol=self.protocol,
                baseline=self.baseline,
                gate_measurements=[{
                    "name": "eval_loss",
                    "value": 3.10,
                    "execution_id": "job-01",
                }],
                protected_paths=[harness],
                expected_sha256={str(harness): good_sha},
            )
            self.assertEqual(tampered_res["verdict"], "invalid")
            self.assertIn("checksum mismatch", tampered_res["summary"])

    def test_check_artifact_paths_enforces_run_dir_and_forbids_cwd_or_package_writes(
            self):
        run_id = "20260923T180000Z-abcdef012345"
        valid_art = f"~/.tpu-agent-suite/runs/{run_id}/artifacts/tpu-performance-optimization/a1/metrics.json"
        self.assertEqual(
            evaluate_arm.check_artifact_paths([valid_art], run_id=run_id),
            [],
        )

        # Reviewer bypasses: .. traversal, cross-run escapes, bare CWD, and package writes.
        bypass_paths = [
            f"~/.tpu-agent-suite/runs/{run_id}/artifacts/../../../../etc/passwd",
            f"~/.tpu-agent-suite/runs/{run_id}/artifacts/../../OTHERRUN/artifacts/x",
            "metrics.json",
            "../metrics.json",
            "scripts/tpu-performance-optimization/benchmark.py",
            "~/.tpu-agent-suite/runs/20260923T000000Z-000000000000/artifacts/out.json",
        ]
        violations = evaluate_arm.check_artifact_paths(bypass_paths,
                                                       run_id=run_id)
        self.assertEqual(len(violations), len(bypass_paths))
        self.assertTrue(any("path traversal" in v for v in violations))

        bad_res = evaluate_arm.evaluate_arm(
            arm="a1",
            execution_id="job-01",
            replicates=[1050.0, 1055.0, 1052.0],
            protocol=self.protocol,
            baseline=self.baseline,
            gate_measurements=[{
                "name": "eval_loss",
                "value": 3.10,
                "execution_id": "job-01",
            }],
            artifacts=[
                f"~/.tpu-agent-suite/runs/{run_id}/artifacts/../../../../etc/passwd"
            ],
            run_id=run_id,
        )
        self.assertEqual(bad_res["verdict"], "invalid")
        self.assertIn("path traversal", bad_res["summary"])

        good_res = evaluate_arm.evaluate_arm(
            arm="a1",
            execution_id="job-01",
            replicates=[1050.0, 1055.0, 1052.0],
            protocol=self.protocol,
            baseline=self.baseline,
            gate_measurements=[{
                "name": "eval_loss",
                "value": 3.10,
                "execution_id": "job-01",
            }],
            artifacts=[valid_art],
            run_id=run_id,
        )
        self.assertEqual(good_res["verdict"], "win")
        round_res = evaluate_arm.to_round_result(good_res)
        self.assertEqual(round_res["artifacts"], [valid_art])

    def test_an_artifact_inside_the_installed_package_is_refused(self):
        """Whatever the folder is called: the rule is the run, not a list of
        package folder names, which the layout by kind made stale."""
        run_id = "20260923T180000Z-abcdef012345"
        package = f"~/.tpu-agent-suite/packages/tpu-agent-suite@{utils.TPU_AGENT_SUITE_VERSION}"
        for inside in (
                f"{package}/agents/tpu-performance-optimization.md",
                f"{package}/skills/tpu-agent/SKILL.md",
                f"{package}/data/tpu-knowledge-base/pages/out.md",
                f"{package}/schemas/arm-result.json",
                f"{package}/scripts/tpu-performance-optimization/artifacts/x.json"
        ):
            violations = evaluate_arm.check_artifact_paths([inside],
                                                           run_id=run_id)
            self.assertEqual(len(violations), 1, inside)

    def test_a_folder_named_like_a_family_is_allowed_under_the_run(self):
        run_id = "20260923T180000Z-abcdef012345"
        for name in ("tpu-kernel-development", "tpu-performance-optimization",
                     "agents", "scripts"):
            path = f"~/.tpu-agent-suite/runs/{run_id}/artifacts/{name}/out.json"
            self.assertEqual(
                evaluate_arm.check_artifact_paths([path], run_id=run_id), [],
                path)

    def test_artifacts_without_a_run_id_are_refused(self):
        """Without the run id, `artifacts` in a path proves nothing."""
        path = "~/.tpu-agent-suite/runs/20260923T180000Z-abcdef012345/artifacts/x"
        for run_id in (None, ""):
            violations = evaluate_arm.check_artifact_paths([path, path],
                                                           run_id=run_id)
            self.assertEqual(len(violations), 2)
            self.assertIn("--run-id", violations[0])

    def test_artifact_containment_blocks_mid_path_dotdot_traversal_escapes(
            self):
        run_id = "20260923T180000Z-abcdef012345"
        escape_passwd = (
            f"/home/user/.tpu-agent-suite/runs/{run_id}/artifacts/../../../../etc/passwd"
        )
        escape_other_run = (
            f"/home/user/.tpu-agent-suite/runs/{run_id}/artifacts/../../OTHERRUN/artifacts/x"
        )
        escape_home_relative = (
            f"~/.tpu-agent-suite/runs/{run_id}/artifacts/../../OTHERRUN/artifacts/x"
        )
        for bad in (escape_passwd, escape_other_run, escape_home_relative,
                    "../report.md"):
            v = evaluate_arm.check_artifact_paths([bad], run_id=run_id)
            self.assertEqual(len(v), 1, f"Expected traversal block for {bad}")
            self.assertIn("traversal", v[0])

    def test_harness_integrity_fails_closed_when_untracked_without_checksum(
            self):
        import hashlib
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            untracked = pathlib.Path(tmpdir) / "outside_git_harness.py"
            untracked.write_text("THRESHOLD = 0.99999\n")
            sha = hashlib.sha256(untracked.read_bytes()).hexdigest()

            # Without expected_sha256 and outside git, check fails closed as unverifiable.
            v_unverifiable = evaluate_arm.check_harness_integrity([untracked])
            self.assertEqual(len(v_unverifiable), 1)
            self.assertIn("unverifiable", v_unverifiable[0])

            res_unverifiable = evaluate_arm.evaluate_arm(
                arm="a-unverifiable",
                execution_id="job-01",
                replicates=[1200.0, 1210.0, 1205.0],
                protocol=self.protocol,
                baseline=self.baseline,
                gate_measurements=[{
                    "name": "eval_loss",
                    "value": 3.10,
                    "execution_id": "job-01",
                }],
                protected_paths=[untracked],
            )
            self.assertEqual(res_unverifiable["verdict"], "invalid")
            self.assertIn("unverifiable", res_unverifiable["summary"])

            # Inline PATH:SHA256 syntax succeeds when file matches digest.
            v_inline_ok = evaluate_arm.check_harness_integrity(
                [f"{untracked}:{sha}"])
            self.assertEqual(v_inline_ok, [])

    def test_compute_arm_fingerprint_coerces_numeric_strings(self):
        fp_int = evaluate_arm.compute_arm_fingerprint("kernel-block", {
            "chunk_size": 128,
            "scale": 0.5
        })
        fp_str = evaluate_arm.compute_arm_fingerprint("kernel-block", {
            "chunk_size": "128",
            "scale": "0.50"
        })
        self.assertEqual(fp_int, fp_str)


if __name__ == "__main__":
    unittest.main()
