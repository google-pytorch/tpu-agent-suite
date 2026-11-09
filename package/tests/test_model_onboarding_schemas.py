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
"""Tests for TPU onboarding schemas and instruction schema bindings."""

from __future__ import annotations

import pathlib
import sys
import unittest

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework import jsonio, utils  # noqa: E402
from framework.builder import resolve_manifest_composition  # noqa: E402


class TestModelOnboardingSchemas(unittest.TestCase):
    """Verifies schema definitions and example validation for onboarding agents."""

    MODEL_ONBOARDING_SCHEMAS = [
        "schemas/onboarding-request.json",
        "schemas/onboarding-result.json",
        "schemas/unwiring-request.json",
        "schemas/unwiring-result.json",
        "schemas/bench-request.json",
    ]

    EXPECTED_AGENT_SCHEMAS = {
        "tpu-model-onboarding": (
            "schemas/onboarding-request.json",
            "schemas/onboarding-result.json",
        ),
        "tpu-unwiring": (
            "schemas/unwiring-request.json",
            "schemas/unwiring-result.json",
        ),
    }

    def test_all_schemas_exist_and_are_registered_in_package_json(self):
        # Every schemas/*.json is registered by the compiler; the manifest
        # lists none itself.
        manifest = resolve_manifest_composition(
            _PACKAGE_ROOT,
            jsonio.load_json(_PACKAGE_ROOT / "package.release.json"))
        registered_schemas = set(manifest.get("schemas", []))

        for schema_rel in self.MODEL_ONBOARDING_SCHEMAS:
            schema_path = _PACKAGE_ROOT / schema_rel
            self.assertTrue(schema_path.is_file(),
                            f"Missing schema file: {schema_rel}")
            self.assertIn(
                schema_rel, registered_schemas,
                f"Schema {schema_rel} not in package.release.json schemas")

    def test_schema_definitions_are_structurally_valid(self):
        for schema_rel in self.MODEL_ONBOARDING_SCHEMAS:
            schema = jsonio.load_json(_PACKAGE_ROOT / schema_rel)
            errors = utils.validate_schema_definition(schema)
            self.assertEqual(
                errors, [],
                f"Invalid schema structure for {schema_rel}: {errors}")

    def test_schema_examples_validate_against_their_schemas(self):
        for schema_rel in self.MODEL_ONBOARDING_SCHEMAS:
            schema = jsonio.load_json(_PACKAGE_ROOT / schema_rel)
            examples = schema.get("examples", [])
            self.assertTrue(
                len(examples) > 0,
                f"Schema {schema_rel} should provide at least one example")
            for i, example in enumerate(examples):
                errors = utils.validate_instance(example, schema)
                self.assertEqual(
                    errors, [],
                    f"Example #{i} failed validation in {schema_rel}: {errors}"
                )

    def test_model_onboarding_agents_bind_expected_schemas(self):
        manifest = resolve_manifest_composition(
            _PACKAGE_ROOT,
            jsonio.load_json(_PACKAGE_ROOT / "package.release.json"))
        agent_map = {a["name"]: a for a in manifest["agents"]}

        for agent_name, (
                expected_accepts,
                expected_produces) in self.EXPECTED_AGENT_SCHEMAS.items():
            self.assertIn(agent_name, agent_map,
                          f"Agent {agent_name} missing from manifest")
            agent = agent_map[agent_name]
            self.assertEqual(
                agent.get("accepts"),
                expected_accepts,
                f"Agent {agent_name} accepts mismatch: expected {expected_accepts}, got {agent.get('accepts')}",
            )
            self.assertEqual(
                agent.get("produces"),
                expected_produces,
                f"Agent {agent_name} produces mismatch: expected {expected_produces}, got {agent.get('produces')}",
            )

    def test_model_onboarding_ships_only_the_expected_agents(self):
        """Workflows and steps are references; only these two become agents.

        Every agent is installed as its own subagent file, so a step document
        registered as an agent again would bring back the nesting this package
        was flattened to avoid.
        """
        manifest = resolve_manifest_composition(
            _PACKAGE_ROOT,
            jsonio.load_json(_PACKAGE_ROOT / "package.release.json"))
        agent_names = {a["name"] for a in manifest["agents"]}
        self.assertLessEqual(set(self.EXPECTED_AGENT_SCHEMAS), agent_names)
        step_documents = {
            p.stem
            for p in (_PACKAGE_ROOT / "references" /
                      "tpu-model-onboarding").glob("*.md")
        }
        self.assertTrue(step_documents)
        self.assertEqual(step_documents & agent_names, set())

    def test_schema_validation_rejects_malformed_payloads(self):
        schema = jsonio.load_json(_PACKAGE_ROOT / "schemas" /
                                  "onboarding-request.json")
        # Empty payload still fails: prompt is required.
        errors = utils.validate_instance({}, schema)
        self.assertTrue(
            len(errors) > 0,
            "Empty payload should fail required properties validation")
        # Invalid mode enum is still rejected.
        errors = utils.validate_instance({
            "prompt": "x",
            "mode": "train"
        }, schema)
        self.assertTrue(len(errors) > 0, "Invalid mode should be rejected")
        # The optimization round's candidate measurement has its own mode.
        self.assertEqual(
            utils.validate_instance(
                {
                    "prompt": "x",
                    "accelerator_type": "tpu7x",
                    "mode": "candidate"
                }, schema), [])

        # Unwiring needs the model, the preflight report and the target.
        schema = jsonio.load_json(_PACKAGE_ROOT / "schemas" /
                                  "unwiring-request.json")
        complete = {
            "model_id": "state-spaces/mamba-2.8b-hf",
            "cuda_kernels_report": "/tmp/cuda_kernels_report.json",
            "target": "server",
        }
        self.assertEqual(utils.validate_instance(complete, schema), [])
        for field in complete:
            payload = {k: v for k, v in complete.items() if k != field}
            self.assertTrue(utils.validate_instance(payload, schema),
                            f"Missing {field} should fail validation")
        self.assertTrue(
            utils.validate_instance(dict(complete, target="training"), schema),
            "Unknown target should be rejected")

    def test_model_onboarding_request_needs_the_accelerator_type(self):
        """The prompt can be plain text, but the TPU it targets is required:
        tpu-agent probes it (query understanding) before delegating."""
        schema = jsonio.load_json(_PACKAGE_ROOT / "schemas" /
                                  "onboarding-request.json")
        for payload in (
            {
                "prompt": "Help me onboard Llama-3 to TPU",
                "accelerator_type": "tpu7x"
            },
            {
                "prompt": "Run Llama 3 8B on TPU",
                "accelerator_type": "tpu-v6e-slice",
                "run_id": "20260924T163000Z-abc"
            },
                # No mode: tpu-model-onboarding must be able to ask the user.
            {
                "prompt": "Run Llama 3 8B on TPU",
                "accelerator_type": "tpu7x",
                "model": {
                    "model_id": "meta-llama/Llama-3.1-8B"
                }
            },
        ):
            self.assertEqual(utils.validate_instance(payload, schema), [],
                             f"Should accept {payload}")
        for payload in ({
                "prompt": "Help me onboard Llama-3 to TPU"
        }, {
                "prompt": "Help me onboard Llama-3 to TPU",
                "accelerator_type": ""
        }):
            self.assertTrue(utils.validate_instance(payload, schema),
                            f"Should reject {payload}")

    FAILURE_RESULTS = {
        "unwiring-result.json": [
            {
                "status": "blocked",
                "error": "selective_scan_cuda still imported after 5 attempts"
            },
            {
                "status": "failed",
                "error": "no TPU reachable"
            },
        ],
        "onboarding-result.json": [{
            "status": "blocked_on_cuda_kernel",
            "summary": "unwiring blocked",
            "unwiring_status": "blocked",
            "parity_verdict": "not_run",
        }],
    }

    def test_result_schemas_accept_failure_without_deliverables(self):
        """A failed run has no results file, script, manifest or metrics to report."""
        for schema_rel, payloads in self.FAILURE_RESULTS.items():
            schema = jsonio.load_json(_PACKAGE_ROOT / "schemas" / schema_rel)
            for payload in payloads:
                self.assertEqual(utils.validate_instance(payload, schema), [],
                                 f"{schema_rel} should accept {payload}")

    def test_result_schemas_still_type_check_deliverables(self):
        """Optional deliverables are still validated when present."""
        for schema_rel, payload in (
            ("unwiring-result.json", {
                "status": "ok",
                "patched_modules": "backbone.layers.0.mixer"
            }),
            ("unwiring-result.json", {
                "status": "ok",
                "attempts": -1
            }),
            ("onboarding-result.json", {
                "status": "success",
                "summary": "ok",
                "parity_verdict": "LOOKS_FINE"
            }),
        ):
            schema = jsonio.load_json(_PACKAGE_ROOT / "schemas" / schema_rel)
            self.assertTrue(utils.validate_instance(payload, schema),
                            f"{schema_rel} should reject {payload}")

    def test_model_onboarding_result_hands_off_the_serving_configuration(self):
        """The manifest and recipe paths copy into tpu-kernel-development's request."""
        result = jsonio.load_json(_PACKAGE_ROOT / "schemas" /
                                  "onboarding-result.json")
        request = jsonio.load_json(_PACKAGE_ROOT / "schemas" /
                                   "kernel-request.json")
        inference = "/runs/r/artifacts/tpu-model-onboarding/inference"
        output_files = {
            "deployments_json": f"{inference}/deployments.json",
            "launch_recipe": f"{inference}/launch_vllm.sh",
        }
        self.assertEqual(
            utils.validate_instance(
                {
                    "status": "success",
                    "summary": "serving",
                    "output_files": output_files
                }, result), [])
        self.assertEqual(
            utils.validate_instance(output_files,
                                    request["properties"]["serving_config"]),
            [])

    def test_model_onboarding_is_told_to_return_the_serving_configuration(
            self):
        text = " ".join((_PACKAGE_ROOT / "agents" /
                         "tpu-model-onboarding.md").read_text().split())
        self.assertIn("set `output_files.deployments_json`", text)
        self.assertIn("`output_files.launch_recipe`", text)

    def test_checkpoint_bucket_is_an_optional_gs_url(self):
        schema = jsonio.load_json(_PACKAGE_ROOT / "schemas" /
                                  "onboarding-request.json")
        base = {
            "prompt": "Run Qwen/Qwen3-8B on TPU",
            "accelerator_type": "tpu7x"
        }
        self.assertEqual(utils.validate_instance(base, schema), [])
        self.assertEqual(
            utils.validate_instance(
                dict(base, checkpoint_bucket="gs://my-ckpts"), schema), [])
        self.assertTrue(
            utils.validate_instance(dict(base, checkpoint_bucket="/mnt/ckpts"),
                                    schema),
            "A checkpoint_bucket that is not gs:// should be rejected")

    def test_onboarding_stages_the_checkpoint_and_records_the_benchmark(self):
        text = " ".join(
            (_PACKAGE_ROOT / "references" / "tpu-model-onboarding" /
             "torch-tpu-inference.md").read_text().split())
        self.assertIn("Step 0.2b: Stage the checkpoint", text)
        # The bucket must be one the gke_job pods can read (compute's pod
        # token is limited to the profile's staging_bucket).
        self.assertIn("`staging_bucket`", text)
        self.assertIn("`pod_service_account`", text)
        # A lifecycle rule can age the copy out before a later job loads it.
        self.assertIn("right before any later job that loads the model", text)
        self.assertIn("`matchesPrefix`", text)
        self.assertIn("{{script:tpu-model-onboarding/stage_checkpoint.py}}",
                      text)
        self.assertIn("return `needs_input`", text)
        self.assertIn(
            "`{{runtime}} compute setup plan --fields staging_bucket`", text)
        self.assertIn("Step 5c: Record the benchmark", text)
        self.assertIn("{{schema:bench-request.json}}", text)
        self.assertIn("`output_files.bench_request`", text)

    def test_bench_request_hands_off_through_the_result(self):
        """A bench_request path fits output_files, and each route validates."""
        result = jsonio.load_json(_PACKAGE_ROOT / "schemas" /
                                  "onboarding-result.json")
        bench = jsonio.load_json(_PACKAGE_ROOT / "schemas" /
                                 "bench-request.json")
        inference = "/runs/r/artifacts/tpu-model-onboarding/inference"
        self.assertEqual(
            utils.validate_instance(
                {
                    "status": "success",
                    "summary": "served and measured",
                    "output_files": {
                        "bench_request": f"{inference}/bench_request.json",
                        "results_json":
                        f"{inference}/serving_bench/summary.json",
                    },
                }, result), [])
        minimal = {
            "route": "port_forward",
            "client_args": ["--engine", "vllm", "--isl-osl", "1024:128"],
            "server": {
                "job_id": "job-1"
            },
            "results_json": f"{inference}/serving_bench/summary.json",
        }
        self.assertEqual(utils.validate_instance(minimal, bench), [])
        for field in ("route", "client_args", "server", "results_json"):
            payload = {k: v for k, v in minimal.items() if k != field}
            self.assertTrue(utils.validate_instance(payload, bench),
                            f"Missing {field} should fail validation")
        self.assertTrue(
            utils.validate_instance(dict(minimal, route="yeet"), bench),
            "An unknown route should be rejected")

    def test_model_onboarding_result_output_files_accepts_unnamed_paths(self):
        """output_files takes extra named paths, as long as each is a string."""
        schema = jsonio.load_json(_PACKAGE_ROOT / "schemas" /
                                  "onboarding-result.json")
        inference = "/runs/r/artifacts/tpu-model-onboarding/inference"
        base = {"status": "success", "summary": "ok"}
        self.assertEqual(
            utils.validate_instance(
                dict(base,
                     output_files={
                         "benchmark_report": f"{inference}/report.md",
                         "server_log": f"{inference}/server.log",
                     }), schema), [])
        for bad in (
            {
                "server_log": 1
            },
            {
                "server_log": [f"{inference}/server.log"]
            },
            {
                "server_log": {
                    "path": f"{inference}/server.log"
                }
            },
                # Declared names keep their own type check.
            {
                "benchmark_report": 1
            },
        ):
            self.assertTrue(
                utils.validate_instance(dict(base, output_files=bad), schema),
                f"Should reject output_files={bad}")

    def test_model_onboarding_result_artifacts_is_path_list(self):
        """Named paths live in output_files; artifacts is a path string list."""
        schema = jsonio.load_json(_PACKAGE_ROOT / "schemas" /
                                  "onboarding-result.json")
        items = schema["properties"]["artifacts"]["items"]
        self.assertEqual(items.get("type"), "string")

        artifact_path = "artifacts/tpu-model-onboarding/benchmark_report.md"
        for payload in (
            {
                "status": "success",
                "summary": "ok",
                "output_files": {
                    "benchmark_report": artifact_path
                },
            },
            {
                "status": "success",
                "summary": "ok",
                "artifacts": [artifact_path],
            },
            {
                "status": "failed",
                "summary": "preflight failed",
                "error": "boom",
            },
        ):
            self.assertEqual(utils.validate_instance(payload, schema), [],
                             f"Should accept {payload}")
        # An object under `artifacts` is rejected.
        self.assertTrue(
            utils.validate_instance(
                {
                    "status": "success",
                    "summary": "ok",
                    "artifacts": [{
                        "file": "benchmark_report.md"
                    }],
                }, schema))


if __name__ == "__main__":
    unittest.main()
