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
"""Validation unit tests for tpu-hello-world request and result schemas."""

import pathlib
import sys
import unittest

# Ensure repository root directory is in sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import jsonio, utils  # noqa: E402
from framework.builder import package_discovery, package_loader  # noqa: E402

Path = pathlib.Path
PACKAGE_DIR = REPO_ROOT / "package"


class TestHelloWorldSchemas(unittest.TestCase):
    """Unit tests verifying tpu-hello-world schemas and contract definitions."""

    @classmethod
    def setUpClass(cls):
        cls.package_manifest = jsonio.load_json(PACKAGE_DIR /
                                                "package.release.json")
        cls.request_schema_path = PACKAGE_DIR / "schemas" / "hello-world-request.json"
        cls.result_schema_path = PACKAGE_DIR / "schemas" / "hello-world-result.json"

        cls.request_schema = jsonio.load_json(cls.request_schema_path)
        cls.result_schema = jsonio.load_json(cls.result_schema_path)

    def test_schemas_exist_and_are_registered_in_package_json(self):
        """Schemas exist on disk and are registered in package.release.json schemas list."""
        self.assertTrue(
            self.request_schema_path.is_file(),
            "hello-world-request.json missing from schemas directory")
        self.assertTrue(
            self.result_schema_path.is_file(),
            "hello-world-result.json missing from schemas directory")

        # Every schemas/*.json is registered by the compiler; the manifest
        # lists none itself.
        manifest = package_loader.resolve_manifest_composition(
            PACKAGE_DIR, self.package_manifest)
        declared_schemas = set(manifest.get("schemas", []))
        self.assertIn("schemas/hello-world-request.json", declared_schemas)
        self.assertIn("schemas/hello-world-result.json", declared_schemas)

    def test_schema_definitions_are_structurally_valid(self):
        """Both schemas conform to JSON Schema Draft 2020-12 specifications."""
        req_errors = utils.validate_schema_definition(self.request_schema)
        self.assertEqual(
            req_errors, [],
            f"hello-world-request.json schema definition errors: {req_errors}")

        res_errors = utils.validate_schema_definition(self.result_schema)
        self.assertEqual(
            res_errors, [],
            f"hello-world-result.json schema definition errors: {res_errors}")

    def test_tpu_hello_world_instruction_contract(self):
        """tpu-hello-world agent frontmatter declares the new typed schemas."""
        agent_path = PACKAGE_DIR / "agents" / "tpu-hello-world.md"
        self.assertTrue(agent_path.is_file())

        agent = package_discovery.load_agent_document(agent_path, PACKAGE_DIR)
        self.assertEqual(agent.get("name"), "tpu-hello-world")
        self.assertEqual(agent.get("accepts"),
                         "schemas/hello-world-request.json")
        self.assertEqual(agent.get("produces"),
                         "schemas/hello-world-result.json")

    def test_request_schema_accepts_valid_payloads(self):
        """Request schema accepts prompt and optional execution settings."""
        # Minimal payload
        minimal = {"prompt": "Check TPU availability"}
        self.assertEqual(utils.validate_instance(minimal, self.request_schema),
                         [])

        # Full payload with execution settings
        full = {
            "prompt": "Run small matmul on TPU and verify parity against CPU",
            "run_id": "20260914T120000Z-000000000000",
            "cluster_cell": "us-central1-a",
            "timeout_seconds": 900,
        }
        self.assertEqual(utils.validate_instance(full, self.request_schema),
                         [])

        # Embedded examples in the schema itself must be valid
        examples = self.request_schema.get("examples", [])
        self.assertTrue(examples,
                        "Request schema must define at least one example")
        for example in examples:
            self.assertEqual(
                utils.validate_instance(example, self.request_schema), [])

    def test_request_schema_rejects_invalid_payloads(self):
        """Request schema rejects missing prompt, unknown keys, or invalid types."""
        # Missing required field 'prompt'
        self.assertTrue(
            utils.validate_instance(
                {"run_id": "20260914T120000Z-000000000000"},
                self.request_schema))

        # Additional unknown properties forbidden
        self.assertTrue(
            utils.validate_instance(
                {
                    "prompt": "test",
                    "unknown_field": "invalid"
                }, self.request_schema))

        # Invalid timeout_seconds type or value
        self.assertTrue(
            utils.validate_instance(
                {
                    "prompt": "test",
                    "timeout_seconds": "900"
                }, self.request_schema))
        self.assertTrue(
            utils.validate_instance({
                "prompt": "test",
                "timeout_seconds": 0
            }, self.request_schema))
        self.assertTrue(
            utils.validate_instance({
                "prompt": "test",
                "timeout_seconds": -10
            }, self.request_schema))

    def test_result_schema_accepts_completed_outcome(self):
        """Result schema accepts completed verification outcomes."""
        completed_payload = {
            "status":
            "completed",
            "device_verified":
            True,
            "device_type":
            "native-tpu",
            "parity_passed":
            True,
            "run_id":
            "20260914T120000Z-000000000000",
            "summary":
            "Computed on native-tpu and verified against CPU reference.",
        }
        self.assertEqual(
            utils.validate_instance(completed_payload, self.result_schema), [])

    def test_result_schema_accepts_failed_outcomes_with_diagnostic_fields(
            self):
        """Result schema accepts structured failure reasons and error messages."""
        failure_cases = [
            {
                "status": "failed",
                "device_verified": False,
                "device_type": "cpu-fallback",
                "parity_passed": False,
                "failure_reason": "no_device",
                "error_message":
                "No TPU runtime detected: falling back to CPU.",
            },
            {
                "status":
                "failed",
                "device_verified":
                True,
                "device_type":
                "native-tpu",
                "parity_passed":
                False,
                "failure_reason":
                "parity_mismatch",
                "error_message":
                "Numerical mismatch on native-tpu: max diff 1.2 exceeds tolerance 0.5",
            },
            {
                "status": "failed",
                "failure_reason": "dispatch_failed",
                "error_message": "Compute job pending timed out after 900s.",
            },
            {
                "status": "failed",
                "failure_reason": "credential_expired",
                "error_message": "GKE cluster credentials expired.",
            },
        ]
        for payload in failure_cases:
            with self.subTest(reason=payload["failure_reason"]):
                self.assertEqual(
                    utils.validate_instance(payload, self.result_schema), [])

    def test_result_schema_accepts_all_envelope_statuses(self):
        """Result schema supports all envelope pattern statuses."""
        for status in ("completed", "invalid_input", "needs_input", "failed"):
            with self.subTest(status=status):
                payload = {"status": status}
                self.assertEqual(
                    utils.validate_instance(payload, self.result_schema), [])

    def test_result_schema_rejects_invalid_payloads(self):
        """Result schema rejects missing status, invalid enum, extra fields, or type errors."""
        # Missing required field 'status'
        self.assertTrue(
            utils.validate_instance(
                {
                    "device_verified": True,
                    "device_type": "native-tpu"
                }, self.result_schema))

        # Invalid status enum
        self.assertTrue(
            utils.validate_instance({"status": "success"}, self.result_schema))
        self.assertTrue(
            utils.validate_instance({"status": "error"}, self.result_schema))

        # Additional unknown properties forbidden
        self.assertTrue(
            utils.validate_instance({
                "status": "completed",
                "extra_metric": 42
            }, self.result_schema))

        # Invalid field types
        self.assertTrue(
            utils.validate_instance(
                {
                    "status": "completed",
                    "device_verified": "yes"
                }, self.result_schema))
        self.assertTrue(
            utils.validate_instance({
                "status": "completed",
                "parity_passed": 1
            }, self.result_schema))

    def test_result_schema_declared_examples(self):
        """All declared examples in hello-world-result.json validate against the schema."""
        examples = self.result_schema.get("examples", [])
        self.assertTrue(examples,
                        "Result schema must define at least one example")
        for example in examples:
            self.assertEqual(
                utils.validate_instance(example, self.result_schema), [])


if __name__ == "__main__":
    unittest.main()
