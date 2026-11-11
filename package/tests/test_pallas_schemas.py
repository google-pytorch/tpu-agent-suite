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
"""Tests for the tpu-pallas request and result contracts."""

from __future__ import annotations

import pathlib
import sys
import unittest

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework import agent_contracts, compiler, jsonio, utils  # noqa: E402

RUN_ID = "20260921T120000Z-abcdef012345"
REQUEST = "schemas/pallas-request.json"
RESULT = "schemas/pallas-result.json"


def _load(rel: str) -> dict:
    return jsonio.load_json(_PACKAGE_ROOT / rel)


def _errors(payload: dict, schema: dict) -> list[str]:
    """Both halves the SubagentStop hook applies: shape, then status."""
    return (utils.validate_instance(payload, schema) +
            agent_contracts.check_status_payload(schema, payload))


class TestPallasSchemas(unittest.TestCase):
    """The pallas pair and the agent that binds it."""

    def test_schemas_are_registered_and_valid(self):
        registered = set(compiler.load_package(_PACKAGE_ROOT)["schemas"])
        for rel in (REQUEST, RESULT):
            with self.subTest(schema=rel):
                self.assertIn(rel, registered)
                self.assertEqual(utils.validate_schema_definition(_load(rel)),
                                 [])

    def test_examples_validate(self):
        for rel in (REQUEST, RESULT):
            schema = _load(rel)
            examples = schema.get("examples", [])
            self.assertTrue(examples, f"{rel} has no examples")
            for i, example in enumerate(examples):
                with self.subTest(schema=rel, example=i):
                    self.assertEqual(_errors(example, schema), [])

    def test_request_without_stack_head_fails(self):
        schema = _load(REQUEST)
        request = dict(schema["examples"][0])
        del request["stack_head"]
        self.assertNotEqual(utils.validate_instance(request, schema), [])

    def test_request_without_checkout_fails(self):
        """The hill-climb always has the campaign repo; nothing is exported."""
        schema = _load(REQUEST)
        request = dict(schema["examples"][0])
        del request["checkout"]
        self.assertNotEqual(utils.validate_instance(request, schema), [])

    def test_patched_without_plugin_or_applies_to_fails(self):
        schema = _load(RESULT)
        patched = next(e for e in schema["examples"]
                       if e["status"] == "patched")
        for field in ("plugin", "applies_to"):
            with self.subTest(field=field):
                payload = {k: v for k, v in patched.items() if k != field}
                self.assertNotEqual(_errors(payload, schema), [])

    def test_a_base_mismatch_is_failed_not_a_loss(self):
        """Apply exit 4 means the ref is off the image's base: no verdict."""
        schema = _load(RESULT)
        self.assertEqual(
            _errors(
                {
                    "status": "failed",
                    "summary": "kernel commit does not apply on the image",
                    "failure_reason": "base mismatch: apply exit 4",
                    "run_id": RUN_ID,
                }, schema), [])
        patched = next(e for e in schema["examples"]
                       if e["status"] == "patched")
        applies_to = dict(patched["applies_to"], exit_code=4)
        self.assertNotEqual(
            _errors(dict(patched, applies_to=applies_to), schema), [])

    def test_missing_inputs_are_invalid_input(self):
        """The agent cannot ask, so a missing input is the caller's omission."""
        schema = _load(RESULT)
        self.assertNotIn("needs_input", schema["properties"]["status"]["enum"])
        self.assertNotEqual(
            _errors(
                {
                    "status": "invalid_input",
                    "summary": "no stack head",
                    "run_id": RUN_ID,
                }, schema), [])


if __name__ == "__main__":
    unittest.main()
