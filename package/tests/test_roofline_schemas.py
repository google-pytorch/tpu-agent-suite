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
"""Tests for the tpu-roofline request and result contract."""

from __future__ import annotations

import pathlib
import sys
import unittest

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework import agent_contracts, compiler, jsonio, utils  # noqa: E402
from framework.builder import package_discovery  # noqa: E402

RUN_ID = "20260921T120000Z-abcdef012345"
REQUEST = "schemas/roofline-request.json"
RESULT = "schemas/roofline-result.json"


class TestRooflineSchemas(unittest.TestCase):
    """The roofline-request / roofline-result pair and who binds it."""

    @classmethod
    def setUpClass(cls):
        cls.request = jsonio.load_json(_PACKAGE_ROOT / REQUEST)
        cls.result = jsonio.load_json(_PACKAGE_ROOT / RESULT)

    def _result_errors(self, payload: dict) -> list[str]:
        """Both halves the SubagentStop hook applies: shape, then status."""
        return (utils.validate_instance(payload, self.result) +
                agent_contracts.check_status_payload(self.result, payload))

    def test_schemas_are_registered(self):
        registered = set(compiler.load_package(_PACKAGE_ROOT)["schemas"])
        for rel in (REQUEST, RESULT):
            with self.subTest(schema=rel):
                self.assertTrue((_PACKAGE_ROOT / rel).is_file())
                self.assertIn(rel, registered)

    def test_schema_definitions_are_structurally_valid(self):
        for rel, schema in ((REQUEST, self.request), (RESULT, self.result)):
            with self.subTest(schema=rel):
                self.assertEqual(utils.validate_schema_definition(schema), [])

    def test_examples_validate(self):
        for rel, schema in ((REQUEST, self.request), (RESULT, self.result)):
            examples = schema.get("examples", [])
            self.assertTrue(examples, f"{rel} has no examples")
            for i, example in enumerate(examples):
                with self.subTest(schema=rel, example=i):
                    self.assertEqual(utils.validate_instance(example, schema),
                                     [])
                    self.assertEqual(
                        agent_contracts.check_status_payload(schema, example),
                        [])

    def test_modeled_without_its_fields_fails(self):
        modeled = next(e for e in self.result["examples"]
                       if e["status"] == "modeled")
        owed = self.result["status_requires"]["modeled"]
        self.assertTrue(owed)
        for field in owed:
            with self.subTest(field=field):
                payload = {k: v for k, v in modeled.items() if k != field}
                self.assertNotEqual(self._result_errors(payload), [])

    def test_missing_inputs_are_invalid_input(self):
        """Agents cannot ask, so a missing choice is the caller's omission."""
        status = self.result["properties"]["status"]["enum"]
        self.assertNotIn("needs_input", status)
        self.assertNotEqual(
            self._result_errors({
                "status": "invalid_input",
                "summary": "workload not given",
                "run_id": RUN_ID,
            }), [])

    def test_request_carries_affected_rows_and_cell_shapes(self):
        """hill-climb prices a neighbour gap's rows at its plan's cells."""
        request = {
            "prompt":
            "Floor for the rows around MLA QK Rope.",
            "run_id":
            RUN_ID,
            "tpu": {
                "accelerator_type": "tpu7x-8",
                "topology": "2x2x2"
            },
            "rows": ["MLA: QK Rope"],
            "cells": [{
                "name": "decode-b64",
                "tokens": 64,
                "kv_len": 9216,
                "batch": 64,
            }],
        }
        self.assertEqual(utils.validate_instance(request, self.request), [])
        del request["tpu"]
        self.assertNotEqual(utils.validate_instance(request, self.request), [])

    def test_tpu_roofline_binds_the_pair_and_delegates_only_the_kb(self):
        agent = package_discovery.load_agent_document(
            _PACKAGE_ROOT / "agents/tpu-roofline.md", _PACKAGE_ROOT)
        self.assertEqual(agent.get("accepts"), REQUEST)
        self.assertEqual(agent.get("produces"), RESULT)
        self.assertEqual(agent.get("delegates"), ["tpu-knowledge-base"])


if __name__ == "__main__":
    unittest.main()
