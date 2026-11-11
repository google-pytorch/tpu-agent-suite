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
"""Tests for the review request and result contracts."""

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
REVIEW_REQUEST = "schemas/review-request.json"
REVIEW_RESULT = "schemas/review-result.json"
SCHEMAS = (REVIEW_REQUEST, REVIEW_RESULT)
RESULTS = (REVIEW_RESULT, )
OUTCOME = {REVIEW_RESULT: "reviewed"}
BINDINGS = {
    "tpu-critical-review": (REVIEW_REQUEST, REVIEW_RESULT),
}
# Leaves only: it sits at depth 2 or 3 under tpu-agent.
DELEGATES = {
    "tpu-critical-review": {"tpu-diagnostics", "tpu-code-stack"},
}


def _load(rel: str) -> dict:
    return jsonio.load_json(_PACKAGE_ROOT / rel)


def _errors(payload: dict, schema: dict) -> list[str]:
    """Both halves the SubagentStop hook applies: shape, then status."""
    return (utils.validate_instance(payload, schema) +
            agent_contracts.check_status_payload(schema, payload))


class TestReviewSchemas(unittest.TestCase):
    """The review pair and the agent that binds it."""

    def test_schemas_are_registered_and_valid(self):
        registered = set(compiler.load_package(_PACKAGE_ROOT)["schemas"])
        for rel in SCHEMAS:
            with self.subTest(schema=rel):
                self.assertIn(rel, registered)
                self.assertEqual(utils.validate_schema_definition(_load(rel)),
                                 [])

    def test_examples_validate(self):
        for rel in SCHEMAS:
            schema = _load(rel)
            examples = schema.get("examples", [])
            self.assertTrue(examples, f"{rel} has no examples")
            for i, example in enumerate(examples):
                with self.subTest(schema=rel, example=i):
                    self.assertEqual(_errors(example, schema), [])

    def test_outcome_without_its_fields_fails(self):
        for rel in RESULTS:
            schema = _load(rel)
            outcome = next(e for e in schema["examples"]
                           if e["status"] == OUTCOME[rel])
            owed = schema["status_requires"][OUTCOME[rel]]
            for field in owed:
                with self.subTest(schema=rel, field=field):
                    payload = {k: v for k, v in outcome.items() if k != field}
                    self.assertNotEqual(_errors(payload, schema), [])

    def test_reviewed_needs_a_verdict(self):
        schema = _load(REVIEW_RESULT)
        self.assertIn("verdict", schema["status_requires"]["reviewed"])

    def test_missing_inputs_are_invalid_input(self):
        """Agents cannot ask, so a missing input is the caller's omission."""
        for rel in RESULTS:
            schema = _load(rel)
            with self.subTest(schema=rel):
                self.assertNotIn("needs_input",
                                 schema["properties"]["status"]["enum"])
                self.assertNotEqual(
                    _errors(
                        {
                            "status": "invalid_input",
                            "summary": "no claim",
                            "run_id": RUN_ID,
                        }, schema), [])

    def _review_example(self, kind: str) -> dict:
        schema = _load(REVIEW_REQUEST)
        return dict(next(e for e in schema["examples"] if e["kind"] == kind))

    def test_stop_without_its_class_is_rejected(self):
        schema = _load(REVIEW_REQUEST)
        stop = self._review_example("stop")
        self.assertEqual(utils.validate_instance(stop, schema), [])
        for field in ("termination_class", "budget_left"):
            with self.subTest(field=field):
                payload = {k: v for k, v in stop.items() if k != field}
                self.assertNotEqual(utils.validate_instance(payload, schema),
                                    [])

    def test_claim_carrying_stop_fields_is_rejected(self):
        schema = _load(REVIEW_REQUEST)
        claim = self._review_example("claim")
        self.assertEqual(utils.validate_instance(claim, schema), [])
        stop_fields = {
            "termination_class": "floor",
            "budget_left": "6 TPU jobs",
            "unexamined": ["neighbour copy"],
        }
        for field, value in stop_fields.items():
            with self.subTest(field=field):
                self.assertNotEqual(
                    utils.validate_instance({
                        **claim, field: value
                    }, schema), [])

    def test_agents_bind_their_pair_and_delegate_only_leaves(self):
        """Leaf delegates keep nesting depth 3."""
        for name, (request, result) in BINDINGS.items():
            with self.subTest(agent=name):
                agent = package_discovery.load_agent_document(
                    _PACKAGE_ROOT / "agents" / f"{name}.md", _PACKAGE_ROOT)
                self.assertEqual(agent.get("accepts"), request)
                self.assertEqual(agent.get("produces"), result)
                self.assertEqual(set(agent.get("delegates")), DELEGATES[name])


if __name__ == "__main__":
    unittest.main()
