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
"""Tests for the tpu-performance-optimization request and result contracts."""

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

REQUEST = "schemas/perf-triage-request.json"
RESULT = "schemas/perf-triage-result.json"


def _load(rel: str) -> dict:
    return jsonio.load_json(_PACKAGE_ROOT / rel)


def _errors(payload: dict, schema: dict) -> list[str]:
    """Both halves the SubagentStop hook applies: shape, then status."""
    return (utils.validate_instance(payload, schema) +
            agent_contracts.check_status_payload(schema, payload))


def _triaged() -> dict:
    schema = _load(RESULT)
    return next(e for e in schema["examples"] if e["status"] == "triaged")


class TestPerfTriageSchemas(unittest.TestCase):
    """The perf-triage pair and the agent that binds it."""

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

    def test_the_engine_is_explicit_and_only_a_supported_one(self):
        schema = _load(REQUEST)
        request = dict(schema["examples"][0])
        del request["engine"]
        self.assertNotEqual(utils.validate_instance(request, schema), [])
        request["engine"] = "custom_fastapi"
        self.assertNotEqual(utils.validate_instance(request, schema), [])

    def test_request_without_a_baseline_fails(self):
        schema = _load(REQUEST)
        request = dict(schema["examples"][0])
        del request["baseline"]
        self.assertNotEqual(utils.validate_instance(request, schema), [])

    def test_triaged_owes_gaps_and_paths(self):
        schema = _load(RESULT)
        triaged = _triaged()
        for field in ("gaps", "decomposition", "roofline", "traces"):
            with self.subTest(field=field):
                payload = {k: v for k, v in triaged.items() if k != field}
                self.assertNotEqual(_errors(payload, schema), [])

    def test_an_op_without_a_floor_gets_no_headroom(self):
        """No floor is a reason, never an estimated headroom or a rank."""
        schema = _load(RESULT)
        triaged = _triaged()
        no_floor = next(g for g in triaged["gaps"] if "no_floor" in g)
        guessed = dict(no_floor, headroom=0.4)
        self.assertNotEqual(_errors(dict(triaged, gaps=[guessed]), schema), [])
        ranked = next(g for g in triaged["gaps"] if "rank" in g)
        unpriced = {k: v for k, v in ranked.items() if k != "floor_ms"}
        self.assertNotEqual(_errors(dict(triaged, gaps=[unpriced]), schema),
                            [])

    def test_a_hypothesis_names_one_of_the_routes(self):
        schema = _load(RESULT)
        triaged = _triaged()
        gap = next(g for g in triaged["gaps"] if g.get("hypotheses"))
        bad = dict(gap["hypotheses"][0], route="sharding")
        self.assertNotEqual(
            _errors(dict(triaged, gaps=[dict(gap, hypotheses=[bad])]), schema),
            [])

    def test_failures_owe_nothing(self):
        schema = _load(RESULT)
        self.assertEqual(
            _errors(
                {
                    "status": "failed",
                    "summary": "trace export failed",
                    "failure_reason": "trace_empty_or_partial",
                }, schema), [])

    def test_missing_inputs_are_invalid_input(self):
        """The agent cannot ask, so a missing input is the caller's omission."""
        schema = _load(RESULT)
        self.assertNotIn("needs_input", schema["properties"]["status"]["enum"])
        self.assertNotEqual(
            _errors(
                {
                    "status": "invalid_input",
                    "summary": "no baseline traces",
                }, schema), [])

    def test_agent_binds_its_pair_and_delegates(self):
        agent = package_discovery.load_agent_document(
            _PACKAGE_ROOT / "agents/tpu-performance-optimization.md",
            _PACKAGE_ROOT)
        self.assertEqual(agent.get("accepts"), REQUEST)
        self.assertEqual(agent.get("produces"), RESULT)
        self.assertEqual(
            sorted(agent.get("delegates")),
            ["tpu-diagnostics", "tpu-knowledge-base", "tpu-roofline"])


if __name__ == "__main__":
    unittest.main()
