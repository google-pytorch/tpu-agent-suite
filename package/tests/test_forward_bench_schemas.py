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
"""Tests for the tpu-forward-bench request and result contracts."""

# ruff: noqa: E402 -- framework is imported after the repo root joins sys.path
from __future__ import annotations

import pathlib
import sys
import unittest

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework import agent_contracts, compiler, frontmatter, jsonio, utils

REQUEST = "schemas/forward-bench-request.json"
RESULT = "schemas/forward-bench-result.json"


def _load(rel: str) -> dict:
    return jsonio.load_json(_PACKAGE_ROOT / rel)


def _errors(payload: dict, schema: dict) -> list[str]:
    """Both halves the SubagentStop hook applies: shape, then status."""
    return (utils.validate_instance(payload, schema) +
            agent_contracts.check_status_payload(schema, payload))


class TestForwardBenchSchemas(unittest.TestCase):
    """The forward-bench pair and the agent that binds it."""

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

    def test_request_without_arms_or_image_fails(self):
        schema = _load(REQUEST)
        for field in ("arms", "image", "source_job"):
            with self.subTest(field=field):
                request = dict(schema["examples"][0])
                del request[field]
                self.assertNotEqual(utils.validate_instance(request, schema),
                                    [])

    def test_the_engine_is_explicit_and_only_a_supported_one(self):
        schema = _load(REQUEST)
        request = dict(schema["examples"][0])
        del request["engine"]
        self.assertNotEqual(utils.validate_instance(request, schema), [])
        request["engine"] = "custom_fastapi"
        self.assertNotEqual(utils.validate_instance(request, schema), [])

    def test_request_takes_a_moving_tag_as_invalid(self):
        """Every arm is the pinned image: a tag is not a measurement."""
        schema = _load(REQUEST)
        request = dict(schema["examples"][0], image="inferencex/vllm:latest")
        self.assertNotEqual(utils.validate_instance(request, schema), [])

    def test_an_arm_takes_a_plugin_ref_or_registry_patches_not_both(self):
        """`plugin` for the hill-climb's repo, `patches` for a caller without."""
        schema = _load(REQUEST)
        request = schema["examples"][1]
        arm = next(a for a in request["arms"] if "plugin" in a)
        registry = {k: v for k, v in arm.items() if k != "plugin"}
        registry["patches"] = ["scripts/common/vllm_patches/gpt_oss_mxfp4.py"]
        for ok in (arm, registry):
            self.assertEqual(
                utils.validate_instance(dict(request, arms=[ok]), schema), [])
        both = dict(arm, patches=registry["patches"])
        self.assertNotEqual(
            utils.validate_instance(dict(request, arms=[both]), schema), [])

    def test_a_short_plugin_sha_padded_with_zeros_is_refused(self):
        """A hill-climb session padded `2efe916` to 40 characters with zeros; that names no commit."""
        schema = _load(REQUEST)
        request = schema["examples"][1]
        arm = next(a for a in request["arms"] if "plugin" in a)
        padded = dict(arm,
                      plugin=dict(arm["plugin"], sha="2efe916" + "0" * 33))
        self.assertNotEqual(
            utils.validate_instance(dict(request, arms=[padded]), schema), [])

    def test_replay_record_is_a_request(self):
        """forward_bench_request.json reruns as is: plan resolved to a path."""
        schema = _load(REQUEST)
        replays = [
            e for e in schema["examples"] if isinstance(e.get("plan"), str)
        ]
        self.assertTrue(replays)
        for replay in replays:
            self.assertEqual(utils.validate_instance(replay, schema), [])

    def test_measured_owes_arms_cost_model_replay_and_report(self):
        schema = _load(RESULT)
        measured = next(e for e in schema["examples"]
                        if e["status"] == "measured")
        for field in ("arms", "cost_model", "replay", "report"):
            with self.subTest(field=field):
                payload = {k: v for k, v in measured.items() if k != field}
                self.assertNotEqual(_errors(payload, schema), [])

    def test_failures_owe_nothing(self):
        schema = _load(RESULT)
        for status in ("blocked", "failed"):
            with self.subTest(status=status):
                self.assertEqual(
                    _errors(
                        {
                            "status": status,
                            "summary": "stopped",
                            "failure_reason": "no slot",
                        }, schema), [])

    def test_unreviewed_adapter_is_named_with_its_path(self):
        schema = _load(RESULT)
        measured = next(e for e in schema["examples"]
                        if e["status"] == "measured")
        adapter = {"path": "./.work/models/M/model.py", "reviewed": False}
        self.assertEqual(_errors(dict(measured, adapter=adapter), schema), [])
        self.assertNotEqual(
            _errors(dict(measured, adapter={"path": adapter["path"]}), schema),
            [])

    def test_missing_inputs_are_invalid_input(self):
        """The agent cannot ask, so a missing input is the caller's omission."""
        schema = _load(RESULT)
        self.assertNotIn("needs_input", schema["properties"]["status"]["enum"])
        self.assertNotEqual(
            _errors({
                "status": "invalid_input",
                "summary": "no baseline arm",
            }, schema), [])

    def test_agent_binds_its_pair_and_delegates(self):
        agent, _ = frontmatter.parse_frontmatter_document(
            (_PACKAGE_ROOT / "agents/tpu-forward-bench.md").read_text())
        self.assertEqual(agent.get("accepts"), REQUEST)
        self.assertEqual(agent.get("produces"), RESULT)
        self.assertEqual(agent.get("delegates"), ["tpu-diagnostics"])


if __name__ == "__main__":
    unittest.main()
