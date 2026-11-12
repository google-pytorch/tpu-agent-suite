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
"""Tests for the tpu-serving-bench request and result contracts."""

from __future__ import annotations

import pathlib
import sys
import unittest

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework import agent_contracts, compiler, jsonio, utils  # noqa: E402
from framework.builder import package_checks  # noqa: E402

RUN_ID = "20260921T120000Z-abcdef012345"
REQUEST = "schemas/serving-bench-request.json"
RESULT = "schemas/serving-bench-result.json"


def _load(rel: str) -> dict:
    return jsonio.load_json(_PACKAGE_ROOT / rel)


def _errors(payload: dict, schema: dict) -> list[str]:
    """Both halves the SubagentStop hook applies: shape, then status."""
    return (utils.validate_instance(payload, schema) +
            agent_contracts.check_status_payload(schema, payload))


def _example(schema: dict, action: str) -> dict:
    return dict(next(e for e in schema["examples"] if e["action"] == action))


class TestServingBenchSchemas(unittest.TestCase):
    """The serving-bench pair and the agent that binds it."""

    def test_schemas_are_registered_and_valid(self):
        registered = set(compiler.load_package(_PACKAGE_ROOT)["schemas"])
        for rel in (REQUEST, RESULT):
            with self.subTest(schema=rel):
                self.assertIn(rel, registered)
                self.assertEqual(utils.validate_schema_definition(_load(rel)),
                                 [])

    def test_examples_validate_and_cover_every_action(self):
        for rel in (REQUEST, RESULT):
            schema = _load(rel)
            examples = schema.get("examples", [])
            self.assertTrue(examples, f"{rel} has no examples")
            for i, example in enumerate(examples):
                with self.subTest(schema=rel, example=i):
                    self.assertEqual(_errors(example, schema), [])
        actions = {e["action"] for e in _load(REQUEST)["examples"]}
        self.assertEqual(actions, {"bench", "serve", "stop", "eval"})

    def test_the_engine_is_explicit_and_only_a_supported_one(self):
        schema = _load(REQUEST)
        request = _example(schema, "bench")
        del request["engine"]
        self.assertNotEqual(utils.validate_instance(request, schema), [])
        request["engine"] = "custom_fastapi"
        self.assertNotEqual(utils.validate_instance(request, schema), [])

    def test_each_action_needs_its_own_inputs(self):
        schema = _load(REQUEST)
        for action, field in (("bench", "arms"), ("serve", "deployment"),
                              ("stop", "servers")):
            with self.subTest(action=action):
                request = _example(schema, action)
                del request[field]
                self.assertNotEqual(utils.validate_instance(request, schema),
                                    [])
        # A bench names its workload once: a bench_request or a deployment.
        request = _example(schema, "bench")
        request["deployment"] = _example(schema, "serve")["deployment"]
        self.assertNotEqual(utils.validate_instance(request, schema), [])

    def test_an_eval_needs_its_config_and_limit(self):
        """They pick the protocol, the baseline and the gate size."""
        schema = _load(REQUEST)
        for field in ("config", "limit"):
            with self.subTest(field=field):
                request = _example(schema, "eval")
                del request[field]
                self.assertNotEqual(utils.validate_instance(request, schema),
                                    [])

    def test_a_bench_carrying_eval_fields_is_rejected(self):
        schema = _load(REQUEST)
        evaluation = _example(schema, "eval")
        for field in ("limit", "config", "vllm_server_cmd"):
            with self.subTest(field=field):
                request = _example(schema, "bench")
                request[field] = evaluation[field]
                self.assertNotEqual(utils.validate_instance(request, schema),
                                    [])

    def test_an_arm_on_a_moving_tag_is_invalid(self):
        schema = _load(REQUEST)
        request = _example(schema, "bench")
        request["arms"] = [dict(request["arms"][0], image="vllm/vllm:latest")]
        self.assertNotEqual(utils.validate_instance(request, schema), [])

    def test_an_arm_takes_its_own_server_args(self):
        """One arm adds vllm serve flags (an MTP arm); an empty or non-string flag is refused."""
        schema = _load(REQUEST)
        request = _example(schema, "bench")
        arm = dict(request["arms"][-1],
                   server_args=["--speculative-config", '{"method":"mtp"}'])
        ok = dict(request, arms=request["arms"][:-1] + [arm])
        self.assertEqual(utils.validate_instance(ok, schema), [])
        for bad in ([""], [3]):
            with self.subTest(server_args=bad):
                arms = request["arms"][:-1] + [dict(arm, server_args=bad)]
                self.assertNotEqual(
                    utils.validate_instance(dict(request, arms=arms), schema),
                    [])

    def test_a_short_plugin_sha_padded_with_zeros_is_refused(self):
        """A hill-climb session padded `2efe916` to 40 characters with zeros; that names no commit."""
        schema = _load(REQUEST)
        for request in schema["examples"]:
            arms = request.get("arms") or []
            for i, arm in enumerate(arms):
                if "plugin" not in arm:
                    continue
                padded = dict(arm,
                              plugin=dict(arm["plugin"],
                                          sha="2efe916" + "0" * 33))
                bad = dict(request, arms=arms[:i] + [padded] + arms[i + 1:])
                with self.subTest(action=request["action"]):
                    self.assertNotEqual(utils.validate_instance(bad, schema),
                                        [])

    def test_an_arm_takes_a_plugin_ref_or_registry_patches_not_both(self):
        """`plugin` for the hill-climb's repo, `patches` for a caller without."""
        schema = _load(REQUEST)
        request = _example(schema, "bench")
        arm = next(a for a in request["arms"] if "plugin" in a)
        registry = {k: v for k, v in arm.items() if k != "plugin"}
        registry["patches"] = ["scripts/common/vllm_patches/gpt_oss_mxfp4.py"]
        for ok in (arm, registry):
            self.assertEqual(
                utils.validate_instance(dict(request, arms=[ok]), schema), [])
        both = dict(arm, patches=registry["patches"])
        self.assertNotEqual(
            utils.validate_instance(dict(request, arms=[both]), schema), [])

    def test_replay_record_is_a_request(self):
        """serving_bench_request.json re-sends as is, with its plan."""
        schema = _load(REQUEST)
        replays = [e for e in schema["examples"] if "plan" in e]
        self.assertTrue(replays)
        for replay in replays:
            self.assertEqual(utils.validate_instance(replay, schema), [])
            self.assertIn("hardware", replay)

    def test_each_outcome_owes_its_payload(self):
        schema = _load(RESULT)
        for status, fields in (("measured", ("arms", "replay", "report")),
                               ("evaluated", ("arms", "gate", "gate_pp",
                                              "delta_pp", "report")),
                               ("served", ("servers", )), ("stopped",
                                                           ("stopped", ))):
            example = next(e for e in schema["examples"]
                           if e["status"] == status)
            for field in fields:
                with self.subTest(status=status, field=field):
                    payload = {k: v for k, v in example.items() if k != field}
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
                            "run_id": RUN_ID,
                        }, schema), [])

    def test_missing_inputs_are_invalid_input(self):
        """The agent cannot ask, so a missing input is the caller's omission."""
        schema = _load(RESULT)
        self.assertNotIn("needs_input", schema["properties"]["status"]["enum"])
        self.assertNotEqual(
            _errors(
                {
                    "status": "invalid_input",
                    "summary": "no workload",
                    "run_id": RUN_ID,
                }, schema), [])

    def test_agent_binds_its_pair_and_delegates(self):
        agents = {
            a["name"]: a
            for a in compiler.load_package(_PACKAGE_ROOT)["agents"]
        }
        agent = agents["tpu-serving-bench"]
        self.assertEqual(agent.get("accepts"), REQUEST)
        self.assertEqual(agent.get("produces"), RESULT)
        self.assertEqual(agent.get("delegates"), ["tpu-diagnostics"])

    def test_chain_fits_the_depth(self):
        """tpu-agent -> tpu-serving-bench -> tpu-diagnostics."""
        manifest = compiler.load_package(_PACKAGE_ROOT)
        skills = [s for s in manifest["skills"] if s["name"] == "tpu-agent"]
        agents = {a["name"]: a for a in manifest["agents"]}
        self.assertIn("tpu-serving-bench", skills[0]["delegates"])
        package_checks.validate_delegation_graph(skills, agents, set())


if __name__ == "__main__":
    unittest.main()
