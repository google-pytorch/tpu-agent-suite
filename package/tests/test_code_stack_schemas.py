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
"""Tests for the tpu-code-stack request and result contracts."""

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
REQUEST = "schemas/code-stack-request.json"
RESULT = "schemas/code-stack-result.json"
SHA = "0123456789abcdef0123456789abcdef01234567"
REPO = "./.work/glm/hill-climb/j-1/plugin"
# Each action's own fields, beyond action, caller, repo and run_id.
ACTION_FIELDS = {
    "init": {
        "image":
        "img@sha256:3f2a3f2a3f2a3f2a3f2a3f2a3f2a3f2a3f2a3f2a3f2a3f2a3f2a3f2a3f2a3f2a",
        "profile": "v6e-gke"
    },
    "branch": {
        "name": "hyp/2-rpa"
    },
    "commit": {
        "worktree": "./.work/glm/hill-climb/j-1/hyp-2-rpa",
        "message": "rpa: block 256"
    },
    "discard": {
        "name": "hyp/2-rpa",
        "what": "test"
    },
    "win": {
        "name": "hyp/2-rpa"
    },
    "stack": {
        "name": "stack-1-3",
        "branches": ["hyp/1-moe", "hyp/3-rope"]
    },
    "ref": {
        "ref": "stack"
    },
    "show": {
        "ref": "hyp/2-rpa"
    },
    "push": {
        "name": "glm-5-hill-climb"
    },
}
PLUGIN = {
    "repo": REPO,
    "base_sha": SHA,
    "ref": "hyp/2-rpa",
    "sha": SHA,
    "patch": "./.work/glm/hill-climb/j-1/patches/hyp-2-rpa-0123456.patch",
}


def _load(rel: str) -> dict:
    return jsonio.load_json(_PACKAGE_ROOT / rel)


def _errors(payload: dict, schema: dict) -> list[str]:
    """Both halves the SubagentStop hook applies: shape, then status."""
    return (utils.validate_instance(payload, schema) +
            agent_contracts.check_status_payload(schema, payload))


def _request(action: str, caller: str = "hill-climb") -> dict:
    return {
        "action": action,
        "caller": caller,
        "repo": REPO,
        "run_id": RUN_ID,
        **ACTION_FIELDS[action],
    }


class TestCodeStackSchemas(unittest.TestCase):
    """The code-stack pair and the leaf agent that binds it."""

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

    def test_each_action_requires_its_fields(self):
        schema = _load(REQUEST)
        for action, fields in ACTION_FIELDS.items():
            with self.subTest(action=action):
                self.assertEqual(
                    utils.validate_instance(_request(action), schema), [])
            for field in fields:
                with self.subTest(action=action, field=field):
                    payload = _request(action)
                    del payload[field]
                    self.assertNotEqual(
                        utils.validate_instance(payload, schema), [])

    def test_refs_and_paths_cannot_smuggle_git_options(self):
        """The agent passes these to git; a leading dash or `..` is never one."""
        schema = _load(REQUEST)
        for action, field, bad in (("ref", "ref", "--output=/tmp/x"),
                                   ("ref", "ref", "stack/../../x"),
                                   ("win", "name", "hyp/a;rm -rf ~"),
                                   ("commit", "worktree",
                                    "../../etc"), ("commit", "worktree",
                                                   "/tmp/$(id)"),
                                   ("init", "image",
                                    "img:latest"), ("init", "image",
                                                    "-x@sha256:" + "0" * 64),
                                   ("init", "profile",
                                    "p;id"), ("init", "profile",
                                              "-p"), ("init", "profile",
                                                      "a..b")):
            with self.subTest(field=field, value=bad):
                payload = _request(action)
                payload[field] = bad
                self.assertNotEqual(utils.validate_instance(payload, schema),
                                    [])

    def test_a_reviewer_commit_is_the_procedures_to_refuse(self):
        """Who may write is a procedure rule; the schema checks shape only."""
        schema = _load(REQUEST)
        self.assertEqual(
            utils.validate_instance(_request("commit", "reviewer"), schema),
            [])

    def test_skipped_push_carries_its_reason(self):
        schema = _load(RESULT)
        skipped = {
            "status": "skipped",
            "action": "push",
            "summary": "no image registry configured",
            "reason": "IMAGE_REGISTRY_NOT_CONFIGURED",
            "run_id": RUN_ID,
        }
        self.assertEqual(_errors(skipped, schema), [])
        del skipped["reason"]
        self.assertNotEqual(_errors(skipped, schema), [])

    def test_recorded_commit_requires_the_plugin(self):
        schema = _load(RESULT)
        recorded = {
            "status": "recorded",
            "action": "commit",
            "summary": "committed on hyp/2-rpa",
            "plugin": PLUGIN,
            "run_id": RUN_ID,
        }
        self.assertEqual(_errors(recorded, schema), [])
        del recorded["plugin"]
        self.assertNotEqual(_errors(recorded, schema), [])


if __name__ == "__main__":
    unittest.main()
