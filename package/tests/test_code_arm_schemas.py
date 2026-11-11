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
"""Tests for the tpu-code-arm request and result contracts."""

from __future__ import annotations

import copy
import pathlib
import sys
import unittest

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework import agent_contracts, jsonio, utils  # noqa: E402

REQUEST = "schemas/code-arm-request.json"
RESULT = "schemas/code-arm-result.json"


def _load(rel: str) -> dict:
    return jsonio.load_json(_PACKAGE_ROOT / rel)


def _errors(payload: dict, schema: dict) -> list[str]:
    """Both halves the SubagentStop hook applies: shape, then status."""
    return (utils.validate_instance(payload, schema) +
            agent_contracts.check_status_payload(schema, payload))


def _example(schema: dict, status: str) -> dict:
    return next(e for e in schema["examples"] if e["status"] == status)


class TestCodeArmSchemas(unittest.TestCase):
    """The code-arm pair."""

    def test_request_without_checkout_fails(self):
        """The arm edits a worktree tpu-code-stack cut; it never runs git."""
        schema = _load(REQUEST)
        request = copy.deepcopy(schema["examples"][0])
        del request["checkout"]
        self.assertNotEqual(utils.validate_instance(request, schema), [])

    def test_changed_owes_matching_outputs_and_a_review(self):
        """Only the oneOf branch enforces these; status_requires cannot."""
        schema = _load(RESULT)
        mismatched = copy.deepcopy(_example(schema, "changed"))
        mismatched["evidence"]["outputs_match"] = False
        self.assertNotEqual(_errors(mismatched, schema), [])
        unreviewed = copy.deepcopy(_example(schema, "changed"))
        del unreviewed["evidence"]["review"]
        self.assertNotEqual(_errors(unreviewed, schema), [])


if __name__ == "__main__":
    unittest.main()
