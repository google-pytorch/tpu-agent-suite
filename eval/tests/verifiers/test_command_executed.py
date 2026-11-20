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
"""Unit tests for eval/verifiers/functional/command_executed.py."""

import contextlib
import io
import json
import unittest
from typing import Any

from eval.tests.verifiers import helpers
from eval.verifiers.functional import command_executed

_PARAMS = {
    "artifact_globs": ["*sample-report*.json"],
    "target_names": ["sample-agent"],
    "command_regex": r"sample_tool\.py",
}
# The folder the verifier loads from here; a compiled task stages it under
# tests/<name>/ and keys its params by that name.
_SECTION = "functional"

_COMMAND = {"command": "python3 sample_tool.py"}
_DELEGATION = {"Subagents": [{"TypeName": "sample-agent", "Prompt": "go"}]}
_OTHER_DELEGATION = {
    "Subagents": [{
        "TypeName": "other-agent",
        "Prompt": "go"
    }]
}


def _call(name: str, arguments: dict) -> dict:
    return {"tool_calls": [{"function_name": name, "arguments": arguments}]}


class CommandExecutedTest(unittest.TestCase):

    def setUp(self):
        self.harness = self.enterContext(helpers.harness())
        self.workspace = self.harness.workspace
        self.artifacts = self.harness.logs / "artifacts"
        self.artifacts.mkdir()

    def _check(self, steps: list[dict], params: Any = _PARAMS) -> bool:
        """Runs the check with `params` as this verifier's table."""
        return self._check_with_file(steps, {_SECTION: params})

    def _check_with_file(self, steps: list[dict], params_file: Any) -> bool:
        """Runs the check with `params_file` as verifier_params.json."""
        self.harness.write_trajectory({"steps": steps})
        params_path = self.harness.tests / "verifier_params.json"
        if params_file is None:
            params_path.unlink(missing_ok=True)
        else:
            self.harness.write_params(params_file)
        with contextlib.redirect_stdout(io.StringIO()):
            return command_executed.check_command_executed(self.workspace)

    def test_artifact_passes(self):
        (self.workspace / "sample-report.json").write_text("{}",
                                                           encoding="utf-8")
        self.assertTrue(self._check([]))

    def test_artifact_in_harness_artifact_dir_passes(self):
        (self.artifacts / "sample-report.json").write_text("{}",
                                                           encoding="utf-8")
        self.assertTrue(self._check([]))

    def test_empty_file_is_not_an_artifact(self):
        (self.workspace / "sample-report.json").write_text("",
                                                           encoding="utf-8")
        self.assertFalse(self._check([]))

    def test_command_passes(self):
        self.assertTrue(self._check([_call("run_command", _COMMAND)]))

    def test_command_in_stream_log_passes(self):
        event = {
            "step_update": {
                "tool_call": {
                    "name": "run_command",
                    "args": _COMMAND
                }
            }
        }
        (self.harness.logs / "agent" / "stream.jsonl").write_text(
            json.dumps(event) + "\n", encoding="utf-8")
        self.assertTrue(self._check([]))

    def test_delegation_to_target_passes(self):
        self.assertTrue(self._check([_call("invoke_subagent", _DELEGATION)]))

    def test_unrelated_activity_fails(self):
        steps = [
            _call("run_command", {"command": "ls"}),
            _call("invoke_subagent", _OTHER_DELEGATION),
        ]
        self.assertFalse(self._check(steps))

    def test_params_outside_the_verifiers_table_are_ignored(self):
        steps = [_call("run_command", _COMMAND)]
        self.assertFalse(self._check_with_file(steps, _PARAMS))
        self.assertFalse(self._check_with_file(steps, {"engine": _PARAMS}))

    def test_target_names_alone_match_delegations(self):
        params = {"target_names": _PARAMS["target_names"]}
        self.assertTrue(
            self._check([_call("invoke_subagent", _DELEGATION)], params))
        self.assertFalse(self._check([_call("run_command", _COMMAND)], params))
        self.assertFalse(
            self._check([_call("invoke_subagent", _OTHER_DELEGATION)], params))

    def test_missing_or_mistyped_params_fail_without_raising(self):
        (self.workspace / "sample-report.json").write_text("{}",
                                                           encoding="utf-8")
        steps = [
            _call("run_command", _COMMAND),
            _call("invoke_subagent", _DELEGATION),
        ]
        mistyped = {
            "artifact_globs": "*sample-report*.json",
            "target_names": [7, ""],
            "command_regex": 42,
        }
        for params in ({}, mistyped):
            with self.subTest(params=params):
                self.assertFalse(self._check(steps, params))
        for params_file in (None, "not json", "[1, 2]"):
            with self.subTest(params_file=params_file):
                self.assertFalse(self._check_with_file(steps, params_file))


if __name__ == "__main__":
    unittest.main()
