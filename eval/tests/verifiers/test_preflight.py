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
"""Unit tests for canonical preflight verifiers."""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

from eval.tests.verifiers import helpers
from eval.verifiers.functional.preflight import \
    preflight_ambiguous_name as ambiguous
from eval.verifiers.functional.preflight import \
    preflight_full_sequence as full_seq
from eval.verifiers.functional.preflight import \
    preflight_gptq_host_reject as gptq_reject

# The preflight verifiers import this from their own folder, under its bare
# name, as they do in a task.
preflight_common = sys.modules["preflight_common"]


def _trajectory(shell_cmds: list[str]) -> dict:
    return {
        "steps": [{
            "tool_calls": [{
                "function_name": "Bash",
                "arguments": {
                    "command": cmd
                },
            }],
        } for cmd in shell_cmds]
    }


class PreflightVerifiersTest(unittest.TestCase):

    def setUp(self) -> None:
        self.harness = self.enterContext(helpers.harness())
        self.workspace = self.harness.workspace
        # Reads the agent's home in the container, which a test has none of.
        self.enterContext(
            mock.patch.object(preflight_common,
                              "load_agy_transcripts",
                              return_value=[]))

    def test_ambiguous_name_checks(self) -> None:
        self.harness.write_trajectory(
            _trajectory([
                "S=scripts/preflight_check.py && python3 $S --resolve 'llama 3 8b'"
            ]))
        self.assertTrue(ambiguous.resolved_name(self.workspace))
        self.assertTrue(ambiguous.no_check_on_a_guess(self.workspace))

    def test_gptq_host_reject_checks(self) -> None:
        self.harness.write_trajectory(
            _trajectory([
                "python3 preflight_check.py --resolve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
                "python3 preflight_check.py --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --hub_only",
            ]))
        self.assertTrue(gptq_reject.resolved_first(self.workspace))
        self.assertTrue(gptq_reject.hub_checks_on_host(self.workspace))
        self.assertTrue(gptq_reject.rejected_before_chip(self.workspace))

    def test_full_sequence_disqualifies_host_transformers_install(
            self) -> None:
        self.harness.write_trajectory(
            _trajectory([
                "python3 preflight_check.py --resolve openai/gpt-oss-20b",
                "python3 preflight_check.py --model openai/gpt-oss-20b --hub_only",
                "pip install transformers",
                "compute --package tag run python3 preflight_check.py --model openai/gpt-oss-20b",
            ]))
        self.assertTrue(full_seq.resolved_first(self.workspace))
        self.assertTrue(full_seq.hub_checks_before_chip(self.workspace))
        self.assertFalse(full_seq.full_check_on_accelerator(self.workspace))

    def test_workspace_trajectory_is_ignored(self) -> None:
        trajectory = json.dumps(
            _trajectory(["python3 preflight_check.py --resolve 'llama 3 8b'"]))
        (self.workspace / "agent").mkdir()
        for directory in (self.workspace, self.workspace / "agent"):
            Path(directory / "trajectory.json").write_text(trajectory,
                                                           encoding="utf-8")
        self.assertFalse(ambiguous.resolved_name(self.workspace))


if __name__ == "__main__":
    unittest.main()
