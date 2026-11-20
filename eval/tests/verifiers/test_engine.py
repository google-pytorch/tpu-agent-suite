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
"""Unit tests for eval/verifiers/functional/engine.py."""

import importlib
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from eval.tests.verifiers import helpers
from eval.verifiers.functional import engine as engine_check

_VERIFY = "python3 scripts/verify_inference.py"


def _reload_engine_check(params: Any = None) -> None:
    """Loads engine.py again, which reads its params as it loads.

    It loads against an empty harness, or one whose verifier_params.json is
    `params`.
    """
    with helpers.harness() as harness:
        if params is not None:
            harness.write_params(params)
        importlib.reload(engine_check)


class EngineCheckTest(unittest.TestCase):

    def _check(self,
               expected: str,
               commands=(),
               files=(),
               ignore_truncated: bool = False) -> bool:
        other = "custom_fastapi" if expected == "vllm" else "vllm"
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            for name in files:
                (ws / name).write_text("#\n", encoding="utf-8")
            with mock.patch.multiple(
                    engine_check,
                    EXPECTED_ENGINE=expected,
                    OTHER_ENGINE=other,
                    _PARAMS={"ignore_truncated_commands": ignore_truncated},
                    _search_dirs=lambda workspace: [ws],
                    _collect_commands=lambda: list(commands),
            ):
                return engine_check.check_engine_used(ws)

    def test_expected_engine_read_from_the_verifiers_table(self):
        self.addCleanup(_reload_engine_check)
        _reload_engine_check(
            {"functional": {
                "expected_engine": "custom_fastapi"
            }})
        self.assertEqual(engine_check.EXPECTED_ENGINE, "custom_fastapi")
        self.assertEqual(engine_check.OTHER_ENGINE, "vllm")

    def test_other_or_malformed_verifier_params_expect_vllm(self):
        self.addCleanup(_reload_engine_check)
        other_table = {"staging": {"expected_engine": "custom_fastapi"}}
        for params in (other_table, "not json", "[1, 2]"):
            with self.subTest(params=params):
                _reload_engine_check(params)
                self.assertEqual(engine_check.EXPECTED_ENGINE, "vllm")

    def test_commands_read_from_the_harness_trajectory(self):
        with helpers.harness() as harness:
            harness.write_trajectory({
                "steps": [{
                    "tool_calls": [{
                        "function_name": "bash",
                        "arguments": {
                            "command": f"{_VERIFY} --engine vllm"
                        },
                    }]
                }]
            })
            self.assertEqual(engine_check._collect_commands(),
                             [f"{_VERIFY} --engine vllm"])

    def test_expected_engine_passes_only_when_deployed_alone(self):
        self.assertTrue(self._check("vllm", files=["launch_vllm.sh"]))
        self.assertFalse(
            self._check("vllm", files=["launch_vllm.sh", "fastapi_server.py"]))

    def test_engine_read_from_verify_command(self):
        commands = [
            f"{_VERIFY} \\\n  --engine custom_fastapi",
            f"{_VERIFY} --help",
        ]
        self.assertTrue(self._check("custom_fastapi", commands=commands))

    def test_truncated_command_ignored_when_configured(self):
        commands = [
            "compute submit --recipe launch_vllm.sh",
            f"{_VERIFY} --url http://host …",
        ]
        self.assertFalse(self._check("vllm", commands=commands))
        self.assertTrue(
            self._check("vllm", commands=commands, ignore_truncated=True))


if __name__ == "__main__":
    unittest.main()
