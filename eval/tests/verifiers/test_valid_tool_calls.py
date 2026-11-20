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
"""Unit tests for valid_tool_calls verifier."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.tests.verifiers import helpers
from eval.verifiers.efficiency_metrics import \
    valid_tool_calls as valid_tool_calls_mod


class ValidToolCallsTest(unittest.TestCase):

    def test_load_trajectory_missing_and_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            missing = tmp_path / "missing.json"
            self.assertIsNone(valid_tool_calls_mod._load_trajectory(missing))

            corrupt = tmp_path / "corrupt.json"
            corrupt.write_text("{invalid json", encoding="utf-8")
            self.assertIsNone(valid_tool_calls_mod._load_trajectory(corrupt))

            valid_file = tmp_path / "valid.json"
            valid_file.write_text('{"steps": []}', encoding="utf-8")
            self.assertEqual(valid_tool_calls_mod._load_trajectory(valid_file),
                             {"steps": []})

    def test_missing_trajectory_returns_zero(self) -> None:
        with patch.object(valid_tool_calls_mod,
                          "_load_trajectory",
                          return_value=None):
            score = valid_tool_calls_mod.valid_tool_calls(Path("/workspace"))
            self.assertEqual(score, 0.0)

    def test_no_tool_calls_uses_empty_score(self) -> None:
        traj = {
            "steps": [{
                "step_id": 1,
                "source": "user",
                "message": "hello"
            }]
        }
        with (
                patch.object(valid_tool_calls_mod,
                             "_load_trajectory",
                             return_value=traj),
                patch.object(valid_tool_calls_mod,
                             "_load_verifier_params",
                             return_value={}),
        ):
            self.assertEqual(
                valid_tool_calls_mod.valid_tool_calls(Path("/workspace")), 1.0)

        with (
                patch.object(valid_tool_calls_mod,
                             "_load_trajectory",
                             return_value=traj),
                patch.object(
                    valid_tool_calls_mod,
                    "_load_verifier_params",
                    return_value={"empty_score": 0.0},
                ),
        ):
            self.assertEqual(
                valid_tool_calls_mod.valid_tool_calls(Path("/workspace")), 0.0)

    def test_read_only_file_tools_ignore_error_strings_in_file_content(
            self) -> None:
        traj = {
            "steps": [{
                "step_id":
                1,
                "tool_calls": [
                    {
                        "tool_call_id": "tc-1",
                        "function_name": "view_file",
                        "arguments": {
                            "AbsolutePath": "/workspace/app.py"
                        },
                    },
                    {
                        "tool_call_id": "tc-2",
                        "function_name": "grep_search",
                        "arguments": {
                            "Query": "ValueError"
                        },
                    },
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id":
                            "tc-1",
                            "content":
                            ("1: Traceback (most recent call last):\n"
                             "2: raise ValueError('bad value')\n"
                             "3: command not found"),
                        },
                        {
                            "source_call_id": "tc-2",
                            "content":
                            "app.py:2: raise ValueError('bad value')",
                        },
                    ]
                },
            }]
        }
        with (
                patch.object(valid_tool_calls_mod,
                             "_load_trajectory",
                             return_value=traj),
                patch.object(valid_tool_calls_mod,
                             "_load_verifier_params",
                             return_value={}),
        ):
            self.assertEqual(
                valid_tool_calls_mod.valid_tool_calls(Path("/workspace")), 1.0)

    def test_read_only_file_tool_runner_error_is_flagged(self) -> None:
        traj = {
            "steps": [{
                "step_id":
                1,
                "tool_calls": [{
                    "tool_call_id": "tc-1",
                    "function_name": "view_file",
                    "arguments": {
                        "AbsolutePath": "/workspace/nonexistent.py"
                    },
                }],
                "observation": {
                    "results": [{
                        "source_call_id":
                        "tc-1",
                        "content":
                        "Encountered error in tool execution: file does not exist",
                    }]
                },
            }]
        }
        with (
                patch.object(valid_tool_calls_mod,
                             "_load_trajectory",
                             return_value=traj),
                patch.object(valid_tool_calls_mod,
                             "_load_verifier_params",
                             return_value={}),
        ):
            self.assertEqual(
                valid_tool_calls_mod.valid_tool_calls(Path("/workspace")), 0.0)

    def test_failed_and_valid_tool_calls_ratio(self) -> None:
        traj = {
            "steps": [
                # 1. Non-zero command exit code in Antigravity format
                {
                    "step_id":
                    1,
                    "tool_calls": [{
                        "tool_call_id": "1",
                        "function_name": "run_command",
                        "arguments": {
                            "CommandLine": "false"
                        },
                    }],
                    "observation": {
                        "results": [{
                            "source_call_id":
                            "1",
                            "content":
                            "The command exited with code 1.",
                        }]
                    },
                },
                # 2. Claude Code [error] tool reported failure
                {
                    "step_id":
                    2,
                    "tool_calls": [{
                        "tool_call_id": "2",
                        "function_name": "Bash",
                        "arguments": {
                            "command": "bad_cmd"
                        },
                    }],
                    "observation": {
                        "results": [{
                            "source_call_id":
                            "2",
                            "content":
                            "stderr output\n[error] tool reported failure",
                        }]
                    },
                },
                # 3. Step-level extra metadata with non-zero exitCode
                {
                    "step_id":
                    3,
                    "tool_calls": [{
                        "tool_call_id": "3",
                        "function_name": "Bash",
                        "arguments": {
                            "command": "exit 2"
                        },
                    }],
                    "extra": {
                        "tool_result_metadata": {
                            "tool_use_result": {
                                "exitCode": 2
                            }
                        }
                    },
                    "observation": {
                        "results": [{
                            "source_call_id": "3",
                            "content": "failed"
                        }]
                    },
                },
                # 4. Blind API / CLI flag guessing in execution output
                {
                    "step_id":
                    4,
                    "tool_calls": [
                        {
                            "tool_call_id": "4a",
                            "function_name": "bash",
                            "arguments": {
                                "command": "python3 -c 'import foo'"
                            },
                        },
                        {
                            "tool_call_id": "4b",
                            "function_name": "bash",
                            "arguments": {
                                "command": "tool --unknown-flag"
                            },
                        },
                    ],
                    "observation": {
                        "results": [
                            {
                                "source_call_id":
                                "4a",
                                "content":
                                ("Traceback (most recent call last):\n"
                                 '  File "<string>", line 1, in <module>\n'
                                 "ModuleNotFoundError: No module named 'foo'"),
                            },
                            {
                                "source_call_id":
                                "4b",
                                "content":
                                "tool: error: unrecognized arguments: --unknown-flag",
                            },
                        ]
                    },
                },
                # 5. Unknown tool name & malformed JSON arguments
                {
                    "step_id":
                    5,
                    "tool_calls": [
                        {
                            "tool_call_id": "5a",
                            "function_name": "unknown",
                            "arguments": {},
                        },
                        {
                            "tool_call_id": "5b",
                            "function_name": "bash",
                            "arguments": "{bad json",
                        },
                    ],
                },
                # 6. Three valid tool calls
                {
                    "step_id":
                    6,
                    "tool_calls": [
                        {
                            "tool_call_id": "6a",
                            "function_name": "run_command",
                            "arguments": {
                                "CommandLine": "echo hello"
                            },
                        },
                        {
                            "tool_call_id": "6b",
                            "function_name": "bash",
                            "arguments": {
                                "command": "ls"
                            },
                        },
                        {
                            "tool_call_id": "6c",
                            "function_name": "list_dir",
                            "arguments": {
                                "DirectoryPath": "/workspace"
                            },
                        },
                    ],
                    "observation": {
                        "results": [
                            {
                                "source_call_id":
                                "6a",
                                "content":
                                "The command exited with code 0.\nhello",
                            },
                            {
                                "source_call_id": "6b",
                                "content": "file.py\n[exit_code] 0",
                            },
                            {
                                "source_call_id": "6c",
                                "content": '{"name": "file.py"}',
                            },
                        ]
                    },
                },
            ]
        }
        # 3 valid out of 10 total tool calls -> ratio = 0.3
        with (
                patch.object(valid_tool_calls_mod,
                             "_load_trajectory",
                             return_value=traj),
                patch.object(valid_tool_calls_mod,
                             "_load_verifier_params",
                             return_value={}),
        ):
            ratio = valid_tool_calls_mod.valid_tool_calls(Path("/workspace"))
            self.assertAlmostEqual(ratio, 0.3)

    def test_min_ratio_param(self) -> None:
        traj = {
            "steps": [{
                "step_id":
                1,
                "tool_calls": [
                    {
                        "tool_call_id": "1",
                        "function_name": "run_command",
                        "arguments": {},
                    },
                    {
                        "tool_call_id": "2",
                        "function_name": "view_file",
                        "arguments": {},
                    },
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "1",
                            "content": "The command exited with code 1.",
                        },
                        {
                            "source_call_id": "2",
                            "content": "1: ok",
                        },
                    ]
                },
            }]
        }
        # Ratio is 0.5; with min_ratio=0.8 it should return 0.0
        with (
                patch.object(valid_tool_calls_mod,
                             "_load_trajectory",
                             return_value=traj),
                patch.object(
                    valid_tool_calls_mod,
                    "_load_verifier_params",
                    return_value={"min_ratio": 0.8},
                ),
        ):
            self.assertEqual(
                valid_tool_calls_mod.valid_tool_calls(Path("/workspace")), 0.0)

        # Ratio is 0.5; with min_ratio=0.5 it should return 0.5
        with (
                patch.object(valid_tool_calls_mod,
                             "_load_trajectory",
                             return_value=traj),
                patch.object(
                    valid_tool_calls_mod,
                    "_load_verifier_params",
                    return_value={"min_ratio": 0.5},
                ),
        ):
            self.assertEqual(
                valid_tool_calls_mod.valid_tool_calls(Path("/workspace")), 0.5)

    def test_cross_step_observation_matching_and_positional_fallback(
            self) -> None:
        traj = {
            "steps": [
                {
                    "step_id":
                    1,
                    "tool_calls": [
                        {
                            "tool_call_id": "cross-call",
                            "function_name": "bash",
                            "arguments": {},
                        },
                        {
                            "function_name": "bash",
                            "arguments": {},
                        },
                    ],
                    "observation": {
                        "results": [
                            {
                                "content": "placeholder for idx 0"
                            },
                            {
                                "content": "[exit_code] 1"
                            },
                        ]
                    },
                },
                {
                    "step_id": 2,
                    "observation": {
                        "results": [{
                            "source_call_id": "cross-call",
                            "content": "[exit_code] 0",
                        }]
                    },
                },
            ]
        }
        outcomes = valid_tool_calls_mod._collect_tool_call_outcomes(traj)
        self.assertEqual(outcomes, [("bash", True), ("bash", False)])

    def test_load_verifier_params_from_file(self) -> None:
        folder_key = Path(valid_tool_calls_mod.__file__).parent.name
        with helpers.harness() as harness:
            harness.write_params({folder_key: {"min_ratio": 0.75}})
            self.assertEqual(valid_tool_calls_mod._load_verifier_params(),
                             {"min_ratio": 0.75})

    def test_trajectory_read_from_the_harness_logs(self) -> None:
        with helpers.harness() as harness:
            self.assertEqual(
                valid_tool_calls_mod.valid_tool_calls(harness.workspace), 0.0)
            harness.write_trajectory({"steps": []})
            self.assertEqual(
                valid_tool_calls_mod.valid_tool_calls(harness.workspace), 1.0)


if __name__ == "__main__":
    unittest.main()
