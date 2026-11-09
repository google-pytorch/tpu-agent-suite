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
"""Offline tests for unit evaluation deterministic command_executed criteria."""

import importlib.util
import pathlib
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parents[2]
NIGHTLY = REPO / "eval" / "datasets" / "nightly-functional"
SMOKE = REPO / "eval" / "datasets" / "smoke-check"

UNIT_TASKS = (
    NIGHTLY / "torch-inference-coder",
    NIGHTLY / "torch-benchmark-coder",
    NIGHTLY / "engine-compatibility",
    NIGHTLY / "remote-tpu-runner",
    NIGHTLY / "smart-hf-search",
    SMOKE / "preflight-check",
)


def _load_check(path: pathlib.Path, steps: list[dict]):
    rk = types.ModuleType("rewardkit")
    rk.criterion = lambda **_: (lambda fn: fn)
    rk_crit = types.ModuleType("rewardkit.criteria")
    rk_traj = types.ModuleType("rewardkit.criteria._trajectory")
    rk_traj.load_trajectory = lambda p: ({
        "steps": steps
    } if p == "/logs/agent/trajectory.json" else None)
    rk_traj.collect_tool_calls = lambda d: [
        tc for s in d.get("steps", []) for tc in s.get("tool_calls", [])
    ]
    modules = {
        "rewardkit": rk,
        "rewardkit.criteria": rk_crit,
        "rewardkit.criteria._trajectory": rk_traj,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


class UnitEvalChecksTest(unittest.TestCase):

    def setUp(self):
        self.workspace = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.workspace, ignore_errors=True)
        p = mock.patch.object(pathlib.Path,
                              "home",
                              return_value=self.workspace)
        p.start()
        self.addCleanup(p.stop)

    def test_delegation_prompt_alone_never_passes_command_executed(self):
        prompt_only = [{
            "tool_calls": [{
                "function_name": "invoke_subagent",
                "arguments": {
                    "Subagents": [{
                        "TypeName":
                        "tpu-model-onboarding",
                        "Prompt":
                        ("Run inference.py, benchmark_tpu.py, smart_hf_search, "
                         "preflight_check, and engine_compatibility_check."),
                    }]
                },
            }]
        }]
        for task_dir in UNIT_TASKS:
            path = task_dir / "tests" / "command_executed" / "check.py"
            with self.subTest(task=task_dir.name):
                self.assertFalse(
                    _load_check(path, prompt_only).check_command_executed(
                        self.workspace))

    def test_command_executed_task_specific_validation(self):
        # 1. torch-inference-coder
        inf_path = (NIGHTLY / "torch-inference-coder" / "tests" /
                    "command_executed" / "check.py")
        server_file = self.workspace / "fastapi_server.py"
        server_file.write_text("raise NotImplementedError()", encoding="utf-8")
        self.assertFalse(
            _load_check(inf_path, []).check_command_executed(self.workspace))
        server_file.write_text(
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
            "device = get_device('tpu')\n",
            encoding="utf-8",
        )
        self.assertTrue(
            _load_check(inf_path, []).check_command_executed(self.workspace))

        # 2. torch-benchmark-coder
        bench_path = (NIGHTLY / "torch-benchmark-coder" / "tests" /
                      "command_executed" / "check.py")
        bench_file = self.workspace / "benchmark_tpu.py"
        bench_file.write_text(
            "raise NotImplementedError('Bespoke loader not implemented.')",
            encoding="utf-8",
        )
        self.assertFalse(
            _load_check(bench_path, []).check_command_executed(self.workspace))
        bench_file.write_text(
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
            "device = get_device('tpu')\nrun_benchmark_pass()\n",
            encoding="utf-8",
        )
        self.assertTrue(
            _load_check(bench_path, []).check_command_executed(self.workspace))

        # 3. engine-compatibility
        ec_path = (NIGHTLY / "engine-compatibility" / "tests" /
                   "command_executed" / "check.py")
        self.assertTrue(
            _load_check(
                ec_path,
                [{
                    "tool_calls": [{
                        "function_name": "run_command",
                        "arguments": {
                            "command":
                            ("compute run --code engine_compatibility_check.py "
                             "-- --model Qwen/Qwen2.5-7B-Instruct --devices 4")
                        },
                    }]
                }],
            ).check_command_executed(self.workspace))
        self.assertFalse(
            _load_check(
                ec_path,
                [{
                    "tool_calls": [
                        {
                            "function_name": "run_command",
                            "arguments": {
                                "command": "pip install vllm"
                            },
                        },
                        {
                            "function_name": "run_command",
                            "arguments": {
                                "command":
                                ("compute run --code engine_compatibility_check.py "
                                 "-- --model Qwen/Qwen2.5-7B-Instruct --devices 4"
                                 )
                            },
                        },
                    ]
                }],
            ).check_command_executed(self.workspace))

        # 4. remote-tpu-runner
        rtr_path = (NIGHTLY / "remote-tpu-runner" / "tests" /
                    "command_executed" / "check.py")
        self.assertTrue(
            _load_check(
                rtr_path,
                [{
                    "tool_calls": [{
                        "function_name": "run_command",
                        "arguments": {
                            "command": "compute --package pkg inspect --json"
                        },
                    }],
                    "observation": {
                        "results": [{
                            "content":
                            '{"status": "busy", "chips": 8, "available_chips": 0}'
                        }]
                    },
                }],
            ).check_command_executed(self.workspace))
        self.assertFalse(
            _load_check(
                rtr_path,
                [{
                    "tool_calls": [{
                        "function_name": "run_command",
                        "arguments": {
                            "command": "compute --package pkg inspect --json"
                        },
                    }],
                    "observation": {
                        "results": [{
                            "content":
                            '{"status": "no_accelerator", "chips": 0, "available_chips": 0}'
                        }]
                    },
                }],
            ).check_command_executed(self.workspace))


if __name__ == "__main__":
    unittest.main()
