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
"""Tests for runtime schema_checks.jsonl emission and schema_compliance verifier."""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from framework import agent_contracts

REPO = pathlib.Path(__file__).resolve().parents[2]
SOURCE = REPO / "eval" / "verifiers" / "common" / "schema_compliance.py"
NIGHTLY_ONBOARD_OPTIMIZE = (REPO / "eval" / "datasets" /
                            "nightly-onboard-optimize")
TPU_INFERENCE_VLLM = REPO / "eval" / "datasets" / "tpu-inference-vllm"
NIGHTLY_FUNCTIONAL = REPO / "eval" / "datasets" / "nightly-functional"
SMOKE_CHECK = REPO / "eval" / "datasets" / "smoke-check"
ONBOARD_DIR = REPO / "eval" / "verifiers" / "onboard"
HILLCLIMB_DIR = REPO / "eval" / "verifiers" / "onboard_hillclimb"
ONBOARD_CARRIERS = (
    "onboarding-gemma-4-31b-it",
    "onboarding-gpt-oss-20b",
    "onboarding-mistral-small-24b",
    "onboarding-qwen3-8-27b",
)
INFERENCE_CARRIERS = (
    "tpu-inference-gemma-4-31b-it-vllm",
    "tpu-inference-gpt-oss-20b-vllm",
    "tpu-inference-qwen3-8-27b-vllm",
)


def _load():
    stub = types.ModuleType("rewardkit")

    def _criterion(fn=None,
                   *,
                   description: str | None = None,
                   shared: bool = False):
        del description, shared
        return (lambda f: f) if fn is None else fn

    stub.criterion = _criterion
    with mock.patch.dict(sys.modules, {"rewardkit": stub}):
        spec = importlib.util.spec_from_file_location("schema_compliance",
                                                      SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


check = _load()

_SUITE_HOME = pathlib.Path(tempfile.mkdtemp(prefix="schema_compliance_suite_"))


def tearDownModule():
    shutil.rmtree(_SUITE_HOME, ignore_errors=True)


@functools.cache
def compiled() -> pathlib.Path:
    """Compile `package/` into `_SUITE_HOME/packages/tpu-agent-suite@0.1.0`."""
    out = _SUITE_HOME / "packages" / "tpu-agent-suite@0.1.0"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "framework" / "compiler.py"),
            "translate",
            str(REPO / "package"),
            "--harness",
            "claude-code",
            "--output",
            str(out),
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    return out


def example(agent: str, direction: str) -> dict:
    root = compiled()
    contracts = json.loads(
        (root / "tpu-agent-suite-build.json").read_text("utf-8"))["contracts"]
    schema = json.loads(
        (root / contracts[agent][direction]).read_text("utf-8"))
    return schema["examples"][0]


def fenced(payload: dict) -> str:
    return f"Payload:\n```json\n{json.dumps(payload)}\n```\n"


class SchemaComplianceTest(unittest.TestCase):

    def test_runtime_hook_emits_ledger_and_verifier_scores_it(self):
        pkg_root = compiled()
        with tempfile.TemporaryDirectory() as tmp:
            home = pathlib.Path(tmp) / "suite"
            logs_dir = pathlib.Path(tmp) / "logs"
            (home / "packages").mkdir(parents=True)
            (home / "packages" / "tpu-agent-suite@0.1.0").symlink_to(
                pkg_root, target_is_directory=True)

            env = {
                "TPU_AGENT_SUITE_HOME": str(home),
                "LOGS_DIR": str(logs_dir),
            }
            with mock.patch.dict(os.environ, env):
                run_id = "run-123"
                run_ledger = (home / "runs" / run_id / "artifacts" /
                              "schema_checks.jsonl")
                # 1. Uncontracted subagent -> allowed without recording a check;
                #    empty ledger scores 1.0 (score=None, checked=0).
                res = agent_contracts.builtin_validate_schema(
                    package="tpu-agent-suite",
                    workspace=pkg_root,
                    event_payload={
                        "event": "PreToolUse",
                        "tool": "delegate",
                        "subagent": "tpu-agent",
                        "context": {
                            "run_id": run_id
                        },
                        "args": {
                            "prompt": "hi"
                        },
                    },
                )
                self.assertEqual(res["decision"], "allow")
                self.assertFalse(run_ledger.exists())
                self.assertFalse(
                    (home / "runs" / "schema_checks.jsonl").exists())
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        check.schema_compliance(pathlib.Path("/workspace")),
                        1.0)

                # 2. Denied PreToolUse (missing required `query`).
                bad_req = dict(example("tpu-knowledge-base", "accepts"))
                del bad_req["query"]
                res = agent_contracts.builtin_validate_schema(
                    package="tpu-agent-suite",
                    workspace=pkg_root,
                    event_payload={
                        "event": "PreToolUse",
                        "tool": "delegate",
                        "subagent": "tpu-knowledge-base",
                        "context": {
                            "run_id": run_id
                        },
                        "args": {
                            "prompt": fenced(bad_req)
                        },
                    },
                )
                self.assertEqual(res["decision"], "deny")

                # 3. Allowed PreToolUse.
                good_req = example("tpu-knowledge-base", "accepts")
                res = agent_contracts.builtin_validate_schema(
                    package="tpu-agent-suite",
                    workspace=pkg_root,
                    event_payload={
                        "event": "PreToolUse",
                        "tool": "delegate",
                        "subagent": "tpu-knowledge-base",
                        "context": {
                            "run_id": run_id
                        },
                        "args": {
                            "prompt": fenced(good_req)
                        },
                    },
                )
                self.assertEqual(res["decision"], "allow")

                # 4. Denied SubagentStop (no JSON in output).
                res = agent_contracts.builtin_validate_schema(
                    package="tpu-agent-suite",
                    workspace=pkg_root,
                    event_payload={
                        "event": "SubagentStop",
                        "subagent": "tpu-knowledge-base",
                        "context": {
                            "run_id": run_id
                        },
                        "output": "Plain text without JSON",
                    },
                )
                self.assertEqual(res["decision"], "deny")

                # 5. Allowed SubagentStop.
                good_out = example("tpu-knowledge-base", "produces")
                res = agent_contracts.builtin_validate_schema(
                    package="tpu-agent-suite",
                    workspace=pkg_root,
                    event_payload={
                        "event": "SubagentStop",
                        "subagent": "tpu-knowledge-base",
                        "context": {
                            "run_id": run_id
                        },
                        "output": fenced(good_out),
                    },
                )
                self.assertEqual(res["decision"], "allow")
                self.assertTrue(run_ledger.is_file())
                self.assertFalse(
                    (home / "runs" / "schema_checks.jsonl").exists())

                # 6. Stage runs/<run_id>/artifacts/schema_checks.jsonl into
                #    $LOGS_DIR/artifacts/schema_checks.jsonl and score it.
                collected = logs_dir / "artifacts" / "schema_checks.jsonl"
                collected.parent.mkdir(parents=True)
                run_ledger.replace(collected)
                with contextlib.redirect_stdout(io.StringIO()):
                    score = check.schema_compliance(pathlib.Path("/workspace"))

            self.assertAlmostEqual(score, 0.5)
            summary = json.loads((logs_dir / "verifier" /
                                  "schema_compliance.json").read_text("utf-8"))
            self.assertEqual(summary["checked"], 4)
            self.assertEqual(summary["compliant"], 2)
            self.assertEqual(
                summary["by_agent"]["tpu-knowledge-base"],
                {
                    "accepts_checked": 2,
                    "accepts_compliant": 1,
                    "produces_checked": 2,
                    "produces_compliant": 1,
                },
            )
            self.assertEqual(
                summary["failures"],
                [
                    {
                        "subagent": "tpu-knowledge-base",
                        "direction": "accepts",
                        "errors": ["$/query: required property is missing"],
                    },
                    {
                        "subagent": "tpu-knowledge-base",
                        "direction": "produces",
                        "errors": ["no JSON object in output"],
                    },
                ],
            )


class CopiesTest(unittest.TestCase):

    def test_tasks_carry_identical_check_and_regraders_stage_it(self):
        import tomllib  # noqa: PLC0415 - 3.11+ stdlib

        source = SOURCE.read_bytes()
        all_tests = [
            ONBOARD_DIR / "tests",
            *(NIGHTLY_ONBOARD_OPTIMIZE / task / "tests"
              for task in ONBOARD_CARRIERS),
            *(TPU_INFERENCE_VLLM / task / "tests"
              for task in INFERENCE_CARRIERS),
            SMOKE_CHECK / "knowledge-qa-csat006" / "tests",
            NIGHTLY_FUNCTIONAL / "tpu-diagnostics" / "tests",
        ]
        for tests in all_tests:
            copy = tests / "schema_compliance" / "check.py"
            with self.subTest(tests=str(tests.relative_to(REPO))):
                self.assertEqual(copy.read_bytes(), source)
                reward = tomllib.loads(
                    (tests / "reward.toml").read_text("utf-8"))
                self.assertEqual(
                    reward["reward"][0]["weights"]["schema_compliance"], 0.0)

        actual_copies = sorted([
            *REPO.glob("eval/datasets/**/tests/schema_compliance/check.py"),
            *(ONBOARD_DIR / "tests" / "schema_compliance").glob("check.py"),
        ])
        self.assertEqual(
            actual_copies,
            sorted(t / "schema_compliance" / "check.py" for t in all_tests),
        )

        with tempfile.TemporaryDirectory() as tmp:
            scratch = pathlib.Path(tmp)
            for idx, grade_py in enumerate(
                (ONBOARD_DIR / "grade.py", HILLCLIMB_DIR / "grade.py")):
                spec = importlib.util.spec_from_file_location(
                    f"regrade_{idx}", grade_py)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                stage_dir = scratch / f"regrade_{idx}" / "tests"
                mod.stage_verifier_tests(
                    stage_dir,
                    mod.DEFAULT_VERIFIERS,
                    None,
                    "qwen",
                    logs_dir=scratch / f"regrade_{idx}" / "logs",
                )
                self.assertEqual(
                    (stage_dir / "schema_compliance" /
                     "check.py").read_bytes(),
                    source,
                )
                reward = tomllib.loads(
                    (stage_dir / "reward.toml").read_text("utf-8"))
                self.assertEqual(
                    reward["reward"][0]["weights"]["schema_compliance"], 0.0)


if __name__ == "__main__":
    unittest.main()
