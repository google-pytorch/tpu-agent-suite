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
"""The nightly invoked_subagent check, on Claude Code and Antigravity runs.

Offline: rewardkit is stubbed, and each case points the check at a
trajectory on disk. The suite's agent names come, as in a trial, from the
package framework/compiler.py builds from package/, compiled once per harness
for the whole run.

fixtures/invoked_subagent/ holds two real Harbor trajectories, cut down to
their delegation calls and the fields the check reads, one step per line,
with the values the harness wrote (prompts and result text dropped):

- claude_code.trajectory.json: a nightly Claude Code run that delegated to
  the suite's agents, one delegation rejected by a hook and retried.
- antigravity.trajectory.json: a nightly Antigravity run whose every
  invoke_subagent call, suite agents included, failed with "not found or not
  allowed to be invoked".

The hand-written cases below cover what those two runs do not.
"""

import functools
import importlib.util
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parents[2]
SOURCE = REPO / "eval" / "verifiers" / "common" / "invoked_subagent.py"
NIGHTLY_ONBOARD_OPTIMIZE = (REPO / "eval" / "datasets" /
                            "nightly-onboard-optimize")
TPU_INFERENCE_VLLM = REPO / "eval" / "datasets" / "tpu-inference-vllm"
ONBOARDING_FRONTIER = REPO / "eval" / "datasets" / "onboarding-frontier"
NIGHTLY_FUNCTIONAL = REPO / "eval" / "datasets" / "nightly-functional"
FIXTURES = pathlib.Path(
    __file__).resolve().parent / "fixtures" / "invoked_subagent"
CSAT006_TESTS = (REPO / "eval" / "datasets" / "smoke-check" /
                 "knowledge-qa-csat006" / "tests")
ONBOARD_CARRIERS = (
    "onboarding-gemma-4-31b-it",
    "onboarding-gpt-oss-20b",
    "onboarding-mistral-small-24b",
    "onboarding-qwen3-8-27b",
)
TPU_INFERENCE_CARRIERS = (
    "tpu-inference-gemma-4-31b-it-vllm",
    "tpu-inference-gpt-oss-20b-vllm",
    "tpu-inference-qwen3-8-27b-vllm",
)
FRONTIER_CARRIERS = ("deepseek-v4-1-flash", "glm-5-3-flash")
CARRIERS = ONBOARD_CARRIERS + TPU_INFERENCE_CARRIERS + FRONTIER_CARRIERS


def _load():
    stub = types.ModuleType("rewardkit")
    stub.criterion = lambda **_: (lambda fn: fn)
    with mock.patch.dict(sys.modules, {"rewardkit": stub}):
        spec = importlib.util.spec_from_file_location("invoked_subagent",
                                                      SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


check = _load()

_BUILD_ROOT = pathlib.Path(tempfile.mkdtemp(prefix="invoked_subagent_"))


def tearDownModule():
    shutil.rmtree(_BUILD_ROOT, ignore_errors=True)


@functools.cache
def compiled(harness: str) -> pathlib.Path:
    """package/ compiled for `harness`, as eval staging compiles it."""
    out = _BUILD_ROOT / harness
    subprocess.run([
        sys.executable,
        str(REPO / "framework" / "compiler.py"), "translate",
        str(REPO / "package"), "--harness", harness, "--output",
        str(out)
    ],
                   cwd=REPO,
                   check=True,
                   capture_output=True)
    return out


def run(trajectory: pathlib.Path,
        harness: str | None = "claude-code",
        params: pathlib.Path | None = None):
    """The criterion on `trajectory`, with `harness`'s package installed and
    `params` as the task's verifier_params.json (none by default)."""
    manifest = (compiled(harness) /
                "tpu-agent-suite-build.json" if harness else _BUILD_ROOT /
                "absent")
    with mock.patch.multiple(check,
                             _TRAJECTORY=str(trajectory),
                             _BUILD_MANIFESTS=str(manifest),
                             _PARAMS=str(params or _BUILD_ROOT / "absent")):
        return check.invoked_subagent(pathlib.Path("/workspace"))


class SuiteAgentsTest(unittest.TestCase):
    """Which agents are the suite's."""

    def test_manifest_lists_every_agent_the_package_installs(self):
        for harness in ("claude-code", "antigravity"):
            dist = compiled(harness)
            installed = {
                re.search(r"^name:\s*\"?([^\"\n]+?)\"?\s*$",
                          p.read_text("utf-8"), re.MULTILINE).group(1)
                for p in (dist / "agents").glob("*.md")
            }
            manifest = str(dist / "tpu-agent-suite-build.json")
            with self.subTest(harness=harness), mock.patch.object(
                    check, "_BUILD_MANIFESTS", manifest):
                agents = check.suite_agents()
                self.assertIn("tpu-model-onboarding", agents)
                self.assertEqual(agents, installed)

    def test_baseline_without_the_suite_has_no_agents(self):
        with mock.patch.object(check, "_BUILD_MANIFESTS",
                               str(_BUILD_ROOT / "absent")):
            self.assertEqual(check.suite_agents(), set())


class CopiesTest(unittest.TestCase):

    def test_tpu_inference_tasks_carry_the_check(self):
        source = SOURCE.read_bytes()
        task_roots = [
            *((NIGHTLY_ONBOARD_OPTIMIZE, t) for t in ONBOARD_CARRIERS),
            *((TPU_INFERENCE_VLLM, t) for t in TPU_INFERENCE_CARRIERS),
            *((ONBOARDING_FRONTIER, t) for t in FRONTIER_CARRIERS),
        ]
        for dataset_root, task in task_roots:
            tests = dataset_root / task / "tests"
            copy = tests / "invoked_subagent" / "check.py"
            with self.subTest(task=task):
                self.assertTrue(copy.is_file(), f"{copy} is missing")
                self.assertEqual(
                    copy.read_bytes(), source,
                    f"{copy} differs from {SOURCE.relative_to(REPO)}")
                self.assertIn("invoked_subagent = 0.0",
                              (tests / "reward.toml").read_text("utf-8"))
                self.assertIn('"invoked_subagent"',
                              (tests / "test.sh").read_text("utf-8"))

    def test_csat006_carries_it_for_tpu_knowledge_base(self):
        # The smoke check's one knowledge case: a citation alone can be
        # copied, so it also needs the delegation to have happened.
        copy = CSAT006_TESTS / "invoked_subagent" / "check.py"
        self.assertEqual(copy.read_bytes(), SOURCE.read_bytes(),
                         f"{copy} differs from {SOURCE.relative_to(REPO)}")
        params = json.loads(
            (CSAT006_TESTS / "verifier_params.json").read_text("utf-8"))
        self.assertEqual(params["invoked_subagent"],
                         {"agents": ["tpu-knowledge-base"]})

    def test_no_other_nightly_task_carries_it(self):
        carriers = sorted(p.parents[2].name for root in (
            NIGHTLY_ONBOARD_OPTIMIZE,
            TPU_INFERENCE_VLLM,
            NIGHTLY_FUNCTIONAL,
            ONBOARDING_FRONTIER,
        ) for p in root.glob("**/tests/invoked_subagent/check.py"))
        self.assertEqual(carriers, sorted(CARRIERS))


class RealTrajectoryTest(unittest.TestCase):
    """The check on trajectories Harbor wrote for real nightly runs."""

    @staticmethod
    def _delegations(fixture):
        return check.delegations(json.loads((FIXTURES / fixture).read_text()))

    def test_claude_code_run_that_used_the_suite_passes(self):
        # Agent tool, bare subagent_type; the hook-rejected call is flagged
        # (tool_result_is_error) and its retry is not.
        self.assertEqual(self._delegations("claude_code.trajectory.json"),
                         [("tpu-knowledge-base", False),
                          ("tpu-diagnostics", True),
                          ("tpu-diagnostics", False),
                          ("tpu-diagnostics", False)])
        self.assertTrue(run(FIXTURES / "claude_code.trajectory.json"))

    def test_antigravity_run_whose_delegations_all_failed_fails(self):
        found = self._delegations("antigravity.trajectory.json")
        # The right agents, tried bare and in forms no harness accepts, and
        # every one came back "not found or not allowed" (status ERROR). The
        # call with an empty TypeName names no agent.
        self.assertEqual([name for name, _ in found], [
            "tpu-knowledge-base", "tpu-model-onboarding", "tpu-agent",
            "tpu-agent-suite:tpu-model-onboarding",
            "tpu-agent-suite/tpu-model-onboarding", "agent",
            "tpu_model_onboarding", "tpu-hello-world", "coder", "browser",
            "browser_subagent", "tpu-knowledge-base", "tpu-model-onboarding"
        ])
        self.assertTrue(all(failed for _, failed in found))
        self.assertFalse(
            run(FIXTURES / "antigravity.trajectory.json", "antigravity"))


def _claude_step(name, args, *, is_error=None, result=True):
    step = {
        "tool_calls": [{
            "tool_call_id": "toolu_1",
            "function_name": name,
            "arguments": args
        }]
    }
    if result:
        extra = ({
            "tool_result_is_error": is_error
        } if is_error is not None else None)
        step["observation"] = {
            "results": [{
                "source_call_id": "toolu_1",
                "content": "done",
                "extra": extra
            }]
        }
    return step


def _antigravity_step(type_name, *, role="Onboarding", status="DONE"):
    return {
        "tool_calls": [{
            "tool_call_id": "antigravity-3-0",
            "function_name": "invoke_subagent",
            "arguments": {
                "Subagents": [{
                    "TypeName": type_name,
                    "Role": role,
                    "Prompt": "Run the preflight check."
                }]
            }
        }],
        "observation": {
            "results": [{
                "source_call_id": "antigravity-3-0",
                "content": "started",
                "extra": {
                    "type": "INVOKE_SUBAGENT",
                    "status": status
                }
            }]
        },
    }


class InvokedSubagentTest(unittest.TestCase):

    def setUp(self):
        tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        self.trajectory = tmp / "trajectory.json"

    def _passes(self, *steps, harness="claude-code") -> bool:
        self.trajectory.write_text(json.dumps({"steps": list(steps)}))
        return run(self.trajectory, harness)

    # Claude Code

    def test_claude_agent_tool_to_suite_agent(self):
        self.assertTrue(
            self._passes(
                _claude_step("Agent",
                             {"subagent_type": "tpu-model-onboarding"})))

    def test_claude_legacy_task_tool_and_plugin_namespace(self):
        self.assertTrue(
            self._passes(
                _claude_step(
                    "Task",
                    {"subagent_type": "tpu-agent-suite:tpu-knowledge-base"})))

    def test_claude_generic_subagent_is_not_the_suite(self):
        self.assertFalse(
            self._passes(
                _claude_step(
                    "Agent", {
                        "subagent_type": "general-purpose",
                        "description": "tpu-model-onboarding",
                        "prompt": "Act as tpu-model-onboarding."
                    })))

    def test_claude_errored_delegation_does_not_count(self):
        self.assertFalse(
            self._passes(
                _claude_step("Agent",
                             {"subagent_type": "tpu-model-onboarding"},
                             is_error=True)))

    def test_claude_delegation_still_running_counts(self):
        self.assertTrue(
            self._passes(
                _claude_step("Agent",
                             {"subagent_type": "tpu-model-onboarding"},
                             result=False)))

    def test_claude_reading_the_skill_is_not_a_delegation(self):
        self.assertFalse(
            self._passes(
                _claude_step("Skill", {"skill": "tpu-agent"}),
                _claude_step("Read", {
                    "file_path":
                    "/root/.claude/agents/tpu-knowledge-base.md"
                })))

    # Antigravity

    def test_antigravity_invoke_subagent_to_suite_agent(self):
        self.assertTrue(
            self._passes(_antigravity_step("tpu-model-onboarding"),
                         harness="antigravity"))

    def test_antigravity_suite_name_only_as_role_is_not_the_suite(self):
        self.assertFalse(
            self._passes(_antigravity_step("self",
                                           role="tpu-model-onboarding"),
                         harness="antigravity"))

    def test_antigravity_agent_not_found_does_not_count(self):
        self.assertFalse(
            self._passes(_antigravity_step("tpu-model-onboarding",
                                           status="ERROR"),
                         harness="antigravity"))

    # Neither

    def test_no_delegation_fails(self):
        self.assertFalse(
            self._passes(
                _claude_step("Bash",
                             {"command": "python3 preflight_check.py"})))

    def test_step_without_calls_or_observation_is_skipped(self):
        self.assertTrue(
            self._passes(
                {"observation": None},
                _claude_step("Agent",
                             {"subagent_type": "tpu-knowledge-base"})))

    def test_baseline_run_without_the_suite_fails(self):
        # The name alone is not the suite: with nothing installed, no agent
        # is the suite's, whatever the delegation calls it.
        self.assertFalse(
            self._passes(_claude_step(
                "Agent", {"subagent_type": "tpu-model-onboarding"}),
                         harness=None))

    # Required agents (verifier_params.json)

    def _params(self, agents) -> pathlib.Path:
        path = self.trajectory.parent / "verifier_params.json"
        path.write_text(json.dumps({"invoked_subagent": {"agents": agents}}))
        return path

    def test_required_agent_delegated_passes(self):
        self.trajectory.write_text(
            json.dumps({
                "steps": [
                    _claude_step("Agent",
                                 {"subagent_type": "tpu-knowledge-base"})
                ]
            }))
        self.assertTrue(
            run(self.trajectory, params=self._params(["tpu-knowledge-base"])))

    def test_another_suite_agent_does_not_satisfy_a_requirement(self):
        self.trajectory.write_text(
            json.dumps({
                "steps": [
                    _claude_step("Agent",
                                 {"subagent_type": "tpu-model-onboarding"})
                ]
            }))
        self.assertFalse(
            run(self.trajectory, params=self._params(["tpu-knowledge-base"])))

    def test_required_agent_on_antigravity_failed_delegation_fails(self):
        self.trajectory.write_text(
            json.dumps({
                "steps":
                [_antigravity_step("tpu-knowledge-base", status="ERROR")]
            }))
        self.assertFalse(
            run(self.trajectory,
                "antigravity",
                params=self._params(["tpu-knowledge-base"])))

    def test_unreadable_logs_score_zero_instead_of_raising(self):
        # Raised from the criterion, it would fail the whole verifier run.
        self.assertFalse(run(self.trajectory))  # Never written.
        self.trajectory.write_text("{not json")
        self.assertFalse(run(self.trajectory))


if __name__ == "__main__":
    unittest.main()
