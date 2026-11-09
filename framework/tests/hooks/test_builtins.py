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
"""Unit tests for the builtin validate-schema and skill-tools hooks."""

import contextlib
import functools
import io
import json
import os
import pathlib
import sys
import unittest
from typing import Any
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import agent_contracts, paths, runtime  # noqa: E402
from framework.hooks import actions, dispatch, events  # noqa: E402
from framework.tests.hooks import helpers  # noqa: E402


def setUpModule():
    helpers.set_up_module()


def tearDownModule():
    helpers.tear_down_module()


class TestBuiltinValidateSchema(unittest.TestCase):
    """Unit tests for builtin:validate-schema hook execution."""

    def setUp(self):
        super().setUp()
        self.temp_dir, self.workspace_dir = helpers.isolated_workspace(
            self, "central_tpu_agent_suite")
        self.central_home = self.temp_dir / "central_tpu_agent_suite"
        self.central_home.mkdir()

        self.pkg_root = self.central_home / paths.PACKAGES_DIRNAME / "test-pkg@1.0.0"
        (self.pkg_root / "schemas").mkdir(parents=True)
        string = {"type": "string"}

        def schema(name, required, properties, example=None):
            doc = {
                "type": "object",
                "additionalProperties": False,
                "required": required,
                "properties": properties,
            }
            if example:
                doc["examples"] = [example]
            (self.pkg_root / "schemas" / name).write_text(json.dumps(doc),
                                                          encoding="utf-8")

        schema("helper-request.json", ["prompt"], {
            "prompt": string,
            "run_id": string
        })
        schema("helper-result.json", ["status"], {
            "status": string,
            "summary": string,
            "artifacts": {
                "type": "array"
            },
        }, {
            "status": "completed",
            "artifacts": [{
                "file": "summary.json"
            }]
        })
        # Like the shipped request schemas that make run_id mandatory; its
        # placeholder example is what the denial must not let win.
        schema("strict-request.json", ["prompt", "run_id"], {
            "prompt": string,
            "run_id": string
        }, {
            "prompt": "Optimize TPU kernel",
            "run_id": "20260914T120000Z-000000000000"
        })
        (self.pkg_root / "tpu-agent-suite-build.json").write_text(
            json.dumps({
                "contracts": {
                    "helper": {
                        "accepts": "schemas/helper-request.json",
                        "produces": "schemas/helper-result.json",
                    },
                    "strict-worker": {
                        "accepts": "schemas/strict-request.json",
                        "produces": "",
                    },
                    "uncontracted": {
                        "accepts": "",
                        "produces": "",
                    },
                },
            }),
            encoding="utf-8",
        )

    def _validate(self, event_payload: dict) -> dict:
        """`builtin_validate_schema` on the test package and workspace."""
        return agent_contracts.builtin_validate_schema(
            package="test-pkg@1.0.0",
            workspace=self.workspace_dir,
            event_payload=event_payload,
        )

    def _delegate_event(self, tool_input):
        """Build a delegate event the way a harness actually produces one."""
        return events._normalize_tool_event(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Agent",
                "tool_input": tool_input,
                "session_id": "sess-1",
            },
            "claude-code",
            "PreToolUse",
            "test-pkg@1.0.0",
            self.workspace_dir,
        )

    def test_delegate_validation_ignores_harness_routing_keys(self):
        """A plain-text delegation validates as its prompt, not as the harness's argument bag.

        Normalization seeds the canonical args with the harness's own tool_input,
        so `subagent_type` and `description` ride along into the validator. A
        closed accepts schema rejected both, which made every plain-text
        delegation impossible and left a hand-wrapped JSON prompt as the only
        way through. This goes through `_normalize_tool_event` rather than
        around it on purpose: a hand-built args dict is already clean, so it
        cannot reproduce the leak.
        """
        event = self._delegate_event({
            "subagent_type": "helper",
            "description": "Optimize a kernel",
            "prompt": "Optimize TPU kernel",
        })
        self.assertIn("subagent_type", event["args"])
        self.assertEqual(self._validate(event)["decision"], "allow")

        # The keys come from tool-mappings.json, so every harness's are
        # dropped; a dotted spec (Antigravity's `Subagents.TypeName`)
        # contributes its root, which is the key the bag carries. agy 1.4.x
        # sends a single delegation flat, without the Subagents array; the
        # array form still wins when both are present.
        flat = {"TypeName": "flat-agent", "Prompt": "flat"}
        nested = {"Subagents": [{"TypeName": "helper", "Prompt": "nested"}]}
        for args, subagent, prompt in ((flat, "flat-agent", "flat"), ({
                **flat,
                **nested
        }, "helper", "nested")):
            with self.subTest(subagent=subagent):
                event = events.normalize_inbound_event(
                    raw_payload={
                        "toolCall": {
                            "name": "invoke_subagent",
                            "args": args
                        }
                    },
                    harness="antigravity",
                    event="PreToolUse",
                    package="test-pkg@1.0.0",
                    workspace=self.workspace_dir,
                )
                self.assertEqual(event["tool"], "delegate")
                self.assertEqual(event["args"]["subagent"], subagent)
                self.assertEqual(event["args"]["prompt"], prompt)
        self.assertIn("Subagents", event["args"])
        self.assertEqual(self._validate(event)["decision"], "allow")

    def test_missing_required_run_id_is_denied_at_the_caller(self):
        """A run_id the subagent cannot see must be refused while it can be fixed.

        Supplying the context run_id to the validator alone would let the
        delegation through while the subagent still receives a prompt without
        one. It cannot recover it either: subagents bypass SessionStart, so the
        prompt is their only source, and five shipped result schemas require
        the value. The denial has to land on the caller, which holds the
        run_id and can resend.
        """
        run_id = "20260913T120000Z-abcdef123456"
        res = self._validate({
            "event": "PreToolUse",
            "tool": "delegate",
            "subagent": "strict-worker",
            "args": {
                "subagent": "strict-worker",
                "prompt": json.dumps({"prompt": "Optimize TPU kernel"}),
            },
            "context": {
                "run_id": run_id
            },
        })
        self.assertEqual(res["decision"], "deny")
        # The example carries a placeholder run_id, and it is the most
        # copyable thing in the message, so the real one has to be stated.
        self.assertIn(run_id, res["reason"])
        self.assertLess(res["reason"].index(run_id),
                        res["reason"].index("20260914T120000Z-000000000000"))

        supplied = self._validate({
            "event": "PreToolUse",
            "tool": "delegate",
            "subagent": "strict-worker",
            "args": {
                "subagent":
                "strict-worker",
                "prompt":
                json.dumps({
                    "prompt": "Optimize TPU kernel",
                    "run_id": run_id,
                }),
            },
            "context": {
                "run_id": run_id
            },
        })
        self.assertEqual(supplied["decision"], "allow")

    def test_skipped_subagent_stop_says_why_on_stderr(self):
        """A stop the gate filters must not vanish silently.

        Claude Code sends an empty `agent_type` for its own internal agents.
        The hooks skip that stop, and stderr says so and why, so a missing
        study record can be told apart from a hook that never ran.
        """
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            res, code = dispatch.dispatch_hook(
                harness="claude-code",
                package="test-pkg@1.0.0",
                workspace=self.workspace_dir,
                hook_name="validate-subagent-result",
                event="SubagentStop",
                raw_input_text=json.dumps({
                    "hook_event_name": "SubagentStop",
                    "session_id": "sess-internal",
                    "agent_id": "agent-suggest-1",
                    "agent_type": "",
                    "last_assistant_message": "not JSON",
                }),
                builtin="validate-schema",
            )
        self.assertEqual(code, 0)
        self.assertNotEqual(res.get("decision"), "block")
        self.assertIn("[validate-subagent-result] SubagentStop skipped",
                      stderr.getvalue())
        self.assertIn("no subagent name", stderr.getvalue())

    def test_subagent_stop_still_requires_artifacts_when_schema_declares_them(
            self):
        """Narrowing the artifact rule must not switch it off where it applies.

        `helper-result.json` declares an `artifacts` array, so a completed result
        that omits one is still prose pretending to be a deliverable.
        """
        res = self._validate({
            "event":
            "SubagentStop",
            "subagent":
            "helper",
            "output":
            json.dumps({
                "status": "completed",
                "summary": "done",
            }),
        })
        self.assertEqual(res["decision"], "deny")

    def test_subagent_without_contract_allowed(self):
        """Subagents without declared contracts pass through unconditionally."""
        res_pre = self._validate({
            "event": "PreToolUse",
            "tool": "delegate",
            "subagent": "uncontracted",
            "args": {
                "any_arg": "value"
            },
        })
        self.assertEqual(res_pre["decision"], "allow")

        res_stop = self._validate({
            "event": "SubagentStop",
            "subagent": "uncontracted",
            "output": "Raw prose output is fine",
        })
        self.assertEqual(res_stop["decision"], "allow")


class TestBuiltinSkillTools(unittest.TestCase):
    """Tests for builtin:skill-tools, which keeps a skill's tool limit for the session."""

    def setUp(self):
        super().setUp()
        self.temp_dir, self.workspace_dir = helpers.isolated_workspace(
            self, "central_tpu_agent_suite")
        self.central_home = self.temp_dir / "central_tpu_agent_suite"
        self.pkg_root = self.central_home / paths.PACKAGES_DIRNAME / "test-pkg@1.0.0"
        self.pkg_root.mkdir(parents=True)
        self._write_build_info({
            "router": ["ask", "delegate", "read", "Skill", "ToolSearch"],
            "exporter":
            None,
        })

    def _write_build_info(self, skill_allowed_tools):
        build_info = {"version": "1.0.0", "source": "test-pkg"}
        if skill_allowed_tools is not None:
            build_info["skill_allowed_tools"] = skill_allowed_tools
        (self.pkg_root / "tpu-agent-suite-build.json").write_text(
            json.dumps(build_info), encoding="utf-8")

    def _run(self, event, tool, args=None, session="s1", raw_input=None):
        payload: dict[str, Any] = {
            "event": event,
            "tool": tool,
            "args": args or {},
            "context": {
                "session_id": session
            },
            "raw_input": raw_input or {},
        }
        if event == "PostToolUse":
            payload["success"] = True
        return actions._builtin_skill_tools(package="test-pkg@1.0.0",
                                            workspace=self.workspace_dir,
                                            event_payload=payload)

    def _invoke_skill(self, name, **kwargs):
        return self._run("PostToolUse", "Skill", {"skill": name}, **kwargs)

    def _shell(self, **kwargs):
        return self._run("PreToolUse", "shell", {"command": "ls"}, **kwargs)

    def test_limited_skill_still_allows_this_packages_setup_commands(self):
        """The main thread can set compute up; nothing else gets the shell."""
        self._invoke_skill("router")
        cmd = functools.partial(runtime.compute.config.runtime_command,
                                "test-pkg@1.0.0")
        runtime_py = runtime.compute.config.RUNTIME
        # The rendered guide names the runtime as `~/...`.
        home = mock.patch.dict(os.environ, {"HOME": str(runtime_py.parent)})
        for command, decision in (
            (cmd("compute",
                 "setup apply --profile p --answers '{\"a\": 1}'"), "allow"),
            (cmd("config", "show").replace(str(runtime_py),
                                           "~/" + runtime_py.name), "allow"),
            (cmd("compute", "run --script x.py"), "deny"),
            (cmd("compute", "setup verify; rm -rf ~"), "deny"),
            (cmd("compute", "setup verify\nrm -rf ~"), "deny"),
            (cmd("compute", "setup plan --profile $(whoami)"), "deny"),
            (cmd("config", "show").replace("test-pkg@1.0.0",
                                           "other@1"), "deny"),
        ):
            with self.subTest(command=command), home:
                self.assertEqual(
                    self._run("PreToolUse", "shell",
                              {"command": command})["decision"], decision)

    def test_subagents_are_not_limited(self):
        """Events from a subagent are allowed and don't change the main thread's limit."""
        sub = {"agent_id": "a1"}
        self._invoke_skill("router")
        self.assertEqual(self._shell(raw_input=sub)["decision"], "allow")
        self._invoke_skill("exporter", raw_input=sub)
        self.assertEqual(self._shell()["decision"], "deny")

    def test_unlimited_package_skill_lifts_the_limit(self):
        """Loading a package skill with no tool limit lifts it.

        A skill this package doesn't ship leaves the limit in place.
        """
        self._invoke_skill("router")
        self._invoke_skill("other-plugin:helper")
        self.assertEqual(self._shell()["decision"], "deny")
        self._invoke_skill("/exporter")
        self.assertEqual(self._shell()["decision"], "allow")


if __name__ == "__main__":
    unittest.main()
