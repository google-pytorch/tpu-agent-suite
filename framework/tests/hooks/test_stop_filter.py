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
"""Unit tests for the stop filter and the background idle gate."""

import contextlib
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import bg_ledger, paths  # noqa: E402
from framework.hooks import dispatch, events  # noqa: E402
from framework.tests.hooks import helpers  # noqa: E402

Path = pathlib.Path


def setUpModule():
    helpers.set_up_module()


def tearDownModule():
    helpers.tear_down_module()


class TestBackgroundIdleGate(unittest.TestCase):
    """A Claude Code subagent's stop with its own background work still live.

    The runtime records each launch a `PostToolUse` reports, per agent, and
    intersects them with the stop's live-task list; payload shapes are as
    captured on Claude Code 2.1.312. A not-idle stop is skipped when the
    hook's environment shows the harness resumes the agent
    (`CLAUDE_CODE_ENTRYPOINT=cli`, an interactive session) and refused
    otherwise (`sdk-cli`, headless, or unset).
    """

    PKG = "test-pkg@1.0.0"
    SESSION = "sess-bg"
    PARENT = "a80d76da82903c174"
    CHILD = "a7e703afaf7629f00"
    SHELL = "b4xiddsfx"
    GOOD = '{"status": "completed", "summary": "done", "artifacts": [{"file": "out.json"}]}'

    def setUp(self):
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.workspace_dir = self.temp_dir / "workspace"
        self.workspace_dir.mkdir()
        self.central_home = self.temp_dir / "central_tpu_agent_suite"
        # Snapshot the environment before touching it, so cleanup restores
        # the caller's, not this test's. The gate reads the regime off the
        # hook's environment: start every test with no signal, whoever runs
        # the suite (a Claude Code session sets it to `cli`); the
        # interactive test sets it itself.
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ["TPU_AGENT_SUITE_HOME"] = str(self.central_home)
        os.environ.pop("CLAUDE_CODE_ENTRYPOINT", None)
        pkg_root = self.central_home / paths.PACKAGES_DIRNAME / self.PKG
        (pkg_root / "schemas").mkdir(parents=True)
        (pkg_root / "schemas" / "result.json").write_text(json.dumps({
            "type":
            "object",
            "required": ["status"],
            "properties": {
                "status": {
                    "type": "string"
                },
                "summary": {
                    "type": "string"
                },
                "artifacts": {
                    "type": "array"
                },
            },
        }),
                                                          encoding="utf-8")
        (pkg_root / "tpu-agent-suite-build.json").write_text(json.dumps({
            "contracts": {
                "parent-orch": {
                    "accepts": "",
                    "produces": "schemas/result.json"
                }
            },
        }),
                                                             encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def _dispatch(self, event, payload, builtin, **kwargs):
        return dispatch.dispatch_hook(
            harness="claude-code",
            package=self.PKG,
            hook_name=f"test-{builtin}",
            event=event,
            builtin=builtin,
            workspace=self.workspace_dir,
            raw_input_text=json.dumps(payload),
            **kwargs,
        )

    def _post_tool(self, tool_name, tool_input, tool_response, agent_id=None):
        payload = {
            "hook_event_name": "PostToolUse",
            "session_id": self.SESSION,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": tool_response,
            "tool_use_id": "toolu_1",
        }
        if agent_id:
            payload["agent_id"] = agent_id
        return payload

    def _launch_agent(self, agent_id=PARENT, child=CHILD):
        return self._post_tool("Agent", {
            "subagent_type": "child-worker",
            "prompt": "go",
            "run_in_background": True
        }, {
            "status": "async_launched",
            "agentId": child,
            "isAsync": True
        },
                               agent_id=agent_id)

    def _launch_shell(self, agent_id=PARENT, task=SHELL):
        return self._post_tool("Bash", {
            "command": "sleep 10",
            "run_in_background": True
        }, {
            "backgroundTaskId": task,
            "stdout": "",
            "stderr": ""
        },
                               agent_id=agent_id)

    def _record_launches(self):
        for payload in (self._launch_agent(), self._launch_shell()):
            res, code = self._dispatch("PostToolUse", payload,
                                       "track-background")
            self.assertEqual((code, res.get("decision")), (0, None))

    def _stop(self, live, message="waiting for background work", **extra):
        payload = {
            "hook_event_name": "SubagentStop",
            "session_id": self.SESSION,
            "agent_id": self.PARENT,
            "agent_type": "parent-orch",
            "last_assistant_message": message,
            "agent_transcript_path": str(self.temp_dir / "missing.jsonl"),
            "transcript_path": str(self.temp_dir / "parent.jsonl"),
            "stop_hook_active": False,
            **extra,
        }
        if live is not None:
            payload["background_tasks"] = [{
                "id": task_id,
                "type": kind,
                "status": "running"
            } for task_id, kind in live]
        return payload

    def _ledger(self, agent_id=PARENT):
        return sorted(bg_ledger.load_ledger(self.SESSION, agent_id).keys())

    def _normalize_stop(self, payload):
        return events.normalize_inbound_event(payload, "claude-code",
                                              "SubagentStop", self.PKG,
                                              self.workspace_dir)

    def test_background_launches_are_recorded_under_their_launcher(self):
        self._record_launches()
        self.assertEqual(self._ledger(), sorted([self.CHILD, self.SHELL]))
        # A foreground delegate and a foreground command launch nothing.
        for payload in (
                self._post_tool("Agent", {"subagent_type": "x"}, {
                    "status": "completed",
                    "agentId": "a111"
                },
                                agent_id=self.PARENT),
                self._post_tool("Bash", {"command": "ls"}, {"stdout": ""},
                                agent_id=self.PARENT),
        ):
            self._dispatch("PostToolUse", payload, "track-background")
        self.assertEqual(self._ledger(), sorted([self.CHILD, self.SHELL]))
        # The root conversation has no agent id and files under its own key.
        self._dispatch("PostToolUse",
                       self._launch_agent(agent_id=None, child="a222"),
                       "track-background")
        self.assertEqual(self._ledger(agent_id=""), ["a222"])
        self.assertEqual(self._ledger(), sorted([self.CHILD, self.SHELL]))
        # Session end clears every agent's ledger for the session.
        res, code = self._dispatch(
            "SessionEnd", {
                "hook_event_name": "SessionEnd",
                "session_id": self.SESSION,
                "reason": "exit"
            }, "track-background")
        self.assertEqual((code, res.get("decision")), (0, None))
        self.assertFalse(
            bg_ledger.ledger_dir(self.SESSION, self.PARENT).parent.exists())

    def test_pause_on_a_headless_run_is_refused_with_the_live_handles(self):
        """Headless (`sdk-cli`), nothing resumes the agent: the stop is final
        and refused. An unset variable refuses the same way."""
        self._record_launches()
        stderr = io.StringIO()
        with mock.patch.dict(os.environ, {"CLAUDE_CODE_ENTRYPOINT": "sdk-cli"}), \
                contextlib.redirect_stderr(stderr):
            res, code = self._dispatch(
                "SubagentStop",
                self._stop([(self.PARENT, "subagent"),
                            (self.CHILD, "subagent"), (self.SHELL, "shell")]),
                "validate-schema")
        self.assertEqual(code, 0)
        self.assertEqual(res.get("decision"), "block")
        reason = res.get("reason", "")
        self.assertIn(f"subagent {self.CHILD}", reason)
        self.assertIn(f"shell {self.SHELL}", reason)
        self.assertIn("`sleep 20`", reason)
        self.assertNotIn("produces contract", reason)
        self.assertIn("SubagentStop refused", stderr.getvalue())
        # The refusal is the dispatcher's, so every hook on the event gives it.
        res, _ = self._dispatch("SubagentStop",
                                self._stop([(self.CHILD, "subagent")]),
                                "record-event")
        self.assertEqual(res.get("decision"), "block")

    def test_finished_work_is_pruned_and_the_stop_validated(self):
        self._record_launches()
        # An entry that cannot be read is skipped with a note, never deleted.
        stray = (bg_ledger.ledger_dir(self.SESSION, self.PARENT) /
                 "stray.json")
        stray.write_text("not json", encoding="utf-8")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            event = self._normalize_stop(
                self._stop([(self.PARENT, "subagent")]))
        self.assertIs(event["args"]["fully_idle"], True)
        self.assertEqual(event["args"]["fully_idle_source"], "ledger")
        self.assertNotIn("outstanding_work", event["args"])
        self.assertEqual(self._ledger(), [])
        self.assertTrue(stray.exists())
        self.assertIn("ignoring unreadable ledger entry", stderr.getvalue())
        res, _ = self._dispatch("SubagentStop",
                                self._stop([], message=self.GOOD),
                                "validate-schema")
        self.assertNotIn("decision", res)

    def test_missing_live_list_is_announced_and_the_stop_validated(self):
        self._record_launches()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            event = self._normalize_stop(self._stop(None))
            res, _ = self._dispatch("SubagentStop", self._stop(None),
                                    "validate-schema")
        self.assertNotIn("fully_idle", event["args"])
        self.assertIn("carries no 'background_tasks' list", stderr.getvalue())
        self.assertEqual(res.get("decision"), "block")
        self.assertIn("produces", res.get("reason", ""))
        # Nothing is pruned on a payload that could not say what is live.
        self.assertEqual(self._ledger(), sorted([self.CHILD, self.SHELL]))

    def test_an_unfilable_launch_is_a_logged_failure_not_a_silent_allow(self):
        """A launch the recorder cannot file would read as idle at the stop,
        so it is raised; the dispatcher logs it and on_failure still allows."""
        payload = self._launch_agent()
        del payload["session_id"]
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            # The manifest registers the recorder with on_failure allow.
            res, code = self._dispatch("PostToolUse",
                                       payload,
                                       "track-background",
                                       on_failure="allow")
        self.assertEqual(code, 0)
        self.assertNotIn("decision", res)  # allowed, as on_failure says
        self.assertIn("hook failed (exception)", stderr.getvalue())
        self.assertIn("no session id", stderr.getvalue())
        self.assertEqual(self._ledger(), [])

    def test_harness_reported_idle_flag_still_skips_on_antigravity(self):
        # Antigravity says `fullyIdle` itself and resumes the paused agent;
        # the ledger policy must not reach it.
        payload = {
            "typeName": "parent-orch",
            "conversationId": "conv-1",
            "terminationReason": "NO_TOOL_CALL",
            "fullyIdle": False,
            "finalModelOutput": "waiting",
        }
        event = events.normalize_inbound_event(payload, "antigravity",
                                               "SubagentStop", self.PKG,
                                               self.workspace_dir)
        self.assertEqual(event["args"]["fully_idle_source"], "harness")
        self.assertIsNone(bg_ledger.idle_gate_verdict(event, "antigravity"))
        res, code = dispatch.dispatch_hook(
            harness="antigravity",
            package=self.PKG,
            hook_name="test-validate-schema",
            event="SubagentStop",
            builtin="validate-schema",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps(payload),
        )
        self.assertEqual((code, res.get("decision")), (0, "allow"))


if __name__ == "__main__":
    unittest.main()
