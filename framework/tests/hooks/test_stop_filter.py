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
"""Unit tests for the stop filter: subagent handback and the background
idle gate."""

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
from framework.hooks import dispatch, events, stop_filter  # noqa: E402
from framework.tests.hooks import helpers  # noqa: E402

Path = pathlib.Path


def setUpModule():
    helpers.set_up_module()


def tearDownModule():
    helpers.tear_down_module()


class TestSubagentHandback(unittest.TestCase):
    """A Claude Code subagent's result arrives by one of two roads.

    Outside auto mode a subagent ends its turn and the result is its last
    message (native SubagentStop), in the foreground or the background. In
    auto mode every subagent must call the injected `SubagentHandback` tool
    instead, and only that call is delivered. The same SubagentStop hook is
    registered on both, and the contract must be enforced exactly once
    whichever road is taken.
    """

    PKG = "test-pkg@1.0.0"
    RESULT = {"status": "success", "summary": "done", "artifacts": []}

    def setUp(self):
        super().setUp()
        self.temp_dir, self.workspace_dir = helpers.isolated_workspace(
            self, "central_tpu_agent_suite")
        self.central_home = self.temp_dir / "central_tpu_agent_suite"

        scripts_dir = self.central_home / "packages" / self.PKG / "scripts"
        scripts_dir.mkdir(parents=True)
        self.fires = self.temp_dir / "fires.jsonl"
        # Records every fire, and refuses any output without a JSON status.
        (scripts_dir / "check.py").write_text(
            "import sys, json\n"
            "data = json.loads(sys.stdin.read())\n"
            f"with open({str(self.fires)!r}, 'a') as f:\n"
            "    f.write(json.dumps(data) + '\\n')\n"
            "if '\"status\"' in str(data.get('output') or ''):\n"
            "    print(json.dumps({'decision': 'allow'}))\n"
            "else:\n"
            "    print(json.dumps({'decision': 'deny', 'reason': 'bad',\n"
            "                      'followup_message': 'send JSON'}))\n",
            encoding="utf-8",
        )

    def _dispatch(self, payload):
        return dispatch.dispatch_hook(
            harness="claude-code",
            package=self.PKG,
            hook_name="check",
            event="SubagentStop",
            script_rel="scripts/check.py",
            workspace=self.workspace_dir,
            raw_input_text=json.dumps(payload),
        )

    def _handback(self, message, **extra):
        return {
            "hook_event_name": "PreToolUse",
            "tool_name": "SubagentHandback",
            "agent_type": "tpu-model-onboarding",
            "session_id": "sess-1",
            "tool_input": {
                "message": message
            },
            **extra,
        }

    def _fire_count(self):
        if not self.fires.exists():
            return 0
        return len(self.fires.read_text(encoding="utf-8").splitlines())

    def _write_transcript(self, records):
        path = self.temp_dir / "agent-1.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records),
                        encoding="utf-8")
        return path

    @staticmethod
    def _content(block):
        return {"message": {"content": [block]}}

    @staticmethod
    def _prompt(prompt_id, text, meta=False):
        # Every user record Claude Code writes for a run carries its promptId.
        record = {
            "type": "user",
            "promptId": prompt_id,
            "message": {
                "role": "user",
                "content": text
            },
        }
        if meta:
            record["isMeta"] = True
        return record

    def _native_stop(self,
                     transcript,
                     message="Report delivered.",
                     prompt_id=None,
                     permission_mode=None):
        payload = {
            "hook_event_name": "SubagentStop",
            "agent_type": "tpu-model-onboarding",
            "session_id": "sess-1",
            "last_assistant_message": message,
            "agent_transcript_path": str(transcript),
            # The parent session's file on Claude Code; must never be read.
            "transcript_path": str(self.temp_dir / "parent.jsonl"),
        }
        if prompt_id is not None:
            payload["prompt_id"] = prompt_id
        if permission_mode is not None:
            payload["permission_mode"] = permission_mode
        return payload

    def test_one_check_per_delivery_across_both_roads(self):
        # The recorded auto-mode sequence: the subagent writes its report as
        # text and ends its turn before calling the tool, then calls it, then
        # ends its turn again with a closing remark. Only the call reaches
        # the caller, so only the call is checked. The stop's mode decides
        # alone: the transcript, missing here, is not read.
        transcript = self.temp_dir / "missing.jsonl"
        before, _ = self._dispatch(
            self._native_stop(transcript, "draft", permission_mode="auto"))
        handback, _ = self._dispatch(self._handback(json.dumps(self.RESULT)))
        after, _ = self._dispatch(
            self._native_stop(transcript,
                              "Report delivered.",
                              permission_mode="auto"))

        self.assertNotIn("decision", before)
        self.assertEqual(handback["hookSpecificOutput"]["permissionDecision"],
                         "allow")
        self.assertNotIn("decision", after)
        fires = [
            json.loads(line)
            for line in self.fires.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(fires), 1)
        self.assertIs(fires[0]["args"]["handback"], True)
        self.assertEqual(fires[0]["subagent"], "tpu-model-onboarding")

    def _handback_call(self, call_id, result, is_error=False):
        # A SubagentHandback call and the tool result Claude Code records.
        tool_result = {
            "type": "tool_result",
            "tool_use_id": call_id,
            "content": result
        }
        if is_error:
            tool_result["is_error"] = True
        return [
            self._content({
                "type": "tool_use",
                "id": call_id,
                "name": "SubagentHandback"
            }),
            self._content(tool_result),
        ]

    def _auto_run_then_resume(self):
        # Recorded shape of a subagent that handed back in auto mode (run
        # "p1") and was then resumed in default mode (run "p2") in the same
        # transcript: Claude Code withdraws the tool for the resumed run, and
        # its plain final message is what the caller receives.
        return self._write_transcript([
            self._prompt("p1", "harness test, part 1"),
            *self._handback_call("t1", "Report delivered to your caller."),
            self._prompt("p2", "harness test, part 2", meta=True),
            self._prompt("p2",
                         "<system-reminder>SubagentHandback is not available "
                         "in this run ...</system-reminder>",
                         meta=True),
            self._content({
                "type": "text",
                "text": "handshake-2"
            }),
        ])

    def test_stop_without_a_known_run_is_held(self):
        # A payload that cannot name its run, or names one no record bears,
        # proves nothing, like an unreadable transcript: an earlier run's
        # hand-back must not exempt it, so the native stop is checked.
        transcript = self._auto_run_then_resume()
        for prompt_id in (None, "p9"):
            with self.subTest(prompt_id=prompt_id):
                with contextlib.redirect_stderr(io.StringIO()) as stderr:
                    res, _ = self._dispatch(
                        self._native_stop(transcript,
                                          "handshake-2",
                                          prompt_id=prompt_id))
                self.assertEqual(res.get("decision"), "block")
                self.assertIn("cannot locate the stopping run",
                              stderr.getvalue())
        self.assertEqual(self._fire_count(), 2)

    def test_auto_mode_run_without_an_offer_is_held(self):
        # Recorded on Claude Code 2.1.241, headless and interactive alike:
        # auto mode, but the run is never given the tool, so its plain final
        # message is what the caller receives. Skipping it on the mode alone
        # let every result through unchecked. The meta record that withdraws
        # the tool names it too; it must not read as the offer.
        for label, record in (
            ("never offered",
             self._content({
                 "type": "text",
                 "text": "Brief and plan written."
             })),
            ("withdrawn",
             self._prompt("p1",
                          "<system-reminder>SubagentHandback is not available "
                          "in this run ...</system-reminder>",
                          meta=True)),
        ):
            with self.subTest(label):
                transcript = self._write_transcript(
                    [self._prompt("p1", "harness test"), record])
                res, _ = self._dispatch(
                    self._native_stop(transcript,
                                      "handshake",
                                      prompt_id="p1",
                                      permission_mode="auto"))
                self.assertEqual(res.get("decision"), "block")
        self.assertEqual(self._fire_count(), 2)

    def test_auto_mode_run_offered_the_tool_is_a_turn_boundary(self):
        # Recorded on Claude Code 2.1.285: the offer is a meta record at the
        # start of the run, before its first stop, so the draft stop before
        # the call is let through without waiting for the call. The skip is
        # not reported: the offer shows the run has the tool.
        for offer in ("<system-reminder>\nYour final report is delivered "
                      "through SubagentHandback: when your work is "
                      "complete, call SubagentHandback({message: <your "
                      "full report>}).</system-reminder>",
                      "Your final report is delivered through "
                      "SubagentHandback."):
            with self.subTest(offer=offer):
                transcript = self._write_transcript([
                    self._prompt("p1", "reply with pong"),
                    self._prompt("p1", offer, meta=True),
                ])
                with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
                    res, _ = self._dispatch(
                        self._native_stop(transcript,
                                          "draft",
                                          prompt_id="p1",
                                          permission_mode="auto"))
                self.assertNotIn("decision", res)
                self.assertEqual(self._fire_count(), 0)
                self.assertNotIn("SubagentStop skipped", err.getvalue())

    def test_accepted_handback_exempts_stops_in_any_mode(self):
        # Switching out of auto mode after the call was accepted must not
        # hold the closing remark: that is the block-on-every-retry loop.
        transcript = self._write_transcript([
            self._prompt("p1", "harness test"),
            *self._handback_call("t1", "Report delivered to your caller."),
        ])
        res, _ = self._dispatch(
            self._native_stop(transcript,
                              "Report delivered.",
                              prompt_id="p1",
                              permission_mode="default"))
        self.assertNotIn("decision", res)
        self.assertEqual(self._fire_count(), 0)


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
        event = events.normalize_inbound_event(
            {
                "typeName": "parent-orch",
                "conversationId": "conv-1",
                "terminationReason": "NO_TOOL_CALL",
                "fullyIdle": False,
                "finalModelOutput": "waiting",
            }, "antigravity", "SubagentStop", self.PKG, self.workspace_dir)
        self.assertEqual(event["args"]["fully_idle_source"], "harness")
        self.assertEqual(
            stop_filter.subagent_stop_skip_reason(event, "antigravity"),
            "subagent still has work outstanding (not fully idle)")
        self.assertIsNone(bg_ledger.idle_gate_verdict(event, "antigravity"))


if __name__ == "__main__":
    unittest.main()
