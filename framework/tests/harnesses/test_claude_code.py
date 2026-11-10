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
"""Claude Code adaptor."""

from __future__ import annotations

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

from framework import harnesses  # noqa: E402
from framework.hooks import dispatch  # noqa: E402
from framework.tests.harnesses import conformance  # noqa: E402


class ClaudeCodeTest(conformance.HarnessConformance):
    harness = harnesses.require("claude-code")

    def test_stop_reply(self):
        """'retry' becomes `block`, the harness's resume signal."""
        retry = self._reply({
            "decision": "retry",
            "reason": "waited for quota"
        }, "Stop", "resume-after-quota")
        self.assertEqual(retry["decision"], "block")
        self.assertIn("waited", retry["reason"])
        self.assertEqual(retry["hookSpecificOutput"]["hookEventName"], "Stop")

        allow = self._reply({"decision": "allow"}, "Stop",
                            "resume-after-quota")
        self.assertNotIn("decision", allow)
        self.assertEqual(allow["hookSpecificOutput"]["hookEventName"], "Stop")

    def test_tool_replies(self):
        out = self._reply(
            {
                "decision": "allow",
                "updated_args": {
                    "prompt": "Sanitized prompt"
                }
            }, "PreToolUse")
        self.assertEqual(out["hookSpecificOutput"]["hookEventName"],
                         "PreToolUse")
        self.assertEqual(out["hookSpecificOutput"]["permissionDecision"],
                         "allow")
        self.assertEqual(out["hookSpecificOutput"]["updatedInput"],
                         {"prompt": "Sanitized prompt"})

        out = self._reply(
            {
                "decision": "deny",
                "reason": "Kernel check failed",
                "followup_message": "not consulted",  # unlike the stops
                "context": "Please retry with tile size 64"
            },
            "PostToolUse",
            "verifier")
        self.assertEqual(out["decision"], "block")
        self.assertEqual(out["reason"], "Kernel check failed")
        self.assertEqual(out["hookSpecificOutput"]["hookEventName"],
                         "PostToolUse")
        self.assertEqual(out["hookSpecificOutput"]["additionalContext"],
                         "Please retry with tile size 64")
        # Unlike the stops, an empty reason is omitted, not filled in.
        out = self._reply({
            "decision": "deny",
            "followup_message": "f"
        }, "PostToolUse")
        self.assertEqual(out["decision"], "block")
        self.assertNotIn("reason", out)


class ClaudeCodeHandbackTest(unittest.TestCase):
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
        self.temp_dir = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.workspace_dir = self.temp_dir / "workspace"
        self.workspace_dir.mkdir()
        self.central_home = self.temp_dir / "central_tpu_agent_suite"
        env_patch = mock.patch.dict(
            os.environ, {"TPU_AGENT_SUITE_HOME": str(self.central_home)})
        env_patch.start()
        self.addCleanup(env_patch.stop)

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

    def _handback_call(self, call_id, result):
        return [
            self._content({
                "type": "tool_use",
                "id": call_id,
                "name": "SubagentHandback"
            }),
            self._content({
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": result
            }),
        ]

    def test_one_check_per_delivery_across_both_roads(self):
        transcript = self.temp_dir / "missing.jsonl"
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            before, _ = self._dispatch(
                self._native_stop(transcript, "draft", permission_mode="auto"))
            handback, _ = self._dispatch({
                "hook_event_name": "PreToolUse",
                "tool_name": "SubagentHandback",
                "agent_type": "tpu-model-onboarding",
                "session_id": "sess-1",
                "tool_input": {
                    "message": json.dumps(self.RESULT)
                },
            })
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

    def test_native_stop_transcript_and_mode_matrix(self):
        handed_back = [
            self._prompt("p1", "t"),
            *self._handback_call("t1", "Report delivered to your caller."),
        ]
        no_tool = "<system-reminder>SubagentHandback is not available in this run ...</system-reminder>"
        offer_note = (
            "<system-reminder>\nYour final report is delivered through "
            "SubagentHandback: when your work is complete, call "
            "SubagentHandback({message: <your full report>}).</system-reminder>"
        )
        cases = [
            # (label, records, prompt_id, mode, held, expected_err)
            ("missing prompt_id", handed_back, None, None, True,
             "cannot locate the stopping run"),
            ("unknown prompt_id", handed_back, "p9", None, True,
             "cannot locate the stopping run"),
            ("auto never offered tool", [
                self._prompt("p1", "t"),
                self._content({
                    "type": "text",
                    "text": "done"
                }),
            ], "p1", "auto", True, None),
            ("withdrawn tool", [
                self._prompt("p1", "t"),
                self._prompt("p1", no_tool, meta=True),
            ], "p1", "auto", True, None),
            ("auto offered tool", [
                self._prompt("p1", "t"),
                self._prompt("p1", offer_note, meta=True),
            ], "p1", "auto", False, None),
            ("accepted handback", handed_back, "p1", "default", False, None),
        ]
        for label, records, pid, mode, held, expected_err in cases:
            with self.subTest(case=label):
                self.fires.unlink(missing_ok=True)
                transcript = self._write_transcript(records)
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    res, _ = self._dispatch(
                        self._native_stop(transcript,
                                          "draft",
                                          prompt_id=pid,
                                          permission_mode=mode))
                self.assertEqual(res.get("decision"),
                                 "block" if held else None)
                self.assertEqual(self._fire_count(), 1 if held else 0)
                if expected_err:
                    self.assertIn(expected_err, err.getvalue())
                elif not held:
                    self.assertNotIn("SubagentStop skipped", err.getvalue())


if __name__ == "__main__":
    unittest.main()
