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
"""Shared behaviour of the Google harness family (`GoogleHarness`)."""

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

from framework import compute, harnesses, paths, run_state  # noqa: E402
from framework.harnesses import google  # noqa: E402
from framework.hooks import dispatch, events  # noqa: E402


class GoogleHarnessTest(unittest.TestCase):
    PKG = "test-pkg@1.0.0"
    RUN_ID = "20260101T000000Z-777777777777"

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.workspace = self.tmp / "workspace"
        self.workspace.mkdir(parents=True)
        self.fake_home = self.tmp / "home"
        self.store_home = self.tmp / "store"
        self.pkg_root = self.store_home / "packages" / self.PKG
        self.pkg_root.mkdir(parents=True)
        env_patch = mock.patch.dict(
            os.environ, {"TPU_AGENT_SUITE_HOME": str(self.store_home)})
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def test_brain_descriptor_and_parent_run_resolution(self):
        ag = harnesses.require("antigravity")
        parent_cid = "conv-parent-999"
        child_cid = "subagent-child-888"
        role_cid = "test-descriptor-role-8888"
        parent_run_id = "20260913T120000Z-999999999999"

        run_state.record_session_run(
            session_id=parent_cid,
            run_id=parent_run_id,
            package=self.PKG,
        )
        sub_dir = (self.fake_home / ".gemini" / "antigravity-cli" / "brain" /
                   parent_cid / ".system_generated" / "subagents")
        sub_dir.mkdir(parents=True)
        (sub_dir / f"{child_cid}.json").write_text(
            json.dumps({"subagentDescriptor": {
                "typeName": "torch-coder"
            }}),
            encoding="utf-8",
        )
        (sub_dir / f"{role_cid}.json").write_text(
            json.dumps({"subagentDescriptor": {
                "role": "kernel-reviewer"
            }}),
            encoding="utf-8",
        )

        with mock.patch("pathlib.Path.home", return_value=self.fake_home):
            self.assertEqual(ag.resolve_subagent(child_cid), "torch-coder")
            self.assertEqual(ag.resolve_subagent(role_cid), "kernel-reviewer")
            self.assertIsNone(ag.resolve_subagent("root-cid-5678"))
            for unsafe_cid in ("", "*", "subagent-*", "../other",
                               "nested/cid"):
                self.assertIsNone(ag.resolve_subagent(unsafe_cid), unsafe_cid)
            self.assertEqual(
                ag.resolve_subagent(
                    "unknown-cid",
                    "/Users/test/workspace/worktrees/subagent-1/transcript.jsonl",
                ),
                "subagent",
            )

            self.assertEqual(run_state.resolve_session_run_id(child_cid),
                             parent_run_id)
            ev = events.normalize_inbound_event(
                raw_payload={
                    "conversationId": child_cid,
                    "toolCall": {
                        "name": "view_file",
                        "args": {
                            "AbsolutePath": "/test/file.py"
                        },
                    },
                },
                harness="antigravity",
                event="PreToolUse",
                package=self.PKG,
                workspace=self.workspace,
            )
            self.assertEqual(ev["context"]["session_id"], child_cid)
            self.assertEqual(ev["context"]["run_id"], parent_run_id)

    def test_complete_subagent_stop(self):
        ag = harnesses.require("antigravity")
        transcript_file = self.workspace / "transcript_post.jsonl"
        transcript_file.write_text(
            json.dumps({
                "source": "MODEL",
                "content": "TPU kernel done"
            }),
            encoding="utf-8",
        )
        self.assertEqual(
            ag.complete_subagent_stop("my-agent",
                                      None,
                                      transcript_path=str(transcript_file)),
            ("my-agent", "TPU kernel done"),
        )
        self.assertEqual(
            ag.complete_subagent_stop("my-agent",
                                      "already_set",
                                      transcript_path=str(transcript_file)),
            ("my-agent", "already_set"),
        )

    def _dispatch_reap(self, payload):
        run_state.record_session_run("sess-1", self.RUN_ID, package=self.PKG)
        return dispatch.dispatch_hook(
            harness="antigravity",
            package=self.PKG,
            hook_name="session-end-reap",
            event="SessionEnd",
            builtin="reap-jobs",
            workspace=self.workspace,
            raw_input_text=json.dumps(payload),
        )

    @mock.patch("framework.compute.reap_jobs", return_value=([], {}))
    def test_idle_terminal_stop_reaps(self, mock_reap_jobs):
        _, code = self._dispatch_reap({
            "conversationId": "sess-1",
            "terminationReason": "TERMINAL_STEP_TYPE",
            "fullyIdle": True,
        })
        self.assertEqual(code, 0)
        self.assertEqual(mock_reap_jobs.call_args.args[0], self.RUN_ID)

    @mock.patch("framework.compute.reap_jobs")
    def test_subagent_stop_disambiguation_and_reap_bypass(
            self, mock_reap_jobs):
        """A subagent's Stop is classified as SubagentStop and never reaps.

        A subagent finishing arrives on the same Stop event as the root; it is
        still classified as SubagentStop, including from nested toolCall
        metadata.
        """
        nested = {
            "conversationId": "sess-1",
            "toolCall": {
                "typeName": "tpu-diagnostics"
            },
            "terminationReason": "NO_TOOL_CALL",
            "fullyIdle": True,
        }
        event = events.normalize_inbound_event(
            raw_payload=nested,
            harness="antigravity",
            event="Stop",
            package=self.PKG,
            workspace=self.workspace,
        )
        self.assertEqual((event["event"], event["subagent"]),
                         ("SubagentStop", "tpu-diagnostics"))

        for payload in (
                nested,
            {
                "conversationId": "untracked-child",
                "transcriptPath": "/ws/worktrees/subagent-2/transcript.jsonl",
                "terminationReason": "TERMINAL_STEP_TYPE",
                "fullyIdle": True,
            },
        ):
            with self.subTest(payload=payload):
                res, code = self._dispatch_reap(payload)
                self.assertEqual((code, res), (0, {"decision": "stop"}))
        mock_reap_jobs.assert_not_called()

    @mock.patch.object(compute.cli,
                       "check_target_at_session_start",
                       return_value=None)
    def test_multi_hook_context_caching_and_subagent_bypass(self, _check):
        scripts_dir = self.pkg_root / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        telemetry_script = scripts_dir / "telemetry.py"
        telemetry_script.write_text(
            "import sys, json\n"
            "json.dump({'decision': 'allow', 'context': 'TPU ready'}, sys.stdout)\n",
            encoding="utf-8",
        )
        run_init = {"hook_name": "session-run-init", "builtin": "start-run"}
        telemetry = {
            "hook_name": "telemetry-hook",
            "script_rel": "scripts/telemetry.py"
        }
        telemetry_reply = {"injectSteps": [{"ephemeralMessage": "TPU ready"}]}

        def fire(initial_num_steps, invocation_num=0, extra=None, **action):
            with contextlib.redirect_stderr(io.StringIO()):
                res, code = dispatch.dispatch_hook(
                    harness="antigravity",
                    package=self.PKG,
                    event="SessionStart",
                    workspace=self.workspace,
                    raw_input_text=json.dumps({
                        "conversationId": "conv-multi",
                        "invocationNum": invocation_num,
                        "initialNumSteps": initial_num_steps,
                        **(extra or {}),
                    }),
                    **action,
                )
            self.assertEqual(code, 0)
            return res

        first_run = fire(1, **run_init)
        run_id = run_state.resolve_session_run_id("conv-multi")
        self.assertIn(f"TPU agent suite run namespace: {run_id}",
                      first_run["injectSteps"][0]["ephemeralMessage"])
        self.assertEqual(fire(1, **telemetry), telemetry_reply)

        telemetry_script.write_text("import sys\nsys.exit(1)\n",
                                    encoding="utf-8")
        for init_steps, invocation in ((1, 2), (5, 3)):
            with self.subTest(initial_num_steps=init_steps,
                              invocation_num=invocation):
                self.assertEqual(fire(init_steps, invocation, **run_init),
                                 first_run)
                self.assertEqual(fire(init_steps, invocation, **telemetry),
                                 telemetry_reply)
        self.assertEqual(run_state.resolve_session_run_id("conv-multi"),
                         run_id)
        self.assertEqual(fire(3, hook_name="unregistered-hook"),
                         {"injectSteps": []})
        # A subagent's PreInvocation is bypassed without running the script.
        self.assertEqual(
            fire(
                1,
                extra={
                    "conversationId":
                    "subagent-conv-456",
                    "transcriptPath":
                    "/Users/test/workspace/worktrees/subagent-123/transcript.jsonl",
                },
                **telemetry,
            ),
            {"injectSteps": []},
        )
        context_files = sorted(p.name
                               for p in (paths.get_sessions_root() /
                                         "conv-multi.contexts").iterdir())
        self.assertEqual(context_files,
                         ["session-run-init.json", "telemetry-hook.json"])

    def test_posttooluse_context_reaches_a_google_harness_model(self):
        session = "conv-e2e-posttooluse"
        post_out, _ = dispatch.dispatch_hook(
            harness="antigravity",
            package="",
            hook_name="probe-hook",
            event="PostToolUse",
            workspace=self.workspace,
            raw_input_text=json.dumps({
                "conversationId": session,
                "toolCall": {
                    "name": "write_to_file",
                    "args": {
                        "TargetFile": "notes.md"
                    },
                },
            }),
            command=
            "printf '%s' '{\"decision\":\"allow\",\"context\":\"SPOOLED-CONTEXT\"}'",
        )
        self.assertEqual(post_out, {})

        for invocation_num in (1, 2):
            out, _ = dispatch.dispatch_hook(
                harness="antigravity",
                package="",
                hook_name="session-run-init",
                event="SessionStart",
                workspace=self.workspace,
                raw_input_text=json.dumps({
                    "conversationId": session,
                    "invocationNum": invocation_num,
                    "initialNumSteps": 8,
                }),
                command="printf '%s' '{\"decision\":\"allow\"}'",
            )
            self.assertIn({"ephemeralMessage": "SPOOLED-CONTEXT"},
                          out.get("injectSteps", []))

    def test_pending_context_is_capped_and_evicts_oldest(self):
        """Every live entry rides along on every model call, so the spool is bounded."""
        overflow = google.MAX_PENDING_CONTEXT_ENTRIES + 3
        for i in range(overflow):
            google.record_pending_hook_context(
                session_id="conv-cap",
                hook_name="test-hook",
                context_text=f"CONTEXT-{i:02d}",
                package=self.PKG,
            )
        self.assertEqual(google.replay_pending_hook_context("conv-stranger"),
                         "")
        replayed = google.replay_pending_hook_context("conv-cap")
        kept = [
            line for line in replayed.splitlines()
            if line.startswith("CONTEXT-")
        ]
        self.assertEqual(len(kept), google.MAX_PENDING_CONTEXT_ENTRIES)
        self.assertNotIn("CONTEXT-00", kept)
        self.assertIn(f"CONTEXT-{overflow - 1:02d}", kept)


if __name__ == "__main__":
    unittest.main()
