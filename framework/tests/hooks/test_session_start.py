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
"""Unit tests for the session lifecycle: SessionStart dispatch, hook
context, the start-run compute check and lease, and the SessionEnd reap hook.
"""

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

from framework import run_state, runtime  # noqa: E402
from framework.hooks import dispatch, events  # noqa: E402
from framework.tests.hooks import helpers  # noqa: E402

Path = pathlib.Path


def setUpModule():
    helpers.set_up_module()


def tearDownModule():
    helpers.tear_down_module()


class TestPendingContext(helpers.RuntimeTestCase):
    """Pending and post-tool context handed back to the model."""

    def test_pending_context_is_capped_and_evicts_oldest(self):
        """Every live entry rides along on every model call, so the spool is bounded.

        A run writing context in a loop would otherwise grow it without
        limit. The newest survive: they are the context a result is most
        likely still to need. Entries are consumed rather than re-read, so
        another session must neither see nor drain them.
        """
        overflow = run_state.MAX_PENDING_CONTEXT_ENTRIES + 3
        for i in range(overflow):
            run_state.record_pending_hook_context(
                session_id="conv-cap",
                hook_name="test-hook",
                context_text=f"CONTEXT-{i:02d}",
                package="test-pkg@1.0.0",
            )
        self.assertEqual(
            run_state.replay_pending_hook_context("conv-stranger"), "")
        replayed = run_state.replay_pending_hook_context("conv-cap")
        kept = [
            line for line in replayed.splitlines()
            if line.startswith("CONTEXT-")
        ]
        self.assertEqual(len(kept), run_state.MAX_PENDING_CONTEXT_ENTRIES)
        self.assertNotIn("CONTEXT-00", kept)
        self.assertIn(f"CONTEXT-{overflow - 1:02d}", kept)

    def test_posttooluse_context_reaches_a_google_harness_model(self):
        """The whole emulation, end to end.

        Antigravity documents the PostToolUse response as a bare {}, so the
        hook must still answer with exactly that -- and the context it
        produced must nonetheless reach the model, as injectSteps on the
        following invocations. `invocationNum > 0` is the model call right
        after a tool call, which is where hook context is wanted.
        """
        session = "conv-e2e-posttooluse"
        post_payload = json.dumps({
            "conversationId": session,
            "workspacePaths": [str(self.workspace_dir)],
            "toolCall": {
                "name": "write_to_file",
                "args": {
                    "TargetFile": "notes.md"
                }
            },
            "stepIdx": 3,
        })
        post_out, _ = self._dispatch(
            harness="antigravity",
            package="",
            hook_name="probe-hook",
            event="PostToolUse",
            raw_input_text=post_payload,
            command=
            "printf '%s' '{\"decision\":\"allow\",\"context\":\"SPOOLED-CONTEXT\"}'",
        )
        self.assertEqual(post_out, {},
                         "PostToolUse must answer with the documented {}")

        def invoke(invocation_num: int) -> dict:
            out, _ = self._dispatch(
                harness="antigravity",
                package="",
                hook_name="session-run-init",
                event="SessionStart",
                raw_input_text=json.dumps({
                    "conversationId":
                    session,
                    "workspacePaths": [str(self.workspace_dir)],
                    "invocationNum":
                    invocation_num,
                    "initialNumSteps":
                    8,
                }),
                command="printf '%s' '{\"decision\":\"allow\"}'",
            )
            return out

        # Replay remains available across rounds so the agent can execute
        # multiple intermediate tool calls before citing the context.
        for invocation_num in range(1, 6):
            self.assertIn({"ephemeralMessage": "SPOOLED-CONTEXT"},
                          invoke(invocation_num).get("injectSteps", []),
                          f"context missing from invocation {invocation_num}")


class TestSessionStartDispatch(helpers.RuntimeTestCase):
    """SessionStart dispatch."""

    # The compute note is not replayed; its own test below covers that.
    @mock.patch.object(runtime.compute.cli,
                       "check_target_at_session_start",
                       return_value=None)
    def test_dispatch_hook_session_start_antigravity_context_caching(
            self, _check):
        """Antigravity fires SessionStart on every model call: only the first
        runs each hook, and every later call replays that hook's own context.

        The builtin emits a bare `run_id`; the context wrapped around it is
        synthesised during translation, so the cache has to cover `run_id`
        too, or Turn 2+ would allocate a fresh namespace.
        """
        telemetry_script = self._write_script(
            "telemetry.py",
            "import sys, json\n"
            "json.dump({'decision': 'allow', 'context': 'TPU ready'}, sys.stdout)\n",
        )
        run_init = {"hook_name": "session-run-init", "builtin": "start-run"}
        telemetry = {"hook_name": "tel", "script_rel": "scripts/telemetry.py"}
        telemetry_reply = {"injectSteps": [{"ephemeralMessage": "TPU ready"}]}

        def fire(initial_num_steps, invocation_num=0, **action):
            res, code = self._dispatch(
                harness="antigravity",
                event="SessionStart",
                raw_input_text=json.dumps({
                    "conversationId": "conv-multi",
                    "invocationNum": invocation_num,
                    "initialNumSteps": initial_num_steps,
                }),
                **action,
            )
            self.assertEqual(code, 0)
            return res

        # Turn 1: each hook runs and seeds its own cache.
        first_run = fire(1, **run_init)
        run_id = run_state.resolve_session_run_id("conv-multi")
        self.assertIn(f"TPU agent suite run namespace: {run_id}",
                      first_run["injectSteps"][0]["ephemeralMessage"])
        self.assertEqual(fire(1, **telemetry), telemetry_reply)

        # Later calls in turn 1 and later turns replay, running nothing.
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
        # A hook that never ran has nothing to replay.
        self.assertEqual(fire(3, hook_name="unregistered-hook"),
                         {"injectSteps": []})

    def test_dispatch_hook_session_start_antigravity_subagent_bypassed(self):
        self._write_script(
            "fail_on_call.py",
            "import sys, json\n"
            "# If called for a subagent, exit with error\n"
            "sys.exit(1)\n",
        )

        # Worktree transcript path indicates subagent -> should be bypassed with allow
        res, code = self._dispatch(
            harness="antigravity",
            hook_name="subagent-test",
            event="SessionStart",
            script_rel="scripts/fail_on_call.py",
            raw_input_text=json.dumps({
                "conversationId":
                "subagent-conv-456",
                "invocationNum":
                0,
                "initialNumSteps":
                1,
                "transcriptPath":
                "/Users/test/workspace/worktrees/subagent-123/transcript.jsonl",
            }),
        )
        self.assertEqual(code, 0)
        self.assertEqual(res, {"injectSteps": []})


class TestStartRunComputeCheck(helpers.RuntimeTestCase):
    """The compute check that start-run adds to the context."""

    def _start_session(self, session_id):
        res, code = self._dispatch(
            harness="claude-code",
            hook_name="session-run-init",
            event="SessionStart",
            builtin="start-run",
            timeout=30,
            raw_input_text=json.dumps({
                "session_id": session_id,
                "source": "startup",
                "cwd": str(self.workspace_dir),
            }),
        )
        self.assertEqual(code, 0)
        return res.get("hookSpecificOutput", {}).get("additionalContext", "")

    def test_start_run_adds_the_compute_check_to_the_context(self):
        note = "Compute target checked for this session: t (ok)"
        with mock.patch.object(runtime.compute.cli,
                               "check_target_at_session_start",
                               return_value=note) as check:
            context = self._start_session("sess-check-1")
            run_id = run_state.resolve_session_run_id("sess-check-1")
            # Resumed: checked again, but the session keeps its run.
            resumed = self._start_session("sess-check-1")
        self.assertIn(note, context)
        self.assertIn(f"TPU agent suite run namespace: {run_id}", resumed)
        self.assertEqual(check.call_count, 2)
        self.assertEqual(check.call_args.args[-1], 30)  # the hook's timeout

    def test_start_run_survives_a_compute_check_that_raises(self):
        with mock.patch.object(runtime.compute.cli,
                               "check_target_at_session_start",
                               side_effect=RuntimeError("boom")), \
                mock.patch("sys.stderr", io.StringIO()):
            context = self._start_session("sess-check-2")
        self.assertIn("TPU agent suite run namespace:", context)


class TestMultiSessionConcurrency(helpers.RuntimeTestCase):
    """Concurrent sessions and run-id propagation."""

    def test_antigravity_child_conversation_resolves_parent_run_id(self):
        parent_cid = "conv-parent-999"
        child_cid = "subagent-child-888"
        parent_run_id = "20260913T120000Z-999999999999"

        # Record run_id for parent session
        run_state.record_session_run(
            session_id=parent_cid,
            run_id=parent_run_id,
            package="test-pkg@1.0.0",
        )

        # Set up real filesystem descriptor in fake gemini brain home
        fake_home = self.temp_dir / "fake_home"
        subagent_file = (fake_home / ".gemini" / "antigravity-cli" / "brain" /
                         parent_cid / ".system_generated" / "subagents" /
                         f"{child_cid}.json")
        subagent_file.parent.mkdir(parents=True, exist_ok=True)
        subagent_file.write_text("{}", encoding="utf-8")

        with mock.patch("pathlib.Path.home", return_value=fake_home):
            # Verify child resolves to parent run_id via disk discovery
            self.assertEqual(
                run_state.resolve_session_run_id(child_cid),
                parent_run_id,
            )

            # Inbound normalization for child subagent PreToolUse
            payload = {
                "conversationId": child_cid,
                "toolCall": {
                    "name": "view_file",
                    "args": {
                        "AbsolutePath": "/test/file.py"
                    },
                },
            }
            ev = events.normalize_inbound_event(
                raw_payload=payload,
                harness="antigravity",
                event="PreToolUse",
                package="test-pkg@1.0.0",
                workspace=self.workspace_dir,
            )
            self.assertEqual(ev["context"]["session_id"], child_cid)
            self.assertEqual(ev["context"]["run_id"], parent_run_id)


class TestSessionStartLease(unittest.TestCase):
    """`start-run` leases the run, then sweeps the runs of ended sessions."""

    PKG = "test-pkg@1.0.0"
    OWNER = {"pid": 4242, "started": "99"}

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.workspace = self.tmp / "workspace"
        self.workspace.mkdir()
        patcher = mock.patch.dict(os.environ,
                                  {"TPU_AGENT_SUITE_HOME": str(self.tmp)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _start(self, owner):
        """Dispatch start-run; returns the lease the sweep saw."""
        seen = []
        with mock.patch("framework.compute.lease.session_owner",
                        return_value=owner), \
                mock.patch("framework.compute.lease.spawn_sweep",
                           side_effect=lambda run: seen.append(
                               runtime.compute.lease.read_owner(run))):
            _, code = dispatch.dispatch_hook(
                harness="claude-code",
                package=self.PKG,
                hook_name="hook",
                event="SessionStart",
                builtin="start-run",
                workspace=self.workspace,
                raw_input_text=json.dumps({"session_id": "sess-1"}),
            )
        self.assertEqual(code, 0)
        return seen[0] if seen else None

    def test_failures_never_fail_the_session_start(self):
        with mock.patch("framework.compute.lease.claim",
                        side_effect=OSError("disk full")), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertIsNone(self._start(self.OWNER))
        self.assertIn("could not lease run", err.getvalue())


class TestSessionEndReapHook(unittest.TestCase):
    """The `reap-jobs` builtin: fires only when the session is really over.

    Registered on SessionEnd, it is what cancels a run's leftover workloads.
    On Claude Code SessionEnd is native and always means the end. Antigravity
    emulates it on its per-turn Stop, so the same hook fires at
    every turn boundary and has to leave a just-submitted job alone.
    """

    RUN_ID = "20260101T000000Z-777777777777"
    PKG = "test-pkg@1.0.0"

    def setUp(self):
        # Session records are central: keep this class's out of the real store.
        self.tmp, self.workspace = helpers.isolated_workspace(self, "home")

    def _dispatch(self, harness, event, payload, session="sess-1"):
        run_state.record_session_run(session, self.RUN_ID, package=self.PKG)
        return dispatch.dispatch_hook(
            harness=harness,
            package=self.PKG,
            hook_name="session-end-reap",
            event=event,
            builtin="reap-jobs",
            workspace=self.workspace,
            raw_input_text=json.dumps(payload),
        )

    @mock.patch("framework.compute.reap_jobs", return_value=(["job-1"], {}))
    @mock.patch("framework.compute.resolve_driver")
    def test_claude_code_session_end_reaps_the_sessions_run(
            self, mock_resolve_driver, mock_reap_jobs):
        res, code = self._dispatch("claude-code", "SessionEnd", {
            "session_id": "sess-1",
            "reason": "prompt_input_exit",
        })
        self.assertEqual(code, 0)
        mock_reap_jobs.assert_called_once()
        run_id, resolver = mock_reap_jobs.call_args.args
        self.assertEqual(run_id, self.RUN_ID)
        self.assertIn("job-1", res.get("systemMessage", ""))

        # The resolver hands reap a driver for the job's own backend and
        # profile, with the workspace's configuration behind it.
        mock_driver = mock.MagicMock()
        mock_resolve_driver.return_value = mock_driver
        self.assertIs(resolver("job-1"), mock_driver)
        mock_resolve_driver.assert_called_once_with(
            workspace=self.workspace.resolve(),
            package=self.PKG,
            run_id=self.RUN_ID,
            job_id="job-1",
        )

    @mock.patch("framework.compute.reap_jobs")
    def test_claude_code_turn_stop_never_reaps(self, mock_reap_jobs):
        """A Stop is a turn ending, whatever the payload says."""
        _, code = self._dispatch("claude-code", "Stop", {
            "session_id": "sess-1",
            "stop_hook_active": True,
        })
        self.assertEqual(code, 0)
        mock_reap_jobs.assert_not_called()

    @mock.patch("framework.compute.reap_jobs", return_value=([], {}))
    def test_antigravity_idle_terminal_stop_reaps(self, mock_reap_jobs):
        _, code = self._dispatch(
            "antigravity", "SessionEnd", {
                "conversationId": "sess-1",
                "terminationReason": "TERMINAL_STEP_TYPE",
                "fullyIdle": True,
            })
        self.assertEqual(code, 0)
        self.assertEqual(mock_reap_jobs.call_args.args[0], self.RUN_ID)

    @mock.patch("framework.compute.reap_jobs")
    def test_unknown_session_is_a_no_op(self, mock_reap_jobs):
        _, code = self._dispatch("claude-code", "SessionEnd", {
            "session_id": "someone-else",
        })
        self.assertEqual(code, 0)
        mock_reap_jobs.assert_not_called()

    @mock.patch("framework.compute.reap_jobs")
    def test_antigravity_subagent_stop_bypasses_the_hook(self, mock_reap_jobs):
        """A subagent finishing arrives on the same Stop; it is not the end."""
        res, code = self._dispatch(
            "antigravity", "SessionEnd", {
                "conversationId": "sess-1",
                "role": "tpu-diagnostics",
                "terminationReason": "NO_TOOL_CALL",
                "fullyIdle": True,
            })
        self.assertEqual(code, 0)
        self.assertEqual(res, {"decision": "stop"})
        mock_reap_jobs.assert_not_called()

    @mock.patch("framework.compute.reap_jobs",
                side_effect=RuntimeError("kubectl: unauthorized"))
    def test_reap_failure_is_reported_not_raised(self, mock_reap_jobs):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            _, code = self._dispatch("claude-code", "SessionEnd", {
                "session_id": "sess-1",
            })
        self.assertEqual(code, 0)
        self.assertIn("kubectl: unauthorized", stderr.getvalue())
        self.assertIn(self.RUN_ID, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
