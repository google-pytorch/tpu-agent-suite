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
"""Unit tests for the session lifecycle: the start-run compute check and lease,
and the SessionEnd reap hook.
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
from framework.hooks import dispatch  # noqa: E402
from framework.tests.hooks import helpers  # noqa: E402

Path = pathlib.Path


def setUpModule():
    helpers.set_up_module()


def tearDownModule():
    helpers.tear_down_module()


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

    @mock.patch("framework.compute.reap_jobs")
    def test_unknown_session_is_a_no_op(self, mock_reap_jobs):
        _, code = self._dispatch("claude-code", "SessionEnd", {
            "session_id": "someone-else",
        })
        self.assertEqual(code, 0)
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
