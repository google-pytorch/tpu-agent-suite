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
"""Unit tests for hook dispatch and inbound event normalization.

Covers the `hook-dispatch` CLI, hook timeouts, the failure policy and
fan-out, and how harness events (subagent and session-start payloads, stop
disambiguation) are normalized and routed.
"""

import contextlib
import functools
import io
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from typing import Any
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import jsonio, paths, run_state, runtime, utils  # noqa: E402
from framework.builder import hook_config  # noqa: E402
from framework.hooks import actions, dispatch, events  # noqa: E402
from framework.tests.hooks import helpers  # noqa: E402

Path = pathlib.Path


def setUpModule():
    helpers.set_up_module()


def tearDownModule():
    helpers.tear_down_module()


class TestHookDispatchUsageErrors(helpers.RuntimeTestCase):
    """hook-dispatch usage errors and how they are settled."""

    def _hook_failure_records(self) -> list[dict[str, Any]]:
        log = self.central_home / utils.HOOK_FAILURE_LOG
        if not log.is_file():
            return []
        return [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
        ]

    def test_hook_dispatch_usage_error_is_settled_by_on_failure(self):
        """A hook-dispatch whose arguments do not parse answers by policy,
        in the harness's shape, exit 0, and is recorded as `startup` --
        instead of argparse's exit 2, which Claude Code reads as a block."""
        base = [
            "hook-dispatch", "--harness", "claude-code", "--package",
            "test-pkg@1.0.0", "--hook", "lenient", "--event", "PreToolUse",
            "--command", "true"
        ]
        code, out, err = self._main_capturing(base +
                                              ["--on-failure", "allow"] +
                                              ["--no-such-flag"])
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(out)["hookSpecificOutput"]["permissionDecision"],
            "allow")
        self.assertIn("hook failed (startup)", err)
        self.assertIn("--no-such-flag", err)
        [record] = self._hook_failure_records()
        self.assertEqual((record["kind"], record["hook"], record["decision"]),
                         ("startup", "lenient", "allow"))

        # Default policy: a deny in the reply, still exit 0.
        code, out, err = self._main_capturing(base + ["--no-such-flag"])
        self.assertEqual(code, 0)
        h_out = json.loads(out)["hookSpecificOutput"]
        self.assertEqual(h_out["permissionDecision"], "deny")
        self.assertIn("--no-such-flag", h_out["permissionDecisionReason"])

        # A hand-back registration names its tool event on the command line;
        # the reply takes that event's shape, not SubagentStop's, or Claude
        # Code would reject it and run the tool.
        code, out, _ = self._main_capturing([
            "hook-dispatch", "--harness", "claude-code", "--package", "p",
            "--hook", "verify", "--event", "SubagentStop", "--handback-event",
            "PreToolUse", "--command", "true", "--no-such-flag"
        ])
        self.assertEqual(code, 0)
        h_out = json.loads(out)["hookSpecificOutput"]
        self.assertEqual(h_out["hookEventName"], "PreToolUse")
        self.assertEqual(h_out["permissionDecision"], "deny")

    def test_hook_dispatch_usage_error_without_a_shape_exits_by_policy(self):
        """An unknown event (or harness) leaves no shape to answer in, so the
        exit code carries the policy: 0 for allow, 2 for deny. So does a
        shape lookup that fails outright, when the store has lost the schema
        files it reads: the startup path must not crash into a traceback."""
        for extra, expected in ((["--on-failure", "allow"], 0), ([], 2)):
            with self.subTest(policy=extra or "default"):
                code, out, err = self._main_capturing([
                    "hook-dispatch", "--harness", "claude-code", "--package",
                    "p", "--hook", "h", "--event", "NotAnEvent", "--command",
                    "true"
                ] + extra)
                self.assertEqual(code, expected)
                self.assertEqual(out, "")
                self.assertIn("hook failed (startup)", err)
                self.assertIn("NotAnEvent", err)

        # The store has lost event-mappings.json: the strict parser cannot
        # even be built (its --harness choices come from it), and neither
        # can a reply shape. Not a usage error, yet still a startup failure.
        with mock.patch.object(utils,
                               "get_event_mappings",
                               side_effect=utils.TpuAgentSuiteError(
                                   "missing event mappings")):
            code, out, err = self._main_capturing([
                "hook-dispatch", "--harness", "claude-code", "--package", "p",
                "--hook", "h", "--event", "PreToolUse", "--command", "true",
                "--on-failure", "allow"
            ])
        self.assertEqual(code, 0)
        self.assertEqual(out, "")
        self.assertIn("hook failed (startup)", err)
        self.assertIn("missing event mappings", err)
        self.assertNotIn("Traceback", err)

        # The shape is known but the reply cannot be made (here the failure
        # log refuses the store): the exit code still carries the policy,
        # and the reason says both what failed.
        with mock.patch.object(
                dispatch,
                "_dispatch_failure",
                side_effect=utils.TpuAgentSuiteError("store refused")):
            code, out, err = self._main_capturing([
                "hook-dispatch", "--harness", "claude-code", "--package", "p",
                "--hook", "h", "--event", "PreToolUse", "--command", "true",
                "--no-such-flag"
            ])
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("--no-such-flag", err)
        self.assertIn("no reply could be shaped", err)
        self.assertIn("store refused", err)
        self.assertNotIn("Traceback", err)


class TestHookDispatch(helpers.RuntimeTestCase):
    """Hook output validation and command-line args."""

    def test_hook_output_must_match_the_hook_output_schema(self):
        """A reply without `decision` is a malformed verdict, and a guard
        fails closed on it; silence (no output at all) is an allow."""
        self._write_script(
            "invalid.py", "import json\n"
            "print(json.dumps({'status': 'allow'}))\n")
        res, code = self._dispatch(
            harness="antigravity",
            hook_name="invalid-guard",
            event="PreToolUse",
            script_rel="scripts/invalid.py",
            raw_input_text="{}",
        )
        self.assertEqual(code, 0)
        self.assertEqual(res["decision"], "deny")
        self.assertIn("violated hook-output.schema.json", res["reason"])

        res, code = self._dispatch(
            harness="antigravity",
            hook_name="cmd-hook",
            event="PreToolUse",
            command="true",
            raw_input_text="{}",
        )
        self.assertEqual((code, res), (0, {"decision": "allow"}))

    def test_hook_dispatch_cli_remainder_args(self):
        """Arguments after `--` reach the script verbatim, spaces included."""
        self._write_script(
            "cli_args_guard.py",
            "import json, sys\n"
            "if sys.argv[1:] == ['--mode', 'param with spaces']:\n"
            "    print(json.dumps({'decision': 'allow'}))\n"
            "else:\n"
            "    print(json.dumps({'decision': 'deny', 'reason': f'argv: {sys.argv}'}))\n",
        )

        captured = io.StringIO()
        with mock.patch("sys.stdout",
                        captured), mock.patch("sys.stdin", io.StringIO("{}")):
            ret = runtime.main([
                "hook-dispatch",
                "--harness",
                "antigravity",
                "--package",
                "test-pkg@1.0.0",
                "--hook",
                "cli-guard",
                "--event",
                "PreToolUse",
                "--script",
                "scripts/cli_args_guard.py",
                "--workspace",
                str(self.workspace_dir),
                "--",
                "--mode",
                "param with spaces",
            ])
        self.assertEqual(ret, 0)
        self.assertEqual(json.loads(captured.getvalue()),
                         {"decision": "allow"})


class TestHookTimeoutVerdictBeatsHarness(unittest.TestCase):
    """A hung hook's deny must be written before the harness gives up on it.

    The harness is registered with the hook's timeout plus
    `hook_config.HOOK_DISPATCH_HEADROOM_S`, and hook-dispatch gets the bare
    timeout. Dispatch enforces that timeout on the action but finishes a
    little after it -- interpreter start before, the verdict after. Should
    that overrun ever reach the headroom, the harness kills dispatch first,
    the deny is lost, and Claude Code runs the tool the hook was guarding.

    Run as a real subprocess, as the harness runs it, so interpreter start and
    module import are counted in the time.
    """

    TIMEOUT_S = 1

    def setUp(self):
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.workspace_dir = self.temp_dir / "workspace"
        self.workspace_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def test_hung_hook_is_denied_within_the_headroom(self):
        env = {
            **os.environ, "TPU_AGENT_SUITE_HOME": str(self.temp_dir / "home")
        }
        started = time.monotonic()
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(runtime.__file__)),
                "hook-dispatch",
                "--harness",
                "claude-code",
                "--package",
                "test-pkg@1.0.0",
                "--hook",
                "hung-hook",
                "--event",
                "PreToolUse",
                "--timeout",
                str(self.TIMEOUT_S),
                "--workspace",
                str(self.workspace_dir),
                "--command",
                "sleep 30",
            ],
            input=json.dumps({
                "session_id": "s1",
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_input": {
                    "command": "echo hi"
                },
            }),
            capture_output=True,
            text=True,
            env=env,
            # Far past the headroom: a hang here is a failure, not a stall.
            timeout=60,
        )
        elapsed = time.monotonic() - started

        self.assertEqual(proc.returncode, 0, proc.stderr)
        specific = json.loads(proc.stdout)["hookSpecificOutput"]
        self.assertEqual(specific["permissionDecision"], "deny")
        self.assertIn(f"timed out after {self.TIMEOUT_S} seconds",
                      specific["permissionDecisionReason"])
        # The hook's own timeout is what ended it, not something sooner...
        self.assertGreaterEqual(elapsed, self.TIMEOUT_S)
        # ...and the verdict landed while the harness was still listening.
        self.assertLess(elapsed,
                        self.TIMEOUT_S + hook_config.HOOK_DISPATCH_HEADROOM_S)

    # A builtin that never returns and keeps writing to stderr, as a stuck
    # builtin that reports progress would; the child then dispatches to it.
    CHATTERING_BUILTIN = (
        "import sys\n"
        "from framework import runtime\n"
        "from framework.hooks import actions\n"
        "def chatter(**_kwargs):\n"
        "    while True:\n"
        "        sys.stderr.write('x' * 4096)\n"
        "actions.BUILTIN_HOOK_ACTIONS['record-event'] = chatter\n"
        "raise SystemExit(runtime.main(sys.argv[1:]))\n")

    def test_an_abandoned_builtin_does_not_crash_dispatch_at_exit(self):
        """Shutdown aborts on a stream lock the thread holds (SIGABRT).

        The verdict may already be out, but to Claude Code a non-zero exit is
        a non-blocking error: the output is ignored and the tool runs.
        """
        proc = subprocess.run(
            [
                sys.executable, "-c", self.CHATTERING_BUILTIN, "hook-dispatch",
                "--harness", "claude-code", "--package", "test-pkg@1.0.0",
                "--hook", "stuck-builtin", "--event", "PreToolUse",
                "--timeout",
                str(self.TIMEOUT_S), "--workspace",
                str(self.workspace_dir), "--builtin", "record-event"
            ],
            input=json.dumps({
                "session_id": "s1",
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_input": {},
            }),
            capture_output=True,
            text=True,
            cwd=Path(runtime.__file__).parents[1],
            env={
                **os.environ, "TPU_AGENT_SUITE_HOME":
                str(self.temp_dir / "home")
            },
            timeout=60,
        )

        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        specific = json.loads(proc.stdout)["hookSpecificOutput"]
        self.assertEqual(specific["permissionDecision"], "deny")
        self.assertIn("timed out", specific["permissionDecisionReason"])


class _FailureLogTestCase(unittest.TestCase):
    """A workspace and an isolated store whose hook failure log tests read."""

    def setUp(self):
        super().setUp()
        # The failure log is central: point the store at the temp dir so the
        # failures recorded here never reach the real one.
        self.temp_dir, self.workspace = helpers.isolated_workspace(
            self, "home")
        self.failure_log = paths.get_tpu_agent_suite_home(
        ) / utils.HOOK_FAILURE_LOG
        self.release = threading.Event()
        self.addCleanup(self.release.set)

    def _records(self) -> list[dict]:
        return jsonio.read_jsonl(self.failure_log)

    def _hang_until_released(self, **_kwargs) -> dict:
        """A builtin that does not answer until the test lets it go."""
        self.release.wait(30)
        return {"decision": "allow"}


class TestHookFailurePolicy(_FailureLogTestCase):
    """What a hook that fails to reach a verdict turns into.

    A guard must fail closed; a hook that only observes, or that sits where a
    deny keeps the agent from returning, must not hold it up. Either way the
    failure is recorded, since a hook that fails open is otherwise silent.
    """

    PAYLOAD = {
        "event": "PreToolUse",
        "tool": "shell",
        "args": {},
        "context": {
            "session_id": "s1",
            "run_id": "r1",
            "package": "test-pkg@1.0.0",
        },
    }

    def _command(self, command: str, on_failure: str) -> dict:
        with contextlib.redirect_stderr(io.StringIO()):
            return actions.execute_single_hook_action(
                event_payload=self.PAYLOAD,
                hook_name="h",
                workspace=self.workspace,
                package="test-pkg@1.0.0",
                command=command,
                timeout=5,
                on_failure=on_failure,
            )

    def test_non_json_output_is_quoted_in_the_reason(self):
        """What the hook printed is the clue to why it failed."""
        denied = self._command("echo not json", "deny")
        self.assertIn(repr("not json"), denied["reason"])
        # Capped: the reason reaches the log and the model's context.
        noisy = self._command("printf 'x%.0s' $(seq 600)", "deny")
        self.assertIn(repr("x" * 500 + "..."), noisy["reason"])
        self.assertNotIn("x" * 501, noisy["reason"])
        self.assertEqual([r["kind"] for r in self._records()],
                         ["non_json", "non_json"])

    def test_output_that_cannot_be_decoded_follows_the_policy(self):
        # `text=True` decodes strictly: this used to escape as a
        # UnicodeDecodeError and leave dispatch without a verdict.
        command = r"printf '\377\376'"
        self.assertEqual(self._command(command, "allow"),
                         {"decision": "allow"})
        denied = self._command(command, "deny")
        self.assertEqual(denied["decision"], "deny")
        self.assertIn("UnicodeDecodeError", denied["reason"])
        # Only a failure to answer is covered; a refusal is an answer, is
        # honoured whatever the policy, and is not recorded as a failure.
        res = self._command(
            "echo '{\"decision\": \"deny\", \"reason\": \"no\"}'", "allow")
        self.assertEqual(res, {"decision": "deny", "reason": "no"})
        self.assertEqual([r["kind"] for r in self._records()],
                         ["exception", "exception"])

    def test_a_long_reason_keeps_only_its_end(self):
        """The whole stderr of a hook can be long, and can hold secrets."""
        denied = self._command(
            "printf 'secret%.0s' $(seq 1000) >&2; echo ' ValueError: bad' >&2;"
            " exit 3", "deny")
        [record] = self._records()
        # The denial the model reads and the record are the same text.
        self.assertEqual(denied["reason"], record["reason"])
        self.assertEqual(len(record["reason"]),
                         len("...") + actions._HOOK_FAILURE_REASON_CHARS)
        self.assertTrue(record["reason"].startswith("..."))
        self.assertTrue(record["reason"].endswith("ValueError: bad"))
        # The record says which hook failed, where, and under which policy.
        self.assertEqual(
            {
                key: record[key]
                for key in ("hook", "event", "tool", "kind", "session_id",
                            "run_id", "package", "on_failure", "decision")
            }, {
                "hook": "h",
                "event": "PreToolUse",
                "tool": "shell",
                "kind": "exit",
                "session_id": "s1",
                "run_id": "r1",
                "package": "test-pkg@1.0.0",
                "on_failure": "deny",
                "decision": "deny",
            })
        self.assertTrue(record["timestamp"])

    def _pid_running(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    def _kill_if_running(self, pid_file: Path) -> None:
        if pid_file.exists():
            with contextlib.suppress(ProcessLookupError, ValueError):
                os.kill(int(pid_file.read_text()), signal.SIGKILL)

    def test_a_timed_out_hook_takes_its_children_with_it(self):
        """Killing only `sh` left what it had started running, orphaned."""
        pid_file = self.workspace / "child.pid"
        self.addCleanup(self._kill_if_running, pid_file)
        with contextlib.redirect_stderr(io.StringIO()):
            res = actions.execute_single_hook_action(
                event_payload=self.PAYLOAD,
                hook_name="h",
                workspace=self.workspace,
                package="test-pkg@1.0.0",
                command="cat >/dev/null; sleep 15 & echo $! > child.pid; wait",
                timeout=1,
                on_failure="deny",
            )
        self.assertIn("timed out", res["reason"])
        child = int(pid_file.read_text())
        # Its parent is gone too, so init reaps it, not us: give it a moment.
        deadline = time.monotonic() + 2
        while self._pid_running(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(self._pid_running(child))

    def test_an_unwritable_log_does_not_cost_the_verdict(self):
        # A file where the log's directory belongs: the log cannot be created.
        self.failure_log.parent.parent.mkdir(parents=True, exist_ok=True)
        self.failure_log.parent.write_text("", encoding="utf-8")
        self.assertEqual(self._command("exit 3", "allow"),
                         {"decision": "allow"})
        self.assertEqual(self._command("exit 3", "deny")["decision"], "deny")


class TestDispatchFailure(_FailureLogTestCase):
    """An error in dispatch itself is settled by the hook's policy.

    Left to escape, it would exit non-zero with nothing on stdout, and the
    harness would decide: Claude Code runs the tool, Antigravity blocks it.
    """

    CLAUDE_INPUT = json.dumps({
        "session_id": "s1",
        "tool_name": "Bash",
        "tool_input": {},
    })

    def _dispatch(self, on_failure: str, **kwargs) -> tuple[dict, int]:
        arguments = {
            "harness": "claude-code",
            "package": "test-pkg@1.0.0",
            "hook_name": "h",
            "event": "PreToolUse",
            "command": "true",
            "workspace": self.workspace,
            "raw_input_text": self.CLAUDE_INPUT,
            "on_failure": on_failure,
        }
        arguments.update(kwargs)
        with contextlib.redirect_stderr(io.StringIO()):
            return dispatch.dispatch_hook(**arguments)

    def test_a_handback_is_answered_as_its_tool_event(self):
        # Claude Code rejects a SubagentStop-shaped reply to the hand-back's
        # PreToolUse as a hook error and runs the tool, losing the deny. The
        # payload is unreadable here: the command line alone decides.
        with mock.patch.object(events,
                               "normalize_inbound_event",
                               side_effect=RuntimeError("boom")):
            denied, code = self._dispatch("deny",
                                          event="SubagentStop",
                                          raw_input_text="not json",
                                          handback_event="PreToolUse")
        self.assertEqual(code, 0)
        self.assertNotIn("decision", denied)
        verdict = denied["hookSpecificOutput"]
        self.assertEqual(verdict["hookEventName"], "PreToolUse")
        self.assertEqual(verdict["permissionDecision"], "deny")
        self.assertIn("RuntimeError: boom",
                      verdict["permissionDecisionReason"])


class TestHookFanOut(_FailureLogTestCase):
    """A batched delegation must reach its verdict inside the harness budget.

    The hook runs once per subagent, and the harness allows the whole dispatch
    only the hook's own timeout plus headroom; hung subagents must not add up.
    """

    TIMEOUT = 1
    # Enough subagents that, checked in turn, they would outlast the budget.
    SUBAGENTS = tuple(f"agent-{i}"
                      for i in range(TIMEOUT +
                                     hook_config.HOOK_DISPATCH_HEADROOM_S + 1))
    # Hangs on the first subagent only, and answers at once for the others.
    # Every sub-event carries the whole batch too; match the one it is for.
    SLOW_ON_FIRST = ('payload=$(cat); case "$payload" in '
                     '*\'"subagent": "agent-0"\'*) sleep 15;; esac; '
                     'echo \'{"decision": "allow"}\'')

    def _dispatch(self, on_failure: str, **action) -> tuple[dict, float]:
        """Dispatches one delegation of every subagent; returns the timing."""
        raw = {
            "toolCall": {
                "name": "invoke_subagent",
                "args": {
                    "Subagents": [{
                        "TypeName": name,
                        "Prompt": "go"
                    } for name in self.SUBAGENTS]
                },
            }
        }
        started = time.monotonic()
        with contextlib.redirect_stderr(io.StringIO()):
            response, code = dispatch.dispatch_hook(
                harness="antigravity",
                package="test-pkg@1.0.0",
                hook_name="h",
                event="PreToolUse",
                workspace=self.workspace,
                raw_input_text=json.dumps(raw),
                timeout=self.TIMEOUT,
                on_failure=on_failure,
                **action,
            )
        self.assertEqual(code, 0)
        return response, time.monotonic() - started

    def _assert_within_budget(self, elapsed: float) -> None:
        self.assertLess(elapsed,
                        self.TIMEOUT + hook_config.HOOK_DISPATCH_HEADROOM_S)

    def _hung_actions(self) -> dict[str, dict]:
        return {
            "command": {
                "command": "sleep 15"
            },
            "builtin": {
                "builtin": "validate-schema"
            },
        }

    def test_hung_subagents_are_denied_within_the_budget(self):
        with mock.patch.dict(actions.BUILTIN_HOOK_ACTIONS,
                             {"validate-schema": self._hang_until_released}):
            for kind, action in self._hung_actions().items():
                with self.subTest(kind=kind):
                    response, elapsed = self._dispatch("deny", **action)
                    self._assert_within_budget(elapsed)
                    self.assertEqual(response["decision"], "deny")
                    for name in self.SUBAGENTS:
                        self.assertIn(name, response["reason"])

    def test_a_hung_subagent_costs_only_itself(self):
        response, elapsed = self._dispatch("deny", command=self.SLOW_ON_FIRST)
        self._assert_within_budget(elapsed)
        self.assertEqual(response["decision"], "allow")
        self.assertEqual(response["overwrite"]["Subagents"], [{
            "TypeName": name,
            "Prompt": "go"
        } for name in self.SUBAGENTS[1:]])


class TestNormalizeInbound(helpers.RuntimeTestCase):
    """Normalizing inbound harness payloads into hook events."""

    def _assert_normalized(self, cases):
        """Each case is `(harness, native event, payload, {path: value})`;
        a path is dotted into the canonical event."""
        for harness, native_event, payload, expected in cases:
            event = events.normalize_inbound_event(
                raw_payload=payload,
                harness=harness,
                event=native_event,
                package="test-pkg",
                workspace=self.workspace_dir,
            )
            for path, value in expected.items():
                with self.subTest(harness=harness, path=path):
                    got = functools.reduce(lambda d, k: d[k], path.split("."),
                                           event)
                    self.assertEqual(got, value)

    def test_normalize_inbound_session_start(self):
        """Each harness's native session-start payload, in canonical form."""
        self._assert_normalized([
            ("claude-code", "SessionStart", {
                "session_id": "sess-claude-123",
                "source": "resume",
                "cwd": "/workspace/project",
            }, {
                "event": "SessionStart",
                "args.session_id": "sess-claude-123",
                "args.source": "resume",
                "args.cwd": "/workspace/project",
            }),
            # `workspace_roots` is a list, but `cwd` names one directory. Cursor
            # names nothing that started the session, so the default stands.
            ("cursor", "sessionStart", {
                "workspace_roots": ["/workspace/cursor-proj"],
            }, {
                "args.cwd": "/workspace/cursor-proj",
                "args.source": "startup",
            }),
            # Regression: an empty list used to displace the workspace
            # fallback, handing hooks `[]` where a path string is promised.
            ("cursor", "sessionStart", {
                "workspace_roots": [],
            }, {
                "args.cwd": str(self.workspace_dir),
            }),
            ("antigravity", "PreInvocation", {
                "conversationId": "conv-agy-101",
                "workspacePaths": ["/workspace/agy-proj"],
                "invocationNum": 1,
                "initialNumSteps": 0,
            }, {
                "event": "SessionStart",
                "args.session_id": "conv-agy-101",
                "args.cwd": "/workspace/agy-proj",
                "args.invocation_num": 1,
                "args.initial_num_steps": 0,
            }),
        ])


class TestGoogleHarnessSubagentNames(helpers.RuntimeTestCase):
    """Transcript and subagent-name helpers for the Google harnesses."""

    def test_get_google_harness_subagent_name_with_disk_descriptor(self):
        cid = "test-descriptor-cid-9999"
        # A fake home, so the descriptor never lands in the developer's own
        # ~/.gemini brain directory.
        fake_home = self.temp_dir / "fake_home"
        parent_dir = fake_home / ".gemini" / "antigravity-cli" / "brain" / "mock-parent-1234" / ".system_generated" / "subagents"
        parent_dir.mkdir(parents=True, exist_ok=True)
        desc_file = parent_dir / f"{cid}.json"
        desc_file.write_text(json.dumps({
            "conversationId": cid,
            "subagentDescriptor": {
                "typeName": "torch-coder"
            }
        }),
                             encoding="utf-8")
        with mock.patch("pathlib.Path.home", return_value=fake_home):
            name = run_state.get_google_harness_subagent_name(cid, "")
        self.assertEqual(name, "torch-coder")

    def test_google_harness_post_processing(self):
        # 1. Output extracted from transcript when None
        transcript_file = self.workspace_dir / "transcript_post.jsonl"
        transcript_file.write_text(
            json.dumps({
                "source": "MODEL",
                "content": "TPU kernel done"
            }),
            encoding="utf-8",
        )
        subagent, output = events.google_harness_post_processing(
            subagent="my-agent",
            subagent_output=None,
            transcript_path=str(transcript_file),
        )
        self.assertEqual(subagent, "my-agent")
        self.assertEqual(output, "TPU kernel done")

        # 2. Existing output preserved
        _, output_kept = events.google_harness_post_processing(
            subagent="my-agent",
            subagent_output="already_set",
            transcript_path=str(transcript_file),
        )
        self.assertEqual(output_kept, "already_set")


class TestSubagentDispatch(helpers.RuntimeTestCase):
    """Dispatching subagent-stop hooks."""

    def test_dispatch_hook_subagent_stop_e2e(self):
        self._write_script(
            "check_deliverable.py",
            "import sys, json\n"
            "data = json.loads(sys.stdin.read())\n"
            "if 'error' in str(data.get('output', '')).lower():\n"
            "    print(json.dumps({'decision': 'deny', 'reason': 'bad'}))\n"
            "else:\n"
            "    print(json.dumps({'decision': 'allow'}))\n",
        )
        matcher = "torch_coder|tpu-kernel-development"
        err = "Error during compilation."
        # (harness, event, payload, matcher, reply key, expected value)
        cases = [
            ("claude-code", "SubagentStop", {
                "agent_type": "torch_coder",
                "last_assistant_message": "All passed."
            }, None, "decision", None),
            # `status` is what Cursor's mapping points `stop_reason` at; read
            # as a termination reason it would look like a halt and excuse
            # the subagent, so only the Google harnesses are asked.
            ("cursor", "subagentStop", {
                "subagent_type": "torch_coder",
                "status": "completed",
                "summary": err
            }, None, "followup_message", "bad"),
            # Finished on its own terms, and matched by the emulated
            # matcher: held to its contract. A non-matching role is not.
            ("antigravity", "SubagentStop", {
                "role": "torch_coder",
                "terminationReason": "NO_TOOL_CALL",
                "toolOutput": err
            }, matcher, "decision", "continue"),
            ("antigravity", "SubagentStop", {
                "role": "doc_writer",
                "toolOutput": err
            }, matcher, "decision", "allow"),
            # A root session stop bypasses the SubagentStop hook.
            ("antigravity", "SubagentStop", {
                "conversationId": "root-conv-1234",
                "terminationReason": "USER_CANCELED",
                "toolOutput": err,
            }, None, "decision", "allow"),
        ]
        # A subagent cut short has no result to hold to a contract; an
        # unrecognised reason errs toward letting the halt stand.
        for reason in ("HALTED_STEP", "TERMINATION_REASON_FROM_THE_FUTURE"):
            cases.append(("antigravity", "SubagentStop", {
                "role": "torch_coder",
                "terminationReason": reason,
                "toolOutput": err
            }, None, "decision", "allow"))
        for harness, event, payload, match, key, expected in cases:
            with self.subTest(harness=harness, payload=payload):
                res, code = self._dispatch(
                    harness=harness,
                    hook_name="check-deliverable",
                    event=event,
                    script_rel="scripts/check_deliverable.py",
                    matcher=match,
                    raw_input_text=json.dumps(payload),
                )
                self.assertEqual((code, res.get(key)), (0, expected))


class TestStopEventDisambiguation(unittest.TestCase):
    """Antigravity emits one native `Stop` for two different things.

    A subagent finishing and the root loop ending arrive on the same wire
    event, so only the payload separates them. Getting it backwards sends a
    subagent's completion down the root path, where a resume hook would try to
    restart a conversation that already ended.
    """

    def setUp(self):
        self.workspace = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.workspace, ignore_errors=True)

    def _normalize(self, payload):
        return events.normalize_inbound_event(raw_payload=payload,
                                              harness="antigravity",
                                              event="Stop",
                                              package="pkg",
                                              workspace=self.workspace)

    def test_nested_subagent_metadata_is_a_subagent_stop(self):
        """The role can arrive inside the tool call rather than beside it."""
        event = self._normalize({
            "toolCall": {
                "typeName": "tpu-diagnostics"
            },
            "terminationReason": "NO_TOOL_CALL",
        })
        self.assertEqual(event["event"], "SubagentStop")
        self.assertEqual(event["subagent"], "tpu-diagnostics")


if __name__ == "__main__":
    unittest.main()
