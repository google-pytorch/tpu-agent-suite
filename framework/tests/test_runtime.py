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
"""Unit tests for standalone runtime operations."""

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

from framework import (agent_contracts, bg_ledger, jsonio, paths, run_state,
                       runtime, utils)
from framework.builder import hook_config
from framework.hooks import actions, dispatch, events, stop_filter

Path = pathlib.Path

# The shipped package's schemas, from the repository root.
_PACKAGE_SCHEMAS = Path(__file__).resolve().parents[2] / "package" / "schemas"

_no_lease_sweep = mock.patch("framework.compute.lease.spawn_sweep")


def setUpModule():
    # Every builtin start-run starts a detached sweep of the real job store.
    _no_lease_sweep.start()


def tearDownModule():
    _no_lease_sweep.stop()


def _isolated_workspace(test: unittest.TestCase,
                        home_name: str) -> tuple[Path, Path]:
    """A temp dir holding a workspace, with the store pointed beside it.

    Returns the temp dir and its `workspace`; `TPU_AGENT_SUITE_HOME` names
    `home_name` inside the temp dir, which is not created. Both the directory
    and the variable are restored when `test` ends.
    """
    temp_dir = Path(tempfile.mkdtemp())
    test.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
    workspace = temp_dir / "workspace"
    workspace.mkdir()
    home = mock.patch.dict(os.environ,
                           {"TPU_AGENT_SUITE_HOME": str(temp_dir / home_name)})
    home.start()
    test.addCleanup(home.stop)
    return temp_dir, workspace


class TestRuntime(unittest.TestCase):
    """Unit tests for runtime namespace allocation, hooks, and validations."""

    def setUp(self):
        super().setUp()
        self.temp_dir, self.workspace_dir = _isolated_workspace(
            self, "central_tpu_agent_suite")
        self.central_home = self.temp_dir / "central_tpu_agent_suite"
        self.central_home.mkdir()

        # Set up installed package structure: the payload under its version,
        # and a user-scope environment.json at the central store's root.
        self.pkg_root = self.central_home / "packages" / "test-pkg@1.0.0"
        self.pkg_root.mkdir(parents=True)
        (self.central_home / "environment.json").write_text(json.dumps({
            "revision":
            1,
            "target":
            "local"
        }),
                                                            encoding="utf-8")
        (self.pkg_root / "schemas").mkdir()
        (self.pkg_root / "schemas" / "out.json").write_text(
            json.dumps({
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string"
                    }
                },
                "required": ["status"],
            }),
            encoding="utf-8",
        )

    def _write_script(self, name: str, source: str) -> Path:
        """Writes `source` as the package's `scripts/<name>`."""
        script = self.pkg_root / "scripts" / name
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(source, encoding="utf-8")
        return script

    def _dispatch(self, **kwargs) -> tuple[dict, int]:
        """`dispatch_hook` for the test package, in the test workspace."""
        kwargs.setdefault("package", "test-pkg@1.0.0")
        kwargs.setdefault("workspace", self.workspace_dir)
        return dispatch.dispatch_hook(**kwargs)

    def _main_capturing(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = runtime.main(argv)
        return code, out.getvalue(), err.getvalue()

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

    def test_validate_cli_exit_code_and_completed_artifact_rule(self):
        """`validate` exits 0 on a valid instance and non-zero otherwise, and
        applies the runtime's completed-artifact rule, not only the schema."""
        schema = self.temp_dir / "result.json"
        schema.write_text(json.dumps({
            "type": "object",
            "required": ["status"],
            "properties": {
                "status": {
                    "type": "string"
                },
                "artifacts": {
                    "type": "array"
                },
            },
        }),
                          encoding="utf-8")
        instance = self.temp_dir / "instance.json"
        for doc, ok in (({
                "status": "ok"
        }, True), ({
                "status": "completed"
        }, False)):
            with self.subTest(doc=doc):
                instance.write_text(json.dumps(doc), encoding="utf-8")
                code, _, _ = self._main_capturing(
                    ["validate", "--schema",
                     str(schema),
                     str(instance)])
                self.assertEqual(code == 0, ok)

    def _package_schemas(self) -> dict[str, dict]:
        schemas = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(_PACKAGE_SCHEMAS.glob("*.json"))
        }
        self.assertTrue(schemas, f"no schemas under {_PACKAGE_SCHEMAS}")
        return {k: v for k, v in schemas.items() if isinstance(v, dict)}

    def test_status_requires_is_coherent_in_every_package_schema(self):
        """A status that claims an outcome owes its payload; others owe nothing.

        Success-only fields must not be `required`, empty values do not count
        (but `succeeded: false` does), and every other status in the schema's
        own enum must pass without them.
        """
        check = agent_contracts.check_status_payload
        found = {
            name: schema
            for name, schema in self._package_schemas().items()
            if "status_requires" in schema
        }
        self.assertTrue(found, "no schema declares status_requires")
        for name, schema in found.items():
            owed = schema["status_requires"]
            for status, fields in owed.items():
                with self.subTest(schema=name, status=status):
                    self.assertFalse(
                        set(fields) & set(schema["required"]),
                        "a success-only field is required")
                    payload = {"status": status, "run_id": "x"}
                    self.assertTrue(check(schema, payload))
                    payload.update({field: "filled" for field in fields})
                    self.assertFalse(check(schema, payload))
                    for field in fields:
                        empty = [] if field == "artifacts" else ""
                        self.assertTrue(
                            check(schema, {
                                **payload, field: empty
                            }), field)
                    if "succeeded" in fields:
                        self.assertFalse(
                            check(schema, {
                                **payload, "succeeded": False
                            }))
            unfinished = [
                status for status in schema["properties"]["status"]["enum"]
                if status not in owed
            ]
            self.assertTrue(unfinished, f"{name} cannot report failure")
            for status in unfinished:
                with self.subTest(schema=name, status=status):
                    self.assertFalse(
                        check(schema, {
                            "status": status,
                            "failure_reason": "no_device"
                        }))

    def test_schema_examples_satisfy_their_own_schema(self):
        """Denials quote an example back to the agent as the expected output;
        one that does not validate is a retry loop."""
        checked = 0
        for name, schema in self._package_schemas().items():
            for index, example in enumerate(schema.get("examples", [])):
                checked += 1
                with self.subTest(schema=name, example=index):
                    errors = utils.validate_instance(example, schema)
                    errors += agent_contracts.check_status_payload(
                        schema, example, name)
                    self.assertFalse(errors)
        self.assertGreater(checked, 0, "no schema declares an example")

    def test_status_requires_must_name_declared_statuses_and_properties(self):
        """A misspelled name has to fail the build, not the rule.

        The rule fires only on the status it is keyed to, so a typo in either
        the status or a field name enforces nothing while the schema keeps
        validating -- it fails open, quietly, in exactly the case it exists to
        catch. Now that each package names its own statuses there is no shared
        vocabulary to fall back on, so a mistyped status is the likelier of the
        two. Catching both at verify time is why the rule is worth declaring in
        the schema at all.
        """
        good = {
            "type": "object",
            "status_requires": {
                "measured": ["latency_ms"]
            },
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["measured", "failed"]
                },
                "latency_ms": {
                    "type": "number"
                },
            },
        }
        self.assertFalse(utils.validate_schema_definition(good))

        misspelled_field = json.loads(json.dumps(good))
        misspelled_field["status_requires"]["measured"] = [
            "latency_ms", "nonexistent_field"
        ]
        errors = utils.validate_schema_definition(misspelled_field)
        self.assertTrue(errors)
        self.assertIn("nonexistent_field", errors[0])

        misspelled_status = json.loads(json.dumps(good))
        misspelled_status["status_requires"] = {"measurred": ["latency_ms"]}
        errors = utils.validate_schema_definition(misspelled_status)
        self.assertTrue(errors)
        self.assertIn("measurred", errors[0])

    def test_unversioned_package_resolves_to_central_install(self):
        """A caller who names a package without its version must still find it.

        Versions are an install detail no agent tracks, so an unversioned
        lookup is the normal case. Missing the versioned directory sends the
        caller to a workspace path holding only part of the package, and
        anything installed solely in central storage then reads as absent --
        with the error naming that content rather than the resolution.
        """
        central = self.central_home / "packages" / "central-pkg@2.3.0"
        (central / "schemas").mkdir(parents=True)
        (central / "schemas" / "schema.json").write_text(json.dumps(
            {"type": "object"}),
                                                         encoding="utf-8")

        resolved = paths.get_installed_package_root("central-pkg")
        self.assertEqual(resolved.resolve(), central.resolve())
        self.assertTrue((resolved / "schemas" / "schema.json").is_file())
        # Compiled prompts pass --package as <name>@<version>; the tag must
        # resolve to the versioned directory the installer created.
        self.assertEqual(
            paths.get_installed_package_root("test-pkg@1.0.0").resolve(),
            self.pkg_root.resolve())

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

    def test_cli_start_run_reads_workspace_environment(self):
        """`start-run` without --workspace consults the cwd's workspace install."""
        ws = self.temp_dir / "proj_ws"
        pkg_cfg = ws / ".tpu-agent-suite"
        pkg_cfg.mkdir(parents=True)
        (pkg_cfg / "environment.json").write_text(json.dumps({
            "revision": 3,
            "target": "proj"
        }),
                                                  encoding="utf-8")

        old_cwd = Path.cwd()
        buf = io.StringIO()
        try:
            os.chdir(ws)
            with contextlib.redirect_stdout(buf):
                ret = runtime.main(["start-run", "--package", "wspkg@1.0.0"])
        finally:
            os.chdir(old_cwd)

        self.assertEqual(ret, 0)
        doc = json.loads(buf.getvalue())
        self.assertEqual(doc["environment_revision"], 3)

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

    def test_purge_data_clears_runs_and_session_records(self):
        """Purging runs must not leave sessions pointing at deleted namespaces.

        Session records are derived from runs: they name a run_id and cache the
        context advertising its artifact path. Surviving a purge, they would
        hand a live conversation a namespace that no longer exists.
        """
        run_doc = run_state.start_run(package="test-pkg@1.0.0",
                                      workspace=self.workspace_dir)
        run_root = paths.get_run_root(run_doc["run_id"])
        self.assertTrue((run_root / "artifacts").is_dir())
        self.assertEqual(paths.get_active_run_id(self.workspace_dir),
                         run_doc["run_id"])
        run_state.record_session_run(
            session_id="conv-purge",
            run_id=run_doc["run_id"],
            package="test-pkg@1.0.0",
        )
        run_state.record_session_hook_context(
            session_id="conv-purge",
            hook_name="session-run-init",
            context_text="cached context",
            package="test-pkg@1.0.0",
        )
        self.assertTrue(run_root.is_dir())

        with contextlib.redirect_stdout(io.StringIO()):
            run_state.purge_data(force=True)

        self.assertFalse(paths.get_runs_root().is_dir())
        self.assertIsNone(run_state.resolve_session_run_id("conv-purge"))
        self.assertIsNone(
            run_state.resolve_session_hook_context("conv-purge",
                                                   "session-run-init"))

    def test_purge_data_declined_keeps_everything(self):
        run_state.record_session_run(
            session_id="conv-keep-purge",
            run_id="20260913T170000Z-998877665544",
            package="test-pkg@1.0.0",
        )
        with mock.patch("builtins.input", return_value="n"):
            with contextlib.redirect_stderr(io.StringIO()) as err:
                run_state.purge_data()

        self.assertIn("cancelled", err.getvalue())
        self.assertEqual(run_state.resolve_session_run_id("conv-keep-purge"),
                         "20260913T170000Z-998877665544")

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


class TestWorkspaceRootResolution(unittest.TestCase):
    """Where a hook decides the workspace is.

    `hooks.json` installs inside the harness's own directory, so a hook command
    runs with that directory as its cwd rather than the workspace. Resolving
    from there has to climb out, or every run-scoped path the hook computes is
    wrong by one level -- silently, since each path still exists.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.workspace = self.tmp / "workspace"
        (self.workspace / ".tpu-agent-suite").mkdir(parents=True)
        (self.workspace / ".tpu-agent-suite" / "current_run_id").write_text(
            "20260928T225805Z-a09b00bd2610", encoding="utf-8")

    def test_harness_directories_resolve_to_their_parent(self):
        """Every directory a harness owns, including Antigravity's `.agents`."""
        for name in (".agents", ".agent", ".claude", ".cursor"):
            with self.subTest(harness_dir=name):
                harness_dir = self.workspace / name
                harness_dir.mkdir(exist_ok=True)
                with mock.patch.object(pathlib.Path,
                                       "cwd",
                                       return_value=harness_dir):
                    self.assertEqual(events.resolve_workspace_root(),
                                     self.workspace.resolve())

    def test_a_repository_is_not_a_workspace(self):
        """With no active run above the cwd, the cwd answers, not a git root.

        A bare `.tpu-agent-suite` (the installer and logging hooks create one
        where no session started) is not a marker; only the active-run file is.
        """
        repo = self.tmp / "bare-repo"
        (repo / ".git").mkdir(parents=True)
        (repo / ".tpu-agent-suite" / "logs").mkdir(parents=True)
        nested = repo / "src"
        nested.mkdir()
        with mock.patch.object(pathlib.Path, "cwd", return_value=nested):
            self.assertEqual(events.resolve_workspace_root(), nested.resolve())


class TestCompletedArtifactRuleIsConditional(unittest.TestCase):
    """Only a `completed` result must carry an artifact.

    The rule lives in the runtime rather than the schema because the validator
    has no conditional keywords, and stating it unconditionally would be worse
    than not stating it: an agent that failed has nothing to attach, so the only
    way to satisfy the rule would be to claim success.
    """

    def test_only_a_completed_result_owes_an_artifact(self):
        for result, owes in (
            ({
                "status": "completed",
                "summary": "all done"
            }, True),
            ({
                "status": "completed",
                "artifacts": []
            }, True),
            ({
                "status": "completed",
                "artifacts": [{
                    "file": "report.md"
                }]
            }, False),
            ({
                "status": "failed",
                "artifacts": []
            }, False),
            ({
                "status": "needs_input"
            }, False),
            ("not a result object", False),
        ):
            with self.subTest(result=result):
                self.assertEqual(
                    bool(agent_contracts.check_completed_has_artifact(result)),
                    owes)


class TestInstalledPackageResolution(unittest.TestCase):
    """Finding the versioned package directory under the TPU agent suite home."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.packages = self.tmp / "packages"
        self.packages.mkdir()

    def test_the_highest_installed_version_wins(self):
        for version in ("0.9.0", "0.10.0", "0.2.0"):
            (self.packages / f"pkg@{version}").mkdir()
        # None of these is a candidate.
        (self.packages / "other@9.0.0").mkdir()
        (self.packages / "pkg@8.0.0").write_text("not a directory",
                                                 encoding="utf-8")
        (self.packages / "pkg@not-a-version").mkdir()
        (self.packages / "unversioned").mkdir()
        with mock.patch.object(paths,
                               "get_tpu_agent_suite_home",
                               return_value=self.tmp):
            resolved = paths.latest_installed_package("pkg")
        # 0.10.0 beats 0.9.0 numerically, which string ordering gets wrong.
        self.assertEqual(resolved.name, "pkg@0.10.0")


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


class TestBuiltinValidateSchema(unittest.TestCase):
    """Unit tests for builtin:validate-schema hook execution."""

    def setUp(self):
        super().setUp()
        self.temp_dir, self.workspace_dir = _isolated_workspace(
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
        self.temp_dir, self.workspace_dir = _isolated_workspace(
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


class TestMissingPackageDegradesGracefully(unittest.TestCase):
    """A hook whose package has gone missing must still answer the harness.

    Hook commands bake a versioned package tag at compile time, so upgrading,
    renaming, or removing the package mid-session leaves the tag pointing at a
    directory that no longer exists. Package resolution raises for that, and an
    escaping error costs the dispatch its stdout: the harness is handed an
    empty string and a non-zero exit instead of a decision.
    """

    def setUp(self):
        super().setUp()
        # An empty store: nothing the hook names can resolve.
        self.temp_dir, self.workspace_dir = _isolated_workspace(
            self, "empty_home")

    def test_validate_schema_builtin_still_returns_a_decision(self):
        """Validation degrades to "no contract", the way its other failures
        do (absent or unparseable build file, unknown agent) -- but says so,
        since enforcing nothing silently is its own failure."""
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            res = agent_contracts.builtin_validate_schema(
                package="ghost-pkg@9.9.9",
                workspace=self.workspace_dir,
                event_payload={
                    "event": "SubagentStop",
                    "subagent": "helper",
                    "output": json.dumps({"status": "completed"}),
                },
            )
        self.assertEqual(res["decision"], "allow")
        self.assertIn("ghost-pkg@9.9.9", stderr.getvalue())

    def test_script_hook_denies_with_the_reason(self):
        """A script hook denies instead of skipping, as it does for any
        script it cannot locate -- but the harness still gets a decision.
        """
        with contextlib.redirect_stderr(io.StringIO()):
            response, code = dispatch.dispatch_hook(
                harness="claude-code",
                package="ghost-pkg@9.9.9",
                hook_name="some-script-hook",
                event="PreToolUse",
                script_rel="scripts/whatever.py",
                workspace=self.workspace_dir,
                raw_input_text=json.dumps({
                    "session_id": "s1",
                    "tool_name": "Bash",
                    "tool_input": {},
                }),
            )

        self.assertEqual(code, 0)
        specific = response["hookSpecificOutput"]
        self.assertEqual(specific["permissionDecision"], "deny")
        self.assertIn("is not installed", specific["permissionDecisionReason"])


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
        self.temp_dir, self.workspace = _isolated_workspace(self, "home")
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
        self.tmp, self.workspace = _isolated_workspace(self, "home")

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
        self.temp_dir, self.workspace_dir = _isolated_workspace(
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
