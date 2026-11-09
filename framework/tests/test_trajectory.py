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
"""Tests for session trajectory export.

Every test points HOME at a temp dir and builds synthetic transcripts there,
so nothing reads a real session. Uploads go through an injected recorder, so
nothing touches the network.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import tarfile
import tempfile
import unittest
import unittest.mock

from framework import jsonio, paths, run_state, runtime, trajectory, utils
from framework.utils import TpuAgentSuiteError

Path = pathlib.Path


def _assistant(blocks: list[dict]) -> dict:
    """One assistant record carrying the given content blocks."""
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": blocks
        }
    }


def _tool_use(tool_id: str, name: str, payload: dict) -> dict:
    """One tool_use content block."""
    return {"type": "tool_use", "id": tool_id, "name": name, "input": payload}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as JSONL, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records),
                    encoding="utf-8")


def _skill_activation(tool_id: str = "t0") -> dict:
    """An assistant record that activates the tpu-agent skill."""
    return _assistant([_tool_use(tool_id, "Skill", {"skill": "tpu-agent"})])


def _claude_project(home: Path,
                    session: str,
                    records: list[dict],
                    agents: dict[str, dict] | None = None,
                    project: str = "-w") -> Path:
    """Lay out a Claude Code transcript plus its durable subagents directory.

    `agents` maps an agent id to {"meta": ..., "records": ...}; records
    default to one stub user record. Returns the project directory.
    """
    project_dir = home / ".claude" / "projects" / project
    _write_jsonl(project_dir / f"{session}.jsonl", records)
    subagents = project_dir / session / "subagents"
    subagents.mkdir(parents=True, exist_ok=True)
    for agent_id, spec in (agents or {}).items():
        (subagents / f"agent-{agent_id}.meta.json").write_text(
            json.dumps(spec["meta"]), encoding="utf-8")
        _write_jsonl(subagents / f"agent-{agent_id}.jsonl",
                     spec.get("records", [{
                         "type": "user"
                     }]))
    return project_dir


def _skill_file_read(step: int) -> dict:
    """A Google harness step that reads the tpu-agent skill document."""
    return {
        "step_index":
        step,
        "tool_calls": [{
            "name": "view_file",
            "args": {
                "AbsolutePath": "/w/.agents/skills/tpu-agent/SKILL.md"
            },
        }]
    }


def _slash_command(name: str, args: str = "") -> dict:
    """A user record shaped the way Claude Code logs a typed slash command."""
    text = (f"<command-message>{name}</command-message>\n"
            f"<command-name>/{name}</command-name>\n"
            f"<command-args>{args}</command-args>")
    return {"type": "user", "message": {"role": "user", "content": text}}


class HomeIsolatedTestCase(unittest.TestCase):
    """Base that redirects HOME so Path.home() resolves into a temp dir."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        env = unittest.mock.patch.dict(os.environ, {"HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)
        # CLAUDE_CONFIG_DIR overrides the home-relative path, so a caller who
        # has one exported would otherwise send every lookup elsewhere.
        os.environ.pop("CLAUDE_CONFIG_DIR", None)


class GoogleHarnessTestCase(HomeIsolatedTestCase):
    """Base that lays out Google harness brain directories under HOME."""

    def _brain(self,
               conversation_id: str,
               root: str = "antigravity-cli") -> Path:
        """Brain directory for one conversation, created on demand."""
        path = self.home / ".gemini" / root / "brain" / conversation_id
        (path / ".system_generated" / "subagents").mkdir(parents=True,
                                                         exist_ok=True)
        return path

    def _transcript(self,
                    conversation_id: str,
                    records: list[dict],
                    root: str = "antigravity-cli") -> None:
        """Write a conversation's full transcript."""
        brain = self._brain(conversation_id, root)
        _write_jsonl(
            brain / ".system_generated" / "logs" / "transcript_full.jsonl",
            records)

    def _descriptor(self, parent: str, child: str, spawn_step: int,
                    type_name: str) -> None:
        """Register `child` as a subagent spawned by `parent`."""
        brain = self._brain(parent)
        (brain / ".system_generated" / "subagents" /
         f"{child}.json").write_text(json.dumps({
             "conversationId": child,
             "subagentDescriptor": {
                 "typeName": type_name
             },
             "spawnStepIndex": spawn_step,
         }),
                                     encoding="utf-8")


class ClaudeCodeResolverTest(HomeIsolatedTestCase):
    """The Claude Code resolver: exact cut, nested subagents, pre-cut exclusion."""

    SESSION = "sess-1"

    def _build(self, records: list[dict], agents: dict[str, dict]) -> None:
        """Lay out a project transcript plus its durable subagents directory."""
        _claude_project(self.home, self.SESSION, records, agents)

    def test_slash_command_in_text_blocks_counts_but_tool_results_do_not(
            self) -> None:
        """Block-list content is read for text; an echoed transcript is not.

        The latest activation wins across shapes, and the export skill shares
        the prefix but must not move the cut.
        """
        tag = "<command-name>/tpu-agent</command-name>"
        as_text = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": tag
                }]
            }
        }
        as_tool_result = {
            "type": "user",
            "message": {
                "role":
                "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "t0",
                    "content": tag
                }]
            }
        }
        self._build([as_text], {})
        bundle = trajectory.resolve_claude_code(self.SESSION, "tpu-agent")
        self.assertEqual((bundle.cut.basis, bundle.cut.key),
                         ("slash-command", 0))
        self._build([
            _skill_activation(), as_text,
            _slash_command("tpu-agent-export-trajectory", "--all")
        ], {})
        bundle = trajectory.resolve_claude_code(self.SESSION, "tpu-agent")
        self.assertEqual((bundle.cut.basis, bundle.cut.key),
                         ("slash-command", 1))
        self._build([as_tool_result], {})
        bundle = trajectory.resolve_claude_code(self.SESSION, "tpu-agent")
        self.assertFalse(bundle.cut.detected)

    def test_excludes_pre_cut_and_includes_nested_subagents(self) -> None:
        """Delegations before the cut are dropped; nested ones are kept."""
        records = [
            _assistant([_tool_use("pre", "Agent", {})]),
            _skill_activation("cut"),
            _assistant([_tool_use("post", "Agent", {})]),
        ]
        agents = {
            # Launched before the cut, so out of scope.
            "old": {
                "meta": {
                    "toolUseId": "pre",
                    "agentType": "old",
                    "spawnDepth": 1
                }
            },
            # Launched after the cut, and itself launches a nested agent.
            "kid": {
                "meta": {
                    "toolUseId": "post",
                    "agentType": "tpu-model-onboarding",
                    "spawnDepth": 1
                },
                "records": [_assistant([_tool_use("deep", "Agent", {})])],
            },
            # Reachable only through "kid", which is what nesting must find.
            "grandkid": {
                "meta": {
                    "toolUseId": "deep",
                    "agentType": "tpu-kernel-development",
                    "spawnDepth": 2
                }
            },
        }
        self._build(records, agents)
        bundle = trajectory.resolve_claude_code(self.SESSION, "tpu-agent")
        by_label = {ref.label: ref for ref in bundle.transcripts}
        self.assertEqual(
            sorted(by_label),
            ["root", "tpu-kernel-development", "tpu-model-onboarding"])
        # A top-level subagent names the root, so the tree joins on ids.
        root = by_label["root"]
        self.assertEqual((root.agent_id, root.parent_id), (self.SESSION, ""))
        self.assertEqual(by_label["tpu-model-onboarding"].parent_id,
                         root.agent_id)

    def test_missing_transcript_is_an_error(self) -> None:
        """A session with no transcript cannot be exported. An eval trial
        files transcripts under CLAUDE_CONFIG_DIR, not under ~/.claude."""
        with self.assertRaises(TpuAgentSuiteError):
            trajectory.resolve_claude_code("absent", "tpu-agent")
        elsewhere = self.home / "logs" / "agent" / "sessions"
        _write_jsonl(elsewhere / "projects" / "-w" / "absent.jsonl",
                     [_skill_activation()])
        os.environ["CLAUDE_CONFIG_DIR"] = str(elsewhere)
        bundle = trajectory.resolve_claude_code("absent", "tpu-agent")
        self.assertTrue(bundle.cut.detected)

    def test_harness_version_and_models_come_from_the_transcript(self) -> None:
        """Version and model are read back rather than shelled out for."""
        records = [
            {
                "type": "assistant",
                "version": "2.1.267",
                "message": {
                    "role": "assistant",
                    "content": [],
                    "model": "claude-opus-5"
                }
            },
            _skill_activation(),
        ]
        self._build(records, {})
        bundle = trajectory.resolve_claude_code(self.SESSION, "tpu-agent")
        self.assertEqual(bundle.harness_version, "2.1.267")
        self.assertEqual(bundle.models["observed"], ["claude-opus-5"])


class GoogleHarnessResolverTest(GoogleHarnessTestCase):
    """The Google harness resolver: step-indexed cut and recursive descendants."""

    ROOT = "conv-root"

    def test_descendants_are_filtered_by_spawn_step_then_recursed(
            self) -> None:
        """Pre-cut children are dropped; a kept child's own children come along."""
        self._transcript(self.ROOT, [_skill_file_read(2)])
        self._descriptor(self.ROOT, "early", 1, "stale")
        self._descriptor(self.ROOT, "kept", 5, "tpu-sharding")
        self._transcript("early", [{"step_index": 1}])
        self._transcript("kept", [{"step_index": 1}])
        # Nested below "kept" at a step before the root's cut: still in scope,
        # because the cut applies only where spawning happened.
        self._descriptor("kept", "nested", 1, "tpu-diagnostics")
        self._transcript("nested", [{"step_index": 1}])

        bundle = trajectory.resolve_google_harness("antigravity", self.ROOT,
                                                   "tpu-agent")
        by_label = {ref.label: ref for ref in bundle.transcripts}
        self.assertEqual(sorted(by_label),
                         ["root", "tpu-diagnostics", "tpu-sharding"])
        # The manifest must describe the tree, not just the membership.
        self.assertEqual((by_label["tpu-sharding"].parent_id,
                          by_label["tpu-sharding"].depth), (self.ROOT, 1))
        self.assertEqual((by_label["tpu-diagnostics"].parent_id,
                          by_label["tpu-diagnostics"].depth), ("kept", 2))

    def test_brain_roots_of_another_harness_are_not_searched(self) -> None:
        """A root counts only when its name contains the harness name."""
        self._transcript(self.ROOT, [{
            "step_index": 1
        }],
                         root="other-google-harness")
        with self.assertRaises(TpuAgentSuiteError):
            trajectory.resolve_google_harness("antigravity", self.ROOT,
                                              "tpu-agent")


class SlicingAndCapTest(unittest.TestCase):
    """Generic slicing and the per-artifact size cap."""

    def test_field_mode_compares_a_record_field(self) -> None:
        """Google harness cuts compare step_index."""
        records = [{"step_index": 1}, {"step_index": 4}]
        cut = trajectory.Cut(detected=True,
                             basis="skill-file-read",
                             key=3,
                             mode="step_index")
        self.assertEqual(trajectory.slice_root_records(cut, records),
                         [{
                             "step_index": 4
                         }])

    def test_oversized_artifacts_are_dropped_and_reported(self) -> None:
        """A file over the cap is excluded, but never silently."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "small.txt").write_bytes(b"x" * 10)
            (root / "big.bin").write_bytes(b"x" * 100)
            included, dropped = trajectory.collect_run_files(root, cap=50)
            self.assertEqual([p.name for p in included], ["small.txt"])
            self.assertEqual(len(dropped), 1)
            self.assertIn("cap", dropped[0]["reason"])


class PushFeedbackTest(HomeIsolatedTestCase):
    """End-to-end bundling, with uploads captured rather than sent."""

    SESSION = "sess-push"

    def setUp(self) -> None:
        super().setUp()
        _claude_project(self.home, self.SESSION, [
            _assistant([_tool_use("t0", "Bash", {})]),
            _skill_activation("t1"),
        ])

        # A run namespace holding one small artifact.
        self.run_root = self.home / "run"
        (self.run_root / "artifacts").mkdir(parents=True)
        (self.run_root / "artifacts" / "result.txt").write_text("ok")
        # Blob names in upload order, and each blob's uploaded bytes.
        self.uploads: list[str] = []
        self.captured: dict[str, bytes] = {}

    def _record_upload(self, path: Path, bucket: str, blob: str) -> None:
        """Uploader stand-in that keeps what would have been sent."""
        self.uploads.append(blob)
        self.captured[blob] = path.read_bytes()

    def _push(self, **overrides):
        """Run a push with test defaults and a recording uploader."""
        kwargs = dict(
            harness="claude-code",
            session_id=self.SESSION,
            run_root=self.run_root,
            run_id="run-1",
            entrypoint="tpu-agent",
            message="compilation error",
            bucket="my-bucket",
            user_id="me@example.com",
            package=f"tpu-agent-suite@{utils.TPU_AGENT_SUITE_VERSION}",
            environment={},
            uploaded_at="2026-09-22T23:36:39+0000",
            uploader=self._record_upload,
        )
        kwargs.update(overrides)
        return trajectory.export_trajectory(**kwargs)

    def _uploaded(self, suffix: str) -> bytes:
        """The bytes of the one uploaded blob whose name ends in `suffix`."""
        return next(data for blob, data in self.captured.items()
                    if blob.endswith(suffix))

    def _bundle_names(self) -> list[str]:
        """Member names of the uploaded bundle tarball."""
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "b.tar.gz"
            archive.write_bytes(self._uploaded("bundle.tar.gz"))
            with tarfile.open(archive) as tar:
                return sorted(tar.getnames())

    def _write_session_without_activation(self) -> None:
        """Overwrite the session so it holds no tpu-agent activation."""
        _claude_project(self.home, self.SESSION,
                        [_assistant([_tool_use("t0", "Bash", {})])])

    def test_undetected_cut_refuses_without_consent(self) -> None:
        """Shipping a whole session is the user's call, not the tool's."""
        self._write_session_without_activation()
        with self.assertRaises(TpuAgentSuiteError) as caught:
            self._push()
        self.assertIn("--all", str(caught.exception))

    def test_consent_flag_records_its_basis(self) -> None:
        """With --all the manifest says the whole session was sent by consent."""
        self._write_session_without_activation()
        summary = self._push(include_full_session=True)
        self.assertEqual(summary["cut"]["basis"], "user-consented-all")
        self.assertFalse(summary["cut"]["detected"])

    def test_bundle_includes_session_journal_when_present(self) -> None:
        """journal.jsonl and experiment.md sit beside artifacts/ and ride in the tar."""
        from framework import journal
        (self.run_root / journal.JOURNAL_FILE).write_text('{"id": "H1"}\n',
                                                          encoding="utf-8")
        (self.run_root / journal.RENDERED_FILE).write_text(
            "# Session experiment journal\n", encoding="utf-8")
        self._push()
        # The manifest lands last, so seeing it implies a complete bundle.
        self.assertEqual([blob.rsplit("/", 1)[-1] for blob in self.uploads],
                         ["bundle.tar.gz", "manifest.json"])
        names = self._bundle_names()
        self.assertIn("artifacts/result.txt", names)
        self.assertIn(journal.JOURNAL_FILE, names)
        self.assertIn(journal.RENDERED_FILE, names)


class UnsupportedHarnessTest(unittest.TestCase):
    """Codex and Cursor ship the skill but refuse the export."""

    def test_refusal_names_the_harness(self) -> None:
        """A missing resolver entry is what produces the refusal."""
        for harness in ("codex", "cursor"):
            with self.subTest(harness=harness):
                self.assertFalse(trajectory.supported(harness))
                with self.assertRaises(TpuAgentSuiteError) as caught:
                    trajectory.resolve_session(harness, "s", "tpu-agent")
                self.assertIn(harness, str(caught.exception))
                self.assertIn("not supported", str(caught.exception))


class RuntimeCliTest(HomeIsolatedTestCase):
    """The `trajectory export` CLI path a compiled skill actually invokes."""

    RUN_ID = "20260923T070000Z-0123456789ab"
    SESSION = "cli-session"

    def setUp(self) -> None:
        super().setUp()
        # Central store and workspace both live under the redirected HOME.
        os.environ["TPU_AGENT_SUITE_HOME"] = str(self.home / "store")

        self.workspace = self.home / "ws"
        marker = self.workspace / ".tpu-agent-suite"
        marker.mkdir(parents=True)
        (marker / "current_run_id").write_text(self.RUN_ID, encoding="utf-8")
        jsonio.write_json(
            paths.get_sessions_root() / f"{self.SESSION}.json", {
                "session_id": self.SESSION,
                "run_id": self.RUN_ID,
                "created_at": "2026-09-23T07:00:00Z",
            })

        run_root = paths.get_run_root(self.RUN_ID)
        (run_root / "artifacts").mkdir(parents=True)
        (run_root / "artifacts" / "out.txt").write_text("result")

        _claude_project(self.home,
                        self.SESSION, [_skill_activation()],
                        project="-ws")

    def _run(self, *argv: str) -> tuple[int, str]:
        """Invoke the CLI with stdout captured, so tests stay readable."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = runtime.main(list(argv))
        return code, buffer.getvalue()

    def test_export_resolves_session_and_run_without_flags(self) -> None:
        """The session is recovered from the run_id."""
        with unittest.mock.patch.object(trajectory.bundle,
                                        "verify_bucket_exists"), \
             unittest.mock.patch.object(trajectory.bundle,
                                        "_default_uploader"):
            code, output = self._run("trajectory", "export", "--harness",
                                     "claude-code", "--workspace",
                                     str(self.workspace), "--run-id",
                                     self.RUN_ID, "--bucket", "gs://my-bucket",
                                     "--user-id", "a/tester", "--note",
                                     "it broke", "--json")
        self.assertEqual(code, 0)
        summary = json.loads(output)
        # The user id shapes the object path, never carrying a separator.
        self.assertIn("/a_tester/", summary["objects"][0]["name"])
        self.assertEqual(summary["cut"]["basis"], "skill-tool-use")

    def test_passed_run_without_a_session_is_refused(self) -> None:
        """A run no session record names is not exported."""
        unknown = "20260923T090000Z-000000000000"
        with self.assertRaises(TpuAgentSuiteError) as raised:
            runtime.resolve_trajectory_target(run_id=unknown)
        self.assertIn(unknown, str(raised.exception))


class GoogleHarnessTranscriptFallbackTest(GoogleHarnessTestCase):
    """The full transcript is preferred, the compact one is a usable stand-in.

    agy's `--print` mode writes no `transcript_full.jsonl` at all, so a
    resolver that only ever names that file silently yields nothing.
    """

    ROOT = "fallback-root"

    def _logs(self) -> pathlib.Path:
        """The conversation's logs directory, created on demand."""
        logs = self._brain(self.ROOT) / ".system_generated" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        return logs

    def test_compact_transcript_is_used_only_when_full_is_absent(self) -> None:
        """When both exist the untruncated one wins; otherwise the compact one
        is used rather than naming a file that does not exist.

        The fallback is recorded, not silent: a cut missed because of
        truncation must be explicable afterwards.
        """
        logs = self._logs()
        _write_jsonl(logs / "transcript_full.jsonl", [{"step_index": 1}])
        _write_jsonl(logs / "transcript.jsonl", [{"step_index": 1}])
        bundle = trajectory.resolve_google_harness("antigravity", self.ROOT,
                                                   "tpu-agent")
        self.assertEqual(bundle.transcripts[0].source, "transcript_full")

        (logs / "transcript_full.jsonl").unlink()
        bundle = trajectory.resolve_google_harness("antigravity", self.ROOT,
                                                   "tpu-agent")
        root = bundle.transcripts[0]
        self.assertEqual(root.source, "transcript")
        self.assertGreater(root.bytes, 0)
        self.assertTrue(any("compact transcript" in m for m in bundle.missing),
                        bundle.missing)


class GoogleHarnessCycleGuardTest(GoogleHarnessTestCase):
    """The descendant walk terminates on malformed descriptor graphs."""

    def _conversation(self, cid: str) -> None:
        """Create a conversation with a stub transcript."""
        self._transcript(cid, [{"step_index": 1}])

    def _spawn(self, parent: str, child: str) -> None:
        """Register `child` as spawned by `parent`."""
        self._descriptor(parent, child, 1, f"agent-{child}")

    def test_cycles_terminate_and_shared_children_are_collected_once(
            self) -> None:
        """A cycle would otherwise recurse until the stack limit, and two
        parents reaching one conversation must not duplicate it."""
        for cid in ("top", "left", "right", "shared"):
            self._conversation(cid)
        self._spawn("top", "left")
        self._spawn("top", "right")
        self._spawn("left", "shared")
        self._spawn("right", "shared")
        self._spawn("shared", "top")
        bundle = trajectory.resolve_google_harness("antigravity", "top",
                                                   "tpu-agent")
        ids = [r.agent_id for r in bundle.transcripts if r.kind == "subagent"]
        self.assertEqual(sorted(ids), ["left", "right", "shared"])
        self.assertEqual(len(ids), len(set(ids)))


class StageSessionDirTest(HomeIsolatedTestCase):
    """Staging for Harbor: one session per directory, sliced like the bundle."""

    SESSION = "sess-stage"

    def test_layout_matches_what_harbor_globs(self) -> None:
        """Subagents land under subagents/, where harbor's rglob looks.

        A neighbour session in the same project directory is not staged
        (harbor merges every *.jsonl in the directory it is handed), and the
        root transcript drops pre-cut records, matching what write_bundle
        ships.
        """
        project = _claude_project(
            self.home,
            self.SESSION, [
                _assistant([_tool_use("t0", "Bash", {})]),
                _skill_activation(),
                _assistant([_tool_use("call", "Agent", {})]),
            ],
            agents={
                "a1": {
                    "meta": {
                        "toolUseId": "call",
                        "agentType": "tpu-kernel-development",
                        "spawnDepth": 1
                    }
                }
            })
        _write_jsonl(project / "sess-neighbour.jsonl",
                     [_assistant([_tool_use("n0", "Bash", {})])])
        bundle = trajectory.resolve_claude_code(self.SESSION, "tpu-agent")
        with tempfile.TemporaryDirectory() as tmp:
            staged = trajectory.stage_session_dir(bundle, Path(tmp))
            roots = sorted(p.name for p in staged.glob("*.jsonl"))
            nested = sorted(p.name for p in staged.rglob("subagents/*.jsonl"))
            records = jsonio.read_jsonl(staged / f"{self.SESSION}.jsonl")
        self.assertEqual(roots, [f"{self.SESSION}.jsonl"])
        self.assertTrue(nested, "subagent transcript should be staged")
        # The pre-cut Bash call is gone; the activation and its work remain.
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["message"]["content"][0]["input"]["skill"],
                         "tpu-agent")


class HarborAvailabilityTest(unittest.TestCase):
    """Both halves of harbor are required and can be installed apart."""

    def setUp(self) -> None:
        trajectory.harbor_missing.cache_clear()

    def tearDown(self) -> None:
        trajectory.harbor_missing.cache_clear()

    def test_each_missing_half_is_reported(self) -> None:
        """A library-only install would fail the upload with FileNotFoundError;
        a pipx install (binary but not importable) would fail the converter."""
        for importable, on_path, expected in ((True, True, None), (True, False,
                                                                   "PATH"),
                                              (False, True, "importable")):
            with self.subTest(importable=importable, on_path=on_path):
                trajectory.harbor_missing.cache_clear()
                spec = object() if importable else None
                which = "/usr/bin/harbor" if on_path else None
                with unittest.mock.patch("importlib.util.find_spec",
                                         return_value=spec), \
                     unittest.mock.patch("shutil.which", return_value=which):
                    missing = trajectory.harbor_missing()
                if expected is None:
                    self.assertEqual(missing, "")
                else:
                    self.assertIn(expected, missing)


class DestinationSelectionTest(HomeIsolatedTestCase):
    """Harbor replaces the bucket only when the user chose it."""

    def _session(self, records: list[dict] | None = None) -> None:
        """Write session `sess` and give it a recorded run with a directory.

        The transcript holds a tpu-agent activation unless `records` says
        otherwise.
        """
        _claude_project(self.home, "sess", records or [_skill_activation()])
        run_id = "20260923T070000Z-0123456789ab"
        run_state.record_session_run("sess", run_id)
        paths.get_run_root(run_id).mkdir(parents=True)

    def _args(self, **over):
        base = dict(harness="claude-code",
                    workspace=str(self.home),
                    package="",
                    bucket="",
                    user_id="u",
                    share_org="",
                    destination="",
                    session_id="sess",
                    run_id="",
                    entrypoint="tpu-agent",
                    note="",
                    json=False,
                    include_full_session=True)
        base.update(over)
        return unittest.mock.Mock(**base)

    def test_declining_export_does_not_reach_harbor(self) -> None:
        """A blank bucket means declined, so nothing is uploaded anywhere.

        Naming no bucket is a change of mind, not a broken install, and it is
        reported before a missing starting point, so the user is asked whether
        to share before whether to share the whole session.
        """
        self._session([_assistant([_tool_use("t0", "Bash", {})])])
        with unittest.mock.patch.object(trajectory, "export_to_harbor") as hub:
            with self.assertRaises(TpuAgentSuiteError) as caught:
                runtime.handle_trajectory_cli(
                    self._args(include_full_session=False))
        hub.assert_not_called()
        self.assertIn("no destination is configured", str(caught.exception))

    def test_choosing_harbor_skips_the_bucket(self) -> None:
        """The two destinations are exclusive, not a try-then-fallback, and
        availability is not consent: the default destination is the bucket."""
        self._session()
        with unittest.mock.patch.object(trajectory, "harbor_missing",
                                        return_value=""), \
             unittest.mock.patch.object(
                 trajectory, "export_to_harbor",
                 return_value={"steps": 1, "destination": "https://hub/x"}) as hub, \
             unittest.mock.patch.object(trajectory, "export_trajectory",
                                        return_value={
                                            "uri": "gs://b/p",
                                            "objects": [],
                                            "transcripts": 1,
                                            "cut": {"detected": True, "basis": "skill-tool-use"},
                                            "dropped": [],
                                        }) as gcs:
            runtime.handle_trajectory_cli(self._args(bucket="gs://b"))
            hub.assert_not_called()
            gcs.assert_called_once()
            gcs.reset_mock()
            runtime.handle_trajectory_cli(self._args(destination="harbor"))
        hub.assert_called_once()
        gcs.assert_not_called()

    def _stage(self, **kwargs):
        """Stand-in for export_to_harbor, i.e. write a file into the job dir."""
        kwargs["job_dir"].mkdir(parents=True)
        (kwargs["job_dir"] / "trajectory.json").write_text("{}")
        self.staged = kwargs["job_dir"]
        return {
            "steps": 1,
            "destination": "https://hub/x",
            "job_dir": str(kwargs["job_dir"])
        }

    def test_harbor_job_is_staged_outside_runs_and_removed(self) -> None:
        """The job should be staged outside to avoid being read by later
        agents. A failed upload keeps it, since the error names it."""
        self._session()
        with unittest.mock.patch.object(trajectory, "harbor_missing",
                                        return_value=""), \
             unittest.mock.patch.object(trajectory, "export_to_harbor",
                                        side_effect=self._stage):
            runtime.handle_trajectory_cli(
                self._args(destination="harbor", json=True))
        runs = paths.get_run_root("20260923T070000Z-0123456789ab").parent
        self.assertFalse(self.staged.is_relative_to(runs))
        self.assertFalse(self.staged.parent.exists())

        def fail(**kwargs):
            self._stage(**kwargs)
            raise TpuAgentSuiteError("upload failed")

        with unittest.mock.patch.object(trajectory, "harbor_missing",
                                        return_value=""), \
             unittest.mock.patch.object(trajectory, "export_to_harbor",
                                        side_effect=fail):
            with self.assertRaises(TpuAgentSuiteError):
                runtime.handle_trajectory_cli(self._args(destination="harbor"))
        self.assertTrue((self.staged / "trajectory.json").is_file())
        shutil.rmtree(self.staged.parent)

    def test_unusable_harbor_is_an_error_not_a_silent_bucket_export(
            self) -> None:
        """Choosing harbor and not having it is reported, not worked around."""
        self._session()
        with unittest.mock.patch.object(trajectory, "harbor_missing",
                                        return_value="the `harbor` command is"
                                        " not on PATH"), \
             unittest.mock.patch.object(trajectory, "export_trajectory") as gcs:
            with self.assertRaises(TpuAgentSuiteError) as caught:
                runtime.handle_trajectory_cli(self._args(destination="harbor"))
        gcs.assert_not_called()
        self.assertIn("PATH", str(caught.exception))


class HarborPayloadTest(HomeIsolatedTestCase):
    """With no bucket behind it, the job must carry everything the bucket would."""

    def setUp(self) -> None:
        super().setUp()
        _claude_project(self.home, "sess", [_skill_activation()])

    def _export(self, job: Path, trial: Path, stdout: str, **kwargs):
        """export_to_harbor with conversion and the upload subprocess faked.

        The job is written to `trial`; returns the summary and the
        subprocess mock so a test can inspect the upload command.
        """
        bundle = trajectory.resolve_for_export("claude-code", "sess",
                                               "tpu-agent", True)
        with unittest.mock.patch.object(trajectory.bundle, "_build_trajectory",
                                        return_value=unittest.mock.Mock(steps=[])), \
             unittest.mock.patch.object(trajectory.bundle, "write_job_dir",
                                        side_effect=lambda *a, **k:
                                        (trial.mkdir(parents=True,
                                                     exist_ok=True), trial)[1]), \
             unittest.mock.patch.object(trajectory.bundle, "subprocess") as sp:
            sp.TimeoutExpired = subprocess.TimeoutExpired
            sp.run.return_value = unittest.mock.Mock(returncode=0,
                                                     stdout=stdout,
                                                     stderr="")
            summary = trajectory.export_to_harbor(bundle=bundle,
                                                  job_dir=job,
                                                  environment={},
                                                  share_org="my-team",
                                                  **kwargs)
        return summary, sp

    def test_manifest_and_bundle_ride_along(self) -> None:
        """The note, the consent basis and the run's artifacts are not dropped.

        --share-org prompts without --yes, and a captured prompt hangs, so
        the upload passes --yes and a timeout.
        """
        run_root = self.home / "run"
        (run_root / "artifacts").mkdir(parents=True)
        (run_root / "artifacts" / "log.txt").write_text("boom",
                                                        encoding="utf-8")

        trial = self.home / "job" / "sess__session"
        summary, sp = self._export(self.home / "job",
                                   trial,
                                   stdout="https://hub/x",
                                   run_root=run_root,
                                   run_id="r1",
                                   message="a note",
                                   package="p")

        self.assertTrue((trial / trajectory.MANIFEST_NAME).is_file())
        self.assertTrue((trial / trajectory.BUNDLE_NAME).is_file())
        manifest = json.loads((trial / trajectory.MANIFEST_NAME).read_text())
        self.assertEqual(manifest["note"]["text"], "a note")
        self.assertTrue(manifest["note"]["provided"])
        self.assertEqual(manifest["cut"]["basis"], "skill-tool-use")
        with tarfile.open(trial / trajectory.BUNDLE_NAME) as tar:
            names = tar.getnames()
        self.assertIn("artifacts/log.txt", names)
        self.assertEqual(summary["destination"], "https://hub/x")
        cmd = sp.run.call_args[0][0]
        self.assertIn("--yes", cmd)
        self.assertEqual(cmd[cmd.index("--share-org") + 1], "my-team")
        self.assertEqual(sp.run.call_args[1]["timeout"],
                         trajectory.bundle.UPLOAD_TIMEOUT_SECONDS)


if __name__ == "__main__":
    unittest.main()
