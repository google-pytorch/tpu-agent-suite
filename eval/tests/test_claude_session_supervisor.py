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
"""Tests for the Claude Code session supervisor, run against a fake CLI.

Harbor builds a trial's trajectory from the one project directory under
CLAUDE_CONFIG_DIR/projects; with two, it builds none and every check scores
zero. These pin that the supervisor leaves the root session in one.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

_SUPERVISOR = (Path(__file__).resolve().parents[1] / "hooks" / "claude_code" /
               "session_supervisor.sh")

_SESSION = "8161b680-cf5b-480d-bb2f-5e40d7af3a99"

# Stands in for claude-real. FAKE_SESSION, when set, goes in the stream as
# the session id; FAKE_SAVE, when set, is the file the session is saved to,
# as Claude Code saves it.
_FAKE_CLI = textwrap.dedent("""\
    #!/usr/bin/env bash
    echo "$*" >> "$FAKE_ARGS_LOG"
    if [ -n "${FAKE_SESSION:-}" ]; then
      printf '{"type":"system","session_id":"%s"}\\n' "$FAKE_SESSION"
    fi
    printf '{"type":"assistant","message":{"content":[{"type":"text","text":"hello"}]}}\\n'
    if [ -n "${FAKE_SAVE:-}" ]; then
      mkdir -p "$(dirname "$FAKE_SAVE")"
      printf '{"type":"assistant","message":{"content":[{"type":"text","text":"saved"}]}}\\n' > "$FAKE_SAVE"
    fi
    """)


def _slug(path: Path) -> str:
    """The project directory Claude Code names for a cwd."""
    return re.sub(r"[^A-Za-z0-9]", "-", str(path))


@unittest.skipUnless(shutil.which("jq"), "the supervisor needs jq")
class ClaudeSessionSupervisorTest(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.config = self.root / "config"
        self.projects = self.config / "projects"
        self.projects.mkdir(parents=True)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        fake = self.root / "claude-real"
        fake.write_text(_FAKE_CLI, encoding="utf-8")
        fake.chmod(0o755)
        self.args_log = self.root / "args.log"
        self.env = {
            "PATH": os.environ["PATH"],
            "HOME": str(self.root / "home"),
            "CLAUDE_CONFIG_DIR": str(self.config),
            "CLAUDE_REAL_BIN": str(fake),
            "FAKE_ARGS_LOG": str(self.args_log),
        }

    def _run(self, args: list[str], cwd: Path, **env: str) -> None:
        done = subprocess.run(["bash", str(_SUPERVISOR), *args],
                              cwd=cwd,
                              env={
                                  **self.env,
                                  **env
                              },
                              capture_output=True,
                              text=True,
                              timeout=60,
                              check=False)
        self.assertEqual(done.returncode, 0, done.stderr)

    def _project_dirs(self) -> list[str]:
        return sorted(p.name for p in self.projects.iterdir() if p.is_dir())

    def test_nested_run_without_persistence_makes_no_project(self):
        own = self.projects / _slug(self.workspace)
        own.mkdir()
        (own / f"{_SESSION}.jsonl").write_text("{}\n", encoding="utf-8")
        elsewhere = self.root / "knowledge-base-kwa2f3og"
        elsewhere.mkdir()

        self._run([
            "-p", "--no-session-persistence", "--",
            "Summarise this page. --no-session-persistence"
        ], elsewhere)

        self.assertEqual(self._project_dirs(), [own.name])
        self.assertEqual([p.name for p in own.iterdir()],
                         [f"{_SESSION}.jsonl"])
        self.assertIn("--no-session-persistence",
                      self.args_log.read_text(encoding="utf-8"))

    def test_session_is_tagged_where_claude_saved_it(self):
        stray = self.projects / _slug(self.root / "knowledge-base-kwa2f3og")
        stray.mkdir()
        saved = self.projects / _slug(self.workspace) / f"{_SESSION}.jsonl"

        self._run(["--print"],
                  self.workspace,
                  FAKE_SESSION=_SESSION,
                  FAKE_SAVE=str(saved))

        self.assertEqual(list(stray.iterdir()), [])
        record = json.loads(saved.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(record["message"]["content"][0]["text"],
                         "[Depth 0] saved")

    def test_unsaved_session_is_synthesized_under_its_cwd(self):
        stray = self.projects / _slug(self.root / "knowledge-base-kwa2f3og")
        stray.mkdir()

        self._run(["--print"], self.workspace, FAKE_SESSION=_SESSION)

        self.assertEqual(list(stray.iterdir()), [])
        synthesized = (self.projects / _slug(self.workspace) /
                       f"{_SESSION}.jsonl")
        lines = synthesized.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("[Depth 0] hello", lines[1])

    def test_stream_without_session_id_still_names_the_file(self):
        self._run(["--print"], self.workspace)

        own = self.projects / _slug(self.workspace)
        names = [p.name for p in own.iterdir()]
        self.assertEqual(len(names), 1)
        self.assertRegex(names[0], r"^session-\d+\.jsonl$")

    def test_plain_text_early_crash_synthesizes_fallback_assistant_event(self):
        crash_cli = self.root / "claude-crash"
        crash_cli.write_text(
            textwrap.dedent("""\
                #!/usr/bin/env bash
                echo "Error: Failed to initialize Vertex AI credentials" >&2
                exit 1
                """),
            encoding="utf-8",
        )
        crash_cli.chmod(0o755)

        done = subprocess.run(
            ["bash", str(_SUPERVISOR), "--print"],
            cwd=self.workspace,
            env={
                **self.env,
                "CLAUDE_REAL_BIN": str(crash_cli),
                "AGENT_PERSONA": "tpu-agent",
            },
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(done.returncode, 1)

        own = self.projects / _slug(self.workspace)
        files = list(own.glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        self.assertGreater(files[0].stat().st_size, 0)
        record = json.loads(
            files[0].read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(record["type"], "assistant")
        text = record["message"]["content"][0]["text"]
        self.assertIn("[Depth 0 | tpu-agent]", text)
        self.assertIn("Failed to initialize Vertex AI credentials", text)

    def test_error_result_without_assistant_message_synthesizes_fallback_step(
            self):
        err_cli = self.root / "claude-err-json"
        err_cli.write_text(
            textwrap.dedent(f"""\
                #!/usr/bin/env bash
                printf '{{"type":"system","session_id":"{_SESSION}"}}\\n'
                printf '{{"type":"result","is_error":true,"result":"","error":{{"message":"API Error: 401 Unauthorized"}}}}\\n'
                exit 1
                """),
            encoding="utf-8",
        )
        err_cli.chmod(0o755)

        done = subprocess.run(
            ["bash", str(_SUPERVISOR), "--print"],
            cwd=self.workspace,
            env={
                **self.env,
                "CLAUDE_REAL_BIN": str(err_cli),
            },
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(done.returncode, 1)

        synthesized = (self.projects / _slug(self.workspace) /
                       f"{_SESSION}.jsonl")
        records = [
            json.loads(line)
            for line in synthesized.read_text(encoding="utf-8").splitlines()
        ]
        assistant_records = [
            r for r in records if r.get("type") == "assistant"
            and isinstance(r.get("message"), dict)
        ]
        self.assertEqual(len(assistant_records), 1)
        fallback_text = assistant_records[0]["message"]["content"][0]["text"]
        self.assertIn("API Error: 401 Unauthorized", fallback_text)
        self.assertNotIn('"type":"system"', fallback_text)

    def test_sigterm_interrupts_running_cli_and_synthesizes_session(self):
        import signal

        slow_cli = self.root / "claude-slow"
        slow_cli.write_text(
            textwrap.dedent(f"""\
                #!/usr/bin/env bash
                printf '{{"type":"system","session_id":"{_SESSION}"}}\\n'
                printf '{{"type":"assistant","message":{{"role":"assistant","content":"working"}}}}\\n'
                sleep 30
                """),
            encoding="utf-8",
        )
        slow_cli.chmod(0o755)

        proc = subprocess.Popen(
            ["bash", str(_SUPERVISOR), "--print"],
            cwd=self.workspace,
            env={
                **self.env,
                "CLAUDE_REAL_BIN": str(slow_cli),
                "AGENT_PERSONA": "tpu-agent",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert proc.stdout is not None
        proc.stdout.readline()
        proc.stdout.readline()
        proc.send_signal(signal.SIGTERM)
        _, stderr = proc.communicate(timeout=10)

        synthesized = (self.projects / _slug(self.workspace) /
                       f"{_SESSION}.jsonl")
        self.assertTrue(synthesized.is_file())
        lines = synthesized.read_text(encoding="utf-8").splitlines()
        self.assertTrue(
            any("[Depth 0 | tpu-agent] working" in ln for ln in lines))
        self.assertEqual(stderr.count("Session synthesis complete:"), 1)

    def test_task_agent_persona_overrides_agent_persona(self):
        saved = self.projects / _slug(self.workspace) / f"{_SESSION}.jsonl"
        self._run(
            ["--print"],
            self.workspace,
            AGENT_PERSONA="tpu-agent",
            TASK_AGENT_PERSONA="tpu-model-onboarding",
            FAKE_SESSION=_SESSION,
            FAKE_SAVE=str(saved),
        )
        args_text = self.args_log.read_text(encoding="utf-8")
        self.assertIn("--agent tpu-model-onboarding", args_text)
        self.assertNotIn("--agent tpu-agent", args_text)
        record = json.loads(saved.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(
            record["message"]["content"][0]["text"],
            "[Depth 0 | tpu-model-onboarding] saved",
        )


if __name__ == "__main__":
    unittest.main()
