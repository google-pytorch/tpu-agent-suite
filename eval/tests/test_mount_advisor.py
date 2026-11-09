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
"""Tests for SkillMountAdvisor (eval/mount_advisor.py)."""

import tempfile
import unittest
from pathlib import Path

from eval.mount_advisor import SkillMountAdvisor


class TestSkillMountAdvisor(unittest.TestCase):
    """Test suite for SkillMountAdvisor harness-aware mounts and injections."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.staged_ws = Path(self.tmp_dir.name) / "workspace"
        self.staged_ws.mkdir(parents=True)
        self.central_dir = Path(
            self.tmp_dir.name) / "central" / "tpu-agent-suite@0.1.0"
        self.central_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_claude_mounts_and_env(self) -> None:
        """Verify mounts and environment variable synthesis for Claude Code."""
        # Create mock staged files
        (self.staged_ws / ".claude").mkdir()
        (self.staged_ws / "CLAUDE.md").write_text("# Claude instructions")
        (self.staged_ws / ".tpu-agent-suite").mkdir()

        advisor = SkillMountAdvisor(
            staged_workspace=self.staged_ws,
            harness="claude-code",
            central_pkg_dir=self.central_dir,
        )

        mounts = advisor.get_mounts()
        mount_targets = {m["target"]: m for m in mounts}

        # Check Claude targets
        self.assertIn("/workspace/.claude", mount_targets)
        self.assertNotIn("/root/.claude", mount_targets)
        self.assertIn("/workspace/CLAUDE.md", mount_targets)
        self.assertIn("/workspace/.tpu-agent-suite", mount_targets)
        self.assertIn("/usr/local/bin/claude_session_supervisor_hook.sh",
                      mount_targets)
        self.assertIn("/usr/local/bin/claude_session_supervisor.sh",
                      mount_targets)
        self.assertIn("/root/.tpu-agent-suite/packages/tpu-agent-suite@0.1.0",
                      mount_targets)

        # Check read-only vs writable mounts
        writable_targets = {"/workspace/.tpu-agent-suite"}
        for m in mounts:
            self.assertEqual(m["type"], "bind")
            if m["target"] in writable_targets:
                self.assertFalse(m.get("read_only", False),
                                 f"{m['target']} should be writable")
                self.assertNotIn(
                    "read_only", m,
                    f"{m['target']} must omit read_only to conform to Harbor schema"
                )
            else:
                self.assertTrue(m.get("read_only", False),
                                f"{m['target']} should be read_only")

        # Check container env
        env = advisor.get_container_env()
        self.assertEqual(env.get("BASH_ENV"),
                         "/usr/local/bin/claude_session_supervisor_hook.sh")
        self.assertEqual(env.get("AGENT_PERSONA"), "tpu-agent")

    def test_the_central_environment_is_mounted_into_the_store(self) -> None:
        """Staging's central environment.json reaches the container's store.

        It sits at the store's root, beside packages/, so a command run
        outside /workspace resolves it as a user-scope install would.
        """

        def targets() -> dict:
            return {
                m["target"]: m
                for m in SkillMountAdvisor(
                    staged_workspace=self.staged_ws,
                    harness="claude-code",
                    central_pkg_dir=self.central_dir,
                ).get_mounts()
            }

        target = "/root/.tpu-agent-suite/environment.json"
        self.assertNotIn(target, targets())
        central_env = self.central_dir.parent.parent / "environment.json"
        central_env.write_text("{}", encoding="utf-8")
        mount = targets()[target]
        self.assertEqual(mount["source"], str(central_env))
        self.assertTrue(mount["read_only"])

    def test_antigravity_mounts_and_env(self) -> None:
        """Verify mounts and environment variable synthesis for Antigravity."""
        # Create mock staged files
        (self.staged_ws / ".agents").mkdir()
        (self.staged_ws / "AGENTS.md").write_text("# Agy instructions")
        (self.staged_ws / ".tpu-agent-suite").mkdir()

        advisor = SkillMountAdvisor(
            staged_workspace=self.staged_ws,
            harness="antigravity",
            central_pkg_dir=self.central_dir,
        )

        mounts = advisor.get_mounts()
        mount_targets = {m["target"]: m for m in mounts}

        # Check Antigravity targets
        self.assertIn("/workspace/.agents", mount_targets)
        self.assertIn("/root/.agents", mount_targets)
        self.assertIn("/workspace/AGENTS.md", mount_targets)
        self.assertIn("/workspace/.tpu-agent-suite", mount_targets)
        self.assertIn("/usr/local/bin/session_supervisor_hook.sh",
                      mount_targets)
        self.assertIn("/usr/local/bin/session_supervisor.sh", mount_targets)

        # Check read-only vs writable mounts
        writable_targets = {"/workspace/.tpu-agent-suite"}
        for m in mounts:
            self.assertEqual(m["type"], "bind")
            if m["target"] in writable_targets:
                self.assertFalse(m.get("read_only", False),
                                 f"{m['target']} should be writable")
                self.assertNotIn(
                    "read_only", m,
                    f"{m['target']} must omit read_only to conform to Harbor schema"
                )
            else:
                self.assertTrue(m.get("read_only", False),
                                f"{m['target']} should be read_only")

        # Check container env
        env = advisor.get_container_env()
        self.assertEqual(env.get("BASH_ENV"),
                         "/usr/local/bin/session_supervisor_hook.sh")
        self.assertEqual(env.get("AGENT_PERSONA"), "tpu-agent")
        # The CLI is pinned; a self-update would swap the binary mid-trial.
        self.assertEqual(env.get("AGY_CLI_DISABLE_AUTO_UPDATE"), "true")

    def test_empty_arm_mounts_supervisor_by_default(self) -> None:
        """Verify empty/vanilla arm mounts supervisor hooks by default without mounting package files."""
        # Create mock package files in workspace
        (self.staged_ws / ".claude").mkdir(exist_ok=True)
        (self.staged_ws / "CLAUDE.md").write_text("# Claude")

        advisor = SkillMountAdvisor(
            staged_workspace=self.staged_ws,
            harness="claude-code",
            central_pkg_dir=self.central_dir,
            is_empty=True,
        )

        mounts = advisor.get_mounts()
        mount_targets = {m["target"]: m for m in mounts}

        # Package files must NOT be mounted
        self.assertNotIn("/workspace/.claude", mount_targets)
        self.assertNotIn("/workspace/CLAUDE.md", mount_targets)
        self.assertNotIn(
            "/root/.tpu-agent-suite/packages/tpu-agent-suite@0.1.0",
            mount_targets)

        # Supervisor hooks MUST be mounted by default
        self.assertIn("/usr/local/bin/claude_session_supervisor_hook.sh",
                      mount_targets)
        self.assertIn("/usr/local/bin/claude_session_supervisor.sh",
                      mount_targets)

        # BASH_ENV must be configured
        env = advisor.get_container_env()
        self.assertEqual(env.get("BASH_ENV"),
                         "/usr/local/bin/claude_session_supervisor_hook.sh")
        # A control arm has no package, so no persona to run as.
        self.assertNotIn("AGENT_PERSONA", env)

    def test_get_updates_returns_mount_updates(self) -> None:
        """Verify get_updates returns MountUpdates dataclass with mounts and env."""
        advisor = SkillMountAdvisor(
            staged_workspace=self.staged_ws,
            harness="claude-code",
            central_pkg_dir=self.central_dir,
        )
        updates = advisor.get_updates()
        self.assertIsInstance(updates.mounts, list)
        self.assertIsInstance(updates.env, dict)
        self.assertEqual(updates.mounts, advisor.get_mounts())
        self.assertEqual(updates.env, advisor.get_container_env())

    def _user_scope(self, harness: str) -> SkillMountAdvisor:
        for name in (".claude", ".agents", ".tpu-agent-suite"):
            (self.staged_ws / name).mkdir(exist_ok=True)
        (self.staged_ws / "CLAUDE.md").write_text("# project memory")
        (self.staged_ws / "AGENTS.md").write_text("# project memory")
        dist = Path(self.tmp_dir.name) / "dist"
        dist.mkdir(exist_ok=True)
        answers = Path(self.tmp_dir.name) / "answers.json"
        answers.write_text("{}")
        return SkillMountAdvisor(
            staged_workspace=self.staged_ws,
            harness=harness,
            central_pkg_dir=self.central_dir,
            user_scope=True,
            staged_dist=dist,
            answers_file=answers,
        )

    def test_user_scope_mounts_what_the_installer_reads(self) -> None:
        """A user-scope arm installs in the trial: it gets the distribution
        and its answers, never the result of a host-side install."""
        for harness in ("claude-code", "antigravity", "codex"):
            with self.subTest(harness=harness):
                mounts = self._user_scope(harness).get_mounts()
                targets = {m["target"]: m for m in mounts}
                self.assertTrue(
                    targets["/opt/tpu-agent-suite/dist"]["read_only"])
                self.assertTrue(
                    targets["/opt/tpu-agent-suite/answers.json"]["read_only"])
                for staged in (
                        "/workspace/.claude",
                        "/workspace/CLAUDE.md",
                        "/workspace/.agents",
                        "/workspace/AGENTS.md",
                        "/root/.agents",
                        "/workspace/.tpu-agent-suite",
                        "/root/.tpu-agent-suite/packages/tpu-agent-suite@0.1.0",
                ):
                    self.assertNotIn(staged, targets)

    def test_user_scope_keeps_the_session_supervisor(self) -> None:
        targets = {
            m["target"]
            for m in self._user_scope("claude-code").get_mounts()
        }
        self.assertIn("/usr/local/bin/claude_session_supervisor_hook.sh",
                      targets)

    def test_user_scope_without_a_distribution_fails_loudly(self) -> None:
        advisor = SkillMountAdvisor(
            staged_workspace=self.staged_ws,
            harness="claude-code",
            user_scope=True,
            staged_dist=Path(self.tmp_dir.name) / "missing",
            answers_file=Path(self.tmp_dir.name) / "missing.json",
        )
        with self.assertRaises(FileNotFoundError):
            advisor.get_mounts()


if __name__ == "__main__":
    unittest.main()
