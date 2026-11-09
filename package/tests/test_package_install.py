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
"""End-to-end integration tests for tpu-agent-suite package installation across all harnesses.

Verifies the full lifecycle (translate -> install -> filesystem verification -> uninstall)
of the production tpu-agent-suite package for every supported coding harness:
Antigravity, Claude Code, Codex, and Cursor.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import tempfile
import unittest

sys_path_root = pathlib.Path(__file__).resolve().parents[2]
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from framework import installer, jsonio  # noqa: E402
from framework.builder import resolve_manifest_composition  # noqa: E402
from framework.builder import translate  # noqa: E402

Path = pathlib.Path
PACKAGE_DIR = Path(__file__).resolve().parents[1]


class TestPackageMultiHarnessInstallIntegration(unittest.TestCase):
    """End-to-end integration tests verifying installation of tpu-agent-suite."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.temp_dir = Path(
            tempfile.mkdtemp(prefix="torchtpu_install_integration_"))
        cls.tpu_agent_suite_home = cls.temp_dir / "central_tpu_agent_suite"
        cls.prev_tpu_agent_suite_home = os.environ.get("TPU_AGENT_SUITE_HOME")
        os.environ["TPU_AGENT_SUITE_HOME"] = str(cls.tpu_agent_suite_home)

        cls.manifest = resolve_manifest_composition(
            PACKAGE_DIR,
            jsonio.load_json(PACKAGE_DIR / "package.release.json"))
        cls.entry_name = cls.manifest["skills"][0]["name"]
        cls.pkg_name = cls.manifest["name"]

    @classmethod
    def tearDownClass(cls):
        if cls.prev_tpu_agent_suite_home is not None:
            os.environ["TPU_AGENT_SUITE_HOME"] = cls.prev_tpu_agent_suite_home
        else:
            os.environ.pop("TPU_AGENT_SUITE_HOME", None)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
        super().tearDownClass()

    def _create_workspace(self, name: str) -> Path:
        ws = self.temp_dir / name
        ws.mkdir(parents=True, exist_ok=True)
        (ws / ".git").mkdir(exist_ok=True)
        return ws

    def _assert_clean_uninstall(self, workspace: Path) -> None:
        """Verify that uninstall leaves no orphan files in the workspace outside .git."""
        remaining = [
            str(p.relative_to(workspace)) for p in workspace.rglob("*")
            if p.is_file()
            and not str(p.relative_to(workspace)).startswith(".git")
        ]
        self.assertEqual(remaining, [],
                         f"Leftover files after uninstall: {remaining}")

    def test_antigravity_installation_lifecycle(self):
        """Antigravity installs plugin.json, skills, agents, hooks, and memory block."""
        dist_dir = self.temp_dir / "dist_antigravity"
        workspace = self._create_workspace("ws_antigravity")

        translate(PACKAGE_DIR, "antigravity", dist_dir)
        installer.install(
            dist_dir,
            destination=workspace,
            non_interactive=True,
            skip_harness_check=True,
            host_check=False,
        )

        plugin_json = workspace / ".agents" / "plugin.json"
        self.assertTrue(plugin_json.is_file(), "Missing .agents/plugin.json")
        plugin_meta = jsonio.load_json(plugin_json)
        self.assertEqual(plugin_meta["name"], self.pkg_name)

        skill_md = workspace / ".agents" / "skills" / self.entry_name / "SKILL.md"
        self.assertTrue(skill_md.is_file(), "Missing tpu-agent skill")

        agents_dir = workspace / ".agents" / "agents"
        self.assertTrue(agents_dir.is_dir(),
                        "Missing .agents/agents directory")
        for agent in self.manifest["agents"]:
            self.assertTrue(
                (agents_dir / f"{agent['name']}.md").is_file(),
                f"Missing Antigravity agent file for {agent['name']}",
            )

        self.assertTrue((workspace / ".agents" / "hooks.json").is_file(),
                        "Missing .agents/hooks.json")
        self.assertTrue((workspace / "AGENTS.md").is_file(),
                        "Missing AGENTS.md memory file")

        installer.uninstall(package=self.pkg_name, destination=workspace)
        self._assert_clean_uninstall(workspace)

    def test_claude_code_installation_lifecycle(self):
        """Claude Code installs skills, agents, hooks, settings, and memory block."""
        dist_dir = self.temp_dir / "dist_claude"
        workspace = self._create_workspace("ws_claude")

        translate(PACKAGE_DIR, "claude-code", dist_dir)
        installer.install(
            dist_dir,
            destination=workspace,
            non_interactive=True,
            skip_harness_check=True,
            host_check=False,
        )

        skill_md = workspace / ".claude" / "skills" / self.entry_name / "SKILL.md"
        self.assertTrue(skill_md.is_file(),
                        "Missing skill under .claude/skills")

        agents_dir = workspace / ".claude" / "agents"
        self.assertTrue(agents_dir.is_dir(),
                        "Missing .claude/agents directory")
        for agent in self.manifest["agents"]:
            self.assertTrue(
                (agents_dir / f"{agent['name']}.md").is_file(),
                f"Missing Claude Code agent file for {agent['name']}",
            )

        self.assertTrue((workspace / ".claude" / "hooks.json").is_file(),
                        "Missing .claude/hooks.json")
        settings_file = workspace / ".claude" / "settings.json"
        self.assertTrue(settings_file.is_file(),
                        "Missing .claude/settings.json")
        settings = jsonio.load_json(settings_file)
        self.assertIn("approved_directories", settings)

        self.assertTrue((workspace / "CLAUDE.md").is_file(),
                        "Missing CLAUDE.md memory file")

        installer.uninstall(package=self.pkg_name, destination=workspace)
        self._assert_clean_uninstall(workspace)

    def test_codex_installation_lifecycle(self):
        """Codex installs skills into .agents, agents into .codex, hooks, config.toml, and memory block."""
        dist_dir = self.temp_dir / "dist_codex"
        workspace = self._create_workspace("ws_codex")

        translate(PACKAGE_DIR, "codex", dist_dir)
        installer.install(
            dist_dir,
            destination=workspace,
            non_interactive=True,
            skip_harness_check=True,
            host_check=False,
        )

        skill_md = workspace / ".agents" / "skills" / self.entry_name / "SKILL.md"
        self.assertTrue(skill_md.is_file(),
                        "Missing skill under .agents/skills")

        agents_dir = workspace / ".codex" / "agents"
        self.assertTrue(agents_dir.is_dir(), "Missing .codex/agents directory")
        for agent in self.manifest["agents"]:
            self.assertTrue(
                (agents_dir / f"{agent['name']}.toml").is_file(),
                f"Missing Codex agent config for {agent['name']}",
            )

        self.assertTrue((workspace / ".codex" / "hooks.json").is_file(),
                        "Missing .codex/hooks.json")
        config_toml = workspace / ".codex" / "config.toml"
        self.assertTrue(config_toml.is_file(), "Missing .codex/config.toml")
        content = config_toml.read_text(encoding="utf-8")
        self.assertIn("[agents]", content)
        self.assertIn("[sandbox]", content)

        self.assertTrue((workspace / "AGENTS.md").is_file(),
                        "Missing AGENTS.md memory file")

        installer.uninstall(package=self.pkg_name, destination=workspace)
        self._assert_clean_uninstall(workspace)

    def test_cursor_installation_lifecycle(self):
        """Cursor installs skills, rules (.mdc), and hooks."""
        dist_dir = self.temp_dir / "dist_cursor"
        workspace = self._create_workspace("ws_cursor")

        translate(PACKAGE_DIR, "cursor", dist_dir)
        installer.install(
            dist_dir,
            destination=workspace,
            non_interactive=True,
            skip_harness_check=True,
            host_check=False,
        )

        skill_md = workspace / ".cursor" / "skills" / self.entry_name / "SKILL.md"
        self.assertTrue(skill_md.is_file(), "Missing Cursor skill")

        rule_file = workspace / ".cursor" / "rules" / f"{self.pkg_name}.mdc"
        self.assertTrue(rule_file.is_file(), "Missing Cursor .mdc rule file")

        self.assertTrue((workspace / ".cursor" / "hooks.json").is_file(),
                        "Missing .cursor/hooks.json")

        installer.uninstall(package=self.pkg_name, destination=workspace)
        self._assert_clean_uninstall(workspace)


class TestRuntimeHostRequirements(unittest.TestCase):
    """The tpu-agent skill declares host-side dependencies for the runtime system
    itself, and the tpu-agent-export-trajectory skill declares google-cloud-storage,
    so this test checks that the compiled release includes them."""

    def _host_requirements(self, skills=None) -> str:
        temp_dir = Path(tempfile.mkdtemp(prefix="torchtpu_runtime_reqs_"))
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        dist_dir = temp_dir / "dist"
        translate(PACKAGE_DIR, "claude-code", dist_dir, skills=skills)
        return (dist_dir / "requirements-client.txt").read_text()

    def test_the_release_ships_what_the_runtime_imports(self):
        host = self._host_requirements()
        self.assertIn("google-cloud-storage", host)
        self.assertIn("tpu-info", host)

    def test_the_export_skill_alone_ships_what_its_upload_imports(self):
        """The trajectory export uploads through google-cloud-storage. Built
        without tpu-agent, the export skill's own declaration ships it."""
        host = self._host_requirements(skills=["tpu-agent-export-trajectory"])
        self.assertIn("google-cloud-storage", host)
        self.assertNotIn("pydantic", host)


if __name__ == "__main__":
    unittest.main()
