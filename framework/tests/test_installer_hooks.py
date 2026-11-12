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
"""Unit tests for installer hook registration.

Also covers the harness destinations matrix, workspace memory, and the host
check.
"""

import contextlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

# Imports below follow this sys.path setup, so the file runs directly.
# ruff: noqa: E402
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import installer, jsonio, utils
from framework.builder import translators
from framework.tests.test_installer import (_InstallerTestCase,
                                            _TranslatedInstallerTestCase)

Path = pathlib.Path


class TestInstallerHooks(_TranslatedInstallerTestCase):
    """Hook entries are installed, upgraded, rolled back and uninstalled."""

    def _create_pkg_with_hooks(self) -> Path:
        """A copy of the test package, renamed, carrying two hooks."""
        pkg_dir = self.temp_dir / "pkg_hooks"
        shutil.copytree(self.pkg_dir, pkg_dir)
        (pkg_dir / "scripts").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "scripts" / "custom_guard.py").write_text(
            "print('guarded')", encoding="utf-8")
        self._update_manifest(pkg_dir,
                              name="pkg-hooks",
                              description="Package with hooks",
                              hooks=[{
                                  "name": "cmd-guard",
                                  "event": "PreToolUse",
                                  "matchers": ["read"],
                                  "command": "python3 check_input.py",
                                  "args": ["--strict"],
                                  "timeout": 5,
                                  "visibility": "external",
                              }, {
                                  "name": "custom-guard",
                                  "event": "PreToolUse",
                                  "matchers": ["read"],
                                  "action": "script:scripts/custom_guard.py",
                                  "timeout": 10,
                                  "visibility": "external",
                              }],
                              compatibility_targets=[
                                  {
                                      "name": "claude",
                                      "harness": "claude-code",
                                      "versions": ">=2.0.0",
                                  },
                                  {
                                      "name": "antigravity",
                                      "harness": "antigravity",
                                      "versions": ">=1.0.0",
                                  },
                                  {
                                      "name": "codex",
                                      "harness": "codex",
                                      "versions": ">=0.145.0",
                                  },
                              ])
        return pkg_dir

    def test_install_and_uninstall_with_hooks(self):
        # The hook commands themselves are pinned at compile time in
        # builder/test_translators; this covers what the installer does with
        # them.
        pkg_dir = self._create_pkg_with_hooks()
        dist = self.temp_dir / "dist_hooks"
        translators.translate(pkg_dir, "claude-code", dist)

        workspace = self._workspace("workspace_hooks")
        self._install(workspace, distribution=dist)

        hooks_file = workspace / ".claude" / "hooks.json"
        self.assertEqual(
            json.loads(hooks_file.read_text(encoding="utf-8")),
            json.loads(
                (dist / "project" / "hooks.json").read_text(encoding="utf-8")))
        central_script = (self.central_home / "packages" / "pkg-hooks@1.0.0" /
                          "scripts" / "custom_guard.py")
        self.assertTrue(central_script.is_file())

        # Verify ownership tracking
        ownership = json.loads((workspace / ".tpu-agent-suite" /
                                "ownership.json").read_text(encoding="utf-8"))
        tracked_dests = [
            entry["path"] for entry in ownership.get("actions", [])
        ]
        self.assertIn(".claude/hooks.json", tracked_dests)
        # Central payload is governed by the reference ledger, not per-file
        # ownership actions.
        self.assertNotIn(
            ".tpu-agent-suite/packages/pkg-hooks@1.0.0/scripts/custom_guard.py",
            tracked_dests)
        self.assertEqual(ownership.get("central_package"), "pkg-hooks@1.0.0")

        # 2. Test user modification preservation on uninstall
        hooks_file.write_text(json.dumps({"custom": "user-modified"}),
                              encoding="utf-8")
        modified = installer.uninstall(package="pkg-hooks",
                                       destination=workspace)
        self.assertIn(".claude/hooks.json", modified)
        self.assertTrue(
            hooks_file.is_file())  # Preserved because user modified it
        self.assertFalse(central_script.exists())  # GC'd: last reference gone

    def test_rollback_json_records_undoes_only_what_the_ledger_added(self):
        settings = ".claude/settings.json"

        def action(keys, *val, path=settings):
            kind = "json_value" if val else "json_container"
            return dict(kind=kind, keys=keys, path=path, value=val and val[0])

        document = dict(hooks=dict(PreToolUse=["pkg@1"],
                                   Notification=["mine"]),
                        env=dict(PKG_VAR="1", MY_VAR="edited"))
        ownership = dict(actions=[
            action(["hooks"]),
            action(["hooks", "PreToolUse"], ["pkg@1"]),
            action(["env"]),
            action(["env", "PKG_VAR"], "1"),
            action(["env", "MY_VAR"], "original"),
            action(["hooks", "Notification"], ["mine"], path="other.json"),
        ])
        installer.rollback_json_records(document, ownership, settings)
        # The user's Notification hook (not in this file's ledger) and the
        # edited MY_VAR stay; the package's entries go; `hooks` and `env`
        # stay because they are not empty.
        self.assertEqual(
            document,
            dict(hooks=dict(Notification=["mine"]), env=dict(MY_VAR="edited")))

    def test_harness_install_destinations_all_scopes(self):
        pkg_dir = self._create_pkg_with_hooks()

        expected = {
            "claude-code": {
                "project": ".claude",
                "user": ".claude",
            },
            "codex": {
                "project": ".codex",
                "user": ".codex",
            },
            "antigravity": {
                "project": ".agents",
                "user": ".gemini/config",
            },
        }

        for harness, scopes in expected.items():
            for scope, expected_dir in scopes.items():
                with self.subTest(harness=harness, scope=scope):
                    dist_dir = self.temp_dir / f"dist_{harness}_{scope}"
                    translators.translate(pkg_dir, harness, dist_dir)

                    target_dir = self.temp_dir / f"target_{harness}_{scope}"
                    target_dir.mkdir()

                    with mock.patch("framework.installer.Path.cwd", return_value=target_dir), \
                         mock.patch("framework.installer.Path.home", return_value=target_dir):

                        dest = target_dir if scope == "project" else None
                        self._install(dest, scope=scope, distribution=dist_dir)

                        hooks_json = target_dir / expected_dir / "hooks.json"
                        self.assertTrue(hooks_json.is_file(),
                                        f"Expected {hooks_json} to be a file")


class TestHostCheck(unittest.TestCase):
    """The software check install runs on this machine."""

    def setUp(self):
        super().setUp()
        self.dist = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dist)
        (self.dist / installer.ENV_CHECK).parent.mkdir(parents=True)
        for name in (installer.ENV_CHECK, installer.HOST_REQUIREMENTS):
            (self.dist / name).write_text("", encoding="utf-8")

    def test_the_host_check_runs_under_python3_and_never_blocks(self):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("shutil.which", return_value="/usr/bin/python3"), \
                mock.patch("subprocess.run",
                           return_value=subprocess.CompletedProcess(
                               [], 1, "[FAIL] pyyaml", "")) as run, \
                contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            installer.check_host(self.dist)
        self.assertEqual(run.call_args.args[0], [
            "/usr/bin/python3",
            str(self.dist / installer.ENV_CHECK), "--host", "--requirements",
            str(self.dist / "requirements-client.txt")
        ])
        self.assertIn("[FAIL] pyyaml", out.getvalue())
        self.assertIn("WARNING:", err.getvalue())
        self.assertIn("the install goes on", err.getvalue())


class TestWorkspaceMemory(_InstallerTestCase):
    """Tests for managed blocks installed into the workspace memory file."""

    MARKER = "tpu-agent-suite:pkg:kb"

    def setUp(self):
        super().setUp()
        manifest = jsonio.load_json(self.pkg_dir / "package.release.json")
        self._update_manifest(
            self.pkg_dir,
            workspace_memory=[{
                "marker": "kb",
                "content": "memory/contract.md",
            }],
            compatibility_targets=manifest["compatibility_targets"] + [
                {
                    "name": "antigravity",
                    "harness": "antigravity",
                    "versions": ">=1.0.0",
                },
                {
                    "name": "codex",
                    "harness": "codex",
                    "versions": ">=0.145.0",
                },
            ])
        (self.pkg_dir / "memory").mkdir(parents=True, exist_ok=True)
        (self.pkg_dir / "memory" / "contract.md").write_text(
            "Schemas live at {{schema:in.json}}.", encoding="utf-8")

        self.dist_dir = self.temp_dir / "dist_memory"
        translators.translate(self.pkg_dir, "claude-code", self.dist_dir)

    def test_user_scope_writes_where_each_harness_reads_user_memory(self):
        """~/.claude/CLAUDE.md, ~/.codex/AGENTS.md and ~/.gemini/AGENTS.md.

        Not the bare file in ~, which only a session started in $HOME itself
        would read.
        """
        for harness, memory, bare in (
            ("claude-code", ".claude/CLAUDE.md", "CLAUDE.md"),
            ("codex", ".codex/AGENTS.md", "AGENTS.md"),
            ("antigravity", ".gemini/AGENTS.md", "AGENTS.md"),
        ):
            with self.subTest(harness=harness):
                dist_dir = self.temp_dir / f"dist_memory_{harness}"
                translators.translate(self.pkg_dir, harness, dist_dir)
                home = self._workspace(f"home_{harness}")
                with mock.patch.dict(os.environ, {"HOME": str(home)}):
                    self._install(None, distribution=dist_dir, scope="user")
                self.assertIn(f"<!-- {self.MARKER}:begin -->",
                              (home / memory).read_text(encoding="utf-8"))
                self.assertFalse((home / bare).exists())

    def test_reinstall_is_idempotent_and_uninstall_removes_the_created_file(
            self):
        """Re-installing replaces the block instead of appending a second one.

        A memory file the install created goes when nothing else remains.
        """
        workspace = self._workspace("ws_mem_idem")
        self._install(workspace)
        first = (workspace / "CLAUDE.md").read_text(encoding="utf-8")
        self._install(workspace, upgrade=True)
        second = (workspace / "CLAUDE.md").read_text(encoding="utf-8")

        self.assertEqual(first, second)
        self.assertEqual(second.count(f"<!-- {self.MARKER}:begin -->"), 1)

        installer.uninstall(destination=workspace, package="pkg")
        self.assertFalse((workspace / "CLAUDE.md").exists())

    def test_refuses_unmanaged_existing_block(self):
        """An unowned block with the same marker blocks the install."""
        workspace = self._workspace("ws_mem_unmanaged")
        (workspace / "CLAUDE.md").write_text(
            f"<!-- {self.MARKER}:begin -->\nstale\n<!-- {self.MARKER}:end -->\n",
            encoding="utf-8",
        )

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            self._install(workspace)
        self.assertIn("unmanaged block", str(ctx.exception))

    def test_local_block_edit_is_refused_unless_forced(self):
        """A hand-edited block is modified managed content; --force replaces it.

        Uninstall never discards the user's own text or edits.
        """
        workspace = self._workspace("ws_mem_edited")
        memory = workspace / "CLAUDE.md"
        memory.write_text("# Project\n\nMy notes.\n", encoding="utf-8")
        self._install(workspace)
        memory.write_text(
            memory.read_text(encoding="utf-8").replace("Schemas live",
                                                       "Edited"),
            encoding="utf-8",
        )

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            self._install(workspace, upgrade=True)
        self.assertIn("managed content changed", str(ctx.exception))

        self._install(workspace, upgrade=True, force=True)
        self.assertIn("Schemas live", memory.read_text(encoding="utf-8"))

        # Uninstall strips an unedited block but leaves the user's text.
        installer.uninstall(destination=workspace, package="pkg")
        remaining = memory.read_text(encoding="utf-8")
        self.assertIn("My notes.", remaining)
        self.assertNotIn(self.MARKER, remaining)
        # A block edited by hand is left in place rather than discarded.
        self._install(workspace)
        memory.write_text(
            memory.read_text(encoding="utf-8").replace("Schemas live",
                                                       "Edited"),
            encoding="utf-8",
        )
        installer.uninstall(destination=workspace, package="pkg")
        self.assertIn("Edited", memory.read_text(encoding="utf-8"))

    def _dist(self, harness: str) -> Path:
        dist = self.temp_dir / f"dist_user_{harness}"
        if not dist.exists():
            translators.translate(self.pkg_dir, harness, dist)
        return dist

    def _install_user(self, harness: str, home: Path, **env: str) -> None:
        with mock.patch.dict(os.environ, {"HOME": str(home), **env}):
            installer.install(
                distribution=self._dist(harness),
                scope="user",
                non_interactive=True,
                skip_harness_check=True,
            )

    def _assert_one_block(self, memory: Path) -> None:
        content = memory.read_text(encoding="utf-8")
        self.assertEqual(content.count(f"<!-- {self.MARKER}:begin -->"), 1)

    def test_user_scope_codex_honours_codex_home(self):
        """Memory, agents and config all follow a relocated home."""
        home = self._workspace("home_codex_relocated")
        self._install_user("codex",
                           home,
                           CODEX_HOME=str(home / ".config" / "codex"))
        codex_home = home / ".config" / "codex"
        self._assert_one_block(codex_home / "AGENTS.md")
        self.assertTrue((codex_home / "config.toml").is_file())
        self.assertTrue((codex_home / "agents" / "helper.toml").is_file())
        self.assertFalse((home / ".codex").exists())
        self.assertFalse((home / "AGENTS.md").exists())

    def test_user_scope_codex_home_outside_home_is_refused(self):
        """A CODEX_HOME the ledger cannot root fails the install.

        It does not succeed into ~/.codex, which that Codex never reads.
        """
        home = self._workspace("home_codex_outside")
        outside = self._workspace("codex_home_outside")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            self._install_user("codex", home, CODEX_HOME=str(outside))
        self.assertIn("Unset CODEX_HOME", str(ctx.exception))
        self.assertEqual(list(outside.iterdir()), [])
        self.assertFalse((home / ".codex").exists())

    def test_user_scope_codex_home_equal_to_home_is_refused(self):
        """$HOME itself would put Codex's files at the top of the home."""
        home = self._workspace("home_codex_is_home")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            self._install_user("codex", home, CODEX_HOME=str(home))
        self.assertIn("Unset CODEX_HOME", str(ctx.exception))
        self.assertFalse((home / "AGENTS.md").exists())
        self.assertFalse((home / "config.toml").exists())
        self.assertFalse((home / "agents").exists())

    def test_refused_codex_home_leaves_the_previous_install(self):
        """--clean removes nothing when the new CODEX_HOME is refused."""
        home = self._workspace("home_codex_keep")
        self._install_user("codex", home)
        with self.assertRaises(utils.TpuAgentSuiteError):
            with mock.patch.dict(
                    os.environ, {
                        "HOME": str(home),
                        "CODEX_HOME": str(self._workspace("codex_elsewhere")),
                    }):
                installer.install(distribution=self._dist("codex"),
                                  scope="user",
                                  non_interactive=True,
                                  skip_harness_check=True,
                                  clean=True)
        self._assert_one_block(home / ".codex" / "AGENTS.md")

    def test_upgrade_moves_block_from_home_agents_md(self):
        """An install made before the fix recorded its block at ~/AGENTS.md.

        The upgrade strips it there and writes it once where it is read.
        """
        for harness, memory in (
            ("codex", Path(".codex/AGENTS.md")),
            ("antigravity", Path(".gemini/AGENTS.md")),
        ):
            with self.subTest(harness=harness):
                home = self._workspace(f"home_upgrade_{harness}")
                stray = home / "AGENTS.md"
                stray.write_text("# My notes\n", encoding="utf-8")
                with mock.patch.dict(installer.USER_SCOPE_MEMORY_PATHS,
                                     clear=True):
                    self._install_user(harness, home)
                self._assert_one_block(stray)

                self._install_user(harness, home)
                self.assertEqual(stray.read_text(encoding="utf-8"),
                                 "# My notes\n")
                self._assert_one_block(home / memory)

    def test_user_scope_uninstall_strips_memory(self):
        """Uninstall removes the block from wherever the install put it."""
        for harness, memory in (
            ("claude-code", Path(".claude/CLAUDE.md")),
            ("codex", Path(".codex/AGENTS.md")),
            ("antigravity", Path(".gemini/AGENTS.md")),
        ):
            with self.subTest(harness=harness):
                home = self._workspace(f"home_uninstall_{harness}")
                self._install_user(harness, home)
                self.assertTrue((home / memory).is_file())
                with mock.patch.dict(os.environ, {"HOME": str(home)}):
                    installer.uninstall(scope="user", package="pkg")
                self.assertFalse((home / memory).exists())

    def test_project_scope_memory_stays_in_workspace(self):
        """Project scope keeps the memory file at the workspace root."""
        for harness, memory in (
            ("codex", "AGENTS.md"),
            ("antigravity", "AGENTS.md"),
        ):
            with self.subTest(harness=harness):
                workspace = self._workspace(f"ws_project_{harness}")
                self._install(workspace, distribution=self._dist(harness))
                self._assert_one_block(workspace / memory)


if __name__ == "__main__":
    unittest.main()
