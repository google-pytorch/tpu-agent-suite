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
"""Unit tests for installer module (install, upgrade, rollback, uninstall)."""

import contextlib
import functools
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any
from unittest import mock

from framework import installer, jsonio, paths, utils
from framework.builder import translators

Path = pathlib.Path


def _env_without_pythonpath() -> dict[str, str]:
    """This environment, minus PYTHONPATH: run as a user would, not as tests."""
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


class _InstallerTestCase(unittest.TestCase):
    """A scratch central store, the test package, and helpers to install it.

    Subclasses build `self.dist_dir`, the distribution `_install` defaults to.
    """

    def setUp(self):
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.central_home = self.temp_dir / "central_tpu_agent_suite"
        self.central_home.mkdir()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        env = mock.patch.dict(os.environ,
                              {"TPU_AGENT_SUITE_HOME": str(self.central_home)})
        env.start()
        self.addCleanup(env.stop)
        self.pkg_dir = self._create_test_package()

    def _install(self, workspace: Path | None, **kwargs) -> None:
        """Install headlessly into `workspace`, from `self.dist_dir` unless
        `distribution` is given."""
        kwargs.setdefault("distribution", self.dist_dir)
        installer.install(destination=workspace,
                          non_interactive=True,
                          skip_harness_check=True,
                          **kwargs)

    def _workspace(self, name: str) -> Path:
        workspace = self.temp_dir / name
        workspace.mkdir(exist_ok=True)
        return workspace

    @staticmethod
    def _update_manifest(pkg_dir: Path, **fields: Any) -> None:
        """Set top-level `fields` in the package manifest at `pkg_dir`."""
        path = pkg_dir / "package.release.json"
        jsonio.write_json(path, {**jsonio.load_json(path), **fields})

    def _create_test_package(self) -> Path:
        pkg_dir = self.temp_dir / "pkg"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "name":
            "pkg",
            "version":
            "1.0.0",
            "description":
            "Installer test package",
            "skills": ["entry"],
            "copy": ["schemas", "environment"],
            "environment": {
                "questions": "environment/questions.json",
                "schema": "schemas/env.json",
            },
            "compatibility_targets": [{
                "name": "claude",
                "harness": "claude-code",
                "versions": ">=2.0.0",
            }],
        }
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")
        (pkg_dir / "skills" / "entry").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "agents").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "skills" / "entry" / "SKILL.md").write_text(
            "---\n"
            "name: entry\n"
            "description: Entry description\n"
            "delegates:\n"
            "  - helper\n"
            "---\n"
            "Call {{agent:helper}}.",
            encoding="utf-8")
        (pkg_dir / "agents" / "helper.md").write_text(
            "---\n"
            "name: helper\n"
            "description: Helper\n"
            "tools:\n"
            "  - read\n"
            "delegates: []\n"
            "accepts: schemas/in.json\n"
            "produces: schemas/out.json\n"
            "---\n"
            "Help.",
            encoding="utf-8")
        (pkg_dir / "schemas").mkdir(parents=True, exist_ok=True)
        for name in ("in.json", "out.json"):
            (pkg_dir / "schemas" / name).write_text('{"type": "object"}',
                                                    encoding="utf-8")
        # tpu_target, plus what every install writes (as real schemas say).
        props = {
            "tpu_target": {
                "type": "string"
            },
            "revision": {
                "type": "integer"
            },
            "default": {
                "type": ["string", "null"]
            },
            "profiles": {
                "type": "object"
            },
        }
        (pkg_dir / "schemas" / "env.json").write_text(json.dumps({
            "type":
            "object",
            "properties":
            props
        }),
                                                      encoding="utf-8")
        (pkg_dir / "environment").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "environment" / "questions.json").write_text(
            json.dumps({"questions": []}), encoding="utf-8")
        return pkg_dir


class TestInstaller(_InstallerTestCase):
    """Unit tests for installer functions, pre-validations, collisions, and rollback."""

    def setUp(self):
        super().setUp()
        # The host check shells out to python3; its own test is in
        # TestHostCheck.
        host_check = mock.patch.object(installer, "check_host")
        self.check_host = host_check.start()
        self.addCleanup(host_check.stop)
        self.dist_dir = self.temp_dir / "dist"
        translators.translate(self.pkg_dir, "claude-code", self.dist_dir)

    def test_install_without_a_terminal_fails_fast_naming_headless_flags(self):
        """Without a TTY an interactive install would hang on (or die
        mid-prompt reading) stdin; it must refuse before doing anything.
        --force never asks, so it installs."""
        argv = [
            "install",
            str(self.temp_dir), "--destination",
            str(self.temp_dir)
        ]
        with mock.patch.object(installer.sys.stdin, "isatty",
                               return_value=False), \
                mock.patch.object(installer, "install") as install, \
                contextlib.redirect_stderr(io.StringIO()) as err:
            ret = installer.main(argv)
            self.assertEqual(ret, 1)
            install.assert_not_called()
            self.assertIn("--non-interactive", err.getvalue())
            installer.main(argv + ["--force", "--relocated-store"])
            install.assert_called_once()

    def test_install_refuses_a_relocated_store(self):
        """Hooks always run ~/.tpu-agent-suite/.../runtime.py,
        so a store TPU_AGENT_SUITE_HOME moved elsewhere breaks every hook.
        Refuse before writing anything."""
        workspace = self._workspace("workspace_relocated")
        argv = [
            "install",
            str(self.dist_dir), "--destination",
            str(workspace), "--non-interactive", "--skip-harness-check"
        ]
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(installer.main(argv), 1)
        self.assertIn("TPU_AGENT_SUITE_HOME", err.getvalue())
        self.assertEqual(list(workspace.iterdir()), [])
        self.assertEqual(list(self.central_home.iterdir()), [])

        # The store at its default place is not relocated, even when the
        # variable names it (as the eval containers do).
        fake_home = self.temp_dir / "fake_home"
        with mock.patch.object(Path, "home", return_value=fake_home), \
                mock.patch.dict(os.environ, {"TPU_AGENT_SUITE_HOME": str(
                    fake_home / paths.TPU_AGENT_SUITE_DIRNAME)}):
            installer.check_store_not_relocated()

    def test_install_rejects_a_plan_that_drops_agents(self):
        """Shipping agents that the plan never installs must fail loudly.

        Left alone it succeeds silently, and the damage surfaces much later as
        an orchestrator that was told to delegate, found nothing registered,
        and ran the agent's instructions inline instead.
        """
        plan_path = self.dist_dir / "tpu-agent-suite-install.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["copies"] = [
            c for c in plan["copies"] if "agents" not in c["destination"]
        ]
        plan_path.write_text(json.dumps(plan), encoding="utf-8")

        workspace = self._workspace("workspace_dropped_agents")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            self._install(workspace)
        self.assertIn("never installs", str(ctx.exception))
        self.assertIn("helper", str(ctx.exception))

    def test_collision_detection(self):
        workspace = self._workspace("workspace_collision")
        (workspace / ".claude" / "skills" / "entry").mkdir(parents=True)
        (workspace / ".claude" / "skills" / "entry" / "SKILL.md").write_text(
            "EXISTING UNMANAGED FILE", encoding="utf-8")

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            self._install(workspace)
        self.assertIn("refusing to overwrite", str(ctx.exception).lower())

        # Test --force bypasses collision check and installs
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self._install(workspace, force=True, announce=True)
        self.assertTrue((workspace / ".claude" / "skills" / "entry" /
                         "SKILL.md").is_file())
        self.assertIn(f"Restart Claude Code in {workspace.resolve()}",
                      out.getvalue())

    def test_reinstall_keeps_profiles_written_by_config(self):
        workspace = self._workspace("workspace_profiles")
        install = functools.partial(self._install, workspace)
        install()
        plan = json.loads(
            (self.dist_dir / "tpu-agent-suite-install.json").read_text())
        env_path = workspace / plan["environment_path"]
        profiles = {
            "revision": 5,
            "default": "vm",
            "profiles": {
                "vm": {
                    "backend": "local",
                    "params": {}
                }
            }
        }
        env_path.write_text(json.dumps(profiles), encoding="utf-8")
        results = env_path.with_name("verification.json")
        results.write_text('{"h": {"profile": "vm"}}')
        for kwargs in ({}, {"force": True}, {"clean": True}):
            with self.subTest(**kwargs):
                install(**kwargs)
                self.assertEqual(json.loads(env_path.read_text()), profiles)
                self.assertEqual(json.loads(results.read_text()),
                                 {"h": {
                                     "profile": "vm"
                                 }})
        # verify's results go with the package.
        installer.uninstall(destination=workspace, package="pkg")
        self.assertFalse(results.parent.exists())

    def test_user_scope_keeps_one_environment_json_in_the_central_store(self):
        from framework.compute import setup as compute_setup

        home = self.temp_dir / "home"
        central = self.central_home / "environment.json"
        central.parent.mkdir(parents=True, exist_ok=True)
        saved = {
            "revision": 2,
            "default": "a",
            "profiles": {
                "a": {
                    "backend": "local"
                }
            }
        }
        central.write_text(json.dumps(saved))
        install = functools.partial(installer.install,
                                    scope="user",
                                    non_interactive=True,
                                    skip_harness_check=True)
        with mock.patch.dict(os.environ, {"HOME": str(home)}):
            install(distribution=self.dist_dir)
            env, source = paths.resolve_environment("pkg@1.0.0", home)
            # A bare name, as typed by hand in $HOME, finds the same file
            # rather than making a new copy beside it.
            self.assertEqual(compute_setup._environment("pkg", home),
                             (env, source))
        self.assertEqual((env, source.resolve()), (saved, central.resolve()))
        self.assertEqual(
            jsonio.load_json(self.central_home / "ownership.json")["package"],
            "pkg")
        self.assertFalse(
            (home / ".tpu-agent-suite" / "environment.json").exists())
        # The file sits outside every version's directory, so an upgrade
        # finds it where the last version left it.
        self._update_manifest(self.pkg_dir, version="2.0.0")
        dist2 = self.temp_dir / "dist2"
        translators.translate(self.pkg_dir, "claude-code", dist2)
        with mock.patch.dict(os.environ, {"HOME": str(home)}):
            install(distribution=dist2)
            env, source = paths.resolve_environment("pkg@2.0.0", home)
            self.assertEqual((env, source.resolve()),
                             (saved, central.resolve()))
            # A user-scope uninstall removes it: it is in no ledger.
            installer.uninstall(scope="user")
        self.assertFalse(central.exists())
        self.assertFalse((self.central_home / "ownership.json").exists())

    def _other_package_dist(self) -> Path:
        """A distribution of a second package, `other`."""
        other_pkg = self.temp_dir / "other_pkg"
        shutil.copytree(self.pkg_dir, other_pkg)
        self._update_manifest(other_pkg, name="other")
        other_dist = self.temp_dir / "other_dist"
        translators.translate(other_pkg, "claude-code", other_dist)
        return other_dist

    def test_a_workspace_holds_one_package_at_a_time(self):
        """A second package is refused; the installed one still upgrades."""
        workspace = self._workspace("workspace_one_package")
        install = functools.partial(self._install, workspace)
        install(distribution=self.dist_dir)

        with self.assertRaisesRegex(utils.TpuAgentSuiteError,
                                    "already has pkg installed"):
            install(distribution=self._other_package_dist())
        ledger = workspace / ".tpu-agent-suite" / "ownership.json"
        self.assertEqual(jsonio.load_json(ledger)["package"], "pkg")

        install(distribution=self.dist_dir)
        self.assertEqual(jsonio.load_json(ledger)["package"], "pkg")

    def test_uninstall_reads_the_package_from_the_ledger(self):
        """No --package needed; given, it has to match what is installed."""
        workspace = self._workspace("workspace_ledger_uninstall")
        self._install(workspace)

        with self.assertRaisesRegex(utils.TpuAgentSuiteError,
                                    "has pkg installed, not other"):
            installer.uninstall(destination=workspace, package="other")
        installer.uninstall(destination=workspace)
        self.assertFalse(
            (workspace / ".tpu-agent-suite" / "ownership.json").exists())
        with self.assertRaisesRegex(utils.TpuAgentSuiteError,
                                    "nothing is installed"):
            installer.uninstall(destination=workspace)

    def test_the_installed_package_uninstalls_without_its_distribution(self):
        """The central copy of the installer removes the install by itself.

        Both installers run as programs, as a user runs them: the
        distribution's framework/installer.py, and the central one, run after
        the distribution is gone. Releasing the last reference deletes the
        directory that installer runs from.
        """
        self.assertFalse((self.dist_dir / "utils.py").exists())
        workspace = self._workspace("workspace_central_uninstall")

        def run(script: Path, *argv: str) -> None:
            result = subprocess.run([
                sys.executable,
                str(script), *argv, "--destination",
                str(workspace)
            ],
                                    capture_output=True,
                                    text=True,
                                    cwd=self.temp_dir,
                                    env=_env_without_pythonpath(),
                                    check=False)
            self.assertEqual(result.returncode, 0, result.stderr)

        run(self.dist_dir / "framework" / "installer.py", "install",
            "--non-interactive", "--skip-harness-check", "--skip-host-check",
            "--relocated-store")
        central_installer = (paths.get_package_storage_root("pkg", "1.0.0") /
                             "framework" / "installer.py")
        self.assertTrue(central_installer.is_file())

        shutil.rmtree(self.dist_dir)
        # Beside a build file but no plan, it is not a distribution.
        reinstall = subprocess.run([
            sys.executable,
            str(central_installer), "install", "--non-interactive",
            "--destination",
            str(workspace)
        ],
                                   capture_output=True,
                                   text=True,
                                   check=False)
        self.assertEqual(reinstall.returncode, 1)
        self.assertIn("distribution directory is required", reinstall.stderr)
        run(central_installer, "uninstall")
        self.assertFalse(
            (workspace / ".tpu-agent-suite" / "ownership.json").exists())
        self.assertFalse(central_installer.parent.exists())

    def test_a_ledger_of_another_schema_is_left_to_its_own_installer(self):
        """Install and uninstall refuse it, and name the installer that can."""
        workspace = self._workspace("workspace_ledger_schema")
        install = functools.partial(self._install, workspace)
        install()
        ledger = workspace / ".tpu-agent-suite" / "ownership.json"
        self.assertEqual(
            jsonio.load_json(ledger)["ledger_schema"], installer.LEDGER_SCHEMA)
        jsonio.write_json(
            ledger, {
                **jsonio.load_json(ledger), "ledger_schema":
                installer.LEDGER_SCHEMA + 1
            })

        command = re.escape(
            f"python3 {paths.get_package_storage_root('pkg', '1.0.0')}"
            f"/framework/installer.py uninstall --destination {workspace.resolve()}"
        )
        with self.assertRaisesRegex(utils.TpuAgentSuiteError, command):
            install()
        with self.assertRaisesRegex(utils.TpuAgentSuiteError, command):
            installer.uninstall(destination=workspace)
        self.assertEqual(
            jsonio.load_json(ledger)["ledger_schema"],
            installer.LEDGER_SCHEMA + 1)
        self.assertTrue(
            (workspace / ".claude" / "skills" / "entry" / "SKILL.md").exists())

    def test_uninstall_hands_off_to_the_installed_versions_installer(self):
        """Another installer passes uninstall to the one that wrote the ledger."""
        workspace = self._workspace("workspace_handoff")
        self._install(workspace)
        central_installer = (paths.get_package_storage_root("pkg", "1.0.0") /
                             "framework" / "installer.py")
        handed_off = central_installer.with_name("handed_off.json")
        central_installer.write_text(
            "import json, sys\n"
            f"open({str(handed_off)!r}, 'w').write(json.dumps(sys.argv[1:]))\n",
            encoding="utf-8")

        argv = ["uninstall", "--destination", str(workspace)]
        self.assertEqual(installer.main(argv), 0)
        self.assertEqual(jsonio.load_json(handed_off), argv)
        self.assertTrue(
            (workspace / ".tpu-agent-suite" / "ownership.json").exists())

    def _install_with_target(self, workspace: Path, tpu_target: str) -> None:
        """Install the test package into `workspace`, set to `tpu_target`."""
        self._install(workspace)
        jsonio.write_json(workspace / ".tpu-agent-suite" / "environment.json",
                          {"tpu_target": tpu_target})

    def test_second_workspace_install_is_isolated_from_the_first(self):
        """Installing into a second project never points it at the first.

        Both installs share one central payload; everything a harness loads
        from the second workspace must still be its own.
        """
        project_1 = self.temp_dir / "project_1"
        project_2 = self.temp_dir / "project_2"
        project_1.mkdir()
        project_2.mkdir()
        self._install_with_target(project_1, "tpu-one")
        self._install_with_target(project_2, "tpu-two")

        for workspace, target in ((project_1, "tpu-one"), (project_2,
                                                           "tpu-two")):
            env, source = paths.resolve_environment("pkg", workspace)
            self.assertEqual(env["tpu_target"], target)
            self.assertEqual(
                source,
                workspace.resolve() / ".tpu-agent-suite" / "environment.json")

        # No installed file in project_2 mentions project_1, under either
        # spelling of its path (macOS temp dirs resolve through /private).
        p1_spellings = {str(project_1), str(project_1.resolve())}
        for path in project_2.rglob("*"):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for spelling in p1_spellings:
                self.assertNotIn(spelling, text,
                                 f"{path} references project_1")

        # Both workspaces hold a claim on the shared payload, so removing
        # project_1 leaves it in place for project_2.
        central = self.central_home / "packages" / "pkg@1.0.0"
        refs = json.loads((central / ".refs.json").read_text(encoding="utf-8"))
        self.assertEqual(
            refs["workspaces"],
            sorted([str(project_1.resolve()),
                    str(project_2.resolve())]))

        installer.uninstall(project_1, "pkg")
        self.assertTrue(central.is_dir())
        refs = json.loads((central / ".refs.json").read_text(encoding="utf-8"))
        self.assertEqual(refs["workspaces"], [str(project_2.resolve())])
        env, _ = paths.resolve_environment("pkg", project_2)
        self.assertEqual(env["tpu_target"], "tpu-two")
        self.assertTrue(
            (project_2 / ".claude" / "agents" / "helper.md").is_file())

    def test_invalid_scope_and_destination_combinations_are_refused(self):
        workspace = self.temp_dir / "workspace_scope"
        cases = {
            "project_without_destination":
            ("project", None, "--destination is required"),
            "user_with_destination":
            ("user", workspace, "Cannot specify --destination"),
            "unknown_scope": ("local", workspace, "unknown scope 'local'"),
            "project_into_the_distribution":
            ("project", self.dist_dir,
             "Cannot install into the distribution directory"),
        }
        for name, (scope, destination, message) in cases.items():
            with self.subTest(name):
                with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
                    installer.resolve_install_destination(
                        self.dist_dir, scope=scope, destination=destination)
                self.assertIn(message, str(ctx.exception))

    def test_upgrade_keeps_reused_directories(self):
        """A reinstall repopulates .claude/ in place rather than recreating it.

        Recreated, the directory gets a new inode, and a container that
        bind-mounted the old one sees an empty /root/.claude until restarted.
        """
        workspace = self._workspace("workspace_reinstall")
        self._install(workspace)
        claude_dir = workspace / ".claude"
        agents_dir = claude_dir / "agents"
        before = (claude_dir.stat().st_ino, agents_dir.stat().st_ino)

        self._install(workspace)

        self.assertEqual((claude_dir.stat().st_ino, agents_dir.stat().st_ino),
                         before)
        self.assertTrue((agents_dir / "helper.md").is_file())

    def test_upgrade_and_uninstall_leave_unowned_empty_directories(self):
        """Only directories the package emptied are pruned."""
        workspace = self.temp_dir / "workspace_unowned_dirs"
        unowned = [workspace / "Downloads", workspace / ".claude" / "commands"]
        for directory in unowned:
            directory.mkdir(parents=True)
        self._install(workspace)
        self._install(workspace)
        installer.uninstall(workspace, "pkg")

        for directory in unowned:
            self.assertTrue(directory.is_dir(), directory)
        self.assertFalse((workspace / ".claude" / "agents").exists())
        self.assertFalse((workspace / ".tpu-agent-suite").exists())

    def test_uninstall_leaves_empty_run_directories(self):
        """Run data outlives the install, down to a run's empty artifacts/."""
        home = self._workspace("home_runs")
        artifacts = self.central_home / "runs" / "run-1" / "artifacts"
        with mock.patch.dict(os.environ, {"HOME": str(home)}):
            self._install(None, scope="user")
            artifacts.mkdir(parents=True)
            installer.uninstall(package="pkg", scope="user")

        self.assertTrue(artifacts.is_dir())
        self.assertFalse((self.central_home / "ownership.json").exists())

    def test_upgrade_prunes_a_directory_the_new_version_no_longer_uses(self):
        """The other direction: a directory the upgrade left empty still goes."""
        workspace = self._workspace("workspace_dropped_dir")
        self._install(workspace)
        ledger_path = workspace / ".tpu-agent-suite" / "ownership.json"
        ledger = jsonio.load_json(ledger_path)
        dropped = workspace / ".claude" / "dropped" / "old.md"
        dropped.parent.mkdir()
        dropped.write_text("old", encoding="utf-8")
        ledger["actions"].append({
            "kind": "file",
            "path": ".claude/dropped/old.md",
            "sha256": installer.compute_digest(dropped),
        })
        jsonio.write_json(ledger_path, ledger)

        self._install(workspace)

        self.assertFalse(dropped.parent.exists())
        self.assertTrue((workspace / ".claude" / "agents").is_dir())

    def test_user_scope_leaves_the_rest_of_home_alone(self):
        """A user-scope install's destination is $HOME itself.

        Upgrade and uninstall swept the whole destination for empty
        directories, so they deleted empty ones anywhere in the home
        directory.
        """
        home = self.temp_dir / "home"
        unowned = [
            home / "Downloads", home / "projects" / "new",
            home / ".claude" / "commands"
        ]
        for directory in unowned:
            directory.mkdir(parents=True)
        with mock.patch.dict(os.environ, {"HOME": str(home)}):
            self._install(None, scope="user")
            self._install(None, scope="user")
            installer.uninstall(package="pkg", scope="user")

        for directory in unowned:
            self.assertTrue(directory.is_dir(), directory)
        self.assertFalse((home / ".claude" / "agents").exists())

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
        # test_compiler; this covers what the installer does with them.
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
        """~/.claude/CLAUDE.md, ~/.codex/AGENTS.md and ~/.gemini/AGENTS.md,
        not the bare file in ~, which only a session started in $HOME itself
        would read."""
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
        """Re-installing replaces the block instead of appending a second one;
        a memory file the install created goes when nothing else remains."""
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
        """A hand-edited block is modified managed content; --force replaces
        it. Uninstall never discards the user's own text or edits."""
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
        """A CODEX_HOME the ledger cannot root fails the install, rather than
        succeeding into ~/.codex, which that Codex never reads."""
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
        """An install made before the fix recorded its block at ~/AGENTS.md;
        the upgrade strips it there and writes it once where it is read."""
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
