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

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import installer, jsonio, paths, utils  # noqa: E402
from framework.builder import translators  # noqa: E402

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
        """Install headlessly into `workspace`.

        The distribution is `self.dist_dir` unless `distribution` is given.
        """
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


class _TranslatedInstallerTestCase(_InstallerTestCase):
    """The test package translated for claude-code into `self.dist_dir`."""

    def setUp(self):
        super().setUp()
        # The host check shells out to python3; its own test is
        # TestHostCheck in test_installer_hooks.py.
        host_check = mock.patch.object(installer, "check_host")
        self.check_host = host_check.start()
        self.addCleanup(host_check.stop)
        self.dist_dir = self.temp_dir / "dist"
        translators.translate(self.pkg_dir, "claude-code", self.dist_dir)


class TestInstaller(_TranslatedInstallerTestCase):
    """Unit tests for installer functions, pre-validations, collisions, and rollback."""

    def test_install_without_a_terminal_fails_fast_naming_headless_flags(self):
        """Without a TTY, an interactive install refuses before doing anything.

        Otherwise it would hang on (or die mid-prompt reading) stdin. --force
        never asks, so it installs.
        """
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
        """Hooks always run ~/.tpu-agent-suite/.../runtime.py.

        So a store TPU_AGENT_SUITE_HOME moved elsewhere breaks every hook.
        Refuse before writing anything.
        """
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

    def test_a_conflicting_user_setting_stops_the_install_untouched(self):
        """A conflicting setting stops the install before anything is written.

        This holds even with --force.
        """
        workspace = self._workspace("workspace_setting_conflict")
        settings = workspace / ".claude" / "settings.json"
        settings.parent.mkdir()
        original = json.dumps({
            "theme": "dark",
            "env": {
                "BASH_DEFAULT_TIMEOUT_MS": "60000"
            }
        })
        settings.write_text(original, encoding="utf-8")

        for kwargs in ({}, {"force": True}):
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(
                        utils.TpuAgentSuiteError,
                        "refusing to replace existing JSON setting: "
                        "env/BASH_DEFAULT_TIMEOUT_MS"):
                    self._install(workspace, **kwargs)
                self.assertEqual(settings.read_text(encoding="utf-8"),
                                 original)
                self.assertFalse((workspace / ".tpu-agent-suite" /
                                  "ownership.json").exists())
                self.assertFalse((workspace / ".claude" / "skills").exists())
                self.assertFalse((workspace / ".claude" / "agents").exists())
                self.assertFalse((self.central_home / "packages").exists())

    def test_install_refuses_a_harness_version_outside_the_target(self):
        """A harness version outside the target's range is refused.

        A version inside the range installs.
        """
        workspace = self._workspace("workspace_harness_version")
        plan = jsonio.load_json(self.dist_dir / "tpu-agent-suite-install.json")

        def install_with_harness(version_output: str) -> mock.MagicMock:
            with mock.patch.object(installer.subprocess,
                                   "run",
                                   return_value=subprocess.CompletedProcess(
                                       plan["version_command"], 0,
                                       version_output, "")) as run:
                installer.install(distribution=self.dist_dir,
                                  destination=workspace,
                                  non_interactive=True)
            return run

        with self.assertRaisesRegex(
                utils.TpuAgentSuiteError,
                r"'1\.9\.9 \(Claude Code\)' is outside"
                r" >=2\.0\.0"):
            install_with_harness("1.9.9 (Claude Code)\n")
        self.assertEqual(list(workspace.iterdir()), [])
        self.assertFalse((self.central_home / "packages").exists())

        run = install_with_harness("2.1.0 (Claude Code)\n")
        self.assertEqual(run.call_args.args[0], plan["version_command"])
        self.assertTrue(
            (workspace / ".tpu-agent-suite" / "ownership.json").is_file())

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
            self.assertEqual(compute_setup._require_environment("pkg", home),
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
        header = jsonio.load_json(ledger)
        self.assertEqual(
            {
                key: header.get(key)
                for key in ("ledger_schema", "package", "version")
            }, {
                "ledger_schema": installer.LEDGER_SCHEMA,
                "package": "pkg",
                "version": "1.0.0"
            })
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


if __name__ == "__main__":
    unittest.main()
