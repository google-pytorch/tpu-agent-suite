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
"""Tests for per-harness translation, hooks, prompt contracts and archives."""

import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

# Imports below follow this sys.path setup, so the file runs directly.
# ruff: noqa: E402
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import compiler, jsonio, utils
from framework.builder import (harness_rules, hook_config, package_discovery,
                               package_loader, prompt_contracts, translators,
                               verify)
from framework.tests.builder import helpers

Path = pathlib.Path


class TestHarnessTranslation(helpers.CompilerTestCase):
    """Translating a package for each harness."""

    def test_compose_package_subcommand(self):
        pkg_dir = self._create_valid_package()
        out_file = self.temp_dir / "composed" / "package.release.json"
        ret = compiler.main(
            ["compose", str(pkg_dir), "--output",
             str(out_file)])
        self.assertEqual(ret, 0)
        written = json.loads(out_file.read_text(encoding="utf-8"))
        self.assertEqual(written["name"], "valid-pkg")
        self.assertEqual([t["name"] for t in written["script_tools"]],
                         ["custom_runner"])


class TestDelegationContract(helpers.CompilerTestCase):
    """How each harness is told to delegate, or that it cannot."""

    def test_unknown_harness_delegation_is_a_build_error(self):
        """A harness with no declared mechanism must fail the build.

        Otherwise it would ship routing instructions its runtime cannot honor.
        """
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            harness_rules.delegation_tool_for("no-such-harness", ["worker"])
        self.assertIn("no delegation mechanism", str(ctx.exception))


class TestBuiltinTokens(helpers.CompilerTestCase):
    """Built-in tool tokens and the ask token."""

    def test_only_a_skill_may_hold_ask(self):
        """An agent granted `ask` would call a tool nobody answers.

        Its caller is blocked on the result, so the question reaches no one.
        Caught at build time because at run time it fails silently.
        """
        pkg_dir = self._create_valid_package()
        # The fixture declares its own native tools rather than the shipped
        # catalog, so `ask` has to exist before an agent can name it.
        manifest = self._read_manifest(pkg_dir)
        manifest["native_tools"].append({
            "name": "ask",
            "visibility": "external",
            "action_level": {
                "effect": "read",
                "scope": "session"
            },
        })
        self._write_manifest(pkg_dir, manifest)
        self._edit_worker(pkg_dir, "tools:\n", "tools:\n  - ask\n")
        with self.assertRaises(utils.TpuAgentSuiteError) as caught:
            package_loader.load_package(pkg_dir)
        message = str(caught.exception)
        self.assertIn("only a skill may hold", message)
        self.assertIn("needs_input", message)


class TestSkillToolRestriction(helpers.CompilerTestCase):
    """A skill's `tools` limit and how each harness enforces it."""

    def test_unclassified_harness_cannot_build_a_restricted_skill(self):
        skill = {"name": "entry", "tools": ["read"]}
        with self.assertRaisesRegex(utils.TpuAgentSuiteError,
                                    "SKILL_TOOL_RESTRICTION"):
            hook_config.skill_tools_enforced(skill, "no-such-harness")


class TestHookTranslation(helpers.CompilerTestCase):
    """Translating declared hooks into each harness's config."""

    def test_matchers_on_an_event_with_nothing_to_match_are_rejected(self):
        """Only a tool or a subagent can be matched.

        Matchers on any other event used to leave the compiler reusing the
        previous hook's matcher, and the runtime then skipped every fire of
        the hook: it never ran, and nothing said so.
        """
        pkg_dir = self._create_valid_package()
        manifest = self._read_manifest(pkg_dir)
        hook = {"matchers": ["worker"], "command": "true"}
        manifest["hooks"] = [
            {
                **hook, "name": "on-subagent",
                "event": "SubagentStop"
            },
            {
                **hook, "name": "on-loop",
                "event": "Stop"
            },
        ]
        self._write_manifest(pkg_dir, manifest)
        with self.assertRaisesRegex(utils.TpuAgentSuiteError,
                                    "'on-loop' declares matchers"):
            verify.verify_package(pkg_dir)
        with self.assertRaisesRegex(utils.TpuAgentSuiteError,
                                    "'on-loop' declares matchers"):
            hook_config.build_harness_hooks_config("valid-pkg", manifest,
                                                   "antigravity")

    def test_translate_hooks_all_harnesses(self):
        pkg_dir = self._create_valid_package()
        manifest = self._read_manifest(pkg_dir)
        manifest["hooks"] = [{
            "name": "guard",
            "event": "PreToolUse",
            "matchers": ["read", "write", "shell"],
            "action": "script:scripts/custom.py",
            "args": ["--strict", "param with space"],
            "timeout": 5,
        }, {
            "name": "tracker",
            "event": "PostToolUse",
            "command": "python3 track.py",
            "args": ["msg with space"],
        }, {
            "name": "verify",
            "event": "SubagentStop",
            "matchers": ["worker"],
            "command": "true",
        }, {
            "name": "init",
            "event": "SessionStart",
            "command": "true",
        }]
        self._write_manifest(pkg_dir, manifest)
        hooks = {}
        for harness, rel in (("claude-code", ".claude-plugin/hooks.json"),
                             ("antigravity",
                              "hooks.json"), ("cursor", ".cursor/hooks.json"),
                             ("codex", ".codex/hooks.json")):
            translators.translate(pkg_dir, harness, self.temp_dir / harness)
            hooks[harness] = jsonio.load_json(self.temp_dir / harness / rel)

        guard = hooks["claude-code"]["guard"]["PreToolUse"][0]
        self.assertEqual(guard["matcher"], "Read|Write|Bash")
        # Arguments follow the script after `--`, or join the command, each
        # shell-quoted so one with a space stays one argument.
        self.assertEqual(
            guard["hooks"][0]["command"],
            hook_config.fail_safe_hook_command(
                "python3 ~/.tpu-agent-suite/packages/valid-pkg@1.0.0/framework/runtime.py hook-dispatch "
                "--harness claude-code --package valid-pkg@1.0.0 --hook guard "
                "--event PreToolUse --timeout 5 --on-failure deny --script scripts/custom.py "
                "-- --strict 'param with space'", "guard", "deny",
                "PreToolUse"))
        self.assertIn(
            "--command " +
            shlex.quote("python3 track.py " + shlex.quote("msg with space")),
            hooks["claude-code"]["tracker"]["PostToolUse"][0]["hooks"][0]
            ["command"])
        # The harness outwaits dispatch's own --timeout, so dispatch's verdict
        # on a hung hook reaches the harness instead of being killed with it.
        self.assertEqual(guard["hooks"][0]["timeout"],
                         5 + hook_config.HOOK_DISPATCH_HEADROOM_S)
        # A background subagent delivers through the injected SubagentHandback
        # tool, so the SubagentStop body is also registered on that call, with
        # the subagent matcher moved into the command.
        handback = hooks["claude-code"]["verify"]["PreToolUse"][0]
        self.assertEqual(handback["matcher"], "SubagentHandback")
        self.assertIn("--handback-event PreToolUse --matcher worker",
                      handback["hooks"][0]["command"])

        antigravity = hooks["antigravity"]
        self.assertEqual(antigravity["guard"]["PreToolUse"][0]["matcher"],
                         "view_file|write_to_file|run_command")
        self.assertIn("--matcher worker",
                      antigravity["verify"]["Stop"][0]["command"])
        # No SessionStart there: a flat PreInvocation handler, since a matcher
        # group would make it silently never fire.
        self.assertNotIn("matcher", antigravity["init"]["PreInvocation"][0])
        # Cursor: version-tagged, matcher on the definition itself.
        self.assertEqual(hooks["cursor"]["version"], 1)
        self.assertEqual(hooks["cursor"]["hooks"]["preToolUse"][0]["matcher"],
                         "Read|Write|Shell")
        # Codex has no `read` tool: it falls through to the canonical name.
        self.assertEqual(hooks["codex"]["hooks"]["PreToolUse"][0]["matcher"],
                         "read|Write|Bash")
        plan = jsonio.load_json(self.temp_dir / "codex" /
                                "tpu-agent-suite-install.json")
        self.assertIn(
            {
                "path": ".codex/config.toml",
                "section": "hooks",
                "key": "enabled",
                "value": True
            }, plan["toml_sets"])

    def test_hooks_validation_errors(self):
        pkg_dir = self._create_valid_package()
        manifest = self._read_manifest(pkg_dir)
        pre = {"name": "h", "event": "PreToolUse"}
        script = {**pre, "action": "script:scripts/custom.py"}
        for expected_error, hooks in (
            ("duplicate hook name", [{
                **pre, "command": "true"
            }, {
                **pre, "event": "PostToolUse",
                "command": "true"
            }]),
            ("script file does not exist", [{
                **pre, "action":
                "script:scripts/absent.py"
            }]),
            ("is not one of", [{
                **pre, "event": "InvalidEvent",
                "command": "true"
            }]),
                # Per-session hook state lives at
                # sessions/<id>.contexts/<name>.json, so "a:b" and "a_b"
                # would share one file.
            ("does not match", [{
                **pre, "name": "bad:name",
                "command": "true"
            }]),
                # start-run rebinds the session to a fresh run namespace; per
                # tool call it would orphan the artifacts of the run it
                # replaced.
            ("cannot be attached to event 'PreToolUse'", [{
                **pre, "action":
                "builtin:start-run"
            }]),
            ("must specify either an 'action' or 'command'", [pre]),
            ("cannot specify both 'action' and 'command'", [{
                **script, "command":
                "true"
            }]),
            ("action must start with 'script:'", [{
                **pre, "action":
                "scripts/custom.py"
            }]),
            ("on_failure: value is not one of ['deny', 'allow']", [{
                **script, "on_failure":
                "maybe"
            }]),
            ("below minimum", [{
                **script, "timeout": 0
            }]),
        ):
            with self.subTest(expected_error):
                self._write_manifest(pkg_dir, {**manifest, "hooks": hooks})
                with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
                    package_loader.load_package(pkg_dir)
                self.assertIn(expected_error, str(ctx.exception))


class TestHookFailurePolicy(helpers.CompilerTestCase):
    """`on_failure` and the compiled command's fail-safe."""

    def _run_hook_command(self, command: str, home: Path):
        """Run a compiled hook command as a harness would.

        It runs in a shell, with an empty payload on stdin, `~` pointing at
        `home`.
        """
        env = {"HOME": str(home), "PATH": os.environ.get("PATH", "")}
        return subprocess.run(["/bin/sh", "-c", command],
                              input="{}",
                              capture_output=True,
                              text=True,
                              env=env,
                              check=False)

    def test_compiled_hook_command_applies_policy_when_dispatch_cannot_run(
            self):
        """The command exits by `on_failure` when hook-dispatch cannot run.

        Without the fallback a missing runtime.py exits 2, which Claude Code
        reads as a block: every `allow` hook denied every tool. On Stop a
        block means "keep going", so there the fallback exits 0 whatever the
        policy, or a dead dispatcher would keep the turn from ever ending.
        """
        manifest = self._read_manifest(self._create_valid_package())
        manifest["hooks"] = [{
            "name": name,
            "event": event,
            "command": "true",
            "on_failure": policy,
        } for name, event, policy in (("lenient", "PreToolUse", "allow"),
                                      ("strict", "PreToolUse",
                                       "deny"), ("stop", "Stop", "deny"))]
        hooks = hook_config.build_harness_hooks_config("valid-pkg", manifest,
                                                       "claude-code")
        empty_home = self.temp_dir / "empty_home"
        empty_home.mkdir()
        for name, event, code, message in (
            ("lenient", "PreToolUse", 0,
             "hook lenient: hook-dispatch exited 2"),
            ("strict", "PreToolUse", 2, "on_failure=deny"),
            ("stop", "Stop", 0, "allowing: on_failure=deny would loop on"),
        ):
            with self.subTest(hook=name):
                res = self._run_hook_command(
                    hooks[name][event][0]["hooks"][0]["command"], empty_home)
                self.assertEqual(res.returncode, code, res.stderr)
                self.assertIn(message, res.stderr)


class TestContractSchemaTemplates(unittest.TestCase):
    """Test schema examples and contract prompt generation."""

    def test_result_contract_states_only_what_the_schema_declares(self):
        """A generated prompt must not contradict the schema it points at.

        The status line names exactly the statuses the schema declares, and
        the artifact rule is stated only where the runtime enforces it:
        `check_completed_has_artifact` runs only for a schema that declares
        `artifacts`, and fires only on the literal status `completed`.
        """
        schema = {
            "properties": {
                "status": {
                    "enum": ["optimized", "invalid_input", "failed"]
                },
            },
            "status_requires": {
                "optimized": ["best_kernel_artifact", "artifacts"]
            },
        }
        joined = "\n".join(prompt_contracts.result_contract_lines(schema))
        self.assertIn("`optimized`, `invalid_input`, `failed`", joined)
        self.assertIn("An `optimized` result must carry", joined)
        self.assertIn("`best_kernel_artifact`, `artifacts`", joined)
        self.assertNotIn("completed", joined)
        self.assertNotIn("needs_input", joined)

        # Rules but no enum: the statuses keyed in `status_requires` still
        # form the status line.
        del schema["properties"]["status"]["enum"]
        joined = "\n".join(prompt_contracts.result_contract_lines(schema))
        self.assertIn("`optimized`", joined)
        self.assertNotIn("completed", joined)

        completed = {"status": {"enum": ["completed", "failed"]}}
        keeps_it = "\n".join(
            prompt_contracts.result_contract_lines({
                "properties": {
                    **completed, "artifacts": {
                        "type": "array"
                    }
                },
            }))
        self.assertIn("A `completed` result must carry", keeps_it)
        no_artifacts = "\n".join(
            prompt_contracts.result_contract_lines({"properties": completed}))
        self.assertIn("`completed`", no_artifacts)
        self.assertNotIn("at least one artifact", no_artifacts)

        # No statuses in the schema means no status line to state.
        for empty in (None, {}, {
                "properties": {
                    "status": {
                        "type": "string"
                    }
                }
        }):
            with self.subTest(schema=empty):
                self.assertEqual(prompt_contracts.result_contract_lines(empty),
                                 [])

    def test_only_a_unit_with_a_shell_is_told_to_keep_the_journal(self):
        """A skill that lists no tools counts as having a shell."""
        manifest = {"name": "pkg", "version": "1.0.0"}
        section = "## Session journal"
        for tools, expected in ((["shell"], True), (["read"], False)):
            with self.subTest(agent_tools=tools):
                agent = {"name": "worker", "tools": tools, "delegates": []}
                self.assertEqual(
                    section in prompt_contracts.agent_execution_contract(
                        manifest, agent, "claude-code"), expected)
        for tools, expected in (([], True), (["read", "shell"], True),
                                (["read"], False)):
            with self.subTest(skill_tools=tools):
                skill = {"name": "entry", "tools": tools, "delegates": []}
                self.assertEqual(
                    section in prompt_contracts.skill_package_contract(
                        manifest, skill, "claude-code"), expected)


class TestCopyAssetExclusions(unittest.TestCase):
    """copy_asset never ships wikify working state, agent-skill installs, or raw/code."""

    def test_dist_excluded_covers_raw_code_and_tool_state_by_path(self):
        for rel, expected in (
            ("knowledgebase/raw/code/torch_tpu/requirements/requirements_torch.txt",
             True),
            ("knowledgebase/.cache/scip/requirements.txt", True),
            ("knowledgebase/wiki/code/page.md", False),
            ("knowledgebase/raw/docs/requirements-notes.txt", False),
            ("requirements.txt", False),
        ):
            with self.subTest(rel=rel):
                self.assertEqual(package_discovery.dist_excluded(Path(rel)),
                                 expected)


class TestStandaloneArchives(unittest.TestCase):
    """Standalone distribution archives: name and layout are a contract."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_create_archives_packages_distribution(self):
        dist_dir = self.tmp / "dist"
        claude_dist = dist_dir / "claude-code" / "test-pkg"
        claude_dist.mkdir(parents=True)
        jsonio.write_json(
            claude_dist / "tpu-agent-suite-build.json",
            {
                "source": "test-pkg",
                "version": "1.2.3",
                "harness": "claude-code",
            },
        )
        (claude_dist / "test.txt").write_text("hello", encoding="utf-8")

        out_dir = self.tmp / "archives"
        archives = compiler.create_archives(dist_dir,
                                            out_dir,
                                            harnesses=["claude-code"])
        self.assertEqual(len(archives), 1)
        archive_path = archives[0]
        self.assertEqual(archive_path.name,
                         "test-pkg-claude-code-v1.2.3.tar.gz")
        self.assertTrue(archive_path.is_file())

        with tarfile.open(archive_path, "r:gz") as tar:
            names = tar.getnames()
            self.assertIn("test-pkg-claude-code-v1.2.3", names)
            self.assertIn("test-pkg-claude-code-v1.2.3/test.txt", names)


if __name__ == "__main__":
    unittest.main()
