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
"""Unit tests for the package build: compiler.py and framework/builder/."""

import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import unittest

from framework import compiler, jsonio, utils
from framework.builder import (dist_assets, harness_rules, hook_config,
                               package_checks, package_discovery,
                               package_loader, prompt_contracts, translators,
                               verify)

Path = pathlib.Path


class TestCompiler(unittest.TestCase):
    """Unit tests for package verification, delegation graph, and translation."""

    def setUp(self):
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def _create_valid_package(self) -> Path:
        pkg_dir = self.temp_dir / "valid-pkg"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "scripts").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "scripts" / "custom.py").write_text("print('hello')",
                                                       encoding="utf-8")

        manifest = {
            "name":
            "valid-pkg",
            "version":
            "1.0.0",
            "description":
            "Valid test package",
            "native_tools": [
                {
                    "name": "read",
                    "visibility": "external",
                    "action_level": {
                        "effect": "read",
                        "scope": "workspace"
                    },
                },
                {
                    "name": "write",
                    "visibility": "external",
                    "action_level": {
                        "effect": "write",
                        "scope": "workspace"
                    },
                },
            ],
            "script_tools": [{
                "name": "custom_runner",
                "path": "scripts/custom.py",
                "visibility": "internal",
                "action_level": {
                    "effect": "execute",
                    "scope": "host"
                },
                "description": "Custom runner",
            }],
            "mcp_servers": [{
                "name": "query_db",
                "command": "python3",
                "visibility": "external",
                "action_level": {
                    "effect": "read",
                    "scope": "workspace"
                },
            }],
            "skills": ["entry"],
            "copy": ["schemas", "environment", "scripts"],
            "environment": {
                "questions": "environment/questions.json",
                "schema": "schemas/env.json",
            },
            "compatibility_targets": [
                {
                    "name": "claude",
                    "harness": "claude-code",
                    "versions": ">=2.0.0",
                },
                {
                    "name": "codex",
                    "harness": "codex",
                    "versions": ">=0.1.0",
                },
                {
                    "name": "antigravity",
                    "harness": "antigravity",
                    "versions": ">=1.0.0",
                },
                {
                    "name": "cursor",
                    "harness": "cursor",
                    "versions": ">=0.40.0",
                },
            ],
        }
        self._write_manifest(pkg_dir, manifest)
        (pkg_dir / "skills" / "entry").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "agents").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "skills" / "entry" / "SKILL.md").write_text(
            "---\n"
            "name: entry\n"
            "description: Entry description\n"
            "delegates:\n"
            "  - worker\n"
            "---\n"
            "Call {{agent:worker}}.",
            encoding="utf-8")
        (pkg_dir / "agents" / "worker.md").write_text(
            "---\n"
            "name: worker\n"
            "description: Worker\n"
            "tools:\n"
            "  - read\n"
            "  - write\n"
            "  - custom_runner\n"
            "  - query_db\n"
            "delegates: []\n"
            "accepts: schemas/in.json\n"
            "produces: schemas/out.json\n"
            "---\n"
            "Work.",
            encoding="utf-8")
        (pkg_dir / "schemas").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "schemas" / "in.json").write_text(json.dumps(
            {"type": "object"}),
                                                     encoding="utf-8")
        (pkg_dir / "schemas" / "out.json").write_text(json.dumps(
            {"type": "object"}),
                                                      encoding="utf-8")
        (pkg_dir / "schemas" / "env.json").write_text(
            json.dumps({
                "type": "object",
                "properties": {
                    "tpu_target": {
                        "type": "string"
                    }
                },
                "required": ["tpu_target"],
            }),
            encoding="utf-8",
        )
        (pkg_dir / "environment").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "environment" / "questions.json").write_text(
            json.dumps({
                "questions": [{
                    "name": "tpu_target",
                    "prompt": "TPU target",
                    "type": "string",
                    "default": "local",
                }]
            }),
            encoding="utf-8",
        )
        return pkg_dir

    @staticmethod
    def _read_manifest(pkg_dir: Path) -> dict:
        return jsonio.load_json(pkg_dir / "package.release.json")

    @staticmethod
    def _write_manifest(pkg_dir: Path, manifest: dict) -> None:
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")

    def _edit_worker(self, pkg_dir: Path, old: str, new: str) -> None:
        """Replace the first `old` in the worker agent's document."""
        path = pkg_dir / "agents" / "worker.md"
        path.write_text(path.read_text(encoding="utf-8").replace(old, new, 1),
                        encoding="utf-8")

    def _delegate_to(self, pkg_dir: Path, *names: str) -> None:
        """Point the entry skill at `names`, which is how an agent is reached."""
        delegates = "".join(f"  - {name}\n" for name in names)
        routes = "\n".join(f"Call {{{{agent:{name}}}}}." for name in names)
        (pkg_dir / "skills" / "entry" / "SKILL.md").write_text(
            "---\n"
            "name: entry\n"
            "description: Entry description\n" +
            (f"delegates:\n{delegates}" if names else "delegates: []\n") +
            "---\n" + (routes or "Standalone."),
            encoding="utf-8")

    def _declare_dependencies(self, pkg_dir: Path, host=(), tpu=()) -> None:
        """Declare dependencies where they are read from: a unit's frontmatter."""
        path = pkg_dir / "agents" / "worker.md"
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"(?m)^(host|tpu)_dependencies:\n(?:  - .*\n)*", "",
                      text)
        extra = ""
        if host:
            extra += "host_dependencies:\n" + "".join(f"  - {d}\n"
                                                      for d in host)
        if tpu:
            extra += "tpu_dependencies:\n" + "".join(f"  - {d}\n" for d in tpu)
        path.write_text(text.replace("delegates: []\n",
                                     "delegates: []\n" + extra, 1),
                        encoding="utf-8")

    def test_second_manifest_builds_its_own_package_from_the_same_files(self):
        """--manifest picks another manifest file in the same directory."""
        pkg_dir = self._create_valid_package()
        manifest = self._read_manifest(pkg_dir)
        manifest["name"] = "valid-pkg-experimental"
        (pkg_dir / "package.experimental.json").write_text(
            json.dumps(manifest), encoding="utf-8")

        verify.verify_package(pkg_dir,
                              manifest_name="package.experimental.json")
        self.assertEqual(
            compiler.compose_package(pkg_dir)["name"], "valid-pkg")
        self.assertEqual(
            compiler.compose_package(
                pkg_dir, manifest_name="package.experimental.json")["name"],
            "valid-pkg-experimental")

        out = self.temp_dir / "dist_experimental"
        ret = compiler.main([
            "translate",
            str(pkg_dir), "--manifest", "package.experimental.json",
            "--harness", "claude-code", "--output",
            str(out)
        ])
        self.assertEqual(ret, 0)
        build = json.loads(
            (out / "tpu-agent-suite-build.json").read_text(encoding="utf-8"))
        self.assertEqual(build["source"], "valid-pkg-experimental")

    def _write_extending_manifest(self, pkg_dir: Path, **fields) -> None:
        (pkg_dir / "skills" / "extra").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "skills" / "extra" / "SKILL.md").write_text(
            "---\nname: extra\ndescription: Extra description\n---\nExtra.",
            encoding="utf-8")
        (pkg_dir / "package.experimental.json").write_text(json.dumps({
            "extends":
            "package.release.json",
            "name":
            "valid-pkg-experimental",
            **fields
        }),
                                                           encoding="utf-8")

    def test_extending_manifest_inherits_the_base_and_adds_skills(self):
        """`extends` takes every base field; `add_skills` appends to skills."""
        pkg_dir = self._create_valid_package()
        self._write_extending_manifest(pkg_dir, add_skills=["extra"])
        base = package_loader.load_package(pkg_dir)
        variant = package_loader.load_package(
            pkg_dir, manifest="package.experimental.json")

        self.assertEqual(variant["name"], "valid-pkg-experimental")
        self.assertEqual(variant["version"], base["version"])
        self.assertEqual([skill["name"] for skill in variant["skills"]],
                         ["entry", "extra"])
        verify.verify_package(pkg_dir,
                              manifest_name="package.experimental.json")

    def test_extending_manifest_errors(self):
        cases = {
            "repeats skills already listed": {
                "add_skills": ["entry"]
            },
            "must be a list of skill names": {
                "add_skills": "extra"
            },
            "must name another manifest file": {
                "extends": "package.experimental.json"
            },
        }
        for message, fields in cases.items():
            with self.subTest(message):
                pkg_dir = self._create_valid_package()
                self._write_extending_manifest(pkg_dir, **fields)
                with self.assertRaisesRegex(utils.TpuAgentSuiteError, message):
                    package_loader.load_package(
                        pkg_dir, manifest="package.experimental.json")

    def test_add_skills_without_extends_is_rejected(self):
        pkg_dir = self._create_valid_package()
        manifest = self._read_manifest(pkg_dir)
        self._write_manifest(pkg_dir, {**manifest, "add_skills": ["entry"]})
        with self.assertRaisesRegex(utils.TpuAgentSuiteError,
                                    "`add_skills` needs `extends`"):
            package_loader.load_package(pkg_dir)

    def test_extends_is_one_level(self):
        pkg_dir = self._create_valid_package()
        self._write_extending_manifest(pkg_dir)
        (pkg_dir / "package.third.json").write_text(json.dumps({
            "extends": "package.experimental.json",
            "name": "third"
        }),
                                                    encoding="utf-8")
        with self.assertRaisesRegex(utils.TpuAgentSuiteError,
                                    "only one level is supported"):
            package_loader.load_package(pkg_dir, manifest="package.third.json")

    def test_tool_name_uniqueness(self):
        pkg_dir = self._create_valid_package()
        manifest = self._read_manifest(pkg_dir)
        # Introduce duplicate tool name between native and script
        manifest["script_tools"].append({
            "name": "read",
            "path": "scripts/custom.py",
            "visibility": "external",
            "action_level": {
                "effect": "read",
                "scope": "session"
            },
        })
        self._write_manifest(pkg_dir, manifest)

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            verify.verify_package(pkg_dir)
        self.assertIn("duplicate tool name", str(ctx.exception).lower())

    def test_cycle_detection(self):
        pkg_dir = self._create_valid_package()
        (pkg_dir / "agents" / "worker2.md").write_text(
            "---\n"
            "name: worker2\n"
            "description: Worker 2\n"
            "tools:\n"
            "  - read\n"
            "delegates:\n"
            "  - worker\n"
            "---\n"
            "Work 2.",
            encoding="utf-8")
        self._edit_worker(pkg_dir, "delegates: []", "delegates:\n  - worker2")

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            verify.verify_package(pkg_dir)
        self.assertIn("cycle", str(ctx.exception).lower())

    def _chain_manifest(self, pkg_dir, length):
        """Extend the package into a skill -> a1 -> ... -> a<length> chain."""
        names = [f"chain{i}" for i in range(1, length + 1)]
        for i, name in enumerate(names):
            next_agent = [names[i + 1]] if i + 1 < len(names) else []
            delegates_yaml = ("delegates:\n" + "\n".join(
                f"  - {d}"
                for d in next_agent)) if next_agent else "delegates: []"
            (pkg_dir / "agents" / f"{name}.md").write_text(
                f"---\n"
                f"name: {name}\n"
                f"description: Chain link {name}\n"
                f"tools:\n"
                f"  - read\n"
                f"{delegates_yaml}\n"
                f"accepts: schemas/in.json\n"
                f"produces: schemas/out.json\n"
                f"---\n"
                f"Chain work {name}.",
                encoding="utf-8")

        # The skill fixture references {{agent:worker}}, which this chain
        # replaces; repoint it at the head of the chain.
        self._delegate_to(pkg_dir, names[0])
        return pkg_dir

    def test_delegation_depth_limit(self):
        """Levels of work below the skill, which is level 0 and is not counted."""
        pkg_dir = self._chain_manifest(self._create_valid_package(),
                                       package_checks.MAX_DELEGATION_DEPTH)
        verify.verify_package(pkg_dir)
        pkg_dir = self._chain_manifest(pkg_dir,
                                       package_checks.MAX_DELEGATION_DEPTH + 1)
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            verify.verify_package(pkg_dir)
        self.assertIn("levels below skill entry", str(ctx.exception))
        # The harness's nesting limit must match, or a package that builds
        # here fails at the point of delegation there.
        specs = dist_assets.harness_install_specs(["entry"], "pkg")
        env = specs["claude-code"]["json_merges"][0]["value"]["env"]
        self.assertEqual(env["CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH"],
                         str(package_checks.MAX_DELEGATION_DEPTH))

    def test_frontmatter_name_mismatch_rejection(self):
        pkg_dir = self._create_valid_package()
        (pkg_dir / "agents" / "worker.md").write_text(
            "---\nname: invalid\ndescription: Invalid worker\n---\nWork.",
            encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            verify.verify_package(pkg_dir)
        self.assertIn(
            "frontmatter name 'invalid' does not match expected agent name "
            "'worker'", str(ctx.exception))

    def test_verify_package_rejects_unpinned_dependencies(self):
        pkg_dir = self._create_valid_package()
        (pkg_dir / "requirements-client.txt").write_text("pydantic\n",
                                                         encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            verify.verify_package(pkg_dir)
        self.assertIn("Unpinned package in 'requirements-client.txt'",
                      str(ctx.exception))

        (pkg_dir / "requirements-client.txt").write_text(
            "pydantic; python_version >= '3.10'\n", encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            verify.verify_package(pkg_dir)
        self.assertIn("Unpinned package in 'requirements-client.txt'",
                      str(ctx.exception))

    def test_verify_package_rejects_scattered_requirements(self):
        pkg_dir = self._create_valid_package()
        sub_dir = pkg_dir / "subagent" / "references"
        sub_dir.mkdir(parents=True)
        (sub_dir / "requirements.txt").write_text("torch\n", encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            verify.verify_package(pkg_dir)
        self.assertIn("Scattered requirements file found", str(ctx.exception))
        self.assertIn("subagent/references/requirements.txt",
                      str(ctx.exception))

    def test_verify_package_rejects_missing_requirements_file_when_deps_declared(
            self):
        pkg_dir = self._create_valid_package()
        self._declare_dependencies(pkg_dir, host=["pydantic"])
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            verify.verify_package(pkg_dir)
        self.assertIn(
            "host_dependencies but 'requirements-client.txt' is missing",
            str(ctx.exception))

        (pkg_dir / "requirements-client.txt").write_text("pydantic>=2.0.0\n",
                                                         encoding="utf-8")
        self._declare_dependencies(pkg_dir, host=["pydantic"], tpu=["torch"])
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            verify.verify_package(pkg_dir)
        self.assertIn("tpu_dependencies but 'requirements-tpu.txt' is missing",
                      str(ctx.exception))

    def test_verify_package_rejects_undeclared_catalog_dependency(self):
        pkg_dir = self._create_valid_package()
        (pkg_dir / "requirements-client.txt").write_text("pydantic>=2.0.0\n",
                                                         encoding="utf-8")
        self._declare_dependencies(pkg_dir, host=["unknown-lib"])
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            verify.verify_package(pkg_dir)
        self.assertIn("Undeclared host dependency: 'unknown-lib'",
                      str(ctx.exception))

    def test_a_catalog_entry_nothing_declares_is_warned_about(self):
        """It never ships: a package nothing uses, or a declaration somebody
        forgot."""
        pkg_dir = self._create_valid_package()
        (pkg_dir / "requirements-client.txt").write_text(
            "pydantic>=2.0.0\nextra-host>=1.0.0\n", encoding="utf-8")
        (pkg_dir / "requirements-tpu.txt").write_text(
            "torch>=2.4.0\nextra-tpu>=1.0.0\n", encoding="utf-8")
        self._declare_dependencies(pkg_dir, host=["pydantic"], tpu=["torch"])
        manifest = package_loader.load_package(pkg_dir)

        warnings = dist_assets.undeclared_catalog_entries(pkg_dir, manifest)

        self.assertEqual(len(warnings), 2, warnings)
        self.assertIn("requirements-client.txt lists extra-host", warnings[0])
        self.assertIn("requirements-tpu.txt lists extra-tpu", warnings[1])

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

    def test_matchers_on_an_event_with_nothing_to_match_are_rejected(self):
        """Only a tool or a subagent can be matched. Matchers on any other
        event used to leave the compiler reusing the previous hook's matcher,
        and the runtime then skipped every fire of the hook: it never ran,
        and nothing said so."""
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

    def test_unclassified_harness_cannot_build_a_restricted_skill(self):
        skill = {"name": "entry", "tools": ["read"]}
        with self.assertRaisesRegex(utils.TpuAgentSuiteError,
                                    "SKILL_TOOL_RESTRICTION"):
            hook_config.skill_tools_enforced(skill, "no-such-harness")

    def test_unknown_harness_delegation_is_a_build_error(self):
        """A harness with no declared mechanism must fail the build rather than
        ship routing instructions its runtime cannot honor."""
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            harness_rules.delegation_tool_for("no-such-harness", ["worker"])
        self.assertIn("no delegation mechanism", str(ctx.exception))

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

    def _run_hook_command(self, command: str, home: Path):
        """Run a compiled hook command as a harness would: in a shell, with
        an empty payload on stdin, `~` pointing at `home`."""
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
                # Per-session hook state lives at sessions/<id>.contexts/<name>.json,
                # so "a:b" and "a_b" would share one file.
            ("does not match", [{
                **pre, "name": "bad:name",
                "command": "true"
            }]),
                # start-run rebinds the session to a fresh run namespace; per tool
                # call it would orphan the artifacts of the run it replaced.
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

    def _package_with_memory_block(self, entry: dict) -> Path:
        """Build a valid package carrying one workspace_memory declaration."""
        pkg_dir = self._create_valid_package()
        manifest = self._read_manifest(pkg_dir)
        manifest["workspace_memory"] = [entry]
        self._write_manifest(pkg_dir, manifest)
        (pkg_dir / "memory").mkdir(parents=True, exist_ok=True)
        return pkg_dir

    def test_workspace_memory_requires_existing_content_file(self):
        """A block naming a missing content file fails verification."""
        pkg_dir = self._package_with_memory_block({
            "marker":
            "kb",
            "content":
            "memory/absent.md",
        })

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            package_loader.load_package(pkg_dir)
        self.assertIn("workspace memory content", str(ctx.exception))

    def test_workspace_memory_rejects_workspace_resources(self):
        """Memory blocks load from arbitrary cwds, so only central resources may appear."""
        pkg_dir = self._package_with_memory_block({
            "marker":
            "kb",
            "content":
            "memory/contract.md",
        })
        (pkg_dir / "memory" / "contract.md").write_text(
            "Config at .tpu-agent-suite/environment.json.", encoding="utf-8")

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            translators.translate(pkg_dir, "claude-code",
                                  self.temp_dir / "dist_bad_asset")
        self.assertIn("central resources only", str(ctx.exception))

    def test_memory_scope_preamble_names_skills_not_package(self):
        """The preamble scopes by skill names, read as an English list, and
        never by the package name, which no harness recognizes."""

        def preamble(*names):
            return dist_assets.memory_scope_preamble({
                "name":
                "some-pkg",
                "skills": [{
                    "name": n
                } for n in names]
            })

        for names, expected in (
            (("a", ), "inside the `a` skill or its agents"),
            (("a", "b"), "inside the `a` or `b` skills or their agents"),
            (("a", "b", "c"), "inside the `a`, `b`, or `c` skills"),
        ):
            with self.subTest(names=names):
                text = preamble(*names)
                self.assertIn(expected, text)
                self.assertNotIn("some-pkg", text)
                self.assertNotIn("package", text)

    def test_skill_name_cannot_escape_package(self):
        """A skill is named, not located: a path in `skills` is rejected even
        when a skill document really exists where that path points."""
        pkg_dir = self._create_valid_package()
        entry = (pkg_dir / "skills" / "entry" / "SKILL.md").read_text()
        (pkg_dir / "outside").mkdir()
        (pkg_dir / "outside" / "SKILL.md").write_text(entry)
        manifest = self._read_manifest(pkg_dir)
        manifest["skills"] = ["../outside"]
        self._write_manifest(pkg_dir, manifest)
        with self.assertRaisesRegex(utils.TpuAgentSuiteError, r"skills/0"):
            package_loader.load_package(pkg_dir)

    def test_removed_manifest_fields_are_rejected_by_name(self):
        """A field package.release.json once carried fails with where it lives now,
        not as an anonymous unknown property."""
        pkg_dir = self._create_valid_package()
        original = self._read_manifest(pkg_dir)
        values = {
            "entrypoint": "skills/entry/SKILL.md",
            "agents": ["agents/worker.md"],
            "include": ["fragments/tools.json"],
            "schemas": ["schemas/in.json"],
            "host_dependencies": ["pydantic"],
            "tpu_dependencies": ["torch"],
            "required_capabilities": ["subagents"],
        }
        self.assertEqual(set(values),
                         set(package_checks.REMOVED_MANIFEST_FIELDS))
        for field, value in values.items():
            with self.subTest(field=field):
                self._write_manifest(pkg_dir, {**original, field: value})
                with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
                    package_loader.load_package(pkg_dir)
                self.assertIn("no longer supported", str(ctx.exception))
                self.assertIn(f"  {field}: ", str(ctx.exception))

    def test_document_driven_agent_missing_frontmatter_rejected(self):
        """Document-driven agent without leading frontmatter is rejected."""
        pkg_dir = self._create_valid_package()
        (pkg_dir / "agents").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "agents" / "bad.md").write_text("# Just Markdown",
                                                   encoding="utf-8")

        self._delegate_to(pkg_dir, "bad")

        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            verify.verify_package(pkg_dir)
        self.assertIn("must start with '---' YAML frontmatter",
                      str(ctx.exception))


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
