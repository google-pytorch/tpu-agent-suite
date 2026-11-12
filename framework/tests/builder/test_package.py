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
"""Tests for package loading, requirements and verification."""

import json
import pathlib
import sys
import unittest

# Imports below follow this sys.path setup, so the file runs directly.
# ruff: noqa: E402
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import compiler, utils
from framework.builder import (dist_assets, package_checks, package_loader,
                               translators, verify)
from framework.tests.builder import helpers

Path = pathlib.Path


class TestManifestComposition(helpers.CompilerTestCase):
    """Second manifests, `extends`, and strict or removed fields."""

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

    def test_removed_manifest_fields_are_rejected_by_name(self):
        """A field package.release.json once carried says where it lives now.

        It fails with that location, not as an anonymous unknown property.
        """
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


class TestDocumentDrivenLoading(helpers.CompilerTestCase):
    """Agents and skills loaded from their markdown documents."""

    def test_skill_name_cannot_escape_package(self):
        """A skill is named, not located: a path in `skills` is rejected.

        This holds even when a skill document really exists where that path
        points.
        """
        pkg_dir = self._create_valid_package()
        entry = (pkg_dir / "skills" / "entry" / "SKILL.md").read_text()
        (pkg_dir / "outside").mkdir()
        (pkg_dir / "outside" / "SKILL.md").write_text(entry)
        manifest = self._read_manifest(pkg_dir)
        manifest["skills"] = ["../outside"]
        self._write_manifest(pkg_dir, manifest)
        with self.assertRaisesRegex(utils.TpuAgentSuiteError, r"skills/0"):
            package_loader.load_package(pkg_dir)

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


class TestUnitLayout(helpers.CompilerTestCase):
    """skills/<name>/ and agents/ may not contain subdirectories."""

    def test_load_rejects_a_subdirectory_in_a_skill_or_agents(self):
        pkg_dir = self._create_valid_package()
        for nested in (pkg_dir / "skills" / "entry" / "references",
                       pkg_dir / "agents" / "group"):
            nested.mkdir()
            (nested / "doc.md").write_text("# Doc\n", encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            package_loader.load_package(pkg_dir)
        message = str(ctx.exception)
        self.assertIn("may not contain subdirectories", message)
        self.assertIn("  skills/entry/references/", message)
        self.assertIn("  agents/group/", message)


class TestPackageRequirements(helpers.CompilerTestCase):
    """Requirements files and catalog dependencies."""

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
        """It never ships: an unused package or a forgotten declaration."""
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

    def test_a_frontmatter_dependency_is_a_bare_name(self):
        pkg_dir = self._create_valid_package()
        (pkg_dir / "requirements-client.txt").write_text("pydantic>=2.0.0\n",
                                                         encoding="utf-8")
        (pkg_dir / "requirements-tpu.txt").write_text("torch>=2.4.0\n",
                                                      encoding="utf-8")
        self._declare_dependencies(pkg_dir, host=["pydantic"], tpu=["torch"])
        verify.verify_package(pkg_dir)
        for host, tpu, field in (
            (["pydantic>=2"], [], "$/host_dependencies/0"),
            (["pydantic[email]"], [], "$/host_dependencies/0"),
            (["pydantic"], ["torch==2.4.0"], "$/tpu_dependencies/0"),
        ):
            with self.subTest(host=host, tpu=tpu):
                self._declare_dependencies(pkg_dir, host=host, tpu=tpu)
                with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
                    package_loader.load_package(pkg_dir)
                message = str(ctx.exception)
                self.assertIn("invalid YAML frontmatter in agent document",
                              message)
                self.assertIn(f"{field}: string does not match", message)


class TestWorkspaceMemory(helpers.CompilerTestCase):
    """Workspace memory blocks."""

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
        """The preamble scopes by skill names, read as an English list.

        It never scopes by the package name, which no harness recognizes.
        """

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


class TestToolDeclarations(helpers.CompilerTestCase):
    """Tool names, action levels, visibility and references."""

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


class TestDelegationGraph(helpers.CompilerTestCase):
    """Delegation cycles and depth limits."""

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


class TestFrontmatterSchema(helpers.CompilerTestCase):
    """Frontmatter parsing and schema checks."""

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

    def test_each_kind_takes_only_its_own_frontmatter_fields(self):
        """Each kind of entry accepts only its own frontmatter fields.

        A skill can't declare `accepts`, a reference can't declare `tools`,
        and an agent needs a `description`.
        """
        skill = Path("skills") / "entry" / "SKILL.md"
        worker = Path("agents") / "worker.md"
        cases = (
            (skill, "description: Entry description\n",
             "description: Entry description\naccepts: schemas/in.json\n",
             "skill document", "accepts"),
            (worker, "delegates: []\n",
             "delegates: []\nreferences:\n  - references/r.md\n",
             "reference '", "tools"),
            (worker, "description: Worker\n", "", "agent document 'worker.md'",
             "description"),
        )
        pkg_dir = self._create_valid_package()
        (pkg_dir / "references").mkdir(exist_ok=True)
        (pkg_dir / "references" / "r.md").write_text(
            "---\ntools:\n  - read\n---\nR.\n", encoding="utf-8")
        for rel, old, new, label, field in cases:
            with self.subTest(field=field):
                path = pkg_dir / rel
                text = path.read_text(encoding="utf-8")
                self.assertIn(old, text)
                path.write_text(text.replace(old, new, 1), encoding="utf-8")
                try:
                    with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
                        package_loader.load_package(pkg_dir)
                finally:
                    path.write_text(text, encoding="utf-8")
                message = str(ctx.exception)
                self.assertIn("invalid YAML frontmatter in " + label, message)
                self.assertIn(field, message)


if __name__ == "__main__":
    unittest.main()
