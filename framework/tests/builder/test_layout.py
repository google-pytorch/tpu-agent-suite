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
"""The layout by kind: skills/, agents/, references/, scripts/.

A manifest names only its skills. Everything else -- agents, references,
scripts, schemas, dependencies -- is found by following frontmatter from
them. These tests build one small package in that layout and check what the
compiler derives from it.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
import unittest

from framework import lint, utils
from framework.builder import package_checks, package_loader, translators

Path = pathlib.Path


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_package(root: Path) -> Path:
    """A package with two skills, a delegating reference, and a bundle dir."""
    pkg = root / "layout-pkg"
    _write(
        pkg, "package.release.json",
        json.dumps({
            "name":
            "layout-pkg",
            "version":
            "1.0.0",
            "description":
            "Layout test package",
            "skills": ["main", "aux"],
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
        }))
    _write(pkg, "environment/questions.json", json.dumps({"questions": []}))
    _write(pkg, "schemas/env.json", json.dumps({"type": "object"}))
    _write(pkg, "schemas/in.json",
           json.dumps({
               "type": "object",
               "required": ["run_id"]
           }))
    _write(
        pkg, "schemas/out.json",
        json.dumps({
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["completed", "failed"]
                }
            },
        }))
    _write(pkg, "requirements-client.txt", "pyyaml>=6.0\n")
    _write(
        pkg, "skills/main/SKILL.md", "---\n"
        "name: main\n"
        "description: The primary skill.\n"
        "tools:\n  - read\n"
        "delegates:\n  - worker\n  - aux\n"
        "references:\n  - references/guide.md\n  - references/bundle\n"
        "scripts:\n  - scripts/tool.py\n"
        "---\n"
        "Delegate to {{agent:worker}} or hand off to {{skill:aux}}.\n"
        "Run `{{script:tool.py}}`; see `{{reference:guide.md}}`.\n")
    _write(
        pkg, "skills/aux/SKILL.md", "---\n"
        "name: aux\n"
        "description: A leaf skill that drives a runtime command.\n"
        "---\n"
        "Run {{runtime}} something --package {{package_tag}}.\n")
    _write(
        pkg, "agents/worker.md", "---\n"
        "name: worker\n"
        "description: Does the work.\n"
        "tools:\n  - read\n"
        "delegates: []\n"
        "accepts: schemas/in.json\n"
        "produces: schemas/out.json\n"
        "references:\n  - references/loop.md\n"
        "---\n"
        "Read `{{reference:loop.md}}` and follow it.\n")
    _write(
        pkg, "agents/helper.md", "---\n"
        "name: helper\n"
        "description: A helper the loop reference dispatches.\n"
        "tools:\n  - read\n"
        "delegates: []\n"
        "---\n"
        "Help.\n")
    _write(
        pkg, "agents/orphan.md", "---\n"
        "name: orphan\n"
        "description: Nothing reaches this one.\n"
        "tools:\n  - read\n"
        "delegates: []\n"
        "---\n"
        "Unreleased.\n")
    _write(
        pkg, "references/loop.md", "---\n"
        "delegates:\n  - helper\n"
        "scripts:\n  - scripts/helper.py\n"
        "host_dependencies:\n  - pyyaml\n"
        "---\n"
        "Loop: dispatch {{agent:helper}}, then run"
        " `{{script:helper.py}}`.\n")
    _write(
        pkg, "references/guide.md", "---\n"
        "scripts:\n  - scripts/tool.py\n"
        "references:\n  - references/bundle\n"
        "---\n"
        "Tool lives at {{script:tool.py}}; the bundle at"
        " {{reference:bundle/a.md}}.\n"
        "Running on {{harness}}.\n")
    _write(pkg, "references/bundle/a.md", "# A\n")
    _write(pkg, "scripts/tool.py", "print('tool')\n")
    _write(pkg, "scripts/helper.py", "print('helper')\n")
    return pkg


class _LayoutTest(unittest.TestCase):
    """Builds the package once per test; loads it and edits its files."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pkg = build_package(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _load(self, **kwargs):
        return package_loader.load_package(self.pkg, **kwargs)

    def _load_error(self, **kwargs) -> str:
        """Load the package, which must fail; return the error message."""
        with self.assertRaises(utils.TpuAgentSuiteError) as ctx:
            self._load(**kwargs)
        return str(ctx.exception)

    def _append(self, rel: str, text: str) -> None:
        path = self.pkg / rel
        path.write_text(path.read_text() + text)

    def _replace(self, rel: str, old: str, new: str, count: int = -1) -> None:
        path = self.pkg / rel
        path.write_text(path.read_text().replace(old, new, count))


class LayoutDiscoveryTest(_LayoutTest):

    def test_a_delegate_nothing_declares_fails_the_load(self):
        _write(self.pkg, "references/loop.md",
               "---\ndelegates:\n  - nonesuch\n---\nLoop.\n")
        self.assertIn("delegates to unknown agents", self._load_error())

    def test_skill_to_skill_delegation_is_warned_about(self):
        """Allowed, and the guide expects it to be rare. A skill built only
        because another delegates to it gets a second warning."""
        warnings = package_checks.delegation_warnings(self._load())
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn("skill 'main' delegates to skill 'aux'", warnings[0])

        warnings = package_checks.delegation_warnings(
            self._load(skills=["main"]))
        self.assertEqual(len(warnings), 2, warnings)
        self.assertIn("skill 'aux' is not one of the skills this build names",
                      warnings[1])
        self.assertIn("only because main delegates to it", warnings[1])

    def test_unreachable_files_are_reported_not_built(self):
        manifest = self._load()
        self.assertEqual(lint.unreachable_files(self.pkg, manifest),
                         ["agents/orphan.md"])

    def test_lint_is_clean_on_the_layout(self):
        manifest = self._load()
        findings = lint.run_checks(self.pkg,
                                   manifest,
                                   only=[
                                       "bare-paths", "delegation-consistency",
                                       "unused-declarations",
                                       "instruction-links",
                                       "frontmatter-contracts"
                                   ])
        self.assertEqual([str(f) for f in findings], [])


class LayoutTranslationTest(_LayoutTest):

    def setUp(self):
        super().setUp()
        self.dist = self.tmp / "dist"
        translators.translate(self.pkg, "claude-code", self.dist)

    def test_a_shipped_reference_is_rendered_like_a_prompt(self):
        """Frontmatter stripped, and package and harness tokens resolved."""
        loop = (self.dist / "references/loop.md").read_text()
        self.assertFalse(loop.startswith("---"))
        self.assertIn("Loop: dispatch", loop)
        guide = (self.dist / "references/guide.md").read_text()
        self.assertIn(
            "~/.tpu-agent-suite/packages/layout-pkg@1.0.0/scripts/tool.py",
            guide)
        self.assertIn(
            "~/.tpu-agent-suite/packages/layout-pkg@1.0.0/references/bundle/a.md",
            guide)
        self.assertIn("Running on claude-code.", guide)
        self.assertNotIn("{{", guide)

    def test_the_compute_setup_guide_ships_rendered_in_every_build(self):
        """Whatever the skills declare; framework/ gets no raw copy."""
        guide = (self.dist / "references/compute-setup.md").read_text()
        self.assertIn("AskUserQuestion every returned question", guide)
        self.assertNotIn("{{", guide)
        self.assertFalse(
            (self.dist / "framework/compute/compute-setup.md").exists())

    def test_a_package_file_named_by_path_fails_the_load(self):
        """A path typed out is read from the workspace, where it is not. The
        error names the token to write and, since the rule cannot tell a
        package file from the user's, how to write a workspace path."""
        cases = (
            ("agents/worker.md", "\nThen run `python3 scripts/helper.py`.\n",
             ("agent worker names package files by path",
              "scripts/helper.py -> {{script:helper.py}}",
              "./scripts/helper.py or <work>/scripts/helper.py")),
            ("references/guide.md", "See references/bundle/a.md.\n",
             ("reference references/guide.md names package files",
              "-> {{reference:bundle/a.md}}")),
            # Renders to the right place and is checked against nothing.
            ("agents/worker.md", "\nRun {{package}}/scripts/helper.py.\n",
             ("-> {{script:helper.py}}", )),
            # How a stale layout survives.
            ("agents/worker.md", "\nRun {{package}}/fam/scripts/helper.py.\n",
             ("fam/scripts/helper.py -> no token names it", )),
        )
        for rel, text, expected in cases:
            with self.subTest(text=text):
                original = (self.pkg / rel).read_text()
                self._append(rel, text)
                message = self._load_error()
                for part in expected:
                    self.assertIn(part, message)
                (self.pkg / rel).write_text(original)

    def test_skill_tools_are_checked_as_a_skill(self):
        """Delegating or not; the twin agent must not report it instead. A
        harness alias is pointed at its canonical tool."""
        cases = (
            ("skills/main/SKILL.md", "tools:\n  - read\n",
             "tools:\n  - read\n  - bogus\n", "skill 'main'", "bogus", None),
            ("skills/aux/SKILL.md", "description:",
             "tools:\n  - bash\ndescription:", "skill 'aux'", "bash", "shell"),
        )
        for rel, old, new, owner, tool, canonical in cases:
            with self.subTest(owner=owner):
                original = (self.pkg / rel).read_text()
                self._replace(rel, old, new, 1)
                message = self._load_error()
                self.assertIn(f"{owner} references undeclared tool '{tool}'",
                              message)
                if canonical:
                    self.assertIn(
                        f"did you mean canonical tool '{canonical}'?", message)
                else:
                    self.assertNotIn("did you mean", message)
                (self.pkg / rel).write_text(original)

    def test_a_file_declared_outside_its_root_fails_the_load(self):
        """A file declared elsewhere ships where no token can name it."""
        _write(self.pkg, "elsewhere/x.py", "print('x')\n")
        _write(self.pkg, "schemas/extra.json", "{}\n")
        cases = (
            ("agents/worker.md", "---\n",
             "---\nscripts:\n  - elsewhere/x.py\n", "'^scripts/'"),
            ("skills/aux/SKILL.md", "---\n",
             "---\nreferences:\n  - schemas/extra.json\n", "'^references/'"),
        )
        for rel, old, new, expected in cases:
            with self.subTest(rel=rel):
                original = (self.pkg / rel).read_text()
                self._replace(rel, old, new, 1)
                self.assertIn(expected, self._load_error())
                (self.pkg / rel).write_text(original)

    def test_an_unknown_token_kind_fails_the_load(self):
        """`{{scripts:x}}` matched no pattern, so it shipped as written; the
        built-in guard once matched [a-z_] only, so a typo slipped by."""
        cases = (
            ("\nRun `{{scripts:helper.py}}`.\n",
             ("{{scripts:helper.py}}", "did you mean {{script:...}}?")),
            ("\nRead `{{knowledgebase}}/wiki/concepts/x.md`.\n",
             ("retired {{knowledgebase}} token",
              "{{knowledgebase}}/wiki/concepts/x.md")),
            ("\nNow {{built-in:ask_the_user}} about it.\n",
             ("unknown resource tokens", )),
        )
        original = (self.pkg / "agents/worker.md").read_text()
        for text, expected in cases:
            with self.subTest(text=text):
                self._append("agents/worker.md", text)
                message = self._load_error()
                for part in expected:
                    self.assertIn(part, message)
                (self.pkg / "agents/worker.md").write_text(original)

    def _set_memory_block(self, text):
        _write(self.pkg, "memory/note.md", text)
        manifest_path = self.pkg / "package.release.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["workspace_memory"] = [{
            "marker": "note",
            "content": "memory/note.md"
        }]
        manifest_path.write_text(json.dumps(manifest))

    def test_a_memory_block_may_name_only_what_the_build_ships(self):
        self._set_memory_block("Ask {{agent:worker}}; run"
                               " `{{script:helper.py}}`.\n")
        self._load()
        _write(self.pkg, "scripts/unshipped.py", "print('no')\n")
        block = "workspace memory block 'note'"
        for text, expected in (
            ("Run `scripts/helper.py`.\n", (block, "{{script:helper.py}}")),
            ("Run `{{script:unshipped.py}}`.\n", (block, "unshipped.py")),
            ("Ask {{agent:orphan}}.\n", ("unknown agent: orphan", )),
        ):
            with self.subTest(text=text):
                self._set_memory_block(text)
                message = self._load_error()
                for part in expected:
                    self.assertIn(part, message)

    def test_data_and_schema_tokens_name_what_the_package_copies(self):
        _write(self.pkg, "data/store/rows.jsonl", "{}\n")
        self._append(
            "agents/worker.md", "\nRows: {{data:store/rows.jsonl}};"
            " shape: {{schema:in.json}}.\n")
        self.assertIn("package.release.json does not list it under `copy`",
                      self._load_error())
        manifest = json.loads((self.pkg / "package.release.json").read_text())
        manifest["copy"].append("data/store")
        (self.pkg / "package.release.json").write_text(json.dumps(manifest))
        self._load()
        out = self.pkg.parent / "dist-data"
        translators.translate(self.pkg, "claude-code", out)
        built = (out / "project/agents/worker.md").read_text()
        self.assertIn(
            "~/.tpu-agent-suite/packages/layout-pkg@1.0.0/data/store/rows.jsonl",
            built)
        self.assertIn(
            "~/.tpu-agent-suite/packages/layout-pkg@1.0.0/schemas/in.json",
            built)

    def test_every_skill_renders_and_only_delegating_ones_get_a_contract(self):
        """Frontmatter comes from the manifest, once, and tokens resolve."""
        main = (self.dist / "project/skills/main/SKILL.md").read_text()
        aux = (self.dist / "project/skills/aux/SKILL.md").read_text()
        self.assertIn("Installed package contract", main)
        self.assertIn("Delegate to worker or hand off to aux.", main)
        self.assertNotIn("Installed package contract", aux)
        self.assertTrue(aux.startswith("---\nname: aux\n"), aux)
        self.assertEqual(aux.count("name: aux"), 1)
        self.assertIn('description: "A leaf skill', aux)
        self.assertIn("framework/runtime.py something", aux)
        self.assertIn("--package layout-pkg@1.0.0", aux)
        self.assertNotIn("{{", aux)


if __name__ == "__main__":
    unittest.main()
