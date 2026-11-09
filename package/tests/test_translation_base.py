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
"""Harness-agnostic translation invariants, shared by every per-harness suite.

Anything asserted here holds for *every* supported harness: the build marker,
the install plan skeleton, the runtime snapshot, the bundled schemas, and the
rule that no raw instruction source leaks into a distribution. Anything that
differs between harnesses -- file extensions, frontmatter dialect, plugin
metadata layout, tool vocabulary -- belongs in the per-harness suite instead.
"""

from __future__ import annotations

import fnmatch
import pathlib
import shutil
import subprocess
import sys
import tempfile

try:
    import tomllib
except ImportError:
    tomllib = None

# Ensure repository root directory is in sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import jsonio, utils  # noqa: E402
from framework.builder import package_discovery  # noqa: E402
from framework.builder import package_loader  # noqa: E402
from framework.builder import dist_assets, translators  # noqa: E402

Path = pathlib.Path
PACKAGE_DIR = REPO_ROOT / "package"

# Every agent reached from the released skills. Hard-coded so that dropping
# an agent fails loudly instead of quietly shrinking the suite. 7 specialists
# plus the agent built from the `tpu-agent` skill, which is emitted as a
# selectable agent as well as a skill so `--agent` can reach it.
EXPECTED_AGENT_COUNT = 8

# Imported, not restated: `copy_asset` hands exactly these patterns to
# `shutil.ignore_patterns`, so the exclusion and its test cannot drift apart.
FORBIDDEN_NAME_PATTERNS = package_discovery.DIST_EXCLUDED_NAMES

# Top-level directories every translator emits regardless of package content.
# Hard-coded on purpose: these come from `translate`, not from package.release.json, so
# one disappearing is a framework regression that should fail loudly. Package
# directories are derived instead -- see `expected_top_dirs`.
FRAMEWORK_TOP_DIRS = frozenset({
    "agents",
    "docs",
    "environment",
    "framework",
    "schemas",
    "skills",
})


class TranslationParityBase:
    """Invariants every harness distribution must satisfy.

    Subclasses set the class attributes below and mix this in alongside
    `unittest.TestCase`.
    """

    HARNESS: str = ""
    # `compatibility_target` and `version_range` as declared in package.release.json.
    TARGET_ID: str = ""
    VERSION_RANGE: str = ""
    # None for harnesses that declare no version probe (e.g. antigravity).
    VERSION_COMMAND: list[str] | None = None
    # Install destinations for the skills and the agent directory.
    SKILL_INSTALL_DEST: str = ""
    AGENTS_INSTALL_DEST: str = ""
    # Top-level directories this harness emits beyond `FRAMEWORK_TOP_DIRS` --
    # plugin metadata dirs and, for Claude, the `project/` mirror.
    EXTRA_TOP_DIRS: frozenset[str] = frozenset()

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.temp_dir = Path(
            tempfile.mkdtemp(prefix=f"torchtpu_test_{cls.HARNESS}_"))
        cls.dist_dir = cls.temp_dir / f"dist_{cls.HARNESS}"
        translators.translate(
            package_dir=PACKAGE_DIR,
            harness=cls.HARNESS,
            output=cls.dist_dir,
            include_internal=False,
        )
        cls.raw_manifest = jsonio.load_json(PACKAGE_DIR /
                                            "package.release.json")
        cls.manifest = package_loader.resolve_manifest_composition(
            PACKAGE_DIR, cls.raw_manifest)
        cls.pkg_name = cls.manifest["name"]
        cls.pkg_tag = f"{cls.pkg_name}@{cls.manifest.get('version', '')}"
        cls.entry_name = cls.manifest["skills"][0]["name"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
        super().tearDownClass()

    @classmethod
    def expected_top_dirs(cls) -> set[str]:
        """Top-level directories the build should contain, derived from the manifest.

        Only the framework-owned names are fixed; every package directory is
        read back out of the manifest, because that is what actually decides
        which ones get emitted. `copy_runtime_assets` copies each agent's
        declared `scripts`/`references` and each `copy` entry at its
        package-relative path, so the first component of each is a top-level
        directory in the distribution. Deriving them means adding a package
        needs no edit here.
        """
        dirs = set(FRAMEWORK_TOP_DIRS) | set(cls.EXTRA_TOP_DIRS)
        for agent in cls.manifest.get("agents", []):
            for rel in (*agent.get("scripts", []),
                        *agent.get("references", [])):
                dirs.add(Path(rel).parts[0])
        for rel in cls.manifest.get("copy", []):
            dirs.add(Path(rel).parts[0])
        # A hook's script is copied at its package-relative path too, so the
        # directory holding it is a top-level directory in the build. Derived
        # for the same reason as the rest: adding a hook should not need an
        # edit here.
        for hook in cls.manifest.get("hooks", []):
            action = str(hook.get("action", ""))
            if action.startswith("script:"):
                rel = action.split(":", 1)[1]
                if Path(rel).parts:
                    dirs.add(Path(rel).parts[0])
        if cls.manifest.get("hooks"):
            if cls.HARNESS == "codex":
                dirs.add(".codex")
            elif cls.HARNESS == "cursor":
                dirs.add(".cursor")
        return dirs

    def agent_files(self, agent_name: str) -> list[Path]:
        """Every file this harness emits for one agent.

        Claude emits two (plugin and project layouts); codex and antigravity
        emit one each.
        """
        raise NotImplementedError

    def skill_files(self) -> list[Path]:
        """Every document this harness emits for the first released skill."""
        return [self.dist_dir / "skills" / self.entry_name / "SKILL.md"]

    def test_build_metadata(self):
        """tpu-agent-suite-build.json pins the framework, source, and harness target."""
        build = jsonio.load_json(self.dist_dir / "tpu-agent-suite-build.json")
        self.assertEqual(build["version"], utils.TPU_AGENT_SUITE_VERSION)
        self.assertEqual(build["source"], self.manifest["name"])
        self.assertEqual(build["harness"], self.HARNESS)
        self.assertEqual(build["compatibility_target"], self.TARGET_ID)
        self.assertEqual(build["version_range"], self.VERSION_RANGE)
        self.assertTrue(build["generated"])
        self.assertFalse(build["include_internal"])

    def test_install_plan_skeleton(self):
        """The install plan names this harness, its version gate, and its copies."""
        plan = jsonio.load_json(self.dist_dir / "tpu-agent-suite-install.json")
        self.assertEqual(plan["harness"], self.HARNESS)
        self.assertEqual(plan["version_range"], self.VERSION_RANGE)
        self.assertEqual(plan.get("version_command"), self.VERSION_COMMAND)
        # One package per workspace, so its files sit directly in
        # .tpu-agent-suite/, not under the package's name.
        self.assertEqual(plan["environment_path"],
                         ".tpu-agent-suite/environment.json")
        self.assertEqual(plan["ownership_path"],
                         ".tpu-agent-suite/ownership.json")
        # Every harness emits the key; the target file is asserted per harness.
        self.assertIsInstance(plan["markdown_merges"], list)

        by_dest = {c["destination"]: c for c in plan["copies"]}
        for expected, store in (
            (self.SKILL_INSTALL_DEST, "workspace"),
            (self.AGENTS_INSTALL_DEST, "workspace"),
                # framework/ holds the installer too, so an install can be
                # removed without its distribution.
            ("framework", "central"),
            ("docs", "central"),
            ("schemas", "central"),
            ("tpu-agent-suite-build.json", "central"),
        ):
            self.assertIn(expected, by_dest)
            self.assertEqual(by_dest[expected]["store"], store)
        for entry in plan["copies"]:
            self.assertIn(entry["store"], ("workspace", "central"))

    def test_every_memory_block_opens_with_the_scope_preamble(self):
        """A memory block loads in every session, so each one first says it
        applies only inside the package's skills."""
        plan = jsonio.load_json(self.dist_dir / "tpu-agent-suite-install.json")
        preamble = dist_assets.memory_scope_preamble(self.manifest)
        for block in plan["markdown_merges"]:
            with self.subTest(marker=block["marker"]):
                self.assertTrue(block["content"].startswith(preamble + "\n\n"))
                self.assertEqual(block["content"].count(preamble), 1)

    def test_skill_present_and_compiled(self):
        """Skill documents exist and carry the installed package contract."""
        for path in self.skill_files():
            self.assertTrue(path.is_file(), f"Missing skill doc {path}")
            content = path.read_text(encoding="utf-8")
            self.assertTrue(
                content.startswith(f"---\nname: {self.entry_name}\n"),
                f"{path} is missing its name frontmatter",
            )
            self.assertIn("## Installed package contract", content)

    def test_every_agent_is_emitted(self):
        """Each declared agent produces the file(s) this harness expects."""
        agents = self.manifest["agents"]
        self.assertEqual(
            len(agents), EXPECTED_AGENT_COUNT,
            f"Expected {EXPECTED_AGENT_COUNT} agents reached from the "
            f"released skills, found {len(agents)}")
        for agent in agents:
            for path in self.agent_files(agent["name"]):
                self.assertTrue(path.is_file(),
                                f"Missing agent artifact {path}")
                content = path.read_text(encoding="utf-8")
                if path.suffix == ".toml" and tomllib is not None:
                    # TOML escapes quotes, so compare the parsed value.
                    content = tomllib.loads(content).get("description", "")
                self.assertIn(
                    agent["description"], content,
                    f"{path} does not carry the declared description")

    def test_execution_contract_in_every_agent(self):
        """Every agent prompt carries the runtime contract, on every harness."""
        for agent in self.manifest["agents"]:
            for path in self.agent_files(agent["name"]):
                content = path.read_text(encoding="utf-8")
                self.assertIn("## Generated execution contract", content)
                self.assertIn("~/.tpu-agent-suite/runs/<run_id>/artifacts/",
                              content)

    def test_bundled_schemas_are_valid(self):
        """Declared schemas are bundled and structurally valid."""
        for schema_rel in self.manifest["schemas"]:
            schema_file = self.dist_dir / schema_rel
            self.assertTrue(schema_file.is_file(),
                            f"Missing bundled schema {schema_rel}")
            errors = utils.validate_schema_definition(
                jsonio.load_json(schema_file))
            self.assertEqual(errors, [],
                             f"Invalid bundled schema {schema_rel}: {errors}")

    def test_bundled_environment_and_copies(self):
        """Environment questions and every declared copy target are bundled."""
        env_questions = (self.dist_dir /
                         self.manifest["environment"]["questions"])
        self.assertTrue(env_questions.is_file(),
                        "Missing bundled environment questions")
        for copy_item in self.manifest.get("copy", []):
            self.assertTrue((self.dist_dir / copy_item).exists(),
                            f"Missing bundled copy item: {copy_item}")

    def test_wiki_pages_are_bundled_at_their_index_paths(self):
        source = PACKAGE_DIR / "data" / "tpu-knowledge-base" / "pages"
        pages = sorted(source.rglob("*.md"))
        # A directory that is not there globs to nothing, and a loop over
        # nothing passes.
        self.assertTrue(pages, f"no pages under {source}")
        for page in pages:
            relative = page.relative_to(PACKAGE_DIR)
            with self.subTest(page=str(relative)):
                bundled = self.dist_dir / relative
                self.assertTrue(bundled.is_file())
                self.assertEqual(bundled.read_bytes(), page.read_bytes())

    def test_runtime_snapshot_executes(self):
        """The bundled runtime runs hermetically from an unrelated CWD."""
        runtime_py = self.dist_dir / "framework" / "runtime.py"
        self.assertTrue(runtime_py.is_file(), "Missing framework/runtime.py")
        self.assertTrue((self.dist_dir / "framework" / "utils.py").is_file(),
                        "Missing framework/utils.py")
        # The build tooling stays in the repository.
        for build_only in ("compiler.py", "builder", "lint.py",
                           "frontmatter.py", "definitions", "templates",
                           "tests", "schemas/package-manifest.schema.json",
                           "schemas/agent-frontmatter.schema.json"):
            self.assertFalse(
                (self.dist_dir / "framework" / build_only).exists(),
                build_only)
        proc = subprocess.run(
            [sys.executable, str(runtime_py), "--help"],
            capture_output=True,
            text=True,
            cwd=str(self.temp_dir),
            check=True,
        )
        for command in ("start-run", "validate"):
            self.assertIn(command, proc.stdout)

    def test_knowledge_base_data_holds_only_what_may_ship(self):
        """`data/tpu-knowledge-base` ships whole, so everything in it is shipped.

        The golden answers and the checkers must never reach an agent, which
        would be marking its own work. They live under `tests/`; this fails if
        anything but the four data directories appears beside them.
        """
        source = PACKAGE_DIR / "data" / "tpu-knowledge-base"
        expected = {"index", "pages", "prompts", "templates"}
        self.assertEqual({p.name for p in source.iterdir()}, expected)
        shipped = self.dist_dir / "data" / "tpu-knowledge-base"
        self.assertEqual({p.name for p in shipped.iterdir()}, expected)
        for name in ("golden.jsonl", "answers.jsonl", "check_entry.py"):
            self.assertEqual(list(self.dist_dir.rglob(name)), [], name)

    def test_the_shipped_readme_names_what_the_build_holds(self):
        """It said the build contains the knowledge base, which no build
        does, and named a directory by an old path."""
        readme = (self.dist_dir / "README.md").read_text(encoding="utf-8")
        # An agent's description may mention a knowledge base; what the
        # build holds must not be called one.
        self.assertNotIn("optimization knowledge base", readme)
        self.assertNotIn("performance knowledge base", readme)
        self.assertIn("`data/tpu-knowledge-base/pages/`", readme)
        self.assertTrue(
            (self.dist_dir / "data" / "tpu-knowledge-base" / "pages").is_dir())

    def test_docs_named_by_the_readme_are_shipped(self):
        """Each doc the README names is in docs/, byte-for-byte the repo copy."""
        readme = (self.dist_dir / "README.md").read_text(encoding="utf-8")
        repo_docs = PACKAGE_DIR.parent / "docs"
        for doc in dist_assets.DISTRIBUTION_DOCS:
            self.assertIn(f"`docs/{doc}`", readme)
            shipped = self.dist_dir / "docs" / doc
            self.assertTrue(shipped.is_file(), f"docs/{doc} missing")
            self.assertEqual(shipped.read_bytes(),
                             (repo_docs / doc).read_bytes())

    def test_requirements_carry_the_declared_dependencies(self):
        """The shipped catalogs are trimmed to what the built units declare in
        frontmatter, never to nothing: a build with no dependencies at all is
        a package that forgot to declare them."""
        host = (self.dist_dir / "requirements-client.txt").read_text()
        tpu = (self.dist_dir / "requirements-tpu.txt").read_text()
        for dep in self.manifest["host_dependencies"]:
            self.assertIn(dep, host, f"{dep} missing from requirements-client")
        for dep in self.manifest["tpu_dependencies"]:
            self.assertIn(dep, tpu, f"{dep} missing from requirements-tpu")
        self.assertIn("pyyaml", host)
        self.assertIn("torch", tpu)

    def test_top_level_directory_tree(self):
        """Exactly the expected top-level directories -- no more, no fewer.

        Equality rather than per-name containment: the `copy` rule is
        directory-wide, so an over-broad entry pulling an unintended directory
        into the build is a real failure mode and only the `no more` half
        catches it.
        """
        actual_top_dirs = {
            p.name
            for p in self.dist_dir.iterdir() if p.is_dir()
        }
        expected = self.expected_top_dirs()
        self.assertEqual(
            actual_top_dirs, expected,
            f"Top-level directory mismatch in the {self.HARNESS} build: "
            f"{actual_top_dirs ^ expected}")

    def test_no_forbidden_artifacts(self):
        """No bytecode, VCS debris, or raw instruction sources in the build.

        Checks each path component against the same `fnmatch` patterns
        `copy_asset` excludes, rather than substring-matching the whole path:
        that is the guarantee the copy rule actually makes, and a substring
        test both misses `foo.pyc.bak` and trips on a legitimate
        `instructions_for_humans.md`.
        """
        for path in self.dist_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.dist_dir)
            for part in rel.parts:
                for pattern in FORBIDDEN_NAME_PATTERNS:
                    self.assertFalse(
                        fnmatch.fnmatch(part, pattern),
                        f"Forbidden artifact leaked into the {self.HARNESS}"
                        f" distribution: {rel} (component {part!r} matches"
                        f" excluded pattern {pattern!r})")

    def test_knowledgebase_is_not_installed(self):
        """The knowledge base left `include`, so no harness may ship it.

        Asserted per harness rather than on the manifest: dropping the
        subpackage removes its agents, but the `copy` entry it declared is
        what actually placed the tree in the payload, and only the built
        distribution shows whether anything still does.
        """
        offenders = [
            str(path.relative_to(self.dist_dir))
            for path in self.dist_dir.rglob("*")
            if "knowledgebase" in path.name
        ]
        self.assertEqual(
            offenders, [],
            f"knowledgebase leaked into the {self.HARNESS} distribution:"
            f" {offenders}")

    def test_no_unexpanded_template_macros(self):
        """No {{runtime}} or {{agent:...}} macros survive unexpanded into the distribution."""
        for path in self.dist_dir.rglob("*.md"):
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8")
            rel = path.relative_to(self.dist_dir)
            self.assertNotIn(
                "{{runtime}}",
                content,
                f"Unexpanded {{{{runtime}}}} macro leaked into {self.HARNESS} distribution: {rel}",
            )
            self.assertNotIn(
                "{{agent:",
                content,
                f"Unexpanded {{{{agent:...}}}} macro leaked into {self.HARNESS} distribution: {rel}",
            )

    def test_shared_critical_files(self):
        """Files every harness distribution must contain."""
        for rel in (
                "tpu-agent-suite-build.json",
                "tpu-agent-suite-install.json",
                "requirements-client.txt",
                "requirements-tpu.txt",
                f"skills/{self.entry_name}/SKILL.md",
                "framework/runtime.py",
                "framework/utils.py",
                "framework/installer.py",
                "schemas/environment.json",
                "environment/questions.json",
                "references/tpu-model-onboarding/torch-benchmark-coder/causal_lm.md",
                "scripts/tpu-model-onboarding/benchmark_tpu.py",
        ):
            self.assertTrue(
                (self.dist_dir / rel).is_file(),
                f"Critical file missing from {self.HARNESS} build: {rel}")


def parse_yaml_frontmatter(content: str) -> dict[str, str]:
    """Return the raw key/value lines of a leading `---` YAML block.

    Deliberately not a YAML parser: values stay as written so a test can assert
    on exact formatting (quoting, list style) that a parser would normalise away.
    Nested list items are collected under their parent key, newline-joined.
    """
    if not content.startswith("---\n"):
        raise AssertionError("content has no leading frontmatter block")
    body = content[4:]
    end = body.find("\n---\n")
    if end == -1:
        raise AssertionError("frontmatter block is not terminated")
    fields: dict[str, str] = {}
    current: str | None = None
    for line in body[:end].split("\n"):
        if line.startswith("  ") and current is not None:
            fields[current] = (fields[current] + "\n" + line.strip()).strip()
        elif ":" in line:
            key, _, value = line.partition(":")
            current = key.strip()
            fields[current] = value.strip()
    return fields
