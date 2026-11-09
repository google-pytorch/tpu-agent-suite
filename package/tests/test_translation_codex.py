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
"""Codex translation parity for the tpu-agent-suite package.

Shared invariants live in `translation_base`. Codex differs from the other
harnesses in three ways worth pinning:

* agents are TOML documents, not markdown with frontmatter, and the prompt is
  carried in a `developer_instructions` triple-quoted string;
* there is no per-agent tool list. Capability is expressed as a `sandbox_mode`
  and an `approval_policy` derived from the declared tools' action levels, so
  the mapping from tools to sandbox tier is what this suite checks;
* configuration is TOML (`toml_sets`), not the JSON merge Claude Code uses.
"""

from __future__ import annotations

import pathlib
import sys

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_translation_base import PACKAGE_DIR  # noqa: E402
from test_translation_base import TranslationParityBase  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from framework import jsonio  # noqa: E402
from framework.builder import package_checks, package_loader  # noqa: E402

Path = pathlib.Path

SANDBOX_TIERS = ("read-only", "workspace-write", "danger-unrestricted")
APPROVAL_POLICIES = ("never", "on-demand", "always")


def capability_sets(
        declared_tools: dict[str, dict]) -> tuple[set[str], set[str]]:
    """Partition declared tools by the capability their `action_level` grants.

    Derived from the package's declared metadata rather than hardcoded, and
    deliberately *not* from `harness_rules.codex_sandbox_mode`. Importing that
    function would make every assertion below tautological -- it would agree
    with itself no matter how badly the tier logic were broken -- and there is
    no shared constant to import, because the producer decides from
    `action_level` fields rather than from a set of tool names.

    Restating the rule here is the point: it is the independent statement of
    what Codex's sandbox tiers are *supposed* to mean.
    """
    exec_like: set[str] = set()
    write_like: set[str] = set()
    for name, tool in declared_tools.items():
        level = tool.get("action_level", {})
        if level.get("scope") == "host" or level.get("effect") == "execute":
            exec_like.add(name)
        elif level.get("effect") in ("write", "delete"):
            write_like.add(name)
    return exec_like, write_like


class TestCodexTranslationParity(TranslationParityBase, unittest.TestCase):
    """Validates that package/ translates correctly to Codex."""

    HARNESS = "codex"
    TARGET_ID = "codex-0.147"
    VERSION_RANGE = ">=0.145.0,<0.160.0"
    VERSION_COMMAND = ["codex", "--version"]
    SKILL_INSTALL_DEST = ".agents/skills/tpu-agent"
    AGENTS_INSTALL_DEST = ".codex/agents"
    EXTRA_TOP_DIRS = frozenset({".codex-plugin"})

    def agent_files(self, agent_name: str) -> list[Path]:
        return [self.dist_dir / "agents" / f"{agent_name}.toml"]

    def _agent_toml(self, agent_name: str) -> dict:
        if tomllib is None:
            self.skipTest("tomllib is not available on Python < 3.11")
        path = self.dist_dir / "agents" / f"{agent_name}.toml"
        with path.open("rb") as handle:
            return tomllib.load(handle)

    def test_plugin_manifest_metadata(self):
        """.codex-plugin/plugin.json metadata matches package.release.json."""
        plugin_json_path = self.dist_dir / ".codex-plugin" / "plugin.json"
        self.assertTrue(plugin_json_path.is_file(),
                        "Missing .codex-plugin/plugin.json")

        plugin = jsonio.load_json(plugin_json_path)
        self.assertEqual(plugin["name"], self.manifest["name"])
        self.assertEqual(plugin["version"], self.manifest["version"])
        self.assertEqual(plugin["description"], self.manifest["description"])
        self.assertEqual(
            plugin["author"]["name"],
            self.manifest.get("author", "TPU agent suite framework"))
        self.assertIn("interface", plugin)
        self.assertIn("skills", plugin)

    def test_install_plan_uses_toml_configuration(self):
        """Codex is configured through `.codex/config.toml`, not a JSON merge."""
        plan = jsonio.load_json(self.dist_dir / "tpu-agent-suite-install.json")
        self.assertEqual(plan["json_merges"], [],
                         "Codex configures via TOML, not JSON merges")

        # Keyed by section as well as key: `enabled` appears under both
        # `agents` and `hooks`, so keying by name alone silently drops one and
        # asserts against whichever happened to be written last.
        by_key = {
            (entry["section"], entry["key"]): entry
            for entry in plan["toml_sets"]
        }
        self.assertIn(("agents", "enabled"), by_key)
        self.assertIn(("agents", "max_concurrent_threads_per_session"), by_key)
        self.assertIn(("sandbox", "allowed_paths"), by_key)
        for entry in plan["toml_sets"]:
            self.assertEqual(entry["path"], ".codex/config.toml")
            # `hooks` appears once the package declares a lifecycle hook.
            self.assertIn(entry["section"], ("agents", "sandbox", "hooks"))
        self.assertIs(by_key[("agents", "enabled")]["value"], True)
        # A nested chain holds one thread per level.
        self.assertGreaterEqual(
            by_key[("agents", "max_concurrent_threads_per_session")]["value"],
            package_checks.MAX_DELEGATION_DEPTH)
        self.assertEqual(by_key[("sandbox", "allowed_paths")]["value"],
                         ["~/.tpu-agent-suite"])

        # The tpu-knowledge-base routing block targets the AGENTS.md memory file
        # Two blocks: tpu-knowledge-base routing, and the compute rule the
        # main session needs (the compute section only reaches agents).
        markers = [b["marker"] for b in plan["markdown_merges"]]
        prefix = f"tpu-agent-suite:{self.manifest['name']}:"
        self.assertEqual(markers,
                         [prefix + "tpu-knowledge-base", prefix + "compute"])
        compute_block = plan["markdown_merges"][1]
        self.assertEqual(compute_block["path"], "AGENTS.md")
        self.assertIn("framework/runtime.py compute", compute_block["content"])
        block = plan["markdown_merges"][0]
        self.assertEqual(block["path"], "AGENTS.md")
        self.assertEqual(
            block["marker"],
            f"tpu-agent-suite:{self.manifest['name']}:tpu-knowledge-base")
        self.assertNotIn("install_root", block)
        self.assertIn(
            f"~/.tpu-agent-suite/packages/{self.manifest['name']}"
            f"@{self.manifest['version']}/data/tpu-knowledge-base/pages",
            block["content"])
        self.assertNotIn("{install_root}", block["content"])

    def test_agent_toml_is_parseable_and_complete(self):
        """Every agent TOML parses and carries the four required keys."""
        for agent in self.manifest["agents"]:
            config = self._agent_toml(agent["name"])
            self.assertEqual(config["name"], agent["name"])
            self.assertEqual(config["description"], agent["description"])
            self.assertIn("sandbox_mode", config)
            self.assertIn("approval_policy", config)
            self.assertTrue(
                config["developer_instructions"].strip(),
                f"{agent['name']}.toml has an empty developer_instructions")

    def _capabilities(self) -> tuple[set[str], set[str]]:
        return capability_sets(
            package_loader.get_declared_tools(PACKAGE_DIR, self.manifest))

    def test_sandbox_mode_matches_declared_tools(self):
        """Sandbox tier escalates with the agent's declared capability."""
        exec_like, write_like = self._capabilities()
        for agent in self.manifest["agents"]:
            tools = set(agent.get("tools", []))
            if tools & exec_like:
                expected = "danger-unrestricted"
            elif tools & write_like:
                expected = "workspace-write"
            else:
                expected = "read-only"
            self.assertEqual(
                self._agent_toml(agent["name"])["sandbox_mode"], expected,
                f"{agent['name']}: sandbox mode does not match tools {tools}")

    def test_shell_is_the_only_escalating_tool(self):
        """Pins which declared tools grant host/exec capability.

        The assumption the sandbox assertions rest on. If a tool's
        `action_level` changes -- say `edit` gains `scope: host` -- the tier
        expectations above shift with it, and this test says so directly
        instead of leaving the other failures to be puzzled over.
        """
        exec_like, write_like = self._capabilities()
        self.assertEqual(exec_like, {"shell"})
        self.assertEqual(write_like, {"write", "edit"})

    def test_approval_policy_is_a_known_value(self):
        """Approval policy is one of Codex's three accepted settings."""
        for agent in self.manifest["agents"]:
            self.assertIn(
                self._agent_toml(agent["name"])["approval_policy"],
                APPROVAL_POLICIES)

    def test_sandbox_mode_is_a_known_tier(self):
        """Sandbox mode is one of Codex's three accepted tiers."""
        for agent in self.manifest["agents"]:
            self.assertIn(
                self._agent_toml(agent["name"])["sandbox_mode"], SANDBOX_TIERS)

    def test_delegation_targets_present(self):
        """Declared delegates are named in the agent's instructions."""
        for agent in self.manifest["agents"]:
            if not agent.get("delegates"):
                continue
            instructions = self._agent_toml(
                agent["name"])["developer_instructions"]
            for delegate in agent["delegates"]:
                self.assertIn(delegate, instructions)

    def test_no_project_mirror(self):
        """Codex has no `project/` mirror.

        The full top-level tree is asserted by the shared base; this pins the
        one directory whose *absence* is the harness distinction.
        """
        self.assertFalse(
            (self.dist_dir / "project").exists(),
            "Codex builds must not emit the Claude-only project/ mirror")


if __name__ == "__main__":
    unittest.main()
