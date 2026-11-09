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
"""Antigravity translation parity for the tpu-agent-suite package.

Shared invariants live in `translation_base`. Antigravity differs from Claude
Code in the details this suite pins:

* agent frontmatter lists tools as a YAML block sequence, not a comma string,
  and the description is quoted;
* the tool vocabulary is Antigravity's own (`view_file`, `grep_search`,
  `run_command`, ...) and delegation is `invoke_subagent`, not `Agent`;
* agents are marked `subagent: true` rather than `background: false`;
* plugin metadata is a bare top-level `plugin.json` with no `.claude-plugin`
  style directory, and no version probe is declared.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_translation_base import TranslationParityBase  # noqa: E402
from test_translation_base import parse_yaml_frontmatter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from framework import jsonio, utils  # noqa: E402
from framework.builder import harness_rules, translators  # noqa: E402

Path = pathlib.Path

# Imported, not restated: one definition of the vocabulary, in the module that
# owns it. `test_tool_vocabulary_is_pinned` below asserts the literal values,
# so a change to the mapping still has to be made deliberately and shows up as
# one failing assertion rather than passing silently through both sides.
ANTIGRAVITY_TOOL_MAP = translators.GOOGLE_HARNESS_TOOL_MAPS["antigravity"]
DELEGATION_TOOLS = harness_rules.GOOGLE_DELEGATION_TOOLS


class TestAntigravityTranslationParity(TranslationParityBase,
                                       unittest.TestCase):
    """Validates that package/ translates correctly to Antigravity."""

    HARNESS = "antigravity"
    TARGET_ID = "antigravity-1.0"
    VERSION_RANGE = ">=1.0.0,<2.0.0"
    VERSION_COMMAND = ["agy", "--version"]
    # Antigravity discovers subagents at `.agents/agents`; `.agent/` is a
    # customization root for skills and is never scanned for agents.
    SKILL_INSTALL_DEST = ".agents/skills/tpu-agent"
    AGENTS_INSTALL_DEST = ".agents/agents"

    def agent_files(self, agent_name: str) -> list[Path]:
        return [self.dist_dir / "agents" / f"{agent_name}.md"]

    def test_plugin_manifest_metadata(self):
        """Top-level plugin.json metadata matches package.release.json."""
        plugin_json_path = self.dist_dir / "plugin.json"
        self.assertTrue(plugin_json_path.is_file(), "Missing plugin.json")

        plugin = jsonio.load_json(plugin_json_path)
        self.assertEqual(plugin["name"], self.manifest["name"])
        self.assertEqual(plugin["version"], self.manifest["version"])
        self.assertEqual(plugin["description"], self.manifest["description"])

    def test_install_plan_writes_only_workspace_memory(self):
        """Antigravity needs no settings merge or TOML write, only the memory block."""
        plan = jsonio.load_json(self.dist_dir / "tpu-agent-suite-install.json")
        self.assertEqual(plan["json_merges"], [])
        self.assertEqual(plan["toml_sets"], [])

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

    def test_tool_vocabulary_is_pinned(self):
        """The Antigravity tool names are what Antigravity actually accepts.

        The one place the vocabulary is written out literally. Everything else
        imports `translators.GOOGLE_HARNESS_TOOL_MAPS`, so without this a
        rename would propagate to both the generator and its tests and no
        assertion would notice. Changing a name here is fine -- it just has to
        be deliberate, and reviewed against Antigravity's own tool list.
        """
        self.assertEqual(
            ANTIGRAVITY_TOOL_MAP, {
                "read": ["view_file"],
                "search": ["grep_search", "find_by_name"],
                "shell": ["run_command", "manage_task"],
                "write": ["write_to_file"],
                "edit": ["replace_file_content", "multi_replace_file_content"],
                "ask": ["ask_question"],
                "web_search": ["search_web"],
                "web_fetch": ["read_url_content"],
            })
        self.assertEqual(DELEGATION_TOOLS,
                         ["invoke_subagent", "manage_subagents"])
        self.assertEqual(
            sorted(ANTIGRAVITY_TOOL_MAP), sorted(utils.PORTABLE_TOOLS),
            "Every portable tool needs an Antigravity mapping, and no mapping "
            "may exist for a tool the framework does not define")

    def test_agent_frontmatter_and_exact_tool_grants(self):
        """Frontmatter uses the Antigravity vocabulary and grants exactly it.

        Set equality rather than membership: an over-broad grant is the failure
        that matters, and several tool names share substrings.
        """
        for agent in self.manifest["agents"]:
            expected: list[str] = []
            for declared in agent.get("tools", []):
                for mapped in ANTIGRAVITY_TOOL_MAP[declared]:
                    if mapped not in expected:
                        expected.append(mapped)
            if agent.get("delegates"):
                # `invoke_subagent` returns a handle while the work runs and
                # the result message wakes the caller; `manage_subagents`
                # stops a subagent no longer needed. The root -- the skill's
                # twin, which `--agent` starts from -- also gets `schedule`
                # for the heartbeat that keeps a headless run alive.
                extra = list(DELEGATION_TOOLS)
                if harness_rules.is_root_agent(self.manifest, agent):
                    extra += harness_rules.GOOGLE_ROOT_TOOLS
                # A root with `shell` already has `manage_task` from its map.
                expected.extend(t for t in extra if t not in expected)

            path = self.dist_dir / "agents" / f"{agent['name']}.md"
            fields = parse_yaml_frontmatter(path.read_text(encoding="utf-8"))
            self.assertEqual(fields["name"], agent["name"])
            self.assertEqual(fields["subagent"], "true",
                             f"{agent['name']} must be marked subagent: true")

            granted = [
                line.lstrip("- ").strip()
                for line in fields["tools"].split("\n") if line.strip()
            ]
            self.assertEqual(
                sorted(granted), sorted(expected),
                f"{agent['name']} tool grant mismatch: got {granted}, "
                f"expected {expected}")

    def test_description_is_quoted(self):
        """Descriptions are emitted quoted so punctuation cannot break YAML."""
        for agent in self.manifest["agents"]:
            path = self.dist_dir / "agents" / f"{agent['name']}.md"
            fields = parse_yaml_frontmatter(path.read_text(encoding="utf-8"))
            self.assertEqual(
                fields["description"], f'"{agent["description"]}"',
                f"{agent['name']}: description must be double-quoted")

    def test_no_claude_tool_vocabulary_leaked(self):
        """Claude's tool names must not survive into an Antigravity build.

        The clearest signal that a translator fell through to the wrong
        frontmatter generator.
        """
        for agent in self.manifest["agents"]:
            path = self.dist_dir / "agents" / f"{agent['name']}.md"
            fields = parse_yaml_frontmatter(path.read_text(encoding="utf-8"))
            granted = {
                line.lstrip("- ").strip()
                for line in fields["tools"].split("\n") if line.strip()
            }
            for claude_tool in ("Read", "Write", "Edit", "Grep", "Glob",
                                "Bash", "Agent"):
                self.assertNotIn(
                    claude_tool, granted,
                    f"{agent['name']} carries Claude tool {claude_tool!r}")

    def test_delegation_targets_present(self):
        """Declared delegates are named in the agent's prompt body."""
        for agent in self.manifest["agents"]:
            if not agent.get("delegates"):
                continue
            content = (self.dist_dir / "agents" /
                       f"{agent['name']}.md").read_text(encoding="utf-8")
            for delegate in agent["delegates"]:
                self.assertIn(delegate, content)

    def test_no_project_mirror(self):
        """Antigravity has no `project/` mirror.

        It also contributes no `EXTRA_TOP_DIRS`: its plugin metadata is a
        top-level `plugin.json` file, not a directory. The full top-level tree
        is asserted by the shared base.
        """
        self.assertFalse(
            (self.dist_dir / "project").exists(),
            "Antigravity builds must not emit the Claude-only project/ mirror")


if __name__ == "__main__":
    unittest.main()
