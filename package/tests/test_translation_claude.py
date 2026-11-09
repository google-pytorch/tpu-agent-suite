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
"""Claude Code translation parity for the tpu-agent-suite package.

Shared invariants live in `translation_base`; this file covers what is specific
to Claude Code: the `.claude-plugin` metadata directory, the dual plugin/project
layout (the plugin variant addresses agents as `<package>:<agent>`, the project
variant by bare name), the `Read/Write/Edit/Grep/Glob/Bash` tool vocabulary, and
the `settings.local.json` subagent-depth merge.
"""

from __future__ import annotations

import pathlib
import re
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_translation_base import TranslationParityBase  # noqa: E402
from test_translation_base import parse_yaml_frontmatter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from framework import jsonio, utils  # noqa: E402
from framework.builder import harness_rules  # noqa: E402

Path = pathlib.Path

# Imported, not restated: one definition of the vocabulary, in the module that
# owns it. `test_tool_vocabulary_is_pinned` below asserts the literal values,
# so a change to the mapping still has to be made deliberately and shows up as
# one failing assertion rather than passing silently through both sides.
CLAUDE_TOOL_MAP = harness_rules.CLAUDE_TOOL_MAP
DELEGATION_TOOL = harness_rules.CLAUDE_DELEGATION_TOOL


class TestClaudeTranslationParity(TranslationParityBase, unittest.TestCase):
    """Validates that package/ translates correctly to Claude Code."""

    HARNESS = "claude-code"
    TARGET_ID = "claude-code-2.1"
    VERSION_RANGE = ">=2.1.219,<3.0.0"
    VERSION_COMMAND = ["claude", "--version"]
    SKILL_INSTALL_DEST = ".claude/skills/tpu-agent"
    AGENTS_INSTALL_DEST = ".claude/agents"
    # Claude alone emits the plugin manifest dir and the `project/` mirror.
    EXTRA_TOP_DIRS = frozenset({".claude-plugin", "project"})

    def agent_files(self, agent_name: str) -> list[Path]:
        return [
            self.dist_dir / "agents" / f"{agent_name}.md",
            self.dist_dir / "project" / "agents" / f"{agent_name}.md",
        ]

    def skill_files(self) -> list[Path]:
        return [
            self.dist_dir / "skills" / self.entry_name / "SKILL.md",
            self.dist_dir / "project" / "skills" / self.entry_name /
            "SKILL.md",
        ]

    def test_plugin_manifest_metadata(self):
        """.claude-plugin/plugin.json metadata matches package.release.json."""
        plugin_json_path = self.dist_dir / ".claude-plugin" / "plugin.json"
        self.assertTrue(plugin_json_path.is_file(),
                        "Missing .claude-plugin/plugin.json")

        plugin = jsonio.load_json(plugin_json_path)
        self.assertEqual(plugin["name"], self.manifest["name"])
        self.assertEqual(
            plugin["displayName"],
            self.manifest.get("display_name", self.manifest["name"]),
        )
        self.assertEqual(plugin["version"], self.manifest["version"])
        self.assertEqual(plugin["description"], self.manifest["description"])
        self.assertEqual(
            plugin["author"]["name"],
            self.manifest.get("author", "TPU agent suite framework"))

    def test_mcp_manifest_absent_until_an_mcp_tool_is_declared(self):
        """`mcp.json` is written only when an MCP-category tool exists.

        The package declares five native tools and no MCP servers, so the file
        must be absent -- an empty `{"mcpServers": {}}` would make Claude Code
        report a configured-but-empty MCP source. Declaring the first MCP tool
        is expected to flip this assertion; that is the point of pinning it.
        """
        declared = {
            tool.get("category")
            for tool in self.manifest.get("native_tools", [])
        }
        self.assertNotIn(
            "mcp", declared,
            "Package now declares an MCP tool -- invert this assertion and "
            "assert mcp.json contents instead.")
        self.assertFalse(
            (self.dist_dir / ".claude-plugin" / "mcp.json").exists(),
            "mcp.json emitted although no MCP tool is declared")

    def test_prompt_timeouts_fit_shell_timeout(self):
        """No prompt asks for a run longer than the Bash timeout allows."""
        package_dir = Path(__file__).resolve().parents[1]
        over = [
            f"{path.relative_to(package_dir)}: --timeout {seconds}"
            for path in package_dir.rglob("*.md") for seconds in map(
                int,
                re.findall(r"--timeout[ =](\d+)",
                           path.read_text(encoding="utf-8")))
            if seconds > harness_rules.MAX_SYNC_COMPUTE_TIMEOUT_S
        ]
        self.assertEqual(
            over, [], "Raise MAX_SYNC_COMPUTE_TIMEOUT_S, or the Bash call "
            "running these is killed before the run finishes")

    def test_install_plan_settings_merge(self):
        """The plan merges the subagent depth/concurrency caps into settings."""
        plan = jsonio.load_json(self.dist_dir / "tpu-agent-suite-install.json")
        self.assertEqual(len(plan["json_merges"]), 1)
        merge = plan["json_merges"][0]
        self.assertEqual(merge["path"], ".claude/settings.local.json")
        env = merge["value"]["env"]
        self.assertIn("CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH", env)
        self.assertIn("CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS", env)
        self.assertGreaterEqual(
            int(env["CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH"]), 3)
        # A nested chain holds one slot per level, so the concurrency cap has
        # to cover the deepest chain the depth cap allows.
        self.assertGreaterEqual(
            int(env["CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS"]),
            int(env["CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH"]))
        # Background work is allowed; the idle gate keeps a subagent's wait
        # from passing as its result.
        self.assertNotIn("CLAUDE_CODE_DISABLE_BACKGROUND_TASKS", env)
        # A synchronous `compute run` holds its Bash call open, so both the
        # cap and the default must outlast the longest run. Claude Code falls back to 2 minutes when
        # the default is unset, so it is set explicitly.
        self.assertEqual(env["BASH_DEFAULT_TIMEOUT_MS"],
                         str(harness_rules.SHELL_TIMEOUT_MS))
        self.assertEqual(env["BASH_MAX_TIMEOUT_MS"],
                         str(harness_rules.SHELL_TIMEOUT_MS))
        self.assertGreater(harness_rules.SHELL_TIMEOUT_MS,
                           harness_rules.MAX_SYNC_COMPUTE_TIMEOUT_S * 1000)
        self.assertEqual(plan["toml_sets"], [],
                         "Claude Code configures via JSON, not TOML")
        # The tpu-knowledge-base routing block targets Claude Code's memory file
        # Two blocks: tpu-knowledge-base routing, and the compute rule the
        # main session needs (the compute section only reaches agents).
        markers = [b["marker"] for b in plan["markdown_merges"]]
        prefix = f"tpu-agent-suite:{self.manifest['name']}:"
        self.assertEqual(markers,
                         [prefix + "tpu-knowledge-base", prefix + "compute"])
        compute_block = plan["markdown_merges"][1]
        self.assertEqual(compute_block["path"], "CLAUDE.md")
        self.assertIn("framework/runtime.py compute", compute_block["content"])
        block = plan["markdown_merges"][0]
        self.assertEqual(block["path"], "CLAUDE.md")
        self.assertEqual(
            block["marker"],
            f"tpu-agent-suite:{self.manifest['name']}:tpu-knowledge-base")
        self.assertNotIn("install_root", block)
        self.assertIn(
            f"~/.tpu-agent-suite/packages/{self.manifest['name']}"
            f"@{self.manifest['version']}/data/tpu-knowledge-base/pages",
            block["content"])
        self.assertNotIn("{install_root}", block["content"])

        # Check copy mappings for skills, agents, runtime, and schemas
        destinations = [c["destination"] for c in plan["copies"]]
        self.assertIn(".claude/skills/tpu-agent", destinations)
        self.assertIn(".claude/agents", destinations)
        central = {
            c["destination"]
            for c in plan["copies"] if c["store"] == "central"
        }
        for expected in ("framework", "schemas", "tpu-agent-suite-build.json",
                         "data"):
            self.assertIn(expected, central)
        # The knowledge base left package.release.json's `include`; nothing may install it.
        self.assertNotIn("knowledgebase", central)

    def test_background_ledger_hooks_compile_to_native_matchers(self):
        """Launches are recorded after `Agent` and `Bash`, cleared at session
        end, and nothing registers the recorder on SubagentStop."""
        hooks = jsonio.load_json(self.dist_dir / ".claude-plugin" /
                                 "hooks.json")

        record = hooks["track-background-work"]
        self.assertEqual(list(record), ["PostToolUse"])
        [group] = record["PostToolUse"]
        self.assertEqual(sorted(group["matcher"].split("|")),
                         ["Agent", "Bash"])
        [entry] = group["hooks"]
        self.assertIn("--builtin track-background", entry["command"])
        self.assertIn("--event PostToolUse", entry["command"])

        cleanup = hooks["background-ledger-cleanup"]
        self.assertEqual(list(cleanup), ["SessionEnd"])
        [[entry]] = [g["hooks"] for g in cleanup["SessionEnd"]]
        self.assertIn("--builtin track-background", entry["command"])

        for name, events in hooks.items():
            for event, groups in events.items():
                for g in groups:
                    for h in g["hooks"]:
                        if "--builtin track-background" in h["command"]:
                            self.assertNotEqual(event, "SubagentStop", name)

    def test_skill_addresses_agents_per_layout(self):
        """Plugin docs use `<package>:<agent>`; project docs use bare names."""
        plugin_content = (self.dist_dir / "skills" / self.entry_name /
                          "SKILL.md").read_text(encoding="utf-8")
        project_content = (self.dist_dir / "project" / "skills" /
                           self.entry_name /
                           "SKILL.md").read_text(encoding="utf-8")

        self.assertIn(f"{self.pkg_name}:tpu-model-onboarding", plugin_content)
        self.assertIn(
            f"Direct delegation targets: "
            f"{self.pkg_name}:tpu-model-onboarding", plugin_content)
        self.assertIn("Direct delegation targets: tpu-model-onboarding",
                      project_content)
        self.assertNotIn(f"{self.pkg_name}:tpu-model-onboarding",
                         project_content)

        pkg_tag = f"{self.pkg_name}@{self.manifest.get('version', '')}"
        for content in (plugin_content, project_content):
            self.assertIn(
                "Retain the active `run_id` assigned at session start",
                content)
            self.assertIn("~/.tpu-agent-suite/runs/<run_id>/artifacts/",
                          content)

    def test_delegation_targets_per_layout(self):
        """Delegates are prefixed in the plugin layout, bare in the project one."""
        for agent in self.manifest["agents"]:
            if not agent.get("delegates"):
                continue
            plugin_content = (self.dist_dir / "agents" /
                              f"{agent['name']}.md").read_text(
                                  encoding="utf-8")
            project_content = (self.dist_dir / "project" / "agents" /
                               f"{agent['name']}.md").read_text(
                                   encoding="utf-8")
            for delegate in agent["delegates"]:
                self.assertIn(f"{self.pkg_name}:{delegate}", plugin_content)
                self.assertIn(delegate, project_content)

    def test_tool_vocabulary_is_pinned(self):
        """The Claude tool names are what Claude Code actually accepts.

        The one place the vocabulary is written out literally. Everything else
        imports `harness_rules.CLAUDE_TOOL_MAP`, so without this a rename would
        propagate to both the generator and its tests and no assertion would
        notice. Changing a name here is fine -- it just has to be deliberate,
        and reviewed against Claude Code's own tool list.
        """
        self.assertEqual(
            CLAUDE_TOOL_MAP, {
                "read": ["Read"],
                "search": ["Grep", "Glob"],
                "shell": ["Bash"],
                "write": ["Write"],
                "edit": ["Edit"],
                "ask": ["AskUserQuestion"],
                "web_search": ["WebSearch"],
                "web_fetch": ["WebFetch"],
            })
        self.assertEqual(DELEGATION_TOOL, "Agent")
        self.assertEqual(
            sorted(CLAUDE_TOOL_MAP), sorted(utils.PORTABLE_TOOLS),
            "Every portable tool needs a Claude mapping, and no mapping may "
            "exist for a tool the framework does not define")

    def test_agent_frontmatter_and_exact_tool_grants(self):
        """Frontmatter names the agent and grants exactly the mapped tools.

        Set equality, not membership: a substring check passes when an agent is
        granted tools it never declared, which is the failure that matters --
        an over-broad grant is a privilege bug, and `Read` is also a substring
        of several other tool names.
        """
        for agent in self.manifest["agents"]:
            expected: list[str] = []
            for declared in agent.get("tools", []):
                for mapped in CLAUDE_TOOL_MAP[declared]:
                    if mapped not in expected:
                        expected.append(mapped)
            if agent.get("delegates"):
                expected.append(DELEGATION_TOOL)

            for path in self.agent_files(agent["name"]):
                fields = parse_yaml_frontmatter(
                    path.read_text(encoding="utf-8"))
                self.assertEqual(fields["name"], agent["name"])
                self.assertEqual(fields["description"], agent["description"])
                self.assertEqual(
                    fields["background"], "false",
                    f"{path} must pin background: false so subagents run "
                    f"inline")
                granted = [t.strip() for t in fields["tools"].split(",")]
                self.assertEqual(
                    sorted(granted), sorted(expected),
                    f"{path} tool grant mismatch: got {granted}, "
                    f"expected {expected}")

    def test_nested_directory_presence(self):
        """Selectively copied scripts & references land at their package paths.

        A spot-check, not an exhaustive list -- the top-level tree is covered
        by `test_top_level_directory_tree` in the shared base.
        """
        expected_nested_dirs = [
            "project/agents",
            "project/skills/tpu-agent",
            "skills/tpu-agent",
            "references/tpu-model-onboarding/remote-tpu-runner",
            "scripts/tpu-model-onboarding",
            "references/tpu-model-onboarding/torch-benchmark-coder",
            "references/tpu-model-onboarding/torch-inference-coder/fastapi_server",
            "data/tpu-knowledge-base/index",
            "data/tpu-knowledge-base/pages",
            "data/tpu-knowledge-base/prompts",
        ]
        for rel_dir in expected_nested_dirs:
            self.assertTrue(
                (self.dist_dir / rel_dir).is_dir(),
                f"Expected directory missing: {rel_dir}",
            )

    def test_claude_critical_files(self):
        """Claude-specific files on top of the shared critical set."""
        for rel in (
                ".claude-plugin/plugin.json",
                "project/skills/tpu-agent/SKILL.md",
                "references/tpu-model-onboarding/remote-tpu-runner/hardware_profiling.md",
        ):
            self.assertTrue((self.dist_dir / rel).is_file(),
                            f"Critical distribution file missing: {rel}")


if __name__ == "__main__":
    unittest.main()
