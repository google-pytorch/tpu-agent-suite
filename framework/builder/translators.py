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
"""Turns a package into the files one harness needs.

Every harness wants agents and skills in its own format: Claude Code takes
Markdown agent files, Codex takes TOML, Antigravity has its own layout, and
Cursor gets a single rules file. Each harness has a "translator" function that
writes that format; `translate()` runs the right one, then adds the hooks,
runtime files and install plan.
"""

from __future__ import annotations

import pathlib
import shutil
from typing import Any, Callable, Sequence

from framework import harnesses, jsonio, utils
from framework.builder import (dist_assets, harness_rules, hook_config,
                               package_loader, prompt_contracts)

Path = pathlib.Path


def write_mcp_config(path: Path, manifest: dict[str, Any],
                     include_internal: bool) -> None:
    """Write the package's MCP servers to `path`, if a build ships any.

    Every harness reads the same `mcpServers` block; only where it lives
    differs.
    """
    mcp_configs: dict[str, Any] = {}
    for mcp in manifest.get("mcp_servers", []):
        if package_loader.excluded_internal(mcp, include_internal):
            continue
        mcp_configs[mcp["name"]] = {
            "command": mcp["command"],
            "args": mcp.get("args", []),
            "env": mcp.get("env", {}),
        }
    if mcp_configs:
        jsonio.write_json(path, {"mcpServers": mcp_configs})


def translate_claude(
    package_dir: Path,
    output: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    include_internal: bool = False,
) -> None:
    """Translate package into Claude Code plugin and project layouts."""
    declared_tools = package_loader.get_declared_tools(package_dir, manifest)
    plugin = output / ".claude-plugin"
    jsonio.write_json(
        plugin / "plugin.json",
        {
            "name": manifest["name"],
            "displayName": manifest.get("display_name", manifest["name"]),
            "version": manifest["version"],
            "description": manifest["description"],
            "author": {
                "name": manifest.get("author", "TPU agent suite framework")
            },
        },
    )

    plugin_agent_prefix = f"{manifest['name']}:"

    # Write plugin skill and standalone project skill
    prompt_contracts.write_skill_documents(output / "skills", package_dir,
                                           manifest, target,
                                           plugin_agent_prefix)
    prompt_contracts.write_skill_documents(output / "project" / "skills",
                                           package_dir, manifest, target)

    # Write plugin agents and project agents
    for agent in manifest["agents"]:
        resolved_tools = harness_rules.resolve_agent_tools(
            agent, declared_tools, include_internal)
        fm = harness_rules.claude_frontmatter(agent, resolved_tools)
        path = output / "agents" / f"{agent['name']}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            fm + prompt_contracts.agent_prompt(package_dir, manifest, target,
                                               agent, plugin_agent_prefix),
            encoding="utf-8",
        )

        project_agent = output / "project" / "agents" / f"{agent['name']}.md"
        project_agent.parent.mkdir(parents=True, exist_ok=True)
        project_agent.write_text(
            fm + prompt_contracts.agent_prompt(package_dir, manifest, target,
                                               agent),
            encoding="utf-8",
        )

    # Export MCP server configurations if declared
    write_mcp_config(output / ".claude-plugin" / "mcp.json", manifest,
                     include_internal)

    hooks_config = hook_config.build_harness_hooks_config(
        manifest["name"], manifest, "claude-code", include_internal)
    if hooks_config:
        jsonio.write_json(output / ".claude-plugin" / "hooks.json",
                          hooks_config)
        jsonio.write_json(output / "project" / "hooks.json", hooks_config)


def translate_codex(
    package_dir: Path,
    output: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    include_internal: bool = False,
) -> None:
    """Translate package into Codex plugin and agents layouts."""
    declared_tools = package_loader.get_declared_tools(package_dir, manifest)
    hooks_config = hook_config.build_harness_hooks_config(
        manifest["name"], manifest, "codex", include_internal)
    plugin = output / ".codex-plugin"
    plugin_manifest: dict[str, Any] = {
        "name": manifest["name"],
        "version": manifest["version"],
        "description": manifest["description"],
        "author": {
            "name": manifest.get("author", "TPU agent suite framework")
        },
        "skills": "./skills/",
        "interface": {
            "displayName": manifest.get("display_name", manifest["name"]),
            "shortDescription": manifest["description"][:80],
            "longDescription": manifest["description"],
            "developerName": manifest.get("author",
                                          "TPU agent suite framework"),
            "category": "Productivity",
            "capabilities": ["Interactive", "Write"],
            "defaultPrompt": [default_prompt(package_dir)[:128]],
        },
    }
    # Codex defaults to `hooks/hooks.json` at the plugin root, which here holds
    # the package's own hook scripts. Declaring the path keeps the generated
    # file beside the rest of the generated plugin config.
    if hooks_config:
        plugin_manifest["hooks"] = "./.codex-plugin/hooks.json"
    jsonio.write_json(plugin / "plugin.json", plugin_manifest)

    prompt_contracts.write_skill_documents(output / "skills", package_dir,
                                           manifest, target)

    for agent in manifest["agents"]:
        resolved_tools = harness_rules.resolve_agent_tools(
            agent, declared_tools, include_internal)
        config = output / "agents" / f"{agent['name']}.toml"
        config.parent.mkdir(parents=True, exist_ok=True)
        prompt = harness_rules.toml_multiline_string(
            prompt_contracts.agent_prompt(package_dir, manifest, target,
                                          agent))
        sandbox = harness_rules.codex_sandbox_mode(resolved_tools)
        approval = harness_rules.codex_approval_policy(resolved_tools)
        config.write_text(
            f'name = "{harness_rules.toml_basic_string(agent["name"])}"\n'
            f'description = "{harness_rules.toml_basic_string(agent["description"])}"\n'
            f'sandbox_mode = "{sandbox}"\n'
            f'approval_policy = "{approval}"\n'
            f'developer_instructions = """\n{prompt}\n"""\n',
            encoding="utf-8",
        )

    if hooks_config:
        events: dict[str, list[dict[str, Any]]] = {}
        for hook_name in sorted(hooks_config):
            for event, groups in sorted(hooks_config[hook_name].items()):
                events.setdefault(event, []).extend(groups)
        codex_hooks = {
            "description": manifest["description"],
            "hooks": events,
        }
        jsonio.write_json(plugin / "hooks.json", codex_hooks)
        jsonio.write_json(output / ".codex" / "hooks.json", codex_hooks)


def translate_google_harness_layout(
    package_dir: Path,
    output: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    harness: str,
    tool_map: dict[str, list[str]],
    include_internal: bool = False,
) -> None:
    """Translate package into the layout every Google harness shares."""
    declared_tools = package_loader.get_declared_tools(package_dir, manifest)
    jsonio.write_json(
        output / "plugin.json",
        {
            "name": manifest["name"],
            "version": manifest["version"],
            "description": manifest["description"],
        },
    )

    prompt_contracts.write_skill_documents(output / "skills", package_dir,
                                           manifest, target)

    for agent in manifest["agents"]:
        resolved_tools = harness_rules.resolve_agent_tools(
            agent, declared_tools, include_internal)
        fm = harness_rules.google_harness_frontmatter(
            agent,
            resolved_tools,
            tool_map,
            root=harness_rules.is_root_agent(manifest, agent))
        agent_path = output / "agents" / f"{agent['name']}.md"
        agent_path.parent.mkdir(parents=True, exist_ok=True)
        agent_path.write_text(
            fm + prompt_contracts.agent_prompt(package_dir, manifest, target,
                                               agent),
            encoding="utf-8",
        )

    write_mcp_config(output / "mcp_config.json", manifest, include_internal)

    hooks_config = hook_config.build_harness_hooks_config(
        manifest["name"], manifest, harness, include_internal)
    if hooks_config:
        jsonio.write_json(output / "hooks.json", hooks_config)


def google_harness_translator(
    harness: str,
    tool_map: dict[str, list[str]],
) -> Callable[[Path, Path, dict[str, Any], dict[str, Any], bool], None]:
    """Bind a Google harness and its tool map to the family translator.

    The layout a Google harness is compiled into is the family's; only the
    native tool names differ. Binding the pair here is what lets a new member
    be registered by naming it and its tool map, with no translator of its own
    to keep in step with this one.
    """

    def translate_google_harness(
        package_dir: Path,
        output: Path,
        manifest: dict[str, Any],
        target: dict[str, Any],
        include_internal: bool = False,
    ) -> None:
        translate_google_harness_layout(package_dir, output, manifest, target,
                                        harness, tool_map, include_internal)

    return translate_google_harness


DEFAULT_PROMPT_FILE = "default-prompt.md"


def default_prompt(package_dir: Path) -> str:
    """The prompt a harness offers a new user, from `default-prompt.md`.

    Build input, not shipped content: it names what the package is for in one
    line a user can run as written.
    """
    path = package_dir / DEFAULT_PROMPT_FILE
    if not path.is_file():
        return "Run this workflow."
    text = " ".join(path.read_text(encoding="utf-8").split())
    return text or "Run this workflow."


def translate_cursor(
    package_dir: Path,
    output: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    include_internal: bool = False,
) -> None:
    """Translate package into Cursor rules, skills, and MCP configurations."""
    prompt_contracts.write_skill_documents(output / "skills", package_dir,
                                           manifest, target)

    # Generate consolidated Cursor rule (.mdc)
    rules_dir = output / ".cursor" / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)

    rule_content = [
        "---",
        f'description: "{manifest["description"]}"',
        'globs: "*"',
        "alwaysApply: true",
        "---",
        "",
        f"# {manifest.get('display_name', manifest['name'])} Workflow Rules",
        "",
        # Cursor runs one context, so a skill that delegates needs the
        # agents it names in that context with it. A skill that does not
        # delegate is served from `.cursor/skills` like any other.
        *[
            prompt_contracts.skill_document(package_dir, manifest, target,
                                            skill)
            for skill in manifest["skills"] if skill["delegates"]
        ],
        "",
        "## Agent Personas & Execution Contracts",
        "",
    ]

    for agent in manifest["agents"]:
        rule_content.append(f"### Persona: {agent['name']}")
        rule_content.append(
            prompt_contracts.agent_prompt(package_dir, manifest, target,
                                          agent))
        rule_content.append("")

    (rules_dir / f"{manifest['name']}.mdc").write_text("\n".join(rule_content),
                                                       encoding="utf-8")

    write_mcp_config(output / ".cursor" / "mcp.json", manifest,
                     include_internal)

    hooks_config = hook_config.build_harness_hooks_config(
        manifest["name"], manifest, "cursor", include_internal)
    if hooks_config:
        events: dict[str, list[dict[str, Any]]] = {}
        for hook_name in sorted(hooks_config):
            for event, groups in sorted(hooks_config[hook_name].items()):
                for group in groups:
                    for entry in group["hooks"]:
                        events.setdefault(event, []).append({
                            "type":
                            "command",
                            "command":
                            entry["command"],
                            "timeout":
                            entry["timeout"],
                            "matcher":
                            group["matcher"],
                        })
        jsonio.write_json(output / ".cursor" / "hooks.json", {
            "version": 1,
            "hooks": events
        })


Translator = Callable[[Path, Path, dict[str, Any], dict[str, Any], bool], None]

# The native tool vocabulary of each Google harness. The family shares a
# translator; what it does not share is the names the tools go by, so this is
# the whole of what a member has to state about itself.
GOOGLE_HARNESS_TOOL_MAPS: dict[str, dict[str, list[str]]] = {
    harness: harness_rules.load_harness_tool_map(harness)
    for harness in harnesses.GOOGLE_HARNESSES
}

# Which function compiles a package for each harness. Registering a Google
# harness means adding it to `GOOGLE_HARNESS_TOOL_MAPS` above and to
# `GOOGLE_HARNESSES`; the translator, hook shape, and mappings all follow from
# the family.
HARNESS_TRANSLATORS: dict[str, Translator] = {
    "claude-code": translate_claude,
    "codex": translate_codex,
    "cursor": translate_cursor,
    **{
        harness: google_harness_translator(harness, tool_map)
        for harness, tool_map in GOOGLE_HARNESS_TOOL_MAPS.items()
    },
}


def translate(
    package_dir: Path,
    harness: str,
    output: Path,
    include_internal: bool = False,
    allow_extra_fields: bool = False,
    skills: Sequence[str] | None = None,
    manifest_name: str = package_loader.DEFAULT_MANIFEST,
) -> None:
    """Translate a package into a target harness distribution."""
    assert set(GOOGLE_HARNESS_TOOL_MAPS) == harnesses.GOOGLE_HARNESSES, (
        "GOOGLE_HARNESS_TOOL_MAPS and harnesses.GOOGLE_HARNESSES must name the"
        " same harnesses")
    manifest = package_loader.load_package(
        package_dir,
        allow_extra_fields=allow_extra_fields,
        skills=skills,
        manifest=manifest_name)
    target = package_loader.target_for(manifest, harness)

    if output.exists():
        marker = output / "tpu-agent-suite-build.json"
        if not marker.is_file() or not jsonio.load_json(marker).get(
                "generated"):
            raise utils.TpuAgentSuiteError(
                f"refusing to replace non-generated output directory: {output}"
            )
        shutil.rmtree(output)

    output.mkdir(parents=True)

    HARNESS_TRANSLATORS[harness](package_dir, output, manifest, target,
                                 include_internal)
    jsonio.write_json(
        output / "tpu-agent-suite-install.json",
        dist_assets.build_install_plan(package_dir, manifest, target, harness),
    )

    dist_assets.copy_runtime_assets(package_dir, output, manifest, harness)
    dist_assets.emit_distribution_requirements(package_dir, output, manifest)

    contracts = {
        name: {
            "accepts": a.get("accepts", ""),
            "produces": a.get("produces", ""),
        }
        for name, a in package_loader.get_manifest_agent_map(manifest).items()
    }

    jsonio.write_json(
        output / "tpu-agent-suite-build.json",
        {
            "version":
            manifest.get("version", utils.TPU_AGENT_SUITE_VERSION),
            "source":
            manifest["name"],
            "compatibility_target":
            target["name"],
            "harness":
            harness,
            "version_range":
            target["versions"],
            "include_internal":
            include_internal,
            "contracts":
            contracts,
            "skill_allowed_tools":
            hook_config.hook_allowed_skill_tools(manifest, harness),
            "agent_tools": {
                agent["name"]: list(agent.get("tools", []))
                for agent in manifest["agents"]
            },
            "generated":
            True,
        },
    )

    dist_assets.bundle_standalone_distribution(output, manifest, harness)
