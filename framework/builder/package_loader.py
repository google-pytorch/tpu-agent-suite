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
"""Loads a package from disk into one manifest for the rest of the build.

Reads the manifest, applies `extends` and composition, finds every unit with
`package_discovery`, runs the checks in `package_checks`, and returns the
combined manifest. `load_package` is the entry point.
"""

from __future__ import annotations

import pathlib
from typing import Any, Sequence

from framework import frontmatter, jsonio, utils
from framework.builder import package_checks, package_discovery, tokens

Path = pathlib.Path

DEFAULT_MANIFEST = "package.release.json"


def hook_script(action: Any) -> str | None:
    """The package path a `script:` hook action runs, else None."""
    if isinstance(action, str) and action.startswith("script:"):
        return action[len("script:"):]
    return None


def excluded_internal(entry: dict[str, Any], include_internal: bool) -> bool:
    """Whether a build leaves out `entry`: internal, and not asked for."""
    return entry.get("visibility") == "internal" and not include_internal


def get_declared_tools(package_dir: Path,
                       manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index declared tools and assert file existence and global name uniqueness."""
    declared: dict[str, dict[str, Any]] = {}

    # 1. Native tools (from native_tools.json catalog or explicit manifest)
    for tool in manifest.get("native_tools", []):
        name = tool["name"]
        if name in declared:
            raise utils.TpuAgentSuiteError(
                f"duplicate tool name declared: '{name}'")
        declared[name] = {"category": "native", **tool}

    # 2. Script Tools
    for script in manifest.get("script_tools", []):
        name = script["name"]
        if name in declared:
            raise utils.TpuAgentSuiteError(
                f"duplicate tool name declared: '{name}'")
        script_path = utils.rooted(package_dir, script["path"],
                                   f"script tool '{name}'")
        if not script_path.is_file():
            raise utils.TpuAgentSuiteError(
                f"script tool '{name}' file does not exist: {script_path}")
        declared[name] = {"category": "script", **script}

    # 3. MCP Servers
    for mcp in manifest.get("mcp_servers", []):
        name = mcp["name"]
        if name in declared:
            raise utils.TpuAgentSuiteError(
                f"duplicate tool name declared: '{name}'")
        declared[name] = {"category": "mcp", **mcp}

    return declared


def resolve_manifest_extends(package_dir: Path, manifest: dict[str, Any],
                             manifest_name: str) -> dict[str, Any]:
    """Merge a manifest that `extends` another manifest onto that base.

    The base's fields apply first and this manifest's fields replace them.
    `add_skills` appends to the base's `skills`, so a variant package names
    only what it adds. One level only: the base must not extend another.
    """
    if "extends" not in manifest:
        if "add_skills" in manifest:
            raise utils.TpuAgentSuiteError(
                f"{manifest_name}: `add_skills` needs `extends`")
        return manifest
    base_name = manifest["extends"]
    if not isinstance(base_name, str) or base_name == manifest_name:
        raise utils.TpuAgentSuiteError(
            f"{manifest_name}: `extends` must name another manifest file")
    base = jsonio.load_json(
        utils.rooted(package_dir, base_name, "extended manifest"))
    if not isinstance(base, dict):
        raise utils.TpuAgentSuiteError(
            f"{base_name}: manifest must be a JSON object")
    if "extends" in base:
        raise utils.TpuAgentSuiteError(
            f"{manifest_name}: extends {base_name}, which extends another"
            " manifest; only one level is supported")
    add_skills = manifest.get("add_skills", [])
    if not isinstance(add_skills, list) or not all(
            isinstance(name, str) for name in add_skills):
        raise utils.TpuAgentSuiteError(
            f"{manifest_name}: `add_skills` must be a list of skill names")
    merged = {
        **base,
        **{
            key: value
            for key, value in manifest.items() if key not in ("extends", "add_skills")
        }
    }
    skills = list(merged.get("skills", []))
    repeated = sorted(set(add_skills) & set(skills))
    if repeated:
        raise utils.TpuAgentSuiteError(
            f"{manifest_name}: `add_skills` repeats skills already listed:"
            f" {repeated}")
    merged["skills"] = skills + add_skills
    return merged


def resolve_manifest_composition(
    package_dir: Path,
    manifest: dict[str, Any],
    skills: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The manifest plus everything its skills reach.

    Returns a copy carrying the default native tool catalog and what
    `package_discovery.discover_units` found: the loaded skills, the agents
    they reach, the union of dependencies and the registered schemas.

    `skills` replaces the manifest's own list (`--skills`). It may name any
    skill the package holds under `skills/`, released or not, and each is
    built whole: nothing is trimmed from what a skill delegates to.
    """
    composed = dict(manifest)
    if skills is not None:
        composed["skills"] = list(skills)

    if "native_tools" not in composed:
        try:
            default_native = utils.get_native_tools_catalog()
            if "native_tools" in default_native:
                composed["native_tools"] = default_native["native_tools"]
        except utils.TpuAgentSuiteError:
            pass

    package_discovery.discover_units(package_dir, composed)
    return composed


def _require_unit_files(package_dir: Path, unit: dict[str, Any],
                        label: str) -> None:
    """Fail unless every reference and script `unit` declares is a file."""
    for key, kind in (("references", "reference file"), ("scripts",
                                                         "script file")):
        for rel in unit.get(key, []):
            path = utils.rooted(package_dir, rel, f"{label} {kind}")
            if not path.is_file():
                raise utils.TpuAgentSuiteError(
                    f"{label} {kind} does not exist: {path}")


def load_package(package_dir: Path,
                 allow_extra_fields: bool = False,
                 skills: Sequence[str] | None = None,
                 manifest: str = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load, validate, and verify the structural integrity of a package manifest and its graph.

    `manifest` names the manifest file in `package_dir`. Several manifests
    can share one package directory, each releasing its own package from
    the same skills, agents, scripts and references. A manifest that
    `extends` another is merged onto it first.
    """
    manifest_path = utils.rooted(package_dir, manifest, "manifest")
    raw_manifest = resolve_manifest_extends(package_dir,
                                            jsonio.load_json(manifest_path),
                                            manifest)
    package_checks.reject_removed_fields(raw_manifest)
    package_checks.reject_nested_units(package_dir)
    manifest_schema = utils.get_package_manifest_schema()

    raw_schema_errors = utils.validate_instance(
        raw_manifest,
        manifest_schema,
        allow_additional_properties=allow_extra_fields,
    )
    if raw_schema_errors:
        raise utils.TpuAgentSuiteError(f"invalid {manifest}:\n" +
                                       "\n".join(raw_schema_errors))

    # 1. Discover what the released skills reach
    manifest = resolve_manifest_composition(package_dir,
                                            raw_manifest,
                                            skills=skills)

    manifest_name = manifest["name"]

    # 2. Tool indexing & name uniqueness
    declared_tools = get_declared_tools(package_dir, manifest)

    # 3. Agent presence & uniqueness
    agent_names = [agent["name"] for agent in manifest["agents"]]
    if len(agent_names) != len(set(agent_names)):
        raise utils.TpuAgentSuiteError(
            f"Agent names are not unique in package: {manifest_name}")
    name_set = set(agent_names)

    # What package.release.json copies: what a data or schema token may name.
    assets = list(manifest.get("copy", []))

    # 4. Skill validation. Every released skill is a delegation root.
    skills = manifest["skills"]
    skill_names = {skill["name"] for skill in skills}
    for skill in skills:
        # Checked here, on the skill, rather than left to the agent emitted
        # for a skill that delegates: a skill that does not delegate has no
        # such agent, and a message about one names a file nobody wrote.
        for tool_name in skill.get("tools", []):
            package_checks.validate_agent_tool(tool_name,
                                               declared_tools,
                                               skill["name"],
                                               kind="skill")
        unknown = set(skill["delegates"]) - name_set - skill_names
        if unknown:
            raise utils.TpuAgentSuiteError(
                f"skill {skill['name']} delegates to unknown agents:"
                f" {sorted(unknown)}")

    # 5. Agent definitions & files
    agents_by_name = {agent["name"]: agent for agent in manifest["agents"]}
    for agent in manifest["agents"]:
        for tool_name in agent.get("tools", []):
            package_checks.validate_agent_tool(tool_name, declared_tools,
                                               agent["name"])

        unknown_delegates = (set(agent.get("delegates", [])) - name_set -
                             skill_names)
        if unknown_delegates:
            raise utils.TpuAgentSuiteError(
                f"agent {agent['name']} delegates to unknown agents:"
                f" {unknown_delegates}")

        source_inst = utils.rooted(package_dir, agent["instructions"],
                                   "agent instructions")
        if not source_inst.is_file():
            raise utils.TpuAgentSuiteError(
                f"missing agent instructions: {source_inst}")
        fm, body, inst_content = frontmatter.read_frontmatter_document(
            source_inst)
        if inst_content.lstrip().startswith("---"):
            if fm.get("name") == agent["name"]:
                tokens.reject_plain_package_paths(
                    body, f"agent {agent['name']}",
                    tokens.body_first_line(inst_content, body))
                tokens.render_agent_references(body, manifest)
                package_checks.validate_resource_tokens(
                    body, package_dir, f"agent {agent['name']}",
                    package_discovery.shipped_by(agent), assets)
            else:
                raise utils.TpuAgentSuiteError(
                    f"agent instructions file '{source_inst.name}' "
                    f"frontmatter name '{fm.get('name')}' does not match "
                    f"expected agent name '{agent['name']}'")
        else:
            tokens.reject_plain_package_paths(inst_content,
                                              f"agent {agent['name']}")
            tokens.render_agent_references(inst_content, manifest)
            package_checks.validate_resource_tokens(
                inst_content, package_dir, f"agent {agent['name']}",
                package_discovery.shipped_by(agent), assets)

        for schema_key in ("accepts", "produces"):
            val = agent.get(schema_key)
            if not val:
                continue
            schema_path = utils.rooted(package_dir, val, f"agent {schema_key}")
            if not schema_path.is_file():
                raise utils.TpuAgentSuiteError(
                    f"missing agent {schema_key} schema: {schema_path}")

        _require_unit_files(package_dir, agent, f"agent {agent['name']}")

    # 6. Delegation graph traversal: cycle detection, max depth <= 3,
    #    reachability
    package_checks.validate_delegation_graph(skills, agents_by_name, name_set)

    # 6b. `ask` belongs to skills alone. A skill is what the user talks to; an
    # agent's caller is blocked waiting on its result, so a question raised by
    # an agent reaches nobody -- granting the tool there would compile a
    # promise the harness cannot keep. Checked here rather than left to review
    # because the failure is silent at run time: the agent calls a tool it
    # holds and no one answers. The agent built from a skill that delegates
    # is that skill's own document, so it keeps what the skill holds.
    for agent in agents_by_name.values():
        if agent["name"] in skill_names:
            continue
        offending = sorted(
            set(agent.get("tools") or []) & utils.SKILL_ONLY_TOOLS)
        if offending:
            raise utils.TpuAgentSuiteError(
                f"agent {agent['name']} declares {', '.join(offending)}, which"
                " only a skill may hold: an agent has no user to ask. Return"
                " status needs_input and let the caller ask instead.")

    # 7. Skill documents and their references, then environment files
    for skill in skills:
        label = f"skill {skill['name']}"
        entry_inst = utils.rooted(package_dir, skill["instructions"],
                                  f"{label} instructions")
        if not entry_inst.is_file():
            raise utils.TpuAgentSuiteError(
                f"missing {label} instructions: {entry_inst}")
        # package_discovery.load_skill_document has already required
        # frontmatter and checked the name in it, so only the body is left to
        # check.
        _, body, entry_content = frontmatter.read_frontmatter_document(
            entry_inst)
        tokens.reject_plain_package_paths(
            body, label, tokens.body_first_line(entry_content, body))
        tokens.render_agent_references(body, manifest)
        package_checks.validate_resource_tokens(
            body, package_dir, label, package_discovery.shipped_by(skill),
            assets)
        _require_unit_files(package_dir, skill, label)

    # A markdown reference reaches its reader as a prompt does, so its tokens
    # are held to the same rules as a prompt's. It is checked against its own
    # frontmatter, not its reader's, so once is enough however many read it.
    validated: set[str] = set()
    for unit in (*skills, *manifest["agents"]):
        for ref_file in unit.get("references", []):
            if ref_file in validated:
                continue
            validated.add(ref_file)
            ref_path = package_dir / ref_file
            if ref_path.suffix == ".md" and ref_path.is_file():
                _, body, text = frontmatter.read_frontmatter_document(
                    ref_path, errors="ignore")
                tokens.reject_plain_package_paths(
                    body, f"reference {ref_file}",
                    tokens.body_first_line(text, body))
                tokens.render_agent_references(body, manifest)
                package_checks.validate_resource_tokens(
                    body, package_dir, f"reference {ref_file}",
                    package_discovery.shipped_by_reference(
                        package_dir, ref_file), assets)

    questions = utils.rooted(package_dir, manifest["environment"]["questions"],
                             "environment questions")
    env_schema = utils.rooted(package_dir, manifest["environment"]["schema"],
                              "environment schema")
    if not questions.is_file() or not env_schema.is_file():
        raise utils.TpuAgentSuiteError(
            "environment questions or schema is missing")

    # 8. Hook definitions & validation
    hook_names: set[str] = set()
    for hook in manifest.get("hooks", []):
        h_name = hook.get("name")
        if not h_name:
            raise utils.TpuAgentSuiteError(
                "hook declaration is missing 'name'")
        if h_name in hook_names:
            raise utils.TpuAgentSuiteError(
                f"duplicate hook name declared: '{h_name}'")
        hook_names.add(h_name)

        event = hook.get("event")
        supported_events = utils.get_supported_hook_events()
        if event not in supported_events:
            raise utils.TpuAgentSuiteError(
                f"hook '{h_name}' has unsupported event '{event}'; "
                f"expected one of {supported_events}")

        matchers = hook.get("matchers")
        if matchers is not None:
            if not isinstance(matchers, list) or not all(
                    isinstance(m, str) for m in matchers):
                raise utils.TpuAgentSuiteError(
                    f"hook '{h_name}' matchers must be a list of strings")
            package_checks.check_hook_matchers(h_name, event, matchers)

        action = hook.get("action")
        command = hook.get("command")

        if not action and not command:
            raise utils.TpuAgentSuiteError(
                f"hook '{h_name}' must specify either an 'action' or 'command'"
            )
        if action and command:
            raise utils.TpuAgentSuiteError(
                f"hook '{h_name}' cannot specify both 'action' and 'command'")

        if action:
            if not isinstance(action,
                              str) or not (action.startswith("script:")
                                           or action.startswith("builtin:")):
                raise utils.TpuAgentSuiteError(
                    f"hook '{h_name}' action must start with 'script:' or 'builtin:', got '{action}'"
                )
            if action.startswith("builtin:"):
                builtin_target = action[len("builtin:"):]
                allowed_events = package_checks.VALID_BUILTIN_HOOK_ACTIONS.get(
                    builtin_target)
                if allowed_events is None:
                    raise utils.TpuAgentSuiteError(
                        f"hook '{h_name}' unknown builtin action '{builtin_target}', expected one of {sorted(package_checks.VALID_BUILTIN_HOOK_ACTIONS)}"
                    )
                if event not in allowed_events:
                    raise utils.TpuAgentSuiteError(
                        f"hook '{h_name}' builtin action '{builtin_target}' cannot be attached to event '{event}', expected one of {sorted(allowed_events)}"
                    )
            elif (script_rel := hook_script(action)) is not None:
                script_path = utils.rooted(package_dir, script_rel,
                                           f"hook '{h_name}' script file")
                if not script_path.is_file():
                    raise utils.TpuAgentSuiteError(
                        f"hook '{h_name}' script file does not exist: {script_path}"
                    )
        elif command:
            if not isinstance(command, str) or not command.strip():
                raise utils.TpuAgentSuiteError(
                    f"hook '{h_name}' command must be a non-empty string")

        timeout = hook.get("timeout")
        if timeout is not None and (not isinstance(timeout, int) or isinstance(
                timeout, bool) or timeout <= 0):
            raise utils.TpuAgentSuiteError(
                f"hook '{h_name}' timeout must be a positive integer, got {timeout!r}"
            )

        on_failure = hook.get("on_failure", utils.DEFAULT_HOOK_ON_FAILURE)
        if on_failure not in utils.HOOK_FAILURE_POLICIES:
            raise utils.TpuAgentSuiteError(
                f"hook '{h_name}' on_failure must be one of "
                f"{list(utils.HOOK_FAILURE_POLICIES)}, got {on_failure!r}")

    # 9. Workspace memory block content. A block is prose an agent reads in
    #    every session, so it is held to what a prompt is held to. It belongs
    #    to the package rather than to a unit, so a script or reference token
    #    in one may name anything some unit of this build ships.
    shipped_by_any = [
        path for unit in (*skills, *manifest["agents"])
        for path in package_discovery.shipped_by(unit)
    ]
    for entry in manifest.get("workspace_memory", []):
        content = utils.rooted(package_dir, entry["content"],
                               "workspace memory content")
        if not content.is_file():
            raise utils.TpuAgentSuiteError(
                f"missing workspace memory content file: {content}")
        label = f"workspace memory block {entry['marker']!r}"
        block = content.read_text(encoding="utf-8")
        tokens.reject_plain_package_paths(block, label)
        tokens.render_agent_references(block, manifest)
        package_checks.validate_resource_tokens(block, package_dir, label,
                                                shipped_by_any, assets)

    return manifest


def target_for(manifest: dict[str, Any], harness: str) -> dict[str, Any]:
    """Retrieve and validate the compatibility target configuration for a given harness."""
    target = next(
        (t for t in manifest.get("compatibility_targets", [])
         if t.get("harness") == harness),
        None,
    )
    if not target:
        raise utils.TpuAgentSuiteError(
            f"no compatibility target for harness {harness}")

    versions = target.get("versions")
    if versions is None:
        raise utils.TpuAgentSuiteError(
            f"compatibility target for {harness} is missing 'versions'")
    utils.parse_version_range(versions)

    return target


def get_manifest_agent_map(
    manifest: dict[str, Any], ) -> dict[str, dict[str, Any]]:
    """Index the manifest's agents by name."""
    return {a["name"]: a for a in manifest.get("agents", []) if "name" in a}
