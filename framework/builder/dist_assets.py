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
"""Copies everything besides agents and skills into a build's output folder.

That covers the runtime `framework/` (minus build-only files), the package's
scripts, references, data and schemas, the README and docs, the trimmed
requirements files, and the install plan that `installer.py` follows.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
from typing import Any, Callable, Sequence

from framework import compute, frontmatter, harnesses, paths, utils
from framework.builder import (harness_rules, hook_config, package_discovery,
                               package_loader, tokens)

Path = pathlib.Path

# Files under the repository's docs/ that every distribution ships in docs/.
DISTRIBUTION_DOCS = ("install.md", "using.md")

# The package's two requirements catalogs: the file, the manifest key that
# names what a build needs from it, and the word a message calls it by.
REQUIREMENTS_CATALOGS: tuple[tuple[str, str, str], ...] = (
    ("requirements-client.txt", "host_dependencies", "host"),
    ("requirements-tpu.txt", "tpu_dependencies", "TPU"),
)


def dist_ignore(source: Path) -> Callable[[str, list[str]], set[str]]:
    """copytree ignore callable: excluded names anywhere, excluded paths under source."""
    by_name = shutil.ignore_patterns(*package_discovery.DIST_EXCLUDED_NAMES)
    excluded = {Path(p) for p in package_discovery.DIST_EXCLUDED_PATHS}

    def ignore(directory: str, names: list[str]) -> set[str]:
        skipped = set(by_name(directory, names))
        rel = Path(os.path.relpath(directory, source))
        skipped.update(name for name in names if (rel / name) in excluded)
        return skipped

    return ignore


def copy_asset(source: Path, destination: Path) -> None:
    """Copy a file or directory tree to destination, creating parent directories as needed."""
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            dirs_exist_ok=True,
            ignore=dist_ignore(source),
        )
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    else:
        raise utils.TpuAgentSuiteError(f"copy path does not exist: {source}")


# What a build leaves out of framework/: the build tooling, which only runs in
# the repository, the schemas only it reads, and what is for maintainers.
BUILD_ONLY = ("compiler.py", "builder", "lint.py", "frontmatter.py",
              "definitions", "templates", "tests",
              "package-manifest.schema.json", "*-frontmatter.schema.json",
              "*.md", "__pycache__", "*.pyc")


def bundle_framework(destination: Path) -> None:
    """Ship framework/ as is, minus BUILD_ONLY: the runtime and the installer.

    A file keeps its path: framework/compute/cli.py in the repository is
    framework/compute/cli.py in a build and in the central store. Markdown
    stays behind too; compute's setup guide ships rendered under
    references/ (copy_runtime_assets).
    """
    shutil.copytree(utils.get_framework_dir(),
                    destination,
                    ignore=shutil.ignore_patterns(*BUILD_ONLY))
    repo_drivers = compute.drivers.external.repo_google_drivers_dir()
    if repo_drivers is not None and compute.drivers.registry.EXTERNAL_DRIVERS:
        resolved_repo_drivers = repo_drivers.resolve()
        drivers_dest = destination / "compute" / "drivers"
        copied_sources: set[Path] = set()
        for src_str in compute.drivers.registry.EXTERNAL_DRIVERS.values():
            src = Path(src_str).resolve()
            if src in copied_sources or src.parent != resolved_repo_drivers:
                continue
            copied_sources.add(src)
            if src.is_file():
                shutil.copy2(src, drivers_dest / src.name)
            elif src.is_dir():
                shutil.copytree(src,
                                drivers_dest / src.name,
                                ignore=shutil.ignore_patterns(
                                    "__pycache__", "*.pyc"))
    installer = destination / "installer.py"
    installer.chmod(installer.stat().st_mode | 0o111)


def bundle_runtime_schemas(schemas_dir: Path) -> None:
    """Copy the study schemas into the distribution's schemas/.

    The runtime reads every framework schema from framework/schemas/; these
    are here too because the package's own schemas and prompts name them
    beside its request and result schemas.
    """
    schemas_dir.mkdir(parents=True, exist_ok=True)
    framework_schemas_dir = utils.get_schemas_dir()
    for schema_name in ("study-protocol.json", "study-round.json",
                        "study-baseline.json"):
        src = framework_schemas_dir / schema_name
        if src.is_file():
            shutil.copy2(src, schemas_dir / schema_name)


def copy_runtime_assets(package_dir: Path, output: Path,
                        manifest: dict[str, Any], harness: str) -> None:
    """Copy configured package assets, agent references/scripts, and bundle framework runtime worker scripts."""
    for item in manifest.get("copy", []):
        rel = utils.safe_relative(item, "copy path")
        source = utils.rooted(package_dir, str(rel), "copy path")
        copy_asset(source, output / rel)

    # Memory-block sources are compiler input -- rendered and spliced into the
    # harness memory file at install time -- not distributable content, so a
    # copy tree that contains one sheds it (a verbatim copy would carry raw
    # resource tokens).
    for entry in manifest.get("workspace_memory", []):
        shipped = output / utils.safe_relative(entry["content"],
                                               "workspace memory content")
        shipped.unlink(missing_ok=True)

    shipped: set[str] = set()
    units = [
        *[s for s in manifest.get("skills", []) if isinstance(s, dict)],
        *manifest.get("agents", []),
    ]
    for unit in units:
        for ref_file in unit.get("references", []):
            if ref_file in shipped:
                continue
            shipped.add(ref_file)
            rel = utils.safe_relative(ref_file, "reference path")
            source = utils.rooted(package_dir, str(rel), "reference path")
            # Markdown references reach the reading agent exactly as a prompt
            # does, so every token a prompt may use resolves here too, the
            # harness's included, and the frontmatter, which is compiler
            # input, is left behind; other assets ship verbatim.
            if source.is_file() and source.suffix == ".md":
                _, text, _ = frontmatter.read_frontmatter_document(source)
                destination = output / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(tokens.render_markdown(
                    text, manifest, harness),
                                       encoding="utf-8")
            else:
                copy_asset(source, output / rel)

        for script_file in unit.get("scripts", []):
            if script_file in shipped:
                continue
            shipped.add(script_file)
            rel = utils.safe_relative(script_file, "script path")
            source = utils.rooted(package_dir, str(rel), "script path")
            copy_asset(source, output / rel)

    for agent in package_loader.get_manifest_agent_map(manifest).values():
        for s_key in ("accepts", "produces"):
            s_file = agent.get(s_key)
            if s_file:
                rel = utils.safe_relative(s_file, f"agent {s_key} schema path")
                source = utils.rooted(package_dir, str(rel),
                                      f"agent {s_key} schema path")
                copy_asset(source, output / rel)

    for hook in manifest.get("hooks", []):
        script_rel = package_loader.hook_script(hook.get("action", ""))
        if script_rel is not None:
            rel = utils.safe_relative(script_rel, "hook script path")
            source = utils.rooted(package_dir, str(rel), "hook script path")
            copy_asset(source, output / rel)

    for script in manifest.get("script_tools", []):
        if "path" in script:
            rel = utils.safe_relative(script["path"], "script tool path")
            source = utils.rooted(package_dir, str(rel), "script tool path")
            copy_asset(source, output / rel)

    bundle_framework(output / "framework")
    bundle_runtime_schemas(output / "schemas")
    # Session start and the compute contract point at the setup guide, so it
    # ships in every build, whichever skills the build holds.
    guide = output / utils.REFERENCES_DIRNAME / compute.SETUP_GUIDE.name
    guide.parent.mkdir(parents=True, exist_ok=True)
    guide.write_text(tokens.render_markdown(
        compute.SETUP_GUIDE.read_text(encoding="utf-8"), manifest, harness),
                     encoding="utf-8")


def canonical_pkg_name(name: str) -> str:
    """Normalize a package name according to PEP 503 (lowercase, dashes)."""
    stripped = name.strip()
    if not stripped or stripped.startswith("-"):
        return ""
    base = re.split(r"[><=~!;\s\[]", stripped)[0]
    return base.lower().replace("_", "-")


def extract_requirement_names(requirements_text: str) -> set[str]:
    """Extract canonical package names declared in a requirements file."""
    names: set[str] = set()
    for line in requirements_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        pkg = canonical_pkg_name(line)
        if pkg:
            names.add(pkg)
    return names


def filter_requirements_content(source_text: str,
                                allowed_names: set[str]) -> str:
    """Filter requirements lines by package name, preserving comments and header."""
    canonical_allowed = {
        canonical_pkg_name(name)
        for name in allowed_names if name.strip()
    }
    output_lines: list[str] = []
    has_requirements = False
    for line in source_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            continue
        if stripped.startswith("#") or stripped.startswith("-"):
            output_lines.append(line)
            continue
        pkg_name = canonical_pkg_name(stripped)
        if pkg_name in canonical_allowed:
            output_lines.append(line)
            has_requirements = True

    while output_lines and output_lines[-1] == "":
        output_lines.pop()
    if not output_lines or not has_requirements:
        return "# No dependencies required for this distribution.\n"
    return "\n".join(output_lines) + "\n"


def undeclared_catalog_entries(package_dir: Path,
                               manifest: dict[str, Any]) -> list[str]:
    """Catalog entries no built unit declares, as warnings.

    A catalog pins versions; frontmatter says what is needed. An entry that
    nothing declares never ships, which is either a package nothing uses or a
    declaration somebody forgot. Neither can be told from here, so it is
    reported and fails nothing.
    """
    warnings: list[str] = []
    for filename, key, _ in REQUIREMENTS_CATALOGS:
        catalog = package_dir / filename
        if not catalog.is_file():
            continue
        declared = {canonical_pkg_name(name) for name in manifest.get(key, [])}
        listed = extract_requirement_names(catalog.read_text(encoding="utf-8"))
        for name in sorted(listed - declared):
            warnings.append(
                f"{filename} lists {name}, which no built unit declares"
                f" under {key}, so it does not ship")
    return warnings


def emit_distribution_requirements(
    package_dir: Path,
    output: Path,
    manifest: dict[str, Any],
) -> None:
    """Emit full or trimmed requirements files into the distribution root."""
    for filename, key, _ in REQUIREMENTS_CATALOGS:
        catalog = package_dir / filename
        if not catalog.is_file():
            continue
        content = catalog.read_text(encoding="utf-8")
        if key in manifest:
            content = filter_requirements_content(content,
                                                  set(manifest.get(key, [])))
        (output / filename).write_text(content, encoding="utf-8")


def _skill_copies(skill_names: Sequence[str], source_root: str,
                  destination_root: str) -> list[dict[str, str]]:
    """One copy rule per shipped skill."""
    return [{
        "source": f"{source_root}/{name}",
        "destination": f"{destination_root}/{name}",
    } for name in skill_names]


def _google_install_spec(dest_root: str,
                         skill_names: Sequence[str]) -> dict[str, Any]:
    """Build install spec for Google agent harnesses (Antigravity)."""
    return {
        "version_command": ["agy", "--version"],
        "memory_file":
        "AGENTS.md",
        "copies": [
            {
                "source": "plugin.json",
                "destination": f"{dest_root}/plugin.json",
            },
            *_skill_copies(skill_names, "skills", f"{dest_root}/skills"),
            {
                "source": "agents",
                "destination": f"{dest_root}/agents",
            },
        ],
        "hook_copies": [
            {
                "source": "hooks.json",
                "destination": f"{dest_root}/hooks.json",
            },
        ],
        "json_merges": [],
        "toml_sets": [],
    }


def harness_install_specs(
    skill_names: Sequence[str] = (),
    package_name: str = "",
) -> dict[str, dict[str, Any]]:
    """Return harness-specific copy rules, version command, and configuration settings.

    `skill_names` lists every skill the distribution ships; each gets one
    copy rule per harness.
    """
    specs: dict[str, dict[str, Any]] = {
        "claude-code": {
            "version_command": ["claude", "--version"],
            "memory_file":
            "CLAUDE.md",
            "copies": [
                *_skill_copies(skill_names, "project/skills",
                               ".claude/skills"),
                {
                    "source": "project/agents",
                    "destination": ".claude/agents",
                },
            ],
            "hook_copies": [{
                "source": "project/hooks.json",
                "destination": ".claude/hooks.json",
            }],
            "json_merges": [{
                "path": ".claude/settings.local.json",
                "value": {
                    "env": dict(harness_rules.CLAUDE_CODE_SETTINGS_ENV),
                    "approved_directories":
                    [paths.TPU_AGENT_SUITE_HOME_SPELLING],
                },
            }],
            "toml_sets": [],
        },
        "codex": {
            "version_command": ["codex", "--version"],
            "memory_file":
            "AGENTS.md",
            "copies": [
                *_skill_copies(skill_names, "skills", ".agents/skills"),
                {
                    "source": "agents",
                    "destination": ".codex/agents",
                },
            ],
            "hook_copies": [{
                "source": ".codex/hooks.json",
                "destination": ".codex/hooks.json",
            }],
            "json_merges": [],
            "toml_sets": [
                {
                    "path": ".codex/config.toml",
                    "section": "agents",
                    "key": "enabled",
                    "value": True,
                },
                {
                    "path": ".codex/config.toml",
                    "section": "agents",
                    "key": "max_concurrent_threads_per_session",
                    "value": harness_rules.MAX_CONCURRENT_DELEGATES,
                },
                {
                    "path": ".codex/config.toml",
                    "section": "sandbox",
                    "key": "allowed_paths",
                    "value": [paths.TPU_AGENT_SUITE_HOME_SPELLING],
                },
            ],
        },
        "cursor": {
            "version_command": ["cursor", "--version"],
            "copies": [
                *_skill_copies(skill_names, "skills", ".cursor/skills"),
                {
                    "source": f".cursor/rules/{package_name}.mdc",
                    "destination": f".cursor/rules/{package_name}.mdc",
                },
            ],
            "hook_copies": [{
                "source": ".cursor/hooks.json",
                "destination": ".cursor/hooks.json",
            }],
            "json_merges": [],
            "toml_sets": [],
        },
    }

    # `.agents`, not `.agent`. The Google harnesses accept either as a general
    # customization root, but they discover *subagents* only at
    # `.agents/agents/<name>.md` and `~/.gemini/config/agents/`. Installing
    # to `.agent/agents/` leaves every agent on disk and invisible, and the
    # orchestrator's `invoke_subagent` call fails with "not found or not
    # allowed to be invoked" -- after which running the agent's
    # instructions inline is the only path left.
    #
    # One spec for the family: those discovery paths are the family's, so a
    # per-member entry would be a copy of this one under another name.
    for google_harness in sorted(harnesses.GOOGLE_HARNESSES):
        specs[google_harness] = _google_install_spec(".agents", skill_names)

    return specs


# Opens every workspace memory block. A memory file loads in every session,
# so without it a block reads as a rule for all work, not only the package's.
# It scopes by skill names and the task because those are what a harness
# shows the model; neither harness knows what a "package" is. `{skills}`
# lists every released skill, since no one skill is the main one.
MEMORY_SCOPE_PREAMBLE = (
    "This applies only inside the {skills} {noun} or {their} agents, or on a"
    " task {those} {cover}, even before you load {one}. For anything else,"
    " ignore this and just do the task.")

# The words in MEMORY_SCOPE_PREAMBLE that agree with how many skills it names.
_PREAMBLE_FORMS = {
    "singular": {
        "noun": "skill",
        "their": "its",
        "those": "that skill",
        "cover": "covers",
        "one": "it"
    },
    "plural": {
        "noun": "skills",
        "their": "their",
        "those": "those skills",
        "cover": "cover",
        "one": "one"
    },
}


def memory_scope_preamble(manifest: dict[str, Any]) -> str:
    """Render MEMORY_SCOPE_PREAMBLE for the package's released skills.

  The names read as an English list: `a`; `a` or `b`; `a`, `b`, or `c`.
  """
    names = [f"`{s['name']}`" for s in manifest["skills"]]
    if len(names) <= 2:
        skills = " or ".join(names)
    else:
        skills = f"{', '.join(names[:-1])}, or {names[-1]}"
    forms = _PREAMBLE_FORMS["singular" if len(names) == 1 else "plural"]
    return MEMORY_SCOPE_PREAMBLE.format(skills=skills, **forms)


def workspace_memory_merges(
    package_dir: Path,
    manifest: dict[str, Any],
    memory_file: str,
) -> list[dict[str, str]]:
    """Build managed workspace memory blocks declared by the package manifest.

  The block text is package content rather than framework text, so it is
  reviewed and versioned alongside the material it points at, and is inlined
  here so the installer never needs the package sources. Resource references
  render through the same token registry as every instruction file, at
  compile time; the installer applies blocks verbatim. Each block opens with
  MEMORY_SCOPE_PREAMBLE, so no block has to state its own scope.
  """
    preamble = memory_scope_preamble(manifest)
    merges: list[dict[str, str]] = []
    for entry in manifest.get("workspace_memory", []):
        content_path = utils.rooted(package_dir, entry["content"],
                                    "workspace memory content")
        content = tokens.render_agent_references(
            content_path.read_text(encoding="utf-8").strip(), manifest)

        # Memory files load in every session, from arbitrary working
        # directories (a user-scope CLAUDE.md most of all), so only
        # location-independent references survive there: the workspace
        # environment slot must not appear in a block.
        env_path = f".tpu-agent-suite/{paths.ENVIRONMENT_FILE}"
        if env_path in content:
            raise utils.TpuAgentSuiteError(
                f"workspace memory block '{entry['marker']}' references the"
                " workspace environment file; memory blocks may reference"
                " central resources only")

        merges.append({
            "path": memory_file,
            "marker": f"tpu-agent-suite:{manifest['name']}:{entry['marker']}",
            "content": f"{preamble}\n\n{content}",
        })
    return merges


def build_install_plan(
    package_dir: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    harness: str,
) -> dict[str, Any]:
    """Generate tpu-agent-suite-install.json installation plan for target harness."""
    name = manifest["name"]
    has_hooks = bool(manifest.get("hooks"))

    asset_roots: set[str] = set()
    for item in manifest.get("copy", []):
        if Path(item).parts:
            asset_roots.add(Path(item).parts[0])

    units = [
        *[s for s in manifest.get("skills", []) if isinstance(s, dict)],
        *package_loader.get_manifest_agent_map(manifest).values(),
    ]
    for agent in units:
        for ref_file in agent.get("references", []):
            if Path(ref_file).parts:
                asset_roots.add(Path(ref_file).parts[0])
        for script_file in agent.get("scripts", []):
            if Path(script_file).parts:
                asset_roots.add(Path(script_file).parts[0])
        for s_key in ("accepts", "produces"):
            s_file = agent.get(s_key, "")
            if Path(s_file).parts:
                asset_roots.add(Path(s_file).parts[0])

    for hook in manifest.get("hooks", []):
        rel = package_loader.hook_script(hook.get("action", ""))
        if rel is not None:
            if Path(rel).parts:
                asset_roots.add(Path(rel).parts[0])

    for script in manifest.get("script_tools", []):
        script_path = script.get("path", "")
        if Path(script_path).parts:
            asset_roots.add(Path(script_path).parts[0])

    skill_names = [skill["name"] for skill in manifest["skills"]]
    spec = harness_install_specs(skill_names, name)[harness]
    # Placement is declared per entry, never inferred from the path: harness
    # files (skills, agents, hooks) are workspace-store; the payload -- the
    # embedded runtime, contract schemas, build identity, and every shipped
    # asset root -- is central-store, one immutable copy per version with
    # store-relative destinations (the version appears once, in the plan's
    # package identity). The installer roots each entry at its store and
    # rejects entries that do not declare one.
    # The installer travels with the payload, so an install can always be
    # removed by the installer of the version that wrote its ledger, without
    # the distribution it came from. docs/ is copied for the same reason:
    # install.md and using.md remain readable after the distribution is gone.
    payload_sources = {
        "docs", "framework", "schemas", "tpu-agent-suite-build.json"
    } | set(asset_roots)
    if (package_dir / "requirements-tpu.txt").is_file():
        # What `compute setup verify` checks the target's software against.
        payload_sources.add("requirements-tpu.txt")
    if (package_dir / "requirements-client.txt").is_file():
        # What session start checks this machine's software against.
        payload_sources.add("requirements-client.txt")
    copies = [{
        **mapping, "store": "workspace"
    } for mapping in spec["copies"]] + [{
        "source": src,
        "destination": src,
        "store": "central",
    } for src in sorted(payload_sources)]
    if not manifest["agents"]:
        # A build whose skills delegate to nothing emits no agent files, so
        # the harness's agents mapping would point the installer at a
        # directory the distribution does not contain.
        copies = [
            c for c in copies
            if pathlib.PurePosixPath(c["source"]).name != "agents"
        ]
    json_merges = [dict(m) for m in spec["json_merges"]]
    toml_sets = [dict(t) for t in spec["toml_sets"]]

    if has_hooks:
        copies.extend({
            **mapping, "store": "workspace"
        } for mapping in spec.get("hook_copies", []))

        if harness == "claude-code":
            hooks_config = hook_config.build_harness_hooks_config(
                name, manifest, "claude-code")
            if hooks_config:
                claude_settings_hooks: dict[str, list[Any]] = {}
                for h_events in hooks_config.values():
                    for evt, evt_list in h_events.items():
                        if evt not in claude_settings_hooks:
                            claude_settings_hooks[evt] = []
                        claude_settings_hooks[evt].extend(evt_list)

                # Merge hooks into .claude/settings.local.json alongside env settings
                merged_into_existing = False
                for m in json_merges:
                    if m.get("path"
                             ) == ".claude/settings.local.json" and isinstance(
                                 m.get("value"), dict):
                        m["value"]["hooks"] = claude_settings_hooks
                        merged_into_existing = True
                        break
                if not merged_into_existing:
                    json_merges.append({
                        "path": ".claude/settings.local.json",
                        "value": {
                            "hooks": claude_settings_hooks
                        },
                    })

        elif harness == "codex":
            toml_sets.append({
                "path": ".codex/config.toml",
                "section": "hooks",
                "key": "enabled",
                "value": True,
            })

    return {
        "harness":
        harness,
        "version_range":
        target["versions"],
        "environment_path":
        f".tpu-agent-suite/{paths.ENVIRONMENT_FILE}",
        "ownership_path":
        f".tpu-agent-suite/{paths.OWNERSHIP_FILE}",
        "version_command":
        spec["version_command"],
        "copies":
        copies,
        "json_merges":
        json_merges,
        "markdown_merges":
        (workspace_memory_merges(package_dir, manifest, spec["memory_file"])
         if spec.get("memory_file") else []),
        "toml_sets":
        toml_sets,
    }


def bundle_standalone_distribution(
    output: Path,
    manifest: dict[str, Any],
    harness: str,
) -> None:
    """Bundle the standalone installer and documentation into the distribution."""
    framework_dir = utils.get_framework_dir()

    # The installer ships in framework/ (bundle_framework).

    # Generate README.md
    display_name = harnesses.HARNESS_DISPLAY_NAMES.get(harness, harness)
    harness_dir = harnesses.HARNESS_CONFIG_DIRS.get(harness, f".{harness}")
    pkg_name = manifest.get("name", "tpu-agent-suite")
    version = manifest.get("version", utils.TPU_AGENT_SUITE_VERSION)
    desc = manifest.get("description", "").strip()

    template_path = framework_dir / "templates" / "README.md"
    readme_content = template_path.read_text(encoding="utf-8").format(
        title_name=manifest.get("display_name", pkg_name),
        display_name=display_name,
        version=version,
        desc=desc,
        harness_dir=harness_dir,
    )
    (output / "README.md").write_text(readme_content, encoding="utf-8")

    # The README names these docs, so the build fails if one is missing.
    docs_dir = framework_dir.parent / "docs"
    (output / "docs").mkdir(exist_ok=True)
    for doc in DISTRIBUTION_DOCS:
        source = docs_dir / doc
        if not source.is_file():
            raise utils.TpuAgentSuiteError(
                f"Distribution doc not found: {source}")
        shutil.copy2(source, output / "docs" / doc)
