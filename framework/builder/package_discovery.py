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
"""Finds every file a package uses, starting from its skills.

Reads each skill, agent and reference document, checks its frontmatter, and
follows what it declares (agents it calls, references, scripts, schemas) until
nothing new turns up. Also decides which files never ship (`dist_excluded`).
"""

from __future__ import annotations

import fnmatch
import pathlib
from typing import Any, Callable, Iterable, Sequence

from framework import frontmatter, utils

Path = pathlib.Path


def shipped_by(unit: dict[str, Any]) -> list[str]:
    """Everything a skill or agent ships.

    That is its own files and what the references it reads brought.
    """
    return [
        *unit.get("references", []), *unit.get("scripts", []),
        *unit.get("declared_references", []),
        *unit.get("declared_scripts", [])
    ]


def shipped_by_reference(package_dir: Path, rel: str) -> list[str]:
    """Everything a markdown reference ships with it.

    That is itself and what its frontmatter declares, transitively.
    """
    shipped: list[str] = [rel]
    pending, seen = [rel], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        path = package_dir / current
        if path.suffix != ".md" or not path.is_file():
            continue
        frontmatter = load_reference_frontmatter(path)
        shipped.extend(frontmatter.get("scripts", []))
        for entry in frontmatter.get("references", []):
            shipped.append(entry)
            files = reference_files(package_dir, entry,
                                    f"reference {current} reference")
            shipped.extend(files)
            pending.extend(files)
    return _dedupe(shipped)


def _check_frontmatter_schema(fm: dict[str, Any], schema: dict[str, Any],
                              label: str) -> None:
    """Fail unless `fm` satisfies `schema`, naming the document as `label`."""
    schema_errors = utils.validate_instance(fm, schema)
    if schema_errors:
        raise utils.TpuAgentSuiteError(
            f"invalid YAML frontmatter in {label}:\n" +
            "\n".join(schema_errors))


def _load_unit_document(
    doc_path: Path,
    package_dir: Path,
    kind: str,
    get_schema: Callable[[], dict[str, Any]],
    label: str,
    expected: str,
    list_keys: Sequence[str],
    name_subject: str | None = None,
) -> dict[str, Any]:
    """Load a skill or agent from the frontmatter of its document.

    `kind` is "skill" or "agent"; `label` names the document in a schema
    error and `name_subject` (default `label`) in a name mismatch, whose
    frontmatter name must be `expected`. Every key in `list_keys` the
    frontmatter leaves out is an empty list.
    """
    if not doc_path.is_file():
        raise utils.TpuAgentSuiteError(
            f"{kind} document not found: {doc_path}")
    fm, _, _ = frontmatter.read_frontmatter_document(doc_path)
    if not fm:
        raise utils.TpuAgentSuiteError(
            f"{kind} document '{doc_path.name}' must start with '---' YAML"
            " frontmatter")
    _check_frontmatter_schema(fm, get_schema(), label)
    if fm["name"] != expected:
        raise utils.TpuAgentSuiteError(
            f"{name_subject or label} frontmatter name '{fm['name']}' "
            f"does not match expected {kind} name '{expected}'")

    unit = dict(fm)
    unit["instructions"] = relative_to_package(doc_path, package_dir)
    for key in list_keys:
        unit.setdefault(key, [])
    return unit


def load_agent_document(doc_path: Path, package_dir: Path) -> dict[str, Any]:
    """Load an agent definition from a Markdown document with YAML frontmatter."""
    return _load_unit_document(
        doc_path,
        package_dir,
        "agent",
        utils.get_agent_frontmatter_schema,
        label=f"agent document '{doc_path.name}'",
        expected=doc_path.stem,
        list_keys=("tools", "delegates", "scripts", "references"),
        name_subject=f"agent document filename '{doc_path.name}'",
    )


def relative_to_package(doc_path: Path, package_dir: Path) -> str:
    """`doc_path` as a POSIX path relative to `package_dir`, else unchanged."""
    try:
        return doc_path.resolve().relative_to(package_dir.resolve()).as_posix()
    except ValueError:
        return str(doc_path)


def load_skill_document(doc_path: Path, package_dir: Path) -> dict[str, Any]:
    """Load a skill from its SKILL.md.

    The frontmatter is the whole declaration: description, delegates,
    references, scripts and dependencies. The skill's name is the directory
    that holds SKILL.md, and the frontmatter must agree with it.
    """
    return _load_unit_document(
        doc_path,
        package_dir,
        "skill",
        utils.get_skill_frontmatter_schema,
        label=f"skill document '{doc_path}'",
        expected=doc_path.parent.name,
        list_keys=("tools", "delegates", "scripts", "references",
                   "host_dependencies", "tpu_dependencies"),
    )


def load_reference_frontmatter(doc_path: Path) -> dict[str, Any]:
    """The frontmatter of a reference document, or {} when it has none.

    A reference needs no frontmatter at all; when it has one it may bring
    scripts, further references, delegates and dependencies, all of which the
    compiler adds to every unit that reads it.
    """
    if doc_path.suffix != ".md":
        return {}
    fm, _, content = frontmatter.read_frontmatter_document(doc_path,
                                                           errors="ignore")
    if not content.lstrip().startswith("---"):
        return {}
    _check_frontmatter_schema(fm, utils.get_reference_frontmatter_schema(),
                              f"reference '{doc_path}'")
    for key in ("delegates", "scripts", "references", "host_dependencies",
                "tpu_dependencies"):
        fm.setdefault(key, [])
    return fm


def reference_files(package_dir: Path, rel: str, label: str) -> list[str]:
    """The files a `references:` entry brings.

    That is the entry itself, or every shipped file under it when it names a
    directory.
    """
    source = utils.rooted(package_dir, rel, label)
    if source.is_dir():
        return sorted(
            relative_to_package(path, package_dir)
            for path in source.rglob("*") if path.is_file()
            and not dist_excluded(path.relative_to(package_dir)))
    if source.is_file():
        return [rel]
    raise utils.TpuAgentSuiteError(f"{label} does not exist: {source}")


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def discover_units(package_dir: Path, manifest: dict[str, Any]) -> None:
    """Resolve the package's skills, then everything reachable from them.

    `skills` in the manifest names the roots. From each root the compiler
    follows `delegates` to agents (`agents/<name>.md`) or to other skills,
    and `references` to files under `references/`, reading the
    frontmatter of every markdown reference on the way: what a reference
    declares -- scripts, further references, delegates, dependencies -- is
    added to every unit that reads it, transitively. Directory references
    bring every file under them.

    On return each skill and agent carries flat lists: `references` and
    `scripts` are the files it ships, `delegates` the names it may invoke
    (its own plus those its references named, kept apart in
    `own_delegates`), and `declared_references` what it wrote. The manifest
    gains the union of dependencies, every `schemas/*.json`, and a twin agent
    for each skill that delegates, so a harness that starts from an agent
    (`--agent <skill>`) finds one carrying the same delegation contract.
    """
    skills: list[dict[str, Any]] = []
    for item in manifest.get("skills", []):
        doc = utils.rooted(
            package_dir,
            f"{utils.SKILLS_DIRNAME}/{item}/{utils.SKILL_DOCUMENT}",
            f"skill {item}")
        if not doc.is_file():
            raise utils.TpuAgentSuiteError(
                f"skill {item!r} is named but {utils.SKILLS_DIRNAME}/{item}/"
                f"{utils.SKILL_DOCUMENT} does not exist")
        skills.append(load_skill_document(doc, package_dir))
    if not skills:
        raise utils.TpuAgentSuiteError(
            "package releases no skills: name at least one under `skills`")
    names = [skill["name"] for skill in skills]
    if len(names) != len(set(names)):
        raise utils.TpuAgentSuiteError(f"skill names are not unique: {names}")
    skill_names = set(names)
    delegated_skills: list[str] = []

    agents: dict[str, dict[str, Any]] = {}
    host_deps: list[str] = []
    tpu_deps: list[str] = []
    reference_cache: dict[str, dict[str, Any]] = {}

    def closure(unit: dict[str, Any], label: str) -> None:
        """Expand one unit's references transitively, in place."""
        unit.setdefault("own_delegates", list(unit.get("delegates", [])))
        unit.setdefault("declared_references", list(unit.get("references",
                                                             [])))
        unit.setdefault("declared_scripts", list(unit.get("scripts", [])))
        files: list[str] = []
        scripts: list[str] = list(unit.get("scripts", []))
        delegates: list[str] = list(unit.get("delegates", []))
        pending = list(unit.get("references", []))
        seen: set[str] = set()
        while pending:
            rel = pending.pop(0)
            for path in reference_files(package_dir, rel,
                                        f"{label} reference"):
                if path in seen:
                    continue
                seen.add(path)
                files.append(path)
                if path not in reference_cache:
                    reference_cache[path] = load_reference_frontmatter(
                        package_dir / path)
                fm = reference_cache[path]
                scripts.extend(fm.get("scripts", []))
                delegates.extend(fm.get("delegates", []))
                host_deps.extend(fm.get("host_dependencies", []))
                tpu_deps.extend(fm.get("tpu_dependencies", []))
                pending.extend(r for r in fm.get("references", [])
                               if r not in seen)
        unit["references"] = _dedupe(files)
        unit["scripts"] = _dedupe(scripts)
        unit["delegates"] = _dedupe(delegates)
        host_deps.extend(unit.get("host_dependencies", []))
        tpu_deps.extend(unit.get("tpu_dependencies", []))

    worklist: list[tuple[dict[str, Any],
                         str]] = [(skill, f"skill {skill['name']}")
                                  for skill in skills]
    while worklist:
        unit, label = worklist.pop(0)
        closure(unit, label)
        for delegate in unit["delegates"]:
            if delegate in agents or delegate in skill_names:
                continue
            doc = package_dir / utils.AGENTS_DIRNAME / f"{delegate}.md"
            if doc.is_file():
                agent = load_agent_document(doc, package_dir)
                agents[delegate] = agent
                worklist.append((agent, f"agent {delegate}"))
                continue
            # A skill may delegate to a skill. One that is reached is built,
            # whether or not the list named it: a build holds what it reaches.
            doc = package_dir / utils.SKILLS_DIRNAME / delegate / utils.SKILL_DOCUMENT
            if doc.is_file():
                skill = load_skill_document(doc, package_dir)
                skills.append(skill)
                skill_names.add(delegate)
                delegated_skills.append(delegate)
                worklist.append((skill, f"skill {delegate}"))
            # Otherwise not an error here: `package_loader.load_package`
            # reports an unresolved delegate against the unit that named it.

    # A skill that delegates is also selectable as an agent (`--agent <skill>`)
    # carrying the same delegation contract.
    for skill in skills:
        if skill["delegates"] and skill["name"] not in agents:
            agents[skill["name"]] = dict(skill)

    manifest["skills"] = skills
    manifest["delegated_skills"] = delegated_skills
    manifest["agents"] = list(agents.values())
    manifest["host_dependencies"] = _dedupe(host_deps)
    manifest["tpu_dependencies"] = _dedupe(tpu_deps)
    schemas_dir = package_dir / "schemas"
    manifest["schemas"] = sorted(
        f"schemas/{path.name}" for path in schemas_dir.glob(
            "*.json")) if schemas_dir.is_dir() else []


DIST_EXCLUDED_NAMES: tuple[str, ...] = (
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    ".git",
    "instructions",
    # wikify-repo working state and agent-skill installs next to a wiki
    ".cache",
    ".wikify",
    ".agents",
    ".claude",
)

# Paths relative to a copied asset root that never ship, whatever they hold. The
# knowledge base pins source checkouts (some private) under raw/code and cites
# them by URL at the pinned commit, so the dist needs the wiki pages, never the
# code. Matched by path, not by name: a "code" directory anywhere else
# (wiki/code, ...) still ships.
DIST_EXCLUDED_PATHS: tuple[str, ...] = ("raw/code", )


def dist_excluded(rel_path: Path) -> bool:
    """True when a package-relative path never ships.

    It never ships if it has an excluded name anywhere in it, or an excluded
    path pair (`raw/code`) under any asset root. Used by verify so files inside
    a pinned source checkout (their own requirements files, for one) are not
    the package's.
    """
    parts = rel_path.parts
    if any(
            fnmatch.fnmatch(part, pat) for part in parts
            for pat in DIST_EXCLUDED_NAMES):
        return True
    joined = "/" + "/".join(parts) + "/"
    return any(f"/{excluded}/" in joined for excluded in DIST_EXCLUDED_PATHS)
