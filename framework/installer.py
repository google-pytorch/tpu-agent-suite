#!/usr/bin/env python3
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
"""Installer module for TPU agent suite framework.

Handles installing, updating, and cleanly uninstalling translated distributions
in destination workspaces with atomic ownership tracking and rollback support.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Any, Sequence

# Run as a script, this file has its own directory on sys.path, not the one
# that holds the framework package: in the repository the root, in a distribution
# or the central store the package directory. Add it, so `framework` imports.
_PACKAGE_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from framework import harnesses, jsonio, paths, utils  # noqa: E402

Path = pathlib.Path

TpuAgentSuiteError = utils.TpuAgentSuiteError
TPU_AGENT_SUITE_VERSION = utils.TPU_AGENT_SUITE_VERSION
JsonConfigManager = utils.JsonConfigManager
MarkdownBlockManager = utils.MarkdownBlockManager
SemVer = utils.SemVer
TomlConfigManager = utils.TomlConfigManager
compute_digest = utils.compute_digest
digest_text = utils.digest_text
rooted = utils.rooted
safe_relative = utils.safe_relative
validate_package_name = utils.validate_package_name
version_satisfies = utils.version_satisfies

# =====================================================================
# Harness Detection & File Expansion
# =====================================================================


def check_harness(plan: dict[str, Any]) -> str:
    """Execute harness version command and assert compatibility against plan version range."""
    if not plan.get("version_command"):
        return "active"
    try:
        result = subprocess.run(
            plan["version_command"],
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TpuAgentSuiteError(
            f"cannot detect {plan['harness']}: {exc}") from exc

    output = (result.stdout or result.stderr).strip()
    if not version_satisfies(SemVer.parse(output), plan["version_range"]):
        raise TpuAgentSuiteError(
            f"installed {plan['harness']} version {output!r} is outside"
            f" {plan['version_range']}")
    return output


def expand_copies(
    distribution: Path,
    destination: Path,
    plan: dict[str, Any],
    central_root: Path,
) -> list[tuple[Path, Path, str]]:
    """Expand copy specifications into (source_file, dest_file, store) triples.

    Each plan entry declares its store; the entry's destination is relative to
    that store's root (the workspace for "workspace", the central package
    directory for "central"). An entry without a known store is a plan error,
    not a guess -- placement is declared, never inferred from paths.
    """
    roots = {"workspace": destination, "central": central_root}
    expanded: list[tuple[Path, Path, str]] = []
    seen: set[Path] = set()

    for mapping in plan["copies"]:
        store = mapping.get("store")
        if store not in roots:
            raise TpuAgentSuiteError(
                f"install plan entry {mapping.get('destination')!r} declares"
                f" no valid store (got {store!r}); rebuild the distribution")
        source_rel = safe_relative(mapping["source"], "copy source")
        destination_rel = safe_relative(mapping["destination"],
                                        "copy destination")

        source = (distribution if str(source_rel) == "." else rooted(
            distribution, str(source_rel), "copy source"))
        target = rooted(roots[store], str(destination_rel), "copy destination")

        if source.is_file():
            pairs = [(source, target)]
        elif source.is_dir():
            pairs = [(item, target / item.relative_to(source))
                     for item in source.rglob("*") if item.is_file()]
        else:
            raise TpuAgentSuiteError(
                f"install source does not exist: {source}")

        for _, installed in pairs:
            if installed in seen:
                raise TpuAgentSuiteError(
                    f"install plan writes the same file twice: {installed}")
            seen.add(installed)

        expanded.extend((src, tgt, store) for src, tgt in pairs)

    return expanded


# =====================================================================
# Ownership Management & Rollback
# =====================================================================


def prune_empty_parents(destination: Path, owned: Sequence[str]) -> None:
    """Remove the directories that held `owned`, where that left them empty.

    Walks up from each path toward `destination`, stopping at the first
    directory that still holds something, and never removes `destination`
    itself. Scoped to what the package placed rather than sweeping the tree:
    a user-scope destination is $HOME, and an empty directory elsewhere in it
    is not the package's to delete.
    """
    for relative in owned:
        parent = rooted(destination, relative, "owned path").parent
        while destination in parent.parents:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def check_action_modification(destination: Path,
                              action: dict[str, Any]) -> str | None:
    """Check if an ownership action target has been modified by the user."""
    path = rooted(destination, action["path"], "owned path")
    kind = action["kind"]

    if kind == "file":
        if path.is_file() and compute_digest(path) != action["sha256"]:
            return action["path"]
    elif kind == "json_value" and path.is_file():
        exists, value = JsonConfigManager.get_at_path(jsonio.load_json(path),
                                                      action["keys"])
        if exists and value != action["value"]:
            return f"{action['path']}:{'/'.join(action['keys'])}"
    elif kind == "json_container" and path.is_file():
        exists, value = JsonConfigManager.get_at_path(jsonio.load_json(path),
                                                      action["keys"])
        if exists and not isinstance(value, dict):
            return f"{action['path']}:{'/'.join(action['keys'])}"
    elif kind == "toml_value" and path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
        _, value = TomlConfigManager.find_key(lines, action["section"],
                                              action["key"])
        if value is not None and value != action["value"]:
            return f"{action['path']}:[{action['section']}]/{action['key']}"
    elif kind == "markdown_block" and path.is_file():
        try:
            body = MarkdownBlockManager.extract(
                path.read_text(encoding="utf-8"), action["marker"])
        except TpuAgentSuiteError:
            return f"{action['path']}:{action['marker']}"
        if body is not None and digest_text(body) != action["sha256"]:
            return f"{action['path']}:{action['marker']}"

    return None


def detect_owned_modifications(destination: Path,
                               ownership: dict[str, Any]) -> list[str]:
    """Detect any files or configuration entries modified since installation."""
    modified: list[str] = []
    for action in ownership.get("actions", []):
        mod = check_action_modification(destination, action)
        if mod:
            modified.append(mod)
    return modified


def rollback_json_action(
    path: Path,
    keys: Sequence[str],
    expected_val: Any,
    is_container: bool = False,
) -> None:
    """Roll back an added JSON key or container from a config document."""
    if not path.is_file():
        return
    document = jsonio.load_json(path)
    if not JsonConfigManager.remove_added(document, keys, expected_val,
                                          is_container):
        return
    if document:
        jsonio.write_json(path, document)
    else:
        path.unlink(missing_ok=True)


def rollback_json_records(
    document: dict[str, Any],
    ownership: dict[str, Any] | None,
    config_path: str,
) -> None:
    """Undo, in `document`, every JSON record `ownership` holds for `config_path`.

    The in-memory half of what `remove_owned` does to that file on disk, in
    the same reverse order and under the same only-if-unchanged rule.
    """
    for action in reversed((ownership or {}).get("actions", [])):
        if action.get("path") != config_path:
            continue
        if action.get("kind") == "json_value":
            JsonConfigManager.remove_added(document, action["keys"],
                                           action["value"])
        elif action.get("kind") == "json_container":
            JsonConfigManager.remove_added(document,
                                           action["keys"], {},
                                           is_container=True)


def rollback_toml_action(
    path: Path,
    section: str,
    key: str,
    expected_val: Any,
    section_created: bool = False,
) -> None:
    """Roll back an added TOML key and optional section from a config file."""
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    index, value = TomlConfigManager.find_key(lines, section, key)
    if index is not None and value == expected_val:
        del lines[index]
        if section_created:
            section_index = next(
                (i for i, line in enumerate(lines)
                 if line.strip() == f"[{section}]"),
                None,
            )
            if section_index is not None:
                next_section = next(
                    (i for i in range(section_index + 1, len(lines))
                     if re.match(r"^\s*\[[^]]+]\s*$", lines[i])),
                    len(lines),
                )
                if not any(line.strip() and not line.lstrip().startswith("#")
                           for line in lines[section_index + 1:next_section]):
                    del lines[section_index:next_section]

        content = "\n".join(lines).strip()
        if content:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            path.unlink(missing_ok=True)


def rollback_action(destination: Path, action: dict[str, Any]) -> None:
    """Revert a single recorded installation action if unchanged."""
    path = rooted(destination, action["path"], "owned path")
    kind = action["kind"]

    if kind == "file":
        if path.is_file() and compute_digest(path) == action["sha256"]:
            path.unlink()
    elif kind == "json_value":
        rollback_json_action(path,
                             action["keys"],
                             action["value"],
                             is_container=False)
    elif kind == "json_container":
        rollback_json_action(path, action["keys"], {}, is_container=True)
    elif kind == "toml_value":
        rollback_toml_action(
            path,
            action["section"],
            action["key"],
            action["value"],
            action.get("section_created", False),
        )
    elif kind == "markdown_block":
        MarkdownBlockManager.strip(
            path,
            action["marker"],
            action.get("sha256", ""),
            action.get("file_created", False),
        )


# Central-store reference counting. The central store holds ONE immutable
# copy of each name@version (~/.tpu-agent-suite/packages/<name>@<version>/) shared
# by every workspace that installs it. To know when that copy may be
# deleted, each version directory carries a .refs.json ledger listing the
# absolute paths of the workspaces using it: install appends the workspace
# (register_central_ref), uninstall removes it (release_central_ref), and
# removing the last reference garbage-collects the whole version directory.
# A listed workspace whose own ownership ledger has vanished (deleted
# without uninstalling) or records a different version (moved on without
# this ledger being updated) is stale; such entries are pruned on every
# release so they cannot pin the payload forever. Upgrades release the
# previous version's reference explicitly, since the replaced ledger is
# the only record of which central version the workspace was using.
CENTRAL_REFS_FILE = ".refs.json"


def register_central_ref(central_pkg_dir: Path, workspace: Path) -> None:
    """Record a workspace as a user of a central package version.

    This is the install half of the central store's reference counting: one
    immutable copy of each name@version lives in ~/.tpu-agent-suite/packages/,
    shared by every workspace that installs it, and each version directory
    carries a .refs.json ledger listing the absolute paths of the workspaces
    using it. Registering appends the installing workspace so the payload
    stays pinned while anything still needs it.

    The ledger lives beside the payload it protects (.refs.json inside the
    version directory), so a version directory and its reference list can
    never drift apart: neither can exist without the other.
    """
    central_pkg_dir.mkdir(parents=True, exist_ok=True)
    refs_path = central_pkg_dir / CENTRAL_REFS_FILE
    refs = jsonio.load_json(refs_path) if refs_path.is_file() else {
        "workspaces": []
    }
    workspaces = set(refs.get("workspaces", []))
    workspaces.add(str(workspace.resolve()))
    jsonio.write_json(refs_path, {"workspaces": sorted(workspaces)})


def release_central_ref(package: str, version: str, workspace: Path) -> None:
    """Drop a workspace's reference to a central package version; GC when unreferenced.

    The uninstall half of the central store's reference counting: removes
    the workspace's entry from the version directory's .refs.json, and when
    the last reference goes it deletes the whole central version directory
    -- that is the garbage collection. Upgrades call this too, releasing
    the previous version's claim, because the replaced ownership ledger is
    the only record of which central version the workspace was using.

    Stale entries are pruned on every release so they cannot pin the
    payload forever: a recorded workspace whose ownership ledger no longer
    exists was deleted without uninstalling, and one whose ledger records a
    different version has moved on. An unreadable ledger keeps its ref --
    better a leaked payload than a blind GC under a live workspace.
    """
    central_pkg_dir = paths.get_package_storage_root(package, version)
    refs_path = central_pkg_dir / CENTRAL_REFS_FILE
    if not central_pkg_dir.is_dir():
        return

    released = str(workspace.resolve())
    refs = jsonio.load_json(refs_path) if refs_path.is_file() else {
        "workspaces": []
    }
    remaining = []
    for ws in refs.get("workspaces", []):
        if ws == released:
            continue
        ledger = paths.tpu_agent_suite_dir_for(ws) / paths.OWNERSHIP_FILE
        if not ledger.is_file():
            continue  # workspace deleted without uninstalling
        try:
            owner = jsonio.load_json(ledger)
            if (owner.get("package") != package
                    or owner.get("version", "") != version):
                continue  # workspace moved to another package or version
        except (OSError, ValueError):
            pass  # unreadable ledger: keep the ref, don't GC blindly
        remaining.append(ws)
    if remaining:
        jsonio.write_json(refs_path, {"workspaces": remaining})
    else:
        shutil.rmtree(central_pkg_dir, ignore_errors=True)
        packages_root = central_pkg_dir.parent
        if packages_root.is_dir() and not any(packages_root.iterdir()):
            packages_root.rmdir()


# The ownership ledger has a stable header (ledger_schema, package and
# version) that any installer may read: it says what is installed and which
# version's installer wrote it. The rest (scope, central_package, actions)
# is read only by an installer that knows ledger_schema, so none rolls back
# a ledger it would misread. Bump LEDGER_SCHEMA when the rest changes in a
# way an older installer would misread.
LEDGER_SCHEMA = 1


def owner_installer(ownership: dict[str, Any]) -> Path:
    """The installer.py of the package version that wrote this ledger."""
    return paths.get_package_storage_root(
        str(ownership.get("package") or ""), str(
            ownership.get("version") or "")) / "framework" / "installer.py"


def check_ledger_schema(ownership: dict[str, Any], scope: str,
                        destination: Path) -> None:
    """Refuse a ledger this installer can't read; name the one that can."""
    if ownership.get("ledger_schema") == LEDGER_SCHEMA:
        return
    where = ("--scope user"
             if scope == "user" else f"--destination {destination}")
    raise TpuAgentSuiteError(
        f"this installer can't read the install record of "
        f"{ownership.get('package')} {ownership.get('version')} (ledger "
        f"schema {ownership.get('ledger_schema')}; it reads "
        f"{LEDGER_SCHEMA}). Uninstall it with its own installer first: "
        f"python3 {owner_installer(ownership)} uninstall {where}")


def owned_paths(ownership: dict[str, Any]) -> list[str]:
    """The destination-relative paths an ownership record's actions touch."""
    return [action["path"] for action in ownership.get("actions", [])]


def remove_owned(destination: Path,
                 ownership: dict[str, Any],
                 prune: bool = True) -> list[str]:
    """Remove files and rollback config settings tracked in ownership record.

    `prune=False` leaves the emptied directories in place, for an upgrade
    that is about to repopulate them: removing and recreating `.claude/`
    gives it a new inode, and anything holding the old one -- a container
    bind mount, an editor, a shell's cwd -- goes on seeing an empty directory.
    """
    modified = detect_owned_modifications(destination, ownership)

    for action in reversed(ownership.get("actions", [])):
        rollback_action(destination, action)

    if prune:
        prune_empty_parents(destination, owned_paths(ownership))
    return modified


# =====================================================================
# Install & Uninstall Operations
# =====================================================================


def warn_if_workspace_not_discoverable(
    destination: Path,
    copies: Sequence[tuple[Path, Path]],
) -> None:
    """Warn when agents land in a workspace no harness will scan.

    Harnesses that discover workspace customizations walk up from the working
    directory to a repository root and stop there. Without one, the `.agents/`
    tree is never reached, and the agents sit installed and invisible: the
    orchestrator is told to delegate, the call reports the subagent as not
    found, and running the instructions inline is the only path left.

    Warning rather than failing, because a destination can legitimately be
    made a repository afterwards, and because not every harness resolves
    customizations this way.
    """
    installs_agents = any(source.parent.name == "agents"
                          for source, _ in copies)
    if not installs_agents:
        return
    if paths.find_git_root(destination) is not None:
        return
    print(
        f"WARNING: {destination} is not inside a git repository. Harnesses"
        " that discover workspace agents stop walking at a repository root,"
        " so the installed agents may be invisible and delegation to them"
        " will report the subagent as not found.",
        file=sys.stderr,
    )


def verify_agents_reachable(
    distribution: Path,
    copies: Sequence[tuple[Path, Path]],
) -> None:
    """Fail when a distribution ships agents that the install plan never places.

    An install that copies skills but silently drops agents still reports
    success, and the resulting failure surfaces much later and in disguise: the
    orchestrator is told to delegate, finds no agent registered, and falls back
    to reading an agent definition off disk and running it inline. That looks
    like a model mistake in the transcript. It is an install defect, and this
    is where it is cheap to catch.
    """
    shipped = {
        p.stem
        for agents_dir in distribution.rglob("agents") if agents_dir.is_dir()
        for p in agents_dir.iterdir() if p.is_file()
    }
    if not shipped:
        return

    # Compare by agent name, not by path. A distribution may carry the same
    # agent in more than one tree -- Claude Code ships a marketplace copy and
    # installs the one under `project/` -- and installing either satisfies the
    # requirement that something can be delegated to.
    installed = {
        source.stem
        for source, _ in copies if source.parent.name == "agents"
    }
    unreachable = sorted(shipped - installed)
    if unreachable:
        raise TpuAgentSuiteError(
            "install plan ships agents it never installs, so nothing can"
            " delegate to them: " + ", ".join(unreachable))


def _confirmed(prompt: str) -> bool:
    """Ask a y/N question on the terminal; only an explicit yes is a yes.

    No answer at all (EOF, Ctrl-C) counts as no, so a closed stdin never
    approves a destructive step.
    """
    try:
        choice = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return choice in ("y", "yes")


def validate_install_preconditions(
    destination: Path,
    previous_ownership: dict[str, Any] | None,
    upgrade: bool = True,
    clean: bool = False,
    force: bool = False,
    non_interactive: bool = False,
) -> None:
    """Validate destination state and enforce upgrade/collision policies before installation.

    Args:
        destination: Target installation root directory.
        previous_ownership: Recorded ownership ledger from an existing install, or None for a fresh install.
        upgrade: If True (default), permits updating an existing installation. If False (via --no-upgrade), raises TpuAgentSuiteError if already installed.
        clean: If True (via --clean), a fresh reinstallation was requested, skipping modification checks against previous installations.
        force: If True (via --force), automatically overwrites locally modified managed files without interactive prompts.
        non_interactive: If True (via --non-interactive), disables terminal prompts and raises TpuAgentSuiteError if modified files or conflicts exist without --force.
    """
    if destination == Path(destination.anchor):
        raise TpuAgentSuiteError("refusing to install into a filesystem root")

    if not upgrade and previous_ownership and not clean:
        raise TpuAgentSuiteError(
            "package is already installed (use --upgrade to update or --clean to reinstall)"
        )

    if clean:
        return

    # Check for local modifications in previously managed files
    if previous_ownership:
        modified = detect_owned_modifications(destination, previous_ownership)
        if modified and not force:
            if non_interactive:
                raise TpuAgentSuiteError(
                    "refusing upgrade because managed content changed (use --force to overwrite):\n"
                    + "\n".join(modified))
            # Interactive prompt
            print("Managed files have local modifications:\n  " +
                  "\n  ".join(modified),
                  file=sys.stderr)
            if not _confirmed("Overwrite your local changes? [y/N]: "):
                raise TpuAgentSuiteError(
                    "aborted by user: refusing to overwrite modified managed files:\n"
                    + "\n".join(modified))


def check_file_collisions(
    copies: Sequence[tuple[Path, Path]],
    destination: Path,
    previous_ownership: dict[str, Any] | None,
    force: bool = False,
    non_interactive: bool = False,
) -> None:
    """Ensure newly copied files do not overwrite unmanaged user files unless confirmed or forced.

    Args:
        copies: List of (source_path, target_path) copy pairs.
        destination: Root installation target directory.
        previous_ownership: Ownership metadata from prior installation.
        force: When True, overwrites unmanaged collisions without prompting.
        non_interactive: When True, raises TpuAgentSuiteError on collisions without --force.
    """
    if force:
        return

    old_files = {
        action["path"]
        for action in (previous_ownership or {}).get("actions", [])
        if action["kind"] == "file"
    }
    collisions = [
        str(target.relative_to(destination)) for _, target in copies
        if target.exists()
        and str(target.relative_to(destination)) not in old_files
    ]
    if collisions:
        if non_interactive:
            raise TpuAgentSuiteError(
                "refusing to overwrite existing files (use --force to overwrite):\n"
                + "\n".join(collisions))
        # Interactive prompt
        print(
            "Found existing unmanaged files colliding with installation:\n  " +
            "\n  ".join(collisions),
            file=sys.stderr)
        if not _confirmed("Overwrite these files? [y/N]: "):
            raise TpuAgentSuiteError(
                "aborted by user: refusing to overwrite existing files:\n" +
                "\n".join(collisions))


def prevalidate_workspace_memory(
    destination: Path,
    merges: Sequence[dict[str, str]],
    previous_ownership: dict[str, Any] | None,
    force: bool = False,
) -> None:
    """Verify managed memory blocks can be written without discarding local edits."""
    owned = {
        (action["path"], action["marker"]): action.get("sha256", "")
        for action in (previous_ownership or {}).get("actions", [])
        if action["kind"] == "markdown_block"
    }

    for merge in merges:
        path = rooted(destination, merge["path"], "workspace memory path")
        if not path.is_file():
            continue

        # Raises on unbalanced or duplicated markers.
        body = MarkdownBlockManager.extract(path.read_text(encoding="utf-8"),
                                            merge["marker"])
        if body is None or force:
            continue

        recorded = owned.get((merge["path"], merge["marker"]))
        if recorded is None:
            raise TpuAgentSuiteError(
                f"unmanaged block '{merge['marker']}' already present in"
                f" {merge['path']}. Remove it, or pass --force to overwrite.")
        if digest_text(body) != recorded:
            raise TpuAgentSuiteError(
                f"managed block '{merge['marker']}' in {merge['path']} was"
                " edited locally. Revert the edit, pass --force to overwrite,"
                " or install with --no-workspace-memory.")


def resolve_config_merge_path(
    harness: str,
    declared_path: str,
) -> str:
    """Resolve JSON configuration merge destination path based on harness.

    Both 'project' and 'user' scopes route settings to standard settings (*.json).
    """
    path_obj = Path(declared_path)
    filename = path_obj.name

    if harness in ("claude-code",
                   "cursor") or harnesses.is_google_harness(harness):
        if filename in ("settings.local.json", "settings.json"):
            return str(path_obj.with_name("settings.json"))

    return declared_path


def prevalidate_configuration_mutations(
    destination: Path,
    plan: dict[str, Any],
    previous_ownership: dict[str, Any] | None = None,
) -> None:
    """Verify that JSON merges and TOML settings can be applied without conflict.

    Nothing is written: each JSON document is freshly loaded, and on an
    upgrade `previous_ownership`'s records for it are undone in memory first
    (`rollback_json_records`). The previous version's entries carry that
    version in their values (a hook command names `<package>@<version>`), so
    checking against the live file would refuse every upgrade as a conflict
    with a setting that is about to be removed. Only entries that still
    match the ledger are discounted; `force` is deliberately not consulted
    here, so an entry the user edited is the user's setting and conflicts.
    """
    harness = plan.get("harness", "")
    # Each merge is checked against the on-disk file on its own, whereas the
    # install writes between merges. Equivalent while a plan holds at most
    # one merge per resolved path, which the compiler guarantees today.
    for merge in plan.get("json_merges", []):
        merge_path = resolve_config_merge_path(harness, merge["path"])
        path = rooted(destination, merge_path, "JSON config path")
        document = jsonio.load_json(path) if path.exists() else {}
        rollback_json_records(document, previous_ownership, merge_path)
        JsonConfigManager.merge(document, merge["value"], [], [])

    for setting in plan.get("toml_sets", []):
        path = rooted(destination, setting["path"], "TOML config path")
        lines = (path.read_text(
            encoding="utf-8").splitlines() if path.exists() else [])
        _, existing = TomlConfigManager.find_key(lines, setting["section"],
                                                 setting["key"])
        if existing is not None and existing != TomlConfigManager.render_value(
                setting["value"]):
            raise TpuAgentSuiteError(
                f"refusing to replace existing TOML setting: [{setting['section']}]"
                f" {setting['key']}")


# The software check, and the requirements files the compiler writes at the
# distribution root, which are what it checks against.
ENV_CHECK = Path("framework") / "compute" / "env_check.py"
HOST_REQUIREMENTS = utils.HOST_REQUIREMENTS


def _env_check(distribution: Path | None,
               requirements: str) -> tuple[Path, Path] | None:
    """The check script and a requirements file, if the build ships both."""
    if distribution is None:
        return None
    script, reqs = distribution / ENV_CHECK, distribution / requirements
    return (script, reqs) if script.is_file() and reqs.is_file() else None


def check_host(distribution: Path) -> None:
    """Run the software check on this machine; report, never block.

    It runs under `python3` from PATH because that is what `{{runtime}}` runs
    under, rather than under the interpreter running this installer.
    """
    check = _env_check(distribution, HOST_REQUIREMENTS)
    if check is None:
        return
    python = shutil.which("python3")
    if python is None:
        print("WARNING: python3 is not on PATH; the agents' runtime needs it.",
              file=sys.stderr)
        return
    print("\nChecking the software on this machine...\n")
    script, requirements = check
    try:
        proc = subprocess.run([
            python,
            str(script), "--host", "--requirements",
            str(requirements)
        ],
                              capture_output=True,
                              text=True,
                              timeout=60,
                              check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"WARNING: could not check this machine: {exc}", file=sys.stderr)
        return
    print((proc.stdout or proc.stderr).rstrip() + "\n")
    if proc.returncode != 0:
        print(
            "WARNING: some agents need the software marked FAIL above and "
            "will not work until it is installed; the install goes on "
            "without it.\n",
            file=sys.stderr)


def resolve_install_destination(
    distribution: Path,
    scope: str | None,
    destination: Path | None,
) -> tuple[Path, str]:
    """Resolve effective destination directory and scope.

    Supports two installation scopes:
    - 'project': Installs into an explicit destination workspace directory (--destination required).
    - 'user': Installs globally into the user's home directory (--destination prohibited).
    """
    effective_scope = scope or "project"
    if effective_scope == "user":
        if destination is not None:
            raise TpuAgentSuiteError(
                "Cannot specify --destination when --scope is 'user'. "
                "User scope installs globally into the user home directory.")
        return Path.home().resolve(), "user"

    if effective_scope == "project":
        if destination is None:
            raise TpuAgentSuiteError(
                "--destination is required when scope is 'project'. "
                "Specify the target workspace directory with --destination /path/to/workspace."
            )
        dest = destination.resolve()
        if dest == distribution.resolve():
            raise TpuAgentSuiteError(
                "Cannot install into the distribution directory itself with scope 'project'. "
                "Specify a target workspace directory with --destination /path/to/workspace."
            )
        return dest, "project"

    raise TpuAgentSuiteError(
        f"unknown scope '{effective_scope}'. Must be 'project' or 'user'.")


def resolve_user_codex_home(home: Path) -> str:
    """Return the Codex home a user-scope install writes, relative to `home`.

    Codex reads its config, agents, hooks and user memory from $CODEX_HOME,
    ~/.codex when unset, so writing ~/.codex under a relocated home leaves all
    of them unread. Only a home inside $HOME is honoured: the ownership ledger
    records destination-relative paths, and `rooted` refuses any that escape.
    One outside it is refused rather than worked around: an install into
    ~/.codex instead would report success and be invisible to Codex. $HOME
    itself is refused too, or the install would scatter Codex's files across
    the top of the home directory.
    """
    configured = os.environ.get("CODEX_HOME")
    if not configured:
        return ".codex"
    target = Path(configured).expanduser().resolve()
    if home not in target.parents:
        raise TpuAgentSuiteError(
            f"CODEX_HOME={configured} is not a directory under {home}, where"
            " a user-scope install writes, and Codex reads nothing installed"
            " anywhere else. Unset CODEX_HOME, or point it at a directory"
            f" under {home}, and install again.")
    return target.relative_to(home).as_posix()


def resolve_uninstall_destination(
    scope: str | None,
    destination: Path | None,
) -> Path:
    """Resolve effective uninstall destination directory.

    Supports two installation scopes:
    - 'project': Uninstalls from an explicit workspace directory (--destination required).
    - 'user': Uninstalls globally from the user's home directory (--destination prohibited).
    """
    effective_scope = scope or "project"
    if effective_scope == "user":
        if destination is not None:
            raise TpuAgentSuiteError(
                "Cannot specify --destination when --scope is 'user'. "
                "User scope uninstalls globally from the user home directory.")
        return Path.home().resolve()

    if effective_scope == "project":
        if destination is None:
            raise TpuAgentSuiteError(
                "--destination is required when uninstalling with scope 'project'. "
                "Specify the target workspace directory with --destination /path/to/workspace."
            )
        return destination.resolve()

    raise TpuAgentSuiteError(
        f"unknown scope '{effective_scope}'. Must be 'project' or 'user'.")


def register_git_exclude(destination: Path, entry: str) -> None:
    """Register path entry in .git/info/exclude if within a git repository."""
    git_root = paths.find_git_root(destination)
    if git_root is None:
        return

    git_dir = git_root / ".git"
    exclude_file = git_dir / "info" / "exclude" if git_dir.is_dir() else None
    if exclude_file:
        exclude_file.parent.mkdir(parents=True, exist_ok=True)
        existing = (exclude_file.read_text(
            encoding="utf-8").splitlines() if exclude_file.exists() else [])
        if entry not in existing:
            existing.append(entry)
            exclude_file.write_text("\n".join(existing) + "\n",
                                    encoding="utf-8")


def summarize_installation(
    destination: Path,
    build: dict[str, Any],
    scope: str,
    memory_merges: Sequence[dict[str, str]] = (),
    default_profile: str | None = None,
) -> None:
    """Report what was installed, how compute gets set up, and how to load it."""
    harness = build["harness"]
    display = harnesses.HARNESS_DISPLAY_NAMES.get(harness, harness)

    print(f"Installed {build['source']} {build['version']} for"
          f" {display} ({scope} scope) into {destination}.")
    for merge in memory_merges:
        print(f"  Added a managed block to {destination / merge['path']}.")

    # Claude Code and Codex installs pre-authorize ~/.tpu-agent-suite through their
    # harness config; harnesses without a config writer need a one-time grant.
    # Cursor is listed for when its package target becomes buildable -- its
    # install spec writes no config either.
    if harness == "cursor" or harnesses.is_google_harness(harness):
        print(
            f"\nNote: package assets and runs live in"
            f" {paths.get_tpu_agent_suite_home()}. When {display} first asks for"
            " access to that directory, grant it permanently so agents can"
            " read shipped assets without per-session prompts.")

    if default_profile:
        print(f"\nCompute uses the profile {default_profile!r}; sessions check"
              " it before TPU work.")
    else:
        print("\nCompute is not set up yet. The first session sets it up: the"
              " agent asks only what your backend needs, suggests the"
              " answers, and checks that the target works.")

    # A project-scope install is read only by sessions rooted in its
    # workspace. A user-scope install is read from the harness's user config
    # directory by every session, and its destination is $HOME: naming it
    # would send users to start the harness there, which is what user scope
    # exists to avoid.
    where = f" in {destination}" if scope == "project" else ""
    print(f"\nRestart {display}{where} to load the installed skills,"
          " agents, and instructions.\nA session that is already running keeps"
          " the configuration it started with.")


# Where each harness reads user-global memory, relative to $HOME, keyed by
# the project-scope file it replaces. A user-scope install's destination is
# $HOME, but no harness reads ~/CLAUDE.md or ~/AGENTS.md as user memory: they
# walk project memory from the repository root down to the cwd, so a block
# left in $HOME reaches only a session started in $HOME itself. Antigravity
# takes ~/.gemini/AGENTS.md rather than GEMINI.md so the block never lands in
# the file users edit by hand.
USER_SCOPE_MEMORY_PATHS: dict[str, dict[str, str]] = {
    "claude-code": {
        "CLAUDE.md": ".claude/CLAUDE.md"
    },
    "codex": {
        "AGENTS.md": ".codex/AGENTS.md"
    },
    **{
        harness: {
            "AGENTS.md": ".gemini/AGENTS.md"
        }
        for harness in harnesses.GOOGLE_HARNESSES
    },
}


def check_store_not_relocated() -> None:
    """Refuse a TPU_AGENT_SUITE_HOME that moves the store away from ~.

    Hook commands, permission grants and instructions are compiled with the
    ~/.tpu-agent-suite spelling, so a package placed anywhere else is
    unreachable from every hook. Only a caller that mounts the store back at
    ~/.tpu-agent-suite (--relocated-store) may move it.
    """
    default_home = paths.default_tpu_agent_suite_home()
    home = paths.get_tpu_agent_suite_home()
    if home != default_home:
        raise TpuAgentSuiteError(
            f"TPU_AGENT_SUITE_HOME={os.environ.get('TPU_AGENT_SUITE_HOME')!r}"
            f" puts the package store in {home}, but installed hooks,"
            f" permissions and instructions always use"
            f" {paths.TPU_AGENT_SUITE_HOME_SPELLING}, so every hook would fail."
            " TPU_AGENT_SUITE_HOME is for tests only: unset it and install"
            " again.")


def install(
    distribution: Path,
    destination: Path | None = None,
    scope: str | None = None,
    non_interactive: bool = False,
    upgrade: bool = True,
    clean: bool = False,
    purge_data: bool = False,
    force: bool = False,
    skip_harness_check: bool = False,
    workspace_memory: bool = True,
    announce: bool = False,
    host_check: bool = True,
) -> Path:
    """Install or upgrade a translated distribution package into a workspace.

    Sets up nothing about compute: an existing environment.json is kept as
    it is, and a missing one starts with no profiles, for the runtime's
    `compute setup` to fill in. `host_check` runs the software check on this
    machine.
    """
    build = jsonio.load_json(distribution / "tpu-agent-suite-build.json")
    plan = jsonio.load_json(distribution / "tpu-agent-suite-install.json")

    if (build["harness"] != plan["harness"]
            or build["version_range"] != plan["version_range"]):
        raise TpuAgentSuiteError("build metadata and install plan disagree")

    if not skip_harness_check:
        check_harness(plan)

    dest_path, resolved_scope = resolve_install_destination(
        distribution, scope, destination)
    # Resolved before anything is removed, so a CODEX_HOME the install
    # cannot serve leaves the previous install as it was.
    codex_home = (resolve_user_codex_home(dest_path) if
                  (resolved_scope == "user"
                   and build["harness"] == "codex") else ".codex")
    pkg_name = build["source"]
    pkg_version = build.get("version", "")
    pkg_tag = utils.format_package_tag(pkg_name, pkg_version)
    central_pkg_dir = paths.get_package_storage_root(pkg_name, pkg_version)
    # The ledger and environment.json go where uninstall and the runtime
    # look for them: the workspace's .tpu-agent-suite/, or the central store
    # for a user-scope install (and for a project install in $HOME).
    tpu_agent_suite_dir = paths.tpu_agent_suite_dir_for(dest_path)
    ownership_path = tpu_agent_suite_dir / paths.OWNERSHIP_FILE
    environment_path = tpu_agent_suite_dir / paths.ENVIRONMENT_FILE

    previous_ownership = (jsonio.load_json(ownership_path)
                          if ownership_path.exists() else None)
    # One package per workspace (and per user): the ledger has room for one,
    # and so does current_run_id, which two packages' sessions would fight
    # over. Upgrading or reinstalling the same package goes ahead.
    if previous_ownership and previous_ownership.get("package") != pkg_name:
        where = ("your user scope"
                 if resolved_scope == "user" else str(dest_path))
        raise TpuAgentSuiteError(
            f"{where} already has {previous_ownership.get('package')} "
            f"installed, and holds one package at a time. Uninstall it "
            f"first, then install {pkg_name}.")
    if previous_ownership:
        check_ledger_schema(previous_ownership, resolved_scope, dest_path)

    if purge_data:
        if any(p.is_dir()
               for p in (paths.get_runs_root(), paths.get_sessions_root())):
            if non_interactive and not force:
                raise TpuAgentSuiteError(
                    "refusing to purge run data in non-interactive mode without --force"
                )
            if not non_interactive:
                print(
                    "\nWARNING: --purge-data will PERMANENTLY DESTROY all past runs, "
                    "experiment artifacts and session records in "
                    f"{paths.RUNS_HOME_SPELLING}/ and"
                    f" {paths.TPU_AGENT_SUITE_HOME_SPELLING}/{paths.SESSIONS_DIRNAME}/.\n",
                    file=sys.stderr,
                )
                if not _confirmed(
                        "Are you sure you want to permanently delete all run data? [y/N]: "
                ):
                    raise TpuAgentSuiteError("purge-data cancelled by user")

    validate_install_preconditions(
        dest_path,
        previous_ownership,
        upgrade=upgrade,
        clean=clean,
        force=force,
        non_interactive=non_interactive,
    )
    if host_check:
        check_host(distribution)

    if purge_data:
        # Session records name runs, so they go with them.
        for central in (paths.get_runs_root(), paths.get_sessions_root()):
            if central.is_dir():
                shutil.rmtree(central, ignore_errors=True)

    if clean and previous_ownership:
        # Rollback prior installation and harness configs. environment.json,
        # its verify results and the active run stay where they are: the
        # profiles are kept, and so are the results that go with them.
        remove_owned(dest_path, previous_ownership)
        ownership_path.unlink(missing_ok=True)
        previous_ownership = None

    # environment.json belongs to the runtime (`compute setup`, `config`), so
    # it is not in the ledger: only a missing one is written, an existing
    # one is never touched (not by --force or --clean either), and uninstall
    # removes it with the ledger.
    if not environment_path.is_file():
        environment_path.parent.mkdir(parents=True, exist_ok=True)
        jsonio.write_json(environment_path, {
            "revision": 1,
            "default": None,
            "profiles": {}
        })

    if resolved_scope == "user":
        harness = str(build.get("harness"))

        def relocate_codex(path: str) -> str:
            if harness == "codex" and path.split("/", 1)[0] == ".codex":
                return codex_home + path[len(".codex"):]
            return path

        for entry in plan.get("copies", []):
            if entry.get("store") == "workspace":
                dest = entry.get("destination", "")
                if harnesses.is_google_harness(harness) and dest.startswith(
                        ".agents"):
                    dest = dest.replace(".agents", ".gemini/config", 1)
                entry["destination"] = relocate_codex(dest)
        for setting in plan.get("toml_sets", []):
            setting["path"] = relocate_codex(setting["path"])
        memory_paths = USER_SCOPE_MEMORY_PATHS.get(harness, {})
        for merge in plan.get("markdown_merges", []):
            if merge.get("path") in memory_paths:
                merge["path"] = relocate_codex(memory_paths[merge["path"]])

    copies = expand_copies(distribution, dest_path, plan, central_pkg_dir)
    workspace_copies = [(s_, t) for s_, t, store in copies
                        if store == "workspace"]
    central_copies = [(s_, t) for s_, t, store in copies if store == "central"]

    # Agents may be placed by either store, so reachability checks see the
    # full copy plan; discoverability concerns only the workspace tree.
    verify_agents_reachable(distribution,
                            [(s_, t) for s_, t, _store in copies])
    # User-scope agents are discovered from the harness's user config
    # directory, never by walking up to a repository root, so $HOME needs no
    # repository for them to load.
    if resolved_scope == "project":
        warn_if_workspace_not_discoverable(dest_path, workspace_copies)

    check_file_collisions(
        workspace_copies,
        dest_path,
        previous_ownership,
        force=force,
        non_interactive=non_interactive,
    )
    prevalidate_configuration_mutations(dest_path,
                                        plan,
                                        previous_ownership=previous_ownership)

    # Memory blocks arrive fully rendered from the compiler (the token
    # registry resolves every resource at compile time), so they apply
    # verbatim -- no install-time path resolution.
    memory_merges = (plan.get("markdown_merges", [])
                     if workspace_memory else [])
    prevalidate_workspace_memory(dest_path,
                                 memory_merges,
                                 previous_ownership,
                                 force=force)

    # Rollback prior installation if upgrading
    if previous_ownership and not clean:
        # Pruned once the new files are in (below), so only what the new
        # version no longer uses goes, and directories it reuses are kept.
        remove_owned(dest_path, previous_ownership, prune=False)
        ownership_path.unlink(missing_ok=True)
        # The old ledger is the only record of which central version this
        # workspace was using; release that claim now or the old version
        # directory stays pinned forever after the upgrade. Same-version
        # reinstalls keep their ref: it is still correct and is
        # re-registered below.
        if previous_ownership.get("central_package") and (
                previous_ownership.get("version") != build.get("version", "")):
            release_central_ref(previous_ownership["package"],
                                previous_ownership.get("version", ""),
                                dest_path)

    # Perform installation
    dest_path.mkdir(parents=True, exist_ok=True)
    actions: list[dict[str, Any]] = []

    # Copy workspace files (tracked in ownership.json)
    for source, target in workspace_copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        actions.append({
            "kind": "file",
            "path": str(target.relative_to(dest_path)),
            "sha256": compute_digest(target),
        })

    # Copy payload to the central store. Not ownership actions: the central
    # copy is shared, so its lifecycle is governed by the reference ledger
    # (registered below), not by any single workspace's ledger.
    for source, target in central_copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    # Apply JSON merges
    harness = plan.get("harness", "")
    for merge in plan.get("json_merges", []):
        merge_path = resolve_config_merge_path(harness, merge["path"])
        path = rooted(dest_path, merge_path, "JSON config path")
        document = jsonio.load_json(path) if path.exists() else {}
        records: list[dict[str, Any]] = []
        JsonConfigManager.merge(document, merge["value"], [], records)
        jsonio.write_json(path, document)
        actions.extend({**record, "path": merge_path} for record in records)

    # Apply workspace memory blocks
    for merge in memory_merges:
        path = rooted(dest_path, merge["path"], "workspace memory path")
        record = MarkdownBlockManager.apply(path, merge["marker"],
                                            merge["content"])
        actions.append({**record, "path": merge["path"]})

    # Apply TOML sets
    for setting in plan.get("toml_sets", []):
        path = rooted(dest_path, setting["path"], "TOML config path")
        record = TomlConfigManager.apply_set(path, setting["section"],
                                             setting["key"], setting["value"])
        if record:
            actions.append({**record, "path": setting["path"]})

    # Register this workspace in the central store's reference ledger so
    # uninstall can garbage-collect the shared payload once no workspace
    # references this package version.
    register_central_ref(central_pkg_dir, dest_path)

    # Record ownership
    ownership_path.parent.mkdir(parents=True, exist_ok=True)
    jsonio.write_json(
        ownership_path,
        {
            "ledger_schema": LEDGER_SCHEMA,
            "package": build["source"],
            "version": build.get("version", TPU_AGENT_SUITE_VERSION),
            "harness": build["harness"],
            "compatibility_target": build["compatibility_target"],
            "scope": resolved_scope,
            "central_package": pkg_tag,
            "actions": actions,
        },
    )

    if previous_ownership and not clean:
        prune_empty_parents(dest_path, owned_paths(previous_ownership))

    if resolved_scope == "project":
        register_git_exclude(dest_path, ".tpu-agent-suite/")

    if announce:
        summarize_installation(
            dest_path, build, resolved_scope, memory_merges,
            jsonio.load_json_or_empty(environment_path).get("default"))
    return dest_path


def read_installed(dest_path: Path,
                   package: str = "") -> tuple[Path, dict[str, Any]]:
    """The ledger of what dest_path has installed, which must be `package` if given."""
    if package:
        validate_package_name(package)
    ownership_path = paths.tpu_agent_suite_dir_for(
        dest_path) / paths.OWNERSHIP_FILE
    if not ownership_path.is_file():
        raise TpuAgentSuiteError(f"nothing is installed in {dest_path}: "
                                 f"{ownership_path} does not exist")
    ownership = jsonio.load_json(ownership_path)
    if package and ownership.get("package") != package:
        raise TpuAgentSuiteError(
            f"{dest_path} has {ownership.get('package')} installed, not "
            f"{package}")
    return ownership_path, ownership


def uninstall(
    destination: Path | None = None,
    package: str = "",
    scope: str | None = None,
) -> list[str]:
    """Uninstall the package installed in destination, preserving user-modified files.

    The ownership ledger names the package and version, so `package` is
    only a check: given, it must match what the ledger records.
    """
    dest_path = resolve_uninstall_destination(scope, destination)
    ownership_path, ownership = read_installed(dest_path, package)
    check_ledger_schema(ownership,
                        str(ownership.get("scope") or scope or "project"),
                        dest_path)
    modified = remove_owned(dest_path, ownership)
    ownership_path.unlink()
    # The runtime's files, not in the ledger: environment.json (`compute
    # setup`, `config`) and its verify results. Left behind, an
    # environment.json without a ledger would serve any package.
    for runtime_file in (paths.ENVIRONMENT_FILE, paths.VERIFICATION_FILE):
        ownership_path.with_name(runtime_file).unlink(missing_ok=True)

    # Release this workspace's claim on the shared central payload; the
    # version directory is garbage-collected once nothing references it.
    if ownership.get("central_package"):
        release_central_ref(ownership["package"], ownership.get("version", ""),
                            dest_path)

    # Only the ledger's directory goes, and only once it is empty: the runs,
    # sessions and studies under it outlive the install, so nothing there is
    # swept, down to a run's empty artifacts/.
    try:
        ownership_path.parent.rmdir()
    except OSError:
        pass
    return modified


def build_installer_parser() -> argparse.ArgumentParser:
    """Construct parser for standalone installer commands."""
    parser = argparse.ArgumentParser(
        prog="installer",
        description=
        "Installation and uninstallation operations for TPU agent suite packages.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # install
    p_install = sub.add_parser(
        "install", help="Install translated package into workspace")
    # Positional distribution is optional; when omitted, installer auto-detects
    # if it is being executed from within an extracted distribution directory.
    p_install.add_argument(
        "distribution",
        nargs="?",
        default=None,
        type=Path,
        help=
        "Path to translated distribution directory (default: current distribution if run from within package)",
    )
    p_install.add_argument(
        "--scope",
        choices=["project", "user"],
        default="project",
        help=
        "Target installation scope: 'project' or 'user' (default: project)",
    )
    p_install.add_argument(
        "--destination",
        type=Path,
        help=
        "Target workspace destination directory (required when scope is 'project')",
    )
    p_install.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never ask before overwriting files or purging run data; fail "
        "instead",
    )
    p_install.add_argument(
        "--upgrade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upgrade existing installation (default: True)",
    )
    p_install.add_argument(
        "--clean",
        action="store_true",
        help=
        "Wipe existing installation and start fresh (preserves run history/artifacts)",
    )
    p_install.add_argument(
        "--purge-data",
        action="store_true",
        help=f"Purge centralized run history in {paths.RUNS_HOME_SPELLING}/",
    )
    p_install.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of locally modified or unmanaged colliding files",
    )
    p_install.add_argument(
        "--workspace-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add the package's managed block to the workspace memory file"
        " (default: True)",
    )
    p_install.add_argument("--skip-host-check",
                           action="store_true",
                           help="Do not check the software on this machine")
    p_install.add_argument("--skip-harness-check",
                           action="store_true",
                           help=argparse.SUPPRESS)
    # For callers that move the store on purpose and mount it back at
    # ~/.tpu-agent-suite before any hook runs: eval staging and tests.
    p_install.add_argument("--relocated-store",
                           action="store_true",
                           help=argparse.SUPPRESS)

    # uninstall
    p_uninstall = sub.add_parser("uninstall",
                                 help="Uninstall package from workspace")
    p_uninstall.add_argument(
        "--package",
        required=False,
        default=None,
        help=
        "The package expected to be installed; must match the ledger. Defaults to this installer's own package (none when run from source)",
    )
    p_uninstall.add_argument(
        "--scope",
        choices=["project", "user"],
        default="project",
        help=
        "Target installation scope: 'project' or 'user' (default: project)",
    )
    p_uninstall.add_argument(
        "--destination",
        type=Path,
        help=
        "Target workspace destination directory (required when scope is 'project')",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entrypoint for standalone installer operations."""
    parser = build_installer_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "install":
            # An interactive install asks before overwriting files unless
            # --force. Without a terminal, that question blocks forever on
            # an open pipe, so refuse before changing anything.
            if not (args.non_interactive or args.force or sys.stdin.isatty()):
                raise TpuAgentSuiteError(
                    "stdin is not a terminal, so the installer cannot ask"
                    " before overwriting files. Run headless with"
                    " --non-interactive (see docs/install.md, 'Headless"
                    " installs').")
            distribution = args.distribution
            if distribution is None:
                # This file is framework/installer.py; a distribution holds
                # the plan beside framework/. The central store has a build
                # file too, but no plan.
                package_dir = Path(__file__).resolve().parents[1]
                if (package_dir / "tpu-agent-suite-install.json").is_file():
                    distribution = package_dir
                else:
                    raise TpuAgentSuiteError(
                        "Path to translated distribution directory is required when not running from within a distribution package."
                    )
            if not args.relocated_store:
                check_store_not_relocated()
            install(
                distribution=distribution,
                destination=args.destination,
                scope=args.scope,
                non_interactive=args.non_interactive,
                upgrade=args.upgrade,
                clean=args.clean,
                purge_data=args.purge_data,
                force=args.force,
                skip_harness_check=args.skip_harness_check,
                host_check=not args.skip_host_check,
                workspace_memory=args.workspace_memory,
                announce=True,
            )

        elif args.command == "uninstall":
            # A package's installer (in its distribution or the central
            # store) uninstalls only that package; one run from source
            # takes whatever the ledger names. --package overrides either.
            own_build = (Path(__file__).resolve().parents[1] /
                         "tpu-agent-suite-build.json")
            package = args.package or (jsonio.load_json(own_build)["source"]
                                       if own_build.is_file() else "")
            _, installed = read_installed(
                resolve_uninstall_destination(args.scope, args.destination),
                package)
            # The installed version's own installer, when it is another
            # file, uninstalls: it wrote the ledger, so it reads it right.
            owner = owner_installer(installed)
            if (owner.is_file()
                    and owner.resolve() != Path(__file__).resolve()):
                forwarded = sys.argv[1:] if argv is None else list(argv)
                return subprocess.run([sys.executable,
                                       str(owner), *forwarded]).returncode
            modified = uninstall(
                destination=args.destination,
                package=package,
                scope=args.scope,
            )
            if modified:
                print(
                    "Preserved modified managed content:\n" +
                    "\n".join(modified),
                    file=sys.stderr,
                )

        return 0
    except TpuAgentSuiteError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
