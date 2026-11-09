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
"""Applies private build-time overlays onto a temporary copy of `package/`.

An overlay lets an internal repository add private skills, references, scripts,
or schema fields to a built distribution while keeping `package/` itself clean
and ready to publish as open source.

When `<git_root>/google/overlays/package/` exists,
`stage_package_with_overlays()` copies `package/` to a temporary directory,
applies every file in the overlay directory onto that copy, and hands the copy
to the compiler (`compose`, `verify`, `lint`, and `translate`). `package/` on
disk is never modified. In a public checkout where `google/overlays/` does not
exist -- or when `--no-overlays` is passed -- this step does nothing and hands
`package/` directly to the compiler without copying anything.

Each file under `google/overlays/package/` maps to the same relative path under
`package/` using one of four conventions:

- `<path>.patch.json`: merges JSON into existing file `<path>`. Dictionaries
  merge key-by-key, lists of `{"name": ...}` objects merge by `"name"`, plain
  lists append new items, and `{"$replace": value}` replaces a value outright.
- `<path>.edits.json`: replaces exact `"target"` strings with `"replacement"`
  strings in existing text file `<path>` (raises `TpuAgentSuiteError` if a
  target string is not found, so stale edits fail loudly).
- `<path>.append.md`: appends markdown text to the end of existing file `<path>`.
- Any other `<path>`: copies the file directly into `<path>` in the temporary
  package copy (for example adding a private skill directory).
"""

from __future__ import annotations

import contextlib
import copy
import pathlib
import shutil
import tempfile
from typing import Any, Iterator

from framework import jsonio, paths, utils

Path = pathlib.Path

DEFAULT_OVERLAYS_ROOT = Path("google") / "overlays"
PATCH_JSON_SUFFIX = ".patch.json"
EDITS_JSON_SUFFIX = ".edits.json"
APPEND_MD_SUFFIX = ".append.md"


def default_overlay_dir(package_dir: Path) -> Path | None:
    """Return `<git_root>/google/overlays/<package_rel>` if it exists."""
    resolved_pkg = package_dir.resolve()
    git_root = paths.find_git_root(resolved_pkg)
    if git_root is not None:
        try:
            pkg_rel = resolved_pkg.relative_to(git_root)
        except ValueError:
            return None
        candidate = git_root / DEFAULT_OVERLAYS_ROOT / pkg_rel
    else:
        candidate = (resolved_pkg.parent / DEFAULT_OVERLAYS_ROOT /
                     resolved_pkg.name)
    return candidate if candidate.is_dir() else None


def _is_named_object_list(items: list[Any]) -> bool:
    return bool(items) and all(
        isinstance(item, dict) and isinstance(item.get("name"), str)
        for item in items)


def deep_merge_json(base: Any, patch: Any) -> Any:
    """Recursively merge `patch` onto `base` without mutating inputs."""
    if isinstance(patch, dict) and set(patch.keys()) == {"$replace"}:
        return copy.deepcopy(patch["$replace"])

    if isinstance(base, dict) and isinstance(patch, dict):
        merged = copy.deepcopy(base)
        for key, patch_val in patch.items():
            if key in merged:
                merged[key] = deep_merge_json(merged[key], patch_val)
            else:
                merged[key] = copy.deepcopy(patch_val)
        return merged

    if isinstance(base, list) and isinstance(patch, list):
        combined = base + patch
        if _is_named_object_list(combined):
            merged_list = [copy.deepcopy(item) for item in base]
            index_by_name = {
                item["name"]: idx
                for idx, item in enumerate(merged_list)
            }
            for patch_item in patch:
                name = patch_item["name"]
                if name in index_by_name:
                    idx = index_by_name[name]
                    merged_list[idx] = deep_merge_json(merged_list[idx],
                                                       patch_item)
                else:
                    index_by_name[name] = len(merged_list)
                    merged_list.append(copy.deepcopy(patch_item))
            return merged_list

        merged_primitives = [copy.deepcopy(item) for item in base]
        for item in patch:
            if item not in merged_primitives:
                merged_primitives.append(copy.deepcopy(item))
        return merged_primitives

    return copy.deepcopy(patch)


def _apply_edits_json(dest: Path, overlay_file: Path, rel_label: str) -> None:
    if not dest.is_file():
        raise utils.TpuAgentSuiteError(
            f"Overlay '{rel_label}' targets missing file '{dest.name}'")
    raw = jsonio.load_json(overlay_file)
    edits = raw.get("edits") if isinstance(raw, dict) else raw
    if not isinstance(edits, list) or not edits:
        raise utils.TpuAgentSuiteError(
            f"Overlay '{rel_label}' must contain a non-empty list of edits")

    content = dest.read_text(encoding="utf-8")
    for idx, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise utils.TpuAgentSuiteError(
                f"Overlay '{rel_label}' edit #{idx} must be an object")
        target = edit.get("target")
        replacement = edit.get("replacement")
        if not isinstance(target, str) or not target:
            raise utils.TpuAgentSuiteError(
                f"Overlay '{rel_label}' edit #{idx} requires a non-empty "
                "'target' string")
        if not isinstance(replacement, str):
            raise utils.TpuAgentSuiteError(
                f"Overlay '{rel_label}' edit #{idx} requires a 'replacement' "
                "string")
        if target not in content:
            raise utils.TpuAgentSuiteError(
                f"Overlay '{rel_label}' edit #{idx} target snippet not found "
                f"in '{dest.name}'")
        content = content.replace(target, replacement)
    dest.write_text(content, encoding="utf-8")


def _iter_overlay_entries(overlay_root: Path) -> list[Path]:
    """Return sorted overlay files while rejecting symlinks anywhere inside."""
    entries: list[Path] = []
    if overlay_root.is_symlink():
        raise utils.TpuAgentSuiteError(
            f"Overlay root must not be a symlink: {overlay_root}")
    for candidate in sorted(overlay_root.rglob("*")):
        if candidate.is_symlink():
            rel = candidate.relative_to(overlay_root)
            raise utils.TpuAgentSuiteError(
                f"Symlinks are not permitted inside overlays: '{rel}'")
        if candidate.is_file():
            entries.append(candidate)
    return entries


def apply_overlays_in_place(staged_dir: Path, overlay_root: Path) -> int:
    """Apply all overlay operations from `overlay_root` onto `staged_dir`."""
    entries = _iter_overlay_entries(overlay_root)
    applied = 0
    for entry in entries:
        rel = entry.relative_to(overlay_root)
        rel_posix = rel.as_posix()

        if rel_posix.endswith(PATCH_JSON_SUFFIX):
            target_rel = rel_posix[:-len(PATCH_JSON_SUFFIX)]
            dest = utils.rooted(staged_dir, target_rel, "overlay target path")
            if not dest.is_file():
                raise utils.TpuAgentSuiteError(
                    f"Overlay '{rel_posix}' targets missing JSON file "
                    f"'{target_rel}'")
            base_doc = jsonio.load_json(dest)
            patch_doc = jsonio.load_json(entry)
            jsonio.write_json(dest, deep_merge_json(base_doc, patch_doc))
        elif rel_posix.endswith(EDITS_JSON_SUFFIX):
            target_rel = rel_posix[:-len(EDITS_JSON_SUFFIX)]
            dest = utils.rooted(staged_dir, target_rel, "overlay target path")
            _apply_edits_json(dest, entry, rel_posix)
        elif rel_posix.endswith(APPEND_MD_SUFFIX):
            target_rel = rel_posix[:-len(APPEND_MD_SUFFIX)]
            dest = utils.rooted(staged_dir, target_rel, "overlay target path")
            if not dest.is_file():
                raise utils.TpuAgentSuiteError(
                    f"Overlay '{rel_posix}' targets missing markdown file "
                    f"'{target_rel}'")
            existing = dest.read_text(encoding="utf-8")
            addition = entry.read_text(encoding="utf-8")
            sep = "" if (not existing or existing.endswith("\n")) else "\n"
            dest.write_text(f"{existing}{sep}{addition}", encoding="utf-8")
        else:
            dest = utils.rooted(staged_dir, rel_posix, "overlay target path")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, dest)
        applied += 1
    return applied


@contextlib.contextmanager
def without_external_drivers() -> Iterator[None]:
    """Temporarily hide external compute drivers so `--no-overlays` builds stay public."""
    from framework.compute.drivers import registry

    if not registry.EXTERNAL_DRIVERS:
        yield
        return

    saved_drivers = dict(registry.DRIVER_REGISTRY)
    saved_external = dict(registry.EXTERNAL_DRIVERS)
    saved_cli_flags = {
        k: list(v)
        for k, v in registry.DRIVER_CLI_FLAGS.items()
    }
    saved_req_flags = {
        k: set(v)
        for k, v in registry.DRIVER_REQUIRED_FLAGS.items()
    }
    saved_all_flags = set(registry.REGISTERED_DRIVER_CLI_FLAGS)
    try:
        for name in list(registry.EXTERNAL_DRIVERS):
            registry.DRIVER_REGISTRY.pop(name, None)
            registry.DRIVER_CLI_FLAGS.pop(name, None)
            registry.DRIVER_REQUIRED_FLAGS.pop(name, None)
        registry.EXTERNAL_DRIVERS.clear()
        registry.REGISTERED_DRIVER_CLI_FLAGS.clear()
        for flags in registry.DRIVER_CLI_FLAGS.values():
            registry.REGISTERED_DRIVER_CLI_FLAGS.update(flags)
        yield
    finally:
        registry.DRIVER_REGISTRY.clear()
        registry.DRIVER_REGISTRY.update(saved_drivers)
        registry.EXTERNAL_DRIVERS.clear()
        registry.EXTERNAL_DRIVERS.update(saved_external)
        registry.DRIVER_CLI_FLAGS.clear()
        registry.DRIVER_CLI_FLAGS.update(saved_cli_flags)
        registry.DRIVER_REQUIRED_FLAGS.clear()
        registry.DRIVER_REQUIRED_FLAGS.update(saved_req_flags)
        registry.REGISTERED_DRIVER_CLI_FLAGS.clear()
        registry.REGISTERED_DRIVER_CLI_FLAGS.update(saved_all_flags)


@contextlib.contextmanager
def stage_package_with_overlays(
    package_dir: Path,
    overlay_dir: Path | None = None,
    enabled: bool = True,
) -> Iterator[Path]:
    """Yield `package_dir` directly, or a temporary copy with overlays applied."""
    if not enabled:
        with without_external_drivers():
            yield package_dir
        return
    if not package_dir.is_dir():
        yield package_dir
        return

    resolved_overlay = (overlay_dir if overlay_dir is not None else
                        default_overlay_dir(package_dir))
    if resolved_overlay is None or not resolved_overlay.exists():
        yield package_dir
        return

    entries = _iter_overlay_entries(resolved_overlay)
    if not entries:
        yield package_dir
        return

    with tempfile.TemporaryDirectory(prefix="tpu-agent-suite-overlay-") as tmp:
        staged_pkg = Path(tmp) / package_dir.resolve().name
        shutil.copytree(package_dir, staged_pkg, symlinks=True)
        apply_overlays_in_place(staged_pkg, resolved_overlay)
        yield staged_pkg
