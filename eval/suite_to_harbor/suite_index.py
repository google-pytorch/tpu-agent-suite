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
"""Resolves evaluation suites by path or declared name."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_YAML_SUFFIXES = (".yaml", ".yml")
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


class SuiteLookupError(LookupError):
    """A suite could not be resolved (not found or ambiguous)."""


def _is_suite_file(path: Path) -> bool:
    return path.suffix.lower() in _YAML_SUFFIXES and path.is_file()


def _is_suite_target(path: Path) -> bool:
    return _is_suite_file(path) or (path.is_dir() and any(
        _is_suite_file(f) for f in path.iterdir()))


def _path_candidates(base: Path | None, spec: str) -> list[Path]:
    path = Path(spec) if base is None else base / spec
    # Skip suffix expansion for explicit directories or YAML paths.
    if (not path.name or spec in (".", "..") or spec.endswith(
        ("/", "/.", "/..")) or path.suffix.lower() in _YAML_SUFFIXES):
        return [path]
    return [
        path,
        *(path.parent / f"{path.name}{ext}" for ext in _YAML_SUFFIXES),
    ]


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_YAML_LOADER)
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _declared_name(data: dict[str, Any]) -> str | None:
    for key in ("suite_name", "name"):
        if (val := data.get(key)) is not None and (s := str(val).strip()):
            return s
    return None


def _scan_suites(suites_dir: Path) -> list[tuple[Path, str, str | None]]:
    """Returns sorted (path, rel_posix, declared_name) entries under suites_dir."""
    if not suites_dir.is_dir():
        return []
    parsed: list[tuple[Path, dict[str, Any], int]] = []
    cases_by_dir: dict[Path, list[int]] = {}
    for path in sorted(suites_dir.rglob("*")):
        if not _is_suite_file(path):
            continue
        data = _read_yaml_mapping(path)
        ev = data.get("evaluation")
        if isinstance(ev, dict) and isinstance(ev.get("cases"), list):
            n = len(ev["cases"])
            parsed.append((path, data, n))
            cases_by_dir.setdefault(path.parent, []).append(n)

    # A directory of >=2 single-case files is one suite known by its dir path.
    task_dirs = {
        d
        for d, counts in cases_by_dir.items()
        if d != suites_dir and len(counts) >= 2 and max(counts) == 1
    }
    entries = [(d, d.relative_to(suites_dir).as_posix(), None)
               for d in task_dirs]
    entries.extend((p, p.relative_to(suites_dir).with_suffix("").as_posix(),
                    _declared_name(data)) for p, data, _ in parsed
                   if p.parent not in task_dirs)
    return sorted(entries, key=lambda item: item[0])


def _names_from_entries(
        entries: list[tuple[Path, str, str | None]]) -> set[str]:
    declared = [name for _, _, name in entries if name]
    return {rel
            for _, rel, _ in entries
            } | {name
                 for name in declared if declared.count(name) == 1}


def suite_names(suites_dir: Path) -> set[str]:
    """Every suite's path under `suites_dir`, plus each unique declared name."""
    return _names_from_entries(_scan_suites(suites_dir))


def suite_sources(resolved: Path) -> list[Path]:
    """Returns the ordered suite YAML file(s) for a resolved suite path."""
    if _is_suite_file(resolved):
        return [resolved]
    if resolved.is_dir():
        sources = sorted(f for f in resolved.iterdir() if _is_suite_file(f))
        if sources:
            return sources
        raise ValueError(
            f"Suite directory {resolved} holds no .yaml/.yml files.")
    raise ValueError(f"Suite path is not a YAML file or directory: {resolved}")


def suite_dataset_dir(
    resolved: Path,
    suites_dir: Path,
    datasets_dir: Path,
) -> Path:
    """Computes the target dataset directory under `datasets_dir` for a suite."""
    resolved_p = resolved.resolve()
    suites_root = suites_dir.resolve()
    if resolved_p.is_relative_to(suites_root):
        sub_rel = resolved_p.relative_to(suites_root)
        if sub_rel == Path("."):
            raise ValueError(
                f"Suite path cannot be the suites root itself: {resolved}")
        return datasets_dir / (sub_rel if resolved_p.is_dir() else
                               sub_rel.with_suffix(""))

    name = (resolved_p.name if resolved_p.is_dir() else
            _declared_name(_read_yaml_mapping(resolved_p)) or resolved_p.stem)
    if (not name or Path(name).name != name or name in (".", "..")
            or "\\" in name):
        raise ValueError(
            f"Suite name {name!r} is not a valid single directory segment")
    return datasets_dir / name


def resolve_suite(
    spec: str,
    suites_dir: Path,
    repo_root: Path | None = None,
    *,
    allow_cwd: bool = True,
) -> Path:
    """Resolve a suite from a path, path under `suites_dir`, or declared name."""
    clean = spec.strip()
    if clean:
        if Path(clean).is_absolute():
            bases: list[Path | None] = [None]
        else:
            # allow_cwd=False skips Path.cwd() so relative paths only search
            # repo_root and suites_dir (e.g. when inspecting another checkout).
            cwd_first = allow_cwd and (clean in (".", "..")
                                       or clean.startswith(("./", "../")))
            raw = ((Path.cwd(), repo_root, suites_dir) if cwd_first else
                   (repo_root, suites_dir, Path.cwd() if allow_cwd else None))
            bases = list(
                dict.fromkeys(b.resolve() for b in raw if b is not None))
        for base in bases:
            for cand in _path_candidates(base, clean):
                if _is_suite_target(cand):
                    return cand.resolve()

    entries = _scan_suites(suites_dir)
    if clean:
        declared = [path for path, _, name in entries if name == clean]
        if len(declared) > 1:
            root_p = repo_root.resolve() if repo_root is not None else None
            listed = "\n".join(
                f"  {p.resolve().relative_to(root_p).as_posix() if root_p and p.resolve().is_relative_to(root_p) else p}"
                for p in declared)
            raise SuiteLookupError(
                f"Suite '{clean}' is ambiguous; these suites declare that name:\n"
                f"{listed}\nPass the file path instead.")
        if declared:
            return declared[0].resolve()

    available = ", ".join(sorted(_names_from_entries(entries))) or "(none)"
    raise SuiteLookupError(f"Suite '{spec}' not found.\n"
                           f"Available suites: {available}")
