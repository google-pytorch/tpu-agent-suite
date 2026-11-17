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
"""Verifier version registry and runtime/streaming provenance helpers.

Used consistently by:
- ``eval/run.py`` (evaluation launch checks and fallback stamping)
- ``eval/adapter/sitecustomize.py`` (in-process Trial/Job Hub provenance stamping)
"""

from __future__ import annotations

import fnmatch
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

try:
    import tomllib
except ImportError:
    tomllib = None  # type: ignore[assignment]
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VERSIONS_PATH = (Path(__file__).resolve().parent / "verifiers" /
                         "versions.yaml")
_SKIP_HASH_DIRS = {
    "__pycache__", ".pytest_cache", ".venv", ".git", "node_modules"
}
_SKIP_HASH_FILES = {".DS_Store"}


def normalize_version_tag(version: str | int | None) -> str:
    """Normalize a version identifier (e.g. '2', 'v2', 'V02') to 'v<int>'."""
    if version is None:
        return "v1"
    if isinstance(version, int):
        return f"v{max(1, version)}"
    raw = str(version).strip().lower()
    if not raw:
        return "v1"
    m = re.match(r"^v?(\d+)$", raw)
    if m:
        return f"v{int(m.group(1))}"
    sanitized = re.sub(r"[^a-z0-9.-]+", "-", raw).strip("-")
    if not sanitized:
        return "v1"
    return sanitized if sanitized.startswith("v") else f"v{sanitized}"


def parse_verifier_version(version: str | int | None) -> int:
    """Parse a verifier version string ('v10', '2') into an integer for sorting."""
    if version is None:
        return 1
    if isinstance(version, int):
        return version
    raw = str(version).strip().lower()
    m = re.search(r"v?(\d+)", raw)
    return int(m.group(1)) if m else 1


def load_verifier_versions(
    versions_path: Path | None = None, ) -> dict[str, Any]:
    """Load the checked-in ``eval/verifiers/versions.yaml`` registry."""
    path = versions_path or DEFAULT_VERSIONS_PATH
    if not path.is_file():
        return {
            "version": 1,
            "families": {
                "default": {
                    "current": "v1",
                    "history": {
                        "v1": "initial"
                    }
                }
            }
        }
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {
            "version": 1,
            "families": {
                "default": {
                    "current": "v1",
                    "history": {
                        "v1": "initial"
                    }
                }
            }
        }
    return data


_ONBOARDING_ROUTING_MARKERS = (
    "onboarding-routing",
    "optimization-heuristics",
    "plain-tp-plan",
    "why-tp-plan",
    "verify-and-bench-0p6b",
)
_CSET_CATEGORY_MARKERS = (
    "kernel_optimization",
    "gpu_kernel_to_pytorch_api_conversion",
    "hardware_profiling_and_debugging",
)
_KNOWLEDGEBASE_BARE_TASKS = frozenset({
    "ask-xprof-remote",
    "hosted-xprof",
    "xprof-capture",
    "q-csat006-kv-block",
})
_XPROF_TASK_MARKERS = (
    "tpu-hello-world-xprof",
    "xprof-attention",
    "xprof-kernel-visibility",
    "xprof-attention-kernel-tuning",
    "kernel-tuning",
    "xprof-compare",
    "xprof-hbm",
    "xprof-summary",
)
_JAXBENCH_CASE_RE = re.compile(r"\b\d+[pk]-[a-z][a-z0-9-]*\b")
_TPU_INFERENCE_VLLM_RE = re.compile(
    r"(?<![a-z0-9-])tpu-inference(?:-[a-z0-9.-]+)*\b")
_CSET_WORD_RE = re.compile(r"(?<![a-z0-9])cset(?![a-z0-9])")


def _is_tpu_knowledge_base_task(task_spec: str) -> bool:
    clean = task_spec.strip().lower()
    if not clean:
        return False
    slug = clean.rsplit("/", 1)[-1]
    if slug in _KNOWLEDGEBASE_BARE_TASKS:
        return True
    if ("/" not in clean or clean.startswith("tpu-knowledge-base/")):
        if slug and not slug.startswith("."):
            return (REPO_ROOT / "eval" / "datasets" / "tpu-knowledge-base" /
                    slug).is_dir()
    return False


def resolve_verifier_family(
    target: str | None = None,
    task_names: Sequence[str] | None = None,
    explicit_family: str | None = None,
) -> str:
    """Infer the verifier family slug from an explicit value, target, or task names."""
    if explicit_family and explicit_family.strip():
        return explicit_family.strip().lower().replace("-", "_")

    tasks = [
        str(t).strip().lower() for t in (task_names or ()) if str(t).strip()
    ]
    haystack = " ".join([target or "", *tasks]).lower()

    if ("hillclimb" in haystack or "onboard-optimize" in haystack
            or "close-gaps" in haystack
            or _TPU_INFERENCE_VLLM_RE.search(haystack)):
        return "onboard_hillclimb"
    if any(m in haystack for m in _ONBOARDING_ROUTING_MARKERS):
        return "onboarding_routing"
    if "onboard" in haystack or "migration" in haystack:
        return "onboard"
    if _CSET_WORD_RE.search(haystack) or any(m in haystack
                                             for m in _CSET_CATEGORY_MARKERS):
        return "cset"
    if "knowledge" in haystack or any(
            _is_tpu_knowledge_base_task(t) for t in tasks):
        return "knowledgebase"
    if "xprof" in haystack or any(m in haystack for m in _XPROF_TASK_MARKERS):
        return "xprof"
    if ("jaxbench" in haystack or "kernel-set1" in haystack
            or "suites/jaxbench" in haystack
            or _JAXBENCH_CASE_RE.search(haystack)):
        return "jaxbench"
    if "csat" in haystack:
        return "cset"
    return "default"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_digest(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _is_skipped_hash_path(path: Path) -> bool:
    if path.name in _SKIP_HASH_FILES or path.suffix == ".pyc":
        return True
    return any(part in _SKIP_HASH_DIRS for part in path.parts)


def _collect_dir_hashes(root: Path, key_prefix: str = "") -> dict[str, str]:
    """Deterministically hash all files under ``root`` keyed by POSIX relpath."""
    if not root.exists():
        return {}
    hashes: dict[str, str] = {}
    if root.is_file():
        if not _is_skipped_hash_path(root):
            key = f"{key_prefix}/{root.name}" if key_prefix else root.name
            hashes[key.lstrip("/")] = _sha256_file(root)
        return hashes
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        rel = item.relative_to(root)
        if _is_skipped_hash_path(rel):
            continue
        rel_posix = rel.as_posix()
        key = f"{key_prefix}/{rel_posix}" if key_prefix else rel_posix
        hashes[key.lstrip("/")] = _sha256_file(item)
    return dict(sorted(hashes.items()))


def _split_source_specs(
    source_paths: Sequence[str] | None, ) -> tuple[list[str], list[str]]:
    includes: list[str] = []
    excludes: list[str] = []
    for raw in source_paths or ():
        spec = str(raw).strip()
        if not spec:
            continue
        if spec.startswith("!"):
            ex = spec[1:].strip()
            if ex:
                excludes.append(ex)
        else:
            includes.append(spec)
    return includes, excludes


def _expand_single_repo_spec(spec: str, root: Path) -> list[Path]:
    p = Path(spec)
    if p.is_absolute():
        return [p] if p.exists() else []
    if any(ch in spec for ch in ("*", "?", "[")):
        return [
            Path(m)
            for m in sorted(glob.glob(str(root / spec), recursive=True))
        ]
    cand = root / spec
    return [cand] if cand.exists() else []


def _build_exclude_matcher(
    exclude_specs: Sequence[str] | None,
    root: Path,
):
    r_root = root.resolve()
    excluded_paths: set[Path] = set()
    exclude_globs: list[str] = []
    for raw in exclude_specs or ():
        ex = str(raw).strip()
        if not ex:
            continue
        exclude_globs.append(ex.rstrip("/"))
        for match in _expand_single_repo_spec(ex, r_root):
            excluded_paths.add(match.resolve())

    def _is_excluded(cand: Path) -> bool:
        if not excluded_paths and not exclude_globs:
            return False
        rc = cand.resolve()
        if rc in excluded_paths or any(parent in excluded_paths
                                       for parent in rc.parents):
            return True
        try:
            rel_parts = rc.relative_to(r_root).parts
        except ValueError:
            rel_parts = rc.parts
        for i in range(1, len(rel_parts) + 1):
            prefix = "/".join(rel_parts[:i])
            for pat in exclude_globs:
                if fnmatch.fnmatch(prefix, pat):
                    return True
        return False

    return _is_excluded


def _is_shared_verifier_relpath(rel_path: str) -> bool:
    norm = rel_path.strip().lstrip("/")
    return not (norm.startswith("eval/datasets/")
                or norm.startswith("eval/.datasets/")
                or norm.startswith("eval/suites/"))


def _expand_repo_source_paths(
    source_paths: Sequence[str] | None,
    repo_root: Path | None = None,
) -> list[Path]:
    root = (repo_root or REPO_ROOT).resolve()
    includes, excludes = _split_source_specs(source_paths)
    is_excluded = _build_exclude_matcher(excludes, root)
    resolved: list[Path] = []
    seen: set[Path] = set()
    for spec in includes:
        for c in _expand_single_repo_spec(spec, root):
            rc = c.resolve()
            if rc not in seen and not is_excluded(rc):
                seen.add(rc)
                resolved.append(rc)
    return resolved


def parse_scoring_configs(
    tests_dir: Path | None = None,
    source_paths: Sequence[Path] | None = None,
    *,
    repo_root: Path | None = None,
    exclude_specs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Parse structured reward weights, judge expectations, and verifier_params."""
    reward_tomls: dict[str, Any] = {}
    judge_configs: dict[str, Any] = {}
    verifier_params: dict[str, Any] = {}

    scan_pairs: list[tuple[str, Path]] = []
    if tests_dir and tests_dir.is_dir():
        for p in sorted(tests_dir.rglob("*")):
            if p.is_file() and not _is_skipped_hash_path(
                    p.relative_to(tests_dir)):
                scan_pairs.append(
                    (f"tests/{p.relative_to(tests_dir).as_posix()}", p))
    repo_resolved = (repo_root or REPO_ROOT).resolve()
    is_excluded = _build_exclude_matcher(exclude_specs, repo_resolved)
    for sp in source_paths or ():
        if is_excluded(sp):
            continue
        if sp.is_file():
            try:
                lbl = f"repo/{sp.resolve().relative_to(repo_resolved).as_posix()}"
            except ValueError:
                lbl = sp.name
            scan_pairs.append((lbl, sp))
        elif sp.is_dir():
            for p in sorted(sp.rglob("*")):
                if p.is_file() and not _is_skipped_hash_path(
                        p.relative_to(sp)) and not is_excluded(p):
                    try:
                        lbl = f"repo/{p.resolve().relative_to(repo_resolved).as_posix()}"
                    except ValueError:
                        lbl = f"{sp.name}/{p.relative_to(sp).as_posix()}"
                    scan_pairs.append((lbl, p))

    seen_files: set[Path] = set()
    for label, path in scan_pairs:
        rpath = path.resolve()
        if rpath in seen_files:
            continue
        seen_files.add(rpath)
        name_lower = path.name.lower()
        if name_lower == "reward.toml":
            try:
                if tomllib is None:
                    raise RuntimeError("tomllib unavailable")
                reward_tomls[label] = tomllib.loads(
                    path.read_text(encoding="utf-8"))
            except Exception:
                reward_tomls[label] = {"raw_sha256": _sha256_file(path)}
        elif name_lower.endswith("judge.toml") or (
                path.parent.name == "expectations"
                and path.suffix.lower() == ".toml"):
            try:
                if tomllib is None:
                    raise RuntimeError("tomllib unavailable")
                judge_configs[label] = tomllib.loads(
                    path.read_text(encoding="utf-8"))
            except Exception:
                judge_configs[label] = {"raw_sha256": _sha256_file(path)}
        elif "verifier_params" in name_lower or "verifer_params" in name_lower:
            if path.suffix.lower() == ".json":
                parsed = _read_json_dict(path)
                if parsed:
                    verifier_params[label] = parsed

    out: dict[str, Any] = {}
    if reward_tomls:
        out["reward_toml"] = (next(iter(reward_tomls.values()))
                              if len(reward_tomls) == 1 else reward_tomls)
    if judge_configs:
        out["expectations"] = (next(iter(judge_configs.values()))
                               if len(judge_configs) == 1 else judge_configs)
    if verifier_params:
        out["verifier_params"] = (next(iter(verifier_params.values()))
                                  if len(verifier_params) == 1 else
                                  verifier_params)
    return out


def compute_family_inputs_manifest(
    family: str,
    versions_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Hash all repository source files declared for ``family`` in versions.yaml."""
    root = (repo_root or REPO_ROOT).resolve()
    registry = load_verifier_versions(versions_path=versions_path)
    families = registry.get("families") if isinstance(registry.get("families"),
                                                      dict) else {}
    fam_key = (family or "default").strip().lower().replace("-", "_")
    entry = families.get(fam_key)
    if not isinstance(entry, dict):
        entry = {}

    raw_paths = entry.get("source_paths") if isinstance(
        entry.get("source_paths"), list) else []
    _, exclude_specs = _split_source_specs(raw_paths)
    is_excluded = _build_exclude_matcher(exclude_specs, root)
    expanded = _expand_repo_source_paths(raw_paths, repo_root=root)

    file_hashes: dict[str, str] = {}
    for sp in expanded:
        if is_excluded(sp):
            continue
        try:
            rel_base = sp.relative_to(root).as_posix()
        except ValueError:
            rel_base = sp.name
        if sp.is_file():
            if not _is_skipped_hash_path(sp):
                file_hashes[rel_base] = _sha256_file(sp)
        elif sp.is_dir():
            for child in sorted(sp.rglob("*")):
                if not child.is_file():
                    continue
                if _is_skipped_hash_path(
                        child.relative_to(sp)) or is_excluded(child):
                    continue
                try:
                    c_rel = child.relative_to(root).as_posix()
                except ValueError:
                    c_rel = f"{rel_base}/{child.relative_to(sp).as_posix()}"
                file_hashes[c_rel] = _sha256_file(child)

    file_hashes = dict(sorted(file_hashes.items()))
    inputs_sha = _canonical_digest(file_hashes) if file_hashes else ""
    parsed_cfg = parse_scoring_configs(
        source_paths=expanded,
        repo_root=root,
        exclude_specs=exclude_specs,
    )
    return {
        "verifier_family": fam_key,
        "source_paths": [str(p) for p in raw_paths],
        "file_count": len(file_hashes),
        "file_hashes": file_hashes,
        "inputs_sha256": inputs_sha,
        **parsed_cfg,
    }


def compute_scoring_inputs_manifest(
    tests_dir: Path | None = None,
    family: str | None = None,
    versions_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Compute a complete, canonical scoring-inputs manifest across scripts & weights."""
    has_tests_dir = bool(tests_dir and tests_dir.is_dir())
    test_hashes = (_collect_dir_hashes(tests_dir, key_prefix="tests")
                   if has_tests_dir else {})
    tests_sha = _canonical_digest(test_hashes) if test_hashes else ""

    fam_manifest: dict[str, Any] = {}
    if family:
        fam_manifest = compute_family_inputs_manifest(
            family=family,
            versions_path=versions_path,
            repo_root=repo_root,
        )
    raw_fam_hashes = fam_manifest.get("file_hashes") or {}
    if has_tests_dir:
        # Per-trial manifests include only shared repository verifier helpers
        # (e.g. eval/verifiers/...), keeping trial digests decoupled from
        # unrelated sibling datasets/suites in the same family.
        fam_hashes = {
            f"repo/{k}": v
            for k, v in raw_fam_hashes.items()
            if _is_shared_verifier_relpath(k)
        }
        raw_sources = fam_manifest.get("source_paths") or []
        parse_source_specs = [
            s for s in raw_sources if str(s).strip().startswith("!")
            or _is_shared_verifier_relpath(str(s).strip())
        ]
    else:
        fam_hashes = {f"repo/{k}": v for k, v in raw_fam_hashes.items()}
        parse_source_specs = fam_manifest.get("source_paths") or []

    _, parse_excludes = _split_source_specs(parse_source_specs)
    fam_sha = str(fam_manifest.get("inputs_sha256") or "")

    merged_hashes = dict(sorted({**fam_hashes, **test_hashes}.items()))
    verifier_digest = _canonical_digest({"files": merged_hashes
                                         }) if merged_hashes else ""

    parsed = parse_scoring_configs(
        tests_dir=tests_dir,
        source_paths=_expand_repo_source_paths(
            parse_source_specs,
            repo_root=repo_root,
        ),
        repo_root=repo_root,
        exclude_specs=parse_excludes,
    )

    return {
        "verifier_digest": verifier_digest,
        "family_inputs_sha256": fam_sha,
        "tests_sha256": tests_sha,
        "family_file_count": len(fam_hashes),
        "test_file_count": len(test_hashes),
        "file_hashes": merged_hashes,
        **parsed,
    }


def check_version_drift(
    versions_path: Path | None = None,
    repo_root: Path | None = None,
    family: str | None = None,
) -> list[dict[str, Any]]:
    """Compare live scoring-input hashes against recorded hashes in versions.yaml."""
    registry = load_verifier_versions(versions_path=versions_path)
    families = registry.get("families") if isinstance(registry.get("families"),
                                                      dict) else {}
    target_family = (family.strip().lower().replace("-", "_")
                     if family and family.strip() else None)
    report: list[dict[str, Any]] = []
    for fam_key, entry in families.items():
        if not isinstance(entry, dict):
            continue
        if target_family is not None and fam_key != target_family:
            continue
        current = normalize_version_tag(entry.get("current") or "v1")
        recorded_map = entry.get("inputs_sha256") if isinstance(
            entry.get("inputs_sha256"), dict) else {}
        recorded_sha = str(recorded_map.get(current) or "").strip()
        fam_manifest = compute_family_inputs_manifest(
            family=fam_key,
            versions_path=versions_path,
            repo_root=repo_root,
        )
        live_sha = str(fam_manifest.get("inputs_sha256") or "")
        drifted = (recorded_sha != live_sha) if (live_sha
                                                 or recorded_sha) else False
        report.append({
            "family": fam_key,
            "current_version": current,
            "recorded_sha256": recorded_sha,
            "live_sha256": live_sha,
            "file_count": fam_manifest.get("file_count", 0),
            "drifted": drifted,
        })
    return report


_VERSIONS_HEADER = (
    "# Copyright 2026 Google LLC\n"
    "#\n"
    "# Licensed under the Apache License, Version 2.0 (the \"License\");\n"
    "# you may not use this file except in compliance with the License.\n"
    "# You may obtain a copy of the License at\n"
    "#\n"
    "#     http://www.apache.org/licenses/LICENSE-2.0\n"
    "#\n"
    "# Unless required by applicable law or agreed to in writing, software\n"
    "# distributed under the License is distributed on an \"AS IS\" BASIS,\n"
    "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
    "# See the License for the specific language governing permissions and\n"
    "# limitations under the License.\n"
    "#\n"
    "# Single source of truth for verifier version provenance and scoring-input\n"
    "# hashes. Checked in so evaluation runs deterministically track verifier\n"
    "# scripts, reward weights, and params.\n\n")


def _write_verifier_versions(registry: dict[str, Any],
                             versions_path: Path) -> None:
    versions_path.write_text(
        _VERSIONS_HEADER +
        yaml.safe_dump(registry, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _resolve_git_head_commit(repo_root: Path | None = None) -> str:
    root = repo_root or REPO_ROOT
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass
    return (os.environ.get("GITHUB_SHA") or "").strip() or "HEAD"


def _dirty_source_paths(
    source_paths: Sequence[str],
    repo_root: Path | None = None,
) -> list[str]:
    """Return uncommitted git paths overlapping a verifier family's source_paths."""
    root = repo_root or REPO_ROOT
    includes, excludes = _split_source_specs(source_paths)
    is_excluded = _build_exclude_matcher(excludes, root)
    expanded: list[str] = []
    for pat in includes:
        if any(ch in pat for ch in ("*", "?", "[")):
            for match in sorted(root.glob(pat)):
                if is_excluded(match):
                    continue
                try:
                    rel = match.relative_to(root).as_posix()
                except ValueError:
                    continue
                if rel and rel not in expanded:
                    expanded.append(rel)
        elif pat not in expanded and not is_excluded(root / pat):
            expanded.append(pat)
    if not expanded:
        return []
    try:
        proc = subprocess.run(
            ["git", "-C",
             str(root), "status", "--porcelain", "--", *expanded],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0:
            return []
        dirty: list[str] = []
        for line in proc.stdout.splitlines():
            if len(line) <= 3:
                continue
            raw_path = line[3:].strip()
            if " -> " in raw_path:
                raw_path = raw_path.split(" -> ", 1)[1].strip()
            if raw_path.startswith('"') and raw_path.endswith('"'):
                raw_path = raw_path[1:-1]
            if not raw_path or is_excluded(root / raw_path):
                continue
            dirty.append(raw_path)
        return dirty
    except Exception:
        return []


def bump_verifier_version(
    family: str,
    *,
    version: str | None = None,
    commit: str | None = None,
    versions_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Bump ``family`` current version, record git commit in history, and refresh inputs_sha256."""
    path = versions_path or DEFAULT_VERSIONS_PATH
    registry = load_verifier_versions(versions_path=path)
    families = registry.get("families") if isinstance(registry.get("families"),
                                                      dict) else {}
    fam_key = (family or "").strip().lower().replace("-", "_")
    if fam_key not in families or not isinstance(families[fam_key], dict):
        valid = ", ".join(sorted(families.keys()))
        raise KeyError(
            f"Unknown verifier family '{family}' in {path}. Valid families: {valid}"
        )

    entry = families[fam_key]
    prev_ver = normalize_version_tag(entry.get("current") or "v1")
    next_ver = (normalize_version_tag(version)
                if version else f"v{parse_verifier_version(prev_ver) + 1}")
    resolved_commit = (commit or
                       _resolve_git_head_commit(repo_root=repo_root)).strip()
    if commit is None:
        raw_sources = entry.get("source_paths")
        source_paths = list(raw_sources) if isinstance(raw_sources,
                                                       list) else []
        dirty = _dirty_source_paths(source_paths, repo_root=repo_root)
        if dirty:
            preview = ", ".join(dirty[:3]) + (f" (+{len(dirty) - 3} more)"
                                              if len(dirty) > 3 else "")
            print(
                f"Warning: Family '{fam_key}' has uncommitted changes ({preview}); "
                f"history[{next_ver}] recorded pre-edit HEAD ({resolved_commit[:12]}). "
                "Commit scoring changes before bumping so git SHA matches inputs_sha256.",
                file=sys.stderr,
            )

    entry["current"] = next_ver
    history = entry.get("history")
    if not isinstance(history, dict):
        history = {}
        entry["history"] = history
    history[next_ver] = resolved_commit

    fam_manifest = compute_family_inputs_manifest(
        family=fam_key,
        versions_path=path,
        repo_root=repo_root,
    )
    live_sha = str(fam_manifest.get("inputs_sha256") or "")
    sha_map = entry.get("inputs_sha256")
    if not isinstance(sha_map, dict):
        sha_map = {}
        entry["inputs_sha256"] = sha_map
    if live_sha:
        sha_map[next_ver] = live_sha

    _write_verifier_versions(registry, path)
    return {
        "family": fam_key,
        "previous_version": prev_ver,
        "current_version": next_ver,
        "commit": resolved_commit,
        "inputs_sha256": live_sha,
        "file_count": fam_manifest.get("file_count", 0),
    }


def refresh_verifier_versions(
    versions_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Update ``inputs_sha256[current]`` in ``versions.yaml`` for all families."""
    path = versions_path or DEFAULT_VERSIONS_PATH
    registry = load_verifier_versions(versions_path=path)
    families = registry.get("families") if isinstance(registry.get("families"),
                                                      dict) else {}
    for fam_key, entry in families.items():
        if not isinstance(entry, dict):
            continue
        current = normalize_version_tag(entry.get("current") or "v1")
        fam_manifest = compute_family_inputs_manifest(
            family=fam_key,
            versions_path=path,
            repo_root=repo_root,
        )
        live_sha = str(fam_manifest.get("inputs_sha256") or "")
        if live_sha:
            sha_map = entry.get("inputs_sha256")
            if not isinstance(sha_map, dict):
                sha_map = {}
                entry["inputs_sha256"] = sha_map
            sha_map[current] = live_sha

    _write_verifier_versions(registry, path)
    return registry


def resolve_verifier_info(
    family: str | None = None,
    version: str | None = None,
    versions_path: Path | None = None,
    tests_dir: Path | None = None,
    strict: bool = False,
    strict_family: bool = False,
    strict_version: bool = False,
) -> dict[str, Any]:
    """Resolve verifier provenance including version, commit, and scoring digest."""
    registry = load_verifier_versions(versions_path=versions_path)
    families = registry.get("families") if isinstance(registry.get("families"),
                                                      dict) else {}
    fam_key = (family or "default").strip().lower().replace("-", "_")
    if (strict or
            strict_family) and family is not None and fam_key not in families:
        valid_fams = ", ".join(sorted(families.keys()))
        raise ValueError(
            f"Unknown verifier family '{family}' in versions.yaml. "
            f"Valid families: {valid_fams}")
    entry = families.get(fam_key) or families.get("default") or {}
    if not isinstance(entry, dict):
        entry = {}

    resolved_ver = normalize_version_tag(version or entry.get("current")
                                         or "v1")
    history = entry.get("history") if isinstance(entry.get("history"),
                                                 dict) else {}
    inputs_map = entry.get("inputs_sha256") if isinstance(
        entry.get("inputs_sha256"), dict) else {}
    if (strict or strict_version) and version is not None:
        known_vers = {
            normalize_version_tag(k)
            for k in ([entry.get("current")] if entry.get("current") else []) +
            list(history.keys()) + list(inputs_map.keys())
        }
        if resolved_ver not in known_vers:
            raise ValueError(
                f"Unknown verifier version '{resolved_ver}' for family '{fam_key}' in versions.yaml. "
                f"Known versions: {', '.join(sorted(known_vers)) or 'none'}")

    commit = str(
        history.get(resolved_ver) or os.environ.get("GITHUB_SHA")
        or "").strip() or "initial"

    recorded_sha = str(inputs_map.get(resolved_ver) or "").strip()

    manifest = compute_scoring_inputs_manifest(
        tests_dir=tests_dir,
        family=fam_key,
        versions_path=versions_path,
    )
    verifier_digest = str(
        manifest.get("verifier_digest") or recorded_sha or "")

    return {
        "verifier_family":
        fam_key,
        "verifier_version":
        resolved_ver,
        "verifier_commit":
        commit,
        "verifier_digest":
        verifier_digest,
        "family_inputs_sha256":
        str(manifest.get("family_inputs_sha256") or recorded_sha),
        "recorded_inputs_sha256":
        recorded_sha,
    }


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json_dict(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _resolve_trial_tests_dir(
    trial_dir: Path,
    task_search_roots: Sequence[Path] | None = None,
) -> Path | None:
    """Resolve the source task ``tests/`` directory for a Harbor trial directory."""
    trial_cfg = _read_json_dict(trial_dir / "config.json")
    trial_res = _read_json_dict(trial_dir / "result.json")

    raw_paths: list[str] = []
    for cand in (
        (trial_cfg.get("task") or {}).get("path") if isinstance(
            trial_cfg.get("task"), Mapping) else None,
        ((trial_res.get("config") or {}).get("task")
         or {}).get("path") if isinstance(
             (trial_res.get("config") or {}).get("task"), Mapping) else None,
        (trial_res.get("task_id") or {}).get("path") if isinstance(
            trial_res.get("task_id"), Mapping) else None,
    ):
        if cand and str(cand).strip():
            raw_paths.append(str(cand).strip())

    for raw in raw_paths:
        p = Path(raw)
        for base in ((p, ) if p.is_absolute() else (p, REPO_ROOT / p)):
            tests_cand = base if base.name == "tests" else (base / "tests")
            if tests_cand.is_dir():
                return tests_cand
        raw_norm = raw.replace("\\", "/")
        for marker in ("eval/.datasets/", "eval/datasets/"):
            if marker in raw_norm:
                rel = raw_norm.split(marker, 1)[1].strip("/")
                if rel:
                    mapped = REPO_ROOT / marker.rstrip("/") / rel
                    tests_cand = (mapped if mapped.name == "tests" else
                                  (mapped / "tests"))
                    if tests_cand.is_dir():
                        return tests_cand

    task_name = str(
        trial_res.get("task_name")
        or ((trial_cfg.get("task") or {}).get("name") if isinstance(
            trial_cfg.get("task"), Mapping) else "")
        or trial_dir.name.split("__")[0]).strip()
    if task_name:
        short_name = task_name.rsplit("/", 1)[-1]
        roots: list[Path] = []
        seen_roots: set[Path] = set()
        for cand_root in (
                *(task_search_roots or ()),
                REPO_ROOT / "eval" / "datasets",
                REPO_ROOT / "eval" / ".datasets",
        ):
            if cand_root and cand_root not in seen_roots:
                seen_roots.add(cand_root)
                roots.append(cand_root)
        for root in roots:
            if not root.is_dir():
                continue
            for rel_name in dict.fromkeys((task_name, short_name)):
                cand = root / rel_name / "tests"
                if cand.is_dir():
                    return cand
            for pattern in (
                    f"*/{short_name}/tests",
                    f"*/*/{short_name}/tests",
                    f"*/*/*/{short_name}/tests",
            ):
                for nested in sorted(root.glob(pattern)):
                    if nested.is_dir():
                        return nested
    return None


def stamp_trial_verifier_provenance(
    trial_dir: Path,
    verifier_family: str,
    verifier_version: str,
    verifier_commit: str | None = None,
    *,
    job_dir: Path | None = None,
    job_name: str | None = None,
    tests_dir: Path | None = None,
    scoring_manifest: dict[str, Any] | None = None,
    task_search_roots: Sequence[Path] | None = None,
    stamped_at: str | None = None,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stamp verifier provenance onto a trial's result.json and sidecar file."""
    if not trial_dir.is_dir():
        return {}

    effective_job_dir = job_dir or trial_dir.parent
    effective_job_name = job_name or effective_job_dir.name
    resolved_stamped_at = stamped_at or datetime.now(timezone.utc).isoformat()

    trial_tests_dir: Path | None = (tests_dir if
                                    (tests_dir is not None
                                     and tests_dir.is_dir()) else None)
    if trial_tests_dir is None and not isinstance(scoring_manifest, dict):
        trial_tests_dir = _resolve_trial_tests_dir(
            trial_dir,
            task_search_roots=task_search_roots,
        )

    if isinstance(info, Mapping):
        info_dict = dict(info)
    else:
        info_dict = resolve_verifier_info(
            family=verifier_family,
            version=verifier_version,
            tests_dir=trial_tests_dir,
        )
    if verifier_commit:
        info_dict["verifier_commit"] = verifier_commit

    trial_manifest = (dict(scoring_manifest) if isinstance(
        scoring_manifest, dict) else compute_scoring_inputs_manifest(
            tests_dir=trial_tests_dir,
            family=info_dict["verifier_family"],
        ))
    trial_digest = str(
        trial_manifest.get("verifier_digest")
        or info_dict.get("verifier_digest") or "")

    trial_meta: dict[str, Any] = {
        "job_name": effective_job_name,
        "trial_name": trial_dir.name,
        "verifier_family": info_dict["verifier_family"],
        "verifier_version": info_dict["verifier_version"],
        "verifier_commit": info_dict["verifier_commit"],
        "verifier_digest": trial_digest,
        "stamped_at": resolved_stamped_at,
        "scoring_inputs": trial_manifest,
    }

    trial_res_path = trial_dir / "result.json"
    trial_res = _read_json_dict(
        trial_res_path) if trial_res_path.is_file() else {}
    if trial_res_path.is_file() and trial_res:
        trial_res["verifier_family"] = info_dict["verifier_family"]
        trial_res["verifier_version"] = info_dict["verifier_version"]
        trial_res["verifier_commit"] = info_dict["verifier_commit"]
        if trial_digest:
            trial_res["verifier_digest"] = trial_digest
        _write_json_dict(trial_res_path, trial_res)
        if trial_res.get("id"):
            trial_meta["trial_id"] = trial_res["id"]

    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    _write_json_dict(verifier_dir / "verifier_metadata.json", trial_meta)
    return trial_meta


def stamp_job_verifier_provenance(
    job_dir: Path,
    verifier_family: str,
    verifier_version: str,
    verifier_commit: str | None = None,
    *,
    tests_dir: Path | None = None,
    scoring_manifest: dict[str, Any] | None = None,
    per_trial_scoring: Mapping[str, dict[str, Any]] | None = None,
    task_search_roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Stamp verifier provenance onto a Harbor job directory and its trials.

    Preserves existing timestamps and scores in ``result.json``, records
    ``verifier_family``, ``verifier_version``, ``verifier_commit``, and
    ``verifier_digest`` on job/trial ``result.json``, and writes complete
    ``scoring_inputs`` into ``verifier_metadata.json`` sidecars.
    """
    if not job_dir.is_dir():
        return {}

    info = resolve_verifier_info(
        family=verifier_family,
        version=verifier_version,
        tests_dir=tests_dir,
    )
    if verifier_commit:
        info["verifier_commit"] = verifier_commit

    trial_dirs = [
        child for child in sorted(job_dir.iterdir())
        if child.is_dir() and not child.name.startswith(".") and any(
            (child / marker).exists()
            for marker in ("result.json", "verifier", "agent", "artifacts"))
    ]

    effective_per_trial: dict[str, dict[str, Any]] = ({
        k: dict(v)
        for k, v in per_trial_scoring.items()
    } if isinstance(per_trial_scoring, Mapping) else {})
    if not effective_per_trial and tests_dir is None:
        for child in trial_dirs:
            t_tests = _resolve_trial_tests_dir(
                child, task_search_roots=task_search_roots)
            if t_tests is not None:
                effective_per_trial[
                    child.name] = compute_scoring_inputs_manifest(
                        tests_dir=t_tests,
                        family=info["verifier_family"],
                    )

    if isinstance(scoring_manifest, dict):
        resolved_manifest = dict(scoring_manifest)
    elif tests_dir is None and effective_per_trial:
        unique_digests = {
            str(m.get("verifier_digest") or ""): m
            for m in effective_per_trial.values() if m.get("verifier_digest")
        }
        if len(unique_digests) == 1:
            resolved_manifest = dict(next(iter(unique_digests.values())))
        else:
            base_manifest = compute_scoring_inputs_manifest(
                tests_dir=None,
                family=info["verifier_family"],
            )
            trial_digests = {
                k: str(m.get("verifier_digest") or "")
                for k, m in sorted(effective_per_trial.items())
            }
            resolved_manifest = {
                **base_manifest,
                "trial_digests":
                trial_digests,
                "verifier_digest":
                _canonical_digest({
                    "family_inputs_sha256":
                    base_manifest.get("family_inputs_sha256", ""),
                    "trial_digests":
                    trial_digests,
                }),
            }
    else:
        resolved_manifest = compute_scoring_inputs_manifest(
            tests_dir=tests_dir,
            family=info["verifier_family"],
        )

    verifier_digest = str(
        resolved_manifest.get("verifier_digest") or info.get("verifier_digest")
        or "")
    stamped_at = datetime.now(timezone.utc).isoformat()
    meta_payload: dict[str, Any] = {
        "job_name": job_dir.name,
        "verifier_family": info["verifier_family"],
        "verifier_version": info["verifier_version"],
        "verifier_commit": info["verifier_commit"],
        "verifier_digest": verifier_digest,
        "stamped_at": stamped_at,
        "scoring_inputs": resolved_manifest,
    }

    root_res_path = job_dir / "result.json"
    if root_res_path.is_file():
        root_res = _read_json_dict(root_res_path)
        if root_res:
            root_res["verifier_family"] = info["verifier_family"]
            root_res["verifier_version"] = info["verifier_version"]
            root_res["verifier_commit"] = info["verifier_commit"]
            if verifier_digest:
                root_res["verifier_digest"] = verifier_digest
            _write_json_dict(root_res_path, root_res)

    _write_json_dict(job_dir / "verifier_metadata.json", meta_payload)

    for child in trial_dirs:
        trial_manifest_arg = (effective_per_trial[child.name] if child.name
                              in effective_per_trial else resolved_manifest)
        stamp_trial_verifier_provenance(
            child,
            verifier_family=info["verifier_family"],
            verifier_version=info["verifier_version"],
            verifier_commit=info["verifier_commit"],
            job_dir=job_dir,
            job_name=job_dir.name,
            tests_dir=tests_dir,
            scoring_manifest=trial_manifest_arg,
            task_search_roots=task_search_roots,
            stamped_at=stamped_at,
            info=info,
        )

    return meta_payload


def _cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=
        "Inspect or refresh verifier scoring-input hashes in versions.yaml.")
    parser.add_argument(
        "--versions-path",
        type=Path,
        default=DEFAULT_VERSIONS_PATH,
        help=f"Path to versions.yaml (default: {DEFAULT_VERSIONS_PATH}).",
    )
    parser.add_argument(
        "--check-verifier-versions",
        "--check-versions",
        dest="check_versions",
        action="store_true",
        help=
        "Exit non-zero if any verifier family's scoring inputs drifted from versions.yaml.",
    )
    parser.add_argument(
        "--refresh-versions",
        action="store_true",
        help=
        "Update inputs_sha256[current] in versions.yaml to match live files.",
    )
    parser.add_argument(
        "--bump-version",
        dest="bump_families",
        action="append",
        default=[],
        help=
        "Bump current version (vN -> vN+1), record git HEAD commit, and refresh inputs_sha256 for this family (repeatable).",
    )
    parser.add_argument(
        "--to-version",
        type=str,
        default=None,
        help=
        "Explicit target version tag when using --bump-version (defaults to next vN+1).",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help=
        "Explicit commit SHA when using --bump-version (defaults to git rev-parse HEAD).",
    )
    args = parser.parse_args(argv)

    for fam in args.bump_families:
        bump_verifier_version(
            fam,
            version=args.to_version,
            commit=args.commit,
            versions_path=args.versions_path,
        )

    if args.refresh_versions:
        refresh_verifier_versions(versions_path=args.versions_path)

    report = check_version_drift(versions_path=args.versions_path)
    print(json.dumps(report, indent=2))
    drifted = [item for item in report if item.get("drifted")]
    if args.check_versions and drifted:
        summary = ", ".join(f"{item['family']} ({item['current_version']})"
                            for item in drifted)
        print(f"Drift detected in verifier families: {summary}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
