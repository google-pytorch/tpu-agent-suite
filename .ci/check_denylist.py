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
"""PR diff and repository denylist scanner for internal references."""

from __future__ import annotations

import argparse
import base64
import dataclasses
import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "google" / "denylist.yaml"
CACHE_CONFIG_PATH = (Path.home() / ".cache" / "tpu-agent-suite" /
                     "denylist.yaml")
ENV_DENYLIST_B64 = "INTERNAL_DENYLIST_B64"
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclasses.dataclass(frozen=True)
class Rule:
    """A compiled denylist regex rule."""

    id: str
    regex: re.Pattern[str]
    description: str
    must_match: str = ""
    must_not_match: str = ""


def _esc(s: str) -> str:
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


@dataclasses.dataclass(frozen=True)
class Finding:
    """One denylist match in a specific file and 1-based line."""

    rule_id: str
    file: str
    line: int
    matched: str
    description: str

    def format(self, github: bool = False, redact: bool = False) -> str:
        detail = (
            f"[{self.rule_id}] {self.description}" if redact else
            f"[{self.rule_id}] matched {self.matched!r} -- {self.description}")
        if not github:
            return f"{self.file}:{self.line}: {detail}"
        return f"::error file={_esc(self.file)},line={self.line}::{_esc(detail)}"


def load_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    env_b64: str | None = None,
) -> tuple[list[str], list[Rule]]:
    """Loads `(exclude_globs, rules)` from `$INTERNAL_DENYLIST_B64`, `config_path`, or cache."""
    b64 = (env_b64 if env_b64 is not None else os.environ.get(
        ENV_DENYLIST_B64, "")).strip()
    if b64:
        raw_text = base64.b64decode(b64).decode("utf-8")
    elif config_path.is_file():
        raw_text = config_path.read_text(encoding="utf-8")
    elif config_path == DEFAULT_CONFIG_PATH and CACHE_CONFIG_PATH.is_file():
        raw_text = CACHE_CONFIG_PATH.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError(f"Denylist config not found: {config_path}")

    data = yaml.safe_load(raw_text) or {}
    excludes = list(data.get("exclude_paths", []))
    rules = [
        Rule(
            id=r["id"],
            regex=re.compile(r["pattern"],
                             0 if r.get("case_sensitive") else re.IGNORECASE),
            description=r["description"].strip(),
            must_match=r.get("must_match", ""),
            must_not_match=r.get("must_not_match", ""),
        ) for r in data.get("rules", [])
    ]
    return excludes, rules


def is_excluded(rel_path: str, exclude_globs: Sequence[str]) -> bool:
    """Returns True if `rel_path` matches any exclude glob."""
    norm = rel_path.lstrip("/")
    return any(fnmatch.fnmatch(norm, pat) for pat in exclude_globs)


def scan_line(rel_path: str, line_no: int, line: str,
              rules: Sequence[Rule]) -> list[Finding]:
    """Scans a single line against `rules`."""
    return [
        Finding(r.id, rel_path, line_no, m.group(0), r.description)
        for r in rules if (m := r.regex.search(line))
    ]


def parse_unified_diff(diff_text: str, exclude_globs: Sequence[str],
                       rules: Sequence[Rule]) -> list[Finding]:
    """Scans added lines (`+`) and added/renamed file paths in a unified diff."""
    findings: list[Finding] = []
    cur_file: str | None = None
    line_no = 0

    for raw in diff_text.splitlines():
        if raw.startswith("diff --git "):
            line_no = 0
            parts = raw.split(" b/", 1)
            target = parts[1].strip() if len(parts) == 2 else ""
            if not target or is_excluded(target, exclude_globs):
                cur_file = None
            else:
                cur_file = target
                findings.extend(scan_line(cur_file, 1, cur_file, rules))
        elif line_no == 0 and raw.startswith("+++ "):
            target = raw[4:].strip()
            target = target[2:] if target.startswith("b/") else target
            if target == "/dev/null" or is_excluded(target, exclude_globs):
                cur_file = None
            elif cur_file != target:
                cur_file = target
                findings.extend(scan_line(cur_file, 1, cur_file, rules))
        elif cur_file is None:
            continue
        elif m := _HUNK_RE.match(raw):
            line_no = int(m.group(1))
        elif line_no == 0:
            continue
        elif raw.startswith("+"):
            findings.extend(scan_line(cur_file, line_no, raw[1:], rules))
            line_no += 1
        elif not raw.startswith(("-", "\\")):
            line_no += 1

    return findings


def scan_paths(root: Path, targets: Sequence[Path],
               exclude_globs: Sequence[str],
               rules: Sequence[Rule]) -> list[Finding]:
    """Scans all UTF-8 text files under `targets` (relative to `root`)."""
    findings: list[Finding] = []
    root_resolved = root.resolve()
    files: list[Path] = []
    for t in targets:
        p = (t if t.is_absolute() else root / t).resolve()
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.is_file()))
        else:
            raise FileNotFoundError(f"Scan target does not exist: {t}")

    for fpath in files:
        try:
            rel = fpath.relative_to(root_resolved).as_posix()
        except ValueError:
            rel = fpath.as_posix()
        if is_excluded(rel, exclude_globs):
            continue
        findings.extend(scan_line(rel, 1, rel, rules))
        data = fpath.read_bytes()
        if b"\x00" in data[:8192]:
            continue
        text = data.decode("utf-8", errors="replace")
        for idx, line in enumerate(text.splitlines(), start=1):
            findings.extend(scan_line(rel, idx, line, rules))
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--diff", metavar="RANGE")
    group.add_argument("--staged", action="store_true")
    group.add_argument("--paths",
                       nargs="+",
                       type=Path,
                       metavar="PATH",
                       help="Files or directories to scan directly")
    parser.add_argument("--output-format",
                        choices=("text", "github"),
                        default="text")
    args = parser.parse_args(argv)

    in_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    used_secret = bool(os.environ.get(ENV_DENYLIST_B64, "").strip())
    try:
        excludes, rules = load_config(args.config)
    except FileNotFoundError as exc:
        if args.staged and in_ci:
            print("Skipping staged denylist check in GitHub Actions.")
            return 0
        print(
            f"{exc} (also checked ${ENV_DENYLIST_B64} and {CACHE_CONFIG_PATH}).\n"
            f"Run: mkdir -p {CACHE_CONFIG_PATH.parent} && "
            f"cp <path-to-denylist.yaml> {CACHE_CONFIG_PATH}\n"
            f"(or export {ENV_DENYLIST_B64}; see docs/developing.md).",
            file=sys.stderr,
        )
        return 1
    source = f"${ENV_DENYLIST_B64}" if used_secret else str(args.config.name)
    if args.paths:
        findings = scan_paths(args.root, args.paths, excludes, rules)
        target_label = "paths (" + ", ".join(str(p) for p in args.paths) + ")"
    else:
        from_ref = os.environ.get("PRE_COMMIT_FROM_REF", "").strip()
        to_ref = os.environ.get("PRE_COMMIT_TO_REF", "").strip()
        if args.staged and from_ref and to_ref:
            diff_target = f"{from_ref}...{to_ref}"
        else:
            diff_target = "--cached" if args.staged else args.diff
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(args.root),
                "diff",
                "--unified=0",
                "--no-color",
                "--no-ext-diff",
                "--diff-filter=ACMRT",
                diff_target,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        findings = parse_unified_diff(proc.stdout, excludes, rules)
        target_label = f"diff ({diff_target})"
    if not findings:
        print(f"Denylist check passed for {target_label} "
              f"({len(rules)} rules from {source}, 0 findings).")
        return 0

    redact = used_secret and in_ci
    for f in findings:
        print(f.format(github=args.output_format == "github", redact=redact))
    print(f"\nDenylist check failed: {len(findings)} finding(s).",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
