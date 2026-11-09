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
"""Command-line entry point for building TPU agent suite packages.

Runs the `compose`, `verify`, `lint`, `translate` and `archive` commands. The
build itself lives in `framework/builder/`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import tarfile
from typing import Any, Sequence

# Run as a script, this file has its own directory on sys.path, not the one
# that holds the framework package: in the repository the root, and the compiler
# only runs there. Add it, so `framework` imports.
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from framework import (  # noqa: E402  # isort: skip
    jsonio, lint, paths, utils)
from framework.builder import (  # noqa: E402  # isort: skip
    DEFAULT_MANIFEST, HARNESS_TRANSLATORS, dist_assets, load_package,
    package_checks, stage_package_with_overlays, translate, verify_package)

Path = pathlib.Path


def create_archives(
    dist_dir: Path,
    output_dir: Path,
    harnesses: Sequence[str] | None = None,
) -> list[Path]:
    """Package compiled harness distributions into standalone .tar.gz archives."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    target_harnesses = set(harnesses) if harnesses else set(
        HARNESS_TRANSLATORS)

    if not dist_dir.is_dir():
        raise utils.TpuAgentSuiteError(
            f"Distribution directory does not exist: {dist_dir}")

    for harness in sorted(target_harnesses):
        harness_dir = dist_dir / harness
        if not harness_dir.is_dir():
            continue
        for pkg_dir in sorted(harness_dir.iterdir()):
            if not pkg_dir.is_dir():
                continue
            build_file = pkg_dir / "tpu-agent-suite-build.json"
            if not build_file.is_file():
                continue
            build_meta = jsonio.load_json(build_file)
            source_name = build_meta.get("source", pkg_dir.name)
            version = build_meta.get("version", utils.TPU_AGENT_SUITE_VERSION)

            archive_base = f"{source_name}-{harness}-v{version}"

            # Create .tar.gz
            tar_path = output_dir / f"{archive_base}.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(pkg_dir, arcname=archive_base)
            created.append(tar_path)

    if not created:
        raise utils.TpuAgentSuiteError(
            f"No distribution packages found in '{dist_dir}' for harnesses: {', '.join(sorted(target_harnesses))}"
        )

    return created


def compose_package(
    package_dir: Path,
    output_file: Path | None = None,
    allow_extra_fields: bool = False,
    skills: Sequence[str] | None = None,
    manifest_name: str = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Compose and validate a modular package into a single harness-agnostic manifest."""
    manifest = load_package(package_dir,
                            allow_extra_fields=allow_extra_fields,
                            skills=skills,
                            manifest=manifest_name)
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        jsonio.write_json(output_file, manifest)
    return manifest


def run_lint(
    package_dir: Path,
    checks: Sequence[str] | None = None,
    output_format: str = "text",
    allow_extra_fields: bool = False,
    manifest_name: str = DEFAULT_MANIFEST,
    source_package_dir: Path | None = None,
) -> int:
    """Report content problems; returns 1 when any are found.

    These are advisory checks -- the ones strict enough to gate on live in
    `verify.verify_package`, and what remains here is heuristic and can
    legitimately fire on correct content. The non-zero exit is still the honest
    signal: a linter that always succeeds is indistinguishable from one finding
    nothing. Keep this out of the *required* status checks rather than muting
    it.

    `--output-format github` emits workflow annotations, so findings surface
    inline on the changed lines of a pull request.

    """
    try:
        manifest = load_package(package_dir,
                                allow_extra_fields=allow_extra_fields,
                                manifest=manifest_name)
    except utils.TpuAgentSuiteError as exc:
        print(f"Error loading package for linting: {exc}")
        return 1
    findings = lint.run_checks(package_dir, manifest, only=checks)
    package_checks.print_delegation_warnings(manifest)
    for warning in dist_assets.undeclared_catalog_entries(
            package_dir, manifest):
        print(f"WARNING: {warning}", file=sys.stderr)
    for rel in lint.unreachable_files(package_dir, manifest):
        print(
            f"WARNING: {rel} is not reachable from any released skill and"
            " will not ship",
            file=sys.stderr)

    # Workflow annotations are resolved from the repository root, not from the
    # job's working-directory, so anchor paths there when a checkout is present.
    anchor = (source_package_dir or package_dir).resolve()
    git_root = paths.find_git_root(anchor)
    root = (anchor.relative_to(git_root)
            if git_root is not None else Path(anchor.name))

    for finding in sorted(findings, key=lambda f: (f.check, f.file, f.line)):
        if output_format == "github":
            # ::warning file=<path>,line=<n>,title=<check>::<detail>
            detail = finding.detail.replace("\n", " ")
            print(
                f"::warning file={(root / finding.file).as_posix()},"
                f"line={max(finding.line, 1)},title={finding.check}::{detail}")
        else:
            print(finding)

    if not findings:
        print("No findings!")
        return 0
    print(f"\n{len(findings)} finding(s).")
    return 1


def parse_skill_list(value: str) -> list[str]:
    """Parse a comma-separated --skills value into skill names."""
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("--skills must name at least one"
                                         " skill")
    for name in names:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            raise argparse.ArgumentTypeError(
                f"--skills takes skill names, not paths: {name!r}")
    return names


def build_compiler_parser() -> argparse.ArgumentParser:
    """Construct parser for standalone compiler commands."""
    parser = argparse.ArgumentParser(
        prog="compiler",
        description=
        "Verification and translation compiler for TPU agent suite packages.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compose
    p_compose = sub.add_parser(
        "compose",
        help="Print the manifest with everything discovered from its skills",
    )
    p_compose.add_argument("package",
                           type=Path,
                           help="Path to package directory")
    p_compose.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (prints formatted JSON to stdout if omitted)",
    )

    # verify
    p_verify = sub.add_parser("verify",
                              help="Verify package structure and schemas")
    p_verify.add_argument("package",
                          type=Path,
                          help="Path to package directory")

    # lint
    p_lint = sub.add_parser(
        "lint",
        help="Report content problems in instructions and bundled scripts",
    )
    p_lint.add_argument("package", type=Path, help="Path to package directory")
    p_lint.add_argument(
        "--output-format",
        choices=["text", "github"],
        default="text",
        help="'github' emits ::warning annotations for pull-request review",
    )
    p_lint.add_argument(
        "--check",
        action="append",
        choices=sorted(lint.CHECKS),
        default=None,
        help="Run only this check (repeatable; all checks run by default)",
    )

    # translate
    p_translate = sub.add_parser("translate",
                                 help="Translate package to target harness")
    p_translate.add_argument("package",
                             type=Path,
                             help="Path to package directory")
    p_translate.add_argument(
        "--harness",
        choices=sorted(HARNESS_TRANSLATORS),
        required=True,
        help="Target harness",
    )
    p_translate.add_argument("--output",
                             type=Path,
                             required=True,
                             help="Output directory")
    p_translate.add_argument(
        "--include-internal",
        action="store_true",
        help=
        "Include internal scripts and MCP servers in compiled distribution",
    )

    for p in (p_compose, p_verify, p_lint, p_translate):
        p.add_argument(
            "--manifest",
            default=DEFAULT_MANIFEST,
            help="Manifest file in the package directory that names the"
            f" package to build (default: {DEFAULT_MANIFEST})",
        )
        p.add_argument(
            "--allow-extra-fields",
            action="store_true",
            help="Allow unrecognized or extra properties in package manifest",
        )
        p.add_argument(
            "--no-overlays",
            action="store_true",
            help="Ignore google/overlays/<package> even when present",
        )

    # Not p_lint: it reports on the package as authored, which is what
    # package.release.json releases.
    for p in (p_compose, p_verify, p_translate):
        p.add_argument(
            "--skills",
            type=parse_skill_list,
            default=None,
            help="Comma-separated skill names to build in place of the"
            " `skills` package.release.json lists. Each is built whole, with"
            " everything it reaches.",
        )

    # archive
    p_archive = sub.add_parser(
        "archive",
        help="Create standalone .tar.gz archives from built distributions",
    )
    p_archive.add_argument(
        "--dist-dir",
        type=Path,
        default=Path("dist"),
        help="Root directory containing harness distributions (default: dist)",
    )
    p_archive.add_argument(
        "--output",
        type=Path,
        default=Path("dist/archives"),
        help="Output directory for archives (default: dist/archives)",
    )
    p_archive.add_argument(
        "--harness",
        choices=sorted(HARNESS_TRANSLATORS),
        action="append",
        default=None,
        help="Harnesses to archive (repeatable; default: all)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entrypoint for standalone compiler operations."""
    parser = build_compiler_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "compose":
            with stage_package_with_overlays(
                    args.package, enabled=not args.no_overlays) as pkg_dir:
                manifest = compose_package(
                    pkg_dir,
                    args.output,
                    allow_extra_fields=args.allow_extra_fields,
                    skills=args.skills,
                    manifest_name=args.manifest,
                )
            if args.output is None:
                print(json.dumps(manifest, indent=2))

        elif args.command == "verify":
            with stage_package_with_overlays(
                    args.package, enabled=not args.no_overlays) as pkg_dir:
                verify_package(
                    pkg_dir,
                    allow_extra_fields=args.allow_extra_fields,
                    skills=args.skills,
                    manifest_name=args.manifest,
                )
            print("Successfully verified!")

        elif args.command == "lint":
            with stage_package_with_overlays(
                    args.package, enabled=not args.no_overlays) as pkg_dir:
                return run_lint(
                    pkg_dir,
                    checks=args.check,
                    output_format=args.output_format,
                    allow_extra_fields=args.allow_extra_fields,
                    manifest_name=args.manifest,
                    source_package_dir=args.package,
                )

        elif args.command == "translate":
            with stage_package_with_overlays(
                    args.package, enabled=not args.no_overlays) as pkg_dir:
                translate(
                    pkg_dir,
                    args.harness,
                    args.output,
                    include_internal=args.include_internal,
                    allow_extra_fields=args.allow_extra_fields,
                    skills=args.skills,
                    manifest_name=args.manifest,
                )

        elif args.command == "archive":
            created = create_archives(
                dist_dir=args.dist_dir,
                output_dir=args.output,
                harnesses=args.harness,
            )
            for path in created:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"Created archive: {path} ({size_mb:.2f} MB)")

        return 0
    except utils.TpuAgentSuiteError as exc:
        print(f"compiler: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
