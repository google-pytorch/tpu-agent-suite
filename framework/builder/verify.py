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
"""The checks behind `compiler.py verify`, the package's CI gate.

`verify_package` fails unless:
- the package loads cleanly (see `package_loader`),
- every JSON schema it ships is a valid schema,
- every harness in `compatibility_targets` is supported,
- the strict content checks pass: links in instructions and references
  resolve, and the environment contract holds,
- Python dependencies live only in `requirements-client.txt` and
  `requirements-tpu.txt`, and every dependency the package declares is listed
  there with a version.
"""

from __future__ import annotations

import pathlib
from typing import Sequence

from framework import jsonio, lint, utils
from framework.builder import (dist_assets, package_checks, package_discovery,
                               package_loader)

Path = pathlib.Path

VERIFY_CONTENT_CHECKS = (
    lint.check_instruction_links,
    lint.check_reference_links,
    lint.check_environment_contract,
    lint.check_workspace_config_paths,
)


def validate_requirements_catalog(req_path: Path) -> None:
    """Validate that every package entry in a requirements catalog specifies a version constraint."""
    if not req_path.is_file():
        return
    for line_no, raw_line in enumerate(
            req_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        line_no_comment = line.split("#", 1)[0].strip()
        req_part = line_no_comment.split(";", 1)[0].strip()
        if not any(ch in req_part for ch in (">", "=", "<", "~", "@")):
            raise utils.TpuAgentSuiteError(
                f"Unpinned package in '{req_path.name}' at line {line_no}: '{raw_line}'. "
                "All dependencies in central requirements catalogs must specify version constraints (e.g. '>=', '==')."
            )


def verify_package(
        package_dir: Path,
        allow_extra_fields: bool = False,
        skills: Sequence[str] | None = None,
        manifest_name: str = package_loader.DEFAULT_MANIFEST) -> None:
    """Verify package manifest, schema definitions, and target compatibility."""
    # Ensure no scattered requirements files exist in sub-directories
    allowed_reqs = {(package_dir / filename).resolve()
                    for filename, _, _ in dist_assets.REQUIREMENTS_CATALOGS}
    for req_file in package_dir.rglob("*requirements*.txt"):
        rel_path = req_file.relative_to(package_dir)
        if package_discovery.dist_excluded(rel_path):
            continue  # never ships (pinned source checkouts under raw/code, tool state)
        if req_file.resolve() not in allowed_reqs:
            raise utils.TpuAgentSuiteError(
                f"Scattered requirements file found at '{rel_path}'. "
                "All Python dependencies must be declared in "
                "'requirements-client.txt' (host) or 'requirements-tpu.txt' "
                "(TPU workloads) at the package root.")

    for filename, _, _ in dist_assets.REQUIREMENTS_CATALOGS:
        validate_requirements_catalog(package_dir / filename)

    manifest = package_loader.load_package(
        package_dir,
        allow_extra_fields=allow_extra_fields,
        skills=skills,
        manifest=manifest_name)
    package_checks.print_delegation_warnings(manifest)
    for schema_rel in manifest.get("schemas", []):
        schema_path = utils.rooted(package_dir, schema_rel, "schema path")
        schema = jsonio.load_json(schema_path)
        errors = utils.validate_schema_definition(schema)
        if errors:
            raise utils.TpuAgentSuiteError(f"invalid schema {schema_rel}:\n" +
                                           "\n".join(errors))

    target_harnesses = {
        target["harness"]
        for target in manifest["compatibility_targets"]
    }
    for harness in target_harnesses:
        package_loader.target_for(manifest, harness)

    for check in VERIFY_CONTENT_CHECKS:
        findings = check(package_dir, manifest)

        if findings:
            raise utils.TpuAgentSuiteError(f"{findings[0].check}:\n" +
                                           "\n".join(f"  {f}"
                                                     for f in findings))

    # Verify declared dependencies exist in central catalogs
    for filename, key, kind in dist_assets.REQUIREMENTS_CATALOGS:
        declared = manifest.get(key, [])
        if not declared:
            continue
        req_path = package_dir / filename
        if not req_path.is_file():
            raise utils.TpuAgentSuiteError(
                f"Package declares {key} but '{req_path.name}' is missing at the package root."
            )
        known = dist_assets.extract_requirement_names(
            req_path.read_text(encoding="utf-8"))
        for dep in declared:
            if dist_assets.canonical_pkg_name(dep) not in known:
                raise utils.TpuAgentSuiteError(
                    f"Undeclared {kind} dependency: '{dep}'. All {kind} dependencies "
                    f"must be declared with versions in '{req_path.name}' at the package root."
                )
