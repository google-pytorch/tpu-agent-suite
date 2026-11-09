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
"""Shared utilities, version management, schema validation, and config managers."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import pathlib
import re
from typing import Any, Sequence

from framework import errors, jsonio

Path = pathlib.Path

# =====================================================================
# Constants & Configuration
# =====================================================================

TpuAgentSuiteError = errors.TpuAgentSuiteError


def _load_tpu_agent_suite_version() -> str:
    """Load package version from package.release.json or build metadata."""
    here = Path(__file__).resolve().parent
    for candidate in (
            # Repository layout: framework/ beside package/.
            here.parent / "package" / "package.release.json",
            # Installed bundle layout: framework/ beside the build metadata.
            here.parent / "tpu-agent-suite-build.json",
    ):
        if candidate.is_file():
            try:
                data = jsonio.load_json(candidate)
                if "version" in data:
                    return str(data["version"])
            except Exception:
                pass
    raise TpuAgentSuiteError(
        "Cannot determine TPU_AGENT_SUITE_VERSION: neither package.release.json nor "
        "tpu-agent-suite-build.json containing 'version' was found relative to utils.py"
    )


TPU_AGENT_SUITE_VERSION = _load_tpu_agent_suite_version()
DEFAULT_HOOK_TIMEOUT = 10
# What a hook's failure to reach a verdict -- timing out, crashing, printing
# something other than a valid reply -- turns into. `deny` for a hook that
# guards; `allow` for one whose failure must not hold the agent up, above all
# on the events where a deny keeps it from returning (Stop, SubagentStop).
HOOK_FAILURE_POLICIES: tuple[str, ...] = ("deny", "allow")
DEFAULT_HOOK_ON_FAILURE = "deny"
# Every such failure is recorded here, under the central store
# (paths.get_tpu_agent_suite_home()), whichever way the policy decided it: a hook that fails open is otherwise
# silent, since harnesses keep a hook's stderr out of sight when it exits 0.
HOOK_FAILURE_LOG = "logs/hook_failures.jsonl"

RUN_ID_PATTERN = r"^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$"
PACKAGE_NAME_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
# A study is named by a human and lives for weeks, so unlike a run id it is a
# slug rather than a timestamp: it has to be typeable at a prompt and readable
# in a path. Same shape as a package name so one validator style covers both.
STUDY_ID_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
SEMVER_REGEX = r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)"
VERSION_CONSTRAINT_REGEX = r"^\s*(>=|<=|>|<|==)\s*(\d+\.\d+\.\d+)\s*$"

PORTABLE_TOOLS: frozenset[str] = frozenset({
    "read", "search", "shell", "write", "edit", "ask", "web_search",
    "web_fetch"
})

# `ask` is a skill's alone. Only the unit in the user's conversation has a
# user to put a question to: an agent's caller is waiting on a result, so a
# question asked from below it arrives nowhere. The compiler enforces this;
# SKILL_ONLY_TOOLS is the list it checks.
SKILL_ONLY_TOOLS: frozenset[str] = frozenset({"ask"})

# The layout a package declares its work units in. A skill is a directory
# under skills/ holding SKILL.md; an agent is one file under agents/; a
# reference is any file under references/ (subdirectories allowed); a script
# is any file under scripts/. The manifest names only the skills; everything
# else is found by following frontmatter from them.
SKILLS_DIRNAME = "skills"
AGENTS_DIRNAME = "agents"
REFERENCES_DIRNAME = "references"
SCRIPTS_DIRNAME = "scripts"
SKILL_DOCUMENT = "SKILL.md"
DATA_DIRNAME = "data"
SCHEMAS_DIRNAME = "schemas"

# Lifecycle events a package may declare a hook for. Mirrors the enum in
# package-manifest.schema.json, which is authoritative at build time; this
# exists because the runtime ships without that schema and still has to know.
SUPPORTED_HOOK_EVENTS: tuple[str, ...] = ("PreToolUse", "PostToolUse",
                                          "SubagentStop", "SessionStart",
                                          "Stop", "SessionEnd")

JSON_SCHEMA_KEYWORDS: frozenset[str] = frozenset({
    "$id",
    "$schema",
    "additionalProperties",
    "const",
    "default",
    "dependentRequired",
    "description",
    "enum",
    "examples",
    "format",  # annotation only; not enforced
    "items",
    "maximum",
    "minimum",
    "minLength",
    "oneOf",
    "pattern",
    "properties",
    "required",
    # The one conditional rule the validator understands: per status value, the
    # fields a result carrying that status owes. It cannot be expressed with
    # `required`, which would demand them of every outcome including the
    # failures, and the validator carries no `if`/`then`. Keyed on the status
    # rather than on a fixed success word, so a package is free to name its own
    # outcomes -- there is no shared status vocabulary for the framework to
    # assume, and no table of package schema names for it to keep.
    "status_requires",
    "title",
    "type",
})

SCHEMA_TYPES: frozenset[str] = frozenset({
    "object",
    "array",
    "string",
    "integer",
    "number",
    "boolean",
    "null",
})

# =====================================================================
# Exceptions & Version Management
# =====================================================================


@dataclasses.dataclass(frozen=True, order=True)
class SemVer:
    """Semantic version (major.minor.patch) supporting full ordering comparisons."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: str) -> SemVer:
        """Parse a semantic version string."""
        match = re.search(SEMVER_REGEX, value)
        if not match:
            raise TpuAgentSuiteError(
                f"cannot parse semantic version from: {value.strip()}")
        return cls(*(int(part) for part in match.groups()))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclasses.dataclass(frozen=True)
class VersionConstraint:
    """A single version constraint, e.g. '>=1.0.0'."""

    operator: str
    target: SemVer

    def is_satisfied_by(self,
                        version: SemVer | tuple[int, int, int] | str) -> bool:
        """Evaluate if the given version satisfies this constraint."""
        if isinstance(version, tuple):
            version = SemVer(*version)
        elif isinstance(version, str):
            version = SemVer.parse(version)

        if self.operator == ">=":
            return version >= self.target
        elif self.operator == "<=":
            return version <= self.target
        elif self.operator == ">":
            return version > self.target
        elif self.operator == "<":
            return version < self.target
        elif self.operator == "==":
            return version == self.target
        return False


def parse_version_range(expression: str) -> list[VersionConstraint]:
    """Parse comma-separated version constraints, e.g. '>=1.0.0,<=2.0.0'."""
    constraints: list[VersionConstraint] = []
    for cond in expression.split(","):
        cond = cond.strip()
        if not cond:
            continue
        match = re.fullmatch(VERSION_CONSTRAINT_REGEX, cond)
        if not match:
            raise TpuAgentSuiteError(f"unsupported version constraint: {cond}")
        op, ver_str = match.groups()
        constraints.append(VersionConstraint(op, SemVer.parse(ver_str)))
    if not constraints:
        raise TpuAgentSuiteError(
            f"unsupported version constraint: {expression}")
    return constraints


def version_satisfies(version: SemVer | tuple[int, int, int] | str,
                      expression: str) -> bool:
    """Check if a version satisfies all constraints in a version range expression."""
    return all(
        c.is_satisfied_by(version) for c in parse_version_range(expression))


# =====================================================================
# Path & JSON / File Utilities
# =====================================================================


def safe_relative(value: str | Path, label: str) -> Path:
    """Ensure a path is relative and contains no directory traversal components."""
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise TpuAgentSuiteError(
            f"{label} must be a safe relative path: {value}")
    return candidate


def rooted(root: Path, value: str | Path, label: str) -> Path:
    """Resolve a relative path against a root directory, preventing directory traversal and symlink escapes."""
    rel_path = safe_relative(value, label)
    candidate = root / rel_path
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise TpuAgentSuiteError(
            f"{label} escapes its root through a symlink: {value}") from exc
    return candidate


def compute_digest(path: Path) -> str:
    """Compute the SHA-256 hexadecimal digest of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_text(text: str) -> str:
    """Compute the SHA-256 hexadecimal digest of a text fragment."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Environment questions (questions.json), read by `compute setup` and lint.


def parse_answer(question: dict[str, Any], raw: Any) -> Any:
    """Convert a raw answer to the type its question declares, and check it.

    A string answer to a boolean question becomes True/False (yes/no,
    true/false, y/n); to an integer question, an int. If the question lists
    `choices`, the answer must be one of them. Raises TpuAgentSuiteError naming the
    field otherwise.
    """
    key = question["name"]
    if question.get("type") == "boolean" and isinstance(raw, str):
        if raw.lower() not in {"yes", "no", "true", "false", "y", "n"}:
            raise TpuAgentSuiteError(f"{key} must be yes/no")
        raw = raw.lower() in {"yes", "true", "y"}

    if question.get("type") == "integer" and isinstance(raw, str):
        try:
            raw = int(raw)
        except ValueError as exc:
            raise TpuAgentSuiteError(f"{key} must be an integer") from exc

    if "choices" in question and raw not in question["choices"]:
        raise TpuAgentSuiteError(f"{key} must be one of {question['choices']}")

    return raw


def question_path(question: dict[str, Any]) -> list[str]:
    """Where a question's answer lands in environment.json, as key segments.

    A question declares `"path": "compute.params.zone"` to nest its answer;
    without one the answer sits at the top level under the question's name.
    `compute` stands for a compute profile, which `compute setup apply`
    stores under `profiles.<name>`.
    """
    return str(question.get("path") or question["name"]).split(".")


def is_profile_question(question: dict[str, Any]) -> bool:
    """Whether a question is a compute profile field rather than a setting."""
    return question_path(question)[0] == "compute"


def extract_all_json_objects(raw: Any) -> list[dict[str, Any]]:
    """Extract all candidate JSON dictionaries from raw input.

    Finds JSON dictionaries whether unadorned, embedded in prose, or enclosed
    in markdown code fences (```json ... ``` or ````json ... ````).
    """
    if isinstance(raw, dict):
        return [raw]
    if not isinstance(raw, str):
        return []

    clean = raw.strip()
    if not clean:
        return []

    # If the entire string parses cleanly as a JSON dictionary, that single
    # object is the unambiguous candidate.
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return [parsed]
    except Exception:
        pass

    candidates: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    idx = 0
    clean_len = len(clean)

    while idx < clean_len:
        start = clean.find("{", idx)
        if start == -1:
            break

        try:
            obj, end = decoder.raw_decode(clean, start)
            if isinstance(obj, dict):
                candidates.append(obj)
                idx = end
            else:
                idx = start + 1
        except Exception:
            idx = start + 1

    return candidates


# =====================================================================
# Framework Schemas & Definitions
# =====================================================================


def get_framework_dir() -> Path:
    """Locate the framework directory: the one that holds utils.py."""
    return Path(__file__).resolve().parent


def get_schemas_dir() -> Path:
    """Locate the framework schemas directory relative to utils.py."""
    return Path(__file__).resolve().parent / "schemas"


def get_definitions_dir() -> Path:
    """Locate the framework definitions directory relative to utils.py."""
    return Path(__file__).resolve().parent / "definitions"


def _load_framework_file(path: Path, what: str) -> Any:
    """Load a JSON file the framework ships, naming `what` if it is missing."""
    if not path.is_file():
        raise TpuAgentSuiteError(f"missing {what}: {path}")
    return jsonio.load_json(path)


def get_package_manifest_schema() -> dict[str, Any]:
    """Load package-manifest.schema.json from the schemas directory."""
    return _load_framework_file(
        get_schemas_dir() / "package-manifest.schema.json",
        "package manifest schema")


def get_agent_frontmatter_schema() -> dict[str, Any]:
    """Load agent-frontmatter.schema.json from the schemas directory."""
    return _load_framework_file(
        get_schemas_dir() / "agent-frontmatter.schema.json",
        "agent frontmatter schema")


def get_skill_frontmatter_schema() -> dict[str, Any]:
    """Load skill-frontmatter.schema.json from the schemas directory."""
    return _load_framework_file(
        get_schemas_dir() / "skill-frontmatter.schema.json",
        "skill frontmatter schema")


def get_reference_frontmatter_schema() -> dict[str, Any]:
    """Load reference-frontmatter.schema.json from the schemas directory."""
    return _load_framework_file(
        get_schemas_dir() / "reference-frontmatter.schema.json",
        "reference frontmatter schema")


def get_tool_mappings() -> dict[str, Any]:
    """Load canonical tool mappings from schemas/tool-mappings.json."""
    return _load_framework_file(get_schemas_dir() / "tool-mappings.json",
                                "tool mappings schema").get("tools", {})


def get_event_mappings() -> dict[str, Any]:
    """Load canonical hook event mappings from schemas/event-mappings.json."""
    return _load_framework_file(get_schemas_dir() / "event-mappings.json",
                                "event mappings schema").get("events", {})


def get_hook_output_schema() -> dict[str, Any]:
    """Load hook-output.schema.json from the schemas directory."""
    return _load_framework_file(get_schemas_dir() / "hook-output.schema.json",
                                "hook output schema")


def get_supported_hook_events() -> list[str]:
    """Supported lifecycle hook event names.

    Read from the manifest schema when it is available, which is the case at
    build time, and from SUPPORTED_HOOK_EVENTS otherwise. The runtime ships
    inside installed packages without that schema, and it needs this list to
    parse its own hook-dispatch arguments -- so sourcing it only from the
    schema turns every dispatch on an installed package into a crash.

    The constant and the schema are asserted to agree by the test suite, so
    the fallback cannot quietly diverge from what packages may declare.
    """
    try:
        schema = get_package_manifest_schema()
    except TpuAgentSuiteError:
        return list(SUPPORTED_HOOK_EVENTS)
    return list(
        schema.get("properties", {}).get("hooks", {}).get("items", {}).get(
            "properties", {}).get("event", {}).get("enum",
                                                   SUPPORTED_HOOK_EVENTS))


def get_native_tools_catalog() -> dict[str, Any]:
    """Load native_tools.json catalog from the definitions directory."""
    return _load_framework_file(get_definitions_dir() / "native_tools.json",
                                "native tools catalog")


def validate_package_name(package: str) -> None:
    """Validate that a package name is lower-case hyphen-separated."""
    if not re.fullmatch(PACKAGE_NAME_PATTERN, package):
        raise TpuAgentSuiteError("package name must be lower-case hyphen-case")


def validate_run_id(run_id: str) -> None:
    """Validate that a run ID matches the expected timestamp-uuid format."""
    if not re.fullmatch(RUN_ID_PATTERN, run_id):
        raise TpuAgentSuiteError(
            f"invalid run ID: {run_id!r} -- run IDs are made when a session "
            "starts, never chosen: a UTC timestamp, a dash and 12 hex digits, "
            "e.g. 20261007T151000Z-0123456789ab. Omit --run-id to use the "
            "active run, or copy one from `compute job list`.")


def validate_study_id(study_id: str) -> None:
    """Validate that a study ID is a lower-case hyphen-case slug."""
    if not re.fullmatch(STUDY_ID_PATTERN, study_id):
        raise TpuAgentSuiteError(
            f"invalid study ID: {study_id!r} --"
            " expected lower-case hyphen-case, e.g. qwen35-397b-ttft")


def format_package_tag(package: str, version: str = "") -> str:
    """Format canonical package tag (e.g. name@version or name)."""
    return f"{package}@{version}" if version else package


def parse_package_tag(tag: str) -> tuple[str, str]:
    """Split a package tag into (name, version); version is empty when untagged."""
    name, _, version = tag.partition("@")
    return name, version


def runtime_schema_path(name: str) -> Path:
    """Locate a runtime schema: framework/schemas/, in the repository and a build alike."""
    path = get_schemas_dir() / name
    if not path.is_file():
        raise TpuAgentSuiteError(
            f"runtime schema {name!r} is not beside the runtime; looked in"
            f" {path.parent}")
    return path


def safe_path_component(value: str) -> str:
    """Reduce an identifier to characters that are safe inside a file name."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", value)


def extract_by_path(data: Any, path: str) -> Any:
    """Resolve a dot-separated path against data, unwrapping list containers automatically.

    Example:
        >>> extract_by_path({"Subagents": [{"TypeName": "torch_coder"}]}, "Subagents.TypeName")
        'torch_coder'
    """
    curr = data
    for part in path.split("."):
        if isinstance(curr, list) and curr:
            curr = curr[0]
        if isinstance(curr, dict):
            curr = curr.get(part)
        else:
            return None
        if curr is None:
            return None
    return curr


def extract_first_field(
    data: Any,
    candidate_spec: str | list[str],
    container: dict[str, Any] | None = None,
) -> Any:
    """Extract first matching field from data or optional container using dot-path traversal.

    Array-valued fields are unwrapped to their first element: several harnesses
    report a scalar concept as a one-element list (Cursor's `workspace_roots`,
    Antigravity's `workspacePaths`). An *empty* array carries no value at all,
    so it is treated as absent rather than returned as a list -- otherwise the
    caller receives `[]` where every candidate promised a scalar.

    Args:
        data: Primary dictionary or object to search.
        candidate_spec: Field name or list of field names in priority order.
        container: Optional secondary container to search if not found in data.

    Returns:
        The extracted value, or None if no candidate matches.
    """
    candidates = candidate_spec if isinstance(candidate_spec,
                                              list) else [candidate_spec]
    for field in candidates:
        val = extract_by_path(data, field)
        if val is None and container:
            val = extract_by_path(container, field)
        if val is None:
            continue
        if isinstance(val, list):
            if not val:
                continue
            val = val[0]
        return val
    return None


def utc_now_iso() -> str:
    """Return the current UTC time as a Zulu-suffixed ISO-8601 timestamp."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z")


# The packages the TPU target needs; the compiler ships it with the runtime.
TARGET_REQUIREMENTS = "requirements-tpu.txt"
# The packages the agents' own machine needs; checked at install and at
# every session start, so the compiler ships it with the runtime too.
HOST_REQUIREMENTS = "requirements-client.txt"

# =====================================================================
# JSON Schema Validator (Deterministic Subset)
# =====================================================================


class JsonSchemaValidator:
    """Validates data structures against the deterministic JSON Schema subset."""

    @staticmethod
    def is_type(instance: Any, schema_type: str) -> bool:
        """Check whether instance conforms to JSON schema type string."""
        if schema_type == "null":
            return instance is None
        elif schema_type == "boolean":
            return isinstance(instance, bool)
        elif schema_type == "integer":
            return isinstance(instance, int) and not isinstance(instance, bool)
        elif schema_type == "number":
            return isinstance(instance,
                              (int, float)) and not isinstance(instance, bool)
        elif schema_type == "string":
            return isinstance(instance, str)
        elif schema_type == "array":
            return isinstance(instance, list)
        elif schema_type == "object":
            return isinstance(instance, dict)
        return False

    @classmethod
    def _validate_status_requires(cls, schema: dict[str, Any],
                                  properties: dict[str, Any],
                                  path: str) -> list[str]:
        """Check a `status_requires` declaration names real statuses and fields.

        A rule under a status the schema cannot emit never fires, and neither
        does one naming a property that does not exist -- the schema keeps
        validating, and the stage it was meant to hold to account goes
        unchecked until someone reads the JSON. Both are silent, and both are
        caught here, at build time, which is the advantage of declaring the
        rule in the schema over keeping a table in the framework.
        """
        if "status_requires" not in schema:
            return []

        declared = schema["status_requires"]
        if not isinstance(declared, dict):
            return [
                f"{path}/status_requires: expected an object keyed by status"
            ]

        errors: list[str] = []
        known_statuses = properties.get("status", {})
        known_statuses = (known_statuses.get("enum") if isinstance(
            known_statuses, dict) else None)

        for status, fields in declared.items():
            where = f"{path}/status_requires/{status}"
            if not isinstance(fields, list) or not all(
                    isinstance(field, str) for field in fields):
                errors.append(f"{where}: expected an array of strings")
                continue
            if isinstance(known_statuses,
                          list) and status not in known_statuses:
                errors.append(
                    f"{where}: not a value of this schema's status enum"
                    f" ({', '.join(map(str, known_statuses))})")
            unknown = [field for field in fields if field not in properties]
            if unknown:
                errors.append(f"{where}: names undeclared propert"
                              f"{'y' if len(unknown) == 1 else 'ies'}"
                              f" {', '.join(sorted(unknown))}")
        return errors

    @classmethod
    def validate_schema_definition(cls,
                                   schema: Any,
                                   path: str = "$") -> list[str]:
        """Validate schema structure and keyword conformance."""
        if not isinstance(schema, dict):
            return [f"{path}: schema must be an object"]

        errors: list[str] = []
        unknown = set(schema) - JSON_SCHEMA_KEYWORDS
        if unknown:
            errors.append(
                f"{path}: unsupported schema keyword(s): {', '.join(sorted(unknown))}"
            )

        if "type" in schema:
            schema_type = schema["type"]
            if isinstance(schema_type, list):
                if not schema_type:
                    errors.append(f"{path}/type: type array must not be empty")
                for item in schema_type:
                    if not isinstance(item, str) or item not in SCHEMA_TYPES:
                        errors.append(
                            f"{path}/type: unsupported type {item!r}")
                if len(set(map(repr, schema_type))) != len(schema_type):
                    errors.append(
                        f"{path}/type: type array must contain unique values")
            elif not isinstance(schema_type,
                                str) or schema_type not in SCHEMA_TYPES:
                errors.append(f"{path}/type: unsupported type {schema_type!r}")

        if "required" in schema and not isinstance(schema["required"], list):
            errors.append(f"{path}/required: expected an array")

        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            errors.append(f"{path}/properties: expected an object")
        else:
            errors.extend(
                cls._validate_status_requires(schema, properties, path))
            for key, child in properties.items():
                errors.extend(
                    cls.validate_schema_definition(child,
                                                   f"{path}/properties/{key}"))

        if "additionalProperties" in schema:
            additional_props = schema["additionalProperties"]
            if isinstance(additional_props, dict):
                errors.extend(
                    cls.validate_schema_definition(
                        additional_props, f"{path}/additionalProperties"))
            elif not isinstance(additional_props, bool):
                errors.append(
                    f"{path}/additionalProperties: expected a boolean or schema object"
                )

        if "items" in schema:
            errors.extend(
                cls.validate_schema_definition(schema["items"],
                                               f"{path}/items"))

        if "oneOf" in schema:
            alternatives = schema["oneOf"]
            if not isinstance(alternatives, list) or not alternatives:
                errors.append(
                    f"{path}/oneOf: expected a non-empty array of schemas")
            else:
                for index, alternative in enumerate(alternatives):
                    errors.extend(
                        cls.validate_schema_definition(
                            alternative, f"{path}/oneOf/{index}"))

        if "dependentRequired" in schema:
            dependents = schema["dependentRequired"]
            if not isinstance(dependents, dict):
                errors.append(f"{path}/dependentRequired: expected an object"
                              " keyed by property name")
            else:
                for key, fields in dependents.items():
                    if not isinstance(fields, list) or not all(
                            isinstance(field, str) for field in fields):
                        errors.append(f"{path}/dependentRequired/{key}:"
                                      " expected an array of strings")

        return errors

    @classmethod
    def validate_instance(
        cls,
        instance: Any,
        schema: dict[str, Any],
        path: str = "$",
        allow_additional_properties: bool = False,
    ) -> list[str]:
        """Validate a data instance against a valid JSON Schema definition."""
        definition_errors = cls.validate_schema_definition(schema, path)
        if definition_errors:
            return definition_errors

        errors: list[str] = []
        expected = schema.get("type")

        if expected:
            # The schema definition is already validated, so every entry is a
            # supported type string.
            expected_types = expected if isinstance(expected,
                                                    list) else [expected]
            if not any(cls.is_type(instance, t) for t in expected_types):
                expected_desc = " or ".join(expected_types)
                return [
                    f"{path}: expected {expected_desc}, got {type(instance).__name__}"
                ]

        if "const" in schema and instance != schema["const"]:
            errors.append(f"{path}: expected constant {schema['const']!r}")

        if "enum" in schema and instance not in schema["enum"]:
            errors.append(f"{path}: value is not one of {schema['enum']!r}")

        if isinstance(instance, str):
            if len(instance) < schema.get("minLength", 0):
                errors.append(f"{path}: string is too short")
            if "pattern" in schema and not re.search(schema["pattern"],
                                                     instance):
                errors.append(
                    f"{path}: string does not match {schema['pattern']!r}")

        if isinstance(instance,
                      (int, float)) and not isinstance(instance, bool):
            if "minimum" in schema and instance < schema["minimum"]:
                errors.append(f"{path}: value is below minimum")
            if "maximum" in schema and instance > schema["maximum"]:
                errors.append(f"{path}: value is above maximum")

        if isinstance(instance, dict):
            properties = schema.get("properties", {})
            for key in schema.get("required", []):
                if key not in instance:
                    errors.append(
                        f"{path}/{key}: required property is missing")

            for key, fields in schema.get("dependentRequired", {}).items():
                if key not in instance:
                    continue
                for field in fields:
                    if field not in instance:
                        errors.append(f"{path}/{field}: required property is"
                                      f" missing when {key} is present")

            additional_props = schema.get("additionalProperties")

            # Validate typed additionalProperties if specified as a schema object (e.g. env vars)
            if isinstance(additional_props, dict):
                for key in instance.keys() - properties.keys():
                    errors.extend(
                        cls.validate_instance(
                            instance[key],
                            additional_props,
                            f"{path}/{key}",
                            allow_additional_properties=
                            allow_additional_properties,
                        ))
            else:
                # Extra properties are only allowed if the schema marked additionalProperties: true AND CLI flag is passed
                allow_extra = allow_additional_properties and (additional_props
                                                               is True)
                if not allow_extra and properties:
                    for key in instance.keys() - properties.keys():
                        errors.append(
                            f"{path}/{key}: additional property is not allowed"
                        )

            for key, child in properties.items():
                if key in instance:
                    errors.extend(
                        cls.validate_instance(
                            instance[key],
                            child,
                            f"{path}/{key}",
                            allow_additional_properties=
                            allow_additional_properties,
                        ))

        if isinstance(instance, list) and "items" in schema:
            for index, item in enumerate(instance):
                errors.extend(
                    cls.validate_instance(
                        item,
                        schema["items"],
                        f"{path}/{index}",
                        allow_additional_properties=allow_additional_properties,
                    ))

        if "oneOf" in schema:
            alternatives = schema["oneOf"]
            matched = sum(
                1 for alternative in alternatives if not cls.validate_instance(
                    instance,
                    alternative,
                    path,
                    allow_additional_properties=allow_additional_properties,
                ))
            if matched != 1:
                # When every alternative is a bare `required` list, name the
                # properties: that is what a caller reading a denial needs.
                hint = ""
                if all(
                        set(alternative) <= {"required", "description"}
                        and isinstance(alternative.get("required"), list)
                        for alternative in alternatives):
                    hint = " (one of: " + " | ".join(
                        ", ".join(alternative["required"])
                        for alternative in alternatives) + ")"
                errors.append(
                    f"{path}: must match exactly one of {len(alternatives)}"
                    f" oneOf alternatives, matched {matched}{hint}")

        return errors


# Backward-compatible function aliases
validate_schema_definition = JsonSchemaValidator.validate_schema_definition
validate_instance = JsonSchemaValidator.validate_instance


def load_schema_example(
    schema_or_dir: dict[str, Any] | Path | str | None,
    schema_path: str = "",
) -> Any | None:
    """Extract first example from a schema dict, file path, or (package_dir, rel_path)."""
    if isinstance(schema_or_dir, dict):
        examples = schema_or_dir.get("examples")
        if isinstance(examples, list) and examples:
            return examples[0]
        return None

    if not schema_or_dir:
        return None

    full_path = (Path(schema_or_dir) /
                 schema_path) if schema_path else Path(schema_or_dir)
    if full_path.is_file():
        try:
            schema_data = jsonio.load_json(full_path)
            if isinstance(schema_data, dict):
                examples = schema_data.get("examples")
                if isinstance(examples, list) and examples:
                    return examples[0]
        except Exception:
            return None
    return None


# =====================================================================
# Configuration Managers: JSON & TOML
# =====================================================================


class JsonConfigManager:
    """Manages deep merges and rollbacks for structured JSON configurations."""

    @staticmethod
    def get_at_path(document: Any, keys: Sequence[str]) -> tuple[bool, Any]:
        """Navigate nested keys in a dictionary. Returns (exists, value)."""
        current = document
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return False, None
            current = current[key]
        return True, current

    @classmethod
    def remove_added(
        cls,
        document: dict[str, Any],
        keys: Sequence[str],
        expected_val: Any,
        is_container: bool = False,
    ) -> bool:
        """Undo one recorded merge action in `document`, in place.

        The inverse of one `merge` record: removes `keys[-1]` only if it still
        holds what the installer put there (`expected_val`), or, for a
        container record, if nothing is left inside it. A value the user has
        since changed is left alone. Returns whether anything was removed.
        """
        if not keys:
            return False
        exists, value = cls.get_at_path(document, keys)
        if not exists:
            return False
        if not ((is_container and not value) or
                (not is_container and value == expected_val)):
            return False
        # get_at_path succeeded along `keys`, so every prefix is a dict that
        # holds the next key.
        parent: Any = document
        for key in keys[:-1]:
            parent = parent[key]
        del parent[keys[-1]]
        return True

    @classmethod
    def merge(
        cls,
        current: dict[str, Any],
        desired: dict[str, Any],
        key_path: list[str],
        records: list[dict[str, Any]],
    ) -> None:
        """Deep merge desired dictionary into current, recording added paths.

    Raises:
      TpuAgentSuiteError: If an existing setting conflicts with the new value.
    """
        for key, value in desired.items():
            path = key_path + [key]
            if key not in current:
                if isinstance(value, dict):
                    current[key] = {}
                    records.append({"kind": "json_container", "keys": path})
                    cls.merge(current[key], value, path, records)
                else:
                    current[key] = value
                    records.append({
                        "kind": "json_value",
                        "keys": path,
                        "value": value
                    })
            elif isinstance(current[key], dict) and isinstance(value, dict):
                cls.merge(current[key], value, path, records)
            elif current[key] != value:
                raise TpuAgentSuiteError(
                    f"refusing to replace existing JSON setting: {'/'.join(path)}"
                )


class TomlConfigManager:
    """Lightweight comment-preserving TOML section and value manipulator."""

    @staticmethod
    def render_value(value: Any) -> str:
        """Render a Python primitive into TOML string representation."""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        return json.dumps(value)

    @classmethod
    def find_key(cls, lines: list[str], section: str,
                 key: str) -> tuple[int | None, str | None]:
        """Locate key within a section. Returns (line_index, raw_value_str)."""
        active_section = ""
        for index, line in enumerate(lines):
            section_match = re.match(r"^\s*\[([^]]+)]\s*$", line)
            if section_match:
                active_section = section_match.group(1)
                continue
            key_match = re.match(rf"^\s*{re.escape(key)}\s*=\s*(.*?)\s*$",
                                 line)
            if active_section == section and key_match:
                return index, key_match.group(1)
        return None, None

    @classmethod
    def apply_set(cls, path: Path, section: str, key: str,
                  value: Any) -> dict[str, Any] | None:
        """Apply a setting under [section] key = value.

    Returns ownership action record if modified.

    Raises:
      TpuAgentSuiteError: If an existing setting conflicts with the new value.
    """
        lines = (path.read_text(
            encoding="utf-8").splitlines() if path.exists() else [])
        rendered = cls.render_value(value)
        index, existing = cls.find_key(lines, section, key)

        if index is not None:
            if existing != rendered:
                raise TpuAgentSuiteError(
                    f"refusing to replace existing TOML setting: [{section}] {key}"
                )
            return None

        section_index = next(
            (i for i, line in enumerate(lines)
             if line.strip() == f"[{section}]"),
            None,
        )
        section_created = section_index is None

        if section_created:
            if lines and lines[-1].strip():
                lines.append("")
            lines.extend([f"[{section}]", f"{key} = {rendered}"])
        else:
            lines.insert(section_index + 1, f"{key} = {rendered}")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {
            "kind": "toml_value",
            "section": section,
            "key": key,
            "value": rendered,
            "section_created": section_created,
        }


class MarkdownBlockManager:
    """Manages marker-delimited managed blocks inside workspace memory files.

  A managed block is package-owned content bounded by HTML comment markers, so
  it stays invisible when rendered, is unambiguous to locate, and can be
  replaced or removed without disturbing the user's own content in the file.
  """

    @staticmethod
    def markers(marker: str) -> tuple[str, str]:
        """Return the begin and end marker comment lines for a managed block."""
        return (f"<!-- {marker}:begin -->", f"<!-- {marker}:end -->")

    @classmethod
    def locate(cls, text: str, marker: str) -> tuple[int, int] | None:
        """Locate a managed block, returning (start, end) line indices or None.

    Raises:
      TpuAgentSuiteError: If the markers are unbalanced, inverted, or duplicated.
    """
        begin, end = cls.markers(marker)
        lines = text.splitlines()
        starts = [i for i, line in enumerate(lines) if line.strip() == begin]
        ends = [i for i, line in enumerate(lines) if line.strip() == end]

        if not starts and not ends:
            return None
        if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
            raise TpuAgentSuiteError(
                f"malformed managed block markers for '{marker}': expected one"
                " begin marker followed by one end marker")
        return starts[0], ends[0]

    @classmethod
    def extract(cls, text: str, marker: str) -> str | None:
        """Return the body between a managed block's markers, or None if absent."""
        span = cls.locate(text, marker)
        if span is None:
            return None
        lines = text.splitlines()
        return "\n".join(lines[span[0] + 1:span[1]]).strip()

    @classmethod
    def apply(cls, path: Path, marker: str, body: str) -> dict[str, Any]:
        """Insert or replace a managed block, returning its ownership record."""
        begin, end = cls.markers(marker)
        block = [begin, *body.strip().splitlines(), end]
        file_created = not path.is_file()
        text = "" if file_created else path.read_text(encoding="utf-8")
        span = cls.locate(text, marker)

        if span is None:
            lines = text.splitlines()
            if lines and lines[-1].strip():
                lines.append("")
            lines.extend(block)
        else:
            lines = text.splitlines()
            lines[span[0]:span[1] + 1] = block

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {
            "kind": "markdown_block",
            "marker": marker,
            "sha256": digest_text(body.strip()),
            "file_created": file_created,
        }

    @classmethod
    def strip(cls,
              path: Path,
              marker: str,
              expected_digest: str = "",
              file_created: bool = False) -> None:
        """Remove a managed block, deleting the file if the install created it.

    A block whose body no longer matches expected_digest was edited by hand and
    is left in place, so local edits are never silently discarded.
    """
        if not path.is_file():
            return
        text = path.read_text(encoding="utf-8")
        try:
            span = cls.locate(text, marker)
        except TpuAgentSuiteError:
            return
        if span is None:
            return
        body = "\n".join(text.splitlines()[span[0] + 1:span[1]]).strip()
        if expected_digest and digest_text(body) != expected_digest:
            return

        lines = text.splitlines()
        del lines[span[0]:span[1] + 1]
        while lines and not lines[-1].strip():
            lines.pop()

        if file_created and not lines:
            path.unlink(missing_ok=True)
            return
        path.write_text("\n".join(lines) + "\n" if lines else "",
                        encoding="utf-8")
