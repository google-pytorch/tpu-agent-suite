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
"""Strict Pydantic v2 schema definitions for evaluation suites.

Defines data models for evaluating AI agents:
  - environment: Base container, packages, and workspace files.
  - evaluation: Judge model, timeouts, lifecycle hooks, verifiers, and cases.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal, TextIO, TypeVar

import pydantic
import yaml

from eval.utils import find_repo_root

from . import gcs_utils

_REPO_ROOT = find_repo_root()

DEFAULT_BASE_IMAGE = (
    "us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/nexus-host:latest")

RewardAggregation = Literal[
    "weighted_mean",
    "weighted-mean",
    "weighted_sum",
    "weighted-sum",
    "all_pass",
    "all-pass",
    "any_pass",
    "any-pass",
]

# Matches double-brace template placeholders `{{var_name}}` (with optional
# inner whitespace). Any placeholder whose key is not in `prompt_vars` is left
# untouched; a `prompt_vars` key that no placeholder uses is rejected.
_PROMPT_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def resolve_path(
    path_str: str,
    source_root: Path | None = None,
    cache_root: Path | None = None,
    repo_root: Path | None = None,
) -> Path | None:
    """Resolves file paths relative to suite directory, repo root (//), cwd, absolute paths, or GCS URIs."""
    if not path_str or not path_str.strip():
        return None
    clean_path = path_str.strip()
    if clean_path.startswith("gs://"):
        return gcs_utils.fetch_gcs_asset(clean_path, cache_root=cache_root)
    eff_repo_root = (repo_root or _REPO_ROOT).resolve()
    if clean_path.startswith("//"):
        rel = clean_path.removeprefix("//").lstrip("/")
        if not rel:
            return None
        cand = (eff_repo_root / rel).resolve()
        if not cand.is_relative_to(eff_repo_root):
            return None
        return cand if cand.exists() else None
    p = Path(clean_path)
    if p.is_absolute():
        return p.resolve() if p.exists() else None
    search_roots = ((source_root,
                     eff_repo_root) if source_root is not None else
                    (eff_repo_root, Path.cwd()))
    for root in search_roots:
        if root:
            cand = (root / p).resolve()
            if cand.exists():
                return cand
    return None


def _clean_str_list(v: Any, *, allow_none: bool = False) -> Any:
    """Normalizes string-or-list inputs, stripping and dropping blank strings."""
    if v is None:
        return None if allow_none else []
    if isinstance(v, str):
        stripped = v.strip()
        return [stripped] if stripped else []
    if isinstance(v, (list, tuple)):
        return [
            item.strip() if isinstance(item, str) else item for item in v
            if not isinstance(item, str) or item.strip()
        ]
    return v


def _none_to_list(v: Any) -> Any:
    return [] if v is None else v


def _none_to_dict(v: Any) -> Any:
    return {} if v is None else v


_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")

NonNullList = Annotated[list[_T], pydantic.BeforeValidator(_none_to_list)]
NonNullDict = Annotated[dict[_K, _V], pydantic.BeforeValidator(_none_to_dict)]
CleanList = Annotated[list[_T], pydantic.BeforeValidator(_clean_str_list)]


class StrictModel(pydantic.BaseModel):
    """Base class enforcing strict validation and forbidding unknown fields."""

    model_config = pydantic.ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )


class FileStaging(StrictModel):
    """File or directory to stage into the container workspace or test directory."""

    src: str = pydantic.Field(
        ...,
        description="Host source path (relative to suite YAML or absolute).",
    )
    dest: str = pydantic.Field(
        default=".",
        description=("Destination path relative to target dir (default: '.' or"
                     " source filename)."),
    )

    @classmethod
    def coerce(cls, value: str | FileStaging | dict[str, Any]) -> FileStaging:
        """Normalizes string shorthand, dict, or instance into a FileStaging."""
        if isinstance(value, FileStaging):
            return value
        if isinstance(value, str):
            if "://" in value:
                scheme, rest = value.split("://", 1)
                if ":" in rest:
                    src_rest, dest = rest.split(":", 1)
                    return cls(
                        src=f"{scheme}://{src_rest.strip()}",
                        dest=dest.strip(),
                    )
                fallback_dest = Path(rest.strip("/")).name or "."
                return cls(src=value.strip(), dest=fallback_dest)
            if ":" in value:
                src, dest = value.split(":", 1)
                return cls(src=src.strip(), dest=dest.strip())
            return cls(src=value, dest=value)
        if isinstance(value, dict):
            return cls.model_validate(value)
        raise TypeError(f"Cannot coerce {type(value)} to FileStaging")


class ScriptCheck(StrictModel):
    """Configuration for a verifier (.py script or .toml judge)."""

    script: str | list[str] = pydantic.Field(
        ...,
        validation_alias=pydantic.AliasChoices("files", "script"),
        description=(
            "Path or list of paths to verifier scripts or judge.toml rubrics "
            "(relative or absolute)."),
    )
    deps: CleanList[str] = pydantic.Field(
        default_factory=list,
        description=(
            "Optional helper modules or support files staged into the verifier "
            "group directory and added to PYTHONPATH."),
    )
    args: NonNullList[str] = pydantic.Field(
        default_factory=list,
        description=(
            "Deprecated: prefer verifier_params. Command-line arguments "
            "passed to the verifier script."),
    )
    weight: float = pydantic.Field(
        default=1.0,
        validation_alias=pydantic.AliasChoices("reward_weight", "weight"),
        allow_inf_nan=False,
        description="Relative scoring weight in reward aggregation.",
    )
    name: str | None = pydantic.Field(
        default=None,
        description="Optional custom folder name under tests/.",
    )
    timeout_sec: float | None = pydantic.Field(
        default=None,
        gt=0.0,
        description="Optional execution timeout in seconds.",
    )
    verifier_params: NonNullDict[str, Any] = pydantic.Field(
        default_factory=dict,
        description=(
            "Key-value parameters to be written to tests/verifier_params.json."
        ),
    )

    @pydantic.field_validator("script", mode="before")
    @classmethod
    def _validate_script_paths(cls, v: Any) -> str | list[str]:
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                raise ValueError("Verifier path must not be empty")
            return stripped
        if isinstance(v, (list, tuple)):
            items = [str(item).strip() for item in v if str(item).strip()]
            if not items:
                raise ValueError("Verifier path list must not be empty")
            return items
        return v

    @property
    def scripts(self) -> list[str]:
        """Returns verifier source path(s) normalized as a list of strings."""
        return ([self.script]
                if isinstance(self.script, str) else list(self.script))

    @property
    def primary_script(self) -> str:
        """Returns the first verifier source path."""
        return self.scripts[0]

    @classmethod
    def coerce(cls, value: str | ScriptCheck | dict[str, Any]) -> ScriptCheck:
        """Normalizes string shorthand, dict, or instance into a ScriptCheck instance."""
        if isinstance(value, ScriptCheck):
            return value
        if isinstance(value, str):
            return cls(script=value.strip())
        if isinstance(value, dict):
            return cls.model_validate(value)
        raise TypeError(f"Cannot coerce {type(value)} to ScriptCheck")


class ExpectationCriterion(StrictModel):
    """Detailed configuration for an LLM judge rubric criterion."""

    criterion: str = pydantic.Field(
        ...,
        validation_alias=pydantic.AliasChoices("criterion", "description"),
        description="Natural language evaluation criterion for LLM judge.",
    )
    weight: float = pydantic.Field(
        default=1.0,
        validation_alias=pydantic.AliasChoices("reward_weight", "weight"),
        ge=0.0,
        allow_inf_nan=False,
        description="Relative scoring weight in reward aggregation.",
    )
    type: Literal["binary", "numeric", "likert"] = pydantic.Field(
        default="binary",
        description="Evaluation scale type ('binary', 'numeric', or 'likert').",
    )

    @classmethod
    def coerce(
        cls, value: str | ExpectationCriterion | dict[str, Any]
    ) -> ExpectationCriterion:
        """Normalizes string shorthand, dict, or instance into an ExpectationCriterion."""
        if isinstance(value, ExpectationCriterion):
            return value
        if isinstance(value, str):
            return cls(criterion=value)
        if isinstance(value, dict):
            return cls.model_validate(value)
        raise TypeError(f"Cannot coerce {type(value)} to ExpectationCriterion")


class CaseEnvironmentOverride(StrictModel):
    """Optional case-level overrides for container image, packages, and resources."""

    base_image: str | None = pydantic.Field(
        default=None,
        description="Per-case base Docker image override.",
    )
    apt_packages: list[str] | None = pydantic.Field(
        default=None,
        description=(
            "Per-case apt packages override (replaces suite apt_packages when "
            "not None)."),
    )
    pip_packages: list[str] | None = pydantic.Field(
        default=None,
        description=(
            "Per-case pip packages override (replaces suite pip_packages when "
            "not None)."),
    )
    setup_scripts: list[str] | None = pydantic.Field(
        default=None,
        description=(
            "Per-case container setup scripts override (replaces suite "
            "setup_scripts when not None)."),
    )
    cpus: int | None = pydantic.Field(
        default=None,
        gt=0,
        description="Per-case CPU core limit override.",
    )
    memory_mb: int | None = pydantic.Field(
        default=None,
        gt=0,
        description="Per-case memory allocation override in MB.",
    )
    storage_mb: int | None = pydantic.Field(
        default=None,
        gt=0,
        description="Per-case storage allocation override in MB.",
    )
    build_timeout_sec: float | None = pydantic.Field(
        default=None,
        gt=0.0,
        description="Per-case container build timeout override in seconds.",
    )
    workdir: str | None = pydantic.Field(
        default=None,
        description="Per-case container working directory override.",
    )
    env_vars: dict[str, str] | None = pydantic.Field(
        default=None,
        description=(
            "Per-case environment variables merged onto suite env_vars, "
            "written to task.toml's [environment.env]."),
    )

    @pydantic.field_validator("base_image", "workdir", mode="before")
    @classmethod
    def _validate_optional_nonempty_str(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                raise ValueError("Value must not be empty")
            return stripped
        return v

    @pydantic.field_validator(
        "apt_packages",
        "pip_packages",
        "setup_scripts",
        mode="before",
    )
    @classmethod
    def _coerce_optional_pkg_list(cls, v: Any) -> Any:
        return _clean_str_list(v, allow_none=True)


class TestCase(StrictModel):
    """Individual test case in an evaluation suite."""

    name: str = pydantic.Field(
        ...,
        min_length=1,
        description="Unique identifier for the test case.",
    )
    description: str | None = pydantic.Field(
        default=None,
        description=(
            "Optional task description written to task.toml [task]. Defaults "
            "to the first non-empty line of the resolved prompt."),
    )
    prompt: str | None = pydantic.Field(
        default=None,
        description=
        "Instruction prompt provided to the agent in instruction.md.",
    )
    prompt_file: str | None = pydantic.Field(
        default=None,
        description=
        "Path to markdown/text file containing prompt (e.g. TASK.md).",
    )
    prompt_vars: NonNullDict[str, Any] = pydantic.Field(
        default_factory=dict,
        description=(
            "Template variables substituted into {{var}} placeholders in prompt"
            " or prompt_file when present. Placeholders not in prompt_vars are"
            " left untouched. A key that no placeholder uses is an error."),
    )
    environment: CaseEnvironmentOverride | None = pydantic.Field(
        default=None,
        description="Optional per-case container/resource overrides.",
    )
    workspace_files: NonNullList[str | FileStaging] = pydantic.Field(
        default_factory=list,
        description=
        "Case-specific files or directories to stage into workspace.",
    )
    verifier_files: NonNullList[str | FileStaging] = pydantic.Field(
        default_factory=list,
        validation_alias=pydantic.AliasChoices("verifier_files", "test_files"),
        description=(
            "Case-specific static files or directories to stage into tests/ "
            "after agent execution (e.g., golden answers or schemas)."),
    )
    expectations: NonNullList[str | ExpectationCriterion] = pydantic.Field(
        default_factory=list,
        description=(
            "List of LLM judge criteria (strings or ExpectationCriterion"
            " objects)."),
    )
    script_verifiers: NonNullList[str | ScriptCheck] = pydantic.Field(
        default_factory=list,
        validation_alias=pydantic.AliasChoices("verifiers",
                                               "script_verifiers"),
        description=(
            "Case-specific verifiers (.py scripts or .toml rubrics)."),
    )
    timeout_sec: float | None = pydantic.Field(
        default=None,
        gt=0.0,
        description=("Execution timeout for the agent in seconds (inherits "
                     "evaluation.timeout_sec when None)."),
    )
    verifier_timeout_sec: float | None = pydantic.Field(
        default=None,
        gt=0.0,
        description=("Execution timeout for the verifier in seconds (inherits "
                     "evaluation.verifier_timeout_sec when None)."),
    )
    verifier_env: NonNullDict[str, str] = pydantic.Field(
        default_factory=dict,
        description=(
            "Case-level verifier environment variables merged onto "
            "evaluation.verifier_env, written to task.toml's [verifier.env]."),
    )
    metadata: dict[str, Any] | None = pydantic.Field(
        default=None,
        description="Optional metadata dictionary merged into [metadata].",
    )
    verifier_params: NonNullDict[str, Any] = pydantic.Field(
        default_factory=dict,
        description=(
            "Case-level parameter overrides for tests/verifier_params.json."),
    )
    task_name: str | None = pydantic.Field(
        default=None,
        description="Optional explicit full task name override.",
    )
    pre_verify: list[str] | None = pydantic.Field(
        default=None,
        description=(
            "Case-level pre-verification scripts executed before RewardKit "
            "(overrides evaluation.pre_verify when not None)."),
    )
    post_verify: list[str] | None = pydantic.Field(
        default=None,
        description=(
            "Case-level post-verification scripts executed after RewardKit "
            "(overrides evaluation.post_verify when not None)."),
    )
    reward_aggregation: RewardAggregation | None = pydantic.Field(
        default=None,
        description=(
            "Case-level RewardKit aggregation strategy override for generated "
            "tests/reward.toml."),
    )

    @pydantic.model_validator(mode="after")
    def _validate_prompt_present(self) -> TestCase:
        has_prompt = bool(self.prompt and self.prompt.strip())
        has_prompt_file = bool(self.prompt_file and self.prompt_file.strip())
        if not has_prompt and not has_prompt_file:
            raise ValueError(
                f"TestCase '{self.name}' must specify either 'prompt' or 'prompt_file'"
            )
        return self

    def resolve_prompt(self, source_root: Path | None = None) -> str:
        """Resolves instruction prompt text and applies prompt_vars if set."""
        if self.prompt and self.prompt.strip():
            raw = self.prompt.strip()
        elif self.prompt_file:
            p = resolve_path(self.prompt_file, source_root=source_root)
            if p is None or not p.is_file():
                raise FileNotFoundError(
                    f"prompt_file not found: {self.prompt_file}")
            raw = p.read_text(encoding="utf-8").rstrip()
        else:
            raise ValueError(
                f"No prompt source found for test case '{self.name}'")

        return self.render_prompt(raw)

    def render_prompt(self, raw: str) -> str:
        """Substitutes prompt_vars into the {{var}} placeholders of raw text.

        A placeholder with no matching key is left untouched, so a prompt can
        carry literal `{{text}}`. A key that no placeholder uses is an error.

        Raises:
            ValueError: If prompt_vars holds a key the text never references.
        """
        if not self.prompt_vars:
            return raw

        used: set[str] = set()

        def _substitute(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in self.prompt_vars:
                return match.group(0)
            used.add(key)
            return str(self.prompt_vars[key])

        rendered = _PROMPT_VAR_RE.sub(_substitute, raw)
        unused = sorted(set(self.prompt_vars) - used)
        if unused:
            raise ValueError(
                f"TestCase '{self.name}': prompt_vars keys not used by the "
                f"prompt: {unused}")
        return rendered

    def normalized_expectations(self) -> list[ExpectationCriterion]:
        """Returns all expectations coerced to ExpectationCriterion objects."""
        return [ExpectationCriterion.coerce(e) for e in self.expectations]

    def normalized_script_verifiers(self) -> list[ScriptCheck]:
        """Returns all verifiers coerced to ScriptCheck objects."""
        return [ScriptCheck.coerce(sv) for sv in self.script_verifiers]

    def normalized_workspace_files(self) -> list[FileStaging]:
        """Returns all workspace files coerced to FileStaging objects."""
        return [FileStaging.coerce(f) for f in self.workspace_files]

    def normalized_verifier_files(self) -> list[FileStaging]:
        """Returns all verifier files coerced to FileStaging objects."""
        return [FileStaging.coerce(f) for f in self.verifier_files]


class EnvironmentConfig(StrictModel):
    """Environment and container configuration."""

    base_image: str = pydantic.Field(
        default=DEFAULT_BASE_IMAGE,
        description="Base Docker image.",
    )
    apt_packages: CleanList[str] = pydantic.Field(
        default_factory=list,
        description="System apt packages to install during container build.",
    )
    pip_packages: CleanList[str] = pydantic.Field(
        default_factory=list,
        description="Python pip packages to install during container build.",
    )
    setup_scripts: CleanList[str] = pydantic.Field(
        default_factory=list,
        description=(
            "Container setup scripts (.py or .sh) executed during Docker build "
            "and removed before the agent starts."),
    )
    workspace_files: CleanList[str | FileStaging] = pydantic.Field(
        default_factory=list,
        description=(
            "Suite-level files or directories to stage into the workspace."),
    )
    cpus: int = pydantic.Field(
        default=2,
        gt=0,
        description="CPU cores limit for task container.",
    )
    memory_mb: int = pydantic.Field(
        default=4096,
        gt=0,
        description="Memory allocation in MB.",
    )
    storage_mb: int = pydantic.Field(
        default=10240,
        gt=0,
        description="Storage allocation in MB.",
    )
    build_timeout_sec: float | None = pydantic.Field(
        default=None,
        gt=0.0,
        description="Optional container build timeout in seconds.",
    )
    workdir: str = pydantic.Field(
        default="/workspace",
        description="Working directory in container.",
    )
    env_vars: NonNullDict[str, str] = pydantic.Field(
        default_factory=dict,
        description=(
            "Environment variables set in task containers, written to "
            "task.toml's [environment.env]."),
    )

    @pydantic.field_validator("base_image", "workdir", mode="before")
    @classmethod
    def _validate_nonempty_str(cls, v: Any) -> Any:
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                raise ValueError("Value must not be empty")
            return stripped
        return v

    def normalized_workspace_files(self) -> list[FileStaging]:
        """Returns suite-level workspace files coerced to FileStaging objects."""
        return [FileStaging.coerce(f) for f in self.workspace_files]

    def merged_with(
        self,
        override: CaseEnvironmentOverride | None,
    ) -> EnvironmentConfig:
        """Returns an EnvironmentConfig with non-None case overrides applied."""
        if override is None:
            return self
        return EnvironmentConfig(
            base_image=(override.base_image if override.base_image is not None
                        else self.base_image),
            apt_packages=(list(override.apt_packages) if override.apt_packages
                          is not None else list(self.apt_packages)),
            pip_packages=(list(override.pip_packages) if override.pip_packages
                          is not None else list(self.pip_packages)),
            setup_scripts=(list(override.setup_scripts)
                           if override.setup_scripts is not None else list(
                               self.setup_scripts)),
            workspace_files=list(self.workspace_files),
            cpus=override.cpus if override.cpus is not None else self.cpus,
            memory_mb=(override.memory_mb
                       if override.memory_mb is not None else self.memory_mb),
            storage_mb=(override.storage_mb if override.storage_mb is not None
                        else self.storage_mb),
            build_timeout_sec=(override.build_timeout_sec
                               if override.build_timeout_sec is not None else
                               self.build_timeout_sec),
            workdir=(override.workdir
                     if override.workdir is not None else self.workdir),
            env_vars=({
                **self.env_vars,
                **override.env_vars,
            } if override.env_vars is not None else dict(self.env_vars)),
        )


class EvaluationConfig(StrictModel):
    """Evaluation configuration, suite-wide defaults, and test cases."""

    judge_model: str = pydantic.Field(
        default="gemini/gemini-3.6-flash",
        description="Default LLM judge model identifier.",
    )
    timeout_sec: float = pydantic.Field(
        default=300.0,
        gt=0.0,
        description="Default agent execution timeout in seconds.",
    )
    verifier_timeout_sec: float = pydantic.Field(
        default=300.0,
        gt=0.0,
        description="Default verifier execution timeout in seconds.",
    )
    verifier_env: NonNullDict[str, str] = pydantic.Field(
        default_factory=dict,
        description=("Suite-level verifier environment variables written to "
                     "task.toml's [verifier.env]."),
    )
    pre_verify: NonNullList[str] = pydantic.Field(
        default_factory=list,
        description=(
            "Suite-level scripts executed before RewardKit in test.sh."),
    )
    post_verify: NonNullList[str] = pydantic.Field(
        default_factory=list,
        description=(
            "Suite-level scripts executed after RewardKit in test.sh."),
    )
    reward_aggregation: RewardAggregation = pydantic.Field(
        default="weighted_mean",
        description=(
            "Default RewardKit aggregation strategy for generated reward.toml."
        ),
    )
    script_verifiers: NonNullList[str | ScriptCheck] = pydantic.Field(
        default_factory=list,
        validation_alias=pydantic.AliasChoices("verifiers",
                                               "script_verifiers"),
        description=(
            "Suite-level verifiers applied across all test cases (.py scripts "
            "or .toml rubrics)."),
    )
    verifier_files: NonNullList[str | FileStaging] = pydantic.Field(
        default_factory=list,
        validation_alias=pydantic.AliasChoices("verifier_files", "test_files"),
        description=(
            "Suite-level static files or directories staged into tests/ "
            "after agent execution across all cases."),
    )
    cases: NonNullList[TestCase] = pydantic.Field(
        ...,
        description="List of test cases.",
    )

    def normalized_script_verifiers(self) -> list[ScriptCheck]:
        """Returns suite-level verifiers coerced to ScriptCheck objects."""
        return [ScriptCheck.coerce(sv) for sv in self.script_verifiers]

    def normalized_verifier_files(self) -> list[FileStaging]:
        """Returns suite-level verifier files coerced to FileStaging objects."""
        return [FileStaging.coerce(f) for f in self.verifier_files]


class EvalSuite(StrictModel):
    """Root evaluation suite document."""

    version: str | None = pydantic.Field(
        default=None,
        description="Optional legacy schema version string.",
    )
    name: str | None = pydantic.Field(
        default=None,
        description="Name of the evaluation suite.",
    )
    suite_name: str | None = pydantic.Field(
        default=None,
        description="Alias for suite name.",
    )
    environment: EnvironmentConfig = pydantic.Field(
        default_factory=EnvironmentConfig,
        description="Environment and container configuration.",
    )
    evaluation: EvaluationConfig = pydantic.Field(
        ...,
        description=
        ("Evaluation configuration including judge model, verifiers, and test"
         " cases."),
    )

    @pydantic.model_validator(mode="after")
    def _resolve_name(self) -> EvalSuite:
        if not self.name and self.suite_name:
            self.name = self.suite_name
        elif not self.suite_name and self.name:
            self.suite_name = self.name
        return self

    @classmethod
    def from_file(cls, file_path: str | Path) -> EvalSuite:
        """Loads and validates an EvalSuite from a .yaml or .yml file path."""
        p = Path(file_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Suite YAML file not found: {file_path}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected YAML mapping at root of {file_path}, got {type(data).__name__}"
            )
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, yaml_content: str | Path | TextIO) -> EvalSuite:
        """Loads and validates an EvalSuite from a YAML string, file path, or file stream."""
        if isinstance(yaml_content, Path):
            with open(yaml_content, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        elif isinstance(yaml_content, str):
            path_candidate = Path(yaml_content)
            if "\n" not in yaml_content and path_candidate.is_file():
                with open(path_candidate, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            elif "\n" not in yaml_content and (yaml_content.endswith(".yaml")
                                               or
                                               yaml_content.endswith(".yml")):
                raise FileNotFoundError(
                    f"Suite YAML file not found: {yaml_content}")
            else:
                data = yaml.safe_load(yaml_content)
        elif hasattr(yaml_content, "read"):
            data = yaml.safe_load(yaml_content.read())
        else:
            raise TypeError(
                f"Unsupported YAML content type: {type(yaml_content).__name__}"
            )

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected YAML mapping at root, got {type(data).__name__}")
        return cls.model_validate(data)


def normalize_script_verifier(
    item: str | ScriptCheck | dict[str, Any], ) -> ScriptCheck:
    """Normalizes a string path, dict, or ScriptCheck instance into a ScriptCheck."""
    return ScriptCheck.coerce(item)


def normalize_expectation(
    item: str | ExpectationCriterion | dict[str, Any],
) -> ExpectationCriterion:
    """Normalizes a string rubric, dict, or ExpectationCriterion into an ExpectationCriterion."""
    return ExpectationCriterion.coerce(item)


def normalize_file_staging(
    item: str | FileStaging | dict[str, Any], ) -> FileStaging:
    """Normalizes a string path, dict, or FileStaging instance into a FileStaging."""
    return FileStaging.coerce(item)
