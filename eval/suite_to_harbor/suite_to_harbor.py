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

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pydantic>=2.0",
#     "pyyaml>=6.0",
# ]
# ///
"""YAML to Harbor task generator and CLI."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import gcs_utils, schema, template_generators

resolve_path = schema.resolve_path


def sanitize_name(name: str) -> str:
    """Sanitizes strings for safe directory names, identifiers, and container tags."""
    clean = name.strip().lower()
    clean = re.sub(r"[^a-z0-9_.-]+", "_", clean)
    clean = re.sub(r"_+", "_", clean)
    return clean.strip("_.-") or "eval_task"


def link_or_copy(src: str | Path, dest: str | Path) -> None:
    """Hardlinks cached assets to dest or copies local repo files."""
    src_p = Path(src).resolve()
    dest_p = Path(dest)
    # Cleanup dest
    if dest_p.exists():
        if dest_p.is_dir():
            shutil.rmtree(dest_p)
        else:
            dest_p.unlink()
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    cache_root = gcs_utils.get_gcs_cache_root().resolve()
    # Hardlink cache data into generated tasks.
    if cache_root in src_p.parents:
        try:
            os.link(src_p, dest_p)
            return
        except OSError as e:
            logging.debug(
                "Could not hardlink %s to %s (%s); falling back to copy.",
                src_p,
                dest_p,
                e,
            )
    shutil.copy2(src_p, dest_p)


def copy_tree_with_patterns(src: Path, dest: Path, *ignore_patterns:
                            str) -> None:
    """Recursively copies/links src to dest using glob ignore patterns."""
    if not src.is_dir():
        return
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns(*ignore_patterns),
        copy_function=link_or_copy,
        dirs_exist_ok=True,
    )


def copy_filtered_tree(src: Path, dest: Path) -> None:
    """Recursively copies files, ignoring hidden and test files."""
    copy_tree_with_patterns(src, dest, ".*", "*_test.py", "test_*")


def stage_file(
    staging: schema.FileStaging,
    dest_dir: Path,
    source_root: Path | None = None,
) -> None:
    """Stages a source file or directory into dest_dir based on FileStaging.

    Raises:
        FileNotFoundError: If the source path does not exist.
    """
    resolved = resolve_path(staging.src, source_root=source_root)
    if not resolved:
        raise FileNotFoundError(f"File path not found: {staging.src}")

    dest_rel = staging.dest.strip()
    if dest_rel in ("/workspace", "/tests"):
        dest_rel = ""
    elif dest_rel.startswith("/workspace/"):
        dest_rel = dest_rel[len("/workspace/"):]
    elif dest_rel.startswith("workspace/"):
        dest_rel = dest_rel[len("workspace/"):]
    elif dest_rel.startswith("/tests/"):
        dest_rel = dest_rel[len("/tests/"):]
    elif dest_rel.startswith("tests/"):
        dest_rel = dest_rel[len("tests/"):]
    is_dir_dest = dest_rel.endswith("/") or dest_rel in ("", ".")
    dest_rel = dest_rel.strip("/")
    if ".." in Path(dest_rel).parts:
        raise ValueError(
            f"Destination path escapes target directory: {staging.dest!r}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()
    dest_resolved = (dest_root / dest_rel).resolve() if dest_rel else dest_root
    if not dest_resolved.is_relative_to(dest_root):
        raise ValueError(
            f"Destination path escapes target directory: {staging.dest!r}")

    if resolved.is_dir():
        target_dir = dest_dir / dest_rel if dest_rel else dest_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        copy_filtered_tree(src=resolved, dest=target_dir)
    else:
        dest_path = dest_dir / dest_rel if dest_rel else dest_dir
        if is_dir_dest or dest_path.is_dir():
            target_file = dest_path / resolved.name
        else:
            target_file = dest_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy(resolved, target_file)


def _dedup_name(stem: str, used: set[str], suffix: str = "") -> str:
    """Returns `stem + suffix` or `f"{stem}_{idx}{suffix}"` and registers it in `used`."""
    cand = f"{stem}{suffix}"
    idx = 2
    while cand in used:
        cand = f"{stem}_{idx}{suffix}"
        idx += 1
    used.add(cand)
    return cand


def stage_lifecycle_hooks(
    hooks: Sequence[str],
    tests_dir: Path,
    source_root: Path | None = None,
    reserved_names: set[str] | None = None,
) -> list[str]:
    """Stages pre_verify / post_verify scripts into tests/ and returns basenames."""
    clean_hooks = [h for h in hooks if h and h.strip()]
    if not clean_hooks:
        return []
    staged_names: list[str] = []
    used = reserved_names if reserved_names is not None else set()
    tests_dir.mkdir(parents=True, exist_ok=True)

    for hook_str in clean_hooks:
        resolved = resolve_path(hook_str, source_root=source_root)
        if not resolved or not resolved.is_file():
            raise FileNotFoundError(
                f"Lifecycle hook script not found: {hook_str}")

        cand_name = _dedup_name(resolved.stem, used, resolved.suffix)
        dest_file = tests_dir / cand_name
        link_or_copy(resolved, dest_file)
        os.chmod(dest_file, 0o755)
        staged_names.append(cand_name)

    return staged_names


def bundle_workspace_data(
    env: schema.EnvironmentConfig,
    task_dir: Path,
    source_root: Path | None = None,
    case: schema.TestCase | None = None,
) -> bool:
    """Stages workspace files into environment/data/."""
    data_dir = task_dir / "environment" / "data"

    # 1. Suite level files
    for wf in env.normalized_workspace_files():
        stage_file(wf, dest_dir=data_dir, source_root=source_root)

    # 2. Case level files
    if case is not None:
        for wf in case.normalized_workspace_files():
            stage_file(wf, dest_dir=data_dir, source_root=source_root)

    if data_dir.is_dir():
        has_files = any(f.is_file() for f in data_dir.rglob("*"))
        if not has_files:
            shutil.rmtree(data_dir, ignore_errors=True)
            return False
        return True
    return False


def bundle_setup_scripts(
    env: schema.EnvironmentConfig,
    task_dir: Path,
    source_root: Path | None = None,
) -> list[str]:
    """Stages container setup scripts into environment/setup/ and returns filenames."""
    scripts = [s for s in env.setup_scripts if s and s.strip()]
    if not scripts:
        return []

    setup_dir = task_dir / "environment" / "setup"
    setup_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    staged_names: list[str] = []

    for script_str in scripts:
        resolved = resolve_path(script_str, source_root=source_root)
        if not resolved or not resolved.is_file():
            raise FileNotFoundError(f"Setup script not found: {script_str}")

        cand_name = _dedup_name(resolved.stem, used_names, resolved.suffix)
        dest_file = setup_dir / cand_name
        link_or_copy(resolved, dest_file)
        os.chmod(dest_file, 0o755)
        staged_names.append(cand_name)

    return staged_names


def bundle_script_verifiers(
    verifiers: Sequence[Any],
    task_dir: Path,
    source_root: Path | None = None,
) -> list[tuple[str, schema.ScriptCheck]]:
    """Stages verifiers (.py scripts or .toml judges) into tests/."""
    if not verifiers:
        return []

    tests_dir = task_dir / "tests"
    used_names: set[str] = ({p.name
                             for p in tests_dir.iterdir()}
                            if tests_dir.is_dir() else set())
    bundled: list[tuple[str, schema.ScriptCheck]] = []
    for raw_v in verifiers:
        sc = schema.ScriptCheck.coerce(raw_v)
        script_list = sc.scripts

        clean_stem = sanitize_name(sc.name or Path(sc.primary_script).stem)
        folder_name = _dedup_name(clean_stem, used_names)
        bundled.append((folder_name, sc))

        check_dir = tests_dir / folder_name
        check_dir.mkdir(parents=True, exist_ok=True)
        seen_toml = False
        staged_files: set[str] = set()
        for script_item in script_list:
            resolved = resolve_path(script_item, source_root=source_root)
            if not resolved or not resolved.is_file():
                raise FileNotFoundError(
                    f"Verifier file not found: {script_item}")
            is_toml = resolved.suffix.lower() == ".toml"
            if is_toml:
                if seen_toml:
                    raise ValueError(
                        f"Verifier {folder_name!r} specifies multiple .toml "
                        "files; split them into separate verifier entries.")
                seen_toml = True
                dest_file = check_dir / "judge.toml"
            else:
                dest_file = check_dir / resolved.name

            if dest_file.name in staged_files:
                raise ValueError(
                    f"Verifier {folder_name!r} has duplicate staged filename "
                    f"{dest_file.name!r}.")
            staged_files.add(dest_file.name)

            if not is_toml and sc.args:
                logging.debug(
                    "ScriptCheck.args is deprecated for %s; prefer verifier_params.",
                    script_item,
                )
                content = resolved.read_text(encoding="utf-8")
                # Prepending an assignment pushes the file's own imports down,
                # so ruff reads every one of them as E402. The generated file
                # is machine-written and cannot be edited to satisfy that, and
                # a bundle the repo's own hooks reject is a bundle somebody has
                # to patch by hand after every regeneration.
                header = (f"_SPECIFIED_ARGS = {json.dumps(list(sc.args))}\n"
                          "# ruff: noqa: E402\n\n")
                # After the licence header *and* the module docstring, not
                # before them. Prepending at position 0 puts code above the
                # licence, which `addlicense` reports as missing, and above the
                # docstring, which stops being one -- `__doc__` goes None and
                # anything reading it breaks.
                lines = content.splitlines(keepends=True)
                at = 0
                for n, line in enumerate(lines):
                    if line.startswith("#") or not line.strip():
                        at = n + 1
                    else:
                        break
                try:
                    tree = ast.parse(content)
                    if (tree.body and isinstance(tree.body[0], ast.Expr)
                            and isinstance(tree.body[0].value, ast.Constant)
                            and isinstance(tree.body[0].value.value, str)):
                        at = max(at, tree.body[0].end_lineno)
                except SyntaxError:
                    pass
                dest_file.write_text("".join(lines[:at]) + header +
                                     "".join(lines[at:]),
                                     encoding="utf-8")
            else:
                shutil.copy2(resolved, dest_file)

        for dep_item in sc.deps:
            resolved_dep = resolve_path(dep_item, source_root=source_root)
            if not resolved_dep or not resolved_dep.is_file():
                raise FileNotFoundError(
                    f"Verifier dependency not found: {dep_item}")
            if resolved_dep.name in staged_files:
                raise ValueError(
                    f"Verifier {folder_name!r} dependency {resolved_dep.name!r} "
                    "collides with an already staged file.")
            staged_files.add(resolved_dep.name)
            shutil.copy2(resolved_dep, check_dir / resolved_dep.name)
    return bundled


def _write_task_files(
    suite_name: str,
    case: schema.TestCase,
    env: schema.EnvironmentConfig,
    eval_cfg: schema.EvaluationConfig,
    task_dir: Path,
    source_root: Path | None,
) -> None:
    """Writes instruction.md and task.toml for a compiled case."""
    prompt_text = case.resolve_prompt(source_root=source_root)
    (task_dir / "instruction.md").write_text(prompt_text, encoding="utf-8")
    (task_dir / "task.toml").write_text(
        template_generators.generate_task_toml(
            suite_name=suite_name,
            case=case,
            env=env,
            eval_cfg=eval_cfg,
            prompt_text=prompt_text,
        ),
        encoding="utf-8",
    )


def _bundle_environment(
    effective_env: schema.EnvironmentConfig,
    case: schema.TestCase,
    task_dir: Path,
    source_root: Path | None,
) -> None:
    """Stages workspace data and setup scripts, and writes environment/Dockerfile."""
    has_data = bundle_workspace_data(
        env=effective_env,
        task_dir=task_dir,
        source_root=source_root,
        case=case,
    )
    staged_setup = bundle_setup_scripts(
        env=effective_env,
        task_dir=task_dir,
        source_root=source_root,
    )
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text(
        template_generators.generate_dockerfile(
            env=effective_env,
            has_data=has_data,
            setup_scripts=staged_setup,
        ),
        encoding="utf-8",
    )


def _stage_verifier_assets(
    case: schema.TestCase,
    eval_cfg: schema.EvaluationConfig,
    task_dir: Path,
    tests_dir: Path,
    judge_model_override: str | None,
    source_root: Path | None,
) -> tuple[list[tuple[str, schema.ScriptCheck]], bool]:
    """Stages verifier files, expectation rubrics, and script/toml verifiers."""
    tests_dir.mkdir(parents=True, exist_ok=True)
    for vf in eval_cfg.normalized_verifier_files():
        stage_file(staging=vf, dest_dir=tests_dir, source_root=source_root)
    for vf in case.normalized_verifier_files():
        stage_file(staging=vf, dest_dir=tests_dir, source_root=source_root)

    judge_model = judge_model_override or eval_cfg.judge_model
    judge_toml = template_generators.generate_judge_toml(
        expectations=case.normalized_expectations(),
        judge_model=judge_model,
    )
    if judge_toml:
        expectations_dir = tests_dir / "expectations"
        expectations_dir.mkdir(parents=True, exist_ok=True)
        (expectations_dir / "judge.toml").write_text(judge_toml,
                                                     encoding="utf-8")

    all_verifiers = list(eval_cfg.normalized_script_verifiers()) + list(
        case.normalized_script_verifiers())
    bundled_verifiers = bundle_script_verifiers(
        all_verifiers,
        task_dir=task_dir,
        source_root=source_root,
    )
    return bundled_verifiers, bool(judge_toml)


def _write_reward_toml(
    case: schema.TestCase,
    eval_cfg: schema.EvaluationConfig,
    tests_dir: Path,
    bundled_verifiers: Sequence[tuple[str, schema.ScriptCheck]],
    has_judge_toml: bool,
) -> None:
    """Writes tests/reward.toml unless verifier_files already staged one."""
    reward_toml_path = tests_dir / "reward.toml"
    if reward_toml_path.exists():
        return
    eff_aggregation = case.reward_aggregation or eval_cfg.reward_aggregation
    eff_weights: dict[str, float] = {}
    if any(sc.weight != 1.0 for _, sc in bundled_verifiers):
        if has_judge_toml:
            eff_weights["expectations"] = 1.0
        for folder_name, sc in bundled_verifiers:
            eff_weights[folder_name] = sc.weight
    reward_toml_path.write_text(
        template_generators.generate_reward_toml(
            aggregation=eff_aggregation,
            weights=eff_weights or None,
        ),
        encoding="utf-8",
    )


def _write_verifier_params(
    case: schema.TestCase,
    tests_dir: Path,
    bundled_verifiers: Sequence[tuple[str, schema.ScriptCheck]],
) -> None:
    """Writes tests/verifier_params.json keyed by verifier folder when non-empty."""
    effective_params: dict[str, Any] = {}
    if bundled_verifiers:
        for folder_name, sc in bundled_verifiers:
            is_toml = (len(sc.scripts) == 1
                       and (tests_dir / folder_name / "judge.toml").is_file())
            params: dict[str, Any] = {}
            if sc.verifier_params:
                params.update(sc.verifier_params)
            if case.verifier_params:
                params.update(case.verifier_params)
            if not is_toml or params:
                effective_params[folder_name] = params
    elif case.verifier_params:
        effective_params = dict(case.verifier_params)

    if effective_params:
        (tests_dir / "verifier_params.json").write_text(
            json.dumps(effective_params, indent=2) + "\n",
            encoding="utf-8",
        )


def _write_test_runner(
    case: schema.TestCase,
    eval_cfg: schema.EvaluationConfig,
    tests_dir: Path,
    bundled_verifiers: Sequence[tuple[str, schema.ScriptCheck]],
    source_root: Path | None,
) -> None:
    """Stages pre/post lifecycle hooks and writes executable tests/test.sh."""
    eff_pre_verify = (case.pre_verify
                      if case.pre_verify is not None else eval_cfg.pre_verify)
    eff_post_verify = (case.post_verify if case.post_verify is not None else
                       eval_cfg.post_verify)
    reserved_hook_names = {
        "test.sh",
        "reward.toml",
        "verifier_params.json",
    } | {p.name
         for p in tests_dir.iterdir()}
    staged_pre = stage_lifecycle_hooks(
        eff_pre_verify,
        tests_dir=tests_dir,
        source_root=source_root,
        reserved_names=reserved_hook_names,
    )
    staged_post = stage_lifecycle_hooks(
        eff_post_verify,
        tests_dir=tests_dir,
        source_root=source_root,
        reserved_names=reserved_hook_names,
    )

    pythonpath_dirs = [
        folder_name for folder_name, sc in bundled_verifiers if sc.deps
    ]
    test_sh = tests_dir / "test.sh"
    test_sh.write_text(
        template_generators.generate_test_sh(
            pre_verify=staged_pre,
            post_verify=staged_post,
            pythonpath_dirs=pythonpath_dirs,
        ),
        encoding="utf-8",
    )
    os.chmod(test_sh, 0o755)


def compile_case(
    suite_name: str,
    case: schema.TestCase,
    env: schema.EnvironmentConfig,
    eval_cfg: schema.EvaluationConfig,
    output_dir: Path,
    judge_model_override: str | None = None,
    source_root: Path | None = None,
) -> Path:
    """Compiles a single TestCase into a self-contained Harbor task bundle."""
    task_dir = output_dir / sanitize_name(case.name)
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    _write_task_files(
        suite_name=suite_name,
        case=case,
        env=env,
        eval_cfg=eval_cfg,
        task_dir=task_dir,
        source_root=source_root,
    )
    _bundle_environment(
        effective_env=env.merged_with(case.environment),
        case=case,
        task_dir=task_dir,
        source_root=source_root,
    )

    tests_dir = task_dir / "tests"
    bundled_verifiers, has_judge_toml = _stage_verifier_assets(
        case=case,
        eval_cfg=eval_cfg,
        task_dir=task_dir,
        tests_dir=tests_dir,
        judge_model_override=judge_model_override,
        source_root=source_root,
    )
    _write_reward_toml(
        case=case,
        eval_cfg=eval_cfg,
        tests_dir=tests_dir,
        bundled_verifiers=bundled_verifiers,
        has_judge_toml=has_judge_toml,
    )
    _write_verifier_params(
        case=case,
        tests_dir=tests_dir,
        bundled_verifiers=bundled_verifiers,
    )
    _write_test_runner(
        case=case,
        eval_cfg=eval_cfg,
        tests_dir=tests_dir,
        bundled_verifiers=bundled_verifiers,
        source_root=source_root,
    )

    return task_dir


def compile_suite(
    suite_path: str | Path,
    output_dir: str | Path,
    judge_model_override: str | None = None,
    task_name_filter: str | None = None,
) -> list[Path]:
    """Compiles a YAML evaluation suite into Harbor task bundles."""
    output_dir_p = Path(output_dir).resolve()
    output_dir_p.mkdir(parents=True, exist_ok=True)

    if isinstance(suite_path, Path) or (isinstance(suite_path, str)
                                        and "\n" not in suite_path):
        cand_p = Path(suite_path)
        if cand_p.exists():
            content = cand_p.read_text(encoding="utf-8")
            suite = schema.EvalSuite.from_yaml(content)
            suite_name = suite.suite_name or suite.name or cand_p.stem
            source_root = cand_p.parent if cand_p.is_file() else Path.cwd()
        else:
            resolved_suite = resolve_path(str(suite_path))
            if resolved_suite and resolved_suite.exists():
                content = resolved_suite.read_text(encoding="utf-8")
                suite = schema.EvalSuite.from_yaml(content)
                suite_name = suite.suite_name or suite.name or resolved_suite.stem
                source_root = resolved_suite.parent
            else:
                suite = schema.EvalSuite.from_yaml(str(suite_path))
                suite_name = suite.suite_name or suite.name or "evaluation"
                source_root = Path.cwd()
    else:
        suite = schema.EvalSuite.from_yaml(suite_path)
        suite_name = suite.suite_name or suite.name or "evaluation"
        source_root = Path.cwd()

    cases = [
        c for c in suite.evaluation.cases
        if not task_name_filter or c.name == task_name_filter
        or sanitize_name(c.name) == sanitize_name(task_name_filter) or (
            c.task_name and c.task_name == task_name_filter)
    ]

    if task_name_filter and not cases:
        available = [c.name for c in suite.evaluation.cases]
        raise ValueError(
            f"No test case found matching task_name filter '{task_name_filter}'. "
            f"Available cases in suite: {available}")

    return [
        compile_case(
            suite_name=suite_name,
            case=case,
            env=suite.environment,
            eval_cfg=suite.evaluation,
            output_dir=output_dir_p,
            judge_model_override=judge_model_override,
            source_root=source_root,
        ) for case in cases
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for converting YAML suites to Harbor bundles."""
    parser = argparse.ArgumentParser(
        description="Convert YAML evaluation suites to Harbor task bundles.")
    parser.add_argument(
        "-s",
        "--suite",
        type=str,
        required=False,
        help="Path to YAML evaluation suite definition file.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=False,
        help="Output directory to write Harbor task bundle(s).",
    )
    parser.add_argument(
        "-j",
        "--judge_model",
        type=str,
        default=None,
        help="Judge model override (e.g. 'gemini/gemini-3.6-flash').",
    )
    parser.add_argument(
        "-t",
        "--task_name",
        type=str,
        default=None,
        help="Filter to convert only a single test case matching this name.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        force=True,
    )

    if args.suite and args.output_dir:
        print(f"Compiling Harbor evaluation suite from: {args.suite}")
        converted = compile_suite(
            suite_path=args.suite,
            output_dir=args.output_dir,
            judge_model_override=args.judge_model,
            task_name_filter=args.task_name,
        )
        for p in converted:
            print(f"  ✓ {p.name}")

        print(
            f"\nSuccessfully compiled {len(converted)} Harbor task bundles to:"
            f" {args.output_dir}")
        return 0

    if args.suite and not args.output_dir:
        parser.error("--output_dir is required when --suite is specified.")

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
