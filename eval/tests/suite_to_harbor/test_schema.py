# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for evaluation suite schema definitions."""

from __future__ import annotations

import pathlib
import tempfile
import unittest
from typing import Any
from unittest import mock

import pydantic

from eval.suite_to_harbor import schema

Path = pathlib.Path


class SchemaTest(unittest.TestCase):

    def test_default_evaluation_and_environment_configs(self) -> None:
        env_cfg = schema.EnvironmentConfig()
        self.assertEqual(env_cfg.base_image, schema.DEFAULT_BASE_IMAGE)

        eval_cfg = schema.EvaluationConfig(
            cases=[schema.TestCase(name="c1", prompt="p1")])
        self.assertEqual(eval_cfg.judge_model, "gemini/gemini-3.6-flash")
        self.assertEqual(eval_cfg.timeout_sec, 300.0)
        self.assertEqual(eval_cfg.verifier_timeout_sec, 300.0)
        self.assertEqual(eval_cfg.pre_verify, [])
        self.assertEqual(eval_cfg.post_verify, [])
        self.assertEqual(eval_cfg.reward_aggregation, "weighted_mean")
        self.assertEqual(eval_cfg.script_verifiers, [])
        self.assertIsNone(eval_cfg.cases[0].timeout_sec)
        self.assertIsNone(eval_cfg.cases[0].verifier_timeout_sec)

    def test_extra_forbidden_on_models(self) -> None:
        cases: list[tuple[str, type[pydantic.BaseModel], dict[str, Any]]] = [
            (
                "eval_suite_extra_field",
                schema.EvalSuite,
                {
                    "evaluation": {
                        "cases": [{
                            "name": "c1",
                            "prompt": "p1"
                        }]
                    },
                    "unknown_pillar": {},
                },
            ),
            (
                "test_case_extra_field",
                schema.TestCase,
                {
                    "name": "c1",
                    "prompt": "p1",
                    "forbidden_field": 123,
                },
            ),
            (
                "case_env_override_extra_field",
                schema.CaseEnvironmentOverride,
                {
                    "memory_mb": 8192,
                    "unknown_key": True,
                },
            ),
        ]
        for label, model_cls, data in cases:
            with self.subTest(case=label):
                with self.assertRaises(pydantic.ValidationError):
                    model_cls.model_validate(data)

    def test_expectation_criterion_alias_choices(self) -> None:
        crit1 = schema.ExpectationCriterion.model_validate({
            "criterion": "Criterion 1",
            "weight": 2.0,
            "type": "binary",
        })
        self.assertEqual(crit1.criterion, "Criterion 1")
        self.assertEqual(crit1.weight, 2.0)
        crit2 = schema.ExpectationCriterion.model_validate({
            "description": "Description alias",
            "weight": 1.5,
            "type": "numeric",
        })
        self.assertEqual(crit2.criterion, "Description alias")
        self.assertEqual(crit2.weight, 1.5)
        self.assertEqual(crit2.type, "numeric")
        crit3 = schema.ExpectationCriterion.model_validate({
            "criterion": "Likert criterion",
            "type": "likert",
        })
        self.assertEqual(crit3.type, "likert")

    def test_missing_required_fields_raise_error(self) -> None:
        cases: list[tuple[str, type[pydantic.BaseModel], dict[str, Any]]] = [
            ("empty_suite", schema.EvalSuite, {}),
            ("empty_evaluation", schema.EvaluationConfig, {}),
            ("missing_prompt_case", schema.TestCase, {
                "name": "c1"
            }),
            ("whitespace_prompt_case", schema.TestCase, {
                "name": "c1",
                "prompt": "   ",
            }),
        ]
        for label, model_cls, data in cases:
            with self.subTest(case=label):
                with self.assertRaises(pydantic.ValidationError):
                    model_cls.model_validate(data)

    def test_negative_boundary_values_raise_error(self) -> None:
        cases: list[tuple[str, type[pydantic.BaseModel], dict[str, Any]]] = [
            (
                "negative_weight_expectation",
                schema.ExpectationCriterion,
                {
                    "criterion": "c",
                    "weight": -1.0,
                },
            ),
            (
                "nan_weight_script",
                schema.ScriptCheck,
                {
                    "script": "s.py",
                    "weight": float("nan"),
                },
            ),
            (
                "inf_weight_expectation",
                schema.ExpectationCriterion,
                {
                    "criterion": "c",
                    "weight": float("inf"),
                },
            ),
            (
                "negative_timeout_case",
                schema.TestCase,
                {
                    "name": "c",
                    "prompt": "p",
                    "timeout_sec": -10.0
                },
            ),
            (
                "negative_verifier_timeout_case",
                schema.TestCase,
                {
                    "name": "c",
                    "prompt": "p",
                    "verifier_timeout_sec": -5.0,
                },
            ),
            (
                "negative_eval_timeout",
                schema.EvaluationConfig,
                {
                    "timeout_sec": 0.0,
                    "cases": [{
                        "name": "c",
                        "prompt": "p"
                    }],
                },
            ),
            ("zero_cpus", schema.EnvironmentConfig, {
                "cpus": 0
            }),
            ("none_cpus", schema.EnvironmentConfig, {
                "cpus": None
            }),
            ("zero_override_memory", schema.CaseEnvironmentOverride, {
                "memory_mb": 0
            }),
        ]
        for label, model_cls, data in cases:
            with self.subTest(case=label):
                with self.assertRaises(pydantic.ValidationError):
                    model_cls.model_validate(data)

    def test_from_yaml_string(self) -> None:
        yaml_content = """
name: test_suite
environment:
  base_image: python:3.12-slim
  pip_packages:
    - pytest
  workspace_files:
    - "shared/setup.sh:setup.sh"
evaluation:
  judge_model: gemini/gemini-2.0-flash
  timeout_sec: 1200.0
  verifier_timeout_sec: 600.0
  post_verify:
    - "hooks/post_verify.py"
  cases:
    - name: test_case_1
      prompt: |-
        Solve problem line 1.
        Solve problem line 2.
      expectations:
        - "Valid JSON output"
        - criterion: "Passes logic"
          weight: 2.0
      script_verifiers:
        - "tests/check.py"
        - script: "tests/check2.py"
          args: ["--arg1"]
          weight: 1.5
    - name: test_case_2
      prompt_file: "tasks/a1/TASK.md"
      verifier_timeout_sec: 900.0
      workspace_files:
        - "tasks/a1/TASK.md:TASK.md"
        - "traces/j3:inputs/trace_a"
"""
        expected = schema.EvalSuite(
            name="test_suite",
            environment=schema.EnvironmentConfig(
                base_image="python:3.12-slim",
                pip_packages=["pytest"],
                workspace_files=["shared/setup.sh:setup.sh"],
            ),
            evaluation=schema.EvaluationConfig(
                judge_model="gemini/gemini-2.0-flash",
                timeout_sec=1200.0,
                verifier_timeout_sec=600.0,
                post_verify=["hooks/post_verify.py"],
                cases=[
                    schema.TestCase(
                        name="test_case_1",
                        prompt="Solve problem line 1.\nSolve problem line 2.",
                        expectations=[
                            "Valid JSON output",
                            schema.ExpectationCriterion(
                                criterion="Passes logic", weight=2.0),
                        ],
                        script_verifiers=[
                            "tests/check.py",
                            schema.ScriptCheck(
                                script="tests/check2.py",
                                args=["--arg1"],
                                weight=1.5,
                            ),
                        ],
                    ),
                    schema.TestCase(
                        name="test_case_2",
                        prompt_file="tasks/a1/TASK.md",
                        verifier_timeout_sec=900.0,
                        workspace_files=[
                            "tasks/a1/TASK.md:TASK.md",
                            "traces/j3:inputs/trace_a",
                        ],
                    ),
                ],
            ),
        )
        suite = schema.EvalSuite.from_yaml(yaml_content)
        self.assertEqual(suite, expected)

    def test_from_yaml_file_path_and_legacy_version(self) -> None:
        yaml_content = """
version: "1.0"
suite_name: legacy_suite
evaluation:
  cases:
    - name: case
      prompt: "prompt"
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = pathlib.Path(tmp_dir)
            yaml_file = temp_path / "suite.yaml"
            yaml_file.write_text(yaml_content, encoding="utf-8")

            suite_from_path = schema.EvalSuite.from_yaml(yaml_file)
            self.assertEqual(suite_from_path.name, "legacy_suite")
            self.assertEqual(suite_from_path.version, "1.0")
            self.assertEqual(suite_from_path.evaluation.cases[0].name, "case")

    def test_file_staging_coercion(self) -> None:
        # Docker style colon shorthand
        fs1 = schema.FileStaging.coerce("artifacts/traces/j3:inputs/trace_a")
        self.assertEqual(fs1.src, "artifacts/traces/j3")
        self.assertEqual(fs1.dest, "inputs/trace_a")

        # Single string shorthand
        fs2 = schema.FileStaging.coerce("TASK.md")
        self.assertEqual(fs2.src, "TASK.md")
        self.assertEqual(fs2.dest, "TASK.md")

        # Explicit dict
        fs3 = schema.FileStaging.coerce({
            "src": "path/to/src",
            "dest": "path/to/dest"
        })
        self.assertEqual(fs3.src, "path/to/src")
        self.assertEqual(fs3.dest, "path/to/dest")

        # FileStaging instance
        fs4 = schema.FileStaging(src="a", dest="b")
        self.assertIs(schema.FileStaging.coerce(fs4), fs4)

        # Default dest
        fs5 = schema.FileStaging(src="only_src.txt")
        self.assertEqual(fs5.src, "only_src.txt")
        self.assertEqual(fs5.dest, ".")

        # GCS URI with colon destination
        fs_gcs1 = schema.FileStaging.coerce(
            "gs://xprof_oss_skills/traces/v1/j1_baseline:inputs/trace")
        self.assertEqual(fs_gcs1.src,
                         "gs://xprof_oss_skills/traces/v1/j1_baseline")
        self.assertEqual(fs_gcs1.dest, "inputs/trace")

        # GCS URI without destination
        fs_gcs2 = schema.FileStaging.coerce(
            "gs://xprof_oss_skills/traces/v1/trace.pb")
        self.assertEqual(fs_gcs2.src,
                         "gs://xprof_oss_skills/traces/v1/trace.pb")
        self.assertEqual(fs_gcs2.dest, "trace.pb")

        # Invalid type
        with self.assertRaises(TypeError):
            schema.FileStaging.coerce(123)

    def test_environment_workspace_files_and_overrides(self) -> None:
        env = schema.EnvironmentConfig(
            apt_packages=["openssh-client"],
            pip_packages=["transformers", "pytest"],
            memory_mb=8192,
            env_vars={
                "SUITE_VAR": "1",
                "SHARED": "suite"
            },
            workspace_files=[
                "common/data:data",
                "TASK.md",
                {
                    "src": "extra",
                    "dest": "extra_dest"
                },
            ],
        )
        norm = env.normalized_workspace_files()
        self.assertEqual(len(norm), 3)
        self.assertEqual(norm[0].src, "common/data")
        self.assertEqual(norm[0].dest, "data")

        # Explicit list override replaces suite packages
        override = schema.CaseEnvironmentOverride(
            pip_packages=["pytest"],
            setup_scripts=["assets/setup/init_env.py"],
            memory_mb=65536,
            env_vars={
                "SHARED": "case",
                "CASE_VAR": "2"
            },
        )
        merged = env.merged_with(override)
        self.assertEqual(merged.apt_packages, ["openssh-client"])
        self.assertEqual(merged.pip_packages, ["pytest"])
        self.assertEqual(
            merged.setup_scripts,
            ["assets/setup/init_env.py"],
        )
        self.assertEqual(merged.memory_mb, 65536)
        self.assertEqual(
            merged.env_vars,
            {
                "SUITE_VAR": "1",
                "SHARED": "case",
                "CASE_VAR": "2"
            },
        )
        self.assertIs(env.merged_with(None), env)

        # Test None coercion
        env_none = schema.EnvironmentConfig.model_validate(
            {"workspace_files": None})
        self.assertEqual(env_none.workspace_files, [])

        # Blank items in package/script lists are stripped and dropped
        env_blanks = schema.EnvironmentConfig(
            apt_packages=[" curl ", "", "   "],
            pip_packages=["pytest", " "],
            setup_scripts=[" ", "setup.sh"],
        )
        self.assertEqual(env_blanks.apt_packages, ["curl"])
        self.assertEqual(env_blanks.pip_packages, ["pytest"])
        self.assertEqual(env_blanks.setup_scripts, ["setup.sh"])

        override_blanks = schema.CaseEnvironmentOverride(
            apt_packages=["", "  "],
            pip_packages=["numpy", " "],
        )
        self.assertEqual(override_blanks.apt_packages, [])
        self.assertEqual(override_blanks.pip_packages, ["numpy"])

        for bad_kwargs in (
            {
                "base_image": "  "
            },
            {
                "workdir": ""
            },
        ):
            with self.assertRaises(pydantic.ValidationError):
                schema.EnvironmentConfig(**bad_kwargs)
            with self.assertRaises(pydantic.ValidationError):
                schema.CaseEnvironmentOverride(**bad_kwargs)

    def test_case_resolve_prompt_and_prompt_vars(self) -> None:
        # Resolve prompt from prompt string
        tc = schema.TestCase(name="tc", prompt="hello-world")
        self.assertEqual(tc.resolve_prompt(), "hello-world")

        # Prompt vars with embedded JSON, shell ${VAR}, single-brace {token}, and unmatched {{token}}
        tc_vars = schema.TestCase(
            name="tc_vars",
            prompt=
            ("Deploy {{service_name}} ({{ endpoint_url }}) with prefix {{job_prefix}}.\n"
             "Shell: ${HOSTNAME}\n"
             'JSON: {"config": "Baseline", "tps": 100}\n'
             "Single: {run_id}\n"
             "Unmatched: {{literal_token}}"),
            prompt_vars={
                "service_name": "demo-service",
                "endpoint_url": "https://example.com/demo",
                "job_prefix": "test-job-",
            },
        )
        rendered = tc_vars.resolve_prompt()
        self.assertIn(
            "Deploy demo-service (https://example.com/demo) with prefix test-job-.",
            rendered,
        )
        self.assertIn("Shell: ${HOSTNAME}", rendered)
        self.assertIn('JSON: {"config": "Baseline", "tps": 100}', rendered)
        self.assertIn("Single: {run_id}", rendered)
        self.assertIn("Unmatched: {{literal_token}}", rendered)

        with tempfile.TemporaryDirectory() as tmp_dir:
            suite_dir = Path(tmp_dir)
            task_md = suite_dir / "tasks" / "TASK.md"
            task_md.parent.mkdir(parents=True, exist_ok=True)
            task_md.write_text("# Task: {{target}}\nDo this step.",
                               encoding="utf-8")

            # Resolve relative to source_root
            tc_file = schema.TestCase(
                name="tc_file",
                prompt_file="tasks/TASK.md",
                prompt_vars={"target": "example-target"},
            )
            self.assertEqual(
                tc_file.resolve_prompt(source_root=suite_dir),
                "# Task: example-target\nDo this step.",
            )

            tc_repo_root = schema.TestCase(
                name="tc_repo_root",
                prompt_file="//tasks/TASK.md",
                prompt_vars={"target": "example-target"},
            )
            with mock.patch.object(schema, "_REPO_ROOT", suite_dir):
                self.assertEqual(tc_repo_root.resolve_prompt(),
                                 "# Task: example-target\nDo this step.")

            # Nonexistent prompt_file raises FileNotFoundError
            tc_nonexistent = schema.TestCase(name="tc_bad",
                                             prompt_file="missing/TASK.md")
            with self.assertRaises(FileNotFoundError):
                tc_nonexistent.resolve_prompt(source_root=suite_dir)

    def test_prompt_vars_key_unused_by_prompt_raises(self) -> None:
        tc = schema.TestCase(
            name="tc",
            prompt="Deploy {{service_name}}.",
            prompt_vars={
                "service_name": "demo-service",
                "endpoint_url": "https://example.com/demo",
            },
        )
        with self.assertRaisesRegex(ValueError, "endpoint_url"):
            tc.resolve_prompt()

    def test_case_workspace_and_test_files(self) -> None:
        tc = schema.TestCase(
            name="tc",
            prompt="do task",
            workspace_files=["tasks/a1/TASK.md:TASK.md"],
            test_files=[
                "configs/schema.json:schema.json",
                "configs/key.json:key.json",
            ],
        )
        self.assertEqual(len(tc.normalized_workspace_files()), 1)
        self.assertEqual(len(tc.normalized_verifier_files()), 2)

        # Test None coercion
        tc_none = schema.TestCase.model_validate({
            "name": "tc",
            "prompt": "p",
            "workspace_files": None,
            "test_files": None,
        })
        self.assertEqual(tc_none.workspace_files, [])
        self.assertEqual(tc_none.verifier_files, [])

    def test_script_check_verifier_params(self) -> None:
        sc_default = schema.ScriptCheck(script="tests/check.py")
        self.assertEqual(sc_default.verifier_params, {})

        sc_custom = schema.ScriptCheck(
            script="tests/check.py",
            verifier_params={
                "target_agent": "test",
                "retries": 3
            },
        )
        self.assertEqual(
            sc_custom.verifier_params,
            {
                "target_agent": "test",
                "retries": 3
            },
        )

        sc_coerced = schema.ScriptCheck.coerce({
            "script": "tests/check.py",
            "verifier_params": None,
        })
        self.assertEqual(sc_coerced.verifier_params, {})

    def test_verifiers_alias_and_script_key(self) -> None:
        suite = schema.EvalSuite.from_yaml("""
name: unified_verifiers
evaluation:
  verifiers:
    - script: verifiers/shared.py
      name: shared_check
  cases:
    - name: c1
      prompt: Run preflight
      verifiers:
        - "assets/checks/preflight.py"
        - name: expectations
          script: assets/expectations/rubric.toml
""")
        self.assertEqual(len(suite.evaluation.script_verifiers), 1)
        eval_v = suite.evaluation.normalized_script_verifiers()[0]
        self.assertEqual(eval_v.script, "verifiers/shared.py")
        self.assertEqual(eval_v.name, "shared_check")

        case_vs = suite.evaluation.cases[0].normalized_script_verifiers()
        self.assertEqual(len(case_vs), 2)
        self.assertEqual(case_vs[0].script, "assets/checks/preflight.py")
        self.assertIsNone(case_vs[0].name)
        self.assertEqual(case_vs[1].script, "assets/expectations/rubric.toml")
        self.assertEqual(case_vs[1].name, "expectations")

    def test_files_deps_and_verifier_files_aliases(self) -> None:
        suite = schema.EvalSuite.from_yaml("""
name: modern_suite
evaluation:
  verifier_files:
    - "assets/shared_schema.json:schema.json"
  verifiers:
    - name: checks
      files:
        - "verifiers/check_a.py"
        - "verifiers/check_b.py"
      deps:
        - "verifiers/common.py"
  cases:
    - name: c1
      prompt: Run check
      verifier_files:
        - "assets/answer.json:answer.json"
""")
        self.assertEqual(
            [vf.src for vf in suite.evaluation.normalized_verifier_files()],
            ["assets/shared_schema.json"],
        )
        eval_v = suite.evaluation.normalized_script_verifiers()[0]
        self.assertEqual(
            eval_v.scripts,
            ["verifiers/check_a.py", "verifiers/check_b.py"],
        )
        self.assertEqual(eval_v.deps, ["verifiers/common.py"])
        case = suite.evaluation.cases[0]
        self.assertEqual(
            [vf.src for vf in case.normalized_verifier_files()],
            ["assets/answer.json"],
        )

    def test_reward_aggregation_enforces_known_values(self) -> None:
        for valid in (
                "weighted_mean",
                "weighted-mean",
                "weighted_sum",
                "weighted-sum",
                "all_pass",
                "all-pass",
                "any_pass",
                "any-pass",
        ):
            tc = schema.TestCase(
                name="c",
                prompt="p",
                reward_aggregation=valid,
            )
            self.assertEqual(tc.reward_aggregation, valid)

        with self.assertRaises(pydantic.ValidationError):
            schema.TestCase(
                name="c",
                prompt="p",
                reward_aggregation="unknown_strategy",  # type: ignore[arg-type]
            )

        with self.assertRaises(pydantic.ValidationError):
            schema.EvaluationConfig(
                reward_aggregation="unknown_strategy",  # type: ignore[arg-type]
                cases=[schema.TestCase(name="c", prompt="p")],
            )


if __name__ == "__main__":
    unittest.main()
