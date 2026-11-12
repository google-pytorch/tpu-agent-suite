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
"""Unit tests for the suite to Harbor task compiler and orchestrator."""

from __future__ import annotations

import json
import os
import pathlib
import stat
import subprocess
import tempfile
import tomllib
import unittest
from unittest import mock

from eval.suite_to_harbor import schema, suite_to_harbor, template_generators

Path = pathlib.Path


class SuiteToHarborTest(unittest.TestCase):

    def test_sanitize_name(self) -> None:
        self.assertEqual(suite_to_harbor.sanitize_name("My Task! #1"),
                         "my_task_1")
        self.assertEqual(suite_to_harbor.sanitize_name("___hello___"), "hello")
        self.assertEqual(suite_to_harbor.sanitize_name("   "), "eval_task")
        self.assertEqual(suite_to_harbor.sanitize_name("task-name.v2"),
                         "task-name.v2")

    def test_resolve_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            file_in_root = root / "script.py"
            file_in_root.write_text("print(1)", encoding="utf-8")

            # Absolute path resolution
            self.assertEqual(
                suite_to_harbor.resolve_path(str(file_in_root.resolve())),
                file_in_root.resolve(),
            )
            # Relative to source_root
            self.assertEqual(
                suite_to_harbor.resolve_path("script.py", source_root=root),
                file_in_root.resolve(),
            )
            # Empty / Nonexistent
            self.assertIsNone(suite_to_harbor.resolve_path(""))
            self.assertIsNone(
                suite_to_harbor.resolve_path("nonexistent.py",
                                             source_root=root))

    def test_copy_filtered_tree_filters_hidden_and_test_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src = tmp_path / "src"
            dest = tmp_path / "dest"
            src.mkdir()

            (src / "valid_script.py").write_text("# code", encoding="utf-8")
            (src / "doc.txt").write_text("# doc", encoding="utf-8")
            (src / ".hidden_file").write_text("# hidden", encoding="utf-8")
            (src / "module_test.py").write_text("# test", encoding="utf-8")

            nested = src / "subdir"
            nested.mkdir()
            (nested / "nested_script.py").write_text("# nested",
                                                     encoding="utf-8")
            (nested / "nested_test.py").write_text("# test", encoding="utf-8")

            suite_to_harbor.copy_filtered_tree(src, dest)

            # Valid files should be copied (independent inodes since outside cache_root)
            self.assertTrue((dest / "valid_script.py").exists())
            self.assertEqual(
                (dest / "valid_script.py").read_text(encoding="utf-8"),
                "# code",
            )
            self.assertTrue((dest / "doc.txt").exists())
            self.assertTrue((dest / "subdir" / "nested_script.py").exists())
            self.assertNotEqual(
                (src / "valid_script.py").stat().st_ino,
                (dest / "valid_script.py").stat().st_ino,
            )

            # Filtered files should NOT be copied
            self.assertFalse((dest / ".hidden_file").exists())
            self.assertFalse((dest / "module_test.py").exists())
            self.assertFalse((dest / "subdir" / "nested_test.py").exists())

    def test_link_or_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cache_dir = tmp_path / "cache"
            cache_dir.mkdir()
            cached_src = cache_dir / "trace.pb"
            cached_src.write_text("cached trace data", encoding="utf-8")

            local_src = tmp_path / "local.py"
            local_src.write_text("print('local')", encoding="utf-8")

            dest_cached = tmp_path / "dest_cached.pb"
            dest_local = tmp_path / "dest_local.py"

            with mock.patch.dict(os.environ,
                                 {"EVAL_GCS_CACHE_DIR": str(cache_dir)}):
                # 1. Cached GCS asset is hardlinked (same inode)
                suite_to_harbor.link_or_copy(cached_src, dest_cached)
                self.assertTrue(dest_cached.is_file())
                self.assertEqual(cached_src.stat().st_ino,
                                 dest_cached.stat().st_ino)

                # 2. Local workspace/repo file is copied (different inode)
                suite_to_harbor.link_or_copy(local_src, dest_local)
                self.assertTrue(dest_local.is_file())
                self.assertEqual(dest_local.read_text(encoding="utf-8"),
                                 "print('local')")
                self.assertNotEqual(local_src.stat().st_ino,
                                    dest_local.stat().st_ino)

                # 3. Overwriting existing dest for cached asset
                suite_to_harbor.link_or_copy(cached_src, dest_cached)
                self.assertTrue(dest_cached.is_file())
                self.assertEqual(cached_src.stat().st_ino,
                                 dest_cached.stat().st_ino)

                # 4. Fallback to copy2 on OSError for cached asset
                dest_fallback = tmp_path / "dest_fallback.pb"
                with mock.patch(
                        "os.link",
                        side_effect=OSError(18, "Invalid cross-device link"),
                ):
                    suite_to_harbor.link_or_copy(cached_src, dest_fallback)
                    self.assertTrue(dest_fallback.is_file())
                    self.assertEqual(dest_fallback.read_text(encoding="utf-8"),
                                     "cached trace data")

    def test_bundle_script_verifiers_with_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            task_dir = tmp_path / "task"
            source_root = tmp_path / "source"
            source_root.mkdir()

            verifier_py = source_root / "check_tpu.py"
            verifier_py.write_text("print('verify')", encoding="utf-8")

            verifiers = [
                schema.ScriptCheck(
                    script="check_tpu.py",
                    args=["--timeout", "10"],
                    name="tpu_check",
                )
            ]

            suite_to_harbor.bundle_script_verifiers(verifiers,
                                                    task_dir=task_dir,
                                                    source_root=source_root)

            dest_script = task_dir / "tests" / "tpu_check" / "check_tpu.py"
            self.assertTrue(dest_script.exists())
            content = dest_script.read_text(encoding="utf-8")
            self.assertIn('_SPECIFIED_ARGS = ["--timeout", "10"]', content)
            self.assertIn("print('verify')", content)

    def test_compile_case_stages_test_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "output"
            source_root = tmp_path / "source"
            source_root.mkdir()

            schema_file = source_root / "answer.schema.json"
            schema_file.write_text('{"type": "object"}', encoding="utf-8")

            key_file = source_root / "answer_key.json"
            key_file.write_text('{"score": 100}', encoding="utf-8")

            case = schema.TestCase(
                name="c1",
                prompt="Prompt 1",
                test_files=[
                    "answer.schema.json:schema.json",
                    "answer_key.json:key.json",
                ],
            )
            env = schema.EnvironmentConfig()
            eval_cfg = schema.EvaluationConfig(cases=[case])

            task_dir = suite_to_harbor.compile_case(
                suite_name="test_suite",
                case=case,
                env=env,
                eval_cfg=eval_cfg,
                output_dir=output_dir,
                source_root=source_root,
            )

            tests_dir = task_dir / "tests"
            dest_schema = tests_dir / "schema.json"
            dest_key = tests_dir / "key.json"

            self.assertTrue(dest_schema.exists())
            self.assertEqual(dest_schema.read_text(encoding="utf-8"),
                             '{"type": "object"}')
            self.assertTrue(dest_key.exists())
            self.assertEqual(dest_key.read_text(encoding="utf-8"),
                             '{"score": 100}')

    def test_compile_case_generates_all_harbor_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "output"

            case = schema.TestCase(
                name="basic_case",
                prompt="Fix TPU communication latency.",
                expectations=["TPU test passes", "Latency under 5ms"],
            )
            env = schema.EnvironmentConfig(
                base_image="python:3.12-slim",
                pip_packages=["torch", "libtpu"],
            )
            eval_cfg = schema.EvaluationConfig(
                judge_model="gemini/gemini-3.6-flash",
                cases=[case],
            )

            task_dir = suite_to_harbor.compile_case(
                suite_name="tpu_suite",
                case=case,
                env=env,
                eval_cfg=eval_cfg,
                output_dir=output_dir,
            )

            self.assertTrue(task_dir.exists())
            self.assertEqual(task_dir.name, "basic_case")

            # 1. instruction.md
            instruction = (task_dir /
                           "instruction.md").read_text(encoding="utf-8")
            self.assertIn("Fix TPU communication latency.", instruction)

            # 2. task.toml
            task_toml = (task_dir / "task.toml").read_text(encoding="utf-8")
            self.assertIn('name = "tpu_suite/basic_case"', task_toml)

            # 3. environment/Dockerfile
            dockerfile = (task_dir / "environment" /
                          "Dockerfile").read_text(encoding="utf-8")
            self.assertIn("python:3.12-slim", dockerfile)
            self.assertIn("torch libtpu", dockerfile)
            self.assertIn("harbor-rewardkit", dockerfile)

            # 4. tests/reward.toml
            self.assertTrue((task_dir / "tests" / "reward.toml").exists())

            # 5. tests/expectations/judge.toml
            judge_toml = (task_dir / "tests" / "expectations" /
                          "judge.toml").read_text(encoding="utf-8")
            self.assertIn('judge = "gemini/gemini-3.6-flash"', judge_toml)
            self.assertIn("TPU test passes", judge_toml)
            self.assertIn("Latency under 5ms", judge_toml)

            # 6. tests/test.sh (must be executable)
            test_sh = task_dir / "tests" / "test.sh"
            self.assertTrue(test_sh.exists())
            mode = test_sh.stat().st_mode
            self.assertTrue(bool(mode & stat.S_IXUSR))

    def test_compile_suite_from_yaml_string(self) -> None:
        yaml_content = """
name: my_suite
environment:
  base_image: python:3.12-slim
evaluation:
  cases:
    - name: case_1
      prompt: "First task"
    - name: case_2
      prompt: "Second task"
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "output"
            tasks = suite_to_harbor.compile_suite(
                suite_path=yaml_content,
                output_dir=output_dir,
            )

            self.assertEqual(len(tasks), 2)
            self.assertTrue((output_dir / "case_1").exists())
            self.assertTrue((output_dir / "case_2").exists())

    @mock.patch("shutil.which", return_value="/usr/bin/gcloud")
    def test_compile_case_with_gcs_workspace_file(
            self, mock_which: mock.MagicMock) -> None:
        yaml_content = """
name: gcs_suite
environment:
  base_image: python:3.12-slim
evaluation:
  cases:
    - name: gcs_case
      prompt: "Analyze the trace."
      workspace_files:
        - "gs://xprof_bucket/traces/v1/sample.pb:inputs/trace/sample.pb"
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            custom_cache = tmp_root / "cache"
            output_dir = tmp_root / "output"

            def fake_download(cmd, **kwargs):
                dest = Path(cmd[-1])
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "sample.pb").write_bytes(b"trace_binary_content")
                return mock.MagicMock(returncode=0)

            with mock.patch.dict(os.environ,
                                 {"EVAL_GCS_CACHE_DIR": str(custom_cache)}):
                with mock.patch("subprocess.run", side_effect=fake_download):
                    tasks = suite_to_harbor.compile_suite(
                        suite_path=yaml_content,
                        output_dir=output_dir,
                    )

            self.assertEqual(len(tasks), 1)
            staged_file = (output_dir / "gcs_case" / "environment" / "data" /
                           "inputs" / "trace" / "sample.pb")
            self.assertTrue(staged_file.is_file())
            self.assertEqual(staged_file.read_bytes(), b"trace_binary_content")

    def test_compile_suite_with_task_name_filter(self) -> None:
        yaml_content = """
name: filtered_suite
evaluation:
  cases:
    - name: target_case
      prompt: "Target task"
    - name: other_case
      prompt: "Other task"
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "output"
            tasks = suite_to_harbor.compile_suite(
                suite_path=yaml_content,
                output_dir=output_dir,
                task_name_filter="target_case",
            )

            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].name, "target_case")
            self.assertFalse((output_dir / "other_case").exists())

    def test_compile_suite_filter_not_found_raises_value_error(self) -> None:
        yaml_content = """
name: test_suite
evaluation:
  cases:
    - name: case_a
      prompt: "Task A"
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "output"
            with self.assertRaises(ValueError) as ctx:
                suite_to_harbor.compile_suite(
                    suite_path=yaml_content,
                    output_dir=output_dir,
                    task_name_filter="nonexistent_case",
                )
            self.assertIn("nonexistent_case", str(ctx.exception))

    def test_stage_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_root = tmp_path / "source"
            source_root.mkdir()
            task_dir = tmp_path / "task"
            dest_dir = task_dir / "environment" / "data"

            # Prepare host files
            src_file = source_root / "task_guide.md"
            src_file.write_text("# My Task", encoding="utf-8")

            traces_dir = source_root / "traces"
            traces_dir.mkdir()
            (traces_dir / "trace1.json").write_text('{"trace": 1}',
                                                    encoding="utf-8")
            (traces_dir / "trace2.json").write_text('{"trace": 2}',
                                                    encoding="utf-8")

            # 1. Directory remap: "traces:inputs/trace_a"
            fs_dir = schema.FileStaging(src="traces", dest="inputs/trace_a")
            suite_to_harbor.stage_file(fs_dir,
                                       dest_dir=dest_dir,
                                       source_root=source_root)
            self.assertTrue((task_dir / "environment" / "data" / "inputs" /
                             "trace_a" / "trace1.json").exists())
            self.assertTrue((task_dir / "environment" / "data" / "inputs" /
                             "trace_a" / "trace2.json").exists())

            # 2. File rename: "task_guide.md:TASK.md"
            fs_file_rename = schema.FileStaging(src="task_guide.md",
                                                dest="TASK.md")
            suite_to_harbor.stage_file(fs_file_rename,
                                       dest_dir=dest_dir,
                                       source_root=source_root)
            target_md = task_dir / "environment" / "data" / "TASK.md"
            self.assertTrue(target_md.is_file())
            self.assertEqual(target_md.read_text(encoding="utf-8"),
                             "# My Task")

            # 3. File into folder: "task_guide.md:inputs/"
            fs_file_folder = schema.FileStaging(src="task_guide.md",
                                                dest="inputs/")
            suite_to_harbor.stage_file(fs_file_folder,
                                       dest_dir=dest_dir,
                                       source_root=source_root)
            self.assertTrue((task_dir / "environment" / "data" / "inputs" /
                             "task_guide.md").exists())

            # 4. File to root "."
            fs_file_root = schema.FileStaging(src="task_guide.md", dest=".")
            suite_to_harbor.stage_file(fs_file_root,
                                       dest_dir=dest_dir,
                                       source_root=source_root)
            self.assertTrue(
                (task_dir / "environment" / "data" / "task_guide.md").exists())

            # 5. Nonexistent file is an error, not a skip
            fs_missing = schema.FileStaging(src="missing.txt",
                                            dest="missing.txt")
            with self.assertRaises(FileNotFoundError):
                suite_to_harbor.stage_file(fs_missing,
                                           dest_dir=dest_dir,
                                           source_root=source_root)

    def test_compile_with_prompt_file_and_workspace_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            suite_dir = Path(tmp_dir) / "suite"
            output_dir = Path(tmp_dir) / "output"
            suite_dir.mkdir()

            task_dir = suite_dir / "tasks" / "triage"
            task_dir.mkdir(parents=True)
            (task_dir / "TASK.md").write_text(
                "# Bottleneck Triage\nTriage issue.", encoding="utf-8")

            trace_dir = suite_dir / "artifacts" / "traces" / "j3"
            trace_dir.mkdir(parents=True)
            (trace_dir / "trace.json").write_text("{}", encoding="utf-8")

            yaml_file = suite_dir / "suite.yaml"
            yaml_file.write_text(
                """
version: "1.0"
name: "eval"
environment:
  base_image: "python:3.12-slim"
evaluation:
  cases:
    - name: "triage"
      prompt_file: "tasks/triage/TASK.md"
      workspace_files:
        - "tasks/triage/TASK.md:TASK.md"
        - "artifacts/traces/j3:inputs/trace_a"
""",
                encoding="utf-8",
            )

            tasks = suite_to_harbor.compile_suite(
                suite_path=yaml_file,
                output_dir=output_dir,
            )

            self.assertEqual(len(tasks), 1)
            t_dir = tasks[0]
            self.assertEqual(t_dir.name, "triage")

            self.assertEqual(
                (t_dir / "instruction.md").read_text(encoding="utf-8"),
                "# Bottleneck Triage\nTriage issue.",
            )

            self.assertTrue(
                (t_dir / "environment" / "data" / "TASK.md").exists())
            self.assertTrue((t_dir / "environment" / "data" / "inputs" /
                             "trace_a" / "trace.json").exists())

            dockerfile = (t_dir / "environment" /
                          "Dockerfile").read_text(encoding="utf-8")
            self.assertIn("COPY data/ /workspace/", dockerfile)

    def test_cli_execution_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            suite_dir = Path(tmp_dir) / "suite"
            output_dir = Path(tmp_dir) / "output"
            suite_dir.mkdir()

            yaml_file = suite_dir / "suite.yaml"
            yaml_file.write_text(
                """
version: "1.0"
name: "cli_suite"
environment:
  base_image: "python:3.12-slim"
evaluation:
  cases:
    - name: "cli_task"
      prompt: "Execute task"
""",
                encoding="utf-8",
            )

            exit_code = suite_to_harbor.main([
                "-s",
                str(yaml_file),
                "-o",
                str(output_dir),
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "cli_task" / "task.toml").is_file())
            self.assertTrue(
                (output_dir / "cli_task" / "instruction.md").is_file())

    def test_generate_dockerfile_quotes_pip_packages_with_specifiers(
            self) -> None:
        env = schema.EnvironmentConfig(
            base_image="python:3.12-slim",
            pip_packages=[
                "torch>=2.0.0",
                "scipy<1.10",
                "numpy!=1.24.0",
                "flax",
            ],
        )
        dockerfile = template_generators.generate_dockerfile(env)
        self.assertIn(
            "RUN pip install --no-cache-dir harbor-rewardkit 'torch>=2.0.0' 'scipy<1.10' 'numpy!=1.24.0' flax",
            dockerfile,
        )

    def test_generate_test_sh_exits_with_the_verifier_code(self) -> None:
        """A verifier that fails every attempt fails the trial with its code,
        rather than the 0 an `if` leaves in `$?`."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bin_dir = pathlib.Path(tmp_dir)
            for name, body in (("rewardkit", "echo run >> {log}; exit 3"),
                               ("sleep", "exit 0")):
                tool = bin_dir / name
                tool.write_text("#!/bin/sh\n" +
                                body.format(log=bin_dir / "runs") + "\n")
                tool.chmod(0o755)
            test_sh = bin_dir / "test.sh"
            test_sh.write_text(template_generators.generate_test_sh())
            logs_dir = bin_dir / "logs"
            ws_dir = bin_dir / "ws"
            ws_dir.mkdir()
            result = subprocess.run(
                ["bash", str(test_sh)],
                env={
                    **os.environ,
                    "PATH": f"{bin_dir}:{os.environ['PATH']}",
                    "LOGS_DIR": str(logs_dir),
                    "WORKSPACE_DIR": str(ws_dir),
                },
                capture_output=True,
                check=False,
            )
            self.assertEqual(3, result.returncode)
            self.assertEqual(5, len((bin_dir / "runs").read_text().split()))

    def test_resolve_path_with_gcs_uri(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir)
            cached_file = cache_root / "bucket" / "file.txt"
            cached_file.parent.mkdir(parents=True, exist_ok=True)
            cached_file.write_text("hello", encoding="utf-8")

            resolved = suite_to_harbor.resolve_path(
                "gs://bucket/file.txt",
                cache_root=cache_root,
            )
            self.assertEqual(resolved, cached_file)

    def test_compile_case_case_level_verifier_params_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_root = tmp_path / "source"
            source_root.mkdir()
            output_dir = tmp_path / "output"

            v1_file = source_root / "check_agent.py"
            v1_file.write_text("print('agent')", encoding="utf-8")

            case = schema.TestCase(
                name="override_case",
                prompt="Override params at case level.",
                verifier_params={
                    "target_agent": "overridden_agent",
                    "extra_key": "val",
                },
                script_verifiers=[
                    schema.ScriptCheck(
                        script="check_agent.py",
                        name="check_agent",
                        verifier_params={
                            "target_agent": "default_agent",
                            "base_key": 1,
                        },
                    ),
                ],
            )
            eval_cfg = schema.EvaluationConfig(cases=[case])
            env = schema.EnvironmentConfig()

            task_dir = suite_to_harbor.compile_case(
                suite_name="override_suite",
                case=case,
                env=env,
                eval_cfg=eval_cfg,
                output_dir=output_dir,
                source_root=source_root,
            )

            params_file = task_dir / "tests" / "verifier_params.json"
            self.assertTrue(params_file.is_file())
            loaded_params = json.loads(params_file.read_text(encoding="utf-8"))
            self.assertEqual(
                loaded_params,
                {
                    "check_agent": {
                        "base_key": 1,
                        "target_agent": "overridden_agent",
                        "extra_key": "val",
                    },
                },
            )

    def test_compile_case_without_script_verifiers_omits_verifier_params_json(
            self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "output"

            case = schema.TestCase(
                name="no_verifier_case",
                prompt="No script verifiers.",
            )
            eval_cfg = schema.EvaluationConfig(cases=[case])
            env = schema.EnvironmentConfig()

            task_dir = suite_to_harbor.compile_case(
                suite_name="no_verifier_suite",
                case=case,
                env=env,
                eval_cfg=eval_cfg,
                output_dir=output_dir,
            )

            params_file = task_dir / "tests" / "verifier_params.json"
            self.assertFalse(params_file.exists())

    def test_compile_suite_with_verifier_params_yaml(self) -> None:
        yaml_content = """
version: "1.0"
name: "xprof_suite"
environment:
  base_image: "python:3.12-slim"
evaluation:
  script_verifiers:
    - name: "check_agent_utilized"
      script: "check_agent_utilized.py"
      verifier_params:
        target_agent: "test"
  cases:
    - name: "a2_roofline"
      prompt: "Analyze roofline."
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            suite_dir = tmp_path / "suite"
            suite_dir.mkdir()
            output_dir = tmp_path / "output"

            (suite_dir / "check_agent_utilized.py").write_text(
                "print('utilized')", encoding="utf-8")
            yaml_file = suite_dir / "suite.yaml"
            yaml_file.write_text(yaml_content, encoding="utf-8")

            tasks = suite_to_harbor.compile_suite(
                suite_path=yaml_file,
                output_dir=output_dir,
            )
            self.assertEqual(len(tasks), 1)
            task_dir = tasks[0]

            params_file = task_dir / "tests" / "verifier_params.json"
            self.assertTrue(params_file.is_file())
            loaded = json.loads(params_file.read_text(encoding="utf-8"))
            self.assertEqual(
                loaded,
                {"check_agent_utilized": {
                    "target_agent": "test"
                }},
            )


class TaskEnvTest(unittest.TestCase):
    """Suite and case `environment.env_vars` reach the container through task.toml."""

    def _toml(self, **case_fields) -> str:
        case = schema.TestCase(name="c", prompt="p", **case_fields)
        return template_generators.generate_task_toml(
            case, "suite", schema.EnvironmentConfig())

    def test_env_is_written_under_environment(self):
        parsed = tomllib.loads(
            self._toml(environment=schema.CaseEnvironmentOverride(
                env_vars={
                    "TPU_AGENT_SUITE_KNOWLEDGE_BASE_JAX_VERSION": "0.5.3"
                })))
        self.assertEqual(
            {"TPU_AGENT_SUITE_KNOWLEDGE_BASE_JAX_VERSION": "0.5.3"},
            parsed["environment"]["env"])

    def test_no_env_writes_no_line(self):
        rendered = self._toml()
        self.assertNotIn("env =", rendered)
        self.assertNotIn("[environment.env]", rendered)

    def test_dumps_toml_roundtrips_all_generators(self):
        reward_toml = template_generators.generate_reward_toml(
            aggregation="weighted-sum",
            weights={
                "check_one": 0.6,
                "check_two": 0.4,
            },
        )
        self.assertEqual(
            tomllib.loads(reward_toml),
            {
                "reward": [{
                    "name": "reward",
                    "aggregation": "weighted-sum",
                    "weights": {
                        "check_one": 0.6,
                        "check_two": 0.4,
                    },
                }]
            },
        )
        self.assertEqual(
            tomllib.loads(
                template_generators.generate_reward_toml(
                    aggregation="weighted_sum")),
            {"reward": [{
                "name": "reward",
                "aggregation": "weighted-sum",
            }]},
        )

        judge_toml = template_generators.generate_judge_toml(
            expectations=[
                "Check 1",
                {
                    "criterion": "Check 2",
                    "weight": 0.25,
                },
            ],
            judge_model="gemini/gemini-3.6-pro",
        )
        assert judge_toml is not None
        self.assertEqual(
            tomllib.loads(judge_toml),
            {
                "judge": {
                    "judge": "gemini/gemini-3.6-pro",
                    "atif-trajectory": "/logs/agent/trajectory.json",
                },
                "criterion": [
                    {
                        "description": "Check 1",
                        "type": "binary",
                        "weight": 1.0,
                    },
                    {
                        "description": "Check 2",
                        "type": "binary",
                        "weight": 0.25,
                    },
                ],
                "scoring": {
                    "aggregation": "weighted_mean",
                },
            },
        )


class CompilerUpgradeFeaturesTest(unittest.TestCase):
    """Tests nexus-host Dockerfiles, timeouts, reward preservation, and hooks."""

    def test_nexus_host_dockerfile_minimal_and_constraints_pinned(
            self) -> None:
        env = schema.EnvironmentConfig(
            base_image=schema.DEFAULT_BASE_IMAGE,
            apt_packages=["openssh-client"],
            pip_packages=["transformers", "pytest>=8.0"],
        )
        df = template_generators.generate_dockerfile(env, has_data=True)
        self.assertIn(
            f"FROM --platform=linux/amd64 {schema.DEFAULT_BASE_IMAGE}", df)
        self.assertIn("openssh-client", df)
        self.assertIn(
            "RUN pip install --no-cache-dir -c /etc/nexus-host/constraints.txt transformers 'pytest>=8.0'",
            df,
        )
        self.assertIn("COPY data/ /workspace/", df)
        # Must NOT re-install kubectl, gcloud, or harbor-rewardkit on nexus-host
        self.assertNotIn("kubectl", df)
        self.assertNotIn("harbor-rewardkit", df)

        # Bare nexus-host with zero extra packages emits no apt-get or pip RUN
        bare_df = template_generators.generate_dockerfile(
            schema.EnvironmentConfig(), has_data=False)
        self.assertNotIn("apt-get", bare_df)
        self.assertNotIn("pip install", bare_df)
        self.assertNotIn("/tmp/_harbor_setup", bare_df)

        # Setup scripts stage into /tmp/_harbor_setup, run, and self-delete
        setup_env = schema.EnvironmentConfig(
            setup_scripts=["assets/setup/init_env.py"], )
        setup_df = template_generators.generate_dockerfile(setup_env,
                                                           has_data=False)
        self.assertIn("COPY setup/ /tmp/_harbor_setup/", setup_df)
        self.assertIn(
            "RUN python3 /tmp/_harbor_setup/init_env.py && rm -rf /tmp/_harbor_setup",
            setup_df,
        )

    def test_prebuilt_image_discovery_pins_matching_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_repo = Path(tmp_dir)
            host_df = (fake_repo / "eval" / "images" / "custom-host" /
                       "Dockerfile")
            host_df.parent.mkdir(parents=True)
            host_df.write_text(
                "RUN touch /etc/custom-host/constraints.txt\n",
                encoding="utf-8",
            )
            worker_df = (fake_repo / "eval" / "images" / "worker-tpu" /
                         "Dockerfile")
            worker_df.parent.mkdir(parents=True)
            worker_df.write_text("FROM python:3.12-slim\n", encoding="utf-8")

            with mock.patch.object(template_generators,
                                   "find_repo_root",
                                   return_value=fake_repo):
                prebuilt_df = template_generators.generate_dockerfile(
                    schema.EnvironmentConfig(
                        base_image=
                        "example.invalid/org/custom-host@sha256:abcdef",
                        pip_packages=["pytest"],
                    ),
                    has_data=False,
                )
                self.assertIn(
                    "RUN pip install --no-cache-dir -c /etc/custom-host/constraints.txt pytest",
                    prebuilt_df,
                )
                self.assertNotIn("kubectl", prebuilt_df)
                self.assertNotIn("harbor-rewardkit", prebuilt_df)

                worker_df_out = template_generators.generate_dockerfile(
                    schema.EnvironmentConfig(
                        base_image="example.invalid/org/worker-tpu:v1",
                        pip_packages=["pytest"],
                    ),
                    has_data=False,
                )
                self.assertIn("harbor-rewardkit pytest", worker_df_out)

    def test_verifier_env_merges_into_task_toml(self) -> None:
        case = schema.TestCase(
            name="case_a",
            prompt="Run sample task.",
            verifier_env={
                "SHARED": "case",
                "CASE_VAR": "https://example.com/asset",
            },
        )
        eval_cfg = schema.EvaluationConfig(
            verifier_env={
                "SUITE_VAR": "test-project",
                "SHARED": "suite",
            },
            cases=[case],
        )
        parsed = tomllib.loads(
            template_generators.generate_task_toml(
                case,
                "sample_suite",
                schema.EnvironmentConfig(),
                eval_cfg=eval_cfg,
            ))
        self.assertEqual(
            parsed["verifier"]["env"],
            {
                "CASE_VAR": "https://example.com/asset",
                "SHARED": "case",
                "SUITE_VAR": "test-project",
            },
        )

    def test_negative_verifier_weight_requires_weighted_sum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src = tmp_path / "src"
            src.mkdir()
            (src / "score.py").write_text("print('ok')\n", encoding="utf-8")
            (src / "penalty.py").write_text("print('penalty')\n",
                                            encoding="utf-8")

            case = schema.TestCase(
                name="case_a",
                prompt="Run sample task.",
                script_verifiers=[
                    schema.ScriptCheck(script="score.py",
                                       name="primary_check",
                                       weight=1.0),
                    schema.ScriptCheck(script="penalty.py",
                                       name="penalty_check",
                                       weight=-1.0),
                ],
            )
            task_dir = suite_to_harbor.compile_case(
                suite_name="sample_suite",
                case=case,
                env=schema.EnvironmentConfig(),
                eval_cfg=schema.EvaluationConfig(
                    reward_aggregation="weighted_sum",
                    cases=[case],
                ),
                output_dir=tmp_path / "out",
                source_root=src,
            )

            reward_toml = tomllib.loads(
                (task_dir / "tests" /
                 "reward.toml").read_text(encoding="utf-8"))
            self.assertEqual(
                reward_toml["reward"][0]["weights"],
                {
                    "primary_check": 1.0,
                    "penalty_check": -1.0,
                },
            )

        with self.assertRaisesRegex(ValueError, "weighted-sum"):
            template_generators.generate_reward_toml(
                aggregation="weighted_mean",
                weights={
                    "primary_check": 1.0,
                    "penalty_check": -1.0,
                },
            )

    def test_timeout_hierarchy_and_metadata_merging(self) -> None:
        case_inherit = schema.TestCase(
            name="inherit",
            prompt="Task prompt",
            metadata={"required_env": ["API_TOKEN"]},
        )
        case_override = schema.TestCase(
            name="override",
            prompt="Task prompt",
            timeout_sec=1200.0,
            verifier_timeout_sec=2400.0,
            description="Custom task description",
        )
        eval_cfg = schema.EvaluationConfig(
            timeout_sec=21600.0,
            verifier_timeout_sec=1800.0,
            cases=[case_inherit, case_override],
        )
        env = schema.EnvironmentConfig()

        t1 = tomllib.loads(
            template_generators.generate_task_toml(case_inherit,
                                                   "my-suite",
                                                   env,
                                                   eval_cfg=eval_cfg))
        self.assertEqual(t1["agent"]["timeout_sec"], 21600.0)
        self.assertEqual(t1["verifier"]["timeout_sec"], 1800.0)
        self.assertEqual(t1["metadata"], {
            "category": "my-suite",
            "required_env": ["API_TOKEN"],
        })

        t2 = tomllib.loads(
            template_generators.generate_task_toml(case_override,
                                                   "my-suite",
                                                   env,
                                                   eval_cfg=eval_cfg))
        self.assertEqual(t2["agent"]["timeout_sec"], 1200.0)
        self.assertEqual(t2["verifier"]["timeout_sec"], 2400.0)
        self.assertEqual(t2["task"]["description"], "Custom task description")

    def test_staged_reward_toml_preserved_and_legacy_test_sh_overwritten(
            self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src = tmp_path / "src"
            src.mkdir()
            out = tmp_path / "out"

            custom_reward = (
                '[[reward]]\nname = "reward"\naggregation = "weighted-sum"\n\n'
                "[reward.weights]\ncheck_one = 0.6\n")
            (src / "reward.toml").write_text(custom_reward, encoding="utf-8")
            (src / "legacy_test.sh").write_text("#!/bin/bash\necho legacy\n",
                                                encoding="utf-8")
            (src / "post_hook.py").write_text(
                "import pathlib, os\n"
                "p = pathlib.Path(os.environ['LOGS_DIR']) / 'post_ran.txt'\n"
                "p.write_text('ok', encoding='utf-8')\n",
                encoding="utf-8",
            )
            (src / "pre_hook.sh").write_text(
                "#!/bin/bash\n"
                'echo "pre" > "${LOGS_DIR}/pre_ran.txt"\n',
                encoding="utf-8",
            )
            (src / "setup_script.py").write_text(
                "print('setup')\n",
                encoding="utf-8",
            )

            case = schema.TestCase(
                name="hooked_case",
                prompt="Run {{target}}",
                prompt_vars={"target": "demo-app"},
                environment=schema.CaseEnvironmentOverride(
                    setup_scripts=["setup_script.py"], ),
                verifier_params={"expected_mode": "fast"},
                test_files=[
                    "reward.toml:reward.toml",
                    "legacy_test.sh:test.sh",
                ],
            )
            eval_cfg = schema.EvaluationConfig(
                pre_verify=["pre_hook.sh"],
                post_verify=["post_hook.py"],
                cases=[case],
            )
            task_dir = suite_to_harbor.compile_case(
                suite_name="suite",
                case=case,
                env=schema.EnvironmentConfig(),
                eval_cfg=eval_cfg,
                output_dir=out,
                source_root=src,
            )

            tests_dir = task_dir / "tests"
            # 1. Staged reward.toml was NOT overwritten by default reward.toml
            self.assertEqual(
                (tests_dir / "reward.toml").read_text(encoding="utf-8"),
                custom_reward,
            )
            # 2. Case-level verifier_params written even with no script_verifiers
            self.assertEqual(
                json.loads(
                    (tests_dir /
                     "verifier_params.json").read_text(encoding="utf-8")),
                {"expected_mode": "fast"},
            )
            # 3. Generated test.sh overwrote staged legacy_test.sh and runs all 3 phases
            test_sh_text = (tests_dir / "test.sh").read_text(encoding="utf-8")
            self.assertNotIn("echo legacy", test_sh_text)
            self.assertIn("pre_hook.sh", test_sh_text)
            self.assertIn("post_hook.py", test_sh_text)

            # 4. Setup script staged into environment/setup/ and invoked in Dockerfile
            self.assertTrue((task_dir / "environment" / "setup" /
                             "setup_script.py").is_file())
            df_text = (task_dir / "environment" /
                       "Dockerfile").read_text(encoding="utf-8")
            self.assertIn("COPY setup/ /tmp/_harbor_setup/", df_text)
            self.assertIn(
                "RUN python3 /tmp/_harbor_setup/setup_script.py && rm -rf /tmp/_harbor_setup",
                df_text,
            )

            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            rk = bin_dir / "rewardkit"
            rk.write_text("#!/bin/sh\necho rk > \"${LOGS_DIR}/rk_ran.txt\"\n",
                          encoding="utf-8")
            rk.chmod(0o755)
            logs_dir = tmp_path / "logs"
            ws_dir = tmp_path / "ws"
            ws_dir.mkdir()

            res = subprocess.run(
                ["bash", str(tests_dir / "test.sh")],
                env={
                    **os.environ,
                    "PATH": f"{bin_dir}:{os.environ['PATH']}",
                    "TESTS_DIR": str(tests_dir),
                    "LOGS_DIR": str(logs_dir),
                    "WORKSPACE_DIR": str(ws_dir),
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr)
            self.assertTrue((logs_dir / "pre_ran.txt").is_file())
            self.assertTrue((logs_dir / "rk_ran.txt").is_file())
            self.assertTrue((logs_dir / "post_ran.txt").is_file())

    def test_unified_verifiers_stages_scripts_and_toml_rubrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src = tmp_path / "src"
            src.mkdir()
            out = tmp_path / "out"

            (src / "check_main.py").write_text("print('check')\n",
                                               encoding="utf-8")
            (src / "helper_utils.py").write_text("COMMON = 1\n",
                                                 encoding="utf-8")
            (src / "metric_a.py").write_text("print('a')\n", encoding="utf-8")
            (src / "metric_b.py").write_text("print('b')\n", encoding="utf-8")
            (src / "rubric.toml").write_text(
                '[judge]\njudge = "gemini"\n',
                encoding="utf-8",
            )
            (src / "answer.json").write_text('{"ok": true}\n',
                                             encoding="utf-8")

            suite_yaml = src / "suite.yaml"
            suite_yaml.write_text(
                """
name: unified_suite
evaluation:
  cases:
    - name: script_and_toml
      prompt: Run verification
      verifier_files:
        - "answer.json:answer.json"
      verifiers:
        - name: checks
          files:
            - check_main.py
          deps:
            - helper_utils.py
          verifier_params:
            strict: true
        - name: efficiency
          files:
            - metric_a.py
            - metric_b.py
        - name: expectations
          files: rubric.toml
          reward_weight: 0.0
""",
                encoding="utf-8",
            )

            tasks = suite_to_harbor.compile_suite(suite_yaml, out)
            self.assertEqual(len(tasks), 1)

            t1 = out / "script_and_toml" / "tests"
            self.assertTrue((t1 / "answer.json").is_file())
            self.assertTrue((t1 / "checks" / "check_main.py").is_file())
            self.assertTrue((t1 / "checks" / "helper_utils.py").is_file())
            self.assertTrue((t1 / "efficiency" / "metric_a.py").is_file())
            self.assertTrue((t1 / "efficiency" / "metric_b.py").is_file())
            self.assertTrue((t1 / "expectations" / "judge.toml").is_file())
            test_sh_text = (t1 / "test.sh").read_text(encoding="utf-8")
            self.assertIn(
                'export PYTHONPATH="${TESTS_DIR}/checks:${TESTS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"',
                test_sh_text,
            )
            reward_toml_text = (t1 / "reward.toml").read_text(encoding="utf-8")
            self.assertIn("checks = 1.0", reward_toml_text)
            self.assertIn("efficiency = 1.0", reward_toml_text)
            self.assertIn("expectations = 0.0", reward_toml_text)
            self.assertEqual(
                json.loads(
                    (t1 / "verifier_params.json").read_text(encoding="utf-8")),
                {
                    "checks": {
                        "strict": True
                    },
                    "efficiency": {},
                },
            )

            # Multiple .toml files in a single verifier entry must raise ValueError
            with self.assertRaises(ValueError):
                suite_to_harbor.bundle_script_verifiers(
                    [
                        schema.ScriptCheck(
                            name="bad_rubric",
                            script=["rubric.toml", "rubric.toml"],
                        )
                    ],
                    task_dir=out / "bad_task",
                    source_root=src,
                )

    def test_resolve_path_supports_repo_root_double_slash_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_repo = Path(tmp) / "repo"
            shared = fake_repo / "eval" / "verifiers" / "common" / "check.py"
            shared.parent.mkdir(parents=True)
            shared.write_text("# check\n", encoding="utf-8")

            suite_dir = fake_repo / "eval" / "suites" / "my_suite"
            suite_dir.mkdir(parents=True)
            # Even if a same-named relative file exists under suite_dir, // anchors at repo_root.
            shadow = suite_dir / "eval" / "verifiers" / "common" / "check.py"
            shadow.parent.mkdir(parents=True)
            shadow.write_text("# shadow\n", encoding="utf-8")

            resolved = suite_to_harbor.resolve_path(
                "//eval/verifiers/common/check.py",
                source_root=suite_dir,
                repo_root=fake_repo,
            )
            self.assertEqual(resolved, shared.resolve())
            self.assertIsNone(
                suite_to_harbor.resolve_path(
                    "//eval/verifiers/common/missing.py",
                    source_root=suite_dir,
                    repo_root=fake_repo,
                ))

    def test_missing_staged_paths_fail_compilation(self) -> None:
        missing = {
            "script_verifiers": ["check_main.py"],
            "verifier_files": ["golden_answer.json"],
            "workspace_files": ["input_data.txt"],
        }
        for field, paths in missing.items():
            with self.subTest(
                    field=field), tempfile.TemporaryDirectory() as tmp:
                case = schema.TestCase(name="case_a",
                                       prompt="Do the thing.",
                                       **{field: paths})
                with self.assertRaises(FileNotFoundError):
                    suite_to_harbor.compile_case(
                        suite_name="demo_suite",
                        case=case,
                        env=schema.EnvironmentConfig(),
                        eval_cfg=schema.EvaluationConfig(cases=[case]),
                        output_dir=Path(tmp) / "out",
                        source_root=Path(tmp),
                    )

    def test_verifier_params_keyed_by_staged_folder_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "suite"
            for group in ("group_a", "group_b"):
                (src / group).mkdir(parents=True)
                (src / group / "check_main.py").write_text("# check\n",
                                                           encoding="utf-8")
            # Two verifiers with the same file stem are staged into different
            # folders, so each must keep its own entry.
            case = schema.TestCase(
                name="case_a",
                prompt="Do the thing.",
                script_verifiers=[
                    schema.ScriptCheck(script="group_a/check_main.py",
                                       verifier_params={"threshold": 1}),
                    schema.ScriptCheck(script="group_b/check_main.py",
                                       verifier_params={"threshold": 2}),
                ],
            )
            task_dir = suite_to_harbor.compile_case(
                suite_name="demo_suite",
                case=case,
                env=schema.EnvironmentConfig(),
                eval_cfg=schema.EvaluationConfig(cases=[case]),
                output_dir=Path(tmp) / "out",
                source_root=src,
            )
            tests_dir = task_dir / "tests"
            params = json.loads(
                (tests_dir / "verifier_params.json").read_text("utf-8"))
            self.assertEqual(sorted(p["threshold"] for p in params.values()),
                             [1, 2])
            for folder_name in params:
                self.assertTrue((tests_dir / folder_name).is_dir())

    def test_stage_file_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src = tmp_path / "src"
            src.mkdir()
            (src / "data.txt").write_text("ok", encoding="utf-8")
            dest_dir = tmp_path / "task" / "environment" / "data"

            for bad_dest in ("../escape.txt", "nested/../../escape.txt"):
                with self.assertRaises(ValueError):
                    suite_to_harbor.stage_file(
                        schema.FileStaging(src="data.txt", dest=bad_dest),
                        dest_dir=dest_dir,
                        source_root=src,
                    )

    def test_resolve_path_with_source_root_ignores_cwd_and_escapes(
            self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_repo = tmp_path / "repo"
            fake_repo.mkdir()
            suite_dir = fake_repo / "suite"
            suite_dir.mkdir()
            cwd_dir = tmp_path / "cwd"
            cwd_dir.mkdir()
            (cwd_dir / "missing.txt").write_text("stray", encoding="utf-8")
            outside = tmp_path / "secret.txt"
            outside.write_text("secret", encoding="utf-8")

            old_cwd = Path.cwd()
            try:
                os.chdir(cwd_dir)
                # Relative lookup with source_root must not fall back to Path.cwd().
                self.assertIsNone(
                    suite_to_harbor.resolve_path(
                        "missing.txt",
                        source_root=suite_dir,
                        repo_root=fake_repo,
                    ))
                # Missing absolute path must not fall back to relative search roots.
                self.assertIsNone(
                    suite_to_harbor.resolve_path(
                        "/nonexistent_abs_missing.txt",
                        source_root=cwd_dir,
                        repo_root=fake_repo,
                    ))
                # Repo-root // path must stay inside repo_root.
                self.assertIsNone(
                    suite_to_harbor.resolve_path(
                        "//../secret.txt",
                        source_root=suite_dir,
                        repo_root=fake_repo,
                    ))
            finally:
                os.chdir(old_cwd)

    def test_bundle_verifiers_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src = tmp_path / "src"
            (src / "a").mkdir(parents=True)
            (src / "b").mkdir(parents=True)
            (src / "a" / "check.py").write_text("print('a')\n",
                                                encoding="utf-8")
            (src / "b" / "check.py").write_text("print('b')\n",
                                                encoding="utf-8")

            task_dir = tmp_path / "task"
            # Pre-existing verifier_files folder/file in tests/ reserves the name.
            (task_dir / "tests" / "checks").mkdir(parents=True)
            (task_dir / "tests" / "checks" / "marker.txt").write_text(
                "keep", encoding="utf-8")

            bundled = suite_to_harbor.bundle_script_verifiers(
                [schema.ScriptCheck(name="checks", script="a/check.py")],
                task_dir=task_dir,
                source_root=src,
            )
            self.assertEqual(bundled[0][0], "checks_2")
            self.assertTrue(
                (task_dir / "tests" / "checks_2" / "check.py").is_file())

            # Duplicate staged basenames within a single verifier must fail.
            with self.assertRaises(ValueError):
                suite_to_harbor.bundle_script_verifiers(
                    [
                        schema.ScriptCheck(
                            name="dup_scripts",
                            script=["a/check.py", "b/check.py"],
                        )
                    ],
                    task_dir=tmp_path / "dup1",
                    source_root=src,
                )
            with self.assertRaises(ValueError):
                suite_to_harbor.bundle_script_verifiers(
                    [
                        schema.ScriptCheck(
                            name="dup_dep",
                            script="a/check.py",
                            deps=["b/check.py"],
                        )
                    ],
                    task_dir=tmp_path / "dup2",
                    source_root=src,
                )

    def test_reward_normalization_and_dockerfile_root_workdir(self) -> None:
        for raw, expected in (
            ("weighted_mean", "weighted-mean"),
            ("weighted_sum", "weighted-sum"),
            ("all_pass", "all-pass"),
            ("any_pass", "any-pass"),
        ):
            parsed = tomllib.loads(
                template_generators.generate_reward_toml(aggregation=raw))
            self.assertEqual(parsed["reward"][0]["aggregation"], expected)

        df = template_generators.generate_dockerfile(
            schema.EnvironmentConfig(
                base_image="python:3.12-slim",
                pip_packages=["pytest", "pytest"],
                workdir="/",
            ),
            has_data=True,
        )
        self.assertIn("COPY data/ /", df)
        self.assertIn("pip install --no-cache-dir harbor-rewardkit pytest\n",
                      df)


if __name__ == "__main__":
    unittest.main()
