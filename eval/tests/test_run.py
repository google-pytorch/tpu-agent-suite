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
"""Unit tests for eval/run.py utility functions."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import ANY, MagicMock, patch

import yaml

from eval.mount_advisor import MountUpdates
from eval.staging import StagingResult

# Dynamically import eval/run.py
_RUN_PY_PATH = Path(__file__).resolve().parent.parent / "run.py"
_SPEC = importlib.util.spec_from_file_location("eval_run", _RUN_PY_PATH)
assert _SPEC and _SPEC.loader
_RUN_MOD = importlib.util.module_from_spec(_SPEC)

sys.modules["eval_run"] = _RUN_MOD
_SPEC.loader.exec_module(_RUN_MOD)

resolve_template = _RUN_MOD.resolve_template
resolve_config_files = _RUN_MOD.resolve_config_files
get_harbor_hub_url = _RUN_MOD.get_harbor_hub_url
load_config = _RUN_MOD.load_config
check_gemini_api_key = _RUN_MOD.check_gemini_api_key
check_task_requirements = _RUN_MOD.check_task_requirements
stage_package_if_needed = _RUN_MOD.stage_package_if_needed
apply_mount_updates = _RUN_MOD.apply_mount_updates
setup_vertex_if_needed = _RUN_MOD.setup_vertex_if_needed
export_git_ref = _RUN_MOD.export_git_ref
setup_gke_credentials_if_needed = _RUN_MOD.setup_gke_credentials_if_needed
build_parser = _RUN_MOD.build_parser
GkeCredentials = _RUN_MOD.GkeCredentials
check_job_scores = _RUN_MOD.check_job_scores
failed_criteria = _RUN_MOD.failed_criteria
CONFIGS_DIR = _RUN_MOD.CONFIGS_DIR
DATASETS_DIR = _RUN_MOD.DATASETS_DIR
require_dataset = _RUN_MOD.require_dataset
apply_dataset = _RUN_MOD.apply_dataset
resolve_task_arg = _RUN_MOD.resolve_task_arg
apply_tasks = _RUN_MOD.apply_tasks
task_specs = _RUN_MOD.task_specs
render_config = _RUN_MOD.render_config
run_dataset_setup_if_present = _RUN_MOD.run_dataset_setup_if_present
check_dependencies = _RUN_MOD.check_dependencies
setup_host_auth = _RUN_MOD.setup_host_auth
publish_github_step_summary = _RUN_MOD.publish_github_step_summary
HarborJobMonitor = _RUN_MOD.HarborJobMonitor
resolve_target_slug = _RUN_MOD.resolve_target_slug
resolve_sut_slug = _RUN_MOD.resolve_sut_slug
resolve_accelerator_slug = _RUN_MOD.resolve_accelerator_slug
resolve_env_slug = _RUN_MOD.resolve_env_slug
generate_job_name = _RUN_MOD.generate_job_name
SUITES_DIR = _RUN_MOD.SUITES_DIR
DOT_DATASETS_DIR = _RUN_MOD.DOT_DATASETS_DIR
available_suites = _RUN_MOD.available_suites
resolve_suite = _RUN_MOD.resolve_suite
compile_suite_if_needed = _RUN_MOD.compile_suite_if_needed


class TestResolveTemplate(unittest.TestCase):
    """Unit tests for template resolution (${VAR} and ${VAR:-default})."""

    def test_none_input(self) -> None:
        self.assertIsNone(resolve_template(None))

    def test_literal_string_unchanged(self) -> None:
        self.assertEqual(resolve_template("literal_value"), "literal_value")
        self.assertEqual(resolve_template("https://aiplatform.googleapis.com"),
                         "https://aiplatform.googleapis.com")

    def test_primitive_types_converted(self) -> None:
        self.assertEqual(resolve_template(123), "123")
        self.assertEqual(resolve_template(True), "True")

    def test_env_var_substitution(self) -> None:
        with patch.dict(os.environ, {"TEST_ENV_VAR": "my_val"}):
            self.assertEqual(resolve_template("${TEST_ENV_VAR}"), "my_val")

    def test_env_var_missing_returns_none(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_template("${NONEXISTENT_VAR}"))

    def test_env_var_with_default_used(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                resolve_template("${NONEXISTENT_VAR:-default_val}"),
                "default_val")

    def test_env_var_with_default_overridden(self) -> None:
        with patch.dict(os.environ, {"TEST_VAR": "custom_val"}):
            self.assertEqual(resolve_template("${TEST_VAR:-default_val}"),
                             "custom_val")

    def test_env_var_with_empty_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_template("${NONEXISTENT_VAR:-}"), "")


class TestExportHarnessPins(unittest.TestCase):
    """run.py exports .ci/pins.conf so the configs can install the pinned CLI."""

    def test_exports_the_claude_code_pin_without_overriding(self) -> None:
        pinned = _RUN_MOD.read_harness_pins(
            _RUN_MOD.HARNESS_PINS_FILE)["CLAUDE_CODE_VERSION"]
        with patch.dict(os.environ, {}, clear=True):
            _RUN_MOD.export_harness_pins()
            self.assertEqual(os.environ["CLAUDE_CODE_VERSION"], pinned)
        with patch.dict(os.environ, {"CLAUDE_CODE_VERSION": "0.0.1"}):
            _RUN_MOD.export_harness_pins()
            self.assertEqual(os.environ["CLAUDE_CODE_VERSION"], "0.0.1")


class TestResolveConfigFile(unittest.TestCase):
    """Unit tests for resolving configuration paths from CLI args."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_explicit_config_file_exists(self) -> None:
        cfg = self.temp_dir / "custom.yaml"
        cfg.write_text("job_name: test", encoding="utf-8")
        args = argparse.Namespace(config_file=cfg, package=None, config=None)
        self.assertEqual(resolve_config_files(args), [cfg.resolve()])

    def test_explicit_config_file_missing_raises(self) -> None:
        args = argparse.Namespace(config_file=self.temp_dir / "missing.yaml",
                                  package=None,
                                  config=None)
        with self.assertRaises(RuntimeError) as ctx:
            resolve_config_files(args)
        self.assertIn("Config file not found", str(ctx.exception))

    def test_config_name_resolves_from_configs_dir(self) -> None:
        args = argparse.Namespace(config_file=None,
                                  config="claude-code/agent-suite")
        resolved = resolve_config_files(args)
        self.assertEqual(len(resolved), 1)
        self.assertTrue(resolved[0].is_file())
        self.assertEqual(str(resolved[0].relative_to(CONFIGS_DIR)),
                         "claude-code/agent-suite.yaml")
        self.assertEqual(resolved[0].parent, CONFIGS_DIR / "claude-code")

    def test_config_missing_raises_and_lists_available(self) -> None:
        args = argparse.Namespace(config_file=None,
                                  config="nonexistent_config")
        with self.assertRaises(RuntimeError) as ctx:
            resolve_config_files(args)
        self.assertIn("Config 'nonexistent_config' not found",
                      str(ctx.exception))
        self.assertIn("Available configs:", str(ctx.exception))

    def test_no_args_raises_usage(self) -> None:
        args = argparse.Namespace(config_file=None,
                                  config=None,
                                  positional_args=[])
        with self.assertRaises(RuntimeError) as ctx:
            resolve_config_files(args)
        self.assertIn("Missing configuration", str(ctx.exception))

    def test_positional_config_file_exists(self) -> None:
        cfg = self.temp_dir / "positional.yaml"
        cfg.write_text("job_name: test", encoding="utf-8")
        args = argparse.Namespace(config_file=None,
                                  config=None,
                                  positional_args=[str(cfg)])
        self.assertEqual(resolve_config_files(args), [cfg.resolve()])

    def test_positional_config_name(self) -> None:
        args = argparse.Namespace(config_file=None,
                                  config=None,
                                  positional_args=["claude-code/agent-suite"])
        resolved = resolve_config_files(args)
        self.assertEqual([str(p.relative_to(CONFIGS_DIR)) for p in resolved],
                         ["claude-code/agent-suite.yaml"])

    def test_positional_unknown_config_raises(self) -> None:
        args = argparse.Namespace(config_file=None,
                                  config=None,
                                  positional_args=["tpu-hello-world"])
        with self.assertRaises(RuntimeError) as ctx:
            resolve_config_files(args)
        self.assertIn("Config 'tpu-hello-world' not found", str(ctx.exception))


class TestConfigsNameNoDataset(unittest.TestCase):
    """A config names the system under test and nothing else.

    Harbor runs a config with neither tasks nor datasets and reports success
    over an empty set, so a dataset key silently reintroduced into a config
    would make `--dataset` optional again without anything failing.
    """

    def test_no_shipped_config_selects_a_dataset(self) -> None:
        offenders = []
        for cfg in sorted(CONFIGS_DIR.glob("*.yaml")):
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            for key in ("tasks", "datasets"):
                if data.get(key):
                    offenders.append(f"{cfg.name}:{key}")
        self.assertEqual(offenders, [],
                         f"configs must not select a dataset: {offenders}")

    def test_every_shipped_config_requires_a_dataset(self) -> None:
        for cfg in sorted(CONFIGS_DIR.glob("*.yaml")):
            with self.subTest(config=cfg.name):
                data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
                with self.assertRaises(RuntimeError) as ctx:
                    require_dataset(data, cfg)
                self.assertIn("No dataset selected", str(ctx.exception))


class TestCheckTaskRequirements(unittest.TestCase):
    """Preflight on task.toml [metadata].required_env."""

    GATED = '[metadata]\nrequired_env = ["HF_TOKEN"]\n'

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _task(self, name: str, metadata: str = "") -> Path:
        task_dir = self.root / name
        task_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text(
            f'schema_version = "1.4"\n[task]\nname = "t/{name}"\n'
            f"[agent]\ntimeout_sec = 28800.0\n[verifier]\ntimeout_sec = 60.0\n"
            f"{metadata}",
            encoding="utf-8")
        return task_dir

    def test_dataset_skips_gated_task_and_keeps_the_rest(self) -> None:
        # GitHub exports a missing secret as "", and ${HF_TOKEN:-} forwards
        # "" rather than failing: the empty case is the one that matters.
        self._task("ds/gated", metadata=self.GATED)
        self._task("ds/plain")
        self._task("ds/old-llama", metadata=self.GATED)  # already excluded
        entry = {
            "path": str(self.root / "ds"),
            "exclude_task_names": ["*llama*"]
        }
        with patch.dict(os.environ, {"HF_TOKEN": ""}, clear=True), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            skipped = check_task_requirements({"datasets": [entry]})
        self.assertEqual(entry["exclude_task_names"], ["*llama*", "gated"])
        self.assertEqual(len(skipped), 1)
        self.assertIn("gated skipped: HF_TOKEN unset or empty", skipped[0])
        self.assertIn("WARNING", err.getvalue())

    def test_dataset_with_nothing_left_refuses(self) -> None:
        self._task("ds/gated", metadata=self.GATED)
        with patch.dict(os.environ, {}, clear=True), \
                contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(RuntimeError) as ctx:
            check_task_requirements(
                {"datasets": [{
                    "path": str(self.root / "ds")
                }]})
        self.assertIn("Nothing left to run", str(ctx.exception))

    def test_explicit_task_refuses(self) -> None:
        gated = self._task("gated", metadata=self.GATED)
        plain = self._task("plain")
        tasks = [{"path": str(gated)}, {"path": str(plain)}]
        with patch.dict(os.environ, {}, clear=True), \
                self.assertRaises(RuntimeError) as ctx:
            check_task_requirements({"tasks": tasks})
        self.assertIn("gated requires HF_TOKEN", str(ctx.exception))

    def test_satisfied_is_a_no_op(self) -> None:
        self._task("ds/gated", metadata=self.GATED)
        self._task("ds/plain")
        entry = {"path": str(self.root / "ds")}
        with patch.dict(os.environ, {"HF_TOKEN": "hf_x"}, clear=True):
            self.assertEqual(check_task_requirements({"datasets": [entry]}),
                             [])
        self.assertNotIn("exclude_task_names", entry)

    def test_shipped_gated_onboarding_tasks_require_token(self) -> None:
        gated = {
            "gemma-3-27b-it",
            "llama-3-3-70b",
        }
        import tomllib
        for task_toml in sorted(DATASETS_DIR.glob("**/task.toml")):
            if task_toml.parent.name not in gated:
                continue
            data = tomllib.loads(task_toml.read_text(encoding="utf-8"))
            meta = data.get("metadata") or {}
            self.assertIn("HF_TOKEN", meta.get("required_env", []), task_toml)


class TestStageBaseDigests(unittest.TestCase):
    """A task FROM nexus-tpu:latest gets a new image tag when :latest moves."""

    def test_digest_moves_the_tag_only_for_latest_based_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, base in (
                ("overlay", "--platform=linux/amd64 us-docker.pkg.dev/p/"
                 "nexus-dev/nexus-tpu:latest"),
                ("pinned",
                 "us-docker.pkg.dev/p/nexus-dev/nexus-tpu@sha256:" + "0" * 64),
            ):
                (root / name / "environment").mkdir(parents=True)
                (root / name / "task.toml").write_text(
                    f'[task]\nname = "ds/{name}"\n', encoding="utf-8")
                (root / name / "environment" / "Dockerfile").write_text(
                    f"FROM {base}\n", encoding="utf-8")
            kwargs = {
                "registry_name": "r",
                "registry_location": "us",
                "project_id": "p"
            }
            config = {"environment": {"kwargs": kwargs}, "datasets": [tmp]}

            def tag() -> str | None:
                return _RUN_MOD.harbor_image_tag(root / "overlay",
                                                 "environment", kwargs)

            before = tag()
            describe = MagicMock(stdout="sha256:aaa\n")
            with patch.object(_RUN_MOD.subprocess, "run",
                              return_value=describe) as run_cmd, \
                    patch.object(_RUN_MOD, "get_adc_path",
                                 return_value=root / "no-adc.json"):
                _RUN_MOD.stage_base_digests(config)
                first = tag()
                describe.stdout = "sha256:bbb\n"
                _RUN_MOD.stage_base_digests(config)
            self.assertNotEqual(before, first)
            self.assertNotEqual(first, tag())
            self.assertFalse((root / "pinned" / "environment" /
                              _RUN_MOD.BASE_DIGEST_FILE).exists())
            # A GKE config builds with --pull; only a local config pulls here.
            self.assertNotIn("pull",
                             [c.args[0][1] for c in run_cmd.call_args_list])


class TestLoadConfig(unittest.TestCase):
    """Unit tests for YAML config loading."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_valid_yaml(self) -> None:
        cfg = self.temp_dir / "test.yaml"
        cfg.write_text("job_name: sample_job\nagents:\n  - name: test_agent\n",
                       encoding="utf-8")
        data = load_config(cfg)
        self.assertEqual(data["job_name"], "sample_job")
        self.assertEqual(len(data["agents"]), 1)

    def test_load_empty_yaml_returns_dict(self) -> None:
        cfg = self.temp_dir / "empty.yaml"
        cfg.write_text("", encoding="utf-8")
        self.assertEqual(load_config(cfg), {})


class TestCheckGeminiApiKey(unittest.TestCase):
    """Unit tests for GEMINI_API_KEY preflight validation."""

    def test_gemini_not_needed_passes(self) -> None:
        config = {
            "agents": [{
                "name": "anthropic/claude-opus-5",
                "model_name": "claude"
            }],
            "verifier": {},
        }
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise
            check_gemini_api_key(config)

    def test_agy_ci_on_metadata_auth_needs_no_key(self) -> None:
        # CI runs agy/agent-suite-remote.yaml under GKE_AUTH_ACCOUNT=metadata:
        # ADC from the metadata server, with no ADC file and no API key.
        config = load_config(CONFIGS_DIR / "agy" / "agent-suite-remote.yaml")
        self.assertEqual(config["environment"]["env"]["AGY_ADC_AUTH"], "true")
        with patch.dict(os.environ, {"GKE_AUTH_ACCOUNT": "metadata"},
                        clear=True), \
                patch.object(_RUN_MOD, "get_adc_path",
                             return_value=Path("/nonexistent/adc.json")):
            check_gemini_api_key(config)

    def test_gemini_in_model_name_missing_key_raises(self) -> None:
        config = {
            "agents": [{
                "model_name": "gemini/gemini-2.0-flash"
            }],
        }
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                check_gemini_api_key(config)
            self.assertIn("GEMINI_API_KEY environment variable is not set",
                          str(ctx.exception))

    def test_gemini_in_agent_env_missing_key_raises(self) -> None:
        config = {
            "agents": [{
                "model_name": "other",
                "env": {
                    "GEMINI_API_KEY": "${GEMINI_API_KEY}"
                }
            }],
        }
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                check_gemini_api_key(config)
            self.assertIn("GEMINI_API_KEY environment variable is not set",
                          str(ctx.exception))

    def test_gemini_in_verifier_env_missing_key_raises(self) -> None:
        config = {
            "verifier": {
                "env": {
                    "GEMINI_API_KEY": "${GEMINI_API_KEY}"
                }
            },
        }
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                check_gemini_api_key(config)
            self.assertIn("GEMINI_API_KEY environment variable is not set",
                          str(ctx.exception))

    def test_gemini_present_passes(self) -> None:
        config = {
            "agents": [{
                "model_name": "gemini/gemini-2.0-flash"
            }],
            "verifier": {
                "env": {
                    "GEMINI_API_KEY": "${GEMINI_API_KEY}"
                }
            },
        }
        with patch.dict(os.environ, {"GEMINI_API_KEY": "valid-api-key"}):
            check_gemini_api_key(config)


class TestSetupVertexIfNeeded(unittest.TestCase):
    """Unit tests for Vertex AI endpoint setup and environment sanitization."""

    def test_vertex_not_needed_no_op(self) -> None:
        config = {"agents": [{"env": {"FOO": "BAR"}}]}
        with patch.dict(os.environ, {}, clear=True):
            setup_vertex_if_needed(config)

    def test_missing_project_raises(self) -> None:
        config = {"agents": [{"env": {"CLAUDE_CODE_USE_VERTEX": "1"}}]}
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                setup_vertex_if_needed(config)
            self.assertIn("VERTEXAI_PROJECT", str(ctx.exception))

    def test_strips_anthropic_direct_api_keys(self) -> None:
        config = {
            "agents": [{
                "env": {
                    "CLAUDE_CODE_USE_VERTEX": "1",
                    "VERTEXAI_PROJECT": "my-project",
                }
            }]
        }
        env_setup = {
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "ANTHROPIC_AUTH_TOKEN": "token-test",
            "CLAUDE_CODE_OAUTH_TOKEN": "oauth-test",
        }
        with patch.dict(os.environ, env_setup):
            with patch.object(_RUN_MOD,
                              "get_adc_path",
                              return_value=Path(__file__)):
                setup_vertex_if_needed(config)
                self.assertNotIn("ANTHROPIC_API_KEY", os.environ)
                self.assertNotIn("ANTHROPIC_AUTH_TOKEN", os.environ)
                self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", os.environ)


class TestExportGitRef(unittest.TestCase):
    """The repo commit a trial lock records."""

    def _run(self, stdouts: dict[str, str], **env: str) -> str:

        def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
            key = " ".join(cmd[3:])
            return subprocess.CompletedProcess(cmd,
                                               0,
                                               stdout=stdouts.get(key, ""))

        with patch.dict(os.environ, env, clear=False):
            if "EVAL_GIT_COMMIT" not in env:
                os.environ.pop("EVAL_GIT_COMMIT", None)
            with patch.object(_RUN_MOD.subprocess, "run",
                              side_effect=fake_run) as run:
                export_git_ref()
            self.calls = run.call_count
            return os.environ["EVAL_GIT_COMMIT"]

    def test_clean_tree(self) -> None:
        self.assertEqual(
            self._run({
                "rev-parse HEAD": "abc123\n",
                "status --porcelain --untracked-files=no": "",
            }), "abc123")

    def test_dirty_tree_is_marked(self) -> None:
        self.assertEqual(
            self._run({
                "rev-parse HEAD":
                "abc123",
                "status --porcelain --untracked-files=no":
                " M eval/run.py",
            }), "abc123-dirty")

    def test_no_branch_reaches_the_agent(self) -> None:
        """A branch name can name the product to the baseline agent."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EVAL_GIT_BRANCH", None)
            self._run({"rev-parse HEAD": "abc123"})
            self.assertNotIn("EVAL_GIT_BRANCH", os.environ)
        for config in (_RUN_MOD.REPO_ROOT / "eval" /
                       "configs").rglob("*.yaml"):
            self.assertNotIn("EVAL_GIT_BRANCH",
                             config.read_text(encoding="utf-8"), config)

    def test_untracked_files_do_not_mark_tree_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run([
                "git", "-C",
                str(repo), "-c", "user.name=Test", "-c",
                "user.email=test@example.com", "commit", "--allow-empty",
                "-qm", "initial"
            ],
                           check=True)
            (repo / "scratch.txt").write_text("scratch")
            with patch.object(_RUN_MOD, "REPO_ROOT",
                              repo), patch.dict(os.environ, {}, clear=False):
                os.environ.pop("EVAL_GIT_COMMIT", None)
                export_git_ref()
                self.assertFalse(
                    os.environ["EVAL_GIT_COMMIT"].endswith("-dirty"))

    def test_a_commit_already_set_wins_without_asking_git(self) -> None:
        """CI sets the PR head; the checkout is a synthetic merge commit."""
        self.assertEqual(
            self._run({"rev-parse HEAD": "synthetic-merge"},
                      EVAL_GIT_COMMIT="pr-head-sha"), "pr-head-sha")
        self.assertEqual(self.calls, 0)

    def test_missing_git_is_not_fatal(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EVAL_GIT_COMMIT", None)
            with patch.object(_RUN_MOD.subprocess,
                              "run",
                              side_effect=FileNotFoundError):
                export_git_ref()
            self.assertEqual(os.environ["EVAL_GIT_COMMIT"], "")


class TestSetupGkeCredentialsIfNeeded(unittest.TestCase):
    """Unit tests for GKE kubeconfig minting and PyYAML token embedding."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.adc_file = self.temp_dir / "adc.json"
        self.adc_file.write_text("{}", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_no_kubeconfig_returns_none(self) -> None:
        config = {"agents": [{"env": {"SOME_VAR": "val"}}]}
        res = setup_gke_credentials_if_needed(config, self.temp_dir)
        self.assertIsNone(res)

    def test_local_backend_skips_gke_credentials(self) -> None:
        config = {
            "agents": [{
                "env": {
                    "COMPUTE_BACKEND": "local",
                    "KUBECONFIG": "/root/.kube/config",
                }
            }]
        }
        res = setup_gke_credentials_if_needed(config, self.temp_dir)
        self.assertIsNone(res)

    def test_gce_tpu_vm_backend_skips_gke_credentials(self) -> None:
        config = {
            "agents": [{
                "env": {
                    "COMPUTE_BACKEND": "gce_tpu_vm",
                    "KUBECONFIG": "/root/.kube/config",
                }
            }]
        }
        res = setup_gke_credentials_if_needed(config, self.temp_dir)
        self.assertIsNone(res)

    def test_kubeconfig_missing_cluster_raises(self) -> None:
        config = {"agents": [{"env": {"KUBECONFIG": "/root/.kube/config"}}]}
        with self.assertRaises(RuntimeError) as ctx:
            setup_gke_credentials_if_needed(config, self.temp_dir)
        self.assertIn("GKE_CLUSTER is not set", str(ctx.exception))

    def test_kubeconfig_missing_project_or_location_raises(self) -> None:
        config = {
            "agents": [{
                "env": {
                    "KUBECONFIG": "/root/.kube/config",
                    "GKE_CLUSTER": "my-cluster",
                }
            }]
        }
        with self.assertRaises(RuntimeError) as ctx:
            setup_gke_credentials_if_needed(config, self.temp_dir)
        self.assertIn("requires GKE_PROJECT and GKE_LOCATION",
                      str(ctx.exception))

    @patch("builtins.print")
    @patch("subprocess.run")
    def test_successful_kubeconfig_adc_exec_plugin(
            self, mock_run: MagicMock, mock_print: MagicMock) -> None:
        config = {
            "agents": [{
                "env": {
                    "KUBECONFIG": "/root/.kube/config",
                    "GKE_CLUSTER": "my-cluster",
                    "GKE_PROJECT": "my-project",
                    "GKE_LOCATION": "us-central1",
                    # Every config defaults to ADC.
                    "GKE_AUTH_ACCOUNT": "adc",
                }
            }]
        }

        kubeconfig_file = self.temp_dir / "kubeconfig"
        raw_kube_content = {
            "apiVersion":
            "v1",
            "users": [{
                "name": "gke_user",
                "user": {
                    "exec": {
                        "command": "gke-gcloud-auth-plugin",
                        "args": ["--use_application_default_credentials"],
                    }
                },
            }],
        }

        def fake_run(cmd, *args, **kwargs):
            self.assertNotIn("print-access-token", cmd)
            if "get-credentials" in cmd:
                env = kwargs["env"]
                self.assertEqual(
                    env["CLOUDSDK_CONTAINER_USE_APPLICATION_DEFAULT_CREDENTIALS"],
                    "true",
                )
                self.assertEqual(
                    env["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"],
                    str(self.adc_file),
                )
                kubeconfig_file.write_text(yaml.safe_dump(raw_kube_content),
                                           encoding="utf-8")
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_run

        with patch.object(_RUN_MOD, "get_adc_path",
                          return_value=self.adc_file):
            creds = setup_gke_credentials_if_needed(config, self.temp_dir)
        self.assertIsNotNone(creds)
        self.assertEqual(creds.kubeconfig_path, kubeconfig_file)

        # Verify that the gke-gcloud-auth-plugin exec block is preserved intact
        updated_data = yaml.safe_load(
            kubeconfig_file.read_text(encoding="utf-8"))
        user_obj = updated_data["users"][0]["user"]
        self.assertIn("exec", user_obj)
        self.assertEqual(user_obj["exec"]["command"], "gke-gcloud-auth-plugin")
        self.assertEqual(user_obj["exec"]["args"],
                         ["--use_application_default_credentials"])
        self.assertNotIn("token", user_obj)

    @patch("builtins.print")
    @patch("subprocess.run")
    def test_metadata_auth_uses_ambient_metadata_server(
            self, mock_run: MagicMock, mock_print: MagicMock) -> None:
        config = {
            "agents": [{
                "env": {
                    "KUBECONFIG": "/root/.kube/config",
                    "GKE_CLUSTER": "my-cluster",
                    "GKE_PROJECT": "my-project",
                    "GKE_LOCATION": "us-central1",
                    "GKE_AUTH_ACCOUNT": "metadata",
                }
            }]
        }
        kubeconfig_file = self.temp_dir / "kubeconfig"

        def fake_run(cmd, *args, **kwargs):
            self.assertNotIn("print-access-token", cmd)
            if "get-credentials" in cmd:
                env = kwargs["env"]
                self.assertEqual(
                    env["CLOUDSDK_CONTAINER_USE_APPLICATION_DEFAULT_CREDENTIALS"],
                    "true",
                )
                self.assertNotIn("CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE", env)
                self.assertNotIn("CLOUDSDK_AUTH_ACCESS_TOKEN", env)
                kubeconfig_file.write_text(yaml.safe_dump(
                    {"users": [{
                        "name": "u",
                        "user": {}
                    }]}),
                                           encoding="utf-8")
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_run
        with patch.dict(os.environ, {}, clear=False), \
             patch.object(_RUN_MOD, "metadata_access_token") as mock_token, \
             patch.object(_RUN_MOD, "metadata_server_get",
                          return_value="sa@proj.iam.gserviceaccount.com"):
            os.environ.pop("GKE_AUTH_KEY_FILE", None)
            os.environ.pop("GKE_IMPERSONATE_SA", None)
            creds = setup_gke_credentials_if_needed(config, self.temp_dir)
        mock_token.assert_not_called()
        assert creds is not None
        self.assertEqual(creds.kubeconfig_path, kubeconfig_file)

    @patch("builtins.print")
    @patch("subprocess.run")
    def test_metadata_auth_uses_key_file_when_set(
            self, mock_run: MagicMock, mock_print: MagicMock) -> None:
        config = {
            "agents": [{
                "env": {
                    "KUBECONFIG": "/root/.kube/config",
                    "GKE_CLUSTER": "my-cluster",
                    "GKE_PROJECT": "my-project",
                    "GKE_LOCATION": "us-central1",
                    "GKE_AUTH_ACCOUNT": "metadata",
                }
            }]
        }
        key_file = self.temp_dir / "key.json"
        key_file.write_text(json.dumps({"client_email": "key@p.iam"}),
                            encoding="utf-8")
        kubeconfig_file = self.temp_dir / "kubeconfig"

        def fake_run(cmd, *args, **kwargs):
            self.assertNotIn("print-access-token", cmd)
            if "get-credentials" in cmd:
                self.assertEqual(
                    kwargs["env"]["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"],
                    str(key_file))
                kubeconfig_file.write_text(yaml.safe_dump(
                    {"users": [{
                        "name": "u",
                        "user": {}
                    }]}),
                                           encoding="utf-8")
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_run
        with patch.dict(os.environ, {"GKE_AUTH_KEY_FILE": str(key_file)}), \
             patch.object(_RUN_MOD, "metadata_access_token") as mock_token:
            creds = setup_gke_credentials_if_needed(config, self.temp_dir)
        mock_token.assert_not_called()
        assert creds is not None
        self.assertEqual(creds.kubeconfig_path, kubeconfig_file)

    @patch("builtins.print")
    @patch("subprocess.run")
    def test_metadata_auth_impersonates_when_gke_impersonate_sa_set(
            self, mock_run: MagicMock, mock_print: MagicMock) -> None:
        config = {
            "agents": [{
                "env": {
                    "KUBECONFIG": "/root/.kube/config",
                    "GKE_CLUSTER": "bodaborg-tpu7x-nap",
                    "GKE_PROJECT": "cloud-tpu-shared-capacity",
                    "GKE_LOCATION": "us-central1",
                    "GKE_AUTH_ACCOUNT": "metadata",
                }
            }]
        }
        sa = "harbor-ci-runner@tpu-nexus-767207.iam.gserviceaccount.com"
        kubeconfig_file = self.temp_dir / "kubeconfig"

        def fake_run(cmd, *args, **kwargs):
            self.assertNotIn("print-access-token", cmd)
            if "get-credentials" in cmd:
                env = kwargs["env"]
                self.assertEqual(
                    env["CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT"], sa)
                self.assertNotIn("CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE", env)
                kubeconfig_file.write_text(
                    yaml.safe_dump({"users": [{
                        "name": "u",
                        "user": {}
                    }]}),
                    encoding="utf-8",
                )
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_run
        with patch.dict(os.environ, {"GKE_IMPERSONATE_SA": sa}, clear=False), \
             patch.object(_RUN_MOD, "metadata_access_token") as mock_token:
            os.environ.pop("GKE_AUTH_KEY_FILE", None)
            creds = setup_gke_credentials_if_needed(config, self.temp_dir)
        mock_token.assert_not_called()
        assert creds is not None
        self.assertEqual(creds.kubeconfig_path, kubeconfig_file)

    def _gke_config(self, **extra: str) -> dict:
        return {
            "agents": [{
                "env": {
                    "KUBECONFIG": "/root/.kube/config",
                    "GKE_CLUSTER": "my-cluster",
                    "GKE_PROJECT": "my-project",
                    "GKE_LOCATION": "us-central1",
                    **extra,
                }
            }]
        }

    def _record_gcloud(self, mock_run: MagicMock) -> list:
        calls = []
        kubeconfig_file = self.temp_dir / "kubeconfig"

        def fake_run(cmd, *args, **kwargs):
            calls.append((list(cmd), kwargs.get("env") or {}))
            if "get-credentials" in cmd:
                kubeconfig_file.write_text(yaml.safe_dump({"users": []}),
                                           encoding="utf-8")
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_run
        return calls

    @patch("builtins.print")
    @patch("subprocess.run")
    def test_dns_endpoint_is_opt_in(self, mock_run: MagicMock,
                                    mock_print: MagicMock) -> None:
        for value, expected in (("1", True), ("", False)):
            with self.subTest(GKE_DNS_ENDPOINT=value), patch.dict(
                    os.environ, {"GKE_DNS_ENDPOINT": value}):
                calls = self._record_gcloud(mock_run)
                setup_gke_credentials_if_needed(self._gke_config(),
                                                self.temp_dir)
                get_creds = [c for c, _ in calls if "get-credentials" in c]
                self.assertEqual(len(get_creds), 1)
                self.assertEqual("--dns-endpoint" in get_creds[0], expected)

    @patch("builtins.print")
    @patch("subprocess.run")
    def test_adc_ignores_a_container_credentials_path(
            self, mock_run: MagicMock, mock_print: MagicMock) -> None:
        # env.txt sets GOOGLE_APPLICATION_CREDENTIALS to the container's path
        # for sandbox.sh; the host has no such file, and gcloud honours the
        # variable over the ADC run.py found.
        adc = self.temp_dir / "adc.json"
        adc.write_text("{}", encoding="utf-8")
        calls = self._record_gcloud(mock_run)
        with patch.dict(
                os.environ, {
                    "EVAL_VERTEX_ADC": str(adc),
                    "GOOGLE_APPLICATION_CREDENTIALS":
                    "/root/.config/gcloud/adc.json",
                }):
            setup_gke_credentials_if_needed(
                self._gke_config(GKE_AUTH_ACCOUNT="adc"), self.temp_dir)
        creds_envs = [env for c, env in calls if "get-credentials" in c]
        self.assertEqual(len(creds_envs), 1)
        self.assertEqual(
            creds_envs[0]["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"], str(adc))

    @patch("builtins.print")
    @patch("subprocess.run")
    def test_adc_keeps_a_credentials_file_that_exists(
            self, mock_run: MagicMock, mock_print: MagicMock) -> None:
        # CI points GOOGLE_APPLICATION_CREDENTIALS at a service account key
        # that can reach the cluster; the ADC there is for Vertex and gets a
        # 403 on container.clusters.get.
        adc = self.temp_dir / "adc.json"
        adc.write_text("{}", encoding="utf-8")
        key = self.temp_dir / "sa-key.json"
        key.write_text("{}", encoding="utf-8")
        calls = self._record_gcloud(mock_run)
        with patch.dict(
                os.environ, {
                    "EVAL_VERTEX_ADC": str(adc),
                    "GOOGLE_APPLICATION_CREDENTIALS": str(key),
                }):
            setup_gke_credentials_if_needed(
                self._gke_config(GKE_AUTH_ACCOUNT="adc"), self.temp_dir)
        creds_envs = [env for c, env in calls if "get-credentials" in c]
        self.assertEqual(len(creds_envs), 1)
        self.assertEqual(
            creds_envs[0]["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"], str(key))


class TestSetupHarborGkeKubeconfigIfNeeded(unittest.TestCase):
    """Unit tests for the kubeconfig Harbor runs trial pods with."""

    CONFIG = {
        "environment": {
            "type": "gke",
            "kwargs": {
                "cluster_name": "harbor-nexus-cluster",
                "region": "us-central1",
                "project_id": "tpu-nexus-767207",
            },
        }
    }

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _fake_get_credentials(self, expect_adc_override: bool):

        def fake_run(cmd, *args, **kwargs):
            self.assertNotIn("print-access-token", cmd)
            if "get-credentials" in cmd:
                env = kwargs["env"]
                self.assertNotIn("CLOUDSDK_AUTH_ACCESS_TOKEN", env)
                if expect_adc_override:
                    self.assertIn("CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE",
                                  env)
                else:
                    self.assertNotIn("CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE",
                                     env)
                Path(env["KUBECONFIG"]).write_text(yaml.safe_dump({
                    "users": [{
                        "name": "gke",
                        "user": {
                            "exec": {
                                "command": "gke-gcloud-auth-plugin"
                            }
                        },
                    }]
                }),
                                                   encoding="utf-8")
            return MagicMock(returncode=0)

        return fake_run

    def test_not_gke_returns_none(self) -> None:
        self.assertIsNone(
            _RUN_MOD.setup_harbor_gke_kubeconfig_if_needed(
                {"environment": {
                    "type": "docker"
                }}, self.temp_dir))

    @patch("builtins.print")
    def test_adc_writes_refreshing_gcp_auth_provider(
            self, mock_print: MagicMock) -> None:
        with patch.dict(os.environ, {"GKE_AUTH_ACCOUNT": "adc"}, clear=False), \
             patch("subprocess.run",
                   side_effect=self._fake_get_credentials(True)):
            os.environ.pop("HARBOR_IMPERSONATE_SA", None)
            path = _RUN_MOD.setup_harbor_gke_kubeconfig_if_needed(
                self.CONFIG, self.temp_dir)
        self.assertEqual(path, self.temp_dir / "harbor_kubeconfig")
        assert path is not None
        users = yaml.safe_load(path.read_text(encoding="utf-8"))["users"]
        # No baked-in token to expire: the client refreshes it itself.
        self.assertEqual(users[0]["user"], {"auth-provider": {"name": "gcp"}})

    @patch("builtins.print")
    def test_metadata_writes_refreshing_gcp_auth_provider(
            self, mock_print: MagicMock) -> None:
        with patch.dict(os.environ, {"GKE_AUTH_ACCOUNT": "metadata"}, clear=False), \
             patch("subprocess.run",
                   side_effect=self._fake_get_credentials(False)):
            os.environ.pop("HARBOR_IMPERSONATE_SA", None)
            path = _RUN_MOD.setup_harbor_gke_kubeconfig_if_needed(
                self.CONFIG, self.temp_dir)
        assert path is not None
        users = yaml.safe_load(path.read_text(encoding="utf-8"))["users"]
        self.assertEqual(users[0]["user"], {"auth-provider": {"name": "gcp"}})

    @patch("builtins.print")
    def test_harbor_impersonate_sa_writes_refreshing_gcp_auth_provider(
            self, mock_print: MagicMock) -> None:
        sa = "141197224043-compute@developer.gserviceaccount.com"

        def fake_run(cmd, *args, **kwargs):
            self.assertNotIn("print-access-token", cmd)
            if "get-credentials" in cmd:
                env = kwargs["env"]
                self.assertEqual(
                    env["CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT"], sa)
                self.assertNotIn("CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE", env)
                self.assertNotIn("CLOUDSDK_AUTH_ACCESS_TOKEN", env)
                Path(env["KUBECONFIG"]).write_text(
                    yaml.safe_dump({
                        "users": [{
                            "name": "gke",
                            "user": {
                                "exec": {
                                    "command": "gke-gcloud-auth-plugin"
                                }
                            },
                        }]
                    }),
                    encoding="utf-8",
                )
            return MagicMock(returncode=0)

        with patch.dict(
                os.environ,
            {
                "GKE_AUTH_ACCOUNT": "metadata",
                "HARBOR_IMPERSONATE_SA": sa,
            },
                clear=False,
        ), patch("subprocess.run", side_effect=fake_run):
            path = _RUN_MOD.setup_harbor_gke_kubeconfig_if_needed(
                self.CONFIG, self.temp_dir)
        assert path is not None
        users = yaml.safe_load(path.read_text(encoding="utf-8"))["users"]
        self.assertEqual(
            users[0]["user"],
            {
                "auth-provider": {
                    "name": "gcp",
                    "config": {
                        "cmd-path":
                        "gcloud",
                        "cmd-args": ("config config-helper --format=json"
                                     f" --impersonate-service-account={sa}"),
                        "token-key":
                        "{.credential.access_token}",
                        "expiry-key":
                        "{.credential.token_expiry}",
                    },
                }
            },
        )


class TestDropHostAdc(unittest.TestCase):
    """Unit tests for removing the host ADC from metadata-auth runs."""

    def test_drops_adc_mount_and_env(self) -> None:
        adc_target = "/root/.config/gcloud/application_default_credentials.json"
        config = {
            "environment": {
                "mounts": [
                    {
                        "type": "bind",
                        "source": "${HOST_ADC}",
                        "target": adc_target
                    },
                    {
                        "type": "bind",
                        "source": "${EVAL_KUBECONFIG}",
                        "target": "/root/.kube/config"
                    },
                ],
            },
            "agents": [{
                "env": {
                    "GOOGLE_APPLICATION_CREDENTIALS": adc_target,
                    "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE": adc_target,
                    "KUBECONFIG": "/root/.kube/config",
                }
            }],
        }
        with patch("builtins.print"):
            _RUN_MOD.drop_host_adc(config)
        self.assertEqual(
            [m["source"] for m in config["environment"]["mounts"]],
            ["${EVAL_KUBECONFIG}"])
        self.assertEqual(config["agents"][0]["env"],
                         {"KUBECONFIG": "/root/.kube/config"})


class TestBuildParser(unittest.TestCase):
    """Unit tests for command-line parser defaults and upload flags."""

    def test_default_upload_is_true(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["-c", "cfg"])
        self.assertTrue(args.upload)

    def test_no_upload_flag_disables_upload(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["-c", "cfg", "--no-upload"])
        self.assertFalse(args.upload)

    def test_explicit_upload_flag_enables_upload(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["-c", "cfg", "--upload"])
        self.assertTrue(args.upload)

    def test_default_share_org_is_google_pytorch(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["-c", "cfg"])
        self.assertEqual(args.share_org, "google-pytorch")

    def test_custom_share_org(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["-c", "cfg", "--share-org", "custom-org"])
        self.assertEqual(args.share_org, "custom-org")

    def test_default_min_score_is_one(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["-c", "cfg"])
        self.assertEqual(args.min_score, 1.0)

    def test_custom_min_score(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["-c", "cfg", "--min-score", "0.8"])
        self.assertEqual(args.min_score, 0.8)


class TestCheckJobScores(unittest.TestCase):
    """Unit tests for check_job_scores."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.job_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_missing_result_json_returns_false(self) -> None:
        passed, summaries = check_job_scores(self.job_dir)
        self.assertFalse(passed)
        self.assertEqual(summaries, [])

    def test_all_trials_passing(self) -> None:
        job_result = {
            "trial_results": [
                {
                    "trial_name": "trial_1",
                    "verifier_result": {
                        "rewards": {
                            "reward": 1.0
                        }
                    },
                },
                {
                    "trial_name": "trial_2",
                    "verifier_result": {
                        "rewards": {
                            "reward": 1.0
                        }
                    },
                },
            ]
        }
        (self.job_dir / "result.json").write_text(json.dumps(job_result))
        passed, summaries = check_job_scores(self.job_dir, min_score=1.0)
        self.assertTrue(passed)
        self.assertEqual(len(summaries), 2)
        self.assertTrue(all(s["passed"] for s in summaries))

    def test_trial_less_than_one_fails(self) -> None:
        job_result = {
            "trial_results": [
                {
                    "trial_name": "trial_1",
                    "verifier_result": {
                        "rewards": {
                            "reward": 1.0
                        }
                    },
                },
                {
                    "trial_name": "trial_2",
                    "verifier_result": {
                        "rewards": {
                            "reward": 0.8
                        }
                    },
                },
            ]
        }
        (self.job_dir / "result.json").write_text(json.dumps(job_result))
        passed, summaries = check_job_scores(self.job_dir, min_score=1.0)
        self.assertFalse(passed)
        self.assertEqual(len(summaries), 2)
        self.assertTrue(summaries[0]["passed"])
        self.assertFalse(summaries[1]["passed"])

    def test_per_trial_subdirectories_fallback(self) -> None:
        (self.job_dir / "result.json").write_text(json.dumps({}))
        t1_dir = self.job_dir / "t1"
        t1_dir.mkdir()
        (t1_dir / "result.json").write_text(
            json.dumps({
                "trial_name": "t1",
                "verifier_result": {
                    "rewards": {
                        "reward": 1.0
                    }
                }
            }))
        passed, summaries = check_job_scores(self.job_dir, min_score=1.0)
        self.assertTrue(passed)
        self.assertEqual(len(summaries), 1)


class TestFailedCriteria(unittest.TestCase):
    """Unit tests for failed_criteria."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.trial_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_details(self, details: Any) -> None:
        verifier = self.trial_dir / "verifier"
        verifier.mkdir()
        (verifier / "reward-details.json").write_text(json.dumps(details),
                                                      encoding="utf-8")

    def test_lists_only_criteria_below_full_marks(self) -> None:
        # The shape RewardKit writes: a programmatic dimension and a judge.
        self._write_details({
            "checks": {
                "score":
                0.5,
                "kind":
                "programmatic",
                "criteria": [
                    {
                        "name": "resolved_first",
                        "value": 1.0,
                        "description": "resolved first"
                    },
                    {
                        "name": "image_probed",
                        "value": 0.0,
                        "description": "probe ran on the TPU"
                    },
                ],
            },
            "expectations": {
                "score":
                0.5,
                "kind":
                "llm",
                "criteria": [{
                    "name": "criterion_1",
                    "value": 0.0,
                    "description": "asks which repository",
                    "reasoning": "No   link\nwas given. " + "x" * 400,
                }],
            },
        })
        lines = failed_criteria(self.trial_dir)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0],
                         "checks/image_probed = 0.00: probe ran on the TPU")
        self.assertTrue(lines[1].startswith(
            "expectations/criterion_1 = 0.00: asks which repository"
            " -- judge: No link was given. x"))
        self.assertTrue(lines[1].endswith("..."))

    def test_walks_nested_groups(self) -> None:
        self._write_details({
            "reward": {
                "kind":
                "group",
                "components": [{
                    "name": "checks",
                    "detail": {
                        "criteria": [{
                            "name": "c",
                            "value": 0.5,
                            "description": "d"
                        }]
                    },
                }],
            }
        })
        self.assertEqual(failed_criteria(self.trial_dir),
                         ["reward/checks/c = 0.50: d"])

    def test_missing_or_unreadable_details_list_nothing(self) -> None:
        self.assertEqual(failed_criteria(self.trial_dir), [])
        (self.trial_dir / "verifier").mkdir()
        (self.trial_dir / "verifier" / "reward-details.json").write_text(
            "not json", encoding="utf-8")
        self.assertEqual(failed_criteria(self.trial_dir), [])


class TestHarborPassthrough(unittest.TestCase):
    """Unit tests ensuring arbitrary Harbor flags pass through with or without '--'."""

    def test_harbor_flags_passthrough_without_double_dash(self) -> None:
        parser = build_parser()
        args, harbor_args = parser.parse_known_args(
            ["-c", "antigravity-cli-no-kb", "--n-concurrent", "1", "--debug"])
        self.assertEqual(args.config, ["antigravity-cli-no-kb"])
        self.assertEqual(harbor_args, ["--n-concurrent", "1", "--debug"])

    def test_harbor_flags_passthrough_with_double_dash(self) -> None:
        parser = build_parser()
        args, harbor_args = parser.parse_known_args(
            ["-c", "antigravity-cli-no-kb", "--", "--n-concurrent", "1"])
        self.assertEqual(args.config, ["antigravity-cli-no-kb"])
        filtered_harbor = [a for a in harbor_args if a != "--"]
        self.assertEqual(filtered_harbor, ["--n-concurrent", "1"])

    def test_positional_config_with_harbor_flags(self) -> None:
        parser = build_parser()
        args, harbor_args = parser.parse_known_args(
            ["claude-code/agent-suite", "--n-concurrent", "1"])
        harbor_args = list(harbor_args)
        cfg_paths = resolve_config_files(args, harbor_args)
        self.assertEqual(len(cfg_paths), 1)
        self.assertTrue(cfg_paths[0].is_file())
        self.assertEqual(str(cfg_paths[0].relative_to(CONFIGS_DIR)),
                         "claude-code/agent-suite.yaml")
        self.assertEqual(harbor_args, ["--n-concurrent", "1"])


class TestMultipleConfigs(unittest.TestCase):
    """Unit tests for multiple configs resolution and Harbor Hub URL extraction."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_resolve_multiple_configs_named_space_separated(self) -> None:
        parser = build_parser()
        args, harbor_args = parser.parse_known_args([
            "-c", "agy/agent-suite", "claude-code/agent-suite",
            "--n-concurrent", "1"
        ])
        cfgs = resolve_config_files(args, list(harbor_args))
        self.assertEqual(len(cfgs), 2)
        self.assertEqual(str(cfgs[0].relative_to(CONFIGS_DIR)),
                         "agy/agent-suite.yaml")
        self.assertEqual(str(cfgs[1].relative_to(CONFIGS_DIR)),
                         "claude-code/agent-suite.yaml")

    def test_resolve_multiple_configs_named_repeated_flag(self) -> None:
        parser = build_parser()
        args, harbor_args = parser.parse_known_args(
            ["-c", "agy/agent-suite", "-c", "claude-code/agent-suite"])
        cfgs = resolve_config_files(args, list(harbor_args))
        self.assertEqual(len(cfgs), 2)
        self.assertEqual(str(cfgs[0].relative_to(CONFIGS_DIR)),
                         "agy/agent-suite.yaml")
        self.assertEqual(str(cfgs[1].relative_to(CONFIGS_DIR)),
                         "claude-code/agent-suite.yaml")

    def test_resolve_multiple_configs_comma_separated(self) -> None:
        parser = build_parser()
        args, harbor_args = parser.parse_known_args(
            ["-c", "agy/agent-suite,claude-code/agent-suite"])
        cfgs = resolve_config_files(args, list(harbor_args))
        self.assertEqual(len(cfgs), 2)
        self.assertEqual(str(cfgs[0].relative_to(CONFIGS_DIR)),
                         "agy/agent-suite.yaml")
        self.assertEqual(str(cfgs[1].relative_to(CONFIGS_DIR)),
                         "claude-code/agent-suite.yaml")

    def test_resolve_multiple_config_files(self) -> None:
        cfg1 = self.temp_dir / "c1.yaml"
        cfg2 = self.temp_dir / "c2.yaml"
        cfg1.write_text("job_name: 1\n", encoding="utf-8")
        cfg2.write_text("job_name: 2\n", encoding="utf-8")

        parser = build_parser()
        args, harbor_args = parser.parse_known_args(
            ["-f", str(cfg1), str(cfg2)])
        cfgs = resolve_config_files(args, list(harbor_args))
        self.assertEqual(cfgs, [cfg1.resolve(), cfg2.resolve()])

    def test_resolve_multiple_positional_configs(self) -> None:
        parser = build_parser()
        args, harbor_args = parser.parse_known_args([
            "agy/agent-suite", "claude-code/agent-suite", "--n-concurrent", "1"
        ])
        harbor_args = list(harbor_args)
        cfgs = resolve_config_files(args, harbor_args)
        self.assertEqual(len(cfgs), 2)
        self.assertEqual(str(cfgs[0].relative_to(CONFIGS_DIR)),
                         "agy/agent-suite.yaml")
        self.assertEqual(str(cfgs[1].relative_to(CONFIGS_DIR)),
                         "claude-code/agent-suite.yaml")
        self.assertEqual(harbor_args, ["--n-concurrent", "1"])

    def test_resolve_multiple_positional_yaml_files(self) -> None:
        cfg1 = self.temp_dir / "pos1.yaml"
        cfg2 = self.temp_dir / "pos2.yaml"
        cfg1.write_text("job_name: 1\n", encoding="utf-8")
        cfg2.write_text("job_name: 2\n", encoding="utf-8")

        parser = build_parser()
        args, harbor_args = parser.parse_known_args(
            [str(cfg1), str(cfg2), "--n-concurrent", "1"])
        harbor_args = list(harbor_args)
        cfgs = resolve_config_files(args, harbor_args)
        self.assertEqual(cfgs, [cfg1.resolve(), cfg2.resolve()])
        self.assertEqual(harbor_args, ["--n-concurrent", "1"])

    def test_get_harbor_hub_url_present(self) -> None:
        job_dir = self.temp_dir / "my_job"
        job_dir.mkdir()
        (job_dir / "result.json").write_text('{"id": "abc-123-xyz"}',
                                             encoding="utf-8")
        url = get_harbor_hub_url(job_dir)
        self.assertEqual(url,
                         "https://hub.harborframework.com/jobs/abc-123-xyz")

    def test_get_harbor_hub_url_missing(self) -> None:
        job_dir = self.temp_dir / "empty_job"
        job_dir.mkdir()
        self.assertIsNone(get_harbor_hub_url(job_dir))

    def test_get_harbor_hub_url_custom_env(self) -> None:
        job_dir = self.temp_dir / "custom_job"
        job_dir.mkdir()
        (job_dir / "result.json").write_text('{"id": "job-789"}',
                                             encoding="utf-8")
        custom_base = "https://harbor-eval-viewer-internal.a.run.app/jobs"
        with patch.dict(os.environ, {"HARBOR_VIEWER_JOBS_URL": custom_base}):
            url = get_harbor_hub_url(job_dir)
            self.assertEqual(url, f"{custom_base}/job-789")

    def test_get_harbor_hub_url_custom_viewer_url_env(self) -> None:
        job_dir = self.temp_dir / "custom_job_2"
        job_dir.mkdir()
        (job_dir / "result.json").write_text('{"id": "job-456"}',
                                             encoding="utf-8")
        custom_base = "https://harboreval-viewer-g3-6zn3wa2dfa-uc.a.run.app/jobs/"
        with patch.dict(os.environ, {"HARBOR_VIEWER_URL": custom_base}):
            url = get_harbor_hub_url(job_dir)
            self.assertEqual(
                url,
                "https://harboreval-viewer-g3-6zn3wa2dfa-uc.a.run.app/jobs/job-456",
            )


class TestRenderConfigCarriesDataset(unittest.TestCase):
    """The dataset must survive into the file Harbor actually reads.

    Harbor is handed a path, not our in-memory config, so a --dataset applied
    only to the dict reaches nothing. That failed at Harbor with "Either
    datasets or tasks must be provided" long after run.py had printed the
    dataset it thought it had selected.
    """

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_applied_dataset_is_written_out(self) -> None:
        cfg_path = CONFIGS_DIR / "claude-code" / "agent-suite.yaml"
        config = load_config(cfg_path)
        apply_dataset(config, "smoke-check")

        # The config mounts variables exported by the credential setup and staging.
        mock_env = {v: f"/tmp/{v}" for v in _RUN_MOD._MOUNT_VARS}
        with patch.dict(os.environ, mock_env):
            rendered = render_config(config, cfg_path, self.temp_dir)
        self.assertNotEqual(rendered, cfg_path,
                            "must not hand Harbor the on-disk config")

        written = yaml.safe_load(rendered.read_text(encoding="utf-8"))
        self.assertTrue(
            written.get("tasks") or written.get("datasets"),
            "rendered config reached Harbor without a dataset")
        self.assertEqual(written["datasets"],
                         [{
                             "path": "eval/datasets/smoke-check"
                         }])

    def test_rendered_even_without_mount_tokens(self) -> None:
        cfg_path = self.temp_dir / "plain.yaml"
        cfg_path.write_text("job_name: plain\n", encoding="utf-8")
        config = {"job_name": "plain", "datasets": [{"path": "x"}]}
        out_dir = self.temp_dir / "out"
        out_dir.mkdir()

        rendered = render_config(config, cfg_path, out_dir)
        self.assertEqual(rendered.parent, out_dir)
        self.assertEqual(
            yaml.safe_load(rendered.read_text(encoding="utf-8"))["datasets"],
            [{
                "path": "x"
            }])

    def test_empty_default_renders_as_empty_string(self) -> None:
        """`${VAR:-}` with VAR unset must stay "", not become YAML null.

        Substituting into the raw text turned `GKE_QUEUE: ${GKE_QUEUE:-}` into
        `GKE_QUEUE:`, and Harbor's JobConfig rejected the resulting None.
        """
        cfg_path = self.temp_dir / "empty.yaml"
        cfg_path.write_text("job_name: empty\n", encoding="utf-8")
        config = {
            "job_name":
            "empty",
            "agents": [{
                "env": {
                    "GKE_QUEUE": "${GKE_QUEUE:-}",
                    "GKE_PRIORITY": "${GKE_PRIORITY:-}",
                    "SET_VAR": "${SET_VAR:-fallback}",
                    "NO_DEFAULT": "${NO_DEFAULT_VAR}",
                    "PORT": 8080,
                }
            }],
        }
        env = {"SET_VAR": "chosen"}
        with patch.dict(os.environ, env, clear=False):
            for k in ("GKE_QUEUE", "GKE_PRIORITY", "NO_DEFAULT_VAR"):
                os.environ.pop(k, None)
            rendered = render_config(config, cfg_path, self.temp_dir)

        written = yaml.safe_load(rendered.read_text(encoding="utf-8"))
        agent_env = written["agents"][0]["env"]
        self.assertEqual(agent_env["GKE_QUEUE"], "")
        self.assertEqual(agent_env["GKE_PRIORITY"], "")
        self.assertEqual(agent_env["SET_VAR"], "chosen")
        self.assertEqual(agent_env["NO_DEFAULT"], "${NO_DEFAULT_VAR}")
        self.assertEqual(agent_env["PORT"], 8080)


class TestResolveTaskArg(unittest.TestCase):
    """--task selects one task; a bare name must be unambiguous."""

    def test_unique_bare_name(self) -> None:
        key, entry = resolve_task_arg("xprof-summary")
        self.assertEqual(key, "tasks")
        self.assertEqual(entry["path"],
                         "eval/datasets/tpu-hello-world-xprof/xprof-summary")

    def test_dataset_qualified_name(self) -> None:
        _, entry = resolve_task_arg("smoke-check/preflight-check")
        self.assertEqual(entry["path"],
                         "eval/datasets/smoke-check/preflight-check")

    def test_path(self) -> None:
        _, entry = resolve_task_arg(
            "eval/datasets/smoke-check/preflight-check")
        self.assertEqual(entry["path"],
                         "eval/datasets/smoke-check/preflight-check")

    def test_ambiguous_bare_name_raises(self) -> None:
        # `baseline` exists under both tpu-hello-world and hf-onboarding-torchtpu.
        with self.assertRaises(RuntimeError) as ctx:
            resolve_task_arg("baseline")
        msg = str(ctx.exception)
        self.assertIn("ambiguous", msg)
        self.assertIn("tpu-hello-world/baseline", msg)
        self.assertIn("hf-onboarding-torchtpu/baseline", msg)

    def test_dataset_scopes_an_ambiguous_name(self) -> None:
        for ds in ("tpu-hello-world", "hf-onboarding-torchtpu"):
            with self.subTest(dataset=ds):
                _, entry = resolve_task_arg("baseline", ds)
                self.assertEqual(entry["path"], f"eval/datasets/{ds}/baseline")

    def test_unknown_task_raises(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            resolve_task_arg("nope")
        self.assertIn("Available tasks:", str(ctx.exception))

    def test_task_not_in_named_dataset_raises(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            resolve_task_arg("xprof-summary", "smoke-check")
        self.assertIn("not found in dataset 'smoke-check'", str(ctx.exception))

    def test_unknown_dataset_raises(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            resolve_task_arg("preflight-check", "nope-dataset")
        self.assertIn("Dataset 'nope-dataset' not found", str(ctx.exception))

    def test_apply_tasks_sets_tasks_key(self) -> None:
        config = {"job_name": "x"}
        apply_tasks(config, ["preflight-check"])
        self.assertEqual(
            config["tasks"],
            [{
                "path": "eval/datasets/smoke-check/preflight-check"
            }])
        self.assertNotIn("datasets", config)

    def test_apply_tasks_selects_all_as_one_job(self) -> None:
        config = {"job_name": "x", "datasets": [{"path": "old"}]}
        apply_tasks(config, [
            "nightly-functional/smart-hf-search",
            "nightly-functional/remote-tpu-runner",
        ])
        self.assertEqual(config["tasks"], [
            {
                "path": "eval/datasets/nightly-functional/smart-hf-search"
            },
            {
                "path": "eval/datasets/nightly-functional/remote-tpu-runner"
            },
        ])
        self.assertNotIn("datasets", config)

    def test_apply_tasks_collapses_duplicates(self) -> None:
        config = {"job_name": "x"}
        apply_tasks(config, [
            "nightly-functional/remote-tpu-runner",
            "eval/datasets/nightly-functional/remote-tpu-runner",
        ])
        self.assertEqual(
            config["tasks"],
            [{
                "path": "eval/datasets/nightly-functional/remote-tpu-runner"
            }])

    def test_apply_tasks_unknown_task_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            apply_tasks({}, [
                "nightly-functional/remote-tpu-runner",
                "nightly-functional/nope",
            ])

    def test_parser_accepts_several_tasks(self) -> None:
        parser = build_parser()
        for argv in (["-c", "claude-code/agent-suite", "--task", "a", "b"],
                     ["-c", "claude-code/agent-suite", "-t", "a", "-t", "b"]):
            with self.subTest(argv=argv):
                args, _ = parser.parse_known_args(argv)
                self.assertEqual(args.task, ["a", "b"])

    def test_task_specs_normalizes(self) -> None:
        self.assertEqual(task_specs(None), [])
        self.assertEqual(task_specs("a"), ["a"])
        self.assertEqual(task_specs(["a", "b"]), ["a", "b"])

    def test_target_slug_for_several_tasks(self) -> None:
        args = argparse.Namespace(task=["nightly/a", "nightly/b"],
                                  suite=None,
                                  dataset=None)
        self.assertEqual(resolve_target_slug({}, args), "2-tasks")
        args.task = ["nightly/smart-hf-search"]
        self.assertEqual(resolve_target_slug({}, args), "smart-hf-search")


class TestMountVarsExpand(unittest.TestCase):
    """Every ${VAR} a config mounts must be one run.py substitutes.

    Harbor expands ${VAR} under agents[].env but not under environment.mounts,
    so a name run.py does not know reaches Harbor as a literal bind path. The
    agy configs used to depend on the eval_run.sh chain exporting nine such
    names; run.py produces them itself now, so the assertion is against
    _MOUNT_VARS alone and that chain is gone.
    """

    def test_every_mounted_var_is_substitutable(self) -> None:
        known = set(_RUN_MOD._MOUNT_VARS)
        pattern = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
        for cfg in sorted(CONFIGS_DIR.glob("**/*.yaml")):
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            for mount in (data.get("environment") or {}).get("mounts") or []:
                fields = f'{mount.get("source", "")} {mount.get("target", "")}'
                for var in pattern.findall(fields):
                    with self.subTest(config=cfg.name, var=var):
                        self.assertIn(
                            var, known,
                            f"{cfg.name} mounts ${{{var}}}, which _MOUNT_VARS "
                            "does not substitute; Harbor would receive it "
                            "literally")

    def test_host_auth_exports_what_configs_mount(self) -> None:
        """setup_host_auth must actually produce the names it promises."""
        with patch.dict(os.environ, {}, clear=False):
            _RUN_MOD.setup_host_auth()
            for var in ("HOST_ADC", "HOST_KUBECONFIG", "EVAL_VERTEX_ADC",
                        "HOST_AGY_SESSION_SUPERVISOR_HOOK",
                        "HOST_AGY_SESSION_SUPERVISOR",
                        "HOST_CLAUDE_SESSION_SUPERVISOR_HOOK",
                        "HOST_CLAUDE_SESSION_SUPERVISOR"):
                with self.subTest(var=var):
                    self.assertTrue(os.environ.get(var),
                                    f"{var} not exported by setup_host_auth")


class TestStagingAnswers(unittest.TestCase):
    """The installer is answered with the config's agent compute environment."""

    def test_agent_env_becomes_the_answers(self) -> None:
        cfg = {
            "agents": [{
                "name": "claude-code",
                "env": {
                    "COMPUTE_BACKEND": "gke_pod",
                    "GKE_POD": "my-test-pod",
                    "GKE_NAMESPACE": "test-ns",
                },
            }],
        }
        with patch.object(_RUN_MOD.subprocess, "run"), \
             patch.dict(os.environ, {}, clear=False):
            stage_package_if_needed(cfg, Path("/tmp/cfg.yaml"))
            written = Path(os.environ["STAGED_WORKSPACE"]).parent
        answers = yaml.safe_load((written / "answers.json").read_text())
        self.assertEqual(answers["backend"], "gke_pod")
        self.assertEqual(answers["pod"], "my-test-pod")
        self.assertEqual(answers["namespace"], "test-ns")

    def test_local_backend_env_becomes_answers(self) -> None:
        cfg = {
            "agents": [{
                "name": "claude-code",
                "env": {
                    "COMPUTE_BACKEND": "local",
                },
            }],
        }
        with patch.object(_RUN_MOD.subprocess, "run"), \
             patch.dict(os.environ, {}, clear=False):
            stage_package_if_needed(cfg, Path("/tmp/cfg2.yaml"))
            written = Path(os.environ["STAGED_WORKSPACE"]).parent
        answers = yaml.safe_load((written / "answers.json").read_text())
        self.assertEqual(answers, {"backend": "local"})


class TestDatasetSetupHook(unittest.TestCase):
    """A dataset shipping setup.sh gets it run before Harbor reads its tasks.

    Only jaxbench needs this: its task directories ship task.toml and base.py,
    while the Dockerfile and instruction.md are staged from common/ and
    gitignored. See the TODO on run_dataset_setup_if_present.
    """

    def _ran(self, config, has_setup_sh: bool = True):
        with (
                patch.object(_RUN_MOD.subprocess, "run") as mock_run,
                patch.object(Path, "is_file", return_value=has_setup_sh),
        ):
            run_dataset_setup_if_present(config)
            return [c[0][0] for c in mock_run.call_args_list]

    def test_dataset_with_setup_sh_is_staged(self) -> None:
        calls = self._ran(
            {"datasets": [{
                "path": "eval/datasets/jaxbench-level1"
            }]})
        self.assertEqual(len(calls), 1)
        self.assertTrue(
            str(calls[0][1]).endswith("datasets/jaxbench-level1/setup.sh"))

    def test_single_task_stages_its_dataset(self) -> None:
        """--task selects a leaf, but setup.sh lives at the dataset root."""
        calls = self._ran(
            {"tasks": [{
                "path": "eval/datasets/jaxbench-level1/8p-gemm"
            }]})
        self.assertEqual(len(calls), 1)
        self.assertTrue(
            str(calls[0][1]).endswith("datasets/jaxbench-level1/setup.sh"))

    def test_dataset_without_setup_sh_is_a_no_op(self) -> None:
        self.assertEqual(
            self._ran({"datasets": [{
                "path": "eval/datasets/smoke-check"
            }]},
                      has_setup_sh=False), [])

    def test_path_outside_datasets_is_ignored(self) -> None:
        self.assertEqual(
            self._ran({"tasks": [{
                "path": "/tmp/elsewhere"
            }]},
                      has_setup_sh=False), [])


class TestPinnedImageHasDockerfile(unittest.TestCase):
    """A task pinning docker_image must also restate it as a Dockerfile.

    Harbor's GKE environment builds from environment/ and never reads
    docker_image (gke.py resolves the pod image from the Cloud Build output),
    so a pin with no Dockerfile fails at run time with FileNotFoundError --
    which is how six tpu-hello-world tasks were found to be unrunnable on GKE.
    """

    def test_pin_and_dockerfile_agree(self) -> None:
        import tomllib
        for task_toml in sorted(DATASETS_DIR.glob("**/task.toml")):
            data = tomllib.loads(task_toml.read_text(encoding="utf-8"))
            pinned = (data.get("environment") or {}).get("docker_image")
            if not pinned:
                continue
            dockerfile = task_toml.parent / "environment" / "Dockerfile"
            rel = task_toml.parent.relative_to(DATASETS_DIR)
            with self.subTest(task=str(rel)):
                self.assertTrue(
                    dockerfile.is_file(),
                    f"{rel} pins docker_image but has no environment/Dockerfile;"
                    " Harbor's GKE path builds from environment/ and will fail"
                )
                froms = [
                    ln.split(None, 1)[1].strip()
                    for ln in dockerfile.read_text(
                        encoding="utf-8").splitlines()
                    if ln.startswith("FROM ")
                ]
                self.assertIn(
                    pinned, froms,
                    f"{rel} Dockerfile FROM does not match its docker_image pin"
                )


class TestCheckDependencies(unittest.TestCase):
    """Unit tests for runtime dependency verification."""

    def test_harbor_present(self) -> None:
        with patch("shutil.which", return_value="/usr/local/bin/harbor"):
            check_dependencies()

    def test_harbor_missing_raises_runtime_error(self) -> None:
        with patch("shutil.which", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                check_dependencies()
            self.assertIn("harbor", str(ctx.exception))
            self.assertIn("./eval/setup.sh", str(ctx.exception))


class TestSetupHostAuthSupervisorVariables(unittest.TestCase):
    """Unit tests for supervisor hook variables and quota project export."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.adc_file = self.temp_dir / "adc.json"
        self.adc_file.write_text(
            '{"client_id": "test", "quota_project_id": "test-p"}',
            encoding="utf-8")
        self.kube_file = self.temp_dir / "kubeconfig"
        self.kube_file.write_text("apiVersion: v1", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_supervisor_hooks_exported(self) -> None:
        with patch.dict(os.environ, {
                "GOOGLE_APPLICATION_CREDENTIALS": str(self.adc_file),
                "KUBECONFIG": str(self.kube_file),
                "VERTEXAI_PROJECT": "my-project",
        },
                        clear=False):
            setup_host_auth()
            self.assertEqual(os.environ.get("QUOTA_PROJECT"), "my-project")
            self.assertIn("HOST_AGY_SESSION_SUPERVISOR_HOOK", os.environ)
            self.assertIn("HOST_AGY_SESSION_SUPERVISOR", os.environ)
            self.assertIn("HOST_CLAUDE_SESSION_SUPERVISOR_HOOK", os.environ)
            self.assertIn("HOST_CLAUDE_SESSION_SUPERVISOR", os.environ)


class TestRenderConfigPathNormalization(unittest.TestCase):
    """Unit tests for path normalization in render_config."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cfg_path = self.temp_dir / "cfg.yaml"
        self.cfg_path.touch()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dataset_relative_path_normalized(self) -> None:
        raw_config = {
            "job_name": "test",
            "datasets": [{
                "path": "datasets/smoke-check"
            }],
        }
        rendered_path = render_config(raw_config, self.cfg_path, self.temp_dir)
        rendered = yaml.safe_load(rendered_path.read_text(encoding="utf-8"))
        self.assertEqual(rendered["datasets"][0]["path"],
                         "eval/datasets/smoke-check")

    def test_task_relative_path_normalized(self) -> None:
        raw_config = {
            "job_name": "test",
            "tasks": [{
                "path": "datasets/smoke-check/preflight-check"
            }],
        }
        rendered_path = render_config(raw_config, self.cfg_path, self.temp_dir)
        rendered = yaml.safe_load(rendered_path.read_text(encoding="utf-8"))
        self.assertEqual(rendered["tasks"][0]["path"],
                         "eval/datasets/smoke-check/preflight-check")

    def test_already_normalized_path_preserved(self) -> None:
        raw_config = {
            "job_name": "test",
            "datasets": [{
                "path": "eval/datasets/smoke-check"
            }],
        }
        rendered_path = render_config(raw_config, self.cfg_path, self.temp_dir)
        rendered = yaml.safe_load(rendered_path.read_text(encoding="utf-8"))
        self.assertEqual(rendered["datasets"][0]["path"],
                         "eval/datasets/smoke-check")


class TestPublishGithubStepSummary(unittest.TestCase):
    """Unit tests for publishing Harbor Hub links to GITHUB_STEP_SUMMARY."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.summary_file = self.temp_dir / "step_summary.md"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_noop_when_no_path_and_no_env(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            publish_github_step_summary([{
                "job_name": "test"
            }],
                                        upload_enabled=True)
            self.assertFalse(self.summary_file.exists())

    def test_noop_when_empty_results(self) -> None:
        publish_github_step_summary([],
                                    upload_enabled=True,
                                    summary_path=self.summary_file)
        self.assertFalse(self.summary_file.exists())

    def test_writes_summary_with_hub_link_and_trials(self) -> None:
        results = [{
            "config_path":
            Path("eval/configs/agy/agent-suite.yaml"),
            "job_name":
            "test_job_123",
            "job_dir":
            self.temp_dir / "test_job_123",
            "hub_url":
            "https://harbor.example.com/jobs/123",
            "upload_succeeded":
            True,
            "eval_passed":
            True,
            "trial_summaries": [
                {
                    "trial_name": "trial_1",
                    "score": 1.0,
                    "passed": True
                },
                {
                    "trial_name": "trial_2",
                    "score": 0.0,
                    "passed": False
                },
            ],
        }]

        publish_github_step_summary(results,
                                    upload_enabled=True,
                                    summary_path=self.summary_file)

        self.assertTrue(self.summary_file.exists())
        content = self.summary_file.read_text(encoding="utf-8")
        self.assertIn("Harbor Evaluation: `test_job_123`", content)
        self.assertIn("✅ Passed", content)
        self.assertIn(
            "[https://harbor.example.com/jobs/123](https://harbor.example.com/jobs/123)",
            content)
        self.assertIn("| `trial_1` | 1.00 | ✅ PASS |", content)
        self.assertIn("| `trial_2` | 0.00 | ❌ FAIL |", content)

    def test_writes_summary_upload_disabled(self) -> None:
        results = [{
            "config_path":
            Path("eval/configs/claude-code/agent-suite.yaml"),
            "job_name":
            "test_job_456",
            "job_dir":
            self.temp_dir / "test_job_456",
            "hub_url":
            None,
            "upload_succeeded":
            False,
            "eval_passed":
            False,
            "trial_summaries": [],
        }]

        publish_github_step_summary(results,
                                    upload_enabled=False,
                                    summary_path=self.summary_file)

        content = self.summary_file.read_text(encoding="utf-8")
        self.assertIn("Harbor Evaluation: `test_job_456`", content)
        self.assertIn("❌ Failed", content)
        self.assertIn("upload disabled; job: `test_job_456`", content)

    def test_writes_summary_upload_failed(self) -> None:
        results = [{
            "config_path": Path("eval/configs/agy/agent-suite.yaml"),
            "job_name": "test_job_789",
            "job_dir": self.temp_dir / "test_job_789",
            "hub_url": "https://harbor.example.com/jobs/789",
            "upload_succeeded": False,
            "eval_passed": True,
            "trial_summaries": [],
        }]

        publish_github_step_summary(results,
                                    upload_enabled=True,
                                    summary_path=self.summary_file)

        content = self.summary_file.read_text(encoding="utf-8")
        self.assertIn("upload failed", content)
        self.assertIn(
            "[https://harbor.example.com/jobs/789](https://harbor.example.com/jobs/789)",
            content)

    def _summary_with(self, trials: list[dict]) -> str:
        publish_github_step_summary(
            [{
                "config_path": Path("eval/configs/agy/agent-suite.yaml"),
                "job_name": "test_job_fr",
                "hub_url": None,
                "upload_succeeded": False,
                "eval_passed": False,
                "trial_summaries": trials,
            }],
            upload_enabled=False,
            summary_path=self.summary_file)
        return self.summary_file.read_text(encoding="utf-8")

    def test_writes_failure_report_only_for_failed_trials(self) -> None:
        content = self._summary_with([
            {
                "trial_name": "ok_trial",
                "score": 1.0,
                "passed": True,
                "failure_report": "PASSING REPORT",
            },
            {
                "trial_name": "bad_trial",
                "score": 0.5,
                "passed": False,
                "failure_report": "**Verdict:** broke at Step 7.",
            },
        ])
        self.assertIn("<details open>", content)
        self.assertIn("Explanation of failures</b>: <code>bad_trial</code>",
                      content)
        self.assertIn("**Verdict:** broke at Step 7.", content)
        self.assertNotIn("PASSING REPORT", content)
        self.assertLess(content.index("| `bad_trial` |"),
                        content.index("broke at Step 7"))
        self.assertLess(content.index("broke at Step 7"),
                        content.rindex("---"))

    def test_failed_trial_without_report_adds_nothing(self) -> None:
        content = self._summary_with([{
            "trial_name": "bad_trial",
            "score": 0.0,
            "passed": False,
        }])
        self.assertIn("| `bad_trial` | 0.00 | ❌ FAIL |", content)
        self.assertNotIn("<details", content)

    def test_failure_report_is_truncated_for_summary(self) -> None:
        content = self._summary_with([{
            "trial_name": "bad_trial",
            "score": 0.0,
            "passed": False,
            "failure_report": "y" * 200_000,
        }])
        self.assertIn(
            "truncated; see the trial's verifier/reward-details.json", content)
        self.assertLess(len(content), 100_000)


class TestReadFailureReport(unittest.TestCase):
    """Unit tests for extracting the failure_report judge's output."""

    def setUp(self) -> None:
        self.job_dir = Path(tempfile.mkdtemp(prefix="test_failure_report_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.job_dir, ignore_errors=True)

    def _trial(self, name: str, reward: float, details: object | None) -> Path:
        trial_dir = self.job_dir / name
        (trial_dir / "verifier").mkdir(parents=True)
        (trial_dir / "result.json").write_text(json.dumps({
            "trial_name": name,
            "verifier_result": {
                "rewards": {
                    "reward": reward
                }
            },
        }),
                                               encoding="utf-8")
        if details is not None:
            text = details if isinstance(details, str) else json.dumps(details)
            (trial_dir / "verifier" / "reward-details.json").write_text(
                text, encoding="utf-8")
        return trial_dir

    @staticmethod
    def _details(reasoning: str) -> dict:
        # The shape rewardkit writes for a tests/failure_report/judge.toml.
        return {
            "expectations": {
                "score": 1.0,
                "criteria": [{
                    "name": "x",
                    "reasoning": "ok"
                }]
            },
            "failure_report": {
                "score":
                0.0,
                "kind":
                "llm",
                "criteria": [{
                    "name": "failure_report",
                    "value": 0.0,
                    "reasoning": reasoning,
                }],
            },
        }

    def test_reads_reasoning(self) -> None:
        trial = self._trial("t", 0.0, self._details("**Verdict:** Step 3."))
        self.assertEqual(_RUN_MOD.read_failure_report(trial),
                         "**Verdict:** Step 3.")

    def test_none_when_task_has_no_failure_report_judge(self) -> None:
        details = self._details("x")
        del details["failure_report"]
        trial = self._trial("t", 0.0, details)
        self.assertIsNone(_RUN_MOD.read_failure_report(trial))

    def test_none_when_details_missing_or_malformed(self) -> None:
        self.assertIsNone(
            _RUN_MOD.read_failure_report(self._trial("a", 0.0, None)))
        self.assertIsNone(
            _RUN_MOD.read_failure_report(self._trial("b", 0.0, "{not json")))
        self.assertIsNone(
            _RUN_MOD.read_failure_report(self._trial("c", 0.0, [1, 2])))
        self.assertIsNone(
            _RUN_MOD.read_failure_report(
                self._trial("d", 0.0, {"failure_report": {
                    "criteria": "?"
                }})))
        self.assertIsNone(
            _RUN_MOD.read_failure_report(
                self._trial("e", 0.0, self._details("  "))))
        self.assertIsNone(_RUN_MOD.read_failure_report(self.job_dir / "nope"))

    def test_check_job_scores_attaches_report_when_present(self) -> None:
        self._trial("with_report", 0.5, self._details("REPORT"))
        self._trial("without_report", 0.0, {"expectations": {}})
        self._trial("no_details", 1.0, None)
        (self.job_dir / "result.json").write_text("{}", encoding="utf-8")

        _, summaries = check_job_scores(self.job_dir)
        by_name = {s["trial_name"]: s for s in summaries}
        self.assertEqual(by_name["with_report"]["failure_report"], "REPORT")
        self.assertNotIn("failure_report", by_name["without_report"])
        self.assertNotIn("failure_report", by_name["no_details"])


class TestFailureReportJudges(unittest.TestCase):
    """Every failure_report judge must be excluded from its task's score."""

    def test_judge_and_zero_weight_come_together(self) -> None:
        import tomllib

        for reward_toml in sorted(DATASETS_DIR.glob("*/*/tests/reward.toml")):
            tests_dir = reward_toml.parent
            has_judge = (tests_dir / "failure_report" / "judge.toml").is_file()
            specs = tomllib.loads(reward_toml.read_text(encoding="utf-8")).get(
                "reward", [])
            weights = [(s.get("weights") or {}).get("failure_report")
                       for s in specs]
            with self.subTest(
                    task=str(tests_dir.parent.relative_to(DATASETS_DIR))):
                if has_judge:
                    # A child directory counts with weight 1.0 unless the
                    # parent overrides it; [judge].weight does not apply.
                    self.assertTrue(specs, "reward.toml has no [[reward]]")
                    self.assertTrue(all(w == 0.0 for w in weights), weights)
                else:
                    # rewardkit rejects weights naming a missing input.
                    self.assertTrue(all(w is None for w in weights), weights)


class TestHarborJobMonitor(unittest.TestCase):
    """Unit tests for HarborJobMonitor live log streaming and heartbeats."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_job_monitor_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parser_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--config", "claude-code/agent-suite", "--dataset", "smoke-check"
        ])
        self.assertFalse(args.stream_logs)
        self.assertEqual(args.heartbeat_interval, 30.0)

        args_stream = parser.parse_args([
            "--config", "claude-code/agent-suite", "--dataset", "smoke-check",
            "--stream-logs", "--heartbeat-interval", "10.0"
        ])
        self.assertTrue(args_stream.stream_logs)
        self.assertEqual(args_stream.heartbeat_interval, 10.0)

    def test_monitor_stages(self) -> None:
        monitor = HarborJobMonitor(self.temp_dir, "test_job")
        trial_dir = self.temp_dir / "my_trial__abc123"
        trial_dir.mkdir(parents=True)

        # 1. Config present -> env stage
        (trial_dir / "config.json").write_text("{}", encoding="utf-8")
        monitor._poll()
        self.assertEqual(monitor._stages.get("my_trial__abc123"), "env")

        # 2. Agent output present -> agent stage
        agent_dir = trial_dir / "agent"
        agent_dir.mkdir()
        (agent_dir / "claude-code.txt").write_text("output", encoding="utf-8")
        monitor._poll()
        self.assertEqual(monitor._stages.get("my_trial__abc123"), "agent")

        # 3. Verifier output present -> verifier stage
        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir()
        (verifier_dir / "test-stdout.txt").write_text("test", encoding="utf-8")
        monitor._poll()
        self.assertEqual(monitor._stages.get("my_trial__abc123"), "verifier")

        # 4. Result present -> finished stage
        res_file = trial_dir / "result.json"
        res_file.write_text(
            json.dumps({
                "trial_name": "my_trial__abc123",
                "verifier_result": {
                    "rewards": {
                        "reward": 1.0
                    }
                }
            }),
            encoding="utf-8",
        )
        monitor._poll()
        self.assertEqual(monitor._stages.get("my_trial__abc123"), "finished")

    def test_monitor_failed_trial(self) -> None:
        monitor = HarborJobMonitor(self.temp_dir, "test_job")
        trial_dir = self.temp_dir / "failed_trial__xyz"
        trial_dir.mkdir(parents=True)
        res_file = trial_dir / "result.json"
        res_file.write_text(
            json.dumps({
                "trial_name": "failed_trial__xyz",
                "exception_info": {
                    "type": "TimeoutError"
                }
            }),
            encoding="utf-8",
        )

        monitor._poll()
        self.assertEqual(monitor._stages.get("failed_trial__xyz"), "finished")

    def test_monitor_lifecycle(self) -> None:
        monitor = HarborJobMonitor(self.temp_dir,
                                   "test_job",
                                   heartbeat_interval_sec=0.1)
        trial_dir = self.temp_dir / "test_trial"
        trial_dir.mkdir(parents=True)
        (trial_dir / "config.json").write_text("{}", encoding="utf-8")

        monitor.start()
        (trial_dir / "result.json").write_text("{}", encoding="utf-8")
        monitor.stop()
        self.assertEqual(monitor._stages.get("test_trial"), "finished")


class TestWholePackage(unittest.TestCase):
    """Every config installs the whole package; a control arm, nothing."""

    def test_skills_picks_a_debugging_subset(self) -> None:
        args = build_parser().parse_args(
            ["-c", "claude-code/agent-suite", "--skills", "tpu-agent"])
        self.assertEqual(["tpu-agent"], args.skills)

    def test_a_subset_is_named_in_the_job(self) -> None:
        path = CONFIGS_DIR / "claude-code" / "agent-suite.yaml"
        name = generate_job_name(load_config(path),
                                 path,
                                 build_parser().parse_args(
                                     ["-c", "claude-code/agent-suite"]),
                                 timestamp="t",
                                 skills=["tpu-agent"])
        self.assertIn("_cc-agent-suite-subset-agent-v6e_", name)

    def test_another_package_is_named_in_the_job(self) -> None:
        # PACKAGE_MANIFEST=package.experimental.json is another system under
        # test, so its runs must not join the release runs on a dashboard.
        path = CONFIGS_DIR / "claude-code" / "agent-suite.yaml"
        args = build_parser().parse_args(["-c", "claude-code/agent-suite"])
        with patch.dict(os.environ,
                        {"PACKAGE_MANIFEST": "package.experimental.json"}):
            name = generate_job_name(load_config(path),
                                     path,
                                     args,
                                     timestamp="t")
        self.assertIn("_cc-agent-suite-experimental-v6e_", name)
        with patch.dict(os.environ, {"PACKAGE_MANIFEST": ""}):
            self.assertIn(
                "_cc-agent-suite-v6e_",
                generate_job_name(load_config(path), path, args,
                                  timestamp="t"))

    def test_a_control_arm_is_not_named_for_a_package(self) -> None:
        # It installs nothing, whatever PACKAGE_MANIFEST says.
        path = CONFIGS_DIR / "claude-code" / "baseline.yaml"
        args = build_parser().parse_args(["-c", "claude-code/baseline"])
        with patch.dict(os.environ,
                        {"PACKAGE_MANIFEST": "package.experimental.json"}):
            name = generate_job_name(load_config(path),
                                     path,
                                     args,
                                     timestamp="t")
        self.assertNotIn("experimental", name)

    def _main(self, *argv: str, **env: str) -> str:
        task = Path(tempfile.mkdtemp()) / "task1"
        task.mkdir()
        (task / "task.toml").write_text("", encoding="utf-8")
        err = io.StringIO()
        with patch("sys.argv", ["run.py", *argv, "-t", str(task)]), \
             patch.dict(os.environ, env), \
             patch.object(_RUN_MOD, "check_dependencies"), \
             patch.object(_RUN_MOD, "setup_host_auth"), \
             patch.object(_RUN_MOD, "stage_package_if_needed") as stage, \
             contextlib.redirect_stderr(err), \
             self.assertRaises(SystemExit) as exited:
            _RUN_MOD.main()
        self.assertEqual(1, exited.exception.code)
        stage.assert_not_called()
        return err.getvalue()

    def test_a_subset_of_a_baseline_is_refused(self) -> None:
        self.assertIn(
            "control arm",
            self._main("-c", "claude-code/baseline", "--skills", "tpu-agent"))

    def test_a_subset_in_ci_is_refused(self) -> None:
        self.assertIn(
            "CI runs the whole package",
            self._main("-c",
                       "claude-code/agent-suite",
                       "--skills",
                       "tpu-agent",
                       GITHUB_RUN_ID="1"))

    def test_an_unknown_skill_is_refused(self) -> None:
        self.assertIn(
            "tpu-nope",
            self._main("-c", "claude-code/agent-suite", "--skills",
                       "tpu-nope"))

    def test_an_experimental_skill_needs_the_experimental_package(
            self) -> None:
        # close-gaps ships in package.experimental.json only, so the release
        # package refuses it; PACKAGE_MANIFEST=package.experimental.json's
        # check is the same call against its own manifest (TestPackageManifest
        # in test_utils.py).
        self.assertIn(
            "close-gaps",
            self._main("-c", "claude-code/agent-suite", "--skills",
                       "close-gaps"))

    def test_render_config_filters_custom_eval_keys(self) -> None:
        cfg = {
            "agents": [{
                "name": "claude-code"
            }],
            "compute": {
                "backend": "local"
            },
            "version": "v2",
            "datasets": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            rendered_path = render_config(cfg, Path("/tmp/dummy.yaml"),
                                          Path(tmpdir))
            written = yaml.safe_load(rendered_path.read_text())
            self.assertNotIn("compute", written)
            self.assertNotIn("version", written)
            self.assertIn("agents", written)

    def test_a_key_that_once_chose_skills_is_refused(self) -> None:
        for cfg in ({
                "stage_skills": ["tpu-agent"]
        }, {
                "agents": [{
                    "kwargs": {
                        "package_dir": []
                    }
                }]
        }):
            with self.subTest(cfg=cfg):
                with self.assertRaisesRegex(RuntimeError, "no longer exist"):
                    _RUN_MOD.reject_removed_keys(cfg, Path("x.yaml"))

    def test_an_agent_suite_config_stages_the_whole_package(self) -> None:
        cfg = {
            "agents": [{
                "name": "claude-code"
            }],
            "environment": {
                "mounts": []
            },
            "compute": {
                "backend": "local"
            },
        }
        with patch.object(_RUN_MOD, "stage_composite_package") as mock_stage:
            mock_stage.return_value = StagingResult(
                staged_root=Path("/tmp/staged_root"),
                staged_workspace=Path("/tmp/staged_root/workspace"),
                staged_dist=Path("/tmp/staged_root/claude-code/pkg"),
                central_pkg_dir=Path("/tmp/central"),
                is_empty=False,
            )
            updates = stage_package_if_needed(cfg,
                                              Path("/tmp/agent-suite.yaml"))
            apply_mount_updates(cfg, updates)
            mock_stage.assert_called_once_with(
                "claude-code",
                control_arm=False,
                skills=None,
                compute_answers=ANY,
                repo_root=ANY,
                manifest="package.release.json",
            )
            self.assertTrue(len(cfg["environment"]["mounts"]) > 0)

    def test_stage_package_if_needed_experimental_stages_its_manifest(
            self) -> None:
        cfg = {
            "agents": [{
                "name": "claude-code",
                "env": {
                    "PACKAGE_MANIFEST":
                    "${PACKAGE_MANIFEST:-package.release.json}"
                },
            }],
            "environment": {
                "mounts": []
            }
        }
        with patch.object(_RUN_MOD, "stage_composite_package") as mock_stage:
            mock_stage.return_value = StagingResult(
                staged_root=Path("/tmp/staged_root"),
                staged_workspace=Path("/tmp/staged_root/workspace"),
                staged_dist=Path("/tmp/staged_root/claude-code/pkg"),
                central_pkg_dir=Path("/tmp/central"),
                is_empty=False,
            )
            with patch.dict(os.environ,
                            {"PACKAGE_MANIFEST": "package.experimental.json"}):
                stage_package_if_needed(cfg, Path("/tmp/agent-suite.yaml"))
            mock_stage.assert_called_once_with(
                "claude-code",
                control_arm=False,
                skills=None,
                compute_answers=ANY,
                repo_root=ANY,
                manifest="package.experimental.json",
            )

    def test_stage_package_if_needed_refuses_an_unknown_manifest(self) -> None:
        # A typo must not quietly run the release package.
        cfg = {
            "agents": [{
                "name": "claude-code",
                "env": {
                    "PACKAGE_MANIFEST": "experimentl"
                },
            }],
        }
        with patch.object(_RUN_MOD, "stage_composite_package") as mock_stage:
            with self.assertRaisesRegex(RuntimeError, "experimentl"):
                stage_package_if_needed(cfg, Path("/tmp/agent-suite.yaml"))
            mock_stage.assert_not_called()

    def test_stage_package_if_needed_control_arm_ignores_the_manifest(
            self) -> None:
        # A control arm installs nothing, so a bad value must not stop it.
        cfg = {
            "agents": [{
                "name": "claude-code",
                "env": {
                    "PACKAGE_MANIFEST": "experimentl"
                },
            }],
        }
        with patch.object(_RUN_MOD, "stage_composite_package") as mock_stage:
            mock_stage.return_value = StagingResult(
                staged_root=Path("/tmp/staged_root"),
                staged_workspace=Path("/tmp/staged_root/workspace"),
                staged_dist=Path("/tmp/staged_root/claude-code/pkg"),
                central_pkg_dir=Path("/tmp/central"),
                is_empty=True,
            )
            stage_package_if_needed(cfg, Path("/tmp/baseline.yaml"))
            self.assertTrue(mock_stage.call_args.kwargs["control_arm"])

    def test_stage_package_if_needed_baseline_mounts_supervisor_by_default(
            self) -> None:
        cfg = {
            "agents": [{
                "name": "claude-code"
            }],
            "environment": {
                "mounts": []
            },
        }
        with patch.object(_RUN_MOD, "stage_composite_package") as mock_stage:
            mock_stage.return_value = StagingResult(
                staged_root=Path("/tmp/staged_root"),
                staged_workspace=Path("/tmp/staged_root/workspace"),
                staged_dist=Path("/tmp/staged_root/claude-code/pkg"),
                central_pkg_dir=Path("/tmp/central"),
                is_empty=True,
            )
            updates = stage_package_if_needed(
                cfg, Path("/tmp/baseline_claude_v7.yaml"))
            apply_mount_updates(cfg, updates)
            mock_stage.assert_called_once_with(
                "claude-code",
                control_arm=True,
                skills=None,
                compute_answers=ANY,
                repo_root=ANY,
                manifest="package.release.json",
            )
            targets = {m["target"] for m in cfg["environment"]["mounts"]}
            self.assertIn("/usr/local/bin/claude_session_supervisor_hook.sh",
                          targets)
            self.assertIn("/usr/local/bin/claude_session_supervisor.sh",
                          targets)
            self.assertEqual(
                cfg["environment"]["env"]["BASH_ENV"],
                "/usr/local/bin/claude_session_supervisor_hook.sh",
            )


class TestJobNameGeneration(unittest.TestCase):
    """Unit tests for descriptive Harbor job name generation."""

    def test_target_always_first_with_task(self) -> None:
        args = argparse.Namespace(task="smoke-check/preflight-check",
                                  suite=None,
                                  dataset=None,
                                  job_name=None)
        cfg = {"agents": [{"name": "claude-code"}]}
        cfg_path = Path("eval/configs/claude-code/agent-suite.yaml")
        name = generate_job_name(cfg,
                                 cfg_path,
                                 args,
                                 timestamp="20260930-050000")
        self.assertTrue(name.startswith("preflight-check_cc-agent-suite_"))
        self.assertEqual(
            name, "preflight-check_cc-agent-suite_local_20260930-050000")

    def test_target_always_first_with_dataset(self) -> None:
        args = argparse.Namespace(task=None,
                                  suite=None,
                                  dataset="nightly",
                                  job_name=None)
        cfg = {"agents": [{"name": "claude-code"}]}
        cfg_path = Path("eval/configs/claude-code/agent-suite.yaml")
        name = generate_job_name(cfg,
                                 cfg_path,
                                 args,
                                 timestamp="20260930-050000")
        self.assertEqual(name, "nightly_cc-agent-suite_local_20260930-050000")

    def test_underscore_strictly_separates_four_segments(self) -> None:
        args = argparse.Namespace(task="gemma-onboarding",
                                  suite=None,
                                  dataset=None,
                                  job_name=None)
        cfg = {"agents": [{"name": "antigravity"}]}
        cfg_path = Path("eval/configs/agy/baseline.yaml")
        name = generate_job_name(cfg,
                                 cfg_path,
                                 args,
                                 timestamp="20260930-050000")
        segments = name.split("_")
        self.assertEqual(len(segments), 4)
        self.assertEqual(segments[0], "gemma-onboarding")
        self.assertEqual(segments[1], "agy-baseline")
        self.assertEqual(segments[2], "local")
        self.assertEqual(segments[3], "20260930-050000")

    def test_the_shipped_configs_name_their_arm(self) -> None:
        for name, sut in (("claude-code/agent-suite", "cc-agent-suite"),
                          ("claude-code/baseline", "cc-baseline"),
                          ("agy/agent-suite", "agy-agent-suite"),
                          ("agy/baseline", "agy-baseline")):
            path = CONFIGS_DIR / f"{name}.yaml"
            with self.subTest(config=name):
                self.assertEqual(resolve_sut_slug(load_config(path), path),
                                 sut)

    def test_env_slug_github_runner(self) -> None:
        with patch.dict(os.environ, {"GITHUB_RUN_ID": "12345678"}):
            self.assertEqual(resolve_env_slug(), "gh-12345678")

    def test_env_slug_local(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_env_slug(), "local")

    def test_github_nightly_job_name(self) -> None:
        args = argparse.Namespace(task=None,
                                  suite=None,
                                  dataset="nightly",
                                  job_name=None)
        cfg = {"agents": [{"name": "claude-code"}]}
        cfg_path = Path("eval/configs/claude-code/agent-suite.yaml")
        with patch.dict(os.environ, {"GITHUB_RUN_ID": "98765432"}):
            name = generate_job_name(cfg,
                                     cfg_path,
                                     args,
                                     timestamp="20260930-050000")
            self.assertEqual(
                name, "nightly_cc-agent-suite_gh-98765432_20260930-050000")

    def test_explicit_job_name_override(self) -> None:
        args = argparse.Namespace(task=None,
                                  suite=None,
                                  dataset="nightly",
                                  job_name="custom_eval_job")
        cfg = {"agents": [{"name": "claude-code"}]}
        cfg_path = Path("eval/configs/claude-code/agent-suite.yaml")
        name = generate_job_name(cfg,
                                 cfg_path,
                                 args,
                                 timestamp="20260930-050000")
        self.assertEqual(name, "custom_eval_job_20260930-050000")

    def test_max_length_safeguard(self) -> None:
        long_task = "a" * 150
        args = argparse.Namespace(task=long_task,
                                  suite=None,
                                  dataset=None,
                                  job_name=None)
        cfg = {"agents": [{"name": "claude-code"}]}
        cfg_path = Path("eval/configs/claude-code/agent-suite.yaml")
        name = generate_job_name(cfg,
                                 cfg_path,
                                 args,
                                 timestamp="20260930-050000",
                                 max_length=128)
        self.assertLessEqual(len(name), 128)
        self.assertTrue(name.endswith("_cc-agent-suite_local_20260930-050000"))

    def test_the_agy_config_generates_agy_agent_suite(self) -> None:
        args = argparse.Namespace(task=None,
                                  suite=None,
                                  dataset="smoke-check",
                                  job_name=None)
        cfg_path = CONFIGS_DIR / "agy" / "agent-suite.yaml"
        cfg = load_config(cfg_path)
        name = generate_job_name(cfg,
                                 cfg_path,
                                 args,
                                 timestamp="20260930-050000")
        self.assertEqual(
            name, "smoke-check_agy-agent-suite-v6e_local_20260930-050000")

    def test_accelerator_slug_resolution(self) -> None:
        self.assertEqual(resolve_accelerator_slug({}), "")
        self.assertEqual(
            resolve_accelerator_slug({
                "agents": [{
                    "env": {
                        "GKE_TPU_ACCELERATOR": "tpu-v6e-slice"
                    }
                }]
            }),
            "v6e",
        )
        self.assertEqual(
            resolve_accelerator_slug(
                {"agents": [{
                    "env": {
                        "GKE_TPU_ACCELERATOR": "tpu7x"
                    }
                }]}),
            "v7",
        )
        self.assertEqual(
            resolve_accelerator_slug({
                "agents": [{
                    "env": {
                        "GKE_TPU_ACCELERATOR": "tpu-v5-lite-podslice"
                    }
                }]
            }),
            "v5e",
        )
        with patch.dict(os.environ, {"ACCELERATOR": "v7"}):
            self.assertEqual(resolve_accelerator_slug({}), "v7")
        with patch.dict(os.environ, {"EVAL_ACCELERATOR": "tpu7x-8"}):
            self.assertEqual(
                resolve_accelerator_slug({
                    "agents": [{
                        "env": {
                            "GKE_TPU_ACCELERATOR":
                            "${GKE_TPU_ACCELERATOR:-tpu-v6e-slice}"
                        }
                    }]
                }),
                "v7",
            )
        # gce_tpu_vm on v7x must NOT inherit GKE_TPU_ACCELERATOR:-tpu-v6e-slice
        with patch.dict(os.environ, {
                "COMPUTE_BACKEND": "gce_tpu_vm",
                "GCE_TPU_NAME": "my-tpu7x-vm",
        }):
            self.assertEqual(
                resolve_accelerator_slug({
                    "agents": [{
                        "env": {
                            "COMPUTE_BACKEND":
                            "${COMPUTE_BACKEND:-gke_job}",
                            "GKE_TPU_ACCELERATOR":
                            "${GKE_TPU_ACCELERATOR:-tpu-v6e-slice}",
                        }
                    }]
                }),
                "v7",
            )
        # local CPU backend must NOT inherit GKE_TPU_ACCELERATOR:-tpu-v6e-slice
        with patch.dict(os.environ, {"COMPUTE_BACKEND": "local"}, clear=False):
            os.environ.pop("TPU_ACCELERATOR_TYPE", None)
            self.assertEqual(
                resolve_accelerator_slug({
                    "agents": [{
                        "env": {
                            "COMPUTE_BACKEND":
                            "${COMPUTE_BACKEND:-gke_job}",
                            "GKE_TPU_ACCELERATOR":
                            "${GKE_TPU_ACCELERATOR:-tpu-v6e-slice}",
                        }
                    }]
                }),
                "",
            )
        self.assertEqual(
            resolve_accelerator_slug({
                "agents": [{
                    "env": {
                        "GKE_TPU_ACCELERATOR":
                        "${GKE_TPU_ACCELERATOR:-tpu-v6e-slice}"
                    }
                }]
            }),
            "v6e",
        )

    def test_shared_capacity_config_generates_v7_job_name(self) -> None:
        args = argparse.Namespace(task=None,
                                  suite=None,
                                  dataset="smoke-check",
                                  job_name=None)
        cfg_path = CONFIGS_DIR / "agy" / "agent-suite-shared-capacity.yaml"
        cfg = load_config(cfg_path)
        name = generate_job_name(cfg,
                                 cfg_path,
                                 args,
                                 timestamp="20260930-050000")
        self.assertEqual(
            name, "smoke-check_agy-agent-suite-v7_local_20260930-050000")


class TestRunSuiteOption(unittest.TestCase):
    """Unit tests for -s / --suite evaluation suite option in run.py."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_suite_"))
        # main() asks git for the commit; these tests stub subprocess.run for
        # harbor, and TestExportGitRef covers the git side.
        git_ref = patch.object(_RUN_MOD, "export_git_ref")
        git_ref.start()
        self.addCleanup(git_ref.stop)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parser_suite_flags(self) -> None:
        parser = build_parser()
        args, _ = parser.parse_known_args(
            ["-s", "knowledgebase/tpu_knowledge_qa"])
        self.assertEqual(args.suite, "knowledgebase/tpu_knowledge_qa")

        args, _ = parser.parse_known_args(["--suite", "my_suite.yaml"])
        self.assertEqual(args.suite, "my_suite.yaml")

    def test_available_suites(self) -> None:
        suites = available_suites().split(", ")
        self.assertIn("knowledgebase/tpu_knowledge_qa", suites)
        # A directory of task files is listed once, not once per task.
        self.assertIn("cset/tpu_knowledge_qa", suites)
        self.assertNotIn("cset/tpu_knowledge_qa/csat002", suites)

    def test_resolve_suite_directory(self) -> None:
        resolved = resolve_suite("cset/tpu_knowledge_qa")
        self.assertTrue(resolved.is_dir())
        self.assertEqual(resolved,
                         (SUITES_DIR / "cset" / "tpu_knowledge_qa").resolve())

    def test_resolve_suite_bare_name(self) -> None:
        resolved = resolve_suite("knowledgebase/tpu_knowledge_qa")
        self.assertTrue(resolved.is_file())
        self.assertEqual(resolved.stem, "tpu_knowledge_qa")
        self.assertEqual(resolved.suffix, ".yaml")

    def test_resolve_suite_with_extension(self) -> None:
        resolved = resolve_suite("knowledgebase/tpu_knowledge_qa.yaml")
        self.assertTrue(resolved.is_file())
        self.assertEqual(resolved.name, "tpu_knowledge_qa.yaml")

    def test_resolve_suite_direct_path(self) -> None:
        suite_path = SUITES_DIR / "knowledgebase" / "tpu_knowledge_qa.yaml"
        resolved = resolve_suite(str(suite_path))
        self.assertEqual(resolved, suite_path.resolve())

    def test_resolve_suite_unknown_raises(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            resolve_suite("nonexistent_suite_name")
        self.assertIn("Suite 'nonexistent_suite_name' not found",
                      str(ctx.exception))
        self.assertIn("Available suites:", str(ctx.exception))

    def test_compile_suite_if_needed_compilation_and_caching(self) -> None:
        mock_suites_root = self.temp_dir / "suites"
        mock_sub_dir = mock_suites_root / "test_cat"
        mock_sub_dir.mkdir(parents=True)
        suite_yaml = mock_sub_dir / "test_suite.yaml"
        suite_yaml.write_text("name: test_suite\n", encoding="utf-8")

        mock_dot_datasets = self.temp_dir / ".datasets"

        with patch.object(_RUN_MOD, "SUITES_DIR", mock_suites_root), \
             patch.object(_RUN_MOD, "DOT_DATASETS_DIR", mock_dot_datasets), \
             patch("eval.suite_to_harbor.suite_to_harbor.compile_suite"
                   ) as mock_compile:

            def fake_compile(src, dst, task_name_filter=None):
                t_dir = dst / "test_task"
                t_dir.mkdir(parents=True, exist_ok=True)
                (t_dir / "task.toml").write_text("", encoding="utf-8")

            mock_compile.side_effect = fake_compile

            # 1. First compilation
            target_dir = compile_suite_if_needed(suite_yaml)
            self.assertEqual(target_dir,
                             mock_dot_datasets / "test_cat" / "test_suite")
            self.assertEqual(mock_compile.call_count, 1)

            # 2. Second call should reuse cached task.toml without recompiling
            target_dir2 = compile_suite_if_needed(suite_yaml)
            self.assertEqual(target_dir2, target_dir)
            self.assertEqual(mock_compile.call_count, 1)

            # 3. Call with new task_filter not present should recompile
            compile_suite_if_needed(suite_yaml, task_filter="another_task")
            self.assertEqual(mock_compile.call_count, 2)
            mock_compile.assert_called_with(suite_yaml,
                                            target_dir,
                                            task_name_filter="another_task")

            # 4. Touching suite_yaml should detect staleness and recompile
            import time
            time.sleep(0.01)
            suite_yaml.touch()
            compile_suite_if_needed(suite_yaml)
            self.assertEqual(mock_compile.call_count, 3)

    def test_compile_suite_if_needed_sibling_suites_are_independent(
            self) -> None:
        mock_suites_root = self.temp_dir / "suites"
        mock_sub_dir = mock_suites_root / "cset"
        mock_sub_dir.mkdir(parents=True)
        suite_a = mock_sub_dir / "suite_a.yaml"
        suite_b = mock_sub_dir / "suite_b.yaml"
        top_level = mock_suites_root / "top-level.yaml"
        for suite in (suite_a, suite_b, top_level):
            suite.write_text("suite_name: shared\n", encoding="utf-8")

        mock_dot_datasets = self.temp_dir / ".datasets"

        with patch.object(_RUN_MOD, "SUITES_DIR", mock_suites_root), \
             patch.object(_RUN_MOD, "DOT_DATASETS_DIR", mock_dot_datasets), \
             patch("eval.suite_to_harbor.suite_to_harbor.compile_suite"
                   ) as mock_compile:

            def fake_compile(src, dst, task_name_filter=None):
                t_dir = dst / src.stem
                t_dir.mkdir(parents=True, exist_ok=True)
                (t_dir / "task.toml").write_text("", encoding="utf-8")

            mock_compile.side_effect = fake_compile

            target_a = compile_suite_if_needed(suite_a)
            # suite_a's freshly compiled tasks must not satisfy suite_b.
            target_b = compile_suite_if_needed(suite_b)
            target_top = compile_suite_if_needed(top_level)

            self.assertEqual(target_a, mock_dot_datasets / "cset" / "suite_a")
            self.assertEqual(target_b, mock_dot_datasets / "cset" / "suite_b")
            self.assertEqual(target_top, mock_dot_datasets / "top-level")
            self.assertEqual(mock_compile.call_count, 3)

    def test_compile_suite_if_needed_directory_is_one_dataset(self) -> None:
        mock_suites_root = self.temp_dir / "suites"
        suite_dir = mock_suites_root / "cset" / "qa"
        suite_dir.mkdir(parents=True)
        task_a = suite_dir / "task_a.yaml"
        task_b = suite_dir / "task_b.yaml"
        for task in (task_a, task_b):
            task.write_text("suite_name: cset\n", encoding="utf-8")

        mock_dot_datasets = self.temp_dir / ".datasets"

        with patch.object(_RUN_MOD, "SUITES_DIR", mock_suites_root), \
             patch.object(_RUN_MOD, "DOT_DATASETS_DIR", mock_dot_datasets), \
             patch("eval.suite_to_harbor.suite_to_harbor.compile_suite"
                   ) as mock_compile:

            def fake_compile(src, dst, task_name_filter=None):
                t_dir = dst / src.stem
                t_dir.mkdir(parents=True, exist_ok=True)
                (t_dir / "task.toml").write_text("", encoding="utf-8")
                return [t_dir]

            mock_compile.side_effect = fake_compile

            target = compile_suite_if_needed(suite_dir)
            self.assertEqual(target, mock_dot_datasets / "cset" / "qa")
            self.assertEqual(mock_compile.call_count, 2)
            self.assertEqual(sorted(p.name for p in target.iterdir()),
                             ["task_a", "task_b"])

            # Cached: a task filter on a directory suite never recompiles.
            compile_suite_if_needed(suite_dir, task_filter="task_a")
            self.assertEqual(mock_compile.call_count, 2)

            # Removing a task file drops its compiled task.
            import time
            time.sleep(0.01)
            task_b.unlink()
            compile_suite_if_needed(suite_dir)
            self.assertEqual(mock_compile.call_count, 3)
            self.assertEqual([p.name for p in target.iterdir()], ["task_a"])

    def test_compile_suite_if_needed_directory_rejects_duplicate_tasks(
            self) -> None:
        mock_suites_root = self.temp_dir / "suites"
        suite_dir = mock_suites_root / "dup"
        suite_dir.mkdir(parents=True)
        for name in ("one.yaml", "two.yaml"):
            (suite_dir / name).write_text("suite_name: dup\n",
                                          encoding="utf-8")

        with patch.object(_RUN_MOD, "SUITES_DIR", mock_suites_root), \
             patch.object(_RUN_MOD, "DOT_DATASETS_DIR",
                          self.temp_dir / ".datasets"), \
             patch("eval.suite_to_harbor.suite_to_harbor.compile_suite"
                   ) as mock_compile:
            mock_compile.side_effect = lambda src, dst: [dst / "same"]
            with self.assertRaises(RuntimeError) as ctx:
                compile_suite_if_needed(suite_dir)
            self.assertIn("both compile to task same", str(ctx.exception))

    def test_cset_task_files_compile_to_harbor_dataset(self) -> None:
        mock_dot_datasets = self.temp_dir / ".datasets"
        with patch.object(_RUN_MOD, "DOT_DATASETS_DIR", mock_dot_datasets):
            target = compile_suite_if_needed(
                resolve_suite("cset/tpu_knowledge_qa"))
        self.assertEqual(target,
                         mock_dot_datasets / "cset" / "tpu_knowledge_qa")
        tasks = sorted(p.parent.name for p in target.glob("*/task.toml"))
        self.assertEqual(
            tasks,
            sorted(p.stem for p in (SUITES_DIR / "cset" /
                                    "tpu_knowledge_qa").glob("*.yaml")))

    def test_require_dataset_error_message_mentions_suite(self) -> None:
        config = {"job_name": "test"}
        with self.assertRaises(RuntimeError) as ctx:
            require_dataset(config, Path("dummy.yaml"))
        msg = str(ctx.exception)
        self.assertIn("--suite <name>", msg)
        self.assertIn("Available suites:", msg)

    def test_resolve_task_arg_with_suite_target_dir(self) -> None:
        target_dir = self.temp_dir / "knowledgebase"
        t_dir = target_dir / "my_task"
        t_dir.mkdir(parents=True)
        (t_dir / "task.toml").write_text("", encoding="utf-8")

        key, entry = resolve_task_arg("my_task", str(target_dir))
        self.assertEqual(key, "tasks")
        self.assertEqual(entry["path"], str(t_dir.resolve()))

    def test_main_with_suite_compiles_and_sets_dataset(self) -> None:
        fake_target_dir = self.temp_dir / "fake_dataset"
        fake_target_dir.mkdir()
        t_dir = fake_target_dir / "task1"
        t_dir.mkdir()
        (t_dir / "task.toml").write_text("", encoding="utf-8")

        dummy_kube = self.temp_dir / "kubeconfig"
        dummy_kube.write_text("apiVersion: v1\nclusters: []", encoding="utf-8")
        creds = _RUN_MOD.GkeCredentials(kubeconfig_path=dummy_kube,
                                        token="dummy-token")
        with patch("sys.argv", [
            "run.py", "-c", "claude-code/agent-suite", "-s",
            "knowledgebase/tpu_knowledge_qa", "--no-upload", "--no-report"
        ]), patch.dict(os.environ, {
            "HOST_ADC": str(self.temp_dir / "adc.json")
        }), patch.object(_RUN_MOD,
                         "compile_suite_if_needed",
                         return_value=fake_target_dir) as mock_compile, \
           patch.object(_RUN_MOD, "check_dependencies"), \
           patch.object(_RUN_MOD, "setup_host_auth"), \
           patch.object(_RUN_MOD, "setup_gke_credentials_if_needed", return_value=creds), \
           patch.object(_RUN_MOD, "setup_harbor_gke_kubeconfig_if_needed", return_value=dummy_kube), \
           patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            try:
                _RUN_MOD.main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)
            mock_compile.assert_called_once()

    def test_main_exports_agy_adc_auth_when_configured(self):
        """When AGY_ADC_AUTH is declared in config environment.env, export it to harbor_env."""
        cfg_file = self.temp_dir / "agy_adc.yaml"
        cfg_file.write_text(
            "agents:\n"
            "  - name: antigravity\n"
            "environment:\n"
            "  env:\n"
            "    AGY_ADC_AUTH: 'true'\n",
            encoding="utf-8",
        )
        task_dir = self.temp_dir / "task1"
        task_dir.mkdir()
        (task_dir / "task.toml").write_text("", encoding="utf-8")

        with patch("sys.argv", [
            "run.py", "-c", str(cfg_file), "-t", str(task_dir),
            "--no-upload", "--no-report"
        ]), patch.object(_RUN_MOD, "check_dependencies"), \
           patch.object(_RUN_MOD, "setup_host_auth"), \
           patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            try:
                _RUN_MOD.main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)

            harbor_calls = [
                c for c in mock_run.call_args_list
                if c.args and isinstance(c.args[0], list)
                and len(c.args[0]) >= 2 and Path(c.args[0][0]).name == "harbor"
                and c.args[0][1] == "run"
            ]
            self.assertEqual(len(harbor_calls), 1)
            called_env = harbor_calls[0].kwargs.get("env", {})
            self.assertEqual(called_env.get("AGY_ADC_AUTH"), "true")

    def test_main_stages_each_config_with_its_own_subset(self) -> None:
        # Each config's --skills is checked against its own package; the
        # second staging pass must use that config's subset, not the one
        # the last config left behind.
        configs = []
        for name in ("arm_a", "arm_b"):
            cfg_file = self.temp_dir / f"{name}.yaml"
            cfg_file.write_text("agents:\n  - name: claude-code\n",
                                encoding="utf-8")
            configs.append(str(cfg_file))
        task_dir = self.temp_dir / "task1"
        task_dir.mkdir()
        (task_dir / "task.toml").write_text("", encoding="utf-8")

        env = {k: v for k, v in os.environ.items() if k != "GITHUB_RUN_ID"}
        with patch("sys.argv", [
            "run.py", "-c", *configs, "-t", str(task_dir), "--skills",
            "tpu-agent", "--no-upload", "--no-report"
        ]), patch.dict(os.environ, env, clear=True), \
           patch.object(_RUN_MOD, "check_dependencies"), \
           patch.object(_RUN_MOD, "setup_host_auth"), \
           patch.object(_RUN_MOD, "skill_subset",
                        side_effect=[["skill-a"], ["skill-b"]]), \
           patch.object(_RUN_MOD, "stage_package_if_needed",
                        return_value=MountUpdates(mounts=[], env={})) as stage, \
           patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            try:
                _RUN_MOD.main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)

        staged = [(Path(c.args[1]).stem, c.args[2])
                  for c in stage.call_args_list]
        self.assertEqual([("arm_a", ["skill-a"]), ("arm_b", ["skill-b"]),
                          ("arm_a", ["skill-a"]), ("arm_b", ["skill-b"])],
                         staged)

    def test_main_passes_streaming_upload_flags_to_harbor_run(self) -> None:
        cfg_file = self.temp_dir / "upload_cfg.yaml"
        cfg_file.write_text(
            "agents:\n  - name: antigravity\n",
            encoding="utf-8",
        )
        task_dir = self.temp_dir / "task1"
        task_dir.mkdir()
        (task_dir / "task.toml").write_text("", encoding="utf-8")

        with patch(
                "sys.argv",
            [
                "run.py",
                "-c",
                str(cfg_file),
                "-t",
                str(task_dir),
                "--upload",
                "--share-org",
                "my-org",
                "--no-report",
            ],
        ), patch.object(_RUN_MOD, "check_dependencies"), \
             patch.object(_RUN_MOD, "setup_host_auth"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            try:
                _RUN_MOD.main()
            except SystemExit:
                pass

            harbor_run_calls = [
                c.args[0] for c in mock_run.call_args_list
                if c.args and isinstance(c.args[0], list)
                and len(c.args[0]) > 1 and Path(c.args[0][0]).name == "harbor"
                and c.args[0][1] == "run"
            ]
            self.assertEqual(len(harbor_run_calls), 1)
            run_cmd = harbor_run_calls[0]
            self.assertIn("--upload", run_cmd)
            self.assertIn("--private", run_cmd)
            self.assertIn("--yes", run_cmd)
            self.assertIn("--share-org", run_cmd)
            self.assertIn("my-org", run_cmd)

        with patch(
                "sys.argv",
            [
                "run.py",
                "-c",
                str(cfg_file),
                "-t",
                str(task_dir),
                "--upload",
                "--no-report",
                "--dry-run",
            ],
        ), patch.object(_RUN_MOD, "check_dependencies"), \
             patch.object(_RUN_MOD, "setup_host_auth"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            try:
                _RUN_MOD.main()
            except SystemExit:
                pass

            harbor_run_calls = [
                c.args[0] for c in mock_run.call_args_list
                if c.args and isinstance(c.args[0], list)
                and len(c.args[0]) > 1 and Path(c.args[0][0]).name == "harbor"
                and c.args[0][1] == "run"
            ]
            self.assertEqual(len(harbor_run_calls), 1)
            self.assertNotIn("--upload", harbor_run_calls[0])

    def test_main_salvages_completed_trials_on_interrupt(self) -> None:
        cfg_file = self.temp_dir / "salvage_cfg.yaml"
        cfg_file.write_text(
            "agents:\n  - name: antigravity\n",
            encoding="utf-8",
        )
        task_dir = self.temp_dir / "task1"
        task_dir.mkdir()
        (task_dir / "task.toml").write_text("", encoding="utf-8")
        summary_file = self.temp_dir / "step_summary.md"

        def fake_subprocess_run(cmd, *args, **kwargs):
            if (isinstance(cmd, list) and len(cmd) > 1
                    and Path(cmd[0]).name == "harbor" and cmd[1] == "run"):
                job_idx = cmd.index("--job-name")
                job_name = cmd[job_idx + 1]
                job_dir = _RUN_MOD.REPO_ROOT / "jobs" / job_name
                job_dir.mkdir(parents=True, exist_ok=True)
                (job_dir / "result.json").write_text(
                    json.dumps({"id": "job-uuid-salvage-999"}),
                    encoding="utf-8",
                )
                trial_dir = job_dir / "trial_1__abc"
                trial_dir.mkdir(parents=True, exist_ok=True)
                (trial_dir / "result.json").write_text(
                    json.dumps({
                        "trial_name": "trial_1__abc",
                        "verifier_result": {
                            "rewards": {
                                "reward": 1.0
                            }
                        },
                    }),
                    encoding="utf-8",
                )
                raise KeyboardInterrupt("Simulated SIGTERM")
            res = MagicMock()
            res.returncode = 0
            return res

        job_name_fixed = "test_salvage_job_20261004-000000"
        job_dir_path = _RUN_MOD.REPO_ROOT / "jobs" / job_name_fixed
        try:
            with patch(
                    "sys.argv",
                [
                    "run.py",
                    "-c",
                    str(cfg_file),
                    "-t",
                    str(task_dir),
                    "--upload",
                    "--no-report",
                ],
            ), patch.dict(
                    os.environ,
                {"GITHUB_STEP_SUMMARY": str(summary_file)},
                    clear=False,
            ), patch.object(_RUN_MOD, "check_dependencies"), \
                 patch.object(_RUN_MOD, "setup_host_auth"), \
                 patch.object(_RUN_MOD, "generate_job_name", return_value=job_name_fixed), \
                 patch("subprocess.run", side_effect=fake_subprocess_run) as mock_run:
                with self.assertRaises(SystemExit) as ctx:
                    _RUN_MOD.main()
                self.assertEqual(ctx.exception.code, 1)

                upload_calls = [
                    c.args[0] for c in mock_run.call_args_list
                    if c.args and isinstance(c.args[0], list) and len(
                        c.args[0]) > 1 and Path(c.args[0][0]).name == "harbor"
                    and c.args[0][1] == "upload"
                ]
                self.assertEqual(len(upload_calls), 1)
                self.assertTrue(summary_file.is_file())
                summary_text = summary_file.read_text(encoding="utf-8")
                self.assertIn("job-uuid-salvage-999", summary_text)
                self.assertIn("trial_1__abc", summary_text)
        finally:
            shutil.rmtree(job_dir_path, ignore_errors=True)

    def test_monitor_prints_harbor_hub_url_once(self) -> None:
        job_dir = self.temp_dir / "monitor_hub_job"
        job_dir.mkdir(parents=True)
        (job_dir / "result.json").write_text(
            json.dumps({"id": "hub-job-12345"}),
            encoding="utf-8",
        )
        monitor = _RUN_MOD.HarborJobMonitor(
            job_dir=job_dir,
            job_name="monitor_hub_job",
            heartbeat_interval_sec=600,
        )
        import io
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            monitor._poll()
            monitor._poll()
        out = buf.getvalue()
        self.assertEqual(out.count("==> Harbor Hub URL:"), 1)
        self.assertIn("hub-job-12345", out)

    def test_sitecustomize_scrubs_trial_before_trial_end_event(self) -> None:
        import asyncio

        from harbor.trial.hooks import TrialEvent
        from harbor.trial.trial import Trial as _HarborTrial

        from eval.adapter import sitecustomize

        self.assertTrue(
            getattr(_HarborTrial, "_eval_pre_emit_scrub_patched", False))

        order = []

        class DummyTrial:

            def _scrub_jobs_dir(self):
                order.append("scrub")

            async def _emit(self, event):
                order.append(f"emit:{event.value}")

        sitecustomize._patch_trial_pre_emit_scrub(DummyTrial, TrialEvent)
        dummy = DummyTrial()
        asyncio.run(DummyTrial._emit(dummy, TrialEvent.START))
        asyncio.run(DummyTrial._emit(dummy, TrialEvent.END))

        self.assertEqual(order, ["emit:start", "scrub", "emit:end"])

    def test_sitecustomize_fails_closed_when_scrub_throws_or_methods_missing(
            self) -> None:
        import asyncio

        from harbor.trial.hooks import TrialEvent

        from eval.adapter import sitecustomize

        order = []
        warnings = []

        class FakeLogger:

            def warning(self, msg: str) -> None:
                warnings.append(msg)

        class FailingScrubTrial:
            logger = FakeLogger()

            def _scrub_jobs_dir(self):
                order.append("scrub_failed")
                raise RuntimeError("disk error during scrub")

            async def _emit(self, event):
                order.append(f"emit:{event.value}")

        sitecustomize._patch_trial_pre_emit_scrub(FailingScrubTrial,
                                                  TrialEvent)
        asyncio.run(
            FailingScrubTrial._emit(FailingScrubTrial(), TrialEvent.END))

        # TrialEvent.END must NOT be emitted when _scrub_jobs_dir raises
        self.assertEqual(order, ["scrub_failed"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("skipping TrialEvent.END", warnings[0])

        # If Harbor renames _emit or _scrub_jobs_dir, patching raises RuntimeError
        class RenamedEmitTrial:

            def _scrub_jobs_dir(self):
                pass

        class RenamedScrubTrial:

            async def _emit(self, event):
                pass

        with self.assertRaises(RuntimeError):
            sitecustomize._patch_trial_pre_emit_scrub(RenamedEmitTrial,
                                                      TrialEvent)
        with self.assertRaises(RuntimeError):
            sitecustomize._patch_trial_pre_emit_scrub(RenamedScrubTrial,
                                                      TrialEvent)

    def test_sitecustomize_stages_suite_artifacts_before_collection(
            self) -> None:
        import asyncio
        from types import SimpleNamespace

        from eval.adapter import sitecustomize

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_art = root / "runs" / "r1" / "artifacts" / "sub"
            run_art.mkdir(parents=True)
            (run_art / "trace.xplane.pb").write_bytes(b"xplane")
            env_art = root / "logs" / "artifacts"

            async def _exec(cmd: str, **_kw):
                subprocess.run(["sh", "-c", cmd], check=True)
                return SimpleNamespace(return_code=0)

            class DummyTrial:
                agent_environment = SimpleNamespace(exec=_exec)
                agent_env_paths = SimpleNamespace(artifacts_dir=env_art)

                async def _collect_artifacts_phased(self, **_kw) -> None:
                    pass

            with patch.object(sitecustomize, "SUITE_RUNS_DIR",
                              str(root / "runs")):
                sitecustomize._patch_trial_stage_suite_artifacts(DummyTrial)
                asyncio.run(
                    DummyTrial._collect_artifacts_phased(DummyTrial(),
                                                         artifacts_dir=root /
                                                         "out"))
            self.assertEqual(
                (env_art / "sub" / "trace.xplane.pb").read_bytes(), b"xplane")


if __name__ == "__main__":
    unittest.main()


class TestNoSharedKnowledgeBaseCache(unittest.TestCase):
    """No config binds a host directory over the provenance cache.

    One host directory bound into every trial was one cache for all of them:
    with `n_concurrent_trials: 6` the first trial to build a page handed it
    to the rest, so a suite of overlapping questions scored most of them
    against pages a sibling had fetched. It was there to spare a
    rate-limited source, back when the lookup fetched pages itself over
    HTTP. It no longer does -- a url is summarised by the harness CLI, from
    the CLI's own backend -- so the mount only blurred what was measured.
    """

    def test_no_config_mounts_the_cache(self) -> None:
        for path in sorted(CONFIGS_DIR.rglob("*.yaml")):
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text())
                mounts = ((config.get("environment") or {}).get("mounts")
                          or [])
                targets = [
                    str(m.get("target")) for m in mounts
                    if isinstance(m, dict)
                ]
                self.assertFalse([
                    t for t in targets
                    if "tpu-agent-suite" in t or "provenance" in t
                ], "a shared cache mount is back")

    def test_the_tpu_agent_suite_home_matches_what_tasks_collect(self) -> None:
        """Every task.toml collects /root/.tpu-agent-suite/runs and nothing
        sets HOME, so the home has to resolve there or the harness
        collects an empty directory without saying so. Pinned, not
        relocated: compiled hook commands hardcode the ~/.tpu-agent-suite
        spelling and do not honour this variable, so a value disagreeing
        with ~ breaks them."""
        for path in sorted(CONFIGS_DIR.rglob("*.yaml")):
            config = yaml.safe_load(path.read_text())
            homes = [(block.get("env") or {}).get("TPU_AGENT_SUITE_HOME")
                     for block in (config.get("agents") or [])]
            with self.subTest(config=path.name):
                self.assertTrue(homes, "no agent in this config")
                for home in homes:
                    self.assertEqual(home, "/root/.tpu-agent-suite")
