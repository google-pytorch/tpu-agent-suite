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
"""Harbor-free tests for eval/job_naming.py & eval/verifiers/versions.yaml.

Designed to run in standard CI (`make test-eval`) using only Python stdlib + PyYAML.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import job_naming  # noqa: E402


class TestJobNamingCI(unittest.TestCase):

    def test_versions_registry_loads_and_has_valid_families(self) -> None:
        reg = job_naming.load_verifier_versions()
        self.assertIn("families", reg)
        families = reg["families"]
        self.assertIn("default", families)
        for fam_name, fam_cfg in families.items():
            cur = fam_cfg.get("current")
            self.assertTrue(cur, f"Family '{fam_name}' missing 'current'")
            history = fam_cfg.get("history") or {}
            self.assertIn(
                cur, history,
                f"Family '{fam_name}' current '{cur}' missing from history")
            info = job_naming.resolve_verifier_info(family=fam_name,
                                                    version=None)
            self.assertEqual(info["verifier_family"], fam_name)
            self.assertEqual(info["verifier_version"], cur)
            self.assertTrue(info["verifier_digest"].startswith("sha256:"))

    def test_versions_registry_no_unrecorded_drift(self) -> None:
        drift_items = job_naming.check_version_drift()
        drifted = [d["family"] for d in drift_items if d.get("drifted")]
        self.assertEqual(
            drifted,
            [],
            f"Verifier inputs drifted vs eval/verifiers/versions.yaml: {drifted}",
        )

    def test_naming_and_stamp_job_verifier_provenance(self) -> None:
        self.assertEqual(job_naming.normalize_version_tag("V02"), "v2")
        self.assertEqual(job_naming.parse_verifier_version("v12"), 12)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tests_dir = root / "tasks" / "sample-task" / "tests"
            tests_dir.mkdir(parents=True)
            (tests_dir / "test.sh").write_text("#!/bin/bash\necho ok\n",
                                               encoding="utf-8")

            job_dir = root / "sample_job"
            trial_dir = job_dir / "sample-task__t1"
            trial_dir.mkdir(parents=True)
            (job_dir / "result.json").write_text(json.dumps({"id": "job-1"}),
                                                 encoding="utf-8")
            (trial_dir / "result.json").write_text(
                json.dumps({
                    "trial_name": "sample-task__t1",
                    "task_name": "sample-task"
                }),
                encoding="utf-8",
            )

            job_naming.stamp_job_verifier_provenance(
                job_dir=job_dir,
                verifier_family="default",
                verifier_version="v2",
                task_search_roots=[root / "tasks"],
            )
            stamped_trial = json.loads(
                (trial_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(stamped_trial["verifier_family"], "default")
            self.assertEqual(stamped_trial["verifier_version"], "v2")
            self.assertTrue(
                stamped_trial["verifier_digest"].startswith("sha256:"))
            self.assertTrue((job_dir / "verifier_metadata.json").is_file())
            self.assertTrue(
                (trial_dir / "verifier" / "verifier_metadata.json").is_file())

    def test_bump_verifier_version_increments_records_commit_and_hashes(
            self) -> None:
        import yaml

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src_dir = root / "eval" / "verifiers" / "demo"
            src_dir.mkdir(parents=True)
            (src_dir / "score.py").write_text("SCORE = 2\n", encoding="utf-8")
            versions_path = root / "eval" / "verifiers" / "versions.yaml"
            versions_path.write_text(
                yaml.safe_dump(
                    {
                        "version": 1,
                        "families": {
                            "demo": {
                                "current": "v1",
                                "description": "Demo family",
                                "source_paths": ["eval/verifiers/demo"],
                                "history": {
                                    "v1": "initial"
                                },
                                "inputs_sha256": {
                                    "v1": "sha256:old"
                                },
                            }
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            res = job_naming.bump_verifier_version(
                "demo",
                commit="abc1234deadbeef",
                versions_path=versions_path,
                repo_root=root,
            )
            self.assertEqual(res["previous_version"], "v1")
            self.assertEqual(res["current_version"], "v2")
            self.assertEqual(res["commit"], "abc1234deadbeef")
            self.assertTrue(res["inputs_sha256"].startswith("sha256:"))

            drift = job_naming.check_version_drift(
                versions_path=versions_path,
                repo_root=root,
            )
            self.assertEqual(len(drift), 1)
            self.assertEqual(drift[0]["current_version"], "v2")
            self.assertFalse(drift[0]["drifted"])

            with self.assertRaises(KeyError):
                job_naming.bump_verifier_version(
                    "nonexistent",
                    versions_path=versions_path,
                    repo_root=root,
                )

            with self.assertRaises(ValueError):
                job_naming.resolve_verifier_info(
                    family="nonexistent",
                    versions_path=versions_path,
                    strict=True,
                )
            with self.assertRaises(ValueError):
                job_naming.resolve_verifier_info(
                    family="demo",
                    version="v99",
                    versions_path=versions_path,
                    strict=True,
                )

            import contextlib
            import io
            from unittest import mock

            err = io.StringIO()
            with (
                    mock.patch.object(job_naming,
                                      "_resolve_git_head_commit",
                                      return_value="111122223333"),
                    mock.patch.object(
                        job_naming,
                        "_dirty_source_paths",
                        return_value=["eval/verifiers/demo/score.py"]),
                    contextlib.redirect_stderr(err),
            ):
                job_naming.bump_verifier_version(
                    "demo",
                    versions_path=versions_path,
                    repo_root=root,
                )
            self.assertIn("uncommitted changes", err.getvalue())
            self.assertIn("Commit scoring changes before bumping",
                          err.getvalue())

    def test_dirty_source_paths_expands_glob_patterns(self) -> None:
        import subprocess

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            test_file = root / "eval" / "datasets" / "fam" / "task_a" / "tests" / "test.sh"
            test_file.parent.mkdir(parents=True)
            test_file.write_text("echo v1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
            subprocess.run([
                "git", "-C",
                str(root), "config", "user.email", "test@example.com"
            ],
                           check=True)
            subprocess.run(
                ["git", "-C",
                 str(root), "config", "user.name", "Test"],
                check=True)
            subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
            subprocess.run(
                ["git", "-C",
                 str(root), "commit", "-q", "-m", "init"],
                check=True)

            self.assertEqual(
                job_naming._dirty_source_paths(["eval/datasets/fam/*/tests"],
                                               repo_root=root),
                [],
            )
            test_file.write_text("echo v2\n", encoding="utf-8")
            self.assertEqual(
                job_naming._dirty_source_paths(["eval/datasets/fam/*/tests"],
                                               repo_root=root),
                ["eval/datasets/fam/task_a/tests/test.sh"],
            )

    def test_resolve_verifier_family_cset_vs_knowledgebase_and_drift_on_empty_tree(
            self) -> None:
        self.assertEqual(job_naming.resolve_verifier_family(None, ["csat007"]),
                         "cset")
        self.assertEqual(
            job_naming.resolve_verifier_family(
                "eval/suites/cset/tpu_knowledge_qa", ["csat002"]),
            "cset",
        )
        self.assertEqual(
            job_naming.resolve_verifier_family(None, ["knowledge-qa-csat006"]),
            "knowledgebase",
        )
        self.assertEqual(
            job_naming.resolve_verifier_family(
                "eval/suites/knowledgebase/tpu_knowledge_qa.yaml",
                ["csat006"]),
            "knowledgebase",
        )

        import yaml

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a").mkdir()
            (root / "b").mkdir()
            (root / "a" / "reward.toml").write_text("w = 1\n",
                                                    encoding="utf-8")
            (root / "b" / "reward.toml").write_text("w = 2\n",
                                                    encoding="utf-8")
            parsed = job_naming.parse_scoring_configs(
                source_paths=[root / "a", root / "b"])
            self.assertEqual(len(parsed["reward_toml"]), 2)

            vpath = root / "versions.yaml"
            vpath.write_text(
                yaml.safe_dump({
                    "version": 1,
                    "families": {
                        "demo": {
                            "current": "v1",
                            "source_paths": ["missing_dir"],
                            "history": {
                                "v1": "init"
                            },
                            "inputs_sha256": {
                                "v1": "sha256:abc"
                            },
                        }
                    },
                }),
                encoding="utf-8",
            )
            drift = job_naming.check_version_drift(versions_path=vpath,
                                                   repo_root=root)
            self.assertTrue(drift[0]["drifted"])

            import contextlib
            import io

            with contextlib.redirect_stdout(
                    io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    job_naming._cli([
                        "--versions-path",
                        str(vpath), "--check-verifier-versions"
                    ]), 1)
                self.assertEqual(
                    job_naming._cli(
                        ["--versions-path",
                         str(vpath), "--check-versions"]), 1)

    def test_resolve_verifier_family_routing_collisions(self) -> None:
        cases = [
            # CSET markers & word boundary vs knowledgebase q-csat006 collision
            ("cset", None, "cset"),
            (None, ["gpu_kernel_to_pytorch_api_conversion/task1"], "cset"),
            (None, ["hardware_profiling_and_debugging/task2"], "cset"),
            (None, ["kernel_optimization/task3"], "cset"),
            (None, ["q-csat006-kv-block"], "knowledgebase"),
            # Knowledgebase vs XProf marker precedence
            (None, ["ask-xprof-remote"], "knowledgebase"),
            (None, ["hosted-xprof"], "knowledgebase"),
            (None, ["xprof-capture"], "knowledgebase"),
            ("knowledgebase/tpu_knowledge_qa", None, "knowledgebase"),
            (None, ["tpu-knowledge-base/custom-q"], "knowledgebase"),
            # XProf markers
            ("xprof", None, "xprof"),
            (None, ["tpu-hello-world-xprof"], "xprof"),
            (None, ["xprof-attention-kernel-tuning"], "xprof"),
            (None, ["xprof-hbm"], "xprof"),
            (None, ["xprof-compare"], "xprof"),
            (None, ["xprof-summary"], "xprof"),
            # JAXBench markers & case slug regex
            ("jaxbench", None, "jaxbench"),
            ("suites/jaxbench/level1", None, "jaxbench"),
            (None, ["kernel-set1/foo"], "jaxbench"),
            (None, ["1p-softmax"], "jaxbench"),
            (None, ["32p-gemm"], "jaxbench"),
            # Onboarding routing must win before generic 'onboard'
            ("onboarding-routing", None, "onboarding_routing"),
            (None, ["optimization-heuristics"], "onboarding_routing"),
            (None, ["plain-tp-plan"], "onboarding_routing"),
            (None, ["why-tp-plan"], "onboarding_routing"),
            (None, ["verify-and-bench-0p6b"], "onboarding_routing"),
            ("onboard/vllm-gemma", None, "onboard"),
            # Hillclimb & close-gaps vs torch-tpu-inference false match
            (None, ["nightly-close-gaps/close-gaps-gemma-4-31b-it"],
             "onboard_hillclimb"),
            (None, ["close-gaps-gates"], "onboard_hillclimb"),
            (None, ["tpu-inference-vllm"], "onboard_hillclimb"),
            (None, ["torch-tpu-inference"], "default"),
            # Negative word-boundary check & underscore delimiter
            (None, ["abcsetxyz"], "default"),
            (None, ["eval_cset_suite"], "cset"),
        ]
        for target, tasks, expected in cases:
            with self.subTest(target=target, tasks=tasks):
                self.assertEqual(
                    job_naming.resolve_verifier_family(target, tasks),
                    expected,
                )

    def test_expand_source_paths_and_dirty_paths_respect_exclusions(
        self, ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keep_file = root / "eval" / "verifiers" / "fam" / "score.py"
            skip_dir_file = (root / "eval" / "verifiers" / "fam" / "fixtures" /
                             "sample.json")
            skip_md_file = root / "eval" / "verifiers" / "fam" / "notes.md"
            for fpath in (keep_file, skip_dir_file, skip_md_file):
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text("data\n", encoding="utf-8")

            specs = [
                "eval/verifiers/fam/*.py",
                "eval/verifiers/fam/**/*.json",
                "eval/verifiers/fam/*.md",
                "!eval/verifiers/fam/fixtures/**",
                "!eval/verifiers/fam/*.md",
            ]
            expanded = job_naming._expand_repo_source_paths(specs,
                                                            repo_root=root)
            self.assertEqual(expanded, [keep_file.resolve()])

            # Verify directory recursion in parse_scoring_configs also honors !exclusions
            dir_specs = [
                "eval/verifiers/fam",
                "!eval/verifiers/fam/fixtures/**",
                "!eval/verifiers/fam/*.md",
            ]
            ver_file = root / "eval" / "verifiers" / "versions.yaml"
            ver_file.write_text(
                "version: 1\n"
                "families:\n"
                "  fam:\n"
                "    current: v1\n"
                "    history:\n"
                "      v1: init\n"
                "    source_paths:\n"
                "      - eval/verifiers/fam\n"
                "      - '!eval/verifiers/fam/fixtures/**'\n"
                "      - '!eval/verifiers/fam/*.md'\n",
                encoding="utf-8",
            )
            manifest = job_naming.compute_family_inputs_manifest(
                "fam",
                versions_path=ver_file,
                repo_root=root,
            )
            self.assertIn("eval/verifiers/fam/score.py",
                          manifest["file_hashes"])
            self.assertNotIn("eval/verifiers/fam/fixtures/sample.json",
                             manifest["file_hashes"])
            self.assertNotIn("eval/verifiers/fam/notes.md",
                             manifest["file_hashes"])

            # Verify _dirty_source_paths honors !exclusions in real git repo
            for cmd in (
                ["git", "-C", str(root), "init"],
                ["git", "-C", str(root), "add", "."],
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "-m",
                    "init",
                ],
            ):
                proc = subprocess.run(cmd,
                                      capture_output=True,
                                      text=True,
                                      check=False)
                self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            for fpath in (keep_file, skip_dir_file, skip_md_file):
                fpath.write_text("modified\n", encoding="utf-8")
            dirty = job_naming._dirty_source_paths(dir_specs, repo_root=root)
            self.assertEqual(dirty, ["eval/verifiers/fam/score.py"])

    def test_compute_scoring_inputs_manifest_decouples_sibling_trials(
        self, ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / "eval" / "verifiers" / "fam" / "score.py"
            shared.parent.mkdir(parents=True, exist_ok=True)
            shared.write_text("WEIGHT = 1.0\n", encoding="utf-8")
            ver_file = root / "eval" / "verifiers" / "versions.yaml"
            ver_file.write_text(
                "version: 1\n"
                "families:\n"
                "  fam:\n"
                "    current: v1\n"
                "    history:\n"
                "      v1: init\n"
                "    source_paths:\n"
                "      - eval/verifiers/fam\n",
                encoding="utf-8",
            )

            task_a_tests = root / "eval" / "datasets" / "task-a" / "tests"
            task_b_tests = root / "eval" / "datasets" / "task-b" / "tests"
            task_a_tests.mkdir(parents=True, exist_ok=True)
            task_b_tests.mkdir(parents=True, exist_ok=True)
            (task_a_tests / "test.sh").write_text("echo A\n", encoding="utf-8")
            (task_b_tests / "test.sh").write_text("echo B\n", encoding="utf-8")

            m_a1 = job_naming.compute_scoring_inputs_manifest(
                family="fam",
                tests_dir=task_a_tests,
                versions_path=ver_file,
                repo_root=root,
            )
            m_b1 = job_naming.compute_scoring_inputs_manifest(
                family="fam",
                tests_dir=task_b_tests,
                versions_path=ver_file,
                repo_root=root,
            )
            self.assertEqual(
                m_a1["family_inputs_sha256"],
                m_b1["family_inputs_sha256"],
            )
            self.assertNotEqual(m_a1["verifier_digest"],
                                m_b1["verifier_digest"])

            # Mutating sibling B must only change B's per-trial digest, not A's or family hash.
            (task_b_tests / "test.sh").write_text("echo B2\n",
                                                  encoding="utf-8")
            m_a2 = job_naming.compute_scoring_inputs_manifest(
                family="fam",
                tests_dir=task_a_tests,
                versions_path=ver_file,
                repo_root=root,
            )
            m_b2 = job_naming.compute_scoring_inputs_manifest(
                family="fam",
                tests_dir=task_b_tests,
                versions_path=ver_file,
                repo_root=root,
            )
            self.assertEqual(m_a2["verifier_digest"], m_a1["verifier_digest"])
            self.assertEqual(
                m_a2["family_inputs_sha256"],
                m_a1["family_inputs_sha256"],
            )
            self.assertNotEqual(m_b2["verifier_digest"],
                                m_b1["verifier_digest"])

            # Mutating shared family input changes family hash and both trial digests.
            shared.write_text("WEIGHT = 2.0\n", encoding="utf-8")
            m_a3 = job_naming.compute_scoring_inputs_manifest(
                family="fam",
                tests_dir=task_a_tests,
                versions_path=ver_file,
                repo_root=root,
            )
            self.assertNotEqual(
                m_a3["family_inputs_sha256"],
                m_a1["family_inputs_sha256"],
            )
            self.assertNotEqual(m_a3["verifier_digest"],
                                m_a1["verifier_digest"])

    def test_resolve_trial_tests_dir_container_paths_and_nested_depths(
        self, ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_repo = Path(tmp).resolve()
            legacy_tests = (fake_repo / "eval" / ".datasets" / "legacy-ds" /
                            "task-d0" / "tests")
            deep_tests = (fake_repo / "eval" / "datasets" / "grp" / "sub" /
                          "deep-task" / "tests")
            flat_tests = fake_repo / "eval" / "datasets" / "flat-task" / "tests"
            for tdir in (legacy_tests, deep_tests, flat_tests):
                tdir.mkdir(parents=True, exist_ok=True)
                (tdir / "test.sh").write_text("#!/bin/sh\n", encoding="utf-8")

            with mock.patch.object(job_naming, "REPO_ROOT", fake_repo):
                # 1. Container path mapping for eval/.datasets/
                t_legacy = fake_repo / "job" / "task-d0__1"
                t_legacy.mkdir(parents=True)
                (t_legacy / "result.json").write_text(
                    json.dumps({
                        "task_name": "task-d0",
                        "config": {
                            "task": {
                                "path": "/app/eval/.datasets/legacy-ds/task-d0"
                            }
                        },
                    }),
                    encoding="utf-8",
                )
                self.assertEqual(
                    job_naming._resolve_trial_tests_dir(t_legacy),
                    legacy_tests.resolve(),
                )

                # 2. Container path mapping for /workspace/eval/datasets/ nested depth
                t_deep = fake_repo / "job" / "deep-task__2"
                t_deep.mkdir(parents=True)
                (t_deep / "result.json").write_text(
                    json.dumps({
                        "task_name": "deep-task",
                        "config": {
                            "task": {
                                "path":
                                "/workspace/eval/datasets/grp/sub/deep-task"
                            }
                        },
                    }),
                    encoding="utf-8",
                )
                self.assertEqual(
                    job_naming._resolve_trial_tests_dir(t_deep),
                    deep_tests.resolve(),
                )

                # 3. Name-based fallback searching 3 levels under eval/datasets & eval/.datasets
                t_fallback_deep = fake_repo / "job" / "deep-task__3"
                t_fallback_deep.mkdir(parents=True)
                (t_fallback_deep / "result.json").write_text(
                    json.dumps({"task_name": "deep-task"}),
                    encoding="utf-8",
                )
                self.assertEqual(
                    job_naming._resolve_trial_tests_dir(t_fallback_deep),
                    deep_tests.resolve(),
                )
                t_fallback_legacy = fake_repo / "job" / "task-d0__4"
                t_fallback_legacy.mkdir(parents=True)
                (t_fallback_legacy / "result.json").write_text(
                    json.dumps({"task_name": "task-d0"}),
                    encoding="utf-8",
                )
                self.assertEqual(
                    job_naming._resolve_trial_tests_dir(t_fallback_legacy),
                    legacy_tests.resolve(),
                )
                t_fallback_flat = fake_repo / "job" / "flat-task__5"
                t_fallback_flat.mkdir(parents=True)
                (t_fallback_flat / "result.json").write_text(
                    json.dumps({"task_name": "flat-task"}),
                    encoding="utf-8",
                )
                self.assertEqual(
                    job_naming._resolve_trial_tests_dir(t_fallback_flat),
                    flat_tests.resolve(),
                )

    def test_stamp_trial_verifier_provenance_and_family_drift_filter(
        self, ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "test-job"
            t1 = job_dir / "task_one__1"
            (t1 / "verifier").mkdir(parents=True, exist_ok=True)
            (t1 / "result.json").write_text(
                json.dumps({
                    "id": "uuid-t1",
                    "trial_name": t1.name,
                    "task_name": "ask-xprof-remote",
                    "started_at": "2026-11-10T00:00:00Z",
                    "finished_at": "2026-11-10T00:05:00Z",
                    "verifier_result": {
                        "rewards": {
                            "reward": 1.0
                        }
                    },
                }),
                encoding="utf-8",
            )
            (job_dir / "result.json").write_text(
                json.dumps({
                    "id": "uuid-j1",
                    "started_at": "2026-11-10T00:00:00Z",
                }),
                encoding="utf-8",
            )

            manifest = job_naming.compute_scoring_inputs_manifest(
                family="xprof")
            meta = job_naming.stamp_trial_verifier_provenance(
                trial_dir=t1,
                verifier_family="xprof",
                verifier_version="v2",
                verifier_commit="abc1234",
                job_dir=job_dir,
                scoring_manifest=manifest,
            )
            self.assertEqual(meta["verifier_version"], "v2")
            self.assertEqual(meta["trial_id"], "uuid-t1")
            t1_res = json.loads(
                (t1 / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(t1_res["verifier_version"], "v2")
            self.assertEqual(t1_res["started_at"], "2026-11-10T00:00:00Z")
            self.assertEqual(t1_res["finished_at"], "2026-11-10T00:05:00Z")
            self.assertEqual(t1_res["verifier_result"]["rewards"]["reward"],
                             1.0)
            self.assertTrue(
                (t1 / "verifier" / "verifier_metadata.json").is_file())

            job_meta = job_naming.stamp_job_verifier_provenance(
                job_dir=job_dir,
                verifier_family="xprof",
                verifier_version="v2",
                verifier_commit="abc1234",
                scoring_manifest=manifest,
            )
            self.assertEqual(job_meta["verifier_version"], "v2")
            job_res = json.loads(
                (job_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(job_res["verifier_family"], "xprof")
            self.assertEqual(job_res["verifier_version"], "v2")
            self.assertEqual(job_res["started_at"], "2026-11-10T00:00:00Z")
            self.assertTrue((job_dir / "verifier_metadata.json").is_file())

            single_fam = job_naming.check_version_drift(family="xprof")
            self.assertEqual([d["family"] for d in single_fam], ["xprof"])
            self.assertFalse(single_fam[0]["drifted"])


if __name__ == "__main__":
    unittest.main()
