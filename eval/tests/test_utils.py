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
"""Which configs are control arms, which package each installs, and that none selects skills."""

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from eval.utils import (RELEASE_MANIFEST, find_repo_root, harness_for_config,
                        installs_at_user_scope, is_control_arm, package_label,
                        package_manifest, read_manifest, released_skills,
                        resolve_manifest, skill_subset)

CONFIGS = find_repo_root() / "eval" / "configs"

# Written out, so a control arm named without `baseline` -- which would
# quietly install the whole package -- fails here rather than in a score.
CONTROL_ARMS = {
    "agy/baseline.yaml",
    "agy/baseline-remote.yaml",
    "agy/baseline-shared-capacity-remote.yaml",
    "claude-code/baseline.yaml",
    "claude-code/baseline-remote.yaml",
    "claude-code/baseline-shared-capacity-remote.yaml",
}

EXPERIMENTAL_MANIFEST = "package.experimental.json"


class TestIsControlArm(unittest.TestCase):

    def configs(self) -> list[Path]:
        return sorted(CONFIGS.rglob("*.yaml"))

    def test_exactly_the_baselines_are_control_arms(self) -> None:
        found = {
            str(p.relative_to(CONFIGS))
            for p in self.configs() if is_control_arm(p)
        }
        self.assertEqual(CONTROL_ARMS, found)

    def test_a_baseline_part_is_needed_not_a_substring(self) -> None:
        self.assertTrue(is_control_arm("x/true_baseline_gke.yaml"))
        self.assertTrue(is_control_arm("x/baseline-opus.yaml"))
        self.assertTrue(is_control_arm("x/opus-Baseline.yaml"))
        self.assertFalse(is_control_arm("x/baselinez_v1.yaml"))
        self.assertFalse(is_control_arm("baseline/agent-suite.yaml"))

    def test_each_pair_matches_apart_from_job_name(self) -> None:
        for harness in ("claude-code", "agy"):
            configs = {
                name:
                yaml.safe_load(
                    (CONFIGS / harness / f"{name}.yaml").read_text())
                for name in ("agent-suite", "baseline", "agent-suite-remote",
                             "baseline-remote",
                             "agent-suite-shared-capacity-remote",
                             "baseline-shared-capacity-remote")
            }
            for config in configs.values():
                config.pop("job_name")
            for base, name in (("agent-suite", "baseline"),
                               ("agent-suite-remote", "baseline-remote"),
                               ("agent-suite-shared-capacity-remote",
                                "baseline-shared-capacity-remote")):
                with self.subTest(harness=harness, config=name):
                    self.assertEqual(configs[base], configs[name])

    def test_each_remote_config_matches_its_local_one_apart_from_where(
            self) -> None:
        for path in self.configs():
            if not path.stem.endswith("-remote"):
                continue
            local_path = path.with_name(
                path.stem.removesuffix("-remote") + ".yaml")
            if not local_path.exists():
                local_path = path.with_name(
                    path.stem.replace("baseline-", "agent-suite-",
                                      1).removesuffix("-remote") + ".yaml")
            remote = yaml.safe_load(path.read_text())
            local = yaml.safe_load(local_path.read_text())
            with self.subTest(config=str(path.relative_to(CONFIGS))):
                self.assertEqual("gke", remote["environment"].pop("type"))
                for key in ("force_build", "kwargs"):
                    remote["environment"].pop(key)
                for key in ("agent_setup_timeout_multiplier",
                            "timeout_multiplier"):
                    remote.pop(key)
                for config in (remote, local):
                    config.pop("job_name")
                self.assertEqual(local, remote)

    def test_every_agy_config_authenticates_with_adc(self) -> None:
        adc = "/root/.config/gcloud/application_default_credentials.json"
        for path in sorted((CONFIGS / "agy").glob("*.yaml")):
            config = yaml.safe_load(path.read_text())
            environment = config["environment"]
            with self.subTest(config=path.name):
                self.assertIn(adc,
                              [m["target"] for m in environment["mounts"]])
                self.assertEqual(
                    adc, config["agents"][0]["env"]
                    ["GOOGLE_APPLICATION_CREDENTIALS"])
                self.assertEqual("true", environment["env"]["AGY_ADC_AUTH"])

    def test_no_config_selects_skills(self) -> None:
        for path in self.configs():
            config = yaml.safe_load(path.read_text()) or {}
            with self.subTest(config=str(path.relative_to(CONFIGS))):
                for key in ("stage_skills", "packages", "package_dir"):
                    self.assertNotIn(key, config)
                for agent in config.get("agents") or []:
                    self.assertNotIn("package_dir", agent.get("kwargs") or {})


class TestHarnessForConfig(unittest.TestCase):

    def test_each_config_directory_is_its_harness(self) -> None:
        for directory, harness in (("claude-code", "claude-code"),
                                   ("agy", "antigravity"), ("codex", "codex")):
            for path in sorted((CONFIGS / directory).glob("*.yaml")):
                with self.subTest(config=str(path.relative_to(CONFIGS))):
                    self.assertEqual(
                        harness,
                        harness_for_config(yaml.safe_load(path.read_text())))


class TestPackageManifest(unittest.TestCase):
    """Which package a config installs: the manifest its PACKAGE_MANIFEST names."""

    def test_every_config_defaults_to_the_release_package(self) -> None:
        # Overridable from the environment, release when it is not set.
        for path in sorted(CONFIGS.rglob("*.yaml")):
            with self.subTest(config=str(path.relative_to(CONFIGS))):
                env = yaml.safe_load(path.read_text())["agents"][0]["env"]
                self.assertEqual("${PACKAGE_MANIFEST:-package.release.json}",
                                 env.get("PACKAGE_MANIFEST"))

    def test_unset_or_empty_is_release(self) -> None:
        self.assertEqual(RELEASE_MANIFEST, package_manifest({}))
        self.assertEqual(RELEASE_MANIFEST,
                         package_manifest({"PACKAGE_MANIFEST": ""}))
        self.assertEqual(
            EXPERIMENTAL_MANIFEST,
            package_manifest(
                {"PACKAGE_MANIFEST": f" {EXPERIMENTAL_MANIFEST} "}))

    def test_any_manifest_in_package_is_chosen_by_its_file_name(self) -> None:
        for path in sorted(
            (find_repo_root() / "package").glob("package.*.json")):
            with self.subTest(manifest=path.name):
                self.assertEqual(path.name, resolve_manifest(path.name))

    def test_a_value_that_is_not_a_manifest_is_refused(self) -> None:
        # Not read as release: a typo would otherwise run the wrong package,
        # and a path must not reach outside package/.
        for manifest in ("package.experimentl.json", "experimental", "0.6.0",
                         "../package/package.release.json",
                         "package/package.release.json", ""):
            with self.subTest(manifest=manifest):
                with self.assertRaisesRegex(
                        ValueError, "available: .*package.release.json"):
                    resolve_manifest(manifest)

    def test_another_json_file_in_package_is_not_a_manifest(self) -> None:
        # Only package.<label>.json: package_label needs that shape, and a
        # schema or data file dropped in package/ must not be chosen.
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "package"
            package.mkdir()
            for name in (RELEASE_MANIFEST, "other.json", "package.json"):
                (package / name).write_text("{}")
            self.assertEqual(RELEASE_MANIFEST,
                             resolve_manifest(RELEASE_MANIFEST, Path(tmp)))
            for name in ("other.json", "package.json"):
                with self.subTest(manifest=name):
                    with self.assertRaisesRegex(
                            ValueError, r"available: package\.release\.json$"):
                        resolve_manifest(name, Path(tmp))

    def test_a_job_name_label_comes_from_the_file_name(self) -> None:
        self.assertEqual("experimental", package_label(EXPERIMENTAL_MANIFEST))
        self.assertEqual("other", package_label("other.json"))

    def test_the_experimental_package_adds_to_the_release_one(self) -> None:
        release = read_manifest(manifest=RELEASE_MANIFEST)
        experimental = read_manifest(manifest=EXPERIMENTAL_MANIFEST)
        raw = json.loads((find_repo_root() / "package" /
                          EXPERIMENTAL_MANIFEST).read_text(encoding="utf-8"))
        self.assertEqual("tpu-agent-suite-experimental", experimental["name"])
        self.assertEqual(release["version"], experimental["version"])
        self.assertEqual(release["skills"] + raw["add_skills"],
                         experimental["skills"])
        self.assertIn("close-gaps",
                      released_skills(manifest=EXPERIMENTAL_MANIFEST))
        self.assertNotIn("close-gaps", released_skills())


class TestInstallsAtUserScope(unittest.TestCase):

    def test_only_the_user_scope_configs_install_in_the_trial(self) -> None:
        for path in sorted(CONFIGS.rglob("*.yaml")):
            with self.subTest(config=str(path.relative_to(CONFIGS))):
                self.assertEqual(
                    path.stem.startswith("user-scope"),
                    installs_at_user_scope(yaml.safe_load(path.read_text())))


class TestSkillSubset(unittest.TestCase):
    """The subset mode's skills are checked against what package.release.json releases."""

    RELEASED = ["tpu-agent", "tpu-query-understanding"]

    def test_comma_and_space_separated_are_the_same(self) -> None:
        self.assertEqual(
            self.RELEASED,
            skill_subset(["tpu-query-understanding,tpu-agent"], self.RELEASED))
        self.assertEqual(
            self.RELEASED,
            skill_subset(["tpu-agent", "tpu-query-understanding"],
                         self.RELEASED))

    def test_an_unknown_or_empty_subset_is_refused(self) -> None:
        for names in (["tpu-nope"], [","]):
            with self.subTest(names=names):
                with self.assertRaises(ValueError):
                    skill_subset(names, self.RELEASED)

    def test_the_released_skills_include_the_entrypoint(self) -> None:
        self.assertIn("tpu-agent", released_skills())


if __name__ == "__main__":
    unittest.main()
