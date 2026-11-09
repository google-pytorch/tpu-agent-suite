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
"""Tests for shipping the export-trajectory skill: emission, install, config.

Covers the packaging half of trajectory export -- that a manifest-declared
skill is rendered, installed on every harness, and kept consistent with its
instruction document.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from framework import lint, runtime, utils
from framework.builder import dist_assets

Path = pathlib.Path
REPO_PACKAGE = Path(__file__).resolve().parents[2] / "package"


class SkillFrontmatterLintTest(unittest.TestCase):
    """Manifest and skill document may not drift apart.

    The agent half of this check is covered in test_lint.py; a skill takes
    its own branch (its own name check, its own existence check, and the
    skill tool list), which is what these cases pin.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.manifest = {
            "name":
            "p",
            "version":
            "1.0.0",
            "description":
            "package description",
            "agents": [],
            "skills": [{
                "name": "p-export",
                "description": "save the trajectory",
                "instructions": "skills/p-export/SKILL.md",
            }],
        }

    def test_skill_drift_is_reported(self) -> None:
        cases = (
            (None, "skill instructions file does not exist"),
            ("---\nname: other\ndescription: d\n---\n\nBody\n",
             "does not match skill name"),
            ("---\nname: p-export\ndescription: d\n"
             "tools:\n  - read\n  - bash\n---\n\nBody\n",
             "invalid tool grants: ['bash']"),
        )
        doc = self.tmp / "skills" / "p-export" / "SKILL.md"
        for body, expected in cases:
            with self.subTest(expected=expected):
                if body is None:
                    doc.unlink(missing_ok=True)
                else:
                    doc.parent.mkdir(parents=True, exist_ok=True)
                    doc.write_text(body, encoding="utf-8")
                findings = lint.check_frontmatter_contracts(
                    self.tmp, self.manifest)
                self.assertTrue(any(expected in f.detail for f in findings),
                                findings)


class ShippedPackageTest(unittest.TestCase):
    """The real package declares the export-trajectory skill coherently."""

    def test_skill_body_drives_the_runtime_not_the_model(self) -> None:
        """The body is one runtime call, with tokens rather than literals."""
        body = (REPO_PACKAGE / "skills" / "tpu-agent-export-trajectory" /
                "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("{{runtime}} trajectory export", body)
        self.assertIn("--harness {{harness}}", body)
        # The consent flag must never be presented as something to add freely.
        self.assertIn("Never add `--all` on your own initiative", body)

    def test_sharing_is_declined_by_default(self) -> None:
        """One gate gets the whole feature, and it defaults to off."""
        questions = json.loads((REPO_PACKAGE / "environment" /
                                "questions.json").read_text())["questions"]
        by_name = {q["name"]: q for q in questions}
        self.assertIn("trajectory_enabled", by_name)
        self.assertIs(by_name["trajectory_enabled"]["default"], False)
        # Every other trajectory question hangs off that gate, directly or
        # through the sink, so declining it ends the flow.
        for name, question in by_name.items():
            if name.startswith("trajectory") and name != "trajectory_enabled":
                with self.subTest(question=name):
                    self.assertIn("when", question)

    def test_environment_schema_accepts_a_declined_bucket(self) -> None:
        """An empty bucket must validate, or declining breaks the install."""
        schema = json.loads(
            (REPO_PACKAGE / "schemas" / "environment.json").read_text())
        params = schema["properties"]["trajectory"]["properties"]["params"]
        self.assertEqual(params["properties"]["bucket"]["type"], "string")
        self.assertNotIn("pattern", params["properties"]["bucket"])
        self.assertNotIn("minLength", params["properties"]["bucket"])


class UserIdResolutionTest(unittest.TestCase):
    """Which identifier uploaded trajectories are filed under."""

    def test_configured_wins_then_gcloud_unset_falls_back_to_user(
            self) -> None:
        self.assertEqual(runtime.resolve_trajectory_user_id(" me@x.com "),
                         "me@x.com")
        unset = subprocess.CompletedProcess([], 0, stdout="(unset)\n")
        with mock.patch.object(runtime.subprocess, "run", return_value=unset), \
                mock.patch.dict(os.environ, {"USER": "alice"}):
            self.assertEqual(runtime.resolve_trajectory_user_id(""), "alice")


class ShippedFrameworkTest(unittest.TestCase):
    """framework/ as a build ships it imports and runs with nothing else.

    In the repository the tests also see the repository root and the build
    tooling; a copy made the way the compiler makes it, run on its own,
    shows the shipped runtime needs neither.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.package = self.tmp / "pkg"
        dist_assets.bundle_framework(self.package / "framework")
        (self.package / "tpu-agent-suite-build.json").write_text(
            json.dumps({
                "version": utils.TPU_AGENT_SUITE_VERSION,
                "source": "tpu-agent-suite"
            }),
            encoding="utf-8",
        )
        self.elsewhere = self.tmp / "elsewhere"
        self.elsewhere.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        return subprocess.run([sys.executable, *argv],
                              cwd=cwd,
                              env=env,
                              capture_output=True,
                              text=True,
                              timeout=60)

    def test_the_build_tooling_is_left_out(self) -> None:
        for name in ("compiler.py", "builder", "lint.py", "frontmatter.py",
                     "tests", "schemas/package-manifest.schema.json",
                     "schemas/skill-frontmatter.schema.json"):
            self.assertFalse((self.package / "framework" / name).exists(),
                             name)

    def test_gcs_helper_resolves(self) -> None:
        """The lazy gcs_staging import finds its sibling package."""
        result = self._run([
            "-c", "from framework.trajectory import bundle as b; "
            "print(b._clean_bucket('gs://x/'))"
        ], self.package)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "x")

    def test_runtime_runs_as_a_script_from_anywhere(self) -> None:
        """The hooks run framework/runtime.py by path, from any directory."""
        result = self._run([
            str(self.package / "framework" / "runtime.py"), "trajectory",
            "export", "--harness", "codex"
        ], self.elsewhere)
        self.assertEqual(result.returncode, 1)
        self.assertIn("not supported", result.stderr)


if __name__ == "__main__":
    unittest.main()
