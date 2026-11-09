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
"""Cursor translation parity for the tpu-agent-suite package."""

from __future__ import annotations

import json
import pathlib
import shutil
import sys
import tempfile
import unittest

# Ensure repository root directory is in sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.builder import translate  # noqa: E402

Path = pathlib.Path
PACKAGE_DIR = REPO_ROOT / "package"


class TestCursorTranslationParity(unittest.TestCase):
    """Validates that package/ translates correctly to Cursor."""

    HARNESS = "cursor"
    TARGET_ID = "cursor-0.40"
    VERSION_RANGE = ">=0.40.0,<1.0.0"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="torchtpu_test_cursor_"))
        cls.dist_dir = cls.temp_dir / "dist_cursor"
        translate(PACKAGE_DIR,
                  cls.HARNESS,
                  cls.dist_dir,
                  include_internal=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
        super().tearDownClass()

    def test_build_metadata(self):
        build_path = self.dist_dir / "tpu-agent-suite-build.json"
        self.assertTrue(build_path.is_file())
        build = json.loads(build_path.read_text(encoding="utf-8"))
        self.assertEqual(build["harness"], self.HARNESS)
        self.assertEqual(build["compatibility_target"], self.TARGET_ID)
        self.assertEqual(build["version_range"], self.VERSION_RANGE)

    def test_install_plan_structure(self):
        plan_path = self.dist_dir / "tpu-agent-suite-install.json"
        self.assertTrue(plan_path.is_file())
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(plan["harness"], self.HARNESS)
        self.assertEqual(plan["version_range"], self.VERSION_RANGE)
        self.assertEqual(plan["version_command"], ["cursor", "--version"])

        copy_dests = [c["destination"] for c in plan["copies"]]
        self.assertIn(".cursor/skills/tpu-agent", copy_dests)
        self.assertIn(".cursor/rules/tpu-agent-suite.mdc", copy_dests)

    def test_cursor_rules_generated(self):
        rule_file = self.dist_dir / ".cursor" / "rules" / "tpu-agent-suite.mdc"
        self.assertTrue(rule_file.is_file())
        content = rule_file.read_text(encoding="utf-8")
        self.assertIn("alwaysApply: true", content)
        self.assertIn('globs: "*"', content)
        self.assertIn("## Agent Personas & Execution Contracts", content)
        self.assertIn("### Persona: tpu-agent", content)

    def test_skill_generated(self):
        skill_file = self.dist_dir / "skills" / "tpu-agent" / "SKILL.md"
        self.assertTrue(skill_file.is_file())
        content = skill_file.read_text(encoding="utf-8")
        self.assertIn("name: tpu-agent", content)


if __name__ == "__main__":
    unittest.main()
