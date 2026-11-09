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
"""Unit tests for scripts/check_denylist.py."""

from __future__ import annotations

import base64
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_denylist.py"

_SPEC = importlib.util.spec_from_file_location("check_denylist", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
check_denylist = importlib.util.module_from_spec(_SPEC)
sys.modules["check_denylist"] = check_denylist
_SPEC.loader.exec_module(check_denylist)


class TestDenylistScanner(unittest.TestCase):
    """Tests diff parsing, secret config loading, and rule self-tests."""

    def test_parse_unified_diff_and_exclusions(self) -> None:
        bad = "forbidden_token_xyz"
        raw_yaml = yaml.safe_dump({
            "exclude_paths": ["google/**"],
            "rules": [
                {
                    "id": "synthetic-leak",
                    "pattern": bad,
                    "description": "No synthetic leaks.",
                },
                {
                    "id": "case-sensitive-rule",
                    "pattern": r"Exact\s+Title",
                    "case_sensitive": True,
                    "description": "Case-sensitive check.",
                },
            ],
        })
        b64 = base64.b64encode(raw_yaml.encode("utf-8")).decode("ascii")
        excludes, rules = check_denylist.load_config(
            Path("/nonexistent/denylist.yaml"), env_b64=b64)

        diff_text = "\n".join([
            "diff --git a/README.md b/README.md",
            "index 1111111..2222222 100644",
            "--- a/README.md",
            "+++ b/README.md",
            "@@ -10,2 +10,4 @@",
            f"-removed line with {bad} is ignored",
            "+clean line 10 with exact title in lowercase",
            f"+++ added line 11 starting with ++ and {bad}",
            "+added line 12 with Exact Title match",
            "diff --git a/google/denylist.yaml b/google/denylist.yaml",
            "+++ b/google/denylist.yaml",
            "@@ -0,0 +1,1 @@",
            f"+excluded file with {bad}",
        ])
        findings = check_denylist.parse_unified_diff(diff_text, excludes,
                                                     rules)
        self.assertEqual(len(findings), 2)
        self.assertEqual((findings[0].file, findings[0].line),
                         ("README.md", 11))
        self.assertEqual((findings[1].file, findings[1].line),
                         ("README.md", 12))
        self.assertIn(bad, findings[0].format(github=True, redact=False))
        self.assertNotIn(bad, findings[0].format(github=True, redact=True))

    def test_configured_rules_self_test(self) -> None:
        try:
            _, rules = check_denylist.load_config(
                check_denylist.DEFAULT_CONFIG_PATH)
        except FileNotFoundError:
            self.skipTest(
                "Neither google/denylist.yaml nor INTERNAL_DENYLIST_B64 is available."
            )
        self.assertGreaterEqual(len(rules), 8)
        for rule in rules:
            with self.subTest(rule_id=rule.id):
                self.assertTrue(rule.must_match)
                self.assertTrue(rule.must_not_match)
                self.assertIsNotNone(rule.regex.search(rule.must_match))
                self.assertIsNone(rule.regex.search(rule.must_not_match))

    def test_missing_config_fails_closed(self) -> None:
        with mock.patch.dict(
                os.environ,
            {
                "INTERNAL_DENYLIST_B64": "",
                "GITHUB_ACTIONS": ""
            },
                clear=False,
        ):
            rc = check_denylist.main(
                ["--config", "/nonexistent/denylist.yaml", "--staged"])
        self.assertEqual(rc, 1)

    def test_scan_paths_directory_and_binary_skip(self) -> None:
        bad = "forbidden_token_xyz"
        raw_yaml = yaml.safe_dump({
            "exclude_paths": ["dist/excluded/**"],
            "rules": [{
                "id": "synthetic-leak",
                "pattern": bad,
                "description": "No synthetic leaks.",
            }],
        })
        b64 = base64.b64encode(raw_yaml.encode("utf-8")).decode("ascii")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist = root / "dist"
            (dist / "claude-code").mkdir(parents=True)
            (dist / "excluded").mkdir(parents=True)
            (dist / "claude-code" / "clean.md").write_text("clean content\n",
                                                           encoding="utf-8")
            (dist / "claude-code" /
             "archive.tar.gz").write_bytes(b"\x00\x1f\x8b" +
                                           bad.encode("utf-8"))
            (dist / "excluded" / "skip.md").write_text(f"skip {bad}\n",
                                                       encoding="utf-8")
            with mock.patch.dict(os.environ, {"INTERNAL_DENYLIST_B64": b64},
                                 clear=False):
                rc_clean = check_denylist.main(
                    ["--root", str(root), "--paths", "dist"])
                self.assertEqual(rc_clean, 0)
                (dist / "claude-code" / "leak.md").write_text(
                    f"line1\n{bad}\n", encoding="utf-8")
                rc_leak = check_denylist.main(
                    ["--root", str(root), "--paths", "dist"])
                self.assertEqual(rc_leak, 1)

    def test_scan_paths_unreadable_file_raises_oserror(self) -> None:
        raw_yaml = yaml.safe_dump({
            "exclude_paths": [],
            "rules": [{
                "id": "synthetic-leak",
                "pattern": "forbidden_token_xyz",
                "description": "No synthetic leaks.",
            }],
        })
        b64 = base64.b64encode(raw_yaml.encode("utf-8")).decode("ascii")
        excludes, rules = check_denylist.load_config(
            Path("/nonexistent/denylist.yaml"), env_b64=b64)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "file.txt"
            target.write_text("clean\n", encoding="utf-8")
            with mock.patch.object(Path,
                                   "read_bytes",
                                   side_effect=OSError("permission denied")):
                with self.assertRaises(OSError):
                    check_denylist.scan_paths(root, [target], excludes, rules)


if __name__ == "__main__":
    unittest.main()
