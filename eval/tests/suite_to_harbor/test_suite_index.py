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
"""Tests for eval/suite_to_harbor/suite_index.py on synthetic suite trees."""

from __future__ import annotations

import os
import pathlib
import tempfile
import unittest

from eval.suite_to_harbor import suite_index

Path = pathlib.Path


def _suite(*cases: str, **names: str) -> str:
    head = "".join(f"{k}: {v}\n" for k, v in names.items())
    body = "".join(f"\n    - name: {c}\n      prompt: Do {c}." for c in cases)
    return f"{head}evaluation:\n  cases:{body or ' []'}\n"


class SuiteIndexTest(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name).resolve()
        self.suites = self.repo / "eval" / "suites"
        self.datasets = self.repo / "eval" / ".datasets"
        self.suites.mkdir(parents=True)

    def _write(self, rel: str, text: str, root: Path | None = None) -> Path:
        path = (root or self.suites) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def _resolve(self, spec: str, *, allow_cwd: bool = True) -> Path:
        return suite_index.resolve_suite(spec,
                                         self.suites,
                                         self.repo,
                                         allow_cwd=allow_cwd)

    def test_suite_names_lists_paths_unique_declared_and_task_dirs(
            self) -> None:
        self.assertEqual(suite_index.suite_names(self.repo / "missing"), set())
        self._write("g/f.yaml", _suite("a", "b", name="sec", suite_name="pri"))
        self._write("dup/a.yaml", _suite("a", "b", name="shared"))
        self._write("dup/b.yaml", _suite("c", "d", name="shared"))
        self._write("qa/one.yaml", _suite("a", name="task-dir"))
        self._write("qa/two.yaml", _suite("b", name="task-dir"))
        self._write("assets/settings.yaml", "key: value\n")
        self._write("assets/broken.yaml", "evaluation: [unclosed\n")
        self.assertEqual(
            suite_index.suite_names(self.suites),
            {"g/f", "pri", "dup/a", "dup/b", "qa"},
        )

    def test_resolve_suite_paths_declared_names_and_precedence(self) -> None:
        suite = self._write("s/a.yaml",
                            _suite("a", "b", name="sec", suite_name="pri"))
        yml_suite = self._write("s/alt.yml", _suite("a", "b"))
        dotted = self._write("s/suite.v1.yaml", _suite("a"))
        by_path = self._write("second.yaml", _suite("c", "d", name="x"))
        self._write("other/f.yaml", _suite("e", "f", name="second"))

        for spec, expected in (
            (str(suite), suite),
            (str(suite.with_suffix("")), suite),
            (str(suite.parent), suite.parent),
            ("s/a.yaml", suite),
            ("s/a", suite),
            ("s/alt", yml_suite),
            ("s/suite.v1", dotted),
            ("s", suite.parent),
            ("pri", suite),
            ("second", by_path),
        ):
            self.assertEqual(self._resolve(spec), expected)

    def test_resolve_suite_repo_root_and_allow_cwd_isolation(self) -> None:
        ext = self._write("extra/suite.yaml", _suite("a"), root=self.repo)
        suites_local = self._write("local.yaml", _suite("a"))
        cwd_dir = self.repo / "work"
        cwd_suite = self._write("local.yaml", _suite("b"), root=cwd_dir)
        self._write("only_cwd.yaml", _suite("c"), root=cwd_dir)
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(cwd_dir)

        self.assertEqual(self._resolve("extra/suite"), ext)
        self.assertEqual(self._resolve("./local.yaml"), cwd_suite)
        self.assertEqual(self._resolve("./local.yaml", allow_cwd=False),
                         suites_local)
        with self.assertRaises(suite_index.SuiteLookupError):
            self._resolve("only_cwd.yaml", allow_cwd=False)

    def test_resolve_suite_rejects_non_yaml_empty_missing_and_ambiguous(
            self) -> None:
        readme = self._write("README.md", "# docs\n", root=self.repo)
        (self.suites / "empty_dir").mkdir()
        self._write("nested/leaf/task.yaml", _suite("a"))
        self._write("a/one.yaml", _suite("a", "b", suite_name="dup"))
        self._write("b/two.yaml", _suite("c", "d", suite_name="dup"))

        for bad in (str(readme), "README.md", "empty_dir", "nested", "   "):
            with self.assertRaises(suite_index.SuiteLookupError):
                self._resolve(bad)

        with self.assertRaises(suite_index.SuiteLookupError) as missing:
            self._resolve("no-such")
        self.assertIn("Suite 'no-such' not found.\nAvailable suites:",
                      str(missing.exception))

        with self.assertRaises(suite_index.SuiteLookupError) as ambig:
            self._resolve("dup")
        self.assertEqual(
            str(ambig.exception).splitlines()[1:3],
            ["  eval/suites/a/one.yaml", "  eval/suites/b/two.yaml"],
        )

    def test_resolve_suite_dot_does_not_match_sibling_yaml(self) -> None:
        self._write("empty_dir.yaml", _suite("a"))
        (self.suites / "empty_dir").mkdir()
        with self.assertRaises(suite_index.SuiteLookupError):
            self._resolve("empty_dir/.")

    def test_suite_sources_and_dataset_dir(self) -> None:
        yaml_a = self._write("g/a.yaml", _suite("a"))
        yml_b = self._write("g/b.yml", _suite("b"))
        readme = self._write("g/README.md", "# notes\n")
        self.assertEqual(suite_index.suite_sources(yaml_a), [yaml_a])
        self.assertEqual(suite_index.suite_sources(yaml_a.parent),
                         [yaml_a, yml_b])
        with self.assertRaises(ValueError):
            suite_index.suite_sources(readme)

        self.assertEqual(
            suite_index.suite_dataset_dir(yaml_a, self.suites, self.datasets),
            self.datasets / "g" / "a",
        )
        self.assertEqual(
            suite_index.suite_dataset_dir(yaml_a.parent, self.suites,
                                          self.datasets),
            self.datasets / "g",
        )
        with self.assertRaises(ValueError):
            suite_index.suite_dataset_dir(self.suites, self.suites,
                                          self.datasets)

        ext = self.repo / "ExternalDir"
        named = self._write("f.yaml",
                            "name: Sec\nsuite_name: Sample-AbC\n",
                            root=ext)
        unnamed = self._write("Stem_X.yaml",
                              "evaluation:\n  cases: []\n",
                              root=ext)
        self.assertEqual(
            suite_index.suite_dataset_dir(named, self.suites, self.datasets),
            self.datasets / "Sample-AbC",
        )
        self.assertEqual(
            suite_index.suite_dataset_dir(unnamed, self.suites, self.datasets),
            self.datasets / "Stem_X",
        )
        self.assertEqual(
            suite_index.suite_dataset_dir(ext, self.suites, self.datasets),
            self.datasets / "ExternalDir",
        )
        for bad in ("../escape", "a/b", r"a\b", ".."):
            bad_file = self._write("bad.yaml",
                                   f"suite_name: '{bad}'\n",
                                   root=ext)
            with self.assertRaises(ValueError):
                suite_index.suite_dataset_dir(bad_file, self.suites,
                                              self.datasets)


if __name__ == "__main__":
    unittest.main()
