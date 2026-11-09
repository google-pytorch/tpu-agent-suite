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
"""Unit tests for build-time package overlays (`framework/builder/overlays.py`)."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from framework import compiler, utils
from framework.builder import overlays

Path = pathlib.Path


class OverlaysTest(unittest.TestCase):

    def test_no_overlay_is_zero_copy_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "package"
            pkg.mkdir()
            (pkg / "hello.txt").write_text("hi\n", encoding="utf-8")
            with overlays.stage_package_with_overlays(pkg) as staged:
                self.assertEqual(staged, pkg)

    def test_disabled_flag_skips_existing_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            pkg = root / "package"
            pkg.mkdir()
            (pkg / "doc.md").write_text("base\n", encoding="utf-8")

            ov = root / "google" / "overlays" / "package"
            ov.mkdir(parents=True)
            (ov / "doc.md.append.md").write_text("extra\n", encoding="utf-8")

            with overlays.stage_package_with_overlays(pkg,
                                                      enabled=False) as staged:
                self.assertEqual(staged, pkg)
                self.assertEqual(
                    (staged / "doc.md").read_text(encoding="utf-8"), "base\n")

    def test_patch_edits_append_and_plain_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            pkg = root / "package"
            (pkg / "environment").mkdir(parents=True)
            (pkg / "agents").mkdir(parents=True)

            base_questions = {
                "questions": [
                    {
                        "name": "backend",
                        "choices": ["local", "gke_job"],
                    },
                    {
                        "name": "kube_context",
                        "when": {
                            "backend": ["gke_job"]
                        },
                    },
                ]
            }
            (pkg / "environment" / "questions.json").write_text(
                json.dumps(base_questions), encoding="utf-8")
            (pkg / "agents" / "worker.md").write_text(
                "Header\n\nAllowed: a, b.\n", encoding="utf-8")

            ov = root / "google" / "overlays" / "package"
            (ov / "environment").mkdir(parents=True)
            (ov / "agents").mkdir(parents=True)
            (ov / "extra").mkdir(parents=True)

            patch_questions = {
                "questions": [
                    {
                        "name": "backend",
                        "choices": ["custom_cluster"],
                    },
                    {
                        "name": "kube_context",
                        "when": {
                            "backend": ["custom_cluster"]
                        },
                    },
                    {
                        "name": "custom_recipe",
                        "path": "compute.params.custom_recipe",
                        "when": {
                            "backend": ["custom_cluster"]
                        },
                    },
                ]
            }
            (ov / "environment" / "questions.json.patch.json").write_text(
                json.dumps(patch_questions), encoding="utf-8")
            (ov / "agents" / "worker.md.edits.json").write_text(
                json.dumps({
                    "edits": [{
                        "target": "Allowed: a, b.",
                        "replacement": "Allowed: a, b, c.",
                    }]
                }),
                encoding="utf-8",
            )
            (ov / "agents" / "worker.md.append.md").write_text(
                "\n## Extra Section\nInternal note.\n", encoding="utf-8")
            (ov / "extra" / "note.txt").write_text("copied\n",
                                                   encoding="utf-8")

            with overlays.stage_package_with_overlays(pkg) as staged:
                self.assertNotEqual(staged, pkg)
                merged_q = json.loads(
                    (staged / "environment" /
                     "questions.json").read_text(encoding="utf-8"))
                by_name = {q["name"]: q for q in merged_q["questions"]}
                self.assertEqual(by_name["backend"]["choices"],
                                 ["local", "gke_job", "custom_cluster"])
                self.assertEqual(by_name["kube_context"]["when"]["backend"],
                                 ["gke_job", "custom_cluster"])
                self.assertIn("custom_recipe", by_name)

                worker_text = (staged / "agents" /
                               "worker.md").read_text(encoding="utf-8")
                self.assertIn("Allowed: a, b, c.", worker_text)
                self.assertIn("## Extra Section", worker_text)
                self.assertEqual((staged / "extra" /
                                  "note.txt").read_text(encoding="utf-8"),
                                 "copied\n")

            # Source package directory remains untouched.
            orig_q = json.loads((pkg / "environment" /
                                 "questions.json").read_text(encoding="utf-8"))
            self.assertEqual(orig_q["questions"][0]["choices"],
                             ["local", "gke_job"])

    def test_missing_edit_target_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "package"
            pkg.mkdir()
            (pkg / "doc.md").write_text("hello world\n", encoding="utf-8")

            ov = root / "overlay"
            ov.mkdir()
            (ov / "doc.md.edits.json").write_text(
                json.dumps([{
                    "target": "does-not-exist",
                    "replacement": "boom"
                }]),
                encoding="utf-8",
            )

            with self.assertRaises(utils.TpuAgentSuiteError):
                with overlays.stage_package_with_overlays(pkg, overlay_dir=ov):
                    pass

    def test_symlink_inside_overlay_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "secret.txt"
            outside.write_text("secret\n", encoding="utf-8")

            pkg = root / "package"
            pkg.mkdir()
            (pkg / "doc.md").write_text("ok\n", encoding="utf-8")

            ov = root / "overlay"
            ov.mkdir()
            (ov / "leak.txt").symlink_to(outside)

            with self.assertRaises(utils.TpuAgentSuiteError):
                with overlays.stage_package_with_overlays(pkg, overlay_dir=ov):
                    pass

    def test_compiler_parser_accepts_no_overlays(self) -> None:
        parser = compiler.build_compiler_parser()
        args = parser.parse_args(["verify", "package", "--no-overlays"])
        self.assertTrue(args.no_overlays)


if __name__ == "__main__":
    unittest.main()
