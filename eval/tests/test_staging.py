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
"""Tests for staging.py (eval/staging.py)."""

import unittest
from unittest.mock import patch

from eval.staging import (compute_staging_digest, find_repo_root,
                          stage_composite_package)
from eval.utils import package_identity


class TestStaging(unittest.TestCase):
    """Test suite for composite staging digest calculation and staging flows."""

    def setUp(self) -> None:
        self.repo_root = find_repo_root()

    def test_digest_deterministic_and_unique(self) -> None:
        """Digest calculation is deterministic and sensitive to harness and answers."""
        d1 = compute_staging_digest(
            self.repo_root,
            harness="claude-code",
            compute_answers={"backend": "local"},
        )
        d2 = compute_staging_digest(
            self.repo_root,
            harness="claude-code",
            compute_answers={"backend": "local"},
        )
        # Deterministic
        self.assertEqual(d1, d2)
        self.assertEqual(len(d1), 16)

        # A subset must not reuse the whole package's cache.
        d_subset = compute_staging_digest(
            self.repo_root,
            harness="claude-code",
            compute_answers={"backend": "local"},
            skills=["tpu-agent"],
        )
        self.assertNotEqual(d1, d_subset)

        # Sensitive to harness
        d_diff_harness = compute_staging_digest(
            self.repo_root,
            harness="antigravity",
            compute_answers={"backend": "local"},
        )
        self.assertNotEqual(d1, d_diff_harness)

        # Sensitive to compute answers
        d_diff_answers = compute_staging_digest(
            self.repo_root,
            harness="claude-code",
            compute_answers={
                "backend": "gke_pod",
                "pod": "test-pod"
            },
        )
        self.assertNotEqual(d1, d_diff_answers)

        # The experimental package must not reuse the release package's cache.
        d_experimental = compute_staging_digest(
            self.repo_root,
            harness="claude-code",
            compute_answers={"backend": "local"},
            manifest="package.experimental.json",
        )
        self.assertNotEqual(d1, d_experimental)

    def test_a_control_arm_stages_nothing(self) -> None:
        """A control arm returns is_empty=True without compiling anything."""
        with patch("subprocess.run") as run:
            res = stage_composite_package("claude-code", control_arm=True)
        self.assertTrue(res.is_empty)
        run.assert_not_called()

    def test_a_subset_is_compiled_alone(self) -> None:
        """The subset mode passes its skills to the compiler."""
        with patch("subprocess.run") as run:
            stage_composite_package("claude-code",
                                    skills=["tpu-agent"],
                                    repo_root=self.repo_root)
        [translate] = [
            c.args[0] for c in run.call_args_list if "translate" in c.args[0]
        ]
        self.assertEqual("tpu-agent",
                         translate[translate.index("--skills") + 1])

    def test_the_whole_package_is_compiled(self) -> None:
        """No --skills: every skill package.release.json releases is built."""
        with patch("subprocess.run") as run:
            stage_composite_package("claude-code", repo_root=self.repo_root)
        [translate] = [
            c.args[0] for c in run.call_args_list if "translate" in c.args[0]
        ]
        self.assertNotIn("--skills", translate)
        self.assertEqual("package.release.json",
                         translate[translate.index("--manifest") + 1])

    def test_the_experimental_manifest_is_compiled(self) -> None:
        """An experimental config's manifest reaches the compiler."""
        with patch("subprocess.run") as run:
            res = stage_composite_package("claude-code",
                                          repo_root=self.repo_root,
                                          manifest="package.experimental.json")
        [translate] = [
            c.args[0] for c in run.call_args_list if "translate" in c.args[0]
        ]
        self.assertEqual("package.experimental.json",
                         translate[translate.index("--manifest") + 1])
        self.assertIn("packages/tpu-agent-suite-experimental@",
                      str(res.central_pkg_dir))

    def test_central_pkg_dir_scoped_to_staged_root(self) -> None:
        """Verify central package directory is scoped inside staged_root to prevent cross-run pollution."""
        with patch("subprocess.run"):
            res = stage_composite_package(
                harness="claude-code",
                repo_root=self.repo_root,
            )
            self.assertTrue(
                str(res.central_pkg_dir).startswith(str(res.staged_root)))
            # Named as the installer names it, from the manifest's fields.
            name, version = package_identity(repo_root=self.repo_root)
            self.assertEqual(f"{name}@{version}", res.central_pkg_dir.name)

    def test_staging_skips_duplicate_bundled_repo_drivers(self) -> None:
        """Drivers already bundled into framework/compute/drivers/ are not duplicated into staged_root/drivers/."""
        import shutil
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_ext = root / "already_bundled.py"
            fake_ext.write_text("VALUE = 1\n", encoding="utf-8")
            with patch(
                    "framework.compute.drivers.external.find_sources",
                    return_value=([fake_ext], []),
            ):
                res = stage_composite_package(
                    harness="claude-code",
                    compute_answers={"backend": "local"},
                    skills=["tpu-agent"],
                    repo_root=self.repo_root,
                )
                try:
                    # `fake_ext` is not bundled into framework/compute/drivers, so it IS staged.
                    self.assertTrue((res.staged_root / "drivers" /
                                     "already_bundled.py").is_file())
                    # Any repo driver under google/drivers/ bundled into central_pkg_dir must NOT be duplicated.
                    bundled_dir = (res.central_pkg_dir / "framework" /
                                   "compute" / "drivers")
                    repo_drivers = self.repo_root / "google" / "drivers"
                    if repo_drivers.is_dir():
                        for drv in repo_drivers.glob("*.py"):
                            if not drv.name.startswith("_"):
                                self.assertTrue(
                                    (bundled_dir / drv.name).is_file())
                                self.assertFalse((res.staged_root / "drivers" /
                                                  drv.name).exists())
                finally:
                    shutil.rmtree(res.staged_root, ignore_errors=True)

    def test_digest_tracks_package_directory_driver_contents(self) -> None:
        """compute_staging_digest changes when files inside a package-directory driver change."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            pkg_drv = Path(tmp) / "custom_pkg_driver"
            pkg_drv.mkdir()
            init_py = pkg_drv / "__init__.py"
            helper_py = pkg_drv / "helper.py"
            init_py.write_text("from .helper import V\n", encoding="utf-8")
            helper_py.write_text("V = 1\n", encoding="utf-8")
            with patch(
                    "framework.compute.drivers.external.find_sources",
                    return_value=([pkg_drv], []),
            ):
                d_before = compute_staging_digest(
                    self.repo_root,
                    harness="claude-code",
                    compute_answers={"backend": "local"},
                )
                helper_py.write_text("V = 2\n", encoding="utf-8")
                d_after = compute_staging_digest(
                    self.repo_root,
                    harness="claude-code",
                    compute_answers={"backend": "local"},
                )
            self.assertNotEqual(d_before, d_after)


if __name__ == "__main__":
    unittest.main()
