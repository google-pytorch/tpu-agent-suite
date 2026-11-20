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
"""Unit tests for `framework.hc.git_ops`."""

from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import tempfile
import unittest
from typing import Sequence
from unittest import mock

from framework.hc import constants, git_ops

Path = pathlib.Path


def _wt(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


class TestGitOps(unittest.TestCase):
    """Tests for bare repo init, worktree hygiene, candidate DAG, and merges."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.home, self.ws, self.code_dir = (self.base / "h", self.base / "ws",
                                             self.base / "src_in")
        for d in (self.home, self.ws, self.code_dir):
            d.mkdir()
        p = mock.patch.dict(os.environ,
                            {"TPU_AGENT_SUITE_HOME": str(self.home)})
        p.start()
        self.addCleanup(p.stop)

    def _propose(self,
                 cid: int,
                 init_sha: str,
                 cand_id: int | None = None) -> dict:
        res = git_ops.propose_candidate_commit(cid,
                                               self.ws,
                                               init_sha,
                                               message="c")
        if cand_id is not None:
            git_ops.finalize_candidate_ref(cid, cand_id, res["sha"],
                                           res["tree_sha"])
        return res

    def _prep(self,
              cid: int,
              parents: Sequence[str],
              ids: Sequence[int] | None = None,
              *,
              discard: bool = False) -> dict:
        return git_ops.prepare_worktree(cid,
                                        self.ws,
                                        parents,
                                        parent_ids=ids,
                                        discard=discard)

    def test_init_hygiene_and_workspace_subpaths(self) -> None:
        _wt(self.code_dir / "model.py", "x = 1\n")
        (self.code_dir / ".git").mkdir()
        _wt(self.code_dir / ".git/HEAD", "ref: refs/heads/main\n")

        info = git_ops.init_campaign_repo(1, self.code_dir, self.ws)
        git_ops.pin_candidate_ref(1, 1, info["sha"])
        src = git_ops.get_worktree_dir(self.ws)
        self.assertFalse((src / ".git").exists())
        self.assertEqual(
            ((src / constants.FLAGS_FILENAME).read_text(encoding="utf-8"),
             info["patch_sha"],
             git_ops.read_worktree_state(1)["prepared_tree_sha"],
             git_ops.read_flags_from_commit(1, info["sha"])),
            ("{}\n", hashlib.sha256(b"").hexdigest(), info["tree_sha"], {}),
        )
        proc = subprocess.run(["git", "rev-parse", "--git-dir"],
                              cwd=src,
                              capture_output=True,
                              check=False)
        self.assertNotEqual(proc.returncode, 0)

        (src / "weights.bin").write_bytes(b"\x00\x01\n=======\n\x00")
        _wt(src / "README.rst", "Install\n=======\nRun pip.\n")
        git_ops.verify_worktree_hygiene(
            src, conflicted_paths=["weights.bin", "README.rst"])

        _wt(src / "model.py",
            "<<<<<<< HEAD\nx = 1\n=======\nx = 2\n>>>>>>> b\n")
        # Unconflicted files are not read_bytes()-scanned, while conflicted_paths are checked.
        git_ops.verify_worktree_hygiene(src)
        with self.assertRaises(constants.ValidationError):
            git_ops.verify_worktree_hygiene(src, conflicted_paths=["model.py"])

        _wt(src / "model.py", "x = 2\n")
        outside = self.base / "secret.txt"
        _wt(outside, "secret")
        (src / "escape_link").symlink_to(outside)
        with self.assertRaises(constants.ValidationError):
            git_ops.verify_worktree_hygiene(src)
        (src / "escape_link").unlink()

        ws_sym_src = self.base / "ws_sym_src"
        ws_sym_src.mkdir()
        (ws_sym_src / "src").symlink_to(self.code_dir)
        for cid, bad_ws in ((20, ws_sym_src), (21, self.code_dir / "nested")):
            with self.assertRaises(constants.ValidationError):
                git_ops.init_campaign_repo(cid, self.code_dir, bad_ws)
        self.assertTrue((self.code_dir / ".git").is_dir())

    def test_candidate_dag_diffs_discard_and_merges(self) -> None:
        ops_t = "def op_top():\n    return {t}\n\n\ndef op_bottom():\n    return {b}\n"
        _wt(self.code_dir / "ops.py", ops_t.format(t=1, b=2))
        for name, body in (("cfg.py", "TILE = 64\n"), ("keep.py", "K = 1\n"),
                           ("old.py", "DEAD = 1\n"), ("mod_del.py",
                                                      "M = 1\n")):
            _wt(self.code_dir / name, body)
        (self.code_dir / "blob.bin").write_bytes(b"\x00\x01")
        c1 = git_ops.init_campaign_repo(2, self.code_dir, self.ws)
        git_ops.finalize_candidate_ref(2, 1, c1["sha"], c1["tree_sha"])
        src, patch_path = git_ops.get_worktree_dir(
            self.ws), self.base / "c.patch"

        # c2 (flags-only): worktree_state stays at c1 until finalize; patch export omits flags.
        _wt(src / constants.FLAGS_FILENAME,
            '{"env": {"XLA_FLAGS": "--xla_a=true"}}\n')
        c2 = self._propose(2, c1["sha"])
        self.assertEqual(
            git_ops.read_worktree_state(2)["staged_parents"], [c1["sha"]])
        git_ops.finalize_candidate_ref(2, 2, c2["sha"], c2["tree_sha"])
        self.assertIsNone(
            git_ops.export_code_patch(2, c1["sha"], c2["sha"], patch_path))
        self.assertNotEqual(git_ops.compute_patch_sha(2, c1["sha"], c2["sha"]),
                            c1["patch_sha"])
        code_d, flags_d = git_ops.compute_candidate_diffs(
            2, c1["sha"], c2["sha"])
        self.assertTrue(code_d == "" and "XLA_FLAGS" in flags_d)

        # c3 (Branch A from c2): code + `.gitignore`-matched file + deletions.
        _wt(src / "ops.py", ops_t.format(t=10, b=2))
        _wt(src / "cfg.py", "TILE = 128\n")
        _wt(src / ".gitignore", "ignored_kernel.py\n")
        _wt(src / "ignored_kernel.py", "IK = 1\n")
        for victim in ("old.py", "mod_del.py", "blob.bin"):
            (src / victim).unlink()
        c3 = self._propose(2, c1["sha"], cand_id=3)
        self.assertEqual(
            git_ops.export_code_patch(2, c1["sha"], c3["sha"], patch_path),
            patch_path)
        ptxt = patch_path.read_text(encoding="utf-8")
        self.assertTrue("ignored_kernel.py" in ptxt
                        and constants.FLAGS_FILENAME not in ptxt)
        self.assertEqual(
            git_ops.compute_candidate_diffs(2, c2["sha"], c3["sha"])[1], "")

        # Dirty check blocks prepare; --discard succeeds even from transient subdir cwd.
        _wt(src / "ignored_kernel.py", "IK = 9\n")
        self.assertTrue(git_ops.worktree_has_uncommitted_changes(2, self.ws))
        transient = src / "transient_dir"
        transient.mkdir()
        _wt(transient / "scratch.txt", "tmp\n")
        with self.assertRaises(constants.InvalidStateError):
            self._prep(2, [c1["sha"]], [1])
        keep_mtime, prev_cwd = (src / "keep.py").stat().st_mtime_ns, Path.cwd()
        os.chdir(transient)
        try:
            prep = self._prep(2, [c1["sha"]], [1], discard=True)
        finally:
            os.chdir(prev_cwd)
        self.assertFalse(prep["has_conflicts"] or transient.exists()
                         or (src / "ignored_kernel.py").exists())
        self.assertEqual((src / "keep.py").stat().st_mtime_ns, keep_mtime)
        for bad_parents in ([], [c1["sha"], c1["sha"]], ["a", "b", "c"]):
            with self.assertRaises(constants.ValidationError):
                self._prep(2, bad_parents)

        # Binary and file/directory conflicts fail cleanly before mutating worktree.
        (src / "blob.bin").write_bytes(b"\x00\x02")
        c_bin = self._propose(2, c1["sha"])
        with self.assertRaisesRegex(constants.GitOperationError, "binary"):
            self._prep(2, [c_bin["sha"], c3["sha"]], discard=True)
        self._prep(2, [c1["sha"]], [1], discard=True)
        (src / "cfg.py").unlink()
        (src / "cfg.py").mkdir()
        _wt(src / "cfg.py/nested.py", "X = 1\n")
        c_dir = self._propose(2, c1["sha"])
        with self.assertRaisesRegex(constants.GitOperationError,
                                    "file/directory"):
            self._prep(2, [c_dir["sha"], c3["sha"]], discard=True)

        # c4 (Branch B from c1) -> 2-parent merge with c3 -> resolve -> c5 + lineage queries.
        self._prep(2, [c1["sha"]], [1], discard=True)
        _wt(src / "ops.py", ops_t.format(t=1, b=20))
        _wt(src / "cfg.py", "TILE = 256\n")
        _wt(src / "mod_del.py", "M = 2\n")
        c4 = self._propose(2, c1["sha"], cand_id=4)

        merge_res = self._prep(2, [c4["sha"], c3["sha"]], [4, 3])
        self.assertTrue(merge_res["has_conflicts"])
        self.assertEqual(
            git_ops.read_worktree_state(2)["conflicted_paths"],
            ["cfg.py", "mod_del.py"])
        self.assertFalse((src / "old.py").exists()
                         or (src / "blob.bin").exists())
        self.assertIn("(deleted)",
                      (src / "mod_del.py").read_text(encoding="utf-8"))
        m_ops = (src / "ops.py").read_text(encoding="utf-8")
        self.assertTrue("return 10" in m_ops and "return 20" in m_ops
                        and "<<<<<<<" not in m_ops)
        self.assertIn("<<<<<<< ", (src / "cfg.py").read_text(encoding="utf-8"))
        with self.assertRaises(constants.ValidationError):
            self._propose(2, c1["sha"])

        _wt(src / "cfg.py", "TILE = 512\n")
        (src / "mod_del.py").unlink()
        c5 = self._propose(2, c1["sha"], cand_id=5)
        self.assertEqual(c5["staged_parent_ids"], [4, 3])
        self.assertEqual(
            git_ops.read_worktree_state(2)["conflicted_paths"], [])
        self.assertEqual(
            [
                git_ops.is_ancestor(2, a, c5["sha"])
                for a in (c1["sha"], c3["sha"], c4["sha"])
            ] + [git_ops.is_ancestor(2, c3["sha"], c4["sha"])],
            [True, True, True, False],
        )
        self.assertEqual(
            (git_ops.candidates_on_path(2, c1["sha"], c5["sha"]),
             git_ops.candidates_on_path(2, c4["sha"], c5["sha"]),
             git_ops.candidates_on_path(2, c5["sha"], c5["sha"])),
            ([2, 3, 4, 5], [2, 3, 5], []),
        )


if __name__ == "__main__":
    unittest.main()
