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
"""Tests for plugin_source.py, against a stand-in vllm_torchtpu on PYTHONPATH.

The stand-in is the image's plugin: a git checkout whose last commit no remote
has, as an earlier agent would leave it.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "plugin_source.py")
APPLIER = os.path.join(HERE, "vllm_patches", "apply_vllm_patch.sh")
GIT = ("git", "-c", "user.name=t", "-c", "user.email=t@t", "-c",
       "commit.gpgsign=false")
IMAGE = "registry.example/vllm@sha256:abc"


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _git(*args):
    return subprocess.run([*GIT, *args],
                          check=True,
                          capture_output=True,
                          text=True).stdout.strip()


class PluginSourceTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.image = os.path.join(self.dir, "image")
        self.probe_dir = os.path.join(self.dir, "probe")
        self.repo = os.path.join(self.dir, "repo")
        os.makedirs(self.probe_dir)

    def _plugin(self, prefix="", git=True):
        """The image's plugin; returns the directory to put on PYTHONPATH."""
        root = os.path.join(self.image, prefix)
        pkg = os.path.join(root, "vllm_torchtpu")
        # Importing it would fail: the plugin must be located without running.
        _write(os.path.join(pkg, "__init__.py"),
               'raise RuntimeError("TPU init")\n')
        _write(os.path.join(pkg, "ops", "kernel.py"), "BLOCK = 128\n")
        if git:
            _git("init", "-q", self.image)
            _git("-C", self.image, "add", "-A")
            _git("-C", self.image, "commit", "-qm", "upstream")
            _write(os.path.join(pkg, "ops", "kernel.py"), "BLOCK = 96\n")
            _git("-C", self.image, "commit", "-qam", "earlier agent")
        return root

    def _run(self, *args, pythonpath, **env_extra):
        env = dict(os.environ, PYTHONPATH=pythonpath, **env_extra)
        return subprocess.run([sys.executable, SCRIPT, *args],
                              cwd=self.probe_dir,
                              env=env,
                              capture_output=True,
                              text=True)

    def _probe(self, root):
        return self._run("probe", "--output", "plugin.bundle", pythonpath=root)

    def _base(self, root):
        r = self._run("base",
                      self.repo,
                      self.probe_dir,
                      "--image",
                      IMAGE,
                      pythonpath=root)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def _commit(self, path, text, message):
        _write(os.path.join(self.repo, path), text)
        _git("-C", self.repo, "add", path)
        _git("-C", self.repo, "commit", "-qm", message)
        return _git("-C", self.repo, "rev-parse", "HEAD")

    def _diff(self, ref, root, *args):
        patch = os.path.join(self.dir, "arm.patch")
        r = self._run("diff", self.repo, ref, patch, *args, pythonpath=root)
        return r, patch

    def _apply(self, patch, root):
        return subprocess.run(["bash", APPLIER, patch, root],
                              capture_output=True,
                              text=True)

    def test_unpushed_commits_become_the_base_and_a_branch_applies(self):
        root = self._plugin()
        _write(os.path.join(self.image, "notes.txt"), "scratch\n")
        sha = _git("-C", self.image, "rev-parse", "HEAD")
        r = self._probe(root)
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = json.loads(r.stdout)
        self.assertEqual(meta["sha"], sha)
        self.assertEqual(meta["untracked"], ["notes.txt"])

        record = self._base(root)
        self.assertEqual(record["base_sha"], sha)
        # `compute image commit --path` reads the in-image checkout from here.
        self.assertEqual(record["path"], meta["path"])
        self.assertEqual(_git("-C", self.repo, "rev-parse", "base"), sha)
        _git("-C", self.repo, "switch", "-qc", "hyp/block", "base")
        arm = self._commit("vllm_torchtpu/ops/kernel.py", "BLOCK = 256\n",
                           "block 256")
        r, patch = self._diff("hyp/block", root)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)["sha"], arm)
        r = self._apply(patch, root)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(
            _read(os.path.join(root, "vllm_torchtpu", "ops", "kernel.py")),
            "BLOCK = 256\n")

    def test_a_plugin_below_the_repo_root_gets_a_relative_patch(self):
        root = self._plugin(prefix="python")
        self.assertEqual(self._probe(root).returncode, 0)
        self.assertEqual(self._base(root)["prefix"], "python")
        self._commit("python/vllm_torchtpu/ops/kernel.py", "BLOCK = 512\n",
                     "block 512")
        r, patch = self._diff("HEAD", root)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._apply(patch, root).returncode, 0)
        self.assertEqual(
            _read(os.path.join(root, "vllm_torchtpu", "ops", "kernel.py")),
            "BLOCK = 512\n")

    def test_no_git_or_modified_tracked_files_exit_7(self):
        root = self._plugin(git=False)
        self.assertEqual(self._probe(root).returncode, 7)
        shutil.rmtree(self.image)
        root = self._plugin()
        _write(os.path.join(root, "vllm_torchtpu", "ops", "kernel.py"),
               "BLOCK = 64\n")
        self.assertEqual(self._probe(root).returncode, 7)
        self.assertFalse(
            os.path.exists(os.path.join(self.probe_dir, "plugin.bundle")))

    def test_a_local_copy_without_git_is_its_own_repo(self):
        root = self._plugin(prefix="src", git=False)
        # An ignored build copy, nearer the top than the plugin itself.
        _write(os.path.join(self.image, ".gitignore"), "b/\n")
        _write(os.path.join(self.image, "b", "vllm_torchtpu", "__init__.py"),
               "\n")
        shas = []
        for name, date in (("a", "2026-01-01T00:00:00Z"),
                           ("b", "2026-06-01T00:00:00Z")):
            copy = os.path.join(self.dir, name)
            shutil.copytree(self.image, copy)
            r = self._run("local",
                          copy,
                          "--image",
                          IMAGE,
                          pythonpath="",
                          GIT_AUTHOR_DATE=date,
                          GIT_COMMITTER_DATE=date)
            self.assertEqual(r.returncode, 0, r.stderr)
            record = json.loads(r.stdout)
            self.assertEqual(record["prefix"], "src")
            shas.append(record["base_sha"])
        # A redone init must not look like another image's base (exit 4).
        self.assertEqual(shas[0], shas[1])

        self.repo = copy
        _git("-C", copy, "switch", "-qc", "hyp/block", "base")
        self._commit("src/vllm_torchtpu/ops/kernel.py", "BLOCK = 256\n",
                     "block 256")
        r, patch = self._diff("hyp/block", root)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._apply(patch, root).returncode, 0)
        self.assertEqual(
            _read(os.path.join(root, "vllm_torchtpu", "ops", "kernel.py")),
            "BLOCK = 256\n")

    def test_not_importable_exits_3(self):
        self.assertEqual(
            self._probe(os.path.join(self.dir, "empty")).returncode, 3)

    def test_a_ref_off_base_or_outside_the_plugin_is_refused(self):
        root = self._plugin()
        self._probe(root)
        self._base(root)
        _git("-C", self.repo, "switch", "-q", "--orphan", "other")
        self._commit("vllm_torchtpu/ops/kernel.py", "BLOCK = 64\n", "orphan")
        r, patch = self._diff("other", root)
        self.assertEqual(r.returncode, 6, r.stderr)

        _git("-C", self.repo, "switch", "-qc", "hyp/setup", "base")
        self._commit("setup.py", "\n", "outside")
        r, patch = self._diff("hyp/setup", root)
        self.assertEqual(r.returncode, 6, r.stderr)
        self.assertFalse(os.path.exists(patch))

    def test_a_repo_keeps_one_base(self):
        root = self._plugin()
        self._probe(root)
        self._base(root)
        _write(os.path.join(root, "vllm_torchtpu", "ops", "kernel.py"),
               "BLOCK = 32\n")
        _git("-C", self.image, "commit", "-qam", "another image")
        self.assertEqual(self._probe(root).returncode, 0)
        r = self._run("base",
                      self.repo,
                      self.probe_dir,
                      "--image",
                      IMAGE,
                      pythonpath=root)
        self.assertEqual(r.returncode, 2, r.stderr)


if __name__ == "__main__":
    unittest.main()
