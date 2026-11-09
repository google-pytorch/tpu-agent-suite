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
"""CPU tests for core/forward_run.py and core/cachelocal.py: node prep, compile cache, twin inputs and KV fill.

Run from scripts/workload-microbenchmark/: `python3 -m unittest discover tests`.
"""

import json
import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)


class NodePrep(unittest.TestCase):

    def test_node_prep_prints_only_exports_of_the_run_env(self):
        import subprocess
        import tempfile
        plan = {
            "twin": {
                "env": {
                    "LIBTPU_INIT_ARGS": "--a=1 --b='x y'",
                    "FOO": "bar"
                }
            }
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as fh:
            json.dump(plan, fh)
        out = subprocess.run([
            sys.executable,
            os.path.join(SCRIPTS, "core", "forward_run.py"), "--steps",
            fh.name, "--node-prep"
        ],
                             capture_output=True,
                             text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = out.stdout.strip().splitlines()
        self.assertTrue(all(line.startswith("export ") for line in lines))
        got = subprocess.run([
            "bash", "-c",
            out.stdout + '\nprintf "%s|%s|%s" "$LIBTPU_INIT_ARGS" '
            '"$WMB_NODE_PREPPED" "$VLLM_CACHE_ROOT"'
        ],
                             capture_output=True,
                             text=True)
        self.assertEqual(got.stdout, "--a=1 --b='x y'|1|/tmp/wmb_cache")


class ForwardCompat(unittest.TestCase):

    def test_unknown_flag_becomes_env(self):
        import argparse

        from core import forward_run
        p = argparse.ArgumentParser()
        p.add_argument("--tensor-parallel-size")
        os.environ.pop("VLLM_DCP_Q_REPLICATE", None)
        argv, notes = forward_run._compat(
            ["--tensor-parallel-size", "8", "--no-dcp-q-replicate"], p)
        self.assertEqual(argv, ["--tensor-parallel-size", "8"])
        self.assertEqual(os.environ.pop("VLLM_DCP_Q_REPLICATE"), "0")
        self.assertEqual(len(notes), 1)

    def test_known_flag_is_kept(self):
        import argparse

        from core import forward_run
        p = argparse.ArgumentParser()
        p.add_argument("--no-dcp-q-replicate", action="store_true")
        self.assertEqual(forward_run._compat(["--no-dcp-q-replicate"], p),
                         (["--no-dcp-q-replicate"], []))

    def test_cache_key_includes_env(self):
        from core import cachelocal
        self.assertNotEqual(
            cachelocal.key("img", ["--a"], {"LIBTPU_INIT_ARGS": "--x"}),
            cachelocal.key("img", ["--a"], {"LIBTPU_INIT_ARGS": "--y"}))


class CachePrune(unittest.TestCase):

    def test_cold_save_keeps_tier3_drops_own_dir_restored_save_the_reverse(
            self):
        import tempfile

        from core import forward_run as fr
        d = tempfile.mkdtemp()
        tcc = os.path.join(d, "torch_compile_cache")
        os.makedirs(os.path.join(tcc, "torch_aot_compile", "a"))
        os.makedirs(os.path.join(d, "tier3", "k"))
        open(os.path.join(d, "tier3", "k", "small.bin"),
             "wb").write(b"0" * 100)
        open(os.path.join(d, "tier3", "k", "big.bin"),
             "wb").write(b"0" * (fr.TIER3_KEEP_BYTES + 1))
        os.makedirs(os.path.join(tcc, "cold1"))  # made by the cold boot
        r1 = fr._prune_for_save(d, set(), restored=False)
        # the next (restored) boot can't load cold1 but can reuse the big Tier-3 executable
        self.assertEqual(
            (r1["compile_dirs_dropped"], r1["tier3_large_dropped"]),
            (["cold1"], 0))
        self.assertTrue(
            os.path.isfile(os.path.join(d, "tier3", "k", "big.bin")))
        before = fr._compile_dirs(d)
        os.makedirs(os.path.join(tcc, "warm1"))  # restored boot's own dir
        r2 = fr._prune_for_save(d, before, restored=True)
        self.assertEqual(r2["tier3_large_dropped"],
                         1)  # warm1 holds those graphs now
        self.assertEqual(sorted(os.listdir(tcc)),
                         ["torch_aot_compile", "warm1"])
        self.assertEqual(os.listdir(os.path.join(d, "tier3", "k")),
                         ["small.bin"])


class CacheParts(unittest.TestCase):

    def test_round_trip_and_old_format(self):
        import hashlib
        import tempfile

        from core import forward_run as fr
        src = tempfile.mkdtemp()
        for i in range(20):
            d = os.path.join(src, f"d{i % 3}")
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, f"f{i}"),
                 "wb").write(os.urandom(1000 * (i + 1)))

        def digest(root):
            return sorted((
                os.path.relpath(os.path.join(r, f), root),
                hashlib.md5(open(os.path.join(r, f), "rb").read()).hexdigest())
                          for r, _, fs in os.walk(root) for f in fs)

        out = tempfile.mktemp(suffix=".tgz")
        fr._save_parts(src, out)
        self.assertNotEqual(open(out, "rb").read(2),
                            b"\x1f\x8b")  # outer tar, not gzip
        dst = tempfile.mkdtemp()
        fr._restore_parts(out, dst)
        self.assertEqual(digest(src), digest(dst))
        old = tempfile.mktemp(suffix=".tgz")
        os.system(f"tar -czf {old} -C {src} .")
        dst2 = tempfile.mkdtemp()
        fr._restore_parts(old, dst2)
        self.assertEqual(digest(src), digest(dst2))

    def test_a_relative_output_path_is_read_from_the_working_directory(self):
        """Saves to a relative path, as a job does with the default `--out`.

        `forward_run.py` passes `wmb_out/wmb/forward/compile_cache.tgz`, which
        is relative to the job's working directory. `_save_parts` must write
        the file there, not under its own temporary directory.
        """
        import tempfile

        from core import forward_run as fr
        src = tempfile.mkdtemp()
        open(os.path.join(src, "f"), "wb").write(os.urandom(1000))
        work = tempfile.mkdtemp()
        os.makedirs(os.path.join(work, "wmb_out", "wmb", "forward"))
        cwd = os.getcwd()
        os.chdir(work)
        try:
            fr._save_parts(src, "wmb_out/wmb/forward/compile_cache.tgz")
        finally:
            os.chdir(cwd)
        self.assertTrue(
            os.path.isfile(
                os.path.join(work, "wmb_out", "wmb", "forward",
                             "compile_cache.tgz")))

    def test_a_failed_save_raises(self):
        """Raises when tar cannot write the output file.

        Before this check, a failed tar went unnoticed, and the job crashed
        later with a less clear error.
        """
        import tempfile

        from core import forward_run as fr
        src = tempfile.mkdtemp()
        open(os.path.join(src, "f"), "wb").write(os.urandom(1000))
        missing = os.path.join(tempfile.mkdtemp(), "no_such_dir", "out.tgz")
        with self.assertRaises(RuntimeError):
            fr._save_parts(src, missing)

    def test_an_empty_cache_saves_an_empty_archive(self):
        """Saves an empty cache directory as an archive with no parts."""
        import tempfile

        from core import forward_run as fr
        out = os.path.join(tempfile.mkdtemp(), "out.tgz")
        fr._save_parts(tempfile.mkdtemp(), out)
        dst = tempfile.mkdtemp()
        fr._restore_parts(out, dst)
        self.assertEqual(os.listdir(dst), [])

    def test_scratch_space_is_private_and_removed(self):
        """Keeps the scratch files in a private temporary directory.

        A fixed path such as `/tmp/wmb_parts_out` raises PermissionError on a
        shared VM when another user already owns it. Both functions must
        create their scratch directory under `tempfile.gettempdir()` and
        remove it when they finish.
        """
        import subprocess
        import tempfile
        from unittest import mock

        from core import forward_run as fr
        src = tempfile.mkdtemp()
        open(os.path.join(src, "f"), "wb").write(os.urandom(1000))
        out = os.path.join(tempfile.mkdtemp(), "out.tgz")
        dst = tempfile.mkdtemp()
        scratch = tempfile.mkdtemp()
        parts = []  # every part archive a tar command writes or reads
        real_popen = subprocess.Popen

        def spy(args, *rest, **kw):
            parts.extend(a for a in args
                         if os.path.isabs(a) and a.endswith("part0.tgz"))
            return real_popen(args, *rest, **kw)

        with mock.patch.object(tempfile, "tempdir", scratch), \
                mock.patch.object(subprocess, "Popen", spy):
            fr._save_parts(src, out)
            self.assertEqual(os.listdir(scratch), [])
            fr._restore_parts(out, dst)
            self.assertEqual(os.listdir(scratch), [])
        self.assertEqual(len(parts), 2)  # one save, one restore
        for p in parts:
            self.assertTrue(p.startswith(scratch + os.sep), p)
        self.assertEqual(os.listdir(dst), ["f"])


class CachePruneGuards(unittest.TestCase):
    """Restored saves: a dir tagged "cold" in the manifest (e.g. from an older tarball) is
    dropped only if this boot flipped keys, and never if a worker reports using it."""

    def _cache(self):
        import json
        import tempfile

        from core import forward_run as fr
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "torch_compile_cache", "same_hash"))
        json.dump({"same_hash": "cold"}, open(os.path.join(d, fr.MANIFEST),
                                              "w"))
        return d

    def test_no_flip_keeps_cold_dirs(self):
        from core import forward_run as fr
        d = self._cache()
        r = fr._prune_for_save(d, fr._compile_dirs(d),
                               restored=True)  # restored, no new dir
        self.assertEqual(r["compile_dirs_dropped"], [])
        self.assertTrue(
            os.path.isdir(os.path.join(d, "torch_compile_cache", "same_hash")))

    def test_used_dir_is_kept_even_after_a_flip(self):
        from core import forward_run as fr
        d = self._cache()
        before = fr._compile_dirs(d)
        os.makedirs(os.path.join(d, "torch_compile_cache",
                                 "flipped"))  # a flip happened
        r = fr._prune_for_save(d, before, restored=True, used={"same_hash"})
        self.assertEqual(r["compile_dirs_dropped"], [])

    def test_unused_cold_dir_dropped_after_a_flip(self):
        # negative control for the two above: the same setup without the guard's reason drops it
        from core import forward_run as fr
        d = self._cache()
        before = fr._compile_dirs(d)
        os.makedirs(os.path.join(d, "torch_compile_cache", "flipped"))
        r = fr._prune_for_save(d, before, restored=True, used=set())
        self.assertEqual(r["compile_dirs_dropped"], ["same_hash"])


class TwinInputs(unittest.TestCase):
    """forward_run feeds each request real, distinct inputs: a shared zero page table made every
    request read page 0 (decode RPA 2.5x slower than served) and collapsed MoE routing."""
    GROUPS = [{
        "gid": 0,
        "block_size": 1152,
        "is_mamba": True,
        "width": 232,
        "ckpt_window": 4
    }, {
        "gid": 1,
        "block_size": 1152,
        "is_mamba": False,
        "width": 228
    }]

    def test_pages_distinct_nonnull_and_sized(self):
        from core.forward_run import _page_plan
        rows = [[4, 159232], [4, 95498], [16376, 16376 + 30000]]
        pp = _page_plan(rows, self.GROUPS, 4007)
        attn, mamba = pp["tables"][1], pp["tables"][0]
        for (q, sl), row in zip(rows, attn):
            ids = [x for x in row if x]
            self.assertEqual(len(ids), -(-sl // 1152))  # cdiv(seq_len, page)
            self.assertEqual(row[:len(ids)], ids)  # a prefix of the row
            self.assertEqual(len(row), 228)
        for (q, sl), row in zip(rows, mamba):  # state block + 3 checkpoints
            so = (sl - 1) // 1152
            self.assertEqual([j for j, x in enumerate(row) if x],
                             list(range(so, so + 4)))
        every = [
            x for t in pp["tables"].values() for row in t for x in row if x
        ]
        self.assertEqual(len(every),
                         len(set(every)))  # no block shared, any group
        self.assertNotIn(0, every)  # 0 is the null block
        self.assertEqual(pp["used"], len(every))

    def test_kernel_pages_never_alias_another_groups_block(self):
        """j-61d110ab: the attention tables count 128-token kernel pages, the pool hands out
        1152-token blocks (9 kernel pages each) shared with the GDN state; numbering kernel pages
        from 1 put attention KV inside the GDN state blocks (and the output went non-finite)."""
        from core.forward_run import _page_plan
        g = [{
            "gid": 0,
            "block_size": 1152,
            "is_mamba": True,
            "width": 231,
            "ckpt_window": 4
        }, {
            "gid": 1,
            "block_size": 128,
            "is_mamba": False,
            "width": 2052
        }, {
            "gid": 2,
            "block_size": 128,
            "is_mamba": False,
            "width": 2052
        }]
        pp = _page_plan([[4, 159236], [4, 95498]], g, 4007, alloc_block=1152)
        owner = {}
        for gid, rows in pp["tables"].items():
            r_ = 1 if gid == 0 else 9
            for i, row in enumerate(rows):
                ids = [x for x in row if x]
                if gid:
                    self.assertEqual(len(ids), -(- [159236, 95498][i] // 128))
                    self.assertEqual(ids, list(range(ids[0],
                                                     ids[0] + len(ids))))
                    self.assertEqual(ids[0] % 9, 0)  # block-aligned
                for b in {x // r_ for x in ids}:  # pool block of each entry
                    self.assertNotIn(b, owner)  # one owner per pool block
                    owner[b] = (gid, i)
        self.assertNotIn(0, owner)
        self.assertEqual(pp["used"], len(owner))

    def test_kv_pages_never_land_in_another_batchs_state_blocks(self):
        """State blocks come from the top of the pool, KV blocks from the bottom, so a decode
        batch's KV pages never hold what a mixed batch's GDN state wrote (and vice versa)."""
        from core.forward_run import _page_plan
        g = [{
            "gid": 0,
            "block_size": 1152,
            "is_mamba": True,
            "width": 231,
            "ckpt_window": 4
        }, {
            "gid": 1,
            "block_size": 128,
            "is_mamba": False,
            "width": 2052
        }]
        mixed = _page_plan([[4, 118182], [4, 118182], [16376, 72824]],
                           g,
                           4007,
                           alloc_block=1152)
        dec = _page_plan([[4, 118182], [4, 118182]], g, 4007, alloc_block=1152)

        def state(pp):
            return {x for row in pp["tables"][0] for x in row if x}

        def kv(pp):
            return {x // 9 for row in pp["tables"][1] for x in row if x}

        self.assertFalse(kv(dec) & state(mixed))
        self.assertFalse(kv(mixed) & state(dec))
        self.assertTrue(min(state(dec)) > max(kv(mixed)))

    def test_pool_overflow_and_row_width_are_errors(self):
        from core.forward_run import _page_plan
        with self.assertRaises(ValueError):
            _page_plan([[4, 159232]] * 30, self.GROUPS,
                       4007)  # 30 x 139 pages > pool
        with self.assertRaises(ValueError):
            _page_plan([[4, 300000]], self.GROUPS,
                       4007)  # > the row's 228 pages

    def test_dcp_virtual_block(self):
        from core.forward_run import _page_plan
        g = [{"gid": 0, "block_size": 256, "is_mamba": False, "width": 1024}]
        pp = _page_plan([[4, 100000]], g, 10000, dcp=8)
        self.assertEqual(pp["used"], -(-100000 // (256 * 8)))

    def test_positions_and_state_slots(self):
        from core.forward_run import _mamba_slots, _positions
        self.assertEqual(_positions([[4, 100], [1, 7]], 8),
                         [96, 97, 98, 99, 6, 0, 0, 0])
        self.assertEqual(_mamba_slots(3, 4),
                         [1, 5, 9])  # slot 0 is null; stride 1+k


class ForwardCmdDefaults(unittest.TestCase):

    def test_iters_default_is_10(self):
        # 3 iterations let one slow host iteration move a repeat by 20-45%
        src = open(os.path.join(SCRIPTS, "core", "cli.py")).read()
        self.assertIn('("iters", 10, 10)', src)
        self.assertRegex(
            open(os.path.join(SCRIPTS, "core", "forward_run.py")).read(),
            r'"--iters",\s*type=int,\s*default=10\b')


class Supervise(unittest.TestCase):
    """Tests for forward_run.supervise.

    Each test replaces forward_run.py with a short stand-in script. Like vLLM, the
    stand-in starts its engine as a separate process. Each test runs supervise in a new
    Python interpreter, the way the recipe does, so that the test can check what
    supervise prints.
    """

    READY = ("[t+    1.0s] engine init (weights + KV + compile of buckets "
             "[16]): 1.0 s")
    ENGINE_ERROR = (
        "(EngineCore pid=786) RuntimeError: Worker failed with error "
        "'failed to compile MLIR module: CompileTimeScopedVmemOom', "
        "please check the stack trace above for the root cause")

    def _fake(self, body):
        """Writes a stand-in for forward_run.py; returns (stand-in, OUT, PIDS)."""
        import tempfile
        d = tempfile.mkdtemp()
        fake, out, pids = (os.path.join(d, n)
                           for n in ("fake_forward_run.py", "out", "pids"))
        with open(fake, "w") as f:
            f.write("import os, subprocess, sys, time\n"
                    f"OUT = {out!r}\nPIDS = {pids!r}\n" + body)
        return fake, out, pids

    def _supervise_cmd(self, fake, out, timeout_s, grace_s):
        """Returns the command that runs supervise around `fake`, as the recipe does."""
        code = ("import sys; from core.forward_run import supervise; "
                f"sys.exit(supervise([], {out!r}, {timeout_s}, {grace_s}, "
                f"[sys.executable, {fake!r}]))")
        return [sys.executable, "-c", code]

    def _run(self, body, timeout_s=60, grace_s=0.5):
        """Runs supervise around a stand-in; returns (process, failure.json, PIDS)."""
        import subprocess
        fake, out, pids = self._fake(body)
        p = subprocess.run(self._supervise_cmd(fake, out, timeout_s, grace_s),
                           cwd=SCRIPTS,
                           capture_output=True,
                           text=True,
                           timeout=120)
        failure = os.path.join(out, "failure.json")
        f = json.load(open(failure)) if os.path.exists(failure) else None
        ids = ([int(x) for x in open(pids).read().split()]
               if os.path.exists(pids) else [])
        return p, f, ids

    #: A stand-in whose engine process prints ERR to standard error, and then neither
    #: exits nor logs anything else. The stand-in waits for its engine, as vLLM's main
    #: process does.
    STUCK_ENGINE = (
        "print('ENGINE ARGS: --model x', flush=True)\n"
        "eng = subprocess.Popen([sys.executable, '-c', 'import sys, time; "
        "print(sys.argv[1], file=sys.stderr, flush=True); time.sleep(600)', "
        "ERR])\n"
        "open(PIDS, 'w').write(f'{os.getpid()} {eng.pid}')\n"
        "eng.wait()\n")

    def assertExited(self, pids):
        from core import forward_run as fr
        self.assertEqual(len(pids), 2)
        for pid in pids:
            self.assertIn(fr._proc_state(pid), (None, "Z"))

    def test_an_engine_that_logs_a_start_up_error_and_stays_alive_is_stopped(
            self):
        from core import forward_run as fr
        p, f, pids = self._run(f"ERR = {self.ENGINE_ERROR!r}\n" +
                               self.STUCK_ENGINE)
        self.assertEqual(p.returncode, fr.STOPPED_RC)
        self.assertEqual((f["stage"], f["reason"]),
                         ("engine start-up", "start_up_error"))
        self.assertEqual(f["detail"], self.ENGINE_ERROR)
        self.assertEqual(f["cause_context"][0], self.ENGINE_ERROR)
        # supervise copies the run's output, unchanged, to its standard output.
        self.assertIn("ENGINE ARGS: --model x\n" + self.ENGINE_ERROR, p.stdout)
        self.assertIn(
            "WMB FAILED during engine start-up. Reason: start_up_error.",
            p.stderr)
        self.assertExited(pids)

    def test_a_start_up_that_hangs_stops_at_the_limit(self):
        # The early ValueError says nothing about where start-up stopped, so
        # cause_context stays empty and `collect` prints the end of log_tail.
        from core import forward_run as fr
        p, f, pids = self._run("ERR = 'still compiling'\n"
                               "print('ValueError: harmless', flush=True)\n" +
                               self.STUCK_ENGINE,
                               timeout_s=0.5)
        self.assertEqual(p.returncode, fr.STOPPED_RC)
        self.assertEqual((f["reason"], f["start_up_limit_s"]),
                         ("start_up_timeout", 0.5))
        self.assertEqual(f["cause_context"], [])
        self.assertEqual(f["log_tail"][-1], "still compiling")
        self.assertExited(pids)

    def test_a_cancel_of_the_supervisor_stops_the_run(self):
        # `compute job cancel` sends SIGTERM to the process group of the recipe. The
        # run leads a process group of its own, so supervise must stop it.
        import signal
        import subprocess
        fake, out, pids_file = self._fake("ERR = 'compiling'\n" +
                                          self.STUCK_ENGINE)
        sup = subprocess.Popen(self._supervise_cmd(fake, out, 600, 600),
                               cwd=SCRIPTS,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               start_new_session=True)
        try:
            deadline, pids = time.time() + 60, []
            while len(pids) < 2 and time.time() < deadline:
                time.sleep(0.2)
                if os.path.exists(pids_file):
                    pids = [int(x) for x in open(pids_file).read().split()]
            os.killpg(sup.pid, signal.SIGTERM)
            self.assertEqual(sup.wait(timeout=60), 128 + signal.SIGTERM)
        finally:
            if sup.poll() is None:
                sup.kill()
        self.assertExited(pids)
        self.assertFalse(os.path.exists(os.path.join(out, "failure.json")))

    def test_a_run_whose_engine_starts_is_not_stopped(self):
        # vLLM logs a start-up error, but the engine becomes ready anyway. Then the
        # same error is logged again, and the steps run past both the grace period
        # and the start-up limit.
        p, f, _ = self._run(
            "print('ENGINE ARGS: --model x', flush=True)\n"
            f"print({self.ENGINE_ERROR!r}, file=sys.stderr, flush=True)\n"
            "time.sleep(0.2)\n"
            f"print({self.READY!r}, flush=True)\n"
            "time.sleep(0.2)\n"
            f"print({self.ENGINE_ERROR!r}, file=sys.stderr, flush=True)\n"
            "time.sleep(1)\n"
            "os.makedirs(OUT, exist_ok=True)\n"
            "open(os.path.join(OUT, 'result.json'), 'w').write('{{}}')\n",
            timeout_s=0.5)
        self.assertEqual(p.returncode, 0)
        self.assertIsNone(f)

    def test_the_recipe_runs_forward_run_under_supervise_and_packs_the_failure(
            self):
        # This test runs the real recipe and the real forward_run.py, with arguments
        # that forward_run.py rejects.
        import shutil
        import subprocess
        import tarfile
        import tempfile

        from core import cli
        d = tempfile.mkdtemp()
        shutil.copy(os.path.join(SCRIPTS, "core", "forward_run.py"), d)
        recipe = os.path.join(d, "recipe.py")
        with open(recipe, "w") as f:
            f.write(
                cli._RECIPE.format(plan="x.steps.json",
                                   out=cli.OUT_TGZ,
                                   image="img",
                                   args=["--no-such-flag"],
                                   start_timeout_s=45 * 60))
        p = subprocess.run([sys.executable, recipe],
                           capture_output=True,
                           text=True,
                           timeout=120)
        # The recipe passes on the exit code from argparse.
        self.assertEqual(p.returncode, 2)
        with tarfile.open(os.path.join(d, cli.OUT_TGZ)) as t:
            f = json.load(t.extractfile("./wmb/forward/failure.json"))
        self.assertEqual((f["stage"], f["reason"]),
                         ("before engine start-up", "exited"))
        self.assertTrue(
            any("required: --steps" in line for line in f["log_tail"]))


class HarnessSalt(unittest.TestCase):

    def test_salt_moves_the_compile_key(self):
        from core import cachelocal
        from core import forward_run as fr
        tw = {"hf_overrides": {"a": 1}, "server_args": {}, "env": {}}
        argv = fr.engine_argv(tw, "/m", [16], "/p")
        self.assertIn(fr.HARNESS_SALT, " ".join(argv))
        fr.HARNESS_SALT, old = "other", fr.HARNESS_SALT
        try:
            self.assertNotEqual(
                cachelocal.key("img", argv),
                cachelocal.key("img", fr.engine_argv(tw, "/m", [16], "/p")))
        finally:
            fr.HARNESS_SALT = old


try:
    import torch  # noqa: F401
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False


@unittest.skipUnless(
    _HAVE_TORCH, "needs torch (run with vllm-torchtpu/uv_venv/bin/python)")
class FillPages(unittest.TestCase):
    """The KV fill decides what the indexer scores and therefore who owns the top-k under DCP.
    Unfilled integer caches (fp8 stored as u8/u32) or repeated pages tie every score, the top-k
    breaks ties by index, and one DCP rank owns nearly every slot: a twin-only straggler."""

    @staticmethod
    def _caches():
        import torch
        return [
            torch.zeros(40, 8, 16, dtype=torch.bfloat16),
            torch.zeros(40, 4, 8, dtype=torch.float8_e4m3fn),
            torch.zeros(40, 6, 4,
                        dtype=torch.uint8),  # indexer K (fp8 as bytes)
            torch.zeros(40, 6, 4, dtype=torch.int32)
        ]  # SparseCore MLA layout (packed)

    def _fill(self, rank):
        import types
        from unittest import mock

        from core.forward_run import _fill_pages
        r = types.SimpleNamespace(kv_caches=self._caches())
        grp = types.SimpleNamespace(rank=rank)
        with mock.patch("vllm.distributed.get_world_group",
                        return_value=grp,
                        create=True):
            out = _fill_pages(r, list(range(5, 35)))
        return r.kv_caches, out

    def test_every_cache_filled_distinct_finite_and_only_its_pages(self):
        import torch
        caches, out = self._fill(0)
        self.assertEqual(out["tensors"],
                         [0, 1, 2, 3])  # integer caches included
        for t in caches:
            b = t.contiguous().view(torch.uint8)
            self.assertTrue(bool(
                b[:5].eq(0).all()))  # pages outside the list untouched
            pages = {bytes(b[j].flatten().tolist()) for j in range(5, 35)}
            self.assertEqual(len(pages), 30)  # no page repeats another
            vals = t[5:35].float() if torch.is_floating_point(t) else \
                b[5:35].view(torch.float8_e4m3fn).float()
            self.assertTrue(bool(
                torch.isfinite(vals).all()))  # no e4m3 NaN bytes

    def test_ranks_get_different_contents(self):
        import torch
        a, _ = self._fill(0)
        b, _ = self._fill(3)
        for x, y in zip(a, b):
            self.assertFalse(
                torch.equal(x.contiguous().view(torch.uint8),
                            y.contiguous().view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
