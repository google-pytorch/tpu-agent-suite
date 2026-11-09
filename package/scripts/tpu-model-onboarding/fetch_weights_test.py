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
"""Which files the weight fetch downloads.

gpt-oss-20b downloaded its 13.8 GB twice because `*.safetensors` also matched
`original/model.safetensors`. The layouts below are taken from the real
repos' file listings.
"""

import contextlib
import errno
import io
import json
import os
import posixpath
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import fetch_weights as fw

GiB = 1024**3
SHA = "0123abcd"


def fake_hub(sizes: dict[str, int]) -> types.ModuleType:
    """A stand-in for huggingface_hub, so these tests run without it."""
    hub = types.ModuleType("huggingface_hub")
    hub.constants = types.SimpleNamespace(HF_HUB_CACHE="")
    hub.HfApi = mock.Mock()
    info = hub.HfApi.return_value.model_info.return_value
    info.sha = SHA
    info.siblings = [
        types.SimpleNamespace(rfilename=name, size=size)
        for name, size in sizes.items()
    ]
    hub.hf_hub_download = mock.Mock()
    hub.snapshot_download = mock.Mock()
    return hub


@contextlib.contextmanager
def uses_hub(hub: types.ModuleType, cache: str):
    hub.constants.HF_HUB_CACHE = cache
    with mock.patch.dict(sys.modules, {
            "huggingface_hub": hub,
            "huggingface_hub.constants": hub.constants,
    }):
        yield


def write(path: str, size: int) -> None:
    with open(path, "wb") as f:
        f.write(b"x" * size)


def never(_):
    raise AssertionError("no index should be read")


class SelectFilesTest(unittest.TestCase):

    def test_gpt_oss_skips_original_and_metal(self):
        shards = [f"model-0000{i}-of-00002.safetensors" for i in range(3)]
        sizes = {s: 4 * GiB for s in shards}
        sizes.update({
            "model.safetensors.index.json": 40_000,
            "config.json": 1_000,
            "tokenizer.json": 27_000_000,
            "original/model.safetensors": 13 * GiB,
            "original/config.json": 500,
            "metal/model.bin": 13 * GiB,
        })
        sel = fw.select_files(sizes, lambda _: shards * 100)
        self.assertEqual(sel.layout, "model.safetensors.index.json")
        self.assertNotIn("original/model.safetensors", sel.files)
        self.assertNotIn("metal/model.bin", sel.files)
        for name in shards + ["config.json", "tokenizer.json"]:
            self.assertIn(name, sel.files)
        self.assertEqual(sel.skipped_bytes, 26 * GiB)

    def test_mistral_skips_consolidated(self):
        shards = [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]
        sizes = {s: 5 * GiB for s in shards}
        sizes.update({
            "model.safetensors.index.json": 24_000,
            "consolidated.safetensors": 14 * GiB,
            "tokenizer.model.v3": 587_000,
        })
        sel = fw.select_files(sizes, lambda _: shards)
        self.assertNotIn("consolidated.safetensors", sel.files)
        self.assertIn("tokenizer.model.v3", sel.files)

    def test_single_file_prefers_safetensors_over_bin(self):
        sizes = {
            "model.safetensors": GiB,
            "pytorch_model.bin": GiB,
            "tf_model.h5": GiB,
            "flax_model.msgpack": GiB,
            "config.json": 600,
        }
        sel = fw.select_files(sizes, never)
        self.assertEqual(sel.files, ["config.json", "model.safetensors"])

    def test_index_naming_a_missing_shard_falls_through(self):
        # A stale index must not select files the repo does not have.
        sizes = {
            "model.safetensors.index.json": 100,
            "model.safetensors": GiB,
        }
        sel = fw.select_files(sizes, lambda _: ["gone.safetensors"])
        self.assertEqual(sel.layout, "model.safetensors")

    def test_fallback_keeps_subfolder_weights_and_drops_copies(self):
        # A diffusers-style repo: no top-level transformers checkpoint.
        sizes = {
            "unet/diffusion_pytorch_model.safetensors": 3 * GiB,
            "unet/diffusion_pytorch_model.bin": 3 * GiB,
            "vae/diffusion_pytorch_model.safetensors": GiB,
            "text_encoder/model.onnx": GiB,
            "original/v1.ckpt": 4 * GiB,
            "model_index.json": 500,
        }
        sel = fw.select_files(sizes, never)
        self.assertEqual(sel.layout, "fallback")
        self.assertEqual(sel.files, [
            "model_index.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ])

    def test_fallback_never_selects_no_weights(self):
        sizes = {"original/consolidated.00.pth": 16 * GiB, "params.json": 200}
        sel = fw.select_files(sizes, never)
        self.assertIn("original/consolidated.00.pth", sel.files)

    def test_large_non_weight_files_are_skipped(self):
        sizes = {"model.safetensors": GiB, "demo.mp4": 300 * 1024 * 1024}
        sel = fw.select_files(sizes, never)
        self.assertEqual(sel.files, ["model.safetensors"])


class FormatProgressTest(unittest.TestCase):

    def test_line(self):
        line = fw.format_progress(int(13.1e9), int(65.5e9), 50e6)
        self.assertEqual(
            line, "WEIGHTS PROGRESS 13.1/65.5 GB (20%) 50 MB/s eta 17m")

    def test_no_eta_when_stalled(self):
        self.assertNotIn("eta", fw.format_progress(1, 10, 0.0))


class FetchTest(unittest.TestCase):

    def test_local_directory_is_ready_without_network(self):
        with tempfile.TemporaryDirectory() as d:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(fw.fetch(d), d)
            self.assertTrue(out.getvalue().startswith(fw.READY))

    def test_path_file_names_the_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            path_file = os.path.join(d, "weights_path.txt")
            with contextlib.redirect_stdout(io.StringIO()):
                fw.main([d, "--path-file", path_file])
            with open(path_file, encoding="utf-8") as f:
                self.assertEqual(f.read().strip(), d)

    def test_downloads_only_the_selection_and_prints_ready(self):
        hub = fake_hub({
            "model.safetensors": 10,
            "original/model.safetensors": 10,
            "config.json": 1,
        })
        hub.snapshot_download.return_value = "/snap"
        with tempfile.TemporaryDirectory() as cache, uses_hub(hub, cache):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(fw.fetch("org/m", interval=0.01), "/snap")
        kwargs = hub.snapshot_download.call_args.kwargs
        self.assertEqual(sorted(kwargs["allow_patterns"]),
                         ["config.json", "model.safetensors"])
        self.assertIsNone(kwargs["revision"])
        self.assertIn(f"{fw.READY} /snap", out.getvalue())

    def test_progress_counts_only_this_selection(self):
        # The blob store already holds 50 bytes of another revision, and one
        # of the two selected files (40 of 100 bytes) is complete in this
        # revision's snapshot. Progress must start at 40/100, not at 100%.
        hub = fake_hub({"a.safetensors": 40, "b.safetensors": 60})
        with tempfile.TemporaryDirectory() as cache, uses_hub(hub, cache):
            repo = os.path.join(cache, "models--org--m")
            os.makedirs(os.path.join(repo, "blobs"))
            write(os.path.join(repo, "blobs", "other-revision"), 50)
            write(os.path.join(repo, "blobs", "a-blob"), 40)
            os.makedirs(os.path.join(repo, "snapshots", SHA))
            os.symlink(os.path.join(repo, "blobs", "a-blob"),
                       os.path.join(repo, "snapshots", SHA, "a.safetensors"))
            arrived = threading.Event()

            def download(*_args, **_kwargs):
                write(os.path.join(repo, "blobs", "b.incomplete"), 30)
                arrived.set()
                time.sleep(0.2)
                return "/snap"

            hub.snapshot_download.side_effect = download
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                fw.fetch("org/m", interval=0.05)
        progress = [
            line for line in out.getvalue().splitlines()
            if line.startswith(fw.PROGRESS)
        ]
        self.assertTrue(arrived.is_set())
        self.assertTrue(progress)
        # 40 already in the snapshot + 30 written by this run.
        self.assertIn("(70%)", progress[-1])

    def test_a_model_that_cannot_fit_fails_before_downloading(self):
        # Gemma-4-31B's 62.55 GB went onto a pod disk with ~23 GB free, and
        # the pod was evicted with exit 137 part-way through the download.
        hub = fake_hub({"model.safetensors": 100, "config.json": 1})
        with tempfile.TemporaryDirectory() as cache, uses_hub(hub, cache), \
                mock.patch.object(fw, "cache_space",
                                  return_value=fw.Space(50)):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), \
                    self.assertRaises(OSError) as ctx:
                fw.fetch("org/m")
        self.assertEqual(ctx.exception.errno, errno.ENOSPC)
        hub.snapshot_download.assert_not_called()
        self.assertIn(f"{fw.FAILED}: org/m needs 0.0 GB more", out.getvalue())

    def test_space_is_measured_where_the_cache_is(self):
        # A shared volume (/hf-cache) or a cache the user moved is where the
        # download goes, so that, not /dev/shm, is what must have room.
        hub = fake_hub({"model.safetensors": 100})
        hub.snapshot_download.return_value = "/snap"
        with tempfile.TemporaryDirectory() as cache, uses_hub(hub, cache), \
                mock.patch.object(fw, "cache_space",
                                  return_value=fw.Space(1000)) as space:
            with contextlib.redirect_stdout(io.StringIO()):
                fw.fetch("org/m", interval=0.01)
        space.assert_called_once_with(os.path.join(cache, "models--org--m"))

    def test_space_counts_only_what_is_still_missing(self):
        # 40 of the 100 bytes are already in this revision's snapshot.
        hub = fake_hub({"a.safetensors": 40, "b.safetensors": 60})
        hub.snapshot_download.return_value = "/snap"
        with tempfile.TemporaryDirectory() as cache, uses_hub(hub, cache), \
                mock.patch.object(fw, "cache_space",
                                  return_value=fw.Space(60)):
            repo = os.path.join(cache, "models--org--m")
            os.makedirs(os.path.join(repo, "blobs"))
            write(os.path.join(repo, "blobs", "a-blob"), 40)
            os.makedirs(os.path.join(repo, "snapshots", SHA))
            os.symlink(os.path.join(repo, "blobs", "a-blob"),
                       os.path.join(repo, "snapshots", SHA, "a.safetensors"))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(fw.fetch("org/m", interval=0.01), "/snap")

    def test_failure_is_logged_then_raised(self):
        hub = fake_hub({})
        hub.HfApi.return_value.model_info.side_effect = OSError("401 gated")
        with tempfile.TemporaryDirectory() as cache, uses_hub(hub, cache):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), \
                    self.assertRaises(OSError):
                fw.fetch("org/gated")
        self.assertIn(f"{fw.FAILED}: could not list org/gated: 401 gated",
                      out.getvalue())


class CompileReserveTest(unittest.TestCase):

    def test_the_reserve_covers_the_measured_caches(self):
        # vLLM 0.29 on TPU, measured on /dev/shm once the server was up:
        # (layers, cache bytes). The reserve keeps at least 1.5x each.
        for layers, measured in ((48, 7_910_490_178), (60, 9_690_612_534),
                                 (60, 10_473_139_876)):
            self.assertGreaterEqual(fw.compile_reserve(layers), 1.5 * measured,
                                    layers)

    def test_layers_come_from_the_config(self):
        cases = [
            ({
                "num_hidden_layers": 32
            }, 32),
            # A multimodal model names its text model's depth.
            ({
                "text_config": {
                    "num_hidden_layers": 60
                },
                "vision_config": {
                    "num_hidden_layers": 27
                }
            }, 60),
            ({
                "n_layer": 24
            }, 24),
            ({}, None),
            ({
                "num_hidden_layers": "60"
            }, None),
        ]
        for config, layers in cases:
            with self.subTest(config=config):
                self.assertEqual(fw.config_layers(config), layers)
        self.assertEqual(fw.compile_reserve(None), fw.COMPILE_RESERVE_UNKNOWN)

    def test_a_config_that_cannot_be_read_takes_the_unknown_reserve(self):
        # It only sizes the reserve, so it never fails the fetch.
        hub = fake_hub({})
        hub.hf_hub_download.side_effect = OSError("offline")
        with tempfile.TemporaryDirectory() as cache, uses_hub(hub, cache):
            self.assertIsNone(fw._layers("org/m", None, {"config.json": 1}))
        self.assertIsNone(fw._layers("org/m", None, {}))


class SpaceProblemTest(unittest.TestCase):
    # Gemma-4-31B: 62.58 GB of weights in 60 layers.
    WEIGHTS = 62_578_686_256
    RESERVE = 17 * GiB

    def test_dev_shm_keeps_room_for_compile_caches(self):
        # 58.3 GiB of weights alone fit the default 64 GiB; with room kept
        # for compile caches they do not, and the line says what to set.
        shm = fw.Space(64 * GiB, shm_limit=64 * GiB, shm_used=0)
        problem = fw.space_problem("google/gemma-4-31b-it", "/dev/shm/hub",
                                   self.WEIGHTS, self.RESERVE, shm)
        self.assertIn("18.3 GB for compile caches", problem)
        self.assertIn("evicted part-way", problem)
        self.assertIn("Set shm_size_gb to at least 76,", problem)
        roomy = fw.Space(128 * GiB, shm_limit=128 * GiB, shm_used=0)
        self.assertIsNone(
            fw.space_problem("m", "/dev/shm/hub", self.WEIGHTS, self.RESERVE,
                             roomy))

    def test_the_reserve_holds_when_the_weights_are_cached(self):
        # Nothing left to download, but the caches still need room.
        shm = fw.Space(1 * GiB, shm_limit=64 * GiB, shm_used=63 * GiB)
        problem = fw.space_problem("m", "/dev/shm/hub", 0, self.RESERVE, shm)
        self.assertIn("Set shm_size_gb to at least 80,", problem)

    def test_elsewhere_only_the_weights_are_sized(self):
        # A PVC or a user's volume runs out of space (ENOSPC); nothing is
        # evicted, and the compile caches are not on it.
        self.assertIsNone(
            fw.space_problem("m", "/hf-cache", 100, self.RESERVE,
                             fw.Space(100)))
        problem = fw.space_problem("m", "/hf-cache", 100, self.RESERVE,
                                   fw.Space(50))
        self.assertIn("would run out of space part-way", problem)
        self.assertNotIn("evicted", problem)
        self.assertNotIn("shm_size_gb", problem)


class CacheSpaceTest(unittest.TestCase):
    # statvfs on a pod's memory-backed /dev/shm can report the node's whole
    # memory (709 GB) rather than the volume's cap.
    BLOCK = 4096
    NODE = os.statvfs_result(
        (BLOCK, BLOCK, 709 * 10**6, 700 * 10**6, 700 * 10**6, 0, 0, 0, 0, 255))
    FREE = 700 * 10**6 * BLOCK

    def free(self, path: str, limit: str | None) -> int:
        return self.space(path, limit).free

    def space(self, path: str, limit: str | None) -> fw.Space:
        env = {} if limit is None else {fw.SHM_LIMIT_ENV: limit}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(fw.os, "statvfs", return_value=self.NODE):
            if limit is None:
                os.environ.pop(fw.SHM_LIMIT_ENV, None)
            return fw.cache_space(path)

    def test_dev_shm_is_held_to_its_cap(self):
        # 9e6 blocks of the node's tmpfs pages are this volume's own.
        cap = 64 * GiB
        used = 9 * 10**6 * self.BLOCK
        self.assertEqual(
            self.space("/dev/shm/cache/hub/models--org--m", str(cap)),
            fw.Space(cap - used, shm_limit=cap, shm_used=used))

    def test_symlinks_are_followed(self):
        # /hf-cache's statvfs, no cap, though the path names /dev/shm; and
        # the cap for a path that only leads to /dev/shm.
        cap = 64 * GiB
        with mock.patch.object(fw.os.path,
                               "realpath",
                               return_value="/hf-cache/hub"):
            self.assertEqual(self.free("/dev/shm/cache/hub", str(cap)),
                             self.FREE)
        with mock.patch.object(fw.os.path,
                               "realpath",
                               return_value="/dev/shm/cache/hub"):
            self.assertEqual(self.free("/data/hub", str(cap)),
                             cap - 9 * 10**6 * self.BLOCK)

    def test_other_paths_and_no_cap_use_statvfs(self):
        self.assertEqual(self.free("/data/hub", str(64 * GiB)), self.FREE)
        self.assertEqual(self.free("/dev/shm/cache", None), self.FREE)


# A checkpoint folder as the vertex-model-garden-public-us bucket really
# lays one out: the same model in five formats, and the folder nested in
# itself. Only the safetensors and the small files are what a loader reads.
URL = "gs://bucket/models/xlm-roberta-base"
FOLDER = URL + "/"
ROBERTA = {
    "README.md": 5_238,
    "config.json": 615,
    "flax_model.msgpack": 1_113_187,
    "model.onnx": 1_881_470,
    "model.safetensors": 1_115_567,
    "pytorch_model.bin": 1_115_590,
    "sentencepiece.bpe.model": 50_690,
    "tf_model.h5": 1_112_441,
    "tokenizer.json": 90_967,
    "tokenizer_config.json": 25,
    "xlm-roberta-base/config.json": 615,
    "xlm-roberta-base/model.safetensors": 1_115_567,
}


class FakeStorage:
    """Stands in for gcloud/gsutil/the library: a dict of object sizes.

    `contents` gives the bytes of the files fetch_gcs reads (config, index);
    every other object is `size` bytes of filler.
    """

    def __init__(self,
                 sizes,
                 config=None,
                 contents=None,
                 fail_on=None,
                 short=()):
        self.contents = dict(contents or {})
        self.sizes = dict(sizes)
        if "config.json" in self.sizes:
            self.contents.setdefault(
                "config.json",
                json.dumps(config or {
                    "num_hidden_layers": 12
                }).encode("utf-8"))
        for name, data in self.contents.items():
            self.sizes[name] = len(data)
        self.fail_on = fail_on
        self.short = set(short)
        self.copied = []

    def list(self, folder_url):
        return dict(self.sizes) if folder_url == FOLDER else {}

    def copy_many(self, srcs, folder):
        for src in srcs:
            self.copy(src, os.path.join(folder, posixpath.basename(src)))

    def copy(self, src, dst):
        name = src[len(FOLDER):]
        if self.fail_on and name == self.fail_on[0]:
            raise fw._GcsError(self.fail_on[1])
        self.copied.append(name)
        if name in self.contents:
            with open(dst, "wb") as f:
                f.write(self.contents[name])
            return
        write(dst, self.sizes[name] - (1 if name in self.short else 0))


@contextlib.contextmanager
def gcs(storage, cache, env=None):
    """Runs fetch_gcs against `storage`, copying under `cache`."""
    env = dict(env or {})
    env.setdefault(fw.CHECKPOINT_DIR_ENV, cache)
    with mock.patch.object(fw, "_gcs_storage", return_value=storage), \
            mock.patch.dict(os.environ, env), \
            mock.patch.object(fw, "cache_space",
                              return_value=fw.Space(100 * GiB)):
        yield


def run(fn, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = fn(*args, **kwargs)
    return result, out.getvalue()


class ParseGcsUrlTest(unittest.TestCase):

    def test_bucket_and_folder(self):
        self.assertEqual(fw.parse_gcs_url("gs://b/models/m/"),
                         ("b", "models/m"))
        self.assertEqual(fw.parse_gcs_url("gs://b"), ("b", ""))

    def test_no_bucket_or_not_gcs_is_refused(self):
        for url in ("gs://", "gs:///models", "/data/m", "s3://b/m"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                fw.parse_gcs_url(url)


class GcsDestTest(unittest.TestCase):

    def test_checkpoint_dir_wins(self):
        with mock.patch.dict(os.environ, {
                fw.CHECKPOINT_DIR_ENV: "/ckpt",
                "HF_HUB_CACHE": "/hf-cache"
        }):
            dest = fw.gcs_dest(URL)
        self.assertTrue(dest.startswith("/ckpt/"))
        self.assertTrue(dest.endswith("/xlm-roberta-base"))

    def test_otherwise_beside_the_hub_cache(self):
        # gke_job points HF_HUB_CACHE at the shared PVC, or at /dev/shm: the
        # copy must land there too, never on the pod's own disk.
        for cache in ("/hf-cache", "/dev/shm/cache/huggingface/hub"):
            with self.subTest(cache=cache), mock.patch.dict(
                    os.environ, {"HF_HUB_CACHE": cache}):
                os.environ.pop(fw.CHECKPOINT_DIR_ENV, None)
                self.assertTrue(
                    fw.gcs_dest(URL).startswith(os.path.join(cache, "gcs")))

    def test_same_folder_name_in_two_buckets_does_not_collide(self):
        with mock.patch.dict(os.environ, {fw.CHECKPOINT_DIR_ENV: "/c"}):
            a = fw.gcs_dest("gs://team-a/models/Qwen3-8B")
            b = fw.gcs_dest("gs://team-b/models/Qwen3-8B")
            c = fw.gcs_dest("gs://team-a/models/Qwen3-8B/")
        self.assertNotEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(os.path.basename(a), "Qwen3-8B")


class FetchGcsTest(unittest.TestCase):

    def test_copies_only_what_a_loader_reads(self):
        storage = FakeStorage(ROBERTA)
        with tempfile.TemporaryDirectory() as cache, gcs(storage, cache):
            path, out = run(fw.fetch_gcs, URL, interval=0.01)
            self.assertTrue(path.startswith(cache))
            self.assertTrue(
                os.path.isfile(os.path.join(path, "model.safetensors")))
        for skipped in ("flax_model.msgpack", "model.onnx", "tf_model.h5",
                        "pytorch_model.bin",
                        "xlm-roberta-base/model.safetensors"):
            self.assertNotIn(skipped, storage.copied)
        for kept in ("model.safetensors", "config.json", "tokenizer.json",
                     "sentencepiece.bpe.model"):
            self.assertIn(kept, storage.copied)
        self.assertIn(f"{fw.READY} {path}", out)
        self.assertIn("WEIGHTS PLAN", out)

    def test_small_only_copies_no_weights(self):
        storage = FakeStorage(ROBERTA)
        with tempfile.TemporaryDirectory() as cache, gcs(storage, cache):
            run(fw.fetch_gcs, URL, small_only=True, interval=0.01)
        self.assertFalse(
            [n for n in storage.copied if fw._ext(n) in fw.WEIGHT_EXTS])
        self.assertIn("tokenizer.json", storage.copied)

    def test_sharded_checkpoint_follows_the_index(self):
        shards = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors"
        ]
        sizes = {s: 1000 for s in shards}
        sizes.update({"config.json": 0, "consolidated.safetensors": 2000})
        index = json.dumps({"weight_map": {"a": shards[0], "b": shards[1]}})
        storage = FakeStorage(
            sizes,
            contents={"model.safetensors.index.json": index.encode("utf-8")})
        with tempfile.TemporaryDirectory() as cache, gcs(storage, cache):
            run(fw.fetch_gcs, URL, interval=0.01)
        self.assertEqual(sorted(n for n in storage.copied if "of-0" in n),
                         shards)
        self.assertNotIn("consolidated.safetensors", storage.copied)
        # The index and config are copied first, before any weights.
        self.assertEqual(set(storage.copied[:2]),
                         {"config.json", "model.safetensors.index.json"})

    def test_a_checkpoint_that_cannot_fit_fails_before_copying_weights(self):
        storage = FakeStorage(ROBERTA)
        with tempfile.TemporaryDirectory() as cache, gcs(storage, cache), \
                mock.patch.object(fw, "cache_space",
                                  return_value=fw.Space(1000)):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), \
                    self.assertRaises(OSError) as ctx:
                fw.fetch_gcs(URL)
        self.assertEqual(ctx.exception.errno, errno.ENOSPC)
        self.assertNotIn("model.safetensors", storage.copied)
        self.assertIn(f"{fw.FAILED}: {URL} needs", out.getvalue())
        self.assertIn(fw.CHECKPOINT_DIR_ENV, out.getvalue())

    def test_dev_shm_keeps_the_compile_reserve_sized_by_the_config(self):
        storage = FakeStorage(ROBERTA, config={"num_hidden_layers": 48})
        seen = []

        def problem(model_id, cache, needed, reserve, space):
            seen.append(reserve)
            return None

        with tempfile.TemporaryDirectory() as cache, gcs(storage, cache), \
                mock.patch.object(fw, "space_problem", side_effect=problem):
            run(fw.fetch_gcs, URL, interval=0.01)
            run(fw.fetch_gcs, URL, small_only=True, interval=0.01)
        self.assertEqual(seen, [fw.compile_reserve(48), 0])

    def test_files_already_whole_are_not_copied_again(self):
        storage = FakeStorage(ROBERTA)
        with tempfile.TemporaryDirectory() as cache, gcs(storage, cache):
            run(fw.fetch_gcs, URL, interval=0.01)
            storage.copied.clear()
            # A staging folder left by a job killed after the copy finished.
            dest = fw.gcs_dest(URL)
            os.makedirs(dest + fw.PARTIAL_SUFFIX)
            write(os.path.join(dest + fw.PARTIAL_SUFFIX, "stale"), 5)
            _, out = run(fw.fetch_gcs, URL, interval=0.01)
            self.assertFalse(os.path.exists(dest + fw.PARTIAL_SUFFIX))
        self.assertEqual(storage.copied, [])
        self.assertIn("copied 0.0 GB", out)

    def test_space_counts_only_what_is_still_missing(self):
        storage = FakeStorage(ROBERTA)
        needed = []

        def problem(model_id, cache, n, reserve, space):
            needed.append(n)
            return None

        with tempfile.TemporaryDirectory() as cache, gcs(storage, cache), \
                mock.patch.object(fw, "space_problem", side_effect=problem):
            dest = fw.gcs_dest(URL)
            os.makedirs(dest)
            write(os.path.join(dest, "model.safetensors"),
                  ROBERTA["model.safetensors"])
            run(fw.fetch_gcs, URL, interval=0.01)
        self.assertNotIn("model.safetensors", storage.copied)
        self.assertLess(needed[0], ROBERTA["model.safetensors"])

    def test_a_partial_copy_left_by_a_killed_job_is_copied_again(self):
        storage = FakeStorage(ROBERTA)
        with tempfile.TemporaryDirectory() as cache, gcs(storage, cache):
            dest = fw.gcs_dest(URL)
            os.makedirs(dest + fw.PARTIAL_SUFFIX)
            write(os.path.join(dest + fw.PARTIAL_SUFFIX, "model.safetensors"),
                  10)
            os.makedirs(dest)
            write(os.path.join(dest, "model.safetensors"), 10)
            run(fw.fetch_gcs, URL, interval=0.01)
            self.assertEqual(
                os.path.getsize(os.path.join(dest, "model.safetensors")),
                ROBERTA["model.safetensors"])
            self.assertFalse(os.path.exists(dest + fw.PARTIAL_SUFFIX))
        self.assertIn("model.safetensors", storage.copied)

    def test_files_are_staged_beside_the_checkpoint_not_in_it(self):
        storage = FakeStorage(ROBERTA)
        seen = []
        real_copy = storage.copy

        def copy(src, dst):
            seen.append(dst)
            real_copy(src, dst)

        storage.copy = copy
        with tempfile.TemporaryDirectory() as cache, gcs(storage, cache):
            dest = fw.gcs_dest(URL)
            run(fw.fetch_gcs, URL, interval=0.01)
        self.assertTrue(seen)
        for dst in seen:
            self.assertTrue(dst.startswith(dest + fw.PARTIAL_SUFFIX + os.sep))

    def test_one_copy_call_per_subfolder(self):
        storage = FakeStorage(ROBERTA)
        calls = []
        real_many = storage.copy_many

        def copy_many(srcs, folder):
            calls.append(list(srcs))
            real_many(srcs, folder)

        storage.copy_many = copy_many
        with tempfile.TemporaryDirectory() as cache, gcs(storage, cache):
            run(fw.fetch_gcs, URL, interval=0.01)
        # The metadata first, then the top folder and the nested one.
        self.assertEqual(len(calls), 3)
        self.assertEqual([posixpath.basename(s) for s in calls[0]],
                         ["config.json"])

    def test_a_file_that_arrives_short_fails(self):
        storage = FakeStorage(ROBERTA, short={"model.safetensors"})
        with tempfile.TemporaryDirectory() as cache, gcs(storage, cache):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), \
                    self.assertRaises(OSError) as ctx:
                fw.fetch_gcs(URL, interval=0.01)
        self.assertEqual(ctx.exception.errno, errno.EIO)
        self.assertIn("did not arrive whole", out.getvalue())

    def test_no_permission_says_what_to_do(self):
        storage = FakeStorage(ROBERTA)
        storage.list = mock.Mock(side_effect=fw._GcsError(
            "ERROR: [vm-sa@p.iam] does not have storage.objects.list access "
            "to the Google Cloud Storage bucket. Permission 'storage.objects."
            "list' denied on resource (or it may not exist). 403"))
        with tempfile.TemporaryDirectory() as cache, gcs(storage, cache):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), \
                    self.assertRaises(PermissionError):
                fw.fetch_gcs(URL)
        self.assertIn(f"{fw.FAILED}: [Errno 13] no permission to read {URL}",
                      out.getvalue())
        self.assertIn(fw.GCS_TOKEN_ENV, out.getvalue())

    def test_permission_lost_mid_copy_is_reported_the_same_way(self):
        storage = FakeStorage(ROBERTA,
                              fail_on=("model.safetensors", "HTTP 403"))
        with tempfile.TemporaryDirectory() as cache, gcs(storage, cache):
            with contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaises(PermissionError):
                fw.fetch_gcs(URL, interval=0.01)

    def test_an_empty_or_missing_folder_fails(self):
        storage = FakeStorage({})
        with tempfile.TemporaryDirectory() as cache, gcs(storage, cache):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), \
                    self.assertRaises(FileNotFoundError):
                fw.fetch_gcs(URL)
        self.assertIn(f"nothing at {URL}", out.getvalue())
        storage.list = mock.Mock(side_effect=fw._GcsError(
            "ERROR: (gcloud.storage.du) One or more URLs matched no objects."))
        with tempfile.TemporaryDirectory() as cache, gcs(storage, cache):
            with contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaises(FileNotFoundError):
                fw.fetch_gcs(URL)

    def test_object_names_that_leave_the_folder_are_dropped(self):
        storage = FakeStorage({"model.safetensors": 10, "config.json": 0})
        storage.sizes["../escape.json"] = 5
        with tempfile.TemporaryDirectory() as cache, gcs(storage, cache):
            run(fw.fetch_gcs, URL, interval=0.01)
            self.assertFalse(os.path.exists(os.path.join(cache,
                                                         "escape.json")))
        self.assertNotIn("../escape.json", storage.copied)
        with self.assertRaises(ValueError):
            fw._local("/c/x", "../y")

    def test_progress_is_printed_while_copying(self):
        storage = FakeStorage(ROBERTA)
        real_copy = storage.copy

        def slow_copy(src, dst):
            real_copy(src, dst)
            if src.endswith("model.safetensors"):
                time.sleep(0.2)

        storage.copy = slow_copy
        with tempfile.TemporaryDirectory() as cache, gcs(storage, cache):
            _, out = run(fw.fetch_gcs, URL, interval=0.05)
        self.assertIn(fw.PROGRESS, out)

    def test_concurrent_copies_of_one_url_take_turns(self):
        # Two Jobs on one shared volume: the second must not read a file the
        # first is still writing, nor copy it a second time. The second
        # starts only once the first is inside its copy, holding the lock,
        # so the test does not depend on how the threads are scheduled.
        storage = FakeStorage(ROBERTA)
        active, overlap = [0], []
        real_copy = storage.copy
        lock = threading.Lock()
        first_is_copying = threading.Event()
        errors = []

        def copy(src, dst):
            with lock:
                active[0] += 1
                overlap.append(active[0])
            if not first_is_copying.is_set():
                first_is_copying.set()
                time.sleep(0.3)  # long enough for the second to queue
            real_copy(src, dst)
            with lock:
                active[0] -= 1

        def job():
            try:
                fw.fetch_gcs(URL, interval=0.01)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                errors.append(exc)

        storage.copy = copy
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as cache, gcs(storage, cache), \
                contextlib.redirect_stdout(out):
            first = threading.Thread(target=job)
            second = threading.Thread(target=job)
            first.start()
            self.assertTrue(first_is_copying.wait(5))
            second.start()
            first.join(10)
            second.join(10)
        self.assertEqual(errors, [])
        self.assertEqual(max(overlap), 1)
        self.assertEqual(storage.copied.count("model.safetensors"), 1)
        # The one that waited says so, so its log does not look hung.
        self.assertIn("WEIGHTS WAIT", out.getvalue())

    def test_fetch_of_a_gcs_url_never_asks_the_hub(self):
        with mock.patch.object(fw, "fetch_gcs", return_value="/c/m") as copy, \
                mock.patch.dict(sys.modules, {"huggingface_hub": None}):
            path, out = run(fw.fetch, URL, revision="main")
        self.assertEqual(path, "/c/m")
        copy.assert_called_once_with(URL, interval=fw.DEFAULT_INTERVAL_SEC)
        self.assertIn("--revision main ignored", out)

    def test_fetch_in_hf_hub_offline_mode_resolves_cached_snapshot(self):
        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.constants = types.SimpleNamespace()
        fake_hf.snapshot_download = mock.Mock(return_value="/cache/snap/abc")
        with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}), \
                mock.patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            path, out = run(fw.fetch, "org/model", revision="main")
        self.assertEqual(path, "/cache/snap/abc")
        fake_hf.snapshot_download.assert_called_once_with(
            "org/model", revision="main", local_files_only=True)
        self.assertIn("cached snapshot, nothing to download", out)

    def test_main_writes_the_local_path(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(fw, "fetch_gcs", return_value="/c/m"):
            path_file = os.path.join(d, "path.txt")
            run(fw.main, [URL, "--path-file", path_file])
            with open(path_file, encoding="utf-8") as f:
                self.assertEqual(f.read().strip(), "/c/m")


class GcsProblemTest(unittest.TestCase):
    # What the tools really printed, on a workstation and in a pod.

    def test_no_credentials_is_a_permission_problem_not_a_missing_folder(self):
        # A failed token refresh quotes the metadata server's 404: that must
        # not read as "nothing at" the URL.
        problem = fw._gcs_problem(
            URL, "ERROR: (gcloud.storage.du) There was a problem refreshing "
            "your current auth tokens: ('Failed to retrieve http://metadata."
            "google.internal/computeMetadata/v1/instance/service-accounts/"
            "default/token from the Google Compute Engine metadata service. "
            "Status: 404 Response:\\nb\"No service account scopes "
            "specified.\"')")
        self.assertIsInstance(problem, PermissionError)
        self.assertIn(fw.GCS_TOKEN_ENV, str(problem))

    def test_invalid_credentials_and_anonymous_callers_are_denied(self):
        for detail in ("Your credentials are invalid. Please run\n$ gcloud "
                       "auth login",
                       "AccessDeniedException: 401 Anonymous caller does not "
                       "have storage.objects.list access"):
            with self.subTest(detail=detail):
                self.assertIsInstance(fw._gcs_problem(URL, detail),
                                      PermissionError)

    def test_missing_objects(self):
        for detail in (
                "ERROR: (gcloud.storage.du) One or more URLs matched "
                "no objects.",
                "CommandException: One or more URLs matched no objects.",
                "NotFound: 404 GET https://storage.googleapis.com/x"):
            with self.subTest(detail=detail):
                self.assertIsInstance(fw._gcs_problem(URL, detail),
                                      FileNotFoundError)

    def test_anything_else_is_reported_as_said(self):
        problem = fw._gcs_problem(URL, "ServiceUnavailable: 503 backend")
        self.assertIsInstance(problem, RuntimeError)
        self.assertIn("503 backend", str(problem))


class GcsListingTest(unittest.TestCase):
    """What preflight and the engine check read about a gs:// folder."""

    def test_listing_drops_names_that_leave_the_folder(self):
        storage = FakeStorage({"config.json": 0, "model.safetensors": 10})
        storage.sizes["../x"] = 1
        with mock.patch.object(fw, "_gcs_storage", return_value=storage):
            names = fw.gcs_listing(URL)
        self.assertEqual(set(names), {"config.json", "model.safetensors"})

    def test_listing_errors_say_what_to_do(self):
        storage = FakeStorage({})
        with mock.patch.object(fw, "_gcs_storage", return_value=storage):
            with self.assertRaises(FileNotFoundError):
                fw.gcs_listing(URL)
        storage.list = mock.Mock(side_effect=fw._GcsError("HTTP 403"))
        with mock.patch.object(fw, "_gcs_storage", return_value=storage):
            with self.assertRaisesRegex(PermissionError, fw.GCS_TOKEN_ENV):
                fw.gcs_listing(URL)

    def test_weight_bytes_follow_the_selection_fetch_gcs_copies(self):
        # Five formats and a nested copy: only model.safetensors counts.
        storage = FakeStorage(ROBERTA)
        with tempfile.TemporaryDirectory() as local, \
                mock.patch.object(fw, "_gcs_storage", return_value=storage):
            self.assertEqual(fw.gcs_weight_bytes(URL, local),
                             ROBERTA["model.safetensors"])

    def test_weight_bytes_of_a_sharded_folder_read_its_index(self):
        shards = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors"
        ]
        storage = FakeStorage({
            shards[0]: 300,
            shards[1]: 200,
            "consolidated.safetensors": 500,
            "model.safetensors.index.json": 10,
        })
        with tempfile.TemporaryDirectory() as local, \
                mock.patch.object(fw, "_gcs_storage", return_value=storage):
            with open(os.path.join(local, "model.safetensors.index.json"),
                      "w",
                      encoding="utf-8") as f:
                json.dump({"weight_map": {"a": shards[0], "b": shards[1]}}, f)
            self.assertEqual(fw.gcs_weight_bytes(URL, local), 500)


class GcsStorageChoiceTest(unittest.TestCase):

    def choose(self, tools, env=None):
        with mock.patch.object(fw.shutil, "which",
                               side_effect=lambda t: t if t in tools else None), \
                mock.patch.dict(os.environ, env or {}), \
                mock.patch.object(fw, "_LibraryStorage",
                                  side_effect=lambda token: ("library", token)):
            if not env:
                os.environ.pop(fw.GCS_TOKEN_ENV, None)
            return fw._gcs_storage()

    def test_gcloud_then_gsutil_then_the_library(self):
        self.assertEqual(self.choose({"gcloud", "gsutil"}).tool, "gcloud")
        self.assertEqual(self.choose({"gsutil"}).tool, "gsutil")
        self.assertEqual(self.choose(set()), ("library", None))

    def test_a_token_file_goes_to_gcloud_by_path_and_skips_gsutil(self):
        with tempfile.NamedTemporaryFile("w", suffix=".tok") as tok:
            tok.write("fake-secret-token\n")
            tok.flush()
            env = {fw.GCS_TOKEN_ENV: tok.name}
            gcloud = self.choose({"gcloud"}, env)
            self.assertEqual(gcloud.env[fw.GCLOUD_TOKEN_ENV], tok.name)
            self.assertNotIn("fake-secret-token", repr(gcloud.env))
            self.assertEqual(self.choose({"gsutil"}, env),
                             ("library", tok.name))

    def test_a_token_file_that_is_missing_fails_clearly(self):
        with self.assertRaises(FileNotFoundError):
            self.choose({"gcloud"}, {fw.GCS_TOKEN_ENV: "/nope/token"})


class CliStorageTest(unittest.TestCase):

    def test_du_output_is_parsed_and_folders_skipped(self):
        out = (f"615          {FOLDER}config.json\n"
               f"1115567652   {FOLDER}model.safetensors\n"
               f"0            {FOLDER}sub/\n"
               f"615          {FOLDER}sub/config.json\n"
               f"9            gs://bucket/models/other/x\n"
               f"1116183882   {FOLDER}\n")
        self.assertEqual(
            fw._parse_du(out, FOLDER), {
                "config.json": 615,
                "model.safetensors": 1115567652,
                "sub/config.json": 615,
            })

    def test_commands(self):
        calls = []

        def fake_run(argv, **kwargs):
            calls.append((argv, kwargs["env"]))
            return mock.Mock(returncode=0, stdout=f"5  {FOLDER}a.json\n")

        with mock.patch.object(fw.subprocess, "run", side_effect=fake_run):
            gcloud = fw._CliStorage("gcloud", "/run/token")
            self.assertEqual(gcloud.list(FOLDER), {"a.json": 5})
            gcloud.copy_many([FOLDER + "a.json", FOLDER + "b.json"],
                             "/d.partial")
            fw._CliStorage("gsutil", None).copy_many([FOLDER + "a.json"],
                                                     "/d.partial/sub/")
        self.assertEqual(
            calls[0][0],
            ["gcloud", "storage", "du", FOLDER, "--verbosity=error"])
        self.assertEqual(calls[0][1][fw.GCLOUD_TOKEN_ENV], "/run/token")
        self.assertEqual(calls[1][0], [
            "gcloud", "storage", "cp", FOLDER + "a.json", FOLDER + "b.json",
            "/d.partial/", "--verbosity=error"
        ])
        self.assertEqual(
            calls[2][0],
            ["gsutil", "-q", "-m", "cp", FOLDER + "a.json", "/d.partial/sub/"])

    def test_a_failed_command_raises_with_what_it_said(self):
        with mock.patch.object(fw.subprocess,
                               "run",
                               return_value=mock.Mock(returncode=1,
                                                      stdout="",
                                                      stderr="AccessDenied")):
            with self.assertRaisesRegex(fw._GcsError, "AccessDenied"):
                fw._CliStorage("gsutil", None).list(FOLDER)


class LibraryStorageTest(unittest.TestCase):

    def fake_modules(self, made, blobs, downloads):

        class Blob:

            def __init__(self, name, size=0):
                self.name, self.size = name, size

            def download_to_filename(self, dst):
                downloads.append((self.name, dst))

        class Bucket:

            def __init__(self, name):
                self.name = name

            def list_blobs(self, prefix):
                return [Blob(n, s) for n, s in blobs if n.startswith(prefix)]

            def blob(self, name):
                return Blob(name)

        storage = types.ModuleType("google.cloud.storage")
        storage.Client = lambda **kw: made.append(kw) or mock.Mock(bucket=
                                                                   Bucket)
        cloud = types.ModuleType("google.cloud")
        cloud.storage = storage
        creds = types.ModuleType("google.oauth2.credentials")
        creds.Credentials = lambda token: ("creds", token)
        return {
            "google.cloud": cloud,
            "google.cloud.storage": storage,
            "google.oauth2.credentials": creds,
        }

    def test_lists_and_copies_with_a_token_from_the_file(self):
        made, downloads = [], []
        blobs = [("models/xlm-roberta-base/config.json", 615),
                 ("models/xlm-roberta-base/", 0),
                 ("models/xlm-roberta-base-v2/config.json", 1)]
        with tempfile.NamedTemporaryFile("w") as tok, mock.patch.dict(
                sys.modules, self.fake_modules(made, blobs, downloads)):
            tok.write("fake-access-token\n")
            tok.flush()
            lib = fw._LibraryStorage(tok.name)
            self.assertEqual(lib.list(FOLDER), {"config.json": 615})
            lib.copy_many([FOLDER + "config.json"], "/d.partial")
        self.assertEqual(made[0]["credentials"],
                         ("creds", "fake-access-token"))
        self.assertEqual(downloads, [
            ("models/xlm-roberta-base/config.json", "/d.partial/config.json")
        ])

    def test_errors_carry_the_exception_name(self):

        class Forbidden(Exception):
            pass

        made, downloads = [], []
        modules = self.fake_modules(made, [], downloads)
        with mock.patch.dict(sys.modules, modules):
            lib = fw._LibraryStorage(None)
            lib.client = mock.Mock()
            lib.client.bucket.return_value.list_blobs.side_effect = Forbidden(
                "403 GET")
            with self.assertRaisesRegex(fw._GcsError, "Forbidden: 403"):
                lib.list(FOLDER)


if __name__ == "__main__":
    unittest.main()
