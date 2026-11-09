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
"""Staging a checkpoint once per bucket, and reusing only what verifies."""

import contextlib
import datetime
import hashlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import fetch_weights as fw
import stage_checkpoint as sc

REPO = "org/tiny"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
DEST = f"checkpoints/{REPO}/{COMMIT[:12]}/"


class PreconditionFailed(Exception):
    """What google-cloud-storage raises when if_generation_match fails."""
    code = 412


def crc(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()[:8]  # stands in for crc32c


class FakeBlob:
    """An object handle: properties are read from, and written to, the
    bucket's store, as with the real client."""

    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name
        self.metadata = None
        self._load()

    def _load(self):
        stored = self.bucket.store.get(self.name)
        self.size = len(stored["data"]) if stored else None
        self.crc32c = stored["crc32c"] if stored else None
        self.generation = stored["generation"] if stored else None
        self.time_created = stored["created"] if stored else None
        if stored and self.metadata is None:
            self.metadata = dict(stored["metadata"] or {})

    def _put(self, data, if_generation_match=None):
        current = self.bucket.store.get(self.name)
        if if_generation_match is not None:
            if (current["generation"]
                    if current else 0) != if_generation_match:
                raise PreconditionFailed(self.name)
        self.bucket.generation += 1
        self.bucket.store[self.name] = {
            "data": data,
            "crc32c": crc(data),
            "generation": self.bucket.generation,
            "created": self.bucket.now(),
            "metadata": dict(self.metadata or {}),
        }
        self.bucket.writes.append(self.name)
        self._load()

    def upload_from_filename(self, path, if_generation_match=None):
        with open(path, "rb") as f:
            self._put(f.read(), if_generation_match)

    def upload_from_string(self,
                           data,
                           content_type=None,
                           if_generation_match=None):
        del content_type
        self._put(data.encode() if isinstance(data, str) else data,
                  if_generation_match)

    def download_as_bytes(self):
        return self.bucket.store[self.name]["data"]

    def delete(self, if_generation_match=None):
        current = self.bucket.store.get(self.name)
        if current is None:
            raise type("NotFound", (Exception, ), {"code": 404})(self.name)
        if (if_generation_match is not None
                and current["generation"] != if_generation_match):
            raise PreconditionFailed(self.name)
        del self.bucket.store[self.name]


class FakeBucket:

    def __init__(self, name="bucket"):
        self.name = name
        self.store = {}
        self.generation = 0
        self.writes = []
        # The code under test ages a lock against the real clock, as GCS
        # stamps objects with real time.
        self.clock = datetime.datetime.now(datetime.timezone.utc)

    def now(self):
        return self.clock

    def blob(self, name):
        return FakeBlob(self, name)

    def get_blob(self, name):
        return FakeBlob(self, name) if name in self.store else None

    def list_blobs(self, prefix=""):
        return [
            FakeBlob(self, n) for n in sorted(self.store)
            if n.startswith(prefix)
        ]

    def put(self, name, data, metadata=None):
        blob = self.blob(name)
        blob.metadata = metadata
        blob.upload_from_string(data)
        return blob


CONTENT = {
    "config.json": json.dumps({
        "num_hidden_layers": 2
    }).encode(),
    "tokenizer.json": b"{\"vocab\": {}}",
    "model.safetensors": b"W" * 4096,
}


def digest_of(name, data):
    if name.endswith(".safetensors"):
        return "sha256:" + hashlib.sha256(data).hexdigest()
    header = f"blob {len(data)}\0".encode()
    return "git-sha1:" + hashlib.sha1(header + data).hexdigest()


def make_plan(content=None, commit=COMMIT, repo=REPO):
    content = content or CONTENT
    return sc.Plan(repo=repo,
                   commit=commit,
                   files={
                       n: sc.FileSpec(len(d), digest_of(n, d))
                       for n, d in content.items()
                   },
                   config=content.get("config.json"))


class Hub:
    """Stands in for the hub: a plan, and downloads that count themselves."""

    def __init__(self, content=None, served=None):
        self.content = dict(content or CONTENT)
        # What a download returns, when it should differ from the plan.
        self.served = dict(served or self.content)
        self.fetched = []

    def plan(self, model, revision=None):
        del revision
        return make_plan(self.content, repo=model)

    def fetch(self, model, commit, name, folder):
        del model, commit
        self.fetched.append(name)
        path = os.path.join(folder, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.served[name])
        return path


def staged(bucket, prefix=DEST, content=None, commit=COMMIT, repo=REPO):
    """Puts a finished, verified copy at `prefix`, as stage() would."""
    content = content or CONTENT
    files = {}
    for name, data in content.items():
        blob = bucket.put(prefix + name, data,
                          {sc.DIGEST_KEY: digest_of(name, data)})
        files[name] = {
            "size": len(data),
            "digest": digest_of(name, data),
            "crc32c": blob.crc32c,
        }
    bucket.put(prefix + sc.MANIFEST,
               json.dumps({
                   "repo": repo,
                   "revision": commit,
                   "files": files
               }))


class Watching(Hub):
    """Records, at each download, whether a manifest is up at DEST."""

    def __init__(self, bucket, **kwargs):
        super().__init__(**kwargs)
        self.bucket = bucket
        self.manifest_seen = []

    def fetch(self, model, commit, name, folder):
        self.manifest_seen.append(DEST + sc.MANIFEST in self.bucket.store)
        return super().fetch(model, commit, name, folder)


def no_wait(sec):
    raise AssertionError(
        f"stage() waited {sec}s for a lock: a lock was left behind")


def run(bucket, hub=None, sleep=no_wait, **kwargs):
    """stage() with a fake hub. A wait fails at once unless the test expects
    one, so a lock left held fails the test rather than hanging it."""
    hub = hub or Hub()
    with contextlib.redirect_stderr(io.StringIO()) as err, \
            mock.patch.object(sc.time, "sleep", side_effect=sleep):
        out = sc.stage(bucket, REPO, plan=hub.plan, fetch=hub.fetch, **kwargs)
    return out, hub, err.getvalue()


class StageTest(unittest.TestCase):

    def test_stages_each_file_with_its_digest_and_the_manifest_last(self):
        bucket = FakeBucket()
        out, hub, _ = run(bucket)
        self.assertEqual(out["checkpoint"], f"gs://bucket/{DEST}")
        self.assertFalse(out["reused"])
        self.assertTrue(out["verified"])
        self.assertEqual(out["uploaded"]["files"], 3)
        self.assertEqual(sorted(hub.fetched), sorted(CONTENT))
        for name, data in CONTENT.items():
            stored = bucket.store[DEST + name]
            self.assertEqual(stored["data"], data)
            self.assertEqual(stored["metadata"][sc.DIGEST_KEY],
                             digest_of(name, data))
        # The manifest is what says the copy is whole: written after every
        # file, and the lock is gone.
        data_writes = [w for w in bucket.writes if not w.endswith(".lock")]
        self.assertEqual(data_writes[-1], DEST + sc.MANIFEST)
        self.assertNotIn(DEST.rstrip("/") + ".lock", bucket.store)

    def test_a_staged_copy_is_reused_without_asking_the_hub_for_files(self):
        bucket = FakeBucket()
        run(bucket)
        out, hub, _ = run(bucket)
        self.assertTrue(out["reused"])
        self.assertTrue(out["verified"])
        self.assertEqual(hub.fetched, [])
        self.assertEqual(out["uploaded"], {"files": 0, "bytes": 0})

    def test_a_copy_staged_elsewhere_in_the_folder_is_reused_if_it_verifies(
            self):
        bucket = FakeBucket()
        staged(bucket, prefix="checkpoints/older-layout/tiny/")
        out, hub, _ = run(bucket)
        self.assertEqual(out["checkpoint"],
                         "gs://bucket/checkpoints/older-layout/tiny/")
        self.assertTrue(out["verified"])
        self.assertEqual(hub.fetched, [])

    def test_an_interrupted_upload_is_not_reused_and_resumes(self):
        # The first run dies after two of the three files.
        bucket = FakeBucket()

        class Dies(Hub):

            def fetch(self, model, commit, name, folder):
                if name == "tokenizer.json":
                    raise sc.StageError("connection reset")
                return super().fetch(model, commit, name, folder)

        with self.assertRaises(sc.StageError):
            run(bucket, hub=Dies())
        self.assertNotIn(DEST + sc.MANIFEST, bucket.store)
        self.assertNotIn(DEST.rstrip("/") + ".lock", bucket.store)

        out, hub, _ = run(bucket)
        self.assertFalse(out["reused"])
        # config.json and model.safetensors came first (sorted) and were
        # recorded as checked: only the file that failed is fetched again.
        self.assertEqual(hub.fetched, ["tokenizer.json"])
        self.assertEqual(out["uploaded"]["files"], 1)
        self.assertIn(DEST + sc.MANIFEST, bucket.store)
        self.assertNotIn(DEST.rstrip("/") + ".progress.json", bucket.store)
        self.assertIn("no manifest", out["candidates"][0]["not_used"])

    def test_a_file_whose_metadata_claims_the_right_digest_is_not_trusted(
            self):
        # Same name and size, and a digest in its metadata, but this tool
        # never recorded it: metadata can be set by anyone.
        bucket = FakeBucket()
        bucket.put(DEST + "model.safetensors",
                   b"X" * 4096,
                   metadata={
                       sc.DIGEST_KEY:
                       digest_of("model.safetensors",
                                 CONTENT["model.safetensors"])
                   })
        _, hub, _ = run(bucket)
        self.assertIn("model.safetensors", hub.fetched)
        self.assertEqual(bucket.store[DEST + "model.safetensors"]["data"],
                         CONTENT["model.safetensors"])

    def test_same_names_and_sizes_without_a_manifest_need_trust(self):
        # A copy someone put in the folder by hand: by names, sizes and
        # config.json it matches, but nothing says it is these weights.
        bucket = FakeBucket()
        for name, data in CONTENT.items():
            bucket.put("checkpoints/by-hand/tiny/" + name, data)
        out, hub, _ = run(bucket)
        self.assertFalse(out["reused"])
        self.assertTrue(hub.fetched)
        self.assertIn("--trust-unverified", out["candidates"][0]["not_used"])

        bucket = FakeBucket()
        for name, data in CONTENT.items():
            bucket.put("checkpoints/by-hand/tiny/" + name, data)
        out, hub, _ = run(bucket, trust_unverified=True)
        self.assertTrue(out["reused"])
        self.assertFalse(out["verified"])
        self.assertEqual(out["checkpoint"],
                         "gs://bucket/checkpoints/by-hand/tiny/")
        self.assertEqual(hub.fetched, [])

    def test_a_different_config_is_never_trusted(self):
        bucket = FakeBucket()
        for name, data in CONTENT.items():
            if name == "config.json":
                data = json.dumps({"num_hidden_layers": 3}).encode()
                data = data[:len(CONTENT[name])].ljust(len(CONTENT[name]))
            bucket.put("checkpoints/by-hand/tiny/" + name, data)
        out, _, _ = run(bucket, trust_unverified=True)
        self.assertFalse(out["reused"])
        self.assertIn("config.json differs", out["candidates"][0]["not_used"])

    def test_a_staged_file_replaced_since_does_not_verify(self):
        # Same size, different bytes: the crc32c recorded at upload catches
        # it, and the file is staged again from the hub.
        bucket = FakeBucket()
        run(bucket)
        bucket.put(DEST + "model.safetensors",
                   b"Z" * 4096,
                   metadata={
                       sc.DIGEST_KEY:
                       digest_of("model.safetensors",
                                 CONTENT["model.safetensors"])
                   })
        out, hub, _ = run(bucket, hub=Watching(bucket))
        self.assertFalse(out["reused"])
        self.assertIn("changed since it was staged",
                      out["candidates"][0]["not_used"])
        # While it is staged again, no manifest says the folder is finished.
        self.assertEqual(hub.manifest_seen, [False])
        # Its metadata still claims the right digest, but its crc32c is not
        # the one recorded at upload: it is fetched and staged again, and
        # only that file.
        self.assertEqual(hub.fetched, ["model.safetensors"])
        self.assertEqual(bucket.store[DEST + "model.safetensors"]["data"],
                         CONTENT["model.safetensors"])
        out, _, _ = run(bucket)
        self.assertTrue(out["reused"] and out["verified"])

    def test_same_sizes_but_a_different_hub_digest_does_not_verify(self):
        # A new revision whose weights kept their size.
        bucket = FakeBucket()
        staged(bucket, prefix="checkpoints/elsewhere/")
        new = dict(CONTENT, **{"model.safetensors": b"V" * 4096})
        out, hub, _ = run(bucket, hub=Hub(new))
        self.assertFalse(out["reused"])
        self.assertIn("differs from the hub's",
                      out["candidates"][0]["not_used"])
        self.assertIn("model.safetensors", hub.fetched)

    def test_a_manifest_for_another_revision_does_not_verify(self):
        bucket = FakeBucket()
        staged(bucket, prefix="checkpoints/elsewhere/", commit="f" * 40)
        out, _, _ = run(bucket)
        self.assertFalse(out["reused"])
        self.assertIn("revision ffffffffffff",
                      out["candidates"][0]["not_used"])

    def test_a_download_that_does_not_match_the_hub_is_never_staged(self):
        bucket = FakeBucket()
        hub = Hub(served=dict(CONTENT, **{"model.safetensors": b"T" * 4096}))
        with self.assertRaisesRegex(sc.StageError,
                                    "does not match the hub's own digest"):
            run(bucket, hub=hub)
        self.assertNotIn(DEST + "model.safetensors", bucket.store)
        self.assertNotIn(DEST + sc.MANIFEST, bucket.store)
        # The lock is released on failure too.
        self.assertNotIn(DEST.rstrip("/") + ".lock", bucket.store)

    def test_only_the_checkpoints_folder_is_searched(self):
        bucket = FakeBucket()
        staged(bucket, prefix="somewhere-else/tiny/")
        out, hub, _ = run(bucket)
        self.assertFalse(out["reused"])
        self.assertTrue(hub.fetched)
        self.assertEqual(out["candidates"], [])

    def test_a_custom_folder(self):
        bucket = FakeBucket()
        out, _, _ = run(bucket, folder="team/models/")
        self.assertEqual(out["checkpoint"],
                         f"gs://bucket/team/models/{REPO}/{COMMIT[:12]}/")

    def test_a_model_with_no_loadable_files_is_refused(self):

        class Empty(Hub):

            def plan(self, model, revision=None):
                return sc.Plan(model, COMMIT, {}, None)

        with self.assertRaisesRegex(sc.StageError, "no files"):
            run(FakeBucket(), hub=Empty())


class LockTest(unittest.TestCase):
    LOCK = DEST.rstrip("/") + ".lock"

    def test_a_second_session_waits_then_reuses_the_first_ones_copy(self):
        bucket = FakeBucket()
        bucket.put(self.LOCK, "{}")  # the first session, mid-upload
        sleeps = []

        def first_session_finishes(sec):
            sleeps.append(sec)
            staged(bucket)
            bucket.blob(self.LOCK).delete()

        out, hub, err = run(bucket, poll_sec=7, sleep=first_session_finishes)
        self.assertEqual(sleeps, [7])
        self.assertTrue(out["reused"])
        self.assertEqual(hub.fetched, [])
        self.assertIn("STAGE WAIT", err)

    def test_a_stale_lock_is_broken(self):
        bucket = FakeBucket()
        bucket.put(self.LOCK, "{}")
        bucket.store[self.LOCK]["created"] -= datetime.timedelta(hours=13)
        out, hub, err = run(bucket)
        self.assertFalse(out["reused"])
        self.assertTrue(hub.fetched)
        self.assertIn("breaking a lock held for 13.0 h", err)
        self.assertNotIn(self.LOCK, bucket.store)

    def test_a_fresh_lock_is_not_broken(self):
        bucket = FakeBucket()
        bucket.put(self.LOCK, "{}")
        bucket.store[self.LOCK]["created"] -= datetime.timedelta(hours=11)
        with mock.patch.object(sc.time, "monotonic", side_effect=[0, 0,
                                                                  10**6]):
            with self.assertRaises(sc.StageError):
                run(bucket, wait_sec=60, sleep=lambda sec: None)
        self.assertIn(self.LOCK, bucket.store)

    def test_waiting_gives_up_after_the_limit(self):
        bucket = FakeBucket()
        bucket.put(self.LOCK, "{}")
        with mock.patch.object(sc.time, "monotonic", side_effect=[0, 0,
                                                                  10**6]):
            with self.assertRaisesRegex(sc.StageError, "gave up after"):
                run(bucket, wait_sec=60, sleep=lambda sec: None)
        self.assertIn(self.LOCK, bucket.store)  # not ours to delete

    def test_release_never_deletes_a_lock_another_session_took_over(self):
        bucket = FakeBucket()
        lock = sc._Lock(bucket, self.LOCK)
        self.assertTrue(lock.try_acquire())
        # Ours went stale and was taken over.
        bucket.blob(self.LOCK).delete()
        bucket.put(self.LOCK, "{\"other\": true}")
        lock.release()
        self.assertEqual(bucket.store[self.LOCK]["data"], b"{\"other\": true}")

    def test_only_one_session_can_hold_the_lock(self):
        bucket = FakeBucket()
        first, second = sc._Lock(bucket,
                                 self.LOCK), sc._Lock(bucket, self.LOCK)
        self.assertTrue(first.try_acquire())
        self.assertFalse(second.try_acquire())
        first.release()
        self.assertTrue(second.try_acquire())


class DigestTest(unittest.TestCase):

    def test_git_blob_id_and_sha256_match_what_the_hub_reports(self):
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            f.write(b"hello\n")
        try:
            # `git hash-object` of "hello\n".
            self.assertEqual(
                sc.file_digest(f.name, "git-sha1"),
                "git-sha1:ce013625030ba8dba906f756967f9e9ca394464a")
            self.assertEqual(
                sc.file_digest(f.name, "sha256"),
                "sha256:5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e"
                "846f6be03")
        finally:
            os.remove(f.name)


def fake_hub_module(siblings, index=None, config=b"{}", error=None):
    hub = types.ModuleType("huggingface_hub")
    info = types.SimpleNamespace(sha=COMMIT, siblings=siblings)
    api = mock.Mock()
    if error:
        api.model_info.side_effect = error
    else:
        api.model_info.return_value = info
    hub.HfApi = mock.Mock(return_value=api)
    tmp = tempfile.mkdtemp()

    def download(model, name, revision=None, local_dir=None):
        del model, revision, local_dir
        path = os.path.join(tmp, name.replace("/", "_"))
        with open(path, "wb") as f:
            f.write(
                json.dumps({
                    "weight_map": index
                }).encode() if name.endswith("index.json") else config)
        return path

    hub.hf_hub_download = download
    return hub


def sib(name, size, sha256=None, blob_id="b" * 40):
    lfs = types.SimpleNamespace(sha256=sha256) if sha256 else None
    return types.SimpleNamespace(rfilename=name,
                                 size=size,
                                 lfs=lfs,
                                 blob_id=blob_id)


class HubPlanTest(unittest.TestCase):

    def test_digests_selection_and_what_a_server_never_loads(self):
        shards = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors"
        ]
        siblings = [
            sib(shards[0], 10, "a" * 64),
            sib(shards[1], 10, "c" * 64),
            sib("model.safetensors.index.json", 50),
            sib("consolidated.safetensors", 20, "d" * 64),
            sib("config.json", 5, blob_id="e" * 40),
            sib("README.md", 5),
            sib("LICENSE", 5),
            sib(".gitattributes", 5),
        ]
        hub = fake_hub_module(siblings,
                              index={
                                  "a": shards[0],
                                  "b": shards[1]
                              },
                              config=b"{\"num_hidden_layers\": 4}")
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            plan = sc.hub_plan(REPO)
        self.assertEqual(plan.commit, COMMIT)
        self.assertEqual(
            sorted(plan.files),
            sorted(shards + ["model.safetensors.index.json", "config.json"]))
        self.assertEqual(plan.files[shards[0]].digest, "sha256:" + "a" * 64)
        self.assertEqual(plan.files["config.json"].digest,
                         "git-sha1:" + "e" * 40)
        self.assertEqual(plan.config, b"{\"num_hidden_layers\": 4}")

    def test_a_gated_model_says_what_to_do_and_never_shows_the_token(self):
        gated = type(
            "GatedRepoError", (Exception, ),
            {})("401 Client Error. Access to model org/tiny is restricted.")
        hub = fake_hub_module([], error=gated)
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}), \
                mock.patch.dict(os.environ, {"HF_TOKEN": "hf_secretvalue"}):
            with self.assertRaises(sc.StageError) as ctx:
                sc.hub_plan(REPO)
        self.assertIn("gated or private", str(ctx.exception))
        self.assertIn("HF_TOKEN", str(ctx.exception))
        self.assertNotIn("hf_secretvalue", str(ctx.exception))

    def test_a_missing_repo_is_named(self):
        missing = type("RepositoryNotFoundError", (Exception, ),
                       {})("404 Repository Not Found for url")
        hub = fake_hub_module([], error=missing)
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            with self.assertRaisesRegex(sc.StageError,
                                        "RepositoryNotFoundError"):
                sc.hub_plan("org/none")


class MainTest(unittest.TestCase):

    def test_writes_the_result_and_says_ready(self):
        bucket, hub = FakeBucket(), Hub()
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(sc, "_client_bucket", return_value=bucket), \
                mock.patch.object(sc, "hub_plan", side_effect=hub.plan), \
                mock.patch.object(sc, "hub_fetch", side_effect=hub.fetch):
            # stage() binds the defaults at definition: pass them through.
            real_stage = sc.stage

            def stage(*a, **k):
                return real_stage(*a, plan=hub.plan, fetch=hub.fetch, **k)

            with mock.patch.object(sc, "stage", side_effect=stage), \
                    contextlib.redirect_stdout(io.StringIO()) as out, \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                code = sc.main(
                    [REPO, "--bucket", "gs://bucket", "--out", d + "/r.json"])
            with open(d + "/r.json", encoding="utf-8") as f:
                written = json.load(f)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue()), written)
        self.assertEqual(written["checkpoint"], f"gs://bucket/{DEST}")
        self.assertIn("STAGE READY", err.getvalue())

    def test_a_failure_exits_non_zero_with_a_reason(self):
        with mock.patch.object(sc, "_client_bucket",
                               return_value=FakeBucket()), \
                mock.patch.object(sc, "stage",
                                  side_effect=sc.StageError("gated")), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(sc.main([REPO, "--bucket", "gs://bucket"]), 1)
        self.assertIn("STAGE FAILED: gated", err.getvalue())

    def test_a_storage_error_is_explained(self):
        denied = type("Forbidden", (Exception, ),
                      {})("403 POST: does not have storage.objects.create")
        with mock.patch.object(sc, "_client_bucket",
                               return_value=FakeBucket()), \
                mock.patch.object(sc, "stage", side_effect=denied), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(sc.main([REPO, "--bucket", "gs://bucket"]), 1)
        self.assertIn("no permission", err.getvalue())


class BucketStorage:
    """fetch_weights' storage interface over a FakeBucket, so a job can
    read what stage() wrote."""

    def __init__(self, bucket):
        self.bucket = bucket

    def list(self, folder_url):
        prefix = folder_url[len(f"gs://{self.bucket.name}/"):]
        return {
            n[len(prefix):]: len(v["data"])
            for n, v in self.bucket.store.items() if n.startswith(prefix)
        }

    def copy_many(self, srcs, folder):
        for src in srcs:
            name = src[len(f"gs://{self.bucket.name}/"):]
            with open(os.path.join(folder, os.path.basename(src)), "wb") as f:
                f.write(self.bucket.store[name]["data"])


class StagedThenFetchedTest(unittest.TestCase):

    def test_a_job_copies_what_was_staged(self):
        bucket = FakeBucket()
        out, _, _ = run(bucket)
        with tempfile.TemporaryDirectory() as cache, \
                mock.patch.object(fw, "_gcs_storage",
                                  return_value=BucketStorage(bucket)), \
                mock.patch.dict(os.environ, {fw.CHECKPOINT_DIR_ENV: cache}), \
                contextlib.redirect_stdout(io.StringIO()):
            path = fw.fetch_gcs(out["checkpoint"], interval=0.01)
            for name, data in CONTENT.items():
                with open(os.path.join(path, name), "rb") as f:
                    self.assertEqual(f.read(), data)


class LocalStageTest(unittest.TestCase):

    def _make_local_model(self, root):
        os.makedirs(os.path.join(root, ".cache"), exist_ok=True)
        files = {
            "config.json": json.dumps({
                "num_hidden_layers": 2
            }).encode(),
            "tokenizer.json": b'{"vocab": {}}',
            "model.safetensors": b"W" * 4096,
            "README.md": b"Not loaded",
            ".cache/ignored.bin": b"Ignored cache file",
        }
        for rel, data in files.items():
            p = os.path.join(root, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(data)
        return files

    def test_stages_local_directory_and_preserves_source_files(self):
        bucket = FakeBucket()
        with tempfile.TemporaryDirectory() as d:
            model_dir = os.path.join(d, "my-local-model")
            os.makedirs(model_dir)
            files = self._make_local_model(model_dir)
            out_file = os.path.join(d, "out.json")

            with mock.patch.object(sc, "_client_bucket", return_value=bucket), \
                    contextlib.redirect_stdout(io.StringIO()) as out, \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                code = sc.main(
                    [model_dir, "--bucket", "gs://bucket", "--out", out_file])

            self.assertEqual(code, 0)
            with open(out_file, encoding="utf-8") as f:
                res = json.load(f)
            self.assertEqual(json.loads(out.getvalue()), res)
            self.assertFalse(res["reused"])
            self.assertTrue(res["verified"])
            self.assertEqual(res["uploaded"]["files"], 3)
            self.assertEqual(res["repo"], "local/my-local-model")
            expected_prefix = f"gs://bucket/checkpoints/local/my-local-model/{res['revision'][:12]}/"
            self.assertEqual(res["checkpoint"], expected_prefix)
            self.assertIn("STAGE READY", err.getvalue())

            # Source files must not be deleted after upload.
            for rel in files:
                self.assertTrue(os.path.isfile(os.path.join(model_dir, rel)))

            # README.md and .cache files must not be uploaded.
            dest = f"checkpoints/local/my-local-model/{res['revision'][:12]}/"
            self.assertNotIn(dest + "README.md", bucket.store)
            self.assertNotIn(dest + ".cache/ignored.bin", bucket.store)
            self.assertIn(dest + sc.MANIFEST, bucket.store)

            # A job can fetch the staged files back.
            with tempfile.TemporaryDirectory() as cache, \
                    mock.patch.object(fw, "_gcs_storage", return_value=BucketStorage(bucket)), \
                    mock.patch.dict(os.environ, {fw.CHECKPOINT_DIR_ENV: cache}), \
                    contextlib.redirect_stdout(io.StringIO()):
                path = fw.fetch_gcs(res["checkpoint"], interval=0.01)
                for name in ("config.json", "tokenizer.json",
                             "model.safetensors"):
                    with open(os.path.join(path, name), "rb") as f:
                        self.assertEqual(f.read(), files[name])

    def test_reuses_verified_local_directory_on_second_run(self):
        bucket = FakeBucket()
        with tempfile.TemporaryDirectory() as d:
            model_dir = os.path.join(d, "my-local-model")
            os.makedirs(model_dir)
            self._make_local_model(model_dir)

            with mock.patch.object(sc, "_client_bucket", return_value=bucket), \
                    contextlib.redirect_stdout(io.StringIO()):
                code1 = sc.main([model_dir, "--bucket", "gs://bucket"])
                self.assertEqual(code1, 0)

            with mock.patch.object(sc, "_client_bucket", return_value=bucket), \
                    contextlib.redirect_stdout(io.StringIO()) as out, \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                code2 = sc.main([model_dir, "--bucket", "gs://bucket"])

            self.assertEqual(code2, 0)
            res = json.loads(out.getvalue())
            self.assertTrue(res["reused"])
            self.assertTrue(res["verified"])
            self.assertEqual(res["uploaded"], {"files": 0, "bytes": 0})
            self.assertIn("STAGE READY", err.getvalue())

    def test_modified_local_file_produces_new_revision_and_stages_fresh_copy(
            self):
        bucket = FakeBucket()
        with tempfile.TemporaryDirectory() as d:
            model_dir = os.path.join(d, "my-local-model")
            os.makedirs(model_dir)
            self._make_local_model(model_dir)

            with mock.patch.object(sc, "_client_bucket", return_value=bucket), \
                    contextlib.redirect_stdout(io.StringIO()) as out1:
                sc.main([model_dir, "--bucket", "gs://bucket"])
            res1 = json.loads(out1.getvalue())

            # Modify model.safetensors.
            with open(os.path.join(model_dir, "model.safetensors"), "wb") as f:
                f.write(b"Z" * 8192)

            with mock.patch.object(sc, "_client_bucket", return_value=bucket), \
                    contextlib.redirect_stdout(io.StringIO()) as out2:
                sc.main([model_dir, "--bucket", "gs://bucket"])
            res2 = json.loads(out2.getvalue())

            self.assertFalse(res2["reused"])
            self.assertNotEqual(res1["revision"], res2["revision"])
            self.assertNotEqual(res1["checkpoint"], res2["checkpoint"])

    def test_local_directory_with_no_loadable_files_is_refused(self):
        bucket = FakeBucket()
        with tempfile.TemporaryDirectory() as d:
            model_dir = os.path.join(d, "empty-model")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "README.md"), "wb") as f:
                f.write(b"# Just a readme")

            with mock.patch.object(sc, "_client_bucket", return_value=bucket), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                code = sc.main([model_dir, "--bucket", "gs://bucket"])

            self.assertEqual(code, 1)
            self.assertIn("STAGE FAILED: no files", err.getvalue())


if __name__ == "__main__":
    unittest.main()
