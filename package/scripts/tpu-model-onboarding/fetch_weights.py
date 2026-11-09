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
"""Download a Hugging Face model's weights before the engine loads them.

Runs on the target, inside the serving job, before the engine starts. It
fixes two things the engines' own downloads get wrong:

-   **What is downloaded.** Many repos carry the same weights more than once.
    `openai/gpt-oss-20b` has its 13.8 GB of Hugging Face shards, the same
    weights again under `original/` for OpenAI's reference package, and again
    under `metal/`; Mistral repos add a `consolidated.safetensors` beside the
    shards. An allow pattern such as `*.safetensors` matches across `/`, so it
    fetches `original/model.safetensors` too. This script downloads only the
    shards the repo's own index lists, which is what transformers and vLLM
    load, plus the small config and tokenizer files.
-   **What the log shows.** The library's progress bar counts files, not
    bytes, and redraws one line in place, so a multi-GB shard looks like a
    hang. This prints a plain `WEIGHTS PROGRESS` line every interval with
    bytes, rate and ETA, then `WEIGHTS READY` once every file is in the
    cache. `compute job wait --until "log:WEIGHTS READY"` waits on that line,
    so a download is never mistaken for a compile.

Before downloading, it checks that what is still missing fits where the cache
is, and fails at once with `WEIGHTS FAILED` if not, rather than part-way
through a download the pod is then evicted for.

Files already in the hub cache (`HF_HUB_CACHE`) are not downloaded again;
that is the library's own behaviour. On gke_job with `hf_cache_pvc` set, that
cache is a volume every pod shares, so a model downloads once. Without it,
each pod starts empty and downloads again; this script makes that smaller and
visible, it does not share it.

Load the model from the snapshot path this returns, not by its repo id, and
run the engine with `HF_HUB_OFFLINE=1`. Offline, a lookup by repo id whose
patterns match a file this skipped (vLLM asking for `*.safetensors` matches
`original/model.safetensors`) fails with `IncompleteSnapshotError`; a local
directory is read as it is.

A `gs://` checkpoint folder -- for example one stage_checkpoint.py staged in
the session's bucket -- is copied to local disk instead, and the hub is
not asked. The same selection applies -- the index's shards plus the small
config and tokenizer files, not the folder's other formats or nested copies
-- and so does the space check, before any weights are copied. The copy goes
under `CHECKPOINT_DIR` when that is set, otherwise beside the hub cache
(`<HF_HUB_CACHE>/gcs/`), so it lands on the volume a hub download would: the
shared `hf_cache_pvc`, or /dev/shm on gke_job, never the pod's own disk. One
folder per URL, and a lock around the copy, so Jobs sharing that volume
neither collide nor read a file another is still writing. A file is renamed
into place only once it is whole, and a file already there at its full size
is not copied again.

The copy uses `gcloud storage`, else `gsutil`, else the google-cloud-storage
library, whichever the job has, with the job's own credentials (the TPU VM's
service account, or Workload Identity on gke_job). A pod with no identity for
the bucket can be given an access token in a file named by `GCS_TOKEN_FILE`;
the token is read, never printed.

Usage, as a script:  python3 fetch_weights.py <model_id | gs://bucket/dir>
                         [--path-file FILE]
Usage, as a module:  path = fetch_weights.fetch(model_id)  # local snapshot
                     path = fetch_weights.fetch_gcs(url, small_only=True)
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import errno
import fcntl
import glob
import hashlib
import json
import math
import os
import posixpath
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator

# The library's bars redraw one line with \r: in a pod log that is either
# noise or nothing. Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

READY = "WEIGHTS READY"
PROGRESS = "WEIGHTS PROGRESS"
FAILED = "WEIGHTS FAILED"
DEFAULT_INTERVAL_SEC = 30

WEIGHT_EXTS = (".safetensors", ".bin", ".pth", ".pt", ".ckpt", ".h5",
               ".msgpack", ".ot", ".gguf", ".onnx", ".tflite", ".mlmodel")
# Formats no PyTorch engine here loads (TF, Flax, Rust, llama.cpp, ONNX...).
NON_TORCH_EXTS = (".h5", ".msgpack", ".ot", ".gguf", ".onnx", ".tflite",
                  ".mlmodel")
# Whole-model copies kept for other runtimes, never read by transformers.
DUPLICATE_DIRS = ("original", "metal")
# A non-weight file larger than this is a sample image, a video or an
# archive, not something a loader reads.
MAX_AUX_BYTES = 100 * 1024 * 1024

# The layouts transformers loads, in the order it prefers them: a sharded
# checkpoint names its shards in the index; a single file is the whole model.
PRIMARY_LAYOUTS = (
    ("model.safetensors.index.json", "model.safetensors"),
    ("pytorch_model.bin.index.json", "pytorch_model.bin"),
)

GB = 1e9
GiB = 1024**3

# The memory-backed /dev/shm gke_job mounts, and the env var naming its cap
# in bytes. Inside the pod, statvfs on it can report the node's whole memory
# rather than that cap, and the kubelet evicts the pod past the cap.
SHM_MOUNT = "/dev/shm"
SHM_LIMIT_ENV = "TPU_AGENT_SUITE_SHM_LIMIT_BYTES"
# Without hf_cache_pvc, the engine's compile caches land on /dev/shm beside
# the weights, after they are in, so room is kept for them. They grow with
# the model's layers, not its weights, nor its TP x DP layout: measured on
# vLLM 0.29, 0.16-0.18 GB per layer (Gemma-4-12B, 48 layers: 7.9 GB;
# Gemma-4-31B, 60 layers: 9.7 GB at TP2 x DP4 and 10.5 GB at TP8 x DP1).
# This keeps 1.5-1.9x that: a reserve too small fails a model that would
# have fit, while one too large only asks for a little more cap than was
# used. A model whose layer count cannot be read gets
# COMPILE_RESERVE_UNKNOWN.
COMPILE_RESERVE_BASE = 2 * 1024**3
COMPILE_RESERVE_PER_LAYER = 1024**3 // 4
COMPILE_RESERVE_UNKNOWN = 16 * 1024**3
# Where configs name the layer count: top level, or under the text model's
# own config for a multimodal one.
LAYER_KEYS = ("num_hidden_layers", "num_layers", "n_layer")

GCS_SCHEME = "gs://"
# Where gs:// checkpoints are copied when set; otherwise beside the hub cache.
CHECKPOINT_DIR_ENV = "CHECKPOINT_DIR"
# A file holding an access token, for a pod with no identity of its own for
# the bucket. Read, never printed.
GCS_TOKEN_ENV = "GCS_TOKEN_FILE"
# gcloud reads an access token from the file this property names.
GCLOUD_TOKEN_ENV = "CLOUDSDK_AUTH_ACCESS_TOKEN_FILE"
# Files are copied into `<dest>.partial/` and renamed into `<dest>` once
# whole, so a reader of `<dest>` never sees a partial copy.
PARTIAL_SUFFIX = ".partial"
# Copied before anything else: they size the compile reserve and name the
# shards, and are a few kB.
METADATA_FILES = ("config.json", ) + tuple(index
                                           for index, _ in PRIMARY_LAYOUTS)
# Substrings of a storage error that mean "not allowed" or "not there", from
# gcloud, gsutil and the client library. Denied is checked first: a failed
# credential refresh can quote a metadata server's 404.
GCS_DENIED = ("403", "401", "accessdenied", "forbidden", "permission",
              "does not have storage.", "unauthorized", "unauthenticated",
              "anonymous caller", "credentials", "auth tokens",
              "reauthentication")
GCS_MISSING = ("404", "notfound", "not found", "matched no objects",
               "no urls matched")


@dataclasses.dataclass
class Space:
    """Room under a cache path: `free` bytes; on /dev/shm, also its cap and
    what the volume already holds, else both None."""
    free: int
    shm_limit: int | None = None
    shm_used: int | None = None


@dataclasses.dataclass
class Selection:
    files: list[str]
    total_bytes: int
    skipped: list[str]
    skipped_bytes: int
    layout: str


def _ext(name: str) -> str:
    return posixpath.splitext(name)[1].lower()


def _top_dir(name: str) -> str:
    return name.split("/", 1)[0] if "/" in name else ""


def _fallback_weights(weights: list[str]) -> list[str]:
    """Every weight file a PyTorch loader might read, minus known copies."""
    kept = [
        w for w in weights
        if _top_dir(w) not in DUPLICATE_DIRS and _ext(w) not in NON_TORCH_EXTS
    ]
    # Where a folder ships safetensors, its .bin/.pth are the same weights in
    # the older format.
    has_safetensors = {
        posixpath.dirname(w)
        for w in kept if _ext(w) == ".safetensors"
    }
    kept = [
        w for w in kept if _ext(w) == ".safetensors"
        or posixpath.dirname(w) not in has_safetensors
    ]
    if kept:
        return kept
    # Never select nothing: an unfamiliar layout downloads in full rather
    # than starting an engine that has no weights.
    return [w for w in weights if _ext(w) not in NON_TORCH_EXTS] or weights


def select_files(sizes: dict[str, int],
                 read_index: Callable[[str], list[str]]) -> Selection:
    """Choose which repo files to download.

    `sizes` maps every file in the repo to its size in bytes. `read_index`
    returns the shard names an index file lists; it is called only for an
    index the repo has.
    """
    chosen: list[str] | None = None
    layout = "fallback"
    for index, single in PRIMARY_LAYOUTS:
        if index in sizes:
            shards = sorted(set(read_index(index)))
            missing = [s for s in shards if s not in sizes]
            if shards and not missing:
                chosen, layout = shards, index
                break
        if single in sizes:
            chosen, layout = [single], single
            break
    if chosen is None:
        weights = [f for f in sizes if _ext(f) in WEIGHT_EXTS]
        chosen = _fallback_weights(weights)

    aux = [
        f for f in sizes
        if _ext(f) not in WEIGHT_EXTS and sizes[f] <= MAX_AUX_BYTES
    ]
    files = sorted(set(chosen) | set(aux))
    skipped = sorted(set(sizes) - set(files))
    return Selection(files=files,
                     total_bytes=sum(sizes[f] for f in files),
                     skipped=skipped,
                     skipped_bytes=sum(sizes[f] for f in skipped),
                     layout=layout)


def _cache_bytes(blobs_dir: str) -> int:
    """Bytes in the repo's blob store, including partial `.incomplete` files."""
    total = 0
    try:
        with os.scandir(blobs_dir) as entries:
            for entry in entries:
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
    except FileNotFoundError:
        pass
    return total


def _cached_bytes(snapshot: str, files: list[str], sizes: dict[str,
                                                               int]) -> int:
    """Bytes of `files` already complete in a snapshot directory.

    A snapshot entry is a symlink to its blob, written only once the blob is
    whole, so an existing entry is a finished file.
    """
    return sum(sizes[f] for f in files
               if os.path.exists(os.path.join(snapshot, f)))


def cache_space(path: str) -> Space:
    """Room under `path`, which may not exist.

    statvfs's free space where `path` really is: a shared volume
    (`hf_cache_pvc`) or a cache the user moved is measured as itself. Only
    on /dev/shm, no more than its cap (`SHM_LIMIT_ENV`) less what the volume
    already holds. Symlinks are followed both ways, into /dev/shm and out.
    """
    path = existing = os.path.abspath(path)
    while not os.path.exists(existing):
        existing = os.path.dirname(existing)
    # The part not made yet keeps its place under the resolved one.
    real = os.path.normpath(
        os.path.join(os.path.realpath(existing),
                     os.path.relpath(path, existing)))
    st = os.statvfs(existing)
    free = st.f_bavail * st.f_frsize
    limit = os.environ.get(SHM_LIMIT_ENV, "")
    if limit.isdigit() and (real + "/").startswith(SHM_MOUNT + "/"):
        used = (st.f_blocks - st.f_bfree) * st.f_frsize
        return Space(min(free, max(int(limit) - used, 0)), int(limit), used)
    return Space(free)


def compile_reserve(layers: int | None) -> int:
    """Bytes kept on /dev/shm for the compile caches of a `layers`-deep
    model; `COMPILE_RESERVE_UNKNOWN` when that is not known."""
    if not layers:
        return COMPILE_RESERVE_UNKNOWN
    return COMPILE_RESERVE_BASE + COMPILE_RESERVE_PER_LAYER * layers


def config_layers(config: dict) -> int | None:
    """The layer count a model's config.json names, or None."""
    for scope in (config.get("text_config") or {}, config):
        for key in LAYER_KEYS:
            if isinstance(scope.get(key), int) and scope[key] > 0:
                return scope[key]
    return None


def space_problem(model_id: str, cache: str, needed: int, reserve: int,
                  space: Space) -> str | None:
    """Why `needed` more bytes of weights cannot go in `cache`, or None when
    they fit.

    On /dev/shm the compile caches follow the weights there, so `reserve`
    bytes are kept for them (see `compile_reserve`), and the answer names
    the `shm_size_gb` that would hold both. Elsewhere only the weights are
    sized: on a `hf_cache_pvc` the compile caches are kept across Jobs, one
    folder per TPU shape and image, so a run adds to them only when it
    compiles for a new one, and a cache the user moved is theirs to size.
    """
    if space.shm_limit is None:
        if needed <= space.free:
            return None
        return (f"{model_id} needs {needed / GB:.1f} GB more in {cache}, "
                f"which has {space.free / GB:.1f} GB free, so the download "
                f"would run out of space part-way. Free space there, or "
                f"point HF_HUB_CACHE at a larger volume")
    if needed + reserve <= space.free:
        return None
    at_least = math.ceil((space.shm_used + needed + reserve) / GiB)
    return (f"{model_id} needs {needed / GB:.1f} GB more for its weights "
            f"and {reserve / GB:.1f} GB for compile caches in {cache}, on "
            f"/dev/shm, which has {space.free / GB:.1f} GB free under its "
            f"{space.shm_limit / GiB:.0f} GiB cap, so the pod would be "
            f"evicted part-way. Set shm_size_gb to at least {at_least}, or "
            f"use hf_cache_pvc")


def format_progress(done: int, total: int, rate: float) -> str:
    pct = 100.0 * done / total if total else 100.0
    line = (f"{PROGRESS} {done / GB:.1f}/{total / GB:.1f} GB ({pct:.0f}%) "
            f"{rate / 1e6:.0f} MB/s")
    if rate > 0 and total > done:
        line += f" eta {int((total - done) / rate / 60)}m"
    return line


def _say(line: str) -> None:
    # flush: pod logs are a pipe, and a buffered line is a line not seen.
    print(line, flush=True)


def _layers(model_id: str, revision: str | None,
            sizes: dict[str, int]) -> int | None:
    """The layer count the repo's config.json names, or None. It only sizes
    the compile reserve, so a config that cannot be read is not an error."""
    if "config.json" not in sizes:
        return None
    # pylint: disable=g-import-not-at-top
    import huggingface_hub
    try:
        path = huggingface_hub.hf_hub_download(model_id,
                                               "config.json",
                                               revision=revision)
        with open(path, encoding="utf-8") as f:
            return config_layers(json.load(f))
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def parse_gcs_url(url: str) -> tuple[str, str]:
    """(bucket, folder) of a `gs://bucket/folder` URL; folder may be ''."""
    if not url.startswith(GCS_SCHEME):
        raise ValueError(f"{url!r} is not a gs:// URL")
    bucket, _, folder = url[len(GCS_SCHEME):].partition("/")
    if not bucket:
        raise ValueError(f"{url!r} names no bucket")
    return bucket, folder.strip("/")


def _hub_cache() -> str:
    """Where huggingface_hub would cache, without importing it."""
    return (os.environ.get("HF_HUB_CACHE")
            or os.environ.get("HUGGINGFACE_HUB_CACHE") or os.path.join(
                os.environ.get("HF_HOME")
                or os.path.expanduser("~/.cache/huggingface"), "hub"))


def gcs_dest(url: str) -> str:
    """The local folder a gs:// checkpoint is copied to.

    Keyed on the whole URL, not only its last folder: two buckets can each
    hold a `Qwen3-8B/`, and a cache shared between Jobs must keep them apart.
    """
    bucket, folder = parse_gcs_url(url)
    base = os.environ.get(CHECKPOINT_DIR_ENV) or os.path.join(
        _hub_cache(), "gcs")
    key = hashlib.sha256(f"{bucket}/{folder}".encode("utf-8")).hexdigest()[:16]
    return os.path.join(base, key, posixpath.basename(folder) or bucket)


def _gcs_problem(url: str, detail: str) -> OSError | RuntimeError:
    """The exception a storage error stands for, with a message that says
    what to do about it."""
    text = detail.lower()
    if any(m in text for m in GCS_DENIED):
        return PermissionError(
            errno.EACCES,
            f"no permission to read {url} with this job's credentials: run "
            f"the job with an identity that can read the bucket (the TPU "
            f"VM's service account, or Workload Identity on gke_job), or "
            f"mount an access token and name its file in {GCS_TOKEN_ENV} "
            f"({detail.strip()[-300:]})")
    if any(m in text for m in GCS_MISSING):
        return FileNotFoundError(
            errno.ENOENT, f"nothing at {url}: name the checkpoint's folder "
            f"({detail.strip()[-300:]})")
    return RuntimeError(f"could not read {url}: {detail.strip()[-300:]}")


def _parse_du(out: str, folder_url: str) -> dict[str, int]:
    """{name under the folder: size} from the `du` of gcloud or gsutil.

    Each line is `<bytes>  gs://...`; a folder placeholder ends in `/`.
    """
    sizes: dict[str, int] = {}
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        path = parts[1].strip()
        if path.startswith(folder_url) and not path.endswith("/"):
            sizes[path[len(folder_url):]] = int(parts[0])
    return sizes


class _GcsError(Exception):
    """A storage call failed; its text is what the tool said."""


class _CliStorage:
    """Lists and copies objects with the gcloud or gsutil CLI."""

    def __init__(self, tool: str, token_file: str | None) -> None:
        self.tool = tool
        self.env = dict(os.environ)
        if token_file:
            # gcloud's auth/access_token_file property; the path, not the
            # token, is what goes in the environment.
            self.env[GCLOUD_TOKEN_ENV] = token_file

    def _run(self, args: list[str]) -> str:
        argv = ([self.tool, "storage"] + args + ["--verbosity=error"]
                if self.tool == "gcloud" else [self.tool, "-q"] + args)
        proc = subprocess.run(argv,
                              env=self.env,
                              capture_output=True,
                              text=True,
                              check=False)
        if proc.returncode:
            raise _GcsError(proc.stderr or proc.stdout
                            or f"{self.tool} exited {proc.returncode}")
        return proc.stdout

    def list(self, folder_url: str) -> dict[str, int]:
        return _parse_du(self._run(["du", folder_url]), folder_url)

    def copy_many(self, srcs: list[str], folder: str) -> None:
        """Copies each object into `folder`, in one call."""
        args = ["cp"] + srcs + [folder.rstrip("/") + "/"]
        if self.tool == "gsutil":
            # gsutil copies one file at a time unless told otherwise; gcloud
            # already copies in parallel.
            args = ["-m"] + args
        self._run(args)


class _LibraryStorage:
    """Lists and copies objects with the google-cloud-storage library."""

    def __init__(self, token_file: str | None) -> None:
        # pylint: disable=g-import-not-at-top
        from google.cloud import storage
        credentials = None
        if token_file:
            from google.oauth2.credentials import Credentials
            with open(token_file, encoding="utf-8") as f:
                credentials = Credentials(f.read().strip())
        # Bucket and object calls need no project, but the client asks for
        # one, and a host with no default project would fail here.
        project = (os.environ.get("GOOGLE_CLOUD_PROJECT")
                   or os.environ.get("CLOUDSDK_CORE_PROJECT") or "none")
        self.client = storage.Client(project=project, credentials=credentials)

    def _bucket(self, url: str):
        return self.client.bucket(parse_gcs_url(url)[0])

    def list(self, folder_url: str) -> dict[str, int]:
        folder = parse_gcs_url(folder_url)[1]
        prefix = f"{folder}/" if folder else ""
        try:
            return {
                b.name[len(prefix):]: int(b.size or 0)
                for b in self._bucket(folder_url).list_blobs(prefix=prefix)
                if not b.name.endswith("/")
            }
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise _GcsError(f"{type(exc).__name__}: {exc}") from exc

    def copy_many(self, srcs: list[str], folder: str) -> None:
        """Copies each object into `folder`."""
        try:
            for src in srcs:
                self._bucket(src).blob(
                    parse_gcs_url(src)[1]).download_to_filename(
                        os.path.join(folder, posixpath.basename(src)))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise _GcsError(f"{type(exc).__name__}: {exc}") from exc


def _gcs_storage() -> _CliStorage | _LibraryStorage:
    """The way this job reads GCS: gcloud, else gsutil, else the library.

    gsutil cannot take an access token from a file, so with one it is
    passed over for the library.
    """
    token_file = os.environ.get(GCS_TOKEN_ENV) or None
    if token_file and not os.path.isfile(token_file):
        raise FileNotFoundError(
            errno.ENOENT, f"{GCS_TOKEN_ENV} names {token_file}, which is "
            f"not a file")
    if shutil.which("gcloud"):
        return _CliStorage("gcloud", token_file)
    if shutil.which("gsutil") and not token_file:
        return _CliStorage("gsutil", None)
    return _LibraryStorage(token_file)


@contextlib.contextmanager
def _locked(path: str, waiting: str = "") -> Iterator[None]:
    """Holds an exclusive lock on `path`.lock for the duration; says
    `waiting` first when another holder makes it wait."""
    with open(path + ".lock", "a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if waiting:
                _say(waiting)
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _local(dest: str, name: str) -> str:
    """`name`'s path under `dest`; refuses a name that would leave it."""
    path = os.path.normpath(os.path.join(dest, name))
    if os.path.isabs(
            name) or not path.startswith(os.path.normpath(dest) + os.sep):
        raise ValueError(f"refusing object name {name!r}: it leaves {dest}")
    return path


def _complete(dest: str, name: str, size: int) -> bool:
    path = _local(dest, name)
    return os.path.isfile(path) and os.path.getsize(path) == size


def _tree_bytes(root: str) -> int:
    """Bytes under `root`, partial copies included."""
    total = 0
    for folder, _, files in os.walk(root):
        for name in files:
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(folder, name))
    return total


def _copy_files(storage: _CliStorage | _LibraryStorage, folder_url: str,
                dest: str, names: list[str]) -> None:
    """Copies the objects into a staging folder beside `dest`, one call per
    subfolder, then renames each whole file into place.

    Nothing under `dest` is ever a partial file, and a copy killed part-way
    leaves only the staging folder, which the next copy clears.
    """
    staging = dest + PARTIAL_SUFFIX
    groups: dict[str, list[str]] = {}
    for name in names:
        _local(dest, name)  # refuse a name that would leave `dest`
        groups.setdefault(posixpath.dirname(name), []).append(name)
    for sub, group in sorted(groups.items()):
        stage = os.path.join(staging, sub)
        os.makedirs(stage, exist_ok=True)
        storage.copy_many([folder_url + n for n in group], stage)
        for name in group:
            path = _local(dest, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            os.replace(os.path.join(stage, posixpath.basename(name)), path)
    shutil.rmtree(staging, ignore_errors=True)


def _folder_url(url: str) -> str:
    """`gs://bucket/folder/`, the prefix every object under `url` starts with."""
    bucket, folder = parse_gcs_url(url)
    return f"{GCS_SCHEME}{bucket}/" + (f"{folder}/" if folder else "")


def _list_folder(storage: _CliStorage | _LibraryStorage, url: str,
                 folder_url: str) -> dict[str, int]:
    """{name under the folder: size}; names that would leave it are dropped.

    Raises the exception `_gcs_problem` maps a storage error to, or
    FileNotFoundError when the folder holds nothing.
    """
    try:
        sizes = storage.list(folder_url)
    except _GcsError as exc:
        raise _gcs_problem(url, str(exc)) from exc
    sizes = {n: s for n, s in sizes.items() if ".." not in n.split("/")}
    if not sizes:
        raise FileNotFoundError(
            errno.ENOENT, f"nothing at {url}: name the checkpoint's folder")
    return sizes


def gcs_listing(url: str) -> dict[str, int]:
    """{file name under the gs:// folder: size in bytes}, read with the job's
    credentials. Raises PermissionError, FileNotFoundError or RuntimeError
    with a message that says what to do."""
    return _list_folder(_gcs_storage(), url, _folder_url(url))


def gcs_weight_bytes(url: str, local: str) -> int | None:
    """Bytes of the weights a loader reads from the gs:// folder at `url`, by
    the selection `fetch_gcs` copies, or None when it selects none.

    `local` is a copy of the folder's small files (`fetch_gcs(url,
    small_only=True)`), which holds the index the selection follows.
    """
    sizes = gcs_listing(url)

    def read_index(name: str) -> list[str]:
        with open(_local(local, name), encoding="utf-8") as f:
            return list(json.load(f).get("weight_map", {}).values())

    sel = select_files(sizes, read_index)
    return sum(sizes[f] for f in sel.files if _ext(f) in WEIGHT_EXTS) or None


def fetch_gcs(url: str,
              dest: str | None = None,
              small_only: bool = False,
              interval: float = DEFAULT_INTERVAL_SEC) -> str:
    """Copy a gs:// checkpoint folder to local disk; return the local path.

    `small_only` copies the config and tokenizer files and no weights, for a
    job that only tokenizes.
    """
    started = time.monotonic()
    try:
        folder_url = _folder_url(url)
        dest = dest or gcs_dest(url)
        storage = _gcs_storage()
    except (ValueError, OSError) as exc:
        _say(f"{FAILED}: {exc}")
        raise
    os.makedirs(dest, exist_ok=True)
    staging = dest + PARTIAL_SUFFIX
    with _locked(
            dest, f"WEIGHTS WAIT {url}: another job is copying it to "
            f"{dest}; this one goes on when that one is done"):
        # Left by a copy that was killed part-way; nothing reads it.
        shutil.rmtree(staging, ignore_errors=True)
        try:
            sizes = _list_folder(storage, url, folder_url)
        except (OSError, RuntimeError) as exc:
            _say(f"{FAILED}: {exc}")
            raise

        try:
            # The few kB that name the shards and the layer count first.
            _copy_files(storage, folder_url, dest, [
                n for n in METADATA_FILES
                if n in sizes and not _complete(dest, n, sizes[n])
            ])

            def read_index(name: str) -> list[str]:
                with open(_local(dest, name), encoding="utf-8") as f:
                    return list(json.load(f).get("weight_map", {}).values())

            sel = select_files(sizes, read_index)
        except _GcsError as exc:
            problem = _gcs_problem(url, str(exc))
            _say(f"{FAILED}: {problem}")
            raise problem from exc

        files = [
            f for f in sel.files if not (small_only and _ext(f) in WEIGHT_EXTS)
        ]
        total = sum(sizes[f] for f in files)
        todo = [f for f in files if not _complete(dest, f, sizes[f])]
        needed = sum(sizes[f] for f in todo)
        _say(f"WEIGHTS PLAN {url}: {len(files)} files, {total / GB:.1f} GB "
             f"({'small files only' if small_only else sel.layout}); "
             f"skipping {len(sizes) - len(files)} files, "
             f"{(sum(sizes.values()) - total) / GB:.1f} GB")

        layers = None
        if "config.json" in sizes:
            with contextlib.suppress(OSError, ValueError):
                with open(_local(dest, "config.json"), encoding="utf-8") as f:
                    layers = config_layers(json.load(f))
        # A job that only tokenizes compiles nothing.
        reserve = 0 if small_only else compile_reserve(layers)
        space = cache_space(dest)
        problem_text = space_problem(url, dest, needed, reserve, space)
        if problem_text:
            problem_text = problem_text.replace(
                "point HF_HUB_CACHE",
                f"point {CHECKPOINT_DIR_ENV} or HF_HUB_CACHE")
            _say(f"{FAILED}: {problem_text}")
            raise OSError(errno.ENOSPC, problem_text)
        cap = (f" under its {space.shm_limit / GiB:.0f} GiB cap"
               if space.shm_limit is not None else "")
        _say(f"WEIGHTS SPACE {needed / GB:.1f} GB to copy, "
             f"{space.free / GB:.1f} GB free{cap}")

        result: dict[str, BaseException] = {}

        def copy() -> None:
            try:
                _copy_files(storage, folder_url, dest, todo)
            except BaseException as exc:  # pylint: disable=broad-exception-caught
                result["error"] = exc

        at_start = _tree_bytes(dest)
        worker = threading.Thread(target=copy, daemon=True)
        worker.start()
        last, last_t = at_start, time.monotonic()
        while True:
            worker.join(interval)
            if not worker.is_alive():
                break
            now = _tree_bytes(dest) + _tree_bytes(staging)
            now_t = time.monotonic()
            done = min(total - needed + max(now - at_start, 0), total)
            _say(
                format_progress(done, total,
                                (now - last) / max(now_t - last_t, 1e-6)))
            last, last_t = now, now_t

        error = result.get("error")
        if isinstance(error, _GcsError):
            problem = _gcs_problem(url, str(error))
            _say(f"{FAILED}: {problem}")
            raise problem from error
        if error is not None:
            _say(f"{FAILED}: {error}")
            raise error
        short = [f for f in files if not _complete(dest, f, sizes[f])]
        if short:
            problem_text = (f"{len(short)} of {len(files)} files did not "
                            f"arrive whole from {url}, e.g. {short[0]}")
            _say(f"{FAILED}: {problem_text}")
            raise OSError(errno.EIO, problem_text)

    _say(f"{READY} {dest} ({total / GB:.1f} GB; copied {needed / GB:.1f} GB "
         f"from {url} in {int(time.monotonic() - started)}s)")
    return dest


def fetch(model_id: str,
          revision: str | None = None,
          interval: float = DEFAULT_INTERVAL_SEC) -> str:
    """Download what the engine loads and return the local snapshot path."""
    if model_id.startswith(GCS_SCHEME):
        if revision:
            _say(f"WEIGHTS NOTE {model_id}: --revision {revision} ignored; "
                 f"a gs:// folder holds one revision")
        return fetch_gcs(model_id, interval=interval)
    if os.path.isdir(model_id):
        _say(f"{READY} {model_id} (local directory, nothing to download)")
        return model_id

    # pylint: disable=g-import-not-at-top
    import huggingface_hub
    from huggingface_hub import constants

    if os.environ.get("HF_HUB_OFFLINE"):
        dest = huggingface_hub.snapshot_download(model_id,
                                                 revision=revision,
                                                 local_files_only=True)
        _say(f"{READY} {dest} (cached snapshot, nothing to download)")
        return dest

    started = time.monotonic()
    try:
        info = huggingface_hub.HfApi().model_info(model_id,
                                                  revision=revision,
                                                  files_metadata=True)
        sizes = {s.rfilename: s.size or 0 for s in info.siblings}

        def read_index(name: str) -> list[str]:
            path = huggingface_hub.hf_hub_download(model_id,
                                                   name,
                                                   revision=revision)
            with open(path, encoding="utf-8") as f:
                return list(json.load(f).get("weight_map", {}).values())

        sel = select_files(sizes, read_index)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _say(f"{FAILED}: could not list {model_id}: {exc}")
        raise

    _say(f"WEIGHTS PLAN {model_id}: {len(sel.files)} files, "
         f"{sel.total_bytes / GB:.1f} GB ({sel.layout}); skipping "
         f"{len(sel.skipped)} files, {sel.skipped_bytes / GB:.1f} GB")
    for name in sel.skipped:
        if sizes[name] >= 100 * 1024 * 1024:
            _say(f"  skip {name} ({sizes[name] / GB:.1f} GB)")

    repo_cache = os.path.join(constants.HF_HUB_CACHE,
                              "models--" + model_id.replace("/", "--"))
    blobs = os.path.join(repo_cache, "blobs")
    # Progress counts this selection only. The blob store can also hold other
    # revisions and files this run skips, so its size alone overstates it:
    # count what this run adds to the store, plus the selected files that
    # are already complete in this revision's snapshot.
    at_start = _cache_bytes(blobs)
    cached = _cached_bytes(os.path.join(repo_cache, "snapshots", info.sha),
                           sel.files, sizes) if info.sha else 0
    needed, space = sel.total_bytes - cached, cache_space(repo_cache)
    problem = space_problem(
        model_id, constants.HF_HUB_CACHE, needed,
        compile_reserve(_layers(model_id, revision, sizes)), space)
    if problem:
        _say(f"{FAILED}: {problem}")
        raise OSError(errno.ENOSPC, problem)
    cap = (f" under its {space.shm_limit / GiB:.0f} GiB cap"
           if space.shm_limit is not None else "")
    _say(f"WEIGHTS SPACE {needed / GB:.1f} GB to download, "
         f"{space.free / GB:.1f} GB free{cap}")
    result: dict[str, object] = {}

    def download() -> None:
        try:
            # Exact names, escaped: allow_patterns are fnmatch patterns.
            # `revision` as given, not the resolved sha: only a download by
            # branch writes the refs/<branch> file an offline engine reads.
            result["path"] = huggingface_hub.snapshot_download(
                model_id,
                revision=revision,
                allow_patterns=[glob.escape(f) for f in sel.files])
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            result["error"] = exc

    worker = threading.Thread(target=download, daemon=True)
    worker.start()
    last, last_t = at_start, time.monotonic()
    while True:
        worker.join(interval)
        if not worker.is_alive():
            break
        now, now_t = _cache_bytes(blobs), time.monotonic()
        done = min(cached + max(now - at_start, 0), sel.total_bytes)
        _say(
            format_progress(done, sel.total_bytes,
                            (now - last) / max(now_t - last_t, 1e-6)))
        last, last_t = now, now_t

    if "error" in result:
        _say(f"{FAILED}: {result['error']}")
        raise result["error"]  # type: ignore[misc]
    fetched = max(_cache_bytes(blobs) - at_start, 0)
    _say(f"{READY} {result['path']} ({sel.total_bytes / GB:.1f} GB; "
         f"downloaded {fetched / GB:.1f} GB in "
         f"{int(time.monotonic() - started)}s)")
    return str(result["path"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_id",
                        help="Hugging Face repo id, local dir, or gs:// "
                        "checkpoint folder.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--interval",
                        type=float,
                        default=DEFAULT_INTERVAL_SEC,
                        help="Seconds between progress lines.")
    parser.add_argument("--path-file",
                        default=None,
                        help="Write the local snapshot path to this file, "
                        "for a launcher to serve from.")
    args = parser.parse_args(argv)
    path = fetch(args.model_id, revision=args.revision, interval=args.interval)
    if args.path_file:
        with open(args.path_file, "w", encoding="utf-8") as f:
            f.write(path + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
