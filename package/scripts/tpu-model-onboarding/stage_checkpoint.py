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
"""Put a Hugging Face or local model's checkpoint in a GCS bucket once, and find it there.

Runs on the host, with the user's own credentials: the only place that
writes to the checkpoints folder. Jobs then read it with
`fetch_weights.py <gs:// path>`, which copies it to their own disk instead of
downloading from the hub, so a model is downloaded from the hub once per
bucket rather than once per job.

    stage_checkpoint.py Qwen/Qwen3-8B --bucket gs://my-bucket \\
        [--folder checkpoints] [--revision REV] [--out result.json]

A staged copy lives at `gs://<bucket>/<folder>/<repo id>/<commit, 12>/`, so
two revisions never share a folder. It is reused only when it can be
verified:

-   its `.hf_source.json` manifest, written last, names this repo and commit
    and the same files the hub lists for it, each with the hub's own digest
    (the LFS sha256, or the git blob id of a small file); and
-   every object is still there at that size, with the crc32c it had when it
    was uploaded, so a file changed or replaced since does not pass.

A copy without a manifest -- left by an interrupted upload, or put in the
folder some other way -- is never reused on names and sizes alone: it is
listed under `candidates` with why it was not used, unless
`--trust-unverified` says to accept a copy whose names, sizes and
config.json match.

Each file is downloaded from the hub at the resolved commit, checked against
the hub's digest, and uploaded before the next is downloaded, so local disk
never holds more than one file. An interrupted upload resumes: a file is
skipped only when it still has the crc32c this tool recorded right after
checking and uploading it (in a progress record beside the folder, or an
earlier manifest); an object's own metadata can be set by anyone, so it is
kept as provenance, not trusted. Two sessions staging the same model take
turns through a lock object beside the folder; the second waits, then reuses
the first one's copy.

Prints, and writes to --out, a JSON result: `checkpoint` (the gs:// path to
load), `reused`, `verified`, `revision` (the resolved commit), `uploaded`
(files and bytes this run uploaded) and `candidates`.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import fnmatch
import hashlib
import json
import os
import pathlib
import posixpath
import re
import socket
import sys
import tempfile
import time
from collections.abc import Callable
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch_weights  # noqa: E402  # pylint: disable=wrong-import-position

MANIFEST = ".hf_source.json"
DEFAULT_FOLDER = "checkpoints"
# Repo files a server never loads; a copy without them still matches.
NOT_LOADED = (".gitattributes", "*.md", "LICENSE*", "*.png", "*.jpg", "*.jpeg",
              "*.gif")
# Object metadata keys an uploaded file carries.
DIGEST_KEY = "hf-digest"
SOURCE_KEY = "hf-source"
DEFAULT_WAIT_SEC = 6 * 3600
DEFAULT_POLL_SEC = 30
# A lock older than this belongs to a session that died without releasing
# it; uploading a 60 GB model takes well under this.
DEFAULT_STALE_SEC = 12 * 3600
HASH_CHUNK = 8 * 1024 * 1024


class StageError(Exception):
    """Staging cannot go on; the message says why and what to do."""


@dataclasses.dataclass(frozen=True)
class FileSpec:
    size: int
    # "sha256:<hex>" for an LFS file, "git-sha1:<hex>" for a small one.
    digest: str


@dataclasses.dataclass
class Plan:
    """What the hub says the model is, at one commit."""
    repo: str
    commit: str
    files: dict[str, FileSpec]
    config: bytes | None


def _say(line: str) -> None:
    # stderr: stdout is the JSON result.
    print(line, file=sys.stderr, flush=True)


def _loaded(name: str) -> bool:
    base = posixpath.basename(name)
    return not any(fnmatch.fnmatch(base, p) for p in NOT_LOADED)


def _hub_problem(model: str, exc: BaseException) -> StageError:
    """A hub error as what to do about it, never with the token."""
    kind = type(exc).__name__
    text = str(exc).splitlines()[0][:300] if str(exc) else kind
    if kind == "GatedRepoError" or re.search(r"\b(401|403)\b", text):
        return StageError(
            f"{model} is gated or private on the hub: accept its license with "
            f"the account whose token is in HF_TOKEN, then retry ({kind})")
    if kind in ("RepositoryNotFoundError", "RevisionNotFoundError"):
        return StageError(f"{model}: {text} ({kind})")
    return StageError(f"could not read {model} from the hub: {text} ({kind})")


def hub_plan(model: str, revision: str | None = None) -> Plan:
    """The files a server loads, with sizes and digests, from the hub."""
    # pylint: disable=g-import-not-at-top
    import huggingface_hub
    try:
        info = huggingface_hub.HfApi().model_info(model,
                                                  revision=revision,
                                                  files_metadata=True)
        siblings = {s.rfilename: s for s in info.siblings}
        sizes = {name: s.size or 0 for name, s in siblings.items()}

        def read_index(name: str) -> list[str]:
            path = huggingface_hub.hf_hub_download(model,
                                                   name,
                                                   revision=info.sha)
            with open(path, encoding="utf-8") as f:
                return list(json.load(f).get("weight_map", {}).values())

        selection = fetch_weights.select_files(sizes, read_index)
        config = None
        if "config.json" in siblings:
            config = pathlib.Path(
                huggingface_hub.hf_hub_download(
                    model, "config.json", revision=info.sha)).read_bytes()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise _hub_problem(model, exc) from exc
    files = {}
    for name in selection.files:
        if not _loaded(name):
            continue
        s = siblings[name]
        lfs = getattr(s, "lfs", None)
        sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
        digest = (f"sha256:{sha256}"
                  if sha256 else f"git-sha1:{getattr(s, 'blob_id', '')}")
        files[name] = FileSpec(size=sizes[name], digest=digest)
    return Plan(repo=model, commit=info.sha, files=files, config=config)


def file_digest(path: str, kind: str) -> str:
    """`kind:<hex>` of the file at `path`, the way the hub computes it."""
    if kind == "sha256":
        h = hashlib.sha256()
    elif kind == "git-sha1":
        # The id git gives a blob: sha1 over a header and the content.
        h = hashlib.sha1(usedforsecurity=False)
        h.update(f"blob {os.path.getsize(path)}\0".encode("ascii"))
    else:
        raise ValueError(f"unknown digest kind {kind!r}")
    with open(path, "rb") as f:
        while chunk := f.read(HASH_CHUNK):
            h.update(chunk)
    return f"{kind}:{h.hexdigest()}"


def hub_fetch(model: str, commit: str, name: str, folder: str) -> str:
    """Downloads one file of the repo at `commit` into `folder`."""
    # pylint: disable=g-import-not-at-top
    import huggingface_hub
    try:
        return huggingface_hub.hf_hub_download(model,
                                               name,
                                               revision=commit,
                                               local_dir=folder)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise _hub_problem(model, exc) from exc


def local_plan(model_dir: str, revision: str | None = None) -> Plan:
    """The files a server loads, with sizes and sha256 digests, from a local directory."""
    del revision
    root = pathlib.Path(model_dir).resolve()
    if not root.is_dir():
        raise StageError(f"{model_dir} is not a local directory")
    sizes: dict[str, int] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        sizes[rel.as_posix()] = p.stat().st_size

    def read_index(name: str) -> list[str]:
        with open(root / name, encoding="utf-8") as f:
            return list(json.load(f).get("weight_map", {}).values())

    try:
        selection = fetch_weights.select_files(sizes, read_index)
    except ValueError as exc:
        raise StageError(f"{model_dir}: {exc}") from exc
    config = (root /
              "config.json").read_bytes() if "config.json" in sizes else None
    files: dict[str, FileSpec] = {}
    tree = hashlib.sha256()
    for name in sorted(selection.files):
        if not _loaded(name):
            continue
        digest = file_digest(str(root / name), "sha256")
        files[name] = FileSpec(size=sizes[name], digest=digest)
        tree.update(f"{name}\0{sizes[name]}\0{digest}\n".encode("utf-8"))
    repo = f"local/{root.name or 'checkpoint'}"
    return Plan(repo=repo, commit=tree.hexdigest(), files=files, config=config)


def local_fetch(model_dir: str) -> Callable[[str, str, str, str], str]:
    """Returns a fetch callable that reads files directly from `model_dir`."""
    root = pathlib.Path(model_dir).resolve()

    def _fetch(repo: str, commit: str, name: str, folder: str) -> str:
        del repo, commit, folder
        return str(root / name)

    return _fetch


def staged_prefix(folder: str, model: str, commit: str) -> str:
    """`<folder>/<repo id>/<commit, 12>/` within the bucket."""
    folder = folder.strip("/")
    return (f"{folder}/" if folder else "") + f"{model}/{commit[:12]}/"


def _listing(bucket, prefix: str) -> dict[str, Any]:
    """{object name: blob} under `prefix`, folder placeholders left out."""
    return {
        b.name: b
        for b in bucket.list_blobs(prefix=prefix) if not b.name.endswith("/")
    }


def verify_copy(bucket, prefix: str, plan: Plan,
                listing: dict[str, Any]) -> str | None:
    """Why the copy at `prefix` is not `plan`'s checkpoint, or None if it is.

    `listing` holds at least every object under `prefix`.
    """
    manifest_blob = listing.get(prefix + MANIFEST)
    if manifest_blob is None:
        return "no manifest: not a finished copy"
    try:
        manifest = json.loads(
            bucket.blob(prefix + MANIFEST).download_as_bytes())
    except (ValueError, OSError) as exc:
        return f"manifest unreadable: {exc}"
    if manifest.get("repo") != plan.repo:
        return f"manifest is for {manifest.get('repo')!r}"
    if manifest.get("revision") != plan.commit:
        return (
            f"manifest is for revision {str(manifest.get('revision'))[:12]}"
            f", not {plan.commit[:12]}")
    recorded = manifest.get("files") or {}
    for name, spec in sorted(plan.files.items()):
        entry = recorded.get(name)
        if entry is None:
            return f"manifest does not list {name}"
        if entry.get("size") != spec.size or entry.get(
                "digest") != spec.digest:
            return f"{name} in the manifest differs from the hub's"
        blob = listing.get(prefix + name)
        if blob is None:
            return f"{name} is missing"
        if blob.size != spec.size:
            return f"{name} is {blob.size} bytes, not {spec.size}"
        if blob.crc32c != entry.get("crc32c"):
            return f"{name} changed since it was staged (crc32c differs)"
    return None


def unverified_match(bucket, prefix: str, plan: Plan,
                     listing: dict[str, Any]) -> str | None:
    """Why a copy without a manifest is not `plan`'s checkpoint by names,
    sizes and config.json, or None if it matches on all three."""
    missing = [n for n in plan.files if prefix + n not in listing]
    if missing:
        return f"missing {len(missing)} files, e.g. {sorted(missing)[0]}"
    wrong = [
        n for n in sorted(plan.files)
        if listing[prefix + n].size != plan.files[n].size
    ]
    if wrong:
        return f"{len(wrong)} files differ in size, e.g. {wrong[0]}"
    if plan.config is not None and bucket.blob(
            prefix + "config.json").download_as_bytes() != plan.config:
        return "config.json differs from the requested revision"
    return None


def _candidates(listing: dict[str, Any], model: str) -> list[str]:
    """Folders under the checkpoints folder that could hold a checkpoint:
    those with a manifest or a config.json; ones named like the model first.
    """
    found = set()
    for name in listing:
        base = posixpath.basename(name)
        if base in (MANIFEST, "config.json"):
            found.add(name[:-len(base)])
    short = model.rsplit("/", 1)[-1].lower()
    return sorted(found, key=lambda p: (short not in p.lower(), p))


def _is_precondition_failed(exc: BaseException) -> bool:
    return (getattr(exc, "code", None) == 412
            or type(exc).__name__ == "PreconditionFailed")


def _is_not_found(exc: BaseException) -> bool:
    return getattr(exc, "code",
                   None) == 404 or type(exc).__name__ == "NotFound"


def _age_sec(blob) -> float:
    created = getattr(blob, "time_created", None)
    if created is None:
        return 0.0
    return (datetime.datetime.now(datetime.timezone.utc) -
            created).total_seconds()


class _Lock:
    """An object beside the staged folder that only one session can create.

    Created with `if_generation_match=0`, so the create fails if it exists;
    released by deleting the generation this session created, so a stale
    lock taken over by another session is never deleted by this one.
    """

    def __init__(self, bucket, name: str) -> None:
        self.bucket, self.name, self.generation = bucket, name, None

    def try_acquire(self) -> bool:
        blob = self.bucket.blob(self.name)
        owner = json.dumps({
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started": time.time(),
        })
        try:
            blob.upload_from_string(owner,
                                    content_type="application/json",
                                    if_generation_match=0)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if _is_precondition_failed(exc):
                return False
            raise
        self.generation = blob.generation
        return True

    def break_if_stale(self, stale_sec: float) -> bool:
        """Deletes the lock if it is older than `stale_sec`; True if gone."""
        blob = self.bucket.get_blob(self.name)
        if blob is None:
            return True
        if _age_sec(blob) < stale_sec:
            return False
        _say(f"STAGE NOTE breaking a lock held for {_age_sec(blob) / 3600:.1f}"
             f" h: gs://{self.bucket.name}/{self.name}")
        try:
            blob.delete(if_generation_match=blob.generation)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if not (_is_precondition_failed(exc) or _is_not_found(exc)):
                raise
        return True

    def release(self) -> None:
        if self.generation is None:
            return
        try:
            self.bucket.blob(
                self.name).delete(if_generation_match=self.generation)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if not (_is_precondition_failed(exc) or _is_not_found(exc)):
                raise
        self.generation = None


def _progress_name(prefix: str) -> str:
    """The record of files uploaded so far, beside (not in) the folder."""
    return prefix.rstrip("/") + ".progress.json"


def _known_crcs(bucket, prefix: str, plan: Plan) -> dict[str, str]:
    """{file: crc32c} this tool recorded right after checking and uploading
    each file: from an earlier manifest, then the progress record.

    Only these let a file be skipped on a resumed run. An object's own
    metadata can be set by anyone, so it is provenance, not proof.
    """
    known: dict[str, str] = {}
    for name in (prefix + MANIFEST, _progress_name(prefix)):
        blob = bucket.get_blob(name)
        if blob is None:
            continue
        try:
            record = json.loads(blob.download_as_bytes())
        except (ValueError, OSError):
            continue
        if record.get("repo", plan.repo) != plan.repo or record.get(
                "revision", plan.commit) != plan.commit:
            continue
        for file, entry in (record.get("files") or {}).items():
            spec = plan.files.get(file)
            if spec and entry.get("digest") == spec.digest:
                known[file] = entry.get("crc32c")
    return known


def _upload(bucket, prefix: str, plan: Plan, fetch: Callable[..., str],
            listing: dict[str, Any]) -> tuple[dict[str, str], int, int]:
    """Uploads every file not already staged; returns ({name: crc32c},
    files uploaded, bytes uploaded)."""
    known = _known_crcs(bucket, prefix, plan)
    if prefix + MANIFEST in listing:
        # A copy that no longer verifies is being staged again: withdraw its
        # manifest first, so nothing takes the folder for finished meanwhile.
        try:
            bucket.blob(prefix + MANIFEST).delete()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if not _is_not_found(exc):
                raise
    crcs: dict[str, str] = {}
    count = size = 0
    names = sorted(plan.files)
    progress = bucket.blob(_progress_name(prefix))
    with tempfile.TemporaryDirectory() as tmp:
        for i, name in enumerate(names, 1):
            spec = plan.files[name]
            have = listing.get(prefix + name)
            if (have is not None and have.size == spec.size
                    and have.crc32c is not None
                    and known.get(name) == have.crc32c):
                # Checked and uploaded by an earlier run, and unchanged since.
                crcs[name] = have.crc32c
                continue
            local = fetch(plan.repo, plan.commit, name, tmp)
            got = file_digest(local, spec.digest.split(":", 1)[0])
            if os.path.getsize(local) != spec.size or got != spec.digest:
                raise StageError(
                    f"{name} downloaded from the hub does not match the hub's "
                    f"own digest ({got} != {spec.digest}); nothing was staged "
                    f"for it")
            blob = bucket.blob(prefix + name)
            blob.metadata = {
                DIGEST_KEY: spec.digest,
                SOURCE_KEY: f"{plan.repo}@{plan.commit}/{name}",
            }
            blob.upload_from_filename(local)
            crcs[name] = blob.crc32c
            if pathlib.Path(local).resolve().is_relative_to(
                    pathlib.Path(tmp).resolve()):
                os.remove(local)
            count, size = count + 1, size + spec.size
            progress.upload_from_string(json.dumps({
                "repo": plan.repo,
                "revision": plan.commit,
                "files": {
                    n: {
                        "digest": plan.files[n].digest,
                        "crc32c": c
                    }
                    for n, c in crcs.items()
                },
            }),
                                        content_type="application/json")
            _say(f"STAGE uploaded {i}/{len(names)} {name} "
                 f"({spec.size / fetch_weights.GB:.2f} GB)")
    return crcs, count, size


def _write_manifest(bucket, prefix: str, plan: Plan, crcs: dict[str,
                                                                str]) -> None:
    """Written last: its presence is what says the copy is whole."""
    manifest = {
        "repo": plan.repo,
        "revision": plan.commit,
        "staged_by": "stage_checkpoint.py",
        "staged_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "files": {
            name: {
                "size": spec.size,
                "digest": spec.digest,
                "crc32c": crcs[name]
            }
            for name, spec in sorted(plan.files.items())
        },
    }
    bucket.blob(prefix + MANIFEST).upload_from_string(
        json.dumps(manifest, indent=2), content_type="application/json")


def stage(bucket,
          model: str,
          revision: str | None = None,
          folder: str = DEFAULT_FOLDER,
          *,
          plan: Callable[..., Plan] = hub_plan,
          fetch: Callable[..., str] = hub_fetch,
          trust_unverified: bool = False,
          wait_sec: float = DEFAULT_WAIT_SEC,
          poll_sec: float = DEFAULT_POLL_SEC,
          stale_sec: float = DEFAULT_STALE_SEC) -> dict[str, Any]:
    """Finds `model`'s checkpoint in `bucket`'s checkpoints folder, or
    stages it there; returns the result described in the module docstring.
    """
    if plan is hub_plan and fetch is hub_fetch and os.path.isdir(model):
        plan = local_plan
        fetch = local_fetch(model)
    want = plan(model, revision)
    if not want.files:
        raise StageError(
            f"no files a server loads found in {model}" if os.path.isdir(model)
            else f"the hub lists no files a server loads for {model}")
    root = folder.strip("/") + "/" if folder.strip("/") else ""
    dest = staged_prefix(folder, want.repo, want.commit)
    url = f"gs://{bucket.name}/"

    def result(path: str,
               reused: bool,
               verified: bool,
               tried: list,
               uploaded=(0, 0)) -> dict[str, Any]:
        return {
            "checkpoint": url + path,
            "reused": reused,
            "verified": verified,
            "repo": want.repo,
            "revision": want.commit,
            "files": len(want.files),
            "bytes": sum(s.size for s in want.files.values()),
            "uploaded": {
                "files": uploaded[0],
                "bytes": uploaded[1]
            },
            "candidates": tried,
        }

    listing = _listing(bucket, root)
    tried: list[dict[str, str]] = []
    trusted = None
    for prefix in sorted(_candidates(listing, want.repo),
                         key=lambda p: p != dest):
        why = verify_copy(bucket, prefix, want, listing)
        if why is None:
            return result(prefix, True, True, tried)
        if prefix + MANIFEST not in listing:
            loose = unverified_match(bucket, prefix, want, listing)
            if loose is None:
                if trust_unverified and trusted is None:
                    trusted = prefix
                    continue
                why = ("matches by names, sizes and config.json but has no "
                       "manifest to verify it; pass --trust-unverified to "
                       "use it")
            else:
                why = f"no manifest, and {loose}"
        tried.append({"path": url + prefix, "not_used": why})
    if trusted is not None:
        return result(trusted, True, False, tried)

    lock = _Lock(bucket, dest.rstrip("/") + ".lock")
    deadline = time.monotonic() + wait_sec
    said_wait = False
    while not lock.try_acquire():
        if not said_wait:
            _say(f"STAGE WAIT another session is staging {url}{dest}; "
                 f"waiting for it to finish")
            said_wait = True
        if verify_copy(bucket, dest, want, _listing(bucket, dest)) is None:
            return result(dest, True, True, tried)
        if lock.break_if_stale(stale_sec):
            continue
        if time.monotonic() > deadline:
            raise StageError(
                f"gave up after {wait_sec / 3600:.1f} h waiting for the "
                f"session holding gs://{bucket.name}/{lock.name}")
        time.sleep(poll_sec)
    try:
        here = _listing(bucket, dest)
        # It may have been finished between our look and the lock.
        if verify_copy(bucket, dest, want, here) is None:
            return result(dest, True, True, tried)
        _say(
            f"STAGE {want.repo}@{want.commit[:12]}: {len(want.files)} files, "
            f"{sum(s.size for s in want.files.values()) / fetch_weights.GB:.1f}"
            f" GB to {url}{dest}")
        crcs, count, size = _upload(bucket, dest, want, fetch, here)
        _write_manifest(bucket, dest, want, crcs)
        with contextlib.suppress(Exception):
            bucket.blob(_progress_name(dest)).delete()
    finally:
        lock.release()
    return result(dest, False, True, tried, (count, size))


def _client_bucket(url: str):
    """The bucket `url` names, with the user's default credentials."""
    # pylint: disable=g-import-not-at-top
    from google.cloud import storage
    bucket, _ = fetch_weights.parse_gcs_url(url)
    # Bucket and object calls need no project, but the client asks for one,
    # and a host with no default project would fail here.
    project = (os.environ.get("GOOGLE_CLOUD_PROJECT")
               or os.environ.get("CLOUDSDK_CORE_PROJECT") or "none")
    return storage.Client(project=project).bucket(bucket)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "model",
        help="Hugging Face repo id (e.g. Qwen/Qwen3-8B) or local checkpoint"
        " directory")
    ap.add_argument("--bucket",
                    required=True,
                    help="gs://<bucket>; only the bucket is used")
    ap.add_argument("--folder",
                    default=DEFAULT_FOLDER,
                    help="the shared checkpoints folder within the bucket "
                    f"(default {DEFAULT_FOLDER!r}); only it is searched")
    ap.add_argument("--revision", help="branch, tag or commit (default: main)")
    ap.add_argument("--out", help="also write the JSON result here")
    ap.add_argument("--trust-unverified",
                    action="store_true",
                    help="reuse a copy with no manifest when its names, sizes "
                    "and config.json match")
    ap.add_argument("--wait",
                    type=float,
                    default=DEFAULT_WAIT_SEC,
                    help="seconds to wait for another session staging the "
                    "same model")
    args = ap.parse_args(argv)
    extra = ({
        "plan": local_plan,
        "fetch": local_fetch(args.model)
    } if os.path.isdir(args.model) else {})
    try:
        out = stage(_client_bucket(args.bucket),
                    args.model,
                    args.revision,
                    args.folder,
                    trust_unverified=args.trust_unverified,
                    wait_sec=args.wait,
                    **extra)
    except (StageError, ValueError) as exc:
        _say(f"STAGE FAILED: {exc}")
        return 1
    except Exception as exc:  # pylint: disable=broad-exception-caught
        problem = fetch_weights._gcs_problem(  # pylint: disable=protected-access
            args.bucket, f"{type(exc).__name__}: {exc}")
        _say(f"STAGE FAILED: {problem}")
        return 1
    text = json.dumps(out, indent=2)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    _say(f"STAGE READY {out['checkpoint']} "
         f"({'reused' if out['reused'] else 'staged'}"
         f"{'' if out['verified'] else ', unverified'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
