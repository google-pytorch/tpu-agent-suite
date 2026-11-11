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
"""The image's vllm_torchtpu git history, as the base of a local campaign repo.

  plugin_source.py probe [--output plugin.bundle]       # compute recipe
  plugin_source.py base <repo> <probe dir> --image IMAGE
  plugin_source.py local <checkout> --image IMAGE [--path IN_IMAGE_PATH]
  plugin_source.py diff <repo> <ref> <out.patch> [--from REV]

`probe` runs on the image, with its Python. The installed plugin must be a git
checkout with no modified tracked files; untracked files and commits no remote
has are fine. It writes HEAD's history as a git bundle, and beside it
plugin.json: the sha, the plugin's path inside the repo, the remote, and the
untracked files. `base` fetches the bundle into <repo> (created when missing)
and tags that sha `base`; its record keeps the checkout's in-image `path`.
`local` is `base` for a copy of the image's plugin tree on this host: the copy
is the repo. One with no .git is committed once, as `base`, under a fixed
author and date, so the same tree always gets the same sha.
Every hypothesis is a branch and every test a commit on it; `diff` writes
`git diff base <ref>` (or from REV), relative to the directory holding
`vllm_torchtpu/`, for apply_vllm_patch.sh, refusing a ref that does not
descend from it or touches anything outside the plugin.
`base`, `local` and `diff` print one JSON record.

Exit codes: 0 done; 2 usage; 3 plugin not found; 6 ref refused; 7 the image's
plugin is not a git checkout, or has modified tracked files.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys

PKG = "vllm_torchtpu"
META = "plugin.json"
RECORD = "plugin_source.json"
IGNORED = ("__pycache__/", "*.pyc", PKG + "/.tpu_agent_suite_patches")
BASE_DATE = "2000-01-01T00:00:00Z"
# The image's checkout may belong to another uid than the job's.
GIT = ("git", "-c", "safe.directory=*", "-c", "user.name=plugin_source", "-c",
       "user.email=plugin_source@localhost", "-c", "commit.gpgsign=false")


def _git(cwd, *args):
    try:
        r = subprocess.run([*GIT, "-C", cwd, *args],
                           capture_output=True,
                           text=True)
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _record_path(repo):
    # Inside the git dir, so it is never a change; a worktree has its own.
    return os.path.join(
        _git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"),
        RECORD)


def probe(output):
    spec = importlib.util.find_spec(PKG)
    if spec is None or not spec.submodule_search_locations:
        print(f"{PKG} is not importable with {sys.executable}",
              file=sys.stderr)
        return 3
    pkg_dir = os.path.realpath(list(spec.submodule_search_locations)[0])
    top = _git(pkg_dir, "rev-parse", "--show-toplevel")
    sha = top and _git(top, "rev-parse", "--verify", "HEAD^{commit}")
    if not sha:
        print(f"{pkg_dir} is not in a git checkout with a commit",
              file=sys.stderr)
        return 7
    modified = _git(top, "status", "--porcelain", "--untracked-files=no")
    if modified:
        print(
            f"{top} has modified tracked files; commit them in the image:\n"
            f"{modified}",
            file=sys.stderr)
        return 7
    if _git(top, "bundle", "create", os.path.abspath(output), "HEAD") is None:
        print(f"git bundle of {top} failed (a shallow clone?)",
              file=sys.stderr)
        return 7
    untracked = _git(top, "ls-files", "--others", "--exclude-standard")
    meta = {
        "sha": sha,
        "path": top,
        "prefix": os.path.relpath(os.path.dirname(pkg_dir), top),
        "remote": _git(top, "remote", "get-url", "origin"),
        "untracked": untracked.splitlines() if untracked else [],
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(output)), META),
              "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))
    return 0


def base(repo, probe_dir, image):
    with open(os.path.join(probe_dir, META), encoding="utf-8") as f:
        meta = json.load(f)
    sha = meta["sha"]
    if not os.path.isdir(os.path.join(repo, ".git")):
        os.makedirs(repo, exist_ok=True)
        subprocess.run([*GIT, "-C", repo, "init", "-q"], check=True)
        with open(os.path.join(repo, ".git", "info", "exclude"),
                  "a",
                  encoding="utf-8") as f:
            f.write("\n".join(IGNORED) + "\n")
    current = _git(repo, "rev-parse", "--verify", "-q", "base^{commit}")
    if current and current != sha:
        print(
            f"{repo} already has base {current}, not {sha}: "
            "one image per repo",
            file=sys.stderr)
        return 2
    if _git(repo, "cat-file", "-e", sha + "^{commit}") is None:
        bundle = os.path.join(probe_dir, "plugin.bundle")
        if _git(repo, "fetch", "-q", bundle, "HEAD") is None:
            print(f"cannot fetch {sha} from {bundle}", file=sys.stderr)
            return 2
    _git(repo, "tag", "-f", "base", sha)
    if _git(repo, "rev-parse", "--verify", "-q", "HEAD") is None:
        _git(repo, "checkout", "-q", "--detach", "base")
    record = {
        "image": image,
        "repo": os.path.realpath(repo),
        "base_sha": sha,
        "path": meta["path"],
        "prefix": meta["prefix"],
        "remote": meta["remote"],
        "untracked": meta["untracked"],
    }
    with open(_record_path(repo), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record))
    return 0


def _find_pkg(checkout):
    # Tracked files only: an image's build/lib/ copy is ignored, not the plugin.
    found = _git(checkout, "ls-files", "--", f"*{PKG}/__init__.py")
    paths = [f for f in (found or "").splitlines() if f.split("/")[-2] == PKG]
    if not paths:
        return None
    return os.path.join(checkout, os.path.dirname(min(paths, key=len)))


def local(checkout, image, path):
    checkout = os.path.realpath(checkout)
    if not any(PKG in dirs for _, dirs, _ in os.walk(checkout)):
        print(f"no {PKG}/ under {checkout}", file=sys.stderr)
        return 3
    if not os.path.isdir(os.path.join(checkout, ".git")):
        subprocess.run([*GIT, "-C", checkout, "init", "-q"], check=True)
        with open(os.path.join(checkout, ".git", "info", "exclude"),
                  "a",
                  encoding="utf-8") as f:
            f.write("\n".join(IGNORED) + "\n")
        env = dict(os.environ,
                   GIT_AUTHOR_DATE=BASE_DATE,
                   GIT_COMMITTER_DATE=BASE_DATE)
        subprocess.run([*GIT, "-C", checkout, "add", "-A"], check=True)
        subprocess.run(
            [*GIT, "-C", checkout, "commit", "-q", "-m", f"base: {image}"],
            check=True,
            env=env)
    pkg_dir = _find_pkg(checkout)
    if pkg_dir is None:
        print(f"no tracked {PKG}/ in {checkout}", file=sys.stderr)
        return 3
    current = _git(checkout, "rev-parse", "--verify", "-q", "base^{commit}")
    if current is None:
        modified = _git(checkout, "status", "--porcelain",
                        "--untracked-files=no")
        if modified:
            print(
                f"{checkout} has modified tracked files; commit them first:\n"
                f"{modified}",
                file=sys.stderr)
            return 7
        current = _git(checkout, "rev-parse", "--verify", "HEAD^{commit}")
        _git(checkout, "tag", "base", current)
    record_path = _record_path(checkout)
    if os.path.exists(record_path):
        with open(record_path, encoding="utf-8") as f:
            if json.load(f)["image"] != image:
                print(f"{checkout} already has the base of another image",
                      file=sys.stderr)
                return 2
    untracked = _git(checkout, "ls-files", "--others", "--exclude-standard")
    record = {
        "image": image,
        "repo": checkout,
        "base_sha": current,
        "path": path or checkout,
        "prefix": os.path.relpath(os.path.dirname(pkg_dir), checkout),
        "remote": _git(checkout, "remote", "get-url", "origin"),
        "untracked": untracked.splitlines() if untracked else [],
    }
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record))
    return 0


def diff(repo, ref, output, start):
    sha = _git(repo, "rev-parse", "--verify", ref + "^{commit}")
    if sha is None:
        print(f"{ref} is not a commit in {repo}", file=sys.stderr)
        return 6
    for rev in {"base", start}:
        if _git(repo, "merge-base", "--is-ancestor", rev, sha) is None:
            print(f"{ref} does not descend from {rev}", file=sys.stderr)
            return 6
    with open(_record_path(repo), encoding="utf-8") as f:
        record = json.load(f)
    prefix = "" if record["prefix"] == "." else record["prefix"] + "/"
    outside = [
        f for f in _git(repo, "diff", "--name-only", start, sha).split()
        if not f.startswith(prefix + PKG + "/")
    ]
    if outside:
        print(f"{ref} changes files outside {prefix}{PKG}/: {outside}",
              file=sys.stderr)
        return 6
    relative = ["--relative=" + prefix] if prefix else []
    patch = _git(repo, "diff", *relative, start, sha)
    with open(output, "w", encoding="utf-8") as f:
        f.write(patch + "\n" if patch else "")
    out = {
        "image": record["image"],
        "repo": record["repo"],
        "base_sha": record["base_sha"],
        "ref": ref,
        "sha": sha,
    }
    if start != "base":
        out["from_sha"] = _git(repo, "rev-parse", start)
    print(json.dumps(out))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="on the image: bundle the plugin's git")
    p.add_argument("--output", default="plugin.bundle")
    p = sub.add_parser("base", help="fetch the bundle and tag its sha base")
    p.add_argument("repo")
    p.add_argument("probe_dir")
    p.add_argument("--image", required=True)
    p = sub.add_parser("local", help="a local copy of the plugin as its repo")
    p.add_argument("checkout")
    p.add_argument("--image", required=True)
    p.add_argument("--path", help="the checkout's path in the image")
    p = sub.add_parser("diff", help="a ref as a patch")
    p.add_argument("repo")
    p.add_argument("ref")
    p.add_argument("output")
    p.add_argument("--from", dest="start", default="base")
    args = parser.parse_args()
    if args.cmd == "probe":
        return probe(args.output)
    if args.cmd == "base":
        return base(args.repo, args.probe_dir, args.image)
    if args.cmd == "local":
        return local(args.checkout, args.image, args.path)
    return diff(args.repo, args.ref, args.output, args.start)


if __name__ == "__main__":
    sys.exit(main())
