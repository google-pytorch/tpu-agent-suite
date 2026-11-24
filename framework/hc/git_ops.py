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
"""Detached bare-repository Git operations and worktree hygiene for `runtime.py hc`."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import uuid
from typing import Any, Sequence

from framework import jsonio, utils
from framework.hc import constants, paths, schemas

Path = pathlib.Path

WORKTREE_SUBDIR: str = paths.WORKTREE_SUBDIR
_GIT_IDENTITY_ENV = {
    "GIT_AUTHOR_NAME": "TPU agent suite",
    "GIT_AUTHOR_EMAIL": "hc@tpu-agent-suite.local",
    "GIT_COMMITTER_NAME": "TPU agent suite",
    "GIT_COMMITTER_EMAIL": "hc@tpu-agent-suite.local",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "LC_ALL": "C",
}


def get_worktree_dir(workspace: str | Path) -> Path:
    """Return the detached source worktree directory (`<workspace>/src`)."""
    return Path(workspace) / WORKTREE_SUBDIR


def get_worktree_state_path(campaign_id: int | str) -> Path:
    """Return `campaigns/<campaign-id>/git/worktree_state.json`."""
    return paths.get_campaign_git_dir(
        campaign_id) / constants.WORKTREE_STATE_FILENAME


def get_git_index_path(campaign_id: int | str) -> Path:
    """Return `campaigns/<campaign-id>/git/index`."""
    return paths.get_campaign_git_dir(campaign_id) / "index"


def _run_git(
    git_dir: Path,
    args: Sequence[str],
    *,
    work_tree: Path | None = None,
    index_file: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    """Invoke `git` against `git_dir` with an explicit index and optional worktree."""
    resolved_git = git_dir.resolve()
    resolved_wt = work_tree.resolve() if work_tree is not None else None
    resolved_idx = (index_file.resolve() if index_file is not None else
                    (resolved_git / "index"))
    cmd = ["git", f"--git-dir={resolved_git}"]
    if resolved_wt is not None:
        cmd.append(f"--work-tree={resolved_wt}")
    cmd.extend(args)

    env = os.environ.copy()
    env.update(_GIT_IDENTITY_ENV)
    env["GIT_INDEX_FILE"] = str(resolved_idx)
    # Prevent ambient git environment variables from overriding explicit flags.
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)

    run_cwd = (resolved_wt if resolved_wt is not None and resolved_wt.is_dir()
               else resolved_git if resolved_git.is_dir() else None)
    proc = subprocess.run(cmd,
                          cwd=run_cwd,
                          capture_output=True,
                          env=env,
                          check=False)
    if check and proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise constants.GitOperationError(
            f"git {' '.join(args[:3])} failed (exit {proc.returncode}): {stderr}"
        )
    return proc


def validate_and_canonicalize_flags_file(flags_path: Path) -> dict[str, Any]:
    """Validate `flags_path` as canonical `__flags.json`, rewriting canonical form if needed."""
    if flags_path.is_symlink() or not flags_path.is_file():
        raise constants.ValidationError(
            f"missing required {constants.FLAGS_FILENAME} in worktree root")
    try:
        raw_text = flags_path.read_text(encoding="utf-8")
        parsed = json.loads(raw_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise constants.ValidationError(
            f"invalid JSON in {constants.FLAGS_FILENAME}: {exc}") from exc
    validated = schemas.validate_flags_dict(parsed)
    canonical_text = schemas.format_canonical_flags(validated)
    if raw_text != canonical_text:
        jsonio.write_text_atomic(flags_path, canonical_text, mkdir=False)
    return validated


def _file_has_conflict_marker(path: Path) -> int | None:
    """Return 1-based line number of first unresolved conflict marker, or None."""
    data = path.read_bytes()
    if b"\x00" in data:
        return None  # Binary blob; git never writes conflict markers to binary files.
    # Only flag `<<<<<<< ` / `>>>>>>> ` delimiters; a standalone `=======` line
    # underlines 7-character headings in reStructuredText and setext Markdown.
    for idx, line in enumerate(data.splitlines(), start=1):
        stripped = line.rstrip(b"\r")
        if stripped.startswith(b"<<<<<<< ") or stripped.startswith(
                b">>>>>>> "):
            return idx
    return None


def verify_worktree_hygiene(
    src_dir: Path,
    *,
    conflicted_paths: Sequence[str] = (),
    canonicalize_flags: bool = True,
) -> None:
    """Verify `<workspace>/src/` has no `.git`, no symlinks, no unresolved conflicts, and valid `__flags.json`."""
    if src_dir.is_symlink():
        raise constants.ValidationError(
            f"symlink forbidden at worktree src: {src_dir}")
    if not src_dir.is_dir():
        raise constants.ValidationError(
            f"worktree src directory not found: {src_dir}")
    for dirpath, dirnames, filenames in os.walk(src_dir, followlinks=False):
        current = Path(dirpath)
        if ".git" in dirnames or ".git" in filenames:
            raise constants.ValidationError(
                f"'.git' entry is forbidden inside worktree: {current / '.git'}"
            )
        for name in (*dirnames, *filenames):
            entry = current / name
            if entry.is_symlink():
                raise constants.ValidationError(
                    f"symlink forbidden inside worktree src: {entry}")

    for rel_str in conflicted_paths:
        try:
            rel = utils.safe_relative(rel_str, "conflicted_paths")
        except utils.TpuAgentSuiteError as exc:
            raise constants.GitOperationError(str(exc)) from exc
        fpath = src_dir / rel
        if fpath.is_file():
            marker_line = _file_has_conflict_marker(fpath)
            if marker_line is not None:
                raise constants.ValidationError(
                    f"unresolved conflict marker in {rel.as_posix()}:{marker_line}"
                )

    flags_path = src_dir / constants.FLAGS_FILENAME
    if canonicalize_flags:
        validate_and_canonicalize_flags_file(flags_path)
    else:
        if flags_path.is_symlink() or not flags_path.is_file():
            raise constants.ValidationError(
                f"missing required {constants.FLAGS_FILENAME} in worktree root"
            )
        try:
            schemas.validate_flags_dict(
                json.loads(flags_path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise constants.ValidationError(
                f"invalid JSON in {constants.FLAGS_FILENAME}: {exc}") from exc


def read_worktree_state(campaign_id: int | str) -> dict[str, Any]:
    """Load and validate `git/worktree_state.json`."""
    state_path = get_worktree_state_path(campaign_id)
    if not state_path.is_file():
        raise constants.InvalidStateError(
            f"missing worktree_state.json for campaign {campaign_id}")
    data = jsonio.load_json(state_path)
    if not isinstance(data, dict):
        raise constants.GitOperationError(
            "worktree_state.json must be a JSON object")
    parents = data.get("staged_parents")
    prepared_tree = data.get("prepared_tree_sha")
    parent_ids = data.get("staged_parent_ids")
    if (not isinstance(parents, list) or not (1 <= len(parents) <= 2)
            or not all(isinstance(p, str) and p for p in parents)
            or not isinstance(prepared_tree, str) or not prepared_tree):
        raise constants.GitOperationError("corrupt worktree_state.json")
    if parent_ids is not None:
        if (not isinstance(parent_ids, list) or len(parent_ids) != len(parents)
                or not all(
                    isinstance(pid, int) and not isinstance(pid, bool)
                    for pid in parent_ids)):
            raise constants.GitOperationError(
                "corrupt staged_parent_ids in worktree_state.json")
    raw_conflicts = data.get("conflicted_paths", [])
    if not isinstance(raw_conflicts, list) or not all(
            isinstance(p, str) and bool(p) for p in raw_conflicts):
        raise constants.GitOperationError(
            "corrupt conflicted_paths in worktree_state.json")
    try:
        conflicted_paths = [
            utils.safe_relative(p, "conflicted_paths").as_posix()
            for p in raw_conflicts
        ]
    except utils.TpuAgentSuiteError as exc:
        raise constants.GitOperationError(str(exc)) from exc
    return {
        "staged_parents": list(parents),
        "staged_parent_ids":
        (list(parent_ids) if parent_ids is not None else None),
        "prepared_tree_sha": prepared_tree,
        "conflicted_paths": conflicted_paths,
    }


def write_worktree_state(
        campaign_id: int | str,
        staged_parents: Sequence[str],
        prepared_tree_sha: str,
        *,
        staged_parent_ids: Sequence[int] | None = None,
        conflicted_paths: Sequence[str] = (),
) -> None:
    """Atomically persist `git/worktree_state.json`."""
    payload: dict[str, Any] = {
        "staged_parents": list(staged_parents),
        "prepared_tree_sha": prepared_tree_sha,
        "conflicted_paths": list(conflicted_paths),
    }
    if staged_parent_ids is not None:
        payload["staged_parent_ids"] = [int(pid) for pid in staged_parent_ids]
    jsonio.write_json(get_worktree_state_path(campaign_id), payload)


def verify_init_worktree_inputs(code_dir: Path, workspace: Path) -> bool:
    """Validate `--code-dir` and `<workspace>/src/` before initializing a campaign repo.

    Returns True when `<workspace>/src/` does not yet hold user files and should
    be removed if initialization rolls back, or False for an in-place
    `--code-dir == <workspace>/src` initialization so rollback never deletes user code.
    """
    if workspace.is_symlink():
        raise constants.ValidationError(f"workspace is a symlink: {workspace}")
    paths.verify_no_symlinks(code_dir, "--code-dir", ignore_git_dir=True)
    src_dir = get_worktree_dir(workspace)
    if src_dir.is_symlink() or src_dir.parent.is_symlink():
        raise constants.ValidationError(
            f"worktree path is a symlink: {src_dir}")
    if src_dir.exists() and not src_dir.is_dir():
        raise constants.ValidationError(
            f"worktree path must be a directory, got file: {src_dir}")
    code_resolved = code_dir.resolve()
    src_resolved = src_dir.resolve()
    cleanup_src = False
    if code_resolved != src_resolved:
        if src_resolved.is_relative_to(code_resolved):
            raise constants.ValidationError(
                f"worktree directory ({src_dir}) cannot be nested inside --code-dir ({code_dir})"
            )
        if src_dir.exists() and any(src_dir.iterdir()):
            raise constants.InvalidStateError(
                f"worktree directory {src_dir} already exists and is non-empty"
            )
        cleanup_src = True
    flags_file = code_dir / constants.FLAGS_FILENAME
    if flags_file.exists():
        schemas.validate_flags_dict(
            schemas.load_json_object(flags_file, constants.FLAGS_FILENAME))
    return cleanup_src


def _copy_code_dir_into_worktree(code_dir: Path, workspace: Path) -> Path:
    """Copy `code_dir` into `<workspace>/src/`, omitting any `.git` entries."""
    # Re-checked here to guard direct `init_campaign_repo` callers outside `open_campaign`.
    verify_init_worktree_inputs(code_dir, workspace)
    src_dir = get_worktree_dir(workspace)
    if code_dir.resolve() == src_dir.resolve():
        # Strip any top-level or nested `.git` metadata in place so `<workspace>/src`
        # functions purely as a detached working tree.
        for dirpath, dirnames, filenames in os.walk(src_dir,
                                                    followlinks=False):
            current = Path(dirpath)
            if ".git" in dirnames:
                shutil.rmtree(current / ".git")
                dirnames.remove(".git")
            if ".git" in filenames:
                (current / ".git").unlink()
        return src_dir
    shutil.copytree(
        code_dir,
        src_dir,
        symlinks=False,
        ignore=shutil.ignore_patterns(".git"),
        dirs_exist_ok=True,
    )
    return src_dir


def pin_candidate_ref(campaign_id: int | str, candidate_id: int,
                      sha: str) -> None:
    """Pin `refs/candidates/<candidate_id>` to `sha` so `git gc` never prunes it."""
    git_dir = paths.get_campaign_git_dir(campaign_id)
    _run_git(git_dir, ["update-ref", f"refs/candidates/{candidate_id}", sha])


def _git_diff(
        campaign_id: int | str,
        base_sha: str,
        target_sha: str,
        *,
        binary: bool = False,
        pathspecs: Sequence[str] = (),
) -> bytes:
    """Run deterministic `git diff` between two commits in the bare repository."""
    git_dir = paths.get_campaign_git_dir(campaign_id)
    args = ["diff"]
    if binary:
        args.append("--binary")
    args.extend([
        "--full-index",
        "--no-renames",
        "--no-ext-diff",
        "--no-color",
        base_sha,
        target_sha,
    ])
    if pathspecs:
        args.extend(["--", *pathspecs])
    return _run_git(git_dir, args).stdout


def compute_patch_sha(
    campaign_id: int | str,
    initial_sha: str,
    candidate_sha: str,
) -> str:
    """Return SHA-256 hex digest of canonical git diff from `initial_sha` to `candidate_sha`."""
    raw = _git_diff(campaign_id, initial_sha, candidate_sha, binary=True)
    return hashlib.sha256(raw).hexdigest()


def export_code_patch(
    campaign_id: int | str,
    initial_sha: str,
    candidate_sha: str,
    dest_path: Path,
) -> Path | None:
    """Export code-only `.patch` excluding `__flags.json`; return None when code is unchanged."""
    raw = _git_diff(
        campaign_id,
        initial_sha,
        candidate_sha,
        binary=True,
        pathspecs=[".", f":(exclude){constants.FLAGS_FILENAME}"],
    )
    if not raw:
        return None
    jsonio.write_bytes_atomic(dest_path, raw)
    return dest_path


def compute_candidate_diffs(
    campaign_id: int | str,
    base_sha: str,
    candidate_sha: str,
) -> tuple[str, str]:
    """Return `(code_diff, flags_diff)` between `base_sha` and `candidate_sha`."""
    code_raw = _git_diff(
        campaign_id,
        base_sha,
        candidate_sha,
        pathspecs=[".", f":(exclude){constants.FLAGS_FILENAME}"],
    )
    flags_raw = _git_diff(
        campaign_id,
        base_sha,
        candidate_sha,
        pathspecs=[constants.FLAGS_FILENAME],
    )
    return (
        code_raw.decode("utf-8", errors="replace"),
        flags_raw.decode("utf-8", errors="replace"),
    )


def read_flags_from_commit(campaign_id: int | str, sha: str) -> dict[str, Any]:
    """Read and validate `__flags.json` directly from a commit in the bare repository."""
    git_dir = paths.get_campaign_git_dir(campaign_id)
    proc = _run_git(git_dir, ["show", f"{sha}:{constants.FLAGS_FILENAME}"])
    try:
        parsed = json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise constants.GitOperationError(
            f"corrupt {constants.FLAGS_FILENAME} in commit {sha}: {exc}"
        ) from exc
    return schemas.validate_flags_dict(parsed)


def init_campaign_repo(
    campaign_id: int | str,
    code_dir: str | Path,
    workspace: str | Path,
) -> dict[str, str]:
    """Initialize bare git repo, populate `<workspace>/src/`, and commit `initial_version`."""
    cid = paths.validate_campaign_id(campaign_id)
    git_dir = paths.get_campaign_git_dir(cid)
    if git_dir.exists() and any(git_dir.iterdir()):
        raise constants.InvalidStateError(
            f"bare git repository already exists for campaign {cid}: {git_dir}"
        )

    src_dir = _copy_code_dir_into_worktree(Path(code_dir), Path(workspace))

    flags_path = src_dir / constants.FLAGS_FILENAME
    if not flags_path.exists():
        jsonio.write_text_atomic(flags_path, "{}\n", mkdir=False)

    verify_worktree_hygiene(src_dir)

    git_dir.mkdir(parents=True, exist_ok=True)
    _run_git(git_dir, ["init", "--bare", "--quiet"])
    _run_git(git_dir, ["add", "--force", "-A"], work_tree=src_dir)
    tree_sha = _run_git(git_dir, ["write-tree"]).stdout.decode("ascii").strip()
    sha = _run_git(
        git_dir,
        ["commit-tree", tree_sha, "-m", "initial_version"],
    ).stdout.decode("ascii").strip()

    patch_sha = compute_patch_sha(cid, sha, sha)
    write_worktree_state(cid, [sha], tree_sha)
    return {
        "sha": sha,
        "tree_sha": tree_sha,
        "patch_sha": patch_sha,
    }


def _strip_worktree_anomalies(src_dir: Path) -> None:
    """Remove any symlinks or `.git` entries inside `src_dir` before tree reset."""
    if src_dir.is_symlink():
        src_dir.unlink()
        src_dir.mkdir(parents=True, exist_ok=True)
        return
    if not src_dir.exists():
        src_dir.mkdir(parents=True, exist_ok=True)
        return
    for dirpath, dirnames, filenames in os.walk(src_dir, followlinks=False):
        current = Path(dirpath)
        for dname in list(dirnames):
            entry = current / dname
            if entry.is_symlink():
                entry.unlink()
                dirnames.remove(dname)
            elif dname == ".git":
                shutil.rmtree(entry)
                dirnames.remove(dname)
        for fname in list(filenames):
            entry = current / fname
            if entry.is_symlink() or fname == ".git":
                entry.unlink()


def worktree_has_uncommitted_changes(
    campaign_id: int | str,
    workspace: str | Path,
) -> bool:
    """Return True if `<workspace>/src/` differs from `worktree_state.prepared_tree_sha`."""
    cid = paths.validate_campaign_id(campaign_id)
    state = read_worktree_state(cid)
    src_dir = get_worktree_dir(workspace)
    if src_dir.is_symlink() or not src_dir.is_dir():
        return True
    for dirpath, dirnames, filenames in os.walk(src_dir, followlinks=False):
        if ".git" in dirnames or ".git" in filenames:
            return True
        current = Path(dirpath)
        if any(
            (current / name).is_symlink() for name in (*dirnames, *filenames)):
            return True

    git_dir = paths.get_campaign_git_dir(cid)
    main_index = get_git_index_path(cid)
    tmp_index = git_dir / f".check_index.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    try:
        if main_index.is_file():
            shutil.copy2(main_index, tmp_index)
        _run_git(git_dir, ["add", "--force", "-A"],
                 work_tree=src_dir,
                 index_file=tmp_index)
        current_tree = _run_git(
            git_dir, ["write-tree"],
            index_file=tmp_index).stdout.decode("ascii").strip()
    finally:
        tmp_index.unlink(missing_ok=True)
    return current_tree != state["prepared_tree_sha"]


def _merge_trees(
    git_dir: Path,
    p1_sha: str,
    p2_sha: str,
) -> tuple[str, list[str], dict[str, str]]:
    """Merge `p1_sha` and `p2_sha` via `git merge-tree --write-tree -z`.

    Returns `(tree_sha, conflicted_paths, modify_delete_markers)` where
    `modify_delete_markers` maps each text modify/delete path to its explicit
    conflict-marker payload (`merge-tree` already writes conflict markers into
    `tree_sha` for content and add/add conflicts, while leaving modify/delete
    files unmarked). Binary and file/directory conflicts raise
    `GitOperationError` before touching the worktree.
    """
    proc = _run_git(
        git_dir,
        ["merge-tree", "--write-tree", "-z", p1_sha, p2_sha],
        check=False,
    )
    if proc.returncode not in (0, 1):
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise constants.GitOperationError(
            f"cannot merge {p1_sha[:12]} and {p2_sha[:12]}: {stderr or 'unrelated histories'}"
        )

    parts = proc.stdout.split(b"\x00")
    tree_sha = parts[0].decode("ascii").strip()
    if proc.returncode == 0:
        return tree_sha, [], {}

    # `merge-tree -z` emits: tree_oid \0 (<mode> <oid> <stage>\t<path> \0)* \0 (<n> \0 <path>*n <type> \0 <msg> \0)*
    sep = parts.index(b"", 1) if b"" in parts[1:] else len(parts)
    unmerged: dict[str, dict[int, str]] = {}
    for entry in parts[1:sep]:
        meta_bytes, path_bytes = entry.split(b"\t", 1)
        _mode, blob_sha, stage_str = meta_bytes.decode("ascii").split()
        unmerged.setdefault(path_bytes.decode("utf-8"),
                            {})[int(stage_str)] = blob_sha

    conflicted_paths: list[str] = list(unmerged)
    modify_delete_paths: list[str] = []
    idx = sep + 1
    while idx < len(parts) and parts[idx]:
        count = int(parts[idx].decode("ascii"))
        cpaths = [p.decode("utf-8") for p in parts[idx + 1:idx + 1 + count]]
        ctype = parts[idx + 1 + count].decode("utf-8", errors="replace")
        detail = parts[idx + 2 + count].decode("utf-8",
                                               errors="replace").strip()
        idx += 3 + count
        if not ctype.startswith("CONFLICT"):
            continue
        if ctype in ("CONFLICT (contents)", "CONFLICT (content)",
                     "CONFLICT (add/add)"):
            conflicted_paths.extend(cpaths)
            continue
        if ctype == "CONFLICT (modify/delete)" and len(cpaths) == 1:
            conflicted_paths.append(cpaths[0])
            modify_delete_paths.append(cpaths[0])
            continue
        raise constants.GitOperationError(detail or f"unmergeable {ctype}")

    markers: dict[str, str] = {}
    for rel_path in modify_delete_paths:
        stages = unmerged.get(rel_path, {})
        ours = (_run_git(git_dir, ["cat-file", "blob", stages[2]]).stdout
                if 2 in stages else None)
        theirs = (_run_git(git_dir, ["cat-file", "blob", stages[3]]).stdout
                  if 3 in stages else None)
        surviving = ours if ours is not None else theirs
        if surviving is not None and b"\x00" in surviving:
            raise constants.GitOperationError(
                f"cannot 3-way merge modify/delete conflict on binary file: {rel_path}"
            )
        ours_text = (ours.decode("utf-8", errors="replace")
                     if ours is not None else "")
        theirs_text = (theirs.decode("utf-8", errors="replace")
                       if theirs is not None else "")
        if ours_text and not ours_text.endswith("\n"):
            ours_text += "\n"
        if theirs_text and not theirs_text.endswith("\n"):
            theirs_text += "\n"
        ours_tag = (f"parent1:{p1_sha[:12]}" if ours is not None else
                    f"parent1:{p1_sha[:12]} (deleted)")
        theirs_tag = (f"parent2:{p2_sha[:12]}" if theirs is not None else
                      f"parent2:{p2_sha[:12]} (deleted)")
        markers[rel_path] = (f"<<<<<<< {ours_tag}\n{ours_text}"
                             f"=======\n{theirs_text}"
                             f">>>>>>> {theirs_tag}\n")
    return tree_sha, sorted(set(conflicted_paths)), markers


def prepare_worktree(
    campaign_id: int | str,
    workspace: str | Path,
    parent_shas: Sequence[str],
    *,
    parent_ids: Sequence[int] | None = None,
    discard: bool = False,
) -> dict[str, Any]:
    """Populate `<workspace>/src/` from 1 or 2 parent commits; return state + `has_conflicts`."""
    cid = paths.validate_campaign_id(campaign_id)
    parents = list(parent_shas)
    if not (1 <= len(parents) <= 2):
        raise constants.ValidationError(
            f"candidate prepare requires 1 or 2 parents, got {len(parents)}")
    if len(parents) == 2 and parents[0] == parents[1]:
        raise constants.ValidationError(
            "candidate prepare requires distinct parent commits")
    if parent_ids is not None and len(parent_ids) != len(parents):
        raise constants.ValidationError(
            "parent_ids length must match parent_shas length")

    ws_path = Path(workspace)
    src_dir = get_worktree_dir(ws_path)
    if not discard and worktree_has_uncommitted_changes(cid, ws_path):
        raise constants.InvalidStateError(
            "uncommitted modifications exist in <workspace>/src/; pass --discard to overwrite"
        )

    git_dir = paths.get_campaign_git_dir(cid)
    conflicted_paths: list[str] = []
    md_markers: dict[str, str] = {}
    if len(parents) == 1:
        target_tree = parents[0]
    else:
        target_tree, conflicted_paths, md_markers = _merge_trees(
            git_dir, parents[0], parents[1])
    has_conflicts = bool(conflicted_paths)

    _strip_worktree_anomalies(src_dir)
    # Sync index to on-disk state so `read-tree --reset -u` updates only files that differ.
    _run_git(git_dir, ["add", "--force", "-A"], work_tree=src_dir)
    _run_git(git_dir, ["read-tree", "--reset", "-u", target_tree],
             work_tree=src_dir)
    _run_git(git_dir, ["clean", "-fdxq"], work_tree=src_dir)
    for rel_path, conflict_doc in md_markers.items():
        dest_file = src_dir / rel_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_text(conflict_doc, encoding="utf-8")

    # Normalize index to stage-0 snapshot of the resulting worktree (even if conflict
    # markers are present) so `prepared_tree_sha` tracks post-prepare edits accurately.
    _run_git(git_dir, ["add", "--force", "-A"], work_tree=src_dir)
    prepared_tree_sha = _run_git(
        git_dir, ["write-tree"]).stdout.decode("ascii").strip()
    write_worktree_state(
        cid,
        parents,
        prepared_tree_sha,
        staged_parent_ids=parent_ids,
        conflicted_paths=conflicted_paths,
    )
    return {
        "staged_parents": parents,
        "staged_parent_ids":
        (list(parent_ids) if parent_ids is not None else None),
        "prepared_tree_sha": prepared_tree_sha,
        "conflicted_paths": conflicted_paths,
        "has_conflicts": has_conflicts,
    }


def propose_candidate_commit(
    campaign_id: int | str,
    workspace: str | Path,
    initial_sha: str,
    *,
    message: str = "candidate",
) -> dict[str, Any]:
    """Validate `<workspace>/src/`, write a commit on `staged_parents`, and return commit metadata."""
    cid = paths.validate_campaign_id(campaign_id)
    state = read_worktree_state(cid)
    staged_parents = state["staged_parents"]
    if not (1 <= len(staged_parents) <= 2):
        raise constants.InvalidStateError(
            "worktree_state.json must contain 1 or 2 staged parents before propose"
        )

    src_dir = get_worktree_dir(workspace)
    verify_worktree_hygiene(src_dir,
                            conflicted_paths=state["conflicted_paths"],
                            canonicalize_flags=True)

    git_dir = paths.get_campaign_git_dir(cid)
    _run_git(git_dir, ["add", "--force", "-A"], work_tree=src_dir)
    tree_sha = _run_git(git_dir, ["write-tree"]).stdout.decode("ascii").strip()

    commit_args = ["commit-tree", tree_sha]
    for parent_sha in staged_parents:
        commit_args.extend(["-p", parent_sha])
    commit_args.extend(["-m", message])
    sha = _run_git(git_dir, commit_args).stdout.decode("ascii").strip()
    patch_sha = compute_patch_sha(cid, initial_sha, sha)

    # `worktree_state.json` stays untouched until `finalize_candidate_ref` runs
    # after the `Candidates` row commits in SQLite, so a rolled-back DB insert
    # never leaves `staged_parents` pointing at an unpinned commit.
    return {
        "sha": sha,
        "tree_sha": tree_sha,
        "patch_sha": patch_sha,
        "staged_parents": staged_parents,
        "staged_parent_ids": state["staged_parent_ids"],
    }


def finalize_candidate_ref(
    campaign_id: int | str,
    candidate_id: int,
    sha: str,
    tree_sha: str,
) -> None:
    """Pin `refs/candidates/<candidate_id>` and record `[candidate_id]` in `worktree_state.json`."""
    pin_candidate_ref(campaign_id, candidate_id, sha)
    write_worktree_state(
        campaign_id,
        [sha],
        tree_sha,
        staged_parent_ids=[candidate_id],
    )


def is_ancestor(
    campaign_id: int | str,
    ancestor_sha: str,
    descendant_sha: str,
) -> bool:
    """Return True iff `ancestor_sha` is an ancestor of (or equal to) `descendant_sha`."""
    git_dir = paths.get_campaign_git_dir(campaign_id)
    proc = _run_git(
        git_dir,
        ["merge-base", "--is-ancestor", ancestor_sha, descendant_sha],
        check=False,
    )
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    stderr = proc.stderr.decode("utf-8", errors="replace").strip()
    raise constants.GitOperationError(
        f"git merge-base --is-ancestor failed: {stderr}")


def candidates_on_path(
    campaign_id: int | str,
    prior_champion_sha: str,
    new_champion_sha: str,
) -> list[int]:
    """Return sorted candidate IDs whose commits lie in `git rev-list <prior>..<new>`."""
    if prior_champion_sha == new_champion_sha:
        return []
    if not is_ancestor(campaign_id, prior_champion_sha, new_champion_sha):
        raise constants.InvalidStateError(
            f"new champion {new_champion_sha[:12]} is not a descendant of "
            f"prior champion {prior_champion_sha[:12]}")

    git_dir = paths.get_campaign_git_dir(campaign_id)
    refs_proc = _run_git(
        git_dir,
        [
            "for-each-ref", "--format=%(refname) %(objectname)",
            "refs/candidates/"
        ],
    )
    sha_to_cids: dict[str, list[int]] = {}
    prefix = "refs/candidates/"
    for line in refs_proc.stdout.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        refname, obj_sha = line.split(None, 1)
        if not refname.startswith(prefix):
            continue
        cid_str = refname[len(prefix):]
        if cid_str.isdigit():
            sha_to_cids.setdefault(obj_sha.strip(), []).append(int(cid_str))

    rev_proc = _run_git(
        git_dir, ["rev-list", f"{prior_champion_sha}..{new_champion_sha}"])
    matched: set[int] = set()
    for rev_sha in rev_proc.stdout.decode("ascii").splitlines():
        for cid in sha_to_cids.get(rev_sha.strip(), ()):
            matched.add(cid)
    return sorted(matched)
