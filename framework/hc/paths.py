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
"""The layout of the Hillclimber's protected store, `~/.tpu-agent-suite/campaigns/`.

`framework.paths` knows only that the campaigns root and the sealed-jobs root
exist, since the harness sandbox rules and `compute` need those; everything
inside them is the Hillclimber's own business and is laid out here:

    campaigns/
    ├── campaign.db                 the SQLite coordination store
    ├── logs/events_unscoped.jsonl  rejections before a campaign is resolved
    ├── <campaign_id>/
    │   ├── git/                    the campaign's detached bare repository
    │   ├── scripts/                protocol scripts sealed at campaign open
    │   ├── sweeps/                 sweep working directories
    │   └── logs/events.jsonl       telemetry
    ├── sealed_jobs/<job_id>/       jobs hc.py submitted (see compute's jobs.py)
    │   └── sealed/                 their write-once outputs and log
    └── runs/<run_id>/active_campaign_id
"""

from __future__ import annotations

import os
import pathlib

from framework import jsonio, paths, utils
from framework.compute.drivers import jobs
from framework.hc import constants

Path = pathlib.Path

WORKTREE_SUBDIR = "src"
CAMPAIGN_DB_FILENAME = "campaign.db"
CAMPAIGN_SCRIPTS_DIRNAME = "scripts"
CAMPAIGN_RUNS_DIRNAME = "runs"
# The per-run file naming the active campaign. It lives under the protected
# root, not in the agent-writable run folder, so an agent cannot rebind a
# run's campaign.
ACTIVE_CAMPAIGN_FILE = "active_campaign_id"

# Re-exported so hc code has one import for every campaign path.
get_campaigns_root = paths.get_campaigns_root
get_sealed_jobs_root = paths.get_sealed_jobs_root


def validate_campaign_id(campaign_id: int | str) -> int:
    """Validate and normalize a positive integer campaign ID."""
    raw = campaign_id.strip() if isinstance(campaign_id, str) else None
    cid = (campaign_id if isinstance(campaign_id, int)
           and not isinstance(campaign_id, bool) else
           int(raw) if raw and raw.isascii() and raw.isdecimal() else 0)
    if cid <= 0:
        raise constants.ValidationError(
            f"invalid campaign_id {campaign_id!r}: expected positive integer")
    return cid


def get_campaign_db_path() -> Path:
    """Return the SQLite coordination database path (`campaigns/campaign.db`)."""
    return get_campaigns_root() / CAMPAIGN_DB_FILENAME


def get_campaign_root(campaign_id: int | str) -> Path:
    """Return the directory for `campaign_id` (`campaigns/<campaign-id>/`)."""
    return get_campaigns_root() / str(validate_campaign_id(campaign_id))


def get_campaign_git_dir(campaign_id: int | str) -> Path:
    """Return the bare git repository path for `campaign_id`."""
    return get_campaign_root(campaign_id) / "git"


def get_campaign_scripts_dir(campaign_id: int | str) -> Path:
    """Return the sealed protocol scripts directory for `campaign_id`."""
    return get_campaign_root(campaign_id) / CAMPAIGN_SCRIPTS_DIRNAME


def get_campaign_sweeps_dir(campaign_id: int | str) -> Path:
    """Return the sweeps directory for `campaign_id`."""
    return get_campaign_root(campaign_id) / "sweeps"


def get_campaign_events_log(campaign_id: int | str) -> Path:
    """Return the JSONL telemetry log path (`<campaign_id>/logs/events.jsonl`)."""
    return get_campaign_root(campaign_id) / "logs" / "events.jsonl"


def get_unscoped_events_log() -> Path:
    """Return the fallback telemetry log path (`campaigns/logs/events_unscoped.jsonl`)."""
    return get_campaigns_root() / "logs" / "events_unscoped.jsonl"


def get_sealed_job_artifacts_dir(job_id: str) -> Path:
    """Return the write-once outputs directory of a sealed job (`sealed_jobs/<job_id>/sealed/`).

    Named by `compute`, which writes it; distinct from the `artifacts/`
    directory a local workload writes into beside it.
    """
    try:
        valid_id = jobs.validate_job_id(job_id)
    except utils.TpuAgentSuiteError as exc:
        raise constants.ValidationError(str(exc)) from exc
    return get_sealed_jobs_root() / valid_id / jobs.SEALED_ARTIFACTS_DIRNAME


def get_campaign_run_dir(run_id: str) -> Path:
    """Return the per-run directory (`campaigns/runs/<run_id>/`)."""
    try:
        utils.validate_run_id(run_id)
    except utils.TpuAgentSuiteError as exc:
        raise constants.ValidationError(str(exc)) from exc
    return get_campaigns_root() / CAMPAIGN_RUNS_DIRNAME / run_id


def get_active_campaign_marker(run_id: str) -> Path:
    """Return the active-campaign pointer path for `run_id`."""
    return get_campaign_run_dir(run_id) / ACTIVE_CAMPAIGN_FILE


def record_active_campaign(run_id: str, campaign_id: int | str) -> None:
    """Record `campaign_id` as the active campaign for `run_id`."""
    cid = validate_campaign_id(campaign_id)
    jsonio.write_text_atomic(get_active_campaign_marker(run_id), f"{cid}\n")


def get_active_campaign_id(run_id: str) -> int | None:
    """Return the active campaign ID recorded for `run_id`, or None."""
    marker = get_active_campaign_marker(run_id)
    if not marker.is_file():
        return None
    try:
        return validate_campaign_id(marker.read_text(encoding="utf-8"))
    except (OSError, constants.HcError, utils.TpuAgentSuiteError):
        return None


def verify_no_symlinks(root_dir: Path,
                       label: str,
                       *,
                       ignore_git_dir: bool = False) -> None:
    """Reject any symlink at or beneath `root_dir`."""
    if root_dir.is_symlink() or not root_dir.is_dir():
        raise constants.ValidationError(
            f"{label} must be an existing non-symlink directory: {root_dir}")
    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=False):
        if ignore_git_dir and ".git" in dirnames:
            dirnames.remove(".git")
        current = Path(dirpath)
        for name in (*dirnames, *filenames):
            if ignore_git_dir and name == ".git":
                continue
            if (current / name).is_symlink():
                raise constants.ValidationError(
                    f"symlink forbidden inside {label}: {current / name}")


def verify_workspace_subpath(workspace: Path, target_path: Path,
                             label: str) -> Path:
    """Verify `target_path` contains no symlinks and resolves strictly inside `workspace`."""
    if workspace.is_symlink() or not workspace.is_dir():
        raise constants.ValidationError(
            f"workspace must be an existing non-symlink directory: {workspace}"
        )
    ws_abs = Path(os.path.abspath(workspace))
    ws_resolved = workspace.resolve()
    if target_path.is_absolute():
        abs_target = Path(os.path.abspath(target_path))
        rel = (abs_target.relative_to(ws_abs)
               if abs_target.is_relative_to(ws_abs) else
               abs_target.relative_to(ws_resolved)
               if abs_target.is_relative_to(ws_resolved) else None)
        if rel is None:
            raise constants.ValidationError(
                f"{label} ({target_path}) must resolve strictly within workspace ({ws_resolved})"
            )
    else:
        rel = target_path
    try:
        candidate = utils.rooted(ws_resolved, rel, label)
    except utils.TpuAgentSuiteError as exc:
        raise constants.ValidationError(str(exc)) from exc
    rel_parts = candidate.relative_to(ws_resolved).parts
    if not rel_parts:
        raise constants.ValidationError(
            f"{label} ({target_path}) must resolve strictly within workspace ({ws_resolved})"
        )
    cur = ws_resolved
    for part in rel_parts:
        cur = cur / part
        if cur.is_symlink():
            raise constants.ValidationError(
                f"symlink forbidden along {label}: {cur}")
    resolved_target = candidate.resolve()
    src_resolved = (ws_resolved / WORKTREE_SUBDIR).resolve()
    if (resolved_target == src_resolved
            or resolved_target.is_relative_to(src_resolved)):
        raise constants.ValidationError(
            f"{label} ({target_path}) must not point inside {WORKTREE_SUBDIR}/"
        )
    if resolved_target.exists():
        verify_no_symlinks(resolved_target, label)
    return resolved_target
