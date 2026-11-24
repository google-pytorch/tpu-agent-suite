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
"""Campaign lifecycle operations and shared DB lookup helpers for `runtime.py hc`."""

from __future__ import annotations

import os
import pathlib
import shutil
import sqlite3
from typing import Any, Callable

from framework import items, jsonio
from framework import paths as base_paths
from framework import utils
from framework.hc import constants, db, git_ops, paths, schemas, telemetry

Path = pathlib.Path
_HC_DIR = pathlib.Path(__file__).resolve().parent
_REPO_OR_PKG_ROOT = _HC_DIR.parents[1]
TARGET_RUNNER_FILENAME = "target_runner.py"
SEALED_MANIFEST_FILENAME = "manifest.json"

# Optional hook registered by `jobs_ops` to cancel live runtime jobs
# and detached sweep wrappers before transitioning remaining Dispatched rows.
JobCanceller = Callable[
    [sqlite3.Connection, int, Path, str, telemetry.TelemetryBuffer], None]
_job_canceller: JobCanceller | None = None


def register_job_canceller(canceller: JobCanceller) -> None:
    """Register the live job/sweep cancellation callback used on campaign close."""
    global _job_canceller
    _job_canceller = canceller


def cancel_dispatched_jobs(
    conn: sqlite3.Connection,
    campaign_id: int,
    workspace: Path,
    run_id: str,
    telem: telemetry.TelemetryBuffer,
) -> list[str]:
    """Cancel remaining `Dispatched` jobs for `campaign_id` on campaign close."""
    rows = db.exec_query(
        conn,
        "list_dispatched_jobs",
        campaign_id=campaign_id,
        status=constants.JobStatus.DISPATCHED.value,
    ).fetchall()
    if not rows:
        return []
    handled_ids = [str(r["job_id"]) for r in rows]
    if _job_canceller is not None:
        _job_canceller(conn, campaign_id, workspace, run_id, telem)
        rows = db.exec_query(
            conn,
            "list_dispatched_jobs",
            campaign_id=campaign_id,
            status=constants.JobStatus.DISPATCHED.value,
        ).fetchall()
    if rows:
        with db.transaction(conn):
            for row in rows:
                db.exec_query(
                    conn,
                    "update_job_status",
                    status=constants.JobStatus.CANCELLED.value,
                    job_id=row["job_id"],
                )
                telem.record(
                    "job.settled",
                    round_id=row["round"],
                    campaign_id=campaign_id,
                    job_id=row["job_id"],
                    kind=row["kind"],
                    from_status=constants.JobStatus.DISPATCHED.value,
                    to_status=constants.JobStatus.CANCELLED.value,
                    elapsed_s=None,
                )
    return handled_ids


def insert_protocol(conn: sqlite3.Connection, validated: dict[str,
                                                              Any]) -> int:
    """Insert a `Protocols` row and its `Protocol_Cells` child rows; return `protocol_id`."""
    gate = validated["accuracy_gate"]
    exec_spec = validated["execution_spec"]
    cur = db.exec_query(
        conn,
        "insert_protocol",
        workload_kind=validated["workload_kind"],
        primary_metric=validated["primary_metric"],
        compute_profile=validated["compute_profile"],
        primary_metric_unit=validated["primary_metric_unit"],
        primary_metric_direction=validated["primary_metric_direction"],
        primary_metric_statistic=validated["primary_metric_statistic"],
        min_replicates=validated["min_replicates"],
        min_win_percentage=validated["min_win_percentage"],
        max_cell_regression_percentage=validated[
            "max_cell_regression_percentage"],
        noise_floor_percentage=validated["noise_floor_percentage"],
        max_rounds=validated["max_rounds"],
        max_barren_rounds=validated["max_barren_rounds"],
        target_improvement_percent=validated["target_improvement_percent"],
        has_accuracy_gate=1 if gate is not None else 0,
        accuracy_gate_metric=gate["metric"] if gate is not None else None,
        accuracy_gate_comparator=gate["comparator"]
        if gate is not None else None,
        accuracy_gate_threshold=gate["threshold"]
        if gate is not None else None,
        accuracy_gate_is_relative=(1 if gate["is_relative"] else 0)
        if gate is not None else None,
        accuracy_gate_relative_to=gate["relative_to"]
        if gate is not None else None,
        exec_image=exec_spec["image"],
        exec_microbenchmark_target=exec_spec["microbenchmark_target"],
        exec_base_server_args=exec_spec["base_server_args"],
        exec_base_client_args=exec_spec["base_client_args"],
        exec_base_eval_args=exec_spec["base_eval_args"],
    )
    protocol_id = int(cur.lastrowid)
    for idx, cell in enumerate(validated["measurement_cells"]):
        db.exec_query(
            conn,
            "insert_protocol_cell",
            protocol_id=protocol_id,
            cell_index=idx,
            topology=cell["topology"],
            weight=cell["weight"],
            prefill=cell["prefill"],
            decode=cell["decode"],
            concurrency=cell["concurrency"],
            sequence_length=cell["sequence_length"],
            global_batch_size=cell["global_batch_size"],
            per_device_batch_size=cell["per_device_batch_size"],
        )
    return protocol_id


def insert_candidate(
    conn: sqlite3.Connection,
    *,
    campaign_id: int,
    round_id: int | None,
    sha: str,
    tree_sha: str,
    patch_sha: str,
    short_innovation: str,
    innovation: str,
) -> int:
    """Insert a `Candidates` row and return its primary key."""
    cur = db.exec_query(
        conn,
        "insert_candidate",
        campaign_id=campaign_id,
        round=round_id,
        sha=sha,
        tree_sha=tree_sha,
        patch_sha=patch_sha,
        short_innovation=short_innovation,
        innovation=innovation,
    )
    return int(cur.lastrowid)


def get_campaign_for_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    allow_closed: bool = False,
) -> sqlite3.Row:
    """Fetch the `Campaigns` row recorded for `run_id`."""
    cid = paths.get_active_campaign_id(run_id)
    if cid is None:
        raise constants.InvalidStateError(
            f"no campaign is active in run {run_id!r}; run `runtime.py hc campaign open` first"
        )
    row = db.exec_query(conn, "get_campaign_by_id", id=cid).fetchone()
    if row is None:
        raise constants.NotFoundError(
            f"active campaign {cid} recorded in run {run_id!r} does not exist in campaign.db"
        )
    if not allow_closed and row[
            "state"] == constants.CampaignState.CLOSED.value:
        raise constants.InvalidStateError(
            f"campaign {cid} is Closed (close_reason={row['close_reason']})")
    return row


def get_open_round(conn: sqlite3.Connection,
                   campaign_id: int) -> sqlite3.Row | None:
    """Return the currently `Open` round row for `campaign_id`, or None."""
    return db.exec_query(
        conn,
        "get_open_round",
        campaign_id=campaign_id,
        state=constants.RoundState.OPEN.value,
    ).fetchone()


def require_active_campaign(conn: sqlite3.Connection,
                            run_id: str) -> sqlite3.Row:
    """Return the active `Campaigns` row, requiring `state == Active` and a non-null `champion`."""
    campaign = get_campaign_for_run(conn, run_id)
    if campaign["state"] != constants.CampaignState.ACTIVE.value:
        raise constants.InvalidStateError(
            f"operation requires Active campaign, got {campaign['state']}")
    if campaign["champion"] is None:
        raise constants.InvalidStateError(
            f"campaign {campaign['id']} has no champion; run `runtime.py hc campaign set-baseline` first"
        )
    return campaign


def require_open_round(conn: sqlite3.Connection,
                       campaign_id: int) -> sqlite3.Row:
    """Return the `Open` round row for `campaign_id`, raising `InvalidStateError` if none."""
    rnd = get_open_round(conn, campaign_id)
    if rnd is None:
        raise constants.InvalidStateError(
            f"no open round in campaign {campaign_id}; run `runtime.py hc round open` first"
        )
    return rnd


def bind_telemetry(
    conn: sqlite3.Connection,
    telem: telemetry.TelemetryBuffer,
    campaign_id: int,
) -> None:
    """Populate `telem` campaign and open-round context if not already bound."""
    if telem.campaign_id != campaign_id:
        telem.campaign_id = campaign_id
        open_rnd = get_open_round(conn, campaign_id)
        telem.round_id = int(open_rnd["id"]) if open_rnd is not None else None


def resolve_package_tag(
    run_id: str,
    explicit_package: str | None = None,
) -> str | None:
    """Resolve package tag from CLI override, `run.json`, or installed release metadata."""
    if explicit_package and explicit_package.strip():
        return explicit_package.strip()
    pkg = items.resolve_package(run_id).strip()
    if pkg:
        return pkg
    for meta_rel in ("package.release.json", "package/package.release.json",
                     "tpu-agent-suite-build.json"):
        rel_doc = jsonio.load_json_or_empty(_REPO_OR_PKG_ROOT / meta_rel)
        rel_name = str(rel_doc.get("package") or rel_doc.get("name")
                       or "").strip()
        rel_ver = str(rel_doc.get("version") or "").strip()
        if rel_name:
            return f"{rel_name}@{rel_ver}" if rel_ver else rel_name
    env_pkg = (os.environ.get("TPU_AGENT_SUITE_PACKAGE") or "").strip()
    return env_pkg or None


def _verify_contained_file(root: Path, rel_str: str, label: str) -> Path:
    """Verify `root / rel_str` has no symlinks and resolves to a regular file inside `root`."""
    if root.is_symlink() or not root.is_dir():
        raise constants.ValidationError(
            f"{label} root must be an existing non-symlink directory: {root}")
    try:
        rel = utils.safe_relative(rel_str, label)
        cand = utils.rooted(root, rel, label)
    except utils.TpuAgentSuiteError as exc:
        raise constants.ValidationError(str(exc)) from exc
    cur = root
    for part in rel.parts:
        cur = cur / part
        if cur.is_symlink():
            raise constants.ValidationError(
                f"symlink forbidden along {label}: {cur}")
        if not cur.exists():
            raise constants.ValidationError(f"{label} not found: {cand}")
    if not cand.is_file():
        raise constants.ValidationError(
            f"{label} must be a regular file, got: {cand}")
    return cand.resolve()


def _verify_workspace_file(workspace: Path, rel_str: str, label: str) -> Path:
    """Verify `workspace / rel_str` is a regular file inside `workspace` and outside `src/`."""
    resolved = _verify_contained_file(workspace, rel_str, label)
    src_resolved = (workspace.resolve() / paths.WORKTREE_SUBDIR).resolve()
    if resolved == src_resolved or resolved.is_relative_to(src_resolved):
        raise constants.ValidationError(
            f"{label} ({rel_str}) must not reside inside {paths.WORKTREE_SUBDIR}/"
        )
    return resolved


def resolve_package_hillclimber(
    run_id: str,
    explicit_package: str | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Return `(assets_root, hc_framework_dir, hillclimber_cfg)` from build or repo manifest."""
    pkg_tag = resolve_package_tag(run_id, explicit_package)
    if pkg_tag:
        try:
            pkg_root = base_paths.get_installed_package_root(pkg_tag)
        except utils.TpuAgentSuiteError as exc:
            pkg_root = None
            explicit_clean = (explicit_package or "").strip()
            repo_manifest = jsonio.load_json_or_empty(
                _REPO_OR_PKG_ROOT / "package" / "package.release.json")
            repo_name = str(repo_manifest.get("name") or "").strip()
            if explicit_clean and utils.parse_package_tag(
                    explicit_clean)[0] != repo_name:
                raise constants.ValidationError(str(exc)) from exc
        if pkg_root is not None:
            build_doc = jsonio.load_json_or_empty(pkg_root /
                                                  "tpu-agent-suite-build.json")
            hc_cfg = build_doc.get("hillclimber")
            if not isinstance(hc_cfg, dict):
                raise constants.ValidationError(
                    f"installed package at {pkg_root} does not declare 'hillclimber' scripts"
                )
            hc_dir = pkg_root / "framework" / "hc"
            return pkg_root, (hc_dir if hc_dir.is_dir() else _HC_DIR), hc_cfg

    if (_REPO_OR_PKG_ROOT / "tpu-agent-suite-build.json").is_file():
        doc = jsonio.load_json_or_empty(_REPO_OR_PKG_ROOT /
                                        "tpu-agent-suite-build.json")
        assets_root = _REPO_OR_PKG_ROOT
    else:
        assets_root = _REPO_OR_PKG_ROOT / "package"
        doc = jsonio.load_json_or_empty(assets_root / "package.release.json")
    hc_cfg = doc.get("hillclimber")
    if not isinstance(hc_cfg, dict):
        raise constants.ValidationError(
            "active package manifest does not declare 'hillclimber' scripts")
    return assets_root, _HC_DIR, hc_cfg


def _require_inference_workload(protocol_cfg: dict[str, Any],
                                label: str) -> None:
    if protocol_cfg["workload_kind"] != constants.WorkloadKind.INFERENCE.value:
        raise constants.ValidationError(
            f"{label}: only workload_kind='inference' is currently supported")


def _resolve_campaign_scripts(
    *,
    engine: str,
    has_accuracy_gate: bool,
    run_id: str,
    package: str | None,
) -> tuple[dict[str, Path], Path, dict[str, Path]]:
    """Resolve canonical target scripts and host `e2e_sweep` scripts before DB writes."""
    pkg_root, hc_dir, hc_cfg = resolve_package_hillclimber(run_id, package)
    to_copy: dict[str, Path] = {
        TARGET_RUNNER_FILENAME:
        _verify_contained_file(hc_dir, TARGET_RUNNER_FILENAME,
                               TARGET_RUNNER_FILENAME),
    }

    serving_map = hc_cfg.get("serving_scripts") or {}
    if engine not in serving_map:
        raise constants.ValidationError(
            f"engine {engine!r} is not supported by package hillclimber.serving_scripts"
        )
    to_copy[f"serve_{engine}.sh"] = _verify_contained_file(
        pkg_root, serving_map[engine],
        f"hillclimber.serving_scripts[{engine!r}]")

    if engine == "vllm":
        to_copy["apply_vllm_patch.sh"] = _verify_contained_file(
            pkg_root,
            str(hc_cfg.get("vllm_patch_applier") or ""),
            "hillclimber.vllm_patch_applier",
        )

    if has_accuracy_gate:
        acc_map = hc_cfg.get("accuracy_scripts") or {}
        if engine not in acc_map:
            raise constants.ValidationError(
                f"accuracy_gate is not supported for engine {engine!r}: "
                "no entry in package hillclimber.accuracy_scripts")
        to_copy["eval_recipe.sh"] = _verify_contained_file(
            pkg_root, acc_map[engine],
            f"hillclimber.accuracy_scripts[{engine!r}]")

    sweep_src = _verify_contained_file(
        pkg_root,
        str(hc_cfg.get("e2e_sweep_script") or ""),
        "hillclimber.e2e_sweep_script",
    )
    to_copy["serving_sweep.py"] = sweep_src
    companions: dict[str, Path] = {
        rel:
        _verify_contained_file(pkg_root, rel,
                               f"hillclimber.e2e_sweep_companions[{rel!r}]")
        for rel in hc_cfg.get("e2e_sweep_companions", ())
    }
    return to_copy, sweep_src, companions


def _resolve_microbenchmark_scripts(
    micro_cfg: dict[str, Any],
    *,
    workspace: Path,
    run_id: str,
    package: str | None,
    prefix: str,
) -> tuple[str, Path, Path | None]:
    """Resolve `microbenchmark_target` (plus `forward_run.py` for `.json` plans)."""
    _require_inference_workload(micro_cfg, prefix)
    target_rel: str = micro_cfg["execution_spec"]["microbenchmark_target"]
    target_src = _verify_workspace_file(workspace, target_rel,
                                        f"{prefix}.microbenchmark_target")
    runner_src: Path | None = None
    if target_rel.endswith(".json"):
        pkg_root, _, hc_cfg = resolve_package_hillclimber(run_id, package)
        runner_src = _verify_contained_file(
            pkg_root,
            str(hc_cfg.get("microbenchmark_runner") or ""),
            "hillclimber.microbenchmark_runner",
        )
    return target_rel, target_src, runner_src


def _copy_and_digest(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.is_file():
        shutil.copy2(src, dst)
    return utils.compute_digest(dst)


def _seal_campaign_scripts(
    campaign_id: int,
    *,
    to_copy: dict[str, Path],
    sweep_src: Path,
    companions: dict[str, Path],
) -> dict[str, str]:
    """Write canonical sealed scripts and `manifest.json` under `campaigns/<id>/scripts/`."""
    scripts_root = paths.get_campaign_scripts_dir(campaign_id)
    scripts_root.mkdir(parents=True, exist_ok=True)
    digests = {
        rel: _copy_and_digest(src, scripts_root / rel)
        for rel, src in to_copy.items()
    }
    manifest_doc = {
        "sealed_files": dict(digests),
        "e2e_sweep": {
            "source_path": str(sweep_src),
            "sha256": digests["serving_sweep.py"],
            "companions": {
                rel: {
                    "source_path": str(c_src),
                    "sha256": utils.compute_digest(c_src),
                }
                for rel, c_src in companions.items()
            },
        },
    }
    jsonio.write_json(scripts_root / SEALED_MANIFEST_FILENAME, manifest_doc)
    return digests


def _seal_microbenchmark_scripts(
    conn: sqlite3.Connection,
    *,
    campaign_id: int,
    protocol_id: int,
    target_rel: str,
    target_src: Path,
    runner_src: Path | None,
) -> dict[str, str]:
    """Seal `microbenchmark_target` (and `forward_run.py` if `.json`) and bind `Protocols`."""
    scripts_root = paths.get_campaign_scripts_dir(campaign_id)
    sealed_rel = f"microbenchmark_p{protocol_id}/{target_rel}"
    digests = {
        sealed_rel: _copy_and_digest(target_src, scripts_root / sealed_rel)
    }
    if runner_src is not None:
        digests["forward_run.py"] = _copy_and_digest(
            runner_src, scripts_root / "forward_run.py")
    db.exec_query(
        conn,
        "update_protocol_microbenchmark_target",
        target=sealed_rel,
        id=protocol_id,
    )
    manifest_path = scripts_root / SEALED_MANIFEST_FILENAME
    manifest_doc = jsonio.load_json_or_empty(manifest_path)
    sealed_files = dict(manifest_doc.get("sealed_files") or {})
    sealed_files.update(digests)
    manifest_doc["sealed_files"] = sealed_files
    jsonio.write_json(manifest_path, manifest_doc)
    return digests


def open_campaign(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    workspace: Path,
    campaign_config_path: Path,
    protocol_config_path: Path,
    code_dir: Path,
    microbenchmark_protocol_path: Path | None,
    package: str | None = None,
    telem: telemetry.TelemetryBuffer,
) -> dict[str, Any]:
    """Implement `runtime.py hc campaign open`."""
    existing_cid = paths.get_active_campaign_id(run_id)
    if existing_cid is not None:
        existing_row = db.exec_query(conn,
                                     "get_campaign_by_id",
                                     id=existing_cid).fetchone()
        if (existing_row is not None and existing_row["state"]
                != constants.CampaignState.CLOSED.value):
            raise constants.InvalidStateError(
                f"campaign {existing_cid} is already open in run {run_id!r} "
                f"(state={existing_row['state']})")

    campaign_cfg = schemas.validate_campaign_config(
        schemas.load_json_object(campaign_config_path, "Campaign Config"))
    protocol_cfg = schemas.validate_protocol_config(
        schemas.load_json_object(protocol_config_path, "Protocol Config"),
        is_microbenchmark=False,
    )
    _require_inference_workload(protocol_cfg, "Protocol Config")
    micro_cfg = (schemas.validate_protocol_config(
        schemas.load_json_object(microbenchmark_protocol_path,
                                 "Microbenchmark Protocol Config"),
        is_microbenchmark=True,
    ) if microbenchmark_protocol_path is not None else None)

    to_copy, sweep_src, companions = _resolve_campaign_scripts(
        engine=campaign_cfg["engine"],
        has_accuracy_gate=protocol_cfg["accuracy_gate"] is not None,
        run_id=run_id,
        package=package,
    )
    micro_resolved = (_resolve_microbenchmark_scripts(
        micro_cfg,
        workspace=workspace,
        run_id=run_id,
        package=package,
        prefix="Microbenchmark Protocol Config.execution_spec",
    ) if micro_cfg is not None else None)
    cleanup_src = git_ops.verify_init_worktree_inputs(code_dir, workspace)

    created_at = utils.utc_now_iso()
    created_campaign_root: Path | None = None
    src_dir = git_ops.get_worktree_dir(workspace)
    script_digests: dict[str, str] = {}
    try:
        with db.transaction(conn):
            protocol_id = insert_protocol(conn, protocol_cfg)
            micro_id = (insert_protocol(conn, micro_cfg)
                        if micro_cfg is not None else None)

            # Deferred FK on initial_version lets us allocate campaign_id first,
            # initialize `campaigns/<campaign_id>/git/` and `scripts/`, insert
            # `initial_version`, and bind `Campaigns.initial_version` before COMMIT.
            cur_camp = db.exec_query(
                conn,
                "insert_campaign",
                name=campaign_cfg["name"],
                owner=campaign_cfg["owner"],
                target_model=campaign_cfg["target_model"],
                engine=campaign_cfg["engine"],
                accelerator=campaign_cfg["accelerator"],
                created_at=created_at,
                protocol=protocol_id,
                microbenchmark_protocol=micro_id,
                state=constants.CampaignState.UNANCHORED.value,
            )
            campaign_id = int(cur_camp.lastrowid)
            created_campaign_root = paths.get_campaign_root(campaign_id)

            script_digests.update(
                _seal_campaign_scripts(
                    campaign_id,
                    to_copy=to_copy,
                    sweep_src=sweep_src,
                    companions=companions,
                ))
            if micro_id is not None and micro_resolved is not None:
                target_rel, target_src, runner_src = micro_resolved
                script_digests.update(
                    _seal_microbenchmark_scripts(
                        conn,
                        campaign_id=campaign_id,
                        protocol_id=micro_id,
                        target_rel=target_rel,
                        target_src=target_src,
                        runner_src=runner_src,
                    ))

            git_info = git_ops.init_campaign_repo(campaign_id, code_dir,
                                                  workspace)

            initial_cand_id = insert_candidate(
                conn,
                campaign_id=campaign_id,
                round_id=None,
                sha=git_info["sha"],
                tree_sha=git_info["tree_sha"],
                patch_sha=git_info["patch_sha"],
                short_innovation="Initial baseline model version",
                innovation="Initial baseline model version",
            )
            db.exec_query(
                conn,
                "set_campaign_initial_version",
                initial_version=initial_cand_id,
                id=campaign_id,
            )
            git_ops.finalize_candidate_ref(
                campaign_id,
                initial_cand_id,
                git_info["sha"],
                git_info["tree_sha"],
            )
    except BaseException:
        if created_campaign_root is not None and created_campaign_root.exists(
        ):
            shutil.rmtree(created_campaign_root, ignore_errors=True)
        if cleanup_src and src_dir.exists():
            shutil.rmtree(src_dir, ignore_errors=True)
        raise

    paths.record_active_campaign(run_id, campaign_id)
    telem.campaign_id = campaign_id
    telem.round_id = None
    telem.record(
        "campaign.opened",
        campaign_id=campaign_id,
        protocol_id=protocol_id,
        microbenchmark_protocol_id=micro_id,
        initial_version=initial_cand_id,
        sha=git_info["sha"],
        script_digests=script_digests,
    )
    return {
        "campaign_id": campaign_id,
        "state": constants.CampaignState.UNANCHORED.value,
        "protocol_id": protocol_id,
        "microbenchmark_protocol_id": micro_id,
        "initial_version": initial_cand_id,
        "sha": git_info["sha"],
        "script_digests": script_digests,
    }


def set_microbenchmark_protocol(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    workspace: Path,
    protocol_config_path: Path,
    package: str | None = None,
    telem: telemetry.TelemetryBuffer,
) -> dict[str, Any]:
    """Implement `runtime.py hc campaign set-microbenchmark-protocol`."""
    validated = schemas.validate_protocol_config(
        schemas.load_json_object(protocol_config_path,
                                 "Microbenchmark Protocol Config"),
        is_microbenchmark=True,
    )
    target_rel, target_src, runner_src = _resolve_microbenchmark_scripts(
        validated,
        workspace=workspace,
        run_id=run_id,
        package=package,
        prefix="Microbenchmark Protocol Config.execution_spec",
    )
    created_scope_dir: Path | None = None
    try:
        with db.transaction(conn):
            campaign = get_campaign_for_run(conn, run_id)
            campaign_id = int(campaign["id"])
            state = campaign["state"]
            if state not in (
                    constants.CampaignState.UNANCHORED.value,
                    constants.CampaignState.ACTIVE.value,
            ):
                raise constants.InvalidStateError(
                    f"campaign set-microbenchmark-protocol requires Unanchored or Active campaign, got {state}"
                )
            micro_job_row = db.exec_query(
                conn,
                "count_campaign_jobs_by_kind",
                campaign_id=campaign_id,
                kind=constants.JobKind.MICROBENCHMARK.value,
            ).fetchone()
            if micro_job_row is not None and int(micro_job_row["cnt"]) > 0:
                raise constants.InvalidStateError(
                    "microbenchmark_protocol is immutable once a Microbenchmark job has been submitted"
                )
            replaced_id = campaign["microbenchmark_protocol"]
            new_protocol_id = insert_protocol(conn, validated)
            created_scope_dir = (paths.get_campaign_scripts_dir(campaign_id) /
                                 f"microbenchmark_p{new_protocol_id}")
            script_digests = _seal_microbenchmark_scripts(
                conn,
                campaign_id=campaign_id,
                protocol_id=new_protocol_id,
                target_rel=target_rel,
                target_src=target_src,
                runner_src=runner_src,
            )
            db.exec_query(
                conn,
                "set_campaign_microbenchmark_protocol",
                protocol_id=new_protocol_id,
                id=campaign_id,
            )
    except BaseException:
        if created_scope_dir is not None and created_scope_dir.exists():
            shutil.rmtree(created_scope_dir, ignore_errors=True)
        raise

    bind_telemetry(conn, telem, campaign_id)
    telem.record(
        "campaign.microbenchmark_protocol_set",
        campaign_id=campaign_id,
        protocol_id=new_protocol_id,
        replaced_protocol_id=replaced_id,
        script_digests=script_digests,
    )
    return {
        "campaign_id": campaign_id,
        "microbenchmark_protocol_id": new_protocol_id,
        "replaced_protocol_id": replaced_id,
        "script_digests": script_digests,
    }


def block_campaign(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    reason: str,
    telem: telemetry.TelemetryBuffer,
) -> dict[str, Any]:
    """Implement `runtime.py hc campaign block <reason>`."""
    reason = schemas.require_text(reason, "block reason")

    with db.transaction(conn):
        campaign = get_campaign_for_run(conn, run_id)
        campaign_id = int(campaign["id"])
        state = campaign["state"]
        if state not in (
                constants.CampaignState.UNANCHORED.value,
                constants.CampaignState.ACTIVE.value,
        ):
            raise constants.InvalidStateError(
                f"cannot block campaign in state {state}")
        db.exec_query(
            conn,
            "update_campaign_block_state",
            state=constants.CampaignState.BLOCKED.value,
            block_reason=reason,
            id=campaign_id,
        )

    bind_telemetry(conn, telem, campaign_id)
    telem.record("campaign.blocked", campaign_id=campaign_id, reason=reason)
    return {
        "campaign_id": campaign_id,
        "state": constants.CampaignState.BLOCKED.value,
        "block_reason": reason,
    }


def unblock_campaign(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    telem: telemetry.TelemetryBuffer,
) -> dict[str, Any]:
    """Implement `runtime.py hc campaign unblock`."""
    with db.transaction(conn):
        campaign = get_campaign_for_run(conn, run_id)
        campaign_id = int(campaign["id"])
        if campaign["state"] != constants.CampaignState.BLOCKED.value:
            raise constants.InvalidStateError(
                f"campaign unblock requires Blocked campaign, got {campaign['state']}"
            )
        restored_state = (constants.CampaignState.ACTIVE.value
                          if campaign["entry_baseline"] is not None else
                          constants.CampaignState.UNANCHORED.value)
        db.exec_query(
            conn,
            "update_campaign_block_state",
            state=restored_state,
            block_reason=None,
            id=campaign_id,
        )

    bind_telemetry(conn, telem, campaign_id)
    telem.record("campaign.unblocked",
                 campaign_id=campaign_id,
                 state=restored_state)
    return {
        "campaign_id": campaign_id,
        "state": restored_state,
    }


def close_campaign(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    workspace: Path,
    abort: bool,
    telem: telemetry.TelemetryBuffer,
) -> dict[str, Any]:
    """Implement `runtime.py hc campaign close [--abort]`."""
    campaign = get_campaign_for_run(conn, run_id)
    campaign_id = int(campaign["id"])
    bind_telemetry(conn, telem, campaign_id)

    close_reason = (constants.CloseReason.ABORTED.value
                    if abort else constants.CloseReason.MANUAL.value)
    cancelled_jobs = cancel_dispatched_jobs(conn, campaign_id, workspace,
                                            run_id, telem)
    closed_at = utils.utc_now_iso()
    with db.transaction(conn):
        db.exec_query(
            conn,
            "close_campaign",
            state=constants.CampaignState.CLOSED.value,
            close_reason=close_reason,
            closed_at=closed_at,
            id=campaign_id,
        )

    telem.record(
        "campaign.closed",
        campaign_id=campaign_id,
        close_reason=close_reason,
    )
    return {
        "campaign_id": campaign_id,
        "state": constants.CampaignState.CLOSED.value,
        "close_reason": close_reason,
        "closed_at": closed_at,
        "cancelled_jobs": cancelled_jobs,
    }
