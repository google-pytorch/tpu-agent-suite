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
"""Candidate preparation, proposal, supersession, and abandonment for `runtime.py hc`."""

from __future__ import annotations

import pathlib
import sqlite3
from typing import Any, Sequence

from framework.hc import (campaign_ops, constants, db, git_ops, hypothesis_ops,
                          schemas, telemetry)

Path = pathlib.Path


def get_campaign_candidate(
    conn: sqlite3.Connection,
    campaign_id: int,
    candidate_id: int,
) -> sqlite3.Row:
    """Fetch a `Candidates` row belonging to `campaign_id`, or raise `NotFoundError`."""
    row = db.exec_query(
        conn,
        "get_campaign_candidate",
        id=candidate_id,
        campaign_id=campaign_id,
    ).fetchone()
    if row is None:
        raise constants.NotFoundError(
            f"candidate {candidate_id} not found in campaign {campaign_id}")
    return row


def prepare_candidate(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    workspace: Path,
    parent_ids: Sequence[int],
    discard: bool,
    telem: telemetry.TelemetryBuffer,
) -> dict[str, Any]:
    """Implement `runtime.py hc candidate prepare <p1> [<p2>] [--discard]`."""
    campaign = campaign_ops.require_active_campaign(conn, run_id)
    campaign_id = int(campaign["id"])
    champion_id = int(campaign["champion"])
    champion_row = get_campaign_candidate(conn, campaign_id, champion_id)
    champion_sha = str(champion_row["sha"])

    pids = [int(pid) for pid in parent_ids]
    if not (1 <= len(pids) <= 2):
        raise constants.ValidationError(
            f"candidate prepare requires 1 or 2 parent candidate IDs, got {len(pids)}"
        )
    if len(pids) == 2 and pids[0] == pids[1]:
        raise constants.ValidationError(
            f"duplicate parent candidate ID {pids[0]} in candidate prepare")

    parent_rows = [
        get_campaign_candidate(conn, campaign_id, pid) for pid in pids
    ]
    parent_shas = [str(r["sha"]) for r in parent_rows]

    if not any(
            git_ops.is_ancestor(campaign_id, champion_sha, p_sha)
            for p_sha in parent_shas):
        raise constants.InvalidStateError(
            f"at least one parent must descend from (or equal) current champion {champion_id}"
        )

    prep = git_ops.prepare_worktree(
        campaign_id,
        workspace,
        parent_shas,
        parent_ids=pids,
        discard=discard,
    )

    campaign_ops.bind_telemetry(conn, telem, campaign_id)
    telem.record(
        "candidate.prepared",
        campaign_id=campaign_id,
        parents=pids,
        has_conflicts=bool(prep["has_conflicts"]),
    )
    return {
        "campaign_id": campaign_id,
        "parents": pids,
        "prepared_tree_sha": prep["prepared_tree_sha"],
        "has_conflicts": bool(prep["has_conflicts"]),
    }


def _resolve_staged_parent_rows(
    conn: sqlite3.Connection,
    campaign_id: int,
    state: dict[str, Any],
) -> list[sqlite3.Row]:
    """Resolve `worktree_state.json` staged parents back to `Candidates` rows."""
    staged_shas: list[str] = state["staged_parents"]
    staged_ids: list[int] | None = state.get("staged_parent_ids")
    if staged_ids is None:
        raise constants.GitOperationError(
            "missing staged_parent_ids in worktree_state.json")
    rows = [
        get_campaign_candidate(conn, campaign_id, pid) for pid in staged_ids
    ]
    for row, expected_sha in zip(rows, staged_shas):
        if str(row["sha"]) != expected_sha:
            raise constants.GitOperationError(
                f"worktree_state mismatch for parent candidate {row['id']}")
    return rows


def propose_candidate(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    workspace: Path,
    hypothesis_ids: Sequence[int],
    supersedes_id: int | None,
    innovation: str,
    short_innovation: str,
    telem: telemetry.TelemetryBuffer,
) -> dict[str, Any]:
    """Implement `runtime.py hc candidate propose`."""
    innovation = schemas.require_text(innovation, "--innovation")
    short_innovation = schemas.require_text(
        short_innovation,
        "--short-innovation",
        max_length=constants.MAX_SHORT_LENGTH,
    )

    dedup_hids = list(dict.fromkeys(int(hid) for hid in hypothesis_ids))

    with db.transaction(conn):
        campaign = campaign_ops.require_active_campaign(conn, run_id)
        campaign_id = int(campaign["id"])
        open_rnd = campaign_ops.require_open_round(conn, campaign_id)
        round_id = int(open_rnd["id"])

        round_champ_id = int(open_rnd["current_champion"])
        round_champ_sha = str(
            get_campaign_candidate(conn, campaign_id, round_champ_id)["sha"])
        initial_sha = str(
            get_campaign_candidate(conn, campaign_id,
                                   int(campaign["initial_version"]))["sha"])

        for hid in dedup_hids:
            hyp = hypothesis_ops.get_campaign_hypothesis(
                conn, campaign_id, hid)
            if hyp["state"] != constants.HypothesisState.OPEN.value:
                raise constants.InvalidStateError(
                    f"hypothesis {hid} is {hyp['state']}; only Open hypotheses can be linked to new candidates"
                )

        if supersedes_id is not None:
            sup_row = get_campaign_candidate(conn, campaign_id,
                                             int(supersedes_id))
            if sup_row["abandoned_reason"] is not None:
                raise constants.InvalidStateError(
                    f"cannot supersede Abandoned candidate {supersedes_id}")
            if sup_row["superseded_by"] is not None:
                raise constants.InvalidStateError(
                    f"candidate {supersedes_id} is already superseded by candidate {sup_row['superseded_by']}"
                )
            if git_ops.is_ancestor(campaign_id, str(sup_row["sha"]),
                                   round_champ_sha):
                raise constants.InvalidStateError(
                    f"cannot supersede Promoted candidate {supersedes_id}")

        wt_state = git_ops.read_worktree_state(campaign_id)
        parent_rows = _resolve_staged_parent_rows(conn, campaign_id, wt_state)
        parent_ids = [int(r["id"]) for r in parent_rows]
        parent_shas = [str(r["sha"]) for r in parent_rows]
        if len(parent_ids) == 2 and parent_ids[0] == parent_ids[1]:
            raise constants.ValidationError(
                "staged parents must reference distinct candidates")
        if not any(
                git_ops.is_ancestor(campaign_id, round_champ_sha, p_sha)
                for p_sha in parent_shas):
            raise constants.InvalidStateError(
                f"at least one staged parent must descend from (or equal) round champion {round_champ_id}"
            )

        commit_info = git_ops.propose_candidate_commit(
            campaign_id,
            workspace,
            initial_sha,
            message=short_innovation,
        )

        candidate_id = campaign_ops.insert_candidate(
            conn,
            campaign_id=campaign_id,
            round_id=round_id,
            sha=commit_info["sha"],
            tree_sha=commit_info["tree_sha"],
            patch_sha=commit_info["patch_sha"],
            short_innovation=short_innovation,
            innovation=innovation,
        )

        for idx, pid in enumerate(parent_ids):
            db.exec_query(
                conn,
                "insert_candidate_parent",
                candidate_id=candidate_id,
                parent_index=idx,
                parent_id=pid,
            )

        for hid in dedup_hids:
            db.exec_query(
                conn,
                "insert_hypothesis_candidate",
                hypothesis_id=hid,
                candidate_id=candidate_id,
            )

        if supersedes_id is not None:
            db.exec_query(
                conn,
                "set_candidate_superseded_by",
                superseded_by=candidate_id,
                id=int(supersedes_id),
            )

        git_ops.finalize_candidate_ref(
            campaign_id,
            candidate_id,
            commit_info["sha"],
            commit_info["tree_sha"],
        )

    telem.campaign_id = campaign_id
    telem.round_id = round_id
    telem.record(
        "candidate.proposed",
        campaign_id=campaign_id,
        round_id=round_id,
        candidate_id=candidate_id,
        parents=parent_ids,
        sha=commit_info["sha"],
        patch_sha=commit_info["patch_sha"],
        hypothesis_ids=dedup_hids,
    )
    if supersedes_id is not None:
        telem.record(
            "candidate.superseded",
            campaign_id=campaign_id,
            round_id=round_id,
            candidate_id=int(supersedes_id),
            superseded_by=candidate_id,
        )

    return {
        "candidate_id": candidate_id,
        "round": round_id,
        "parents": parent_ids,
        "hypotheses": dedup_hids,
        "sha": commit_info["sha"],
        "tree_sha": commit_info["tree_sha"],
        "patch_sha": commit_info["patch_sha"],
        "supersedes":
        int(supersedes_id) if supersedes_id is not None else None,
    }


def abandon_candidate(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    candidate_id: int,
    reason: str,
    telem: telemetry.TelemetryBuffer,
) -> dict[str, Any]:
    """Implement `runtime.py hc candidate abandon <id> <reason>`."""
    reason = schemas.require_text(reason, "abandon reason")

    with db.transaction(conn):
        campaign = campaign_ops.require_active_campaign(conn, run_id)
        campaign_id = int(campaign["id"])
        cand = get_campaign_candidate(conn, campaign_id, candidate_id)
        if cand["abandoned_reason"] is not None:
            raise constants.InvalidStateError(
                f"candidate {candidate_id} is already abandoned")
        if cand["superseded_by"] is not None:
            raise constants.InvalidStateError(
                f"cannot abandon Superseded candidate {candidate_id} "
                f"(superseded by {cand['superseded_by']})")
        if int(cand["id"]) == int(campaign["initial_version"]):
            raise constants.InvalidStateError(
                f"cannot abandon initial_version candidate {candidate_id}")
        champion_id = campaign["champion"]
        if champion_id is not None:
            champ_row = get_campaign_candidate(conn, campaign_id,
                                               int(champion_id))
            if git_ops.is_ancestor(campaign_id, str(cand["sha"]),
                                   str(champ_row["sha"])):
                raise constants.InvalidStateError(
                    f"cannot abandon Promoted candidate {candidate_id}")
        db.exec_query(conn,
                      "abandon_candidate",
                      reason=reason,
                      id=candidate_id)

    campaign_ops.bind_telemetry(conn, telem, campaign_id)
    telem.record(
        "candidate.abandoned",
        campaign_id=campaign_id,
        candidate_id=candidate_id,
        reason=reason,
    )
    return {
        "candidate_id": candidate_id,
        "state": constants.CandidateState.ABANDONED.value,
        "abandoned_reason": reason,
    }
