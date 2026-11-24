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
"""Qualitative review creation (`runtime.py hc review add`) for candidates and hypotheses."""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from framework.hc import (campaign_ops, candidate_ops, constants, db,
                          hypothesis_ops, schemas, telemetry)


def resolve_reviewer_identity() -> tuple[str, bool]:
    """Infer `(reviewer, human_authored)` from the invoking harness environment."""
    raw_human = os.environ.get("HC_HUMAN_AUTHORED", "").strip().lower()
    human_authored = raw_human in ("1", "true", "yes")

    explicit_reviewer = (os.environ.get("HC_REVIEWER")
                         or os.environ.get("TPU_AGENT_SUITE_AGENT")
                         or os.environ.get("AGENT_NAME")
                         or os.environ.get("TPU_AGENT_SUITE_HARNESS")
                         or "").strip()
    if explicit_reviewer:
        return explicit_reviewer, human_authored
    if human_authored:
        return (os.environ.get("USER") or "human").strip() or "human", True
    return "agent", False


def add_review(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    candidate_id: int | None,
    hypothesis_id: int | None,
    approval: str,
    comments: str,
    short_comments: str | None = None,
    conditions: str | None = None,
    telem: telemetry.TelemetryBuffer,
) -> dict[str, Any]:
    """Implement `runtime.py hc review add`."""
    if (candidate_id is None) == (hypothesis_id is None):
        raise constants.ValidationError(
            "exactly one of --candidate or --hypothesis must be supplied")

    approval_enum = constants.parse_enum(constants.ReviewApproval, approval,
                                         "approval")
    comments = schemas.require_text(comments, "--comments")
    resolved_short = (comments[:constants.MAX_SHORT_LENGTH]
                      if short_comments is None else schemas.require_text(
                          short_comments,
                          "--short-comments",
                          max_length=constants.MAX_SHORT_LENGTH,
                      ))

    if conditions is not None:
        if not isinstance(conditions, str):
            raise constants.ValidationError("--conditions must be a string")
        conditions = conditions.strip() or None
    if approval_enum == constants.ReviewApproval.CONDITIONAL_PASS:
        if conditions is None:
            raise constants.ValidationError(
                "--conditions is required when --approval is ConditionalPass")
        conditions = schemas.require_text(conditions, "--conditions")
    elif conditions is not None:
        raise constants.ValidationError(
            "--conditions is only allowed when --approval is ConditionalPass")

    reviewer, human_authored = resolve_reviewer_identity()
    with db.transaction(conn):
        campaign = campaign_ops.get_campaign_for_run(conn, run_id)
        campaign_id = int(campaign["id"])
        if campaign["state"] == constants.CampaignState.BLOCKED.value:
            raise constants.InvalidStateError(
                f"cannot add review while campaign {campaign_id} is Blocked")

        if candidate_id is not None:
            candidate_ops.get_campaign_candidate(conn, campaign_id,
                                                 candidate_id)
            target_str = f"candidate:{candidate_id}"
        else:
            assert hypothesis_id is not None
            hypothesis_ops.get_campaign_hypothesis(conn, campaign_id,
                                                   hypothesis_id)
            target_str = f"hypothesis:{hypothesis_id}"

        cur = db.exec_query(
            conn,
            "insert_review",
            candidate=candidate_id,
            hypothesis=hypothesis_id,
            human_authored=1 if human_authored else 0,
            reviewer=reviewer,
            approval=approval_enum.value,
            short_comments=resolved_short,
            conditions=conditions,
            comments=comments,
        )
        review_id = int(cur.lastrowid)

    campaign_ops.bind_telemetry(conn, telem, campaign_id)
    telem.record(
        "review.added",
        campaign_id=campaign_id,
        review_id=review_id,
        target=target_str,
        human_authored=human_authored,
        reviewer=reviewer,
        approval=approval_enum.value,
    )
    return {
        "review_id": review_id,
        "candidate": candidate_id,
        "hypothesis": hypothesis_id,
        "human_authored": human_authored,
        "reviewer": reviewer,
        "approval": approval_enum.value,
        "short_comments": resolved_short,
        "conditions": conditions,
    }
