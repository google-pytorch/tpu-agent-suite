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
"""Hypothesis creation and refutation operations for `runtime.py hc`."""

from __future__ import annotations

import pathlib
import sqlite3
from typing import Any

from framework.hc import campaign_ops, constants, db, schemas, telemetry

Path = pathlib.Path


def get_campaign_hypothesis(
    conn: sqlite3.Connection,
    campaign_id: int,
    hypothesis_id: int,
) -> sqlite3.Row:
    """Fetch a `Hypotheses` row belonging to `campaign_id`, or raise `NotFoundError`."""
    row = db.exec_query(
        conn,
        "get_campaign_hypothesis",
        hypothesis_id=hypothesis_id,
        campaign_id=campaign_id,
    ).fetchone()
    if row is None:
        raise constants.NotFoundError(
            f"hypothesis {hypothesis_id} not found in campaign {campaign_id}")
    return row


def add_hypothesis(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    hypothesis_json_path: Path,
    telem: telemetry.TelemetryBuffer,
) -> dict[str, Any]:
    """Implement `runtime.py hc hypothesis add <path>`."""
    validated = schemas.validate_hypothesis_config(
        schemas.load_json_object(hypothesis_json_path, "Hypothesis"))

    with db.transaction(conn):
        campaign = campaign_ops.require_active_campaign(conn, run_id)
        campaign_id = int(campaign["id"])
        open_rnd = campaign_ops.require_open_round(conn, campaign_id)
        round_id = int(open_rnd["id"])

        cur = db.exec_query(
            conn,
            "insert_hypothesis",
            round_created=round_id,
            lever_kind=validated["lever_kind"],
            target_scope=validated["target_scope"],
            short_claim=validated["short_claim"],
            state=constants.HypothesisState.OPEN.value,
            claim=validated["claim"],
        )
        hyp_id = int(cur.lastrowid)

        for idx, op in enumerate(validated["target_canonical_ops"]):
            db.exec_query(
                conn,
                "insert_hypothesis_canonical_op",
                hypothesis_id=hyp_id,
                op_index=idx,
                layer_kind=op["layer_kind"],
                operator=op["operator"],
            )

        for idx, edge in enumerate(validated["target_relayout_edges"]):
            prod = edge["producer"]
            cons = edge["consumer"]
            db.exec_query(
                conn,
                "insert_hypothesis_relayout_edge",
                hypothesis_id=hyp_id,
                edge_index=idx,
                producer_layer_kind=prod["layer_kind"]
                if prod is not None else None,
                producer_operator=prod["operator"]
                if prod is not None else None,
                consumer_layer_kind=cons["layer_kind"]
                if cons is not None else None,
                consumer_operator=cons["operator"]
                if cons is not None else None,
            )

        for idx, ev in enumerate(validated["evidence"]):
            db.exec_query(
                conn,
                "insert_hypothesis_evidence",
                hypothesis_id=hyp_id,
                evidence_index=idx,
                source_type=ev["source_type"],
                locator=ev["locator"],
                summary=ev["summary"],
            )

        for idx, ref in enumerate(validated["target_code_refs"]):
            db.exec_query(
                conn,
                "insert_hypothesis_code_ref",
                hypothesis_id=hyp_id,
                ref_index=idx,
                code_ref=ref,
            )

    telem.campaign_id = campaign_id
    telem.round_id = round_id
    telem.record(
        "hypothesis.added",
        campaign_id=campaign_id,
        round_id=round_id,
        hypothesis_id=hyp_id,
        lever_kind=validated["lever_kind"],
        target_scope=validated["target_scope"],
        short_claim=validated["short_claim"],
    )
    return {
        "hypothesis_id": hyp_id,
        "round_created": round_id,
        "lever_kind": validated["lever_kind"],
        "target_scope": validated["target_scope"],
        "state": constants.HypothesisState.OPEN.value,
        "short_claim": validated["short_claim"],
    }


def refute_hypothesis(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    hypothesis_id: int,
    reason: str,
    telem: telemetry.TelemetryBuffer,
) -> dict[str, Any]:
    """Implement `runtime.py hc hypothesis refute <id> <reason>`."""
    reason = schemas.require_text(reason, "refutation reason")

    with db.transaction(conn):
        campaign = campaign_ops.require_active_campaign(conn, run_id)
        campaign_id = int(campaign["id"])
        hyp = get_campaign_hypothesis(conn, campaign_id, hypothesis_id)

        if hyp["state"] == constants.HypothesisState.PROMOTED.value:
            raise constants.InvalidStateError(
                f"cannot refute Promoted hypothesis {hypothesis_id}")
        if (hyp["state"] == constants.HypothesisState.REFUTED.value
                or hyp["refutation_reason"] is not None):
            raise constants.InvalidStateError(
                f"hypothesis {hypothesis_id} is already Refuted; refutation_reason is immutable once set"
            )

        db.exec_query(
            conn,
            "refute_hypothesis",
            state=constants.HypothesisState.REFUTED.value,
            reason=reason,
            id=hypothesis_id,
        )

    campaign_ops.bind_telemetry(conn, telem, campaign_id)
    telem.record(
        "hypothesis.refuted",
        campaign_id=campaign_id,
        hypothesis_id=hypothesis_id,
        reason=reason,
    )
    return {
        "hypothesis_id": hypothesis_id,
        "state": constants.HypothesisState.REFUTED.value,
        "refutation_reason": reason,
    }
