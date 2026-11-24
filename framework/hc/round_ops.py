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
"""Round lifecycle operations for `runtime.py hc`."""

from __future__ import annotations

import sqlite3
from typing import Any

from framework.hc import campaign_ops, constants, db, telemetry

# Re-exported so round/job modules can reference active-campaign/open-round guards from either module.
require_active_campaign = campaign_ops.require_active_campaign
require_open_round = campaign_ops.require_open_round


def open_round(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    telem: telemetry.TelemetryBuffer,
) -> dict[str, Any]:
    """Implement `runtime.py hc round open`."""
    with db.transaction(conn):
        campaign = require_active_campaign(conn, run_id)
        campaign_id = int(campaign["id"])
        champion_id = int(campaign["champion"])

        existing_open = campaign_ops.get_open_round(conn, campaign_id)
        if existing_open is not None:
            raise constants.InvalidStateError(
                f"round {existing_open['id']} (round_number={existing_open['round_number']}) "
                f"is already Open in campaign {campaign_id}")

        max_row = db.exec_query(conn,
                                "get_max_round_number",
                                campaign_id=campaign_id).fetchone()
        next_number = int(max_row["max_num"]) + 1
        cur = db.exec_query(
            conn,
            "insert_round",
            campaign_id=campaign_id,
            round_number=next_number,
            current_champion=champion_id,
            state=constants.RoundState.OPEN.value,
        )
        round_id = int(cur.lastrowid)

    telem.campaign_id = campaign_id
    telem.round_id = round_id
    telem.record(
        "round.opened",
        campaign_id=campaign_id,
        round_id=round_id,
        round_number=next_number,
        current_champion=champion_id,
    )
    return {
        "round_id": round_id,
        "campaign_id": campaign_id,
        "round_number": next_number,
        "current_champion": champion_id,
        "state": constants.RoundState.OPEN.value,
    }
