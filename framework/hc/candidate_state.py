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
"""On-the-fly strict-precedence `Candidates` state and `prior_decisions_count`."""

from __future__ import annotations

import sqlite3
from typing import Any, Mapping

from framework.hc import campaign_ops, constants, db, git_ops


def resolve_reference_baseline(
    conn: sqlite3.Connection,
    campaign_row: Mapping[str, Any],
    baseline_measurement_override: int | None = None,
) -> tuple[sqlite3.Row | None, bool]:
    """Resolve `(baseline_measurement_row, is_explicit_override)` for candidate evaluation."""
    campaign_id = int(campaign_row["id"])
    if baseline_measurement_override is not None:
        row = db.exec_query(
            conn,
            "get_campaign_measurement",
            measurement_id=baseline_measurement_override,
            campaign_id=campaign_id,
        ).fetchone()
        if row is None:
            raise constants.NotFoundError(
                f"baseline measurement {baseline_measurement_override} not found in campaign {campaign_id}"
            )
        return row, True

    cur_base_id = campaign_row["current_baseline"]
    if cur_base_id is None:
        return None, False
    row = db.exec_query(
        conn,
        "get_measurement_by_id",
        id=int(cur_base_id),
    ).fetchone()
    return row, False


def _e2e_decision_qualifies(
    drow: Mapping[str, Any],
    *,
    b_id: int | None,
    b_subj_id: int | None,
    main_protocol_id: int,
    explicit_override: bool,
) -> bool:
    if b_id is None:
        return False
    if explicit_override:
        return int(drow["reference_measurement"]) == b_id
    return (int(drow["reference_measurement"]) == b_id
            or (int(drow["ref_subject"]) == b_subj_id
                and int(drow["ref_protocol"]) == main_protocol_id
                and str(drow["ref_trial"]) == str(drow["subj_trial"])))


def get_latest_qualifying_e2e_decision(
    conn: sqlite3.Connection,
    *,
    candidate_id: int,
    main_protocol_id: int,
    baseline_measurement_row: Mapping[str, Any],
    explicit_override: bool,
) -> sqlite3.Row | None:
    """Return the latest qualifying `is_e2e=True` `Decision` for `candidate_id` against `B`."""
    b_id = int(baseline_measurement_row["id"])
    b_subj_id = int(baseline_measurement_row["subject"])
    rows = db.exec_query(
        conn,
        "list_candidate_decisions",
        candidate_id=candidate_id,
    ).fetchall()
    for drow in rows:
        if int(drow["is_e2e"]) == 1 and _e2e_decision_qualifies(
                drow,
                b_id=b_id,
                b_subj_id=b_subj_id,
                main_protocol_id=main_protocol_id,
                explicit_override=explicit_override,
        ):
            return drow
    return None


def get_latest_settled_accuracy(
    conn: sqlite3.Connection,
    candidate_id: int,
) -> bool | None:
    """Return latest non-Null `passed_accuracy` on succeeded `Accuracy` jobs for `candidate_id`."""
    row = db.exec_query(
        conn,
        "get_latest_settled_accuracy",
        candidate_id=candidate_id,
        kind=constants.JobKind.ACCURACY.value,
        status=constants.JobStatus.SUCCEEDED.value,
    ).fetchone()
    if row is None:
        return None
    return bool(row["passed_accuracy"])


def has_e2e_or_accuracy_in_flight(
    conn: sqlite3.Connection,
    *,
    candidate_id: int,
    champion_candidate_id: int | None,
) -> bool:
    """Return True if candidate has Dispatched E2E/Accuracy or paired champion E2E job."""
    own_row = db.exec_query(
        conn,
        "has_dispatched_e2e_or_accuracy_job",
        candidate_id=candidate_id,
        e2e_kind=constants.JobKind.E2E_BENCHMARK.value,
        accuracy_kind=constants.JobKind.ACCURACY.value,
        status=constants.JobStatus.DISPATCHED.value,
    ).fetchone()
    if own_row is not None:
        return True

    if champion_candidate_id is not None and champion_candidate_id != candidate_id:
        paired_champ = db.exec_query(
            conn,
            "has_paired_champion_e2e_in_flight",
            candidate_id=candidate_id,
            e2e_kind=constants.JobKind.E2E_BENCHMARK.value,
            succeeded_status=constants.JobStatus.SUCCEEDED.value,
            champion_candidate_id=champion_candidate_id,
            dispatched_status=constants.JobStatus.DISPATCHED.value,
        ).fetchone()
        if paired_champ is not None:
            return True

    return False


def compute_candidate_state_and_prior_decisions(
    conn: sqlite3.Connection,
    *,
    campaign_row: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
    baseline_measurement_override: int | None = None,
) -> tuple[constants.CandidateState, int]:
    """Compute `(state, prior_decisions_count)` in strict top-to-bottom precedence order."""
    campaign_id = int(campaign_row["id"])
    cand_id = int(candidate_row["id"])
    cand_sha = str(candidate_row["sha"])
    main_protocol_id = int(campaign_row["protocol"])

    b_row, explicit_override = resolve_reference_baseline(
        conn, campaign_row, baseline_measurement_override)
    open_rnd = campaign_ops.get_open_round(conn, campaign_id)
    open_round_id = int(open_rnd["id"]) if open_rnd is not None else None

    # Scan candidate decisions once (newest first) to count prior decisions and
    # capture the latest qualifying E2E and microbenchmark decisions.
    all_decisions = db.exec_query(
        conn,
        "list_candidate_decisions",
        candidate_id=cand_id,
    ).fetchall()

    prior_decisions_count = 0
    latest_e2e_dec: sqlite3.Row | None = None
    latest_micro_dec: sqlite3.Row | None = None
    b_id = int(b_row["id"]) if b_row is not None else None
    b_subj_id = int(b_row["subject"]) if b_row is not None else None
    for drow in all_decisions:
        if int(drow["is_e2e"]) == 1:
            qualifies = _e2e_decision_qualifies(
                drow,
                b_id=b_id,
                b_subj_id=b_subj_id,
                main_protocol_id=main_protocol_id,
                explicit_override=explicit_override,
            )
            if qualifies and latest_e2e_dec is None:
                latest_e2e_dec = drow
        else:
            if explicit_override:
                qualifies = (b_id is not None
                             and int(drow["reference_measurement"]) == b_id)
            elif b_subj_id is not None:
                qualifies = int(drow["ref_subject"]) == b_subj_id
            else:
                qualifies = (open_round_id is not None
                             and drow["round"] is not None
                             and int(drow["round"]) == open_round_id)
            if qualifies and latest_micro_dec is None:
                latest_micro_dec = drow
        if not qualifies:
            prior_decisions_count += 1

    # 1. Promoted: ancestor of (or equal to) B's candidate in git history.
    if b_subj_id is not None:
        b_cand_row = db.exec_query(
            conn,
            "get_campaign_candidate",
            id=b_subj_id,
            campaign_id=campaign_id,
        ).fetchone()
        if b_cand_row is not None and git_ops.is_ancestor(
                campaign_id, cand_sha, str(b_cand_row["sha"])):
            return constants.CandidateState.PROMOTED, prior_decisions_count

    # 2. Superseded.
    if candidate_row["superseded_by"] is not None:
        return constants.CandidateState.SUPERSEDED, prior_decisions_count

    # 3. Abandoned.
    if candidate_row["abandoned_reason"] is not None:
        return constants.CandidateState.ABANDONED, prior_decisions_count

    # 4. E2E_In_Flight.
    champ_for_flight = (b_subj_id if b_subj_id is not None else
                        (int(campaign_row["champion"])
                         if campaign_row["champion"] is not None else None))
    if has_e2e_or_accuracy_in_flight(
            conn,
            candidate_id=cand_id,
            champion_candidate_id=champ_for_flight,
    ):
        return constants.CandidateState.E2E_IN_FLIGHT, prior_decisions_count

    # 5. E2E decision ladder against B.
    if latest_e2e_dec is not None:
        verdict = constants.DecisionVerdict(latest_e2e_dec["verdict"])
        if verdict == constants.DecisionVerdict.WIN:
            main_protocol = db.exec_query(
                conn,
                "get_protocol_by_id",
                id=main_protocol_id,
            ).fetchone()
            assert main_protocol is not None and int(
                main_protocol["has_accuracy_gate"]
            ) == 1, (
                f"main E2E protocol {main_protocol_id} must configure an accuracy_gate"
            )
            acc = get_latest_settled_accuracy(conn, cand_id)
            if acc is True:
                return (constants.CandidateState.E2E_PASSED,
                        prior_decisions_count)
            if acc is False:
                return (constants.CandidateState.E2E_ACCURACY_FAILED,
                        prior_decisions_count)
            return (constants.CandidateState.E2E_PENDING_ACCURACY,
                    prior_decisions_count)
        if verdict == constants.DecisionVerdict.VIABLE:
            return constants.CandidateState.E2E_VIABLE, prior_decisions_count
        # Per spec, Neutral, Loss, and Invalid verdicts map to E2E_Regressed.
        return constants.CandidateState.E2E_REGRESSED, prior_decisions_count

    # 6. Microbenchmark_In_Flight.
    micro_flight = db.exec_query(
        conn,
        "has_dispatched_job_by_kind",
        candidate_id=cand_id,
        kind=constants.JobKind.MICROBENCHMARK.value,
        status=constants.JobStatus.DISPATCHED.value,
    ).fetchone()
    if micro_flight is not None:
        return (constants.CandidateState.MICROBENCHMARK_IN_FLIGHT,
                prior_decisions_count)

    # 7. Microbench_Passed / Microbench_Regressed.
    if latest_micro_dec is not None:
        m_verdict = constants.DecisionVerdict(latest_micro_dec["verdict"])
        if m_verdict in (
                constants.DecisionVerdict.WIN,
                constants.DecisionVerdict.VIABLE,
        ):
            return (constants.CandidateState.MICROBENCH_PASSED,
                    prior_decisions_count)
        return (constants.CandidateState.MICROBENCH_REGRESSED,
                prior_decisions_count)

    # 8. Untested.
    return constants.CandidateState.UNTESTED, prior_decisions_count
