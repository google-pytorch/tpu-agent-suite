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
"""Benchmark comparison verdicts, accuracy gate evaluation, and `Decisions` persistence."""

from __future__ import annotations

import dataclasses
import math
import sqlite3
from typing import Any, Mapping, Sequence

from framework.hc import constants, db, telemetry

_FP_EPS: float = 1e-12


@dataclasses.dataclass(frozen=True)
class BenchmarkEvaluation:
    """Result of comparing a subject measurement against a reference measurement."""

    verdict: constants.DecisionVerdict
    weighted_geomean_delta_pct: float | None
    worst_cell_delta_pct: float | None
    per_cell_deltas_pct: dict[int, float]
    reason: str


def load_measurement_primary_metrics(
    conn: sqlite3.Connection,
    measurement_id: int,
    primary_metric: str,
) -> dict[int, tuple[float, int]]:
    """Return `{cell_id: (value, n)}` for `primary_metric` on `measurement_id`."""
    rows = db.exec_query(
        conn,
        "get_measurement_primary_metrics",
        measurement_id=measurement_id,
        metric_name=primary_metric,
    ).fetchall()
    return {int(r["cell_id"]): (float(r["value"]), int(r["n"])) for r in rows}


def measurement_meets_min_replicates(
    conn: sqlite3.Connection,
    measurement_id: int,
    protocol_row: Mapping[str, Any],
    protocol_cells: Sequence[Mapping[str, Any]],
) -> bool:
    """Return True iff every protocol cell has `n >= protocol.min_replicates`."""
    primary_metric = str(protocol_row["primary_metric"])
    min_reps = int(protocol_row["min_replicates"])
    by_cell = load_measurement_primary_metrics(conn, measurement_id,
                                               primary_metric)
    for cell in protocol_cells:
        cid = int(cell["id"])
        if cid not in by_cell or by_cell[cid][1] < min_reps:
            return False
    return True


def evaluate_benchmark_comparison(
    *,
    protocol_row: Mapping[str, Any],
    protocol_cells: Sequence[Mapping[str, Any]],
    subject_cells: Mapping[int, tuple[float, int]],
    reference_cells: Mapping[int, tuple[float, int]],
) -> BenchmarkEvaluation:
    """Evaluate `Decisions` verdict and deterministic reason under `protocol_row`."""
    if not protocol_cells:
        raise constants.ValidationError("protocol has no measurement cells")
    direction = constants.MetricDirection(
        protocol_row["primary_metric_direction"])
    min_replicates = int(protocol_row["min_replicates"])
    min_win_pct = float(protocol_row["min_win_percentage"])
    noise_floor_pct = float(protocol_row["noise_floor_percentage"])
    max_cell_reg = (float(protocol_row["max_cell_regression_percentage"])
                    if protocol_row["max_cell_regression_percentage"]
                    is not None else None)
    primary_metric = str(protocol_row["primary_metric"])

    def _invalid(reason: str) -> BenchmarkEvaluation:
        return BenchmarkEvaluation(constants.DecisionVerdict.INVALID, None,
                                   None, {}, reason)

    # 1. Completeness & replication gate across all protocol cells.
    for cell in protocol_cells:
        cid, cidx = int(cell["id"]), int(cell["cell_index"])
        if cid not in subject_cells:
            return _invalid(
                f"Invalid: subject measurement missing {primary_metric} "
                f"on cell_index={cidx} (cell_id={cid}).")
        if cid not in reference_cells:
            return _invalid(
                f"Invalid: reference measurement missing {primary_metric} "
                f"on cell_index={cidx} (cell_id={cid}).")
        subj_val, subj_n = subject_cells[cid]
        ref_val, ref_n = reference_cells[cid]
        if subj_n < min_replicates:
            return _invalid(
                f"Invalid: subject cell_index={cidx} under-replicated "
                f"(n={subj_n} < min_replicates={min_replicates}).")
        if ref_n < min_replicates:
            return _invalid(
                f"Invalid: reference cell_index={cidx} under-replicated "
                f"(n={ref_n} < min_replicates={min_replicates}).")
        if subj_val <= 0.0 or ref_val <= 0.0:
            return _invalid(
                f"Invalid: non-positive {primary_metric} on cell_index={cidx} "
                f"(subject={subj_val}, reference={ref_val}).")

    # 2. Per-cell speedup ratios r_i and weight-normalized geometric mean ratio R.
    total_weight = math.fsum(float(cell["weight"]) for cell in protocol_cells)
    if total_weight <= 0.0:
        raise constants.ValidationError(
            "protocol measurement cell weights must sum to a positive value")
    weighted_logs: list[float] = []
    per_cell_deltas: dict[int, float] = {}
    regressed_cell_msg: str | None = None

    for cell in protocol_cells:
        cid, cidx = int(cell["id"]), int(cell["cell_index"])
        weight = float(cell["weight"])
        c_val, b_val = subject_cells[cid][0], reference_cells[cid][0]
        r_i = (c_val / b_val if direction == constants.MetricDirection.MAXIMUM
               else b_val / c_val)
        delta_i = (r_i - 1.0) * 100.0
        per_cell_deltas[cid] = delta_i
        weighted_logs.append(weight * math.log(r_i))

        if (max_cell_reg is not None and delta_i < -(max_cell_reg + _FP_EPS)
                and regressed_cell_msg is None):
            regressed_cell_msg = (
                f"cell_index={cidx} regressed by {-delta_i:.3f}% "
                f"(exceeding max_cell_regression_percentage={max_cell_reg:.3f}%)"
            )

    geomean_delta_pct = (math.exp(math.fsum(weighted_logs) / total_weight) -
                         1.0) * 100.0
    worst_delta_pct = min(per_cell_deltas.values())

    def _verdict(v: constants.DecisionVerdict,
                 reason: str) -> BenchmarkEvaluation:
        return BenchmarkEvaluation(v, geomean_delta_pct, worst_delta_pct,
                                   per_cell_deltas, reason)

    if regressed_cell_msg is not None:
        return _verdict(
            constants.DecisionVerdict.INVALID,
            f"Invalid: {regressed_cell_msg}; "
            f"weighted geomean delta {geomean_delta_pct:+.3f}%.",
        )

    # 3. Primary-metric threshold classification (inclusive noise floor checked first).
    if -(noise_floor_pct + _FP_EPS) <= geomean_delta_pct <= (noise_floor_pct +
                                                             _FP_EPS):
        return _verdict(
            constants.DecisionVerdict.NEUTRAL,
            f"Neutral: weighted geomean {primary_metric} delta "
            f"{geomean_delta_pct:+.3f}% is within noise_floor_percentage "
            f"[-{noise_floor_pct:.3f}%, +{noise_floor_pct:.3f}%] "
            f"(worst cell {worst_delta_pct:+.3f}%).",
        )
    if geomean_delta_pct < -(noise_floor_pct + _FP_EPS):
        return _verdict(
            constants.DecisionVerdict.LOSS,
            f"Loss: weighted geomean {primary_metric} delta "
            f"{geomean_delta_pct:+.3f}% regressed beyond "
            f"-noise_floor_percentage (-{noise_floor_pct:.3f}%; "
            f"worst cell {worst_delta_pct:+.3f}%).",
        )
    if geomean_delta_pct >= (min_win_pct - _FP_EPS):
        return _verdict(
            constants.DecisionVerdict.WIN,
            f"Win: weighted geomean {primary_metric} improved by "
            f"{geomean_delta_pct:+.3f}% (>= min_win_percentage "
            f"{min_win_pct:.3f}% and > noise_floor_percentage "
            f"{noise_floor_pct:.3f}%; worst cell {worst_delta_pct:+.3f}%).",
        )
    return _verdict(
        constants.DecisionVerdict.VIABLE,
        f"Viable: weighted geomean {primary_metric} improved by "
        f"{geomean_delta_pct:+.3f}% (> noise_floor_percentage "
        f"{noise_floor_pct:.3f}% but < min_win_percentage "
        f"{min_win_pct:.3f}%; worst cell {worst_delta_pct:+.3f}%).",
    )


def evaluate_accuracy_gate(
    protocol_row: Mapping[str, Any],
    accuracy_value: float,
) -> bool | None:
    """Evaluate `protocol.accuracy_gate` for `accuracy_value`; None if relative anchor unset."""
    if not int(protocol_row["has_accuracy_gate"]):
        return True
    is_relative = bool(protocol_row["accuracy_gate_is_relative"])
    relative_to = protocol_row["accuracy_gate_relative_to"]
    if is_relative:
        if relative_to is None:
            return None
        effective = float(accuracy_value) - float(relative_to)
    else:
        effective = float(accuracy_value)

    threshold = float(protocol_row["accuracy_gate_threshold"])
    comp = constants.Comparator(protocol_row["accuracy_gate_comparator"])
    if comp == constants.Comparator.GTE:
        return effective >= (threshold - _FP_EPS)
    return effective <= (threshold + _FP_EPS)


def compute_improvement_pct(
    conn: sqlite3.Connection,
    protocol_row: Mapping[str, Any],
    protocol_cells: Sequence[Mapping[str, Any]],
    subject_measurement_id: int,
    reference_measurement_id: int,
) -> float | None:
    """Compute weighted geometric mean improvement % of `subject` over `reference`."""
    primary_metric = str(protocol_row["primary_metric"])
    subj = load_measurement_primary_metrics(conn, subject_measurement_id,
                                            primary_metric)
    ref = load_measurement_primary_metrics(conn, reference_measurement_id,
                                           primary_metric)
    ev = evaluate_benchmark_comparison(
        protocol_row=protocol_row,
        protocol_cells=protocol_cells,
        subject_cells=subj,
        reference_cells=ref,
    )
    return ev.weighted_geomean_delta_pct


def record_decision_if_needed(
    conn: sqlite3.Connection,
    *,
    campaign_id: int,
    round_id: int | None,
    is_e2e: bool,
    subject_measurement_id: int,
    reference_measurement_id: int,
    protocol_row: Mapping[str, Any],
    protocol_cells: Sequence[Mapping[str, Any]],
    telem: telemetry.TelemetryBuffer,
) -> tuple[int, BenchmarkEvaluation]:
    """Evaluate and persist a `Decisions` row unless already recorded for the pair."""
    is_e2e_int = 1 if is_e2e else 0
    primary_metric = str(protocol_row["primary_metric"])
    with db.transaction(conn):
        if is_e2e:
            camp_proto = db.exec_query(
                conn,
                "get_campaign_by_id",
                id=campaign_id,
            ).fetchone()
            if camp_proto is None or int(protocol_row["id"]) != int(
                    camp_proto["protocol"]):
                raise constants.InvalidStateError(
                    "is_e2e=True Decisions may only be recorded under the main E2E protocol"
                )

        existing = db.exec_query(
            conn,
            "get_existing_decision",
            is_e2e=is_e2e_int,
            subject_measurement=subject_measurement_id,
            reference_measurement=reference_measurement_id,
        ).fetchone()

        subj_cells = load_measurement_primary_metrics(conn,
                                                      subject_measurement_id,
                                                      primary_metric)
        ref_cells = load_measurement_primary_metrics(conn,
                                                     reference_measurement_id,
                                                     primary_metric)
        ev = evaluate_benchmark_comparison(
            protocol_row=protocol_row,
            protocol_cells=protocol_cells,
            subject_cells=subj_cells,
            reference_cells=ref_cells,
        )
        if existing is not None:
            return int(existing["id"]), ev

        cur = db.exec_query(
            conn,
            "insert_decision",
            round=round_id,
            is_e2e=is_e2e_int,
            subject_measurement=subject_measurement_id,
            reference_measurement=reference_measurement_id,
            verdict=ev.verdict.value,
            reason=ev.reason,
        )
        decision_id = int(cur.lastrowid)

    telem.record(
        "decision.recorded",
        campaign_id=campaign_id,
        round_id=round_id,
        decision_id=decision_id,
        is_e2e=is_e2e,
        subject_measurement_id=subject_measurement_id,
        reference_measurement_id=reference_measurement_id,
        weighted_geomean_delta_pct=ev.weighted_geomean_delta_pct,
        worst_cell_delta_pct=ev.worst_cell_delta_pct,
        verdict=ev.verdict.value,
        reason=ev.reason,
    )
    return decision_id, ev
