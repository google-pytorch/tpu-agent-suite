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
"""Minimal unit tests for `framework.hc.decisions` and `framework.hc.candidate_state`."""

from __future__ import annotations

import sqlite3
import unittest

from framework.hc import candidate_state, constants, db, decisions, telemetry
from framework.tests.hc import _helpers


class TestDecisionsAndStates(_helpers.HcCliTestCase):
    """Tests benchmark verdicts, accuracy gates, and `CandidateState`."""

    def test_benchmark_verdicts_and_weighted_geomean(self) -> None:
        proto = dict(
            primary_metric="step_ms",
            primary_metric_direction="minimum",
            min_replicates=2,
            min_win_percentage=2.0,
            noise_floor_percentage=0.5,
            max_cell_regression_percentage=1.5,
        )
        cells = [
            dict(id=10, cell_index=0, weight=3.0),
            dict(id=11, cell_index=1, weight=1.0),
        ]
        ref = {10: (100.0, 2), 11: (200.0, 2)}
        for subj, expected in (
            ({
                10: (95.0, 1),
                11: (190.0, 2)
            }, constants.DecisionVerdict.INVALID),
            ({
                10: (90.0, 2),
                11: (204.0, 2)
            }, constants.DecisionVerdict.INVALID),
            ({
                10: (100.0 / 1.005, 2),
                11: (200.0 / 1.005, 2)
            }, constants.DecisionVerdict.NEUTRAL),
            ({
                10: (100.0 / 1.01, 2),
                11: (200.0 / 1.01, 2)
            }, constants.DecisionVerdict.VIABLE),
            ({
                10: (100.0 / 0.99, 2),
                11: (200.0 / 0.99, 2)
            }, constants.DecisionVerdict.LOSS),
            ({
                10: (100.0 / 1.20, 2),
                11: (200.0, 2)
            }, constants.DecisionVerdict.WIN),
        ):
            ev = decisions.evaluate_benchmark_comparison(
                protocol_row=proto,
                protocol_cells=cells,
                subject_cells=subj,
                reference_cells=ref,
            )
            self.assertEqual(ev.verdict, expected)
        assert ev.weighted_geomean_delta_pct is not None
        self.assertAlmostEqual(ev.weighted_geomean_delta_pct,
                               ((1.20**0.75) - 1.0) * 100.0)

    def test_accuracy_gates(self) -> None:
        gate = dict(
            has_accuracy_gate=1,
            accuracy_gate_comparator="gte",
            accuracy_gate_threshold=-0.015,
            accuracy_gate_is_relative=1,
            accuracy_gate_relative_to=None,
        )
        self.assertIsNone(decisions.evaluate_accuracy_gate(gate, 0.80))
        anchored = {**gate, "accuracy_gate_relative_to": 0.80}
        self.assertTrue(decisions.evaluate_accuracy_gate(anchored, 0.785))
        self.assertFalse(decisions.evaluate_accuracy_gate(anchored, 0.784))
        abs_lte = {
            **anchored,
            "accuracy_gate_comparator": "lte",
            "accuracy_gate_is_relative": 0,
            "accuracy_gate_threshold": 2.5,
        }
        self.assertTrue(decisions.evaluate_accuracy_gate(abs_lte, 2.5))
        self.assertFalse(decisions.evaluate_accuracy_gate(abs_lte, 2.501))

    def test_candidate_state_ladder(self) -> None:
        self.assertEqual(
            self._run_hc("campaign", "open", str(self.camp_json),
                         str(self.proto_json), "--code-dir",
                         str(self.code_dir), "--microbenchmark-protocol",
                         str(self.micro_json))[0],
            0,
        )

        def _meas(conn: sqlite3.Connection, trial: str, subj: int, proto: int,
                  cell: int, val: float) -> int:
            with db.transaction(conn):
                cur = conn.execute(
                    "INSERT INTO Measurements (trial, subject, protocol, env_image) "
                    "VALUES (?, ?, ?, 'img')", (trial, subj, proto))
                mid = int(cur.lastrowid)
                conn.execute(
                    "INSERT INTO Measurement_Metrics (measurement_id, cell_id, metric_name, value, n) "
                    "VALUES (?, ?, 'output_tokens_per_s', ?, 2)",
                    (mid, cell, val))
            return mid

        def _state(conn: sqlite3.Connection, cid: int) -> tuple[str, int]:
            camp = conn.execute(
                "SELECT * FROM Campaigns WHERE id = 1").fetchone()
            cand = conn.execute("SELECT * FROM Candidates WHERE id = ?",
                                (cid, )).fetchone()
            st, prior = candidate_state.compute_candidate_state_and_prior_decisions(
                conn, campaign_row=camp, candidate_row=cand)
            return st.value, prior

        with db.open_db() as conn:
            m1 = _meas(conn, "t0", 1, 1, 1, 1000.0)
            with db.transaction(conn):
                conn.execute(
                    "UPDATE Protocols SET accuracy_gate_relative_to = 0.80 WHERE id = 1"
                )
                conn.execute(
                    "UPDATE Campaigns SET state = 'Active', champion = 1, "
                    "entry_baseline = ?, current_baseline = ? WHERE id = 1",
                    (m1, m1))
            self.assertEqual(self._run_hc("round", "open")[0], 0)
            for idx in (2, 3):
                if idx == 3:
                    self._run_hc("candidate", "prepare", "1")
                (self.workspace / "src" / "model.py").write_text(
                    f"X = {idx}\n", encoding="utf-8")
                self._run_hc("candidate", "propose", "--innovation", f"c{idx}",
                             "--short-innovation", f"c{idx}")

            p_row = conn.execute(
                "SELECT * FROM Protocols WHERE id = 1").fetchone()
            p_cells = conn.execute(
                "SELECT * FROM Protocol_Cells WHERE protocol_id = 1").fetchall(
                )
            telem = telemetry.TelemetryBuffer(run_id=self.run_id,
                                              campaign_id=1,
                                              round_id=1)

            self.assertEqual((_state(conn, 1), _state(conn, 2)),
                             (("Promoted", 0), ("Untested", 0)))

            m_c2 = _meas(conn, "te2", 2, 1, 1, 1050.0)
            decisions.record_decision_if_needed(
                conn,
                campaign_id=1,
                round_id=1,
                is_e2e=True,
                subject_measurement_id=m_c2,
                reference_measurement_id=m1,
                protocol_row=p_row,
                protocol_cells=p_cells,
                telem=telem,
            )
            self.assertEqual(_state(conn, 2)[0], "E2E_Pending_Accuracy")

            with db.transaction(conn):
                conn.execute(
                    "INSERT INTO Jobs (job_id, round, candidate, protocol, kind, "
                    "accuracy_value, passed_accuracy, status) "
                    "VALUES ('ja2', 1, 2, 1, 'Accuracy', 0.80, 1, 'Succeeded')"
                )
            self.assertEqual(_state(conn, 2)[0], "E2E_Passed")

            m_c3 = _meas(conn, "te3", 3, 1, 1, 1012.0)
            decisions.record_decision_if_needed(
                conn,
                campaign_id=1,
                round_id=1,
                is_e2e=True,
                subject_measurement_id=m_c3,
                reference_measurement_id=m1,
                protocol_row=p_row,
                protocol_cells=p_cells,
                telem=telem,
            )
            self.assertEqual(_state(conn, 3), ("E2E_Viable", 0))

            with db.transaction(conn):
                conn.execute(
                    "UPDATE Campaigns SET champion = 2, current_baseline = ? WHERE id = 1",
                    (m_c2, ))
            self.assertEqual(_state(conn, 2)[0], "Promoted")
            self.assertEqual(_state(conn, 3), ("Untested", 1))


if __name__ == "__main__":
    unittest.main()
