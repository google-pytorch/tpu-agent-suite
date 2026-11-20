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
"""Unit tests for the normalized `STRICT` SQLite coordination store (`framework.hc.db`)."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from unittest import mock

from framework.hc import constants, db


class TestCoordinationDatabase(unittest.TestCase):
    """Verify WAL, FK enforcement, SAVEPOINT nesting, open_db, and CHECKs."""

    def test_pragmas_strict_tables_and_constraints(self):
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(
                os.environ, {"TPU_AGENT_SUITE_HOME": home}):
            with db.open_db() as conn:
                self.assertEqual(
                    (conn.execute("PRAGMA journal_mode").fetchone()[0].lower(),
                     conn.execute("PRAGMA foreign_keys").fetchone()[0],
                     conn.execute("PRAGMA user_version").fetchone()[0]),
                    ("wal", 1, db.SCHEMA_VERSION),
                )
                rows = conn.execute(
                    "SELECT name, strict FROM pragma_table_list "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
                found = {row["name"]: row["strict"] for row in rows}
                expected = set(
                    "Campaigns Protocols Rounds Hypotheses Candidates Jobs Measurements "
                    "Reviews Decisions Protocol_Cells Hypothesis_Canonical_Ops "
                    "Hypothesis_Relayout_Edges Hypothesis_Evidence Hypothesis_Code_Refs "
                    "Hypothesis_Candidates Candidate_Parents Measurement_Metrics "
                    "Measurement_Artifacts Measurement_Jobs".split())
                self.assertEqual(set(found), expected)
                self.assertTrue(all(strict == 1 for strict in found.values()))

                # Deferred `initial_version` FK (insert placeholder 0 -> insert Candidate -> UPDATE)
                # + nested SAVEPOINT rollback.
                with db.transaction(conn):
                    proto_id = conn.execute(
                        "INSERT INTO Protocols (workload_kind, primary_metric, compute_profile, "
                        "primary_metric_unit, primary_metric_direction, primary_metric_statistic, "
                        "min_replicates, min_win_percentage, noise_floor_percentage, has_accuracy_gate, "
                        "accuracy_gate_metric, accuracy_gate_comparator, accuracy_gate_threshold, "
                        "accuracy_gate_is_relative, exec_image, exec_e2e_script) VALUES "
                        "('inference','tps','gke','tok/s','maximum','mean',3,2.0,1.0,1,'acc','gte',-0.01,1,'img','e2e.sh')"
                    ).lastrowid
                    camp_id = conn.execute(
                        "INSERT INTO Campaigns (name, owner, target_model, engine, accelerator, "
                        "created_at, protocol, initial_version, state) VALUES "
                        "('c1','alice','gemma','vllm','v6e','2026-11-10T00:00:00Z',?,0,'Unanchored')",
                        (proto_id, ),
                    ).lastrowid
                    cand_id = conn.execute(
                        "INSERT INTO Candidates (campaign_id, round, sha, tree_sha, patch_sha, "
                        "short_innovation, innovation) VALUES (?,NULL,'a1','t1','p1','init','init commit')",
                        (camp_id, ),
                    ).lastrowid
                    conn.execute(
                        "UPDATE Campaigns SET initial_version = ? WHERE id = ?",
                        (cand_id, camp_id),
                    )
                    rnd_id = conn.execute(
                        "INSERT INTO Rounds (campaign, round_number, current_champion, state) "
                        "VALUES (?, 1, ?, 'Open')",
                        (camp_id, cand_id),
                    ).lastrowid
                    with self.assertRaises(RuntimeError):
                        with db.transaction(conn):
                            conn.execute(
                                "INSERT INTO Rounds (campaign, round_number, current_champion, state) "
                                "VALUES (?, 2, ?, 'Open')",
                                (camp_id, cand_id),
                            )
                            raise RuntimeError("abort inner savepoint")
                    hyp_id = conn.execute(
                        "INSERT INTO Hypotheses (round_created, lever_kind, target_scope, "
                        "short_claim, state, claim) VALUES (?, 'kernel', 'step', 's', 'Open', 'c')",
                        (rnd_id, ),
                    ).lastrowid
                self.assertEqual(
                    (conn.execute(
                        "SELECT initial_version FROM Campaigns WHERE id = ?",
                        (camp_id, )).fetchone()[0],
                     conn.execute("SELECT COUNT(*) FROM Rounds").fetchone()[0]
                     ),
                    (cand_id, 1),
                )

                # Leaving placeholder `initial_version = 0` un-updated fails at COMMIT and rolls back.
                with self.assertRaises(sqlite3.IntegrityError):
                    with db.transaction(conn):
                        conn.execute(
                            "INSERT INTO Campaigns (name, owner, target_model, engine, accelerator, "
                            "created_at, protocol, initial_version, state) VALUES "
                            "('bad','alice','m','vllm','v6e','2026-11-10T00:00:00Z',?,0,'Unanchored')",
                            (proto_id, ),
                        )
                self.assertFalse(conn.in_transaction)

                rev_sql = (
                    "INSERT INTO Reviews (candidate, hypothesis, human_authored, reviewer, "
                    "approval, short_comments, comments) VALUES (?,?,0,'bot','Pass','s','c')"
                )
                bad_stmts = [
                    ("UPDATE Campaigns SET num_consecutive_barren_rounds = 'bad' WHERE id = 1",
                     ()),
                    ("INSERT INTO Candidates (campaign_id, round, sha, tree_sha, patch_sha, "
                     "short_innovation, innovation) VALUES (1,NULL,'a2','t2','p2',?,'ok')",
                     ("x" * (constants.MAX_SHORT_LENGTH + 1), )),
                    (rev_sql, (1, hyp_id)),
                    (rev_sql, (None, None)),
                    ("INSERT INTO Hypothesis_Relayout_Edges (hypothesis_id, edge_index, "
                     "producer_layer_kind, producer_operator) VALUES (?,0,NULL,'qkv')",
                     (hyp_id, )),
                ]
                for sql, params in bad_stmts:
                    with self.subTest(params=params), self.assertRaises(
                            sqlite3.IntegrityError):
                        with db.transaction(conn):
                            conn.execute(sql, params)
            with self.assertRaises(sqlite3.ProgrammingError):
                conn.execute("SELECT 1")

            with sqlite3.connect(str(db.paths.get_campaign_db_path())) as raw:
                raw.execute("PRAGMA user_version = 99")
            with self.assertRaises(constants.InvalidStateError):
                db.connect()


if __name__ == "__main__":
    unittest.main()
