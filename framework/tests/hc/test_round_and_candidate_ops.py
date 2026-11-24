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
"""Unit tests for rounds, hypotheses, candidates, and qualitative reviews."""

from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from framework import jsonio
from framework.hc import constants, db, paths
from framework.tests.hc import _helpers


class TestRoundHypothesisCandidateAndReviewOps(_helpers.HcCliTestCase):
    """End-to-end tests for round, hypothesis, candidate, and review commands."""

    def setUp(self) -> None:
        super().setUp()
        self.assertEqual(
            self._run_hc("campaign", "open", str(self.camp_json),
                         str(self.proto_json), "--code-dir",
                         str(self.code_dir))[0], 0)

    def test_rounds_and_hypotheses(self) -> None:
        self.assertNotEqual(self._run_hc("round", "open")[0], 0)
        self._anchor_campaign()

        hyp1 = self._write_hyp(
            "hyp1.json",
            target_scope="canonical_ops",
            target_canonical_ops=[
                dict(layer_kind="decoder", operator="flash_attention")
            ],
            evidence=[
                dict(source_type="xplane",
                     locator="runs/trace.xplane.pb",
                     summary="flash_attention occupies 42% of step time.")
            ],
        )
        self.assertNotEqual(self._run_hc("hypothesis", "add", str(hyp1))[0], 0)

        rc_open, out_open = self._run_hc("round", "open")
        self.assertEqual((rc_open, json.loads(out_open)["round_number"]),
                         (0, 1))
        self.assertNotEqual(self._run_hc("round", "open")[0], 0)

        self.assertEqual(self._run_hc("hypothesis", "add", str(hyp1))[0], 0)
        op_qkv = dict(layer_kind="decoder", operator="qkv_proj")
        op_fa = dict(layer_kind="decoder", operator="flash_attention")
        op_out = dict(layer_kind="decoder", operator="o_proj")
        hyp2 = self._write_hyp(
            "hyp2.json",
            lever_kind="sharding",
            target_scope="redundant_relayout",
            target_relayout_edges=[
                dict(producer=op_qkv, consumer=op_fa),
                dict(producer=op_fa, consumer=op_out),
            ],
        )
        self.assertEqual(self._run_hc("hypothesis", "add", str(hyp2))[0], 0)

        self.assertEqual(
            self._run_hc("hypothesis", "refute", "1", "VREG spills.")[0], 0)
        self.assertNotEqual(
            self._run_hc("hypothesis", "refute", "1", "Overwrite.")[0], 0)

        with db.open_db() as conn:
            with db.transaction(conn):
                conn.execute(
                    "UPDATE Hypotheses SET state = 'Promoted' WHERE id = 2")
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM Hypothesis_Canonical_Ops WHERE hypothesis_id = 1"
                ).fetchone()["c"], 1)
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM Hypothesis_Relayout_Edges WHERE hypothesis_id = 2"
                ).fetchone()["c"], 2)
        self.assertNotEqual(
            self._run_hc("hypothesis", "refute", "2", "Promoted.")[0], 0)

    def test_candidates_lineage_merges_supersedes_abandon_and_reviews(
            self) -> None:
        self._anchor_campaign()
        self.assertEqual(self._run_hc("round", "open")[0], 0)
        hyp_json = self._write_hyp()
        for _ in range(2):
            self.assertEqual(
                self._run_hc("hypothesis", "add", str(hyp_json))[0], 0)
        self.assertEqual(
            self._run_hc("hypothesis", "refute", "2", "Out of VMEM")[0], 0)

        src_file = self.workspace / "src" / "model.py"
        src_file.write_text("TILE_M = 256\nTILE_N = 128\n", encoding="utf-8")

        self.assertNotEqual(
            self._run_hc("candidate", "propose", "--hypotheses", "2",
                         "--innovation", "Refuted", "--short-innovation",
                         "Refuted")[0], 0)
        rc_c2, out_c2 = self._run_hc("candidate", "propose", "--hypotheses",
                                     "1", "--innovation", "Set TILE_M = 256.",
                                     "--short-innovation", "Set TILE_M = 256.")
        self.assertEqual((rc_c2, json.loads(out_c2)["candidate_id"]), (0, 2))

        self.assertEqual(self._run_hc("candidate", "prepare", "1")[0], 0)
        src_file.write_text("TILE_M = 128\nTILE_N = 256\n", encoding="utf-8")
        self.assertEqual(
            self._run_hc("candidate", "propose", "--hypotheses", "1",
                         "--innovation", "Set TILE_N = 256.",
                         "--short-innovation", "Set TILE_N = 256.")[0], 0)

        with db.open_db() as conn, db.transaction(conn):
            conn.execute("UPDATE Rounds SET state = 'Progressed' WHERE id = 1")
            conn.execute("UPDATE Campaigns SET champion = 2 WHERE id = 1")
        self.assertEqual(self._run_hc("round", "open")[0], 0)

        self.assertNotEqual(self._run_hc("candidate", "prepare", "3")[0], 0)
        rc_merge, out_merge = self._run_hc("candidate", "prepare", "2", "3")
        self.assertEqual(rc_merge, 0)
        if json.loads(out_merge)["has_conflicts"]:
            src_file.write_text("TILE_M = 256\nTILE_N = 256\n",
                                encoding="utf-8")

        self.assertNotEqual(
            self._run_hc("candidate", "propose", "--supersedes", "2",
                         "--innovation", "x", "--short-innovation", "x")[0], 0)
        rc_c4, out_c4 = self._run_hc("candidate", "propose", "--hypotheses",
                                     "1", "--supersedes", "3", "--innovation",
                                     "Merged 256x256.", "--short-innovation",
                                     "Merged 256x256.")
        self.assertEqual(rc_c4, 0)
        c4 = json.loads(out_c4)
        self.assertEqual((c4["candidate_id"], c4["parents"], c4["supersedes"]),
                         (4, [2, 3], 3))

        for bad_cid in ("1", "2", "3"):
            self.assertNotEqual(
                self._run_hc("candidate", "abandon", bad_cid, "nope")[0], 0)
        self.assertEqual(
            self._run_hc("candidate", "abandon", "4", "Timeout")[0], 0)
        self.assertNotEqual(
            self._run_hc("candidate", "propose", "--supersedes", "4",
                         "--innovation", "x", "--short-innovation", "x")[0], 0)

        for bad_rev in (
            ("--candidate", "4", "--approval", "ConditionalPass", "--comments",
             "x", "--conditions", "   "),
            ("--candidate", "4", "--approval", "Pass", "--comments", "x",
             "--conditions", "extra"),
            ("--candidate", "4", "--hypothesis", "1", "--approval", "Pass",
             "--comments", "XOR"),
        ):
            self.assertNotEqual(self._run_hc("review", "add", *bad_rev)[0], 0)

        with mock.patch.dict(
                os.environ,
            {
                "HC_REVIEWER": "critic-subagent",
                "HC_HUMAN_AUTHORED": "0"
            },
        ):
            rc_rev, out_rev = self._run_hc("review", "add", "--candidate", "4",
                                           "--approval", "ConditionalPass",
                                           "--comments", "  " + ("R" * 620),
                                           "--conditions",
                                           "  Verify bf16 numerics.  ")
        self.assertEqual(rc_rev, 0)
        rev = json.loads(out_rev)
        self.assertEqual(
            (rev["reviewer"], rev["human_authored"], len(
                rev["short_comments"]), rev["conditions"]),
            ("critic-subagent", False, constants.MAX_SHORT_LENGTH,
             "Verify bf16 numerics."),
        )

        # Also exercise hypothesis review + human reviewer identity fallback.
        with mock.patch.dict(
                os.environ,
            {
                "HC_HUMAN_AUTHORED": "1",
                "USER": "alice",
                "HC_REVIEWER": "",
                "TPU_AGENT_SUITE_AGENT": "",
                "AGENT_NAME": "",
                "TPU_AGENT_SUITE_HARNESS": "",
            },
        ):
            rc_hrev, out_hrev = self._run_hc("review", "add", "--hypothesis",
                                             "1", "--approval", "Pass",
                                             "--comments",
                                             "Solid roofline grounding.",
                                             "--short-comments",
                                             "Solid grounding.")
        self.assertEqual((rc_hrev, json.loads(out_hrev)["reviewer"]),
                         (0, "alice"))

        names = [
            e["event_name"]
            for e in jsonio.read_jsonl(paths.get_campaign_events_log(1))
        ]
        for expected in ("round.opened", "hypothesis.added",
                         "hypothesis.refuted", "candidate.prepared",
                         "candidate.proposed", "candidate.superseded",
                         "candidate.abandoned", "review.added"):
            self.assertIn(expected, names)


if __name__ == "__main__":
    unittest.main()
