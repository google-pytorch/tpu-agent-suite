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
"""Arm verdicts, study locking, section caps, and arm claim/release."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import unittest

# Imports below follow this sys.path setup, so the file runs directly.
# ruff: noqa: E402
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import runtime, study
from framework.tests.study.helpers import (RoundRecordTestCase,
                                           StudyStoreTestCase, open_arm)

Path = pathlib.Path


class TestComputeArmVerdict(unittest.TestCase):
    """Unit tests for compute_arm_verdict invariants."""

    def test_verdict_and_delta_against_the_noise_floor(self):
        # (metric_value, direction, gates_passed, replicates, verdict, delta),
        # against a baseline of 100.0, a 1% noise floor and 3 required
        # replicates.
        cases = {
            "win_exceeds_noise_floor_maximize":
            (105.0, "maximize", True, 3, "win", 5.0),
            "win_exceeds_noise_floor_minimize":
            (90.0, "minimize", True, 3, "win", 10.0),
            "tie_within_noise_floor": (100.5, "maximize", True, 3, "tie", 0.5),
            "loss_below_negative_noise_floor":
            (98.0, "maximize", True, 3, "loss", -2.0),
            # A faster number that breaks an admissibility gate is
            # inadmissible ('invalid'), and cannot count.
            "admissibility_gate_failure_is_invalid":
            (120.0, "maximize", False, 3, "invalid", 20.0),
            # Exactly at the noise floor is a tie, not a win.
            "noise_floor_exact_boundary_is_tie": (101.0, "maximize", True, 3,
                                                  "tie", 1.0),
            "insufficient_replicates_remains_tie": (120.0, "maximize", True, 1,
                                                    "tie", 20.0),
        }
        for name, (value, direction, gates_passed, replicates, want_verdict,
                   want_delta) in cases.items():
            with self.subTest(name):
                verdict, delta = study.compute_arm_verdict(
                    metric_value=value,
                    baseline_value=100.0,
                    direction=direction,
                    noise_floor_pct=1.0,
                    gates_passed=gates_passed,
                    replicates=replicates,
                    required_replicates=3,
                )
                self.assertEqual(verdict, want_verdict)
                self.assertEqual(delta, want_delta)
        # A zero baseline or an unknown direction cannot yield a comparison.
        for baseline, direction in ((0.0, "maximize"), (100.0, "unknown")):
            with self.subTest(baseline=baseline, direction=direction):
                verdict, _ = study.compute_arm_verdict(100.0, baseline,
                                                       direction)
                self.assertEqual(verdict, "invalid")


class StudyLockingAndSectionCapsTestCase(StudyStoreTestCase):
    """Advisory file locking and state section cap enforcement."""

    def test_save_study_caps_amendments_and_archives_oldest_to_journal(self):
        self.start("capped-amendments")
        total = study.MAX_STUDY_AMENDMENTS + 3
        for idx in range(total):
            updated = self.protocol(noise_floor_pct=float(idx + 1))
            study.set_protocol("capped-amendments",
                               updated,
                               reason=f"reason-{idx}")

        doc = study.load_study("capped-amendments")
        self.assertEqual(doc["protocol_revision"], total + 1)
        self.assertEqual(len(doc["amendments"]), study.MAX_STUDY_AMENDMENTS)
        self.assertEqual([a["reason"] for a in doc["amendments"]],
                         [f"reason-{i}" for i in range(3, total)])
        # Each amendment keeps the protocol it superseded.
        self.assertEqual(doc["amendments"][-1]["revision"], total)
        self.assertEqual(doc["amendments"][-1]["protocol"]["noise_floor_pct"],
                         float(total - 1))

        journal = study.read_study_journal("capped-amendments")
        self.assertEqual(len(journal), 3)
        self.assertEqual([j["evicted"]["reason"] for j in journal],
                         ["reason-0", "reason-1", "reason-2"])

    def test_save_study_caps_arms_evicting_weakest_oldest_to_journal(self):
        total = study.MAX_STUDY_ARMS + 3
        arms = []
        for idx in range(total):
            if idx == 0:
                status, verdict, delta = "decided", "invalid", None
            elif idx in (1, 2):
                status, verdict, delta = "decided", "loss", -4.0
            elif idx == total - 1:
                status, verdict, delta = "in_flight", None, None
            elif idx == total - 2:
                status, verdict, delta = "open", None, None
            elif idx == 3:
                status, verdict, delta = "decided", "win", 5.0
            else:
                status, verdict, delta = "decided", "tie", 0.5

            arm_entry = open_arm(f"a{idx}", f"claim-{idx}", status=status)
            if verdict is not None:
                arm_entry["result"] = dict(verdict=verdict,
                                           metric=dict(delta_pct=delta))
            arms.append(arm_entry)

        self.start_with_arms("capped-arms", arms)

        saved = study.load_study("capped-arms")
        self.assertEqual(len(saved["arms"]), study.MAX_STUDY_ARMS)
        kept_ids = {a["arm"] for a in saved["arms"]}
        self.assertFalse({"a0", "a1", "a2"} & kept_ids)
        self.assertLessEqual({"a3", f"a{total - 2}", f"a{total - 1}"},
                             kept_ids)

        journal = study.read_study_journal("capped-arms")
        self.assertEqual(len(journal), 3)
        self.assertEqual([j["evicted"]["arm"] for j in journal],
                         ["a0", "a1", "a2"])


class StudyArmClaimAndReleaseTestCase(RoundRecordTestCase):
    """Atomic arm reservation and release for parallel worker fleets."""

    def test_claim_arm_auto_selects_highest_expected_gain_first(self):
        self.start_with_arms("claim-order", [
            open_arm("a0", "c0", share=0.1, headroom=0.2),
            open_arm("a1", "c1", share=0.5, headroom=0.8),
            open_arm("a2", "c2", share=0.3, headroom=0.5),
        ])

        self.assertEqual(
            [study.claim_arm("claim-order")["arm"] for _ in range(3)],
            ["a1", "a2", "a0"])

        with self.assertRaises(study.TpuAgentSuiteError) as ctx:
            study.claim_arm("claim-order")
        self.assertIn("no open arms available", str(ctx.exception))

    def test_claim_arm_refuses_non_open_arm(self):
        self.start_with_arms("claim-refuse", [
            open_arm("a0", "c0", status="decided"),
            open_arm("a1", "c1", status="in_flight"),
        ])

        for arm_id, message in (("a0", "cannot be claimed"),
                                ("a1", "cannot be claimed"), ("nonexistent",
                                                              "not found")):
            with self.subTest(arm=arm_id):
                with self.assertRaisesRegex(study.TpuAgentSuiteError, message):
                    study.claim_arm("claim-refuse", arm_id=arm_id)

    def test_release_arm_transitions_in_flight_to_open_or_blocked(self):
        self.start_with_arms("release-test", [open_arm("a0", "c0")])

        study.claim_arm("release-test", arm_id="a0", worker_id="w-temp")
        released = study.release_arm("release-test",
                                     arm_id="a0",
                                     status="open",
                                     reason="preflight retry")
        self.assertEqual(released["status"], "open")
        self.assertNotIn("claimed_at", released)
        self.assertNotIn("claimed_by", released)
        self.assertEqual(released["release_reason"], "preflight retry")

        study.claim_arm("release-test", arm_id="a0", worker_id="w-temp2")
        blocked = study.release_arm("release-test",
                                    arm_id="a0",
                                    status="blocked",
                                    reason="awaiting cluster capacity")
        self.assertEqual(blocked["status"], "blocked")
        self.assertEqual(blocked["release_reason"],
                         "awaiting cluster capacity")

        with self.assertRaises(study.TpuAgentSuiteError) as ctx:
            study.release_arm("release-test", arm_id="a0")
        self.assertIn("cannot be released", str(ctx.exception))

    def test_failed_dispatches_reopen_then_block_the_arm(self):
        self.start_with_arms("failed-dispatch", [open_arm("a0", "c0")])

        study.claim_arm("failed-dispatch", arm_id="a0")
        first = study.release_arm("failed-dispatch",
                                  arm_id="a0",
                                  failed=True,
                                  summary="recompile per step")
        self.assertEqual(first["status"], "open")
        self.assertEqual(first["failed_dispatches"], 1)
        self.assertNotIn("result", first)

        # A round that rewrites the arms without the count does not reset it.
        study.record_round("failed-dispatch", {"arms": [open_arm("a0", "c0")]})
        self.assertEqual(
            study.load_study("failed-dispatch")["arms"][0]
            ["failed_dispatches"], 1)

        study.claim_arm("failed-dispatch", arm_id="a0")
        second = study.release_arm("failed-dispatch",
                                   arm_id="a0",
                                   failed=True,
                                   summary="NaN in cache update")
        self.assertEqual(second["status"], "blocked")
        self.assertEqual(second["failed_dispatches"],
                         study.MAX_FAILED_DISPATCHES)
        self.assertIn("NaN in cache update", second["blocked_on"])

    def test_failed_release_needs_a_summary_and_no_verdict(self):
        self.start_with_arms("failed-dispatch-edges", [open_arm("a0", "c0")])
        study.claim_arm("failed-dispatch-edges", arm_id="a0")
        with self.assertRaises(study.TpuAgentSuiteError) as ctx:
            study.release_arm("failed-dispatch-edges",
                              arm_id="a0",
                              failed=True)
        self.assertIn("summary", str(ctx.exception))
        with self.assertRaises(study.TpuAgentSuiteError) as ctx:
            study.release_arm("failed-dispatch-edges",
                              arm_id="a0",
                              status="invalid",
                              failed=True,
                              summary="x")
        self.assertIn("no verdict", str(ctx.exception))

    def test_multiprocess_claim_arm_races_without_double_claims(self):
        import subprocess
        import sys

        self.start_with_arms("mp-claims", [
            open_arm(f"mp-arm-{i}",
                     f"c{i}",
                     share=0.5,
                     headroom=float(i + 1) / 10.0) for i in range(4)
        ])

        env = dict(os.environ,
                   PYTHONPATH=str(Path(study.__file__).resolve().parents[1]))

        procs = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    ("import json; from framework import study; "
                     f"print(json.dumps(study.claim_arm('mp-claims', worker_id='p-{i}')))"
                     ),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            ) for i in range(8)
        ]
        claimed_ids: list[str] = []
        refused = 0
        for proc in procs:
            out, err = proc.communicate(timeout=10)
            if proc.returncode == 0:
                claimed_ids.append(json.loads(out.strip())["arm"])
            else:
                self.assertIn("no open arms available", err)
                refused += 1

        self.assertEqual(len(claimed_ids), 4)
        self.assertEqual(len(set(claimed_ids)), 4)
        self.assertEqual(refused, 4)

    def test_cli_claim_and_release_arm(self):
        self.start_with_arms("cli-claim", [open_arm("a0", "c0")])

        def cli(*argv):
            return runtime.main(["study", *argv, "--study-id", "cli-claim"])

        self.assertEqual(cli("claim-arm", "--worker-id", "worker-cli"), 0)
        arm = study.load_study("cli-claim")["arms"][0]
        self.assertEqual((arm["status"], arm["claimed_by"]),
                         ("in_flight", "worker-cli"))
        self.assertEqual(
            cli("release-arm", "--arm", "a0", "--status", "open", "--reason",
                "cli release"), 0)
        arm = study.load_study("cli-claim")["arms"][0]
        self.assertEqual((arm["status"], arm["release_reason"]),
                         ("open", "cli release"))

    def test_record_round_merges_concurrent_arms_without_clobbering_decided_or_in_flight(
            self):
        sid = "merge-concurrent"
        self.start_based(sid, admissibility_gates=[])
        study.record_round(
            sid, {"arms": [open_arm(f"a{i}", f"c{i}") for i in (1, 2, 3)]})
        # Both workers snapshot the study before w2 claims a2.
        stale = {a["arm"]: a for a in study.load_study(sid)["arms"]}
        study.claim_arm(sid, arm_id="a2", worker_id="w2")

        def decided(arm_id, verdict, **metric):
            return dict(open_arm(arm_id, "c", status="decided"),
                        result=dict(verdict=verdict,
                                    execution_id=f"job-{arm_id}",
                                    metric=self.metric(**metric),
                                    gates=[]))

        # w1 records a1 beside its stale (open) copy of a2, omitting a3.
        merged = study.record_round(
            sid, {
                "arms": [
                    decided("a1", "win", value=120.0, delta_pct=20.0),
                    stale["a2"]
                ]
            })
        by_id = {a["arm"]: a for a in merged["arms"]}
        self.assertEqual(by_id["a1"]["status"], "decided")
        # w2's claim is not regressed to open; the omitted a3 survives.
        self.assertEqual((by_id["a2"]["status"], by_id["a2"]["claimed_by"]),
                         ("in_flight", "w2"))
        self.assertEqual(by_id["a3"]["status"], "open")

        # w2 records a2 beside its stale (open) copy of a1.
        merged = study.record_round(
            sid, {"arms": [stale["a1"],
                           decided("a2", "tie", delta_pct=0.0)]})
        by_id = {a["arm"]: a for a in merged["arms"]}
        self.assertEqual(by_id["a1"]["result"]["verdict"], "win")
        self.assertEqual(by_id["a2"]["status"], "decided")

    def test_release_arm_supports_invalid_requires_blocked_on_and_clears_stale_fields(
            self):
        sid = "release-edge-cases"
        self.start_with_arms(sid, [open_arm("a0", "c0"), open_arm("a1", "c")])

        def release(arm_id="a0", **kwargs):
            study.claim_arm(sid, arm_id=arm_id)
            return study.release_arm(sid, arm_id=arm_id, **kwargs)

        # A summary doubles as the reason; a later bare release clears both.
        rel = release(status="open", summary="first failure")
        self.assertEqual((rel["summary"], rel["release_reason"]),
                         ("first failure", "first failure"))
        rel = release(status="open")
        self.assertNotIn("summary", rel)
        self.assertNotIn("release_reason", rel)

        # Blocking needs a blocked_on (or a reason to stand in for one).
        with self.assertRaisesRegex(study.TpuAgentSuiteError, "blocked_on"):
            release(status="blocked")
        rel = study.release_arm(sid,
                                arm_id="a0",
                                status="blocked",
                                blocked_on="awaiting v7x reservation")
        self.assertEqual((rel["status"], rel["blocked_on"]),
                         ("blocked", "awaiting v7x reservation"))

        # Releasing as invalid closes the arm with that verdict.
        rel = release("a1",
                      status="invalid",
                      summary="failed syntax preflight")
        self.assertEqual((rel["status"], rel["result"]["verdict"]),
                         ("decided", "invalid"))
        self.assertEqual(rel["summary"], "failed syntax preflight")

    def test_claim_arm_expires_stale_in_flight_leases_and_formats_pdt_timezone(
            self):
        self.start_with_arms("claim-ttl", [
            open_arm("orphaned-arm",
                     "c0",
                     status="in_flight",
                     claimed_by="crashed-worker",
                     claimed_at="2020-01-01T00:00:00-08:00"),
        ])

        # Claiming with claim_ttl_seconds=60.0 automatically recovers the
        # expired in_flight lease
        reclaimed = study.claim_arm("claim-ttl",
                                    worker_id="new-worker",
                                    claim_ttl_seconds=60.0,
                                    timezone_name="America/Los_Angeles")
        self.assertEqual(reclaimed["arm"], "orphaned-arm")
        self.assertEqual(reclaimed["claimed_by"], "new-worker")
        self.assertTrue(
            reclaimed["claimed_at"].endswith("-07:00")
            or reclaimed["claimed_at"].endswith("-08:00"),
            f"Expected America/Los_Angeles offset in {reclaimed['claimed_at']}",
        )

    def test_arm_expected_gain_falls_back_to_headroom_for_step_kinds(self):
        # Step-function kinds with share=0.0 fall back to headroom
        self.assertAlmostEqual(
            study.arm_expected_gain(
                dict(kind="memory-capacity", share=0.0, headroom=0.65)), 0.65)
        self.assertAlmostEqual(
            study.arm_expected_gain(
                dict(kind="kernel-block", share=0.4, headroom=0.5)), 0.20)


if __name__ == "__main__":
    unittest.main()
