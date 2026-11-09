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
"""Study lifecycle, protocol, identity, rounds, events and CLI."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest
from unittest import mock

# Imports below follow this sys.path setup, so the file runs directly.
# ruff: noqa: E402
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import jsonio, paths, runtime, study, utils
from framework.hooks import actions
from framework.tests.study.helpers import (RoundRecordTestCase,
                                           StudyStoreTestCase, open_arm)
from framework.utils import TpuAgentSuiteError

Path = pathlib.Path


def _hook_events(study_id: str) -> list[dict]:
    """The events a hook wrote, without the attach marker."""
    return [
        e for e in study.read_study_events(study_id) if e["kind"] == "hook"
    ]


class TestStudyLifecycle(StudyStoreTestCase):

    def test_reopen_attempt_is_refused_and_leaves_the_ledger_untouched(self):
        first = self.start("keepme", replicate_count=4)
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            self.start("keepme", replicate_count=999)
        self.assertIn("already exists", str(ctx.exception))
        self.assertEqual(
            study.load_study("keepme")["protocol"], first["protocol"])

    def test_reclosing_does_not_move_closed_at(self):
        # closed_at records when the work ended. A cleanup script run at 06:00
        # must not rewrite it to 06:00 and erase when the study really stopped.
        self.start("restamp")
        first = study.close_study("restamp", outcome="tie")
        doc = study.load_study("restamp")
        doc["closed_at"] = "2020-01-01T00:00:00Z"
        jsonio.write_json(
            paths.get_study_root("restamp") / study.STUDY_DOCUMENT, doc)

        again = study.close_study("restamp")
        self.assertEqual(again["closed_at"], "2020-01-01T00:00:00Z")
        self.assertEqual(again["outcome"], "tie")
        self.assertEqual(first["status"], study.STUDY_STATUS_CLOSED)
        self.assertIsNotNone(first["closed_at"])
        # A late outcome can still be recorded on an already-closed study.
        late = study.close_study("restamp", outcome="halted on plateau")
        self.assertEqual(late["outcome"], "halted on plateau")
        self.assertEqual(late["closed_at"], "2020-01-01T00:00:00Z")


class TestShippedExamples(unittest.TestCase):

    def test_the_shipped_examples_validate(self):
        # The worked examples are what people copy, so they have to be correct.
        for example in study.load_study_protocol_schema().get("examples", []):
            self.assertEqual(study.validate_study_protocol(example), [])
        for schema in (study.load_study_round_schema(),
                       study.load_study_baseline_schema()):
            for example in schema.get("examples", []):
                self.assertEqual(utils.validate_instance(example, schema), [])


class TestStudyIdentity(StudyStoreTestCase):
    """Target and package live on the document and are checked at start."""

    def test_target_and_package_are_required_and_checked(self):
        # The grouping key cannot be recovered from a free-form study id.
        for bad in ("", "   "):
            with self.subTest(target=bad):
                with self.assertRaises(TpuAgentSuiteError) as ctx:
                    study.start_study("untargeted",
                                      protocol=self.protocol(),
                                      package="tpu-agent-suite",
                                      target=bad)
                self.assertIn("--target", str(ctx.exception))
        for bad in ("", "Not A Package", "under_score"):
            with self.subTest(package=bad):
                with self.assertRaises(TpuAgentSuiteError):
                    study.start_study("pkg",
                                      protocol=self.protocol(),
                                      package=bad,
                                      target="m")


class TestStudyListing(StudyStoreTestCase):

    def test_a_broken_ledger_is_reported_not_silently_dropped(self):
        # A study that disappears from the board because its file broke is
        # exactly the failure the board exists to catch.
        self.start("healthy")
        broken = paths.get_study_root("healthy").parent / "broken"
        broken.mkdir()
        (broken / study.STUDY_DOCUMENT).write_text("{not json",
                                                   encoding="utf-8")
        (broken.parent / "hollow").mkdir()

        listed = {d["study_id"]: d for d in study.list_studies()}
        self.assertIn("broken", listed)
        self.assertIn("error", listed["broken"])
        self.assertEqual(listed["hollow"]["error"], "no study.json")
        self.assertIn("healthy", listed)


class TestRecordRound(RoundRecordTestCase):

    def test_a_decided_arm_needs_a_result(self):
        self.start("resultless")
        arm = open_arm("a1", "x", status="decided")
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.record_round("resultless", {"arms": [arm]})
        self.assertIn("decided without a result", str(ctx.exception))

    def test_a_failed_write_leaves_the_state_untouched(self):
        self.start_based("atomic", admissibility_gates=self.GATES)
        study.record_round("atomic", {"arms": [self.arm(arm="keep")]})
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.record_round("atomic", {"arms": [{"arm": "broken"}]})
        self.assertIn("invalid round record", str(ctx.exception))
        self.assertEqual(
            [a["arm"] for a in study.load_study("atomic")["arms"]], ["keep"])


class TestRoundHonoursTheProtocol(RoundRecordTestCase):
    """The checks that make pre-registration mean something."""

    def test_an_omitted_gate_is_not_a_passing_gate(self):
        self.start_based("omit", admissibility_gates=self.GATES)
        arm = self.arm(result=dict(gates=[self.gate("ttft")]))
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.record_round("omit", {"arms": [arm]})
        self.assertIn("accuracy", str(ctx.exception))
        self.assertIn("unmeasured gate", str(ctx.exception))

    def test_a_gate_measured_elsewhere_is_refused(self):
        # The 108-measurements failure, caught by the schema this time.
        self.start_based("split", admissibility_gates=self.GATES)
        arm = self.arm(result=dict(
            gates=[self.gate("ttft"),
                   self.gate("accuracy", "j-2")]))
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.record_round("split", {"arms": [arm]})
        self.assertIn("same_execution", str(ctx.exception))

    def test_a_win_that_fails_a_gate_is_refused(self):
        # A faster number that violates a gate is inadmissible, not a
        # smaller win. This is the champion that was crowned and withdrawn.
        self.start_based("champ", admissibility_gates=self.GATES)
        arm = self.arm(
            result=dict(verdict="win",
                        gates=[
                            self.gate("ttft", passed=False, value=6000.0),
                            self.gate("accuracy", value=0.95)
                        ]))
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.record_round("champ", {"arms": [arm]})
        self.assertIn("inadmissible", str(ctx.exception))

    def test_a_win_inside_the_noise_floor_or_under_replicated_is_refused(self):
        # Exactly at the floor is a tie, agreeing with compute_arm_verdict.
        self.start_based("unproven", admissibility_gates=self.GATES)
        cases = {
            "inside_floor": (self.metric(delta_pct=0.13), "That is a tie"),
            "at_floor": (self.metric(delta_pct=0.4), "That is a tie"),
            "under_replicated": (self.metric(replicates=1,
                                             delta_pct=9.0), "replicate"),
        }
        for name, (metric, message) in cases.items():
            with self.subTest(name):
                arm = self.arm(result={"verdict": "win", "metric": metric})
                with self.assertRaises(TpuAgentSuiteError) as ctx:
                    study.record_round("unproven", {"arms": [arm]})
                self.assertIn(message, str(ctx.exception))


class TestAWinOwesAMeasurement(RoundRecordTestCase):
    """Every bar is skipped when its input is missing, so absence must fail."""

    def test_a_verdict_with_no_metric_is_refused(self):
        # Without this the noise floor and replicate checks are both skipped,
        # so omitting the measurement clears every bar by saying nothing.
        self.start_based("silentwin", admissibility_gates=[])
        for verdict in ("win", "tie"):
            with self.subTest(verdict=verdict):
                arm = self.arm(result={"verdict": verdict, "gates": []})
                del arm["result"]["metric"]
                with self.assertRaises(TpuAgentSuiteError) as ctx:
                    study.record_round("silentwin", {"arms": [arm]})
                self.assertIn("without a metric", str(ctx.exception))
        for missing in ("replicates", "delta_pct"):
            with self.subTest(missing=missing):
                metric = self.metric(delta_pct=9.0)
                del metric[missing]
                arm = self.arm(result={
                    "verdict": "win",
                    "gates": [],
                    "metric": metric
                })
                errors = study.check_round_against_protocol(
                    {"arms": [arm]},
                    study.load_study("silentwin")["protocol"])
                self.assertTrue(any(missing in e for e in errors))

    def test_a_win_must_report_value_for_admissibility_gate(self):
        # A win must report the measurement for every gate declaring a bound.
        # Merely asserting passed=True without a value is inadmissible.
        self.start_based("gateval", admissibility_gates=self.GATES)
        arm = self.arm(result=dict(
            verdict="win",
            metric=self.metric(value=112.0, delta_pct=12.0),
            # ttft omits its "value".
            gates=[self.gate("ttft"),
                   self.gate("accuracy", value=0.95)]))
        errors = study.check_round_against_protocol(
            {"arms": [arm]},
            study.load_study("gateval")["protocol"])
        self.assertTrue(any("ttft" in e and "value" in e for e in errors))


class TestBaseline(RoundRecordTestCase):
    """The number every arm is compared against."""

    def test_an_unusable_baseline_is_refused(self):
        # Without per-op attribution the first round cannot rank arms by
        # headroom; without a fingerprint nothing can tell whether a baseline
        # is reusable, and an e2e run costs hours.
        self.start("incomplete")
        for field in ("profile", "fingerprint"):
            with self.subTest(missing=field):
                bad = self.baseline()
                del bad[field]
                with self.assertRaises(TpuAgentSuiteError) as ctx:
                    study.set_baseline("incomplete", bad)
                self.assertIn("invalid baseline", str(ctx.exception))
        cases = {
            # Every arm is compared against it, so a modelled one turns the
            # study into an evaluation of its own prediction.
            "modelled": ({
                "provenance": "modeled"
            }, "must be measured"),
            "wrong_metric": ({
                "name": "ttft_p50_ms"
            }, "not a delta"),
            "under_replicated": ({
                "replicates": 1
            }, "every comparison rests on"),
        }
        for name, (metric, message) in cases.items():
            with self.subTest(name):
                with self.assertRaises(TpuAgentSuiteError) as ctx:
                    study.set_baseline("incomplete",
                                       self.baseline(metric=metric))
                self.assertIn(message, str(ctx.exception))

    def test_a_delta_without_a_baseline_is_refused(self):
        # delta_pct claimed to compare against "the study's baseline" while no
        # baseline existed anywhere -- a comparison with nothing on one side.
        self.start("nobase", admissibility_gates=[])
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.record_round("nobase",
                               {"arms": [self.arm(result={"gates": []})]})
        self.assertIn("no baseline", str(ctx.exception))


class TestBaselineReplacement(RoundRecordTestCase):

    def test_replacing_a_baseline_with_measured_arms_is_refused(self):
        # Every delta already recorded compares to the old number. Swapping it
        # silently leaves them comparing to something that is no longer there.
        self.start_based("swap", admissibility_gates=[])
        study.record_round("swap", {"arms": [self.arm(result={"gates": []})]})
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.set_baseline("swap", self.baseline(metric={"value": 999.0}))
        self.assertIn("no longer there", str(ctx.exception))
        # Setting the identical baseline again invalidates nothing.
        doc = study.set_baseline("swap", self.baseline())
        self.assertEqual(doc["baseline"]["metric"]["value"], 100.0)


class TestArmMetricMatchesProtocol(RoundRecordTestCase):

    def test_an_arm_reporting_another_metric_is_refused(self):
        # A delta between two different quantities is not a delta.
        self.start_based("mismatch", admissibility_gates=[])
        arm = self.arm(
            result={
                "gates": [],
                "metric":
                self.metric(name="ttft_p50_ms", value=1.0, delta_pct=2.0),
            })
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.record_round("mismatch", {"arms": [arm]})
        self.assertIn("primary metric", str(ctx.exception))


class TestStudyNotes(RoundRecordTestCase):
    """Rendered from the state, so it cannot go stale."""

    def test_notes_report_the_round_terms_and_recorded_arms(self):
        self.start_based("noted", admissibility_gates=[])
        study.attach_run_to_study("noted", "20260917T120000Z-0123456789ab")
        study.record_round("noted", {"arms": [self.arm(result={"gates": []})]})
        text = study.render_study_notes("noted")
        self.assertIn("# study noted", text)
        self.assertIn("round: 1 of 10", text)
        self.assertIn("tokens_per_s", text)
        self.assertIn("**a1**", text)
        self.assertIn("[measured]", text)


class TestSetProtocol(RoundRecordTestCase):

    def test_a_reason_is_required(self):
        # A diff shows what changed and never why, and the amendment worth
        # reviewing later is the one that relaxed a failing bar.
        self.start("silent")
        for blank in ("", "   "):
            with self.subTest(reason=blank):
                with self.assertRaises(TpuAgentSuiteError) as ctx:
                    study.set_protocol("silent", self.protocol(), blank)
                self.assertIn("--reason", str(ctx.exception))

    def test_an_invalid_amendment_is_refused(self):
        self.start("badamend")
        with self.assertRaises(TpuAgentSuiteError):
            study.set_protocol("badamend", {"mode": "auto"}, reason="x")
        self.assertEqual(study.load_study("badamend")["protocol_revision"], 1)


class TestRunToStudyCardinality(StudyStoreTestCase):
    """One run, one study. The rule the rest of the model rests on."""

    RUN = "20260917T120000Z-0123456789ab"

    def test_attaching_the_same_run_twice_is_a_no_op(self):
        # A retried command is not an error, and emits no second marker.
        self.start("idem")
        study.attach_run_to_study("idem", self.RUN)
        doc = study.attach_run_to_study("idem", self.RUN)
        self.assertEqual(doc["runs"], [self.RUN])
        # SessionStart cannot be routed, so the attach carries that signal.
        events = study.read_study_events("idem")
        self.assertEqual([(e["kind"], e["run_id"], e["round"])
                          for e in events], [("attach", self.RUN, 1)])

    def test_attaching_a_run_to_a_second_study_is_refused_without_change(self):
        self.start("first")
        self.start("second")
        study.attach_run_to_study("first", self.RUN)
        with self.assertRaises(TpuAgentSuiteError) as ctx:
            study.attach_run_to_study("second", self.RUN)
        message = str(ctx.exception)
        self.assertIn("already attached", message)
        self.assertIn("first", message)
        self.assertEqual(study.load_study("first")["runs"], [self.RUN])
        self.assertEqual(study.load_study("second")["runs"], [])


class TestEventRouting(StudyStoreTestCase):
    """Strict lookup: no inference, no fan-out."""

    RUN = "20260917T120000Z-0123456789ab"
    OTHER = "20260917T130000Z-abcdef012345"

    def test_a_lone_open_study_does_not_absorb_unrelated_sessions(self):
        # The common case, and the bug: one open study used to receive every
        # session on the machine, including onboarding jobs and scratch work.
        self.start("lonely")
        self.assertEqual(study.studies_for_event(self.OTHER), [])
        self.assertEqual(study.studies_for_event(""), [])

    def test_a_stop_event_reaches_only_its_own_study(self):
        # fully_idle in the wrong stream is a false liveness signal, which
        # inverts the one thing the stream exists to report. It distinguishes
        # "the model stopped" from "no work is outstanding", which is the
        # difference between healthy and stalled, so the termination fields
        # must be carried through.
        self.start("watched")
        self.start("bystander")
        study.attach_run_to_study("watched", self.RUN)
        actions._builtin_record_event(
            "p", {
                "event": "Stop",
                "context": {
                    "workspace": "/ws",
                    "package": "p",
                    "run_id": self.RUN
                },
                "termination": {
                    "reason": "NO_TOOL_CALL",
                    "fully_idle": True
                },
            })
        watched = _hook_events("watched")
        self.assertEqual(len(watched), 1)
        self.assertEqual(watched[0]["stop_reason"], "NO_TOOL_CALL")
        self.assertTrue(watched[0]["fully_idle"])
        self.assertEqual(_hook_events("bystander"), [])


class TestRecordEventBuiltin(StudyStoreTestCase):
    """The hook handler that writes the event stream."""

    RUN = "20260917T120000Z-0123456789ab"
    PAYLOAD = {
        "event": "PreToolUse",
        "tool": "delegate",
        "subagent": "worker",
        "context": {
            "workspace": "/ws",
            "package": "tpu-agent-suite",
            "session_id": "sess-1",
            "run_id": RUN,
        },
    }

    def test_closed_studies_do_not_receive_events(self):
        self.start("done")
        study.attach_run_to_study("done", self.RUN)
        study.close_study("done")
        actions._builtin_record_event("tpu-agent-suite", self.PAYLOAD)
        self.assertEqual(_hook_events("done"), [])

    def test_always_allows_when_the_store_is_unwritable(self):
        # A monitor must never be able to block the work it monitors. The
        # write is forced to fail rather than chmod-ed, which neither root nor
        # an already-existing events file would honour.
        self.start("unwritable")
        study.attach_run_to_study("unwritable", self.RUN)
        with mock.patch.object(study,
                               "append_study_event",
                               side_effect=OSError("read-only")) as append:
            out = actions._builtin_record_event("tpu-agent-suite",
                                                self.PAYLOAD)
        append.assert_called_once()
        self.assertEqual(out["decision"], "allow")


class TestStudyCli(StudyStoreTestCase):

    def _cli_start(self, study_id: str, protocol: dict) -> int:
        """`study start` with `protocol` written to a file for it to read."""
        path = self.home / f"{study_id}.protocol.json"
        path.write_text(json.dumps(protocol), encoding="utf-8")
        return runtime.main([
            "study", "start", "--study-id", study_id, "--protocol",
            str(path), "--package", "tpu-agent-suite", "--target", "m"
        ])

    def test_start_status_attach_list_close_round_trip(self):
        self.assertEqual(self._cli_start("cli-study", self.protocol()), 0)
        run = "20260917T120000Z-0123456789ab"
        for argv in (["status"], ["attach-run", "--run-id", run], ["events"],
                     ["journal", "--limit",
                      "5"], ["close", "--outcome", "tie"]):
            with self.subTest(argv=argv):
                self.assertEqual(
                    runtime.main([
                        "study", argv[0], "--study-id", "cli-study", *argv[1:]
                    ]), 0)
        self.assertEqual(runtime.main(["study", "list"]), 0)

        doc = study.load_study("cli-study")
        self.assertEqual(doc["status"], study.STUDY_STATUS_CLOSED)
        self.assertEqual(doc["outcome"], "tie")
        self.assertEqual(doc["protocol"]["primary_metric"]["name"],
                         "tokens_per_s")
        self.assertEqual(doc["runs"], [run])

    def test_cli_returns_nonzero_on_a_bad_study_id_or_protocol(self):
        self.assertEqual(self._cli_start("Bad_Id", self.protocol()), 1)
        self.assertEqual(self._cli_start("bad-protocol", {"target": "m"}), 1)
        # A rejected protocol leaves nothing behind to block a retry.
        self.assertFalse(paths.get_study_root("bad-protocol").exists())
        # A study id becomes a directory name: traversal is refused on every
        # path, including save_study, which takes its id from the document.
        with self.assertRaises(TpuAgentSuiteError):
            paths.get_study_root("../../etc")
        with self.assertRaises(TpuAgentSuiteError):
            study.save_study({"study_id": "../escape"})


if __name__ == "__main__":
    unittest.main()
