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
"""Tests for the study store: the layer above a run.

Every test points TPU_AGENT_SUITE_HOME at a temp dir, so nothing here touches a real
~/.tpu-agent-suite.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from framework import jsonio, paths, runtime, study, utils
from framework.hooks import actions
from framework.utils import TpuAgentSuiteError

Path = pathlib.Path


class StudyStoreTestCase(unittest.TestCase):
    """Base that isolates the central store into a temp directory."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        env = mock.patch.dict(os.environ, {"TPU_AGENT_SUITE_HOME": tmp.name})
        env.start()
        self.addCleanup(env.stop)

    @staticmethod
    def protocol(**overrides) -> dict:
        """A minimal protocol that satisfies the schema.

        Built here rather than pasted into forty tests so that adding a
        required field breaks one place, not forty, and so no test can
        accidentally assert the behaviour of a protocol that would be
        rejected in practice.
        """
        base = dict(primary_metric=dict(name="tokens_per_s",
                                        unit="tok/s",
                                        direction="maximize"),
                    measurement_cell=dict(topology="v6e-2x2x1"),
                    replicate_count=3,
                    noise_floor_pct=0.4,
                    stopping_rule=dict(max_rounds=10),
                    admissibility_gates=[],
                    mode="approve")
        base.update(overrides)
        return base

    def start(self, study_id: str, **overrides):
        """Open a study with a valid protocol and identity."""
        return study.start_study(study_id,
                                 protocol=self.protocol(**overrides),
                                 package="tpu-agent-suite",
                                 target="test-model",
                                 owner="tester")

    def start_with_arms(self, study_id: str, arms: list[dict]) -> None:
        """Open a study and write `arms` straight into its state."""
        self.start(study_id)
        doc = study.load_study(study_id)
        doc["arms"] = arms
        study.save_study(doc)


def _open_arm(arm_id: str, claim: str, **extra) -> dict:
    """An untested arm, as a round or a claim test seeds it."""
    return {
        "arm": arm_id,
        "kind": "kernel-block",
        "status": "open",
        "claim": claim,
        **extra
    }


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


class RoundRecordTestCase(StudyStoreTestCase):
    """Shared fixtures for the round-record verbs."""

    GATES = [
        dict(name="ttft",
             metric="ttft_p50_ms",
             comparator="lt",
             bound=5000,
             same_execution=True),
        dict(name="accuracy",
             metric="gsm8k",
             comparator="gte",
             bound=0.94,
             same_execution=True),
    ]

    @staticmethod
    def baseline(**overrides) -> dict:
        base = dict(execution_id="j-base",
                    metric=dict(name="tokens_per_s",
                                value=100.0,
                                unit="tok/s",
                                replicates=3,
                                provenance="measured"),
                    profile="baseline/xplane.pb",
                    fingerprint=dict(commit_sha="abc123",
                                     config_digest="sha256:deadbeef",
                                     cell="v6e-2x2x1/8192-1024/c256"))
        metric_overrides = overrides.pop("metric", None)
        base.update(overrides)
        if metric_overrides is not None:
            base["metric"] = {**base["metric"], **metric_overrides}
        return base

    def start_based(self, study_id: str, **overrides):
        """Open a study that already has a baseline, so arms may report a
        delta. Most round tests need one; a delta without a baseline is a
        comparison with nothing on the other side."""
        doc = self.start(study_id, **overrides)
        # The baseline must itself meet the pre-registered replicate count,
        # so derive it rather than hardcoding one that a stricter protocol
        # would reject.
        replicates = doc["protocol"]["replicate_count"]
        study.set_baseline(study_id,
                           self.baseline(metric={"replicates": replicates}))
        return doc

    @staticmethod
    def metric(**overrides) -> dict:
        """An arm's primary-metric report; a 5% measured delta by default."""
        return dict(name="tokens_per_s",
                    value=100.0,
                    replicates=3,
                    delta_pct=5.0,
                    provenance="measured") | overrides

    @staticmethod
    def gate(name: str, execution_id: str = "j-1", **extra) -> dict:
        """A passing gate report measured on `execution_id`."""
        return dict(name=name, passed=True, execution_id=execution_id) | extra

    @classmethod
    def arm(cls, **overrides) -> dict:
        result = dict(verdict="tie",
                      execution_id="j-1",
                      metric=cls.metric(),
                      gates=[
                          cls.gate("ttft", value=4500.0),
                          cls.gate("accuracy", value=0.95)
                      ])
        base = _open_arm("a1", "fuse the epilogue", status="decided")
        base["result"] = result
        result_overrides = overrides.pop("result", None)
        base.update(overrides)
        if result_overrides is not None:
            base["result"] = {**base["result"], **result_overrides}
        return base


class TestRecordRound(RoundRecordTestCase):

    def test_a_decided_arm_needs_a_result(self):
        self.start("resultless")
        arm = _open_arm("a1", "x", status="decided")
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

            arm_entry = _open_arm(f"a{idx}", f"claim-{idx}", status=status)
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
            _open_arm("a0", "c0", share=0.1, headroom=0.2),
            _open_arm("a1", "c1", share=0.5, headroom=0.8),
            _open_arm("a2", "c2", share=0.3, headroom=0.5),
        ])

        self.assertEqual(
            [study.claim_arm("claim-order")["arm"] for _ in range(3)],
            ["a1", "a2", "a0"])

        with self.assertRaises(study.TpuAgentSuiteError) as ctx:
            study.claim_arm("claim-order")
        self.assertIn("no open arms available", str(ctx.exception))

    def test_claim_arm_refuses_non_open_arm(self):
        self.start_with_arms("claim-refuse", [
            _open_arm("a0", "c0", status="decided"),
            _open_arm("a1", "c1", status="in_flight"),
        ])

        for arm_id, message in (("a0", "cannot be claimed"),
                                ("a1", "cannot be claimed"), ("nonexistent",
                                                              "not found")):
            with self.subTest(arm=arm_id):
                with self.assertRaisesRegex(study.TpuAgentSuiteError, message):
                    study.claim_arm("claim-refuse", arm_id=arm_id)

    def test_release_arm_transitions_in_flight_to_open_or_blocked(self):
        self.start_with_arms("release-test", [_open_arm("a0", "c0")])

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
        self.start_with_arms("failed-dispatch", [_open_arm("a0", "c0")])

        study.claim_arm("failed-dispatch", arm_id="a0")
        first = study.release_arm("failed-dispatch",
                                  arm_id="a0",
                                  failed=True,
                                  summary="recompile per step")
        self.assertEqual(first["status"], "open")
        self.assertEqual(first["failed_dispatches"], 1)
        self.assertNotIn("result", first)

        # A round that rewrites the arms without the count does not reset it.
        study.record_round("failed-dispatch",
                           {"arms": [_open_arm("a0", "c0")]})
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
        self.start_with_arms("failed-dispatch-edges", [_open_arm("a0", "c0")])
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
            _open_arm(f"mp-arm-{i}",
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
        self.start_with_arms("cli-claim", [_open_arm("a0", "c0")])

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
            sid, {"arms": [_open_arm(f"a{i}", f"c{i}") for i in (1, 2, 3)]})
        # Both workers snapshot the study before w2 claims a2.
        stale = {a["arm"]: a for a in study.load_study(sid)["arms"]}
        study.claim_arm(sid, arm_id="a2", worker_id="w2")

        def decided(arm_id, verdict, **metric):
            return dict(_open_arm(arm_id, "c", status="decided"),
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
        self.start_with_arms(sid,
                             [_open_arm("a0", "c0"),
                              _open_arm("a1", "c")])

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
            _open_arm("orphaned-arm",
                      "c0",
                      status="in_flight",
                      claimed_by="crashed-worker",
                      claimed_at="2020-01-01T00:00:00-08:00"),
        ])

        # Claiming with claim_ttl_seconds=60.0 automatically recovers the expired in_flight lease
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
