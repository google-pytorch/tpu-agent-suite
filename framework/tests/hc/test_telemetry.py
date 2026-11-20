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
"""Unit tests for `framework.hc.telemetry`."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from framework import jsonio
from framework.hc import constants, paths, telemetry


class TestTelemetryLogger(unittest.TestCase):
    """Verify flat JSONL format, post-commit buffering, rejections, and call-fallback env var."""

    def test_buffering_rejection_reserved_keys_and_fallback(self):
        run_id = "20261110T070000Z-abcdef012345"
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(
                os.environ, {"TPU_AGENT_SUITE_HOME": home}):
            buf = telemetry.TelemetryBuffer(run_id=run_id,
                                            campaign_id=3,
                                            round_id=2)
            with self.assertRaises(ValueError):
                buf.record("bad.event", timestamp="collision")
            with self.assertRaises(ValueError):
                telemetry.append_event(campaign_id=3,
                                       run_id=run_id,
                                       round_id=None,
                                       event_name="bad.event",
                                       event_fields={"run_id": "override"})

            buf.record("decision.recorded",
                       decision_id=11,
                       verdict=constants.DecisionVerdict.WIN)
            log_path = paths.get_campaign_events_log(3)
            self.assertFalse(log_path.exists())
            self.assertEqual(buf.flush(), 1)
            row = jsonio.read_jsonl(log_path)[0]
            self.assertEqual(
                (row["run_id"], row["round_id"], row["event_name"],
                 row["verdict"], row["decision_id"]),
                (run_id, 2, "decision.recorded", "Win", 11),
            )
            self.assertNotIn("event_fields", row)

            rej = telemetry.TelemetryBuffer(run_id=run_id,
                                            campaign_id=5,
                                            round_id=1)
            rej.record("round.opened")
            rej.record_rejection(args=["round", "open"],
                                 error_code="INVALID_STATE",
                                 message="already Open")
            rej_rows = jsonio.read_jsonl(paths.get_campaign_events_log(5))
            self.assertEqual([(r["event_name"], r["error_code"])
                              for r in rej_rows],
                             [("cmd.rejected", "INVALID_STATE")])

            call = telemetry.TelemetryBuffer(run_id=run_id,
                                             campaign_id=9,
                                             round_id=None)
            kw = dict(subcommands=["hypothesis", "search"],
                      args=["hypothesis", "search", "--state", "Open"])
            self.assertFalse(call.maybe_record_call_fallback(**kw))
            with mock.patch.dict(os.environ,
                                 {constants.LOG_ALL_CALLS_ENV: "1"}):
                self.assertEqual([
                    call.maybe_record_call_fallback(**kw),
                    call.maybe_record_call_fallback(**kw),
                ], [True, False])
            self.assertEqual([
                r["event_name"]
                for r in jsonio.read_jsonl(paths.get_campaign_events_log(9))
            ], ["hypothesis.search"])


if __name__ == "__main__":
    unittest.main()
