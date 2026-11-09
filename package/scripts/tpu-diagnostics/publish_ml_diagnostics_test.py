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
"""Unit tests for scripts/tpu-diagnostics/publish_ml_diagnostics.py."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import publish_ml_diagnostics as pub

CLUSTER = "projects/proj/locations/us-central1/clusters/c"
BASE = f"{pub.API}/projects/proj/locations/us-central1/machineLearningRuns"


class PublishTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        mock.patch.object(pub, "POLL_SECONDS", 0).start()
        self.addCleanup(mock.patch.stopall)
        self.runs: list[list[str]] = []
        self.calls: list[tuple[str, str, object]] = []
        # url -> reply; a POST reply may be a list consumed in order.
        self.replies: dict[str, object] = {BASE: (200, {"done": True})}

    def capture(self, name: str, payload: bytes) -> Path:
        d = self.root / name / "plugins" / "profile" / "2026_10_27_00_00_00"
        d.mkdir(parents=True)
        (d / "host.xplane.pb").write_bytes(payload)
        return self.root / name

    def publish(self, *traces: str) -> tuple[int, dict]:

        def run(argv, what, timeout=3600):
            self.runs.append(argv)
            return "token\n"

        def call(method, url, token, body=None):
            self.calls.append((method, url, body))
            reply = self.replies.get(url.split("?")[0], (200, {}))
            return reply.pop(0) if isinstance(reply, list) else reply

        out = io.StringIO()
        with mock.patch.object(pub, "_run", run), \
                mock.patch.object(pub, "_call", call), \
                contextlib.redirect_stdout(out):
            code = pub.main([
                "--cluster", CLUSTER, "--gcs", "gs://b/profiles/",
                "--run-group", "Run_42", *[f"--trace={t}" for t in traces]
            ])
        return code, json.loads(out.getvalue())

    def sessions(self, rid: str, link: str) -> None:
        self.replies[f"{BASE}/{rid}/profilerSessions"] = (200, {
            "profilerSessions": [{
                "dashboardUri": ""
            }, {
                "dashboardUri": link
            }]
        })

    def test_uploads_below_run_id_and_registers_the_prefix_above_it(self):
        src = self.capture("decode", b"\x00/device:TPU:0\x00")
        self.sessions("run-42-decode", "https://xprof/decode")
        code, res = self.publish(f"decode={src}")
        self.assertEqual(code, 0)
        self.assertEqual(res["published"], {"decode": "https://xprof/decode"})
        self.assertIn([
            "gcloud", "storage", "rsync", "-r",
            str(src / "plugins" / "profile"),
            "gs://b/profiles/run-42-decode/plugins/profile/"
        ], self.runs)
        method, url, body = self.calls[0]
        self.assertEqual(
            (method, url),
            ("POST", f"{BASE}?machine_learning_run_id=run-42-decode"))
        self.assertEqual(body["artifacts"], {"gcsPath": "gs://b/profiles"})
        self.assertEqual(body["runSet"], "run-42")
        self.assertEqual(body["tools"], [{"xprof": {}}])

    def test_host_only_capture_is_skipped_without_uploading(self):
        src = self.capture("prefill", b"/host:CPU")
        code, res = self.publish(f"prefill={src}")
        self.assertEqual(code, 0)
        self.assertIn("prefill", res["skipped"])
        self.assertEqual((self.runs, self.calls), ([], []))

    def test_republish_accepts_already_exists_from_the_operation(self):
        src = self.capture("decode", b"/device:TPU:0")
        self.replies[BASE] = (200, {"name": "operations/op1", "done": False})
        self.replies[f"{pub.API}/operations/op1"] = (200, {
            "done": True,
            "error": {
                "code": pub.ALREADY_EXISTS
            }
        })
        self.sessions("run-42-decode", "https://xprof/decode")
        code, res = self.publish(f"decode={src}")
        self.assertEqual((code, res["failed"]), (0, {}))

    def test_failed_register_is_reported_and_others_still_publish(self):
        bad = self.capture("bad", b"/device:TPU:0")
        good = self.capture("good", b"/device:TPU:0")
        self.replies[BASE] = [(200, {
            "done": True,
            "error": {
                "code": 3,
                "message": "bad gcsPath"
            }
        }), (200, {
            "done": True
        })]
        self.sessions("run-42-good", "https://xprof/good")
        code, res = self.publish(f"bad={bad}", f"good={good}")
        self.assertEqual(code, 1)
        self.assertIn("bad gcsPath", res["failed"]["bad"])
        self.assertEqual(res["published"], {"good": "https://xprof/good"})


if __name__ == "__main__":
    unittest.main()
