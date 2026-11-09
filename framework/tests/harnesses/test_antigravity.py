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
"""Antigravity adaptor."""

from __future__ import annotations

import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import harnesses  # noqa: E402
from framework.tests.harnesses import conformance  # noqa: E402


class AntigravityTest(conformance.HarnessConformance):
    harness = harnesses.require("antigravity")

    def test_stop_reply(self):
        """'retry' becomes `continue`, which re-enters the loop and keeps the
        conversation, so a resumed run picks up instead of repeating."""
        resumed = self._reply(
            {
                "decision": "retry",
                "reason": "waited for quota"
            }, "Stop", "resume-after-quota")
        self.assertEqual(resumed["decision"], "continue")
        self.assertIn("waited", resumed["reason"])
        ended = self._reply({"decision": "allow"}, "Stop",
                            "resume-after-quota")
        self.assertEqual(ended["decision"], "stop")

    def test_session_start_reply(self):
        allow = self._reply(
            {
                "decision": "allow",
                "context": "Active profile: TPU v6e."
            }, "SessionStart", "init-guard")
        self.assertEqual(allow["injectSteps"],
                         [{
                             "ephemeralMessage": "Active profile: TPU v6e."
                         }])
        deny = self._reply({
            "decision": "deny",
            "reason": "Invalid workspace"
        }, "SessionStart", "init-guard")
        self.assertEqual(len(deny["injectSteps"]), 1)
        self.assertIn("Invalid workspace",
                      deny["injectSteps"][0]["ephemeralMessage"])


if __name__ == "__main__":
    unittest.main()
