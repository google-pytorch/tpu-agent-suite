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
"""Codex adaptor."""

from __future__ import annotations

import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import harnesses  # noqa: E402
from framework.tests.harnesses import conformance  # noqa: E402


class CodexTest(conformance.HarnessConformance):
    harness = harnesses.require("codex")

    def test_stop_reply(self):
        retry = self._reply({
            "decision": "retry",
            "reason": "waited for quota"
        }, "Stop", "resume-after-quota")
        self.assertEqual(retry["decision"], "block")
        self.assertIn("waited", retry["reason"])
        allow = self._reply({"decision": "allow"}, "Stop",
                            "resume-after-quota")
        self.assertEqual(allow["decision"], "allow")

    def test_session_start_reply(self):
        allow = self._reply(
            {
                "decision": "allow",
                "context": "Context for codex"
            }, "SessionStart", "init-guard")
        self.assertEqual(allow.get("decision"), "allow")
        self.assertEqual(allow.get("additional_context"), "Context for codex")
        deny = self._reply(
            {
                "decision": "deny",
                "reason": "GCE cluster unavailable"
            }, "SessionStart", "init-guard")
        self.assertEqual(deny.get("decision"), "block")
        self.assertEqual(deny.get("reason"), "GCE cluster unavailable")


if __name__ == "__main__":
    unittest.main()
