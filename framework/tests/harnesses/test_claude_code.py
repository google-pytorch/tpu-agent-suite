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
"""Claude Code adaptor."""

from __future__ import annotations

import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import harnesses  # noqa: E402
from framework.tests.harnesses import conformance  # noqa: E402


class ClaudeCodeTest(conformance.HarnessConformance):
    harness = harnesses.require("claude-code")

    def test_stop_reply(self):
        """'retry' becomes `block`, the harness's resume signal."""
        retry = self._reply({
            "decision": "retry",
            "reason": "waited for quota"
        }, "Stop", "resume-after-quota")
        self.assertEqual(retry["decision"], "block")
        self.assertIn("waited", retry["reason"])
        self.assertEqual(retry["hookSpecificOutput"]["hookEventName"], "Stop")

        allow = self._reply({"decision": "allow"}, "Stop",
                            "resume-after-quota")
        self.assertNotIn("decision", allow)
        self.assertEqual(allow["hookSpecificOutput"]["hookEventName"], "Stop")

    def test_tool_replies(self):
        out = self._reply(
            {
                "decision": "allow",
                "updated_args": {
                    "prompt": "Sanitized prompt"
                }
            }, "PreToolUse")
        self.assertEqual(out["hookSpecificOutput"]["hookEventName"],
                         "PreToolUse")
        self.assertEqual(out["hookSpecificOutput"]["permissionDecision"],
                         "allow")
        self.assertEqual(out["hookSpecificOutput"]["updatedInput"],
                         {"prompt": "Sanitized prompt"})

        out = self._reply(
            {
                "decision": "deny",
                "reason": "Kernel check failed",
                "followup_message": "not consulted",  # unlike the stops
                "context": "Please retry with tile size 64"
            },
            "PostToolUse",
            "verifier")
        self.assertEqual(out["decision"], "block")
        self.assertEqual(out["reason"], "Kernel check failed")
        self.assertEqual(out["hookSpecificOutput"]["hookEventName"],
                         "PostToolUse")
        self.assertEqual(out["hookSpecificOutput"]["additionalContext"],
                         "Please retry with tile size 64")
        # Unlike the stops, an empty reason is omitted, not filled in.
        out = self._reply({
            "decision": "deny",
            "followup_message": "f"
        }, "PostToolUse")
        self.assertEqual(out["decision"], "block")
        self.assertNotIn("reason", out)


if __name__ == "__main__":
    unittest.main()
