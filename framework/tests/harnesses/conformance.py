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
"""Checks every adaptor must pass; each `test_<harness>.py` subclasses this."""

from __future__ import annotations

import contextlib
import io
import pathlib
import sys
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import harnesses, utils  # noqa: E402
from framework.hooks import dispatch  # noqa: E402


class HarnessConformance(unittest.TestCase):
    """Shared checks. Skipped when collected as the base class itself."""

    harness: harnesses.Harness

    @classmethod
    def setUpClass(cls) -> None:
        if cls is HarnessConformance:
            raise unittest.SkipTest("abstract conformance suite")
        super().setUpClass()

    def _reply(self, result: dict, event: str, hook: str = "guard") -> dict:
        """`result` translated into this harness's reply, as dispatch does."""
        return dispatch.translate_canonical_output(result, self.harness.name,
                                                   event, hook)

    def _hook_reply(self, event: str, decision: str) -> harnesses.HookReply:
        return harnesses.HookReply(event=event,
                                   hook_name="hk",
                                   decision=decision,
                                   reason="why",
                                   followup_message=None,
                                   message="banner",
                                   context="ctx",
                                   updated_args={"a": 1})

    def test_renders_a_reply_for_every_canonical_event(self):
        """Every canonical event renders to a dict for every verdict.

        The unknown-event path is made to fail for the duration, so a
        canonical event that `render_reply` does not dispatch -- which would
        otherwise land there and still return a dict -- fails here, as does
        a `reply_*` left on the base.
        """

        def not_canonical(reply: harnesses.HookReply) -> dict:
            self.fail(f"{reply.event} fell through to reply_unknown_event")

        with mock.patch.object(self.harness, "reply_unknown_event",
                               not_canonical):
            for event in utils.SUPPORTED_HOOK_EVENTS:
                for decision in ("allow", "deny", "retry"):
                    with self.subTest(event=event, decision=decision), \
                            contextlib.redirect_stderr(io.StringIO()):
                        self.assertIsInstance(
                            self.harness.render_reply(
                                self._hook_reply(event, decision)), dict)

    def test_renders_a_reply_for_an_unknown_event(self):
        """An event outside the canonical set takes the unknown-event path."""
        for decision in ("allow", "deny", "retry"):
            with self.subTest(decision=decision), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    mock.patch.object(self.harness, "reply_unknown_event",
                                      wraps=self.harness.reply_unknown_event
                                      ) as unknown:
                self.assertIsInstance(
                    self.harness.render_reply(
                        self._hook_reply("NoSuchEvent", decision)), dict)
                unknown.assert_called_once()
