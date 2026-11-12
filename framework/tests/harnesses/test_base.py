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
"""The registry and `mapping_block`."""

from __future__ import annotations

import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import harnesses, utils  # noqa: E402
from framework.harnesses import base, google  # noqa: E402
from framework.hooks import dispatch  # noqa: E402


class RegistryTest(unittest.TestCase):

    def test_unknown_harness(self):
        self.assertIsNone(harnesses.get("no-such-harness"))
        with self.assertRaises(utils.TpuAgentSuiteError):
            harnesses.require("no-such-harness")
        with self.assertRaises(utils.TpuAgentSuiteError):
            dispatch.translate_canonical_output({}, "no-such-harness",
                                                "PreToolUse", "hk")

    def test_registry_matches_the_mapping_tables(self):
        """A name in the tables or the registry alone is a half-added harness.

        The registry alone decides what is supported. Each table on its own: a
        block missing from one makes every lookup in it empty.
        """
        for loader in (utils.get_tool_mappings, utils.get_event_mappings):
            table_names: set[str] = set()
            for entry in loader().values():
                table_names.update(entry.get("harnesses", {}))
            self.assertEqual(table_names, set(harnesses.HARNESSES),
                             loader.__name__)
        self.assertEqual(harnesses.supported_harnesses(), harnesses.names())


class MappingBlockTest(unittest.TestCase):
    """`mapping_block` reads the harness's own block, else its family's."""

    def test_family_member_reads_its_own_block_before_the_familys(self):

        class Member(google.GoogleHarness):
            name = "member"
            display_name = "Member"
            brain_dir = "member-cli"

        # The family key is inherited, not replaced by the member's name.
        self.assertEqual(Member.mapping_key, google.GOOGLE_HARNESS_MAPPING_KEY)
        own = {"tools": ["own"]}
        family = {"tools": ["fam"]}
        self.assertEqual(
            Member().mapping_block({
                "member": own,
                "antigravity": family
            }), own)
        self.assertEqual(Member().mapping_block({"antigravity": family}),
                         family)
        self.assertEqual(Member().mapping_block({}), {})

    def test_a_harness_outside_a_family_reads_only_its_own_block(self):

        class Lone(base.Harness):
            name = "lone"
            display_name = "Lone"
            config_dir = ".lone"

        self.assertEqual(Lone().mapping_block({"antigravity": {"x": 1}}), {})
        # An unregistered name behaves the same through the module function.
        blocks = {"other": {"tools": ["o"]}, "antigravity": {"tools": ["a"]}}
        self.assertEqual(harnesses.harness_mapping(blocks, "other"),
                         {"tools": ["o"]})
        self.assertEqual(harnesses.harness_mapping(blocks, "unknown"), {})


if __name__ == "__main__":
    unittest.main()
