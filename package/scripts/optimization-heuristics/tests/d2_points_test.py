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
"""D2's token-bucket ladder and propose's one-candidate-per-kind selection."""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

# pylint: disable=g-import-not-at-top,wrong-import-position
import propose  # noqa: E402
from serving_analysis import context as context_lib  # noqa: E402
from serving_analysis import detectors  # noqa: E402


class TokenPaddingsTest(unittest.TestCase):

    def test_default_ladder_doubles_from_16_to_the_budget(self):
        ctx = context_lib.ServingContext(max_num_batched_tokens=512)
        self.assertEqual(detectors._torchtpu_token_paddings(ctx),
                         [16, 32, 64, 128, 256, 512])

    def test_server_log_ladder_wins(self):
        ctx = context_lib.ServingContext(
            server_log='INFO Using token paddings: [16, 32, 48, 64]\n')
        self.assertEqual(detectors._torchtpu_token_paddings(ctx),
                         [16, 32, 48, 64])


class SelectKindsTest(unittest.TestCase):

    def _ranked(self, kinds):
        return [(0, -i, {
            "cand_id": f"{k}{i}",
            "kind": k
        }) for i, k in enumerate(kinds)]

    def test_a_knob_takes_the_last_mesh_slot(self):
        got = propose.select_kinds(
            self._ranked(["mesh", "mesh", "mesh", "knob"]), 3)
        self.assertEqual([r[2]["cand_id"] for r in got[:3]],
                         ["mesh0", "mesh1", "knob3"])

    def test_one_kind_only_is_unchanged(self):
        ranked = self._ranked(["mesh", "mesh", "mesh", "mesh"])
        self.assertEqual(propose.select_kinds(ranked, 3), ranked)

    def test_both_kinds_already_in_is_unchanged(self):
        ranked = self._ranked(["mesh", "knob", "mesh", "knob"])
        self.assertEqual(propose.select_kinds(ranked, 3), ranked)


if __name__ == "__main__":
    unittest.main()
