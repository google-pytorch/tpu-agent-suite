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
"""Registry sanity checks for the probe suite. Needs neither jax nor a TPU."""

import collections
import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402
import run_probes  # noqa: E402


def _probes():
    with contextlib.redirect_stdout(io.StringIO()):
        run_probes._import_all_check_modules()  # pylint: disable=protected-access
    return harness.registered_probes()


class RegistryTest(unittest.TestCase):

    def test_registry_is_not_empty(self):
        self.assertTrue(_probes())

    def test_probe_names_are_unique(self):
        counts = collections.Counter(p.name for p in _probes())
        duplicates = sorted(n for n, c in counts.items() if c > 1)
        self.assertFalse(duplicates, f"duplicate probe names: {duplicates}")

    def test_every_probe_has_a_question(self):
        missing = [p.name for p in _probes() if not p.question.strip()]
        self.assertFalse(missing, f"probes with an empty question: {missing}")

    def test_list_touches_no_device(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(run_probes.main(["--list"]), 0)
        self.assertIn("dispatch_floor", out.getvalue())


if __name__ == "__main__":
    unittest.main()
