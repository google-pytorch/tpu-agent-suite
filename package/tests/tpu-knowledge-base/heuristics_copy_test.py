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
"""The optimization-heuristics references and their tpu-knowledge-base copies say the same.

tpu-diagnostics reads its references; every other agent reaches the same
evidence through the knowledge base. A number changed in one place only
would leave the two disagreeing.
"""

import os
import unittest

# package/tests/tpu-knowledge-base/ -> package/; run by `make
# test-knowledge-base`, which CI runs.
_PACKAGE = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PAGES = ("serving-trace-attribution", )


def _read(*parts):
    with open(os.path.join(_PACKAGE, *parts), encoding="utf-8") as f:
        return f.read()


class KbCopyTest(unittest.TestCase):

    def test_reference_matches_kb_page_body(self):
        for page in _PAGES:
            with self.subTest(page=page):
                kb = _read("data", "tpu-knowledge-base", "pages", f"{page}.md")
                body = kb[kb.index("\n# ") + 1:]
                self.assertEqual(
                    _read("references", "optimization-heuristics",
                          f"{page}.md"), body,
                    f"edit references/optimization-heuristics/{page}.md and the KB page"
                    " together")

    def test_kb_detector_table_matches_the_reference(self):
        """The KB's serving-heuristics page lists the same detectors."""

        def table(text):
            sec = text[text.index("## All detectors"):]
            sec = sec[:sec.index("\n## ", 4)]
            return [ln for ln in sec.splitlines() if ln.startswith("| `D")]

        ref = _read("references", "optimization-heuristics", "detectors.md")
        kb = _read("data", "tpu-knowledge-base", "pages",
                   "vllm-serving-heuristics.md")
        self.assertEqual(
            table(ref), table(kb),
            "edit the detector table in references/optimization-heuristics/"
            "detectors.md and the KB page vllm-serving-heuristics together")


if __name__ == "__main__":
    unittest.main()
