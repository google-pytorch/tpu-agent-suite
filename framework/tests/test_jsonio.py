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
"""Unit tests for JSON/JSONL reading and atomic writes."""

import pathlib
import tempfile
import unittest

from framework import jsonio

Path = pathlib.Path


class TestJsonio(unittest.TestCase):
    """Unit tests for jsonio functions."""

    def test_tolerant_readers_skip_unusable_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.json"
            self.assertEqual(jsonio.load_json_or_empty(path), {})
            # An atomic rewrite keeps a private file private.
            jsonio.write_json(path, {"a": 1})
            path.chmod(0o600)
            jsonio.write_json(path, {"a": 2})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(jsonio.load_json_or_empty(path), {"a": 2})
            for text in ("{bad json", "[1, 2]"):
                path.write_text(text, encoding="utf-8")
                self.assertEqual(jsonio.load_json_or_empty(path), {})

            path = Path(tmp) / "log.jsonl"
            path.write_bytes(
                b'{"a": 1}\n[1]\n{torn\n{"b": "\xff"}\n{"c": 3}\n')
            self.assertEqual(jsonio.read_jsonl(path, errors="ignore"), [{
                "a": 1
            }, {
                "b": ""
            }, {
                "c": 3
            }])
            self.assertEqual(jsonio.read_jsonl(path, 1, errors="ignore"),
                             [{
                                 "c": 3
                             }])


if __name__ == "__main__":
    unittest.main()
