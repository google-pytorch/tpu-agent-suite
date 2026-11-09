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
"""Tests for the session journal, its `journal` command and its contract rule."""

from __future__ import annotations

import contextlib
import io
import json
import multiprocessing
import os
import tempfile
import unittest
from unittest import mock

from framework import journal, paths, runtime

RUN_ID = "20261014T170000Z-0123456789ab"


def _add_notes(agent: str) -> None:
    for i in range(25):
        journal.write(RUN_ID, agent=agent, kind="note", text=str(i))


class JournalTest(unittest.TestCase):

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = mock.patch.dict(os.environ, {"TPU_AGENT_SUITE_HOME": tmp.name})
        env.start()
        self.addCleanup(env.stop)
        self.dir = paths.get_run_root(RUN_ID)
        # The session creates the run directory, not the journal.
        self.dir.mkdir(parents=True)

    def add(self, **kw):
        return journal.write(RUN_ID, agent="a", **kw)

    def cli(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(
                io.StringIO()):
            return runtime.main(["journal", *argv]), out.getvalue()

    def test_entries_link_and_render(self):
        self.add(kind="hypothesis", text="fp8 KV lifts tok/s")
        journal.write(RUN_ID, agent=" a|b \n", kind="hypothesis", text="two")
        self.add(kind="rejected", parent="H2", text="refuted before")
        self.add(kind="result",
                 parent="H1",
                 verdict="supported",
                 text="+15%",
                 evidence=["summary.json"],
                 metrics={"gain_pct": 15.0})
        self.assertEqual(journal.read(RUN_ID)[1]["agent"], "a|b")
        md = (self.dir / journal.RENDERED_FILE).read_text()
        self.assertIn("| H1 | a | supported | fp8 KV lifts tok/s | +15% |", md)
        self.assertIn("| H2 | a\\|b | rejected |", md)
        self.assertIn("gain_pct = 15.0", md)

    def test_refused_writes_say_what_is_missing(self):
        other = "20261014T170000Z-ffffffffffff"
        with self.assertRaisesRegex(journal.Refused, "has no directory"):
            journal.write(other, agent="a", kind="note", text="x")
        self.assertFalse(paths.get_run_root(other).exists())

        self.add(kind="decision", text="route")
        self.add(kind="hypothesis", text="h")
        for fields, message in [
            ({}, "--parent <id> is not a hypothesis"),
            ({
                "parent": "D1"
            }, "--parent D1 is not a hypothesis"),
            ({
                "parent": "H1",
                "evidence": ["e"]
            }, "--verdict"),
            ({
                "parent": "H1",
                "verdict": "refuted"
            }, "--evidence"),
        ]:
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(journal.Refused, message):
                    self.add(kind="result", text="x", **fields)
        self.assertEqual(len(journal.read(RUN_ID)), 2)

    def test_concurrent_writers_lose_nothing(self):
        procs = [
            multiprocessing.Process(target=_add_notes, args=(f"w{i}", ))
            for i in range(4)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        self.assertEqual(len({e["id"] for e in journal.read(RUN_ID)}), 100)

    def test_cli(self):
        run = ("--run-id", RUN_ID)
        code, added = self.cli("add", *run, "--agent", "a", "--kind",
                               "hypothesis", "--text", "h", "--metric",
                               "k=v=1", "--metric", " n = 2", "--metric",
                               "loss=nan")
        self.assertEqual(code, 0)
        self.assertEqual(self.cli("show", *run, "--open")[1], added)
        self.assertEqual(self.cli("show", *run, "--agent", " a ")[1], added)
        self.assertEqual(
            json.loads(added)["metrics"], {
                "k": "v=1",
                "n": 2.0,
                "loss": "nan"
            })
        for bad in ("k=", "=1"):
            with self.subTest(metric=bad), self.assertRaises(SystemExit):
                self.cli("add", *run, "--agent", "a", "--kind", "note",
                         "--text", "x", "--metric", bad)
        self.assertEqual(self.cli("render", *run)[0], 0)
        self.assertEqual(self.cli("show", "--run-id", "../x")[0],
                         1)  # Can't leave runs/.


if __name__ == "__main__":
    unittest.main()
