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
"""Tests for tpu-diagnostics's locate_xprof_skill.py, which finds the XProf skill.

The skill ships inside the `xprof` wheel, co-versioned with the CLI, and
is read where the wheel put it. These pin what the script promises: it reports
that location and writes nothing, and it refuses -- naming the fix -- every
environment whose skill would not describe a usable binary.
"""

from __future__ import annotations

import builtins
import contextlib
import importlib.util
import io
import json
import pathlib
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _PACKAGE_ROOT / "scripts" / "tpu-diagnostics" / "locate_xprof_skill.py"

_spec = importlib.util.spec_from_file_location("locate_xprof_skill", _SCRIPT)
locate_xprof_skill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(locate_xprof_skill)

_REPO_ROOT = _PACKAGE_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework.builder import translate  # noqa: E402


def _without_packaging():
    """Make `import packaging...` fail, as in a venv holding only XProf."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "packaging" or name.startswith("packaging."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    return mock.patch("builtins.__import__", side_effect=fake_import)


class TestFloor(unittest.TestCase):

    def test_the_floor_is_the_one_in_requirements_client(self):
        """Two copies of the floor agree today and drift on the next bump."""
        text = (_PACKAGE_ROOT /
                "requirements-client.txt").read_text(encoding="utf-8")
        match = re.search(r"^xprof\s*>=\s*(\S+)\s*$", text, re.M)
        self.assertIsNotNone(match,
                             "requirements-client.txt pins no xprof floor")
        self.assertEqual(locate_xprof_skill.MINIMUM_VERSION, match[1])

    def test_every_copy_of_the_floor_follows_requirements_client(self):
        """requirements-client.txt is the one place the floor is set.

        The other copies cannot read it -- the TPU image installs its own
        requirements file, a suite pins its own sandbox -- so each is held to
        it here, and a bump that misses one fails rather than drifting.
        """
        floor = re.search(
            r"^xprof\s*>=\s*(\S+)\s*$",
            (_PACKAGE_ROOT /
             "requirements-client.txt").read_text(encoding="utf-8"), re.M)[1]
        suite = _REPO_ROOT / "eval" / "suites" / "xprof" / "cli"
        copies = {
            "requirements-tpu.txt": (_PACKAGE_ROOT / "requirements-tpu.txt",
                                     r"^xprof\s*>=\s*(\S+)\s*$"),
            "xprof CLI suite sandbox":
            (suite / "tier_a_evals.yaml", r'"xprof>=([^"]+)"'),
        }
        for name, (path, pattern) in copies.items():
            with self.subTest(copy=name):
                found = re.findall(pattern, path.read_text(encoding="utf-8"),
                                   re.M)
                self.assertTrue(found, f"no floor found in {path}")
                self.assertEqual(set(found), {floor}, path)

    def test_xprof_attention_takes_xprof_from_nexus_host(self):
        """The image and its verifier hold no floor of their own: nexus-host
        installs requirements-client.txt and pins it in its constraints."""
        task = (_REPO_ROOT / "eval" / "datasets" / "xprof-attention" /
                "kernel-tuning")
        dockerfile = (task / "environment" /
                      "Dockerfile").read_text(encoding="utf-8")
        verifier = (task / "tests" / "test.sh").read_text(encoding="utf-8")
        for name, text in (("Dockerfile", dockerfile), ("test.sh", verifier)):
            with self.subTest(file=name):
                self.assertNotRegex(text, r"""["']xprof(-nightly)?\s*[<>=~]""")
        self.assertIn("-c /etc/nexus-host/constraints.txt", dockerfile)
        self.assertIn("'^xprof==' /etc/nexus-host/constraints.txt", verifier)

    CASES = (
        ("2.23.2", True),
        ("2.23.3", True),
        ("2.23.3.post1", True),
        ("2.24.0a20260923", True),
        ("2.24.0", False),
        ("2.24.0.post1", False),
        ("2.24.1", False),
        ("2.25.0", False),
    )

    def test_below_floor_with_packaging(self):
        for version, expected in self.CASES:
            with self.subTest(version=version):
                self.assertIs(locate_xprof_skill.below_floor(version),
                              expected)

    def test_below_floor_without_packaging(self):
        """Re-cut `.postN` releases compare too, not just plain ones."""
        with _without_packaging():
            for version, expected in self.CASES:
                with self.subTest(version=version):
                    self.assertIs(locate_xprof_skill.below_floor(version),
                                  expected)
            self.assertIsNone(locate_xprof_skill.below_floor("2.24.0.dev1"))


class TestLocate(unittest.TestCase):
    """The report: where the wheel's skill is, or why there is none to use."""

    def setUp(self):
        self.root = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        # A stand-in for the wheel's package directory.
        self.skills = self.root / "site-packages" / "xprof" / "skills"
        self.skill = self.skills / "xprof"
        (self.skill / "references").mkdir(parents=True)
        (self.skill / "SKILL.md").write_text(
            "skill: see [A](references/a.md) and [B](references/b.md)",
            encoding="utf-8")
        for name in ("a.md", "b.md"):
            (self.skill / "references" / name).write_text(name,
                                                          encoding="utf-8")
        files = mock.patch("importlib.resources.files",
                           return_value=self.skills)
        files.start()
        self.addCleanup(files.stop)

    def _locate(self, dists):
        with mock.patch.object(locate_xprof_skill,
                               "installed_distributions",
                               return_value=dists):
            return locate_xprof_skill.locate()

    def test_reports_the_skill_where_the_wheel_put_it(self):
        report = self._locate({"xprof": "2.24.0"})
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["skill_md"], str(self.skill / "SKILL.md"))
        self.assertEqual(report["skill_dir"], str(self.skill))
        self.assertEqual(report["references"], ["a", "b"])
        self.assertEqual(report["version"], "2.24.0")
        self.assertEqual(report["interpreter"], sys.executable)

    def _run(self, *argv, dists=None):
        with mock.patch.object(locate_xprof_skill,
                               "installed_distributions",
                               return_value=dists if dists is not None else
                               {"xprof": "2.24.0"}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            code = locate_xprof_skill.main(list(argv))
        return code, out.getvalue()

    def test_prints_the_skill_itself(self):
        """The skill arrives in the script's output: no harness has to read
        outside the workspace, where most ask the user first."""
        code, out = self._run()
        self.assertEqual(code, 0)
        self.assertIn("xprof==2.24.0", out)
        self.assertIn("===== SKILL.md =====\nskill: see [A]", out)
        self.assertIn("References: a, b", out)
        self.assertIn("--reference <name>", out)

    def test_prints_a_reference_named_as_skill_md_links_it(self):
        for name in ("references/a.md", "a.md", "a", "./references/a.md"):
            with self.subTest(name=name):
                code, out = self._run("--reference", name)
                self.assertEqual(code, 0)
                self.assertTrue(
                    out.endswith("===== references/a.md =====\na.md\n"))

    def test_every_link_in_skill_md_is_printable(self):
        """The agent follows SKILL.md's links; each must resolve."""
        _, out = self._run()
        links = re.findall(r"\]\((references/[^)#]+\.md)\)", out)
        self.assertEqual(links, ["references/a.md", "references/b.md"])
        for link in links:
            with self.subTest(link=link):
                self.assertEqual(self._run("--reference", link)[0], 0)

    def test_an_unknown_reference_lists_the_known_ones(self):
        """Exit 2, not 1: a wrong name is not an unusable XProf. Nothing
        outside references/ is reachable through it."""
        for name in ("nope", "../SKILL", "../../xprof/__init__"):
            with self.subTest(name=name):
                code, out = self._run("--reference", name)
                self.assertEqual(code, 2)
                self.assertIn("References: a, b", out)

    def test_writes_nothing(self):
        """No copy anywhere: the wheel's skill is the only one."""
        before = {p: p.stat().st_mtime_ns for p in self.root.rglob("*")}
        cwd = pathlib.Path.cwd()
        self.assertEqual(self._run()[0], 0)
        self.assertEqual(self._run("--reference", "a")[0], 0)
        self.assertEqual(
            before, {p: p.stat().st_mtime_ns
                     for p in self.root.rglob("*")})
        self.assertEqual(pathlib.Path.cwd(), cwd)

    def test_refusals_name_the_fix_and_report_no_skill(self):
        cases = {
            "no xprof distribution": {},
            "only supported channel": {
                "xprof": "2.24.0",
                "xprof-nightly": "2.24.3a20261015"
            },
            "`xprof-nightly` distribution is installed": {
                "xprof-nightly": "2.24.3a20261015"
            },
            "older than this package's floor": {
                "xprof": "2.23.2"
            },
        }
        for expected, dists in cases.items():
            with self.subTest(expected=expected):
                report = self._locate(dists)
                self.assertEqual(len(report["errors"]), 1)
                self.assertIn(expected, report["errors"][0])
                # The fix is the user's to run, under this exact interpreter.
                self.assertIn(f"Ask the user to run: {sys.executable} -m pip",
                              report["errors"][0])
                self.assertIn(
                    f"{sys.executable} -m pip install -U "
                    f"'xprof>={locate_xprof_skill.MINIMUM_VERSION}'",
                    report["errors"][0])
                self.assertIsNone(report["skill_md"])

    def test_a_wheel_without_the_skill_is_refused(self):
        shutil.rmtree(self.skill)
        report = self._locate({"xprof": "2.24.0"})
        self.assertIn("install is incomplete", report["errors"][0])
        self.assertIsNone(report["skill_md"])

    def test_an_uncomparable_version_warns_and_goes_on(self):
        with mock.patch.object(locate_xprof_skill,
                               "below_floor",
                               return_value=None):
            report = self._locate({"xprof": "weird"})
        self.assertEqual(report["errors"], [])
        self.assertTrue(
            any("could not compare" in w for w in report["warnings"]))

    def test_an_unusable_xprof_prints_the_fix_and_exits_1(self):
        code, out = self._run(dists={})
        self.assertEqual(code, 1)
        self.assertIn("XProf is unusable here", out)
        self.assertIn(f"Ask the user to run: {sys.executable} -m pip", out)
        self.assertNotIn("SKILL.md =====", out)
        self.assertEqual(self._run("--reference", "a", dists={})[0], 1)


class TestShipping(unittest.TestCase):

    def test_callers_relay_an_unusable_xprof_to_the_user(self):
        """The entry point, the agent and the result contract all hand the
        fix to the user; none lets an agent install XProf itself."""
        tpu_agent_suite = (_PACKAGE_ROOT / "skills" / "tpu-agent" /
                           "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("xprof_unavailable", tpu_agent_suite)
        self.assertIn("{{built-in:ask_user_question}}, quoting the reason",
                      tpu_agent_suite)
        self.assertIn("create a virtual\n   environment", tpu_agent_suite)
        profiling = (_PACKAGE_ROOT / "agents" /
                     "tpu-diagnostics.md").read_text(encoding="utf-8")
        self.assertIn("Your caller relays that to the user.", profiling)
        result = json.loads(
            (_PACKAGE_ROOT / "schemas" /
             "profiling-result.json").read_text(encoding="utf-8"))
        self.assertIn("never installs XProf itself",
                      result["properties"]["failure_reason"]["description"])

    def test_the_agent_ships_the_script_and_nothing_copies_the_skill(self):
        """tpu-diagnostics runs the script from the installed package, and no
        install step copies the skill into a harness: there is no second copy
        to own or to go stale."""
        with tempfile.TemporaryDirectory() as out:
            dist = pathlib.Path(out) / "dist"
            with contextlib.redirect_stderr(io.StringIO()):
                translate(_PACKAGE_ROOT, "claude-code", dist)
            self.assertTrue((dist / "scripts" / "tpu-diagnostics" /
                             "locate_xprof_skill.py").is_file())
            self.assertEqual(list(dist.rglob("skills/xprof")), [])


if __name__ == "__main__":
    unittest.main()
