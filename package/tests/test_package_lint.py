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
"""Content checks run against the real package.

`instruction-links`, `reference-links`, and `environment-contract` are enforced
by `verify.verify_package`, so they are asserted here through `verify` itself
-- that is the gate, and these tests exist so `make test` fails when it would.

The remaining checks are heuristic and deliberately not gates: an agent may name
another agent without invoking it, a reference doc may legitimately mention a
Bazel fallback, and a backticked path may be illustrative rather than
a real reference. They are reported, not asserted.
"""

import json
import pathlib
import unittest

from framework import jsonio, lint
from framework.builder import package_loader, verify

Path = pathlib.Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]

# Kept in step with the tuple `verify.verify_package` iterates. A check listed
# here is a merge gate; everything else in `lint.CHECKS` is advisory.
GATED_CHECKS = ("instruction-links", "reference-links", "environment-contract",
                "workspace-config-paths")


class TestPackageLint(unittest.TestCase):
    """Gate on the decidable checks; report the heuristic ones."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = package_loader.load_package(PACKAGE_DIR)
        cls.findings = lint.run_checks(PACKAGE_DIR, cls.manifest)

    def test_verify_package_passes(self):
        """The gate itself, including the two enforced content checks."""
        verify.verify_package(PACKAGE_DIR)

    def test_no_unresolved_instruction_links(self):
        offenders = [
            f for f in self.findings if f.check == "instruction-links"
        ]
        self.assertFalse(
            offenders,
            "every markdown link in an agent prompt must name a file that agent"
            " declares in scripts/references:\n" +
            "\n".join(f"  {f}" for f in offenders),
        )

    def test_no_unresolved_reference_doc_links(self):
        offenders = [f for f in self.findings if f.check == "reference-links"]
        self.assertFalse(
            offenders,
            "every markdown link in a bundled reference doc must land on a file"
            " some agent declares, or it dangles in the built distribution:\n"
            + "\n".join(f"  {f}" for f in offenders),
        )

    def test_environment_contract_is_satisfiable(self):
        offenders = [
            f for f in self.findings if f.check == "environment-contract"
        ]
        self.assertFalse(
            offenders,
            "every environment key a bundled script reads must be answerable by"
            " a question in questions.json:\n" + "\n".join(f"  {f}"
                                                           for f in offenders),
        )

    def test_no_instruction_to_use_a_harness_only_tool(self):
        """Asserted, unlike the rest of legacy-paths: this one is decidable.

        "use `ask_question`" is an instruction the agent can only follow in
        the one harness that has the tool; the package ships to four. Naming
        a tool is still allowed -- only telling the agent to use it is not.
        """
        offenders = [
            f for f in self.findings if f.check == "legacy-paths"
            and f.detail.startswith("harness-tool-directive:")
        ]
        self.assertFalse(
            offenders,
            "say what to do (\"ask the user\") and let the harness choose the"
            " tool:\n" + "\n".join(f"  {f}" for f in offenders),
        )

    def test_every_check_runs_against_the_package(self):
        """Guards against a check silently becoming a no-op."""
        for name, check in lint.CHECKS.items():
            with self.subTest(check=name):
                check(PACKAGE_DIR, self.manifest)

    def test_gated_checks_match_the_verify_gate(self):
        """GATED_CHECKS drifting from `verify` would silently downgrade a gate."""
        from framework import lint as lint_module
        gated_here = {lint_module.CHECKS[name] for name in GATED_CHECKS}
        self.assertEqual(gated_here, set(verify.VERIFY_CONTENT_CHECKS))

    def test_all_supported_harnesses_declared_in_compatibility_targets(self):
        """Ensures that every harness supported by the compiler has a compatibility target declared."""
        declared = {
            target["harness"]
            for target in self.manifest.get("compatibility_targets", [])
        }
        for harness in ("claude-code", "codex", "antigravity", "cursor"):
            with self.subTest(harness=harness):
                self.assertIn(
                    harness,
                    declared,
                    f"harness '{harness}' is supported by compiler but missing from compatibility_targets in package.release.json",
                )


class GatedDelegatesTest(unittest.TestCase):
    """tpu-kernel-development is routed to from a single routing bullet in SKILL.md."""

    ROUTED = ("tpu-kernel-development", )

    def setUp(self):
        raw = jsonio.load_json(PACKAGE_DIR / "package.release.json")
        self.lines = (PACKAGE_DIR / "skills" / raw["skills"][0] /
                      "SKILL.md").read_text(encoding="utf-8").splitlines()

    def test_each_routed_agent_is_routed_to_from_exactly_one_line(self):
        for name in self.ROUTED:
            with self.subTest(agent=name):
                hits = [
                    ln for ln in self.lines if f"{{{{agent:{name}}}}}" in ln
                ]
                self.assertEqual(
                    len(hits), 1,
                    f"{name} is referenced on {len(hits)} lines; it must be "
                    f"routed to from one routing bullet")


class DefaultImageTest(unittest.TestCase):
    """docs/install.md names the default image in questions.json."""

    def test_the_docs_name_the_default_image(self):
        questions = json.loads(
            (PACKAGE_DIR / "environment" / "questions.json").read_text())
        image = next(q["default"] for q in questions["questions"]
                     if q["name"] == "image")
        docs = (PACKAGE_DIR.parent / "docs" / "install.md").read_text()
        self.assertIn(image, docs)


class TpuAgentSuiteOrchestrationInvariantsTest(unittest.TestCase):
    """Guards against regressions in tpu-agent orchestration rules."""

    def setUp(self):
        raw = jsonio.load_json(PACKAGE_DIR / "package.release.json")
        self.text = (PACKAGE_DIR / "skills" / raw["skills"][0] /
                     "SKILL.md").read_text(encoding="utf-8")

    def test_staged_pipeline_forbids_mode_both(self):
        self.assertIn('do **NOT** pass `mode: "both"`', self.text)
        self.assertIn('mode: "inference"', self.text)
        self.assertIn("omit `--serve`", self.text)

    def test_mandatory_profiling_gate_is_enforced(self):
        self.assertIn("Mandatory profiling gate", self.text)
        self.assertIn("empirical bottleneck `levers`", self.text)

    def test_skill_states_no_waiting_mechanism(self):
        """How to wait for a subagent differs by harness, and the compiler
        renders the right contract into the same prompt. A rule here can only
        agree with one harness and contradict the other."""
        for phrase in ("transcript.jsonl", ".system_generated/", "task-*.log",
                       "poll for subagent status", "Do NOT poll subagents"):
            self.assertNotIn(phrase, self.text)


if __name__ == "__main__":
    unittest.main()
