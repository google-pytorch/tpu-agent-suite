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
"""Unit tests for the package content checks in framework.lint."""

import json
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

from framework import lint

Path = pathlib.Path


class _LintTestCase(unittest.TestCase):
    """Helpers over a fixture package in `self.pkg` described by `self.manifest`.

    Subclasses build both in setUp; agent-a is always the first agent.
    """

    def tearDown(self):
        shutil.rmtree(self.pkg, ignore_errors=True)
        super().tearDown()

    def _write(self, rel, text):
        path = self.pkg / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _run(self, check):
        return lint.CHECKS[check](self.pkg, self.manifest)

    def _labels(self, check):
        """The rule label of each finding `check` reports, in order."""
        return [f.detail.split(":")[0] for f in self._run(check)]

    def _declare_scripts(self, files):
        """Write `files` (path -> text) and make them agent-a's scripts."""
        for rel, text in files.items():
            self._write(rel, text)
        self.manifest["agents"][0]["scripts"] = list(files)

    def _declare_script(self, text):
        """Write `text` as agent-a's one script, agent_a/scripts/tool.py."""
        self._declare_scripts({"agent_a/scripts/tool.py": text})


class TestLint(_LintTestCase):
    """Each check against a violating fixture, plus clean cases that regressed."""

    def setUp(self):
        super().setUp()
        self.pkg = Path(tempfile.mkdtemp())
        self._write("agent_a/references/guide.md", "# Guide")
        # The skill is checked like any other prompt, so the fixture keeps it
        # clean: it names its one delegate with a token. A test that wants
        # findings on the skill overwrites this file.
        self._write("agent_a/instructions/entry.md",
                    "Entry point. Delegate to {{agent:agent-a}}.")
        self._write("agent_a/instructions/a.md", "Do the thing.")
        self.manifest = {
            "name":
            "fixture",
            "skills": [{
                "name": "entry",
                "description": "Entry description",
                "instructions": "agent_a/instructions/entry.md",
                "delegates": ["agent-a"],
            }],
            "agents": [{
                "name": "agent-a",
                "instructions": "agent_a/instructions/a.md",
                "delegates": [],
                "scripts": [],
                "references": ["agent_a/references/guide.md"],
            }],
        }

    # --- instruction-links ------------------------------------------------

    def test_declared_external_and_runtime_links_are_clean(self):
        """Only links naming an undeclared package file are findings.

        Run-time targets regressed twice: `.tpu-agent-suite/` was a known
        runtime root only for `bare-paths`, and the home-relative `~/` spelling
        was not one at all -- and this check gates `verify`, so each false
        positive failed the build. The same link filter serves
        `reference-links`.
        """
        self._write(
            "agent_a/instructions/a.md",
            "See [guide](agent_a/references/guide.md).\n"
            "See [docs](https://example.com) and [top](#heading).\n"
            "See [report](<artifacts>/0/report_0.md),"
            " [state](.tpu-agent-suite/pkg/runs/<run_id>/artifacts/state.md),"
            " and [out](workspace/pkg/runs/<run_id>/artifacts/out.json).\n"
            "Source run: [report](~/.tpu-agent-suite/runs/<run_id>/artifacts/...).",
        )
        self.assertEqual(self._run("instruction-links"), [])

    # --- bare-paths -------------------------------------------------------

    def _add_agent_b(self, *, delegated_from_a: bool) -> None:
        """A second agent owning `agent_b/references/knobs.md`.

        Whether agent-a can reach it by delegating is the only knob.
        """
        self._write("agent_b/references/knobs.md", "# Knobs")
        self._write("agent_b/instructions/b.md", "Nothing here.")
        if delegated_from_a:
            self.manifest["agents"][0]["delegates"] = ["agent-b"]
        self.manifest["agents"].append({
            "name":
            "agent-b",
            "instructions":
            "agent_b/instructions/b.md",
            "delegates": [],
            "scripts": [],
            "references": ["agent_b/references/knobs.md"],
        })

    def test_bare_path_declared_by_a_delegate_is_ignored(self):
        """Describing a delegate's work names the delegate's files, not its own.

        tpu-sharding's orchestrator explains that the Optimizer reads the knob wiki;
        it never opens the file itself, so the path belongs to the delegate's
        declarations, not a gap in the orchestrator's. The exemption follows
        the delegation edge -- it is not package-wide.
        """
        self._add_agent_b(delegated_from_a=True)
        self._write(
            "agent_a/instructions/a.md",
            "Delegate to {{agent:agent-b}}. It reads"
            " `agent_b/references/knobs.md` for allowed parameters.",
        )
        self.assertEqual(self._run("bare-paths"), [])

        self.manifest["agents"][0]["delegates"] = []
        findings = self._run("bare-paths")
        self.assertEqual([f.agent for f in findings], ["agent-a"])
        self.assertIn("not declared", findings[0].detail)

    def test_link_to_a_delegates_file_is_still_flagged(self):
        """`bare-paths` exempts a delegate's file; `instruction-links` must not.

        A link tells the reader to open the target, and translate bundles it for
        the delegate only -- so it dangles here whoever else declares it.
        """
        self._add_agent_b(delegated_from_a=True)
        self._write("agent_a/instructions/a.md",
                    "See [knobs](agent_b/references/knobs.md).")
        findings = self._run("instruction-links")
        self.assertEqual(len(findings), 1)
        self.assertIn("not declared", findings[0].detail)

    # --- reference-links --------------------------------------------------

    def test_link_to_an_undeclared_sibling_is_flagged(self):
        """It resolves on disk, but nothing bundles it, so it dangles at run time."""
        self._write("agent_a/references/other.md", "# Other")
        self._write("agent_a/references/guide.md", "See [other](other.md).")
        findings = self._run("reference-links")
        self.assertEqual(len(findings), 1)
        self.assertIn("other.md", findings[0].detail)

        self.manifest["agents"][0]["references"].append(
            "agent_a/references/other.md")
        self.assertEqual(self._run("reference-links"), [])

    # --- unused-declarations ----------------------------------------------

    def test_only_a_declaration_nothing_names_is_flagged(self):
        """A file is used when the prompt names it, or a sibling names or imports it.

        Indirect use is real: one bundled script starts another. A Python
        dependency is named by its module, not its filename -- tpu-sharding's
        `query_tpu_sharding.py` reaches `entry.py` as `from entry import ...`.
        The import allowance is a match, not a blanket pass for `.py`.
        """
        self._declare_scripts({
            "agent_a/scripts/client.py":
            "SERVER = 'server.py'\nfrom entry import Entry\nimport persistence\n",
            "agent_a/scripts/server.py": "pass\n",
            "agent_a/scripts/entry.py": "class Entry: pass\n",
            "agent_a/scripts/persistence.py": "pass\n",
            "agent_a/scripts/orphan.py": "pass\n",
        })
        self.manifest["agents"][0]["references"] = []
        self._write("agent_a/instructions/a.md", "Run `client.py`.")
        findings = self._run("unused-declarations")
        self.assertEqual([f.file for f in findings],
                         ["agent_a/scripts/orphan.py"])

    # --- delegation-consistency -------------------------------------------

    def test_skill_delegates_are_checked_too(self):
        """A skill that names its delegate in prose declares it silently."""
        self._write("agent_a/instructions/entry.md",
                    "For everything, invoke_subagent agent-a.")
        findings = self._run("delegation-consistency")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].agent, "entry")
        self.assertIn("never references", findings[0].detail)

    def test_token_outside_the_allowed_set_is_flagged(self):
        """An agent that *can* delegate, naming a target it may not invoke."""
        self.manifest["agents"][0]["delegates"] = ["agent-b"]
        self._write("agent_a/instructions/a.md",
                    "Invoke {{agent:agent-b}}, then {{agent:agent-c}}.")
        findings = self._run("delegation-consistency")
        self.assertEqual(len(findings), 1)
        self.assertIn("agent-c", findings[0].detail)
        self.assertIn("does not declare", findings[0].detail)

    # --- compute-contract -------------------------------------------------

    def test_compute_contract_accepts_a_nested_subcommands_flag(self):
        """`setup plan --fields` is the runtime's own fix for a missing field."""
        self._write(
            "agent_a/instructions/a.md",
            "Run `{{runtime}} compute setup plan --fields staging_bucket`,"
            " not `{{runtime}} compute setup plan --no-such-flag`.")
        findings = self._run("compute-contract")
        self.assertEqual(len(findings), 1)
        self.assertIn("--no-such-flag", findings[0].detail)

    def test_compute_contract_flags_unknown_subcommand(self):
        self._write("agent_a/instructions/a.md",
                    "Run `{{runtime}} compute exec --code x.py`.")
        findings = self._run("compute-contract")
        self.assertEqual(len(findings), 1)
        self.assertIn("unknown compute subcommand", findings[0].detail)

    def test_compute_contract_flags_unknown_backend(self):
        """A backend name that is not registered fails at resolve time."""
        self._write(
            "agent_a/instructions/a.md",
            "Run `{{runtime}} compute --backend gke run --code x.py`.\n"
            "Or `{{runtime}} compute run --backend=gke --code x.py`.")
        findings = self._run("compute-contract")
        self.assertEqual(len(findings), 2)
        for finding in findings:
            self.assertIn("unknown backend 'gke'", finding.detail)

    def test_compute_contract_reports_when_the_runtime_cannot_be_introspected(
            self):
        """A runtime that fails to load must not silently switch the check off."""
        self._write(
            "agent_a/instructions/a.md",
            "Run `{{runtime}} compute run --code x.py --no-such-flag`.")
        saved = (lint._COMPUTE_FLAGS_CACHE, lint._COMPUTE_BACKENDS_CACHE,
                 lint._COMPUTE_INTROSPECTION_ERROR)
        lint._COMPUTE_FLAGS_CACHE = None
        lint._COMPUTE_INTROSPECTION_ERROR = None
        try:
            with mock.patch("framework.runtime.build_runtime_parser",
                            side_effect=SyntaxError("broken runtime")):
                findings = self._run("compute-contract")
        finally:
            (lint._COMPUTE_FLAGS_CACHE, lint._COMPUTE_BACKENDS_CACHE,
             lint._COMPUTE_INTROSPECTION_ERROR) = saved
        self.assertEqual(len(findings), 1)
        self.assertIn("introspection failed", findings[0].detail)
        self.assertIn("broken runtime", findings[0].detail)
        self.assertIn("validation skipped", findings[0].detail)

    def test_compute_contract_scans_declared_references_and_scripts(self):
        """A declared reference reaches the agent exactly as a prompt does.

        A bundled tool that opens its own connection defeats the runtime
        whether or not any prompt mentions it, so scripts are scanned as well
        as prompts. This is the shape a package ported from a codebase with its
        own transport arrives in: the prompts get migrated and the tools they
        call keep shelling out.
        """
        self._write(
            "agent_a/references/guide.md",
            "Run `{{runtime}} compute run --code x.py --no-such-flag`.")
        self._declare_script("import subprocess\n"
                             'cmd = f"ssh {host} python3 -u -"\n'
                             "subprocess.run(cmd, shell=True)\n")
        findings = self._run("compute-contract")
        self.assertEqual(
            sorted(f.file for f in findings),
            ["agent_a/references/guide.md", "agent_a/scripts/tool.py"])
        self.assertIn(
            "raw ssh",
            next(f.detail for f in findings if f.file.endswith(".py")))

    def test_legacy_paths_flagged(self):
        """Legacy paths and AGY spellings in a prompt, Bazel even in prose."""
        self._write(
            "agent_a/instructions/a.md",
            "Build it locally to avoid Bazel overhead.\n"
            "Read experimental/tpu_kernel_development/tool.py.\n"
            "Then plan_kernel_agent.\n"
            "Then invoke_subagent torch-coder.\n"
            "Use view_file to read code.\n"
            "Use the Bash tool, then call TodoWrite.\n",
        )
        self.assertEqual(
            set(self._labels("legacy-paths")), {
                "bazel", "experimental-path", "agy-agent-name", "agy-invoke",
                "agy-tool", "harness-tool"
            })

    # --- environment-contract ---------------------------------------------

    def _setup_environment(self, questions, properties, closed=True):
        self._write(
            "environment/questions.json",
            json.dumps({
                "questions": [{
                    "name": q,
                    "type": "string"
                } for q in questions]
            }),
        )
        schema = {"type": "object", "properties": {p: {} for p in properties}}
        if closed:
            schema["additionalProperties"] = False
        self._write("schemas/env.json", json.dumps(schema))
        self.manifest["environment"] = {
            "questions": "environment/questions.json",
            "schema": "schemas/env.json",
        }

    def test_unanswerable_environment_key_is_flagged(self):
        self._setup_environment(["host"], ["host"])
        self._declare_script('p = "environment.json"\nh = env.get("host")\n'
                             'z = env.get("zone", "us-central2-b")\n')
        findings = self._run("environment-contract")
        self.assertEqual(len(findings), 1)
        self.assertIn("'zone'", findings[0].detail)
        self.assertIn("additionalProperties", findings[0].detail)

    def test_nested_question_path_is_resolved_against_the_schema(self):
        """A question landing at `compute.params.zone` needs that nested property."""
        self._setup_environment([], [])
        self._write(
            "environment/questions.json",
            json.dumps({
                "questions": [{
                    "name": "zone",
                    "path": "compute.params.zone"
                }, {
                    "name": "pod",
                    "path": "compute.params.pod",
                    "when": {
                        "flavour": ["x"]
                    }
                }]
            }))
        params = {"properties": {"zone": {}}}
        self._write(
            "schemas/env.json",
            json.dumps({
                "additionalProperties": False,
                "properties": {
                    "compute": {
                        "properties": {
                            "params": params
                        }
                    }
                },
            }))
        details = "\n".join(f.detail
                            for f in self._run("environment-contract"))
        self.assertIn("'pod' lands at 'compute.params.pod'", details)
        self.assertIn("conditional on 'flavour'", details)
        self.assertNotIn("'zone'", details)

    # --- legacy-imports ---------------------------------------------------

    def test_origin_rooted_imports_are_flagged(self):
        """Including in a companion *_test.py, which no frontmatter declares."""
        self._declare_script(
            "from skills.preflight.scripts import preflight\n"
            "import experimental.tpu_kernel_development.tools\n")
        self._write("agent_a/scripts/tool_test.py",
                    "from learning.agents import agent\n")
        findings = self._run("legacy-imports")
        modules = [f.detail.split("'")[1] for f in findings]
        self.assertEqual(modules, [
            "skills.preflight.scripts",
            "experimental.tpu_kernel_development.tools", "learning.agents"
        ])
        self.assertTrue(findings[-1].file.endswith("tool_test.py"))

    def test_unparseable_script_is_reported(self):
        self._declare_script("def broken(:\n")
        findings = self._run("legacy-imports")
        self.assertEqual(len(findings), 1)
        self.assertIn("cannot be parsed", findings[0].detail)

    # --- frontmatter-contracts --------------------------------------------

    def test_frontmatter_contract_violations(self):
        valid_rest = ("description: Desc.\naccepts: schemas/in.json\n"
                      "produces: schemas/out.json\ndelegates: []\n")
        for rel, text, expected in (
            ("a.md", "# Just Markdown",
             "missing leading '---' YAML frontmatter"),
            ("a.md", "---\nname: agent-a\nNo closing delimiter",
             "unclosed YAML frontmatter"),
            ("a.md", "---\nname: agent-a\n---\nPrompt.",
             "missing required contract field 'description'"),
            ("a.md",
             f"---\nname: agent-a\ntools:\n  - bash_tool\n{valid_rest}---\n",
             "invalid tool grants"),
            ("a.md", f"---\nname: wrong-name\n{valid_rest}---\n",
             "does not match agent name"),
            ("entry.md",
             "---\nname: entry\ndelegates: ['agent-a']\n---\nPrompt.",
             "missing required field 'description'"),
        ):
            with self.subTest(expected=expected):
                self._write(f"agent_a/instructions/{rel}", text)
                findings = self._run("frontmatter-contracts")
                self.assertTrue(any(expected in f.detail for f in findings),
                                findings)


class XprofOwnershipChecks(_LintTestCase):
    """Only tpu-diagnostics reads traces or runs the xprof CLI."""

    def setUp(self):
        super().setUp()
        self.pkg = Path(tempfile.mkdtemp())
        self._write(
            "agent_a/instructions/entry.md",
            "Delegate to {{agent:agent-a}} or {{agent:tpu-diagnostics}}.")
        self._write("agent_a/instructions/a.md", "Do the thing.")
        self._write("tpu-diagnostics/instructions/x.md", "Analyse.")
        self.manifest = {
            "name":
            "fixture",
            "skills": [{
                "name": "entry",
                "description": "Entry",
                "instructions": "agent_a/instructions/entry.md",
                "delegates": ["agent-a", "tpu-diagnostics"],
            }],
            "agents": [{
                "name": "agent-a",
                "instructions": "agent_a/instructions/a.md",
                "delegates": [],
                "scripts": [],
                "references": [],
            }, {
                "name": "tpu-diagnostics",
                "instructions": "tpu-diagnostics/instructions/x.md",
                "delegates": [],
                "scripts": [],
                "references": [],
            }],
        }

    # --- xplane-access ----------------------------------------------------

    def test_script_reading_xplane_proto_is_flagged(self):
        self._declare_script(
            "from tensorflow.tsl.profiler.protobuf import xplane_pb2\n"
            "from xprof.convert import raw_to_tool_data\n"
            "import tensorboard_plugin_profile.convert\n")
        findings = self._run("xplane-access")
        self.assertEqual([f.line for f in findings], [1, 2, 3])
        self.assertTrue(all("tpu-diagnostics" in f.detail for f in findings))

    # --- xprof-outside-analyst -------------------------------------------

    def test_the_analyst_is_known_by_name_wherever_its_prompt_sits(self):
        """The exemption followed a directory, which the layout by kind
        removed: the analyst's prompt is `agents/tpu-diagnostics.md` now. A
        prompt beside it is not exempt -- a shared directory is not
        ownership."""
        for name in ("tpu-diagnostics", "agent-a"):
            self._write(f"agents/{name}.md",
                        "Run `xprof get_overview LOGDIR`.")
        for agent in self.manifest["agents"]:
            agent["instructions"] = f"agents/{agent['name']}.md"
        findings = self._run("xprof-outside-analyst")
        self.assertEqual([f.agent for f in findings], ["agent-a"])

    # --- generic-compute-prose -------------------------------------------

    def test_generic_compute_prose_is_flagged(self):
        self._write(
            "agent_a/instructions/a.md",
            "Use `{{runtime}} compute run` for synchronous execution or"
            " `{{runtime}} compute submit` for asynchronous batch jobs.\n"
            "1. **Unified Compute Access**: Execute accelerator operations via"
            " `{{runtime}} compute` (supports `local`, `gce_tpu_vm`,"
            " `gke_pod`).\n"
            "It supports `local`, `gce_tpu_vm`, `gke_pod`.\n"
            "When execution on accelerator compute is required, execute via"
            " `{{runtime}} compute`:\n"
            "- Status: `{{runtime}} compute job status --job-id <job_id>`\n")
        self.assertEqual(self._labels("generic-compute-prose"), [
            "generic-run-vs-submit", "generic-compute-header",
            "generic-backend-list", "generic-compute-intro",
            "generic-job-lifecycle"
        ])


if __name__ == "__main__":
    unittest.main()
