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
"""Unit tests for standalone runtime operations.

Covers the runtime CLI, package and version resolution, centralized
storage, the workspace environment, purge-data, and the agent contracts
(the completed-artifact rule and the stage-result schemas). Hook tests
live in `framework/tests/hooks/`.
"""

import contextlib
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

# Imports below follow this sys.path setup, so the file runs directly.
# ruff: noqa: E402
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import agent_contracts, paths, run_state, runtime, utils
from framework.harnesses import google as google_harness
from framework.hooks import dispatch, events
from framework.tests.hooks import helpers

Path = pathlib.Path

# The shipped package's schemas, from the repository root.
_PACKAGE_SCHEMAS = Path(__file__).resolve().parents[2] / "package" / "schemas"


def setUpModule():
    helpers.set_up_module()


def tearDownModule():
    helpers.tear_down_module()


class TestRuntimeCli(helpers.RuntimeTestCase):
    """The runtime command line."""

    def test_validate_cli_exit_code_and_completed_artifact_rule(self):
        """`validate` exits 0 on a valid instance and non-zero otherwise.

        It applies the runtime's completed-artifact rule, not only the schema.
        """
        schema = self.temp_dir / "result.json"
        schema.write_text(json.dumps({
            "type": "object",
            "required": ["status"],
            "properties": {
                "status": {
                    "type": "string"
                },
                "artifacts": {
                    "type": "array"
                },
            },
        }),
                          encoding="utf-8")
        instance = self.temp_dir / "instance.json"
        for doc, ok in (({
                "status": "ok"
        }, True), ({
                "status": "completed"
        }, False)):
            with self.subTest(doc=doc):
                instance.write_text(json.dumps(doc), encoding="utf-8")
                code, _, _ = self._main_capturing(
                    ["validate", "--schema",
                     str(schema),
                     str(instance)])
                self.assertEqual(code == 0, ok)


class TestAgentContracts(helpers.RuntimeTestCase):
    """The completed-artifact rule and the stage-result schemas."""

    def _package_schemas(self) -> dict[str, dict]:
        schemas = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(_PACKAGE_SCHEMAS.glob("*.json"))
        }
        self.assertTrue(schemas, f"no schemas under {_PACKAGE_SCHEMAS}")
        return {k: v for k, v in schemas.items() if isinstance(v, dict)}

    def test_status_requires_is_coherent_in_every_package_schema(self):
        """A status that claims an outcome owes its payload; others owe nothing.

        Success-only fields must not be `required`, empty values do not count
        (but `succeeded: false` does), and every other status in the schema's
        own enum must pass without them.
        """
        check = agent_contracts.check_status_payload
        found = {
            name: schema
            for name, schema in self._package_schemas().items()
            if "status_requires" in schema
        }
        self.assertTrue(found, "no schema declares status_requires")
        for name, schema in found.items():
            owed = schema["status_requires"]
            for status, fields in owed.items():
                with self.subTest(schema=name, status=status):
                    self.assertFalse(
                        set(fields) & set(schema["required"]),
                        "a success-only field is required")
                    payload = {"status": status, "run_id": "x"}
                    self.assertTrue(check(schema, payload))
                    payload.update({field: "filled" for field in fields})
                    self.assertFalse(check(schema, payload))
                    for field in fields:
                        empty = [] if field == "artifacts" else ""
                        self.assertTrue(
                            check(schema, {
                                **payload, field: empty
                            }), field)
                    if "succeeded" in fields:
                        self.assertFalse(
                            check(schema, {
                                **payload, "succeeded": False
                            }))
            unfinished = [
                status for status in schema["properties"]["status"]["enum"]
                if status not in owed
            ]
            self.assertTrue(unfinished, f"{name} cannot report failure")
            for status in unfinished:
                with self.subTest(schema=name, status=status):
                    self.assertFalse(
                        check(schema, {
                            "status": status,
                            "failure_reason": "no_device"
                        }))

    def test_schema_examples_satisfy_their_own_schema(self):
        """Denials quote an example back to the agent as the expected output.

        One that does not validate is a retry loop.
        """
        checked = 0
        for name, schema in self._package_schemas().items():
            for index, example in enumerate(schema.get("examples", [])):
                checked += 1
                with self.subTest(schema=name, example=index):
                    errors = utils.validate_instance(example, schema)
                    errors += agent_contracts.check_status_payload(
                        schema, example, name)
                    self.assertFalse(errors)
        self.assertGreater(checked, 0, "no schema declares an example")

    def test_status_requires_must_name_declared_statuses_and_properties(self):
        """A misspelled name has to fail the build, not the rule.

        The rule fires only on the status it is keyed to, so a typo in either
        the status or a field name enforces nothing while the schema keeps
        validating -- it fails open, quietly, in exactly the case it exists to
        catch. Now that each package names its own statuses there is no shared
        vocabulary to fall back on, so a mistyped status is the likelier of the
        two. Catching both at verify time is why the rule is worth declaring in
        the schema at all.
        """
        good = {
            "type": "object",
            "status_requires": {
                "measured": ["latency_ms"]
            },
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["measured", "failed"]
                },
                "latency_ms": {
                    "type": "number"
                },
            },
        }
        self.assertFalse(utils.validate_schema_definition(good))

        misspelled_field = json.loads(json.dumps(good))
        misspelled_field["status_requires"]["measured"] = [
            "latency_ms", "nonexistent_field"
        ]
        errors = utils.validate_schema_definition(misspelled_field)
        self.assertTrue(errors)
        self.assertIn("nonexistent_field", errors[0])

        misspelled_status = json.loads(json.dumps(good))
        misspelled_status["status_requires"] = {"measurred": ["latency_ms"]}
        errors = utils.validate_schema_definition(misspelled_status)
        self.assertTrue(errors)
        self.assertIn("measurred", errors[0])


class TestPackageResolution(helpers.RuntimeTestCase):
    """Package and version resolution."""

    def test_unversioned_package_resolves_to_central_install(self):
        """A caller who names a package without its version must still find it.

        Versions are an install detail no agent tracks, so an unversioned
        lookup is the normal case. Missing the versioned directory sends the
        caller to a workspace path holding only part of the package, and
        anything installed solely in central storage then reads as absent --
        with the error naming that content rather than the resolution.
        """
        central = self.central_home / "packages" / "central-pkg@2.3.0"
        (central / "schemas").mkdir(parents=True)
        (central / "schemas" / "schema.json").write_text(json.dumps(
            {"type": "object"}),
                                                         encoding="utf-8")

        resolved = paths.get_installed_package_root("central-pkg")
        self.assertEqual(resolved.resolve(), central.resolve())
        self.assertTrue((resolved / "schemas" / "schema.json").is_file())
        # Compiled prompts pass --package as <name>@<version>; the tag must
        # resolve to the versioned directory the installer created.
        self.assertEqual(
            paths.get_installed_package_root("test-pkg@1.0.0").resolve(),
            self.pkg_root.resolve())


class TestWorkspaceEnvironment(helpers.RuntimeTestCase):
    """Reading the workspace environment."""

    def test_cli_start_run_reads_workspace_environment(self):
        """`start-run` without --workspace consults the cwd's workspace install."""
        ws = self.temp_dir / "proj_ws"
        pkg_cfg = ws / ".tpu-agent-suite"
        pkg_cfg.mkdir(parents=True)
        (pkg_cfg / "environment.json").write_text(json.dumps({
            "revision": 3,
            "target": "proj"
        }),
                                                  encoding="utf-8")

        old_cwd = Path.cwd()
        buf = io.StringIO()
        try:
            os.chdir(ws)
            with contextlib.redirect_stdout(buf):
                ret = runtime.main(["start-run", "--package", "wspkg@1.0.0"])
        finally:
            os.chdir(old_cwd)

        self.assertEqual(ret, 0)
        doc = json.loads(buf.getvalue())
        self.assertEqual(doc["environment_revision"], 3)


class TestPurgeData(helpers.RuntimeTestCase):
    """The purge-data command."""

    def test_purge_data_clears_runs_and_session_records(self):
        """Purging runs must not leave sessions pointing at deleted namespaces.

        Session records are derived from runs: they name a run_id and cache the
        context advertising its artifact path. Surviving a purge, they would
        hand a live conversation a namespace that no longer exists.
        """
        run_doc = run_state.start_run(package="test-pkg@1.0.0",
                                      workspace=self.workspace_dir)
        run_root = paths.get_run_root(run_doc["run_id"])
        self.assertTrue((run_root / "artifacts").is_dir())
        self.assertEqual(paths.get_active_run_id(self.workspace_dir),
                         run_doc["run_id"])
        run_state.record_session_run(
            session_id="conv-purge",
            run_id=run_doc["run_id"],
            package="test-pkg@1.0.0",
        )
        google_harness.record_session_hook_context(
            session_id="conv-purge",
            hook_name="session-run-init",
            context_text="cached context",
            package="test-pkg@1.0.0",
        )
        self.assertTrue(run_root.is_dir())

        with contextlib.redirect_stdout(io.StringIO()):
            run_state.purge_data(force=True)

        self.assertFalse(paths.get_runs_root().is_dir())
        self.assertIsNone(run_state.resolve_session_run_id("conv-purge"))
        self.assertIsNone(
            google_harness.resolve_session_hook_context(
                "conv-purge", "session-run-init"))

    def test_purge_data_declined_keeps_everything(self):
        run_state.record_session_run(
            session_id="conv-keep-purge",
            run_id="20260913T170000Z-998877665544",
            package="test-pkg@1.0.0",
        )
        with mock.patch("builtins.input", return_value="n"):
            with contextlib.redirect_stderr(io.StringIO()) as err:
                run_state.purge_data()

        self.assertIn("cancelled", err.getvalue())
        self.assertEqual(run_state.resolve_session_run_id("conv-keep-purge"),
                         "20260913T170000Z-998877665544")


class TestWorkspaceRootResolution(unittest.TestCase):
    """Where a hook decides the workspace is.

    `hooks.json` installs inside the harness's own directory, so a hook command
    runs with that directory as its cwd rather than the workspace. Resolving
    from there has to climb out, or every run-scoped path the hook computes is
    wrong by one level -- silently, since each path still exists.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.workspace = self.tmp / "workspace"
        (self.workspace / ".tpu-agent-suite").mkdir(parents=True)
        (self.workspace / ".tpu-agent-suite" / "current_run_id").write_text(
            "20260928T225805Z-a09b00bd2610", encoding="utf-8")

    def test_harness_directories_resolve_to_their_parent(self):
        """Every directory a harness owns, including Antigravity's `.agents`."""
        for name in (".agents", ".agent", ".claude", ".cursor"):
            with self.subTest(harness_dir=name):
                harness_dir = self.workspace / name
                harness_dir.mkdir(exist_ok=True)
                with mock.patch.object(pathlib.Path,
                                       "cwd",
                                       return_value=harness_dir):
                    self.assertEqual(events.resolve_workspace_root(),
                                     self.workspace.resolve())

    def test_a_repository_is_not_a_workspace(self):
        """With no active run above the cwd, the cwd answers, not a git root.

        A bare `.tpu-agent-suite` (the installer creates one where no session
        started) is not a marker; only the active-run file is.
        """
        repo = self.tmp / "bare-repo"
        (repo / ".git").mkdir(parents=True)
        (repo / ".tpu-agent-suite" / "logs").mkdir(parents=True)
        nested = repo / "src"
        nested.mkdir()
        with mock.patch.object(pathlib.Path, "cwd", return_value=nested):
            self.assertEqual(events.resolve_workspace_root(), nested.resolve())


class TestCompletedArtifactRuleIsConditional(unittest.TestCase):
    """Only a `completed` result must carry an artifact.

    The rule lives in the runtime rather than the schema because the validator
    has no conditional keywords, and stating it unconditionally would be worse
    than not stating it: an agent that failed has nothing to attach, so the only
    way to satisfy the rule would be to claim success.
    """

    def test_only_a_completed_result_owes_an_artifact(self):
        for result, owes in (
            ({
                "status": "completed",
                "summary": "all done"
            }, True),
            ({
                "status": "completed",
                "artifacts": []
            }, True),
            ({
                "status": "completed",
                "artifacts": [{
                    "file": "report.md"
                }]
            }, False),
            ({
                "status": "failed",
                "artifacts": []
            }, False),
            ({
                "status": "needs_input"
            }, False),
            ("not a result object", False),
        ):
            with self.subTest(result=result):
                self.assertEqual(
                    bool(agent_contracts.check_completed_has_artifact(result)),
                    owes)


class TestInstalledPackageResolution(unittest.TestCase):
    """Finding the versioned package directory under the TPU agent suite home."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.packages = self.tmp / "packages"
        self.packages.mkdir()

    def test_the_highest_installed_version_wins(self):
        for version in ("0.9.0", "0.10.0", "0.2.0"):
            (self.packages / f"pkg@{version}").mkdir()
        # None of these is a candidate.
        (self.packages / "other@9.0.0").mkdir()
        (self.packages / "pkg@8.0.0").write_text("not a directory",
                                                 encoding="utf-8")
        (self.packages / "pkg@not-a-version").mkdir()
        (self.packages / "unversioned").mkdir()
        with mock.patch.object(paths,
                               "get_tpu_agent_suite_home",
                               return_value=self.tmp):
            resolved = paths.latest_installed_package("pkg")
        # 0.10.0 beats 0.9.0 numerically, which string ordering gets wrong.
        self.assertEqual(resolved.name, "pkg@0.10.0")


class TestMissingPackageDegradesGracefully(unittest.TestCase):
    """A hook whose package has gone missing must still answer the harness.

    Hook commands bake a versioned package tag at compile time, so upgrading,
    renaming, or removing the package mid-session leaves the tag pointing at a
    directory that no longer exists. Package resolution raises for that, and an
    escaping error costs the dispatch its stdout: the harness is handed an
    empty string and a non-zero exit instead of a decision.
    """

    def setUp(self):
        super().setUp()
        # An empty store: nothing the hook names can resolve.
        self.temp_dir, self.workspace_dir = helpers.isolated_workspace(
            self, "empty_home")

    def test_validate_schema_builtin_still_returns_a_decision(self):
        """Validation degrades to "no contract", but says so.

        It degrades the way its other failures do (absent or unparseable build
        file, unknown agent), and says so since enforcing nothing silently is
        its own failure.
        """
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            res = agent_contracts.builtin_validate_schema(
                package="ghost-pkg@9.9.9",
                workspace=self.workspace_dir,
                event_payload={
                    "event": "SubagentStop",
                    "subagent": "helper",
                    "output": json.dumps({"status": "completed"}),
                },
            )
        self.assertEqual(res["decision"], "allow")
        self.assertIn("ghost-pkg@9.9.9", stderr.getvalue())

    def test_script_hook_denies_with_the_reason(self):
        """A script hook denies instead of skipping.

        As it does for any script it cannot locate -- but the harness still
        gets a decision.
        """
        with contextlib.redirect_stderr(io.StringIO()):
            response, code = dispatch.dispatch_hook(
                harness="claude-code",
                package="ghost-pkg@9.9.9",
                hook_name="some-script-hook",
                event="PreToolUse",
                script_rel="scripts/whatever.py",
                workspace=self.workspace_dir,
                raw_input_text=json.dumps({
                    "session_id": "s1",
                    "tool_name": "Bash",
                    "tool_input": {},
                }),
            )

        self.assertEqual(code, 0)
        specific = response["hookSpecificOutput"]
        self.assertEqual(specific["permissionDecision"], "deny")
        self.assertIn("is not installed", specific["permissionDecisionReason"])


if __name__ == "__main__":
    unittest.main()
