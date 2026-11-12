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
"""Unit tests for shared utils module."""

import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

from framework import harnesses, jsonio, utils

Path = pathlib.Path


class TempDirTestCase(unittest.TestCase):
    """Base that gives each test a scratch directory, removed afterwards."""

    def setUp(self):
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)


class TestUtils(unittest.TestCase):
    """Unit tests for utils functions and config managers."""

    def test_semver_parsing_and_ranges(self):
        v1 = utils.SemVer.parse("1.2.3")
        v2 = utils.SemVer.parse("1.2.4")
        v3 = utils.SemVer.parse("2.0.0")
        self.assertTrue(v1 < v2 < v3)
        self.assertTrue(utils.version_satisfies(v1, ">=1.0.0,<2.0.0"))
        self.assertTrue(utils.version_satisfies(v1, "==1.2.3"))
        self.assertFalse(utils.version_satisfies(v1, ">1.2.3"))
        self.assertTrue(utils.version_satisfies(v2, ">=1.2.0"))

    def test_schema_validator(self):
        props = dict(name=dict(type="string", pattern=r"^[A-Z][a-z]+$"),
                     age=dict(type="integer"),
                     nick=dict(type=["string", "null"]))
        schema = dict(type="object",
                      required=["name", "age"],
                      properties=props,
                      additionalProperties=False)
        ok = dict(name="Alice", age=30)
        self.assertEqual(utils.validate_instance(dict(ok, nick=None), schema),
                         [])
        for instance, expected in (
            (dict(ok, nick=1), "$/nick: expected string or null, got int"),
            (dict(name="Alice"), "$/age: required property is missing"),
            (dict(ok, age=True), "$/age: expected integer, got bool"),
            (dict(ok, name="alice1"), "$/name: string does not match"),
        ):
            with self.subTest(instance=instance):
                errors = utils.validate_instance(instance, schema)
                self.assertTrue(any(e.startswith(expected) for e in errors))
        # additionalProperties: False wins over the flag; omitted means strict;
        # True needs the flag; a schema value validates the extra values.
        extra = "$/x: additional property is not allowed"
        integer = dict(type="integer")
        for additional, allow, value, expected in (
            (False, True, 1, [extra]),
            (None, False, 1, [extra]),
            (True, False, 1, [extra]),
            (True, True, 1, []),
            (integer, False, 1, []),
            (integer, False, "s", ["$/x: expected integer, got str"]),
        ):
            sub = dict(type="object",
                       properties=dict(name=dict(type="string")))
            if additional is not None:
                sub["additionalProperties"] = additional
            with self.subTest(additional=additional, allow=allow):
                self.assertEqual(
                    utils.validate_instance(dict(x=value),
                                            sub,
                                            allow_additional_properties=allow),
                    expected)

    def test_every_repo_schema_is_valid(self):
        """Checks every schema file, whether or not package.release.json lists it."""
        repo_root = Path(__file__).resolve().parents[2]
        not_schemas = {"event-mappings.json", "tool-mappings.json"}  # data
        paths = [
            p for d in ("package/schemas", "framework/schemas")
            for p in (repo_root / d).glob("*.json")
            if p.name not in not_schemas
        ]
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(schema=path.name):
                self.assertEqual(
                    utils.validate_schema_definition(jsonio.load_json(path)),
                    [])

    def test_event_mappings_cover_every_event_and_harness(self):
        """Every supported event maps, on every harness, to a named event.

        Each has a session_id source (else its run namespace cannot resolve).
        """
        events = utils.get_event_mappings()
        required = {
            "SubagentStop": ("subagent", "output"),
            "SessionStart": ("cwd", )
        }
        for event in utils.get_supported_hook_events():
            self.assertEqual(events[event]["canonical_name"], event)
            for h in harnesses.HARNESSES:
                with self.subTest(event=event, harness=h):
                    block = events[event]["harnesses"][h]
                    self.assertTrue(block["event_name"])
                    fields = block.get("fields", {})
                    for name in ("session_id", *required.get(event, ())):
                        self.assertIsInstance(fields.get(name), list)
                        self.assertTrue(fields[name])
                    # Cursor has no `source`; declaring it empty is a defect.
                    if event == "SessionStart" and "source" in fields:
                        self.assertTrue(fields["source"])

    def test_json_config_manager(self):
        doc = {"a": {"b": 1}}
        records = []
        utils.JsonConfigManager.merge(doc, {
            "a": {
                "c": 2
            },
            "d": 3
        }, [], records)
        self.assertEqual(doc, {"a": {"b": 1, "c": 2}, "d": 3})
        self.assertEqual(len(records), 2)


class TestMarkdownBlockManager(TempDirTestCase):
    """Unit tests for marker-delimited managed blocks in markdown files."""

    MARKER = "tpu-agent-suite:pkg:kb"

    def setUp(self):
        super().setUp()
        self.path = self.temp_dir / "CLAUDE.md"

    def test_malformed_markers_rejected(self):
        self.path.write_text(f"<!-- {self.MARKER}:begin -->\nno end marker\n",
                             encoding="utf-8")
        with self.assertRaises(utils.TpuAgentSuiteError):
            utils.MarkdownBlockManager.locate(
                self.path.read_text(encoding="utf-8"), self.MARKER)


class TestHookEventSources(unittest.TestCase):
    """The runtime and the schema must name the same lifecycle events."""

    def test_lookup_survives_a_missing_schema(self):
        """The fallback constant names the events the manifest schema declares.

        An installed package has no manifest schema beside its runtime, so
        the fallback is all it has.
        """
        self.assertEqual(utils.get_supported_hook_events(),
                         list(utils.SUPPORTED_HOOK_EVENTS))
        with mock.patch.object(
                utils,
                "get_package_manifest_schema",
                side_effect=utils.TpuAgentSuiteError("missing")):
            self.assertEqual(utils.get_supported_hook_events(),
                             list(utils.SUPPORTED_HOOK_EVENTS))


class TestExtractJsonObjects(unittest.TestCase):
    """Tests for extracting JSON dictionaries from raw text, code fences, and prose."""

    def test_extract_all_json_objects_from_text(self):
        """Extracts JSON dictionaries from code fences and unfenced text.

        Both 3- and 4-backtick fences count; prose, empty and non-string
        inputs yield none.
        """
        raw = ("Context config: {\"lr\": 0.01}\n\n"
               "Step 1 output:\n"
               "```json\n"
               '{"step": 1, "nested": {"val": [1, 2]}}\n'
               "```\n\n"
               "Final envelope:\n"
               "````json\n"
               '{"step": 2, "status": "completed"}\n'
               "````\n"
               "All done!")
        objs = utils.extract_all_json_objects(raw)
        self.assertEqual(objs, [{
            "lr": 0.01
        }, {
            "step": 1,
            "nested": {
                "val": [1, 2]
            }
        }, {
            "step": 2,
            "status": "completed"
        }])
        for raw in ("Just prose text", "", None):
            self.assertEqual(utils.extract_all_json_objects(raw), [])


class TestJsonSchemaDefinitionErrors(unittest.TestCase):
    """Malformed schema definitions are reported, not silently accepted."""

    def test_malformed_definitions_are_reported(self):
        """A malformed `type`, `oneOf` or `dependentRequired` is an error.

        It is a definition error, not a schema that silently validates
        nothing.
        """
        validate = utils.JsonSchemaValidator.validate_schema_definition
        for schema, expected in (
            ({
                "type": []
            }, "$/type: type array must not be empty"),
            ({
                "type": ["string", "string"]
            }, "$/type: type array must contain unique values"),
            ({
                "type": ["string", "bogus"]
            }, "$/type: unsupported type 'bogus'"),
            ({
                "oneOf": []
            }, "$/oneOf: expected a non-empty array of schemas"),
            ({
                "oneOf": [{
                    "bogus": 1
                }]
            }, "$/oneOf/0: unsupported schema keyword(s): bogus"),
            ({
                "dependentRequired": ["path"]
            },
             "$/dependentRequired: expected an object keyed by property name"),
            ({
                "dependentRequired": {
                    "slot": "path"
                }
            }, "$/dependentRequired/slot: expected an array of strings"),
        ):
            with self.subTest(schema=schema):
                self.assertEqual(validate(schema), [expected])


if __name__ == "__main__":
    unittest.main()
