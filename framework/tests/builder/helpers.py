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
"""Shared fixtures for the package build tests."""

import json
import pathlib
import re
import shutil
import tempfile
import unittest

from framework import jsonio

Path = pathlib.Path


class CompilerTestCase(unittest.TestCase):
    """Unit tests for package verification, delegation graph, and translation.

    Shared base: a temp dir per test, a small valid package, and helpers
    that edit its manifest and documents.
    """

    def setUp(self):
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def _create_valid_package(self) -> Path:
        pkg_dir = self.temp_dir / "valid-pkg"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "scripts").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "scripts" / "custom.py").write_text("print('hello')",
                                                       encoding="utf-8")

        manifest = {
            "name":
            "valid-pkg",
            "version":
            "1.0.0",
            "description":
            "Valid test package",
            "native_tools": [
                {
                    "name": "read",
                    "visibility": "external",
                    "action_level": {
                        "effect": "read",
                        "scope": "workspace"
                    },
                },
                {
                    "name": "write",
                    "visibility": "external",
                    "action_level": {
                        "effect": "write",
                        "scope": "workspace"
                    },
                },
            ],
            "script_tools": [{
                "name": "custom_runner",
                "path": "scripts/custom.py",
                "visibility": "internal",
                "action_level": {
                    "effect": "execute",
                    "scope": "host"
                },
                "description": "Custom runner",
            }],
            "mcp_servers": [{
                "name": "query_db",
                "command": "python3",
                "visibility": "external",
                "action_level": {
                    "effect": "read",
                    "scope": "workspace"
                },
            }],
            "skills": ["entry"],
            "copy": ["schemas", "environment", "scripts"],
            "environment": {
                "questions": "environment/questions.json",
                "schema": "schemas/env.json",
            },
            "compatibility_targets": [
                {
                    "name": "claude",
                    "harness": "claude-code",
                    "versions": ">=2.0.0",
                },
                {
                    "name": "codex",
                    "harness": "codex",
                    "versions": ">=0.1.0",
                },
                {
                    "name": "antigravity",
                    "harness": "antigravity",
                    "versions": ">=1.0.0",
                },
                {
                    "name": "cursor",
                    "harness": "cursor",
                    "versions": ">=0.40.0",
                },
            ],
        }
        self._write_manifest(pkg_dir, manifest)
        (pkg_dir / "skills" / "entry").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "agents").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "skills" / "entry" / "SKILL.md").write_text(
            "---\n"
            "name: entry\n"
            "description: Entry description\n"
            "delegates:\n"
            "  - worker\n"
            "---\n"
            "Call {{agent:worker}}.",
            encoding="utf-8")
        (pkg_dir / "agents" / "worker.md").write_text(
            "---\n"
            "name: worker\n"
            "description: Worker\n"
            "tools:\n"
            "  - read\n"
            "  - write\n"
            "  - custom_runner\n"
            "  - query_db\n"
            "delegates: []\n"
            "accepts: schemas/in.json\n"
            "produces: schemas/out.json\n"
            "---\n"
            "Work.",
            encoding="utf-8")
        (pkg_dir / "schemas").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "schemas" / "in.json").write_text(json.dumps(
            {"type": "object"}),
                                                     encoding="utf-8")
        (pkg_dir / "schemas" / "out.json").write_text(json.dumps(
            {"type": "object"}),
                                                      encoding="utf-8")
        (pkg_dir / "schemas" / "env.json").write_text(
            json.dumps({
                "type": "object",
                "properties": {
                    "tpu_target": {
                        "type": "string"
                    }
                },
                "required": ["tpu_target"],
            }),
            encoding="utf-8",
        )
        (pkg_dir / "environment").mkdir(parents=True, exist_ok=True)
        (pkg_dir / "environment" / "questions.json").write_text(
            json.dumps({
                "questions": [{
                    "name": "tpu_target",
                    "prompt": "TPU target",
                    "type": "string",
                    "default": "local",
                }]
            }),
            encoding="utf-8",
        )
        return pkg_dir

    @staticmethod
    def _read_manifest(pkg_dir: Path) -> dict:
        return jsonio.load_json(pkg_dir / "package.release.json")

    @staticmethod
    def _write_manifest(pkg_dir: Path, manifest: dict) -> None:
        (pkg_dir / "package.release.json").write_text(json.dumps(manifest),
                                                      encoding="utf-8")

    def _edit_worker(self, pkg_dir: Path, old: str, new: str) -> None:
        """Replace the first `old` in the worker agent's document."""
        path = pkg_dir / "agents" / "worker.md"
        path.write_text(path.read_text(encoding="utf-8").replace(old, new, 1),
                        encoding="utf-8")

    def _delegate_to(self, pkg_dir: Path, *names: str) -> None:
        """Point the entry skill at `names`, which is how an agent is reached."""
        delegates = "".join(f"  - {name}\n" for name in names)
        routes = "\n".join(f"Call {{{{agent:{name}}}}}." for name in names)
        (pkg_dir / "skills" / "entry" / "SKILL.md").write_text(
            "---\n"
            "name: entry\n"
            "description: Entry description\n" +
            (f"delegates:\n{delegates}" if names else "delegates: []\n") +
            "---\n" + (routes or "Standalone."),
            encoding="utf-8")

    def _declare_dependencies(self, pkg_dir: Path, host=(), tpu=()) -> None:
        """Declare dependencies where they are read from: a unit's frontmatter."""
        path = pkg_dir / "agents" / "worker.md"
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"(?m)^(host|tpu)_dependencies:\n(?:  - .*\n)*", "",
                      text)
        extra = ""
        if host:
            extra += "host_dependencies:\n" + "".join(f"  - {d}\n"
                                                      for d in host)
        if tpu:
            extra += "tpu_dependencies:\n" + "".join(f"  - {d}\n" for d in tpu)
        path.write_text(text.replace("delegates: []\n",
                                     "delegates: []\n" + extra, 1),
                        encoding="utf-8")
