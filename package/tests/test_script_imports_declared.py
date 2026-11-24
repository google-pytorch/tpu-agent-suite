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
"""Every package module a declared script imports is declared too.

Only declared scripts ship, so a script whose sibling module is missing from
the same frontmatter imports fine in the repo and fails with ImportError once
installed. That passed every unit test and blocked a hill-climb.
"""

import ast
import pathlib
import unittest

import yaml

PACKAGE = pathlib.Path(__file__).resolve().parents[1]


def _frontmatter(path: pathlib.Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    return yaml.safe_load(text.split("---", 2)[1]) or {}


def _imported(script: str) -> set[str]:
    """Package-relative paths of the local modules `script` imports."""
    path = PACKAGE / script
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    sdir = pathlib.PurePosixPath(script).parent
    bases = [
        sdir, sdir.parent,
        pathlib.PurePosixPath("scripts/common"), sdir.parent.parent
    ]
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node,
                        ast.ImportFrom) and node.module and not node.level:
            mods = [node.module
                    ] + [f"{node.module}.{a.name}" for a in node.names]
        else:
            continue
        for mod in mods:
            for base in bases:
                cand = base.joinpath(*mod.split(".")).with_suffix(".py")
                if (PACKAGE / cand).is_file():
                    if cand.name != "__init__.py":
                        found.add(str(cand))
                    break
    return found


class ScriptImportsDeclaredTest(unittest.TestCase):

    def test_imports_of_declared_scripts_are_declared(self):
        docs = sorted(PACKAGE.glob("agents/*.md")) + sorted(
            PACKAGE.glob("skills/*/SKILL.md"))
        for doc in docs:
            scripts = set(_frontmatter(doc).get("scripts") or [])
            missing = set()
            for script in sorted(s for s in scripts if s.endswith(".py")):
                missing |= _imported(script) - scripts
            with self.subTest(doc=str(doc.relative_to(PACKAGE))):
                self.assertEqual(
                    sorted(missing), [],
                    "imported by a declared script but not declared")


if __name__ == "__main__":
    unittest.main()
