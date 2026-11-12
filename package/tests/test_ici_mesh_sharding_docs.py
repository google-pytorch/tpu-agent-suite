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
"""ici-mesh-sharding's prose quotes only names its scripts emit.

An agent reads SKILL.md and the two references, then looks for the quoted
roles, slot roles, verdicts, error classes and output keys in the scripts'
output. A name the code does not have sends it looking for a field that is
not there, so every such name is checked against the code it comes from.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
import unittest

_PACKAGE = pathlib.Path(__file__).resolve().parents[1]
_SCRIPTS = _PACKAGE / "scripts" / "ici-mesh-sharding"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import collective_breakdown  # noqa: E402
import compile_verdict  # noqa: E402
import mesh_portfolio  # noqa: E402
import mesh_search  # noqa: E402
import mesh_topology  # noqa: E402
import sharding_adapter  # noqa: E402

SKILL = (_PACKAGE / "skills" / "ici-mesh-sharding" /
         "SKILL.md").read_text(encoding="utf-8")
ANALYZER = (_PACKAGE / "references" / "ici-mesh-sharding" /
            "analyzer.md").read_text(encoding="utf-8")
PROPOSAL = (_PACKAGE / "references" / "ici-mesh-sharding" /
            "proposal.md").read_text(encoding="utf-8")
DOCS = {"SKILL.md": SKILL, "analyzer.md": ANALYZER, "proposal.md": PROPOSAL}
# The authored knowledge-base page quotes model constants too.
KB_PAGE = (_PACKAGE / "data" / "tpu-knowledge-base" / "pages" /
           "ici-mesh-collective-attribution.md").read_text(encoding="utf-8")

ROLES = {
    value
    for name, value in vars(collective_breakdown).items()
    if name.startswith("ROLE_")
}
VERDICTS = {verdict.value for verdict in compile_verdict.Verdict}
ERROR_CLASSES = {
    value
    for name, value in vars(compile_verdict).items()
    if name.startswith("ERROR_") and isinstance(value, str)
}
SUMMARY_KEYS = set(collective_breakdown.summarize([], {"data": 2}))


def _slot_roles() -> set[str]:
    """The `role` strings `mesh_portfolio` gives proposals."""
    tree = ast.parse((_SCRIPTS / "mesh_portfolio.py").read_text("utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(
                node.func, "attr", None)
            index = {"_request_slot": 3, "take": 2}.get(name)
            if index is not None and len(node.args) > index:
                found.add(node.args[index])
        elif (isinstance(node, ast.Tuple) and len(node.elts) == 3 and getattr(
                node.elts[1], "id", None) in ("LAYOUT", "SLICES", "PIPELINE")):
            found.add(node.elts[2])
    return {
        node.value
        for node in found
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _section(text: str, title: str) -> str:
    """The body of the `## <n>. <title>` section."""
    match = re.search(rf"^## \d+\. {re.escape(title)}.*?$(.*?)(?=^## |\Z)",
                      text, re.M | re.S)
    if match is None:
        raise AssertionError(f"no section {title!r}")
    return match.group(1)


def _quoted(text: str) -> list[str]:
    return re.findall(r"`([^`]+)`", text)


def _flat(text: str) -> str:
    """`text` with each run of whitespace (line wraps included) as one space."""
    return " ".join(text.split())


def _group(expression: str, mesh: dict[str, int]) -> dict:
    """The breakdown's group for one collective with HLO `expression`."""
    kind = re.search(r"\] ([a-z-]+)\(", expression).group(1)
    op = {
        "operation": f"{kind}.1",
        "category": kind,
        "raw_time_ps": 1e8,
        "occurrences": 1,
        "expression": expression,
        "provenance": "jit(step)/block",
    }
    return collective_breakdown.summarize([op], mesh)["groups"][0]


class IciMeshShardingDocsTest(unittest.TestCase):

    def test_role_table_lists_exactly_the_role_constants(self):
        rows = re.findall(r"^\| `([^`]+)` \| (.*) \| (Yes|No) \|$",
                          _section(ANALYZER, "Collective Role Classification"),
                          re.M)
        self.assertEqual({role for role, _, _ in rows}, ROLES)
        self.assertEqual(
            {role
             for role, _, charged in rows if charged == "No"},
            set(collective_breakdown._NOT_PER_AXIS))

    def test_skill_names_every_role_and_only_roles(self):
        line = next(line for line in SKILL.splitlines()
                    if line.strip().startswith("- `roles`"))
        self.assertEqual(set(_quoted(line)) - {"roles", " (by shape)"}, ROLES)

    def test_breakdown_keys_exist(self):
        block = SKILL.split("Inspect (keys of the JSON output):", 1)[1]
        block = block.split("\n---", 1)[0]
        quoted = []
        for text in (_section(ANALYZER,
                              "Collective Role Classification"), block):
            quoted += re.findall(r"^\s*- `([a-z_]+)`", text, re.M)
        self.assertIn("per_axis", quoted)
        self.assertIn("largest_reducible", quoted)
        self.assertLessEqual(set(quoted), SUMMARY_KEYS)

    def test_slot_roles_exist_and_are_all_documented(self):
        template = _section(PROPOSAL, "The Portfolio Template")
        listed = set()
        for line in template.splitlines():
            if re.match(r"^\d\. ", line):
                listed |= set(_quoted(line))
        listed -= {"k = 2", "k"}
        roles = _slot_roles()
        self.assertTrue(roles)
        self.assertLessEqual(listed, roles)
        self.assertEqual(roles - listed, set())

    def test_verdicts_and_error_classes_exist(self):
        for name, text in DOCS.items():
            with self.subTest(doc=name):
                bold = set(re.findall(r"\*\*`([A-Z_]+)`\*\*", text))
                self.assertLessEqual(bold, VERDICTS)
        self.assertEqual(set(re.findall(r"\*\*`([A-Z_]+)`\*\*", SKILL)),
                         VERDICTS)
        dropped = next(line for line in PROPOSAL.splitlines()
                       if line.startswith("- **`DROPPED`**"))
        self.assertEqual(
            set(_quoted(dropped.split("`error_class` is one of", 1)[1])),
            ERROR_CLASSES)

    def test_quoted_constants_match_the_code(self):
        modules = (mesh_portfolio, collective_breakdown, compile_verdict,
                   mesh_search, mesh_topology, sharding_adapter)
        found = 0
        for name, text in {**DOCS, "KB page": KB_PAGE}.items():
            for constant, value in re.findall(r"`(_?[A-Z][A-Z_]+) = ([\d.]+)`",
                                              text):
                found += 1
                with self.subTest(doc=name, constant=constant):
                    owners = [m for m in modules if hasattr(m, constant)]
                    self.assertTrue(owners)
                    for module in owners:
                        self.assertEqual(float(getattr(module, constant)),
                                         float(value))
        self.assertGreaterEqual(found, 4)

    def test_worked_examples_give_the_documented_results(self):
        kb = _flat(KB_PAGE)
        gather = re.search(r"^%ag = .*$", KB_PAGE, re.M).group(0)
        all_reduce = "%r = f32[8] all-reduce(f32[8] %x), replica_groups="
        for quoted, expression, mesh, key, result in (
            ("`replica_groups=mesh['data'=8,'seq'=4] {'seq'}` (the groups span"
             " `seq`)", all_reduce + "mesh['data'=8,'seq'=4] {'seq'}", {
                 "data": 8,
                 "seq": 4
             }, "axes", "seq"),
            ("`replica_groups={{0,1},{2,3}}`, where the inner list length gives"
             " `group_size = 2`", all_reduce + "{{0,1},{2,3}}", {
                 "fsdp": 2,
                 "tensor": 4
             }, "evidence", "group size 2"),
            ("`fsdp=4, tensor=4`) and neither `mesh_axes` nor provenance tells"
             " them apart, report the time as ambiguous (the breakdown labels"
             " it `ambiguous: fsdp or tensor`)", all_reduce + "[4,4]<=[16]", {
                 "fsdp": 4,
                 "tensor": 4
             }, "axes", "ambiguous: fsdp or tensor"),
            ("= 8 -> group_size = 8", gather, {
                "fsdp": 8,
                "tensor": 2
            }, "evidence", "x8 (d0:1024->8192)"),
        ):
            with self.subTest(example=result):
                self.assertIn(quoted, kb)
                self.assertEqual(_group(expression, mesh)[key], result)
        self.assertIn("`x8 (d0:1024->8192)`", ANALYZER)

        skill = _flat(SKILL)
        self.assertIn(
            "check-topology \\ --mesh '{\"fsdp\":8,\"tensor\":4,\"expert\":2}'"
            " \\ --topology 4x4x4 \\ --platform v5p", skill)
        for doc, quoted, mesh, fits in (
            (kb, "`tensor=4, fsdp=4, expert=4` gives each axis a whole ring", {
                "tensor": 4,
                "fsdp": 4,
                "expert": 4
            }, True),
            (kb, "`tensor=8, fsdp=4, expert=2` does not fit", {
                "tensor": 8,
                "fsdp": 4,
                "expert": 2
            }, False),
            (skill, "`buildable: true` with `fits_without_split: false`", {
                "fsdp": 8,
                "tensor": 4,
                "expert": 2
            }, False),
        ):
            with self.subTest(mesh=mesh):
                self.assertIn(quoted, doc)
                fit = mesh_topology.check_mesh_topology_fit(mesh,
                                                            "4x4x4",
                                                            cores_per_chip=1)
                self.assertTrue(fit["buildable"])
                self.assertEqual(fit["fits_without_split"], fits)


if __name__ == "__main__":
    unittest.main()
