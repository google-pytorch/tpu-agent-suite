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
"""Keeps tpu-kernel-development's design knowledge free of past numbers.

The design references, the strategy rules, the method, the hardware facts,
the code style, the hardware probe catalog and the probe suite are read by
agents as rules. A measured number copied into them is a number from one
machine and one toolchain, quoted as if it held everywhere; a toolchain
version pinned in them goes stale on the next upgrade. Numbers come from
running a hardware probe on the target, never from these files.

The catalog is also checked against the probe registry, since a plan cites
a probe by the name the catalog gives it.
"""

import contextlib
import io
import json
import pathlib
import re
import sys
import unittest

PACKAGE_DIR = pathlib.Path(__file__).resolve().parents[1]
REFERENCES = PACKAGE_DIR / "references" / "tpu-kernel-development"
PROBES_DIR = PACKAGE_DIR / "scripts" / "tpu-kernel-development" / "hw_probes"
# The strategy rules live in the knowledge base, as `file` rows: served by
# kb_lookup like any page, but kept out of full-text ranking, where their
# long bodies outranked existing answers.
STRATEGIES = (PACKAGE_DIR / "data" / "tpu-knowledge-base" / "pages" /
              "kernel-strategies")
STRATEGY_ROWS = (PACKAGE_DIR / "data" / "tpu-knowledge-base" / "index" /
                 "kernel-strategies.jsonl")

TARGETS = [
    REFERENCES / "design",
    STRATEGIES,
    REFERENCES / "tpu_hardware_facts.md",
    REFERENCES / "kernel_design_method.md",
    REFERENCES / "hw_probe_catalog.md",
    REFERENCES / "pallas_code_style.md",
    PROBES_DIR,
]

RULES = [
    ("toolchain version pin",
     re.compile(r"\b(?:jax|jaxlib|libtpu)[ =/@-]+v?\d+\.\d+(?:\.\d+)*", re.I)),
    ("past job id", re.compile(r"\bj-[0-9a-f]{8}\b")),
    ("evidence tag", re.compile(r"\[(?:measured|verified):", re.I)),
]

# A catalog row: | `probe_name` [**(destructive)**] | question | informs |
_CATALOG_ROW = re.compile(r"^\| `([a-z0-9_]+)`( \*\*\(destructive\)\*\*)? \|",
                          re.MULTILINE)


def _files() -> list[pathlib.Path]:
    found = []
    for target in TARGETS:
        if target.is_file():
            found.append(target)
        elif target.is_dir():
            found += sorted(p for p in target.rglob("*")
                            if p.suffix in (".md", ".py") and p.is_file())
    return found


def _registry():
    sys.path.insert(0, str(PROBES_DIR))
    try:
        import harness  # pylint: disable=import-outside-toplevel
        import run_probes  # pylint: disable=import-outside-toplevel
        with contextlib.redirect_stdout(io.StringIO()):
            run_probes._import_all_check_modules()  # pylint: disable=protected-access
        return list(harness.registered_probes())
    finally:
        sys.path.remove(str(PROBES_DIR))


class DesignKnowledgeTest(unittest.TestCase):

    def test_targets_exist(self):
        for target in TARGETS:
            self.assertTrue(target.exists(), target)
        self.assertGreater(len(_files()), 50)

    def test_no_pins_job_ids_or_evidence_tags(self):
        violations = []
        for path in _files():
            text = path.read_text(encoding="utf-8")
            for number, line in enumerate(text.splitlines(), 1):
                for label, pattern in RULES:
                    if pattern.search(line):
                        rel = path.relative_to(PACKAGE_DIR)
                        violations.append(
                            f"{rel}:{number}: {label}: {line.strip()[:120]}")
        self.assertFalse(violations, "\n".join(violations))

    def test_catalog_lists_every_registered_probe_and_nothing_else(self):
        catalog = (REFERENCES /
                   "hw_probe_catalog.md").read_text(encoding="utf-8")
        rows = {
            m.group(1): bool(m.group(2))
            for m in _CATALOG_ROW.finditer(catalog)
        }
        probes = {p.name: p.destructive for p in _registry()}
        self.assertEqual(sorted(set(probes) - set(rows)), [],
                         "registered probes missing from the catalog")
        self.assertEqual(sorted(set(rows) - set(probes)), [],
                         "catalog rows naming no registered probe")
        wrong = sorted(n for n in probes if probes[n] != rows[n])
        self.assertEqual(wrong, [],
                         "destructive marks that disagree with the registry")

    def test_every_cited_probe_is_registered(self):
        """`probe `x``, `verify: probe x` and `--pick x` name real probes."""
        names = {p.name for p in _registry()}
        patterns = [
            re.compile(r"\bprobes? `([a-z0-9_]+)`"),
            re.compile(r"^verify: probe ([a-z0-9_]+)", re.MULTILINE),
            re.compile(r"--pick ([a-z0-9_]+)"),
        ]
        unknown = []
        for path in _files() + sorted(
            (PACKAGE_DIR / "references" / "tpu-kernel-development" /
             "kernel-worker").glob("*.md")):
            if path.suffix != ".md":
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in patterns:
                unknown += [
                    f"{path.relative_to(PACKAGE_DIR)}: {m.group(1)}"
                    for m in pattern.finditer(text) if m.group(1) not in names
                ]
        self.assertEqual(unknown, [])

    def test_every_strategy_page_has_one_row_and_one_index_line(self):
        rows = [
            json.loads(line)
            for line in STRATEGY_ROWS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_file = {pathlib.Path(r["file"]).name: r for r in rows}
        pages = sorted(p.name for p in STRATEGIES.glob("STRAT-*.md"))
        self.assertEqual(len(pages), 28)
        self.assertEqual(sorted(set(by_file) - {"index.md"}), pages)
        self.assertEqual([r["id"] for r in rows if "page" in r], [],
                         "a `page` row is ranked by its full text")
        index = (STRATEGIES / "index.md").read_text(encoding="utf-8")
        for name in pages:
            with self.subTest(page=name):
                key = re.match(r"(STRAT-[A-Z]+-\d+)", name).group(1)
                self.assertIn(key.lower(), by_file[name]["keywords"])
                self.assertIn(f"]({name})", index)
                body = (STRATEGIES / name).read_text(encoding="utf-8")
                self.assertNotIn(
                    "{{", body, "templating does not run on "
                    "knowledge-base pages")

    def test_no_reference_points_at_the_old_strategies_dir(self):
        stale = []
        for path in sorted((PACKAGE_DIR / "references" /
                            "tpu-kernel-development").rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            if "tpu-kernel-development/strategies" in text:
                stale.append(str(path.relative_to(PACKAGE_DIR)))
        self.assertEqual(stale, [])

    def test_every_section_8_definition_requires_the_rules_consulted(self):
        """The plan, the method and the completeness check list one part set.

        The consulted-rules line is the record of which strategy rules a
        plan used; a part one of the three omits is not checked.
        """
        places = {
            "plan-kernel.md": REFERENCES / "kernel-worker" / "plan-kernel.md",
            "kernel_design_method.md": REFERENCES / "kernel_design_method.md",
            "kernel-worker.md": PACKAGE_DIR / "agents" / "kernel-worker.md",
        }
        for name, path in places.items():
            with self.subTest(file=name):
                text = " ".join(path.read_text(encoding="utf-8").split())
                self.assertIn("trategy rules consulted", text)
        worker = " ".join(
            (PACKAGE_DIR / "agents" /
             "kernel-worker.md").read_text(encoding="utf-8").split())
        self.assertIn("all six required parts", worker)


if __name__ == "__main__":
    unittest.main()
