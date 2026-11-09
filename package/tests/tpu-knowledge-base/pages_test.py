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
"""What the shipped pages and the rows that reach them must still hold.

Ported from test_provenance_pages.py and test_provenance_table.py, which
were written against index.py and go with it. What they checked about the
chain -- prompts, templates, `assemble` -- goes too. What they checked
about the pages is still true and still worth failing on, so it is here,
against the catalog instead.
"""

import importlib.util
import pathlib
import re
import unittest

import yaml

_PACKAGE = pathlib.Path(__file__).resolve().parents[2]
_PAGES = _PACKAGE / "data" / "tpu-knowledge-base" / "pages"

_spec = importlib.util.spec_from_file_location(
    "kb_catalog", _PACKAGE / "scripts" / "tpu-knowledge-base" / "catalog.py")
catalog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(catalog)

# `preflight_check.discover_playbook_mappings()` on the knowledge base's
# playbooks, before they moved.
TARGET_IMPORTS = {
    "apex": "apex-to-native-normalization.md",
    "bicubic_interp": "tpu-tensor-layout-and-interp.md",
    "causal_conv1d": "mamba-ssm-unwiring.md",
    "cuda_api": "cuda-api-stripping-and-runtime.md",
    "cuda_extension": "custom-cuda-kernel-unwiring.md",
    "custom_cuda": "custom-cuda-kernel-unwiring.md",
    "deepspeed": "deepspeed-ops-unwiring.md",
    "dynamic_shapes": "playbook-static-shapes-and-tracing.md",
    "flash_attn": "flash-attention-to-sdpa.md",
    "flash_attn_interface": "flash-attention-to-sdpa.md",
    "gpt_oss_mxfp4": "vllm-gpt-oss-mxfp4.md",
    "groupnorm": "tpu-tensor-layout-and-interp.md",
    "hf_check_imports": "huggingface-runtime-compatibility.md",
    "mamba_ssm": "mamba-ssm-unwiring.md",
    "nccl_dist": "playbook-tpu-distributed-and-env-hygiene.md",
    "selective_scan_cuda": "mamba-ssm-unwiring.md",
    "serving_topology": "playbook-size-serving-topology.md",
    "sharding_config": "playbook-size-training-parallelism.md",
    "torch.ops": "custom-cuda-kernel-unwiring.md",
    "tpu_profiling": "playbook-profile-tpu-program.md",
    "triton": "triton-to-pytorch-ops.md",
    "xformers": "xformers-to-sdpa.md",
}


def target_imports(pages: pathlib.Path) -> dict[str, str]:
    """Import -> page name, read the way `preflight_check` reads it."""
    mappings = {}
    for page in sorted(pages.glob("*.md")):
        parts = page.read_text(encoding="utf-8").split("---", 2)
        if len(parts) < 3 or parts[0]:
            continue
        front = yaml.safe_load(parts[1])
        if isinstance(front, dict):
            for target in front.get("target_imports") or []:
                mappings[str(target).strip()] = page.name
    return mappings


class TableTest(unittest.TestCase):
    """Invariants of the rows themselves."""

    def setUp(self):
        self.entries = catalog.rows()

    def test_the_tables_load(self):
        self.assertTrue(self.entries)

    def test_every_keyword_resolves_to_its_own_row(self):
        """A keyword a row lists must reach that row and not a neighbour.

        Two rows claiming one keyword is a silent shadow: the second never
        answers, and the first answers questions meant for it.
        """
        for entry in self.entries:
            for keyword in entry.get("keywords") or []:
                with self.subTest(keyword=keyword):
                    found = catalog.find(self.entries, keyword)
                    self.assertIsNotNone(found, f"{keyword} resolves to none")
                    self.assertEqual(found[0]["id"], entry["id"])

    def test_every_entry_has_exactly_one_source(self):
        for entry in self.entries:
            present = [k for k in catalog.SOURCES if entry.get(k)]
            self.assertEqual(
                len(present), 1,
                f"{entry['id']} must have exactly one of url/file/page")

    def test_a_generation_no_row_lists_still_resolves(self):
        """The templated row, which is what makes next year's chip work."""
        found = catalog.find(self.entries, "v99e")
        self.assertIsNotNone(found)
        self.assertIn("{keyword}", found[0].get("url", ""))


class PageLinkTest(unittest.TestCase):
    """A page's `index.py get X` is how a reader follows its links."""

    def test_every_link_a_page_writes_resolves(self):
        entries = catalog.rows()
        # rglob, not glob: the pages under hw/ and cloud-tpu/ are the ones
        # most likely to carry a stale link.
        pages = sorted(_PAGES.rglob("*.md"))
        # A directory that is not there globs to nothing, and a loop over
        # nothing passes.
        self.assertTrue(pages, f"no pages under {_PAGES}")
        for page in pages:
            for n, line in enumerate(page.read_text().splitlines(), 1):
                for key in re.findall(r"index\.py get ([^\s`)<]+)", line):
                    with self.subTest(page=page.name, line=n):
                        self.assertIsNotNone(
                            catalog.find(entries, key),
                            f"{page.name}:{n}: '{key}' does not resolve")


def page_name(row: dict) -> str | None:
    """The name of the shipped page a row reads, if it reads one."""
    found = catalog.page_of(row)
    return found.name if found else None


class TargetImportsTest(unittest.TestCase):

    def test_the_pages_map_every_import_as_before(self):
        self.assertEqual(TARGET_IMPORTS, target_imports(_PAGES))

    def test_every_mapped_page_is_a_row(self):
        """A playbook reached by preflight_check but by no keyword could not
        be found by an agent that asked."""
        pages = {page_name(r) for r in catalog.rows()} - {None}
        for page in sorted(set(TARGET_IMPORTS.values())):
            with self.subTest(page=page):
                self.assertIn(page, pages)


class HardwarePagesTest(unittest.TestCase):

    def test_memory_questions_distinguish_static_and_runtime_sources(self):
        entries = catalog.rows()
        for keyword, expected in (("tpu-buffer-sizes", "hw-memory-buffers"),
                                  ("memory-profile", "xprof-memory-profile"),
                                  ("memory-viewer", "xprof-memory-viewer")):
            with self.subTest(keyword=keyword):
                found = catalog.find(entries, keyword)
                self.assertIsNotNone(found, f"{keyword} resolves to nothing")
                self.assertEqual(found[0]["id"], expected)

    def test_hardware_entries_read_the_moved_grounded_pages(self):
        entries = {row["id"]: row for row in catalog.rows()}
        for generation in ("v4", "v5e", "v5p", "v6e", "7x", "8t", "8i"):
            with self.subTest(generation=generation):
                row = entries[f"hw-tpu-{generation}"]
                folder = "v7x" if generation == "7x" else generation
                self.assertEqual(
                    row["file"],
                    f"data/tpu-knowledge-base/pages/hw/{folder}/overview.md")
                found = catalog.page_of(row)
                self.assertIsNotNone(found)
                text = found.read_text(encoding="utf-8")
                front = yaml.safe_load(text.split("---", 2)[1])
                self.assertTrue(front["source_ids"])
                # A page carries its own evidence: a declared id must be
                # defined on the page, and the definition must name a url.
                for source_id in front["source_ids"]:
                    cited = re.search(rf"^\[\^{re.escape(source_id)}\]:(.*)$",
                                      text, re.M)
                    self.assertIsNotNone(cited, f"{source_id} is not defined")
                    self.assertIn("](http", cited.group(1),
                                  f"{source_id} cites no url")
                self.assertFalse((_PAGES / f"hw-tpu-{generation}.md").exists())


if __name__ == "__main__":
    unittest.main()
