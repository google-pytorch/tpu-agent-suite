# Hardware knowledge

These are the shipped hardware pages read by TPU provenance. Each generation
has its own `overview.md`; lookup IDs and aliases remain in
`../../tpu-knowledge-base/index/wiki.jsonl`. The index reads these files directly
using package-relative paths. No new loader or generated-page cache is needed.

| Generation | Page | Source ID | Evidence type |
|---|---|---|
| v4 | [Overview](v4/overview.md) | `cloud-tpu-v4` | Specification |
| v5e | [Overview](v5e/overview.md) | `cloud-tpu-v5e` | Specification |
| v5p | [Overview](v5p/overview.md) | `cloud-tpu-v5p` | Specification |
| v6e / Trillium | [Overview](v6e/overview.md) | `cloud-tpu-v6e` | Specification |
| v7x / Ironwood | [Overview](v7x/overview.md) | `cloud-tpu-7x` | Specification |
| 8t | [Overview](8t/overview.md) | `tpu-8th-gen` | Announced vendor claims |
| 8i | [Overview](8i/overview.md) | `tpu-8th-gen` | Announced vendor claims |

Pages cite stable source IDs in footnotes and define each one in full at
the bottom of the page. See [source authoring instructions](../README.md).

## Grounding

Official GCP specifications ground the deployed generations; 8t/8i use the Google
announcement and explicitly distinguish vendor claims from unspecified details.

The sibling `overview.recipe.json` holds the query each page answers. Pages
were reviewed by the coding agent against the fetched evidence; human review
remains pending. v4 is retained unchanged.
The v5p and v6e pages also cite GCP's TPU7x comparison table where their own
specification pages omit a figure.

The v6e and v7x overviews are pinned: each carries a "Roofline inputs" table,
and `critical_fields` in its header names the rows that must stay stated.
`pages_test.py` fails a regenerated or edited page that leaves one
`not stated`, and `kb_summarize` refuses to cache a summary of the
`cloud-tpu-v6e` or `cloud-tpu-7x` url that does.

| Generation | Replay query |
|---|---|
| v5e | [Recipe](v5e/overview.recipe.json) |
| v5p | [Recipe](v5p/overview.recipe.json) |
| v6e | [Recipe](v6e/overview.recipe.json) |
| v7x | [Recipe](v7x/overview.recipe.json) |
| 8t | [Recipe](8t/overview.recipe.json) |
| 8i | [Recipe](8i/overview.recipe.json) |

For compiler layout concepts, use source ID `openxla-tiled-layout`. It explains
tiling and does not establish generation-specific chip figures. The corresponding
lookup entry uses the same catalog source. Older hardware concept pages remain
in `data/tpu-knowledge-base/pages/` for this first migration.

The older authoring copies in `data/knowledgebase/wiki/hardware/` remain
separate. These shipped pages previously lived in `data/tpu-knowledge-base/pages/`
and include later specification additions. Editing an authoring copy does not
update the shipped page automatically. Edit these pages for shipped provenance;
consolidating the authoring tree is a separate migration.

## Memory and profiling reference

[Memory and buffer sizing](concepts/memory-buffers.md) adds scoped VMEM limits,
SparseCore buffer sizing, padding and XProf inspection. This extension used
GitHub file pages only, following OpenXLA `main` and XProf `master`.
The generation overviews link to their detailed memory pages and include scoped limits where established.

## Generation-specific memory pages

| Generation | Memory and buffers | Replay query |
|---|---|---|
| v5e | [Page](v5e/memory-buffers.md) | [Recipe](v5e/memory-buffers.recipe.json) |
| v5p | [Page](v5p/memory-buffers.md) | [Recipe](v5p/memory-buffers.recipe.json) |
| v6e | [Page](v6e/memory-buffers.md) | [Recipe](v6e/memory-buffers.recipe.json) |
| v7x | [Page](v7x/memory-buffers.md) | [Recipe](v7x/memory-buffers.recipe.json) |
