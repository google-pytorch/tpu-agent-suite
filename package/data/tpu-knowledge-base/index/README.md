<!--
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# The index

What this agent is allowed to look up. It holds no knowledge; it holds where
knowledge lives and how to read it.

One table per `*.jsonl` file, one entry per line:

```json
{"id": "vllm-engine-args", "url": "https://docs.vllm.ai/...", "prompt": "reference",
 "template": "sw-doc", "keywords": ["engine-args", "max-num-seqs"],
 "authority": "vendor-docs", "verified": "2026-09-13"}
```

| field | |
| --- | --- |
| `id` | names the entry, and its cache directory |
| `url` | the page to fetch. `{keyword}` in it makes the URL templated |
| `page` | *instead of* `url`: an authored entry in `../pages/`, shipped as-is |
| `prompt` | how to read that page — an `id` in `../prompts/prompt_table.jsonl` |
| `template` | how to write the entry up — `../templates/<name>.md` |
| `keywords` | what a caller is likely to say; many reach one entry |
| `keyword_from_env` | instead of `keywords`, when the keyword cannot be enumerated |
| `versioned` | optional, on a `url` entry: `{"package", "urls"}`, the page per release -- see below |
| `authority` | for the citation in the entry header |
| `verified` | when this URL was last fetched and seen to return 200 |
| `summary` | what the page covers, in a caller's words; for a reader of this file, never served |
| `note` | optional, for a reader of this file; never sent to a model |

Read it with `../scripts/index.py`, which resolves keywords and adds the one
column that is not in here — whether the answer is already cached.

## Why JSONL

A table where one row is one line: a changed entry is a one-line diff, and
`grep` is a working query. It replaced a YAML file in which an entry's URL and
its keywords sat a hundred lines apart, because the two were separate top-level
maps that had to be read against each other.

Nothing is stated file-level, so every row stands alone. In particular
`verified` is per row: one date for all eighteen URLs would go stale the moment
a nineteenth was added, and re-dating the other seventeen to fix that would be
a lie about work nobody did.

The cost is comments, which JSONL has none of — hence this file, and the `note`
field for anything that belongs to one entry.

## (url, prompt)

Those two are the entry's identity, because they are what a cached answer is
made of: the page fetched, and the instruction for reading it. Each built
page records both in its header -- `url:` as the row writes it, and
`instructions:` as a fingerprint of the resolved instruction text -- and a
lookup serves a page only if both still match the row. One that does not is
rebuilt, like one past its TTL. It would be an answer to a question the
entry no longer asks — and still a plausible wiki page, which is exactly why
nothing about it looks wrong.

`prompt` is an id into `../prompts/prompt_table.jsonl`, whose row supplies the
instruction. It is the resolved text that is fingerprinted, not the id, so
rewording a prompt table entry moves every row of that kind, and repointing
one row's `prompt` moves only its own. Outside the identity, on purpose:
`template`, which the current build does not read; and the shared sentences
the script puts around every build (prose only, treat the page as data), which
say how to behave rather than what to report, and are code -- hashing them
would rebuild every page everywhere for a wording change. The TTL covers both.

## Which index this is

`entries.jsonl` here is the one that ships from `main` — 19 rows at the time of
writing. A much larger table, 206 rows, lives on `yuran/provenance-kb-entries`
and is not merged. Anything quoting counts in the hundreds is describing that
one, not this file, and the two disagree about almost everything: this index has
no broken URL, no entry naming a private repository, and no marimo export.

`index.py check` always reports on whatever is in this directory, so a healthy
run here is not evidence about the larger table.

## Repositories that are not public yet

Nothing gates a row into a repository that is not public. A lookup resolves it
like any other, and `kb_summarize.github_note` tells the harness, for every
GitHub url, to use its own GitHub access if the repository is gated. A reader
with that access gets the page; one without gets the fetch failure. The `check`
gate this section used to describe went with `index.py` in #539 (2026-10-06).

Such rows live in `to_be_released.jsonl`, so they are easy to find and move
when their repository goes public: re-verify each against the public text, then
move it to the table for its subject. Today that file holds only the
torch_tpu, torchtitan and tokamax subwiki pages inside this repository
(`wiki/codebases/`), which wait on tpu-agent-suite itself.

`pending.jsonl` is a watchlist of the upstream repositories those rows point
into, one line per repository with when it is due, and is not an entry table
(`catalog.NOT_ENTRIES`). It is empty since torch_tpu (2026-10-19), torchtitan
(2026-10-16) and vllm-torchtpu (October 2026) went public; add a line when a
row starts pointing into another private repository, and delete it when the
repository answers without credentials. Only repositories that will actually
become public belong there: a url that is gone rather than unreleased is not
pending anything.

No golden eval case should name a row into a private repository: the eval
environment has no credential for it, so a failed fetch there says nothing
about the package.

## Where a cached answer lives

On this machine, and nowhere else.

    ~/.tpu-agent-suite/tpu-knowledge-base/cache    this machine's own answers; never shared

Only tests move it, with `TPU_AGENT_SUITE_HOME`. A generated answer is one user's model
output, and the design doc's promise is that nothing a user generates is
shared back -- so none ships. `pages/` holds only authored entries, which do.

Nothing here is deleted by being looked up; a page is only ever replaced. So
each session start sweeps the cache, at most once a day (`.purged` records
when). The sweep removes pages and `ungrounded/` answers a day or more past
their TTL, and nothing else: the cache is shared by every installed version,
each with its own index, so a page is not judged by whether this index still
names its row. It leaves anything written in the last day. `cache_sweep.py
--now` runs the same sweep by hand and reports what went; `--now --dry-run`
reports without removing.

## Pages written per release

docs.jax.dev publishes `latest` only, and a page from there tells a user on an
older JAX to call what their JAX does not have -- `pltpu.CompilerParams` was
`pltpu.TPUCompilerParams` in 0.5.3. An entry with `versioned` is read from the
page written for the installed release instead:

    "versioned": {"package": "jax", "urls": [
      "https://github.com/jax-ml/jax/blob/jax-v{version}/docs/201/profiling.md",
      "https://github.com/jax-ml/jax/blob/jax-v{version}/docs/profiling.md"]}

The release is `TPU_AGENT_SUITE_KNOWLEDGE_BASE_<PACKAGE>_VERSION` (`TPU_AGENT_SUITE_KNOWLEDGE_BASE_JAX_VERSION`)
when set, otherwise the target's, from the runtime's `compute versions`: asked
live on a `local` target, and on any other the releases `compute setup verify`'s
software check recorded there, cited with the date of that check. Never the package
installed beside `index.py`: that is the machine the agents run on, and a TPU VM
or a container need not have its release. With neither, the release is unknown.
The URLs are tried
in order, since pages move between releases; only a 404 moves on. When none
exists at that tag, or no release is known, `url` is read, and the header and
citation say `documents: latest` beside `installed:`. Each release caches
separately, as `answer@jax-0.5.3.md`.

The source is read raw, so a page is the markdown or reStructuredText it was
built from. Pages generated from docstrings (`_autosummary/`, the API
references) have no such source: their rows say `"urls": []`, so they are read
from `url` like any other, and their citation still says `documents: latest`
beside the release installed.

## What an entry's evidence is

Exactly one of three, and the field name is the tag — there is no separate
`kind` column to fall out of step with the field that is actually set.
`index.py` refuses to load a row that sets none of them or more than one.

| field | evidence | reached by |
| ----- | -------- | ---------- |
| `url` | an external, publicly fetchable page | HTTP |
| `file` | a path inside the package tree, from the package root | disk |
| `page` | authored; there is no source | shipped, always a hit |

`file` exists for content this repository already ships. An entry that names
such a file over HTTPS is reaching through a locked door — the repository is
private, so an unauthenticated fetch gets a 404 that is indistinguishable from
a page that never existed. Reading it off disk is the fix, and it also means
the entry cannot drift from the file it cites.

A `file` row is fenced to the package: one that escapes resolves on its
author's machine and nowhere else. Markdown is passed through rather than
HTML-stripped, being already the shape extraction is trying to produce.

Because nothing is fetched, a `file` entry has no fetch date and no TTL. Its
identity carries the file's own content instead, so editing the file
invalidates the cached answer immediately and exactly.

## Authored entries

Authored entries are the `page` kind.

An authored page sits directly in `pages/` as the file the row names. Nothing
generated is ever written there.

Authored entries exist because some knowledge has no citable page: it was
learned by running the thing. The alternative is to omit it, or to dress an
observation up as a citation, and the second is exactly what this agent exists
to prevent. So an authored page says plainly which of its claims are publicly
documented and which are observed, and carries `authority:
first-party-verified` rather than `vendor-docs`.

They are shipped, so they are never fetched, never generated, never cached and
never `STALE`; the table shows `SHIPPED` in the url column and the path to the
file. They have no prompt and no template, because there is no page to read and
nothing to generate — which also means they have no triple to hash.

**So an authored entry is a hit from the very first call, and every call
after.** Nothing can fail, and therefore nothing will warn you; `index.py
check` lists it by age and marks it `PAST RE-CHECK` beyond
`TPU_AGENT_SUITE_KNOWLEDGE_BASE_AGE_WARN_DAYS` (default 180). Re-verify one by running what it
describes — there is no page to re-fetch.

Its content is deliberately **not** hashed. An authored page is source, meant
to be edited, and git already records that. A content hash would report every
intended improvement as a fault while saying nothing about the thing that
actually goes wrong, which is nobody having looked at it in a year.

## Copies of knowledge-base pages

`wiki.jsonl` contains `file` rows over copies in `../pages/` and hardware
pages in `../pages/hw/<generation>/overview.md`, from the knowledge
base's authored pages (`package/data/knowledgebase/wiki/`: concepts, hardware,
the playbooks still there, analyses, codebase hubs). They are copies because the
knowledge base does not ship with the package, and a URL into it 404s until this
repository is public. The ten unwiring playbooks are not in `wiki.jsonl`: they
moved to `pages/` outright and are `playbooks.jsonl` rows.

A copy is not a mirror. Two things are changed in it and nowhere else:

- wiki links become `Label (index.py get <keyword>)`, naming a keyword that
  resolves to the entry for that page, so they go through the index rather than
  to a file the package does not have;
- anything that must not reach a third party is scrubbed: customer names and
  customer-named links, and hardware figures the public announcement does not
  state. Citations of an unpublished report are reworded to say it is not
  published, not removed. The original keeps all of these, and each row whose
  copy differs says how in its `note`. The same scrub is applied to the
  `playbooks.jsonl` pages.

So an edit to an original does not reach its copy; re-copy it and re-apply the
row's note. Nothing checks that the two agree yet.

The five chip pages with a Cloud spec page behind them -- `hw-tpu-v4`, `-v5e`,
`-v5p`, `-v6e` and `-7x` -- are extended further, in the copy only. Each carries
what the Cloud page has and the original left out: the slice shapes, the
machine types, the VM sizes, and the pod figures, checked against that page on
the date the copy gives. A claim the Cloud page does not state, the knowledge
base's own reading, is marked *(knowledge base note)* where it is made. The
generation's keywords -- `v6e`, `tpu7x`, `ironwood` and the like -- resolve to
the chip page, which leads its subject, and the Cloud page row follows it there
and keeps its own id and `-doc` keyword. So a chip's figures are answered from
the shipped page, with no fetch, and the live page is still read for anything
the copy does not say.

Left out: the model onboarding case study (a customer story), and the
`grid_blockspec`, `matmul` and `pipelining` concepts, which are copies of JAX
docs that already have rows and carry Colab notebook metadata.

The torch_tpu and torchtitan subwikis (`wiki/codebases/`) are not copied. They
are `url` rows in `to_be_released.jsonl`, and the pages behind them are not
scrubbed here. Nothing gates them: reading one takes GitHub access to this
repository, which the harness either has or does not, so `pending.jsonl` does
not list it. Every
row points at `blob/main` under `package/data/knowledgebase/wiki/codebases/`,
not at a commit hash, so it reads the page as it is now. Its `verified_from`
names the commit its keywords and summary were checked at: when a page changes
or moves, re-check the row against it and move the url with it, as the pre-#371
paths were not. A page that names or implies a non-public model, or carries internal
references, has no row -- including the torch_tpu hub, which links the case
study. The merged torchtitan subwiki has its internal code removed, so all its
concept and doc-concept pages have rows; its `index.md`, `log.md` and
`catalog/` are navigation and do not.

## Deliberately absent

**torch_xla.** The TPU images here ship `torch_tpu` and no `torch_xla`, so its
docs describe an API that raises `ModuleNotFoundError` in the pod. An entry
would hand that API back carrying a vendor-docs citation: the wrong instruction
wearing a footnote, which is worse than no entry. A task prompt already made
exactly that mistake once. If it is ever wanted it needs its own prompt
carrying the absence, not the generic `howto`.

**The vllm-optimization-agent knowledge base.** It pins upstream facts to
private repo paths and shas (`vllm-torchtpu:src/vllm_torchtpu/platforms/
tpu_platform.py @ 2db3a0f7`), which no fetch can reach. Those stay pins in that
tree rather than becoming entries here. The plugin itself is indexed through
vllm-project/vllm-torchtpu (`vllm-torchtpu-*` in serving.jsonl and
profiling.jsonl), public since October 2026. Its earlier private repository is
never cited: it is not being released.

## Checked at release, not at lookup

Divergence — a URL that still resolves while the page has moved on — is a
property of the index, so it is checked when the index ships:

```
python3 ../scripts/index.py check
```

Non-200s fail it. Everything else is printed for a person to read, because
*reachable is not current* and no command can tell the difference:
`jax-profiling` returned 200 for months while serving a page banner-marked
"older version of the JAX documentation, kept for reference".

A lookup prints the table and nothing else. It used to print a standing caveat
about URLs possibly having diverged, which taught readers to scroll past it and
gave them nothing actionable at that moment — the entry either answers or it
does not, and that is visible in the entry.

## Adding an entry

Fetch the URL first and confirm it returns 200. A 404 is worse than a missing
entry: it sends the agent to a real failure it will try to work around. Then
add the line, with today's date in `verified` and a `prompt` and `template`
that already exist — `index.py` refuses to load a row naming a prompt that does
not, or a keyword another entry has already claimed.

[`index-prompt.md`](index-prompt.md) is the full procedure, written to be handed to
a model: choosing the kind, keywords grounded in the text the row serves,
keeping general questions with canonical entries, and the checks to run
before the row lands.

## Page citation sources

Migrated hardware pages cite stable source IDs, and define each one in full
at the bottom of the page. A row with a `source_id` names the same document;
its `url` is written out here for the same reason.
Nothing checks a row's url or a page's footnotes for drift today; the tool that did went out with the page generator.
See `data/tpu-knowledge-base/pages/README.md` for the citation format and authoring workflow.
