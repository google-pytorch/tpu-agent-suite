---
references:
  - references/code-reference/MAP.md
  - references/write-tpu-kernels/
---

# Code reference

Follow this for every *upstream lookup*: a worked example of a TPU Pallas pattern, the unit test that
would prove such a kernel correct, the file layout to copy for a new kernel and its test, or a check
of a pinned tokamax or jax citation in the wiki.

It makes two upstream repositories, `jax-ml/jax` and `openxla/tokamax`, reachable as a *lookup
table*, not as reading. `<jax>/tests/pallas/` is the largest body of TPU Pallas unit tests that
exists, and tokamax's per-op directories are the best template for how a kernel and its test are
laid out together. Both are large: the `jax` checkout is ~76 MB and its TPU test files run to
thousands of lines each. You carry `path:line-line@sha` citations forward, never pasted source.

## Paths used on this page

- `$ROOT`: `{{runs}}/<run_id>`, the run's work directory. Every command on this page and in
  {{reference:code-reference/MAP.md}} runs from `$ROOT`, so the commands name the clones
  `code-reference/jax` and `code-reference/tokamax`.
- `<code-reference>`: `$ROOT/code-reference/`, which holds both clones and the manifest. Every
  campaign in the run shares it.
- `<jax>`: `<code-reference>/jax`, the clone of `jax-ml/jax`.
- `<tokamax>`: `<code-reference>/tokamax`, the clone of `openxla/tokamax`.

Clone only here, with your own shell. Never clone into the package tree or any git checkout: a
nested repo there is one `git add` away from being committed.

## Decide which path to take

Take the **first** branch that matches, per repo; the two are independent:

| condition | go to |
| --- | --- |
| Both clones exist and pass *Verify* | *Answering a kernel question*. Do not fetch, do not stat the remote. |
| `<code-reference>/<repo>/` is **missing** | *Fetch* |
| Present but **broken** (no `.git`, an interrupted checkout, `git status` errors) | Note what was wrong in `STRATEGY.md`, `rm -rf` that one clone (this run made it), and *Fetch* it once. |

A clone is never updated within a run. It is the pin your citations are written against; moving it
invalidates every `path:line@sha` already in your notes.

## Preconditions

Check these before fetching. A failure means no upstream lookup this run, not a stop: record it in
`STRATEGY.md`, say "upstream unavailable: <error text>" wherever a citation would go, and carry the
affected API usage as `[assumed]`.

| probe | if it fails |
| --- | --- |
| `git --version` — need ≥ 2.19 for `--filter` | Fall back to a plain `--depth 1` clone and record in the manifest that arbitrary-pin resolution is **unavailable**. Pinned tokamax citations then cannot be checked. |
| `git ls-remote --heads https://github.com/openxla/tokamax.git main` (and the same for `jax-ml/jax`) | HTTPS to GitHub is down or proxied. Record the actual error text. Do not retry-loop. |
| `df -h "$ROOT"` — need ≥ 1 GB free | Record the free space. The failure mode is a half-written clone. |

## Fetch

Two different commands, on purpose. `--filter=blob:none` drops *history* blobs, not the working
tree, so dropping `--depth 1` costs only commits and trees:

```bash
cd "$ROOT" && mkdir -p code-reference && cd code-reference

# tokamax — blobless, FULL history. ~1240 commits; measured 14 MB on disk, ~4 s.
git clone --filter=blob:none https://github.com/openxla/tokamax.git

# jax — blobless, depth 1. 187 MB of upstream git history nobody here needs.
git clone --depth 1 --filter=blob:none https://github.com/jax-ml/jax.git
```

Measured 2026-09-10: tokamax 14 MB in 4 s, jax 76 MB in 16 s. A result far off those is a wrong
clone; check the filter was accepted before you trust the tree.

**Do not "fix" tokamax to `--depth 1`.** Full history is what makes `git show 9da2369:<path>`
resolve, the only way to check the pinned `[source: tokamax/…@9da2369]` citations in
{{reference:write-tpu-kernels/}}. jax is depth-1 because its pinned citations are tag-shaped
(`@jax 0.10.2`), and a tag is one cheap fetch away (*Deepen onto a pinned sha or tag*).

Then run *Verify* and write the manifest.

## Verify

`git clone` can exit 0 having produced a repo that does not do what you need, so never trust the
exit code alone:

```bash
cd "$ROOT"
test -f code-reference/jax/tests/pallas/tpu_pallas_test.py
test -d code-reference/tokamax/tokamax/_src/ops/ragged_dot
git -C code-reference/tokamax cat-file -e 9da23691ea784df734946dab03587949477e8573^{commit}
git -C code-reference/jax     log -1 --format='%H %cI %s'
git -C code-reference/tokamax log -1 --format='%H %cI %s'
du -sh code-reference/jax code-reference/tokamax
```

The `cat-file -e` is the one that matters: it distinguishes a working tokamax clone from a
plausible one. If it fails, the clone cannot resolve pinned citations; record that in the manifest.

## The manifest

Write `<code-reference>/MANIFEST.md` **beside** the clones, never inside one. One block per repo:

```markdown
## jax
url:          https://github.com/jax-ml/jax.git
clone mode:   depth-1 blobless          # or: blobless full | depth-1 plain
sha:          <full sha>
committed:    <%cI>
subject:      <%s>
fetched:      <YYYY-MM-DD>
size:         <du -sh>
resolves an arbitrary pinned sha:  no — fetch --depth 1 origin <sha> first
```

Copy both shas into `STRATEGY.md` as well; a citation without the sha it was read at drifts.

## Deepen onto a pinned sha or tag

Use this when checking a `[source: …@<sha>]` or `[source: …@jax 0.10.2]` citation, not for browsing.
GitHub serves arbitrary reachable SHAs and tags.

```bash
# jax, onto a release tag, without moving the working tree
git -C code-reference/jax fetch --depth 1 origin refs/tags/jax-v0.10.2:refs/tags/jax-v0.10.2
git -C code-reference/jax show jax-v0.10.2:jax/_src/pallas/mosaic/tpu_info.py | sed -n '1,60p'

# jax, onto an arbitrary sha
git -C code-reference/jax fetch --depth 1 origin <sha>
git -C code-reference/jax show FETCH_HEAD:<path>
```

tokamax needs none of this: it has full history, so `git show 9da2369:<path>` works directly.

For more than two or three files at a pin, `git worktree add` at that pin rather than checking the
main tree out backwards; a clone left on a detached old commit makes later citations wrong.

## Answering a kernel question

1. **Classify** the question into one row of {{reference:code-reference/MAP.md}}. The highest-traffic
   rows are inlined below, so the common case costs no second file read.
2. **Grep the named subtree first, then Read a window.** Never Read one of these files whole. Grep
   for the symbol or the test name, then Read ±40 lines around the hit.
3. **Budget: at most 5 files opened per question.** If you need more, the question was several
   questions; split it.
4. **If no row fits**, record the nearest row and a proposed new row in your campaign's `wiki/`
   entries, so it is promoted with the rest.
5. Record the answer as the block in *What to record*.

| Question | Go to |
| --- | --- |
| A first Pallas TPU kernel, any shape | `<jax>/jax/experimental/pallas/ops/tpu/example_kernel.py`, `<jax>/docs/pallas/quickstart.md` |
| Mosaic TPU semantics — DMA, semaphores, scratch, `pltpu` primitives | `<jax>/tests/pallas/tpu_pallas_test.py` — grep it, never read it whole |
| Manual pipelining, `emit_pipeline`, overlap | `<jax>/tests/pallas/tpu_pallas_pipeline_test.py` and `tpu_pallas_async_test.py`; `<jax>/docs/pallas/pipelining.md` |
| A **SparseCore** Pallas kernel | `<jax>/docs/pallas/tpu/sparsecore.md` first — runnable worked examples; then `<jax>/tests/pallas/tpu_sparsecore_pallas_test.py`; API in `<jax>/jax/experimental/pallas/tpu_sc.py`. tokamax has none. Pair with §10 of {{reference:write-tpu-kernels/hardware-facts.md}} |
| **Grouped GEMM, MoE, ragged dot** | `<jax>/jax/experimental/pallas/ops/tpu/megablox/` + `<jax>/tests/pallas/tpu_gmm_test.py` + `<tokamax>/tokamax/_src/ops/ragged_dot/` |
| **Attention** — splash, flash | `<jax>/jax/experimental/pallas/ops/tpu/splash_attention/` and `flash_attention.py`; tests `tpu_splash_attention_kernel_test.py`, `tpu_splash_attention_kernel_sharded_test.py`, `tpu_splash_attention_mask_test.py`; `<tokamax>/tokamax/_src/ops/attention/` + `<tokamax>/docs/splash_attention.md` |
| **Paged / ragged-paged attention** — vLLM's decode kernel | `<jax>/jax/experimental/pallas/ops/tpu/paged_attention/` and `ragged_paged_attention/`; tests `tpu_paged_attention_kernel_test.py`, `tpu_ragged_paged_attention_test.py` |
| How to structure a new kernel **and its test** | the tokamax per-op layout — MAP.md § "The tokamax per-op file convention" |

## Never grep the clones repo-wide

- **Every search names a subtree.** Find the file first, then search inside it:

  ```bash
  grep -rn "<symbol>" code-reference/jax/tests/pallas/ --include='tpu_*_test.py' -l
  grep -n  "<symbol>" code-reference/jax/tests/pallas/<the one file>
  ```

- **Skip `gpu_*`, `mgpu_*`, `pallas_triton*`, `pallas_mosaic_gpu*`.** Roughly half of
  `<jax>/tests/pallas/` and most files in each tokamax op directory are GPU and lower through a
  different compiler. `<tokamax>/tokamax/_src/pallas/block.py` is Triton/GPU code, not TPU.
- **Citations, not source.** The unit of record is
  `` `jax/tests/pallas/tpu_gmm_test.py:412-448@<sha>` `` plus one line saying what is there, with
  the path relative to `<code-reference>`. Paste at most ~20 lines, and only when the exact text
  *is* the answer: a signature, a decorator, an argument order.
- **Upstream source is a rung on the evidence ladder, not the top of it.** It proves an API exists
  and a pattern compiles. It says nothing about what that pattern costs at your shapes on this
  hardware; it sits beside a `[verified]` fact measured at another shape, below a probe you ran
  this session.

## What to record

For each kernel question, add this block to `STRATEGY.md`:

```
question:   <restated in one line>
route:      <the MAP.md row used, or "no row — nearest is X, proposed row: …">
citations:  <path:lines@sha> — <what is there>        (1 to 5 of these)
pattern:    <3-8 lines: the shape of the upstream solution, in prose>
caveats:    <version skew, GPU-only siblings, anything you did not verify>
shas:       jax <sha> (<date>) · tokamax <sha> (<date>)
```

A mismatch between a pinned citation's lines and the excerpt in a wiki entry is a finding for your
campaign's `wiki/` entries; do not edit {{reference:write-tpu-kernels/}}, which is read-only.

## Limits

- The clones are checkouts to read, not environments to import from. Nothing here touches a venv.
- Do not run upstream tests. A test worth running becomes a probe, on a slot like any other job.
- Read only the clones under `<code-reference>`. Any other jax or tokamax checkout on the machine
  is not the one your citations resolve against.
