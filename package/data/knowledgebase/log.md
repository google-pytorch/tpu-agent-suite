# Knowledge Base Log

Append-only event log. Newest entries at the top.
Parse recent entries: `grep "^## \[" log.md | head -10`

## [2026-11-06] manual | update TorchTpu warning prefix in hardware-introspection

**Operation**: manual
**Pages created**: none
**Pages updated**: `wiki/codebases/torch_tpu/concepts/hardware-introspection.md` (`<TorchTpu/ApiStage/` -> `//TorchTpu/ApiStage/`)
**Notes**: Aligns warning filter prefix with `torch_tpu` `//TorchTpu/...//` warning category format.

## [2026-11-05] manual | removed stale torch_xla wording

**Operation**: manual
**Pages created**: none
**Pages updated**: `wiki/playbooks/static-shapes-and-tracing.md` (`mark_step` -> step synchronization barriers, tag `mark-step` -> `synchronization`), `wiki/codebases/xprof.md` (PyTorch/XLA capture line -> pointer to `torch_tpu`'s profiler), `wiki/codebases/torch-tpu.md` (dropped the `torch_xla_backend` alias), `index.md` (Static Shapes entry)
**Notes**: `torch_tpu` has no `mark_step`, and `torch_xla` is banned. `human_reviewed` reset to false on the playbook and on `xprof.md`, whose claims changed. The alias removal is metadata only, so `torch-tpu.md` stays `human_reviewed: false`. OpenXLA wording stays: `torch_tpu` compiles through it.

## [2026-11-03] manual | remove legacy private issue links for tpu-agent-suite public cutover

**Operation**: manual (human-directed: remove private issue links from shipped wiki pages and update `to_be_released.jsonl` URLs to `google-pytorch/tpu-agent-suite`)
**Pages created**: none
**Pages updated**: `wiki/kernel/pallas-validation-and-xprof.md`, `wiki/kernel/pallas-core-mental-model.md`, shipped copies in `tpu-knowledge-base/pages/` (`pallas-validation-and-xprof.md`, `pallas-core-mental-model.md`, `hardware_alignment_and_vmem.md`, `moe_serving_and_decode.md`, `vllm-tp-dp-plan.md`), `tpu-knowledge-base/index/README.md`, and `tpu-knowledge-base/index/to_be_released.jsonl`.
**Notes**: Citation scrub only; no technical claim changed, `human_reviewed` unchanged.

## [2026-10-27] ingest | tpu_info chip table; HBM capacities in GiB

**Operation**: ingest (code file, human-directed: state per-chip HBM in one unit, since onboarding now sizes from the knowledge base)
**Pages created**: `wiki/sources/2026-10-tpu-info-chip-table.md` (cites `tpu_info/device.py` by permalink at the pin; code is not snapshotted under `raw/`)
**Pages updated**: `wiki/hardware/tpu-v5e.md` (16 GB -> 16 GiB), `wiki/hardware/tpu-v6e.md` (32 GB -> 32 GiB), `wiki/hardware/tpu-7x.md` (96 GB -> 96 GiB per chiplet); per-chip HBM in GiB on `wiki/concepts/megacore.md`, `wiki/concepts/data-parallelism.md` (v5p 95 GiB, so `HBM / 10` is ~10 B parameters, not ~9 B), `wiki/concepts/prefill-vs-decode.md` (16 GiB / 8.2e11 B/s is ~21 ms; the book's ~19 ms takes 16 GB as 1.6e10 B), `wiki/playbooks/size-serving-topology.md` and `wiki/kernel/pallas-core-mental-model.md`; `index.md`
**Notes**: the Cloud TPU pages mix GB and GiB for HBM capacity; tpu_info's table names the unit (`hbm_gib`) and agrees with the GiB figures (v4 32, v5e 16, v5p 95, v6e 32, 7x 192 per chip). The shipped `tpu-knowledge-base/pages/hw/` overviews for v5e, v6e and v7x were corrected the same way. The Cloud TPU7x page now writes GiB throughout (updated 2026-10-22), so only the v5e and v6e pages still say GB. `human_reviewed` reset to false on `concepts/data-parallelism`, `concepts/prefill-vs-decode`, `playbooks/size-serving-topology`, `hardware/tpu-v5e`, `hardware/tpu-v6e`, `hardware/tpu-7x` (claims changed).

## [2026-10-16] ingest | torch_tpu re-ingested at release/0.1.2 (6d2a4575)

**Operation**: ingest (code repo via wikify-repo 0.4.1; human-directed: re-ingest torch_tpu at the head of `release/0.1.2`, keep the diff small where possible, ingest all project docs, and keep internal identifiers out)
**Pages created**: `wiki/sources/2026-10-torch-tpu-repo.md`; subwiki `wiki/codebases/torch_tpu/`: `concepts/mlir-optimization-passes.md`, `areas/csrc.md`, `changes/6d2a45754b.md`, and 35 new doc-concept pages from the user guide under `docs/markdown/` and the new README files
**Pages updated**: `config/torch_tpu.md` (pin, source URL, Bazel targets, index shards, planner excludes and every subsystem prefix remapped to the new `csrc/` + `src/torch_tpu/` layout); subwiki: 25 concept pages rebuilt in place, 2 relinked only (`dtype-and-precision`, `error-handling`), `overview.md`, 10 existing doc-concept pages (5 rewritten from new documents), `index.md`, `log.md`, and the regenerated symbol index (`catalog/`); hub `wiki/codebases/torch-tpu.md` (pin, counts, routing, sources); `index.md`
**Pages removed**: subwiki doc-concepts `deepseek-decomposition-tests`, `deepseek-v3-model-techniques`, `diffusers-smoke-tests`, `dynamic-shape-compilation`, `quality-benchmark-suite` (their source README files no longer exist and no current document covers them)
**Notes**: `release/0.1.2` shares no git history with the previous pin `4c0c7c57` (no common ancestor), so there is no commit count between them; the code changed substantially, including the move of C++ to `csrc/` and Python to `src/torch_tpu/`, so every citation anchor changed; wikify relinked 1,215 citations for 7,738 moved symbols and the rebuilt pages kept their still-accurate wording. Adversarial verification of about 485 load-bearing claims found 34 wrong; all corrected in place. Doc-versus-code mismatches found in the upstream docs are recorded on the doc-concept pages and the source page. The symbol index's doc column was scrubbed of internal bug-tracker IDs, short links, personal usernames and internal build names (16 rows) after `finalize`; a re-run of `finalize` regenerates the catalog, so the scrub must be repeated. The hub page's claims (pin, counts) changed, so its `human_reviewed` reset to false. After review on 2026-10-16 the overview's human `verified:` stamp was refreshed (the 09-17 stamp predated this rebuild), and the change page's `description:` was quoted so its front matter parses as YAML. The overview's compile-paths section now also names the third path: an FX graph traced with `make_fx` (as TorchTitan's graph trainer does) compiled directly by `StaticCompiler`, without Dynamo. Review follow-up: five pages whose subject is a Python API (torch-compile-backend, synchronization, profiler, scaled-dot-product-attention, eager-dynamic-shapes) had been seeded from the csrc/ half of their code only, so they cited no Python symbols; their config seeds now list the symbols each page cites plus the Python API symbols (TpuBackend, StaticCompiler, synchronize, profile, TpuProfilerConfig, splash_sdpa and others), the four with Python code were updated in place and re-verified (3 of 21 new claims corrected), and the two shipped reference copies were refreshed again. A re-run of `finalize` restored the view-decomposition description in the silo index; `wikify source-links` re-pointed the area page's entry-point links at source (finalize regenerates that block after it re-points citations); the catalog doc column was re-scrubbed (16 rows). The "4,037 commits" figure is withdrawn (the pins share no history); the change page now carries a note on its commit attribution and page bookkeeping; `dtype-and-precision` and `error-handling` carry `updated: 2026-10-13` for the claims verification corrected; the overview's third-compile-path paragraph no longer names `donated_inputs` (documented in the source as internal), and the overview was re-reviewed and re-stamped.
## [2026-10-15] ingest | torchtitan re-ingested at main (033f1313)

**Operation**: ingest (code repo via wikify-repo 0.4.1; human-directed: re-ingest google-pytorch/torchtitan at the head of `main`, keep the diff small where possible, ingest all project docs, and keep internal identifiers out)
**Pages created**: `wiki/sources/2026-10-torchtitan-repo.md`; subwiki `wiki/codebases/torchtitan/`: `changes/033f131322.md`, and 13 new doc-concept pages (`grain-data-pipeline`, `dataset-mixing-and-packing`, `nvfp4-training`, `flexshard-dist-muon`, `full-configurations-and-recipes`, `model-config-transforms`, `llama3-model`, `deepseek-v4-model`, `kimi-k2-7-model`, `kimi-k3-model`, `muse-glimmer-model`, `dapo-math-rl-recipe`, `verifiers-rl-integration`)
**Pages updated**: `config/torchtitan.md` (pin, source URL, three subsystem prefixes remapped to the new packages, `tensor-parallel` / `context-parallel` / `optimizer-and-lr-schedule` re-seeded, the six new model directories added to the planner excludes); subwiki: all 24 concept pages rebuilt in place, `overview.md`, 43 existing doc-concept pages, `index.md`, `log.md`, and the regenerated symbol index (`catalog/`); hub `wiki/codebases/torchtitan.md` (pin, counts, routing, sources); `index.md`; `package/data/tpu-knowledge-base/index/to_be_released.jsonl` (two rows for the removed pages dropped)
**Pages removed**: subwiki doc-concepts `forge-engine` (`experiments/forge/` was deleted upstream) and `tpu-model-parity-tests` (its per-model test README is gone; no current doc covers it)
**Notes**: 180 upstream commits since `63e334d5`, dominated by one rebase onto pytorch/torchtitan (flattened `[T]` activations, token-based batch sizing, `ShardingConfig` TP, CP as a config transform requiring `spmd_types`, packages for checkpointing, optimizers and data). The doc worklist grew from 48 to 68 documents and all were ingested. Adversarial verification checked 436 load-bearing claims plus a 32-claim re-check and found 17 wrong; all corrected in place. Doc-versus-code mismatches are recorded on the doc-concept pages and the source page. The symbol index's doc column needed no scrub at this pin; the generated change page was scrubbed of an upstream commit subject and file paths naming partner packages withheld from this wiki, and both checks must be repeated after any `finalize`. The change page's `description:` is quoted by hand so its front matter parses as YAML (wikify 0.4.1 writes it unquoted, with a colon in the text); a re-`finalize` undoes that too until wikify is fixed. The overview's front-matter reviewer stamp was removed because its claims changed; the hub's claims (pin, counts) changed, `human_reviewed` stays false.

## [2026-10-15] manual | rename wiki/MK/ to wiki/kernel/

**Operation**: manual (human-directed: drop the legacy `MK` directory name in favor of `kernel`)
**Pages created**: none
**Pages updated**: the 15 pages under `wiki/MK/` move unchanged to `wiki/kernel/`; links updated in `SCHEMA.md`, `index.md`, `wiki/kernel-optimization-index.md`, `wiki/kernel/briefs.md`, `wiki/codebases/tokamax.md`, `wiki/sources/2026-10-tokamax-repo.md`, and the `note` fields in `tpu-knowledge-base/index/pallas.jsonl`.
**Notes**: Path rename only; no claim changed, `human_reviewed` unchanged. Earlier entries in this log keep their `wiki/MK/` paths as history.

## [2026-10-14] manual | migrate torch.tpu runtime APIs to torch.accelerator

**Operation**: manual
**Pages created**: none
**Pages updated**: `wiki/codebases/torch_tpu/concepts/device-backend-registration.md`
**Notes**:
- Re-synced API mapping table from upstream `docs/markdown/6_reference_2_torchtpu_api.md`.
- Documented PyTorch 2.11+ unified `torch.accelerator` and PyTorch 2.13+ `torch.accelerator.random` API mappings with legacy `torch.tpu.*` fallback note.
- Updated tpu-knowledge-base pages to recommend canonical `torch.accelerator.*` over legacy `torch.tpu.*`.

## [2026-10-13] manual | pallas-kernel: See also points at pages that exist

**Operation**: manual (human-directed: fix the dead See-also links reported on `wiki/MK/pallas-kernel.md`)
**Pages created**: none
**Pages updated**: `wiki/MK/pallas-kernel.md`
**Notes**: Four of its five See-also targets (`mosaic-kernel.md`, `custom-call.md`, `flash-attention.md`, `attention-block-sizes.md`) were never in this tree; they came over with the MaxKernel import, and three exist only as stubs on the unmerged tokamax-context branch (#360). `splash-attention.md` exists only under the tokamax subwiki. Each now points at the page that covers the topic: the Pallas hub (Mosaic), the MK XProf page (custom calls), tokamax splash attention (also its `SplashConfig` block sizes), and tokamax dot_product_attention (flash attention). Link housekeeping only; no claims changed, `human_reviewed` unchanged.

## [2026-10-12] manual | public-repository hygiene: drop internal references

**Operation**: manual (human-directed: remove internal identifiers ahead of the public repository move)
**Pages created**: none
**Pages updated**: `wiki/codebases/torch_tpu/catalog/symbols.tsv` (doc-line column only, 10 rows: internal bug-tracker IDs and personal usernames dropped from TODO comments, bare sort-marker lines emptied), `wiki/codebases/torchtitan/concepts/dtensor-and-sharding-types.md` (a quoted source comment loses its username), `wiki/sources/2026-09-torchtitan-repo.md` (local checkout path replaced by a description), `log.md` (internal bug and short-link references and a personal branch name and PR reference removed from four older entries)
**Notes**: The symbol-index doc lines are copied verbatim from torch_tpu source comments, so a re-ingest at a new pin reintroduces them unless wikify learns to filter them; that is a wikify-side change, not part of this one. Hand-editing `catalog/` is otherwise disallowed by SCHEMA; this edit touches no anchor, path, line, hash or signature, only the doc text. No claims changed; `human_reviewed` flags preserved.

## [2026-10-12] manual | gpu-vs-tpu: TPU's 2x low-precision type is int8, fp8 only from TPU7x

**Operation**: manual (human-directed: correct a claim found while reviewing #698)
**Pages created**: none
**Pages updated**: `wiki/concepts/gpu-vs-tpu.md`
**Notes**: The page said "fp8 doubles FLOPs on both" GPU and TPU, citing scaling-book Part 12, which says so only for H100/B200; its TPU example (Part 1) is int8 at ~2x bf16. Now: fp8 on H100/B200, int8 on TPU v5e/v6e (`tpu-v5e`, `tpu-v6e`), fp8 only from TPU7x (`tpu-7x`). The claim changed, so `human_reviewed` reset to false. The tpu-knowledge-base copy is corrected in #698.

## [2026-10-11] manual | rename specialist subagents to TPU agent suite branding

**Operation**: manual (human-directed: rename specialist subagents per the TPU agent suite branding guidelines)
**Pages created**: none
**Pages updated**: `CONTRACT.md`, `index.md`, `wiki/analyses/30-model-onboarding-learnings.md`, `wiki/analyses/standard-intelligence-case-study.md`, `wiki/concepts/gpu-vs-tpu.md`, `wiki/kernel-optimization-index.md`
**Notes**: Updated references from `tpu-provenance` to `tpu-knowledge-base` and from `tpu-kernel` to `tpu-kernel-development` across the knowledge base contract, index, and wiki pages. No technical claims changed; `human_reviewed` flags preserved.

## [2026-10-11] manual | flatten repository root and scrub legacy framework references

**Operation**: manual (human-directed: flatten repository directory layout and scrub legacy framework references)
**Pages created**: none
**Pages updated**: `config/tokamax.md`, `config/torch_tpu.md`, `config/torchtitan.md`, `log.md`
**Notes**: Simplified knowledge base ingest config headings and updated paths in `log.md` after moving `package/`, `framework/`, and `eval/` to the repository root. No wiki content changed.

## [2026-10-06] manual | rename 30-model-migration-learnings.md to 30-model-onboarding-learnings.md

**Operation**: manual (human-directed: rename `30-model-migration-learnings.md` to `30-model-onboarding-learnings.md` and update links)
**Pages created**: none
**Pages updated**: `index.md`, `wiki/analyses/30-model-onboarding-learnings.md` (renamed from `wiki/analyses/30-model-migration-learnings.md`), `wiki/codebases/torch-tpu.md`. Additionally, instances of 'migration' were changed to 'onboarding' across the knowledge base where 'migration' referred to making changes to a model to get it to run on TPUs.
**Notes**: Renamed `wiki/analyses/30-model-migration-learnings.md` to `wiki/analyses/30-model-onboarding-learnings.md` (along with its `tpu-provenance` counterpart `analysis-30-model-onboarding-learnings.md`) and updated cross-references in `index.md` and `wiki/codebases/torch-tpu.md`. No claims changed; `human_reviewed` flags preserved.

## [2026-10-03] ingest | tokamax code wiki (openxla/tokamax @ 3a0a3a34)

**Operation**: ingest (human-directed: wikify-ingest-repo, lens and GPU-as-catalog chosen by the user)
**Pages created**: `wiki/codebases/tokamax.md`, `wiki/sources/2026-10-tokamax-repo.md`, the `wiki/codebases/tokamax/` subwiki (overview, 18 concept pages, 2 area pages, 14 doc-concept pages, symbol index)
**Pages updated**: `index.md` (Codebases row, Sources row, tokamax removed from Planned), `wiki/MK/pallas-kernel.md` (Sources and repo-link tail point at the subwiki), `wiki/codebases/index.md` (regenerated by wikify)
**Notes**:
- Pin is main HEAD 3a0a3a34 (2026-10-02); tokamax has no release tags. Apache-2.0.
- Lens: TPU kernels. GPU kernels are index-only by request, but listed with their purpose in `concepts/kernel-catalog.md` (every op x backend); GPU-only ops are sections of `areas/src.md`.
- Concept pages: linter-gated, then adversarially verified (337 claims, 19 refuted and corrected).
- wikify 0.4.0 quirks hit: `agenda_exclude: [dir]` also drops `dir/*` (worked around with `agenda_max: 1`, see `config/tokamax.md`); `finalize` regenerates area blocks after its source-link pass, so run `wikify source-links tokamax` after every `finalize`; lint does not resolve GitHub-style citations on doc-concept pages (checked separately against `catalog/symbols.tsv`).
- Working state moved: `.cache/` and `raw/code/{torch_tpu,torchtitan}` from the old `package/knowledgebase/` to this directory (gitignored either way).
- `human_reviewed: false` on the new hub and source pages; `pallas-kernel.md` was already false.

## [2026-10-02] manual | migrate deprecated env vars in torch_tpu concept pages

**Operation**: manual
**Pages updated**: `wiki/codebases/torch_tpu/overview.md`, `wiki/codebases/torch_tpu/concepts/deferred-execution-and-materialization.md`, `wiki/codebases/torch_tpu/concepts/error-handling.md`
**Notes**: Migrated deprecated environment variables to canonical names (TPU_DEFER_AND_FUSE -> TORCH_TPU_DEFER_AND_FUSE, TPU_LAUNCH_BLOCKING -> TORCH_TPU_LAUNCH_BLOCKING, TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS -> TORCH_TPU_ENABLE_DEBUG_CHECKS).

## [2026-10-01] manual | correct VMEM, SMEM, HBM and device facts in three MK pages

**Operation**: manual (human-directed: issue #400)
**Pages created**: none
**Pages updated**: `wiki/MK/pallas-core-mental-model.md`, `wiki/MK/pallas-validation-and-xprof.md`, `wiki/MK/attention-pallas-data-management.md`
**Notes**:
- Physical VMEM, SMEM and per-device HBM bandwidth now follow JAX `tpu_info.py` (be2c88887fb7) and the `hardware/` pages; the scoped VMEM limit is described separately, as measured on TPU7x with libtpu 0.0.44 / 0.0.44.1 (default 32 MiB; a kernel's `vmem_limit_bytes` wins over the flag).
- The hallucination ceiling takes the higher of the published spec and `tpu_info.py` per generation.
- The tpu-provenance copies of these pages carry the same corrections (#432).

## [2026-09-28] manual | the ten unwiring playbooks moved to tpu-provenance

**Operation**: manual (human-directed: tpu-onboarding could not read them here)
**Pages created**: none
**Pages updated**: `index.md` (ten playbook rows removed)
**Pages moved**: `wiki/playbooks/{apex-to-native-normalization, cuda-api-stripping-and-runtime, custom-cuda-kernel-unwiring, deepspeed-ops-unwiring, flash-attention-to-sdpa, huggingface-runtime-compatibility, mamba-ssm-unwiring, tpu-tensor-layout-and-interp, triton-to-pytorch-ops, xformers-to-sdpa}.md` -> `tpu-provenance/pages/`
**Notes**:
- `tpu-unwiring` reads all ten by path, and every read was broken: `{{knowledgebase}}` resolves to `tpu-provenance`, which has no `wiki/`, and `knowledgebase` is not in the package's `include[]`, so nothing here ships. Moved rather than copied, so there is one copy to keep current.
- Registered in `tpu-provenance/index/playbooks.jsonl` as `page` entries, which `generate()` returns directly: a lookup is a read, not a model call. Keywords are each page's id, its `aliases` and its `target_imports`, so `flash_attn` reaches the playbook that unwires it.
- Nine keywords naming a conversion (`flash-attention-to-sdpa`, `apex-to-native-normalization`, ...) were held by PyTorch API reference rows and by the Mamba paper. They now belong to the playbooks that perform them; the reference rows keep their API vocabulary.
- The five playbooks that no shipped agent reads stay here.

## [2026-09-26] manual | which torchtitan: google-pytorch vs the official pytorch repo; overview reviewed

**Operation**: manual (human-directed: state that the ingested repo is Google's TorchTitan with the TorchTPU backend, not the official one; record the human review of the torchtitan overview)
**Pages updated**: `wiki/codebases/torchtitan.md` (hub: "Which torchtitan" note, pin table names both repos), `wiki/codebases/torchtitan/overview.md` (short form of the note; a human reviewer stamp in `verified:`), `wiki/sources/2026-09-torchtitan-repo.md` (identity), `index.md` (TorchTitan row), `SCHEMA.md` (note on google-pytorch links)
**Notes**:
- Checked on GitHub and in the clone: `google-pytorch/torchtitan` is not a GitHub fork and has `torchtitan/experiments/tpu` (240 files); `pytorch/torchtitan` returns 404 for that path. The two share no git history (the Google repo's history starts 2026-03-10; upstream changes are ported by hand). Of the pin's 804 files, 301 exist at the same path in upstream `main` (c8a3e766, 2026-09-25), 75 of them identical.
- Concept pages were not edited, so their recorded verification stands.

## [2026-09-25] manual | route public TPU facts to tpu-provenance

**Operation**: manual (human-directed: stop the contract competing with tpu-provenance)
**Pages updated**: `CONTRACT.md`
**Notes**: The contract told every agent to read `index.md` first for any TPU
hardware, performance or optimization claim, so agents read the knowledge base
and never asked tpu-provenance. Public facts about TPU hardware, runtimes and
software -- specs, defaults, supported versions -- now go to tpu-provenance
first. The knowledge base stays the source for playbooks, the codebase map and
the Pallas concept pages. No wiki content changed.

## [2026-09-25] ingest | torchtitan @ 63e334d5 as a wikify subwiki

**Operation**: ingest (code repo via wikify-repo 0.3.4; human-directed: ingest torchtitan from main HEAD, add a page on the TorchTPU backend, finalize the ingest)
**Pages created**: `wiki/codebases/torchtitan/` (overview, 24 concept pages, 45 doc-concept pages, generated `index.md`, `log.md` and the `catalog/` symbol index), `wiki/codebases/torchtitan.md` (hub: pin and routing), `wiki/sources/2026-09-torchtitan-repo.md` (provenance), `config/torchtitan.md` (ingest config)
**Pages updated**: `index.md` (TorchTitan row routes to the overview; removed from the Planned list; source entry)
**Notes**:
- Ingested from a clean clone of `main` at `63e334d5` (2026-09-23), not from the existing developer working copy, which carried untracked scratch files.
- Agenda curated from the planner's proposal: one page per parallelism axis, and the six per-model directories folded into `model-definitions` (variants of one mechanism set, concept rule 4).
- wikify's planner excludes every `experiments/` path from concept pages and the list is not configurable, which would have hidden `experiments/tpu`. The `torchtpu-backend` page was seeded with explicit symbols instead, which resolve against the whole graph. To report upstream to wikify.
- `finalize` was green on the first pass. Adversarial verification checked all 428 load-bearing claims against the source; 24 wrong statements were corrected. Seven upstream bugs were confirmed from source and are listed on the overview under "Known issues at this pin".
- The subwiki's generated `index.md` contains non-ASCII punctuation from wikify's own template; it is tool-owned and regenerated by `finalize`, so it was not hand-edited.
- Withheld from the published wiki: partner model packages, Google-internal launch and test infrastructure, private plugin names, and the doc pages derived from them (wikify 0.4.1 `source_exclude` / `symbol_exclude`, listed in the uncommitted `.wikify/torchtitan.exclude.yaml`). Excluded code never enters the graph, so it is absent from every page, index row and edge; page prose that referred to it was rewritten (two doc-concept pages that existed only for it were deleted), lint re-checked every citation, and the 10 edited load-bearing claims were re-verified against the source.
- Citations link the source at the pin, as for torch_tpu: `wikify source-links` (wikify 0.4.0) rewrote all 2,409 citations from the shipped `catalog/symbols.tsv` to `[label](<GitHub blob at 63e334d5>/<path>#L<line> "<index key>")`. No catalog pages (tier `index`), no prose changed; every href matches its index row, lint is clean, and a 40-link sample lands on the symbol's line.

## [2026-09-25] manual | torch_tpu citations link the source at the pin

**Operation**: manual (human-directed: citations in generated pages point at the source code instead of catalog pages, from the original ingest's pin and index)
**Pages updated**: `wiki/codebases/torch_tpu/` (1,785 citations in 27 concept and 15 doc-concept pages), `wiki/codebases/torch-tpu.md` (hub: citation form), `wiki/sources/2026-09-torch-tpu-repo.md` (provenance), `SCHEMA.md` (codebase subwikis: citation form; wikify block regenerated from wikify 0.4.0)
**Notes**:
- Every citation pointed at a `catalog/<module>.md#<Symbol>` page that the `index` catalog tier does not render. Each is now `[label](<GitHub blob at 4c0c7c57>/<path>#L<line> "<index key>")`, rewritten by `wikify source-links` from the shipped `catalog/symbols.tsv` of the original ingest: no re-index, no catalog pages, no prose changed. All 1,785 hrefs match their index row; a 40-link sample lands on the symbol's line.
- The repo is private: the link works for readers with access. Agents are told not to fetch it and to grep the title in `catalog/symbols.tsv`, and to read a body from a local checkout or one `gh api` call at the pin.
- `SCHEMA.md` states that links into `https://github.com/google-pytorch/` repos are private (404 to a plain fetch) and are read through `gh` with access.

## [2026-09-24] manual | make the knowledge base a deselectable subpackage

**Operation**: manual (human-directed: allow builds with no knowledge base, for ablation)
**Pages created**: `knowledgebase.json`
**Pages updated**: `CONTRACT.md`
**Notes**: `knowledgebase` moves from the package manifest's `copy` list to
`include`, so `--packages` selects it like any other subpackage and a build can
leave it out. `knowledgebase.json` carries the `copy` entry and the workspace
memory block, which previously sat at the package root and rendered even when
the tree was absent. `CONTRACT.md` gains the `SCHEMA.md` routing sentence that
was duplicated in tpu-agent's instructions, which now state the rule once.
No wiki content changed.

## [2026-09-22] manual | add custom CUDA kernel unwiring playbook and parity verification workflow

**Operation**: manual (human-directed: add playbook for custom CUDA unwiring and parity verification)
**Pages created**: `wiki/playbooks/custom-cuda-kernel-unwiring.md`
**Pages updated**: `index.md`
**Notes**: Adds playbook detailing reverse-engineering custom/uncataloged CUDA kernels into native PyTorch fallbacks, executing teacher-forced probe validation with `numerical-parity`, and escalating unresolvable kernels to MaxKernel Pallas synthesis.

## [2026-09-22] manual | fix torch.compile backend and remove deprecated XLA_USE_BF16

**Operation**: manual (human-directed: update torch.compile backend to tpu and remove deprecated XLA_USE_BF16)
**Pages created**: none
**Pages updated**: `index.md`, `wiki/playbooks/cuda-api-stripping-and-runtime.md`, `wiki/playbooks/static-shapes-and-tracing.md`
**Notes**: Updated `backend="openxla"` to `backend="tpu"` in `static-shapes-and-tracing.md`, removed deprecated `XLA_USE_BF16=1` from `cuda-api-stripping-and-runtime.md` and `index.md`, and replaced with native PyTorch precision controls (`torch.bfloat16`, `torch.autocast`).

## [2026-09-22] manual | public naming standard: tpu-* components, onboarding vocabulary

**Operation**: manual (human-directed: rename components to the PM-agreed public naming standard)
**Pages created**: none
**Pages updated**: `SCHEMA.md` (taxonomy row `#migration` -> `#onboarding`; playbook subsection heading; illustrative filename `migrate-cuda-streams.md` -> `convert-cuda-streams.md`), `index.md` (section heading, agent/package names, link text), `wiki/kernel-optimization-index.md`, and the pages carrying component names or the `#migration` tag: `wiki/analyses/30-model-migration-learnings.md`, `wiki/analyses/standard-intelligence-case-study.md`, `wiki/concepts/gpu-vs-tpu.md`, `wiki/sources/2025-scaling-book-gpus.md`, and 10 `wiki/playbooks/` pages.
**Notes**: No claim changed; this is naming and vocabulary only. Component renames: maxshard -> tpu-sharding, maxkernel -> tpu-kernel, maxauto -> tpu-performance, xprof -> tpu-profiling, provenance -> tpu-provenance, torchtpu-migration -> tpu-onboarding, hello-world -> tpu-hello-world. Customer-facing "migration" vocabulary became "onboarding" for positioning (the suite must not read as being about moving workloads off GPUs); the four X-to-Y playbook titles use "Conversion" instead, since "Onboarding" is ungrammatical there. Deliberately unchanged: the CUDA->TorchTPU construct mappings (reference material an engineer needs, not positioning); the real XProf tool's names; `wiki/codebases/xprof.md` and `wiki/MK/`; the filename of `30-model-migration-learnings.md` (this log links to it); and all captured sources under `raw/`. Human review flags preserved: these were naming-only edits, not claim-touching.

## [2026-09-21] manual | deprecate torch_xla and standardize on torch_tpu APIs

**Operation**: manual (human-directed: enforce torch_tpu and deprecate torch_xla across docs and playbooks)
**Pages created**: none
**Pages updated**: `wiki/codebases/torch-tpu.md`, `wiki/playbooks/cuda-api-stripping-and-runtime.md`, `wiki/playbooks/static-shapes-and-tracing.md`
**Notes**: Updated device acquisition to `torch.device('tpu')`, step synchronization to `torch.accelerator.synchronize()`, and added explicit deprecation notices for `torch_xla`.

## [2026-09-17] manual | eliminate subagent setup references in playbooks and case studies

**Operation**: manual (human-directed: centralize environment setup and eliminate setup_tpu_env)
**Pages created**: none
**Pages updated**: `wiki/analyses/standard-intelligence-case-study.md`, `wiki/playbooks/tpu-distributed-and-env-hygiene.md`
**Notes**: Updated environment provisioning steps to reference centralized requirements (`requirements-tpu.txt`) instead of deprecated `setup-tpu-env` subagent.

## [2026-09-16] ingest | torch_tpu @ 4c0c7c57 as a wikify subwiki (rebuilt with wikify 0.3.0)

**Operation**: ingest (code repo via wikify-repo 0.3.0; human-directed: rebuild the torch_tpu silo from a clean clone and open a fresh PR; supersedes the earlier 2026-09-05 build)
**Pages created**:
- Subwiki `wiki/codebases/torch_tpu/` at commit `4c0c7c5711368001ed94830a4fe76652d1cb6c31`: `overview.md`, 27 concept pages under `concepts/`, 15 doc-concept pages under `doc-concepts/` (from 12 README files), and the symbol index under `catalog/` (`symbols/<dir>.tsv`: 13,328 rows in 8 shards, full profile with signatures and doc lines; `edges/<dir>.tsv`: every caller edge; `index.md`: module map over 556 modules; 100% of documentable symbols represented); generated silo `index.md` and `log.md`; `wiki/codebases/index.md` (generated silo listing).
- `wiki/sources/2026-09-torch-tpu-repo.md` (provenance record), `config/torch_tpu.md` (ingest config: pin, languages, bazel targets, `catalog: index`, `index_profile: full`, curated concept list).
**Pages updated**: `wiki/codebases/torch-tpu.md` (hub: pin, routing into the subwiki, symbol-index description), `SCHEMA.md` (codebase-subwiki conventions for the index tier; wikify 0.3.0 retrieval block), `index.md` (TorchTPU row routes to the overview; source entry), `scripts/validate_kb.py` (subwiki pages excluded from the catalog-completeness check; each subwiki must be reachable through its `overview.md`), `package/tests/test_knowledgebase_playbooks.py` (subwiki pages carry wikify front matter and are gated by `wikify finalize`, so the KB schema and link checks stop at the subwiki boundary), `framework/compiler.py` (+ tests: the dist never ships `raw/code/`, `.cache/`, `.wikify/`, `.agents/`, `.claude/`), `.gitignore` (same paths never enter git).
**Notes**:
- Why a rebuild: the 2026-09-05 build was indexed from a developer working copy, so untracked Jupyter checkpoints and an empty-path shard leaked into 560 per-module catalog pages, overloaded functions showed their first `@overload` stub (`jax_op(name: str, fn: None = None,, *, ...)`), enum comments were inherited across members, and a leading underscore was stripped from a symbol-base prefix. wikify 0.3.0 fixes all of these and ships the catalog as a greppable symbol index instead of pages (rationale: `docs/catalog-index.md` in wikify-repo — agents never opened catalog pages when the source was on disk).
- Reconcile at the same pin from the clean clone: 26 concept pages unchanged, 1 marked stale by merged-record hashes (`pallas-custom-kernels`); its citations all resolve in the new graph and the source is unchanged, so the verified content was kept and re-stamped. The 15 doc-concept pages were kept (worklist unchanged). No area pages: the curated 27-unit agenda covers the plan.
- wikify nit to report upstream: the silo `log.md` entry for a same-ref rebuild links `changes/4c0c7c5711.md`, which `finalize` did not write.
- Size: the symbol index is ~9.4 MB (edges 7.0 MB, symbols 2.4 MB) against ~5 MB of pages it replaces; shipped as-is, flagged in the PR.
- Log operation token is `ingest` (the schema enum); the superseded branch used `ingest-code`.

## [2026-09-15] manual | index.md page-count header removed; section counts dropped

**Operation**: manual (tooling; human-directed after PR review feedback on the hand-maintained page count)
**Pages created**: none
**Pages updated**: `index.md` (the `*Last updated: … — N pages total*` line and the `(N)` section counts are gone), `SCHEMA.md` (index skeleton without them; rationale recorded; rule 2 names the validator), `scripts/validate_kb.py`, `package/tests/test_validate_kb.py`, `Makefile` (`validate-kb` target, prerequisite of `verify`), `.github/workflows/ci.yml` (PR log-entry check against the base branch), `.pre-commit-config.yaml` (hook passes `--check-git`)
**Notes**: the count was redundant — the validator already requires every page under `wiki/` to be cataloged and every index link to resolve, which implies it — and it was a merge conflict on every concurrent knowledge base PR; the date was a second conflict on the same line and is carried by `log.md` and by each page's `updated:` field. The validator now rejects log operation tokens outside the schema enum (`ingest | query | lint | manual`), and keeps the catalog-completeness, link-resolution and ordering checks. The log-entry rule (`--check-git`, now with `--base REF` for comparing a branch to its base) is enforced at commit time by pre-commit and on every PR by the CI workflow; previously nothing invoked it. No wiki page content changed.

## [2026-09-10] ingest | MaxKernel Pallas Kernel Optimization Playbooks & Concepts

**Operation**: ingest (human-directed: integrate MaxKernel local wiki optimization playbooks & concepts)
**Pages created**:
- 16 playbooks: `wiki/kernel-optimization-index.md`, `wiki/MK/pallas-core-mental-model.md`, `wiki/MK/pallas-tpu-reference.md`, `wiki/MK/pallas-validation-and-xprof.md`, `wiki/MK/matmul-pallas-architectures.md`, `wiki/MK/attention-pallas-data-management.md`, `wiki/MK/design-class-catalog.md`, `wiki/MK/kernel-attention.md`, `wiki/MK/kernel-cross-chip-collective.md`, `wiki/MK/kernel-dense-near-roofline.md`, `wiki/MK/kernel-gemm-conv-epilogue.md`, `wiki/MK/kernel-grouped-ragged-indirection.md`, `wiki/MK/kernel-state-carry-scan.md`, `wiki/MK/kernel-streaming-reduction.md`, `wiki/MK/briefs.md`, `wiki/MK/pallas-kernel.md`.
- 3 concept pages: `wiki/concepts/grid_blockspec.md`, `wiki/concepts/matmul.md`, `wiki/concepts/pipelining.md`.
**Pages updated**: `index.md` (catalog updated to 63 pages total, Pallas kernels playbook section and concepts populated).
**Notes**: Added SOTA mental model guides, TPU v6e data management, validation/XProf profiling workflows, and category-specific Pallas kernel design classes to support local LLMWiki kernel optimization.

## [2026-09-04] ingest | How To Scale Your Model — remaining chapters (Parts 1, 3–10, 12, intro, conclusion)

**Operation**: ingest (human-directed: "do ingestion for scaling book now"; guidance: v6e and TPU7x constants take priority over the book's v5e/v5p ones, worked problems are not worth capturing, no ASCII restriction on page style, orient toward what the delegate agents need without over-indexing on them)
**Pages created**:
- sources (11): `wiki/sources/2025-scaling-book.md` (book hub; covers intro + conclusion), `2025-scaling-book-{roofline,sharding,transformers,training,applied-training,inference,applied-inference,profiling,jax-stuff,gpus}.md`
- concepts (17): parallelism — `parallelism-strategies` (hub), `sharding-notation`, `collective-costs`, `data-parallelism`, `fsdp`, `tensor-parallelism`, `pipeline-parallelism`, `expert-parallelism`, `collective-matmul`; performance — `critical-batch-size`, `mfu`, `transformer-flops-accounting`, `kv-cache`, `prefill-vs-decode`, `gradient-checkpointing`; compilation — `hlo`; architecture — `gpu-vs-tpu`
- codebases (2): `jax`, `xprof` (both with canonical URLs marked not yet ingested)
- playbooks (3, new "End-to-end optimization" subsection): `profile-tpu-program`, `size-training-parallelism`, `size-serving-topology`
**Pages updated**: `roofline-analysis` (formal model, network rooflines), `ici` (collective costs, ICI arithmetic intensity, latency floor), `quantization` (effect on rooflines / memory fit), `hbm` (decode step-time formula), `vmem` (HLO memory space, Flash Attention residency, vs SMEM/L2), `vpu` (vs CUDA cores), `mxu` (vs Tensor Cores), `compute-communication-overlap` (critical-path vs async collectives), `dcn` (cross-slice DP threshold), `padding` (see-also HLO), `pallas` (book positions it as the escape hatch; Pallas series fetch target), `2025-scaling-book-tpus` (re-fetched 2026-09-04, text identical, pin recorded, hub link), hardware `tpu-v5e/v5p/v6e/7x` (see-also to critical-batch-size only), `index.md` (44 → 77 pages; Concepts gained Parallelism and Compilation subsections; Playbooks split into Onboarding / Pallas kernels (empty) / End-to-end optimization; Sources grouped).
**Notes**:
- Raw: `raw/docs/2025-scaling-book-{index,roofline,sharding,transformers,training,applied-training,inference,applied-inference,profiling,jax-stuff,gpus,conclusion}.html`, fetched 2026-09-04 from the live site. Upstream HEAD at fetch `ac72f02` (2026-09-03); per-chapter last-change commits recorded on each source page. Decision: per-page HTML snapshots (matching the 2026-08-22 Part 2 ingest) rather than a wikify docs-mode repo ingest — the docs-mode output conflicts with this KB's frontmatter/index/link contracts and would ship ~76 MB of Jekyll assets; wikify is deferred to a code-repo pilot.
- Editorial: the book is JAX-first — mechanism pages stay framework-neutral and every place a TorchTPU spelling would go is marked *not yet ingested* rather than inferred from JAX. Hardware numbers: formulas cite the book, constants cite the Cloud-docs hardware pages; the per-generation threshold table on `critical-batch-size` (HBM and ICI arithmetic intensities for v5e/v5p/v6e/TPU7x, and the resulting DP/FSDP/TP floors) is arithmetic on cited numbers and is flagged *derived — verify before relying*. Its headline: FLOPs grew ~5× from v5p to TPU7x on the same 1.8e11 bidi ICI per axis, so ICI-bound thresholds are ~5× tighter on TPU7x (~2× on v6e). Human review of that table is the top follow-up.
- Granularity calls: sequence/context parallelism and 2D weight-stationary sharding → sections of `tensor-parallelism`; continuous batching, disaggregation, speculative sampling → sections of `prefill-vs-decode`; prefix caching and paged attention → sections of `kv-cache`; arithmetic intensity stays a section of `roofline-analysis`; the four collectives share one page (`collective-costs`, split past ~500 lines); `parallelism-strategies` is a hub with a comparison table only. Worked problems (58) deliberately not captured; a handful of their conclusions were lifted and are listed under each source page's Residue.
- Playbooks carry synthetic `target_imports` keys (`tpu_profiling`, `sharding_config`, `serving_topology`) to satisfy the package test's discovery contract; they are not real import triggers and preflight will not match them.
- JetStream's archival status (from a sibling wiki) is noted on the Part 7 source page as *unverified*. Gaps recorded: TorchTPU mapping for every mechanism; how far expert parallelism can span over ICI on v6e/TPU7x; Pallas; xprof beyond the three tabs; PyTorch/XLA profile capture.
- human_reviewed **reset to false** (claim-touching edits): `roofline-analysis`, `ici`, `quantization`, `hbm`, `vmem`, `vpu`, `compute-communication-overlap`, `dcn`. Preserved: hardware pages (see-also housekeeping only; `updated:` bumped), `mxu`/`padding`/`pallas` were already false.
- Citation labels (housekeeping, same day): every scaling-book reference now names the book — Sources bullets read `How To Scale Your Model, Part N: <chapter> (scaling-book)`, inline citations read `scaling-book, Part N`; 32 pages, link text only, no claims or `human_reviewed` flags touched.
- Adversarial verification and review flip (same day, human-directed): the human stated they had reviewed all sources and authorized the agent to set `human_reviewed: true` on their behalf for this session. Agent pass first: (1) every number on the 33 new pages and the 10 edited pages cross-checked against the raw chapter text — the only unmatched figures are Cloud-docs inputs and the explicitly derived ratios on `critical-batch-size`/`gpu-vs-tpu`; (2) load-bearing qualitative claims re-read against the cited chapters. Defects fixed before flipping: `pipeline-parallelism` named DualPipe/GPipe/1F1B, which the source does not (removed); `gpu-vs-tpu` A100/H200 intensities and `kv-cache` LLaMA-2 per-token bytes marked derived/computed; `kv-cache` GQA throughput gain quantified from the book's table (964 → 4,529 tokens/s); `parallelism-strategies` rule 1 restated as the book states it; `expert-parallelism` AllToAll sentence tightened to the source; `2025-scaling-book-gpus` commit attribution corrected (spine-switch `60a6fa6`, fat-tree `ed80595`); `profile-tpu-program` step 6 heuristics marked agent-derived; `gradient-checkpointing` trade-off rule cited to Part 6; `critical-batch-size` gained the `β = W_hbm/W_ici` column that `tensor-parallelism` references. **human_reviewed set to true on 41 pages** (33 created today + the 8 reset earlier today). Still worth a second look: the derived per-generation threshold table on `critical-batch-size` (arithmetic on cited inputs; the TPU7x per-axis ICI figure comes from the book's Part 2 table). `mxu`, `padding`, `pallas`, `megacore` stay false (pre-existing status).
- Suggested next sources: TPU7x collective/ICI benchmark or microarchitecture doc (to ground the derived table); torch_tpu repo snapshot (SPMD/FSDP/remat/profiler spellings); JAX distributed-arrays guide; Pallas TPU kernel series; xprof repo; *Efficiently Scaling Transformer Inference* (arXiv 2211.05102).

## [2026-08-28] ingest | TorchTPU Knowledge Base & CUDA Unwiring Playbooks

**Operation**: ingest (human-directed: port TorchTPU playbooks & SI customer findings)
**Pages created**:
- 11 playbooks: `wiki/playbooks/flash-attention-to-sdpa.md`, `mamba-ssm-unwiring.md`, `xformers-to-sdpa.md`, `triton-to-pytorch-ops.md`, `apex-to-native-normalization.md`, `deepspeed-ops-unwiring.md`, `cuda-api-stripping-and-runtime.md`, `tpu-distributed-and-env-hygiene.md`, `tpu-tensor-layout-and-interp.md`, `static-shapes-and-tracing.md`, `huggingface-runtime-compatibility.md`.
- 1 codebase page: `wiki/codebases/torch-tpu.md`.
- 2 analysis pages: `wiki/analyses/30-model-migration-learnings.md`, `wiki/analyses/standard-intelligence-case-study.md`.
**Pages updated**: `index.md` (catalog updated to 44 pages total, playbooks, codebase and analysis sections populated).
**Notes**: Implements verified unwiring recipes for FlashAttention-2/3, Mamba SSM associative scans, xFormers, Triton JIT kernels, APEX normalization, DeepSpeed optimizers, and runtime precision/layout patches. Incorporates empirical receipts from 30-model TPU runs (>97% XLA cache hits) and Standard Intelligence TPU v7 onboarding report (FA3 unwiring, avoiding `uv sync` dependency clobbering, `nccl` -> `tpu_dist`).

## [2026-08-24] manual | lint follow-ups: megacore adjudicated, MXU claim attribution, rule-10 amendment

**Operation**: manual (human-directed: "address those")
**Pages created**: none
**Pages updated**: `megacore` — `[!warning]` resolved as **human-adjudicated**: official v6e docs win over the scaling-book's "v6 is megacore" claim; kept as a resolved-contradiction note for provenance. `mxu` + `padding` — the 256×256 claim now attributes v6e to the scaling-book and "later generations" to a dated human-reviewed assertion (no ingested source states it yet; an Ironwood microarchitecture source would ground it). SCHEMA.md — human_reviewed reset rule amended: claim-touching agent edits reset the flag; pure housekeeping (metadata counts, link formatting, section scaffolding) preserves it, still logged.
**Notes**: flags — `megacore`, `padding` reset to false (claim-touching edits; human may re-flip); `tpu-v6e`, `sparsecore` restored to true (previous resets were housekeeping-only, exempt under the amended rule; the human had set them). Zero unresolved `[!warning]` blocks remain.

## [2026-08-24] lint | full lint pass after human review sweep

**Operation**: lint (human-requested)
**Pages created**: none
**Pages updated**: `mxu` (sources 5→6, user-added TPU7x citation formatted into list), `tpu-v6e` (sources 1→2), `sparsecore` + `pallas` (missing `## Sources` sections added). **human_reviewed reset to false** by these agent edits on `mxu`, `tpu-v6e`, `sparsecore` (rule 10) — metadata-only fixes; human may re-flip.
**Notes**: clean: links, orphans, frontmatter/type/dates, index sync (30/30). Human review sweep (committed 0f15099) marked 22 pages reviewed and extended the 256×256-MXU claim to "v6e and more recent chips" (mxu, padding) citing TPU7x docs — human-asserted; the 7x docs don't state MXU size explicitly. Open items: megacore `[!warning]` awaits explicit adjudication (docs presumed authoritative); stubs `pallas`/`sparsecore` legitimately blocked on sources; source pages have no inline citations by design (provenance layer). Suggested next sources: Pallas docs (canonical URL on stub), torch_tpu repo snapshot, scaling-book Part 3 (sharding), SparseCore documentation, 8t/8i Cloud docs when published (supersession watch).

## [2026-08-24] manual | Pallas stub fixes: canonical URL, de-scoped VMEM-tunability claim

**Operation**: manual (human review of the stub)
**Pages created**: none
**Pages updated**: `pallas` (added canonical docs URL https://docs.jax.dev/en/latest/pallas/index.html marked "not yet ingested"; VMEM-tunability rephrased); `vmem` (same claim de-scoped from "on TPU7x" to generation-neutral, with `*(unverified — agent inference)*` on the generalization); SCHEMA.md (codebase pages carry a canonical docs/repo URL as a fetch target, not a citation).
**Notes**: the "tunable VMEM" claim was cited correctly but scoped misleadingly — the TPU7x docs state it, yet it's a Pallas/XLA-stack property, not 7x silicon; wording implied exclusivity. Single-source attribution bias; the generalization stays marked unverified until a Pallas source confirms. Routing decision affirmed: Pallas stays under codebases/ (named software, future repo/docs ingest) — kernel techniques from it will land in concepts/playbooks.

## [2026-08-22] lint | review pass: Pallas stub + stale quantization index hook

**Operation**: lint (human-requested review; fixes approved)
**Pages created**: `wiki/codebases/pallas.md` (stub — first codebases page; was mentioned on `vmem`/`tpu-7x` without a page, violating the extraction rules; central to the kernel-optimization use case)
**Pages updated**: `vmem`, `tpu-7x`, `2026-08-cloud-tpu-7x-docs` (Pallas mentions → links, connections); `index.md` (quantization hook corrected — speedup varies by generation, v4/v5p have none; Codebases section opened; count 30)
**Notes**: mechanical lint was clean (29/29 frontmatter, 0 orphans, index in sync). Remaining known gaps, deliberate backlog: codebases and playbooks largely empty (use cases 2–5); most concept pages single-source; Pathways/Sax remain inline mentions (fail the page-worthiness test for now).

## [2026-08-22] manual | date-prefix convention: natural granularity (yyyy-mm for snapshots)

**Operation**: manual (naming refinement, human-approved)
**Pages created**: none
**Pages updated**: SCHEMA.md naming table — source/raw prefixes now use the source's natural date granularity: `<yyyy>` publication year for one-off works, `<yyyy-mm>` snapshot month for living sources (docs sites, repos), `<yyyy-mm-dd>` for analyses; raw files share their source page's prefix; raw code dirs `<repo>-<yyyy-mm>`. Renamed `2026-cloud-tpu-{v4,v5e,v5p,v6e,7x}-docs` (source pages + raw HTML) to `2026-08-…`; all wiki/index references updated. Historical log entries keep old names (log records what was true at the time).
**Notes**: fixes a latent collision — living docs re-ingested later the same year would have collided under year-only prefixes; publications keep citation-style year identity (no fake months). Karpathy's gist prescribes no naming; this formalizes tpu_wiki's implicit practice (its repo snapshots were already `2026-04-…`).

## [2026-08-22] manual | source pages redefined as provenance records (residue rule)

**Operation**: manual (schema refinement, human-approved)
**Pages created**: none
**Pages updated**: SCHEMA.md (INGEST step 4 + page-types `source` row: source pages are provenance records — identity/date/authority/gaps/connections; knowledge lives once on wiki pages; residue rule for deliberately-unextracted content; rich many-page sources may keep a short claims digest; 1:1 sources ≈15 lines); slimmed 6 source pages (`2026-cloud-tpu-{v4,v5e,v5p,v6e,7x}-docs`, `2026-google-blog-tpu-8th-gen`) — dropped Key claims duplicating hardware/concept pages, kept authority + caveats + connections; `2025-scaling-book-tpus` residue note updated (book spec tables superseded by Cloud docs; book still sole holder of v3 numbers); `hbm` stale "pending specs ingestion" note → links to hardware pages.
**Notes**: matches Karpathy's original gist, which specifies only a thin per-source "summary page"; the heavy Key-claims template was a tpu_wiki elaboration useful for rich sources only. 8th-gen blog page gains an explicit supersession watch: re-derive tpu-8t/8i from Cloud docs when they appear.

## [2026-08-22] ingest | TPU specs: Cloud docs v4/v5e/v5p/v6e/7x + 8th-gen announcement

**Operation**: ingest (6 sources, human-directed "ingest these for tpu stats")
**Pages created**: hardware: `tpu-v4`, `tpu-v5e`, `tpu-v5p`, `tpu-v6e`, `tpu-7x`, `tpu-8t`, `tpu-8i`; concepts: `sparsecore` (stub); sources: `2026-cloud-tpu-{v4,v5e,v5p,v6e,7x}-docs`, `2026-google-blog-tpu-8th-gen`
**Pages updated**: `megacore` (**first `[!warning]` contradiction**: book says v6 is megacore, v6e docs say 1 TensorCore/chip; + TPU7x dual-chiplet detail), `ici` (twisted torus ~70%/40% bisection gains, ICI resiliency), `vmem` (TPU7x tunable VMEM / Pallas block-size constraint), `quantization` (per-generation low-precision speedup table — v4/v5p get NO speedup), `mxu` (MXUs-per-TensorCore by generation), `index.md`
**Notes**: TPU 8t/8i pages carry explicit "vendor announcement claims, no datasheet, don't use for roofline math" warnings (121 EF/pod has unstated precision). Naming convention applied: `tpu-<version>.md`, codenames only in title/aliases. Cloud docs pages all "last updated 2026-08-11". Raw: `raw/docs/2026-cloud-tpu-*-docs.html`, `2026-google-blog-tpu-8th-gen.html`.

## [2026-08-22] manual | hardware/ redefined as generation spec pages; blocks moved to concepts/

**Operation**: manual (schema refinement, human-directed)
**Pages created**: none
**Pages updated**: moved `tensorcore`, `mxu`, `vpu`, `vmem`, `hbm`, `ici`, `dcn`, `pcie` from `wiki/hardware/` to `wiki/concepts/` (frontmatter `type: hardware` → `concept`; all cross-links rewritten); SCHEMA.md (page-types table, directory layout, extraction rule 5, naming table); index.md.
**Notes**: the hardware/concepts boundary was fuzzy (megacore-vs-tensorcore inconsistency). New rule: `hardware/` holds only per-generation spec pages — volatile cited numbers; `concepts/` holds all timeless mechanism, hardware blocks included. Naming convention for generation pages: `wiki/hardware/tpu-<version>.md` (official version tag lowercased: `tpu-v5p`, `tpu-v6e`, `tpu-7x`); codenames (Trillium, Ironwood) only in `title`/`aliases`, never in filenames.

## [2026-08-22] ingest | How to Think About TPUs (scaling-book, Part 2)

**Operation**: ingest
**Pages created**: `wiki/sources/2025-scaling-book-tpus.md`; hardware: `tensorcore` (hub), `mxu`, `vpu`, `vmem`, `hbm`, `ici`, `dcn`, `pcie`; concepts: `systolic-array`, `roofline-analysis`, `compute-communication-overlap`, `padding`, `quantization`, `megacore`
**Pages updated**: `index.md`
**Notes**: Human guidance: skip per-generation spec pages — a dedicated detailed specs ingestion will follow; ingest core concepts instead. Spec tables remain in the raw snapshot and are flagged as deliberately-not-paged on the source page. Granularity calls: VMEM prefetching filed as a section of `vmem` (variant, will split on a second source); scalar core/SMEM/VREGs/XLU as sections of `vpu`; torus/wraparound rules as sections of `ici`; worked roofline problems kept on the source page as future playbook fodder. `quantization` written as an ML concept from the TPU angle (native int8/int4 throughput), methods left for future sources. Raw: `raw/docs/2025-scaling-book-tpus.html`.

## [2026-08-22] manual | added concept-extraction guidance to SCHEMA.md

**Operation**: manual
**Pages created**: none
**Pages updated**: SCHEMA.md — new "Concept extraction (during ingest)" section: per-candidate decision list (already paged? → TPU optimization concept: always page → ML concept: page from its TPU angle → variant: section → else route by type), a "Concept granularity" rubric (decision-sized technique litmus + too-coarse/right/too-fine table + hub pages for umbrella terms), stub graduation criteria, link-on-creation and relations-in-prose rules; INGEST step 6 references it.
**Notes**: closes a gap — the schema said "create stubs for named things" without defining what qualifies or at what granularity.

## [2026-08-22] manual | knowledge base bootstrapped

**Operation**: manual
**Pages created**: SCHEMA.md, index.md, log.md; empty directory tree (`raw/{docs,papers,code,assets}`, `wiki/{hardware,codebases,concepts,playbooks,sources,analyses}`)
**Pages updated**: none
**Notes**: structure only — no content ingested. Schema adapted from the `tpu_wiki` prototype, extended with the `playbook` page type (serving CUDA→TorchTPU onboarding, Pallas kernel optimization, and end-to-end optimization use cases), the `human_reviewed` frontmatter flag, and the explicit two-hop citation chain (wiki page → source page → raw file).
