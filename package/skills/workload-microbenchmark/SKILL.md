---
name: workload-microbenchmark
description: Turns a finished serving job (e.g. a GLM-5.2 AgentX run) into representative, per-rank microbenchmark configs that run on a small slice (2x2x1, scaled mode) or on the job's own slice with its own sharding (mirror mode, e.g. 2x2x2) — representative cells and weights derived from the job's own requests, per-rank shapes, a per-rank "twin" of the model with one layer per layer kind, the component benchmarks needed where the twin deviates, a feasibility gate, and which benchmark level can decide the question. Users can override the deployment ("remove DCP", "dcp=8", "mnbt=8192", a kernel env) and every dependent value is re-derived. Use when asked to "build a microbenchmark for job j-…", "make a representative benchmark for this AgentX run", "onboard <model> for microbenchmarking", "what if we drop DCP", "derive L1 configs", or when a kernel or sharding-implementation change needs an A/B at the workload's real shapes. Onboarding a new model writes code for review under <work>/models/<MODEL_DIR>/; day-to-day configs go to <work>/ (default ./.work).
scripts:
- scripts/common/serving_benchmark.py
- scripts/workload-microbenchmark/GLM-5.2-FP8/config/agentx-2x2x2-tp16-dcp16-ep.bars.json
- scripts/workload-microbenchmark/GLM-5.2-FP8/config/agentx-2x2x2-tp16-dcp16-ep.json
- scripts/workload-microbenchmark/GLM-5.2-FP8/config/agentx-2x2x2-tp16-dcp16-ep.source.json
- scripts/workload-microbenchmark/GLM-5.2-FP8/config/agentx-2x2x2-tp16-dcp16-ep.twin8.json
- scripts/workload-microbenchmark/GLM-5.2-FP8/model.py
- scripts/workload-microbenchmark/Qwen3.5-397B-A17B-FP8/config/qwen3.5-397b-agentx-singlehost-2x2x1-tp8-ep.bars.json
- scripts/workload-microbenchmark/Qwen3.5-397B-A17B-FP8/config/qwen3.5-397b-agentx-singlehost-2x2x1-tp8-ep.json
- scripts/workload-microbenchmark/Qwen3.5-397B-A17B-FP8/config/qwen3.5-397b-agentx-singlehost-2x2x1-tp8-ep.source.json
- scripts/workload-microbenchmark/Qwen3.5-397B-A17B-FP8/config/qwen3.5-397b-agentx-singlehost-2x2x1-tp8-ep.twin8.json
- scripts/workload-microbenchmark/Qwen3.5-397B-A17B-FP8/model.py
- scripts/workload-microbenchmark/core/cachelocal.py
- scripts/workload-microbenchmark/core/cells.py
- scripts/workload-microbenchmark/core/cli.py
- scripts/workload-microbenchmark/core/costmodel.py
- scripts/workload-microbenchmark/core/deploy.py
- scripts/workload-microbenchmark/core/derive.py
- scripts/workload-microbenchmark/core/evidence.py
- scripts/workload-microbenchmark/core/forward_run.py
- scripts/workload-microbenchmark/core/jobdata.py
- scripts/workload-microbenchmark/core/jobdata_agentx.py
- scripts/workload-microbenchmark/core/mix.py
- scripts/workload-microbenchmark/core/project.py
- scripts/workload-microbenchmark/core/report.py
- scripts/workload-microbenchmark/core/steps.py
- scripts/workload-microbenchmark/core/xprof_steps.py
- scripts/workload-microbenchmark/tests/test_costmodel.py
- scripts/workload-microbenchmark/tests/test_derive.py
- scripts/workload-microbenchmark/tests/test_forward_run.py
- scripts/workload-microbenchmark/tests/test_mix.py
- scripts/workload-microbenchmark/tests/test_sampling.py
- scripts/workload-microbenchmark/tests/test_xprof.py
references:
- references/workload-microbenchmark
delegates:
- tpu-diagnostics
host_dependencies:
- pyyaml
---

# `workload-microbenchmark`

Derive benchmarks that stand in for a dynamic serving workload, from a job that already ran. How
the code fits together is in {{reference:workload-microbenchmark/core/README.md}}.

In this skill, `cli.py` stands for `python3 {{script:workload-microbenchmark/core/cli.py}}`, run
from the workspace. `<work>` is its work directory (`cli.py --work`, default `./.work`), and
`<run_id>` is this session's run id.

## The idea in four lines

- A dynamic run reduces to step types × counts: `T_wall ≈ N_A·t_A + N_B·t_B`. Here A is a mixed
  prefill step (one chunk on a cached prefix plus the running decodes) and B is a decode step.
  Check that closure first.
- **L1 answers one question:** does change X make the units it touches faster, and still correct,
  at the workload's shapes?
  - It never loads the whole model. It runs one rank's slice.
  - It uses the real vllm-torchtpu engine as a **twin**: the model with one layer of each kind.
    In scaled mode it runs on 2x2x1 with split dims shrunk so each rank keeps the served per-rank
    shapes; in mirror mode it runs the served sharding on the served slice (see "Two modes").
  - It adds **component** benchmarks only where the twin's shapes deviate from serving.
- L2 (fixed-work server cells) and L3 (the real run) are *generated* here but *run* as serving
  jobs with `{{runtime}} compute submit --serve` and a colocated client.
- Every output says which level can decide it: kernel/env changes → L1; sharding or scheduler
  changes → L1 per-layer Δ, and the verdict needs L2.

## When not to use it

This skill times the model's forward steps on dummy weights, with no server around them. Don't use
it for questions that live outside that:

- **vLLM orchestration.** There is no scheduler, no API server, no async-scheduling overlap, no
  detokenization, and no request admission. (The sample stage itself, LM head + sampler +
  rejection sampler, does run when the plan asks for it: see "Sampling".) `forward_run` drives the engine's model
  runner directly with pre-built batches. A bug or slowdown in scheduling, batching decisions,
  host dispatch, Ray / DP coordination, or the request path is not reproduced. Use a server-level
  run (a `{{runtime}} compute submit --serve` job and its client) and its logs.
- **Imbalance that depends on real weights.** Weights are randomly initialized (`--load-format
  dummy`); routers are re-drawn (`ROUTER`, target and drafter) so experts are used, and every
  request gets its own token ids, true positions, KV pages and state slots, but the
  token-to-expert distribution is synthetic. The KV pages a batch reads are filled with finite
  values first (a never-written page holds whatever the pool's memory held, which made the forward
  NaN), and the drafter replay's zero hidden-state inputs (`run_dp_dummy_draft`) are swapped for
  seeded random ones (`_NonZeroHidden`), so the drafter's MoE routes per token too. Expert-load imbalance across EP ranks, hot experts, routing skew,
  and anything that depends on the trained weights' statistics (including MTP acceptance) do not
  reflect the real model. Debug those on a run with real weights.

## Sharding and shapes: mirroring the served slice onto the small one

Serving runs its parallelism on the job's slice, e.g. TP16 × DCP16 × EP16 on 16 cores. The twin
runs the **same parallelism on the small slice** (TP8 × DCP8 × EP8 on the 8 cores of one 2x2x1).
A smaller world alone would double every per-rank tensor. So `project` scales every model
dimension that is split across ranks by the same ratio (here 8/16) in `hf_overrides`, keeping
each rank's share exactly the served one: `global' / degree' = global / degree`.
- **Scaled with the world:** query heads (64 → 32: 4 per rank either way), routed experts and
  top-k together (256 / top-8 → 128 / top-4: 16 experts and T × 8/16 = T × 4/8 (token, expert)
  pairs per rank), dense FFN width (12288 → 6144: 768 per rank).
- **Context:** × 0.5, so a rank holds L / 16 KV tokens for a context of L in both.
- **Tokens per step:** not split across ranks (every rank sees all T tokens), so they stay as
  served.
- **KV page size and pool:** the served block size (e.g. 256 tokens per page), with enough pages
  that the per-rank KV pool matches serving, so page-granular work matches too. The prefix-cache
  block (block × DCP) comes out smaller in the twin, but `forward_run` has no scheduler or prefix
  cache; `points` applies the served cache rule before the run.
- **Layers:** one per layer kind, in the served order (GLM-5.2: 78 + MTP → 3). The count comes from
  the model's layer-kind rule, not a constant.

What cannot match on fewer cores, and is listed as a deviation in the twin config and the report:
- collectives span 8 ranks on one host, not 16 across hosts;
- attention and the indexer merge work over the DCP group's heads and candidates (8-way, not 16);
- shared-expert width per rank;
- routing statistics (experts and top-k are scaled together);
- with MTP IndexShare, the served drafter's first draft step computes its own sparse top-k, but
  the drafter replay `forward_run` times has no context, so that top-k is missing from the measured
  drafter. The projection adds it as a *modelled* term: a twin layer with an indexer minus the
  same FFN kind without one, at the same step.

This is why a projection to the served model is always labelled *modelled*.

**The KV fill decides who owns the top-k.** Dummy weights mean the cached context is whatever
`forward_run._fill_pages` writes. With a sparse-attention indexer under DCP, tied indexer scores make
the top-k break ties by index, so one DCP rank owns nearly every slot and becomes a straggler every
later collective waits for. The fill therefore writes every KV tensor, integer-stored fp8 caches
included (u8 indexer K, u32 SparseCore MLA layout), with distinct values per page and a seed per rank
({{script:workload-microbenchmark/tests/test_forward_run.py}} `FillPages`, with a positive control). Before 2026-10-02 it skipped integer caches
and tiled 16 pages with one seed on every rank: GLM-5.2 runs from then show DCP rank 0 owning ~all
slots (rank 0's sparse-MLA 90 µs vs 8 µs elsewhere), which misjudges any ownership-sensitive change
(sparse-MLA width or overflow paths); ops that walk the full top-k width were unaffected.

## Two modes: scaled (2x2x1) and mirror (the served slice)

The **twin** is always the served model shrunk to one layer per layer kind. The mode says how it
fits the slice it runs on:

| | **scaled** (default, `--mode scaled`) | **mirror** (`--mode mirror`) |
| --- | --- | --- |
| slice | one 2x2x1 host, 8 cores (`--twin-world 8`) | the served slice, e.g. 2x2x2, 16 cores |
| model | split dims scaled by `twin world / served world` so every rank keeps its served shape | nothing scaled: served heads, experts, top-k, widths, context |
| sharding | the served one at the smaller degree (TP16 × DCP16 → TP8 × DCP8) | the served one as is |
| deviations | collectives over 8 ranks on one host; DCP merge width; shared-expert width; routing statistics (see below) | none from sharding; the per-rank table is all `match` |
| runs on | one host: one `{{runtime}} compute submit` job (`forward-cmd`) | two hosts on one Ray cluster; `forward-cmd` does not run it (see below) |
| availability | easy to get | scarcer pool; one two-host job per run |
| use it for | kernel work, checking an op got faster, most cost-model A/Bs | sharding choices, collective and DCP-width effects, an accurate served projection, calibrating scaled mode |

**Which mode to use:**
- **Prefer scaled mode (the 2x2x1 twin) when writing or tuning a kernel,** and whenever 2x2x2 or
  larger pools are busy. Per-rank shapes match serving, so a kernel sees its real inputs, and the
  pool is easy to get. It is also enough to check that an op got faster.
- **Use mirror mode for accurate roofline or served-time analysis, and for sharding questions**
  (collective sizes and degrees, DCP merge width, cross-host traffic, comparing shardings at their
  true degree), **when the current pool can return the results in time.** If the 2x2x2 queue is
  long, run scaled mode now and calibrate it with mirror later; don't block on the bigger pool.

- **Pick the mode at `derive` or `project`:** `derive … --mode mirror` writes `<name>.twin16.json`
  beside `<name>.twin8.json`. `points` on it writes its plans under
  `<sharding>+mirror-mix-<job>/`, so the two modes' plans never overwrite each other.
- **`forward-cmd` reads the mode from the plan.** A mirror plan whose twin spans two hosts needs
  one Ray cluster across both. On each host, a launcher runs `forward_run.py --node-prep` before
  `ray start`: it restores the compile cache and exports the run's env, so host 1's Ray workers get
  it too. Host 0 then runs `forward_run.py` with `--distributed-executor-backend ray`, added outside
  the compile-cache key. The recipe `forward-cmd` writes runs on one host, so `forward-cmd`
  refuses a two-host plan (exit 2). The two-host launcher of the source was specific to its job platform,
  which has no counterpart in this package. A mirror plan of a one-host served slice (twin world 8
  or less) runs as usual. Check with `{{runtime}} compute inspect` how many cores one host of the
  target has.
- **Mirror-mode limits, for now:**
  - the compile cache is saved from host 0 only, so host 1 may recompile on every run;
  - the profile and `result.json` come from host 0;
  - only host 0's profile ranks (0–7) are copied to the outputs; rank 0 is what `collect` reads.
- **Smoke test passed** (`j-f96adeda-ccb5-4a6e-b5b2`, cell `mixed-b2-p0-50`): Ray on both hosts,
  layer census 6/6, and the served projection within 8% of the source job (mixed 1,448 vs 1,367 ms
  measured; decode 31.5 vs 34.2 ms).

## Modes

Every mode takes the global options `--work <work>` and `--jobs <dir>` (fetched job directories,
default `<work>/jobs`) before the mode name:

```bash
cli.py onboard  <MODEL_DIR> --job <job-id> [--name N]    # reference config; reviewed before use
cli.py derive   <MODEL_DIR> --job <job-id> [--override k=v ...] [--mode scaled|mirror]   # working config -> <work>/
cli.py project  <config.json> [--twin-world 8 | --mode mirror]  # real workload -> twin config (onboard/derive do it too)
cli.py show     <config.json>
cli.py points   <twin.json> [--override dcp=8 ...] [--no-mtp] [--min-share 0.002] \
                [--point NAME=cached/input/output@C ... [--with-mix]] [--top K | --combined] \
                [--sampling | --no-sampling]   # step mix -> one plan per cell (--top/--combined: one plan)
cli.py mix-rollup --plans <dir> <job-id> ... [--baseline <ids>]   # the cost model -> cost_model.json
cli.py forward-cmd --steps <f.steps.json> [--image <path|job> | --interpreter <python>] [--repeats N] [--no-profile] \
                   --compute "{{runtime}} compute"   # writes <plan>.recipe.py, prints the compute submit command
                                             # (default: the source job's image; profile on)
cli.py collect <job-id> --steps <f.steps.json>   # after job artifacts: files the cache, prints
                                 # phase/step times, splits the profile, writes evidence.json
cli.py cache-pull <job-id>       # file the job's compile cache under ~/.xla_cache
cli.py selftest                  # re-derives every reference config and checks its bars
python3 -P -m unittest discover -s {{script:workload-microbenchmark/tests}}   # CPU tests
cli.py run ...                   # L1 A/B driver: NOT BUILT YET
```

- **Before any mode,** fetch the job's files into `<work>/jobs/<job-id>/`:
  `{{runtime}} compute job artifacts --job-id <job-id> --out-dir <work>/jobs/<job-id>`. A source
  serving job that returns its outputs as one archive needs one more step: also run
  `tar -xzf <work>/jobs/<job-id>/outputs.tgz -C <work>/jobs/<job-id>`. The CLI
  never calls `{{runtime}}`. A source serving job's directory must hold its Kubernetes JobSet
  manifest as `recipe.yaml` (the served command and env are read from it), and under `outputs/`
  its aiperf records and metrics and its `vllm.log`, or the server log under `logs/`.
- **A server this package started** (`{{runtime}} compute submit --serve`) needs no
  `recipe.yaml`: it is read from vLLM's own log.
  - **Args:** the API server's `non-default args: {...}` line, as `vllm serve` flags.
  - **Prefix caching:** the engine config line (`enable_prefix_caching=`), when the flag was left
    at vLLM's default.
  - **Page size:** vllm_torchtpu's `Using KV cache block size: N` line.
  - **Checkpoint config:** `config.json` from the model path in those args, or this host's Hugging
    Face cache.
  - **Env:** vLLM logs none. Env the model always needs comes from its model.py `TWIN_ENV`
    (source `model` in the ENV block); anything else is `--override env.NAME=value`.

  Fetch the server job's artifacts into `<work>/jobs/<server-job-id>/`, and the client job's into
  its `outputs/`:
  - **aiperf** (`profile_export.jsonl`, per-request records): sets the measured window; the step
    mix is built as for any job.
  - **the package's own client** ({{script:common/serving_benchmark.py}}, `summary.json`, no
    per-request records): each cell is fixed lengths at a fixed concurrency with every prompt new,
    so there is no step mix to recover. The snapshot keeps the cells (`summary_cells`) and models
    the requests from them (a closed loop, labelled), and `points` makes one point per cell
    (`<cell>=0/<isl>/<osl>@<concurrency>`). The server counters need the aiperf window, so MTP
    acceptance falls back to the ceiling (1 + spec tokens). The pod cannot see its image, so
  `forward-cmd` needs `--image` (container backends) or `--interpreter` (host backends). ITL
  percentiles are not logged, so the G0 closure is skipped.
- **Overrides:** `tp`, `dp`, `dcp`, `pcp`, `ep`, `block_size`, `mnbt`, `mns`, `max_model_len`,
  `prefix_caching`, `spec_tokens`, …, or `env.<NAME>=<value>`. Unknown keys are an error.
- **An infeasible override is refused** (exit 3) with alternatives: the minimum DCP for the recorded
  peak, or the largest max-model-len that fits.

## The steps, and what each writes

1. **Read the job**: a snapshot of recipe, checkpoint config, requests and server metrics.
2. **Derive the real workload**: deployment, per-rank shapes, cells and counts (checked against
   the server), feasibility. Writes `<name>.json`.
3. **Project onto the small slice**: the twin rule (per-rank sizes kept, collectives across the
   twin's ranks) applied for `--twin-world` cores. Writes `<name>.twin8.json`:
   - per-rank table, real vs twin (`match` / `deviation` / `by design`);
   - full twin engine args and env;
   - memory per rank, real vs twin, with the ballast that would equalize them;
   - MTP coverage.
4. **Plan the step mix** (`points --top K`): the job's step mix ({{script:workload-microbenchmark/core/mix.py}}) → the K cells
   carrying the most busy time, as fixed shapes in **one** self-contained plan with the twin
   embedded. Writes `<work>/<MODEL_DIR>/workload-microbenchmark/steps/<sharding>[+mirror]-top<K>-<job>.steps.json`.
   (Without `--top`: one plan per cell, see "A run, end to end".)
5. **Run the plan as one job** (`forward_run.py --steps`, on the TPU host, through the recipe
   `forward-cmd` writes): boot the twin with dummy weights, compile the plan's buckets only, time
   every shape back to back and profile every one.
6. **Cost model** (`mix-rollup`): `T_mix = Σ steps × step time` over the plan's cells; with
   `--baseline`, the speedup, guardrails and noise. Writes `cost_model.json`.

## The step mix: the shapes to run, and their weights

**Why not P50 / P90 points.** A percentile point `cached / input / output @ C` runs C identical
requests per step. An agentic trace is not that. GLM-5.2 AgentX at client concurrency 4
(`j-58d59d77`): 54% of wall time is decode steps carrying **1–2** requests, 15% 3–4, 3% 5–8,
18% prefill chunks, 10% the client thinking. Its P50 / P90 points ran 7–8 requests per decode
step, which is 3% of that job, and never compiled the bucket-16 program that runs half of it.

**What `points` builds** ({{script:workload-microbenchmark/core/mix.py}}, from the snapshot):
- **Classify every second of the measured window.** The inputs are aiperf's 1-s scrape of
  `vllm:iteration_tokens_total` (observed when a step *ends*) and `num_requests_running`, kept in
  the snapshot as `step_slices`. A decoding request is 1+k tokens under MTP k (1 without), so each
  histogram bin up to mns × (1+k) tokens is the decode class of the requests it can hold
  (`mix.decode_classes`): under MTP 3 with mns 8, ≤8 / ≤16 / ≤32 tokens are **D1-2 / D3-4 / D5-8**;
  without MTP, ≤1 / ≤8 are **D1 / D2-8**. A longer step carries a prefill chunk (**mixed**). The
  bins are the histogram's (1, 8, 16, 32, 512), so a class can span several batch sizes; its batch
  comes from the requests decoding in its seconds. Class step times are fitted on fully busy decode-only seconds. The
  rest of a second is mixed when a long step is in flight, else idle (client think time, which no
  benchmark prices).
- **Validated** on `j-c964a8b6`, whose server also logged every scheduler step (corrected for async
  scheduling's one-step lag): within about 5 points, with the error toward mixed.
- **Cells.** Each class is split into three context bins (0–50%, 50–90%, 90–100% of its time, each
  at its time-weighted median context, from the requests' decode intervals), and into the two
  integer batches bracketing its mean D. A mixed cell's prefix is the prefilling request's cache
  hit under this deployment (the same rule as `cells.with_hit`), binned the same way. Its chunk
  fills the budget after its decode rows (`mnbt − B × (1+k)`), and its decode rows are also timed
  alone, so the chunk's own cost is the difference.
- **Weights.** Every cell carries its step count and its share of measured busy time. A result
  rolls up as `Σ steps × step time`, per class and in total, beside the job's measured busy time.
  `--min-share` (default 0.002) drops cells below that share; the kept share is printed (99.8% on
  `j-58d59d77`).
- **Buckets come from the job, not constants.** `points` rebuilds the served runner's bucket list
  from the job's mnbt, `TPU_TOKEN_BUCKET_EXTRA` and any explicit `compile_sizes`
  (`serving_token_paddings`), and each step takes the first bucket that fits. On `j-58d59d77`:
  `[16, 32, 4096]`.
- **One job: `--top K`** (K = 2–5). The heaviest cells by busy share go into one plan, run back
  to back and each profiled, so one job per arm covers most of busy time (63.5% for the top 5 on
  `j-58d59d77`). The cost model counts only those cells; the report states the coverage.
- **Full coverage: one plan per cell** (`points` without `--top`), one job per cell, launched in
  parallel. Every cell plan compiles the union of the mix's buckets, so all of them are one engine
  config and share one compile cache. `mix-rollup` sums them back into the workload.
  `--combined` puts every cell in one plan (one job; the profile holds only the heaviest cell of
  each class).
- **A snapshot without `step_slices`** (taken before this model, e.g. the `j-c964a8b6`
  reference) is refused with that reason. Re-snapshot the job (`derive` / `onboard`), or pass
  `--point`.
- **`--point NAME=cached/input/output@C`** is still accepted as an explicit what-if shape (one
  request with C−1 neighbours: its full chunks plus a C-request decode step). It replaces the
  mix unless `--with-mix`. It is not a model of the job.
- **A different sharding** is `--override` on `points` (e.g. `dcp=8`, `tp=8 dp=2`,
  `ep=false`). It re-derives the deploy (cache hit, chunking, feasibility) and re-projects the
  twin (real DCP8 → twin DCP4 at context × 0.5, so KV per rank stays L/8). The step counts stay
  the job's. An infeasible sharding is refused with alternatives.
- **MTP follows the source job.** If the job served with speculative decoding (e.g. MTP 3), the
  plan keeps the served speculative config and times the drafter's propose replay after every
  step; `points` turns this on by itself and says so. `--no-mtp` leaves the drafter out (the
  header records it); `--mtp` on a job without speculative decoding is refused.
- **Sampling follows the source job** (see "Sampling" below): on when it sampled non-greedily,
  `--no-sampling` leaves it out.
- **Results are twin ms** (N layers, per-rank exact). They sit beside the job's numbers and are
  not a substitute for them.

## Sampling: the step's sample stage

A served step does not end at the forward. `execute_model` computes the target LM head on the
sampled rows and `sample_tokens` samples: under MTP, `_sample_spec_verify_chunk` (the bonus token
through `_build_padded_sampling_params` + Gumbel `sample_from_logits`, the drafts through the
rejection sampler); without MTP, `sample_from_logits`. At temperature > 0 with top-k / top-p this
is not small (Qwen3.5 AgentX, served trace: 0.704 ms of a ~9.95 ms b1 decode step), and a sampler
change is invisible to a twin that times only `forward_model`.

- **Plan block.** `points` writes `sampling`: `{run, temperature, top_k, top_p, greedy, source,
  why}`. The params come from the source job's server log ("Default vLLM sampling parameters have
  been overridden by the model's generation_config.json: {...}", read into the snapshot as
  `sampling_defaults`, or from the synced log), else from model.py `SAMPLING`. `run` defaults to
  on when the job sampled non-greedily; `points --no-sampling` (or `forward_run --no-sampling`)
  turns it off. Unknown params: off, and `--sampling` is refused.
- **On the TPU host** (`forward_run._prepare_sampling` / `_sample_stage`): the runner's own functions.
  Per batch, once and outside any timed region: the params go into the persistent input batch
  rows `_build_padded_sampling_params` reads, and the MTP verify metadata comes from
  `spec_decode_manager.get_spec_decode_metadata` (as `_prepare_inputs` builds it: B requests ×
  (1 + k) rows, bonus rows padded to the request bucket, target rows to the token bucket). Per
  step, after the forward: `compute_selected_logits` → `_sample_spec_verify_chunk` (or
  `sample_from_logits`). Works on images whose `_build_padded_sampling_params` returns 3 tensors
  or 3 + the static `k_bound` ({{script:workload-microbenchmark/tests/test_sampling.py}} `SamplingGlue`). Not included: logprobs,
  structured output, the mamba read-offset update after the verify.
- **Timed apart** from the forward: `sampling_ms` per step (wall), next to `median_after_first`.
- **In the profile:** the stage runs between two marker programs (outputs `s32[7919]` and
  `s32[7907]`, `SAMPLING_MARKS`); `xprof_steps` books the modules between them as the step's
  `sampling_ms` / `sampling_busy_ms` / `sampling_modules`, apart from the backbone and the drafter.
  torch-tpu ran both marker sizes through one program (the trace shows `s32[7919]` twice), so the
  markers are read as a toggle: the first after a backbone opens, the second closes. A step whose
  plan sampled but has fewer than 2 markers gets `sampling_error`.
- **Warm-up:** every timed shape runs forward + stage 3× untimed before repeat 0 (the stage's
  programs compile per shape; unwarmed, they slowed repeat 0's forwards to 15–275 ms).
- **Measured** (Qwen3.5 AgentX top5, B = 1 decode, device busy per step): 0.86 ms on 0fcc1f696
  (LM head 0.12, bonus+target gather 0.06, bonus `sample_from_logits` 0.35, rejection sampler
  0.30, RNG/index ~0.03), 0.37 ms on the chunked-top-k stack c8ee856e5 (sampler calls 0.068 +
  0.092). The stage's host wall is ~1.9 ms on both: the win is visible on device time only.
- **In the cost model:** the stage's device busy once per step on `served` and `served_device`
  (never × n_layers), its wall on `twin`. `compare` flags arms that differ in whether it ran
  (`sampling_mismatch`).

## Cost model: one number to optimize, and when to trust it

**The number.** The step mix turns the workload into cells, each with a measured step count from
the source job. The cost of a run is the job's engine-busy time at the twin's shapes:

```
T_mix = Σ over cells i of  steps_i × t_i        t_i = the cell's twin step time, median over repeats
```

Under MTP, `t_i` includes the drafter, once per step and never multiplied by a layer count. On
the twin basis that is the propose replay's wall, timed right after the step (`draft_ms`). On the
served basis it is the drafter's **device busy** from the profile (`draft_busy_ms`, the union of
its ops): the replay is driven from the host module by module, and its wall (3.9-4.8 ms on
Qwen3.5) is mostly host gaps a served drafter does not have (served busy 1.29 ms). `cost_model.json`
keeps every part per cell (`backbone_ms`, `draft_wall_ms`, `draft_busy_ms`, `drafter_basis`) and
the rule (`drafter_rule`); `compare` flags arms that differ in whether the drafter ran
(`drafter_mismatch`).

**An A/B is a ratio** of two runs of the same cells (e.g. the incumbent image vs a candidate):

```
speedup = T_mix(baseline) / T_mix(candidate)
```

**Two bases; `served` is primary.** The twin has one layer per kind, so its wall step does
**not** carry over by ratio:
- a saving inside the layers is `n_layers` times larger in serving (GLM-5.2: 3 / 18 / 57 per
  kind), but only once in the twin;
- a per-step saving (host dispatch, embedding, step entry and exit) is the same absolute amount,
  and so is a much larger share of the twin's step;
- the twin weights every layer kind equally; serving does not.

So `mix-rollup` rebuilds each cell's step from its profile ({{script:workload-microbenchmark/core/costmodel.py}}, *modelled*):

```
t_i_served = fixed_i + Σ_kind n_kind × layer_kind,i + drafter_busy_i   n_kind from the model
fixed_i    = backbone_wall_i − Σ_kind layer_kind,i       host dispatch + before/after, counted once
```

- **Per-step ops are not layer work.** Some ops run once per served step but land inside a twin
  layer (Qwen3.5: the RoPE cache copy, `rpa_metadata_schedule`, the metadata `copy-done`s, the
  mixed step's mRoPE gather). The model lists them (`TRACE["per_step_ops"]`); `xprof_steps`
  subtracts them from `layer_kind,i`, so they fall into `fixed_i` and count once.

- A cell with no profile of its own goes through its class's profiled cell: its layer body is
  its wall minus that cell's `fixed`, scaled by that cell's served/twin layer ratio. A class with
  no profiled cell is listed missing; nothing is borrowed across classes.
- **One profile is one step, and its per-layer noise is multiplied by the layer count.** On the
  GLM-5.2 mirror twin with EP, the decode `moe` layer flipped between 0.275 and 0.323 ms from
  one profile to the next (dummy-routing imbalance), which is ±9% of the served decode step.
  So profile each cell several times (`forward-cmd --profile-iters 5`), and pool profiles from
  more jobs of the same arm (`mix-rollup --profiles`, `--baseline-profiles`). The per-kind median
  is used, and the profile-to-profile spread of `T_mix` (`profile_spread`) joins the repeat spread
  in the significance test.
- **A repeat's deviation from the median counts once, unscaled.** The repeats share the
  profiled layers (stable to a few µs), so their difference is per-step host jitter. Scaling it by
  the served/twin layer ratio (~13× on GLM-5.2) once inflated a 0.3% repeat spread to 6%.
  Boot-to-boot layer noise is a separate term: profile each arm on 2 boots and pool them, and
  judge an A/B against the larger of the repeat spread, `profile_spread`, and an A/A of two
  baseline boots (1.6% on the GLM-5.2 AgentX mirror twin with MTP, from dummy-routing modes).
- `basis: twin` (the wall step as run) is kept beside it in `cost_model.json`. The two disagree
  whenever the change is per-layer or per-step, which is most changes.

**Third basis, `served_device`: primary for kernel changes.** `served` takes `fixed` from the host
wall (`backbone wall − layers`), so every cell carries the host's per-step jitter (±0.3 ms at
~10 ms on Qwen3.5). That hid a measured −1.4% layer-time kernel win (RPA v4: device basis −1.43%,
wall basis 1.0012×). `served_device` takes every term from device time:

```
t_i_device = before busy + after busy + per-step ops (once) + Σ_kind n_kind × layer_kind,i
             + drafter device busy + sampling device busy          per profile, then pooled
```

- Pooled like `served` (median within a job, then over jobs). `by_repeat` holds one value per
  profile, so `t_mix_by_repeat_s` is a T_mix per profile and `spread_pct` its range.
- `compare.significance`: Welch's t on the two arms' per-profile T_mix samples (t, df, two-sided
  p, a 95% interval on the speedup). Profiles of one boot share its dummy-routing mode, so the
  test alone is anticonservative: an A/A of two baseline boots (j-3e69c2dd vs j-36af7327, Qwen3.5
  top5) gave 0.9963× at p = 0.014. So `significant` = p < 0.05 **and** |speedup − 1| beats the
  boot-noise floor: an arm's own per-boot spread (`boot_spread`, with ≥ 2 boots) or
  `mix-rollup --noise-floor-pct` from an A/A (0.37% there). With neither, the output says
  `boot_noise: unmeasured`.
- `mix-rollup` prints all three bases and marks `primary_for`: `served_device` for kernel /
  device-time changes, `served` (or `twin`) for host / per-step changes, which device time cannot
  see. A cell without its own profile goes through its class's profiled cell, scaled by the
  backbone wall ratio (labelled). An `xprof_steps.json` written before this basis (no
  `before_busy_ms`) gives no device sample: re-run `collect`.

**What it gets right.** Each cell counts by how much of the job it is. On the concurrency-4
GLM-5.2 AgentX mix, 10% off the 1–2-request decode step (~60% of busy time) is worth about 6%,
and 10% off the 5–8-request step (~4%) is worth about 0.4%. A change that speeds one class up and
slows another shows its net effect.

**Always reported beside it:**

| | what | why |
| --- | --- | --- |
| **per-class delta** | seconds each class saved or cost, and its share of the net change | says *where* the win is |
| **guardrails** | the heaviest decode cell (typical TPOT) and every mixed cell (TTFT, and the TPOT tail: decodes wait through a prefill step), baseline vs candidate | a lower `T_mix` bought by slower prefill steps worsens latency; a regressed guardrail is named |
| **noise** | `T_mix` per repeat and its spread; `significant` when `|speedup − 1|` beats the larger arm's spread | a margin inside the spread is two candidates, not a win |
| **coverage** | the share of busy time the measured cells cover, and any cell without a job | a missing cell is listed, never counted as zero |
| **served gain (modelled)** | `(busy + idle) / (busy * (covered_share / speedup + 1 - covered_share) + idle)` with the source job's measured busy and idle | throughput also contains unmeasured cells and the client's think time, which no kernel shortens |

**When it is valid.** `T_mix` holds the job's step counts fixed, so it decides changes that make a
step of the same shape cheaper: kernels, collectives, env flags, layouts. It does **not** decide
changes that alter the schedule, because those change how many steps of each shape run:
- chunk size or `mnbt`;
- the number of speculative tokens, or anything that changes MTP acceptance (the same output then
  takes a different number of decode steps);
- a re-sharding that changes admission or the running set;
- scheduler policy.
For those, `T_mix` is at most a per-step hint, and the verdict needs L2 (fixed-work server cells)
or L3 (the real run).

**It is modelled, not measured.** `T_mix` is a microbenchmark prediction. An optimization is only
`accepted` after {{skill:critical-review}} and an accuracy gate (`serving_eval.py`, as {{skill:close-gaps}} runs it); the end-to-end run is the last word.

## Image and environment: audit both before trusting a number

How the sharding communicates is set by env and `LIBTPU_INIT_ARGS`, not only by `--tensor-parallel-size`
(hierarchical all-reduce, SparseCore collective offload, the fused-EP row gather, DCP q-replicate).
The twin's env has these sources, and every output says which one each key came from.

| source | what | where you see it |
| --- | --- | --- |
| **container** | the recipe's `env:` list | `points` ENV block, `source` column |
| **script** | what the recipe script sets before `vllm serve`: `export`s, `if` branches, `${X:-default}` | same. The script is evaluated in restricted bash (nothing runs, nothing is written); `serve_parse: bash-eval` in the snapshot proves it was |
| **override** | `--override env.NAME=value` | same |
| **image** | the Dockerfile's env and vllm-torchtpu's `envs.py` defaults | only inside the job: `result.json` → `effective_env`, each key tagged `plan` / `forward_run` / `image/pod` |

- **The code under test is always named: an image or an environment, by backend.** Check the
  profile's backend first (`{{runtime}} compute inspect`):
  - **Container backend (`gke_job`): `--image`.** Default `--image job` is the image the
    source job ran (`source_image` in the plan), so the run measures the code the workload was
    recorded on. An A/B names each arm's build: `--image <repo:tag@sha256:…>`. A server read from
    its vLLM log has no image record, so its plans need an explicit `--image`.
  - **Host backends (`local`, `gce_tpu_vm`, `gke_pod`): `--interpreter`.** These run the job on the
    host and **ignore `--image`**. An arm is the Python environment holding that arm's vllm-torchtpu:
    `--interpreter <python>`, passed to `compute submit --interpreter`. `forward-cmd` refuses both
    together.
  - **Never a moving tag (`:latest`)** as an arm: it is not a fixed build, and it is not ours to retag.
- **Image tags are pinned to their digest.** `forward-cmd` resolves the image to `…@sha256:…`
  through Artifact Registry. The job runs that exact image, its recipe records it, and the compile
  cache is keyed by it, so a rebuilt tag never reuses an older build's cache. If the registry
  can't be read, it warns and runs the tag as is. An `--interpreter` arm keys the cache by its path.
- **A different image runs the source job's args and env.** Flags the image's vLLM does not know
  are translated (`--no-dcp-q-replicate` → `VLLM_DCP_Q_REPLICATE=0` on newer vLLM) and listed in
  `result.json` → `compat`. Anything else fails at argument parsing; it does not pass silently.
- **Plumbing is dropped, and listed:** the job platform's output paths and bookkeeping vars,
  Ray/`TPU_PROCESS_*` multi-host wiring,
  `TPU_NODE_ID`/`TPU_WORKER_ID`, HF cache, `MODEL_PATH`. The source job may run on more hosts than
  the twin, so its multi-host collective behaviour is a deviation by design, not something the env
  can copy.
- **A sharding override does not recompute the env that depends on sharding.** Recipe scripts
  compute those keys FROM the sharding. `points --override tp=… / dcp=… / ep=…` marks each carried
  sharding key `REVIEW` (the list is `derive.SHARDING_ENV`). Take the value for the new sharding
  from a served config of that sharding, or set it with `--override env.NAME=...`. A plan with open
  REVIEW lines is not a measurement of the new sharding.
- **To audit a run,** put three things side by side: the source job's env (`points` ENV block), the
  twin's (`result.json` → `effective_env`) and `compat`. Every difference must be plumbing, an
  override you passed, or `image/pod` for an image you picked on purpose.

## Change → build → microbenchmark

The workflow for measuring a vllm-torchtpu change (a kernel, a collective, an env default) against
the workload. The two arms differ **only by the change**: both are built the same way from
vllm-torchtpu commits, and both run the same cell plans. An arm is an image on a container
backend and a Python environment on a host backend (see "Image and environment").

**0. The representative config** (once per source job; reuse it for every change):
```bash
{{runtime}} compute job artifacts --job-id <job-id> --out-dir <work>/jobs/<job-id>   # the latest full benchmark
cli.py derive <MODEL_DIR> --job <job-id>               # snapshot -> config -> twin
cli.py points <twin.json>                              # step mix -> <dir>/<cell>.steps.json
```

**1. Make the change** in a vllm-torchtpu checkout (`<vllm-torchtpu>`), on a branch off the
commit you are comparing against (`BASE`), and commit it. If your image build clones from GitHub,
**push the branch**: a local-only commit fails the checkout.

**2. Preflight** the change: the cheapest check that it does what it
claims and is still correct (unit test, op microbenchmark). Don't build or queue TPU jobs on an
unchecked change.

**3. Build both arms** the same way, one per commit (`BASE`, then `CHANGE`):
- **Container backend:** one image per commit, from the build that makes your serving images. This
  package ships no image build; if the caller has none and supplies no images, that is a blocker.
  Record each `repo:tag@sha256:…`; the report names them.
- **Host backend:** one Python environment per commit on the host, each a copy of the profile's
  `python_interpreter` environment with vllm-torchtpu installed from that commit. Record each
  interpreter path and commit; `result.json` → `image_source` confirms what each job ran.
- To compare against what the source job ran, the baseline can be `--image job` (container
  backends). The arms then differ by everything between that image and your commit, not only
  your change; say so in the report.

**4. Run one job per arm, both at once** (on a host backend with one TPU host, one after the
other: each job holds the chips). The same `--top K` plan on each arm. Each arm has its own
compile cache. A cold job is a complete measurement: compile
time is spent in engine init, outside the timed steps, so **no warm run is required**.
```bash
cli.py forward-cmd --steps <dir>/<plan>.steps.json --image <baseline repo:tag@sha256:…> --compute "{{runtime}} compute" \
    --recipe <dir>/<plan>.baseline.recipe.py
cli.py forward-cmd --steps <dir>/<plan>.steps.json --image <candidate repo:tag@sha256:…> --compute "{{runtime}} compute" \
    --recipe <dir>/<plan>.candidate.recipe.py
# host backend: --interpreter <baseline env>/bin/python3 and <candidate env>/bin/python3 instead of --image
```
Each prints a `{{runtime}} compute submit --recipe … --image …` (or `--interpreter …`) `--input …
--output wmb_out.tgz` command; run both.
- **`gke_job` needs a `staging_bucket`** for these jobs: inputs over 900 KiB and outputs over
  1 MiB travel through it, and a restored compile cache (up to ~2 GiB in) and `wmb_out.tgz` (the
  cache save and profiles, out) are both over. Without one the job cannot return its results;
  set it up first (the install guide's "Large files on `gke_job`"). Give each arm its own `--recipe`: both arms share the plan, and the default
recipe path is next to it. For full coverage, run the per-cell plans instead: one job per cell
per arm, all launched at once.
- **A failed job reports why it failed.** vLLM's engine is the process that compiles and
  runs the model. The engine can log a fatal error during start-up and then keep running.
  vLLM's own wait for the engine does not cover every stage of start-up, so such a job
  can hold its chips until someone cancels it. To prevent this, the recipe runs `forward_run.py` under `supervise`.
  `supervise` stops the job in two cases:
  - vLLM logs a start-up error, and the job has not exited one minute later.
  - Engine start-up lasts longer than the limit that `forward-cmd --engine-start-timeout`
    sets. By default, the limit is the plan's `VLLM_ENGINE_READY_TIMEOUT_S` if it sets
    one, and otherwise 45 minutes. Set the limit longer than a cold compile of the
    plan takes. A cold compile cannot reuse cached compile results, and it can print
    nothing for many minutes. If the plan does not set `VLLM_ENGINE_READY_TIMEOUT_S`,
    vLLM's own default wait can end a slow start-up first, so set it in the plan's
    `twin.env` for a plan that compiles for a long time.

  A job that ends without `result.json` returns `wmb/forward/failure.json` instead. The
  file records the reason that the job ended, the cause of the failure, and the last lines
  of the log. The cause is the first error line of the job, together with the four lines
  that follow it. If the engine became ready, only the lines after that point count.
  `collect` prints the cause. Read the cause before you resubmit the job, because the
  same plan on the same image is likely to fail the same way.

**5. Collect and score.** Fetch (`{{runtime}} compute job artifacts --job-id <job-id> --out-dir
<work>/jobs/<job-id>`) and `collect` every job, have the profiles read ("Reading the profile"),
then run the cost model over both arms:
```bash
cli.py mix-rollup --plans <dir> <candidate job ids…> --baseline <baseline ids,…>
```
It writes `cost_model.json`: the speedup, its significance against the repeat spread, per-class
deltas, the guardrails, and the modelled served gain (see "Cost model").

**6. Report and gate.** Write REPORT.md from `cost_model.json`, `result.json`, `xprof_steps.json`
and `evidence.json` (see below). A microbenchmark win is modelled: run {{skill:critical-review}} on
it, and take the change to the end-to-end run (a `{{runtime}} compute submit --serve` job and its client) and
its `serving_eval.py` accuracy gate before calling it `accepted`. If the change alters the schedule (chunk size, `mnbt`, speculative tokens,
admission), `T_mix` cannot decide it; go to L2 / L3 directly.

## A run, end to end

**One job** (`--top K`, the standard run): the K (2–5) cells carrying the most busy time, as
fixed shapes in one plan, run back to back and every one profiled (`--iters 10 --repeats 2
--profile-iters 5` by default). The cost model counts only those K cells, and the report states
their coverage of busy.
```bash
cli.py points <twin.json> --top 5 [--override ...]   # one plan; move it into its own dir
cli.py forward-cmd --steps <dir>/<plan>.steps.json [--image <repo:tag@sha256:…>] --compute "{{runtime}} compute"
JOB_ID=$(<the compute submit command it printed>)                              # one job
{{runtime}} compute job wait --job-id $JOB_ID
{{runtime}} compute job artifacts --job-id $JOB_ID --out-dir <work>/jobs/$JOB_ID
cli.py collect $JOB_ID --steps <dir>/<plan>.steps.json          # times, cache, evidence.json
# have tpu-diagnostics read the trace ("Reading the profile"), then collect again: profile split
cli.py mix-rollup --plans <dir>/<plan>.steps.json $JOB_ID [--baseline <id>]   # cost model -> cost_model.json
# then the agent writes REPORT.md over the job (see below)
```

**Full coverage, one job per cell** (`points` without `--top`), when the cells outside the top K
matter:
```bash
cli.py points <twin.json> [--override ...]   # one plan per cell under <dir>
for f in <dir>/*.steps.json; do cli.py forward-cmd --steps $f --repeats 3 --compute "{{runtime}} compute"; done
# run every compute submit command it printed: launch all at once
{{runtime}} compute job artifacts --job-id <job-id> --out-dir <work>/jobs/<job-id>   # each, when finished
cli.py collect <job-id> --steps <dir>/<cell>.steps.json         # each; again after the profile is read
cli.py mix-rollup --plans <dir> <job-id> ... [--baseline <ids>]  # exits 1 if a cell has no job
```

| run shape | jobs | profiled | use it for |
| --- | --- | --- | --- |
| `--top K` (standard) | one | every one of the K | the heavy cells; an A/B is one job per arm |
| one plan per cell | one per cell, launched at once | every cell, its own job | the whole mix, full coverage |
| `--combined` | one | the heaviest cell per class | every cell timed, one profile per class |

- **No warm run.** Every job runs cold. Compile happens in engine init, before the timed steps,
  so a cold job's step times are as valid as a warm one's; a cold job costs about 7 min of init
  instead of about 1. Do not add a separate warm rerun to a measurement, and with per-cell plans
  do not hold the other cells back behind one warming cell.
- **Each profiled step is labelled** `<cell>/decode` or `<cell>/mixed/chunk0` in `profile_order`.
- **`--point` plans** run one job per point, sharing one compile cache.

**Typical cost of one job** (GLM-5.2 twin, backbone, no `--mtp`; about 50% more with it):

| phase (in `forward_run`, host 0) | mirror `--top 5`, cold (s) | mirror `--top 5`, cache restored (s) |
| --- | --- | --- |
| import vllm | 16.5 | 6.8 |
| engine init (weights, KV, compile of buckets `[16, 4096]`) | 253.8 | 204.1 |
| timed steps, 2 repeats × 10 iterations × 5 shapes | about 1.8 | about 1.8 |
| profile window, 30 profiled steps (5 per shape) | 73.0 | 74.5 |
| cache save | 14.9 | — |
| **total** | **361.5** | **288.7** |

- **Compile and the profile window are the cost**; the timed steps are seconds. Fewer profiled
  steps (`--profile-iters`) is the lever after compile.
- **A restored cache saves little in mirror mode**: about 50 s of init, because the restored boot
  rebuilds its graphs under its own key and host 1 gets no saved cache. That is one more reason
  not to run a warm job first.
- Scaled mode on one host (2x2x1), measured earlier with buckets `[32, 4096]`: about 7 min of
  engine init cold, about 40 s fully warm.
- Add 1–2 min of job creation (uploading the inputs and any cache) and the queue.

## REPORT.md: every measurement ends in one, written by the agent

A run without a report is not handed over. **The agent writes REPORT.md**; no script generates it.
What a report has to explain depends on the run: what changed, what the numbers show, which gap
matters, and which env var or deviation bears on this particular question. Code only gathers
the facts. Put it at `{{runs}}/<run_id>/artifacts/workload-microbenchmark/<job ids>/REPORT.md`, and
write one report for jobs that are meant to be compared: the cells of one step mix, and the same
mix on two images or with and without `--mtp`.

The report describes where the time went; it does not rank what to fix. For that, run
{{skill:decompose-optimization}} on the job: it reads `evidence.json`, `xprof_steps.json` and
`plan.steps.json`, censuses each profiled step separately, and ranks candidates by their served
cost (per-layer × served layer count, weighted by phase share of served wall). The top ops listed
per layer in `xprof_steps.json` are raw material, not a ranking.

**Read, per job** (all written by `collect`, next to the job's `result.json`):

| file | what it holds |
| --- | --- |
| `result.json` | cache state (`cache.warm`), phase times (`times_s`), every repeat's step times (`repeats`, merged in `points` with `by_repeat` / `spread_pct`; per cell `twin_model`: step, chunk's own cost, `steps × step`), the roll-up `mix` (`Σ steps × t` per class vs the job's measured busy), drafter times, the env the pod actually ran with (`effective_env`), compile-key inputs, the image's source record (`image_source`) |
| `xprof_steps.json` | the profiled steps split into layers: device ms per segment, top ops with output ← operand shapes, collectives with shapes, the layer census; served projection (`served`) and the source job's measured numbers |
| `evidence.json` | layer kinds, served counts and **served layer ids** (from `model.py`); sharding labels, `hf_overrides`, per-rank served-vs-twin table and deviations; the layer census check that tests the boundary rule; the HLO op categories (from {{agent:tpu-diagnostics}}'s `hlo_categories.json`); SparseCore offload threshold with every collective's twin and served size and any flip; `image_source` |
| `cost_model.json` | written by `mix-rollup` next to the cell plans: `T_mix` per arm (cells, classes, per-repeat values, spread, coverage, missing cells) and, with `--baseline`, `compare` (speedup, significance, per-class deltas, guardrails, modelled served gain) |
| `plan.steps.json` | the plan the job ran: the step mix (`mix`: measured shares, class step times and counts, share kept) and one entry per cell (`cell`: class, batch, context, chunk, prefix, steps, share), what each batch contains, buckets, twin env |

For "what does this env var / op / shape do", read the code the image ran (the functions saved in
`image_source/`, or vllm-torchtpu at the image's commit) and cite `file:line`. Don't guess.

**Reproduce: always the first section**, before the answers. Someone who reruns this later has
only this section to go on; `<work>` is not tracked, and a separate `JOBS.md` gets lost.
- **How the plans were made**: the source job id, the `derive`/`project` and `points` commands
  verbatim (every `--override`, `--no-mtp`, `--mode`), and this package's version
  (`{{package_tag}}`).
- **One row per job**: the cell, its shape in served terms (`2 requests × 4 tokens @ 76,994 /
  349,092`, or `4,088-token chunk @ 16,384 ctx + 2 decode requests`), its weight or step count, the
  job id, the plan file the job ran (its `plan.steps.json`), and the image digest. Include
  any job that failed or was rerun, marked as such.
- **Each job's command, as submitted**: the `{{runtime}} compute submit` command verbatim, and the
  recipe file it ran (`forward-cmd` writes it next to the plan, or at `--recipe`). When the
  commands differ only in `--steps`, show one in full and the per-job difference.
- **The roll-up**: the `mix-rollup` command, with every job id and `--baseline` it took.

**Must cover**, after Reproduce, in whatever order and depth the run calls for:
- **Layer count first.** "This is an N-layer per-rank twin, not the M-layer (+ MTP) served model",
  with N, the kinds and M taken from the evidence; they change with the model. Label every measured
  table "twin (N layers)".
- **Layers and sharding, before any number.**
  - Which layers the twin runs: each kind, what it contains, how many served layers it stands
    for, and their ids.
  - How the served sharding is mirrored onto the small slice (see "Sharding and shapes" above,
    with this plan's own numbers).
  - The per-rank served-vs-twin table and the deviations.
- **Setup:**
  - the step mix: the job's measured split of wall time by class, and every cell (batch × rows @
    context, chunk @ prefix, steps, share of busy), with the share the plan kept;
  - MTP drafter run or not, and what its replay covers;
  - the compiled buckets, explained. A bucket is the token count of one forward step; each step is
    padded to the smallest compiled bucket that fits. The served list is derived from the job
    (`serving_token_paddings` in the plan): sizes doubling from 16 up to the step budget mnbt,
    which is always the largest bucket, plus `TPU_TOKEN_BUCKET_EXTRA`, or the job's explicit
    `compile_sizes`. Mixed steps fill mnbt; decode lands in the smallest bucket ≥ B × (1 + k). Say
    whether any rule pads decode higher (EP-DP lockstep, only with DP > 1), and what the plan
    skips because its bucket is not compiled;
  - cache warm or cold;
  - repeats × iterations;
  - the image under test: the pinned digest, whether it is the source job's image or a named build
    (its vllm-torchtpu commit and base), and which vllm-torchtpu source it ran (`image_source`).
    When it differs from the source job's image, name the difference and any `compat` translations.
- **Environment:** the vars that pick code paths or collectives, with their implications for this
  run, and the offload check.
- **Time breakdown** of the job: restore, import, engine init, steps per repeat, profile, save, total.
- **Step times** with every repeat, median and spread. Describe each step so a reader needs no
  internals: new tokens, the cached context it reads in the served request's terms (never the
  twin's scaled count), and KV per rank.
- **The cost model** (from `cost_model.json`, see "Cost model"):
  - `T_mix` (twin s), its per-repeat values and spread, the coverage, and each class's share of it;
  - for an A/B: the speedup with its significance against the noise, the per-class deltas, every
    guardrail (flag a regression), and the modelled served output gain, labelled *modelled*;
  - whether the change is one `T_mix` can decide (same step counts) or one that needs L2/L3.
- **Profile step map**, one table per job: every block on the `XLA Modules` line, with its start
  (`start_ms` after the first step), device ms, `program` id, `profile_order` label and bar
  (decode / prefill_mixed). Every block has the same module name; the program id separates the
  buckets within one trace and changes between boots. The shape inside a bucket comes from
  position only. Name the trace file each table reads.
- **Per-layer breakdown** of the profiled steps:
  - device time per layer;
  - top ops with shapes, explained where they matter (e.g. `s32[8388608]` = 4,096 tokens × top-k 2,048);
  - collectives with shapes;
  - the evidence for the layer boundaries: the **layer census** (`layer_census_check`: each twin
    layer holds exactly the op families its kind should, per the model's `TRACE`), which fails if
    the boundary rule is wrong. The events come from one reader ({{agent:tpu-diagnostics}}'s
    export); there is no second, independent read of the trace.
- **Projection to the served model**, labelled *modelled*, next to the source job's *measured* numbers,
  with the gaps named.

**Rules:**
- Every table column header names its unit: (ms), (s), (tokens), (MiB), (%), (count).
- *Measured* (twin) and *modelled* (served) are never mixed in one column.
- A number without a job id behind it doesn't go in.
- Name steps the same way everywhere, by cell ("dec1-2-b1-p0-50", "mixed-b2-p50-90").

**Keep `<work>/jobs/<job-id>/` until the report is written.** `collect` reads the rank-0
`.xplane.pb`, `trace_events.json` and `hlo_categories.json` from there, and does not fetch a
deleted trace back.

## Reading the profile: ask tpu-diagnostics

This skill never reads a trace itself. When `collect` reports a profiled job without
`trace_events.json`, invoke {{agent:tpu-diagnostics}} with a `profiling-request` JSON object, all
paths fully resolved. `<res>` is the directory of the job's `result.json`, which `collect`
prints; the trace is the rank-0 `.xplane.pb` it names:

```json
{"prompt": "EXPORT ONLY. No analysis, no report, no findings. Read the rank-0 trace of workload-microbenchmark job <job-id> and write exactly two files, then hand back immediately.\n(1) <res>/trace_events.json: every event of the 'XLA Modules' and 'XLA Ops' lines of the /device:TPU:0 plane, with no event limit, as {\"events\": [{\"line\": <line name>, \"name\": <the full event name, HLO text included>, \"start_ns\": <start>, \"duration_ns\": <duration>}]}.\n(2) <res>/hlo_categories.json: the HLO op profile by category, IDLE included, as [{\"category\": ..., \"total_self_time_ms\": ..., \"op_count\": ..., \"top_op\": ...}].\nRules:\n- Do only what is needed for these two files. Do not compute metrics, rank ops, read graphs, open kernel source, or explain anything.\n- Wrap every shell command in `timeout 120`. Never run `xprof` without a subcommand (it starts a server and never exits); for help, use `<subcommand> --help` under `timeout 30`.\n- Total budget 10 minutes. If a file cannot be written in that time, stop and hand back `failed` with the last command and its error.\n- Do not install, upgrade or remove any package. If XProf is unusable, hand back `failed` at once.\n- Hand back as soon as both files exist: their absolute paths, sizes and event counts, nothing else.", "run_id": "<run_id>", "traces": [{"path": "<rank-0 .xplane.pb>", "label": "<job-id>"}]}
```

- **Send no `fields`, `top_ops` or `report_path`.** Each one asks for analysis beyond the two
  files, and the agent then goes looking for tools to do it. An earlier version of this request
  asked for step time, device busy and the top 20 ops. The export finished in about a minute, but
  the agent then spent over an hour hunting for those numbers, 72 minutes of it on one bare `xprof`
  call that never exited.
- **The caller enforces the budget; the prompt only asks for it.** If the delegation has not handed
  back after 10 minutes, stop it, and treat the job as having no profile split.
- **Never install XProf to unblock it.** If the agent reports XProf missing, tell the user. The
  interpreter it uses is usually the one the server runs on, and installing XProf there changes
  shared packages (numpy, protobuf, setuptools).
- **Check the files yourself; do not trust the hand-back.** Whatever the result's `status`, check
  that both files exist and are non-empty. The agent keeps session output under the run's
  artifacts, so it may write them there instead of `<res>`; copy them into `<res>`. Then run
  `collect` again: it splits the profile (`xprof_steps.json`) and rebuilds `evidence.json`.
- **Missing files mean no profile split.** If either file is missing (agent `failed`,
  `needs_input`, or stopped), say so in the report. The twin basis of the cost model still holds,
  and the served bases list the job's cells as missing.
- One request per job. Several jobs can go in one request, one `traces` entry each with its
  job id as `label`, when the prompt names each job's `<res>`.

## Compile cache (local, `~/.xla_cache`)

- **Layout:** `~/.xla_cache/<image-tag>/<key>/compile_cache.tgz`, where the key = hash(image +
  exact engine args, buckets included + the plan's env, since `LIBTPU_INIT_ARGS`/`XLA_FLAGS` change
  what is compiled).
- **Get the command:** `cli.py forward-cmd --steps <plan> [--image <path|job>]` writes the recipe
  and prints the `{{runtime}} compute submit` command. On a cache hit it adds the entry as an
  `--input` and passes `--cache-in` to `forward_run`. Entries are keyed by the pinned image digest;
  a new image starts cold.
- **After the job:** `{{runtime}} compute job artifacts --job-id <job> --out-dir <work>/jobs/<job>`,
  then `cli.py collect` (or `cache-pull`) files the new cache.
- **A new config compiles once.** vLLM's torch.compile dir is `sha256([env, config, code,
  compiler])`. env/config/compiler hashes are identical across boots, but `code_hash` (traced
  files) differs, deterministically, between a boot with no cache and a boot that restored one.
  So the second run can't load the first run's graphs and rebuilds them under its own key. Tier-3
  (the persistent cache of compiled XLA executables) makes that rebuild cheap:
  - run 1 (no cache) compiles from scratch; its save keeps the Tier-3 executables and drops
    its own torch.compile dir, which no later run can load;
  - run 2 (restores run 1) rebuilds under the restored-boot key, with every XLA executable served
    from Tier-3; its save keeps its own dir and drops the now-duplicate large Tier-3 binaries;
  - run 3 onward loads the graphs directly.

  `forward_run` writes the key inputs to `result.json` (`compile_key_factors`). With `--mtp` the
  drafter compiles into a torch.compile dir of its own, which follows the same rule.
- **Tier-3 settings:** `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` (inside the saved dir) with
  `TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK=0`. Without the second, a Tier-3
  hit still starts a full "backup" compile. Tier-3 also keeps the ~140 small eager programs a boot
  compiles (dummy weight init, copies). Neither setting is part of any compile key.
- **Format and size.** `compile_cache.tgz` is a plain tar of 16 gzip parts, written and read in
  parallel (the image has no pigz/zstd); single-gzip tarballs also restore. Each save stays under
  2 GiB, so that it uploads quickly as the next job's `--input`: it drops what the next run can't
  use (`_prune_for_save`, kinds of boot tracked in `wmb_manifest.json`):
  - a cold save drops its own torch.compile dirs;
  - a restored save drops Tier-3 binaries over 16 MB, plus leftover cold dirs (only after a key
    flip, never a dir the workers report using).

  AOT entries are kept: both kinds of boot load them. Typical sizes: about 0.7 GiB (cold save) and
  about 1 GiB (restored save) for the backbone, about 1.5 GiB with `--mtp`.
- **Bytecode:** `PYTHONPYCACHEPREFIX` points inside the saved dir. The image runs vLLM from source,
  so without it every job recompiles it (`import vllm` about 30 s, versus about 10 s with it).
- **Saving:** `forward-cmd` adds `--save-cache` only on a local-cache miss or with `--resave`.
  Every run captures a profile window by default: a `--top 5` plan profiles each shape 5 times
  (30 steps, about 75 s on the mirror twin); a per-cell plan profiles its cell once.
  `--no-profile` skips it for timing-only runs.

## Layout

In the package (the directory above `core/` of {{script:workload-microbenchmark/core/cli.py}}):

```
core/                    model-agnostic: jobdata (job -> snapshot), deploy (args -> per-rank shapes),
                         cells (requests -> cells, counts, closure), derive (+ feasibility),
                         project (-> twin), steps (points -> step batches), report, cli,
                         forward_run (on the TPU host), cachelocal (~/.xla_cache), xprof_steps
                         (event export -> steps/layers), evidence (facts for REPORT.md; the agent
                         writes the report)
<MODEL_DIR>/             per model, reviewed at onboarding (GLM-5.2-FP8, Qwen3.5-397B-A17B-FP8):
  model.py               every model-specific fact and rule (see "Onboarding a new model")
  config/<name>.json     reference config        <name>.source.json  its snapshot (offline inputs)
  config/<name>.bars.json selftest expectations
tests/test_*.py          CPU tests, one file per area of core/
```

The worked derivations of the shipped models are
{{reference:workload-microbenchmark/GLM-5.2-FP8/examples/agentx-2x2x2-tp16-dcp16-ep.md}} and
{{reference:workload-microbenchmark/Qwen3.5-397B-A17B-FP8/examples/qwen3.5-397b-agentx-singlehost-2x2x1-tp8-ep.md}}.

In the workspace:

```
<work>/models/<MODEL_DIR>/                           a model being onboarded (same layout, plus
                                                     examples/<name>.md)
<work>/jobs/<job-id>/                                fetched job files (`cli.py --jobs`)
<work>/<MODEL_DIR>/workload-microbenchmark/configs/   working configs from `derive` (not merged)
<work>/<MODEL_DIR>/workload-microbenchmark/steps/     plans from `points`, and their recipes
{{runs}}/<run_id>/artifacts/workload-microbenchmark/  REPORT.md per set of jobs
```

## Onboarding a new model

1. Write `<work>/models/<MODEL_DIR>/model.py`, using
   {{script:workload-microbenchmark/GLM-5.2-FP8/model.py}} as the template. **Every
   model-specific fact lives here; `core/` only counts, divides and formats.** Cite a source for
   each value.

   | hook | required | what it provides |
   | --- | --- | --- |
   | `facts(cfg, kv_dtype)` | yes | canonical `Facts`, including `kinds` (the layer kind of every layer, in the model's own names), `kv_bytes_token` (KV bytes per token, all layers), `state_bytes_seq` (recurrent state per sequence, for linear-attention / mamba layers), `mtp_kv_bytes_token` |
   | `twin_overrides(f, real, twin_world, pool, ckpt)` | yes | the twin's `hf_overrides`, engine args, context scale, deviations |
   | `twin_facts(f, hf)` | yes | the twin as `Facts` (same fields, the twin's layers) |
   | `kernels(env, d)` | yes | which implementation each op runs, from the recipe env |
   | `param_bytes_per_rank(f, d)` | yes | resident weight bytes per rank |
   | `per_rank_rows(f, d, pool)` | no | per-rank rows beyond core's generic ones (attention variant, indexer, recurrent state) |
   | `PROJECTION_ROWS` | no | which of those rows the twin check compares, real vs twin |
   | `TRACE` | yes, to read profiles | sync all-reduces before layer 0 and per layer (the layer-boundary rule), the op families per layer kind for the census, and optionally a `drafter_topk_pair` |
   | `ROUTER` | MoE with dummy weights | router parameter names to re-draw, so experts are used as with real weights |
   | `DRAFTER` | for `--mtp` | the drafter's replay entry point in vllm-torchtpu, its gate, and its source functions to record |
   | `cache_hit(r, d)` | no | the prefix-cache hit rule, if it isn't vLLM's full-attention rule (e.g. hybrid models that resume only from saved state) |
   | `feasibility_checks(f, d)` | no | model-specific problems (head / KV-head divisibility, unsupported DCP, ...) |
   | `state_bytes_per_seq_per_rank(f, d)` | models with recurrent state | recurrent state per running sequence per rank, counted against the KV budget and in twin memory |
   | `COMPONENTS` | yes | the exact-shape component benchmarks for the twin's deviations |
   | `TWIN_ENV` | no | env the model always runs with (e.g. a kernel switch it needs on TPU), applied under the job's own env and overrides; the only env source for a server read from its vLLM log |

2. `cli.py onboard <MODEL_DIR> --job <id>`, then `cli.py selftest` and the unit tests.
3. Read the generated `<work>/models/<MODEL_DIR>/examples/<name>.md`. Every number must be
   traceable. **First-time onboarding is reviewed before anyone relies on it.** To ship the model
   with this package, add its directory beside the shipped models in a reviewed change.
4. On the first TPU run, check that the layer census passes (`evidence.json`
   `layer_census_check`). A failure means `TRACE` doesn't describe this model's profile yet.

## Rules

- **Depend on vllm-torchtpu, never copy it.** The code under test comes from the image at the
  arm's commit; the harness only builds inputs, times and checks. Read kernel paths from the recipe
  env, never assume them. One exception: `_run_batch` builds a serving step's inputs for any batch
  (seeded token ids, true positions, distinct non-null KV pages written to the runner's block
  table, distinct state slots); it calls the runner's own `stage_block_table_uploads`, metadata
  builder and `forward_model`. `result.json` records each batch's inputs (`inputs`).
- **Know which source the image ran.** An image tag records no commit, and the package version's
  `+g<sha>` may not match the code actually installed. Every run saves the image's vllm-torchtpu
  version, its git commit if present, and the exact source and line range of each function
  `_run_batch` and `_run_draft` rely on (`result.json` `image_source`,
  `outputs/.../image_source/*.py`). Check parity and cite line numbers from there, not from a local
  checkout.
- **Per-rank, not a smaller world.** Twin rule: `global' / degree' == global / degree` for every
  scaled tensor. The deviations table lists what cannot match and the component that covers it.
- **Cells come from the job, with a server cross-check** (computed tokens within 3% of the server's
  `local_compute` counter). A weight with no data behind it is a bug.
- **A roofline is not needed and not run.** It is an optional side-by-side, never an input.
- **`selftest` must pass** before a derived config is used. Its bars catch drift when recipes,
  job formats or vllm-torchtpu defaults change.

## Status

| milestone | state |
| --- | --- |
| M0: core, onboard, derive, selftest, GLM-5.2-FP8 reference | **done** |
| M1: twin on 2x2x1; points plan (mixed + decode per point, any sharding, optional MTP drafter); evidence; one job per point | **done** |
| M1.6: mirror mode (the served sharding on the served slice, 2 hosts over Ray) beside scaled mode | **done**; smoke test `j-f96adeda` |
| M1.5: the step mix replaces P50/P90 points (classes × context bins × batch, weighted by measured busy share; one job per cell in parallel, `mix-rollup`) | **done** |
| M1.7: `points --top K`: the heaviest K cells in one plan, one job, every one profiled | **done**; first run on GLM-5.2 AgentX c4 mirror |
| M2: twin on the incumbent | — |
| M3: first A/B | — |
| M4: exact-shape components | — |
| M5: the check against L2 | — |

`run` (the A/B driver) is not built. A plan is not a measurement; only `forward_run` results are.
Not covered yet:
- host/scheduler overhead per step (L2);
- the MTP drafter's step-0 top-k, which is modelled, not measured (the replay has no context);
- collectives at the served group size: covered by mirror mode, once its smoke test passes;
- EP-DP lockstep decode padding for DP > 1 deployments;
- a re-sharded plan keeps the job's step counts: a faster sharding that would change the schedule
  itself (fewer overlapping steps, a different running set) is not modelled;
- the GLM-5.2 reference config is still `j-c964a8b6`, whose snapshot has no `step_slices`;
- dummy fp8 weights dequantize to ~1e-6 per element (requantized per-channel scales ~4e-9), so
  every projection is tiny and the residual stream is mostly the embedding; outputs are finite and
  routing is per token, but activation magnitudes are not the served model's (timing is not
  value-dependent, except through routing).
