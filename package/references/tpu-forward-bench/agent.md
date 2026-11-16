---
scripts:
- scripts/common/tpu_slots.py
- scripts/common/vllm_patches/apply_vllm_patch.sh
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
- scripts/workload-microbenchmark/core/mix.py
- scripts/workload-microbenchmark/core/project.py
- scripts/workload-microbenchmark/core/report.py
- scripts/workload-microbenchmark/core/steps.py
- scripts/workload-microbenchmark/core/xprof_steps.py
- scripts/workload-microbenchmark/generic/model.py
- scripts/workload-microbenchmark/generic/tests/fixtures/qwen3-30b-a3b.config.json
- scripts/workload-microbenchmark/generic/tests/fixtures/qwen3-32b.config.json
- scripts/workload-microbenchmark/generic/tests/test_generic.py
- scripts/workload-microbenchmark/tests/test_costmodel.py
- scripts/workload-microbenchmark/tests/test_derive.py
- scripts/workload-microbenchmark/tests/test_forward_run.py
- scripts/workload-microbenchmark/tests/test_mix.py
- scripts/workload-microbenchmark/tests/test_sampling.py
- scripts/workload-microbenchmark/tests/test_xprof.py
references:
- references/workload-microbenchmark
---

# Forward-pass microbenchmark

Engines: vllm.

Derive benchmarks that stand in for a dynamic serving workload, from a job that already ran, and
measure arms on them. How the code fits together is in
{{reference:workload-microbenchmark/core/README.md}}.

Here `cli.py` stands for `python3 {{script:workload-microbenchmark/core/cli.py}}`, run from the
workspace. `<work>` is its work directory (`cli.py --work`, default `./.work`), `<run_id>` the
request's run, and `<out>` is `{{runs}}/<run_id>/artifacts/tpu-forward-bench/<source_job>/<label>/`.

## Scope: check these before anything else

Each failure returns before any job.

- **The engine has a section below.** A request whose `engine` has no `## <engine>` section
  returns `invalid_input`: `engine <x> is not supported here`.
- **Adapter: model-specific, else generic, else written per "When no adapter fits".** `derive`
  loads `<MODEL_DIR>/model.py` from `<work>/models/`, then from the package (`GLM-5.2-FP8`,
  `Qwen3.5-397B-A17B-FP8`). Without one it falls back to
  {{script:workload-microbenchmark/generic/model.py}} when its `classify` finds the checkpoint
  a standard decoder-only model (dense or MoE, full GQA/MHA attention on every layer: Llama,
  Mistral, Mixtral, Qwen2, Qwen3 and their MoE variants). Otherwise `derive` fails naming the
  `classify` reasons (hybrid or linear attention, MLA, a sparse-attention indexer, MTP layers,
  sliding windows, cross-attention, multimodal), and you write the adapter yourself.
- **`gke_job` profiles only.** Each job is a fresh container started from the pinned `--image`, so
  a patch applied in one arm never reaches the next. Read the backend, and return `blocked` with
  the reason for any other:

  ```bash
  {{runtime}} compute --package {{package_tag}} preflight --json \
    | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["backend"], r["status"])'
  ```

- **Every job goes through `{{runtime}} compute`** (`submit`, `job wait`, `job logs`,
  `job artifacts`, `job cancel`), under a slot. Never launch a container or a TPU process any
  other way.
- **`selftest` must pass** (`cli.py selftest`) before a derived config is used; a failure is
  `failed` with its output.

## vllm

The source job served with vllm, and the twin runs vllm-torchtpu's engine. Everything below is
this engine's procedure.

### The run

1. **Plan.** With `plan` a path, use it as is, and an arm's own `plan` when given. Otherwise fetch
   the source job and derive:

   ```bash
   {{runtime}} compute job artifacts --job-id <source_job> --out-dir <work>/jobs/<source_job>
   cli.py derive <model_dir> --job <source_job> [--mode mirror]
   cli.py points <twin.json> --top <top_k, default 5> --out <out>/plan.steps.json
   ```

   `plan.full` drops `--top` and writes one plan per cell under `--out <out>/plan/`. An arm with
   `overrides` gets its own plan, the same `points` with `--override <o>` per entry, at
   `<out>/<arm>.steps.json` (`<out>/<arm>/` when full). A two-host plan (`forward-cmd` exit 2) is
   `blocked`.
2. **Slots.** One per job, holder `forward-bench:<source_job>:<label>:<arm>:r<n>`
   (`r1`…`r<repeats>`; with a full plan, `<arm>.<cell>`): `claim` hands a holder back the slot it
   already holds, so a shared holder would put every job on one slot.

   ```bash
   python3 {{script:common/tpu_slots.py}} claim --run-id <run_id> \
     --holder forward-bench:<source_job>:<label>:<arm>:r<n> --hardware <hardware>
   python3 {{script:common/tpu_slots.py}} set-job --run-id <run_id> --resource <resource> \
     --holder forward-bench:<source_job>:<label>:<arm>:r<n> --job-id <job_id>
   python3 {{script:common/tpu_slots.py}} release --run-id <run_id> --resource <resource> \
     --holder forward-bench:<source_job>:<label>:<arm>:r<n>
   ```

   - Without `hardware` in the request, the first claim takes `--most-available` instead; every
     later claim uses the key it prints, which the result returns.
   - Exit 2 (no `parallel.json`): run `discover --runtime "{{runtime}}" --package
     {{package_tag}}` once and claim again; a second exit 2 is `blocked`. Exit 3 (refused): wait
     for one of your jobs to end; holding none, it is `blocked`.
   - **Release on every exit:** the job ended, the submit failed, the patch did not apply, the job
     failed or was cancelled, before a relaunch, and before you return.
3. **Run.** Per arm, one recipe, submitted `repeats` times, every arm launched together:

   ```bash
   cli.py forward-cmd --steps <plan>.steps.json --image <image> [--patch <p> ...] \
     [--no-profile | --kernel-trace [--profile-options <json>]] \
     --compute "{{runtime}} compute" --recipe <out>/<arm>.recipe.py
   ```

   It prints the `{{runtime}} compute submit` command; run it with the `profile` the claim
   printed, then `set-job`. `--no-profile` only when `profile` is false.
   With `kernel_trace`, the request is an unscored capture of the inside of Pallas/Mosaic kernels:
   one job per arm, `--kernel-trace` (LLO debug info and runtime custom-call tracing in
   `LIBTPU_INIT_ARGS`), `--profile-iters 1` (custom-call tracing grows traces 6–12× and can drop
   HLO events), and on tpu7x `--profile-options '{"tpu_enable_periodic_counter_sampling": true}'`
   for hardware counters. Check `xprof get_llo_analysis` reports `"success": true` with LLO data
   for the kernels in question before tpu-diagnostics reads the trace, and return `unscored`:
   no verdict or `T_mix` from it is ever an A/B, since the flags change the compile key and
   instrument every kernel. A plan with no custom calls has nothing for them to trace. A full plan runs one
   `forward-cmd --repeats 3` per cell plan and arm, every cell job at once.
4. **Wait** on each job; exit 3 means run the wait again.

   ```bash
   {{runtime}} compute job wait --job-id <job_id> --until done
   ```

5. **Failed jobs.** Read `{{runtime}} compute job logs --job-id <job_id>`. Applier exit 4 is the
   arm's `base_mismatch`, never a `loss`: the patch does not fit this image (`plugin.patch` is cut
   from `plugin.base_sha`, so it fails only when that is not this image's plugin commit). No
   relaunch, no metric; on the baseline arm the request is `failed`. Exits 2 and 3, and other
   infrastructure failures, relaunch, at most twice per job; past that the arm is a `loss` with the
   reason, and for the baseline arm the request is `failed`.
6. **Collect** each job, export its trace through tpu-diagnostics ("Reading the profile" below),
   then `collect` again:

   ```bash
   {{runtime}} compute job artifacts --job-id <job_id> --out-dir <work>/jobs/<job_id>
   cli.py collect <job_id> --steps <arm plan>.steps.json
   ```
7. **Score.** The baseline alone, then each candidate against it, every job of an arm listed:

   ```bash
   cli.py mix-rollup --plans <plan> <baseline jobs…> --json <out>/<baseline>.cost_model.json
   cli.py mix-rollup --plans <arm plan> <arm jobs…> --baseline <baseline jobs,…> \
     [--noise-floor-pct <noise_floor_pct>] --json <out>/<arm>.cost_model.json
   ```

   With a full plan, `--plans` is the plan directory, and `mix-rollup` exits 1 if a cell has no
   job. An arm's verdict is `win` when `significant` on the primary basis and no guardrail
   regressed,
   `loss` when significantly slower or a guardrail regressed, otherwise `tie`.
8. **Report and replay.** Write `<out>/REPORT.md` (below) and `<out>/forward_bench_request.json`:
   the request with `plan` set to the baseline plan's path, each override arm's `plan`, and
   `hardware`, each arm's `plugin` or `patches` as sent. Sent back as a request, it reruns the
   identical test.

### When no adapter fits

Use this when `derive` fails because neither `<MODEL_DIR>/model.py` nor
{{script:workload-microbenchmark/generic/model.py}} applies. The error names what the generic
adapter does not model; the new adapter has to model exactly that.

**Find the vLLM source.** The paths below are relative to the vLLM repo root. First look for a
local copy: a vLLM checkout in the workspace, or the installed package
(`python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))"`, whose parent is the repo
root for `vllm/...` paths; `tests/...` paths are not in an installed package). Otherwise read them
on GitHub at https://github.com/vllm-project/vllm (default branch; raw files at
`https://raw.githubusercontent.com/vllm-project/vllm/<branch>/<path>`), or shallow-clone it into
the run's work dir, never the package tree. Cite every fact with where you read it: the local path,
or `vllm@<commit sha> <path>:<lines>` for GitHub. Read the vllm_torchtpu plugin's kernel and env
selection read only: in the campaign repo's work dir when an arm's `plugin` names one
(`plugin.repo`, on `stack`), else from the plugin the image installed, found as for vLLM above.
Never run git or plugin_source.py on it; tpu-code-stack owns the repo.

| adapter piece | read |
| --- | --- |
| architecture -> model module and class | `vllm/model_executor/models/registry.py` |
| layer structure and kinds, `TRACE` op families | `vllm/model_executor/models/<module>.py` |
| text config of a nested (multimodal) checkpoint | `vllm/config/model.py` (`get_model_arch_config`), `vllm/transformers_utils/config.py`, `vllm/transformers_utils/configs/<model>.py` |
| field-name quirks, twin overrides | `tests/models/utils.py` (`dummy_hf_overrides`), `tests/models/registry.py`; keep every size, cut layers to one per kind |
| per-rank splits, KV-head replication | `vllm/model_executor/layers/linear.py` (`QKVParallelLinear`, column / row parallel) |
| KV and recurrent-state bytes | `vllm/v1/kv_cache_interface.py`, `vllm/model_executor/layers/mamba/mamba_utils.py` |

**Write it** to `<work>/models/<MODEL_DIR>/model.py`; `cli.py` searches `<work>/models/` first.
Start from {{script:workload-microbenchmark/generic/model.py}} as the worked example, and borrow
the exotic pieces from the reviewed adapters:
{{script:workload-microbenchmark/GLM-5.2-FP8/model.py}} (MLA, indexer, MTP drafter) and
{{script:workload-microbenchmark/Qwen3.5-397B-A17B-FP8/model.py}} (hybrid linear attention,
recurrent state, multimodal text config). Mark it unreviewed in its docstring.

**Validate it before any number counts:**
- `cli.py onboard <MODEL_DIR> --job <source_job>` writes its reference config, then `cli.py
  selftest` passes.
- `derive`'s projection table: every per-rank row is `match`, or a deviation the adapter names
  in `deviations`.
- KV and state bytes against the source job's server log: the logged `GPU KV cache size` and any
  logged page or state size (e.g. `mamba_raw_state_bytes`) agree with the adapter's
  `kv_bytes_per_token_per_rank` and `state_bytes_per_seq_per_rank`, or the gap is explained.

A failed check stops the request: `blocked`, naming the failing rows. Every result produced with
an unreviewed adapter carries `adapter: {path, reviewed: false}`.

### The idea in four lines

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

### What it cannot answer

It times the model's forward steps on dummy weights, with no server around them.

- **vLLM orchestration.** No scheduler, API server, async-scheduling overlap, detokenization or
  request admission. (The sample stage does run when the plan asks for it: see "Sampling".)
  `forward_run` drives the engine's model runner directly with pre-built batches, so scheduling,
  batching decisions, host dispatch, Ray / DP coordination and the request path are not
  reproduced. Use a server-level run (a `{{runtime}} compute submit --serve` job and its client).
- **Imbalance that depends on real weights.** Weights are random (`--load-format dummy`); routers
  are re-drawn (`ROUTER`, target and drafter) so experts are used, and every request gets its own
  token ids, positions, KV pages and state slots, but the token-to-expert distribution is
  synthetic. KV pages a batch reads are filled with finite values first, and the drafter replay's
  zero hidden-state inputs are swapped for seeded random ones (`_NonZeroHidden`). Expert-load
  imbalance across EP ranks, hot experts, routing skew and anything that depends on trained
  weights (MTP acceptance included) do not reflect the real model.

### Sharding and shapes: mirroring the served slice onto the small one

Serving runs its parallelism on the job's slice, e.g. TP16 × DCP16 × EP16 on 16 cores. The twin
runs the **same parallelism on the small slice** (TP8 × DCP8 × EP8 on the 8 cores of one 2x2x1).
A smaller world alone would double every per-rank tensor, so `project` scales every model
dimension split across ranks by the same ratio (here 8/16) in `hf_overrides`, keeping each rank's
share exactly the served one: `global' / degree' = global / degree`.
- **Scaled with the world:** query heads (64 → 32: 4 per rank either way), routed experts and
  top-k together (256 / top-8 → 128 / top-4), dense FFN width (12288 → 6144: 768 per rank).
- **Context:** × 0.5, so a rank holds L / 16 KV tokens for a context of L in both.
- **Tokens per step:** not split across ranks, so they stay as served.
- **KV page size and pool:** the served block size, with enough pages that the per-rank KV pool
  matches serving. `forward_run` has no scheduler or prefix cache; `points` applies the served
  cache rule before the run.
- **Layers:** one per layer kind, in the served order (GLM-5.2: 78 + MTP → 3), from the model's
  layer-kind rule.

What cannot match on fewer cores, listed as a deviation in the twin config and the report:
- collectives span 8 ranks on one host, not 16 across hosts;
- attention and the indexer merge work over the DCP group (8-way, not 16);
- shared-expert width per rank;
- routing statistics (experts and top-k are scaled together);
- with MTP IndexShare, the drafter's first draft step computes its own sparse top-k, which the
  context-free replay misses; the projection adds it as a *modelled* term.

So a projection to the served model is always labelled *modelled*.

**The KV fill decides who owns the top-k.** With a sparse-attention indexer under DCP, tied
indexer scores make one DCP rank own nearly every slot and straggle. The fill therefore writes
every KV tensor, integer-stored fp8 caches included, with distinct values per page and a seed per
rank ({{script:workload-microbenchmark/tests/test_forward_run.py}} `FillPages`). GLM-5.2 runs
before 2026-10-02 show DCP rank 0 owning ~all slots and misjudge any ownership-sensitive change.

### Two modes: scaled (2x2x1) and mirror (the served slice)

| | **scaled** (default, `--mode scaled`) | **mirror** (`--mode mirror`) |
| --- | --- | --- |
| slice | one 2x2x1 host, 8 cores (`--twin-world 8`) | the served slice, e.g. 2x2x2, 16 cores |
| model | split dims scaled by `twin world / served world` | nothing scaled |
| sharding | the served one at the smaller degree | the served one as is |
| deviations | collectives over 8 ranks on one host; DCP merge width; shared-expert width; routing statistics | none from sharding |
| runs on | one host: one `{{runtime}} compute submit` job (`forward-cmd`) | two hosts on one Ray cluster; `forward-cmd` does not run it |
| use it for | kernel work, checking an op got faster, most cost-model A/Bs | sharding choices, collective and DCP-width effects, an accurate served projection |

- **Prefer scaled mode for kernel and flag arms**, and whenever larger pools are busy: per-rank
  shapes match serving and the pool is easy to get. Use mirror mode for accurate served-time
  analysis and sharding questions when the pool can return results in time; otherwise run scaled
  now and calibrate with mirror later.
- **Pick the mode at `derive` or `project`:** `derive … --mode mirror` writes `<name>.twin16.json`
  beside `<name>.twin8.json`; `points` on it writes under `<sharding>+mirror-mix-<job>/`.
- **`forward-cmd` reads the mode from the plan** and refuses a two-host plan (exit 2): this
  package has no two-host launcher. A mirror plan of a one-host served slice (twin world 8 or
  less) runs as usual. Check with `{{runtime}} compute inspect` how many cores one host has.

### CLI

Every mode takes the global options `--work <work>` and `--jobs <dir>` (fetched job directories,
default `<work>/jobs`) before the mode name:

```bash
cli.py derive   <MODEL_DIR> --job <job-id> [--override k=v ...] [--mode scaled|mirror]   # working config -> <work>/
cli.py project  <config.json> [--twin-world 8 | --mode mirror]  # real workload -> twin config (derive does it too)
cli.py show     <config.json>
cli.py points   <twin.json> [--override dcp=8 ...] [--no-mtp] [--min-share 0.002] \
                [--point NAME=cached/input/output@C ... [--with-mix]] [--top K | --combined] \
                [--sampling | --no-sampling]   # step mix -> one plan per cell (--top/--combined: one plan)
cli.py mix-rollup --plans <dir> <job-id> ... [--baseline <ids>]   # the cost model -> cost_model.json
cli.py forward-cmd --steps <f.steps.json> [--image <path|job>] [--patch <file>]... [--repeats N] \
                   [--no-profile | --kernel-trace [--profile-options <json>]] [--recipe <path>] \
                   --compute "{{runtime}} compute"
                   # writes <plan>.recipe.py, prints the compute submit command
cli.py collect <job-id> --steps <f.steps.json>   # after job artifacts: files the cache, prints
                                 # phase/step times, splits the profile, writes evidence.json
cli.py cache-pull <job-id>       # file the job's compile cache under ~/.xla_cache
cli.py selftest                  # re-derives every reference config and checks its bars
python3 -P -m unittest discover -s {{script:workload-microbenchmark/tests}}   # CPU tests
python3 -P -m unittest discover -s {{script:workload-microbenchmark/generic/tests}}   # generic adapter
```

- **Before any mode,** fetch the job's files:
  `{{runtime}} compute job artifacts --job-id <job-id> --out-dir <work>/jobs/<job-id>`, and for a
  job that returns one archive, `tar -xzf <work>/jobs/<job-id>/outputs.tgz -C <work>/jobs/<job-id>`.
  The CLI never calls `{{runtime}}`. A source serving job's directory must hold its JobSet manifest
  as `recipe.yaml`, and under `outputs/` its aiperf records and metrics and its `vllm.log` (or the
  server log under `logs/`).
- **Overrides:** `tp`, `dp`, `dcp`, `pcp`, `ep`, `block_size`, `mnbt`, `mns`, `max_model_len`,
  `prefix_caching`, `spec_tokens`, …, or `env.<NAME>=<value>`. Unknown keys are an error.
- **An infeasible override is refused** (exit 3) with alternatives.

### The steps, and what each writes

1. **Read the job**: a snapshot of recipe, checkpoint config, requests and server metrics.
2. **Derive the real workload**: deployment, per-rank shapes, cells and counts (checked against
   the server), feasibility. Writes `<name>.json`.
3. **Project onto the small slice** for `--twin-world` cores. Writes `<name>.twin8.json`: the
   per-rank table real vs twin (`match` / `deviation` / `by design`), the twin engine args and
   env, memory per rank, MTP coverage.
4. **Plan the step mix** (`points --top K`): the K cells carrying the most busy time, as fixed
   shapes in **one** self-contained plan. Writes
   `<work>/<MODEL_DIR>/workload-microbenchmark/steps/<sharding>[+mirror]-top<K>-<job>.steps.json`.
5. **Run the plan as one job** through the recipe `forward-cmd` writes: boot the twin with dummy
   weights, compile the plan's buckets, time every shape back to back and profile every one.
6. **Cost model** (`mix-rollup`): `T_mix = Σ steps × step time`; with `--baseline`, the speedup,
   guardrails and noise. Writes `cost_model.json`.

### The step mix: the shapes to run, and their weights

**Why not P50 / P90 points.** An agentic trace is not C identical requests per step. GLM-5.2
AgentX at concurrency 4 (`j-58d59d77`): 54% of wall time is decode steps carrying 1–2 requests;
its P50 / P90 points ran 7–8 requests per decode step, 3% of that job.

**What `points` builds** ({{script:workload-microbenchmark/core/mix.py}}):
- **Every second of the measured window is classified** from aiperf's 1-s scrape of
  `vllm:iteration_tokens_total` and `num_requests_running` (`step_slices`). Under MTP k a decoding
  request is 1+k tokens, so steps of ≤8 / ≤16 / ≤32 tokens are the **D1-2 / D3-4 / D5-8** decode
  classes, and a longer step carries a prefill chunk (**mixed**). The rest of a second is mixed
  when a long step is in flight, else idle (client think time).
- **Cells.** Each class splits into three context bins (0–50%, 50–90%, 90–100% of its time) and
  the two integer batches bracketing its mean D. A mixed cell's chunk fills the budget after its
  decode rows (`mnbt − B × (1+k)`); its decode rows are also timed alone, so the chunk's own cost
  is the difference.
- **Weights.** Every cell carries its step count and share of measured busy time. `--min-share`
  (default 0.002) drops small cells; the kept share is printed.
- **Buckets come from the job:** mnbt, `TPU_TOKEN_BUCKET_EXTRA` and any `compile_sizes`
  (`serving_token_paddings`); each step takes the first bucket that fits.
- **One job: `--top K`** (K = 2–5): the heaviest cells in one plan, each profiled. The cost model
  counts only those cells; the report states the coverage.
- **Full coverage:** `points` without `--top`, one plan and one job per cell, all sharing one
  engine config and compile cache; `mix-rollup` sums them. `--combined` puts every cell in one
  plan (the profile holds only the heaviest cell of each class).
- **A snapshot without `step_slices`** is refused; re-snapshot with `derive`, or pass `--point`.
- **`--point NAME=cached/input/output@C`** is an explicit what-if shape, not a model of the job.
  It replaces the mix unless `--with-mix`.
- **A different sharding** is `--override` on `points` (e.g. `dcp=8`, `tp=8 dp=2`, `ep=false`).
  The step counts stay the job's.
- **MTP and sampling follow the source job.** `--no-mtp` / `--no-sampling` leave them out (the
  header records it); `--mtp` on a job without speculative decoding is refused.
- **Results are twin ms** (N layers, per-rank exact), beside the job's numbers, not a substitute.

### Sampling: the step's sample stage

A served step ends with the LM head on the sampled rows and the sampler (under MTP,
`_sample_spec_verify_chunk`: bonus token plus rejection sampler; without, `sample_from_logits`).
At temperature > 0 with top-k / top-p this is not small (Qwen3.5 AgentX: 0.704 ms of a ~9.95 ms
b1 decode step), and a twin that timed only `forward_model` would not see a sampler change.

- **Plan block** `sampling`: `{run, temperature, top_k, top_p, greedy, source, why}`, from the
  source job's server log, else model.py `SAMPLING`. On by default when the job sampled
  non-greedily; unknown params: off, and `--sampling` is refused.
- **On the TPU host** the runner's own functions run once per step after the forward, timed apart
  as `sampling_ms`. Not included: logprobs, structured output, the mamba read-offset update.
- **In the profile** the stage sits between two marker programs (`SAMPLING_MARKS`, read as a
  toggle); `xprof_steps` books it as `sampling_ms` / `sampling_busy_ms`. A sampled step with
  fewer than 2 markers gets `sampling_error`.
- **Warm-up:** every timed shape runs forward + stage 3× untimed before repeat 0.
- **The win is visible on device time only** (host wall ~1.9 ms either way). In the cost model the
  stage's device busy counts once per step on `served` and `served_device`; `compare` flags arms
  that differ in whether it ran (`sampling_mismatch`).

### Cost model: one number to optimize, and when to trust it

**The number.** The cost of a run is the job's engine-busy time at the twin's shapes:

```
T_mix = Σ over cells i of  steps_i × t_i        t_i = the cell's twin step time, median over repeats
```

Under MTP, `t_i` includes the drafter once per step, never multiplied by a layer count: its
replay wall (`draft_ms`) on the twin basis, its **device busy** (`draft_busy_ms`) on the served
basis, since the replay's wall is mostly host gaps a served drafter does not have. `compare` flags
arms that differ in whether the drafter ran (`drafter_mismatch`).

**An A/B is a ratio** of two runs of the same cells: `speedup = T_mix(baseline) / T_mix(candidate)`.

**`served` basis.** The twin has one layer per kind, so its wall step does not carry over by ratio:
a saving inside the layers is `n_layers` times larger in serving (GLM-5.2: 3 / 18 / 57 per kind);
a per-step saving is the same absolute amount; serving weights layer kinds unequally. So
`mix-rollup` rebuilds each cell's step from its profile
({{script:workload-microbenchmark/core/costmodel.py}}, *modelled*):

```
t_i_served = fixed_i + Σ_kind n_kind × layer_kind,i + drafter_busy_i   n_kind from the model
fixed_i    = backbone_wall_i − Σ_kind layer_kind,i       host dispatch + before/after, counted once
```

- **Per-step ops are not layer work.** Ops that run once per step but land inside a twin layer
  (`TRACE["per_step_ops"]`) are subtracted from `layer_kind,i` and count once in `fixed_i`.
- A cell with no profile goes through its class's profiled cell; a class with no profiled cell is
  listed missing. Nothing is borrowed across classes.
- **One profile is one step, and its per-layer noise is multiplied by the layer count** (GLM-5.2
  decode `moe` flipped ±9% of the served step between profiles). Profile each cell several times
  (`--profile-iters 5`) and pool profiles over jobs of the same arm (`mix-rollup --profiles`,
  `--baseline-profiles`). `profile_spread` joins the repeat spread in the significance test.
- **A repeat's deviation from the median counts once, unscaled** (it is host jitter). Profile each
  arm on 2 boots, and judge an A/B against the largest of the repeat spread, `profile_spread`, and
  an A/A of two baseline boots.
- `basis: twin` (the wall step as run) is kept beside it.

**`served_device` basis: primary for kernel changes.** `served` takes `fixed` from the host wall,
which carries ±0.3 ms of per-step jitter and once hid a −1.4% kernel win. `served_device` takes
every term from device time:

```
t_i_device = before busy + after busy + per-step ops (once) + Σ_kind n_kind × layer_kind,i
             + drafter device busy + sampling device busy          per profile, then pooled
```

- `compare.significance`: Welch's t on per-profile T_mix samples. Profiles of one boot share its
  routing mode, so `significant` = p < 0.05 **and** |speedup − 1| beats the boot-noise floor
  (`boot_spread` with ≥ 2 boots, or `mix-rollup --noise-floor-pct` from an A/A). With neither the
  output says `boot_noise: unmeasured`.
- `mix-rollup` prints all three bases and marks `primary_for`: `served_device` for kernel /
  device-time changes, `served` (or `twin`) for host / per-step changes. An `xprof_steps.json`
  with no `before_busy_ms` gives no device sample: re-run `collect`.

**Always reported beside it:**

| | what | why |
| --- | --- | --- |
| **per-class delta** | seconds each class saved or cost, and its share of the net change | where the win is |
| **guardrails** | the heaviest decode cell (TPOT) and every mixed cell (TTFT, TPOT tail), baseline vs candidate | a lower `T_mix` bought by slower prefill worsens latency; a regressed guardrail is named |
| **noise** | `T_mix` per repeat and its spread; `significant` as above | a margin inside the spread is not a win |
| **coverage** | the busy share the measured cells cover, and any cell without a job | a missing cell is listed, never counted as zero |
| **served gain (modelled)** | `(busy + idle) / (busy / speedup + idle)` with the source job's busy and idle | throughput also contains client think time |

**When it is valid.** `T_mix` holds the job's step counts fixed, so it decides changes that make a
step of the same shape cheaper: kernels, collectives, env flags, layouts. It does **not** decide
changes that alter the schedule (chunk size or `mnbt`, speculative tokens or MTP acceptance, a
re-sharding that changes admission, scheduler policy); for those the verdict needs L2 or L3.

**It is modelled, not measured.** The caller's review and end-to-end run are the last word.

### Image, patches and environment: audit before trusting a number

How the sharding communicates is set by env and `LIBTPU_INIT_ARGS`, not only by
`--tensor-parallel-size`. The twin's env has these sources, and every output says which one each
key came from:

| source | what | where you see it |
| --- | --- | --- |
| **container** | the recipe's `env:` list | `points` ENV block, `source` column |
| **script** | what the recipe script sets before `vllm serve`, evaluated in restricted bash (`serve_parse: bash-eval`) | same |
| **override** | `--override env.NAME=value` | same |
| **image** | the Dockerfile's env and vllm-torchtpu's `envs.py` defaults | `result.json` → `effective_env` |

- **The image is always pinned, never a moving tag.** Default `--image job` is the image the source
  job ran (`source_image`). `forward-cmd` resolves a tag to `…@sha256:…`; the job, its recipe and
  the compile-cache key all carry that digest. Never use `inferencex/vllm:latest` as an arm: it is
  a stale shared base image, not the current code.
- **A plugin arm is the pinned image plus one `--patch`, no build.** The arm's `plugin` is a ref
  in the campaign repo, as tpu-code-stack cut it; pass its `plugin.patch`, the
  `git diff base..sha` already cut, and never read the repo. An arm with `patches` (registry files,
  for a caller with no repo) ships them in order the same way, one `--patch` per file. The job
  applies them before the model loads, and each sha256 joins the cache key. `collect` writes them
  to `evidence.json` → `patches`. Applier exit 4 (`patch does not apply`) is `base_mismatch` either
  way: the patch does not fit the image. No metric; exits 2 and 3 are `infrastructure failure`.
- **A flag or env arm is a plan override**: `points --override env.NAME=value` (or an engine arg),
  same image and plugin ref as its baseline.
- **A different image runs the source job's args and env.** Flags its vLLM does not know are
  translated and listed in `result.json` → `compat`; anything else fails at argument parsing.
- **Plumbing is dropped, and listed:** platform output paths and bookkeeping vars, Ray /
  `TPU_PROCESS_*` wiring, `TPU_NODE_ID` / `TPU_WORKER_ID`, HF cache, `MODEL_PATH`.
- **A sharding override does not recompute sharding-dependent env.** `points --override tp=… /
  dcp=… / ep=…` marks each such key `REVIEW` (`derive.SHARDING_ENV`); set it with
  `--override env.NAME=...`. A plan with open REVIEW lines is not a measurement of the new sharding.
- **To audit a run,** compare the source job's env (`points` ENV block), the twin's
  (`effective_env`) and `compat`. Every difference must be plumbing, an override you passed, or
  `image/pod` for an image you picked on purpose.

### Arms and run shapes

The arms of an A/B differ **only by the change** and run the same cell plans. Each arm is a pinned
image, an optional plugin ref and a set of overrides; nothing is built. A flag or env arm is the
same image and plugin ref with its own `points --override` plan. A cold job is a complete
measurement (compile is in engine init, outside the timed steps), so **no warm run**.

A microbenchmark win is modelled: the caller reviews it and confirms it end to end. If the change
alters the schedule, `T_mix` cannot decide it (see "When it is valid").

| run shape | jobs | profiled | use it for |
| --- | --- | --- | --- |
| `--top K` (standard) | one | every one of the K | the heavy cells; an A/B is one job per arm |
| one plan per cell | one per cell | every cell, its own job | the whole mix |
| `--combined` | one | the heaviest cell per class | every cell timed, one profile per class |

- **Each profiled step is labelled** `<cell>/decode` or `<cell>/mixed/chunk0` in `profile_order`.
- **Typical cost of one job** (GLM-5.2 mirror `--top 5`, cold): about 6 min, almost all engine
  init (~250 s) and the profile window (~75 s); the timed steps are seconds. Fewer profiled steps
  (`--profile-iters`) is the lever after compile. Add 1–2 min of job creation and the queue.

### REPORT.md: every measurement ends in one

You write REPORT.md; no script generates it. Put it at `<out>/REPORT.md`, one report for every
job of the request.

The report describes where the time went; it does not rank what to fix. Ranking is the caller's.

**Read, per job** (all written by `collect`, next to the job's `result.json`):

| file | what it holds |
| --- | --- |
| `result.json` | cache state, phase times (`times_s`), every repeat's step times, the roll-up `mix`, drafter times, `effective_env`, compile-key inputs, `image_source` |
| `xprof_steps.json` | the profiled steps split into layers: device ms per segment, top ops with shapes, collectives, the layer census; served projection |
| `evidence.json` | layer kinds, served counts and layer ids; sharding, `hf_overrides`, per-rank table and deviations; layer census check; HLO op categories (from {{agent:tpu-diagnostics}}'s `hlo_categories.json`); SparseCore offload check; `patches` |
| `cost_model.json` | `T_mix` per arm and, with `--baseline`, `compare` (speedup, significance, per-class deltas, guardrails, modelled served gain) |
| `plan.steps.json` | the plan the job ran: the step mix, one entry per cell, buckets, twin env |

For what an env var, op or shape does, read the code the image ran (`image_source/`) and cite
`file:line`.

**Reproduce: always the first section.** The source job id; the `derive` / `points` commands
verbatim (every `--override`, `--no-mtp`, `--mode`); the package version (`{{package_tag}}`); one
row per job (cell, shape in served terms, weight, job id, plan file, image digest, plugin ref and
sha or patches, failed or rerun marked); each `{{runtime}} compute submit` command and recipe as
submitted; the `mix-rollup` command.

**Must cover**, after Reproduce:
- **Layer count first:** "This is an N-layer per-rank twin, not the M-layer (+ MTP) served model",
  from the evidence. Label every measured table "twin (N layers)".
- **Layers and sharding, before any number:** each layer kind, how many served layers it stands
  for, how the served sharding maps onto the small slice, the per-rank table and deviations.
- **Setup:** the step mix and every cell with the share kept; MTP drafter on or off; the compiled
  buckets and what the plan skips; cache warm or cold; repeats × iterations; the image digest and
  plugin ref and sha, and any `compat` translations.
- **Environment:** the vars that pick code paths or collectives, and the offload check.
- **Time breakdown** of the job, and **step times** with every repeat, median and spread, each step
  described in served terms.
- **The cost model:** `T_mix`, spread, coverage, per-class shares; for an A/B the speedup with its
  significance, per-class deltas, every guardrail, the modelled served gain; and whether `T_mix`
  can decide this change.
- **Profile step map** per job (every `XLA Modules` block: start, device ms, `program`,
  `profile_order` label), naming the trace file.
- **Per-layer breakdown:** device time per layer, top ops and collectives with shapes, and the
  layer census as the evidence for the boundaries.
- **Projection to the served model**, *modelled*, beside the source job's *measured* numbers.

**Rules:** every column header names its unit; *measured* (twin) and *modelled* (served) never
share a column; no number without a job id behind it; name steps by cell everywhere.

Keep `<work>/jobs/<job-id>/` until the report is written: `collect` reads the rank-0 trace and
the exports from there and does not fetch a deleted trace back.

### Reading the profile: ask tpu-diagnostics

Never read a trace yourself. When `collect` reports a profiled job without `trace_events.json`,
send {{agent:tpu-diagnostics}} a `profiling-request`, all paths resolved. `<res>` is the directory
of the job's `result.json`, which `collect` prints; the trace is the rank-0 `.xplane.pb` it names:

```json
{"prompt": "Read the rank-0 trace of workload-microbenchmark job <job-id> and write two files. (1) <res>/trace_events.json: every event of the 'XLA Modules' and 'XLA Ops' lines of the /device:TPU:0 plane, with no event limit, as {\"events\": [{\"line\": <line name>, \"name\": <the full event name, HLO text included>, \"start_ns\": <start>, \"duration_ns\": <duration>}]}, from one query_events_db SQL query (category AS line, kernel_name AS name, start_ns, end_ns - start_ns AS duration_ns, WHERE device = 'TPU:0' AND category IN ('XLA Modules', 'XLA Ops') ORDER BY start_ns), not an event-listing tool or a parsing script. (2) <res>/hlo_categories.json: the HLO op profile by category, IDLE included, as [{\"category\": ..., \"total_self_time_ms\": ..., \"op_count\": ..., \"top_op\": ...}], from get_hlo_stats. Then report the fields below and the top ops.", "run_id": "<run_id>", "traces": [{"path": "<rank-0 .xplane.pb>", "label": "<job-id>"}], "report_path": "<out>/<job-id>/tpu_profiling_report.md", "fields": ["step_time_ms", "device_busy_ms"], "top_ops": 20}
```

- On `reported`, check both files exist, then run `collect` again: it splits the profile and
  rebuilds `evidence.json`.
- On `failed` or `needs_input`, the job has no profile split. Say so in the report; the twin basis
  still holds and the served bases list the job's cells as missing.
- Put every job of one arm set in one request, one `traces` entry per job with its job id as
  `label`, when the prompt names each job's `<res>`.
- Copy each returned trace's `ml_diagnostics_url`, when tpu-diagnostics published it, into its
  arm's `ml_diagnostics_urls`, in `traces` order: the caller links each A/B to its capture.

### Compile cache (local, `~/.xla_cache`)

- **Layout:** `~/.xla_cache/<image-tag>/<key>/compile_cache.tgz`, the key a hash of the image
  digest, the engine args (buckets included), the plan's env and the patch hashes.
- `forward-cmd` adds a hit as an `--input` with `--cache-in`, and `--save-cache` only on a miss or
  with `--resave`. After the job, `collect` (or `cache-pull`) files the new cache. A new image or
  plugin patch starts cold.
- **A new config compiles once.** The torch.compile dir key differs between a boot with no cache
  and one that restored one, so run 2 rebuilds its graphs, cheaply, from the Tier-3 executable
  cache; run 3 onward loads them directly. `result.json` → `compile_key_factors` records the
  inputs.
- **Saves stay under 2 GiB** (`_prune_for_save`), so they upload quickly as the next job's input.

### Layout

```
core/                    model-agnostic: jobdata, deploy, cells, derive, project, steps, report, cli,
                         forward_run (on the TPU host), cachelocal, xprof_steps, evidence
<MODEL_DIR>/             a model-specific adapter: GLM-5.2-FP8/,
                         Qwen3.5-397B-A17B-FP8/, or one in <work>/models/:
  model.py               every model-specific fact and rule
  config/<name>.json     reference config; <name>.source.json its snapshot; <name>.bars.json selftest bars
generic/model.py         standard decoder-only models without their own model.py; tests in generic/tests/
tests/test_*.py          CPU tests
```

The worked derivations of the model-specific adapters are
{{reference:workload-microbenchmark/GLM-5.2-FP8/examples/agentx-2x2x2-tp16-dcp16-ep.md}} and
{{reference:workload-microbenchmark/Qwen3.5-397B-A17B-FP8/examples/qwen3.5-397b-agentx-singlehost-2x2x1-tp8-ep.md}}.

In the workspace: `<work>/jobs/<job-id>/` (fetched job files),
`<work>/<MODEL_DIR>/workload-microbenchmark/configs/` (from `derive`),
`<work>/<MODEL_DIR>/workload-microbenchmark/steps/` (plans and recipes),
`<out>` (recipes, cost models, REPORT.md and the replay record per request).

### Rules

- **Depend on vllm-torchtpu, never copy it.** The code under test comes from the arm's image and
  patches; the harness only builds inputs, times and checks. Read kernel paths from the recipe env.
- **Know which source the image ran.** Every run saves the image's vllm-torchtpu version, commit
  and the exact source of the functions it relies on (`result.json` `image_source`). Cite line
  numbers from there, not from a local checkout.
- **Per-rank, not a smaller world:** `global' / degree' == global / degree` for every scaled
  tensor.
- **Cells come from the job, with a server cross-check** (computed tokens within 3% of the
  server's `local_compute` counter).
- **A roofline is not an input here.**
- **A plan is not a measurement**; only `forward_run` results are. There is no A/B driver: arms
  are run as above.
- **Not modelled:** host and scheduler overhead per step (L2); the MTP drafter's step-0 top-k
  (modelled); EP-DP lockstep decode padding for DP > 1; a re-sharding's effect on the schedule.
  Dummy fp8 weights give tiny activations; timing is not value-dependent except through routing.
