---
references:
- references/write-tpu-kernels/
- references/tpu-pallas/code-reference.md
scripts:
- scripts/common/tpu_slots.py
- scripts/common/vllm_patches/apply_vllm_patch.sh
---

# Pallas kernel procedure

Turn a block of compute into a custom TPU kernel that runs near its hardware floor at its original
precision, in every phase it is called in, and hand it back as a `vllm_torchtpu` commit on the
gap's branch, on the caller's stack head — or deliver an evidenced verdict that no kernel is
warranted.

You run to the end without asking anyone. Every input is in the request; there are no human gates
and no subagents beyond your declared delegates. Probes, upstream reading and trace parsing happen
in your own shell.

## Core rules

**Design**

1. **70/30 design before code, per stage.** Steps 0–3 get most of the effort, and the re-freeze
   that opens stage B is design time too. A wrong dataflow costs ten times more to fix in code than
   on paper. Design-time probes that answer a *fact* — does this lower, what does this primitive
   cost, what did the compiler build — are design, and are encouraged. Timing a candidate is not,
   and waits for the freeze.
2. **Deliver in two stages.** Stage A reaches the floor of today's decomposition. Stage B moves
   toward the floor of full fusion. Both are designed up front; A ships first, and A must not rule
   out B.
3. **Ask two questions, not one.** *What is this computation and where must it run* — the op DAG,
   the floors, the cost classes. **And** *what does a competent kernel of this shape contain* — the
   implementation affordances. The first finds the right physics; only the second finds a blocking
   write.
4. **Elimination before tuning.** Before padding or retiling a structure, ask whether a native
   primitive, a fold into an existing matmul, or a different unit deletes it.
5. **Every phase, every time.** The kernel serves prefill, `decode_first` and `decode_last`. Floors,
   gates, predictions and results are per phase, and a phase that regresses is a regression.

**Evidence**

6. **A correctness claim needs a fault the oracle catches.** A green oracle is evidence only after
   an injected fault has turned it red on the cell class production runs.
7. **A performance claim needs a removal that moves the clock.** "X costs N" is a hypothesis until
   an arm without X measured about N less. Instruments that agree are independent only if they
   reach the answer by different routes.
8. **Every number carries its origin.** `[measured: <job_id>]`, `[modelled]`, `[fact: <id>]`,
   `[source: path@sha]`, `[assumed]` — in notes too. A roofline is always `[modelled]`.
9. **A null holds only on the axis and configuration it was measured on.** Record both, or it will
   harden into a false constraint.

**Sources and process**

10. **Learn the API from upstream, never from memory.** Pallas and Mosaic APIs move between
    releases, and a remembered signature is the most common source of wasted compiles. Every API
    the kernel uses is copied in shape from a working, cited example (an *upstream lookup*, below).
11. **Upstream is an example, not an oracle.** `jax` and `tokamax` kernels target other generations,
    precisions and shapes, and are not optimal for yours. Take the mechanism, the API usage and the
    test structure; re-derive the buffer math.
12. **Zero hardcoded constants or dtypes.** Dtypes from the spec; `pltpu.get_tpu_info()` and
    `pltpu.get_sublane_tiling(dtype)` at runtime.
13. **Keep raw logs out of your context.** Read job logs, traces and upstream trees through `grep`,
    scripts and the probe record; carry forward the record, not the log.
14. **Only the plugin.** The commit you return touches `vllm_torchtpu/` and nothing else. A fix that
    needs torch_tpu, JAX, libtpu, vLLM core or the image is not yours to make: stop `blocked` with
    `fix outside plugin: <where and why>`.

**Upstream lookup.** Follow {{reference:tpu-pallas/code-reference.md}}: it clones `jax` and
`tokamax` once per run into `{{runs}}/<run_id>/code-reference/` and routes a question to the
upstream file that answers it, cited as `path:line-line@sha`.

## Conventions

| | |
| --- | --- |
| **Time delta** | `(new − old) / old × 100`. **Negative is faster.** |
| **Speedup** | `old / new`. `> 1.0×` is faster. |
| **Phases** | `prefill`, `decode_first`, `decode_last`, exactly as {{agent:tpu-roofline}} names them. |
| **Precision** | storage and accumulator dtypes stated separately (e.g. `fp8_e4m3fn` in, `fp32` accumulate, `bf16` out). |
| **Carried rules** | a wiki rule labelled for another chip or unit is `[assumed]` until re-measured. |
| **Frozen text** | probe READMEs and whiteboard versions are never edited after their md5 is recorded; corrections go in the next version. |

## Workspace and the wiki

The campaign workspace is `<campaign>` = `{{runs}}/<run_id>/artifacts/tpu-pallas/<slug>/`, with
`<slug>` from the block's function name and the gap.

```
<campaign>/
├── WHITEBOARD.v1.md   design; frozen at the gate. Re-freezes add v2, v3 … beside it.
├── STRATEGY.md        run log: every win, rejection and null, as it happens
├── roofline/          tpu-roofline requests and results for F1 and F2
├── instruments/       accuracy test + microbenchmark, frozen before Job #1
├── incumbent/         compile-only profile of the incumbent
├── probes/<nn>-<slug>/  probe.py, run.sh, frozen README.md, <job_id>.txt, record.yaml
├── variants/          candidates, one file each — created only after the freeze gate
├── final/             kernel_patch.diff (base..kernel commit), apply check, final_report.md
└── wiki/              entries promoted at the end, for merging into the wiki
```

The wiki is {{reference:write-tpu-kernels/}}, read-only. At each step, open
{{reference:write-tpu-kernels/INDEX.md}}, take the rows its by-stage list gives for that step, and
read only those bodies. Check each entry's `hardware:` label against `target.chip` before using a
number; a mismatch means re-measure, not carry. Cite what you use as `[fact: <id>]`.

| step | wiki stage |
| --- | --- |
| 0–1 — context, operating points, instruments | `hardware-facts.md` whole + stage 1 |
| 2 — floors, dataflow, cost classes, affordances | stage 2 |
| 3 — algebra, unit trading, freeze | stage 3 |
| 4 — lowering | stage 4 — the APIs that lie |
| 5 — probing | stage 5 + `hardware-facts.md` "Instruments that lie" — **before** dispatch |

**Constants have one source each.** Throughput numbers — HBM bandwidth, MXU and VPU rates,
SparseCore bandwidth, per-call intercept — come from {{agent:tpu-roofline}}, so two floors can never
disagree. {{reference:write-tpu-kernels/hardware-facts.md}} owns limits and behaviour: VMEM
capacity, tiling legality, relayout costs, what lowers. If the two disagree on a shared number, stop
and report it rather than picking one.

## TPU jobs: stack head, slots, budget

**Every TPU job runs on the stack head.** Plugin changes are git commits, and you never run git:
{{agent:tpu-code-stack}} does. The request's `checkout` is a worktree of the campaign repo on
`checkout.branch`, the gap's branch, starting at `checkout.revision`, the stack head's commit
(`stack_head.plugin.sha`); a `ref` of the branch whose `sha` is not `checkout.revision` at the start
is `invalid_input`. `checkout.revision` is the incumbent. Edit files only in `checkout.path`. Each
variant you wire into production code there is committed by a `commit` delegation (`caller:
writer`, the lever as `message`); a lost one is edited back and committed again, since tags and
resets are your caller's. A job ships the applier and one patch from `base`: the last commit's
`plugin.patch`, or the patch a `ref` delegation cuts. Its recipe applies it before anything
imports the plugin.

```bash
{{runtime}} compute submit --profile <profile> --image <stack_head.image> \
  --recipe <campaign>/probes/<nn>-<slug>/run.sh --interpreter bash \
  --input {{script:common/vllm_patches/apply_vllm_patch.sh}} --input <campaign>/probes/<nn>-<slug>/plugin.patch \
  --input <campaign>/probes/<nn>-<slug>/probe.py --input <campaign>/probes/<nn>-<slug>/README.md \
  --input <incumbent file> --input <campaign>/variants/<name>.py --input <instruments> \
  --output <result file> --run-id <run_id> --name tpu-pallas-<slug>-<nn>
```

Save `{{runtime}} compute job logs --job-id <job_id>` as `probes/<nn>-<slug>/<job_id>.txt`.

**Claim a slot before every submit**, from the request's run, under a holder unique to that job:
`tpu-pallas:<slug>:<nn>`. `claim` hands a holder back the slot it already holds, so a shared
holder would put every job on one slot.

```bash
python3 {{script:common/tpu_slots.py}} claim --run-id <run_id> \
  --holder tpu-pallas:<slug>:<nn> --hardware <target.hardware>
python3 {{script:common/tpu_slots.py}} set-job --run-id <run_id> --resource <resource> \
  --holder tpu-pallas:<slug>:<nn> --job-id $JOB_ID
python3 {{script:common/tpu_slots.py}} release --run-id <run_id> --resource <resource> \
  --holder tpu-pallas:<slug>:<nn>
```

Submit with the `profile` the claim prints. Exit 2 (no `parallel.json`): run `discover` once
(`--runtime "{{runtime}}" --package {{package_tag}}`), then claim again; a second exit 2 returns
`blocked`. Exit 3 (refused): wait for one of your own jobs to end and release it; if you hold none,
return `blocked` with the `show` output — your caller cannot free more while it waits on you.

**Release on every exit**: when a job ends, on a failed submit, on a cancelled or failed job, before
every delegation, and before you return, whatever the status.

**Budget.** `budget.jobs` counts every submit, the apply check included; keep one for it. When one
job is left, stop probing and deliver what is measured. An apply exit 2 or 3, or a job that fails
for infrastructure, relaunches at most twice; past that, return `failed`.

---

# The flow

```
STEP 0  SYSTEM CONTEXT (paper)
  op DAG by class · eliminate by unit capability · 2× bytes per unit boundary
  · denominator T_block/T_step · is the inherited decomposition still right?
STEP 1  OPERATING POINTS, INSTRUMENTS, BASELINE
  cell → roofline phases → accuracy test + microbenchmark (paged KV, ragged)
  → freeze → incumbent profile from artifacts → Job #1 per phase
STEP 2  TWO FLOORS, AND THE DESIGN OF BOTH STAGES
  F1 · F2 (legality-marked) · per-phase gate · examples before API
  · dataflow + working set · primitives · cost classes · affordances · stage-B ladder
STEP 3  ALGEBRA, UNIT TRADING, FREEZE GATE
STAGE A  Step 4 lower + affordance audit → Step 5 probe → re-freeze on refutation → deliver
STAGE B  (A delivered, fusion prize F1 → F2 above noise) re-freeze on A's numbers;
         fusion ladder, one boundary per rung: Step 4 → Step 5 → deliver, per rung
FINISH   patch → apply check → reviews → result
```

Start from the request's `floor` and `diagnosis`: they say which gap brought you here and what the
caller's profile already showed. They are inputs to Step 0, not a substitute for Steps 1–2.

---

## Step 0 — System context

**On paper, before Step 1.** Steps 1–5 make a decomposition fast; Step 0 asks whether it is the
right one.

1. **Write the op DAG.** Every op with shapes and dtypes, classified as exactly one of: dense
   contraction · elementwise · reduction · transcendental · data-dependent addressing · layout
   change. The class, not the op's name, is what maps to silicon.
2. **Eliminate by capability.** Most ops have one legal unit. Check each unit's hard limits
   (`hardware-facts.md`) before costing anything — a unit without `dot_general` cannot hold a
   matmul, whatever its bandwidth.
3. **Price every unit boundary as a full HBM round trip**: `2 × intermediate_bytes`. A placement
   locally worse on one op that deletes a boundary usually wins. Note which boundaries *could* be
   removed and what would have to lower for that; this is the raw material for stage B.
4. **Record the denominator** — `T_block / T_step` per phase — as `measured` (cite the diagnosis
   report), `modelled` (show the arithmetic), or `UNKNOWN`. Concurrent units cost `max()`, not
   `sum()`: an op already hidden under another is worth at most `max(0, T_hidden − T_cover)`. Under
   `UNKNOWN` the deliverable is device time only, and establishing the denominator is the top
   carried item.
5. **Name the assumption behind the inherited decomposition** and check whether anything since —
   the wiki, a new primitive, a toolchain bump — has falsified it.

---

## Step 1 — Operating points, instruments, baseline

### 1.1 Operating points — the roofline's phases, not round numbers

Derive the served config from `model`, `cell` and `mesh` (and `cell.serving_flags` when given),
then send {{agent:tpu-roofline}} a `roofline-request` with that config, the three phases as
`cells`, the block, and `tpu` from `target`: `topology` from the `hardware` key, and
`accelerator_type` the `chip` with that topology's chip count (`tpu7x` on `2x2x2` → `tpu7x-8`). Its phase list is the kernel's operating points:

| phase | what the kernel sees | `kv_len` |
| --- | --- | --- |
| `prefill` | a chunk of `mnbt` tokens | the context before the chunk |
| `decode_first` | `batch` requests, one token each | ISL |
| `decode_last` | the same | ISL + OSL |

Take the **per-device** tokens, batch and `kv_len` the roofline prints after sharding. Record them
in `WHITEBOARD.v1.md` with the dtypes from `precisions` and the serving flags that shape them.

### 1.2 Start from an upstream test, not a blank file

Before writing either instrument, find upstream the kernel nearest this block and **the unit test
and benchmark that prove it** — the paged or ragged paged attention tests for a decode attention
kernel, the megablox / ragged-dot tests for a grouped GEMM. Borrow their structure, not their
shapes: one correctness suite the incumbent and every candidate both run; one shape/dtype spec
(here, the three phases) that the test and the benchmark both consume; upstream's realistic inputs
(page tables, ragged lengths, masks) and its comparison helpers.

### 1.3 The accuracy test — `<campaign>/instruments/test_accuracy.py`

Three parts, run on every phase cell plus edge cells:

| part | role |
| --- | --- |
| **math oracle** | `float64` (NumPy, or JAX `Precision.HIGHEST`). Separates algorithmic drift from rounding when the math is reordered. |
| **native reference** | the op in its original dtypes with its real accumulator — calibrates `atol` / `rtol`. |
| **fault control** | deliberately broken variants the test must reject — one per fault class (wrong mask, off-by-one page, stale accumulator, dropped tail) — with the catch rate reported **per cell class**. |

Edge cells: extents not a multiple of the block size; `kv_len` of 0, 1, and one either side of a
page boundary; a one-token prefill chunk; the maximum batch. A fault class that is blind on the
cell class production runs means the gate has not been built, however many cells pass.

### 1.4 The microbenchmark — `<campaign>/instruments/bench.py`

Its cells are the three phases, exactly, so every measured number joins a modelled one:

- **KV cache, when the kernel reads one** — paged, at production page size and KV dtype, filled to
  the phase's `kv_len`, with block tables drawn from a **shuffled** page pool.
- **Prefill** runs as a chunk with its prior context already in the cache.
- **Ragged decode** is one extra cell, request lengths spread across `[decode_first, decode_last]`.
- **Kernels without KV** (GEMM, MoE, norms): cells reduce to token counts.
- **Value-dependent cost** — routing skew, top-k, masks — uses production's distribution.
- **Reporting:** per rep, per cell: time, compile seconds, and the median with rep 0 held out.
  Obey M1 and M2 (Step 5) from the first version.

**Freeze `instruments/` before Job #1** and record each file's md5 in the whiteboard.

### 1.5 The incumbent's profile — from artifacts, not source

One compile-only job records what the compiler actually built:

1. Write `<campaign>/incumbent/capture.py`. It sets
   `LIBTPU_INIT_ARGS="--xla_xprof_register_llo_debug_info=true"` before it imports jax, builds the
   incumbent exactly as `bench.py` does, warms it up outside the trace, runs one input size under
   `jax.profiler.trace`, and packs the trace directory into `incumbent_trace.tar.gz`. Submit it on
   a slot as above; unpack the archive under `<campaign>/incumbent/trace/` and delete it.
2. Send {{agent:tpu-diagnostics}} a `profiling-request` ({{schema:profiling-request.json}}):

   ```json
   {"prompt": "Read the compile and trace data of the incumbent kernel <name>. Report what the compiler built.", "run_id": "<run_id>", "traces": [{"path": "<campaign>/incumbent/trace", "label": "incumbent"}], "report_path": "<campaign>/incumbent/tpu_profiling_report.md", "fields": ["program_or_scope_name", "tile_parameters", "grid", "vmem_allocations", "buffer_counts", "tpu_custom_call_count_per_call", "xla_ops_in_jitted_region"]}
   ```

3. Proceed only on `status: reported`.

Identify the incumbent by program id or by the shape string in its own name, never by a phase
label or a directory name. Default to the trace-side capture. Use the dump-side capture
(`--xla_jf_dump_to`, `--xla_mosaic_dump_to`) only for a question the trace cannot answer: write to
the target's `/tmp`, copy back only the kernel's own final-pass files in one archive, and stop the
job past 1 GB.

### 1.6 Job #1 — the baseline, per phase

Run the incumbent — the file the stack head serves, verbatim, md5-recorded — at every cell. Print
the **XLA op breakdown of the measured region**: the template every candidate's region is compared
against (M4). Record per-phase times as `[measured: <job_id>]`.

---

## Step 2 — Two floors, and the design of both stages

### 2.1 Two floors via {{agent:tpu-roofline}}

Send it the model, precisions, mesh, `tpu` as in 1.1, the three phases, and the block as
`file:function` with its op DAG from Step 0. Ask for two numbers per phase, and keep its per-op binding resource:

| | **F1 — today's decomposition** | **F2 — full fusion** |
| --- | --- | --- |
| boundaries | today's op and kernel boundaries kept | none inside the block |
| HBM | every intermediate written and read back | the block's external inputs, weights, KV and outputs only |
| time | Σ over ops of `max(HBM, compute)` | `max(total HBM / BW, total compute)` |
| its gap | **baseline → F1 is tuning headroom** | **F1 → F2 is the fusion prize** |

Check F1 against the request's `floor`; a disagreement is a finding to explain, not to average.
**F2 is a target only if fusion is legal.** If Step 0 put two ops on different units with no
handoff that lowers without HBM, F2 is a `projection`: carry it and plan no stage B toward it.

### 2.2 The gate, per phase

| per-phase condition | action |
| --- | --- |
| baseline ≤ 1.25 × F1, and the fusion prize is under noise or F2 is a projection | **stop at floor** — through a `kind: stop` {{agent:tpu-critical-review}} |
| baseline ≤ 1.25 × F1, fusion prize above noise | skip stage A for this phase; stage B |
| baseline > 1.25 × F1 | stage A; plan stage B if the prize is above noise |
| bytes already at the theoretical minimum | **gate inapplicable** — say so; rank rows, descriptors and events with the fixed-cost fit (cost class 6) |

"Above noise" is absolute: `(F1 − F2) × calls per step` against the benchmark's own spread, and
against the denominator.

### 2.3 Examples before API

Design and write from working examples. Through upstream lookups, collect:

1. **Design exemplars** — the shipped kernels nearest this block in shape class. Read them for
   *mechanism*: pipelining, tiling, ragged tails, scratch layout, grid split. Ask which of their
   choices were forced by a limitation your target does not share.
2. **One working example per API** the design uses — `pallas_call` and its grid spec, `BlockSpec`
   and index maps, `PrefetchScalarGridSpec`, `emit_pipeline`, `make_async_copy` and semaphores,
   memory spaces, `pl.when`, scratch shapes, SparseCore primitives.
3. **Lowering questions** — answered by the test that covers them, before any job is spent.

Record an **API ledger** in the whiteboard: one row per API, with its citation and the exact form
the kernel will copy. An API with no row is written from memory and goes back to lookup before
Step 4.

### 2.4 Dataflow, working set, primitives

- **Pipeline diagram** — DMA in, compute and DMA out across steps T−1, T, T+1, with each stage's
  buffer. **Name the pipelining axis.**
- **Working set** — `Buffer_Count × Σ(tile elements × sizeof(dtype)) + scratch ≤ VMEM limit`, per
  phase, arithmetic shown. A hard gate: declared scratch that does not fit may fail to compile.
- **Accumulators** — dtype and residency stated.
- **Primitive audit** — the target unit's native primitives (`hardware-facts.md`, the exemplars):
  scans, cross-lane reductions, masked loads, indirect DMA, packing intrinsics. Before copying an
  idiom, ask which primitive the original was avoiding and whether your unit has the same hole.
- **Structural deletion test** — which buffers, tables or scratch structures can be removed
  outright by a native primitive?

### 2.5 Cost classes — each with a critical-path verdict

Fill every row as `examined: <work> → <critical-path contribution> [by: ablation | overlap
measurement | argued]` or `UNEXAMINED: <why>`. Inside a pipelined kernel, work concurrent with a
longer transfer contributes `max(0, T_work − T_cover)`. An `argued` verdict is `[modelled]` and must
be backed (M8) before it ranks a lever or appears as an attribution.

| # | cost class | instrument that reveals the work |
| --- | --- | --- |
| 1 | structural elimination | the primitive audit and deletion test |
| 2 | bulk memory traffic | F1/F2 bytes against measured time |
| 3 | per-element ALU in the innermost body | the body's op list × trip count |
| 4 | cross-lane serializing ops | a per-chunk cost model |
| 5 | bank / port conflicts | latency that moves with input *distribution* at constant op count |
| 6 | launch / fixed overhead | the intercept of an M-sweep (M2) |
| 7 | boundary and layout handoff | shapes and pads at the wrapper; `2 ×` bytes per boundary |
| 8 | inter-unit concurrency | `max()`, not `sum()` |

Rows 1, 3, 4 and 5 have no profiler view, which is why **the unexamined rows are a deliverable.**

### 2.6 Implementation affordances — declared now, audited in Step 4

Score every row `planned`, `NA: <reason>` or `absent: <reason>`. Add rows; never remove them.

| affordance | how Step 4 checks it |
| --- | --- |
| every critical-path HBM→VMEM transfer is double- or multi-buffered | buffer count ≥ 2, or the ring depth of a manual loop |
| every VMEM→HBM write is async, waited on as late as possible | no `.wait()` directly after its own `.start()` |
| tail and ragged tiles run a narrower body | a distinct path or masked shape for the last grid step |
| zero-trip and common-case fast paths | a branch ahead of loop setup |
| no host-computable work inside the jit | the region's XLA breakdown has only the custom call(s) plus declared ops |
| per-call setup hoisted | built outside the jitted callable, or cached |
| the epilogue is chunked or overlapped with the next load | it does not serialize after the last DMA |
| the accumulator stays resident across the contraction loop | no HBM round trip of partial sums |
| grid dimension semantics permit the core split | `parallel` on independent axes |
| data-dependent indices arrive by scalar prefetch | not loaded inside the kernel body |

### 2.7 Design both stages

- **Stage A** — designed in full: the changes that reach F1, boundaries kept.
- **Stage B** — a **ladder of boundary removals**, ranked by bytes saved, each rung with its own
  modelled floor. Look up how upstream fuses across the same boundary before sketching a rung.
- **Does A rule out B?** Check every stage-A interface choice — output layout and dtype, tiling
  axes, grid order. If the fused kernel could not consume it, change A now.

---

## Step 3 — Algebra, unit trading, and the freeze gate

### 3.1 Algebraic rewrites

Prove each against the `float64` oracle before relying on it: concatenating narrow contractions
into one wide one; factoring shared terms (`Q·A + Q·(K·B) → Q·(A + K·B)`); online accumulation
instead of materialized intermediates.

### 3.2 Unit trading — a search, not a list

Moving work between units — above all VPU work onto an idle MXU — is one of the largest and least
obvious levers, and no profile suggests it. **Split each op into its linear and non-linear parts**:
any linear map is a matmul by some constant matrix. Try the **folds** first — a scale folded into a
weight, a transpose into `dot_general` dimension numbers.

**Four costs, per phase:** (1) padding — price the padded FLOPs with {{agent:tpu-roofline}};
(2) critical path and slack — it pays only if the VPU pass is on the critical path (M8) *and* the
MXU has slack in that phase, so allow per-phase dispatch; (3) precision — the oracle decides;
(4) the constant matrix's VMEM and source. The rule runs in reverse too, and covers a transpose
into dimension numbers, a gather onto a SparseCore, a small matmul off the MXU.

**The search runs here, in your own context.** Every VPU or XLU pass on the critical path, and every
op whose unit is still a choice after Step 0, gets a row in the whiteboard's **unit-trading table**:

| op | linear part | candidate formulation | padded cost per phase | slack per phase | precision plan | matrix source | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |

Search upstream for how jax and tokamax express the same ops (a `dot` or `einsum` against a
constant, work folded into dimension numbers), the wiki's stage-3 rows, and first principles. Every
row ends in a candidate, "linear part too small to pay" with the number, or "no formulation found"
with what was searched. A surviving candidate becomes a probe arm, beside a removal arm showing the
pass it replaces is on the critical path.

### 3.3 The freeze gate — stop before coding

No file appears under `variants/` until every item is checked in the current whiteboard version.
Then record its md5.

| # | item | what proves it |
| --- | --- | --- |
| 1 | operating points locked | three phases, per-device shapes, exact dtypes, from the roofline config |
| 2 | floors per phase | F1 and F2, with F2's legality marked |
| 3 | primitives audited, deletion evaluated | the inventory, and why each surviving buffer survives |
| 4 | algebra proved | every rewrite checked against the `float64` oracle |
| 5 | unit-trading table complete | every row ends in a verdict |
| 6 | incumbent profile recorded | program name, tiles, VMEM, buffer counts, from artifacts |
| 7 | API ledger complete | every API has a cited upstream example in the form it will be used |
| 8 | affordances declared | every row scored |
| 9 | working set proven, per phase | exact bytes ≤ VMEM limit, arithmetic shown |
| 10 | pipeline axis named | which axis, and what overlaps on it |
| 11 | stage A does not rule out stage B | the interface check from 2.7 |
| 12 | cost classes filled | every row a verdict or `UNEXAMINED: <why>` |
| 13 | fault control designed | the faults, and which cell classes discriminate each, per phase |
| 14 | predictions written | signed, falsifiable, per lever, per phase (M5) |
| 15 | sweep drivers computed | the work driver at every cell of every planned sweep (M6) |
| 16 | denominator recorded | `measured` / `modelled` / `UNKNOWN`, plus the overlap bound |

---

## Step 4 — Lowering, then the affordance audit

1. **Translate** the frozen design into `variants/<name>.py`, copying each API's form from its
   ledger row. Keep the wrapper and the kernel body in separate functions. Earlier variants stay
   importable beside the new one.
2. **Interpreter dry-run** — `pltpu.InterpretParams()` on CPU, in your shell, at no TPU cost.
3. **A compile or lowering error goes to upstream first.** Find the test that exercises the failing
   API or op and diff your usage against it before trying variations.
4. **Accuracy test** — the candidate passes at every phase cell **and** every injected fault fails.
5. **Affordance audit** — check the code and its compiled artifact against every row of 2.6. An
   affordance declared `planned` and absent in code is a bug.

---

## Step 5 — Probe, measure, deliver

### 5.1 Arms

One job, one arm per chip, one frozen question in `probes/<nn>-<slug>/README.md`. Read the chip
count from `{{runtime}} compute inspect --json`. The recipe `run.sh` applies the job's plugin
patch, runs `python -u probe.py` and writes the result file; submit it as in "TPU jobs".

- **Arm 0** is always the incumbent, verbatim (M4).
- The other arms are levers from different cost classes, or removal arms backing an attribution
  (M8). A one-parameter sweep is allowed only when the question *is* a knee, and M6 holds.

Submit every independent probe before harvesting any, each on its own slot, and wait with
`{{runtime}} compute job wait` in the background (M3).

### 5.2 The measurement rules

**M1 — Build every callable before the timing loop.** JAX caches on the identity of the wrapped
callable, and `functools.partial` has no value equality — hoist the function object, not just the
`jit`. Compile every `(arm, cell)` into a dict first, then time by indexing it. Print per-unit
compile seconds; hold rep 0 out of the median. A late rep costing what rep 0 cost is recompiling.

**M2 — The chained clock, and the hoist check that guards it.** Barrier the **loop carry**, never
the loop-invariant operands, or XLA hoists the call and M invocations run once. All three
defences: (1) assert the HLO's `tpu_custom_call` count equals the expected multiple of M, per unit;
(2) per-call cost from an M-sweep least-squares slope, never divide-by-M; (3) treat a raw
`T(M_hi) / T(1)` near 1, or any reading under the floor, as a hoist. Never stack heavy cell groups
on one chip.

**M3 — Dispatch, never block.** Never `sleep` in the foreground. A dependent job waits: a
correctness gate that has not returned has not passed.

**M4 — Same artifact, same job, same surrounding work.** Arm 0 is the stack head's file,
byte-identical and md5-gated; incumbent and candidate run in the same job, interleaved on the same
chips. Print both measured regions' XLA op breakdowns and diff them; an unaccounted non-kernel op
in one region fails the job.

**M5 — Freeze predictions before dispatch.** Signed, falsifiable, per lever and per phase, plus the
abort condition. Record the README's md5 before submit and re-check it on the target.

**M6 — The sweep must move its driver.** Compute grid steps, tiles, bytes moved or DMA count at
every cell, from the code that builds the kernel, and assert it moves the way the prediction
assumes. A flat driver voids the sweep before it costs a job.

**M7 — Correctness: show the oracle can fail.** Report the fault control's catch rate per cell
class with every correctness claim. A class blind on production's cells is not a pass.

**M8 — Performance: show the cost can be removed.** *"X costs N µs"* needs a removal arm (timing
only) whose delta is about N at that phase. A gap split across parts is a sum of direct per-part
arms against an independently measured whole; the difference is **unattributed**. Instruments that
agree count once unless they reach the answer by different routes. Until backed, an attribution is
`[modelled]` and may not justify a redesign or appear as a result.

**M9 — Measure at the scope the work lives, and compose with a ladder.** Ship on whole-call time per
phase against the incumbent. Confirm a stack of levers with a forward ladder plus at least one
leave-one-out; disagreement over ~3 pp means the levers interact. A residual over 3% between the
ladder's top and the full candidate voids the attribution. No phase may regress.

### 5.3 The probe record

After each probe, write `probes/<nn>-<slug>/record.yaml` from the job, never from a template, and
work from it rather than the log:

```yaml
probe_id: "<slug>"
question: "<the frozen question>"
prediction_freeze: { readme_md5: "<md5>", frozen_at: "<iso8601, before submit>" }
job_id: "<job_id>"
status: "SUCCESS | FAILED_ALIGNMENT | COMPILER_REJECT | TIMEOUT | VOID"
clock_check: { custom_call_count: "<expected vs observed>", m_sweep_slope_used: <bool>, hoist_ratio: "<T(M_hi)/T(1)>" }
region_symmetry: "<non-kernel op diff: none | listed>"
driver_check: "<pass | VOID>"
arms:
  - { name: "<arm>", per_phase: { prefill: "<us, delta>", decode_first: "<>", decode_last: "<>" }, oracle: "<PASS | FAIL>" }
fault_control: "<caught n/N per cell class; blind classes named>"
attributions: [{ claim: "<X costs N us at phase>", backing: "<removal arm id | UNBACKED>" }]
profiling_summary: "<what binds, per phase, from the tpu-diagnostics report>"
predictions_scored: "<within range | refuted, with the number>"
refreeze_triggered: "<no | which modelled number>"
route_action: "Adopt | Re-freeze | Step 0 | Step 2 | Step 3 | Step 5"
distilled_strategy: |
  lever: <>; hardware: <chip>; phase: <phases>
  result: <delta> [measured: <job_id>]; axis/config: <required for a null>
  mechanism: <>; scope: <>
```

Copy `distilled_strategy` into `STRATEGY.md`, **including when the arm lost**.

### 5.4 Re-freeze when the design's numbers are refuted

**Trigger:** a measurement lands outside the frozen range of a modelled number the lever ranking
depends on. **Before the next probe:** write `WHITEBOARD.v<n+1>.md` beside the untouched `v<n>`,
open it with a modelled → measured table, recompute the floors and the lever ranking, re-run the
freeze gate on what changed, and send {{agent:tpu-critical-review}} (`kind: claim`) the new version's
attributions with the raw artifacts behind them. Predictions made under `v<n>` are scored against `v<n>`. **Stage B
always opens with a re-freeze** on stage A's measured numbers.

### 5.5 Routing from the profiling result

Capture the probe's trace inside the probe job, as in 1.5, and send it to {{agent:tpu-diagnostics}}
with `fields` naming every signal below. Route on what binds:

| signal | route |
| --- | --- |
| XLU transposes / relayouts | Step 3 — layout, scan order, or a unit trade into dimension numbers |
| VPU saturated, on the critical path | Step 3.2 — unit trading |
| DMA wait | Step 2.6 — buffering depth, async writes, the pipelining axis |
| spills | tile knee, in Step 5 |
| ratio to floor **widens** as work is removed | fixed cost binds — back to Step 0 or cost class 6 |

Cap each question at 2–3 probes. Not converging means the whiteboard is wrong, not the parameters.

### 5.6 Deliver a stage

A stage — and each stage-B rung — is delivered when it has: a whole-call, per-phase result against
the incumbent in one job, with replicates; a correctness job with the fault control; every
attribution backed per M8; and a {{agent:tpu-critical-review}} `review-request`, `kind: claim`, carrying the job ids,
artifact paths, recipe paths and image digests — the reviewer recomputes, it does not read prose.
Then apply the stage-B gate: A delivered, and the fusion prize above noise against the updated F1.

---

## Finishing: patch, apply check, reviews, result

### 1. Commit and cut the patch

Write the delivered variant into the checkout's `block.file`, wired the way production calls it,
and send {{agent:tpu-code-stack}} a `commit` with `worktree` `checkout.path`, `caller: writer`
and the lever as `message`. It refuses a change outside `vllm_torchtpu/` or off `base`; fix the
files and send it again. Copy its `plugin.patch` to `<campaign>/final/kernel_patch.diff`; its
`plugin` is the result's `plugin`.

### 2. Apply check on the stack head

Before returning, prove the patch applies on the requested head, in a compute job on a slot. The
recipe runs {{script:common/vllm_patches/apply_vllm_patch.sh}} on `kernel_patch.diff` on
`stack_head.image` and prints `apply-exit <code>`:

- **0**: `applies_to.exit_code` 0.
- **4**: no verdict. The patch is cut from `base`, the image's own plugin commit, so it fails only
  when the campaign repo's `base` is not this image's: return `failed` with
  `failure_reason: base mismatch: <base_sha> vs <stack_head.image>`, whatever the probes measured.
- **2 or 3**: infrastructure, not a verdict. Relaunch within the limit, then return `failed`.

Record the image, `checkout.revision` as `stack_sha`, the code and the job id as `applies_to`, and
save the log under `final/`.

### 3. Promote `STRATEGY.md` into the wiki

Follow {{reference:write-tpu-kernels/PROTOCOL.md}}. The filter: *would this change what someone
optimizing a different kernel does next month?* Also sweep the report's rejections, refuted
predictions, correctness findings, unit-trading verdicts, API-ledger surprises and carried items.
The installed wiki is read-only: write new `raw/` files, `strategies/` entries and `INDEX.md` rows
under `<campaign>/wiki/` with the same layout, and list them in the report.

### 4. Write `<campaign>/final/final_report.md`

Cover, per phase: block and per-device shapes, dtypes, sharding; chip; denominator and overlap
bound; F1, F2, F2 legality and what binds `[modelled]`; stage A and B status, job ids and review
verdicts; baseline vs candidate µs and speedup for each phase and ragged decode, with replicates;
correctness with fault-control catch rates and blind classes; levers with mechanism, delta and
backing arm; rejected levers with the axis of each null; cost-class coverage with the unexamined
rows; affordances absent; unit-trading verdicts; re-freezes; carried findings with their owner;
promoted wiki entries; the branch, kernel commit, patch path and the apply check. The unexamined
rows, absent affordances and carried findings are never omitted.

### 5. Reviews before returning

| outcome | before returning |
| --- | --- |
| a candidate (`patched`) | **{{agent:tpu-critical-review}} with `kind: claim`, always**, on `final_report.md` with the raw artifacts: job ids, file paths, recipe paths. You do not review your own claims. |
| `at_floor` | {{agent:tpu-critical-review}} with `kind: stop`, `termination_class: floor` — always if cost classes are unexamined or affordances absent. Give it the whiteboard versions, both floors and the measured values. |
| `blocked` on the kernel | {{agent:tpu-critical-review}} with `kind: stop`, `termination_class: blocked`, the verbatim failure, a minimal reproduction, the repairs tried, and the upstream examples the failing usage was diffed against. |

If the stop review returns `EXIT PREMATURE`, run its top continuation within the budget before
returning.

**Accuracy.** The oracle (M7) gates the kernel's numerics. A win still owes an end-to-end mmlu_pro
on the served stack, which the caller's tpu-serving-bench eval runs on the combined stack, so list
that check under `carried`.

### 6. Verdict

Against the verbatim incumbent, whole-call, per phase, on the stack head:

| verdict | when |
| --- | --- |
| `win` | faster above the benchmark's spread in at least one phase, no phase regressed, oracle and fault control pass, {{agent:tpu-critical-review}} passed, apply exit 0 |
| `tie` | every phase within the spread |
| `loss` | any phase regressed |
| `invalid` | the oracle failed, a fault class is blind on production's cells, or {{agent:tpu-critical-review}} did not pass |

Apply exit 4 is no verdict: it is `failed`, base mismatch (step 2).
