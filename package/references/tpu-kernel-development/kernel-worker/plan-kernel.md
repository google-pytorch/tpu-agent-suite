---
scripts:
- scripts/tpu-kernel-development/search_api.py
- scripts/tpu-kernel-development/ledger.py
- scripts/tpu-knowledge-base/kb_lookup.py
- scripts/tpu-knowledge-base/catalog.py
- scripts/tpu-knowledge-base/kb_summarize.py
- scripts/tpu-knowledge-base/rank.py
references:
- references/tpu-kernel-development/cuda-to-pallas.md
- references/tpu-kernel-development/cuda-portability.md
- references/tpu-kernel-development/concepts/grid_blockspec.md
- references/tpu-kernel-development/concepts/matmul.md
- references/tpu-kernel-development/concepts/pipelining.md
---

# Plan the kernel

The kernel-worker reads this in Phase 1 step 1 to create or revise a detailed
optimization plan for a Pallas kernel. You write the plan yourself, in this
context, with your own tools.

### What the run is built on, and what you may read

`<work>/state.json` carries the run's shape. Read it before anything else:

*   `state.primary.language` — `jax` or `pytorch`. When it is `pytorch`,
    `<work>/base.py` is a **port**, not the user's code.
*   `state.primary.context_path` — `<work>/torch_context.md`. Present only
    when the primary is not `jax`.
*   `state.primary.source_path` — the user's untouched input.
*   `state.ideas_ledger_path` and `state.reference_alignment_path` — present
    only when the user supplied a reference kernel; `null` otherwise.
*   `state.library_check` — what tpu-kernel-development found when it checked Tokamax
    for this op. If `result` is `"build"`, Tokamax already has a TPU kernel
    and the user still wants a new one. `summary` describes that kernel. Plan
    to beat it, or explain why a custom kernel is needed. You don't need to
    look it up again.
*   **`<work>/ref/` is OFF LIMITS to you.** The reference briefs in there have
    already been reconciled against the primary and distilled into the ledger,
    with each idea tagged by how much it can be trusted. Reading the raw CUDA
    behind the ledger's back reintroduces exactly the anchoring the ledger
    exists to prevent.

### Step 1: Determine Your Task

Identify whether you are creating a **NEW plan** (Iteration 0) or performing a **REVISION / NEXT ITERATION plan**.

*   **NEW Plan (Iteration 0):**
    *   First iteration of optimization on `{base_kernel_path}`.
*   **REVISION / NEXT ITERATION Plan:**
    *   Previous plan and code files are provided (`{previous_kernel_plan}`, `{previous_optimized_kernel}`).
    *   Execution results from prior iterations are provided (`{previous_compilation_status}`, `{previous_testing_status}`, `{previous_autotune_summary}`, `{previous_profile_summary}`).

### Step 2: Gather Context and Decide Optimization Base

**Inputs Provided:**
- Original Base Kernel: `{base_kernel_path}`
- Shared Test Harness: `{test_file_path}`
- Previous Kernel Plan: `{previous_kernel_plan}`
- Previous Optimized Kernel: `{previous_optimized_kernel}`
- Previous Compilation Status: `{previous_compilation_status}`
- Previous Testing Status: `{previous_testing_status}`
- Previous Autotune Summary: `{previous_autotune_summary}`
- Previous Profile Summary: `{previous_profile_summary}`

**For NEW and REVISION Plans alike:**

1.  **Read base kernel:** Use `read` to inspect the original source code at `{base_kernel_path}`.
1b. **Read the primary context brief, when there is one.** Read
    `state.primary.language` from `<work>/state.json` first:

    *   **`"jax"`** — `state.primary.context_path` is `null`. `base.py` *is*
        the user's own code; nothing else to read. Skip to step 2.
    *   **`"pytorch"` or `"cuda"`** — `base.py` is a **port**, not the user's
        code. You MUST `read` `state.primary.context_path` before planning
        anything. It was written once, at the front of the run, by an agent
        that read the original source line by line; it is the only place the
        original's intent survives. Planning from `base.py` alone throws that
        away.

    For a **PyTorch** brief specifically, the two highest-value sections are:
    *   **§4 Materialization Map** — the fusion boundaries. Every intermediate
        the op graph writes to HBM and reads back is a round trip a single
        Pallas kernel can collapse, and that is where this kernel most likely
        earns its keep.
    *   **§6 Dynamic-Shape Hazards** — anything with no clean TPU form. Read it
        before you commit to a design, not after the compiler rejects one.

    Also check `state.primary.port_verified`:
    *   `true` — `base.py` was checked against values captured by running the
        user's own module. You can trust it as the specification.
    *   `false` or `null` — the port was never independently verified. Treat
        §2 of the brief as authoritative over `base.py` wherever they differ,
        and say so in your plan rather than silently following one.

    How to use a CUDA brief — this is where planners most often go wrong:

    *   **Section 2 ("What It Computes") is your specification.** It, not
        `base.py`, defines the semantics you must preserve. If the port in
        `base.py` and the spec in Section 2 disagree, say so in your plan
        rather than silently following one — a contradiction there means the
        run's baseline is wrong and every measurement after it is meaningless.
    *   **Section 7 ("CUDA → TPU Translation Notes") is a source of hypotheses,
        not a design.** Treat its mapping table as candidate strategies to
        evaluate against your roofline analysis, not as a plan to execute.
    *   **Section 9 ("Optimization Opportunities") is the highest-value input
        you get.** The fusions the CUDA author already performed are exactly
        the HBM round trips the JAX baseline reintroduces, and collapsing them
        back is usually where a Pallas kernel earns its keep here. Start there.
    *   **Sections 4 and 5 describe SIMT bookkeeping. Do NOT transliterate
        them.** There is no TPU counterpart to a thread index, a warp shuffle,
        a `__syncthreads()`, or bank-conflict padding, and a kernel that
        imitates the CUDA block structure will be slower than the baseline, not
        faster. Take the *data flow* from those sections — what is reused, what
        stays resident, what is read once — and re-derive the tiling from TPU
        hardware limits yourself.
    *   **Section 6 constrains your numerics.** Match the accumulator dtypes
        the original used. Do not "improve" precision the source deliberately
        gave up, and do not give up precision it kept.
    *   The original's block and tile constants are **evidence about reuse, not
        values to copy.** A CUDA `BLOCK=256` says something about the working
        set; it says nothing about what fits in the VMEM of the chip you are
        targeting or aligns to the MXU's 128×128.

    A PyTorch brief reads the same way, with Section 7 naming the fusion
    boundaries XLA will leave behind and Section 9 naming the intermediates
    that get materialized to HBM for no reason.
1c. **Read `state.fusion_context` when it is set — it is the served graph, not
    this harness.** The caller's profile of the real workload recorded which
    fusion the target sits in there (`fusion`) and what XLA fused with it
    (`fused_with`). Your kernel is one custom call, and a custom call is a
    fusion boundary: whatever XLA fused into the target in production -- a
    norm, an activation, a residual add -- is computed separately once your
    kernel replaces it, with its own HBM round trip. A decode MLP kernel that
    measured 1.23x faster than XLA standalone ran 2.5-3.2% slower end to end,
    in part because it broke the RMSNorm's fusion into the first matmul. So
    either take the fused neighbours into the kernel, or count the lost fusion
    against the expected gain in §6 of the plan. Its `per_call_us` is the
    production time of the target op: when the harness times the base kernel
    far from it, say so in the plan, because the harness is then not measuring
    the op the profile ranked.
1d. **Read `state.served_calls` and `<work>/input_provenance.md` when they
    exist.** The harness configs
    labelled with a served-call case are the calls the server makes, under the
    configuration `state.serving_config` names; the others are coverage. Weigh the cases by their
    `share_pct` when you choose what to optimize -- a large speedup on a case
    that is a small share of the step buys little end to end -- and do not
    trade a regression on a served-call case for a gain on a coverage case. A
    served-call case with no `share_pct` is a call the server makes that no
    lever ranked: keep it correct and do not regress it, but do not weigh it.
    Where the provenance file marks a value as an assumption (sequence counts,
    lengths, a smaller pool), name that assumption wherever the plan's
    reasoning depends on it.
2.  **Read the shared test harness:** Use `read` on `{test_file_path}`. This
    file was generated once, before this loop started, and is reused unchanged
    for every iteration. It is important because its `get_inputs()` function
    defines the EXACT input shapes, dtypes, and static arguments (e.g.
    `block_size`) every iteration's kernel will be tested and benchmarked
    against. Use these shapes to inform grid sizes and tiling choices so the
    plan you produce is implementable against the real test inputs.
    -   **Do not overfit to the specific shapes/values in `get_inputs()`.**
        Design tiling and grid logic that generalizes across different sizes
        and edge cases (e.g. divisibility, small/large inputs), not one that
        only works for the exact numbers the harness happens to use. The
        harness's job is to sample representative cases, not to define the
        universe of valid inputs.

**For REVISIONS / NEXT ITERATION Plans:**

1.  **Read previous artifacts:** Use `read` to read `{previous_kernel_plan}`, `{previous_optimized_kernel}`, `{previous_autotune_summary}` (if not `none`), and `{previous_profile_summary}` (if not `none`).
    When `{previous_profile_summary}` is `none`, no profile of the previous
    iteration exists: say "no profile" in the plan. Cite a profile, a trace or
    an XProf finding only from a file you actually read -- a citation to
    evidence that was never produced reads later as a measurement.
2.  **Review execution results:** Analyze `{previous_compilation_status}`, `{previous_testing_status}`, `{previous_autotune_summary}`, and `{previous_profile_summary}` to identify what failed or what performance bottlenecks remain.
3.  **Decide Optimization Base — an explicit numerical rule, not a judgement
    call.**

    Read `<work>/state.json`, find the entry in `state.history` whose
    `iteration` equals `N-1`, and call it `prev`. Extract exactly three values:

    *   `prev.compile_ok` — boolean
    *   `prev.test_ok` — boolean
    *   `prev.speedup` — JSON number or `null`; iteration `N-1`'s measured
        speedup over `base.py`, from a paired measurement the harness took in a
        single process.

    Apply this rule literally. Evaluate top to bottom and stop at the first row
    that matches:

    | # | Condition on iteration `N-1` | Base you MUST use |
    |---|---|---|
    | 1 | `prev` missing, OR `prev.compile_ok != true`, OR `prev.test_ok != true`, OR `prev.speedup` is `null` | `{base_kernel_path}` |
    | 2 | `prev.speedup > 1.0` | `{previous_optimized_kernel}` |
    | 3 | `prev.speedup <= 1.0` | `{base_kernel_path}` |

    In words: **iteration `N-1`'s kernel is worth building on only if it was
    correct and actually beat the baseline.** `speedup > 1.0` means it ran
    faster than `base.py`, so carry it forward. `speedup <= 1.0` means it was no
    faster than the unoptimized code, so there is nothing in it worth keeping —
    restart from `{base_kernel_path}`.

    How to apply it:

    *   The threshold is the literal number `1.0`. Do not substitute a different
        one, and do not soften `<=` into "approximately 1".
    *   A speedup of exactly `1.0` selects `base.py` (row 3): the kernel matched
        the baseline and earned nothing.
    *   Compare the values as read from `state.json`. Do NOT re-derive a speedup
        from `base_time_ms` / `optimized_time`, and do NOT read one out of the
        prose in `{previous_profile_summary}` or `{previous_autotune_summary}` —
        summaries round, restate and omit. `state.json` is the only authority.
    *   This rule fixes the **starting code** only; it does not constrain your
        strategy. When row 1 or row 3 sends you back to `base.py`, you still
        have the full bottleneck diagnosis from the profile summary — use it to
        plan a *different* approach. Restarting from `base.py` is not licence to
        repeat iteration `N-1`'s plan.
    *   You MUST report the numbers you compared in Section 2 of your plan (see
        the Output Requirement below). Phase 5 parses your stated choice back
        into `state.json` as `base_choice`, and the implementation step (Phase 1
        step 2) reads it to know which file to open.

### Step 2b: Consult the Ideas Ledger — AFTER the roofline, never before

**Skip entirely when `state.ideas_ledger_path` is `null` or
`state.reference_trust == "rejected"`.** Most runs have no reference and this
step does not exist for them.

The user supplied a reference kernel — an independent implementation of roughly
the same computation, by someone who already thought hard about it. Phase 0.7
reconciled it against the primary and distilled it into
`<work>/ideas_ledger.json`. That ledger is the *only* channel through which the
reference may reach you; `<work>/ref/` is off limits.

#### The read order is a rule, and it is the point

**You must have written your roofline classification and your VMEM arithmetic
before you open the ledger.** Not "considered" — written, into the plan.

This is the anti-anchoring mechanism and it is the difference between learning
from a reference and being captured by one. A planner who reads a clever CUDA
trick first will construct a plan around it and find reasons afterwards. A
planner who has already written *"memory-bandwidth bound; the naive form moves
4.2 GB per call; the floor is 0.9 GB"* evaluates that same trick against a
number, and can tell whether it attacks the actual bottleneck or an irrelevant
one. If you have not done the roofline yet, go and do it.

#### Read the alignment verdict first

`read` `state.reference_alignment_path`. The verdict decides what a ledger
entry is worth to you:

| `reference_trust` | What a `LEDGER-*` provenance buys you |
| --- | --- |
| `aligned` | The idea is usable as a primary hypothesis on its own. |
| `partial` | Usable, **except** for entries with a non-null `depends_on_difference`. Read that difference in the alignment doc: if the primary does not have it, the idea's justification does not apply here and adopting it is how you get a slower kernel with no explanation. |
| `divergent` | The reference solves a related but materially different problem. A ledger citation is **not sufficient on its own** — every hypothesis must also carry an independent `ROOFLINE` justification that stands without the reference. |
| `rejected` | Do not read the ledger at all. |

#### List what you may actually adopt

```bash
python3 {{script:tpu-kernel-development/ledger.py}} list <work>/ideas_ledger.json --adoptable
```

This filters out what you must not touch: `NON_PORTABLE` entries, ideas
already refuted by an earlier iteration's trace, and ideas already adopted.
Use the full listing (`list` with no flag) to *read* the refuted and
non-portable entries — they are informative — but never to adopt one.

#### Treat each class completely differently

*   **`ALGORITHMIC`** — the reason to read a reference at all. These are
    statements about the *problem*, not about the GPU: an online/streaming
    recurrence, the choice of what to recompute rather than store, which
    operand stays resident across the loop, the fusion boundary the author
    picked, a numerical reformulation, exploited block structure. They survive
    the hardware change intact. Evaluate them against your roofline and adopt
    the ones that attack the bottleneck you identified.

*   **`STRUCTURAL`** — a real decision whose *value* is hardware-specific.
    **Never adopt the constant.** A CUDA `BLOCK_M=128, BLOCK_N=64` is evidence
    about the shape of the working set and about what the author believed fit
    in fast memory. It says nothing about what fits in this chip's VMEM or
    aligns to a 128×128 MXU. Take the *ratio and the reason*; re-derive your
    own tile sizes from the VMEM budget on the hardware page you loaded, and
    cite the reference only as corroboration. A plan that copies a tile
    constant has transliterated, not translated.

*   **`NON_PORTABLE`** — warp shuffles, `__syncthreads`, bank-conflict padding,
    per-thread register blocking, occupancy targets, atomics-as-reduction.
    `ledger.py` will refuse to adopt these. Read them anyway: knowing that the
    author spent forty lines on a warp-level transpose *because* shared-memory
    bank conflicts punish the naive layout tells you that the layout mattered
    to them, while the transpose itself has no TPU analogue at all. Take the
    fact, discard the machinery.

When an entry's TPU translation is unclear,
`{{reference:tpu-kernel-development/cuda-to-pallas.md}}` is the lookup table the reconciler
used: §1–6 by construct, §7 by kernel family (what a GEMM, FlashAttention,
paged-attention, normalization, quantized-GEMM or MoE design usually keeps
versus drops). `{{reference:tpu-kernel-development/cuda-portability.md}}` gives the
principle behind the split.

#### Record every adoption through the tool

For each idea your plan actually builds on:

```bash
python3 {{script:tpu-kernel-development/ledger.py}} adopt <work>/ideas_ledger.json \
  --id LEDGER-003 --iteration <N> --note "<how this plan uses it>"
```

Never hand-edit the ledger JSON. `ledger.py` enforces the transitions — it is
what stops a `NON_PORTABLE` entry being adopted, stops a refuted idea being
re-proposed, and keeps the history that the final report is built from.

#### Every hypothesis in your plan carries a provenance tag

This is mandatory for all plans, with or without a ledger:

```markdown
### Hypothesis 2 — fuse the rescale into the KV loop
Provenance: LEDGER-003 (ALGORITHMIC, trust=aligned)
Mechanism: removes one HBM pass over S; predicted 4.2 GB -> 2.8 GB per call
Falsifiable as: XProf HBM bytes/call drop >= 30% vs iter 2
```

Allowed values: `LEDGER-<id>`, `ROOFLINE`, `PROFILE-iter<k>`, `WIKI-<page>`.
A hypothesis may carry more than one — and under `divergent` trust, a
`LEDGER-*` tag **must** be paired with a `ROOFLINE` one.

The tag is not bookkeeping. It is what lets the run's final report say which
borrowed ideas actually produced speedups, instead of asserting that the
reference helped. An untagged hypothesis is an unfinished one.

### Step 3: Create or Update the Plan

Create or update a comprehensive optimization plan for the kernel code. The plan
should be structured as a markdown document with the following sections:

## 1. Current Kernel Analysis

-   **Source Provenance** (one line, always present):
    `Input language: <pytorch|jax>` — and when it is not `jax`, the brief
    you read (`<work>/torch_context.md`) plus a
    one-sentence statement of what the original kernel computes, taken from its
    Section 2. If Section 2 and `{base_kernel_path}` disagree about the
    semantics, flag it here explicitly.
-   Brief description of what the kernel does
-   Current implementation approach
-   Identified performance bottlenecks or issues from previous execution/profiling

## 2. Optimization Strategy

-   High-level optimization approach
-   **Every hypothesis carries a `Provenance:` line.** One of `LEDGER-<id>`,
    `ROOFLINE`, `PROFILE-iter<k>`, `WIKI-<page>`, or several. Under
    `reference_trust == "divergent"`, a `LEDGER-*` tag must be paired with a
    `ROOFLINE` one. An untagged hypothesis is unfinished.
-   **For a `pytorch` / `cuda` primary**: which opportunity from Section 9 of
    the primary's context brief this iteration attacks, and — if you are
    deliberately *not* reproducing something the original did — why that
    choice does not transfer to TPU.
-   **When a ledger exists**: a short table of the ideas you adopted this
    iteration and, for any adoptable idea you passed over, one line on why.
    Name any `NON_PORTABLE` mechanism you considered and discarded — showing
    it was weighed is what demonstrates the design was not transliterated.

    ```
    Ledger adoptions (trust = partial)
      LEDGER-001  ALGORITHMIC  adopted  -- unnormalized accumulator, attacks the
                                           HBM bound identified in the roofline
      LEDGER-004  ALGORITHMIC  passed   -- depends on D2 (fused epilogue), which
                                           this primary does not have
      LEDGER-007  NON_PORTABLE noted    -- warp-shuffle reduction; no TPU
                                           counterpart, expressed as a jnp
                                           reduction over the axis instead
    ```
-   **Optimization Base Choice**: Report the Step 2.3 decision with the numbers
    you compared, in exactly this form so Phase 5 and the implementation step
    can parse it:

    ```
    Optimization Base Choice: iter<N-1>
      prev.compile_ok = true
      prev.test_ok    = true
      prev.speedup    = 1.3500
      Rule applied    = row 2 (speedup > 1.0) -> build on <work>/iter<N-1>/optimized.py
    ```

    The first line must be `Optimization Base Choice: iter<k>` or
    `Optimization Base Choice: base` and nothing else. For iteration 1 there is
    no previous iteration — write `Optimization Base Choice: base (iteration 1)`.
-   Key transformations to apply
-   Rationale for each optimization

## 3. Memory Layout and Tiling

-   Proposed block sizes (bM, bK, bN, etc.)
-   Memory layout strategy (HBM, VMEM, SMEM usage)
-   Justification based on TPU specs

## 4. TPU-Specific Optimizations

-   Use of pipelining
-   Prefetching strategies
-   Use of TPU-specific features (matmul units, vector units)
-   Synchronization and memory fence placement

## 5. Implementation Details

-   Grid specification
-   BlockSpec configuration
-   Any special considerations or edge cases

## 6. Expected Performance Impact

-   Expected speedup or performance characteristics
-   When `state.fusion_context` names fused neighbours the kernel leaves
    out: the fusion lost, and the gain net of it
-   Potential risks or limitations
-   Alternative approaches if this doesn't work

## 7. Documentation Requirements

-   All variables in the kernel should have shape comments (e.g., `# Shape:
    (batch_size, seq_len, hidden_dim)`)
-   Memory space annotations for key variables (e.g., `# Memory: HBM`, `#
    Memory: VMEM`, `# Memory: SMEM`)
-   Comments explaining memory transfers between spaces (e.g., `# Transfer from
    HBM to VMEM`, `# Load from VMEM to registers`)
-   Rationale for block dimensions and tiling choices
-   Explanation of any non-obvious indexing or memory access patterns

### Tool Usage & Knowledge Retrieval Strategy

You have three primary tools: `read` (to inspect kernels and documentation),
`write` (to output plans/code), and `shell` (to execute helper scripts like
`{{script:tpu-kernel-development/search_api.py}}`).

Follow this deterministic 4-step workflow:

#### Step 1: Open the Kernel Optimization Router
Get it with `python3 {{script:tpu-knowledge-base/kb_lookup.py}} kernel-optimization-index` and `read` the file it names in `paths`.
This is your main router. It details the core thesis, the three sinks of lost time, tiling theory, and the category routing for your specific kernel type.

These pages come from tpu-knowledge-base. Reach every page, authored or fetched, by keyword with `python3 {{script:tpu-knowledge-base/kb_lookup.py}} <keyword>`; it prints a JSON record whose `paths` name the pages, already on disk; read those. Resolve by keyword rather than guessing or listing paths yourself: generated and version-resolved answers never live in `pages/`. A key the catalog does not serve comes back in `unmatched` with `status: uncited`: go on without it, and name it in what you return.

#### Step 2: Load Mandatory Baseline Files (Required for ALL kernels)
You MUST always load the following files using the 'read' tool: `{{reference:tpu-kernel-development/concepts/grid_blockspec.md}}`, `{{reference:tpu-kernel-development/concepts/matmul.md}}`, `{{reference:tpu-kernel-development/concepts/pipelining.md}}`. You MUST also load the hardware page for the chip you are targeting, named below.

To pick the hardware page, read `state.tpu_spec` from `<work>/state.json`; the kernel-worker resolved it once in Phase 0 step 3 from `{{runtime}} compute inspect --json`. Look up the page it names with `python3 {{script:tpu-knowledge-base/kb_lookup.py}} <state.tpu_spec.hw_page>` and read the file it returns, and size against its `accelerator`, `topology`, and `hbm_gb_per_device` (or `hbm_gb` when that is `null`). Do not probe the target yourself and never run `jax.devices()` locally: that describes the machine your shell is on, not necessarily the target. Phase 0 step 3 stops the run when the chip cannot be resolved, so `state.tpu_spec` is always set by the time you get here; if it is not, stop and report that rather than assuming a chip.

#### Step 3: Retrieve Category-Specific Concepts & Documentation
Use `read` on the router and the pages and references named in this prompt EXTENSIVELY to retrieve Pallas/JAX/TPU documentation, optimization patterns, and examples. This is your PRIMARY source for:
- **Tiling strategies and block size recommendations** for specific operations (e.g., "matmul tiling", "reduction block sizes")
- **Memory layout patterns (HBM, VMEM, SMEM) and best practices**
- **TPU-specific optimization techniques (pipelining, prefetching, memory barriers)**
- **TPU architecture details (HBM, VMEM, SMEM, MXU capabilities, vector units)**
- **API signatures and usage examples (`pl.pallas_call`, `BlockSpec`, `program_id`, etc.)**
- **Performance tuning guidelines and profiling strategies**
- **Common patterns for specific kernel types (matmul, convolution, reduction, etc.)**

**Retrieval Strategy:**
1. **Identify the kernel type first**: use the *Kernel categories → the routing table* section of the router (`kernel-optimization-index`) you opened in Step 1 to classify the kernel into exactly one of the seven categories -- `attention`, `gemm-conv-epilogue`, `streaming-reduction`, `grouped-ragged-indirection`, `state-carry-scan`, `dense-near-roofline`, `cross-chip-collective` -- then fetch the class page that row names with `python3 {{script:tpu-knowledge-base/kb_lookup.py}} <keyword>`, using the keyword the row gives, and `read` the path it prints. One primary page; at most one secondary for straddlers.
2. **Identify specific optimizations**: Check techniques like pipelining, double buffering (`emit_pipeline`), or Megacore splitting.
3. **Verify memory management**: Check VMEM capacity and tiling constraints against your target chip.

#### Step 4: Write the Plan
Use the `write` tool to save your plan to `{kernel_plan_path}`.

1.  `read` and `write`: To read the source kernel, read documentation, and to write your
    plan.

2.  `shell`: For looking up specific API definitions and signatures when
    you need precise technical details.

    *   Invoke via CLI:

        ```bash
        python3 {{script:tpu-kernel-development/search_api.py}} -- "<api_name>"
        ```

        where `<api_name>` is the API you want to search. Run this command
        every time these instructions say to "query".

IMPORTANT: You MUST consult the `kernel-optimization-index` page and load the required concept/hardware pages while creating your plan to ensure accuracy. Do not rely on pre-trained knowledge alone - always verify with current documentation. CRITICAL: Do not use the `search` tool as a substitute for reading the `kernel-optimization-index` page or running `{{script:tpu-kernel-development/search_api.py}}` anywhere during the kernel optimization process.

### Output Requirement

**For NEW plans:**

1.  You **must** use the `write` tool to write the plan as a markdown
    file.
    -   **CRITICAL**: Save the plan to the exact path provided in
        `{kernel_plan_path}`.
2.  After successfully writing the file, go on to Phase 1 step 2 and
    implement it.

**For REVISIONS:**

1.  You **must** use the `write` tool to **overwrite** the existing plan
    file at `{kernel_plan_path}` with your revised version.
2.  After successfully overwriting the file, go on to Phase 1 step 2 and
    implement it.

### TPU Hardware Context

TPU generation and topology come from `state.tpu_spec` in `<work>/state.json`:
`accelerator`, `topology`, `chips`, `hbm_gb`, `devices_per_chip`,
`hbm_gb_per_device` and `hw_page`. The kernel-worker resolves them once, in
Phase 0 step 3, from `{{runtime}} compute inspect --json`, so planning,
implementation and every repair size tiles against the same answer. VMEM size,
core count and MXU shape come from the hardware page `hw_page` names.

The older `{tpu_specs}` placeholder is not populated by anything and never
was — read state instead.

There is no fallback. Phase 0 step 3 stops the run when the chip cannot be
resolved, so do not probe the target yourself, do not run `jax.devices()`
locally, and do not assume a generation.

If neither state nor the runtime can tell you, name the chip explicitly in the
plan and say that you assumed it, rather than planning against an unstated
guess.

### Example Plan Structure:

```markdown
# Kernel Optimization Plan: Matrix Multiplication

## 1. Current Kernel Analysis
The current implementation performs a basic matrix multiplication using JAX's `jnp.matmul`. This is functional but doesn't leverage TPU-specific optimizations available through Pallas.

Current approach: Simple matmul with no blocking or tiling.

Performance bottlenecks:
- No explicit memory hierarchy management
- Missing TPU matmul unit utilization
- No pipelining or prefetching

## 2. Optimization Strategy
We will implement a blocked matrix multiplication kernel using Pallas with the following key optimizations:
1. Tile the computation into blocks that fit in VMEM
2. Use explicit accumulation in output blocks
3. Leverage TPU matmul units through proper block sizing
4. Add pipelining for overlapping compute and memory operations

## 3. Memory Layout and Tiling
- Block sizes: bM=128, bK=128, bN=128
  - Rationale: Aligns with TPU matmul unit dimensions (128x128)
  - Fits in VMEM: ~128KB per block with float32
- Grid: (M//bM, N//bN, K//bK)
- BlockSpecs:
  - A: (bM, bK) moving along M and K dimensions
  - B: (bK, bN) moving along K and N dimensions
  - C: (bM, bN) accumulating along K dimension

## 4. TPU-Specific Optimizations
- Initialize output block to zero only on first K iteration (program_id(2) == 0)
- Use in-place accumulation (+=) to leverage matmul units
- Potential for pipelining in future iterations

## 5. Implementation Details
- Grid: 3D grid (M//bM, N//bN, K//bK)
- Input BlockSpecs with dimension selection lambdas
- Output BlockSpec with accumulation semantics
- Zero initialization guard using pl.when

## 6. Expected Performance Impact
- Expected: 2-5x speedup over naive jnp.matmul for large matrices
- Benefits increase with matrix size due to better memory locality
- Risks: May need tuning of block sizes for optimal performance on specific TPU version
- Alternative: If performance is not satisfactory, consider smaller blocks or adding explicit pipelining

## 7. Documentation Requirements
- All tensor shapes documented inline: A: (M, K), B: (K, N), C: (M, N)
- Memory hierarchy annotations: Input blocks (A_block, B_block) loaded from HBM to VMEM
- Block references: a_ref (bM, bK) in VMEM, b_ref (bK, bN) in VMEM, c_ref (bM, bN) accumulator
- Memory transfer comments: Document when data moves from HBM→VMEM→registers
- Grid indexing explanation: program_id(0)=M block, program_id(1)=N block, program_id(2)=K iteration
```

Remember: Focus on creating a clear, actionable plan.
