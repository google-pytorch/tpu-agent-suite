---
scripts:
- scripts/tpu-kernel-development/search_api.py
- scripts/tpu-kernel-development/hw_probe.py
- scripts/tpu-knowledge-base/kb_lookup.py
- scripts/tpu-knowledge-base/catalog.py
- scripts/tpu-knowledge-base/kb_summarize.py
- scripts/tpu-knowledge-base/rank.py
references:
- references/tpu-kernel-development/concepts
- references/tpu-kernel-development/tpu_hardware_facts.md
- references/tpu-kernel-development/design/sparsecore.md
- references/tpu-kernel-development/hw_probe_catalog.md
---

# Fix kernel compilation

The kernel-worker reads this in Phase 2 when `{optimized_kernel_path}` fails to
compile and attempts remain. You fix the compilation errors yourself, in this
context, while preserving the optimization strategy.

**TPU VM Execution Requirement**: This validation phase requires execution on
the TPU VM.

-   When execution on accelerator compute is required, execute via `{{runtime}} compute run --code {optimized_kernel_path} --timeout 180`.

## Context

-   Kernel file path: `{optimized_kernel_path}`
-   Optimization plan path: `{kernel_plan_path}`
-   Current Compilation Error: `{error_trace}`, from Phase 2's compile run

-   Compilation History (All Attempts): `{compilation_history}`

-   TPU Hardware Context: `state.tpu_spec` in `<work>/state.json`

## Tools

You have three primary tools to fix the kernel: `read` (to inspect kernels and
documentation), `write` (to output the fixed kernel), and `shell` (to execute
helper scripts like `{{script:tpu-kernel-development/search_api.py}}`).

Follow this deterministic workflow:

### Step 1: Open the Kernel Optimization Router
Get it with `python3 {{script:tpu-knowledge-base/kb_lookup.py}} kernel-optimization-index` and `read` the file it names in `paths`.
This is your main router. It details the core thesis, the three sinks of lost time, tiling theory, and category routing.

These pages come from tpu-knowledge-base. Reach every page, authored or fetched, by keyword with `python3 {{script:tpu-knowledge-base/kb_lookup.py}} <keyword>`; it prints a JSON record whose `paths` name the pages, already on disk; read those. Resolve by keyword rather than guessing or listing paths yourself: generated and version-resolved answers never live in `pages/`. A key the catalog does not serve comes back in `unmatched` with `status: uncited`: go on without it, and name it in what you return.

### Step 2: Load Mandatory Baseline Files (Required for ALL kernels)
You MUST always load the following files using the 'read' tool: `{{reference:tpu-kernel-development/concepts/grid_blockspec.md}}`, `{{reference:tpu-kernel-development/concepts/matmul.md}}`, `{{reference:tpu-kernel-development/concepts/pipelining.md}}`. You MUST also load the hardware page for the chip you are targeting, named below.

To pick the hardware page, read `state.tpu_spec` from `<work>/state.json`; the kernel-worker resolved it once in Phase 0 step 3 from `{{runtime}} compute inspect --json`. Look up the page it names with `python3 {{script:tpu-knowledge-base/kb_lookup.py}} <state.tpu_spec.hw_page>` and read the file it returns, and size against its `accelerator`, `topology`, and `hbm_gb_per_device` (or `hbm_gb` when that is `null`). Do not probe the target yourself and never run `jax.devices()` locally: that describes the machine your shell is on, not necessarily the target. Phase 0 step 3 stops the run when the chip cannot be resolved, so `state.tpu_spec` is always set by the time you get here; if it is not, stop and report that rather than assuming a chip.

### Step 3: Retrieve Category-Specific Concepts & Documentation
Use `read` on the router and the pages and references named in this prompt EXTENSIVELY to retrieve Pallas/JAX/TPU documentation, optimization patterns, and examples. Consult the relevant concept docs (e.g., `{{reference:tpu-kernel-development/concepts/grid_blockspec.md}}` for index map/rank issues, `{{reference:tpu-kernel-development/concepts/pipelining.md}}` for async DMA/pipeline lowering, the `vmem` page (`python3 {{script:tpu-knowledge-base/kb_lookup.py}} concept-vmem`) for memory allocation, the `padding` page (`python3 {{script:tpu-knowledge-base/kb_lookup.py}} padding`) for MXU alignment).

1.  `read` and `write`: To read the kernel file, read documentation, and to write your fixed kernel.

2.  `shell`: To run `{{script:tpu-kernel-development/search_api.py}}` for looking up specific
    API definitions and signatures when you need precise technical details.

    *   Invoke via CLI:

        ```bash
        python3 {{script:tpu-kernel-development/search_api.py}} -- "<api_name>"
        ```

        where `<api_name>` is the API you want to search. Run this command
        every time these instructions say to "query".

3.  `shell`: To run a lowering hardware probe through
    `{{script:tpu-kernel-development/hw_probe.py}}`, which settles whether a
    construct lowers at all on this TPU and toolchain, instead of guessing
    from the error.

    *   Lowering and legality probes live in the `lowering`, `hardware` and
        `sparsecore` groups;
        {{reference:tpu-kernel-development/hw_probe_catalog.md}} lists each
        with its question and the three steps (`build`, `compute run`,
        `record`) that run one. Use one when the error is about whether an
        op, dtype or index form is supported, not about your usage of it.
        Never pass `--allow-destructive`.

IMPORTANT: You MUST consult the `kernel-optimization-index` page and load the required concept/hardware pages while diagnosing and fixing compilation errors to ensure accuracy. Do not rely on pre-trained knowledge alone - always verify with current documentation.

--------------------------------------------------------------------------------

## Your Task

Fix ONLY the compilation errors in the kernel file. DO NOT modify the
optimization strategy unless it directly causes compilation failures.

### Step 0: Check for System/Connection Errors

If `{error_trace}` contains the string `FATAL_CONNECTION_ERROR`:

-   This is an unsolvable infrastructure error (e.g., SSH tunnel down, TPU
    unresponsive).
-   **DO NOT** attempt to write any code fixes or research the error.
-   **Immediately halt**: record `ESCALATE_SYSTEM_ERROR: <details of the
    error>` and follow step 3 of the kernel-worker's Strict Debuggability
    protocol, which stops the workflow.
-   **STOP HERE**.

### Step 1: Read Current State

1.  Use the `read` tool to read the kernel file at
    `{optimized_kernel_path}`.
2.  If `{kernel_plan_path}` is provided, read the optimization plan using
    `read` to understand the intended strategy.

### Step 2: Analyze Compilation Errors and History

Review the **current compilation error** and the **compilation history** to
understand:

-   What errors occurred in previous attempts (with summaries).
-   What fixes have already been tried (explicitly listed in history).
-   Whether errors are improving, staying the same, or getting worse.
-   Patterns that might indicate the root cause.

**Learn from previous fix attempts:** If previous attempts show a pattern (e.g.,
"Fix: Changed block size" but error persists), this suggests the approach isn't
working and you need a different strategy.

### Step 2.4: Check the Known Legality Rules and Error Catalogue

Before changing code, check whether the error is a known rule:

-   {{reference:tpu-kernel-development/tpu_hardware_facts.md}} covers
    block-shape legality, what may be a runtime value (DMA extents and
    offsets, lane vs sublane indices), and DMAs that must be waited.
-   For a SparseCore kernel,
    {{reference:tpu-kernel-development/design/sparsecore.md}} has a table of
    error messages and the fix for each.
-   **If an error message names a compiler flag** (for example
    `needs_layout_passes`), retry with that flag before concluding the
    construct is unsupported.
-   A rejected construct is a fact about your usage until an upstream test
    (via `{{script:tpu-kernel-development/search_api.py}}`) or a lowering
    probe (via `{{script:tpu-kernel-development/hw_probe.py}}`) shows it
    failing too.

### Step 2.5: MANDATORY Research (DO THIS BEFORE WRITING ANY CODE)

**BEFORE making any changes, you MUST consult the `kernel-optimization-index` page and/or run `{{script:tpu-kernel-development/search_api.py}}` to research:**

1.  **Check the compilation error against known patterns & concept docs:**
    -   Map the error symptom directly to the relevant concept file under `{{reference:tpu-kernel-development/concepts}}/` and inspect its code examples:
        *   **BlockSpec / Rank / Index mismatch** (`ValueError`, shape/rank errors) → Read `{{reference:tpu-kernel-development/concepts/grid_blockspec.md}}` for valid `index_map` and dimension-squeezing examples (`None`).
        *   **Pipelining / Lowering failure** (`emit_pipeline`, async DMA) → Read `{{reference:tpu-kernel-development/concepts/pipelining.md}}` and the `compute-communication-overlap` page (`python3 {{script:tpu-knowledge-base/kb_lookup.py}} concept-compute-communication-overlap`).
        *   **Matmul / Contraction errors** → Read `{{reference:tpu-kernel-development/concepts/matmul.md}}`.
        *   **VMEM Overflow / Resource allocation** (`Cannot allocate VMEM`) → Consult the `vmem` page (`python3 {{script:tpu-knowledge-base/kb_lookup.py}} concept-vmem`) and the hardware page you loaded in Step 2 (shrink tile sizes by factor of 2; keep multiples of 128 on ≤v5p or 256 on v6e+).
        *   **MXU Alignment / Padding failure** → Read the `padding` page (`python3 {{script:tpu-knowledge-base/kb_lookup.py}} padding`).
2.  **Query each Pallas API involved (Run `{{script:tpu-kernel-development/search_api.py}}` for definitions, router/concepts for patterns):**
    -   Identify Pallas functions in the error trace (`pl.pallas_call`, `pl.BlockSpec`, `pltpu.emit_pipeline`, etc.).
    -   Run `{{script:tpu-kernel-development/search_api.py}}` with the exact API name to get its definition and parameters.
    -   Consult the `kernel-optimization-index` page and concept docs for broader documentation.
    -   Example for search_api:
        `python3 {{script:tpu-kernel-development/search_api.py}} -- "jax.experimental.pallas.pallas_call"`

3.  **Search for examples (Use concept docs or `{{script:tpu-kernel-development/search_api.py}}`):**
    -   Consult the `{{reference:tpu-kernel-development/concepts}}/*.md` files for end-to-end working code examples of the pattern you need.
    -   Alternatively, run `{{script:tpu-kernel-development/search_api.py}}` with the API name, as the generated definition may include usage examples.
    -   Example for search_api:
        `python3 {{script:tpu-kernel-development/search_api.py}} -- "jax.experimental.pallas.pallas_call"`

**DO NOT skip this step.** Even if you think you know the answer, verify it
first. API signatures and requirements change between versions.

### Step 3: Fix the Compilation Errors

**MANDATORY REQUIREMENT: You MUST write a fixed version of the kernel file in
this step.** Even if you're uncertain about the fix, you must attempt a
correction and write the file. The validation loop will test your fix and give
you another chance if it fails.

**REMEMBER:** You have already completed Step 2.5 and gathered research
information. Use those results to guide your fixes.

**CRITICAL - Learn from History:**

-   If this is attempt 2 or 3, review what was tried before.
-   DO NOT repeat the same fix if it already failed.
-   If errors are getting worse, consider reverting to a simpler approach.
-   If errors are similar across attempts, the root cause may be deeper than
    syntax.
-   **If you're stuck after multiple attempts with the same error, try a
    completely different approach** (e.g., simplify block sizes, change grid
    config, modify memory layout).

**CRITICAL RULES:**

1.  **Preserve Optimization Intent:**
    -   DO NOT change block sizes (bM, bK, bN, etc.) unless they directly cause
        compilation errors.
    -   DO NOT change grid dimensions unless incorrect for the operation.
    -   DO NOT change memory layout strategy (how data flows through memory
        hierarchy).
    -   DO NOT change the algorithmic approach (e.g., tiled matmul pattern,
        accumulation strategy).
2.  **Fix Only What's Broken:**
    -   Correct Python syntax errors.
    -   Fix import statements and module paths.
    -   Correct Pallas/JAX API usage (consult the `kernel-optimization-index` page and/or run
        `{{script:tpu-kernel-development/search_api.py}}` to verify correct APIs).
    -   Fix variable references and scoping issues.
    -   Ensure the `main` function is present and correctly calls the kernel.
    -   **Consider the progression of errors** - if the same type of error
        persists, try a different approach.
3.  **Maintain Structure:**
    -   Keep the two-level structure: `kernel` function and `computation`
        function.
    -   Preserve the `main` function that demonstrates kernel usage.
    -   Keep all inline documentation and shape/memory annotations.
    -   Maintain variable names from the original implementation.
    -   **CRITICAL: DO NOT add try-except blocks** - Let errors surface
        naturally for proper validation.
4.  **Enable Debug Mode:**
    -   **ALWAYS add `debug=True` to the `pl.pallas_call()` invocation**
    -   This provides better error messages and helps diagnose compilation
        issues.
5.  **Add Debug Output When Needed:**
    -   For difficult-to-diagnose errors, add regular `print()` statements in
        the `main()` function or computation setup.
    -   Print intermediate shapes, values, or configuration before calling the
        kernel.
    -   Keep the kernel function clean - debug prints should be outside the
        kernel in the main function.

### Step 4: Write the Fixed Kernel (REQUIRED - NO EXCEPTIONS)

Use the `write` tool to overwrite the kernel file. Save the plan to the
exact path provided in `{optimized_kernel_path}`.

The fixed kernel MUST include:

-   All necessary imports.
-   The `kernel` function with correct Pallas syntax.
-   The `computation` function that calls `pl.pallas_call`.
-   A `main` function that:
    -   Creates sample input arrays.
    -   Calls the computation function.
    -   Prints or validates output.
    -   Demonstrates the kernel works standalone.

### Step 5: Summary (REQUIRED)

After writing the fixed file (which you MUST do in Step 4), write a
**concise** summary in the following format. Phase 2 appends it to
`{compilation_history}`, then compiles again:

```
FIX_SUMMARY:
- Error: [What compilation error occurred in THIS attempt]
- Cause: [What caused the error]
- Fix: [What change was made to fix it]
```

**CRITICAL - This summary is required because you wrote a fix:**

-   **ONLY describe the changes made in THIS current attempt**
-   **DO NOT summarize previous attempts or list all fixes tried so far**
-   **DO NOT reference the compilation history in your summary**
-   Structure as three separate lines: **Error**, **Cause**, and **Fix**
-   **Error**: The specific compilation error from THIS attempt (e.g.,
    "ValueError: Block shape must have same number of dimensions")
-   **Cause**: The root cause of THIS error (e.g., "BlockSpec used 2D
    block_shape (64, 128) but input array q is 3D")
-   **Fix**: The specific change made in THIS attempt (e.g., "Updated
    block_shape to (None, 64, 128) to match 3D input and squeeze first
    dimension")
-   Keep each line brief but specific.

**Example:** `FIX_SUMMARY: - Error: ValueError: Block shape for args[0] (=
(Blocked(block_size=4), Blocked(block_size=64))) must have the same number of
dimensions as the array shape (8, 4, 64) - Cause: BlockSpec block_shape was 2D
(4, 64) but input q is 3D with shape (seq_len=8, num_heads=4, head_dim=64),
causing a rank mismatch - Fix: Updated q_spec block_shape from (4, 64) to (None,
4, 64) to match the 3D input dimensions, using None to squeeze the seq_len
dimension`

--------------------------------------------------------------------------------

## Important Notes

-   If the optimization plan fundamentally conflicts with Pallas capabilities,
    record that in `<work>/tpu_kernel_debug_history.md`.
-   If you must change block sizes or grid config to fix compilation, explain
    why.
-   Always verify API usage with the `kernel-optimization-index` page or `{{script:tpu-kernel-development/search_api.py}}` before making
    changes.
-   The goal is a compiling kernel that still achieves the planned
    optimizations.
-   Compilation errors are often simple syntax/API issues that don't require
    strategy changes.
