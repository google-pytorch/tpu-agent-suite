---
scripts:
- scripts/tpu-kernel-development/search_api.py
- scripts/tpu-knowledge-base/kb_lookup.py
- scripts/tpu-knowledge-base/catalog.py
- scripts/tpu-knowledge-base/kb_summarize.py
- scripts/tpu-knowledge-base/rank.py
references:
- references/tpu-kernel-development/pallas_code_style.md
- references/tpu-kernel-development/kernel_design_method.md
- references/tpu-kernel-development/tpu_hardware_facts.md
- references/tpu-kernel-development/design/sparsecore.md
- references/tpu-kernel-development/concepts/grid_blockspec.md
- references/tpu-kernel-development/concepts/matmul.md
- references/tpu-kernel-development/concepts/pipelining.md
---

# Implement the kernel

The kernel-worker reads this in Phase 1 step 2 to implement a Pallas kernel
following the optimization plan written in step 1. You write the kernel
yourself, in this context, with your own tools.

### ⚠️ CRITICAL: NO ERROR HANDLING

**DO NOT add try-except blocks, error handling, or any exception catching to
your implementation.**

-   Try-except blocks hide compilation errors and break the validation loop
-   Let errors surface naturally so they can be caught and fixed properly
-   The validation system needs to see raw errors to diagnose issues
-   Even "helpful" error handling (logging errors, fallbacks, etc.) breaks
    validation

**If you add try-except blocks, the kernel will appear to compile successfully
but actually have hidden failures.**

### Optimization Plan

You must read and follow the optimization plan from this file:
**{kernel_plan_path}**

Phase 1 step 1 wrote it. If the file is missing, step 1 did not finish: go
back and plan again rather than implementing without a plan.

Use the `read` tool to read this plan file. It
contains the detailed strategy, tiling configuration, and implementation details
you must follow.

### Source Kernel

The source kernel to be optimized is located at: **{base_kernel_path}**

If this is not available, check the plan file for the source kernel path.

Use the `read` tool to read the source kernel file.

#### When the baseline is a port, not the user's own code

Read `state.primary.language` from `<work>/state.json`. When it is
`"pytorch"`, `{base_kernel_path}` is a JAX **port** of the user's original
source, not the original itself. It is still the contract you must match — the
harness compares against it and nothing else. Read
`state.primary.context_path` (`<work>/torch_context.md`) only when the plan
cites it or when the baseline
leaves a dtype, layout or edge case ambiguous; its Section 2 is the
authoritative specification and its numerics section fixes the accumulator
precision. Do not port source constructs across literally — the plan already
decided what the TPU shape of this kernel is.

#### `<work>/ref/` IS OFF LIMITS. Do not read it, ever.

That directory holds reference kernels the user supplied for the *planner* to
mine. They have already been reconciled against the primary and distilled into
ledger entries that were translated into TPU terms; the plan cites the ones it
adopted, and those citations are everything you are entitled to.

This is not bureaucratic. Opening a `.cu` file at implementation time is the
single likeliest way this whole feature produces a slower kernel: the natural
next step is to reproduce what you see, and what you see is warp shuffles,
`__syncthreads`, per-thread register blocking and bank-conflict padding —
machinery with no TPU counterpart, which becomes scalar `jnp` busywork when
transliterated. If you find yourself wanting to read `ref/`, the answer is that
the plan is underspecified. Say so in your report instead.

#### The starting point the plan selected

Section 2 of the plan carries an `Optimization Base Choice:` line naming it —
either `base` (start from `{base_kernel_path}`) or `iter<k>` (build on
`<work>/iter<k>/optimized.py`). Open the file it names and start from that one.
The planner reached that choice by a numerical rule against the previous
iteration's measured `speedup`, so it is not a suggestion to weigh again. If
the line is missing, start from `{base_kernel_path}` and say so in your report.

### Test Harness & Inputs

The optimized kernel MUST have the exact same function signature as the source
kernel, and its `computation` function MUST match the interface expected by
the shared test harness located at: **{test_file_path}**

This harness was generated once, before this loop started, and is reused
unchanged for every iteration. Use the `read` tool to read it. Its
`get_inputs()` function defines the exact tensor shapes, dtypes, and static
arguments (e.g. `block_size`) that will be passed into your `computation`
function every time it's tested or benchmarked. Align your implementation's
argument order, shapes, and any static (non-traced) parameters exactly with
what `get_inputs()` produces — a mismatch here fails every test for reasons
unrelated to the kernel's actual correctness.

### Your Task

Implement the optimized Pallas kernel by:

1.  **Reading** the approved optimization plan from the plan file
2.  **Reading** the source kernel code and the shared test harness to confirm
    the exact inputs/shapes/static args your `computation` function must
    accept
3.  **Implementing** the optimizations specified in the plan
4.  **Following** the exact specifications from the plan (block sizes, grid
    configuration, memory layout, etc.)
5.  **Writing it in the house style** of
    {{reference:tpu-kernel-development/pallas_code_style.md}}: name dimensions
    once, shape comments, extents derived from refs, and hardware constants
    read from `pltpu.get_tpu_info()` / `pltpu.get_sublane_tiling(dtype)` at
    run time, never hardcoded. A block size is a design choice, not a hardware
    constant: keep it the ALL_CAPS module-level literal the plan
    chose.
6.  **Auditing affordances** against the plan's Section 8: every affordance
    the plan marks `planned` (double-buffered critical-path transfers, async
    writes waited as late as possible, a narrower tail-tile path, hoisted
    per-call setup, a resident accumulator, `parallel` semantics on
    independent axes, scalar-prefetched indices) must be visible in your code.
    An affordance declared `planned` and missing from the code is a bug; see
    {{reference:tpu-kernel-development/kernel_design_method.md}} §6 and §10.

### Critical Requirements

-   Follow the plan EXACTLY - do not deviate from the approved strategy
-   If the plan specifies block sizes (e.g., bM=128, bK=128, bN=128), use those
    exact values
-   If the plan specifies a grid structure, implement it as described
-   If the plan mentions specific optimizations (pipelining, prefetching, etc.),
    include them
-   Preserve all initialization and setup code outside both the kernel and
    computation functions
-   **Always use a two-level structure:**
    1.  **`kernel` function** (with exact name "kernel") at module level -
        contains the core computation logic that operates on memory references
        (e.g., `x_ref`, `y_ref`, `z_ref`)
    2.  **`computation` function** (with exact name "computation") at module
        level - sets up parameters and invokes `pl.pallas_call` with the kernel
        function

### Documentation Requirements (CRITICAL)

Your implementation MUST include comprehensive inline documentation:

1.  **Shape Annotations**: Every significant variable must have a shape comment

    -   Function parameters: `def kernel(x_ref, y_ref, z_ref): # x_ref: (bM,
        bK), y_ref: (bK, bN), z_ref: (bM, bN)`
    -   Local variables: `block_data = x_ref[...] # Shape: (bM, bK)`
    -   Intermediate results: `partial_sum = jnp.sum(block_data, axis=1) #
        Shape: (bM,)`

2.  **Memory Space Annotations**: Document which memory hierarchy level
    variables occupy

    -   `# Memory: HBM` - Data in High Bandwidth Memory (main DRAM)
    -   `# Memory: VMEM` - Data in Vector Memory (on-chip SRAM)
    -   `# Memory: SMEM` - Data in Scalar Memory
    -   `# Memory: Registers` - Data in register file

3.  **Memory Transfer Comments**: Explain data movement between memory levels

    -   `# Transfer: HBM → VMEM` when loading blocks via BlockSpec
    -   `# Load: VMEM → Registers` when accessing x_ref[...] or similar
    -   `# Store: Registers → VMEM` when writing to output references
    -   `# Write back: VMEM → HBM` happens automatically at kernel completion

4.  **Computation Comments**: Explain the purpose of each major operation

    -   Why specific block dimensions were chosen
    -   How grid indices map to tensor coordinates
    -   Purpose of conditional logic (e.g., boundary handling)
    -   Accumulation patterns and their correctness

**Example of well-documented kernel:**

```python
def kernel(a_ref, b_ref, c_ref):
    '''Matrix multiplication kernel for blocks.

    Args:
        a_ref: Input A block  # Shape: (bM, bK), Memory: VMEM
        b_ref: Input B block  # Shape: (bK, bN), Memory: VMEM
        c_ref: Output C block  # Shape: (bM, bN), Memory: VMEM (accumulator)
    '''
    # Get block indices in the grid
    i = pl.program_id(0)  # M dimension block index
    j = pl.program_id(1)  # N dimension block index
    k = pl.program_id(2)  # K dimension iteration index

    # Initialize output block to zero on first K iteration
    # This is necessary because we accumulate across K dimension
    @pl.when(k == 0)
    def _init():
        c_ref[...] = jnp.zeros_like(c_ref)  # Shape: (bM, bN)

    # Load blocks from VMEM to registers
    a_block = a_ref[...]  # Shape: (bM, bK), Load: VMEM → Registers
    b_block = b_ref[...]  # Shape: (bK, bN), Load: VMEM → Registers

    # Compute matrix multiplication for this block
    # This uses the TPU MXU (Matrix Multiply Unit) for efficiency
    partial_result = a_block @ b_block  # Shape: (bM, bN), Compute in Registers

    # Accumulate result into output block
    c_ref[...] += partial_result  # Shape: (bM, bN), Store: Registers → VMEM
```

### Output Requirement

When you have implemented the optimized kernel:

1.  You **must** use the `write` tool to write the *entire* optimized
    script to a file. Save the plan to the exact path provided in
    `{optimized_kernel_path}`.
2.  Note the key optimizations applied, and the path written, in
    `<work>/tpu_kernel_debug_history.md` under this iteration's heading.

**IMPORTANT:** Once you have written the optimized kernel file, this step is
COMPLETE. Go on to Phase 2 and compile it.

### TPU-Specific Constraints (Mosaic Backend)

When targeting TPU (which is the default for these kernels), you must follow
these Mosaic lowering constraints:

-   **Matmul Accumulator Type**: Any `tpu.matmul` operation (or `@` operator
    lowered to it) requires a **32-bit accumulator**. Ensure you use appropriate
    types (e.g., `acc_dtype=jnp.float32` or similar) or ensure inputs are cast
    correctly if needed, although accumulation is usually 32-bit.
-   **Block Spec Divisibility**: The Pallas TPU lowering requires that the last
    two dimensions of your block shape are divisible by **8 and 128**
    respectively, or that they match the array dimensions exactly.
-   **Rank Constraint**: The Pallas TPU lowering supports only blocks of rank
    **>= 1**. Do not generate zero-dimensional blocks.
-   **More legality rules**: what may be a runtime value (DMA extents and
    offsets, lane vs sublane indices), when a DMA must be waited, and
    SparseCore memory-space rules are in
    {{reference:tpu-kernel-development/tpu_hardware_facts.md}}. For a SparseCore
    kernel, read {{reference:tpu-kernel-development/design/sparsecore.md}} first.

### Important Notes

-   If you encounter any ambiguity in the plan, use your best judgment to
    resolve it rather than making assumptions or asking the user.
-   If the plan seems to have issues or contradictions, attempt to resolve them
    or proceed with the most logical approach. Do not stop to ask the user.
-   DO NOT change code outside the `kernel` and `computation` functions unless
    the plan explicitly specifies to do so
-   The `kernel` function must be named exactly "kernel" (not "mlp_kernel",
    "matmul_kernel", etc.) and should be defined at module level
-   The `computation` function must be named exactly "computation" (not
    "mlp_computation", "matmul_computation", etc.) and should be defined at
    module level
-   Maintain the same variable names and overall structure as the source kernel

### Required Code Structure

Your implementation must follow this structure:

```python
# Imports
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# Initialization (if needed)
# ... constants, helper functions, etc ...

# Kernel function (module level)
def kernel(x_ref, y_ref, z_ref, ...):
    """Core kernel logic that operates on memory references."""
    # Kernel body here - memory operations, computations, etc.
    # Example: z_ref[...] = x_ref[...] @ y_ref[...]
    pass

# Computation function (module level)
def computation(A: jnp.ndarray, B: jnp.ndarray, ...) -> jnp.ndarray:
    """Sets up and invokes the Pallas kernel."""
    # Set up block sizes, grid configuration, etc.
    bM, bK, bN = 128, 128, 128

    # Call the kernel via pallas_call
    # IMPORTANT: Always include debug=True for better error diagnostics
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(...),
        grid=...,
        in_specs=[...],
        out_specs=...,
        debug=True,  # Always enable for better compilation error messages
    )(A, B, ...)

# Main function (REQUIRED - module level)
def main():
    """Demonstrates kernel usage with sample inputs.

    This function is REQUIRED to make the kernel file self-contained and testable.
    It should create sample inputs, call the computation function, and validate output. Do not include any correctness testing.

    The main function tests compilation in two stages:
    1. First runs without JIT to verify basic compilation
    2. Then runs with JIT to verify optimized compilation
    Both stages must **NOT** use try-catch blocks.
    """
    # Create sample input arrays with appropriate shapes
    # Example: x = jax.random.normal(jax.random.PRNGKey(0), (1024, 512))

    # Stage 1: Test without JIT (verifies basic compilation)
    # print("Testing without JIT...")
    # result = computation(x, y, ...)
    # print(f"Result shape: {result.shape}")
    # print(f"Result sample: {result[:5, :5]}")

    # Stage 2: Test with JIT (verifies optimized compilation)
    # print("\nTesting with JIT...")
    # jitted_computation = jax.jit(computation)
    # result_jit = jitted_computation(x, y, ...)
    # print(f"JIT result shape: {result_jit.shape}")
    # print(f"JIT result sample: {result_jit[:5, :5]}")
    pass

if __name__ == "__main__":
    main()
```

### Example Implementation Flow

1.  Read the plan file to understand the optimization strategy
2.  Read the source kernel to understand the current implementation
3.  Apply the optimizations from the plan:
    -   Define the `kernel` function at module level with the core computation
        logic
    -   Set up the specified block sizes in the `computation` function
    -   Create the `pallas_call` invocation with the planned memory operations
    -   Configure the grid_spec, in_specs, and out_specs as specified
    -   Add any special optimizations (zero initialization guards, accumulation,
        etc.) in the `kernel` function
    -   **CRITICAL:** Add a `main` function that:
        -   Creates sample inputs
        -   First tests the computation without JIT to verify basic compilation
        -   Then tests with JIT to verify optimized compilation
        -   Validates and prints outputs from both stages
4.  Write the complete optimized kernel to a new file following the required
    structure
5.  Go on to Phase 2 with the new file

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
Use `read` on the router and the pages and references named in this prompt EXTENSIVELY throughout implementation to retrieve Pallas/JAX/TPU documentation. Essential for:
- **Finding implementation examples for specific operations** (matmul, reductions, etc.)
- **Looking up memory reference operations** (`.load()`, `.store()`, `[...]`)
- **Understanding grid specifications and BlockSpec patterns with `index_map` lambdas**
- **Checking correct usage of TPU-specific features** (`pl.when`, memory barriers, etc.)
- **Looking up TPU architecture details** (memory hierarchy, MXU specs, vector units)
- **Debugging compilation or runtime issues with specific API calls**

**Retrieval Strategy:**
1. **Identify the kernel type first**: use the *Kernel categories → the routing table* section of the router (`kernel-optimization-index`) you opened in Step 1 to classify the kernel into exactly one of the seven categories -- `attention`, `gemm-conv-epilogue`, `streaming-reduction`, `grouped-ragged-indirection`, `state-carry-scan`, `dense-near-roofline`, `cross-chip-collective` -- then fetch the class page that row names with `python3 {{script:tpu-knowledge-base/kb_lookup.py}} <keyword>`, using the keyword the row gives, and `read` the path it prints. One primary page; at most one secondary for straddlers.
2. **Identify specific optimizations**: Check techniques like pipelining, double buffering (`emit_pipeline`), or Megacore splitting.
3. **Verify memory management**: Check VMEM capacity and tiling constraints against your target chip.

#### Step 4: Write the Kernel
Use the `write` tool to write your final, optimized kernel to `{optimized_kernel_path}`.

Your implementation summary and code header MUST attest:
`loaded: [<provenance keywords and reference files read>]`

1.  `read` and `write`: To read the plan file, read the source
    kernel, and to write your final, optimized kernel.

2.  `shell`: For looking up specific API definitions and signatures when
    you need precise technical details.

    *   Invoke via CLI:

        ```bash
        python3 {{script:tpu-kernel-development/search_api.py}} -- "<api_name>"
        ```

        where `<api_name>` is the API you want to search. Run this command
        every time these instructions say to "query".

IMPORTANT: You MUST consult the `kernel-optimization-index` page and load the required concept/hardware pages while implementing the kernel to ensure accuracy. Do not rely on pre-trained knowledge alone - always verify with current documentation.


### Final Checklist Before Writing the File

Before you call `write`, verify your implementation has:

-   ✅ **NO try-except blocks** - This is critical for validation
-   ✅ **NO error handling** - Let errors surface naturally
-   ✅ `kernel` function at module level
-   ✅ `computation` function at module level
-   ✅ Comprehensive shape and memory annotations
-   ✅ Follows exact specifications from the plan
-   ✅ Every affordance the plan marks `planned` is present in the code
-   ✅ No hardcoded hardware constants (lane width, sublane tiling, VMEM size)
-   ✅ Includes a `main()` function for testing

**REMEMBER: The validation loop depends on seeing raw compilation errors. Do not
hide them with try-except blocks!**
