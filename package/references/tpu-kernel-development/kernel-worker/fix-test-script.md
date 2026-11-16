---
scripts:
- scripts/tpu-kernel-development/assemble_test_harness.py
- scripts/tpu-kernel-development/test_harness_template.py
- scripts/tpu-kernel-development/kernel_exec_common.py
---

# Fix the test script

The kernel-worker reads this in Phase 0.10 step 5 when harness validation
failed and retries remain. You check the validation results and fix errors in
`{get_inputs_path}` yourself, in this context. This is the only file you can
touch: the rest of the shared harness at `{test_file_path}` is assembled deterministically by
`{{script:tpu-kernel-development/assemble_test_harness.py}}` from `{get_inputs_path}` plus
`{base_kernel_path}` plus the fixed `{{script:tpu-kernel-development/test_harness_template.py}}`, so it
cannot itself contain a bug that a fix here would address -- if validation
keeps failing after `{get_inputs_path}` looks correct, the problem is likely a
mismatch between what `get_inputs()` returns and what `{base_kernel_path}`'s
`computation` actually expects.

**TPU VM Execution Requirement**: Mock validation runs the *assembled*
harness (produced fresh each retry by `assemble_test_harness.py` from your
latest `{get_inputs_path}`) on the TPU VM -- not CPU. Neither `base.py` nor
the fixed harness template ever set `interpret=True` on any `pl.pallas_call`,
so running the assembled harness on a CPU-only backend fails outright with
`Only interpret mode is supported on CPU backend`, regardless of whether
`get_inputs()` and the base kernel are otherwise correct. Don't burn a retry
"fixing" `{get_inputs_path}` in response to that error -- it means the
harness was run on the wrong backend, not that your file is wrong.

-   Mock validation is an interactive test loop that needs stdout, stderr and the exit code back immediately, so run the assembled harness synchronously:
    `{{runtime}} compute run --code {test_file_path}`
    Pass `--interpreter` (or ensure `python_interpreter` is configured in the compute profile in `environment.json`) pointing to the Python environment where `jax[tpu]` and Pallas are installed (e.g. `~/tpu-kernel-development_venv/bin/python`).

## Context

`get_inputs()` file: `{get_inputs_path}`
Assembled harness (read-only, do not edit): `{test_file_path}`

**Validation Results** (from Phase 0.10 step 5):

-   Current validation error: `{validation_error}` -- which check failed
    (syntax, import, or mock execution) and its output
-   Earlier errors and fixes: `{validation_history}`

## First: Check if the File Exists

**If `{get_inputs_path}` does not exist:**

-   There is nothing to fix: Phase 0.10 step 3 did not produce it. Go back to
    step 3 and generate it first.
-   **STOP HERE**

## Second: Check for System/Connection Errors

**If the validation error contains the string `FATAL_CONNECTION_ERROR`:**

-   This is an unsolvable infrastructure error (e.g., SSH tunnel down, TPU
    unresponsive).
-   **DO NOT** attempt to write any code fixes.
-   **Immediately halt**: record `ESCALATE_SYSTEM_ERROR: <details of the
    error>` and follow step 3 of the kernel-worker's Strict Debuggability
    protocol, which stops the workflow.
-   **STOP HERE**.

## Third: Check if Fixes are Needed

1.  If there is no current validation error -- syntax, import and mock
    execution all passed:

    -   **All validations passed! No fixes needed.**
    -   Go back to Phase 0.10 step 5, which breaks the loop.
    -   **STOP HERE - do not modify the file**

2.  If any check failed → proceed to "Your Task" below.

## Tool Usage

1.  `read`: To read `{get_inputs_path}`, `{base_kernel_path}`, and
    `{test_file_path}` (for context on the error only -- never write to the
    latter).
2.  `write`: To overwrite `{get_inputs_path}` with the corrected
    version.

## Your Task (Only if Fixes are Needed)

### Step 1: Read the Current File and the Error Context

Use `read` on `{get_inputs_path}`. If the error trace references the
assembled harness (`{test_file_path}`), read it too, but only to understand
*where* `get_inputs()`'s output diverges from what `base_computation` expects
-- not to edit it.

### Step 2: Identify and Fix the Error

-   **Syntax Errors**: fix Python syntax in `get_inputs()`.
-   **Import Errors**: fix/add imports `get_inputs()` needs.
-   **Mock Execution Errors**: usually a shape/arity mismatch between what
    `get_inputs()` returns and what `base_kernel_path`'s `computation`
    expects — e.g. wrong number of `dynamic_args`/`static_args`, or a
    `(dynamic_args, static_args)` tuple malformed (must be exactly 2
    elements).

### Step 3: Write the Fixed File

Use `write` to overwrite `{get_inputs_path}` with the corrected
version.

**CRITICAL RULES:**

1.  **Only ever write to `{get_inputs_path}`.** Never write to
    `{test_file_path}` — Phase 0.10 step 4 regenerates it deterministically
    from this file after you're done.
2.  **DO NOT invent a new optimized kernel or `opt_computation` stub.**
3.  Keep the required return shape: a list of `(dynamic_args, static_args)`
    tuples.

**After writing:**

-   Append what you fixed to `{validation_history}`
-   Go back to Phase 0.10 step 4 to re-assemble the harness and validate
    again

## Important Notes

-   We are NOT fixing kernel bugs — only `get_inputs()`.
-   After your fix, Phase 0.10 re-runs `assemble_test_harness.py` and
    validation runs again.
-   Unlike per-iteration kernel work, exhausting retries here is a
    **run-blocking failure**: nothing downstream (planning, implementation,
    testing, profiling) can proceed without a valid harness. Say so plainly
    if you reach max retries.
