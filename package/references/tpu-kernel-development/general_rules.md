# tpu-kernel-development General Rules

These are the invariants every tpu-kernel-development unit works under. Most of them are
already stated where they bite — in a phase of the kernel-worker, or in a
reference it follows — and this page collects them so that a step that does not
restate a rule is not read as permission to break it.

Where a phase or reference is more specific than a rule here, follow the phase:
it names the exact status to return, the exact file to write, the exact retry
budget. Nothing here licenses something a phase forbids.

`<work>` is this kernel's slot, `{{runs}}/<run_id>/artifacts/tpu-kernel-development/<slot>`,
exactly as the agent that sent you here defines it. Other slots under
`{{runs}}/<run_id>/artifacts/tpu-kernel-development` belong to other kernels: never read
or write them.

## Who follows which rule

| Rule | `tpu-kernel-development` | `kernel-worker` and every reference it follows | `analyze-cuda-reference` |
| :--- | :---: | :---: | :---: |
| 1. Preserve the caller's inputs | ✓ | ✓ | — |
| 2. Stop rather than guess, substitute or fabricate | ✓ | ✓ | ✓ |
| 3. A missing measurement is `null` | ✓ | ✓ | ✓ |

The kernel-worker follows its references in its own context, so a reference is
bound by everything the kernel-worker is bound by. `analyze-cuda-reference`
never builds inputs or sets tolerances, which is why rule 1 does not reach it.

## 1. Preserve the caller's inputs

-   **Problem shapes, dtypes and static arguments are the caller's.** The
    shapes in `get_inputs()` come from the golden manifest, from a
    `get_inputs()` the caller's source already defines, from the tensor table
    of a `specification` input, or from the context brief — in that order —
    and are copied, not chosen. Never shrink a dimension, a batch size or a
    sequence length to get past an OOM, a VMEM overflow or a timeout. Shrink
    the **tile**, never the problem.
-   **Autotuning searches implementation parameters only** — block sizes,
    grid order, buffering depth. A search space that varies a problem
    dimension is measuring a different computation.
-   **`atol`, `rtol`, `max_iterations` and `seed` are the values resolved in
    setup and recorded in `state.json`.** Never loosen a tolerance to make a
    gate or a test pass; a gate that cannot pass stops the run.
-   A kernel that cannot handle the caller's shapes is a failed iteration, not
    a reason to change the shapes.

## 2. Stop rather than guess, substitute or fabricate

-   **A missing input stops the step; it is never reconstructed.** A missing
    `run_id`, an unreadable `base_kernel_path`, a `<work>/state.json` that is
    not there when the step expects it, a `base.py` a phase owed but did not
    produce — stop and return a result whose `error_message` names the path.
    Use `invalid_input` when the caller did not supply something, and `failed`
    when a phase that owed the artifact did not produce it. Never substitute a
    file found nearby and never write the missing artifact yourself.
-   **Do not guess what the run is.** `input_language`, the
    entry point and the slot each file fills are settled by the request, by
    `classify_inputs.py`, or by the user — never by inference. The same holds
    for the hardware: generation, topology, HBM and the hardware page come from
    `state.tpu_spec`, recorded once in the kernel-worker's Phase 0 step 3 from
    `{{runtime}} compute inspect --json`. When the target cannot report its
    chip the run stops there; no step assumes a generation.
-   **Never fabricate a result.** Compilation status, correctness, latency and
    speedup come from the stdout of a `{{runtime}} compute run` and from
    nothing else. Read them off the output; do not infer them from a clean
    exit, from wall-clock time, or from how the code looks.

## 3. A missing measurement is `null`

-   When a value could not be obtained, write `null` and say why in the
    accompanying `.md` or in `<work>/tpu_kernel_debug_history.md`. In a result
    object, omit the measurement field instead. Downstream steps read `null`
    as "not measured" and read any number as measured — `state.history`'s
    `speedup` drives the next planner's build-or-restart choice, and a guessed
    one silently changes what gets optimized.
-   Never compute a speedup yourself; take the harness's `SPEEDUP` line. Never
    substitute a spec-sheet figure from a knowledge page for a measurement:
    a peak bandwidth is a ceiling, not what this kernel did.
-   **The one sentinel:** `state.best_optimized_time` starts as the string
    `"Infinity"`, meaning "no kernel that beat the baseline kept yet", as `tpu-kernel-development`'s initial state
    defines. Compare it as numeric infinity. Never write `"Infinity"` anywhere
    else, and never into a result.
