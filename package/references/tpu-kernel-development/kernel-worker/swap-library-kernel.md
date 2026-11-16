# Use the Tokamax kernel

Read this in Phase 1 when `state.library_check.result` is `"use"`. It replaces
planning and implementing.

The user chose an existing Tokamax kernel, named in
`state.library_check.module` (for example `tokamax.ragged_dot`). Instead of
writing a kernel, you write a small file that calls it. The rest of the
iteration then tests and measures that file like any other kernel.

## Step 1: Read the op's source

Look at the Tokamax source installed on the TPU before writing anything. Print
it with `{{runtime}} compute run`, for example:

```python
import inspect
import tokamax._src.ops.ragged_dot.api as api
print(inspect.getsource(api))
```

Some ops have no `api.py` (for example `gmm_v2` and `fused_moe`). For those,
also print the op's test file, which shows how the kernel is called.

If the import fails, see **If Tokamax or the op is missing** below.

## Step 2: Write the kernel file

Write `{optimized_kernel_path}` with a module-level `computation` function. It
takes the inputs that `{test_file_path}`'s `get_inputs()` produces, returns
the same output as `{base_kernel_path}`, and calls Tokamax:

```python
import jax.numpy as jnp
import tokamax


def computation(lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray) -> jnp.ndarray:
    return tokamax.ragged_dot(lhs, rhs, group_sizes, implementation="mosaic")
```

Guidelines:

-   Use the top-level function (`tokamax.ragged_dot`, `tokamax.layer_norm`,
    and so on) when there is one. Experimental ops don't have one, so import
    them from `tokamax._src` and note that in the plan.
-   Pass the TPU `implementation=` value, as in the example, so the call runs
    the TPU kernel and not plain XLA. If `state.library_check.module` names a
    value, use that one.
-   Match `base.py`'s call exactly, including the output dtype
    (`preferred_element_type`). Only reshape or cast around the call as much
    as `base.py`'s inputs and output require.
-   Don't add a `kernel` function or error handling.

Then write `{kernel_plan_path}` with three lines:

1.  The call you made.
2.  `Optimization Base Choice: base`
3.  A note that the kernel comes from Tokamax.

## Step 3: Compile and test (Phases 2 and 3)

Run Phases 2 and 3 as usual. The test checks the Tokamax kernel against
`base.py` within `atol`/`rtol` and measures its speedup. The kernel runs with
the configuration Tokamax chooses for these shapes; this loop does not tune
it. Phase 5 records Phase 3's numbers.

Phase 4 then captures a trace of the kernel.

## If Tokamax or the op is missing

If the import fails with `No module named 'tokamax'`, or the installed version
doesn't have this op, editing the file won't help. Skip the compile fixer, go
to Phase 5 with `compile_ok` set to false, and return `failed` with
`failure_reason: "compilation_error"`.

In `error_message`, say what is missing and how to fix it. Tokamax is not part
of the package's TPU requirements. Give the user this command for the TPU
environment; it keeps the `xprof` that tpu-diagnostics uses at the package's
floor:

```bash
pip install tokamax==0.0.15 'xprof>=2.24.0'
```

Leave the install to the user.
