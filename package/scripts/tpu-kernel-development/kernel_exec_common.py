# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The compile path shared by the benchmark and the profile run.

Both `test_harness_template.py` (which times a kernel) and
`profile_harness_template.py` (which traces it) have to reach the accelerator
with **the same compilation**. If they do not, the time the test harness
measured (`RESULT_TIME`) describes one executable and the trace describes
another: the trace can contain layout copies, or a differently tiled Pallas
call, that the timed run never executed
-- and the planning step (`plan-kernel`) then optimises against a bottleneck the benchmark never
saw. Nothing about that failure is visible in either output.

Keeping the compile in one place is what makes that impossible by
construction, so neither template should JIT a kernel itself.

This file is never imported. Both scripts run on the compute target as a
single file (`compute run --code`), so the assemblers concatenate this source
into what they emit -- the same mechanism `assemble_test_harness.py` already
uses for the template itself. `jax` is supplied by the enclosing script.
"""

# `jax` comes from the assembled file's namespace, not from here -- see the
# module docstring. Suppressed rather than left as IDE noise.
# pyright: reportUndefinedVariable=false

import math
import re
import warnings


def input_configs(raw_inputs):
    """Normalize `get_inputs()` into a list of (dynamic_args, static_args).

    One `get_inputs()` serves correctness, benchmarking and profiling, so the
    shape it is allowed to return is defined once, here.
    """
    if isinstance(raw_inputs, list):
        configs = raw_inputs
    elif isinstance(raw_inputs, tuple) and len(raw_inputs) == 2:
        configs = [raw_inputs]
    else:
        raise ValueError("get_inputs() must return a list of tuples or a "
                         "single (dynamic_args, static_args) tuple.")
    for config in configs:
        if not isinstance(config, tuple) or len(config) != 2:
            raise ValueError(
                "Each input config must be exactly "
                f"(dynamic_args, static_args). Got: {type(config)}")
    return configs


def _alias_map(compiled):
    """`{param_index: output_leaf_index}` for buffers the executable reuses.

    Read from the `input_output_alias` attribute of the compiled HLO module,
    which is XLA's final word on which donated inputs it actually wrote the
    output into. An empty map -- including when the text is unavailable or
    in a form this does not parse -- means "no donation", which is always
    safe: it costs the copy, never correctness.
    """
    try:
        header = compiled.as_text().split("\n", 1)[0]
    except Exception:  # pylint: disable=broad-except
        return {}
    match = re.search(r"input_output_alias=\{(.*?)\}(?:,\s*[a-z_]+=|$)",
                      header)
    if not match:
        return {}
    aliases = {}
    for out_idx, param, param_idx in re.findall(
            r"\{([\d,\s]*)\}:\s*\((\d+),\s*\{([\d,\s]*)\}", match.group(1)):
        out_idx = out_idx.strip()
        if param_idx.strip() or "," in out_idx:
            # Tupled parameters or nested outputs: jit flattens both, so
            # neither is expected; do not guess at a mapping.
            return {}
        aliases[int(param)] = int(out_idx) if out_idx else 0
    return aliases


def _fresh(*xs):
    """Identity that is guaranteed to return new buffers.

    What makes donating them safe: a jit whose output is literally its input
    may be forwarded by jax without running, handing back the caller's own
    array -- which the first donated call would then free. The barrier is an
    op, so its outputs are new values and XLA must materialize them.
    """
    return jax.lax.optimization_barrier(xs)


def _copied_params(compiled):
    """Parameter indices the compiled program copies aside, unchanged.

    What donation is for is removing such a copy, but it can add one: when a
    kernel still reads an input while its output is written into that input's
    buffer, XLA first copies the input, as is, somewhere else. That is the
    pattern counted here -- a top-level copy of the parameter itself with the
    parameter's exact type. A copy that reshapes or re-lays out an input is
    the kernel's own work, present with or without donation, and is not
    counted. An unreadable program reports none.
    """
    try:
        text = compiled.as_text()
        entry = text[text.index("\nENTRY"):]
        entry = entry[:entry.index("\n}")]
    except Exception:  # pylint: disable=broad-except
        return set()
    params = {
        name: (int(index), type_)
        for name, type_, index in re.findall(
            r"%([\w.\-]+) = (\S+) parameter\((\d+)\)", entry)
    }
    copied = set()
    for type_, src in re.findall(r"= (\S+) copy(?:-start)?\(%([\w.\-]+)\)",
                                 entry):
        if src in params and params[src][1] == type_:
            copied.add(params[src][0])
    return copied


def _alias_candidates(fn, leaves, arrays):
    """The array inputs XLA could write an output into, and safely may.

    Safely: once an input is donated, XLA may write *any* output of its size
    into it -- not only the update the kernel meant -- and `step` feeds that
    output back in as the input. Float data survives that as drifting values;
    integer data does not: lengths, page tables and indices would come back as
    some other output's values, and the next timed call would read past its
    buffers. So only floating-point inputs are offered. KV caches are floating
    point; an integer-quantized cache keeps the copy, as it did before
    donation.

    Of those, only the ones the size of an output: no other buffer can be
    reused, so offering it costs a compile and finds nothing. Matched on bytes
    rather than shape so a kernel that returns its cache reshaped still
    qualifies. `eval_shape` traces without compiling; if it cannot run, every
    floating-point array is offered.
    """
    floats = [
        i for i in arrays
        if jax.numpy.issubdtype(leaves[i].dtype, jax.numpy.floating)
    ]
    try:
        sizes = {
            math.prod(out.shape) * out.dtype.itemsize
            for out in jax.tree_util.tree_leaves(jax.eval_shape(fn, *leaves))
        }
    except Exception:  # pylint: disable=broad-except
        return floats
    return [
        i for i in floats
        if math.prod(leaves[i].shape) * leaves[i].dtype.itemsize in sizes
    ]


def compile_and_align(func, args, static_argnums):
    """Compile `func` the way it runs in production and return a timed step.

    Returns `step`, a zero-argument callable that runs the executable once and
    returns its result. Two things make one call of `step` measure the kernel
    and nothing else:

    - **Layouts.** The executable is compiled with AUTO layouts and its inputs
      are moved into the layouts it wants, so XLA inserts no layout copies.
      Falls back to a plain JIT when the layout API is unavailable, so an
      older jax still runs -- consistently for both callers, which is the
      point.
    - **Donation.** A kernel that updates an input in place (a Pallas
      `input_output_aliases`, or `.at[].set` on a cache) can only do so when
      the caller donates that input; otherwise XLA copies the whole buffer
      first, on every call, and that copy is what gets timed -- for a KV-cache
      update it can be most of the measurement. Every floating-point input the
      size of an output is offered for donation (`_alias_candidates` says why
      only those; a kernel with none compiles once, undonated), any offer XLA
      answers with a copy is withdrawn (`_copied_params`), and `_alias_map`
      reads back which ones XLA actually aliased. `step` then feeds each aliased output back in as
      the next call's input, the way a serving loop threads its cache, so a
      donated buffer is never reused after it was consumed. Values in those
      buffers therefore evolve across calls; correctness is checked on a
      separate, undonated run.

    The inputs `step` donates are private copies, so the caller's `args` stay
    valid for the next `compile_and_align`.
    """
    dyn_positions = [i for i in range(len(args)) if i not in static_argnums]
    leaves, treedef = jax.tree_util.tree_flatten(
        tuple(args[i] for i in dyn_positions))
    # A host array (numpy) cannot be donated, so a `get_inputs()` that returns
    # one would keep the copy this function exists to remove. Upload it once,
    # here, rather than on every timed call.
    array_type = getattr(jax, "Array", ())
    leaves = [
        jax.device_put(leaf) if hasattr(leaf, "__array__")
        and not isinstance(leaf, array_type) else leaf for leaf in leaves
    ]

    # Takes the flattened dynamic leaves so an HLO parameter index is a leaf
    # index; static args are closed over rather than passed.
    def wrapped(*flat):
        all_args = list(args)
        for i, arg in zip(dyn_positions,
                          jax.tree_util.tree_unflatten(treedef, flat)):
            all_args[i] = arg
        return func(*all_args)

    try:
        from jax.experimental import layout as jax_layout

        auto = jax_layout.Format(jax_layout.Layout.AUTO)
        layout_kwargs = {"in_shardings": auto, "out_shardings": auto}
    except Exception:  # pylint: disable=broad-except
        layout_kwargs = {}

    def compile_with(donate):
        # keep_unused: a pruned argument would shift every parameter index
        # after it, and `_alias_map` indexes by parameter.
        return jax.jit(wrapped,
                       donate_argnums=tuple(donate),
                       keep_unused=True,
                       **layout_kwargs).lower(*leaves).compile()

    arrays = [
        i for i, leaf in enumerate(leaves) if isinstance(leaf, array_type)
    ]
    candidates = _alias_candidates(wrapped, leaves, arrays)

    def build(donate):
        with warnings.catch_warnings():
            # jax warns about each offered input XLA did not alias. Expected
            # here: offering is how we find out which ones it will.
            warnings.filterwarnings(
                "ignore", message="Some donated buffers were not usable")
            compiled = compile_with(donate)
        aliases = _alias_map(compiled) if donate else {}

        state = list(leaves)
        if layout_kwargs and hasattr(compiled, "input_formats"):
            arg_formats, _ = compiled.input_formats

            state = list(
                jax.jit(
                    _fresh,
                    out_shardings=arg_formats).lower(*state).compile()(*state))
        elif donate:
            fresh = jax.jit(_fresh)(*(state[i] for i in donate))
            for i, buf in zip(donate, fresh):
                state[i] = buf
        return compiled, aliases, state

    try:
        compiled_func, aliases, state = build(candidates)
    except Exception:  # pylint: disable=broad-except
        if not layout_kwargs:
            raise
        layout_kwargs.clear()
        compiled_func, aliases, state = build(candidates)

    # A donation XLA answered with a copy of that input (`_copied_params`)
    # adds the cost this function exists to remove. Drop those inputs and
    # compile again; if what is left still copies, donate nothing -- the
    # executable then matches an undonated compile exactly.
    donate = candidates
    copied = _copied_params(compiled_func) & set(donate)
    if copied:
        donate = [i for i in donate if i not in copied]
        compiled_func, aliases, state = build(donate)
        if _copied_params(compiled_func) & set(donate):
            donate = []
            compiled_func, aliases, state = build(())

    def step():
        res = compiled_func(*state)
        if aliases:
            out_leaves = jax.tree_util.tree_leaves(res)
            for param, out_idx in aliases.items():
                state[param] = out_leaves[out_idx]
        return res

    # An offered input XLA did not alias was not written, so it can be passed
    # again -- unless this jax consumes every donated buffer regardless, which
    # one call shows. Only then pay a second compile, donating just the
    # aliased inputs; if even that one aliases differently, pay the copy.
    unaliased = [i for i in donate if i not in aliases]
    if unaliased:
        jax.block_until_ready(step())
        if any(state[i].is_deleted() for i in unaliased):
            donate = sorted(aliases)
            compiled_func, aliases, state = build(donate)
            if sorted(aliases) != donate:
                compiled_func, aliases, state = build(())

    return step


def warm_up(step, num_warmups):
    """Run `step` `num_warmups` times and block.

    Always call this before timing or tracing: the first call compiles, so a
    measurement that includes it reports compilation as kernel time. `res` is
    bound before the loop so `num_warmups == 0` is a no-op rather than a
    NameError.
    """
    res = None
    for _ in range(num_warmups):
        res = step()
    if res is not None:
        jax.block_until_ready(res)
