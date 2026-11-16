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
"""Template for the per-iteration profile run (never executed as-is).

`assemble_profile_run.py` copies this file byte-for-byte, substitutes the three
constants below, then appends the run's `get_inputs()` source and an exec-bound
`opt_computation` -- the same shape as `test_harness_template.py`, and for the
same reason: the parts an LLM authors stay small and the parts that must be
exact are copied by a tool.

Two things in here are load-bearing and easy to get wrong by hand:

  * `LIBTPU_INIT_ARGS` is set BEFORE `import jax`. libtpu reads it once, at
    initialization. Set it afterwards and the trace still captures, still looks
    complete, and simply has no LLO data in it -- which XProf reports as
    `status: UNAVAILABLE` / `reason: LLO_DATA_ABSENT`, i.e. a Pallas kernel
    shows up as one opaque block with no kernel-level detail. That is the whole
    reason tpu-kernel-development profiles at all.
  * Warm-up runs happen OUTSIDE `start_trace`. The first call compiles, so a
    trace that includes it attributes compilation time to the kernel.

`get_inputs`, `opt_computation` and `main` reference each other across the
append boundary, so `# noqa: F821` marks the names this file cannot see.
"""

import os

# Substituted by assemble_profile_run.py. Both paths are relative on purpose:
# `compute run` addresses staged files by a path relative to the script's
# working directory on the target, and refuses anything absolute or escaping
# it. TRACE_ARCHIVE is what comes back, because `--output` carries one FILE and
# a profile run is a `plugins/profile/<session>/` tree.
TRACE_DIR = "tensorboard"
TRACE_ARCHIVE = "trace.tar.gz"
NUM_WARMUPS = 3
NUM_RUNS = 3

# LLO debug info: without it a Pallas kernel is a single opaque op in the
# trace. Not --xla_xprof_enable_custom_call_tracing: it adds only runtime LLO
# trace detail, not the source map get_llo_analysis reads, and skews the
# kernel's own stats (openxla/xprof#3566, #3574). Appended rather than
# overwritten so an environment that already set flags keeps them.
_LLO_FLAGS = "--xla_xprof_register_llo_debug_info=true"
_existing = os.environ.get("LIBTPU_INIT_ARGS", "")
os.environ["LIBTPU_INIT_ARGS"] = " ".join(
    [_existing] + [f
                   for f in _LLO_FLAGS.split() if f not in _existing]).strip()

import sys  # noqa: E402
import tarfile  # noqa: E402
import traceback  # noqa: E402

import jax  # noqa: E402
# `get_inputs()` is appended into THIS module's namespace and builds its arrays
# from these two names without importing them itself -- same arrangement as
# `test_harness_template.py`. Unused here on purpose; do not drop it.
import jax.numpy as jnp  # noqa: E402,F401


def _profile_options():
    """The options the old generate-profile-script stage set, when available.

    `ProfileOptions` and `advanced_configuration` are not in every jax release
    this package runs against. A missing option costs detail in the trace; a
    hard failure here costs the whole profile, so an older jax degrades to the
    default trace instead.
    """
    if not hasattr(jax.profiler, "ProfileOptions"):
        return None
    options = jax.profiler.ProfileOptions()
    for name, value in (
        ("python_tracer_level", 0),
        ("host_tracer_level", 2),
        ("advanced_configuration", {
            "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"
        }),
    ):
        if hasattr(options, name):
            setattr(options, name, value)
    return options


def main():
    try:
        steps = []
        for dynamic_args, static_args in input_configs(
                get_inputs()):  # noqa: F821
            args = tuple(dynamic_args) + tuple(static_args)
            static_argnums = tuple(range(len(dynamic_args), len(args)))
            # The benchmark's compile path, not a fresh jax.jit: the trace has
            # to describe the executable `RESULT_TIME` was measured on.
            step = compile_and_align(  # noqa: F821
                opt_computation, args, static_argnums)  # noqa: F821
            # Warm up outside the trace: compile here, measure there.
            warm_up(step, NUM_WARMUPS)  # noqa: F821
            steps.append(step)

        options = _profile_options()
        if options is None:
            jax.profiler.start_trace(TRACE_DIR)
        else:
            jax.profiler.start_trace(TRACE_DIR, profiler_options=options)
        try:
            for step in steps:
                res = None
                for _ in range(NUM_RUNS):
                    res = step()
                if res is not None:
                    jax.block_until_ready(res)
        finally:
            jax.profiler.stop_trace()

        # The agent reads this line to learn where the trace landed, and the
        # count so an empty capture is caught here rather than downstream.
        traces = []
        for root, _, files in os.walk(TRACE_DIR):
            traces.extend(
                os.path.join(root, f) for f in files
                if f.endswith(".xplane.pb"))
        print(f"TRACE_DIR: {TRACE_DIR}")
        print(f"XPLANE_FILES: {len(traces)}")
        for path in sorted(traces):
            print(f"  {path}")
        if not traces:
            print("PROFILE_OK: False -- no .xplane.pb was written")
            return 1

        # Pack the run into one file so `compute run --output` can carry it:
        # staging moves a single file, and this is a directory tree. Archive
        # the tree itself, not its parent, so unpacking into the caller's
        # trace directory reproduces `plugins/profile/<session>/`.
        with tarfile.open(TRACE_ARCHIVE, "w:gz") as archive:
            for root, _, files in os.walk(TRACE_DIR):
                for name in files:
                    full = os.path.join(root, name)
                    archive.add(full, arcname=os.path.relpath(full, TRACE_DIR))
        print(f"TRACE_ARCHIVE: {TRACE_ARCHIVE}")
        print("PROFILE_OK: True")
        return 0
    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
        print("PROFILE_OK: False")
        return 1
