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
"""Locks the two invariants the assembled profile script exists for.

Both fail silently in production -- the run succeeds, the trace is written, and
the numbers are wrong or the kernel detail is missing -- so they are asserted
here rather than left to review.
"""

import ast
import os
import runpy
import sys
import tarfile
import tempfile
import textwrap
import unittest

scripts_dir = os.path.dirname(__file__)
sys.path.insert(0, scripts_dir)

import assemble_profile_run  # noqa: E402

OPTIMIZED_KERNEL = textwrap.dedent("""
    import jax
    import jax.numpy as jnp


    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0


    def computation(x, scale):
        return x * scale
    """)

GET_INPUTS = textwrap.dedent("""
    def get_inputs():
        x = jnp.ones((8, 8), dtype=jnp.float32)
        return ([x], [2.0])
    """)

# A jax stand-in that records the order it was driven in. Enough surface for
# the harness, nothing more -- the point is call ordering, not numerics.
STUB_JAX = textwrap.dedent('''
    import os
    CALLS = []


    class _Profiler:

        class ProfileOptions:

            def __init__(self):
                self.python_tracer_level = None
                self.host_tracer_level = None
                self.advanced_configuration = None

        @staticmethod
        def start_trace(logdir, profiler_options=None):
            CALLS.append(("start_trace", profiler_options))
            session = os.path.join(logdir, "plugins", "profile", "s1")
            os.makedirs(session, exist_ok=True)
            with open(os.path.join(session, "host0.xplane.pb"), "w") as handle:
                handle.write("trace")

        @staticmethod
        def stop_trace():
            CALLS.append(("stop_trace", None))


    profiler = _Profiler()


    class _Compiled:

        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args):
            CALLS.append(("call", None))
            return self.fn(*args)


    class _Lowered:

        def __init__(self, fn):
            self.fn = fn

        def compile(self):
            return _Compiled(self.fn)


    class _Jitted:

        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args):
            CALLS.append(("call", None))
            return self.fn(*args)

        def lower(self, *args):
            return _Lowered(self.fn)


    # jax.experimental.layout is deliberately absent, so compile_and_align
    # takes its documented fallback rather than the AUTO-layout path.
    def jit(fn=None, **kwargs):
        if fn is None:
            return lambda f: _Jitted(f)
        return _Jitted(fn)


    def block_until_ready(value):
        CALLS.append(("block", None))
        return value


    class tree_util:

        @staticmethod
        def tree_flatten(tree):
            return list(tree), len(tree)

        @staticmethod
        def tree_unflatten(treedef, leaves):
            return tuple(leaves)

        @staticmethod
        def tree_leaves(tree):
            return list(tree) if isinstance(tree, (list, tuple)) else [tree]
    ''')

STUB_JNP = textwrap.dedent('''
    class _Arr:

        def __init__(self, shape):
            self.shape = shape

        def __mul__(self, other):
            return self

        __rmul__ = __mul__


    def ones(shape, dtype=None):
        return _Arr(shape)


    float32 = "float32"
    ''')


class TestAssembleProfileRun(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name

        self.kernel_path = os.path.join(root, "optimized.py")
        self.inputs_path = os.path.join(root, "get_inputs.py")
        self.output_path = os.path.join(root, "profile_kernel.py")
        # TRACE_DIR/TRACE_ARCHIVE are relative, as `compute run` requires,
        # so the script must be executed from a known working directory.
        self.cwd = os.path.join(root, "rundir")
        os.makedirs(self.cwd)
        with open(self.kernel_path, "w") as handle:
            handle.write(OPTIMIZED_KERNEL)
        with open(self.inputs_path, "w") as handle:
            handle.write(GET_INPUTS)

        stub_root = os.path.join(root, "stub")
        os.makedirs(os.path.join(stub_root, "jax"))
        with open(os.path.join(stub_root, "jax", "__init__.py"),
                  "w") as handle:
            handle.write(STUB_JAX)
        with open(os.path.join(stub_root, "jax", "numpy.py"), "w") as handle:
            handle.write(STUB_JNP)
        self.stub_root = stub_root

    def _assemble(self, **kwargs):
        assemble_profile_run.assemble(self.kernel_path, self.inputs_path,
                                      self.output_path, **kwargs)
        with open(self.output_path) as handle:
            return handle.read()

    def _run_assembled(self):
        """Execute the assembled script against the stub and return jax.CALLS."""
        sys.path.insert(0, self.stub_root)
        origin = os.getcwd()
        os.chdir(self.cwd)
        saved_jax = {
            name: module
            for name, module in sys.modules.items()
            if name == "jax" or name.startswith("jax.")
        }
        for name in saved_jax:
            del sys.modules[name]
        saved_env = os.environ.get("LIBTPU_INIT_ARGS")
        os.environ.pop("LIBTPU_INIT_ARGS", None)
        try:
            namespace = runpy.run_path(self.output_path, run_name="not_main")
            exit_code = namespace["main"]()
            import jax  # the stub, per sys.path above
            return exit_code, list(jax.CALLS), os.environ["LIBTPU_INIT_ARGS"]
        finally:
            os.chdir(origin)
            sys.path.remove(self.stub_root)
            for name in [
                    n for n in sys.modules
                    if n == "jax" or n.startswith("jax.")
            ]:
                del sys.modules[name]
            sys.modules.update(saved_jax)
            if saved_env is None:
                os.environ.pop("LIBTPU_INIT_ARGS", None)
            else:
                os.environ["LIBTPU_INIT_ARGS"] = saved_env

    def test_assembled_script_is_valid_python(self):
        ast.parse(self._assemble())

    def test_profile_uses_the_benchmarks_compile_path(self):
        """The trace must describe the executable `RESULT_TIME` was measured on.

        A fresh `jax.jit(opt_computation, ...)` here compiles with default
        layouts, while the benchmark compiles with AUTO layouts and re-lays out
        its inputs -- so the trace could hold layout copies, or a differently
        tiled Pallas call, that the timed run never executed. Both paths go
        through `compile_and_align` so that cannot happen.
        """
        source = self._assemble()
        self.assertIn("def compile_and_align", source,
                      "kernel_exec_common.py was not embedded")
        self.assertIn("compile_and_align(", source)

        # No direct jit of the kernel anywhere in the emitted script.
        tree = ast.parse(source)
        direct_jits = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(
                node.func, ast.Attribute) and node.func.attr == "jit" and any(
                    isinstance(a, ast.Name) and a.id == "opt_computation"
                    for a in node.args)
        ]
        self.assertEqual(
            direct_jits, [],
            "profile run compiles the kernel itself instead of "
            "going through the benchmark's compile path")

    def test_llo_flags_are_set_before_jax_is_imported(self):
        """libtpu reads LIBTPU_INIT_ARGS once, at init.

        Set it after `import jax` and the capture still succeeds and simply
        carries no LLO data, which XProf reports as LLO_DATA_ABSENT -- a Pallas
        kernel becomes one opaque block. Ordering is the whole fix.
        """
        source = self._assemble()
        tree = ast.parse(source)
        env_write_line = min(
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Subscript) and isinstance(
                node.value, ast.Attribute) and node.value.attr == "environ")
        jax_import_line = min(
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Import) and any(alias.name == "jax"
                                                    for alias in node.names))
        self.assertLess(env_write_line, jax_import_line)

    def test_warmups_run_outside_the_trace(self):
        """A trace that includes the first call attributes compilation to the kernel."""
        self._assemble(warmups=3, runs=2)
        exit_code, calls, libtpu_args = self._run_assembled()

        self.assertEqual(exit_code, 0)
        names = [name for name, _ in calls]
        start = names.index("start_trace")
        self.assertEqual(names[:start].count("call"), 3)
        self.assertEqual(names[start:].count("call"), 2)
        self.assertEqual(names[-1], "stop_trace")

        self.assertIn("--xla_xprof_register_llo_debug_info=true", libtpu_args)
        # It skews the kernel's own stats (openxla/xprof#3566).
        self.assertNotIn("xla_xprof_enable_custom_call_tracing", libtpu_args)

        options = dict(calls)["start_trace"]
        self.assertEqual(options.host_tracer_level, 2)
        self.assertEqual(options.advanced_configuration,
                         {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"})

    def test_trace_is_packed_into_one_returnable_archive(self):
        """`compute run --output` carries a file; a profile run is a tree.

        The archive holds the tree's contents rather than the tree itself, so
        unpacking into the caller's trace directory reproduces
        `plugins/profile/<session>/` rather than nesting it one deeper.
        """
        self._assemble()
        exit_code, _, _ = self._run_assembled()
        self.assertEqual(exit_code, 0)

        archive_path = os.path.join(self.cwd, "trace.tar.gz")
        self.assertTrue(os.path.isfile(archive_path),
                        "no archive for --output to carry back")
        with tarfile.open(archive_path) as archive:
            names = archive.getnames()
        self.assertTrue(any(n.endswith(".xplane.pb") for n in names), names)
        self.assertTrue(all(n.startswith("plugins/profile/") for n in names),
                        names)

    def test_empty_capture_is_reported_as_failure(self):
        """A clean run that wrote no trace is a failed capture, not a caveat."""
        self._assemble()
        with open(os.path.join(self.stub_root, "jax", "__init__.py"),
                  "a") as handle:
            handle.write(
                textwrap.dedent("""

                    def _no_op_start(logdir, profiler_options=None):
                        CALLS.append(("start_trace", profiler_options))


                    profiler.start_trace = _no_op_start
                    """))
        exit_code, _, _ = self._run_assembled()
        self.assertEqual(exit_code, 1)

    def test_rejects_a_profile_with_no_warm_up(self):
        """Zero warm-ups traces the compilation; fail loudly at assembly time."""
        with self.assertRaises(ValueError) as ctx:
            self._assemble(warmups=0)
        self.assertIn("at least 1", str(ctx.exception))

    def test_rejects_sources_missing_their_entry_points(self):
        with open(self.kernel_path, "w") as handle:
            handle.write("def not_computation():\n    pass\n")
        with self.assertRaises(ValueError) as ctx:
            self._assemble()
        self.assertIn("computation", str(ctx.exception))

        with open(self.kernel_path, "w") as handle:
            handle.write(OPTIMIZED_KERNEL)
        with open(self.inputs_path, "w") as handle:
            handle.write("def not_get_inputs():\n    pass\n")
        with self.assertRaises(ValueError) as ctx:
            self._assemble()
        self.assertIn("get_inputs", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
