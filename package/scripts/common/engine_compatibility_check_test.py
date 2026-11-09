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
"""Tests for the TP/DP plan; they need no vLLM or TPU."""

import contextlib
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import unittest.mock

import engine_compatibility_check as check


def _info(files=(), parameters=None):
    return types.SimpleNamespace(
        siblings=[
            types.SimpleNamespace(rfilename=name, size=size)
            for name, size in files
        ],
        safetensors=types.SimpleNamespace(parameters=parameters))


class TestWeightsBytes(unittest.TestCase):

    def test_parameters_are_scaled_by_dtype(self):
        # safetensors.total would say 10; bf16 weights are 2 bytes each.
        info = _info(parameters={"BF16": 8, "F32": 2})
        self.assertEqual(check.weights_bytes(info), 24)

    def test_duplicate_checkpoints_are_not_double_counted(self):
        """Mistral-7B-Instruct-v0.3 ships a consolidated and a split copy."""
        gib = 1024**3
        info = _info(files=[("consolidated.safetensors", 13.5 * gib),
                            ("model-00001-of-00003.safetensors", 4.61 * gib),
                            ("model-00002-of-00003.safetensors", 4.66 * gib),
                            ("model-00003-of-00003.safetensors", 4.23 * gib)],
                     parameters={"BF16": 7_248_023_552})
        self.assertAlmostEqual(check.weights_bytes(info) / gib, 13.5, 1)
        # With an unknown dtype, the index's total wins over the file sum.
        info.safetensors.parameters = {"Q4": 1}
        self.assertEqual(check.weights_bytes(info, index_total=7), 7)

    def test_file_sizes_are_the_last_resort(self):
        info = _info(files=[("a.safetensors", 3), ("b.safetensors",
                                                   4), ("config.json", 100)])
        self.assertEqual(check.weights_bytes(info), 7)

    def test_unknown_dtype_is_unknown_size(self):
        self.assertIsNone(check.weights_bytes(_info(parameters={"Q4": 8})))
        self.assertIsNone(check.weights_bytes(_info()))

    def test_local_directory(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(check.local_weights_bytes(d))
            for name, size in (("consolidated.safetensors",
                                10), ("model-00001-of-00002.safetensors", 5),
                               ("model-00002-of-00002.safetensors", 5)):
                with open(os.path.join(d, name), "wb") as f:
                    f.write(b"0" * size)
            self.assertEqual(check.local_weights_bytes(d), 20)
            with open(os.path.join(d, "model.safetensors.index.json"),
                      "w",
                      encoding="utf-8") as f:
                json.dump({"metadata": {"total_size": 10}}, f)
            self.assertEqual(check.local_weights_bytes(d), 10)


def _layouts(counts):
    """A torch_tpu topology lookup that knows layouts for `counts` devices."""

    def lookup(n):
        if n not in counts:
            raise RuntimeError(f"No TPU topology found for count: {n}")
        return f"layout-{n}"

    return lookup


class TestPlanParallelism(unittest.TestCase):

    def test_recommendations(self):
        """Largest fitting TP; replicas fill devices the heads leave over."""
        cases = [
            # name, (devices, hbm, heads, kv, weights), (tp, dp, smallest tp)
            ("8B on one v6e device", (1, 32.0, 32, 8, 15.0), (1, 1, 1)),
            ("8B on eight v6e devices", (8, 32.0, 32, 8, 15.0), (8, 1, 1)),
            ("70B on a tpu7x-4 (8 devices)", (8, 96.0, 64, 8, 131.0), (8, 1,
                                                                       2)),
            ("70B needs TP 8 on v6e", (8, 32.0, 64, 8, 131.0), (8, 1, 8)),
            # TP 1 fits the weights, but its hybrid state slots exhausted one
            # device at startup.
            ("Qwen3.8-27B on a tpu7x-4", (8, 96.0, 24, 4, 51.75), (8, 1, 1)),
            # TP 1 fits too, but 8 replicas compiling on one host outlast the
            # engine-ready timeout.
            ("GPT-OSS-20B on a tpu7x-4", (8, 96.0, 64, 8, 37.3), (8, 1, 1)),
            ("12 heads rule out TP 8", (8, 32.0, 12, 12, 60.0), (4, 2, 4)),
            ("weight size unknown takes the largest TP", (8, 32.0, 32, 8,
                                                          None), (8, 1, None)),
        ]
        for name, inputs, (tp, dp, smallest) in cases:
            with self.subTest(name):
                plan = check.plan_parallelism(*inputs)
                self.assertTrue(plan["ok"], plan["reason"])
                self.assertEqual(plan["source"], "recommended")
                self.assertEqual(plan["tensor_parallel_size"], tp)
                self.assertEqual(plan["data_parallel_size"], dp)
                self.assertEqual(plan["smallest_fitting_tp"], smallest)
                self.assertEqual(tp * dp, inputs[0])

    def test_reason_names_what_replicas_cost(self):
        """The weight-fit check leaves out what each replica holds."""
        plan = check.plan_parallelism(8, 32.0, 12, 12, 60.0)
        self.assertEqual(plan["data_parallel_size"], 2)
        self.assertIn("each of the 2 replicas compiles its own graphs",
                      plan["reason"])
        plan = check.plan_parallelism(8, 32.0, 32, 8, 15.0, user_tp=2)
        self.assertIn("each of the 4 replicas", plan["reason"])
        plan = check.plan_parallelism(8, 32.0, 32, 8, 15.0)
        self.assertNotIn("replicas compiles", plan["reason"])

    def test_reason_names_a_smaller_tp_that_fits(self):
        plan = check.plan_parallelism(8, 32.0, 32, 8, 15.0)
        self.assertIn("TP 1 also fits", plan["reason"])
        plan = check.plan_parallelism(8, 32.0, 64, 8, 131.0)
        self.assertNotIn("also fits", plan["reason"])

    def test_kv_heads_split_or_replicate(self):
        self.assertIsNone(check.head_problem(4, 32, 2))  # replicated
        self.assertIsNone(check.head_problem(4, 32, 8))  # split
        self.assertIn("KV heads", check.head_problem(2, 24, 3))  # neither

    def test_query_and_kv_failures_are_named_apart(self):
        self.assertIn("does not divide 28 attention heads",
                      check.head_problem(8, 28, 4))
        # Qwen2-VL-2B: vLLM's config check accepts TP 3, the layer does not.
        problem = check.head_problem(3, 12, 2)
        self.assertNotIn("attention heads", problem)
        self.assertIn("2 KV heads", problem)

    def test_linear_attention_heads_must_split(self):
        """Gated-DeltaNet heads are split across shards like query heads."""
        self.assertIsNone(check.head_problem(8, 24, 4, linear_heads=(16, 48)))
        self.assertIn("6 linear-attention key heads",
                      check.head_problem(4, 24, 4, linear_heads=(6, 48)))
        self.assertIn("20 linear-attention value heads",
                      check.head_problem(8, 24, 4, linear_heads=(16, 20)))
        plan = check.plan_parallelism(8,
                                      96.0,
                                      24,
                                      4,
                                      51.75,
                                      linear_heads=(16, 20))
        self.assertEqual(plan["tensor_parallel_size"], 4)
        self.assertEqual(plan["linear_attention_heads"], [16, 20])

    def test_unknown_size_names_no_smaller_tp(self):
        """With the fit unchecked, no smaller TP is known to fit."""
        plan = check.plan_parallelism(8, 32.0, 32, 8, None)
        self.assertIsNone(plan["smallest_fitting_tp"])
        self.assertNotIn("also fits", plan["reason"])

    def test_mla_skips_the_kv_rule(self):
        self.assertIsNotNone(check.head_problem(2, 24, 3))
        self.assertIsNone(check.head_problem(2, 24, 3, use_mla=True))

    def test_falcon_style_kv_count_limits_tp(self):
        """24 heads with 3 KV heads: only TP 1 survives on 8 devices."""
        plan = check.plan_parallelism(8, 32.0, 24, 3, 60.0)
        self.assertFalse(plan["ok"])
        plan = check.plan_parallelism(8, 96.0, 24, 3, 60.0)
        self.assertEqual(plan["tensor_parallel_size"], 1)
        self.assertEqual(plan["smallest_fitting_tp"], 1)

    def test_vllm_rejection_falls_through_to_next_tp(self):
        plan = check.plan_parallelism(8,
                                      32.0,
                                      32,
                                      8,
                                      15.0,
                                      rejected={8: "vLLM rejected TP 8: no"})
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["tensor_parallel_size"], 4)
        self.assertEqual(plan["data_parallel_size"], 2)
        self.assertEqual(plan["smallest_fitting_tp"], 1)
        self.assertIn("vLLM rejected TP 8: no", plan["reason"])

    def test_vllm_rejection_of_user_tp_fails(self):
        plan = check.plan_parallelism(8,
                                      32.0,
                                      32,
                                      8,
                                      15.0,
                                      user_tp=4,
                                      rejected={4: "vLLM rejected TP 4: no"})
        self.assertFalse(plan["ok"])
        self.assertIn("vLLM rejected TP 4: no", plan["reason"])

    def test_tp_without_a_device_layout_is_skipped(self):
        """Qwen3.8-27B on tpu7x: torch_tpu lays out 2, 4 or 8 devices only."""
        rejected = check.topology_rejections(8, _layouts({2, 4, 8}))
        self.assertEqual(sorted(rejected), [1])
        plan = check.plan_parallelism(8, 96.0, 24, 4, 51.75, rejected=rejected)
        self.assertTrue(plan["ok"], plan["reason"])
        self.assertEqual(plan["tensor_parallel_size"], 8)
        self.assertEqual(plan["data_parallel_size"], 1)
        self.assertEqual(plan["smallest_fitting_tp"], 2)
        self.assertIn("torch_tpu has no device layout for TP 1",
                      plan["reason"])

    def test_layouts_come_from_the_installed_lookup(self):
        """v6e lays out 1, 4 and 8 devices, so there TP 2 is the one skipped."""
        rejected = check.topology_rejections(8, _layouts({1, 4, 8}))
        self.assertEqual(sorted(rejected), [2])
        plan = check.plan_parallelism(8, 32.0, 32, 8, 40.0, rejected=rejected)
        self.assertEqual(plan["tensor_parallel_size"], 8)
        # Without a layout for 8, the largest TP left is taken, with replicas.
        rejected = check.topology_rejections(8, _layouts({1, 2, 4}))
        plan = check.plan_parallelism(8, 32.0, 32, 8, 40.0, rejected=rejected)
        self.assertEqual(plan["tensor_parallel_size"], 4)
        self.assertEqual(plan["data_parallel_size"], 2)
        self.assertIn("2 replicas use all 8 devices", plan["reason"])

    def test_a_lookup_that_returns_nothing_rules_the_tp_out(self):
        rejected = check.topology_rejections(2, lambda n: None
                                             if n == 1 else "x")
        self.assertEqual(sorted(rejected), [1])
        self.assertIn("no layout returned", rejected[1])

    def test_user_tp_without_a_device_layout_fails(self):
        rejected = check.topology_rejections(8, _layouts({2, 4, 8}))
        plan = check.plan_parallelism(8,
                                      96.0,
                                      24,
                                      4,
                                      51.75,
                                      user_tp=1,
                                      rejected=rejected)
        self.assertFalse(plan["ok"])
        self.assertIn("no device layout for TP 1", plan["reason"])

    def test_every_tp_refused_says_so(self):
        rejected = check.topology_rejections(8, _layouts(set()))
        plan = check.plan_parallelism(8, 96.0, 24, 4, 51.75, rejected=rejected)
        self.assertFalse(plan["ok"])
        self.assertIn("every head-compatible TP [1, 2, 4, 8] was rejected",
                      plan["reason"])

    def test_refusal_is_found_anywhere_in_the_message(self):

        def lookup(n):
            if n == 1:
                raise RuntimeError("topology lookup failed\n"
                                   "No TPU topology found for count: 1")
            return "x"

        rejected = check.topology_rejections(8, lookup)
        self.assertEqual(sorted(rejected), [1])
        self.assertIn("(topology lookup failed)", rejected[1])

    def test_other_lookup_failures_rule_nothing_out(self):
        """A changed signature or a busy runtime is not a missing layout."""
        failures = (TypeError("_get_tpu_topology() missing 1 required "
                              "positional argument"),
                    RuntimeError("TPU initialization failed: device busy"),
                    ValueError("No TPU devices found."))
        for error in failures:
            with self.subTest(error):

                def lookup(n, error=error):
                    raise error

                with self.assertRaises(check.TopologyLookupError) as caught:
                    check.topology_rejections(8, lookup)
                self.assertIn(str(error), str(caught.exception))

    def test_plugin_without_the_lookup_rules_nothing_out(self):
        self.assertIsNone(check.find_topology_lookup())

    def test_vllm_error_drops_pydantic_framing(self):
        exc = ValueError(
            "1 validation error for VllmConfig\n  Value error, Total number "
            "of attention heads (28) must be divisible by tensor parallel "
            "size (8). [type=value_error, input_value=ArgsKwargs((), {})]")
        self.assertEqual(
            check.vllm_error(exc),
            "Total number of attention heads (28) must be divisible by "
            "tensor parallel size (8).")
        self.assertEqual(check.vllm_error(RuntimeError("boom\nmore")), "boom")

    def test_nothing_fits(self):
        plan = check.plan_parallelism(1, 16.0, 32, 8, 131.0)
        self.assertFalse(plan["ok"])
        self.assertEqual(plan["recommended_engine"], "custom_fastapi")
        self.assertIsNone(plan["tensor_parallel_size"])
        self.assertIn("do not fit", plan["reason"])

    def test_user_tp_wins(self):
        plan = check.plan_parallelism(8, 32.0, 32, 8, 15.0, user_tp=4)
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["recommended_engine"], "vllm")
        self.assertEqual(plan["source"], "user")
        self.assertEqual(plan["tensor_parallel_size"], 4)
        self.assertEqual(plan["data_parallel_size"], 2)
        self.assertIn("stated by the user", plan["reason"])

    def test_user_tp_records_unused_devices_and_oom_risk(self):
        plan = check.plan_parallelism(8, 32.0, 24, 24, 131.0, user_tp=3)
        self.assertTrue(plan["ok"])
        self.assertIn("2 of 8 devices unused", plan["reason"])
        self.assertIn("expect OOM", plan["reason"])

    def test_user_tp_with_unknown_size_says_fit_unchecked(self):
        plan = check.plan_parallelism(8, 32.0, 32, 8, None, user_tp=2)
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["tensor_parallel_size"], 2)
        self.assertIn("memory fit was not checked", plan["reason"])

    def test_user_tp_that_vllm_would_refuse(self):
        for user_tp, words in ((3, "does not divide"), (16, "exceeds")):
            with self.subTest(user_tp):
                plan = check.plan_parallelism(8, 32.0, 32, 8, 15.0, user_tp)
                self.assertFalse(plan["ok"])
                self.assertIn(words, plan["reason"])

    def test_sglang_engine_plan_and_rejections(self):
        plan = check.plan_parallelism(8,
                                      32.0,
                                      32,
                                      8,
                                      15.0,
                                      rejected={8: "unsupported"},
                                      engine="sglang")
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["recommended_engine"], "sglang")
        self.assertEqual(plan["tensor_parallel_size"], 4)
        self.assertEqual(plan["data_parallel_size"], 2)
        self.assertEqual(plan["smallest_fitting_tp"], 1)
        self.assertIn("SGLang rejected TP 8: unsupported", plan["reason"])

        user_plan = check.plan_parallelism(8,
                                           32.0,
                                           32,
                                           8,
                                           15.0,
                                           user_tp=3,
                                           engine="sglang")
        self.assertFalse(user_plan["ok"])
        self.assertIn("SGLang will refuse it", user_plan["reason"])

    def test_sglang_main_validates_registry_topology_and_trust_remote_code(
            self):
        resolved_calls = []
        seen_trust = []

        class FakeServerArgs:

            def __init__(self,
                         model_path,
                         tp_size=1,
                         dp_size=1,
                         device="tpu",
                         trust_remote_code=False):
                self.model_path = model_path
                self.tp_size = tp_size
                self.dp_size = dp_size
                self.device = device
                self.trust_remote_code = trust_remote_code
                seen_trust.append(trust_remote_code)

            def resolve_once(self):
                resolved_calls.append((self.tp_size, self.dp_size))
                if self.tp_size == 8:
                    raise ValueError("TP 8 not supported for this test")

        class FakeAttentionArch:
            MLA = "MLA"
            MHA = "MHA"

        class FakeModelConfig:
            arch_name = "Qwen3MoeForCausalLM"

            @classmethod
            def from_server_args(cls, server_args):
                hf_cfg = types.SimpleNamespace(
                    architectures=[cls.arch_name],
                    text_config=types.SimpleNamespace(
                        num_attention_heads=32,
                        num_key_value_heads=8,
                    ),
                    to_dict=lambda: {"architectures": [cls.arch_name]},
                )
                return types.SimpleNamespace(
                    hf_config=hf_cfg,
                    hf_text_config=hf_cfg.text_config,
                    num_attention_heads=32,
                    get_total_num_kv_heads=lambda: 8,
                    attention_arch=FakeAttentionArch.MHA,
                    context_len=32768,
                    quantization=None,
                )

        class FakeModelRegistry:
            models = {"Qwen3MoeForCausalLM": object}

            @classmethod
            def get_supported_archs(cls):
                return cls.models.keys()

        sglang_mod = types.ModuleType("sglang")
        srt_mod = types.ModuleType("sglang.srt")
        configs_mod = types.ModuleType("sglang.srt.configs")
        mc_mod = types.ModuleType("sglang.srt.configs.model_config")
        mc_mod.AttentionArch = FakeAttentionArch
        mc_mod.ModelConfig = FakeModelConfig
        models_mod = types.ModuleType("sglang.srt.models")
        reg_mod = types.ModuleType("sglang.srt.models.registry")
        reg_mod.ModelRegistry = FakeModelRegistry
        sa_mod = types.ModuleType("sglang.srt.server_args")
        sa_mod.ServerArgs = FakeServerArgs
        sglang_tpu_mod = types.ModuleType("sglang_tpu")
        patches_mod = types.ModuleType("sglang_tpu.patches")
        patches_mod._resolve_single_host_tpu_topology = _layouts({1, 4, 8})

        modules = {
            "sglang": sglang_mod,
            "sglang.srt": srt_mod,
            "sglang.srt.configs": configs_mod,
            "sglang.srt.configs.model_config": mc_mod,
            "sglang.srt.models": models_mod,
            "sglang.srt.models.registry": reg_mod,
            "sglang.srt.server_args": sa_mod,
            "sglang_tpu": sglang_tpu_mod,
            "sglang_tpu.patches": patches_mod,
        }
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "model.safetensors"), "wb") as f:
                f.write(b"0" * 1024)
            plan_path = os.path.join(tmp, "plan.json")
            argv = [
                "check", "--engine", "sglang", "--model", tmp, "--devices",
                "8", "--hbm-gb-per-device", "32", "--plan-json", plan_path
            ]
            with unittest.mock.patch.dict(sys.modules, modules), \
                    unittest.mock.patch.object(sys, "argv", argv), \
                    contextlib.redirect_stdout(io.StringIO()) as out, \
                    self.assertRaises(SystemExit) as cm:
                check.main()
            self.assertEqual(cm.exception.code, 0, out.getvalue())
            self.assertTrue(resolved_calls)
            self.assertEqual(set(seen_trust), {False})
            with open(plan_path, encoding="utf-8") as f:
                plan = json.load(f)
            # TP 8 rejected by ServerArgs.resolve_once(), TP 2 rejected by
            # sglang_tpu topology lookup -> selects TP 4 x DP 2.
            self.assertEqual(plan["recommended_engine"], "sglang")
            self.assertEqual(plan["tensor_parallel_size"], 4)
            self.assertEqual(plan["data_parallel_size"], 2)
            self.assertIn("SGLang rejected TP 8", plan["reason"])
            self.assertIn("torch_tpu has no device layout for TP 2",
                          plan["reason"])

            # Unsupported architecture in ModelRegistry fails pre-flight.
            FakeModelConfig.arch_name = "UnsupportedArchForCausalLM"
            with unittest.mock.patch.dict(sys.modules, modules), \
                    unittest.mock.patch.object(sys, "argv", argv + ["--trust-remote-code"]), \
                    contextlib.redirect_stdout(io.StringIO()) as out2, \
                    self.assertRaises(SystemExit) as cm2:
                check.main()
            self.assertEqual(cm2.exception.code, 1)
            self.assertIn("not supported by SGLang", out2.getvalue())
            self.assertIn(True, seen_trust)


APPLIER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "vllm_patches", "apply_vllm_patch.sh")
_PATCH = """--- a/vllm_torchtpu/quant.py
+++ b/vllm_torchtpu/quant.py
@@ -1 +1,2 @@
 MXFP4 = "mxfp4"
+GPT_OSS_MXFP4 = "gpt_oss_mxfp4"
"""


class TestApplyVllmPatch(unittest.TestCase):
    """The patch is applied to whichever plugin the target imports."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        plugin = os.path.join(self.tmp, "site", "vllm_torchtpu")
        os.makedirs(plugin)
        # The real plugin starts the TPU runtime on import and prints
        # banners to stdout; nothing may read that output as a path.
        with open(os.path.join(plugin, "__init__.py"), "w") as f:
            f.write('print("Device type: tpu, Device index: default")\n')
        self.quant = os.path.join(plugin, "quant.py")
        with open(self.quant, "w") as f:
            f.write('MXFP4 = "mxfp4"\n')
        inputs = os.path.join(self.tmp, "inputs")
        os.makedirs(inputs)
        self.patch = os.path.join(inputs, "fix.patch")
        with open(self.patch, "w") as f:
            f.write(_PATCH)
        shutil.copy(APPLIER, inputs)
        path = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = os.path.join(self.tmp, "site")
        self.addCleanup(
            lambda: os.environ.__setitem__("PYTHONPATH", path)
            if path is not None else os.environ.pop("PYTHONPATH", None))
        # The check looks the plugin up in its own process, the shell applier
        # in a child: each finds it through its own path.
        sys.path.insert(0, os.path.join(self.tmp, "site"))
        self.addCleanup(sys.path.remove, os.path.join(self.tmp, "site"))
        importlib.invalidate_caches()

    def test_applies_then_is_idempotent(self):
        self.assertIsNone(check.apply_vllm_patch(self.patch))
        self.assertIsNone(check.apply_vllm_patch(self.patch))
        with open(self.quant) as f:
            self.assertEqual(f.read().count("GPT_OSS_MXFP4"), 1)

    def test_drifted_plugin_is_refused(self):
        with open(self.quant, "w") as f:
            f.write('MXFP4 = "mxfp4-v2"\n')
        problem = check.apply_vllm_patch(self.patch)
        self.assertIn("does not apply", problem)

    def test_applier_finds_the_plugin_without_importing_it(self):
        """The shell applier's own lookup, as launch_vllm.sh calls it."""
        done = subprocess.run(["bash", APPLIER, self.patch],
                              capture_output=True,
                              text=True,
                              env=dict(os.environ,
                                       TPU_AGENT_SUITE_PYTHON=sys.executable),
                              check=False)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn("Device type", done.stdout)
        with open(self.quant) as f:
            self.assertIn("GPT_OSS_MXFP4", f.read())

    def test_missing_plugin_is_reported(self):
        os.environ["PYTHONPATH"] = self.tmp
        site = os.path.join(self.tmp, "site")
        sys.path.remove(site)
        self.addCleanup(sys.path.insert, 0, site)  # setUp's cleanup removes it
        importlib.invalidate_caches()
        self.assertIn("not importable", check.apply_vllm_patch(self.patch))
        done = subprocess.run(["bash", APPLIER, self.patch],
                              capture_output=True,
                              text=True,
                              env=dict(os.environ,
                                       TPU_AGENT_SUITE_PYTHON=sys.executable),
                              check=False)
        self.assertEqual(done.returncode, 3)
        self.assertIn("not importable", done.stderr)

    def test_applier_must_travel_beside_the_patch(self):
        os.remove(os.path.join(self.tmp, "inputs", "apply_vllm_patch.sh"))
        self.assertIn("same directory", check.apply_vllm_patch(self.patch))


class TestGcsModel(unittest.TestCase):
    """A gs:// model: vLLM reads a local copy, and the size comes from GCS."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.copies = []
        module = types.ModuleType("fetch_weights")

        def fetch_gcs(url, small_only=False):
            self.copies.append((url, small_only))
            return self.tmp

        module.fetch_gcs = fetch_gcs
        module.gcs_weight_bytes = lambda url, local: 7 * 1024**3
        self.module = module

    def test_the_index_total_wins(self):
        with open(os.path.join(self.tmp, "model.safetensors.index.json"),
                  "w",
                  encoding="utf-8") as f:
            json.dump({"metadata": {"total_size": 5 * 1024**3}}, f)
        with unittest.mock.patch.dict(sys.modules,
                                      {"fetch_weights": self.module}):
            local, size = check.gcs_model("gs://b/m")
        self.assertEqual((local, size), (self.tmp, 5 * 1024**3))
        self.assertEqual(self.copies, [("gs://b/m", True)])

    def test_without_an_index_the_listing_sizes_it(self):
        with unittest.mock.patch.dict(sys.modules,
                                      {"fetch_weights": self.module}):
            self.assertEqual(check.gcs_model("gs://b/m")[1], 7 * 1024**3)

    def test_main_gives_vllm_the_copy_never_the_url(self):
        seen = []

        class EngineArgs:

            def __init__(self, model, tensor_parallel_size, data_parallel_size,
                         enforce_eager):
                seen.append(model)

            def create_engine_config(self):
                mc = unittest.mock.MagicMock()
                mc.model_arch_config.total_num_attention_heads = 8
                mc.get_total_num_kv_heads.return_value = 8
                mc.use_mla = False
                mc.hf_text_config = types.SimpleNamespace()
                return types.SimpleNamespace(model_config=mc)

        arg_utils = types.ModuleType("vllm.engine.arg_utils")
        arg_utils.EngineArgs = EngineArgs
        modules = {
            "fetch_weights": self.module,
            "vllm_torchtpu": types.ModuleType("vllm_torchtpu"),
            "vllm": types.ModuleType("vllm"),
            "vllm.engine": types.ModuleType("vllm.engine"),
            "vllm.engine.arg_utils": arg_utils,
        }
        plan = os.path.join(self.tmp, "plan.json")
        argv = [
            "check", "--model", "gs://b/m", "--devices", "8",
            "--hbm-gb-per-device", "16", "--plan-json", plan
        ]
        with unittest.mock.patch.dict(sys.modules, modules), \
                unittest.mock.patch.object(sys, "argv", argv), \
                unittest.mock.patch.object(check, "find_topology_lookup",
                                           return_value=None), \
                contextlib.redirect_stdout(io.StringIO()) as out, \
                self.assertRaises(SystemExit) as cm:
            check.main()
        self.assertEqual(cm.exception.code, 0, out.getvalue())
        self.assertTrue(seen)
        self.assertEqual(set(seen), {self.tmp})
        self.assertIn("Base Model Size on Disk: ~7.0 GiB", out.getvalue())
        # No --points: nothing to size --max-num-seqs by.
        with open(plan, encoding="utf-8") as f:
            written = json.load(f)
        self.assertIsNone(written["max_num_seqs"])
        self.assertEqual(written["max_num_seqs_other_layouts"], [])

    def test_main_fails_clearly_when_the_folder_cannot_be_read(self):

        def denied(url, small_only=False):
            raise PermissionError(13, f"no permission to read {url}")

        self.module.fetch_gcs = denied
        modules = {
            "fetch_weights": self.module,
            "vllm_torchtpu": types.ModuleType("vllm_torchtpu"),
        }
        argv = [
            "check", "--model", "gs://b/m", "--devices", "8",
            "--hbm-gb-per-device", "16"
        ]
        with unittest.mock.patch.dict(sys.modules, modules), \
                unittest.mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(io.StringIO()) as out, \
                self.assertRaises(SystemExit) as cm:
            check.main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("could not read gs://b/m", out.getvalue())
        self.assertIn("no permission", out.getvalue())


_CONFIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata",
                        "oss_configs")
# tpu7x: 8 devices of 94.75 GiB.
_DEVICES, _HBM = 8, 94.75


def _shape(rules, name):
    with open(os.path.join(_CONFIGS, name), encoding="utf-8") as f:
        return rules.ModelShape.from_hf_config(json.load(f))


class TestKvCheck(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rules = check.load_mesh_rules()

    def test_kv_warning_when_a_point_does_not_fit(self):
        shape = _shape(self.rules, "gemma-4-31B-it.json")
        short = check.kv_warnings(self.rules, shape, 8, 1, _HBM, 58.0,
                                  [(8192, 1024, 2048)])
        self.assertEqual(len(short), 1)
        self.assertGreater(short[0]["kv_gib_needed"], short[0]["kv_gib_free"])
        self.assertEqual(
            check.kv_warnings(self.rules, shape, 8, 1, _HBM, 58.0,
                              [(1024, 1024, 1)]), [])

    def test_max_num_seqs_covers_the_top_point_per_replica(self):
        shape = _shape(self.rules, "gemma-4-31B-it.json")
        sized = check.max_num_seqs_plan(self.rules, shape, 4, 2, _HBM, 58.0,
                                        [(1024, 1024, 1), (1024, 1024, 256)])
        self.assertEqual(sized["target_per_replica"], 128)
        self.assertEqual(sized["value"], 128)
        self.assertFalse(sized["capped"])
        # A DP that does not divide the concurrency rounds up.
        sized = check.max_num_seqs_plan(self.rules, shape, 4, 3, _HBM, 58.0,
                                        [(1024, 1024, 100)])
        self.assertEqual(sized["value"], 34)

    def test_max_num_seqs_is_capped_by_the_kv_cache(self):
        shape = _shape(self.rules, "gemma-4-31B-it.json")
        sized = check.max_num_seqs_plan(self.rules, shape, 8, 1, _HBM, 58.0,
                                        [(8192, 1024, 2048)])
        fit = sized["points"][0]["kv_fit_per_replica"]
        self.assertLess(fit, 2048)
        self.assertEqual(sized["value"], fit)
        self.assertTrue(sized["capped"])

    def test_a_long_point_does_not_cap_a_short_one(self):
        shape = _shape(self.rules, "gemma-4-31B-it.json")
        long_only = check.max_num_seqs_plan(self.rules, shape, 8, 1, _HBM,
                                            58.0, [(8192, 1024, 256)])
        self.assertLess(long_only["value"], 256)
        both = check.max_num_seqs_plan(self.rules, shape, 8, 1, _HBM, 58.0,
                                       [(8192, 1024, 256), (1024, 1024, 256)])
        self.assertEqual(both["value"], 256)
        self.assertFalse(both["capped"])

    def test_no_value_when_no_point_asks_for_more_than_the_floor(self):
        shape = _shape(self.rules, "gemma-4-31B-it.json")
        # One measurement at c1, or c16 over two replicas: vLLM's default
        # holds them, and a server sized for them would cap a later sweep.
        for dp, points in ((1, [(1024, 128, 1)]), (2, [(1024, 1024, 16)])):
            with self.subTest(points=points):
                self.assertIsNone(
                    check.max_num_seqs_plan(self.rules, shape, 8 // dp, dp,
                                            _HBM, 58.0, points))
        self.assertIsNone(
            check.max_num_seqs_plan(self.rules, shape, 8, 1, _HBM, 58.0, []))

    def test_floor_when_one_point_fits_fewer_and_another_more(self):
        shape = _shape(self.rules, "gemma-4-31B-it.json")
        # 40 GiB devices: the 256k point holds 3, the short one asks for 4.
        sized = check.max_num_seqs_plan(self.rules, shape, 8, 1, 40.0, 58.0,
                                        [(262144, 1024, 100), (1024, 128, 4)])
        fits = [p["kv_fit_per_replica"] for p in sized["points"]]
        self.assertLess(fits[0], check.MIN_MAX_NUM_SEQS)
        self.assertGreater(fits[1], check.MIN_MAX_NUM_SEQS)
        self.assertEqual(sized["value"], check.MIN_MAX_NUM_SEQS)
        self.assertTrue(sized["capped"])

    def test_no_value_when_the_kv_cache_holds_fewer_than_the_floor(self):
        shape = _shape(self.rules, "gemma-4-31B-it.json")
        # 16 GiB devices with 58 GiB of weights at TP 8: little cache left.
        self.assertIsNone(
            check.max_num_seqs_plan(self.rules, shape, 8, 1, 16.0, 58.0,
                                    [(131072, 1024, 64)]))

    def test_linear_attention_adds_no_cap(self):
        shape = self.rules.ModelShape(num_layers=4,
                                      num_heads=8,
                                      num_kv_heads=8,
                                      head_dim=128,
                                      layer_types=("linear_attention", ) * 4)
        sized = check.max_num_seqs_plan(self.rules, shape, 8, 1, _HBM, 58.0,
                                        [(8192, 1024, 512)])
        self.assertEqual(sized["value"], 512)
        self.assertIsNone(sized["points"][0]["kv_fit_per_replica"])
        # Hybrid: only the full-attention layer takes per-token cache, so
        # the value is the largest any point asks for.
        hybrid = self.rules.ModelShape(num_layers=4,
                                       num_heads=8,
                                       num_kv_heads=8,
                                       head_dim=128,
                                       layer_types=("linear_attention", ) * 3 +
                                       ("full_attention", ))
        sized = check.max_num_seqs_plan(self.rules, hybrid, 8, 1, _HBM, 58.0,
                                        [(8192, 1024, 512), (1024, 128, 64)])
        self.assertEqual(sized["value"], 512)
        self.assertFalse(sized["capped"])

    def test_other_layouts_one_replica_and_the_passed_over_one(self):
        plan = {
            "tensor_parallel_size": 4,
            "data_parallel_size": 2,
            "traffic_rule": {
                "instead_of": {
                    "tensor_parallel_size": 8,
                    "data_parallel_size": 1
                }
            }
        }
        layouts = check.max_num_seqs_layouts(
            plan, lambda tp, dp: {"value": 256 // dp})
        self.assertEqual(
            [(x["tensor_parallel_size"], x["data_parallel_size"], x["value"])
             for x in layouts], [(4, 1, 256), (8, 1, 256)])
        # TP 8 x DP 1 already is one replica: nothing else to size.
        self.assertEqual(
            check.max_num_seqs_layouts(
                {
                    "tensor_parallel_size": 8,
                    "data_parallel_size": 1
                }, lambda tp, dp: None), [])

    def test_the_halving_is_short_only_where_the_larger_tp_is_not(self):
        plan = {
            "tensor_parallel_size": 4,
            "data_parallel_size": 2,
            "traffic_rule": {
                "instead_of": {
                    "tensor_parallel_size": 8,
                    "data_parallel_size": 1
                }
            }
        }
        a = {"isl": 8192, "osl": 1024, "concurrency": 256}
        b = {"isl": 8192, "osl": 1024, "concurrency": 512}
        # Both points short at TP 4 x DP 2; TP 8 is short at the larger too.
        short = {(4, 2): [a, b], (8, 1): [b]}
        self.assertEqual(
            check.halving_kv_short(plan, lambda tp, dp: short[tp, dp]), [a])
        # Short in both layouts: TP 8 would not fix it, so the rule stays.
        short = {(4, 2): [b], (8, 1): [b]}
        self.assertEqual(
            check.halving_kv_short(plan, lambda tp, dp: short[tp, dp]), [])


class TestTrafficLayout(unittest.TestCase):
    """One TP halving for long prompts at high concurrency, measured."""

    # Gemma-4-31B on tpu7x 2x2x1: 32 heads, 16 KV heads, ~58 GiB weights.
    GEMMA4 = (8, 94.75, 32, 16, 58.0)

    def test_long_prompts_at_high_concurrency_halve_tp(self):
        plan = check.plan_parallelism(*self.GEMMA4,
                                      points=[(1024, 1024, 32),
                                              (8192, 1024, 128)])
        self.assertEqual(
            (plan["tensor_parallel_size"], plan["data_parallel_size"]), (4, 2))
        rule = plan["traffic_rule"]
        self.assertEqual(rule["points"], ["8192:1024:128"])
        self.assertEqual(rule["instead_of"]["tensor_parallel_size"], 8)
        self.assertIn("time to first token", plan["reason"])

    def test_latency_keeps_the_largest_tp(self):
        plan = check.plan_parallelism(*self.GEMMA4,
                                      points=[(8192, 1024, 128)],
                                      objective="latency")
        self.assertEqual(plan["tensor_parallel_size"], 8)
        self.assertNotIn("traffic_rule", plan)

    def test_short_prompts_or_few_users_keep_the_largest_tp(self):
        for points in ([(1024, 1024, 256)], [(8192, 1024, 32)], None):
            with self.subTest(points=points):
                plan = check.plan_parallelism(*self.GEMMA4, points=points)
                self.assertEqual(plan["tensor_parallel_size"], 8)
                self.assertNotIn("traffic_rule", plan)

    def test_halving_needs_the_smaller_tp_to_fit_and_run(self):
        # Weights too big for TP 4: stay at TP 8.
        plan = check.plan_parallelism(8,
                                      94.75,
                                      32,
                                      16,
                                      400.0,
                                      points=[(8192, 1024, 128)])
        self.assertEqual(plan["tensor_parallel_size"], 8)
        # Qwen3.8-27B: TP 8 ruled out, so TP 4 x DP 2 is planned; a plan
        # that already runs replicas is not halved again.
        plan = check.plan_parallelism(8,
                                      94.75,
                                      24,
                                      4,
                                      51.75,
                                      rejected={8: "qwen38_tp8_engine_init"},
                                      linear_heads=(16, 48),
                                      points=[(8192, 1024, 128)])
        self.assertEqual(plan["tensor_parallel_size"], 4)
        self.assertNotIn("traffic_rule", plan)

    def test_a_stated_tp_still_wins(self):
        plan = check.plan_parallelism(*self.GEMMA4,
                                      user_tp=8,
                                      points=[(8192, 1024, 128)])
        self.assertEqual(plan["tensor_parallel_size"], 8)
        self.assertNotIn("traffic_rule", plan)


class TestValidatedPlan(unittest.TestCase):
    """Review on #898: every plan, the KV fallback included, goes to vLLM."""

    GEMMA4 = (8, 94.75, 32, 16, 58.0)
    POINTS = [(1024, 1024, 32), (8192, 1024, 256)]

    def _run(self, kv_short, refuse=()):
        rejected, accepted, validated = {}, set(), []

        def make_plan(use_traffic):
            return check.plan_parallelism(
                *self.GEMMA4,
                rejected=rejected,
                points=self.POINTS if use_traffic else None)

        def validate(tp, dp):
            validated.append((tp, dp))
            if tp in refuse:
                raise ValueError(f"TP {tp} refused")

        plan = check.validated_plan(make_plan, validate, rejected, accepted,
                                    kv_short)
        return plan, validated, rejected

    def test_the_kv_fallback_is_validated_by_vllm(self):
        plan, validated, _ = self._run(
            lambda p: p["tensor_parallel_size"] == 4)
        self.assertEqual(
            (plan["tensor_parallel_size"], plan["data_parallel_size"]), (8, 1))
        self.assertIn("traffic_rule_dropped", plan)
        self.assertNotIn("traffic_rule", plan)
        self.assertIn((8, 1), validated)  # the fallback went through vLLM

    def test_a_fallback_vllm_refuses_is_replanned(self):
        plan, validated, rejected = self._run(
            lambda p: p["tensor_parallel_size"] == 4, refuse=(8, ))
        self.assertIn(8, rejected)
        self.assertNotEqual(plan["tensor_parallel_size"], 8)
        self.assertIn((8, 1), validated)

    def test_the_rule_stays_when_the_kv_cache_holds(self):
        plan, validated, _ = self._run(lambda p: False)
        self.assertEqual(
            (plan["tensor_parallel_size"], plan["data_parallel_size"]), (4, 2))
        self.assertIn("traffic_rule", plan)
        self.assertEqual(set(validated), {(4, 2), (4, 1)})


class TestParsePoints(unittest.TestCase):

    def test_points(self):
        self.assertEqual(check.parse_points("1024:1024:128, 8192:1024:64"),
                         [(1024, 1024, 128), (8192, 1024, 64)])
        self.assertEqual(check.parse_points(None), [])

    def test_bad_point(self):
        with self.assertRaises(ValueError):
            check.parse_points("1024:0:8")


if __name__ == "__main__":
    unittest.main()
