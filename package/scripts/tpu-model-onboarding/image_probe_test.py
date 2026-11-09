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
"""Tests for the task image probe."""

import importlib.metadata
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

import image_probe


def _installed(versions: dict[str, str]):
    """A stand-in for importlib.metadata.version over `versions`."""

    def version(name: str) -> str:
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    return version


def _importable(names: set[str]):
    """A stand-in for importlib.util.find_spec that finds only `names`."""
    return lambda name: object() if name in names else None


# The image the gpt-oss-20b onboarding ran in.
TORCH_TPU_IMAGE = {
    "torch": "2.9.0",
    "torch_tpu": "0.1.1.dev20260714",
    "transformers": "4.57.1",
    "huggingface_hub": "0.36.0",
}

# The vLLM stacks of the old and the new tpu-model-onboarding images.
OLD_SERVING_IMAGE = {
    "torch": "2.11.0",
    "vllm": "0.22.1",
    "vllm_torchtpu": "0.0.0",
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
}
NEW_SERVING_IMAGE = {
    "torch": "2.13.0+cpu",
    "vllm": "0.29.0",
    "vllm_torchtpu": "0.0.0",
    "jax": "0.11.2",
    "jaxlib": "0.11.2",
}

# The entry point the vLLM TorchTPU plugin declares in its pyproject.toml.
TORCHTPU_ENTRY_POINT = importlib.metadata.EntryPoint(
    "torchtpu", "vllm_torchtpu.platforms:register_tpu_platform",
    "vllm.platform_plugins")


class ImageProbeTest(unittest.TestCase):

    def _probe(self, versions: dict[str, str], modules: set[str],
               **kwargs) -> dict:
        with mock.patch.object(importlib.metadata, "version",
                               _installed(versions)), \
                mock.patch.object(image_probe.importlib.util, "find_spec",
                                  _importable(modules)):
            return image_probe.probe(**kwargs)

    def test_torch_tpu_image(self):
        report = self._probe(
            TORCH_TPU_IMAGE,
            {"torch", "torch_tpu", "transformers", "huggingface_hub"},
            check_device=False)
        self.assertEqual(report["distributions"], TORCH_TPU_IMAGE)
        self.assertTrue(report["torch_tpu_present"])
        self.assertEqual(report["device"]["status"], "skipped")

    def test_image_without_torch_tpu(self):
        report = self._probe({"torch": "2.9.0"}, {"torch"}, check_device=False)
        self.assertFalse(report["torch_tpu_present"])
        self.assertFalse(report["sglang_present"])

    def test_sglang_image_reported(self):
        report = self._probe(
            {
                "torch": "2.9.0",
                "sglang": "0.5.3",
                "sgl-kernel": "0.1.0"
            }, {"torch", "torch_tpu", "sglang"},
            check_device=False)
        self.assertTrue(report["sglang_present"])
        self.assertEqual(report["distributions"]["sglang"], "0.5.3")
        self.assertEqual(report["distributions"]["sgl-kernel"], "0.1.0")

    def test_modules_are_never_imported(self):
        # Importing torch_tpu would initialise the TPU runtime; only
        # --touch_device may do that.
        sys.modules.pop("torch_tpu", None)
        with mock.patch("builtins.__import__",
                        side_effect=AssertionError("imported")) as imp:
            image_probe.module_presence(("torch_tpu", ))
        imp.assert_not_called()
        self.assertNotIn("torch_tpu", sys.modules)

    def test_preflight_runnable_needs_all_preflight_modules(self):
        full = {"absl", "huggingface_hub", "transformers", "yaml"}
        self.assertTrue(
            self._probe({}, full, check_device=False)["preflight_runnable"])
        self.assertFalse(
            self._probe({}, full - {"transformers"},
                        check_device=False)["preflight_runnable"])

    def test_custom_fastapi_runnable_needs_fastapi_pydantic_and_uvicorn(self):
        full = {"fastapi", "pydantic", "uvicorn"}
        report = self._probe({}, full, check_device=False)
        self.assertTrue(report["custom_fastapi_runnable"])
        for missing in full:
            report = self._probe({}, full - {missing}, check_device=False)
            self.assertFalse(report["custom_fastapi_runnable"])
            self.assertFalse(report["modules"][missing])

    def test_tpu_env_reported(self):
        with mock.patch.dict(os.environ, {
                "TPU_ACCELERATOR_TYPE": "tpu7x-8",
                "UNRELATED": "x"
        }):
            report = self._probe({}, set(), check_device=False)
        self.assertEqual(report["tpu_env"],
                         {"TPU_ACCELERATOR_TYPE": "tpu7x-8"})

    def test_touch_device_without_torch_fails(self):
        with mock.patch.dict(sys.modules, {"torch": None}):
            result = image_probe.touch_device()
        self.assertEqual(result["status"], "fail")
        self.assertTrue(result["detail"].startswith("ModuleNotFoundError"))

    def _torch(self, *outcomes):
        """A torch whose allocations meet `outcomes` in turn: None or an error."""
        return types.SimpleNamespace(
            zeros=mock.Mock(side_effect=list(outcomes)),
            device=lambda kind: f"device:{kind}")

    def test_touch_device_relies_on_autoload(self):
        # A workload only asks for the device; torch_tpu must load itself.
        torch = self._torch(None)
        with mock.patch.dict(sys.modules, {"torch": torch}), \
                mock.patch("builtins.__import__", wraps=__import__) as imp:
            sys.modules.pop("torch_tpu", None)
            result = image_probe.touch_device()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["loaded_by"], "autoload")
        torch.zeros.assert_called_once_with(1, device="device:tpu")
        self.assertNotIn("torch_tpu",
                         [c.args[0] for c in imp.call_args_list if c.args])

    def test_touch_device_falls_back_to_importing_torch_tpu(self):
        torch = self._torch(RuntimeError("Unknown device type: tpu"), None)
        with mock.patch.dict(sys.modules, {
                "torch": torch,
                "torch_tpu": types.ModuleType("torch_tpu")
        }):
            result = image_probe.touch_device()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["loaded_by"], "import")
        self.assertIn("Unknown device type: tpu", result["detail"])
        self.assertEqual(torch.zeros.call_count, 2)

    def test_touch_device_without_torch_tpu_reports_the_autoload_error(self):
        torch = self._torch(RuntimeError("Unknown device type: tpu"))
        with mock.patch.dict(sys.modules, {"torch": torch}), \
                mock.patch.object(image_probe, "_importable",
                                  return_value=False):
            result = image_probe.touch_device()
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["detail"],
                         "RuntimeError: Unknown device type: tpu")

    def test_touch_device_runtime_error_fails(self):
        torch = self._torch(RuntimeError("no TPU devices"),
                            RuntimeError("no TPU devices"))
        with mock.patch.dict(sys.modules, {
                "torch": torch,
                "torch_tpu": types.ModuleType("torch_tpu")
        }):
            result = image_probe.touch_device()
        self.assertEqual(result["status"], "fail")
        self.assertTrue(
            result["detail"].startswith("RuntimeError: no TPU devices"))
        self.assertIn("without importing torch_tpu", result["detail"])

    def _platform(self,
                  entry_points: list,
                  env: dict | None = None,
                  load: bool = False) -> dict:
        with mock.patch.object(importlib.metadata, "entry_points",
                               lambda group: entry_points), \
                mock.patch.dict(os.environ, env or {}, clear=False):
            if env is None:
                os.environ.pop("VLLM_PLUGINS", None)
            return image_probe.vllm_platform(load=load)

    def test_vllm_platform_registered(self):
        report = self._platform([TORCHTPU_ENTRY_POINT])
        self.assertTrue(report["registered"])
        self.assertEqual(
            report["entry_points"],
            ["torchtpu = vllm_torchtpu.platforms:register_tpu_platform"])
        self.assertNotIn("load", report)

    def test_vllm_platform_without_the_plugin(self):
        # vLLM would then pick its own TPU platform wherever libtpu imports.
        report = self._platform([])
        self.assertFalse(report["registered"])
        self.assertEqual(
            report["problems"],
            ["no vllm.platform_plugins entry point is in vllm_torchtpu"])

    def test_vllm_platform_left_out_by_vllm_plugins(self):
        report = self._platform([TORCHTPU_ENTRY_POINT],
                                env={"VLLM_PLUGINS": "lora_resolver"})
        self.assertFalse(report["registered"])
        self.assertIn("leaves out torchtpu", report["problems"][0])

    def test_vllm_platform_allowed_by_vllm_plugins(self):
        report = self._platform([TORCHTPU_ENTRY_POINT],
                                env={"VLLM_PLUGINS": "torchtpu"})
        self.assertTrue(report["registered"])

    def test_vllm_platform_with_a_rival_plugin(self):
        rival = importlib.metadata.EntryPoint("other",
                                              "other_platform:register",
                                              "vllm.platform_plugins")
        report = self._platform([TORCHTPU_ENTRY_POINT, rival])
        self.assertFalse(report["registered"])
        self.assertIn("other platform plugins would also load: ['other']",
                      report["problems"])

    def test_vllm_platform_load_under_touch_device(self):
        plugin = types.SimpleNamespace(
            name="torchtpu",
            value="vllm_torchtpu.platforms:register_tpu_platform",
            load=lambda: (lambda: "vllm_torchtpu.platforms.tpu_platform."
                          "TpuPlatform"))
        report = self._platform([plugin], load=True)
        self.assertTrue(report["registered"])
        self.assertEqual(report["load"]["platform"],
                         "vllm_torchtpu.platforms.tpu_platform.TpuPlatform")

    def test_vllm_platform_load_failure_is_not_registered(self):
        plugin = types.SimpleNamespace(
            name="torchtpu",
            value="vllm_torchtpu.platforms:register_tpu_platform",
            load=mock.Mock(side_effect=ImportError("No module named "
                                                   "'portpicker'")))
        report = self._platform([plugin], load=True)
        self.assertFalse(report["registered"])
        self.assertEqual(report["load"]["status"], "fail")
        self.assertIn("portpicker", report["load"]["detail"])

    def test_main_writes_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "image_probe.json")
            out = io.StringIO()
            with redirect_stdout(out):
                code = image_probe.main(["--output", path])
            self.assertEqual(code, 0)
            with open(path, encoding="utf-8") as f:
                written = json.load(f)
        self.assertEqual(written, json.loads(out.getvalue()))
        for key in ("distributions", "modules", "torch_tpu_present",
                    "sglang_present", "preflight_runnable",
                    "custom_fastapi_runnable", "vllm_platform", "dependencies",
                    "device"):
            self.assertIn(key, written)

    def test_main_touch_device_flag(self):
        with mock.patch.object(image_probe,
                               "touch_device",
                               return_value={
                                   "status": "ok",
                                   "detail": ""
                               }) as touch, redirect_stdout(io.StringIO()):
            image_probe.main(["--touch_device"])
        touch.assert_called_once()


class VllmStackTest(unittest.TestCase):

    def test_no_vllm_is_absent(self):
        report = image_probe.vllm_stack(TORCH_TPU_IMAGE, [])
        self.assertEqual(report["verdict"], "absent")
        self.assertEqual(report["problems"], [])

    def test_old_image_is_broken(self):
        report = image_probe.vllm_stack(OLD_SERVING_IMAGE, [])
        self.assertEqual(report["verdict"], "broken")
        self.assertIn("jax API that no longer exists", report["problems"][0])
        self.assertIn("vision models", report["problems"][0])

    def test_new_image_is_tested(self):
        report = image_probe.vllm_stack(NEW_SERVING_IMAGE, [])
        self.assertEqual(report["verdict"], "tested")
        self.assertEqual(report["image"],
                         "tpu-onboarding-torch_2.13.0-vllm_0.29.0-jax_0.11.2")
        self.assertEqual(report["problems"], [])

    def test_legacy_call_is_broken_whatever_the_versions(self):
        calls = ["pltpu.repeat in /root/torchtpu-vllm/vllm_torchtpu/a.py:3"]
        report = image_probe.vllm_stack(NEW_SERVING_IMAGE, calls)
        self.assertEqual(report["verdict"], "broken")
        self.assertIn("pltpu.repeat in", report["problems"][0])
        self.assertEqual(report["legacy_calls"], calls)

    def test_unknown_stack_is_untested(self):
        versions = dict(NEW_SERVING_IMAGE, jax="0.12.0", jaxlib="0.12.0")
        report = image_probe.vllm_stack(versions, [])
        self.assertEqual(report["verdict"], "untested")
        self.assertIn("jax 0.12.0", report["problems"][0])

    def test_version_prefix_and_local_label(self):
        self.assertTrue(image_probe._matches("2.13", "2.13.0+cpu"))
        self.assertTrue(image_probe._matches("2.13.0", "2.13.0+cpu"))
        self.assertFalse(image_probe._matches("2.1", "2.13.0"))


class PluginLegacyCallsTest(unittest.TestCase):

    def _plugin(self, files: dict[str, str]):
        """A vllm_torchtpu package on disk, found by spec as an editable
        install would be."""
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        package = os.path.join(root, "vllm_torchtpu")
        for rel, text in files.items():
            path = os.path.join(package, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        spec = types.SimpleNamespace(submodule_search_locations=[package])
        return package, mock.patch.object(image_probe.importlib.util,
                                          "find_spec",
                                          return_value=spec)

    def test_finds_pltpu_repeat(self):
        package, patch = self._plugin({
            "__init__.py":
            "",
            "ops/attention.py": ("from jax.experimental.pallas import tpu as"
                                 " pltpu\n\nx = pltpu.repeat(y, 2, axis=0)\n"),
        })
        with patch:
            calls = image_probe.plugin_legacy_calls()
        self.assertEqual(calls, [
            f"pltpu.repeat in {os.path.join(package, 'ops', 'attention.py')}:3"
        ])

    def test_current_plugin_makes_no_legacy_call(self):
        _, patch = self._plugin({
            "__init__.py": "",
            "ops/attention.py": "x = jnp.repeat(y, 2, axis=0)\n",
            "notes.txt": "pltpu.repeat was removed\n",
        })
        with patch:
            self.assertEqual(image_probe.plugin_legacy_calls(), [])

    def test_plugin_absent(self):
        with mock.patch.object(image_probe.importlib.util,
                               "find_spec",
                               return_value=None):
            self.assertEqual(image_probe.plugin_legacy_calls(), [])


class VllmTimeoutEnvsTest(unittest.TestCase):

    def _vllm(self, envs_py: str | None):
        """A vllm package on disk, with `envs_py` as its envs.py if given."""
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        package = os.path.join(root, "vllm")
        os.makedirs(package)
        if envs_py is not None:
            with open(os.path.join(package, "envs.py"), "w",
                      encoding="utf-8") as f:
                f.write(envs_py)
        spec = types.SimpleNamespace(submodule_search_locations=[package])
        return mock.patch.object(image_probe.importlib.util,
                                 "find_spec",
                                 return_value=spec)

    def test_all_read(self):
        text = "".join(
            f'    "{name}":\n        lambda: int(os.getenv("{name}",'
            ' "600")),\n' for name in image_probe.VLLM_TIMEOUT_ENVS)
        with self._vllm(text):
            report = image_probe.vllm_timeout_envs()
        self.assertEqual(report,
                         dict.fromkeys(image_probe.VLLM_TIMEOUT_ENVS, True))

    def test_renamed_one_reads_false(self):
        text = ('"VLLM_ENGINE_READY_TIMEOUT_S": ...\n'
                '"VLLM_ENGINE_ITERATION_TIMEOUT_S": ...\n'
                '"VLLM_EXECUTE_MODEL_TIMEOUT_S": ...\n')
        with self._vllm(text):
            report = image_probe.vllm_timeout_envs()
        self.assertFalse(report["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"])
        self.assertTrue(report["VLLM_ENGINE_READY_TIMEOUT_S"])

    def test_no_envs_py(self):
        with self._vllm(None):
            report = image_probe.vllm_timeout_envs()
        self.assertEqual(report,
                         dict.fromkeys(image_probe.VLLM_TIMEOUT_ENVS, None))

    def test_vllm_absent(self):
        with mock.patch.object(image_probe.importlib.util,
                               "find_spec",
                               return_value=None):
            report = image_probe.vllm_timeout_envs()
        self.assertEqual(report,
                         dict.fromkeys(image_probe.VLLM_TIMEOUT_ENVS, None))

    def test_spec_without_locations(self):
        # The probe's own tests stand in an object() for a found spec.
        with mock.patch.object(image_probe.importlib.util,
                               "find_spec",
                               return_value=object()):
            report = image_probe.vllm_timeout_envs()
        self.assertEqual(report,
                         dict.fromkeys(image_probe.VLLM_TIMEOUT_ENVS, None))


class PipCheckTest(unittest.TestCase):

    def _check(self, returncode: int, stdout: str = "", stderr: str = ""):
        done = subprocess.CompletedProcess([], returncode, stdout, stderr)
        with mock.patch.object(image_probe.subprocess,
                               "run",
                               return_value=done):
            return image_probe.pip_check()

    def test_clean(self):
        report = self._check(0, "No broken requirements found.\n")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["stack_conflicts"], [])

    def test_conflicts_split_by_who_declares_them(self):
        report = self._check(
            1, "vllm 0.22.1 has requirement torch==2.7.0, but you have"
            " torch 2.11.0.\n"
            "vllm-torchtpu 0.0.0 requires portpicker, which is not"
            " installed.\n"
            "some-tool 1.0 requires click, which is not installed.\n")
        self.assertEqual(report["status"], "conflicts")
        self.assertEqual(len(report["stack_conflicts"]), 2)
        self.assertTrue(
            report["stack_conflicts"][1].startswith("vllm-torchtpu"))
        self.assertEqual(report["other_conflicts"], 1)

    def test_no_pip(self):
        report = self._check(1, "", "/usr/bin/python3: No module named pip\n")
        self.assertEqual(report["status"], "unavailable")
        self.assertIn("No module named pip", report["detail"])

    def test_timeout(self):
        with mock.patch.object(image_probe.subprocess,
                               "run",
                               side_effect=subprocess.TimeoutExpired(
                                   "pip", 120)):
            report = image_probe.pip_check()
        self.assertEqual(report["status"], "unavailable")


class VllmPatchesRegistryIntegrationTest(unittest.TestCase):
    """Tests for registry.json integration and closest-version patch selection."""

    def test_loads_real_registry_and_integrates_with_vllm_stack(self):
        patch_dir, registry = image_probe.load_patch_registry()
        self.assertIsNotNone(patch_dir)
        self.assertIn("openai/gpt-oss-20b", registry.get("models", {}))
        self.assertTrue(
            any(s["versions"].get("vllm") == "0.29.0"
                for s in image_probe.registry_tested_stacks(registry)))
        custom_reg = {
            "models": {
                "m": {
                    "patch": "f.py",
                    "validated": {
                        "vllm": "0.28.0",
                        "torch": "2.12.0",
                        "jax": "0.11.0"
                    }
                }
            }
        }
        report = image_probe.vllm_stack(
            {
                "torch": "2.12.0+cpu",
                "vllm": "0.28.0",
                "jax": "0.11.0"
            }, [],
            registry=custom_reg)
        self.assertEqual(report["verdict"], "tested")

    def test_select_patch_candidates_closest_before_and_after(self):
        _, registry = image_probe.load_patch_registry()
        entry = registry["models"]["openai/gpt-oss-20b"]
        cases = [
            ("0.29.0", "exact", [("0.29.0", "exact")]),
            ("0.22.1+tpu", "exact", [("0.22.1", "exact")]),
            ("0.30.1rc1.dev385+g8039d3cbf", "newer", [("0.29.0", "before")]),
            ("0.20.0", "older", [("0.22.1", "after")]),
            ("0.24.0", "between", [("0.22.1", "before"), ("0.29.0", "after")]),
            ("0.27.0", "between", [("0.29.0", "after"), ("0.22.1", "before")]),
        ]
        for ver, exp_match, exp_cands in cases:
            with self.subTest(ver=ver):
                match, cands = image_probe.select_patch_candidates(ver, entry)
                self.assertEqual(match, exp_match)
                self.assertEqual([(c["validated_vllm"], c["position"])
                                  for c in cands], exp_cands)

    def test_dry_run_diff_and_python_patch_without_writing_files(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        pkg = os.path.join(root, "vllm_torchtpu")
        os.makedirs(pkg)
        init_py = os.path.join(pkg, "__init__.py")
        with open(init_py, "w", encoding="utf-8") as f:
            f.write("X = 1\n")

        diff_path = os.path.join(root, "fix.patch")
        with open(diff_path, "w", encoding="utf-8") as f:
            f.write("--- a/vllm_torchtpu/__init__.py\n"
                    "+++ b/vllm_torchtpu/__init__.py\n"
                    "@@ -1 +1,2 @@\n X = 1\n+Y = 2\n")
        self.assertEqual(
            image_probe.dry_run_patch(diff_path, root)["status"], "applies")
        with open(init_py, encoding="utf-8") as f:
            self.assertEqual(f.read(), "X = 1\n")

        py_patch = os.path.join(root, "fix.py")
        with open(py_patch, "w", encoding="utf-8") as f:
            f.write(
                "import os, pathlib\n"
                "class Sources:\n"
                "    def __init__(self, r):\n"
                "        self.p = os.path.join(r, 'vllm_torchtpu', '__init__.py')\n"
                "        self.orig = self.cur = pathlib.Path(self.p).read_text('utf-8')\n"
                "    def changed(self): return [self.p] if self.cur != self.orig else []\n"
                "def plan(s):\n"
                "    if 'X = 2' in s.cur: return ['ok'], []\n"
                "    if 'X = 1' not in s.cur: return [], ['missing']\n"
                "    s.cur = s.cur.replace('X = 1', 'X = 2'); return ['ok'], []\n"
            )
        self.assertEqual(
            image_probe.dry_run_patch(py_patch, root)["status"], "applies")
        with open(init_py, encoding="utf-8") as f:
            self.assertEqual(f.read(), "X = 1\n")

        with open(init_py, "w", encoding="utf-8") as f:
            f.write("X = 2\n")
        self.assertEqual(
            image_probe.dry_run_patch(diff_path, root)["status"],
            "does_not_apply")
        self.assertEqual(
            image_probe.dry_run_patch(py_patch, root)["status"],
            "already_applied")

    def test_vllm_patches_needs_user_confirmation_when_absent_or_unapplicable(
            self):
        absent = image_probe.vllm_patches(
            TORCH_TPU_IMAGE)["openai/gpt-oss-20b"]
        self.assertEqual(absent["version_match"], "vllm_absent")
        self.assertEqual(absent["action"], "needs_user_confirmation")

        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        pkg = os.path.join(root, "vllm_torchtpu")
        os.makedirs(pkg)
        with open(os.path.join(pkg, "__init__.py"), "w",
                  encoding="utf-8") as f:
            f.write("")
        spec = types.SimpleNamespace(submodule_search_locations=[pkg])
        with mock.patch.object(image_probe.importlib.util,
                               "find_spec",
                               return_value=spec):
            rep = image_probe.vllm_patches(
                {"vllm": "0.30.1rc1.dev385+g8039d3cbf"})["openai/gpt-oss-20b"]
        self.assertEqual(rep["version_match"], "newer")
        self.assertEqual([c["validated_vllm"] for c in rep["candidates"]],
                         ["0.29.0"])
        self.assertEqual(rep["applicable_candidates"], 0)
        self.assertEqual(rep["action"], "needs_user_confirmation")


if __name__ == "__main__":
    unittest.main()
