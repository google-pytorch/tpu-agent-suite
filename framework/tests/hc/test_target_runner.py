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
"""Unit tests for `framework.hc.target_runner`."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from framework.hc import target_runner

Path = pathlib.Path


class TestTargetRunner(unittest.TestCase):
    """Tests for `target_runner` serving, patching, and lm-eval reduction."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workdir = Path(self._tmp.name)
        orig_cwd = Path.cwd()
        self.addCleanup(lambda: os.chdir(orig_cwd))

    def test_serve_vllm_and_env_validation(self) -> None:
        (self.workdir / "__hc_context.json").write_text(
            json.dumps({
                "target_model":
                "meta-llama/Llama-3.3-70B-Instruct",
                "engine":
                "vllm",
                "base_server_args":
                "--tensor-parallel-size 8 --enforce-eager --port=8000",
            }),
            encoding="utf-8",
        )
        (self.workdir / "__flags.json").write_text(
            json.dumps({
                "env": {
                    "VLLM_ASYNC_SCHEDULING": 1,
                    "ENABLE_CUSTOM_KERNEL": True,
                },
                "server_args": ["--max-num-seqs", "64"],
            }),
            encoding="utf-8",
        )
        (self.workdir / "candidate.patch").write_text(
            "--- a/x.py\n+++ b/x.py\n", encoding="utf-8")
        stub_serve = self.workdir / "serve_vllm.sh"
        stub_serve.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

        captured: dict[str, object] = {}
        with mock.patch(
                "os.execvpe",
                side_effect=lambda f, a, e: captured.update(
                    file=f, argv=a, env=e),
        ):
            target_runner.main([
                "--workdir",
                str(self.workdir),
                "serve-vllm",
                "--",
                "--gpu-memory-utilization",
                "0.92",
                "--port",
                "8081",
            ])

        self.assertEqual(
            captured["argv"],
            [
                "bash",
                str(stub_serve.resolve()),
                "meta-llama/Llama-3.3-70B-Instruct",
                "--tensor-parallel-size",
                "8",
                "--enforce-eager",
                "--max-num-seqs",
                "64",
                "--gpu-memory-utilization",
                "0.92",
                "--port",
                "8081",
                "--served-model-name",
                "meta-llama/Llama-3.3-70B-Instruct",
            ],
        )
        env = captured["env"]
        assert isinstance(env, dict)
        self.assertEqual(
            (env["VLLM_ASYNC_SCHEDULING"], env["ENABLE_CUSTOM_KERNEL"]),
            ("1", "true"))
        self.assertEqual(
            (self.workdir / "vllm_patches.txt").read_text(encoding="utf-8"),
            "candidate.patch\n")

        self.assertEqual(target_runner._launcher(self.workdir, "vllm"),
                         stub_serve)
        with self.assertRaises(FileNotFoundError):
            target_runner._launcher(self.workdir, "sglang")

        (self.workdir / "__flags.json").write_text(json.dumps(
            {"env": {
                "BAD-NAME": "1"
            }}),
                                                   encoding="utf-8")
        with self.assertRaises(ValueError):
            target_runner.serve(self.workdir, "vllm", [])

        # Whitespace/empty patch is ignored; missing __hc_context.json fails fast.
        (self.workdir / "vllm_patches.txt").unlink()
        (self.workdir / "candidate.patch").write_text("   \n",
                                                      encoding="utf-8")
        (self.workdir / "__flags.json").unlink()
        target_runner.serving_args(self.workdir, engine="vllm", cli_args=[])
        self.assertFalse((self.workdir / "vllm_patches.txt").exists())
        target_runner.apply_patch(self.workdir)

        (self.workdir / "__flags.json").write_text("{bad", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "__flags.json"):
            target_runner.serving_args(self.workdir,
                                       engine="vllm",
                                       cli_args=[])
        (self.workdir / "__flags.json").unlink()
        (self.workdir / "__hc_context.json").unlink()
        with self.assertRaises(FileNotFoundError):
            target_runner.serving_args(self.workdir,
                                       engine="vllm",
                                       cli_args=[])

    def test_sglang_patch_and_package_discovery(self) -> None:
        fake_site = self.workdir / "fake_site"
        fake_sglang = fake_site / "sglang"
        fake_sglang.mkdir(parents=True)
        (fake_sglang / "__init__.py").write_text("", encoding="utf-8")
        worker = fake_sglang / "worker.py"
        worker.write_text("X = 1\n", encoding="utf-8")
        (self.workdir / "candidate.patch").write_text(
            "--- a/sglang/worker.py\n+++ b/sglang/worker.py\n@@ -1 +1 @@\n-X = 1\n+X = 2\n",
            encoding="utf-8",
        )
        target_runner.apply_patch(self.workdir, target_dir=fake_site)
        target_runner.apply_patch(self.workdir, target_dir=fake_site)
        self.assertEqual(worker.read_text(encoding="utf-8"), "X = 2\n")

        worker.write_text("X = 1\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HC_PATCH_TARGET_DIR", None)
            with mock.patch("importlib.util.find_spec", return_value=None):
                with self.assertRaises(RuntimeError):
                    target_runner.apply_patch(self.workdir)
            sys.path.insert(0, str(fake_site))
            try:
                target_runner.apply_patch(self.workdir)
                target_runner.apply_patch(self.workdir)
            finally:
                sys.path.remove(str(fake_site))
        self.assertEqual(worker.read_text(encoding="utf-8"), "X = 2\n")

    def test_eval_vllm_python_resolution_and_lm_eval_summary(self) -> None:
        (self.workdir / "__hc_context.json").write_text(
            json.dumps({
                "target_model": "meta-llama/Llama-3.3-70B-Instruct",
                "engine": "vllm",
                "accuracy_metric": "exact_match,custom-extract",
                "base_eval_args": "--num_fewshot 5",
            }),
            encoding="utf-8",
        )
        # Candidate __flags.json tampering keys are ignored for model and eval config.
        (self.workdir / "__flags.json").write_text(
            json.dumps({
                "model": "tiny-fake-model",
                "eval_tasks": "trivial_task",
                "eval_limit": "1",
                "base_eval_args": ["--limit", "1"],
                "eval_args": ["--limit", "1"],
            }),
            encoding="utf-8",
        )
        (self.workdir / "target_runner.py").write_text("# staged\n",
                                                       encoding="utf-8")
        stub_eval = self.workdir / "eval_recipe.sh"
        stub_eval.write_text(
            "#!/usr/bin/env bash\n"
            "mkdir -p eval_out/model\n"
            "cat > eval_out/model/results_2026.json <<'EOF'\n"
            '{"groups": {"mmlu_pro": {"exact_match,custom-extract": 0.825}}, '
            '"results": {"mmlu_pro_biology": {"exact_match,custom-extract": 0.95}}}\n'
            "EOF\n",
            encoding="utf-8",
        )
        stub_eval.chmod(0o755)

        # Shell value in TPU_AGENT_SUITE_PYTHON falls back to python3.
        with mock.patch.dict(os.environ,
                             {"TPU_AGENT_SUITE_PYTHON": "/bin/bash"}):
            self.assertEqual(
                target_runner.main([
                    "--workdir",
                    str(self.workdir),
                    "eval-vllm",
                    "SERVER_PORT=8123",
                ]),
                0,
            )
        summary = json.loads(
            (self.workdir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["scores"],
                         {"exact_match,custom-extract": 0.825})
        self.assertEqual(
            (self.workdir / "vllm_server_cmd.txt").read_text(encoding="utf-8"),
            "exec python3 target_runner.py serve-vllm --port 8123\n",
        )
        self.assertIn("--num_fewshot 5",
                      (self.workdir /
                       "eval_cmd.txt").read_text(encoding="utf-8"))
        self.assertNotIn("--limit 1",
                         (self.workdir /
                          "eval_cmd.txt").read_text(encoding="utf-8"))
        eval_argv = target_runner.eval_args(self.workdir, [])
        self.assertIn("EVAL_TASKS=mmlu_pro", eval_argv)
        self.assertIn("EVAL_LIMIT=100", eval_argv)

        # Missing protocol metric in results_*.json marks summary failed.
        (self.workdir / "__hc_context.json").write_text(
            json.dumps({
                "target_model": "meta-llama/Llama-3.3-70B-Instruct",
                "engine": "vllm",
                "accuracy_metric": "nonexistent_metric",
            }),
            encoding="utf-8",
        )
        missing = target_runner.summarize(self.workdir, 0)
        self.assertEqual(missing["status"], "failed")
        self.assertEqual(missing["scores"], {})

        (self.workdir / "__flags.json").write_text(
            json.dumps({"server_args": ["--model", "tiny"]}),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            target_runner.serving_args(self.workdir,
                                       engine="vllm",
                                       cli_args=[])

        # Leading tilde in TPU_AGENT_SUITE_PYTHON is expanded before quoting.
        with mock.patch.dict(os.environ,
                             {"TPU_AGENT_SUITE_PYTHON": "~/venv/bin/python3"}):
            self.assertTrue(
                target_runner._resolve_python().endswith("/venv/bin/python3"))
            self.assertFalse(target_runner._resolve_python().startswith("~"))

        bad_res = self.workdir / "eval_out" / "model" / "results_9999.json"
        bad_res.write_text("{corrupt", encoding="utf-8")
        with self.assertRaises(ValueError):
            target_runner._extract_metric(bad_res,
                                          "exact_match,custom-extract")
        self.assertEqual(
            target_runner.summarize(self.workdir, 0)["status"], "failed")

        # Multi-host worker pods exit with eval_recipe.sh's rc without requiring results_*.json,
        # while leader host 0 still writes summary.json.
        shutil.rmtree(self.workdir / "eval_out")
        (self.workdir / "summary.json").unlink()
        (self.workdir / "__flags.json").unlink()
        stub_eval.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        with mock.patch.dict(
                os.environ,
            {
                "TPU_NUM_HOSTS": "2",
                "JOB_COMPLETION_INDEX": "1",
            },
        ):
            os.environ.pop("TPU_WORKER_ID", None)
            self.assertEqual(target_runner.run_eval(self.workdir, []), 0)
        self.assertFalse((self.workdir / "summary.json").exists())

        with mock.patch.dict(
                os.environ,
            {
                "TPU_NUM_HOSTS": "2",
                "TPU_WORKER_ID": "0",
            },
        ):
            self.assertEqual(target_runner.run_eval(self.workdir, []), 1)
        self.assertTrue((self.workdir / "summary.json").is_file())

    def test_run_microbenchmark_forward_run_and_runner_mode_dispatch(
            self) -> None:
        (self.workdir / "plan.steps.json").write_text("{}", encoding="utf-8")
        (self.workdir / "forward_run.py").write_text(
            "import json, pathlib, sys\n"
            "out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "pt = [{'point': {'input': 1024, 'output': 1024, 'concurrency': 16}, "
            "'twin_model': {'output_tok_per_s': 980.0}}]\n"
            "(out / 'result.json').write_text(json.dumps({'repeats': [pt, pt], 'points': pt}))\n",
            encoding="utf-8",
        )
        (self.workdir / "__hc_context.json").write_text(
            json.dumps({
                "runner_mode": "run-microbenchmark",
                "engine": "vllm",
                "image": "img@sha256:cafe",
                "microbenchmark_target": "plan.steps.json",
                "min_replicates": 2,
            }),
            encoding="utf-8",
        )
        self.assertEqual(target_runner.main(["--workdir",
                                             str(self.workdir)]), 0)
        summary = json.loads(
            (self.workdir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(
            [
                r["output_tokens_per_s"]
                for r in summary["cells"][0]["replicates"]
            ],
            [980.0, 980.0],
        )
        self.assertEqual(summary["environment"]["image_digest"], "sha256:cafe")


if __name__ == "__main__":
    unittest.main()
