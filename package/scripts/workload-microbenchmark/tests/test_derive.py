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
"""CPU tests for core/jobdata.py, core/derive.py, core/cells.py and core/deploy.py: serve args, overrides, reference derivations, recipes and env audit.

Run from scripts/workload-microbenchmark/: `python3 -m unittest discover tests`.
"""

import glob
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

from core import cells, derive, jobdata, jobdata_agentx, project  # noqa: E402
from core.deploy import Deploy, apply_overrides  # noqa: E402


class ServeArgs(unittest.TestCase):

    def test_array_form_with_appended_flag_and_redirect(self):
        script = '''VLLM_ARGS=(
  vllm serve "$MODEL_PATH"
  --tensor-parallel-size=16
  --block-size=256
  --async-scheduling
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  "${OFFLOAD_ARGS[@]}"
)
"${VLLM_ARGS[@]}" --no-dcp-q-replicate 2>&1 | tee "$LOG"
'''
        args, model = jobdata_agentx._parse_serve_args(script)
        self.assertEqual(model, ["$MODEL_PATH"])
        self.assertEqual(args["tensor-parallel-size"], "16")
        self.assertIs(args["async-scheduling"], True)
        self.assertIs(args["no-dcp-q-replicate"], True)  # not "2>&1"
        self.assertEqual(
            json.loads(args["speculative-config"])["method"], "mtp")

    def test_plain_line_form(self):
        args, _ = jobdata_agentx._parse_serve_args(
            "vllm serve m --max-num-seqs 8 --enable-prefix-caching > log")
        self.assertEqual(args, {
            "max-num-seqs": "8",
            "enable-prefix-caching": True
        })


class Recompute(unittest.TestCase):
    D = Deploy(block_size=256,
               dcp=16,
               prefix_caching=True,
               spec_method="mtp",
               spec_tokens=3)

    def _req(self, conv, isl, osl, phase="profiling"):
        return {"conv": conv, "isl": isl, "osl": osl, "phase": phase}

    def test_floor_is_one_block_plus_misalignment_with_eagle_drop(self):
        # prior turn 100,000 + 200 out; next prompt appends 500 tokens
        reqs = [
            self._req("a", 100_000, 200, "warmup"),
            self._req("a", 100_700, 50)
        ]
        r = cells.recompute(reqs, self.D)[0]
        G = 4096
        self.assertEqual(r["hit"], (100_200 // G) * G -
                         G)  # full blocks of the prefix, minus one
        self.assertTrue(G < r["computed"] <= 2 * G + 700)

    def test_no_eagle_no_drop(self):
        d = Deploy(block_size=256, dcp=16, prefix_caching=True)
        reqs = [
            self._req("a", 100_000, 200, "warmup"),
            self._req("a", 100_700, 50)
        ]
        self.assertEqual(
            cells.recompute(reqs, d)[0]["hit"], (100_200 // 4096) * 4096)

    def test_cold_request_computes_everything(self):
        r = cells.recompute([self._req("b", 50_000, 10)], self.D)[0]
        self.assertEqual(r["computed"], 50_000)

    def test_block_granularity_scales_with_dcp(self):
        self.assertEqual(self.D.cache_block_tokens, 4096)
        self.assertEqual(
            apply_overrides(self.D, {
                "dcp": "8"
            }).cache_block_tokens, 2048)


class Overrides(unittest.TestCase):

    def test_unknown_key_is_an_error(self):
        with self.assertRaises(ValueError):
            apply_overrides(Deploy(), {"dcpp": "8"})

    def test_env_overrides_do_not_touch_deploy(self):
        self.assertEqual(apply_overrides(Deploy(dcp=2), {"env.X": "1"}),
                         Deploy(dcp=2))


def _reference():
    p = sorted(
        glob.glob(
            os.path.join(SCRIPTS, "GLM-5.2-FP8", "config", "*.source.json")))
    return json.load(open(p[0])) if p else None


@unittest.skipIf(_reference() is None, "no GLM-5.2-FP8 reference snapshot")
class GlmReference(unittest.TestCase):
    snap = _reference()

    def _twin(self, world=8):
        cfg = derive.derive(self.snap, "GLM-5.2-FP8")
        return cfg, project.project(cfg, self.snap,
                                    derive.load_model("GLM-5.2-FP8"), world)

    def test_mirror_mode_scales_nothing_and_spans_hosts(self):
        """twin_world == the served world: only the layer count shrinks; nothing deviates."""
        cfg, tw = self._twin(16)
        f, hf, sv = cfg["facts"], tw["twin"]["hf_overrides"], tw["twin"][
            "server_args"]
        self.assertEqual(tw["mode"], "mirror")
        self.assertEqual((tw["twin"]["multihost"], tw["twin"]["hosts"]),
                         (True, 2))
        self.assertEqual(tw["context_scale"], 1.0)
        self.assertEqual((hf["num_attention_heads"], hf["n_routed_experts"],
                          hf["num_experts_per_tok"], hf["intermediate_size"]),
                         (f["heads"], f["experts"], f["topk"], f["dense_ffn"]))
        self.assertEqual(
            (sv["tensor-parallel-size"], sv["decode-context-parallel-size"]),
            (cfg["deploy"]["tp"], cfg["deploy"]["dcp"]))
        self.assertEqual(hf["num_hidden_layers"], 3)
        self.assertEqual([
            r["per_rank"]
            for r in tw["projection"] if r["status"] == "deviation"
        ], [])
        _, sc = self._twin(8)
        self.assertEqual((sc["mode"], sc["twin"]["multihost"]),
                         ("scaled", False))

    def test_twin_rule_preserves_per_rank_shapes(self):
        """global' / degree' == global / degree for every scaled tensor."""
        cfg, tw = self._twin()
        d, f, t = cfg["deploy"], cfg["facts"], tw
        hf, sv = tw["twin"]["hf_overrides"], tw["twin"]["server_args"]
        ep_real, ep_twin = d["tp"] * d["dp"], sv["tensor-parallel-size"] * sv[
            "data-parallel-size"]
        self.assertEqual(
            hf["num_attention_heads"] // sv["tensor-parallel-size"],
            f["heads"] // d["tp"])
        self.assertEqual(hf["n_routed_experts"] // ep_twin,
                         f["experts"] // ep_real)
        self.assertEqual(hf["num_experts_per_tok"] / ep_twin,
                         f["topk"] / ep_real)
        self.assertEqual(hf["intermediate_size"] // sv["tensor-parallel-size"],
                         f["dense_ffn"] // d["tp"])
        # the per-rank KV page is the served one (page-granular work matches), and so is the
        # per-rank KV pool; the prefix-cache block (block x DCP) is not used by forward_run
        self.assertEqual(sv["block-size"], d["block_size"])
        pool = {
            r["per_rank"]: r
            for r in tw["projection"]
        }["kv_pool_tokens_per_rank"]
        self.assertEqual(pool["real"], pool["twin"])
        self.assertEqual(t["layers"], ["dense_idx", "moe_idx", "moe"])
        # every row the rule controls matches; only group-size rows and the shared expert deviate
        dev = {
            r["per_rank"]
            for r in tw["projection"] if r["status"] == "deviation"
        }
        self.assertEqual(
            dev, {
                "heads_in_attention", "indexer_merge_width",
                "shared_expert_ffn_per_rank"
            })
        self.assertEqual(hf["index_topk_pattern"], ["F", "F", "S"])

    def test_fp8_skip_list_follows_layer_kind(self):
        """Routers, indexer projections and MTP projections keep their served dtype in the twin."""
        q = self._twin()[1]["twin"]["hf_overrides"]["quantization_config"]
        skip = set(q["modules_to_not_convert"])
        self.assertIn("model.layers.1.mlp.gate",
                      skip)  # twin layer 1 = MoE + indexer
        self.assertIn("model.layers.1.self_attn.indexers_proj", skip)
        self.assertNotIn("model.layers.2.self_attn.indexers_proj",
                         skip)  # reuse layer
        self.assertIn("model.layers.3.eh_proj",
                      skip)  # MTP layer is id 3 in the twin
        self.assertFalse(any(m.startswith("model.layers.4") for m in skip))

    def test_counts_reproduce_the_server(self):
        c = derive.derive(self.snap, "GLM-5.2-FP8")["cells"]
        self.assertAlmostEqual(c["recompute"]["model_over_server"],
                               1.0,
                               delta=0.03)
        self.assertAlmostEqual(sum(p["weight"] for p in c["prefill_cells"]),
                               1.0,
                               delta=0.01)
        self.assertAlmostEqual(sum(q["weight"] for q in c["decode_cells"]),
                               1.0,
                               delta=0.01)

    def test_dcp1_is_refused_with_alternatives(self):
        cfg = derive.derive(self.snap, "GLM-5.2-FP8", {"dcp": "1"})
        self.assertFalse(cfg["feasibility"]["feasible"])
        self.assertNotIn("cells", cfg)
        self.assertGreater(
            cfg["feasibility"]["kv"]["min_dcp_for_recorded_peak"], 1)

    def test_scheduling_override_is_labelled_L2(self):
        cfg = derive.derive(self.snap, "GLM-5.2-FP8", {"mnbt": "8192"})
        self.assertTrue(cfg["decidable_at"].startswith("L1 per-layer"))
        self.assertEqual(
            derive.derive(
                self.snap, "GLM-5.2-FP8",
                {"env.TPU_STREAMIDX_CHUNK_TOKENS": "128"})["decidable_at"],
            "L1")


class RecipeEval(unittest.TestCase):
    """The recipe script is evaluated as bash would, and nothing in it actually runs."""
    SCRIPT = r'''set -euo pipefail
# Wait for vllm serve to come up
[ -f "/nonexistent-pvc/config.json" ] || { echo "missing"; exit 1; }
gsutil -m rsync -r gs://x /root/ckpt || exit 1
/bin/touch "$MARK"
echo hi > "$MARK.redirect"
if [ "${USE_MOE_FUSED_EP_KERNEL:-0}" = "1" ]; then
  export TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER="${TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER:-1}"
fi
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+$LIBTPU_INIT_ARGS }--flag=$(( 2 * 4096 ))"
MNBT="16384"
if [ "$JOBSET_JOB_INDEX" = "0" ]; then
  until ray status 2>/dev/null | grep -q "16.0 TPU"; do sleep 5; done
  vllm serve \
    /root/ckpt \
    --tensor-parallel-size=8 \
    --max-num-batched-tokens=${MNBT} \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' 2>&1 | tee "$MARK.log" &
else
  ray start --block
fi
'''

    def _run(self, env):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            mark = os.path.join(tmp, "ran")
            ev = jobdata_agentx._eval_script(self.SCRIPT, {
                "JOBSET_JOB_INDEX": "0",
                "MARK": mark,
                **env
            })
            self.assertFalse(os.path.exists(mark),
                             "an absolute-path command ran")
            self.assertFalse(os.path.exists(mark + ".redirect"),
                             "a file redirect was written")
            self.assertFalse(os.path.exists(mark + ".log"))
            return ev

    def test_args_expanded_and_env_from_branches(self):
        argv, env = self._run({
            "USE_MOE_FUSED_EP_KERNEL": "1",
            "LIBTPU_INIT_ARGS": "--base"
        })
        args, model = jobdata_agentx._args_from_tokens(argv)
        self.assertEqual(model, ["/root/ckpt"])
        self.assertEqual(args["max-num-batched-tokens"],
                         "16384")  # shell var expanded
        self.assertEqual(
            json.loads(args["speculative-config"])["method"], "mtp")
        self.assertEqual(env["TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER"],
                         "1")  # branch taken
        self.assertEqual(env["LIBTPU_INIT_ARGS"], "--base --flag=8192")

    def test_branch_not_taken(self):
        _, env = self._run({"USE_MOE_FUSED_EP_KERNEL": "0"})
        self.assertNotIn("TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER", env)
        self.assertEqual(env["LIBTPU_INIT_ARGS"], "--flag=8192")

    def test_never_reaches_vllm(self):
        self.assertIsNone(
            jobdata_agentx._eval_script(
                "python3 -m vllm.entrypoints.api_server --x 1\n", {}))

    def test_fallback_parser_and_comment_guard(self):
        self.assertFalse(jobdata_agentx._is_server_script("# vllm serve /x\n"))
        self.assertTrue(jobdata_agentx._is_server_script(self.SCRIPT))
        env = {"VLLM_SERVER_CMD": "cd /ws; vllm serve /m --port=8000"}
        self.assertFalse(
            jobdata_agentx._is_server_script('echo "${VLLM_SERVER_CMD}"\n',
                                             env))
        self.assertTrue(
            jobdata_agentx._is_server_script('eval "${VLLM_SERVER_CMD}"\n',
                                             env))
        args, pos = jobdata_agentx._parse_serve_args(env["VLLM_SERVER_CMD"])
        self.assertEqual((pos, args["port"]), (["/m"], "8000"))
        args, pos = jobdata_agentx._parse_serve_args(self.SCRIPT)
        self.assertEqual((pos, args["tensor-parallel-size"]),
                         (["/root/ckpt"], "8"))
        # In bash, a `#` comment inside a `\`-continuation swallows its own trailing `\`
        # and terminates the command, so `--tensor-parallel-size=8` is NOT passed to `vllm`.
        broken = ("vllm serve /ckpt \\\n"
                  "  # --enable-prefix-caching \\\n"
                  "  --tensor-parallel-size=8\n")
        argv, _ = jobdata_agentx._eval_script(broken, {})
        self.assertEqual(jobdata_agentx._args_from_tokens(argv),
                         ({}, ["/ckpt"]))
        self.assertEqual(jobdata_agentx._parse_serve_args(broken),
                         ({}, ["/ckpt"]))


class RecipeSource(unittest.TestCase):
    """`jobdata_agentx.read_recipe` picks the server's source in one place: an AgentX job's
    `recipe.yaml`, else the vLLM log of a server this package started (`jobdata.read_recipe`,
    which reads only the log)."""

    MANIFEST = {
        "metadata": {
            "name": "j-agentx"
        },
        "spec": {
            "replicatedJobs": [{
                "name": "server",
                "replicas": 1,
                "template": {
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [{
                                    "name":
                                    "vllm",
                                    "image":
                                    "repo/vllm@sha256:abc",
                                    "args": [
                                        "vllm serve /ckpt --tensor-parallel-size 4"
                                    ],
                                    "env": [{
                                        "name": "X",
                                        "value": "1"
                                    }],
                                }]
                            }
                        }
                    }
                }
            }]
        },
    }
    LOG = "INFO non-default args: {'model': '/m', 'tensor_parallel_size': 8}\n"

    def _job(self, tmp, manifest=False, log=False):
        job = os.path.join(tmp, "j")
        os.makedirs(os.path.join(job, "logs"))
        if manifest:
            with open(os.path.join(job, "recipe.yaml"), "w") as f:
                json.dump(self.MANIFEST, f)  # JSON is YAML
        if log:
            with open(os.path.join(job, "logs", "server.log"), "w") as f:
                f.write(self.LOG)
        return job

    def test_manifest_is_read_by_agentx_reader(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            # a vLLM log beside it does not matter: the manifest wins
            rec = jobdata_agentx.read_recipe(
                self._job(tmp, manifest=True, log=True))
        self.assertEqual(rec["job_id"], "j-agentx")
        self.assertEqual(rec["serve_parse"], "bash-eval")
        self.assertEqual(rec["image"], "repo/vllm@sha256:abc")
        self.assertEqual(rec["model_path"], "/ckpt")
        self.assertEqual(rec["server_args"]["tensor-parallel-size"], "4")
        self.assertEqual(rec["env_container"], {"X": "1"})

    def test_no_manifest_reads_vllm_log(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            rec = jobdata_agentx.read_recipe(self._job(tmp, log=True))
        self.assertEqual(rec["serve_parse"], "vllm-log")
        self.assertEqual(rec["server_args"]["tensor-parallel-size"], 8)

    def test_base_reader_ignores_manifest(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            rec = jobdata.read_recipe(self._job(tmp, manifest=True, log=True))
        self.assertEqual(rec["serve_parse"], "vllm-log")

    def test_neither_is_a_clear_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                    FileNotFoundError, "no recipe.yaml .* no vLLM server log"):
                jobdata_agentx.read_recipe(self._job(tmp))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError,
                                        "no vLLM server log"):
                jobdata.read_recipe(self._job(tmp))


class VllmServerLog(unittest.TestCase):
    """A server with no recipe.yaml is read from vLLM's own log: the `non-default args` line for
    the recipe, the engine config line for prefix caching, the checkpoint's config.json from the
    model path, and the page size from vllm_torchtpu's "Using KV cache block size" line. vLLM
    logs no env: the model's TWIN_ENV supplies what it needs."""

    ARGS = (
        "(APIServer pid=1) INFO 10-14 20:16:02 [api_utils.py:286] non-default args: "
        "{'model_tag': '%s', 'host': '0.0.0.0', 'model': '%s', 'max_model_len': 8192, "
        "'served_model_name': ['openai/gpt-oss-20b'], 'attention_backend': 'CUSTOM', "
        "'tensor_parallel_size': 2, 'max_num_batched_tokens': 4096, "
        "'profiler_config': ProfilerConfig(profiler='torch', torch_profiler_dir='/p')}"
    )

    def _job(self, tmp):
        model = os.path.join(tmp, "ckpt")
        os.makedirs(model)
        with open(os.path.join(model, "config.json"), "w") as f:
            json.dump({"num_hidden_layers": 24}, f)
        job = os.path.join(tmp, "j-srv")
        os.makedirs(os.path.join(job, "logs"))
        lines = [
            self.ARGS % (model, model),
            "(EngineCore pid=2) INFO [core.py:112] Initializing a V1 LLM engine (v0.29.0) "
            "with config: model='%s', tensor_parallel_size=2, enable_prefix_caching=True, "
            "enable_chunked_prefill=True" % model,
            "(Worker_TP0 pid=3) INFO [tpu_platform.py:1112] Using KV cache block size: 256",
            "(EngineCore pid=2) INFO [kv_cache_utils.py:2032] GPU KV cache size: 1,000 tokens",
        ]
        with open(os.path.join(job, "logs", "server.log"), "w") as f:
            f.write("\n".join(lines) + "\n")
        out = os.path.join(job, "outputs", "serving_bench")
        os.makedirs(out)
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump(
                {
                    "settings": {},
                    "cells": [{
                        "cell":
                        "isl1024_osl128_c1",
                        "isl":
                        1024,
                        "osl":
                        128,
                        "concurrency":
                        1,
                        "status":
                        "ok",
                        "replicates": [{
                            "completed": 4,
                            "duration_s": 2.0,
                            "ttft_mean_ms": 35.0
                        }]
                    }]
                }, f)
        return job, model

    def test_args_line_keeps_non_literal_values(self):
        d = jobdata.parse_args_line(self.ARGS % ("/m", "/m"))
        self.assertEqual(d["tensor_parallel_size"], 2)
        self.assertEqual(d["served_model_name"], ["openai/gpt-oss-20b"])
        self.assertTrue(d["profiler_config"].startswith("ProfilerConfig("))
        self.assertIsNone(jobdata.parse_args_line("INFO no args here"))

    def test_recipe_from_vllm_log(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            job, model = self._job(tmp)
            rec = jobdata.read_recipe(job)
        self.assertEqual(rec["job_id"], "j-srv")
        self.assertEqual(rec["serve_parse"], "vllm-log")
        self.assertEqual(rec["model_path"], model)
        self.assertEqual(rec["server_args"]["tensor-parallel-size"], 2)
        self.assertEqual(rec["server_args"]["max-num-batched-tokens"], 4096)
        self.assertEqual(rec["server_args"]["attention-backend"], "CUSTOM")
        self.assertNotIn("model", rec["server_args"])
        # left at vLLM's default, so taken from the engine config line
        self.assertIs(rec["server_args"]["enable-prefix-caching"], True)
        self.assertEqual(rec["env"], {})
        self.assertIsNone(rec["image"])
        rows = derive.env_provenance(rec, {
            "USE_MOE_SPARSE_CORE": "0",
            "X": "1"
        }, {"env.X": "1"}, {"USE_MOE_SPARSE_CORE": "0"})
        self.assertEqual([(r["name"], r["source"]) for r in rows],
                         [("USE_MOE_SPARSE_CORE", "model"), ("X", "override")])

    ENGINE = (
        "(EngineCore pid=2) INFO [core.py:112] Initializing a V1 LLM engine (v0.29.0) "
        "with config: model='/m', enable_prefix_caching=%s, enable_chunked_prefill=True"
    )

    def _recipe(self, lines, name="server.log"):
        """_recipe_from_vllm_log on a log holding `lines` (gzipped when `name` ends in .gz)."""
        import gzip
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            job = os.path.join(tmp, "j-srv")
            os.makedirs(os.path.join(job, "logs"))
            log = os.path.join(job, "logs", name)
            with (gzip.open(log, "wt") if name.endswith(".gz") else open(
                    log, "w")) as f:
                f.write("\n".join(lines) + "\n")
            return jobdata._recipe_from_vllm_log(job, log)

    def test_explicit_prefix_caching_flag_wins_over_engine_line(self):
        args = (
            "INFO non-default args: {'model': '/m', 'enable_prefix_caching': False}"
        )
        rec = self._recipe([args, self.ENGINE % "True"])
        self.assertIs(rec["server_args"]["enable-prefix-caching"], False)

    def test_prefix_caching_from_engine_line(self):
        args = "INFO non-default args: {'model': '/m'}"
        rec = self._recipe([args, self.ENGINE % "False"])
        self.assertIs(rec["server_args"]["enable-prefix-caching"], False)
        # no engine line: the flag is left out, not guessed
        rec = self._recipe([args])
        self.assertNotIn("enable-prefix-caching", rec["server_args"])

    def test_first_args_line_only(self):
        # a restarted server logs a second args line; the first one is the run being read
        rec = self._recipe([
            "INFO non-default args: {'model': '/m', 'tensor_parallel_size': 2}",
            "INFO non-default args: {'model': '/m', 'tensor_parallel_size': 8}",
        ])
        self.assertEqual(rec["server_args"]["tensor-parallel-size"], 2)

    def test_model_tag_when_no_model_key(self):
        rec = self._recipe([
            "INFO non-default args: {'model_tag': 'openai/gpt-oss-20b', "
            "'max_model_len': 8192}"
        ])
        self.assertEqual(rec["model_path"], "openai/gpt-oss-20b")
        self.assertEqual(rec["server_args"], {"max-model-len": 8192})

    def test_gzipped_log(self):
        rec = self._recipe([
            "INFO non-default args: {'model': '/m', 'tensor_parallel_size': 4}"
        ],
                           name="server.log.gz")
        self.assertEqual(rec["server_args"]["tensor-parallel-size"], 4)
        self.assertEqual(rec["launch_log"],
                         os.path.join("logs", "server.log.gz"))

    def test_snapshot_from_vllm_log(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            job, model = self._job(tmp)
            snap = jobdata.snapshot("j-srv", root=tmp)
        self.assertEqual(snap["checkpoint_config"]["num_hidden_layers"], 24)
        self.assertTrue(snap["checkpoint_config"]["_source"].startswith(
            os.path.join(model, "config.json")))
        self.assertEqual(snap["log"]["block_size_resolved"], 256)
        self.assertEqual(snap["log"]["kv_pool_tokens"], 1000)
        # no aiperf export: no server counters and no step slices (no metrics poller)
        self.assertEqual(snap["server"], {})
        self.assertIsNone(snap["step_slices"])


class ServingBenchmarkSummary(unittest.TestCase):
    """A serving_benchmark.py job (summary.json, no per-request records): its cells, and requests
    modelled from them as a closed loop."""

    def test_cells_and_modelled_requests(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "j", "outputs", "serving_bench")
            os.makedirs(out)
            rep = {"completed": 8, "duration_s": 10.0, "ttft_mean_ms": 200.0}
            with open(os.path.join(out, "summary.json"), "w") as f:
                json.dump(
                    {
                        "settings": {},
                        "cells": [{
                            "cell": "isl100_osl10_c4",
                            "isl": 100,
                            "osl": 10,
                            "concurrency": 4,
                            "status": "ok",
                            "replicates": [rep]
                        }, {
                            "cell": "skipped",
                            "isl": 9,
                            "osl": 9,
                            "concurrency": 1,
                            "status": "skipped",
                            "replicates": []
                        }]
                    }, f)
            cells = jobdata.read_summary_cells(os.path.join(tmp, "j"))
            reqs = jobdata.read_requests(os.path.join(tmp, "j"))
        self.assertEqual([c["cell"] for c in cells], ["isl100_osl10_c4"])
        self.assertEqual(len(reqs), 8)
        # 8 requests on 4 slots in 10 s: each lasts 5 s, the second wave starts at 5 s
        self.assertEqual([(r["t0"], r["t1"]) for r in reqs[3:5]],
                         [(0.0, 5.0), (5.0, 10.0)])
        self.assertTrue(
            all(r["isl"] == 100 and r["osl"] == 10 and r["ttft_ms"] == 200.0
                for r in reqs))


class SnapshotFormat(unittest.TestCase):
    """Snapshots are written one record per line and read back unchanged."""

    def test_round_trip_and_layout(self):
        snap = {
            "job_id": "j-1",
            "requests": [{
                "conv": i,
                "isl": 100 + i,
                "osl": 20
            } for i in range(3)],
            "step_slices": {
                "rows": [[float(i), 0, 1] for i in range(40)]
            },
            "recipe": {
                "script": "x" * 300
            },
        }
        text = jobdata.dumps_snapshot(snap, width=80)
        self.assertEqual(json.loads(text), snap)
        lines = text.splitlines()
        # A record that fits stays on one line.
        self.assertIn('  {"conv": 0, "isl": 100, "osl": 20},', lines)
        # Short flat rows share a line.
        rows = [line for line in lines if line.strip().startswith("[0.0")]
        self.assertIn("[1.0, 0, 1]", rows[0])
        # An object too long for the width is expanded, a long string is not split.
        self.assertTrue(
            any(line.startswith('  "script": "xxx') for line in lines))


class PrefixChainAndConcurrency(unittest.TestCase):

    def test_chain_and_gap_fallback(self):
        reqs = [
            {
                "conv": "a",
                "turn": 0,
                "phase": "warmup",
                "t0": 0,
                "isl": 100,
                "osl": 50
            },
            {
                "conv": "a",
                "turn": 1,
                "phase": "profiling",
                "t0": 1,
                "isl": 200,
                "osl": 80
            },
            # shorter than turn 1's prompt+output (280): falls back to turn 0 (150)
            {
                "conv": "a",
                "turn": 2,
                "phase": "profiling",
                "t0": 2,
                "isl": 260,
                "osl": 10
            }
        ]
        r = jobdata.prefix_chain(reqs)
        self.assertEqual([(x["reusable"], x["src_turn"], x["gap"]) for x in r],
                         [(0, None, False), (150, 0, False), (150, 0, True)])
        self.assertEqual(r[2]["new"], 110)

    def test_concurrency_time_weighted(self):
        reqs = [{
            "phase": "profiling",
            "t0": 0,
            "t1": 10
        }, {
            "phase": "profiling",
            "t0": 0,
            "t1": 2
        }, {
            "phase": "warmup",
            "t0": 0,
            "t1": 10
        }]
        c = jobdata.concurrency(reqs)
        self.assertEqual(c["time_share"], {1: 0.8, 2: 0.2})
        self.assertEqual((c["p50"], c["p90"], c["max"]), (1, 2, 2))


class EnvAudit(unittest.TestCase):
    REC = {
        "serve_parse": "bash-eval",
        "env_from_script": {
            "LIBTPU_INIT_ARGS": "--x"
        }
    }
    ENV = {
        "LIBTPU_INIT_ARGS": "--x",
        "USE_MOE_SPARSE_CORE": "1",
        "VLLM_DCP_Q_REPLICATE": "0"
    }

    def test_sources(self):
        rows = {
            r["name"]: r
            for r in derive.env_provenance(self.REC, self.ENV,
                                           {"env.USE_MOE_SPARSE_CORE": "1"})
        }
        self.assertEqual(rows["LIBTPU_INIT_ARGS"]["source"], "script")
        self.assertEqual(rows["USE_MOE_SPARSE_CORE"]["source"], "override")
        self.assertEqual(rows["VLLM_DCP_Q_REPLICATE"]["source"], "container")
        self.assertFalse(any("review" in r
                             for r in rows.values()))  # no resharding

    def test_resharding_flags_sharding_env_for_review(self):
        rows = {
            r["name"]: r
            for r in derive.env_provenance(self.REC, self.ENV, {
                "dcp": "8",
                "env.VLLM_DCP_Q_REPLICATE": "1"
            })
        }
        self.assertIn("review", rows["LIBTPU_INIT_ARGS"])
        self.assertNotIn("review",
                         rows["VLLM_DCP_Q_REPLICATE"])  # set explicitly
        self.assertNotIn("review",
                         rows["USE_MOE_SPARSE_CORE"])  # not a sharding knob

    def test_unevaluated_script_is_labelled(self):
        rows = derive.env_provenance({"serve_parse": "regex-fallback"},
                                     {"A": "1"}, {})
        self.assertIn("NOT evaluated", rows[0]["source"])

    def test_node_ids_are_plumbing(self):
        self.assertEqual(
            derive.server_env({
                "TPU_NODE_ID": "0",
                "TPU_WORKER_ID": "0",
                "X": "1"
            }), {"X": "1"})


if __name__ == "__main__":
    unittest.main()
