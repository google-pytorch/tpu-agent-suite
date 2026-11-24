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
"""Project a real-workload L1 config onto a small slice: the per-rank twin config.

Input: an L1 config from `derive`/`onboard` (the real deployment, its per-rank shapes and
cells). Output: a standalone twin config that the runners consume and nothing else:

  twin              engine arguments + env for the twin
  projection        per-rank table: real vs twin, each row marked match / deviation
  memory            per-rank HBM, real vs twin (weights + KV) and the ballast that would equalize
  mtp               what the twin does and does not reproduce of speculative decoding
  run               how to run it (points plan -> forward_run.py)

Rule (design-notes P20-P24): every per-rank tensor keeps its real size,
global' / degree' == global / degree; the collectives sync across the twin's ranks.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from typing import Any

from .deploy import Deploy, kv_per_token_rank, per_rank

GiB = 1 << 30

#: Served flags every twin carries, whatever the adapter: they set the dtype of the recurrent
#: state the twin's engine allocates. Without `mamba-ssm-cache-dtype` a hybrid model's GDN state
#: ran float32 against a bfloat16 server (twice the bytes per page), and Qwen3.5-397B's fused
#: GDN kernel ran out of scoped VMEM at start-up.
CORE_CARRY_ARGS = ("mamba-ssm-cache-dtype", "mamba-cache-dtype")


def _cmp(real: Any, twin: Any) -> str:
    if real == twin:
        return "match"
    if isinstance(real, (int, float)) and isinstance(twin,
                                                     (int, float)) and real:
        rel = (twin - real) / real
        if abs(rel) <= 0.005:  # integer page rounding and the like
            return f"match ({rel:+.2%})"
    return "deviation"


#: TPU cores one host gives the job (the source's tpu7x 2x2x1 host: 4 chips x 2 cores; check the
#: target with `compute inspect`). A twin wider than this spans hosts and runs on Ray.
CORES_PER_HOST = 8


def project(cfg: dict, snap: dict, mod, twin_world: int = 8) -> dict[str, Any]:
    """The twin (one layer per kind) on `twin_world` cores. Two modes fall out of the one rule
    `global' / degree' == global / degree`:
      scaled   twin_world < the served world (default 8, one 2x2x1 host): the model's split
               dimensions scale by twin_world / world so every rank keeps its served shape
      mirror   twin_world == the served world: nothing scales; the served sharding runs as is,
               on the served slice, and only the layer count shrinks"""
    if not cfg["feasibility"]["feasible"]:
        raise ValueError("config is infeasible; nothing to project")
    d = Deploy(**cfg["deploy"])
    ckpt = snap["checkpoint_config"]
    f = mod.facts(ckpt, d.kv_cache_dtype)
    pool = cfg["per_rank"].get("kv_pool_tokens")
    t = mod.twin_overrides(f, d, twin_world, pool, ckpt)
    hf, sv = t["hf_overrides"], t["server_args"]
    ft = mod.twin_facts(f, hf)
    dt = replace(d,
                 tp=sv["tensor-parallel-size"],
                 dp=sv["data-parallel-size"],
                 dcp=sv["decode-context-parallel-size"],
                 block_size=sv["block-size"],
                 max_model_len=sv["max-model-len"])
    scale = t["context_scale"]
    pool_t = sv.get("num-gpu-blocks-override",
                    0) * dt.block_size * dt.dcp or None
    rows_hook = getattr(mod, "per_rank_rows", None)
    kv_rank = getattr(mod, "kv_bytes_per_token_per_rank", None)
    pr, prt = per_rank(f, d, pool, rows_hook,
                       kv_rank), per_rank(ft, dt, pool_t, rows_hook, kv_rank)

    # --- per-rank table: the rule says every row that a rank holds on its own matches
    rows = []
    for key in [
            "q_heads_projected", "experts_per_rank", "dense_ffn_per_rank",
            "shared_expert_ffn_per_rank", "cache_block_tokens",
            "kv_pool_tokens_per_rank", *getattr(mod, "PROJECTION_ROWS", []),
            "kv_divisor"
    ]:
        if key in pr:
            rows.append({
                "per_rank": key,
                "real": pr[key],
                "twin": prt.get(key),
                "status": _cmp(pr[key], prt.get(key))
            })
    rows.append({
        "per_rank": "kv_tokens_for_context_L",
        "real": f"L / {d.dcp}",
        "twin": f"(L x {scale}) / {dt.dcp} = L / {int(round(dt.dcp / scale))}",
        "status": _cmp(d.dcp, round(dt.dcp / scale))
    })
    rows.append({
        "per_rank": "routed_pairs_per_rank",
        "real": f"T x {f.topk}/{d.ep_group}",
        "twin": f"T x {ft.topk}/{dt.ep_group}",
        "status": _cmp(f.topk / d.ep_group, ft.topk / dt.ep_group)
    })
    rows.append({
        "per_rank": "collective_degree",
        "real": d.tp,
        "twin": dt.tp,
        "status": _cmp(d.tp, dt.tp)
    })
    rows.insert(
        0, {
            "per_rank": "kv_page_tokens",
            "real": d.block_size,
            "twin": dt.block_size,
            "status": _cmp(d.block_size, dt.block_size)
        })
    # kv_divisor and collective degree differ by design (that is the projection); flag them
    for r in rows:
        if r["per_rank"] in ("kv_divisor", "collective_degree"):
            r["status"] = "by design"
        if r["per_rank"] == "cache_block_tokens" and r["status"] != "match":
            # the prefix-cache hash block matters only to a scheduler; forward_run has none and
            # the plan applies the served cache rule (block x DCP) before the run
            r["status"] = "not used by forward_run (plan applies the served rule)"

    # --- memory per rank
    wr = mod.param_bytes_per_rank(f, d)
    wt = mod.param_bytes_per_rank(ft, dt)
    state = getattr(mod, "state_bytes_per_seq_per_rank", None)
    st_real = d.mns * state(f, d) if state else 0
    st_twin = d.mns * state(ft, dt) if state else 0

    # per token per rank; the MTP layer's KV splits like a backbone layer's
    def kv_tok(ff, dd):
        return kv_per_token_rank(ff, dd, kv_rank) * dd.dcp

    mtp_tok = (kv_tok(f, d) * f.mtp_kv_bytes_token / f.kv_bytes_per_token
               if f.kv_bytes_per_token else f.mtp_kv_bytes_token)
    kv_real = (pool // d.dcp if pool else 0) * kv_tok(f, d) + st_real
    kv_twin = (pool_t // dt.dcp if pool_t else 0) * (kv_tok(ft, dt) +
                                                     mtp_tok) + st_twin
    memory = {
        "weights_real_gib":
        round(wr["total"] / GiB, 2),
        "weights_twin_gib":
        round(wt["total"] / GiB, 2),
        "kv_real_gib":
        round(kv_real / GiB, 2),
        "kv_twin_gib":
        round(kv_twin / GiB, 2),
        "ballast_to_match_gib":
        round((wr["total"] + kv_real - wt["total"] - kv_twin) / GiB, 2),
        "note":
        f"estimates; per-layer footprint matches, totals do not ({ft.layers} vs "
        f"{f.layers} layers{' + MTP' if f.mtp_layers else ''}). Ballast is reported, not "
        "applied (needs an in-worker hook).",
    }

    mtp = {
        "served": {
            "method": d.spec_method,
            "num_speculative_tokens": d.spec_tokens
        },
        "twin_engine":
        f"decode rows carry 1 + {d.spec_tokens} tokens and the drafter's block "
        "drop is in the hit rule; the drafter itself runs only in plans made with "
        "`points --mtp` (its propose replay, timed after every step)",
        "not_reproduced":
        "acceptance (the plan uses the job's measured acceptance; the sample "
        "stage runs only in plans with sampling.run), "
        "and whatever the drafter replay leaves out (see the model's DRAFTER)",
    } if d.spec_method else None

    server_args = {
        **sv,
        "max-num-seqs":
        d.mns,
        "max-num-batched-tokens":
        d.mnbt,
        "kv-cache-dtype":
        d.kv_cache_dtype,
        "async-scheduling":
        True,
        "cp-kv-cache-interleave-size":
        d.interleave if dt.dcp > 1 else None,
        "no-dcp-q-replicate": (dt.dcp > 1 and not d.dcp_q_replicate) or None,
        "enable-expert-parallel":
        d.ep or None,
        "enable-prefix-caching":
        d.prefix_caching or None,
        "speculative-config":
        json.dumps({
            "method": d.spec_method,
            "num_speculative_tokens": d.spec_tokens
        }) if d.spec_method else None,
        "skip-tokenizer-init":
        True,
        # Setting this makes vllm-torchtpu skip its memory-profiling warmup (no warmup forward and
        # no extra graph compile). The block count stays pinned by num-gpu-blocks-override; this
        # only has to be large enough for it (and for any recurrent state).
        "kv-cache-memory-bytes":
        int(math.ceil(kv_twin * 1.25 / (1 << 20))) << 20 if kv_twin else None,
    }
    # served flags the model needs carried verbatim (attention backend, cache mode, ...)
    served = (snap.get("recipe") or {}).get("server_args") or {}
    carried = {
        k: served[k]
        for k in (*CORE_CARRY_ARGS, *getattr(mod, "TWIN_CARRY_ARGS", []))
        if k in served
    }
    server_args = {**carried, **server_args}
    server_args = {k: v for k, v in server_args.items() if v is not None}
    mode = "mirror" if twin_world == d.world else "scaled"
    return {
        "_what":
        f"workload-microbenchmark twin config ({mode}): real {d.label()} on "
        f"{twin_world} cores (generated; do not hand-edit)",
        "mode":
        mode,
        "_header":
        cfg.get("_header"),
        "job_id":
        cfg["job_id"],
        "model_dir":
        cfg["model_dir"],
        "overrides":
        cfg["overrides"],
        "real_deploy":
        asdict(d),
        "twin_deploy":
        asdict(dt),
        "twin_world":
        twin_world,
        "context_scale":
        scale,
        "layers":
        t["layers"],
        "twin": {
            "hf_overrides": hf,
            "server_args": server_args,
            "env": {
                **cfg.get("server_env", {}),
                **cfg["kernels"].get("env", {})
            },
            "multihost": twin_world > CORES_PER_HOST,
            "hosts": math.ceil(twin_world / CORES_PER_HOST)
        },
        "projection":
        rows,
        "deviations":
        t["deviations"],
        "problems":
        t["problems"],
        "verify_before_boot":
        t["verify_before_boot"],
        "memory":
        memory,
        "mtp":
        mtp,
        "components":
        cfg.get("components"),
        "run":
        "cli.py points <this>  ->  cli.py forward-cmd --steps <dir>/<cell>.steps.json",
    }
