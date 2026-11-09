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
"""GLM-5.2-FP8 adapter for workload-microbenchmark. Reviewed code: change via PR.

Everything model-specific lives here and nowhere else; core/ only counts, divides and formats:
  facts()          checkpoint config.json -> canonical Facts (incl. per-layer kinds, KV bytes)
  twin_overrides() hf_overrides that give a small-slice twin the big deployment's per-rank
                   shapes, one layer per kind
  twin_facts()     the twin as Facts
  per_rank_rows()  model-specific per-rank rows (attention variant, indexer)
  kernels()        which implementation variant each op runs, read from the recipe env
  PROJECTION_ROWS  per-rank rows the twin check compares beyond the generic ones
  TRACE            how to read this model's profile: layer boundaries, op families per layer kind
  ROUTER           router parameters to re-draw on dummy weights
  DRAFTER          the MTP drafter's replay entry point in vllm-torchtpu
  COMPONENTS       the exact-shape benchmarks this model needs where the twin deviates

Sources:
  checkpoint  gs://<bucket>/ckpt/hf/zai-org/GLM-5.2-FP8/config.json
  layer kinds vLLM 0.27 deepseek_v2.py:1096-1121 (index_topk_freq / index_skip_topk_offset,
              or index_topk_pattern with "S" = skip); first_k_dense_replace
  sharding    deepseek_v2.py: fused q_a/kv_a and indexer projections replicated
              (:912-925, :512, :669-683); shared expert and dense FFN TP-sharded (:232-263)
  kernels     vllm-torchtpu: USE_MOE_FUSED_EP_KERNEL / USE_MOE_SPARSE_CORE select the
              MoE path (fused_moe.py); linears use per-channel xla_quantized_matmul unless
              ENABLE_QUANTIZED_MATMUL_KERNEL (fp8.py:20-37)
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from core.deploy import Deploy, Facts

MODEL_DIR = "GLM-5.2-FP8"
MODEL_TYPES = ("glm_moe_dsa", )


def _indexer_ids(cfg: dict) -> tuple[int, ...]:
    n = cfg["num_hidden_layers"]
    pat = cfg.get("index_topk_pattern")
    if pat:
        return tuple(i for i in range(n) if i < len(pat) and pat[i] != "S")
    f = cfg.get("index_topk_freq", 1)
    o = cfg.get("index_skip_topk_offset", 2)
    return tuple(i for i in range(n) if max(i - o + 1, 0) % f == 0)


def _kinds(n: int, dense_first: int, idx: set[int]) -> tuple[str, ...]:
    """Layer kind per layer: FFN (first `dense_first` dense, the rest MoE) x own indexer."""
    return tuple(
        ("dense" if i < dense_first else "moe") + ("_idx" if i in idx else "")
        for i in range(n))


def facts(cfg: dict, kv_cache_dtype: str = "fp8") -> Facts:
    if cfg.get("model_type") not in MODEL_TYPES:
        raise ValueError(
            f"{MODEL_DIR}: unexpected model_type {cfg.get('model_type')!r}")
    kv_b = 1 if "fp8" in kv_cache_dtype else 2
    ihd = cfg.get("index_head_dim", 0)
    n = cfg["num_hidden_layers"]
    idx = _indexer_ids(cfg)
    latent = (cfg.get("kv_lora_rank", 0) +
              cfg.get("qk_rope_head_dim", 0)) * kv_b
    # DeepseekV32IndexerCache: head_dim fp8 + head_dim//128 fp32 scales (deepseek_v2.py:714-718)
    idx_b = ihd + (ihd // 128) * 4 if ihd else 0
    mtp = cfg.get("num_nextn_predict_layers", 0)
    return Facts(
        name=MODEL_DIR,
        layers=cfg["num_hidden_layers"],
        dense_layers=cfg.get("first_k_dense_replace", 0),
        hidden=cfg["hidden_size"],
        heads=cfg["num_attention_heads"],
        kv_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        mla=cfg.get("kv_lora_rank", 0) > 0,
        kv_lora_rank=cfg.get("kv_lora_rank", 0),
        qk_rope_head_dim=cfg.get("qk_rope_head_dim", 0),
        qk_nope_head_dim=cfg.get("qk_nope_head_dim", 0),
        v_head_dim=cfg.get("v_head_dim", 0),
        q_lora_rank=cfg.get("q_lora_rank") or 0,
        experts=cfg.get("n_routed_experts", 0),
        topk=cfg.get("num_experts_per_tok", 0),
        expert_ffn=cfg.get("moe_intermediate_size", 0),
        shared_ffn=cfg.get("moe_intermediate_size", 0) *
        cfg.get("n_shared_experts", 0),
        dense_ffn=cfg.get("intermediate_size", 0),
        indexer_heads=cfg.get("index_n_heads", 0),
        indexer_head_dim=ihd,
        indexer_topk=cfg.get("index_topk", 0),
        indexer_layer_ids=idx,
        indexer_cache_bytes_per_token=idx_b,
        kv_cache_bytes_per_token_per_layer=latent,
        mtp_layers=mtp,
        vocab=cfg.get("vocab_size", 0),
        kinds=_kinds(n, cfg.get("first_k_dense_replace", 0), set(idx)),
        kv_bytes_token=latent * n +
        idx_b * len(idx),  # latent on every layer + indexer K
        mtp_kv_bytes_token=(latent + idx_b) *
        mtp,  # the MTP layer has both caches
    )


def per_rank_rows(f: Facts,
                  d: Deploy,
                  pool_tokens: int | None = None) -> dict[str, Any]:
    """MLA under DCP and the DSA indexer, per rank."""
    tp = max(1, d.tp)
    out: dict[str, Any] = {
        # DCP attends with the whole DCP group's heads over this rank's KV shard;
        # q-replicate only changes whether that costs a per-step Q all-gather.
        "heads_in_attention": (f.heads // tp) * d.dcp,
        "q_head_allgather": d.dcp > 1 and not d.dcp_q_replicate,
    }
    if f.mla:
        out["mla_kv_bytes_per_token_per_layer"] = f.kv_cache_bytes_per_token_per_layer
    if f.indexer_layer_ids:
        import math
        out["indexer_topk_local"] = f.indexer_topk
        out["indexer_merge_width"] = d.dcp * f.indexer_topk if d.dcp > 1 else None
        out["indexer_score_row_padded"] = (
            math.ceil(d.max_model_len / d.cache_block_tokens) *
            d.block_size) if d.max_model_len else None
    return out


#: Per-rank rows (beyond core's generic ones) the twin check compares, real vs twin.
PROJECTION_ROWS = [
    "indexer_score_row_padded", "heads_in_attention", "indexer_merge_width"
]

#: How to read this model's profile (carried in every plan; used by xprof_steps / evidence).
#:  embed_all_reduces / all_reduces_per_layer: synchronous all-reduces before layer 0 and per
#:    layer (attention output + FFN output, TP row-parallel). Layer i ends at the end of
#:    all-reduce embed_all_reduces - 1 + all_reduces_per_layer * (i + 1).
#:  census: op families (regex on the HLO op name), the layer part they belong to (ops before the
#:    layer's first all-reduce = "attention", after it = "ffn"), and per kind which families must
#:    be present; every other family must be absent. Kernel names are vllm-torchtpu's.
#:  drafter_topk_pair: (kind with an indexer, same kind without): their difference models the
#:    MTP drafter's step-0 sparse top-k (IndexShare), which the drafter replay can't show.
#:  boundary_regex: small decode TP all-reduces can lower to Pallas
#:    `small_all_reduce_{oneshot,twoshot}_r<rows>` or fused `fp8_oproj_all_reduce*` kernels, while
#:    large steps can lower to async `all-reduce.N.cloned.M.call-done` (with optional XLA
#:    `.refs_k_` suffixes).
TRACE = {
    "embed_all_reduces":
    1,
    "all_reduces_per_layer":
    2,
    "boundary_regex":
    (r"^(?:all-reduce\.\d+(?:\.cloned\.\d+\.call-done)?"
     r"|small_all_reduce_(?:oneshot|twoshot)_r\d+(?:\.\d+)?"
     r"|fp8_oproj_all_reduce\w*(?:\.\d+)?)(?:\.refs_\d+_)*$"),
    "census": {
        "families": {
            "indexer": {
                "regex": r"StreamIdx|gather_fusion|scatter_custom_fusion",
                "part": "attention"
            },
            "experts": {
                "regex": r"gmm|router_topk|ragged",
                "part": "ffn"
            }
        },
        "kinds": {
            "dense_idx": ["indexer"],
            "dense": [],
            "moe_idx": ["indexer", "experts"],
            "moe": ["experts"]
        },
    },
    "drafter_topk_pair": ["moe_idx", "moe"],
}

#: Router parameters re-drawn on dummy weights, so tokens spread over experts as real ones do.
ROUTER = {
    "weight_suffix": "mlp.gate.weight",
    "zero_suffixes": ["mlp.gate.e_score_correction_bias"],
    "std": 0.05
}

#: The MTP drafter's replay in vllm-torchtpu (spec_decode/eagle3.py): replays the propose trace
#: for one step; `gate` is the method that limits it to DP-lockstep runs (lifted on the worker).
DRAFTER = {
    "file": "spec_decode/eagle3.py",
    "cls": "Eagle3Proposer",
    "replay": "run_dp_dummy_draft",
    "gate": "_dp_lockstep_sharded",
    "source_funcs":
    ["Eagle3Proposer.run_dp_dummy_draft", "Eagle3Proposer.propose"]
}


def kernels(env: dict[str, str], d: Deploy) -> dict[str, Any]:
    moe = (
        "fused_ep" if env.get("USE_MOE_FUSED_EP_KERNEL") == "1" else
        "gmm_sparsecore" if env.get("USE_MOE_SPARSE_CORE") == "1" else "gmm")
    knobs = {
        k: v
        for k, v in sorted(env.items())
        if k.startswith(("TPU_STREAMIDX", "TPU_MOE", "TPU_SCTOPK", "USE_MOE",
                         "ONEHOT_MOE", "ENABLE_QUANTIZED", "VLLM_DCP",
                         "TPU_ROPE"))
    }
    return {
        "mla_writer": "dcp" if d.dcp > 1 else "plain",
        "indexer": "dcp" if d.dcp > 1 else "plain",
        "moe": moe,
        "linear": "block"
        if env.get("ENABLE_QUANTIZED_MATMUL_KERNEL") == "1" else "per_channel",
        "env": knobs,
    }


#: Twin layer stack: one layer of each kind this model has, in an order vLLM can build.
#: A reuse ("S") layer needs a preceding full-indexer layer.
TWIN_LAYERS = {
    "dense_idx": ("dense", "F"),
    "moe_idx": ("moe", "F"),
    "moe": ("moe", "S"),
}


def _remap_not_convert(ckpt: dict, f: Facts, kinds: list[str]) -> dict | None:
    """The fp8 skip-list names layers by index. Give twin layer i the entries of a real layer of
    the same kind (and the MTP layer the real MTP layer's), so routers, norms, indexer and MTP
    projections keep their served dtype."""
    import re
    q = ckpt.get("quantization_config") or {}
    skip = q.get("modules_to_not_convert")
    if not skip:
        return None
    first = {}
    for i, k in enumerate(f.kinds):
        first.setdefault(k, i)
    src = [first[k]
           for k in kinds] + [f.layers + j for j in range(f.mtp_layers)]
    out = [m for m in skip if not re.match(r"model\.layers\.\d+\.", m)]
    for twin_i, real_i in enumerate(src):
        pre = f"model.layers.{real_i}."
        out += [
            f"model.layers.{twin_i}." + m[len(pre):] for m in skip
            if m.startswith(pre)
        ]
    return {**q, "modules_to_not_convert": sorted(set(out))}


def twin_overrides(f: Facts,
                   real: Deploy,
                   twin_world: int,
                   pool_tokens: int | None = None,
                   ckpt: dict | None = None) -> dict[str, Any]:
    """Shrink the MODEL so a `twin_world`-core deployment gives each rank the real per-rank
    shapes (design-notes P22-P24). Rule: global' / degree' == global / degree."""
    r = real.world // twin_world
    if real.world % twin_world or r < 1:
        raise ValueError(
            f"twin world {twin_world} must divide real world {real.world}")
    tw = replace(real,
                 tp=real.tp // r if real.tp >= r else real.tp,
                 dcp=max(1, real.dcp // r),
                 dp=real.dp if real.tp >= r else real.dp // r)
    tp_r = real.tp // tw.tp
    ep_r = real.ep_group // tw.ep_group if real.ep else 1
    dcp_r = real.dcp // tw.dcp
    problems = []
    for name, glob, div in (("heads", f.heads, tp_r), ("experts", f.experts,
                                                       ep_r),
                            ("topk", f.topk, ep_r), ("dense_ffn", f.dense_ffn,
                                                     tp_r)):
        if glob and glob % div:
            problems.append(f"{name} {glob} not divisible by {div}")
    kinds = [k for k in TWIN_LAYERS if k in f.layer_kinds]
    dense_first = sum(1 for k in kinds if TWIN_LAYERS[k][0] == "dense")
    pattern = [TWIN_LAYERS[k][1] for k in kinds]
    hf = {
        "num_hidden_layers": len(kinds),
        "first_k_dense_replace": dense_first,
        "index_topk_pattern": pattern,
        "num_attention_heads": f.heads // tp_r,
        "num_key_value_heads": f.kv_heads // tp_r,
        "n_routed_experts": f.experts // ep_r,
        "num_experts_per_tok": f.topk // ep_r,
        "intermediate_size": f.dense_ffn // tp_r,
    }
    qc = _remap_not_convert(ckpt or {}, f, kinds)
    if qc:
        hf["quantization_config"] = qc
    # The served KV page size: page-granular work per rank (KV-pool relayouts, the indexer's
    # page walk) then matches serving. The prefix-cache block (block x DCP) comes out smaller in
    # the twin, but forward_run runs no scheduler or prefix cache; `points` applies the served
    # cache rule before the run.
    twin_block = real.block_size
    ctx_scale = tw.dcp / real.dcp if real.dcp > 1 else 1.0
    server = {
        "tensor-parallel-size": tw.tp,
        "data-parallel-size": tw.dp,
        "decode-context-parallel-size": tw.dcp,
        "block-size": twin_block,
        # KV per rank = L / DCP: scaling context by DCP'/DCP keeps it, and keeps the
        # indexer's padded row width (max_model_len / DCP) too
        "max-model-len": int(real.max_model_len * ctx_scale),
        "load-format": "dummy",
    }
    if pool_tokens:
        # same per-rank cache size as serving: whole-cache ops (relayouts) scale with it
        per_rank = pool_tokens // real.dcp
        server["num-gpu-blocks-override"] = per_rank // twin_block
    deviations = []
    if dcp_r > 1:
        deviations.append({
            "what": "heads inside DCP attention",
            "real": (f.heads // real.tp) * real.dcp,
            "twin": (f.heads // tp_r // tw.tp) * tw.dcp,
            "covered_by": "component sparse_mla (real heads)"
        })
        if f.indexer_layer_ids:
            deviations.append({
                "what": "indexer merge width",
                "real": real.dcp * f.indexer_topk,
                "twin": tw.dcp * f.indexer_topk,
                "covered_by": "component indexer_merge"
            })
        deviations.append({
            "what":
            "DCP collective degree",
            "real":
            real.dcp,
            "twin":
            tw.dcp,
            "covered_by":
            "component collective (DCP 2/4/8 scaling)"
        })
    if f.shared_ffn and tp_r > 1:
        deviations.append({
            "what": "shared expert width per rank",
            "real": f.shared_ffn // real.tp,
            "twin": f.shared_ffn // tw.tp,
            "covered_by": "none (small; noted)"
        })
    if ep_r > 1:
        deviations.append({
            "what":
            "router: experts and top-k scaled together",
            "real":
            f"{f.experts} / top-{f.topk}",
            "twin":
            f"{f.experts // ep_r} / top-{f.topk // ep_r}",
            "covered_by":
            "per-rank routed pairs match; routing statistics differ"
        })
    return {
        "twin_world":
        twin_world,
        "twin_deploy":
        tw.label(),
        "layers":
        kinds,
        "hf_overrides":
        hf,
        "server_args":
        server,
        "context_scale":
        round(ctx_scale, 4),
        "deviations":
        deviations,
        "problems":
        problems,
        "verify_before_boot": [
            f"MLA/indexer kernels accept {f.heads // tp_r} heads",
            f"sparse paths accept {real.block_size}-token pages at DCP {tw.dcp}",
            "index_topk_pattern honored by the image's vLLM (else fall back to freq/offset)",
            "load-format dummy + hf_overrides boot through vllm-torchtpu",
        ],
    }


def twin_facts(f: Facts, hf: dict) -> Facts:
    """The twin model as Facts, from the overrides above: used to check per-rank shapes."""
    from dataclasses import replace as _r
    n = hf["num_hidden_layers"]
    pat = hf.get("index_topk_pattern") or []
    idx = tuple(i for i in range(n) if i < len(pat) and pat[i] != "S")
    return _r(f,
              layers=n,
              dense_layers=hf["first_k_dense_replace"],
              heads=hf["num_attention_heads"],
              kv_heads=hf["num_key_value_heads"],
              experts=hf["n_routed_experts"],
              topk=hf["num_experts_per_tok"],
              dense_ffn=hf["intermediate_size"],
              indexer_layer_ids=idx,
              kinds=_kinds(n, hf["first_k_dense_replace"], set(idx)),
              kv_bytes_token=f.kv_cache_bytes_per_token_per_layer * n +
              f.indexer_cache_bytes_per_token * len(idx))


def param_bytes_per_rank(f: Facts,
                         d: Deploy,
                         layers: dict[str, int] | None = None) -> dict:
    """Estimated resident weight bytes on one rank (fp8 weights, bf16 embeddings/head/norms
    ignored beyond embeddings). Replicated: q_a/kv_a and the indexer projections; TP-sharded:
    q_b, kv_b, o_proj, dense FFN, shared expert, LM head, embeddings; EP-sharded: routed experts.
    Sources: deepseek_v2.py (ReplicatedLinear / ColumnParallel / RowParallel, see module doc)."""
    kinds = layers or f.layer_kinds
    H, D = f.heads, f.hidden
    attn_rep = D * (f.q_lora_rank + f.kv_lora_rank + f.qk_rope_head_dim)
    attn_tp = (f.q_lora_rank * H *
               (f.qk_nope_head_dim + f.qk_rope_head_dim) + f.kv_lora_rank * H *
               (f.qk_nope_head_dim + f.v_head_dim) +
               H * f.v_head_dim * D) / d.tp
    idx = (f.q_lora_rank * f.indexer_heads * f.indexer_head_dim +
           D * f.indexer_head_dim + D * f.indexer_heads)
    moe = (f.experts / max(1, d.ep_group)) * 3 * D * f.expert_ffn + 3 * D * f.shared_ffn / d.tp \
        + D * f.experts
    dense = 3 * D * f.dense_ffn / d.tp
    per = {
        "dense_idx": attn_rep + attn_tp + idx + dense,
        "dense": attn_rep + attn_tp + dense,
        "moe_idx": attn_rep + attn_tp + idx + moe,
        "moe": attn_rep + attn_tp + moe
    }
    layer_b = sum(per[k] * n for k, n in kinds.items())
    mtp_b = f.mtp_layers * (per["moe_idx"] + 2 * D * D)  # + eh_proj
    emb_b = 2 * 2 * f.vocab * D / d.tp  # embed + lm_head, bf16
    return {
        "layers": int(layer_b),
        "mtp": int(mtp_b),
        "embed_head": int(emb_b),
        "total": int(layer_b + mtp_b + emb_b)
    }


#: Exact-shape component benchmarks, needed only for the twin's deviations (P24).
#: Implemented in M4; declared here so the report can say what is covered.
COMPONENTS = {
    "sparse_mla":
    "sparse attention at the real per-rank heads x L/DCP (rank-local)",
    "indexer_merge":
    "sparsecore_topk over [T, DCP x index_topk]",
    "collective":
    "DCP collectives at 2/4/8 on the small slice, fit to the real degree",
}
