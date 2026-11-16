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
"""Generic adapter for standard decoder-only checkpoints. Reviewed code: change via PR.

core/derive.py load_model uses it when no `<MODEL_DIR>/model.py` exists and `classify(cfg)`
returns no reasons. Same interface as the model-specific adapters (facts, twin_overrides,
twin_facts, per_rank_rows, kernels, kv_bytes_per_token_per_rank, param_bytes_per_rank,
PROJECTION_ROWS, TWIN_CARRY_ARGS, TRACE, ROUTER, COMPONENTS); no DRAFTER, no SAMPLING.

Covers dense and MoE transformers with full GQA/MHA attention on every layer: Llama, Mistral,
Mixtral, Qwen2, Qwen2-MoE, Qwen3, Qwen3-MoE (MODEL_TYPES lists the families checked against
their vLLM model files; `classify` is the only gate). Layer kinds: `dense` and `moe`. The twin
keeps every per-rank size and cuts the stack to one layer per kind, unlike vLLM's test helper,
which also shrinks experts.

Sources: vllm@fc0cfcdc05b1717cadb17a0e24a6b1f611aec456 (GitHub, default branch `main`), paths
relative to the vLLM repo root:
  model files   vllm/model_executor/models/registry.py:154,178,180,199-202 (arch -> module)
  text config   vllm/transformers_utils/config.py:1257-1271 (get_hf_text_config: the nested
                text_config of a multimodal checkpoint)
  field quirks  vllm/transformers_utils/config.py:56-86 (CONFIG_FIELD_ALIASES: KV heads,
                experts, top-k, shared experts); tests/models/utils.py:469-478, 534-545
                (dummy_hf_overrides: n_group / num_expert_group, num_experts_per_tok / _token,
                num_local_experts, n_routed_experts, first_k_dense_replace, get_text_config)
  head_dim      vllm/transformers_utils/model_arch_config_convertor.py:104-118 (head_dim, else
                hidden / heads); :152-157 (KV heads default to query heads)
  layer kinds   vllm/model_executor/models/qwen3_moe.py:385-392 and qwen2_moe.py:313-319 (MoE
                unless in mlp_only_layers or off decoder_sparse_step); glm4_moe.py:353-356
                (dense below first_k_dense_replace); mixtral.py:246-250 (every layer MoE, experts
                intermediate_size wide)
  shared expert qwen3_moe.py:177-195 (shared_expert_intermediate_size); glm4_moe.py:173
                (moe_intermediate_size x n_shared_experts)
  sharding      vllm/model_executor/layers/linear.py:1066-1071 (QKVParallelLinear: KV heads
                replicate when TP >= KV heads, one per rank)
  KV bytes      vllm/v1/kv_cache_interface.py:570-579 (page = KV heads x (K + V head size) x dtype)
  non-standard  llama.py:185-204 (layer_types "sliding_attention"); model_arch_config_convertor.py
                :294-340 (MLA via kv_lora_rank)
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from core.deploy import Deploy, Facts

MODEL_DIR = "generic"
MODEL_TYPES = ("llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen3",
               "qwen3_moe")

#: vllm/transformers_utils/config.py:56-86, first present wins.
KV_HEADS = ("n_head_kv", "num_kv_heads", "num_key_value_heads",
            "multi_query_group_num", "num_attention_groups")
EXPERTS = ("num_experts", "moe_num_experts", "n_routed_experts",
           "num_local_experts")
TOPK = ("num_experts_per_tok", "num_experts_per_token", "top_k_experts",
        "moe_topk", "moe_top_k", "top_k")
DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


def _text(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg


def _first(t: dict, names: tuple[str, ...], default: Any = 0) -> Any:
    for n in names:
        if t.get(n) is not None:
            return t[n]
    return default


def classify(cfg: dict) -> list[str]:
    """Why this checkpoint is NOT a standard decoder-only model ([] = standard). Each reason
    names what a model-specific adapter must model."""
    t = _text(cfg)
    why = []
    if not all(
            t.get(k) for k in ("num_hidden_layers", "hidden_size",
                               "num_attention_heads")):
        why.append("not a decoder-only LM config (no num_hidden_layers / "
                   "hidden_size / num_attention_heads)")
    other = sorted(set(t.get("layer_types") or []) - {"full_attention"})
    if other:
        why.append(f"hybrid layer_types {other} (linear/sliding/recurrent "
                   "layers need their own kinds and state bytes)")
    if any(
            t.get(k) for k in ("hybrid_override_pattern", "attn_layer_indices",
                               "block_configs", "layers_block_type")):
        why.append("hybrid attention/recurrent stack")
    if t.get("kv_lora_rank"):
        why.append("MLA (kv_lora_rank): latent KV, not per-head")
    if any(t.get(k) for k in ("index_topk", "index_n_heads", "indexer_types")):
        why.append("DSA sparse-attention indexer")
    if any(
            t.get(k)
            for k in ("num_nextn_predict_layers", "mtp_num_hidden_layers")):
        why.append("MTP layers (num_nextn_predict_layers / "
                   "mtp_num_hidden_layers) need a DRAFTER")
    if (t.get("sliding_window") and t.get("use_sliding_window") is not False) \
            or t.get("sliding_window_pattern") or t.get("attention_chunk_size"):
        why.append("sliding-window / chunked local attention")
    if cfg.get("is_encoder_decoder") or t.get("cross_attention_layers"):
        why.append("cross-attention / encoder-decoder")
    if any(cfg.get(k) for k in ("vision_config", "audio_config")):
        why.append("multimodal checkpoint (vision/audio tower, text_config "
                   "overrides, --language-model-only)")
    if any(
            t.get(k) for k in ("interleave_moe_layer_step", "moe_layer_freq",
                               "moe_layer_start_index")):
        why.append("MoE layer placement this adapter does not read")
    return why


def _is_moe(t: dict, i: int, experts: int) -> bool:
    return (experts > 0 and i >= (t.get("first_k_dense_replace") or 0)
            and i not in (t.get("mlp_only_layers") or [])
            and (i + 1) % (t.get("decoder_sparse_step") or 1) == 0)


def _wbytes(cfg: dict) -> float:
    q = str((cfg.get("quantization_config") or {}).get("quant_method", ""))
    if "fp8" in q:
        return 1
    if q in ("awq", "gptq", "awq_marlin", "gptq_marlin"):
        return 0.5
    return DTYPE_BYTES.get(str(_dtype(cfg)), 2)


def _dtype(cfg: dict) -> str:
    t = _text(cfg)
    return t.get("torch_dtype") or t.get("dtype") or cfg.get(
        "torch_dtype") or "bfloat16"


#: The checkpoint values Facts has no field for, kept by facts().
_CFG: dict[str, Any] = {}


def facts(cfg: dict, kv_cache_dtype: str = "auto") -> Facts:
    why = classify(cfg)
    if why:
        raise ValueError(
            "generic adapter: not a standard decoder-only model: " +
            "; ".join(why))
    t = _text(cfg)
    n, heads = t["num_hidden_layers"], t["num_attention_heads"]
    hd = t.get("head_dim") or t["hidden_size"] // heads
    kv = _first(t, KV_HEADS, heads)
    experts = _first(t, EXPERTS)
    kinds = tuple("moe" if _is_moe(t, i, experts) else "dense"
                  for i in range(n))
    kv_b = 1 if "fp8" in kv_cache_dtype else (DTYPE_BYTES.get(
        str(_dtype(cfg)), 2) if kv_cache_dtype == "auto" else 2)
    kv_layer = 2 * kv * hd * kv_b  # K + V, all KV heads
    eff = t.get("moe_intermediate_size") or t.get("intermediate_size", 0)
    shared = (t.get("shared_expert_intermediate_size") or eff *
              (t.get("n_shared_experts") or 0)) if experts else 0
    _CFG.clear()
    _CFG.update(head_dim=hd,
                weight_bytes=_wbytes(cfg),
                tied=bool(cfg.get("tie_word_embeddings")),
                keys=set(t),
                quant=(cfg.get("quantization_config")
                       or {}).get("quant_method"))
    return Facts(
        name=f"generic:{cfg.get('model_type')}",
        layers=n,
        dense_layers=kinds.count("dense"),
        hidden=t["hidden_size"],
        heads=heads,
        kv_heads=kv,
        mla=False,
        experts=experts,
        topk=_first(t, TOPK) if experts else 0,
        expert_ffn=eff if experts else 0,
        shared_ffn=shared,
        dense_ffn=t.get("intermediate_size", 0) if "dense" in kinds else 0,
        kv_cache_bytes_per_token_per_layer=kv_layer,
        vocab=t.get("vocab_size", 0),
        kinds=kinds,
        kv_bytes_token=kv_layer * n,
    )


def _kv_heads_rank(f: Facts, d: Deploy) -> int:
    return max(1, f.kv_heads // max(1, d.tp))


def kv_bytes_per_token_per_rank(f: Facts, d: Deploy) -> float:
    """Paged KV one rank holds per token: its KV heads (one, replicated, when TP >= KV heads)
    on every layer, split by DCP."""
    return f.kv_bytes_token * _kv_heads_rank(f, d) / f.kv_heads / d.dcp


def feasibility_checks(f: Facts, d: Deploy) -> list[str]:
    tp = max(1, d.tp)
    if f.kv_heads % tp and tp % f.kv_heads:
        return [
            f"kv heads {f.kv_heads} and tp {tp}: neither divides the other"
        ]
    return []


def per_rank_rows(f: Facts,
                  d: Deploy,
                  pool_tokens: int | None = None) -> dict[str, Any]:
    return {
        "attn_kv_heads_per_rank":
        _kv_heads_rank(f, d),
        "attn_kv_replicated":
        max(1, d.tp) > f.kv_heads,
        "kv_bytes_per_token_per_layer_per_rank":
        round(kv_bytes_per_token_per_rank(f, d) / f.layers, 1),
    }


PROJECTION_ROWS = [
    "attn_kv_heads_per_rank", "kv_bytes_per_token_per_layer_per_rank"
]

#: Served flags the twin must carry: the attention backend, the quantization and the dtype the
#: weights are built with.
TWIN_CARRY_ARGS = ["attention-backend", "quantization", "dtype"]

#: Profile rules for the vllm-torchtpu runtime, from the reviewed Qwen3.5 adapter's served
#: profile (j-9fb52480): 1 embedding all-reduce + attention and FFN out-projection per layer;
#: ragged paged attention `RPA*`, MoE `gmm` / `router_topk`; boundary all-reduce may lower to an
#: async `call-done` on large steps. Not yet confirmed on a generic-family profile:
#: census_check reports any mismatch.
TRACE = {
    "embed_all_reduces":
    1,
    "all_reduces_per_layer":
    2,
    "boundary_regex":
    r"^all-reduce\.\d+(\.cloned\.\d+\.call-done)?$",
    "census": {
        "families": {
            "attn": {
                "regex": r"^RPA[a-z]*-",
                "part": "attention"
            },
            "experts": {
                "regex": r"gmm|router_topk",
                "part": "ffn"
            }
        },
        "kinds": {
            "dense": ["attn"],
            "moe": ["attn", "experts"]
        },
    },
    "per_step_ops": [
        {
            "what": "RoPE cos/sin cache copy",
            "op": r"^copy$",
            "out": r"^bf16\[\d{5,},\d+\]$"
        },
        {
            "what": "RPA metadata schedule",
            "op": r"^rpa_metadata_schedule$"
        },
        {
            "what": "attention metadata H2D copy",
            "op": r"^copy-done$",
            "out": r"^s32\[\d+\]$"
        },
    ],
}

#: Router weights re-drawn on dummy weights: `mlp.gate` (Qwen MoE) and `block_sparse_moe.gate`
#: (Mixtral); the leading dot keeps `shared_expert_gate` out. Dense models match nothing.
ROUTER = {"weight_suffix": ".gate.weight", "zero_suffixes": [], "std": 0.05}


def kernels(env: dict[str, str], d: Deploy) -> dict[str, Any]:
    moe = (
        "fused_ep" if env.get("USE_MOE_FUSED_EP_KERNEL") == "1" else
        "gmm_sparsecore" if env.get("USE_MOE_SPARSE_CORE") == "1" else "gmm")
    knobs = {
        k: v
        for k, v in sorted(env.items())
        if k.startswith(("TPU_MOE", "USE_MOE", "ONEHOT_MOE", "MOE_",
                         "ENABLE_QUANTIZED", "TPU_ROPE", "TPU_RPA",
                         "VLLM_KV_CACHE_LAYOUT", "VLLM_TPU_FUSED",
                         "TPU_TOKEN_BUCKET"))
    }
    return {
        "attention": "ragged paged attention",
        "moe": moe,
        "linear": "block"
        if env.get("ENABLE_QUANTIZED_MATMUL_KERNEL") == "1" else "per_channel",
        "quantization": _CFG.get("quant"),
        "env": knobs,
    }


def _remap_not_convert(ckpt: dict, f: Facts, kinds: list[str]) -> dict | None:
    """The quantization skip-list names layers by index: give twin layer i the entries of the
    first real layer of its kind."""
    q = ckpt.get("quantization_config") or {}
    skip = q.get("modules_to_not_convert")
    if not skip:
        return None
    first: dict[str, int] = {}
    for i, k in enumerate(f.kinds):
        first.setdefault(k, i)
    out = [m for m in skip if not re.match(r"model\.layers\.\d+\.", m)]
    for twin_i, k in enumerate(kinds):
        pre = f"model.layers.{first[k]}."
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
    """One layer per kind (dense first, then moe); `global' / degree' == global / degree` for
    heads, KV heads, FFN widths and experts when the twin world is smaller (TP shrinks, else DP)."""
    r = real.world // twin_world
    if real.world % twin_world or r < 1:
        raise ValueError(
            f"twin world {twin_world} must divide real world {real.world}")
    tp = real.tp // r if real.tp >= r else real.tp
    tw = replace(real,
                 tp=tp,
                 dp=real.dp if real.tp >= r else real.dp // r,
                 dcp=max(1, real.dcp // (real.tp // tp)))
    tp_r = real.tp // tw.tp
    ep_r = real.ep_group // tw.ep_group if real.ep else 1
    problems = [
        f"{name} {glob} not divisible by {div}"
        for name, glob, div in (("heads", f.heads, tp_r), ("experts",
                                                           f.experts, ep_r),
                                ("topk", f.topk, ep_r), ("dense_ffn",
                                                         f.dense_ffn, tp_r))
        if glob and glob % div
    ]
    if real.tp < r and real.dp % r:
        problems.append(f"dp {real.dp} not divisible by {r}")
    kinds = [k for k in ("dense", "moe") if k in f.layer_kinds]
    keys = _CFG.get("keys", set())
    hf: dict[str, Any] = {"num_hidden_layers": len(kinds)}
    if "layer_types" in keys:
        hf["layer_types"] = ["full_attention"] * len(kinds)
    if len(kinds) == 2:
        if "first_k_dense_replace" in keys:
            hf["first_k_dense_replace"] = 1
        else:
            hf["mlp_only_layers"] = [0]
            hf["decoder_sparse_step"] = 1
    if tp_r > 1:
        hf["num_attention_heads"] = f.heads // tp_r
        hf["num_key_value_heads"] = max(1, f.kv_heads // tp_r)
        hf["head_dim"] = _CFG["head_dim"]  # hidden / heads would change it
        if f.dense_ffn or not f.experts:
            hf["intermediate_size"] = f.dense_ffn // tp_r
        if f.shared_ffn:
            hf["shared_expert_intermediate_size"] = f.shared_ffn // tp_r
        if f.experts and not real.ep:  # experts TP-sharded on their FFN width
            hf["moe_intermediate_size" if "moe_intermediate_size" in
               keys else "intermediate_size"] = f.expert_ffn // tp_r
    if ep_r > 1:
        name = next((k for k in EXPERTS if k in keys), "num_experts")
        hf[name] = f.experts // ep_r
        hf[next((k for k in TOPK if k in keys),
                "num_experts_per_tok")] = f.topk // ep_r
    qc = _remap_not_convert(ckpt or {}, f, kinds)
    if qc:
        hf["quantization_config"] = qc
    ctx_scale = tw.dcp / real.dcp if real.dcp > 1 else 1.0
    server: dict[str, Any] = {
        "tensor-parallel-size": tw.tp,
        "data-parallel-size": tw.dp,
        "decode-context-parallel-size": tw.dcp,
        "block-size": real.block_size,
        "max-model-len": int(real.max_model_len * ctx_scale),
        "load-format": "dummy",
    }
    if pool_tokens:
        server[
            "num-gpu-blocks-override"] = pool_tokens // real.dcp // real.block_size
    deviations = []
    if tw.tp != real.tp:
        deviations.append({
            "what":
            "TP collective degree",
            "real":
            real.tp,
            "twin":
            tw.tp,
            "covered_by":
            "none: per-rank shapes match; all-reduce cost differs"
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
            f"layer census: {' + '.join('1 ' + k for k in kinds)}",
            "load-format dummy + hf_overrides boot through vllm-torchtpu",
            "TRACE census families (RPA, gmm) present in the twin profile",
        ],
    }


def twin_facts(f: Facts, hf: dict) -> Facts:
    n = hf["num_hidden_layers"]
    kinds = tuple(["dense", "moe"][:n] if hf.get("mlp_only_layers")
                  or hf.get("first_k_dense_replace") else [f.kinds[0]] * n)
    heads = hf.get("num_attention_heads", f.heads)
    kv = hf.get("num_key_value_heads", f.kv_heads)
    experts = next((hf[k] for k in EXPERTS if k in hf), f.experts)
    kv_layer = f.kv_cache_bytes_per_token_per_layer * kv // f.kv_heads
    # Mixtral-style experts are `intermediate_size` wide (no dense layers to share the key)
    expert_ffn = hf.get("moe_intermediate_size") or (
        hf.get("intermediate_size")
        if f.experts and not f.dense_ffn else None) or f.expert_ffn
    return replace(f,
                   layers=n,
                   dense_layers=kinds.count("dense"),
                   heads=heads,
                   kv_heads=kv,
                   experts=experts,
                   topk=next((hf[k] for k in TOPK if k in hf), f.topk),
                   expert_ffn=expert_ffn,
                   dense_ffn=hf.get("intermediate_size", f.dense_ffn)
                   if f.dense_ffn else 0,
                   shared_ffn=hf.get("shared_expert_intermediate_size",
                                     f.shared_ffn),
                   kinds=kinds,
                   kv_cache_bytes_per_token_per_layer=kv_layer,
                   kv_bytes_token=kv_layer * n)


def param_bytes_per_rank(f: Facts,
                         d: Deploy,
                         layers: dict[str, int] | None = None) -> dict:
    """Estimated resident weight bytes on one rank at the checkpoint's weight width (bf16
    embeddings, LM head and routers). TP-sharded: q/k/v/o (KV heads replicated past TP), dense
    and shared FFN, embeddings, LM head; routed experts EP-sharded, else TP-sharded."""
    kinds = layers or f.layer_kinds
    D, tp, hd, wb = f.hidden, max(1,
                                  d.tp), _CFG["head_dim"], _CFG["weight_bytes"]
    attn = (2 * D * f.heads * hd + 2 * D * _kv_heads_rank(f, d) * hd * tp) / tp
    experts = (f.experts / d.ep_group if d.ep else f.experts / tp)
    moe = experts * 3 * D * f.expert_ffn + 3 * D * f.shared_ffn / tp
    per = {
        "dense": (attn + 3 * D * f.dense_ffn / tp) * wb,
        "moe": (attn + moe) * wb + 2 * D * f.experts
    }
    layer_b = sum(per[k] * n for k, n in kinds.items())
    emb_b = (1 if _CFG.get("tied") else 2) * 2 * f.vocab * D / tp
    return {
        "layers": int(layer_b),
        "mtp": 0,
        "embed_head": int(emb_b),
        "total": int(layer_b + emb_b)
    }


COMPONENTS = {}
