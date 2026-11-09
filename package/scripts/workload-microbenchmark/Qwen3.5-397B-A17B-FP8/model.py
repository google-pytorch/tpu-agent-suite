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
"""Qwen3.5-397B-A17B-FP8 adapter for workload-microbenchmark. Reviewed code: change via PR.

Everything model-specific lives here and nowhere else; core/ only counts, divides and formats:
  facts()          checkpoint config.json -> canonical Facts (per-layer kinds, KV and GDN state bytes)
  twin_overrides() hf_overrides for the twin: one layer per kind, served per-rank shapes
  twin_facts()     the twin as Facts
  per_rank_rows()  model-specific per-rank rows (GQA KV replication, GDN heads, GDN state)
  kernels()        which implementation variant each op runs, read from the recipe env
  kv_bytes_per_token_per_rank(), state_bytes_per_seq_per_rank()  per-rank KV / recurrent state
  PROJECTION_ROWS  per-rank rows the twin check compares beyond the generic ones
  TWIN_CARRY_ARGS  served `vllm serve` flags the twin must carry verbatim
  TRACE            how to read this model's profile: layer boundaries, op families per layer kind
  ROUTER           router parameters to re-draw on dummy weights
  DRAFTER          the MTP drafter's replay entry point in vllm-torchtpu
  COMPONENTS       the exact-shape benchmarks this model needs where the twin deviates

A hybrid model: 60 layers, every 4th full attention (gated GQA, 32 q / 2 KV heads, head_dim 256),
the other 45 Gated DeltaNet (linear attention with a fixed per-request recurrent state). Every
layer's FFN is MoE (512 experts, top-10, plus one gated shared expert). One MTP layer (full
attention + MoE). The vision tower is never built (`--language-model-only`).

Sources:
  checkpoint  gs://<bucket>/ckpt/hf/Qwen/Qwen3.5-397B-A17B-FP8/config.json, generation
              1778713242141391;
              the language model's keys live under `text_config`, quantization_config at top level
  layer kinds vLLM 0.29 qwen3_5.py:144-162 (`layer_types[i]` picks linear_attention / full_attention;
              the MLP is Qwen3NextSparseMoeBlock on every layer of qwen3_5_moe_text);
              configs/qwen3_5_moe.py:94-101 (layer_types built from full_attention_interval at init,
              so the twin overrides `layer_types` itself, not only num_hidden_layers)
  GDN state   vLLM 0.29 mamba_utils.py:258-279 (conv [conv_dim/TP, kernel-1+k], SSM [v_heads/TP,
              v_dim, k_dim]); conv bf16, SSM `mamba_ssm_dtype` float32 (config) -- vllm-torchtpu
              tpu_block_size_utils.py:_hybrid_mamba_state_layout. Job j-9fb52480 logged
              mamba_raw_state_bytes=542720 per layer per rank at TP8, k=3, which this reproduces
  page size   the TPU runtime sizes the hybrid page so one page holds a GDN state
              (tpu_block_size_utils.py:_derive_tpu_block_slot_config): j-9fb52480 resolved
              `--block-size 256` to 1152. core/derive.py takes the resolved size from the log
  sharding    GQA KV heads replicate when TP > KV heads (vLLM QKVParallelLinear), so a TP8 rank
              holds 1 of the 2 KV heads; GDN q/k/v heads split evenly over TP (16/8, 64/8)
  kernels     vllm-torchtpu: USE_MOE_FUSED_EP_KERNEL / USE_MOE_SPARSE_CORE select the MoE path;
              GDN runs the fused_conv1d_gdn Pallas kernels (kernels/gdn/v3), attention the ragged
              paged attention (RPA) kernels under `--attention-backend CUSTOM`
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from core.deploy import Deploy, Facts

MODEL_DIR = "Qwen3.5-397B-A17B-FP8"
MODEL_TYPES = ("qwen3_5_moe", )

#: Layer kind per `layer_types` entry (the model's own names, shortened).
KIND = {"linear_attention": "gdn", "full_attention": "attn"}
#: Twin layer stack: one layer of each kind, in served order (layer 0 is GDN, layer 3 attention).
TWIN_ORDER = ("gdn", "attn")


def _text(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg


def _layer_types(t: dict) -> list[str]:
    lt = t.get("layer_types")
    if lt:
        return list(lt)
    k = t.get("full_attention_interval", 4)
    return [
        "full_attention" if (i + 1) % k == 0 else "linear_attention"
        for i in range(t["num_hidden_layers"])
    ]


def _gdn_state_bytes(t: dict, tp: int = 1, spec: int = 0) -> int:
    """GDN state per sequence per layer on one TP rank: conv (bf16) + SSM (mamba_ssm_dtype)."""
    kh, vh = t["linear_num_key_heads"], t["linear_num_value_heads"]
    kd, vd = t["linear_key_head_dim"], t["linear_value_head_dim"]
    conv_dim = kd * kh * 2 + vd * vh
    conv = conv_dim // tp * (t["linear_conv_kernel_dim"] - 1 + spec) * 2
    ssm_b = 4 if str(t.get("mamba_ssm_dtype", "float32")) == "float32" else 2
    return conv + vh // tp * vd * kd * ssm_b


#: The checkpoint's text config, kept by facts() for the GDN dims Facts has no field for.
_GDN: dict[str, Any] = {}


def facts(cfg: dict, kv_cache_dtype: str = "fp8") -> Facts:
    if cfg.get("model_type") not in MODEL_TYPES:
        raise ValueError(
            f"{MODEL_DIR}: unexpected model_type {cfg.get('model_type')!r}")
    t = _text(cfg)
    kv_b = 1 if "fp8" in kv_cache_dtype else 2
    lt = _layer_types(t)
    kinds = tuple(KIND[x] for x in lt)
    n_attn = kinds.count("attn")
    hd = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    kv_layer = 2 * t["num_key_value_heads"] * hd * kv_b  # K + V, all KV heads
    mtp = t.get("mtp_num_hidden_layers", 0)
    _GDN.clear()
    _GDN.update({
        k: t[k]
        for k in ("linear_num_key_heads", "linear_num_value_heads",
                  "linear_key_head_dim", "linear_value_head_dim",
                  "linear_conv_kernel_dim")
    })
    _GDN["mamba_ssm_dtype"] = t.get("mamba_ssm_dtype", "float32")
    _GDN["head_dim"] = hd
    return Facts(
        name=MODEL_DIR,
        layers=t["num_hidden_layers"],
        dense_layers=0,
        hidden=t["hidden_size"],
        heads=t["num_attention_heads"],
        kv_heads=t["num_key_value_heads"],
        mla=False,
        experts=t.get("num_experts", 0),
        topk=t.get("num_experts_per_tok", 0),
        expert_ffn=t.get("moe_intermediate_size", 0),
        shared_ffn=t.get("shared_expert_intermediate_size", 0),
        dense_ffn=0,
        kv_cache_bytes_per_token_per_layer=kv_layer,
        mtp_layers=mtp,
        vocab=t.get("vocab_size", 0),
        kinds=kinds,
        kv_bytes_token=kv_layer * n_attn,  # paged KV on attention layers only
        state_bytes_seq=_gdn_state_bytes(t) * kinds.count("gdn"),
        mtp_kv_bytes_token=kv_layer * mtp,  # the MTP layer is full attention
    )


def _kv_heads_rank(f: Facts, d: Deploy) -> int:
    return max(1, f.kv_heads // max(1, d.tp))


def kv_bytes_per_token_per_rank(f: Facts, d: Deploy) -> float:
    """Paged KV one rank holds per token: its KV heads (replicated when TP > KV heads) on every
    attention layer, split by DCP."""
    return f.kv_bytes_token * _kv_heads_rank(f, d) / f.kv_heads / d.dcp


def state_bytes_per_seq_per_rank(f: Facts, d: Deploy) -> float:
    """GDN state per running sequence on one rank, with the conv window widened by the
    speculative tokens (mamba_utils.py:268-272)."""
    n_gdn = f.kinds.count("gdn")
    return _gdn_state_bytes(_GDN, max(1, d.tp), d.spec_tokens) * n_gdn


def per_rank_rows(f: Facts,
                  d: Deploy,
                  pool_tokens: int | None = None) -> dict[str, Any]:
    tp = max(1, d.tp)
    return {
        "attn_kv_heads_per_rank":
        _kv_heads_rank(f, d),
        "attn_kv_replicated":
        tp > f.kv_heads,
        "gdn_k_heads_per_rank":
        _GDN["linear_num_key_heads"] // tp,
        "gdn_v_heads_per_rank":
        _GDN["linear_num_value_heads"] // tp,
        "gdn_state_bytes_per_seq_per_layer_per_rank":
        _gdn_state_bytes(_GDN, tp, d.spec_tokens),
        "kv_bytes_per_token_per_rank_gqa":
        round(kv_bytes_per_token_per_rank(f, d), 1),
    }


#: Per-rank rows (beyond core's generic ones) the twin check compares, real vs twin.
PROJECTION_ROWS = [
    "attn_kv_heads_per_rank", "gdn_k_heads_per_rank", "gdn_v_heads_per_rank",
    "gdn_state_bytes_per_seq_per_layer_per_rank"
]

#: Served flags the twin must carry: without CUSTOM vLLM falls back to FLASH_ATTN; the hybrid
#: prefix cache needs mamba-cache-mode align; the checkpoint is a VL model and the twin runner
#: refuses multimodal inputs, so the vision tower must not be built.
TWIN_CARRY_ARGS = [
    "attention-backend", "mamba-cache-mode", "language-model-only",
    "quantization"
]

#: How to read this model's profile (carried in every plan; used by xprof_steps / evidence).
#:  The served backbone step of j-9fb52480 has 121 sync all-reduces = 1 (vocab-parallel
#:  embedding) + 2 x 60 (attention / GDN out-projection, MoE output); each MTP draft forward 3.
#:  census families (op names from that profile, per step): fused_conv1d_gdn_* 45x (GDN),
#:  RPAd/RPAm 15x (ragged paged attention), gmm_v2 / router_topk 60x (MoE).
#:  backbone_by_all_reduces: the MTP draft forward compiles to the same XLA module name as the
#:  backbone (tt_jit_layernorm_..._fused_add_rms_norm_add in that profile), so a backbone step
#:  is told apart by its all-reduce count, not its name.
#:  boundary_regex: decode steps lower the boundary all-reduce to a sync `all-reduce.N`; a
#:  16,384-token step lowers the same all-reduce to an async `...cloned.1.call-start/-done`
#:  pair (twin j-5bddc288: 5 call-done per mixed step = 1 + 2 x 2 layers).
TRACE = {
    "embed_all_reduces":
    1,
    "all_reduces_per_layer":
    2,
    "backbone_by_all_reduces":
    True,
    "boundary_regex":
    r"^all-reduce\.\d+(?:\.cloned\.\d+\.call-done)?(?:\.refs_\d+_)*$",
    "census": {
        "families": {
            "gdn": {
                "regex": r"fused_conv1d_gdn",
                "part": "attention"
            },
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
            "gdn": ["gdn", "experts"],
            "attn": ["attn", "experts"]
        },
    },
    # Ops the served step runs ONCE although the twin places them inside a layer (the twin's
    # first use of the attention metadata / RoPE cache falls in its only gdn or attn layer).
    # Served j-9fb52480 decode b1 @ ~140k (its served-trace decomposition):
    # the RoPE cos/sin cache copy `copy bf16[262144,64]` once in L3, `rpa_metadata_schedule`
    # s32[12288] + s32[16384] and `copy-done s32[1277952]` once in the before segment; twin
    # j-b8df7060 / j-1393feab put them in the attn (decode) or gdn (mixed) layer. The mixed
    # step's mRoPE position gather `gather_offload_async_done bf16[49152,64]` (3 x 16,384
    # tokens x rotary 64) sits in the twin's attn layer. Counted once on the served basis
    # (xprof_steps.project), not x 45 / x 15. 1-D s32 copy-dones are the metadata H2D copies;
    # the MoE route tensors (s32[163840] on mixed) move by gather_offload, not copy-done.
    "per_step_ops": [
        {
            "what": "RoPE cos/sin cache copy",
            "op": r"^copy$",
            "out": r"^bf16\[\d{5,},64\]$"
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
        {
            "what": "mRoPE position gather",
            "op": r"^gather_offload_async_done$",
            "out": r"^bf16\[\d+,64\]$"
        },
    ],
}

#: Router parameters re-drawn on dummy weights (Qwen3NextSparseMoeBlock.gate, ReplicatedLinear,
#: no correction bias), so tokens spread over experts as real ones do.
ROUTER = {"weight_suffix": "mlp.gate.weight", "zero_suffixes": [], "std": 0.05}

#: The sampling the served job ran with when its snapshot does not say: the checkpoint's
#: generation_config.json, which vLLM applies as the default (server log j-9fb52480,
#: vllm-server.log 10-01 18:38:58: "Default vLLM sampling parameters have been overridden by the
#: model's `generation_config.json`: `{'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}`").
SAMPLING = {
    "temperature":
    0.6,
    "top_k":
    20,
    "top_p":
    0.95,
    "source":
    "model.py default: the checkpoint's generation_config.json (logged by "
    "j-9fb52480's server)"
}

#: The MTP drafter's replay in vllm-torchtpu (spec_decode/eagle3.py @ 6eb383e4c: propose :1029,
#: _dp_lockstep_sharded :1882, run_dp_dummy_draft :1991).
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
        if k.startswith(("TPU_MOE", "USE_MOE", "ONEHOT_MOE", "MOE_",
                         "ENABLE_QUANTIZED", "TPU_ROPE", "TPU_GDN",
                         "TPU_ENABLE_GDN", "TPU_RPA", "VLLM_KV_CACHE_LAYOUT",
                         "VLLM_TPU_FUSED", "TPU_TOKEN_BUCKET"))
    }
    return {
        "gdn":
        "fused_conv1d_gdn (kernels/gdn/v3)" +
        (", dynamic tiling"
         if env.get("TPU_ENABLE_GDN_DYNAMIC_TILING") == "1" else ""),
        "attention":
        "ragged paged attention (CUSTOM)",
        "moe":
        moe,
        "linear":
        "block"
        if env.get("ENABLE_QUANTIZED_MATMUL_KERNEL") == "1" else "per_channel",
        "env":
        knobs,
    }


def _remap_not_convert(ckpt: dict, f: Facts, kinds: list[str]) -> dict | None:
    """The fp8 skip-list names layers by index (`model.language_model.layers.<i>.`). Give twin
    layer i the entries of the first real layer of the same kind, so routers, the shared-expert
    gate and the GDN conv1d / in_proj_a / in_proj_b keep their served dtype. `mtp.*` entries are
    not indexed by backbone layer and are kept as they are."""
    import re
    q = ckpt.get("quantization_config") or {}
    skip = q.get("modules_to_not_convert")
    if not skip:
        return None
    first: dict[str, int] = {}
    for i, k in enumerate(f.kinds):
        first.setdefault(k, i)
    pat = re.compile(r"(model\.language_model\.layers\.)(\d+)\.")
    out = [m for m in skip if not pat.match(m)]
    for twin_i, k in enumerate(kinds):
        pre = f"model.language_model.layers.{first[k]}."
        out += [
            f"model.language_model.layers.{twin_i}." + m[len(pre):]
            for m in skip if m.startswith(pre)
        ]
    return {**q, "modules_to_not_convert": sorted(set(out))}


def twin_overrides(f: Facts,
                   real: Deploy,
                   twin_world: int,
                   pool_tokens: int | None = None,
                   ckpt: dict | None = None) -> dict[str, Any]:
    """One layer per kind, served per-rank shapes. Only the served world is supported (mirror):
    the source jobs run TP8 on one 2x2x1 host, which is also the small slice, so scaled and
    mirror coincide. A wider served world would need GDN heads, attention heads and experts
    scaled together (and GDN heads must stay whole per rank); not built."""
    if twin_world != real.world:
        raise ValueError(
            f"{MODEL_DIR}: twin world {twin_world} != served world {real.world}; "
            "only the served world (mirror) is supported for this model")
    kinds = [k for k in TWIN_ORDER if k in f.layer_kinds]
    inv = {v: k for k, v in KIND.items()}
    hf: dict[str, Any] = {
        "text_config": {
            "num_hidden_layers": len(kinds),
            "layer_types": [inv[k] for k in kinds]
        }
    }
    qc = _remap_not_convert(ckpt or {}, f, kinds)
    if qc:
        hf["quantization_config"] = qc
    server: dict[str, Any] = {
        "tensor-parallel-size": real.tp,
        "data-parallel-size": real.dp,
        "decode-context-parallel-size": real.dcp,
        # the served page (the runtime's resolved size, see core/derive.py); passed as the floor
        # it re-resolves to the same size, since per-rank GDN state and KV bytes are unchanged
        "block-size": real.block_size,
        "max-model-len": real.max_model_len,
        "load-format": "dummy",
    }
    deviations = []
    if pool_tokens:
        server[
            "num-gpu-blocks-override"] = pool_tokens // real.dcp // real.block_size
        deviations.append({
            "what":
            "KV pool blocks",
            "real":
            "num_blocks from the server log (vLLM's reported pool tokens exclude the "
            "hybrid padding layers)",
            "twin":
            f"{server['num-gpu-blocks-override']} = reported pool tokens / page",
            "covered_by":
            "none: distinct pages per request, dummy KV values; no whole-pool op on this model's step path"
        })
    deviations.append({
        "what":
        "GDN recurrent state",
        "real":
        "per-request slots",
        "twin":
        "distinct non-null slot per request (forward_run _mamba_slots); "
        "dummy state values",
        "covered_by":
        "state read/write cost per request is the same; values differ"
    })
    return {
        "twin_world":
        twin_world,
        "twin_deploy":
        real.label(),
        "layers":
        kinds,
        "hf_overrides":
        hf,
        "server_args":
        server,
        "context_scale":
        1.0,
        "deviations":
        deviations,
        "problems": [],
        "verify_before_boot": [
            "hf_overrides.text_config.layer_types honoured (layer census: 1 gdn + 1 attn)",
            f"resolved page size {real.block_size} logged again by tpu_block_size_utils",
            "load-format dummy + language-model-only boot through vllm-torchtpu",
        ],
    }


def twin_facts(f: Facts, hf: dict) -> Facts:
    t = hf["text_config"]
    kinds = tuple(KIND[x] for x in t["layer_types"])
    return replace(
        f,
        layers=t["num_hidden_layers"],
        kinds=kinds,
        kv_bytes_token=f.kv_cache_bytes_per_token_per_layer *
        kinds.count("attn"),
        state_bytes_seq=(f.state_bytes_seq // max(1, f.kinds.count("gdn"))) *
        kinds.count("gdn"))


def param_bytes_per_rank(f: Facts,
                         d: Deploy,
                         layers: dict[str, int] | None = None) -> dict:
    """Estimated resident weight bytes on one rank (fp8 weights; bf16 embeddings, LM head,
    routers). TP-sharded: attention q/k/v/o (KV heads replicated past 2), GDN in/out
    projections, shared expert, embeddings and LM head; EP-sharded: routed experts.
    Replicated: routers and the shared-expert gate (small)."""
    kinds = layers or f.layer_kinds
    D, tp = f.hidden, max(1, d.tp)
    hd = _GDN["head_dim"]
    attn = (
        D * f.heads * hd * 2  # q + output gate
        + 2 * D * _kv_heads_rank(f, d) * hd * tp  # k, v (per rank x tp)
        + f.heads * hd * D) / tp
    kh, vh = _GDN["linear_num_key_heads"], _GDN["linear_num_value_heads"]
    kd, vd = _GDN["linear_key_head_dim"], _GDN["linear_value_head_dim"]
    gdn = (D * (2 * kh * kd + 2 * vh * vd) +
           vh * vd * D) / tp  # in_proj qkvz + out_proj
    moe = (f.experts / max(1, d.ep_group)) * 3 * D * f.expert_ffn + 3 * D * f.shared_ffn / tp \
        + 2 * D * f.experts
    per = {"gdn": gdn + moe, "attn": attn + moe}
    layer_b = sum(per[k] * n for k, n in kinds.items())
    mtp_b = f.mtp_layers * (per["attn"] + 2 * D * D)  # + fc (2D -> D)
    emb_b = 2 * 2 * f.vocab * D / tp  # embed + lm_head, bf16
    return {
        "layers": int(layer_b),
        "mtp": int(mtp_b),
        "embed_head": int(emb_b),
        "total": int(layer_b + mtp_b + emb_b)
    }


#: Exact-shape component benchmarks for the twin's deviations. Mirror mode has none from
#: sharding; declared so the report can say so.
COMPONENTS = {}
