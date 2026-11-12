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
"""Sharding nomenclature adapter, roofline helpers, and HBM capacity estimator.

Provides bidirectional translation, validation, and HBM footprint checks for:
1. Shortnames and canonical mesh keys (`tp`, `dp`, `ep`, `sp`, `fsdp`, `cp`,
   `pcp`, `dcp`, `pp`, `dpl`, `tensor`, `data`, `expert`, `sequence`,
   `pipeline`, `data_local`, `model`, `model_q`, `model_kv`, `seq`, `replica`).
2. MaxText ICI parallelism configuration (`ici_*_parallelism`).
3. vLLM TPU serving CLI flags (`--tensor-parallel-size`, `--data-parallel-size`,
   `--expert-parallel-size`, `--prefill-context-parallel-size`,
   `--decode-context-parallel-size`, `--pipeline-parallel-size`,
   `--data-parallel-size-local`).
4. Dense and MoE (including Multi-Head Latent Attention, shared experts, and
   leading dense layers) HBM memory footprint and roofline ridge-point helpers.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import tpu_specs  # noqa: E402

# Usable fraction of nameplate HBM; the remainder covers XLA runtime scratch,
# collectives, and allocator fragmentation.
HBM_UTILIZATION_SAFETY_FACTOR = 0.85

CANONICAL_SHARDING_MAP: dict[str, str] = {
    # vLLM / MaxText / OSS shortnames & fullnames
    "tp": "tensor",
    "tensor": "tensor",
    "tensor_parallel": "tensor",
    "tensor_parallelism": "tensor",
    "ici_tensor_parallelism": "tensor",
    "tensor_parallel_size": "tensor",
    "dp": "data",
    "data": "data",
    "data_parallel": "data",
    "data_parallelism": "data",
    "ici_data_parallelism": "data",
    "data_parallel_size": "data",
    "fsdp": "fsdp",
    "fsdp_parallel": "fsdp",
    "fsdp_parallelism": "fsdp",
    "ici_fsdp_parallelism": "fsdp",
    "ici_fsdp_transpose_parallelism": "fsdp",
    "ep": "expert",
    "expert": "expert",
    "expert_parallel": "expert",
    "expert_parallelism": "expert",
    "ici_expert_parallelism": "expert",
    "expert_parallel_size": "expert",
    "sp": "sequence",
    "sequence": "sequence",
    "sequence_parallel": "sequence",
    "sequence_parallelism": "sequence",
    "ici_sequence_parallelism": "sequence",
    "sequence_parallel_size": "sequence",
    "pcp": "prefill_context",
    "prefill_context": "prefill_context",
    "prefill_context_parallel": "prefill_context",
    "prefill_context_parallel_size": "prefill_context",
    "dcp": "decode_context",
    "decode_context": "decode_context",
    "decode_context_parallel": "decode_context",
    "decode_context_parallel_size": "decode_context",
    "cp": "context",
    "context": "context",
    "context_parallel": "context",
    "context_parallelism": "context",
    "ici_context_parallelism": "context",
    "pp": "pipeline",
    "pipeline": "pipeline",
    "pipeline_parallel": "pipeline",
    "pipeline_parallelism": "pipeline",
    "pipeline_parallel_size": "pipeline",
    "dpl": "data_local",
    "data_local": "data_local",
    "data_parallel_local": "data_local",
    "data_parallel_size_local": "data_local",
    "autoregressive": "data",
    "ici_autoregressive_parallelism": "data",
    # JAX trainer mesh keys (including split query / KV tensor axes)
    "model": "tensor",
    "model_q": "tensor_q",
    "model_kv": "tensor_kv",
    "replica": "data",
    "data2": "data",
    "model_expert": "expert",
    "seq": "sequence",
}

# Canonical configuration attribute aliases across HuggingFace, MaxText, and JAX.
EMBEDDING_DIM_ALIASES: tuple[str, ...] = (
    "d_model",
    "hidden_size",
    "emb_dim",
    "base_emb_dim",
    "model_dim",
)

FFN_DIM_ALIASES: tuple[str, ...] = (
    "d_ff",
    "intermediate_size",
    "ffn_hidden_size",
    "mlp_dim",
    "base_mlp_dim",
    "moe_mlp_dim",
    "base_moe_mlp_dim",
    "ffw_hidden",
)

# MoE configs carry two feed-forward widths: the dense `intermediate_size` and
# the narrower per-expert `moe_intermediate_size`. Consult these first for
# routed expert blocks; configs publishing only the dense width fall back.
MOE_FFN_DIM_ALIASES: tuple[str, ...] = (
    "moe_intermediate_size",
    "moe_ffn_hidden_size",
    "expert_intermediate_size",
    "moe_mlp_dim",
    "base_moe_mlp_dim",
)

# Shared (always-on) experts are replicated on every expert-parallel rank rather
# than partitioned across them, so they are costed like a dense MLP.
SHARED_EXPERTS_ALIASES: tuple[str, ...] = (
    "n_shared_experts",
    "num_shared_experts",
    "shared_experts",
    "moe_num_shared_experts",
)

# Leading layers that keep a dense MLP instead of an MoE block; these use the
# dense feed-forward width, not the expert width.
DENSE_LAYER_COUNT_ALIASES: tuple[str, ...] = (
    "first_k_dense_replace",
    "num_dense_layers",
    "n_dense_layers",
)

HEADS_ALIASES: tuple[str, ...] = (
    "num_heads",
    "num_attention_heads",
    "num_query_heads",
    "base_num_query_heads",
    "model_q",
)

KV_HEADS_ALIASES: tuple[str, ...] = (
    "num_kv_heads",
    "num_key_value_heads",
    "multi_query_group_num",
    "base_num_kv_heads",
    "model_kv",
)

# Per-head dimension. Most HuggingFace configs omit this and derive it as
# `hidden_size / num_attention_heads`; MLA configs split it into
# `qk_nope_head_dim` + `qk_rope_head_dim` and `v_head_dim` instead.
HEAD_DIM_ALIASES: tuple[str, ...] = (
    "head_dim",
    "head_size",
    "attention_head_dim",
    "qk_head_dim",
    "kv_channels",
)

SEQ_LEN_ALIASES: tuple[str, ...] = (
    "max_model_len",
    "sequence_length",
    "seq_length",
    "seq_len",
    "max_sequence_length",
    "max_position_embeddings",
    "max_target_length",
    "context_length",
)

LAYERS_ALIASES: tuple[str, ...] = (
    "num_layers",
    "num_hidden_layers",
    "num_decoder_layers",
    "base_num_decoder_layers",
)

# Unambiguously global: DP splits these, so a per-rank consumer divides by DP.
EXPLICIT_GLOBAL_BATCH_SIZE_ALIASES: tuple[str, ...] = (
    "global_batch_size",
    "train_batch_size",
    "eval_batch_size",
)

# `batch_size` is ambiguous: MaxText means global batch, vLLM means
# `max_num_seqs` (per-engine). The DP divisibility rule reads it as global;
# the per-rank KV-cache term reads it as per-rank.
AMBIGUOUS_BATCH_SIZE_ALIASES: tuple[str, ...] = ("batch_size", )

GLOBAL_BATCH_SIZE_ALIASES: tuple[str, ...] = (
    "global_batch_size",
    "batch_size",
    "train_batch_size",
    "eval_batch_size",
)

PER_DEVICE_BATCH_SIZE_ALIASES: tuple[str, ...] = ("per_device_batch_size", )

BATCH_SIZE_ALIASES: tuple[str, ...] = (GLOBAL_BATCH_SIZE_ALIASES +
                                       PER_DEVICE_BATCH_SIZE_ALIASES)

MM_EMBED_DIM_ALIASES: tuple[str, ...] = (
    "mm_embed_dim",
    "multimedia_embed_dim",
    "mm_hidden_size",
    "image_dim",
)

AUDIO_EMBED_DIM_ALIASES: tuple[str, ...] = (
    "audio_embed_dim",
    "audio_hidden_size",
)

NUM_EXPERTS_ALIASES: tuple[str, ...] = (
    "num_experts",
    "n_routed_experts",
    "num_local_experts",
    "num_routed_experts",
    "total_experts",
    "moe_num_experts",
    "n_experts",
    "experts",
)

NUM_EXPERTS_PER_TOK_ALIASES: tuple[str, ...] = (
    "num_experts_per_datapoint",
    "num_experts_per_tok",
    "num_experts_per_token",
    "num_selected_experts",
    "top_k_experts",
)

VOCAB_SIZE_ALIASES: tuple[str, ...] = (
    "vocab_size",
    "padded_vocab_size",
    "vocabulary_size",
    "vocab",
)


class ShardingNomenclatureAdapter:
    """Adapter to parse, validate, and convert sharding proposals."""

    @classmethod
    def detect_workload_type(cls, model_cfg: Mapping[str, Any]) -> str:
        """Detects whether a configuration targets a trainer or serving runtime."""
        engine = cls.detect_runtime_engine(model_cfg)
        if engine in ("jax", "maxtext", "pytorch_xla"):
            return "trainer"
        return "serving"

    # Keys that only appear in a training configuration.
    TRAINING_ONLY_KEYS = (
        "opt_type",
        "optimizer",
        "learning_rate",
        "adam_b1",
        "adam_b2",
        "adam_eps",
        "weight_decay",
        "gradient_accumulation_steps",
        "grad_accum_steps",
        "warmup_steps",
        "remat_policy",
    )

    @classmethod
    def detect_training_or_inference(
            cls, model_cfg: Mapping[str, Any] | None) -> str:
        """Returns 'training' or 'inference' for a model configuration.

        Defaults to 'inference' when there is no positive evidence of training,
        so that an unannotated config is never charged for optimizer state it
        does not hold.
        """
        if not model_cfg:
            return "inference"
        explicit = (str(
            model_cfg.get("workload_type") or model_cfg.get("workload")
            or "").strip().lower())
        if explicit in ("training", "train", "pretrain", "finetune", "sft"):
            return "training"
        if explicit in ("inference", "serving", "serve", "decode", "prefill"):
            return "inference"
        if cls.detect_runtime_engine(model_cfg) in ("vllm", "sglang"):
            return "inference"
        if any(key in model_cfg for key in cls.TRAINING_ONLY_KEYS):
            return "training"
        return "inference"

    @classmethod
    def optimizer_state_bytes_per_param(
            cls, model_cfg: Mapping[str, Any] | None) -> float:
        """Bytes of persistent optimizer state held per (sharded) parameter."""
        opt = (str((model_cfg or {}).get("opt_type")
                   or (model_cfg or {}).get("optimizer")
                   or "adamw").strip().lower())
        if "adafactor" in opt:
            return 4.0
        if opt in ("sgd", "momentum", "nesterov"):
            return 4.0
        # Adam / AdamW: fp32 master weights + first moment + second moment
        # = 3 x 4 bytes per parameter.
        return 12.0

    @classmethod
    def detect_runtime_engine(
        cls,
        model_cfg: Mapping[str, Any],
        profile_path: str | None = None,
    ) -> str:
        """Detects runtime engine ('vllm', 'sglang', 'maxtext', 'jax', 'pytorch_xla', 'pallas')."""
        for key in ("library", "runtime_engine", "serving_framework",
                    "framework"):
            val = model_cfg.get(key)
            if val and isinstance(val, str):
                clean_val = val.strip().lower()
                if clean_val in (
                        "vllm",
                        "sglang",
                        "maxtext",
                        "jax",
                        "pytorch_xla",
                        "pallas",
                ):
                    return clean_val

        if profile_path and isinstance(profile_path, str):
            clean_path = profile_path.lower()
            if "vllm" in clean_path or "tpu_inference" in clean_path:
                return "vllm"
            if "sglang" in clean_path:
                return "sglang"
            if "maxtext" in clean_path:
                return "maxtext"
            if "torch_xla" in clean_path or "pytorch" in clean_path:
                return "pytorch_xla"
            if "jax" in clean_path:
                return "jax"

        if "max_model_len" in model_cfg or "kv_cache_dtype" in model_cfg:
            return "vllm"
        if "per_device_batch_size" in model_cfg or "ici_mesh_shape" in model_cfg:
            return "maxtext"

        return "vllm"

    @classmethod
    def is_mla_model(cls, model_cfg: Mapping[str, Any] | None) -> bool:
        """Detects whether a model configuration uses Multi-Head Latent Attention.

        Detection is structural first (`kv_lora_rank > 0` or both
        `qk_rope_head_dim > 0` and `qk_nope_head_dim > 0`), falling back to
        explicit `attention_type` or family metadata (`deepseek`, `mla`).
        """
        if not isinstance(model_cfg, Mapping):
            return False

        def _positive_int(key: str) -> bool:
            raw = model_cfg.get(key)
            if raw is None:
                return False
            try:
                return int(raw) > 0
            except (TypeError, ValueError):
                return False

        if _positive_int("kv_lora_rank"):
            return True
        if _positive_int("qk_rope_head_dim") and _positive_int(
                "qk_nope_head_dim"):
            return True

        attn_type = str(model_cfg.get("attention_type", "")).lower()
        if "mla" in attn_type or "latent" in attn_type:
            return True
        for key in (
                "model",
                "model_name",
                "model_family",
                "name",
                "architecture",
                "model_type",
        ):
            val = str(model_cfg.get(key, "")).lower()
            if "deepseek" in val or "mla" in val:
                return True
        return False

    @classmethod
    def resolve_accelerator_type(cls,
                                 model_cfg: Mapping[str, Any],
                                 fallback: str = "tpu7x-8") -> str:
        """Resolves accelerator string across model config fields."""
        return resolve_accelerator_type(model_cfg, fallback=fallback)

    @classmethod
    def get_cores_per_chip(cls, tpu_type: str = "tpu7x-8") -> int:
        """Returns number of logical XLA devices (TensorCores) per physical chip."""
        devices = tpu_specs.get_devices_per_chip(tpu_type)
        return devices if devices is not None else 1

    @classmethod
    def get_target_mesh_capacity(cls, model_cfg: Mapping[str, Any],
                                 total_tpu_chips: int) -> tuple[int, str]:
        """Calculates total mesh capacity based on runtime engine and hardware."""
        engine = cls.detect_runtime_engine(model_cfg)
        tpu_type = cls.resolve_accelerator_type(model_cfg, fallback="tpu7x-8")
        cores_per_chip = cls.get_cores_per_chip(tpu_type)

        capacity = total_tpu_chips * cores_per_chip
        desc = (
            f"{engine.title()} Logical TensorCore Mesh ({total_tpu_chips} chips x"
            f" {cores_per_chip} cores/chip = {capacity} cores)")
        return capacity, desc

    CANONICAL_TO_TRAINER: dict[str, str] = {
        "tensor": "model",
        "tensor_q": "model_q",
        "tensor_kv": "model_kv",
        "data": "data",
        "expert": "expert",
        "sequence": "seq",
        "pipeline": "pipeline",
        "fsdp": "fsdp",
        "data_local": "data_local",
        "prefill_context": "prefill_context",
        "decode_context": "decode_context",
        "context": "context",
    }

    CANONICAL_TO_OSS: dict[str, str] = {
        "tensor": "tp",
        "data": "dp",
        "expert": "ep",
        "sequence": "sp",
        "pipeline": "pp",
        "fsdp": "fsdp",
        "data_local": "dpl",
        "prefill_context": "pcp",
        "decode_context": "dcp",
        "context": "cp",
    }

    @classmethod
    def derived_expert_parallelism(cls, standardized: Mapping[str,
                                                              int]) -> int:
        """Returns the EP degree vLLM derives for a given mesh.

        vLLM computes `flatten_tp_size = dp_size * pcp_size * tp_size` and, when
        expert parallelism is enabled, sets `tp_size=1, ep_size=flatten_tp_size`.
        EP therefore never adds devices to the mesh, and neither does
        `decode_context_parallel_size` (DCP), which reuses TP (or PCP / TPxPCP)
        ranks.
        """
        tensor = (standardized["tensor"] * standardized["tensor_q"] *
                  standardized["tensor_kv"])
        return (tensor * standardized["prefill_context"] *
                standardized["data"] * standardized["fsdp"] *
                standardized["data_local"])

    @classmethod
    def vllm_dcp_error(cls, standardized: Mapping[str, int]) -> str | None:
        """Returns vLLM's objection to the mesh's DCP size, or None if valid.

        Mirrors `ParallelConfig` (`vllm/config/parallel.py`): without PCP, DCP
        reuses TP ranks and needs `tp_size % dcp_size == 0`; with PCP, DCP must
        be 1, `pcp`, or `tp * pcp`.
        """
        tensor = (standardized["tensor"] * standardized["tensor_q"] *
                  standardized["tensor_kv"])
        pcp = standardized["prefill_context"]
        dcp = standardized["decode_context"]
        if dcp == 1:
            return None
        if pcp == 1:
            if tensor % dcp == 0:
                return None
            return (
                f"Decode Context Parallelism DCP={dcp} must divide TP={tensor}"
                " in vLLM: without PCP, DCP reuses the TP ranks.")
        if dcp in (pcp, tensor * pcp):
            return None
        return (
            f"Decode Context Parallelism DCP={dcp} is not valid with PCP={pcp}"
            f" in vLLM: DCP must be 1, PCP ({pcp}) or TP x PCP"
            f" ({tensor * pcp}).")

    @classmethod
    def compute_mesh_product(cls, standardized: Mapping[str, int],
                             engine: str) -> int:
        """Returns the number of logical devices a standardized mesh occupies.

        vLLM: world size is `tp * dp * pcp * pp` (`dcp` reuses TP/PCP ranks and
        `ep` is derived as `tp * dp * pcp`).
        MaxText / JAX / PyTorch-XLA: orthogonal product across all independent
        mesh axes.
        """
        effective_dp = (standardized["data"] * standardized["fsdp"] *
                        standardized["data_local"])
        context = max(standardized["prefill_context"],
                      standardized["decode_context"])
        tensor = (standardized["tensor"] * standardized["tensor_q"] *
                  standardized["tensor_kv"])
        if engine == "vllm":
            return (tensor * standardized["prefill_context"] * effective_dp *
                    standardized["pipeline"])
        return (tensor * standardized["sequence"] * context *
                standardized["pipeline"] * effective_dp *
                standardized["expert"])

    @classmethod
    def standardize_proposal(
            cls, raw_proposal: Mapping[str, int]) -> dict[str, int]:
        """Standardizes a raw sharding proposal dict into canonical keys."""
        canonical: dict[str, int] = {
            "tensor": 1,
            "data": 1,
            "fsdp": 1,
            "expert": 1,
            "sequence": 1,
            "prefill_context": 1,
            "decode_context": 1,
            "pipeline": 1,
            "data_local": 1,
            "tensor_q": 1,
            "tensor_kv": 1,
        }
        for k, v in raw_proposal.items():
            clean_k = str(k).strip().lower()
            if clean_k in CANONICAL_SHARDING_MAP:
                canonical_key = CANONICAL_SHARDING_MAP[clean_k]
                if canonical_key == "context":
                    canonical["prefill_context"] = int(v)
                    canonical["decode_context"] = int(v)
                else:
                    canonical[canonical_key] = int(v)
            elif clean_k in canonical:
                canonical[clean_k] = int(v)
        return canonical

    @classmethod
    def get_config_attribute(
        cls,
        config: Mapping[str, Any] | None,
        aliases: tuple[str, ...],
        default: Any = None,
    ) -> Any:
        """Retrieves a configuration value checking multiple framework aliases."""
        if not isinstance(config, Mapping):
            return default
        for key in aliases:
            if key in config and config[key] is not None:
                val = config[key]
                if isinstance(val, Mapping):
                    nested_val = cls.get_config_attribute(val, aliases)
                    if nested_val is not None:
                        return nested_val
                    if len(val) == 1:
                        return next(iter(val.values()))
                return val
        return default

    @classmethod
    def safe_int(
        cls,
        val: Any,
        aliases: tuple[str, ...] | None = None,
        default: int | None = None,
    ) -> int | None:
        """Safely extracts an integer from a scalar or nested dictionary."""
        if val is None:
            return default
        if isinstance(val, Mapping):
            if aliases:
                nested_attr = cls.get_config_attribute(val, aliases)
                if nested_attr is not None:
                    res = cls.safe_int(nested_attr, aliases)
                    if res is not None:
                        return res
            if len(val) == 1:
                res = cls.safe_int(next(iter(val.values())), aliases)
                if res is not None:
                    return res
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    @classmethod
    def get_config_int(
        cls,
        config: Mapping[str, Any] | None,
        aliases: tuple[str, ...],
        default: int | None = None,
    ) -> int | None:
        """Retrieves and safely converts a config attribute to an integer."""
        raw_val = cls.get_config_attribute(config, aliases)
        if raw_val is None:
            return default
        return cls.safe_int(raw_val, aliases, default=default)

    @classmethod
    def normalize_model_config(
            cls, model_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
        """Normalizes model config keys across HuggingFace, MaxText, and JAX."""
        if not isinstance(model_cfg, Mapping):
            return {}
        normalized = dict(model_cfg)

        int_d_model = cls.get_config_int(normalized, EMBEDDING_DIM_ALIASES)
        if int_d_model is not None:
            normalized["d_model"] = int_d_model
            normalized["hidden_size"] = int_d_model

        int_d_ff = cls.get_config_int(normalized, FFN_DIM_ALIASES)
        if int_d_ff is not None:
            normalized["d_ff"] = int_d_ff
            normalized["intermediate_size"] = int_d_ff

        int_heads = cls.get_config_int(normalized, HEADS_ALIASES)
        if int_heads is not None:
            normalized["num_heads"] = int_heads

        int_kv_heads = cls.get_config_int(normalized, KV_HEADS_ALIASES)
        if int_kv_heads is not None:
            normalized["num_kv_heads"] = int_kv_heads

        int_layers = cls.get_config_int(normalized, LAYERS_ALIASES)
        if int_layers is not None:
            normalized["num_layers"] = int_layers

        int_batch_size = cls.get_config_int(normalized, BATCH_SIZE_ALIASES)
        if int_batch_size is not None:
            normalized["batch_size"] = int_batch_size

        int_mm_embed = cls.get_config_int(normalized, MM_EMBED_DIM_ALIASES)
        if int_mm_embed is not None:
            normalized["mm_embed_dim"] = int_mm_embed

        int_audio_embed = cls.get_config_int(normalized,
                                             AUDIO_EMBED_DIM_ALIASES)
        if int_audio_embed is not None:
            normalized["audio_embed_dim"] = int_audio_embed

        int_num_experts = cls.get_config_int(normalized, NUM_EXPERTS_ALIASES)
        if int_num_experts is not None:
            normalized["num_experts"] = int_num_experts

        int_experts_per_tok = cls.get_config_int(normalized,
                                                 NUM_EXPERTS_PER_TOK_ALIASES)
        if int_experts_per_tok is not None:
            normalized["num_experts_per_tok"] = int_experts_per_tok

        int_vocab_size = cls.get_config_int(normalized, VOCAB_SIZE_ALIASES)
        if int_vocab_size is not None:
            normalized["vocab_size"] = int_vocab_size

        return normalized

    @classmethod
    def normalize_to_internal(
        cls, mesh_dict: Mapping[str, Any] | None
    ) -> tuple[dict[str, Any] | None, str]:
        """Converts OSS or trainer mesh dict to normalized trainer representation."""
        if not isinstance(mesh_dict, Mapping):
            return mesh_dict, "TRAINER"
        normalized: dict[str, Any] = {}
        detected_style = "TRAINER"
        for k, v in mesh_dict.items():
            k_lower = str(k).strip().lower()
            if isinstance(v, (int, float)) or (isinstance(v, str)
                                               and v.isdigit()):
                val: Any = int(v)
            else:
                val = v

            if k_lower in CANONICAL_SHARDING_MAP:
                canonical_key = CANONICAL_SHARDING_MAP[k_lower]
                internal_key = cls.CANONICAL_TO_TRAINER.get(
                    canonical_key, canonical_key)
                normalized[internal_key] = val
                if k_lower in (
                        "tp",
                        "dp",
                        "ep",
                        "sp",
                        "pp",
                        "cp",
                        "pcp",
                        "dcp",
                        "fsdp",
                        "dpl",
                ):
                    detected_style = "OSS"
            else:
                normalized[k_lower] = val
        return normalized, detected_style

    @classmethod
    def format_to_style(cls, normalized_mesh: Mapping[str, Any],
                        target_style: str) -> dict[str, Any]:
        """Formats normalized mesh into `'TRAINER'` or `'OSS'` key style."""
        if not isinstance(normalized_mesh, Mapping):
            return {}
        if target_style.upper() == "TRAINER":
            return dict(normalized_mesh)
        out: dict[str, Any] = {}
        for k, v in normalized_mesh.items():
            k_lower = str(k).strip().lower()
            canonical_key = CANONICAL_SHARDING_MAP.get(k_lower, k_lower)
            target_key = cls.CANONICAL_TO_OSS.get(canonical_key, k)
            out[target_key] = v
        return out

    @classmethod
    def validate_feasibility(
        cls,
        proposal: Mapping[str, int],
        total_tpu_chips: int,
        model_cfg: Mapping[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Validates whether the sharding dimension product matches device capacity."""
        standardized = cls.standardize_proposal(proposal)
        model_config = model_cfg or {}
        engine = cls.detect_runtime_engine(model_config)
        mesh_product = cls.compute_mesh_product(standardized, engine)

        expected_capacity, desc = cls.get_target_mesh_capacity(
            model_config, total_tpu_chips)

        if mesh_product != expected_capacity:
            return (
                False,
                (f"Infeasible sharding product: {mesh_product} does not match"
                 f" expected {desc} capacity {expected_capacity} (mesh:"
                 f" {standardized})"),
            )

        for k, val in standardized.items():
            if val < 1:
                return False, f"Invalid dimension value for {k}: {val} < 1"
            if (val & (val - 1)) != 0:
                return False, f"Non-power-of-two sharding dimension for {k}: {val}"

        return True, f"Valid sharding mesh for {desc}"

    @classmethod
    def to_trainer_mesh(cls, proposal: Mapping[str, int]) -> dict[str, int]:
        """Converts proposal into standard JAX trainer mesh keys."""
        std = cls.standardize_proposal(proposal)
        trainer_mesh: dict[str, int] = {
            "replica": 1,
            "data": std["data"] * std["fsdp"] * std["data_local"],
            "seq": std["sequence"],
        }
        if std["tensor_q"] > 1 or std["tensor_kv"] > 1:
            trainer_mesh["model_q"] = std["tensor_q"] * std["tensor"]
            trainer_mesh["model_kv"] = std["tensor_kv"]
        else:
            trainer_mesh["model"] = std["tensor"]
        if std["expert"] > 1:
            trainer_mesh["expert"] = std["expert"]
        if std["pipeline"] > 1:
            trainer_mesh["pipeline"] = std["pipeline"]
        return trainer_mesh

    @classmethod
    def to_maxtext_config(cls, proposal: Mapping[str, int]) -> dict[str, int]:
        """Converts proposal into MaxText ICI parallelism configuration."""
        std = cls.standardize_proposal(proposal)
        return {
            "ici_tensor_parallelism":
            (std["tensor"] * std["tensor_q"] * std["tensor_kv"]),
            "ici_data_parallelism":
            (std["data"] * std["fsdp"] * std["data_local"]),
            "ici_expert_parallelism":
            std["expert"],
            "ici_sequence_parallelism":
            std["sequence"],
            "ici_context_parallelism":
            max(std["prefill_context"], std["decode_context"]),
        }

    @classmethod
    def to_vllm_flags(cls, proposal: Mapping[str, int]) -> dict[str, Any]:
        """Converts proposal into vLLM CLI arguments."""
        standardized = cls.standardize_proposal(proposal)
        effective_dp = standardized["data"] * standardized["fsdp"]

        flags = {
            "tensor_parallel_size": standardized["tensor"],
            "data_parallel_size": effective_dp,
            "expert_parallel_size": standardized["expert"],
            "prefill_context_parallel_size": standardized["prefill_context"],
            "decode_context_parallel_size": standardized["decode_context"],
            "pipeline_parallel_size": standardized["pipeline"],
            "data_parallel_size_local": standardized["data_local"],
        }
        cli_args = [
            f"--tensor-parallel-size={flags['tensor_parallel_size']}",
            f"--data-parallel-size={flags['data_parallel_size']}",
        ]
        if flags["expert_parallel_size"] > 1:
            cli_args.append(
                f"--expert-parallel-size={flags['expert_parallel_size']}")
        if flags["prefill_context_parallel_size"] > 1:
            cli_args.append("--prefill-context-parallel-size="
                            f"{flags['prefill_context_parallel_size']}")
        if flags["decode_context_parallel_size"] > 1:
            cli_args.append("--decode-context-parallel-size="
                            f"{flags['decode_context_parallel_size']}")
        if flags["pipeline_parallel_size"] > 1:
            cli_args.append(
                f"--pipeline-parallel-size={flags['pipeline_parallel_size']}")
        if flags["data_parallel_size_local"] > 1:
            cli_args.append(
                f"--data-parallel-size-local={flags['data_parallel_size_local']}"
            )

        return {
            "flags": flags,
            "cli_args": cli_args,
            "cli_string": " ".join(cli_args),
        }

    @classmethod
    def get_model_sharding(cls, model_cfg: Mapping[str,
                                                   Any]) -> dict[str, int]:
        """Extracts standard sharding dictionary from model configuration."""
        return {
            "tp": int(model_cfg.get("tensor_parallel_size", 1)),
            "dp": int(model_cfg.get("data_parallel_size", 1)),
            "ep": int(model_cfg.get("expert_parallel_size", 1)),
            "sp": int(model_cfg.get("sequence_parallel_size", 1)),
            "pcp": int(model_cfg.get("prefill_context_parallel_size", 1)),
            "dcp": int(model_cfg.get("decode_context_parallel_size", 1)),
            "pp": int(model_cfg.get("pipeline_parallel_size", 1)),
            "dpl": int(model_cfg.get("data_parallel_size_local", 1)),
        }

    @classmethod
    def apply_sharding_overrides(
        cls,
        model_cfg: Mapping[str, Any],
        tp: int | None = None,
        dp: int | None = None,
        ep: int | None = None,
        sp: int | None = None,
        pcp: int | None = None,
        dcp: int | None = None,
        pp: int | None = None,
        dpl: int | None = None,
    ) -> dict[str, Any]:
        """Creates an updated model config copy with requested sharding overrides."""
        updated = dict(model_cfg)
        if tp is not None:
            updated["tensor_parallel_size"] = tp
        if dp is not None:
            updated["data_parallel_size"] = dp
        if ep is not None:
            updated["expert_parallel_size"] = ep
        if sp is not None:
            updated["sequence_parallel_size"] = sp
        if pcp is not None:
            updated["prefill_context_parallel_size"] = pcp
        if dcp is not None:
            updated["decode_context_parallel_size"] = dcp
        if pp is not None:
            updated["pipeline_parallel_size"] = pp
        if dpl is not None:
            updated["data_parallel_size_local"] = dpl
        return updated


# ---------------------------------------------------------------------------
# Roofline & HBM Capacity Helpers
# ---------------------------------------------------------------------------


def resolve_accelerator_type(
    model_cfg: Mapping[str, Any] | None = None,
    fallback: str = "",
) -> str:
    """Resolves accelerator string across model config fields."""
    if not model_cfg:
        return fallback
    for key in (
            "accelerator_type",
            "device_type",
            "tpu_type",
            "platform",
            "system_name",
            "system",
    ):
        val = model_cfg.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return fallback


def resolve_platform_spec(accelerator_str: str) -> dict[str, Any] | None:
    """Resolves an accelerator string (e.g. 'tpu7x-8', 'v5p-128', 'trillium') to its spec dict."""
    spec = tpu_specs.get_platform_spec(accelerator_str)
    if spec is None:
        return None
    out = dict(spec)
    out.setdefault("hbm_gib_per_chip", out["hbm_gib"])
    out.setdefault("tcs_per_chip", out["devices_per_chip"])
    return out


def resolve_platform_hbm_capacity_gib(accelerator_str: str) -> float:
    """Resolves chip HBM capacity in GiB from `tpu_specs.PLATFORM_SPECS`."""
    spec = resolve_platform_spec(accelerator_str)
    if not spec:
        return 0.0
    return float(spec.get("hbm_gib", 0.0))


def resolve_platform_hbm_capacity_per_core_gib(accelerator_str: str) -> float:
    """Resolves per-TensorCore (per logical device) HBM capacity in GiB."""
    spec = resolve_platform_spec(accelerator_str)
    if not spec:
        return 0.0
    return float(spec.get("hbm_gib_per_core", 0.0))


def get_roofline_ridge_point(accelerator_str: str) -> float:
    """Returns BF16 roofline ridge point in FLOPs/byte for `accelerator_str`, or 0.0."""
    spec = resolve_platform_spec(accelerator_str)
    if not spec:
        return 0.0
    bw_bytes = float(spec.get("mem_bw_gbs", 0.0)) * 1e9
    if bw_bytes <= 0.0:
        return 0.0
    return float(spec.get("peak_flops", 0.0)) / bw_bytes


def compute_arithmetic_intensity(flops: float, bytes_accessed: float) -> float:
    """Returns operational arithmetic intensity in FLOPs/byte."""
    if bytes_accessed <= 0:
        return float("inf") if flops > 0 else 0.0
    return float(flops) / float(bytes_accessed)


def classify_bottleneck(flops: float, bytes_accessed: float,
                        accelerator_str: str) -> str:
    """Classifies an operation as 'memory_bound', 'compute_bound', or 'unknown'."""
    ridge = get_roofline_ridge_point(accelerator_str)
    if ridge <= 0.0:
        return "unknown"
    intensity = compute_arithmetic_intensity(flops, bytes_accessed)
    return "memory_bound" if intensity < ridge else "compute_bound"


_WEIGHT_WIDTH_RULES: tuple[tuple[tuple[str, ...], float], ...] = (
    (("nvfp4", "fp4", "int4", "w4a8", "w4a16", "awq", "gptq"), 0.5),
    (("fp8", "float8", "int8", "w8a8"), 1.0),
    (("float32", "fp32"), 4.0),
    (("bfloat16", "bf16", "float16", "fp16"), 2.0),
)

_DECLARED_WEIGHT_DTYPE_KEYS: tuple[str, ...] = (
    "weight_dtype",
    "quantization",
    "quant_method",
    "quantization_config",
    "dtype",
    "torch_dtype",
    "checkpoint_dtype",
)

_NAME_WEIGHT_DTYPE_KEYS: tuple[str, ...] = (
    "model",
    "model_name",
    "model_id",
    "served_model_name",
)


def resolve_weight_bytes_per_param(
    model_cfg: Mapping[str, Any] | None, ) -> float:
    """Returns bytes per stored weight, honouring quantized checkpoints (default 2.0 for bf16)."""
    if not model_cfg:
        return 2.0
    for keys in (_DECLARED_WEIGHT_DTYPE_KEYS, _NAME_WEIGHT_DTYPE_KEYS):
        blob = " ".join(str(model_cfg.get(key, "")).lower() for key in keys)
        for needles, width in _WEIGHT_WIDTH_RULES:
            if any(needle in blob for needle in needles):
                return width
    return 2.0


def estimate_hbm_capacity(
    proposal: Mapping[str, int],
    model_cfg: Mapping[str, Any] | None = None,
    tpu_type: str = "",
    hidden_size: int = 0,
    num_layers: int = 0,
    vocab_size: int = 0,
    num_experts: int = 0,
    intermediate_size: int = 0,
) -> list[str]:
    """Estimates HBM memory usage to check for OOM risk on Dense and MoE models.

    Accounts for:
    - Routed experts (`moe_intermediate_size`), shared experts (`n_shared_experts`),
      and leading dense layers (`first_k_dense_replace`).
    - Declared quantization / checkpoint storage width (`fp4`, `fp8`, `bf16`, `fp32`).
    - Multi-Head Latent Attention (MLA) non-TP-divided compressed KV cache vs GQA
      TP-sharded KV cache (`max(1, num_kv_heads // tp)`).
    - Optimizer state and gradient buffers when `model_cfg` indicates training.

    Returns:
      List of error strings if estimated memory exceeds 85% safe chip HBM capacity.
    """
    errors: list[str] = []
    norm_cfg: dict[str, Any] = {}
    adapter_cls = ShardingNomenclatureAdapter

    if model_cfg:
        norm_cfg = adapter_cls.normalize_model_config(model_cfg)
        hidden_size = (adapter_cls.get_config_int(
            norm_cfg, EMBEDDING_DIM_ALIASES, default=hidden_size) or 0)
        num_layers = (adapter_cls.get_config_int(
            norm_cfg, LAYERS_ALIASES, default=num_layers) or 0)
        vocab_size = (adapter_cls.get_config_int(
            norm_cfg, VOCAB_SIZE_ALIASES, default=vocab_size) or 0)
        num_experts = (adapter_cls.get_config_int(
            norm_cfg, NUM_EXPERTS_ALIASES, default=num_experts) or 0)
        intermediate_size = (adapter_cls.get_config_int(
            norm_cfg, FFN_DIM_ALIASES, default=intermediate_size) or 0)
        if intermediate_size <= 0 and hidden_size > 0:
            intermediate_size = hidden_size * 4
        moe_intermediate_size = (adapter_cls.get_config_int(
            norm_cfg, MOE_FFN_DIM_ALIASES, default=0) or 0)
        num_shared_experts = (adapter_cls.get_config_int(
            norm_cfg, SHARED_EXPERTS_ALIASES, default=0) or 0)
        num_dense_layers = (adapter_cls.get_config_int(
            norm_cfg, DENSE_LAYER_COUNT_ALIASES, default=0) or 0)
        tpu_type = resolve_accelerator_type(norm_cfg, fallback=tpu_type)
    else:
        hidden_size = max(0, hidden_size)
        num_layers = max(0, num_layers)
        vocab_size = max(0, vocab_size)
        num_experts = max(0, num_experts)
        intermediate_size = max(0, intermediate_size)
        moe_intermediate_size = 0
        num_shared_experts = 0
        num_dense_layers = 0

    if hidden_size <= 0 or num_layers <= 0:
        return []

    if intermediate_size <= 0:
        intermediate_size = hidden_size * 4

    normalized = adapter_cls.standardize_proposal(proposal)
    tp = max(
        1,
        int(normalized.get("tensor", 1)) * int(normalized.get("tensor_q", 1)) *
        int(normalized.get("tensor_kv", 1)),
    )
    ep = max(1, int(normalized.get("expert", 1)))
    fsdp = max(1, int(normalized.get("fsdp", 1)))
    pcp = max(1, int(normalized.get("prefill_context", 1)))
    dcp = max(1, int(normalized.get("decode_context", 1)))
    cp_factor = max(1, pcp, dcp)
    effective_dp = (max(1, int(normalized.get("data", 1))) * fsdp *
                    max(1, int(normalized.get("data_local", 1))))

    # 1. Weights parameter calculation (Dense vs MoE)
    dense_params = (vocab_size * hidden_size * 2 + num_layers * 4 *
                    (hidden_size**2))

    if num_experts > 0:
        expert_ffn_dim = moe_intermediate_size or intermediate_size
        dense_layers = min(max(0, num_dense_layers), num_layers)
        moe_layers = num_layers - dense_layers
        moe_params = num_experts * moe_layers * 3 * hidden_size * expert_ffn_dim
        mlp_params = dense_layers * 3 * hidden_size * intermediate_size + (
            num_shared_experts * moe_layers * 3 * hidden_size * expert_ffn_dim)
    else:
        mlp_params = num_layers * 3 * hidden_size * intermediate_size
        moe_params = 0

    if model_cfg and adapter_cls.detect_runtime_engine(model_cfg) == "vllm":
        moe_shards = max(ep, tp * effective_dp * max(1, pcp))
    else:
        moe_shards = ep if ep > 1 else tp
    sharded_params = (dense_params + mlp_params) / (tp * fsdp) + moe_params / (
        moe_shards * fsdp)
    weight_bytes = resolve_weight_bytes_per_param(model_cfg)
    weights_memory_gib = (sharded_params * weight_bytes) / (1024**3)

    workload = adapter_cls.detect_training_or_inference(model_cfg)
    optimizer_memory_gib = 0.0
    if workload == "training":
        optimizer_bytes_per_param = adapter_cls.optimizer_state_bytes_per_param(
            model_cfg)
        optimizer_memory_gib = (sharded_params *
                                (optimizer_bytes_per_param + weight_bytes)) / (
                                    1024**3)

    # 2. KV Cache Footprint Calculation (if batch size & context length present)
    kv_cache_gib = 0.0
    if model_cfg:
        kv_cfg = norm_cfg if norm_cfg else model_cfg

        def _parse_cfg_int(key: str, default_val: int = 0) -> int:
            try:
                val = kv_cfg.get(key)
                if val is None:
                    return default_val
                return max(0, int(val))
            except (ValueError, TypeError):
                return default_val

        def _parse_aliased_int(aliases: tuple[str, ...],
                               default_val: int = 0) -> int:
            resolved = adapter_cls.get_config_int(kv_cfg, aliases)
            return max(0, resolved) if resolved is not None else default_val

        def _parse_raw_batch(aliases: tuple[str, ...]) -> int:
            resolved = adapter_cls.get_config_int(model_cfg, aliases)
            return max(0, resolved) if resolved is not None else 0

        batch_size = _parse_raw_batch(
            PER_DEVICE_BATCH_SIZE_ALIASES) or _parse_raw_batch(
                AMBIGUOUS_BATCH_SIZE_ALIASES)
        if batch_size <= 0:
            global_batch = _parse_raw_batch(EXPLICIT_GLOBAL_BATCH_SIZE_ALIASES)
            batch_size = (max(1, global_batch //
                              effective_dp) if global_batch > 0 else 0)
        max_model_len = _parse_aliased_int(SEQ_LEN_ALIASES, 0)
        num_heads = _parse_aliased_int(HEADS_ALIASES, 0)
        num_kv_heads = _parse_aliased_int(KV_HEADS_ALIASES, num_heads)
        head_dim = _parse_aliased_int(HEAD_DIM_ALIASES, 0)
        if head_dim <= 0:
            head_dim = (hidden_size // num_heads) if num_heads > 0 else 128

        kv_dtype = str(model_cfg.get("kv_cache_dtype", "bfloat16")).lower()
        kv_bytes = 1 if "fp8" in kv_dtype else 2

        is_mla = adapter_cls.is_mla_model(model_cfg)
        latent_scored = False
        if is_mla:
            kv_lora_rank = _parse_cfg_int("kv_lora_rank", 0)
            qk_rope_head_dim = _parse_cfg_int("qk_rope_head_dim", 0)
            latent_per_token = kv_lora_rank + qk_rope_head_dim
            if latent_per_token > 0:
                latent_scored = True
                if batch_size > 0 and max_model_len > 0:
                    kv_cache_elements = (num_layers * latent_per_token *
                                         max_model_len * batch_size)
                    kv_cache_bytes = (kv_cache_elements * kv_bytes) / cp_factor
                    kv_cache_gib = kv_cache_bytes / (1024**3)
        if (not latent_scored and batch_size > 0 and max_model_len > 0
                and num_kv_heads > 0 and tp > 0):
            kv_heads_per_rank = max(1, num_kv_heads // tp)
            kv_cache_elements = (num_layers * 2 * kv_heads_per_rank *
                                 head_dim * max_model_len * batch_size)
            kv_cache_bytes = (kv_cache_elements * kv_bytes) / cp_factor
            kv_cache_gib = kv_cache_bytes / (1024**3)

    # 3. Dynamic Platform HBM Capacity Lookup
    chip_capacity_gib = resolve_platform_hbm_capacity_gib(tpu_type)
    if chip_capacity_gib <= 0.0:
        return []
    max_safe_memory = chip_capacity_gib * HBM_UTILIZATION_SAFETY_FACTOR

    total_estimated_gib = (weights_memory_gib + kv_cache_gib +
                           optimizer_memory_gib)

    if total_estimated_gib > max_safe_memory:
        breakdown = f"{weights_memory_gib:.1f} GiB weights"
        if optimizer_memory_gib > 0:
            breakdown += f" + {optimizer_memory_gib:.1f} GiB optimizer/gradients"
        if kv_cache_gib > 0:
            breakdown += f" + {kv_cache_gib:.1f} GiB KV cache"
        errors.append(
            f"OOM Risk: Estimated memory ({total_estimated_gib:.1f} GiB = "
            f"{breakdown}) exceeds 85% safe capacity of {tpu_type} "
            f"({chip_capacity_gib:.0f} GiB). Increase TP, EP, or Context"
            " Parallelism (PCP/DCP).")

    return errors
