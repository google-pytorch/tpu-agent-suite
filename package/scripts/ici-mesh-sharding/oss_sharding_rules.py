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
"""Model sharding rules and architectural constraint validator for TPU meshes.

Validates sharding proposals for Open-Source (OSS) LLMs, MoE, and Multimodal
models (Qwen, Gemma, Llama, DeepSeek, Kimi, GLM, Mixtral, GPT-OSS) as well as
JAX/MaxText trainer configurations against hardware capacity, divisibility,
Multi-Head Latent Attention (MLA), derived vLLM EP/DCP semantics, and HBM limits.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import sharding_adapter  # noqa: E402

# Re-export HBM and roofline helpers for single-module callers.
classify_bottleneck = sharding_adapter.classify_bottleneck
compute_arithmetic_intensity = sharding_adapter.compute_arithmetic_intensity
get_roofline_ridge_point = sharding_adapter.get_roofline_ridge_point
resolve_platform_hbm_capacity_gib = (
    sharding_adapter.resolve_platform_hbm_capacity_gib)
resolve_platform_hbm_capacity_per_core_gib = (
    sharding_adapter.resolve_platform_hbm_capacity_per_core_gib)
resolve_weight_bytes_per_param = sharding_adapter.resolve_weight_bytes_per_param


def is_power_of_two(val: int) -> bool:
    """Checks if a positive integer is a power of 2."""
    return val > 0 and (val & (val - 1)) == 0


def validate_oss_model_sharding(
    proposal: Mapping[str, int],
    model_cfg: Mapping[str, Any],
    total_tpu_chips: int,
) -> tuple[bool, list[str], list[str]]:
    """Validates a model sharding proposal against architectural and HBM constraints.

    Args:
      proposal: Parallelism dimensions (`tp`, `dp`, `ep`, `sp`, `fsdp`, `cp`,
        `pcp`, `dcp`, `pp`, `dpl`, or JAX trainer keys).
      model_cfg: Model specification dictionary (`num_heads`, `num_kv_heads`,
        `num_experts`, `kv_lora_rank`, `accelerator_type`, etc.).
      total_tpu_chips: Total available physical TPU chips.

    Returns:
      A tuple `(is_valid, errors, warnings)`.
    """
    errors: list[str] = []
    warnings: list[str] = []
    adapter_cls = sharding_adapter.ShardingNomenclatureAdapter

    std_proposal = adapter_cls.standardize_proposal(proposal)
    tp = std_proposal.get("tensor", 1)
    dp = std_proposal.get("data", 1)
    ep = std_proposal.get("expert", 1)
    sp = std_proposal.get("sequence", 1)
    fsdp = std_proposal.get("fsdp", 1)
    pcp = std_proposal.get("prefill_context", 1)
    dcp = std_proposal.get("decode_context", 1)
    pp = std_proposal.get("pipeline", 1)
    dpl = std_proposal.get("data_local", 1)
    tq = std_proposal.get("tensor_q", 1)
    tkv = std_proposal.get("tensor_kv", 1)

    # --- 1. Power of 2 check ---
    dimensions = {
        "tp": tp,
        "dp": dp,
        "ep": ep,
        "sp": sp,
        "fsdp": fsdp,
        "pcp": pcp,
        "dcp": dcp,
        "pp": pp,
        "dpl": dpl,
        "tensor_q": tq,
        "tensor_kv": tkv,
    }
    has_dim_error = False
    for name, val in dimensions.items():
        if val < 1:
            errors.append(f"Dimension {name} must be >= 1, got {val}")
            has_dim_error = True
        elif not is_power_of_two(val):
            errors.append(f"Dimension {name}={val} is not a power of 2")

    if has_dim_error:
        return False, errors, warnings

    # --- 2. Mesh product capacity check (vLLM vs MaxText/JAX) ---
    engine = adapter_cls.detect_runtime_engine(model_cfg)
    tpu_val = sharding_adapter.resolve_accelerator_type(model_cfg, fallback="")
    tpu_type = str(tpu_val).lower() if tpu_val else ""
    cores_per_chip = adapter_cls.get_cores_per_chip(
        tpu_type) if tpu_type else 1

    if engine == "vllm":
        dcp_error = adapter_cls.vllm_dcp_error(std_proposal)
        if dcp_error:
            errors.append(dcp_error)

    if total_tpu_chips > 0:
        expected_capacity = total_tpu_chips * cores_per_chip
        cap_desc = f"TensorCore logical mesh capacity ({expected_capacity} cores)"
        if engine == "vllm" and sp > 1:
            errors.append(
                "Sequence Parallelism (SP) is currently not supported by vLLM on"
                " TPU. Use pcp/dcp for sequence parallelism in vLLM.")

        mesh_product = adapter_cls.compute_mesh_product(std_proposal, engine)
        if mesh_product != expected_capacity:
            errors.append(
                f"Mesh product ({mesh_product}) does not equal expected {cap_desc}"
            )

        if engine == "vllm" and ep > 1:
            derived_ep = adapter_cls.derived_expert_parallelism(std_proposal)
            if ep != derived_ep:
                errors.append(
                    f"Expert Parallelism EP={ep} is not realizable in vLLM: EP is"
                    " derived as tp * dp * pcp and equals"
                    f" {derived_ep} for this mesh. Use ep=1 (expert parallelism"
                    f" disabled) or ep={derived_ep}.")
    else:
        warnings.append(
            "Mesh capacity check skipped: total_tpu_chips not provided.")

    # --- 3. Attention Heads Divisibility Constraint ---
    num_heads = adapter_cls.get_config_int(model_cfg,
                                           sharding_adapter.HEADS_ALIASES)
    q_shards = tp * tq
    kv_shards = tp * tkv
    if num_heads is not None and num_heads > 0:
        if q_shards > num_heads:
            errors.append(
                f"Tensor Parallelism TP={q_shards} exceeds total query heads"
                f" ({num_heads})")
        elif q_shards > 0 and num_heads % q_shards != 0:
            errors.append(
                f"Attention query heads ({num_heads}) not divisible by"
                f" TP={q_shards}")
    elif tp > 1:
        warnings.append(
            "Attention heads check skipped: num_heads missing in model_cfg.")

    # --- 4. Grouped Query Attention (GQA) & MLA KV Heads Constraint ---
    is_mla = adapter_cls.is_mla_model(model_cfg)
    num_kv_heads = adapter_cls.get_config_int(
        model_cfg, sharding_adapter.KV_HEADS_ALIASES)
    if not is_mla and num_kv_heads is not None and num_kv_heads > 0 and kv_shards > 0:
        if kv_shards > num_kv_heads and kv_shards % num_kv_heads != 0:
            warnings.append(
                f"TP={kv_shards} > KV heads ({num_kv_heads}); replication"
                " required.")
        elif kv_shards <= num_kv_heads and num_kv_heads % kv_shards != 0:
            errors.append(
                f"KV heads ({num_kv_heads}) not divisible by TP={kv_shards}")

    # --- 5. MoE Experts Divisibility & Routing Constraints ---
    num_experts = adapter_cls.get_config_int(
        model_cfg, sharding_adapter.NUM_EXPERTS_ALIASES)
    num_experts_per_tok = adapter_cls.get_config_int(
        model_cfg, sharding_adapter.NUM_EXPERTS_PER_TOK_ALIASES)

    if num_experts is not None and num_experts > 0:
        if ep > num_experts:
            errors.append(f"Expert Parallelism EP={ep} exceeds total experts"
                          f" ({num_experts})")
        elif ep > 0 and num_experts % ep != 0:
            errors.append(
                f"Total experts ({num_experts}) is not divisible by EP={ep}")
        elif ep == 1 and total_tpu_chips >= 8:
            warnings.append(
                f"MoE model with {num_experts} experts uses EP=1; consider"
                " Expert Parallelism (EP > 1) to distribute expert weights"
                " across accelerators.")
    elif ep > 1:
        warnings.append(
            "MoE experts check skipped: num_experts missing in model_cfg.")

    if num_experts_per_tok is not None:
        if num_experts_per_tok <= 0:
            errors.append(
                f"num_experts_per_tok must be > 0, got {num_experts_per_tok}")
        elif (num_experts is not None and num_experts > 0
              and num_experts_per_tok > num_experts):
            errors.append(
                f"num_experts_per_tok ({num_experts_per_tok}) exceeds total"
                f" experts ({num_experts})")
        elif (ep > 1 and num_experts is not None and num_experts > 0
              and num_experts % ep == 0):
            experts_per_rank = num_experts // ep
            if num_experts_per_tok > experts_per_rank:
                warnings.append(
                    f"num_experts_per_tok ({num_experts_per_tok}) > experts per"
                    f" rank ({experts_per_rank}); cross-rank expert routing"
                    " (All-to-All) required.")

    # --- 6. Batch Size Divisibility Constraint ---
    batch_size = (adapter_cls.get_config_int(
        model_cfg,
        sharding_adapter.GLOBAL_BATCH_SIZE_ALIASES,
        default=0,
    ) or 0)
    effective_dp = dp * fsdp * dpl
    if batch_size > 0 and effective_dp > 0 and batch_size % effective_dp != 0:
        warnings.append(
            f"Batch size ({batch_size}) not divisible by effective DP"
            f" ({effective_dp})")

    # --- 7. Pipeline Parallelism (PP) Layer Divisibility Constraint ---
    if pp > 1:
        num_layers = adapter_cls.get_config_int(
            model_cfg, sharding_adapter.LAYERS_ALIASES)
        if num_layers is not None and num_layers > 0 and num_layers % pp != 0:
            errors.append(
                f"Number of layers ({num_layers}) is not divisible by Pipeline"
                f" Parallelism PP={pp}")

    # --- 8. HBM Capacity & OOM Check (Dense & MoE) ---
    hbm_errors = sharding_adapter.estimate_hbm_capacity(proposal,
                                                        model_cfg=model_cfg,
                                                        tpu_type=tpu_type)
    errors.extend(hbm_errors)

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def estimate_hbm_capacity(
    proposal: Mapping[str, int],
    model_cfg: Mapping[str, Any] | None = None,
    tpu_type: str = "",
) -> list[str]:
    """Wrapper delegating to `sharding_adapter.estimate_hbm_capacity`."""
    return sharding_adapter.estimate_hbm_capacity(proposal,
                                                  model_cfg=model_cfg,
                                                  tpu_type=tpu_type)
