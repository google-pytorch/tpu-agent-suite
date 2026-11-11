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
"""Pluggable model specifications and composable architecture packs.

Separates roofline and step-cost modeling into two layers:
  1. Composable `ArchPack` building blocks (`FineGrainedMoEPack`,
     `SparseMLADcpPack`, `SparseCoreIndexerPack`, `GDNHybridAttentionPack`,
     `MTPSpecDecodePack`) contributing op-classification regexes, first-principles
     HBM/ICI/MXU roofline floors, and concrete optimization levers.
  2. Public declarative `ModelPlugin` specs (`glm5_mla_moe`,
     `qwen3_5_hybrid_moe`, `deepseek_mla_moe`, plus custom JSON overlays).
"""

import dataclasses
import json
import pathlib
import re
from collections.abc import Mapping
from typing import Any


@dataclasses.dataclass(frozen=True)
class HardwareSpec:
    """Peak bandwidth and matrix compute parameters for one chip."""

    name: str = "tpu7x"
    chips_per_host: int = 4
    cores_per_chip: int = 2
    hbm_bw_tbps_per_chip: float = 7.372
    fp8_tflops_per_chip: float = 4614.0
    bf16_tflops_per_chip: float = 2307.0
    ici_bw_tbps_per_chip: float = 1.20
    # Measured VPU 4-bit weight unpack rate (billions of weights per ms per
    # TensorCore). No default: 4-bit compute floors leave out the unpack time
    # unless the caller supplies it (`roofline --vpu-unpack-gweights-per-ms`).
    vpu_unpack_gweights_per_ms_per_tc: float | None = None


TPU_7X = HardwareSpec()


@dataclasses.dataclass(frozen=True)
class ModelDims:
    """Architectural dimensions of a Transformer or hybrid MoE serving model."""

    hidden_size: int
    num_layers: int
    moe_layers: int
    num_experts: int = 0
    top_k: int = 0
    moe_intermediate_size: int = 0
    shared_experts: int = 1
    kv_lora_rank: int = 0
    qk_rope_dim: int = 0
    q_lora_rank: int = 1536
    num_heads: int = 64
    v_head_dim: int = 128
    gdn_layers: int = 0
    full_attn_layers: int = 0
    indexer_layers: int = 0
    mtp_draft_tokens: int = 3
    vocab_size: int = 151936


@dataclasses.dataclass(frozen=True)
class ServingTopology:
    """Parallelism and slice geometry for a serving deployment."""

    mode: str  # disagg_pd | single_host
    tp_size: int
    ep_size: int = 1
    dcp_size: int = 1
    prefill_slices: int = 1
    decode_slices: int = 1
    chips_per_slice: int = 8

    @property
    def total_chips(self) -> int:
        return (self.prefill_slices +
                self.decode_slices) * self.chips_per_slice

    @property
    def moe_mode(self) -> str:
        return "ep" if self.ep_size > 1 else "tp"


@dataclasses.dataclass(frozen=True)
class StepContext:
    """Runtime step conditions used to evaluate an `ArchPack` roofline."""

    batch_reqs: int = 16
    padded_rows: int = 64
    context_tokens: int = 50400
    weight_dtype: str = "fp8"
    quant_block_size: int = 0
    row_compacted: bool = False
    moe_mode: str = ""
    routing_correlation: float = 1.0


@dataclasses.dataclass(frozen=True)
class OpRule:
    """Regex rule mapping trace ops/groups into an architectural category."""

    category: str
    bucket: str
    pattern: str
    scaling: str

    def matches(self, text: str) -> bool:
        return re.search(self.pattern, text, flags=re.IGNORECASE) is not None


@dataclasses.dataclass(frozen=True)
class PlaybookLever:
    """Concrete kernel or configuration lever owned by an `ArchPack`."""

    name: str
    target_categories: tuple[str, ...]
    title: str
    mechanism: str
    action: str
    # Hand-set prior range (low, high) for the share of the target categories'
    # measured time that this lever could remove. It is a heuristic estimate,
    # not a measured result; `trace_roofline` multiplies it by the measured
    # category time and labels the printed saving range as heuristic.
    est_saving_share: tuple[float, float]
    requires_accuracy_gate: bool = False


@dataclasses.dataclass(frozen=True)
class PackRoofline:
    """Analytical HBM / ICI / MXU roofline floor produced by one `ArchPack`."""

    pack_name: str
    buckets: tuple[str, ...]
    streamed_bytes_per_chip_step: float
    ici_bytes_per_chip_step: float
    flops_per_chip_step: float
    hbm_floor_ms_per_step: float
    ici_floor_ms_per_step: float
    compute_floor_ms_per_step: float
    floor_ms_per_step: float
    limiting_resource: str
    active_experts_per_layer: float | None = None
    per_call_hbm_floor_us: float | None = None
    warnings: tuple[str, ...] = ()
    details: Mapping[str,
                     float | str] = dataclasses.field(default_factory=dict)


class ArchPack:
    """Base interface for composable model-architecture packs."""

    name: str = "base"

    def op_rules(self) -> tuple[OpRule, ...]:
        return ()

    def levers(self) -> tuple[PlaybookLever, ...]:
        return ()

    def roofline(
        self,
        dims: ModelDims,
        topology: ServingTopology,
        hw: HardwareSpec,
        ctx: StepContext,
    ) -> PackRoofline:
        raise NotImplementedError


def expected_active_experts(
    rows: float,
    num_experts: int,
    top_k: int,
    routing_correlation: float = 1.0,
) -> float:
    """Computes `E * (1 - (1 - topk / E) ** (rows * routing_correlation))`."""
    if num_experts <= 0 or top_k <= 0 or rows <= 0:
        return 0.0
    if top_k >= num_experts:
        return float(num_experts)
    eff_rows = max(0.0, float(rows) * routing_correlation)
    prob_untouched = (1.0 - float(top_k) / float(num_experts))**eff_rows
    return float(num_experts) * (1.0 - prob_untouched)


def weight_bytes_per_param(weight_dtype: str,
                           quant_block_size: int = 0) -> float:
    """Effective HBM bytes per weight parameter including block scales."""
    dtype = weight_dtype.lower().strip()
    if dtype == "fp8":
        return 1.0 + (4.0 / quant_block_size if quant_block_size > 0 else 0.0)
    if dtype in ("w4a8", "fp4"):
        return 0.5 + (4.0 / quant_block_size if quant_block_size > 0 else 0.0)
    if dtype == "mxfp4":
        block = quant_block_size if quant_block_size > 0 else 32
        return 0.5 + 1.0 / block
    if dtype in ("nvfp4", "w4a16"):
        block = quant_block_size if quant_block_size > 0 else 16
        return 0.5 + 1.0 / block
    if dtype in ("bf16", "fp16"):
        return 2.0
    raise ValueError(f"Unsupported weight_dtype {weight_dtype!r}.")


def _dominant_resource(hbm_ms: float, ici_ms: float,
                       compute_ms: float) -> tuple[float, str]:
    best = max(hbm_ms, ici_ms, compute_ms)
    if compute_ms >= hbm_ms and compute_ms >= ici_ms and compute_ms > 0:
        return best, "compute"
    if ici_ms >= hbm_ms and ici_ms > 0:
        return best, "ici"
    return best, "hbm"


class FineGrainedMoEPack(ArchPack):
    """Fine-grained routed Mixture-of-Experts pack (TP stream & fused EP)."""

    name = "fine_grained_moe"

    def op_rules(self) -> tuple[OpRule, ...]:
        return (
            OpRule(
                category="moe_expert",
                bucket="MoE",
                pattern=(
                    r"moe_tp_decode|fused_ep_moe|adaptive_fused_moe|megablox|"
                    r"gmm_v2|fused_moe|moe expert kernel|^moe$"),
                scaling="expert_coverage",
            ),
            OpRule(
                category="moe_router",
                bucket="MoE",
                pattern=r"moe router|gate matmul|router_topk|topk_softmax",
                scaling="padded_rows",
            ),
        )

    def levers(self) -> tuple[PlaybookLever, ...]:
        return (
            PlaybookLever(
                name="moe_w4a8_compacted_weights",
                target_categories=("moe_expert", ),
                title="Row-compacted W4A8 (FP4) expert weights",
                mechanism=(
                    "Fine-grained MoE decode reached 96.7% of the FP8 HBM "
                    "roofline in a GLM-5.2-FP8 decode trace, so halving expert "
                    "weight bytes with routed row compaction would cut HBM "
                    "streaming time nearly in half. On TP decode it lost "
                    "instead: `TPU_MOE_TP_DECODE_COMPACT=1;"
                    "TPU_MOE_TP_DECODE_W4A8=1` against FP8 without either "
                    "flag, same commit and shape, was -10.1% throughput per "
                    "chip at c=48 cap=20 and -13.4% at c=32 cap=8 (GLM-5.2 "
                    "1P1D)."),
                action=(
                    "Do not relaunch compacted W4A8 TP decode unchanged; "
                    "compare its per-layer MoE time with FP8 first. Otherwise "
                    "run `DECODE_MOE=ep` with "
                    "`USE_MOE_FUSED_EP_KERNEL=1;MOE_FUSED_EP_ENABLE_W4A8=1`, "
                    "gated on downstream accuracy."),
                est_saving_share=(0.43, 0.50),
                requires_accuracy_gate=True,
            ),
            PlaybookLever(
                name="moe_tp_decode_row_buckets_and_ici_pieces",
                target_categories=("moe_expert", ),
                title=
                "Extend fused TP MoE `ROW_BUCKETS` and ICI exchange pieces",
                mechanism=
                ("Decode steps landing outside admitted `ROW_BUCKETS` fall "
                 "back to a 2.5x slower unfused GMM+psum path (80 rows: 175 "
                 "vs 447 us per MoE layer on tpu7x). Admitting 80 rows let "
                 "c=48 run cap=20: +3.0% tok/s/chip as a single change, and "
                 "+5.3% throughput per chip on top of vllm-torchtpu #2681 "
                 "(both in vllm-torchtpu #2852). In-kernel ICI pieces hide "
                 "shard exchange latency."),
                action=("Admit the target `cap*(1+K)` rows in `ROW_BUCKETS` "
                        "(`(16, 32, 64, 80, 96)` with vllm-torchtpu #2852; "
                        "97-128 rows still fall back), add that count to "
                        "`DECODE_TOKEN_BUCKET_EXTRA`, and set "
                        "`TPU_MOE_TP_DECODE_EXCHANGE_PIECES=16;"
                        "TPU_MOE_TP_DECODE_AG_PAIRS=64`."),
                est_saving_share=(0.05, 0.20),
                requires_accuracy_gate=False,
            ),
        )

    def roofline(
        self,
        dims: ModelDims,
        topology: ServingTopology,
        hw: HardwareSpec,
        ctx: StepContext,
    ) -> PackRoofline:
        moe_mode = (ctx.moe_mode or topology.moe_mode).lower()
        dtype = ctx.weight_dtype.lower().strip()
        is_4bit = dtype in ("w4a8", "fp4", "mxfp4", "nvfp4", "w4a16")
        bpp = weight_bytes_per_param(dtype, ctx.quant_block_size)
        shared_bpp = 1.0 if dtype == "fp8" else bpp

        params_per_expert = 3.0 * dims.hidden_size * dims.moe_intermediate_size
        sharding = max(1, topology.chips_per_slice)
        expert_bytes_per_chip = params_per_expert * bpp / sharding
        shared_bytes_per_chip = (dims.shared_experts * params_per_expert *
                                 shared_bpp / sharding)

        live_rows = max(1,
                        ctx.batch_reqs * (1 + max(0, dims.mtp_draft_tokens)))
        eff_step_rows = max(live_rows, ctx.padded_rows)
        active_main = expected_active_experts(live_rows, dims.num_experts,
                                              dims.top_k,
                                              ctx.routing_correlation)
        imbalance = 1.08 if moe_mode == "ep" else 1.0

        layer_bytes_main = (active_main * expert_bytes_per_chip * imbalance +
                            shared_bytes_per_chip)
        mtp_full_passes = 1 if dims.mtp_draft_tokens >= 1 else 0
        mtp_small_passes = max(0, dims.mtp_draft_tokens - 1)
        active_small = expected_active_experts(ctx.batch_reqs,
                                               dims.num_experts, dims.top_k,
                                               ctx.routing_correlation)
        layer_bytes_small = (active_small * expert_bytes_per_chip * imbalance +
                             shared_bytes_per_chip)

        full_calls = dims.moe_layers + mtp_full_passes
        total_bytes = (full_calls * layer_bytes_main +
                       mtp_small_passes * layer_bytes_small)
        peak_hbm_bps = hw.hbm_bw_tbps_per_chip * 1e12
        hbm_floor_ms = total_bytes / peak_hbm_bps * 1e3
        per_call_hbm_us = layer_bytes_main / peak_hbm_bps * 1e6

        warnings: list[str] = []
        routed_token_experts = live_rows * (dims.top_k + dims.shared_experts)
        compacted_flops_layer = (2.0 * routed_token_experts *
                                 params_per_expert / sharding)
        if (is_4bit and moe_mode == "tp" and eff_step_rows >= 64
                and not ctx.row_compacted):
            uncompacted_flops_layer = (2.0 * eff_step_rows *
                                       (active_main + dims.shared_experts) *
                                       params_per_expert / sharding)
            flops_step = (full_calls * uncompacted_flops_layer +
                          mtp_small_passes * compacted_flops_layer)
            tflops = (hw.bf16_tflops_per_chip
                      if dtype == "w4a16" else hw.fp8_tflops_per_chip)
            compute_floor_ms = flops_step / (tflops * 1e12) * 1e3
            vpu_note = ""
            if hw.vpu_unpack_gweights_per_ms_per_tc:
                weights_per_tc_step = (total_bytes / bpp /
                                       max(1, hw.cores_per_chip))
                compute_floor_ms += weights_per_tc_step / (
                    hw.vpu_unpack_gweights_per_ms_per_tc * 1e9)
            else:
                vpu_note = "; VPU widen time not included: no unpack rate"
            relation = ">" if compute_floor_ms > hbm_floor_ms else "<="
            warnings.append(
                f"UNCOMPACTED_W4A8_TP_COMPUTE_WALL: padded_rows={eff_step_rows}"
                f" >= 64 under {dtype.upper()} TP decode with "
                "row_compacted=False hits the tpu7x VPU widen + MXU wall "
                f"(compute floor {compute_floor_ms:.2f} ms {relation} HBM "
                f"floor {hbm_floor_ms:.2f} ms{vpu_note}).")
        else:
            flops_step = (full_calls +
                          mtp_small_passes) * compacted_flops_layer
            tflops = (hw.bf16_tflops_per_chip
                      if dtype == "w4a16" else hw.fp8_tflops_per_chip)
            compute_floor_ms = flops_step / (tflops * 1e12) * 1e3
            if dtype == "w4a16":
                if hw.vpu_unpack_gweights_per_ms_per_tc:
                    weights_per_tc_step = (total_bytes / bpp /
                                           max(1, hw.cores_per_chip))
                    compute_floor_ms += 1.35 * (
                        weights_per_tc_step /
                        (hw.vpu_unpack_gweights_per_ms_per_tc * 1e9))
                warnings.append(
                    "W4A16_CAST_PENALTY: fp4->bf16 cast routes through f32 on "
                    "tpu7x; prefer W4A8.")

        ici_bytes = full_calls * eff_step_rows * dims.hidden_size * 2.0
        ici_floor_ms = ici_bytes / (hw.ici_bw_tbps_per_chip * 1e12) * 1e3
        floor_ms, limiting = _dominant_resource(hbm_floor_ms, ici_floor_ms,
                                                compute_floor_ms)
        return PackRoofline(
            pack_name=self.name,
            buckets=("MoE", ),
            streamed_bytes_per_chip_step=total_bytes,
            ici_bytes_per_chip_step=ici_bytes,
            flops_per_chip_step=flops_step,
            hbm_floor_ms_per_step=hbm_floor_ms,
            ici_floor_ms_per_step=ici_floor_ms,
            compute_floor_ms_per_step=compute_floor_ms,
            floor_ms_per_step=floor_ms,
            limiting_resource=limiting,
            active_experts_per_layer=active_main,
            per_call_hbm_floor_us=per_call_hbm_us,
            warnings=tuple(warnings),
            details={
                "active_expert_fraction":
                active_main / max(1, dims.num_experts),
                "bytes_per_param": bpp,
                "full_moe_calls_per_step": float(full_calls),
                "moe_mode": moe_mode,
            },
        )


class SparseMLADcpPack(ArchPack):
    """Multi-head Latent Attention (MLA) with Decode Context Parallelism (DCP)."""

    name = "sparse_mla_dcp"

    def op_rules(self) -> tuple[OpRule, ...]:
        return (
            OpRule(
                category="mla_dcp_combine",
                bucket="MLA_DCP",
                pattern=(
                    r"dcp all-gather|dcp reduce-scatter|attn dcp comm|"
                    r"distributed_reduce_scatter_tensor|all_gather_into_tensor|"
                    r"lse_rescale|attn lse merge|dcp_online_lse_combine"),
                scaling="padded_rows",
            ),
            OpRule(
                category="mla_oproj",
                bucket="MLA_DCP",
                pattern=
                (r"fp8_uv_oproj|o-proj|small_all_reduce|distributed_all_reduce"
                 ),
                scaling="fixed",
            ),
            OpRule(
                category="mla_proj",
                bucket="MLA_DCP",
                pattern=(r"fused_norm_qkva|_prequantized_matmul|absorb_q|"
                         r"attn q_a/kv_a|attn q_b|attn absorb_q|attn proj|"
                         r"norms \(standalone"),
                scaling="fixed",
            ),
            OpRule(
                category="mla_core",
                bucket="MLA_DCP",
                pattern=(
                    r"mla-sparsecore|attn sparse mla|sc_native_sc_gather|"
                    r"sparsecore_cache|attn kv gather|attn prep|attn core|"
                    r"^attention$"),
                scaling="live_rows",
            ),
        )

    def levers(self) -> tuple[PlaybookLever, ...]:
        return (
            PlaybookLever(
                name="mla_dcp_fused_online_lse_rs",
                target_categories=("mla_dcp_combine", ),
                title="Fused DCP online-LSE reduce-scatter / UV-before-merge",
                mechanism=
                ("Separate LSE all-gather + lse_rescale + partial "
                 "reduce-scatter pay three collective/kernel launches per "
                 "layer (4.42 ms per step at DCP16 in a GLM-5.2-FP8 decode "
                 "trace). The Pallas `dcp_online_lse_combine` (vllm-torchtpu "
                 "#2851, `pairs` mode) took 223.9 -> 68.1 us/layer at 64 rows "
                 "(DCP=8 micro-benchmark); end to end it shipped bundled with "
                 "other changes for +1.0% at c=48 cap=20, within noise. A "
                 "token-chunked variant for >96 rows measured slower than the "
                 "XLA path (196.76 vs 149.87 us), so keep the combine at <=96 "
                 "rows."),
                action=(
                    "With the vllm-torchtpu #2851 code, enable "
                    "`TPU_DCP_ONLINE_LSE_COMBINE=1` with "
                    "`TPU_DCP_ONLINE_LSE_COMBINE_MAX_TOKENS=96`, "
                    "`TPU_DCP_MLA_OUT_VMEM_MAX_TOKENS=96` and "
                    "`TPU_FUSED_OPROJ_AR_MAX_ROWS=96`; otherwise "
                    "`TPU_DCP_MLA_UV_BEFORE_MERGE=1` or "
                    "`TPU_DCP_MERGE_VSPACE_FUSED=1` at decode row counts."),
                est_saving_share=(0.45, 0.62),
                requires_accuracy_gate=False,
            ),
            PlaybookLever(
                name="mla_fused_qb_absorb_layout",
                target_categories=("mla_proj", ),
                title=
                "Fuse `q_b` projection with `W_UK` absorb and direct layout",
                mechanism=(
                    "Eliminates intermediate `absorb_q` relayout copy while "
                    "staying under the 64 MiB tpu7x VMEM ceiling."),
                action=("Fuse `q_b` + `W_UK` epilogue with tiled head blocks "
                        "(avoiding full-head 68 MiB VMEM windows at DCP16)."),
                est_saving_share=(0.24, 0.37),
                requires_accuracy_gate=False,
            ),
            PlaybookLever(
                name="mla_oproj_ar_pipeline_and_row_cap",
                target_categories=("mla_oproj", ),
                title="Pipeline `fp8_uv_oproj_all_reduce` and extend max rows",
                mechanism=(
                    "`fp8_uv_oproj_all_reduce` drops to unfused fallback when "
                    "`padded_rows > TPU_FUSED_OPROJ_AR_MAX_ROWS`."),
                action=
                ("Raise `TPU_FUSED_OPROJ_AR_MAX_ROWS` / "
                 "`TPU_SMALL_ALL_REDUCE_MAX_ROWS` to cover target `cap*(1+K)`."
                 ),
                est_saving_share=(0.15, 0.30),
                requires_accuracy_gate=False,
            ),
        )

    def roofline(
        self,
        dims: ModelDims,
        topology: ServingTopology,
        hw: HardwareSpec,
        ctx: StepContext,
    ) -> PackRoofline:
        sharding = max(1, topology.chips_per_slice)
        dcp = max(1, topology.dcp_size)
        kv_rank = dims.kv_lora_rank or 512
        rope_dim = dims.qk_rope_dim or 64
        q_rank = dims.q_lora_rank or 1536

        proj_weights_layer = (
            dims.hidden_size *
            (q_rank + kv_rank + rope_dim) + q_rank * dims.num_heads *
            (kv_rank + rope_dim) + dims.num_heads * kv_rank * dims.v_head_dim +
            dims.num_heads * dims.v_head_dim * dims.hidden_size) / sharding

        attended_tokens = (min(ctx.context_tokens, 2048)
                           if dims.indexer_layers > 0 else ctx.context_tokens)
        kv_bytes_layer = (ctx.batch_reqs * attended_tokens *
                          (kv_rank + rope_dim) / dcp)
        hbm_bytes_step = dims.num_layers * (proj_weights_layer +
                                            kv_bytes_layer)
        hbm_floor_ms = hbm_bytes_step / (hw.hbm_bw_tbps_per_chip * 1e12) * 1e3

        if dcp > 1:
            rs_bytes_layer = (dims.num_heads * ctx.padded_rows * kv_rank *
                              2.0 * (dcp - 1) / dcp)
            lse_bytes_layer = dims.num_heads * ctx.padded_rows * 4.0 * dcp
        else:
            rs_bytes_layer = 0.0
            lse_bytes_layer = 0.0
        ar_bytes_layer = ctx.padded_rows * dims.hidden_size * 2.0
        ici_bytes_step = dims.num_layers * (rs_bytes_layer + lse_bytes_layer +
                                            ar_bytes_layer)
        ici_floor_ms = ici_bytes_step / (hw.ici_bw_tbps_per_chip * 1e12) * 1e3

        flops_step = (
            2.0 * dims.num_layers * ctx.padded_rows * proj_weights_layer +
            4.0 * dims.num_layers * ctx.batch_reqs *
            (dims.num_heads / sharding) * attended_tokens * kv_rank)
        compute_floor_ms = flops_step / (hw.fp8_tflops_per_chip * 1e12) * 1e3
        floor_ms, limiting = _dominant_resource(hbm_floor_ms, ici_floor_ms,
                                                compute_floor_ms)
        return PackRoofline(
            pack_name=self.name,
            buckets=("MLA_DCP", ),
            streamed_bytes_per_chip_step=hbm_bytes_step,
            ici_bytes_per_chip_step=ici_bytes_step,
            flops_per_chip_step=flops_step,
            hbm_floor_ms_per_step=hbm_floor_ms,
            ici_floor_ms_per_step=ici_floor_ms,
            compute_floor_ms_per_step=compute_floor_ms,
            floor_ms_per_step=floor_ms,
            limiting_resource=limiting,
            details={
                "attended_tokens_per_req": float(attended_tokens),
                "dcp_size": float(dcp),
                "proj_mb_per_chip_layer": proj_weights_layer / 1e6,
            },
        )


class SparseCoreIndexerPack(ArchPack):
    """SparseCore top-k / StreamIdx token-selection indexer pack."""

    name = "sparsecore_indexer"

    def op_rules(self) -> tuple[OpRule, ...]:
        return (
            OpRule(
                category="indexer_sc_wait",
                bucket="SparseCore_Indexer",
                pattern=r"sc_topk|sc_gather_pack|indexer sc top-k",
                scaling="padded_rows_x_ctx",
            ),
            OpRule(
                category="indexer_score",
                bucket="SparseCore_Indexer",
                pattern=r"streamidx|indexer streamidx",
                scaling="padded_rows_x_ctx",
            ),
            OpRule(
                category="indexer_comm",
                bucket="SparseCore_Indexer",
                pattern=r"indexer all_to_all|streamidx_topk_dcp.*all_",
                scaling="fixed",
            ),
            OpRule(
                category="indexer_proj",
                bucket="SparseCore_Indexer",
                pattern=r"indexer projections|indexer misc|^indexer$",
                scaling="padded_rows",
            ),
        )

    def levers(self) -> tuple[PlaybookLever, ...]:
        return (PlaybookLever(
            name="indexer_overlap_sc_waits_and_merge_comm",
            target_categories=(
                "indexer_sc_wait",
                "indexer_comm",
                "indexer_score",
            ),
            title=
            "Overlap SparseCore top-k waits behind `q_b` and merge indexer collectives",
            mechanism=(
                "TensorCore stalls ~42 us/indexer-layer on SparseCore "
                "`call-done` barriers while independent `q_b` matmuls sit "
                "scheduled after the wait."),
            action=("Schedule `q_b` inside the SparseCore async top-k window: "
                    "enable `TPU_STREAMIDX_DECODE_RING=1;"
                    "TPU_QB_OVERLAP_HONORS_SKIP_TOPK=1`, and with the "
                    "vllm-torchtpu #2851 code also "
                    "`TPU_DCP_INDEXER_A2A_SMALL_COLL=1;"
                    "TPU_INDEXER_OVERLAP_QB_ABSORB=1` (DCP=8 micro-benchmark: "
                    "137.4 -> 82.6 us/layer at 64 rows, 200.4 -> 170.3 us at "
                    "80 rows)."),
            est_saving_share=(0.28, 0.42),
            requires_accuracy_gate=False,
        ), )

    def roofline(
        self,
        dims: ModelDims,
        topology: ServingTopology,
        hw: HardwareSpec,
        ctx: StepContext,
    ) -> PackRoofline:
        idx_layers = dims.indexer_layers or 0
        dcp = max(1, topology.dcp_size)
        index_bytes_layer = ctx.batch_reqs * ctx.context_tokens * 128.0 / dcp
        proj_bytes_layer = (dims.hidden_size * 1024.0 /
                            max(1, topology.chips_per_slice))
        hbm_bytes_step = idx_layers * (index_bytes_layer + proj_bytes_layer)
        hbm_floor_ms = hbm_bytes_step / (hw.hbm_bw_tbps_per_chip * 1e12) * 1e3

        ici_bytes_step = idx_layers * ctx.padded_rows * 2048.0 * 4.0
        ici_floor_ms = ici_bytes_step / (hw.ici_bw_tbps_per_chip * 1e12) * 1e3
        flops_step = (2.0 * idx_layers * ctx.padded_rows *
                      (ctx.context_tokens / dcp) * 128.0)
        compute_floor_ms = flops_step / (hw.fp8_tflops_per_chip * 1e12) * 1e3
        floor_ms, limiting = _dominant_resource(hbm_floor_ms, ici_floor_ms,
                                                compute_floor_ms)
        return PackRoofline(
            pack_name=self.name,
            buckets=("SparseCore_Indexer", ),
            streamed_bytes_per_chip_step=hbm_bytes_step,
            ici_bytes_per_chip_step=ici_bytes_step,
            flops_per_chip_step=flops_step,
            hbm_floor_ms_per_step=hbm_floor_ms,
            ici_floor_ms_per_step=ici_floor_ms,
            compute_floor_ms_per_step=compute_floor_ms,
            floor_ms_per_step=floor_ms,
            limiting_resource=limiting,
            details={"indexer_layers": float(idx_layers)},
        )


class GDNHybridAttentionPack(ArchPack):
    """Hybrid Gated DeltaNet recurrent state + paged full attention."""

    name = "gdn_hybrid_attention"

    def op_rules(self) -> tuple[OpRule, ...]:
        return (
            OpRule(
                category="gdn_recurrent",
                bucket="GDN_Hybrid",
                pattern=r"gdn|deltanet|delta_rule|recurrent_state|conv1d_state",
                scaling="live_rows",
            ),
            OpRule(
                category="hybrid_full_attn",
                bucket="GDN_Hybrid",
                pattern=
                r"ragged_paged_attention|rpa|kv_split|paged_attn|flash_attn",
                scaling="live_rows",
            ),
            OpRule(
                category="hybrid_linear_proj",
                bucket="GDN_Hybrid",
                pattern=r"qkvz|qkv_proj|o_proj|out_proj|rms_norm|all_reduce",
                scaling="fixed",
            ),
        )

    def levers(self) -> tuple[PlaybookLever, ...]:
        return (
            PlaybookLever(
                name="gdn_w8a16_skinny_and_kv_split",
                target_categories=("hybrid_linear_proj", "hybrid_full_attn"),
                title=
                "Enable skinny W8A16 decode linears and KV-split paged attention",
                mechanism=(
                    "At <=64 decode rows, dynamic FP8 activation quantization "
                    "costs more than W8A16 weight-bound matmuls save; KV-split "
                    "parallelizes long-context paged reads."),
                action=(
                    "Set `TPU_FP8_LINEAR_W8A16_MAX_ROWS=64`, "
                    "`TPU_GDN_FRONT_IN_KERNEL=1`, and `QWEN35_KV_SPLIT_READ=1`."
                ),
                est_saving_share=(0.08, 0.22),
                requires_accuracy_gate=True,
            ),
            PlaybookLever(
                name="single_host_decode_priority_throttle",
                target_categories=("gdn_recurrent", "hybrid_full_attn"),
                title=
                "Throttle mixed chunked-prefill steps (`DECODE_PRIORITY`)",
                mechanism=(
                    "On single-host mixed serving, 8K-16K prefill chunks share "
                    "steps with active decodes and inflate p90 ITL."),
                action=(
                    "Enable `VLLM_TPU_DECODE_PRIORITY=4096:2:scale=8192:minb=4:"
                    "deadline=2000`."),
                est_saving_share=(0.10, 0.25),
                requires_accuracy_gate=False,
            ),
        )

    def roofline(
        self,
        dims: ModelDims,
        topology: ServingTopology,
        hw: HardwareSpec,
        ctx: StepContext,
    ) -> PackRoofline:
        sharding = max(1, topology.chips_per_slice)
        gdn_layers = dims.gdn_layers
        attn_layers = dims.full_attn_layers or max(
            0, dims.num_layers - gdn_layers)

        state_bytes_per_req_layer = dims.num_heads * 128.0 * 128.0 * 2.0 * 2.0
        gdn_state_bytes = (gdn_layers * ctx.batch_reqs *
                           state_bytes_per_req_layer / sharding)
        kv_bytes_per_tok_layer = 2.0 * 8.0 * dims.v_head_dim
        attn_kv_bytes = (attn_layers * ctx.batch_reqs * ctx.context_tokens *
                         kv_bytes_per_tok_layer / sharding)
        proj_bytes_layer = 4.0 * dims.hidden_size * dims.hidden_size / sharding
        proj_bytes = dims.num_layers * proj_bytes_layer

        hbm_bytes_step = gdn_state_bytes + attn_kv_bytes + proj_bytes
        hbm_floor_ms = hbm_bytes_step / (hw.hbm_bw_tbps_per_chip * 1e12) * 1e3
        ici_bytes_step = dims.num_layers * ctx.padded_rows * dims.hidden_size * 2.0
        ici_floor_ms = ici_bytes_step / (hw.ici_bw_tbps_per_chip * 1e12) * 1e3
        flops_step = 2.0 * dims.num_layers * ctx.padded_rows * proj_bytes_layer
        compute_floor_ms = flops_step / (hw.fp8_tflops_per_chip * 1e12) * 1e3
        floor_ms, limiting = _dominant_resource(hbm_floor_ms, ici_floor_ms,
                                                compute_floor_ms)
        return PackRoofline(
            pack_name=self.name,
            buckets=("GDN_Hybrid", ),
            streamed_bytes_per_chip_step=hbm_bytes_step,
            ici_bytes_per_chip_step=ici_bytes_step,
            flops_per_chip_step=flops_step,
            hbm_floor_ms_per_step=hbm_floor_ms,
            ici_floor_ms_per_step=ici_floor_ms,
            compute_floor_ms_per_step=compute_floor_ms,
            floor_ms_per_step=floor_ms,
            limiting_resource=limiting,
            details={
                "gdn_layers": float(gdn_layers),
                "full_attn_layers": float(attn_layers),
            },
        )


class MTPSpecDecodePack(ArchPack):
    """Multi-Token Prediction (MTP) drafter, `lm_head`, and rejection sampler."""

    name = "mtp_spec_decode"

    def op_rules(self) -> tuple[OpRule, ...]:
        return (
            OpRule(
                category="lm_head",
                bucket="MTP_Sampler",
                pattern=r"lm_head|,9680\]|,151936\]|,154880\]",
                scaling="fixed",
            ),
            OpRule(
                category="mtp_sampler_glue",
                bucket="MTP_Sampler",
                pattern=(r"sampler|verify|argmax|draft glue|lm head\+sampler|"
                         r"vocab_parallel_argmax|layernorm_l21"),
                scaling="fixed",
            ),
        )

    def levers(self) -> tuple[PlaybookLever, ...]:
        return (PlaybookLever(
            name="mtp_fused_draft_verify_loop",
            target_categories=("mtp_sampler_glue", "lm_head"),
            title="Fold MTP draft loop and trim rejection-sampler rows",
            mechanism=(
                "Separate XLA launches for MTP passes 1..K, `lm_head`, and "
                "rejection verify incur ~0.7-1.5 ms/step of dispatch and "
                "padded-row reductions."),
            action=("Enable `VLLM_TPU_FUSED_DRAFT_LOOP=1`, "
                    "`TPU_SAMPLER_VERIFY_TRIM_ROWS=32,40,48`, and "
                    "`TPU_LM_HEAD_HBM=drafter`."),
            est_saving_share=(0.25, 0.45),
            requires_accuracy_gate=False,
        ), )

    def roofline(
        self,
        dims: ModelDims,
        topology: ServingTopology,
        hw: HardwareSpec,
        ctx: StepContext,
    ) -> PackRoofline:
        sharding = max(1, topology.chips_per_slice)
        lm_head_passes = 2 if dims.mtp_draft_tokens > 0 else 1
        lm_head_bytes = (lm_head_passes * dims.hidden_size * dims.vocab_size /
                         sharding)
        hbm_floor_ms = lm_head_bytes / (hw.hbm_bw_tbps_per_chip * 1e12) * 1e3
        ici_bytes = (1 + dims.mtp_draft_tokens) * ctx.batch_reqs * 256.0
        ici_floor_ms = ici_bytes / (hw.ici_bw_tbps_per_chip * 1e12) * 1e3
        flops_step = (2.0 *
                      (ctx.padded_rows +
                       max(0, dims.mtp_draft_tokens - 1) * ctx.batch_reqs) *
                      dims.hidden_size * dims.vocab_size / sharding)
        compute_floor_ms = flops_step / (hw.fp8_tflops_per_chip * 1e12) * 1e3
        floor_ms, limiting = _dominant_resource(hbm_floor_ms, ici_floor_ms,
                                                compute_floor_ms)
        return PackRoofline(
            pack_name=self.name,
            buckets=("MTP_Sampler", ),
            streamed_bytes_per_chip_step=lm_head_bytes,
            ici_bytes_per_chip_step=ici_bytes,
            flops_per_chip_step=flops_step,
            hbm_floor_ms_per_step=hbm_floor_ms,
            ici_floor_ms_per_step=ici_floor_ms,
            compute_floor_ms_per_step=compute_floor_ms,
            floor_ms_per_step=floor_ms,
            limiting_resource=limiting,
            details={"mtp_draft_tokens": float(dims.mtp_draft_tokens)},
        )


PACK_REGISTRY: Mapping[str, type[ArchPack]] = {
    FineGrainedMoEPack.name: FineGrainedMoEPack,
    SparseMLADcpPack.name: SparseMLADcpPack,
    SparseCoreIndexerPack.name: SparseCoreIndexerPack,
    GDNHybridAttentionPack.name: GDNHybridAttentionPack,
    MTPSpecDecodePack.name: MTPSpecDecodePack,
}


@dataclasses.dataclass(frozen=True)
class Classification:
    """Classification of one trace op/group by a `ModelPlugin`."""

    pack_name: str
    bucket: str
    category: str
    scaling: str


@dataclasses.dataclass(frozen=True)
class ModelPlugin:
    """Declarative specification of a model architecture and serving topology."""

    name: str
    display_name: str
    dims: ModelDims
    topology: ServingTopology
    packs: tuple[ArchPack, ...]
    default_acceptance_length: float = 2.99
    hardware: HardwareSpec = TPU_7X
    aliases: tuple[str, ...] = ()

    def classify_op(self,
                    op_text: str,
                    group_text: str = "") -> Classification:
        combined = f"{op_text} {group_text}".strip()
        for candidate_text in (op_text, group_text, combined):
            if not candidate_text:
                continue
            for pack in self.packs:
                for rule in pack.op_rules():
                    if rule.matches(candidate_text):
                        return Classification(
                            pack_name=pack.name,
                            bucket=rule.bucket,
                            category=rule.category,
                            scaling=rule.scaling,
                        )
        if re.search(r"copy|slice|dma|reshape|transpose", combined,
                     re.IGNORECASE):
            return Classification("misc", "Misc", "dma_or_copy", "fixed")
        return Classification("misc", "Misc", "unclassified", "fixed")

    def rooflines(self, ctx: StepContext) -> dict[str, PackRoofline]:
        return {
            pack.name: pack.roofline(self.dims, self.topology, self.hardware,
                                     ctx)
            for pack in self.packs
        }

    def all_levers(self) -> tuple[PlaybookLever, ...]:
        levers: list[PlaybookLever] = []
        for pack in self.packs:
            levers.extend(pack.levers())
        return tuple(levers)


GLM5_MLA_MOE = ModelPlugin(
    name="glm5_mla_moe",
    display_name="GLM-5.2-FP8 Disaggregated P/D",
    dims=ModelDims(
        hidden_size=6144,
        num_layers=78,
        moe_layers=75,
        num_experts=256,
        top_k=8,
        moe_intermediate_size=2048,
        shared_experts=1,
        kv_lora_rank=512,
        qk_rope_dim=64,
        q_lora_rank=1536,
        num_heads=64,
        v_head_dim=128,
        indexer_layers=21,
        mtp_draft_tokens=3,
        vocab_size=154880,
    ),
    topology=ServingTopology(
        mode="disagg_pd",
        tp_size=16,
        ep_size=1,
        dcp_size=16,
        prefill_slices=1,
        decode_slices=1,
        chips_per_slice=8,
    ),
    packs=(
        FineGrainedMoEPack(),
        SparseMLADcpPack(),
        SparseCoreIndexerPack(),
        MTPSpecDecodePack(),
    ),
    default_acceptance_length=2.984,
    aliases=("glm52_pd", "glm5_pd", "1p1d_long_context"),
)

QWEN3_5_HYBRID_MOE = ModelPlugin(
    name="qwen3_5_hybrid_moe",
    display_name="Qwen3.5-397B-A17B-FP8 Single-Host",
    dims=ModelDims(
        hidden_size=4096,
        num_layers=60,
        moe_layers=60,
        num_experts=512,
        top_k=10,
        moe_intermediate_size=1024,
        shared_experts=1,
        gdn_layers=45,
        full_attn_layers=15,
        num_heads=32,
        v_head_dim=128,
        mtp_draft_tokens=3,
        vocab_size=151936,
    ),
    topology=ServingTopology(
        mode="single_host",
        tp_size=8,
        ep_size=8,
        dcp_size=1,
        prefill_slices=0,
        decode_slices=1,
        chips_per_slice=4,
    ),
    packs=(
        FineGrainedMoEPack(),
        GDNHybridAttentionPack(),
        MTPSpecDecodePack(),
    ),
    default_acceptance_length=3.39,
    aliases=("qwen35_single", "qwen3_5_single"),
)

DEEPSEEK_MLA_MOE = ModelPlugin(
    name="deepseek_mla_moe",
    display_name="DeepSeek-R1 FP8 Disaggregated P/D",
    dims=ModelDims(
        hidden_size=7168,
        num_layers=61,
        moe_layers=58,
        num_experts=256,
        top_k=8,
        moe_intermediate_size=2048,
        shared_experts=1,
        kv_lora_rank=512,
        qk_rope_dim=64,
        q_lora_rank=1536,
        num_heads=128,
        v_head_dim=128,
        indexer_layers=61,
        mtp_draft_tokens=3,
        vocab_size=129280,
    ),
    topology=ServingTopology(
        mode="disagg_pd",
        tp_size=16,
        ep_size=16,
        dcp_size=16,
        prefill_slices=1,
        decode_slices=1,
        chips_per_slice=8,
    ),
    packs=(
        FineGrainedMoEPack(),
        SparseMLADcpPack(),
        SparseCoreIndexerPack(),
        MTPSpecDecodePack(),
    ),
    default_acceptance_length=3.10,
    aliases=("deepseek_r1_pd", "deepseek_pd"),
)

DEFAULT_PLUGIN_NAME = GLM5_MLA_MOE.name

_PLUGINS: dict[str, ModelPlugin] = {}


def register_plugin(plugin: ModelPlugin) -> ModelPlugin:
    """Registers `plugin` under its canonical name and aliases."""
    _PLUGINS[plugin.name] = plugin
    for alias in plugin.aliases:
        _PLUGINS[alias] = plugin
    return plugin


for _builtin in (GLM5_MLA_MOE, QWEN3_5_HYBRID_MOE, DEEPSEEK_MLA_MOE):
    register_plugin(_builtin)

PLUGINS: Mapping[str, ModelPlugin] = _PLUGINS


def load_plugin_from_dict(raw: Mapping[str, Any]) -> ModelPlugin:
    """Builds a `ModelPlugin` from a declarative dictionary or JSON overlay."""
    base_name = str(raw.get("base", DEFAULT_PLUGIN_NAME))
    base = _PLUGINS.get(base_name, GLM5_MLA_MOE)

    dims_raw = raw.get("dims")
    dims = base.dims
    if isinstance(dims_raw, Mapping):
        dims_kwargs = {
            field.name: int(dims_raw[field.name])
            for field in dataclasses.fields(ModelDims)
            if field.name in dims_raw
        }
        dims = dataclasses.replace(base.dims, **dims_kwargs)

    topo_raw = raw.get("topology")
    topology = base.topology
    if isinstance(topo_raw, Mapping):
        topo_kwargs: dict[str, Any] = {}
        for field in dataclasses.fields(ServingTopology):
            if field.name not in topo_raw:
                continue
            val = topo_raw[field.name]
            topo_kwargs[field.name] = (str(val)
                                       if field.name == "mode" else int(val))
        topology = dataclasses.replace(base.topology, **topo_kwargs)

    if "packs" in raw:
        packs_list: list[ArchPack] = []
        for pack_name in raw["packs"]:
            key = str(pack_name)
            if key not in PACK_REGISTRY:
                known = ", ".join(sorted(PACK_REGISTRY))
                raise KeyError(f"Unknown ArchPack {key!r}; known: {known}")
            packs_list.append(PACK_REGISTRY[key]())
        packs = tuple(packs_list)
    else:
        packs = base.packs

    aliases = (tuple(
        str(a) for a in raw["aliases"]) if "aliases" in raw else base.aliases)
    return ModelPlugin(
        name=str(raw.get("name", base.name)),
        display_name=str(raw.get("display_name", base.display_name)),
        dims=dims,
        topology=topology,
        packs=packs,
        default_acceptance_length=float(
            raw.get("default_acceptance_length",
                    base.default_acceptance_length)),
        hardware=base.hardware,
        aliases=aliases,
    )


def get_plugin(name_or_path: str = DEFAULT_PLUGIN_NAME) -> ModelPlugin:
    """Looks up a registered `ModelPlugin` by name/alias or loads JSON path."""
    if name_or_path in _PLUGINS:
        return _PLUGINS[name_or_path]
    candidate = pathlib.Path(name_or_path)
    if name_or_path.endswith(".json") or candidate.is_file():
        raw = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"Plugin JSON at {name_or_path} must be an object.")
        return load_plugin_from_dict(raw)
    known = ", ".join(sorted({p.name for p in _PLUGINS.values()}))
    raise KeyError(f"Unknown model plugin {name_or_path!r}; known: {known}")
