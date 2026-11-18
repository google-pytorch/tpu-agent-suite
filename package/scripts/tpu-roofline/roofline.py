#!/usr/bin/env python3
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
"""The roofline script. Fixed arithmetic, model supplied as data.

    python3 scripts/roofline.py scripts/configs/<cfg>.json
    python3 scripts/roofline.py <cfg>.json --phase prefill --csv out.csv
    python3 scripts/roofline.py <cfg>.json --layer-csv layers.csv
    python3 scripts/roofline.py --selftest

Two tables come out, and they are the same numbers grouped two ways: the flat
phase table (every op, by cost, multiplicity applied) says which op is
expensive; the per-layer tables (`layer_rows`, one representative layer of each
KIND, in dataflow order) say how a layer executes. This file prints the layer
kind's key and never a prose name -- naming is the tpu-roofline prompt's, because it is model
vocabulary and not arithmetic. README.md s1 has the row roles and the two
properties a merged row does not have.

Why this is a script and not an agent's arithmetic
==================================================
Measured 2026-09-10: five agents given the same pins, hardware and instruction
returned prefill anchors of 19 ms, 6,216, 7,641, 10,554 and 2,410,850 ms. A
byte-for-byte re-dispatch moved one config by 1,118x. The same agents reproduced
the *pins* to within 4%. Agents transcribe configuration reliably and derive
cost unreliably. So: this file holds the arithmetic and is not regenerated per
model; a config holds only data. Adding a model is transcription.

The model, in one line
======================
    roofline_ms = max(hbm_ms, compute_ms) + ici_ms          per op, per layer

Compute and HBM are assumed perfectly overlapped *within* an op; collectives are
assumed fully exposed. See README.md for every term and its caveats.

See also: model/common/<op>/ for op shapes, parallelism/<scheme>/ for which
collective fires and how often, COLLECTIVE.md for how one is priced, tpu/ for
hardware pins.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import MISSING, dataclass, field, fields, replace
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cost_estimator import collective as cc  # noqa: E402

GiB, MiB = 1 << 30, 1 << 20

# ============================================================ config

#: The EP dispatch strategies `Model.moe_dispatch_strategy` selects between.
#: Declared here so `Model.__post_init__` can validate against it.
MOE_DISPATCH = ("all_to_all", "all_gather", "hierarchical_ag")

#: The DCP indexer candidate exchanges `Model.dcp_indexer_gather` selects between.
DCP_INDEXER_GATHER = ("two_ag", "single_ag")

#: The PCP KV layouts `Model.pcp_kv_strategy` selects between.
PCP_KV_STRATEGY = ("ring", "allgather")


@dataclass(frozen=True)
class LayerClass:
    """One KIND of transformer layer, and how many of it the model stacks.

    `op_classes` is the set of `Op.layer_class` values that run inside this kind, which
    is what lets one representative layer be assembled from the flat op list. The two
    vocabularies are deliberately different: an op says *which layers run me*, a layer
    kind says *which ops run in me*, and the roll-up checks they agree.

    `key` is an identifier, not a display name. What a reader should be shown --
    "Full Indexer MoE Layer" for `moe_idx` -- is a reporting decision and lives in
    the tpu-roofline prompt, because it is the one part of this that is model-vocabulary rather than
    arithmetic. Nothing in this file prints a prose label.
    """

    key: str  # "dense_idx" | "dense" | "moe_idx" | "moe"
    count: int  # how many layers of this kind
    op_classes: frozenset[str]  # Op.layer_class values that run in it


@dataclass(frozen=True)
class Model:
    name: str
    layers: int
    dense_layers: int  # first_k_dense_replace
    indexer_layers: int  # layers that actually RUN the indexer
    hidden: int
    heads: int
    # MLA (kv_lora_rank > 0) or GQA
    kv_lora_rank: int = 0
    q_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    kv_heads: int = 0
    head_dim: int = 0
    # DSA indexer
    indexer_heads: int = 0
    indexer_head_dim: int = 0
    indexer_topk: int = 0
    indexer_cache_bytes_per_token: float = 0.0
    #: WHICH layers run the indexer. vLLM builds one on layer l iff
    #: max(l - offset + 1, 0) % freq == 0 (vllm/model_executor/models/deepseek_v2.py:
    #: 1104-1105; offset defaults to 2 when the config omits it). GLM-5.2: freq 4,
    #: offset 3 -> layers 0,1,2,6,10,...,74. 0 = unknown: the `indexer_layers` count is
    #: spread evenly instead, a guess that only a pipeline partition can see.
    indexer_freq: int = 0
    indexer_offset: int = 2
    #: Does the indexer kernel fuse QK -> score-aggregation -> top-K, keeping the
    #: [T, S] score matrix off HBM? `model/common/attention/dsa` asserts it does,
    #: unconditionally. That holds only while the matrix fits VMEM (8 MiB on v7x)
    #: -- at T=1024, S=128K it is 256 MiB, 32x over, and the scores
    #: must spill. False (the default) charges the spill: written by the weighted
    #: sum, read back by top-K. See `dsa_weighted_sum`.
    indexer_fused_scores: bool = False
    #: How sparse MLA reaches the KV cache. Governs BOTH the keys each query scans
    #: (compute) and the distinct entries pulled from HBM (memory) -- see `Ctx` and
    #: `kv_fetched`. The two strategies differ by up to 64x at prefill, so this is
    #: never a detail.
    #:   "gathered"     Sparse Gathered QK+AV. Fetch and compute over the top-K
    #:                  entries only: scan = K_idx, fetch = min(S, T_seq * K_idx).
    #:   "masked_dense" Masked Dense QK+AV. Fetch and compute the whole context and
    #:                  mask the unselected: scan = fetch = S.
    sparse_attention_strategy: str = "gathered"
    #: How EP gets each token to the device holding its experts.
    #: `parallelism/ep/ep.md` s4 has the derivation and the shapes.
    #:   "all_to_all"      A. All-to-all the EXPANDED [B, kT, D] set over the EP
    #:                     group. Payload is `topk`x bigger than the other two.
    #:   "all_gather"      B. All-gather the UNEXPANDED [B, T, D] tokens, then sort
    #:                     to expert order and slice locally. The default.
    #:   "hierarchical_ag" C. As B, but as two explicit phases -- gather ACROSS chips
    #:                     over ICI first, then the chip's own cores over d2d last.
    #:                     An all-gather's payload grows, so the fast link goes last;
    #:                     `coll_ep_dispatch` has the reasoning.
    moe_dispatch_strategy: str = "all_gather"
    #: How the DCP indexer exchanges each rank's top-K candidates. Read only when
    #: `dcp > 1` on an indexer model. `parallelism/dcp/dcp.md` s7.2.
    #:   "two_ag"     Indices and scores gathered separately, [T, K_idx] each, 8 B per
    #:                candidate. What vllm-torchtpu main runs today. The default.
    #:   "single_ag"  Scores plus one source-rank column, [T, K_idx + 1], in ONE
    #:                gather -- a rank only scores its own shard, so the indices never
    #:                need to leave it. ~4 B per candidate. vllm-torchtpu PR #1230.
    dcp_indexer_gather: str = "two_ag"
    #: How PCP gives each rank the keys its local queries need. `parallelism/pcp/
    #: pcp.md` s4-s6. Read only when `pcp > 1`. MLA has NO PCP path on TPU; this is
    #: what an MLA row is priced as, for evaluation.
    #:   "ring"       Striped cache (page p on rank p % PCP), KV pages streamed around
    #:                the PCP ring inside the kernel. KV / device divides by PCP; the
    #:                ring moves every page the step reads. What main runs for GQA.
    #:   "allgather"  Replicated cache; each rank all-gathers the step's NEW rows. No
    #:                KV division; the wire carries new tokens only. Not on main.
    pcp_kv_strategy: str = "ring"
    #: Is the gated activation (SwiGLU) a fused epilogue of the up/gate GEMM?
    #: True  -- the GEMM keeps gate and up in VMEM, applies SiLU(gate)*up and writes
    #:          ONE [.., F] tensor. No separate SwiGLU op exists.
    #: False -- it writes both halves [2, .., F] and a separate SwiGLU op reads them
    #:          back. 2x the output write plus a whole extra pass.
    #: Worth 144 MiB/layer on GLM-5.2's MoE block. Applies to dense FFN, routed
    #: experts and the shared expert alike -- they are the same kernel shape.
    #:
    #: **Defaults False: fusion is a claim about the kernel, so it is opted into.**
    #: Asserting it by default understates HBM on every gated FFN in every model,
    #: and the evidence is not in the op tables -- a reference sheet's activation
    #: column is usually a tensor INVENTORY (LHS + full logical output), which reads
    #: the same either way. The discriminator is the Memory Time: on GLM-5.2's MoE
    #: FFi, 1.93953 ms means fused and 1.94951 means not.
    #: Split per block because they are separate kernels and GLM-5.2's reference
    #: sheet fuses one and not the other: dense unfused, both MoE paths fused. A
    #: single flag could not say that.
    dense_fused_activation: bool = False
    moe_fused_activation: bool = False
    # FFN / MoE
    dense_ffn: int = 0
    n_experts: int = 0
    topk: int = 0
    expert_ffn: int = 0
    n_shared_experts: int = 0
    shared_ffn: int = 0
    vocab: int = 0
    tie_embeddings: bool = False

    @property
    def moe_layers(self) -> int:
        return self.layers - self.dense_layers

    @property
    def use_mla(self) -> bool:
        return self.kv_lora_rank > 0

    def indexer_layer_ids(self) -> tuple[int, ...]:
        """The layers that run the indexer, by vLLM's rule when it is pinned."""
        if not self.indexer_layers:
            return ()
        if self.indexer_freq:
            f, o = self.indexer_freq, self.indexer_offset
            return tuple(i for i in range(self.layers)
                         if max(i - o + 1, 0) % f == 0)
        n, k = self.layers, self.indexer_layers  # unpinned: spread evenly
        return tuple(i for i in range(n) if (i * k) // n != ((i + 1) * k) // n)

    @property
    def counts(self) -> dict[str, int]:
        """Layer multiplicities. `moe_idx` layers run the indexer, `moe_plain` reuse it."""
        idx_on_dense = min(self.dense_layers, self.indexer_layers)
        return {
            "dense": self.dense_layers,
            "moe_idx": self.indexer_layers - idx_on_dense,
            "moe_plain":
            self.moe_layers - (self.indexer_layers - idx_on_dense),
        }

    def layer_classes(self) -> tuple[LayerClass, ...]:
        """The distinct KINDS of transformer layer, each with how many of it there are.

        `counts` answers "how many layers of each multiplicity term", which is what the
        roll-up needs. This answers "what kinds of layer does this model have", which is
        what a reader needs: one table per kind beats 78 identical ones, and the kinds
        are exactly where the per-layer cost actually differs.

        The split is two independent binary facts -- dense vs MoE FFN, and indexer vs
        index-reuse -- so there are up to four, and which ones are non-empty is a
        property of the checkpoint, not of this code. GLM-5.2 (78 layers, 3 dense,
        indexer on 21) gives three: 3 x dense+indexer, 18 x MoE+indexer, 57 x MoE
        reusing the indices. A model whose indexer skipped some dense layers would
        light up the fourth. Nothing here is keyed on a model name.

        Empty classes are dropped, so a plain dense model returns one row and a plain
        MoE model two.
        """
        idx_on_dense = min(self.dense_layers, self.indexer_layers)
        spec = (
            ("dense_idx", idx_on_dense, ("all", "dense", "indexer")),
            ("dense", self.dense_layers - idx_on_dense, ("all", "dense")),
            ("moe_idx", self.indexer_layers - idx_on_dense, ("all", "moe",
                                                             "indexer")),
            ("moe", self.moe_layers - (self.indexer_layers - idx_on_dense),
             ("all", "moe")),
        )
        return tuple(
            LayerClass(k, n, frozenset(oc)) for k, n, oc in spec if n > 0)

    def __post_init__(self) -> None:
        if self.dense_layers > self.layers or self.indexer_layers > self.layers:
            raise ValueError(f"{self.name}: layer counts exceed `layers`")
        if sum(self.counts.values()) != self.layers:
            raise ValueError(
                f"{self.name}: layer classes do not sum to {self.layers}")
        if sum(lc.count for lc in self.layer_classes()) != self.layers:
            raise ValueError(
                f"{self.name}: layer kinds do not sum to {self.layers}")
        if self.indexer_freq and len(
                self.indexer_layer_ids()) != self.indexer_layers:
            raise ValueError(
                f"{self.name}: indexer_freq={self.indexer_freq}, indexer_offset="
                f"{self.indexer_offset} selects {len(self.indexer_layer_ids())} layers, "
                f"but indexer_layers={self.indexer_layers}")
        if self.moe_dispatch_strategy not in MOE_DISPATCH:
            raise ValueError(
                f"{self.name}: moe_dispatch_strategy must be one of "
                f"{sorted(MOE_DISPATCH)}, got {self.moe_dispatch_strategy!r}")
        if self.dcp_indexer_gather not in DCP_INDEXER_GATHER:
            raise ValueError(
                f"{self.name}: dcp_indexer_gather must be one of "
                f"{sorted(DCP_INDEXER_GATHER)}, got {self.dcp_indexer_gather!r}"
            )
        if self.pcp_kv_strategy not in PCP_KV_STRATEGY:
            raise ValueError(
                f"{self.name}: pcp_kv_strategy must be one of "
                f"{sorted(PCP_KV_STRATEGY)}, got {self.pcp_kv_strategy!r}")
        if self.sparse_attention_strategy not in ("gathered", "masked_dense"):
            raise ValueError(
                f"{self.name}: sparse_attention_strategy must be 'gathered' or "
                f"'masked_dense', got {self.sparse_attention_strategy!r}")


#: `act_in` is the quantized GEMM input (fp8 under w8a8); `act_out` is the
#: accumulator write, normally wider. Conflating them understates prefill HBM.
#: `indexer_w_proj{,_act}` exist because the indexer's head-weights projection is not
#: quantized with the rest of it: the reference sheet has the K and Q projections at
#: w8a8 and this one at w16a16. Worth 0.004 GiB and ~0.0002 ms/layer on GLM-5.2, so it
#: is here for fidelity, not for the number -- but a site that does not exist cannot be
#: stated, and an unstatable difference is one nobody can check.
SITES = ("weights", "moe_weights", "kv_cache", "indexer_cache", "act_in",
         "act_out", "attn_qk", "attn_av", "indexer_qk", "indexer_av",
         "indexer_w_proj", "indexer_w_proj_act", "kv_write", "indexer_write",
         "indexer_indices", "tp_allreduce", "expert_dispatch",
         "expert_collect", "router_logits", "dcp_gather", "dcp_lse",
         "dcp_combine")

#: A site that falls back to ANOTHER site rather than to `Numerics.default`.
#: `kv_write` is the dtype the KV projection's GEMM emits. Left unset it is the
#: cache's own dtype -- i.e. a fused quantize epilogue, the GEMM writing cache
#: entries directly. Set it wider (bf16 over an fp8 cache) and the GEMM is writing a
#: STAGING buffer that something must still quantize; `gemm` detects that and says so.
_SITE_FALLBACK = {"kv_write": "kv_cache", "indexer_write": "indexer_cache"}

#: A site whose sensible default is NOT `Numerics.default`. Top-K emits positions,
#: not values, so it never follows the model's activation dtype -- see
#: TYPE_PROMOTION.md "Selection outputs are not promoted values".
_SITE_DEFAULT = {
    "indexer_indices": "int32",
    # DCP's merge: LSEs are cast to f32 and the LSE-weighted partials are
    # reduce-scattered in f32, cast back only afterwards
    # (vllm-torchtpu `cp_mla_attention.py:93-110`). Not the activation dtype.
    "dcp_lse": "fp32",
    "dcp_combine": "fp32"
}


@dataclass(frozen=True)
class Numerics:
    default: str = "bf16"
    sites: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bad = set(self.sites) - set(SITES)
        if bad:
            raise ValueError(
                f"unknown numerics site(s) {sorted(bad)}; known {list(SITES)}")

    def b(self, site: str) -> float:
        return cc.dtype_bytes(self.dt(site))

    def dt(self, site: str) -> str:
        if site not in SITES:
            raise ValueError(f"unknown site {site!r}")
        if site in self.sites:
            return self.sites[site]
        fb = _SITE_FALLBACK.get(site)
        if fb:
            return self.dt(fb)
        return _SITE_DEFAULT.get(site, self.default)

    @property
    def unset(self) -> list[str]:
        return [s for s in SITES if s not in self.sites]


@dataclass(frozen=True)
class Hardware:
    # Every hardware value is required and comes from the config, so a run uses
    # exactly the numbers its config names and nothing hidden here. Values are
    # per core (chiplet). The config's `_provenance.hardware` records the source
    # of each one: the knowledge base through tpu-knowledge-base for published
    # figures, a measurement with its job id, or `assumed`.
    gen: str
    slice_dims: tuple[int, ...]
    d2d: bool
    d2d_bw: float  # die-to-die link, one way
    d2d_hop_latency: float  # per die-to-die hop [s]
    ici_bw: float  # one ICI link, one way
    #: Divides each chiplet's share of an ICI link: 2.0 when the two chiplets of a
    #: package share its links. Applied only with `d2d`.
    ici_contention: float
    hop_latency: float  # per ICI hop [s]; collectives and the PP handoff
    hbm_bw: float
    hbm_capacity: float
    peak_fp8: float
    peak_bf16: float
    # An elementwise op runs on the VPU, not the MXU.
    peak_vpu_bf16: float
    peak_vpu_fp32: float
    mxu_min_dim: int  # contracting dim is ceil-padded to this
    #: Pooled VMEM per SparseCore -- the working-set denominator that decides whether
    #: an intermediate can be kept off HBM. Every "fused" / "kept in VMEM" claim in
    #: `model/common/` is a comparison against this (REQUIREMENTs.md s Fusion).
    sparsecore_vmem_bytes: float
    # vLLM's `--gpu-memory-utilization` (that name on TPU too). The KV pool is
    #   cap = hbm_capacity * memory_utilization
    #   kv  = cap - weights - resident_overhead
    # per vllm-torchtpu/src/vllm_torchtpu/utils.py:171 compute_hbm_budget, where
    # `total_used` is everything already resident: weights, compiled graphs, ...
    memory_utilization: float
    resident_overhead_gb: float  # compiled graphs + workspace, beyond weights
    launch_overhead: float = 0.0
    kernel_efficiency: float = 1.0  # 1.0 => a ceiling, not a prediction

    def vpu_peak(self, dtype: str) -> float:
        p = self.peak_vpu_fp32 if cc.dtype_bytes(
            dtype) >= 4.0 else self.peak_vpu_bf16
        return p * self.kernel_efficiency

    def peak(self, *dtypes: str) -> float:
        """MXU rate for a matmul, from ALL its operands.

        The narrow rate needs *every* operand narrow. A w8a16 GEMM -- fp8
        weights, bf16 activations -- runs at the bf16 rate, not fp8: one wide
        operand is enough to halve the pass. Keying this on the weight dtype
        alone makes such a model 2x optimistic on every GEMM.
        """
        widest = max(cc.dtype_bytes(d) for d in dtypes)
        p = self.peak_fp8 if widest <= 1.0 else self.peak_bf16
        return p * self.kernel_efficiency

    def machine(self) -> cc.Machine:
        kw: dict[str, Any] = {"hop_latency": self.hop_latency}
        if self.launch_overhead:
            kw["launch_overhead"] = self.launch_overhead
        return cc.tpu(self.gen,
                      self.slice_dims,
                      link_bw_oneway=self.ici_bw,
                      d2d=self.d2d,
                      d2d_bw=self.d2d_bw,
                      d2d_hop_latency=self.d2d_hop_latency,
                      ici_contention=self.ici_contention,
                      **kw)

    @property
    def world(self) -> int:
        n = 1
        for d in self.slice_dims:
            n *= d
        return n * (2 if self.d2d else 1)


@dataclass(frozen=True)
class Sharding:
    tp: int = 1
    dp: int = 1
    pp: int = 1
    pcp: int = 1
    dcp: int = 1
    ep: bool = False
    sp: bool = False
    #: Layers per pipeline stage -- `VLLM_PP_LAYER_PARTITION` (vllm/distributed/
    #: utils.py:143). Empty: `get_pp_indices`'s default. parallelism/pp/pp.md s3.
    pp_layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        # a JSON list arrives as a list; the dataclass is frozen and used as a key
        object.__setattr__(self, "pp_layers",
                           tuple(int(x) for x in self.pp_layers))

    @property
    def label(self) -> str:
        s = f"tp{self.tp}-dp{self.dp}"
        for k, v in (("pp", self.pp), ("pcp", self.pcp), ("dcp", self.dcp)):
            if v > 1:
                s += f"-{k}{v}"
        if self.pp_layers:
            s += "[" + ",".join(map(str, self.pp_layers)) + "]"
        return s + ("-ep" if self.ep else "") + ("-sp" if self.sp else "")

    def ep_group(self) -> int:
        return self.tp * self.pcp * self.dp if self.ep else 1

    def check(self, hw: Hardware, model: "Model | None" = None) -> str | None:
        """Describe why vLLM would refuse this sharding, or None. Informational:
        nothing in this script declines to price a config.

        `model` adds the rules that depend on the checkpoint. parallelism/parallelism.md
        Legality has each one with its source."""
        if self.tp * self.pcp * self.pp * self.dp != hw.world:
            return (f"TP*PCP*PP*DP = {self.tp*self.pcp*self.pp*self.dp}"
                    f" != world {hw.world}")
        if self.pcp > 1 and self.dp > 1:
            return "PCP does not support data parallelism"
        if self.pcp == 1 and self.tp % self.dcp:
            return f"DCP reuses TP ranks: tp {self.tp} % dcp {self.dcp} != 0"
        if self.sp and self.pp > 1:
            return "the SP MoE path applies at PP=1"
        if self.pcp > 1 and self.dcp > 1:
            return "PCP runner path does not support DCP (vllm-torchtpu pcp_validation)"
        if self.pcp > 1 and self.pp > 1:
            return "PCP runner path does not support PP (vllm-torchtpu pcp_validation)"
        if self.pp > 1 and self.tp > 1:
            return "PP does not support tensor parallelism inside a stage (pp_validation)"
        if self.pp_layers:
            if len(self.pp_layers) != self.pp or min(self.pp_layers) < 1:
                return (
                    f"pp_layers {list(self.pp_layers)} needs {self.pp} entries, "
                    f"each >= 1")
            if model is not None and sum(self.pp_layers) != model.layers:
                return (
                    f"pp_layers sums to {sum(self.pp_layers)}, the model has "
                    f"{model.layers} layers")
        if model is not None and self.pcp > 1:
            if model.n_experts and model.moe_layers and not self.ep:
                return "PCP on an MoE model requires --enable-expert-parallel"
            if model.use_mla:
                return (
                    f"MLA has no PCP path on TPU; priced as "
                    f"pcp_kv_strategy={model.pcp_kv_strategy!r} for evaluation"
                )
            if model.head_dim == 64:
                return "the PCP ring kernel does not support head_dim 64"
        if model is not None and self.dcp > 1:
            if not model.use_mla:
                # vllm/config/model.py:1335-1362 -- the DCP group must share one KV
                # head TP had replicated
                hkv, hq = max(1, model.kv_heads), model.heads
                if self.tp <= hkv:
                    return f"GQA DCP needs tp {self.tp} > H_kv {hkv}"
                if self.dcp > self.tp // hkv:
                    return f"GQA DCP needs dcp {self.dcp} <= tp // H_kv = {self.tp // hkv}"
                if (hq // hkv) % self.dcp:
                    return f"GQA DCP needs (H_q / H_kv) {hq // hkv} % dcp {self.dcp} == 0"
            elif not model.indexer_topk:
                return (
                    "dense MLA has no DCP path on TPU (the non-sparse branch never "
                    "reads dcp_size); not refused, not implemented")
        return None


@dataclass(frozen=True)
class Phase:
    """One forward pass shape, per core."""
    name: str  # "prefill" | "decode..." -- the decode rules key off the prefix
    tokens: int = 0  # tokens this core processes in the pass. 0 => `mnbt`.
    #  Both phases derive from the same budget, because both spend it:
    #    prefill  one pass IS the budget, so `tokens = mnbt` and `batch` is 1.
    #    decode   the budget buys one token per request, so `batch = mnbt` and
    #             `tokens` equals it.
    #  Give `tokens` explicitly on prefill only to model a WHOLE prompt that `mnbt`
    #  then chunks -- `tokens=8192, mnbt=1024` is eight passes of 1024, which is a
    #  different question from one pass of 1024.
    kv_len: int = 0  # context length each query attends over
    batch: int = 0  # sequences on this core -- the DECODE BATCH SIZE. 0 => derive
    #  A decode step advances every running request by one token, so `batch` IS the
    #  running set and `tokens` equals it. Left at 0 on a decode phase, `batch`
    #  defaults to `mnbt`: the step's token budget is spent one token per request, so
    #  the budget and the batch are the same number. Set it to override -- which you
    #  will often need to, because **mnbt is rarely the binding gate**. The KV pool
    #  usually admits far fewer (4 req/core at 128K on GLM-5.2 against mnbt 1024),
    #  and this dataclass does not know the pool. `kv_pool_bytes`/`kv_capacity_reqs`
    #  do; `serving.py` turns them into a real admission gate.
    mnbt: int = 0  # the per-STEP token budget this phase gets.
    #  vLLM: `token_budget = max_num_scheduled_tokens`, which defaults to
    #  `max_num_batched_tokens` (config/scheduler.py:59-61). The scheduler
    #  decrements it across ALL requests in the step -- running (decode) first,
    #  then waiting (prefill) -- so a prefill chunk is `mnbt - decode_tokens`,
    #  not `mnbt` (v1/core/sched/scheduler.py:459,485,637,687).
    #  For a PURE-prefill phase there are no decode tokens, so chunk = mnbt.
    #  For a mixed step, set mnbt to the budget left after decode.
    #  `long_prefill_token_threshold` caps one request's chunk further; 0=off.
    #  0 or >= tokens => a single un-chunked pass.
    kv_fetched: int = 0  # DISTINCT KV tokens read from HBM this pass, per core.

    #  Sparsity buys compute, not always bandwidth. With `tokens` queries each
    #  selecting `indexer_topk` keys, prefill's union is usually the whole
    #  context (so kv_fetched = kv_len) while decode fetches topk per request
    #  (so kv_fetched = batch * min(kv_len, topk)). 0 => derive that way.

    def __post_init__(self) -> None:
        """Fill `batch` and `tokens` where the config left them to be derived.

        Done here rather than in `load` so a `Phase` built in code -- by a step
        model, a sweep, a test -- resolves the same way a config does. Frozen, so
        `object.__setattr__`; the alternative is every construction site repeating
        the rule and one of them getting it wrong.
        """
        b, t = self.batch, self.tokens
        if self.name.startswith("decode"):
            b = b or self.mnbt or 1
            t = t or b  # one token per running request
        else:
            b = b or 1
            t = t or self.mnbt  # one pass IS the token budget
        if t <= 0:
            raise ValueError(
                f"{self.name}: cannot derive `tokens` -- give it, or give `mnbt` "
                f"(prefill: one pass of the budget; decode: one token per request)"
            )
        object.__setattr__(self, "batch", b)
        object.__setattr__(self, "tokens", t)


@dataclass(frozen=True)
class Workload:
    """The request shape. Needed for TTFT/TPOT/throughput -- see README s5."""
    isl: int  # input (prompt) tokens per request
    osl: int  # output tokens per request
    max_num_seqs: int | None = None  # --max-num-seqs per DP engine; serving.py


@dataclass(frozen=True)
class Config:
    model: Model
    numerics: Numerics
    hardware: Hardware
    sharding: Sharding
    phases: tuple[Phase, ...]
    workload: Workload | None = None


# ============================================================ one op


@dataclass
class Op:
    """One priced operation, per core, for one pass.

    HBM traffic is carried as three named parts rather than one total, and `hbm_bytes`
    is their sum. That is what makes the per-layer table's occupancy columns checkable:
    `activations + weights + KV` is exactly the byte count behind `mem_ms`, so a reader
    can divide by the HBM bandwidth and land on the Memory Time column. Keeping only
    the total, as this did, forced every consumer that needed the split -- `mixed_step`
    charging one fused step's weights once, the occupancy columns -- to re-derive it.

    ICI is carried the same way. `collective.py` already separates the alpha term from
    the beta term; collapsing them to `total` at the boundary threw away the one fact
    that says whether a collective shrinks when you make the payload smaller.
    """

    name: str
    layer_class: str  # "all" | "dense" | "moe" | "indexer"
    flops: float  # 2*M*K*N -- an operation count, dtype-independent
    dtype: str  # operand A. Picks the MXU peak, with operand B...
    # ---- HBM traffic, per core, this pass. Weights do not scale with the tokens in
    #      the pass; activations and KV do. `mixed_step` needs them apart.
    w_bytes: float = 0.0
    act_bytes: float = 0.0
    kv_bytes: float = 0.0  # KV cache and the DSA indexer cache -- TRAFFIC,
    #  the bytes this op actually reads, so it feeds `mem_ms`.
    act_inventory_bytes: float | None = None  # the activation TENSORS this op
    #  involves -- inputs plus the FULL logical output, gate and up both counted even
    #  when a fused epilogue means only one reaches HBM. The occupancy column reports
    #  this; `act_bytes` is the traffic and feeds `mem_ms`. None => same as traffic,
    #  which is every op that is not a gated GEMM.
    kv_resident_bytes: float = 0.0  # the cache this op ATTENDS OVER, resident.
    #  A different quantity, and the one the occupancy column reports. Under sparse
    #  attention they are 64x apart at 128K: the op reads `B * topk` tokens out of a
    #  `B * S` cache. Reporting traffic in a column called "Memory Occupancy" put a
    #  number 64x off next to a layer-sums column of the same name.
    # ---- collective, split as `collective.py` splits it. `t_fixed` (launch overhead)
    #      is folded into latency: it is a fixed cost that no payload shrinks.
    ici_lat_ms: float = 0.0
    ici_bw_ms: float = 0.0
    dtype_b: str | None = None  # ...when they differ. Widest operand wins.
    unit: str = "mxu"  # "mxu" | "vpu" -- 16x apart on this chip
    kind: str = "compute"  # "compute" | "collective"
    group: str | None = None  # parent row in the per-layer table, or None
    order: int = 0  # dataflow position; `build_ops` assigns it
    comment: str = ""  # why this row looks like it does. See `Op.comment`
    #  Reserved for what the CONFIG chose -- the fused-vs-unfused epilogue, the
    #  dispatch schedule, the sparse-attention strategy -- and what the alternative
    #  would have cost. A number whose value depends on a knob is unreadable without
    #  it, and re-running to find out is how a reader stops checking.

    @property
    def hbm_bytes(self) -> float:
        return self.w_bytes + self.act_bytes + self.kv_bytes

    @property
    def act_occupancy(self) -> float:
        """Activation bytes for the OCCUPANCY column -- inventory, not traffic."""
        return self.act_bytes if self.act_inventory_bytes is None \
            else self.act_inventory_bytes

    @property
    def ici_ms(self) -> float:
        return self.ici_lat_ms + self.ici_bw_ms

    def dtypes(self) -> tuple[str, ...]:
        return (self.dtype, ) if self.dtype_b is None else (self.dtype,
                                                            self.dtype_b)

    def comp_ms(self, hw: Hardware) -> float:
        rate = (hw.vpu_peak(self.dtype) if self.unit == "vpu" else hw.peak(
            *self.dtypes()))
        return self.flops / rate * 1e3

    def mem_ms(self, hw: Hardware) -> float:
        return self.hbm_bytes / hw.hbm_bw * 1e3

    def roofline_ms(self, hw: Hardware) -> float:
        return max(self.mem_ms(hw), self.comp_ms(hw)) + self.ici_ms

    def bound(self, hw: Hardware) -> str:
        if self.ici_ms > max(self.mem_ms(hw), self.comp_ms(hw)):
            return "ICI"
        if self.mem_ms(hw) >= self.comp_ms(hw):
            return "HBM"
        return "VPU" if self.unit == "vpu" else "MXU"


@dataclass
class Ctx:
    """Everything an op function needs, derived once per pass."""
    cfg: Config
    ph: Phase

    def __post_init__(self) -> None:
        m, s, ph = self.cfg.model, self.cfg.sharding, self.ph
        self.m, self.n, self.hw, self.s = m, self.cfg.numerics, self.cfg.hardware, s
        # PCP ranks replicate TP's shards and split the step's TOKENS (parallelism/
        # pcp s3): heads divide by TP alone, and every per-token op sees T / PCP.
        self.tp = s.tp
        self.pcp = float(s.pcp)
        self.ring = s.pcp > 1 and m.pcp_kv_strategy == "ring"
        self.T = float(ph.tokens) / self.pcp  # tokens this core, this pass
        self.B = float(max(1, ph.batch))  # sequences in the engine's step
        # sequences with a query on THIS rank: all of them when each spans every
        # rank (prefill), ~B/PCP when each has one token (decode)
        self.Bq = min(self.B, max(1.0, self.T)) if s.pcp > 1 else self.B
        self.S = float(ph.kv_len)  # context each query sees
        self.D = float(m.hidden)
        self.H = max(1.0, m.heads / self.tp)  # local query heads
        self.Hkv = max(1.0, m.kv_heads / min(self.tp, max(1, m.kv_heads))) \
            if not m.use_mla else 0.0
        self.dc = float(m.kv_lora_rank)
        self.qr = float(m.q_lora_rank)
        self.nope = float(m.qk_nope_head_dim)
        self.rope = float(m.qk_rope_head_dim or m.head_dim)
        self.Dv = float(m.v_head_dim or m.head_dim)
        self.dh = float(m.head_dim)
        # compute scans `scan` keys per query; HBM reads `fetch` DISTINCT keys.
        # Under "masked_dense" every query scans the whole context and the mask throws
        # the rest away, so sparsity buys nothing on either axis.
        self.scan = (float(m.indexer_topk) if m.indexer_topk
                     and m.sparse_attention_strategy == "gathered" else self.S)
        # distinct KV tokens the engine's step reads -- the ring streams all of them
        self.fetch_all = kv_fetched(m, ph)
        self.fetch = (self.fetch_all if s.pcp == 1 or self.ring else
                      _kv_fetched(m, float(ph.kv_len), self.T, self.Bq))
        # DCP re-spends `dcp` of TP's ways on context positions (parallelism/dcp s4):
        # the attention core runs the whole DCP group's heads over THIS rank's shard,
        # so heads go up by dcp and keys scanned, keys fetched and keys resident go
        # down by it. FLOPs are unchanged. `H` stays TP-local: the projections, RoPE
        # and o_proj never see DCP. Sparse: ~K_idx/dcp owned winners, balanced.
        self.dcp = float(s.dcp)
        self.Ha = self.H * self.dcp  # heads in the attention core
        self.scan = self.scan / self.dcp  # keys per query, this rank
        self.fetch = self.fetch / self.dcp  # distinct KV read, this rank's shard
        # PCP ring: each rank holds and reads 1/PCP of every sequence's pages; the
        # rest arrives over ICI (`coll_pcp`). Replicated: every rank holds it all.
        kv_div = self.dcp * (self.pcp if self.ring else 1.0)
        self.fetch = self.fetch / (self.pcp if self.ring else 1.0)
        self.Sl = self.S / kv_div  # context resident on this rank
        self.Sk = self.S / self.dcp  # keys the indexer scores per query
        self.Bk = self.Bq if s.pcp > 1 and not self.ring else self.B  # seqs whose
        #                                            indexer keys this rank reads
        self.Hi = float(m.indexer_heads)
        self.Di = float(m.indexer_head_dim)


def gemm(c: Ctx,
         name: str,
         cls: str,
         tokens: float,
         contract: float,
         out: float,
         *,
         w: float,
         i: float,
         o: float,
         w_site: str = "weights",
         act_site: str = "act_in",
         n_mat: int = 1,
         out_site: str = "act_out",
         cache_site: str | None = None,
         group: str | None = None,
         comment: str = "",
         n_mat_out: int | None = None) -> Op:
    """FLOPs = 2*tokens*pad(contract)*out*n_mat;  HBM = weights + act in + act out.

    `w_site`/`act_site` are the two halves of a `wNaM` numerics cell, and both are
    needed: `peak()` takes the WIDEST operand, so a GEMM whose weights are fp8 and
    whose activations are bf16 runs at the bf16 rate. Keying only the weight site
    makes such a GEMM 2x optimistic.

    `out_site` names the numerics site giving the OUTPUT dtype (default `act_out`).
    `cache_site` names the cache this output feeds, for the projections that build
    one -- the MLA KV down-projection, the indexer's K projection.

    **Whether those bytes land in the KV column is decided, not declared:** they do
    exactly when the write dtype equals the cache dtype. Equal means the GEMM emitted
    cache entries directly (a fused quantize epilogue). Wider means it emitted a
    staging buffer that something must still narrow, so it is an activation and the
    quantize pass is a real op -- one this file does not model and says so in the
    row's comment rather than pretending either that the staging write is free or
    that it was already the cache.

    This is also what makes prefill and decode asymmetric: prefill GENERATES `tokens`
    of cache, decode reads a cache already there and appends one entry per sequence.
    """
    k = max(contract, c.hw.mxu_min_dim)
    # `n_mat_out` decouples the OUTPUT width from the number of weight matrices. A
    # gated FFN has n_mat=2 (gate and up) but writes one tensor when the activation
    # is a fused epilogue -- see `Model.dense_fused_activation` / `moe_fused_activation`.
    ob = tokens * o * (n_mat
                       if n_mat_out is None else n_mat_out) * c.n.b(out_site)
    ob_full = tokens * o * n_mat * c.n.b(out_site)  # inventory: never halved
    to_cache = cache_site is not None and c.n.dt(out_site) == c.n.dt(
        cache_site)
    if cache_site is not None and not to_cache:
        comment = (
            f"writes {c.n.dt(out_site)} STAGING, not the {c.n.dt(cache_site)} "
            f"cache: charged to activations, not KV. A quantize pass must "
            f"follow and is NOT modelled")
    return Op(name,
              cls,
              2 * tokens * k * out * n_mat,
              c.n.dt(w_site),
              w_bytes=w * n_mat * c.n.b(w_site),
              act_bytes=tokens * i * c.n.b(act_site) +
              (0.0 if to_cache else ob),
              act_inventory_bytes=tokens * i * c.n.b(act_site) +
              (0.0 if to_cache else ob_full),
              kv_bytes=ob if to_cache else 0.0,
              dtype_b=c.n.dt(act_site),
              group=group,
              comment=comment)


def vpu(c: Ctx,
        name: str,
        cls: str,
        flops: float,
        elems: float,
        site: str = "act_out",
        group: str | None = None,
        comment: str = "") -> Op:
    return Op(name,
              cls,
              flops,
              c.n.dt(site),
              act_bytes=elems * c.n.b("act_out"),
              unit="vpu",
              group=group,
              comment=comment)


# ---------------------------------------------- MLA   model/common/attention/mla
# Flow B (weight absorption). See README s10 for the unresolved Flow A question.


def mla_q_down(c):  # "Query Latent Down-Projection (c_q)"
    return gemm(c,
                "MLA: Query Down Proj",
                "all",
                c.T,
                c.D,
                c.qr,
                w=c.D * c.qr,
                i=c.D,
                o=c.qr)


def mla_q_up(c):  # "Query Up-Projection for both RoPE and NoPE (Q_c)"
    o = c.H * (c.nope + c.rope)
    return gemm(c,
                "MLA: Query Up Proj",
                "all",
                c.T,
                c.qr,
                o,
                w=c.qr * o,
                i=c.qr,
                o=o)


def mla_kv_down(c):  # "KV Down-Projection (c_kv and K_r combined)"
    # its output IS the KV cache entry -- written at the cache dtype, to the KV column
    o = c.dc + c.rope
    return gemm(c,
                "MLA: KV Down Proj",
                "all",
                c.T,
                c.D,
                o,
                w=c.D * o,
                i=c.D,
                o=o,
                out_site="kv_write",
                cache_site="kv_cache",
                comment="output IS the KV cache entry, charged to KV")


def mla_q_to_k(c):  # "Q Proj to K (Equivalence for Q)"
    return gemm(c,
                "MLA: Q Proj to K (absorb)",
                "all",
                c.T,
                c.nope,
                c.H * c.dc,
                w=c.H * c.nope * c.dc,
                i=c.H * c.nope,
                o=c.H * c.dc)


def _sparse_note(c) -> str:
    if c.m.sparse_attention_strategy == "gathered":
        return (f"sparse gathered: scans {c.scan:,.0f} keys, fetches "
                f"{c.fetch:,.0f}. Masked dense would scan/fetch {c.Sl:,.0f}" +
                (f" (per rank, dcp{c.s.dcp})" if c.s.dcp > 1 else ""))
    return (
        f"masked dense: scans and fetches all {c.Sl:,.0f}; gathered would use topk"
        + (f" (per rank, dcp{c.s.dcp})" if c.s.dcp > 1 else ""))


#: The three MLA attention sub-ops and the softmax are one fused flash kernel on the
#: device. They are priced separately because their shapes differ, and reported under
#: one parent row because that is the thing you would go and optimise.
MLA_FLASH = "MLA: Flash Attention QK+AV"


def mla_qk_nope(c):  # "QK Phase (nope) with flash attention" -- write 0
    return Op("MLA: QK Nope",
              "all",
              2 * c.T * c.Ha * max(c.dc, c.hw.mxu_min_dim) * c.scan,
              c.n.dt("attn_qk"),
              act_bytes=c.T * c.Ha * c.dc * c.n.b("attn_qk"),
              kv_bytes=c.fetch * c.dc * c.n.b("kv_cache"),
              kv_resident_bytes=c.B * c.Sl * c.dc * c.n.b("kv_cache"),
              group=MLA_FLASH,
              comment=_sparse_note(c))


def mla_qk_rope(c):  # "QK Phase (rope) with flash attention"
    return Op("MLA: QK Rope",
              "all",
              2 * c.T * c.Ha * max(c.rope, c.hw.mxu_min_dim) * c.scan,
              c.n.dt("attn_qk"),
              act_bytes=c.T * c.Ha * c.rope * c.n.b("attn_qk"),
              kv_bytes=c.fetch * c.rope * c.n.b("kv_cache"),
              kv_resident_bytes=c.B * c.Sl * c.rope * c.n.b("kv_cache"),
              group=MLA_FLASH)


def mla_av(c):  # "AV Phase" -- O_latent = Score @ c_kv
    """Writes O_latent. Charges NO KV read: `mla_qk_nope` already read `c_kv`.

    This is what weight absorption *is*. The latent `c_kv` serves as both K and V --
    QK Nope computes `Q'_c @ c_kv^T`, AV computes `Score @ c_kv` -- so one tensor is
    used twice and fetched once. Charging it in both ops read 136 MiB/layer from a
    72 MiB cache at 128K: 1.89x more KV than the layer physically holds.

    Not true of GQA (`gqa_av`), where K and V are genuinely separate tensors and both
    reads are real. The asymmetry is the point; do not "fix" that one to match.
    """
    return Op(
        "MLA: AV",
        "all",
        2 * c.T * c.Ha * max(c.scan, c.hw.mxu_min_dim) * c.dc,
        c.n.dt("attn_av"),
        act_bytes=c.T * c.Ha * c.dc * c.n.b("attn_av"),
        group=MLA_FLASH,
        comment="no c_kv read: QK Nope already fetched it (weight absorption)")


def mla_v_up(c):  # "V Up Proj (Deferred V Projection)"
    return gemm(c,
                "MLA: V Up Proj (absorb)",
                "all",
                c.T,
                c.dc,
                c.H * c.Dv,
                w=c.H * c.dc * c.Dv,
                i=c.H * c.dc,
                o=c.H * c.Dv)


def mla_o_proj(c):  # "Output Projection"
    return gemm(c,
                "MLA: O Proj",
                "all",
                c.T,
                c.H * c.Dv,
                c.D,
                w=c.H * c.Dv * c.D,
                i=c.H * c.Dv,
                o=c.D)


#: Split three ways so `build_ops` can interleave the indexer between the projections
#: and the attention core, which is the order the layer actually runs in.
MLA_PROJ = [mla_q_down, mla_q_up, mla_q_to_k, mla_kv_down]
MLA_ATTN = [mla_qk_nope, mla_qk_rope, mla_av]
MLA_OUT = [mla_v_up, mla_o_proj]

# ------------------------------------- GQA/MHA  model/common/attention/{qkv,out}_proj

GQA_FLASH = "Attn: Flash Attention QK+AV"


def gqa_qkv(c):
    o = (c.H + 2 * c.Hkv) * c.dh
    return gemm(c, "Attn: QKV Proj", "all", c.T, c.D, o, w=c.D * o, i=c.D, o=o)


def gqa_qk(c):
    return Op("Attn: QK",
              "all",
              2 * c.T * c.Ha * max(c.dh, c.hw.mxu_min_dim) * c.scan,
              c.n.dt("attn_qk"),
              act_bytes=c.T * c.Ha * c.dh * c.n.b("attn_qk"),
              kv_bytes=c.fetch * c.Hkv * c.dh * c.n.b("kv_cache"),
              kv_resident_bytes=c.B * c.Sl * c.Hkv * c.dh * c.n.b("kv_cache"),
              group=GQA_FLASH)


def gqa_av(c):
    return Op("Attn: AV",
              "all",
              2 * c.T * c.Ha * max(c.scan, c.hw.mxu_min_dim) * c.dh,
              c.n.dt("attn_av"),
              act_bytes=c.T * c.Ha * c.dh * c.n.b("attn_av"),
              kv_bytes=c.fetch * c.Hkv * c.dh * c.n.b("kv_cache"),
              kv_resident_bytes=c.B * c.Sl * c.Hkv * c.dh * c.n.b("kv_cache"),
              group=GQA_FLASH)


def gqa_o_proj(c):
    return gemm(c,
                "Attn: O Proj",
                "all",
                c.T,
                c.H * c.dh,
                c.D,
                w=c.H * c.dh * c.D,
                i=c.H * c.dh,
                o=c.D)


GQA_PROJ = [gqa_qkv]
GQA_ATTN = [gqa_qk, gqa_av]
GQA_OUT = [gqa_o_proj]

# ---------------------------------------------- DSA indexer  model/common/attention/dsa


def dsa_k_proj(c):  # writes the indexer K cache, not a transient activation
    return gemm(c,
                "DSA Indexer: K Proj",
                "indexer",
                c.T,
                c.D,
                c.Di,
                w=c.D * c.Di,
                i=c.D,
                o=c.Di,
                out_site="indexer_write",
                cache_site="indexer_cache",
                comment="output IS the indexer K cache entry, charged to KV")


def dsa_head_weights(c):  # w16a16 on the reference sheet, unlike K/Q at w8a8
    return gemm(
        c,
        "DSA Indexer: Head Weights",
        "indexer",
        c.T,
        c.D,
        c.Hi,
        w=c.D * c.Hi,
        i=c.D,
        o=c.Hi,
        w_site="indexer_w_proj",
        act_site="indexer_w_proj_act",
        comment="w16a16 per the reference sheet; the K and Q projections "
        "are w8a8")


def dsa_q_proj(c):
    src = c.qr or c.D
    return gemm(c,
                "DSA Indexer: Q Proj",
                "indexer",
                c.T,
                src,
                c.Hi * c.Di,
                w=src * c.Hi * c.Di,
                i=src,
                o=c.Hi * c.Di)


def _score_note(c, verb: str) -> str:
    mib = c.T * c.Sk * c.n.b("indexer_av") / MiB
    if c.m.indexer_fused_scores:
        return (
            f"scores fused in SparseCore VMEM ({c.hw.sparsecore_vmem_bytes / MiB:,.0f} MiB);"
            f" spilled "
            f"they would be {mib:,.2f} MiB")
    return (
        f"scores spill to HBM, {verb} {mib:,.2f} MiB "
        f"(SparseCore VMEM is {c.hw.sparsecore_vmem_bytes / MiB:,.0f} MiB);"
        f" fused this is 0")


DSA_FLASH = "DSA Indexer: Flash Attention QK+AV"


def dsa_qk(c):  # "QK Logits": pairwise similarity using ReLU
    """Scans EVERY cached key -- this is the op that computes the sparsity, so it
    cannot use it.

    The cache read is `B * S`, not `S`. At prefill B=1 and the T queries are one
    sequence sharing one cache, so the two agree. At decode the T queries are T
    DIFFERENT sequences, each with its own S-token cache, and dropping B under-read
    it by the batch -- 16 MiB against 16 GiB at B=1024, S=128K. `mla_qk_nope` never
    had the bug because `kv_fetched` carries the batch for it.
    """
    return Op("DSA Indexer: QK",
              "indexer",
              2 * c.T * c.Hi * max(c.Di, c.hw.mxu_min_dim) * c.Sk,
              c.n.dt("indexer_qk"),
              act_bytes=c.T * c.Hi * c.Di * c.n.b("act_in"),
              kv_bytes=c.Bk * c.Sl * c.Di * c.n.b("indexer_cache"),
              kv_resident_bytes=c.B * c.Sl * c.Di * c.n.b("indexer_cache"),
              group=DSA_FLASH)


def dsa_weighted_sum(c):  # "Score Aggregation": weighted sum over heads
    """Collapses the per-head logits to one [T, S] score per query, and WRITES it.

    Three separate things were wrong here and they pulled in different directions:

    * It re-read the indexer K cache (`S * Di`). `dsa_qk` already paid for that; this
      counted the whole cache twice -- 16 MiB/layer at 128K.
    * It never charged its OUTPUT. The [T, S] score matrix is 256 MiB at T=1024,
      S=128K -- far too big for VMEM, so it lands in HBM. `dsa_topk` was already
      reading a score matrix that, in this model, nothing had written.
    * It read the [T, Hi] head weights twice.

    The [T, Hi, S] per-head logits stay inside the fused kernel and are not charged.
    Whether the collapsed [T, S] score matrix does too is `Model.indexer_fused_scores`
    -- the spec says always, VMEM says only when it fits. At T=1024/S=128K it is 256
    MiB against 8 MiB of VMEM, so the default spills it.
    """
    # It is a matmul contracting over H_idx, so it pays MXU padding like any other:
    # H_idx is 32 on GLM-5.2 against a 256 minimum, so SEVEN EIGHTHS of this op is
    # zeros. The spec's `2*B*S^2*H_idx` is the useful work, not the charged work --
    # this was the one raw `Op` in the file that forgot to pad, and it under-counted
    # the op 8x.
    return Op(
        "DSA Indexer: Weighted Sum",
        "indexer",
        2 * c.T * c.Sk * max(c.Hi, float(c.hw.mxu_min_dim)),
        c.n.dt("indexer_av"),
        act_bytes=c.T * c.Hi * c.n.b("act_out")  # head weights, read
        + (
            0.0 if c.m.indexer_fused_scores  # [T, S] scores, written
            else c.T * c.Sk * c.n.b("indexer_av")),
        group=DSA_FLASH,
        comment=_score_note(c, "writes"))


def dsa_topk(c):  # "Top-K Selection" -- a sort; the FLOP count is a stand-in
    """Writes K_idx indices per query; reads the scores back unless they were fused.

    The two terms were previously one `T * (S + topk)` lump, which silently assumed
    the spill while `dsa_weighted_sum` assumed the fusion -- so the scores were read
    from HBM without ever being written there.

    Built raw rather than through `vpu()`, which charges every element at `act_out`.
    The two terms have different widths and the helper cannot say so: the scores it
    reads are activations (bf16), but what it WRITES is indices -- integers, and
    `int32` at that, because picking K_idx of S needs ceil(log2(S)) bits and S is
    131072 here. Charging them at bf16 under-counted the write 2x.
    """
    scores = 0.0 if c.m.indexer_fused_scores else c.T * c.Sk * c.n.b(
        "indexer_av")
    # Under DCP every rank then merges the DCP*K_idx gathered candidates to the exact
    # global top-K (streamindex_topk stage 3) -- identical work on every rank.
    merge = c.T * c.s.dcp * float(c.m.indexer_topk) if c.s.dcp > 1 else 0.0
    return Op("DSA Indexer: TopK",
              "indexer",
              c.T * c.Sk + merge,
              c.n.dt("indexer_av"),
              act_bytes=scores +
              c.T * float(c.m.indexer_topk) * c.n.b("indexer_indices"),
              unit="vpu",
              comment=_score_note(c, "reads back") +
              f"; indices written at {c.n.dt('indexer_indices')}")


DSA_PROJ = [dsa_k_proj, dsa_head_weights, dsa_q_proj]
DSA_ATTN = [dsa_qk, dsa_weighted_sum, dsa_topk]

# ---------------------------------------------- dense FFN  model/common/ffn


def _gated(c, tokens: float, F: float, fused: bool) -> tuple[int, str]:
    """(output multiplicity, comment) for a gated FFN's up/gate GEMM.

    Fused: gate and up stay in VMEM, `SiLU(gate)*up` is the epilogue, one [.., F]
    tensor is written and no SwiGLU op exists. Unfused: both halves go to HBM and a
    separate pass reads them back. The comment carries what the other choice costs,
    because the row is unreadable without knowing which one produced it.
    """
    one = tokens * F * c.n.b("act_out") / MiB
    if fused:
        return 1, (
            f"fused SwiGLU epilogue: writes {one:,.2f} MiB. Unfused writes "
            f"{2 * one:,.2f} MiB + a separate SwiGLU op")
    return 2, (
        f"unfused: writes both halves, {2 * one:,.2f} MiB, and a separate "
        f"SwiGLU op follows. Fused would write {one:,.2f} MiB")


def dense_ffi(c):
    F = float(c.m.dense_ffn) / c.tp
    n_out, note = _gated(c, c.T, F, c.m.dense_fused_activation)
    return gemm(c,
                "Dense: FFi + FFg",
                "dense",
                c.T,
                c.D,
                F,
                w=c.D * F,
                i=c.D,
                o=F,
                n_mat=2,
                n_mat_out=n_out,
                comment=note)


def dense_ffo(c):
    F = float(c.m.dense_ffn) / c.tp
    return gemm(c, "Dense: FFo", "dense", c.T, F, c.D, w=F * c.D, i=F, o=c.D)


def dense_swiglu(c):
    if c.m.dense_fused_activation:
        return None  # folded into `dense_ffi`'s epilogue
    F = float(c.m.dense_ffn) / c.tp
    return vpu(c,
               "Dense: SwiGLU",
               "dense",
               6 * c.T * F,
               3 * c.T * F,
               comment="separate pass; fused into FFi+FFg when "
               "dense_fused_activation / moe_fused_activation")


DENSE_OPS = [dense_ffi, dense_ffo, dense_swiglu]

# ---------------------------------------------- MoE  model/common/moe


def _moe_dims(c):
    ep = c.s.ep_group() or 1
    local_e = c.m.n_experts / ep if c.s.ep else c.m.n_experts
    Fe = (float(c.m.expert_ffn) if c.s.ep else float(c.m.expert_ffn) /
          (c.tp * c.pcp))  # TP flattened across PCP
    return local_e, Fe, float(c.m.topk)


def moe_router(c):  # s3.1 Router (Gating)
    E = float(c.m.n_experts)
    return gemm(c,
                "MoE: Router (gating)",
                "moe",
                c.T,
                c.D,
                E,
                w=c.D * E,
                i=c.D,
                o=E)


def moe_ffi(c):  # s3.2 RoutedMoE W1 (Gate/Up)
    le, Fe, K = _moe_dims(c)
    n_out, note = _gated(c, c.T * K, Fe, c.m.moe_fused_activation)
    return gemm(c,
                "MoE: FFi + FFg",
                "moe",
                c.T * K,
                c.D,
                Fe,
                w=le * c.D * Fe,
                i=c.D,
                o=Fe,
                w_site="moe_weights",
                n_mat=2,
                n_mat_out=n_out,
                comment=note)


def moe_ffo(c):  # s3.3 RoutedMoE W2 (Down)
    le, Fe, K = _moe_dims(c)
    return gemm(c,
                "MoE: FFo",
                "moe",
                c.T * K,
                Fe,
                c.D,
                w=le * Fe * c.D,
                i=Fe,
                o=c.D,
                w_site="moe_weights")


def moe_swiglu(c):
    if c.m.moe_fused_activation:
        return None  # folded into `moe_ffi`'s epilogue
    _, Fe, K = _moe_dims(c)
    return vpu(c,
               "MoE: SwiGLU",
               "moe",
               6 * c.T * K * Fe,
               3 * c.T * K * Fe,
               comment="separate pass; fused into FFi+FFg when "
               "dense_fused_activation / moe_fused_activation")


#: The routing data movement, as THREE ops, because that is what the device runs.
#: `j-5444b22d-79a3-440c-bd5b` shows `gather_fusion` -> [expert GEMMs] -> `gather` ->
#: `convert_reduce_fusion`, 75x per step, all HBM-bound, 20.84 ms/step at prefill --
#: 59% of what the expert GEMMs themselves cost. Splitting them is worth only
#: +0.56 ms of modelled prefill over one lumped `4 * T*K*D` row; the reason to do it
#: is that the unpermute's 2 full passes are the only thing anyone can optimise here,
#: and a lumped row gives `decompose-optimization` nothing to point at.
#:
#: In units of one full [T*K, D] pass, U:
#:     permute    read the [T, D] table (0.125U), write [T*K, D]  -> 1.125U
#:     unpermute  inverse-permutation gather, full in AND out     -> 2.000U
#:     combine    weighted reduce K->1, write [T, D]              -> 1.125U
#: 4.25U in total. The earlier single op charged 4U, which was close by luck: it
#: assumed the unpermute wrote [T*K, D] and no combine existed. The combine's write
#: really is the cheap [T, D] -- but it is a SEPARATE fusion fed by a separate full
#: gather, so the cheap write is paid on top of an extra pass, not instead of one.
def moe_permute(c):  # s3.6, first leg
    _, _, K = _moe_dims(c)
    return vpu(
        c,
        "MoE: permute",
        "moe",
        0.0,
        c.T * c.D + c.T * K * c.D,
        comment="gather into expert-major order for the grouped GEMM: reads "
        "the [T, D] table, writes [T*K, D]. Local HBM, not a "
        "collective -- it exists with or without EP")


def moe_unpermute(c):  # s3.6, second leg
    _, _, K = _moe_dims(c)
    return vpu(
        c,
        "MoE: unpermute",
        "moe",
        0.0,
        2 * c.T * K * c.D,
        comment="inverse-permutation gather, [T*K, D] in AND out -- two full "
        "passes. Fusing it into the combine as one gather-reduce "
        "would remove both")


def moe_combine(c):  # s3.6, third leg
    _, _, K = _moe_dims(c)
    return vpu(
        c,
        "MoE: combine",
        "moe",
        2 * c.T * K * c.D,
        c.T * K * c.D + c.T * c.D,
        comment="weighted reduce of the K expert copies back to one row per "
        "token: reads [T*K, D], writes [T, D]")


def moe_router_softmax(c):
    E = float(c.m.n_experts)
    return vpu(c, "MoE: router softmax + topk", "moe", 3 * c.T * E,
               2 * c.T * E)


def moe_shared_ffi(c):  # s3.4 SharedExpert W1
    if not c.m.n_shared_experts:
        return None
    Fs = float(c.m.shared_ffn) * c.m.n_shared_experts / c.tp
    n_out, note = _gated(c, c.T, Fs, c.m.moe_fused_activation)
    return gemm(c,
                "MoE: Shared FFi + FFg",
                "moe",
                c.T,
                c.D,
                Fs,
                w=c.D * Fs,
                i=c.D,
                o=Fs,
                n_mat=2,
                n_mat_out=n_out,
                comment=note)


def moe_shared_ffo(c):  # s3.5 SharedExpert W2
    if not c.m.n_shared_experts:
        return None
    Fs = float(c.m.shared_ffn) * c.m.n_shared_experts / c.tp
    return gemm(c,
                "MoE: Shared FFo",
                "moe",
                c.T,
                Fs,
                c.D,
                w=Fs * c.D,
                i=Fs,
                o=c.D)


def moe_shared_swiglu(c):
    if not c.m.n_shared_experts or c.m.moe_fused_activation:
        return None  # folded into `moe_shared_ffi`'s epilogue
    Fs = float(c.m.shared_ffn) * c.m.n_shared_experts / c.tp
    return vpu(c,
               "MoE: Shared SwiGLU",
               "moe",
               6 * c.T * Fs,
               3 * c.T * Fs,
               comment="separate pass; fused into Shared FFi+FFg when "
               "dense_fused_activation / moe_fused_activation")


# ---------------------------------------------- per-layer elementwise (every layer)


def norm_rmsnorm(c):
    return vpu(c, "Norm: RMSNorm x2", "all", 2 * c.T * c.D * 4,
               2 * 2 * c.T * c.D)


def norm_residual(c):
    return vpu(c, "Norm: residual add x2", "all", 2 * c.T * c.D,
               2 * 3 * c.T * c.D)


def attn_rope(c):
    return vpu(c, "Attn: RoPE", "all", (c.H + 1) * c.T * c.rope * 6,
               2 * c.T * c.H * c.rope)


def attn_softmax(c):  # fused -> no HBM, but the VPU work is real
    return Op("Attn: softmax",
              "all",
              3 * c.T * c.Ha * c.scan,
              c.n.dt("attn_av"),
              unit="vpu",
              group=MLA_FLASH if c.m.use_mla else GQA_FLASH)


def kv_fetched(m: Model, ph: Phase) -> float:
    """Distinct KV tokens read from HBM this pass, per core. See Phase.kv_fetched.

    Per SEQUENCE, the T_seq queries in this pass each select `K_idx` keys, so their
    union is `min(S, T_seq * K_idx)` -- capped at the context because they share one
    cache. Then times the batch, because sequences do NOT share: each has its own
    cache and its own union.

    The cap bites at `T_seq = S / K_idx` (64 tokens at S=128K, K_idx=2048): above it
    a pass reads the whole context and sparsity buys no bandwidth, below it the read
    scales with the chunk. The previous form keyed off `tokens > batch` as a proxy
    for "is this prefill", which got the right answer either side of that crossover
    by accident and over-charged a 16-token chunk 4x.
    """
    if ph.kv_fetched:
        return float(ph.kv_fetched)
    return _kv_fetched(m, float(ph.kv_len), float(ph.tokens),
                       float(max(1, ph.batch)))


def _kv_fetched(m: Model, S: float, tokens: float, B: float) -> float:
    """`kv_fetched`'s arithmetic for `tokens` queries over `B` sequences of `S`."""
    if not m.indexer_topk or m.sparse_attention_strategy == "masked_dense":
        return S * B
    return B * min(S, tokens / B * m.indexer_topk)


def build_ops(cfg: Config, ph: Phase) -> list[Op]:
    """Assemble the op list this model's config calls for, IN DATAFLOW ORDER.

    One function per spec op, and the order is the order the layer runs in -- norms,
    attention projections, RoPE, the indexer, the attention core, the output
    projection and its all-reduce, the FFN with its dispatch/collect around it, the
    residual. `Op.order` records the position.

    The order is load-bearing only for the per-layer table, which a reader follows down
    the layer: sorting those rows by cost, as the flat phase table does, hides that the
    indexer runs *before* the attention it selects for. Every consumer that sums or
    ranks is order-independent, so this costs them nothing.

    Collectives are interleaved here rather than appended, for the same reason: `MoE:
    Dispatch` belongs above `MoE: FFi + FFg`, not in a block at the bottom.
    """
    c, m, s = Ctx(cfg, ph), cfg.model, cfg.sharding
    mach = cfg.hardware.machine()
    tp, ep = c.tp > 1, s.ep and m.moe_layers
    out: list[Op] = []

    def add(fns, *, coll=False):
        for f in fns:
            o = f(c, mach) if coll else f(c)
            if o is not None:
                o.order = len(out)
                out.append(o)

    add([norm_rmsnorm])
    add(MLA_PROJ if m.use_mla else GQA_PROJ)
    add([attn_rope])
    if m.indexer_layers:
        add(DSA_PROJ)
        add(DSA_ATTN)
        if s.dcp > 1:
            add([coll_dcp_indexer], coll=True)
        if s.pcp > 1:
            add([coll_pcp_indexer], coll=True)
    if s.dcp > 1:
        add([coll_dcp_q_gather], coll=True)
    add(MLA_ATTN if m.use_mla else GQA_ATTN)
    add([attn_softmax])
    if s.pcp > 1:
        add([coll_pcp], coll=True)
    if s.dcp > 1:
        add([coll_dcp_merge], coll=True)
    add(MLA_OUT if m.use_mla else GQA_OUT)
    if tp:
        add([coll_tp_attn], coll=True)
    if m.dense_layers:
        add(DENSE_OPS)
    if m.moe_layers:
        add([moe_router, moe_router_softmax])
        if ep:
            add([coll_ep_router, coll_ep_dispatch], coll=True)
        add([
            moe_permute, moe_ffi, moe_swiglu, moe_ffo, moe_unpermute,
            moe_combine
        ])
        if ep:
            add([coll_ep_collect], coll=True)
        add([moe_shared_ffi, moe_shared_swiglu, moe_shared_ffo])
    if tp:
        add([coll_tp_ffn], coll=True)
    add([norm_residual])
    return out


def compute_ops(cfg: Config, ph: Phase) -> list[Op]:
    """The GEMM and VPU ops this model's config calls for -- no collectives."""
    return [o for o in build_ops(cfg, ph) if o.kind != "collective"]


def collectives(cfg: Config, ph: Phase) -> list[Op]:
    """Only the collectives. `parallelism/<scheme>/<scheme>.md` says which ones fire."""
    return [o for o in build_ops(cfg, ph) if o.kind == "collective"]


# ------------------------------------- collectives  parallelism/<scheme>/<scheme>.md
# Volume comes from the scheme's spec; the price comes from collective.py.


def coll(name: str,
         cls: str,
         *ests: cc.Estimate,
         scale: float = 1.0,
         note: str = "") -> Op:
    """A collective op, keeping `collective.py`'s alpha/beta split instead of `.total`.

    `t_fixed` is folded into latency. It is the per-call barrier cost left over after
    hop latency (`Interconnect.launch_overhead`), so like latency and unlike bandwidth
    it does not shrink when the payload does -- which is the whole question the two
    columns exist to answer. `lat + bw` is `sum(e.total)` exactly, so nothing about the
    priced cost changes by splitting it.
    """
    return Op(name,
              cls,
              0.0,
              "bf16",
              kind="collective",
              comment=note,
              ici_lat_ms=scale * sum(e.t_latency + e.t_fixed
                                     for e in ests) * 1e3,
              ici_bw_ms=scale * sum(e.t_bandwidth for e in ests) * 1e3)


def coll_tp_attn(c, mach):  # tp s2: attention exit, every layer
    return coll(
        "TP: All-Reduce (attn exit)", "all",
        cc.all_reduce(mach, elems=c.T * c.D, dtype=c.n.dt("tp_allreduce")))


def coll_tp_ffn(c, mach):  # tp s2: FFN exit -- dense layers only when EP is on
    return coll(
        "TP: All-Reduce (FFN exit)", "dense" if c.s.ep else "all",
        cc.all_reduce(mach, elems=c.T * c.D, dtype=c.n.dt("tp_allreduce")))


def _chip_axes(mach) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(intra-chip d2d axes, inter-chip ICI axes), each dropping size-1 axes.

    The on-package d2d link is ~7x the ICI link on v7x, which is the entire reason
    a hierarchical schedule is worth naming: do the cheap hop first and the
    expensive one carries fewer, larger messages.
    """
    d2d = tuple(i for i, a in enumerate(mach.axes)
                if a.name == "d2d" and a.size > 1)
    ici = tuple(i for i, a in enumerate(mach.axes)
                if a.name != "d2d" and a.size > 1)
    return d2d, ici


def _axes_size(mach, axes) -> int:
    n = 1
    for i in axes:
        n *= mach.axes[i].size
    return n


def coll_ep_dispatch(c,
                     mach):  # ep s4. parallelism/ep/ep.md s4 has the shapes.
    """Get every token to the device holding the experts it routed to.

    Three schedules, and the first differs from the other two by `topk` on the wire:

      all_to_all       ships the EXPANDED [B, kT, D] set -- each token already
                       replicated once per expert it chose. Per device kT*D/EP.
      all_gather       ships the UNEXPANDED [B, T, D] tokens once, and every device
                       sorts to expert order and slices out its own share locally.
                       8x less wire traffic than A on GLM-5.2 (topk=8), which is why
                       it is the default.
      hierarchical_ag  same payload as B, but as two named phases: gather ACROSS
                       chips over ICI first, then gather the chip's own cores over
                       d2d last.

    The phase order in C is not arbitrary and it is the opposite of the intuitive
    one. An all-gather's payload GROWS with every phase -- after gathering over a
    group of n you are carrying n times what you started with -- so the phase you
    want on the fast on-package link is the LAST one, not the first. Doing d2d first
    spends the 600 GB/s link on the smallest message and leaves the full gathered
    chunk to the 100 GB/s ICI link; measured 1.73x worse than the right order.
    Reduce-scatter is the mirror image (payload shrinks, so put d2d first), which is
    why `MoE: Collect` must not simply copy this.
    """
    strat, ep = c.m.moe_dispatch_strategy, c.s.ep_group()
    dt = c.n.dt("expert_dispatch")
    if strat == "all_to_all":
        return coll("MoE: Dispatch (all-to-all)",
                    "moe",
                    cc.all_to_all(mach,
                                  elems=c.T * float(c.m.topk) * c.D,
                                  dtype=dt),
                    note=f"strategy A: all-to-all the EXPANDED kT set, "
                    f"{c.m.topk}x the payload of the all-gather strategies")
    if strat == "all_gather":
        return coll(
            "MoE: Dispatch (all-gather)",
            "moe",
            cc.all_gather(mach, elems_full=c.T * ep * c.D, dtype=dt),
            note="strategy B: one flat all-gather of the unexpanded tokens; "
            "cost_estimator picks the axis order")
    # hierarchical: ICI across chips first, then d2d inside the chip carrying the
    # full gathered chunk on the fast link. Order matters -- see the docstring.
    d2d, ici = _chip_axes(mach)
    if not d2d or not ici:  # one level only -> it degenerates to B
        return coll("MoE: Dispatch (hierarchical AG)",
                    "moe",
                    cc.all_gather(mach, elems_full=c.T * ep * c.D, dtype=dt),
                    note="strategy C, but the slice has one link class so it "
                    "degenerates to the flat all-gather")
    n_ici = _axes_size(mach, ici)
    return coll(
        "MoE: Dispatch (hierarchical AG)",
        "moe",
        cc.all_gather(mach, elems_full=c.T * n_ici * c.D, dtype=dt, axes=ici),
        cc.all_gather(mach, elems_full=c.T * ep * c.D, dtype=dt, axes=d2d),
        note=
        "strategy C: AG1 across chips over ICI, then AG2 over d2d carrying the "
        "full chunk. Phases summed -- a real one overlaps all but the last")


def coll_ep_collect(c, mach):
    t = c.T * c.s.ep_group()
    return coll(
        "MoE: Collect (reduce-scatter)", "moe",
        cc.reduce_scatter(mach, elems=t * c.D, dtype=c.n.dt("expert_collect")))


def coll_ep_router(c, mach):  # ep s4b -- width is E, not D. Easy to omit.
    t = c.T * c.s.ep_group()
    return coll(
        "MoE: Router logits (all-gather)", "moe",
        cc.all_gather(mach,
                      elems_full=t * c.m.n_experts,
                      dtype=c.n.dt("router_logits")))


def coll_dcp_indexer(c, mach):  # dcp s7.2 -- indexer layers only
    k, d = float(c.m.indexer_topk), c.s.dcp
    if c.m.dcp_indexer_gather == "single_ag":
        return coll(
            "DCP: indexer candidate gather",
            "indexer",
            cc.all_gather(mach,
                          elems_full=c.T * (k + 1) * d,
                          dtype=c.n.dt("dcp_gather")),
            note="single_ag (vllm-torchtpu PR #1230): scores + a source-rank "
            "column in one gather; indices stay home")
    return coll("DCP: indexer candidate gather",
                "indexer",
                cc.all_gather(mach,
                              elems_full=c.T * k * d,
                              dtype=c.n.dt("dcp_gather")),
                scale=2.0,
                note="two_ag (main): indices and scores gathered separately; "
                "single_ag (#1230) would halve the bytes in one collective")


def coll_dcp_q_gather(c,
                      mach):  # dcp s6 / s7.1 step 1 -- every attention layer
    """Widen this rank's query heads to the whole DCP group's: H -> H * dcp."""
    e = c.T * c.Ha  # gathered heads x tokens
    if c.m.use_mla:  # ql_nope and q_pe, two calls
        return coll(
            "DCP: query head gather",
            "all",
            cc.all_gather(mach, elems_full=e * c.dc, dtype=c.n.dt("attn_qk")),
            cc.all_gather(mach, elems_full=e * c.rope,
                          dtype=c.n.dt("attn_qk")),
            note="" if c.m.indexer_topk else
            "dense MLA: no DCP path on TPU -- priced as the sparse merge would be"
        )
    return coll(
        "DCP: query head gather", "all",
        cc.all_gather(mach, elems_full=e * c.dh, dtype=c.n.dt("attn_qk")))


def coll_dcp_merge(c,
                   mach):  # dcp s6 / s7.1 steps 3-4 -- every attention layer
    """Fold the ranks' partial attention into the exact whole-context result.

    Not absorbed by `o_proj`: the merge ends TP-local again, and the TP reduction
    then runs as it would without DCP.
    """
    e, d = c.T * c.Ha, c.s.dcp  # partial: [T, H*dcp, width]
    if c.m.use_mla:
        # LSE all-gather, then reduce-scatter the LSE-weighted partials along heads
        return coll(
            "DCP: LSE merge",
            "all",
            cc.all_gather(mach, elems_full=e * d, dtype=c.n.dt("dcp_lse")),
            cc.reduce_scatter(mach,
                              elems=e * c.dc,
                              dtype=c.n.dt("dcp_combine")),
            note="LSE all-gather + f32 reduce-scatter of the weighted partials"
        )
    # GQA: all-gather partials AND LSEs along tokens, fold, slice own heads -- the
    # partial gather stacks dcp copies of an already dcp-wide head axis: dcp^2
    return coll("DCP: LSE merge",
                "all",
                cc.all_gather(mach,
                              elems_full=e * c.dh * d,
                              dtype=c.n.dt("attn_qk")),
                cc.all_gather(mach, elems_full=e * d, dtype=c.n.dt("dcp_lse")),
                note="partials all-gathered along tokens: T * H * dcp^2 * D_h")


def _pcp_note(c, what: str) -> str:
    return what + ("; MLA has no PCP path on TPU -- priced for evaluation"
                   if c.m.use_mla else "")


def coll_pcp(c, mach):  # pcp s4 / s6 -- every attention layer
    """The keys each rank's local queries need but does not hold.

    "ring": the striped cache streams around the PCP ring inside the attention
    kernel -- point-to-point, but a ring all-gather of every page the step reads in
    volume. It scales with the CONTEXT. "allgather": replicated cache, so only the
    step's new rows move -- it scales with the TOKENS.
    """
    w = (c.dc + c.rope) if c.m.use_mla else 2 * c.Hkv * c.dh
    if c.ring:
        return coll("PCP: KV ring",
                    "all",
                    cc.all_gather(mach,
                                  elems_full=c.fetch_all * w,
                                  dtype=c.n.dt("kv_cache")),
                    note=_pcp_note(
                        c, "in-kernel ring over the striped cache; hidden "
                        "behind attention on hardware, charged serially"))
    return coll("PCP: KV all-gather",
                "all",
                cc.all_gather(mach,
                              elems_full=c.T * c.pcp * w,
                              dtype=c.n.dt("kv_cache")),
                note=_pcp_note(
                    c, "replicated cache: the step's new rows only. "
                    "Not on main"))


def coll_pcp_indexer(c, mach):  # pcp s4 / s6 -- indexer layers; MLA only
    if c.ring:  # the indexer scores every key, so all of it streams
        e = c.B * c.S * c.Di
        name, what = "PCP: indexer K ring", "the indexer K cache streams in full"
    else:
        e = c.T * c.pcp * c.Di
        name, what = "PCP: indexer K all-gather", "replicated indexer cache: new rows"
    return coll(name,
                "indexer",
                cc.all_gather(mach,
                              elems_full=e,
                              dtype=c.n.dt("indexer_cache")),
                note=_pcp_note(c, what))


# ============================================================ roll-up


def multiplicity(m: Model, cls: str) -> int:
    c = m.counts
    return {
        "all": m.layers,
        "dense": m.dense_layers,
        "moe": m.moe_layers,
        "indexer": m.indexer_layers
    }[cls] // max(1, 1) if cls in ("all", "dense", "moe", "indexer") else 0


# ------------------------------------------------ pipeline stages  parallelism/pp/pp.md


def pp_default_partition(n_layers: int, k: int) -> list[int]:
    """Layers per stage under `get_pp_indices` (vllm/distributed/utils.py:156-161).

    `L // k` each; the remainder goes to the stages counting back from the
    SECOND-to-last -- `partitions[-i] += 1` for i = 2 .. r+1 -- so the first and last,
    which carry the embeddings, are spared. Remainder-on-the-last is the wrong answer.
    """
    if k <= 1:
        return [n_layers]
    n, r = divmod(n_layers, k)
    out = [n] * k
    order = list(range(k - 2, 0, -1)) + [k - 1, 0]
    for j in range(r):
        out[order[j % len(order)]] += 1
    return out


def stage_layers(cfg: Config) -> list[int]:
    """Layers per stage: `Sharding.pp_layers` when set, else the default."""
    s = cfg.sharding
    return list(s.pp_layers) if s.pp_layers else pp_default_partition(
        cfg.model.layers, max(1, s.pp))


def layer_costs(cfg: Config, ops: list[Op]) -> list[float]:
    """Roofline ms of each transformer layer, unsharded by PP.

    Layers are not interchangeable: the first `dense_layers` run a dense FFN and the
    rest an MoE block (~43x heavier on GLM-5.2), and only the indexer layers run the
    indexer. That asymmetry is why a partition can be unbalanced.
    """
    hw, m = cfg.hardware, cfg.model
    per = {"all": 0.0, "dense": 0.0, "moe": 0.0, "indexer": 0.0}
    for op in ops:
        if op.layer_class in per:
            per[op.layer_class] += op.roofline_ms(hw)
    idx = set(m.indexer_layer_ids())
    return [
        per["all"] + (per["dense"] if i < m.dense_layers else per["moe"]) +
        (per["indexer"] if i in idx else 0.0) for i in range(m.layers)
    ]


def stage_costs(cfg: Config,
                ops: list[Op],
                layers: list[int] | None = None) -> list[float]:
    """Per-stage ms of one pass, before the hand-off."""
    lc = layer_costs(cfg, ops)
    out, i = [], 0
    for take in (layers or stage_layers(cfg)):
        out.append(sum(lc[i:i + take]))
        i += take
    return out


def pp_handoff_ms(cfg: Config, bucket: float) -> float:
    """One pipeline hand-off: a collective-permute every stage launches every forward.

    Carries hidden+residual summed, [bucket, D] in the model dtype, plus -- on a
    sparse model -- the top-k table [bucket, K_idx] int32, which crosses EVERY
    boundary. `bucket` is the PADDED buffer: every launch runs one program sized to
    the largest token bucket, i.e. `max_num_batched_tokens` (pp_wave.py). Priced
    directly: `collective.py` has no permute primitive. parallelism/pp/pp.md s4.
    """
    if cfg.sharding.pp <= 1:
        return 0.0
    m, n, hw = cfg.model, cfg.numerics, cfg.hardware
    per_tok = m.hidden * n.b("act_out")
    if m.indexer_topk and m.indexer_layers:
        per_tok += m.indexer_topk * n.b("indexer_indices")
    return (bucket * per_tok / hw.ici_bw + hw.hop_latency) * 1e3


def stage_ms(cfg: Config,
             ph: Phase,
             bucket: float | None = None,
             layers: list[int] | None = None) -> list[float]:
    """Per-stage ms of one pass, hand-off included on every stage (pp > 1)."""
    st = stage_costs(cfg, build_ops(cfg, ph), layers)
    if cfg.sharding.pp <= 1:
        return st
    h = pp_handoff_ms(cfg, bucket if bucket else ph.tokens)
    return [x + h for x in st]


def pp_cycle_ms(stages: list[float], v: float) -> float:
    """Decode: every request advances one token per cycle. `v` batches in flight, no
    two in the slowest stage at once, none back at stage 0 before a full traversal."""
    return max(sum(stages), v * max(stages))


def pp_ttft_ms(chunk_stages: list[list[float]]) -> float:
    """One prompt's chunks pipeline across stages: one chunk per wave at the slowest
    stage, then the last drains through the rest. parallelism/pp/pp.md s6."""
    last = chunk_stages[-1]
    return sum(max(c) for c in chunk_stages) + sum(last) - max(last)


def best_pp_partition(cfg: Config,
                      ph: Phase,
                      k: int | None = None,
                      max_stage_bytes: float | None = None) -> tuple[int, ...]:
    """The contiguous partition minimising the slowest stage for pass `ph`.

    Exact min-max linear partition by dynamic programming over the real per-layer
    costs. `max_stage_bytes` (default: the usable HBM less resident overhead) bounds
    each stage's weights, embedding/LM head included; the tie-break is the lighter
    heaviest stage. Returns `Sharding.pp_layers` for `k` stages.
    """
    m, hw = cfg.model, cfg.hardware
    k = k or cfg.sharding.pp
    L = m.layers
    if k <= 1:
        return (L, )
    lc = layer_costs(cfg, build_ops(cfg, ph))
    lw = _layer_weight_bytes(cfg)
    emb_first, emb_last = _embedding_bytes(cfg, k)
    if max_stage_bytes is None:
        max_stage_bytes = (hw.hbm_capacity * hw.memory_utilization -
                           hw.resident_overhead_gb * GiB)
    pc = [0.0]
    pw = [0.0]
    for c, w in zip(lc, lw):
        pc.append(pc[-1] + c)
        pw.append(pw[-1] + w)
    INF = (float("inf"), float("inf"))
    # best[j][i]: (slowest stage, heaviest stage) for layers [0, i) in j stages
    best = [[INF] * (L + 1) for _ in range(k + 1)]
    cut = [[0] * (L + 1) for _ in range(k + 1)]
    best[0][0] = (0.0, 0.0)
    for j in range(1, k + 1):
        for i in range(j, L - (k - j) + 1):
            for a in range(j - 1, i):
                if best[j - 1][a] == INF:
                    continue
                w = pw[i] - pw[a] + (emb_first if a == 0 else 0.0) \
                    + (emb_last if i == L else 0.0)
                if w > max_stage_bytes:
                    continue
                cand = (max(best[j - 1][a][0],
                            pc[i] - pc[a]), max(best[j - 1][a][1], w))
                if cand < best[j][i]:
                    best[j][i], cut[j][i] = cand, a
    if best[k][L] == INF:
        raise ValueError(f"no {k}-stage partition of {L} layers fits "
                         f"{max_stage_bytes / GiB:.1f} GiB per stage")
    out, i = [], L
    for j in range(k, 0, -1):
        a = cut[j][i]
        out.append(i - a)
        i = a
    return tuple(reversed(out))


@dataclass
class PhaseResult:
    phase: Phase
    rows: list[dict[str, Any]]
    total_ms: float
    weights_gib: float
    kv_gib: float
    layer_rows: list[dict[str, Any]] = field(default_factory=list)


#: Columns of the per-layer op table, in order, with the titles a report uses.
#: Times and bytes are for ONE layer of that kind, on ONE core, summed over every pass
#: the phase runs (so a chunked prefill's 8 passes are already in the number).
LAYER_COLUMNS = (
    ("operator", "Operator"),
    ("memory_ms", "Memory Time (ms)"),
    ("compute_ms", "Compute Time (ms)"),
    ("ici_latency_ms", "ICI Latency Time (ms)"),
    ("ici_bw_ms", "ICI BW Time (ms)"),
    ("roofline_ms", "Roofline Time (ms)"),
    ("boundness", "Boundness"),
    ("act_mib", "Memory Occupancy - Activations (MiB)"),
    ("weight_mib", "Memory Occupancy - Weights (MiB)"),
    ("kv_mib", "Memory Occupancy - KV (MiB)"),
    ("comment", "Comment"),
)


def layer_rows(cfg: Config, ph: Phase) -> list[dict[str, Any]]:
    """One representative layer of each KIND, priced op by op.

    78 layers of GLM-5.2 are three distinct layers repeated, so this prices the three
    and says how many of each there are. That is the readable form: a reader follows
    one layer down the dataflow instead of reading a phase total that has already
    averaged the indexer across layers that do not run it.

    Every row is fully summed here on purpose. Three `role`s come back --

        op      one operator
        fused   a parent over sub-ops that are one kernel on the device (the three
                MLA attention ops and their softmax are one flash kernel). Its
                children follow it with the same `parent`, and are NOT also counted:
                a consumer showing the parent must skip rows with a `parent` set.
        total   the whole layer kind, and the row the Layer Sums table is built from

    -- so turning this into a report is filtering, ordering and naming, never
    arithmetic. `roofline.py` does the sums because agents were measured getting
    exactly this wrong; the tpu-roofline prompt does the naming because that is model vocabulary.

    `layer_total_ms` is `roofline_ms * count`, the kind's contribution to the phase.
    Summing it over the `total` rows reproduces `PhaseResult.total_ms` exactly --
    `selftest::test_the_layer_tables_reconcile_with_the_phase_total` pins that.
    """
    hw, m = cfg.hardware, cfg.model
    stage = 1.0 / max(1, cfg.sharding.pp)
    passes = chunks_of(ph)

    # per op name, summed over the passes in this phase. Shapes differ between chunks
    # (pass i attends over more context than pass i-1), so this is a sum, not a scale.
    acc: dict[str, dict[str, Any]] = {}
    for sub in passes:
        for op in build_ops(cfg, sub):
            a = acc.setdefault(
                op.name, {
                    "operator": op.name,
                    "layer_class": op.layer_class,
                    "parent": op.group or "",
                    "order": op.order,
                    "unit": op.unit,
                    "comment": op.comment,
                    "memory_ms": 0.0,
                    "compute_ms": 0.0,
                    "ici_latency_ms": 0.0,
                    "ici_bw_ms": 0.0,
                    "roofline_ms": 0.0,
                    "act_mib": 0.0,
                    "weight_mib": 0.0,
                    "kv_mib": 0.0
                })
            a["order"] = min(a["order"], op.order)
            a["memory_ms"] += op.mem_ms(hw)
            a["compute_ms"] += op.comp_ms(hw)
            a["ici_latency_ms"] += op.ici_lat_ms
            a["ici_bw_ms"] += op.ici_bw_ms
            a["roofline_ms"] += op.roofline_ms(hw)
            # Occupancy is what is live at once, and passes run one after another:
            # the MAX over passes, never the sum. Summing made a 256-pass prefill
            # report 256x the weights. Weights are equal per pass; KV peaks on the last.
            a["act_mib"] = max(a["act_mib"], op.act_occupancy / MiB)
            a["weight_mib"] = max(a["weight_mib"], op.w_bytes / MiB)
            a["kv_mib"] = max(a["kv_mib"], op.kv_resident_bytes / MiB)

    out: list[dict[str, Any]] = []
    for lc in m.layer_classes():
        mine = sorted(
            (a for a in acc.values() if a["layer_class"] in lc.op_classes),
            key=lambda r: r["order"])
        rows: list[dict[str, Any]] = []
        done: set[str] = set()
        for a in mine:
            g = a["parent"]
            if not g:
                rows.append(_row(lc, a, "op"))
                continue
            if g in done:  # emitted as a block with its first child
                continue
            done.add(g)
            kids = [k for k in mine if k["parent"] == g]
            rows.append(_row(lc, _merge(g, kids), "fused"))
            rows += [_row(lc, k, "sub") for k in kids]
        counted = [r for r in rows if r["role"] != "sub"]
        rows.append(_row(lc, _merge("TOTAL", counted, act_peak=True), "total"))
        out += rows
    return [
        dict(r, layer_total_ms=r["roofline_ms"] * r["layer_count"] * stage)
        for r in out
    ]


#: The numeric columns, which are all additive. Kept in one place so `_merge` and
#: `_row` cannot disagree about what a row carries.
_NUM = ("memory_ms", "compute_ms", "ici_latency_ms", "ici_bw_ms",
        "roofline_ms", "act_mib", "weight_mib", "kv_mib")


def _merge(name: str,
           rows: list[dict[str, Any]],
           *,
           act_peak: bool = False) -> dict[str, Any]:
    """Add up rows into the one row that stands for them -- a fused parent, or a TOTAL.

    `roofline_ms` is the SUM of the parts, not `max(mem, compute) + ici` recomputed
    over the summed columns. The parts are priced as ops that run one after another,
    so summing is the composition this file uses everywhere else; taking the max over
    the totals would quietly claim an overlap nothing here models. It also means a
    merged row's own columns do not satisfy `max(mem, comp) + ici = roofline`, which is
    correct and worth saying out loud in any report that shows both.
    """
    s = {k: sum(r[k] for r in rows) for k in _NUM}
    # Activations are TRANSIENT -- one op's working set is live at a time -- so a
    # layer's activation occupancy is the PEAK, not the sum. Weights and KV are
    # resident together and do add. Within a FUSED kernel the sub-ops' buffers DO
    # coexist, so a parent row still sums; only the layer TOTAL peaks.
    if act_peak:
        s["act_mib"] = max((r["act_mib"] for r in rows), default=0.0)
    # a merged row inherits a comment only when every part agrees; otherwise the
    # children carry their own and the parent says nothing rather than something
    # true of only some of it.
    notes = {r.get("comment", "") for r in rows}
    note = notes.pop() if len(notes) == 1 else ""
    # A merged row is only VPU work if every part is. Mixed -> "mxu", which is what
    # `_boundness` needs to avoid labelling a whole layer by its softmax. This is why
    # `_row` carries `unit`: TOTAL merges rows that `_row` already built, and dropping
    # it there made every total default to "mxu" without deriving it.
    unit = "vpu" if rows and all(r["unit"] == "vpu" for r in rows) else "mxu"
    return {
        "operator": name,
        "parent": "",
        "order": 0,
        "unit": unit,
        "comment": note,
        **s
    }


def _boundness(a: dict[str, Any]) -> str:
    if a["ici_latency_ms"] + a["ici_bw_ms"] > max(a["memory_ms"],
                                                  a["compute_ms"]):
        return "ICI"
    if a["memory_ms"] >= a["compute_ms"]:
        return "HBM"
    return "VPU" if a.get("unit") == "vpu" else "MXU"


def _row(lc: LayerClass, a: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "layer_kind": lc.key,
        "layer_count": lc.count,
        "role": role,
        "operator": a["operator"],
        "parent": a["parent"],
        "unit": a["unit"],
        "comment": a.get("comment", ""),
        "boundness": _boundness(a),
        **{
            k: a[k]
            for k in _NUM
        }
    }


def chunks_of(ph: Phase) -> list[Phase]:
    """Split a phase into the passes the engine actually runs.

    Chunked prefill: an ISL-token prompt becomes ceil(ISL/mnbt) passes of mnbt
    tokens, and pass i attends over roughly i*chunk + chunk/2 of context. The
    per-pass weight read is unchanged, which is exactly why a small chunk is
    expensive -- see README s3.1.
    """
    if not ph.mnbt or ph.mnbt >= ph.tokens:
        return [ph]
    n = math.ceil(ph.tokens / ph.mnbt)
    out = []
    for i in range(n):
        tok = min(ph.mnbt, ph.tokens - i * ph.mnbt)
        ctx = i * ph.mnbt + tok / 2.0
        out.append(replace(ph, tokens=int(tok), kv_len=int(ctx), mnbt=0))
    return out


def run_phase(cfg: Config, ph: Phase) -> PhaseResult:
    m, hw = cfg.model, cfg.hardware
    stage = 1.0 / cfg.sharding.pp
    passes = chunks_of(ph)
    agg: dict[str, dict[str, Any]] = {}
    total = 0.0
    for sub in passes:
        for op in build_ops(cfg, sub):
            cnt = multiplicity(m, op.layer_class) * stage
            r_ms = op.roofline_ms(hw)
            a = agg.setdefault(
                op.name, {
                    "op": op.name,
                    "layers": cnt,
                    "flops_per_layer": 0.0,
                    "hbm_mib_per_layer": 0.0,
                    "comp_ms": 0.0,
                    "mem_ms": 0.0,
                    "ici_ms": 0.0,
                    "roofline_ms": 0.0,
                    "bound": op.bound(hw),
                    "total_ms": 0.0
                })
            a["flops_per_layer"] += op.flops
            a["hbm_mib_per_layer"] += op.hbm_bytes / MiB
            a["comp_ms"] += op.comp_ms(hw)
            a["mem_ms"] += op.mem_ms(hw)
            a["ici_ms"] += op.ici_ms
            a["roofline_ms"] += r_ms
            a["total_ms"] += r_ms * cnt
            if r_ms * cnt > 0:
                a["bound"] = op.bound(hw)
            total += r_ms * cnt
    rows = list(agg.values())
    for r in rows:
        r["pct"] = 100.0 * r["total_ms"] / total if total else 0.0
    rows.sort(key=lambda r: -r["total_ms"])
    return PhaseResult(ph, rows, total,
                       weight_bytes(cfg) / GiB,
                       kv_bytes_per_token(cfg) * ph.kv_len * ph.batch / GiB,
                       layer_rows(cfg, ph))


def _attn_params(m: Model) -> tuple[float, float]:
    """(TP-sharded, replicated) attention parameters of ONE layer.

    MLA's two down-projections -- `q_a_proj` [D, q_lora] and `kv_a_proj_with_mqa`
    [D, kv_lora + rope] -- project into the shared latent, which has no head axis to
    split, so vLLM replicates them (`DeepSeekV2FusedQkvAProjLinear`, disable_tp=True).
    parallelism/tp/tp.md s3. Everything else is head-sharded.
    """
    D = float(m.hidden)
    if m.use_mla:
        rep = D * m.q_lora_rank + D * (m.kv_lora_rank + m.qk_rope_head_dim)
        sh = (m.q_lora_rank * m.heads *
              (m.qk_nope_head_dim + m.qk_rope_head_dim) +
              m.heads * m.kv_lora_rank * (m.qk_nope_head_dim + m.v_head_dim) +
              m.heads * m.v_head_dim * D)
        return sh, rep
    return D * (m.heads +
                2 * m.kv_heads) * m.head_dim + m.heads * m.head_dim * D, 0.0


def _layer_weight_bytes(cfg: Config) -> list[float]:
    """Resident weight bytes of each layer, per core, before PP."""
    m = cfg.model
    t = _per_layer_weight_terms(cfg)
    idx = set(m.indexer_layer_ids())
    return [
        t["attn"][0] + (t["indexer"][0] if i in idx else 0.0) +
        (t["dense"][0] if i < m.dense_layers else t["moe"][0])
        for i in range(m.layers)
    ]


def _embedding_bytes(cfg: Config, k: int) -> tuple[float, float]:
    """(stage 0, last stage) embedding / LM-head bytes per core. Tied embeddings are
    one tensor at pp=1 and a copy on each end otherwise."""
    m, s = cfg.model, cfg.sharding
    one = m.vocab * m.hidden / s.tp * cfg.numerics.b("weights")
    if k <= 1:
        return one * (1 if m.tie_embeddings else 2), 0.0
    return one, one


def stage_weight_bytes(cfg: Config) -> list[float]:
    """Per-core weight bytes of each pipeline stage. parallelism/pp/pp.md s5."""
    lw, out, i = _layer_weight_bytes(cfg), [], 0
    part = stage_layers(cfg)
    first, last = _embedding_bytes(cfg, len(part))
    for j, take in enumerate(part):
        out.append(
            sum(lw[i:i + take]) + (first if j == 0 else 0.0) +
            (last if j == len(part) - 1 and len(part) > 1 else 0.0))
        i += take
    return out


def weight_bytes(cfg: Config) -> float:
    """Per-core weight bytes -- the HEAVIEST pipeline stage, which is what fits or
    does not. parallelism/{tp,ep,pp}/<scheme>.md s3/s5."""
    if cfg.sharding.pp > 1:
        return max(stage_weight_bytes(cfg))
    return _weight_bytes_flat(cfg)


def _weight_bytes_flat(cfg: Config) -> float:
    """Per-core weight bytes without PP. parallelism/{tp,ep}/<scheme>.md s3."""
    return sum(t.local_bytes for t in weight_terms(cfg))


@dataclass(frozen=True)
class WeightTerm:
    """One weight tensor family, whole-model and per core, with the rule that divides it.

    `global_bytes / divisor * stage = local_bytes`, and summing `local_bytes` over
    `weight_terms` IS `_weight_bytes_flat` -- the aggregate is built from these, so a
    report that shows the terms cannot disagree with the admission number.
    """
    tensor: str
    layers: int  # how many layers carry it (1 for the embedding)
    global_bytes: float  # unsharded, all layers
    divisor: int  # the shard count on the per-core copy
    divisor_expr: str  # the divisor as axes, e.g. "TP*PCP*DP" or "1"
    rule: str  # why that divisor. Cites parallelism/<axis>/<axis>.md
    stage: float  # 1/pp -- the MEAN stage; the heaviest is `stage_weight_bytes`

    @property
    def local_bytes(self) -> float:
        return self.global_bytes / self.divisor * self.stage


def weight_terms(cfg: Config) -> list[WeightTerm]:
    """Per-core weight bytes, term by term, before PP balancing.

    parallelism/{tp,ep}/<scheme>.md s3. Zero-byte terms (no indexer, no dense layers) are
    dropped, so a dense GQA model returns only what it has.
    """
    m, n, s = cfg.model, cfg.numerics, cfg.sharding
    tp, stage = s.tp, 1.0 / s.pp  # PCP replicates TP's shards
    wb, mb = n.b("weights"), n.b("moe_weights")
    D = float(m.hidden)
    attn, attn_rep = _attn_params(m)
    # The head-weights projection carries its own numerics site (w16a16 against the
    # K/Q projections' w8a8), so it cannot ride along on `wb` -- `dsa_head_weights`
    # would then disagree with this aggregate, which is what
    # `test_weight_bytes_agrees_with_the_op_table_on_the_indexer` exists to catch.
    idx = ((m.q_lora_rank or D) * m.indexer_heads * m.indexer_head_dim +
           D * m.indexer_head_dim) if m.indexer_layers else 0.0
    idx_wp = D * m.indexer_heads if m.indexer_layers else 0.0
    # Routed experts: EP gives each device E/EP whole experts; without EP every
    # device holds every expert but only F_expert/TP of each. `_moe_dims` already
    # says this on the op side -- parallelism/tp/tp.md:88. Dividing by
    # ep_group() alone leaves the non-EP case undivided, which on GLM-5.2 reports
    # 676 GiB/core against 43 and drives kv_pool_bytes negative.
    if s.ep:
        expert_div, expert_expr = s.ep_group(), "TP*PCP*DP"
        expert_rule = (
            f"EP on: each core holds {m.n_experts}/{s.ep_group()} whole "
            f"experts; the EP group spans TP*PCP*DP (parallelism/ep §3)")
    else:
        expert_div, expert_expr = tp * s.pcp, "TP*PCP"
        expert_rule = (
            "EP off: every core holds every expert, 1/(TP*PCP) of each "
            "one's F columns (parallelism/tp §3)")
    head_rule = "head-sharded by TP; PCP replicates TP's shards (parallelism/tp §3)"
    terms = [
        WeightTerm("attention, head-sharded projections", m.layers,
                   attn * m.layers * wb, tp, "TP", head_rule, stage),
        WeightTerm("attention, latent down-projections", m.layers,
                   attn_rep * m.layers * wb, 1, "1",
                   "no head axis to split -- replicated (parallelism/tp §3)",
                   stage),
        # the indexer's projections are replicated too: `wq_b` is ReplicatedLinear and
        # `wk_weights_proj` is MergedColumnParallelLinear(disable_tp=True)
        # (vllm/model_executor/models/deepseek_v2.py:669-683)
        WeightTerm("indexer K/Q projections", m.indexer_layers,
                   idx * m.indexer_layers * wb, 1, "1",
                   "ReplicatedLinear -- replicated (deepseek_v2.py:669-683)",
                   stage),
        WeightTerm("indexer head weights", m.indexer_layers,
                   idx_wp * m.indexer_layers * n.b("indexer_w_proj"), 1, "1",
                   "disable_tp=True -- replicated (deepseek_v2.py:669-683)",
                   stage),
        WeightTerm("dense FFN", m.dense_layers,
                   3 * D * m.dense_ffn * m.dense_layers * wb, tp, "TP",
                   "column/row-sharded by TP (parallelism/tp §3)", stage),
        WeightTerm("routed experts", m.moe_layers,
                   3 * D * m.expert_ffn * m.n_experts * m.moe_layers * mb,
                   expert_div, expert_expr, expert_rule, stage),
        WeightTerm(
            "shared expert", m.moe_layers,
            3 * D * m.shared_ffn * m.n_shared_experts * m.moe_layers * wb, tp,
            "TP", "sharded by TP like a dense FFN (parallelism/tp §3)", stage),
        WeightTerm("router", m.moe_layers, D * m.n_experts * m.moe_layers * wb,
                   1, "1", "replicated", stage),
        WeightTerm("embedding + LM head", 1,
                   m.vocab * D * (1 if m.tie_embeddings else 2) * wb, tp, "TP",
                   "vocab-sharded by TP (parallelism/tp §3)", stage),
    ]
    return [t for t in terms if t.global_bytes > 0]


@dataclass
class KindOccupancy:
    """RESIDENT bytes for all layers of one kind. Not the op tables' traffic.

    The two are different questions and the same row cannot answer both. The op table's
    occupancy columns are what an op MOVES in a pass -- they divide by HBM bandwidth to
    give its Memory Time, and under a chunked prefill the weight term is counted once
    per pass because it really is re-read once per pass. This is what SITS in HBM: the
    weight read once, the KV for the whole context, counted once however many passes
    touch it.

    `local` is this core. `global_` is the unsharded whole, so `global_ / world / local`
    says how close the partition is to the perfectly-sharded floor -- 100% when every
    byte is split, 1/world when nothing is.
    """

    weights_local: float = 0.0
    weights_global: float = 0.0
    kv_local: float = 0.0
    kv_global: float = 0.0
    act_peak_local: float = 0.0  # largest single op's activation bytes, one layer

    def eff(self, world: int, which: str) -> float:
        g = getattr(self, f"{which}_global")
        loc = getattr(self, f"{which}_local")
        return (g / world / loc) if loc > 0 else 0.0


def _per_layer_weight_terms(cfg: Config) -> dict[str, tuple[float, float]]:
    """Resident weight bytes of ONE layer, per term, as (local, global).

    Split out of `weight_bytes` rather than restated, so the aggregate and the per-kind
    table cannot disagree -- they are the same parameters counted once.
    """
    m, n, s = cfg.model, cfg.numerics, cfg.sharding
    tp = s.tp  # PCP replicates TP's shards
    wb, mb = n.b("weights"), n.b("moe_weights")
    D = float(m.hidden)
    attn_sh, attn_rep = (x * wb for x in _attn_params(m))
    attn = attn_sh + attn_rep
    idx = (
        (m.q_lora_rank or D) * m.indexer_heads * m.indexer_head_dim + D *
        m.indexer_head_dim) * wb + D * m.indexer_heads * n.b("indexer_w_proj")
    dense = 3 * D * m.dense_ffn * wb
    experts = 3 * D * m.expert_ffn * m.n_experts * mb
    shared = 3 * D * m.shared_ffn * m.n_shared_experts * wb
    router = D * m.n_experts * wb
    expert_div = s.ep_group(
    ) if s.ep else tp * s.pcp  # TP flattened across PCP
    return {
        "attn": (attn_sh / tp + attn_rep, attn),
        "indexer": (idx, idx),  # replicated, as in weight_bytes
        "dense": (dense / tp, dense),
        # routed experts shard by the EP group with EP on and by TP without it --
        # `_moe_dims` says the same thing on the op side
        "moe": ((experts / expert_div + shared / tp + router),
                experts + shared + router),
    }


#: Which weight terms a layer kind carries. Mirrors `LayerClass.op_classes`.
_KIND_TERMS = {
    "dense_idx": ("attn", "indexer", "dense"),
    "dense": ("attn", "dense"),
    "moe_idx": ("attn", "indexer", "moe"),
    "moe": ("attn", "moe"),
}


def layer_occupancy(cfg: Config, ph: Phase) -> dict[str, KindOccupancy]:
    """Resident weights and KV for ALL layers of each kind, local and global."""
    m, n, s = cfg.model, cfg.numerics, cfg.sharding
    terms = _per_layer_weight_terms(cfg)
    cp = float(s.dcp * _pcp_kv_div(m, s))
    if m.use_mla:
        kv_per_tok, kv_shard = ((m.kv_lora_rank + m.qk_rope_head_dim) *
                                n.b("kv_cache"), cp)
    else:
        kv_per_tok = 2 * m.kv_heads * m.head_dim * n.b("kv_cache")
        kv_shard = float(min(s.tp, max(1, m.kv_heads))) * cp
    ctx = float(ph.kv_len) * max(1, ph.batch)
    idx_per_tok = float(m.indexer_cache_bytes_per_token)

    # peak activation, per layer kind: the largest single op, from the op table
    # max over the rows that COUNT (op + fused), matching the op table's TOTAL. A
    # fused parent's buffers coexist inside its kernel, so the parent is the peak for
    # that kernel; its children must not be taken individually.
    peak: dict[str, float] = {}
    for r in layer_rows(cfg, ph):
        if r["role"] in ("op", "fused"):
            peak[r["layer_kind"]] = max(peak.get(r["layer_kind"], 0.0),
                                        r["act_mib"] * MiB)

    out: dict[str, KindOccupancy] = {}
    for lc in m.layer_classes():
        c = float(lc.count)
        wl = sum(terms[t][0] for t in _KIND_TERMS[lc.key]) * c
        wg = sum(terms[t][1] for t in _KIND_TERMS[lc.key]) * c
        kvl = kv_per_tok * ctx / kv_shard * c
        kvg = kv_per_tok * ctx * c
        if "indexer" in _KIND_TERMS[lc.key]:
            kvl += idx_per_tok * ctx / cp * c
            kvg += idx_per_tok * ctx * c
        out[lc.key] = KindOccupancy(wl / s.pp, wg, kvl / s.pp, kvg,
                                    peak.get(lc.key, 0.0))
    return out


def layer_summary(cfg: Config, res: PhaseResult) -> list[dict[str, Any]]:
    """The layer-sums table: time and resident footprint, one row per layer kind."""
    occ = layer_occupancy(cfg, res.phase)
    world = cfg.hardware.world
    rows = []
    for t in [x for x in res.layer_rows if x["role"] == "total"]:
        o = occ[t["layer_kind"]]
        rows.append({
            "layer_kind":
            t["layer_kind"],
            "layers":
            t["layer_count"],
            "per_layer_ms":
            t["roofline_ms"],
            "all_layer_ms":
            t["layer_total_ms"],
            "pct_total":
            100.0 * t["layer_total_ms"] /
            res.total_ms if res.total_ms else 0.0,
            "act_local_mib":
            o.act_peak_local / MiB,
            "weights_local_mib":
            o.weights_local / MiB,
            "kv_local_mib":
            o.kv_local / MiB,
            "act_global_mib":
            o.act_peak_local / MiB,
            "weights_global_mib":
            o.weights_global / MiB,
            "kv_global_mib":
            o.kv_global / MiB,
            "weights_eff":
            o.eff(world, "weights"),
            "kv_eff":
            o.eff(world, "kv"),
        })
    tot = {
        "layer_kind": "TOTAL",
        "layers": sum(r["layers"] for r in rows),
        "per_layer_ms": 0.0,
        "all_layer_ms": sum(r["all_layer_ms"] for r in rows),
        "pct_total": sum(r["pct_total"] for r in rows),
        # activations are transient -- the peak, not the sum. Weights and KV are
        # resident in every layer at once, so those do add.
        "act_local_mib": max((r["act_local_mib"] for r in rows), default=0.0),
        "act_global_mib": max((r["act_global_mib"] for r in rows),
                              default=0.0),
    }
    for k in ("weights_local_mib", "kv_local_mib", "weights_global_mib",
              "kv_global_mib"):
        tot[k] = sum(r[k] for r in rows)
    for w in ("weights", "kv"):
        g, loc = tot[f"{w}_global_mib"], tot[f"{w}_local_mib"]
        tot[f"{w}_eff"] = (g / world / loc) if loc else 0.0
    return rows + [tot]


@dataclass(frozen=True)
class KVTerm:
    """One cache, per token: whole-model and per core, with the rule that divides it.
    Summing `local_bytes_per_token` over `kv_terms` IS `kv_bytes_per_token` at pp=1."""
    cache: str
    layers: int
    bytes_per_token_layer: float  # one token, one layer, unsharded
    divisor: int
    divisor_expr: str
    rule: str

    @property
    def global_bytes_per_token(self) -> float:
        return self.bytes_per_token_layer * self.layers

    @property
    def local_bytes_per_token(self) -> float:
        return self.global_bytes_per_token / self.divisor


def kv_terms(cfg: Config) -> list[KVTerm]:
    """Per-token cache bytes, cache by cache, before PP. parallelism/{tp,dcp,pcp}/<scheme>.md."""
    m, n, s = cfg.model, cfg.numerics, cfg.sharding
    pk = _pcp_kv_div(m, s)
    cp = s.dcp * pk
    cp_expr = "DCP*PCP" if pk > 1 else "DCP"
    cp_rule = (f"split along the sequence by DCP={s.dcp}" +
               (f" and PCP={s.pcp} (ring)" if pk > 1 else "") +
               "; TP does not split it (parallelism/dcp §3)")
    out = []
    if m.use_mla:
        out.append(
            KVTerm(
                "MLA latent (c_kv + k_rope)", m.layers,
                (m.kv_lora_rank + m.qk_rope_head_dim) * n.b("kv_cache"), cp,
                cp_expr,
                "one latent per token shared by every head, so no head axis "
                "-- " + cp_rule))
    else:
        h = min(s.tp, max(1, m.kv_heads))
        out.append(
            KVTerm("K and V", m.layers,
                   2 * m.kv_heads * m.head_dim * n.b("kv_cache"), h * cp,
                   f"min(TP, kv_heads)*{cp_expr}",
                   f"KV heads split by TP up to {m.kv_heads}; " + cp_rule))
    if m.indexer_layers and m.indexer_cache_bytes_per_token:
        out.append(
            KVTerm("DSA indexer K cache", m.indexer_layers,
                   float(m.indexer_cache_bytes_per_token), cp, cp_expr,
                   cp_rule))
    return out


def kv_bytes_per_token(cfg: Config) -> float:
    """Per core, all layers. parallelism/{tp,dcp}/<scheme>.md s3."""
    m, n, s = cfg.model, cfg.numerics, cfg.sharding
    cp = s.dcp * _pcp_kv_div(m, s)
    if m.use_mla:
        per, shard = (m.kv_lora_rank +
                      m.qk_rope_head_dim) * n.b("kv_cache"), cp
    else:
        per = 2 * m.kv_heads * m.head_dim * n.b("kv_cache")
        shard = min(s.tp, max(1, m.kv_heads)) * cp
    idx_per = m.indexer_cache_bytes_per_token
    if s.pp <= 1:
        return sum(t.local_bytes_per_token for t in kv_terms(cfg))
    # the HEAVIEST stage: its attention layers, and only its own indexer layers
    # (`indexer_cache_bytes_per_token` is per indexer layer)
    idx, out, i = set(m.indexer_layer_ids()), [], 0
    for take in stage_layers(cfg):
        ni = sum(1 for lid in range(i, i + take) if lid in idx)
        out.append(per * take / shard + idx_per * ni / cp)
        i += take
    return max(out)


def _pcp_kv_div(m: Model, s: Sharding) -> int:
    """PCP's divisor on KV per device: PCP under the ring's striped cache, 1 under a
    replicated one. parallelism/pcp/pcp.md s5."""
    return s.pcp if s.pcp > 1 and m.pcp_kv_strategy == "ring" else 1


# ============================================================ io

#: Hardware fields a config must set. Nothing falls back to a value in this file.
HARDWARE_REQUIRED = tuple(f.name for f in fields(Hardware)
                          if f.default is MISSING)


def load(path: str) -> Config:
    raw = json.load(open(path))
    hw = raw.get("hardware", {})
    missing = [k for k in HARDWARE_REQUIRED if k not in hw]
    if missing:
        raise ValueError(
            f"{path}: `hardware` is missing {', '.join(missing)}. Every hardware"
            " value comes from the config; take published figures from"
            " tpu-knowledge-base and record each source in `_provenance.hardware`."
        )
    if "slice_dims" in hw:
        hw = {**hw, "slice_dims": tuple(hw["slice_dims"])}
    return Config(
        model=Model(**raw["model"]),
        numerics=Numerics(**raw.get("numerics", {})),
        hardware=Hardware(**hw),
        sharding=Sharding(**raw.get("sharding", {})),
        phases=tuple(Phase(**p) for p in raw["phases"]),
        workload=Workload(**raw["workload"]) if "workload" in raw else None,
    )


def report(cfg: Config, results: list[PhaseResult], top: int) -> None:
    m, hw, s = cfg.model, cfg.hardware, cfg.sharding
    c = m.counts
    print(
        f"\n{m.name}   {hw.gen} {'x'.join(map(str,hw.slice_dims))} = {hw.world} cores"
        f"   sharding {s.label}")
    print(
        f"layers {m.layers} = {c['dense']} dense + {c['moe_idx']} MoE+indexer"
        f" + {c['moe_plain']} MoE   |   indexer runs on {m.indexer_layers}")
    bad = s.check(hw, m)
    if bad:
        print(f"  note: vLLM would not accept this sharding ({bad}). "
              f"Priced anyway -- this script computes, it does not gate.")
    if cfg.numerics.unset:
        print(f"  ! numerics defaulted to {cfg.numerics.default} at: "
              f"{', '.join(cfg.numerics.unset)}")
    if hw.kernel_efficiency >= 1.0:
        print(
            "  ! kernel_efficiency 1.0 and collectives fully exposed -- these are "
            "ceilings. Use ratios.")

    for r in results:
        ph = r.phase
        np_ = len(chunks_of(ph))
        chunk = f"  mnbt={ph.mnbt} -> {np_} passes of {min(ph.mnbt, ph.tokens)}" \
            if np_ > 1 else ""
        print(
            f"\n=== {ph.name}   T={ph.tokens}/core  B={ph.batch}  S={ph.kv_len}"
            f"{chunk}   E2E {r.total_ms:.4f} ms")
        print(
            f"    HBM: weights {r.weights_gib:.2f} GiB + KV {r.kv_gib:.2f} GiB"
            f" = {r.weights_gib + r.kv_gib:.2f} of {hw.hbm_capacity/GiB:.0f} GiB"
            f" ({100*(r.weights_gib+r.kv_gib)/(hw.hbm_capacity/GiB):.1f}%)")
        h = (f"{'operator':<34}{'ms/layer':>10}{'x':>5}{'total ms':>10}"
             f"{'%':>7}{'bound':>6}{'HBM ms':>9}{'MXU ms':>9}{'ICI ms':>9}")
        print("    " + h)
        print("    " + "-" * len(h))
        for row in r.rows[:top]:
            if row["layers"] == 0:
                continue
            print(
                f"    {row['op']:<34}{row['roofline_ms']:>10.5f}"
                f"{row['layers']:>5.0f}{row['total_ms']:>10.4f}{row['pct']:>7.2f}"
                f"{row['bound']:>6}{row['mem_ms']:>9.5f}{row['comp_ms']:>9.5f}"
                f"{row['ici_ms']:>9.5f}")
        by = {}
        for row in r.rows:
            by[row["bound"]] = by.get(row["bound"], 0.0) + row["total_ms"]
        print("    " +
              "  ".join(f"{k} {100*v/r.total_ms:.1f}%"
                        for k, v in sorted(by.items(), key=lambda x: -x[1])))
        print(f"    phase rate: {ph.tokens * hw.world / (r.total_ms/1e3):,.0f}"
              f" tok/s over {hw.world} cores (this phase alone)")


def kv_pool_bytes(cfg: Config) -> float:
    """HBM left for the KV cache, per core. vllm-torchtpu utils.py:171."""
    hw = cfg.hardware
    cap = hw.hbm_capacity * hw.memory_utilization
    return cap - weight_bytes(cfg) - hw.resident_overhead_gb * GiB


def kv_capacity_reqs(cfg: Config) -> float:
    """Requests a core can hold, sized at FULL context. The admission gate."""
    w = cfg.workload
    per_req = kv_bytes_per_token(cfg) * (w.isl + w.osl)
    pool = kv_pool_bytes(cfg)
    return pool / per_req if per_req > 0 else float("inf")


@dataclass
class Serving:
    """README s5. Every field is derived from T_prefill and T_step."""
    t_prefill_ms: float  # whole prompt, all chunks
    t_step_ms: float  # one decode step
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    reqs_per_core_prefill: float
    reqs_per_core_decode: float
    qps_prefill: float
    qps_decode: float
    qps_combined: float
    tok_s_per_core: float
    tok_s_fleet: float
    prefill_share: float
    kv_pool_gib: float = 0.0
    kv_cap_reqs: float = 0.0
    gate: str = ""


def serving(cfg: Config, results: list[PhaseResult]) -> Serving | None:
    w = cfg.workload
    pre = next((r for r in results if r.phase.name == "prefill"), None)
    dec = next((r for r in results if r.phase.name == "decode"), None)
    if not (w and pre and dec):
        return None
    Tp, Ts = pre.total_ms, dec.total_ms
    # requests a core prefills in one prefill pass, and holds in decode
    rp = pre.phase.tokens / w.isl
    rd = float(dec.phase.batch)
    qp = rp / (Tp / 1e3)  # req/s/core if it only prefilled
    qd = rd / (w.osl * Ts / 1e3)  # req/s/core if it only decoded
    qc = 1.0 / (1.0 / qp + 1.0 / qd)  # it must do both, serially
    cap = kv_capacity_reqs(cfg)
    return Serving(
        kv_pool_gib=kv_pool_bytes(cfg) / GiB,
        kv_cap_reqs=cap,
        gate="kv" if cap < rd else "offered",
        t_prefill_ms=Tp,
        t_step_ms=Ts,
        ttft_ms=Tp,
        tpot_ms=Ts,
        e2e_ms=Tp + (w.osl - 1) * Ts,
        reqs_per_core_prefill=rp,
        reqs_per_core_decode=rd,
        qps_prefill=qp,
        qps_decode=qd,
        qps_combined=qc,
        tok_s_per_core=qc * (w.isl + w.osl),
        tok_s_fleet=qc * (w.isl + w.osl) * cfg.hardware.world,
        prefill_share=(1.0 / qp) / (1.0 / qp + 1.0 / qd),
    )


def report_serving(cfg: Config, sv: Serving) -> None:
    w = cfg.workload
    print(f"\n=== serving   ISL {w.isl} / OSL {w.osl}   (README s5)")
    print(
        f"    T_prefill  = {sv.t_prefill_ms:10.3f} ms   whole prompt, all chunks"
    )
    print(f"    T_step     = {sv.t_step_ms:10.3f} ms   one decode step")
    print(
        f"    TTFT       = T_prefill                       = {sv.ttft_ms:10.1f} ms"
    )
    print(
        f"    TPOT       = T_step                          = {sv.tpot_ms:10.3f} ms"
    )
    print(
        f"    E2E/req    = TTFT + (OSL-1)*TPOT             = {sv.e2e_ms/1e3:10.2f} s"
    )
    print(f"    QPS_pre    = {sv.reqs_per_core_prefill:.3g} req / T_prefill"
          f"           = {sv.qps_prefill:10.4f} req/s/core")
    print(f"    QPS_dec    = {sv.reqs_per_core_decode:.3g} req / (OSL*T_step)"
          f"     = {sv.qps_decode:10.4f} req/s/core")
    print(
        f"    QPS        = 1/(1/QPS_pre + 1/QPS_dec)       = {sv.qps_combined:10.4f}"
        f" req/s/core")
    print(
        f"    throughput = QPS * (ISL+OSL)                 = {sv.tok_s_per_core:10.0f}"
        f" tok/s/core   ({sv.tok_s_fleet:,.0f} fleet)")
    who = "prefill" if sv.prefill_share > 0.5 else "decode"
    print(f"    prefill takes {100*sv.prefill_share:.1f}% of the time budget"
          f" -> {who} dominates throughput")
    hw = cfg.hardware
    print(
        f"\n    KV pool    = {hw.hbm_capacity/GiB:.0f} GiB x {hw.memory_utilization}"
        f" - weights - {hw.resident_overhead_gb:.0f} GiB"
        f"   = {sv.kv_pool_gib:8.2f} GiB   (vllm-torchtpu utils.py:171)")
    print(f"    KV gate    = pool / (kv_per_token x (ISL+OSL))"
          f" = {sv.kv_cap_reqs:8.1f} req/core")
    pct = 100 * sv.reqs_per_core_decode / sv.kv_cap_reqs if sv.kv_cap_reqs else 0.0
    print(f"    configured   {sv.reqs_per_core_decode:.0f} req/core"
          f"                        = {pct:8.0f}% of the KV gate")


def report_layers(cfg: Config, results: list[PhaseResult]) -> None:
    """The per-layer op tables, one per layer kind per phase.

    Deliberately plain. This prints the `layer_kind` KEY, not a name -- naming and the
    shape of the finished report are the tpu-roofline prompt's, and a pretty label baked in here would
    be a second place to change when a model needs different vocabulary.
    """
    hdr = (
        f"{'operator':<38}{'mem ms':>9}{'comp ms':>9}{'ICI lat':>9}{'ICI bw':>9}"
        f"{'roofline':>10}{'bound':>7}{'act MiB':>10}{'wt MiB':>10}{'KV MiB':>10}"
    )
    for r in results:
        if not r.layer_rows:
            continue
        print(f"\n=== {r.phase.name}: per-layer op tables"
              f"   (one layer of each kind, one core, all"
              f" {len(chunks_of(r.phase))} pass(es) of the phase)")
        for kind in dict.fromkeys(x["layer_kind"] for x in r.layer_rows):
            block = [x for x in r.layer_rows if x["layer_kind"] == kind]
            n = block[0]["layer_count"]
            print(f"\n  [{kind}]  x{n}")
            print("  " + hdr)
            print("  " + "-" * len(hdr))
            notes = []
            for x in block:
                if x["role"] == "total":
                    print("  " + "-" * len(hdr))
                nm = ("  --> " +
                      x["operator"]) if x["role"] == "sub" else x["operator"]
                mark = ""
                if x["comment"]:
                    notes.append((len(notes) + 1, x["operator"], x["comment"]))
                    mark = f" [{len(notes)}]"
                print(
                    f"  {nm + mark:<38}{x['memory_ms']:>9.4f}{x['compute_ms']:>9.4f}"
                    f"{x['ici_latency_ms']:>9.4f}{x['ici_bw_ms']:>9.4f}"
                    f"{x['roofline_ms']:>10.4f}{x['boundness']:>7}"
                    f"{x['act_mib']:>10.2f}{x['weight_mib']:>10.2f}"
                    f"{x['kv_mib']:>10.2f}")
            # Comments as footnotes, not a column: they are sentences, and a column
            # wide enough for them makes the numbers unreadable. The CSV carries the
            # same text in a real `Comment` column for a spreadsheet.
            for i, op, note in notes:
                print(f"    [{i}] {op}: {note}")

        print(
            f"\n  layer sums, {r.phase.name}   (occupancy is RESIDENT bytes, not the"
            f" op tables' traffic)")
        h2 = (
            f"  {'layer kind':<12}{'#':>4}{'ms/layer':>10}{'all ms':>10}{'%':>7}"
            f"{'act L':>10}{'wt L':>12}{'KV L':>11}"
            f"{'act G':>10}{'wt G':>12}{'KV G':>11}{'wt eff':>8}{'KV eff':>8}")
        print(h2)
        print("  " + "-" * (len(h2) - 2))
        for x in layer_summary(cfg, r):
            ms = f"{x['per_layer_ms']:>10.4f}" if x[
                "per_layer_ms"] else f"{'':>10}"
            print(
                f"  {x['layer_kind']:<12}{x['layers']:>4}{ms}"
                f"{x['all_layer_ms']:>10.4f}{x['pct_total']:>7.2f}"
                f"{x['act_local_mib']:>10.2f}{x['weights_local_mib']:>12.2f}"
                f"{x['kv_local_mib']:>11.2f}{x['act_global_mib']:>10.2f}"
                f"{x['weights_global_mib']:>12.2f}{x['kv_global_mib']:>11.2f}"
                f"{100 * x['weights_eff']:>7.2f}%{100 * x['kv_eff']:>7.2f}%")


def write_csv(path: str, results: list[PhaseResult]) -> None:
    rows = [{"phase": r.phase.name, **row} for r in results for row in r.rows]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path} ({len(rows)} rows)")


def write_layer_csv(path: str, results: list[PhaseResult]) -> None:
    """The per-layer tables as one long-form CSV -- a phase column, a layer-kind column,
    one row per operator. Long form rather than one file per table so it pivots."""
    head = (["Phase", "Layer Kind", "Layer Count", "Role"] +
            [t for _, t in LAYER_COLUMNS] + ["Layer Total (ms)"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        n = 0
        for r in results:
            for x in r.layer_rows:
                nm = ("--> " +
                      x["operator"]) if x["role"] == "sub" else x["operator"]
                w.writerow([
                    r.phase.name, x["layer_kind"], x["layer_count"], x["role"]
                ] + [nm] + [x[k] for k, _ in LAYER_COLUMNS[1:]] +
                           [x["layer_total_ms"]])
                n += 1
    print(f"wrote {path} ({n} rows)")


# ============================================================ selftest


def selftest() -> int:
    import unittest
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = load(os.path.join(here, "configs", "glm-5.2-fp8-8k1k-2x2x2.json"))
    hw, n, m = cfg.hardware, cfg.numerics, cfg.model
    # The reference workbook models ONE un-chunked 8192-token pass. Validate the
    # op formulas against that shape regardless of the config's mnbt, which only
    # decides how many passes the report sums.
    pre = replace(next(p for p in cfg.phases if p.name == "prefill"), mnbt=0)
    ops = {o.name: o for o in build_ops(cfg, pre)}

    class T(unittest.TestCase):
        """Compute times cross-checked against a hand-built workbook roofline.

        The workbook used a 1785.62 TFLOP/s fp8 peak. Each value here is the
        workbook value x 1785.62 / 2307, the published per-core peak.
        """

        def ms(self, name):
            return ops[name].comp_ms(hw)

        def test_o_proj(self):
            self.assertAlmostEqual(self.ms("MLA: O Proj"), 0.71490, places=4)

        def test_qk_nope(self):
            self.assertAlmostEqual(self.ms("MLA: QK Nope"), 0.95320, places=4)

        def test_qk_rope_pays_4x_for_mxu_padding(self):
            """D_rope=64 padded to 256. Three quarters of this op is padding."""
            self.assertAlmostEqual(self.ms("MLA: QK Rope"), 0.47660, places=4)
            unpadded = 2 * pre.tokens * m.heads * m.qk_rope_head_dim * m.indexer_topk
            self.assertAlmostEqual(ops["MLA: QK Rope"].flops / unpadded,
                                   4.0,
                                   places=6)

        def test_av(self):
            self.assertAlmostEqual(self.ms("MLA: AV"), 0.95320, places=4)

        def test_q_down_up_kv_down(self):
            self.assertAlmostEqual(self.ms("MLA: Query Down Proj"),
                                   0.08936,
                                   places=4)
            self.assertAlmostEqual(self.ms("MLA: Query Up Proj"),
                                   0.23830,
                                   places=4)
            self.assertAlmostEqual(self.ms("MLA: KV Down Proj"),
                                   0.02513,
                                   places=4)

        def test_absorption_projections(self):
            self.assertAlmostEqual(self.ms("MLA: Q Proj to K (absorb)"),
                                   0.05957,
                                   places=4)
            self.assertAlmostEqual(self.ms("MLA: V Up Proj (absorb)"),
                                   0.05957,
                                   places=4)

        def test_moe_and_dense_ffn(self):
            self.assertAlmostEqual(self.ms("MoE: FFi + FFg"),
                                   1.42980,
                                   places=4)
            self.assertAlmostEqual(self.ms("MoE: FFo"), 0.71490, places=4)
            self.assertAlmostEqual(self.ms("MoE: Shared FFi + FFg"),
                                   0.17872,
                                   places=4)
            self.assertAlmostEqual(self.ms("Dense: FFi + FFg"),
                                   1.07235,
                                   places=4)

        def test_indexer(self):
            self.assertAlmostEqual(self.ms("DSA Indexer: QK"),
                                   0.47660,
                                   places=4)
            self.assertAlmostEqual(self.ms("DSA Indexer: Q Proj"),
                                   0.05957,
                                   places=4)

        def test_query_down_proj_hbm_counts_activations(self):
            """92 MiB = 12 weights + 48 act in + 32 act out. Weights alone is 13%."""
            self.assertAlmostEqual(ops["MLA: Query Down Proj"].hbm_bytes / MiB,
                                   92.0,
                                   places=3)

        def test_layer_classes_sum(self):
            self.assertEqual(sum(m.counts.values()), m.layers)
            self.assertEqual(m.counts["dense"], 3)
            self.assertEqual(m.indexer_layers, 21)

        def test_ep_issues_three_collectives(self):
            names = [o.name for o in collectives(cfg, pre)]
            # the dispatch op's NAME carries the strategy -- match the role, not
            # one schedule, or setting a config's strategy fails an unrelated test
            self.assertTrue(any(x.startswith("MoE: Dispatch") for x in names),
                            names)
            self.assertIn("MoE: Collect (reduce-scatter)", names)
            self.assertIn("MoE: Router logits (all-gather)", names)

        def test_no_tp_means_no_tp_allreduce(self):
            self.assertFalse(
                any(o.name.startswith("TP:") for o in collectives(cfg, pre)))

        def test_sharding_is_legal(self):
            self.assertIsNone(cfg.sharding.check(hw, m))

        def test_a_decode_phase_derives_its_batch_from_mnbt_and_can_override(
                self):
            """`batch` is the decode running set. Omitted it follows the token
            budget, because a decode step spends one token per request."""
            d = Phase("decode", kv_len=9216, mnbt=1024)
            self.assertEqual((d.batch, d.tokens), (1024, 1024))
            o = Phase("decode", kv_len=9216, mnbt=1024, batch=4)
            self.assertEqual((o.batch, o.tokens), (4, 4))  # override wins
            n_ = Phase("decode", kv_len=9216)  # no budget, no batch
            self.assertEqual((n_.batch, n_.tokens), (1, 1))
            # prefill derives tokens from the same budget: one pass of it
            p = Phase("prefill", kv_len=131072, mnbt=1024)
            self.assertEqual((p.batch, p.tokens), (1, 1024))
            self.assertEqual(len(chunks_of(p)), 1)  # one pass, not chunked
            # ...but an explicit `tokens` still means a WHOLE prompt that mnbt chunks
            w = Phase("prefill", tokens=8192, kv_len=8192, mnbt=1024)
            self.assertEqual((w.batch, w.tokens), (1, 8192))
            self.assertEqual(len(chunks_of(w)), 8)
            # neither given -> refuse rather than compute zero
            with self.assertRaises(ValueError):
                Phase("prefill", kv_len=8192)
            # decode_first / decode_last share the prefix, so they share the rule
            self.assertEqual(
                Phase("decode_last", kv_len=9216, mnbt=512).batch, 512)

        def test_mnbt_is_not_the_binding_gate_at_long_context(self):
            """The reason `batch` must stay overridable. At 128K the KV pool admits
            single digits against a 1024-token budget -- deriving from mnbt and
            stopping there would model a step that cannot be scheduled."""
            c2 = replace(cfg,
                         sharding=Sharding(tp=1, dp=16, ep=True),
                         workload=Workload(isl=130048, osl=1024))
            per_seq = kv_bytes_per_token(c2) * 131072
            gate = kv_pool_bytes(c2) / per_seq
            self.assertGreater(gate, 0.0)
            self.assertLess(gate, 32.0, msg=f"{gate:.1f} req/core at 128K")

        def test_every_phase_has_positive_total(self):
            for ph in cfg.phases:
                self.assertGreater(run_phase(cfg, ph).total_ms, 0.0)

        # ---- the per-layer tables: they must be the SAME numbers, regrouped

        def test_the_layer_tables_reconcile_with_the_phase_total(self):
            """One representative layer per kind, times how many of that kind there
            are, is the phase. If this drifts, the per-layer table has become a second
            cost model and the two would be quoted against each other.
            """
            for ph in cfg.phases:
                r = run_phase(cfg, ph)
                got = sum(x["layer_total_ms"] for x in r.layer_rows
                          if x["role"] == "total")
                self.assertAlmostEqual(got / r.total_ms,
                                       1.0,
                                       places=9,
                                       msg=ph.name)

        def test_layer_kind_counts_reconcile_with_the_flat_multiplicity(self):
            """`layer_classes` and `multiplicity` are two answers to the same question
            -- how many layers run this op -- reached from opposite directions."""
            lcs = m.layer_classes()
            self.assertEqual([lc.key for lc in lcs],
                             ["dense_idx", "moe_idx", "moe"])
            self.assertEqual([lc.count for lc in lcs], [3, 18, 57])
            for cls in ("all", "dense", "moe", "indexer"):
                self.assertEqual(sum(lc.count for lc in lcs
                                     if cls in lc.op_classes),
                                 multiplicity(m, cls),
                                 msg=cls)

        def test_the_routing_data_movement_is_three_ops_totalling_4_25_passes(
                self):
            """`j-5444b22d-79a3-440c-bd5b` runs gather_fusion -> [expert GEMMs] ->
            gather -> convert_reduce_fusion, 75x/step. The unpermute is its own full
            [T*K, D] in-and-out gather; the combine's cheap [T, D] write is paid on
            top of it, not instead of a pass. A single lumped op charged 4U and hid
            the 2U that is the only optimisable part.
            """
            ops = {o.name: o for o in build_ops(cfg, pre)}
            T, K, D, b = float(pre.tokens), float(m.topk), float(m.hidden), \
                n.b("act_out")
            U = T * K * D * b
            for name, want in (("MoE: permute", T * D * b + U),
                               ("MoE: unpermute", 2 * U), ("MoE: combine",
                                                           U + T * D * b)):
                self.assertAlmostEqual(ops[name].hbm_bytes,
                                       want,
                                       places=3,
                                       msg=name)
            total = sum(ops[k].hbm_bytes
                        for k in ("MoE: permute", "MoE: unpermute",
                                  "MoE: combine"))
            self.assertAlmostEqual(total / U, 4.25, places=9)
            # and they sit either side of the expert GEMMs in dataflow order
            order = {o.name: o.order for o in build_ops(cfg, pre)}
            self.assertLess(order["MoE: permute"], order["MoE: FFi + FFg"])
            self.assertLess(order["MoE: FFo"], order["MoE: unpermute"])
            self.assertLess(order["MoE: unpermute"], order["MoE: combine"])

        def test_the_activation_total_is_a_PEAK_and_the_others_are_sums(self):
            """Activations are transient -- one op's working set at a time -- so a
            layer's activation occupancy is the largest single row, not the sum.
            Weights and KV are resident together and do add. Summing activations
            reported 1,117 MiB on GLM-5.2's dense decode layer against a true peak
            of 262, and the layer-sums table had it right while the op table did not.
            """
            for ph in cfg.phases:
                for lc in m.layer_classes():
                    rows = [
                        x for x in layer_rows(cfg, ph)
                        if x["layer_kind"] == lc.key
                    ]
                    tot = next(x for x in rows if x["role"] == "total")
                    cnt = [x for x in rows if x["role"] in ("op", "fused")]
                    self.assertAlmostEqual(
                        tot["act_mib"],
                        max(x["act_mib"] for x in cnt),
                        places=6,
                        msg=f"{ph.name}/{lc.key} act is not a peak")
                    for col in ("weight_mib", "kv_mib", "roofline_ms"):
                        self.assertAlmostEqual(tot[col],
                                               sum(x[col] for x in cnt),
                                               places=6,
                                               msg=f"{ph.name}/{lc.key}/{col}")
                    # a fused parent still SUMS its children -- they coexist in-kernel
                    for p in [x for x in rows if x["role"] == "fused"]:
                        kids = [
                            x for x in rows if x["parent"] == p["operator"]
                        ]
                        self.assertAlmostEqual(p["act_mib"],
                                               sum(k["act_mib"] for k in kids),
                                               places=6,
                                               msg=p["operator"])

        def test_the_op_table_and_layer_sums_agree_on_the_activation_peak(
                self):
            """Two tables, one number. They disagreed once already."""
            for ph in cfg.phases:
                occ = layer_occupancy(cfg, ph)
                for lc in m.layer_classes():
                    tot = next(
                        x for x in layer_rows(cfg, ph)
                        if x["layer_kind"] == lc.key and x["role"] == "total")
                    self.assertAlmostEqual(occ[lc.key].act_peak_local / MiB,
                                           tot["act_mib"],
                                           places=6,
                                           msg=f"{ph.name}/{lc.key}")

        def test_a_fused_parent_is_the_sum_of_its_children_and_they_are_not_double_counted(
                self):
            """`sub` rows are shown under their parent and must not also be summed;
            a consumer that forgets skips nothing visible but doubles the kernel."""
            rows = run_phase(cfg, pre).layer_rows
            kinds = {x["layer_kind"] for x in rows}
            for kind in kinds:
                block = [x for x in rows if x["layer_kind"] == kind]
                for p in [x for x in block if x["role"] == "fused"]:
                    kids = [x for x in block if x["parent"] == p["operator"]]
                    self.assertGreater(len(kids), 1, msg=p["operator"])
                    for col in ("memory_ms", "compute_ms", "roofline_ms",
                                "kv_mib"):
                        self.assertAlmostEqual(p[col],
                                               sum(k[col] for k in kids),
                                               places=9,
                                               msg=f"{p['operator']}.{col}")
                tot = next(x for x in block if x["role"] == "total")
                counted = [x for x in block if x["role"] in ("op", "fused")]
                self.assertAlmostEqual(tot["roofline_ms"],
                                       sum(x["roofline_ms"] for x in counted),
                                       places=9,
                                       msg=kind)

        def test_memory_time_comes_from_TRAFFIC_not_from_the_occupancy_column(
                self):
            """`Op.kv_bytes` is what the op READS; `kv_resident_bytes` is the cache it
            attends over. Only the first may reach `mem_ms`.

            Must be checked on a phase where they DIFFER. At prefill the union of the
            queries' selections is the whole context, so traffic == residency and any
            confusion of the two passes silently -- which is how the op table came to
            print traffic under a column the layer sums filled with residency, 64x
            apart, both labelled KV.
            """
            dec = Phase("decode", kv_len=131072, batch=1024)
            qk = {o.name: o for o in build_ops(cfg, dec)}["MLA: QK Nope"]
            self.assertLess(qk.kv_bytes * 60,
                            qk.kv_resident_bytes)  # genuinely apart
            for ph in (pre, dec):
                for o in build_ops(cfg, ph):
                    want = (o.w_bytes + o.act_bytes +
                            o.kv_bytes) / hw.hbm_bw * 1e3
                    self.assertAlmostEqual(o.mem_ms(hw),
                                           want,
                                           places=12,
                                           msg=o.name)

        def test_the_kv_occupancy_column_is_the_resident_cache(self):
            """What a reader means by "Memory Occupancy - KV": the cache the op
            attends over, `B * S * width`, not the sparse slice it happens to fetch.
            The op table and the layer-sums table must agree on that."""
            dec = Phase("decode", kv_len=130048, batch=1024)
            B, S = dec.batch, dec.kv_len
            rs = {
                x["operator"]: x
                for x in layer_rows(cfg, dec) if x["layer_kind"] == "moe"
            }
            for op_name, width in (("MLA: QK Nope", m.kv_lora_rank),
                                   ("MLA: QK Rope", m.qk_rope_head_dim)):
                want = B * S * width * n.b("kv_cache") / MiB
                self.assertAlmostEqual(rs[op_name]["kv_mib"],
                                       want,
                                       places=3,
                                       msg=op_name)
            self.assertEqual(rs["MLA: AV"]["kv_mib"], 0.0)  # QK Nope holds it
            # the flash group's residency == the layer's whole MLA cache
            self.assertAlmostEqual(rs["MLA: Flash Attention QK+AV"]["kv_mib"],
                                   B * S *
                                   (m.kv_lora_rank + m.qk_rope_head_dim) *
                                   n.b("kv_cache") / MiB,
                                   places=3)

        def test_the_ici_split_sums_to_what_collective_py_priced(self):
            """Splitting latency from bandwidth must not change the total. `t_fixed`
            goes to latency -- it is the term a smaller payload does not shrink."""
            mach = cfg.hardware.machine()
            # pin the FLAT schedule here: this test is about the lat/bw split summing
            # to `collective.py`'s total, not about which schedule the config picks.
            # `test_the_three_dispatch_strategies_differ_as_ep_s4c_says` owns that.
            c = Ctx(
                replace(cfg,
                        model=replace(m, moe_dispatch_strategy="all_gather")),
                pre)
            for fn, est in ((coll_ep_dispatch,
                             cc.all_gather(mach,
                                           elems_full=c.T * c.s.ep_group() *
                                           c.D,
                                           dtype=n.dt("expert_dispatch"))),
                            (coll_ep_collect,
                             cc.reduce_scatter(mach,
                                               elems=c.T * c.s.ep_group() *
                                               c.D,
                                               dtype=n.dt("expert_collect")))):
                op = fn(c, mach)
                self.assertAlmostEqual(op.ici_ms, est.total * 1e3, places=12)
                self.assertGreater(op.ici_bw_ms, 0.0)
                self.assertGreater(op.ici_lat_ms, 0.0)

        def test_every_matmul_pads_its_contracting_dim_to_the_mxu_minimum(
                self):
            """buglist B6. `REQUIREMENTs.md` s MXU requires it; `dsa_weighted_sum`
            was the one raw `Op` that skipped it, under-counting 8x because H_idx is
            32. Checks the rule everywhere rather than that one op, so the next raw
            `Op` cannot reintroduce it."""

            def P(k):
                return max(float(k), float(hw.mxu_min_dim))

            T, S = float(pre.tokens), float(pre.kv_len)
            ops2 = {o.name: o for o in build_ops(cfg, pre)}
            want = {
                "DSA Indexer: QK":
                2 * T * m.indexer_heads * P(m.indexer_head_dim) * S,
                "DSA Indexer: Weighted Sum":
                2 * T * S * P(m.indexer_heads),
                "MLA: QK Nope":
                2 * T * m.heads * P(m.kv_lora_rank) * m.indexer_topk,
                "MLA: QK Rope":
                2 * T * m.heads * P(m.qk_rope_head_dim) * m.indexer_topk,
                "MLA: AV":
                2 * T * m.heads * P(m.indexer_topk) * m.kv_lora_rank,
            }
            for k, v in want.items():
                self.assertAlmostEqual(ops2[k].flops / v, 1.0, places=9, msg=k)
            # and the two that actually pay for it are padded, not coincidences
            self.assertAlmostEqual(ops2["DSA Indexer: Weighted Sum"].flops /
                                   (2 * T * S * m.indexer_heads),
                                   8.0,
                                   places=9)
            self.assertAlmostEqual(
                ops2["MLA: QK Rope"].flops /
                (2 * T * m.heads * m.qk_rope_head_dim * m.indexer_topk),
                4.0,
                places=9)

        def test_the_indexer_head_weights_proj_is_w16a16_not_w8a8(self):
            """The reference sheet quantizes the indexer's K and Q projections to w8a8
            and leaves the head-weights projection at w16a16. It is a small op -- 0.004
            GiB/core and ~0.0002 ms/layer -- but before `indexer_w_proj` existed the
            config could not say so, and it was silently priced at w8a8.
            """
            ops2 = {o.name: o for o in build_ops(cfg, pre)}
            hwp, kp = ops2["DSA Indexer: Head Weights"], ops2[
                "DSA Indexer: K Proj"]
            self.assertEqual((hwp.dtype, hwp.dtype_b), ("bf16", "bf16"))
            self.assertEqual((kp.dtype, kp.dtype_b), ("fp8", "fp8"))
            # so it runs at the bf16 MXU rate, not fp8 -- the point of pairing the sites
            self.assertAlmostEqual(hwp.comp_ms(hw) * hw.peak_bf16 / hwp.flops,
                                   1e3,
                                   places=6)

        def test_weight_bytes_agrees_with_the_op_table_on_the_indexer(self):
            """`weight_bytes` and `dsa_head_weights` read the same parameters at the
            same dtype. They are two expressions of one fact and must not drift."""
            s = Sharding(tp=16, dp=1)
            c2 = replace(cfg, sharding=s)
            ops2 = {o.name: o for o in compute_ops(c2, pre)}
            got = sum(ops2[k].w_bytes
                      for k in ("DSA Indexer: K Proj", "DSA Indexer: Q Proj",
                                "DSA Indexer: Head Weights"))
            want = ((m.q_lora_rank * m.indexer_heads * m.indexer_head_dim +
                     m.hidden * m.indexer_head_dim) * n.b("weights") +
                    m.hidden * m.indexer_heads * n.b("indexer_w_proj"))
            # all three are replicated, in the op table and in the aggregate alike
            self.assertAlmostEqual(got / want, 1.0, places=9)

        def test_the_indexer_reads_every_sequences_cache_not_just_one(self):
            """`dsa_qk` scans all S keys for each of the B sequences. Missing the
            batch factor made decode's dominant op look like a rounding error."""
            for B in (1, 8, 1024):
                ph = Phase("decode", kv_len=131072, batch=B)
                op = {o.name: o for o in build_ops(cfg, ph)}["DSA Indexer: QK"]
                want = B * 131072 * m.indexer_head_dim * n.b("indexer_cache")
                self.assertAlmostEqual(op.kv_bytes / want,
                                       1.0,
                                       places=9,
                                       msg=f"B={B}")
            # prefill is the degenerate case: T queries, ONE shared cache
            self.assertAlmostEqual(
                {o.name: o
                 for o in build_ops(cfg, pre)}["DSA Indexer: QK"].kv_bytes,
                pre.kv_len * m.indexer_head_dim * n.b("indexer_cache"),
                places=3)

        def test_no_op_group_reads_more_kv_than_the_cache_holds(self):
            """The physical bound, and the one that catches a double-read.

            A layer's attention cannot fetch more KV bytes than that layer's cache
            contains for the tokens it touches. `mla_av` re-read `c_kv` after
            `mla_qk_nope` had already read it -- 136 MiB against a 72 MiB cache,
            1.89x -- and nothing failed, because every per-op formula was defensible
            on its own. Only the aggregate is impossible.
            """
            for ph in cfg.phases:
                fetched = kv_fetched(m, ph)
                ops2 = build_ops(cfg, ph)
                mla = sum(o.kv_bytes for o in ops2
                          if o.group == MLA_FLASH and "KV Down" not in o.name)
                cap = fetched * (m.kv_lora_rank +
                                 m.qk_rope_head_dim) * n.b("kv_cache")
                self.assertLessEqual(mla,
                                     cap * 1.000001,
                                     msg=f"{ph.name}: MLA flash")
                idx = sum(o.kv_bytes for o in ops2 if o.group == DSA_FLASH)
                icap = (max(1, ph.batch) * ph.kv_len * m.indexer_head_dim *
                        n.b("indexer_cache"))
                self.assertLessEqual(idx,
                                     icap * 1.000001,
                                     msg=f"{ph.name}: indexer")

        def test_mla_av_does_not_refetch_the_latent_but_gqa_av_does(self):
            """Weight absorption makes c_kv both K and V; GQA's K and V are distinct.
            The asymmetry is deliberate -- pinned so neither is 'fixed' to match."""
            pre_ops = {o.name: o for o in build_ops(cfg, pre)}
            self.assertEqual(pre_ops["MLA: AV"].kv_bytes, 0.0)
            self.assertGreater(pre_ops["MLA: QK Nope"].kv_bytes, 0.0)
            gqa = replace(m,
                          kv_lora_rank=0,
                          q_lora_rank=0,
                          indexer_layers=0,
                          kv_heads=8,
                          head_dim=128,
                          qk_nope_head_dim=0,
                          qk_rope_head_dim=0,
                          v_head_dim=0,
                          dense_layers=m.layers,
                          indexer_topk=0)
            g = {o.name: o for o in build_ops(replace(cfg, model=gqa), pre)}
            self.assertGreater(g["Attn: AV"].kv_bytes, 0.0)

        def test_the_three_dispatch_strategies_differ_as_ep_s4c_says(self):
            """A ships `topk`x the wire traffic; C's phase order is load-bearing."""
            ep = R_cfg = replace(cfg, sharding=Sharding(tp=1, dp=16, ep=True))
            got = {}
            for st in MOE_DISPATCH:
                c2 = replace(ep, model=replace(m, moe_dispatch_strategy=st))
                got[st] = next(
                    o for o in build_ops(c2, replace(pre, tokens=1024))
                    if o.name.startswith("MoE: Dispatch")).ici_ms
            # A moves the EXPANDED set, so it is far the most expensive despite
            # all-to-all's smaller per-device payload -- its phases do not shrink.
            self.assertGreater(got["all_to_all"], 4.0 * got["all_gather"])
            # B and C carry the same bytes, so they land within ~10% of each other.
            self.assertLess(
                abs(got["hierarchical_ag"] / got["all_gather"] - 1), 0.10)

        def test_hierarchical_dispatch_puts_the_fast_link_on_the_last_phase(
                self):
            """An all-gather's payload GROWS, so d2d belongs last. Reversing it was
            1.62x worse, and nothing about the code shape makes that obvious."""
            mach = cfg.hardware.machine()
            d2d, ici = _chip_axes(mach)
            self.assertTrue(d2d and ici,
                            "need both link classes to test the order")
            T, D, dt = 1024.0, float(m.hidden), n.dt("expert_dispatch")
            n_ici, ep = _axes_size(mach, ici), 16
            right = (
                cc.all_gather(
                    mach, elems_full=T * n_ici * D, dtype=dt, axes=ici).total +
                cc.all_gather(mach, elems_full=T * ep * D, dtype=dt,
                              axes=d2d).total)
            wrong = (
                cc.all_gather(mach, elems_full=T * 2 * D, dtype=dt,
                              axes=d2d).total +
                cc.all_gather(mach, elems_full=T * ep * D, dtype=dt,
                              axes=ici).total)
            self.assertLess(right, wrong)
            self.assertGreater(wrong / right, 1.5)

        def test_hbm_bytes_is_exactly_its_three_parts(self):
            for op in build_ops(cfg, pre):
                self.assertAlmostEqual(op.hbm_bytes,
                                       op.w_bytes + op.act_bytes + op.kv_bytes,
                                       places=9,
                                       msg=op.name)
            # KV traffic lands on the attention ops that READ the cache, plus the
            # projections that WRITE it -- but a writer counts only when it writes at
            # the cache's OWN dtype. Wider means it emitted a staging buffer, which
            # is an activation. `MLA: AV` is absent because `mla_qk_nope` already
            # fetched `c_kv`; `DSA Indexer: Weighted Sum` because `dsa_qk` already
            # fetched the indexer cache. See the two tests above.
            kv = {o.name for o in build_ops(cfg, pre) if o.kv_bytes > 0}
            readers = {"MLA: QK Nope", "MLA: QK Rope", "DSA Indexer: QK"}
            self.assertEqual(kv & readers, readers)
            self.assertNotIn("MLA: AV", kv)
            self.assertNotIn("DSA Indexer: Weighted Sum", kv)
            writers = (("MLA: KV Down Proj", "kv_write", "kv_cache"),
                       ("DSA Indexer: K Proj", "indexer_write",
                        "indexer_cache"))
            for op_name, w, cache in writers:
                self.assertEqual(
                    op_name in kv,
                    n.dt(w) == n.dt(cache),
                    msg=f"{op_name}: {n.dt(w)} vs cache {n.dt(cache)}")
            # narrowing the write site to the cache dtype must flip it back to KV
            for op_name, w, cache in writers:
                c2 = replace(cfg,
                             numerics=Numerics(n.default, {
                                 **n.sites, w: n.dt(cache)
                             }))
                op = {o.name: o for o in build_ops(c2, pre)}[op_name]
                self.assertGreater(op.kv_bytes, 0.0, msg=op_name)
                self.assertNotIn("STAGING", op.comment, msg=op_name)

        # ---- routed-expert weights are sharded in BOTH regimes

        def test_non_ep_moe_weights_are_tp_sharded(self):
            """Without EP each device holds every expert, TP-sharded -- not whole.

            `_moe_dims` has always said so on the op side; `weight_bytes` divided by
            `ep_group()`, which is 1 when EP is off, and reported 676 GiB/core against
            43. That drove `kv_pool_bytes` NEGATIVE, so a sharding sweep silently
            dropped every non-EP row before ranking it.
            """
            a = weight_bytes(replace(cfg, sharding=Sharding(tp=16, dp=1)))
            b = weight_bytes(
                replace(cfg, sharding=Sharding(tp=16, dp=1, ep=True)))
            # at tp16-dp1 the EP group IS the TP group, so the two must agree exactly
            self.assertAlmostEqual(a / b, 1.0, places=9)
            self.assertLess(a / GiB, 96.0)
            self.assertGreater(
                kv_pool_bytes(replace(cfg, sharding=Sharding(tp=16, dp=1))),
                0.0)

        def test_weight_bytes_agrees_with_the_op_table_on_the_moe_terms(self):
            """The aggregate and the per-op weight reads are the same parameters.

            They reconcile exactly on the MoE and dense terms -- which is what the
            `expert_div` fix above is about -- and disagree by a known, fully
            attributed 1.27 GiB on three attention projections. See the next test.
            """
            s = Sharding(tp=16, dp=1)
            c2 = replace(cfg, sharding=s)
            per = {
                "all": m.layers,
                "dense": m.dense_layers,
                "moe": m.moe_layers,
                "indexer": m.indexer_layers
            }
            ops2 = {o.name: o for o in compute_ops(c2, pre)}
            moe_ops = sum(ops2[k].w_bytes * per[ops2[k].layer_class]
                          for k in ("MoE: FFi + FFg", "MoE: FFo",
                                    "MoE: Router (gating)",
                                    "MoE: Shared FFi + FFg", "MoE: Shared FFo",
                                    "Dense: FFi + FFg", "Dense: FFo"))
            D, tp, wb, mb = m.hidden, s.tp, n.b("weights"), n.b("moe_weights")
            moe_agg = (
                3 * D * m.dense_ffn * m.dense_layers / tp * wb +
                3 * D * m.expert_ffn * m.n_experts * m.moe_layers / tp * mb +
                3 * D * m.shared_ffn * m.n_shared_experts * m.moe_layers / tp *
                wb + D * m.n_experts * m.moe_layers * wb)
            self.assertAlmostEqual(moe_ops / moe_agg, 1.0, places=9)

        def test_weight_bytes_agrees_with_the_op_table_on_every_weight(self):
            """The aggregate and the op table are two expressions of one fact.

            They used to disagree on three projections -- MLA's `q_a`/`kv_a` and the
            DSA indexer's -- which the op table replicated and `weight_bytes` divided
            by TP. vLLM replicates all three (`deepseek_v2.py:912-927, 669-683`), so
            the aggregate was wrong: 1.27 GiB/core at tp16. Checked at several TP
            degrees because a `/tp` on a replicated term is invisible at tp1.
            """
            per = {
                "all": m.layers,
                "dense": m.dense_layers,
                "moe": m.moe_layers,
                "indexer": m.indexer_layers
            }
            for tp in (1, 4, 16):
                s = Sharding(tp=tp, dp=16 // tp, ep=True)
                c2 = replace(cfg, sharding=s)
                tot = sum(o.w_bytes * per[o.layer_class]
                          for o in compute_ops(c2, pre)
                          if o.layer_class in per)
                emb = m.vocab * m.hidden * 2 / s.tp * n.b("weights")
                self.assertAlmostEqual((tot + emb) / weight_bytes(c2),
                                       1.0,
                                       places=9,
                                       msg=f"tp{tp}")

        def test_dcp_rebalances_the_attention_core_without_changing_its_flops(
                self):
            """DCP runs dcp x the heads over 1/dcp of the keys: FLOPs equal, KV read
            and KV resident down by dcp. parallelism/dcp/dcp.md s4."""
            base = {
                o.name: o
                for o in build_ops(replace(cfg, sharding=Sharding(tp=16)), pre)
            }
            dcp = {
                o.name: o
                for o in build_ops(
                    replace(cfg, sharding=Sharding(tp=16, dcp=8)), pre)
            }
            for name in ("MLA: QK Nope", "MLA: QK Rope", "MLA: AV"):
                self.assertAlmostEqual(dcp[name].flops / base[name].flops,
                                       1.0,
                                       places=9)
            for name in ("MLA: QK Nope", "MLA: QK Rope"):
                self.assertAlmostEqual(base[name].kv_bytes /
                                       dcp[name].kv_bytes,
                                       8.0,
                                       places=9)
            self.assertAlmostEqual(base["DSA Indexer: QK"].kv_bytes /
                                   dcp["DSA Indexer: QK"].kv_bytes,
                                   8.0,
                                   places=9)
            # projections never see DCP
            self.assertEqual(dcp["MLA: O Proj"].flops,
                             base["MLA: O Proj"].flops)

        def test_sparse_mla_dcp_issues_the_merge_on_every_layer(self):
            """Query gather and LSE merge fire on all L layers; the candidate
            exchange only on the indexer's. The merge is NOT absorbed by o_proj, so
            the TP reduction is still there. parallelism/dcp/dcp.md s7."""
            c2 = replace(cfg, sharding=Sharding(tp=16, dcp=8))
            by = {o.name: o for o in collectives(c2, pre)}
            self.assertEqual(by["DCP: query head gather"].layer_class, "all")
            self.assertEqual(by["DCP: LSE merge"].layer_class, "all")
            self.assertEqual(by["DCP: indexer candidate gather"].layer_class,
                             "indexer")
            self.assertIn("TP: All-Reduce (attn exit)", by)
            none = {
                o.name
                for o in collectives(replace(cfg, sharding=Sharding(
                    tp=16)), pre)
            }
            self.assertFalse(any(k.startswith("DCP:") for k in none))

        def test_single_ag_halves_the_indexer_exchange(self):
            """#1230: one gather of [T, K+1] against two of [T, K]. The bandwidth
            term halves, less 1/K; the latency term drops a whole collective."""

            def gather(strategy):
                c2 = replace(cfg,
                             sharding=Sharding(tp=16, dcp=8),
                             model=replace(m, dcp_indexer_gather=strategy))
                return {
                    o.name: o
                    for o in collectives(c2, pre)
                }["DCP: indexer candidate gather"]

            two, one = gather("two_ag"), gather("single_ag")
            k = m.indexer_topk
            self.assertAlmostEqual(one.ici_bw_ms / two.ici_bw_ms,
                                   (k + 1) / (2 * k),
                                   places=6)
            self.assertLess(one.ici_lat_ms, two.ici_lat_ms)

        def test_gqa_dcp_merge_gathers_dcp_squared_partials(self):
            """Each rank's partial already has H*dcp heads and the gather stacks dcp
            of them. parallelism/dcp/dcp.md s6."""
            g = replace(m,
                        kv_lora_rank=0,
                        q_lora_rank=0,
                        qk_nope_head_dim=0,
                        qk_rope_head_dim=0,
                        v_head_dim=0,
                        kv_heads=2,
                        head_dim=128,
                        indexer_layers=0,
                        indexer_topk=0,
                        indexer_heads=0,
                        indexer_head_dim=0,
                        indexer_cache_bytes_per_token=0.0)

            def merge(dcp):
                c2 = replace(cfg, model=g, sharding=Sharding(tp=16, dcp=dcp))
                return {
                    o.name: o
                    for o in collectives(c2, pre)
                }["DCP: LSE merge"]

            # the partial payload goes as dcp^2; a big-payload ring is ~bandwidth,
            # so doubling dcp should roughly quadruple the partial bytes
            self.assertIsNone(Sharding(tp=16, dcp=8).check(hw, g))
            self.assertGreater(merge(8).ici_bw_ms / merge(4).ici_bw_ms, 3.0)

        def test_pcp_divides_tokens_not_weights(self):
            """PCP ranks replicate TP's shards and split the step's tokens
            (parallelism/pcp s3). So tp8-pcp2-ep holds exactly tp8-dp2-ep's weights --
            dense /8, experts /16 either way -- and every per-token op halves."""
            pcp = replace(cfg, sharding=Sharding(tp=8, pcp=2, ep=True))
            dp = replace(cfg, sharding=Sharding(tp=8, dp=2, ep=True))
            self.assertAlmostEqual(weight_bytes(pcp) / weight_bytes(dp),
                                   1.0,
                                   places=12)
            one = {
                o.name: o
                for o in build_ops(
                    replace(cfg, sharding=Sharding(tp=8, ep=True)), pre)
            }
            two = {o.name: o for o in build_ops(pcp, pre)}
            for name in ("MLA: O Proj", "MLA: Query Up Proj",
                         "MoE: FFi + FFg"):
                self.assertAlmostEqual(one[name].flops / two[name].flops,
                                       2.0,
                                       places=9,
                                       msg=name)
            # heads do NOT divide by PCP
            base = replace(cfg, sharding=Sharding(tp=8, ep=True))
            self.assertEqual(Ctx(pcp, pre).H, Ctx(base, pre).H)

        def test_pcp_ring_stripes_kv_and_allgather_replicates_it(self):
            """The ring buys capacity; the replicated cache buys none. pcp s5."""
            ring = replace(cfg, sharding=Sharding(tp=8, pcp=2, ep=True))
            ag = replace(ring, model=replace(m, pcp_kv_strategy="allgather"))
            base = replace(cfg, sharding=Sharding(tp=8, ep=True))
            self.assertAlmostEqual(kv_bytes_per_token(base) /
                                   kv_bytes_per_token(ring),
                                   2.0,
                                   places=9)
            self.assertAlmostEqual(kv_bytes_per_token(ag),
                                   kv_bytes_per_token(base),
                                   places=6)

        def test_pcp_ring_scales_with_context_allgather_with_tokens(self):
            """At decode the ring streams the whole cache every layer; the all-gather
            moves one new row per request. pcp s4 vs s6."""
            dec = Phase("decode", kv_len=131072, batch=64)
            ring = replace(cfg, sharding=Sharding(tp=8, pcp=2, ep=True))
            ag = replace(ring, model=replace(m, pcp_kv_strategy="allgather"))
            r = {o.name: o for o in collectives(ring, dec)}["PCP: KV ring"]
            a = {o.name: o for o in collectives(ag, dec)}["PCP: KV all-gather"]
            self.assertGreater(r.ici_bw_ms, 100 * a.ici_bw_ms)
            self.assertIn("PCP: indexer K ring",
                          {o.name
                           for o in collectives(ring, dec)})

        def test_pcp_legality_follows_main(self):
            """pcp_validation.py on vllm-torchtpu fde6c3994; MLA has no PCP path."""
            self.assertIn("requires --enable-expert-parallel",
                          Sharding(tp=8, pcp=2).check(hw, m) or "")
            self.assertIn("MLA has no PCP path",
                          Sharding(tp=8, pcp=2, ep=True).check(hw, m) or "")
            self.assertIn("PP", Sharding(tp=4, pcp=2, pp=2).check(hw) or "")
            g = replace(m,
                        kv_lora_rank=0,
                        kv_heads=8,
                        head_dim=128,
                        n_experts=0,
                        indexer_layers=0,
                        indexer_topk=0)
            self.assertIsNone(Sharding(tp=8, pcp=2).check(hw, g))
            self.assertIn(
                "head_dim 64",
                Sharding(tp=8, pcp=2).check(hw, replace(g, head_dim=64)) or "")

        def test_pp_layers_is_the_partition_knob_and_is_checked(self):
            """`Sharding.pp_layers` = VLLM_PP_LAYER_PARTITION. parallelism/pp s3."""
            part = (6, 4, 4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 4, 8, 4
                    )  # j-78adee7b
            s16 = Sharding(pp=16, pp_layers=list(part))
            self.assertEqual(s16.pp_layers, part)  # list -> tuple
            self.assertIsNone(s16.check(hw, m))
            self.assertEqual(stage_layers(replace(cfg, sharding=s16)),
                             list(part))
            self.assertEqual(
                stage_layers(replace(cfg, sharding=Sharding(pp=16))),
                [4] + [5] * 14 + [4])
            self.assertIn(
                "sums to",
                Sharding(pp=16, pp_layers=(5, ) * 16).check(hw, m) or "")
            self.assertIn("entries",
                          Sharding(pp=16, pp_layers=(78, )).check(hw, m) or "")
            self.assertIn("tensor parallelism inside a stage",
                          Sharding(tp=2, pp=8).check(hw) or "")

        def test_glm_indexer_layers_follow_vllms_rule(self):
            """deepseek_v2.py:1104-1105, freq 4 offset 3 -> 0,1,2,6,...,74."""
            ids = m.indexer_layer_ids()
            self.assertEqual(ids[:5], (0, 1, 2, 6, 10))
            self.assertEqual(ids[-1], 74)
            self.assertEqual(len(ids), m.indexer_layers)

        def test_pp_hbm_is_the_heaviest_stage(self):
            """parallelism/pp s5: the stages sum to the unpipelined model, and the
            one that has to fit is the heaviest, not the mean."""
            part = (6, 4, 4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 4, 8, 4)
            c16 = replace(cfg, sharding=Sharding(pp=16, pp_layers=part))
            st = stage_weight_bytes(c16)
            flat = _weight_bytes_flat(replace(cfg, sharding=Sharding()))
            self.assertAlmostEqual(sum(st) / flat, 1.0, places=9)
            self.assertEqual(weight_bytes(c16), max(st))
            self.assertGreater(weight_bytes(c16), 1.5 * flat / 16)
            kv1 = kv_bytes_per_token(replace(cfg, sharding=Sharding()))
            self.assertGreater(kv_bytes_per_token(c16), 1.5 * kv1 / 16)

        def test_the_partition_search_never_loses_to_the_default(self):
            """best_pp_partition is the exact min-max split over the real per-layer
            costs, so its slowest stage is at most the default's."""
            dec = Phase("decode", kv_len=9216, batch=64)
            for k, cap in ((2, float("inf")), (4, float("inf")), (16, None)):
                c2 = replace(cfg, sharding=Sharding(pp=k))
                best = best_pp_partition(c2, dec, max_stage_bytes=cap)
                self.assertEqual(len(best), k)
                self.assertEqual(sum(best), m.layers)
                ops = build_ops(c2, dec)
                self.assertLessEqual(max(stage_costs(c2, ops, list(best))),
                                     max(stage_costs(c2, ops)) + 1e-12)
            # at pp4, tp1, no EP a quarter of GLM-5.2 does not fit a core: say so
            with self.assertRaises(ValueError):
                best_pp_partition(replace(cfg, sharding=Sharding(pp=4)), dec)

        def test_dcp_legality_follows_vllm_and_the_tpu_paths(self):
            """vllm/config/model.py:1335-1362 for GQA; dense MLA has no TPU path."""
            g = replace(m,
                        kv_lora_rank=0,
                        kv_heads=8,
                        heads=64,
                        indexer_layers=0,
                        indexer_topk=0)
            self.assertIn("tp 8 > H_kv",
                          Sharding(tp=8, dp=2, dcp=2).check(hw, g) or "")
            self.assertIn("dcp 4 <= tp // H_kv",
                          Sharding(tp=16, dcp=4).check(hw, g) or "")
            self.assertIsNone(Sharding(tp=16, dcp=2).check(hw, g))
            dense_mla = replace(m, indexer_layers=0, indexer_topk=0)
            self.assertIn("dense MLA",
                          Sharding(tp=16, dcp=2).check(hw, dense_mla) or "")
            self.assertIsNone(Sharding(tp=16, dcp=16).check(hw, m))
            self.assertIn("PCP",
                          Sharding(tp=8, pcp=2, dcp=2).check(hw, m) or "")

    r = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestLoader().loadTestsFromTestCase(T))
    return 0 if r.wasSuccessful() else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--phase", help="only this phase")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--csv", help="the flat per-phase op table")
    ap.add_argument("--layer-csv", help="the per-layer op tables, long form")
    ap.add_argument("--no-layers",
                    action="store_true",
                    help="skip the per-layer op tables")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.config:
        ap.error("need a config path, or --selftest")
    cfg = load(a.config)
    phases = [p for p in cfg.phases if not a.phase or p.name == a.phase]
    res = [run_phase(cfg, p) for p in phases]
    report(cfg, res, a.top)
    if not a.no_layers:
        report_layers(cfg, res)
    sv = serving(cfg, res)
    if sv:
        report_serving(cfg, sv)
    elif cfg.workload is None:
        print("\n  (no `workload` block -> no TTFT/throughput; see README s5)")
    if a.csv:
        write_csv(a.csv, res)
    if a.layer_csv:
        write_layer_csv(a.layer_csv, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
