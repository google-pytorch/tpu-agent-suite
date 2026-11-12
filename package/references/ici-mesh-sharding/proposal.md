---
name: ici-mesh-sharding-proposal
description: Pairwise factor moves, analytical communication-volume scaling, portfolio ranking, torus topology fit, OSS/serving rules, and AOT compile verification.
scripts:
- scripts/ici-mesh-sharding/mesh_search.py
- scripts/ici-mesh-sharding/mesh_portfolio.py
- scripts/ici-mesh-sharding/mesh_topology.py
- scripts/ici-mesh-sharding/tpu_specs.py
- scripts/ici-mesh-sharding/sharding_adapter.py
- scripts/ici-mesh-sharding/oss_sharding_rules.py
- scripts/ici-mesh-sharding/compile_verdict.py
---

# ICI Mesh Sharding — Candidate Enumeration, Portfolio Ranking & Compile Verification

This reference documents the candidate generation, analytical cost model, portfolio template, topology checks, OSS serving rules, and AOT compile verdicts implemented in `ici-mesh-sharding`. Names in `code` are the keys and strings the scripts emit.

## 1. Pairwise Power-of-Two Factor Moves

{{script:ici-mesh-sharding/mesh_search.py}} (`cli.py enumerate`) lists every mesh one pairwise move away from the baseline (`--max-moves` allows up to `MAX_MOVES_LIMIT = 3`). A move divides a donor axis by a power of two `k` and multiplies a receiver axis by `k` (`donor/k -> receiver xk`), so the device count is unchanged.

- **Bracket points** (`bracketed_pairs`): each (donor, receiver) pair at its first step (normally `k = 2`), at its boundary (the largest legal `k`; the donor collapses to 1 when legal), and, for sibling axes such as `data` and `data2`, at the size swap.
- **Legality**: a changed axis must stay a power of two; `--batch-size` must be divisible by the batch shards (`replica`, `data`, `data2`, `fsdp`, `dp`), `--sequence-length` by the sequence shards (`seq`, `seq1` to `seq4`, `sp`, `pcp`, `dcp`, `sequence`, `context`), and `--num-heads` by the query-head shards (the tensor axes except `model2`).
- **KV heads** only add warnings: replication when the KV-head shards exceed `--num-kv-heads`, and strict SplashAttention validators when they do not divide it. `rank-portfolio` leaves out the meshes whose KV-head shards do not divide `--num-kv-heads` and says how many in its notes.
- `--fixed-axes` defaults to `intrachip`, `replica`, `stage` and `pp`, which keep their size.

## 2. Analytical Per-Device Communication-Volume Model

{{script:ici-mesh-sharding/mesh_portfolio.py}} estimates each candidate's step-time change without hardware. It scales every baseline collective group's share of the step by the change in its per-device communication volume:

| Collective group | Volume scales with |
| --- | --- |
| Tensor-parallel activation collectives | local tokens x `(t - 1) / t` over the tensor shards `t` |
| KV gathers and ring attention | per-device batch x KV heads per device x `(s - 1) / s` over the group |
| MoE routing (all-to-all) | local tokens x `(g - 1) / g` x the group's mean hop count on the torus |
| Other activation collectives | local tokens x `(g - 1) / g` |
| FSDP weight gathers and gradient reductions | `1 / (tensor x expert shards)` x `(g - 1) / g` |
| Cross-slice (DCN) reductions | `(r - 1) / r` over the slices `r`, divided by the growth of the per-slice mesh |

Each is multiplied by `_SHARED_DIM_PENALTY = 2.0` when the group's axes do not fill whole torus dimensions on their own. The mean hop count takes `L / 4` per dimension of length `L`, the mean on a ring with wrap-around links, whether or not the slice has them (section 4). The factor of 2 and the hop counts are model assumptions, not measurements. Global reductions, host transfers and compute are held constant. The estimates rank candidates; they do not predict the new step time.

## 3. The Portfolio Template

`rank-portfolio` (`build_portfolio`) first scores the hypotheses the breakdown supports (`hypotheses`): `routing`, `tensor`, `sequence`, `weights`, `batch` and `mixed`, each kept at 2% of the step or more (`MIN_HYPOTHESIS_PCT = 2.0`), plus `layout`, `slices` and `pipeline` when the evidence or the flags allow them. It then fills up to `--num-proposals` slots (default `DEFAULT_NUM_PROPOSALS = 7`) in this order, skipping meshes already taken (`_fill`):

1. `strongest move, first step` and `strongest move, boundary`: the strongest hypothesis' best (donor, receiver) pair at `k = 2` and at the largest `k` that passes the memory check.
2. `same bottleneck, another axis, first step` and `same bottleneck, another axis, boundary`: the same bottleneck relieved through another donor or receiver, for example the other axis of an MoE routing group.
3. `second hypothesis, first step`.
4. One placement proposal, the first of `pipeline depth, first step`, `fewer, larger slices`, `sibling re-factor` and `re-shaped slice` that the evidence or the flags support.
5. `compound remedy` (one receiver fed by two donors, one donor split across two receivers, or a move plus a sibling fold), `second hypothesis, compound remedy` (when that hypothesis has 15% of the step or more), `third hypothesis, first step`, and `second hypothesis, boundary`.
6. The remaining placement options, `next first step`, `strongest move, intermediate factor`, and then `best remaining estimate`.

A proposal's `role` is one of these strings, and `hypothesis` names the hypothesis it serves. With `--allow-reshape`, a move can be re-proposed on the slice shape that suits it (`change: reshape`). Proposals are listed with `oom_risk` ones last, then regressions (estimated gain below -0.5%, `_REGRESSION_PCT = 0.5`), then `layout` and `pipeline` proposals, then by `estimated_gain_pct`. A proposal also carries `mesh`, `moves`, `topology`, `num_slices`, `change` (`move`, `reshape` or `slices`), `bracket`, `effects` (the groups it shrinks or grows, with `scale`), `local_tokens_ratio`, `weight_bytes_ratio`, `allow_split_physical_axes`, `memory_fraction`, `oom_risk`, `warnings`, `torus_mapping` and `legality`.

**Memory check.** When the baseline's `peak_fraction` is known (from `--memory-profile`; serving profiles have none) and the token axes are known (`--token-axes`, or `token_axes` in `--arch-json`), `memory_fraction = peak_fraction x (parameter_fraction x weight_bytes_ratio + (1 - parameter_fraction) x local_tokens_ratio)`, and `oom_risk` is true above the larger of `_OOM_FRACTION = 0.95` and the baseline peak. Otherwise `oom_risk` is true when the larger of the two ratios reaches `_MEMORY_RISK_RATIO = 8.0` (exceeds it, when the token ratio is only an upper bound), and the Markdown table adds "(check memory)" from `_MEMORY_CAUTION_RATIO = 4.0`. First steps, boundaries and compound remedies are taken only from candidates without `oom_risk`. The memory estimate (linear in the two ratios) and its thresholds are model assumptions, not measurements.

## 4. Physical Torus Fit & Target Slice Planning

- **`check-topology`** (`check_mesh_topology_fit` in {{script:ici-mesh-sharding/mesh_topology.py}}) follows `jax.experimental.mesh_utils.create_device_mesh` with its default `allow_split_physical_axes=False`: every physical axis must go whole to one mesh axis. `--platform` sets the devices per chip (`devices_per_chip` in {{script:ici-mesh-sharding/tpu_specs.py}}: 2 on `tpu7x`, 1 elsewhere). It returns `device_count_matches`, `buildable`, `fits_without_split`, `allow_split_physical_axes` (true when the mesh can be built only with split axes), `assignment` and `reason`. With `--partition-config`, it also says where a stacked `RING` / `GRID` partition config places each axis (`partition_layout`).
- **Wrap-around links**: `torus_mapping` and `partition_layout` call an axis that owns a whole dimension a ring, and say whether that dimension has wrap-around links by `mesh_search.wraparound_dims`: every dimension of a 3D slice made of whole 4x4x4 cubes (documented for v4 and v5p, assumed for `tpu7x`), and only the 16-chip dimensions of a 2D slice (v5e, v6e).
- **`plan-topology`** (`plan_target_topology`) sizes a target slice from `--source-chips`, `--source-platform`, `--target-platform` and optionally `--memory-per-chip-gib`. It takes the larger of the chips needed for peak BF16 FLOP/s parity and the chips whose HBM, at 85% (`HBM_UTILIZATION_SAFETY_FACTOR = 0.85`), holds the source's memory, rounded up to a power of two. It recommends a balanced 3D shape (`v4`, `v5p`, `tpu7x`) or a near-square 2D shape (`v5e`, `v6e`; above 256 chips, several 256-chip slices) and reports both chip counts.

## 5. OSS / Serving Parallelism Rules (`TP`, `DP`, `EP`, `PCP`, `DCP`, `PP`)

{{script:ici-mesh-sharding/sharding_adapter.py}} and {{script:ici-mesh-sharding/oss_sharding_rules.py}} validate serving and OSS framework sharding configurations before launch:

- **World size (vLLM)**: `TP * DP * PCP * PP`, where `DP` includes `fsdp`. `DCP` reuses TP (or PCP) ranks, and `EP` is derived.
- **Derived `EP` (vLLM)**: vLLM sets `EP = TP * DP * PCP` when expert parallelism is on, so a proposal with `ep > 1` must use exactly that value. On every engine, `EP` must divide the number of experts.
- **`DCP` (vLLM)**: without `PCP`, `DCP` must divide `TP`; with `PCP`, `DCP` must be 1, `PCP` or `TP * PCP`.
- **Multi-Head Latent Attention (MLA)**: detected from the config's structure (`kv_lora_rank > 0`, or both `qk_rope_head_dim` and `qk_nope_head_dim` positive), else from `attention_type` or the model family (`ShardingNomenclatureAdapter.is_mla_model`). A GQA KV cache divides by `TP` (`num_kv_heads / TP` per rank), whereas MLA's latent cache (`kv_lora_rank + qk_rope_head_dim` per token) is not divided by `TP`; only context parallelism (`DCP` / `PCP`) divides it. `estimate_hbm_capacity` applies this when it checks the 85% HBM threshold.

## 6. AOT Compile Landing Verification (`verify-compile`)

Changing a mesh setting in a training or serving launcher can silently do nothing (a config parser ignores the flag) or fail to compile. {{script:ici-mesh-sharding/compile_verdict.py}} reads the compile log and the `--xla_dump_to` artifacts (`*after_optimizations*` HLO and `*-memory-usage-report.pb`, parsed with the standard library) and returns `verdict`, `error_class`, `evidence` and per-module evidence:

- **`LANDED`**: compiled, and the optimized HLO differs from the baseline in sharding, mesh devices, instruction or collective counts, or peak HBM. Without a baseline dump to compare against, the evidence says `(unverified)`.
- **`NO_OP`**: compiled to the same program as the baseline: the optimized HLO is byte-identical, or its sharding, mesh devices, op counts and peak HBM equal the baseline's.
- **`DROPPED`**: the compile failed, produced no optimized HLO, or the trial wrote config paths absent from the baseline plan. `error_class` is one of `hbm_oom`, `vmem_oom`, `sharding_error`, `mesh_split_required`, `explicit_prefetch_ret_check`, `tiling_failure`, `unknown_flag`, `unknown_config_path`, `no_hlo_dump`, `timeout` or `compile_error`.
