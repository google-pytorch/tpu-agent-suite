---
name: ici-mesh-sharding
description: Diagnose ICI/DCN collective bottlenecks from XProf exports, check 2D/3D TPU torus topology alignment, validate OSS/serving parallelism and MLA/MoE HBM fit, rank a 7-proposal mesh sharding portfolio, and verify AOT HLO compile landing. Use when asked to tune SPMD/Shardy mesh axes (data, fsdp, tensor, sequence, expert, replica), reduce exposed all-gather/reduce-scatter/all-to-all time, onboard a model mesh onto v4/v5e/v5p/v6e/tpu7x topologies, or verify whether a sharding change landed in XLA HLO.
delegates:
- tpu-diagnostics
- tpu-knowledge-base
scripts:
- scripts/ici-mesh-sharding/__init__.py
- scripts/ici-mesh-sharding/cli.py
- scripts/ici-mesh-sharding/tpu_specs.py
- scripts/ici-mesh-sharding/mesh_search.py
- scripts/ici-mesh-sharding/collective_breakdown.py
- scripts/ici-mesh-sharding/mesh_topology.py
- scripts/ici-mesh-sharding/mesh_portfolio.py
- scripts/ici-mesh-sharding/sharding_adapter.py
- scripts/ici-mesh-sharding/oss_sharding_rules.py
- scripts/ici-mesh-sharding/compile_verdict.py
- scripts/ici-mesh-sharding/xprof_parsers.py
- scripts/ici-mesh-sharding/rank_mesh_proposals.py
references:
- references/ici-mesh-sharding/analyzer.md
- references/ici-mesh-sharding/proposal.md
---

# ICI Mesh Sharding — Attribution, Topology Fit, Portfolio Ranking & Compile Verification

Optimize multi-chip and multi-slice TPU sharding (`data`, `fsdp`, `tensor`, `sequence`/`context`, `expert`, `replica`, plus serving `TP`/`DP`/`EP`/`PCP`/`DCP`/`PP`) from empirical profile evidence through physical torus mapping to AOT HLO compile verification.

> [!IMPORTANT]
> **Division of responsibility.**
> - Never parse raw `.xplane.pb` traces directly or invoke profiler binaries here. When starting from a live server or raw capture directory without exported JSON tables, delegate trace extraction to {{agent:tpu-diagnostics}} first, and consult {{agent:tpu-knowledge-base}} (`ici-mesh-collective-attribution`, `openxla-shardy`, `hw-tpu-v5p`, `hw-tpu-v6e`, `hw-tpu-7x`) for hardware and compiler invariants.
> - For pre-benchmark serving sweeps priced through the analytical op-level roofline (`points.csv` TTFT vs throughput curves), use `sharding-strategy` + `roofline-analysis`. Use **this skill (`ici-mesh-sharding`)** when attributing measured XProf collective time to mesh axes, checking 2D/3D physical torus split-axis penalties, onboarding a mesh across TPU generations (`v4`, `v5e`, `v5p`, `v6e`, `tpu7x`), ranking a profile-driven 7-slot mesh portfolio, or verifying that a sharding config actually landed in XLA HLO dumps.

---

## Workflow Overview

```
Phase 1: Profile & Architecture Attribution  ──►  cli.py breakdown
Phase 2: Topology Fit & 7-Slot Portfolio     ──►  cli.py check-topology / plan-topology / rank-portfolio
Phase 3: OSS / Serving & HBM Gate            ──►  cli.py validate-model
Phase 4: AOT HLO Compile Verification        ──►  cli.py verify-compile
```

Read {{reference:ici-mesh-sharding/analyzer.md}} before interpreting collective attribution or serving HBM headroom, and read {{reference:ici-mesh-sharding/proposal.md}} before customizing portfolio slots, split-axis overrides, or AOT compile verdicts.

---

## Phase 1 — Profile & Architecture Attribution

1. **Obtain exported profile JSONs**:
   - If the workspace or `{{runs}}/<run_id>/artifacts/` already contains exported XProf JSON files (`<work_dir>/op_profile.json`, optional `<work_dir>/memory_profile.json`, `<work_dir>/memory_viewer.json`, `<work_dir>/megascale_stats.json`, `<work_dir>/overview_page.json`), pass them directly to {{script:ici-mesh-sharding/cli.py}}.
   - Otherwise, delegate to {{agent:tpu-diagnostics}} to export the HLO op profile and memory breakdown JSON artifacts from the target capture.
2. **Attribute collective time to mesh axes and semantic roles**:

   ```bash
   python3 {{script:ici-mesh-sharding/cli.py}} breakdown \
     --ops-json <work_dir>/op_profile.json \
     --mesh '{"fsdp":8,"tensor":4,"expert":2}' \
     --output-format json
   ```

   Inspect (keys of the JSON output):
   - `collective_pct`: collective and async-wait time as % of profiled op time; `groups`: each collective group with its `kind`, `role`, `axes` label and `pct`. An `axes` label of `ambiguous: <a> or <b>` means two axis subsets have the same size and nothing in the HLO tells them apart.
   - `roles`: % per role, using the strings `MoE routing`, `KV gather (sequence)`, `ring / permute`, `tensor: collective matmul`, `activations / other`, `weights: FSDP gather / gradient`, `global (all devices)` and `host offload` (suffixed ` (by shape)` when inferred from shapes).
   - `per_axis` and `largest_reducible`: the time a move between mesh axes can shrink, per axis and for the largest (role, axes) group. They leave out `weights: FSDP gather / gradient`, `global (all devices)` and `host offload`, and any group labeled `all axes (global)`.

---

## Phase 2 — Topology Alignment & Ranked 7-Slot Portfolio

1. **Verify physical 2D/3D torus mapping and slice sizing**:

   ```bash
   python3 {{script:ici-mesh-sharding/cli.py}} check-topology \
     --mesh '{"fsdp":8,"tensor":4,"expert":2}' \
     --topology 4x4x4 \
     --platform v5p \
     --output-format json
   ```

   `buildable: true` with `fits_without_split: false` means the mesh needs `allow_split_physical_axes=True`; here `fsdp=8` cannot own whole 4-chip dimensions.

   When onboarding a workload onto a new TPU generation (`v4`, `v5e`, `v5p`, `v6e`, `tpu7x`), generate a hardware-aware slice and torus onboarding plan:

   ```bash
   python3 {{script:ici-mesh-sharding/cli.py}} plan-topology \
     --source-chips 64 \
     --source-platform v5p \
     --target-platform tpu7x \
     --memory-per-chip-gib 70.0 \
     --output-format json
   ```

2. **Enumerate legal factor moves or generate the ranked 7-proposal portfolio**:

   - To inspect every legal single-step pairwise power-of-two factor move (`donor ÷ k`, `receiver × k`):

     ```bash
     python3 {{script:ici-mesh-sharding/cli.py}} enumerate \
       --mesh '{"data":1,"fsdp":8,"tensor":4,"sequence":1,"expert":2}' \
       --topology 4x4x4 \
       --batch-size 128 \
       --sequence-length 8192 \
       --num-heads 64 \
       --num-kv-heads 8 \
       --output-format table
     ```

   - To combine XProf op-profile attribution, dynamic HBM headroom, DCN stall metrics, physical torus alignment, and architecture divisibility into a diverse 7-slot portfolio (`--num-proposals`, default `DEFAULT_NUM_PROPOSALS = 7`):

     ```bash
     python3 {{script:ici-mesh-sharding/cli.py}} rank-portfolio \
       --ops-json <work_dir>/op_profile.json \
       --mesh '{"data":1,"fsdp":8,"tensor":4,"sequence":1,"expert":2}' \
       --topology 4x4x4 \
       --batch-size 128 \
       --sequence-length 8192 \
       --num-heads 64 \
       --num-kv-heads 8 \
       --arch-json <work_dir>/arch.json \
       --memory-profile <work_dir>/memory_profile.json \
       --memory-viewer <work_dir>/memory_viewer.json \
       --megascale-stats <work_dir>/megascale_stats.json \
       --overview-page <work_dir>/overview_page.json \
       --output-format json
     ```

   > [!NOTE]
   > **Serving mesh guard.** `rank-portfolio` optimizes SPMD trainer collective axes (`data`, `fsdp`, `tensor`, `sequence`, `expert`, `replica`). If every axis of `--mesh` is a serving axis (`tp`, `dp`, `ep`, `sp`, `pcp`, `dcp`, `pp`), it stops with an error that points to `validate-model` (Phase 3) and {{reference:ici-mesh-sharding/proposal.md}}.

---

## Phase 3 — OSS / Serving Sharding & MLA / MoE HBM Validation

Before launching any serving (`vLLM`) or OSS (`MaxText`, `TorchTitan`) sharding candidate, validate axis divisibility, derived `EP` / `DCP` rules, MLA non-TP-sharded KV cache behavior, and the 85% HBM ceiling (`HBM_UTILIZATION_SAFETY_FACTOR = 0.85`):

```bash
python3 {{script:ici-mesh-sharding/cli.py}} validate-model \
  --proposal '{"tp":4,"dp":2,"pcp":1,"dcp":2,"pp":1,"ep":8}' \
  --model-config <work_dir>/model_config.json \
  --total-tpu-chips 8 \
  --tpu-type v6e \
  --output-format json
```

Key checks enforced by {{script:ici-mesh-sharding/sharding_adapter.py}} and {{script:ici-mesh-sharding/oss_sharding_rules.py}}:
- Power-of-two checks on `TP`, `DP`, `EP`, `PCP`, `DCP`, and `PP`.
- Attention query-head and KV-head divisibility by `TP`.
- vLLM derives `EP = TP * DP * PCP`, so `ep` must be 1 or exactly that; `EP` must divide the number of experts. Without `PCP`, `DCP` must divide `TP` (it reuses TP ranks and adds no devices); with `PCP`, `DCP` must be 1, `PCP` or `TP * PCP`.
- Multi-Head Latent Attention (MLA) detection (`kv_lora_rank > 0`, among others): the compressed KV latent state (`kv_lora_rank + qk_rope_head_dim`) is **not** divided by `TP`, only by context parallelism (`DCP` / `PCP`).

---

## Phase 4 — AOT HLO Compile Landing Verification

After editing a workload's mesh configuration and running an AOT compile (with `--xla_dump_to=<candidate_dump_dir>`), verify that the sharding change actually reached XLA and did not silently no-op or trigger an OOM/partitioner fallback:

```bash
python3 {{script:ici-mesh-sharding/cli.py}} verify-compile \
  --candidate-dir <work_dir>/candidate_hlo_dump \
  --baseline-dir <work_dir>/baseline_hlo_dump \
  --compile-log <work_dir>/candidate_compile.log \
  --exit-code 0 \
  --output-format json
```

- **`LANDED`**: Proceed to hardware benchmarking (`tpu-model-onboarding` or `workload-microbenchmark`).
- **`NO_OP`**: Stop before spending TPU slice time — inspect the launcher / config parser wiring, because the candidate compiled to the baseline's program (byte-identical optimized HLO, or the same sharding, mesh devices, op counts and peak HBM).
- **`DROPPED`**: Inspect `error_class` and consult {{agent:tpu-knowledge-base}} (for example `xla-error-e1001` or `openxla-flags-guidance`) before picking the next portfolio candidate.

---

## Files

| File | Role |
| --- | --- |
| {{script:ici-mesh-sharding/__init__.py}} | Package marker and exported pure-stdlib module overview |
| {{script:ici-mesh-sharding/cli.py}} | Unified CLI (`enumerate`, `breakdown`, `rank-portfolio`, `check-topology`, `plan-topology`, `validate-model`, `verify-compile`) |
| {{script:ici-mesh-sharding/tpu_specs.py}} | Public Cloud TPU hardware specs (`v4`, `v5e`, `v5p`, `v6e`, `tpu7x`: HBM, BF16 TFLOP/s, ICI topology) |
| {{script:ici-mesh-sharding/mesh_search.py}} | Pairwise power-of-two mesh candidate generator, legality guards, and torus block checks |
| {{script:ici-mesh-sharding/collective_breakdown.py}} | 4-tier HLO collective axis attribution (`mesh_axes`, named groups, decoded replica groups, shape ratios) and role classifier |
| {{script:ici-mesh-sharding/mesh_topology.py}} | Physical 2D/3D torus fit verification, split-axis decision logic, stacked partition layouts, and `TopologyOnboardingPlan` |
| {{script:ici-mesh-sharding/mesh_portfolio.py}} | Analytical communication-volume model and 7-slot hypothesis portfolio builder |
| {{script:ici-mesh-sharding/sharding_adapter.py}} | `ShardingNomenclatureAdapter` (`MaxText` / `vLLM`), MLA structural detector, roofline ridge point, and HBM estimator |
| {{script:ici-mesh-sharding/oss_sharding_rules.py}} | Power-of-two, head/expert divisibility, derived `EP`/`DCP`, and 85% HBM safety validator |
| {{script:ici-mesh-sharding/compile_verdict.py}} | Pure-stdlib HLO protobuf wire parser and `LANDED` / `NO_OP` / `DROPPED` AOT compile verifier |
| {{script:ici-mesh-sharding/xprof_parsers.py}} | Pure-stdlib parsers for exported XProf JSON artifacts (`op_profile`, `memory_profile`, `memory_viewer`, `megascale_stats`, `overview_page`) |
| {{script:ici-mesh-sharding/rank_mesh_proposals.py}} | End-to-end file-based glue joining XProf evidence, architecture specs, and `mesh_portfolio` |
| {{reference:ici-mesh-sharding/analyzer.md}} | Detailed reference on collective axis attribution, role classification, serving KV reservation, and DCN stalls |
| {{reference:ici-mesh-sharding/proposal.md}} | Detailed reference on factor moves, analytical comm-volume scaling, 7-slot portfolio, OSS/MLA rules, and compile verdicts |
