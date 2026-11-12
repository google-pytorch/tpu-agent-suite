---
name: ici-mesh-sharding-analyzer
description: Collective attribution, role classification, memory headroom, and DCN stall parsing for ICI mesh sharding analysis.
scripts:
- scripts/ici-mesh-sharding/collective_breakdown.py
- scripts/ici-mesh-sharding/xprof_parsers.py
---

# ICI Mesh Sharding — Profile & Collective Attribution Reference

This reference documents how {{script:ici-mesh-sharding/collective_breakdown.py}} and {{script:ici-mesh-sharding/xprof_parsers.py}} turn exported XProf JSON tables into collective time per mesh axis and role, HBM headroom, and cross-slice (DCN) stall evidence. Names in `code` are the keys and strings the scripts emit.

## 1. Collective Axis Attribution

`collective_kind` picks out collectives (`all-reduce`, `all-gather`, `reduce-scatter`, `all-to-all`, `ragged-all-to-all`, `collective-permute`, send/recv, barrier semaphores, and async-done waits). `extract_evidence` then takes the first of these that names the mesh axes:

1. **Kernel metadata**: `mesh_axes: [...]` in the op's HLO expression (SparseCore / Pallas kernels such as MoE ragged all-to-all routing). Evidence: `kernel mesh_axes=<axes>`.
2. **Named replica groups**: `replica_groups=mesh['data'=8,'seq'=4] {'seq'}`. Evidence: `groups over <axes>`. Positional names (`axis_0`, ...) are mapped onto the device order the profile's named groups give.
3. **Decoded replica groups**: iota `[G,S]<=[dims]T(perm)` or explicit `{{0,1,...}}`. With a device order from named groups, the axes along which group 0's devices vary are exact. Otherwise the group size is matched against products of up to three active axes of `--mesh` (`axes_for_size`).
4. **Shape factor**: for `all-gather`, `reduce-scatter` and `all-to-all`, the ratio of the one dimension that differs between operand and result (`x8 (d0:1024->8192)`).

A group's `axes` label is then:

- the axes (`expert`, `fsdp x tensor`) when exactly one subset matches;
- `<axes> (from dN)` when a same-size tie is broken by the activation dimension the collective gathers or scatters (d0 batch, d1 sequence, d2 tensor);
- `ambiguous: <a> or <b>` when several subsets match and nothing breaks the tie; these rows count toward `ambiguous_pct` and are never charged to an axis;
- `all axes (global)` when the group spans every device;
- `no axis product = <n>` or `unattributed` otherwise.

## 2. Collective Role Classification

`classify_role` gives each group one role. It tests these rules in order, and the first match wins. The role strings are the `ROLE_*` constants:

| Role | First matching rule | Charged to its axes in `per_axis`? |
| --- | --- | --- |
| `host offload` | the op name contains `host-` or `host_` | No |
| `MoE routing` | kind `ragged-all-to-all`; `ragged_all_to_all` in the op name, provenance or source file; `barrier_semaphore` in the op name; or a source file starting with `ragged_all_to_all`, `scheduling_groups`, `sparse_a2a`, `dense_a2a`, `fused_moe`, `moe_routing`, `token_dispatcher`, `megablox` or `gmm` | Yes |
| `KV gather (sequence)` | `k_hbm_ref` or `v_hbm_ref` in the op name | Yes |
| `weights: FSDP gather / gradient` | a source file starting with `fsdp_utils`, `fsdp`, `fully_sharded`, `zero_`, `layer_stack`, `rmsprop`, `adam`, `optimizer`, `optax` or `maxtext_fsdp` | No |
| `tensor: collective matmul` | `matmul` in the op name | Yes |
| `global (all devices)` | the group's label is `all axes (global)` | No |
| `ring / permute` | kind `collective-permute`, or `permute` in the op name | Yes |
| `activations / other` | anything else | Yes |

With `--sequence-length`, an `activations / other` all-gather or reduce-scatter can instead get `KV gather (sequence)` or `weights: FSDP gather / gradient` from its shapes (`_shape_role`). Such roles print with the suffix ` (by shape)`; confirm them with the op's provenance.

`_is_mesh_reducible` charges a group to its axes only when its exact axes are known, its label is not `all axes (global)`, and its role is not in `_NOT_PER_AXIS` (`weights: FSDP gather / gradient`, `global (all devices)`, `host offload`). Weight gathers and gradient reductions move about the same bytes per device whatever their group size, so moving factors between batch and sequence axes does not shrink them; more tensor or expert shards do, which is how `rank-portfolio`'s `weights` hypothesis treats them.

`cli.py breakdown --output-format json` returns the `summarize` dict:

- `collective_pct`: collective and async-wait time, as % of profiled op time.
- `groups`: rows with `kind`, `role`, `by_shape`, `axes`, `exact_axes`, `evidence`, `pct`, `ops`, `occurrences`, `source` and `example`.
- `axes` and `roles`: % of profiled op time per axes label and per role.
- `per_axis`: mesh-reducible % per axis. A group over several axes counts toward each, so rows overlap.
- `largest_reducible`: the largest mesh-reducible (role, axes) group, with `role`, `axes`, `devices`, `pct`, `kinds`, `evidence` and `example`.
- `ambiguous_pct`, `unattributed_pct`, `global`, `tensor_ar`, `device_order` and `mesh_mismatch`.

## 3. HBM Headroom & The Serving KV-Cache Reservation Pitfall

`parse_memory_profile` reads the device allocators of `memory_profile.json` (`memoryProfilePerAllocator.<allocator>.profileSummary`; host allocators are skipped): `memoryCapacity`, and the larger of `peakStats.peakBytesInUse` (`peakStats.peakBytesUsageHbm` in the `memoryProfileSummary` format) and `peakBytesUsageLifetime`, for the device with the highest peak. `parse_memory_viewer_module` reads `moduleName`, `totalBufferAllocationMib`, `peakHeapMib`, `entryComputationParametersMib` and the largest buffers of `memory_viewer.json`.

`summarize_memory_headroom` combines them into `peak_fraction` (peak / capacity) and `parameter_fraction` (`entryComputationParametersMib / peakHeapMib` of the largest module, else heap / peak; the basis is in `parameter_fraction_basis`).

Serving engines such as vLLM reserve most of HBM for the paged KV cache at startup, so a serving profile's peak reads near full whatever the mesh. `is_serving_modules` flags a serving profile when a module name contains `vllm`, `paged_attention` or `ragged_paged_attention`, or starts with `tt_jit_`, `jit_step_fun`, `jit_run_model`, `jit__decode_core`, `jit_generate` or `jit_insert_with_scratchpad`. Then `is_serving` is `true`, `peak_fraction` and `parameter_fraction` are `null`, and the raw fraction is reported as `reserved_fraction`, so the reservation is never read as a training peak.

## 4. Cross-Slice DCN Stall Evidence

`parse_megascale_stats` reads the `megascale_stats.json` DataTable into one entry per collective (`rendezvous_name`): `occurrences`, `observed_us_total`, `stall_us_total`, `slack_us_avg`, `bytes` and `required_bandwidth_gbps`. `summarize_dcn_stats` ranks them by `stall_us_total`, then `observed_us_total`, and computes `dcn_pct`: the total stall per host (observed time when no stall is recorded, an upper bound) over `num_steps * step_time_ms` from `overview_page.json`, or over the profile duration when there is no step data (a lower bound). `dcn_pct_basis` says which.

`rank-portfolio` uses `dcn_pct` only on a multi-slice run (`--num-slices` above 1) with `--allow-slice-change`: it then adds the `slices` hypothesis ("fewer, larger slices"), scored by `dcn_pct`. `--dcn-pct` overrides the value derived from `--megascale-stats`.
