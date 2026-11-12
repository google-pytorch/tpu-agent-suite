<!--
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
---
entry: ici-mesh-collective-attribution
keyword: ici-mesh-collective-attribution
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-11-05
source_updated: not applicable
generator: authored
---

# Attributing XLA TPU collectives to logical mesh axes and physical ICI tori

When an SPMD or Shardy program (`index.py get openxla-shardy`, `index.py get
gspmd-paper`) spends device time in exposed `all-gather`, `reduce-scatter`,
`all-reduce`, `all-to-all`, `ragged-all-to-all`, or `collective-permute`
instructions (`index.py get openxla-operation-semantics`), the op category alone
does not tell you which mesh axis (`data`, `fsdp`, `tensor`, `sequence`,
`expert`, or `replica`) caused the stall. Attributing each collective to its
mesh axis and role separates the time a mesh change can shrink from fixed
global synchronization, and avoids proposing meshes that split physical ICI
torus dimensions or misreading pre-reserved serving KV memory.

Section 2's scaling rules and section 3's hop counts and `2x` factor are the
`ici-mesh-sharding` scripts' model, not measurements; the Evidence section
lists which rules are documented and where.

## 1. Resolving HLO collectives to logical mesh axes

Exported XProf HLO op profiles and XLA HLO text/proto dumps (`index.py get
openxla-hlo-dumps`) encode collective group membership at four levels of
specificity. The `ici-mesh-sharding` breakdown resolves each collective
instruction in this order:

1. **Explicit kernel metadata (`mesh_axes`).** Custom collective and Pallas
   kernels (for example MoE ragged all-to-all routing) can carry
   `mesh_axes: [...]` in the instruction's HLO text. Trust explicit axis
   annotations over group cardinality.
2. **Named replica groups.** Some HLO keeps mesh axis names in the replica
   groups, as in `replica_groups=mesh['data'=8,'seq'=4] {'seq'}` (the groups
   span `seq`).
3. **Materialized `replica_groups` cardinality.** Optimized HLO lowers mesh
   names to integer device lists:
   - Compact iota form: `replica_groups=[num_groups,group_size]<=[...]`, where
     the second dimension is the participant count per group.
   - Explicit nested braces: `replica_groups={{0,1},{2,3}}`, where the inner
     list length gives `group_size = 2`.

   Match `group_size` against active mesh axes (`size > 1`). If a single axis
   has `size == group_size`, attribution is exact. If two active axes share
   the same size (for example `fsdp=4, tensor=4`) and neither `mesh_axes` nor
   provenance tells them apart, report the time as ambiguous (the breakdown
   labels it `ambiguous: fsdp or tensor`) rather than assigning all of it to
   one axis.
4. **Operand/output gather and scatter shape ratios.** When profile exports
   truncate long `replica_groups` strings, `all-gather` output-to-operand
   element count ratios and `reduce-scatter` operand-to-output element count
   ratios recover `group_size`:

```hlotxt
%ag = bf16[8192,4096] all-gather(bf16[1024,4096] %param), dimensions={0}
// output_elements / operand_elements = 8 -> group_size = 8
```

A `collective-permute` names its peers in `source_target_pairs` instead of
replica groups; trace the directed pairs by hand to recover a ring's length.
The breakdown does not, and leaves such ops unattributed.

## 2. Which collectives a mesh move can shrink

Summing raw collective time overstates what a mesh factor move (`donor / k ->
receiver * k`, device count unchanged) can save. Classify each collective by
its role first:

- **Fixed under such a move**:
  - **Reductions over every device** (loss logging, gradient-clipping norm,
    optimizer step counters): the group stays all `N = prod(mesh)` devices.
  - **Host transfers** waiting on infeed or checkpoint outfeed.
- **Shrink or grow with the sizes of the axes they span** (the breakdown's
  `per_axis` and `largest_reducible`):
  - **Tensor-parallel activation collectives** (`all-reduce`, or
    `reduce-scatter` + `all-gather`, on MLP and attention projections).
  - **KV gathers and ring attention** over sequence axes.
  - **MoE token routing** (`all-to-all` or `ragged-all-to-all`).
- **FSDP weight gathers and gradient reductions** move about the same bytes
  per device whatever their group size, so moving factors between batch and
  sequence axes does not shrink them; more tensor or expert shards do.

A ring collective over `n` devices moves `(n - 1) / n` of the data per device
(`index.py get concept-collective-costs`). On top of that, the scripts' model
scales tensor-parallel activation collectives as local tokens x `(t - 1) / t`,
KV gathers as per-device batch x KV heads per device x `(s - 1) / s`, MoE
routing as local tokens x `(e - 1) / e` x the group's mean hop count on the
torus, and FSDP traffic as `1 / (t * e)` x `(g - 1) / g`, where `g = data *
fsdp`.

## 3. Physical 2D and 3D ICI torus alignment

Cloud TPU slices wire chips into 2D tori on `v5e` and `v6e` (`index.py get
hw-tpu-v6e`) and 3D tori on `v4`, `v5p` (`index.py get hw-tpu-v5p`) and
`tpu7x` (`index.py get hw-tpu-7x`, 2 TensorCores per chip). Only some slices
have wrap-around links (`index.py get scaling-book-tpus`, `index.py get
cloud-tpu-v5p`, `index.py get concept-ici`):

- `v5e` and `v6e` get a wrap-around link only on an axis of 16 chips: a `v6e`
  `4x4` slice (16 chips) has none, and `8x16` has one on its long axis only.
- 3D `v4` and `v5p` slices get them on every axis once the slice is made of
  whole `4x4x4` cubes: `4x4x4` and `4x4x8` are full 3D tori, `2x2x4` has none.
  `tpu7x` topologies larger than 64 chips are built from the same cubes
  (`index.py get hw-tpu-7x`); the scripts assume the same rule for them.

A ring without wrap-around links takes about twice as long as one with them
(`index.py get concept-ici`).

Logical mesh axes map cleanly only when each axis owns whole physical
dimensions, which JAX's `create_device_mesh` requires by default
(`allow_split_physical_axes=False`). On a `v5p` `4x4x4` slice, `tensor=4,
fsdp=4, expert=4` gives each axis a whole ring with wrap-around links.
`tensor=8, fsdp=4, expert=2` does not fit: `tensor` needs one dimension plus
half of another, which `expert` also uses, so the mesh builds only with
`allow_split_physical_axes=True` and its collectives run as strided or partial
rings over shared links. Prefer a whole-axis factor move when one exists, and
split physical axes only when parameter or KV memory rules out every
whole-axis mesh.

The scripts' estimate charges such shared-dimension groups a factor of
`_SHARED_DIM_PENALTY = 2.0` and counts `L / 4` hops per dimension of length
`L`, the mean on a ring with wrap-around links, on every slice. Both are
model assumptions. Its placement text applies the wrap-around rule above; its
hop counts do not, so on slices without wrap-around links they are low.

## 4. Two memory traps when ranking mesh moves

1. **Static KV-cache reservation in serving profiles.** vLLM sizes its KV
   cache by budget: it takes the `--gpu-memory-utilization` fraction of
   device memory (default `0.92`) and allocates what the weights leave as
   cache blocks (`index.py get vllm-engine-args`). A serving profile's peak
   memory therefore sits near that fraction whatever the mesh. On serving
   profiles, ignore the reserved peak and bound candidate meshes with
   parameter bytes from `memory_viewer` plus an analytical KV-cache footprint
   at the target batch and sequence length.
2. **MLA KV cache under `TP`.** In standard GQA, per-rank KV cache shrinks
   linearly with `TP` up to `num_kv_heads`; above that, vLLM replicates each
   KV head `TP / num_kv_heads` times (`index.py get
   concept-tensor-parallelism`). Multi-Head Latent Attention caches one
   compressed latent per token, shared by all heads (`index.py get
   deepseek-v2-paper`; `kv_lora_rank + qk_rope_head_dim` values in model
   config terms), so splitting heads across `TP` ranks does not divide it:
   each rank holds it whole. Only context parallelism (`DCP` / `PCP`,
   `index.py get vllm-torchtpu-parallelism`) divides MLA KV memory per chip.

## What to check

- Before changing mesh axes, read the breakdown's `per_axis` and
  `largest_reducible`. If reductions over every device or host transfers
  dominate collective time, no mesh move with the same device count shrinks
  them; FSDP weight traffic shrinks only with more tensor or expert shards.
- Check whether `tensor` or `expert` owns whole physical dimensions on the
  target slice, and whether those dimensions have wrap-around links: a `v6e`
  `4x4` slice (16 chips) has whole dimensions but no wrap-around links; a
  `v5p` `4x4x4` slice (64 chips) has them on all three dimensions.
- After editing launcher mesh flags, run an AOT `--xla_dump_to` compile and
  diff optimized HLO `replica_groups` and peak parameter bytes against
  baseline to catch silent `NO_OP` flag drops before benchmarking on hardware.

## Evidence

Documented, in public sources:

| Claim | Source |
| --- | --- |
| Replica groups, `source_target_pairs`, and the shapes of `all-gather` / `reduce-scatter` results | `index.py get openxla-operation-semantics` |
| Named mesh axes and SPMD partitioning | `index.py get openxla-shardy`, `index.py get gspmd-paper` |
| HLO dumps from `--xla_dump_to` | `index.py get openxla-hlo-dumps` |
| 2D tori on `v5e` / `v6e`, 3D tori on `v4` / `v5p` / `tpu7x`; 2 TensorCores per `tpu7x` chip; `tpu7x` topologies above 64 chips built from `4x4x4` cubes; 16 chips in a `v6e` `4x4` slice, 64 in a `v5p` `4x4x4` slice | `index.py get hw-tpu-v6e`, `index.py get hw-tpu-v5p`, `index.py get hw-tpu-7x` |
| Wrap-around links only on 16-chip axes of `v5e` / `v6e`, and on 3D `v4` / `v5p` slices of whole `4x4x4` cubes | `index.py get scaling-book-tpus`, `index.py get cloud-tpu-v5p`, `index.py get concept-ici` |
| A ring without wrap-around takes about `2x` as long; ring cost `(n - 1) / n` | `index.py get concept-ici`, `index.py get concept-collective-costs` |
| vLLM sizes the KV cache by `--gpu-memory-utilization` (default `0.92`) | `index.py get vllm-engine-args` |
| vLLM replicates each KV head `TP / num_kv_heads` times when `TP` exceeds the KV-head count | `index.py get concept-tensor-parallelism` |
| MLA caches one compressed latent per token, shared by all heads | `index.py get deepseek-v2-paper` |
| Prefill and decode context parallelism in vLLM on TPU | `index.py get vllm-torchtpu-parallelism` |

`allow_split_physical_axes=False` is the default of JAX's
`jax.experimental.mesh_utils.create_device_mesh` (JAX API reference; no index
row).

The HLO lines, replica groups and meshes in sections 1 and 3 are worked
examples, not profile data: the scripts return the group sizes, labels and
fit results given for them, and the scripts' tests check those results.

Model assumptions of the `ici-mesh-sharding` scripts, not measured: the
resolution order in section 1 and the `mesh['data'=8,'seq'=4] {'seq'}` group
form it reads; the role classes and per-role scaling rules in section 2; the
`L / 4` mean hop count per dimension; and the `2x` shared-dimension factor
(`_SHARED_DIM_PENALTY`). No benchmark backs this page.
