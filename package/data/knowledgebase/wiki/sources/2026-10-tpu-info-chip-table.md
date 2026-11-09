---
title: "tpu_info chip table (cloud-accelerator-diagnostics)"
type: source
tags: [hardware]
created: 2026-10-27
updated: 2026-10-27
sources: 1
human_reviewed: false
---

The `TpuChip` table in `tpu_info/device.py` from Google's
cloud-accelerator-diagnostics repository: HBM per chip in GiB (`hbm_gib`) and the
devices the runtime exposes per chip. **Authority: Google open-source code.** It is
the table `compute inspect` sizes from (`hbm_gb_per_device` = `hbm_gib` /
`devices_per_chip`). Pinned at commit `35b5377566aa` (2026-10-05); the same values
ship in the `tpu-info` 0.14.2 package.

| Generation | HBM per chip | Devices per chip |
|---|---|---|
| v4 | 32 GiB | 1 |
| v5e | 16 GiB | 1 |
| v5p | 95 GiB | 1 |
| v6e | 32 GiB | 1 |
| 7x | 192 GiB | 2 |

## Gaps & caveats

- The Cloud TPU pages write some of these capacities as GB (v5e 16 GB; v6e 32 GB in
  the spec) and others as GiB (v4, v5p, and TPU7x, whose page uses 192 GiB and 96 GiB
  throughout since its 2026-10-22 update). The capacities are binary; this table names
  the unit (`hbm_gib`), so the hardware pages give GiB.
- These are nominal capacities, not the HBM free to an application after the runtime's
  reservations.

## Connections

Updated [TPU v5e](../hardware/tpu-v5e.md), [TPU v6e](../hardware/tpu-v6e.md) and
[TPU7x](../hardware/tpu-7x.md) (HBM capacity in GiB).

## Sources

- `TpuChip` in `tpu_info/device.py`, permalink at the pinned commit:
  https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/blob/35b5377566aa/tpu_info/tpu_info/device.py#L51-L60 (read 2026-10-27). Code is cited by permalink, not snapshotted under `raw/`,
  as for the other code sources.
