---
title: "XProf hardware counters that mislead"
type: concept
tags: [tpu-kernel-development, profiling, xprof, llo, counters]
created: 2026-11-17
updated: 2026-11-17
sources: 0
human_reviewed: false
---

# XProf hardware counters that mislead

Read this before putting any LLO hardware counter from an XProf capture into a ratio, a
utilization figure or a comparison between kernels. Every statement here is an observation on the
capture path and toolchain it was found on, not documented behaviour.

These are the xprof/LLO hardware counter planes (`..._TC_...` on `/device:TPU:N`). The facts below
hold for the capture path and toolchain they were found on; re-check them on a new one with the
determinism control below. How a profile of a Pallas kernel is captured and read is in
[pallas-validation-and-xprof](pallas-validation-and-xprof.md).

**`COUNT_CYCLES` is a free-running counter over an enable window, not a measure of execution.**
The window is decoupled from the kernel, module and capture. It can run far longer than the
program, and its implied clock rate varies wildly between captures. **Never put it in a
denominator**: `X / COUNT_CYCLES` divides a real quantity by a meaningless one.

**`COUNT_MXU_BUSY_<k>` counts MXU *idle*, not busy.** It equals
`COUNT_CYCLES − COUNT_MATPUSH_CYCLES_MXU_<k>` exactly, and the `MXU_BUSY_*` counters sum exactly to
`COUNT_CYCLES`. They partition the enable window and inherit its non-determinism. `XLU_BUSY_*` reads
the same way. With `num_mxus` MXUs receiving identical push counts, one MXU's counter is only its
share of the traffic, not a per-unit breakdown.

**Pinning a different `jax`/`jaxlib` wheel changes neither compiler**: Mosaic TPU and XLA:TPU
both live in `libtpu.so`. **An A/B across jax versions that holds libtpu fixed is null by
construction**, and its near-zero delta does not clear the compiler.

**Run a determinism control before reading any counter.** Run the same variant and shape in two
jobs, or two replicate captures in one, and discard every counter that moves.
- `COUNT_CYCLES` and `COUNT_MXU_BUSY_*` moved between clean replicates, and moved much more when
  the host stalled.
- These stayed bit-identical: `COUNT_MATPUSH_MATRICES_MXU_*`, `COUNT_MATPUSH_CYCLES_MXU_*`,
  `COUNT_MATMUL_VREG_F32_MXU_*`, `COUNT_VECTOR_ISSUE`, `COUNT_SCALAR_ISSUE`,
  `COUNT_BUNDLES_INTO_VIF`, `COUNT_VST_INSTRUCTION`, `COUNT_VLD_INSTRUCTION_*`,
  `COUNT_ROTATE_PERMUTE_INSTRUCTION_XLU_*`. **They are sound as raw counts and as same-job deltas
  between arms.** They are not sound as fractions. They are whole-chip aggregates that include
  non-kernel ops, so a per-time rate built from one is not an occupancy.

**A percentage looks normalised by something real.** Sanity-check the implied physical constant
(cycles over wall time is a clock speed; that check exposed `COUNT_CYCLES`). Prefer a raw count
with its denominator stated inline: a "percent of cycles" once turned out to be
`COUNT_VECTOR_ISSUE / COUNT_SCALAR_ISSUE`. **A resource at exactly 100.0% is a quantity divided by
itself, never a finding.**
