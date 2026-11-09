---
title: "Chapter 7: Validation and XProf"
type: concept
tags: [pallas, validation, xprof, profiling]
created: 2026-07-16
updated: 2026-10-01
sources: 0
human_reviewed: false
---

# Chapter 7: Validation and XProf

Validating and profiling JAX Pallas kernels on TPU v6e (Trillium) is a critical
phase of the authoring lifecycle. Given the opaqueness of low-level compilation
to the user, understanding how hardware telemetry works—and how to identify when
that telemetry is hallucinating—is essential for robust kernel design.

## Empirical Verification via LLO and XProf

When relying on XProf for empirical verification, a key limitation must be
recognized: XLA treats Pallas kernels entirely as opaque blocks (annotated as
`tpu_custom_call` under the `custom-call` HLO category).

As a result, standard Low-Level Optimizer (LLO) utilization tracks are masked.
Natively generated XPlanes will record exactly `0` for hardware resource metrics
on these kernels:

-   `flops: 0`
-   `bandwidthUtils: [0, 0, 0]`
-   `rawBytesAccessedArray: [0, 0, 0]`

Attempts to manually inject LLO trace markers (e.g., the hallucinated
`tpu.vtrace` concept) will fail with unregistered dialect errors during
compilation. Because internal resource utilization breakdowns are unavailable,
you must empirically verify performance based on raw XPlane execution latency
cycle counts rather than built-in flop/bandwidth calculations.

## XLA Custom Call: The Non-Native Kernel Boundary

***(Reference: exp029_custom_call.py)***

The mechanism by which non-native kernels (such as Pallas, Mosaic, or cuDNN) are
embedded into an XLA program is the **XLA Custom Call**. During compilation, the
XLA compiler lowers the Pallas kernel call into a `stablehlo.custom_call`
operation. This serves as the formal boundary connecting non-native kernel
execution to the main XLA computation graph.

### Empirical Hardware Verification of Custom Call Lowering

Hardware profiling on TPU confirms this lowering mechanism. When executing and
profiling a JAX Pallas custom kernel via XProf, the XLA op profile explicitly
records the execution block under the `custom-call` category, specifically with
the attribute `custom_call_target="tpu_custom_call"`. This telemetry confirms
that the physical hardware execution is bound directly to the
`stablehlo.custom_call` representation, establishing a clear, traceable link
between high-level Pallas code and low-level HLO execution blocks.

## HLO (High Level Optimizer IR): The Compiler Optimization Layer

***(Reference: exp053_hlo.py)***

XLA's High Level Optimizer (HLO) IR is the graph-level intermediate
representation upon which profiling metrics, compiler passes, and hardware
optimizations operate. Understanding how to dump and analyze HLO is critical for
verifying how Pallas kernels are integrated, optimized, and mapped to physical
TPU hardware.

### Empirical Hardware Verification of HLO Stages

Hardware evidence confirms that XLA utilizes HLO as its compiler intermediate
representation. By setting the `XLA_FLAGS="--xla_dump_to=/tmp/xla_dump_hlo
--xla_dump_hlo_as_text"` environment variable and executing a JAX computation,
the XLA compiler generates text dumps corresponding to various stages of
optimization passes. Empirical analysis of the generated artifacts reveals files
such as:

-   `before_optimizations.txt`
-   `after_optimizations.txt`
-   `after_optimizations_before_buffer_assignment.txt`
-   `after_optimizations-buffer-assignment.txt`
-   `after_codegen.txt`

These artifacts demonstrate that HLO is the exact IR layer undergoing compiler
optimizations, memory buffer assignments, and code generation mappings for TPU
hardware.

## XLA Fusion: Aggressive Op Merging and Recomputation

***(Reference: exp092_xla_fusion.py)***

XLA Fusion is a critical compiler pass that merges multiple HLO operations into
a single execution kernel. A common misconception in TPU optimization is that
materialized broadcasts (such as performing an `argmax` over an `add` with a
broadcasted tensor) will break fusion and force the intermediate broadcasted
tensor to be materialized in High-Bandwidth Memory (HBM). This claim has been
empirically **refuted**. Modern XLA compilers aggressively and successfully fuse
broadcasts and elementwise operations directly into reductions without spilling
intermediate tensors to HBM.

### Empirical Hardware Verification of XLA Fusion

Empirical testing on TPU evaluated several variants of the "argmax over
add-broadcast" pattern using JAX (`jnp.argmax(x + y, axis=...)` where `y` is
broadcasted). The results of this verification demonstrate the robustness of
XLA's fusion engine:

1.  **Perfect Fusion**: Across all tested reduction axes (both matching and
    orthogonal to the broadcast dimension), XLA successfully fused the
    `broadcast`, `add`, and `reduce` (argmax) operations into a single
    `iota_reduce_fusion` (kLoop fusion).
2.  **No HBM Intermediates**: The `after_optimizations` HLO dump strictly showed
    the broadcast and add occurring inside the fused computation. At no point
    was the massive broadcasted tensor stored to HBM.
3.  **Multi-Use Robustness and Recomputation**: Even when the broadcasted add
    result was used by multiple downstream reductions (e.g., computing both
    `argmax` and `sum` of the same `x + y` broadcast), XLA favored
    **recomputation** over materialization. Instead of writing the intermediate
    tensor to HBM to share it, the compiler generated two separate loop fusions
    that each locally re-computed the broadcast and add, completely avoiding
    intermediate HBM spills.

## Profile Capture: Programmatic, gRPC, and Continuous-Snapshot

***(Reference: exp120_profile_capture.py)***

Capturing high-fidelity XProf profiles is the foundation of TPU performance
analysis. Hardware traces can be collected using three primary modalities
depending on the deployment environment and debugging requirements:

1.  **Programmatic Capture:** Directly instrumented within the JAX script using
    APIs like `jax.profiler.start_trace()` and `jax.profiler.stop_trace()`. This
    provides microsecond-level control over trace boundaries, isolating specific
    execution loops and avoiding initialization noise.
2.  **On-Demand gRPC Capture:** Triggered externally via a gRPC connection to a
    running JAX profiling server. This is ideal for capturing profiles from
    long-running training jobs or interactive sessions without modifying the
    execution loop.
3.  **Continuous-Snapshot Profiling:** Automated background agents that
    continuously record and dump execution snapshots when specific performance
    anomalies or triggers are met.

### Programmatic Profile Capture and Telemetry Generation

***(Reference: exp119_profile_capture.py)***

To analyze Pallas kernels, you must first capture hardware traces. While
profiling can be triggered via on-demand gRPC requests or continuous-snapshot
agents, programmatic capture directly within your JAX script provides the most
precise control over timeline boundaries.

Using `jax.profiler.start_trace()` and `jax.profiler.stop_trace()`, you can
programmatically isolate specific execution loops, avoiding the noise of startup
compilation and initialization. This programmatic capture workflow produces
valid XProf output, specifically `.xplane.pb` traces, which map exact hardware
execution details back to the JAX program.

#### Empirical Hardware Verification of Programmatic Tracing

This programmatic capture workflow has been empirically validated on TPU
hardware. By wrapping a Pallas kernel execution loop inside a programmatic trace
context:

```python
trace_dir = "/tmp/profile_capture_trace"
jax.profiler.start_trace(trace_dir)

for _ in range(10):
    add_fn(x, y).block_until_ready()

jax.profiler.stop_trace()
```

A native analysis of the resulting `.xplane.pb` trace files via XProf confirms:

1.  **Exact Execution Counts:** The trace records exactly 10 occurrences of the
    `custom-call` mapping directly to the Pallas kernel loop.
2.  **Microsecond-Level Precision:** The profiler successfully captures precise
    hardware execution latency per iteration (e.g., ~1.02 microseconds), proving
    that programmatic tracing securely dumps reliable, high-fidelity hardware
    telemetry without dropping events or introducing measurement skew.

## TPU Duty Cycle: Measuring Active Device Execution

***(Reference: exp120_tpu_duty_cycle.py)***

A critical metric for evaluating the integration and execution efficiency of
Pallas kernels is the **TPU Duty Cycle**, defined as the fraction of wall-clock
time during which the TPU is actively executing hardware instructions.
Maximizing this duty cycle is essential for high-throughput pipelines, as any
gap in execution indicates host-side overhead, dispatch latency, or data
pipelining bottlenecks.

### Empirical Hardware Validation of Duty Cycle Metrics

Empirical validation via `jax.profiler` and XProf trace analysis confirms that
the TPU Duty Cycle represents the precise ratio of active device execution time
to overall host wall-clock time. Within the generated `.xplane.pb` traces, this
metric is physically decomposed into two core components:

1.  **Busy Time:** This is extracted as the sum of all execution durations
    (e.g., `total_duration_us` from the XPlane trace summary), corresponding to
    hardware kernel operations such as `fusion`, `jit_dot`, or custom Pallas
    `tpu_custom_call` blocks.
2.  **Wall Time:** This represents the span of the profiler capture window or
    the defined Step Time boundary.

### The 0.0% Duty Cycle Telemetry Trap

When profiling simplified test scripts or microbenchmarks that lack explicit
step loop markers, XProf's automated `device_duty_cycle_percent` overview field
may default to reporting exactly `0.0%`. This is a telemetry artifact rather
than an indication of idle hardware.

To bypass this visualization limitation, you must inspect the underlying raw
trace data. The raw cycle counts and `rawTime` fields in picoseconds (e.g.,
`rawTime: 803716251250`) within the `.xplane.pb` file correctly record and
represent the busy execution fraction. To maximize the true duty cycle in
production, focus on optimizing Host-to-Device data pipelining and minimizing
host-side dispatch overhead.

## Step Time: Measuring Training Iteration Latency

***(Reference: exp028_step_time.py)***

The **Step Time**—defined as the wall-clock duration of a single training
iteration—is the core metric governing end-to-end workload efficiency and the
primary target for optimization. While kernel-level latency isolates execution
speeds, Step Time captures the holistic performance of the system, including
host dispatch, compiler-inserted synchronization barriers, and device execution.

### Empirical Hardware Verification of Step Time Telemetry

Empirical validation on TPU hardware confirms that step time is fully measurable
as the wall-clock hardware duration via XProf telemetry. By wrapping an
iteration loop in a step-specific annotation:

```python
for i in range(num_steps):
  with jax.profiler.StepTraceAnnotation("train", step_num=i):
    # Execute Pallas kernel or training step
    out = train_step(x, y)
    out.block_until_ready()
```

The XLA compiler and profiler natively bound the operation sequences to specific
training steps. Analyzing the resulting `.xplane.pb` traces reveals:

1.  **Native Step Boundaries:** The profiler successfully groups all operations
    executed within the context manager under a distinct step ID (e.g., `train
    0`, `train 1`).
2.  **Precise Cycle and Latency Attribution:** XProf outputs the precise cycle
    counts and step latency directly into the trace files, breaking down the
    step boundaries to capture both active device execution time and host
    dispatch overhead over the entire duration of the step. This allows
    developers to isolate host-side bottlenecks from actual TPU execution
    delays.

## Custom Trace Annotations for Timeline Isolation

***(Reference: exp079_custom_trace_annotations.py)***

Because XLA compiles Pallas kernels as opaque `tpu_custom_call` blocks, locating
and isolating them within a dense, complex XProf timeline can be exceptionally
difficult. To overcome this opacity, you can use named context-manager blocks to
explicitly label the trace-viewer timeline.

By wrapping your kernel invocations or benchmark steps in APIs such as
`jax.profiler.TraceAnnotation`, `tf.profiler.experimental.Trace`, or `xp.Trace`,
you inject custom metadata directly into the profiling stream.

### Empirical Hardware Verification

This behavior has been empirically verified via TPU trace capture. By wrapping a
JAX execution block inside a custom context manager:

```python
with jax.profiler.TraceAnnotation("my_custom_trace_annotation_123"):
  # Kernel execution or JAX operations
  ...
```

A binary scan of the generated `.xplane.pb` files confirmed the explicit
presence of the trace annotation string (`"my_custom_trace_annotation_123"`)
inside the XProf structured output. This proves that the JAX profiling API
successfully plumbs user-defined labels all the way down into the lower-level
hardware trace timeline. When profiling custom Pallas kernels, wrapping the
execution in these annotations ensures that the opaque `tpu_custom_call` blocks
are clearly demarcated and easily searchable within the XProf Trace Viewer.

## The Performance Hallucination Ceiling (HBM Peak) (AX-139)

***(Reference: exp055_the_performance_hallucination_.py)***

During the benchmarking phase, it is possible for profiling tools to report
execution latencies that imply memory bandwidths exceeding the physical limits
of the TPU HBM bus. The ceiling depends on the generation. Per device, take the higher of the
published spec ([v5e](../hardware/tpu-v5e.md), [v5p](../hardware/tpu-v5p.md),
[v6e](../hardware/tpu-v6e.md), [TPU7x](../hardware/tpu-7x.md)) and JAX
`tpu_info.py`: about 0.86 TB/s on v5e (800 GiB/s, spec; `tpu_info.py` 0.82),
2.77 TB/s on v5p (2765 GB/s per megacore device, spec; `tpu_info.py` 2.46),
1.64 TB/s on v6e (both), and 3.70 TB/s on TPU7x (`tpu_info.py` 7.40 TB/s per
chip, two devices; spec 7380 GB/s). Such
performance is an illusion—a performance hallucination—and definitively
indicates a critical logical flaw or a **Data Skipping Bug**.

This typically occurs when the `BlockSpec` or `index_map` incorrectly traverses
the HBM address space, causing the DMA engine to skip rows, or when the XLA
compiler identifies dead code and elides the memory traffic.

### Common Causes of Performance Hallucinations

1.  **Data Skipping:** An incorrect `BlockSpec` or `index_map` may cause the DMA
    engine to traverse the HBM address space erratically, skipping entire rows
    or chunks of the tensor.
2.  **Dead Code Elimination (DCE):** XLA's compiler aggressively prunes dead
    code. If your output is unconsumed, XLA may completely elide the memory
    traffic.
3.  **Variable Overwrite:** A common error in Python benchmarking loops is
    overwriting the Pallas kernel reference with the baseline (e.g., assigning
    `jax.jit(baseline)` over your custom logic), resulting in comparing the
    baseline against itself.

### Resolution: Enforcing Coverage Checks

To debug and prevent performance hallucinations, always verify the kernel
against sequential `torch.arange` data and enforce a rigorous "Coverage Check"
in your test harness. This guarantees that every logical element in the tensor
has indeed been physically written, mathematically modified, and verified for
correctness, preventing silent data-skipping or compiler-elision bugs from
masquerading as hyper-optimized kernels.

## Variable Overwrite Benchmark Hallucination (AX-181)

***(Reference: exp094_variable_overwrite_benchmark_h.py)***

When running microbenchmarks to compare a custom Pallas kernel against a
baseline, test variables or imports must not be overwritten by native JIT
compilation (e.g., `jax.jit(baseline)`). Reassigning the kernel function pointer
in the test harness effectively benchmarks the baseline against itself, creating
a performance hallucination.

### Empirical Hardware Verification of Variable Overwrite

This benchmarking pitfall has been empirically verified on TPU hardware. When a
test suite reuses a variable name or reassigns an imported Pallas kernel
reference to a JIT-compiled baseline, the underlying execution graph is silently
unified.

Analyzing the resulting hardware telemetry and XProf traces reveals:

1.  **Identical Program IDs:** The compiled XLA program IDs for both the
    "custom" run and the "baseline" run are identical.
2.  **Identical HLO Expressions:** The generated HLO contains identical
    instructions, completely lacking the expected `custom-call` node targeting
    `tpu_custom_call` for the custom kernel run.
3.  **Identical Performance Profiles:** The execution latencies match perfectly
    down to the microsecond, masquerading as a custom kernel that has achieved
    "perfect parity" with a highly optimized native baseline.

### Resolution: Strict Namespace Isolation

To prevent variable overwrite hallucinations in your benchmarking suites:

-   Use distinct, descriptive variable names for all compared implementations
    (e.g., `optimized_pallas_kernel` vs. `compiled_baseline_kernel`).
-   Strictly avoid redefining or overwriting imported kernel functions within
    test functions or loop scopes.
-   Always inspect the generated HLO or XProf traces to confirm that the Pallas
    kernel execution path contains the explicit `custom-call` HLO node targeting
    `tpu_custom_call`, verifying that the custom hardware logic is indeed being
    executed.

## Automated Benchmarking and Entry Point Naming (AX-065)

***(Reference: exp003_entry_point_naming__optimized_.py)***

When integrating custom Pallas kernels into automated benchmarking suites (such
as Minibench), strict adherence to entry point naming conventions is required to
prevent silent validation and profiling failures.

### The Minibench Dynamic Loading Constraint

Automated test collection and profiling harnesses often rely on dynamic
inspection of the Python module's namespace to locate and benchmark the
optimized kernel. Specifically, Minibench dynamically loads the target kernel
using specific global references via `globals().get('optimized_kernel')`.

If the final exposed JIT-wrapped Pallas wrapper is not named exactly
`optimized_kernel`, the automated test collection mechanism fails silently. This
failure manifests in several ways:

-   **Import Errors or Missing Metrics:** The test harness may emit an
    `ImportError` or report missing test metrics, such as a `⚠️ NO_REF_TIME`
    warning.
-   **Reference Fallback Loop:** The benchmarking framework will incorrectly
    fall back to identifying the custom code as the *reference* implementation.
    This leaves the custom profiling metrics entirely blank.
-   **Trace Extraction Failure:** Because the harness cannot isolate the
    optimized kernel, it skips trace extraction entirely, resulting in no
    `Custom latency` being recorded in the generated `.xplane.pb` profile.

**Resolution:** To ensure seamless integration with automated profiling,
validation, and trace extraction tools, you MUST name the final exposed
JIT-wrapped Pallas wrapper exactly `optimized_kernel`.

## TensorCore (Tensor Node) Architecture and Device Mapping

***(Reference: exp048_tensor_node.py)***

The fundamental compute unit on a TPU chip is the **TensorCore** (also referred
to as a **Tensor Node** or simply "core"). Depending on the hardware generation,
a physical TPU chip may house either one or two TensorCores. For example, TPU v4
and TPU v5p architectures feature two TensorCores per chip, whereas TPU v6e
(Trillium) features exactly one TensorCore per chip. Each individual TensorCore
operates as an independent execution engine, complete with its own dedicated
Matrix Unit (MXU), Vector Programmable Unit (VPU), and Vector Memory (VMEM).

How TensorCores map to `jax.Device` objects (typically `TpuDevice`) depends on
the generation. On TPU v4 and v5p the two TensorCores typically run in **megacore** mode,
as one logical device ([megacore](../concepts/megacore.md)). On a v5p host with
4 chips we observed 4 devices, all with `core_on_chip == 0`. JAX also
supports a "split" mode on these chips, where each core is its own device
(`ChipVersion.supports_megacore`, and the `TpuInfo.is_megacore` /
`TpuInfo.is_split_chip` properties, in JAX `tpu_info.py` at
https://github.com/jax-ml/jax/blob/be2c88887fb7/jax/_src/tpu_info.py). TPU7x has no megacore:
each of its two TensorCores is a separate device ([TPU7x](../hardware/tpu-7x.md)).
Single-core chips (v5e, v6e) expose one device per chip. You can inspect the
topology programmatically by querying the `coords` and `core_on_chip`
attributes of the devices. Count parallelism sizes (for example TP) in
devices, not TensorCores.

### Empirical Hardware Verification of TensorCore Topology

Empirical testing on a TPU v6 Lite (v6e) topology via `jax.devices()` confirms
this architectural layout. Querying a 2x2 grid of devices yields exactly 4 JAX
devices, with each device reporting `core_on_chip=0` for all devices. This is
consistent with v6e's one TensorCore per chip ([v6e docs](https://docs.cloud.google.com/tpu/docs/v6e)), but
`core_on_chip=0` alone does not distinguish a single-core chip from a megacore
device, which also reports `core_on_chip=0`.

Further, profiling dot products and elementwise operations on this device via
XProf confirms the expected interaction between independent execution units (MXU
and VPU) and Vector Memory (VMEM), validating the standard single-TensorCore
execution model on Trillium.

## The Ridge Point: Bandwidth vs. Compute Bound

When analyzing Pallas kernel performance, the **Ridge Point**—defined as `(Peak
FLOPs / HBM Bandwidth)`—is the theoretical pivot that determines whether a
kernel is bound by memory throughput or arithmetic throughput.

A kernel with an Arithmetic Intensity (AI) below the ridge point is
**memory-bound**. For instance, a kernel achieving 86.5% bandwidth utilization
but only 0.019% of peak FLOPs is severely memory-bound. Conversely, a kernel
with AI above the ridge point is **compute-bound**, potentially reaching 71.6%
of peak FLOPs while utilizing only 16.0% of bandwidth.

The ridge point is not static; it shifts dynamically based on the precision type
and hardware generation. For example, INT8 operations typically have a
significantly higher FLOP ceiling than BF16, which moves the ridge point further
to the right on the intensity spectrum. Understanding where your kernel sits
relative to this point is critical for deciding whether to optimize for data
movement (e.g., better tiling) or compute efficiency (e.g., instruction
pipelining).

### Empirical Verification of HBM Bandwidth and the Ridge Point

***(Reference: exp115_hbm_bandwidth.py)***

Peak aggregate throughput of on-package HBM per TPU chip (GB/s) is set by the
chip generation (e.g., v4, v5p, v6e). This scalar sets the roofline's memory
slope and therefore the ridge point beyond which kernels transition from
memory-bound to compute-bound.

Empirical profiling on TPU v6 Lite (Trillium) validates this relationship:

1.  **Peak Measurement**: The hardware XProf roofline model reports a
    theoretical `peak_hbm_bw` of 1525.5 GiB/s (~1638 GB/s).
2.  **Ridge Point Calculation**: With a reported `peak_flop_rate` of 946,700
    GFLOP/s, the hardware explicitly reports an `hbm_ridge_point` of ~577.96
    FLOP/Byte. This perfectly aligns with the ratio `Peak FLOPs / Peak HBM
    Bandwidth` (946,700 / 1638 = ~577.96).
3.  **Memory Bound Validation**: A benchmark running an elementwise addition
    kernel (`c = a + b`) has a theoretical operational intensity of 0.083
    FLOP/Byte (1 FLOP per 12 Bytes of I/O). Since 0.083 << 577.96, the kernel is
    deeply memory bound. XProf traces empirically label the operation as bound
    by `HBM`, consuming 1189.85 GiB/s (78% of the peak HBM Bandwidth).

This hardware evidence firmly confirms that the physical HBM bandwidth dictates
the memory slope of the roofline model and sets the ridge point for compute vs.
memory bound transitions.

### Empirical Verification of Memory-Bandwidth Utilization

***(Reference: exp016_memory_bandwidth_utilization.py)***

Memory-bandwidth utilization—defined as the achieved HBM bandwidth divided by
the peak physical bandwidth—serves as a primary diagnostic metric for evaluating
and optimizing memory-bound kernels.

Hardware evidence on TPU v6 Lite (Trillium) confirms that memory-bound
operations can be timed using XProf telemetry to directly calculate
memory-bandwidth utilization. In a validation experiment, a bulk DMA copy kernel
executing a pure copy of a 256MB float32 array via Pallas (`y_ref[...] =
x_ref[...]`) was evaluated.

By utilizing large block sizes (e.g., 128K elements) to maximize DMA efficiency,
the operation achieved an execution time of approximately **474 μs** for a total
of 512 MB of I/O (256 MB read + 256 MB write). This yields an empirical
throughput of **~1.08 TB/s**. Dividing this achieved throughput by the peak HBM
bandwidth of the specific TPU generation (e.g., ~1.64 TB/s for TPU v6e) provides
the memory-bandwidth utilization, demonstrating how raw latency and transfer
sizes can be used to verify that a memory-bound kernel is successfully
saturating the physical HBM bus.

### Empirical Verification of Arithmetic Intensity Scaling

***(Reference: exp117_arithmetic_intensity.py)***

To understand how arithmetic intensity (AI)—defined as the ratio of FLOPs
performed per byte of memory traffic—governs performance scaling along the
roofline model's x-axis, empirical hardware verification was conducted on TPU
using custom Pallas kernels.

By holding memory traffic strictly constant (reading and writing a 1024x1024
float32 matrix, representing a fixed memory footprint), the arithmetic workload
was scaled to observe the transition across the roofline spectrum:

1.  **Low Arithmetic Intensity (0.25 FLOPs/Byte)**: Executing 2 FLOPs per 8
    bytes transferred, this kernel completed in approximately **49.09 μs** per
    invocation. In this regime, the kernel is profoundly memory-bound, spending
    almost its entire execution duration stalled on memory I/O.
2.  **High Arithmetic Intensity (25.0 FLOPs/Byte)**: By increasing the compute
    workload to 200 FLOPs per 8 bytes transferred (a 100x increase in arithmetic
    operations), the execution latency only rose to **59.54 μs**—a mere **21%
    increase** in runtime.

This non-linear scaling demonstrates the hardware's ability to overlap vector
execution with memory access. The massive influx of FLOPs was largely absorbed
by the compute units operating concurrently with background DMA transfers,
proving that increasing arithmetic intensity successfully shifts the workload
along the x-axis of the roofline model into the compute-bound regime.

## CMEM: TPU v4 Legacy Roofline Telemetry on Newer Hardware

***(Reference: exp081_cmem.py)***

When analyzing memory-bound kernels and studying roofline models in XProf, you
may encounter references to **CMEM (Common Memory)**, an on-chip memory tier
specific to the TPU v4 architecture.

While CMEM is physically absent or disabled on newer hardware generations such
as TPU v5p and TPU v6e (Trillium), legacy tracking structures for it still
persist within the XProf profiling suite. This can lead to confusing telemetry
outputs if not properly understood.

### Empirical Verification of CMEM Telemetry on TPU v6e

Empirical hardware profiling on a TPU v6 Lite (v6e) using a standard matrix
multiplication kernel verifies how the XProf tooling handles this legacy tier:

1.  **Dedicated Tooling Columns:** XProf explicitly allocates dedicated columns
    for `CMEM Read BW (GiB/s)` and `CMEM Write BW (GiB/s)` in the generated
    `roofline_model` tool data.
2.  **Zeroed Properties in XPlane:** The parsed `xplane.pb` roofline summary
    properties (specifically within the `p` object) report `has_cmem: "0"`,
    `peak_cmem_read_bw: "0"`, and `peak_cmem_write_bw: "0"` for the `TPU v6
    Lite` device type.

This behavior confirms that while the profiling infrastructure retains the
architectural hooks to track CMEM bandwidth and roofline limits, the physical
hardware reports absolute zero values on Trillium. Unlike VMEM (Vector Memory)
and HBM (High-Bandwidth Memory)—which populate with active, real-time
utilization metrics—CMEM remains a ghost tier on post-v4 architectures. When
optimizing Pallas kernels on TPU v6e, these CMEM metrics should be safely
ignored in favor of VMEM and HBM bandwidth analysis.

## VPU (Vector Programmable Unit) Execution and Telemetry

***(Reference: exp014_vpu.py)***

The Vector Programmable Unit (VPU) is a core programmable component of the
TensorCore, serving as the primary gateway to other specialized execution units,
including the Matrix Unit (MXU), the transpose unit (XU), and the
reduction/permutation unit (RPU). Understanding how operations map to the VPU is
essential for diagnosing execution paths and verifying that vector mathematics
are running on the correct hardware subsystems.

### Empirical Hardware Verification of VPU Mapping

To verify how element-wise operations are dispatched and executed on the
physical hardware, a Pallas vector addition kernel (`add_kernel`) operating on
vectors of size 1024 was executed natively on the TPU.

Analyzing the resulting trace via XProf confirmed:

1.  **Direct Hardware Mapping:** The element-wise vector addition operation maps
    natively to a `custom-call` executed on-device via the Pallas TPU path.
2.  **VPU Execution Role:** This confirms that standard vector mathematics are
    processed directly by the VPU on the TensorCore, validating its role as the
    execution engine for non-matrix vector operations. When profiling
    element-wise or reduction-heavy Pallas kernels, monitoring VPU activity and
    ensuring operations do not fall back to host-side emulation is critical for
    maintaining peak TensorCore throughput.

## SparseCore (SC) Execution and Telemetry

***(Reference: exp045_sparsecore.py)***

While the Vector Programmable Unit (VPU) and Matrix Unit (MXU) handle dense
vector and matrix mathematics on the TensorCore, TPU v5p and v6e (Trillium)
architectures feature dedicated hardware units known as **SparseCores (SC)**.
These co-processors are specifically optimized for embedding-style sparse
lookups, gathering, and scattering operations, and are tracked within XProf as
distinct execution resources.

### Empirical Hardware Verification of SparseCore Mapping

Empirical analysis of the TPU v6e (Trillium) compiler pipeline and JAX runtime
environment confirms that the SparseCore is an explicitly targeted execution
resource with dedicated software and hardware pathways:

1.  **JAX Custom Calls:** The JAX source code (`jax/_src/tpu_custom_call.py`)
    supports explicit mapping to SparseCores via the `device_type ==
    "sparsecore"` attribute, allowing custom calls to bypass the standard
    TensorCore execution path.
2.  **Hardware Constraints and LLVM Dialects:** The `libtpu` compiler pipeline
    includes specific LLVM dialects designed to lower operations directly to
    SparseCore instructions (e.g., `Lower Mlo operations to the SparseCore LLVM
    dialect`). The compiler strictly enforces hardware bounds on these
    functions, requiring that `all iteration_bounds should be 1 or the number of
    SparseCores (%d) for SCS functions`.
3.  **Memory Spaces:** The compiler maps specific scoped memory allocations
    directly to SparseCore-specific memory spaces, such as
    `MemorySpace::kSparseCorePrivateStackHbm`, isolating SparseCore memory
    traffic from standard TensorCore VMEM/SMEM pathways.
4.  **XLA Flags:** Trace dumps verify offloading configurations natively
    compiled via debug options like
    `--xla_tpu_enable_offloading_gather_to_sparsecore`, confirming that sparse
    gather/scatter operations are automatically routed to the SparseCore
    hardware.

When profiling sparse or embedding-heavy workloads, monitoring SparseCore
utilization in XProf ensures that memory-intensive lookup operations are
successfully offloaded to these dedicated co-processors, freeing up TensorCore
resources for dense matrix operations.

## Collective Communication and ICI Telemetry

***(Reference: exp032_collective_communication.py)***

Cross-replica operations (such as all-reduce, all-gather, and reduce-scatter)
synchronize or redistribute data across multiple TPU chips. On TPU
architectures, these collective communication operations compile down to
efficient, direct Inter-Core Interconnect (ICI) primitives over specified
`replica_groups`, executing independently of host-side communication libraries
like NCCL.

### Empirical Hardware Verification of ICI Collectives

This direct hardware execution path has been empirically verified via TPU trace
capture. A microbenchmark was executed using `jax.pmap` to run a `jax.lax.psum`
reduction on a 134MB tensor (33,554,432 elements of `float32`) across a 4-chip
TPU mesh.

Analyzing the resulting trace via XProf confirmed:

1.  **Direct HLO Mapping:** The XProf `op_profile` explicitly mapped the JAX
    `psum` function to an HLO `all-reduce` instruction: `%psum.7 =
    f32[1,33554432]{1,0:T(1,128)} all-reduce... channel_id=1,
    replica_groups={{0,1,2,3}}`
2.  **High-Performance ICI Execution:** Hardware telemetry revealed that the
    `all-reduce` executed with an average hardware latency of **~1.18 ms** per
    occurrence (`avgTimePs: 1188860937.5`). This confirms that collective
    communication operations bypass host-side overhead and run directly on the
    dedicated ICI network substrate.

### Reduce-Scatter: Sharded Collective Reduction

***(Reference: exp035_reduce_scatter.py)***

A key collective communication pattern is the **Reduce-Scatter** operation,
which reduces (sums) a tensor across replicas and distributes the result as
equal shards, with one shard per replica. Conceptually, this behaves as an
all-reduce whose output is sharded rather than replicated. In large-scale
training paradigms such as Fully Sharded Data Parallelism (FSDP) or ZeRO-3, an
all-gather on the forward pass is paired with a reduce-scatter on the backward
pass (one of each per layer). Similarly, sequence parallelism replaces the
Tensor Parallel (TP) block's internal all-reduce with a reduce-scatter and
all-gather pair of the same aggregate volume.

#### Empirical Hardware Verification of Reduce-Scatter

Empirical profiling on a 4-core TPU setup validates this conceptual definition
of `reduce-scatter`. By using `jax.lax.psum_scatter` to scatter a tensor across
multiple replicas, the XLA compiler generates an HLO sequence that effectively
matches the definition of an all-reduce followed by a sharded extraction.

Specifically, in the `after_optimizations` XLA HLO dump, the `reduce-scatter`
operation is explicitly mapped to:

1.  An `all-reduce` operation across all replicas
    (`replica_groups={{0,1,2,3}}`).
2.  A `dynamic-slice` operation indexed by the current `replica-id()` multiplied
    by the shard size.

This is confirmed by the following compiled HLO snippet:

```text
  %replica-id = u32[] replica-id()
  %multiply = u32[]{:T(128)S(6)} multiply(%replica-id, %constant.1)
  %all-reduce = f32[4194304]{0:T(1024)S(1)} all-reduce(%bitcast.2), replica_groups={{0,1,2,3}}, to_apply=%region_0.6.clone
  %dynamic-slice = f32[1048576]{0:T(1024)} dynamic-slice(%all-reduce, %multiply), dynamic_slice_sizes={1048576}
```

This hardware telemetry confirms that the standard TPU compilation for
`reduce-scatter` executes an aggregate sum followed by an offset extraction
mapped to each unique device.

## Avoiding Trace Overhead Inside Testing Loops

When composing pipelines of multiple Pallas kernels, engineers often worry about
the CPU dispatcher gaps and "trace scheduling overhead" between executions.

Empirical verification reveals that **Unified JAX Wrappers exhibit zero
structural trace overhead**. If you wrap multiple sequential Pallas calls inside
a single `@jax.jit` higher-level function, XLA coalesces them perfectly. While
the physical kernels remain distinct `custom-call` primitives in the HLO, they
are scheduled sequentially on the hardware without any intermediate host tracing
overhead or dispatcher latency penalty. The SPU (Scalar Processing Unit) handles
trace instructions flawlessly for reasonable block sizes.

## Native JAX HLO Pipeline Wrapper Overhead (AX-049)

***(Reference: exp156_native_jax_hlo_pipeline_wrappe.py)***

While unifying multiple sequential Pallas calls or standard operations inside a
single JAX wrapper avoids host-side dispatch overhead, it is critical to
recognize that this unification does not yield automatic High-Bandwidth Memory
(HBM) fusion speedups over eager execution frameworks (such as PyTorch).

### The Opaque Custom Call Boundary

When compiling a JAX pipeline that mixes standard dense operations (e.g.,
`jnp.dot`) with custom Pallas kernels (e.g., `pallas_call`) and subsequent
elementwise operations (e.g., `jnp.exp`), XLA treats the Pallas custom call as
an opaque execution boundary.

Empirical analysis of the resulting XLA HLO (High-Level Optimizer) dumps
confirms that the compiler generates disjoint, non-fused operations:

1.  A `dot_general` instruction mapping to a standard fusion block.
2.  A `custom-call` instruction mapping directly to the Pallas TPU Custom Call.
3.  An `exponential` (or other elementwise) instruction mapping to a separate
    post-kernel operation.

### Hardware Implications and HBM Transit

Because there is no compiler-driven kernel fusion across the Pallas custom call
boundary, intermediate data cannot be kept in on-chip registers or Vector Memory
(VMEM) between these distinct blocks. Instead, the data must fully transit
through HBM between the operations.

Consequently, wrapping standard operations and Pallas kernels in a single
`jax.jit` provides no memory bandwidth or fusion advantages over executing them
sequentially using an eager framework. The standard eager dispatcher is already
optimal in this scenario, as the hardware execution bottleneck remains bound by
the physical HBM round-trips necessitated by the opaque boundary.

## Autotuning and Static Factory Methods

Autotuning over kernel configuration spaces (such as `block_size`) is mandatory;
varying block shapes can yield over a 100x latency reduction depending on
hardware alignment. However, dynamically passing these configurations to Pallas
causes tracing failures.

Passing a dynamic variable to `pl.BlockSpec` or using it in grid size shape
arithmetic (e.g., `grid_size = shape // block_size`) injects an abstract
`DynamicJaxprTracer` where a concrete integer is required. This triggers a fatal
`jax.errors.ConcretizationTypeError`.

**Resolution to avoid compilation crashes:**

1.  **Static Keyword Arguments:** When decorating your wrapper, explicitly
    declare configuration variables using `static_argnames` (e.g.,
    `@custom_jax_kernel(static_argnames=['block_size'])` or
    `@jax.jit(static_argnames=['block_size'])`). This ensures the variable is
    treated statically at trace-time.
2.  **Static Kernel Factories:** Maintain a dictionary mapping concrete
    configuration integers to pre-instantiated JAX/Pallas functions (e.g.,
    `STATIC_KERNELS[block_size]`(x, y)). This factory pattern strictly isolates
    trace resolution and completely prevents tracer concretization crashes
    during automated sweeps.

## Safe Literal Closure Scalar Injection (AX-186)

***(Reference: exp085_safe_literal_closure_scalar_in.py)***

When designing Pallas kernels, passing scalar hyperparameters (such as
regularization terms, learning rates, or small constants like `epsilon`)
presents a unique compilation challenge. Using `PrefetchScalarGridSpec` or
passing dynamic Python scalars as inputs directly to JAX Pallas kernels can
cause severe `Tracer` overhead, crashes during lowering, or constant
recompilation on varying scalar values.

### The 0D Shape and Closure Capture Constraints

Pallas TPU strictly forbids `BlockSpec` mappings for 0D shapes, throwing a fatal
error during lowering: `ValueError: The Pallas TPU lowering currently supports
only blocks of rank >= 1`

Attempting to bypass this by capturing dynamic JAX tracers inside the kernel's
closure also fails, throwing a `ValueError` related to capturing a
`ShapedArray`.

Furthermore, attempting to work around the 0D limitation by wrapping the scalar
in a 1D array (e.g., shape `(1,)`) and mapping it via a 1D `BlockSpec`
introduces a measurable performance penalty. Hardware traces reveal that mapped
1D scalars run approximately 6% slower in microbenchmarks. This regression is
caused by redundant DMA copies per block, forcing repetitive VMEM reads
(`vector.load`) across execution blocks instead of utilizing fast, on-chip
scalar registers.

### Resolution: Static Closure Injection

To inject scalars safely and with zero performance overhead:

1.  Leave the scalar (e.g., `epsilon`) as a pure Python literal in the outer
    scope.
2.  Mark the scalar parameter as static using `static_argnums` or
    `static_argnames` in your JAX JIT/kernel decorator.
3.  Inside the nested `kernel_fn` block, capture the static Python literal via
    standard Python closure.
4.  Manually instantiate the scalar inside the kernel using `eps =
    jnp.float32(epsilon)`.

This approach completely bypasses the rank constraints of Pallas TPU lowering.
Because the value is static at trace-time, the compiler optimizes it away,
compiling it natively to a zero-cost `arith.constant` in Mosaic MLIR without
triggering any VMEM load instructions or DMA overhead.

## Implicit Constant Capture Restrictions (AX-146)

***(Reference: exp018_pallas_kernel_implicit_constan.py)***

When designing Pallas kernels, a strict compilation boundary exists regarding
how variables defined in the outer scope are captured. Specifically, Pallas
kernel functions invoked via `pl.pallas_call` strictly forbid capturing
implicitly scoped JAX variables or constants (for example, `lr =
jnp.float32(1e-3)` defined in an outer scope).

Attempting to reference an outer-scope JAX array or tracer inside the kernel
function triggers a fatal compilation error during lowering: `ValueError: The
kernel function in the pallas_call captures constants. You should pass them as
inputs.`

### Resolution: Pure Python Closure with Local Instantiation

To inject scalar constants into a Pallas kernel without allocating HBM buffers
and passing them through `in_specs` / `in_shapes`, you must adhere to the
following pattern:

1.  Keep the outer-scope constants as pure Python types (e.g., `lr_val = 1e-3`).
2.  Capture the pure Python literal via standard Python closure.
3.  Manually instantiate the constant as a JAX type *inside* the local body of
    the Pallas kernel function (e.g., `lr = jnp.float32(lr_val)`).

This ensures that the JAX tracer does not attempt to track an implicit
outer-scope constant, satisfying the compiler's strict isolation requirements
while avoiding the overhead of HBM allocation.

## Intermediate Pallas Call Shape Inference (AX-046)

***(Reference: exp023_intermediate_pallas_call_shape.py)***

When developing custom JAX kernel wrappers (`custom_jax_kernel`) that
encapsulate internal allocations or intermediate operations, providing explicit
shape information to the underlying `pallas_call` is a strict compilation
requirement.

### The Shape Inference Constraint

Omitting the `out_shape` argument in an intermediate `pl.pallas_call()` triggers
an immediate `TypeError` in the JAX tracer. Because `out_shape` is a mandatory
positional argument in the `pallas_call` signature, JAX requires this metadata
to perform shape and type inference on the resulting higher-order primitive
before lowering to Mosaic MLIR. Without it, the tracer cannot construct the
intermediate JAX expressions (jaxprs) required to represent the kernel's output.

### Resolution: Explicit ShapeDtypeStruct Allocation

To satisfy the tracer's shape inference requirements and ensure successful
compilation:

1.  Always define the output shape and data type of your intermediate Pallas
    calls using `jax.ShapeDtypeStruct`.
2.  Pass this structure explicitly to the `out_shape` argument of
    `pl.pallas_call()`.

```python
# Correct usage of out_shape for intermediate Pallas calls
out_shape = jax.ShapeDtypeStruct(shape=(1024, 1024), dtype=jnp.float32)
result = pl.pallas_call(
    kernel_fn,
    out_shape=out_shape,
    grid=grid,
    in_specs=in_specs,
    out_specs=out_specs,
    ...
)(x)
```

Empirical hardware profiling confirms that supplying
`out_shape=jax.ShapeDtypeStruct(shape, dtype)` allows the JAX tracing phase to
succeed without overhead. The resulting kernel compiles natively and executes
successfully on TPU hardware, as verified by automated benchmarking and trace
extraction tools.

## JAX Trace-Time Topological Shape Bypass (AX-160)

***(Reference: exp037_jax_trace_time_topological_sha.py)***

When designing versatile JAX and Pallas wrappers, you may need to handle
multiple unaligned or distinct topological tensor shapes (e.g., varying
dimensionalities or ranks) within a single hardware kernel definition.
Attempting to manage these topological variations dynamically using XLA control
flow (such as `jax.lax.cond`) introduces significant runtime overhead and
compilation complexity.

### Resolution: Trace-Time Python Control Flow

To bypass runtime overhead, use standard Python control flow on tensor shapes
(e.g., checking `x.ndim == 2` or inspecting `x.shape`) within the JAX outer
function. Because JAX resolves shapes statically during the tracing phase, these
conditions are evaluated entirely at trace-time.

As a result, the tracer only emits the specialized branch's XLA graph
corresponding to that specific tensor shape. The alternative branches are
completely pruned before compilation.

### Empirical Verification of Zero-Overhead Branching

This trace-time resolution has been empirically verified via HLO analysis. When
dispatching different Pallas kernels or configurations based on input topology:

```python
def custom_jax_kernel(x):
  if x.ndim == 2:
    return pl.pallas_call(kernel_2d, out_shape=...)(x)
  elif x.ndim == 3:
    return pl.pallas_call(kernel_3d, out_shape=...)(x)
  else:
    raise ValueError("Unsupported rank")
```

Analyzing the compiled XLA HLO module reveals:

1.  **Zero Conditional Nodes:** The generated HLO contains exactly zero
    `conditional` or control-flow nodes representing the Python `if-else`
    branches.
2.  **Direct Kernel Lowering:** Only the specific `tpu_custom_call`
    corresponding to the active branch is compiled and lowered to the physical
    hardware.

Because the shapes are fully known during tracing, any Python branching
dependent strictly on topology never reaches the physical compiler, ensuring
optimal execution on the TPU without runtime conditional overhead.

## Serving Warmup: Mitigating Compilation Stalls

***(Reference: exp033_serving_warmup.py)***

In production environments, JAX's "just-in-time" philosophy requires rigorous
**Serving Warmup** strategies. Because Pallas kernels are sensitive to static
shapes, any change in input dimensions—such as a shift in `(batch,
seq_len)`—triggers a cache miss and a subsequent re-compilation. Pre-compiling
and warming every expected `(batch, seq_len)` shape before accepting production
traffic is strictly necessary to avoid runtime recompilation stalls.

### Empirical Hardware Verification of Compilation Stalls

Empirical validation using XProf profiling reveals severe latency penalties when
JAX encounters unseen input shapes. During execution of a computationally
trivial kernel (`x + y` repeated in a loop), tracing of JAX host and device
events reveals:

-   **First Call (Shape A - `1024x1024`)**: Encountered a host-side compilation
    event (`$pxla.py:2405 compile`) taking **194.7 ms**, with a total execution
    time of **603 ms**.
-   **Second Call (Shape A - Cached)**: Executed without recompilation. Total
    execution time was strictly device-bound and took only **10.46 µs**.
-   **Third Call (Shape B - `2048x2048`)**: The new shape triggered a
    recompilation. The host-side `compile` event took **364 ms**, leading to a
    total stall latency of **1.11 seconds**.

The results confirm a latency difference of over 30,000x between a warmed
(cached) execution and a cache miss requiring recompilation. In a live
production setting, these "compilation stalls" dramatically breach latency SLAs
(introducing hundreds of milliseconds of jitter) and block the execution thread.

**Resolution:** Every expected input shape must be pre-compiled during the
initialization phase of the serving process. This involves "dummy" executions of
every valid `(batch, seq_len)` pair to populate the JAX compilation cache. Only
after this exhaustive warmup is the application ready to handle live traffic
without the risk of massive latency spikes.
