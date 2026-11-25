# Task: Onboard and Hill-Climb GLM-5.3-Flash on TPU via torch-tpu

Execute this task in **two strictly sequential phases**:

1. **Phase 1 — Onboarding (Baseline Bring-Up):** Complete Phase 1 end-to-end first, **identically to a standalone onboarding run**. Do **not** start hill-climbing, custom kernel authoring, or experimental optimization sweeps during Phase 1. Once Phase 1's Definition of Done holds, freeze your Phase 1 serving config, raw benchmark outputs, reproduction commands, and markdown report (`onboarding/REPORT.md`) under `onboarding/` in your artifact directory (and mirror to `/logs/artifacts/onboarding/` if `/logs/artifacts` exists).
2. **Phase 2 — Throughput Hill-Climbing (Maximize `output_tps_per_chip`):** Starting from your frozen Phase 1 deployment, iteratively profile and optimize the serving stack to maximize **per-chip output throughput (`output_tps_per_chip`)** on the `8k/1k` workload, capture a **post-hill-climb XProf profile** to document remaining bottlenecks and roofline gaps, and write `benchmark_results.json`.

## Phase 1: Onboarding

Onboard zai-org/GLM-5.3-Flash (https://huggingface.co/zai-org/GLM-5.3-Flash) to run as an inference server on Cloud TPU, using the `torch_tpu` backend.
Model: zai-org/GLM-5.3-Flash

### Phase 1 Definition of done

Phase 1 is done when all eight hold:

1. A vLLM server is running on TPU hardware serving the model.
2. It answers a completion request over HTTP with coherent output, and you show and save the request and the raw JSON response under `onboarding/` (for example in `onboarding/REPORT.md` or a raw response file).
3. You report measured performance in markdown tables (format below) — real numbers from your run, not estimates.
4. You give the exact commands to reproduce the result from scratch.
5. Your code changes are a series of local git commits, and the log reads as the story of how you got there.
6. Every artifact is retained locally, with versions pinned (see below).
7. Within the harness, use Opus 5 (1M) context to do this task (when applicable to your harness; otherwise use the harness's configured primary model).
8. Use a tpu-v7 slice when selecting hardware on a shared cluster, or the TPU slice your environment is pre-configured for. Check the environment for the correct cluster.

If you cannot reach a running server, say so plainly and report how far you got, what blocked you, and what you ruled out. A truthful partial result is worth more than a claimed success.

### Phase 1 Artifacts and versions

Everything you produce must survive the run. A result that cannot be inspected or reproduced afterwards does not count.

- **Keep artifacts on local disk**, under one directory, organised and named so a reader can tell what each file is. Not scattered in `/tmp`, not left inside a container that gets torn down, not only in a scrollback buffer.
- **Retain at minimum**: raw benchmark output (not just the summary numbers), server logs, any generated or patched source, the serving config, and any profiles you captured.
- **Pin every version you depended on**, recorded in one file alongside the artifacts:
  - container image — the **digest**, not just the tag
  - `torch_tpu` build / wheel version
  - vLLM version
  - Tokamax commit SHA, if used
  - Hugging Face model revision (the commit hash, not `main`)
  - TPU runtime / accelerator type and topology
- **Every benchmark row must be traceable** to the raw output it came from and the version set it ran against. State which.
- Do not commit credentials or model weights. If ADC or a token is mounted, keep it out of the artifact directory and out of git history.

### Phase 1 Reporting benchmarks

Report numbers in tables (saved in `onboarding/REPORT.md` as well as your final report), not prose. One row per configuration you measured.

| Config | Topology | dtype | Batch | In/Out tokens | TTFT (ms) | ITL (ms) | Throughput (tok/s) |
|---|---|---|---|---|---|---|---|

Rules for the table:

- Every number is measured. If a cell is an estimate or a single sample, mark it. If you did not measure it, leave it blank — never fill it in plausibly.
- State how many runs each number averages, and the variance if you have it.
- If you measured a baseline (CPU, or the model before your changes), give it its own row so the comparison is visible rather than asserted.

Put failures in a second table — what you tried, what happened, whether you resolved it:

| Attempt | Config | Outcome | Resolution |
|---|---|---|---|

## Phase 2: Throughput Hill-Climbing (`output_tps_per_chip`)

Start Phase 2 **only after** Phase 1 is complete and its baseline artifacts are saved under `onboarding/` (and mirrored to `/logs/artifacts/onboarding/` if `/logs/artifacts` exists). Starting from your finished Phase 1 baseline serving recipe and deployment, run a structured, hypothesis-driven optimization loop on the compute backend used in Phase 1 to maximize per-chip output throughput (`output_tps_per_chip`). Save Phase 2 artifacts under `hillclimb/` without overwriting Phase 1 files.

### 1. Optimization Metric & Workload (`output_tps_per_chip`)

- **Primary metric to maximize:** **`output_tps_per_chip = output_tps / num_chips`** (output tokens/second per physical TPU chip).
  - **`num_chips` definition:** Total **physical TPU chips** allocated to the deployment (on `tpu7x`, 1 physical chip = 2 logical TensorCores/devices, so `2x2x1` = `4` physical chips, `2x2x2` = `8` physical chips, and `2x2x4` = `16` physical chips; for disaggregated prefill/decode, `num_chips = prefill_chips + decode_chips`). Never divide by logical TensorCore count on `tpu7x`.
  - **Efficiency rule:** Rank, select, and stack optimizations strictly by **`output_tps_per_chip`**, not plain `output_tps`. Scaling to more chips without improving `output_tps_per_chip` is a regression, and leaving allocated chips on a slice idle (e.g., `TP` smaller than the slice without `DP` replicas) penalizes `output_tps_per_chip`.
- **Target serving benchmark (`8k/1k`):** Benchmark over HTTP with random fixed-length prompts (`ISL = 8192`, `OSL = 1024`, `--ignore-eos` so every request generates 1024 tokens, **prefix caching disabled**, `--max-model-len >= 9216`). Sweep every consecutive power-of-2 concurrency `1, 2, 4, 8, 16, 32, 64, 128, ...` without skipping intermediate powers of 2 (`>= 4x` concurrency prompts per point) until `output_tps_per_chip` improves `< 10%` over the previous concurrency or requests fail/OOM.
  - **Paired baseline check:** If Phase 1 did not already run this exact fixed-length `8k/1k` sweep, measure your **unmodified Phase 1 server** on this `8k/1k` sweep first before applying any Phase 2 optimization so `"baseline"` and `"optimized"` compare on identical points.

### 2. Hill-Climbing Loop

1. **Roofline & Baseline Profiling Triage:** Compute the theoretical hardware roofline (`output_tps` and `output_tps_per_chip`) for GLM-5.3-Flash on the target TPU slice at `8k/1k`, capture a baseline XProf trace (`.xplane.pb`) of the running server at peak baseline concurrency, and rank the primary bottlenecks and optimization hypotheses.
2. **Hypothesis-Driven Iteration:** Propose, test, and stack improvements across rounds. Any legitimate optimization that preserves output quality is in scope:
   - Sharding, mesh layout, and parallelism (`TP`, `DP`, `EP`, `DCP`, multi-host placement, or disaggregated `1P1D`)
   - `vLLM` serving, scheduler, bucketing, and KV-cache configuration (`--max-num-batched-tokens`, `--max-num-seqs`, chunked prefill, KV dtype, speculative/MTP decoding when supported by model weights)
   - Compiler and runtime flags (`LIBTPU_INIT_ARGS`, `XLA_FLAGS`, static compile scope, compile cache, host-sync/copy elimination)
   - Custom or tuned Pallas / Tokamax kernels, verified for numerical parity (`L_inf <= 1e-2`, `L2 <= 1e-3`, zero NaNs/Infs) before serving integration
3. **Ledger & Quality Gate:** Log every tested hypothesis arm (change, `num_chips`, `output_tps`, `output_tps_per_chip`, TTFT, ITL, parity/accuracy result, and keep/discard reason) in `experiment_log.md`. Verify accuracy/quality on the winning stack (for example via reference parity or standard evaluation prompts). Reserve sufficient wall-clock time before session timeout to finish Steps 4 and 5 cleanly.
4. **Final Comparative Sweep:** Re-run the identical `8k/1k` concurrency sweep on the final winning stack across every Phase 1 baseline concurrency point plus any higher concurrencies unlocked before saturation.
5. **Post-Hill-Climb XProf Capture & Bottlenecks/Gaps Report:** **Before tearing down the final winning server**, capture a **post-hill-climb XProf trace (`.xplane.pb`)** at peak `output_tps_per_chip` concurrency. Save the `.xplane.pb` file (and XProf / ML Diagnostics URL if available) under `hillclimb/xprof/` and write **`xprof_bottlenecks_and_gaps.md`** in your artifact directory covering:
   - **Roofline & Efficiency Summary:** Baseline vs. Optimized `output_tps_per_chip` vs. theoretical roofline `output_tps_per_chip` (`% roofline` achieved and remaining gap).
   - **Post-Hill-Climb Device Breakdown:** Top-5 operators/kernels by `%` of TPU step time (prefill and decode), MXU/VPU compute vs. HBM bandwidth vs. ICI/DCN collective time vs. host/scheduler gaps, and how bottlenecks shifted from the baseline XProf trace.
   - **Ranked Residual Gaps:** Actionable list of remaining bottlenecks limiting `output_tps_per_chip` (kernel, sharding/collectives, vLLM framework, or compiler/platform) and why they remain open.

### 3. Phase 2 Reporting & `benchmark_results.json`

Report a same-concurrency comparison table in Markdown (one row per measured concurrency point):

| Regime | Concurrency | Baseline Chips | Opt Chips | Baseline `output_tps` | Opt `output_tps` | Baseline `output_tps_per_chip` | Opt `output_tps_per_chip` | Per-Chip Speedup | Optimization Applied |
|---|---|---|---|---|---|---|---|---|---|

Write **`benchmark_results.json`** in your artifact directory (and mirror to `/logs/artifacts/benchmark_results.json` if `/logs/artifacts` exists). Every benchmark entry **must** include raw numeric values for **`"output_tps"`**, **`"num_chips"`**, and **`"output_tps_per_chip"`** (illustrative schema example):

```json
{
  "optimization_metric": "output_tps_per_chip",
  "baseline": [
    {
      "config": "Baseline",
      "topology": "2x2x1",
      "num_chips": 4,
      "dtype": "bf16",
      "regime": "8k/1k",
      "concurrency": 16,
      "ttft_ms": 310.4,
      "itl_ms": 12.3,
      "output_tps": 1180.0,
      "output_tps_per_chip": 295.0,
      "raw_output_path": "onboarding/raw_benchmarks/bench_8k1k_c16.json"
    }
  ],
  "optimized": [
    {
      "config": "Optimized",
      "topology": "2x2x1",
      "num_chips": 4,
      "dtype": "bf16",
      "regime": "8k/1k",
      "concurrency": 16,
      "ttft_ms": 195.0,
      "itl_ms": 8.5,
      "output_tps": 1845.0,
      "output_tps_per_chip": 461.25,
      "raw_output_path": "hillclimb/raw_benchmarks/bench_8k1k_c16.json"
    }
  ],
  "comparison": [
    {
      "regime": "8k/1k",
      "regime_label": "Prefill",
      "concurrency": 16,
      "num_chips": 4,
      "baseline_num_chips": 4,
      "optimized_num_chips": 4,
      "baseline_tps": 1180.0,
      "optimized_tps": 1845.0,
      "speedup_ratio": 1.5636,
      "baseline_tps_per_chip": 295.0,
      "optimized_tps_per_chip": 461.25,
      "per_chip_speedup_ratio": 1.5636,
      "primary_bottleneck_op": "...",
      "optimization_applied": "..."
    }
  ],
  "roofline": {
    "regime": "8k/1k",
    "concurrency": 16,
    "num_chips": 4,
    "output_tps": 2650.0,
    "output_tps_per_chip": 662.5,
    "achieved_roofline_pct": 69.62,
    "source": "hillclimb/roofline_report.md"
  },
  "post_hillclimb_xprof": {
    "regime": "8k/1k",
    "concurrency": 16,
    "trace_path": "hillclimb/xprof/final_optimized.xplane.pb",
    "bottlenecks_and_gaps_report": "xprof_bottlenecks_and_gaps.md",
    "top_bottleneck_ops": ["..."]
  },
  "failures": []
}
```

## Shared Rules, Constraints & Environment (Both Phases)

### How to work

- **Commit as you go, locally.** Each commit should be one coherent step that builds or runs. Do not deliver one large commit at the end.
- **The commit message carries the reasoning** — what changed and why it was necessary. This is where the narrative lives.
- **Comment the flow in code where it is not obvious**: why a kernel was swapped, why a shape is padded, why an op was replaced. Explain *why*, not *what* — a line restating the code is noise. Keep it short.
- No need to push anything. Local history is the deliverable.
- Proceed autonomously without waiting for user confirmation. Never run shell `sleep` commands or polling loops of your own; wait only with bounded wait commands that time out and return on their own (such as a job wait with a timeout). A full or auto-provisioning cluster is not a failure: if a submitted TPU job is queued or pending while capacity scales up or frees, leave it queued and keep re-running bounded waits until your job is admitted and finishes.

### Hard constraints

- **Use `torch_tpu` with `vLLM`.** Do NOT use `torch_xla` / `pytorch_xla`, do NOT use the `tpu-inference` repo, and do NOT fall back to toy HTTP wrappers (`FastAPI`, `Flask`, single-stream loops).
- **Do not reuse existing work.** Write everything this task needs yourself, except for packaged patches from the `vllm_patches` directory (use a matching patch if present; the directory or a patch for this model may not exist).
  - You MAY read and use: the Hugging Face model card and config, the pre-staged Hugging Face checkpoint (see Environment & Allowances), the `torch_tpu` and `vLLM` source inside the provided container, Tokamax, public vLLM/TPU documentation, and packaged patches from the `vllm_patches` directory (if available).
  - Except for packaged patches from the `vllm_patches` directory and the pre-staged Hugging Face checkpoint, you MUST NOT read or copy onboarding or optimization code, scripts, notebooks, configs, or output artifacts that already exist on any VM, disk, bucket, or container you did not create during this task. Reading such material is a FAIL, even if you do not copy from it.
  - If you encounter pre-existing work, stop reading it and say so in your report.
- **Do not contaminate pre-existing containers or snoop on other jobs.** Create your own containers/jobs. Do not install into, modify, inspect logs of, or exec against any container or job you did not start.
- **Job naming, logging, and teardown:** Name every job with `"agent"` in the job name so runs are identifiable, record every cluster job you launch (with start time, end time, and `num_chips`) in `job_log.md`, and **tear down every cluster job you started** once final Phase 2 benchmarking and post-hill-climb XProf capture are complete.
- **Artifact organization:** At the end of the task, organize your artifact directory (and mirror all deliverables to `/logs/artifacts/` if `/logs/artifacts` exists) with `onboarding/`, `hillclimb/`, `benchmark_results.json`, `experiment_log.md`, `xprof_bottlenecks_and_gaps.md`, `job_log.md`, pinned versions, profiles, and logs ready to be zipped for analysis.

### Environment & Allowances

- Because this 321B model's FP8 checkpoint exceeds 300 GB, stage weights into shared memory (`/dev/shm`, sizing container shared memory to `>= 450 GB` so pods do not fail on smaller defaults) rather than small node ephemeral disks, and keep concurrency sweep warmups and prompt counts compact (`>= 4x` concurrency prompts per point) so full sweeps finish within cluster pod time caps. Reuse the pre-staged checkpoint at `gs://agent-suite-models/ckpt/hf/zai-org/GLM-5.3-Flash/` (or `gs://tpu-agent-eval/checkpoints/zai-org/GLM-5.3-Flash/`), Hugging Face revision `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`, rather than re-downloading all 62 shards from Hugging Face if reachable.
- `torch_tpu` is not public; it is packaged in the container image specified by `$TASK_DOCKER` in your environment. Resolve `$TASK_DOCKER` to a digest before you start, and report the digest you ran against. Inspect your environment before assuming what is available: check for TPU access, a GKE cluster or pre-configured cluster runner, and quota. Authenticate with ADC. Prefer a clean cluster job setup (or your environment's configured cluster runner); a TPU VM is acceptable if the environment is clean. Size the topology for a 321B total / 18B active FP8 MoE model yourself and state your reasoning.
- vLLM is the intended inference engine. If `vllm serve` hits GPU-only dependencies or custom hybrid-architecture gaps at worker import, KV-cache layout/group planning, or `EngineCore` `profile_run` / `_dummy_run` (for example `ImportError: tilelang is required for mhc`, Sparse MLA / indexer KV layout & page-size grouping, or KDA linear-attention `GDNAttentionMetadata` assertions), do **not** stop early: inspect the `vllm_torchtpu` source inside `$TASK_DOCKER` (e.g. `/workspace/vllm-torchtpu/src/vllm_torchtpu/`, including `models/vllm/` such as `kimi_k3/` and `deepseek_v4/`, `layers/adapter/custom_ops/` such as `kda_attention_op.py` and `mla_attention_op.py`, and `patch_registry.py`), author a `vllm_torchtpu` adapter patch wiring the model's KDA, Sparse MLA/indexer, and norm layers to TPU custom ops (validated rapidly with `--load-format dummy` before loading 328 GB weights, and verified with CPU-vs-TPU numerical parity) to serve via `vLLM` in Phase 1. You may also use or adapt Pallas kernels where they apply (including Tokamax: https://github.com/openxla/tokamax).
- Cite the source for each non-obvious decision — which doc, which file, which measurement. Where you inferred or guessed, say that you did.
