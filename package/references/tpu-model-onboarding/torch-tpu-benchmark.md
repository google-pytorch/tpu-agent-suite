---
scripts:
  - scripts/tpu-model-onboarding/preflight_check.py
  - scripts/tpu-model-onboarding/run_benchmarks.py
  - scripts/tpu-model-onboarding/smart_hf_search.py
  - scripts/tpu-model-onboarding/benchmark_tpu.py
references:
  - references/tpu-model-onboarding/remote-tpu-runner.md
  - references/tpu-model-onboarding/smart-hf-search.md
  - references/tpu-model-onboarding/preflight-check.md
  - references/tpu-model-onboarding/torch-benchmark-coder.md
  - references/tpu-model-onboarding/numerical-parity.md
---
<!-- disableFinding(LINE_OVER_80) -->

# TorchTPU Hugging Face Benchmark Workflow

The end-to-end onboarding of Hugging Face PyTorch
model inference to Google Cloud TPU using the `torch_tpu` API, verifying
correctness, and benchmarking performance against a CPU/GPU baseline. You run
these steps yourself; only CUDA unwiring is delegated.

## Paths

`<artifacts>` and `<work>` are as tpu-model-onboarding defines them.

## Step-by-Step Onboarding Workflow

Always adhere strictly to the following phases. **Never bypass steps** for
perceived efficiency.

### 📋 Mandatory Orchestration Workflow Checklist

You **MUST** follow this step-by-step checklist during execution. At each phase
of your work, check off the step you are executing and ensure you complete the
next step:

-   [ ] **Step 0.1: Accelerator Runtime Verification**: Check that compute is
    configured via `{{runtime}} config show --json --package {{package_tag}}`.
-   [ ] **Step 0.2: Hugging Face Requirements**: Skip gating/auth prompts for
    public models.
-   [ ] **Step 0.3: Hardware Sizing**: Read
    `{{reference:tpu-model-onboarding/remote-tpu-runner.md}}` and size the accelerator.
-   [ ] **Step 0.4: Model Discovery**: Read
    `{{reference:tpu-model-onboarding/smart-hf-search.md}}` and discover candidate models.
-   [ ] **Step 0.5: Pre-flight Compatibility Check**: **MANDATORY**: Read
    `{{reference:tpu-model-onboarding/preflight-check.md}}` AND execute `{{script:tpu-model-onboarding/preflight_check.py}}` with
    `--allow_gpu_imports --cuda_report_json=<artifacts>/cuda_kernels_report.<model_slug>.json --mode=benchmark` (one report per model) for
    **every single candidate model** (locally or remotely). **NEVER SKIP THIS
    STEP.**
-   [ ] **Step 1: Configure Models JSON**: Write candidate model list to
    `<work>/models.json`.
-   [ ] **Step 2: Unwire & Stage Benchmark Scripts**: For every model preflight flagged `ONBOARDING_REQUIRED`, delegate to
    {{agent:tpu-unwiring}} (`target: benchmark`). Then read `{{reference:tpu-model-onboarding/torch-benchmark-coder.md}}` and synthesize `<work>/benchmark_tpu.py`, importing each returned patches module rather than inlining patches.
-   [ ] **Step 3: Execute Benchmarks**: Execute `{{script:tpu-model-onboarding/run_benchmarks.py}}` via
    `{{runtime}} compute run`, inputting `<work>/benchmark_tpu.py` and `<work>/models.json`, outputting to `<work>/results.json`.
-   [ ] **Step 3.5: Numerical Parity Verification**: If CUDA unwiring patches were applied, follow `{{reference:tpu-model-onboarding/numerical-parity.md}}` (`capture_parity.py` and `compare_parity.py`) to verify parity against CPU reference.
-   [ ] **Step 4: Collect Results & Report**: Collect `<work>/results.json` and compile final `<work>/benchmark_report.md` hand-off artifact (including Hit/Miss and Numerical Parity tables).

### Step 0: Environment Setup & Information Gathering

### Step 0.1: Accelerator Runtime Verification

Before executing the onboarding, verify that accelerator compute is configured
and reachable. The runtime resolves the installed environment configuration
itself -- do not go looking for the file, and do not consult SSH configuration
or ask the user for host details. When the configured target is this machine,
it *is* the accelerator host and there is nothing remote to reach; an agent that assumes
otherwise concludes it is on a CPU machine and hard-stops asking for
credentials to the machine it is already on.

#### Verify Accelerator Compute Is Configured

Before proceeding to model discovery, script generation, or benchmarking, read
the active compute profile (do not run `compute inspect`):

```bash
{{runtime}} config show --json --package {{package_tag}}
```

1. If it names an active profile, proceed to Step 0.2. A connectivity error
   surfaces at the first job; treat it as below.
2. If there is no active profile, or a job returns `COMPUTE_NOT_CONFIGURED` or a connectivity error:
   - **STOP IMMEDIATELY AND ASK** the user to set up compute, following the compute setup guide named under **Accelerator Compute Access**. Never create or edit the environment configuration yourself.
   - **DO NOT** proceed to model discovery, script generation, or benchmarking until accelerator compute access has been verified.

### Step 0.2: **MANDATORY** HuggingFace Pre-requisites setup

Before you discovery model and profile hardware, please make to complete these
steps to ensure seamless access to models:

1.  Ask the user if they would like to setup
    huggingface authentication to access gated models. If the user agrees,
    proceed to step 2. Else, explicitly print a CAUTION message for them stating
    that gated models will not be accessible.

2.  **IMPORTANT** DO NOT run the commands yourself. Instead request the user to
    login to HuggingFace using the `huggingface_hub` CLI for gated model access
    on both local machine and remote VM.

    Ask user to authenticate using the `huggingface_hub` CLI:

    ```bash
    hf auth login
    ```

### Step 0.3: Hardware Sizing

Before discovering or validating models, size the target's accelerator and
memory capacity as
`{{reference:tpu-model-onboarding/remote-tpu-runner/hardware_profiling.md}}`
says: the accelerator from the request, its specs from the knowledge base, the
chips from the compute profile. The memory limit (`hbm_gb`) is HBM per chip x
chips.

### Step 0.4: Model Discovery & Capacity Integration

If the user request specifies criteria (e.g., "trending transformer models")
instead of a specific model ID, follow
`{{reference:tpu-model-onboarding/smart-hf-search.md}}` to find candidates.

To prevent Out-Of-Memory (OOM) compilation crashes on the remote TPU
accelerator, you **MUST** apply safe parameter limits on the search query based
on the `hbm_gb` limit resolved in Step 0.3.

#### Real TPU Memory-to-Parameter Boundary Calculations (Relaxed but Conservative)

Models loaded in `bfloat16` format require $2 \times \text{Params}$ GB memory.
To ensure safe compilation and KV cache serving headroom buffers during
PyTorch/XLA execution:

Actual TPU Accelerator Configuration                  | Total HBM Capacity | Safe Search Parameter Limit (`--max_params`) | Weight Footprint (bfloat16) | Notes / Serving Limit
:---------------------------------------------------- | :----------------: | :------------------------------------------: | :-------------------------: | :--------------------
**TPU v5e-1 (1 chip)**                                | `16 GB`            | **`7B`**                                     | `~14 GB`                    | Maximally relaxed. Fits 7B architectures (e.g. Mistral 7B / Qwen 7B) with restricted sequence bounds.
**TPU v6e-1 (1 chip)**                                | `32 GB`            | **`15B`**                                    | `~30 GB`                    | Relaxed limit. Fits standard 7B-14B scales (e.g. Llama 3.1 8B / Qwen 2.5 14B).
**TPU v5e-4 (4 chips)**                               | `64 GB`            | **`30B`**                                    | `~60 GB`                    | Relaxed limit. Fits higher-scale models (e.g. Gemma 2 27B / Llama 2 13B).
**TPU v6e-4 (4 chips) / TPU v4-8 (8 cores, 4 chips)** | `128 GB`           | **`50B`**                                    | `~100 GB`                   | Relaxed limit. Accommodates Mixture-of-Experts and advanced LLMs (e.g. Mixtral 8x7B / Qwen 32B).
**TPU v5p-8 (8 cores, 4 chips)**                      | `380 GB`           | **`130B`**                                   | `~260 GB`                   | Relaxed limit. Fits large-scale serving foundation targets (e.g. Llama 3 70B / Command-R+).

#### Inter-Skill Event Mapping

When parsing candidates returned in JSON format by the search skill, route
metadata fields as follows:

-   **`library` & `pipeline` metadata**: Match the candidate's pipeline tags
    (`text-generation`, `fill-mask`, `feature-extraction`) to determine model
    architecture parameters for the subsequent phases.

#### High-Diversity Model Gathering & Interleaving (Generic & Scalable)

When the user requests a verified list of `{C}` compatible Hugging Face models
from a broad representation of `{L}` PyTorch libraries (e.g., `{C}=20`
compatible models from `{L}=9` distinct library tags including `transformers`,
`diffusers`, `timm`, `peft`, `trl`, `sentence-transformers`, `pyannote.audio`,
`open_clip`, and `speechbrain` / `espnet`; or general large scales such as
`{C}=500` models from `{L}=20` libraries):

1.  **Perform Redundant Search by Library Tag**:

    *   Run the smart-hf-search script
        once for each of the `{L}` distinct library tags.
    *   Since local pre-flight checks can reject models due to gating issues,
        incompatible formats, custom forbidden imports, or unsupported GPU
        dependencies, you **MUST query a redundant/buffer number of candidates
        `{K}` per library tag** to ensure you have a robust buffer of backup
        models to choose from.
    *   *Sizing Guideline*: Choose `{K}` such that `{K} >= 1.5 * (target models
        expected per library)`. For example, if aiming for `{C}=20` across 9
        libraries, choose `{K}=10` or `{K}=15` per tag. If scaling up to
        `{C}=500` models across 20 libraries (averaging ~25 models per library),
        select a buffer of `{K}=40` or `{K}=50` raw matches per tag.

2.  **Interleave the Candidates for Maximum Tag Diversity**:

    *   To prevent a single domain or library (e.g., `transformers`) from
        filling your entire quota of verified models early on and destroying tag
        diversity, you **MUST NOT** perform pre-flight checks sequentially
        library-by-library.
    *   Incorporate all raw candidates and **interleave them using a round-robin
        strategy** across the libraries:
        *   Round 1: 1st model from Library 1, 1st model from Library 2, ...,
            1st model from Library N.
        *   Round 2: 2nd model from Library 1, 2nd model from Library 2, ...,
            2nd model from Library N.
        *   Continue this pattern until all retrieved candidate lists are
            completely exhausted.

3.  **Progressive Pre-flight Filtering**:

    *   Execute the preflight check
        sequentially on this interleaved list of candidate models.
    *   Track the models that successfully pass all validation steps.
    *   Feed models one-by-one through the validator until the requested target
        quota `{C}` of verified, compatible models is successfully accumulated,
        or the interleaved candidate list is fully exhausted.

#### Handling Batch Model Family Resolution (e.g. `important models`)

When the user requests benchmarking a predefined list of model families (e.g.,
`important models` list in the prompt) rather than concrete model IDs:

1.  **Parse the list** of model families directly from the prompt.
2.  **Build the search tool** locally to run it without build overhead:

    ```bash
        ```

3.  **Resolve families in parallel**: Create a temporary Python scratch script
    (in `<work>/search_scratch.py`) to call `{{script:tpu-model-onboarding/smart_hf_search.py}}` concurrently
    (e.g., using `ThreadPoolExecutor` with 10 workers) for each family. Use
    `--k=3` to retrieve multiple candidates for redundancy:

    ```bash
    python3 {{script:tpu-model-onboarding/smart_hf_search.py}} \
      --search={family} \
      --sort=downloads \
      --k=3 \
      --json
    ```

4.  **Filter by HBM Capacity**: Parse the JSON outputs to get concrete model
    IDs. Immediately discard any resolved models whose parameter size exceeds
    the safe TPU HBM limit resolved during diagnostics (Step 0.3) to avoid OOMs
    during compilation. Keep the remaining compatible candidates ordered by
    popularity for each family.

### Step 0.5: Pre-flight Compatibility Check

Before generating benchmark scripts or executing remote runs, you **MUST**
validate the target model's compatibility and pipeline properties locally, or on
the target host via compute runtime as a backup option if local execution fails.

1.  **Load Instructions**: You **MUST** read
    `{{reference:tpu-model-onboarding/preflight-check.md}}`
    to understand how to validate Hugging Face model compatibility. Never bypass
    reading it.
2.  **Execute Preflight Validation**: Follow those instructions to run preflight compatibility checks on each
    candidate model, ensuring `--cuda_report_json=<artifacts>/cuda_kernels_report.<model_slug>.json`
    is written under `<artifacts>` before proceeding to full benchmarking.

### Step 1: Create a file with the list of models

We will start by creating the list of compatible models to benchmark under
`<work>/models.json`. Here is an example file that benchmarks models on `cpu`
and `tpu`:

```json
{
    "models": [
        {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "library": "transformers",
            "pipeline_tag": "text-generation",
            "device": "cpu",
            "clean_cache": false
        },
        {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "library": "transformers",
            "device": "tpu",
            "pipeline_tag": "text-generation",
            "clean_cache": true
        },
        {
            "model_id": "stabilityai/stable-diffusion-3-5-large",
            "library": "diffusers",
            "device": "cpu",
            "pipeline_tag": "text-to-image",
            "clean_cache": false
        },
        {
            "model_id": "stabilityai/stable-diffusion-3-5-large",
            "library": "diffusers",
            "device": "tpu",
            "pipeline_tag": "text-to-image",
            "clean_cache": true
        }
    ]
}
```

Similarly, you will need to create a separate file with `cuda` (and optionally
`cpu`) as device for benchmarking on GPU VM.

**TIP**: As shown in the above sample JSON, the models file can contain multiple
entries for the same model_id and library, but with different devices and
clean_cache values. If a single VM has multiple accelerators that you are
interested in benchmarking, you can add multiple entries for the same model_id
and library with different device values.

### Step 2: Benchmark Script Generation & Unwiring

For every model preflight flagged `[STATUS: ONBOARDING_REQUIRED]`, delegate to
{{agent:tpu-unwiring}} first, one model at a time, with `model_id`, `run_id`,
`target: benchmark`, and the fully expanded `cuda_kernels_report` path. Each
returns its own `patches_path`. A model that comes back `blocked` or `failed`
is dropped from `<work>/models.json`, recorded as `BLOCKED_ON_CUDA_KERNEL` in
that model's `<artifacts>/cuda_kernels_report.<model_slug>.json`, and listed in the report; the other
models still run.

Then follow `{{reference:tpu-model-onboarding/torch-benchmark-coder.md}}` to synthesize the benchmark script at
`<work>/benchmark_tpu.py` based on the shipped template `{{script:tpu-model-onboarding/benchmark_tpu.py}}`.
Never overwrite files in the shipped package tree.

1.  Identify the model's architecture/task (mapping to Causal LM, SpeechSeq2Seq,
    Feature Embedding, or Vision-Language patterns).
2.  Incorporate standard PyTorch-to-TPU device placement dynamically using
    `api.tpu_device()`.
3.  Ensure proper warmup phases are added to exclude compilation/graph overhead
    from speedup calculations.
4.  For each unwired model, import its patches module and call `apply()` before that model is constructed. Never inline the patches into `<work>/benchmark_tpu.py`: the parity check imports the same module, and cannot import patches that live inside the benchmark script.

### Step 3: Remote Running & Verification

Follow `{{reference:tpu-model-onboarding/remote-tpu-runner.md}}` to run targets via the unified compute runtime:

1.  Execute the orchestrator script `{{script:tpu-model-onboarding/run_benchmarks.py}}`
    passing `<work>/benchmark_tpu.py` and `<work>/models.json` via `--input`, and
    collecting results via `--output results.json:<work>/results.json`:
    ```bash
    {{runtime}} compute run --code {{script:tpu-model-onboarding/run_benchmarks.py}} \
      --timeout 3600 \
      --input <work>/benchmark_tpu.py:benchmark_tpu.py \
      --input <work>/models.json:models.json \
      --output results.json:<work>/results.json \
      -- --models_json models.json --output_json results.json
    ```
    For each unwired model, add `--input <patches_path>:patches/<model_slug>.py`
    (`<model_slug>` is the model ID with every `/` replaced by `__`)
    so `benchmark_tpu.py` can import it on the target.
2.  This script will automatically handle iterating over the models, running them on the appropriate devices,
    pre-fetching weights, handling errors, and saving results to `results.json`.
3.  Collect execution stdout/stderr payloads into `<work>/benchmark.log` and verify `<work>/results.json`.

### Step 3.5: Numerical Parity Verification (Mandatory for Unwired Models)

For every model that relies on unwired CUDA patches (hits or misses), follow `{{reference:tpu-model-onboarding/numerical-parity.md}}` (`capture_parity.py` and `compare_parity.py`) with frozen teacher-forced input probes to verify that the unwired PyTorch implementation matches CPU reference outputs within noise floor tolerances. {{agent:tpu-unwiring}} already built the probe and the candidate capture for each model (its `probe_path` and `candidate_capture`); run only the reference capture and the compare, in that directory. If parity fails, update that model's `<artifacts>/cuda_kernels_report.<model_slug>.json` with `status: "BLOCKED_ON_CUDA_KERNEL"` and halt.

### Step 4: Speedup Report Generation & Onboarding Hand-Off Document

1.  **Report Generation**: Compile `<work>/benchmark_report.md` including:
    - Comparative performance timing, speedup ratios, and latency distributions.
    - **Playbook Hit/Miss Breakdown Table** (`HIT` vs `MISS` per kernel).
    - **Numerical Parity Verification Table** (`max_rel`, `top-1 agree`, per-module verdict).
2.  **Kernel Optimization Handoff**: If custom kernels could not be unwired or if running with eager fallback, advise user to dispatch `tpu-kernel-development` to synthesize Pallas TPU kernels.

--------------------------------------------------------------------------------

## Verification & Benchmarking Rules

*   **Warmup Execution**: You **must** trigger a warmup run on the remote host
    before recording execution times. XLA compiles graphs on-the-fly; first run
    timings represent compile latencies rather than actual runtime performance.
*   **Determinism and Correctness Check**: For text generation models where
    sampling is turned off (`do_sample=False`), enforce strict CPU vs TPU
    semantic comparisons. For sampling-enabled configurations, compare outputs
    using semantic coherence verification.
