---
scripts:
  - scripts/common/engine_compatibility_check.py
  - scripts/tpu-model-onboarding/fetch_weights.py
  - scripts/common/oss_meshes.py
  - scripts/common/vllm_patches/registry.json
  - scripts/common/vllm_patches/apply_vllm_patch.sh
  - scripts/common/vllm_patches/gpt_oss_mxfp4.py
---
# Engine Compatibility Check

This skill performs a fast pre-flight validation to ensure a requested
HuggingFace model is fully compatible with the currently installed version of
`vllm-torchtpu` or `sglang-torchtpu` on a TPU VM.

It dynamically inspects the remote vLLM installation to detect unsupported
features (like missing Pallas kernels for exotic attention topologies,
quantizations, or unsupported positional embeddings) before you waste time and
bandwidth downloading the model weights or compiling XLA graphs.

## When to use this skill
- Run this **before** deploying the final inference daemon.
- When an engineer asks to deploy a new or uniquely configured HuggingFace model
   on TPU and wants to know if the PyTorch/XLA backend can natively support it.

## Execution Steps

1. **Size the Accelerator**: The check needs `<JOB_DEVICES>`, the devices one
   deployment gets, and `<HBM_GB_PER_DEVICE>`. Size from devices, never from
   chips: a tpu7x chip is two devices. Onboarding works them out as
   tpu-model-onboarding's sizing says (per-chip specs from the knowledge base,
   chips from the compute profile) and does not run `compute inspect`.
   Without both numbers, stop -- TP cannot be sized on unknown hardware.
2. **Run Preflight Check via Unified Compute**: Each compute job takes a whole
   TPU node. When onboarding's preflight job already ran this check for this
   model and these devices in this run, its `<inference_work>/tp_plan.json` is
   this step's result, and its console output is in that job's log (the
   `tail` in its `preflight.json` if it failed): go on to Analyze Output. Otherwise, or to re-plan with another `--tp`, execute the
   preflight check script on the target accelerator environment using
   `{{runtime}} compute run`:
   ```bash
   {{runtime}} compute run --code {{script:common/engine_compatibility_check.py}} \
     --input {{script:common/oss_meshes.py}}:oss_meshes.py \
     --output tp_plan.json:<inference_work>/tp_plan.json \
     -- --model <MODEL_ID> --devices <JOB_DEVICES> --hbm-gb-per-device <HBM_GB_PER_DEVICE> \
        --points <ISL:OSL:C,...>
   ```
   `oss_meshes.py` holds the KV-cache estimate; ship it with every run of
   the check. `--points` lists the benchmark points the request names (input
   length, output length, concurrency), so the plan can say which of them
   the KV cache cannot hold and pick the layout for that traffic; leave it
   out when the request names none.
   If {{script:common/vllm_patches/registry.json}} lists the model
   by its exact Hugging Face id, the installed `vllm_torchtpu` rejects it
   until its patch is applied, and the check must see the patched plugin.
   Read `vllm_patches[<MODEL_ID>]` from `<artifacts>/image_probe.json`, which
   selects the closest validated patch version(s) (`before` and `after` when
   between two versions; only the last when `vllm` is newer; only the first
   when older). If `action` is `"needs_user_confirmation"`, stop and return
   `status: "needs_input"` (do not run the check, fall back to
   `custom_fastapi`, or spend compute cycles fixing the patch without user
   confirmation). Otherwise, ship the first candidate with `applies: true` and
   the applier, and name the patch:
   ```bash
   {{runtime}} compute run --code {{script:common/engine_compatibility_check.py}} \
     --input {{script:common/oss_meshes.py}}:oss_meshes.py \
     --input {{script:common/vllm_patches/apply_vllm_patch.sh}}:apply_vllm_patch.sh \
     --input {{script:common/vllm_patches/gpt_oss_mxfp4.py}}:gpt_oss_mxfp4.py \
     --output tp_plan.json:<inference_work>/tp_plan.json \
     -- --model <MODEL_ID> --devices <JOB_DEVICES> --hbm-gb-per-device <HBM_GB_PER_DEVICE> \
        --points <ISL:OSL:C,...> --vllm-patch gpt_oss_mxfp4.py
   ```
   For a `gs://` checkpoint folder as `<MODEL_ID>`, also pass
   `--input {{script:tpu-model-onboarding/fetch_weights.py}}:fetch_weights.py`.
   The job copies the folder's config and tokenizer (no weights) and gives
   vLLM that copy, never the URL; the weight size for the plan comes from the
   folder's index or listing. A `CRITICAL ERROR: could not read gs://...`
   names what failed, such as the job having no permission to read the
   bucket.
   If the check fails on the first candidate and `candidates` has a second
   applicable bracketing candidate (`before`/`after` when `version_match` is
   `"between"`), try it once. If no candidate passes, **end your work and
   return `status: "needs_input"`** reporting the tried patch version(s) and
   failure; do not fall back to `custom_fastapi` or spend cycles fixing the
   patch until the user confirms.
   Add `--tp <N>` only when the user stated a tensor parallel size; a stated
   TP always wins, and the script records what it costs rather than
   overriding it. Without `--tp`, the script recommends one. When deploying
   with SGLang (`engine: "sglang"`), pass `--engine sglang` (and
   `--trust-remote-code` only when the user or model requires custom Hub code)
   so the check validates `ServerArgs.resolve_once()`, `ModelConfig.from_server_args()`,
   architecture registration in `ModelRegistry`, and TPU topology, recording
   `"recommended_engine": "sglang"` in `tp_plan.json`.
3. **Analyze Output**:
   - The script will perform a dry-run initialization.
   - If it reports that vLLM is not installed, that is an environment problem
   rather than a model incompatibility. Do not install vLLM yourself; report to
   the user that they need to install vLLM on the target.
   - If it throws a `NotImplementedError`, explain to the user exactly which
   operation is missing a Pallas kernel implementation.
   - `<inference_work>/tp_plan.json` holds the parallelism plan:
   `tensor_parallel_size` x `data_parallel_size` equals the device count, the TP
   divides the model's attention heads (and a hybrid model's linear-attention
   heads, recorded in `linear_attention_heads`), and `reason` says why this TP
   was chosen. The recommendation is the largest TP that suits the heads and
   whose weight shard fits one device, so a single engine serves the model.
   Data-parallel replicas are added only when the head counts, the device
   layouts, or vLLM rule out a TP that spans every device. A smaller TP with
   more replicas is not the default even when the weights fit: the fit check
   leaves out the KV cache and hybrid models' per-sequence state, and every
   replica compiles its own graphs at startup. Pass `--tp` for that layout only
   when the user asks for it; `smallest_fitting_tp` names the smallest TP the
   weights fit (null when the weight size is unknown), and `reason` says when it
   is below the recommendation. A TP the installed torch_tpu has no device
   layout for is never recommended: the script asks the installed torch_tpu
   which TP sizes it can lay out, and `reason` names each TP it skipped. When
   the weight size cannot be read (a local path the script cannot size, an
   unreachable hub, an unrecognised dtype), the plan takes the largest
   head-compatible TP without checking the fit, and `reason` says so; tell the
   user.
   - `kv_warnings` lists the `--points` whose sequences do not fit the
   planned TP x DP's KV cache, with the GiB per device each needs and the
   GiB left for the cache; those points will queue or preempt.
   Sliding-window layers count only their window. The check needs the weight
   size for it, so it is skipped when the size could not be read. The script
   prints `WARNING: oss_meshes.py was not shipped` when it ran without the
   estimate: rerun it with the input.
   - `max_num_seqs` sizes vLLM's `--max-num-seqs` for each replica of the
   planned layout from the same estimate. Each point asks for its
   concurrency split over the replicas, capped at 80% of what the KV cache
   holds at that point's own length, so a long prompt's cap never stops a
   short prompt's sweep; `value` is the largest of those, and at least 8.
   `capped` is true when the KV cache, not the points, set it. Linear-attention
   state is not counted, so a hybrid model's value is effectively the
   points'. Launching with it lets a sweep reach its top point on the first
   server. It is null without `--points`, without the estimate (the weight
   size unreadable, or `oss_meshes.py` missing), when no point asks a
   replica for more than 8 (vLLM's default holds them, and a server sized
   for one request would cap any later sweep against it), or when the KV
   cache holds fewer than 8 sequences at every point (`kv_warnings` names
   those); the server then keeps vLLM's own default.
   `max_num_seqs_other_layouts` gives the value for one replica of the
   planned TP and for a traffic rule's `instead_of`, the layouts a
   deployment may use instead (tpu-model-onboarding's
   `torch-inference-coder.md`, Task 1, picks among them).
   - With `--points` that include a long prompt at high concurrency (input
   at least 4x the output, 128 or more concurrent requests) and the default
   `--objective throughput`, a plan whose largest TP spans every device
   halves it once and runs two replicas: `traffic_rule` names the points,
   the layout it passed over (`instead_of`) and the trade-off (fewer
   tokens/s below about 32 users, higher time to first token at long prompts
   up to about 128 users). A sweep that doubles concurrency to 256 reaches
   128, so the rule fires for almost every sweep with a long-prompt pair,
   low-concurrency points included: it picks the layout for the sweep's
   peak. `--objective latency` keeps the largest TP. When the halved
   layout's KV cache is short at a target point where the largest TP's is
   not, the plan keeps the largest TP and records `traffic_rule_dropped`.
   On images where DP replicas hang (see `vllm-tp-dp-plan`), the halved
   layout starts and then never answers; serve `instead_of` there.
   - The rule behind the plan, why it takes the largest TP rather than the
   smallest, the evidence for it and its open caveats are one entry,
   `vllm-tp-dp-plan` in {{agent:tpu-knowledge-base}}. `reason` does not carry
   the caveats. Ask for that entry before you deploy a plan or pass `--tp` to
   change it, and cite it when you explain the plan.
   - If `ok` is false, the model cannot be deployed as planned on this
   hardware; report `reason` to the user.

## Artifact Deliverable
After a successful execution, summarize the model's TPU physical readiness to
the user. Do not proceed to `torch-inference-coder` if the dry-run validation
fails.

## Fallback Mechanism

A model in the patch registry is compatible once patched; its patch is not a
reason to fall back. If the model isn't compatible with vLLM for inference, fallback to using native pytorch inference using the `custom_fastapi` engine to
deploy the model for inference. Before falling back, inform the user of this
decision explicitly to ensure transparency.
