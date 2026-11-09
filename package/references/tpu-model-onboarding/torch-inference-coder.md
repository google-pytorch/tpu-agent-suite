---
scripts:
  - scripts/tpu-model-onboarding/inference.py
  - scripts/tpu-model-onboarding/fetch_weights.py
  - scripts/common/discover_vllm_flags.sh
  - scripts/common/discover_sglang_flags.sh
  - scripts/common/serve_vllm.sh
  - scripts/common/serve_sglang.sh
  - scripts/tpu-model-onboarding/inference_test.py
  - scripts/tpu-model-onboarding/health_probe.py
  - scripts/tpu-model-onboarding/verify_inference.py
  - scripts/tpu-model-onboarding/tpu_lock_reset.py
  - scripts/common/vllm_patches/registry.json
  - scripts/common/vllm_patches/apply_vllm_patch.sh
  - scripts/common/vllm_patches/gpt_oss_mxfp4.py
references:
  - references/tpu-model-onboarding/torch-inference-coder/fastapi_server/text_generation.md
  - references/tpu-model-onboarding/torch-inference-coder/fastapi_server/embeddings.md
  - references/tpu-model-onboarding/torch-inference-coder/fastapi_server/vlm.md
---
# Torch Inference Coder Skill

This skill guides the generation, deployment, and monitoring of HTTP Inference
engines on TPU wrapped around a model provided by the user.

## Artifact Root (`<artifacts>`)

`<artifacts> = {{runs}}/<run_id>/artifacts`, `<work> = <artifacts>/tpu-model-onboarding`, `<inference_work> = <work>/inference`.
Return `invalid_input` if `run_id` is missing. Write only to `<inference_work>`, never to CWD or package files.

Every file this skill produces -- the manifest, the generated server, the
launcher, and every report retrieved from the target -- lands under
`<inference_work>`. The unwiring patches module is the one exception: it is
produced by tpu-unwiring, which reports its `patches_path`. Package-relative paths such as
`{{script:tpu-model-onboarding/inference.py}}` name shipped files
to read or execute, never output locations.

## Inputs for Skill

-   `model`: Identifier for the model to be deployed. This can be one of the following:
        a. HuggingFace Model ID
        b. Local path to model code
        c. Local path to checkpoint file
        d. URL to checkpoint file or model repo
-   **`engine`**(Optional): The engine to be used for setting up the model for
    inference. Default: `vllm` for supported CausalLM architectures (`sglang`
    when explicitly requested); `custom_fastapi` for non-LLMs, unwired models,
    or fallback.
-   **`port`**(Optional): The port to use to setup the API for inference.

## How This Skill Reaches The Target

Everything here runs on the accelerator host, and nothing in this skill puts it
there by hand. You do not open a shell on the target, copy files to it, or
drive its cluster directly; the runtime owns every transport. Four mechanisms
carry all of it:

-   **`compute run`** copies a script over with its `--input` files, runs it to
    completion, brings `--output` files back, and returns. Use it for anything
    that finishes.
-   **`compute submit --serve`** launches a detached daemon that outlives the
    call and returns a `job_id`. `--serve` marks it a service, which exempts it
    from the session-end reaper, so the endpoint survives the session. It takes
    `--input` too, so the manifest travels beside the recipe.
-   **`compute port-forward`** brings the engine's port to this host, so health
    checks and test requests are ordinary local HTTP calls. Without it,
    `localhost` here is not the serving host.
-   **`compute job wait`** blocks until the deployment is running, has
    printed its ready line, or has visibly failed. It is the only way to wait
    on a job: never write a `sleep` or polling loop.
-   **`compute job status` / `logs` / `cancel`** follow and end the deployment.

### Capturing the job id

Do **not** write ``JOB_ID=`compute submit ...` `` or `JOB_ID=$(compute submit
...)`. The daemon inherits the command-substitution pipe and holds it open for
its whole life, so the shell blocks until the server exits -- which for a
service is never. `submit` itself returns immediately; it is the capture that
hangs.

Do not redirect the id to a file either. Instead give the job a name you choose
and read the id back from the runtime's own job record:

```bash
{{runtime}} compute submit --package {{package_tag}} ... --name <model>-serve
JOB_ID=$({{runtime}} compute job list --package {{package_tag}} --json --no-refresh \
  | python3 -c "import json,sys;print(next(r['job_id'] for r in json.load(sys.stdin) if r['job_name']=='<model>-serve'))")
```

Substitution is safe here: with `--no-refresh`, `job list` reads the record
store and exits without asking any backend, so nothing holds the pipe open. Pick a `--name` unique to this deployment; if two
jobs share it, the filter is ambiguous and you must disambiguate on `status`.

## Weights: Fetch Before The Engine Loads

Neither engine downloads the weights itself. The serving job first runs
`{{script:tpu-model-onboarding/fetch_weights.py}}`, which travels as an `--input`
beside the recipe (Task 3a), and only then starts the engine. It:

-   downloads only the shards the repo's own index lists, plus config and
    tokenizer files. Copies kept for other runtimes are skipped:
    `original/` and `metal/` in `openai/gpt-oss-20b` (27.5 GB of its 41 GB),
    a Mistral `consolidated.safetensors`, TF/Flax/ONNX formats.
-   prints a `WEIGHTS PROGRESS` line with bytes, rate and ETA every 30
    seconds, then `WEIGHTS READY <snapshot path>`.
-   skips files already in the Hugging Face cache. On `gke_job` with
    `hf_cache_pvc` configured, that cache is shared by every pod, so a model
    downloads once; without it, each new pod starts empty and downloads
    everything again, into a cache `gke_job` puts on the pod's `/dev/shm`
    (capped at the profile's `shm_size_gb`) rather than its own disk, which
    can hold less than one model.

The engine then loads the **snapshot path**, never the repo id, with
`HF_HUB_OFFLINE=1`. Offline, a lookup by repo id fails with
`IncompleteSnapshotError` when its patterns match a file the fetch skipped;
a local directory is read as it is. Offline mode also turns any download the
engine still attempts into an error in the log instead of a silent multi-GB
transfer.

The vLLM launcher exports `HF_HUB_OFFLINE=1` before vLLM starts. The FastAPI
server cannot: `huggingface_hub` reads the variable once, at import, and
`fetch` has already imported it, so setting it afterwards does nothing. The
server passes `local_files_only=True` to `from_pretrained` instead, to the
same effect.

## Task 1: Generate `deployments.json`

Create the manifest at `<inference_work>/deployments.json`.

### Dynamic Discovery for vLLM

If the engine is `vllm`, discover the flags the installed build accepts before
writing the manifest -- do not assume them. Preflight's Step 3 job normally
captured them already: when its `preflight.json` has a
`discover_vllm_flags.exit_code` of `0`, use `<inference_work>/vllm_flags.txt`
and skip this job, which would take a TPU node again for the same answer.
Otherwise run:

```bash
{{runtime}} compute run --package {{package_tag}} \
  --code {{script:common/discover_vllm_flags.sh}} \
  --timeout 120 \
  --output vllm_flags.txt:<inference_work>/vllm_flags.txt \
  -- vllm_flags.txt
```

Read `<inference_work>/vllm_flags.txt` and populate the manifest only with keys
that appear there. The flags come only from this run's own result: if it
exits non-zero or the file was not produced, report its output and stop
rather than writing the manifest from flags you remember. Do not activate a
virtualenv: pass `--interpreter <venv>/bin/python3` if the vLLM install is not
on the target's default interpreter, and the script resolves `vllm` from
beside it, else from `PATH`.

Populate the `deployments.json` with keys mapping to standard CLI arguments.
For `vllm` and `sglang`, take the parallelism from
`<inference_work>/tp_plan.json`, written by the `engine-compatibility` check.
The plan names its fields the vLLM way whichever engine it was run for; the
manifest uses the engine's own keys, and the flags file is the one discovered
for that engine (for `sglang`, run step 1 of the `sglang` recipe before
writing the manifest):

| Meaning | `tp_plan.json` field | `vllm` key (flag) | `sglang` key (flag) |
|---|---|---|---|
| Flags file | | `vllm_flags.txt` | `sglang_flags.txt` |
| TP | `tensor_parallel_size` | `tensor_parallel_size` (`--tensor-parallel-size`) | `tp_size` (`--tp-size`) |
| DP (replicas) | `data_parallel_size` | `data_parallel_size` (`--data-parallel-size`) | `dp_size` (`--dp-size`) |
| Requests per replica | `max_num_seqs` | `max_num_seqs` (`--max-num-seqs`) | `max_running_requests` (`--max-running-requests`) |
| Context length | | `max_model_len` (`--max-model-len`) | `context_length` (`--context-length`) |

Below, "the TP key", "the DP flag" and so on mean that engine's column. Steps
that read the manifest later (the benchmark's concurrency limit, a sweep's
`--max-model-len`) read the same keys. SGLang on TPU may lower its requests
per replica at startup; Step 3d checks for that and updates the manifest.

Apply these in order; the first that matches decides:

1.  **The user stated a TP** (the plan's `source` is `"user"`): use its
    `tensor_parallel_size` as the TP key, whatever else holds. Copy
    `data_parallel_size` as the DP key too when the DP flag appears in the
    flags file; when it does not, omit it and run one replica, leaving the
    other devices unused.
2.  **The DP flag appears in the flags file**: copy the plan's
    `tensor_parallel_size` and `data_parallel_size` as the TP and DP keys.
    Their product is the plan's `devices`, so no device sits idle.
3.  **Otherwise** the engine cannot run replicas: copy the TP key and omit
    the DP key. The plan's TP is already the largest that fits, so a larger
    one would not start either; any devices the replicas would have used sit
    idle.

Record the choice in a `parallelism` object: `source` and `reason` from the
plan, plus, under rule 1 without data parallelism or under rule 3, that data
parallelism is unavailable in this engine build. When the plan's
`smallest_fitting_tp` is below the TP you use, record it in `parallelism` too:
benchmark numbers for this deployment are then measured on the single-engine
layout, and any report of them must say that TP `smallest_fitting_tp` with
more replicas also fits and may serve faster. `parallelism` is a record, not
an engine flag.

Set the requests-per-replica key when its flag appears in the flags file,
from the first that applies:

1.  A value the user states, or the patch registry's `serving` settings name.
2.  The plan's value for the layout you deploy: `max_num_seqs.value` when
    you deploy the plan's TP and DP, else the
    `max_num_seqs_other_layouts` entry with your TP and DP (a manifest
    without the DP key runs one replica: DP 1). A null there means omit the
    key; do not go on to item 3.
3.  A layout neither field lists (the registry's, say): the highest
    `--points` concurrency divided by your DP, rounded up, when that is
    above 8, and at least 8; otherwise omit the key.

The value is per replica for both engines, so the server admits that value
x DP requests at once. (SGLang divides `--max-running-requests` among
replicas only under DP attention, which these recipes do not enable.) A null
plan value means no `--points`, no KV estimate, no point asking a replica
for more than 8, or a cache that holds fewer than 8 sequences at every
point. With the key omitted, the engine keeps its own default; a sweep then
runs without `--max-concurrency-limit`, and its top points may measure
queueing rather than the server.

**Check the layout with tpu-knowledge-base: required for every `vllm`
deployment, before you write the manifest.** Delegate to
{{agent:tpu-knowledge-base}} for `vllm-tp-dp-plan`, naming the model, the
hardware, the serving image, the TP x DP you are about to use,
`smallest_fitting_tp`, and the input length, output length and concurrency
the request targets (the entry's rule for TP versus replicas depends on
them). The
plan's `reason` states the rule but not what is known to go wrong with it;
that entry records the rule's evidence and its open caveats, such as a model
that failed at the TP the rule picks. Do not skip the delegation because
`reason` already explains the choice, or because the user did not ask.

-   Copy every caveat that names this model, this TP or this hardware into
    `parallelism.caveats`, and tell the user before launching. Copy there too
    every `kv_warnings` entry of `tp_plan.json`: each names a target point
    the KV cache cannot hold. When the plan has a `traffic_rule`, copy it
    too, with its
    `tradeoff` and the layout it passed over (`instead_of`), and tell the
    user: the plan chose fewer TP ranks for the target traffic. If
    `--data-parallel-size` is not in `vllm_flags.txt`, use `instead_of`
    rather than the plan's TP, since a halved TP without replicas leaves
    half the devices idle.
-   When the user asks why this TP, or wants a different one, answer from
    that entry and cite it. It uses the same terms (device, TP, replica, DP)
    as `tp_plan.json`.
-   A caveat changes nothing by itself: TP and DP still come from the rules
    above, or from the patch registry's `serving` settings for a model it
    lists (step 5 of Task 2's `vllm` recipe), and Step 3e's retries still
    decide what happens if the server fails.

The key is **`model_id`**, matching the Pydantic schema in
`{{script:tpu-model-onboarding/inference.py}}`. A manifest
using `model` fails validation at server startup.

Example for `vllm`:
```json
{
  "deployments": [
    {
      "model_id": "meta-llama/Llama-3.1-8B-Instruct",
      "engine": "vllm",
      "port": 8000,
      "max_model_len": 2048,
      "tensor_parallel_size": 8,
      "data_parallel_size": 1,
      "max_num_seqs": 256,
      "parallelism": {
        "source": "recommended",
        "reason": "largest TP that divides 32 heads and whose weight shard (1.9 GiB) fits 80% of 32 GiB, leaving the most HBM for the KV cache and compiling one engine; TP 1 also fits and, with more replicas, may serve faster: pass --tp to measure it, at the cost of startup time and memory headroom",
        "smallest_fitting_tp": 1
      }
    }
  ]
}
```

Example for `sglang`:
```json
{
  "deployments": [
    {
      "model_id": "meta-llama/Llama-3.1-8B-Instruct",
      "engine": "sglang",
      "port": 8000,
      "context_length": 2048,
      "tp_size": 1,
      "dp_size": 8,
      "max_running_requests": 32,
      "parallelism": {
        "source": "user",
        "reason": "the user asked for TP 1"
      }
    }
  ]
}
```

Example for `custom_fastapi`:
```json
{
  "deployments": [
    {
      "model_id": "meta-llama/Llama-3.1-8B-Instruct",
      "engine": "custom_fastapi",
      "port": 8000,
      "max_model_len": 2048
    }
  ]
}
```

## Task 2: Synthesize API Server Script / Launcher

Check the `engine` specified in the `deployments.json`. Generate the
appropriate engine launcher. The manifest is **not** inlined into the script --
it travels as an `--input` at submit time, so the script reads
`deployments.json` by bare name from its working directory.

### Recipe: `custom_fastapi` (Native PyTorch)

If native PyTorch execution is requested:

1.  Start from the backbone execution runtime,
    `{{script:tpu-model-onboarding/inference.py}}`, as your
    template. Synthesize a new file at `<inference_work>/fastapi_server.py`;
    never modify the shipped one, which is read-only package content.
2.  Determine the model's pipeline type (e.g. Causal LM, Embedding,
    Vision-Language).
3.  Reference the corresponding Python API overrides template:
    *   Text Generation / Causal LM:
        `{{reference:tpu-model-onboarding/torch-inference-coder/fastapi_server/text_generation.md}}`
    *   Embeddings: `{{reference:tpu-model-onboarding/torch-inference-coder/fastapi_server/embeddings.md}}`
    *   Vision-Language (VLM): `{{reference:tpu-model-onboarding/torch-inference-coder/fastapi_server/vlm.md}}`
4.  Implement the `AGENT TODO` gaps precisely following the logic block
    templates provided in your chosen reference. You must:
    -   Fill in `load_model()` (ensure `get_device('tpu')` is used to
        bootstrap safely). Load from `fetch(config.model_id)`, as the
        reference shows, never from `config.model_id` directly, and never at
        module level: the server runs `load_model()` at startup, and the test
        below imports the module without it.
    -   Setup the `Payload Schema` request model targeting your pipeline format.
    -   Write the `Generate Endpoint API`.
5.  **(Optional Verification)**: check Pydantic parsing and endpoint routing
    before launching a daemon. The test writes its own dummy
    `deployments.json` and then does `import inference`, so your generated
    server must arrive under that name. It takes no arguments:

    ```bash
    {{runtime}} compute run --package {{package_tag}} \
      --code {{script:tpu-model-onboarding/inference_test.py}} \
      --timeout 300 \
      --input <inference_work>/fastapi_server.py:inference.py \
      --input {{script:tpu-model-onboarding/fetch_weights.py}}:fetch_weights.py
    ```

### Recipe: `vllm` (TPU Pallas)

If the vLLM engine is requested:

1.  Do not generate custom Python FastAPI code. vLLM implements its own
    OpenAI-compatible FastAPI server internally.
2.  Generate a `<inference_work>/launch_vllm.sh` bash script. Iterate through
    the extra keys in the manifest and append them as CLI flags to the vLLM command
    (changing `tensor_parallel_size` to `--tensor-parallel-size`, etc.), using
    only flags that appeared in `vllm_flags.txt`. Skip `parallelism`; it
    records why, and is not a flag.
3.  You MUST append `--host 0.0.0.0` to the generated bash script, and
    `--attention-backend CUSTOM` unless the manifest sets `attention_backend`
    (a serving candidate's manifest can; step 2 has then appended it, and a
    second `--attention-backend` would override it). `--host 0.0.0.0` is what
    makes the port forwardable; a server bound to `127.0.0.1` inside a pod
    cannot be reached. `CUSTOM` selects vllm-torchtpu's
    `PallasBatchedRPAAttentionBackend`, which runs the plugin's
    `kernels/experimental/batched_rpa` rather than RPA v3; without it the
    plugin falls back to `FLASH_ATTN`, and Gemma-4-31B-it fails to compile
    (ask {{agent:tpu-knowledge-base}} for `custom-attention-backend`). A
    candidate's `FLASH_ATTN` runs RPA v3 deliberately, with the decode
    blocks its env sets.
4.  Start the server with `exec bash serve_vllm.sh <the vllm serve
    arguments>`, never with `vllm serve` directly (see step 7). It resolves
    `vllm` from beside `$TPU_AGENT_SUITE_PYTHON`, the interpreter the framework
    selected, rather than sourcing a virtualenv; run any Python the recipe
    needs with that interpreter too, as the example does.
5.  Look the model up in {{script:common/vllm_patches/registry.json}}
    by its exact Hugging Face id, and read `vllm_patches[<model_id>]` from
    `<artifacts>/image_probe.json`. A listed model serves only with its
    `vllm_torchtpu` patch applied. Use the first candidate in
    `vllm_patches[<model_id>].candidates` with `applies: true` (closest
    `before`/`after` when between validated versions; only the last when `vllm`
    is newer; if none has `applies: true`, stop and return
    `status: "needs_input"` waiting for user confirmation before spending
    cycles fixing the patch). The task image is never rebuilt, so the
    recipe applies the patch every time it starts: begin it with `set -eo
    pipefail` and run `bash apply_vllm_patch.sh <patch>` before `vllm serve`,
    so a patch that does not apply stops the server instead of serving the
    unpatched model. Ship both files as inputs in Task 3a. The entry's
    `provenance` keyword names the playbook with the failures the patch
    fixes; ask {{agent:tpu-knowledge-base}} for it when a patched server fails.
    Tell the user the patch is applied, and which candidate stack version
    (`validated_vllm`) it comes from. That candidate's `serving` names the
    settings the stack needs: use them unless the user stated others. They
    take precedence over the TP and DP in `tp_plan.json`; record
    `parallelism.source` as `registry` in the manifest.
6.  Fetch the weights first, then serve the snapshot path offline (see
    *Weights* above). Pass `--served-model-name <model_id>` when it appears in
    `vllm_flags.txt`, so clients and Task 4 still address the model by its
    repo id; when it does not, the served name is the snapshot path, and
    Task 4 must pass that path as `--model-id`.
7.  `serve_vllm.sh` ({{script:common/serve_vllm.sh}}) is what makes the
    server profilable. vLLM can record a profile only when its profiler was
    configured at server start, so a server started without it can be profiled
    only by restarting it -- which on a preemptible cluster risks losing the
    slice and the running baseline. The script arms vllm_torchtpu's phased
    profiler on every start, at no cost until a client calls
    `POST /start_profile`, writing one short trace per inference phase to
    `artifacts/vllm_profile/<phase>/` under the job's working directory. It
    also serves that directory on port 8790, because the GKE job backend's
    `job artifacts` cannot read a running pod's files. Do not reproduce what
    it does inline: which setting this vLLM build reads is the script's to
    work out. {{agent:tpu-diagnostics}} captures the server through a `server`
    request naming this job's id and port. On `gke_job` with `hf_cache_pvc`,
    the pod's `VLLM_CACHE_ROOT`, `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` and
    `TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT` point at
    `/hf-cache/compile-cache/<accelerator>-<topology>/<image key>/{vllm,torch_tpu}`
    (or the pod's /dev/shm when the volume is 90% full), so a restart on the
    same image skips most of compiling; image folders unused for
    `compile_cache_max_age_days` (default 14) are removed. A build patched
    with `apply_vllm_patch.sh` keeps the image key, so `serve_vllm.sh` moves
    those caches to sibling `-patch-<key>` folders by itself; don't set them
    in `launch_vllm.sh`. It also raises vLLM's GPU-sized timeouts for TPU
    compilation (`VLLM_ENGINE_READY_TIMEOUT_S=3600`,
    `VLLM_ENGINE_ITERATION_TIMEOUT_S=1800`,
    `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`, and
    `--distributed-timeout-seconds 1800` where the build has the flag),
    keeping any the recipe sets; don't lower them. If the image probe's
    `vllm_timeout_envs` reports one as `false`, this vLLM renamed it: say so
    in the report.

Example `<inference_work>/launch_vllm.sh` generated from JSON:
```bash
#!/bin/bash
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
PY="${TPU_AGENT_SUITE_PYTHON:-python3}"
PY="$(command -v "${PY/#\~/$HOME}")" \
  || { echo "no Python at ${TPU_AGENT_SUITE_PYTHON:-python3} on the target" >&2; exit 1; }
"$PY" fetch_weights.py "meta-llama/Llama-3.1-8B-Instruct" \
  --path-file weights_path.txt || exit 1
export HF_HUB_OFFLINE=1
exec bash serve_vllm.sh "$(cat weights_path.txt)" \
  --served-model-name "meta-llama/Llama-3.1-8B-Instruct" \
  --port 8000 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --max-model-len 2048 \
  --host 0.0.0.0 \
  --attention-backend CUSTOM
```

The same recipe for a model listed in the patch registry
(`openai/gpt-oss-20b`), with its entry's `validated.serving` settings:
```bash
#!/bin/bash
set -eo pipefail
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export USE_MOE_SPARSE_CORE=0
bash apply_vllm_patch.sh gpt_oss_mxfp4.py
exec bash serve_vllm.sh "openai/gpt-oss-20b" \
  --port 8000 \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --attention-backend CUSTOM
```

### Recipe: `sglang`

If the `sglang` engine is requested:

1.  Discover the flags the installed build accepts with
    `{{script:common/discover_sglang_flags.sh}}` before writing
    `deployments.json`. Preflight's Step 3 job normally captured them already:
    when its `preflight.json` has a `discover_sglang_flags.exit_code` of `0`,
    use `<inference_work>/sglang_flags.txt` and skip this job. Otherwise run:
    ```bash
    {{runtime}} compute run --package {{package_tag}} \
      --code {{script:common/discover_sglang_flags.sh}} \
      --timeout 120 \
      --output sglang_flags.txt:<inference_work>/sglang_flags.txt \
      -- sglang_flags.txt
    ```
2.  Generate `<inference_work>/launch_sglang.sh` and start the server with
    `exec bash serve_sglang.sh <the sglang.launch_server arguments>`
    ({{script:common/serve_sglang.sh}}), which arms
    `SGLANG_TORCH_PROFILER_DIR` and serves traces on port 8790. It also
    raises SGLang's GPU-sized timeouts for TPU compilation
    (`SGLANG_WARMUP_TIMEOUT=3600`, and `--watchdog-timeout 3600` and
    `--dist-timeout 1800` where the build has the flag), keeping any the
    recipe sets; don't lower them. Pass the manifest's `tp_size`, `dp_size`,
    `max_running_requests` and `context_length` as `--tp-size`, `--dp-size`,
    `--max-running-requests` and `--context-length` (each only when the key is
    in the manifest; Task 1 already checked them against
    `sglang_flags.txt`), plus `--served-model-name <model_id>`,
    `--device tpu`, and `--host 0.0.0.0`. Skip `parallelism`; it is not a
    flag:
    ```bash
    #!/bin/bash
    set -eo pipefail
    PY="${TPU_AGENT_SUITE_PYTHON:-python3}"
    PY="$(command -v "${PY/#\~/$HOME}")" \
      || { echo "no Python at ${TPU_AGENT_SUITE_PYTHON:-python3} on the target" >&2; exit 1; }
    "$PY" fetch_weights.py "meta-llama/Llama-3.1-8B-Instruct" \
      --path-file weights_path.txt || exit 1
    export HF_HUB_OFFLINE=1
    exec bash serve_sglang.sh "$(cat weights_path.txt)" \
      --served-model-name "meta-llama/Llama-3.1-8B-Instruct" \
      --port 8000 \
      --tp-size 1 \
      --dp-size 8 \
      --max-running-requests 32 \
      --context-length 2048 \
      --device tpu \
      --host 0.0.0.0
    ```
3.  Submit `<inference_work>/launch_sglang.sh` in Step 3a with
    `--input {{script:common/serve_sglang.sh}}:serve_sglang.sh` and
    `--input {{script:tpu-model-onboarding/fetch_weights.py}}:fetch_weights.py`.

## Task 3: Launch and Monitor Deployment

### Step 3a: Submit the daemon

`submit` handles the daemonisation this task used to spell out by hand: it runs
the recipe detached, records the pid on the target, captures stdout and stderr,
and writes a durable job record.

**Pass `--serve` only when the caller asked for a running endpoint.** It marks
the job exempt from the session-end reaper, which is right for a service the
user wants kept and wrong for a deployment that exists only to be verified --
omit it and the reaper releases the chips even if this workflow dies before
tearing down.

**Reuse a live server before starting another.** Decide from this run's own
job records, not from the cluster: `compute job list --run-id <run_id>` reads
compute's store of the jobs this run submitted, and never lists anyone
else's. A `RUNNING` record is this deployment's server only if it is the
`job_id` an earlier onboarding result returned for it, or carries the
`--name <model>-serve` this run gave it. A new server spends 15-30 minutes
starting up, and two hold twice the chips. Reuse it if any runtime limit
(the compute profile's `declared_duration_minutes`, from `config show`) still covers
what is left to do: the limit less the time since the `start_time` that
`job status` shows.

Both recipes take `fetch_weights.py` as an input. For `custom_fastapi`:

```bash
{{runtime}} compute submit --package {{package_tag}} \
  --recipe <inference_work>/fastapi_server.py \
  [--serve] [--shm-size-gb <N>] \
  --input <inference_work>/deployments.json:deployments.json \
  --input {{script:tpu-model-onboarding/fetch_weights.py}}:fetch_weights.py \
  [--input <patches_path>:model_patches.py] \
  --name <model>-serve
```

For `vllm`, submit `<inference_work>/launch_vllm.sh`. Every vLLM server ships `serve_vllm.sh`; a model in the patch registry also
ships its patch and the applier:

```bash
{{runtime}} compute submit --package {{package_tag}} \
  --recipe <inference_work>/launch_vllm.sh \
  [--serve] [--shm-size-gb <N>] \
  --input {{script:common/serve_vllm.sh}}:serve_vllm.sh \
  [--input {{script:common/vllm_patches/apply_vllm_patch.sh}}:apply_vllm_patch.sh \
   --input {{script:tpu-model-onboarding/fetch_weights.py}}:fetch_weights.py \
   --input {{script:common/vllm_patches/gpt_oss_mxfp4.py}}:gpt_oss_mxfp4.py] \
  --name <model>-serve
```

Pass `--shm-size-gb` only on a resubmission Step 3e asks for: it overrides
the profile's `shm_size_gb` for this Job alone.

Then find the job id the same way:

```bash
JOB_ID=$({{runtime}} compute job list --package {{package_tag}} --json --no-refresh \
  | python3 -c "import json,sys;print(next(r['job_id'] for r in json.load(sys.stdin) if r['job_name']=='<model>-serve'))")
```

Keep `$JOB_ID`. It replaces the PID this task previously tracked in
`inference_pid.txt`: the pid is meaningful only on the target, while the job id
addresses the deployment from anywhere, including a later session.

### Step 3b: Wait for the server to come up

Wait with `compute job wait`, never with a `sleep` or polling loop of your
own. A hand-written loop is killed without a word when it outlives the shell's
ten-minute limit, and one that watches only for success misses an engine that
is crashing while its job still reads `RUNNING`. `job wait` checks the job's
status and its logs together, gives up on its own after nine minutes, and
exits with a code per outcome.

Wait in three stages: the job is admitted and starts, the weights are in
the cache, the server is up. Waiting on the weights separately is what tells
a download from a compile -- both are silent to the server's ready line.

```bash
{{runtime}} compute job wait --package {{package_tag}} --job-id $JOB_ID \
  --until running --json
{{runtime}} compute job wait --package {{package_tag}} --job-id $JOB_ID \
  --until "log:WEIGHTS READY" --fail-on-log 'WEIGHTS FAILED' \
  --fail-on-log 'Traceback \(most recent call last\)' --json; rc=$?
{{runtime}} compute job logs --package {{package_tag}} --job-id $JOB_ID \
  | grep -E '^WEIGHTS (PLAN|PROGRESS|FAILED)' | tail -n 2
echo "weights wait exit: $rc"
```

Exit 3 here means still downloading: report the last `WEIGHTS PROGRESS`
line and run the wait again. If two waits in a row show the same byte count,
the download has stalled; treat it as an infrastructure failure (Step 3e).
A gated model fails at once with `WEIGHTS FAILED` naming the 401; that is a
missing `HF_TOKEN` on the target, not a configuration to change. A model
too large for the cache also fails before downloading anything, with
`WEIGHTS FAILED` naming the GB it needs, the GB free and, on `/dev/shm`, the
`shm_size_gb` that would hold it: that is the *Disk* failure in Step 3e.

Both engines run under uvicorn, which prints `Application startup complete.`
once the server accepts requests:

```bash
{{runtime}} compute job wait --package {{package_tag}} --job-id $JOB_ID \
  --until "log:Application startup complete" \
  --fail-on-log 'Traceback \(most recent call last\)' --json; rc=$?
{{runtime}} compute job logs --package {{package_tag}} --job-id $JOB_ID \
  > <inference_work>/serve_log_<attempt>.txt 2>&1
echo "job wait exit: $rc"
```

Run the ready-line wait and the `job logs` as one command, as above. A serving
engine that crashes is restarted in place within seconds, and `job logs`
shows only the current container: fetched in a later command, the log is
the restarted engine's start-up, and the exception that ended the wait is
gone.

For a patched `vllm` recipe, also pass `--fail-on-log 'vllm patch does not
apply'`: the applier exits before vLLM starts, which raises no `Traceback`.

Always pass the `Traceback` pattern. The built-in patterns catch exhausted
memory and a full disk, not a Python exception, and a serving job whose engine
raises at startup is restarted in place with the same arguments while its
status keeps reading `RUNNING`. Without the pattern, the wait sees nothing
wrong until the restarts run out.

| Exit | Meaning | What to do |
| :--- | :--- | :--- |
| 0 | Reached | Continue to Step 3c. |
| 1 | `job wait` itself failed: a bad flag, or the job's status could not be read three times in a row (an unknown job id, an unreachable cluster) | Read the error on stderr. Do not treat the job as ended; check `job status` once the cause is fixed. |
| 2 | A failure line appeared while the job still runs | `line` is only the matched line; read the log tail for the exception itself. Go to Step 3e. Rerun with `--no-default-fail-patterns` or without the `Traceback` pattern only if you know the workload recovers from this exception. |
| 3 | Timed out, still waiting | Read `next` in the result and do what it says. "Run the same `job wait` again" is the usual answer: the weights are already in (the previous wait saw `WEIGHTS READY`), so this is the engine loading and compiling, and a vLLM engine compiling on a cold cache takes well past nine minutes. Any other `next` names why the job is still `PENDING` -- held by a queue, bound to a node, or with no node and no queue -- and what to do; follow it, and report its reason rather than waiting on. |
| 4 | The job ended without reaching it | Read `line` and `error`, then the log tail. Go to Step 3e. |

The ready line is uvicorn's own start-up message, printed at its default `info`
log level. A recipe that lowers uvicorn's log level never prints it, and every
wait times out; keep the default.

Read logs with `compute job logs`, and bring only the tail into the
conversation.

### Step 3c: Forward the port

```bash
{{runtime}} compute port-forward --package {{package_tag}} \
  --port <port> --job-id $JOB_ID --background
```

Always pass `--background`. The foreground form blocks until interrupted, which
for an agent means forever. It prints a PID to kill later; on the `local`
backend there is no forward to make and no PID is printed, which is expected,
not a failure.

### Step 3d: Poll health

This is now an ordinary local HTTP call, because the port is here:

```bash
python3 {{script:tpu-model-onboarding/health_probe.py}} \
  --port <port> --timeout 300 --output <inference_work>/health.json
```

Read `<inference_work>/health.json`. Step 3b already waited out the engine's
start-up, including XLA compilation, so this only has to confirm the forward
works; a short timeout keeps it under the shell's ten-minute limit. If it
times out, the server said it was up, so connection-refused points at the
forward: check that its PID is alive and re-establish it with Step 3c.

For `sglang`, once health passes, check whether SGLang lowered the requests
per replica to fit its compiled batch sizes. Its log names the value it
runs with:

```bash
{{runtime}} compute job logs --package {{package_tag}} --job-id $JOB_ID \
  | grep 'TPU batch limits after reconciliation' | tail -n 1
```

When that line's `max_running_requests` is below the manifest's, set
`max_running_requests` in `deployments.json` and `--max-running-requests`
in `launch_sglang.sh` to the line's value, and add the change to
`parallelism.caveats`. Do not relaunch: the running server already uses the
lower value, and later steps read it from the manifest. With no such line,
or a value equal to the manifest's, change nothing.

### Step 3e: When a deployment fails

Come here from Step 3b exit 2 or 4. Resubmitting the same deployment reproduces
the same failure, so every resubmission must answer a line from the log.

1.  **Read the saved logs.** Step 3b saved them to
    `<inference_work>/serve_log_<attempt>.txt` in the same command as the
    wait. Read the exception from that file; a fresh `job logs` may already
    show a restarted engine.
2.  **Cancel, even if the job already reads `FAILED`.** The backend may still
    be retrying a serving job with the same arguments -- on GKE, up to six
    new pods -- and each retry holds the chips and fails the same way.
    `job cancel` (Task 5) stops the retries.
3.  **Decide which kind of failure it is.**
    -   *Patched registry model (`registry.json`)*: when the model is listed
        in {{script:common/vllm_patches/registry.json}} and the deployment
        with the current candidate from `vllm_patches[<model_id>].candidates`
        fails (`vllm patch does not apply`, startup/compile crash, or Task 4
        verification failure): if `candidates` has a second untried bracketing
        candidate with `applies: true` (`before`/`after` when `version_match`
        is `"between"`), update `deployments.json` and `launch_vllm.sh` to
        that candidate's `patch` and `serving` settings and submit once under
        a new `--name`. Otherwise (including when `version_match` is
        `"newer"`, where only the last validated version is tried, or when
        both candidates fail), **end your work immediately and return
        `status: "needs_input"`** with the installed `vllm` version, tried
        candidate(s), and failure; do not run generic retries, fall back to
        `custom_fastapi`, or spend compute cycles fixing the patch without
        user confirmation.
    -   *Disk*: the log's `WEIGHTS FAILED` line says the model needs more
        GB than the cache has free, or the wait's `error`, or `job status`,
        reads `pod was evicted` with `EmptyDir volume`, `ephemeral-storage`
        or `ephemeral local storage` in it, or the wait stopped (exit 2) on
        `No space left on device` or `Raising signal 7`. For an eviction,
        the log shows no exception and the container exited 137, but the
        recipe filled the pod's storage. This is a configuration failure.
        -   `WEIGHTS FAILED` ending `Set shm_size_gb to at least <N>`: the
            weights and the room kept for compile caches do not fit
            `/dev/shm`, where `gke_job` puts them when there is no
            `hf_cache_pvc`. Resubmit under a new `--name` with
            `--shm-size-gb <N>` (Step 3a). Do not size it yourself: `df` on
            `/dev/shm` shows the node's whole memory, not the cap.
        -   `EmptyDir volume`, or, with no `hf_cache_pvc`, `No space left on
            device` or `Raising signal 7` after `WEIGHTS READY`: the weights
            fit, and the compile caches likely outgrew the room kept for
            them. Where `/dev/shm` is capped at its size, a full one fails
            the engine's writes instead of evicting the pod, so the line may
            sit in a cache warning before the `Traceback` it causes.
            Resubmit once under a new `--name` with `--shm-size-gb` twice
            the cap the log's `WEIGHTS SPACE` line names. A full disk
            elsewhere (a cache the image or launcher moved) can raise the
            same line: if it comes back with twice the cap, `/dev/shm` was
            not what filled. Report the line, and stop.
        -   After either resubmission, the pod is evicted naming `memory`:
            the node cannot back that much `/dev/shm`. Report that the
            model needs `hf_cache_pvc`, and stop. A resubmission that stays
            `PENDING` is not this: `/dev/shm` does not affect scheduling,
            so it is waiting on the queue or for chips (Step 3b).
        -   `WEIGHTS FAILED` that names a cache outside `/dev/shm` (a
            `hf_cache_pvc`, or a cache the launcher or the image moved): that
            volume is out of space, which no resubmission changes. Report
            it, and stop.
        -   `ephemeral-storage` or `ephemeral local storage`: something
            wrote to the container's own disk. Find where the launcher or
            the engine sets a cache or temporary directory (`HF_HUB_CACHE`,
            `HF_HOME`, `VLLM_CACHE_ROOT`, `XDG_CACHE_HOME`,
            `TORCHINDUCTOR_CACHE_DIR`, `TMPDIR`, a `download_dir`) outside
            `/dev/shm`, leave it at the pod's default, and resubmit under a
            new `--name`.
    -   *Infrastructure*: the log shows no exception and the job was killed
        from outside -- exit code 137 or 143 with no `Traceback` and no disk
        eviction named -- or the job stays `PENDING` for want of chips.
        Node preemption, taint eviction and memory pressure on the node look
        like this. Report it and stop; the configuration is not at fault,
        so there is nothing to change.
    -   *Timeout*: the log names one of the engine's timeout lines below,
        while the lines before it still show the engine compiling or loading
        (`Compiling FX graph`, `TpuCompilerAdaptor`, weights loading) and no
        exception other than the timeout itself. The server was healthy and
        slow, not misconfigured: keep the model's configuration. Resubmit once
        under a new `--name` with the timeout that line names doubled,
        exported in the launcher before its `exec bash serve_<engine>.sh`:
        -   vLLM (`launch_vllm.sh`): `Timed out waiting for engine core
            processes to start`, or a `TimeoutError` from an RPC or a
            collective (`VLLM_ENGINE_READY_TIMEOUT_S`,
            `VLLM_ENGINE_ITERATION_TIMEOUT_S`,
            `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` or
            `VLLM_DISTRIBUTED_TIMEOUT_S`).
        -   SGLang (`launch_sglang.sh`): `Scheduler watchdog timeout` with
            `self.soft=False` (`SGLANG_WATCHDOG_TIMEOUT_S`), `Initialization
            failed. warmup error` followed by a `ReadTimeout`
            (`SGLANG_WARMUP_TIMEOUT`), or a `TimeoutError` from a collective
            (`SGLANG_DIST_TIMEOUT_S`).

        If it times out again at the doubled value, report the last compile
        line before it, and stop.
    -   *Configuration*: the log names an exception (`ValueError`,
        `RESOURCE_EXHAUSTED`, a vLLM argument error). Change the manifest or
        launcher so it addresses that exception, then resubmit under a new
        `--name`. A memory error at the planned TP is answered by a larger
        TP, and a larger TP needs fewer replicas: TP x DP must still equal
        the device count. Rerun the `engine-compatibility` check with
        `--tp <the larger TP>` (the next divisor of the device count above
        the failed one) and the same `--points`, and take TP, DP and
        requests per replica from the new `tp_plan.json`, as in Task 1; never
        change TP alone. When no larger divisor exists, or
        the new plan's `ok` is false (the heads or the device layouts rule
        that TP out), keep the planned TP and DP and shrink what each
        engine holds instead. A hybrid model (the plan's
        `linear_attention_heads` is set) keeps a state slot per concurrent
        sequence, so lower the requests-per-replica key first when its flag
        appears in the flags file (Task 1's table); otherwise, or for any
        other model, lower the context-length key. Update
        `parallelism.reason` to name the
        memory error.
    -   *Unknown*: `job logs` could not read the logs. Say so, include its
        error, and stop. Do not look for them another way.
4.  **Record each attempt** in `<inference_work>/attempts.json`: the job id,
    the exception line, and what you changed. A change that does not answer
    that line is not a new attempt.
5.  **Stop after two resubmissions.** Report every attempt from
    `attempts.json` and hand the decision back to the user.

## Task 4: Output Verification

```bash
python3 {{script:tpu-model-onboarding/verify_inference.py}} \
  --port <port> --engine <engine> --model-id <model_id> \
  --output <inference_work>/verification.json
```

`verified: true` requires a 200 **and** a completion that is neither empty nor
degenerate: not one character repeated (`!!!!` is what NaN logits decode to),
not a phrase looping (`the the the`), and not mostly `<unk>`/`<pad>`/U+FFFD.
`failures` in the report names which check failed. A 200 that fails any of
them is unverified, and you should report it that way too: the engine
answered, the model did not.

The request is greedy and goes through the model's chat template, so rerunning
the check gives the same verdict. After a pass, the script also asks a
known-answer question; a wrong answer lands in `warnings`, not `failures`.
Report each warning to the user as-is. Do not treat it as a failed deployment,
and do not hide it.

This is a functional check only. Fluent output can still be numerically wrong;
that is what the numerical parity step exists for.

A model served with a registry patch needs more than a non-empty completion:
an unpadded MoE returns `!!!!` (token 0) and still passes. Ask
{{agent:tpu-knowledge-base}} for the entry's `provenance` keyword and send the
playbook's verification prompts with `temperature: 0` and `logprobs: 5`, eight
at a time so every data-parallel replica answers. Report the deployment
verified only when every response is identical, matches the playbook's
known-good text, and the logprob requests return 200.

## Task 5: Teardown

Cancelling the job is the teardown. It signals the whole process group the
recipe was launched in, which reaches workers the framework spawned --
including the ones vLLM disguises under names like `VLLM::Worker_TP0`:

```bash
{{runtime}} compute job cancel --package {{package_tag}} --job-id $JOB_ID
```

Read the logs before cancelling if you still need them. Kill the port-forward
PID too, if one was printed.

If the service was submitted with `--serve`, nothing cancels it for you -- that
is the point of the flag, and it means a forgotten deployment holds its chips
until someone ends it. Cancel deliberately when the user is done.

Without `--serve`, the session-end reaper will cancel it, but do not rely on
that: it fires only when the session ends, so a verification-only deployment
left running holds the chips for the rest of the session. Cancel it as soon as
verification is done.

Only if a later deployment then fails with `Device or resource busy` is there
anything more to do. That means a process survived the signal, or an earlier
crash orphaned one, and it is still holding the TPU device nodes. Report first:

```bash
{{runtime}} compute run --package {{package_tag}} \
  --code {{script:tpu-model-onboarding/tpu_lock_reset.py}} \
  --timeout 120 --output tpu_locks.json:<inference_work>/tpu_locks.json \
  -- --output tpu_locks.json
```

If `clean` is false, read which processes are listed before clearing them --
on a shared host one of them may be someone else's workload. Pass `--force`
only once you have checked:

```bash
{{runtime}} compute run --package {{package_tag}} \
  --code {{script:tpu-model-onboarding/tpu_lock_reset.py}} \
  --timeout 120 --output tpu_locks.json:<inference_work>/tpu_locks.json \
  -- --force --output tpu_locks.json
```

## Unwired Model Serving Support

Models flagged `[STATUS: ONBOARDING_REQUIRED]` (e.g. Mamba SSM, SigLIP with
FlashAttention, or custom uncataloged CUDA ops) are unwired by tpu-unwiring
before you get here. You do not write patches. It returns `patches_path`, an
importable module exposing `PATCHED_MODULES`, `APPLIED_PLAYBOOKS` and
`apply()`, already proven to run a forward pass on the accelerator.

-   The generated server imports `model_patches` and calls `apply()` **before**
    `AutoModel.from_pretrained()` is called. Never copy the patches into the
    server script: the numerical parity check imports the same module in its
    own process, and cannot import patches that live inside a running server.
-   The module travels to the target as a second `--input` on the submit in
    Task 3a, beside the manifest: `--input <patches_path>:model_patches.py`.
-   Unwired models are served with the `custom_fastapi` recipe; the `vllm`
    launcher does not load a patches module.
