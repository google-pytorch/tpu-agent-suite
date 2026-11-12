---
scripts:
  - scripts/common/tpu_slots.py
  - scripts/tpu-model-onboarding/stage_checkpoint.py
references:
  - references/tpu-model-onboarding/preflight-check.md
  - references/common/engine-compatibility.md
  - references/tpu-model-onboarding/torch-inference-coder.md
  - references/tpu-model-onboarding/numerical-parity.md
  - references/common/inference-serving-benchmark.md
---
# Torch TPU Inference Workflow

The end-to-end deployment of a model for HTTP inference on Cloud TPU. You run
these steps yourself; only CUDA unwiring is delegated.

## Required Inputs

-   `model`: Identifier for the model to be deployed. This can be one of the following:
        a. HuggingFace Model ID
        b. Local path to model code
        c. Local path to checkpoint file
        d. URL to checkpoint file or model repo
-   `engine` (Optional): The inference engine to use. When omitted, defaults to
    `vllm` for supported LLM / CausalLM architectures, and automatically falls
    back to `custom_fastapi` for non-LLMs (vision/audio/embeddings), unwired
    models, or unsupported architectures. Supported: `vllm`, `sglang`,
    `custom_fastapi`.

## Paths

`<artifacts>` and `<work>` are as tpu-model-onboarding defines them;
`<inference_work> = <work>/inference`.

## Two Modes: Read Your Request First

This workflow deploys and verifies a model either way. Whether the deployment
survives the run is decided by the mode you chose from the request:

-   **Serving mode** -- the user asked for a running endpoint. Submit with
    `--serve`, which exempts the job from the session-end reaper, and leave it
    running. Their endpoint persists, which is what they asked for; nothing
    cleans it up, so the hand-off must say that cancelling is manual.
-   **Verification mode** -- verification only, tear down when done; the
    default when the request says neither. Submit **without** `--serve`,
    verify, then cancel at Step 6. A request that did not ask for a server
    must not leave one holding the chips.

Omitting `--serve` in verification mode is load-bearing, not a shortcut: a job
without it is cancelled by the session-end reaper, so if this workflow dies
between deploying and tearing down, the chips are still released. `--serve`
would survive that failure and hold them indefinitely.

The endpoint is reached through a **port forward** that this workflow establishes.
That forward is a process on this host and does not survive the session. In
serving mode the service does, and a later session re-establishes the forward
with `compute port-forward --job-id <id> --background`.

## Step-by-Step Inference Workflow

You MUST follow this checklist in order. Check off each step and proceed to the
next.

-   [ ] **Step 0.1: Compute Runtime Setup & Hardware Diagnostics**: When the
    request asks for numbers at more than one `isl:osl` shape, the mode is
    serving mode and the session has a `gke_job` profile, do Step 0.1b first: it picks the TPU this
    deployment runs on, and everything below sizes for that TPU. Read the
    active profile's `backend` and `params` with
    `{{runtime}} config show --json --package {{package_tag}}`; do not run
    `compute inspect`. Work out `<JOB_DEVICES>` and `<HBM_GB_PER_DEVICE>` for
    Step 0.2 as tpu-model-onboarding's sizing says: per-chip specs from the
    knowledge base, chips from the profile.

-   [ ] **Step 0.1b: Pick the TPU and book its slot (only for a sweep of
    several shapes that Step 5b runs here, on `gke_job` in serving mode;
    before the rest of Step 0.1; never for vLLM unless `handback.run_here`,
    since Step 2b claims no slot)**: Step 5b will
    run the shapes on several servers at once, every one of them the same
    kind of TPU as this deployment: one accelerator, topology and image. So
    put this deployment on the kind with the most room for servers like it.
    The session's slots are the ones its compute profiles declare (step 1
    of "A sweep on several servers" in
    `{{reference:common/inference-serving-benchmark.md}}`). Take the slot
    with the most free slots of its kind beside it; with no `parallel.json`
    yet, the claim writes it from the profiles:

    ```bash
    python3 {{script:common/tpu_slots.py}} claim --run-id <run_id> \
      --holder tpu-model-onboarding:<deployment name> --most-available \
      --runtime "{{runtime}}" --package {{package_tag}}
    ```

    It prints the `resource`, its `profile`, its `hardware` key and
    `free_like_this`, the slots of that kind still free for the sweep's other
    servers. Keep all four. Point this run at that profile, so Step 0.1's
    sizing and every job after it use it:

    ```bash
    {{runtime}} config use <profile> --package {{package_tag}}
    ```

    Say in your result which TPU you picked and why. A `needs_input` is your
    caller's to answer: return it with the `error`. On `refused` (no slot of
    a known kind is free), go on with the configured profile and one server;
    Step 5b then measures the shapes one after another in one client. Skip
    this step for one shape, verification mode, or another backend: Step 5b
    measures those with one client, one shape after another.
-   [ ] **Step 0.2: Pre-flight Compatibility Check**:
    **Execution order between Step 0.2 and Step 0.2b depends on `model`:**

    | `model` input | Execution Order | Why |
    | :--- | :--- | :--- |
    | **Not a local checkpoint directory** (Hugging Face Hub ID or `gs://` folder) | **Step 0.2 first, then Step 0.2b** | Preflight runs first; if compatible, Step 0.2b stages Hub weights to GCS before serving jobs run. |
    | **Local checkpoint directory** (and backend is not `local`) | **Step 0.2b first, then Step 0.2** | Remote preflight jobs cannot read the host disk; stage the local directory to GCS in Step 0.2b first, then run Step 0.2 on the staged `gs://` folder. |

    Read `{{reference:tpu-model-onboarding/preflight-check.md}}` and
    `{{reference:common/engine-compatibility.md}}`
    and follow them to complete preflight compatibility checks on the
    model (or on the staged `gs://` checkpoint folder if Step 0.2b ran first)
    with `--allow_gpu_imports --cuda_report_json=<artifacts>/cuda_kernels_report.<model_slug>.json --mode=inference`.
-   [ ] **Step 0.2b: Stage the checkpoint (Hugging Face models or local checkpoint directories, when a bucket is known)**:
    **Ordering:** Run **after Step 0.2** when `model` is a Hugging Face Hub ID (skip for an already-staged `gs://` folder), or **before Step 0.2** when `model` is a local checkpoint directory on a remote backend (if you already ran Step 0.2b before Step 0.2, check this off and move to Step 0.3).

    Put the weights in a bucket once, or reuse the copy already there, so no
    job after this downloads the model from the Hub again (and so remote jobs
    can read a local checkpoint directory). The bucket must be
    one the serving jobs can read. On `gke_job`, read the params of the
    profile this run uses (`{{runtime}} config show --json --package
    {{package_tag}}`): its pods reach GCS through the profile's
    `pod_service_account` when it has one, and otherwise only through a token
    that reads the `checkpoints/` folder under its `staging_bucket` (and writes
    only the job's own folder). Take the first that applies:

    1. The request's `checkpoint_bucket`, when the jobs can read it: any
       backend other than `gke_job`, a `gke_job` profile with a
       `pod_service_account`, or the same bucket as its `staging_bucket`.
    2. The profile's `staging_bucket`, when set. When it names a folder
       (`gs://<bucket>/<folder>`), pass `--folder <folder>/checkpoints`; the
       script uses only the bucket from `--bucket`. When it replaces a
       `checkpoint_bucket` the pods could not read, say so in your result.
    3. Neither: for a Hugging Face model, skip this step and use the Hub id as
       before. For a local checkpoint directory on a remote backend, **stop and
       return `needs_input`**: remote jobs cannot read the host disk without a
       GCS bucket. In `summary`, explain that staging the local checkpoint
       requires a `gs://` bucket and give `{{runtime}} compute setup plan --fields staging_bucket`
       (on `gke_job`) or ask for a `gs://` `checkpoint_bucket` the TPU target
       can read (on other remote backends). Do not run the setup command
       yourself; setup is done with the user in the main session.

    ```bash
    python3 {{script:tpu-model-onboarding/stage_checkpoint.py}} <model_id> \
      --bucket <bucket> [--folder <folder>/checkpoints] [--revision <revision>] \
      --out <artifacts>/checkpoint.json
    ```

    Its `checkpoint` (a `gs://` folder) is **the model from here on**: pass it
    wherever these steps say model, model path or model id, and keep the Hub
    id (or local directory's base name) as `--served-model-name`. A sweep's followers then read it too,
    instead of each downloading the model again. Say in your result which
    bucket you used and whether the copy was `reused` or staged now.

    A running server is not affected if the copy goes away later: each job
    copies the checkpoint to its own disk when it starts. A job that starts
    later can be, when a lifecycle rule on the bucket ages the copy out (a
    staging bucket usually has one). So:

    -   Run this command again, with the same arguments, right before any later
        job that loads the model (a relaunch after a failure, a sweep): it
        only checks the copy when it is there and stages it again when it is
        gone.
    -   When the bucket's lifecycle rule covers the whole bucket, say in your
        result that it can age out `checkpoints/`, and that scoping the rule
        to the staging folder (a `matchesPrefix` condition) avoids it.
-   [ ] **Step 0.3: CUDA Unwiring (only if `ONBOARDING_REQUIRED`)**: If
    preflight flagged the model `[STATUS: ONBOARDING_REQUIRED]`, delegate to
    {{agent:tpu-unwiring}} with `model_id`, `run_id`, `target: server`, and the
    fully expanded `cuda_kernels_report` path. It returns `patches_path`, a
    patches module that has run a forward pass on the accelerator. Unwired models are served with `custom_fastapi`; the `vllm`
    engine does not load a patches module, so switch the engine and say so.
    On `blocked` or `failed`, update `<artifacts>/cuda_kernels_report.<model_slug>.json`
    with `status: "BLOCKED_ON_CUDA_KERNEL"`, report `blocked_on_cuda_kernel`,
    and stop.
-   [ ] **Step 1: Create deployments.json**: Read
    `{{reference:tpu-model-onboarding/torch-inference-coder.md}}`
    and, following it, create a declarative service manifest at
    `<inference_work>/deployments.json` defining the model ID, target engine,
    sequence lengths, and port. Resolve the engine using the 3-tier hierarchy:
    1. *vLLM / SGLang Supported*: When `engine` is omitted and the model is a CausalLM supported
       by vLLM (verified in Step 0.2), set `engine: "vllm"` (or `engine: "sglang"` when the user requested SGLang).
    2. *Golden Model / Patchable*: If minor gaps have an available playbook,
       proceed with `engine: "vllm"` (or `engine: "sglang"`).
    3. *Automated Fallback to Native PyTorch*: If the model is a non-LLM, unwired
       (Step 0.3), or rejected by engine compatibility, automatically set
       `engine: "custom_fastapi"` without asking the user.
-   [ ] **Step 2: Inference Code Generation**: Generate the required serving
    wrapper under `<inference_work>/` as per the torch-inference-coder
    instructions -- `fastapi_server.py`, `launch_vllm.sh`, or `launch_sglang.sh`. If Step 0.3 ran,
    the server imports the module at `patches_path`. The
    manifest is not baked into the script; it travels beside it as an
    `--input` when the job is submitted.
-   [ ] **Step 2b: Hand a vLLM deployment back (vLLM on `gke_job`, with
    the image pinned by digest; not in candidate mode)**: launch nothing and
    claim no slot; the work you line up claims its own when it runs. Any
    other backend, or an image named by tag only, continues at Step 3. Run
    Step 4 (parity) when it applies and skip Steps 0.1b, 3, 5, 5b and 5c.
    Line up the serving as tpu-serving-bench requests
    ({{schema:serving-bench-request.json}}), each with:
    -   this `run_id`, `engine: "vllm"`, `model`, `tpu`, and a `label`
        unique to the request and the attempt (`onboard-bench-1`,
        `onboard-serve-1`; the next attempt after a `handback.failure`
        uses `-2`);
    -   one `baseline: true` arm on the deployment's image by digest,
        with `patches` the registry patches the recipe applies (each
        file `apply_vllm_patch.sh` is given), and the recipe's own
        `apply_vllm_patch.sh` lines removed, since serve_vllm.sh applies
        the arm's patches;
    -   `deployment`: `launch_recipe`, `deployments_json`, `port`,
        `max_model_len` (the context-length key in `deployments.json`),
        `inputs` (every `--input` Step 3 would submit, `fetch_weights.py`
        included, except `serve_vllm.sh` and the patch files), and
        `interpreter` and `submit_args` when Step 3 would pass them.

    First a `bench` when the request asks for numbers, with `profile:
    false` and `client_args` the client flags Step 5b would pass (its
    measurement flags, `--max-concurrency-limit`, `--model` and
    `--tokenizer`); then a `serve`, which verifies the deployment. Write
    them, in that order, as a JSON list to
    `<inference_work>/serving_requests.json`, and return `status:
    "serve_pending"` with `output_files.deployments_json`,
    `output_files.launch_recipe` and `output_files.serving_requests`.
    tpu-agent sends them in order and wakes you with `handback`. On that
    wake-up, `<inference_work>` already holds Steps 0 to 2 and 4. Copy
    `handback.verification` to `<inference_work>/verification.json` as Step
    5's report, record the benchmark's files as Step 5c says, and go to Step
    6. A `handback.failure` goes back to Step 2 with its logs, as Step 3's
    failure path says, and returns `serve_pending` again with new labels.
    With `handback.run_here`, continue at Step 3 as for any other engine.
-   [ ] **Step 3: Deploy, Forward, Monitor Health (not for vLLM, Step 2b,
    unless `handback.run_here`)**: Using the
    torch-inference-coder instructions, submit the recipe (with `--serve` in
    serving mode only) under a `--name` you choose, then recover the **`job_id`** with
    `compute job list --json` -- never by command substitution on `submit`,
    which hangs on a daemon, and never by redirecting it to a file. Wait for
    the server with `compute job wait` -- `--until running`, then
    `--until "log:WEIGHTS READY"` (the weight download), then
    `--until "log:Application startup complete"` -- never with a sleep or
    polling loop of your own; exit code 3 means run it again. Pass
    `--fail-on-log 'Traceback \(most recent call last\)'` so an engine
    crashing while its job reads `RUNNING` ends the wait, and save
    `job logs` in the same command, before the engine restarts. Then forward
    the port with `--background` and probe health into
    `<inference_work>/health.json`. On a failure, follow torch-inference-coder
    Step 3e: save the logs, cancel even a `FAILED` job, resubmit only with a
    change that answers the logged exception, and stop after two
    resubmissions or on an infrastructure failure.
-   [ ] **Step 4: Numerical Parity**: **MANDATORY before concluding the model works.**
    Read `{{reference:tpu-model-onboarding/numerical-parity.md}}` and
    verify the unwired model still computes the same numbers as the model's own
    plain-PyTorch path. A server that starts, answers, and produces fluent text
    can still be computing the wrong thing; nothing before this step would
    notice. If Step 0.3 ran, use its `patches_path` as `--patches`, and its
    `probe_path` and `candidate_capture` already exist: run only the reference
    capture and the compare, in the directory that holds them. You judge
    the verdict, not the agent that wrote the patches.
-   [ ] **Step 5: Output Verification**: Using the torch-inference-coder
    instructions, send a test request through the forwarded port and write the
    report to `<inference_work>/verification.json`. Treat a 200 with an empty
    or degenerate completion (its `failures`) as unverified, and pass any
    `warnings` on to the user.
-   [ ] **Step 5b: Serving Benchmark (only if the request asked for numbers)**:
    Follow `{{reference:common/inference-serving-benchmark.md}}` before
    Step 6, with `<bench_work>` = `<inference_work>`, `--engine` set to the
    deployment's engine, and never a client you write yourself. If the
    request supplies its own benchmark, run that instead, as the reference's
    "A benchmark the caller supplies" says. Skip the step when no numbers were
    asked for. Every backend that serves gets a benchmark; take the first
    route that applies:

    | When | Route | Shapes |
    | :--- | :--- | :--- |
    | Step 0.1b booked slots (`gke_job`, serving mode, several shapes) | the reference's "A sweep on several servers" | side by side on several servers |
    | `gke_job` in serving mode | its client beside the server ("Beside the server") | one after another, in one client |
    | any other backend that serves (`local`, `gce_tpu_vm`, `gke_pod`), or `gke_job` in verification mode | its client on this host through Step 3's port forward ("Through a port forward") | one after another, in one client |

    One client takes every shape the request names in one run, with the
    flags "The measurement's flags" gives for the request: the
    single-measurement flags for one measurement, the onboarding sweep's for
    several shapes or concurrencies (`--max-concurrency-limit` = requests
    per replica x DP from `deployments.json`: `max_num_seqs` x
    `data_parallel_size` for `vllm`, `max_running_requests` x `dp_size` for
    `sglang`, with DP 1 when the DP key is absent; left out when the
    manifest has no requests-per-replica key; and the request's `--avoid` when it
    states one). It measures the shapes one after another,
    each stopped early on its own, so the result is the same measurement as a
    sweep on several servers, only slower; say in your result which route ran
    and, for one client, that the shapes ran one after another.

    For a sweep on several servers, this server (submitted with `--serve` in
    serving mode) is its `--leader`, and the recipe, `--input`s, image and
    other flags Step 3 submitted it with are the sweep's server recipe. Pass
    it, after `--`, the same client flags one client would get, and
    `--max-model-len` from the context-length key in `deployments.json`
    (`max_model_len` for `vllm`, `context_length` for `sglang`). When the
    request lists `servers` that an earlier onboarding in this session left
    serving, add each whose `deployments_json` and `launch_recipe` files have
    the same contents as yours as another `--leader <job_id>@<profile>`;
    never one that differs, whose numbers would not compare. Return the plan
    as `output_files.sweep_plan`, and the merged `summary.json` as the
    measurement. If the sweep ends `invalid_input` or `refused` before any
    server took a shape, measure with one client instead, by the routes
    above.

    Measure with the recipe Step 3 deployed; do not relaunch with higher
    requests per replica (`--max-num-seqs`, `--max-running-requests`) or
    another TP or DP. The preflight check sized requests per replica for
    the sweep's top point, and a new server re-measures every point from
    the start. The limit stops a sweep short of its top point only when
    requests per replica x DP in `deployments.json` is below the highest
    `--points` concurrency. Then report it as the result: the concurrency
    the sweep stopped at, and what set the value: the KV cache (the plan's
    or the layout's value below that concurrency), Step 3e lowering it for
    a memory error, SGLang lowering it at startup (torch-inference-coder's
    Step 3d), or the user or
    the patch registry. A
    higher limit or another layout is the user's call, and a separate
    measurement.
-   [ ] **Step 5c: Record the benchmark (only if Step 5b ran a client here,
    or `handback` brought the benchmark's files)**: For handed-back files,
    return them as `output_files.bench_request`, `results_json` and
    `benchmark_report` and go to Step 6. Otherwise write
    `<inference_work>/bench_request.json` as
    `{{schema:bench-request.json}}` describes, so a later stage can measure a
    change on exactly the same points: the `route` Step 5b took, the
    client's arguments verbatim as one client run takes them (for a sweep,
    the ones after `--`), and the server it measured. For a sweep, add what
    `serving_sweep.py run` was given besides them: the plan path, recipe,
    `--server-input`s and `--max-model-len`. Return its path as
    `output_files.bench_request`, and the client's `summary.json` (a sweep's
    merged one) as `output_files.results_json`.
-   [ ] **Step 6: Hand-off**: branch on the mode you read at the top.
    -   *Serving mode*: leave the service running. Give the user the `job_id`,
        the forwarded port, the backend, the parity verdict, and the commands
        to inspect it (`compute job status`, `compute job logs`) and to end it
        (`compute job cancel`). Say that it persists beyond this session and
        that cancelling it is manual. Set `compute_profile` to the profile
        it runs under. If Step 0.1b claimed a slot, it stays held for the
        server: name its `resource`.
    -   *Verification mode*: tear the deployment down now, via
        torch-inference-coder Task 5, and kill the port forward. If Step 0.1b
        claimed a slot, release it, also when the workflow ends early:
        `python3 {{script:common/tpu_slots.py}} release --run-id <run_id>
        --resource <resource> --holder tpu-model-onboarding:<deployment name>`.
        Report the parity verdict, the verification result, and that the
        deployment was ended and the chips released. Do not hand back a
        `job_id` as if it were live.

> **CRITICAL**: In serving mode, do NOT tear down the inference server after
> verification unless explicitly commanded by the user. In verification mode,
> tearing it down is the last required step -- a run that ends without it has
> left TPU chips held.

## Step 7: Teardown & Lock Eviction (When Requested)

When the user explicitly requests teardown, or when you are restarting a
crashed server, follow torch-inference-coder Task 5. In short:

1. Read the logs first if they are still needed -- cancelling discards the
   process.
2. `{{runtime}} compute job cancel --package {{package_tag}} --job-id $JOB_ID`.
   This signals the whole process group, which is what reaches the detached
   workers frameworks like vLLM spawn. There is no separate `kill` step and no
   PID file to clean up. Kill the port-forward PID as well, if one was printed.
3. Only if a subsequent deployment fails with `Device or resource busy`, run
   the TPU lock report, read which processes are listed, and clear them with
   `--force` if none of them is a workload worth keeping.
