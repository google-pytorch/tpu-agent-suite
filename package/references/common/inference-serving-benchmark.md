---
scripts:
  - scripts/common/serving_benchmark.py
  - scripts/common/benchmark_fastapi_server.py
  - scripts/common/serving_sweep.py
  - scripts/common/serving_sweep_client.sh
  - scripts/common/tpu_slots.py
  - scripts/tpu-model-onboarding/fetch_weights.py
references:
  - references/tpu-model-onboarding/torch-inference-coder.md
---
# Inference Serving Benchmark

How to measure a model served over HTTP from the client's side of its
endpoint: latency per request and throughput, with every prompt new to the
server. Any workflow with a server already running uses it -- tpu-model-onboarding
for one measurement or a concurrency sweep of a fresh deployment. It never
deploys, configures or tears down the server.

## What the caller provides

-   The server: a `$JOB_ID` and `<port>` to reach it at, as "Run the client"
    below describes.
-   `<engine>`: `vllm`, or `custom_fastapi` for a server built from
    torch-inference-coder's FastAPI recipe
    ({{reference:tpu-model-onboarding/torch-inference-coder.md}}).
-   `<model_id>`: the Hugging Face id the server answers to.
-   `<bench_work>`: the directory the client's recipe and results go in.

Run it after the server has passed its own verification, and before
teardown.

## Why the client is fixed

A client written for the occasion tends to reuse one prompt. With vLLM's
prefix cache on, every repeat after the first finds the prompt's KV blocks
already computed and skips the prefill, so it reports a cache lookup as time
to first token and inflates throughput with it. The numbers look plausible and
nothing flags them. Leave prefix caching as the deployment has it -- real
traffic runs with it -- and send prompts the server has not seen.
{{script:common/serving_benchmark.py}} does that for every run it makes:

-   Every prompt is random, drawn from `--seed`, and every request runs to
    exactly the output length (`--ignore-eos`): random prompts are
    meaningless text, which a model may stop on at once or run on from.
-   No run hits an earlier run's prompts. Against vLLM it resets the prefix
    cache (`POST /reset_prefix_cache`) before every run; vLLM serves that
    endpoint only when the server starts with `VLLM_SERVER_DEV_MODE=1`.
    Without it, each run gets its own seed (42, 43, ...) instead.
    `settings.prefix_cache` in `summary.json` records which was used. The
    FastAPI recipe keeps no prefix cache, so it records `none`.
-   `--num-warmups` requests (default: the concurrency) are sent and
    discarded first, so the compilation of new shapes is not timed.
-   vLLM's warmups resend the run's first prompt, so one measured request per
    run is a cache hit even after a reset. Send enough prompts that one does
    not count: `--num-prompts 200` for a single measurement.

**Give every invocation against the same server its own `--seed`** when the
server has no cache reset: a second invocation with the first one's seed
resends its prompts, which are now cached.

Unless the caller supplies a benchmark of its own (next section), measure only
with this client. Never write a client of your own.

## A benchmark the caller supplies

A user may bring their own benchmark: a script, a command, or a dataset of
prompts. Their benchmark is the default's replacement, not its competitor:

-   **Run it as given.** Do not edit it, rewrite it into this client, or
    change its settings. Run the default client as well only when the request
    asks for both.
-   **Run it the same way.** It is a client like this one: write it, or a
    recipe that calls it, as `<bench_work>/bench_client.sh`, pass its files
    with `--input`, and submit it beside the server as "Run the client"
    shows. Its result files come back through `--output`.
-   **Check it for repeated prompts before quoting its numbers.** Against a
    `vllm` server, a benchmark that sends the same prompt more than once, or
    reruns a fixed prompt set, measures cache hits from the second request
    on. Do not fix it: run it, and say beside its numbers that its TTFT and
    throughput include prefix-cache hits.
-   **Report its numbers under its own name and keys**, as "the user's
    benchmark", with the command it ran. Map them onto the metric names below
    only where a name means the same thing; leave the rest as the benchmark
    reported them.

## The client

`serving_benchmark.py` sweeps input/output length x concurrency cells
(`--isl-osl`, `--concurrency`), runs each `--replicates` times, and writes
`summary.json` and a table in `summary.md`. `--engine` picks what drives each
run:

| `--engine` | Run by | Model and tokenizer |
| :--- | :--- | :--- |
| `vllm` (default) | `vllm bench serve`, the client vllm-torchtpu's own perf CI uses | Read from `/v1/models`. Pass `--tokenizer <model_id>` whenever the server loaded the model from a path: that path is on the server's filesystem, and the client exits `2` naming it when its own host lacks it. |
| `custom_fastapi` | {{script:common/benchmark_fastapi_server.py}}, beside it | `--model <model_id>` is required: the recipe has no `/v1/models`. `--tokenizer <model_id>` makes prompts exact in tokens and counts the output. |

For `custom_fastapi`, also pass `--task` with the recipe the server was built
from: `generate` (default), `embed`, or `vqa` with `--image-url`. And know
what it can measure:

-   **End-to-end latency only.** The recipe does not stream, so the first
    token is never observable on its own: there is no TTFT, TPOT or ITL. Do
    not estimate them.
-   **Concurrency above 1 measures queueing.** The recipe serves each request
    with its own `model.generate` call and no batching. Report a concurrency
    above 1 only when the request asks for it, and say so beside the numbers.
-   **Output length is held by `ignore_eos`.** The recipe honours the
    request's `ignore_eos` field. A server generated before the recipe had
    that field drops it without an error and stops at end-of-sequence: with
    `--tokenizer`, a `total_output_tokens` well below `completed` x the output
    length is that server. Report the shortfall; do not compare its latency
    with vLLM's.
-   **Without `--tokenizer`** (it cannot be loaded, say for a gated model with
    no token on the client's host), prompts become random words and every
    token count and token rate is absent. Say so in the report.
-   It is always greedy, and `--endpoint` and `--request-rate` do not apply.

**A model staged in GCS** has no Hugging Face id to pass: give `--tokenizer`
the `gs://` checkpoint folder instead, for either engine. The client copies
the folder's tokenizer and config files (no weights) to its own host with
{{script:tpu-model-onboarding/fetch_weights.py}}, using that host's
credentials, and reads the copy. A server that loaded the model from a
`gs://` folder is read the same way without `--tokenizer`.

Defaults keep the measurement deterministic: greedy decoding (`--temperature
0`), `--ignore-eos`, fixed lengths (`--range-ratio 0`) and seed 42. For vLLM,
`--endpoint chat` measures `/v1/chat/completions` instead of
`/v1/completions`.

### The AgentX workload

By default, the client sends random prompts of a fixed length. With
`--workload agentx`, the client runs the AgentX benchmark instead.

AgentX is an agentic-coding benchmark published by SemiAnalysis. It replays
recorded Claude Code sessions against the server. A session has many turns,
and each turn sends the whole conversation so far. The prompt therefore grows
with every turn, and consecutive turns share most of their prompt. AgentX
measures how the server handles this traffic: long prompts, and prompts whose
beginning the server has already processed and can reuse from its prefix
cache. The random workload does not test either.

The server needs no changes. Start it with `serve_vllm.sh` as for the random
workload, with the settings in [Server requirements](#server-requirements).

#### How a run works

1.  The client downloads InferenceX, the SemiAnalysis repository that holds
    the AgentX scripts, at a pinned commit. InferenceX includes aiperf, the
    load generator that sends the requests.
2.  For each value of `--concurrency`, the client starts one AgentX run. The
    concurrency is the number of sessions that aiperf replays at the same
    time.
3.  aiperf sends each recorded turn as a request to the server's
    OpenAI-compatible chat endpoint, `/v1/chat/completions`. InferenceX sets
    this endpoint; the client only passes the server's base URL.
4.  Each run has a warmup phase and then a measured phase of fixed length.
    The metrics come from the measured phase only.
5.  InferenceX validates the run and computes its results. The client copies
    the results into `summary.json` under the usual metric names.

The client calls the same InferenceX functions that the published AgentX
results use, so the requests, the validation and the result computation are
the same as AgentX's.

#### Server requirements

-   **Prefix caching on.** vLLM enables it by default, so do not pass
    `--no-enable-prefix-caching`. Without it, the server recomputes every
    turn from the start, and the results do not reflect normal serving.
-   **A long context.** `--max-model-len` decides which recorded sessions the
    server can take. The client reads it from `/v1/models` and picks the
    recordings as follows:

    | `max_model_len` | Recordings replayed |
    | :--- | :--- |
    | 1,000,000 or more | The full set. Its longest sessions reach about 1M tokens. |
    | 262,144 to 999,999 | The set capped at 256k tokens. aiperf skips any session longer than `max_model_len`. |
    | Under 262,144 | As above. The client prints a warning: more sessions are skipped, so the load is lighter than AgentX's. |

    To pick a set yourself, pass `--agentx-corpus`.
-   **Optional: `VLLM_SERVER_DEV_MODE=1`.** With it, the client can empty the
    server's prefix cache before each run. See
    [Prefix cache between runs](#prefix-cache-between-runs).

#### Client requirements

-   `git`, and network access to GitHub, PyPI and the Hugging Face Hub.
-   About 1.5 GB of disk in `--agentx-cache-dir` (default
    `<output-dir>/agentx_cache`). The first run downloads InferenceX,
    installs aiperf in its own Python environment and downloads the
    recordings there. Later runs reuse these files, so keep the directory
    between runs, on a disk with space.
-   No TPU, and no vLLM installation. The client only sends HTTP requests.

#### Duration

A run lasts the measured phase plus setup and warmup:

-   The measured phase lasts `--agentx-duration` seconds. The default, 1800,
    is the AgentX standard.
-   Loading the recordings and the warmup can add up to an hour.
-   `--agentx-fast` selects the AgentX quick mode for smoke tests: a shorter
    warmup and a 1200-second measured phase.
-   `--run-timeout` defaults to the measured phase plus 7200 seconds.

#### Comparing with published AgentX results

Only a standard run can be compared with published AgentX results. A
standard run has a measured phase of 1800 seconds, does not use
`--agentx-fast` and does not use `--agentx-coverage-ratio`.
`settings.agentx.canonical` in `summary.json` is `true` for a standard run.

AgentX also checks that the latency measurements cover at least 95% of the
measured phase, and fails the run otherwise. A slow server may fail this
check. `--agentx-coverage-ratio` lowers the required share, and makes the run
non-standard.

#### Metrics

`summary.json` uses the same metric names as for the random workload, with
these differences:

-   AgentX reports latency percentiles p50, p90 and p95. The summary has
    `*_p95_ms` metrics and no `*_p99_ms` metrics.
-   AgentX computes TPOT and ITL the same way, so `tpot_*` and `itl_*` are
    equal.
-   `prefix_cache_hit_rate` is the share of prompt tokens that the server
    found in its prefix cache, from the server's Prometheus metrics.
-   `theoretical_prefix_cache_hit_rate` is the highest hit rate the
    recordings allow.

A cell fails when more than `--max-failed-pct` of its requests fail. For
AgentX the default is 10%, the AgentX limit. A cell also fails when AgentX's
own validation rejects the run. The raw aiperf output and the AgentX result
of each run are in `raw/agentx_c<N>_r<replicate>/`.

#### Prefix cache between runs

When the server runs with `VLLM_SERVER_DEV_MODE=1`, the client empties its
prefix cache before each run, so each run starts with an empty cache.
Without it, a run can reuse cache entries that the previous run left behind,
and `settings.prefix_cache` in `summary.json` is `not reset`.

#### Unsupported options

-   `--isl-osl`, `--avoid`, `--max-concurrency-limit`,
    `--stop-below-gain-pct` and `--stop-on-failure` apply only to the random
    workload. The client exits with an error if you pass them.
-   `--engine custom_fastapi` is not supported: that server has no chat
    endpoint.
-   `serving_sweep.py` does not support AgentX. Run `serving_benchmark.py`
    once against the server instead.

## Run the client

The client only sends HTTP requests and holds no chips. Run it one of two
ways, and keep to one for every run that will be compared: the port forward
adds a roughly constant latency to TTFT and end-to-end latency that the
in-cluster client does not see.

Either way, the client usually runs where there are no chips: a pod beside
the server, a CPU-only node, or this host. On the torchtpu image the plain
`vllm` CLI does not start there: `import torch` autoloads torch_tpu, which
brings up libtpu, and importing vllm fails with `Expected one of cpu, cuda,
... device type: tpu`. `serving_benchmark.py` runs vllm as itself
(`serving_benchmark.py --as-vllm-cli`) whenever its interpreter has vllm,
which keeps the TPU stack out of the client; otherwise it warns on stderr and
runs the plain CLI. **Never call `vllm bench serve` directly on a pod or node
without chips**: run it through `serving_benchmark.py`, or as
`python3 serving_benchmark.py --as-vllm-cli bench serve <its arguments>`.
That mode is for `vllm bench serve` only: with no TPU platform, vllm
commands that build an engine config, even `--version`, fail with `Failed
to infer device type`.

### Beside the server (`gke_job`)

A `gke_job` server, with or without `--serve`, is reachable in its namespace
at `http://$JOB_ID:<port>` from a `--colocate-with $JOB_ID` client. Measure
the server that was deployed and verified; never tear it down and submit a
fresh one for the benchmark, which pays its startup again and gives up the
node in between. Submit the client as a second job on the server's node:
`--colocate-with` puts it there without chips, and `--colocate-requests` sets
the CPU and memory it asks for instead. Submit it rather than `compute run`
it, so a long run is waited on rather than cut off, and omit `--serve`: the
client ends on its own.

Write `<bench_work>/bench_client.sh`:

```bash
#!/bin/bash
set -eo pipefail
PY="${TPU_AGENT_SUITE_PYTHON:-$(command -v python3)}"
exec "$PY" serving_benchmark.py "$@"
```

```bash
{{runtime}} compute submit --package {{package_tag}} \
  --colocate-with $JOB_ID --colocate-requests cpu=4,memory=16Gi \
  --recipe <bench_work>/bench_client.sh \
  --input {{script:common/serving_benchmark.py}}:serving_benchmark.py \
  --input {{script:common/benchmark_fastapi_server.py}}:benchmark_fastapi_server.py \
  --input {{script:tpu-model-onboarding/fetch_weights.py}}:fetch_weights.py \
  --output serving_bench/summary.json:serving_bench_<seed>/summary.json \
  --output serving_bench/summary.md:serving_bench_<seed>/summary.md \
  --name <model>-bench-<seed> \
  -- --base-url http://$JOB_ID:<port> --engine <engine> \
     --model <model_id> --tokenizer <model_id> \
     --output-dir serving_bench --seed <seed> <the measurement's flags>
BENCH_ID=$({{runtime}} compute job list --package {{package_tag}} --json --no-refresh \
  | python3 -c "import json,sys;print(next(r['job_id'] for r in json.load(sys.stdin) if r['job_name']=='<model>-bench-<seed>'))")
{{runtime}} compute job wait --package {{package_tag}} --job-id $BENCH_ID \
  --until done --json
{{runtime}} compute job artifacts --package {{package_tag}} --job-id $BENCH_ID \
  --out-dir <bench_work>
```

On `job wait` exit 3 the run is still going: read `next` in the result and do
what it says, which is usually to run the same wait again. Save the client's
log with `compute job logs` either way; it carries the table and any error.

If the client's job ends early (a timeout, preemption, or its node lost),
`job artifacts` still brings back `summary.json` as it last was: the client
rewrites it after every cell, and compute sends it back as it changes. Its
`status` is then `partial`; report the cells it lists as measured and the
rest as not run.

### Through a port forward (every backend)

Every backend (`local`, `gce_tpu_vm`, `gke_pod`, `gke_job`) supports a port
forward.

The client runs on this host and needs `vllm` here for `--engine vllm`.
Run it with an interpreter that has vllm, so it runs vllm as itself as "Run
the client" says; `--vllm-cmd` points it at another vllm, which then runs
plain. Do not run it with
`compute run`: on `gke_job` that dispatches a new pod that requests chips,
which waits behind the server holding them, or lands on another node where
`localhost` has no server.

1.  **Forward the server's port.** `--background` returns once the forward is
    serving, and prints the forward's PID:

    ```bash
    {{runtime}} compute port-forward --port <port> --job-id $JOB_ID --background
    ```

2.  **Run the client on this host:**

    ```bash
    python3 {{script:common/serving_benchmark.py}} \
      --base-url http://localhost:<port> --engine <engine> \
      --model <model_id> --tokenizer <model_id> \
      --output-dir <bench_work>/serving_bench <the measurement's flags>
    ```

3.  **Kill the forward's PID** when the client is done, whatever it
    returned: the forward is detached and serves until it is killed.

## A sweep inside a time limit

When the cluster stops jobs a fixed time after they start running
(the compute profile's `declared_duration_minutes`, from `config show`, when set,
or a limit the prompt states), the server and every client run against it must
end before that; a server can spend 15-30 minutes starting up. A server
stopped mid-sweep fails the remaining cells; the client keeps the ones it
finished (above). Size the work to fit:

-   **Before submitting a server**, look in this run's job records, not the
    cluster: `compute job list --run-id <run_id>` lists only the jobs this run
    submitted. Use a `RUNNING` one only if it is the job id or `--name` this
    run recorded for the same deployment, and only if what is left of its
    limit covers the next chunk: the limit less the time since the
    `start_time` that `job status` shows.
-   **One server per load regime**, or per chunk of concurrency points that
    fits in about 75% of the limit with its startup. Never one server for a
    sweep that cannot finish on it.
-   **One client run per chunk**, each with its own `--output-dir` and
    `--seed`, so each chunk's `summary.json` is collected as it finishes.
-   **On resume, skip the chunks** whose `summary.json` already came back
    with every cell measured, and run only the rest.

## A sweep on several servers (`gke_job`)

A sweep that measures several input/output lengths -- 1k/1k, 1k/8k, 8k/1k --
runs them one after another on one server, so it takes as long as all of them
together. {{script:common/serving_sweep.py}} starts more copies of the same
server and measures a different length on each copy at the same time, so the
sweep takes about as long as its slowest length.

Terms used below:

| Term | Meaning |
| :--- | :--- |
| shape | One `isl:osl` pair and all of its concurrency levels. |
| leader | The server you already deployed and verified. The sweep measures on it and never stops it. |
| follower | A copy of the leader that the sweep starts from the same launch recipe, and stops when it is done. |
| slot | Room for one server on the cluster, booked in the session's `parallel.json` (by {{script:common/tpu_slots.py}}) so that two agents never start servers on the same chips. |
| kind | The accelerator, topology, chips per server and image a server runs on. Numbers from servers of different kinds are not comparable. |

Use it on `gke_job` when the request asks for more than one shape, the leader
was submitted with `--serve`, and the session has more than one slot.

Everywhere else -- `local`, `gce_tpu_vm`, `gke_pod`, a `gke_job` server
without `--serve`, or a session with one slot -- run the client once, as
above (beside the server, or through a port forward), with every shape in its
`--isl-osl`. It measures the shapes one after another, with the same flags
and seeds, so the summary is the same measurement, only slower.

How the sweep works:

-   **Each shape runs on one server.** The client may stop raising the
    concurrency of a shape once throughput stops improving, which it decides
    by comparing each level with the levels before it. So all of one shape's
    levels must run in order on the same server. The shapes wait in a queue,
    longest first, and each server takes the next shape when it finishes one.
-   **The result is one measurement.** Every shape gets the same client flags,
    the full `--concurrency` list and the seeds it would get in a single
    run. The sweep then merges the shapes into one `summary.json`, in the
    client's usual layout, with a `shards` list saying which server measured
    which shape. Report it as one measurement, and say how many servers ran
    it.
-   **The leader goes first.** Pass the server you verified as `--leader`. It
    must be a job of this run submitted with `--serve`, because only those
    get the in-cluster address a client reaches; a leader that was not is
    left out, with the reason in `sweep_result.json`'s `ignored_leaders`.
    Followers start only once the first server is serving, so the compile
    cache and weights it left on a shared volume are ready for them. Each
    follower takes a slot, is submitted with `--serve`, and is stopped when
    the sweep ends, however it ends.
-   **Every server is the leader's kind.** A follower only takes a slot of
    the leader's kind, even when another kind -- a v6e cluster next to a v7x
    one -- has free slots, and every follower runs the leader's image. The
    merge refuses shapes measured on more than one kind. So put the leader on
    the kind with the most free slots (`claim --most-available`), where its
    followers will find room. To compare kinds of TPU, run one sweep per kind
    and compare the summaries; never merge them.
-   **Each server loads the weights.** With `hf_cache_pvc`, every server reads
    one shared copy. Without it, each new pod downloads the model again into
    its own `/dev/shm`; stage a `gs://` copy once and serve that instead
    (onboarding's Step 0.2b does this with `stage_checkpoint.py`).

1.  **Book the slots**, once per session, from the compute profiles. Each
    `gke_job` profile gets one slot per server its node pool can still start
    (the free chips `tpu_slots.py` reads itself, divided by the chips one of
    its Jobs takes; don't run `compute inspect` for them), and each slot
    records its kind of TPU. Profiles on the same nodes share those chips:
    each lists them, and a claim stops at what the
    nodes can hold. `--gke-slots` is what a profile gets when its free chips
    cannot be read, or its pool has no nodes yet; give it no more than the
    shapes to measure. `parallel`, the cap on slots held at once, starts at
    the session's capacity and changes only with `--parallel`:

    ```bash
    python3 {{script:common/tpu_slots.py}} discover --run-id <run_id> \
      --runtime "{{runtime}}" --package {{package_tag}} --gke-slots <N>
    ```

    `status: needs_input` names what only the user can answer; ask them. The
    result's `hardware` lists the free slots per kind of TPU, and `sized`
    how each profile was counted.

2.  **Run it detached** into a new directory, one no earlier sweep used: a
    sweep takes hours, longer than any one command may block. Pass the server
    recipe exactly as Step 3 of the deployment submitted it: the same recipe,
    each `--input` as `--server-input`, its image as `--server-image`, its
    `--interpreter` only if it had one, and any other flag of that submit
    (`--shm-size-gb`, say) as `--server-submit-arg=<flag>`. After `--` go the
    client's flags, all of them, as one `serving_benchmark.py` run would take
    them: `--isl-osl` with every shape, `--concurrency`, `--engine`,
    `--model`, `--tokenizer`, and the measurement's flags below (for an
    onboarding sweep `--max-concurrency-limit <requests per replica x DP>`:
    `max_num_seqs` x `data_parallel_size` for vLLM, `max_running_requests`
    x `dp_size` for SGLang, left out when the deployment sets neither
    `max_num_seqs` nor `max_running_requests`; and
    `--stop-below-gain-pct 10 --stop-on-failure`). They are read by the
    client's own parser, so they default as they would there; the sweep sets
    `--base-url` and `--output-dir`, and each shape's `--isl-osl` and
    `--seed`.
    `--max-model-len` lets it skip a shape that cannot fit before giving it a
    server. On a cluster that stops jobs after a fixed time (the section
    above), add `--server-budget-s` at about 75% of the limit; each leader's
    age comes from its job record:

    ```bash
    W=<bench_work>/sweep && mkdir -p "$W" && nohup python3 \
      {{script:common/serving_sweep.py}} run \
      --runtime "{{runtime}}" --package {{package_tag}} --run-id <run_id> \
      --name <deployment name> --port <port> --leader $JOB_ID \
      --recipe <the launch recipe, e.g. launch_vllm.sh> \
      --server-input <each --input of the server's submit> \
      [--server-image <image>] [--server-submit-arg=<flag>] \
      --max-model-len <max_model_len> --out-dir "$W" \
      -- --isl-osl <isl:osl,...> --concurrency <c,...> --engine <engine> \
      --model <model_id> --tokenizer <model_id> <the measurement's flags> \
      > "$W/sweep.out" 2>&1 < /dev/null &
    ```

    It writes the shapes it queued, longest first, to `$W/sweep_plan.json`
    before it starts a server; `serving_sweep.py plan --out <file>` with the
    same `--max-model-len` and client flags shows them without running
    anything. A server an earlier onboarding in this run left serving, with
    the same deployment, can take shapes too: add it as
    `--leader <job_id>@<profile>`.

3.  **Poll**, at most nine minutes per call, until `sweep_result.json`
    exists; it is written whole, so once it exists it is complete.
    `sweep_status.json` says which server runs which shape meanwhile:

    ```bash
    for i in $(seq 54); do [ -f "$W/sweep_result.json" ] && break; sleep 10; done
    cat "$W/sweep_status.json"
    ```

    If no `serving_sweep.py run` process is left and there is still no
    result, it died before it could write one: report the last lines of
    `$W/sweep.out`, and cancel the servers it started and free their slots,
    which nothing else will:

    ```bash
    python3 {{script:common/serving_sweep.py}} cleanup --runtime "{{runtime}}" \
      --package {{package_tag}} --run-id <run_id> --out-dir "$W"
    ```

4.  **Read** `$W/serving_bench/summary.json` as "Report" below says, beside
    `$W/sweep_result.json`, whose `warnings` carry each shape's. A `server`
    that `failed` or was `lost` gave its shapes to the others; a shape no
    server was left to run is `not_run`, and the sweep is then `failed`, with
    each such cell's reason in `error`. A shape whose server stopped serving
    went back in the queue, once, for another server; one whose client could
    not start (a missing tokenizer, say) failed without one, since it would
    fail again.

| Exit | Meaning |
| :--- | :--- |
| `0` | Every shape measured, by the client's rule. |
| `1` | A shape failed or was not run, on servers that did serve; see `error` and each shape's state. |
| `2` | No server ever served the sweep, or it never started: `status` says why (`invalid_input`, `refused` when no slot was free, `needs_input` when the session has no slots). |

## The measurement's flags

| Use | Flags |
| :--- | :--- |
| One measurement of a deployment (tpu-model-onboarding) | `--isl-osl 1024:128 --concurrency 1 --replicates 1 --num-prompts 200`, or the lengths and concurrency the request names. The lengths must fit the server's maximum model length together. |
| Concurrency sweep of a deployment (tpu-model-onboarding) | `--max-concurrency-limit <requests per replica * DP> --stop-below-gain-pct 10 --stop-on-failure` on top of the `--isl-osl` and `--concurrency` sweep. Requests per replica x DP is `max_num_seqs` x `data_parallel_size` for vLLM and `max_running_requests` x `dp_size` for SGLang (DP 1 when absent). Leave out `--max-concurrency-limit` when the deployment sets neither, so the sweep runs to its top point against the engine's default (points above it measure queueing). |

### Sweep design: static filtering and dynamic early-stopping

`serving_benchmark.py` represents a sweep as a list of `(isl, osl, c_range)` items, where `c_range` is sorted in ascending order:

1. **Static filtering (`filter_c_range`)**: Before running an `(isl, osl)` shape, `c_range` is filtered against:
   - `max_model_len`: skips the shape if `isl + osl` (including `--range-ratio`) exceeds the server's `max_model_len`.
   - `--max-concurrency-limit <int>`: skips concurrency levels `c > max_concurrency_limit` (e.g., `max_num_seqs * dp`, where additional requests only sit in the client HTTP queue).
   - `--avoid <isl:osl:c,...>`: skips points matching comma-separated wildcard rules (`*` matches any value, e.g. `8192:*:64`).
2. **Sequential execution & dynamic early-stopping (`check_early_stop`)**: The kept `c_range` runs sequentially from lowest to highest concurrency. After each concurrency point:
   - If `--stop-on-failure` is set and the cell fails while the server is still alive, or if `--stop-below-gain-pct <float>` (e.g. `10.0`) is set and `output_tokens_per_s` improves by less than that percentage over the best throughput so far for two consecutive steps, all remaining higher concurrencies in `c_range` for that `(isl, osl)` shape are marked `status: "skipped"` with the reason in `error`, and the sweep moves to the next `(isl, osl)` shape.
   - `--stop-below-gain-pct`, `--stop-on-failure`, and `--avoid` are for exploratory and onboarding sweeps only (`settings.sweep_mode: "exploratory"`), never for a study or A/B comparison.

### SGLang client, Pareto curves, and BKC export

- **`--bench-client sglang`**: Runs `python3 -m sglang.bench_serving` (override with `--sglang-cmd`) against an OpenAI-compatible server (`--endpoint completions` $\rightarrow$ `sglang-oai`, `--endpoint chat` $\rightarrow$ `sglang-oai-chat`), flushing the radix prefix cache via `POST /flush_cache` when `/reset_prefix_cache` returns 404/405 and normalizing SGLang's JSONL output to the same summary keys.
- **Pareto curve & BKC verification**: `summary.json` includes `curves["<isl>:<osl>"]["pareto_frontier"]` (non-dominated points over `output_tokens_per_s` and `output_tokens_per_s_per_user`). Pass `--reference-curve <ref_summary.json> [--tolerance-pct 5.0] [--export-bkc <bkc.json>]` to verify Pareto reproduction within `--tolerance-pct` and export the Best Known Config (`bkc.json`) manifest when consistent.

## Exit codes and failures

Neither engine's client can hang on a server that stops answering: a run that
takes longer than `--run-timeout` (default 1800 s) is killed and its cell
failed with the reason. After any failure the client probes `/health`; if the
server no longer answers, the rest of the sweep is not run (each such cell
says why), the summary is still written, and the client exits `2`.

| Exit | Meaning |
| :--- | :--- |
| `0` | Every cell measured. |
| `1` | A cell failed or was skipped; see its `error`. A cell where no request completed, or where more than `--max-failed-pct` (default 0) failed, is `failed`. A cell whose lengths cannot fit in the server's `max_model_len` is `skipped` and not run. |
| `2` | The server was unreachable, listed no model, loaded a path this host lacks with no `--tokenizer` given, or stopped answering `/health` mid-sweep (`summary.json` names the cell, in `error`). |

Read `status` in `summary.json`, never trust a run's own exit code: `vllm
bench serve` exits 0 even when every request fails. `partial` means the sweep
did not finish: the summary is the one saved after its last finished cell.

## Report

Read `summary.json`, never the console. Each cell's `metrics` carries these
names, with `mean`, `stdev`, `cv_pct` and the per-replicate `values`; the
value to quote is `cells[i].metrics[<name>].mean`:

| Metric | Meaning | `vllm` | `custom_fastapi` |
| :--- | :--- | :--- | :--- |
| `ttft_{mean,p50,p90,p99}_ms` | Time to first token | yes | not measured: no streaming |
| `tpot_{mean,p50,p90,p99}_ms` | Time per output token after the first, per request | yes | not measured: no streaming |
| `itl_{mean,p50,p90,p99}_ms` | Inter-token latency, across every token gap | yes | not measured: no streaming |
| `e2el_{mean,p50,p90,p99}_ms` | End-to-end request latency | yes | yes |
| `output_tokens_per_s` | Aggregate output throughput | yes | with `--tokenizer` |
| `total_tokens_per_s` | Aggregate input + output throughput | yes | with `--tokenizer` |
| `output_tokens_per_s_per_user` | `1000 / tpot_mean_ms`: decode speed one user sees | yes | not measured: no TPOT |
| `requests_per_s` | Completed requests per second | yes | yes |
| `completed`, `failed` | Request counts | yes | yes |

Report one table with these rows for the engine that ran, and write "not
measured" with the reason in a row it cannot fill; never leave it blank or
fill it in. Beside it give the settings from `summary.json` (`settings`, and
each cell's lengths and concurrency), the backend, and how the client ran
(beside the server, or through a forward). A cell with `failed` above zero is
reported with that count, not as a clean run. Cached TTFT is not part of
this measurement; report it only if the request asks for it, as its own
labelled run.
