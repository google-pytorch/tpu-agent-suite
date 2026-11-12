---
scripts:
- scripts/common/serve_vllm.sh
- scripts/common/serving_sweep.py
- scripts/common/tpu_slots.py
- scripts/common/vllm_patches/apply_vllm_patch.sh
- scripts/optimization-heuristics/report.py
- scripts/tpu-model-onboarding/health_probe.py
- scripts/tpu-model-onboarding/verify_inference.py
references:
- references/common/inference-serving-benchmark.md
- references/tpu-serving-bench/accuracy.md
---

# Serving bench

Engines: vllm.

Start, measure and stop inference servers of one model, each arm a pinned image plus an optional
`plugin`: a ref in the campaign repo, as tpu-code-stack cut it, whose `plugin.patch`, the
`git diff base..sha` already cut, is the one patch its servers apply; never read the repo. An arm
with `patches` (registry files, for a caller with no repo) ships them in order the same way, one
`--patch` per file.
`<out>` is `{{runs}}/<run_id>/artifacts/tpu-serving-bench/<label>/`, `<arm>` an arm's label.

## Checks before any job

Each failure returns before any job.

- **The engine has a section below.** A request whose `engine` has none returns `invalid_input`:
  `engine <x> is not supported here`. A workload that runs another engine than the one named
  returns `invalid_input` too: `engine <x> is not <engine>; not handled here`. A FastAPI server
  (`benchmark_fastapi_server.py`, `--engine custom_fastapi` in the client args) is
  never one of these engines.
- **The workload is one the sweep runs.** A `--workload` other than `random` in the client args,
  or a benchmark the caller supplies (its own client or script), returns `invalid_input`:
  `workload <x> is not handled here`.
- **The action's inputs** are there, as the agent's Inputs section lists; a bench has exactly one
  baseline arm. Every path named exists, each `plugin.patch` and `patches` file included. Each
  gap is a `missing` entry.
- **`gke_job` profiles only.** Each server is a fresh container from the arm's image, so one arm's
  patches never reach another's. Any other backend returns `blocked`:

  ```bash
  {{runtime}} compute --package {{package_tag}} preflight --json \
    | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["backend"], r["status"])'
  ```

- **Every job goes through `{{runtime}} compute`**, under a slot.

## The workload

From `bench_request` (a path, or the object): `client_args` are the client's flags; `server`
gives `launch_recipe` and `deployments_json`; `sweep` gives `server_inputs` and `max_model_len`.
Otherwise `deployment` and `client_args` as given. No launch recipe either way is `invalid_input`.
Client args without `--model` or `--tokenizer` get `--model <model> --tokenizer <model>`: the server
reports its local weights path, which the client pod does not have.
With `shapes`, keep only those `isl:osl` pairs in `--isl-osl`; a listed shape the workload lacks is
`invalid_input`. The arm's `image` and patches replace any image or patches the record names.

`hardware` is the request's. Without one, run the first arm's sweep without `--hardware`: it writes
the session's `parallel.json` if there is none and keeps every server on one kind, which its merged
summary names in `shards[].hardware`. Every later arm passes that key, and the result returns it.
Claim nothing to find the key. When `parallel.json` already lists more than one hardware key
(`python3 {{script:common/tpu_slots.py}} show --run-id <run_id>`), return `invalid_input` naming
them, since the caller must pick one. The session's slots are the ones its compute profiles declare, never measured: the
first sweep writes `parallel.json` from them when there is none. A Job waiting in its Kueue queue is waiting, not `blocked`; no slot of the
key in any profile is `blocked`.

## vllm

### The launch recipe

It must start the server with `exec bash serve_vllm.sh <vllm serve args>`, so the profiler is armed
and the arm's patch applies before vLLM starts. If it calls `vllm serve` directly, write
`<out>/launch.sh` from it with that line as `exec bash serve_vllm.sh <args>`, and drop any line
that runs `apply_vllm_patch.sh`: the patch comes only from the arm. Nothing in it edits source. Ship
{{script:common/serve_vllm.sh}} as `serve_vllm.sh` beside the recipe's own inputs.

An arm with `server_args` gets its own copy, `<out>/<arm>/launch.sh`: the recipe with those flags
appended to its `exec bash serve_vllm.sh` line, and nothing else changed. Write
`<out>/<arm>/launch.diff` against the shared recipe, and give that arm's sweep the copy as
`--recipe`. Every other arm runs the shared recipe as it is.

### bench

Follow {{reference:common/inference-serving-benchmark.md}} "A sweep on several servers" for each
arm; the sweep books every slot and starts every server, and you claim nothing yourself.

Arms run side by side only when the request names `hardware` and that key has a free slot for
every arm's servers (`tpu_slots.py show`). Otherwise run them one at a time, each through steps 2
to 7 (sweep, wait, check, capture, release) before the next starts: without `hardware`, the first
arm's sweep finds the key every later arm passes; with too few slots, and with `profile` on, an
arm's kept servers hold its slot until its own step 7.

1. **Plan once**, unless the request has `plan`, so every arm runs the same cells and seeds:

   ```bash
   python3 {{script:common/serving_sweep.py}} plan --out <out>/sweep_plan.json \
     [--max-model-len <max_model_len>] -- <client_args>
   ```

2. **One sweep per arm**, started together or one after another as above, each detached into its
   own new out-dir:

   ```bash
   W=<out>/<arm>/sweep && mkdir -p "$W" && nohup python3 \
     {{script:common/serving_sweep.py}} run \
     --runtime "{{runtime}}" --package {{package_tag}} --run-id <run_id> \
     --name <label>-<arm> --port <port> --recipe <launch recipe> \
     --server-input {{script:common/serve_vllm.sh}}:serve_vllm.sh \
     --server-input <each other input> [--interpreter <interpreter>] \
     [--server-submit-arg=<flag> ...] --server-image <arm image> \
     [--patch <patch> ...] --hardware <hardware> [--keep-servers] \
     [--server-budget-s <s>] \
     --plan <plan> --out-dir "$W" > "$W/sweep.out" 2>&1 < /dev/null & echo $! > "$W/pid"
   ```

   `--patch` for `plugin.patch`, or each `patches` file in order. `--keep-servers` when `profile`
   is true. The sweep starts every server it measures on, on a slot it claims, and stops them and
   frees their slots when it ends; it never measures on a server another request left running.
   A sweep that ends `refused` is rerun into a new out-dir once another arm has finished step 7;
   refused with no other arm holding a slot is `blocked`.
3. **Wait** on every running arm's sweep at once; it blocks up to nine minutes, so give the
   command a ten-minute timeout, and prints one line per arm, so poll nothing else:

   ```bash
   python3 {{script:common/serving_sweep.py}} wait --out-dir <out>/<baseline arm>/sweep \
     --out-dir <out>/<arm>/sweep ...
   ```

   Exit 3 means wait again. A run whose status is `died` left no result: clean it up as step 7
   says, report the last lines of its `sweep.out`, and rerun that arm once into a new out-dir.
4. **Patch outcome.** A server whose `error` in `sweep_status.json` reads `patch does not apply:
   <name>` stopped the arm: `patch: {exit: 4, outcome: "base_mismatch: <name>"}`, no cells, no
   rerun. Exit 4 means the patch does not fit this image (for `plugin`, `base_sha` is not its plugin
   commit): an input error, never a slower arm. On the baseline arm the request is `failed`.
   Otherwise `exit: 0`, outcome `applied`, or `none` with nothing to apply.
5. **Check** each merged `serving_bench/summary.json` is `ok` and every `shards[].hardware` has the
   key's accelerator, topology and chips. Any other, or `failed` with no patch failure: rerun that
   arm once into a new out-dir. A baseline arm failing again is `failed`; another arm keeps its
   `status` and the reason in `summary`.
6. **Capture** each arm with `profile` on, after its sweep: send {{agent:tpu-diagnostics}} one
   `profiling-request` with `server: {job_id, port, files_port: 8790, load}` for the first server
   `sweep_status.json` marks `kept`, `load` the arm's heaviest cell as client flags, and `image` the
   arm's image. Check it verified `/device:TPU:` planes, and record its traces, with each trace's
   `ml_diagnostics_url`, when published, in the arm's `ml_diagnostics_urls`. The capture is
   never a measurement.
7. **Release.** For every arm, whatever happened:

   ```bash
   python3 {{script:common/serving_sweep.py}} cleanup --runtime "{{runtime}}" \
     --package {{package_tag}} --run-id <run_id> --out-dir <out>/<arm>/sweep
   ```

8. **Compare.** Per arm, one `cells` row per summary cell: `output_tokens_per_s`, `ttft_p50_ms`,
   `tpot_p50_ms`, `e2el_p50_ms`, each its `metrics[<name>].mean`. The deltas come from the script,
   never by hand:

   ```bash
   python3 {{script:optimization-heuristics/report.py}} \
     --baseline <out>/<baseline arm>/sweep/serving_bench/summary.json \
     --measured <arm>=<out>/<arm>/sweep/serving_bench/summary.json ... --out <out>/report.json
   ```

   Each arm's `deltas` are its `points` in `report.json`: per cell, each metric's `delta_pct` and
   the throughput `verdict` (`win` or `loss` beyond the noise, else `tie`), each `cell` named as in
   `cells` (`1024:1024:32` is `isl1024_osl1024_c32`).
9. **Write** `<out>/<arm>/bench_request.json` per arm ({{schema:bench-request.json}}: route
   `sweep`, the client args, the first server, `patches` as `[plugin.patch]`, the arm's `patches`,
   or none, `hardware`, `sweep`, `results_json`), `<out>/REPORT.md` (one table per shape: cells x
   arms, deltas, servers per arm, each arm's plugin ref and sha or patches, its `server_args` and its patch outcome,
   hardware), and `<out>/serving_bench_request.json`: the request with `hardware` and `plan`
   filled, each arm's `plugin` or `patches` as sent. It must validate as a request: sent back, it
   reruns the identical bench.

### serve

Per arm, one server left running:

1. **Slot**, holder `tpu-serving-bench:<label>:<arm>`:

   ```bash
   python3 {{script:common/tpu_slots.py}} claim --run-id <run_id> \
     --holder tpu-serving-bench:<label>:<arm> --hardware <hardware> \
     --runtime "{{runtime}}" --package {{package_tag}}
   ```

2. **Submit** with the profile the claim printed. With `plugin` or `patches`, write
   `<out>/<arm>/vllm_patches.txt`, one patch file name per line in order, and ship it, each patch
   and the applier:

   ```bash
   {{runtime}} compute submit --package {{package_tag}} --profile <profile> --run-id <run_id> \
     --recipe <launch recipe> [--interpreter <interpreter>] --serve --image <arm image> \
     --input {{script:common/serve_vllm.sh}}:serve_vllm.sh --input <each other input> \
     [--input <patch>:<name> ... \
      --input {{script:common/vllm_patches/apply_vllm_patch.sh}}:apply_vllm_patch.sh \
      --input <out>/<arm>/vllm_patches.txt:vllm_patches.txt] [<submit flag> ...] \
     --name <label>-<arm>
   python3 {{script:common/tpu_slots.py}} set-job --run-id <run_id> --resource <resource> \
     --holder tpu-serving-bench:<label>:<arm> --job-id <job_id>
   {{runtime}} compute job wait --package {{package_tag}} --job-id <job_id> \
     --until "log:Application startup complete" \
     --fail-on-log "serve_vllm: patch (does not apply|failed)|Traceback \(most recent call last\)"
   ```

   Exit 3 means wait again. `patch does not apply` is `failed` with `base mismatch: <patch>`, and
   `patch failed` is `failed` with the log line, both with no relaunch. Any other failure: save
   `compute job logs` to `<out>/<arm>/server.log`, cancel, and relaunch at most twice; past that
   `failed`, with that file as `logs`.
3. **Verify** through a port forward, then kill the forward's PID:

   ```bash
   {{runtime}} compute port-forward --package {{package_tag}} --port <port> --job-id <job_id> \
     --background
   python3 {{script:tpu-model-onboarding/health_probe.py}} --port <port> --timeout 300 \
     --output <out>/<arm>/health.json
   python3 {{script:tpu-model-onboarding/verify_inference.py}} --port <port> --engine vllm \
     --model-id <model> --output <out>/<arm>/verification.json
   ```

   `verified: false` cancels the server and releases its slot: `failed` with the report's
   `failures`. A known-answer warning goes in `summary`, not a failure.
4. **Return** a handle per arm: `job_id`, `profile`, `port`, `files_port` 8790, `engine`, `image`,
   `plugin` or `patches`, `deployments_json`, `launch_recipe`, `hardware`, `resource`, `holder`,
   `verification`. The server and its slot stay held until a `stop`.

### eval

Follow {{reference:tpu-serving-bench/accuracy.md}}: one eval job per arm under
`vllm_server_cmd`, on this section's slots and patches, and the 1.5pp gate. It takes no
workload: of "The workload", only `hardware` applies.

### stop

Per server: with `sweep_dir`, run the sweep's `cleanup` there (bench step 7). Otherwise:

```bash
{{runtime}} compute job cancel --package {{package_tag}} --job-id <job_id>
python3 {{script:common/tpu_slots.py}} release --run-id <run_id> --resource <resource> \
  --holder <holder>
```

A handle with neither `sweep_dir` nor `resource` was not started here: `invalid_input`. Return the
job ids cancelled.
