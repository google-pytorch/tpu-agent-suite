---
scripts:
- scripts/accuracy-check/eval_recipe.sh
- scripts/common/tpu_slots.py
- scripts/common/vllm_patches/apply_vllm_patch.sh
---

# Accuracy eval (`action: eval`)

Per arm, one job boots a vLLM server under `vllm_server_cmd` and runs `lm_eval` against it; the
stack arm's score is compared with the baseline recorded for that config, or with the baseline
arm's. The output is a gate a caller can act on, not a number.

This is the **expensive** tier. A real gate run is hours.

**Why one job per arm, not `serve` plus a client.** {{script:accuracy-check/eval_recipe.sh}} runs
server and `lm_eval` in one job: the harness version comes from the image, the endpoint is
`localhost`, and the job's exit code is `lm_eval`'s. That is how both baselines were measured. A
client against a `serve`d server would run `lm_eval` on your host through a port forward for
hours, under a harness version the baselines never saw. The slot handling and the patch handling
are this agent's own (agent.md "serve" steps 1 and 2).

## One size: `--limit 100`

`mmlu_pro` runs at `--limit 100`: 100 questions per subject, 14 subjects, **~1400 samples**. That
is the gate. The full set (`limit: full`, 12,032 samples) stays available for an absolute quality
number; it is never a precondition for a pass.

`--limit` is **per subject**, which is why 100 gives ~1400 and `--limit 5` gives 70. Anything
below 100 is a plumbing test: `gate: inconclusive`, never compared as a result.

**Baseline and stack must be at the same size.** A `--limit 100` score checked against a
full-set baseline is not a paired comparison.

## The gate is 1.5pp, and it is not configurable

**A drop of more than 1.5 percentage points against the `mmlu_pro` baseline is `gate: fail`.**
Fixed for every model and config, and calibrated on `mmlu_pro` at `--limit 100`; not a generic
tolerance to carry to another task.

## Earlier job records

The baseline and evidence job ids here (`j-…`) are records of runs on an earlier job system,
for provenance only. `{{runtime}} compute job` cannot read them back.

## Step 0: Get the baseline

A baseline, the protocol that produced it, and the job that measured it belong together. Take
them, in this order, from:

1. the request's `baseline`;
2. the accuracy record of an earlier eval on the same config at the same size, under
   `{{runs}}/<earlier run_id>/artifacts/tpu-serving-bench/*/accuracy-<config>.md` (Step 5 writes
   it);
3. the tables in Step 1 and "Server command and baseline, per model".

You need score, stderr, `n`, size, the server shape it was measured at and its `EVAL_CMD`, **for
`mmlu_pro`**. Discard baselines on any other task. With one, the baseline arm is not run: its
result arm carries the recorded numbers, `source` the record, `job_ids` empty.

With none, the baseline arm runs live first, on the unpatched image under the same protocol. Its
score, through Step 5's record, becomes this config's baseline for every later eval, and the stack
is gated against it as usual. A stack is never its own reference. Say in the report that the
baseline is n=1 and unreplicated. A model Step 1 has no protocol for uses the
Qwen3.5 1-node command verbatim, with `num_concurrent` the server's `--max-num-seqs` times its
data-parallel size; record it as the protocol.

### The task is `mmlu_pro`. You will be offered others; do not take them.

**Qwen3.5-397B-A17B-FP8** has two accuracy baselines:

| | `mmlu_pro` (**use this**) | `gsm8k` (do not use) |
|---|---|---|
| Recipe | `vllm-torchtpu-generic-eval` | `vllm-torchtpu-generic-**bench**`, `BENCH_CMD` swapped |
| Task source | `--tasks mmlu_pro` | `<InferenceX>/utils/evals/gsm8k.yaml`, cloned at runtime |
| Shape | limit-100, 1400 samples, thinking off | full set, 5-shot, `temperature=0` |
| Score | 0.8250 ± 0.0100 · `j-8846a1e0-1b60-4785-944c` | 0.9689 ± 0.0048 · `j-52e7603c-99b7-48a8-b1eb` |

A gsm8k check is a serving job with the gsm8k client in place of the benchmark client, not this
eval: `invalid_input` rather than approximating it.

## Inputs

- **`arms`**: the stack, plus the baseline (`baseline: true`) when the config has no recorded
  baseline (Step 0), which then runs live. Each image is pinned by digest; nothing is built.
- **`config`**, e.g. `qwen3.5-397b-fp8-tp1-dp8-ep`. It selects the protocol and the baseline, and
  must match the layout served. Not the model id: one model has several configs.
- **`vllm_server_cmd`**: the server under test, the **same** command the gated benchmark used, read
  back from that job's record. Both arms are served under it.
- **`model`**: the HuggingFace id, org included, passed as `SERVED_MODEL_NAME`. A gated model needs
  the target's HuggingFace token; check what the target offers with
  `{{runtime}} compute --package {{package_tag}} inspect --json`.
- **`limit`**: `EVAL_LIMIT`: `100` is the gate, `full` the whole set.

## Step 1: The protocol is fixed. Copy it, do not rebuild it

**There is one protocol per model: whichever one produced the baseline.** Read it out of the
baseline record and copy it verbatim. Do not merge the two below or assume either generalizes.
The recipe has no default `EVAL_CMD` and exits non-zero without one.

| | **Qwen3.5-397B-A17B-FP8** | **GLM-5.2-FP8** |
|---|---|---|
| Task | `mmlu_pro` | `mmlu_pro` |
| Baseline recipe | `vllm-torchtpu-generic-eval` | `vllm-torchtpu-generic-eval-2node` |
| Slice | 1 node, TP=1 / DP=8 / EP | 2 nodes, DP=16 / EP over Ray |
| `--model_args` form | JSON | comma-kv |
| `num_concurrent` | 512 | 32 |
| Thinking disabled | yes, in two places | no |
| `--log_samples` | yes | no; **add it to a candidate** |
| `until` | task default | `["<\|im_end\|>"]` |
| Wall clock, `--limit 100` | 1h29m | 3h53m |
| Baseline | 0.8250 ± 0.0100 · `j-8846a1e0-1b60-4785-944c` | 0.7500 ± 0.0113 · `j-012c51a9-a247-4d76-914b` |

Both baselines: commit `ec4bcecf78`, lm_eval 0.5.0.dev1, n=1400, **0 null-content answers**.

### Qwen3.5-397B-A17B-FP8, 1 node

```
lm_eval --model local-chat-completions \
  --model_args "{\"model\": \"${SERVED_MODEL_NAME}\", \"base_url\": \"${EVAL_BASE_URL}\", \"enable_thinking\": false, \"num_concurrent\": 512}" \
  --tasks ${EVAL_TASKS} --apply_chat_template --log_samples \
  --gen_kwargs "{\"chat_template_kwargs\": {\"enable_thinking\": false}}" \
  --output_path ${EVAL_OUTPUT_DIR} ${EVAL_LIMIT_ARG} --seed 42
```

Only `${EVAL_TASKS}` and `${EVAL_LIMIT_ARG}` vary. Every other part is load-bearing:

- **`enable_thinking: false` appears twice, deliberately**: in `model_args` and in `gen_kwargs`.
- **`--log_samples` is not optional.** When a score comes back wrong, the samples tell you whether
  the model emits `<think>`, hits the token cap, or answers fine and is mis-extracted.
- **JSON `--model_args` requires lm_eval 0.5.x.** 0.4.x splits it on `,` and `=` into garbage
  keys. `vllm/vllm-tpu:nightly` ships 0.4.12; the torchtpu image and the recipe's pin give
  0.5.0.dev1.

### GLM-5.2-FP8, 2 nodes

```
lm_eval --model local-chat-completions --model_args base_url=${EVAL_BASE_URL},model=${SERVED_MODEL_NAME},tokenizer=${SERVED_MODEL_NAME},max_gen_toks=4096,max_length=32768,num_concurrent=32,timeout=7200 --apply_chat_template --tasks ${EVAL_TASKS} --seed 42 ${EVAL_LIMIT_ARG} --gen_kwargs "max_new_tokens=4096,until=[\"<|im_end|>\"]" --output_path ${EVAL_OUTPUT_DIR}
```

**The baseline is sound.** From the eval log of `j-012c51a9-a247-4d76-914b`:

```
lm_eval version: 0.5.0.dev1
smoke: finish_reason=stop completion_tokens=130
smoke: content=391
eval-summary: null-content answers=0
eval-summary: mmlu_pro exact_match,custom-extract=0.75 ... sample_len=1400
```

Two properties travel with the command, neither impugning the score:

1. **`until=["<|im_end|>"]` replaces `mmlu_pro`'s `until: ['Question:']`** with a token GLM does
   not emit, so every sample runs to the 4096-token cap. That costs wall clock (3h53m), not
   accuracy.
2. **No `--log_samples`.** Add it to a candidate: it changes what is written, never what is
   generated. Without it the recipe reports the null-content count as `unknown`.

Thinking is not disabled here, against `--reasoning-parser=glm45`. With zero nulls there is no
evidence it cost anything, and none that thinking was off. Say which you know.

### Rules for both

- **Reuse the baseline's command; do not "improve" on it.** A stack measured under another command
  is not comparable to its baseline.
- **Both JSON arguments are wrapped in `"…"` with inner quotes escaped** (see Quoting).
- **Record the harness version; do not assume a mapping.** Read the lm_eval version and the
  `mmlu_pro` group and subtask versions from the results table (and the `versions` map in the
  results JSON) into the record. When stack and baseline differ on either, say so and do not
  treat the delta as clean.

### Server command and baseline, per model

The **authority for a server command is the record of a job that ran it** (the
`effective VLLM_SERVER_CMD:` line the recipe prints), not a copy here. No numbers are kept in this
table; it says where to look.

| Model / config | Baseline recipe | Baseline job (`--limit 100`) |
|---|---|---|
| `qwen3.5-397b-fp8-tp1-dp8-ep`, agent shape | `vllm-torchtpu-generic-eval` | `j-8846a1e0-1b60-4785-944c` |
| `glm-5.2-fp8-dp16-ep` | `vllm-torchtpu-generic-eval-2node` | `j-012c51a9-a247-4d76-914b` |

- **To reproduce a baseline exactly, pin its image.** A run on another image reproduces nothing.
- **Job ids are full ids on purpose**; the short form resolves nowhere. Config names use
  `<model>-<quant>-<sharding>` so one name selects one baseline.
- **Only generic-recipe jobs count**: `vllm-torchtpu-generic-eval` or `-2node`, or an eval of
  this recipe with a recorded protocol. Per-model recipes, numbers under
  `<vllm-torchtpu>/scripts/vllm/benchmarking/baselines/eval/`, and protocols derived from
  `<vllm-torchtpu>/configs/<config>.sh` or `run_eval_flow.sh` are **not** comparators.

## Step 2: Run each arm's eval job

Measure the arms one after the other, the baseline first when Step 0 needs it, and release one
arm's slot before claiming the next.

1. **Slot**, as agent.md "serve" step 1: holder `tpu-serving-bench:<label>:<arm>`, `--hardware`
   the request's key, or the one agent.md "The workload" picks. Exit 3 is `blocked`.
2. **Inputs**, in `<out>/<arm>/`:
   - `eval_cmd.txt`: the Step 1 command for this model.
   - `vllm_server_cmd.txt`: `vllm_server_cmd`. With `plugin` or `patches`, prefix
     `bash apply_vllm_patch.sh <name> && ` once per patch, in order (`plugin.patch`, or each
     `patches` file); an arm with neither runs the command as is.
3. **Confirm the image exists** before submitting: a digest that does not resolve reserves the
   slice and then fails to pull.
4. **Submit** with the claim's profile, then `set-job` as agent.md "serve" step 2:

   ```bash
   {{runtime}} compute submit --package {{package_tag}} --profile <profile> --run-id <run_id> \
     --recipe {{script:accuracy-check/eval_recipe.sh}} --interpreter bash --image <arm image> \
     --input <out>/<arm>/vllm_server_cmd.txt:vllm_server_cmd.txt \
     --input <out>/<arm>/eval_cmd.txt:eval_cmd.txt \
     [--input <patch>:<name> ... \
      --input {{script:common/vllm_patches/apply_vllm_patch.sh}}:apply_vllm_patch.sh] \
     --output eval_out.tar.gz --output server.log --name <label>-<arm> \
     -- SERVED_MODEL_NAME=<model> EVAL_TASKS=mmlu_pro EVAL_LIMIT=<limit>
   ```

The recipe accepts only `SERVED_MODEL_NAME`, `EVAL_TASKS`, `EVAL_LIMIT`, `SERVER_PORT`,
`SERVER_WAIT_TIMEOUT_S` and `LM_EVAL_VERSION`, and exits non-zero on any other key.

- **`SERVER_PORT`** (default 8000) builds `${EVAL_BASE_URL}`, so `--port` in the server command
  must match it.
- **Server environment goes in the server command**: `export A=1 B=2 && vllm serve …`. That is how
  `VLLM_ENGINE_READY_TIMEOUT_S=9600` is set; raise `SERVER_WAIT_TIMEOUT_S` (default 3600) to
  match, or the recipe stops waiting first.
- **`EVAL_CMD` is run verbatim**, nothing appended. It must mention `base_url`, or the recipe
  exits before starting the server. Use `${EVAL_BASE_URL}` (the full `/v1/chat/completions` URL),
  `${EVAL_OUTPUT_DIR}`, `${EVAL_TASKS}`, `${EVAL_LIMIT_ARG}` and `${SERVED_MODEL_NAME}`.

**The recipe is single-host.** The GLM-5.2 protocol needs two hosts and a recipe that creates the
Ray workload; this agent ships none. If `compute inspect --json` shows the target cannot hold the
baseline's layout, release the slot and return `blocked` with that reason; never run another
layout and compare it. A 2-node GLM-5.2 run must carry its baseline's values:
`SERVED_MODEL_NAME=zai-org/GLM-5.2-FP8`, `EVAL_LIMIT=100`, `VLLM_ENGINE_READY_TIMEOUT_S=9600` with
a server wait of the same length, the server env `VLLM_DISABLE_COMPILE_CACHE=1
VLLM_WORKER_MULTIPROC_METHOD=spawn TORCHINDUCTOR_AUTOGRAD_CACHE=0`, and the GLM `EVAL_CMD`.

### Quoting

The eval command is `eval`'d in the container:

- **JSON needs its double quotes escaped as `\"`.** Unescaped, `{"enable_thinking":false}`
  arrives as `{enable_thinking:false}` and dies in argument parsing, which is how `j-09ab11f8`
  died. JSON wrapped in outer `"…"` may contain spaces; unwrapped JSON must not.
- **A list-valued `gen_kwargs` keeps the baseline's spelling**:
  `--gen_kwargs "max_new_tokens=4096,until=[\"<|im_end|>\"]"`. `lm_eval` parses it with
  `literal_eval`; the `|` characters are safe inside the quotes.

## Step 3: Watch the job

```bash
{{runtime}} compute job wait --package {{package_tag}} --job-id <job_id> \
  --until "log:vllm-server is up" \
  --fail-on-log 'vllm patch does not apply|^eval exited rc=[1-9]' --json
{{runtime}} compute job wait --package {{package_tag}} --job-id <job_id> --until done --json
{{runtime}} compute job logs --package {{package_tag}} --job-id <job_id>
```

Exit 3 from `job wait` means still waiting: run it again. When the server exits before `/health`,
the recipe prints the last 200 lines of `server.log` into the job log.

**`vllm patch does not apply: <name>`** is the applier's exit 4: the patch does not fit this image
(for `plugin`, `base_sha` is not its plugin commit). The arm is `patch: {exit: 4, outcome:
"base_mismatch: <name>"}` with no score and no rerun, and the request is `failed` with
`failure_reason: base_mismatch: <name>`. Otherwise `exit: 0`, `applied` or `none`.

A server that fails in a way its log does not explain and a profile would goes to
{{agent:tpu-diagnostics}} as a `profiling-request`. A job that exits 0 is necessary, not
sufficient. Check the job log, in order, and read back `effective EVAL_LIMIT:`, `effective
EVAL_CMD:` and `effective VLLM_SERVER_CMD:`; the report quotes the last two:

1. `lm_eval version: <v>`: expect **0.5.0.dev1**. A preceding `lm_eval not present; installing`
   lands on the same pin, but notes the image lacks the harness. A version other than the
   baseline's makes the delta unclean.
2. `vllm version: <v>` and `vllm-torchtpu commit: <sha>`, both into the report. `unknown` means
   the image sets no `VLLM_TORCHTPU_COMMIT`: record the commit from the image and say where from.
3. `Eval endpoint: http://…/v1/chat/completions`
4. `smoke: content=<...>`. A reasoning model may print `content=<null>` then `smoke: content was
   null; reasoning=…`; neither is diagnostic alone.
5. `eval-summary: null-content answers=<n>`: the line that decides.
6. `eval-summary: <task> exact_match,…=<score> … sample_len=<n>`, then one row per subtask.
7. `eval exited rc=0`

Release the arm's slot as agent.md "stop" does, on every exit: job done, submit failed, job
failed, and before any return.

## Step 4: Verify the results artifact

```bash
{{runtime}} compute job artifacts --package {{package_tag}} --job-id <job_id>
```

`eval_out.tar.gz` and `server.log` land at `{{runs}}/<run_id>/artifacts/<job_id>/`. Unpack the
archive there: `results_<ts>.json`, and `samples_*.jsonl` with `--log_samples`. A run that failed
before lm_eval wrote it has an empty archive; the `eval-summary:` lines are a complete substitute.

- **`sample_len` matches the size**: ~1400 for `--limit 100`, 12,032 for the full set. A mismatch
  is `failed`, like a mistyped key.
- **Sanity-check the magnitude.** MMLU-Pro is 10-way: a score near 0.10 is chance and means
  answers are not parsed; read a sample. A score far below baseline is more often a protocol error
  (Step 1) than a regression.

A baseline arm that fails, or fails these checks, makes the request `failed`.

## Step 5: Report

`delta_pp` = (stack − baseline) × 100. `gate`: `fail` below −1.5, else `pass`; `inconclusive`
below `--limit 100` or against a baseline at another size, whatever the score. Write
`<out>/accuracy-<config>.md` and return it as `report`:

```
accuracy:
  config:      <config name>, <1node | 2node>
  image:       <arm images and plugin ref/sha or patches>
  recipe:      accuracy-check eval recipe · <package tag>
  server:      <VLLM_SERVER_CMD, as read back>
  eval:        <EVAL_CMD, as read back>, protocol from <baseline record>
  job:         <job id per arm> · <job status>
  harness:     lm_eval <version, from the log>
  score:       <task> <metric>=<value> ± <stderr> (n=<sample_len>)
  baseline:    <value> ± <stderr> (n=<samples>) from <record or arm job>, at <server shape>
  size:        <--limit 100 (the gate) | full (n=12032) | plumbing (smaller)>
  paired:      <same size as the baseline? if not, the delta is not comparable>
  gate:        1.5pp: <delta_pp> → <pass | fail | inconclusive> (<why>)
```

`not recorded` is not a value: an empty field says in words why the work did not happen. Quote
numbers and commands as read back from the job log. A qualifier on the baseline goes inside the
gate line, never in place of it. Report a regression plainly: it is the outcome this eval exists
to produce. This record is the next run's Step 0 baseline.

## Verified mechanisms

These come from runs of the earlier generic eval recipes and establish the protocol and harness.

> **The eval path, 2026-09-01**, `j-fa9c39fa-e4de-4c0f-a8c9` (`Succeeded`, 1 node,
> Qwen3.5-35B-A3B-FP8 DP8/EP, `--limit 5`): lm_eval 0.5.0.dev1 from the image, server up after
> 955 s, 14.4 min for 70 samples, `results_*.json` and `eval-summary` lines present. Score 0.400,
> later shown to be a protocol error.
>
> **The protocol, 2026-09-01**, `j-0a9b1f22-f43d-4c6d-a4e1`, same config with only
> `--gen_kwargs {\"chat_template_kwargs\":{\"enable_thinking\":false}}` added: 0.7571 ± 0.0515 at
> n=70, against 0.400 with thinking on. The evidence for the `enable_thinking` rule.
>
> **Qwen3.5-397B, 2026-09-02**, sweep `j-b1817590` / `j-8846a1e0` / `j-3c80a2b6` (TP=1/DP=8/EP,
> lm_eval 0.5.0.dev1, `mmlu_pro` group 2.0 / subtasks 3.0): 0.8314 ± 0.0139 (n≈700),
> 0.8250 ± 0.0100 (n≈1400), **0.8201 ± 0.0034 (n=12,032)**, each inside the previous interval.
> `j-8846a1e0` was deleted afterwards; its score was recovered from its archived eval log.
>
> **GLM-5.2's baseline**, `j-012c51a9-a247-4d76-914b`, generic 2-node, `--limit 100`,
> 0.7500 ± 0.0113 at n=1400, 0 null-content answers, 3h53m. No full-set run exists.
>
> **Two failure modes, both fixed by rules above.** `j-bd2a68c6-fbad-410f-9e05`: `Weight
> input_size_per_partition = 64 is not divisible by weight quantization block_k = 128`; TP=8
> shards the expert `down_proj` below the fp8 block floor, DP8/TP1 works.
> `j-09ab11f8-b120-49c0-bed2`: `--limit-mm-per-prompt: Value image:0 cannot be converted`, the
> unescaped-JSON trap.
