---
name: accuracy-check
description: Measures a model's accuracy on real hardware against a recorded baseline, using the eval recipe this skill ships and the protocol that produced that baseline, then reports whether the score clears the 1.5pp gate. Use before calling an optimization `accepted`, when asked to "run the accuracy check", "run MMLU-Pro", "check the eval score", "is this change quality-neutral", or to establish the correctness evidence a benchmarked run needs.
scripts:
- scripts/accuracy-check/eval_recipe.sh
references:
- references/common/accuracy-protocol.md
---

# Accuracy check on TPU

You boot one vLLM server on a TPU slice and run `lm_eval` against it, then compare the score to
the baseline recorded for that config. The output is a verdict a caller can act on,
not a number.

This is the **expensive** tier. A real gate run is hours. {{skill:change-preflight}} owns the cheap tier
and is not gated by this skill; reach for that first.

## The protocol

The gate size, the 1.5pp gate, earlier job records, Step 0 (the baseline), Step 1 (the per-model
protocol and server commands), Step 4 (verifying the results artifact) and the verified
mechanisms are in {{reference:common/accuracy-protocol.md}}. Follow it as written; the steps
below are this skill's own.

## Inputs you need

- **`DOCKER_IMAGE`** — reuse the image the caller gave you. **Do not build one.** A rebuild is
  the caller's decision, not a step you take silently. If no image was given, say so and stop.

  **The repo to use is your team's `vllm-torchtpu` image repository:**

  ```
  <registry>/vllm-torchtpu:<commit sha>
  ```

  **Prefer a commit-sha tag over `:latest`.** `:latest` is rolling — on 2026-09-04 it was
  `sha256:00990af4…`, built 2026-09-01, carrying vllm-torchtpu `27ff56aa0a295cdb43a5c835b4cb5f58b4cc7389`
  — and a comparison against a baseline is only about your change if the image is held fixed. If
  you must use `:latest`, record the commit the eval job prints, never the tag.
- **The config name** — e.g. `qwen3.5-35b-fp8-dp8-ep`. It selects both the protocol and the
  baseline, and it must match the layout actually being served. Not the model id: the same model
  has several configs and they do not share baselines.
- **`VLLM_SERVER_CMD`** — the server under test. When this run is gating a benchmark, pass the
  **same** command that benchmark used, read back from that benchmark job's own record of what ran.
  An accuracy number measured at a different shape describes a configuration nobody benchmarked.
- **`MODEL_ID`** — the HuggingFace id, org included. Serve it and pass
  `--served-model-name=<MODEL_ID>`; the recipe takes the same value as `SERVED_MODEL_NAME`.
  `vllm serve <MODEL_ID>` downloads the weights from HuggingFace on the target. A gated model needs
  the target's HuggingFace token. If the target mounts a shared HuggingFace cache, the weights
  download once; check what the target offers with `{{runtime}} compute inspect --json`.
- **the size** — `EVAL_LIMIT=100`, unless the caller explicitly asked for the full set.

## Step 2: Submit the job

The recipe is {{script:accuracy-check/eval_recipe.sh}}. In one job it starts the server from
`vllm_server_cmd.txt`, waits for `/health`, prints the versions, sends one smoke request, runs
`eval_cmd.txt` verbatim, prints the `eval-summary:` lines, and exits with `lm_eval`'s exit code.
The two commands travel as files, so they reach the recipe without a second layer of shell quoting.

Write the two files into `<work>`:

- `<work>/vllm_server_cmd.txt` — the server under test, one shell command line.
- `<work>/eval_cmd.txt` — the command from Step 1 of {{reference:common/accuracy-protocol.md}}, for THIS model.

Then submit:

```bash
{{runtime}} compute submit \
  --recipe {{script:accuracy-check/eval_recipe.sh}} --interpreter bash \
  --image <image> \
  --input <work>/vllm_server_cmd.txt:vllm_server_cmd.txt \
  --input <work>/eval_cmd.txt:eval_cmd.txt \
  --output eval_out.tar.gz --output server.log \
  --run-id <run_id> --name accuracy-check-<config> \
  -- SERVED_MODEL_NAME=<HF id> EVAL_TASKS=mmlu_pro EVAL_LIMIT=100
JOB_ID=$({{runtime}} compute job list --run-id <run_id> --json --no-refresh \
  | python3 -c "import json,sys;print(next(r['job_id'] for r in json.load(sys.stdin) if r['job_name']=='accuracy-check-<config>'))")
```

Pass `EVAL_LIMIT=full` only when the caller asked for the full set. The recipe accepts only the
keys `SERVED_MODEL_NAME`, `EVAL_TASKS`, `EVAL_LIMIT`, `SERVER_PORT`, `SERVER_WAIT_TIMEOUT_S` and
`LM_EVAL_VERSION`, and exits non-zero on any other key.

**The recipe is single-host.** The GLM-5.2 protocol needs two hosts. A 2-node run needs a
multi-host backend and a recipe that creates the Ray workload itself; this skill does not ship
one. Check the configured target with `{{runtime}} compute inspect --json`.
If it cannot hold the model's baseline layout, report that as the blocker and stop. Do not run the
model at a different layout and compare it to the baseline. A 2-node GLM-5.2 run must carry
the values its baseline ran with:

- `MODEL_ID=zai-org/GLM-5.2-FP8`, `SERVED_MODEL_NAME=zai-org/GLM-5.2-FP8`
- `EVAL_TASKS=mmlu_pro`, `EVAL_LIMIT=100`
- `VLLM_ENGINE_READY_TIMEOUT_S=9600`, and a server wait of the same length
- the server env `VLLM_DISABLE_COMPILE_CACHE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn TORCHINDUCTOR_AUTOGRAD_CACHE=0`
- `EVAL_CMD` = the GLM command from Step 1 of {{reference:common/accuracy-protocol.md}}

**Always pass `EVAL_CMD` explicitly.** The recipe has no default for it and exits non-zero without
`eval_cmd.txt`. The baselines' original recipes carried a default `EVAL_CMD` that matched neither
baseline; do not reconstruct one.

Two more parameters matter when porting a command from a dedicated recipe:

- **`SERVER_PORT`** (default 8000) — the recipe builds `${EVAL_BASE_URL}` from it, so the
  `--port` in `VLLM_SERVER_CMD` must match. Porting GLM's command meant changing its 47903 to
  8000 (passing `SERVER_PORT=47903` would have worked equally).
- **Server environment** — put it in the server command: `export A=1 B=2 && vllm serve …` inside
  `VLLM_SERVER_CMD`. That has no limit on spaces in a value (`LIBTPU_INIT_ARGS`). An `export`
  there is how `VLLM_ENGINE_READY_TIMEOUT_S=9600` is set. When you raise it, raise the recipe's
  `SERVER_WAIT_TIMEOUT_S` (default 3600) to match, or the recipe stops waiting first.

**The recipe guards `EVAL_CMD` before it starts the server.** It checks that the string mentions
`base_url` and exits non-zero if not. It accepts both spellings, comma-kv **and** JSON.

**`EVAL_CMD` is run verbatim — nothing is appended to it.** This is the difference from a bench
recipe, which appends the server host: `vllm bench serve` takes its endpoint as `--host`,
`lm_eval` takes it inside `--model_args`. Use `${EVAL_BASE_URL}` (already the full
`/v1/chat/completions` URL), `${EVAL_OUTPUT_DIR}`, `${EVAL_TASKS}`, `${EVAL_LIMIT_ARG}` and
`${SERVED_MODEL_NAME}`; the recipe exports all of them before the `eval`.

### Quoting: the rule that still applies

The eval command is `eval`'d in the container. That means:

- **JSON needs its double quotes escaped as `\"`.** `eval` performs quote removal, so an
  unescaped `{"enable_thinking":false}` reaches the program as `{enable_thinking:false}` and dies
  in argument parsing. Escaped, it arrives intact:

  ```
  --gen_kwargs {\"chat_template_kwargs\":{\"enable_thinking\":false}}
  ```

  Verified both directions on 2026-09-01: escaped survives as one argv element; unescaped splits
  into `image:0`, `video:0`, which is exactly how job `j-09ab11f8` died.

  **The rule is about wrapping, not about spaces.** JSON wrapped in outer `"…"` may contain
  spaces freely — the baseline `EVAL_CMD` and its `--limit-mm-per-prompt` both do, and both work.
  It is *unwrapped* JSON that must not contain them. Do not strip spaces out of a working wrapped
  command to satisfy a rule it is not subject to.

- **A list-valued `gen_kwargs` keeps the baseline's spelling.** The GLM stop list is written with
  escaped double quotes inside the wrapped value:

  ```
  --gen_kwargs "max_new_tokens=4096,until=[\"<|im_end|>\"]"
  ```

  `lm_eval` parses the value with `literal_eval`, so `["<|im_end|>"]` and `['<|im_end|>']` are the
  same list. It reaches lm_eval as one argv element, `max_new_tokens=4096,until=["<|im_end|>"]`.
  The `|` characters are safe because they are inside the quotes.

**Confirm the image tag exists before submitting.** A tag that does not resolve schedules the
slice and then fails to pull the image, having reserved the hardware to do it:

```bash
gcloud artifacts docker tags list <repo path> --filter=tag:<tag> --format='value(tag)'
```

**Record the recipe version at submit time.** The recipe is part of this package, so the package
tag, `{{package_tag}}`, identifies what ran. Carry it into the report.

**Read back what was actually submitted**, on every job. The recipe prints every effective value
before it starts the server:

```bash
{{runtime}} compute job logs --job-id $JOB_ID | grep '^effective '
```

**Check the keys.** The recipe rejects an unknown key, so a mistyped `KEY=VALUE` fails the job
instead of being ignored. Still confirm `effective EVAL_LIMIT:` — that is the one that costs a
gate. Do the same for `effective EVAL_CMD:` and `effective VLLM_SERVER_CMD:`; those two are what
the report must quote.

## Step 3: Watch the job

```bash
{{runtime}} compute job wait --job-id $JOB_ID \
  --until "log:vllm-server is up" --fail-on-log '^eval exited rc=[1-9]' --json
{{runtime}} compute job wait --job-id $JOB_ID --until done --json
{{runtime}} compute job logs --job-id $JOB_ID
```

A gate run takes hours. Exit 3 from `job wait` means it is still waiting; do what its `next`
says, which while the gate runs is to run the same command again.

The job log carries the recipe's own lines. The server's output goes to `server.log`, which comes
back with the artifacts (Step 4 of {{reference:common/accuracy-protocol.md}}). When the server exits or does not answer `/health` in time, the
recipe prints the last 200 lines of `server.log` into the job log before it exits.

The recipe's exit code is the job's status, so a failed job is a real failure. A successful job is
necessary and not sufficient: it says the process exited 0, not that the score is meaningful.

Lines to find in the job log, in order:

1. `lm_eval version: <v>` — expect **0.5.0.dev1**. A preceding `lm_eval not present; installing`
   is not a problem in itself: the recipe pins the fallback
   (`uv pip install --system --no-config "lm-eval[api]==0.5.0.dev1"`), so both paths land on the
   same version. It does tell you the image lacks the harness, which is worth noting. What
   actually matters is the version *printed*: if it differs from the baseline's, say so and do not
   treat the delta as clean.
2. `vllm version: <v>` and `vllm-torchtpu commit: <sha>` — the serving stack that produced the
   score. A score without them is not reproducible; carry both into the report. The recipe reads
   the commit from the image's `VLLM_TORCHTPU_COMMIT` variable and prints `unknown` when the image
   does not set it; then record the commit from the image tag and say where it came from.
3. `Eval endpoint: http://…/v1/chat/completions`
4. `smoke: content=<...>` — one real answer before the run commits hours of TPU time. A
   reasoning model may instead print `content=<null>` followed by `smoke: content was null;
   reasoning=…`, which is the chain of thought landing in `reasoning` rather than a broken
   server. Do not treat either as diagnostic on its own: GLM-5.2's baseline, served with
   `--reasoning-parser=glm45`, printed `content=391` and `finish_reason=stop`. The line that
   decides is `eval-summary: null-content answers=<n>` at the end.
5. `eval-summary: null-content answers=<n>` — see "Record the harness version" in {{reference:common/accuracy-protocol.md}}.
6. `eval-summary: <task> exact_match,…=<score> … sample_len=<n>`, then one row per subtask.
7. `eval exited rc=0`

## Step 5: Report

```
accuracy:
  config:      <config name>, <1node | 2node>
  image:       <image path>
  recipe:      accuracy-check eval recipe · <package tag>
  server:      <VLLM_SERVER_CMD, as read back from the `effective VLLM_SERVER_CMD:` line>
  eval:        <EVAL_CMD, as read back>, protocol derived from <files read>
  job:         <job id> · <job status>
  harness:     lm_eval <version, from the log>
  score:       <task> <metric>=<value> ± <stderr> (n=<sample_len>)
  baseline:    <value> ± <stderr> (n=<samples>) from <baseline record>,
               measured at <which server shape>
  size:        <--limit 100 (n~1400, the gate) | full (no --limit, n=12032) | plumbing (smaller)>
  paired:      <is the baseline at the same size? if not, say so — the delta is not comparable>
  gate:        1.5pp drop — <delta vs baseline, and whether it clears>
  verdict:     <✅ passed | ❌ failed | ⚠️ inconclusive> — <why>
```

`not recorded` is not a value here. If a field is empty because that work did not happen, say so
in words — a hollow block read quickly looks like a check that ran.

**Quote numbers, never summarize them**, and quote the commands as read back from the job log
rather than as you meant to send them.

`⚠️ inconclusive` is the honest verdict for any run below `--limit 100`, whatever the score was,
and for a run whose baseline was measured at a different size — that delta is not a paired
comparison. It is not a failure of the run; it is a statement about what the run can support.

`verdict:` always takes one of the three values above. Do not use `provisional` as a verdict.
If the baseline carries a limitation, write the qualifier **inside** the verdict rather than
replacing it: `✅ passed (provisional — <limitation>)`.

Report a regression plainly. A score that drops is the outcome this check exists to produce, and
it is worth more to your caller than the hours it saves.

**Then close the loop: record the run.** Write the report block to the run's artifacts directory,
`{{runs}}/<run_id>/artifacts/accuracy-check-<config>-<job_id>.md` — score, stderr, `n`, tier, the
baseline it was checked against, the protocol, and the job id. That is what makes this run's
number the next run's baseline, and it is the reason Step 0 has anything to read.
