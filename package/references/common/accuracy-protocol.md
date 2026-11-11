# Accuracy protocol

The fixed part of an accuracy check: the gate size and the 1.5pp gate, the baseline to compare
against (Step 0), the per-model protocol and server commands (Step 1), how to verify the results
artifact (Step 4), and the mechanisms verified so far. The accuracy-check skill follows it; its
Steps 2, 3 and 5 (submitting, watching and reporting) are its own.

## One size: `--limit 100`

`mmlu_pro` runs at `--limit 100` — 100 questions per subject, 14 subjects, **~1400 samples**.
That is the gate: there is no sizing decision to make and no larger run required before a change
is checked in. The full set (omit `--limit`, 12,032 samples) stays available for an absolute
quality number; it is never a precondition for `accepted`.

`--limit` is applied **per subject**, not overall, which is why 100 gives ~1400 and `--limit 5`
gives 70. Anything below the gate size is a plumbing test, not a result: say so in the report and
do not compare it to a baseline.

**The one rule that replaces sizing: baseline and candidate must be at the same size.** A
`--limit 100` score checked against a full-set baseline is not a paired comparison, and the
mismatch is worth more than the gate.

## The gate is 1.5pp, and it is not configurable

**A drop of more than 1.5 percentage points against the `mmlu_pro` baseline fails.** Fixed, for
every model and every config. The 1.5pp figure is calibrated on `mmlu_pro` at `--limit 100`; it is
not a generic tolerance to carry to another task.

## Earlier job records

The baseline and evidence job ids in this file (`j-…`) are records of runs on an earlier job
system. They are identifiers for provenance only. `{{runtime}} compute job` cannot read them back.

## Step 0: Get the baseline

**Get the baseline before anything else.** A baseline, the protocol that produced it, and the job
that measured it belong together. Take them, in this order, from:

1. the caller, if the caller gave you a baseline;
2. the accuracy record of an earlier run of this skill on the same config, in that run's artifacts
   directory, `{{runs}}/<earlier run_id>/artifacts/` (Step 5 writes it);
3. the per-model tables in Step 1 and "Server command and baseline, per model".

You need the baseline **for `task: mmlu_pro`** — score, stderr, `n`, tier, the server shape it was
measured at, and the `EVAL_CMD` that produced it. If a source offers baselines on other tasks,
discard them.

If no source has an accuracy baseline for the model, see "Server command and baseline, per model"
below — you are establishing one rather than checking one, and that changes what you report.

### The task is `mmlu_pro`. You will be offered others — do not take them.

**This skill's gate is defined against `mmlu_pro` and nothing else.** Ignore accuracy baselines on
any other task, no matter how they are described.

You will hit this on **Qwen3.5-397B-A17B-FP8**, which has two accuracy baselines:

| | `mmlu_pro` — **use this** | `gsm8k` — do not use |
|---|---|---|
| Recipe | `vllm-torchtpu-generic-eval` | `vllm-torchtpu-generic-**bench**`, `BENCH_CMD` swapped |
| Task source | `--tasks mmlu_pro` | `<InferenceX>/utils/evals/gsm8k.yaml`, cloned at runtime |
| Shape | limit-100, 1400 samples, thinking off | full set, 5-shot, `temperature=0` |
| Score | 0.8250 ± 0.0100 · `j-8846a1e0-1b60-4785-944c` | 0.9689 ± 0.0048 · `j-52e7603c-99b7-48a8-b1eb` |

If a caller explicitly asks for gsm8k, that is a serving job (`{{runtime}} compute submit --serve`)
with the gsm8k client in place of the benchmark client, not this skill. Say so rather than approximating it.

## Step 1: The protocol is fixed — copy it, do not rebuild it

**There is one protocol per model, and it is whichever one produced the baseline you are comparing
against.** Read it out of the baseline record from Step 0 and copy it verbatim.
Do not rebuild it, do not merge the two below, and do not assume either generalizes — they differ
in every field that matters. The recipe this skill ships has no default `EVAL_CMD`: it exits
non-zero without one (see Step 2).

| | **Qwen3.5-397B-A17B-FP8** | **GLM-5.2-FP8** |
|---|---|---|
| Task | `mmlu_pro` (see Step 0) | `mmlu_pro` |
| Baseline recipe | `vllm-torchtpu-generic-eval` | `vllm-torchtpu-generic-eval-2node` |
| Slice | 1 node, TP=1 / DP=8 / EP | 2 nodes, DP=16 / EP over Ray |
| `--model_args` form | JSON | comma-kv |
| `num_concurrent` | 512 | 32 |
| Thinking disabled | yes, in two places | no |
| `--log_samples` | yes | no — **add it to a candidate** |
| `until` | task default | `["<\|im_end\|>"]` |
| Wall clock, `--limit 100` | 1h29m | 3h53m |
| Baseline | 0.8250 ± 0.0100 · `j-8846a1e0-1b60-4785-944c` | 0.7500 ± 0.0113 · `j-012c51a9-a247-4d76-914b` |

Both baselines: commit `ec4bcecf78`, lm_eval 0.5.0.dev1, n=1400, **0 null-content answers**.

---

### Qwen3.5-397B-A17B-FP8 — 1 node

```
lm_eval --model local-chat-completions \
  --model_args "{\"model\": \"${SERVED_MODEL_NAME}\", \"base_url\": \"${EVAL_BASE_URL}\", \"enable_thinking\": false, \"num_concurrent\": 512}" \
  --tasks ${EVAL_TASKS} --apply_chat_template --log_samples \
  --gen_kwargs "{\"chat_template_kwargs\": {\"enable_thinking\": false}}" \
  --output_path ${EVAL_OUTPUT_DIR} ${EVAL_LIMIT_ARG} --seed 42
```

Only `${EVAL_TASKS}` and `${EVAL_LIMIT_ARG}` vary. Every other part is load-bearing:

- **`enable_thinking: false` appears twice, deliberately** — once in `model_args`, once in
  `gen_kwargs`.
- **`--log_samples` is not optional.** When a score comes back wrong, the samples are what tell
  you whether the model is emitting `<think>`, hitting the token cap, or answering fine and being
  mis-extracted. Without it that costs another multi-hour job.
- **JSON `--model_args` requires lm_eval 0.5.x.** 0.4.x parses it with `simple_parse_args_string`,
  which splits on `,` and `=` and silently produces garbage keys. `vllm/vllm-tpu:nightly` ships
  0.4.12; the torchtpu image and the recipe's fallback pin give 0.5.0.dev1.

### GLM-5.2-FP8 — 2 nodes

```
lm_eval --model local-chat-completions --model_args base_url=${EVAL_BASE_URL},model=${SERVED_MODEL_NAME},tokenizer=${SERVED_MODEL_NAME},max_gen_toks=4096,max_length=32768,num_concurrent=32,timeout=7200 --apply_chat_template --tasks ${EVAL_TASKS} --seed 42 ${EVAL_LIMIT_ARG} --gen_kwargs "max_new_tokens=4096,until=[\"<|im_end|>\"]" --output_path ${EVAL_OUTPUT_DIR}
```

**The baseline is sound — read its own log before believing otherwise.** From the eval log of
`j-012c51a9-a247-4d76-914b`:

```
lm_eval version: 0.5.0.dev1
smoke: finish_reason=stop completion_tokens=130
smoke: content=391
eval-summary: null-content answers=0
eval-summary: mmlu_pro exact_match,custom-extract=0.75 ... sample_len=1400
```

Two properties of the command travel with it, neither of which impugns the score:

1. **`until=["<|im_end|>"]` replaces `mmlu_pro`'s own `until: ['Question:']`** with a token GLM
   does not emit, so generation runs to the 4096-token cap on every sample. That costs wall clock,
   not accuracy — it is why the run takes 3h53m.
2. **No `--log_samples`**, so individual answers cannot be inspected afterwards. **Add it to a
   candidate run**: it changes what is written, never what is generated, and is the one safe
   departure from the baseline's command. The recipe this skill ships also counts null-content
   answers from the samples file, so without `--log_samples` it reports that count as `unknown`.

Thinking is not explicitly disabled here, against a server running `--reasoning-parser=glm45`.
With zero nulls there is no evidence this cost anything, and no evidence thinking was off — the
chain of thought may have gone to `reasoning` rather than `content`. Say which you know.

### Rules that apply to both

- **Reuse the baseline's command; do not "improve" on it.** A candidate measured under a different
  command is not comparable to its baseline, whatever either number looks like.
- **Both JSON arguments are wrapped in `"…"` with inner quotes escaped.** Escaping alone is not
  enough when the JSON contains spaces — the outer quotes keep it one argv element through the
  recipe's `eval`. Full quoting rules in Step 2.
- **A model with no generic-recipe baseline has no protocol to copy.** You are establishing one,
  not checking against one — see "Server command and baseline, per model".

### Record the harness version; do not assume a mapping

Read the **lm_eval version** and the **group and subtask versions** `mmlu_pro` reports out of the
results table (and the `versions` map in the results JSON), and record both in the accuracy
record's protocol field. When a candidate and its baseline differ on either, say so and do not
treat the delta as clean.

That is the whole rule. Do **not** carry a version-to-schema mapping here — the schema strings
have been observed to differ between lm_eval versions on the same task, so a table written today
would be wrong tomorrow. Record what the run printed and compare it to what the baseline printed.

The one version constraint that *is* confirmed is above: JSON `--model_args` needs 0.5.x. The pin
itself belongs in the recipe, not here.

### Server command and baseline, per model

The **authority for a server command is the record of a job that ran it** — the
`effective VLLM_SERVER_CMD:` line the recipe prints (Step 2), or the equivalent read-back of the
reference job — not a copy in this file. A second copy drifts, and the read-back is what actually
ran. Take the command from the reference job, change only what the caller is testing.

**Numbers are not kept in this file.** The baseline, the protocol that produced it and the
commands live in the baseline record; this table only says where to look. A number copied here
is a second copy that drifts.

| Model / config | Baseline recipe | Baseline job (`--limit 100`) |
|---|---|---|
| `qwen3.5-397b-fp8-tp1-dp8-ep`, agent shape | `vllm-torchtpu-generic-eval` | `j-8846a1e0-1b60-4785-944c` |
| `glm-5.2-fp8-dp16-ep` | `vllm-torchtpu-generic-eval-2node` | `j-012c51a9-a247-4d76-914b` |

Two consequences, and neither is optional:

- **To reproduce a baseline exactly, pass that tag.** A run that does not pin `DOCKER_IMAGE` to
  the baseline's image reproduces nothing.

**Job ids are full ids here on purpose.** The short form appears everywhere in prose and resolves
nowhere. Config names use the canonical `<model>-<quant>-<sharding>` form so that one name selects
one baseline.

**Only generic-recipe jobs count.** A baseline, and the protocol that produced it, is a
`vllm-torchtpu-generic-eval` or `-2node` job, or a run of this skill's recipe, with a recorded
protocol. Runs on a dedicated per-model recipe, numbers in
`<vllm-torchtpu>/scripts/vllm/benchmarking/baselines/eval/`, and protocols derived from
`<vllm-torchtpu>/configs/<config>.sh` or `run_eval_flow.sh` are **not** comparators — they were
measured at a different shape, under a different harness, or by a job that is attributable to no
commit.

**So there are exactly two things a score can be checked against:**

1. **The recorded `mmlu_pro` baseline** for this model — a generic-recipe job, measured at
   the agent's server shape, carrying the protocol and the job id that produced it. This is the
   normal case, since the accuracy run inherits the benchmark's `VLLM_SERVER_CMD`.
2. **A baseline you measure yourself**, when there is none. That is a run of this same skill on
   the baseline commit at the same size, recorded in the run's artifacts directory,
   `{{runs}}/<run_id>/artifacts/`. Say in the report that the baseline is yours and n=1,
   unreplicated.

**If the model has no generic-recipe run you are in case 2** — establishing a baseline, not
checking one. Measure it, record it, and report it as a baseline rather than as a gate result.
There is no protocol to derive from elsewhere.

## Step 4: Verify the results artifact

A score printed is not a score computed over what you asked for. Before reporting:

```bash
{{runtime}} compute job artifacts --job-id $JOB_ID
```

`eval_out.tar.gz` and `server.log` land at `{{runs}}/<run_id>/artifacts/<job_id>/`. Unpack
`eval_out.tar.gz` there: it holds `results_<ts>.json`, and `samples_*.jsonl` only with
`--log_samples`.

**Do not assume `results_*.json` exists.** A run that failed before lm_eval wrote it has an
empty archive. The `eval-summary:` lines in the job log carry score, stderr and
`sample_len`, so the log is a complete substitute for the JSON.

Confirm that **`sample_len` matches the size you ran** — ~1400 for `--limit 100`, 12,032 for the full set. A mismatch means the run answered a different question — treat it exactly as the bench skill treats a typo'd recipe key.

**Sanity-check the magnitude before believing it.** MMLU-Pro is 10-way, so a score near 0.10 is
chance and means the answers are not being parsed at all — read a sample rather than reporting it.
A score far below baseline is more often a protocol error than a regression; step 1 is where to
look first.

## Verified mechanisms

These records come from runs of the earlier generic eval recipes. They establish the protocol
and the harness behaviour this skill relies on.

> **The eval path was verified end to end on 2026-09-01** by `j-fa9c39fa-e4de-4c0f-a8c9`
> (`Succeeded`, 1 node, Qwen3.5-35B-A3B-FP8 DP8/EP, `--limit 5`). `lm_eval version: 0.5.0.dev1`
> came from the image with no runtime install; checkpoint copy **4.0 GiB/s** for 35 GB;
> `vllm-server is up after 955s`; eval 14.4 min for 70 samples; `results_*.json` landed in
> the job's output directory; the `eval-summary` lines printed. Score 0.400 — later shown to be a protocol
> error, not a model result.
>
> **The protocol was validated on 2026-09-01** by `j-0a9b1f22-f43d-4c6d-a4e1` (`Succeeded`, same
> config, single change: `--gen_kwargs {\"chat_template_kwargs\":{\"enable_thinking\":false}}`).
> `mmlu_pro exact_match,custom-extract = 0.7571 ± 0.0515` at n=70, against 0.400 for the same
> config with thinking left on. Server up in 600 s. That is the evidence for the `enable_thinking`
> rule above; it establishes the field matters, not that any external number is reproduced.
>
> **The full path was validated end to end on Qwen3.5-397B on 2026-09-02** by the three-tier sweep
> `j-b1817590` / `j-8846a1e0` / `j-3c80a2b6` (TP=1/DP=8/EP, agent shape, lm_eval 0.5.0.dev1,
> `mmlu_pro` group 2.0 / subtasks 3.0, as read from `results_*.json`). Scores 0.8314 ± 0.0139 (n≈700), 0.8250 ± 0.0100 (n≈1400),
> **0.8201 ± 0.0034 (n=12,032)** — converging monotonically, every tier inside the previous one's
> interval. All three are generic-recipe jobs, which is what makes them the authority here, and
> they are the evidence behind the size section at the top of this file. `j-8846a1e0` was deleted
> afterwards; its score was recovered from its archived eval log.
>
> **GLM-5.2's baseline** is `j-012c51a9-a247-4d76-914b`, generic 2-node, `--limit 100`,
> 0.7500 ± 0.0113 at n=1400 with **0 null-content answers**, lm_eval 0.5.0.dev1, 3h53m. There is
> no generic-recipe full-set run for this model.
>
> **Two failure modes cost a job each, both fixed by rules above.**
> `j-bd2a68c6-fbad-410f-9e05` died at `ValueError: Weight input_size_per_partition = 64 is not
> divisible by weight quantization block_k = 128` — TP=8 shards this model's expert `down_proj`
> below the fp8 block-quant floor; the DP8/TP1 layout is the one that works.
> `j-09ab11f8-b120-49c0-bed2` died at `--limit-mm-per-prompt: Value image:0 cannot be converted`,
> the unescaped-JSON trap. Both were red, correctly: the recipe propagates the exit code.
>
> Correctness of the *model* was not at issue in any of these — they establish the harness.
