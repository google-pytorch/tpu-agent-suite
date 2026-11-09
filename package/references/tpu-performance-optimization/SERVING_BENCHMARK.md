---
references:
  - references/common/inference-serving-benchmark.md
  - references/tpu-model-onboarding/torch-inference-coder.md
scripts:
  - scripts/common/serving_benchmark.py
  - scripts/tpu-performance-optimization/serving_eval.py
---

# Serving Benchmark

How a study measures a model served behind an OpenAI-compatible endpoint
(`vllm serve` on `vllm-torchtpu` or `python3 -m sglang.launch_server` on
SGLang): one client for speed and one for accuracy, each with one invocation,
the same for the baseline and for every arm.

## 1. The speed client

The speed half is the serving benchmark every agent shares,
{{script:common/serving_benchmark.py}}. What it measures, its metric names
(the names a study protocol's `primary_metric.name` and
`admissibility_gates[].metric` use), how it keeps prefix-cache hits out of
every run (`POST /reset_prefix_cache` on vLLM or `POST /flush_cache` on
SGLang), and its exit codes are in
{{reference:common/inference-serving-benchmark.md}}. This document adds what a
study needs on top of it.

## 2. Invocation

Run the client through a port forward, as the common reference's "Through a
port forward" shows, against a server started from the baseline recipe:
submit it with `compute submit` under a `--name`, recover its `job_id` from
`compute job list --json`, and wait with `compute job wait --until running`,
then `--until "log:Application startup complete"`. Never pipe or capture
`submit`'s output: the server holds the pipe open and the call blocks until
the server exits (see {{reference:tpu-model-onboarding/torch-inference-coder.md}}).
For example:

```bash
python3 {{script:common/serving_benchmark.py}} \
  --base-url http://localhost:8000 \
  --tokenizer <hf_model_id> \
  --output-dir <work>/serving_bench \
  --isl-osl 8192:1024 --concurrency 1,4,8,16,32 --replicates 3 \
  --reference-curve <work>/reference_curve.json \
  --tolerance-pct 5.0 \
  --export-bkc <work>/bkc.json
```

When every point on the reference Pareto frontier is reproduced within
`--tolerance-pct`, `summary.json` records
`reproduction: {"consistent": true, "promote_to_bkc": true, ...}` and writes the
verified BKC manifest to `--export-bkc`.

Run the accuracy eval (Section 4) through the same forward, and kill the
forward's PID when both are done, whatever they returned. Start the baseline
recipe's server with `VLLM_SERVER_DEV_MODE=1`, so the client can reset the
prefix cache before every run; whether it could is in
`settings.prefix_cache`, and a baseline and an arm measured differently do not
compare.

## 3. Using it in a study

- **Pre-register the cell.** `measurement_cell.shape` is the `ISL-OSL` pair and
  `measurement_cell.concurrency` the concurrency level, as passed to
  `--isl-osl` and `--concurrency`. The primary metric is one of the names
  above, for example `output_tokens_per_s` (`maximize`) or `ttft_p50_ms`
  (`minimize`).
- **Let the baseline server batch.** The client sends requests concurrently
  (`--concurrency`, and the eval's `--num-concurrent`), but the server runs
  at most `--max-num-seqs` of them per step and queues the rest. A baseline
  launched with `--max-num-seqs 1` serves one request at a time: the eval
  takes most of an hour instead of minutes, every concurrency above 1
  measures queueing, and the first arm that raises it claims a speedup that
  is only the recipe's own limit. Launch the baseline with `--max-num-seqs`
  of at least 8 and at least the highest `--concurrency` measured. Raise it
  further, up to the eval's `--num-concurrent`, if the server still starts
  and serves without running out of HBM; fix the value in the baseline
  recipe before the baseline runs, as the protocol's other terms are.
- **Set the noise floor from the baseline's spread.** Run the baseline with
  `--replicates` of at least `protocol.replicate_count` and read `cv_pct` for
  the primary metric. Throughput at a fixed concurrency is usually steady to
  about 1%. A TTFT of a few tens of milliseconds can vary by tens of percent
  between replicates, so a TTFT primary metric needs a long prompt, more
  replicates, or a wider floor.
- **Measure every arm with the baseline's exact invocation.** Record the full
  command line in the baseline recipe. An arm changes the server (its flags,
  environment, kernels or code), never the client: a changed `--isl-osl`,
  `--concurrency`, `--num-prompts`, `--temperature` or `--endpoint` measures a
  different cell, and the comparison is void. Never pass exploratory sweep
  flags (`--stop-below-gain-pct`, `--stop-on-failure` or `--avoid`) in a study:
  they can stop the baseline and an arm at different concurrencies. This script
  is a benchmark harness: it is never edited for an arm.
- **Keep the accuracy gate on the same execution.** A throughput number from
  this client and an accuracy number from another server launch do not
  describe the same configuration. Run the accuracy eval (Section 4) against
  the same running server before tearing it down, and record both under that
  server's `execution_id`.
- **Report from `summary.json`**, not from the client's console output. Quote
  the primary metric's `mean` with its `cv_pct` and the replicate count.

## 4. Accuracy eval

A change that makes the server faster can also make it wrong: a kernel with a
numerics bug, a quantization step, a scheduler change that corrupts a batch.
`{{script:tpu-performance-optimization/serving_eval.py}}` is the accuracy half of the
measurement. It runs lm-evaluation-harness against the server's
`/v1/chat/completions` with the invocation `vllm-torchtpu`'s own eval flow (`run_eval_flow.sh`)
uses: 5-shot examples sent as prior
chat turns, reasoning switched off with `enable_thinking: false`, and `--limit`
questions per subject. Like the benchmark, it names no model: the served name
comes from `/v1/models`, and the server applies the chat template, so no
tokenizer is needed on this host.

```bash
python3 {{script:tpu-performance-optimization/serving_eval.py}} \
  --base-url http://localhost:8000 \
  --output-dir <work>/serving_eval \
  --tasks mmlu_llama --limit 100
```

It writes `eval_summary.json`, whose `scores` are keyed by lm-eval task name:
`scores.mmlu_llama.value` is the MMLU score (fraction correct) and
`scores.mmlu_llama.stderr` its standard error. Groups (`mmlu_llama_stem`,
`mmlu_llama_humanities`, ...) are `headline` scores and are gated; single
subjects are kept but not gated, because a score over a few dozen questions is
mostly noise. The task name is the gate's `metric` in the protocol, for
example `{"name": "mmlu-floor", "metric": "mmlu_llama", "comparator": "gte",
"bound": <baseline - allowed drop>, "same_execution": true}`. Any lm-eval task
works through `--tasks` (`mmlu_pro`, `gsm8k`, ...).

Pass the baseline's `eval_summary.json` with `--baseline` and the client also
applies `vllm-torchtpu`'s regression gate (`check_regression.py --mode eval`):
a headline score fails when it drops by more than
`max(--tolerance, 1.645 x hypot(stderr_baseline, stderr_run))`. That is a
one-sided 95% test on the difference, so the run-to-run noise of a few hundred
questions does not fail it. The default tolerance is 0.015 (1.5 points).

Exit codes: `0` every task scored and, with `--baseline`, every headline score
passed; `1` the gate failed (`status: "gate_failed"`, see `gate`); `2` the
server was unreachable or stopped answering `/health` after a failed task, or
`--baseline` was unreadable, not `status: "ok"` or
had no headline score; `3` lm-eval failed or timed out, wrote no score for a task, or more
than `--max-unscored-pct` (10%) of answers were empty or unparseable (see
`errors` and `unscored_answers`). An unscored answer counts as wrong, so a run
full of them reports the reply format, not the model: a score of 0 is that
failure, never a result.

**Reasoning models** (gpt-oss, and any model whose template has no
`enable_thinking` switch) do not fit `mmlu_llama`: it allows 10 tokens, stops
at the first `.`, and scores an exact continuation of "The best answer is",
so the reasoning never reaches an answer. Use a task that extracts the answer
with a regex and give the model room, for example:

```bash
python3 {{script:tpu-performance-optimization/serving_eval.py}} --base-url http://localhost:8000 \
  --output-dir <work>/serving_eval --tasks mmlu_pro --num-fewshot 0 --single-turn \
  --system-instruction 'Think step by step, then finish your reply with "The answer is (X)" where X is the letter of the correct option.' \
  --gen-kwargs '{"max_gen_toks": 1024, "until": [], "reasoning_effort": "low"}'
```

The server's `max_model_len` must hold the prompt plus `max_gen_toks`; 5-shot
MMLU prompts alone run to about 3,000 tokens, and a refused request fails the
run with the server's reason in `errors`.

The client needs `lm-eval` on this host (`pip install "lm-eval[api]"`), or
`--lm-eval-cmd` pointing at an environment that has it.

- **Size it from the baseline.** `--limit 100` scores 5,700 MMLU questions and
  gives a standard error of about half a point; `--limit 20` scores 1,140 in
  about a minute on a small model at the cost of about twice the error, and a
  correspondingly wider allowed drop. Pick it before the baseline runs, as the
  protocol's other terms are.
- **Expect small differences on an unchanged server.** Batched inference is
  not bit-exact even with greedy decoding, so a rerun of the same server
  changes a handful of answers. The stderr term in the gate is what absorbs
  that; do not tighten `--tolerance` to zero.
- **The eval is a harness too.** Run the baseline's exact invocation for every
  arm and never edit the script, `--tasks`, `--limit`, the few-shot format or
  `--gen-kwargs` for an arm.
