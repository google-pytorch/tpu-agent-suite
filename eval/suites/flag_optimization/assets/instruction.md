# Speed up {{model}} pre-training on {{slice}}

I have a MaxText pre-training job for {{model}} running on a {{slice}} slice with a default config. I want the highest training throughput I can get out of it (tokens per second, i.e. the lowest steady-state step time) without changing what it computes. Please tune only:

{{scope}}

Write the best config you find to **`/workspace/output/best_config.json`** (format below).

You have **{{budget}}** for this (`flagopt timer` shows what's left). Keep `/workspace/output/best_config.json` holding your best verified config at all times; whatever is in it when your time is up is what I'll evaluate.{{note}}

## Setup

- MaxText v0.2.6 from the official `tpu_pre_training` image. The same install is in your container at `/deps/src/maxtext` if you want to read the code or the config schema.
- Model: MaxText `model_name={{maxtext_model}}` (Hugging Face [`{{hf_repo}}`](https://huggingface.co/{{hf_repo}})).
- Global batch: {{batch}}.
- {{precision}}; AdamW. The data is MaxText's synthetic batch with a fixed seed and dropout is off, so two correct configs produce the same loss and gradient-norm curves step by step.
- TPU work runs as Kubernetes Jobs on the shared GKE cluster `bodaborg-tpu7x-nap` (project `cloud-tpu-shared-capacity`, `us-central1`), namespace `priority-dev`, Kueue queue `default`. `kubectl` in your container is already authenticated and only sees your own jobs. The `GKE_TPU_ACCELERATOR` / `GKE_TPU_TOPOLOGY` environment variables don't describe this slice, so don't size or schedule work from them.
- `flagopt` is installed to run, measure and compare training runs on the `tpu7x` `{{topology}}` slice; its guide is in `/opt/flagopt/README.md`.

## What you can and can't change

`flagopt example` prints the current config. Only these MaxText overrides may appear in your config (knobs you leave out keep their current value):

{{knobs}}

{{rules}}

Everything else is fixed: the model (size, layers, hidden size, experts, vocab), precision and quantization, batch size and sequence length, the optimizer and its hyperparameters, the data, the checkpoint and the step-time measurement. `flagopt spec` lists the fixed settings and `flagopt validate` rejects any change to them. Don't write or replace kernels; choosing among MaxText's existing kernel options and their hyperparameters is fine where the list above includes them.

## Deliverable

Write **`/workspace/output/best_config.json`**: a JSON object with exactly these keys:

```json
{
  "maxtext_overrides": {"<knob>": <value>, "...": "..."},
  "xla_flags": ["--xla_flag=value", "..."],
  "notes": "optional free text"
}
```

- `maxtext_overrides` maps tunable knob names to values of the types `flagopt spec` lists (integers as JSON integers, booleans as `true`/`false`, enums as strings).
- `xla_flags` is a list of strings (may be empty).
- `flagopt validate /workspace/output/best_config.json` must print `VALID`.

Also write `/workspace/output/report.md`: a table of every run you made (config, p50 / p90 step time, tokens/s, pass/fail), the best config and its measured speedup over the current config, and the reasoning behind it.

## How I'll evaluate it

After your session ends, I'll run your `best_config.json` in a fresh Job on the same slice type with the model checkpoint, measured the same way as `flagopt run`, and compare it with the current config measured the same way.

- The config must be valid, compile, fit in memory and finish the run.
- Accuracy gate: over the first 15 steps, the per-step training loss must match the current config's within 0.1% (relative) and the gradient norm within 5%. Performance knobs shouldn't change the math; a config that does is rejected.
- Your score is the throughput improvement: current median step time / your median step time - 1 (0.25 for 25% more tokens per second). There is no target to hit: the more you improve throughput, the higher the score. A config that is slower than the current one, invalid, crashes or fails the gate scores 0.

## Ground rules

- It's a shared cluster: only create, inspect or delete Jobs you started yourself, and create them with `flagopt run` so each one is recorded in `/workspace/runs/job_log.md`. Never touch, read the logs of, or preempt workloads you didn't start, and delete every Job you created before you finish.
- Wait only with bounded commands (`flagopt run` has queue and run timeouts); don't write your own sleep / polling loops.
- Go ahead without asking me for confirmation. If something blocks you (no capacity, a failing baseline), say so plainly in `report.md` and still leave the best config you could verify in `best_config.json`.
