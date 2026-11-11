---
scripts:
- scripts/common/tpu_slots.py
- scripts/common/vllm_patches/apply_vllm_patch.sh
---

# Code arm procedure

Test one `code` hypothesis on a gap: the smallest change to `vllm_torchtpu` code that is not a
Pallas kernel, committed on the gap's branch, on the caller's stack head, with probes that show the
plugin imports, the op's outputs match the incumbent and what the op now costs per occurrence. The
whole-model A/B is the caller's (hill-climb 4.4); never run one.

You run to the end without asking anyone. `<slug>` is `checkout.branch` less `hyp/`; `<arm>` is
`{{runs}}/<run_id>/artifacts/tpu-code-arm/<slug>`.

## 1. Read

Read the plugin source in `checkout.path` at the diagnosis's `where`, its callers and the op's
shapes and dtypes from the gap's `evidence`. A change the hypothesis needs outside `vllm_torchtpu/`
(torch_tpu, XLA, vLLM itself) is `blocked` with what it needs, before any job.

## 2. Change and commit

Make the smallest change that tests the hypothesis, editing files in `checkout.path` only. You never
run git: send {{agent:tpu-code-stack}} a `commit` with `repo`, `worktree` `checkout.path`,
`caller: writer` and the lever as `message`. Its `plugin` is the candidate; keep each commit's
`sha` for `evidence.commits`. `refused` for a path
outside the plugin is fixed in the files and sent once more; a second `refused`, or one off `base`,
returns `blocked` with its reason.

## 3. TPU jobs: stack head, slots, budget

**Every probe runs on `stack_head.image`.** The incumbent is the stack head's own plugin
(`stack_head.plugin.patch`, or the image as is while `stack` is at `base`); the candidate is the
last commit's `plugin.patch`. Each job ships the applier and one patch; its recipe `run.sh` runs
`bash apply_vllm_patch.sh plugin.patch`, prints `apply-exit <code>`, then `python -u probe.py`. The
incumbent at `base` ships no patch: its `run.sh` skips the applier and prints `apply-exit none`, and
its submit drops that `--input`. A candidate also ships the incumbent's `outputs.pt` as
`incumbent.pt`.

```bash
{{runtime}} compute submit --profile <profile> --image <stack_head.image> \
  --recipe <arm>/probes/<nn>-<side>/run.sh --interpreter bash \
  --input {{script:common/vllm_patches/apply_vllm_patch.sh}} --input <arm>/probes/<nn>-<side>/plugin.patch \
  --input <arm>/probes/<nn>-<side>/probe.py --output result.json --output outputs.pt \
  --run-id <run_id> --name tpu-code-arm-<slug>-<nn>
```

Save `{{runtime}} compute job logs --job-id <job_id>` as `probes/<nn>-<side>/<job_id>.txt`.

**Claim a slot before every submit**, from the request's run, under a holder unique to that job:
`tpu-code-arm:<slug>:<nn>`. `claim` hands a holder back the slot it already holds, so a shared
holder would put every job on one slot.

```bash
python3 {{script:common/tpu_slots.py}} claim --run-id <run_id> \
  --holder tpu-code-arm:<slug>:<nn> --hardware <target.hardware>
python3 {{script:common/tpu_slots.py}} set-job --run-id <run_id> --resource <resource> \
  --holder tpu-code-arm:<slug>:<nn> --job-id $JOB_ID
python3 {{script:common/tpu_slots.py}} release --run-id <run_id> --resource <resource> \
  --holder tpu-code-arm:<slug>:<nn>
```

Submit with the `profile` the claim prints. Exit 2 (no `parallel.json`): run `discover` once
(`--runtime "{{runtime}}" --package {{package_tag}}`), then claim again; a second exit 2 returns
`blocked`. Exit 3 (refused): wait for one of your own jobs to end and release it; if you hold none,
return `blocked` with the `show` output — your caller cannot free more while it waits on you.

**Release on every exit**: when a job ends, on a failed submit, on a cancelled or failed job, before
every delegation, and before you return, whatever the status. No raw docker or podman.

**Budget.** `budget.jobs` counts every submit. An apply exit 2 or 3, or a job that fails for
infrastructure, relaunches at most twice; past that, return `failed`. Apply exit 4 is no verdict:
the patch is cut from `base`, so the campaign repo's `base` is not this image's. Return `failed`
with `failure_reason: base mismatch: <base_sha> vs <stack_head.image>`, never `refuted`.

## 4. Probe

`probe.py` imports `vllm_torchtpu`, builds the op's inputs on the gap's shapes and dtypes from a
fixed seed, calls the function at `where`, saves its outputs moved to CPU to `outputs.pt`, and times
it per occurrence: warm up, then the median and spread of at least 50 synced calls, in
`result.json`.

Run the incumbent first, once, and reuse it. The host has no torch, so the comparison runs in the
candidate's job: its probe compares its outputs to `incumbent.pt`, bitwise for a change that only
moves or reorders data, else `torch.testing.assert_close` at the dtype's default tolerance, and
writes `outputs_match` to `result.json`.

A probe that fails to import, crashes or mismatches with a clear fix gets one more edit and commit
(step 2) and one more candidate probe. Outputs that still do not match, or a plugin that still does
not import, are `refuted`. A candidate not faster than the incumbent by more than the larger of
`noise_floor_pct` and the two probes' spreads is `refuted` too. When the time moves the wrong way and the cause is unclear, capture a trace in the
probe and send it to {{agent:tpu-diagnostics}} before deciding; such a probe adds
`--output trace.tar.gz`, unpacked under `probes/<nn>-<side>/trace/`, whose path goes as `traces`.

## 5. Review and return

Before returning `changed`, release your slots and send {{agent:tpu-critical-review}} a
`kind: claim`: the op's before and after per occurrence in the claim, the probe results, logs and
the diff as `artifacts`, every probe `job_ids`, `image_digests` for both sides, the probes'
`recipe.yaml` read-backs as `recipe_paths`, and `noise_floor` with `spread_pct` the threshold of
step 4 and `source` the probe job ids and the caller's step 0 spread. You do not review your own
claims. A review that does not pass is fixed within the budget, or the result is `blocked` with
`reason: review did not pass: <report path>`.

Leave the branch as it is on `refuted`: tags and resets are your caller's.

| status | owes |
| --- | --- |
| `changed` | `plugin`, `evidence` with every probe job, every commit, `op_before_ms`, `op_after_ms`, `outputs_match: true` and the review |
| `refuted` | `evidence`: the probe jobs, `outputs_match` when outputs were produced, and whatever times were measured |
| `blocked` | `reason`: outside the plugin, tpu-code-stack refused, no slot, or the review did not pass |
| `failed` | `failure_reason`, e.g. `base mismatch: …` |
