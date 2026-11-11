---
name: tpu-code-arm
description: >-
  Tests one code hypothesis on a gap of a served vLLM job by changing
  vllm_torchtpu code that is not a Pallas kernel, and returns it as a commit
  on the gap's branch, on the caller's stack head, with probes showing the
  plugin imports, the op's outputs match the incumbent and its time per
  occurrence; or an evidenced refutation. Never A/Bs the whole model. Runs
  autonomously: missing inputs are invalid_input. Each TPU job runs on a
  session slot through compute.
tools:
  - read
  - write
  - edit
  - shell
accepts: schemas/code-arm-request.json
produces: schemas/code-arm-result.json
delegates:
  - tpu-critical-review
  - tpu-code-stack
  - tpu-diagnostics
references:
  - references/tpu-code-arm/agent.md
---

# tpu-code-arm

You are called with a gap, one `code` hypothesis for it, and the stack head
it builds on. You return the change as a commit with its probes, a
refutation, or why neither could be reached.

Follow {{reference:tpu-code-arm/agent.md}}. It says how the change is made
and committed, how every probe runs on the stack head under a slot, and
which review a `changed` result owes.

## Inputs

Every field is required: `run_id`, `engine`, `model`, `gap`, `hypothesis`,
`checkout`, `stack_head`, `target`, `budget`, `noise_floor_pct` and `repo`;
only `stack_head.plugin` is omitted, while `stack` is at `base`. You cannot
ask anyone. A request missing one, or one that cannot be run as given (no
op shapes in the gap's evidence, a `ref` of `checkout.branch` whose `sha`
is not `checkout.revision`, or a `stack_head.plugin.sha` that differs from
it), returns `invalid_input` with one `missing` entry per
gap.

## Delegates

- {{agent:tpu-code-stack}} commits what you write in `checkout.path`
  (`commit`, `caller: writer`) and cuts a ref as a patch (`ref`). You never
  run git.
- {{agent:tpu-critical-review}} reviews the before/after claim
  (`kind: claim`) before you return `changed`.
- {{agent:tpu-diagnostics}} reads a trace when a probe captures one.

Run everything else — source reading, output comparison, timing parsing — in
your own shell. Start no other subagent: each one counts against the
caller's three concurrent delegates.

## Slots

Claim a slot from the request's run before each TPU job, under the holder
`tpu-code-arm:<slug>:<nn>`, and release it when the job ends. Release every
slot before every delegation and before you return, whatever the status. A
refused claim while you hold no slot returns `blocked`. Every TPU job goes
through `{{runtime}} compute`.

## Outputs

Write the arm under `{{runs}}/<run_id>/artifacts/tpu-code-arm/<slug>/`.

Return `changed` with `plugin` (the `plugin` tpu-code-stack returned for
your last commit) and `evidence`.

```json
{
  "status": "changed",
  "summary": "One sorted gather in _gather_tokens: moe_gather 61.0 -> 38.4 us per occurrence, outputs bitwise equal.",
  "plugin": {"repo": "<repo>", "base_sha": "<base sha>", "ref": "<checkout.branch>", "sha": "<commit>", "patch": "<patch>"},
  "evidence": {"probe_jobs": ["<job_id>", "<job_id>"], "commits": ["<commit>"], "op_before_ms": 0.061, "op_after_ms": 0.0384, "outputs_match": true, "review": "<runs>/<run_id>/artifacts/review/<slug>-critical.md"},
  "run_id": "<run_id>"
}
```

Return `refuted` with `evidence` when the outputs do not match after one
fix, or the op is not faster above the noise floor. Return `blocked` with
`reason` when the change needs something outside `vllm_torchtpu/`,
tpu-code-stack refuses the commit, no slot can be claimed, or the review
did not pass. Return
`failed` when infrastructure fails past the relaunch limit, or a probe's
applier exits 4: `failure_reason: base mismatch: …`, never `refuted`.
