# Serving Onboarding & Optimization (`nightly-onboard-optimize`)

Serve a Hugging Face model on Cloud TPU with `torch_tpu` and vLLM, and prove
it. Where [`hf-onboarding-torchtpu`](../hf-onboarding-torchtpu/) asks for an
analysis and a benchmark script, this dataset asks for a server that answers
over HTTP, measured numbers from that server, and a commit history that shows
how it got there.

## Tasks

### Standalone Onboarding (`onboarding-*`)

One onboarding-only task per model, scored with the Onboarding-Only RewardKit
suite. The verifier is identical across all four, and so are the brief and
image of gemma, mistral and gpt-oss; Qwen3.8-27B, the control for
`onboarding-frontier`, runs that dataset's overlay image and brief; the model
is the variable, so "size the topology yourself" has more than one right answer
and a run that hardcodes a slice stops scoring.

| Task | Model | Params | Gated | Reachable with our key |
| :--- | :--- | :--- | :--- | :--- |
| [`onboarding-qwen3-8-27b`](./onboarding-qwen3-8-27b/) | `Qwen/Qwen3.8-27B` | 27B dense | no | yes |
| [`onboarding-gemma-4-31b-it`](./onboarding-gemma-4-31b-it/) | `google/gemma-4-31B-it` | 31.3B dense | no | yes |
| [`onboarding-mistral-small-24b`](./onboarding-mistral-small-24b/) | `mistralai/Mistral-Small-24B-Instruct-2501` | 24B dense | no | yes |
| [`onboarding-gpt-oss-20b`](./onboarding-gpt-oss-20b/) | `openai/gpt-oss-20b` | 20.2B / 4B MoE | no | yes |

### Two-Phase Onboarding + Throughput Hill-Climbing (`onboard-hillclimb-*`)

End-to-end two-phase tasks pairing standalone `torch_tpu` + vLLM bring-up
(Phase 1, frozen under `onboarding/`) with hypothesis-driven throughput
hill-climbing on the `8k/1k` serving workload to maximize per-chip output
throughput (`output_tps_per_chip = output_tps / num_chips`), followed by a
post-hill-climb `.xplane.pb` profile and residual bottlenecks report
(`xprof_bottlenecks_and_gaps.md`) (Phase 2, saved under `hillclimb/`). Prompts
are written neutrally so both the `agent-suite` arm and the `baseline` control
arm run the exact same specification.

| Task | Model | Params | Gated | Reachable with our key |
| :--- | :--- | :--- | :--- | :--- |
| [`onboard-hillclimb-glm-5-2-fp8`](./onboard-hillclimb-glm-5-2-fp8/) | `zai-org/GLM-5.2-FP8` | 753B / ~40B FP8 MoE | no | yes |
| [`onboard-hillclimb-gemma-4-26b-a4b-it`](./onboard-hillclimb-gemma-4-26b-a4b-it/) | `google/gemma-4-26B-A4B-it` | 26B / 4B MoE | no | yes |
| [`onboard-hillclimb-qwen3-5-27b`](./onboard-hillclimb-qwen3-5-27b/) | `Qwen/Qwen3.5-27B` | 27B dense | no | yes |
| [`onboard-hillclimb-qwen3-6-27b`](./onboard-hillclimb-qwen3-6-27b/) | `Qwen/Qwen3.6-27B` | 27B dense | no | yes |
| [`onboard-hillclimb-glm-5-3-flash`](./onboard-hillclimb-glm-5-3-flash/) | `zai-org/GLM-5.3-Flash` | 321B / 18B FP8 MoE | no | yes |

Combined onboarding and kernel-optimization tasks (`tpu-inference-*-vllm`) live
in [`tpu-inference-vllm`](../tpu-inference-vllm/), while kernel optimization on
an already-onboarded server lives in
[`nightly-close-gaps`](../nightly-close-gaps/).

The model catalogue lives in
[`hf-onboarding-torchtpu/README.md`](../hf-onboarding-torchtpu/README.md) --
categories, target slices, recommended TP and gating for every model in scope.

Gating is per-account and manual for any marked model: the token in use has
`canReadGatedRepos`, so a 403 there means the account has not accepted that
model's terms, not that the key is wrong. Accept at the model page, then check
`https://huggingface.co/settings/gated-repos`. The briefs for gated models say
to report a 403 and stop rather than substitute another model -- a silent
substitution scores as a success on a model nobody asked for, and the verifier
cannot tell the difference.

## Arms

| Config | installs | |
| :--- | :--- | :--- |
| `claude-code/agent-suite` | the whole package | the agent-suite arm |
| `claude-code/baseline` | nothing | the control |

Every config installs the whole package; one whose name contains `baseline`
is a control arm and installs nothing.

## Environment

- **`HF_MODEL_URL`** -- set per task in its `task.toml`; the agent reads it
  from the environment rather than from the prompt.
- **`HF_TOKEN`** -- required for gated models, ignored by the rest. Any gated
  task declares `required_env = ["HF_TOKEN"]` under `[metadata]`. When it is
  unset *or empty* on the host, `run.py` drops those tasks from a dataset run
  (announced on stderr) so the rest still run, and refuses a task named alone
  with `--task`. A missing CI secret arrives as `""`, which `${HF_TOKEN:-}`
  forwards unchanged, and the trial would otherwise end at the first 401 inside
  the pod.
- Both `onboarding-*` and `onboard-hillclimb-*` tasks build from `nexus-host`
  (except `onboarding-qwen3-8-27b`, which builds from `nexus-tpu:latest` as the
  `onboarding-frontier` control); workloads run inside `$TASK_DOCKER` on TPU.
  Harbor's GKE path never reads `task.toml`'s `docker_image` (`run.py:286`), so
  no image is pinned there.
- The tasks need a real v6e or v7 slice. Export the `GKE_*` variables for the
  shared cluster before running; `run.py` does not read a `.env` file.

## Scoring

- For `onboarding-*`, `tests/test.sh` runs the Onboarding-Only RewardKit suite
  (`onboarding_success`, `efficiency`, and `cheating`, plus diagnostic
  `regime_sweep`, `execution_report`, `invoked_subagent`, and
  `schema_compliance`).
- For `onboard-hillclimb-*`, `tests/test.sh` and `tests/reward.toml` run the
  canonical Onboard + Hill-Climb RewardKit suite with multiplicative gating:
  - `reward = (0.3 * onboarding_success + 0.3 * hillclimb_success + 0.3 * speedup_score + 0.1 * efficiency) * (1 - cheating)`
  - `speedup` is computed deterministically on verified per-chip output
    throughput (`output_tps_per_chip = output_tps / num_chips`) across the
    `8k/1k` concurrency sweep from `benchmark_results.json` cross-checked
    against backing `raw_output_path` files (`tests/speedup/calculate_speedup.py`).
  - `hillclimb_success` requires both `onboarding_success == 1` and verified
    per-chip speedup `>= 1.01`; `speedup_score` and `efficiency` are gated on
    both `onboarding_success` and `hillclimb_success`, and `cheating == 1`
    zeroes the final reward.
  - Diagnostic criteria (`execution_report`, `invoked_subagent`,
    `schema_compliance`, and `post_hillclimb_xprof` in `reward-details.json`)
    carry weight `0.0`.

## Running

```bash
# one standalone onboarding task, agent-suite arm
python3 eval/run.py -c claude-code/agent-suite -t nightly-onboard-optimize/onboarding-qwen3-8-27b

# one two-phase onboarding + hill-climb task, agent-suite arm
python3 eval/run.py -c claude-code/agent-suite -t nightly-onboard-optimize/onboard-hillclimb-gemma-4-26b-a4b-it

# the same model, control arm
python3 eval/run.py -c claude-code/baseline -t nightly-onboard-optimize/onboarding-qwen3-8-27b

# every task in the nightly-onboard-optimize dataset
python3 eval/run.py -c claude-code/agent-suite -d nightly-onboard-optimize
```

### Provenance pages between trials

Each trial caches tpu-knowledge-base pages inside its own container, under
`/root/.tpu-agent-suite`, and they go when the container does. A trial is
scored on what it retrieved, not on what a sibling left behind.
`TPU_AGENT_SUITE_HOME` pins that path rather than moving it: every task
collects its artifacts from `/root/.tpu-agent-suite/runs` and nothing sets
`HOME`. Keep it equal to `~` for the container's user -- compiled hook
commands hardcode the `~/.tpu-agent-suite` spelling and do not honour it.

There used to be a shared host directory here, so that one trial's fetch
spared the next: every onboarding reads the same few vllm pages, and
docs.vllm.ai refused the third request in quick succession from one IP.
The lookup no longer fetches anything -- a url is summarised by the harness
CLI, from the CLI's own backend, not from this VM -- so there is no longer
a rate limit to share around, and nothing to configure.
