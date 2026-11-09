# close-gaps Optimization of an Onboarded Server (`nightly-close-gaps`)

This dataset tests how well an agent optimizes a vLLM server on TPU. The
agent-suite arm uses the `close-gaps` skill, which measures where a served
model loses time, ranks those gaps, and works through them one by one. The
control arm, which installs no skills, optimizes on its own.

The `tpu-inference-*-vllm` tasks in
[`tpu-inference-vllm`](../tpu-inference-vllm/) ask the agent to
onboard the model and then optimize it. The tasks in this dataset skip the
onboarding. Each task image already contains a finished onboarding in
`environment/onboarded/`, which its own README describes. As a result, every
trial starts from the same baseline and spends its time on the optimization.

## Tasks

| Task | Model | Params | Gated |
| :--- | :--- | :--- | :--- |
| [`close-gaps-gemma-4-31b-it`](./close-gaps-gemma-4-31b-it/) | `google/gemma-4-31B-it` | 31B dense | no |

## Scoring

The agent writes `/logs/artifacts/benchmark_results.json` with the 1k/1k
output tokens/s of the baseline and the optimized server, the whole-model
roofline tokens/s it computed (with roofline-analysis when it is installed),
and its final speedup. `tests/speedup/calculate_speedup.py` reads the file,
and `test.sh` uses the same reader for every reported number.

```text
reward = (speedup - 1) / agent_hours x (1 - cheating)
```

The reward is the 1k/1k gain per hour of agent wall-clock time. For example,
a 1.5x speedup in 4 hours scores 0.5 / 4 = 0.125. A slowdown scores below 0.
The reward is 0 if the speedup or `agent_hours` is unknown.

- `speedup` is the optimized output tokens/s divided by the baseline's, as
  written in `benchmark_results.json`.
- `cheating` is 1 when the cheating judge finds a violation of one of its
  rules in `tests/cheating/judge.toml`. It sets the reward to 0. The rules are:
  a number in `benchmark_results.json` differs from the trajectory; a number
  presented as measured was calculated instead; or the final accuracy check
  was missing, failed or manipulated, or the final speedup does not come from
  full benchmark runs.

`execution_report` is reported in `reward.json`, but does not change the
reward.

### Campaign metrics

`test.sh` also writes four campaign metrics to `reward.json`. They are
reported only.

| Metric | Columns | Source |
| :--- | :--- | :--- |
| Roofline saturation | `roofline_saturation`, `roofline_tps` | `optimized_tps / roofline_tps`, from `benchmark_results.json` |
| Throughput lift | `speedup`, `baseline_tps`, `optimized_tps` | `optimized_tps / baseline_tps`, from `benchmark_results.json`. Both servers use the same chips, so this is also the per-chip lift. |
| Token economy | `total_tokens`, `prompt_tokens`, `completion_tokens`, `cached_tokens` | `final_metrics` in the agent's trajectory |
| Wall-clock time | `agent_hours` | the first to the last timestamped step of the agent's trajectory |

The environment is described in
[`nightly-onboard-optimize/README.md`](../nightly-onboard-optimize/README.md).

## Refreshing the onboarding

To refresh the onboarding, follow these steps:

1. Replace `environment/onboarded/` with the output of a newer onboarding run.
   Keep the same directory layout.
2. Update the job IDs of the baseline server and its benchmark clients in
   `instruction.md`, and the server's job ID in the judges under `tests/`.

## Arms

An arm is one configuration that the dataset runs, so that its scores can be
compared with the other arms.

close-gaps and the skills it calls ship only in `package.experimental.json`.
For this reason, the agent-suite arm runs with
`PACKAGE_MANIFEST=package.experimental.json`, which tells the agent-suite
configs to install that package. See
[`eval/configs/README.md`](../../configs/README.md).

Both arms run on tpu7x `2x2x1`, the slice the nightly uses.

| Config | Installs | Role |
| :--- | :--- | :--- |
| `claude-code/agent-suite-shared-capacity-remote`, with `PACKAGE_MANIFEST=package.experimental.json` | the experimental package | the agent-suite arm |
| `claude-code/baseline-shared-capacity-remote` | nothing | the control |

The scheduled nightly evaluation runs this dataset with the experimental package
and the `agent-suite-shared-capacity-remote` configs, once for Claude Code and
once for Antigravity.

## Running

```bash
# agent-suite arm
PACKAGE_MANIFEST=package.experimental.json python3 eval/run.py -c claude-code/agent-suite-shared-capacity-remote -t nightly-close-gaps/close-gaps-gemma-4-31b-it

# control arm
python3 eval/run.py -c claude-code/baseline-shared-capacity-remote -t nightly-close-gaps/close-gaps-gemma-4-31b-it
```
