# Serving Onboarding & Optimization (`nightly-onboard-optimize`)

Serve a Hugging Face model on Cloud TPU with `torch_tpu` and vLLM, and prove
it. Where [`hf-onboarding-torchtpu`](../../datasets/hf-onboarding-torchtpu/)
asks for an analysis and a benchmark script, this suite asks for a server that
answers over HTTP, measured numbers from that server, and a commit history that
shows how it got there.

The suite is split into two YAML files in this directory — both declaring
`name: nightly-onboard-optimize` so `--suite nightly-onboard-optimize` compiles
all 9 cases together:

- [`onboard.yaml`](./onboard.yaml): 4 standalone onboarding-only tasks (`onboard`
  scoring).
- [`onboard_hillclimb.yaml`](./onboard_hillclimb.yaml): 5 two-phase onboarding +
  per-chip throughput hill-climbing tasks (`onboard_hillclimb` scoring).

## Tasks

### Standalone Onboarding ([`onboard.yaml`](./onboard.yaml))

The first three cases share
[`onboarding_template.md`](./assets/prompts/onboarding_template.md) on
`nexus-host`; `onboarding-qwen3-8-27b` uses its dedicated frontier-parity brief
[`onboarding-qwen3-8-27b.md`](./assets/prompts/onboarding-qwen3-8-27b.md) and
builds from
[`assets/setup/Dockerfile.nexus-tpu-host`](./assets/setup/Dockerfile.nexus-tpu-host).

| Case | Model | Params | Default Slices |
| :--- | :--- | :--- | :--- |
| `onboarding-gemma-4-31b-it` | `google/gemma-4-31B-it` | 31.3B dense | `tpu7x` (`2x2x1`), `tpu-v6e-slice` (`2x4`) |
| `onboarding-gpt-oss-20b` | `openai/gpt-oss-20b` | 20.2B / 4B MoE | `tpu7x` (`2x2x1`), `tpu-v6e-slice` (`2x4`) |
| `onboarding-mistral-small-24b` | `mistralai/Mistral-Small-24B-Instruct-2501` | 24B dense | `tpu7x` (`2x2x1`), `tpu-v6e-slice` (`2x4`) |
| `onboarding-qwen3-8-27b` | `Qwen/Qwen3.8-27B` | 27B dense | `tpu7x` (`2x2x1`), `tpu-v6e-slice` (`2x4`) |

### Two-Phase Onboarding + Hill-Climbing ([`onboard_hillclimb.yaml`](./onboard_hillclimb.yaml))

All 5 cases share
[`onboard_hillclimb_template.md`](./assets/prompts/onboard_hillclimb_template.md)
on `nexus-host`. Phase 1 brings up and freezes the unmodified vLLM baseline;
Phase 2 iteratively optimizes **`output_tps_per_chip = output_tps / num_chips`**
on the `8k/1k` concurrency sweep and captures a post-hill-climb XProf trace.

| Case | Model | Params | Nightly Routing |
| :--- | :--- | :--- | :--- |
| `onboard-hillclimb-gemma-4-26b-a4b-it` | `google/gemma-4-26B-A4B-it` | 26B / 4B MoE | Daily (`tpu7x` `2x2x1`) |
| `onboard-hillclimb-qwen3-5-27b` | `Qwen/Qwen3.5-27B` | 27B dense | Daily (`tpu7x` `2x2x1`) |
| `onboard-hillclimb-qwen3-6-27b` | `Qwen/Qwen3.6-27B` | 27B dense | Daily (`tpu7x` `2x2x1`) |
| `onboard-hillclimb-glm-5-2-fp8` | `zai-org/GLM-5.2-FP8` | 753B / 40B MoE FP8 | Explicit `--task` / `--include-all-tasks` (`tpu7x` `2x2x2`) |
| `onboard-hillclimb-glm-5-3-flash` | `zai-org/GLM-5.3-Flash` | 321B / 18B MoE FP8 | Explicit `--task` / `--include-all-tasks` (`tpu7x` `2x2x2`) |

## Arms

| Config | Installs | Role |
| :--- | :--- | :--- |
| `claude-code/agent-suite` | Full package | Agent-suite arm |
| `claude-code/baseline` | Nothing | Control arm |

## Environment

- **`HF_MODEL_URL`** -- set per case in `environment.env_vars` (and `verifier_env`
  for hill-climb cases) in `task.toml`.
- **`HF_TOKEN`** -- required only for gated models when `required_env` is set.
- Export `GKE_*` variables for the shared TPU cluster before running.

## Scoring

Compiled tasks execute a two-phase verification pipeline in `tests/test.sh`:

- **Standalone onboarding (`onboard.yaml`):**
  1. `rewardkit` evaluates `onboarding_success`, `efficiency`, `cheating`,
     `regime_sweep`, `execution_report`, `invoked_subagent`, and
     `schema_compliance`.
  2. `eval/verifiers/onboard/aggregate_reward.py` applies multiplicative
     onboarding + anti-cheating gates and writes `reward.json`.
- **Two-phase hill-climb (`onboard_hillclimb.yaml`):**
  1. `rewardkit` evaluates `onboarding_success`, `speedup`
     (`calculate_speedup.py` with `primary_regime="8k/1k"`, `per_chip=true`,
     `require_manifest=true`), `efficiency`, `cheating`, `execution_report`,
     `invoked_subagent`, and `schema_compliance`.
  2. `eval/verifiers/onboard_hillclimb/aggregate_reward.py` applies
     multiplicative onboarding, hill-climb (`>= 1.01x`), log-speedup, and
     anti-cheating gates, records diagnostic `post_hillclimb_xprof` status in
     `reward-details.json`, and writes `reward.json`.

## Running

```bash
# Validate suite compilation and schemas
./eval/verify.sh --suite nightly-onboard-optimize

# Single case, agent-suite arm
python3 eval/run.py -c claude-code/agent-suite --suite nightly-onboard-optimize --task onboard-hillclimb-gemma-4-26b-a4b-it

# Full suite (all 9 cases)
python3 eval/run.py -c claude-code/agent-suite --suite nightly-onboard-optimize
```
