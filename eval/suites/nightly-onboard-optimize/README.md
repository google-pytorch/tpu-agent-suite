# Serving Onboarding (`nightly-onboard-optimize`)

Serve a Hugging Face model on Cloud TPU with `torch_tpu` and vLLM, and prove
it. Where [`hf-onboarding-torchtpu`](../../datasets/hf-onboarding-torchtpu/)
asks for an analysis and a benchmark script, this suite asks for a server that
answers over HTTP, measured numbers from that server, and a commit history that
shows how it got there.

## Tasks

One case per model, compiled from [`suite.yaml`](./suite.yaml). The first three
cases share [`onboarding_template.md`](./assets/prompts/onboarding_template.md)
on `nexus-host`; `onboarding-qwen3-8-27b` uses its dedicated frontier-parity
brief [`onboarding-qwen3-8-27b.md`](./assets/prompts/onboarding-qwen3-8-27b.md)
and builds from [`assets/setup/Dockerfile.nexus-tpu-host`](./assets/setup/Dockerfile.nexus-tpu-host).

| Case | Model | Params | Gated |
| :--- | :--- | :--- | :--- |
| `onboarding-gemma-4-31b-it` | `google/gemma-4-31B-it` | 31.3B dense | no |
| `onboarding-gpt-oss-20b` | `openai/gpt-oss-20b` | 20.2B / 4B MoE | no |
| `onboarding-mistral-small-24b` | `mistralai/Mistral-Small-24B-Instruct-2501` | 24B dense | no |
| `onboarding-qwen3-8-27b` | `Qwen/Qwen3.8-27B` | 27B dense | no |

## Arms

| Config | Installs | Role |
| :--- | :--- | :--- |
| `claude-code/agent-suite` | Full package | Agent-suite arm |
| `claude-code/baseline` | Nothing | Control arm |

## Environment

- **`HF_MODEL_URL`** -- set per case in `environment.env_vars` (`task.toml`).
- **`HF_TOKEN`** -- required only for gated models when `required_env` is set.
- Export `GKE_*` variables for the shared TPU cluster before running.

## Scoring

Compiled tasks execute a two-phase verification pipeline in `tests/test.sh`:

1. `rewardkit` evaluates deterministic trajectory/artifact checks and LLM
   judges (`onboarding_success`, `efficiency`, `cheating`, `regime_sweep`,
   `execution_report`, `invoked_subagent`, and `schema_compliance`).
2. `aggregate_reward.py` applies multiplicative success and anti-cheating gates
   and writes the finalized `reward.json` payload.

## Running

```bash
# Validate suite compilation and schemas
./eval/verify.sh --suite nightly-onboard-optimize

# Single case, agent-suite arm
python3 eval/run.py -c claude-code/agent-suite --suite nightly-onboard-optimize --task onboarding-qwen3-8-27b

# Full suite
python3 eval/run.py -c claude-code/agent-suite --suite nightly-onboard-optimize
```
