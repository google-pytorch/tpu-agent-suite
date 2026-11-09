# Serving Onboarding + Kernel Optimization (`tpu-inference-vllm`)

End-to-end tasks that require both onboarding a model onto `torch_tpu` + vLLM
and optimizing its serving throughput by profiling with XProf, authoring and
autotuning a custom Pallas kernel, integrating it into vLLM, and re-running
the full serving benchmark sweep.

For standalone onboarding evaluation see
[`nightly-onboard-optimize`](../nightly-onboard-optimize/); for kernel
optimization on an already-onboarded server see
[`nightly-close-gaps`](../nightly-close-gaps/).

## Tasks

| Task | Model | Params | Gated |
| :--- | :--- | :--- | :--- |
| [`tpu-inference-gemma-4-31b-it-vllm`](./tpu-inference-gemma-4-31b-it-vllm/) | `google/gemma-4-31B-it` | 31B dense | no |
| [`tpu-inference-gpt-oss-20b-vllm`](./tpu-inference-gpt-oss-20b-vllm/) | `openai/gpt-oss-20b` | 20.2B / 4B MoE | no |
| [`tpu-inference-qwen3-8-27b-vllm`](./tpu-inference-qwen3-8-27b-vllm/) | `Qwen/Qwen3.8-27B` | 27B dense | no |

## Running

```bash
# one model, agent-suite arm
python3 eval/run.py -c claude-code/agent-suite -t tpu-inference-vllm/tpu-inference-gemma-4-31b-it-vllm

# every task in the tpu-inference-vllm dataset
python3 eval/run.py -c claude-code/agent-suite -d tpu-inference-vllm
```
