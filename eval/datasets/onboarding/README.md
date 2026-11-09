# Serving Onboarding (`onboarding`)

Serve a Hugging Face model on Cloud TPU with `torch_tpu` and vLLM, and prove
it. Where [`hf-onboarding-torchtpu`](../hf-onboarding-torchtpu/) asks for an
analysis and a benchmark script, this asks for a server that answers over
HTTP, measured numbers from that server, and a commit history that shows how
it got there.

## Models

One task per model. The brief, the image and the verifier are identical
across all five; the model is the variable, so "size the topology yourself"
has more than one right answer and a run that hardcodes a slice stops
scoring.

| Task | Model | Params | Gated | Reachable with our key |
| :--- | :--- | :--- | :--- | :--- |
| [`qwen3-32b`](./qwen3-32b/) | `Qwen/Qwen3-32B` | 32B dense | no | yes |
| [`gemma-3-27b-it`](./gemma-3-27b-it/) | `google/gemma-3-27b-it` | 27B dense | yes | yes, terms accepted |
| [`mistral-small-24b`](./mistral-small-24b/) | `mistralai/Mistral-Small-24B-Instruct-2501` | 24B dense | no | yes |
| [`llama-3-3-70b`](./llama-3-3-70b/) | `meta-llama/Llama-3.3-70B-Instruct` | 70B dense | yes | **not yet — awaiting repo-author review** |
| [`gpt-oss-20b`](./gpt-oss-20b/) | `openai/gpt-oss-20b` | 20.2B / 4B MoE | no | yes |

The model catalogue lives in
[`hf-onboarding-torchtpu/README.md`](../hf-onboarding-torchtpu/README.md) --
categories, target slices, recommended TP and gating for every model in scope.
The five here are a subset of it, not a second list.

Gating is per-account and manual for the two marked: the token in use has
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
- **`HF_TOKEN`** -- required for the gated models, ignored by the rest. The
  gated tasks declare `required_env = ["HF_TOKEN"]` under `[metadata]`. When
  it is unset *or empty* on the host, `run.py` drops those tasks from a
  dataset run (announced on stderr) so the rest still run, and refuses a task
  named alone with `--task`. A missing CI secret arrives as `""`, which `${HF_TOKEN:-}` forwards unchanged, and the
  trial would otherwise end at the first 401 inside the pod.
- The image is built from each task's `environment/Dockerfile`: the
  `vllm-on-host` image (`eval/images/vllm-on-host`, which includes tokamax),
  pinned by digest. Harbor's GKE path never reads `task.toml`'s
  `docker_image` (`run.py:286`), so no image is pinned there.
- The tasks need a real v7 slice. Export the `GKE_*` variables for the
  shared cluster before running; `run.py` does not read a `.env` file.

## Scoring

`tests/test.sh` scores the checkable half of the brief and says which half
that is: artifacts under one directory, the image digest and HF revision
pinned, both required tables with a measured row, raw output retained, an HTTP
response carrying content, and three or more local commits. "Coherent output"
and a log that "reads as the story" are left to a reader.

Two prohibitions are disqualifying gates rather than points -- `torch_xla` or
`tpu_inference`, and committed credentials or weights -- so a run that
produced nothing earns nothing for satisfying them.

## Running

```bash
# one model, agent-suite arm
python3 eval/run.py -c claude-code/agent-suite -t onboarding/qwen3-32b

# the same model, control arm
python3 eval/run.py -c claude-code/baseline -t onboarding/qwen3-32b

# every model in the dataset
python3 eval/run.py -c claude-code/agent-suite -d onboarding
```
