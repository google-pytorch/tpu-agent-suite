# Training Flag Optimization (`flag-optimization`)

Make a MaxText pre-training job on a Cloud TPU slice faster without changing
its math. Each task fixes the model, batch, precision, optimizer and data,
and lists the performance knobs the agent may change: the device mesh,
rematerialization, MaxText framework flags, kernel hyperparameters and, where
allowed, XLA / libtpu flags. The deliverable is one file of knob values,
`/workspace/output/best_config.json`; the verifier runs it on the same slice
and rewards the throughput improvement over the baseline.

## Tasks

| Task | Model | Slice | Knobs | Budget |
| :--- | :--- | :--- | :--- | :--- |
| `flagopt-gemma4-31b-v7x-2x2x2-sharding` | Gemma 4 31B (dense) | TPU7x `2x2x2`, 8 chips | sharding (3) | 6 h |
| `flagopt-gemma4-31b-v7x-2x2x2-sharding-remat` | Gemma 4 31B (dense) | TPU7x `2x2x2`, 8 chips | sharding, remat (15) | 6 h |
| `flagopt-gemma4-31b-v7x-2x2x2-full-stack` | Gemma 4 31B (dense) | TPU7x `2x2x2`, 8 chips | sharding, remat, framework, kernel (35) + XLA | 6 h |
| `flagopt-qwen3-5-397b-a17b-v7x-4x4x4-all` | Qwen 3.5 397B-A17B (MoE) | TPU7x `4x4x4`, 64 chips | all (60) + XLA | 3 h |
| `flagopt-deepseek-v3-671b-v7x-4x4x4-all` | DeepSeek-V3 671B (MoE) | TPU7x `4x4x4`, 64 chips | all (63) + XLA | 6 h |
| `flagopt-deepseek-v3-671b-v7x-8x8x8-all` | DeepSeek-V3 671B (MoE) | TPU7x `8x8x8`, 512 chips | all (63) + XLA | 3 h |

Every task runs MaxText `maxtext-v0.2.6` from the same pinned
`maxtext-images/tpu_pre_training` image, initialised from the model's Hugging
Face checkpoint, on the shared cluster `bodaborg-tpu7x-nap`. The DeepSeek-V3
`8x8x8` baseline is `DP=2 x FSDP=512`: pure FSDP over 1024 devices does not
compile, since some MLA weight dimensions are not divisible by 1024.

## Files

| Path | What it is |
| :--- | :--- |
| `flag_optimization.yaml` | The suite: the image, the verifier and one case per task (`FLAGOPT_TASK`, timeouts, instruction values) |
| `assets/flagopt/tasks.json` | The task catalog, the source of truth: shared settings (MaxText, measurement, cluster, fixed training settings), topologies, models (facts, fixed settings, applicable knobs, defaults), the 63 knobs, and one short entry per task (model, topology, batch, unlocked knob groups, baseline mesh, time budget) |
| `assets/flagopt/flagopt.py` | The agent's toolkit and the verifier's library (stdlib only) |
| `assets/flagopt/README.md` | The `flagopt` guide the instruction points the agent to |
| `assets/install_flagopt.sh` | Image setup: installs `flagopt` into a root-owned, read-only `/opt/flagopt` |
| `assets/instruction.md`, `instruction_vars.py` | The instruction template, and the derivation of each case's `prompt_vars` from its spec |
| `assets/baselines/`, `record_baseline.py` | Recorded baselines (see [Baselines](#baselines)) and the script that records them |
| `../../verifiers/flag_optimization/` | The rewardkit verifier: `measure.py` and one file per criterion |

`eval/run.py` compiles the suite into six Harbor tasks under
`eval/.datasets/flag_optimization/`. Their `environment/` directories are
identical, so all six run on one image; they differ only in `task.toml` and
the instruction. After editing `tasks.json`, run
`python3 eval/suites/flag_optimization/instruction_vars.py` and update the
YAML's `prompt_vars` from its output;
`eval/tests/test_flag_optimization_suite.py` fails if they disagree.

## How a task works

- **Image.** The base image is the MaxText image the TPU pods run, so
  `flagopt aot` compiles a config on the CPU against the same MaxText, JAX
  and libtpu, and the agent can read the MaxText source under `/deps`.
- **`flagopt`** resolves the task named by `$FLAGOPT_TASK` from `tasks.json`
  (`flagopt spec --json` prints it). It validates a config (only tunable
  knobs, allowed values, a mesh covering every device once, tokens per step
  unchanged, well-formed XLA flags), AOT-compiles it, submits it as a
  multi-host Job under a unique name, waits within bounded queue and run
  timeouts, parses the per-step metrics, deletes the Job, and compares two
  runs.
- **Cluster access** is the trial's standard kube-gate
  ([Trial Isolation](../../README.md#trial-isolation)); `flagopt` and the
  verifier call plain `kubectl`.
- **Slice shape** comes from the task spec, not from the config's
  `GKE_TPU_ACCELERATOR` / `GKE_TPU_TOPOLOGY` (the config's default slice,
  which reaches the agent's environment ahead of the task's).
  `FLAGOPT_TPU_ACCELERATOR` / `FLAGOPT_TPU_TOPOLOGY` override it for local
  testing; namespace, queue and priority follow the config's `GKE_*`.
- **Measurement.** A run trains 26 steps; its step time is the median over
  steps 5..25. A run counts only if it exited 0 and logged every step.

## Scoring

The verifier is a [rewardkit](https://github.com/harbor-framework/harbor/tree/main/examples/tasks/reward-kit-example)
suite. Each criterion is one dimension of `/logs/verifier/reward.json`:

| Dimension | Score | Weight |
| :--- | :--- | :--- |
| `config_valid` | `best_config.json` exists and passes the task spec | 0 |
| `candidate_ran` | the config compiled, fit and finished every step of its run | 0 |
| `accuracy_gate` | per-step loss within 0.1% and gradient norm within 5% of the baseline over the first 15 steps; a missing or non-finite value in either run fails | 0 |
| `throughput_improvement` | `speedup - 1` over the baseline (0.25 for 25% more tokens/s), uncapped; 0 unless the three above hold, and 0 for a config slower than the baseline | 1 |
| `infra_error` | 1 when a zero is the infrastructure's fault (failed baseline, queue timeout, a run cut short by the verifier's deadline) | 0 |

`reward` is the throughput improvement: there is no target or pass
threshold. All criteria read one measurement (`measure.py`): it validates the
config (an invalid one submits nothing), deletes Jobs the agent left behind,
takes the baseline (see below), runs the agent's config in a fresh Job with
the checkpoint weights (once more if its step time is noisy), and computes
the accuracy gate and `speedup` = baseline p50 / candidate p50.
`verification.json` / `.md` hold every number and the per-step gate table.

## Guardrails

- The agent runs as an unprivileged user; `flagopt.py` and `tasks.json` are
  root-owned in `/opt/flagopt`, and the verifier reads those, re-validates
  the config and re-runs it itself, ignoring the agent's own numbers.
  rewardkit runs with `PYTHONNOUSERSITE=1`, so nothing the agent installs in
  its home changes what the verifier imports.
- Fixed settings (model, precision, batch, optimizer, data, seeds) are applied
  after the agent's overrides and rejected by validation; the accuracy gate
  catches a config that changes the math.
- Every run has queue and run timeouts, Jobs have `activeDeadlineSeconds`,
  a TTL and no retries, and each verifier run's timeout is its share of the
  verifier's remaining deadline.
- XLA flags must be `--xla_...`, `--libtpu_...` or `--megascale_...`;
  dump / debug / SDC-checker / CPU / GPU flags are rejected.

## Baselines

> [!NOTE]
> TODO: no task has a recorded baseline yet (recording needs TPU access).
> The tasks run without one: the verifier then measures the baseline live,
> in the same trial, before the agent's config. That costs one more TPU Job
> per trial (two if its step time is noisy), and each trial is compared with
> its own baseline measurement.

With a recording in `assets/baselines/<task_id>.json` (staged into each
task's `tests/baselines/`), the verifier runs only the agent's config and
every trial is scored against the same baseline. Record one per task from a
machine whose kubectl reaches the cluster, and commit it:

```bash
python3 eval/suites/flag_optimization/record_baseline.py <task_id>
```

It runs the baseline config exactly as the verifier runs a candidate, with
the checkpoint weights and with random init, keeping the least noisy of up
to two runs each. Re-record after changing a task's image, measurement or
baseline; the verifier ignores a recording that does not match the spec.

## Running

Run under the shared-capacity configs, whose kubeconfig targets
`bodaborg-tpu7x-nap`: `claude-code/agent-suite-shared-capacity[-remote]` or
`agy/agent-suite-shared-capacity[-remote]` for the agent-suite arm, and
`claude-code/baseline-shared-capacity-remote` or
`agy/baseline-shared-capacity-remote` for the control (it gets the same
`flagopt`, so both arms can score).

```bash
# one task
python3 eval/run.py -c claude-code/agent-suite-shared-capacity-remote \
  -s flag_optimization -t flagopt-gemma4-31b-v7x-2x2x2-sharding

# every task
python3 eval/run.py -c claude-code/agent-suite-shared-capacity-remote \
  -s flag_optimization
```

`eval/run.py` recompiles the suite when the YAML changes; after editing only
an asset or the verifier, delete `eval/.datasets/flag_optimization/` first.
The `4x4x4` and `8x8x8` tasks hold 64 and 512 chips per Job.
