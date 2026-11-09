# Eval Tooling

This directory contains tooling for running evaluations on TPU agents using the Harbor framework.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) (recommended for Python environment management)
- **Docker**: Required for running evaluations in isolated containers. Ensure the Docker daemon is running.
- **API Keys**: Required for LLM calls during evaluation.

## Setup

We use `uv` to manage the virtual environment and dependencies.

**Run the setup script** to create a virtual environment and install dependencies:

```bash
./eval/setup.sh
```

This will create a `.venv` directory under `eval` and install the required packages (including `harbor`).

**Activate the virtual environment**:

```bash
source eval/.venv/bin/activate
```

*Note: Always run the verification and evaluation scripts with this virtual environment activated.*

## Verification

(Optional) Before running evaluations, verify your environment configuration and validate the schemas of your configs and datasets.

Run the verification script from the repository root:

```bash
# Verify a specific package
./eval/verify.sh <package_name>
```

The script performs the following checks:
- Verifies that `harbor` and `python` binaries are accessible.
- Checks if the Docker daemon is running.
- Checks if the required api key environment variables are set.
- Validates every YAML config under `eval/configs/`.
- Validates every `task.toml` in the eval tree. Pass a dataset name
  (`./eval/verify.sh smoke-check`) to scope the task check to one suite;
  an unknown name exits non-zero.

## Running Evaluation Tasks

To run evaluation tasks, use `run.py`. It automatically ensures that required packages under `package/` are compiled and staged, mints ephemeral single-context GKE cluster credentials when needed, resolves environment variables, sets `PYTHONPATH`, and drives Harbor.

**Set your API Key**:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

**Run the evaluation**:

A config names the **system under test**; the dataset is given per run. Neither
is optional, and no config carries a `tasks:` or `datasets:` key.

```bash
# A config by name, against a dataset by name
./eval/run.py --config claude-code/agent-suite --dataset tpu-hello-world

# Positional config name works too
./eval/run.py claude-code/agent-suite --dataset tpu-hello-world

# Several configs over the same dataset -- this is how an A/B is run
./eval/run.py --config claude-code/agent-suite claude-code/baseline --dataset tpu-hello-world

# A config file outside configs/, and a dataset by path
./eval/run.py --config-file /tmp/custom.yaml --dataset eval/datasets/tpu-hello-world

# One task instead of a whole dataset
./eval/run.py --config claude-code/agent-suite --task torch-benchmark-coder

# Several tasks as one Harbor job (one Hub link, one pass/fail)
./eval/run.py --config claude-code/agent-suite --task nightly-functional/smart-hf-search nightly-functional/remote-tpu-runner

# Anything unrecognised is passed straight to Harbor
./eval/run.py --config claude-code/agent-suite --dataset tpu-hello-world --verbose
```

Configs resolve against `eval/configs/`, datasets against
`eval/datasets/` or a path. `eval/configs/README.md` maps the
`{scope}_{agent}_{launch}_{v}` naming onto what each config actually stages.

## Running One Task Without Harbor

To iterate on a single task, `sandbox.sh` prepares a container you drive by
hand. A config is required: it decides the packages, the cluster and the arm.

```bash
./eval/sandbox.sh -c claude-code/agent-suite
./eval/sandbox.sh -c claude-code/agent-suite -t tpu-hello-world/baseline
./eval/sandbox.sh --baseline                      # the control arm
./eval/sandbox.sh -c claude-code/agent-suite --sleep   # no shell
```

The container always runs as a daemon and the shell you get is a `docker exec`
into it, inside a **host** tmux session named `tpu-sandcastle`. Ctrl-b d
detaches and whatever is running keeps running; `tmux attach -t tpu-sandcastle`,
or re-running `sandbox.sh`, gets you back. That is what a long install or a
multi-hour agent run needs, and it means a dropped ssh session costs you the
view rather than the work. Re-running `sandbox.sh` removes the previous
container first, so export anything you want to keep before you do.

Nothing is installed for you. From the shell:

| Command | What it does |
| :--- | :--- |
| `install_claude` | Install the package for Claude Code, as a user does; the first session sets up compute. |
| `install_agy` | The same for Antigravity, then mirrors into the `~/.gemini` roots agy scans. |
| `install_codex` | The same for Codex. |
| `run_claude` / `run_agy` / `run_codex` | Start the agent with the sandbox environment loaded. |

All three harnesses are compiled on the host in Step 7 and copied in, because
the container has no `compiler.py` to produce another one later.

The task chooses the image, through the `docker_image` its `task.toml` pins.
The repository is **not** mounted: `/workspace` is empty and the agent's own. A
control arm that could read `package` is not a control arm — an agent
asked to do TPU work simply grepped it and followed the instructions. What
crosses into the container is the generated env, the task prompt and the
compiled dists, all by `docker cp` rather than a bind mount.

The config chooses between the whole package (`agent-suite`) and nothing
(`baseline`); `--skills` installs a subset for debugging (see
[configs/README.md](configs/README.md#what-gets-installed-all-none-or-subset)).
The config supplies the environment (`agents[].env`), which the sandbox writes to
`/root/sandbox/env.txt` for the container to source.

The sandbox mints no cluster credentials: it mounts your own `~/.kube/config`
if you have one, and otherwise none. A config that names a cluster is a `run.py`
run, which does mint.

## Trial Isolation

### Why it exists

An eval run starts several **trials** at once. A trial is one agent working on
one task in its own container. To use a TPU, each agent submits Kubernetes
Jobs to a shared GKE cluster. All trials use the same cluster, the same
namespace (`default`) and the same credentials. Without isolation, an agent in
one trial could list, read the logs of, or delete the Jobs of another trial
running at the same time, and change that trial's result.

### How it works

Each trial container is set up so that the agent can reach the cluster only
through a wrapper around `kubectl`, and the wrapper only lets it touch its own
trial's resources.

1. **The agent runs as an ordinary user.** The agent process runs as the
   Linux user `agent` (uid 1000), not as `root`, so Linux file permissions
   apply to it. Its home folder is `/root` (owned by `agent`), so task and
   config paths under `/root/...` work as before.
2. **The agent cannot read the cluster credentials.** The kubeconfig and the
   Google Cloud credentials used for the cluster are in `/etc/kube-gate/`.
   Only a separate system user, `kube-gate`, can read them. The real
   `kubectl` binary is moved to `/usr/libexec/kubectl-real`, and only
   `kube-gate` can run it.
3. **`kubectl` goes through a wrapper.** `/usr/local/bin/kubectl` is a
   two-line script. It uses `sudo` to run
   [`kubectl_wrapper.py`](kubectl_wrapper.py) (installed as
   `/usr/libexec/kubectl-wrapper`) as `kube-gate`. The sudo rule allows that
   one command and nothing else. The agent types `kubectl` commands as usual.

Every trial gets an ID (Harbor's pod or session name). The wrapper uses it as
a Kubernetes label, `eval-trial-id=<trial ID>`, and handles each command like
this:

| Command | What the wrapper does |
| :--- | :--- |
| `apply`, `create`, `run` | Adds the `eval-trial-id` label to every resource, including the pods a Job or CronJob starts. Refuses if the name already belongs to another trial. |
| `get`, `delete`, `wait`, `logs`, `top` without a name (e.g. `kubectl get pods`) | Adds `-l eval-trial-id=<trial ID>`, so only this trial's resources are listed or affected. |
| A command that names a resource (e.g. `kubectl logs job/foo`, `exec`, `attach`, `cp`, `port-forward`) | Looks up the resource's label first. If it belongs to another trial, answers `NotFound` and does not run the command. |
| Read-only cluster information (`version`, `config`, `get nodes`, `get namespaces`, Kueue queues, ...) | Runs it unchanged. |
| Anything else (`edit`, `replace`, `proxy`, `drain`, ...) | Refuses with `Forbidden`. |

The setup runs as `root` when each trial starts:

| File | Role |
| :--- | :--- |
| [`adapter/kube_gate_setup.sh`](adapter/kube_gate_setup.sh) | Creates the users, installs the wrapper and the sudo rule, moves the credentials, writes the trial ID. |
| [`adapter/sitecustomize.py`](adapter/sitecustomize.py) | Runs `kube_gate_setup.sh` in every Harbor trial (`run.py`). |
| [`sandbox.sh`](sandbox.sh) | Does the same setup for the local sandbox. |
| [`kubectl_wrapper.py`](kubectl_wrapper.py) | The wrapper. Tests: [`tests/test_kubectl_wrapper.py`](tests/test_kubectl_wrapper.py). |

### What you will notice in a trial

- `kubectl get pods` shows only this trial's pods. Another trial's Job or pod
  is reported as `NotFound`, even when it exists.
- A name another trial already uses cannot be reused: `kubectl create` or
  `apply` fails with `AlreadyExists ... belongs to another trial`. The
  agent-suite job driver detects this and adds a random suffix to the name.
- `/usr/local/bin` and the system Python (`/usr/local/lib/python3.12`) belong
  to `root`. The wrapper is a Python script that runs as `kube-gate`, so if the
  agent could change Python or its libraries, it could run its own code as
  `kube-gate` and read the credentials.
- `pip install` still works. pip prints `Defaulting to user installation
  because normal site-packages is not writeable` and installs into
  `~/.local`. Commands it installs go to `~/.local/bin`, which is on `PATH`.
- The agent cannot use `sudo` for anything except `kubectl`.

## Adding Datasets and Configs

- Tasks live at `eval/datasets/<dataset>/<task>/` with a `task.toml`,
  `instruction.md`, `environment/Dockerfile`, and `tests/`. Configs live flat
  in `eval/configs/<config>.yaml` and must not name a dataset.
- Dataset and task directory names must match `^[a-z0-9-]+$`; CI rejects
  anything else. Names starting with `_` or `.` are skipped, so `_template` is
  allowed.
- Run `./eval/verify.sh <package>` to validate the config YAML and every
  `task.toml` before running.
- Tasks under `eval/datasets/nightly-functional/` run in the scheduled nightly
  evaluation under `claude-code/agent-suite-remote` and
  `agy/agent-suite-remote`, so any functional nightly task must pass under both
  configs. Pull requests run the `smoke-check` dataset automatically when
  package, framework, or evaluation files change.

## Interactive Claude Code Sandbox

You can launch an interactive Claude Code session inside an isolated container with cluster credentials and TPU tooling mounted.

1. **Configure Environment**:
   Fill up `eval/env_example.txt` with your cluster, project, and TPU details, and rename (or copy) it to `eval/env.txt` for a working environment inside the sandbox:

   ```bash
   cp eval/env_example.txt eval/env.txt
   # Fill up eval/env.txt with your configuration
   ```

2. **Launch Sandbox**:

   The arm comes from the config, never from a flag, so the sandbox cannot
   disagree with the config about which arm it is. A control arm is a config
   whose name contains `baseline`.

   ```bash
   # Control arm baseline (no package tools)
   ./eval/sandbox.sh --baseline

   # With the package tools available to install
   ./eval/sandbox.sh -c claude-code/agent-suite
   ```

3. **Install and run**, from the shell it drops you in:

   ```bash
   install_claude   # or install_agy, install_codex
   run_claude       # or run_agy, run_codex
   ```
