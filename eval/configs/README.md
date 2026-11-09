# Evaluation configs

A config names the **system under test** and nothing else: which agent harness
runs, what of the package is installed into it, and where the work executes.
It does not name a dataset. No config here has a `tasks:` or `datasets:` key.

The dataset is supplied per run:

```bash
python3 eval/run.py --config <sut> --dataset <dataset>
```

A dataset is run under one config, so a whole dataset is one job. The two axes
are independent, which is what makes an A/B possible: hold the dataset fixed
and vary the SUT.

## The SUTs

Each harness directory holds the same eight:

| SUT (`--config`) | what runs |
| --- | --- |
| `claude-code/agent-suite.yaml` | Claude Code with the whole package, on `tpu-nexus-cluster` |
| `claude-code/agent-suite-shared-capacity.yaml` | the same, on cloud-tpu-shared-capacity |
| `claude-code/agent-suite-remote.yaml` | `agent-suite.yaml`, with the agent in a GKE pod; what CI's smoke check and nightly eval run, on the runner's GKE metadata server (`GKE_AUTH_ACCOUNT=metadata`) |
| `claude-code/agent-suite-shared-capacity-remote.yaml` | `agent-suite-shared-capacity.yaml`, with the agent in a GKE pod |
| `claude-code/baseline.yaml` | Claude Code with nothing installed |
| `claude-code/baseline-remote.yaml` | `baseline.yaml`, with the agent in a GKE pod |
| `claude-code/baseline-shared-capacity-remote.yaml` | `agent-suite-shared-capacity-remote.yaml`, with nothing installed |
| `agy/agent-suite.yaml` | Antigravity with the whole package, on `tpu-nexus-cluster` |
| `agy/agent-suite-shared-capacity.yaml` | the same, on cloud-tpu-shared-capacity |
| `agy/agent-suite-remote.yaml` | `agent-suite.yaml`, with the agent in a GKE pod; what CI's smoke check and nightly eval run, on the runner's GKE metadata server (`GKE_AUTH_ACCOUNT=metadata`) |
| `agy/agent-suite-shared-capacity-remote.yaml` | `agent-suite-shared-capacity.yaml`, with the agent in a GKE pod |
| `agy/baseline.yaml` | Antigravity with nothing installed |
| `agy/baseline-remote.yaml` | `baseline.yaml`, with the agent in a GKE pod |
| `agy/baseline-shared-capacity-remote.yaml` | `agent-suite-shared-capacity-remote.yaml`, with nothing installed |

### User-scope installs

Every SUT above is handed a project-scope install, staged on the host and
mounted into `/workspace`. The `user-scope*.yaml` configs instead run
`installer.py install --scope user` inside the trial, through the agents in
`eval/agents/user_scope.py`, and start the harness in `/workspace`,
outside `$HOME`. They go with the `user-scope-install` dataset, which scores
whether the harness loaded the managed memory block and can delegate to an
installed agent:

| SUT (`--config`) | what runs |
| --- | --- |
| `claude-code/user-scope.yaml` | Claude Code after a user-scope install, in a pod on `harbor-nexus-cluster`; no TPU cluster |
| `agy/user-scope.yaml` | Antigravity after a user-scope install, in a pod on `harbor-nexus-cluster`; no TPU cluster |
| `codex/user-scope.yaml` | Codex after a user-scope install, in local docker; needs `OPENAI_API_KEY`, so workstation only |
| `codex/user-scope-codex-home.yaml` | `codex/user-scope.yaml`, with `CODEX_HOME` set under `$HOME` for the install and the session |

Scheduled nightly evaluations run the first two (`claude-code/user-scope.yaml`
and `agy/user-scope.yaml`) against `user-scope-install`.

## What gets installed: all, none, or subset

A run installs one of three things:

| mode | installs | chosen by |
| --- | --- | --- |
| **all** | the whole package `PACKAGE_MANIFEST` names: every skill its manifest releases (`tpu-agent` is the entry point) | the config: `agent-suite.yaml` |
| **none** | nothing; the control arm | the config: `baseline.yaml` |
| **subset** | only the skills you name, for debugging | `--skills` on `run.py` or `sandbox.sh` |

**all** and **none** come from the config and nothing else. A config named
`baseline` installs nothing and every other config installs the whole package;
`run.py` and `sandbox.sh` share that rule (`eval/utils.py`
`is_control_arm`). A test keeps each `baseline.yaml` identical to its
`agent-suite.yaml`, and `baseline-remote.yaml` to `agent-suite-remote.yaml`, apart from `job_name`, so the only difference between the two arms
is the package.

### Which package: `PACKAGE_MANIFEST`

Each config sets `PACKAGE_MANIFEST: ${PACKAGE_MANIFEST:-package.release.json}`
in its agent env. Its value is the file name of the manifest in `package/` an
agent-suite arm installs:

| `PACKAGE_MANIFEST` | installs |
| --- | --- |
| `package.release.json` (the default) | tpu-agent-suite |
| `package.experimental.json` | tpu-agent-suite-experimental: the release package plus the vLLM optimization skills (close-gaps, decompose-optimization, workload-microbenchmark, accuracy-check, ...) |

Export it to run any config with another package; nothing else changes:

```bash
PACKAGE_MANIFEST=package.experimental.json python3 eval/run.py -c claude-code/agent-suite -d <dataset>
```

A new package needs only its manifest in `package/`, named
`package.<label>.json`, to be chosen the same way; `<label>` is its short name
in job names. A value that is not a manifest there is refused rather
than read as release (`eval/utils.py` `resolve_manifest`). A run on any
package but the release one says so in its job name
(`cc-agent-suite-experimental`, from the manifest's file name), so dashboards
keep it apart from the release runs. A control arm installs nothing, whatever
the value.

**subset** is a command-line choice, never a config key:

```bash
python3 eval/run.py -c claude-code/agent-suite --skills tpu-agent -d smoke-check
./eval/sandbox.sh -c claude-code/agent-suite --skills tpu-agent
```

The names are checked against what the config's package releases, and each
skill is built whole, with everything it reaches. `--skills` is refused with a
`baseline` config, which has nothing to choose from, and in CI, which always
runs the whole package. A subset's job name says so (`cc-agent-suite-subset-agent`),
so it is never read as a scored agent-suite run.

`agent-suite`, `baseline` and `ci` run on the shared `tpu-nexus-cluster`
(project `tpu-nexus-767207`): a kubeconfig is mounted and work is submitted
directly, with no Kueue queue. `agent-suite-shared-capacity` runs on
`bodaborg-tpu7x-nap` (project `cloud-tpu-shared-capacity`, tpu7x 2x2x1), which
admits work through Kueue (`priority-dev` namespace, `default` queue); models
are still served from `tpu-nexus-767207`.

Every other config runs the agent in local docker. A `-remote` config runs it
in a pod on `harbor-nexus-cluster` (project `tpu-nexus-767207`) instead, so a
long run does not hold up the machine; the TPU work goes where its local
counterpart sends it. `run.py` builds and pushes the task images to
`harbor-nexus-eval` first, and `adapter/sitecustomize.py` uploads the mounts
into the pod. A test keeps each `-remote` config identical to its local one
apart from `job_name`, its longer timeouts and `environment`.
CI's smoke check and nightly eval run the `ci` configs, which are the
`agent-suite` configs under a fixed name. The
Hugging Face variables (`HF_TOKEN`, `HF_MODEL_URL`) pass through for the
onboarding datasets and are empty unless exported.

## Datasets

Any dataset runs under either arm; the `baseline` arm measures the package's
absence. A dataset that needs hardware other than the shared cluster's
overrides the cluster through the environment (below).

The `xprof-*` tasks are the routing examples: each is a plainly worded
profiling question ("summarise this profile", "break down HBM") that a user
would hand to `tpu-agent` and that `tpu-agent` should hand to
`tpu-diagnostics`. They touch no hardware — the profiles they read are baked into
the image from openxla/xprof's public demo set, so every graded row is checked
against what the xprof CLI itself reports. The reward is for getting the tools'
answer; whether the run routed through the analyst is reported beside it rather
than scored.

## Running

```bash
# A/B the tpu-hello-world package: same dataset, two arms
python3 eval/run.py --config claude-code/agent-suite    --dataset tpu-hello-world
python3 eval/run.py --config claude-code/baseline --dataset tpu-hello-world

# Two harnesses over one dataset
python3 eval/run.py --config claude-code/agent-suite agy/agent-suite --dataset smoke-check

# One task, to debug one failure without running the suite
python3 eval/run.py --config claude-code/agent-suite --dataset smoke-check/preflight-check
```

Omitting `--dataset` is an error. Harbor would otherwise accept a config with
no tasks, run nothing, and report success over an empty set. Unrecognised
flags pass through to Harbor, so `-n 3`, `--no-upload` and `--print-config`
work as usual.

## Cluster configuration

The cluster settings read from the environment, with the shared cluster as the
default in each config's `env` block:

```bash
GKE_PROJECT=... GKE_CLUSTER=... GKE_LOCATION=... \
GKE_TPU_ACCELERATOR=... GKE_TPU_TOPOLOGY=... \
python3 eval/run.py --config claude-code/agent-suite --dataset <dataset>
```

## Staging

- Tasks pin their base image in `task.toml`.
- The whole package is compiled on the host and staged, cached by commit,
  harness and compute answers. A control arm stages nothing.
- run.py bind-mounts the staged package (`mount_advisor.py`); the config
  mounts the ADC credentials and the kubeconfig run.py mints.
- No per-run `docker build` is needed to bake packages into images.
