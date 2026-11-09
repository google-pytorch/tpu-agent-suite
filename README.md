# TPU agent suite

Autonomous TPU development and onboarding agents for Google Cloud TPUs.

This repository builds and distributes **TPU agent suite packages**: declarative bundles of multi-agent skills, routing protocols, and tools that compile and install natively across major agent harnesses—including **Claude Code** and **Antigravity**, with **Codex** and **Cursor** coming soon.

---

## Architecture

| Component | Path | Description |
| :--- | :--- | :--- |
| **Framework** | [`framework/`](framework/) | Compiler, installer, runtime and compute access layer. Lets us develop agents that run on multiple harnesses. |
| **Package** | [`package/`](package/) | The TPU agent suite, released as `tpu-agent-suite`. |
| **Evaluations** | [`eval/`](eval/) | Harbor suites and configs that score the package. |

Users talk to the **`tpu-agent`** skill. It routes each request to a
specialist: onboarding, profiling, a hardware check, cited
TPU facts, and kernels. See the [design docs](docs/design/README.md) for how the framework
and the suite are built.

---

## Quickstart

### 1. Prerequisites

- Python 3.12+ and the host-side packages the agents run with, XProf among
  them, in a virtualenv (`pip install -r package/requirements-client.txt`;
  see [docs/install.md](docs/install.md#1-host-prerequisites)). PyYAML alone
  builds and installs the package, but leaves the agents without XProf.
- Standalone CLI tools (backend-dependent):
  - **`gcloud`** ([Google Cloud CLI](https://cloud.google.com/sdk/docs/install)) for Cloud TPU VMs, GCS staging, and GCP credentials (`gcloud auth login && gcloud auth application-default login`)
  - **`kubectl`** ([Kubernetes CLI](https://kubernetes.io/docs/tasks/tools/)) and **`gke-gcloud-auth-plugin`** (`gcloud components install gke-gcloud-auth-plugin`) for GKE Pod and Job workloads (`gcloud container clusters get-credentials <CLUSTER_NAME> --region=<REGION>`)

The installer checks only the host-side Python packages. The first session
sets up your backend and checks its tools, credentials and target (see
[docs/install.md](docs/install.md#check-the-setup)).

### 2. Build

Compile the canonical agent package into native distributions for every supported harness:

```bash
make build
```

This writes `dist/<harness>/tpu-agent-suite/` for `claude-code`, `codex`, `antigravity`, and `cursor`.

### 3. Install

See [`docs/install.md`](docs/install.md) ([headless / CI](docs/install.md#headless-installs-ci-containers)).

### 4. Use

Ensure your [virtual environment](docs/install.md#1-host-prerequisites) is activated whenever you use this agentic suite. Once inside the session, the `tpu-agent` skill is activated by any relevant natural language request, so no explicit invocation is needed. For example:

- **Onboarding & Benchmarking**:
  > *"Onboard Qwen/Qwen3.8-27B to run as a vLLM inference server on Cloud TPU using the torch_tpu backend. Send it a test completion request, then benchmark it and report TTFT, ITL and throughput in a table."*

- **Profiling**:
  > *"Analyze the XProf profile at /path/to/profile and identify memory bottlenecks and TPU compute utilization."*

- **Kernel Optimization**:
  > *"Optimize this custom Pallas flash-attention kernel for TPU v6e and run a self-refinement benchmark loop."*

What each request routes to, where workloads run, and where outputs land are in [`docs/using.md`](docs/using.md).

---

## Documentation

| Guide | For |
| :--- | :--- |
| [`docs/install.md`](docs/install.md) | Host and TPU prerequisites, build, install scopes and backends, upgrade/uninstall |
| [`docs/using.md`](docs/using.md) | Prompting guide, what `tpu-agent` routes each request to, and where outputs land |
| [`docs/developing.md`](docs/developing.md) | Dev environment, `make` loop, CI gates, adding agents, tpu-knowledge-base sources and eval contributions |
| [`docs/design/`](docs/design/README.md) | How the framework and the suite are designed |
| [`framework/README.md`](framework/README.md) | Compiler, installer, hooks, and runtime reference |
| [`framework/compute/README.md`](framework/compute/README.md) | Compute CLI and backend drivers |
| [`package/README.md`](package/README.md) | Converting an existing skill into a TPU agent suite package, step by step |
| [`eval/README.md`](eval/README.md) | Running evaluations |

---

## Running Evaluations

Evaluations are orchestrated using [Harbor](https://github.com/harbor-framework/harbor) and configured under [`eval/`](eval/):

```bash
# Set up evaluation virtual environment
./eval/setup.sh
source eval/.venv/bin/activate

# Verify evaluation configuration and dataset schemas
./eval/verify.sh smoke-check

# Run evaluation with Harbor
export GEMINI_API_KEY="your-api-key"
./eval/run.py --config claude-code/agent-suite --dataset tpu-hello-world
```

> **Automated PR Evaluations**: Smoke-check evaluations run automatically in CI whenever a pull request modifies agent packages, framework code, or evaluation files.

For the full list of configs, credentials, and the interactive sandbox, see [`eval/README.md`](eval/README.md).

---

## Contribution & Code Review Policy

All code contributions, tests, and agent packages live in this repository. Every pull request runs the following CI checks:

- **`make verify`**: manifest, delegation graph, schema, and knowledge base checks.
- **`make test`**: framework and package test suites (translation parity per harness).
- **`make build`**: distributions for all harnesses compile cleanly.
- **`make lint`**: advisory content checks, reported as annotations.
- **pre-commit**: formatting and static checks (`.pre-commit-config.yaml`).

Local setup for these checks, and the short path for common contributions, is in [`docs/developing.md`](docs/developing.md). Note that external contributions are not being accepted at this time (see [`CONTRIBUTING.md`](CONTRIBUTING.md)).

---

## Security

This project is eligible for the [Google Open Source Software Vulnerability Rewards Program (OSS VRP)](https://bughunters.google.com/about/rules/6521337925468160/google-open-source-software-vulnerability-reward-program-rules).

If you suspect a security vulnerability in this project, please report it via [g.co/vulnz](https://g.co/vulnz). Please do not report security vulnerabilities through public GitHub issues.
