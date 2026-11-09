# Installing TPU agent suite

This guide covers installing TPU agent suite into your workspace and
configuring TPU hardware compute targets. It assumes you have a standalone
distribution for your harness, extracted to
`tpu-agent-suite-<harness>-v<version>/` (for example `v0.1.0`). This file is
`docs/install.md` inside that directory. To get one, download the release
archive for your harness from GitHub Releases in this repository and extract it:

```bash
tar -xzf tpu-agent-suite-<harness>-v<version>.tar.gz
```

GitHub Releases publish prebuilt archives for `antigravity` and `claude-code`.
To build a distribution from a repository clone (including for `codex` or
`cursor`), see
[developing.md](https://github.com/google-pytorch/tpu-agent-suite/blob/main/docs/developing.md#building-a-package).

For what the agents do once installed, see [using.md](using.md). For the
framework internals behind these commands, see
[`README.md`](https://github.com/google-pytorch/tpu-agent-suite/blob/main/README.md) and
[`framework/README.md`](https://github.com/google-pytorch/tpu-agent-suite/blob/main/framework/README.md).

---

## 1. Host prerequisites

The host is the machine your harness (Claude Code, Codex, Antigravity, or
Cursor) runs on: a workstation or the TPU VM itself.

1. **Python 3.12+**: `python3 --version`.
2. **Build dependency**: `pip install pyyaml` is all the compiler and installer
   need.
3. **Host-side agent dependencies**: create a
   [virtual environment](https://docs.python.org/3/library/venv.html) and
   install the distribution's `requirements-client.txt` into it:

   ```bash
   python3 -m pip install -r tpu-agent-suite-<harness>-v<version>/requirements-client.txt
   ```

   Ensure your environment is activated whenever you install or use this
   agentic suite.

4. **Standalone CLI tools** (backend-dependent):
   - **`gcloud`** ([Google Cloud CLI](https://cloud.google.com/sdk/docs/install)):
     Required for `gce_tpu_vm` (to SSH/SCP into TPU VMs), and whenever
     accessing GKE clusters or Google Cloud APIs (Vertex AI, BigQuery). Authenticate with:

     ```bash
     gcloud auth login
     gcloud auth application-default login
     ```

   - **`kubectl`** ([Kubernetes CLI](https://kubernetes.io/docs/tasks/tools/))
     and **`gke-gcloud-auth-plugin`**, which kubectl needs to authenticate to
     GKE (`gcloud components install gke-gcloud-auth-plugin`): required for
     `gke_job` and `gke_pod` to dispatch Jobs, exec into pods, and
     inspect TPU cluster nodes. If using GKE, configure cluster credentials
     with:

     ```bash
     gcloud container clusters get-credentials <CLUSTER_NAME> --region=<REGION>
     ```

     This makes the cluster kubectl's current context. Setup suggests it as
     the profile's `kube_context`, so a later `kubectl config use-context`
     does not move `gke_job` or `gke_pod` work to another cluster.

The installer checks only the host-side packages above. The first session
sets up your backend and checks its tools, credentials and target; see
[Check the setup](#check-the-setup).

---

## 2. Install the distribution

### Install
With your virtual environment activated, run the distribution's self-contained
installer:

```bash
python3 tpu-agent-suite-<harness>-v<version>/framework/installer.py install --destination /path/to/project
```

*To install globally across all projects for that harness, omit `--destination` and pass `--scope user`.*

The installer will not overwrite unmanaged existing files or files modified locally by the user unless `--force` is specified.

### Upgrade
Download the new release archive and pass `--upgrade`:

```bash
tar -xzf tpu-agent-suite-<harness>-v<new-version>.tar.gz
python3 tpu-agent-suite-<harness>-v<new-version>/framework/installer.py install --upgrade --destination /path/to/project
```

Running `installer.py install` on an existing install upgrades in place and retains existing `environment.json` configuration without wiping run history.

> **Upgrading from a release that kept its files in
> `.tpu-agent-suite/tpu-agent-suite/`:** uninstall with that release's installer
> first, then install the new one. The new installer does not read the old
> layout.

A workspace holds one package at a time, and so does the user scope: installing
a different package where one is already installed fails until you uninstall
the first.

### Uninstall
Every install keeps a copy of its installer in the central store, so you do not
need the distribution to remove it:

```bash
python3 ~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/framework/installer.py uninstall --destination /path/to/project
```

*(If uninstalling a user-wide installation, pass `--scope user` instead of
`--destination`).* The installer in the extracted distribution works the same
way. A package's installer removes only that package: if the workspace holds
another one, it stops. Whichever installer you run, the installed version's own
installer does the uninstall.

Running `installer.py uninstall` performs a clean rollback of all installed files, configuration merges, and managed instruction blocks based on `ownership.json`.

---

## 3. Environment configuration

The installer asks no questions about compute. A fresh install writes an
`environment.json` with no profiles to
`<workspace>/.tpu-agent-suite/environment.json` (or, for user scope,
`~/.tpu-agent-suite/environment.json`, the only copy):

```json
{"revision": 1, "default": null, "profiles": {}}
```

The first session sets compute up: the agent asks which backend you use, then
only what that backend needs, suggests answers where it can find them, checks
the target, and saves the result as a named profile. A reinstall or upgrade
keeps your profiles and settings.

| Backend | Meaning | Backend-specific parameters |
| :--- | :--- | :--- |
| `local` | The harness is running on the TPU VM; workloads run in-process. | *(none)* |
| `gce_tpu_vm` | Reach a Cloud TPU VM over `gcloud ... ssh`. | `tpu_name`, `zone`, `project` |
| `gke_pod` | `kubectl exec` into an already-running TPU pod. | `namespace`, `pod`, optional `container`, `kube_context` |
| `gke_job` | Dispatch a TPU Job via `kubectl`, optionally through Kueue; a multi-host slice runs as one pod per node, host 0 the leader. | `image`, `tpu_accelerator`, `tpu_topology`, optional `namespace`, `queue`, `priority`, `declared_duration_minutes`, `kubeconfig`, `kube_context`, `hf_cache_pvc`, `staging_bucket`, `pod_service_account`, `staging_service_account`, `staging_service_account_key` |

Trajectory sharing does not depend on where workloads run, and it is off
until you turn it on. The first time you run
`/tpu-agent-export-trajectory`, the agent asks whether and where to share,
and saves your answer with `config set ... --user-confirmed`. A "no" is saved
as `trajectory_enabled=false`, so you are not asked again; the export then
refuses until you ask to share. The settings:

- `trajectory_enabled`: share session trajectories via
  `/tpu-agent-export-trajectory` (default: no). Turning it off also
  removes the sink, bucket and org below.
- `trajectory_sink` (only if enabled): `gcs` saves to a bucket **you own**;
  `harbor` uploads to the Harbor hub, which renders the session as a browsable
  trajectory (default: `gcs`).
- `trajectory_bucket` (only for `gcs`): e.g. `gs://my-bucket`. Without one,
  the export reports that no destination is configured and saves nothing.
- `trajectory_share_org` (only for `harbor`): Harbor organization to share
  uploads with; blank keeps the upload private. When a new name is saved
  with `config set`, it is checked with Harbor, and one that doesn't exist is
  refused
  (`harbor auth org list` shows yours); if it can't check (not logged in,
  offline), it keeps the name and warns.

Every profile can also set `python_interpreter`, and `xprof_interpreter` is a
setting of its own:
- `python_interpreter`: Path to the Python interpreter running workloads on the target (e.g. `~/tpu_venv/bin/python3` on a TPU VM, or `/opt/conda/bin/python` in a container). If omitted, defaults to the target's ambient `python3` (or `sys.executable` for `local`).
- `xprof_interpreter`: Separate Python interpreter for XProf analysis when it does not share the workload's environment (e.g. a dedicated venv for `xprof`).

### Headless installs (CI, containers)

In CI, a container, or an IDE/agent shell without a terminal, pass
`--non-interactive`: the installer then fails instead of asking before it
overwrites changed files or purges run data. Without a terminal and without
`--non-interactive` or `--force`, the installer refuses to start rather than
wait on stdin.

The installer sets up no compute. To have a profile ready before the first
session, create it with the runtime after the install, the same way a session
does:

```bash
python3 tpu-agent-suite-<harness>-v<version>/framework/installer.py install \
    --destination /path/to/project --non-interactive
cd /path/to/project
python3 ~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/framework/runtime.py \
    compute --package tpu-agent-suite@<version> setup apply \
    --profile default --answers answers.json
```

`answers.json` is a flat object keyed by the question names above (not the
nested `environment.json` path). A missing required field or an invalid value
is reported as INCOMPLETE, exit status 1, and nothing is written; a field with
a default (such as `gke_job`'s `image`) takes it when left out. The first
profile saved becomes the default. `setup apply` does not check the target;
`compute setup verify` does, and sessions check the profile before TPU work.
The full list of questions is in
`~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/environment/questions.json`.
Example per backend:

```jsonc
// local
{"backend": "local"}

// gce_tpu_vm
{"backend": "gce_tpu_vm", "tpu_name": "my-tpu-vm", "zone": "us-central2-b", "project": "my-project", "python_interpreter": "~/tpu_venv/bin/python3"}

// gke_pod
{"backend": "gke_pod", "namespace": "default", "pod": "my-tpu-pod"}

// gke_job (add "queue"/"priority" to admit through Kueue; leave out "image"
// to use the default nexus-tpu image, see "Container image" below)
{"backend": "gke_job", "namespace": "default",
 "tpu_accelerator": "tpu-v6e-slice", "tpu_topology": "2x2"}
```

To opt in to trajectory sharing, set it with `config` (whoever runs it agrees
on the user's behalf):

```bash
python3 .../runtime.py config --package tpu-agent-suite@<version> set \
    trajectory_enabled=true trajectory_sink=gcs trajectory_bucket=gs://my-bucket \
    --user-confirmed
```

### Changing your answers later

Ask the agent, or use `runtime config set` and `runtime compute setup apply`.
Use these rather than editing `environment.json` by hand: they check a change
the same way setup does. After a change, the next
compute command in a session checks the profile again (see
[Check the setup](#check-the-setup)), or run `compute setup verify`. The shape
the runtime reads is:

```json
{"revision": 1, "default": "default", "profiles": {"default": {"backend": "gce_tpu_vm", "params": {"tpu_name": "my-tpu-vm", "zone": "us-central2-b", "project": "my-project", "python_interpreter": "~/tpu_venv/bin/python3"}}}, "trajectory": {"params": {"enabled": true, "sink": "gcs", "bucket": "gs://my-bucket"}}}
```

The export uses these settings only while `trajectory.params.enabled` is
`true`.

---

## 4. What gets installed

For Claude Code at project scope:

```
.claude/skills/tpu-agent/SKILL.md      # the skill the harness activates
.claude/skills/tpu-agent-export-trajectory/SKILL.md   # see using.md
.claude/agents/*.md                       # one compiled prompt per agent
.claude/settings.json                     # subagent limits + pre-authorized ~/.tpu-agent-suite path
CLAUDE.md                                 # managed block with package-wide instructions
.tpu-agent-suite/
    environment.json                      # your compute profiles
    ownership.json                        # ledger of installed files

~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/   # framework (runtime, installer), docs, schemas, scripts, data
~/.tpu-agent-suite/runs/<run_id>/artifacts/              # outputs from each agent run
```

The managed memory block goes where each harness reads memory for that scope:

| Harness | `--scope project` | `--scope user` |
| :--- | :--- | :--- |
| Claude Code | `<workspace>/CLAUDE.md` | `~/.claude/CLAUDE.md` |
| Codex | `<workspace>/AGENTS.md` | `$CODEX_HOME/AGENTS.md` (default `~/.codex/AGENTS.md`) |
| Antigravity | `<workspace>/AGENTS.md` | `~/.gemini/AGENTS.md` |

A user-scope install never writes `~/CLAUDE.md` or `~/AGENTS.md`: those are
project memory, which a harness reads only for a session started in `$HOME`.
For Codex, a `CODEX_HOME` inside your home directory receives the memory
block, agents, `hooks.json` and `config.toml`. A `CODEX_HOME` outside it is
not supported: the install stops before changing anything and asks you to
unset `CODEX_HOME` or point it inside your home directory.
Reinstalling over a user-scope install made before this layout moves the
block out of `~/AGENTS.md` and leaves the rest of that file alone. Pass
`--no-workspace-memory` to skip the managed memory block.

> **Antigravity:** this harness has no config file the installer can
> pre-authorize, so the first time an agent reads `~/.tpu-agent-suite` the harness
> asks for permission. Grant it permanently once.

---

## 5. Set up the compute target

### Cloud TPU VM (`local` and `gce_tpu_vm`)

For `local` and `gce_tpu_vm`, the TPU VM needs a Python 3.12 environment with `torch_tpu`
and the TPU workload dependencies.

> **Important**: `torch_tpu` is published on [PyPI](https://pypi.org/project/torch-tpu/)
> and installs a compatible PyTorch build. Do **not** install PyTorch manually
> prior to `torch_tpu`.

```bash
# On the TPU VM (for gce_tpu_vm, first: gcloud compute tpus tpu-vm ssh <TPU_NAME> --zone=<ZONE>)
sudo apt update && sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv ~/tpu_venv && source ~/tpu_venv/bin/activate
pip install --upgrade pip

# 1. Install torch_tpu (PyTorch installs automatically; do NOT install torch manually)
pip install torch_tpu

# 2. Install remaining TPU workload dependencies and utilities
pip install -r tpu-agent-suite-<harness>-v<version>/requirements-tpu.txt
pip install portpicker marimo scikit-learn pandas tensorboard

# 3. Verify TPU hardware and driver visibility
tpu-info

# 4. Verify native PyTorch can allocate on the TPU device
python3 -c "import torch, torch_tpu; torch.zeros(1, device='tpu'); print('TPU Device OK:', torch.device('tpu'))"
```

Then point the profile at that interpreter: answer `python_interpreter` with
`~/tpu_venv/bin/python3` when the agent sets up compute, or run
`config set python_interpreter=~/tpu_venv/bin/python3`.

**vLLM serving** (optional): use a pre-configured vLLM TPU image, or install
`vllm-torchtpu` into `~/tpu_venv` (from `vllm-project/vllm-torchtpu`). Native
PyTorch serving (`custom_fastapi`) needs nothing extra.

**Disk space for model weights.** Models are downloaded to
`~/.cache/huggingface/hub` on the VM, for the user the agent connects as, and
stay there between runs, so each model downloads once. Size the VM's disk for
the models you plan to use: large models take tens of GB each, and nothing
deletes them for you. See [Model weights](using.md#model-weights).

### Container image (`gke_job`, `gke_pod`)

For containerized backends (`gke_job` and `gke_pod`), workloads execute
inside a container image pre-baked with Python 3.12, `torch_tpu`, and the workload
dependencies:

- **Image requirements**: The image must contain Python 3.12, `torch_tpu`
  (installed via Google Cloud Artifact Registry alongside its compatible PyTorch
  build), and the dependencies declared in
  `~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/requirements-tpu.txt`.
- **Interpreter configuration**: If your container uses a custom virtualenv or
  conda environment (e.g. `/opt/venv/bin/python3`), set `python_interpreter` to
  that path.
- **Filesystem layout**: The `gke_job` driver mounts workload execution scripts
  under `/app`, leaving `/workspace` free.

The default answer to `image` is a `nexus-tpu` image that meets these
requirements. The `nexus-tpu` image is generated nightly, but the default image
is pinned to a specific version rather than the absolute latest nightly to avoid
scenarios where the image version silently changes and hinders reproducibility.

The default version is:

```
us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/nexus-tpu@sha256:a13cfe0129b78c7752289c1a26b4e55cd2cb4ced3373dc524cf0332e8726e7ee
```

The `nexus-tpu` image is built nightly with two tags: `dev<YYYYMMDD-HHMM>`,
which names that one build, and `latest`, which always refers to the newest build.

To see which different nightlies are available, list the builds with:

```
gcloud artifacts docker images list \
  us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/nexus-tpu --include-tags
```

To use a version other than the default one, simply respond to the `gke_job` container
`image` question with the name of the desired image. I.e.,

```
us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/nexus-tpu@sha256:<desired_image_id>
```

To simply use the absolute latest nightly, you can respond with:

```
us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/nexus-tpu:latest
```

For example, these answers set up a `tpu7x` cluster that admits work through
the Kueue LocalQueue `default`:

| Question | Answer |
| :--- | :--- |
| `backend` | `gke_job` |
| `namespace` | *(blank: `default`)* |
| `image` | *(blank: the image above)* |
| `tpu_accelerator` | `tpu7x` |
| `tpu_topology` | `2x2x1` |
| `queue` | `default` |
| `kube_context` | *(suggested: the current kubectl context)* |
| `priority`, `declared_duration_minutes`, `kubeconfig` | *(blank)* |
| `staging_bucket` and the service-account questions after it | *(blank, unless inputs exceed 900 KiB or outputs 1 MiB; see below)* |
| `python_interpreter`, `xprof_interpreter` | *(blank: the image's `python3` has `torch_tpu`)* |

### Large files on `gke_job`: the staging bucket

Inputs over 900 KiB and outputs over 1 MiB travel through a GCS bucket
(`staging_bucket`) on signed URLs, so the TPU pod needs no Google credentials
of its own. Smaller files don't need a bucket.

Only a service account can sign those URLs. Your personal login can't, so set
**`staging_service_account`** to a service account for staging to act as:

```json
"staging_bucket": "gs://my-staging-bucket",
"staging_service_account": "stager@my-project.iam.gserviceaccount.com"
```

- You need `roles/iam.serviceAccountTokenCreator` on that service account, and
  it needs `roles/storage.objectAdmin` on the bucket.
- Only staging acts as it. `kubectl` keeps running as you.
- With no personal login (for example in CI), set
  **`staging_service_account_key`** to the account's key file instead. Set one
  or the other, not both.
- **Don't** point `GOOGLE_APPLICATION_CREDENTIALS` at a key to fix signing:
  `kubectl` reads that variable too, so it would run as the service account and
  usually fail with a 403 "forbidden".

#### On a cluster with Workload Identity: no signing

If the cluster has [Workload Identity](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity),
let the pod move staged files with its own token instead. Nothing is signed,
so no service account or key is needed, and your own login is enough for
the host's side (it needs `roles/storage.objectUser` or `objectAdmin` on the
bucket):

```json
"staging_bucket": "gs://my-staging-bucket",
"pod_service_account": "tpu-staging-pod"
```

One-time setup:

1. Workload Identity is on for the cluster (`--workload-pool=<project>.svc.id.goog`).
2. The TPU node pool runs the GKE metadata server
   (`--workload-metadata=GKE_METADATA`). Pods are scheduled only onto nodes
   that have it; `preflight` names the TPU node pools that don't.
3. Create the Kubernetes service account in the job's namespace:
   `kubectl create serviceaccount tpu-staging-pod -n <namespace>`.
4. Give it `roles/storage.objectAdmin` on the bucket, as the principal
   `principal://iam.googleapis.com/projects/<project-number>/locations/global/workloadIdentityPools/<project>.svc.id.goog/subject/ns/<namespace>/sa/tpu-staging-pod`.
   (Or annotate it with `iam.gke.io/gcp-service-account` to act as a Google
   service account that holds the role.)

### Shared model cache on `gke_job`: `hf_cache_pvc`

Every `gke_job` pod starts with an empty filesystem, so without a shared cache
each Job that loads a model downloads it again: the parity captures, the
server, every restart. Set **`hf_cache_pvc`** to a ReadWriteMany
PersistentVolumeClaim in the Job's namespace, and every pod mounts it at
`/hf-cache` with `HF_HUB_CACHE` pointing there, so a model downloads once.
Without it, each pod downloads into, and keeps its compile caches on, its
memory-backed `/dev/shm`, not its own disk, which on a TPU node can hold less
than one model; raise `shm_size_gb` (128 GiB by default) above the largest
model's weights plus its compile cache. The size is a cap, not a
reservation: the node's memory backs what the pod writes there, until it
is deleted, so keep it below the node's memory. This applies to every
`gke_job` pod, not only those that load a model: all of a pod's caches
(`HF_HUB_CACHE`, `VLLM_CACHE_ROOT`, `XDG_CACHE_HOME`,
`TORCHINDUCTOR_CACHE_DIR`) and its temporary files (`TMPDIR`) count against
`shm_size_gb`, and are node memory.
Each is a default the pod sets only when the image and the recipe have not:
an image that sets its own, or `HF_HOME`, keeps it.
Compute setup asks for the claim's name (leave it blank if there is none).
To add or change it on an existing profile, set it yourself, then run
`compute setup verify`:

```bash
python3 .../runtime.py config --package tpu-agent-suite@<version> set hf_cache_pvc=hf-cache
```

A cluster admin creates the claim once; nothing in the package creates
storage. On a cluster with the Filestore CSI driver, for example:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: hf-cache
  namespace: default
spec:
  accessModes: [ReadWriteMany]
  storageClassName: standard-rwx   # a ReadWriteMany Filestore class
  resources:
    requests:
      storage: 1Ti
```

- It must be **ReadWriteMany**: a server and its parity check run on
  different nodes at once, and with ReadWriteOnce the second stays Pending.
  `preflight` fails a claim that is missing or not ReadWriteMany, and warns
  while one is Pending under a class that binds on first use (the first Job
  creates the volume, which for Filestore takes several minutes).
- Bucket-backed claims (GCS FUSE) are not supported yet: GKE mounts them only
  in pods annotated `gke-gcsfuse/volumes: "true"`, which the driver does not
  add.
- **Every Job in the namespace can write to it, and later pods load what is
  there**, including model code run with `trust_remote_code`. Share a claim
  only among people who trust each other's Jobs, as you would a shared home
  directory; give each team its own namespace and claim otherwise.
- **Models are never pruned.** Size it for the models you use, and have its
  owner delete unused `models--*` folders when it fills. A Job that runs out of
  space logs `No space left on device`, which `compute job wait` reports as a
  failure. The benchmark scripts never delete models from it: they only
  clean up a cache that is private to the pod.
- **It also holds compiled programs.** Every pod keeps vLLM's and torch_tpu's
  compile caches (`VLLM_CACHE_ROOT`, `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`
  and `TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT`) in the claim's
  `compile-cache/<accelerator>-<topology>/<image key>/{vllm,torch_tpu}` folder,
  so a restart on the same image skips most of compiling (on Qwen3-0.6B, vLLM's
  torch.compile went from 15 s to 2 s and XLA compile from 16 s to 3 s). An
  image that sets these variables itself keeps its own values. A vLLM server
  patched with `apply_vllm_patch.sh` keeps the image key, so `serve_vllm.sh`
  moves its caches to `-patch-<key>` folders next to `vllm` and `torch_tpu`.
- **Old compile caches are pruned.** Each pod touches a `.last_used` file in
  its `<image key>` folder at start, then, in the background, removes the other
  `compile-cache/*/<image key>` folders in which no file (`.last_used`
  included) changed in `compile_cache_max_age_days`
  (`--compile-cache-max-age-days`, default 14; 0 turns pruning off), so
  folders from before this existed age out too. It never removes the folder it
  uses, and a failure is only logged. A pod that starts with the volume over
  90% full keeps its compile caches in `/dev/shm` instead, and logs why.
- **Tip:** pin the KV cache size of a server you restart
  (`--num-gpu-blocks-override` or `--kv-cache-memory-bytes`). Left unpinned, it
  can come out different on each start, and a new size compiles again and adds
  to the cache; pinned, a restart reuses it (torch.compile went from 285 s to
  8 s).

### Check the setup

`compute setup verify` checks a profile: the tools it needs are installed,
you have credentials, and the TPU VM, pod or cluster answers. On `local`,
`gce_tpu_vm` and `gke_pod`, where running something takes seconds, it also
runs the software check
(`~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/framework/compute/env_check.py`)
on the target, which
checks Python, `torch_tpu` and every package in `requirements-tpu.txt`
against its version bounds. On `local` it does so only when
`python_interpreter` is set, so that it checks the workload's Python rather
than the runtime's.

On `gke_job`, where running something means scheduling a Job, it can also
submit a smoke-test Job that runs the same software check in your `image`:
with `--smoke-test`, which the agent asks you about first. That is the one
check that shows
the nodes can pull the image and that it has `torch_tpu` and the packages
the agents need; it can take several minutes while a TPU node scales up. A
Job whose image pulls have kept failing for a minute fails with the
registry's reason, rather than waiting out the timeout; a Job still
queued at the timeout leaves the image unverified rather than failed.
Preflight also checks that an Artifact Registry `image` exists, as your
account.

Each check prints one line, with how to fix a failure: a command, or the
section of this guide that covers it. Commands are filled in where the
cluster is known: from a `gke_<project>_<location>_<cluster>` kubectl
context, or else `GKE_CLUSTER` / `GKE_LOCATION` / `GKE_PROJECT`.

The installer checks only this machine: `python3`, the interpreter the
agents' runtime uses, and the packages in `requirements-client.txt`,
including any package that conflicts with one of them, such as
`xprof-nightly` next to `xprof`. The installer does not change your
environment; it prints each failure with the command that fixes it, and each
session start asks you to run that command. `--skip-host-check` skips this
check.

Each session starts by telling the agent whether the run's compute profile is
set up, verified, failing or ready (checking a ready one again), and the next
compute command checks the target again after `environment.json` changes, so
the agent finds out about a broken setup before it tries to use it.

On `gke_pod` and `gke_job`, a cluster that refuses your gcloud account (403)
is tried again with application default credentials
(`gcloud auth application-default login`): a token from
`gcloud auth application-default print-access-token`, passed to gcloud as
`CLOUDSDK_AUTH_ACCESS_TOKEN`. If ADC is let in, compute commands use it
and the checks say so; if it is refused too, you get an error naming both
identities and the commands that fix it.

To run the compute checks yourself, use `compute preflight` (`--json` for a
machine-readable report). It exits 1 when a check fails:

```bash
python3 ~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/framework/runtime.py \
    compute --package tpu-agent-suite@<version> preflight
```

To run the software check yourself inside an image, copy
`framework/compute/env_check.py` and `requirements-tpu.txt` from
`~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/` into the image, then
run there:

```bash
python3 env_check.py --requirements requirements-tpu.txt
```

and on this machine:

```bash
python3 ~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/framework/compute/env_check.py --host \
    --requirements ~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/requirements-client.txt
```
