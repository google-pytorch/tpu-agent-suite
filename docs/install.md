# Installing TPU agent suite

This guide covers installing TPU agent suite into your workspace and
configuring TPU hardware compute targets. Download the pre-compiled
`tpu-agent-suite-<harness>-v<version>.tar.gz` archive (`claude-code` or
`antigravity`) from the release locations at the top of
[`README.md`](../README.md) (to build from source, including for `codex` or
`cursor`, see
[developing.md](https://github.com/google-pytorch/tpu-agent-suite/blob/main/docs/developing.md#building-a-package)).

## Quickstart (< 5 minutes)

```bash
# 1. Extract the release archive for your harness (claude-code or antigravity)
tar -xzf tpu-agent-suite-<harness>-v<version>.tar.gz

# 2. Create a Python 3.12+ virtual environment and install host dependencies
python3 -m venv ~/.venv/tpu-agent-suite
source ~/.venv/tpu-agent-suite/bin/activate
python3 -m pip install -r tpu-agent-suite-<harness>-v<version>/requirements-client.txt

# 3. Install into a project workspace (or omit --destination and pass --scope user)
python3 tpu-agent-suite-<harness>-v<version>/framework/installer.py install \
    --destination /path/to/project
```

Restart your harness in `/path/to/project` with `~/.venv/tpu-agent-suite`
activated. Your first session sets up and verifies your TPU compute profile
interactively. For what the agents do once installed, see [using.md](using.md);
for framework internals, see
[`framework/README.md`](https://github.com/google-pytorch/tpu-agent-suite/blob/main/framework/README.md).

---

## 1. Host prerequisites

The host is the machine your harness (Claude Code, Codex, Antigravity, or
Cursor) runs on: a workstation or the TPU VM itself.

1. **Python 3.12+**: `python3 --version`.
2. **Host-side agent dependencies**: create a
   [virtual environment](https://docs.python.org/3/library/venv.html) and
   install the distribution's `requirements-client.txt` into it:

   ```bash
   python3 -m pip install -r tpu-agent-suite-<harness>-v<version>/requirements-client.txt
   ```

   Keep this environment activated whenever you run sessions with the suite
   (the installer itself runs on standard-library Python and warns if host
   dependencies are missing).

3. **Standalone CLI tools** (backend-dependent):
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
Extract the new release archive and run `installer.py install` again (`--upgrade`
is enabled by default):

```bash
tar -xzf tpu-agent-suite-<harness>-v<new-version>.tar.gz
python3 tpu-agent-suite-<harness>-v<new-version>/framework/installer.py install --destination /path/to/project
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
| `gke_job` | Dispatch a TPU Job via `kubectl`, optionally through Kueue; a multi-host slice runs as one pod per node, host 0 the leader. | `image`, `tpu_accelerator`, `tpu_topology`, optional `namespace`, `queue`, `priority`, `declared_duration_minutes`, `kubeconfig`, `kube_context`, `hf_cache_pvc`, `compile_cache_url`, `staging_bucket`, `pod_service_account`, `staging_service_account`, `staging_service_account_key` |

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
  trajectory (default: `gcs`; `harbor` requires optional
  `pip install harbor==0.23.0` and `harbor auth login` in your host
  environment).
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
- `xprof_interpreter`: Python interpreter for XProf analysis on the host. Defaults to the python3 your harness started with (never python_interpreter, even on local); set only if xprof lives in a separate environment.

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
.claude/skills/tpu-agent/SKILL.md         # the skill the harness activates
.claude/skills/tpu-agent-export-trajectory/SKILL.md   # see using.md
.claude/agents/*.md                       # one compiled prompt per agent
.claude/hooks.json                        # lifecycle and schema-validation hooks
.claude/settings.json                     # subagent limits, hooks, ~/.tpu-agent-suite approval, and campaign/package deny rules
CLAUDE.md                                 # managed blocks with package-wide instructions
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

> **Sandbox & directory permissions:** Claude Code pre-authorizes
> `~/.tpu-agent-suite` in `approved_directories` while adding `permissions.deny`
> rules blocking agent reads/writes under `~/.tpu-agent-suite/campaigns/**` and
> writes under `~/.tpu-agent-suite/packages/**` (merging by union with any
> existing `permissions.deny` entries). Codex pre-authorizes
> `~/.tpu-agent-suite` in `[sandbox] allowed_paths`.
>
> **Antigravity:** installs skills, agents, `hooks.json`, and `plugin.json`
> under `.agents/` (project scope) or `~/.gemini/config/` (user scope), but has
> no setting for pre-authorizing external directories—so the first time an
> agent reads `~/.tpu-agent-suite`, grant access permanently once.

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

# 2. Install TPU workload dependencies
pip install -r tpu-agent-suite-<harness>-v<version>/requirements-tpu.txt

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

For `gke_job` and `gke_pod`, workloads run inside a container image with Python
3.12, `torch_tpu` (and its matching PyTorch build), and the packages in
`~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/requirements-tpu.txt`:

- **Interpreter**: If the container uses a custom virtualenv or conda
  environment (e.g. `/opt/venv/bin/python3`), set `python_interpreter` to that
  path.
- **Filesystem**: `gke_job` mounts execution scripts under `/app`, leaving
  `/workspace` free.
- **Default `nexus-tpu` image**: `image` defaults to a reproducible pinned
  digest rather than a moving tag:

  ```
  us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/nexus-tpu@sha256:a13cfe0129b78c7752289c1a26b4e55cd2cb4ced3373dc524cf0332e8726e7ee
  ```

  Nightly builds publish `dev<YYYYMMDD-HHMM>` and `latest` tags. List available
  builds or override `image` with another digest or `:latest`:

  ```bash
  gcloud artifacts docker images list \
    us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/nexus-tpu --include-tags
  # Example overrides for the `image` question:
  # us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/nexus-tpu@sha256:<desired_image_id>
  # us-docker.pkg.dev/tpu-nexus-767207/nexus-dev/nexus-tpu:latest
  ```

Example answers for a `tpu7x` cluster admitting work through Kueue LocalQueue
`default`:

| Question | Answer |
| :--- | :--- |
| `backend` | `gke_job` |
| `namespace` | *(blank: `default`)* |
| `image` | *(blank: the default digest above)* |
| `tpu_accelerator` / `tpu_topology` | `tpu7x` / `2x2x1` |
| `queue` | `default` |
| `kube_context` | *(suggested: current `kubectl` context)* |
| `priority`, `declared_duration_minutes`, `kubeconfig` | *(blank)* |
| `staging_bucket` and service-account fields | *(blank unless inputs > 900 KiB or outputs > 1 MiB; see below)* |
| `python_interpreter` | *(blank: image `python3` has `torch_tpu`)* |

### Large files on `gke_job`: the staging bucket

Inputs over 900 KiB and outputs over 1 MiB transfer through `staging_bucket` via
signed URLs, so TPU pods need no GCP credentials of their own (smaller files
need no bucket). Because personal user logins cannot sign URLs, set
**`staging_service_account`** for staging to impersonate:

```json
"staging_bucket": "gs://my-staging-bucket",
"staging_service_account": "stager@my-project.iam.gserviceaccount.com"
```

- Grant yourself `roles/iam.serviceAccountTokenCreator` on that service account,
  and grant it `roles/storage.objectAdmin` on the bucket (`kubectl` continues
  running as you).
- Without a personal login (e.g. in CI), set **`staging_service_account_key`**
  to a key file instead (set one or the other, never both).
- **Do not** set `GOOGLE_APPLICATION_CREDENTIALS` to fix URL signing: `kubectl`
  reads it too and will likely fail cluster auth with 403.

#### On a cluster with Workload Identity: no signing

With [Workload Identity](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity),
pods access `staging_bucket` with their own token—no URL signing, service
account impersonation, or key file is needed (your host login only needs
`roles/storage.objectUser` or `objectAdmin` on the bucket):

```json
"staging_bucket": "gs://my-staging-bucket",
"pod_service_account": "tpu-staging-pod"
```

One-time setup:

1. Enable Workload Identity on the cluster (`--workload-pool=<project>.svc.id.goog`).
2. Run the GKE metadata server on TPU node pools (`--workload-metadata=GKE_METADATA`);
   `preflight` flags pools without it.
3. Create the Kubernetes service account:
   `kubectl create serviceaccount tpu-staging-pod -n <namespace>`.
4. Grant `roles/storage.objectAdmin` on the bucket to
   `principal://iam.googleapis.com/projects/<project-number>/locations/global/workloadIdentityPools/<project>.svc.id.goog/subject/ns/<namespace>/sa/tpu-staging-pod`
   (or annotate the KSA with `iam.gke.io/gcp-service-account`).

### Shared model cache on `gke_job`: `hf_cache_pvc`

`gke_job` pods start with an empty filesystem. Set **`hf_cache_pvc`** to a
ReadWriteMany PVC in the Job namespace so every pod mounts `/hf-cache`
(`HF_HUB_CACHE=/hf-cache`) and downloads weights once:

```bash
python3 .../runtime.py config --package tpu-agent-suite@<version> set hf_cache_pvc=hf-cache
```

Without `hf_cache_pvc`, caches (`HF_HUB_CACHE`, `VLLM_CACHE_ROOT`,
`XDG_CACHE_HOME`, `TORCHINDUCTOR_CACHE_DIR`) and `TMPDIR` live on memory-backed
`/dev/shm` (capped by `shm_size_gb`, default 128 GiB; raise above weights +
compile cache, staying below node memory). Images or recipes that already set
these variables or `HF_HOME` keep their values.

A cluster admin creates the PVC once (e.g. with Filestore CSI):

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: hf-cache
  namespace: default
spec:
  accessModes: [ReadWriteMany]
  storageClassName: standard-rwx
  resources:
    requests:
      storage: 1Ti
```

- **ReadWriteMany required** (so concurrent server and parity pods can mount it;
  `preflight` fails `ReadWriteOnce` or missing claims, and warns on `Pending`
  first-consumer volumes). GCS FUSE PVCs are not yet supported (`gke-gcsfuse/volumes`
  annotation is not added).
- **Shared trust boundary**: Every Job in the namespace can read/write the PVC
  (including `trust_remote_code` repos); isolate untrusted teams in separate
  namespaces/PVCs.
- **Weight retention**: Models are never auto-pruned; delete unused `models--*`
  directories if disk fills (`No space left on device`).
- **Compile cache & pruning**: Pods store `VLLM_CACHE_ROOT`,
  `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`, and
  `TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT` under
  `compile-cache/<accelerator>-<topology>/<image key>/{vllm,torch_tpu}` (or
  `-patch-<key>` when patched via `apply_vllm_patch.sh`), pruning sibling
  `<image key>` folders unused for `compile_cache_max_age_days` (default 14; `0`
  disables; falls back to `/dev/shm` if volume > 90% full). Pin KV cache size
  (`--num-gpu-blocks-override` or `--kv-cache-memory-bytes`) so restarts reuse
  compiled graphs.

### Check the setup

- **Target verification (`compute setup verify` / `compute preflight`)**: Checks
  CLI tools, credentials, and target reachability—each check prints one line
  with how to fix a failure (a command or the section of this guide that covers
  it). On `local` (when `python_interpreter` is set), `gce_tpu_vm`, and
  `gke_pod`, it runs `framework/compute/env_check.py` against
  `requirements-tpu.txt`. On `gke_job`, pass `--smoke-test` (prompted during
  setup) to schedule a Job verifying image pull and `torch_tpu` packages (a Job
  whose image pulls keep failing for a minute fails early with the registry's
  reason rather than waiting out the timeout, while a Job still queued at the
  timeout leaves the image unverified rather than failed), and preflight checks
  Artifact Registry image existence. If `gke_pod`/`gke_job` returns 403 for your
  `gcloud` login, compute automatically retries with Application Default
  Credentials (`gcloud auth application-default login` via
  `CLOUDSDK_AUTH_ACCESS_TOKEN`).
- **Host check**: `installer.py` checks local `python3` and
  `requirements-client.txt` (including conflicting packages like `xprof-nightly`
  beside `xprof`) without modifying your environment (`--skip-host-check` skips
  this). Sessions re-verify compute profiles at start and whenever
  `environment.json` changes.

Run checks manually at any time:

```bash
# Compute preflight (exits 1 on failure; pass --json for machine-readable output)
python3 ~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/framework/runtime.py \
    compute --package tpu-agent-suite@<version> preflight

# Standalone image / TPU environment check (after copying env_check.py + requirements-tpu.txt)
python3 env_check.py --requirements requirements-tpu.txt

# Standalone host environment check
python3 ~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/framework/compute/env_check.py --host \
    --requirements ~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/requirements-client.txt
```
