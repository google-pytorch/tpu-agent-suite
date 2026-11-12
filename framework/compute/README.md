# Unified Compute Framework

The Unified Compute Framework provides a standardized compute interface for the agent framework, centralizing workload execution and hardware inspection across diverse compute types (local host, remote Cloud TPU VMs, GKE pods, GKE Jobs, and other internal backends) so that packages never need to re-implement compute functionality.

It provides agents and developers with a unified command interface (`{{runtime}} compute ...`) for:
- Hardware topology & device contention inspection (`inspect`)
- Package releases on the target, for citing per-release docs (`versions`)
- Synchronous script execution (`run`)
- Asynchronous batch and serving workload submission (`submit`)
- Port forwarding to remote devices, pods, and jobs (`port-forward`)
- Asynchronous job lifecycle management (`job status`, `job logs`, `job artifacts`, `job cancel`, `job reap`)

---

## Supported Backends

The framework ships with 4 built-in compute drivers:

| Backend | Description | Required Parameters (`environment.json` / CLI) | Optional Parameters |
| :--- | :--- | :--- | :--- |
| `local` | Direct execution on the local host machine. | None | `python_interpreter` (`--interpreter`, default `sys.executable`) |
| `gce_tpu_vm` | Remote Cloud TPU VM via GCE API / SSH. | `tpu_name` (`--tpu-name`), `zone` (`--zone`), `project` (`--project`) | `python_interpreter` (`--interpreter`, default `python3`) |
| `gke_pod` | Exec into an existing Kubernetes pod on GKE via `kubectl exec`. | `pod` (`--pod`) | `namespace` (`--namespace`, default `"default"`), `container` (`--container`), `kube_context` (`--kube-context`), `python_interpreter` (`--interpreter`, default `python3`) |
| `gke_job` | Dispatch a TPU Job onto GKE via `kubectl`, optionally scheduled via Kueue (see [Multi-host execution](#multi-host-execution-on-gke_job)). | `image` (`--image`), `tpu_accelerator` (`--tpu-accelerator`), `tpu_topology` (`--tpu-topology`) | Cluster, Kueue, staging, cache, and placement settings — see [`gke_job` optional parameters](#gke_job-optional-parameters) |

### Multi-host execution on `gke_job`

- **Indexed Job layout**: A topology larger than one node (e.g. `tpu7x` `2x2x2` or `v6e` `4x4`) runs as a Kubernetes Indexed Job with one pod per node behind a headless Service of the Job's name. Every pod runs the same script and inputs.
- **Worker environment**: Each pod has `TPU_NUM_HOSTS`, `TPU_WORKER_ID` / `TPU_NODE_ID` (`0` is the leader), `TPU_CHIPS_PER_HOST`, `TPU_HEAD_HOST`, and `TPU_WORKER_HOSTNAMES` set, and `MEGASCALE_*` blanked.
- **Chips per node**: On `v5e` and `v6e`, chips per pod come from the cluster's nodes automatically when `tpu_chips` is unset, since a `2x4` topology can be either one 8-chip node or two 4-chip nodes.
- **Status, logs & outputs**: `status` reports `RUNNING` once every host is running and `FAILED` (naming the host) when any host fails. `logs`, `wait --until log:`, and `port-forward` read from the leader (host `0`), and only the leader returns `--output` files.

### `gke_job` optional parameters

#### Cluster, Kueue & Naming

| Parameter (`environment.json`) | CLI Flag | Default | Behavior |
| :--- | :--- | :--- | :--- |
| `namespace` | `--namespace` | `"default"` | Kubernetes namespace to dispatch into. When `queue` is set, this must be the namespace containing the Kueue `LocalQueue`. |
| `kubeconfig` | `--kubeconfig` | ambient | Path to a kubeconfig file to use instead of the ambient `$KUBECONFIG` / `~/.kube/config`. |
| `kube_context` | `--kube-context` | current context | Passed as `--context` on every `kubectl` call so `kubectl config use-context` elsewhere does not move the profile to another cluster. `compute setup plan` suggests the current context; `setup verify` warns when the current context differs and fails when the kubeconfig lacks it. |
| `tpu_chips` | `--tpu-chips` | derived from topology (`4`) | Override chips requested per pod. |
| `queue` | `--queue` | unset | Kueue `LocalQueue` name to admit the Job through (omit on clusters without Kueue). |
| `priority` | `--priority` | unset | Kueue workload priority class; leaving this unset leaves the Job unlabelled so it takes the cluster default. |
| `declared_duration_minutes` | `--declared-duration-minutes` | unset | Declared runtime in minutes for Kueue suspension enforcement. |
| `job_name_prefix` | `--job-name-prefix` | unset | Prefix (1–30 lowercase letters, digits, or `-`) added to every generated Job name, e.g. `agent`. |
| `slots` | none (`config set slots=N`) | `1` | How many of this profile's servers a serving sweep may run side by side. Read by `tpu_slots.py`, not by the driver. |
| `mldiagnostics` | `--mldiagnostics` | `false` | Label the Job for GKE managed ML diagnostics so the workload appears in the Cloud Console Diagnostics view. |
| `python_interpreter` | `--interpreter` | `python3` | Python interpreter inside the container. |

#### File Staging & Credentials

| Parameter (`environment.json`) | CLI Flag | Default | Behavior |
| :--- | :--- | :--- | :--- |
| `staging_bucket` | `--staging-bucket` | unset | GCS bucket URI (`gs://bucket` or `gs://bucket/folder`) used to transfer large inputs (>900 KiB) and outputs (>1 MiB); see [Synchronous Code Execution (`run`)](#3-synchronous-code-execution-run) and [`docs/install.md`](../../docs/install.md#large-files-on-gke_job-the-staging-bucket). |
| `pod_service_account` | `--pod-service-account` | unset | Kubernetes service account for the Job's pod. On clusters with Workload Identity, the pod transfers staged files using its own token without signed URLs. |
| `staging_service_account` | `--staging-service-account` | unset | Google service account impersonated (using your personal login, with no key file) solely to sign GCS staging URLs; `kubectl` keeps your own identity. |
| `staging_service_account_key` | `--staging-service-account-key` | unset | Key file for the staging service account when impersonation is unavailable (e.g. in CI), without setting `GOOGLE_APPLICATION_CREDENTIALS` globally. |

#### Shared Memory (`/dev/shm`) & Model / Compile Cache

| Parameter (`environment.json`) | CLI Flag | Default | Behavior |
| :--- | :--- | :--- | :--- |
| `shm_size_gb` | `--shm-size-gb` | `128` | Size in GiB of the memory-backed `/dev/shm` mounted in every pod (since container runtimes default to 64 MiB, which is too small for vLLM). Recorded in `TPU_AGENT_SUITE_SHM_LIMIT_BYTES` because `statvfs` inside a pod can report total node memory. Without `hf_cache_pvc`, `/dev/shm` also backs model downloads, caches, and temp files (`HF_HUB_CACHE`, `VLLM_CACHE_ROOT`, `XDG_CACHE_HOME`, and `TORCHINDUCTOR_CACHE_DIR` default to `/dev/shm/cache` and `TMPDIR` to `/dev/shm`, only when unset by the image or recipe). Pass `--shm-size-gb` on `submit` to override for a single Job. |
| `hf_cache_pvc` | `--hf-cache-pvc` | unset | Name of an admin-created `ReadWriteMany` PersistentVolumeClaim in the namespace, mounted at `/hf-cache` in every pod with `HF_HUB_CACHE=/hf-cache` and `VLLM_CACHE_ROOT` + `torch_tpu` tier-3 compilation caches under `/hf-cache/compile-cache/<accelerator>-<topology>/<image key>/`. Model weights download once and compiled programs persist across Jobs (models are never auto-pruned; see [`docs/install.md`](../../docs/install.md#shared-model-cache-on-gke_job-hf_cache_pvc)). |
| `compile_cache_max_age_days` | `--compile-cache-max-age-days` | `14` | With `hf_cache_pvc`, each pod prunes in the background any compile-cache image folders untouched for this many days (never its own; `0` disables pruning). If the PVC is over 90% full at pod start, compile caches fall back to `/dev/shm`. |

#### Pod Placement (Colocated Clients & CPU-Only Jobs)

| Parameter (`environment.json`) | CLI Flag | Default | Behavior |
| :--- | :--- | :--- | :--- |
| `colocate_with` | `--colocate-with JOB_ID` | unset | Schedule onto the same node as running Job `JOB_ID` without requesting TPU chips (since `JOB_ID` holds them). Used for client workloads (e.g. a benchmark hitting a running server at `http://<job_id>:<port>` without `kubectl exec` or port-forwarding; creates an owner-referenced headless Service for the server Job if needed). Requires `colocate_requests`. |
| `colocate_requests` | `--colocate-requests NAME=QTY,...` | unset (required with `colocate_with`) | CPU and memory requests (e.g. `cpu=2,memory=8Gi`) used in place of TPU chips for a colocated client pod, since Kueue rejects pods with zero resource requests. |
| `cpu_requests` | `--cpu-requests NAME=QTY,...` | unset | Run on a CPU node without booking a TPU node (mutually exclusive with `colocate_with`). Requests the specified CPU/memory with no TPU node selector or toleration so TPU taints keep it off TPU nodes. Exposes the profile's TPU generation (`v6e`, `tpu7x`, ...) in `TPU_AGENT_SUITE_CHIPLESS_TPU`; if a `run` exits `75` or never starts, it automatically retries once on a TPU node (recording why in `details.cpu_fallback`). |

---

## Command Reference

These subcommands are designed to be invoked inline by agents within their instructions (via `{{runtime}} compute <action>`) as well as directly by developers (if needed) from the shell (`python3 runtime.py compute <action>`).

### Common Flags

These flags are supported across all `compute` actions:
- `--backend <backend>`: Explicitly select or override the compute backend (e.g. `--backend gce_tpu_vm`).
- `--workspace <path>`: Workspace root whose `.tpu-agent-suite/environment.json` is consulted (default: the current directory or, outside any workspace with an active run, the workspace of the run leased to this session).
- `--package <name@version>`: Package whose `environment.json` supplies the configuration. Every `{{runtime}} compute` in a compiled instruction carries this automatically; only hand-typed invocations need to pass it.
- `--interpreter <path>` (`run`, `submit` and `versions` only): Optional path to Python interpreter (e.g. `/opt/conda/bin/python`).
- `--json`: Format output as JSON (supported by every subcommand for structured agent parsing).

Backend-specific flags registered by drivers (e.g. `--tpu-name`, `--zone`, `--project`, `--pod`, `--namespace`) can also be passed directly on the CLI and outrank configured values.

### Running by hand

The examples below use the compiled-instruction form (`{{runtime}} compute ...`), where `--package` is added for you. From a shell, call the installed runtime and pass `--package` yourself. Without it no `environment.json` is consulted, and any command that needs the configured backend fails with `NO_COMPUTE_BACKEND`:

```bash
python3 ~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/framework/runtime.py \
    compute --package tpu-agent-suite@<version> inspect --workspace <PATH>
```

### 1. Inspect Hardware Topology (`inspect`)

Probes accelerator device availability, chip topology, model allocation, and health:

```bash
{{runtime}} compute inspect
```

On `gke_job`, `chips` counts the whole pool. What one Job gets is in `details`:
`job_chips`, `job_devices` and `hbm_gb_per_device`, with `devices_per_chip`
between them. With no node matching yet (an autoscaled pool at zero), `status`
is `no_accelerator` and the top-level fields say so, but `details` still
carries the profile's `tpu_accelerator` and these sizes, with a `nodes_note`. Size tensor parallelism to `job_devices`: a tpu7x chip is two
devices of 96 GB each, so a 2x2x1 Job has 8, not 4. If a package these
numbers come from is missing on the controller, they are left out, `hbm_gb`
is 0, and `hbm_gb_note` tells the agent to ask the user to install it
(naming the runtime's interpreter) and run `inspect` again.

Every session start also checks this machine against the package's
`requirements-client.txt` (shipped with the runtime) and, if anything is
missing or too old, opens with a `## Host software` note asking the user to
install it with one `<runtime python> -m pip install ...` command. The
installer runs the same check once; this repeats it each session until it
passes.

### 2. Check Prerequisites (`preflight`)

Checks that the configured
backend's tools are installed, that there are credentials, and that the target
answers. Exits 1 when a check fails:

```bash
{{runtime}} compute preflight [--json]
```

The report is the same for every backend: `backend`, `target` (a description a
person can recognise), `reachable`, `status` (`ok`, `warn` or `fail`, the worst
of the checks), and `checks`, each with a `status`, a `detail` and, for a
problem, the `fix` that resolves it.

| Backend | Checks |
| :--- | :--- |
| `local` | This host has TPU devices, and no other process holds them. |
| `gce_tpu_vm` | `gcloud` and an active account; the TPU VM exists and is `READY`. |
| `gke_pod` | `kubectl`, and `gke-gcloud-auth-plugin` when the kubeconfig calls it; the cluster answers; the gcloud account and `$GOOGLE_APPLICATION_CREDENTIALS` (below); the pod is `Running`. With `kube_context`: the kubeconfig has that context (a warning when kubectl's current context is another one). |
| `gke_job` | `kubectl`, and `gke-gcloud-auth-plugin` when the kubeconfig calls it; the cluster answers; the gcloud account and `$GOOGLE_APPLICATION_CREDENTIALS` (below); `image`, `tpu_accelerator` and `tpu_topology` are set; an Artifact Registry `image` exists (a warning when it cannot be checked), and, when it is in another project than the cluster's, whether the nodes' service account can read it (a warning unless the repository's policy shows it); the namespace exists and this account may create Jobs in it. With `kube_context`: the kubeconfig has that context (a warning when kubectl's current context is another one). With `queue`: the Kueue `LocalQueue` exists, and with `priority`, the `WorkloadPriorityClass`. With `staging_bucket`: `google-cloud-storage` imports, the staging credential (or else Application Default Credentials) exists and can sign a URL (a failure if not, unless `pod_service_account` is set: the pod cannot reach the bucket, so the profile is not READY), and the bucket, or the folder in it, can be listed. With `hf_cache_pvc`: the claim exists and is ReadWriteMany, and is Bound (a warning while Pending under a storage class that binds on first use; a failure otherwise). |

A check the account is not allowed to make (reading a cluster-scoped object,
say) is a warning, not a failure.

On the GKE backends, a cluster that cannot be reached is diagnosed as
precisely as the machine allows:

- kubectl reports a kubeconfig that does not exist as merely having no
  current context. Preflight names the missing file instead (`kubeconfig`,
  or the paths in `$KUBECONFIG`), and its `get-credentials` fix writes to
  that file, since writing the default one would leave kubectl still
  reading the missing path.
- `get-credentials` fixes name a real cluster where one is known: from
  `GKE_CLUSTER`, `GKE_LOCATION` and `GKE_PROJECT` when any is set (as CI
  and eval environments set them), otherwise from a
  `gke_<project>_<location>_<cluster>` kubectl context, never a mix of the
  two, with placeholders for whatever the chosen source lacks. Checks of a
  cluster that answers (such as who its nodes pull the image as) use the
  context first, since that is the cluster kubectl reached.
- The gcloud account (`gcloud:account`) is checked on every backend that uses
  gcloud, when gcloud is installed, since `gke-gcloud-auth-plugin` authenticates kubectl through it.
  When the account is this VM's (or GKE node's) own service account, the
  check also reads the VM's OAuth scopes from the metadata server: without
  `cloud-platform`, its tokens are refused whatever IAM grants. On
  `gke_pod` and `gke_job`, a missing account or scope is a warning, not a
  failure, when another credential works: the cluster answers, or
  `$GOOGLE_APPLICATION_CREDENTIALS` holds a usable credential.
- `$GOOGLE_APPLICATION_CREDENTIALS`, when set, is reported
  (`credentials:gac`) with its type (`authorized_user`, `service_account`,
  ...) and account, always as a warning: `gke-gcloud-auth-plugin` honours
  it, so it also changes who kubectl authenticates as, and a credential
  without RBAC on the cluster gets an opaque HTML 403.

Preflight runs in `compute setup verify`, at the start of a session whose
profile is verified, when `environment.json` changes (below), and whenever you
call it. In `verify`, on backends where `run` takes seconds (`local`,
`gce_tpu_vm` and `gke_pod`, the drivers with `quick_run` set), it then runs
`env_check.py` on the target. It checks the installed software against the
distribution's `requirements-tpu.txt` (see `docs/install.md`). On `gke_job`
(`scheduled_run`), where `run` schedules a Job, `--smoke-test` runs the same
check as a smoke-test Job instead: the one check that the nodes can pull
`image` and that it holds the right software.

`compute setup plan [--profile NAME] [--backend X] [--fields a,b]` works out
what a profile needs and suggests a value for each: with no backend, just the
backend (and the existing profiles); otherwise the backend's `SETUP_FIELDS`, or
just `--fields`. Each comes as a question with its current value and a
suggestion found by
read-only probes (`probes.py`: `gcloud`, `kubectl`, and the host's TPU devices), each limited to `PROBE_TIMEOUT_SEC` and run in parallel. A
probe that finds nothing leaves its questions without a suggestion, and one
that finds several values with none clearly meant (TPU VMs in a shared
project, the user's own first, at most `MAX_CHOICES`) lists them as the
question's `choices` instead, each TPU VM as `{"tpu_name", "zone"}` since a
name is unique only within a zone (`zone` is suggested only if every VM is
in it); a
missing tool, a timeout, or a chosen cluster with no TPU nodes is also listed
in `probe_warnings`. With `--fields` (asked again after a failed verify), the
current value is shown but not suggested.
`compute setup apply` then saves the answers.

`compute setup verify [--profile NAME] [--smoke-test]` runs these checks on
a saved profile. The smoke test runs only with `--smoke-test`: verify never
prompts, since the agent that runs it would answer for the user. Skipping it
still counts as a pass, recorded with
`smoke_test: false`. It prints progress on stderr and returns the
checks, plus the `fields` to ask about again (each failed check's
`PreflightCheck.fields`, from the driver's `CHECK_FIELDS`). The result is
saved in `verification.json` beside `environment.json`, under a hash of the
profile's settings, so changing any setting makes that profile unverified
again. `config show` reports each profile's `setup_state`: `NOT_CONFIGURED`
(incomplete), `UNVERIFIED`, `FAILED` or `READY`.

#### At session start, and when environment.json changes

- **Session start** (new or resumed): the `start-run` SessionStart hook works
  out the run's profile (`--profile`, then the run's, then the default) and
  its `setup_state` (`setup.setup_state`, the one `config show` uses), and
  adds one message to the agent's context. It never blocks the session.
  - `NOT_CONFIGURED`: no profile is selected, or the selected one is
    incomplete. The message gives the setup steps (plan, ask in one round,
    apply until ok, verify, asking before `--smoke-test` on backends that
    schedule Jobs) with full commands, and names the shared guide,
    `compute-setup.md`. If profiles exist but none is selected, it lists them
    and gives `config use <name>`.
  - `UNVERIFIED`: run `compute setup verify`; on backends that schedule Jobs,
    ask before adding `--smoke-test`.
  - `FAILED`: the failing checks from the last verify (verify keeps them in
    `verification.json`), their fixes and the fields to ask about again.
  - `READY`: the only state that runs preflight, within the hook's timeout
    minus `SESSION_CHECK_HEADROOM_S` (5 s). If it passes, the note says
    "Compute ready" and, using the same driver, adds the real hardware from
    `inspect`: accelerator type, device count and HBM per device. A failed
    check reads as `FAILED`. If the hardware can't be read in time, the note
    says so and points to `compute inspect --json`; the next compute command
    runs the check.

  The other states need no network, so session start stays fast. On the
  Google harnesses, later turns don't replay the message, so after the first
  turn the agent learns setup is pending only from a compute command's
  `COMPUTE_NOT_CONFIGURED`.
  - On Claude Code, a package skill's tool limit still lets the main thread
    run this package's `runtime.py compute --package <pkg> setup ...` and
    `runtime.py config --package <pkg> ...` commands (nothing chained or
    redirected; the runtime may be named by its full path or `~/...`, as the
    guide does), so setup works after a skill is active (`_is_setup_command`
    in `hooks/actions.py`).
- **`environment.json` changes**: each check saves a hash of the profile it
  checked in `{{runs}}/<run_id>/compute-preflight.json`. The next compute
  command (other than `preflight`) whose profile hashes differently checks the
  target first, prints the result on stderr, then runs as usual. Changes to
  other profiles or to settings outside the profile don't count. With
  `--profile`, it checks that profile's target, so switching between
  `--profile` and the run's own profile checks again.

Without `--backend`, a compute command fails with `COMPUTE_NOT_CONFIGURED`
when no profile is selected or the selected one is one `compute setup apply`
would not save; the error lists the fields in `missing` and the
`compute setup plan` command in `fix`. A field asked only when a command
needs it (`SETUP_FIELDS.on_demand`, such as `staging_bucket`) fails with
`<GROUP>_NOT_CONFIGURED`, with `missing` and `fix: ... compute setup plan
--profile <name> --fields <field>`. Drivers raise it with
`self.not_configured(field)`: the group is the `CHECK_FIELDS` check the field
belongs to, before any `:` (`staging:bucket` -> `STAGING_NOT_CONFIGURED`),
else the field itself; the reason is its `on_demand` entry. The CLI fills in
`fix` with the profile the driver came from.

Each job record keeps the `compute_profile` it was submitted under. Job
commands (`status`, `logs`, `wait`, `cancel`, `list`, `reap`, ...) resolve
their driver from that profile and the record's `backend`, so `config use`
does not move a running job to another target. `--profile` still wins. A
record from before this field, or whose profile was deleted (with a warning),
uses the run's profile, so it can still be cancelled.

**ADC retry** (`gke_pod`, `gke_job`): if the cluster refuses the gcloud
account, kubectl is retried once with application default credentials: a token
from `gcloud auth application-default print-access-token`, passed to gcloud
(and so `gke-gcloud-auth-plugin`) as `CLOUDSDK_AUTH_ACCESS_TOKEN`. If ADC
works, the rest of
the command uses it and preflight adds a `credentials:adc` warning. If ADC is
refused too, or there is none, preflight fails the `cluster` check and compute
commands raise `CLUSTER_ACCESS_DENIED`, naming both identities and the fix.
Only kubectl's own refusal messages count, and a `kubectl exec` is retried only
if kubectl was refused before the pod's shell started, so a workload that hits
a 403 is never re-run. `port-forward` can't be retried, so it picks the
credential with a `get` first. The helpers are in `drivers/auth_utils.py`.

### 3. Synchronous Code Execution (`run`)

Executes a local Python script or shell script on the target accelerator environment and streams standard output and standard error:

```bash
{{runtime}} compute run --code scripts/train.py --timeout 300 -- --batch-size 32 --lr 1e-4
```

Only `--code` travels to the target on its own. A script that reads a sibling
file, or writes one the caller needs, declares them:

```bash
{{runtime}} compute run --code scripts/sweep.py --timeout 2000 \
  --input workspace/payload.json \
  --output results.json:workspace/results.json \
  -- payload.json results.json
```

- `--input LOCAL[:REMOTE]` copies a local file next to the script before it
  runs. `REMOTE` is relative to the script's working directory on the target
  and defaults to the file's name. Repeatable.
- `--output REMOTE[:LOCAL]` copies a file the script wrote back to this host
  after it exits. `LOCAL` defaults to the same spelling. Repeatable. A run
  that exits 0 without producing a declared output is reported as exit 1
  with the missing file named on stderr.
- On `gke_job`, `--timeout` starts counting when the Job's pod starts
  running, so time spent queued (waiting for Kueue or a node) doesn't count.
  The queue wait has its own limit of the same length. If a Job times out,
  the error says whether it ever ran and how long it was queued
  (`details.queued_sec`); one that never ran is deleted and reported with
  `details.never_started`, no logs read.
- With either flag the script runs in a fresh working directory holding the
  inputs -- on every backend, `local` included -- so pass bare names after
  `--`. Without them it runs in the caller's directory, as before.
- `--name NAME` names the Job on `gke_job` (other backends ignore it), with
  a random suffix so parallel runs don't collide: `qwen-bench-1a2b3c`. Every
  gke_job name, from `run` or `submit`, is made a valid Kubernetes name
  (`My_Job` → `my-job`), given `job_name_prefix` if set, and trimmed to 63
  characters, so use the `job_id` that is printed.
- On `gke_job`, small files need no bucket: `--input` files that fit in
  the Job's ConfigMap with the script (900 KiB in all) travel in it, and
  `--output` files come back framed in the pod's log (1 MiB in all), checked
  against their sha256; an output over that is reported missing, not cut
  short. An output whose upload the bucket refuses comes back the same
  way. Larger inputs, and outputs whenever the bucket can take them, go
  through `staging_bucket` on V4 signed URLs, so the workload image needs no
  GCP credentials. `staging_bucket` may name a folder in the bucket
  (`gs://bucket/folder`).

  **Who signs the URLs.** Only a service account can sign them; a person's
  own `gcloud auth application-default login` cannot. So staging acts as a
  service account of its own, and only staging does -- kubectl keeps running
  as you:
  - `staging_service_account` (`--staging-service-account`): staging uses
    your login to act as this account, with no key file. You need
    `roles/iam.serviceAccountTokenCreator` on it, and it needs
    `roles/storage.objectAdmin` on the bucket. This is the usual choice.
  - `staging_service_account_key` (`--staging-service-account-key`): a key
    file for that account, for setups with no personal login (e.g. CI).

  With neither set, staging uses the host's Application Default Credentials,
  which can sign only if they are a service account themselves: a GCE/GKE
  host's own account (holding `roles/iam.serviceAccountTokenCreator` on
  itself), or `gcloud auth application-default login
  --impersonate-service-account=<sa>` (gcloud's name for acting as `<sa>`).
  Don't point `GOOGLE_APPLICATION_CREDENTIALS` at a key to make signing work:
  kubectl reads that variable too, so every kubectl call would then run as
  the service account and usually fail with a 403. Without a credential that
  can sign, inputs too large for the ConfigMap are refused, and outputs fall
  back to the log, so those over 1 MiB in all are lost; the error then
  prints the `setup plan` command that asks for a way for the pod to
  upload them. `setup verify` fails on such a profile for the same reason.

  **Or sign nothing: `pod_service_account`** (`--pod-service-account`). On a
  cluster with Workload Identity, name a Kubernetes service account in the
  job's namespace that holds `roles/storage.objectAdmin` on the bucket. Pods
  then run as it (`serviceAccountName`), are scheduled only onto nodes with
  the GKE metadata server (`iam.gke.io/gke-metadata-server-enabled=true`, i.e.
  a node pool with `--workload-metadata=GKE_METADATA`), and fetch a token from
  it to read and write plain `storage.googleapis.com` object URLs. The host
  uploads and downloads with its own credentials, so a personal login works
  and neither setting above is needed. `preflight` checks the account
  exists. Setup is in [docs/install.md](../../docs/install.md).

  The pod's stderr is merged into its stdout, and each frame is marked with
  the Job's name, so neither the pod's own messages nor a workload that
  prints another run's log can break one. Staged objects are deleted when the run finishes; a lifecycle rule on the bucket
  covers the run that dies before it can.
- Staged objects of an asynchronous `submit` are cleaned up differently,
  because nothing is waiting to collect them: inputs go with the Job when it
  is cancelled or reaped, and outputs are deleted by `job artifacts` once all
  of them are on disk. A job whose outputs are never fetched, or only partly
  fetched, keeps them under `gs://<staging_bucket>/<job_id>/out/` -- results
  nobody has collected yet are not the reaper's to destroy. Set a lifecycle
  rule on the bucket to age those out.
- On `gke_job`, a finished Job's logs are copied into its job record the
  first time `job status` (or `run`, or a cancel) sees it finished, and `job
  logs`/`job artifacts` read that copy: the pod is garbage-collected once the
  autoscaler removes its TPU node. With a `staging_bucket`, the pod also
  uploads its output to `gs://<staging_bucket>/<job_id>/out/`, which covers a
  job nobody watched finish. With neither copy (a pod evicted or killed
  before its upload ran), `job logs` reads the last
  `CLOUD_LOGGING_MAX_LINES` lines from Cloud Logging with `gcloud logging
  read`, under a first line saying so; when gcloud is missing or Cloud
  Logging has nothing, it fails with `LOGS_UNAVAILABLE`, saying why and
  giving the query to run by hand. The query matches the Job's UID, kept at
  submit, so a Job cancelled and resubmitted under the same `--name` never
  shows the other's output; a record without one is bounded to exactly its
  own start and end.
- On `gke_job`, a failed Job's error carries the pod's own reason as well
  as the exit code: an eviction reads `pod was evicted: The node was low on
  resource: ephemeral-storage... (container exited with code 137)`, not a
  bare 137. A Job removed outside compute fails `job status` with
  `JOB_GONE`, saying where it was last seen (status, node, pod, time),
  from the `last_seen.json` each status read writes beside its record.
- On `gke_job`, a Job whose image cannot be pulled is reported `FAILED`,
  with the registry's reason, and deleted: by `run`, and by `job status` for
  a submitted one, since its pod would otherwise retry the pull forever
  while holding a TPU node. An image name that can never pull
  (`InvalidImageName`) fails at once; a pull that is retrying fails once its
  events show failed pulls a minute apart, so one registry error followed by
  a long retry is ridden out. Without access to the pod's events it is never failed
  early, and the wait runs its course. Once such a Job is deleted, `job
  status` answers from its record, with the error, and `job logs`/`job
  artifacts` fail with `JOB_NEVER_STARTED`: no container ran, so there is
  no output to find.
- On `gke_job`, a Job Kueue has suspended is `PENDING` with `suspended` set
  while its Kueue Workload waits -- for quota, behind a queue its
  `stopPolicy` holds, or requeued after a preemption (`kueue_reason`
  `Preempted`, the message naming the preemptor). Kueue's message is passed
  on in the error and in `kueue_message`. A Job Kueue will never admit is
  `FAILED` at once instead, with `kueue_reason` and Kueue's message, and
  deleted like a pull failure: its Workload is inadmissible (a LocalQueue or
  ClusterQueue that does not exist or is misconfigured, a request no flavor
  or quota can hold), or Kueue made no Workload for it within a minute of
  its creation (usually a WorkloadPriorityClass the cluster does not
  define). A Workload deactivated by a cluster time-limit controller
  (`quota-system/eviction-reason`) is `FAILED` with the controller's message
  and `eviction_reason`, and kept so `job logs` still works. If the Workloads
  cannot be read -- no Kueue CRD, no right to list
  them -- the Job stays `PENDING` and the wait runs its course.

> **_NOTE:_**
> **The `--` Delimiter**: Always separate `runtime` flags from the script's arguments with `--`. Any arguments following `--` are forwarded verbatim to the target script.

### 4. Asynchronous Workload Submission (`submit`)

Submits an asynchronous batch workload or long-lived serving daemon and prints the resulting `job_id`:

```bash
# Standard batch job
{{runtime}} compute submit --recipe recipes/pretrain.yaml --name llama-run-01 -- --learning-rate 0.001

# Serving daemon mode: exempts the job from automatic session-end reaping
{{runtime}} compute submit --recipe server.py --serve --input deployments.json --name llama-server

# Multi-file staging with declared outputs
{{runtime}} compute submit --recipe recipes/train.py \
  --input config.json \
  --output checkpoint.pt \
  -- --epochs 5
```

- `--serve`: Marks the workload as a persistent service. Unlike batch jobs, serving jobs are **exempt from the session-end `reap-jobs` hook**, allowing services to remain running across sessions until explicitly cancelled with `compute job cancel` or `compute job reap --include-serving`. On `gke_job` a serving workload is dispatched with `restartPolicy: OnFailure` and a `backoffLimit` of 6, so a crash is restarted rather than ending the service -- bounded, not infinite, so a container that cannot start stops holding the chips instead of crash-looping on them. (Note: On queued clusters such as Kueue-managed GKE, cluster-level admission bounds like `declared_duration_minutes` still apply).
  - On `gke_job`, a serving Job also gets a headless Service of the same name, owned by the Job (so cancelling deletes it): other pods in the namespace reach it at `http://<job_id>:<port>`. To benchmark it without `exec` or a port-forward, run the client with `--colocate-with <job_id>`, which puts it on the server's node without chips.
  - A batch Job gets the same Service when a `--colocate-with` client of it is submitted, so a server this run measures and then tears down needs no `--serve`: it stays reapable, and a crashed run does not leave it holding the chips.
  - `--cpu-requests NAME=QTY,...` (`gke_job`): a Job that holds no TPU node. It runs in the task image on a CPU node, requesting that CPU and memory, for checks that need the image's software but not a chip. It finds the profile's TPU generation in `TPU_AGENT_SUITE_CHIPLESS_TPU`. With `run`, exiting 75 asks for the same run again on a TPU node, and a run no CPU node takes gets one too.
- `--input LOCAL[:REMOTE]`: Stages files next to the recipe before it starts. Repeatable.
- `--output REMOTE[:LOCAL]`: Declares files produced by the recipe to be fetched when running `compute job artifacts`. Repeatable. The file lands under the artifact directory at its target-side path, so `--output eval/metrics.json` and `--output train/metrics.json` stay two files. `LOCAL` is a path on the submitting host and is not consulted by `job artifacts`, which may run from a different workspace or session entirely; only `run`, which fetches in the same breath as it executes, honours it.
  - On `gke_job`, the pod also sends each output again whenever its file changes while the recipe runs (every 30 s at most), and passes SIGTERM (a timeout or preemption) on to the recipe and then sends the outputs. So a job that ends early, even with its node lost, still returns its outputs as they last were. A recipe that rewrites an output as it goes should replace the file whole (write a temporary file, then rename it).

### 5. Port Forwarding (`port-forward`)

Establishes a port forward from a local port to a remote workload (Cloud TPU VM, GKE pod, or GKE job):

```bash
# Forward local port 8000 to remote port 8000 (blocks in foreground)
{{runtime}} compute port-forward --port 8000

# Map local port 8080 to remote port 8000 on a specific GKE Job
{{runtime}} compute port-forward --port 8080:8000 --job-id <job_id>

# Run in background to avoid blocking the agent or shell
{{runtime}} compute port-forward --port 8080:8000 --job-id <job_id> --background
```

- `--background` (`-b`): Runs the port forward in the background and prints its PID (`Port forward established in background (PID: <pid>)`), allowing automated agents and scripts to proceed without blocking. It returns once something accepts a connection on the local port, not after a fixed interval: a tunnel that exits, or that never binds within 30s, is reported as `PORT_FORWARD_FAILED` with the child's output quoted and its log path, rather than handing back a pid for a forward nothing is listening behind. The background process is not tracked after that -- nothing cancels it at session end, so stop it by pid when you are done with it.
- On `gce_tpu_vm`: Establishes an SSH tunnel via `gcloud compute tpus tpu-vm ssh ... -N -L <local>:localhost:<remote>`.
- On `gke_pod`: Establishes a port-forward to the target pod via `kubectl port-forward pod/<pod> <local>:<remote>`.
- On `gke_job`: Automatically discovers the job's running pod (of a multi-host job, the leader's, host 0) and forwards to it, falling back to `job/<job_id>` when no pod is up yet. `--job-id` is required.
- On `local`: A no-op -- the workload runs on this host and is already reachable, so nothing is tunnelled. A mapping to a *different* local port is refused with `UNSUPPORTED_OPERATION`, since nothing would be listening on it.

### 6. Job Lifecycle Management (`job`)

Query or control asynchronous jobs submitted via `submit`:

```bash
# Check job status (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
{{runtime}} compute job status --job-id <id>

# Wait until the job runs, finishes, or prints a line -- and stop early if
# its logs show it failing while it still reads RUNNING. Exits 0 reached,
# 2 a failure line appeared, 3 timed out (default 540s, under an agent shell's
# ten-minute limit; `next` says whether to run it again or, for a job long
# PENDING with no node and no queue, to inspect or cancel), 4 the job ended
# without reaching it, and 1 on an error -- including a status that could
# not be read three times in a row, e.g. an unknown job id. Built-in failure
# patterns are RESOURCE_EXHAUSTED, SIGBUS, `RuntimeError: cancelled` and "No
# space left on device"; add with --fail-on-log, drop with
# --no-default-fail-patterns.
{{runtime}} compute job wait --job-id <job_id> [--until running|done|log:<regex>] [--timeout <sec>]

# Fetch stdout/stderr execution logs
{{runtime}} compute job logs --job-id <id>

# Download job output artifacts (defaults to {{runs}}/<run_id>/artifacts/<job_id>/)
{{runtime}} compute job artifacts --job-id <id> [--out-dir ./artifacts]
# Exits 1 and lists `missing_outputs` if a declared --output was never
# produced, even when the job itself reports COMPLETED.

# List tracked compute jobs for a run or the current workspace. The run's
# unfinished jobs are first asked for their live status; one whose backend
# cannot answer keeps its recorded status plus `refresh_error` in --json.
{{runtime}} compute job list [--run-id <run_id>] [--no-refresh]

# Cancel an active or queued job. Only a job submitted from here, by this run
# (or by the run named with --run-id), is cancelled; any other id fails with
# NOT_OWN_JOB, since on a shared cluster it is another run's or a person's.
{{runtime}} compute job cancel --job-id <id>

# Cancel every job the run left active, except those submitted with
# `--serve`. The package's `session-end-reap` hook (builtin `reap-jobs`, on
# SessionEnd) runs this for the session's run when the session is over; this
# is the manual form. Reaped and skipped ids are both reported.
{{runtime}} compute job reap [--run-id <run_id>]

# Clear the run completely, serving jobs included. A serving job the reaper
# refused to stop holds TPU chips with nothing left pointing at it, so an
# explicit reap can be told to take them too.
{{runtime}} compute job reap [--run-id <run_id>] --include-serving

# Cancel the batch jobs of every run whose session ended without SessionEnd
# (see "Sessions that never end"). The session-start hook runs it detached.
{{runtime}} compute job reap --orphaned

# Wait until the run's session has been gone for the grace period, then cancel
# its batch jobs. `submit` runs it detached.
{{runtime}} compute job reap --watch [--run-id <run_id>]
```

Every job belongs to a run. `submit` tags it with `--run-id` when given,
otherwise with the active run of the workspace containing the current
directory, which the session-start hook records. From outside any workspace
(an agent that has `cd`'d to `/tmp`, say) it falls back to the run leased to
this session's harness process, and reads that run's workspace and its
`environment.json` too. With none of these it refuses, since a job tagged with
no run is one the reaper can never find. `job artifacts` and `job reap`
resolve the run the same way. One caveat: the active run is per workspace, so
two sessions open in the same workspace at once share it, and a job submitted
without `--run-id` in one is reaped when the other ends.

### 7. Package Releases on the Target (`versions`)

Reports the releases of jax, jaxlib, libtpu, torch and torch_tpu on the
target, for an agent citing documentation written per release. It never
answers from the machine the agents run on, whose packages need not be the
target's:

```bash
{{runtime}} compute versions [--json]
```

- `local`: asks the configured `python_interpreter` (or `--interpreter`) now,
  so the answer is current. `source` is `live`.
- Any other backend: asking would be a remote job per question, so the answer
  is what the last `compute setup verify` of the profile to check the
  software found, kept in its verification.json record as `target_versions`
  with the image it was read from. `source` is `recorded`, `checked` is when
  verify read them, and `image` the image. Re-run verify after changing the
  target's software.
- On `gke_job` (any backend that schedules a Job), verify checks the software
  only with `--smoke-test`, so the releases are recorded only then. Without
  it, as in a scripted setup, `compute versions` reports unknown (once the
  knowledge base reads it, per-release docs then fall back to `latest`). A
  later verify without it keeps the releases already recorded for the
  unchanged profile, with the time they were read.
- The record is kept under the profile's hash, so editing the profile (a new
  `image`, say, or an eval's `TASK_DOCKER` written into a fresh install) leaves
  it behind: `source` is null, and `stale` holds what was recorded, when, for
  which image, and the image now configured. A recorded release never stands
  in for an image it was not read from. Software that changes inside the same
  image tag cannot be seen this way; `checked` dates the answer for that.
- Otherwise: `source` is null and `packages` is empty. The command still exits
  0: an unknown release is an answer, not an error.

### Sessions that never end

A killed session (Ctrl+C, a closed terminal, a crash) never fires SessionEnd,
so each run is also leased to the harness process that started it (`claude`
or `codex`; under any other harness the run goes unleased, and a compute
command run from outside the workspace has no session run to fall back to).
`submit` starts a detached `compute job reap --watch`, which reaps the run's
batch jobs once that process has been gone for 120 seconds, retrying failed cancels for up to
30 minutes, and exits once the run has no batch jobs left. A session resumed
within the window takes its run back. Should the watcher die too (a reboot),
the next session start sweeps them in the background, once it has claimed
its own run, and gives up on a run after three failed sweeps in a row, until
a session claims it again. Both cancel with the workspace and package the run
was started under, so a job is cancelled on the cluster it went to. Serving
jobs are exempt, a run whose owner could not be identified is never reaped
this way, and a lease taken on another host (a shared home directory) is left
to that host. Neither helps if the submitting machine never comes back.

Once a session has ended, `submit` no longer falls back to its run as the
workspace's active run: the job would be reaped as that session's leftover.
Pass `--run-id` instead.

Reaping only ever touches jobs the runtime submitted: it works from the runtime's own
job records, and deletes cluster workloads by their recorded name. Jobs
tracked by pid (`local` and the remote shell backends) record the process's
start time too, and a pid now held by a different process is neither reported
as running nor signalled.

---

## Key Architectural Principles

1. **Self-Registering Drivers (`@register_driver`)**:
   Every driver annotates its class with `@register_driver("<backend_name>")`.
   When `framework/compute/drivers` initializes, `_auto_discover_drivers()` uses `pkgutil.iter_modules` to dynamically discover and import all driver files in the `drivers/` directory. This allows for the easy addition of new drivers without modifying any existing files (especially for internal-only drivers).

2. **Self-Construction & Parameter Validation (`from_params`)**:
   Rather than centralizing validation in `config.py`, each driver implements `from_params(params: dict[str, Any])`. The driver validates its own required parameters (e.g. `gce_tpu_vm` requires `tpu_name`, `zone`, `project`; `gke_pod` requires `pod`; `gke_job` validates `image`, `tpu_accelerator`, `tpu_topology` at dispatch) and raises `ComputeConfigError` with clear error codes.

3. **CLI Inversion of Control (`register_cli_args`)**:
   `cli_parser.py` does not hardcode backend-specific flags. Instead, each driver implements `register_cli_args(parser)` using `cls.add_cli_arg(parser, "--flag", ...)`.
   - Flags are safely deduplicated if shared across backends (e.g. `--zone` and `--project`).
   - The framework automatically tracks registered options into a **positive whitelist** (`REGISTERED_DRIVER_CLI_FLAGS`), so any newly introduced driver argument automatically passes cleanly through to `from_params()`.

---

## Driver Contract (Extending Compute)

To add a new compute driver to the package, create a module in `framework/compute/drivers/<backend>.py` implementing `ComputeDriver`. A large driver can be a package instead (`drivers/<backend>/`), with an `__init__.py` that imports the registered class; see `gke_job/`. For profiles to use it, also add the backend to the package schema's `backend` enum and declare its `SETUP_FIELDS` (see [Valid backends](#valid-backends)). To add a driver without changing the package, see [External drivers](#external-drivers).

```python
import argparse
from pathlib import Path
from typing import Any
from . import base, registry


@registry.register_driver("my_backend")
class MyBackendDriver(base.ComputeDriver):
  # The one declaration of what a complete profile needs. `--target` is
  # marked required in generated docs because it is listed here.
  SETUP_FIELDS = base.SetupFields(required=("target",))

  def __init__(self, target: str, **kwargs: Any) -> None:
    self.target = target

  @classmethod
  def from_params(cls, params: dict[str, Any]) -> "MyBackendDriver":
    target = params.get("target")
    if not target:
      raise base.ComputeConfigError(
          "Backend 'my_backend' requires 'target'.", error_code="MISSING_FIELD"
      )
    return cls(target=target)

  @classmethod
  def register_cli_args(cls, parser: Any) -> None:
    # Do not pass `default=`: see "Flag defaults" below.
    cls.add_cli_arg(
        parser,
        "--target",
        help="Target execution cluster or device",
    )

  def inspect(self) -> base.HardwareProfile:
    ...

  def run(self, code_path: str, ...) -> base.SynchronousExecutionResult:
    ...

  def submit(self, recipe_path: str, ...) -> str:
    ...

  def get_status(self, job_id: str) -> base.AsynchronousExecutionResult:
    ...

  def fetch_logs(self, job_id: str) -> str:
    ...

  def fetch_artifacts(self, job_id: str, dest_dir: Path) -> list[Path]:
    ...

  def cancel(self, job_id: str) -> None:
    ...

  # Optional. Implement only if the target can be tunnelled to; the base class
  # owns what running a forward means (foreground vs `--background`), so a
  # driver supplies the command and a label for its log file and nothing else.
  # Leaving it out makes `compute port-forward` refuse this backend cleanly.
  def _port_forward_command(
      self, local_port: int, remote_port: int, job_id: str | None
  ) -> tuple[list[str], str]:
    ...
```

---

## Configuration Resolution

When an agent or user invokes `{runtime} compute ...`, driver parameters come
from two layers: **CLI flags** (e.g. `--backend gce_tpu_vm --tpu-name node-1`)
override values from the **resolved environment file**. Unspecified CLI
arguments do not override configured values.

The environment file itself is not searched for — it is resolved by how the
package named with `--package` was installed. A workspace holds one package and
so does the user scope, so each install's `environment.json` sits directly in
its `.tpu-agent-suite/` beside the `ownership.json` that names the package:

- **Workspace install**: `<workspace>/.tpu-agent-suite/environment.json`. A
  workspace install is authoritative — if its ownership ledger exists but
  `environment.json` is missing, resolution is a hard error rather than a
  fall-through to central. A ledger naming another package means this one is
  not installed in the workspace.
- **User-scope install** (no workspace install of this package):
  `~/.tpu-agent-suite/environment.json`. It is not version-namespaced, so a bare
  name reaches it as well as a `name@version` tag.

Without `--package` there is no environment file to consult — configuration
comes from CLI flags alone.

### Configuration Format (`environment.json`)

Compute configuration lives in named profiles, each a `backend` and its `params`; `default` names the one a command uses when neither `--profile` nor the run's `compute_profile` picks another. This is the only shape the runtime reads. A fresh install writes `{"revision": 1, "default": null, "profiles": {}}` and otherwise leaves the file alone, and the first `compute setup apply` creates a profile and makes it the default. Each question in questions.json declares where its answer lands in a profile (`compute.backend`, `compute.params.<name>`). Host-side settings that are not compute parameters (`xprof_interpreter`, `trajectory`) sit beside `profiles` at the top level.

```json
{
  "revision": 1,
  "default": "default",
  "profiles": {
    "default": {
      "backend": "gce_tpu_vm",
      "params": {
        "tpu_name": "v4-8-node-1",
        "zone": "us-central2-b",
        "project": "my-tpu-project"
      }
    }
  }
}
```

### Flag defaults

`add_cli_arg` always registers flags with `argparse.SUPPRESS`, and ignores any
`default=` a driver passes. An argparse default would be present in the
namespace on every invocation, and the CLI forwards every registered driver flag
it finds into `param_overrides` -- which outrank `environment.json`. A default
declared here would therefore silently override the user's configured value.

Put backend defaults in `from_params` instead, where they apply only when
neither the CLI nor `environment.json` supplied a value:

```python
zone = params.get("zone", "us-central2-b")
```

Flag destinations owned by the runtime (`--workspace`, `--backend`, `--json`,
`--timeout`, `--code`, `--recipe`, `--name`, `--job-id`, `--out-dir`,
`--interpreter`, `--input`, `--output`, and the REMAINDER captures) are reserved; registering one
raises `ValueError`.

### Discovery failures

`_auto_discover_drivers()` never raises -- one broken driver must not disable
the others -- but failures are recorded in `DISCOVERY_ERRORS` and appended to
the `INVALID_BACKEND` message. A hydrated internal driver whose dependencies are
missing is reported as a load failure with its import error, rather than
appearing as an unknown backend.

### External drivers

An **external driver** lets you add a custom compute backend — for example, for an internal job scheduler or custom cluster wrapper — without modifying this repository or the installed package. Because external drivers live outside the package directory, they stay intact when you upgrade or reinstall the suite.

#### Where external drivers are loaded from

After loading the built-in backends, the runtime searches two locations in order:

1. **`~/.tpu-agent-suite/drivers/`** — Place a driver here to make it available automatically in every session. Even processes that do not load your shell profile (such as Claude Code hooks) will find it here.
2. **`TPU_AGENT_SUITE_COMPUTE_DRIVERS`** — A `:`-separated list of extra paths. Each entry can point to a single `.py` file, a Python package directory (containing `__init__.py`), or a folder of drivers.

When scanning a folder of drivers, the runtime loads every `*.py` file and every subdirectory containing an `__init__.py`, skipping anything whose name starts with `_` or `.`. Put helper modules inside a driver package or prefix their filenames with `_` so the runtime does not treat them as standalone drivers.

#### Writing an external driver

An external driver implements the same `ComputeDriver` interface as a built-in driver and registers itself with `@registry.register_driver("<name>")`:

```python
# ~/.tpu-agent-suite/drivers/my_backend.py
from framework.compute.drivers import base, registry


@registry.register_driver("my_backend")
class MyBackendDriver(base.ComputeDriver):
    SETUP_FIELDS = base.SetupFields(required=("target",))
    ...  # implement the ComputeDriver methods shown above
```

Once loaded, `my_backend` works anywhere a built-in backend works: `--backend my_backend`, `compute setup plan --backend my_backend`, and profiles in `environment.json`.

- **Interactive setup & validation**: `compute setup` automatically accepts your backend name and any fields listed in `SETUP_FIELDS`. Because custom fields do not have entries in `questions.json` or fixed types in the package schema, setup asks a generic prompt (`Value for 'target'`) and leaves type validation up to your driver's `from_params()` method.

#### Troubleshooting load errors

A broken external driver never crashes the runtime or disables other backends. Instead, the failure is recorded in `DISCOVERY_ERRORS` under the file or directory path and shown if someone tries to use that backend (see [Discovery failures](#discovery-failures)). A driver fails to load when:

- Importing it raises an exception (any partial registration from that source is rolled back).
- It finishes importing without calling `registry.register_driver(...)`.
- It tries to register a backend name that is already taken (for example `gke_job`) — the existing driver keeps the name.
- Two external drivers have the same filename or package directory name — the first one found stays loaded.
- A path listed in `TPU_AGENT_SUITE_COMPUTE_DRIVERS` does not exist on disk.

#### Security and privacy

- **Only load trusted code**: External drivers are imported with your user permissions on every runtime command. Only put code you trust in `~/.tpu-agent-suite/drivers/`, and never point `TPU_AGENT_SUITE_COMPUTE_DRIVERS` at untrusted directories, downloads, or job outputs.
- **Keep private details out of driver output**: Text returned by a driver — including `describe_target()`, error messages, and job logs — is visible to the agent and can be copied into reports, commit messages, or exported trajectories. Avoid returning secrets or sensitive internal details if commits or trajectories leave your environment.

### Valid backends

The driver registry is the only source of truth for which backends `--backend`
can name. A compute profile also needs its backend in the package schema's
`backend` enum (the one `compute setup apply` validates against); without it,
every profile on that backend is `COMPUTE_NOT_CONFIGURED`, and the backend
works only through `--backend`. External drivers are the exception: setup adds
their backend names and setup fields to the schema when it loads it (see
[External drivers](#external-drivers)). `--backend` is intentionally not constrained by an argparse `choices` list:
validation happens in `resolve_driver`, which returns a structured
`INVALID_BACKEND` payload listing the registered backends and any driver that
failed to load.
