---
name: tpu-hello-world
description: >-
  Dispatches one PyTorch operation onto a TPU node of the configured GKE cluster
  and verifies the result against the same operation on the CPU, then reports
  which kind of device actually did the work.
tools:
  - read
  - shell
accepts: schemas/hello-world-request.json
produces: schemas/hello-world-result.json
delegates: []
scripts:
  - scripts/tpu-hello-world/hello_tpu.py
  - scripts/tpu-hello-world/hello_tpu_with_profile.py
references: []
tpu_dependencies:
  - torch
  - tpu-info
---

# HelloTpu Skill

This skill answers one question: *can PyTorch actually use a real TPU?* It runs
a single operation on the device, verifies the answer against the CPU, and
reports which kind of device really did the work.

The chips live on a GKE cluster, so the work is dispatched onto a TPU node
rather than run locally -- the machine this skill runs on has no accelerator.
Dispatch is the compute framework's job, not yours: `{{runtime}} compute`
creates the Job, mounts the script into the stock image, waits, reads the logs
and cleans up. You choose nothing about how that happens. Your work is reading
the result.

It is self-contained: it runs no model benchmarks and depends on no other
skill.

## Preconditions & Inputs

Everything about *where* the work goes comes from the package's configured
compute environment — view it with
`{{runtime}} compute inspect --package {{package_tag}} --json`. You do not
assemble that configuration and you do not override it — treat it as
read-only and never copy it elsewhere:

- **`backend`** must be `gke_job` or an external cluster backend (not `local`,
  `gce_tpu_vm`, or `gke_pod`). If it is one of those unsupported backends, the
  workspace is not configured for this check -- report that as the blocker
  rather than switching it.
- **`image`, `tpu_accelerator`, `tpu_topology`** name the container image and
  the node pool. **Use the cluster the profile names (its `kube_context`,
  else the kubeconfig's current context) and no other.** Do not go looking
  for other clusters, projects or zones, do not switch kubectl contexts, and
  do not provision any.
- **`queue`** is set only on a cluster that admits work through Kueue (e.g.
  `default`, `priority-dev`, `ubench-regression-tests` on shared capacity clusters).
  Its presence or absence is already correct for the cluster; it changes how long
  admission takes, which matters when you read a timeout. Workload labels
  (`namespace`, `priority`, `declared-duration-minutes`) and Kueue queue labels
  are handled automatically by the driver.

**Build nothing and install nothing.** No image build, no venv, no `pip`.
There is no Docker daemon here, by design. A missing `torch_tpu` *inside the
configured image* is a finding to report, not a problem to fix --
provisioning here would hide the very thing being checked.

--------------------------------------------------------------------------------

## Workflow

Perform these steps in order. **When a step says to report a blocker, that
means: stop, report what failed and at which step, and do not run the steps
after it.** Report what was already established rather than discarding it.

### Step 1: Check the Hardware Is There

```bash
{{runtime}} compute inspect --package {{package_tag}} --json
```

Read three fields before dispatching anything:

- `status: no_accelerator` means the node selector matches nothing in the
  cluster. Report it; a dispatch would sit `Pending` until it timed out and
  tell you the same thing fifteen minutes later.
- `available_chips: 0` means every chip is taken. If the details show a
  `queue`, the dispatch waits in it for its turn. If not, it sits Pending
  with no place in line. Either way, if the run times out, report that the
  cluster was full: a full cluster is not a broken one.
- `available_chips: null` means contention could not be measured -- this
  identity cannot list pods. Not a blocker, but it means a timeout below is
  ambiguous, and your report should say so.

### Step 2: Run the Check

```bash
{{runtime}} compute run --package {{package_tag}} \
  --code {{script:tpu-hello-world/hello_tpu.py}} \
  --timeout 900
```

One command, and it blocks until the answer is back. It dispatches the script
into the configured image, waits for the Job, prints what the pod printed, and
deletes the Job and its ConfigMap on the way out -- including when the run
fails, because a Job left holding a TPU node keeps the next run `Pending`.

The timeout is generous because the image is several GB and a node that has
not pulled it before spends most of the wall clock doing that. The check
itself takes seconds.

The exit status is the script's own: non-zero on any failure, including when
no TPU runtime was found.

If it times out, the reason is in the job status, and there are three
different ones. Do not report "the TPU did not work" without distinguishing
them:

| What you see | What it means |
| :--- | :--- |
| `suspended: true` in the status details | Kueue has not admitted the Job yet (waiting for quota/capacity), or hard time limit (`declared-duration-minutes`) exceeded. The cluster is queued, not broken. |
| Pending with no suspension | Nothing matches the node selector, or the image is still pulling. |
| Running, then cut off | The pod started but did not finish. This one is about the workload. |

A Job Kueue will never admit does not time out: it fails at once, with
`kueue_reason` in the status details and Kueue's own message in the error (a
LocalQueue or ClusterQueue that does not exist, a WorkloadPriorityClass the
cluster does not define). That is the cluster's configuration, not the TPU
or the script: report the message, since resubmitting the same Job changes
nothing. `kueue_reason: Preempted` is different: Kueue evicted the Job for
another workload and requeued it, and the error names the preemptor.

To see it, submit and poll rather than blocking, so the job is still there to
ask about. Pass the `job_id` that `submit` prints to the later commands, not
the `--name`: the backend may prefix or replace the name.

```bash
{{runtime}} compute submit --package {{package_tag}} --recipe {{script:tpu-hello-world/hello_tpu.py}} --name tpu-hello-world-probe
{{runtime}} compute job status --package {{package_tag}} --job-id <job_id> --json
{{runtime}} compute job logs --package {{package_tag}} --job-id <job_id>
{{runtime}} compute job cancel --package {{package_tag}} --job-id <job_id>
```

Cancel it when you are done, even after a failure, and read the logs first:
cancelling a job that is still running deletes the pod and the logs go with it.

### Step 3: Capture a Profile — only if profiling was asked for

Skip this entirely unless the caller asked for a profile or a trace. The check
above answers *did it run*; this answers *where the time went*.

**Establish what is available first.** Two things, and neither should be
assumed:

*Does the cluster have managed ML Diagnostics?* It may or may not, and both are
normal. From the control plane:

```bash
gcloud container clusters describe "$GKE_CLUSTER" \
  --location "$GKE_LOCATION" --project "$GKE_PROJECT" \
  --format='value(managedMachineLearningDiagnosticsConfig.enabled)'
```

`True` means enabled. If that call is refused for scope or permission reasons,
treat the feature as unavailable. Don't query the cluster with `kubectl`
instead: cluster access goes through `{{runtime}} compute` only.

Use it if it is there. If it is not, capture and collect the trace yourself and
say which of the two happened; the trace is the deliverable either way.

*Does your profiling tooling work?* Installed is not working. Exercise it once
before you rely on it.

**Run the profiling script.** Same dispatch as Step 2, different code:

```bash
{{runtime}} compute run --package {{package_tag}} \
  --code {{script:tpu-hello-world/hello_tpu_with_profile.py}} \
  --output xprof.xplane.pb:{{runs}}/<run_id>/artifacts/tpu-hello-world/xprof/hello_tpu.xplane.pb \
  --timeout 900
```

It does what Step 2 does and wraps the matmul in an xprof session, warms up
first, writes the trace under `xprof/` in the job's working directory on the
TPU node (override with `XPROF_DIR`), checks on disk that a non-empty
`.xplane.pb` actually landed, and copies it to `xprof.xplane.pb` beside itself.
Use it rather than writing your own capture: the parts that are easy to get
wrong are already decided in it, and they are the parts below.

**The `--output` is what brings the trace back.** The script runs on the TPU
node, not here, and on GKE that is a pod whose filesystem is deleted with it: a
trace left there is gone when the run returns, however cleanly it was written.
`--output` copies `xprof.xplane.pb` into your run's artifact directory, where it
is kept and collected. Without it the capture succeeds and nothing is left to
read. Read the trace from there after the run.

The session is `torch_tpu._internal.profiler` -- a context manager shaped like
`torch.profiler`, with `xprof_trace_handler` routing the output:

```python
from torch_tpu._internal import profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.TPU],
    on_trace_ready=profiler.xprof_trace_handler(dir_name="xprof"),
):
    ...
```

Two near-misses worth naming, because both look right. This runtime is
`torch_tpu`, not `torch_xla`, so guidance written for PyTorch/XLA names an
import that raises `ModuleNotFoundError` here. And
`torch.profiler.tensorboard_trace_handler` is the shape `xprof_trace_handler`
resembles, not a substitute for it.

There is no public page for this API; the authority is the runtime's own
docstrings, so confirm against `help(profiler.profile)` in the image rather
than against this page. For the capture *model* -- programmatic versus
on-demand, and sessions within a run --
<https://openxla.org/xprof/capturing_profiles> is public and citable, but it
documents PyTorch/XLA, so cite it for the mechanism and not for the call.

**Profile a warm call.** The first execution of a shape compiles, so a trace of
a cold call is mostly the compiler and its device time is not the device time a
loop would see. Run the operation a few times first, then record.

A successful capture writes the standard xprof layout, and that is how you
confirm it worked rather than merely returned:

```
<dir>/plugins/profile/<session_id>/<hostname>.xplane.pb
```

A missing or empty `.xplane.pb` means the region recorded nothing — usually the
operation sat outside the traced block, or was still queued when the trace
stopped. Report that as a failed capture. Do not report a trace you did not
read.

### Step 4: Report the Verdict

Say plainly whether PyTorch can use the TPU, and what that rests on.

**The device label is the finding.** Report it verbatim and say what it means:

| Label | What it means |
| :--- | :--- |
| `native-tpu` | `torch_tpu` drove real hardware. This is the only pass. |
| `cpu-fallback` | No TPU runtime at all. Nothing about the TPU was verified. |
| `cpu-simulated` | `MOCK_TPU=1` was set. A test of this skill, not of a TPU. |

A `cpu-fallback` run is **not** a pass: the arithmetic was done on the CPU, so
the script prints no difference (it says the CPU comparison is not
meaningful) and fails. Say so; do not report a difference for it.
The usual cause is a pod scheduled without TPU chips in its resource limits,
not an image that lacks a runtime -- and since the driver always requests
them, on this path it more often means the topology is multi-host. A single
pod cannot initialise a multi-host slice: `libtpu` wants every worker's
hostname before it will open a device, so it returns nothing and the script
falls through to the CPU.

Report the measured difference alongside the tolerance the script used, not
bare. On real TPU hardware a difference of order 0.1 is normal -- float32
matmuls accumulate in reduced precision on the systolic array -- which is why
the tolerance is 0.5 rather than something tighter. A tpu7x measures 0.113 on
this operation through the native backend. A difference far below that on a
`native-tpu` run is worth remarking on, since it suggests the
work did not land on the MXU.

The script prints its findings as a markdown table. Copy those rows verbatim
rather than restating them from memory.

State what you could not determine. This skill checks one operation on one
node; it says nothing about how many chips the cluster has, how fast they are,
whether multi-host collectives work, or whether the chips are contended.
