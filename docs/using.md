# Using TPU agent suite

Once the package is [installed](install.md) and your
[virtual environment](install.md#1-host-prerequisites) is activated, inside the
session you talk to one agent: **`tpu-agent`**. It is activated by its
description, so any natural-language request in the TPU domain reaches it
without an explicit invocation. It reads the request and hands it to the
specialist that owns the workflow; the specialist delegates further as needed.
You never address a specialist directly.

---

## Prompting Guide

The TPU agent suite package lets you make requests to onboard, optimize or perform related tasks on an AI model to run on TPUs via simple natural language commands. For most tasks, there is no specific format you need to follow or skill names to memorize.
That being said, it is important to provide the right information in the prompt to ensure the agent understands your request. The more context and details you provide, the better the agent will be able to assist you.

The following list can be used to help you build your prompt:
- Task description (required): Tell the model what you want to do. Multiple tasks can be chained together (e.g. onboard, optimize, and serve). Example tasks include:
  - Onboarding/Porting to TPUs
  - Optimizing/Accelerating on TPUs
  - Setting up the model for inference serving on TPUs
  - Benchmarking on TPUs
  - Checking that compute is set up correctly
  - Querying information about the model or environment
- Environment configuration: users are responsible for configuring the environment that their model needs on the compute backend (where the TPUs are provisioned). If your compute profile names a container image, it is used automatically. Otherwise, a good starting point is the python dependencies in the distribution's `requirements-tpu.txt`. You could also choose to build a custom docker container, as exemplified in [`eval`](https://github.com/google-pytorch/tpu-agent-suite/tree/main/eval).
- Resource limits: You can specify the maximum number of profiling runs, optimization attempts, etc.
- Output: You can specify the format in which you want the output to be delivered (e.g. markdown, json, etc.). By default, the output is placed in the artifacts directory of the current run, under `~/.tpu-agent-suite/runs/<run_id>/artifacts/`. However, you can manually specify wherever you like.

### Example Prompts

#### Detailed Prompts

We recommend prompts that are as detailed as possible to ensure that the agent performs the task in the way that you want. A good strategy for detailed prompts is to write the prompt in a text or markdown file and reference it in the model's chat box when starting the task. Some detailed prompts that have been shown to work well can be found in [`eval/datasets`](https://github.com/google-pytorch/tpu-agent-suite/tree/main/eval/datasets), for example:

- Onboarding: [`eval/datasets/onboarding/gemma-3-27b-it/instruction.md`](https://github.com/google-pytorch/tpu-agent-suite/blob/main/eval/datasets/onboarding/gemma-3-27b-it/instruction.md)
- Kernel Optimization: [`eval/datasets/nightly-functional/jaxbench-8p-gemm/instruction.md`](https://github.com/google-pytorch/tpu-agent-suite/blob/main/eval/datasets/nightly-functional/jaxbench-8p-gemm/instruction.md)
- Onboarding + Benchmarking: [`eval/datasets/nightly-functional/torch-tpu-benchmark/instruction.md`](https://github.com/google-pytorch/tpu-agent-suite/blob/main/eval/datasets/nightly-functional/torch-tpu-benchmark/instruction.md)

#### Simple Prompts

While we recommend more descriptive prompts, using a simpler style of prompt can also work. The agent may stop more frequently to ask you for clarification or may make what it deems to be reasonable assumptions for aspects of the task that are not defined in the prompt.

**Onboard and serve**
> Onboard Qwen3-32B (https://huggingface.co/Qwen/Qwen3-32B) to run as an inference server on Cloud TPU on vLLM using 2x2x1 v7x TPUs.

**Onboard and optimize, but do not serve**
> I'd like to onboard google/gemma-4-31B-it to run on TPUs, specifically the v6e. Can you do that for me? You can find it on HuggingFace at https://huggingface.co/google/gemma-4-31B-it. After you do that, can you optimize it to be as efficient as possible on that hardware? Use vLLM and this docker image: <link to docker image>.

**Query for information**
> How much HBM does one 4-chip TPU7x VM have in total, in GiB? Save the JSON reply, exactly as printed, to ~/answer.json.

**Request a specific type of optimization**
> The attention layer in my model at `<path/to/model>` is the hot spot in my model and it is slower than I think it should be. Make it faster on the TPU, and show me the measurements that justify what you changed.

You can find more example prompts in the [`eval/datasets/`](https://github.com/google-pytorch/tpu-agent-suite/tree/main/eval/datasets) directory.

### Prompting Tips
- If the agent requires authorization on HuggingFace, make sure your environment exports `HF_TOKEN` before launching the prompt.
- If the agent needs additional information, it will ask for it. Watch for clarification questions for a few minutes after it starts.

## What each request routes to

| You want to... | Routed to | Example |
| :--- | :--- | :--- |
| Run a HuggingFace PyTorch model on TPU, benchmark it, or serve it | `tpu-model-onboarding` | *"Onboard Qwen/Qwen3.8-27B to run as a vLLM inference server on Cloud TPU using the torch_tpu backend. Send it a test completion request, then benchmark it and report TTFT, ITL and throughput in a table."* |
| Optimize a custom Pallas TPU kernel | `tpu-kernel-development` | *"Optimize this custom Pallas flash-attention kernel for TPU v5e and run a self-refinement benchmark loop."* |
| Analyze an XProf profile, or capture one | `tpu-diagnostics` | *"Analyze the XProf profile at /path/to/profile and identify memory bottlenecks and TPU compute utilization."* |
| Get a cited fact about TPU hardware or runtimes | `tpu-knowledge-base` | *"How much HBM does a TPU v6e chip have? Cite the source."* |

---

## Where things run

Workloads run on the compute backend of the run's compute profile (`local`,
`gce_tpu_vm`, `gke_pod`, or `gke_job`), kept in
`.tpu-agent-suite/environment.json`. Don't edit that file by
hand: the agent sets profiles up with you through `compute setup`, and
`config show` / `config use` list and switch them. Each session starts by
checking whether the profile is set up and verified. If it is not, the agent
offers to set it up before any TPU work: it asks its questions in one round,
with suggested answers, then saves and verifies the profile, asking before
a smoke-test Job that takes minutes (you can say skip). A verified profile
is checked again at session start, and the next
compute command checks it after the file changes; run `compute preflight` to
check it yourself.

Host-side work (HuggingFace search, Vertex AI calls, XProf analysis) runs on the
machine your harness is on, using `python3` from the environment your harness
was started in (unless `xprof_interpreter` says otherwise for XProf).

---

## Model weights

Weights are downloaded on the compute target, where the model runs, not on
your machine.

**What you'll see.** Before the engine starts, the job downloads only the
files the model loads: the weight shards the model's index lists, plus its
config and tokenizer files. Extra copies that some repos ship for other
runtimes, such as `original/` and `metal/` or a `consolidated.safetensors`,
are skipped. The job's log shows `WEIGHTS PROGRESS` lines with the amount
done, the rate and an estimate of time left, then `WEIGHTS READY`. A slow
first start is usually two waits in a row: the download, then the compile.

**Where they're kept between runs.** Files already in the Hugging Face cache
are not downloaded again. Whether the cache survives depends on the backend:

| Backend | Cache | Kept between runs? |
| :--- | :--- | :--- |
| `local`, `gce_tpu_vm` | `~/.cache/huggingface/hub` on that machine, for the user the job runs as | Yes. Nothing to set up. |
| `gke_pod` | The pod's own filesystem | While that pod lives; lost when it restarts. |
| `gke_job` | Empty in every Job | Only with a shared cache claim (`hf_cache_pvc`). |

**What you need to do.**

- **Leave room on the disk.** Large models take tens of GB each, and nothing
  deletes them for you. When a disk fills, jobs fail with
  `No space left on device`. Delete the `models--<org>--<name>` folders you no
  longer need from the cache directory above.
- **On `gke_job`, set up a shared cache.** Without one, every Job downloads
  the full model again. Ask a cluster admin for a ReadWriteMany
  PersistentVolumeClaim in the Job's namespace, then run
  `config set hf_cache_pvc=<claim name>`. Setup and what the checks report
  are in
  [Shared model cache on `gke_job`](install.md#shared-model-cache-on-gke_job-hf_cache_pvc).
- **If the weights are already on the target,** for example on a local disk or
  a mounted bucket, give the agent that path instead of the HuggingFace model
  id. Nothing is downloaded, and the agent should not copy weights that are
  already there.

---

## Where outputs land

Each harness session allocates a run under `~/.tpu-agent-suite/runs/<run_id>/` (a
`SessionStart` hook does this). Durable outputs (generated benchmark scripts,
profiles, reports, kernel variants) are written to
`~/.tpu-agent-suite/runs/<run_id>/artifacts/`, and agents pass around paths
pointing into that directory rather than copying payloads. Ask the agent for
the run id, or take the newest directory under `~/.tpu-agent-suite/runs/`, to find
the artifacts afterwards.

Every agent with a shell also keeps the run's journal, beside `artifacts/`:
`journal.jsonl` records each hypothesis, experiment, result and decision with
the evidence behind it, and `experiment.md` beside it is a readable view (a
table of hypotheses with their status, then a timeline). Read `experiment.md`
to see what a session tried and what it found.

Compute jobs started by a run are tracked, and any still active when the
session ends are cancelled by the package's `SessionEnd` hook -- or, for a
session that was killed rather than ended, about two minutes later, or when
the next session starts at the latest. Serving jobs
submitted with `--serve` are exempt, so a service outlives the session that
started it -- it keeps its accelerators until you stop it with `compute job
cancel`, or clear the whole run with `compute job reap --include-serving`
(cluster-level admission bounds like Kueue `declared_duration_minutes` still
apply on top). To cancel or inspect a job manually, use the compute CLI
documented in
[`framework/compute/README.md`](https://github.com/google-pytorch/tpu-agent-suite/blob/main/framework/compute/README.md#command-reference).

---

## Saving a session's trajectory

When something goes wrong -- a compilation error, a bad optimization, a wrong
answer -- you can archive the whole session for later diagnosis:

```
/tpu-agent-export-trajectory "run failed with a compilation error"
```

The note is optional; `/tpu-agent-export-trajectory` on its own works too.

### What is collected

Everything from the point `tpu-agent` was last activated onwards: its own
transcript, every subagent transcript beneath it at any depth, and any artifact
under 10 MB. Work you did earlier in the same session, before `tpu-agent` was
invoked, is **not** included.

Alongside it goes a manifest recording your note, the harness and its
version, the models used, your host OS, and your `environment.json`.

### Where it goes

To the bucket **you** chose (`trajectory.params.bucket`), under:

```
gs://<your-bucket>/tpu-agent-trajectories/<user-id>/<hostname>/<timestamp>-<session-id-prefix>/
    manifest.json     # readable without downloading the bundle
    bundle.tar.gz     # transcripts and artifacts
```

`<session-id-prefix>` is the first eight characters of the session id; the
full id is in `manifest.json`.

Nothing is sent to Google. Sharing is off until you turn it on: the first
time you export, the agent asks whether and where to share, and saves your
answer. If you decline, nothing is exported, and it doesn't ask again unless
you ask it to share. See
[install.md](install.md#3-environment-configuration) for the settings.

### Two things it will ask you about

- **If it cannot tell where `tpu-agent`'s work began** it refuses rather
  than saving, because the export would cover your entire session. It will
  say so and ask; only then is the whole session saved, and the manifest
  records that you consented.
- **On some harnesses trajectory export is not supported yet.** The command
  says so and saves nothing.

The bundle is not redacted. It contains your transcripts and artifacts
verbatim, which is why it goes only to a bucket you control.

---

## Sources

Agents do not answer public TPU facts (hardware specs, supported versions,
defaults) from memory. They ask the `tpu-knowledge-base` agent, which reads the
public page, quotes it and returns the URL to cite; a fact no page states is
reported as not stated. The first answer for a source is generated and cached
under `~/.tpu-agent-suite/tpu-knowledge-base/cache/`, so later asks are fast.

The onboarding playbooks and the Pallas kernel pages are shipped pages,
installed read-only under
`~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/data/tpu-knowledge-base/pages/`.

The curated knowledge base (hardware specs, parallelism concepts, onboarding
playbooks) is kept in the repository under
[`package/data/knowledgebase/`](https://github.com/google-pytorch/tpu-agent-suite/tree/main/package/data/knowledgebase)
and is not installed. To add a source, see
[developing.md](https://github.com/google-pytorch/tpu-agent-suite/blob/main/docs/developing.md#5-contributing-a-tpu-knowledge-base-source).
