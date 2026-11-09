# Task: serve Mistral-Small-24B-Instruct-2501 on TPU via torch-tpu

Onboard Mistral-Small-24B-Instruct-2501 (https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) to run as an
inference server on Cloud TPU, using the torch_tpu backend.
Model: Mistral-Small-24B-Instruct-2501

## Definition of done

You are done when all eleven hold:

1. A vLLM server is running on TPU hardware serving the model.
2. It answers a completion request over HTTP with coherent output, and you
show the request and the response.
3. You report measured performance in markdown tables (format below) — real
numbers from your run, not estimates.
4. You give the exact commands to reproduce the result from scratch.
5. Your code changes are a series of local git commits, and the log reads as
the story of how you got there.
6. Every artifact is retained locally, with versions pinned (see below).
7. Within the harness, use Opus 5 (1M) context do this task.
8. Use the TPU slice your environment is configured for. Check the
environment for the correct cluster.
9. All jobs that are launched on the cluster MUST be recorded in job_log.md in your artifact directory (use /logs/artifacts/job_log.md only if your agent environment does not already specify an artifact directory). Peeking into jobs you did not start is considered cheating. For each of those jobs record when it was started and when you terminated it in case you did.
10. All job names should be prefixed `golden-mistral-<run-id>-<customize as needed>`, where `<run-id>` is a short identifier from `$HOSTNAME` (e.g. `$(hostname -s | sed 's/-env$//' | tail -c 8)`) (Kubernetes names allow no underscores)
11. Before you finish, tear down every job you started, the final server included, and record when each one ended in job_log.md in your artifact directory (use /logs/artifacts/job_log.md only if your agent environment does not already specify an artifact directory).

If you cannot reach a running server, say so plainly and report how far you
got, what blocked you, and what you ruled out. A truthful partial result is
worth more than a claimed success.

## Artifacts and versions

Everything you produce must survive the run. A result that cannot be inspected
or reproduced afterwards does not count.

- **Keep artifacts on local disk**, under one directory, organised and named so
a reader can tell what each file is. Not scattered in /tmp, not left inside a
container that gets torn down, not only in a scrollback buffer.
- **Retain at minimum**: raw benchmark output (not just the summary numbers),
server logs, any generated or patched source, the serving config, and any
profiles you captured.
- **Pin every version you depended on**, recorded in one file alongside the
artifacts:
- container image — the **digest**, not just the tag
- torch_tpu build / wheel version
- vLLM version
- Tokamax commit SHA, if used
- Hugging Face model revision (the commit hash, not `main`)
- TPU runtime / accelerator type and topology
- **Every benchmark row must be traceable** to the raw output it came from and
the version set it ran against. State which.
- Do not commit credentials or model weights. If ADC or a token is mounted,
keep it out of the artifact directory and out of git history.
- At the end of the task run a artifacts organization step, where
your artifact directory (or /logs/artifacts only if your agent environment does not already specify an artifact directory) is organized with all generated code, profiles,
relevant logs as so on. Ready to be zipped for analysis.

## Benchmarks to run

Benchmark the running server over HTTP with a serving benchmark, using random
prompts of fixed length. Every request must generate exactly its output
sequence length, not stop early at end-of-sequence. Measure both ISL/OSL
(input/output sequence length) combinations, where 1k is 1024 tokens and 8k
is 8192:

| ISL/OSL | ISL (tokens) | OSL (tokens) | Stresses |
|---|---|---|---|
| 1k/1k | 1024 | 1024 | balanced |
| 8k/1k | 8192 | 1024 | prefill |

For each ISL/OSL combination, sweep the number of concurrent requests to find
where the slice is fully utilized: start at 1 and double (1, 2, 4, 8, ...)
until output throughput improves by less than 10% over the previous step, or
requests start failing or being preempted. Send at least 4x the concurrency in
prompts at each point so it reaches steady state.

Configure the server for the sweep: it must accept sequences of at least 9216
tokens so the 8k/1k combination fits, and it must be able to schedule as many
concurrent sequences as the highest concurrency you test. Make sure no prefix
caching is used while benchmarking with random inputs. Record the serving
configuration with the artifacts. If an ISL/OSL combination or concurrency
cannot run, put it in the failure table instead of dropping it.

## Reporting benchmarks

Report numbers in tables, not prose. One row per (ISL/OSL, concurrency)
point you measured.

| Config | Topology | dtype | ISL/OSL | Concurrency | TTFT (ms) | ITL (ms) | Throughput (tok/s) |
|---|---|---|---|---|---|---|---|

Rules for the table:
- Every bench config is its own job, named with the prefix `golden-mistral-<run-id>-bench-`. Only touch cluster resources you created.
- TTFT and ITL are medians; Throughput is output tokens per second across all
  requests. Mark the peak-throughput row for each
  ISL/OSL combination.
- Every number is measured. If a cell is an estimate or a single sample, mark
  it. If you did not measure it, leave it blank — never fill it in plausibly.
- State how many runs each number averages, and the variance if you have it.
- If you measured a baseline (CPU, or the model before your changes), give it
  its own row so the comparison is visible rather than asserted.

Put failures in a second table — what you tried, what happened, whether you
resolved it:

| Attempt | Config | Outcome | Resolution |
|---|---|---|---|

## How to work

- **Commit as you go, locally.** Each commit should be one coherent step that
  builds or runs. Do not deliver one large commit at the end.
- **The commit message carries the reasoning** — what changed and why it was
  necessary. This is where the narrative lives.
- **Comment the flow in code where it is not obvious**: why a kernel was
  swapped, why a shape is padded, why an op was replaced. Explain *why*, not
  *what* — a line restating the code is noise. Keep it short.
- No need to push anything. Local history is the deliverable.

## Hard constraints

- **Use torch_tpu.** Do NOT use torch_xla / pytorch_xla, and do NOT use the
  tpu-inference repo. Both are older paths and are out of scope. If you find
  yourself importing either, stop and reconsider the approach.
- **Do not reuse existing work.** Write everything this task needs yourself,
  except for packaged patches from the `vllm_patches` directory (use a matching
  patch if present; the directory or a patch for this model may not exist).
  - You MAY read and use: the Hugging Face model card and config, the
    torch_tpu source inside the provided container, Tokamax, public vLLM/TPU
    documentation, and packaged patches from the `vllm_patches` directory (if
    available).
  - Except for packaged patches from the `vllm_patches` directory, you MUST NOT
    read or copy onboarding code, scripts, notebooks, configs, or output
    artifacts that already exist on any VM, disk, bucket, or container you did
    not create during this task. Reading such material is a FAIL, even if you
    do not copy from it.
  - If you encounter pre-existing work, stop reading it and say so in your
    report.
- **Do not contaminate pre-existing containers.** Create your own. Do not
  install into, modify, or exec against a container you did not start.
- **Name every job with the `golden-mistral-<run-id>-` prefix**, so runs are identifiable (Kubernetes names allow no underscores).

## Execution Etiquette

- Proceed without checking in; do not stop or wait for user confirmation.
- Never run shell sleep commands or polling loops of your own; wait only with bounded wait commands that time out and return on their own (such as a job wait with a timeout).
- A full cluster is not a failure. If a submitted TPU job is queued waiting for capacity, leave it queued and keep re-running the bounded wait until it is admitted.

## Environment

- torch_tpu is not public. It is packaged in the container image specified by
  `$TASK_DOCKER` in your environment. Resolve it to a digest before you start,
  and report the digest you ran against.
- Inspect your environment before assuming what is available: check for TPU
  access, a GKE cluster, and quota. Authenticate with ADC.
- Prefer a clean GKE setup. A TPU VM is acceptable if the environment is clean.
- Size the topology for a 24B model yourself and state your reasoning.

## Allowances

- vLLM is the intended inference engine.
- You may use existing Pallas kernels where they apply. Tokamax has some:
  https://github.com/openxla/tokamax

## Reporting

Cite the source for each non-obvious decision — which doc, which file, which
measurement. Where you inferred or guessed, say that you did.
