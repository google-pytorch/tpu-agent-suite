# TPU agent suite

> **Design:** [Overview](README.md) · [Package](package.md) · [Framework](framework.md) · [Runtime](runtime.md) · **TPU agent suite**

The TPU agent suite is the package in `package/`, released as
`tpu-agent-suite`. A user starts with one skill, `tpu-agent`, for all TPU
work. Behind it, specialist prose units move PyTorch models onto Cloud TPU,
write and tune Pallas kernels, profile workloads with XProf, and answer TPU
questions with a citation.

## In one minute

- The package releases skills, and a user can start any of them. For TPU work
  the one to start is `tpu-agent`: it can read and write files and ask the
  user questions, and it routes every request to a specialist rather than
  doing the work itself, because a router that runs workloads tends to do
  the work instead of delegating. Its prompt limits its shell access to
  checking the TPU and environment before routing (query understanding). A
  second skill, `tpu-agent-export-trajectory`, saves a session for later
  diagnosis. Behind tpu-agent sit specialist agents. [The specialists](#the-specialists)
  lists them and what each one does.
- Nobody yet knows the best way to build these agents. So we build competing
  versions side by side and let evals decide which one ships.
- A tenet for every agent we develop: the performance numbers it claims come
  from runs on real TPU hardware. When a number is an estimate, the agent says
  so; when it could not measure something, it reports that rather than
  guessing.

## What the suite does

- **Onboard** a Hugging Face PyTorch model to Cloud TPU using `torch_tpu`:
  benchmark it against a CPU or GPU baseline, or deploy it as an inference
  service.
- **Write and tune kernels:** turn one computation, given as JAX, PyTorch or a
  written description, into a fast Pallas kernel.
- **Profile** a workload with XProf and report exactly the metrics asked for.
- **Answer** questions about TPU hardware and software, citing a public URL.
- **Work in any supported harness** the user already has: Claude Code, Codex,
  Antigravity or Cursor.

## The specialists

| Specialist | What it does | How it works |
| --- | --- | --- |
| tpu-model-onboarding | The onboarding workflow: check the hardware, find candidate models, run preflight checks, benchmark or deploy, and report. | Follows a fixed checklist. Each step's details live in a reference. |
| tpu-unwiring | Replaces CUDA-only code in a model with native PyTorch, and proves the patch works by running it. | Patches, runs and fixes until the model runs. |
| tpu-kernel-development | Converts one computation into a fast Pallas kernel. | A loop with a fixed iteration budget, keeping its state in a file. |
| tpu-diagnostics | Captures and reads XProf traces. | Returns exactly the metrics requested. |
| tpu-knowledge-base | Answers a factual question with a citation. | Looks the question up in an index of public sources, fetches one page, and returns its URL. |

## The principle: build more than one, measure, ship the best

> Nobody yet knows the best way to build a good agent. Rather than choose one
> design on a whiteboard, we build more than one, run them on the same tasks,
> and ship the winner.

- **A new idea is cheap to try.** Every prose unit follows the same contract,
  so a new approach arrives as a new prose unit rather than a rewrite.
- **Evals decide.** Harbor runs the same tasks against every version.
- **A control arm keeps the evals honest.** It uses the same image and tools
  with no package installed. A version has to beat the plain harness model
  (without TPU agent suite), not just work.
- **A release is defined by a file.** `package.release.json` names the released
  skills, and a user gets exactly what those skills reach. Experiments can live
  in the repository without reaching any user.
- **What we avoid:** a design document that blocks a working prototype, or a
  rewrite proposal before the current version has an eval score.

## Knowledge

Every claim the agents make about TPUs must cite something a reader can check.

- **tpu-knowledge-base ships with every install.** Its index maps keywords to
  public URLs, along with the prompts and templates it uses to answer. The
  index and cached pages live under `data/tpu-knowledge-base/`. A memory block
  tells every session doing the suite's work to ask tpu-knowledge-base for TPU
  facts rather than answer from memory.
- **The citation comes from a live fetch.** tpu-knowledge-base fetches one page
  during the session and cites that URL. The failure it exists to prevent is a
  URL the model recalls from memory without fetching it, which may be wrong or
  may not exist.
- **Some reference pages ship directly.** Model onboarding recipes, Pallas
  kernel guides, tuning heuristics, and step-by-step agent procedures live
  under `references/` so agents can read them in full.
- **The curated knowledge base does not ship.** It stays in the repository, in
  `data/knowledgebase/`, and CI still validates it. Whether to keep it has not
  been decided. We will likely use a subset of the pages as a fallback if there
  is no URL mapping in tpu-knowledge-base.

## Trust

- **The user sees one assistant.** If a user has to learn a specialist's name
  to get work done, that is a bug.
- **Claimed performance numbers come from real TPU runs.** This is the tenet
  every agent is written to. An agent that cannot produce a number on a real
  TPU says so and stops, rather than reporting one it did not measure.
- **Measured values are kept apart from estimates.** tpu-diagnostics reports
  values the profiler measured, and values computed from those measurements
  together with the formula. A metric that no tool measures comes back as
  `null` rather than as a guess.
- **Contracts are enforced.** The runtime checks every request and result
  against its schema, using hooks.
- **Long sessions resume rather than restart.** Model access is limited by a
  per-minute token quota, so a long session can be cut off partway through.
  When that happens, the work resumes under the same `run_id` instead of
  starting over.

## Evals and CI

- **Harbor suites** score what the hardware and the tools report, not what the
  agent claims.
- **Eval configs pair the package with a control arm,** on Claude Code and on
  Antigravity, both locally and on GKE.
- **Pull requests that change the package or the eval configs run smoke
  checks** on both harnesses with a small set of tasks.
- **Package CI** runs `verify`, `lint`, a build for every harness, the package
  tests, and the knowledge base validator.
- **Leaderboard** shows the performance on an eval dataset over a period of
  time as we make changes.
