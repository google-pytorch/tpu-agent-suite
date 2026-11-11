# Runtime: what the framework ships to the user's machine

> **Design:** [Overview](README.md) · [Package](package.md) · [Framework](framework.md) · **Runtime** · [TPU agent suite](tpu-agent-suite.md)

Most of what an install puts on a user's machine comes from the package: the
skills, and everything those skills reach. A small part comes from the
framework instead, and is the same for every package. This page describes that
part, which we call the runtime: what it contains, who calls it, and what it
does.

## In one minute

- The runtime is a set of Python scripts that every build carries, so an
  installed workspace never needs a copy of the framework's source.
- Three callers use it: the harness, through **hooks**; agents, through
  commands in their instructions; and the user, through skills such as the
  trajectory export.
- It offers six things: hook dispatch, runs, contract checks, compute,
  studies and trajectory export.
- A package configures it: the package's hooks, schemas and environment settings
  tell it what to check and where to run.

## What ships

`framework/` is one Python package, and every build ships it as is, minus the
build tools that never leave the repository (`compiler.py` and `builder/`, the
linter, the frontmatter parser, `definitions/` and `templates/`). A file has
the same path in the repository and in a build, and its imports have one form,
`from framework import utils`.

| Path | What it is |
| --- | --- |
| `framework/runtime.py` | The command-line entry point for everything on this page. |
| `framework/hooks/` | Hook dispatch: reading the harness's event, running the hook, and replying. |
| `framework/agent_contracts.py` | Contract checks for delegations and subagent results. |
| `framework/run_state.py` | Run namespaces and the session records that point at them. |
| `framework/compute/` | The compute commands and their backend drivers. |
| `framework/study.py` | The store and commands for long-running studies. |
| `framework/journal.py` | The session journal: one experiment timeline per run. |
| `framework/bg_ledger.py` | The per-agent ledger of background work and the idle gate: a subagent does not finish while a child or command it launched is still running. |
| `framework/trajectory/` | The trajectory export. |
| `framework/utils.py` | Shared helpers: schema validation and safe path handling. |
| `framework/paths.py` | Locating the central store, runs, sessions, studies, and installed packages. |
| `framework/schemas/` | The schemas the runtime checks hooks, studies and contracts against. |
| `framework/installer.py` | Installs and uninstalls the build. See [Framework: Installer](framework.md#installer). |

At install time the build's `framework/` directory is copied into the central
store, at `~/.tpu-agent-suite/packages/<name>@<version>/framework/`. None of these
files comes from the package, and none is found by following a skill's
frontmatter. The compiler adds them to every build.

## How it is called

- **By the harness, through hooks.** The build registers the package's hooks
  with the harness. When an event fires, such as a session starting or a tool
  about to run, the harness runs `runtime.py hook-dispatch`.
- **By agents, through their instructions.** A prompt writes `{{runtime}}`,
  and the compiler replaces it with the command that runs the installed
  runtime. An agent runs a workload with `{{runtime}} compute run …`, for
  example.
- **By the user, through a skill.** A skill can be a thin wrapper around one
  runtime command. `/tpu-agent-export-trajectory` is one: its instructions
  tell the model to run a single `trajectory export` command and report what
  it prints.

## What it offers

| Area | Commands | Used by |
| --- | --- | --- |
| Hooks | `hook-dispatch` | the harness |
| Runs | `start-run`, `purge-data` | hooks, the user |
| Contract checks | `validate` | hooks, agents |
| Compute | `compute inspect`, `preflight`, `run`, `submit`, `job`, `port-forward` | agents |
| Compute setup | `compute setup plan`, `apply`, `verify`; `config show`, `set`, `use` | agents, the session-start hook, headless installs |
| Studies | `study start`, `status`, `notes`, `list` and others | agents |
| Session journal | `journal add`, `show`, `render` | agents |
| Trajectory export | `trajectory export` | the user, through a skill |

### Hooks

A package writes each hook once, as a script or a shell command, and it runs on
every harness. The harnesses differ in what they send a hook and what they
expect back, so the runtime sits in between:

- It converts the harness's event into one common shape, described by
  `hook-input.schema.json`, and passes it to the hook on stdin.
- It runs the hook from the workspace root, whatever directory the harness
  started it in.
- It enforces the hook's timeout.
- It converts the hook's decision (`allow` or `deny`, with a reason) back into
  the format the harness expects.

The runtime also provides six built-in hooks that a package can enable in its
`package.release.json`:

| Built-in | What it does |
| --- | --- |
| `start-run` | Creates the session's run and its `run_id`. |
| `validate-schema` | Checks each delegation request against the callee's `accepts` schema, and each agent result against its `produces` schema. |
| `reap-jobs` | When a session ends, cancels compute jobs that it left running. |
| `record-event` | Appends lifecycle events to the study a run belongs to. |
| `skill-tools` | Keeps the main conversation to the active skill's `tools` list. |
| `track-background` | Records each background agent or command an agent launches, refuses a background delegation from an agent with no shell to wait with, and clears the record when the session ends. The stop gate reads the record to tell a pause from a finished subagent. |

A hook can fail without reaching a decision: it may crash, time out, or print
something that is not valid output. Each hook declares what such a failure
means through its `on_failure` setting. The default is `deny`, which suits a
guard, where silence must not count as approval. `allow` suits a logger or a
recorder, which should never block the work. The TPU agent suite sets `allow`
on all of its hooks today. Every hook failure is logged either way.

### Runs

A run is the namespace for one session's work. The `start-run` hook creates it
when a session starts, under `~/.tpu-agent-suite/runs/<run_id>/`, and every
durable output an agent writes goes under that run's `artifacts/` directory.

A session here means the user's top-level conversation. The agents it starts
do not get runs of their own: they work under the session's run, and the
`run_id` is passed to them in each request. If the hook fires again in the
same session, it reuses the run it already created.

- A run id is a UTC timestamp followed by 12 random hex digits, for example
  `20260930T163833Z-7a3b9f12c4d5`.
- The workspace records which run is active, in
  `<workspace>/.tpu-agent-suite/current_run_id`. Which run each session used is
  recorded centrally, under `~/.tpu-agent-suite/sessions/`, so a subagent finds
  its parent's run from any directory.
- The runtime rejects attempts by one package to use a run that another
  package created.
- `purge-data` deletes the run history and the session records, for every
  workspace. It asks for confirmation first.
- `journal` keeps the run's experiment journal, `journal.jsonl` and its
  rendering `experiment.md`, beside `artifacts/`. Every agent with a shell is
  told to use it by its generated contract.

### Contract checks

Agents exchange JSON, and the runtime checks it. The `validate-schema` hook
does this automatically at each delegation: a request that does not match the
callee's schema is refused before the callee starts, and a result that does
not match is sent back to the agent that produced it. The same check is
available as a command, `validate --schema <schema> <file>`, for an agent or a
developer who wants to check a file directly.

Two rules apply to every result. Its `status` must come from the list the
schema allows, and each status names the fields that must accompany it. A
`completed` result must point to at least one artifact file under its run.

### Compute

Agents never write SSH, `kubectl` or job-polling logic into their prompts. They
call one command family, and the run's compute profile decides which backend
runs the work: `local`, `gce_tpu_vm`, `gke_pod` or `gke_job`. The
profile is `--profile` if given, else the one the run uses, else the default
in the package's `environment.json`.

| Command | What it does |
| --- | --- |
| `compute inspect` | Reports the hardware: chip type, topology and health. A value it cannot determine is reported as `null`, never `0`. |
| `compute preflight` | Checks the backend's tools, credentials and target. The session-start hook runs it for a verified profile, and `compute setup verify` runs it first. |
| `compute run` | Runs a script on the accelerator and waits for it to finish. |
| `compute submit` | Starts a batch job and returns immediately. |
| `compute job` | Manages batch jobs: `status`, `logs`, `artifacts`, `wait`, `cancel`, `list` and `reap`. |
| `compute port-forward` | Forwards a local port to a remote workload, for example a model server. |

When a command passes any `--input` or `--output`, the script runs in a fresh
directory on every backend, with its inputs copied in. That is what lets the
same command work locally and on a remote machine. If a declared output is not
produced, the command fails even when the script itself exited successfully.

Each backend is one driver module that registers itself, so adding a backend
does not change the commands above.

### Studies

A run covers one session. Some optimization work lasts much longer, so the
runtime has a second, longer-lived scope called a study.

Studies are used by the tpu-performance-optimization agent, which is still in development.
Most sessions today do not involve a study.

| Level | Term | What it is |
| --- | --- | --- |
| L1 | study | the optimization effort as a whole; the only thing that outlives a session |
| L2 | round | one step of the study: plan, measure, record, stop |
| L3 | arm | one candidate change being tested |
| L4 | execution | one compute job that measures an arm |

A run is not one of these levels. It is the session that carries out a round,
and runs exist whether or not a study does.

- **Where a study lives.** In `~/.tpu-agent-suite/studies/<id>/`, beside `runs/`.
  Its state is in `study.json`. Hooks append what happened to `events.jsonl`.
  Some sections of `study.json` have a size limit; when a round adds more than
  fits, the oldest entries move to `journal.jsonl`, so nothing is lost.
- **A round is meant to be one run.** The study agent does one round and
  exits. Because the next round is a new session, it starts with an empty
  context and reads only `study.json`, whose size is capped, rather than the
  full history. Round 60 therefore costs about as much as round 2. A retry is
  simply a new round.
- **The framework counts rounds by counting runs.** It does not stop a second
  round from being recorded in the same session, but that round would not be
  counted.
- **The round number is computed, not stored.** It is derived from the runs
  attached to the study, so it cannot drift. It counts rounds attempted, not
  rounds that succeeded, so a study whose rounds all fail still reaches its
  stopping rule.
- **A run belongs to at most one study,** so every event lands in exactly one
  study's timeline. The `record-event` hook writes these events, so no agent
  has to remember to.
- **A study needs a protocol before it starts.** The metric, the pass/fail
  gates, the noise floor and the stopping rule are fixed before the first
  measurement.
- **Some wins are refused.** The runtime will not record a win that failed a
  gate, that falls inside the noise floor, or whose metric and gates came from
  different executions.

Agents reach studies only through the runtime. The commands below are written
as you would run them from the repository root; an installed agent runs
the same commands through `{{runtime}}`.

```bash
# Open a study. All four flags are required; --protocol names the protocol file.
python3 framework/runtime.py study start --study-id <slug> \
    --package <pkg> --target <what-is-optimized> --protocol protocol.json

python3 framework/runtime.py study status --study-id <slug>   # the state, as JSON
python3 framework/runtime.py study notes  --study-id <slug>   # a human-readable view
python3 framework/runtime.py study list                       # every study, newest first
```

The remaining commands record progress during a round: `attach-run`,
`record-round`, `claim-arm`, `release-arm`, `set-protocol`, `set-baseline`,
`append-journal`, `events`, `journal` and `close`.

### Trajectory export

A trajectory is the record of a session: what the user asked, what each agent
did, and what came back. When something goes wrong, the user can save it for
later diagnosis with one command, which a skill runs for them.

- **What it collects.** The transcript from the point the package's skill was
  last started, the transcript of every agent beneath it, the run's artifacts
  up to a size limit, and a manifest describing the environment. Work the user
  did earlier in the same session is left out.
- **Where it goes.** To the destination the user chose: a storage bucket
  they own, or Harbor. Sharing is off until the user turns it on; the skill
  asks the first time they export and saves the answer with
  `config set --user-confirmed`. The runtime never picks a destination on its
  own.
- **How it finds the session.** The `start-run` hook records each session
  under `~/.tpu-agent-suite/sessions/`. The export looks up the session's
  record from its `run_id`, so the user does not have to supply a session id.
- **Consent for more than was asked.** If the runtime cannot find where the
  skill's work began, it refuses rather than exporting the whole session. The
  user can then agree to export everything, and the manifest records that they
  did.
- **Harness support.** Reading a transcript is specific to each harness. A
  harness without a reader yet gets a clear refusal, and nothing is saved.

The user-facing details, including the bucket layout, are in
[Using: Saving a session's trajectory](../using.md#saving-a-sessions-trajectory).

## How the runtime finds the workspace

Several of the features above keep state under `<workspace>/.tpu-agent-suite/`,
so the runtime has to know which directory the workspace is.

- **A hook is told.** The harness includes the session's directory in the
  event it sends, and the runtime uses that.
- **A command has to look.** A command an agent or user runs has no such
  event, so the runtime searches upward from the current directory for a
  `.tpu-agent-suite` directory.

Both must arrive at the same directory. Otherwise a command looks for a
session's records in a place the hook never wrote them.
