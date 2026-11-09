# Framework: from package to running agents

> **Design:** [Overview](README.md) · [Package](package.md) · **Framework** · [Runtime](runtime.md) · [TPU agent suite](tpu-agent-suite.md)

The framework lives in `framework/`. It allows us to develop agents that
can run on multiple harnesses: it checks that a package is consistent, builds
it for each harness, installs it in a way that can be cleanly undone, and
enforces the package's contracts while its agents run.

## In one minute

- A package goes through four stages: **package → compiler → installer →
  runtime**. Each stage runs in a different place and catches a different kind
  of mistake.
- The **compiler** runs in CI. It can see the whole package, including the
  whole delegation graph.
- The **installer** runs on the user's machine. It can see the user's
  workspace and settings.
- The **runtime** ships inside every build and runs during agent sessions. It
  can see each request and result as it happens.
- **Compute** gives every agent one family of commands, `compute …`, that works
  the same way across four built-in backends (plus external drivers).

## Why a framework

A multi-agent package without a framework tends to break in the same ways:

- **Harness drift.** Every harness has its own agent file format, tool names
  and way of delegating. A prompt tuned for one harness is wrong for the next.
- **Delegation nobody can check.** Dozens of agents that delegate to each other
  form a graph too large to verify by reading. Cycles and unreachable agents
  show up only in a live session.
- **Prose as an interface.** When agents pass each other summaries, they start
  inventing results. A contract between agents has to be a schema that a
  machine checks.
- **Compute rewritten in every agent.** SSH, kubectl, job polling and file
  staging end up reimplemented, slightly differently, in each prompt that
  touches a TPU.
- **Installs that cannot be undone.** Files get dropped into a user's workspace
  with no record of what was added or what was there before.

## The pipeline

A package passes through four stages on its way to a running agent:

| Stage | Where it runs | What it works on | What it produces |
| --- | --- | --- | --- |
| Package | the repository | what the author writes | skills, agents, references, scripts, schemas and data |
| Compiler | CI, or the author's machine | the whole package | one build per harness, under `dist/<harness>/` |
| Installer | the user's machine | one build and the user's workspace | skills and agents in the harness directory; everything else in the central directory and workspace directory |
| Runtime | each agent session | each hook event, request and result | runs, artifacts, and compute jobs on the configured backend |

| Stage | What it guarantees |
| --- | --- |
| Compiler | Delegation has no cycles, is at most 3 levels deep, and reaches every agent.<br>Every tool a prose unit uses is declared.<br>Every script, reference and schema a prose unit names exists.<br>Prose names package files only through tokens.<br>Every requirement has a version pin.<br>Only a skill can ask the user. |
| Installer | It never overwrites a file it did not add, or replaces a setting the user already has.<br>The harness version is one the package supports. |
| Runtime | Every request matches the callee's `accepts` schema, and every result its `produces` schema.<br>A completed result points to an artifact.<br>A package cannot use another package's run.<br>No path escapes its root directory. |

Each stage is the last place where a particular mistake is still cheap to
catch. A delegation cycle costs nothing to find in CI and a great deal to find
in a user's session. A file collision, on the other hand, is invisible in CI
and obvious on the user's machine.

## Compiler

`compiler.py` runs in CI and on the author's machine. The file is the
command-line entry point; the build behind it lives in `framework/builder/`.
It has five commands:

- **`compose`** prints the manifest together with everything the compiler
  discovered from it. Use it to see exactly what a build will contain.
- **`verify`** is the CI gate. It checks the delegation graph, tools, files,
  tokens, schemas, hooks, version pins and harness targets.
- **`lint`** checks the prose and scripts for problems `verify` cannot decide:
  broken links, stale paths, unused declarations, and files that no released
  skill reaches. A few of its checks are strict enough to run inside `verify`
  as errors; the rest are advisory.
- **`translate`** builds the package for one harness. It replaces tokens, maps
  portable tool names to the harness's own, and appends a **generated execution
  contract** to every prompt.
- **`archive`** packages a build as a release archive.

The generated execution contract is written from the prose unit's frontmatter,
so an agent's instructions cannot drift from its schemas. There are two
versions of it:

- An **agent's contract** assumes a caller. It tells the agent to require the
  `run_id` in its request, where to write its outputs, which statuses its
  result may return and which fields each status requires, and which agents it
  may call.
- A **skill's contract** assumes the skill is where the session starts. It
  tells the skill to read the workspace's `environment.json` before
  delegating, to keep the `run_id` assigned at session start and include it in
  every request, and which agents it may call.

Either version also gets a **session journal** section when the agent or skill
has a shell: how to record hypotheses, experiments and results with
`runtime.py journal`, in one log per run. No agent's own text has to mention it.

## Installer

`installer.py` runs on the user's machine, once per workspace or once per user.

- **Two scopes.** `project` installs into a single workspace. `user` installs
  into the harness's home directory, so the agents are available in every
  project.
- **Three locations.** Skills, agents and hooks go into the harness's
  directory. If it is a `project` installation, then environment.json and
  ownership.json go into the workspace directory. Everything they read or run
  goes into the central directory, which holds one copy of each version and
  counts how many workspaces use it. If it is a `user` installation, then
  environment.json and ownership.json also go into the central directory.
- **One package per workspace.** environment.json and ownership.json sit
  directly in `.tpu-agent-suite/`, so a workspace (or the user scope) holds one
  package. Installing a different one fails until the first is uninstalled.
- **Uninstalls from the central store.** Each version's
  `framework/installer.py` is copied into its central directory, so an install can be removed without the
  distribution it came from. The ledger names the package and version.
- **Asks nothing.** The installer checks only the Python packages on the
  user's machine. Compute is set up and checked by the first session (the
  `compute-setup` guide, through `runtime compute setup plan|apply|verify`).
- **Merges, never replaces.** Changes to the harness's config files only add
  entries. If an entry would conflict with an existing setting, the install
  stops with an error.
- **Exact uninstall.** `ownership.json` records every file, config key and
  memory block the installer added. Uninstalling removes exactly those, and
  leaves anything the user edited alone.
  Its header (`ledger_schema`, package, version) never changes shape; only
  an installer that knows `ledger_schema` reads the rest, and uninstall goes
  to the installed version's own installer.

## Runtime

`runtime.py` ships inside every build, so an installed workspace never needs a
copy of the framework's source. It dispatches hooks, creates the run for each
session, checks every request and result against its schema, runs workloads on
the configured compute backend, keeps long-running studies, and exports a
session's trajectory. [Runtime](runtime.md) describes each of these.

## Harness targets

A harness is supported when the framework can build for it and the package's
`package.release.json` declares a range of versions for it, under
`compatibility_targets`. The ranges change with each harness release, so they
are kept there rather than repeated here.

| Harness | Agent file format | Delegation tool | Config the installer writes |
| --- | --- | --- | --- |
| Claude Code | `agents/<name>.md` | `Agent` | `.claude/settings.json` |
| Codex | `agents/<name>.toml` | `spawn_agent`, which has not yet been validated end to end, so `translate` prints a warning | `.codex/config.toml` |
| Antigravity | `agents/<name>.md` | `invoke_subagent` | none |
| Cursor | `.cursor/rules/<package>.mdc` and `skills/`; the rule file inlines each skill that delegates | `Task` | `.cursor/hooks.json` |

- Tool names and hook event names for each harness live in two tables,
  `tool-mappings.json` and `event-mappings.json`. Each harness has one
  adaptor module under `framework/harnesses/`; a test keeps the adaptors
  and the tables naming the same harnesses.
- Each harness has a parity test suite. It checks that every prose unit is
  built, that every prompt has its execution contract, and that no token is
  left unreplaced.

## Guarantees

Beyond what each stage guarantees, these hold for the framework as a whole, for
every package on every harness.

- **Contracts are generated.** The execution contract in every prompt is
  written from frontmatter, and an agent's `tools` are enforced as an upper
  bound.
- **Installs are reversible.** The install record lists every change, and
  uninstalling reverses exactly those changes.
- **The runtime is self-contained.** Every build carries its own copy.
- **Extensions are additions, not rewrites.** A new compute driver, harness
  mapping, built-in hook or lint check is one new entry.
