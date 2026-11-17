# Design

> **Design:** **Overview** · [Package](package.md) · [Framework](framework.md) · [Runtime](runtime.md) · [TPU agent suite](tpu-agent-suite.md)

This repository has three parts: a **framework** that allows us to develop
agents that can run on multiple harnesses, a **package** of TPU agents built
with that framework, and an **eval** harness that scores the agents. These
pages explain how the framework and the package are designed, and why. If you
want to install, use or develop the agents, the guides in [`docs/`](..) are the
better starting point.

| Part | Directory | What it is |
| --- | --- | --- |
| Framework | `framework/` | The compiler, installer, runtime and compute access layer. It builds a package for every supported coding assistant and enforces the package's contracts while agents run. |
| Package | `package/` | The TPU agent suite, released as `tpu-agent-suite`: its skills, agents, references, scripts, schemas and data. |
| Evals | `eval/` | Harbor suites and configs that score the package, each run beside a control arm that has no package installed. |

## Reading order

1. **[Package](package.md)** describes
   the kinds of file a package is made of and when to use each one.
2. **[Framework](framework.md)** follows a package from source to running
   agents, and explains what each stage checks.
3. **[Runtime](runtime.md)** describes the part of the framework that ships to
   the user's machine: hooks, runs, compute, studies and trajectory export.
4. **[TPU agent suite](tpu-agent-suite.md)** describes what the package does
   and why it is organized the way it is.

## Terms

### Packages and harnesses

| Term | Meaning |
| --- | --- |
| package | A set of prose units and resources released together, described by its `package.release.json`. |
| harness | The coding assistant that runs the agents: Claude Code, Codex, Antigravity or Cursor. |
| build | The package compiled for one harness, under `dist/<harness>/<package>/`. It is what the installer installs. |

### What a package is made of

| Term | Meaning |
| --- | --- |
| prose unit | A skill, an agent, a reference, or a memory block: a markdown file of instructions that a model reads. |
| skill | A prose unit that a person starts. It takes and returns plain text. |
| agent | A prose unit that only a skill or another agent starts. It gets its own context window and exchanges JSON with its caller. Also called a subagent or specialist. |
| reference | A markdown document that a skill or agent loads into its own context when its instructions say to. |
| memory block | Text that the installer adds to the harness's memory file (`CLAUDE.md` or `AGENTS.md`), so it loads into every session. A package keeps each block as a file under `memory/`. |
| resource | Everything in a package that is not a prose unit: scripts, schemas and data. |
| token | A placeholder such as `{{script:tpu-knowledge-base/index.py}}` in a prose unit's text. The compiler replaces it with the real installed path. |
| released skill | A skill listed in `package.release.json`. A build includes only the prose units reachable from released skills. |

### What happens while agents work

| Term | Meaning |
| --- | --- |
| workspace | The directory a session is working in, usually the user's project. An install adds a `.tpu-agent-suite/` directory to it. |
| session | The user's top-level conversation in the harness, identified by the session id the harness assigns. The agents it starts are part of the same session, even though a harness also gives each one an id of its own: an agent id in Claude Code, a separate conversation id in Antigravity. |
| run | The namespace for a session's work. A hook creates it when the session starts, gives it a `run_id`, and gives it a directory for its outputs. A session has exactly one run, and the agents the session starts all work under it. |
| artifact | A file an agent writes under its run's `artifacts/` directory. An agent's result must point to its artifact files; a prose summary on its own is not a valid deliverable. |
| study | An optimization effort meant to continue across sessions. It has a protocol fixed before its first measurement, and it keeps its own state outside any run. It records the runs that worked on it; a run can belong to only one study. |
| protocol | The terms of a study, written down before its first measurement: the one metric a win is judged on, the conditions every number is measured under, how many times each candidate is measured, the noise floor below which a difference counts as a tie, the gates a candidate must pass, and the rule that ends the study. |
| round | One step of a study: plan, measure one candidate, record the result, stop. A round is meant to take one run, and a study counts its rounds as the number of runs that worked on it. |
| arm | One candidate change that a study is testing. |
| execution | One compute job that measures an arm. |

Study, round, protocol, arm and execution are all concepts used by the
tpu-performance-optimization agent for hill-climbing, and that agent is still in
development. We expect most users, for now, to start a session and run one or
two specialists, such as onboarding or kernel authoring. In that case only the
session and its run exist.

How these terms relate:

- **A session has one run.** The run is created when the session starts, and
  the agents the session starts all work under it.
- **A study is worked on by many runs, and a run belongs to at most one study.**
  The framework enforces the second half: attaching a run to a second study is
  refused.
- **A round is meant to be one run.** The study agent does one round and exits,
  so that each round starts with a fresh context and costs about the same as
  the first. The framework counts rounds by counting runs. It does not stop a
  second round from being recorded in the same session, but that round would
  not be counted.
- **A round tests arms, and each measurement of an arm is an execution.**
- **A run exists without a study.** Most sessions are not part of one.

### Where things are stored

| Term | Meaning |
| --- | --- |
| harness directory | Where a harness looks for skills and agents, such as `.claude/` or `.agents/`. It is inside the workspace for a project install, and in the home directory for a user install. |
| central directory | `~/.tpu-agent-suite/`. It holds everything that is not specific to one workspace. |
| central store | `~/.tpu-agent-suite/packages/<name>@<version>/`, the part of the central directory that holds one copy of each installed package version, shared by every workspace. |

## Where files live

An install writes to three places. The rule for choosing between them: if the
harness must find a file itself, the file goes in the harness directory. If a
file belongs to one workspace, it goes in that workspace. Everything else goes
in the central directory, under the package version when it depends on the
version and beside it when it does not.

The central directory, shared by every workspace:

```
~/.tpu-agent-suite/
├── packages/<name>@<version>/     one copy of each installed version (the central store)
│   ├── references/                documents that skills and agents load
│   ├── scripts/                   programs they run
│   ├── schemas/                   the JSON schemas requests and results are checked against
│   ├── data/                      files that scripts read, such as the tpu-knowledge-base index
│   ├── environment/               the settings questions and their defaults
│   └── framework/                 the runtime, and this version's installer, so uninstall
│                                  needs no distribution
├── environment.json               a user install's compute profiles and settings
├── ownership.json                 a user install's record of what it added
├── runs/<run_id>/                 one directory per session
│   ├── run.json                   which package and session the run belongs to
│   ├── journal.jsonl              the session journal: hypotheses, experiments, results, decisions
│   ├── experiment.md              the journal rendered for people to read
│   ├── items/                     delegated work: items.json, and events.jsonl with one record per write
│   └── artifacts/                 everything the session's agents produced
├── runs/jobs/<job_id>/            the record and log of each compute job
├── studies/<study_id>/            one directory per study
│   ├── study.json                 the study's state
│   ├── events.jsonl               what the hooks recorded
│   └── journal.jsonl              entries moved out of study.json when it filled up
├── sessions/<session_id>.json     which run each session used, and the context its hooks cached
├── logs/hook_failures.jsonl       every hook failure, from every workspace
└── tpu-knowledge-base/cache/      pages tpu-knowledge-base has fetched; a package's own cache
```

A workspace, for a project install:

```
<workspace>/
├── .claude/ (or the harness's own directory)
│   ├── skills/<name>/SKILL.md     the skills, in the harness's format
│   ├── agents/<name>.md           the agents, in the harness's format
│   └── hooks and settings         the package's hooks, and the settings the installer merged
├── CLAUDE.md (or AGENTS.md)       the workspace memory file, with the package's memory blocks
└── .tpu-agent-suite/
    ├── environment.json           the compute profiles and settings, set up by the runtime
    ├── ownership.json             every file and setting the installer added, so uninstall is exact
    └── current_run_id             the run of the most recent session
```

A workspace holds one package, and so does the user scope, which is why
`environment.json` and `ownership.json` need no package directory around them.

A user install puts the skills and agents in the harness's directory under the
home directory instead, so they are available in every project, and keeps its
`environment.json` and `ownership.json` in the central directory. The
`.tpu-agent-suite/` directory with the active run is still created in each
workspace where a session runs.

[Package](package.md#where-everything-lives) shows how each kind of prose unit
and resource gets from the repository to these places, and
[Runtime](runtime.md) describes runs, studies and the hooks that write the
session records.
