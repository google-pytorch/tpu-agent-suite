# Converting a skill into a TPU agent suite package

## Overview

This guide walks through converting an existing skill into a TPU agent suite package.
Its worked examples start from a skill written for Antigravity (AGY) before
the framework existed; the same steps apply to a skill from any harness.
The design it follows is in [`docs/design/`](../docs/design/README.md):

- [Package](../docs/design/package.md): the package layout by kind, what each
  unit's frontmatter declares, tokens, and how the compiler discovers units.
- [TPU agent suite](../docs/design/tpu-agent-suite.md): what each specialist
  owns, and why `tpu-agent` routes while `tpu-performance-optimization` drives.

## Step 1. Restructure Your Package

### Overview

To make your existing AGY Agent compatible with the framework, restructure it to match the layout in
[Where everything lives](../docs/design/package.md#where-everything-lives):

1. Every subagent markdown file becomes `package/agents/<agent-name>.md`;
   a user-invocable skill becomes `package/skills/<skill-name>/SKILL.md`.
2. Every reference (non-instructions MD file) goes under
   `package/references/<dir>/`, and every script (Python / code file)
   under `package/scripts/<dir>/`. The directory groups files by the
   family that uses them (`references/tpu-model-onboarding/`, `scripts/tpu-kernel-development/`);
   it does not decide who gets them. A file reaches a skill or agent only when
   that unit, or a reference it reads, declares it in frontmatter.
3. A reference that needs scripts, other references or a delegate of its own
   declares them in its own frontmatter once; every reader inherits them.

### Worked Example: tpu-kernel-development

1. AGY's `tpu-kernel-development/subagents/plan_kernel_agent.md` becomes `references/tpu-kernel-development/kernel-worker/plan-kernel.md`, a reference the kernel-worker follows in its own context
2. The source harness's flat `tools/` directory moves under the skill's own `scripts/` directory: for example, `tools/generate_profile_script.py` becomes `scripts/tpu-kernel-development/assemble_profile_run.py` (used by the kernel-worker's profiling phase).
3. Python dependencies are declared centrally at the package root in `requirements-client.txt` (host) and `requirements-tpu.txt` (remote TPU workloads).
4. Anything that talked to the accelerator directly (AGY's `tools/tpu_client.py` and `server/tpu_server.py`) has no counterpart: sub-agents reach compute through `{{runtime}} compute` instead of shipping their own transport.

> **Note:** Placing a file under `scripts/` or `references/` controls *where it
> lives*, not who gets it. A skill or agent gets what its frontmatter names, plus
> what the references it reads name in theirs. Declaring a directory under
> `references` brings every file under it.

## Step 2. Format SKILLs/Subagents MD Files

### Overview

1. **Inject Frontmatter Metadata (`--- ... ---`)**: Every subagent markdown file must start with leading YAML frontmatter declaring its manifest metadata. When converting existing skills (which typically only declare `name` and `description`), you must **add** the required contract fields for agents that use structured contracts:
   * **`name`**: Unique lowercase hyphen-case identifier (e.g. `kernel-worker`).
   * **`description`**: Human-readable explanation of the agent's role.
   * **`tools`**: Primitive tools permitted for this subagent (e.g. `["read", "write", "edit", "shell", "search"]`).
   * **`accepts`**: Relative path to the input contract schema in `schemas/`.
   * **`produces`**: Relative path to the output contract schema in `schemas/`.
   * **`delegates`**: List of subagent names this agent may invoke (empty `[]` for leaf agents).
   * Optional: `scripts` (executable files under `scripts/`), `references` (files or directories under `references/`), and `host_dependencies` / `tpu_dependencies` (unpinned package names the scripts need; pinned centrally, see Step 5).

2. **Template Agent References**: Replace hardcoded file paths or skill names with TPU agent suite template tokens (`{{agent:<name>}}`). Agent names referenced this way must resolve to a declared agent name in the package.

3. **Name Package Files By Token**: A token is the only way to name a package file in an instruction or a reference. Write `{{script:tpu-kernel-development/assemble_test_run.py}}`, which renders to the installed path `~/.tpu-agent-suite/packages/<name>@<version>/scripts/tpu-kernel-development/assemble_test_run.py`. A path typed out, such as `scripts/tpu-kernel-development/assemble_test_run.py`, is read from the agent's workspace, where the file is not, so `verify` rejects it. Frontmatter lists are declarations, not prose, and keep plain paths.

4. Route all remote execution and accelerator operations through Unified Compute (`{{runtime}} compute`).

   **Never write your own transport commands (`ssh`, `scp`, `kubectl`, `xpk workload`, `gcloud compute`) in prompts or reference docs.** Instead, use the `{{runtime}}` macro, which automatically expands during compilation to the path of the embedded `runtime.py` runner in the target workspace:

   * **Synchronous script execution**:

     ```bash
     {{runtime}} compute run --code <path/to/script.py> [--name <job_name>] [--timeout <sec>] [--interpreter <python>] [--input <local>[:<remote>]]... [--output <remote>[:<local>]]... -- [script args...]
     ```

     *Note: All flags before `--` configure the runtime; everything after `--` belongs to the target script.*
   * **Hardware & connectivity inspection**:

     ```bash
     {{runtime}} compute inspect --json
     ```

   * **Batch & serving (asynchronous) workloads**:

     ```bash
     JOB_ID=$({{runtime}} compute submit --recipe <path/to/recipe.sh> [--serve] [--input <local>[:<remote>]]... [--output <remote>]... [--run-id <run_id>])
     {{runtime}} compute job status --job-id $JOB_ID
     {{runtime}} compute job wait --job-id $JOB_ID [--until running|done|log:<regex>]  # never a sleep loop; exit 3: do what `next` says
     {{runtime}} compute job logs --job-id $JOB_ID
     {{runtime}} compute job artifacts --job-id $JOB_ID [--out-dir <dir>]  # each --output lands at {{runs}}/<run_id>/artifacts/<job_id>/<remote> (or <dir>/<remote>)
     {{runtime}} compute job list [--run-id <run_id>] [--no-refresh]  # --no-refresh: recorded status, no backend calls
     {{runtime}} compute job cancel --job-id $JOB_ID
     {{runtime}} compute job reap [--include-serving]  # --serve jobs survive a plain reap
     ```

   * **Port forwarding**:

     ```bash
     {{runtime}} compute port-forward --port [LOCAL:]REMOTE [--job-id <id>] [--background]
     ```

   Compute target configuration (such as TPU VM details or GKE cluster/pod/job) is automatically resolved from the run's compute profile in `.tpu-agent-suite/environment.json` (or chosen with `--profile <name>`, or `--backend <backend>` plus driver CLI flags).

   > **Do not author generic compute instructions in package prompts**:
   > The compiled accelerator contract (appended by `compiler.py` to every skill and agent prompt in the package) owns generic compute instructions: run vs. submit semantics, dynamic backend listings, interpreter resolution, default artifact directory routing, and job lifecycle commands. Manually authoring generic compute prose in package prompts causes documentation drift and is flagged by `compiler.py lint` (`generic-compute-prose`). Author only the specific execution command required for the agent's task.

   > **_NOTE:_**
   > **How `environment.json` is populated**:
   > The installer asks nothing: it writes `<workspace>/.tpu-agent-suite/environment.json` (or `~/.tpu-agent-suite/environment.json` for user scope) with no compute profiles, and the first session sets compute up from the questions declared in `environment/questions.json` (backend type, TPU VM name, zone, project, or GKE cluster/pod/job), saving the answers as a named profile. For automated or scripted setups, run `runtime.py compute setup apply --profile default --answers <path.json>` after the install. If compute is not set up at runtime, compute commands fail with `COMPUTE_NOT_CONFIGURED`, and the session-start message and the compiled accelerator contract point agents at the shared setup guide (`framework/compute/compute-setup.md`, shipped to every harness as `references/compute-setup.md`), which sets it up with the user through `runtime.py compute setup`. Point package prompts at that guide rather than writing setup steps of their own.

5. **Use Resource Tokens for Installed Assets**: The installer decides where assets land (central store vs workspace), so instructions must never spell an installed location literally — `verify` rejects any `.tpu-agent-suite/...` or `~/.tpu-agent-suite/...` path in authored prose. Name *what* you want with a token and let the compiler route *where*:

   | Token | Expands to |
   | :--- | :--- |
   | `{{package}}` | the central payload directory (`~/.tpu-agent-suite/packages/<name>@<version>`) |
   | `{{runs}}` | the central runs directory (`~/.tpu-agent-suite/runs`) |
   | `{{runtime}}` | the command that invokes the embedded runtime |
   | `{{package_tag}}` | the `name@version` identifier for `--package` flags (identity, not a path) |
   | `{{script:<path>}}` | the installed path of `scripts/<path>` |
   | `{{reference:<path>}}` | the installed path of `references/<path>`, a file or a directory |
   | `{{data:<path>}}` | the installed path of `data/<path>` |
   | `{{schema:<path>}}` | the installed path of `schemas/<path>` |
   | `{{skill:<name>}}` | the name of a skill this build holds |

   These four are the only way to name a package file. `verify` rejects a path under `scripts/`, `references/`, `data/` or `schemas/` that is typed out in prose, and one built on `{{package}}`, and says which token to write instead. A path that is not a package file, such as `<work>/state.json`, is left alone. `verify` cannot tell whether a bare `scripts/train.py` means the package's file or one in the user's workspace, so it rejects it; write a workspace path as one, with something in front of it: `./scripts/train.py` or `<work>/scripts/train.py`.

   A `{{script:..}}` or `{{reference:..}}` token must name something the document that uses it ships: a file in its own frontmatter, a file under a directory it declared, or a file a reference it reads declared. A file that exists but that only another unit declares fails `verify`, because a build that holds this document need not hold that one. The same rule applies to a token inside a reference, against that reference's own frontmatter. A `{{data:..}}` or `{{schema:..}}` token must name something `package.release.json` lists under `copy`.

   Environment configuration deliberately has **no** token — its location depends on install scope, which is unknowable at compile time. Prose that needs effective config directs the agent to `{{runtime}} compute inspect --package {{package_tag}} --json`.

   **Migrating an existing skill**: replace every literal installed path (runs-root literals, version-guessing shell globs like `~/.tpu-agent-suite/packages/<pkg>@*/scripts/...`, any `environment.json` mention) with the matching token, and drop `asset_root` from `workspace_memory` entries — memory blocks render at compile time and may reference central resources only. Run `python3 framework/compiler.py verify package` to catch anything left behind.

### Worked Example: tpu-kernel-development

1. Define subagent metadata in the YAML frontmatter of the markdown file (`package/agents/kernel-worker.md`):

```markdown
---
name: kernel-worker
description: Coordinates one full iteration cycle of kernel optimization.
tools:
- read
- write
- edit
- shell
accepts: schemas/kernel-worker-request.json
produces: schemas/kernel-worker-result.json
delegates:
- analyze-cuda-reference
scripts:
- scripts/tpu-kernel-development/assemble_test_harness.py
- scripts/tpu-kernel-development/assemble_test_run.py
references:
- references/tpu-kernel-development/kernel-worker/plan-kernel.md
- references/tpu-kernel-development/kernel-worker/implement-kernel.md
---

# kernel-worker agent

You are the kernel-worker agent, running ONE iteration of the kernel
optimization loop...
```

2. Replace all subagent file references with `{{agent:<name>}}`, or with
   `{{reference:<path>}}` where the unit became a reference the caller follows
   itself, as tpu-kernel-development's planner and implementer did.

Before (`tpu-kernel-development/subagents/worker.md`):

```markdown
### Phase 1: Planning & Implementation

1.  **Plan the Optimization**:
    -   Invoke
        [`plan_kernel_agent`](subagents/plan_kernel_agent.md)
        subagent. Pass `{base_kernel_path}`, `{test_file_path}`...
2.  **Implement the Kernel**:
    -   Invoke
        [`implement_kernel_agent`](subagents/implement_kernel_agent.md)
        subagent.
```

After (`package/agents/kernel-worker.md`):

```markdown
### Phase 1: Planning & Implementation

1.  **Plan the Optimization**:
    -   Read and follow {{reference:tpu-kernel-development/kernel-worker/plan-kernel.md}}
        yourself. Its inputs are `{base_kernel_path}`, `{test_file_path}`...
2.  **Implement the Kernel**:
    -   Read and follow
        {{reference:tpu-kernel-development/kernel-worker/implement-kernel.md}} yourself...
```

3. Name each tool by its token (e.g. `tpu-kernel-development/tools/assemble_test_run.py` -> `{{script:tpu-kernel-development/assemble_test_run.py}}`):

Before (`tpu-kernel-development/subagents/worker.md`)

```
...

### Phase 3: Test Execution
1.  Assemble the runnable test script:
    `python3 tpu-kernel-development/tools/assemble_test_run.py {test_file_path} {optimized_kernel_path} {test_run_path}`
...
```

After (`package/agents/kernel-worker.md`)

```
...

### Phase 3: Test Execution
1.  Assemble the runnable test script:
    `python3 {{script:tpu-kernel-development/assemble_test_run.py}} {test_file_path} {optimized_kernel_path} {test_run_path}`
...
```

4. Route accelerator execution through `{{runtime}} compute run` instead of legacy client scripts:

Before (`tpu-kernel-development/subagents/worker.md`)

```
### Phase 2: Compilation & Repair Loop

1.  **Loop (up to 6 attempts)**:
    -   Compile the kernel code via `tpu_client.py`:
        `python3 tpu-kernel-development/tools/tpu_client.py --action compilation_test --code_file {optimized_kernel_path}`
```

After (`package/agents/kernel-worker.md`)

```
### Phase 2: Compilation & Repair Loop

1.  **Loop (up to 6 attempts)**:
    -   Compile the kernel code via `{{runtime}} compute run`:
        `{{runtime}} compute run --code {optimized_kernel_path} --timeout 180`
```

## Step 3. Create Expected Schemas

The framework requires formal JSON Schema definitions for agent communication, under `schemas/`.
>

### Overview

Agents that use structured contracts declare an explicit contract for input (`accepts`) and output (`produces`), and `verify` fails if either file is missing from disk:

```json
{
   "$schema": "https://json-schema.org/draft/2020-12/schema",
   "type": "object",
   "additionalProperties": false,
   "required": ["model_id", "run_id"],
   "properties": {
      "model_id": { "type": "string", "minLength": 1 },
      "run_id": {
         "type": "string",
         "pattern": "^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$"
      }
   }
}
```

The validator supports a subset of JSON Schema: `type` (a name or a list such as `["string", "null"]`), `properties`, `required`, `additionalProperties`, `items`, `enum`, `const`, `pattern`, `minLength`, `minimum`, `maximum`, plus the annotations `$schema`, `$id`, `title`, `description`, `default`, `examples`, and `format` (accepted, not enforced). Any other keyword fails `verify`, as does a `schemas/*.json` file that no manifest lists under `schemas`.

### Choosing a schema strategy

Currently, the agents in this package take different (but valid) approaches:

| | Approach | When it fits |
| :---- | :---- | :---- |
| `tpu-model-onboarding` | Subagents use free-form interaction without schema contracts | Subagents take a free-form prompt and return a status/summary. Cheapest to author and maintain. |
| `tpu-kernel-development` | All 3 agents across 3 dedicated request/result schema pairs, the driver included; everything inside one kernel-worker iteration -- planning, implementing, the repair loops, porting, formatting -- is a reference the kernel-worker follows itself | Subagents pass structured state between pipeline stages (kernel artifacts, iteration counts, latencies) where a typed contract catches real errors. |
| `tpu-diagnostics` | One pair, `profiling-request` / `profiling-result`, that types only what the package owns | A shared specialist several orchestrators call. The contract types the status, the failure reason and the artifact paths, and passes XProf's own field names and values through, because those change between nightlies. |

Introduce a dedicated schema pair when subagents exchange structured fields
that a typed contract would actually validate — and note that a pair can be
shared by several subagents doing the same kind of work.

The converse is worth stating too: a typed pair earns its keep only if
something reads the fields. The four summary steps format a record into prose
for a human, and the kernel-worker branches on the stdout it parsed itself rather than
on anything they report — so they are not agents at all, but references under
`references/tpu-kernel-development/kernel-worker/` that the kernel-worker follows in its own context.

### Worked Example: tpu-kernel-development

Agents with structured contracts declare:

* **`accepts`**: A JSON file defining its input contract
* **`produces`**: A JSON file defining its output contract.

`tpu-kernel-development` organizes all 3 of its agents into 3 schema contracts under `package/schemas/`, the `tpu-kernel-development` driver included: it accepts a `kernel-request` from `tpu-agent` and returns a `kernel-result`, so the boundary into the package states what it needs and what it measured rather than passing an untyped prompt. Profiling is not a `tpu-kernel-development` stage at all: after each iteration `tpu-kernel-development` delegates it to the `tpu-diagnostics` package's `tpu-diagnostics` with a `profiling-request`, and gets back a `profiling-result` (see [Step 4.5](#step-45-profiling-ask-tpu-diagnostics-dont-run-xprof)).

| Schema Pair (`accepts` / `produces`) | Subagents Using This Contract |
| :---- | :---- |
| kernel-request.json kernel-result.json | tpu-kernel-development |
| kernel-worker-request.json kernel-worker-result.json | kernel-worker |
| summary-request.json summary-result.json | analyze-cuda-reference |

For example, for the `tpu-kernel-development` `kernel-worker` (`schemas/kernel-worker-request.json`), we have the following request/accepts JSON:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "kernel-worker-request.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["prompt"],
  "properties": {
    "prompt": { "type": "string" },
    "run_id": {
      "type": "string",
      "pattern": "^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$"
    },
    "iteration": {
      "type": "integer",
      "minimum": 1
    },
    "base_kernel_path": {
      "type": "string",
      "minLength": 1
    },
    "work_dir": {
      "type": "string",
      "minLength": 1
    }
  }
}
```

`prompt` is required and is the only required field, on every request schema in
this package. The delegation hook treats `prompt` as routing metadata and
strips it unless the target schema declares it as a property — so a schema
without it is handed an empty object and fails on every required field it has.
Callers dispatch prose naming file paths; the typed
fields stay available for a caller that can supply them, and a caller that
cannot is still reachable.

And the following kernel-worker result example (`schemas/kernel-worker-result.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "iteration",
    "run_id",
    "status"
  ],
  "status_requires": {
    "optimized": ["optimized_kernel_artifact"]
  },
  "properties": {
    "iteration": { "type": "integer", "minimum": 1 },
    "run_id": { "type": "string", "pattern": "^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$" },
    "status": {
      "type": "string",
      "enum": ["optimized", "invalid_input", "failed"]
    },
    "latency_ms": { "type": "number" },
    "speedup_factor": { "type": "number" },
    "optimized_kernel_artifact": { "type": "string" }
  }
}
```

Note what is *not* in `required`. `optimized_kernel_artifact` is owed by the
`optimized` outcome and by nothing else, so it sits under `status_requires`
instead — an iteration whose kernel never compiled has no artifact to hand
back, and a schema that demanded one would leave it no honest way to say so.
The status enum names what this stage produces rather than reusing a generic
`completed`, which said only that the agent stopped.

In each subagent markdown file (e.g. `package/agents/kernel-worker.md`), the frontmatter points directly to its shared contract files:

```yaml
---
name: kernel-worker
accepts: schemas/kernel-worker-request.json
produces: schemas/kernel-worker-result.json
...
---
```

### Where schemas are declared

Only an agent names a schema, and only through `accepts` and `produces`.
Nothing else declares one:

| Kind | Names a schema? |
| :---- | :---- |
| Agent | Yes, in `accepts` and `produces`. |
| Skill | No. A skill takes and returns plain text. |
| Reference | No. Its frontmatter carries scripts, references, delegates and dependencies only. |
| Script | No. A script has no frontmatter of its own. |

There is no list of schemas to maintain. Every `schemas/*.json` is registered
and validated by `verify`, and the whole directory ships with every build
because `package.release.json` names `schemas` under `copy`. A `schemas` field in
`package.release.json` is rejected.

This is looser than scripts and references, which ship only when a released
skill reaches them: a schema nothing names still ships, and `lint` does not
report it. That is deliberate while the only reader is the contract hook pair,
which looks an agent's `accepts` and `produces` up in the build manifest. A
script that needs a schema reads it from the installed `schemas/` directory.
If scripts start doing that, add a `schemas` list to frontmatter, ship only
what is reached, and drop `schemas` from `copy`.

## Step 4. Declare metadata in frontmatter

There is no per-domain manifest. Each skill, agent and reference declares
its own metadata in leading YAML frontmatter, and the compiler gathers the
transitive closure from the skills named in `package.release.json`. Dependencies are
declared where they are needed -- in the frontmatter of the document whose
scripts import them -- as unpinned canonical package names; versions are
managed centrally at the package root (Step 5).

### Frontmatter Fields

Each referenced markdown file must declare its metadata in leading YAML frontmatter (`--- ... ---`):

| Field | Type | Required? | Description |
| :---- | :---- | :---- | :---- |
| **`name`** | `string` | Yes | Lowercase hyphen-case identifier (e.g. `kernel-worker`). Must be unique across the entire package. |
| **`description`** | `string` | Yes | Human-readable explanation of the agent's specific role. |
| **`tools`** | `array[string]` | Yes | Primitive tools permitted for this subagent. Allowed: `["read", "write", "edit", "shell", "search"]`. Grant the minimum the subagent needs: a summarizing subagent that only reads files should get `["read"]`, not the full set. |
| **`accepts`** | `string` | Yes | Relative path to the input JSON Schema in `schemas/`. |
| **`produces`** | `string` | Yes | Relative path to the output JSON Schema in `schemas/`. |
| **`delegates`** | `array[string]` | Yes | List of agent names this agent is allowed to invoke (empty `[]` for leaf agents). |
| **`scripts`** | `array[string]` | Optional | Executable Python files bundled for this agent (e.g. `scripts/tpu-kernel-development/assemble_test_run.py`). Every entry must be under `scripts/`. |
| **`references`** | `array[string]` | Optional | Files or directories under `references/` this agent reads. A directory brings every file under it. Every entry must be under `references/`; a schema or data file is not declared here, it is named by `{{schema:..}}` or `{{data:..}}`. |
| **`host_dependencies`** | `array[string]` | Optional | Unpinned package names the agent's scripts need on the host. |
| **`tpu_dependencies`** | `array[string]` | Optional | Unpinned package names the agent's scripts need in the TPU workload environment. |

A skill's frontmatter (`skills/<name>/SKILL.md`) takes the same fields except
`accepts` and `produces`: a skill takes and returns plain text. A reference's
frontmatter is optional and may carry `scripts`, `references`, `delegates`,
`host_dependencies` and `tpu_dependencies`; whatever it declares is added to
every unit that reads it, and a delegate it names must be an agent under
`agents/` or a released skill. The frontmatter is stripped when the reference
ships.

## Step 4.5. Profiling: Ask `tpu-diagnostics`, Don't Run XProf

**XProf is centralized. Your package never invokes it directly — it asks
`tpu-diagnostics`, which writes a report and returns a `profiling-result`.**

That is the whole rule. The reason is version drift: the XProf CLI's interface
changes between releases, and the guidance documenting it ships *inside* the
xprof wheel, co-versioned with the binary. One agent reads that co-versioned
guidance on every invocation; anything else would restate an interface that
moves underneath it.

### What you get back

A `profiling-result` (`schemas/profiling-result.json`) and a markdown report at
a path you choose — `<artifacts>/<N>/tpu_profiling_report.md` by convention.
The report contains the fields **you asked for**, the top HLO ops by device
time, what the profiler could not see, and ranked levers with the number
justifying each. The result carries the `status`, the report's path as
`profiling_report_artifact`, the XProf version, and `traces`: one entry per
trace read, in the order you sent them, then any it captured. Each entry is
`{path, session_id?, label?, captured, metrics, levers?, tool_outputs,
read_duration_seconds}`, where `captured` is `true` only for a trace this call
captured from your `workload`, `metrics` holds a value for every field you
asked for, and `levers` ranks the trace's bottlenecks, biggest first. With one
trace, read `traces[0]`.

Each metric is `{value, source_tool, unit?, tool_status?, tool_reason?,
provenance?, note?}`, copied from the tool untouched: `value` may be a number, XProf's
own `"N/A"` string, or `null` when no tool produced the field. `tool_status` /
`tool_reason` (e.g. `UNAVAILABLE` / `LLO_DATA_ABSENT`) and `provenance` are
XProf's strings, not enums. The package types only what it owns and passes through
what XProf owns, so a new XProf release does not need a schema change.

Each lever is `{kind, summary, share_pct, source_tool, phase?,
pct_of_roofline?, trace_name?, source_file?, next_step?}`. `kind` is one of the
study's bottleneck kinds (`study.BOTTLENECK_KINDS`) or `other`, and a
`kernel-block` names the file the server runs for it in `source_file`.

Branch on `status`. `reported` means every trace you asked about was read, and
your workload captured if you sent one, even when some fields came back
unavailable; `invalid_input` means no traces and no workload, or a trace that
does not exist; `needs_input` means a capture needs something only you have;
`failed` comes with a `failure_reason` (`trace_empty_or_partial`,
`capture_failed`, `xprof_unavailable`, `read_timeout`). A trace you send is
never replaced by a capture: an empty or partial one fails the call. Only
`reported` has a report to link.

"The fields you asked for" is load-bearing: the analyst reports what the
request names and nothing else, so a metric you forget to ask for simply will
not exist. Name every field you need.

### What you give it

A `profiling-request` (`schemas/profiling-request.json`) JSON object: `prompt`
and `run_id`, which are required, plus **traces** to read (`traces`, each
`{path, session_id?, label?}`), **a workload** to capture (`workload`), or both,
and the report path. Several traces, each with a `label`, ask for a
comparison; traces plus a workload ask for a before/after. A prose-only
delegation is denied at the boundary, because it carries no `run_id`.

```json
{"prompt": "Analyse the XProf profile of run 3.", "run_id": "<run_id>", "traces": [{"path": "<artifacts>/3/tensorboard"}], "report_path": "<artifacts>/3/tpu_profiling_report.md", "fields": ["duty_cycle_percent", "mxu_utilization_percent", "roofline_utilization", "peak_hbm_gib"], "top_ops": 30}
```

`fields` is an open list of names, returned under the same names; which XProf
tool produces each is the analyst's business, not yours.

Hand over a trace whenever you already have one — that half of the contract is
cheaper and more reliable than asking it to capture.

### When you capture the trace yourself

The analyst can capture through `{{runtime}} compute`, but only for a workload
it can construct on its own. If the thing worth profiling needs inputs, shapes,
or a harness that lives in your package, you capture and hand over the
directory. `tpu-kernel-development` is the worked example: `optimized.py` is a kernel
definition, meaningless without that run's `get_inputs()`.

Three things to get right, all of which fail *silently* — the run succeeds and
the numbers are simply wrong:

-   Set the capture flags the XProf skill's `collect_profile` reference gives,
    **before** jax is imported, or the trace contains no kernel-level data. Take
    them from the skill rather than from this page: they are part of the
    interface that moves between releases.
-   Warm up outside `start_trace`, or the first call's compilation is
    attributed to the kernel.
-   Compile exactly the way your benchmark compiles, or the trace describes a
    different executable than the one you timed.

`compute run --output` stages one **file** and a profile run is a
`plugins/profile/<session>/` tree, so pack it into an archive and unpack it
your side. See `scripts/tpu-kernel-development/assemble_profile_run.py`,
`profile_harness_template.py` and the shared `kernel_exec_common.py`.

### Two things CI will reject

| Check | Fails on |
| :---- | :---- |
| `xplane-access` | A script importing `xplane_pb2`, `tsl.profiler`, `xprof.convert` or `tensorboard_plugin_profile` — parsing a trace yourself instead of asking. |
| `xprof-outside-analyst` | Any prompt outside `tpu-diagnostics/` spelling out an `xprof get_*` / `list_*` / `aggregate_*` command. |

Both are the same rule as the heading: don't invoke XProf, ask for a report.

## Step 4.6. Asking the User: Only a Skill Can

**`ask` is a skill's tool and no one else's. The compiler rejects any agent
that declares it.**

That is not a style rule. `tpu-agent` is the only level with a user
listening: when it delegates, it blocks on the result, so a specialist running
underneath has no one to put a question to. Granting it the tool would compile
a promise the harness cannot keep, and the failure is silent -- the agent calls
a tool it holds and nothing answers.

There is also no way to answer a running subagent. Delegation gives a parent
`invoke_subagent`, a `state` bit from `manage_subagents`, and the child's
result when it arrives. All of it points upward. Nothing writes to a child
that has already started.

### Naming the tool portably

Two of the four harnesses have a tool for this, under different names, and two
have none. Write `{{built-in:ask_user_question}}` and the compiler resolves it:

| Harness | Renders as |
| :--- | :--- |
| Claude Code | `AskUserQuestion` |
| Antigravity | `ask_question` |
| Codex, Cursor | `ask the user` |

Phrase the sentence so both readings work -- `"{{built-in:ask_user_question}}
whether to use compile or eager mode"` reads correctly as a tool name and as
prose. `"use {{built-in:ask_user_question}}"` does not.

### What a specialist does instead

**Resolve it before delegating.** The skill collects what is missing and
sends a complete request. An optional field whose description says what to do
when it is unset is the usual shape -- `onboarding-request.mode`:

```json
"mode": {
  "enum": ["benchmark", "inference", "both"],
  "description": "Optional. Omit only when tpu-agent could not resolve the user's intent; tpu-model-onboarding cannot ask, so an unset mode is a question its caller still owes."
}
```

**Or stop and hand the question back.** For a gap the request could not have
anticipated, return `status: needs_input` and end. Every result schema that can
hit this already carries the status -- `onboarding-result`, `study-result`,
`arm-result` and the four `sharding-*` results -- and the generated prompt
names it in the agent's status vocabulary.

Today the question travels in `summary`, since those schemas are
`additionalProperties: false` and none uses `status_requires` to demand more:

```json
{
  "status": "needs_input",
  "summary": "Benchmark mode not specified. Ask whether to run in compile or eager mode, then delegate again saying which."
}
```

Be aware of the cost: the subagent ends. When the answer comes back it starts
from nothing and redoes its work up to that point, so this is a fallback, not a
conversation. Prefer resolving it before delegating.

A structured `questions[]` payload, declared through `status_requires` so the
compiler renders it into every prompt, is a follow-up. Until it lands, nothing
instructs a caller what to do on receiving `needs_input` -- treat handling it
as your agent's business to spell out.

### What CI will reject

| Check | Fails on |
| :---- | :---- |
| `skill-only tool` | Any agent declaring `ask`. |
| `harness-tool-directive` | Any prompt or bundled reference telling the agent to *use* `ask_question`, `AskUserQuestion` or `request_user_input` by name. Use the token instead. |

## Step 5. Update Top-Level Files

Update the following top-level files:

### `requirements-client.txt` and `requirements-tpu.txt`

Central Python dependency catalogs at the package root:
* **`requirements-client.txt`**: Declares all host-side dependencies with explicit version constraints (e.g. `pydantic>=2.0.0`).
* **`requirements-tpu.txt`**: Declares all remote TPU workload dependencies with explicit version constraints (e.g. `torch>=2.4.0`, and the `xprof` floor). `requirements-client.txt` is the single source of the `xprof` floor; a test holds every other copy to it.

All package dependencies declared in any frontmatter must be present in these root catalogs with explicit version bounds. The compiler rejects unpinned entries (an entry with no version constraint) during `verify`. Every build's requirements are trimmed to what the built units declare. What the shipped runtime imports on the host (`google-cloud-storage`, `tpu-info`) is declared in the `tpu-agent` skill's frontmatter, so a `--skills` build that leaves `tpu-agent` out must declare those itself. `--skills` changes which units are built, and so what is kept. Pip option flags (e.g. `-f`, `--extra-index-url`) and comments are preserved.

`framework/compute/env_check.py` checks installed versions against both catalogs: the installer checks `requirements-client.txt` on the installing machine, and `compute setup verify` checks `requirements-tpu.txt` on the compute target where `run` takes seconds, and in a smoke-test Job on `gke_job` once agreed to. It skips lines that name no package (pip options, URLs, paths) and does not evaluate environment markers.

### `package/skills/tpu-agent/SKILL.md`

Add a routing instruction in this file indicating that `tpu-agent` can delegate to your new agent, e.g.:
`- For optimizing Pallas TPU kernels and executing self-refinement optimization loops, delegate to {{agent:tpu-kernel-development}}.`

Also ensure `tpu-kernel-development` is listed in the YAML frontmatter `delegates` array of `skills/tpu-agent/SKILL.md`.

> **Note:** Name the agent with a `{{agent:<name>}}` token, not with prose like `invoke_subagent <name>`.

### `package.release.json`

`package.release.json` is the release manifest. It names the skills the package
releases and what cannot be derived from frontmatter: identity, verbatim
assets, lifecycle hooks, post-install steps, the environment questionnaire
and the harness version ranges. To add a new agent family you normally touch
nothing here: wire the family's orchestrator into `skills/tpu-agent/SKILL.md`
and the compiler finds the rest. Add an entry to `skills` only for a new
user-invocable skill, and to `copy` only for data that no frontmatter reaches.

```json
{
  "name": "tpu-agent-suite",
  "version": "0.1.0",
  "skills": ["tpu-agent", "tpu-agent-export-trajectory"],
  "copy": ["schemas", "environment", "data/tpu-knowledge-base"],
  "workspace_memory": [{"marker": "tpu-knowledge-base", "content": "memory/tpu-knowledge-base.md"}],
  "environment": {"questions": "environment/questions.json", "schema": "schemas/environment.json"},
  "hooks": [],
  "compatibility_targets": [{"harness": "claude-code", "name": "claude-code-2.1", "versions": ">=2.1.219,<3.0.0"}]
}
```

No skill is special, and the order means nothing. Every released skill is a
root of the delegation graph, is listed by the description in its own
frontmatter, and gets its own install rule.

`entrypoint`, `agents`, `include`, `schemas`, `host_dependencies`,
`tpu_dependencies` and `required_capabilities` are no longer read from
`package.release.json`. A manifest that declares one fails `verify` with where that
content lives now.

## Step 6. Declarative Lifecycle Hooks & Action Kinds

### When to Use a Hook

When building agents, developers often rely solely on prompt instructions (e.g., *"Never delete production checkpoints"* or *"Verify working directory before starting training"*). However, LLMs can hallucinate, skip advisory constraints, or execute hazardous actions before prompt rules intervene.

**Lifecycle hooks** provide deterministic, non-bypassable guardrails and automated operations executed directly by the host agent harness (`claude-code`, `antigravity`, `codex`, `cursor`) before or after tool execution:

| Use Case | Hook Event | What It Does | Why Not a Prompt or Subagent? |
| :--- | :--- | :--- | :--- |
| **Safety Guardrails** | `PreToolUse` | Intercept dangerous or unvetted commands (e.g. `rm -rf /`, force push, writing outside sandbox) and deny execution before the tool runs. | Prompts are advisory; hooks are non-bypassable, programmatic, and fail closed by default (`on_failure: "deny"`). |
| **Feedback & Diagnostics** | `PostToolUse` | Run linter/validator after an edit or shell command and inject diagnostic context directly into the conversation. | Guarantees immediate feedback to the agent for closed-loop repair. |
| **Subagent Deliverable Verification** | `SubagentStop` | Audit or validate subagent outputs and artifacts before returning results to the parent coordinator or ending the turn. | Automated quality gates guaranteeing deliverables before parent resumes or session completes. |
| **Environment Bootstrap & Context Injection** | `SessionStart` | Verify environment prerequisites, check hardware/cluster readiness, and inject dynamic system context at session launch. | Executes before user turns begin, enforcing pre-flight requirements deterministically. |

> [!NOTE]
> **When NOT to use a hook**: Do not use hooks for regular agent workflow logic, multi-step problem solving, or tasks that require dynamic LLM reasoning. Those belong in subagent prompts and tools.

---

### Hook Action Kinds

1. **Script Hooks (`action: "script:<path>"`)**:
   * Executes a dedicated Python script shipped inside the package (e.g. `scripts/tpu_safety_guard.py`).
   * The embedded runtime dispatcher runs the script via `python3` with normalized event JSON passed on `sys.stdin` (`event`, `tool`, `args`, `context`).
   * The script inspects the event and writes a canonical JSON decision object to `sys.stdout` (such as `{"decision": "allow"}` or `{"decision": "deny", "reason": "..."}`).
   * **Best for**: Rich validation logic, complex parameter inspection/rewriting, external API checks, or conditional multi-field evaluations.

2. **Command Hooks (`command: "<command>"`)**:
   - Executes a direct shell command line, utility, or pipeline (e.g. `git diff --quiet`, `ruff check .`, or `test -f /tmp/tpu.lock`).
   - Uses standard Unix exit code semantics without requiring JSON boilerplate:
     - **Exit Code `0` (Success)**: Treated as `"allow"`. If stdout is empty, the runtime automatically translates it to a harness-native allow response. If stdout emits JSON, it validates and forwards the decision.
     - **Non-zero Exit Code (Failure)**: Settled by `on_failure`: `"deny"` under the default, with `stderr` as the `reason`. To block under either policy, print a JSON deny instead.
   - **Best for**: Fast pre-flight checks, existing linters/formatters, git status assertions, or command-line verification tools.

---

### Worked Example: TPU Safety Guard Hook

In this example, we add a `PreToolUse` lifecycle hook that inspects shell commands before execution and blocks unsafe or unauthorized TPU reset operations.

#### 1. Declare the Hook in `package/package.release.json`

Add the hook entry under the top-level `"hooks"` array:

```json
{
  "name": "tpu-agent-suite",
  "version": "0.2.0",
  "skills": ["tpu-agent"],
  "hooks": [
    {
      "name": "tpu-safety-guard",
      "event": "PreToolUse",
      "matchers": ["shell"],
      "action": "script:scripts/tpu_safety_guard.py",
      "timeout": 10,
      "visibility": "external"
    }
  ]
}
```

#### 2. Implement the Guard Script (`package/scripts/tpu_safety_guard.py`)

Hook scripts receive normalized JSON on `stdin` and emit a canonical decision JSON object to `stdout`:

```python
#!/usr/bin/env python3
"""TPU Safety Guard: PreToolUse hook preventing destructive host commands."""
import json
import sys

FORBIDDEN_COMMANDS = [
    "rm -rf /",
    "killall -9 vllm",
    "mkfs",
    "dd if=",
]

def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}

    tool = payload.get("tool")
    args = payload.get("args", {})
    command = args.get("command", "")

    if tool == "shell":
        for forbidden in FORBIDDEN_COMMANDS:
            if forbidden in command:
                response = {
                    "decision": "deny",
                    "reason": f"Command contains forbidden operation: '{forbidden}'",
                    "message": "Blocked by TPU Safety Guardrail policy."
                }
                print(json.dumps(response))
                return 0

    # Allow tool execution to proceed
    print(json.dumps({"decision": "allow"}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

## Step 7. Verify and Test
### Overview

First confirm the package is well-formed:

```bash
python3 framework/compiler.py verify package
```

On success this prints `Successfully verified!` and exits 0. It validates the
manifest schema, checks that every referenced instruction/schema/script/
reference file exists on disk, and enforces the delegation rules (no cycles,
max depth 3 below a released skill, every agent reachable).

Then build and run the test suites, updating any that your changes broke:

```bash
make build      # compile distributions for all target harnesses
make test       # framework + package test suites
```

> **Note:** `verify` only checks `{{agent:...}}` tokens — it does **not**
> validate ordinary markdown links or prose inside instructions. After
> onboarding, grep your instruction files for leftover AGY paths
> (`subagents/`, `tools/`, internal origin-tree paths, `experimental/`) and for links to files
> that no longer exist.

Then run the content lint:

```bash
python3 framework/compiler.py lint package
```

`lint` checks that packages follow framework conventions in ways `verify` cannot —
prose, links, paths, and delegation, rather than manifest structure. CI runs it
on every PR and posts findings as annotations.

Several checks are heuristics and can produce false positives (a reference doc
may legitimately *mention* a Bazel path; a backticked path may be
illustrative), so most findings are **advisory** — but you are strongly
encouraged to fix them. Four are merge gates: `instruction-links`,
`reference-links`, `environment-contract` and `workspace-config-paths`.

The ones most likely to fire on a fresh onboarding:

| Check | Catches |
| :---- | :---- |
| `instruction-links` / `reference-links` | A markdown link to a file the agent does not declare, or that does not exist. |
| `legacy-paths` / `legacy-imports` | `subagents/`, `tools/`, internal origin-tree paths, `experimental/` left over from the origin tree. |
| `delegation-consistency` | Prose naming an agent the frontmatter does not list in `delegates`. |
| `workspace-config-paths` | An installed path spelled literally instead of via a resource token. |
| `environment-contract` | A bundled script reading an environment key no question can answer. |
| `generic-compute-prose` | Duplicating generic compute instructions (run vs submit, backend lists, job lifecycle commands) that the compiled accelerator contract already injects. |
| `xplane-access` / `xprof-outside-analyst` | Profiling run directly instead of asked for — see [Step 4.5](#step-45-profiling-ask-tpu-diagnostics-dont-run-xprof). |

### Manual completeness review — the PR author's job

**A green `verify` does not mean the onboarding is complete.** `verify` checks
that the package is *structurally* correct: the manifest matches the schema,
referenced files exist, the delegation graph is legal. It cannot tell that a
subagent, a script, or half a prompt never made it across, because it only ever
sees the converted tree — nothing in it knows what the original contained.

So before opening a PR, the author is responsible for confirming feature parity
against the source package. Read the original and the conversion side by side,
or put a harness to work on it — Antigravity, Claude Code, or whatever you use:

> Compare the original package at `<path>` with the converted package at
> `package/<name>/`. List anything present in the original that is
> missing, truncated, or altered in the conversion: subagents, scripts,
> reference docs, tool grants, and prompt sections.

We recommend recording what you changed or did not port in the PR description.

## Missing framework features

If the conversion cannot preserve a feature because the framework has no equivalent —
a tool category, a delegation pattern, a harness capability, raise it with the Core Framework Team (GitHub ID: prksandeep, learning-to-play, and jrplatin)!

## Step 8. Package Migration Checklist

Before publishing or running your converted package, ensure you have verified the following:

- [ ] **Package Name:** Hyphen-case and lowercase (`^[a-z0-9]+(?:-[a-z0-9]+)*$`).
- [ ] **Graph Depth:** No delegation branch exceeds 3 levels below a released skill (`skill → L1 → L2 → L3`).
- [ ] **Reachability:** Every agent under `agents/` that the release needs is reached from a released skill; `lint` warns about the rest, which do not ship.
- [ ] **Acyclic:** No recursion or cycles in agent delegation lists.
- [ ] **Portable Tools:** All agent `tools` use only valid portable tools (`read`, `search`, `shell`, `write`, `edit`).
- [ ] **YAML Frontmatter:** Every `agents/*.md` starts with valid `---` YAML frontmatter declaring `name`, `description`, `tools`, `accepts`, `produces`, and `delegates`; every `skills/*/SKILL.md` declares `name` and `description`.
- [ ] **Template Tokens:** All markdown instructions use `{{agent:<name>}}` syntax with agent names that resolve in the composed manifest.
- [ ] **Resource Tokens:** No instruction file spells an installed path literally (`.tpu-agent-suite/...`, `~/.tpu-agent-suite/...`); installed assets are referenced via `{{package}}`, `{{runs}}`, `{{runtime}}`, `{{package_tag}}`, and environment config only through `{{runtime}} compute inspect`.
- [ ] **Schema Compliance:** All JSON files in `schemas/` pass JSON Schema draft 2020-12 validation; every one is registered automatically.
- [ ] **Files Named By Token:** Every package file an instruction or reference names is written as `{{script:..}}`, `{{reference:..}}`, `{{data:..}}` or `{{schema:..}}`; `verify` rejects a path typed out.
- [ ] **Scripts & References Registered:** Scripts live under `scripts/`, references under `references/`, and each is declared in the frontmatter of the unit that uses it or of a reference that unit reads (companion `*_test.py` files excluded).
- [ ] **Top-Level Wiring:** `skills/tpu-agent/SKILL.md` routes to the new orchestrator, and its frontmatter `delegates` contains the **orchestrator agent name**.
- [ ] **Lifecycle Hooks:** If declared in `package.release.json`, hooks specify valid `event`, matchers, action/command, and timeout.
- [ ] **No Stale AGY Paths:** No instruction file still references `subagents/`, `tools/`, or other paths, and every markdown link resolves to a file that exists.
- [ ] **Artifact Contracts:** Artifact references are paths located under the run's artifacts directory (`{{runs}}/<run_id>/artifacts/`).
- [ ] **Unified Compute Routing:** All accelerator operations route through `{{runtime}} compute` commands (no raw `ssh`, `scp`, `kubectl`, or `xpk` invocations).
- [ ] **No Generic Compute Duplication:** Prompts do not duplicate generic compute run/submit semantics, backend lists, or job lifecycle prose (`generic-compute-prose` clean).
- [ ] **Profiling Delegated:** No agent runs XProf directly. Anything needing profile numbers declares `tpu-diagnostics` in `delegates`, sends it a `profiling-request` naming every field in `fields`, and reads the `profiling-result` it returns. See [Step 4.5](#step-45-profiling-ask-tpu-diagnostics-dont-run-xprof).
- [ ] **Skill Stands Alone:** for a new skill, `python3 framework/compiler.py verify package --skills <your-skill>` exits 0: the skill and everything it reaches build without the rest.
- [ ] **Linter Clean:** `python3 framework/compiler.py lint package` reports 0 findings for compute contract, legacy paths, or dangling links.
- [ ] **Verification & Tests Clean:** `python3 framework/compiler.py verify package` and `make test` exit with return code 0.
- [ ] **Feature Parity Confirmed:** The converted package has been compared against the original (manually or via a harness) and the result is recorded in the PR description — see [Manual completeness review](#manual-completeness-review--the-pr-authors-job).
- [ ] **Framework Gaps Raised:** Anything the conversion could not preserve has been raised with the Core Framework Team (GitHub ID: prksandeep, learning-to-play, and jrplatin) rather than worked around silently.

