# TPU agent suite framework

The **TPU agent suite framework** provides a domain-neutral, harness-agnostic
foundation for defining, compiling, installing, and executing collaborative
multi-agent packages across target agent harnesses (**Claude Code**,
**Codex**, **Antigravity**, and **Cursor**).

The framework acts as a pure compiler and runtime orchestrator: it enforces
deterministic schema contracts, validates hierarchical delegation graphs, and
guarantees isolated execution namespaces and artifact provenance without
embedding domain-specific prompts or agent logic.

This page is the command and contract reference. The design, and why it is
shaped this way, is in [`docs/design/`](../docs/design/README.md).

---

## Modular Architecture

The framework is organized into distinct, specialized tools and modules:

1. **`compiler.py`** *(CI / Build)*: The CLI that verifies a package and builds it for each harness. The build code lives in **`builder/`** (`package_loader`, `package_discovery`, `package_checks`, `tokens`, `prompt_contracts`, `harness_rules`, `hook_config`, `translators`, `dist_assets`, `verify`, `overlays`). Other code should use its public API (`load_package`, `verify_package`, `translate`, ...), exported from `builder/__init__.py`.
2. **`installer.py`** *(User Host)*: Distribution installation, environment setup, and uninstallation rollback.
3. **`runtime.py`** *(Target Workspace)*: CLI entry point invoked by installed hooks and agents. Implementation is split across `hooks/`, `agent_contracts.py`, and `run_state.py`.
4. **`hooks/`** *(Target Workspace)*: What `runtime.py hook-dispatch` runs for each harness event: `events.py` turns the harness payload into a canonical event, `dispatch.py` filters it and translates the reply, and `actions.py` runs the hook script, command or builtin.
5. **`compute/`** *(Unified Compute)*: Standardized compute interface for the framework, centralizing workload execution and hardware inspection across backends (local host, Cloud TPU VMs, GKE pods/jobs/clusters) so packages do not need to re-implement this. See [compute/README.md](compute/README.md).
6. **`study.py`** *(Target Workspace)*: The study store and its `study` CLI verbs. A study outlives the session that opens it, so its state, lifecycle events and command surface live apart from the runtime that spends runs on it.
7. **`journal.py`** *(Target Workspace)*: The session journal and its `journal` CLI verbs: one experiment timeline per run that every agent with a shell writes, rendered to `experiment.md`. The compiler injects the rule for using it into those agents' contracts.
8. **`bg_ledger.py`** *(Target Workspace)*: The per-agent ledger of background work and the idle gate built on it: records each launch, tells a stop made with the agent's own work live from a finished one, and refuses it on a run the harness will not resume. Wired in by `hooks/`.
9. **`harnesses/`** *(Shared)*: One adaptor module per harness (`google.py` holds what the Google family shares) and their registry. An adaptor holds a harness's identity -- name, display name, config directory -- which block of the mapping tables it reads, and the inbound, dispatch, and outbound sides of a hook call (`canonical_event`, `before_dispatch`, `after_dispatch`, `render_reply`); the harness-keyed lookups (`harness_mapping`, `to_harness_event`, ...) live here too. Per-harness compile, install and trajectory logic still sits in `builder/`, `installer.py` and `trajectory/`, and is being moved into the adaptors.
10. **`utils.py`** *(Shared)*: Common semver, JSON schema validation, safe path resolution, and configuration merging.
11. **`jsonio.py`** *(Shared)*: Reading JSON and JSONL files, and atomic file writes.
12. **`paths.py`** *(Shared)*: Locating the enclosing git repository and the `~/.tpu-agent-suite` store (runs, sessions, studies, packages, `environment.json`), including the installed version of a package.

---

## Package Lifecycle Workflow

The canonical workflow comprises four core stages: **Verification**,
**Translation**, **Installation**, and **Uninstallation**.

```
┌─────────────────────────┐       ┌────────────────────┐       ┌─────────────────────┐
│ Canonical               │ verify│ Harness-Specific   │install│ Active Project /    │
│ Package Source          ├──────►│ Distribution       ├──────►│ Workspace           │
│ (package.release.json)  │trans- │ (dist/<harness>/)  │       │ (.tpu-agent-suite/) │
└─────────────────────────┘  late └────────────────────┘       └─────────────────────┘
```

### Manifest Composition (Linking)
A diagnostic step rather than one of the four stages: prints the manifest
together with everything discovery found from its skills (agents, references,
scripts, schemas, dependencies), so you can inspect exactly what a package
resolves to before verifying or translating it:

```bash
# Pretty-print composed manifest to stdout
python3 framework/compiler.py compose path/to/package

# Write composed package.release.json to file
python3 framework/compiler.py compose path/to/package --output dist/package.release.json
```

### Variant Packages (`--manifest`, `extends`, `add_skills`)
One package directory can release more than one package. Each package has its
own manifest file in the package directory. `compose`, `verify`, `lint` and
`translate` read `package.release.json` unless `--manifest <file>` names another one:

```bash
python3 framework/compiler.py verify package --manifest package.experimental.json
```

A variant manifest names only what differs from a base manifest:

```json
{
  "extends": "package.release.json",
  "name": "tpu-agent-suite-experimental",
  "display_name": "TPU agent suite experimental",
  "description": "...",
  "add_skills": ["close-gaps", "decompose-optimization"]
}
```

`load_package` merges it, in `resolve_manifest_extends`, before any other
check runs:

1. Every field of the base applies first.
2. Every field the variant sets replaces the base's field of that name, whole.
   The replacement is per top-level field: a variant that sets `hooks` or
   `copy` replaces the base's whole list.
3. `add_skills` is appended to the resulting `skills`: the base's list, or the
   variant's own `skills` if it sets one.
4. `extends` and `add_skills` are removed, and the merged manifest is
   validated against the manifest schema like any other.

A change to the base, such as a new hook or a version bump, reaches every
variant without an edit.

The merge rejects:

- `add_skills` without `extends`;
- `extends` that does not name another manifest file;
- a base that itself declares `extends`: only one level is supported;
- `add_skills` that is not a list of skill names, or that repeats a skill
  already in `skills`.

### 1. Package Verification
Strictly validates manifest schema (disallowing unrecognized properties by default),
instructions integrity, tool permissions, compatibility targets, and delegation
topologies (detecting circular delegations, more than three levels below a
released skill, and unreachable subagents):

```bash
python3 framework/compiler.py verify path/to/package

# Optionally permit extra/unrecognized manifest properties
python3 framework/compiler.py verify path/to/package --allow-extra-fields
```

### 2. Harness Translation
Compiles a canonical package into a standalone distribution tailored to a
specific agent harness (`antigravity`, `claude-code`, `codex`, or `cursor`).
Generates native manifests, agent system prompts, skill contracts, and bundles
the portable runtime snapshot:

```bash
python3 framework/compiler.py translate path/to/package \
    --harness [antigravity | claude-code | codex | cursor] \
    --output dist/<harness>/<package-name> \
    [--skills skill1,skill2] \
    [--include-internal] \
    [--allow-extra-fields]
```

`--skills` builds the skills it names in place of the ones `package.release.json`
lists. It may name any skill the package holds under `skills/`, released or
not. Each is built whole: the agents it delegates to, the skills it delegates
to, and every reference and script they reach all come along, and nothing is
trimmed from a skill's own delegates. `--skills tpu-agent-export-trajectory`
builds that one skill and no agents. `verify` and `compose` take the same flag.

To build less, name fewer skills. You cannot build a skill with only part of
its delegation tree, because that would test a different skill from the one
users install.

#### Harness-scoped tokens

Most `{{token}}`s expand from the manifest, in `render_agent_references`. A
few can't, because their value depends on the harness being built and nothing
in the manifest knows which that is: `{{harness}}` itself, and
`{{built-in:ask_user_question}}`, which names the tool that puts a question to
the user on the two harnesses that have one and becomes the plain instruction
on the two that don't. They are expanded later, by `render_harness_tokens`,
once `translate` has picked a target — and deliberately left alone during
`verify`, where there is no target at all.

### 3. Content Lint
`verify` proves a package is structurally sound: declared files exist, the
delegation graph is legal, tokens resolve. It says nothing about the prose
inside instruction files, which is where a migrated package rots — links into
the old source tree, paths from the origin build system, agents referred to by
their former names, or a capability re-implemented instead of delegated.

```bash
python3 framework/compiler.py lint path/to/package [--output-format github]
```

Checks live in `lint.CHECKS`; each returns `Finding`s and the caller decides
what is fatal. `verify_package` gates on the decidable ones
(`instruction-links`, `reference-links`, `environment-contract`,
`workspace-config-paths`); the rest are reported, since several are heuristics
that can fire on legitimate content. CI runs the full set as annotations.

`lint` takes no `--skills`: it reports on the package as authored, which is
what `package.release.json` releases.

---

## Tool Taxonomy & Permission Model

Packages declare all tools at the top level of `package.release.json`. Tool names must be globally unique across categories:

1. **`native_tools`**: Standard portable harness primitives (`read`, `write`, `edit`, `search`, `shell`). **Optional** — when a package omits this key entirely, the compiler injects the standard catalog from `definitions/native_tools.json`. Most packages rely on that default and never declare it.
2. **`script_tools`**: Executables shipped in the package, specifying `visibility` (`internal` | `external`) and paired `action_level` (`effect` + `scope`).
3. **`mcp_servers`**: Model Context Protocol servers with `visibility` and `action_level`.
4. **`hooks`**: Declarative lifecycle hooks (`PreToolUse`, `PostToolUse`, `SubagentStop`, `SessionStart`, `Stop`, `SessionEnd`) executing runtime actions or custom guard scripts.

An agent's `tools` list must be a subset of the names declared this way;
`verify` rejects any agent referencing an undeclared tool.

### Action Levels (`effect` + `scope` + `approval`)
- **`effect`**: `read` | `write` | `delete` | `execute`
- **`scope`**: `session` | `workspace` | `host`
- **`approval`** *(optional)*: `always` | `never` | `on-demand` (interactive user confirmation on step-over)

Example:

```json
{
  "script_tools": [
    {
      "name": "run_analysis",
      "path": "scripts/analyze.py",
      "visibility": "internal",
      "action_level": { "effect": "execute", "scope": "host", "approval": "on-demand" },
      "description": "Run an analysis pass on the host"
    }
  ],
  "mcp_servers": [
    {
      "name": "results_db",
      "command": "python3",
      "args": ["-m", "db_server"],
      "visibility": "external",
      "action_level": { "effect": "read", "scope": "workspace", "approval": "never" }
    }
  ],
  "hooks": [
    {
      "name": "sensitive-guard",
      "event": "PreToolUse",
      "matchers": ["read", "write", "shell"],
      "action": "script:scripts/guard.py",
      "timeout": 5,
      "visibility": "external"
    }
  ]
}
```

### Universal Lifecycle Hooks Contract

Packages declare lifecycle hooks using portable primitives. Hooks can be declared as **Script Hooks** (file-backed) or **Command Hooks** (direct shell execution):

1. **Script Hooks (`action: "script:<path>"`)**: A script shipped within the package (e.g., `scripts/guard.py`).
2. **Command Hooks (`command: "<command>"`)**: A direct shell command, utility, pipeline, or inline script (e.g., `git diff --quiet`, `flake8`, or `python3 -c "..."`).

At runtime, the embedded hook dispatcher (`runtime.py hook-dispatch`) wraps both types into a unified cross-harness lifecycle:
- **Normalized Inbound Payload**: Normalizes inbound payloads from Claude Code, Codex, Antigravity (`agy`), and Cursor into a single unified JSON schema on `stdin`.
- **Deterministic Working Directory**: Always executes hook actions with the target **workspace root** as `cwd`, avoiding harness-specific working directory discrepancies (such as `.agent/` vs `.`).
- **Timeout Enforcement**: Enforces an execution timeout (`--timeout <seconds>`, defaulting to 10s or configured via `timeout` in `package.release.json`). If hook execution exceeds the timeout, the child process is terminated and the hook is settled by its `on_failure` policy (below). The harness itself is registered with the declared timeout plus `HOOK_DISPATCH_HEADROOM_S` (5s), so the dispatcher's verdict reaches the harness before the harness gives up on the hook and kills it.
- **Batched Delegations**: When one tool call delegates to several subagents at once (Antigravity's `invoke_subagent` with more than one entry in `Subagents`), each `PreToolUse` hook runs once per subagent, concurrently, each run under the hook's own timeout, so the batch fits the same harness budget as a single call. Subagents the hook denies are removed from the call and the rest proceed; if it denies all of them, the call is denied. A hook script may therefore run alongside other copies of itself, and must be safe to do so.
- **Outbound Adaptation**: Validates `stdout` against `hook-output.schema.json` and adapts it to the host harness's native contract.

#### Hook Failure Policy (`on_failure`)

A hook *fails* when it does not return a verdict of its own: it times out, exits non-zero, prints non-JSON or JSON that violates `hook-output.schema.json`, or its script cannot be found. A builtin fails when it raises or outruns its timeout. Each hook, builtins included, declares what a failure becomes:

- **`"on_failure": "deny"`** (default): the failure is a `"deny"`, with the failure as its reason. Use it for guards whose silence must not be taken as consent.
- **`"on_failure": "allow"`**: the failure is an `"allow"`, so a broken or hung hook cannot trap the agent (a deny on `Stop` or `SubagentStop` keeps it running). Use it for loggers, recorders, and validators whose absence should not stop the work.

The policy only covers failures: a hook that prints `{"decision": "deny", ...}` is denied under either policy. Any non-zero exit, `2` included, is a failure, since common errors (a shell syntax error, an `argparse` usage error) exit `2` too. Every failure, under either policy, is reported on the dispatcher's `stderr` and appended to `~/.tpu-agent-suite/logs/hook_failures.jsonl`; each record names the workspace it came from.

The policy also covers the dispatcher's own failure to start. `hook-dispatch` puts its verdict in JSON and always exits `0`, so a non-zero exit from it is never a verdict — yet Claude Code reads `2` as a block and Antigravity refuses any non-zero. Three layers keep the declared policy in charge:

- The compiled command ends in a shell fallback, `... || { echo ... >&2; exit N; }`, which needs nothing installed. On tool events `N` is the policy's code: `0` for `allow`, `2` for `deny`. On `Stop`, `SubagentStop` and session events it is always `0`, since a block there means "keep going" and would loop with no way out.
- A `hook-dispatch` whose arguments do not parse is answered in the harness's shape and recorded as a `startup` failure.
- A `runtime.py` whose own imports fail exits by the policy with the reason on `stderr`.

#### Command Hook Execution Semantics

Command hooks allow running arbitrary CLI commands with standard Unix exit code semantics without writing harness-specific adapter code:

- **Exit Code `0` (Success)**:
  - **Empty Output (`""`)**: Automatically synthesized into an `"allow"` decision tailored for the host harness (`hookSpecificOutput.permissionDecision = "allow"` for Claude Code, `{"decision": "allow"}` for Antigravity, `{}` for Cursor).
  - **JSON Output**: Validated against `hook-output.schema.json` and translated to the host harness.
- **Non-Zero Exit Code (Failure)**:
  - Settled by `on_failure`: a `"deny"` under the default, with `stderr` (or a fallback exit code message) as the `reason`. To block under either policy, print a JSON deny instead.

```json
{
  "name": "require-clean-working-tree",
  "event": "PreToolUse",
  "matchers": ["shell"],
  "command": "git diff --quiet",
  "timeout": 5,
  "visibility": "external"
}
```

#### Normalized Inbound Schema (`sys.stdin`)
Every hook receives this unified JSON payload defined by `schemas/hook-input.schema.json`:
- **`event`**: `"PreToolUse"` | `"PostToolUse"` | `"SubagentStop"` | `"SessionStart"` | `"Stop"` | `"SessionEnd"`.
- **`tool`**: Canonical tool name (`"shell"`, `"read"`, `"write"`, `"edit"`, `"search"`, `"web_search"`, `"web_fetch"`, `"delegate"`).
- **`subagent`**: Subagent identifier (populated for `SubagentStop` or delegation events).
- **`args`**: Normalized parameter dictionary (e.g. `{"command": "..."}`, `{"path": "..."}`, or `{"subagent": "...", "prompt": "..."}` when tool is `"delegate"`).
- **`output`**: Execution output or deliverable (`null` for `PreToolUse`, populated for `PostToolUse` and `SubagentStop`).
- **`context`**: Dictionary containing `"workspace"` and `"package"`.

> **Note on `SubagentStop`**: In native harnesses (such as Claude Code), subagent stop events fire on *every* turn pause, model wait, or user interruption. In this framework, canonical `SubagentStop` triggers **only when a subagent actually finishes** and owes a deliverable. Turn pauses and user-halted subagents are automatically filtered by the runtime dispatcher so hook scripts only validate completed deliverables.

> **Note on hand-back tools**: Claude Code delivers a subagent's result by one of two roads. Outside auto mode, the subagent's last message is the result (native `SubagentStop`). In auto mode, Claude Code gives every locally run subagent, foreground or background, the `SubagentHandback` tool, and only that call reaches the caller. A harness declares such a tool as a `handback` entry on its `SubagentStop` mapping in `schemas/event-mappings.json`, and the compiler registers every `SubagentStop` hook on it too (for Claude Code, a `PreToolUse` hook on `SubagentHandback`). Hook bodies see the same canonical event on both roads (`args.handback` is `true` on the tool road), and a refusal comes back to the subagent as a tool denial. The hand-back command also carries `--handback-event PreToolUse`, so a dispatch that fails before reading the payload still answers in the tool event's shape. To check each result once, the native stops of a run that hands back are let through unchecked. A native stop is let through if the stopping run was given the tool in auto mode (`requires_mode`), where Claude Code does not deliver plain text and re-prompts the subagent to call the tool, or if the stopping run's hand-back call was accepted. Auto mode alone is not enough: Claude Code 2.1.241 gives the tool to no subagent and delivers its plain final message, so an auto-mode run whose transcript shows no offer of the tool (`transcript_offer`) is checked; when the transcript cannot be read, the mode decides, and the hook reports the skip on stderr, since a run without the tool goes unchecked then. Only that run's transcript records count (`transcript_prompt_id`), so a resumed subagent is judged on its current run. Outside auto mode, a stop whose run cannot be located, like one whose transcript cannot be read, is checked. A run started outside auto mode and switched into it mid-run never gets the tool, so its plain result goes unchecked.

> **Note on background work**: A subagent that stops while a background child or command it launched is still running is not finished. Antigravity says so on the stop (`fullyIdle`) and resumes the agent when the work completes, so the stop is skipped. Claude Code reports no flag, so the runtime derives one: the `track-background` builtin records each launch a `PostToolUse` reports (`tool_response.status == "async_launched"` for `Agent`, `tool_response.backgroundTaskId` for `Bash`) under the launching `agent_id`, and the stop intersects that ledger with the payload's `background_tasks` -- the list alone has no owner, the ledger alone goes stale. Hook bodies see `args.fully_idle`, `args.fully_idle_source` (`harness` or `ledger`) and, when not idle, `args.outstanding_work`. Whether a not-idle stop is a pause depends on the run (measured on Claude Code 2.1.312–2.1.314): an interactive session resumes the agent on each completion, so the stop is skipped; a headless run (`claude -p`) does not -- the pause text becomes the caller's result and the work is orphaned -- so the stop is **refused**, naming the live tasks. The refusal tells the agent to run `sleep 20` and try again, because headless a child's result reaches the agent only with a tool result; Claude Code blocks a bare `sleep` of 25 s or more. An agent with no shell has no such call to make, so its compiled contract says to launch in the foreground and the `track-background` builtin refuses a background delegation from it at `PreToolUse` (the build file's `agent_tools` says which agents have a shell). The gate runs before the hand-back skip. The regime comes from the hook's environment (`pause_resumes` in the mapping's `idle` block): `CLAUDE_CODE_ENTRYPOINT` is `cli` interactive and `sdk-cli` headless; anything else refuses, since a wrong refusal costs a wait and a wrong skip loses the result. A payload without the live list is checked as before, with a line on stderr. The ledger is one file per launch under the sessions directory, cleared at `SessionEnd`.

#### Universal Outbound Schema (`sys.stdout`)
Every hook emitting output uses this unified JSON payload defined by `schemas/hook-output.schema.json`:

```json
{
  "decision": "allow",
  "reason": "Why denied (if decision is deny)",
  "message": "Optional user notification banner",
  "context": "Optional context injected into LLM conversation",
  "followup_message": "Optional feedback directed to the subagent to continue or repair deliverables",
  "updated_args": { ... },
  "updated_output": "..."
}
```

#### Minimalist Developer Script Hook Template

```python
#!/usr/bin/env python3
"""A universal lifecycle hook running identically on Claude Code, Antigravity, Cursor, and Codex."""
import json
import sys

def main():
    data = json.load(sys.stdin)
    tool = data.get("tool")
    args = data.get("args", {})

    if tool == "shell" and "rm -rf /" in args.get("command", ""):
        print(json.dumps({
            "decision": "deny",
            "reason": "Destructive shell command prohibited."
        }))
        return

    print(json.dumps({
        "decision": "allow",
        "message": f"Verified {tool} execution."
    }))

if __name__ == "__main__":
    main()
```

### Units, frontmatter and layout

What each kind of unit is, what its frontmatter declares, how the compiler
discovers units from the released skills, and where each one lives before
and after install: [Package](../docs/design/package.md).

---

### 4. Installation
Installs a compiled distribution into a destination workspace. Sets up no
compute: an existing `environment.json` is kept as it is, a missing one is
written with no profiles, and the first session (or `runtime.py compute setup
apply`, for headless installs) sets compute up. Pre-validates file
collisions, checks this machine's packages, executes non-destructive
JSON/TOML configuration merges, and records an atomic ownership ledger
(`ownership.json`):

```bash
# Project scope (installs to explicit workspace destination; requires --destination)
python3 framework/installer.py install dist/codex/<package-name> --destination /path/to/project

# User scope (installs globally to ~/.<harness>/, e.g. ~/.agents for Antigravity)
python3 framework/installer.py install dist/codex/<package-name> --scope user

# Non-interactive install (CI, containers)
python3 framework/installer.py install dist/claude-code/<package-name> \
    --destination /path/to/project \
    --non-interactive

# Upgrade an existing installation
python3 framework/installer.py install dist/claude-code/<package-name> \
    --destination /path/to/project \
    --upgrade

# Install without checking this machine's packages
python3 framework/installer.py install dist/claude-code/<package-name> \
    --scope project \
    --skip-host-check

# Install without touching the workspace memory file (CLAUDE.md / AGENTS.md)
python3 framework/installer.py install dist/claude-code/<package-name> \
    --destination /path/to/project \
    --no-workspace-memory
```

#### Workspace Memory Blocks

A package may declare `workspace_memory` entries in its manifest, each naming a
markdown file whose contents are installed as a marker-delimited block in the
harness's memory file. This is how package-wide instructions reach every
session rather than only the sessions that invoke a skill. Because of that
reach, the compiler opens every block with one shared preamble saying it
applies only inside the package's skills (each one named) and their agents, so
a block does not state its own scope. The preamble names skills, not the
package: a harness lists skills to the model, but has no notion of a package.

| Harness | `--scope project` | `--scope user` |
| :--- | :--- | :--- |
| `claude-code` | `<destination>/CLAUDE.md` | `~/.claude/CLAUDE.md` |
| `codex` | `<destination>/AGENTS.md` | `$CODEX_HOME/AGENTS.md`, default `~/.codex/AGENTS.md` |
| `antigravity` | `<destination>/AGENTS.md` | `~/.gemini/AGENTS.md` |

A user-scope install's destination is `$HOME`, but no harness reads
`~/CLAUDE.md` or `~/AGENTS.md` as user memory: those are project memory, read
only by a session started in `$HOME` itself. The user-scope paths live in
`installer.USER_SCOPE_MEMORY_PATHS`. For Codex, a `CODEX_HOME` under `$HOME`
relocates the memory block, `agents/`, `hooks.json` and `config.toml` together.
One outside `$HOME`, or `$HOME` itself, fails the install before anything is removed or
written: the ownership ledger records destination-relative paths, and
`~/.codex` in its place would be an install Codex never reads. The ledger
records each block where it was written, so an upgrade strips a block that an
older install left in `~/AGENTS.md` before writing the new one.

The file is created if absent and otherwise appended to. Blocks render
through the compile-time resource registry (the same tokens as instructions)
and apply verbatim at install time -- no install-time path resolution. Because
memory files load in every session from arbitrary working directories, a
block may reference central resources only; the compiler rejects references
to the workspace environment file. Re-installing replaces the block in place, uninstalling
strips it (deleting the file only when the install created it and nothing else
remains), and a block edited locally is reported as modified managed content
rather than being silently overwritten. Pass `--no-workspace-memory` to skip
this step entirely, or `--force` to overwrite a locally edited block.

### 5. Uninstallation
Cleanly removes all managed files, rolls back modified JSON/TOML
configuration entries, and strips installed workspace memory blocks, while
preserving user-created artifacts and unmodified settings. The ledger names
the installed package, so `--package` is only a check:

```bash
# Uninstall from project scope (requires --destination)
python3 framework/installer.py uninstall --destination /path/to/project

# Uninstall from user scope
python3 framework/installer.py uninstall --scope user
```

---

## Embedded Runtime Architecture

Every generated distribution ships `framework/` as is, minus the build tools
(`compiler.py`, `builder/`, `lint.py`, `frontmatter.py`, `definitions/`, `templates/`), so
the runtime has the same paths and imports as here. Installed agent entrypoints invoke
this local runtime to manage execution sessions and artifact lifecycles without
relying on external framework installations.

### Run Namespace Isolation
Before initiating subagent delegation, a released skill allocates a
collision-resistant, timestamped run directory in central storage under
`~/.tpu-agent-suite/runs/<run-id>/`. (`$TPU_AGENT_SUITE_HOME` moves the store
for tests only; the installer refuses it, because installed hooks always run
from `~/.tpu-agent-suite`.)

```bash
python3 ~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/framework/runtime.py start-run
```

Output:

```json
{
   "created_at" : "2026-08-14T05:00:00Z",
   "environment_revision" : 1,
   "version" : "<version>",
   "package" : "tpu-agent-suite",
   "run_id" : "20260814T050000Z-7a3b9f12c4d5"
}
```

### Contract & Payload Validation
Validates input and output instances against package schemas using the
deterministic JSON Schema validator:

```bash
python3 ~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/framework/runtime.py validate \
    --schema ~/.tpu-agent-suite/packages/tpu-agent-suite@<version>/schemas/analysis-request.json \
    instance.json
```

---

## Dependency Packaging & Requirements Trimming

Packages define Python dependencies centrally to eliminate scattered requirements files and avoid conflicting per-agent ranges:

1. **Central Catalogs (`requirements-client.txt` and `requirements-tpu.txt`)**:
   - `requirements-client.txt`: Declares all host-side dependencies with explicit version constraints (e.g. `pydantic>=2.0.0`).
   - `requirements-tpu.txt`: Declares remote TPU workload dependencies with explicit version constraints (e.g. `torch>=2.11.0,<2.14`, `libtpu>=0.0.44`).
   - Package entries in central catalogs must specify version constraints; unpinned entries fail `compiler.py verify`.
   - Pip options (such as `-f` or `--extra-index-url`) and comments are preserved when emitting distributions.
   - `compute/env_check.py` checks the installed versions against them, and fails a requirement when a distribution listed for it in `CONFLICTS` is installed too: `requirements-client.txt` on the installing machine at install, `requirements-tpu.txt` on the compute target in `compute setup verify` (see `compute/README.md`).

2. **Frontmatter Declarations**:
   - A skill, agent or reference declares only unpinned canonical package names in its `host_dependencies` and `tpu_dependencies` frontmatter arrays (e.g. `["google-cloud-aiplatform", "vertexai"]`); the compiler unions them across the package.
   - All declared dependency names must resolve against the root catalogs during `compiler.py verify`.

3. **Trimming**:
   - The compiler trims `requirements-client.txt` and `requirements-tpu.txt` to the dependencies the built units declare, omitting the rest.

---

## Core Guarantees

The guarantees every package gets on every harness are listed in
[Framework](../docs/design/framework.md#guarantees).
