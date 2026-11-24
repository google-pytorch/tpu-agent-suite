# Developing TPU agent suite

How to set up a development environment, the groundrules every change follows,
what the CI gates are, and the short
path for the common contributions: a new agent, a tpu-knowledge-base source, an
eval dataset. External pull requests are not accepted at this time (see
[`CONTRIBUTING.md`](../CONTRIBUTING.md)); this guide is for maintainers.

Reference material this guide points into:

- [`docs/design/`](design/README.md): how the framework and the suite are designed; [Package](design/package.md) covers the unit kinds, frontmatter and tokens.
- [`README.md`](../README.md): package/framework layout, harness matrix, install walkthrough.
- [`framework/README.md`](../framework/README.md): compiler, installer, hooks, runtime reference.
- [`package/README.md`](../package/README.md): the full guide to converting an existing skill into a TPU agent suite package.
- [`eval/README.md`](../eval/README.md): the evaluation runner.

---

## Groundrules

These apply to every change merged to `main`.

1. **New prose units follow the package format.** A new agent, skill or
   reference carries the frontmatter that [Package](design/package.md)
   defines. A skill is listed under `skills` or `add_skills` in the manifest
   that ships it. Agents and references are not listed: the compiler finds
   them through the `delegates` and `references` frontmatter of the units
   that use them.
2. **No bespoke task-agnostic scripts.** Functionality that more than one agent
   needs goes through the runtime or a shared agent. For example, TPU access
   goes through `runtime.py compute` commands, not a script of your own. Every
   agent declares meaningful `accepts` and `produces` schemas.
3. **`package.release.json` ships only thoroughly tested functionality.**
   Skills listed there reach every user, so adding one is a product decision
   as well as an engineering one. New functionality is added to it only in
   the version-bump PR (the PR that sets the next `"version"`), and only once:
   - its user journeys show consistently performant eval results;
   - the product owners approve releasing it;
   - it is described in that release's `release-notes/vX.Y.Z.md`.
4. **Experimental work uses a separate manifest.** Experimental work lives
   outside `package.release.json`: `package.experimental.json` (`extends`
   `package.release.json` via `add_skills`) for serving/inference optimization
   skills, `package.train.json` (standalone `skills`) for training optimization
   skills, or a manifest of your own (see
   [The experimental package](#the-experimental-package-tpu-agent-suite-experimental)
   and [The training package](#the-training-package-tpu-agent-suite-train)).
   Keep training-only skills in `package.train.json` rather than
   `package.experimental.json` so they do not pull in inference agents or
   collide with inference skills in the same workspace. Experimental work merges
   to `main` as long as it does not break anything already released.
5. **Reuse first; duplicate only with a plan.** New prose units and code should
   reuse what already exists, within reason. Duplication is acceptable when
   reuse would risk breaking existing functionality, or would significantly slow
   down introducing or validating new functionality. In that case, the PR
   states what is duplicated and the plan for removing it.

### Versioning

Releases follow [Semantic Versioning](https://semver.org). The interface is
what users depend on: skill names, the install layout, configuration and the
runtime CLI. Before 1.0.0:

- a regular release bumps the minor version (0.8.0 → 0.9.0) and may change
  the interface;
- a bug-fix release bumps the patch version (0.8.0 → 0.8.1) and must not.

1.0.0 marks the commitment to a stable interface. After that, an incompatible
change needs a major version bump.

---

## 1. Environment

Three environments exist, and they are not interchangeable:

| Purpose | Install | Used by |
| :--- | :--- | :--- |
| Build, verify, and test the package | `pip install -r cicd-requirements.txt` | `make verify`, `make test`, `make build`, `make lint`, `make coverage` |
| Run the installed agents on a host | `pip install -r package/requirements-client.txt` | Agent scripts at runtime (see [install.md](install.md#1-host-prerequisites)) |
| Run evaluations | `./eval/setup.sh` (uses `uv`, needs Docker) | `eval/run.py`, `verify.sh`, `sandbox.sh` |

For day-to-day work on the package you need only the first:

```bash
python3 -m venv ~/.venv/tpu_agent_suite_dev && source ~/.venv/tpu_agent_suite_dev/bin/activate
pip install -r cicd-requirements.txt
pip install pre-commit
pre-commit install
```

`.pre-commit-config.yaml` at the repo root declares `pre-commit`, `pre-push`,
and `commit-msg` hook types, so a plain `pre-commit install` sets up everything.
Among other things the hooks add a `Signed-off-by:` trailer to each commit, run
`ruff`/`yapf`/`isort`, fix markdown, validate the knowledge base, and run
`.ci/check_denylist.py` on both `git commit` and `git push`. Under
`framework/`, `google/drivers/` and `google/tests/`, `ruff` also enforces the
docstring convention (its `D` rules, Google style): public functions, methods
and classes carry a docstring whose first line is a one-sentence summary ending
in a period. In checkouts
where the repository denylist is not present on disk, place your local copy at
`~/.cache/tpu-agent-suite/denylist.yaml` before committing:

```bash
mkdir -p ~/.cache/tpu-agent-suite && cp <path-to-denylist.yaml> ~/.cache/tpu-agent-suite/denylist.yaml
```

Agent instruction files (`package/**/*.md`), knowledge base `raw/` sources and
the generated codebase symbol catalogs are excluded from formatting, since they
are consumed byte-for-byte.

---

## 2. The development loop

All `make` targets run from the repository root. `make help` lists them.

```bash
make verify   # gate: manifest schema, delegation graph, referenced files, knowledge base index/log
make test     # framework, package (translation parity), and tpu-knowledge-base unit tests
make lint     # content lint checks on instruction prose and scripts (enforced in CI + PR annotations)
make build    # compile dist/<harness>/tpu-agent-suite for all four harnesses
```

What each one catches:

- **`verify`** is the structural merge gate. It validates `package.release.json` against the
  schema, checks that every referenced instruction/schema/script/reference
  exists, enforces the delegation rules (no cycles, at most 3 levels below a
  released skill, every agent reachable), and runs `validate-kb`. A *released
  skill* is one that `package.release.json` lists under `skills`, or that `--skills`
  names for a build; those are the roots the compiler starts from. It also runs the
  four lint checks that are decidable (`instruction-links`, `reference-links`,
  `environment-contract`, `workspace-config-paths`).
- **`test`** runs `make test-framework`, `make test-package`, and
  `make test-knowledge-base`. The package tests include one translation parity
  suite per harness; if you change a translator or an agent's frontmatter, these
  are what break.
- **`lint`** checks instruction and script content rules not covered by `verify`.
  Findings exit non-zero (enforced in CI and surfaced as inline PR annotations),
  while informational `WARNING:` lines for unreachable files or undeclared
  catalog entries do not fail the check.
- **`build`** is what `dist/` and the evaluation staging consume.

`make test-all` additionally runs the companion `*_test.py` suites next to
package scripts; those need the host requirements installed and are local
only. `make coverage` enforces the framework coverage floor;
`make coverage-diff DIFF_BASE=origin/main` checks only your changed lines.

To try a change end to end, build and install into a scratch workspace, then
open your harness there:

```bash
make build
python3 framework/installer.py install dist/claude-code/tpu-agent-suite \
    --destination /tmp/scratch-workspace --non-interactive
```

The first session there sets compute up, or see
[Headless installs](install.md#headless-installs-ci-containers) to create a
profile from the command line.

### Building a package

To build the distribution for a specific harness, run:

```bash
python3 framework/compiler.py translate package --harness <harness> --output dist/<harness>/tpu-agent-suite
```

*(Replace `<harness>` with `claude-code`, `codex`, `antigravity`, or `cursor`).*

The compiled distribution directory bundles translated skills, compiled agent prompts, harness configuration, lifecycle hooks, the runtime and installer (in `framework/`), dependency manifests, and a README.

If you also want to package that single distribution into an archive (`.tar.gz`):

```bash
python3 framework/compiler.py archive --dist-dir dist --output dist/archives --harness <harness>
```

To install the result, follow [install.md](install.md#2-install-the-distribution).

#### The experimental package (`tpu-agent-suite-experimental`)

`package/package.experimental.json` builds `tpu-agent-suite-experimental`, a package for testing. It releases everything `tpu-agent-suite` releases, plus the vLLM optimization skills (`close-gaps`, `decompose-optimization`, `write-tpu-kernels` and others).

It is a complete package, not an add-on. It has its own runtime, compute environment and hooks, and it builds every agent its skills use. A workspace holds one package at a time, so **install either `tpu-agent-suite` or `tpu-agent-suite-experimental`, not both**. To switch, uninstall the installed package first.

```bash
make build-experimental
python3 dist/<harness>/tpu-agent-suite-experimental/framework/installer.py install --destination /path/to/project
```

Both packages are built from the same files in `package/`. `--manifest` selects the manifest file, and works with `compose`, `verify`, `lint` and `translate`. It defaults to `package.release.json`.

`package.experimental.json` declares `"extends": "package.release.json"`. It takes every field of `package.release.json`, replaces `name`, `display_name` and `description`, and lists only the skills it adds under `add_skills`. A change to `package.release.json`, such as a new hook or a version bump, therefore reaches both packages. Only one level of `extends` is supported.

#### The training package (`tpu-agent-suite-train`)

`package/package.train.json` is a standalone manifest building `tpu-agent-suite-train`, an experimental training optimization package (`tpu-training-tuner` and `tpu-agent-export-trajectory`). Build and verify it with `make build-train` and `make verify-train` (or `--manifest package.train.json`).

---

## 3. What CI runs

Every pull request runs the following checks:

| Gate | Runs | Required? |
| :--- | :--- | :--- |
| **Package & unit tests** | `make verify`, `make verify-experimental`, `make verify-train`, `make validate-kb KB_BASE=origin/<base>`, `make test`, `make test-eval`, `make build`, `compiler.py lint` (plus `--no-overlays` `verify` / `verify-experimental` / `verify-train` / `lint`, and `google/tests` when present), `make coverage`, `make coverage-diff` | yes (`coverage-diff` advisory) |
| **Formatting & denylist** | The `pre-commit` hook set plus `.ci/check_denylist.py` on PR diffs, commit messages, and PR titles | yes |
| **Dataset naming** | Eval dataset and task folder naming (`^[a-z0-9-]+$`) | yes, when `eval/**` changes |
| **Code owners** | Code-owner approval on protected onboarding, framework, documentation, and infrastructure paths | yes |
| **Smoke evaluation** | The `smoke-check` Harbor evaluation when package, framework, or evaluation files change | reported |

Release builds validate `release-notes/v<version>.md` (copied from
`release-notes/TEMPLATE.md`), compile standalone per-harness archives without
internal overlays, scan the compiled `dist/` with `.ci/check_denylist.py`,
smoke-test installing the built archive(s) in a clean workspace, and emit
`SHA256SUMS`.

Run the same commands locally before pushing; CI adds nothing you cannot
reproduce with the commands in the table above plus `pre-commit run --all-files`.

---

## 4. Adding an agent to an existing package

`package/README.md` walks through converting a whole skill from scratch;
[Choosing a kind](design/package.md#choosing-a-kind) decides between a skill, an
agent, a reference, or inline prose. To scaffold one with an agent harness
(Claude Code, Antigravity, Codex, or Cursor), prompt it with:

```text
Follow @docs/developing.md (§4–§4c) to scaffold a new [agent | skill] named `<name>` that <one-line purpose>, wired into `<caller-or-manifest>`.
```

Adding one agent to a domain that already exists:

1. **Write the instruction file** at `package/agents/<agent-name>.md`
   (validated against `framework/schemas/agent-frontmatter.schema.json`; never
   grant skill-only tool `ask` to an agent):

   ```yaml
   ---
   name: my-agent                       # unique, lowercase hyphen-case
   description: One line on what it does.
   tools: [read]                        # minimum of read | write | edit | shell | search | web_search | web_fetch
   accepts: schemas/my-agent-request.json   # typed input contract under schemas/
   produces: schemas/my-agent-result.json
   delegates: []
   scripts: []                          # under scripts/, copied into the dist
   references: []                       # files or directories under references/
   host_dependencies: []                # unpinned names; pinned in requirements-client.txt
   tpu_dependencies: []                 # unpinned names; pinned in requirements-tpu.txt
   ---
   ```

   In prose, name other agents with `{{agent:<name>}}` and package files with
   `{{script:..}}`, `{{reference:..}}`, `{{data:..}}` or `{{schema:..}}`, never
   by a typed-out path. Prefix workspace paths with `<workspace>/...`,
   `<work>/...`, or `workspace/...` (for example `` `<workspace>/scripts/train.py` ``
   — `./...` passes `verify`, whereas inline backticked `` `./config.json` ``
   still trips `lint`'s `bare-paths` check). The full list is in
   [Tokens](design/package.md#tokens).

   Don't write generic compute lifecycle prose (`run` vs. `submit`, job
   wait/logs/artifacts) or session-journal instructions: the compiler appends
   the compute contract automatically (`lint`'s `generic-compute-prose` flags
   duplicated compute prose) and appends the journal contract when `tools`
   includes `shell`.

2. **Nothing to register; place scripts and references under their `<family>/` directory.**
   The compiler finds an agent by following `delegates` from the released
   skills, so wiring the delegation (next step) is what makes it part of the
   build. Pin any new pip dependencies in `package/requirements-client.txt` /
   `requirements-tpu.txt` (unpinned entries fail `verify`). Scripts go under
   `package/scripts/<family>/` (with companion `*_test.py` files beside them)
   and references under `package/references/<family>/`; a reference may declare
   its own scripts, references, delegates and dependencies in optional
   frontmatter (`framework/schemas/reference-frontmatter.schema.json`), and
   every reader inherits them.

   When porting an external tool or adding a multi-file Python package, flatten
   it into a single self-contained `package/scripts/<family>/` directory (like
   `package/scripts/workload-microbenchmark/`) — no nested `skills/.../scripts/`
   or single-implementation `backends/` wrappers, no standalone
   `pyproject.toml`/`setup.py`, no `..` `sys.path` walks across package roots,
   and **never** place nested `AGENTS.md` or `CLAUDE.md` files under `package/`
   or `scripts/` (harnesses auto-load nested `AGENTS.md` files into context; put
   provenance or citations in `SOURCES.md` or `references/`). Agent-invoked CLIs
   should fail fast on unknown flags (`argparse` `parse_args`, not
   `parse_known_args`).

3. **Wire the delegation.** Add the name to the `delegates` list of the agent
   that will call it, and mention it in that agent's prose with an
   `{{agent:my-agent}}` token. If it is a new top-level workflow, add a routing
   line in `package/skills/tpu-agent/SKILL.md` and to that file's `delegates`.

4. **Add schemas and contract tests** under `package/schemas/` and
   `package/tests/test_<agent>_schemas.py`. Every `schemas/*.json` is registered
   automatically. Set `"additionalProperties": false`, include valid
   `"examples"` for each outcome, and express per-status required fields with
   `"status_requires"` or `"oneOf"` (`framework/utils.py`'s
   `JSON_SCHEMA_KEYWORDS` omits `$ref`, `if`/`then`, and `allOf`;
   every `status_requires` key must appear in `properties.status.enum` and its
   fields must not be in top-level `required`; `agent_contracts.result_problems(schema, payload)`
   enforces non-empty values for `status_requires` fields and, when `artifacts`
   is a property, non-empty `artifacts` on `"status": "completed"`; leave
   `"failed"` out of `status_requires` so `test_runtime.py` confirms at least
   one failure status can always be returned):

   ```json
   {
     "$schema": "https://json-schema.org/draft/2020-12/schema",
     "$id": "my-agent-result.json",
     "title": "my-agent Result",
     "type": "object",
     "additionalProperties": false,
     "required": ["status", "summary", "run_id"],
     "status_requires": {
       "completed": ["artifacts"],
       "invalid_input": ["missing"]
     },
     "properties": {
       "status": { "type": "string", "enum": ["completed", "invalid_input", "failed"] },
       "summary": { "type": "string", "minLength": 1 },
       "run_id": { "type": "string", "pattern": "^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{12}$" },
       "artifacts": { "type": "array", "items": { "type": "string", "minLength": 1 } },
       "missing": { "type": "array", "items": { "type": "string", "minLength": 1 } },
       "failure_reason": { "type": "string" }
     },
     "examples": [
       {
         "status": "completed",
         "summary": "Wrote report.",
         "artifacts": ["~/.tpu-agent-suite/runs/20260921T120000Z-abcdef012345/artifacts/my-agent/report.md"],
         "run_id": "20260921T120000Z-abcdef012345"
       },
       {
         "status": "invalid_input",
         "summary": "Missing model config.",
         "missing": ["model_config_path"],
         "run_id": "20260921T120000Z-abcdef012345"
       },
       {
         "status": "failed",
         "summary": "Target unreachable.",
         "failure_reason": "compute_run_failed",
         "run_id": "20260921T120000Z-abcdef012345"
       }
     ]
   }
   ```

   Follow [`package/schemas/roofline-request.json`](../package/schemas/roofline-request.json),
   [`package/schemas/roofline-result.json`](../package/schemas/roofline-result.json)
   (or [`package/schemas/code-arm-result.json`](../package/schemas/code-arm-result.json)
   when `oneOf` is needed), and
   [`package/tests/test_roofline_schemas.py`](../package/tests/test_roofline_schemas.py)
   for a complete request/result pair and contract test.

5. **Profiling?** Do not run XProf. Delegate to `{{agent:tpu-diagnostics}}`
   instead; `lint` flags direct XProf use. See
   [`package/README.md`, Step 4.5](../package/README.md#step-45-profiling-ask-tpu-diagnostics-dont-run-xprof).

6. `make verify && make test && make lint`, then build and install into a
   scratch workspace to try it.

---

## 4b. Adding a user-invocable skill

Typically, an agent is reached by delegation; a **skill** is reached by the user typing
`/<name>`. No harness in `compatibility_targets` lets an agent register a
command that users invoke directly, so a lightweight skill is how the framework gets
that effect: it is the one user-invocable surface all four harnesses share.

Use one for an action that should be close to deterministic --
`tpu-agent-export-trajectory` is the worked example. Close to, not actually: the
model still reads the skill and composes the call, so determinism is a
property of how narrowly the body is written, never something the mechanism
enforces.

1. **Write the instruction file** at
   `package/skills/<skill-name>/SKILL.md`, with frontmatter carrying
   `name` and `description` (and, if it delegates, `delegates`):

   ```yaml
   ---
   name: my-skill
   description: One line the harness shows when listing skills.
   ---
   ```

   The same resource tokens apply, plus `{{harness}}`, which resolves to the
   harness being compiled for. A skill that drives a runtime command should
   spell it once:

   ```
   {{runtime}} my-command --harness {{harness}} --package {{package_tag}}
   ```

2. **Register it in a manifest.** Skills are what users see and invoke, so
   where you register one decides who gets it. (Agents and references are
   not listed; a skill pulls them in through `delegates` and `references`.)

   - **For experimental serving / inference optimization skills,** list it under
     `add_skills` in `package/package.experimental.json` (which `extends`
     `package.release.json`; see
     [The experimental package](#the-experimental-package-tpu-agent-suite-experimental)).
   - **For experimental training skills,** list it under `skills` in
     `package/package.train.json` (standalone manifest; see
     [The training package](#the-training-package-tpu-agent-suite-train)), or
     create a separate manifest for a distinct initiative so skills from
     unrelated workflows do not collide in one workspace.
   - **To release it to every user,** it moves to `skills` in
     `package/package.release.json`. That is a product decision as much as an
     engineering one: it needs approval, and happens only in a version-bump
     PR (see [Groundrules](#groundrules)).

   ```json
   "add_skills": ["my-skill"]
   ```

   The name is the directory under `skills/`, and the document's frontmatter
   must agree with it, so the two cannot drift apart. No skill is special:
   the order in either list means nothing.

3. **Write the body to leave as little to judgement as possible.** State the
   single command to run, say what to do with each failure the command can
   report, and say plainly what the model must *not* invent -- consent flags
   especially. These narrow the model's choices rather than removing them, so
   anything that must actually be enforced belongs in the runtime the skill
   calls, not in its prose.

4. `make verify && make test && make lint && make build`. The compiler emits
   `dist/<harness>/tpu-agent-suite/skills/<skill-name>/SKILL.md` for all four
   harnesses, and `tpu-agent-suite-install.json` gains a copy rule per harness.
   Nothing else needs changing: the install specs emit one copy rule for
   every declared skill. For another manifest, pass `--manifest <file>` to
   `compose`, `verify`, `lint` and `translate`.

---

## 4c. Pre-PR checklist for skills, agents, and references

- [ ] **Frontmatter & layout**: frontmatter validates against `framework/schemas/{agent,skill,reference}-frontmatter.schema.json`; `agents/` and `skills/<name>/` contain no subdirectories; `scripts/<family>/` is flattened into one self-contained directory (no `..` `sys.path` walks, standalone `pyproject.toml`, or nested `AGENTS.md`/`CLAUDE.md` files); `host_dependencies` / `tpu_dependencies` are unpinned names with version constraints pinned in `package/requirements-{client,tpu}.txt`.
- [ ] **Declaration & link parity**: every `scripts`, `references`, and `delegates` entry is referenced in prose (`unused-declarations`, `delegation-consistency`), and every markdown link resolves to a declared bundled file (`instruction-links`, `reference-links`).
- [ ] **Deterministic code & prose rigor**: deterministic logic lives in Python/runtime/schemas rather than prompt prose, and CLI entry points fail fast on unknown flags (`parse_args`); prompts use `{{runtime}} compute ...` without restating generic compute (`generic-compute-prose`) or journal mechanics, use `{{built-in:ask_user_question}}` in skills rather than harness-specific tools (`ask_question`, `AskUserQuestion`, `request_user_input`, `invoke_subagent`, `Bash tool`; `legacy-paths`), state no harness-specific subagent waiting (`transcript.jsonl`), cite public TPU facts via `{{agent:tpu-knowledge-base}}`, and contain zero internal/non-public references (`.ci/check_denylist.py`).
- [ ] **Local gates pass**: `make verify && make verify-experimental && make verify-train && make lint && make test && pre-commit run --all-files` (for experimental or training units, also check `python3 framework/compiler.py lint package --manifest <manifest>` reports no findings or warnings on your files).

---

## 5. Contributing a tpu-knowledge-base source

Agents do not answer public TPU facts from memory: they ask the `tpu-knowledge-base`
agent, which quotes the public page and returns its URL. What it may look up is
the index at `package/data/tpu-knowledge-base/index/*.jsonl`, one row per source;
[`index/README.md`](../package/data/tpu-knowledge-base/index/README.md) defines the
row and how its answer is cached. This is the checklist.

1. **A public page**: fetch the URL and confirm it returns 200, then add a row
   with today's date in `verified`, unique `keywords`, and a `prompt` id from
   `package/data/tpu-knowledge-base/prompts/prompt_table.jsonl`
   (`catalog_test.py` fails on an unknown `prompt` or a keyword another row
   already claims).
2. **Knowledge with no citable page** (a playbook, an observed behaviour): an
   authored page under `package/data/tpu-knowledge-base/pages/` and a `page` row
   naming it. The page says which of its claims are documented and which
   observed. A playbook lists the imports it removes under `target_imports`,
   which `preflight_check` reads.
3. `make test-knowledge-base` (included in `make test`) checks the index,
   keyword resolution, and authored pages without fetching anything;
   `make test-package` checks `target_imports` across playbooks.

The knowledge base at `package/data/knowledgebase/` stays in the repository
but is not packaged, and no agent reads it. `make validate-kb` still checks its
index and log, and a change to it still needs a `log.md` entry.

---

## 6. Adding an evaluation dataset or config

Evaluations live under `eval/` as Harbor tasks. The runner
compiles and stages the package for you; see
[`eval/README.md`](../eval/README.md) for running them.

- **Layout**: `eval/datasets/<dataset>/<task>/` holding
  `task.toml`, `instruction.md`, `environment/Dockerfile`, and `tests/`
  (`test.sh` plus `reward.toml` or expectation files). Configs go in
  `eval/configs/<harness>/<name>.yaml`.
- **Naming**: dataset and task directory names must match `^[a-z0-9-]+$`
  (enforced in CI). Directories starting with `_` or `.`
  are ignored, so `_template` is fine.
- **Validate** before running anything: `./eval/verify.sh <dataset>`
  checks the config YAML and every `task.toml` against their schemas.
- **Iterate on one task without Harbor**: `eval/sandbox.sh` prepares a
  container you drive by hand. A config is required, and it decides the arm:

  ```bash
  ./eval/sandbox.sh -c claude-code/agent-suite
  ./eval/sandbox.sh -c claude-code/agent-suite -t tpu-hello-world/baseline
  ./eval/sandbox.sh --baseline                      # the control arm
  ```

  The container runs as a daemon and the shell is a `docker exec` inside a host
  tmux session, so an install or an agent run outlives the terminal that
  started it. Install with `install_claude` (or `install_agy`, `install_codex`),
  then `run_claude` (or `run_agy`, `run_codex`).

  The task's `task.toml` chooses the image and the config chooses the package
  and the environment. The repository is not mounted and no cluster credentials
  are minted, so it is a development loop rather than a faithful replica of a
  scored run.

- **Control arm**: use a config whose name contains `baseline`; it installs
  nothing, so the baseline does not quietly ship the agents you are measuring.
  `eval/configs/README.md` shows the pattern.
