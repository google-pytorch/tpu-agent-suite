# Developing TPU agent suite

How to set up a development environment, what the CI gates are, and the short
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
`scripts/check_denylist.py` on both `git commit` and `git push`. In checkouts
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
make test     # framework unit tests + package translation parity tests
make lint     # advisory content checks on instruction prose (links, legacy paths, delegation)
make build    # compile dist/<harness>/tpu-agent-suite for all four harnesses
```

What each one catches:

- **`verify`** is the merge gate. It validates `package.release.json` against the
  schema, checks that every referenced instruction/schema/script/reference
  exists, enforces the delegation rules (no cycles, at most 3 levels below a
  released skill, every agent reachable), and runs `validate-kb`. A *released
  skill* is one that `package.release.json` lists under `skills`, or that `--skills`
  names for a build; those are the roots the compiler starts from. It also runs the
  four lint checks that are decidable (`instruction-links`, `reference-links`,
  `environment-contract`, `workspace-config-paths`).
- **`test`** runs `make test-framework` and `make test-package`. The package
  tests include one translation parity suite per harness; if you change a
  translator or an agent's frontmatter, these are what break.
- **`lint`** reports everything else. Several checks are heuristics, so it is
  advisory (kept out of required status checks), but CI posts its findings as
  PR annotations and you are expected to clear the real ones.
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

---

## 3. What CI runs

Every pull request runs the following checks:

| Gate | Runs | Required? |
| :--- | :--- | :--- |
| **Package & unit tests** | `make verify`, `make verify-experimental`, `make validate-kb KB_BASE=origin/<base>`, `make test`, `make test-eval`, `make build`, `compiler.py lint`, `make coverage`, `make coverage-diff` | yes (`coverage-diff` advisory) |
| **Formatting & denylist** | The `pre-commit` hook set plus `scripts/check_denylist.py` on PR diffs and commit messages | yes |
| **Dataset naming** | Eval dataset and task folder naming (`^[a-z0-9-]+$`) | yes, when `eval/**` changes |
| **Code owners** | Code-owner approval on protected onboarding, framework, documentation, and infrastructure paths | yes |
| **Smoke evaluation** | The `smoke-check` Harbor evaluation when package, framework, or evaluation files change | reported |

Release builds compile standalone per-harness archives without internal
overlays, scan the compiled `dist/` and curated `release-notes/v<version>.md`
(copied from `release-notes/TEMPLATE.md`) with `scripts/check_denylist.py`,
smoke-test installing the `claude-code` archive in a clean workspace, and emit
`SHA256SUMS`.

Run the same commands locally before pushing; CI adds nothing you cannot
reproduce with the loop above plus `pre-commit run --all-files`.

---

## 4. Adding an agent to an existing package

`package/README.md` walks through converting a whole skill from scratch.
Adding one agent to a domain that already exists is shorter:

1. **Write the instruction file** at `package/agents/<agent-name>.md`.
   The YAML frontmatter is the manifest:

   ```yaml
   ---
   name: my-agent                       # unique, lowercase hyphen-case
   description: One line on what it does.
   tools: [read]                        # minimum of read | write | edit | shell | search
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
   by a typed-out path. A path in the user's workspace needs something in front
   of it: `./scripts/train.py`. The full list is in
   [Tokens](design/package.md#tokens).

   Don't write instructions for the session journal: an agent with `shell`
   gets them in its generated contract.

2. **Nothing to register.** The compiler finds an agent by following
   `delegates` from the released skills, so wiring the delegation (next step)
   is what makes it part of the build. Pin any new pip dependencies in
   `package/requirements-client.txt` / `requirements-tpu.txt`
   (unpinned entries fail `verify`). Scripts go under
   `package/scripts/<family>/`, references under
   `package/references/<family>/`; a reference may declare its own
   scripts, references and delegates in optional frontmatter, and every
   reader inherits them.

3. **Wire the delegation.** Add the name to the `delegates` list of the agent
   that will call it, and mention it in that agent's prose with an
   `{{agent:my-agent}}` token. If it is a new top-level workflow, add a routing
   line in
   `package/skills/tpu-agent/SKILL.md` and to that
   file's `delegates`.

4. **Add schemas** under `schemas/` if you introduced typed contracts; every
   `schemas/*.json` is registered and validated automatically.

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

2. **Register it** in `package/package.release.json` under `skills`, by name:

   ```json
   "skills": ["tpu-agent", "tpu-agent-export-trajectory", "my-skill"]
   ```

   The name is the directory under `skills/`, and the document's frontmatter
   must agree with it, so the two cannot drift apart. No skill is special:
   the order in `skills` means nothing.

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
   every declared skill.

---

## 5. Contributing a tpu-knowledge-base source

Agents do not answer public TPU facts from memory: they ask the `tpu-knowledge-base`
agent, which quotes the public page and returns its URL. What it may look up is
the index at `package/data/tpu-knowledge-base/index/*.jsonl`, one row per source;
[`index/README.md`](../package/data/tpu-knowledge-base/index/README.md) defines the
row and how its answer is cached. This is the checklist.

1. **A public page**: fetch the URL and confirm it returns 200, then add a row
   with today's date in `verified` and a `prompt` and `template` that exist.
   `index.py` refuses a row whose prompt does not exist or whose keyword
   another row already claims.
2. **Knowledge with no citable page** (a playbook, an observed behaviour): an
   authored page under `package/data/tpu-knowledge-base/pages/` and a `page` row
   naming it. The page says which of its claims are documented and which
   observed. A playbook lists the imports it removes under `target_imports`,
   which `preflight_check` reads.
3. `make test-package` checks the index, the authored pages and the golden
   rows without fetching anything.

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
  `eval/configs/<config>.yaml`.
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
