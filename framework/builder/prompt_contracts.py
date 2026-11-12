# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Writes the standard instructions added to agent prompts and skills.

The build appends generated sections to what the author wrote: how to run TPU
jobs, how to use the run journal, which agents it can call and how, and what
its result has to contain. They're built from each agent's or skill's
frontmatter, so they always match its schemas.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from framework import compute, frontmatter, journal, jsonio, paths, utils
from framework.builder import (harness_rules, hook_config, package_loader,
                               tokens)

Path = pathlib.Path


def compute_contract_section(runtime: str, pkg_tag: str) -> str:
    """Render the accelerator-access contract injected into agent prompts.

    Backends and their flags come from the driver registry rather than a
    hand-written list, so the prompt can never advertise a backend this build
    does not have -- an agent following a stale list gets INVALID_BACKEND.

    Every command carries `--package`, because the package's compute
    profiles (environment.json, written by `compute setup apply`) are found
    per package, and the resolver only looks for them when it is told which
    package to look for. Without the flag a set-up package resolves to
    NO_COMPUTE_BACKEND, which reads as "nothing is configured" rather than
    "you did not say whose configuration to read".
    """
    env_hint = (f"the package's installed environment (view it with"
                f" `{runtime} compute inspect --package {pkg_tag} --json`)")
    backends = compute.describe_backends()
    required = compute.describe_required_flags()
    if backends:
        lines = []
        for name, flags in backends.items():
            needed = required.get(name, set())
            rendered = " ".join(f"`{flag} <value>`" +
                                ("" if flag in needed else " (optional)")
                                for flag in flags)
            lines.append(
                f"- `{name}`: {rendered or '(no backend-specific flags)'}")
        backend_text = (
            f"Configuration inherits from {env_hint}. When "
            "overriding `--backend` via CLI, supply that backend's arguments:\n"
            + "\n".join(lines))
    else:
        backend_text = (
            f"Configuration inherits from {env_hint}. This "
            f"build registers no compute backends; run `{runtime} compute "
            "--help` to see what is available before using these commands.")

    compute_cmd = f"{runtime} compute --package {pkg_tag}"
    # `wait` has its own line: it takes its own flags, and its point -- use it
    # instead of a sleep loop -- is lost in a list of action names.
    job_actions = "|".join(a for a in compute.JOB_ACTIONS if a != "wait")
    cli = compute.cli
    setup_guide = (
        f"{paths.central_home_spelling_for(*utils.parse_package_tag(pkg_tag))}"
        f"/{utils.REFERENCES_DIRNAME}/{compute.SETUP_GUIDE.name}")
    return f"""## Accelerator Compute Access
Whenever you need to interact with TPU or compute, execute via `{compute_cmd}`.
Every job, server, log and port-forward goes through it. Don't run `kubectl`, `ssh`, `scp`, `xpk` or `gcloud compute` yourself, not even to read; a recipe that compute runs may. The cluster and its namespace are shared with other runs and with people: compute only acts on jobs this run submitted, and a raw command can't tell which those are. Anything already there belongs to someone else: don't read, copy, benchmark, delete or reuse it. When a compute command fails, read its error, which names the cause and usually the next command; if it still fails, stop and report the command and its error rather than working around it with raw cluster commands.
{backend_text}

Commands:
- Inspect hardware and health: `{compute_cmd} inspect [--json]`
- Releases of jax, jaxlib, libtpu, torch and torch_tpu on the target, for citing docs written for that release: `{compute_cmd} versions [--json]`
- Run synchronous workloads: `{compute_cmd} run --code <path> [--name <job_name>] [--timeout <s>] [--interpreter <python>] [--input <local>[:<name>]]... [--output <name>[:<local>]]... [-- <script args>]`
- Submit asynchronous batch workloads: `{compute_cmd} submit --recipe <path> [--name <job_name>] [--run-id <run_id>] [--interpreter <python>] [--serve] [--input <local>[:<name>]]... [--output <name>]... [-- <recipe args>]`
- Manage jobs and pull artifacts: `{compute_cmd} job [{job_actions}] [--job-id <id>] [--run-id <id>] [--include-serving]`
- Wait for a job instead of sleeping or polling: `{compute_cmd} job wait --job-id <id> [--until running|done|log:<regex>] [--fail-on-log <regex>]... [--timeout <s>]`. Exit {cli.WAIT_REACHED}: reached; {cli.WAIT_FAILURE_LOGGED}: a failure line was logged; {cli.WAIT_TIMED_OUT}: timed out after `--timeout` (default {cli.WAIT_TIMEOUT_SEC}s), so run it again; {cli.WAIT_JOB_ENDED}: the job ended first; 1: `wait` itself failed (read stderr).
- Forward ports to remote jobs: `{compute_cmd} port-forward --port [LOCAL:]REMOTE [--job-id <id>] [--background]`

For later `job` commands, use the `job_id` that `run` and `submit` print, not your `--name`: on `gke_job` the two can differ. Batch jobs are tagged with the session's run automatically; `--run-id <run_id>` overrides that when your request names a different run to work in. Workloads launched with `--serve` are exempt from automatic session reaping: stop one with `job cancel --job-id <id>`, or clear a run's serving jobs with `job reap --include-serving`. Port forwarding blocks in the foreground by default; pass `--background` (or run with `&` in shell) to detach and print the PID so the command does not hang. `job artifacts` defaults to the run's artifact directory (`{paths.RUNS_HOME_SPELLING}/<run_id>/artifacts/<job_id>/`); pass `--out-dir` to override. Each `--output <name>` of a `submit` lands there as `<dir>/<name>` and nowhere else: unlike `run`, `submit` takes no `:<local>` destination, so when you need a file in your work directory, copy it from the path `job artifacts` prints (or pass `--out-dir <work>`). It exits non-zero and names the files in `missing_outputs` when the workload never produced something it declared with `--output`, however the job's own status reads: report that the workload did not produce its result and stop, rather than continuing from whatever the artifact directory happens to hold. Only `--code` and `--recipe` reach the target by themselves: declare files the workload reads with `--input` and files it writes with `--output`; both name the file relative to the workload's working directory on the target, so pass bare names after `--`.

A recipe (`--recipe`, like `run`'s `--code`) is one local script file, copied to the target and executed there with the arguments after `--`, in the working directory that holds its `--input` files and collects its `--output` files. The runner is picked by extension: a `.sh` file runs under `bash`; any other file is a Python script. There is no recipe schema or config format: the script is the whole workload. Python scripts run under the interpreter set in compute setup (`python_interpreter` in the compute profile's `params` in the workspace's `environment.json`), or the target's `python3` if none was configured; `--interpreter <python>` overrides it for one call. Nothing activates a virtualenv -- name its `bin/python3` directly. A `.sh` recipe gets that same interpreter as `$TPU_AGENT_SUITE_PYTHON`: call it instead of a bare `python3`, and look for tools installed beside it (such as a virtualenv's commands) in its directory. For example, a `train.sh` holding `exec "$TPU_AGENT_SUITE_PYTHON" train.py "$1"` runs with `{compute_cmd} submit --recipe train.sh --input train.py -- <config>`; then `job wait --job-id <id> --until done`.

Session start says whether compute is set up for this run, and the message is in your context. `COMPUTE_NOT_CONFIGURED` (no usable profile) and the other `*_NOT_CONFIGURED` codes such as `STAGING_NOT_CONFIGURED` (a field asked only when first needed; `missing` names it and `fix` is the command that asks for it) are fixed with the user by following the compute setup guide, `{setup_guide}`. Only the main session asks the user: elsewhere, stop and report the error with its `missing` and `fix`. A compute command checks the target again first if its profile changed; a failed check lists each failure and its fix, which needs a person (credentials, cluster access, configuration), so report it to the user rather than working around it. `CLUSTER_ACCESS_DENIED` means the cluster refused both the gcloud account and application default credentials; report its `fix` to the user.

`run` holds your shell call open until the workload returns, and a shell call that hits its own timeout kills the run with it. Pass `--timeout <s>`, and if your shell tool takes a per-call timeout, set it to at least `<s>` + {harness_rules.SHELL_TIMEOUT_HEADROOM_S} seconds. Keep `--timeout` at or under {harness_rules.MAX_SYNC_COMPUTE_TIMEOUT_S}; longer work goes through `submit`.

A shared cluster may stop any job a fixed time after it starts running (`declared_duration_minutes` in `compute inspect`'s `details` when configured, or a limit the prompt or an eviction message states), counted from when the job started running (`job status` shows its `start_time`):
- Budget: everything a job does must fit inside the limit, a server's startup (often 15-30 minutes) included. Plan on about 75% of it as usable, and before starting a long phase on a server that is already up, check what is left of it: the limit less the time since its `start_time`.
- Waits: wait in `job wait` calls of `--timeout` 600 or less and run the same one again on exit {cli.WAIT_TIMED_OUT}, never in one long blocking wait or a sleep loop, so a job the cluster stops is noticed within minutes.
- Exit {cli.WAIT_JOB_ENDED}: the job ended first; read its `error`. If the job had run and the `error` says the cluster stopped or deactivated it, or names a time limit, it outgrew the limit: split it into shorter jobs or resume from a checkpoint rather than resubmitting it as it was. If it never started (its pod never became Ready, or Kueue would not admit it), fix that cause -- image, topology, a crash, the queue or priority -- before submitting again.
- If `job status` says Kueue preempted and requeued the job, it is waiting to run again, not lost: run the same wait again, or cancel and resubmit if what is left of the limit no longer fits the work."""


def journal_contract_section(runtime: str, agent_name: str) -> str:
    """Return the prompt section that tells an agent to keep the session journal.

    The session journal is one shared log per run (`journal.jsonl`, shown as
    `experiment.md`) where agents record what they tried, why, and what
    happened, so the user and the next agent can follow the work. Writing it
    needs the `runtime.py journal` command, so this section goes into every
    agent and skill that has a shell, with `agent_name` filled in as the
    entry's author. Agents' own instructions don't need to mention it.
    """
    run_dir = f"{paths.RUNS_HOME_SPELLING}/<run_id>"
    journal_cmd = f"{runtime} journal"
    kinds = "|".join(journal.KINDS)
    return f"""## Session journal
Keep the session's experiment journal, so the user, your caller and the next agent can see what was tried, why, and what it showed. One per run, shared by every agent in it: `{run_dir}/{journal.JOURNAL_FILE}`, re-rendered after every entry to `{run_dir}/{journal.RENDERED_FILE}`. Both sit beside `artifacts/`, not in it: they are the one exception to keeping every output under `artifacts/` or your own work directory. Write them only with these commands, never by editing either file:
- `{journal_cmd} show --run-id <run_id> [--open] [--agent <name>] [--kind <kind>]`
- `{journal_cmd} add --run-id <run_id> --agent {agent_name} --kind <{kinds}> --text "<one or two sentences>" [--parent <id>] [--verdict supported|refuted|inconclusive] [--evidence <path or job id>]... [--metric key=value]...`

`add` prints the entry as JSON; keep its `id` to refer to it later.
- `hypothesis`: before an experiment, what will change, by how much and why; give a number when there is one.
- `experiment`: when it launches; `--parent` its hypothesis, `--evidence` the job id or config path.
- `result`: when it is measured; `--parent` its hypothesis, `--verdict`, `--evidence` the file or job the numbers came from, and the numbers as `--metric`. Say whether they are measured or modelled.
- `rejected`: a hypothesis dropped untested; `--parent` it, and why.
- `decision`: a route, an accept or revert, a stop, with the reason.
- `blocked`: you cannot go on at all; say what would unblock it. A failure you or your caller can retry is a `note`.
- `note`: anything else the next reader needs.

Run `show` before you start, and don't re-run a hypothesis this run already refuted unless the new one says what is different. Write as you go, not at the end: the journal is how a crash, a compaction or the next agent picks up the work. It is a record, not a dependency: if a write is refused, fix the call (the message says what is missing); if the command itself fails, say so in your result and carry on. Never put credentials or tokens in it."""


def skill_frontmatter(name: str, description: str) -> str:
    """Generate YAML frontmatter for a skill."""
    escaped = description.replace('"', "'")
    return f'---\nname: {name}\ndescription: "{escaped}"\n---\n\n'


def delegation_contract_section(
    manifest: dict[str, Any],
    delegates: list[str],
    package_dir: Path | None = None,
    agent_prefix: str = "",
) -> str:
    """Generate delegation request contract section showing required JSON payload shapes."""
    if not delegates:
        return ""
    all_agents = package_loader.get_manifest_agent_map(manifest)

    groups: dict[str, list[str]] = {}
    for d in delegates:
        clean_name = (d[len(agent_prefix):]
                      if agent_prefix and d.startswith(agent_prefix) else d)
        agent_obj = all_agents.get(d) or all_agents.get(clean_name)
        if not agent_obj:
            continue
        accepts = agent_obj.get("accepts")
        if accepts:
            groups.setdefault(accepts, []).append(d)

    if not groups:
        return ""

    lines = [
        "### Delegation request payloads",
        "",
        "When delegating, format the request as a structured JSON object in the delegation prompt/message parameter matching the target contract.",
        "Provide task inputs only; do NOT specify output format templates, response schemas, or examples in the delegation prompt.",
        "",
    ]
    for schema_path, targets in groups.items():
        example = utils.load_schema_example(package_dir, schema_path)
        targets_str = ", ".join(f"`{t}`" for t in targets)
        lines.append(f"- Targets: {targets_str} (`{schema_path}`):")
        if example is not None:
            lines.append("```json")
            lines.append(json.dumps(example, indent=2))
            lines.append("```")
        lines.append("")
    return "\n".join(lines).strip()


def response_deliverable_section(
    agent: dict[str, Any],
    package_dir: Path | None = None,
) -> str:
    """Generate response deliverable contract section showing required JSON result shape."""
    produces = agent.get("produces")
    if not produces:
        return ""
    example = utils.load_schema_example(package_dir, produces)
    lines = [
        "### Response deliverable format",
        "",
        f"When completing your task, your final response MUST be a JSON codeblock conforming to `{produces}`. Runtime validation rejects a response without it.",
        "",
    ]
    if example is not None:
        lines.append("```json")
        lines.append(json.dumps(example, indent=2))
        lines.append("```")
    return "\n".join(lines).strip()


def skill_package_contract(
    manifest: dict[str, Any],
    skill: dict[str, Any],
    harness: str,
    package_dir: Path | None = None,
    agent_prefix: str = "",
) -> str:
    """Build the runtime execution contract section for an installed skill package."""
    delegates = [agent_prefix + name for name in skill.get("delegates", [])]
    # The journal asks the skill to run commands, so it follows what the skill
    # declares, on every harness. Compute guidance is a guard on commands the
    # skill can run at all, so it also goes where the harness can't remove
    # the shell.
    declares_shell = not skill.get("tools") or "shell" in skill["tools"]
    shell_available = (declares_shell
                       or hook_config.SKILL_TOOL_RESTRICTION.get(harness)
                       == hook_config.SKILL_TOOLS_UNENFORCED)
    targets = ", ".join(delegates) or "none"
    pkg_tag, pkg_path, runtime = tokens.package_runtime_paths(manifest)

    sections = [
        "## Installed package contract",
        "",
        f"- Read `.tpu-agent-suite/{paths.ENVIRONMENT_FILE}` (workspace installs) or `{paths.TPU_AGENT_SUITE_HOME_SPELLING}/{paths.ENVIRONMENT_FILE}` (user-scope installs) before delegating — whichever exists; never any other location. It records which machine the work runs on and which interpreter to use for what; guessing either wastes the run. Never create, copy, move, or edit an environment.json by hand: change it only through `{runtime} compute setup` and `{runtime} config`, following `{pkg_path}/{utils.REFERENCES_DIRNAME}/{compute.SETUP_GUIDE.name}`. If the configured location is missing, report that as the blocker.",
        "- Retain the active `run_id` assigned at session start, include it in every delegation request, and use only that run's artifact references.",
    ]

    delegation_mech = harness_rules.delegation_mechanism_lines(
        harness, delegates, root=True).strip()
    if delegation_mech:
        sections.append(delegation_mech)

    sections.extend([
        f"- Direct delegation targets: {targets}. Do not invoke any other package agent directly.",
        "- Run one branch at a time. A subagent may wait for its one nested helper; do not run sibling branches in parallel.",
        f"- Store durable outputs below `{paths.RUNS_HOME_SPELLING}/<run_id>/artifacts/`; never overwrite an existing artifact.",
    ])

    delegation_payloads = delegation_contract_section(
        manifest, delegates, package_dir, agent_prefix=agent_prefix)
    if delegation_payloads:
        sections.extend(["", delegation_payloads])

    if declares_shell:
        sections.extend(["", journal_contract_section(runtime, skill["name"])])

    compute_sec = (compute_contract_section(runtime, pkg_tag)
                   if shell_available else "")
    if compute_sec:
        sections.extend(["", compute_sec])

    return "\n\n" + "\n".join(sections).strip() + "\n"


def skill_document(
    package_dir: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    skill: dict[str, Any],
    agent_prefix: str = "",
) -> str:
    """Generate the SKILL.md of one skill.

    A skill that delegates carries the installed package contract, which
    names its delegates and how to reach them. A skill that does not is a
    plain document: it drives a runtime command rather than orchestrating.
    """
    header = skill_frontmatter(skill["name"], skill["description"])
    # Called for its warning when the harness can't enforce `tools`.
    hook_config.skill_tools_enforced(skill, target["harness"])

    source_path = utils.rooted(package_dir, skill["instructions"],
                               f"instructions for skill {skill['name']}")
    if not source_path.is_file():
        raise utils.TpuAgentSuiteError(
            f"skill {skill['name']!r} declares instructions that do not "
            f"exist: {skill['instructions']}")
    _, body, _ = frontmatter.read_frontmatter_document(source_path)
    rendered_body = tokens.render_harness_tokens(
        tokens.render_agent_references(body, manifest, agent_prefix).rstrip(),
        target["harness"])

    if not skill.get("delegates"):
        return f"{header}{rendered_body}\n"
    contract = skill_package_contract(
        manifest,
        skill,
        target["harness"],
        package_dir=package_dir,
        agent_prefix=agent_prefix,
    )

    return f"{header}{rendered_body}{contract}"


def iter_skill_documents(
    package_dir: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    agent_prefix: str = "",
) -> list[tuple[str, str]]:
    """List (skill name, SKILL.md text) for every released skill."""
    return [(skill["name"],
             skill_document(package_dir, manifest, target, skill,
                            agent_prefix)) for skill in manifest["skills"]]


def write_skill_documents(
    skills_root: Path,
    package_dir: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    agent_prefix: str = "",
) -> None:
    """Write every skill's SKILL.md beneath `skills_root`."""
    for name, document in iter_skill_documents(package_dir, manifest, target,
                                               agent_prefix):
        path = skills_root / name / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding="utf-8")


def load_produces_schema(
    package_dir: Path | None,
    produces: str | None,
) -> dict[str, Any] | None:
    """Load an agent's produces schema, or None if it cannot be read."""
    if not package_dir or not produces:
        return None
    path = Path(package_dir) / produces
    if not path.is_file():
        return None
    try:
        schema = jsonio.load_json(path)
    except (utils.TpuAgentSuiteError, ValueError):
        return None
    return schema if isinstance(schema, dict) else None


def result_contract_lines(schema: dict[str, Any] | None) -> list[str]:
    """State the status vocabulary and payload rules an agent's schema declares.

    Everything is read off the schema, so the generated prompt cannot
    contradict the schema it tells the agent to conform to. The statuses are
    the `status` enum plus any status `status_requires` keys a rule on; a
    schema naming none gets no lines.

    Returns:
        the status line first, then one line per payload rule. Empty when the
        schema names no statuses.
    """
    statuses: list[str] = []
    payload_rules: dict[str, Any] = {}
    declares_artifacts = False
    if schema:
        properties = schema.get("properties", {})
        declares_artifacts = isinstance(properties,
                                        dict) and "artifacts" in properties
        status_prop = properties.get("status", {}) if isinstance(
            properties, dict) else {}
        if isinstance(status_prop, dict) and isinstance(
                status_prop.get("enum"), list):
            statuses = [str(value) for value in status_prop["enum"]]
        if isinstance(schema.get("status_requires"), dict):
            payload_rules = schema["status_requires"]
            # A rule keyed on a status the enum omits is still a status this
            # schema expects, so it belongs in the sentence naming them.
            statuses += [
                str(status) for status in payload_rules
                if str(status) not in statuses
            ]

    if not statuses:
        return []

    vocabulary = ", ".join(f"`{status}`" for status in statuses)
    lines = [f"- Return a result whose `status` is one of {vocabulary}."]

    if payload_rules:
        for status, fields in payload_rules.items():
            if not isinstance(fields, list) or not fields:
                continue
            owed = ", ".join(f"`{field}`" for field in fields)
            article = "An" if status[:1].lower() in "aeiou" else "A"
            lines.append(
                f"- {article} `{status}` result must carry {owed}, each"
                " non-empty; the runtime rejects it otherwise. If the work did"
                f" not produce them, return another status, not `{status}`.")
    elif declares_artifacts and "completed" in statuses:
        # The runtime enforces this only for a schema declaring `artifacts`,
        # and only on the literal status `completed`; anywhere else the line
        # would promise a check that never runs.
        lines.append(
            "- A `completed` result must carry at least one artifact; the"
            " runtime rejects it otherwise. Write findings to a file under the"
            " run's artifacts directory, not only into the summary.")
    return lines


def agent_execution_contract(
    manifest: dict[str, Any],
    agent: dict[str, Any],
    harness: str,
    package_dir: Path | None = None,
    agent_prefix: str = "",
) -> str:
    """Build the runtime execution contract section for an agent prompt."""
    delegates = [agent_prefix + name for name in agent.get("delegates", [])]
    targets = ", ".join(delegates) or "none"
    pkg_tag, _, runtime = tokens.package_runtime_paths(manifest)

    sections = [
        "## Generated execution contract",
        "",
    ]
    if agent.get("accepts"):
        sections.append("- Require the request's `run_id`.")

    produces_schema = load_produces_schema(package_dir, agent.get("produces"))
    result_lines = result_contract_lines(produces_schema)

    if agent.get("produces"):
        sections.extend(result_lines[:1])

    has_shell = "shell" in (agent.get("tools") or [])

    sections.extend([
        f"- Keep every output under `{paths.RUNS_HOME_SPELLING}/<run_id>/artifacts/`.",
        f"- Read only your own run's directory under `{paths.RUNS_HOME_SPELLING}/`, or a run the user specifies. Other runs hold other sessions' work, possibly other people's, so don't search or read them.",
        "- Never overwrite an existing artifact.",
        "- Reject artifact paths that escape the run's artifacts directory.",
        "- Materialize durable outputs.",
        "- Wait for slow work inside the command that started it, not by asking again in a later turn. Every turn resends the whole conversation, so polling for a result costs a full context each time to learn one bit.",
        "- When a command will outlast your shell call's timeout or wait window, raise it for the call past the command's own `--timeout`, or sleep inside the command. Polling is what is left when neither is done, not a technique in its own right.",
    ])

    if agent.get("produces"):
        sections.extend(result_lines[1:])

    delegation_mech = harness_rules.delegation_mechanism_lines(
        harness,
        delegates,
        root=harness_rules.is_root_agent(manifest, agent),
        shell="shell" in agent.get("tools", [])).strip()
    if delegation_mech:
        sections.append(delegation_mech)

    sections.extend([
        f"- Allowed delegation targets: {targets}.",
        "- Run sequentially. Never weaken inherited permissions or approvals.",
    ])

    delegation_payloads = delegation_contract_section(
        manifest, delegates, package_dir, agent_prefix=agent_prefix)
    if delegation_payloads:
        sections.extend(["", delegation_payloads])

    response_deliverable = response_deliverable_section(agent, package_dir)
    if response_deliverable:
        sections.extend(["", response_deliverable])

    if has_shell:
        sections.extend(["", journal_contract_section(runtime, agent["name"])])

    compute_sec = compute_contract_section(runtime, pkg_tag)
    if compute_sec:
        sections.extend(["", compute_sec])

    return "\n\n" + "\n".join(sections).strip() + "\n"


def agent_prompt(
    package_dir: Path,
    manifest: dict[str, Any],
    target: dict[str, Any],
    agent: dict[str, Any],
    agent_prefix: str = "",
) -> str:
    """Generate agent instruction prompt document with generated execution contract."""
    source_path = utils.rooted(package_dir, agent["instructions"],
                               "agent instructions")
    _, body, _ = frontmatter.read_frontmatter_document(source_path)
    rendered_body = tokens.render_harness_tokens(
        tokens.render_agent_references(body, manifest, agent_prefix).rstrip(),
        target["harness"])

    contract = agent_execution_contract(
        manifest,
        agent,
        target["harness"],
        package_dir=package_dir,
        agent_prefix=agent_prefix,
    )

    return f"{rendered_body}{contract}"
