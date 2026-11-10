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
"""Runs one hook action and settles its failures.

An action is a package script, a shell command, or one of the builtins in
`BUILTIN_HOOK_ACTIONS`. A hook that times out, crashes, or prints something
that is not a valid reply gets the decision its `on_failure` policy names.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import os
import pathlib
import shlex
import signal
import subprocess
import sys
import threading
from typing import Any, Callable

from framework import (agent_contracts, bg_ledger, compute, harnesses, jsonio,
                       paths, run_state, study, utils)
from framework.hooks import events

Path = pathlib.Path
TpuAgentSuiteError = utils.TpuAgentSuiteError

_HOOK_FAILURE_REPORT_LOCK = threading.Lock()

# How much of a failure's reason is kept: the end, where a traceback names
# its error. An `exit` reason is the hook's whole stderr, which can be long
# and can hold secrets; a denial puts it in the model's context, and the log
# is never rotated.
_HOOK_FAILURE_REASON_CHARS = 1000


def hook_failure(
    kind: str,
    reason: str,
    *,
    hook_name: str,
    on_failure: str,
    event_payload: dict[str, Any],
    workspace: Path,
) -> dict[str, Any]:
    """Record a hook failure and return the verdict specified by `on_failure`.

    `kind` identifies the failure mode (`timeout`, `exit`, `non_json`, `schema`,
    `exception`, `missing`, `dispatch`, or `startup`). Logs to stderr and
    appends a JSONL record to `HOOK_FAILURE_LOG` (best-effort).
    """
    context = event_payload.get("context") or {}
    if len(reason) > _HOOK_FAILURE_REASON_CHARS:
        reason = "..." + reason[-_HOOK_FAILURE_REASON_CHARS:]
    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hook": hook_name,
        "event": event_payload.get("event", ""),
        "tool": event_payload.get("tool"),
        "subagent": event_payload.get("subagent"),
        "kind": kind,
        "reason": reason,
        "on_failure": on_failure,
        "decision": on_failure,
        "session_id": context.get("session_id", ""),
        "run_id": context.get("run_id", ""),
        "package": context.get("package", ""),
        "workspace": str(workspace),
    }
    with _HOOK_FAILURE_REPORT_LOCK:
        print(
            f"runtime: [{hook_name}] hook failed ({kind}): {reason}; "
            f"on_failure={on_failure}",
            file=sys.stderr)
        try:
            log_file = paths.get_tpu_agent_suite_home(
            ) / utils.HOOK_FAILURE_LOG
            log_file.parent.mkdir(parents=True, exist_ok=True)
            # Use a single atomic O_APPEND write to avoid interleaving across processes.
            line = (json.dumps(record) + "\n").encode("utf-8")
            fd = os.open(log_file, os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                         0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        except OSError:
            pass

    if on_failure == "allow":
        return {"decision": "allow"}
    return {"decision": "deny", "reason": reason}


# Maximum leading characters of non-JSON stdout included in failure reasons.
_HOOK_OUTPUT_SHOWN_CHARS = 500


def _run_hook_process(args: str | list[str], *, shell: bool, stdin_text: str,
                      cwd: Path,
                      timeout: float) -> subprocess.CompletedProcess[str]:
    """Run a hook subprocess in its own process group, killing the group on timeout."""
    with subprocess.Popen(args,
                          shell=shell,
                          stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          text=True,
                          cwd=str(cwd),
                          start_new_session=True) as proc:
        try:
            stdout, stderr = proc.communicate(stdin_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            # Use wait() rather than communicate() so a child that left the
            # process group cannot block pipe EOF.
            proc.wait()
            raise
    return subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)


def execute_single_hook_action(
    event_payload: dict[str, Any],
    hook_name: str,
    workspace: Path,
    package: str,
    command: str | None = None,
    script_rel: str | None = None,
    timeout: float = utils.DEFAULT_HOOK_TIMEOUT,
    script_args: list[str] | None = None,
    on_failure: str = utils.DEFAULT_HOOK_ON_FAILURE,
) -> dict[str, Any]:
    """Execute a single hook action (command or script) and validate its output."""
    failure = {
        "hook_name": hook_name,
        "on_failure": on_failure,
        "event_payload": event_payload,
        "workspace": workspace,
    }

    if not command and not script_rel:
        raise TpuAgentSuiteError(
            "Either script_rel or command must be provided")

    try:
        if command:
            p = _run_hook_process(command,
                                  shell=True,
                                  stdin_text=json.dumps(event_payload),
                                  cwd=workspace,
                                  timeout=timeout)
        else:
            try:
                package_dir = paths.get_installed_package_root(package)
                script_path = utils.rooted(package_dir, script_rel,
                                           f"hook '{hook_name}' script")
            except TpuAgentSuiteError as exc:
                return hook_failure("missing", str(exc), **failure)

            if not script_path.is_file():
                return hook_failure(
                    "missing",
                    f"Hook '{hook_name}' script not found at {script_rel}",
                    **failure)

            cmd_args = [sys.executable, str(script_path), *(script_args or [])]
            p = _run_hook_process(cmd_args,
                                  shell=False,
                                  stdin_text=json.dumps(event_payload),
                                  cwd=workspace,
                                  timeout=timeout)
    except subprocess.TimeoutExpired:
        return hook_failure(
            "timeout", f"Hook '{hook_name}' timed out after {timeout} seconds",
            **failure)
    except (OSError, ValueError) as exc:
        return hook_failure(
            "exception", f"Hook '{hook_name}' could not be run: "
            f"{type(exc).__name__}: {exc}", **failure)

    if p.returncode != 0:
        err_r = p.stderr.strip(
        ) or f"Hook script exited with code {p.returncode}"
        return hook_failure("exit", err_r, **failure)

    stdout_clean = p.stdout.strip()
    if not stdout_clean:
        return {"decision": "allow"}

    try:
        out_obj = json.loads(stdout_clean)
    except json.JSONDecodeError:
        shown = stdout_clean[:_HOOK_OUTPUT_SHOWN_CHARS]
        if len(stdout_clean) > _HOOK_OUTPUT_SHOWN_CHARS:
            shown += "..."
        return hook_failure(
            "non_json",
            f"Hook '{hook_name}' emitted non-JSON output on stdout: {shown!r}",
            **failure)

    if not isinstance(out_obj, dict):
        return hook_failure(
            "schema", f"Hook '{hook_name}' output must be a JSON object",
            **failure)

    if "decision" not in out_obj:
        out_obj["decision"] = "allow"

    try:
        schema = utils.get_hook_output_schema()
        s_errors = utils.validate_instance(out_obj, schema)
    except Exception as exc:
        print(f"runtime: Warning - schema validation skipped: {exc}",
              file=sys.stderr)
        s_errors = []
    if s_errors:
        return hook_failure(
            "schema",
            f"Hook '{hook_name}' output violated hook-output.schema.json: " +
            "; ".join(s_errors), **failure)

    return out_obj


_ABANDONED_HOOK_THREADS: list[threading.Thread] = []


def abandoned_hook_threads() -> list[threading.Thread]:
    """Return timed-out builtin hook threads that are still alive.

    When any daemon thread is still running, the caller should exit via
    `os._exit` after flushing stdout so interpreter shutdown does not abort on
    held stdio buffer locks.
    """
    return [t for t in _ABANDONED_HOOK_THREADS if t.is_alive()]


def _run_builtin_on_thread(handler: Callable[..., dict[str, Any]],
                           kwargs: dict[str, Any], outcome: dict[str,
                                                                 Any]) -> None:
    """Invoke `handler(**kwargs)` and store its return value or exception in `outcome`."""
    try:
        outcome["result"] = handler(**kwargs)
    except Exception as exc:  # noqa: BLE001 - becomes the hook's failure
        outcome["error"] = exc


def run_builtin_hook_action(
    handler: Callable[..., dict[str, Any]],
    *,
    package: str,
    workspace: Path,
    event_payload: dict[str, Any],
    hook_name: str,
    timeout: float = utils.DEFAULT_HOOK_TIMEOUT,
    on_failure: str = utils.DEFAULT_HOOK_ON_FAILURE,
) -> dict[str, Any]:
    """Run a builtin hook on a daemon thread with `timeout` and `on_failure` handling."""
    outcome: dict[str, Any] = {}
    handler_kwargs = {
        "package": package,
        "workspace": workspace,
        "event_payload": event_payload,
        "hook_name": hook_name,
        # start-run fits its compute check inside this.
        "timeout": timeout,
    }
    worker = threading.Thread(target=_run_builtin_on_thread,
                              args=(handler, handler_kwargs, outcome),
                              name=f"hook-{hook_name}",
                              daemon=True)
    worker.start()
    worker.join(timeout)

    failure = {
        "hook_name": hook_name,
        "on_failure": on_failure,
        "event_payload": event_payload,
        "workspace": workspace,
    }
    if worker.is_alive():
        _ABANDONED_HOOK_THREADS.append(worker)
        return hook_failure(
            "timeout", f"Hook '{hook_name}' timed out after {timeout} seconds",
            **failure)
    if "error" in outcome:
        exc = outcome["error"]
        return hook_failure(
            "exception",
            f"Hook '{hook_name}' raised {type(exc).__name__}: {exc}",
            **failure)
    return outcome["result"]


def _run_hook_action(
    event_payload: dict[str, Any],
    *,
    hook_name: str,
    workspace: Path,
    package: str,
    timeout: float,
    on_failure: str,
    builtin_handler: Callable[..., dict[str, Any]] | None,
    command: str | None,
    script_rel: str | None,
    script_args: list[str] | None,
) -> dict[str, Any]:
    """One hook action for one event: a builtin, a script or a command."""
    if builtin_handler:
        return run_builtin_hook_action(builtin_handler,
                                       package=package,
                                       workspace=workspace,
                                       event_payload=event_payload,
                                       hook_name=hook_name,
                                       timeout=timeout,
                                       on_failure=on_failure)
    return execute_single_hook_action(event_payload=event_payload,
                                      hook_name=hook_name,
                                      workspace=workspace,
                                      package=package,
                                      command=command,
                                      script_rel=script_rel,
                                      timeout=timeout,
                                      script_args=script_args,
                                      on_failure=on_failure)


def run_hook_action_per_subagent(sub_events: list[dict[str, Any]],
                                 **action: Any) -> list[dict[str, Any]]:
    """Run the hook action concurrently for each subagent in a batched delegation."""
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(sub_events),
            thread_name_prefix=f"hook-{action['hook_name']}") as pool:
        futures = [
            pool.submit(_run_hook_action, sub_event, **action)
            for sub_event in sub_events
        ]
        return [future.result() for future in futures]


def _builtin_start_run(
    package: str,
    workspace: Path,
    event_payload: dict[str, Any] | None = None,
    timeout: float = utils.DEFAULT_HOOK_TIMEOUT,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Builtin hook handler to allocate a new run namespace and emit its run_id."""
    session_id = ""
    if event_payload:
        session_id = events.event_session_id(event_payload)

    # 1. Idempotency check: reuse existing run_id if this session already initialized one
    if session_id:
        existing_run_id = run_state.resolve_session_run_id(session_id)
        if existing_run_id:
            _lease_run(package, workspace, existing_run_id)
            return _allow_run(
                package, workspace, existing_run_id,
                f"Reusing TPU agent suite run namespace: {existing_run_id}",
                timeout)

    # 2. First-time initialization for this session
    run_doc = run_state.start_run(package=package, workspace=workspace)
    run_id = run_doc["run_id"]
    if session_id:
        run_state.record_session_run(session_id, run_id, package)
    _lease_run(package, workspace, run_id)

    return _allow_run(package, workspace, run_id,
                      f"Initialized TPU agent suite run namespace: {run_id}",
                      timeout)


def _allow_run(package: str, workspace: Path, run_id: str, message: str,
               timeout: float) -> dict[str, Any]:
    """Allow the session, adding the compute check to the run context."""
    result = {"decision": "allow", "run_id": run_id, "message": message}
    try:
        note = compute.cli.check_target_at_session_start(
            package, workspace, run_id, timeout)
    except Exception as exc:  # noqa: BLE001 - must never fail session start
        print(f"warning: compute check at session start failed: {exc}",
              file=sys.stderr)
        note = None
    if note:
        run_context = run_state.format_run_context(run_id)
        result["context"] = f"{run_context}\n\n{note}"
        # Later turns replay only the run context, since the check result can go stale.
        result["replay_context"] = run_context
    return result


def _lease_run(package: str, workspace: Path, run_id: str) -> None:
    """Claim the run lease for the current harness process and spawn an orphan sweep."""
    try:
        compute.lease.claim(run_id, compute.lease.session_owner(), workspace,
                            package)
        compute.lease.spawn_sweep(run_id)
    except Exception as exc:  # noqa: BLE001 - reported, the reply still owed
        print(f"runtime: could not lease run {run_id} -- {exc}",
              file=sys.stderr)


def _is_session_over(event_payload: dict[str, Any]) -> bool:
    """Return True if a canonical SessionEnd event marks the end of the session."""
    harness = str(event_payload.get("context", {}).get("harness") or "")
    adaptor = harnesses.get(harness)
    if adaptor is None:
        return event_payload.get("event") == "SessionEnd"
    return adaptor.is_session_over(event_payload)


def _builtin_reap_jobs(
    package: str,
    workspace: Path,
    event_payload: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Builtin hook handler: cancel compute jobs left running when a session ends."""
    event_payload = event_payload or {}
    if not _is_session_over(event_payload):
        return {"decision": "allow"}
    session_id = events.event_session_id(event_payload)
    run_id = (run_state.resolve_session_run_id(session_id)
              if session_id else None)
    if not run_id:
        return {"decision": "allow"}

    def _resolve_driver(job_id: str) -> Any:
        return compute.resolve_driver(workspace=workspace,
                                      package=package,
                                      run_id=run_id,
                                      job_id=job_id)

    try:
        reaped, failed = compute.reap_jobs(run_id, _resolve_driver)
    except Exception as exc:  # noqa: BLE001 - reported, the reply still owed
        print(f"runtime: reap of run {run_id} failed -- {exc}",
              file=sys.stderr)
        return {"decision": "allow"}
    for job_id, why in failed.items():
        print(
            f"runtime: could not cancel job {job_id} of run {run_id} -- "
            f"{why}",
            file=sys.stderr)
    result: dict[str, Any] = {"decision": "allow"}
    if reaped:
        result["message"] = (f"Cancelled {len(reaped)} compute job(s) left "
                             f"running by run {run_id}: {', '.join(reaped)}")
    return result


def _builtin_record_event(
    package: str,
    event_payload: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Builtin hook handler: append a lifecycle event to every open study (best-effort)."""
    payload = event_payload or {}
    context = payload.get("context") or {}
    record = {
        "event": payload.get("event", ""),
        "tool": payload.get("tool", ""),
        "subagent": payload.get("subagent", ""),
        "session_id": context.get("session_id", ""),
        "run_id": context.get("run_id", ""),
        "package": package,
    }
    termination = payload.get("termination") or {}
    if termination:
        record["stop_reason"] = termination.get("reason", "")
        record["fully_idle"] = termination.get("fully_idle")

    try:
        for study_id in study.studies_for_event(record.get("run_id", "")):
            study.append_study_event(study_id, "hook", **record)
    except (TpuAgentSuiteError, OSError):
        pass
    return {"decision": "allow"}


def _installed_build_table(package: str, key: str,
                           what: str) -> dict[str, Any]:
    """Return table `key` of the installed build file, or {} if unavailable.

    `what` names the check that goes without it in the stderr line.
    """
    try:
        package_root = paths.get_installed_package_root(package)
        build_info = jsonio.load_json(package_root /
                                      "tpu-agent-suite-build.json")
    except (TpuAgentSuiteError, OSError, ValueError) as exc:
        print(f"runtime: {what} skipped -- {exc}", file=sys.stderr)
        return {}
    table = build_info.get(key)
    return table if isinstance(table, dict) else {}


def _installed_skill_allowed_tools(
        package: str, workspace: Path) -> dict[str, list[str] | None]:
    """Return the installed build's skill_allowed_tools table, or {} if unavailable."""
    del workspace
    return _installed_build_table(package, "skill_allowed_tools",
                                  "skill tool limits")


def _is_setup_command(package: str, command: str) -> bool:
    """Whether `command` is only this package's `compute setup` or `config`."""
    if any(c in command for c in "$`\n\r"):
        return False
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    if any(set(t) <= set(lexer.punctuation_chars) for t in tokens):
        return False
    # The rendered guide names the runtime as `~/...`; errors use the full path.
    tokens = [os.path.expanduser(t) for t in tokens]
    prefixes = (compute.config.runtime_command(package, "compute", "setup"),
                compute.config.runtime_command(package, "config", ""))
    return any(tokens[:len(p)] == p for p in map(shlex.split, prefixes))


def _builtin_skill_tools(
    package: str,
    workspace: Path,
    event_payload: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Keep the main thread limited to its skill's tools for the rest of the session."""
    payload = event_payload or {}
    raw_input = payload.get("raw_input") or {}
    # Claude Code sets agent_id on events from subagents; only the main thread is limited.
    if isinstance(raw_input, dict) and raw_input.get("agent_id"):
        return {"decision": "allow"}
    session_id = events.event_session_id(payload)
    if not session_id:
        return {"decision": "allow"}
    active_file = run_state.get_active_skill_file(session_id)
    tool = payload.get("tool", "")

    if payload.get("event") == "PostToolUse":
        # Skill isn't in tool-mappings, so it keeps Claude Code's native name and input.
        if tool != "Skill" or payload.get("success") is False:
            return {"decision": "allow"}
        args = payload.get("args") or {}
        invoked = str(args.get("skill") or args.get("command") or "")
        # Plugin installs name skills `<plugin>:<skill>`.
        name = invoked.rsplit(":", 1)[-1].lstrip("/").strip()
        table = _installed_skill_allowed_tools(package, workspace)
        if name not in table:
            # Skills from other packages don't change anything.
            return {"decision": "allow"}
        if table[name] is None:
            # A package skill with no tool limit (e.g. trajectory export) lifts it.
            active_file.unlink(missing_ok=True)
        else:
            jsonio.write_json(
                active_file, {
                    "skill": name,
                    "allowed": list(table[name]),
                    "package": package,
                    "recorded_at": utils.utc_now_iso(),
                })
        return {"decision": "allow"}

    if payload.get("event") != "PreToolUse" or not tool:
        return {"decision": "allow"}
    active = jsonio.load_json_or_empty(active_file)
    allowed = active.get("allowed")
    # An allow list, so tools we've never heard of are blocked too.
    if not isinstance(allowed, list) or tool in allowed:
        return {"decision": "allow"}
    if tool == "shell" and _is_setup_command(
            package, str((payload.get("args") or {}).get("command") or "")):
        return {"decision": "allow"}
    skill = active.get("skill") or "the active skill"
    return {
        "decision":
        "deny",
        "reason":
        (f"The {skill} skill doesn't allow `{tool}`. Delegate this step to"
         " the specialist agent that handles it. If the user asked for a"
         " different skill, invoke that skill first."),
    }


def _builtin_track_background(
    package: str,
    event_payload: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Builtin hook handler keeping the per-agent ledger of background work.

    `PreToolUse`: refuse a background launch from an agent that has no
    shell to wait with (`bg_ledger.background_launch_without_shell`); the one thing
    this handler denies, since the launch would lose its report.
    `PostToolUse`: record a launch the harness reported
    (`bg_ledger.reported_launch`) under the agent the hook fired in.
    `SessionEnd`: clear the session's ledger. The recorder itself never
    denies: one that could veto what it records would turn its own failure
    into a blocked tool. A payload it cannot file is raised instead of
    allowed quietly, since a launch it misses reads as idle at the stop --
    the dispatcher logs the failure and the hook's `on_failure` (allow)
    still decides the verdict.
    """
    payload = event_payload or {}
    harness = str(payload.get("context", {}).get("harness") or "")
    event = payload.get("event")
    if event == "PreToolUse":
        raw_input = payload.get("raw_input")
        if not isinstance(raw_input, dict):
            return {"decision": "allow"}
        agent_type = bg_ledger.background_launch_without_shell(
            raw_input, str(payload.get("tool") or ""), harness,
            _installed_build_table(package, "agent_tools",
                                   "foreground-only check"))
        if agent_type:
            return {
                "decision":
                "deny",
                "reason":
                (f"Launch it in the foreground: `{agent_type}` has no shell,"
                 " so it cannot wait for background work and the report"
                 " would never reach it."),
            }
        return {"decision": "allow"}
    session_id = events.event_session_id(payload)
    if not session_id:
        raise TpuAgentSuiteError(
            "track-background: payload carries no session id; a launch "
            "made here would go unrecorded")
    if event == "SessionEnd":
        if _is_session_over(payload):
            bg_ledger.clear_ledger(session_id)
        return {"decision": "allow"}
    if event != "PostToolUse":
        return {"decision": "allow"}
    raw_input = payload.get("raw_input")
    if not isinstance(raw_input, dict):
        raise TpuAgentSuiteError(
            "track-background: PostToolUse payload carries no raw_input "
            "object; a launch made here would go unrecorded")
    launch = bg_ledger.reported_launch(raw_input,
                                       str(payload.get("tool") or ""), harness)
    if launch is None:
        return {"decision": "allow"}
    idle_mapping = harnesses.subagent_idle_mapping(harness) or {}
    agent_id = str(
        events.extract_canonical_field(raw_input,
                                       idle_mapping.get("agent_id", [])) or "")
    task_id, task_type = launch
    bg_ledger.record_launch(session_id, agent_id, task_id, task_type)
    return {"decision": "allow"}


BUILTIN_HOOK_ACTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "start-run": _builtin_start_run,
    "validate-schema": agent_contracts.builtin_validate_schema,
    "reap-jobs": _builtin_reap_jobs,
    "record-event": _builtin_record_event,
    "skill-tools": _builtin_skill_tools,
    "track-background": _builtin_track_background,
}
