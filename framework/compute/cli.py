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
"""CLI command dispatcher for unified compute subcommands."""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import functools
import json
import pathlib
import re
import subprocess
import sys
import threading
import time
from typing import Any, Callable

from framework import jsonio, paths, utils

from . import config, lease, setup
from .drivers import (ComputeConfigError, ComputeDriver, NotConfigured,
                      StagedFile, base, get_registered_driver_cli_flags, jobs,
                      require_file)

Path = pathlib.Path

format_json = jsonio.format_json

DriverResolver = Callable[..., ComputeDriver]

# `job wait` gives up before an agent's shell (ten minutes) kills it, so it
# can say it timed out and be run again.
WAIT_TIMEOUT_SEC = 540
WAIT_POLL_SEC = 10
# Log lines that mean a workload is failing while its job still reads
# RUNNING; each stalled a serving run for minutes. A match ends the wait, not
# the job.
WAIT_FAIL_PATTERNS = (
    r"RESOURCE_EXHAUSTED",
    r"Raising signal 7",  # SIGBUS; a full /dev/shm is one cause
    r"RuntimeError: cancelled",
    r"No space left on device",
)
# PENDING this long since submission, `job wait` stops saying only "wait
# again" and says why, by the backend's details: the fourth consecutive
# default-length wait. Time running never counts. Advisory only: same exit
# code, nothing cancelled.
WAIT_PENDING_GIVE_UP_SEC = 1800
# Exit codes per outcome. 1 is the runtime's error code, so no outcome uses
# it and a wait that errored never reads as a job that ended.
WAIT_REACHED = 0
WAIT_FAILURE_LOGGED = 2
WAIT_TIMED_OUT = 3
WAIT_JOB_ENDED = 4
# Failed status reads in a row (a raise, or UNKNOWN with an error) before the
# wait errors: rides out an API blip, but fails a mistyped job id fast.
WAIT_READ_ATTEMPTS = 3
# The error code of a job whose record is live but which its backend no
# longer has: another run or a person deleted it. Retrying cannot bring it
# back, so `job wait` stops at once and `job list` reports it per job.
JOB_GONE = "JOB_GONE"
# The error code of a `job cancel` refused because the job is not this run's.
NOT_OWN_JOB = "NOT_OWN_JOB"
# Every `compute job` action
JOB_ACTIONS = ("status", "logs", "artifacts", "cancel", "list", "reap", "wait")


def _strip_leading_delimiter(raw_args: list[str] | None) -> list[str]:
    """Drop the POSIX '--' delimiter argparse.REMAINDER captures verbatim.

    When callers separate runtime flags from the target script's flags
    (`runtime.py compute run --code train.py -- --batch-size 32`), argparse
    hands back `['--', '--batch-size', '32']`. The delimiter must not reach the
    target script as an argument of its own.
    """
    if raw_args and raw_args[0] == "--":
        return raw_args[1:]
    return raw_args or []


def _print_result(payload: dict[str, Any], as_json: bool) -> None:
    print(json.dumps(payload) if as_json else format_json(payload))


# `setup.ENV_CHECK` is asked of a local target's interpreter by
# `compute versions`.
VERSIONS_TIMEOUT_SEC = 60


def _ask_interpreter(python: str) -> dict[str, Any] | None:
    """`env_check.py --versions` under `python`; None if it can't answer."""
    try:
        proc = subprocess.run(
            [python, str(setup.ENV_CHECK), "--versions"],
            capture_output=True,
            text=True,
            timeout=VERSIONS_TIMEOUT_SEC,
            check=False)
        # The last line, in case a sitecustomize printed before it.
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        found = (json.loads(lines[-1])
                 if proc.returncode == 0 and lines else None)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return found if isinstance(found, dict) else None


def target_versions(package: str | None,
                    workspace: Path | str | None = None,
                    *,
                    profile: str | None = None,
                    run_id: str | None = None,
                    backend: str | None = None,
                    interpreter: str | None = None) -> dict[str, Any]:
    """The releases of the per-release-documented packages on the target.

    A local target is asked directly, with the interpreter workloads run
    under, so the answer is current (`source` "live"). Any other target would
    take a remote job per question, so the answer is what the last
    `compute setup verify` of the profile found there (`source` "recorded"),
    with when it ran and, for a container, the image. That record is kept
    under the profile's hash, so once the profile is edited (a new image,
    say) it no longer applies: `source` is None and `stale` says what was
    recorded and for which image. Unknown is never a guess from this
    machine, whose packages need not be the target's.
    """
    env, path = _environment(package, workspace)
    try:
        profiles, _ = config.normalize_environment_profiles(env)
        name, _ = config.resolve_profile_name(env, profile, run_id)
    except ComputeConfigError:
        profiles, name = {}, None
    stored = profiles.get(name or "") or {}
    params = stored.get("params") if isinstance(stored.get("params"),
                                                dict) else {}
    configured = stored.get("backend")
    result: dict[str, Any] = {
        "backend": backend or configured,
        "profile": name,
        "source": None,
        "checked": None,
        "packages": {},
    }
    if (backend or configured) == "local":
        # The local driver's own default when none is configured.
        python = interpreter or params.get(
            "python_interpreter") or sys.executable
        found = _ask_interpreter(python)
        if found is not None:
            result.update(source="live",
                          checked=utils.utc_now_iso(),
                          packages=found,
                          interpreter=python)
        return result
    if path is None or not stored or (backend and backend != configured):
        # Nothing recorded, or another backend than the one checked.
        return result
    results = jsonio.load_json_or_empty(path.parent / paths.VERIFICATION_FILE)
    record = results.get(config.profile_hash(stored))
    recorded = (record.get("target_versions")
                if isinstance(record, dict) else None)
    if isinstance(recorded, dict) and isinstance(recorded.get("packages"),
                                                 dict):
        # A verify that didn't check the software carries the releases
        # forward with the time they were read.
        result.update(source="recorded",
                      checked=recorded.get("checked_at",
                                           record.get("checked_at")),
                      packages=recorded["packages"],
                      image=recorded.get("image"))
        return result
    # A record for an earlier version of this profile: say so, rather than
    # answering with releases of an image the profile no longer runs.
    for old in results.values():
        if (isinstance(old, dict) and old.get("profile") == name
                and isinstance(old.get("target_versions"), dict)):
            then = old["target_versions"]
            result["stale"] = {
                "checked": then.get("checked_at", old.get("checked_at")),
                "image": then.get("image"),
                "configured_image": params.get("image"),
                "packages": then.get("packages") or {},
            }
            break
    return result


def _action_versions(args: argparse.Namespace) -> int:
    """Print the target's package releases; never fails for an unknown one."""
    found = target_versions(getattr(args, "package", None),
                            getattr(args, "workspace", None),
                            profile=getattr(args, "profile", None),
                            run_id=_resolve_run_id(args),
                            backend=getattr(args, "backend", None),
                            interpreter=getattr(args, "interpreter", None))
    if getattr(args, "json", False):
        print(json.dumps(found, sort_keys=True))
        return 0
    backend = found["backend"] or "not configured"
    if found["source"] is None:
        stale = found.get("stale")
        images = (f" (image then: {stale['image']}, now: "
                  f"{stale['configured_image']})" if stale and
                  (stale["image"] or stale["configured_image"]) else "")
        why = (f"the profile changed after the releases were recorded at "
               f"{stale['checked']}{images}. Re-run `compute setup verify` "
               "(with --smoke-test on gke_job) to record them again."
               if stale else
               "no local interpreter answered, and `compute setup verify` "
               "recorded none for this profile.")
        print(f"Backend: {backend}\nReleases on the target are unknown: "
              f"{why}")
        return 0
    how = ("asked just now" if found["source"] == "live" else
           f"recorded by `compute setup verify` at {found['checked']}")
    image = f", image {found['image']}" if found.get("image") else ""
    print(f"Backend: {backend} ({how}{image})")
    for name, version in sorted(found["packages"].items()):
        print(f"  {name:<10} {version or 'not installed'}")
    return 0


def _action_inspect(driver: ComputeDriver, args: argparse.Namespace) -> int:
    _print_result(driver.inspect().to_dict(), getattr(args, "json", False))
    return 0


def format_preflight(report: dict[str, Any]) -> str:
    """A preflight report as a person reads it: one line per check."""
    lines = [
        f"Backend:   {report['backend']}", f"Target:    {report['target']}",
        f"Reachable: {'yes' if report['reachable'] else 'no'}", ""
    ]
    width = max((len(c["name"]) for c in report["checks"]), default=0)
    for check in report["checks"]:
        mark = "ok" if check["status"] == "ok" else check["status"].upper()
        lines.append(
            f"[{mark:<4}] {check['name']:<{width}}  {check['detail']}")
        if check["fix"]:
            lines.append(f"       {'':<{width}}  fix: {check['fix']}")
    return "\n".join(lines + ["", f"Status: {report['status']}"])


def _action_preflight(driver: ComputeDriver, args: argparse.Namespace) -> int:
    """Check the backend's prerequisites; exit 1 when one fails."""
    report = driver.preflight().to_dict()
    print(
        json.dumps(report) if getattr(args, "json", False
                                      ) else format_preflight(report))
    return 1 if report["status"] == "fail" else 0


# Seconds of the session-start hook's timeout kept for the rest of the hook.
SESSION_CHECK_HEADROOM_S = 5
# Per-run file holding the hash of the profile the last check saw.
LAST_CHECK_FILE = "compute-preflight.json"
# The setup procedure every harness gets; the compiler renders it into the
# package's references/.
SETUP_GUIDE = Path(__file__).resolve().parent / "compute-setup.md"


def _environment(
        package: str | None,
        workspace: Path | str | None) -> tuple[dict[str, Any], Path | None]:
    """The package's environment.json and its path; ({}, None) if there is none."""
    if not package:
        return {}, None
    try:
        return paths.resolve_environment(
            package, workspace if workspace is not None else Path.cwd())
    except (utils.TpuAgentSuiteError, OSError, ValueError):
        return {}, None


def _profile_hash(env: dict[str, Any], profile: str | None,
                  run_id: str | None) -> str | None:
    """The hash of the profile a command would use; None without a file.

    `profile` is the `--profile` override. A profile that doesn't exist hashes
    as an empty one, so switching to it still counts as a change.
    """
    if not env:
        return None
    try:
        profiles, _ = config.normalize_environment_profiles(env)
        name, _ = config.resolve_profile_name(env, profile, run_id)
    except ComputeConfigError:
        return None
    return config.profile_hash(profiles.get(name or "") or {})


def _last_check_path(run_id: str | None) -> Path | None:
    """The run's last-check file, or None if the run has no directory."""
    if not run_id:
        return None
    try:
        run_root = paths.get_run_root(run_id)
    except utils.TpuAgentSuiteError:
        return None
    return run_root / LAST_CHECK_FILE if run_root.is_dir() else None


def _last_checked_hash(run_id: str | None) -> str | None:
    """The profile hash the run's last check saw, if any."""
    path = _last_check_path(run_id)
    return (jsonio.load_json_or_empty(path)
            if path else {}).get("profile_hash")


def _resolve_target_driver(package: str | None,
                           workspace: Path | str | None,
                           run_id: str | None,
                           profile: str | None = None) -> ComputeDriver | None:
    """The configured target's driver, or None if it is not set up.

    A profile that is not set up returns None quietly; the compute command
    itself then fails with COMPUTE_NOT_CONFIGURED.
    """
    try:
        return config.resolve_driver(workspace=workspace,
                                     package=package,
                                     profile_override=profile,
                                     run_id=run_id)
    except ComputeConfigError:
        return None
    except Exception as exc:  # noqa: BLE001 - a check must not break callers
        print(f"warning: could not check the compute target: {exc}",
              file=sys.stderr)
        return None


def _check_configured_target(
        driver: ComputeDriver,
        package: str | None,
        workspace: Path | str | None,
        run_id: str | None,
        budget: float | None = None,
        profile: str | None = None) -> dict[str, Any] | None:
    """Preflight `driver`'s target and record which profile was checked.

    `profile` is the `--profile` override `driver` was resolved with, if any.
    Returns the report, or None if the check didn't finish.
    """
    env_hash = _profile_hash(
        _environment(package, workspace)[0], profile, run_id)
    report = _within_budget(lambda: driver.preflight().to_dict(), budget,
                            "compute check")
    if report is None:
        return None
    path = _last_check_path(run_id)
    if path and env_hash:
        try:
            jsonio.write_json(path, {"profile_hash": env_hash})
        except OSError:
            pass
    return report


def _within_budget(call: Callable[[], Any], budget: float | None,
                   what: str) -> Any | None:
    """Run `call()` with a time limit; None if it fails or runs out of time.

    `what` names the call in the warning printed when it doesn't finish.
    """
    if budget is not None and budget <= 0:
        return None
    outcome: dict[str, Any] = {}

    def _call() -> None:
        try:
            outcome["result"] = call()
        except Exception as exc:  # noqa: BLE001 - reported below
            outcome["error"] = exc

    thread = threading.Thread(target=_call, daemon=True)
    thread.start()
    thread.join(budget)
    if "result" not in outcome:
        reason = outcome.get("error") or f"took over {budget}s"
        print(f"warning: the {what} did not finish ({reason})",
              file=sys.stderr)
    return outcome.get("result")


def describe_hardware(profile: base.HardwareProfile | None,
                      package: str | None) -> str:
    """The hardware line for the session-start note.

    Gives the probed accelerator type, device count and HBM per device. If
    any of them is missing, says so instead of guessing from a name.
    """
    details = profile.details if profile else {}
    accelerator = (details.get("accelerator_type")
                   or (profile.accelerator if profile else None))
    devices = details.get("job_devices")
    hbm = details.get("hbm_gb_per_device")
    if accelerator and accelerator != "none" and devices and hbm:
        return (f"Hardware: accelerator_type {accelerator}, {devices} "
                f"devices, {hbm:g} GB HBM per device. Plan from these "
                "numbers, not from the profile, VM or cluster name.")
    note = details.get("hbm_gb_note")
    inspect = config.runtime_command(package, "compute", "inspect --json")
    return (f"Hardware not read at session start"
            f"{f' ({note})' if note else ''}. Don't guess chips or HBM from "
            "the profile, VM or cluster name. Get the real numbers from "
            f"`{inspect}` before sizing anything; if you can't run commands, "
            "leave the sizing to a specialist that can.")


def describe_check(report: dict[str, Any]) -> str:
    """The check's result for the agent: one line if ok, else the report."""
    if not setup.failed_checks(report):
        warnings = [
            c["name"] for c in report["checks"] if c["status"] == "warn"
        ]
        return (
            f"Compute target checked: {report['target']} "
            f"({'warnings: ' + ', '.join(warnings) if warnings else 'ok'})")
    return ("Compute target FAILED its check, so compute commands will fail "
            "until it is fixed. This matters only if your task runs compute "
            "commands; if it does, tell the user what is marked FAIL and its "
            "fix (`compute preflight` checks again after a fix), otherwise "
            "carry on:\n\n" + format_preflight(report))


def _setup_guide() -> Path:
    """compute-setup.md: installed under references/, else the source copy."""
    installed = (config.RUNTIME.parent.parent / utils.REFERENCES_DIRNAME /
                 SETUP_GUIDE.name)
    return installed if installed.is_file() else SETUP_GUIDE


# Verify can submit a smoke-test Job on backends that schedule one (gke_job).
SMOKE_TEST_ASK = (
    "can also run a smoke-test Job, which takes minutes: ask the "
    "user first, and add `--smoke-test` only if they agree.")


def _setup_note(body: str) -> str:
    """A session-start note: a header, `body`, then where the guide is."""
    return (f"## Compute setup\n{body} The full procedure is in "
            f"{_setup_guide()}.")


def _setup_message(package: str | None, lead: str, profile: str | None) -> str:
    """The NOT_CONFIGURED message: `lead`, then the setup steps.

    Every step after the first plan names the profile, so a run or default
    pointing at a missing profile can't send verify to it.
    """
    run = functools.partial(config.runtime_command, package, "compute")
    scheduled = sorted(
        name for name, driver in config.get_registered_drivers().items()
        if driver.scheduled_run)
    smoke = (f" On {', '.join(scheduled)}, it {SMOKE_TEST_ASK}"
             if scheduled else "")
    target = profile or "<name>"
    plan = run(
        f"setup plan{f' --profile {profile}' if profile else ''} --json")
    replan = run(f"setup plan --profile {target} --backend <backend> --json")
    apply = run(f"setup apply --profile {target} --answers '<json>'")
    verify = run(f"setup verify --profile {target}")
    name = ("" if profile else " `<name>` names the new profile; pick it "
            "yourself: the backend's name, or `default` for the first one.")
    return _setup_note(
        f"{lead} Before any TPU work:\n"
        f"1. Run `{plan}`. Show the user any `probe_warnings`. If the only "
        f"question is `backend`, ask the user which backend to use, even "
        f"though it has a suggestion; never choose it yourself. Once they "
        f"answer, run `{replan}` for its questions.{name}\n"
        "2. Ask the user every returned question, optional ones too, in one "
        "round, with "
        "suggestions as defaults. Never answer a question yourself unless "
        "the user accepted its suggestion. Ask a question with no suggestion "
        "as it is; don't run your own commands to find an answer.\n"
        f"3. Run `{apply}` with the answers as a JSON object, including "
        "`backend` if it was asked in step 1. Repeat 2-3 until it returns "
        "ok.\n"
        f"4. Run `{verify}`.{smoke} On failure, ask again for the fields it "
        "names.\n"
        "If the user says skip, continue; compute commands fail with "
        "COMPUTE_NOT_CONFIGURED until setup is done.")


def _failed_message(package: str | None, name: str,
                    checks: list[dict[str, Any]], verify: str) -> str:
    """The FAILED message: each failing check with its fix, then what to do.

    Fields to ask again go through `setup plan --fields`, so the agent asks
    with the questions' prompts, choices and suggestions.
    """
    lines = "".join(
        f"\n- {c['name']}: {(c.get('detail') or '').partition(chr(10))[0]}" +
        (f" Fix: {c['fix']}" if c.get("fix") else "") for c in checks)
    fields = ",".join(setup.fields_of(checks))
    run = functools.partial(config.runtime_command, package, "compute")
    plan = run(f"setup plan --profile {name} --fields {fields} --json")
    apply = run(f"setup apply --profile {name} --answers '<json>'")
    steps = (f"run `{plan}`, ask the user its questions, run `{apply}` with "
             "the answers, then run"
             if fields else "fix what failed, then run")
    return _setup_note(f"Compute profile '{name}' failed its check.{lines}\n"
                       f"Before any TPU work, {steps} `{verify}`. If the user "
                       "says skip, continue.")


def check_target_at_session_start(package: str | None,
                                  workspace: Path | str | None, run_id: str,
                                  hook_timeout: float) -> str | None:
    """The session-start note: this machine's missing software, if any,
    then the compute setup state (`_compute_state_note`)."""
    missing = setup.host_requirements_note()
    compute_note = _compute_state_note(package, workspace, run_id,
                                       hook_timeout)
    parts = [f"## Host software\n{missing}" if missing else "", compute_note]
    return "\n\n".join(part for part in parts if part) or None


def _compute_state_note(package: str | None, workspace: Path | str | None,
                        run_id: str, hook_timeout: float) -> str | None:
    """Work out the run's compute setup state; return one note for the agent.

    NOT_CONFIGURED (no profile selected, an unknown one, or one apply would
    not save) gets the setup steps; UNVERIFIED gets `compute setup verify`;
    FAILED gets the failing checks, their fixes and how to ask again. Only a
    READY profile gets the quick preflight, within the hook's timeout: a pass
    says compute is ready, a failure reads as FAILED. None without an
    environment.json.
    """
    env, path = _environment(package, workspace)
    if path is None:
        return None
    try:
        profiles, _ = config.normalize_environment_profiles(env)
        name, source = config.resolve_profile_name(env, None, run_id)
    except ComputeConfigError as exc:
        return _setup_note(exc.message)
    verify = config.runtime_command(package, "compute", "setup verify")
    if name not in profiles:
        lead = (
            f"{config.unknown_profile_error(name, source, profiles).message}"
            f" {config.use_hint(package)}" if name else config.no_profile_lead(
                package, profiles))
        return _setup_message(package, lead, None)
    results = jsonio.load_json_or_empty(path.parent / paths.VERIFICATION_FILE)
    state, record = setup.setup_state(name, profiles[name], results)
    if state == "NOT_CONFIGURED":
        return _setup_message(
            package, f"Compute profile '{name}' is not fully set up.", name)
    if state == "FAILED":
        return _failed_message(package, name, record.get("failed_checks", []),
                               verify)
    if state == "UNVERIFIED":
        smoke = (f" It {SMOKE_TEST_ASK}" if config.get_driver(
            profiles[name]["backend"]).scheduled_run else "")
        return _setup_note(f"Compute profile '{name}' is set up but not "
                           f"verified. Before any TPU work, run `{verify}`."
                           f"{smoke}")
    ready = f"Compute ready: profile '{name}'."
    unread = f"{ready}\n{describe_hardware(None, package)}"
    budget = hook_timeout - SESSION_CHECK_HEADROOM_S
    if budget <= 0:
        print(
            f"warning: a {hook_timeout}s hook timeout leaves no time for the "
            "compute check; the next compute command runs it",
            file=sys.stderr)
        return unread
    deadline = time.monotonic() + budget
    driver = _resolve_target_driver(package, workspace, run_id)
    if driver is None:
        return unread
    report = _check_configured_target(driver, package, workspace, run_id,
                                      budget)
    if report is None:
        return unread
    failed = setup.failed_checks(report)
    if failed:
        return _failed_message(package, name, failed, verify)
    # The coordinator plans from this note, so it carries the real hardware.
    hardware = _within_budget(driver.inspect, deadline - time.monotonic(),
                              "hardware inspect")
    return (f"{ready} {describe_check(report)}\n"
            f"{describe_hardware(hardware, package)}")


def _recheck_if_environment_changed(args: argparse.Namespace) -> None:
    """Check the target again if its profile changed since the last check."""
    run_id = _resolve_run_id(args)
    if _last_check_path(run_id) is None:
        return
    package = getattr(args, "package", None)
    workspace = getattr(args, "workspace", None)
    profile = getattr(args, "profile", None)
    current = _profile_hash(
        _environment(package, workspace)[0], profile, run_id)
    if current is None or current == _last_checked_hash(run_id):
        return
    print(
        "the compute profile changed since the last compute check; checking "
        "the target again...",
        file=sys.stderr)
    driver = _resolve_target_driver(package, workspace, run_id, profile)
    if driver is None:
        return
    report = _check_configured_target(driver, package, workspace, run_id, None,
                                      profile)
    if report is not None:
        print(describe_check(report), file=sys.stderr)


def _require_local_file(path: str, flag: str) -> None:
    """Reject a missing local file before a driver stages it remotely.

    Drivers validate again for programmatic callers; checking here first means
    the failure is reported against the CLI flag the user typed, rather than
    surfacing midway through SSH or cloud staging.
    """
    require_file(path, flag)


def _staged_files(
        args: argparse.Namespace) -> tuple[list[StagedFile], list[StagedFile]]:
    """The `--input`/`--output` specs of `run` or `submit`, parsed and checked.

    Both commands take the same pair of flags with the same meaning, and a
    difference between how they read them would be a difference nothing here
    intends.
    """
    inputs = [
        StagedFile.parse_input(spec) for spec in getattr(args, "input", [])
    ]
    outputs = [
        StagedFile.parse_output(spec) for spec in getattr(args, "output", [])
    ]
    for staged in inputs:
        _require_local_file(staged.local, "--input")
    return inputs, outputs


def _setup_fix(driver: Any, args: argparse.Namespace,
               fields: list[str]) -> str:
    """The `setup plan` command that asks for `fields` on `driver`'s profile.

    The profile the driver came from, which for a job command may not be the
    run's.
    """
    profile = getattr(driver, "compute_profile", None)
    return config.runtime_command(
        getattr(args, "package", None), "compute",
        f"setup plan{f' --profile {profile}' if profile else ''} "
        f"--fields {','.join(fields)}")


def _action_run(driver: ComputeDriver, args: argparse.Namespace) -> int:
    _require_local_file(args.code, "--code")
    inputs, outputs = _staged_files(args)
    # A scheduled driver tags the job its `run` creates with this run (see
    # `ComputeDriver.run_id`), so a bad id is refused before anything starts.
    run_id = _resolve_run_id(args)
    if run_id:
        _require_valid_run_id(run_id)
    res = driver.run(
        code_path=args.code,
        timeout=args.timeout,
        args=_strip_leading_delimiter(args.run_args),
        python_interpreter=getattr(args, "interpreter", None),
        inputs=inputs,
        outputs=outputs,
        job_name=getattr(args, "name", None),
    )
    fields = res.details.get("setup_fields")
    if fields:
        # An output too large for the pod's log, which only a pod that can
        # upload brings back: say how to set that up, as NotConfigured does.
        fix = _setup_fix(driver, args, fields)
        res.details["fix"] = fix
        res.stderr += f"Fix, then rerun: {fix}\n"
    if getattr(args, "json", False):
        print(json.dumps(res.to_dict()))
    else:
        if res.stdout:
            print(res.stdout, end="" if res.stdout.endswith("\n") else "\n")
        if res.stderr:
            print(
                res.stderr,
                file=sys.stderr,
                end="" if res.stderr.endswith("\n") else "\n",
            )
    return res.exit_code


def _adopt_session(args: argparse.Namespace) -> None:
    """Point `args` at the session's leased run and workspace, unless
    `--workspace` is set.

    The active-run marker is one file per workspace, so with two sessions
    open there it names whichever started last, and the other's `job list`
    and `job reap` would act on a run not theirs. The lease names the harness
    process, so when there is one it wins wherever the cwd is: inside the
    workspace, in /tmp, or in another workspace, since a job launched from
    this session is this session's to reap. To act on another workspace's
    run, pass `--workspace` or `--run-id`.
    """
    if getattr(args, "workspace", None) is not None:
        return
    try:
        found = lease.session_run()
    except Exception:  # noqa: BLE001 - a lookup, never a reason to fail
        return
    if found is None:
        return
    args.session_run_id, args.workspace = found


def _resolve_run_id(args: argparse.Namespace) -> str | None:
    """The run a job operation belongs to: `--run-id`, else the active run.

    The active run is the one leased to this session, or, outside a leased
    session, the one the session-start hook recorded for the workspace, so
    an agent that omits the flag still lands in its own run rather than in
    nobody's. Returns None when neither is known.
    """
    return (getattr(args, "run_id", None)
            or getattr(args, "session_run_id", None)
            or paths.get_active_run_id(getattr(args, "workspace", None)))


def _no_run_message(args: argparse.Namespace, command: str,
                    purpose: str) -> str:
    """Why `command` has no run, naming the directory it looked from."""
    start = getattr(args, "workspace", None) or Path.cwd()
    return (f"{command} needs a run {purpose}, but {start} is not inside a "
            "workspace with an active run. Run it from the workspace "
            "directory, or pass --run-id <run_id> (and --workspace <dir> if "
            "the environment is a workspace install).")


def _require_valid_run_id(run_id: str) -> None:
    """Raise INVALID_FIELD unless `run_id` is a well-formed run id.

    Checked before a job is dispatched, never after: a job created under a
    malformed id outlives the command that made it, since nothing can look
    its run up to report or reap it.
    """
    try:
        utils.validate_run_id(run_id)
    except utils.TpuAgentSuiteError as exc:
        raise base.ComputeConfigError(str(exc),
                                      error_code="INVALID_FIELD") from exc


def _action_submit(driver: ComputeDriver, args: argparse.Namespace) -> int:
    _require_local_file(args.recipe, "--recipe")
    inputs, outputs = _staged_files(args)
    run_id = _resolve_run_id(args)
    if not run_id:
        # A job without a run is a job nothing can reap: the run-end hook
        # finds jobs by run id, so refusing here is what keeps a workload
        # from outliving the session that launched it.
        raise base.ComputeConfigError(_no_run_message(args, "compute submit",
                                                      "to tag the job with"),
                                      error_code="MISSING_FIELD")
    _require_valid_run_id(run_id)
    if not getattr(args, "run_id", None) and lease.has_ended(run_id):
        # The active run outlives its session; a job tagged with it now would
        # be reaped as that session's leftover.
        raise base.ComputeConfigError(
            f"the workspace's active run {run_id} belongs to a session that "
            "has ended, and its jobs are reaped: pass --run-id to submit into "
            "a run explicitly.",
            error_code="MISSING_FIELD",
        )
    serve = getattr(args, "serve", False)
    job_id = driver.submit(
        recipe_path=args.recipe,
        job_name=args.name,
        args=_strip_leading_delimiter(args.submit_args),
        python_interpreter=getattr(args, "interpreter", None),
        run_id=run_id,
        inputs=inputs,
        outputs=outputs,
        serve=serve,
    )
    payload: dict[str, Any] = {"job_id": job_id, "status": "PENDING"}
    if serve:
        payload["serve"] = True
    else:
        try:
            lease.spawn_watcher(run_id)
        except Exception as exc:  # noqa: BLE001 - the job's id is still owed
            print(f"Could not start the lease watcher: {exc}", file=sys.stderr)
    try:
        payload["status"] = driver.get_status(job_id).status
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        # The job was submitted, so the id must still reach the caller. Report
        # why its initial status is unknown rather than passing off the
        # optimistic "PENDING" as an observed one.
        payload["status_error"] = f"{type(exc).__name__}: {exc}"
    landing = paths.get_run_artifacts_dir(run_id) / job_id
    if outputs:
        payload["outputs"] = {
            staged.remote: str(landing / staged.remote)
            for staged in outputs
        }

    if getattr(args, "json", False):
        print(json.dumps(payload))
    else:
        # stdout carries the id alone so `JOB_ID=$(... compute submit ...)`
        # captures something `--job-id` accepts; the prose goes to stderr.
        print(job_id)
        status_msg = f"Submitted job {job_id} (status: {payload['status']})"
        if serve:
            status_msg += " [serving mode: exempt from session reap]"
        print(status_msg, file=sys.stderr)
        if "status_error" in payload:
            print(f"Could not read initial status: {payload['status_error']}",
                  file=sys.stderr)
    # An output always lands in the run's artifact directory, never at the
    # LOCAL of `--output REMOTE:LOCAL`: `job artifacts` may run later, from
    # another directory or session, where LOCAL would point somewhere else.
    # Warn now, so the file isn't looked for at LOCAL.
    for staged in outputs:
        if staged.local != staged.remote:
            print(
                f"warning: submit ignores ':{staged.local}'; {staged.remote} "
                f"will land at {landing / staged.remote} (or pass "
                "`job artifacts --out-dir <dir>`)",
                file=sys.stderr)
    return 0


def _port(value: str, spec: str) -> int:
    """One port of a mapping, checked for range as well as for digits.

    A port outside 1-65535 parses fine and then fails somewhere inside
    kubectl or ssh, where the message is about neither the flag nor the
    number; this is the layer that can say which.
    """
    try:
        port = int(value)
    except ValueError as exc:
        raise base.ComputeConfigError(
            f"Invalid port {spec!r}: ports must be integers.",
            error_code="INVALID_FIELD",
        ) from exc
    if not 1 <= port <= 65535:
        raise base.ComputeConfigError(
            f"Invalid port {spec!r}: {port} is outside the range 1-65535.",
            error_code="INVALID_FIELD",
        )
    return port


def _parse_port_spec(spec: str) -> tuple[int, int]:
    """Parse [LOCAL:]REMOTE port mapping into integer tuple."""
    if ":" in spec:
        local_s, remote_s = spec.split(":", 1)
        return _port(local_s, spec), _port(remote_s, spec)
    port = _port(spec, spec)
    return port, port


def _action_port_forward(driver: ComputeDriver,
                         args: argparse.Namespace) -> int:
    local_port, remote_port = _parse_port_spec(args.port)
    background = getattr(args, "background", False)
    as_json = getattr(args, "json", False)
    rc = driver.port_forward(
        local_port=local_port,
        remote_port=remote_port,
        job_id=getattr(args, "job_id", None),
        background=background,
    )
    if background:
        pid = rc
        # A driver with nothing to tunnel (the local backend) returns no pid.
        # Reporting that as a running forward invites the caller to wait on a
        # process that does not exist.
        if as_json:
            print(
                json.dumps({
                    "status": "RUNNING" if pid else "NOT_NEEDED",
                    "pid": pid or None,
                    "local_port": local_port,
                    "remote_port": remote_port,
                }))
        else:
            if pid:
                print(f"Port forward established in background (PID: {pid}).")
            else:
                print(f"No forward needed: already reachable on "
                      f"localhost:{remote_port}.")
        return 0
    return rc


def _missing_declared_outputs(record: jobs.JobRecord | None,
                              out_dir: Path) -> list[str]:
    """Declared `--output` files of a job that are not in `out_dir` after a fetch.
    """
    declared = record.extra.get("outputs") if record else None
    if not declared:
        return []
    return [
        spec["remote"] for spec in declared
        if not (Path(out_dir) / spec["remote"]).exists()
    ]


def _action_job(driver: ComputeDriver, args: argparse.Namespace) -> int:
    """Run a job lifecycle action.

    Every action honours --json: an agent polling a job should never have to
    scrape prose to learn its state.
    """
    job_action = args.job_action
    as_json = getattr(args, "json", False)
    if not getattr(args, "job_id", None):
        raise base.ComputeConfigError(
            f"--job-id is required for job {job_action}",
            error_code="MISSING_FIELD")

    if job_action == "status":
        _print_result(driver.get_status(args.job_id).to_dict(), as_json)
    elif job_action == "logs":
        logs = driver.fetch_logs(args.job_id)
        if as_json:
            print(json.dumps({"job_id": args.job_id, "logs": logs}))
        else:
            print(logs)
    elif job_action == "artifacts":
        record = jobs.JobRecord.load(args.job_id)
        out_dir = args.out_dir
        if out_dir is None:
            # The job's own run first: artifacts belong where the job was
            # launched from, whichever session is fetching them.
            run_id = (getattr(args, "run_id", None)
                      or (record.run_id if record else "")
                      or getattr(args, "session_run_id", None)
                      or paths.get_active_run_id(
                          getattr(args, "workspace", None)))
            out_dir = (paths.get_run_artifacts_dir(run_id)
                       if run_id else Path.cwd()) / args.job_id
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        retrieved = driver.fetch_artifacts(args.job_id, out_dir)
        missing = _missing_declared_outputs(record, out_dir)
        declared = (record.extra.get("outputs") if record else None) or []
        landed = {
            spec["remote"]: str(out_dir / spec["remote"])
            for spec in declared if spec["remote"] not in missing
        }
        if as_json:
            print(
                json.dumps({
                    "job_id": args.job_id,
                    "out_dir": str(out_dir),
                    "artifacts": [str(p) for p in retrieved],
                    "outputs": landed,
                    "missing_outputs": missing,
                }))
        else:
            print(f"Downloaded {len(retrieved)} artifacts to {out_dir}")
            for remote, path in landed.items():
                print(f"  {remote} -> {path}")
            if missing:
                print(
                    f"Job {args.job_id} declared {len(missing)} output(s) it "
                    f"never produced: {', '.join(missing)}",
                    file=sys.stderr)
        # A workload that exits 0 without writing what it declared has not
        # succeeded from the caller's side: the next thing to run is whatever
        # reads that file, and an empty artifact directory reported as a
        # success is what it reads instead. `run` already promotes this to a
        # failure as it happens; nothing was watching an asynchronous job, so
        # the fetch is the first moment it can be said.
        return 1 if missing else 0
    elif job_action == "cancel":
        _ensure_job_is_this_runs(args)
        driver.cancel(args.job_id)
        if as_json:
            print(json.dumps({"job_id": args.job_id, "status": "CANCELLED"}))
        else:
            print(f"Cancelled job {args.job_id}")
    elif job_action == "wait":
        return _action_job_wait(driver, args)
    return 0


def _ensure_job_is_this_runs(args: argparse.Namespace) -> None:
    """Raise NOT_OWN_JOB unless this run submitted the job."""
    record = jobs.JobRecord.load(args.job_id)
    if record is None:
        raise base.ComputeConfigError(
            f"Job {args.job_id} was not submitted from here: there is no job "
            "record for it, so it belongs to another run or to a person "
            "sharing the cluster. compute only cancels its own jobs; see "
            "yours with `compute job list`. Don't stop it another way: if "
            "the user asked you to, tell them compute can't, so they can "
            "stop it themselves.",
            error_code=NOT_OWN_JOB)
    run_id = _resolve_run_id(args)
    if record.run_id and run_id and record.run_id != run_id:
        raise base.ComputeConfigError(
            f"Job {args.job_id} belongs to run {record.run_id}, not to this "
            f"run ({run_id}). To cancel it anyway, pass --run-id "
            f"{record.run_id}.",
            error_code=NOT_OWN_JOB)


def _compile(pattern: str, flag: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise base.ComputeConfigError(
            f"{flag} is not a valid regular expression: {pattern!r} ({exc})",
            error_code="INVALID_FIELD") from exc


def _first_match(patterns: list[re.Pattern[str]], text: str) -> str | None:
    for line in text.splitlines():
        if any(p.search(line) for p in patterns):
            return line.strip()
    return None


def _wait_outcome(status: str, logs: str, ready: re.Pattern[str] | None,
                  fail: list[re.Pattern[str]],
                  until: str) -> tuple[str, int, str | None] | None:
    """The (outcome, exit code, log line) one poll settles, or None."""
    if status in jobs.TERMINAL_STATUSES:
        # A finished job's own verdict outranks its log lines: a server that
        # printed its ready line and then died is not up. A failure line, if
        # any, says why it ended.
        line = _first_match([ready], logs) if ready else None
        if status == "COMPLETED" and (ready is None or line is not None):
            return "reached", WAIT_REACHED, line
        return "job_ended", WAIT_JOB_ENDED, _first_match(fail, logs)
    # While it runs, the ready line comes first: a server that came up after
    # a failed attempt (a TP fallback, say) has still come up.
    if ready and (line := _first_match([ready], logs)) is not None:
        return "reached", WAIT_REACHED, line
    if (line := _first_match(fail, logs)) is not None:
        return "failure_logged", WAIT_FAILURE_LOGGED, line
    if until == "running" and status == "RUNNING":
        return "reached", WAIT_REACHED, None
    return None


def _action_job_wait(driver: ComputeDriver, args: argparse.Namespace) -> int:
    """Block until a job reaches a state, logs a line, or visibly fails.

    Replaces agents' own sleep loops, which outlive the shell's limit and miss
    a job crashing while it still reads RUNNING. Exits per WAIT_*.
    """
    until = getattr(args, "until", None) or "done"
    ready: re.Pattern[str] | None = None
    if until.startswith("log:") and until[4:]:
        ready = _compile(until[4:], "--until log:")
    elif until not in ("running", "done"):
        raise base.ComputeConfigError(
            f"--until must be running, done or log:<regex>, got {until!r}",
            error_code="INVALID_FIELD")
    defaults = () if getattr(args, "no_default_fail_patterns",
                             False) else WAIT_FAIL_PATTERNS
    fail = [
        _compile(p, "--fail-on-log")
        for p in (*defaults, *(getattr(args, "fail_on_log", None) or ()))
    ]
    timeout = getattr(args, "timeout", None)
    started = time.monotonic()
    deadline = started + (WAIT_TIMEOUT_SEC if timeout is None else timeout)
    status, last, failures, slowest = "UNKNOWN", None, 0, 0.0
    details: dict[str, Any] = {}

    while True:
        poll_started = time.monotonic()
        try:
            read = driver.get_status(args.job_id)
            error = read.error if read.status == "UNKNOWN" else None
        except base.ComputeConfigError as exc:
            if exc.error_code == JOB_GONE:
                raise
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 - retried, then raised
            error = str(exc)
        if error:
            failures += 1
            if failures >= WAIT_READ_ATTEMPTS:
                raise base.ComputeConfigError(
                    f"job wait could not read {args.job_id}'s status "
                    f"{failures} times in a row: {error}",
                    error_code="JOB_STATUS_UNREADABLE")
            settled = None
        else:
            failures, status, error = 0, read.status, read.error
            details = read.details or {}
            if status != last:  # stderr, so stdout stays the one result
                print(f"{args.job_id}: {status}", file=sys.stderr)
                last = status
            logs = ""
            if status != "PENDING":
                try:
                    logs = driver.fetch_logs(args.job_id)
                except Exception:  # noqa: BLE001 - no logs this poll
                    pass
            settled = _wait_outcome(status, logs, ready, fail, until)
        if settled:
            outcome, code, line = settled
            break
        # Stop while the next poll, as slow as the slowest yet, would still
        # finish in time; one slow poll must not carry it past the shell.
        slowest = max(slowest, time.monotonic() - poll_started)
        if time.monotonic() + WAIT_POLL_SEC + slowest > deadline:
            outcome, code, line = "timed_out", WAIT_TIMED_OUT, None
            break
        time.sleep(WAIT_POLL_SEC)

    result: dict[str, Any] = {
        "job_id": args.job_id,
        "outcome": outcome,
        "status": status,
        "waited_sec": round(time.monotonic() - started),
    }
    if line is not None:
        result["line"] = line
    if error:
        result["error"] = error
    if outcome == "timed_out":
        since_submit = (_since_submit_sec(args.job_id, result["waited_sec"])
                        if status == "PENDING" else None)
        if since_submit is not None:
            result["since_submit_sec"] = since_submit
        result["next"] = _wait_next(args.job_id, since_submit, error, details)
    _print_result(result, getattr(args, "json", False))
    return code


def _since_submit_sec(job_id: str, waited_sec: int) -> int:
    """Seconds since the job was submitted, per its record.

    Accumulates across `job wait` calls, unlike `waited_sec`. Without a
    record, this call's wait alone.
    """
    record = jobs.JobRecord.load(job_id)
    if record is None:
        return waited_sec
    try:
        since = datetime.datetime.fromisoformat(record.start_time)
    except (TypeError, ValueError):
        return waited_sec
    if since.tzinfo is None:
        since = since.replace(tzinfo=datetime.timezone.utc)
    now = datetime.datetime.now(datetime.timezone.utc)
    return max(waited_sec, round((now - since).total_seconds()))


WAIT_AGAIN = "Still waiting: run the same `job wait` again."


def _wait_next(job_id: str, since_submit_sec: int | None, error: str | None,
               details: dict[str, Any]) -> str:
    """The `next` hint after a timed-out wait.

    Under WAIT_PENDING_GIVE_UP_SEC since submission, or not PENDING: "wait
    again". Past it, by the backend's `pending_reason`:

    - queued, starting, admitted_no_node: still "wait again", with context.
    - unscheduled: nothing holds a place for this job, so the hint lays out
      the choice: inspect the cluster, cancel, or wait knowingly.
    - none: the backend cannot say, so "wait again".
    """
    if since_submit_sec is None or since_submit_sec < WAIT_PENDING_GIVE_UP_SEC:
        return WAIT_AGAIN
    since = f"{since_submit_sec // 60} min since submission"
    said = f" Backend: {error.rstrip('.')}." if error else ""
    reason = details.get("pending_reason")
    if reason == base.PENDING_QUEUED:
        return (f"Held by the queue, {since}; it admits in order, so waiting "
                f"is how this gets a turn.{said} {WAIT_AGAIN}")
    if reason == base.PENDING_STARTING:
        return f"On a node but not yet running, {since}.{said} {WAIT_AGAIN}"
    if reason == base.PENDING_ADMITTED_NO_NODE:
        return (f"Admitted by the queue but no node fits, {since}: quota is "
                f"held, nodes are not.{said} Run `compute inspect` to see "
                f"whether the pool is full; the queue may requeue it itself. "
                f"{WAIT_AGAIN}")
    if reason == base.PENDING_UNSCHEDULED:
        return (f"No node and no queue, {since}: nothing holds chips for "
                f"this job, and a later submission can take freed ones first."
                f"{said} Run `compute inspect`; if no chips are free, "
                f"`compute job cancel --job-id {job_id}` and report the "
                f"cluster busy, or wait again knowingly.")
    return f"PENDING {since}.{said} {WAIT_AGAIN}"


def _job_target(record: jobs.JobRecord) -> tuple[str, str | None]:
    """The target a job runs on: its backend (untagged is local) and profile."""
    return (record.backend or "local", record.compute_profile)


def _job_drivers(
        args: argparse.Namespace, resolve: DriverResolver,
        run_id: str | None) -> Callable[[jobs.JobRecord], ComputeDriver]:
    """A job's driver, built once per target (`_job_target`)."""
    drivers: dict[tuple[str, str | None], ComputeDriver] = {}

    def driver_for(record: jobs.JobRecord) -> ComputeDriver:
        target = _job_target(record)
        if target not in drivers:
            drivers[target] = resolve(
                workspace=getattr(args, "workspace", None),
                package=getattr(args, "package", None),
                profile_override=getattr(args, "profile", None),
                run_id=run_id,
                job_id=record.job_id)
        return drivers[target]

    return driver_for


def _refresh_statuses(
        records: list[jobs.JobRecord],
        driver_for: Callable[[jobs.JobRecord],
                             ComputeDriver]) -> dict[str, str]:
    """Bring unfinished records up to what their backends report now.

    Most drivers write a record only at submit and when a status read finds
    the job finished, so a running job reads PENDING until then. Drivers
    settle finished jobs inside `get_status`; only PENDING<->RUNNING is written
    here. `driver_for` gives a record's driver. A target's (backend and
    compute profile's) first failure stands for all its records, so one dead
    cluster costs one timeout. Returns why each skipped record was skipped.
    """
    unreachable: dict[tuple[str, str | None], str] = {}
    not_refreshed: dict[str, str] = {}
    for record in records:
        if record.status in jobs.TERMINAL_STATUSES:
            continue
        target = _job_target(record)
        if target not in unreachable:
            try:
                live = driver_for(record).get_status(record.job_id)
                # Drivers that catch their own transport errors (shell)
                # answer UNKNOWN with the error instead of raising it.
                if live.status == "UNKNOWN" and live.error:
                    unreachable[target] = live.error
            except base.ComputeConfigError as exc:
                # A job deleted outside compute is reported alone: the
                # cluster still answered for the others.
                if exc.error_code == JOB_GONE:
                    not_refreshed[record.job_id] = exc.message
                    continue
                unreachable[target] = str(exc)
            except Exception as exc:  # noqa: BLE001 - reported per job
                unreachable[target] = str(exc)
        if target in unreachable:
            not_refreshed[record.job_id] = (
                f"its backend could not be asked: {unreachable[target]}")
            continue
        current = jobs.JobRecord.load(record.job_id) or record
        if (live.status in ("PENDING", "RUNNING")
                and current.status != live.status
                and current.status not in jobs.TERMINAL_STATUSES):
            current.mark(live.status)
    return not_refreshed


def _action_job_list(args: argparse.Namespace, resolve: DriverResolver) -> int:
    """List job records, first asking the backend of each unfinished one.

    Only the run in hand (`--run-id`, else the session's active run) is asked
    about: drivers are built from today's workspace config, and a record from
    an older run may name a cluster or host that config no longer points at.
    `--no-refresh` reads the store alone, for when no backend is reachable.
    """
    not_refreshed: dict[str, str] = {}
    if not getattr(args, "no_refresh", False):
        run_id = _resolve_run_id(args)
        not_refreshed = _refresh_statuses(jobs.list_jobs(run_id=run_id),
                                          _job_drivers(args, resolve, run_id))
    records = jobs.list_jobs(run_id=getattr(args, "run_id", None),
                             status=getattr(args, "status", None))
    if getattr(args, "json", False):
        rows = [dataclasses.asdict(r) for r in records]
        for row in rows:
            if row["job_id"] in not_refreshed:
                row["refresh_error"] = not_refreshed[row["job_id"]]
        print(json.dumps(rows))
    elif not records:
        print("No compute jobs found.")
    else:
        header = f"{'JOB_ID':<28} {'STATUS':<12} {'RUN_ID':<36} {'NAME'}"
        print(header)
        print("-" * len(header))
        for r in records:
            print(f"{r.job_id:<28} {r.status:<12} {(r.run_id or '-'):<36} "
                  f"{r.job_name}")
        for jid, why in not_refreshed.items():
            print(f"  {jid}: showing its last recorded status; {why}",
                  file=sys.stderr)
    return 0


def _action_job_reap(args: argparse.Namespace, resolve: DriverResolver) -> int:
    """Cancel a run's leftover jobs, each through its own backend's driver."""
    if getattr(args, "orphaned", False):
        return _report_lease_reap(args, *lease.reap_orphaned_jobs(resolve))
    run_id = _resolve_run_id(args)
    if not run_id:
        raise base.ComputeConfigError(_no_run_message(args, "compute job reap",
                                                      "to reap"),
                                      error_code="MISSING_FIELD")
    if getattr(args, "watch", False):
        ids, failed = lease.watch(run_id, resolve)
        return _report_lease_reap(args, {run_id: ids} if ids else {}, failed)

    driver_for = _job_drivers(args, resolve, run_id)
    include_serving = getattr(args, "include_serving", False)
    # Read before the reap: afterwards these records are terminal either way.
    serving = [] if include_serving else jobs.serving_jobs(run_id)
    reaped, failed = jobs.reap_jobs(
        run_id,
        lambda job_id: driver_for(jobs.JobRecord.load(job_id)),
        include_serving=include_serving)
    if getattr(args, "json", False):
        print(
            json.dumps({
                "run_id": run_id,
                "reaped": reaped,
                "failed": failed,
                "skipped_serving": serving,
            }))
    else:
        print(f"Reaped {len(reaped)} active compute jobs for run {run_id}")
        for jid in reaped:
            print(f"  Cancelled {jid}")
        # A serving job left holding chips is the reaper's most expensive
        # silence: say what was left and how to clear it.
        for jid in serving:
            print(f"  Left serving job {jid} running "
                  f"(pass --include-serving to cancel it)")
        for jid, why in failed.items():
            print(f"  Could not cancel {jid}: {why}", file=sys.stderr)
    return 0


def _report_lease_reap(args: argparse.Namespace, reaped: dict[str, list[str]],
                       failed: dict[str, str]) -> int:
    """Report a reap of runs whose session ended unseen (see `lease`)."""
    if getattr(args, "json", False):
        print(json.dumps({"reaped": reaped, "failed": failed}))
    else:
        for run_id, ids in reaped.items():
            print(f"Reaped {', '.join(ids)} (run {run_id})")
        for jid, why in failed.items():
            print(f"Could not cancel {jid}: {why}", file=sys.stderr)
    return 0


_ACTIONS: dict[str, Callable[[ComputeDriver, argparse.Namespace], int]] = {
    "inspect": _action_inspect,
    "preflight": _action_preflight,
    "run": _action_run,
    "submit": _action_submit,
    "job": _action_job,
    "port-forward": _action_port_forward,
}


def _print_setup_result(args: argparse.Namespace, result: dict[str,
                                                               Any]) -> int:
    """Print a setup/config result; exit 1 unless its status is ok."""
    print(
        json.dumps(result) if getattr(args, "json", False) else jsonio.
        format_json(result))
    return 0 if result.get("status", "ok") == "ok" else 1


def handle_config_cli(args: argparse.Namespace) -> int:
    """Run `runtime config [show|set|use|remove]`.

    The run is found the way compute commands find it: `--run-id`, else the
    session's run.
    """
    _adopt_session(args)
    where = {"package": args.package, "workspace": args.workspace}
    action = args.config_action or "show"
    if action == "show":
        result = setup.config_show(**where,
                                   run_id=_resolve_run_id(args),
                                   profile_override=args.profile)
    elif action == "set":
        result = setup.config_set(**where,
                                  assignments=args.assignments,
                                  profile=args.profile,
                                  run_id=_resolve_run_id(args),
                                  user_confirmed=args.user_confirmed)
    elif action == "remove":
        result = setup.config_remove(**where,
                                     profile=args.profile_name,
                                     user_confirmed=args.user_confirmed,
                                     run_id=_resolve_run_id(args))
    else:
        result = setup.config_use(**where,
                                  profile=args.profile_name,
                                  as_default=args.default,
                                  run_id=_resolve_run_id(args))
    return _print_setup_result(args, result)


def handle_compute_cli(
    args: argparse.Namespace,
    resolve: DriverResolver | None = None,
) -> int:
    """Execute compute CLI subcommands (inspect, run, submit, job).

    Extracts a positive whitelist of parameters registered by compute drivers,
    resolves the appropriate ComputeDriver via 3-tier precedence, and executes
    the specified subaction.

    Args:
        args: Parsed runtime arguments.
        resolve: Driver resolver, injectable for tests. Defaults to
            `config.resolve_driver`.
    """
    resolve = resolve or config.resolve_driver
    # The session's workspace even after a `cd` out of it: `versions` reads
    # that workspace's environment.json, like every other action.
    _adopt_session(args)
    # Read from environment.json or a local interpreter: no driver, no
    # tools, and no re-check of a remote target to answer it.
    if args.compute_action == "versions":
        return _action_versions(args)
    if args.compute_action == "setup":
        where = {"package": args.package, "workspace": args.workspace}
        if args.setup_action == "plan":
            result = setup.plan_setup(**where,
                                      profile=args.profile,
                                      backend=args.backend,
                                      fields=args.fields,
                                      run_id=_resolve_run_id(args))
        elif args.setup_action == "verify":
            result = setup.verify_setup(**where,
                                        profile=args.profile,
                                        run_id=_resolve_run_id(args),
                                        smoke_test=args.smoke_test)
        else:
            result = setup.apply_setup(**where,
                                       profile=args.profile,
                                       answers=setup.read_answers(
                                           args.answers),
                                       dry_run=args.dry_run)
        return _print_setup_result(args, result)
    if args.compute_action != "preflight":
        _recheck_if_environment_changed(args)

    # Extract only parameter keys that were registered by compute drivers
    param_overrides = {
        param: getattr(args, param)
        for param in get_registered_driver_cli_flags()
        if hasattr(args, param) and getattr(args, param) is not None
    }
    if getattr(args, "interpreter", None) is not None:
        param_overrides["python_interpreter"] = args.interpreter

    # Two job actions span backends rather than address one: each resolves a
    # driver per job, from the backend its record names.
    if args.compute_action == "job" and args.job_action == "list":
        return _action_job_list(args, resolve)
    if args.compute_action == "job" and args.job_action == "reap":
        return _action_job_reap(args, resolve)

    driver = resolve(
        backend_override=getattr(args, "backend", None),
        workspace=getattr(args, "workspace", None),
        param_overrides=param_overrides,
        package=getattr(args, "package", None),
        profile_override=getattr(args, "profile", None),
        run_id=_resolve_run_id(args),
        job_id=getattr(args, "job_id", None),
    )

    action = _ACTIONS.get(args.compute_action)
    if action is None:
        return 0
    # Preflight reports missing tools as checks of its own; everything else
    # shells out to them and needs them first.
    if args.compute_action != "preflight":
        base.require_tools(driver.needed_tools() if isinstance(
            driver, ComputeDriver) else getattr(driver, "required_tools", ()))
    try:
        return action(driver, args)
    except NotConfigured as exc:
        exc.set_fix(_setup_fix(driver, args, exc.missing))
        raise
