#!/usr/bin/env python3
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
"""Command-line entry point for the TPU agent suite runtime.

Installed hooks and agent instructions run this file by path, so it stays
here. The commands live elsewhere: hook dispatch in `hooks/`, runs and
sessions in `run_state`, contract checks in `agent_contracts`, and compute in
`compute/`.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Any


def _is_hook_dispatch(argv: list[str]) -> bool:
    """Whether `argv` (without the program name) invokes `hook-dispatch`."""
    return argv[:1] == ["hook-dispatch"]


def _hook_dispatch_argv_hints(argv: list[str]) -> dict[str, str | None]:
    """Best-effort parse of `hook-dispatch` flags needed to format a startup failure."""
    lenient = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for flag in ("--harness", "--hook", "--event", "--on-failure",
                 "--handback-event", "--workspace"):
        lenient.add_argument(flag, default=None)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            known, _ = lenient.parse_known_args(argv[1:])
    except SystemExit:
        return {}
    return vars(known)


# Ensure the parent of `framework/` is on sys.path when invoked as a script.
_PACKAGE_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

try:
    from framework import (agent_contracts, compute, harnesses, journal,
                           jsonio, paths, run_state, study, trajectory, utils)
    from framework.hooks import actions, dispatch, events
except Exception as _import_error:  # noqa: BLE001 - becomes the hook's failure
    # If `hook-dispatch` fails during import (e.g. half-removed package or
    # missing package.release.json), honor `--on-failure` instead of crashing
    # with exit 1.
    if not (__name__ == "__main__" and _is_hook_dispatch(sys.argv[1:])):
        raise
    _hints = _hook_dispatch_argv_hints(sys.argv[1:])
    _on_failure = _hints.get("on_failure")
    if not _on_failure:
        raise
    print(
        f"runtime: [{_hints.get('hook') or '?'}] hook failed (startup): "
        f"{type(_import_error).__name__}: {_import_error}; "
        f"on_failure={_on_failure}",
        file=sys.stderr)
    raise SystemExit(0 if _on_failure == "allow" else 2)

Path = pathlib.Path
TpuAgentSuiteError = utils.TpuAgentSuiteError


def resolve_trajectory_target(session_id: str = "",
                              run_id: str = "") -> tuple[str, str, Path]:
    """Return `(session_id, run_id, run_root)` for a trajectory export.

    Accepts either `run_id`, `session_id`, or both, resolving whichever is
    omitted from session records.
    """
    session_id = session_id.strip()
    run_id = run_id.strip()
    if not run_id:
        if not session_id:
            raise TpuAgentSuiteError("no run id given; pass --run-id")
        run_id = run_state.resolve_session_run_id(session_id) or ""
        if not run_id:
            raise TpuAgentSuiteError(
                f"no run is recorded for session {session_id}; pass --run-id")
    if not session_id:
        session_id = run_state.session_for_run(run_id)

    run_root = paths.get_run_root(run_id)
    if not run_root.is_dir():
        raise TpuAgentSuiteError(
            f"run {run_id} has no directory at {run_root}")
    return session_id, run_id, run_root


def resolve_trajectory_user_id(configured: str = "") -> str:
    """Resolve user ID for trajectory export: `configured`, gcloud account, or `$USER`."""
    if configured.strip():
        return configured.strip()
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "account"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        account = (result.stdout or "").strip()
        if account and account != "(unset)":
            return account
    except (OSError, subprocess.SubprocessError):
        pass
    return os.environ.get("USER") or "unknown"


def handle_trajectory_cli(args: argparse.Namespace) -> int:
    """Run `trajectory export`."""
    if not trajectory.supported(args.harness):
        raise TpuAgentSuiteError(trajectory.unsupported_message(args.harness))

    workspace = events.resolve_workspace_root(args.workspace)
    package = args.package or ""
    environment: dict[str, Any] = {}
    if package:
        environment, _ = paths.get_environment(workspace, package)

    params = (environment.get("trajectory") or {}).get("params") or {}
    if params.get("enabled") is not True:
        if params.get("enabled") is False and not (args.destination
                                                   or args.bucket):
            raise TpuAgentSuiteError(
                "trajectory sharing is off: the user chose not to share "
                "sessions, so nothing was exported. It can be turned on "
                "with `config set trajectory_enabled=true ... "
                "--user-confirmed`.")
        params = {}
    share_org = args.share_org or str(params.get("share_org") or "")
    destination = args.destination or ("harbor" if params.get("sink")
                                       == "harbor" else "bucket")

    session_id, run_id, run_root = resolve_trajectory_target(
        args.session_id, args.run_id or "")

    if destination == "harbor":
        missing = trajectory.harbor_missing()
        if missing:
            raise TpuAgentSuiteError(
                f"trajectory destination is 'harbor' but {missing}.")
        bundle = trajectory.resolve_for_export(args.harness, session_id,
                                               args.entrypoint,
                                               args.include_full_session)
        staging = pathlib.Path(
            tempfile.mkdtemp(prefix="tpu-agent-suite-harbor-"))
        harbor_summary = trajectory.export_to_harbor(
            bundle=bundle,
            job_dir=staging / session_id,
            run_root=run_root,
            run_id=run_id,
            message=args.note,
            package=package,
            environment=environment,
            share_org=share_org,
        )
        shutil.rmtree(staging, ignore_errors=True)
        for key in ("job_dir", "trajectory", "manifest", "bundle"):
            harbor_summary.pop(key, None)
        if args.json:
            print(jsonio.format_json(harbor_summary))
        else:
            print(f"Logged session {session_id} to Harbor"
                  f" ({harbor_summary['steps']} steps):"
                  f" {harbor_summary['destination']}")
        return 0

    bucket = args.bucket or str(params.get("bucket") or "")
    user_id = resolve_trajectory_user_id(args.user_id
                                         or str(params.get("user_id") or ""))

    summary = trajectory.export_trajectory(
        harness=args.harness,
        session_id=session_id,
        run_root=run_root,
        run_id=run_id,
        entrypoint=args.entrypoint,
        message=args.note,
        bucket=bucket,
        user_id=user_id,
        package=package,
        environment=environment,
        include_full_session=args.include_full_session,
    )

    if args.json:
        print(jsonio.format_json(summary))
        return 0

    count = len(summary["objects"])
    print(f"Exported {count} objects to {summary['uri']}")
    for entry in summary["objects"]:
        print(f"  {entry['name']} ({entry['bytes']} bytes)")
    print(f"  transcripts: {summary['transcripts']}, "
          f"scoped: {summary['cut']['basis']}")
    for entry in summary["dropped"]:
        print(f"  dropped {entry['path']}: {entry['reason']}")
    return 0


def build_runtime_parser() -> argparse.ArgumentParser:
    """Construct parser for standalone runtime commands."""
    parser = argparse.ArgumentParser(
        prog="runtime",
        description=
        "Embedded runtime execution operations for TPU agent suite packages.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # start-run
    p_start = sub.add_parser(
        "start-run", help="Initialize a new run namespace for a package")
    p_start.add_argument("--package", default="", help="Optional package name")
    p_start.add_argument("--workspace",
                         type=Path,
                         default=None,
                         help="Optional workspace root directory override")
    p_start.add_argument(
        "--profile",
        default=None,
        metavar="PROFILE",
        help="Pin this run to a named compute profile from environment.json",
    )

    # study
    study.register_study_parser(sub)

    # journal
    journal.register_journal_parser(sub)

    # validate
    p_validate = sub.add_parser(
        "validate", help="Validate a JSON instance against a schema")
    p_validate.add_argument("--schema",
                            type=Path,
                            required=True,
                            help="Path to schema JSON file")
    p_validate.add_argument("instance",
                            type=Path,
                            help="Path to JSON instance file to validate")

    # purge-data
    p_purge = sub.add_parser(
        "purge-data",
        help="Purge centralized run history, artifacts, and session records")
    p_purge.add_argument("--force",
                         action="store_true",
                         help="Purge without interactive confirmation")

    # hook-dispatch
    p_hook = sub.add_parser(
        "hook-dispatch",
        help="Normalize and dispatch lifecycle hook invocation to user script",
    )
    p_hook.add_argument(
        "--harness",
        required=True,
        choices=harnesses.supported_harnesses(),
        help="Target agent harness executing the hook",
    )
    p_hook.add_argument("--package", required=True, help="Package name")
    p_hook.add_argument("--hook", required=True, help="Hook name")
    p_hook.add_argument(
        "--event",
        required=True,
        choices=[
            "PreToolUse",
            "PostToolUse",
            "SubagentStop",
            "subagentStop",
            "Stop",
            "SessionStart",
            "sessionStart",
            "PreInvocation",
            "SessionEnd",
            "sessionEnd",
        ],
        help="Lifecycle phase",
    )
    grp_action = p_hook.add_mutually_exclusive_group(required=True)
    grp_action.add_argument(
        "--script",
        help="Script path relative to package directory",
    )
    grp_action.add_argument(
        "--command",
        help="Shell command line to execute directly",
    )
    grp_action.add_argument(
        "--builtin",
        help="Builtin runtime action name (e.g. start-run)",
    )
    p_hook.add_argument(
        "--timeout",
        type=int,
        default=utils.DEFAULT_HOOK_TIMEOUT,
        help=
        f"Hook execution timeout in seconds (default: {utils.DEFAULT_HOOK_TIMEOUT}s)",
    )
    p_hook.add_argument(
        "--on-failure",
        choices=utils.HOOK_FAILURE_POLICIES,
        default=utils.DEFAULT_HOOK_ON_FAILURE,
        help=("Decision when the hook fails to reach one: times out, exits "
              "non-zero, or prints no valid reply "
              f"(default: {utils.DEFAULT_HOOK_ON_FAILURE})"),
    )
    p_hook.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace root directory path",
    )
    p_hook.add_argument(
        "--matcher",
        default=None,
        help="Optional subagent or tool matcher pattern for event filtering",
    )
    p_hook.add_argument(
        "--handback-event",
        default=None,
        help=("Native tool event a SubagentStop hook is registered on as a "
              "hand-back; shapes the reply when dispatch itself fails"),
    )
    p_hook.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Trailing arguments forwarded to the hook script",
    )

    # The compute flags default to SUPPRESS, so these fill in whatever a
    # command line leaves out.
    parser.set_defaults(
        workspace=None,
        backend=None,
        package=None,
        profile=None,
        run_id=None,
    )

    # compute, config
    compute.register_compute_parser(sub)
    compute.register_config_parser(sub)

    # trajectory
    p_trajectory = sub.add_parser(
        "trajectory",
        help="Export a session's trajectory to a bucket you own or to Harbor")
    trajectory_sub = p_trajectory.add_subparsers(dest="trajectory_action",
                                                 required=True,
                                                 help="Trajectory operation")

    # trajectory export
    p_export = trajectory_sub.add_parser(
        "export", help="Bundle and export this session's trajectory")
    p_export.add_argument("--harness",
                          required=True,
                          choices=harnesses.supported_harnesses(),
                          help="Harness the session ran on")
    p_export.add_argument(
        "--note",
        default="",
        help="Optional note from the user, stored in the manifest")
    p_export.add_argument(
        "--entrypoint",
        default="tpu-agent",
        help="Entrypoint skill whose activation starts the bundle")
    p_export.add_argument("--session-id", default="", help="Session to export")
    p_export.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace root, defaulting to the current directory")
    p_export.add_argument(
        "--package",
        default=None,
        help="Package whose environment supplies the destination")
    p_export.add_argument("--run-id",
                          default=None,
                          help="Run whose artifacts ship with the trajectory")
    p_export.add_argument(
        "--bucket",
        default="",
        help="Destination bucket, overriding the configured one")
    p_export.add_argument(
        "--destination",
        default="",
        choices=["", "bucket", "harbor"],
        help="Where to send the trajectory, overriding the configured choice")
    p_export.add_argument(
        "--share-org",
        default="",
        help="Organization to share the uploaded Harbor job with")
    p_export.add_argument("--user-id",
                          default="",
                          help="User identifier used in the object path")
    # Consent for shipping a whole session when the activation point could not
    # be found. Never set this on the user's behalf.
    p_export.add_argument(
        "--all",
        action="store_true",
        dest="include_full_session",
        help="Send the entire session when its starting point cannot be found")
    p_export.add_argument("--json",
                          action="store_true",
                          help="Output raw JSON format")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entrypoint for standalone runtime operations."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if not _is_hook_dispatch(argv):
        args = build_runtime_parser().parse_args(argv)
    else:
        # Route parse or startup errors through `--on-failure` instead of
        # exiting with argparse's code 2.
        usage = io.StringIO()
        try:
            with contextlib.redirect_stderr(usage):
                args = build_runtime_parser().parse_args(argv)
        except SystemExit as exc:
            if exc.code == 0:
                raise
            reason = usage.getvalue().strip().splitlines()
            return dispatch.hook_dispatch_startup_failure(
                _hook_dispatch_argv_hints(argv),
                reason[-1] if reason else f"exit {exc.code}")
        except Exception as exc:  # noqa: BLE001 - becomes the hook's failure
            return dispatch.hook_dispatch_startup_failure(
                _hook_dispatch_argv_hints(argv),
                f"{type(exc).__name__}: {exc}")

    cmd = getattr(args, "subcommand", None)
    try:
        if cmd == "start-run":
            run_doc = run_state.start_run(
                package=args.package,
                workspace=args.workspace or Path.cwd(),
                compute_profile=getattr(args, "profile", None),
            )
            print(jsonio.format_json(run_doc))

        elif cmd == "validate":
            errors = agent_contracts.validate_json_file(
                args.schema, args.instance)
            if errors:
                print("\n".join(errors), file=sys.stderr)
                return 2

        elif cmd == "purge-data":
            run_state.purge_data(force=args.force)

        elif cmd == "hook-dispatch":
            script_args = (args.script_args[1:] if args.script_args
                           and args.script_args[0] == "--" else
                           (args.script_args or []))
            response, code = dispatch.dispatch_hook(
                harness=args.harness,
                package=args.package,
                hook_name=args.hook,
                event=args.event,
                script_rel=args.script,
                command=args.command,
                builtin=args.builtin,
                timeout=args.timeout,
                workspace=args.workspace,
                script_args=script_args,
                matcher=args.matcher,
                on_failure=args.on_failure,
                handback_event=args.handback_event,
            )
            print(json.dumps(response), flush=True)
            if actions.abandoned_hook_threads():
                # Shutdown could abort on a lock the thread holds and turn
                # the verdict into a crash; the verdict is out, so skip it.
                os._exit(code)
            return code

        elif cmd == "study":
            study.dispatch_study_command(args)

        elif cmd == "journal":
            journal.dispatch_journal_command(args)

        elif cmd == "trajectory":
            return handle_trajectory_cli(args)

        elif cmd == "compute":
            return compute.handle_compute_cli(args)

        elif cmd == "config":
            return compute.handle_config_cli(args)

        return 0
    except compute.ComputeConfigError as exc:
        # Machine-readable only when the caller asked for it; `args` may be
        # unbound if the failure happened before parsing completed.
        wants_json = getattr(locals().get("args"), "json", False)
        print(json.dumps(exc.to_dict())
              if wants_json else f"runtime compute: {exc.message}",
              file=sys.stderr)
        return 1
    except TpuAgentSuiteError as exc:
        print(f"runtime: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
