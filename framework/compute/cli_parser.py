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
"""The argument parsers for `runtime.py compute` and `runtime.py config`."""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

from . import config
from .cli import JOB_ACTIONS
from .drivers import get_registered_drivers

Path = pathlib.Path


def register_compute_parser(sub: Any) -> None:
    """Attach `compute` and its subcommands to the runtime's subparsers."""
    # Common compute flags shared between the top-level 'compute' parser and its subcommands
    compute_common = argparse.ArgumentParser(add_help=False)
    compute_common.add_argument(
        "--workspace",
        type=Path,
        default=argparse.SUPPRESS,
        help="Optional workspace root directory override",
    )
    # Backends are validated in `resolve_driver` rather than via `choices` so
    # unknown or broken drivers produce a structured INVALID_BACKEND error.
    compute_common.add_argument(
        "--package",
        default=argparse.SUPPRESS,
        help="Package whose environment.json supplies compute configuration",
    )
    compute_common.add_argument(
        "--profile",
        default=argparse.SUPPRESS,
        metavar="PROFILE",
        help=
        "Named compute profile from environment.json to use for this command",
    )
    compute_common.add_argument(
        "--backend",
        default=argparse.SUPPRESS,
        metavar="BACKEND",
        help=("Override compute backend selection. Registered backends: " +
              (", ".join(config.get_valid_backends()) or "none")),
    )
    compute_common.add_argument(
        "--run-id",
        default=argparse.SUPPRESS,
        metavar="RUN_ID",
        help="Run ID associated with this compute operation",
    )

    # Dynamically register backend-specific CLI flags from all registered drivers
    for driver_cls in get_registered_drivers().values():
        driver_cls.register_cli_args(compute_common)

    # compute
    p_compute = sub.add_parser(
        "compute",
        parents=[compute_common],
        help="Execute compute workloads and inspect accelerator hardware",
    )
    compute_sub = p_compute.add_subparsers(dest="compute_action",
                                           required=True,
                                           help="Compute operation")

    # compute inspect
    p_inspect = compute_sub.add_parser(
        "inspect",
        parents=[compute_common],
        help="Inspect hardware topology, chip allocation, and health")
    p_inspect.add_argument("--json",
                           action="store_true",
                           help="Output raw JSON format")

    # compute versions
    p_versions = compute_sub.add_parser(
        "versions",
        parents=[compute_common],
        help=("Releases of jax, jaxlib, libtpu, torch and torch_tpu on the "
              "target: asked live on a local target, as `compute setup "
              "verify` recorded them elsewhere, else unknown"))
    p_versions.add_argument("--json",
                            action="store_true",
                            help="Output raw JSON format")
    p_versions.add_argument(
        "--interpreter",
        default=None,
        help="Python to ask on a local target (default: the configured one)")

    # compute preflight
    p_preflight = compute_sub.add_parser(
        "preflight",
        parents=[compute_common],
        help=("Check the configured backend's tools, credentials and target; "
              "exits 1 when a check fails"))
    p_preflight.add_argument("--json",
                             action="store_true",
                             help="Output raw JSON format")

    # compute run
    p_run = compute_sub.add_parser(
        "run",
        parents=[compute_common],
        help="Synchronously execute code on the target accelerator")
    p_run.add_argument("--code",
                       required=True,
                       help="Path to code/script file to execute")
    p_run.add_argument("--timeout",
                       type=int,
                       default=None,
                       help=("Execution timeout in seconds. On gke_job it is "
                             "timed from when the pod starts, not including "
                             "Kueue queue time, which gets a separate budget "
                             "of the same length"))
    p_run.add_argument("--name", help="Optional job name (gke_job only)")
    p_run.add_argument(
        "--interpreter",
        default=None,
        help="Path to python interpreter (e.g. /opt/conda/bin/python)")
    p_run.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="LOCAL[:REMOTE]",
        help=("Copy a local file next to the script before it runs; REMOTE "
              "is a path relative to the script's working directory and "
              "defaults to the file's name. Repeatable."))
    p_run.add_argument(
        "--output",
        action="append",
        default=[],
        metavar="REMOTE[:LOCAL]",
        help=("Copy a file the script wrote back to this host after it "
              "exits; REMOTE is relative to the script's working directory, "
              "LOCAL defaults to the same spelling. Repeatable."))
    p_run.add_argument("--json",
                       action="store_true",
                       help="Output raw JSON format")
    p_run.add_argument("run_args",
                       nargs=argparse.REMAINDER,
                       help="Arguments passed to code")

    # compute submit
    p_submit = compute_sub.add_parser(
        "submit",
        parents=[compute_common],
        help="Submit asynchronous batch workload recipe")
    p_submit.add_argument("--recipe",
                          required=True,
                          help="Path to batch recipe script")
    p_submit.add_argument("--name", default=None, help="Optional job name")
    p_submit.add_argument(
        "--interpreter",
        default=None,
        help="Path to python interpreter (e.g. /opt/conda/bin/python)")
    p_submit.add_argument(
        "--serve",
        action="store_true",
        default=False,
        help=("Launch workload in long-lived serving mode; exempts the job "
              "from session-end automatic reaping."))
    p_submit.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="LOCAL[:REMOTE]",
        help=("Copy a local file next to the recipe before it runs; REMOTE "
              "is a path relative to the recipe's working directory and "
              "defaults to the file's name. Repeatable."))
    p_submit.add_argument(
        "--output",
        action="append",
        default=[],
        metavar="REMOTE",
        help=("Fetch a file the recipe wrote when `job artifacts` runs; "
              "REMOTE is relative to the recipe's working directory and "
              "lands at <artifact dir>/REMOTE (or <out-dir>/REMOTE). A "
              "':LOCAL' suffix is ignored. Repeatable."))
    p_submit.add_argument("--json",
                          action="store_true",
                          help="Output raw JSON format")
    p_submit.add_argument("submit_args",
                          nargs=argparse.REMAINDER,
                          help="Arguments passed to recipe")

    # compute port-forward
    p_port_forward = compute_sub.add_parser(
        "port-forward",
        parents=[compute_common],
        help="Forward a local port to a remote workload port",
    )
    p_port_forward.add_argument(
        "--port",
        required=True,
        help="Port mapping: [LOCAL_PORT:]REMOTE_PORT (e.g. 8000 or 8080:8000)",
    )
    p_port_forward.add_argument(
        "--job-id",
        default=None,
        help=
        "Target job ID (required for gke_job unless pod is directly specified)",
    )
    p_port_forward.add_argument(
        "--background",
        "-b",
        action="store_true",
        help="Run the port forward in the background and print its PID",
    )
    p_port_forward.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format",
    )

    # compute job
    p_job = compute_sub.add_parser(
        "job",
        parents=[compute_common],
        help="Manage asynchronous jobs and lifecycle")
    p_job.add_argument(
        "job_action",
        choices=list(JOB_ACTIONS),
        help="Job lifecycle action",
    )
    p_job.add_argument("--job-id", default=None, help="Target job ID")
    p_job.add_argument("--status",
                       default=None,
                       help="Filter jobs by status (for list)")
    p_job.add_argument(
        "--until",
        help="For wait: running, done (default), or log:<regex>",
    )
    p_job.add_argument(
        "--fail-on-log",
        action="append",
        default=None,
        help="For wait: another log regex that ends the wait (repeatable)",
    )
    p_job.add_argument(
        "--no-default-fail-patterns",
        action="store_true",
        help="For wait: drop the built-in failure log patterns",
    )
    p_job.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="For wait: seconds before exit 3 (default: 540)",
    )
    p_job.add_argument(
        "--no-refresh",
        action="store_true",
        help=("For list: show each job's recorded status without asking its "
              "backend, e.g. when the cluster is unreachable"),
    )
    p_job.add_argument("--out-dir",
                       type=Path,
                       default=None,
                       help="Destination directory for artifacts")
    p_job.add_argument(
        "--include-serving",
        action="store_true",
        help=("For reap: also cancel jobs submitted with --serve, which the "
              "session-end reaper leaves running"),
    )
    p_job.add_argument(
        "--orphaned",
        action="store_true",
        help=("For reap: instead of one run, cancel the batch jobs of every "
              "run whose session has ended without reaping them"),
    )
    p_job.add_argument(
        "--watch",
        action="store_true",
        help=("For reap: wait until the run's session has been gone for the "
              "lease grace period, then reap its batch jobs (`submit` starts "
              "this detached)"),
    )
    p_job.add_argument("--json",
                       action="store_true",
                       help="Output raw JSON format")

    setup_common, config_common = _setup_parents()
    p_setup = compute_sub.add_parser("setup",
                                     help="Configure named compute profiles")
    setup_sub = p_setup.add_subparsers(dest="setup_action",
                                       required=True,
                                       help="Setup operation")
    p_setup_plan = setup_sub.add_parser(
        "plan",
        parents=[config_common],
        help="Show what a profile needs, with values suggested from this "
        "machine; writes nothing")
    p_setup_plan.add_argument("--backend",
                              default=argparse.SUPPRESS,
                              help="Backend to plan for (default: the "
                              "profile's)")
    p_setup_plan.add_argument(
        "--fields",
        type=lambda s: [f.strip() for f in s.split(",") if f.strip()],
        default=[],
        help="Comma-separated fields to ask about, instead of the backend's "
        "setup fields")
    p_setup_apply = setup_sub.add_parser(
        "apply",
        parents=[setup_common],
        help="Validate answers and write a complete profile to "
        "environment.json, or report what is missing and write nothing")
    p_setup_apply.add_argument(
        "--profile",
        required=True,
        help="Name of the compute profile to create or update")
    p_setup_apply.add_argument(
        "--answers",
        required=True,
        help="JSON object of {field: value}: a file, '-' for stdin, or inline")
    p_setup_apply.add_argument("--dry-run",
                               action="store_true",
                               help="Validate without writing")
    p_setup_verify = setup_sub.add_parser(
        "verify",
        parents=[config_common],
        help="Check that a profile's target works and record the result")
    p_setup_verify.add_argument(
        "--smoke-test",
        action="store_true",
        help="On backends that schedule Jobs (gke_job), also submit a "
        "smoke-test Job that checks the image and its software; it can take "
        "several minutes")


def register_config_parser(sub: Any) -> None:
    """Attach `config show|set|use|remove` to the runtime's subparsers."""
    _, config_common = _setup_parents()
    p_config = sub.add_parser(
        "config",
        parents=[config_common],
        help="Show or change compute profiles and settings")
    config_sub = p_config.add_subparsers(
        dest="config_action", help="Config operation (default: show)")
    config_sub.add_parser(
        "show",
        parents=[config_common],
        help="Show the profiles and which one this run resolves to")
    p_config_set = config_sub.add_parser(
        "set",
        parents=[config_common],
        help="Change settings, validated like `compute setup apply`")
    p_config_set.add_argument("assignments",
                              nargs="+",
                              metavar="KEY=VALUE",
                              help="Settings to change; KEY= unsets KEY")
    p_config_set.add_argument(
        "--user-confirmed",
        action="store_true",
        help="The user agreed to this change; required for trajectory-"
        "sharing settings")
    p_config_use = config_sub.add_parser(
        "use",
        parents=[config_common],
        help="Point this run (or, with --default, every run) at a profile")
    p_config_use.add_argument("profile_name", help="Profile to use")
    p_config_use.add_argument(
        "--default",
        action="store_true",
        help="Change environment.json's default instead of this run")
    p_config_remove = config_sub.add_parser(
        "remove",
        parents=[config_common],
        help="Remove a named profile from environment.json")
    p_config_remove.add_argument("profile_name", help="Profile to remove")
    p_config_remove.add_argument(
        "--user-confirmed",
        action="store_true",
        help="The user asked to remove this profile; required")


def _setup_parents(
) -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """Return `(setup_common, config_common)` parent parsers for `compute setup` and `config`."""
    # Separate from compute_common so `setup apply` only accepts its own flags,
    # and use SUPPRESS defaults so flags work before or after the subcommand.
    setup_common = argparse.ArgumentParser(add_help=False)
    setup_common.add_argument("--workspace",
                              type=Path,
                              default=argparse.SUPPRESS,
                              help="Workspace root directory")
    setup_common.add_argument(
        "--package",
        default=argparse.SUPPRESS,
        help="Package whose environment.json to read or update")
    setup_common.add_argument("--json",
                              action="store_true",
                              default=argparse.SUPPRESS,
                              help="Output raw JSON format")
    config_common = argparse.ArgumentParser(add_help=False,
                                            parents=[setup_common])
    config_common.add_argument(
        "--profile",
        default=argparse.SUPPRESS,
        help="show: profile to resolve instead of the run's; set: profile to "
        "change; verify: profile to check; plan: profile to plan for, which "
        "may be new (default: the one this run uses)")
    config_common.add_argument(
        "--run-id",
        default=argparse.SUPPRESS,
        help="Run whose profile to use (default: the session's)")
    return setup_common, config_common
