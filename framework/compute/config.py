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
"""Resolves a ComputeDriver: CLI flags over the resolved environment file.

Explicit CLI flags (e.g. backend='gce_tpu_vm') override values from the
package's environment file, which is resolved by install scope -- a
workspace install's <workspace>/.tpu-agent-suite/environment.json, or a
user-scope install's ~/.tpu-agent-suite/environment.json. The workspace is
consulted first, and the central store when the package is not installed
there; a ledger naming another package is not this one's install (see
paths.resolve_environment).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

from framework import jsonio, paths, utils

from .drivers import (ComputeConfigError, ComputeDriver, NotConfigured,
                      describe_discovery_errors, get_driver,
                      get_registered_drivers, jobs, list_drivers, registry)

Path = pathlib.Path
# The runtime CLI, for commands that errors and messages name or that
# start it again (the lease watcher).
RUNTIME = Path(utils.__file__).resolve().parent / "runtime.py"


def get_valid_backends() -> tuple[str, ...]:
    """Return the backends that can actually be constructed.

    The driver registry is the only source of truth: a name is valid exactly
    when a driver registered under it was imported successfully. Deriving this
    from the package schema instead would let the CLI advertise backends that
    cannot be built, or reject ones that can.
    """
    return tuple(list_drivers())


def describe_backends() -> dict[str, list[str]]:
    """Return each registered backend with the CLI flags it declares.

    Built by asking the drivers themselves, so generated documentation cannot
    advertise a backend that is not installed or a flag it does not accept.
    """
    scratch = argparse.ArgumentParser(add_help=False)
    for driver_cls in get_registered_drivers().values():
        driver_cls.register_cli_args(scratch)
    return {name: _ordered_flags(name) for name in list_drivers()}


def _ordered_flags(backend: str) -> list[str]:
    """Flags for a backend, required ones first, alphabetical within each group."""
    declared = registry.DRIVER_CLI_FLAGS.get(backend, [])
    required = registry.DRIVER_REQUIRED_FLAGS.get(backend, set())
    return sorted(declared, key=lambda flag: (flag not in required, flag))


def describe_required_flags() -> dict[str, set[str]]:
    """Return the flags each backend cannot be constructed without."""
    describe_backends()  # ensure every driver has registered its arguments
    return {
        name: set(registry.DRIVER_REQUIRED_FLAGS.get(name, set()))
        for name in list_drivers()
    }


def find_environment_data(workspace: Path | str | None = None,
                          package: str | None = None) -> dict[str, Any]:
    """Resolve environment.json via the single ownership-keyed resolver.

    One implementation lives in paths.resolve_environment; this wrapper
    adapts the return shape (data alone, {} when the package is not
    installed anywhere) and makes the compute layer's two edge decisions
    explicitly, so the resolver itself never guesses: no package means there
    is nothing to resolve, and an unnamed workspace means the current
    working directory (the compute CLI runs inside the workspace it serves).
    """
    if not package:
        return {}
    ws = workspace if workspace is not None else pathlib.Path.cwd()
    data, _source_path = paths.resolve_environment(package, ws)
    return data


def normalize_environment_profiles(
        env_data: dict[str,
                       Any]) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Return `(profiles, default_profile_name)` from environment.json.

    A malformed `profiles` or `default` is rejected rather than read as
    something else.
    """
    profiles, default = env_data.get("profiles", {}), env_data.get("default")
    if not isinstance(profiles, dict) or not all(
            isinstance(v, dict) for v in profiles.values()):
        raise ComputeConfigError(
            "environment.json 'profiles' must map each profile name to an "
            "object with 'backend' and 'params'.",
            error_code="INVALID_FIELD")
    if default is not None and not (isinstance(default, str)
                                    and default.strip()):
        raise ComputeConfigError(
            "environment.json 'default' must be a profile name or null.",
            error_code="INVALID_FIELD")
    return profiles, default


def profile_hash(profile: dict[str, Any]) -> str:
    """A hash of a profile's settings (backend and params), not its name.

    Any change to a setting changes it, so a check recorded against the old
    hash no longer applies.
    """
    settings = {
        "backend": profile.get("backend"),
        "params": profile.get("params") or {}
    }
    return hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def unknown_profile_error(name: str, source: str,
                          profiles: dict[str, Any]) -> ComputeConfigError:
    """UNKNOWN_PROFILE for `name`, naming the tier (`source`) that chose it."""
    what = {
        "--profile": f"Compute profile '{name}' (--profile)",
        "run": f"Compute profile '{name}' (the run's compute_profile)",
        "default":
        f"Default compute profile '{name}' (environment.json 'default')",
        "config use": f"Compute profile '{name}'",
        "config remove": f"Compute profile '{name}'",
    }[source]
    return ComputeConfigError(
        f"{what} is not defined in environment.json "
        f"(available profiles: {', '.join(sorted(profiles)) or 'none'}).",
        error_code="UNKNOWN_PROFILE")


def runtime_command(package: str | None, command: str, rest: str) -> str:
    """The shell command for `runtime.py <command> [--package] <rest>`.

    Errors and the session-start message name the runtime in full, because
    the agent reading them may not know where it is installed.
    """
    package_flag = f" --package {package}" if package else ""
    return f"python3 {RUNTIME} {command}{package_flag} {rest}"


def use_hint(package: str | None) -> str:
    """How to point this run at an existing profile."""
    return f"To use one, run `{runtime_command(package, 'config', 'use <name>')}`."


def no_profile_lead(package: str | None, profiles: dict[str, Any]) -> str:
    """Says no profile is selected, with the ones there are and how to pick."""
    if not profiles:
        return "Compute is not set up yet."
    return ("No compute profile is selected for this run; environment.json "
            f"has {', '.join(sorted(profiles))}. {use_hint(package)}")


def _run_compute_profile(run_id: str | None) -> str | None:
    """`compute_profile` from run `run_id`'s run.json, if it names one."""
    if not run_id:
        return None
    try:
        run_file = paths.get_run_root(run_id) / "run.json"
    except utils.TpuAgentSuiteError:
        return None
    profile = jsonio.load_json_or_empty(run_file).get("compute_profile")
    return profile.strip() if isinstance(profile,
                                         str) and profile.strip() else None


def resolve_profile_name(env_data: dict[str, Any],
                         profile_override: str | None = None,
                         run_id: str | None = None) -> tuple[str | None, str]:
    """Pick the compute profile and the tier that chose it.

    `--profile` (`profile_override`), else run `run_id`'s `compute_profile`,
    else environment.json's `default`. Returns `(name, source)`, where source
    is "--profile", "run" or "default".
    """
    if profile_override is not None:
        if not profile_override.strip():
            raise ComputeConfigError(
                "--profile must be a non-empty profile name.",
                error_code="INVALID_FIELD")
        return profile_override.strip(), "--profile"
    run_profile = _run_compute_profile(run_id)
    if run_profile:
        return run_profile, "run"
    return normalize_environment_profiles(env_data)[1], "default"


def parse_compute_section(
    env_data: dict[str, Any],
    profile: str | None = None,
    source: str = "default",
) -> tuple[str | None, dict[str, Any]]:
    """Return `(backend, params)` of `profile`, else of the default profile.

    Reads the profiles in environment.json (see normalize_environment_profiles);
    `source` is the tier that chose `profile`, for the UNKNOWN_PROFILE message.
    """
    profiles, default_name = normalize_environment_profiles(env_data)
    target = profile if profile is not None else default_name
    if not target:
        return None, {}
    if target not in profiles:
        raise unknown_profile_error(target, source, profiles)
    params = profiles[target].get("params")
    return profiles[target].get("backend"), (dict(params) if isinstance(
        params, dict) else {})


def resolve_driver(
    backend_override: str | None = None,
    workspace: Path | str | None = None,
    param_overrides: dict[str, Any] | None = None,
    package: str | None = None,
    profile_override: str | None = None,
    run_id: str | None = None,
    job_id: str | None = None,
) -> ComputeDriver:
    """Build the driver for one compute command.

    The profile is chosen by `--profile`, then by the `compute_profile` job
    `job_id` was submitted under, so a job is reached on its own target,
    then run `run_id`'s `compute_profile`, then environment.json's `default`
    (see resolve_profile_name). A job is always reached through its own
    backend. A job whose profile was deleted, or recorded before jobs kept
    one, falls back to the run's or default profile, so it can still be
    cancelled. `--backend` and driver flags then override that profile's
    backend and params.

    Without `--backend`, if no profile exists or the profile is invalid even
    after applying driver flags, COMPUTE_NOT_CONFIGURED is raised.
    """
    env_data = find_environment_data(workspace, package)
    profiles = normalize_environment_profiles(env_data)[0]
    record = (jobs.JobRecord.load(job_id)
              if job_id and profile_override is None else None)
    if record and record.compute_profile in profiles:
        name, source = record.compute_profile, "job"
    else:
        if record and record.compute_profile:
            print(
                f"warning: job {job_id}'s compute profile "
                f"'{record.compute_profile}' no longer exists; using this "
                "run's profile",
                file=sys.stderr)
        name, source = resolve_profile_name(env_data, profile_override, run_id)
    env_backend, env_params = parse_compute_section(env_data, name, source)

    # Merge parameter dictionaries with CLI overrides taking precedence over
    # environment.json. Unset CLI arguments default to None in argparse;
    # filtering out None values ensures that an unspecified CLI flag does not
    # unintentionally erase a configured value from environment.json.
    params = {
        **env_params,
        **{
            k: v
            for k, v in (param_overrides or {}).items() if v is not None
        },
    }

    # A record with no backend predates tagging, when jobs were all local.
    backend = (backend_override or (record and (record.backend or "local"))
               or env_backend)
    if not backend and package:
        raise NotConfigured(no_profile_lead(package, profiles),
                            "COMPUTE_NOT_CONFIGURED", ["backend"],
                            runtime_command(package, "compute", "setup plan"))
    if not backend:
        # No --package means no environment file was read; say so.
        raise ComputeConfigError(
            "No compute backend configured. No --package was given, so no"
            " compute profile was read: pass --package <name@version>"
            " (e.g. --package tpu-agent-suite@<version>), or --backend with"
            " its flags.",
            error_code="NO_COMPUTE_BACKEND",
        )

    driver_cls = get_driver(backend)
    if not driver_cls:
        valid_backends = get_valid_backends()
        available = (f"Must be one of {valid_backends}." if valid_backends else
                     "No compute drivers are registered.")
        raise ComputeConfigError(
            f"Invalid backend '{backend}'. {available}"
            f"{describe_discovery_errors()}",
            error_code="INVALID_BACKEND",
        )
    if not backend_override and not record:
        from . import setup  # setup imports this module

        # Validate with setup-field CLI overrides applied (e.g. `--image`).
        # Run-only flags like `--cpu-requests` are ignored so they aren't
        # rejected.
        fields = driver_cls.SETUP_FIELDS.all_fields()
        checked = setup.validate_profile(
            name, profiles[name], {
                k: v
                for k, v in (param_overrides or {}).items()
                if v is not None and k in fields
            })
        if checked["status"] != "ok":
            missing = [q["name"] for q in checked["missing"]]
            invalid = [e["name"] for e in checked["invalid"]]
            problems = [f"missing {', '.join(missing)}"] if missing else []
            if invalid:
                problems.append(f"invalid {', '.join(invalid)}")
            raise NotConfigured(
                f"Compute profile '{name}' is not set up "
                f"({'; '.join(problems)}).", "COMPUTE_NOT_CONFIGURED",
                missing + invalid,
                runtime_command(package, "compute",
                                f"setup plan --profile {name}"))

    driver = driver_cls.from_params(params)
    # A driver built for another backend than the profile's is not that
    # profile's target, so jobs it submits don't name the profile.
    driver.compute_profile = name if backend == env_backend else None
    driver.run_id = run_id
    return driver
