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
"""Registry of compute driver backends and their CLI flags."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .base import ComputeDriver

DRIVER_REGISTRY: dict[str, type["ComputeDriver"]] = {}
REGISTERED_DRIVER_CLI_FLAGS: set[str] = set()

# Import failures hit while auto-discovering driver modules, keyed by module
# name. Discovery cannot raise -- a driver that fails to import must not take
# the whole CLI down -- but the failure has to stay visible, otherwise a
# hydrated internal driver with a missing dependency silently disappears and
# the user is told only that their backend is "invalid".
DISCOVERY_ERRORS: dict[str, str] = {}

# Backends registered by driver files loaded from outside the package (see
# `external.py`), mapped to the file or directory each one came from. Built-in
# backends are not listed here. The setup schema check accepts these backend
# names and their setup fields, which the package schema cannot list because
# it does not know them.
EXTERNAL_DRIVERS: dict[str, str] = {}

# Option strings each backend declared, keyed by backend name. Lets generated
# documentation list a backend's real flags instead of a hand-maintained copy
# that drifts as drivers come and go.
DRIVER_CLI_FLAGS: dict[str, list[str]] = {}

# Flags a backend cannot be constructed without. Generated documentation marks
# these, so an agent can tell `--pod` (required) from `--container` (optional).
DRIVER_REQUIRED_FLAGS: dict[str, set[str]] = {}

# Flag destinations owned by the runtime and the compute subcommands. A driver
# that claimed one of these would have the runtime's own value swept into its
# `from_params` payload by the CLI whitelist, so registration is rejected.
RESERVED_CLI_DESTS = frozenset({
    "answers",
    "backend",
    "background",
    "code",
    "compute_action",
    "dry_run",
    "fields",
    "input",
    "interpreter",
    "job_action",
    "job_id",
    "json",
    "name",
    "out_dir",
    "output",
    "package",
    "port",
    "profile",
    "recipe",
    "run_args",
    "run_id",
    "serve",
    "setup_action",
    "smoke_test",
    "submit_args",
    "timeout",
    "workspace",
})


def register_driver(name: str) -> Callable[[type], type]:
    """Decorator to register a compute driver backend class."""

    def decorator(cls: type) -> type:
        cls.backend_name = name
        DRIVER_REGISTRY[name] = cls
        return cls

    return decorator


def get_registered_drivers() -> dict[str, type["ComputeDriver"]]:
    """Return dictionary of all registered compute drivers."""
    return dict(DRIVER_REGISTRY)


def get_driver(name: str) -> type["ComputeDriver"] | None:
    """Get registered compute driver class by backend name."""
    return DRIVER_REGISTRY.get(name)


def list_drivers() -> list[str]:
    """List names of all registered compute driver backends."""
    return list(DRIVER_REGISTRY.keys())


def get_registered_driver_cli_flags() -> set[str]:
    """Return set of attribute names for all CLI flags registered by compute drivers.

    Used by the compute CLI dispatcher to extract a positive whitelist of parameters
    from argparse.Namespace into driver `param_overrides`, avoiding brittle negative
    blacklists of general runtime flags.
    """
    return set(REGISTERED_DRIVER_CLI_FLAGS)


def record_cli_flag(dest: str,
                    option_strings: tuple[str, ...] = (),
                    backend: str = "",
                    required: bool = False) -> None:
    """Record a driver-owned CLI destination, rejecting runtime-reserved names."""
    if dest in RESERVED_CLI_DESTS:
        raise ValueError(
            f"CLI flag '--{dest.replace('_', '-')}' is reserved by the runtime; "
            "choose a backend-specific name instead.")
    REGISTERED_DRIVER_CLI_FLAGS.add(dest)
    if backend:
        declared = DRIVER_CLI_FLAGS.setdefault(backend, [])
        required_flags = DRIVER_REQUIRED_FLAGS.setdefault(backend, set())
        for option in option_strings:
            if option.startswith("--") and option not in declared:
                declared.append(option)
            if required and option.startswith("--"):
                required_flags.add(option)


def record_discovery_error(module_name: str, exc: BaseException) -> None:
    """Record why a driver module failed to import during auto-discovery."""
    DISCOVERY_ERRORS[module_name] = f"{type(exc).__name__}: {exc}"


def get_discovery_errors() -> dict[str, str]:
    """Return driver modules that failed to import, keyed by module name."""
    return dict(DISCOVERY_ERRORS)


def describe_discovery_errors() -> str:
    """Render discovery failures as a suffix for user-facing error messages."""
    if not DISCOVERY_ERRORS:
        return ""
    details = "; ".join(f"{name} ({reason})"
                        for name, reason in sorted(DISCOVERY_ERRORS.items()))
    return f" Driver modules that failed to load: {details}."
