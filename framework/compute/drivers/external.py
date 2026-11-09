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
"""Loads compute drivers that live outside the installed package.

An external driver is a Python file, or a directory with an `__init__.py`,
that defines a `ComputeDriver` subclass and registers it with
`registry.register_driver`. It lets a user add a backend for a private
scheduler without changing the package, and the driver keeps working after
the package is reinstalled or upgraded.

The runtime looks for external drivers in this order:

1. The `drivers/` directory of the central store, normally
   `~/.tpu-agent-suite/drivers/`. A missing directory is not an error.
2. `<git_root>/google/drivers/` when running directly from a repository
   checkout that contains internal drivers.
3. Each path listed in the `TPU_AGENT_SUITE_COMPUTE_DRIVERS` environment
   variable, separated by `os.pathsep` (`:` on Linux). A path can name one
   driver file, one driver package directory, or a directory of drivers.
   A listed path that does not exist is reported as a load failure.

In a directory of drivers, every `*.py` file and every subdirectory with an
`__init__.py` is loaded, except names that start with `_` or `.`. Helper
modules therefore go in a driver package or get a leading underscore.

Loading a driver runs its code with the permissions of the runtime, so these
locations must only hold code the user trusts.

Loading never raises. Each failure is recorded in
`registry.DISCOVERY_ERRORS` under the source path, and the `INVALID_BACKEND`
error lists it, in the same way as for a built-in driver that fails to
import. These cases are failures:

- The source raises an exception while it is imported.
- The source registers no backend.
- The source registers a backend name that a driver loaded before it
  already uses, for example `gke_job`. The earlier driver stays registered.
- Two sources have the same module name, for example `a/foo.py` and
  `b/foo.py`. The first one stays loaded.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

from framework import paths

from . import registry

Path = pathlib.Path

ENV_VAR = "TPU_AGENT_SUITE_COMPUTE_DRIVERS"
DRIVERS_DIRNAME = "drivers"

# Prefix of the `sys.modules` name each external source is imported under,
# e.g. `my_driver.py` becomes `_tpu_agent_suite_external_driver_my_driver`.
# The prefix keeps an external module from shadowing or being shadowed by an
# installed module of the same name.
MODULE_PREFIX = "_tpu_agent_suite_external_driver_"


def default_drivers_dir() -> Path:
    """Return the central store's directory for external drivers."""
    return paths.get_tpu_agent_suite_home() / DRIVERS_DIRNAME


def _is_source(path: Path) -> bool:
    """Return True if `path` is a driver file or a driver package directory."""
    if path.name.startswith(("_", ".")):
        return False
    if path.is_file():
        return path.suffix == ".py"
    return path.is_dir() and (path / "__init__.py").is_file()


def repo_google_drivers_dir() -> Path | None:
    """Return `<git_root>/google/drivers` when running from an internal checkout."""
    git_root = paths.find_git_root(Path(__file__).resolve())
    if git_root is None:
        return None
    return git_root / "google" / DRIVERS_DIRNAME


def find_sources() -> tuple[list[Path], list[str]]:
    """Find the external driver sources to load.

    Returns:
        A tuple `(sources, missing)`. `sources` lists each driver file or
        driver package directory once, in load order. `missing` lists the
        entries of the environment variable that name no existing path.
    """
    entries: list[tuple[Path, bool]] = [(default_drivers_dir(), False)]
    repo_drivers = repo_google_drivers_dir()
    if repo_drivers is not None:
        entries.append((repo_drivers, False))
    for raw in os.environ.get(ENV_VAR, "").split(os.pathsep):
        if raw.strip():
            entries.append((Path(raw.strip()).expanduser(), True))

    sources: list[Path] = []
    missing: list[str] = []
    for entry, listed in entries:
        if not entry.exists():
            if listed:
                missing.append(str(entry))
            continue
        if entry.is_file() or (entry / "__init__.py").is_file():
            # A path the user listed by name is loaded even if its name
            # starts with `_`, because the user chose it explicitly.
            candidates = [entry]
        else:
            candidates = [
                child for child in sorted(entry.iterdir()) if _is_source(child)
            ]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in sources:
                sources.append(resolved)
    return sources, missing


def _module_name(source: Path) -> str:
    stem = source.stem if source.is_file() else source.name
    return MODULE_PREFIX + stem


def _load_one(source: Path) -> None:
    """Import one driver source and record what it registered.

    Args:
        source: An absolute path to a driver file or a driver package
            directory.
    """
    label = str(source)
    module_name = _module_name(source)
    existing = sys.modules.get(module_name)
    if existing is not None:
        expected_file = (source if source.is_file() else source /
                         "__init__.py").resolve()
        existing_file = getattr(existing, "__file__", None)
        if (existing_file and Path(existing_file).resolve() == expected_file
                and label in registry.EXTERNAL_DRIVERS.values()):
            return
        registry.record_discovery_error(
            label,
            ImportError(f"another external driver is already loaded as "
                        f"module '{module_name}'; rename one of them"))
        return

    if source.is_file():
        spec = importlib.util.spec_from_file_location(module_name, source)
    else:
        spec = importlib.util.spec_from_file_location(
            module_name,
            source / "__init__.py",
            submodule_search_locations=[str(source)])
    if spec is None or spec.loader is None:
        registry.record_discovery_error(
            label, ImportError("not an importable Python source"))
        return

    before = _snapshot()
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so that a driver package can import its
    # own submodules with relative imports.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        sys.modules.pop(module_name, None)
        _restore(before)
        registry.record_discovery_error(label, exc)
        return

    before_drivers = before[0]
    replaced = [
        name for name, cls in before_drivers.items()
        if registry.DRIVER_REGISTRY.get(name) is not cls
    ]
    if replaced:
        _restore(before)
        sys.modules.pop(module_name, None)
        registry.record_discovery_error(
            label,
            ValueError(f"registers backend name(s) {sorted(replaced)}, which "
                       "a loaded driver already uses; choose another name"))
        return

    added = [
        name for name in registry.DRIVER_REGISTRY if name not in before_drivers
    ]
    if not added:
        _restore(before)
        sys.modules.pop(module_name, None)
        registry.record_discovery_error(
            label,
            ValueError("registers no backend; decorate the driver class "
                       "with registry.register_driver('<name>')"))
        return
    for name in added:
        registry.EXTERNAL_DRIVERS[name] = label


_RegistrySnapshot = tuple[
    dict[str, type],
    dict[str, list[str]],
    dict[str, set[str]],
    set[str],
]


def _snapshot() -> _RegistrySnapshot:
    """Capture driver and CLI-flag registry state before importing a source."""
    return (
        dict(registry.DRIVER_REGISTRY),
        {
            k: list(v)
            for k, v in registry.DRIVER_CLI_FLAGS.items()
        },
        {
            k: set(v)
            for k, v in registry.DRIVER_REQUIRED_FLAGS.items()
        },
        set(registry.REGISTERED_DRIVER_CLI_FLAGS),
    )


def _restore(snapshot: _RegistrySnapshot) -> None:
    """Reset the driver and CLI-flag registries to `snapshot`."""
    drivers, cli_flags, req_flags, all_flags = snapshot
    registry.DRIVER_REGISTRY.clear()
    registry.DRIVER_REGISTRY.update(drivers)
    registry.DRIVER_CLI_FLAGS.clear()
    registry.DRIVER_CLI_FLAGS.update(cli_flags)
    registry.DRIVER_REQUIRED_FLAGS.clear()
    registry.DRIVER_REQUIRED_FLAGS.update(req_flags)
    registry.REGISTERED_DRIVER_CLI_FLAGS.clear()
    registry.REGISTERED_DRIVER_CLI_FLAGS.update(all_flags)


def load_external_drivers() -> None:
    """Load every external driver source. See the module docstring."""
    sources, missing = find_sources()
    for entry in missing:
        registry.record_discovery_error(
            entry,
            FileNotFoundError(f"{ENV_VAR} lists a path that does "
                              "not exist"))
    for source in sources:
        _load_one(source)
