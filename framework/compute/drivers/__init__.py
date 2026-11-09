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
"""Compute driver implementations."""

import importlib
import pathlib
import pkgutil

from . import external
from .base import ComputeDriver, SetupFields, StagedFile
from .contracts import (AsynchronousExecutionResult, HardwareProfile,
                        PreflightCheck, PreflightReport,
                        SynchronousExecutionResult)
from .errors import ClusterAccessDenied, ComputeConfigError, NotConfigured
from .gce_tpu_vm import GceTpuVmDriver
from .gke_job import GkeJobDriver
from .gke_pod import GkePodDriver
from .jobs import JobRecord, jobs_dir
from .local import LocalDriver
from .registry import (DISCOVERY_ERRORS, DRIVER_REGISTRY,
                       REGISTERED_DRIVER_CLI_FLAGS, RESERVED_CLI_DESTS,
                       describe_discovery_errors, get_discovery_errors,
                       get_driver, get_registered_driver_cli_flags,
                       get_registered_drivers, list_drivers, record_cli_flag,
                       record_discovery_error, register_driver)
from .setup_fields import validate_field_format
from .shell import RemoteShellDriver, require_file

_INTERNAL_MODULES = ("base", "contracts", "errors", "external", "gcs_staging",
                     "jobs", "kubectl", "probe", "registry", "setup_fields",
                     "shell")


def _auto_discover_drivers() -> None:
    """Dynamically discover and register any additional compute drivers in this directory.

    Import failures are recorded rather than raised: one broken driver must not
    take down every other backend. `describe_discovery_errors()` surfaces them in
    the `INVALID_BACKEND` message so a driver that failed to load is reported as
    such instead of looking like a configuration mistake.
    """
    package_dir = pathlib.Path(__file__).resolve().parent
    # Subpackages are included: a driver with helper modules can live in
    # `drivers/<backend>/` rather than being forced into one file.
    for _, module_name, _is_pkg in pkgutil.iter_modules([str(package_dir)]):
        if module_name not in _INTERNAL_MODULES:
            try:
                importlib.import_module(f"{__name__}.{module_name}")
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                record_discovery_error(module_name, exc)


_auto_discover_drivers()
# After the built-in drivers, so that an external driver cannot take a
# built-in backend name.
external.load_external_drivers()
