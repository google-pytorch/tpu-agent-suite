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
"""Unified compute framework."""

from . import cli, lease, setup
from .cli import (JOB_ACTIONS, SETUP_GUIDE, handle_compute_cli,
                  handle_config_cli)
from .cli_parser import register_compute_parser, register_config_parser
from .config import (describe_backends, describe_required_flags,
                     get_valid_backends, normalize_environment_profiles,
                     resolve_driver, resolve_profile_name,
                     unknown_profile_error)
from .drivers import (DISCOVERY_ERRORS, DRIVER_REGISTRY,
                      AsynchronousExecutionResult, ComputeConfigError,
                      ComputeDriver, GceTpuVmDriver, GkeJobDriver,
                      GkePodDriver, HardwareProfile, LocalDriver, SetupFields,
                      StagedFile, SynchronousExecutionResult,
                      describe_discovery_errors, get_discovery_errors,
                      get_driver, get_registered_driver_cli_flags,
                      get_registered_drivers, list_drivers, register_driver,
                      validate_field_format)
from .drivers.jobs import reap_jobs
from .setup import (apply_setup, config_remove, config_set, config_show,
                    config_use)
