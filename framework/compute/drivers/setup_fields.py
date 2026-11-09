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
"""Per-backend setup field declarations and format checks.

Each compute driver declares what a complete profile for its backend needs in
one `SetupFields` value on the class (`SETUP_FIELDS`). Every other place that
deals in per-backend fields -- `questions.json` prompts, `from_params`, and
`compute setup plan/apply` -- is checked against `SETUP_FIELDS` so they cannot
drift apart. The `environment.json` schema stays structural (types only) and
does not repeat which fields each backend requires.

This module imports no driver, so `base` and the drivers can import from it.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

# Kueue priority classes a `gke_job` profile may name. Lives here rather than
# in `gke_job` so this module depends on no driver; `gke_job` imports it.
# A tuple, highest first, so `choices` and error text list them in a stable,
# meaningful order (a frozenset's iteration order varies with hash seeding).
VALID_PRIORITIES = ("medium", "low", "very-low")


@dataclasses.dataclass(frozen=True)
class SetupFields:
    """The fields a backend's profile asks for, and when each one applies.

    Each driver declares its setup contract once in `SETUP_FIELDS`.
    `questions.json` prompts, `from_params`, and `compute setup` all check
    against it so required and conditional fields cannot drift.

    Attributes:
        required: Parameter names that must be non-empty before `apply` saves
            a profile for this backend.
        optional: Parameter names asked during initial setup that have a
            working default when left blank (e.g. `namespace`, `queue`,
            `python_interpreter`).
        conditional: `{field: trigger_field}` for fields that become relevant
            only once `trigger_field` is non-empty (e.g. `priority` once
            `queue` is set).
        on_demand: `{field: reason}` for fields that only some workflows need
            and that are asked when a command first requires them rather than
            during initial setup.
    """

    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    conditional: dict[str, str] = dataclasses.field(default_factory=dict)
    on_demand: dict[str, str] = dataclasses.field(default_factory=dict)

    def all_fields(self) -> set[str]:
        """Every parameter name this backend recognises during setup."""
        return (set(self.required)
                | set(self.optional)
                | set(self.conditional)
                | set(self.on_demand))


# GCP zones have a trailing `-<letter>` after the region (`us-central1-a`,
# `europe-west4-b`); a region alone (`us-central1`, `us-east5`) is missing it.
ZONE_PATTERN = re.compile(r"^[a-z]+-[a-z]+\d+-[a-z]$")
REGION_PATTERN = re.compile(r"^[a-z]+-[a-z]+\d+$")
TOPOLOGY_PATTERN = re.compile(r"^[1-9]\d*x[1-9]\d*(?:x[1-9]\d*)?$")

BUCKET_FIELDS = frozenset({"staging_bucket", "trajectory_bucket"})


def validate_field_format(name: str, value: Any) -> dict[str, Any] | None:
    """Check one field's value at setup time before saving `environment.json`.

    Returns `None` when `value` is well-formed (or empty -- emptiness is
    checked separately against `SetupFields.required`), or an error entry
    `{"name": name, "value": value, "reason": ...}` suitable for the
    `invalid` list of an `INCOMPLETE` response.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None

    if name == "zone":
        if REGION_PATTERN.match(text):
            return {
                "name":
                name,
                "value":
                value,
                "reason": (f"{text!r} is a region, not a zone "
                           f"(expected e.g. '{text}-a')"),
                "choices":
                [f"{text}-a", f"{text}-b", f"{text}-c", f"{text}-f"],
            }
        if not ZONE_PATTERN.match(text):
            return {
                "name":
                name,
                "value":
                value,
                "reason":
                (f"{text!r} does not look like a GCP zone "
                 f"(expected '<region>-<zone>', e.g. 'us-central1-a')"),
            }

    if name == "tpu_topology":
        if not TOPOLOGY_PATTERN.match(text):
            return {
                "name":
                name,
                "value":
                value,
                "reason": (f"{text!r} must match NxM or NxMxK "
                           f"(e.g. '2x4' or '2x2x1')"),
            }

    # `tpu_accelerator` is deliberately not checked against a list of known
    # labels: preflight only warns on an unfamiliar one so a newer TPU
    # generation works before this list learns its label.

    if name in BUCKET_FIELDS:
        if not text.startswith("gs://") or len(text) <= len("gs://"):
            return {
                "name": name,
                "value": value,
                "reason": f"{text!r} must be a GCS URI starting with 'gs://'",
            }

    if name == "priority":
        if text not in VALID_PRIORITIES:
            return {
                "name":
                name,
                "value":
                value,
                "reason": (f"{text!r} is not a valid Kueue priority; "
                           f"expected one of {list(VALID_PRIORITIES)}"),
                "choices": ["", *VALID_PRIORITIES],
            }

    return None
