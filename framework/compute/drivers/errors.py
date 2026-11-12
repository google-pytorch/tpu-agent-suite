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
"""Error types raised by the compute framework."""

from __future__ import annotations

from typing import Any


class ComputeConfigError(Exception):
    """Raised when compute configuration is missing or invalid."""

    def __init__(self,
                 message: str,
                 error_code: str = "COMPUTE_CONFIG_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code

    def to_dict(self) -> dict[str, Any]:
        """The error as a JSON-ready dict with status, error_code and message."""
        return {
            "status": "error",
            "error_code": self.error_code,
            "message": self.message,
        }


class ClusterAccessDenied(ComputeConfigError):
    """The cluster refused every identity tried; names them and the fix."""

    def __init__(self, identities: list[str], fix: str) -> None:
        super().__init__(
            f"The cluster refused {' and then '.join(identities)}. "
            f"Fix: {fix}",
            error_code="CLUSTER_ACCESS_DENIED")
        self.identities = identities
        self.fix = fix

    def to_dict(self) -> dict[str, Any]:
        """The base error dict plus the refused identities and the fix."""
        return {
            **super().to_dict(), "identities": self.identities,
            "fix": self.fix
        }


class NotConfigured(ComputeConfigError):
    """A command needs a setting that is not set up yet.

    COMPUTE_NOT_CONFIGURED when there is no usable profile at all, or
    `<GROUP>_NOT_CONFIGURED` (see `ComputeDriver.not_configured`) for a field
    that is asked only when a command first needs it. `missing` names the
    fields to ask for and `fix` the `compute setup plan` command that asks
    for them. A driver does not know which package the command runs for, so
    it leaves `fix` for the CLI to fill in with `set_fix`.
    """

    def __init__(self,
                 message: str,
                 error_code: str,
                 missing: list[str],
                 fix: str | None = None) -> None:
        super().__init__(f"{message} Fix: {fix}" if fix else message,
                         error_code=error_code)
        self.missing, self.fix = missing, fix

    def set_fix(self, fix: str) -> None:
        """Record the fix command and append it to the message."""
        self.fix = fix
        self.message = f"{self.message} Fix: {fix}"

    def to_dict(self) -> dict[str, Any]:
        """The base error dict plus the missing fields and the fix, if set."""
        return {
            **super().to_dict(), "missing": self.missing,
            **({
                "fix": self.fix
            } if self.fix else {})
        }
