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
"""Canonical domain enums, structured CLI errors, and constants for `hc.py`."""

from __future__ import annotations

import enum
from typing import TypeVar

from framework import errors as base_errors

MAX_TEXT_LENGTH: int = 65_536
MAX_SHORT_LENGTH: int = 500
LOG_ALL_CALLS_ENV: str = "TPU_AGENT_SUITE_HC_LOG_ALL_CALLS"
FLAGS_FILENAME: str = "__flags.json"
WORKTREE_STATE_FILENAME: str = "worktree_state.json"
TRUNCATION_SUFFIX: str = "[...]"

_E = TypeVar("_E", bound=enum.StrEnum)


class HcError(base_errors.TpuAgentSuiteError):
    """Domain or CLI error carrying a stable `error_code` for `cmd.rejected` telemetry."""

    def __init__(self,
                 message: str,
                 *,
                 error_code: str = "HC_ERROR",
                 exit_code: int = 1) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.exit_code = exit_code


class InvalidStateError(HcError):
    """Raised when an operation violates a campaign, round, hypothesis, or candidate state rule."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="INVALID_STATE")


class ValidationError(HcError):
    """Raised when an input schema, column filter, or field constraint is violated."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="VALIDATION_ERROR")


class NotFoundError(HcError):
    """Raised when a referenced primary or foreign key does not exist."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="NOT_FOUND")


class GitOperationError(HcError):
    """Raised when a detached working tree or bare git operation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="GIT_ERROR")


class CampaignState(enum.StrEnum):
    """Lifecycle states of an optimization campaign."""

    UNANCHORED = "Unanchored"
    ACTIVE = "Active"
    BLOCKED = "Blocked"
    CLOSED = "Closed"


class CloseReason(enum.StrEnum):
    """Terminal reasons recorded when closing a campaign."""

    TARGET_MET = "TargetMet"
    MAX_ROUNDS = "MaxRounds"
    PLATEAU = "Plateau"
    MANUAL = "Manual"
    ABORTED = "Aborted"


class WorkloadKind(enum.StrEnum):
    """Supported workload kinds for optimization protocols."""

    INFERENCE = "inference"
    TRAINING = "training"


class MetricDirection(enum.StrEnum):
    """Optimization direction for a benchmark metric."""

    MAXIMUM = "maximum"
    MINIMUM = "minimum"


class MetricStatistic(enum.StrEnum):
    """Summary statistic extracted across metric samples."""

    MEAN = "mean"
    MEDIAN = "median"


class Comparator(enum.StrEnum):
    """Comparison operator for guardrail and accuracy thresholds."""

    GTE = "gte"
    LTE = "lte"


class RoundState(enum.StrEnum):
    """Lifecycle states of a hillclimbing round."""

    OPEN = "Open"
    PROGRESSED = "Progressed"
    BARREN = "Barren"


class LeverKind(enum.StrEnum):
    """Optimization lever categories tracked by hypotheses."""

    KERNEL = "kernel"
    SHARDING = "sharding"
    COLLECTIVE = "collective"
    QUANTIZATION = "quantization"
    OFFLOAD = "offload"
    RUNTIME_CONFIG = "runtime_config"
    COMPILER_FLAG = "compiler_flag"


class TargetScope(enum.StrEnum):
    """Execution scope targeted by an optimization hypothesis."""

    CANONICAL_OPS = "canonical_ops"
    REDUNDANT_RELAYOUT = "redundant_relayout"
    STEP = "step"


class HypothesisState(enum.StrEnum):
    """Lifecycle states of an optimization hypothesis."""

    OPEN = "Open"
    PROMOTED = "Promoted"
    REFUTED = "Refuted"


class CandidateState(enum.StrEnum):
    """Derived lifecycle states of a candidate implementation."""

    PROMOTED = "Promoted"
    SUPERSEDED = "Superseded"
    ABANDONED = "Abandoned"
    E2E_IN_FLIGHT = "E2E_In_Flight"
    E2E_PASSED = "E2E_Passed"
    E2E_ACCURACY_FAILED = "E2E_Accuracy_Failed"
    E2E_PENDING_ACCURACY = "E2E_Pending_Accuracy"
    E2E_VIABLE = "E2E_Viable"
    E2E_REGRESSED = "E2E_Regressed"
    MICROBENCHMARK_IN_FLIGHT = "Microbenchmark_In_Flight"
    MICROBENCH_PASSED = "Microbench_Passed"
    MICROBENCH_REGRESSED = "Microbench_Regressed"
    UNTESTED = "Untested"


class JobKind(enum.StrEnum):
    """Categories of benchmark and evaluation jobs managed by `hc.py`."""

    MICROBENCHMARK = "Microbenchmark"
    E2E_BENCHMARK = "E2E_Benchmark"
    ACCURACY = "Accuracy"


class JobStatus(enum.StrEnum):
    """Execution states of a tracked benchmark or accuracy job."""

    DISPATCHED = "Dispatched"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    UNKNOWN = "Unknown"


class ReviewApproval(enum.StrEnum):
    """Reviewer approval outcomes for a candidate diff."""

    PASS = "Pass"
    CONDITIONAL_PASS = "ConditionalPass"
    FAIL = "Fail"


class DecisionVerdict(enum.StrEnum):
    """Verdict categories recorded when judging a candidate."""

    WIN = "Win"
    VIABLE = "Viable"
    NEUTRAL = "Neutral"
    LOSS = "Loss"
    INVALID = "Invalid"


class HeadroomStatus(enum.StrEnum):
    """Profile alignment status for hypothesis headroom matching."""

    EXACT = "exact"
    SUBSUMED = "subsumed"
    UNPROFILED = "unprofiled"


class DiffBase(enum.StrEnum):
    """Reference baseline selector for candidate diff computation."""

    INITIAL = "initial"
    CHAMPION = "champion"


def parse_enum(enum_cls: type[_E], raw: str, field_name: str) -> _E:
    """Parse `raw` into `enum_cls`, raising `ValidationError` on mismatch."""
    if isinstance(raw, str):
        try:
            return enum_cls(raw)
        except ValueError:
            pass
    allowed = [member.value for member in enum_cls]
    raise ValidationError(
        f"invalid {field_name} {raw!r}: expected one of {allowed}")
