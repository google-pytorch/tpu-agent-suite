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
"""Study store: the layer above a run.

Studies outlive the session that opens them, so their state, their lifecycle
events and their command-line surface all live here rather than in the
runtime that happens to spend a run on them.
"""

from __future__ import annotations

import contextlib
import datetime
import fcntl
import json
import os
import pathlib
import sys
import threading
from typing import Any, Iterator, Mapping
from zoneinfo import ZoneInfo

from framework import jsonio, paths, utils

Path = pathlib.Path

TpuAgentSuiteError = utils.TpuAgentSuiteError
TPU_AGENT_SUITE_VERSION = utils.TPU_AGENT_SUITE_VERSION
runtime_schema_path = utils.runtime_schema_path
utc_now_iso = utils.utc_now_iso
validate_instance = utils.validate_instance

# A run is one session. A study is the long-running optimization effort that
# spends many of them, and it is the thing that has to survive a session
# ending, a quota refusal and a machine reboot. So it lives beside runs/ in
# the central store rather than inside any one run, and it references the
# runs it spent by id rather than copying their artifacts.
#
# Two streams, deliberately split by who writes them:
#
#   study.json   the state. The one hot file, read whole at every decision
#                point and rewritten whole, never patched.
#   events.jsonl machine-authored by lifecycle hooks; journal.jsonl is the
#                agent-authored overflow beside it. Both cold: a round reads
#                neither, and neither is a recovery source.
#
# The human-readable view is rendered from the state by `study notes`, not
# stored. The file that used to hold it was written once at creation and
# never updated, so on the morning it mattered it still said "nothing
# measured yet".
#
# `protocol` is pre-registered: the primary metric, the admissibility gates,
# the measurement cell, the replicate count and the stopping rule are fixed
# before the baseline runs. Nothing here enforces that by itself -- what it
# does is make an amendment visible, because `protocol_revision` changes and
# the old protocol stays in `amendments`.

STUDY_DOCUMENT = "study.json"
STUDY_LOCK_FILE = ".study.lock"
STUDY_STATUS_OPEN = "open"
STUDY_STATUS_CLOSED = "closed"

STUDY_PROTOCOL_SCHEMA = "study-protocol.json"
STUDY_ROUND_SCHEMA = "study-round.json"

# Canonical bottleneck taxonomy shared across study-round.json and arm-request.json.
# Asserted to agree with JSON schemas by the test suite so changing the taxonomy
# has a single source of truth.
BOTTLENECK_KINDS: tuple[str, ...] = (
    "kernel-block",
    "exposed-collective",
    "sharding-overhead",
    "load-imbalance",
    "host-critical-path",
    "recompilation-padding",
    "scheduler-batching",
    "memory-capacity",
)

MAX_STUDY_ARMS = 16
MAX_STUDY_AMENDMENTS = 10

_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_LOCK_STATE = threading.local()


@contextlib.contextmanager
def study_lock(study_id: str) -> Iterator[None]:
    """Acquire a re-entrant thread and process advisory lock for `study_id`.

    Protects read-modify-write cycles on `study.json` across concurrent
    subagents, threads, and lifecycle hook processes. Lock files live as hidden
    regular files under `get_studies_root()` so locking a study never creates
    its subdirectory before `start_study` runs.
    """
    if study_id == "__store__":
        lock_path = paths.get_studies_root() / STUDY_LOCK_FILE
    else:
        utils.validate_study_id(study_id)
        lock_path = paths.get_studies_root() / f".{study_id}.lock"

    lock_key = str(lock_path)
    with _THREAD_LOCKS_GUARD:
        rlock = _THREAD_LOCKS.setdefault(lock_key, threading.RLock())

    held: dict[str, tuple[int, int]] = getattr(_LOCK_STATE, "held", None) or {}
    _LOCK_STATE.held = held

    with rlock:
        if lock_key in held:
            fd, depth = held[lock_key]
            held[lock_key] = (fd, depth + 1)
            try:
                yield
            finally:
                held[lock_key] = (fd, depth)
            return

        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            held[lock_key] = (fd, 1)
            try:
                yield
            finally:
                held.pop(lock_key, None)
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass


def load_study_protocol_schema() -> dict[str, Any]:
    """Load the protocol schema that ships beside the runtime.

    It lives with the framework rather than the package because framework
    code is what validates it, and because a study must be openable without
    depending on which packages happen to be installed at that moment.
    """
    return jsonio.load_json(runtime_schema_path(STUDY_PROTOCOL_SCHEMA))


def validate_study_protocol(protocol: dict[str, Any]) -> list[str]:
    """Return every reason `protocol` is unfit to open a study with.

    The schema carries the structure. This adds only what the validator
    cannot express, for the same reason the artifact rule lives in
    `check_completed_has_artifact`: it has no conditional keywords.

    There is deliberately no separate spend cap. `stopping_rule.max_rounds`
    is required of every protocol and a round is one dispatch, so an `auto`
    study is already bounded by a number a human wrote down. A cap phrased in
    "expensive dispatches" would read better and enforce nothing, because
    nothing in the package defines which dispatches are expensive -- and an
    unenforceable brake on the mode that spends without asking is worse than
    an admittedly coarse one.
    """
    errors = validate_instance(protocol, load_study_protocol_schema())
    if errors:
        return errors

    # A stopping rule is required, but a rule that permits unlimited rounds
    # would satisfy the schema while leaving an unattended study unbounded.
    max_rounds = protocol.get("stopping_rule", {}).get("max_rounds")
    if protocol.get("mode") == "auto" and not max_rounds:
        errors.append(
            "protocol: mode 'auto' requires stopping_rule.max_rounds."
            " Auto spends without asking, so the round count is its only brake"
        )

    return errors


def default_owner() -> str:
    """Who to ask about a study, taken from the environment.

    Falls back to "unknown" rather than raising: an unattributed study is a
    nuisance, but refusing to open one because the environment is unusual
    would be worse.
    """
    for var in ("USER", "LOGNAME", "USERNAME"):
        value = os.environ.get(var, "").strip()
        if value:
            return value
    return "unknown"


def start_study(study_id: str,
                protocol: dict[str, Any],
                package: str,
                target: str,
                owner: str = "") -> dict[str, Any]:
    """Create a study namespace in the central store, or fail if it exists.

    `protocol` carries the experiment's terms and nothing else. `target` and
    `owner` are identity, and they live on the document rather than inside
    the protocol on purpose: the protocol is amendment-governed, so every
    change to it bumps `protocol_revision`, pushes the prior version onto
    `amendments` and owes a written reason. Correcting a misspelt owner must
    not produce a formal amendment to what the experiment measures.

    `target` is what is being optimized, and it is required because it is the
    grouping key -- "every study on this model" is the query people actually
    run, and it cannot be recovered from a free-form study id.

    `owner` defaults to the invoking user. An unattended study found running
    at 3am needs someone to ask, and making that a required flag on the one
    command a human types buys nothing when the environment already knows.

    `package` is checked for shape only, not for being installed. Coupling
    study creation to install state would break opening one in CI, or before
    an install finishes, and buys no safety in return.
    """
    utils.validate_package_name(package)

    if not target.strip():
        raise TpuAgentSuiteError(
            "a study needs a --target: the thing being optimized. It is the"
            " grouping key, and a free-form study id cannot stand in for it")

    protocol_errors = validate_study_protocol(protocol)
    if protocol_errors:
        raise TpuAgentSuiteError("invalid study protocol:\n  " +
                                 "\n  ".join(protocol_errors))

    study_root = paths.get_study_root(study_id)
    with study_lock(study_id):
        if study_root.exists():
            raise TpuAgentSuiteError(
                f"study {study_id!r} already exists at {study_root} --"
                " use `study status` to read it, or pick another id")

        study_root.mkdir(parents=True)
        now = utc_now_iso()
        document = {
            "study_id": study_id,
            "status": STUDY_STATUS_OPEN,
            "package": package,
            "target": target.strip(),
            "owner": owner.strip() or default_owner(),
            "version": TPU_AGENT_SUITE_VERSION,
            "created_at": now,
            "updated_at": now,
            "protocol_revision": 1,
            "protocol": protocol,
            "amendments": [],
            "runs": [],
        }
        jsonio.write_json(study_root / STUDY_DOCUMENT, document)
        return document


def study_round(document: dict[str, Any]) -> int:
    """Rounds attempted so far, derived rather than stored.

    A round is one run, enforced by `attach_run_to_study`, so the count of
    attached runs *is* the round number and a stored counter could only ever
    disagree with it.

    This counts **attempted** rounds. A round that attaches its run and then
    dies still counts, which is deliberate: it spent real hardware, and a
    counter that advanced only on success would let a study that fails every
    round run forever against a round-based stopping rule.
    """
    return len(document.get("runs", []))


# ---------------------------------------------------------------------
# The two streams beside the study state
# ---------------------------------------------------------------------
#
# `events.jsonl` is machine-authored. Hooks append to it, so it records what
# actually happened rather than what an agent remembered to report -- an agent
# that forgets to write progress produces exactly the silence a monitor exists
# to detect, so agent discipline is the wrong mechanism here.
#
# `journal.jsonl` is agent-authored. It is where capped state sections
# overflow to, in the same round that overflows them, so a dropped row is
# archived rather than lost.
#
# Both are COLD. A round reads the study state and nothing else; per-round cost has
# to stay flat whether it is round 2 or round 60, and a stream that grows
# without bound cannot be in that path. External readers -- a monitoring
# daemon, a dashboard -- have no context window to spend and may read freely.
#
# Neither stream is a recovery source. The study state is the only thing a round
# resumes from. Rebuilding state from events would create a second source of
# truth, and the one that got believed would be whichever was read last.

STUDY_EVENTS = "events.jsonl"
STUDY_JOURNAL = "journal.jsonl"


def _append_jsonl(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Append one JSON record as a single line under an advisory file lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    return record


def append_study_event(study_id: str, kind: str,
                       **fields: Any) -> dict[str, Any]:
    """Append a machine-authored lifecycle event to a study's event stream."""
    utils.validate_study_id(study_id)
    record = {"at": utc_now_iso(), "kind": kind, **fields}
    return _append_jsonl(paths.get_study_root(study_id) / STUDY_EVENTS, record)


def read_study_events(study_id: str, limit: int = 0) -> list[dict[str, Any]]:
    """Read a study's lifecycle events, oldest first; `limit` keeps the newest."""
    utils.validate_study_id(study_id)
    return jsonio.read_jsonl(
        paths.get_study_root(study_id) / STUDY_EVENTS, limit)


def append_study_journal(study_id: str, entry: dict[str,
                                                    Any]) -> dict[str, Any]:
    """Archive an agent-authored entry, typically a row overflowing the study state."""
    utils.validate_study_id(study_id)
    record = {"at": utc_now_iso(), **entry}
    return _append_jsonl(
        paths.get_study_root(study_id) / STUDY_JOURNAL, record)


def read_study_journal(study_id: str, limit: int = 0) -> list[dict[str, Any]]:
    """Read a study's overflow journal, oldest first; `limit` keeps the newest."""
    utils.validate_study_id(study_id)
    return jsonio.read_jsonl(
        paths.get_study_root(study_id) / STUDY_JOURNAL, limit)


def studies_for_event(run_id: str = "") -> list[str]:
    """The open study this event belongs to, as a list of zero or one.

    A strict lookup, with no inference and no fan-out. A run is attached to
    at most one study, so ownership is a fact to read rather than a guess to
    make.

    An event whose run is attached to nothing is **dropped**. That covers
    every session that is not advancing a study -- an onboarding job, a
    hardware check, someone's scratch session -- and dropping those is the
    point. Broadcasting them was how an unrelated session's `fully_idle`
    landed in a study's stream and made a stalled study look alive, which
    inverts the one thing the stream exists to report.

    Returning a list rather than an optional keeps the caller's loop
    unchanged, and leaves room for a future event that legitimately belongs
    to several studies. Nothing produces one today.
    """
    if not run_id:
        return []
    study_id = study_for_run(run_id)
    if study_id is None:
        return []
    document = load_study(study_id)
    if document.get("status") != STUDY_STATUS_OPEN:
        return []
    return [study_id]


def load_study_round_schema() -> dict[str, Any]:
    """Load the round-record schema that ships beside the runtime."""
    return jsonio.load_json(runtime_schema_path(STUDY_ROUND_SCHEMA))


def check_round_against_protocol(record: dict[str, Any],
                                 protocol: dict[str, Any]) -> list[str]:
    """Check a round record against the terms the study pre-registered.

    The schema says a record is well-formed. This says it honours the bargain
    the study opened with, which is the only thing that makes pre-registration
    worth anything: a protocol whose terms nothing checks is a wish.

    Three rules, each earned:

      * A gate the protocol declares must appear in every decided arm's
        result. An omitted gate is a check that did not happen, and reading
        it as a pass is how a champion gets crowned on one metric and
        withdrawn later.
      * A gate marked `same_execution` must carry the metric's own execution
        id. Throughput from one run and accuracy from another do not describe
        the same configuration, however identical the config.
      * A `win` must clear the noise floor at the declared replicate count,
        and must pass every gate. A faster number that violates a gate is not
        a smaller win, it is inadmissible.
    """
    errors: list[str] = []
    gates = protocol.get("admissibility_gates") or []
    required_gates = {g["name"] for g in gates}
    same_exec = {g["name"] for g in gates if g.get("same_execution")}
    floor = protocol.get("noise_floor_pct")
    want_replicates = protocol.get("replicate_count")

    for arm in record.get("arms", []):
        if arm.get("status") != "decided":
            continue
        name = arm.get("arm", "?")
        result = arm.get("result")
        if not result:
            errors.append(f"arm {name}: decided without a result")
            continue

        verdict = result.get("verdict")
        metric = result.get("metric") or {}
        exec_id = result.get("execution_id", "")
        reported = {g["name"]: g for g in (result.get("gates") or [])}

        missing = sorted(required_gates - set(reported))
        if missing:
            errors.append(
                f"arm {name}: protocol declares gate(s) {', '.join(missing)}"
                " and the result does not report them -- an omitted gate is"
                " an unmeasured gate, not a passing one")

        for gate_name in sorted(same_exec & set(reported)):
            gate_exec = reported[gate_name].get("execution_id", "")
            if gate_exec != exec_id:
                errors.append(
                    f"arm {name}: gate {gate_name!r} is marked same_execution"
                    f" but was measured on {gate_exec!r} while the metric came"
                    f" from {exec_id!r}. Two executions do not describe one"
                    " configuration")

        # A verdict that asserts something about performance owes a
        # measurement. `invalid` does not: it means the number cannot be
        # trusted, which is a different claim from a number that did not help.
        want_metric = (protocol.get("primary_metric") or {}).get("name")
        if want_metric and metric.get(
                "name") and metric["name"] != want_metric:
            errors.append(
                f"arm {name}: reports {metric['name']!r} but the protocol's"
                f" primary metric is {want_metric!r}. A verdict has to be"
                " about the quantity the study is optimizing, or the delta"
                " compares two different things")

        if verdict in ("win", "tie", "loss") and not metric:
            errors.append(
                f"arm {name}: verdict {verdict!r} without a metric. A verdict"
                " about performance is a claim about a measurement, and an"
                " absent measurement cannot support one")

        if verdict != "win":
            continue

        # Checked explicitly rather than left to the absence of a value.
        # Every bar below is skipped when its input is missing, so a win that
        # simply omits the metric would clear all of them by saying nothing.
        for field in ("replicates", "delta_pct"):
            if metric.get(field) is None:
                errors.append(
                    f"arm {name}: a win must report metric.{field}."
                    " Omitting it does not clear the bar, it evades it")

        failed = sorted(n for n, g in reported.items() if not g.get("passed"))
        if failed:
            errors.append(
                f"arm {name}: recorded a win while failing gate(s)"
                f" {', '.join(failed)}. A faster number that violates a gate"
                " is inadmissible, not a smaller win")

        for gate_def in gates:
            g_name = gate_def.get("name")
            if not g_name or g_name not in reported:
                continue
            if "bound" in gate_def and reported[g_name].get("value") is None:
                errors.append(
                    f"arm {name}: gate {g_name!r} declares a bound but the"
                    " result omits 'value'. A win cannot assert a pass without"
                    " the measurement")

        got = metric.get("replicates")
        if want_replicates and got and got < want_replicates:
            errors.append(
                f"arm {name}: recorded a win from {got} replicate(s) when the"
                f" protocol pre-registered {want_replicates}")

        delta = metric.get("delta_pct")
        if floor is not None and delta is not None and abs(delta) <= floor:
            errors.append(
                f"arm {name}: recorded a win on a {delta}% delta inside the"
                f" pre-registered {floor}% noise floor. That is a tie")

    return errors


def compute_arm_verdict(
    metric_value: float | None,
    baseline_value: float,
    direction: str,
    noise_floor_pct: float = 0.0,
    gates_passed: bool = True,
    replicates: int = 1,
    required_replicates: int = 1,
    telemetry_valid: bool = True,
) -> tuple[str, float | None]:
    """Compute the empirical verdict and delta percentage for an arm candidate.

    Follows the canonical study invariants:
      1. Admissibility: Any failed admissibility gate invalidates a performance claim ('invalid').
         A faster configuration that breaks an accuracy or loss gate is inadmissible and cannot count.
      2. Replicates: Measuring fewer replicates than pre-registered leaves the verdict at 'tie'.
      3. Noise floor: A win must strictly exceed the noise floor in the optimizing direction with all gates passing.
      4. Symmetry: Losses fall below -noise_floor_pct; measurements within the floor (abs(delta) <= floor) are ties.
      5. Integrity: Crashes, corrupted telemetry, or invalid baselines produce 'invalid' (untrustworthy number).

    Returns:
        (verdict, delta_pct) where verdict is 'win', 'tie', 'loss', or 'invalid'.
    """
    if not telemetry_valid or metric_value is None or baseline_value <= 0:
        return "invalid", None

    if direction == "maximize":
        delta_pct = round(
            ((metric_value - baseline_value) / baseline_value) * 100.0, 4)
    elif direction == "minimize":
        delta_pct = round(
            ((baseline_value - metric_value) / baseline_value) * 100.0, 4)
    else:
        return "invalid", None

    # Failed admissibility gate: configuration is inadmissible, cannot count at all
    if not gates_passed:
        return "invalid", delta_pct

    # Insufficient replicates: cannot claim a win; remains an unconfirmed tie
    if replicates < required_replicates:
        return "tie", delta_pct

    if delta_pct > noise_floor_pct:
        return "win", delta_pct
    elif abs(delta_pct) <= noise_floor_pct:
        return "tie", delta_pct
    else:
        return "loss", delta_pct


STUDY_BASELINE_SCHEMA = "study-baseline.json"


def load_study_baseline_schema() -> dict[str, Any]:
    """Load the baseline schema that ships beside the runtime."""
    return jsonio.load_json(runtime_schema_path(STUDY_BASELINE_SCHEMA))


def set_baseline(study_id: str, baseline: dict[str, Any]) -> dict[str, Any]:
    """Record the measurement every arm will be compared against.

    Three rules the schema cannot state, checked here:

      * The metric must be the protocol's primary metric. A baseline in some
        other quantity is not a baseline for this study, and a delta computed
        across the two is meaningless in a way nothing downstream can detect.
      * Its provenance must be `measured`. Comparing every arm against a
        modelled number means the study spends weeks measuring its own
        prediction.
      * It must carry at least the pre-registered replicate count. A baseline
        is the number every later comparison rests on, so measuring it less
        carefully than the arms is backwards.

    A profile is required by the schema rather than here, because its absence
    is structural: without per-op attribution the first round cannot rank arms
    by headroom, and it falls back to guessing which layer to attack.
    """
    with study_lock(study_id):
        document = load_study(study_id)
        protocol = document.get("protocol", {})

        errors = validate_instance(baseline, load_study_baseline_schema())
        if errors:
            raise TpuAgentSuiteError("invalid baseline:\n  " +
                                     "\n  ".join(errors))

        metric = baseline["metric"]
        want = (protocol.get("primary_metric") or {}).get("name")
        if want and metric["name"] != want:
            errors.append(
                f"baseline measures {metric['name']!r} but the protocol's primary"
                f" metric is {want!r} -- a delta across two quantities is not a"
                " delta")

        if metric["provenance"] != "measured":
            errors.append(
                f"baseline provenance is {metric['provenance']!r}; a baseline must"
                " be measured. Every arm is compared against it, so a modelled one"
                " makes the study an evaluation of its own prediction")

        want_replicates = protocol.get("replicate_count")
        if want_replicates and metric["replicates"] < want_replicates:
            errors.append(
                f"baseline has {metric['replicates']} replicate(s) against a"
                f" pre-registered {want_replicates}. It is the number every"
                " comparison rests on")

        if errors:
            raise TpuAgentSuiteError("baseline contradicts the protocol:\n  " +
                                     "\n  ".join(errors))

        existing = document.get("baseline")
        if existing and existing != baseline:
            recorded = [
                arm.get("arm", "?") for arm in (document.get("arms") or [])
                if ((arm.get("result") or {}).get("metric") or {}
                    ).get("delta_pct") is not None
            ]
            if recorded:
                raise TpuAgentSuiteError(
                    "this study already has a baseline and arm(s)"
                    f" {', '.join(recorded)} have been measured against it."
                    " Replacing it would leave those deltas comparing to a number"
                    " that is no longer there, silently. Amend the protocol and"
                    " start a new study instead")

        document["baseline"] = baseline
        return save_study(document)


def record_round(study_id: str, record: dict[str, Any]) -> dict[str, Any]:
    """Write a round's outcome into the study state. The only state write.

    Whole-file semantics: `arms` replaces what was there rather than merging
    into it. A patch can leave the state half-updated if the round dies
    mid-write, and that is the one corruption this design refuses to risk.

    Validated twice, because the two questions are different. The schema asks
    whether the record is well-formed. `check_round_against_protocol` asks
    whether it honours the terms the study opened with.
    """
    with study_lock(study_id):
        document = load_study(study_id)
        previous_arms = document.get("arms") or []

        errors = validate_instance(record, load_study_round_schema())
        if errors:
            raise TpuAgentSuiteError("invalid round record:\n  " +
                                     "\n  ".join(errors))

        if not document.get("baseline"):
            deltas = [
                arm.get("arm", "?") for arm in record.get("arms", [])
                if ((arm.get("result") or {}).get("metric") or {}
                    ).get("delta_pct") is not None
            ]
            if deltas:
                raise TpuAgentSuiteError(
                    f"arm(s) {', '.join(deltas)} report a delta but the study has"
                    " no baseline. A delta is a comparison, and there is nothing"
                    " on the other side of it. Set one with `study set-baseline`"
                )

        errors = check_round_against_protocol(record,
                                              document.get("protocol", {}))
        if errors:
            raise TpuAgentSuiteError(
                "round record contradicts the protocol:\n  " +
                "\n  ".join(errors))

        document["arms"] = _merge_round_arms(
            document.get("arms") or [], record["arms"])
        _carry_failed_dispatches(document["arms"], previous_arms)
        if "next" in record:
            document["next"] = record["next"]
        return save_study(document)


def render_study_notes(study_id: str) -> str:
    """Render the human-readable view of a study, from the state, on demand.

    Rendered rather than stored. The file this replaces was written once at
    creation and never updated, so on the morning it mattered it still said
    "nothing measured yet" -- confidently stale, which is worse than absent.
    Deriving it means there is no artifact that can drift.
    """
    document = load_study(study_id)
    protocol = document.get("protocol", {})
    metric = protocol.get("primary_metric", {})
    lines = [
        f"# study {document['study_id']}",
        "",
        f"- target: {document.get('target', '?')}",
        f"- owner: {document.get('owner', '?')}",
        f"- status: {document['status']}   round: {study_round(document)}"
        f" of {protocol.get('stopping_rule', {}).get('max_rounds', '?')}",
        f"- mode: {protocol.get('mode', '?')}",
        f"- optimizing: {metric.get('name', '?')}"
        f" ({metric.get('direction', '?')}, {metric.get('unit', '?')})",
        f"- noise floor: {protocol.get('noise_floor_pct', '?')}%"
        f" at {protocol.get('replicate_count', '?')} replicate(s)",
        f"- updated: {document.get('updated_at', '?')}",
        "",
    ]

    baseline = document.get("baseline")
    if baseline:
        adopted = baseline.get("adopted_from_run")
        lines += [
            "## baseline",
            "",
            f"{baseline['metric']['value']} {baseline['metric'].get('unit','')}"
            f" from {baseline['execution_id']}" +
            (f" (adopted from run {adopted})" if adopted else ""),
            f"cell: {baseline['fingerprint']['cell']}",
            "",
        ]
    else:
        lines += [
            "## baseline", "", "None set. No arm can report a delta.", ""
        ]

    arms = document.get("arms") or []
    lines += ["## arms", ""]
    if not arms:
        lines.append("None recorded yet.")
    for arm in arms:
        result = arm.get("result") or {}
        verdict = result.get("verdict", "-")
        met = result.get("metric") or {}
        value = met.get("value")
        delta = met.get("delta_pct")
        shown = "" if value is None else f"  {value}"
        shown += "" if delta is None else f" ({delta:+}%)"
        prov = f" [{met['provenance']}]" if met.get("provenance") else ""
        lines.append(f"- **{arm['arm']}** ({arm['kind']}) {arm['status']}"
                     f" / {verdict}{shown}{prov} -- {arm['claim']}")

    if document.get("next"):
        lines += ["", "## next", "", document["next"]]
    if document.get("amendments"):
        lines += ["", "## amendments", ""]
        for a in document["amendments"]:
            lines.append(f"- r{a['revision']} superseded {a['superseded_at']}"
                         f": {a['reason']}")
    return "\n".join(lines) + "\n"


def set_protocol(study_id: str, protocol: dict[str, Any],
                 reason: str) -> dict[str, Any]:
    """Amend a study's protocol, keeping the superseded version and the why.

    `reason` is required and not decoration. A diff of two protocol blobs
    shows what changed and never why, and the amendment that matters is the
    one relaxing a gate at round 30 because the gate was failing. Without a
    stated reason that is indistinguishable from a correction.
    """
    if not reason.strip():
        raise TpuAgentSuiteError(
            "an amendment needs --reason. A diff shows what changed; only you"
            " can say why, and the amendment worth reviewing later is the one"
            " that relaxed a bar because the bar was being missed")

    errors = validate_study_protocol(protocol)
    if errors:
        raise TpuAgentSuiteError("invalid study protocol:\n  " +
                                 "\n  ".join(errors))

    with study_lock(study_id):
        document = load_study(study_id)
        document["amendments"].append({
            "protocol": document["protocol"],
            "revision": document["protocol_revision"],
            "superseded_at": utc_now_iso(),
            "reason": reason.strip(),
        })
        document["protocol"] = protocol
        document["protocol_revision"] = document["protocol_revision"] + 1
        return save_study(document)


def study_view(document: dict[str, Any]) -> dict[str, Any]:
    """The study as a reader sees it: stored fields plus derived ones.

    `round` is derived rather than stored, so it has to be added back here or
    it is invisible to everyone outside this module -- which is most readers,
    since the round agent and any monitor reach the study through the CLI.
    Deriving a value and then not surfacing it is worse than storing it: the
    stopping rule becomes uncheckable by the one agent that has to check it.

    The stored document stays canonical. Nothing here is written back.
    """
    view = dict(document)
    view["round"] = study_round(document)
    return view


def load_study(study_id: str) -> dict[str, Any]:
    """Read a study's study state, failing loudly when it is absent."""
    doc_path = paths.get_study_root(study_id) / STUDY_DOCUMENT
    if not doc_path.is_file():
        raise TpuAgentSuiteError(
            f"study {study_id!r} is not in {paths.get_studies_root()} --"
            " start it first")
    return jsonio.load_json(doc_path)


_ARM_VERDICT_PRIORITY = {
    "invalid": 0,
    "loss": 1,
    "tie": 2,
    "win": 5,
}

_STEP_FUNCTION_KINDS = ("memory-capacity", "recompilation-padding")


def arm_expected_gain(arm: Mapping[str, Any]) -> float:
    """Compute expected ceiling gain for an arm, shared by claim_arm and eviction.

    Uses Amdahl `share * headroom` by default, with a fallback to `headroom`
    when `share == 0.0` on step-function kinds (`memory-capacity`,
    `recompilation-padding`) where baseline profile share under-represents the
    unlock (`tpu-performance-optimization.md:147`).
    """
    share = float(arm.get("share") or 0.0)
    headroom = float(arm.get("headroom") or 0.0)
    product = share * headroom
    if product == 0.0 and headroom > 0.0 and str(arm.get(
            "kind", "")) in _STEP_FUNCTION_KINDS:
        return headroom
    return product


def _carry_failed_dispatches(arms: list[dict[str, Any]],
                             previous_arms: list[dict[str, Any]]) -> None:
    """Keep each arm's `failed_dispatches` across a whole-file round rewrite.

    The count is what blocks an arm nobody can implement. A round that rewrites
    the arms without it would reset the cap, so a count can only go up here.
    """
    previous = {
        str(a.get("arm")): int(a.get("failed_dispatches", 0))
        for a in previous_arms if a.get("arm")
    }
    for arm in arms:
        before = previous.get(str(arm.get("arm")), 0)
        if before > int(arm.get("failed_dispatches", 0)):
            arm["failed_dispatches"] = before


def _merge_round_arms(
    existing_arms: list[dict[str, Any]],
    incoming_arms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge round arm updates when multi-arm claims are active without clobbering concurrent workers.

    When no multi-arm claims (`in_flight`, `claimed_by`, `claimed_at`) exist on
    either side, preserves single-worker whole-file replacement semantics.
    When concurrent worker claims exist:
    1. Preserves any arm present in `existing_arms` that is omitted from
       `incoming_arms` (when a parallel worker submits only its own arm subset).
    2. If an arm in `existing_arms` is already `decided` (with a `result`) and
       `incoming_arms` carries a stale `open` or `in_flight` snapshot of that
       same arm, retains the `decided` record so another agent's completed result
       is never overwritten.
    3. If an arm in `existing_arms` was concurrently claimed (`in_flight` with
       `claimed_by` / `claimed_at`) while the caller's snapshot still had it as
       `open`, retains the `in_flight` claim.
    """
    has_concurrent_claims = any(
        a.get("status") == "in_flight" or a.get("claimed_by")
        or a.get("claimed_at") for a in (existing_arms + incoming_arms))
    if not has_concurrent_claims:
        return [dict(inc) for inc in incoming_arms]

    existing_by_id = {
        str(a.get("arm")): dict(a)
        for a in existing_arms if a.get("arm")
    }
    incoming_ids = {str(a.get("arm")) for a in incoming_arms if a.get("arm")}
    merged: list[dict[str, Any]] = []

    for inc in incoming_arms:
        arm_id = str(inc.get("arm", ""))
        prev = existing_by_id.get(arm_id)
        inc_copy = dict(inc)
        if prev is not None:
            prev_status = prev.get("status")
            inc_status = inc_copy.get("status")
            if prev_status == "decided" and inc_status in ("open",
                                                           "in_flight"):
                merged.append(prev)
                continue
            if (prev_status == "in_flight" and inc_status == "open"
                    and (prev.get("claimed_by") or prev.get("claimed_at"))):
                merged.append(prev)
                continue
            for claim_meta in ("claimed_by", "claimed_at"):
                if prev.get(claim_meta) and claim_meta not in inc_copy:
                    inc_copy[claim_meta] = prev[claim_meta]
        merged.append(inc_copy)

    for prev in existing_arms:
        arm_id = str(prev.get("arm", ""))
        if arm_id and arm_id not in incoming_ids:
            merged.append(dict(prev))

    return merged


def _arm_eviction_sort_key(
        indexed_arm: tuple[int, dict[str, Any]]) -> tuple[int, float, int]:
    """Order arms from weakest/oldest (evict first) to strongest/proven.

    Eviction tiers (lowest evicted first, oldest first within each tier):
      0: decided/blocked `invalid`
      1: decided/blocked `loss`
      2: decided/blocked `tie`
      3: untested `open` hypothesis (weakest `arm_expected_gain` first)
      4: active `in_flight` execution
      5: decided `win` (smallest gain first; never evicted while any non-win exists)
    """
    idx, arm = indexed_arm
    status = str(arm.get("status", ""))
    if status == "in_flight":
        return (4, 0.0, idx)
    if status == "open":
        return (3, arm_expected_gain(arm), idx)
    result = arm.get("result") or {}
    verdict = str(result.get("verdict", "tie"))
    tier = _ARM_VERDICT_PRIORITY.get(verdict, 2)
    metric = result.get("metric") or {}
    delta_mag = (abs(float(metric.get("delta_pct") or 0.0))
                 if verdict == "win" else 0.0)
    return (tier, delta_mag, idx)


def _enforce_study_state_caps(document: dict[str, Any]) -> None:
    """Enforce bounded section caps on `study.json`, archiving overflow to `journal.jsonl`."""
    study_id = document.get("study_id", "")
    amendments = document.get("amendments")
    if isinstance(amendments, list) and len(amendments) > MAX_STUDY_AMENDMENTS:
        overflow_count = len(amendments) - MAX_STUDY_AMENDMENTS
        for evicted in amendments[:overflow_count]:
            append_study_journal(
                study_id,
                {
                    "section": "amendments",
                    "evicted": evicted
                },
            )
        document["amendments"] = amendments[overflow_count:]

    arms = document.get("arms")
    if isinstance(arms, list) and len(arms) > MAX_STUDY_ARMS:
        overflow_count = len(arms) - MAX_STUDY_ARMS
        ranked = sorted(enumerate(arms), key=_arm_eviction_sort_key)
        evicted_indices = sorted(idx for idx, _ in ranked[:overflow_count])
        evicted_set = set(evicted_indices)
        for idx in evicted_indices:
            append_study_journal(
                study_id,
                {
                    "section": "arms",
                    "evicted": arms[idx]
                },
            )
        document["arms"] = [
            arm for idx, arm in enumerate(arms) if idx not in evicted_set
        ]


def save_study(document: dict[str, Any]) -> dict[str, Any]:
    """Write a study state back, enforcing section caps and stamping `updated_at`.

    `updated_at` is the liveness signal a watcher reads to tell a study that
    is working from one that died at 01:00, so it comes from the clock here
    rather than from whatever the caller believed the time was.
    """
    study_id = document.get("study_id", "")
    utils.validate_study_id(study_id)
    with study_lock(study_id):
        _enforce_study_state_caps(document)
        document["updated_at"] = utc_now_iso()
        jsonio.write_json(
            paths.get_study_root(study_id) / STUDY_DOCUMENT, document)
        return document


def study_for_run(run_id: str) -> str | None:
    """The study a run belongs to, or None.

    Scans every study, not only the open ones: a run attached to a closed
    study still belongs to it, and letting it be re-attached elsewhere would
    rewrite history rather than extend it.
    """
    utils.validate_run_id(run_id)
    for document in list_studies():
        if run_id in (document.get("runs") or []):
            return document.get("study_id")
    return None


def attach_run_to_study(study_id: str, run_id: str) -> dict[str, Any]:
    """Attach a run to a study. One run, one study, enforced here.

    This is the single rule the whole state model rests on. Without it
    nothing establishes which study a run belongs to, and every consumer
    downstream has to *infer* ownership it cannot know -- which is how one
    session's events ended up in an unrelated study's timeline.

    With it, three things follow rather than being arranged separately: an
    event routes to exactly one study, the round counter can be the count of
    attached runs, and `session` becomes a harness correlation key rather
    than a concept the study layer has to model.

    Re-attaching the same run to the same study is a no-op, because a retried
    command should not be an error. Attaching it to a *different* study is
    refused: a round is one run, and a retry is a new round.
    """
    utils.validate_run_id(run_id)
    with study_lock("__store__"):
        with study_lock(study_id):
            document = load_study(study_id)

            if run_id in document["runs"]:
                return document

            owner = study_for_run(run_id)
            if owner is not None:
                raise TpuAgentSuiteError(
                    f"run {run_id} is already attached to study {owner!r} -- a round"
                    " is one run, and a retry is a new round. Start a fresh session;"
                    " that is what keeps round 60 costing what round 2 cost")

            document["runs"].append(run_id)
            saved = save_study(document)

            # The attach is also the study's "a session joined" marker. SessionStart
            # fires before any agent turn, so at that moment the run exists and
            # belongs to nothing, and a strictly-routed SessionStart event could
            # never reach a study. This event carries the same information at the
            # first moment it can be attributed.
            append_study_event(study_id,
                               "attach",
                               run_id=run_id,
                               round=study_round(saved))
            return saved


def list_studies() -> list[dict[str, Any]]:
    """Return every study state, most recently updated first.

    A study whose directory exists but whose study state is unreadable is reported
    with an `error` field rather than skipped: a study that vanishes from the
    board because its file broke is the failure this listing exists to catch.
    """
    root = paths.get_studies_root()
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        doc_path = child / STUDY_DOCUMENT
        if not doc_path.is_file():
            out.append({"study_id": child.name, "error": "no study.json"})
            continue
        try:
            out.append(jsonio.load_json(doc_path))
        except (TpuAgentSuiteError, ValueError, OSError) as exc:
            # jsonio.load_json wraps a decode failure in TpuAgentSuiteError, so catching
            # only the stdlib errors would let one corrupt study state take the
            # whole board down -- the opposite of what this listing is for.
            out.append({"study_id": child.name, "error": str(exc)})
    out.sort(key=lambda d: str(d.get("updated_at", "")), reverse=True)
    return out


def close_study(study_id: str, outcome: str = "") -> dict[str, Any]:
    """Mark a study closed. Closing an already-closed study is not an error.

    `closed_at` records when the study actually ended and is written once. A
    later repeat -- a cleanup script, a retried round -- must not move it, or
    the record of when work stopped becomes the record of when someone last
    ran the command.
    """
    with study_lock(study_id):
        document = load_study(study_id)
        already_closed = (document.get("status") == STUDY_STATUS_CLOSED
                          and document.get("closed_at"))
        document["status"] = STUDY_STATUS_CLOSED
        if not already_closed:
            document["closed_at"] = utc_now_iso()
        if outcome:
            document["outcome"] = outcome
        return save_study(document)


_TZ_ALIASES = {
    "PDT": "America/Los_Angeles",
    "PST": "America/Los_Angeles",
    "PT": "America/Los_Angeles",
    "UTC": "UTC",
    "Z": "UTC",
}
DEFAULT_CLAIM_TIMEZONE = "America/Los_Angeles"
DEFAULT_CLAIM_TTL_SECONDS = 3600.0
# A failed dispatch is an implementation the executor could not repair. The
# arm reopens for another try, and is blocked once this many have failed.
MAX_FAILED_DISPATCHES = 2


def study_timestamp_iso(timezone_name: str | None = None) -> str:
    """Return an ISO-8601 timestamp in the configured timezone (default: America/Los_Angeles PDT/PST)."""
    raw_tz = (timezone_name or os.environ.get("TPU_AGENT_SUITE_TIMEZONE", "")
              or DEFAULT_CLAIM_TIMEZONE).strip()
    resolved_tz = _TZ_ALIASES.get(raw_tz.upper(), raw_tz)
    if resolved_tz == "UTC":
        return utc_now_iso()
    try:
        tz = ZoneInfo(resolved_tz)
        return datetime.datetime.now(tz).isoformat(timespec="seconds")
    except Exception:
        return utc_now_iso()


def _parse_iso_timestamp(value: str) -> datetime.datetime | None:
    if not value or not isinstance(value, str):
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        dt = datetime.datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt
    except ValueError:
        return None


def _expire_stale_in_flight_arms(
    arms: list[dict[str, Any]],
    claim_ttl_seconds: float,
) -> list[str]:
    """Reclaim orphaned `in_flight` arms whose `claimed_at` exceeds `claim_ttl_seconds`."""
    if claim_ttl_seconds <= 0:
        return []
    now = datetime.datetime.now(datetime.timezone.utc)
    reclaimed: list[str] = []
    for arm in arms:
        if arm.get("status") != "in_flight":
            continue
        claimed_dt = _parse_iso_timestamp(str(arm.get("claimed_at", "")))
        if claimed_dt is None:
            continue
        age_s = (now -
                 claimed_dt.astimezone(datetime.timezone.utc)).total_seconds()
        if age_s >= claim_ttl_seconds:
            arm["status"] = "open"
            arm.pop("claimed_at", None)
            arm.pop("claimed_by", None)
            msg = f"stale in-flight claim expired after {int(age_s)}s"
            arm["summary"] = msg
            arm["release_reason"] = msg
            reclaimed.append(str(arm.get("arm", "")))
    return reclaimed


def claim_arm(
    study_id: str,
    arm_id: str | None = None,
    worker_id: str = "",
    *,
    timezone_name: str | None = None,
    claim_ttl_seconds: float = DEFAULT_CLAIM_TTL_SECONDS,
) -> dict[str, Any]:
    """Atomically reserve an arm for execution under advisory file lock.

    First expires any orphaned `in_flight` claims older than `claim_ttl_seconds`.
    Then transitions an arm from `open` to `in_flight`, clears any stale release
    metadata, stamps `claimed_at` (in PDT/PST by default or `TPU_AGENT_SUITE_TIMEZONE`)
    and optional `claimed_by`, writes the study document back, and returns the
    claimed arm. If `arm_id` is None, claims the highest-priority `open` arm
    ranked by `arm_expected_gain(arm)` descending, then oldest index.
    """
    with study_lock(study_id):
        document = load_study(study_id)
        arms = document.get("arms") or []
        _expire_stale_in_flight_arms(arms, claim_ttl_seconds)

        if arm_id is not None:
            target_arm = None
            for arm in arms:
                if arm.get("arm") == arm_id:
                    target_arm = arm
                    break
            if target_arm is None:
                raise TpuAgentSuiteError(
                    f"arm {arm_id!r} not found in study {study_id!r}")
            if target_arm.get("status") != "open":
                raise TpuAgentSuiteError(
                    f"arm {arm_id!r} cannot be claimed: status is "
                    f"{target_arm.get('status')!r} (expected 'open')")
        else:
            open_arms: list[tuple[float, int, dict[str, Any]]] = []
            for idx, arm in enumerate(arms):
                if arm.get("status") == "open":
                    gain = arm_expected_gain(arm)
                    open_arms.append((-gain, idx, arm))
            if not open_arms:
                raise TpuAgentSuiteError(
                    f"no open arms available to claim in study {study_id!r}")
            if len(open_arms) > 1 and all(g == 0.0 for g, _, _ in open_arms):
                print(
                    f"WARNING: all {len(open_arms)} open arms in study {study_id!r} "
                    "scored 0.0 expected gain (missing share/headroom); falling back to list order.",
                    file=sys.stderr,
                )
            open_arms.sort()
            target_arm = open_arms[0][2]

        for stale_key in ("claimed_at", "claimed_by", "release_reason",
                          "summary", "blocked_on", "result"):
            target_arm.pop(stale_key, None)
        target_arm["status"] = "in_flight"
        target_arm["claimed_at"] = study_timestamp_iso(timezone_name)
        if worker_id:
            target_arm["claimed_by"] = worker_id
        save_study(document)
        return dict(target_arm)


def release_arm(
    study_id: str,
    arm_id: str,
    status: str = "open",
    reason: str = "",
    *,
    summary: str = "",
    blocked_on: str = "",
    execution_id: str = "preflight-rejected",
    failed: bool = False,
) -> dict[str, Any]:
    """Release an `in_flight` arm under advisory file lock.

    Supports transitioning an `in_flight` arm to:
    - `open`: return to the pool for retry.
    - `blocked`: mark blocked (requires `blocked_on` or `summary`/`reason`).
    - `invalid`: close the arm as `decided` with `result.verdict = "invalid"`.

    `failed` records a dispatch that ended `failed` with no verdict. It
    increments `failed_dispatches`, and once that reaches
    `MAX_FAILED_DISPATCHES` an `open` release becomes `blocked` instead.

    Clears stale metadata (`claimed_at`, `claimed_by`, `release_reason`,
    `summary`, `blocked_on`, `result`) before applying the new release state so
    subsequent releases never carry over stale fields from earlier rounds.
    """
    valid_statuses = ("open", "blocked", "invalid")
    if status not in valid_statuses:
        raise TpuAgentSuiteError(
            f"invalid release status {status!r} (expected one of {valid_statuses})"
        )
    if failed and status == "invalid":
        raise TpuAgentSuiteError(
            "a failed dispatch has no verdict; release it as open or blocked,"
            " not invalid")
    if failed and not (summary or reason).strip():
        raise TpuAgentSuiteError(
            "releasing a failed dispatch requires summary (or reason): the next"
            " attempt starts from it")

    effective_summary = (summary or reason).strip()
    effective_blocked_on = (blocked_on or effective_summary).strip()
    if status == "blocked" and not effective_blocked_on:
        raise TpuAgentSuiteError(
            "releasing an arm with status='blocked' requires blocked_on (or summary/reason)"
        )

    with study_lock(study_id):
        document = load_study(study_id)
        arms = document.get("arms") or []
        target_arm = None
        for arm in arms:
            if arm.get("arm") == arm_id:
                target_arm = arm
                break
        if target_arm is None:
            raise TpuAgentSuiteError(
                f"arm {arm_id!r} not found in study {study_id!r}")
        if target_arm.get("status") != "in_flight":
            raise TpuAgentSuiteError(
                f"arm {arm_id!r} cannot be released: status is "
                f"{target_arm.get('status')!r} (expected 'in_flight')")

        for stale_key in ("claimed_at", "claimed_by", "release_reason",
                          "summary", "blocked_on", "result"):
            target_arm.pop(stale_key, None)

        if failed:
            failures = int(target_arm.get("failed_dispatches", 0)) + 1
            target_arm["failed_dispatches"] = failures
            if status == "open" and failures >= MAX_FAILED_DISPATCHES:
                status = "blocked"
                effective_blocked_on = (
                    f"{failures} dispatches failed with no verdict; last:"
                    f" {effective_summary}")

        if status == "invalid":
            target_arm["status"] = "decided"
            target_arm["result"] = {
                "verdict": "invalid",
                "execution_id": execution_id or "preflight-rejected",
            }
        else:
            target_arm["status"] = status

        if status == "blocked":
            target_arm["blocked_on"] = effective_blocked_on

        if effective_summary:
            target_arm["summary"] = effective_summary
            target_arm["release_reason"] = effective_summary

        save_study(document)
        return dict(target_arm)


# =====================================================================
# Command-line surface
# =====================================================================


def register_study_parser(sub: Any) -> None:
    """Attach the `study` command and its verbs to the runtime's subparsers."""
    p_study = sub.add_parser("study",
                             help="Manage long-running optimization studies")
    study_sub = p_study.add_subparsers(dest="study_action", required=True)

    p_study_start = study_sub.add_parser("start", help="Open a new study")
    p_study_start.add_argument("--study-id",
                               required=True,
                               help="Study slug, e.g. qwen35-397b-ttft")
    p_study_start.add_argument("--package",
                               required=True,
                               help="Owning package name")
    p_study_start.add_argument(
        "--target",
        required=True,
        help="What is being optimized, e.g. a model name. The grouping key:"
        " 'every study on this model' is the query people actually run")
    p_study_start.add_argument(
        "--owner",
        default="",
        help="Who to ask about this study. Defaults to the invoking user")
    p_study_start.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="JSON file holding the pre-registered protocol. Required: a study"
        " opened without one has no stopping rule, gates or budget, and every"
        " guard downstream fails silently")

    p_study_status = study_sub.add_parser("status",
                                          help="Print one study's study state")
    p_study_status.add_argument("--study-id", required=True, help="Study slug")

    study_sub.add_parser("list", help="List every study, freshest first")

    p_study_events = study_sub.add_parser(
        "events", help="Read a study's machine-authored lifecycle events")
    p_study_events.add_argument("--study-id", required=True, help="Study slug")
    p_study_events.add_argument("--limit",
                                type=int,
                                default=0,
                                help="Keep only the newest N (0 = all)")

    p_study_journal = study_sub.add_parser(
        "journal", help="Read a study's agent-authored overflow journal")
    p_study_journal.add_argument("--study-id",
                                 required=True,
                                 help="Study slug")
    p_study_journal.add_argument("--limit",
                                 type=int,
                                 default=0,
                                 help="Keep only the newest N (0 = all)")

    p_study_attach = study_sub.add_parser(
        "attach-run", help="Record that a run belongs to a study")
    p_study_attach.add_argument("--study-id", required=True, help="Study slug")
    p_study_attach.add_argument("--run-id", required=True, help="Run ID")

    p_study_record = study_sub.add_parser(
        "record-round",
        help="Write a round's outcome into the study state (the only write)")
    p_study_record.add_argument("--study-id", required=True, help="Study slug")
    p_study_record.add_argument(
        "--round",
        dest="round_file",
        type=Path,
        required=True,
        help="JSON file matching study-round.json. Replaces the arms whole;"
        " a patch can leave the state half-updated if the round dies")

    p_study_amend = study_sub.add_parser(
        "set-protocol", help="Amend the protocol, keeping the superseded one")
    p_study_amend.add_argument("--study-id", required=True, help="Study slug")
    p_study_amend.add_argument("--protocol",
                               type=Path,
                               required=True,
                               help="JSON file holding the new protocol")
    p_study_amend.add_argument(
        "--reason",
        required=True,
        help="Why. A diff shows what changed and never why, and the amendment"
        " worth reviewing later is the one that relaxed a failing bar")

    p_study_baseline = study_sub.add_parser(
        "set-baseline",
        help="Record the measurement every arm is compared against")
    p_study_baseline.add_argument("--study-id",
                                  required=True,
                                  help="Study slug")
    p_study_baseline.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="JSON file matching study-baseline.json. A profile is required:"
        " without per-op attribution the first round cannot rank arms")

    p_study_notes = study_sub.add_parser(
        "notes", help="Render the human-readable view, from the state")
    p_study_notes.add_argument("--study-id", required=True, help="Study slug")

    p_study_append = study_sub.add_parser(
        "append-journal", help="Archive a row that overflowed the state")
    p_study_append.add_argument("--study-id", required=True, help="Study slug")
    p_study_append.add_argument("--row",
                                type=Path,
                                required=True,
                                help="JSON file holding the overflowed row")

    p_study_close = study_sub.add_parser("close", help="Close a study")
    p_study_close.add_argument("--study-id", required=True, help="Study slug")
    p_study_close.add_argument("--outcome",
                               default="",
                               help="One line on how the study ended")

    p_study_claim = study_sub.add_parser(
        "claim-arm",
        help="Atomically claim an open arm for parallel execution")
    p_study_claim.add_argument("--study-id", required=True, help="Study slug")
    p_study_claim.add_argument("--arm",
                               default=None,
                               help="Specific arm ID to claim")
    p_study_claim.add_argument("--worker-id",
                               default="",
                               help="Worker/job identifier")
    p_study_claim.add_argument(
        "--timezone",
        default=None,
        help=
        "Timezone for claimed_at (default: America/Los_Angeles PDT/PST or TPU_AGENT_SUITE_TIMEZONE)"
    )
    p_study_claim.add_argument(
        "--claim-ttl-seconds",
        type=float,
        default=DEFAULT_CLAIM_TTL_SECONDS,
        help="Expire in_flight claims older than this TTL before claiming")

    p_study_release = study_sub.add_parser(
        "release-arm",
        help="Release an in_flight arm back to open, blocked, or invalid")
    p_study_release.add_argument("--study-id",
                                 required=True,
                                 help="Study slug")
    p_study_release.add_argument("--arm",
                                 required=True,
                                 help="Arm ID to release")
    p_study_release.add_argument("--status",
                                 default="open",
                                 choices=["open", "blocked", "invalid"],
                                 help="New status (open, blocked, or invalid)")
    p_study_release.add_argument(
        "--summary",
        default="",
        help="Summary or conclusion for releasing the arm")
    p_study_release.add_argument("--reason",
                                 default="",
                                 help="Alias for --summary")
    p_study_release.add_argument(
        "--blocked-on",
        default="",
        help="Blocker description (required when --status=blocked)")
    p_study_release.add_argument(
        "--failed",
        action="store_true",
        help="The dispatch ended failed with no verdict: count it in"
        " failed_dispatches, and block the arm at"
        f" {MAX_FAILED_DISPATCHES}")


def dispatch_study_command(args: Any) -> None:
    """Run one `study` verb. Printing is the command's only output."""
    action = args.study_action
    if action == "start":
        print(
            jsonio.format_json(
                start_study(study_id=args.study_id,
                            protocol=jsonio.load_json(args.protocol),
                            package=args.package,
                            target=args.target,
                            owner=args.owner)))
    elif action == "status":
        print(jsonio.format_json(study_view(load_study(args.study_id))))
    elif action == "list":
        print(
            jsonio.format_json([
                study_view(d) if "study_id" in d and "runs" in d else d
                for d in list_studies()
            ]))
    elif action == "events":
        print(
            jsonio.format_json(
                read_study_events(args.study_id, limit=args.limit)))
    elif action == "journal":
        print(
            jsonio.format_json(
                read_study_journal(args.study_id, limit=args.limit)))
    elif action == "attach-run":
        print(
            jsonio.format_json(attach_run_to_study(args.study_id,
                                                   args.run_id)))
    elif action == "record-round":
        print(
            jsonio.format_json(
                record_round(args.study_id,
                             jsonio.load_json(args.round_file))))
    elif action == "set-protocol":
        print(
            jsonio.format_json(
                set_protocol(args.study_id, jsonio.load_json(args.protocol),
                             args.reason)))
    elif action == "set-baseline":
        print(
            jsonio.format_json(
                set_baseline(args.study_id, jsonio.load_json(args.baseline))))
    elif action == "notes":
        print(render_study_notes(args.study_id), end="")
    elif action == "append-journal":
        print(
            jsonio.format_json(
                append_study_journal(args.study_id,
                                     jsonio.load_json(args.row))))
    elif action == "close":
        print(
            jsonio.format_json(close_study(args.study_id,
                                           outcome=args.outcome)))
    elif action == "claim-arm":
        print(
            jsonio.format_json(
                claim_arm(args.study_id,
                          arm_id=args.arm,
                          worker_id=args.worker_id,
                          timezone_name=args.timezone,
                          claim_ttl_seconds=args.claim_ttl_seconds)))
    elif action == "release-arm":
        print(
            jsonio.format_json(
                release_arm(args.study_id,
                            arm_id=args.arm,
                            status=args.status,
                            reason=args.reason,
                            summary=args.summary,
                            blocked_on=args.blocked_on,
                            failed=args.failed)))
