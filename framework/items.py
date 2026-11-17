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
"""The shared store of delegated work items.

A delegation is a row in a locked file, not a call that blocks its caller.
One agent queues a request for another; a worker claims it, writes progress
while it works, and answers it when it is done. Every one of those writes is
a state change the rest of the run can see, which is the point: "done" stops
being something a stop hook has to infer from a turn ending and becomes a
status anyone can read.

What the store makes possible, and foreground delegation does not:

- **Fan-out.** Queueing is cheap and non-blocking, so independent work is
  queued together and limited by `capacity`, not by how deep one chain goes.
- **Waiting without holding a slot.** A worker that needs someone else's
  result records itself as a waiter, marks its item `waiting` and stops. The
  result readies it again (`dispatch` reports it as a resume).
- **Acting on a partial result.** A producer publishes named parts of its
  result as it goes, and a consumer queued `--after I1#gaps[0]` starts on the
  first part while the producer is still running.
- **Dead workers.** A claim is a lease. An expired lease turns the item
  `stalled` and surfaces it for a decision instead of losing it.

Nothing here dispatches a hook or spawns an agent: this module only owns the
state and the event log.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import re
import time
from typing import Any, Callable, Iterator

from framework import agent_contracts, jsonio, paths, study, utils

Path = paths.Path

STORE_DOCUMENT = "items.json"
STORE_EVENTS = "events.jsonl"
STORE_VERSION = 1

# `waiting` is not an end state: the worker released its slot and is resumed
# once what it waits on lands.
STATUS_QUEUED = "queued"
STATUS_CLAIMED = "claimed"
STATUS_WAITING = "waiting"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_STALLED = "stalled"

OPEN_STATUSES = frozenset(
    {STATUS_QUEUED, STATUS_CLAIMED, STATUS_WAITING, STATUS_STALLED})
CLOSED_STATUSES = frozenset({STATUS_COMPLETED, STATUS_FAILED})
# Only a claim holds a capacity slot; releasing it is why `waiting` exists.
ACTIVE_STATUSES = frozenset({STATUS_CLAIMED})

# Dispatch policies order the runnable items; they never change which are
# runnable.
POLICY_BREADTH_FIRST = "breadth-first"
POLICY_DEPTH_FIRST = "depth-first"
POLICY_PRIORITY = "priority"
POLICIES = (POLICY_BREADTH_FIRST, POLICY_DEPTH_FIRST, POLICY_PRIORITY)

# With more work than slots, every lead gets a first look before any one is
# followed deeper.
DEFAULT_POLICY = POLICY_BREADTH_FIRST

# At or below the harness's live-subagent limit: a request past it cannot
# spawn, so the store queues it instead.
DEFAULT_CAPACITY = 10

# Matches a study arm's claim (`study.DEFAULT_CLAIM_TTL_SECONDS`): long TPU
# work writes rarely.
DEFAULT_LEASE_SECONDS = 3600.0

# Under the harness's shell cap (`harness_rules.SHELL_TIMEOUT_MS`): a watcher
# the harness killed and one that found nothing look the same to the caller.
DEFAULT_WATCH_TIMEOUT_SECONDS = 1800.0

# Bounds how long a published partial result waits before its consumer starts.
DEFAULT_WATCH_POLL_SECONDS = 2.0

# `I3` for a whole result, `I3#gaps[0]` for one named part of it.
_REF_RE = re.compile(r"^(?P<item>I[0-9]+)(?:#(?P<part>[^\s#]+))?$")


class ItemError(utils.TpuAgentSuiteError):
    """A write the store refuses: a bad reference, or a wrong-state move."""


@contextlib.contextmanager
def store_lock(run_id: str) -> Iterator[None]:
    """Hold the run's item store exclusively; re-entrant, across processes.

    Refuses a run no session started, so no caller builds a second store.
    """
    paths.get_started_run_root(run_id)
    with utils.path_lock(paths.get_items_lock(run_id)):
        yield


def _now() -> str:
    return utils.utc_now_iso()


def parse_ref(ref: str) -> tuple[str, str]:
    """Split a result reference into its item id and optional part name."""
    match = _REF_RE.match((ref or "").strip())
    if not match:
        raise ItemError(
            f"{ref!r} is not a result reference: write I3, or I3#gaps[0] for"
            " one named part of I3's result")
    return match.group("item"), match.group("part") or ""


def document_path(run_id: str) -> Path:
    """Where a run's item document lives."""
    return paths.get_items_dir(run_id) / STORE_DOCUMENT


def events_path(run_id: str) -> Path:
    """Where a run's state-change log lives."""
    return paths.get_items_dir(run_id) / STORE_EVENTS


def _empty_document(run_id: str) -> dict[str, Any]:
    return {
        "version": STORE_VERSION,
        "run_id": run_id,
        "capacity": DEFAULT_CAPACITY,
        "policy": DEFAULT_POLICY,
        "next_id": 1,
        "items": {},
    }


def load(run_id: str) -> dict[str, Any]:
    """Read the run's item document, or an empty one if nothing is queued."""
    utils.validate_run_id(run_id)
    path = document_path(run_id)
    if not path.is_file():
        return _empty_document(run_id)
    document = jsonio.load_json(path)
    if not isinstance(document, dict):
        raise ItemError(f"{path} is not an item store document")
    return document


def save(run_id: str, document: dict[str, Any]) -> dict[str, Any]:
    """Replace the run's item document atomically. Call under `store_lock`."""
    jsonio.write_json(document_path(run_id), document)
    return document


def read_events(run_id: str, limit: int = 0) -> list[dict[str, Any]]:
    """Read the run's state-change log, newest `limit` records (0 = all)."""
    utils.validate_run_id(run_id)
    return jsonio.read_jsonl(events_path(run_id), limit=limit)


def _state_change(
    run_id: str,
    item: dict[str, Any],
    kind: str,
    old_status: str,
    *,
    part: str = "",
    detail: str = "",
) -> dict[str, Any]:
    """Append one `StateChange` record to the run's event log.

    `kind` names the write, which is what a handler dispatches on; `part` is
    the result slot a `progress` write filled, which a waiter on a part keys
    off.
    """
    record = jsonio.append_jsonl(
        events_path(run_id), {
            "at": _now(),
            "event": "StateChange",
            "kind": kind,
            "item": item["id"],
            "to": item["to"],
            "part": part,
            "old_status": old_status,
            "new_status": item["status"],
            "owner": item.get("owner", ""),
            "depth": item.get("depth", 1),
            "detail": detail,
        })
    _record_in_studies(run_id, record)
    return record


def _record_in_studies(run_id: str, record: dict[str, Any]) -> None:
    """Mirror a state change into the timeline of every study on this run.

    Queueing is a CLI write, not a harness event, so without this a study
    stalled on a worker that never answered shows a gap with no cause.
    Best-effort, like `record-event`: a logging failure must never fail a
    delegation.
    """
    # `kind` and `at` are the study stream's own columns; the change's kind
    # travels as `change`.
    fields = {k: v for k, v in record.items() if k not in ("kind", "at")}
    fields["change"] = record.get("kind", "")
    try:
        for study_id in study.studies_for_event(run_id):
            study.append_study_event(study_id, "items", **fields)
    except (utils.TpuAgentSuiteError, OSError):
        pass


def _require(document: dict[str, Any], item_id: str) -> dict[str, Any]:
    item = (document.get("items") or {}).get(item_id)
    if item is None:
        raise ItemError(f"no item {item_id} in this store")
    return item


def _chain(document: dict[str, Any], item_id: str) -> list[str]:
    """The agents on an item's parent chain, closest first."""
    chain: list[str] = []
    seen: set[str] = set()
    current = item_id
    while current and current not in seen:
        seen.add(current)
        item = (document.get("items") or {}).get(current)
        if item is None:
            break
        chain.append(item["to"])
        current = item.get("parent") or ""
    return chain


def ref_satisfied(document: dict[str, Any], ref: str) -> bool:
    """Is there anything left for a waiter on `ref` to wait for?

    A part is satisfied once published; otherwise the producer closing
    settles it, and a failure counts: the waiter wakes and reads the verdict.
    """
    item_id, part = parse_ref(ref)
    item = _require(document, item_id)
    if part and part in (item.get("parts") or {}):
        return True
    return item.get("status") in CLOSED_STATUSES


def _waits_on(document: dict[str, Any], start: str, goal: str) -> bool:
    """Does `start`, through its unsatisfied blockers, wait on `goal`?"""
    stack, seen = [start], set()
    items_by_id = document.get("items") or {}
    while stack:
        current = stack.pop()
        if current == goal:
            return True
        if current in seen or current not in items_by_id:
            continue
        seen.add(current)
        stack.extend(
            parse_ref(ref)[0]
            for ref in blockers(document, items_by_id[current]))
    return False


def blockers(document: dict[str, Any], item: dict[str, Any]) -> list[str]:
    """The references this item still waits on, in declaration order."""
    return [
        ref for ref in item.get("blocked_on") or []
        if not ref_satisfied(document, ref)
    ]


def runnable_action(document: dict[str, Any], item: dict[str,
                                                         Any]) -> str | None:
    """`start`, `resume`, `decide` (a stalled item), or None.

    A stalled item is surfaced for a decision, not restarted: a worker that
    died halfway may have left side effects a blind retry would repeat.
    """
    status = item.get("status")
    if status == STATUS_STALLED:
        return "decide"
    if status not in (STATUS_QUEUED, STATUS_WAITING):
        return None
    if blockers(document, item):
        return None
    return "start" if status == STATUS_QUEUED else "resume"


def _sort_key(policy: str, item: dict[str, Any]) -> tuple[Any, ...]:
    depth = int(item.get("depth") or 1)
    # Larger priority first; created-at keeps the order stable between reads.
    priority = -int(item.get("priority") or 0)
    created = str(item.get("created_at") or "")
    if policy == POLICY_DEPTH_FIRST:
        return (-depth, priority, created)
    if policy == POLICY_PRIORITY:
        return (priority, depth, created)
    return (depth, priority, created)


def dispatch(document: dict[str, Any],
             policy: str = "") -> list[dict[str, Any]]:
    """The ordered list of what to start, resume or decide about now.

    `start` and `resume` are cut to the free slots under `policy`. A stalled
    item is always reported: it holds no slot, and nothing else recovers it.
    """
    items = list((document.get("items") or {}).values())
    active = sum(1 for item in items if item.get("status") in ACTIVE_STATUSES)
    free = max(0, int(document.get("capacity") or DEFAULT_CAPACITY) - active)

    effective = policy or document.get("policy") or DEFAULT_POLICY
    if effective not in POLICIES:
        raise ItemError(f"unknown dispatch policy {effective!r}; pick one of"
                        f" {', '.join(POLICIES)}")

    stalled: list[dict[str, Any]] = []
    ready: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for item in items:
        action = runnable_action(document, item)
        if action is None:
            continue
        entry = {
            "action": action,
            "item": item["id"],
            "to": item["to"],
            "depth": item.get("depth", 1),
            "owner": item.get("owner", ""),
            "status": item["status"],
        }
        if action == "decide":
            entry["reason"] = item.get("failure") or "lease expired"
            stalled.append(entry)
        else:
            ready.append((_sort_key(effective, item), entry))

    ready.sort(key=lambda pair: pair[0])
    return stalled + [entry for _, entry in ready[:free]]


def init(run_id: str,
         *,
         capacity: int = 0,
         policy: str = "",
         max_depth: int = 0) -> dict[str, Any]:
    """Create the run's store, or change its capacity, policy or depth cap.

    Separate from `queue` so the policy can be set before any work exists and
    changed mid-run, e.g. breadth-first until one lead stands out.
    """
    utils.validate_run_id(run_id)
    if policy and policy not in POLICIES:
        raise ItemError(f"unknown dispatch policy {policy!r}; pick one of"
                        f" {', '.join(POLICIES)}")
    if capacity < 0:
        raise ItemError("capacity cannot be negative")
    if max_depth < 0:
        raise ItemError("max depth cannot be negative")
    with store_lock(run_id):
        document = load(run_id)
        if capacity:
            document["capacity"] = capacity
        if policy:
            document["policy"] = policy
        if max_depth:
            document["max_depth"] = max_depth
        return save(run_id, document)


def queue(
    run_id: str,
    *,
    to: str,
    request: Any,
    parent: str = "",
    after: list[str] | None = None,
    priority: int = 0,
    requested_by: str = "",
    max_depth: int = 0,
    package: str = "",
) -> dict[str, Any]:
    """Record a request for `to` and return the item, without blocking.

    `parent` is the item the request was made from; it sets the chain the
    cycle and depth checks run against, and names the requester. The main
    agent always spawns the worker.

    `package` is the build whose graph and contracts admit the request; with
    none, or one built before the graph was published, those checks skip.
    """
    utils.validate_run_id(run_id)
    to = to.strip()
    if not to:
        raise ItemError("a request needs an agent to go to")
    refs = []
    for ref in after or []:
        parse_ref(ref)
        refs.append(ref.strip())

    with store_lock(run_id):
        document = load(run_id)
        items = document.setdefault("items", {})

        depth = 1
        requester = ""
        if parent:
            parent_item = _require(document, parent)
            depth = int(parent_item.get("depth") or 1) + 1
            # The parent item says who is asking; a self-reported
            # `--requested-by` could claim an edge the requester lacks.
            requester = parent_item["to"]
            if to in _chain(document, parent):
                raise ItemError(
                    f"{to} is already on this item's chain ({parent}); a"
                    " cycle. Answer the request that is already in flight"
                    " instead of queueing another")
        graph, roots, graph_depth = (agent_contracts.delegation_graph(package)
                                     if package else ({}, None, 0))
        # The tightest cap wins: an argument can add a limit, never lift the
        # one stored for the run or published by the package.
        caps = [
            value for value in (max_depth, int(document.get("max_depth") or 0),
                                graph_depth) if value > 0
        ]
        if caps and depth > min(caps):
            raise ItemError(f"logical depth would be {depth}; the cap is"
                            f" {min(caps)}")
        # A requester outside the graph (a root skill, or a built-in) may
        # reach only the roots; `--requested-by` is self-reported, so unused.
        allowed = graph.get(requester, roots) if graph else None
        if allowed is not None and to in graph and to not in allowed:
            if requester in graph:
                declared = ", ".join(sorted(allowed)) or "nothing"
                raise ItemError(f"{requester} does not declare {to} in"
                                f" delegates: it may request {declared}")
            who = (f"{requester}, outside the package graph,"
                   if requester else "a request with no --parent")
            raise ItemError(f"no skill declares {to} in delegates, so {who}"
                            " cannot go to it")
        _check_contract(package, to, request, result=False)
        for ref in refs:
            # Raises for an unknown item: such a row could never run.
            ref_satisfied(document, ref)

        item_id = f"I{int(document.get('next_id') or 1)}"
        document["next_id"] = int(document.get("next_id") or 1) + 1
        now = _now()
        item = {
            "id": item_id,
            "to": to,
            "request": request,
            "status": STATUS_QUEUED,
            "owner": "",
            "parent": parent,
            "depth": depth,
            "priority": priority,
            "requested_by": requested_by,
            "blocked_on": refs,
            "parts": {},
            "response": None,
            "failure": "",
            "history": [],
            "lease_expires_at": "",
            "created_at": now,
            "updated_at": now,
        }
        items[item_id] = item
        save(run_id, document)
        _state_change(run_id, item, "queued", "")
        return item


def _check_contract(package: str, to: str, payload: Any, *,
                    result: bool) -> None:
    """Refuse a request (`accepts`) or answer (`produces`) `to` would reject."""
    problems = (agent_contracts.contract_problems(
        package, to, payload, result=result) if package else [])
    if problems:
        side = "answer from" if result else "request to"
        raise ItemError(f"{side} {to} violates its contract: " +
                        "; ".join(problems))


def _write_item(run_id: str,
                item_id: str,
                change: Callable[..., dict[str, str] | None],
                *,
                owner: str | None = None,
                closed_ok: bool = False) -> dict[str, Any]:
    """One verb's locked read-modify-write of an item, and its event.

    `change(document, item)` edits the item in place and returns the
    `_state_change` fields to record (`kind`, optionally `part`, `detail`), or
    None for no event. `owner`, when given, must hold the item; a closed item
    is refused unless `closed_ok`.
    """
    with store_lock(run_id):
        document = load(run_id)
        item = _require(document, item_id)
        if owner is not None:
            _owned(item, owner)
        old = item["status"]
        if old in CLOSED_STATUSES and not closed_ok:
            raise ItemError(f"{item_id} is {old}; it takes no more writes")
        fields = change(document, item)
        item["updated_at"] = _now()
        save(run_id, document)
        if fields is not None:
            _state_change(run_id, item, old_status=old, **fields)
        return item


def _renew(item: dict[str, Any], lease_seconds: float) -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    expires = now + datetime.timedelta(seconds=max(1.0, lease_seconds))
    item["lease_expires_at"] = expires.isoformat(timespec="seconds").replace(
        "+00:00", "Z")


def _require_slot(document: dict[str, Any], item: dict[str, Any]) -> None:
    """Refuse a transition into a claim when every slot is already held."""
    if item["status"] in ACTIVE_STATUSES:
        return
    capacity = int(document.get("capacity") or DEFAULT_CAPACITY)
    active = sum(1 for other in document["items"].values()
                 if other.get("status") in ACTIVE_STATUSES)
    if active >= capacity:
        raise ItemError(f"{item['id']} cannot be claimed: all {capacity}"
                        " slots are held")


def _revive(document: dict[str, Any], item: dict[str, Any]) -> bool:
    """Return a stalled item to its owner's claim; True when it was stalled.

    Called after `_owned`: a write from the owner proves the sweep wrong.
    """
    if item["status"] != STATUS_STALLED:
        return False
    _require_slot(document, item)
    item["status"] = STATUS_CLAIMED
    item["failure"] = ""
    return True


def _owned(item: dict[str, Any], owner: str) -> None:
    """Refuse a write from anyone but the item's owner, including no one."""
    if not owner:
        raise ItemError(f"a write to {item['id']} needs the owner that"
                        " claimed it")
    if item.get("owner") != owner:
        raise ItemError(f"{item['id']} is owned by"
                        f" {item.get('owner') or 'no one'}, not {owner}")


def claim(run_id: str,
          item_id: str,
          *,
          owner: str,
          lease_seconds: float = DEFAULT_LEASE_SECONDS) -> dict[str, Any]:
    """Take ownership of a queued or readied item and start its lease."""
    if not owner.strip():
        raise ItemError("a claim needs an owner")

    def change(document: dict[str, Any], item: dict[str,
                                                    Any]) -> dict[str, str]:
        holder = item.get("owner")
        if item["status"] == STATUS_CLAIMED and holder != owner:
            raise ItemError(f"{item_id} is already claimed by {holder}")
        if item["status"] == STATUS_STALLED and holder != owner:
            # Its owner may only be slow (`_revive`); giving it away would run
            # the work twice.
            raise ItemError(
                f"{item_id} is stalled and still held by {holder}; requeue it"
                " before another worker claims it")
        pending = blockers(document, item)
        if pending:
            raise ItemError(
                f"{item_id} still waits on {', '.join(pending)}; it is not"
                " runnable yet")
        _require_slot(document, item)
        item.update(status=STATUS_CLAIMED,
                    owner=owner,
                    blocked_on=[],
                    failure="")
        _renew(item, lease_seconds)
        return {"kind": "claimed"}

    return _write_item(run_id, item_id, change)


def progress(
    run_id: str,
    item_id: str,
    *,
    owner: str = "",
    step: str = "",
    artifact: str = "",
    part: str = "",
    value: Any = None,
    lease_seconds: float = DEFAULT_LEASE_SECONDS,
) -> dict[str, Any]:
    """Write mid-work state without closing the item, and renew its lease.

    A published part readies anything queued `--after <item>#<part>`. With no
    step, artifact or part this is a heartbeat: it only renews the lease.
    """

    def change(document: dict[str, Any],
               item: dict[str, Any]) -> dict[str, str] | None:
        revived = _revive(document, item)
        if not (step or artifact or part):
            if item["status"] != STATUS_CLAIMED:
                raise ItemError(
                    f"{item_id} is {item['status']}, not claimed; there is no"
                    " lease to renew")
            _renew(item, lease_seconds)
            return {"kind": "claimed"} if revived else None
        if part:
            item.setdefault("parts", {})[part] = value
        entry = {"step": step, "artifact": artifact, "part": part}
        item.setdefault("history", []).append({
            "at": _now(),
            **{
                key: text
                for key, text in entry.items() if text
            }
        })
        # `wait` cleared the lease on purpose; re-arming it would let a second
        # worker resume an item its owner is still writing to.
        if item["status"] != STATUS_WAITING:
            _renew(item, lease_seconds)
        return {"kind": "progress", "part": part, "detail": step or artifact}

    return _write_item(run_id, item_id, change, owner=owner)


def answer(run_id: str,
           item_id: str,
           *,
           response: Any,
           owner: str = "",
           package: str = "") -> dict[str, Any]:
    """Close the item with its final result, which outranks any partial."""

    def change(document: dict[str, Any], item: dict[str,
                                                    Any]) -> dict[str, str]:
        _check_contract(package, item["to"], response, result=True)
        item.update(status=STATUS_COMPLETED,
                    response=response,
                    blocked_on=[],
                    failure="",
                    lease_expires_at="")
        return {"kind": "completed"}

    return _write_item(run_id, item_id, change, owner=owner)


def fail(run_id: str,
         item_id: str,
         *,
         reason: str,
         owner: str = "") -> dict[str, Any]:
    """Close the item as failed, so its waiters wake to a verdict."""
    if not reason.strip():
        raise ItemError("a failure needs a reason")

    def change(document: dict[str, Any], item: dict[str,
                                                    Any]) -> dict[str, str]:
        item.update(status=STATUS_FAILED,
                    failure=reason,
                    blocked_on=[],
                    lease_expires_at="")
        return {"kind": "failed", "detail": reason}

    return _write_item(run_id, item_id, change, owner=owner)


def wait(run_id: str,
         item_id: str,
         *,
         on: list[str],
         owner: str = "") -> dict[str, Any]:
    """Release the item's slot until the results it names have landed.

    The worker stops afterwards and `dispatch` reports a `resume` once every
    reference is satisfied. Waiting on something already satisfied readies
    the item at once, so a worker is safe to wake more than once.
    """
    if not on:
        raise ItemError("a wait needs at least one item or part to wait on")
    refs = [ref.strip() for ref in on]

    def change(document: dict[str, Any], item: dict[str,
                                                    Any]) -> dict[str, str]:
        for ref in refs:
            target, _ = parse_ref(ref)
            if target == item_id:
                raise ItemError(f"{item_id} cannot wait on itself")
            if not ref_satisfied(document, ref) and _waits_on(
                    document, target, item_id):
                raise ItemError(
                    f"{item_id} cannot wait on {ref}: {target} already waits"
                    f" on {item_id}, and neither would ever run")
        item.update(status=STATUS_WAITING,
                    blocked_on=refs,
                    failure="",
                    lease_expires_at="")
        return {"kind": "waiting", "detail": ", ".join(refs)}

    return _write_item(run_id, item_id, change, owner=owner)


def requeue(run_id: str, item_id: str) -> dict[str, Any]:
    """Put a stalled or failed item back in the queue for a fresh worker."""

    def change(document: dict[str, Any], item: dict[str,
                                                    Any]) -> dict[str, str]:
        if item["status"] not in (STATUS_STALLED, STATUS_FAILED):
            raise ItemError(f"{item_id} is {item['status']}; only a stalled"
                            " or failed item can be requeued")
        # The discarded attempt's parts would wake waiters on stale output.
        item.update(status=STATUS_QUEUED,
                    owner="",
                    failure="",
                    parts={},
                    lease_expires_at="")
        return {"kind": "queued"}

    return _write_item(run_id, item_id, change, closed_ok=True)


def expire_leases(run_id: str) -> list[dict[str, Any]]:
    """Turn every claim whose lease ran out into a `stalled` item.

    The lease is the only evidence a worker died. The `dispatch` and `watch`
    commands sweep first; the `dispatch` function stays a pure read.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    stalled: list[dict[str, Any]] = []
    with store_lock(run_id):
        document = load(run_id)
        for item in (document.get("items") or {}).values():
            if item.get("status") != STATUS_CLAIMED:
                continue
            expires = utils.parse_iso_utc(item.get("lease_expires_at"))
            if expires is None or expires > now:
                continue
            item["status"] = STATUS_STALLED
            item["failure"] = (
                f"lease expired at {item.get('lease_expires_at')} with no"
                f" write from {item.get('owner') or 'its worker'}")
            item["updated_at"] = _now()
            stalled.append(item)
        if stalled:
            save(run_id, document)
            for item in stalled:
                _state_change(run_id,
                              item,
                              "stalled",
                              STATUS_CLAIMED,
                              detail=item["failure"])
    return stalled


def open_items(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Every item that is not closed, in id order."""
    return sorted(
        (item for item in (document.get("items") or {}).values()
         if item.get("status") in OPEN_STATUSES),
        key=lambda item: int(item["id"][1:]),
    )


def summary(document: dict[str, Any]) -> dict[str, Any]:
    """A reader's view of the store: counts, capacity, and one row per item."""
    items = sorted((document.get("items") or {}).values(),
                   key=lambda item: int(item["id"][1:]))
    counts: dict[str, int] = {}
    for item in items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {
        "run_id":
        document.get("run_id", ""),
        "capacity":
        document.get("capacity", DEFAULT_CAPACITY),
        "policy":
        document.get("policy", DEFAULT_POLICY),
        "counts":
        counts,
        "items": [{
            "id": item["id"],
            "to": item["to"],
            "status": item["status"],
            "owner": item.get("owner", ""),
            "parent": item.get("parent", ""),
            "depth": item.get("depth", 1),
            "blocked_on": item.get("blocked_on") or [],
            "parts": sorted((item.get("parts") or {}).keys()),
            "steps": len(item.get("history") or []),
        } for item in items],
    }


def watch(run_id: str,
          *,
          timeout_seconds: float = DEFAULT_WATCH_TIMEOUT_SECONDS,
          poll_seconds: float = DEFAULT_WATCH_POLL_SECONDS,
          policy: str = "") -> dict[str, Any]:
    """The doorbell: block until the store has work to dispatch.

    No harness wakes an idle main agent when another process writes this
    store, so the main agent keeps a watcher in the background and the
    watcher's exit is the wake. `reason` is `runnable`, `idle` (no open items,
    so nothing will ever become runnable) or `timeout` (re-arm).

    Expired leases are swept once the earliest is due, so a dead worker rings
    it too. The document is re-read when its mtime or the event log's size
    moves: mtime alone ticks too coarsely to see every write.
    """
    utils.validate_run_id(run_id)
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    interval = max(0.05, poll_seconds)
    path, log = document_path(run_id), events_path(run_id)
    # Not a stamp any store can have: the first tick must read.
    last_seen: tuple[int | None, int | None] | str = "unread"

    next_expiry: datetime.datetime | None = None

    while True:
        stamp = (path.stat().st_mtime_ns if path.is_file() else None,
                 log.stat().st_size if log.is_file() else None)
        if stamp != last_seen:
            last_seen = stamp
            document = load(run_id)
            next_expiry = _earliest_lease(document)
            plan = dispatch(document, policy=policy)
            if plan:
                return {"reason": "runnable", "runnable": plan}
            if not open_items(document):
                return {"reason": "idle", "runnable": []}
        # Sweep only once a lease is due, then re-read whatever it found: a
        # heartbeat logs no event, so the stamp can miss a renewed lease.
        if next_expiry is not None and next_expiry <= datetime.datetime.now(
                datetime.timezone.utc):
            expire_leases(run_id)
            last_seen = "unread"
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"reason": "timeout", "runnable": []}
        time.sleep(min(interval, remaining))


def resolve_package(run_id: str) -> str:
    """The package this run is admitted against, from its run document.

    Deliberately not a flag: a caller that could name its own package could
    name one whose graph permits it.
    """
    try:
        document = jsonio.load_json_or_empty(
            paths.get_run_root(run_id) / "run.json")
    except (utils.TpuAgentSuiteError, OSError):
        return ""
    package = str(document.get("package") or "")
    # `start-run`'s placeholder for a run with no package.
    return "" if package == "no-package" else package


def _earliest_lease(document: dict[str, Any]) -> datetime.datetime | None:
    """When the first claim in `document` expires, or None if none is held."""
    leases = [
        utils.parse_iso_utc(item.get("lease_expires_at"))
        for item in (document.get("items") or {}).values()
        if item.get("status") == STATUS_CLAIMED
    ]
    return min((lease for lease in leases if lease), default=None)


def resolve_run_id() -> str:
    """The workspace's active run, from `.tpu-agent-suite/current_run_id`.

    There is no override: every agent in a session addresses the same store.
    The run must exist, or a worker and the orchestrator could each address
    a different store and read the other as idle.
    """
    run_id = paths.get_active_run_id()
    if not run_id:
        raise ItemError(
            "no active run: run this inside a workspace whose session recorded"
            " one in .tpu-agent-suite/current_run_id")
    paths.get_started_run_root(run_id)
    return run_id


def _json_arg(args: argparse.Namespace, name: str, *, required: bool) -> Any:
    """`--<name>-file`, else inline `--<name>` JSON; None if optional and unset."""
    path = getattr(args, f"{name}_file", None)
    if path:
        return jsonio.load_json(path)
    text = getattr(args, name, None) or ""
    if not text.strip():
        if required:
            raise ItemError(
                f"a {name} is required: pass --{name} or --{name}-file")
        return None
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ItemError(f"--{name} is not JSON: {exc}") from exc


def register_items_parser(sub: Any) -> None:
    """Attach the `items` command and its verbs to the runtime's subparsers."""
    parser = sub.add_parser(
        "items", help="Queue, claim and answer delegated work items")
    verbs = parser.add_subparsers(dest="items_action", required=True)

    def item_verb(name: str,
                  help_text: str,
                  *,
                  owner: bool = True,
                  lease: bool = False) -> argparse.ArgumentParser:
        verb = verbs.add_parser(name, help=help_text)
        verb.add_argument("item", help="Item id, e.g. I3")
        if owner:
            verb.add_argument("--owner", required=True, help="Worker writing")
        if lease:
            verb.add_argument("--lease-seconds",
                              type=float,
                              default=DEFAULT_LEASE_SECONDS,
                              help="How long the claim stays valid without a"
                              " write")
        return verb

    p_init = verbs.add_parser("init",
                              help="Create the store, or set its capacity"
                              " and dispatch policy")
    p_init.add_argument("--capacity",
                        type=int,
                        default=0,
                        help="Workers that may hold a claim at once; set at"
                        " or below the harness's live-subagent limit")
    p_init.add_argument("--policy",
                        default="",
                        choices=["", *POLICIES],
                        help="How dispatch orders runnable items")
    p_init.add_argument("--max-depth",
                        type=int,
                        default=0,
                        help="Logical delegation depth every queue is held"
                        " to (0 = no cap)")

    p_queue = verbs.add_parser("queue", help="Record a request for an agent")
    p_queue.add_argument("--to", required=True, help="Agent to request from")
    p_queue.add_argument("--request", default="", help="Request as JSON")
    p_queue.add_argument("--request-file",
                         type=Path,
                         default=None,
                         help="File holding the request JSON")
    p_queue.add_argument("--parent",
                         default="",
                         help="Item this request is made from; sets the"
                         " chain the cycle and depth checks run against")
    p_queue.add_argument("--after",
                         action="append",
                         default=[],
                         metavar="REF",
                         help="Wait for this result first: I3, or I3#gaps[0]"
                         " for one part of it. Repeatable")
    p_queue.add_argument("--priority",
                         type=int,
                         default=0,
                         help="Higher goes first under the priority policy")
    p_queue.add_argument("--requested-by",
                         default="",
                         help="Agent making the request; recorded, not"
                         " used for admission")
    p_queue.add_argument("--max-depth",
                         type=int,
                         default=0,
                         help="Refuse the request past this logical depth"
                         " (0 = no cap here)")

    item_verb("claim", "Take ownership of an item", lease=True)

    p_progress = item_verb(
        "progress",
        "Write mid-work state without closing the item; with no --step,"
        " --artifact or --part, just renew the lease",
        lease=True)
    p_progress.add_argument("--step", default="", help="Step reached")
    p_progress.add_argument("--artifact", default="", help="Artifact path")
    p_progress.add_argument("--part",
                            default="",
                            help="Name of the result slot this fills, e.g."
                            " gaps[0]; readies anything queued --after it")
    p_progress.add_argument("--value",
                            default=None,
                            help="The part's value, as JSON")

    p_answer = item_verb("answer", "Close the item with its final result")
    p_answer.add_argument("--response", default="", help="Result as JSON")
    p_answer.add_argument("--response-file",
                          type=Path,
                          default=None,
                          help="File holding the result JSON")

    p_fail = item_verb("fail", "Close the item as failed")
    p_fail.add_argument("--reason", required=True, help="Why it failed")

    p_wait = item_verb("wait", "Release the item's slot until a result lands")
    p_wait.add_argument("--on",
                        action="append",
                        required=True,
                        metavar="REF",
                        help="Result to wait for: I3, or I3#gaps[0]."
                        " Repeatable")

    item_verb("requeue",
              "Put a stalled or failed item back in the queue",
              owner=False)

    p_dispatch = verbs.add_parser(
        "dispatch",
        help="Stall expired claims, then list what to start, resume or"
        " decide about now")
    p_dispatch.add_argument("--policy",
                            default="",
                            choices=["", *POLICIES],
                            help="Order this list differently, just once")

    # `watch`, not `wait`: `wait` is the worker's release, and the agents that
    # must not confuse the two would read one name as the same thing.
    p_watch = verbs.add_parser(
        "watch", help="The doorbell: block until there is work to dispatch")
    p_watch.add_argument("--timeout-seconds",
                         type=float,
                         default=DEFAULT_WATCH_TIMEOUT_SECONDS,
                         help="Return `timeout` after this long so the"
                         " caller can re-arm. Keep it under the harness's"
                         " own shell cap")
    p_watch.add_argument("--poll-seconds",
                         type=float,
                         default=DEFAULT_WATCH_POLL_SECONDS,
                         help="How often to look")
    p_watch.add_argument("--policy",
                         default="",
                         choices=["", *POLICIES],
                         help="Order the returned list differently")

    p_status = verbs.add_parser(
        "status", help="Counts and one row per item, or one item in full")
    p_status.add_argument("item",
                          nargs="?",
                          default="",
                          help="Item id to show in full")

    p_events = verbs.add_parser("events", help="The run's state-change log")
    p_events.add_argument("--limit",
                          type=int,
                          default=0,
                          help="Keep only the newest N (0 = all)")


def dispatch_items_command(args: Any) -> None:
    """Run one `items` verb. Printing is the command's only output."""
    run_id = resolve_run_id()
    package = resolve_package(run_id)
    action = args.items_action
    if action == "init":
        out: Any = init(run_id,
                        capacity=args.capacity,
                        policy=args.policy,
                        max_depth=args.max_depth)
    elif action == "queue":
        out = queue(run_id,
                    to=args.to,
                    request=_json_arg(args, "request", required=True),
                    parent=args.parent,
                    after=args.after,
                    priority=args.priority,
                    requested_by=args.requested_by,
                    max_depth=args.max_depth,
                    package=package)
    elif action == "claim":
        out = claim(run_id,
                    args.item,
                    owner=args.owner,
                    lease_seconds=args.lease_seconds)
    elif action == "progress":
        out = progress(run_id,
                       args.item,
                       owner=args.owner,
                       step=args.step,
                       artifact=args.artifact,
                       part=args.part,
                       value=_json_arg(args, "value", required=False),
                       lease_seconds=args.lease_seconds)
    elif action == "answer":
        out = answer(run_id,
                     args.item,
                     response=_json_arg(args, "response", required=True),
                     owner=args.owner,
                     package=package)
    elif action == "fail":
        out = fail(run_id, args.item, reason=args.reason, owner=args.owner)
    elif action == "wait":
        out = wait(run_id, args.item, on=args.on, owner=args.owner)
    elif action == "requeue":
        out = requeue(run_id, args.item)
    elif action == "dispatch":
        expire_leases(run_id)
        out = dispatch(load(run_id), policy=args.policy)
    elif action == "watch":
        out = watch(run_id,
                    timeout_seconds=args.timeout_seconds,
                    poll_seconds=args.poll_seconds,
                    policy=args.policy)
    elif action == "status":
        document = load(run_id)
        out = _require(document, args.item) if args.item else summary(document)
    elif action == "events":
        out = read_events(run_id, limit=args.limit)
    else:
        raise ItemError(f"unknown items verb {action!r}")
    print(jsonio.format_json(out))
