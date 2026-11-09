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
"""The session's TPU slots: which servers may run at once, and who holds each.

A slot is room for one vLLM server: one TPU VM, or the chips one server's Job
takes on a GKE node pool. An agent claims a slot before it starts a server
and releases it when the server ends, so two agents of a session never start
servers on the same chips. The session lists its slots in
`<suite home>/runs/<run_id>/parallel.json`, the file yeet-vllm-bench books
from, so two agents of one session never start servers on the same chips:

    {"parallel": 3,
     "resources": {
       "v6e-gke-1": {"profile": "v6e-gke", "pool": "pool-3f2a...",
                     "hardware": "tpu-v6e-slice|2x4|8|<image>",
                     "job": "sweep:gemma:s2", ...},
       "v6e-gke-2": {"profile": "v6e-gke", ..., "job": null, ...}},
     "pools": {"pool-3f2a...": 2}}

Each slot names the compute profile that reaches it and its `hardware`:
accelerator, topology, chips per job and image, joined by `|`. Servers whose
numbers are compared must share that key, not a profile: two profiles on the
same kind of TPU and image are one kind of server.

A `gce_tpu_vm` profile is one VM, so one slot. A `gke_job` profile gets one
slot per server its node pool can still start: the pool's `available_chips`
from `compute inspect`, divided by the chips one of its Jobs takes. That is
the pool's capacity, kept under `pools` and shared by every profile on the
same nodes, whatever its image: each such profile lists the pool's capacity
in slots, and a claim is refused once the pool's held slots reach it. When
the free chips cannot be read, or the pool has no nodes yet (an autoscaled
pool scaled to zero), the profile gets `--gke-slots` slots and no pool.
Free chips spread over several nodes can count a server more than the nodes
can place; the Job then waits for one.

`parallel` caps how many slots are held at once, across every agent sharing
the file. `discover` sets it to the session's capacity when it writes the
file, and leaves it as it is after that unless `--parallel` says otherwise.

The first server of a set should go where the most servers like it can
follow: `claim --most-available` takes a slot of the hardware key with the
most free slots, and `claim --hardware KEY` a slot of that key only.

The file is read under a shared lock and changed only under an exclusive one,
and replaced whole, so concurrent claims from threads or processes neither
double-book a slot nor leave a half-written file.

    tpu_slots.py discover --run-id R --runtime "<runtime>" --package P \\
        [--gke-slots N] [--parallel N]
    tpu_slots.py claim    --run-id R --holder H \\
        [--resource NAME | --profile P | --hardware KEY | --most-available]
    tpu_slots.py set-job  --run-id R --resource NAME --holder H --job-id J
    tpu_slots.py release  --run-id R --resource NAME [--holder H]
    tpu_slots.py show     --run-id R

Every command prints one JSON object. Exit codes: 0 done, 2 the session lacks
what is asked for (no parallel.json, an unknown slot), 3 refused (every slot
held, or `parallel` reached), 1 anything else.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterator, NamedTuple

Path = pathlib.Path

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NEEDS_INPUT = 2
EXIT_REFUSED = 3

# Backends with room to book: a TPU VM is one server, a gke_job cluster as
# many as its free chips hold. `local` is this host, which compute's own lock
# already serializes; a `gke_pod` is one pod, already running, that no other
# agent starts servers on. A slot on either of them would book nothing.
SLOT_BACKENDS = ("gce_tpu_vm", "gke_job")
CLAIM_FIELDS = ("job", "job_id", "run_id", "since")


class NeedsInput(Exception):
    """The session lacks what the command needs; the user decides it."""


class Refused(Exception):
    """Nothing free to claim right now."""


def state_path(run_id: str | None, explicit: str | None = None) -> Path:
    """parallel.json for this session: `--state`, $VMBENCH_STATE, or the run's."""
    if explicit or os.environ.get("VMBENCH_STATE"):
        return Path(explicit or os.environ["VMBENCH_STATE"])
    if not run_id:
        raise NeedsInput("parallel.json belongs to a run: pass --run-id")
    home = os.environ.get("TPU_AGENT_SUITE_HOME", "").strip()
    root = Path(home) if home else Path.home() / ".tpu-agent-suite"
    return root / "runs" / run_id / "parallel.json"


def write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


@contextlib.contextmanager
def locked_state(path: Path, exclusive: bool = True) -> Iterator[dict]:
    """parallel.json under its lock: shared to read, exclusive to change.

    Yields {"state": dict or None}. On an exclusive hold, a changed state is
    written back whole; leaving it unchanged writes nothing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{path}.lock", "a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            state = (json.loads(path.read_text(
                encoding="utf-8")) if path.exists() else None)
            holder = {"state": state}
            yield holder
            if exclusive and holder["state"] is not None and holder[
                    "state"] != state:
                write_atomic(path,
                             json.dumps(holder["state"], indent=2) + "\n")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _copy(state: dict | None) -> dict:
    return json.loads(json.dumps(state or {}))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Hardware(NamedTuple):
    """A kind of server: what its numbers depend on besides the model."""
    accelerator: str
    topology: str
    chips: str
    image: str

    @property
    def key(self) -> str:
        """The `accelerator|topology|chips|image` a slot records."""
        return "|".join(self)

    @classmethod
    def parse(cls, key: str | None) -> Hardware | None:
        parts = (key or "").split("|", 3)
        return cls(*parts) if len(parts) == 4 else None


def chips_per_job(params: dict[str, Any]) -> str:
    """A gke_job profile's chips per Job: `tpu_chips`, else its topology's."""
    if params.get("tpu_chips"):
        return str(params["tpu_chips"])
    try:
        return str(
            math.prod(
                int(d) for d in str(params.get("tpu_topology")).split("x")))
    except ValueError:
        return "auto"


def hardware_key(profile: dict[str, Any],
                 seen: dict[str, Any] | None = None) -> str | None:
    """`accelerator|topology|chips|image` of a profile's servers, or None.

    A gke_job profile names all four in its params, which hold even when its
    pool has no nodes. Another backend's accelerator and topology come from
    `compute inspect` (`seen`).
    """
    params = (profile or {}).get("params") or {}
    if (profile or {}).get("backend") == "gke_job":
        parts = (params.get("tpu_accelerator"), params.get("tpu_topology"),
                 chips_per_job(params), params.get("image"))
    else:
        seen = seen or {}
        parts = (seen.get("accelerator"), seen.get("topology"),
                 str(seen.get("chips") or "-"), params.get("image") or "-")
    if not all(parts) or {"none", "unknown"} & {str(p) for p in parts[:2]}:
        return None
    return Hardware(*(str(p) for p in parts)).key


def pool_id(nodes: tuple[str, ...]) -> str:
    return "pool-" + hashlib.sha256(",".join(nodes).encode()).hexdigest()[:12]


def plan_slots(
        profiles: dict[str, Any],
        inspect: Callable[[str], dict[str, Any] | None], gke_slots: int
) -> tuple[dict[str, dict], dict[str, dict], dict[str, int]]:
    """The slots the profiles stand for, by name; how each profile was sized;
    and the capacity of each node pool they share."""
    slots: dict[str, dict] = {}
    sized: dict[str, dict] = {}
    pools: dict[str, int] = {}
    for name, profile in sorted(profiles.items()):
        backend = (profile or {}).get("backend")
        if backend not in SLOT_BACKENDS:
            continue
        seen = inspect(name) or {}
        key = hardware_key(profile, seen)
        if backend == "gce_tpu_vm":
            slots[name] = {"profile": name, "hardware": key, "pool": None}
            sized[name] = {"hardware": key, "slots": 1, "by": "one VM"}
            continue
        details = seen.get("details") or {}
        params = (profile or {}).get("params") or {}
        job_chips = int(
            details.get("job_chips") or params.get("tpu_chips") or 0)
        available = seen.get("available_chips")
        nodes = tuple(sorted(details.get("nodes") or ()))
        pool = None
        if available is None or not nodes or not job_chips:
            count, by = gke_slots, "--gke-slots: free chips unknown"
        else:
            pool = pool_id(nodes)
            count = int(available) // job_chips
            if pool in pools:
                # The same nodes as an earlier profile: the same chips.
                count = pools[pool]
                by = f"shares its nodes' {count} server(s) with other profiles"
            else:
                pools[pool] = count
                by = f"{available} free chips / {job_chips} per server"
        for i in range(1, count + 1):
            slots[f"{name}-{i}"] = {
                "profile": name,
                "hardware": key,
                "pool": pool
            }
        sized[name] = {"hardware": key, "slots": count, "by": by}
    return slots, sized, pools


def capacity(state: dict[str, Any]) -> int:
    """Servers the session can run at once: each pool once, and every slot
    outside a pool."""
    resources = (state.get("resources") or {}).values()
    return (sum((state.get("pools") or {}).values()) +
            sum(1 for e in resources if not e.get("pool")))


def pool_room(state: dict[str, Any]) -> dict[str, int]:
    """Servers each node pool can still start: its capacity less its held
    slots."""
    room = dict(state.get("pools") or {})
    for entry in (state.get("resources") or {}).values():
        if entry.get("job") and entry.get("pool") in room:
            room[entry["pool"]] -= 1
    return room


def free_slots(
        state: dict[str, Any],
        fits: Callable[[dict[str, Any]], bool] = lambda _: True) -> list[str]:
    """Free slots that fit, leaving out those of a pool already full."""
    left = pool_room(state)
    return [
        n for n, e in sorted((state.get("resources") or {}).items())
        if not e.get("job") and fits(e) and (
            not e.get("pool") or left.get(e["pool"], 0) > 0)
    ]


def room(state: dict[str, Any], names: list[str]) -> int:
    """How many of `names` could be held at once, pool caps counted."""
    resources = state.get("resources") or {}
    left = pool_room(state)
    per_pool: dict[str, int] = {}
    for n in names:
        pool = resources[n].get("pool") or f"slot:{n}"
        per_pool[pool] = per_pool.get(pool, 0) + 1
    # A slot outside a pool is room for one server on its own.
    return sum(count if pool.
               startswith("slot:") else min(count, max(left.get(pool, 0), 0))
               for pool, count in per_pool.items())


def summarize(state: dict[str, Any]) -> dict[str, dict]:
    """Slots and the servers that could still start, per hardware key."""
    out: dict[str, dict] = {}
    by_key: dict[str, list[str]] = {}
    for name, entry in (state.get("resources") or {}).items():
        key = entry.get("hardware") or "unknown"
        row = out.setdefault(key, {"slots": 0, "free": 0, "profiles": []})
        row["slots"] += 1
        if entry["profile"] not in row["profiles"]:
            row["profiles"].append(entry["profile"])
        by_key.setdefault(key, []).append(name)
    free = set(free_slots(state))
    for key, names in by_key.items():
        out[key]["free"] = room(state, [n for n in names if n in free])
    return out


def discover(path: Path,
             profiles: dict[str, Any],
             gke_slots: int,
             inspect: Callable[[str], dict[str, Any] | None] = lambda _: None,
             parallel: int | None = None) -> dict:
    """Add the slots the profiles can serve now; never remove one.

    A profile's slots already listed keep their claims, and take its current
    hardware key and pool; each pool takes the capacity measured now.
    `parallel` is set to the session's capacity when the file is new, and
    otherwise kept unless `parallel` is given. Returns the file, the slots it
    added, how each profile was sized, the free servers per hardware key, and
    the questions only the user can answer (no TPU profile at all).
    """
    found, sized, pools = plan_slots(profiles, inspect, gke_slots)
    questions = []
    if not sized:
        questions.append(
            "The session has no gce_tpu_vm or gke_job compute profile to run "
            "servers on: which TPU should the benchmark use?")
    with locked_state(path) as held:
        new = held["state"] is None
        state = _copy(held["state"]) or {"parallel": 1, "resources": {}}
        resources = state.setdefault("resources", {})
        added = [name for name in found if name not in resources]
        for name in added:
            resources[name] = {
                **found[name],
                **{
                    k: None
                    for k in CLAIM_FIELDS
                }
            }
        by_profile = {f["profile"]: f for f in found.values()}
        for entry in resources.values():
            if entry.get("profile") in sized:
                entry["hardware"] = sized[entry["profile"]]["hardware"]
                entry["pool"] = (by_profile.get(entry["profile"])
                                 or {}).get("pool")
        state["pools"] = {**(state.get("pools") or {}), **pools}
        if parallel is not None:
            state["parallel"] = parallel
        elif new:
            state["parallel"] = max(capacity(state), 1)
        held["state"] = state
    return {
        "status": "needs_input" if questions else "ok",
        "path": str(path),
        "added": added,
        "sized": sized,
        "parallel": state["parallel"],
        "capacity": capacity(state),
        "hardware": summarize(state),
        "session": state,
        "questions": questions,
    }


def claim(path: Path,
          holder: str,
          resource: str | None = None,
          run_id: str | None = None,
          profile: str | None = None,
          hardware: str | None = None,
          most_available: bool = False) -> dict:
    """Take `resource`, or a free slot, for `holder`.

    `profile` and `hardware` limit which slots fit; a slot whose node pool is
    full never does. `most_available` takes a slot of the hardware key with
    room for the most servers, on the profile with the most of that room, so
    the servers that follow it find room of the same kind. A slot `holder`
    already has is returned again rather than refused, so a retried claim is
    harmless.
    """

    def fits(entry: dict[str, Any]) -> bool:
        return (profile in (None, entry.get("profile"))
                and hardware in (None, entry.get("hardware")))

    with locked_state(path) as held:
        if held["state"] is None:
            raise NeedsInput(f"no slots at {path}: run `tpu_slots.py "
                             "discover` first")
        state = _copy(held["state"])
        resources = state.get("resources") or {}
        mine = [
            n for n, e in sorted(resources.items())
            if e.get("job") == holder and fits(e)
        ]
        wanted = " ".join(f"{k} {v!r}" for k, v in (("profile", profile),
                                                    ("hardware", hardware))
                          if v)
        free = free_slots(state, fits)
        if resource is None and mine:
            name = mine[0]
        elif resource is not None:
            if resource not in resources:
                known = ", ".join(sorted(resources)) or "none"
                raise NeedsInput(f"{resource!r} is not a slot in {path} "
                                 f"(known: {known})")
            if not fits(resources[resource]):
                raise Refused(f"{resource} is not on {wanted}")
            if (not resources[resource].get("job") and resource not in free):
                raise Refused(f"{resource}'s node pool is full")
            name = resource
        else:
            if most_available:
                by_key: dict[str, list[str]] = {}
                for n in free:
                    if resources[n].get("hardware"):
                        by_key.setdefault(resources[n]["hardware"],
                                          []).append(n)
                if by_key:
                    key = min(by_key,
                              key=lambda k: (-room(state, by_key[k]), k))
                    by_profile: dict[str, list[str]] = {}
                    for n in by_key[key]:
                        by_profile.setdefault(resources[n]["profile"],
                                              []).append(n)
                    best = min(by_profile,
                               key=lambda p: (-room(state, by_profile[p]), p))
                    free = by_profile[best]
                else:
                    free = []
            if not free:
                on = f" on {wanted}" if wanted else ""
                raise Refused(f"no slot{on} in {path} is free")
            name = free[0]
        entry = resources[name]
        if entry.get("job") not in (None, holder):
            raise Refused(f"{name} is held by {entry['job']} since "
                          f"{entry.get('since')}")
        busy = sum(1 for e in resources.values() if e.get("job"))
        if entry.get("job") is None and busy >= int(state.get("parallel", 1)):
            raise Refused(f"{busy} slots already held; parallel is "
                          f"{state.get('parallel')}")
        if entry.get("job") is None:
            entry.update(job=holder, job_id=None, run_id=run_id, since=_now())
        held["state"] = state
        key = entry.get("hardware")
        return {
            "status":
            "ok",
            "resource":
            name,
            "profile":
            entry["profile"],
            "hardware":
            key,
            # Servers of the same kind that could still start beside it.
            "free_like_this":
            room(state, free_slots(state, lambda e: e.get("hardware") == key))
            if key else 0,
        }


def set_job(path: Path, resource: str, holder: str, job_id: str) -> dict:
    """Record the compute job that `holder` runs on its slot."""
    with locked_state(path) as held:
        state = _copy(held["state"])
        entry = (state.get("resources") or {}).get(resource)
        if not entry or entry.get("job") != holder:
            raise Refused(f"{resource} is not held by {holder}")
        entry["job_id"] = job_id
        held["state"] = state
    return {"status": "ok", "resource": resource, "job_id": job_id}


def release(path: Path, resource: str, holder: str | None = None) -> dict:
    """Free `resource`; with `holder`, only while that holder still has it."""
    with locked_state(path) as held:
        state = _copy(held["state"])
        entry = (state.get("resources") or {}).get(resource)
        released = bool(entry and entry.get("job")
                        and holder in (None, entry.get("job")))
        if released:
            entry.update({k: None for k in CLAIM_FIELDS})
            held["state"] = state
    return {"status": "ok", "resource": resource, "released": released}


def show(path: Path) -> dict:
    with locked_state(path, exclusive=False) as held:
        state = held["state"]
    if state is None:
        raise NeedsInput(f"no slots at {path}: run `tpu_slots.py discover`")
    return {"status": "ok", "path": str(path), "session": state}


def read_config(runtime: str, package: str | None,
                run_id: str | None) -> dict[str, Any]:
    """`<runtime> config show`: the profiles and the run's active one."""
    argv = shlex.split(runtime) + ["config", "show", "--json"]
    if package:
        argv += ["--package", package]
    if run_id:
        argv += ["--run-id", run_id]
    try:
        proc = subprocess.run(argv,
                              capture_output=True,
                              text=True,
                              timeout=120,
                              check=False)
        return json.loads(proc.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"`{' '.join(argv)}` printed no config: "
                           f"{exc}") from exc


def read_profiles(runtime: str, package: str | None,
                  run_id: str | None) -> dict[str, Any]:
    """The compute profiles from `<runtime> config show`."""
    return read_config(runtime, package, run_id).get("profiles") or {}


def read_inspect(runtime: str, package: str | None,
                 run_id: str | None) -> Callable[[str], dict | None]:
    """`<runtime> compute inspect --profile P`, per profile; None if it fails."""

    def inspect(profile: str) -> dict[str, Any] | None:
        argv = shlex.split(runtime) + [
            "compute", "inspect", "--json", "--profile", profile
        ]
        if package:
            argv += ["--package", package]
        if run_id:
            argv += ["--run-id", run_id]
        try:
            proc = subprocess.run(argv,
                                  capture_output=True,
                                  text=True,
                                  timeout=300,
                                  check=False)
            return json.loads(proc.stdout) if proc.returncode == 0 else None
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None

    return inspect


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-id", default=None)
    common.add_argument("--state",
                        default=None,
                        help="parallel.json to use instead of the run's")
    d = sub.add_parser("discover", parents=[common])
    d.add_argument("--runtime",
                   required=True,
                   help="the command that invokes the runtime")
    d.add_argument("--package", default=None)
    d.add_argument("--gke-slots",
                   type=int,
                   default=1,
                   help="servers for a gke_job profile whose free chips "
                   "cannot be read")
    d.add_argument("--parallel",
                   type=int,
                   default=None,
                   help="cap on slots held at once; default: the session's "
                   "capacity when the file is new, else as it is")
    c = sub.add_parser("claim", parents=[common])
    c.add_argument("--holder", required=True)
    which = c.add_mutually_exclusive_group()
    which.add_argument("--resource", default=None)
    which.add_argument("--most-available",
                       action="store_true",
                       help="a slot of the hardware key with the most free")
    c.add_argument("--profile",
                   default=None,
                   help="only a slot on this compute profile")
    c.add_argument("--hardware",
                   default=None,
                   help="only a slot of this accelerator|topology|image key")
    j = sub.add_parser("set-job", parents=[common])
    j.add_argument("--resource", required=True)
    j.add_argument("--holder", required=True)
    j.add_argument("--job-id", required=True)
    r = sub.add_parser("release", parents=[common])
    r.add_argument("--resource", required=True)
    r.add_argument("--holder", default=None)
    sub.add_parser("show", parents=[common])
    args = p.parse_args(argv)
    if getattr(args, "gke_slots", 1) < 1:
        p.error("--gke-slots must be at least 1")
    if (getattr(args, "parallel", None) or 1) < 1:
        p.error("--parallel must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        path = state_path(args.run_id, args.state)
        if args.command == "discover":
            result = discover(
                path, read_profiles(args.runtime, args.package,
                                    args.run_id), args.gke_slots,
                read_inspect(args.runtime, args.package, args.run_id),
                args.parallel)
        elif args.command == "claim":
            result = claim(path, args.holder, args.resource, args.run_id,
                           args.profile, args.hardware, args.most_available)
        elif args.command == "set-job":
            result = set_job(path, args.resource, args.holder, args.job_id)
        elif args.command == "release":
            result = release(path, args.resource, args.holder)
        else:
            result = show(path)
    except NeedsInput as exc:
        print(json.dumps({"status": "needs_input", "error": str(exc)}))
        return EXIT_NEEDS_INPUT
    except Refused as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}))
        return EXIT_REFUSED
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return EXIT_ERROR
    print(json.dumps(result))
    return EXIT_NEEDS_INPUT if result["status"] == "needs_input" else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
