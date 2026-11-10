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
       "v6e-gke-1": {"profile": "v6e-gke",
                     "hardware": "tpu-v6e-slice|2x4|8|<image>",
                     "job": "sweep:gemma:s2", ...},
       "v6e-gke-2": {"profile": "v6e-gke", ..., "job": null, ...}}}

Each slot names the compute profile that reaches it and its `hardware`:
accelerator, topology, chips per job and image, joined by `|`. Servers whose
numbers are compared must share that key, not a profile: two profiles on the
same kind of TPU and image are one kind of server.

A `gce_tpu_vm` profile is one VM, so one slot. A `gke_job` profile has the
slots it declares in its `slots` param, one if it names none.

`parallel` is the session's slot count, kept for yeet-vllm-bench, which
reads it.

The slots are set once, at the start of the session: the first `claim` given
`--runtime` (and `--package`) writes the file from the compute profiles, and
an existing file is left as it is.

The first server of a set should go where the most servers like it can
follow: `claim --most-available` takes a slot of the hardware key with the
most free slots, and `claim --hardware KEY` a slot of that key only.

The file is read under a shared lock and changed only under an exclusive one,
and replaced whole, so concurrent claims from threads or processes neither
double-book a slot nor leave a half-written file.

    tpu_slots.py claim    --run-id R --holder H \\
        [--resource NAME | --profile P | --hardware KEY | --most-available] \\
        [--runtime "<runtime>" --package P]
    tpu_slots.py set-job  --run-id R --resource NAME --holder H --job-id J
    tpu_slots.py release  --run-id R --resource NAME [--holder H]
    tpu_slots.py show     --run-id R

Every command prints one JSON object. Exit codes: 0 done, 2 the session lacks
what is asked for (no parallel.json, no TPU profile, an unknown slot), 3
refused (every slot held), 1 anything else.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
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

# Backends with room to book: a TPU VM is one server, a gke_job profile as
# many as its `slots` param declares. `local` is this host, which compute's
# own lock already serializes; a `gke_pod` is one pod, already running, that no other
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


def plan_slots(
    profiles: dict[str, Any],
    inspect: Callable[[str], dict[str, Any] | None] = lambda _: None
) -> tuple[dict[str, dict], dict[str, dict]]:
    """The slots the profiles declare, by name, and how each was counted.

    `inspect` names a TPU VM's hardware, which its profile does not; a
    gke_job profile names its own and is never inspected.
    """
    slots: dict[str, dict] = {}
    sized: dict[str, dict] = {}
    for name, profile in sorted(profiles.items()):
        backend = (profile or {}).get("backend")
        if backend not in SLOT_BACKENDS:
            continue
        if backend == "gce_tpu_vm":
            key = hardware_key(profile, inspect(name) or {})
            slots[name] = {"profile": name, "hardware": key}
            sized[name] = {"hardware": key, "slots": 1, "by": "one VM"}
            continue
        key = hardware_key(profile)
        declared = ((profile or {}).get("params") or {}).get("slots")
        count = int(declared or 1)
        for i in range(1, count + 1):
            slots[f"{name}-{i}"] = {"profile": name, "hardware": key}
        sized[name] = {
            "hardware": key,
            "slots": count,
            "by": "its slots param" if declared else "one by default"
        }
    return slots, sized


def free_slots(
        state: dict[str, Any],
        fits: Callable[[dict[str, Any]], bool] = lambda _: True) -> list[str]:
    """Free slots that fit."""
    return [
        n for n, e in sorted((state.get("resources") or {}).items())
        if not e.get("job") and fits(e)
    ]


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
        out[key]["free"] = sum(1 for n in names if n in free)
    return out


def open_slots(
        path: Path,
        profiles: dict[str, Any],
        inspect: Callable[[str],
                          dict[str, Any] | None] = lambda _: None) -> dict:
    """Write the session's slots, the ones the profiles declare, once.

    An existing file is left as it is: slots are set at the start of the
    session. `parallel` is the slot count. With no gce_tpu_vm or gke_job
    profile it writes nothing and asks which TPU to use. Returns the file,
    the slots it added, how each profile was counted and the free servers per
    hardware key.
    """
    found, sized = plan_slots(profiles, inspect)
    if not sized:
        raise NeedsInput(
            "The session has no gce_tpu_vm or gke_job compute profile to run "
            "servers on: which TPU should the benchmark use?")
    added: list[str] = []
    with locked_state(path) as held:
        if held["state"] is None:
            held["state"] = {
                "parallel": len(found),
                "resources": {
                    name: {
                        **slot,
                        **{
                            k: None
                            for k in CLAIM_FIELDS
                        }
                    }
                    for name, slot in found.items()
                },
            }
            added = list(found)
        state = held["state"]
    return {
        "status": "ok",
        "path": str(path),
        "added": added,
        "sized": sized,
        "parallel": state["parallel"],
        "hardware": summarize(state),
        "session": state,
    }


def claim(path: Path,
          holder: str,
          resource: str | None = None,
          run_id: str | None = None,
          profile: str | None = None,
          hardware: str | None = None,
          most_available: bool = False) -> dict:
    """Take `resource`, or a free slot, for `holder`.

    `profile` and `hardware` limit which slots fit. `most_available` takes a
    slot of the hardware key with room for the most servers, on the profile with the most of that room, so
    the servers that follow it find room of the same kind. A slot `holder`
    already has is returned again rather than refused, so a retried claim is
    harmless.
    """

    def fits(entry: dict[str, Any]) -> bool:
        return (profile in (None, entry.get("profile"))
                and hardware in (None, entry.get("hardware")))

    with locked_state(path) as held:
        if held["state"] is None:
            raise NeedsInput(f"no slots at {path}: claim with --runtime (and"
                             " --package) to write them from the session's"
                             " compute profiles")
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
            name = resource
        else:
            if most_available:
                by_key: dict[str, list[str]] = {}
                for n in free:
                    if resources[n].get("hardware"):
                        by_key.setdefault(resources[n]["hardware"],
                                          []).append(n)
                if by_key:
                    key = min(by_key, key=lambda k: (-len(by_key[k]), k))
                    by_profile: dict[str, list[str]] = {}
                    for n in by_key[key]:
                        by_profile.setdefault(resources[n]["profile"],
                                              []).append(n)
                    best = min(by_profile,
                               key=lambda p: (-len(by_profile[p]), p))
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
            len(free_slots(state, lambda e: e.get("hardware") == key))
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
        raise NeedsInput(f"no slots at {path}: claim with --runtime first")
    return {"status": "ok", "path": str(path), "session": state}


def _argv(runtime: str) -> list[str]:
    # A quoted --runtime "python3 ~/..." reaches us with its ~ unexpanded.
    return [os.path.expanduser(t) for t in shlex.split(runtime)]


def read_config(runtime: str, package: str | None,
                run_id: str | None) -> dict[str, Any]:
    """`<runtime> config show`: the profiles and the run's active one."""
    argv = _argv(runtime) + ["config", "show", "--json"]
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
        config = json.loads(proc.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"`{' '.join(argv)}` printed no config: "
                           f"{exc}") from exc
    if proc.returncode != 0 or "profiles" not in config:
        raise RuntimeError(f"`{' '.join(argv)}` failed: "
                           f"{(proc.stderr or proc.stdout).strip()[-500:]}")
    return config


def read_profiles(runtime: str, package: str | None,
                  run_id: str | None) -> dict[str, Any]:
    """The compute profiles from `<runtime> config show`."""
    return read_config(runtime, package, run_id).get("profiles") or {}


def read_inspect(runtime: str, package: str | None,
                 run_id: str | None) -> Callable[[str], dict | None]:
    """`<runtime> compute inspect --profile P`, per profile; None if it fails."""

    def inspect(profile: str) -> dict[str, Any] | None:
        argv = _argv(runtime) + [
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
    c.add_argument("--runtime",
                   default=None,
                   help="the command that invokes the runtime; with no "
                   "parallel.json yet, write it from the compute profiles")
    c.add_argument("--package", default=None)
    j = sub.add_parser("set-job", parents=[common])
    j.add_argument("--resource", required=True)
    j.add_argument("--holder", required=True)
    j.add_argument("--job-id", required=True)
    r = sub.add_parser("release", parents=[common])
    r.add_argument("--resource", required=True)
    r.add_argument("--holder", default=None)
    sub.add_parser("show", parents=[common])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        path = state_path(args.run_id, args.state)
        if args.command == "claim":
            if args.runtime and not path.exists():
                open_slots(
                    path, read_profiles(args.runtime, args.package,
                                        args.run_id),
                    read_inspect(args.runtime, args.package, args.run_id))
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
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
