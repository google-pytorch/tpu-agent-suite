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
"""One serving sweep, spread over several identical servers at once.

serving_benchmark.py measures one server, one (isl, osl) shape after another.
A sweep of several shapes -- 1k/1k, 1k/8k, 8k/1k -- takes the sum of them.
This script runs the shapes side by side on servers started from one launch
recipe, so it takes about as long as the busiest server's share:

    serving_sweep.py run     --runtime "<runtime>" --package P --run-id R \\
        --name gemma --port 8000 --recipe launch_vllm.sh --server-input ... \\
        [--leader JOB[@PROFILE]] [--max-model-len N] [--patch P ...] \\
        [--hardware KEY] [--keep-servers] --out-dir bench_work \\
        -- <serving_benchmark.py's flags: --isl-osl, --concurrency, --engine,
        --model, --tokenizer, --stop-below-gain-pct, --stop-on-failure, ...>
    serving_sweep.py plan    --out sweep_plan.json [--max-model-len N] \\
        -- <the same client flags>       what `run` would queue, run nothing
    serving_sweep.py merge   --out-dir bench_work
    serving_sweep.py wait    --out-dir bench_work [--out-dir ...] \\
        [--max-secs 540]                one line per run, once all end
    serving_sweep.py cleanup --runtime "<runtime>" --package P --run-id R \\
        --out-dir bench_work

The client flags after `--` are serving_benchmark.py's, read by its own
parser, so they mean and default to what they would in one run. This script
sets `--base-url` and `--output-dir` for each shape's client, and its
`--isl-osl` and `--seed`.

The unit of work is a whole shape. serving_benchmark.py stops a shape early
by comparing each concurrency with the ones before it, so one shape's
concurrencies must run in order, in one client, against one server. The
shapes wait in a queue, longest first; each server takes the next one when
it finishes the last, so a shape that stops early frees its server for the
rest.

Each shape is measured as it would be in one unsharded sweep: the same client
flags, the full `--concurrency` list (the client filters it as it would
there), and a base seed that puts every cell at its place in the unsharded
grid. `merge` puts the shapes back into one `summary.json` in
serving_benchmark.py's layout, each shape judged by its `shape_ok`, with a
`shards` list naming the server each shape ran on.

Servers. `--leader` is a server that is already serving the model -- the
deployment onboarding verified, or one an earlier onboarding in this run left
serving with the same deployment -- and is never cancelled here; it may be
given more than once, each as `JOB` or `JOB@PROFILE`. A leader must be a job
of this run submitted with `--serve`: only those get the Service a client
reaches at `http://<job id>:<port>`; one that is not is left out, and so is
one of another kind of TPU than the first. Every other server is started here
from `--recipe`, with `--serve` for the same reason, on a slot `tpu_slots.py`
books in the session's parallel.json, and only after the first server serves,
so the compile cache and weights it leaves on a shared volume are warm for the
rest. They are cancelled however the run ends; a run that died before it could
cancel them is cleaned up with `cleanup`, from its sweep_status.json.

A shape whose server stopped serving goes back in the queue, once, for
another server; a shape whose client could not start (exit 2 with the server
still serving) fails without one, since it would fail the same way again.

Every server is one kind of TPU: tpu_slots.py's hardware key (accelerator,
topology, chips per job, image), whichever profiles reach it, since a shape
measured on other hardware is a different measurement. The kind is the first
leader's; without one, the first server goes to the kind with the most free
slots, so the rest find room beside it. Followers book only slots of that
kind, and every one is submitted with the same image (`--server-image`, else
the kind's), so a profile's own image never stands in for the verified one.
`merge` refuses shapes measured on more than one kind. To compare hardware,
run one sweep per kind and compare the summaries.

`--hardware KEY` pins the kind: every server books a slot of that
tpu_slots.py hardware key, and a leader of another kind is left out.

`--patch P`, repeatable, makes the sweep one arm of a patched A/B: every
server it starts gets each patch, apply_vllm_patch.sh and a vllm_patches.txt
naming them in order, which serve_vllm.sh applies before vLLM starts (so the
recipe must start vLLM through serve_vllm.sh). A patch that does not apply
fails its server and stops the sweep: every other server would fail the same
way. A sweep with `--patch` takes no `--leader`: it could not tell what the
leader runs. The summary lists each patch with its sha256.

`--keep-servers` leaves every server started here that served running, its
slot held, after the sweep, so the caller can profile it; sweep_status.json
marks each `kept`, and `cleanup` cancels them.

On a cluster that stops jobs a fixed time after they start, pass
`--server-budget-s`, the part of that limit a server may spend serving shapes:
a server takes no new shape once its budget is spent, counted from its job's
start (for a leader, as `job list` records it; `--leader-age-s` when it has
none), and the shapes left go to the others.

Only gke_job runs servers side by side here; a server on any other backend
is refused. Its sweep is serving_benchmark.py's, run once with every shape
(beside the server or through a port forward): the same measurement, one
shape after another.

Everything goes through `<runtime> compute`. The threads share one lock, for
the queue and the status file; each writes only its own shape's directory.

`run` writes `<out-dir>/sweep_plan.json` first, `<out-dir>/sweep_status.json`
as it goes and `<out-dir>/sweep_result.json` once it ends, each whole, into an
out-dir no earlier run used. Exit codes: 0 every shape measured cleanly, 1 a
shape failed or was not run, 2 no server could serve the sweep, or the
request was invalid, refused or needs input.

`wait` blocks until every run named has ended (its sweep_result.json exists)
or died (its process is gone with no result), or `--max-secs` pass, and
prints each run's status and shapes done: exit 0 all ended, 3 wait again.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import enum
import hashlib
import io
import json
import math
import os
import pathlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import serving_benchmark  # noqa: E402  # pylint: disable=wrong-import-position
import tpu_slots  # noqa: E402  # pylint: disable=wrong-import-position

Path = pathlib.Path
write_atomic = tpu_slots.write_atomic

# The client's: 0 every shape measured, 1 one failed, 2 nothing could run.
EXIT_OK = serving_benchmark.EXIT_OK
EXIT_FAILED = serving_benchmark.EXIT_CELL_FAILED
EXIT_SETUP = serving_benchmark.EXIT_SETUP
EXIT_WAITING = 3


class Wait(enum.IntEnum):
    """`compute job wait`'s exit codes."""
    REACHED = 0
    FAILURE_LOGGED = 2
    TIMED_OUT = 3
    JOB_ENDED = 4


# One `job wait` call; short, so a stop request is seen within a poll or two.
WAIT_CHUNK_S = 120
# Tries of one shape -- a client submit that failed, or a server lost during
# it -- before the shape counts as failed.
SHAPE_TRIES = 2
PARALLEL_BACKENDS = ("gke_job", )

# How a server is known to serve, or to have failed to.
SERVER_READY = "Application startup complete"
SERVER_FAILED = (r"WEIGHTS FAILED|Traceback \(most recent call last\)|"
                 r"serve_vllm: patch (does not apply|failed)")
PATCH_FAILED = re.compile(r"patch (does not apply|failed)")
PATCHES_APPLIED = "serve_vllm: applied"
APPLIER = HERE / "vllm_patches" / "apply_vllm_patch.sh"

# Each shape's client: a job beside its server running CLIENT_RECIPE, which
# keeps serving_benchmark.py's exit code in a file so the job's files come
# back however the shape went.
CLIENT_RECIPE = HERE / "serving_sweep_client.sh"
CLIENT_DIR = "serving_bench"
CLIENT_INPUTS = (
    HERE / "serving_benchmark.py",
    HERE / "benchmark_fastapi_server.py",
    # The client copies a gs:// tokenizer with it.
    HERE.parent / "tpu-model-onboarding" / "fetch_weights.py",
)
CLIENT_OUTPUTS = ("summary.json", "summary.md", "client.json", "exit_code")
# Client flags this script sets: for every shape, and per shape from the
# caller's (one shape each, at its own seed).
SET_HERE = ("--base-url", "--output-dir")
PER_SHAPE = ("--isl-osl", "--seed")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, json.dumps(data, indent=1) + "\n")


def shape_id(isl: int, osl: int) -> str:
    return f"isl{isl}_osl{osl}"


def with_image(kind: str | None, image: str | None) -> str | None:
    """`kind` with its image replaced: the image the servers run."""
    hardware = tpu_slots.Hardware.parse(kind)
    if hardware is None or not image:
        return kind
    return hardware._replace(image=image).key


def flag_name(arg: str) -> str:
    return arg.split("=", 1)[0]


def without_flags(argv: list[str], names: tuple[str, ...]) -> list[str]:
    """`argv` less every `--name value` and `--name=value` of `names`."""
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in names:
            skip = True
            continue
        if flag_name(arg) in names:
            continue
        out.append(arg)
    return out


# ------------------------------------------------------------------ plan


def parse_client(client_args: list[str]) -> argparse.Namespace:
    """The client flags as serving_benchmark.py reads them, or ValueError."""
    clash = [a for a in client_args if flag_name(a) in SET_HERE]
    if clash:
        raise ValueError(f"{', '.join(clash)} are set here for each shape's "
                         "client; leave them out")
    errors = io.StringIO()
    try:
        with contextlib.redirect_stderr(errors):
            client = serving_benchmark.parse_args(
                list(client_args) + ["--output-dir", CLIENT_DIR])
    except SystemExit as exc:
        raise ValueError("the client flags are not serving_benchmark.py's: " +
                         (errors.getvalue().strip().splitlines() or [str(exc)]
                          )[-1]) from exc
    if client.workload != "random":
        raise ValueError(f"--workload {client.workload} is not supported: "
                         "a sweep spreads ISL:OSL shapes over servers, and "
                         "this workload has none. Run serving_benchmark.py "
                         "once against the server instead.")
    return client


def shape_cost(client: argparse.Namespace, isl: int, osl: int,
               c_range: list[int]) -> float:
    """A rough duration, used only to queue the longest shapes first.

    Waves of requests (prompts / concurrency) times the tokens one request
    takes, with a prefill token counted as an eighth of a decode step: a
    guess, but the order it gives (1k/8k before 8k/1k before 1k/1k) is the
    one that matters.
    """
    return sum(
        math.ceil(serving_benchmark.prompts_for(client, c) / c) *
        (osl + isl / 8) for c in c_range)


def make_plan(client_args: list[str],
              max_model_len: int | None) -> dict[str, Any]:
    """The shapes to run, each with what its client will keep, and its seed."""
    client = parse_client(client_args)
    concurrencies = sorted(set(client.concurrencies))
    shapes = []
    for index, (isl, osl) in enumerate(client.cells):
        kept, skipped = serving_benchmark.filter_c_range(
            isl,
            osl,
            concurrencies,
            max_model_len=max_model_len,
            range_ratio=client.range_ratio,
            max_concurrency_limit=client.max_concurrency_limit,
            avoid_rules=client.avoid_rules,
        )
        shapes.append({
            "id":
            shape_id(isl, osl),
            "isl":
            isl,
            "osl":
            osl,
            # The shape's place in --isl-osl. The client seeds a cell by its
            # place in its own grid, so a base seed past every earlier
            # shape's cells puts it where the unsharded sweep would.
            "index":
            index,
            "seed":
            client.seed + index * len(concurrencies) * client.replicates,
            # What the client will run, for sizing and ordering only: it is
            # sent the full list and filters it the same way itself.
            "c_range":
            kept,
            "skipped":
            skipped,
            "cost":
            round(shape_cost(client, isl, osl, kept)),
        })
    ids = [s["id"] for s in shapes]
    if len(set(ids)) != len(ids):
        raise ValueError(f"--isl-osl names a shape twice: {client.isl_osl}")
    return {
        "created_at":
        utc_now(),
        "client_args":
        list(client_args),
        "max_model_len":
        max_model_len,
        # What the merged summary says when no shape's client wrote one.
        "settings": {
            "engine": client.engine,
            "num_prompts": client.num_prompts,
            "replicates": client.replicates,
            "range_ratio": client.range_ratio,
            "seed": client.seed,
            "max_concurrency_limit": client.max_concurrency_limit,
            "avoid": client.avoid or None,
            "stop_below_gain_pct": client.stop_below_gain_pct,
            "stop_on_failure": client.stop_on_failure,
            "sweep_mode": serving_benchmark.sweep_mode(client),
        },
        # Longest first: the long tail starts early and the short shapes fill
        # in behind it.
        "queue": [
            s["id"]
            for s in sorted(shapes, key=lambda s: (-s["cost"], s["index"]))
            if s["c_range"]
        ],
        "shapes":
        shapes,
    }


# ------------------------------------------------------------------ compute


class Compute:
    """`<runtime> compute ...` calls, each with its output in its own files.

    Files, not pipes: a non-serving `submit` starts a watcher process that
    inherits the caller's stdout, and reading a pipe would wait for it. The
    watcher may write to the file later, so the result is the first JSON line,
    which compute printed before it exited; `submit_job` also finds the job by
    its name in `job list` when that line names none.
    """

    def __init__(self, runtime: str, package: str | None, run_id: str | None,
                 calls_dir: Path):
        self.runtime = runtime
        # A quoted --runtime "python3 ~/..." reaches us with its ~ unexpanded.
        self.base = [os.path.expanduser(t)
                     for t in shlex.split(runtime)] + ["compute"]
        self.package, self.run_id = package, run_id
        self.calls_dir = calls_dir
        self._n = 0
        self._lock = threading.Lock()

    def __call__(self,
                 *args: str,
                 profile: str | None = None,
                 timeout: float | None = None,
                 tail: list[str] | None = None) -> tuple[int, dict[str, Any]]:
        argv = self.base + list(args) + ["--json"]
        if self.package:
            argv += ["--package", self.package]
        if self.run_id:
            argv += ["--run-id", self.run_id]
        if profile:
            argv += ["--profile", profile]
        argv += tail or []
        with self._lock:
            self._n += 1
            stem = self.calls_dir / f"{self._n:04d}-{args[0]}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        out, err = stem.with_suffix(".out"), stem.with_suffix(".err")
        with open(out,
                  "w", encoding="utf-8") as fo, open(err,
                                                     "w",
                                                     encoding="utf-8") as fe:
            try:
                rc = subprocess.run(argv,
                                    stdin=subprocess.DEVNULL,
                                    stdout=fo,
                                    stderr=fe,
                                    timeout=timeout,
                                    check=False).returncode
            except subprocess.TimeoutExpired:
                rc = -1
            except OSError as exc:
                rc = -1
                fe.write(f"{exc}\n")
        body: Any = {}
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                body = json.loads(line)
                break
            except ValueError:
                continue
        if rc and not body:
            body = {"error": err.read_text(encoding="utf-8").strip()[-600:]}
        return rc, body if isinstance(body, dict) else {"result": body}

    def submit_job(self,
                   *args: str,
                   name: str,
                   profile: str | None,
                   tail: list[str] | None = None) -> tuple[str | None, dict]:
        """`submit`, then the job's id: from its output, else `job list`."""
        rc, body = self("submit",
                        *args,
                        profile=profile,
                        timeout=600,
                        tail=tail)
        if body.get("job_id"):
            return body["job_id"], body
        mine = [
            r for r in self.jobs(profile)
            if r.get("job_name") == name and r.get("status") not in (
                "COMPLETED", "FAILED", "CANCELLED")
        ]
        if rc == 0 and mine:
            latest = max(mine, key=lambda r: r.get("start_time") or "")
            return latest["job_id"], body
        return None, body

    def jobs(self, profile: str | None = None) -> list[dict[str, Any]]:
        """This run's job records, as `job list --no-refresh` has them."""
        _, listed = self("job",
                         "list",
                         "--no-refresh",
                         profile=profile,
                         timeout=120)
        rows = listed.get("result")
        return rows if isinstance(rows, list) else []


def age_s(start_time: str | None) -> float | None:
    """Seconds since an ISO-8601 `start_time`, or None if it cannot be read."""
    try:
        started = datetime.datetime.fromisoformat(str(start_time))
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=datetime.timezone.utc)
    now = datetime.datetime.now(datetime.timezone.utc)
    return max((now - started).total_seconds(), 0.0)


# ------------------------------------------------------------------ run


class Sweep:
    """The servers, the queue of shapes, and what each shape returned."""

    def __init__(self, args: argparse.Namespace, plan: dict[str, Any],
                 compute: Callable[..., tuple[int, dict[str, Any]]],
                 profiles: dict[str, dict[str,
                                          Any]], active_profile: str | None):
        self.args, self.plan, self.compute = args, plan, compute
        self.profiles, self.active_profile = profiles, active_profile
        self.backends = {
            name: (prof or {}).get("backend")
            for name, prof in profiles.items()
        }
        self.out = Path(args.out_dir)
        self.lock = threading.Lock()
        # Held from snapshot to write, so a later state is never overwritten
        # by an earlier one.
        self.status_lock = threading.Lock()
        self.stop = threading.Event()
        self.stop_requested = False
        self.first_ready = threading.Event()
        self.queue = list(plan["queue"])
        self.by_id = {s["id"]: s for s in plan["shapes"]}
        self.shapes: dict[str, dict[str, Any]] = {
            sid: {
                "state": "queued",
                "attempts": 0
            }
            for sid in self.queue
        }
        self.servers: list[dict[str, Any]] = []
        self.ignored: list[dict[str, str]] = []
        self.client_jobs: set[tuple[str, str | None]] = set()
        # Found in plan_servers, inside run's error handling.
        self.slots_path: Path | None = None
        self.image: str | None = args.server_image
        self._patch_inputs: list[str] = []
        self.patch_sha256: list[str] = []

    def submit(self,
               *args: str,
               name: str,
               profile: str | None,
               tail: list[str] | None = None) -> tuple[str | None, dict]:
        return self.compute.submit_job(*args,
                                       name=name,
                                       profile=profile,
                                       tail=tail)

    # -------------------------------------------------- status

    def save_status(self) -> None:
        with self.status_lock:
            with self.lock:
                snapshot = {
                    "updated_at":
                    utc_now(),
                    "pid":
                    os.getpid(),
                    "slots":
                    str(self.slots_path) if self.slots_path else None,
                    "servers": [{
                        k: v
                        for k, v in s.items() if not k.startswith("_")
                    } for s in self.servers],
                    "ignored_leaders":
                    list(self.ignored),
                    "queue":
                    list(self.queue),
                    "shapes":
                    copy.deepcopy(self.shapes),
                    **self.pinned(),
                }
            write_json(self.out / "sweep_status.json", snapshot)

    def pinned(self) -> dict[str, Any]:
        """--patch (with each file's sha256) and --hardware, as given."""
        out: dict[str, Any] = {}
        if self.args.patch:
            out["patches"] = list(self.args.patch)
            out["patch_sha256"] = list(self.patch_sha256)
        if self.args.hardware:
            out["hardware"] = self.args.hardware
        return out

    def patch_inputs(self) -> list[str]:
        """Each patch, the applier and the list naming them, as --input."""
        if not self.args.patch:
            return []
        missing = [p for p in self.args.patch if not Path(p).is_file()]
        if missing:
            raise ValueError(f"--patch {', '.join(missing)}: no such file")
        # Hashed once: what every server is sent is what the summary records.
        self.patch_sha256 = [
            hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in self.args.patch
        ]
        names = [Path(p).name for p in self.args.patch]
        # As compute reads --input: LOCAL[:REMOTE], REMOTE defaulting to
        # LOCAL's name.
        taken = {"vllm_patches.txt"}
        for spec in self.args.server_input:
            local, sep, remote = spec.partition(":")
            taken.add(remote if sep and remote else Path(local).name)
        twice = {n for n in names if names.count(n) > 1}
        clash = sorted(twice | (set(names) & taken))
        if clash:
            raise ValueError(f"--patch names {', '.join(clash)} twice, or as "
                             "a --server-input: rename one")
        listing = self.out / "vllm_patches.txt"
        write_atomic(listing, "".join(n + "\n" for n in names))
        inputs = [f"{p}:{n}" for p, n in zip(self.args.patch, names)]
        if "apply_vllm_patch.sh" not in taken - {"vllm_patches.txt"}:
            inputs.append(f"{APPLIER}:apply_vllm_patch.sh")
        return inputs + [f"{listing}:vllm_patches.txt"]

    def note(self, server: dict[str, Any], **fields: Any) -> None:
        with self.lock:
            server.update(fields)
        self.save_status()

    def next_shape(self, server: dict[str, Any]) -> str | None:
        """The next shape for `server`, or None when it should take no more."""
        budget = self.args.server_budget_s
        with self.lock:
            if self.stop.is_set() or not self.queue:
                return None
            if budget and time.monotonic() - server.get("_since", 0) > budget:
                server["state"] = "budget spent"
                return None
            return self.queue.pop(0)

    def set_shape(self, sid: str, **fields: Any) -> None:
        with self.lock:
            self.shapes[sid].update(fields)
        self.save_status()

    def requeue(self, sid: str, why: str) -> bool:
        """Put a shape back for another try; False when it has had its tries."""
        with self.lock:
            shape = self.shapes[sid]
            if shape["attempts"] >= SHAPE_TRIES or self.stop.is_set():
                return False
            shape.update(state="queued", error=why)
            self.queue.insert(0, sid)
        self.save_status()
        return True

    def retry_or_fail(self, sid: str, why: str) -> None:
        if not self.requeue(sid, why):
            self.set_shape(sid, state="failed", error=why)

    # -------------------------------------------------- servers

    def kind_of(self, profile: str | None) -> str | None:
        return tpu_slots.hardware_key(self.profiles.get(profile or "") or {})

    def leaders(self, want: int) -> list[dict[str, Any]]:
        """The usable `--leader`s: this run's serving jobs, of one kind."""
        records = {r.get("job_id"): r for r in self.compute.jobs()}
        default = self.args.leader_profile or self.active_profile
        found: list[dict[str, Any]] = []
        for spec in self.args.leader:
            job_id, _, profile = spec.partition("@")
            profile = profile or default
            record = records.get(job_id)
            why = None
            if record is None:
                why = "not a job of this run"
            elif not record.get("serve"):
                why = ("submitted without --serve, so it has no Service a "
                       "client can reach")
            elif record.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
                why = f"no longer running ({record.get('status')})"
            elif (self.args.hardware
                  and self.kind_of(profile) != self.args.hardware):
                why = (f"another kind of TPU ({self.kind_of(profile)}) than "
                       f"--hardware {self.args.hardware}")
            elif found and self.kind_of(profile) != found[0]["_kind"]:
                why = (f"another kind of TPU ({self.kind_of(profile)}) than "
                       f"{found[0]['job_id']} ({found[0]['_kind']})")
            elif len(found) >= want:
                why = "more leaders than shapes"
            if why:
                self.ignored.append({"job_id": job_id, "why": why})
                self.log(f"leader {job_id}: not used: {why}")
                continue
            # Its own age, from its record; the flag only when it has none.
            age = age_s(record.get("start_time"))
            found.append({
                "label":
                "leader" if not found else f"leader{len(found) + 1}",
                "job_id":
                job_id,
                "profile":
                profile,
                "owned":
                False,
                "state":
                "serving",
                "_kind":
                self.kind_of(profile),
                "_since":
                time.monotonic() -
                (self.args.leader_age_s if age is None else age),
            })
        return found

    def plan_servers(self) -> None:
        """The leaders, then a follower per slot, up to one server per shape."""
        if not self.args.no_slots:
            self.slots_path = tpu_slots.state_path(self.args.run_id,
                                                   self.args.slots)
            if not self.slots_path.exists():
                # The session's slots, set once from its profiles. A sweep
                # books only gke_job slots, so no TPU VM is inspected.
                tpu_slots.open_slots(self.slots_path, self.profiles)
        want = min(len(self.queue), self.args.max_servers)
        self.servers.extend(self.leaders(want))
        # One kind of TPU for every server: the first leader's, else that of
        # the first slot, taken where most slots of its kind are free. A
        # leader whose kind is unknown keeps the followers on its profile.
        kind = (self.servers[0]["_kind"]
                if self.servers else self.args.hardware)
        same_profile = (self.servers[0]["profile"]
                        if self.servers and not kind else None)
        for i in range(len(self.servers), want):
            label = f"s{i + 1}"
            holder = f"serving-sweep:{self.args.name}:{label}"
            profile, resource = same_profile or self.active_profile, None
            got_kind = kind or self.kind_of(profile)
            if self.slots_path is not None:
                try:
                    got = self.claim(holder, kind, same_profile)
                except (tpu_slots.Refused, tpu_slots.NeedsInput) as exc:
                    if not self.servers:
                        raise
                    self.log(f"{label}: no slot ({exc}); "
                             f"running on {len(self.servers)} server(s)")
                    break
                profile, resource = got["profile"], got["resource"]
                got_kind = got.get("hardware")
            if kind is None and same_profile is None:
                kind = got_kind
                same_profile = None if kind else profile
            self.servers.append({
                "label": label,
                "job_id": None,
                "profile": profile,
                "resource": resource,
                "holder": holder,
                "owned": True,
                "state": "waiting",
                "_kind": got_kind,
            })
        # Every follower runs one image: the caller's, else the kind's.
        hardware = tpu_slots.Hardware.parse(kind)
        if not self.image and hardware:
            self.image = hardware.image
        for server in self.servers:
            server["hardware"] = with_image(server["_kind"], self.image)
            backend = self.backends.get(server["profile"] or "")
            if backend not in PARALLEL_BACKENDS:
                raise ValueError(
                    f"server {server['label']} is on profile "
                    f"{server['profile']!r}, backend {backend!r}: servers run "
                    f"side by side only on {', '.join(PARALLEL_BACKENDS)}. "
                    "Measure the sweep on one server instead: run "
                    "serving_benchmark.py once with every shape in its "
                    "--isl-osl, beside the server or through a port forward; "
                    "it runs them one after another.")
        if self.servers and not self.servers[0]["owned"]:
            self.first_ready.set()

    def claim(self, holder: str, kind: str | None,
              same_profile: str | None) -> dict[str, Any]:
        """A slot of `kind`, else on `same_profile`, else where most are free."""
        if kind:
            return tpu_slots.claim(self.slots_path,
                                   holder,
                                   run_id=self.args.run_id,
                                   hardware=kind)
        if same_profile:
            return tpu_slots.claim(self.slots_path,
                                   holder,
                                   run_id=self.args.run_id,
                                   profile=same_profile)
        try:
            return tpu_slots.claim(self.slots_path,
                                   holder,
                                   run_id=self.args.run_id,
                                   most_available=True)
        except tpu_slots.Refused:
            # No slot names its kind: take any, and keep the rest on its
            # profile.
            return tpu_slots.claim(self.slots_path,
                                   holder,
                                   run_id=self.args.run_id)

    def wait(self,
             job_id: str,
             until: str,
             profile: str | None,
             limit_s: float,
             fail_on: str | None = None,
             default_fail: bool = True) -> tuple[int, dict]:
        """`job wait` in short calls until it settles, `limit_s` or a stop."""
        deadline = time.monotonic() + limit_s
        while True:
            tail = ["--until", until, "--timeout", str(WAIT_CHUNK_S)]
            if fail_on:
                tail += ["--fail-on-log", fail_on]
            if not default_fail:
                tail += ["--no-default-fail-patterns"]
            rc, body = self.compute("job",
                                    "wait",
                                    "--job-id",
                                    job_id,
                                    profile=profile,
                                    timeout=WAIT_CHUNK_S + 120,
                                    tail=tail)
            if rc != Wait.TIMED_OUT:
                return rc, body
            if self.stop.is_set() or time.monotonic() > deadline:
                return Wait.TIMED_OUT, body

    def start_server(self, server: dict[str, Any]) -> bool:
        """Submit a follower from the recipe and wait until it serves."""
        args = self.args
        if not self.first_ready.is_set() and server is not self.servers[0]:
            self.note(server, state="waiting for the first server")
            while not self.first_ready.wait(10):
                if self.stop.is_set():
                    return False
        if self.stop.is_set():
            return False
        name = f"{args.name}-{server['label']}"
        # --serve for the Service its clients reach it by; it is cancelled
        # here, or by `cleanup` from the status file.
        submit = ["--recipe", args.recipe, "--name", name, "--serve"]
        if args.interpreter:
            submit += ["--interpreter", args.interpreter]
        for spec in args.server_input + self._patch_inputs:
            submit += ["--input", spec]
        if self.image:
            submit += ["--image", self.image]
        submit += list(args.server_submit_arg)
        self.note(server, state="submitting", name=name)
        job_id, body = self.submit(*submit,
                                   name=name,
                                   profile=server["profile"])
        if not job_id:
            self.note(server,
                      state="failed",
                      error=f"submit failed: {body.get('error') or body}")
            return False
        with self.lock:
            server["_since"] = time.monotonic()
        self.note(server, job_id=job_id, state="starting")
        if self.slots_path is not None and server.get("resource"):
            with contextlib.suppress(tpu_slots.Refused):
                tpu_slots.set_job(self.slots_path, server["resource"],
                                  server["holder"], job_id)
        # Serving means its weights loaded, whatever the recipe printed on
        # the way; a failed fetch or a traceback ends the wait early.
        stages = [("running", SERVER_FAILED if args.patch else None)]
        if args.patch:
            # A recipe that skips serve_vllm.sh would serve the bare image.
            stages.append((f"log:{PATCHES_APPLIED}", SERVER_FAILED))
        stages.append((f"log:{SERVER_READY}", SERVER_FAILED))
        for until, fail_on in stages:
            rc, body = self.wait(job_id, until, server["profile"],
                                 args.start_timeout, fail_on)
            if rc != Wait.REACHED:
                why = body.get("line") or body.get("outcome") or body.get(
                    "error") or f"job wait exit {rc}"
                if PATCH_FAILED.search(str(why)):
                    # Every server gets the same patches: none would serve.
                    self.stop.set()
                self.note(server,
                          state="failed",
                          error=f"did not reach {until}: {why}")
                return False
        self.note(server, state="serving", served=True)
        return True

    def server_running(self, server: dict[str, Any]) -> bool:
        rc, body = self.compute("job",
                                "status",
                                "--job-id",
                                server["job_id"],
                                profile=server["profile"],
                                timeout=300)
        return rc == 0 and body.get("status") == "RUNNING"

    def stop_server(self, server: dict[str, Any]) -> None:
        """Cancel a server started here and free its slot, once; keep its log."""
        with self.lock:
            if not server.get("owned") or server.get("released"):
                return
            if (self.args.keep_servers and server.get("served")
                    and server.get("state") not in ("failed", "lost")
                    and not self.stop_requested):
                server["kept"] = True
                return
            server["released"] = True
        job_id = server.get("job_id")
        if job_id:
            log = self.out / "servers" / f"{server['label']}.log"
            rc, body = self.compute("job",
                                    "logs",
                                    "--job-id",
                                    job_id,
                                    profile=server["profile"],
                                    timeout=300)
            if rc == 0 and body.get("logs"):
                log.parent.mkdir(parents=True, exist_ok=True)
                write_atomic(log, body["logs"])
            self.compute("job",
                         "cancel",
                         "--job-id",
                         job_id,
                         profile=server["profile"],
                         timeout=300)
        if self.slots_path is not None and server.get("resource"):
            tpu_slots.release(self.slots_path, server["resource"],
                              server["holder"])
        self.note(server,
                  state=("stopped" if server.get("state") == "serving" else
                         server.get("state")))

    # -------------------------------------------------- shapes

    def client_argv(self, server: dict[str, Any],
                    shape: dict[str, Any]) -> list[str]:
        """The caller's client flags, for this shape and this server."""
        return without_flags(self.plan["client_args"], PER_SHAPE) + [
            "--base-url", f"http://{server['job_id']}:{self.args.port}",
            "--isl-osl", f"{shape['isl']}:{shape['osl']}", "--seed",
            str(shape["seed"]), "--output-dir", CLIENT_DIR
        ]

    def run_shape(self, server: dict[str, Any], sid: str) -> bool:
        """One shape's client beside `server`. False when the server is gone."""
        shape = self.by_id[sid]
        shard = self.out / "shards" / sid
        # An earlier try's files must not pass for this one's.
        shutil.rmtree(shard, ignore_errors=True)
        name = f"{self.args.name}-{shape['isl']}x{shape['osl']}"
        with self.lock:
            self.shapes[sid]["attempts"] += 1
        self.set_shape(sid, state="running", server=server["label"])
        submit = [
            "--recipe",
            str(CLIENT_RECIPE), "--name", name, "--colocate-with",
            server["job_id"], "--colocate-requests",
            self.args.colocate_requests
        ]
        for path in CLIENT_INPUTS:
            submit += ["--input", f"{path}:{path.name}"]
        for f in CLIENT_OUTPUTS:
            submit += ["--output", f"{CLIENT_DIR}/{f}"]
        job_id, body = self.submit(*submit,
                                   name=name,
                                   profile=server["profile"],
                                   tail=["--"] +
                                   self.client_argv(server, shape))
        if not job_id:
            self.retry_or_fail(
                sid, "client submit failed: "
                f"{body.get('error') or body}")
            return self.server_running(server)
        key = (job_id, server["profile"])
        with self.lock:
            self.client_jobs.add(key)
        self.set_shape(sid, job_id=job_id)
        # The client's log is its own: a server failure line quoted there
        # must not end the wait while the client still runs.
        rc, body = self.wait(job_id,
                             "done",
                             server["profile"],
                             self.args.shape_timeout,
                             default_fail=False)
        if rc not in (Wait.REACHED, Wait.JOB_ENDED):
            # Still running, or its state unknown: it must not load the
            # server beside the next shape's client.
            self.compute("job",
                         "cancel",
                         "--job-id",
                         job_id,
                         profile=server["profile"],
                         timeout=300)
        self.compute("job",
                     "artifacts",
                     "--job-id",
                     job_id,
                     "--out-dir",
                     str(shard),
                     profile=server["profile"],
                     timeout=900)
        lrc, logs = self.compute("job",
                                 "logs",
                                 "--job-id",
                                 job_id,
                                 profile=server["profile"],
                                 timeout=300)
        if lrc == 0 and logs.get("logs"):
            shard.mkdir(parents=True, exist_ok=True)
            write_atomic(shard / "client.log", logs["logs"])
        with self.lock:
            self.client_jobs.discard(key)
        return self.settle(server, sid, shard, rc, body)

    def settle(self, server: dict[str, Any], sid: str, shard: Path, rc: int,
               body: dict[str, Any]) -> bool:
        """Record how a shape's client ended. False when its server is gone.

        A shape whose server stopped serving goes back for another server;
        one whose client could not start fails, since it would again.
        """
        summary_path = shard / CLIENT_DIR / "summary.json"
        client = shard / CLIENT_DIR / "client.json"
        code_path = shard / CLIENT_DIR / "exit_code"
        code = (int(code_path.read_text(encoding="utf-8").strip() or -1)
                if code_path.is_file() else None)
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("error"):
                # It gave up on a server that stopped answering /health.
                self.retry_or_fail(sid, f"server lost: {summary['error']}")
                return False
            if summary.get("status") == "partial":
                # The client's job ended mid-sweep (its node was lost, or it
                # timed out): the cells it finished came back, not the shape.
                why = "the client ended before its last cell"
                if not self.server_running(server):
                    self.retry_or_fail(sid, f"server lost: {why}")
                    return False
                self.retry_or_fail(sid, why)
                return True
            self.set_shape(sid,
                           state="done",
                           status=summary.get("status"),
                           summary=str(summary_path))
            return True
        why = ("timed out" if rc == Wait.TIMED_OUT else body.get("line")
               or body.get("outcome") or f"job wait exit {rc}")
        if client.is_file():
            why = client.read_text(encoding="utf-8").strip()[-400:] or why
        if not self.server_running(server):
            self.retry_or_fail(sid, f"server lost: {why}")
            return False
        if code == serving_benchmark.EXIT_SETUP:
            why = f"the client could not start: {why}"
        self.set_shape(sid,
                       state="failed",
                       error=f"no summary.json came back: {why}")
        return True

    def serve(self, server: dict[str, Any]) -> None:
        """A server's thread: start it if it is ours, then take shapes."""
        sid = None
        try:
            if server.get("owned"):
                ok = self.start_server(server)
                if ok:
                    self.first_ready.set()
                if not ok:
                    return
            while (sid := self.next_shape(server)) is not None:
                if not self.run_shape(server, sid):
                    self.note(server,
                              state="lost",
                              error=f"stopped answering during {sid}")
                    return
        except Exception as exc:  # noqa: BLE001 - recorded, then cleaned up
            why = f"{type(exc).__name__}: {exc}"
            # The shape in hand is failed, not left looking like it runs.
            if sid is not None and self.shapes[sid]["state"] == "running":
                self.set_shape(sid, state="failed", error=why)
            self.note(server, state="failed", error=why)
        finally:
            # However the first server ended, the others wait no longer.
            if self.servers and server is self.servers[0]:
                self.first_ready.set()
            self.stop_server(server)

    def log(self, text: str) -> None:
        print(f"[{utc_now()}] {text}", file=sys.stderr, flush=True)

    def run(self) -> dict[str, Any]:
        used = [
            f for f in ("sweep_status.json", "sweep_result.json")
            if (self.out / f).exists()
        ]
        if used:
            raise ValueError(f"{self.out} holds an earlier run's {used[0]}: "
                             "give each run a new --out-dir")
        self.out.mkdir(parents=True, exist_ok=True)
        write_json(self.out / "sweep_plan.json", self.plan)
        try:
            self._patch_inputs = self.patch_inputs()
            self.plan_servers()
        except BaseException:
            for server in self.servers:
                self.stop_server(server)
            raise
        self.save_status()
        threads = [
            threading.Thread(target=self.serve, args=(s, ), daemon=True)
            for s in self.servers
        ]
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(1)
        finally:
            self.stop.set()
            with self.lock:
                left = list(self.client_jobs)
            for job_id, profile in left:
                self.compute("job",
                             "cancel",
                             "--job-id",
                             job_id,
                             profile=profile,
                             timeout=300)
            # Every compute call has a timeout, so the threads end; waiting
            # for them is what lets their own cleanup run.
            for t in threads:
                t.join()
            for server in self.servers:
                self.stop_server(server)
            with self.lock:
                for sid in self.queue:
                    self.shapes[sid].update(
                        state="not_run",
                        error="no server was left to run it" if
                        not self.stop_requested else "the sweep was stopped")
                self.queue.clear()
            self.save_status()
        return merge(self.plan, self.out, self.shapes, self.servers,
                     self.pinned())

    def exit_code(self, merged: dict[str, Any]) -> int:
        """0 ok, 1 a shape failed or was not run, 2 no server ever served."""
        if merged["status"] == "ok":
            return EXIT_OK
        served = any(not s["owned"] or s.get("served") for s in self.servers)
        return EXIT_FAILED if served else EXIT_SETUP


# ------------------------------------------------------------------ merge


def merge(plan: dict[str, Any],
          out: Path,
          shapes: dict[str, dict[str, Any]] | None = None,
          servers: list[dict[str, Any]] | None = None,
          pinned: dict[str, Any] | None = None) -> dict[str, Any]:
    """The shapes' summaries as one serving_benchmark.py summary.json.

    Each shape is judged as the client judges it, by `shape_ok`; the sweep is
    ok when every shape is, no client lost its server, and every shape ran
    with the same settings, on the same model, on one kind of TPU.
    """
    shapes = shapes if shapes is not None else {}
    servers = servers or []
    by_label = {s["label"]: s for s in servers}
    cells: list[dict[str, Any]] = []
    shards, errors, warnings = [], [], []
    base: dict[str, Any] | None = None
    base_sid = None
    for shape in sorted(plan["shapes"], key=lambda s: s["index"]):
        sid = shape["id"]
        got = shapes.get(sid, {})
        path = out / "shards" / sid / CLIENT_DIR / "summary.json"
        summary = (json.loads(path.read_text(encoding="utf-8"))
                   if path.is_file() and got.get("state") == "done" else None)
        if summary is not None:
            # The client was sent every concurrency and reports each one,
            # the ones its own filter skipped included.
            shape_cells = list(summary.get("cells", []))
            warnings += [f"{sid}: {w}" for w in summary.get("warnings") or []]
            if base is None:
                base, base_sid = summary, sid
            elif _comparable(summary) != _comparable(base):
                errors.append(f"{sid} ran with other settings than "
                              f"{base_sid}: not merged as one sweep")
            stop_on_failure = bool((summary.get("settings")
                                    or {}).get("stop_on_failure"))
            if not serving_benchmark.shape_ok(shape_cells, stop_on_failure):
                errors.append(f"{sid}: no cell passed")
        else:
            shape_cells = list(shape["skipped"])
            if shape["c_range"]:
                # A shape whose client failed is failed; one no server
                # reached was not run.
                reason = got.get("error") or "the sweep ended before it ran"
                failed = got.get("state") == "failed"
                for c in shape["c_range"]:
                    cell = serving_benchmark.skipped_cell(
                        shape["isl"], shape["osl"], c,
                        reason if failed else f"not run: {reason}")
                    if failed:
                        cell["status"] = "failed"
                    shape_cells.append(cell)
            else:
                reason = "the plan left no concurrency to run"
            # As in one sweep: a shape with no measured cell fails it.
            errors.append(f"{sid}: {reason}")
        shape_cells.sort(key=lambda c: c["concurrency"])
        cells += shape_cells
        server = by_label.get(got.get("server", ""), {})
        shards.append({
            "shape":
            sid,
            "server":
            got.get("server"),
            "server_job_id":
            server.get("job_id"),
            "profile":
            server.get("profile"),
            "hardware":
            server.get("hardware"),
            "client_job_id":
            got.get("job_id"),
            "seed":
            shape["seed"],
            "tries":
            got.get("attempts"),
            "state":
            got.get("state") or ("skipped" if not shape["c_range"] else None),
            "base_url":
            summary.get("base_url") if summary else None,
        })
    # A server of unknown kind counts as its profile: never merged with
    # another unless they share it.
    kinds = sorted({
        by_label[sh["server"]].get("hardware")
        or f"profile {by_label[sh['server']].get('profile')}"
        for sh in shards if sh["server"] in by_label and sh["state"] == "done"
    })
    if len(kinds) > 1:
        errors.append(f"shapes were measured on more than one kind of TPU "
                      f"({'; '.join(kinds)}): not one sweep")
    ok = not errors and any(c["status"] == "ok" for c in cells)
    # The plan's settings when no shape's client wrote any.
    settings = {
        **plan.get("settings", {}),
        **((base or {}).get("settings") or {})
    }
    settings["seed"] = plan["settings"]["seed"]
    merged = {
        "status":
        "ok" if ok else "failed",
        "error":
        "; ".join(errors) or None,
        "warnings":
        warnings,
        "created_at":
        utc_now(),
        "base_url":
        None,
        "model": (base or {}).get("model"),
        "client": (base or {}).get("client"),
        "settings":
        settings,
        "cells":
        cells,
        "shards":
        shards,
        "servers": [{
            k: s.get(k)
            for k in ("label", "job_id", "profile", "hardware", "resource",
                      "owned", "state", "error")
        } for s in servers],
        **(pinned or {}),
    }
    write_json(out / CLIENT_DIR / "summary.json", merged)
    write_atomic(out / CLIENT_DIR / "summary.md",
                 serving_benchmark.format_table(cells) + "\n")
    return merged


def _comparable(summary: dict[str, Any]) -> tuple:
    """What must match across shapes for them to be one sweep."""
    settings = {
        k: v
        for k, v in (summary.get("settings") or {}).items() if k != "seed"
    }
    return (json.dumps(settings, sort_keys=True),
            json.dumps(summary.get("model"),
                       sort_keys=True), summary.get("client"))


# ------------------------------------------------------------------ cleanup


def run_state(out: Path) -> dict[str, Any]:
    """One run's progress: shapes done of all, and whether it ended or died."""
    result, status = out / "sweep_result.json", out / "sweep_status.json"
    if result.is_file():
        got = json.loads(result.read_text(encoding="utf-8"))
        return {"ended": True, "status": got.get("status")}
    if not status.is_file():
        # A run that died before its first status: the pid its launcher saved.
        pid = out / "pid"
        if pid.is_file() and not _alive(pid.read_text().strip()):
            return {"ended": True, "status": "died"}
        return {"ended": False, "status": "starting"}
    state = json.loads(status.read_text(encoding="utf-8"))
    shapes = list((state.get("shapes") or {}).values())
    row = {
        "ended": False,
        "status": "running",
        "shapes_done": sum(1 for s in shapes if s.get("state") == "done"),
        "shapes": len(shapes),
    }
    if not _alive(state.get("pid")):
        row.update(ended=True, status="died")
    return row


def _alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        pass
    return True


def wait(outs: list[Path],
         max_secs: float,
         poll_s: float = 10.0) -> dict[str, Any]:
    """Each run's state once every one has ended or died, or `max_secs` on."""
    deadline = time.monotonic() + max_secs
    while True:
        runs = {str(o): run_state(o) for o in outs}
        ended = all(r["ended"] for r in runs.values())
        if ended or time.monotonic() >= deadline:
            return {"status": "ended" if ended else "running", "runs": runs}
        time.sleep(poll_s)


def cleanup(out: Path, compute: Callable[..., tuple[int, dict[str, Any]]],
            slots: Path | None) -> dict[str, Any]:
    """Cancel the servers a run started and left, and free their slots.

    For a run that died before its own cleanup: its sweep_status.json names
    every server it started, its job and its slot. Servers it was handed are
    never touched.
    """
    status = json.loads(
        (out / "sweep_status.json").read_text(encoding="utf-8"))
    slots = slots or (Path(status["slots"]) if status.get("slots") else None)
    done = []
    for server in status.get("servers", []):
        if not server.get("owned") or server.get("released"):
            continue
        if server.get("job_id"):
            # A kept server's log was never fetched; keep it before it goes.
            log = out / "servers" / f"{server.get('label')}.log"
            rc, body = (1, {}) if log.exists() else compute(
                "job",
                "logs",
                "--job-id",
                server["job_id"],
                profile=server.get("profile"),
                timeout=300)
            if rc == 0 and body.get("logs"):
                log.parent.mkdir(parents=True, exist_ok=True)
                write_atomic(log, body["logs"])
            compute("job",
                    "cancel",
                    "--job-id",
                    server["job_id"],
                    profile=server.get("profile"),
                    timeout=300)
        if slots is not None and server.get("resource"):
            tpu_slots.release(slots, server["resource"], server.get("holder"))
        done.append(server.get("job_id") or server.get("label"))
    return {"status": "ok", "cancelled": done}


# ------------------------------------------------------------------ cli


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    client = argparse.ArgumentParser(add_help=False)
    client.add_argument("--max-model-len",
                        type=int,
                        default=None,
                        help="the server's; shapes that cannot fit get none")
    client.add_argument("client_args",
                        nargs=argparse.REMAINDER,
                        help="after --: serving_benchmark.py's flags")

    compute_args = argparse.ArgumentParser(add_help=False)
    compute_args.add_argument("--runtime",
                              required=True,
                              help="the command that invokes the runtime")
    compute_args.add_argument("--package", default=None)
    compute_args.add_argument("--run-id", default=None)
    compute_args.add_argument(
        "--slots",
        default=None,
        help="parallel.json to book from instead of the run's")

    pl = sub.add_parser("plan",
                        parents=[client],
                        help="what `run` would queue; runs nothing")
    pl.add_argument("--out", required=True)

    r = sub.add_parser("run",
                       parents=[compute_args, client],
                       help="run a sweep on servers side by side")
    r.add_argument("--plan",
                   default=None,
                   help="a plan `plan` wrote, instead of the flags after --")
    r.add_argument("--name",
                   required=True,
                   help="names the sweep's jobs and slot claims")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--recipe",
                   required=True,
                   help="the verified launch recipe, e.g. launch_vllm.sh")
    r.add_argument("--server-input",
                   action="append",
                   default=[],
                   metavar="LOCAL[:REMOTE]",
                   help="an --input of the recipe's own submit; repeatable")
    r.add_argument("--interpreter",
                   default=None,
                   help="the recipe's own --interpreter, if it had one")
    r.add_argument("--server-image",
                   default=None,
                   help="the image the recipe was submitted with; default: "
                   "the kind's")
    r.add_argument("--server-submit-arg",
                   action="append",
                   default=[],
                   metavar="FLAG",
                   help="another flag of the recipe's own submit, verbatim, "
                   "e.g. --server-submit-arg=--shm-size-gb=128; repeatable")
    r.add_argument("--leader",
                   action="append",
                   default=[],
                   metavar="JOB[@PROFILE]",
                   help="a server of this run already serving this recipe, "
                   "submitted with --serve; used, never cancelled; repeatable")
    r.add_argument("--leader-profile",
                   default=None,
                   help="profile of a --leader that names none; default: the "
                   "active one")
    r.add_argument("--patch",
                   action="append",
                   default=[],
                   metavar="PATCH",
                   help="a vllm_torchtpu patch every started server applies "
                   "through serve_vllm.sh; repeatable, in apply order")
    r.add_argument("--hardware",
                   default=None,
                   metavar="KEY",
                   help="the tpu_slots.py hardware key every server books")
    r.add_argument("--keep-servers",
                   action="store_true",
                   help="leave the servers started here running, slots "
                   "held, after the sweep; `cleanup` cancels them")
    r.add_argument("--max-servers",
                   type=int,
                   default=8,
                   help="servers at most, the leaders included")
    r.add_argument("--no-slots",
                   action="store_true",
                   help="book nothing: every server uses the active profile")
    r.add_argument("--colocate-requests", default="cpu=4,memory=16Gi")
    r.add_argument("--start-timeout", type=float, default=3600.0)
    r.add_argument("--server-budget-s",
                   type=float,
                   default=None,
                   help="seconds a server may serve shapes before it takes "
                   "no more; for clusters that stop jobs after a fixed time")
    r.add_argument("--leader-age-s",
                   type=float,
                   default=0.0,
                   help="a leader's time running when its record has none")
    r.add_argument("--shape-timeout", type=float, default=4 * 3600.0)
    r.add_argument("--out-dir", required=True)

    m = sub.add_parser("merge", help="merge a run's shapes again")
    m.add_argument("--out-dir", required=True)

    w = sub.add_parser("wait",
                       help="block until the runs end or die, or --max-secs")
    w.add_argument("--out-dir", action="append", required=True)
    w.add_argument("--max-secs", type=float, default=540)

    c = sub.add_parser("cleanup",
                       parents=[compute_args],
                       help="cancel what a dead run left serving")
    c.add_argument("--out-dir", required=True)

    args = p.parse_args(argv)
    if getattr(args, "client_args", None) is not None:
        if args.client_args[:1] == ["--"]:
            args.client_args = args.client_args[1:]
    if args.command == "run":
        if args.max_servers < 1:
            p.error("--max-servers must be at least 1")
        if re.search(r"[^a-z0-9-]", args.name.lower()):
            p.error("--name may hold only letters, digits and '-'")
        if not args.plan and not args.client_args:
            p.error("give the client's flags after --, or --plan")
        if args.patch and args.leader:
            p.error(
                "--patch takes no --leader: a leader's patches are unknown")
        if args.hardware and args.no_slots:
            p.error("--hardware pins the slots it books: not with --no-slots")
    return args


def load_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.plan:
        return json.loads(Path(args.plan).read_text(encoding="utf-8"))
    return make_plan(args.client_args, args.max_model_len)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        try:
            plan = make_plan(args.client_args, args.max_model_len)
        except ValueError as exc:
            print(json.dumps({"status": "invalid_input", "error": str(exc)}))
            return EXIT_SETUP
        write_json(Path(args.out), plan)
        print(
            json.dumps({
                "status":
                "ok",
                "plan":
                args.out,
                "queue":
                plan["queue"],
                "skipped_shapes":
                [s["id"] for s in plan["shapes"] if not s["c_range"]],
            }))
        return EXIT_OK

    if args.command == "wait":
        result = wait([Path(d) for d in args.out_dir], args.max_secs)
        print(json.dumps(result))
        return EXIT_OK if result["status"] == "ended" else EXIT_WAITING

    out = Path(args.out_dir)
    if args.command == "cleanup":
        compute = Compute(args.runtime, args.package, args.run_id,
                          out / "calls")
        try:
            result = cleanup(out, compute,
                             Path(args.slots) if args.slots else None)
        except (OSError, ValueError) as exc:
            result = {"status": "error", "error": str(exc)}
        print(json.dumps(result))
        return EXIT_OK if result["status"] == "ok" else EXIT_SETUP

    if args.command == "merge":
        plan = json.loads(
            (out / "sweep_plan.json").read_text(encoding="utf-8"))
        status = out / "sweep_status.json"
        state = (json.loads(status.read_text(
            encoding="utf-8")) if status.is_file() else {})
        merged = merge(
            plan, out, state.get("shapes"), state.get("servers"), {
                k: state[k]
                for k in ("patches", "patch_sha256", "hardware") if k in state
            })
        print(
            json.dumps({
                "status": merged["status"],
                "error": merged["error"]
            }))
        return EXIT_OK if merged["status"] == "ok" else EXIT_FAILED

    sweep: Sweep | None = None
    result: dict[str, Any]
    try:
        plan = load_plan(args)
        compute = Compute(args.runtime, args.package, args.run_id,
                          out / "calls")
        shown = tpu_slots.read_config(args.runtime, args.package, args.run_id)
        sweep = Sweep(args, plan, compute,
                      shown.get("profiles") or {}, shown.get("active_profile"))

        def on_signal(signum, _frame):
            sweep.stop_requested = True
            sweep.stop.set()
            sweep.log(f"signal {signum}: stopping after cleanup")

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, on_signal)
        merged = sweep.run()
        result = {
            "status": merged["status"],
            "error": merged["error"],
            "warnings": merged["warnings"],
            "summary": str(out / CLIENT_DIR / "summary.json"),
            "shards": merged["shards"],
            "servers": merged["servers"],
            "ignored_leaders": sweep.ignored,
        }
        code = sweep.exit_code(merged)
    except (ValueError, RuntimeError, tpu_slots.Refused,
            tpu_slots.NeedsInput) as exc:
        status = ("refused" if isinstance(
            exc, tpu_slots.Refused) else "needs_input" if isinstance(
                exc, tpu_slots.NeedsInput) else "invalid_input")
        result = {
            "status": status,
            "error": str(exc),
            "ignored_leaders": sweep.ignored if sweep else []
        }
        code = EXIT_SETUP
    # An out-dir an earlier run used keeps that run's result.
    target = out / "sweep_result.json"
    if result["status"] != "invalid_input" or not target.exists():
        write_json(target, result)
    print(
        json.dumps({k: result.get(k)
                    for k in ("status", "error", "summary")}))
    return code


if __name__ == "__main__":
    sys.exit(main())
