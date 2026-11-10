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
"""Unit tests for serving_sweep.py, against a fake `compute` and slot file."""

from __future__ import annotations

import contextlib
import datetime
import importlib.util
import io
import json
import os
import pathlib
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from . import serving_sweep  # noqa: E402
except ImportError:
    _spec = importlib.util.spec_from_file_location(
        "serving_sweep", _SCRIPTS_DIR / "serving_sweep.py")
    serving_sweep = importlib.util.module_from_spec(_spec)
    sys.modules["serving_sweep"] = serving_sweep
    _spec.loader.exec_module(serving_sweep)
tpu_slots = serving_sweep.tpu_slots


def _declaring(profiles, n):
    """`profiles`, each gke_job profile declaring `n` slots."""
    return {
        name: ({
            **p, "params": {
                **(p.get("params") or {}), "slots": n
            }
        } if p.get("backend") == "gke_job" else p)
        for name, p in profiles.items()
    }


serving_benchmark = serving_sweep.serving_benchmark

MODEL = {"id": "m", "root": "/model", "max_model_len": 9216}
IMAGE = "registry.example/vllm@sha256:abc"
CLIENT = [
    "--isl-osl", "1024:1024,1024:8192,8192:1024", "--concurrency", "1,8,64",
    "--num-prompts", "50", "--seed", "0", "--engine", "vllm"
]


def gke(accelerator="tpu7x", topology="2x2x1", image=IMAGE):
    return {
        "backend": "gke_job",
        "params": {
            "tpu_accelerator": accelerator,
            "tpu_topology": topology,
            "image": image,
        },
    }


GKE = {"gke": gke()}
V6E = f"tpu-v6e-slice|2x4|8|{IMAGE}"
V7X = f"tpu7x|2x2x1|4|{IMAGE}"


def _flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def _flags(argv, name):
    return [argv[i + 1] for i, a in enumerate(argv) if a == name]


class FakeCompute:
    """`compute` for a GKE cluster, as far as the sweep can tell.

    A server is reachable only when it was submitted with `--serve`, and a
    client submitted with `--interpreter` runs under it, as on gke_job. The
    knobs, by shape id unless named: `fail_start` servers that never serve;
    `no_summary` shapes whose client brings nothing back; `lost_once` shapes
    whose first client loses its server; `partial_once` shapes whose first
    client ends mid-sweep with a `partial` summary, mapped to whether its
    server is lost too; `setup_error` shapes whose client cannot start; `client_submit_fails` failed submits per shape;
    `client_wait_rc` a shape client's `job wait` exit code; `raise_on_submit`
    servers whose submit raises.
    """

    def __init__(self,
                 *,
                 delay=0.0,
                 client_delay=0.0,
                 fail_start=(),
                 no_summary=(),
                 lost_once=(),
                 partial_once=None,
                 setup_error=(),
                 client_submit_fails=None,
                 client_wait_rc=None,
                 raise_on_submit=()):
        self.lock = threading.Lock()
        self.calls, self.cancelled = [], []
        self.records, self.ready, self.submitted_at = {}, {}, {}
        self.delay, self.client_delay = delay, client_delay
        self.fail_start, self.no_summary = set(fail_start), set(no_summary)
        self.lost_once, self.setup_error = set(lost_once), set(setup_error)
        self.partial_once = dict(partial_once or {})
        self.client_submit_fails = dict(client_submit_fails or {})
        self.client_wait_rc = dict(client_wait_rc or {})
        self.raise_on_submit = set(raise_on_submit)
        self.n = 0
        for job_id in ("lead-1", "lead-2"):
            self.add_server(job_id)

    def add_server(self, job_id, serve=True, start_time=None):
        self.records[job_id] = {
            "name": job_id,
            "client": False,
            "argv": [],
            "serve": serve,
            "status": "RUNNING",
            "start_time": start_time
        }

    def __call__(self, *args, profile=None, timeout=None, tail=None):
        argv = list(args) + (tail or [])
        with self.lock:
            self.calls.append(tuple(argv))
        action, job_id = args[1], _flag(argv, "--job-id")
        job = self.records.get(job_id, {})
        if action == "wait":
            return self.wait(job_id, job, argv)
        if action == "artifacts":
            self.artifacts(job, pathlib.Path(_flag(argv, "--out-dir")))
        elif action == "logs":
            return 0, {"logs": f"log of {job_id}\n"}
        elif action == "cancel":
            with self.lock:
                self.cancelled.append(job_id)
        elif action == "status":
            return 0, {"status": job.get("status", "RUNNING")}
        return 0, {"job_id": job_id}

    def jobs(self, profile=None):
        with self.lock:
            return [{
                "job_id": k,
                "job_name": j["name"],
                "serve": j["serve"],
                "status": j["status"],
                "start_time": j.get("start_time")
            } for k, j in self.records.items()]

    def submit_job(self, *args, name, profile, tail=None):
        argv = list(args) + (tail or [])
        with self.lock:
            self.calls.append(("submit", *argv))
            if name in self.raise_on_submit:
                raise OSError(f"no runtime for {name}")
            client = "--colocate-with" in argv
            if client and self.client_submit_fails.get(self.sid(argv)):
                self.client_submit_fails[self.sid(argv)] -= 1
                return None, {"error": "colocate: no scheduled pod"}
            self.n += 1
            job_id = f"{name}-{self.n}"
            self.records[job_id] = {
                "name": name,
                "client": client,
                "serve": "--serve" in argv,
                "profile": profile,
                "argv": argv,
                "status": "RUNNING"
            }
            self.submitted_at[job_id] = time.monotonic()
        return job_id, {"job_id": job_id}

    @staticmethod
    def sid(argv):
        return serving_sweep.shape_id(
            *(int(x) for x in _flag(argv, "--isl-osl").split(":")))

    def wait(self, job_id, job, argv):
        time.sleep(self.delay)
        if not job["client"]:
            if (job["name"] in self.fail_start
                    and _flag(argv, "--until").startswith("log:")):
                return 2, {"outcome": "failure_logged", "line": "Traceback"}
            with self.lock:
                self.ready.setdefault(job_id, time.monotonic())
            return 0, {"outcome": "reached"}
        time.sleep(self.client_delay)
        rc = self.client_wait_rc.get(self.sid(job["argv"]), 0)
        return rc, {"outcome": "failure_logged" if rc else "reached"}

    def artifacts(self, job, out_dir):
        argv, base = job["argv"], out_dir / serving_sweep.CLIENT_DIR
        sid = self.sid(argv)
        server = self.records[_flag(argv, "--colocate-with")]
        base.mkdir(parents=True, exist_ok=True)

        def client_failed(error, code):
            (base / "client.json").write_text(json.dumps({"error": error}))
            (base / "exit_code").write_text(f"{code}\n")

        if sid in self.lost_once:
            self.lost_once.discard(sid)
            server["status"] = "FAILED"
            return client_failed("server stopped answering", 2)
        if not server["serve"]:
            return client_failed("unreachable", 2)
        if "--interpreter" in argv:
            return client_failed("bash: syntax error", 2)
        if sid in self.setup_error:
            return client_failed("no tokenizer here", 2)
        if sid in self.no_summary:
            return None
        # As the client does: every concurrency, filtered, seeded by place.
        isl, osl = (int(x) for x in _flag(argv, "--isl-osl").split(":"))
        conc = sorted(
            {int(x)
             for x in _flag(argv, "--concurrency").split(",")})
        limit = _flag(argv, "--max-concurrency-limit")
        kept, cells = serving_benchmark.filter_c_range(
            isl,
            osl,
            conc,
            max_model_len=MODEL["max_model_len"],
            max_concurrency_limit=int(limit) if limit else None,
            avoid_rules=serving_benchmark.parse_avoid_rules(
                _flag(argv, "--avoid") or ""))
        seed = int(_flag(argv, "--seed"))
        for c in kept:
            cells.append({
                "cell": f"isl{isl}_osl{osl}_c{c}",
                "isl": isl,
                "osl": osl,
                "concurrency": c,
                "status": "ok",
                "seed": seed + conc.index(c),
                "metrics": {
                    "output_tokens_per_s": {
                        "mean": 100.0 * c,
                        "values": [100.0 * c]
                    }
                },
                "replicates": []
            })
        settings = {
            "engine": "vllm",
            "num_prompts": _flag(argv, "--num-prompts"),
            "seed": seed,
            "stop_on_failure": False
        }
        (base / "summary.json").write_text(
            json.dumps({
                "status": "ok",
                "error": None,
                "model": MODEL,
                "base_url": _flag(argv, "--base-url"),
                "client": "vllm",
                "settings": settings,
                "cells": cells,
                "warnings": [f"note from {sid}"]
            }))
        if sid in self.partial_once:
            # serving_benchmark.py's summary after its first cell, as it
            # comes back when the client's job ends before the last.
            if self.partial_once.pop(sid):
                server["status"] = "FAILED"
            (base / "summary.json").write_text(
                json.dumps({
                    "status": "partial",
                    "error": None,
                    "settings": settings,
                    "cells": cells[:1]
                }))

    def clients(self):
        return [j for j in self.records.values() if j["client"]]

    def servers_started(self):
        return [
            k for k, j in self.records.items()
            if not j["client"] and not k.startswith("lead-")
        ]


class SweepCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.slots = self.root / "parallel.json"
        self.runs = 0
        self.declared = (GKE, GKE)

    def plan(self, *extra, client=CLIENT):
        return serving_sweep.make_plan(list(client) + list(extra), 9216)

    def make_slots(self, n, profiles=None):
        """The profiles' gke_job ones declare `n` slots; the sweep given those
        profiles is given these, and opens the same slots itself."""
        source = profiles or GKE
        self.declared = (source, _declaring(source, n))
        tpu_slots.open_slots(self.slots, self.declared[1])

    def sweep(self,
              plan,
              compute,
              *extra,
              leader=("lead-1", ),
              profiles=None,
              active="gke"):
        self.runs += 1
        argv = [
            "run", "--runtime", "rt", "--name", "gemma", "--port", "8000",
            "--recipe", "launch_vllm.sh", "--slots",
            str(self.slots), "--out-dir",
            str(self.root / f"bench{self.runs}"), "--start-timeout", "5",
            "--shape-timeout", "5"
        ]
        for job in leader:
            argv += ["--leader", job]
        if leader:
            argv += ["--leader-profile", active]
        args = serving_sweep.parse_args(argv + list(extra) + ["--"] +
                                        plan["client_args"])
        source, declared = self.declared
        if profiles is None or profiles is source:
            profiles = declared
        return serving_sweep.Sweep(args, plan, compute, profiles, active)

    def run_quietly(self, sweep):
        with contextlib.redirect_stderr(io.StringIO()):
            return sweep.run()

    def slots_free(self):
        state = json.loads(self.slots.read_text(encoding="utf-8"))
        return all(e["job"] is None for e in state["resources"].values())


class TestPlan(SweepCase):

    def test_the_plan_reads_the_clients_own_flags(self):
        plan = self.plan("--max-concurrency-limit",
                         "32",
                         "--avoid",
                         "8192:*:8",
                         "--replicates",
                         "2",
                         client=[
                             "--isl-osl", "1024:1024,1024:8192,8192:2048",
                             "--concurrency", "1,8,64", "--seed", "42"
                         ])
        by_id = {s["id"]: s for s in plan["shapes"]}
        self.assertEqual(by_id["isl1024_osl1024"]["c_range"], [1, 8])
        self.assertTrue(
            all("max_model_len" in c["error"]
                for c in by_id["isl8192_osl2048"]["skipped"]))
        self.assertEqual(plan["queue"], ["isl1024_osl8192", "isl1024_osl1024"])
        # Three concurrencies x two replicates per shape before it.
        self.assertEqual([s["seed"] for s in plan["shapes"]], [42, 48, 54])
        # Unset, prompts scale with concurrency, as in the client.
        self.assertIsNone(plan["settings"]["num_prompts"])
        self.assertEqual(plan["settings"]["sweep_mode"], "exploratory")
        fixed = self.plan(client=[
            "--isl-osl", "1024:1024", "--concurrency", "64", "--num-prompts",
            "200"
        ])
        scaled = self.plan(
            client=["--isl-osl", "1024:1024", "--concurrency", "64"])
        self.assertGreater(scaled["shapes"][0]["cost"],
                           fixed["shapes"][0]["cost"])

    def test_flags_a_sweep_cannot_honour_are_refused(self):
        for bad in (["--isl-osl", "1024:1024,1024:1024"
                     ], ["--isl-osl", "1024:1024", "--base-url",
                         "http://x"], ["--isl-osl", "1024:1024", "--profile"],
                    ["--isl-osl", "1024:1024", "--no-such-flag"
                     ], ["--workload", "agentx", "--concurrency", "1"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.plan(client=bad)


class TestRun(SweepCase):

    def test_a_sweep_runs_every_shape_once_on_servers_that_serve(self):
        # No parallel.json yet: the sweep writes it from the profile's slots.
        two = gke()
        two["params"] = {**two["params"], "slots": 2}
        fake = FakeCompute(delay=0.01, client_delay=0.1)
        sweep = self.sweep(self.plan(),
                           fake,
                           "--server-submit-arg=--shm-size-gb=128",
                           "--server-input",
                           "deployments.json",
                           "--server-image",
                           "registry.example/patched@sha256:f",
                           profiles={"gke": two})
        merged = self.run_quietly(sweep)
        self.assertTrue(self.slots.exists())
        self.assertEqual(merged["status"], "ok")
        self.assertEqual(len(merged["cells"]), 9)
        ran = {_flag(j["argv"], "--isl-osl"): j for j in fake.clients()}
        self.assertEqual(len(fake.clients()), 3)
        self.assertEqual(
            {
                k: int(_flag(j["argv"], "--seed"))
                for k, j in ran.items()
            }, {
                "1024:1024": 0,
                "1024:8192": 3,
                "8192:1024": 6
            })
        for job in fake.clients():
            argv = job["argv"]
            self.assertEqual(_flag(argv, "--base-url"),
                             f"http://{_flag(argv, '--colocate-with')}:8000")
            self.assertEqual(_flag(argv, "--concurrency"), "1,8,64")
            self.assertEqual(_flags(argv, "--isl-osl"),
                             [_flag(argv, "--isl-osl")])
            self.assertEqual(_flag(argv, "--num-prompts"), "50")
            self.assertNotIn("--interpreter", argv)
            recipe = pathlib.Path(_flag(argv, "--recipe"))
            self.assertEqual(recipe.name, "serving_sweep_client.sh")
            self.assertTrue(recipe.is_file())
            shipped = {i.rsplit(":", 1)[1] for i in _flags(argv, "--input")}
            self.assertEqual(
                shipped, {
                    "serving_benchmark.py", "fetch_weights.py",
                    "benchmark_fastapi_server.py"
                })
        followers = fake.servers_started()
        self.assertEqual(len(followers), 2)
        for job_id in followers:
            argv = fake.records[job_id]["argv"]
            self.assertIn("--serve", argv)
            self.assertIn("--shm-size-gb=128", argv)
            self.assertEqual(_flag(argv, "--image"),
                             "registry.example/patched@sha256:f")
            self.assertEqual(_flags(argv, "--input"), ["deployments.json"])
        self.assertTrue(
            {_flag(j["argv"], "--colocate-with")
             for j in fake.clients()} & set(followers))
        self.assertEqual(sorted(fake.cancelled), sorted(followers))
        self.assertTrue(self.slots_free())
        self.assertEqual({s["hardware"]
                          for s in merged["servers"]},
                         {"tpu7x|2x2x1|4|registry.example/patched@sha256:f"})
        self.assertEqual(len(merged["warnings"]), 3)

    def test_servers_are_at_most_one_per_shape_and_the_slots(self):
        for slots, shapes, started in ((1, "1024:1024,1024:8192,8192:1024", 1),
                                       (4, "1024:1024,1024:8192", 1)):
            with self.subTest(slots=slots, shapes=shapes):
                self.make_slots(slots)
                fake = FakeCompute()
                self.run_quietly(
                    self.sweep(
                        self.plan(
                            client=["--isl-osl", shapes, "--concurrency", "1"
                                    ]), fake))
                self.assertEqual(len(fake.servers_started()), started)

    def test_followers_wait_for_the_first_server_however_it_ends(self):
        for raises in (False, True):
            with self.subTest(first_raises=raises):
                self.make_slots(3)
                fake = FakeCompute(delay=0.05,
                                   raise_on_submit={"gemma-s1"} if raises else
                                   ())
                sweep = self.sweep(self.plan(), fake, leader=())
                done = {}
                runner = threading.Thread(
                    target=lambda: done.update(merged=self.run_quietly(sweep)),
                    daemon=True)
                runner.start()
                runner.join(30)
                self.assertFalse(runner.is_alive(), "the sweep hung")
                self.assertEqual(done["merged"]["status"], "ok")
                if not raises:
                    first, *rest = sorted(fake.servers_started(),
                                          key=fake.submitted_at.get)
                    for job_id in rest:
                        self.assertGreaterEqual(fake.submitted_at[job_id],
                                                fake.ready[first])
                self.assertTrue(self.slots_free())

    def test_each_failure_stays_with_its_own_shape(self):
        first = self.plan()["queue"][0]
        cases = {
            "a follower never serves":
            dict(fail_start={"gemma-s2"}),
            "a client submit fails once":
            dict(client_submit_fails={first: 1}),
            "a server is lost mid-shape":
            dict(lost_once={first}, client_delay=0.05),
            "a client's node is lost with its server":
            dict(partial_once={first: True}, client_delay=0.05),
            "a client ends mid-shape, its server serving":
            dict(partial_once={first: False}),
            "a client submit keeps failing":
            dict(client_submit_fails={first: 9}),
            "a client cannot start":
            dict(setup_error={first}),
            "a client wait ends badly":
            dict(client_wait_rc={first: 2}, no_summary={first}),
        }
        # Which shape fails, and how often it was tried.
        expect = {
            "a follower never serves": ("ok", None, 1),
            "a client submit fails once": ("ok", None, 2),
            "a server is lost mid-shape": ("ok", None, 2),
            "a client's node is lost with its server": ("ok", None, 2),
            "a client ends mid-shape, its server serving": ("ok", None, 2),
            "a client submit keeps failing":
            ("failed", first, serving_sweep.SHAPE_TRIES),
            "a client cannot start": ("failed", first, 1),
            "a client wait ends badly": ("failed", first, 1),
        }
        for name, knobs in cases.items():
            with self.subTest(name):
                self.make_slots(2)
                fake = FakeCompute(delay=0.01, **knobs)
                sweep = self.sweep(self.plan(), fake)
                merged = self.run_quietly(sweep)
                status, failed, tries = expect[name]
                self.assertEqual(merged["status"], status)
                self.assertEqual(sweep.shapes[first]["attempts"], tries)
                states = {s: v["state"] for s, v in sweep.shapes.items()}
                self.assertEqual([s for s, v in states.items() if v != "done"],
                                 [failed] if failed else [])
                self.assertTrue(self.slots_free())
                if name == "a client wait ends badly":
                    client = sweep.shapes[first]["job_id"]
                    self.assertIn(client, fake.cancelled)
                    self.assertTrue(
                        all("--no-default-fail-patterns" in c
                            for c in fake.calls if c[:2] == ("job", "wait")
                            and _flag(list(c), "--job-id") == client))
                if name == "a client cannot start":
                    self.assertIn("could not start",
                                  sweep.shapes[first]["error"])
                    self.assertNotIn("lost",
                                     {s["state"]
                                      for s in merged["servers"]})

    def test_an_exception_or_a_stale_file_fails_only_its_shape(self):
        self.make_slots(1)
        plan = self.plan()
        first = plan["queue"][0]
        fake = FakeCompute(delay=0.01, no_summary={first})
        sweep = self.sweep(plan, fake)
        stale = sweep.out / "shards" / first / serving_sweep.CLIENT_DIR
        stale.mkdir(parents=True)
        (stale / "summary.json").write_text('{"status": "ok", "cells": []}')
        real = fake.artifacts

        def broken(job, out_dir):
            if FakeCompute.sid(job["argv"]) == "isl8192_osl1024":
                raise RuntimeError("disk full")
            real(job, out_dir)

        fake.artifacts = broken
        merged = self.run_quietly(sweep)
        self.assertEqual(sweep.shapes[first]["state"], "failed")
        self.assertFalse((stale / "summary.json").exists())
        self.assertIn("disk full", merged["error"])
        with self.assertRaises(ValueError):
            serving_sweep.Sweep(sweep.args, plan, FakeCompute(), GKE,
                                "gke").run()

    def test_leaders_are_used_only_when_a_client_can_reach_them(self):
        profiles = {"gke": gke(), "v6e": gke("tpu-v6e-slice", "2x4")}
        self.make_slots(1, profiles)
        fake = FakeCompute(delay=0.01)
        fake.add_server("lead-quiet", serve=False)
        sweep = self.sweep(self.plan(),
                           fake,
                           profiles=profiles,
                           leader=("lead-1", "lead-2@gke", "lead-quiet",
                                   "lead-3@v6e", "lead-gone"))
        fake.add_server("lead-3")
        merged = self.run_quietly(sweep)
        self.assertEqual(merged["status"], "ok")
        why = {i["job_id"]: i["why"] for i in sweep.ignored}
        self.assertIn("--serve", why["lead-quiet"])
        self.assertIn("another kind", why["lead-3"])
        self.assertIn("not a job of this run", why["lead-gone"])
        used = {_flag(j["argv"], "--colocate-with") for j in fake.clients()}
        self.assertTrue(used <= {"lead-1", "lead-2"})
        self.assertFalse({"lead-1", "lead-2"} & set(fake.cancelled))

    def test_a_server_past_its_budget_takes_no_new_shape(self):
        old = (datetime.datetime.now(datetime.timezone.utc) -
               datetime.timedelta(hours=2)).isoformat()
        for name, record_start, extra in (("age from its record", old, []),
                                          ("age from the flag", None,
                                           ["--leader-age-s", "7200"])):
            with self.subTest(name):
                self.make_slots(1)
                fake = FakeCompute(delay=0.01)
                fake.add_server("lead-1", start_time=record_start)
                sweep = self.sweep(self.plan(), fake, "--server-budget-s",
                                   "3600", *extra)
                merged = self.run_quietly(sweep)
                servers = {s["label"]: s for s in merged["servers"]}
                self.assertEqual(servers["leader"]["state"], "budget spent")
                self.assertEqual({s["server"]
                                  for s in sweep.shapes.values()}, {"s2"})

    def test_a_backend_that_cannot_serve_side_by_side_is_refused(self):
        self.make_slots(2)
        sweep = self.sweep(self.plan(),
                           FakeCompute(),
                           leader=(),
                           profiles={"gke": {
                               "backend": "gce_tpu_vm"
                           }})
        with self.assertRaises(ValueError) as refused:
            sweep.run()
        # It says how the sweep is measured instead.
        self.assertIn("one after another", str(refused.exception))
        self.assertTrue(self.slots_free())

    def test_a_stop_cancels_clients_and_servers_once_and_cleanup_finishes(
            self):
        self.make_slots(1)
        fake = FakeCompute(delay=0.2)
        sweep = self.sweep(self.plan(), fake)
        stopper = threading.Timer(0.3, sweep.stop.set)
        stopper.start()
        sweep.stop_requested = True
        merged = self.run_quietly(sweep)
        stopper.cancel()
        self.assertIn("not_run", {s["state"] for s in sweep.shapes.values()})
        self.assertEqual(merged["status"], "failed")
        self.assertTrue(self.slots_free())
        follower = next(s for s in sweep.servers if s["owned"])
        sweep.stop_server(follower)
        self.assertEqual(fake.cancelled.count(follower["job_id"]), 1)
        # A run that died first: cleanup cancels what it left, from its file.
        held = tpu_slots.claim(self.slots, "serving-sweep:gemma:s9")
        dead = self.root / "dead"
        dead.mkdir()
        (dead / "sweep_status.json").write_text(
            json.dumps({
                "slots":
                str(self.slots),
                "servers": [{
                    "label": "leader",
                    "job_id": "lead-1",
                    "owned": False
                }, {
                    "label": "s9",
                    "job_id": "gemma-s9-7",
                    "owned": True,
                    "resource": held["resource"],
                    "holder": "serving-sweep:gemma:s9"
                }, {
                    "label": "s3",
                    "job_id": "gemma-s3-8",
                    "owned": True,
                    "released": True
                }]
            }))
        fake = FakeCompute()
        result = serving_sweep.cleanup(dead, fake, None)
        self.assertEqual((result["cancelled"], fake.cancelled),
                         (["gemma-s9-7"], ["gemma-s9-7"]))
        self.assertTrue(self.slots_free())

    def test_many_shapes_on_many_servers_each_run_exactly_once(self):
        self.make_slots(6)
        shapes = ",".join(f"{128 * i}:{64 * i}" for i in range(1, 13))
        fake = FakeCompute(delay=0.005)
        sweep = self.sweep(
            self.plan(client=["--isl-osl", shapes, "--concurrency", "1,8"]),
            fake, "--max-servers", "7")
        self.assertEqual(self.run_quietly(sweep)["status"], "ok")
        ran = [_flag(j["argv"], "--isl-osl") for j in fake.clients()]
        self.assertEqual((len(ran), len(set(ran))), (12, 12))

    def test_exit_codes_tell_a_failed_shape_from_no_server(self):
        plan = self.plan()
        for name, knobs, leader, code in (
            ("every client failed", dict(no_summary=set(plan["queue"])),
             ("lead-1", ), serving_sweep.EXIT_FAILED),
            ("no server served", dict(fail_start={"gemma-s1", "gemma-s2"}), (),
             serving_sweep.EXIT_SETUP)):
            with self.subTest(name):
                self.make_slots(2)
                sweep = self.sweep(plan, FakeCompute(**knobs), leader=leader)
                self.assertEqual(sweep.exit_code(self.run_quietly(sweep)),
                                 code)

    def test_a_run_with_nowhere_to_book_says_so_without_a_traceback(self):
        runtime = self.root / "rt"
        runtime.write_text(
            "#!/bin/sh\ncase \"$1\" in config) echo '{\"profiles\": {\"gke\":"
            " {\"backend\": \"gke_job\"}}, \"active_profile\": \"gke\"}';;\n"
            "compute) echo '[]';; esac\n",
            encoding="utf-8")
        runtime.chmod(0o755)
        out, printed = self.root / "nobook", io.StringIO()
        with unittest.mock.patch.dict(os.environ), \
                contextlib.redirect_stdout(printed), \
                contextlib.redirect_stderr(io.StringIO()):
            os.environ.pop("VMBENCH_STATE", None)
            code = serving_sweep.main([
                "run", "--runtime",
                str(runtime), "--name", "g", "--port", "8000", "--recipe",
                "r.sh", "--out-dir",
                str(out), "--"
            ] + CLIENT)
        self.assertEqual(code, serving_sweep.EXIT_SETUP)
        self.assertEqual(
            json.loads(printed.getvalue())["status"], "needs_input")
        self.assertEqual(
            json.loads((out / "sweep_result.json").read_text())["status"],
            "needs_input")


class TestHardware(SweepCase):

    PROFILES = {
        # A v6e cluster, and two profiles that reach v7x of one kind.
        "gke-v6e": gke("tpu-v6e-slice", "2x4"),
        "gke-v7x-a": gke("tpu7x", "2x2x1"),
        "gke-v7x-b": gke("tpu7x", "2x2x1"),
    }

    def run_on(self, fake, active, leader=("lead-1", )):
        sweep = self.sweep(self.plan(),
                           fake,
                           leader=leader,
                           profiles=self.PROFILES,
                           active=active)
        return sweep, self.run_quietly(sweep)

    def kinds(self, fake):
        return {
            tpu_slots.hardware_key(self.PROFILES[fake.records[k]["profile"]])
            for k in fake.servers_started()
        }

    def test_followers_are_the_leaders_kind_on_any_profile_of_it(self):
        self.make_slots(1, self.PROFILES)
        tpu_slots.claim(self.slots,
                        "tpu-model-onboarding:g",
                        profile="gke-v7x-a")
        fake = FakeCompute(delay=0.01)
        sweep, merged = self.run_on(fake, "gke-v7x-a")
        self.assertEqual(merged["status"], "ok")
        # v7x-a's slot is the leader's: the follower runs on v7x-b, and the
        # free v6e slot is left alone.
        self.assertEqual(
            [fake.records[k]["profile"] for k in fake.servers_started()],
            ["gke-v7x-b"])
        self.assertEqual({s["hardware"] for s in merged["shards"]}, {V7X})

    def test_without_a_leader_the_first_server_goes_to_the_most_free_kind(
            self):
        self.make_slots(1, self.PROFILES)
        fake = FakeCompute(delay=0.01)
        _, merged = self.run_on(fake, "gke-v6e", leader=())
        self.assertEqual(merged["status"], "ok")
        self.assertEqual(self.kinds(fake), {V7X})

    def test_shapes_measured_on_two_kinds_are_not_one_sweep(self):
        self.make_slots(2, self.PROFILES)
        sweep, _ = self.run_on(FakeCompute(delay=0.01), "gke-v6e")
        follower = next(s for s in sweep.servers if s["owned"])
        follower["hardware"] = V7X
        next(iter(sweep.shapes.values()))["server"] = follower["label"]
        merged = serving_sweep.merge(sweep.plan, sweep.out, sweep.shapes,
                                     sweep.servers)
        self.assertIn("more than one kind of TPU", merged["error"])


class TestMerge(SweepCase):

    def test_a_merge_keeps_seeds_skips_warnings_and_the_plans_settings(self):
        self.make_slots(2)
        plan = self.plan("--avoid", "*:*:8")
        sweep = self.sweep(plan, FakeCompute(delay=0.01))
        merged = self.run_quietly(sweep)
        self.assertEqual(merged["status"], "ok")
        seeds = {
            c["cell"]: c.get("seed")
            for c in merged["cells"] if c["status"] == "ok"
        }
        # Where an unsharded sweep puts them: c=64 is third in its shape.
        self.assertEqual(
            (seeds["isl1024_osl1024_c64"], seeds["isl8192_osl1024_c64"]),
            (2, 8))
        self.assertEqual(
            sum(c["status"] == "skipped" for c in merged["cells"]), 3)
        self.assertEqual(sorted(merged["warnings"]),
                         sorted(f"{s}: note from {s}" for s in plan["queue"]))
        # Shapes with other settings are not one sweep.
        path = (sweep.out / "shards" / plan["queue"][0] /
                serving_sweep.CLIENT_DIR / "summary.json")
        summary = json.loads(path.read_text())
        summary["settings"]["num_prompts"] = 999
        path.write_text(json.dumps(summary))
        merged = serving_sweep.merge(plan, sweep.out, sweep.shapes,
                                     sweep.servers)
        self.assertIn("other settings", merged["error"])
        # With nothing measured, the settings are still the plan's.
        nothing = serving_sweep.merge(plan, self.root / "empty", {}, [])
        self.assertEqual(nothing["status"], "failed")
        self.assertEqual(nothing["settings"]["avoid"], "*:*:8")
        self.assertEqual(nothing["settings"]["sweep_mode"], "exploratory")

    def test_a_shape_the_plan_left_empty_fails_the_sweep_as_in_one_run(self):
        self.make_slots(2)
        plan = self.plan(
            client=["--isl-osl", "1024:1024,8192:2048", "--concurrency", "1"])
        merged = self.run_quietly(self.sweep(plan, FakeCompute()))
        self.assertEqual(merged["status"], "failed")
        self.assertIn("isl8192_osl2048", merged["error"])


class TestCompute(unittest.TestCase):

    def runtime(self, tmp, body):
        path = pathlib.Path(tmp) / "rt"
        path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
        path.chmod(0o755)
        return serving_sweep.Compute(str(path), "pkg", "run-1",
                                     pathlib.Path(tmp) / "calls")

    def test_compute_output_is_read_from_files_and_job_ids_are_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            compute = self.runtime(
                tmp, "echo 'a log line' >&2\necho '{\"job_id\": \"j-1\"}'\n")
            self.assertEqual(compute("submit", "--recipe", "r.sh"),
                             (0, {
                                 "job_id": "j-1"
                             }))
            self.assertEqual(
                len(list((pathlib.Path(tmp) / "calls").iterdir())), 2)
        with tempfile.TemporaryDirectory() as tmp:
            compute = self.runtime(
                tmp, "case \"$2\" in submit) echo 'not json';;\n"
                "job) echo '[{\"job_id\": \"old\", \"job_name\": \"g\", "
                "\"status\": \"COMPLETED\", \"start_time\": \"1\"}, "
                "{\"job_id\": \"g-x\", \"job_name\": \"g\", "
                "\"status\": \"PENDING\", \"start_time\": \"2\"}]';;\nesac\n")
            self.assertEqual(
                compute.submit_job("--recipe", "r.sh", name="g",
                                   profile=None)[0], "g-x")
        with tempfile.TemporaryDirectory() as tmp:
            compute = serving_sweep.Compute("/nonexistent/rt", None, None,
                                            pathlib.Path(tmp) / "calls")
            rc, body = compute("job", "status", "--job-id", "x")
            self.assertNotEqual(rc, 0)
            self.assertIn("error", body)


if __name__ == "__main__":
    unittest.main()
